# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cluster split-KV paged decode: 7 dense query tokens of one 16-head GQA group, FP8, SM103.

One CTA per (request [, row group], KV split); the KV splits of a request (row group) form one thread-block cluster
and are merged through distributed shared memory inside the same launch (owner-directed bulk copies, exact fp32
split weights; no global workspace, no reducer kernel). Two schedules, picked by batch size (kernel_config):

PagedDecodeQ7Dual (B <= DUAL_MAX_B and the DUAL_EXTRA batch sizes): G query-row groups of 112 / G rows (M=64 MMAs;
    G = 8 / 4 / 2 for B1 / B2 / the others) x C splits; the two 64-key tiles of a pair are packed into TMEM lanes
    0-15 / 16-31 of every sub-partition. Every K/V tile of the CTA's run is resident in shared memory, so the
    softmax is max-first: pass 1 takes the exact row max from TMEM (tcgen05.ld.red), pass 2 writes P and
    O += P V never needs a rescale. The first `ne` pairs of a run turn into fp32 E = 2^(k (s - m_t)) as their S
    lands (k = scale * log2(e), m_t: the thread's running max of s) and get P = e4m3(448 * E * 2^(k (m_t - m)))
    once the row max m is known.
    Runs longer than the NS resident S slots: 6 pairs (C6: B8-B11) keep the sixth S pair under the P / l columns;
    7 pairs (C5: B12-B14) the seventh in the not yet written O columns, the V tiles of pairs 2-4 reusing their K
    slots once their S is computed; 9 pairs (C4: B16-B18) pairs 6 / 7 in the O columns and pair 8 in slot 0 after
    the early pair 0 moved its E to shared memory, the V tiles filling the K slots in pass-2 order. From B4 on,
    every split pushes its row maxima to the row owners while its last P.V runs (st.async) and the owner
    accumulates the partials in arrival order, one mbarrier per source; B1-B3 merge their small partials behind
    one barrier. 8 softmax warps, 1 MMA warp, 1 TMA warp.
PagedDecodeQ7Online (the other batch sizes): 128-row tiles (112 query rows), K/V pages streamed through a 2-stage
    ring (K one page ahead of V), two softmax warpgroups (64-key halves of every page) with an exact online
    softmax (O and l rescaled in TMEM when the row max moves). Row maxima go to the row owners when the softmax
    loop ends; the partials are accumulated in arrival order, one mbarrier per source. 1 TMA warp, 2 MMA warps.

Numerics: fp32 S / O / l in TMEM, P = e4m3(448 * 2^(s * scale * log2(e) - m)) (the FP8 attention P scaling of
the stock decode), l = P . ones on the tensor core, fp32 split merge, fp16 output. The P exponentials use
ex2.approx.ftz.f32: the same MUFU.EX2 as the non-ftz form, differing only for subnormal inputs and results,
which e4m3 rounding maps to the same value, so the e4m3 P is unchanged. Online corrections, early-pair factors
and split-merge weights use the exact exp2. The row max (pass 1 / online) is scale * max(s) for a positive scale
(pos_scale); the host compiles that form only.

K and V are read through two TMA descriptors built from the caller's K/V views (token and page strides are
runtime values), so one interleaved paged buffer and separate pools are both addressed by strides. Keys past
seq_len in a request's last page are masked; pages past ceil(seq_len / 128) (64-key halves past
ceil(seq_len / 64)) are never read.

A request with no keys (seq_len <= 0, e.g. a padded CUDA-graph row) reads no pages and runs the same cluster
barrier protocol as any other request; its 112 output rows are written as zeros (empty_rows="zero") or left
untouched (empty_rows="skip").
"""

import functools
import math
from typing import NamedTuple

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05, OperandMajorMode
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.typing import Float32, Int32, Int64
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
from cutlass import AddressSpace
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm, nvvm

import cutlass.pipeline as pipeline
from cutlass.pipeline import Agent, CooperativeGroup

SEQ_Q = 7
HEADS = 16
ROWS = SEQ_Q * HEADS  # 112 query rows per request (one GQA group)
D = 128
PAGE = 128
HT = 64  # keys per tile (half page)
MAX_PAGES = 34
LOG2_E = math.log2(math.e)
LOG2_P_SCALE = math.log2(448.0)  # P is scaled into the e4m3 range before the P.V MMA
ROW_F = 132  # floats per pushed partial row: O[128] + (m, l, 0, 0)
ROW_B = ROW_F * 4
NS = 5  # resident S tiles in TMEM
NP = 3  # P stages in TMEM
# TMEM columns
TM_O = 0
TM_S = 128
TM_P = TM_S + NS * HT
TM_L = TM_P + NP * (HT // 4)
TM_COLS = 512

NWG = 2  # softmax warpgroups of PagedDecodeQ7Dual; each warp owns CW columns of every tile
CW = HT // NWG
SM_WARPS = 4 * NWG
SM_THREADS = 32 * SM_WARPS
NTHREADS = SM_THREADS + 64
# PagedDecodeQ7Dual with a seventh pair (huge): the V requests wait for K tile HUGE_VW (or the run's last one) to
# land; the first HUGE_DL tiles (pairs 0, 1) get dedicated V slots, so their V requests do not wait on their S
HUGE_VW = 3
HUGE_DL = 4

warp_threads = 32
warpgroup_threads = 128

# Output of a request with no keys (seq_len <= 0): "zero" writes zeros to its 112 rows, "skip" leaves them as they are.
EMPTY_ROWS_MODES = ("zero", "skip")


def _asm(ret, args, text, cons):
    return llvm.inline_asm(ret, args, text, cons, has_side_effects=True, is_align_stack=False,
                           asm_dialect=llvm.AsmDialect.AD_ATT)


def _asm0(text):
    @dsl_user_op
    def f(*, loc=None, ip=None):
        llvm.inline_asm(None, [], text, "", has_side_effects=True, is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return f


tc_fence_before = _asm0("tcgen05.fence::before_thread_sync;")
tc_fence_after = _asm0("tcgen05.fence::after_thread_sync;")


def _i32(x, loc, ip):
    return Int32(x).ir_value(loc=loc, ip=ip)


def _f32(x, loc, ip):
    return Float32(x).ir_value(loc=loc, ip=ip)


@dsl_user_op
def dsmem_addr(smem_ptr, rank, *, loc=None, ip=None):
    """32-bit shared::cluster address of `smem_ptr` in the CTA with cluster rank `rank`."""
    a = nvvm.mapa(llvm.PointerType.get(AddressSpace.dsmem), smem_ptr.to_llvm_ptr(loc=loc, ip=ip),
                  _i32(rank, loc, ip), loc=loc, ip=ip)
    return Int32(llvm.ptrtoint(T.i32(), a, loc=loc, ip=ip))


@dsl_user_op
def smem_addr(smem_ptr, *, loc=None, ip=None):
    return Int32(llvm.ptrtoint(T.i32(), smem_ptr.to_llvm_ptr(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def gmem_addr(gmem_ptr, *, loc=None, ip=None):
    return Int64(llvm.ptrtoint(T.i64(), gmem_ptr.to_llvm_ptr(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def ld_smem_v4f32(addr, *, loc=None, ip=None):
    st = llvm.StructType.get_literal([T.f32()] * 4)
    r = _asm(st, [_i32(addr, loc, ip)], "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];", "=f,=f,=f,=f,r")
    return [Float32(llvm.extractvalue(T.f32(), r, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def ld_smem_f32(addr, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [_i32(addr, loc, ip)], "ld.shared.f32 $0, [$1];", "=f,r"))


@dsl_user_op
def st_smem_f32(addr, a, *, loc=None, ip=None):
    _asm(None, [_i32(addr, loc, ip), _f32(a, loc, ip)], "st.shared.f32 [$0], $1;", "r,f")


@dsl_user_op
def st_global_f16x4(addr, a, b, c, d, *, loc=None, ip=None):
    """Round 4 fp32 to fp16 (rn) and store them as one 8-byte global store."""
    _asm(None, [Int64(addr).ir_value(loc=loc, ip=ip)] + [_f32(x, loc, ip) for x in (a, b, c, d)],
         "{\n\t.reg .b32 lo, hi;\n\tcvt.rn.f16x2.f32 lo, $2, $1;\n\tcvt.rn.f16x2.f32 hi, $4, $3;\n\t"
         "st.global.v2.b32 [$0], {lo, hi};\n\t}",
         "l,f,f,f,f")


@dsl_user_op
def st_async_f32(addr, a, mbar, *, loc=None, ip=None):
    """4-byte remote (shared::cluster) store, completion counted on the destination CTA's mbarrier."""
    _asm(None, [_i32(addr, loc, ip), _f32(a, loc, ip), _i32(mbar, loc, ip)],
         "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [$0], $1, [$2];", "r,f,r")


@dsl_user_op
def st_smem_v4f32(addr, a, b, c, d, *, loc=None, ip=None):
    _asm(None, [_i32(addr, loc, ip)] + [_f32(x, loc, ip) for x in (a, b, c, d)],
         "st.shared.v4.f32 [$0], {$1, $2, $3, $4};", "r,f,f,f,f")


@dsl_user_op
def bulk_s2dsmem(dst, src, nbytes, mbar, *, loc=None, ip=None):
    """cp.async.bulk smem -> peer smem (shared::cluster), completion via the peer's mbarrier tx count."""
    _asm(None, [_i32(x, loc, ip) for x in (dst, src, nbytes, mbar)],
         "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
         "r,r,r,r")


bulk_commit = _asm0("cp.async.bulk.commit_group;")
bulk_wait_read0 = _asm0("cp.async.bulk.wait_group.read 0;")


def tree_max(vals):
    """Pairwise (log-depth) max of a list of Float32 (plain Python: unrolled at trace time)."""
    while len(vals) > 1:
        vals = [cute.arch.fmax(vals[2 * i], vals[2 * i + 1]) for i in range(len(vals) // 2)]
    return vals[0]


@dsl_user_op
def tmem_ld_x32(taddr, *, loc=None, ip=None):
    """32 consecutive fp32 TMEM columns of this thread's lane (32x32b.x32), waited."""
    st = llvm.StructType.get_literal([T.f32()] * 32)
    regs = ", ".join(f"${i}" for i in range(32))
    r = _asm(st, [_i32(taddr, loc, ip)],
             "tcgen05.ld.sync.aligned.32x32b.x32.b32 {" + regs + "}, [$32];\n\ttcgen05.wait::ld.sync.aligned;",
             ",".join(["=f"] * 32) + ",r")
    return [Float32(llvm.extractvalue(T.f32(), r, [i], loc=loc, ip=ip)) for i in range(32)]


@dsl_user_op
def tmem_ld_epi(ta, tb, tla, tlb, *, loc=None, ip=None):
    """Epilogue loads of both lane halves under one wait: O (16x32bx2.x16, column offset 16) at ta / tb and l
    (16x32bx2.x1) at tla / tlb.  Returns 16 + 16 + 1 + 1 values."""
    st = llvm.StructType.get_literal([T.f32()] * 34)
    ra = ", ".join(f"${i}" for i in range(16))
    rb = ", ".join(f"${i}" for i in range(16, 32))
    r = _asm(st, [_i32(x, loc, ip) for x in (ta, tb, tla, tlb)],
             "tcgen05.ld.sync.aligned.16x32bx2.x16.b32 {" + ra + "}, [$34], 16;\n\t"
             "tcgen05.ld.sync.aligned.16x32bx2.x16.b32 {" + rb + "}, [$35], 16;\n\t"
             "tcgen05.ld.sync.aligned.16x32bx2.x1.b32 {$32}, [$36], 0;\n\t"
             "tcgen05.ld.sync.aligned.16x32bx2.x1.b32 {$33}, [$37], 0;\n\t"
             "tcgen05.wait::ld.sync.aligned;",
             ",".join(["=f"] * 34) + ",r,r,r,r")
    return [Float32(llvm.extractvalue(T.f32(), r, [i], loc=loc, ip=ip)) for i in range(34)]


def _make_ldred(op, w):
    @dsl_user_op
    def f(taddr, *, loc=None, ip=None):
        """Row reduction of w consecutive fp32 TMEM columns of this thread's lane (32x32b.x{w}; values discarded)."""
        regs = ", ".join(f"t{i}" for i in range(w))
        return Float32(_asm(
            T.f32(), [_i32(taddr, loc, ip)],
            "{\n\t.reg .f32 " + regs + ";\n\t"
            "tcgen05.ld.red.sync.aligned.32x32b.x" + str(w) + "." + op + ".f32 {" + regs + "}, $0, [$1];\n\t"
            "tcgen05.wait::ld.sync.aligned;\n\t}",
            "=f,r"))
    return f


def _make_ldred64(op):
    @dsl_user_op
    def f(taddr, *, loc=None, ip=None):
        """Row reduction of 64 consecutive fp32 TMEM columns of this thread's lane (32x32b.x64; values discarded)."""
        regs = ", ".join(f"t{i}" for i in range(64))
        return Float32(_asm(
            T.f32(), [_i32(taddr, loc, ip)],
            "{\n\t.reg .f32 " + regs + ";\n\t"
            "tcgen05.ld.red.sync.aligned.32x32b.x64." + op + ".f32 {" + regs + "}, $0, [$1];\n\t"
            "tcgen05.wait::ld.sync.aligned;\n\t}",
            "=f,r"))
    return f


tmem_ldred64_max = _make_ldred64("max")
tmem_ldred64_min = _make_ldred64("min")


def _make_ldred_vals(op, w):
    @dsl_user_op
    def f(taddr, *, loc=None, ip=None):
        """w consecutive fp32 TMEM columns of this thread's lane and their row reduction (32x32b.x{w}), waited:
        w values, then the reduction."""
        st = llvm.StructType.get_literal([T.f32()] * (w + 1))
        regs = ", ".join(f"${i}" for i in range(w))
        r = _asm(st, [_i32(taddr, loc, ip)],
                 "tcgen05.ld.red.sync.aligned.32x32b.x" + str(w) + "." + op + ".f32 {" + regs + "}, $" + str(w)
                 + ", [$" + str(w + 1) + "];\n\ttcgen05.wait::ld.sync.aligned;",
                 ",".join(["=f"] * (w + 1)) + ",r")
        return [Float32(llvm.extractvalue(T.f32(), r, [i], loc=loc, ip=ip)) for i in range(w + 1)]
    return f


@dsl_user_op
def tmem_st_x32f(taddr, vals, *, loc=None, ip=None):
    regs = ", ".join(f"${i + 1}" for i in range(32))
    _asm(None, [_i32(taddr, loc, ip)] + [_f32(x, loc, ip) for x in vals],
         "tcgen05.st.sync.aligned.32x32b.x32.b32 [$0], {" + regs + "};\n\ttcgen05.wait::st.sync.aligned;",
         "r," + ",".join(["f"] * 32))


@dsl_user_op
def tmem_st_x1f(taddr, v, *, loc=None, ip=None):
    _asm(None, [_i32(taddr, loc, ip), _f32(v, loc, ip)],
         "tcgen05.st.sync.aligned.32x32b.x1.b32 [$0], {$1};\n\ttcgen05.wait::st.sync.aligned;", "r,f")


@dsl_user_op
def tmem_st_x8(taddr, w, *, loc=None, ip=None):
    _asm(None, [_i32(taddr, loc, ip)] + [_i32(x, loc, ip) for x in w],
         "tcgen05.st.sync.aligned.32x32b.x8.b32 [$0], {$1, $2, $3, $4, $5, $6, $7, $8};\n\t"
         "tcgen05.wait::st.sync.aligned;",
         "r,r,r,r,r,r,r,r,r")


@dsl_user_op
def ex2_p(x, *, loc=None, ip=None):
    """2^x for the probabilities P only.  ex2.approx.ftz.f32 and the exact (non-ftz) ex2.approx.f32 issue the same
    MUFU.EX2; they differ only for subnormal inputs (|x| < 2^-126: both give 1.0f) and for results below 2^-126,
    which are far below the smallest e4m3 subnormal (2^-9) and become 0 either way.  P is quantized to e4m3
    (satfinite, rn) right after, so the e4m3 P is bit-identical to the exact path; it just skips the subnormal
    fix-up instructions.  The split-merge and online-correction weights stay on the exact exp2."""
    return Float32(llvm.inline_asm(T.f32(), [_f32(x, loc, ip)], "ex2.approx.ftz.f32 $0, $1;", "=f,f",
                                   has_side_effects=False, is_align_stack=False,
                                   asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def p4_from_s(s0, s1, s2, s3, k, no, *, loc=None, ip=None):
    """4 keys: e4m3(exp2(s * k + no)) packed into one 32-bit word (key 0 in the low byte)."""
    # two packed f32x2 FMAs (each lane an ordinary fma.rn.f32), the MUFU.EX2s (see ex2_p) and the e4m3 packing
    text = ("{\n\t.reg .b64 x01, x23, kk, nn;\n\t.reg .f32 y0, y1, y2, y3;\n\t.reg .b16 lo, hi;\n\t"
            "mov.b64 x01, {$1, $2};\n\tmov.b64 x23, {$3, $4};\n\t"
            "mov.b64 kk, {$5, $5};\n\tmov.b64 nn, {$6, $6};\n\t"
            "fma.rn.f32x2 x01, x01, kk, nn;\n\tfma.rn.f32x2 x23, x23, kk, nn;\n\t"
            "mov.b64 {y0, y1}, x01;\n\tmov.b64 {y2, y3}, x23;\n\t"
            "ex2.approx.ftz.f32 y0, y0;\n\tex2.approx.ftz.f32 y1, y1;\n\t"
            "ex2.approx.ftz.f32 y2, y2;\n\tex2.approx.ftz.f32 y3, y3;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 lo, y1, y0;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 hi, y3, y2;\n\t"
            "mov.b32 $0, {lo, hi};\n\t}")
    return Int32(llvm.inline_asm(T.i32(), [_f32(x, loc, ip) for x in (s0, s1, s2, s3, k, no)], text,
                                 "=r,f,f,f,f,f,f", has_side_effects=False, is_align_stack=False,
                                 asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def p4_from_x(x0, x1, x2, x3, *, loc=None, ip=None):
    """4 keys: e4m3(exp2(x)) packed into one 32-bit word."""
    e0 = ex2_p(Float32(x0))
    e1 = ex2_p(Float32(x1))
    e2 = ex2_p(Float32(x2))
    e3 = ex2_p(Float32(x3))
    text = ("{\n\t.reg .b16 lo, hi;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\n\t"
            "mov.b32 $0, {lo, hi};\n\t}")
    return Int32(_asm(T.i32(), [_f32(x, loc, ip) for x in (e0, e1, e2, e3)], text, "=r,f,f,f,f"))


@dsl_user_op
def e4_from_s(s0, s1, s2, s3, k, nb, *, loc=None, ip=None):
    """4 keys: fp32 E = exp2(s * k + nb) (packed f32x2 FMAs, MUFU.EX2; see ex2_p for the ftz form)."""
    st = llvm.StructType.get_literal([T.f32()] * 4)
    text = ("{\n\t.reg .b64 x01, x23, kk, nn;\n\t.reg .f32 y0, y1, y2, y3;\n\t"
            "mov.b64 x01, {$4, $5};\n\tmov.b64 x23, {$6, $7};\n\t"
            "mov.b64 kk, {$8, $8};\n\tmov.b64 nn, {$9, $9};\n\t"
            "fma.rn.f32x2 x01, x01, kk, nn;\n\tfma.rn.f32x2 x23, x23, kk, nn;\n\t"
            "mov.b64 {y0, y1}, x01;\n\tmov.b64 {y2, y3}, x23;\n\t"
            "ex2.approx.ftz.f32 $0, y0;\n\tex2.approx.ftz.f32 $1, y1;\n\t"
            "ex2.approx.ftz.f32 $2, y2;\n\tex2.approx.ftz.f32 $3, y3;\n\t}")
    r = llvm.inline_asm(st, [_f32(x, loc, ip) for x in (s0, s1, s2, s3, k, nb)], text,
                        "=f,=f,=f,=f,f,f,f,f,f,f", has_side_effects=False, is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
    return [Float32(llvm.extractvalue(T.f32(), r, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def p4_from_e(e0, e1, e2, e3, c, *, loc=None, ip=None):
    """4 keys: e4m3(E * c) packed into one 32-bit word (packed f32x2 multiplies, paired satfinite conversion)."""
    text = ("{\n\t.reg .b64 x01, x23, cc;\n\t.reg .f32 y0, y1, y2, y3;\n\t.reg .b16 lo, hi;\n\t"
            "mov.b64 x01, {$1, $2};\n\tmov.b64 x23, {$3, $4};\n\tmov.b64 cc, {$5, $5};\n\t"
            "mul.rn.f32x2 x01, x01, cc;\n\tmul.rn.f32x2 x23, x23, cc;\n\t"
            "mov.b64 {y0, y1}, x01;\n\tmov.b64 {y2, y3}, x23;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 lo, y1, y0;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 hi, y3, y2;\n\t"
            "mov.b32 $0, {lo, hi};\n\t}")
    return Int32(llvm.inline_asm(T.i32(), [_f32(x, loc, ip) for x in (e0, e1, e2, e3, c)], text,
                                 "=r,f,f,f,f,f", has_side_effects=False, is_align_stack=False,
                                 asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


def _scol(p, mode):
    """TMEM column offset (from TM_S) of pair p's S in the small-batch kernel.  mode 1 (huge): the seventh pair sits
    in the O columns until pass 2; mode 2 (nine): pairs 6 / 7 in the two halves of the O columns and pair 8 in
    slot 0 (pair 0's E moved to shared memory first)."""
    if mode == 0:
        return p * HT
    if isinstance(p, int):
        if p == NS + 1:
            return TM_O - TM_S
        if mode == 2 and p == NS + 2:
            return TM_O + HT - TM_S
        if mode == 2 and p == NS + 3:
            return 0
        return p * HT
    off = cutlass.select_(p == NS + 1, Int32(TM_O - TM_S - (NS + 1) * HT), Int32(0))
    if mode == 2:
        off = off + cutlass.select_(p == NS + 2, Int32(TM_O + HT - TM_S - (NS + 2) * HT), Int32(0))
        off = off + cutlass.select_(p == NS + 3, Int32(-(NS + 3) * HT), Int32(0))
    return p * HT + off


@dsl_user_op
def pin_i32(x, *, loc=None, ip=None):
    """Identity through a side-effecting mov: keeps a value's computation where it is written."""
    return Int32(_asm(T.i32(), [_i32(x, loc, ip)], "mov.b32 $0, $1;", "=r,r"))


def _make_ld_nowait(w):
    @dsl_user_op
    def f(taddr, *, loc=None, ip=None):
        """Issue a 32x32b.x{w} TMEM load; the values must go through the matching wait before use."""
        st = llvm.StructType.get_literal([T.f32()] * w)
        regs = ", ".join(f"${i}" for i in range(w))
        r = _asm(st, [_i32(taddr, loc, ip)], "tcgen05.ld.sync.aligned.32x32b.x" + str(w) + ".b32 {" + regs + "}, [$"
                 + str(w) + "];", ",".join(["=f"] * w) + ",r")
        return [Float32(llvm.extractvalue(T.f32(), r, [i], loc=loc, ip=ip)) for i in range(w)]
    return f


def _make_wait_ld(w):
    @dsl_user_op
    def f(vals, *, loc=None, ip=None):
        """tcgen05.wait::ld with the in-flight registers tied through, so no use can move above the wait."""
        st = llvm.StructType.get_literal([T.f32()] * w)
        r = _asm(st, [_f32(v, loc, ip) for v in vals], "tcgen05.wait::ld.sync.aligned;",
                 ",".join(["=f"] * w) + "," + ",".join(str(i) for i in range(w)))
        return [Float32(llvm.extractvalue(T.f32(), r, [i], loc=loc, ip=ip)) for i in range(w)]
    return f


@dsl_user_op
def tmem_st16x2_x4(taddr, w, *, loc=None, ip=None):
    """tcgen05.st 16x32bx2.x4: threads 0-15 -> lanes base+0..15 at columns [c, c+4), threads 16-31 -> the same lanes at
    columns [c+4, c+8)."""
    _asm(None, [_i32(taddr, loc, ip)] + [_i32(x, loc, ip) for x in w],
         "tcgen05.st.sync.aligned.16x32bx2.x4.b32 [$0], 4, {$1, $2, $3, $4};\n\ttcgen05.wait::st.sync.aligned;",
         "r,r,r,r,r")


# TMEM helpers of PagedDecodeQ7Dual over the CW columns a softmax warp owns of every tile
tmem_ldred_max = _make_ldred("max", CW)
tmem_ldred_min = _make_ldred("min", CW)
tmem_ldredv_max = _make_ldred_vals("max", CW)
tmem_ldredv_min = _make_ldred_vals("min", CW)
tmem_ld_cw_nowait = _make_ld_nowait(CW)
tmem_wait_ld_cw = _make_wait_ld(CW)


class PagedDecodeQ7Dual:
    """Row-split, lane-packed variant for small batches.

    Each request is handled by G row groups x C KV splits.  A CTA owns the 112 / G query rows of its group (one M=64
    MMA tile) and a run of 64-key half pages taken in pairs: the first tile of a pair lives in TMEM lanes 0-15 of
    every sub-partition, the second one in lanes 16-31 (M=64 tcgen05 layout, lane alignment 0 / 16), so every
    softmax lane does useful work while the per-CTA partial that crosses DSMEM is half the size.  The two lane halves
    keep separate O / l accumulators that are added (exact fp32) in the epilogue.

    ``pos_scale``: the row max of s * scale is scale * max(s) (else scale * min(s)). ``empty_rows``: what a request
    with no keys gets in its output rows (see EMPTY_ROWS_MODES). ``groups``: query-row groups per request.
    ``prog``: the progressive split merge (row maxima pushed early, partials accumulated as they land, one barrier per
    source) instead of one barrier for all partials. ``ne``: early pairs of a run.
    """

    def __init__(self, cluster_size: int, pos_scale: bool, empty_rows: str, groups: int, prog: bool, ne: int):
        assert empty_rows in EMPTY_ROWS_MODES
        # split merge: True = row maxima pushed early, partials accumulated as they land (one barrier per source);
        # False = one barrier for all partials, then every load of an item in flight together (tiny partials)
        self.prog = prog
        # query-row groups per request: 2 x 56 rows, or more, smaller groups for the smallest batches (idle SMs;
        # a smaller partial shrinks the DSMEM merge, and the busiest TMEM sub-partition still holds 16 rows a tile)
        self.G = groups
        self.R = (ROWS + groups - 1) // groups  # row stride of the groups; the last one may be shorter
        self.C = cluster_size
        self.pos_scale = pos_scale
        self.zero_empty = empty_rows == "zero"
        self.nk = (2 * MAX_PAGES + cluster_size - 1) // cluster_size  # max half-page tiles per CTA
        self.npair = (self.nk + 1) // 2
        # big: one pair more than the resident S slots.  Its S lives in the P / l columns during pass 1
        # (TM_P == TM_S + NS * HT) and it is the first pair of pass 2.  To fit 12 K + 12 V tiles in shared
        # memory the e4m3 ones tile and the max exchange go over Q after pass 1 and the receive buffer drops the
        # (unused) slot of the CTA's own split.
        self.big = self.npair >= NS + 1
        # huge: a seventh pair, whose S lives in the (not yet written) O columns during pass 1; pass 2 runs pairs
        # 6, 5, 0, 1, ...  To fit 14 K tiles the V tiles of pairs 2-4 reuse the K slots once their S is computed;
        # pairs 5 and 6 (needed first in pass 2) get four dedicated V slots, pairs 0 and 1 (HUGE_DL tiles) too.
        self.huge = self.npair == NS + 2
        # nine: up to nine pairs (C4): pairs 6 / 7 in the O columns, pair 8 in slot 0 once pair 0 (an early pair) has
        # moved its E to shared memory; pass 2 runs pairs npr-1 .. 5, then 0 .. 4; the V tiles fill the K slots in
        # that order, each after the S of the slot's pair
        self.nine = self.npair >= NS + 3
        assert self.npair <= NS + 4
        self.smode = 2 if self.nine else (1 if self.huge else 0)
        self.nslot = self.nk + (4 + HUGE_DL if self.huge else 0)
        assert (not self.big) or (NP * (HT // 4) + 16 == HT)
        self.rows_per = (self.R + cluster_size - 1) // cluster_size
        # early pairs: the first ne pairs turn into fp32 E = 2^(s*k - m*k) as their S lands (MUFU work hidden under the
        # K stream); after the exact row max M is known their P is e4m3(E * 448 * 2^(m*k - M))
        self.ne = ne
        assert (not self.nine) or self.ne >= 1

    @cute.jit
    def _p_last(self, sv, t_p, p_empty, p_full, pp, npr, n, seqlen, h_begin, hw, half, lane, k, no):
        """P of the run's last pair pp (full, partial, or a single tile) from this thread's S values."""
        u = pp
        if cutlass.const_expr(self.big):
            u = cutlass.select_(npr > NS, Int32(0), pp)
        ps = u % NP
        if u >= NP:
            cute.arch.mbarrier_wait(p_empty + ps, phase=((u // NP) - 1) & 1)
        tc_fence_after()
        t = 2 * pp + half
        valid = cutlass.select_(t < n, seqlen - (h_begin + t) * HT - hw * CW, Int32(0))
        if 2 * pp + 1 < n:
            if cute.arch.vote_all_sync(valid >= CW):
                w = [p4_from_s(sv[4 * q], sv[4 * q + 1], sv[4 * q + 2], sv[4 * q + 3], k, no)
                     for q in range(CW // 4)]
                tmem_st_x8(t_p + ps * (HT // 4), w)
            else:
                xs = [cutlass.select_(j < valid, sv[j] * k + no, -Float32.inf) for j in range(CW)]
                w2 = [p4_from_x(xs[4 * q], xs[4 * q + 1], xs[4 * q + 2], xs[4 * q + 3])
                      for q in range(CW // 4)]
                tmem_st_x8(t_p + ps * (HT // 4), w2)
        else:
            # odd run: this pair has no second tile.  Its first tile's CW columns are split over the two thread
            # halves (CW/2 each; the upper half takes them from lane - 16), so no exponentials are spent on the
            # empty lanes 16-31.
            hi = [cute.arch.shuffle_sync(sv[CW // 2 + j], lane % 16) for j in range(CW // 2)]
            src8 = [cutlass.select_(half == 0, sv[j], hi[j]) for j in range(CW // 2)]
            valid8 = seqlen - (h_begin + 2 * pp) * HT - hw * CW - half * (CW // 2)
            xs8 = [cutlass.select_(j < valid8, src8[j] * k + no, -Float32.inf) for j in range(CW // 2)]
            w8 = [p4_from_x(xs8[4 * q], xs8[4 * q + 1], xs8[4 * q + 2], xs8[4 * q + 3])
                  for q in range(CW // 8)]
            tmem_st16x2_x4(t_p + ps * (HT // 4), w8)
        tc_fence_before()
        cute.arch.sync_warp()
        if lane == 0:
            cute.arch.mbarrier_arrive(p_full + ps)

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (112, 128, B) e4m3
        mK: cute.Tensor,  # (128 tok, 128 d, pages) e4m3, strides (token, 1, page), runtime values
        mV: cute.Tensor,  # (128 d, 128 tok, pages) e4m3, strides (1, token, page), runtime values
        mO: cute.Tensor,  # (B*112*128,) fp16
        mSeq: cute.Tensor,  # (B,) int32
        mBT: cute.Tensor,  # (B, 34) int32
        scale_log2: Float32,
        scale_o: Float32,
        stream: cuda.CUstream,
    ):
        mma_dtype = mQ.element_type
        acc_dtype = Float32
        tiled_mma_qk = sm100_utils.make_trivial_tiled_mma(
            mma_dtype, mma_dtype, OperandMajorMode.K, OperandMajorMode.K, acc_dtype,
            tcgen05.CtaGroup.ONE, (64, HT), tcgen05.OperandSource.SMEM,
        )
        tiled_mma_pv = sm100_utils.make_trivial_tiled_mma(
            mma_dtype, mma_dtype, OperandMajorMode.K, OperandMajorMode.MN, acc_dtype,
            tcgen05.CtaGroup.ONE, (64, D), tcgen05.OperandSource.TMEM,
        )
        tiled_mma_pl = sm100_utils.make_trivial_tiled_mma(
            mma_dtype, mma_dtype, OperandMajorMode.K, OperandMajorMode.K, acc_dtype,
            tcgen05.CtaGroup.ONE, (64, 16), tcgen05.OperandSource.TMEM,
        )
        smem_layout_q = sm100_utils.make_smem_layout_a(tiled_mma_qk, (64, HT, D), mma_dtype, 1)
        smem_layout_k = sm100_utils.make_smem_layout_b(tiled_mma_qk, (64, HT, D), mma_dtype, self.nslot)
        smem_layout_v = sm100_utils.make_smem_layout_b(tiled_mma_pv, (64, D, HT), mma_dtype, self.nslot)

        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_q = cute.nvgpu.make_tiled_tma_atom_A(tma_load_op, mQ, smem_layout_q, (64, HT, D), tiled_mma_qk)
        tma_k = cute.nvgpu.make_tiled_tma_atom_B(tma_load_op, mK, smem_layout_k, (64, HT, D), tiled_mma_qk)
        tma_v = cute.nvgpu.make_tiled_tma_atom_B(tma_load_op, mV, smem_layout_v, (64, D, HT), tiled_mma_pv)

        batches = mSeq.shape[0]
        self.kernel(
            mma_dtype, tiled_mma_qk, tiled_mma_pv, tiled_mma_pl, tma_q, tma_k, tma_v,
            mO, mSeq, mBT, scale_log2, scale_o,
        ).launch(
            grid=[self.C, self.G * batches, 1],
            block=[NTHREADS, 1, 1],
            cluster=[self.C, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mma_dtype: cutlass.Constexpr,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_mma_pl: cute.TiledMma,
        tma_q: cute.nvgpu.cpasync.TmaInfo,
        tma_k: cute.nvgpu.cpasync.TmaInfo,
        tma_v: cute.nvgpu.cpasync.TmaInfo,
        mO: cute.Tensor,
        mSeq: cute.Tensor,
        mBT: cute.Tensor,
        scale_log2: Float32,
        scale_o: Float32,
    ):
        C = self.C
        NK = self.nk
        rows_per = self.rows_per
        acc_dtype = Float32
        LANE16 = 16 * 65536  # TMEM address offset of lane 16

        rank, yb, _ = cute.arch.block_idx()
        RS = self.R
        coord_b = yb // self.G
        grp = yb % self.G
        RG = Int32(RS)  # rows of this group
        if cutlass.const_expr(ROWS % self.G != 0):
            RG = cutlass.select_(grp == self.G - 1, Int32(ROWS - (self.G - 1) * RS), Int32(RS))
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(tidx // 32)
        lane = cute.arch.lane_idx()

        tma_warp = SM_WARPS
        mma_warp = SM_WARPS + 1

        # ---------------- shared memory ----------------
        smem = cutlass.utils.SmemAllocator()
        tmem_holder = smem.allocate_array(Int32, 1)
        q_full = smem.allocate_array(Int64, 1)
        k_full = smem.allocate_array(Int64, NK)
        v_full = smem.allocate_array(Int64, NK)
        NSF = max(NS + 1, self.npair)
        s_full = smem.allocate_array(Int64, NSF)
        p_full = smem.allocate_array(Int64, NP)
        p_empty = smem.allocate_array(Int64, NP)
        o_full = smem.allocate_array(Int64, 1)
        recv_mbar = smem.allocate_array(Int64, C)  # one per receive slot: partials are merged as they land
        stats_mbar = smem.allocate_array(Int64, 1)
        sMeta = smem.allocate_array(Int32, 4)
        # row maxima of my rows from every split: [split][row], pushed right after pass 2
        sStats = smem.allocate_tensor(acc_dtype, cute.make_layout(C * rows_per), 16)
        if cutlass.const_expr(not self.big):
            sMax = smem.allocate_tensor(acc_dtype, cute.make_layout(NWG * 64), 16)
        smem_layout_ones = sm100_utils.make_smem_layout_b(tiled_mma_pl, (64, 16, HT), mma_dtype, 1)
        if cutlass.const_expr(not self.big):
            ones_raw = smem.allocate(1024, byte_alignment=1024)
        tAsQ = smem.allocate_tensor(mma_dtype, tma_q.smem_layout.outer, 1024, tma_q.smem_layout.inner)
        if cutlass.const_expr(self.big):
            # both used only after pass 1, when Q is dead
            ones_raw = tAsQ.iterator
            sMax = cute.make_tensor(cute.recast_ptr(tAsQ.iterator + 4096, dtype=acc_dtype), cute.make_layout(NWG * 64))
        if cutlass.const_expr(self.nine):
            sE = smem.allocate_tensor(acc_dtype, cute.make_layout(SM_THREADS * CW), 16)  # pair 0's E: [j][thread]
            e0_bar = smem.allocate_array(Int64, 1)
        if cutlass.const_expr(self.huge or self.nine):
            # one pool of K / V tile slots, K tile t in slot t.  huge: V tile t in slot NK + 4 + t (pairs 0, 1), in
            # slot t once its S is computed (pairs 2-4) or in slot NK + t - 2 NS (pairs 5, 6); nine: the V tiles fill
            # slots 0, 1, ... in pass-2 order, each once the S of the slot's pair is computed
            kv_raw = smem.allocate(self.nslot * HT * D, byte_alignment=1024)
            tBsK = cute.make_tensor(cute.recast_ptr(kv_raw, tma_k.smem_layout.inner, dtype=mma_dtype),
                                    tma_k.smem_layout.outer)
            tBsV = cute.make_tensor(cute.recast_ptr(kv_raw, tma_v.smem_layout.inner, dtype=mma_dtype),
                                    tma_v.smem_layout.outer)
        else:
            tBsK = smem.allocate_tensor(mma_dtype, tma_k.smem_layout.outer, 1024, tma_k.smem_layout.inner)
            tBsV = smem.allocate_tensor(mma_dtype, tma_v.smem_layout.outer, 1024, tma_v.smem_layout.inner)
        NRS = C - 1 if self.big else C  # receive slots (big: none for the own split)
        sRecv = smem.allocate_tensor(acc_dtype, cute.make_layout(NRS * rows_per * ROW_F), 16)

        q_bytes = cute.size_in_bytes(mma_dtype, tma_q.smem_layout)
        kv_bytes = HT * D  # one 64-key K (or V) tile, e4m3

        thr_mma_qk = tiled_mma_qk.get_slice(0)
        thr_mma_pv = tiled_mma_pv.get_slice(0)
        thr_mma_pl = tiled_mma_pl.get_slice(0)
        mcast_coord = (0, 0, 0, 0)
        mcast_layout = cute.make_layout((1, 1, 1, 1))

        # ================= TMA producer: barriers, block table, Q/K/V =================
        if warp_idx == tma_warp:
            # both metadata loads in flight together; Q needs neither and goes out before they land
            pid0 = mBT[coord_b, lane]
            pid1 = Int32(0)
            if lane < MAX_PAGES - 32:
                pid1 = mBT[coord_b, lane + 32]
            seqlen_p = mSeq[coord_b]
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(q_full, 1)
                for i in cutlass.range_constexpr(NK):
                    cute.arch.mbarrier_init(k_full + i, 1)
                    cute.arch.mbarrier_init(v_full + i, 1)
                if cutlass.const_expr(self.huge or self.nine):
                    # S barriers here: with `huge` / `nine` this warp waits on them before reusing K slots for V
                    for i in cutlass.range_constexpr(NSF):
                        cute.arch.mbarrier_init(s_full + i, 1)
                cute.arch.mbarrier_init_fence()
            gQ = cute.local_tile(cute.domain_offset((grp * RS, 0, 0), tma_q.tma_tensor),
                                 tiler=(64, D), coord=(0, 0, coord_b))
            tAgQ = thr_mma_qk.partition_A(gQ)
            tAsQ_tma, tAgQ_tma = cute.nvgpu.cpasync.tma_partition(
                tma_q.atom, mcast_coord, mcast_layout, cute.group_modes(tAsQ, 0, 3), cute.group_modes(tAgQ, 0, 3))
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(q_full, q_bytes)
            cute.copy(tma_q.atom, tAgQ_tma, tAsQ_tma[None, 0], tma_bar_ptr=q_full)

            # a nonpositive length is a request with no keys: no tiles in any split, no page reads
            seqlen_p = cutlass.select_(seqlen_p > 0, seqlen_p, Int32(0))
            nh_p = (seqlen_p + (HT - 1)) // HT
            h_begin_p = (rank * nh_p) // C
            h_end_p = ((rank + 1) * nh_p) // C
            n_p = h_end_p - h_begin_p
            npr_p = (n_p + 1) // 2
            if lane == 0:
                sMeta[0] = seqlen_p
                sMeta[1] = h_begin_p
                sMeta[2] = n_p
                sMeta[3] = npr_p
            cute.arch.barrier_arrive(barrier_id=1, number_of_threads=NTHREADS)
            cute.arch.barrier_arrive(barrier_id=3, number_of_threads=96)
            # peers may target recv_mbar only after the cluster barrier
            cute.arch.cluster_arrive_relaxed()

            gK = cute.local_tile(tma_k.tma_tensor, tiler=(HT, D), coord=(None, 0, None))
            tBgK = thr_mma_qk.partition_B(gK)
            tBsK_tma, tBgK_tma = cute.nvgpu.cpasync.tma_partition(
                tma_k.atom, mcast_coord, mcast_layout, cute.group_modes(tBsK, 0, 3), cute.group_modes(tBgK, 0, 3))
            gV = cute.local_tile(tma_v.tma_tensor, tiler=(D, HT), coord=(0, None, None))
            tBgV = thr_mma_pv.partition_B(gV)
            tBsV_tma, tBgV_tma = cute.nvgpu.cpasync.tma_partition(
                tma_v.atom, mcast_coord, mcast_layout, cute.group_modes(tBsV, 0, 3), cute.group_modes(tBgV, 0, 3))
            for t in cutlass.range(n_p):
                g = h_begin_p + t
                pidx = g // 2
                src = pid0
                if pidx >= 32:
                    src = pid1
                page = cute.arch.shuffle_sync(src, pidx % 32)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(k_full + t, kv_bytes)
                cute.copy(tma_k.atom, tBgK_tma[None, g % 2, page], tBsK_tma[None, t], tma_bar_ptr=k_full + t)
            # V requests wait for the first K tile to land, so the K stream (needed first) is not diluted
            if cutlass.const_expr(self.huge):
                if n_p > 0:
                    cute.arch.mbarrier_wait(k_full + cutlass.select_(n_p > HUGE_VW, Int32(HUGE_VW), n_p - 1), phase=0)
            else:
                if n_p > 0:
                    cute.arch.mbarrier_wait(k_full, phase=0)
            if cutlass.const_expr(self.nine):
                # V tiles in pass-2 order (pairs npr-1 .. 5, then 0 .. 4) into the K slots in order: slot k once the S
                # of its pair k // 2 is complete
                kpos = Int32(0)
                for j in cutlass.range(npr_p):
                    pj = j
                    if npr_p > NS:
                        pj = cutlass.select_(j < npr_p - NS, npr_p - 1 - j, j - (npr_p - NS))
                    for e in cutlass.range_constexpr(2):
                        tv = 2 * pj + e
                        if tv < n_p:
                            cute.arch.mbarrier_wait(s_full + kpos // 2, phase=0)
                            g = h_begin_p + tv
                            pidx = g // 2
                            src = pid0
                            if pidx >= 32:
                                src = pid1
                            page = cute.arch.shuffle_sync(src, pidx % 32)
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive_and_expect_tx(v_full + tv, kv_bytes)
                            cute.copy(tma_v.atom, tBgV_tma[None, g % 2, page], tBsV_tma[None, kpos], tma_bar_ptr=v_full + tv)
                            kpos = kpos + 1
            elif cutlass.const_expr(self.huge):
                # pairs 6 and 5 first (pass-2 order) into their dedicated slots, then pairs 0-4 into their K slots,
                # each once its S MMA has completed
                for e in cutlass.range_constexpr(4):
                    tv = 2 * (NS + 1) + e if e < 2 else 2 * NS + e - 2
                    if tv < n_p:
                        g = h_begin_p + tv
                        pidx = g // 2
                        src = pid0
                        if pidx >= 32:
                            src = pid1
                        page = cute.arch.shuffle_sync(src, pidx % 32)
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(v_full + tv, kv_bytes)
                        cute.copy(tma_v.atom, tBgV_tma[None, g % 2, page], tBsV_tma[None, NK + tv - 2 * NS],
                                  tma_bar_ptr=v_full + tv)
                # the first HUGE_DL tiles (pairs 0, 1) get dedicated slots too: no wait on their S
                for e in cutlass.range_constexpr(HUGE_DL):
                    if e < n_p:
                        g = h_begin_p + e
                        pidx = g // 2
                        src = pid0
                        if pidx >= 32:
                            src = pid1
                        page = cute.arch.shuffle_sync(src, pidx % 32)
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(v_full + e, kv_bytes)
                        cute.copy(tma_v.atom, tBgV_tma[None, g % 2, page], tBsV_tma[None, NK + 4 + e],
                                  tma_bar_ptr=v_full + e)
                n_low = cutlass.select_(n_p > 2 * NS, Int32(2 * NS), n_p)
                for tv in cutlass.range(HUGE_DL, n_low):
                    if tv % 2 == 0:
                        cute.arch.mbarrier_wait(s_full + tv // 2, phase=0)
                    g = h_begin_p + tv
                    pidx = g // 2
                    src = pid0
                    if pidx >= 32:
                        src = pid1
                    page = cute.arch.shuffle_sync(src, pidx % 32)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(v_full + tv, kv_bytes)
                    cute.copy(tma_v.atom, tBgV_tma[None, g % 2, page], tBsV_tma[None, tv], tma_bar_ptr=v_full + tv)
            else:
                # V in pass-2 order: with `big` and a sixth pair, that pair's P.V comes first (its S sits under P stage 0),
                # so its V tiles go first; otherwise P stage 0 is recycled only after the last V tiles land
                nl_p = n_p - 2 * (npr_p - 1)  # tiles of the run's last pair (1 or 2)
                for t in cutlass.range(n_p):
                    tv = t
                    if cutlass.const_expr(self.big):
                        if npr_p > NS:
                            tv = cutlass.select_(t < nl_p, 2 * (npr_p - 1) + t, t - nl_p)
                    g = h_begin_p + tv
                    pidx = g // 2
                    src = pid0
                    if pidx >= 32:
                        src = pid1
                    page = cute.arch.shuffle_sync(src, pidx % 32)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(v_full + tv, kv_bytes)
                    cute.copy(tma_v.atom, tBgV_tma[None, g % 2, page], tBsV_tma[None, tv], tma_bar_ptr=v_full + tv)

        # ================= MMA issue =================
        # every role branches to its own code at kernel start: the cold instruction fetch of the MMA and
        # softmax code overlaps the block-table / Q / K latency instead of following the setup rendezvous
        elif warp_idx == mma_warp:
            with cute.arch.elect_one():
                if cutlass.const_expr(not (self.huge or self.nine)):
                    for i in cutlass.range_constexpr(NSF):
                        cute.arch.mbarrier_init(s_full + i, 1)
                if cutlass.const_expr(self.nine):
                    cute.arch.mbarrier_init(e0_bar, SM_WARPS)
                for i in cutlass.range_constexpr(NP):
                    cute.arch.mbarrier_init(p_full + i, SM_WARPS)
                    cute.arch.mbarrier_init(p_empty + i, 1)
                cute.arch.mbarrier_init(o_full, 1)
                r0 = rank * rows_per
                nrows = RG - r0
                if nrows > rows_per:
                    nrows = Int32(rows_per)
                if nrows < 0:
                    nrows = Int32(0)
                if cutlass.const_expr(self.prog):
                    cute.arch.mbarrier_init(stats_mbar, 1)
                    for i in cutlass.range_constexpr(C - 1 if self.big else C):
                        cute.arch.mbarrier_init(recv_mbar + i, 1)
                    cute.arch.mbarrier_arrive_and_expect_tx(stats_mbar, (C - 1) * nrows * 4)
                    for i in cutlass.range_constexpr(C - 1 if self.big else C):
                        cute.arch.mbarrier_arrive_and_expect_tx(recv_mbar + i, nrows * ROW_B)
                else:
                    cute.arch.mbarrier_init(recv_mbar, 1)
                    cute.arch.mbarrier_arrive_and_expect_tx(recv_mbar, (C - 1) * nrows * ROW_B)
                cute.arch.mbarrier_init_fence()
            cute.arch.sync_warp()
            # one sync site per named barrier: the MMA warp arrives on barrier 1 (the softmax warps sync there)
            # and syncs on barrier 3 (TMA warp: run metadata written; warp 0: TMEM allocated)
            cute.arch.barrier_arrive(barrier_id=1, number_of_threads=NTHREADS)
            cute.arch.barrier(barrier_id=3, number_of_threads=96)
            # peers may target recv_mbar only after the cluster barrier
            cute.arch.cluster_arrive_relaxed()
            # The TMA warp loads and partitions seq_lens once; the role rendezvous above publishes these values.
            seqlen = sMeta[0]
            h_begin = sMeta[1]
            n = sMeta[2]
            npr = sMeta[3]

            tmem_ptr = cute.arch.retrieve_tmem_ptr(acc_dtype, 16, tmem_holder)
            s_lay = tiled_mma_qk.make_fragment_C(tiled_mma_qk.partition_shape_C((64, HT))).layout
            o_lay = tiled_mma_pv.make_fragment_C(tiled_mma_pv.partition_shape_C((64, D))).layout
            p_lay = tiled_mma_pv.make_fragment_A(tiled_mma_pv.partition_shape_A((64, HT))).layout
            l_lay = tiled_mma_pl.make_fragment_C(tiled_mma_pl.partition_shape_C((64, 16))).layout
            # explicit TMEM columns; the "_b" twins sit 16 lanes further (second tile of every pair)
            tCtO = cute.make_tensor(tmem_ptr + TM_O, o_lay)
            tCtO_b = cute.make_tensor(tmem_ptr + (TM_O + LANE16), o_lay)
            tCtL = cute.make_tensor(tmem_ptr + TM_L, l_lay)
            tCtL_b = cute.make_tensor(tmem_ptr + (TM_L + LANE16), l_lay)
            smem_ones = cute.make_tensor(cute.recast_ptr(ones_raw, smem_layout_ones.inner, dtype=mma_dtype),
                                         smem_layout_ones.outer)

            tAsQ_d = thr_mma_qk.make_fragment_A(tAsQ)
            tBsK_d = thr_mma_qk.make_fragment_B(tBsK)
            tBsV_d = thr_mma_pv.make_fragment_B(tBsV)
            tBsOnes_d = thr_mma_pl.make_fragment_B(smem_ones)
            num_kq = cute.size(tBsK.shape[2])
            num_kv = cute.size(tBsV.shape[2])
            cute.arch.mbarrier_wait(q_full, phase=0)
            # pass 1: S of every pair of the run (tile 2p -> lanes 0-15, tile 2p+1 -> lanes 16-31)
            for p in cutlass.range(npr):
                cute.arch.mbarrier_wait(k_full + 2 * p, phase=0)
                tc_fence_after()
                # S column of pair p (huge: the seventh pair sits in the O columns until pass 2)
                scp = _scol(p, self.smode)
                if cutlass.const_expr(self.nine):
                    # pair 8 goes to slot 0: pair 0's E must have left TMEM
                    if p == NS + 3:
                        cute.arch.mbarrier_wait(e0_bar, phase=0)
                        tc_fence_after()
                tCtS_s = cute.make_tensor(tmem_ptr + (TM_S + scp), s_lay)
                tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, False)
                for kb in cutlass.range_constexpr(num_kq):
                    cute.gemm(tiled_mma_qk, tCtS_s, tAsQ_d[None, None, kb, 0], tBsK_d[None, None, kb, 2 * p], tCtS_s)
                    tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, True)
                if 2 * p + 1 < n:
                    cute.arch.mbarrier_wait(k_full + 2 * p + 1, phase=0)
                    tc_fence_after()
                    tCtS_sb = cute.make_tensor(tmem_ptr + (TM_S + LANE16 + scp), s_lay)
                    tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, False)
                    for kb in cutlass.range_constexpr(num_kq):
                        cute.gemm(tiled_mma_qk, tCtS_sb, tAsQ_d[None, None, kb, 0], tBsK_d[None, None, kb, 2 * p + 1],
                                  tCtS_sb)
                        tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(s_full + p)
            # pass 2: O += P V, l += P 1 per lane half
            for u in cutlass.range(npr):
                p = u
                if cutlass.const_expr(self.big):
                    p = cutlass.select_(npr > NS, (u + NS) % (NS + 1), u)
                if cutlass.const_expr(self.huge):
                    p = cutlass.select_(npr > NS + 1,
                                        cutlass.select_(u == 0, Int32(NS + 1), cutlass.select_(u == 1, Int32(NS), u - 2)), p)
                # V slot of the pair's tiles (huge: pairs 5 and 6 in the dedicated slots)
                va = 2 * p
                vb = 2 * p + 1
                if cutlass.const_expr(self.huge):
                    va = cutlass.select_(p >= NS, NK + 2 * p - 2 * NS,
                                         cutlass.select_(2 * p < HUGE_DL, NK + 4 + 2 * p, 2 * p))
                    vb = va + 1
                if cutlass.const_expr(self.nine):
                    p = cutlass.select_(npr > NS, cutlass.select_(u < npr - NS, npr - 1 - u, u - (npr - NS)), u)
                    # V slot = position of the tile in the pass-2 order (the run's last pair, first, may be single)
                    nl = n - 2 * (npr - 1)
                    va = cutlass.select_(npr > NS, cutlass.select_(u > 0, nl + 2 * (u - 1), Int32(0)), 2 * u)
                    vb = va + 1
                ps = u % NP
                cute.arch.mbarrier_wait(v_full + 2 * p, phase=0)
                cute.arch.mbarrier_wait(p_full + ps, phase=(u // NP) & 1)
                tc_fence_after()
                tAtP_s = cute.make_tensor(cute.recast_ptr(tmem_ptr + (TM_P + ps * (HT // 4)), dtype=mma_dtype), p_lay)
                tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, u > 0)
                tiled_mma_pl.set(tcgen05.Field.ACCUMULATE, u > 0)
                for kb in cutlass.range_constexpr(num_kv):
                    cute.gemm(tiled_mma_pv, tCtO, tAtP_s[None, None, kb], tBsV_d[None, None, kb, va], tCtO)
                    cute.gemm(tiled_mma_pl, tCtL, tAtP_s[None, None, kb], tBsOnes_d[None, None, kb, 0], tCtL)
                    tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, True)
                    tiled_mma_pl.set(tcgen05.Field.ACCUMULATE, True)
                if 2 * p + 1 < n:
                    cute.arch.mbarrier_wait(v_full + 2 * p + 1, phase=0)
                    tc_fence_after()
                    tAtP_sb = cute.make_tensor(
                        cute.recast_ptr(tmem_ptr + (TM_P + LANE16 + ps * (HT // 4)), dtype=mma_dtype), p_lay)
                    # odd tiles: with `big`, the reordered first pair may lack one (odd run length)
                    acc_b = u > 0
                    if cutlass.const_expr(self.big):
                        acc_b = u > cutlass.select_((npr > NS) and (2 * (npr - 1) + 1 >= n), Int32(1), Int32(0))
                    tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, acc_b)
                    tiled_mma_pl.set(tcgen05.Field.ACCUMULATE, acc_b)
                    for kb in cutlass.range_constexpr(num_kv):
                        cute.gemm(tiled_mma_pv, tCtO_b, tAtP_sb[None, None, kb], tBsV_d[None, None, kb, vb],
                                  tCtO_b)
                        cute.gemm(tiled_mma_pl, tCtL_b, tAtP_sb[None, None, kb], tBsOnes_d[None, None, kb, 0], tCtL_b)
                        tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, True)
                        tiled_mma_pl.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(p_empty + ps)
            if n > 0:
                with cute.arch.elect_one():
                    tcgen05.commit(o_full)

        # ================= softmax (warps 0-15) =================
        elif warp_idx < SM_WARPS:
            if warp_idx == 0:
                cute.nvgpu.cpasync.prefetch_descriptor(tma_q.atom)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_k.atom)
                cute.nvgpu.cpasync.prefetch_descriptor(tma_v.atom)
                cute.arch.alloc_tmem(TM_COLS, tmem_holder)
                tc_fence_before()
                cute.arch.barrier_arrive(barrier_id=3, number_of_threads=96)
            # e4m3 ones (0x38) as the B operand of the row-sum MMA
            if tidx < 256:
                if cutlass.const_expr(not self.big):
                    ones_i32 = cute.make_tensor(cute.recast_ptr(ones_raw, dtype=Int32), cute.make_layout(256))
                    ones_i32[tidx] = Int32(0x38383838)
                cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier(barrier_id=1, number_of_threads=NTHREADS)
            cute.arch.cluster_arrive_relaxed()
            # The TMA warp loads and partitions seq_lens once; the role rendezvous above publishes these values.
            seqlen = sMeta[0]
            h_begin = sMeta[1]
            n = sMeta[2]
            npr = sMeta[3]

            tmem_ptr = cute.arch.retrieve_tmem_ptr(acc_dtype, 16, tmem_holder)

            sp = warp_idx % 4  # TMEM sub-partition
            hw = warp_idx // 4  # column group of every tile: [CW*hw, CW*hw + CW)
            half = lane // 16  # 0: first tile of a pair, 1: second tile
            row = sp * 16 + lane % 16  # local query row (0..63; < RG real)
            tc_fence_after()
            tbase = cute.make_tensor(tmem_holder, cute.make_layout(1))[0]
            lane_off = (sp * 32) << 16
            t_s = tbase + lane_off + TM_S + hw * CW
            t_p = tbase + lane_off + TM_P + hw * (CW // 4)
            k = scale_log2
            SMODE = self.smode

            # short runs (< 2 tiles) leave lanes 16-31 (or all lanes) of O / l unwritten by the MMAs: zero them
            # before the first P is published, so the epilogue can add the two lane halves unconditionally
            if n < 2:
                for zz in cutlass.range_constexpr(4 // NWG):
                    tmem_st_x32f(tbase + lane_off + TM_O + hw * (128 // NWG) + 32 * zz, [Float32(0)] * 32)
                if hw == 0:
                    tmem_st_x1f(tbase + lane_off + TM_L, Float32(0))
                tc_fence_before()
            # ---- pass 1: exact row max (min for a non-positive scale) over the run ----
            mx = -Float32.inf
            if cutlass.const_expr(not self.pos_scale):
                mx = Float32.inf
            # early pairs (full tiles): each thread only touches its own CW columns, whose E (fp32) goes back over S;
            # mt is the running max of this thread's own values, the per-thread reference of its E
            NE = self.ne
            mt = -Float32.inf
            if cutlass.const_expr(not self.pos_scale):
                mt = Float32.inf
            msl = []
            for uu in cutlass.range_constexpr(NE):
                if uu < npr - 1:
                    cute.arch.mbarrier_wait(s_full + uu, phase=0)
                    tc_fence_after()
                    if cutlass.const_expr(self.pos_scale):
                        rve = tmem_ldredv_max(t_s + uu * HT)
                        mt = cute.arch.fmax(mt, rve[CW])
                    else:
                        rve = tmem_ldredv_min(t_s + uu * HT)
                        mt = cute.arch.fmin(mt, rve[CW])
                    nbe = -(mt * k)
                    eve = []
                    for q in cutlass.range_constexpr(CW // 4):
                        eve = eve + e4_from_s(rve[4 * q], rve[4 * q + 1], rve[4 * q + 2], rve[4 * q + 3], k, nbe)
                    if cutlass.const_expr(self.nine and uu == 0):
                        # nine: pair 0's E goes to shared memory ([j][thread]) and its TMEM slot to pair 8's S
                        for j in cutlass.range_constexpr(CW):
                            sE[j * SM_THREADS + tidx] = eve[j]
                        tc_fence_before()
                        cute.arch.sync_warp()
                        if lane == 0:
                            cute.arch.mbarrier_arrive(e0_bar)
                    else:
                        tmem_st_x32f(t_s + uu * HT, eve)
                msl.append(mt * k)
            # warp hw reduces whole later pairs p = NE + hw, NE + hw + NWG, ... (all 64 columns of its lane in one
            # ld.red; never an early pair, whose other columns hold E), so the per-warp chain of waited TMEM loads
            # is short; the exchange below combines the warps
            t_s0 = tbase + lane_off + TM_S
            for p in cutlass.range(NE + hw, npr - 1, NWG):
                cute.arch.mbarrier_wait(s_full + p, phase=0)
                tc_fence_after()
                t = 2 * p + half
                valid = cutlass.select_(t < n, seqlen - (h_begin + t) * HT, Int32(0))
                if cute.arch.vote_all_sync(valid >= HT):
                    if cutlass.const_expr(self.pos_scale):
                        mx = cute.arch.fmax(mx, tmem_ldred64_max(t_s0 + _scol(p, SMODE)))
                    else:
                        mx = cute.arch.fmin(mx, tmem_ldred64_min(t_s0 + _scol(p, SMODE)))
                else:
                    for hh in cutlass.range_constexpr(2):
                        rv = tmem_ld_x32(t_s0 + _scol(p, SMODE) + hh * 32)
                        for j in cutlass.range_constexpr(32):
                            if cutlass.const_expr(self.pos_scale):
                                mx = cute.arch.fmax(mx, cutlass.select_(hh * 32 + j < valid, rv[j], -Float32.inf))
                            else:
                                mx = cute.arch.fmin(mx, cutlass.select_(hh * 32 + j < valid, rv[j], Float32.inf))
            # every S of the run is complete once the last pair's commit has fired (pass 2 and, with `big`, the
            # ones tile over Q rely on it; this warp may not have waited on that pair above)
            # the last pair (the one on the critical path once its K tiles land) is split over the NWG warps:
            # CW columns each instead of one warp reducing all 64
            if npr > 0:
                cute.arch.mbarrier_wait(s_full + npr - 1, phase=0)
                tc_fence_after()
                t = 2 * (npr - 1) + half
                valid = cutlass.select_(t < n, seqlen - (h_begin + t) * HT - hw * CW, Int32(0))
                if cute.arch.vote_all_sync(valid >= CW):
                    if cutlass.const_expr(self.pos_scale):
                        mx = cute.arch.fmax(mx, tmem_ldred_max(t_s + _scol(npr - 1, SMODE)))
                    else:
                        mx = cute.arch.fmin(mx, tmem_ldred_min(t_s + _scol(npr - 1, SMODE)))
                else:
                    rvl = tmem_ldredv_max(t_s + _scol(npr - 1, SMODE))
                    for j in cutlass.range_constexpr(CW):
                        if cutlass.const_expr(self.pos_scale):
                            mx = cute.arch.fmax(mx, cutlass.select_(j < valid, rvl[j], -Float32.inf))
                        else:
                            mx = cute.arch.fmin(mx, cutlass.select_(j < valid, rvl[j], Float32.inf))
            tc_fence_after()
            if cutlass.const_expr(self.pos_scale):
                mx = cute.arch.fmax(mx, mt)
                mx = cute.arch.fmax(mx, cute.arch.shuffle_sync_bfly(mx, 16))
            else:
                mx = cute.arch.fmin(mx, mt)
                mx = cute.arch.fmin(mx, cute.arch.shuffle_sync_bfly(mx, 16))
            if half == 0:
                sMax[hw * 64 + row] = mx
            # S of the first pass-2 pair is read before the barrier: with `big` it sits under P stage 0
            pr0 = Int32(0)
            if cutlass.const_expr(self.big):
                pr0 = cutlass.select_(npr > NS, npr - 1, Int32(0))
            if cutlass.const_expr(self.nine):
                # pairs 5 (under the P stages / l), 6 and 7 (O columns) are overwritten once pass 2 starts: read them
                # now (pair 8, in slot 0, is read after the barrier)
                sv5_n = tmem_ld_cw_nowait(t_s + NS * HT)
                sv6_n = tmem_ld_cw_nowait(t_s + _scol(NS + 1, SMODE))
                sv7_n = tmem_ld_cw_nowait(t_s + _scol(NS + 2, SMODE))
            else:
                sv0_n = tmem_ld_cw_nowait(t_s + _scol(pr0, SMODE))  # waited after the max exchange
            if cutlass.const_expr(self.huge):
                # with a seventh pair, pair 5 (under the P stages / l) is the second pass-2 pair: read it now too
                sv1_n = tmem_ld_cw_nowait(t_s + NS * HT)
            if cutlass.const_expr(self.big):
                # every S of the run is computed (this thread waited on all of them, so Q is dead): the row-sum
                # ones tile goes over Q; the max-exchange barrier below orders it before any P / P.1 MMA
                ones_q = cute.make_tensor(cute.recast_ptr(ones_raw, dtype=Int32), cute.make_layout(256))
                if tidx < 256:
                    ones_q[tidx] = Int32(0x38383838)
                cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier(barrier_id=2, number_of_threads=SM_THREADS)
            for o_hw in cutlass.range_constexpr(1, NWG):
                other = sMax[((hw + o_hw) % NWG) * 64 + row]
                if cutlass.const_expr(self.pos_scale):
                    mx = cute.arch.fmax(mx, other)
                else:
                    mx = cute.arch.fmin(mx, other)
            # row max of the scaled scores (log2 units); -inf if the run is empty
            m_scaled = cutlass.select_(n > 0, mx * k, -Float32.inf)
            no = LOG2_P_SCALE - m_scaled
            if cutlass.const_expr(self.nine):
                sv5 = tmem_wait_ld_cw(sv5_n)
                sv6 = tmem_wait_ld_cw(sv6_n)
                sv7 = tmem_wait_ld_cw(sv7_n)
            else:
                sv0 = tmem_wait_ld_cw(sv0_n)
            if cutlass.const_expr(self.huge):
                sv1 = tmem_wait_ld_cw(sv1_n)

            # ---- pass 2: P = 448 * exp2(s*k - m) -> e4m3 into TMEM, in the MMA's pair order.  Only the run's last pair
            # can hold a partial or missing tile: the others take the full-tile path, the last one is emitted once
            # (first with `big` and a sixth pair, whose S sits under P stage 0) ----
            if cutlass.const_expr(self.nine):
                # pass-2 order: pairs npr-1 .. 5 (the run's last pair first), then 0 .. 4; pair 0's E from smem
                ntop = cutlass.select_(npr > NS, npr - NS, Int32(0))
                sv8 = tmem_wait_ld_cw(tmem_ld_cw_nowait(t_s))
                for jj in cutlass.range_constexpr(4):
                    if jj < ntop:
                        pt = npr - 1 - jj
                        svp = [cutlass.select_(pt == NS + 3, sv8[i],
                                               cutlass.select_(pt == NS + 2, sv7[i],
                                                               cutlass.select_(pt == NS + 1, sv6[i], sv5[i])))
                               for i in range(CW)]
                        if cutlass.const_expr(jj == 0):
                            self._p_last(svp, t_p, p_empty, p_full, npr - 1, npr, n, seqlen, h_begin, hw, half, lane, k,
                                         no)
                        else:
                            ps = jj % NP
                            if cutlass.const_expr(jj >= NP):
                                cute.arch.mbarrier_wait(p_empty + ps, phase=((jj // NP) - 1) & 1)
                            tc_fence_after()
                            w = [p4_from_s(svp[4 * q], svp[4 * q + 1], svp[4 * q + 2], svp[4 * q + 3], k, no)
                                 for q in range(CW // 4)]
                            tmem_st_x8(t_p + ps * (HT // 4), w)
                            tc_fence_before()
                            cute.arch.sync_warp()
                            if lane == 0:
                                cute.arch.mbarrier_arrive(p_full + ps)
                nbot = cutlass.select_(npr > NS, Int32(NS), npr - 1)
                for ue in cutlass.range_constexpr(NS):
                    if ue < nbot:
                        u = ntop + ue
                        if cutlass.const_expr(ue == 0):
                            sv = [sE[j * SM_THREADS + tidx] for j in range(CW)]
                        else:
                            sv = tmem_wait_ld_cw(tmem_ld_cw_nowait(t_s + ue * HT))
                        ps = u % NP
                        if u >= NP:
                            cute.arch.mbarrier_wait(p_empty + ps, phase=((u // NP) - 1) & 1)
                        tc_fence_after()
                        if cutlass.const_expr(ue < self.ne):
                            ce = cute.math.exp2(msl[ue] - m_scaled, fastmath=False) * Float32(448.0)
                            w = [p4_from_e(sv[4 * q], sv[4 * q + 1], sv[4 * q + 2], sv[4 * q + 3], ce)
                                 for q in range(CW // 4)]
                        else:
                            w = [p4_from_s(sv[4 * q], sv[4 * q + 1], sv[4 * q + 2], sv[4 * q + 3], k, no)
                                 for q in range(CW // 4)]
                        tmem_st_x8(t_p + ps * (HT // 4), w)
                        tc_fence_before()
                        cute.arch.sync_warp()
                        if lane == 0:
                            cute.arch.mbarrier_arrive(p_full + ps)
                if npr <= NS:
                    if npr > 0:
                        svl = tmem_wait_ld_cw(tmem_ld_cw_nowait(t_s + (npr - 1) * HT))
                        self._p_last(svl, t_p, p_empty, p_full, npr - 1, npr, n, seqlen, h_begin, hw, half, lane, k, no)
            else:
                last_first = False
                if cutlass.const_expr(self.big):
                    last_first = npr > NS
                if last_first:
                    self._p_last(sv0, t_p, p_empty, p_full, npr - 1, npr, n, seqlen, h_begin, hw, half, lane, k, no)
                # huge with a seventh pair: pair 5 (always two full tiles then) is the second pass-2 pair, P stage 1
                sh2 = False
                if cutlass.const_expr(self.huge):
                    sh2 = npr > NS + 1
                    if sh2:
                        tc_fence_after()
                        w5 = [p4_from_s(sv1[4 * q], sv1[4 * q + 1], sv1[4 * q + 2], sv1[4 * q + 3], k, no)
                              for q in range(CW // 4)]
                        tmem_st_x8(t_p + 1 * (HT // 4), w5)
                        tc_fence_before()
                        cute.arch.sync_warp()
                        if lane == 0:
                            cute.arch.mbarrier_arrive(p_full + 1)
                nxt = sv0
                if cutlass.const_expr(self.big):
                    if last_first:
                        nxt = tmem_ld_cw_nowait(t_s)
                for ue in cutlass.range_constexpr(self.npair - 1):
                    sv = tmem_wait_ld_cw(nxt)
                    # next pair's S, in flight during this one; clamped to the run's last pair, so after the loop the
                    # prefetched registers hold the last pair's S
                    pn = cutlass.select_(ue + 1 < npr - 1, Int32(ue + 1), npr - 1)
                    nxt = tmem_ld_cw_nowait(t_s + _scol(pn, SMODE))
                    lim = npr - 1
                    if cutlass.const_expr(self.huge):
                        lim = cutlass.select_(sh2, npr - 2, npr - 1)
                    if ue < lim:
                        u = Int32(ue)
                        if cutlass.const_expr(self.big):
                            u = cutlass.select_(last_first, Int32(ue + 1), Int32(ue))
                        if cutlass.const_expr(self.huge):
                            u = cutlass.select_(sh2, Int32(ue + 2), u)
                        ps = u % NP
                        if u >= NP:
                            cute.arch.mbarrier_wait(p_empty + ps, phase=((u // NP) - 1) & 1)
                        tc_fence_after()
                        if cutlass.const_expr(ue < self.ne):
                            ce = cute.math.exp2(msl[ue] - m_scaled, fastmath=False) * Float32(448.0)
                            w = [p4_from_e(sv[4 * q], sv[4 * q + 1], sv[4 * q + 2], sv[4 * q + 3], ce)
                                 for q in range(CW // 4)]
                        else:
                            w = [p4_from_s(sv[4 * q], sv[4 * q + 1], sv[4 * q + 2], sv[4 * q + 3], k, no)
                                 for q in range(CW // 4)]
                        tmem_st_x8(t_p + ps * (HT // 4), w)
                        tc_fence_before()
                        cute.arch.sync_warp()
                        if lane == 0:
                            cute.arch.mbarrier_arrive(p_full + ps)
                if npr > 0:
                    if not last_first:
                        svl = tmem_wait_ld_cw(nxt)
                        self._p_last(svl, t_p, p_empty, p_full, npr - 1, npr, n, seqlen, h_begin, hw, half, lane, k, no)

            # ---- epilogue: add the two lane halves, stage (O, m, l) rows, one bulk DSMEM copy per owner ----
            # copy parameters of this lane's owner (warp 0 lanes 1..C-1 issue the copies), computed off the
            # critical path and pinned here
            o_dst = (rank + lane) % C  # rotated so that owners receive their pieces evenly
            nr_dst = RG - o_dst * rows_per
            if nr_dst > rows_per:
                nr_dst = Int32(rows_per)
            slot_dst = rank
            if cutlass.const_expr(self.big):
                slot_dst = cutlass.select_(rank > o_dst, rank - 1, rank)
            dst_addr = pin_i32(dsmem_addr(sRecv.iterator, o_dst) + slot_dst * (rows_per * ROW_B))
            dst_mbar = pin_i32(dsmem_addr(recv_mbar, o_dst))
            if cutlass.const_expr(self.prog):
                dst_mbar = pin_i32(dsmem_addr(recv_mbar + slot_dst, o_dst))
            src_addr = pin_i32(smem_addr(tBsK.iterator) + o_dst * (rows_per * ROW_B))
            # this row's max goes to its owner while the last P.V runs, so the owner's exact split weights are
            # ready before the partial rows land (peers may be targeted after the cluster barrier)
            if cutlass.const_expr(self.prog):
                if hw == 0:
                    cute.arch.cluster_wait()
                    if half == 0:
                        if row < RG:
                            o_s = row // rows_per
                            st_off = (rank * rows_per + row - o_s * rows_per) * 4
                            if o_s == rank:
                                st_smem_f32(smem_addr(sStats.iterator) + st_off, m_scaled)
                            else:
                                st_async_f32(dsmem_addr(sStats.iterator, o_s) + st_off, m_scaled,
                                             dsmem_addr(stats_mbar, o_s))
            if n > 0:
                cute.arch.mbarrier_wait(o_full, phase=0)
            tc_fence_after()
            # both tiles of a pair for this thread's row: lanes 0-15 (first tile) and 16-31 (second tile) of the
            # sub-partition, read with 16x32bx2 so that thread half h gets columns [16h, 16h + 16) of its group
            # O columns of this thread group: 4 / NWG segments of 32 (16 per thread half)
            o_segs = []
            l_val = Float32(0)
            for sg in cutlass.range_constexpr(4 // NWG):
                ev = tmem_ld_epi(tbase + lane_off + TM_O + hw * (128 // NWG) + 32 * sg,
                                 tbase + lane_off + LANE16 + TM_O + hw * (128 // NWG) + 32 * sg,
                                 tbase + lane_off + TM_L, tbase + lane_off + LANE16 + TM_L)
                o_segs.append([ev[j] + ev[16 + j] for j in range(16)])
                if cutlass.const_expr(sg == 0):
                    l_val = ev[32] + ev[33]
            stage_base = smem_addr(tBsK.iterator)
            if row < RG:
                sb = stage_base + row * ROW_B + hw * (512 // NWG) + half * 64
                for sg in cutlass.range_constexpr(4 // NWG):
                    ov = o_segs[sg]
                    for q in cutlass.range_constexpr(4):
                        st_smem_v4f32(sb + 128 * sg + q * 16, ov[4 * q], ov[4 * q + 1], ov[4 * q + 2], ov[4 * q + 3])
                if hw == 0 and half == 0:
                    st_smem_v4f32(stage_base + row * ROW_B + 512, m_scaled, l_val, Float32(0), Float32(0))
            tc_fence_before()
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier(barrier_id=2, number_of_threads=SM_THREADS)
            if warp_idx == 0:
                if cutlass.const_expr(not self.prog):
                    cute.arch.cluster_wait()
                if lane > 0 and lane < C:
                    if nr_dst > 0:
                        bulk_s2dsmem(dst_addr, src_addr, nr_dst * ROW_B, dst_mbar)
                    bulk_commit()
                    if cutlass.const_expr(not self.prog):
                        bulk_wait_read0()
                # TMEM was last read before the barrier above; free it off the copy-issue path
                cute.arch.relinquish_tmem_alloc_permit()
                cute.arch.dealloc_tmem(tmem_ptr, TM_COLS)

            # ---- owner side: combine the C partials of my rows ----
            r0 = rank * rows_per
            nrows = RG - r0
            if nrows > rows_per:
                nrows = Int32(rows_per)
            if nrows < 0:
                nrows = Int32(0)
            if cutlass.const_expr(self.prog):
                recv_base = smem_addr(sRecv.iterator)
                own_base = stage_base + r0 * ROW_B
                st_base = smem_addr(sStats.iterator)
                out_base = gmem_addr(mO.iterator) + Int64(coord_b * ROWS + grp * RS + r0) * (D * 2)
                cute.arch.mbarrier_wait(stats_mbar, phase=0)
                # one thread per (row, 4 channels).  Exact split weights from the C row maxima (pushed early); the
                # partials are then accumulated in their arrival order (own first; sender s copies to s + 1, s + 2, ...
                # so owner o gets o - 1 first), each behind its own barrier, so only the last one's work trails the
                # last copy
                srcs = [(rank + (C - j)) % C for j in range(C)]
                slots = [x if not self.big else cutlass.select_(x > rank, x - 1, x) for x in srcs]
                # item slots rotated by one warp: warp 0 issues the copies, so it gets the last (mostly unused) slots
                # one thread per (row, 4 * QPI channels): 2 quads per item when the 4-channel items outnumber threads
                QPI = 1 if rows_per * 32 <= SM_THREADS else 2
                NIT = 32 // QPI
                for it in cutlass.range((tidx + SM_THREADS - 32) % SM_THREADS, nrows * NIT, SM_THREADS):
                    lr = it // NIT
                    ch = it % NIT
                    ms = [ld_smem_f32(st_base + (srcs[j] * rows_per + lr) * 4) for j in range(C)]
                    m_all = ms[0]
                    for j in cutlass.range_constexpr(1, C):
                        m_all = cute.arch.fmax(m_all, ms[j])
                    wts = [cute.math.exp2(ms[j] - m_all, fastmath=False) for j in range(C)]
                    acc = [Float32(0)] * (4 * QPI)
                    l_all = Float32(0)
                    for j in cutlass.range_constexpr(C):
                        if cutlass.const_expr(j == 0):
                            base = own_base + lr * ROW_B
                        else:
                            cute.arch.mbarrier_wait(recv_mbar + slots[j], phase=0)
                            base = recv_base + slots[j] * (rows_per * ROW_B) + lr * ROW_B
                        vs = [ld_smem_v4f32(base + (ch + NIT * qq) * 16) for qq in range(QPI)]
                        lc = ld_smem_f32(base + 516)
                        acc = [acc[4 * qq + e] + wts[j] * vs[qq][e] for qq in range(QPI) for e in range(4)]
                        l_all = l_all + wts[j] * lc
                    inv = scale_o / l_all
                    # a request with no keys (every split's m is -inf: NaN weights) gets zeros or, with
                    # empty_rows="skip", no store; its merge waits ran as for any request (every partial has landed)
                    for qq in cutlass.range_constexpr(QPI):
                        o_addr = out_base + Int64(lr * (D * 2) + (ch + NIT * qq) * 8)
                        if seqlen > 0:
                            st_global_f16x4(o_addr, acc[4 * qq] * inv, acc[4 * qq + 1] * inv, acc[4 * qq + 2] * inv,
                                            acc[4 * qq + 3] * inv)
                        if cutlass.const_expr(self.zero_empty):
                            if seqlen <= 0:
                                st_global_f16x4(o_addr, Float32(0), Float32(0), Float32(0), Float32(0))
            else:
                cute.arch.mbarrier_wait(recv_mbar, phase=0)
                recv_base = smem_addr(sRecv.iterator)
                own_base = stage_base + r0 * ROW_B
                out_base = gmem_addr(mO.iterator) + Int64(coord_b * ROWS + grp * RS + r0) * (D * 2)
                # one thread per (row, 4 channels): exact split weights recomputed in-thread from the C (m, l) pairs
                for it in cutlass.range(tidx, nrows * 32, SM_THREADS):
                    lr = it // 32
                    ch = it % 32
                    slots = [c if not self.big else cutlass.select_(c > rank, Int32(c - 1), Int32(c)) for c in range(C)]
                    bases = [cutlass.select_(rank == c, own_base, recv_base + slots[c] * (rows_per * ROW_B)) + lr * ROW_B
                             for c in range(C)]
                    mls = [ld_smem_v4f32(bases[c] + 512) for c in range(C)]
                    vs = [ld_smem_v4f32(bases[c] + ch * 16) for c in range(C)]
                    m_all = mls[0][0]
                    for c in cutlass.range_constexpr(1, C):
                        m_all = cute.arch.fmax(m_all, mls[c][0])
                    wts = [cute.math.exp2(mls[c][0] - m_all, fastmath=False) for c in range(C)]
                    l_all = wts[0] * mls[0][1]
                    for c in cutlass.range_constexpr(1, C):
                        l_all = l_all + wts[c] * mls[c][1]
                    inv = scale_o / l_all
                    a0 = Float32(0)
                    a1 = Float32(0)
                    a2 = Float32(0)
                    a3 = Float32(0)
                    for c in cutlass.range_constexpr(C):
                        a0 = a0 + wts[c] * vs[c][0]
                        a1 = a1 + wts[c] * vs[c][1]
                        a2 = a2 + wts[c] * vs[c][2]
                        a3 = a3 + wts[c] * vs[c][3]
                    # a request with no keys (every split's m is -inf: NaN weights) gets zeros or, with
                    # empty_rows="skip", no store
                    o_addr = out_base + Int64(lr * (D * 2) + ch * 8)
                    if seqlen > 0:
                        st_global_f16x4(o_addr, a0 * inv, a1 * inv, a2 * inv, a3 * inv)
                    if cutlass.const_expr(self.zero_empty):
                        if seqlen <= 0:
                            st_global_f16x4(o_addr, Float32(0), Float32(0), Float32(0), Float32(0))
            if cutlass.const_expr(self.prog):
                if warp_idx == 0:
                    if lane > 0 and lane < C:
                        bulk_wait_read0()
        return


class PagedDecodeQ7Online:
    """128-row variant for larger batches: one CTA per (request, KV split), K/V pages streamed through a ring.

    ``empty_rows``: what a request with no keys gets in its output rows (see EMPTY_ROWS_MODES).
    """

    def __init__(self, cluster_size: int, pos_scale: bool, empty_rows: str, kv_stages: int = 2, k_lead: int = 1):
        assert empty_rows in EMPTY_ROWS_MODES
        self.k_lead = k_lead  # K pages issued ahead of the matching V page
        # the epilogue stages the 112 partial rows over the (then idle) K and V rings
        assert 2 * kv_stages * 128 * 128 >= ROWS * ROW_B
        self.C = cluster_size
        self.pos_scale = pos_scale  # bmm1_scale >= 0: row max of s * scale = scale * max(s), else scale * min(s)
        self.zero_empty = empty_rows == "zero"
        self.kv_stages = kv_stages
        self.threads_per_cta = 384

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (112, 128, B) e4m3
        mK: cute.Tensor,  # (128 tok, 128 d, pages) e4m3, strides (token, 1, page), runtime values
        mV: cute.Tensor,  # (128 d, 128 tok, pages) e4m3, strides (1, token, page), runtime values
        mO: cute.Tensor,  # (B*112*128,) fp16
        mSeq: cute.Tensor,  # (B,) int32
        mBT: cute.Tensor,  # (B, 34) int32
        scale_log2: Float32,
        scale_o: Float32,
        stream: cuda.CUstream,
    ):
        mma_dtype = mQ.element_type
        acc_dtype = Float32
        tile = (128, 128, 128)

        tiled_mma_qk = sm100_utils.make_trivial_tiled_mma(
            mma_dtype, mma_dtype, OperandMajorMode.K, OperandMajorMode.K, acc_dtype,
            tcgen05.CtaGroup.ONE, tile[:2], tcgen05.OperandSource.SMEM,
        )
        tiled_mma_pv = sm100_utils.make_trivial_tiled_mma(
            mma_dtype, mma_dtype, OperandMajorMode.K, OperandMajorMode.MN, acc_dtype,
            tcgen05.CtaGroup.ONE, tile[:2], tcgen05.OperandSource.TMEM,
        )
        # row sums of the quantized P on the tensor core: L = P . ones  (N=16, all columns identical)
        tiled_mma_pl = sm100_utils.make_trivial_tiled_mma(
            mma_dtype, mma_dtype, OperandMajorMode.K, OperandMajorMode.K, acc_dtype,
            tcgen05.CtaGroup.ONE, (128, 16), tcgen05.OperandSource.TMEM,
        )

        smem_layout_q = sm100_utils.make_smem_layout_a(tiled_mma_qk, tile, mma_dtype, 1)
        smem_layout_k = sm100_utils.make_smem_layout_b(tiled_mma_qk, tile, mma_dtype, self.kv_stages)
        smem_layout_v = sm100_utils.make_smem_layout_b(tiled_mma_pv, tile, mma_dtype, self.kv_stages)

        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_q = cute.nvgpu.make_tiled_tma_atom_A(tma_load_op, mQ, smem_layout_q, tile, tiled_mma_qk)
        tma_k = cute.nvgpu.make_tiled_tma_atom_B(tma_load_op, mK, smem_layout_k, tile, tiled_mma_qk)
        tma_v = cute.nvgpu.make_tiled_tma_atom_B(tma_load_op, mV, smem_layout_v, tile, tiled_mma_pv)

        batches = mSeq.shape[0]
        self.kernel(
            mma_dtype, tiled_mma_qk, tiled_mma_pv, tiled_mma_pl, tma_q, tma_k, tma_v,
            mO, mSeq, mBT, scale_log2, scale_o,
        ).launch(
            grid=[self.C, batches, 1],
            block=[self.threads_per_cta, 1, 1],
            cluster=[self.C, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mma_dtype: cutlass.Constexpr,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_mma_pl: cute.TiledMma,
        tma_q: cute.nvgpu.cpasync.TmaInfo,
        tma_k: cute.nvgpu.cpasync.TmaInfo,
        tma_v: cute.nvgpu.cpasync.TmaInfo,
        mO: cute.Tensor,
        mSeq: cute.Tensor,
        mBT: cute.Tensor,
        scale_log2: Float32,
        scale_o: Float32,
    ):
        C = self.C
        acc_dtype = Float32
        mma_tile_m, mma_tile_n, mma_tile_k = 128, 128, 128
        n_threads = self.threads_per_cta
        sm_threads = 2 * warpgroup_threads  # two softmax warpgroups, one per 64-key half of each tile

        rank, coord_b, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(tidx // warp_threads)
        lane = cute.arch.lane_idx()
        tma_warp_id = 0
        mma_warp_qk_id = 1
        mma_warp_pv_id = 2

        # page ids of the request are requested first: the K/V stream waits on them
        pid0 = Int32(0)
        pid1 = Int32(0)
        if warp_idx == tma_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_q.atom)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_k.atom)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_v.atom)
            pid0 = mBT[coord_b, lane]
            if lane < MAX_PAGES - 32:
                pid1 = mBT[coord_b, lane + 32]
        seqlen = mSeq[coord_b]
        # a nonpositive length is a request with no keys: no pages in any split, no page reads
        seqlen = cutlass.select_(seqlen > 0, seqlen, Int32(0))

        thr_mma_qk = tiled_mma_qk.get_slice(0)
        thr_mma_pv = tiled_mma_pv.get_slice(0)
        thr_mma_pl = tiled_mma_pl.get_slice(0)
        mcast_coord_vmnk = (0, 0, 0, 0)
        mcast_layout_vmnk = cute.make_layout((1, 1, 1, 1))

        stensor_align = 1024
        smem = cutlass.utils.SmemAllocator()
        tmem_ptr_smem_ptr = smem.allocate_array(Int32, 1)
        tmem_alloc_cols = 512

        elect_one = CooperativeGroup(Agent.Thread)
        sm_coop = CooperativeGroup(Agent.Thread, sm_threads)

        q_smem_bytes = cute.size_in_bytes(mma_dtype, tma_q.smem_layout)
        q_mbar_ptr = smem.allocate_array(Int64, 1)

        # all pipeline mbarriers are initialised by warp 0 (the TMA warp)
        kv_stages = self.kv_stages
        k_stage_bytes = cute.size_in_bytes(mma_dtype, tma_k.smem_layout) // kv_stages
        k_pipeline_ptr = smem.allocate_array(Int64, kv_stages * 2)
        k_producer, k_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=k_pipeline_ptr, num_stages=kv_stages,
            producer_group=elect_one, consumer_group=elect_one,
            tx_count=k_stage_bytes, cta_layout_vmnk=mcast_layout_vmnk, defer_sync=True,
        ).make_participants()

        v_stage_bytes = cute.size_in_bytes(mma_dtype, tma_v.smem_layout) // kv_stages
        v_pipeline_ptr = smem.allocate_array(Int64, kv_stages * 2)
        v_producer, v_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=v_pipeline_ptr, num_stages=kv_stages,
            producer_group=elect_one, consumer_group=elect_one,
            tx_count=v_stage_bytes, cta_layout_vmnk=mcast_layout_vmnk, defer_sync=True,
        ).make_participants()

        s_stages = 2
        s_pipeline_ptr = smem.allocate_array(Int64, s_stages * 2)
        s_producer, s_consumer = pipeline.PipelineUmmaAsync.create(
            barrier_storage=s_pipeline_ptr, num_stages=s_stages,
            producer_group=elect_one, consumer_group=sm_coop, defer_sync=True,
        ).make_participants()

        p_stages = 2
        p_pipeline_ptr = smem.allocate_array(Int64, p_stages * 2)
        p_producer, p_consumer = pipeline.PipelineAsyncUmma.create(
            barrier_storage=p_pipeline_ptr, num_stages=p_stages,
            producer_group=sm_coop, consumer_group=elect_one, defer_sync=True,
        ).make_participants()

        o_pipeline_ptr = smem.allocate_array(Int64, 2)
        o_producer, o_consumer = pipeline.PipelineUmmaAsync.create(
            barrier_storage=o_pipeline_ptr, num_stages=1,
            producer_group=elect_one, consumer_group=sm_coop, defer_sync=True,
        ).make_participants()

        # split merge: every CTA pushes its partial rows to the rows' owner CTA (rows_per rows per owner)
        rows_per = (ROWS + C - 1) // C
        # Exact split coefficients are computed once per owned row, then shared
        # by all 32 output-channel quads during the DSMEM reduction.
        sW = smem.allocate_tensor(acc_dtype, cute.make_layout(C * rows_per), 16)
        recv_mbar = smem.allocate_array(Int64, C)  # one per source split: partials are merged as they land
        stats_mbar = smem.allocate_array(Int64, 1)
        # row maxima of my rows from every split: [split][row], pushed when a split's softmax loop ends
        sStats = smem.allocate_tensor(acc_dtype, cute.make_layout(C * rows_per), 16)
        # per-tile row max exchange between the two halves of a row: [parity][half][row]
        sMax = smem.allocate_tensor(acc_dtype, cute.make_layout(2 * 2 * 128), 16)

        tAsQ = smem.allocate_tensor(mma_dtype, tma_q.smem_layout.outer, stensor_align, tma_q.smem_layout.inner)
        tBsK = smem.allocate_tensor(mma_dtype, tma_k.smem_layout.outer, stensor_align, tma_k.smem_layout.inner)
        tBsV = smem.allocate_tensor(mma_dtype, tma_v.smem_layout.outer, stensor_align, tma_v.smem_layout.inner)

        # e4m3 ones (0x38) as the B operand of the row-sum MMA
        smem_layout_ones = sm100_utils.make_smem_layout_b(tiled_mma_pl, (128, 16, 128), mma_dtype, 1)
        ones_raw = smem.allocate(2048, byte_alignment=1024)
        tBsOnes = cute.make_tensor(
            cute.recast_ptr(ones_raw, smem_layout_ones.inner, dtype=mma_dtype), smem_layout_ones.outer
        )
        # fp32 partial rows received from the C splits: [C][rows_per][132]: O[0:128], (m, l, 0, 0)
        sRecvO = smem.allocate_tensor(acc_dtype, cute.make_layout(C * rows_per * ROW_F), 16)

        # ---- this CTA's page range (balanced split of the request's pages over the cluster): computed inside each
        # role, so that nothing waits on seq_lens before the TMA warp has put Q in flight.  Remainder pages go to
        # low ranks so the partial tail page belongs to a shorter final split. ----
        mma_dice = (None, None, None)
        npages = Int32(0)
        p_begin = Int32(0)
        p_end = Int32(0)
        n_tiles = Int32(0)

        if warp_idx == tma_warp_id:
            # ================= TMA: Q, then K/V pages (K two tiles ahead of V) =================
            # starts right after its own mbarrier init; no CTA / cluster barrier on this path
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(q_mbar_ptr, 1)
                if cutlass.const_expr(C > 1):
                    for i in cutlass.range_constexpr(C):
                        cute.arch.mbarrier_init(recv_mbar + i, 1)
                    cute.arch.mbarrier_init(stats_mbar, 1)
            cute.arch.mbarrier_init_fence()

            gQ = cute.local_tile(tma_q.tma_tensor, tiler=(mma_tile_m, mma_tile_k), coord=(0, 0, coord_b))
            tAgQ = thr_mma_qk.partition_A(gQ)
            tAsQ_tma, tAgQ_tma = cute.nvgpu.cpasync.tma_partition(
                tma_q.atom, mcast_coord_vmnk, mcast_layout_vmnk,
                cute.group_modes(tAsQ, 0, 3), cute.group_modes(tAgQ, 0, 3),
            )
            cute.copy(tma_q.atom, tAgQ_tma, tAsQ_tma[None, 0], tma_bar_ptr=q_mbar_ptr)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(q_mbar_ptr, q_smem_bytes)
            # rendezvous before anything waits on seq_lens: the other roles only need the barriers initialised
            if cutlass.const_expr(C > 1):
                with cute.arch.elect_one():
                    recv_rows = Int32(rows_per)
                    if cutlass.const_expr(ROWS % C != 0):
                        recv_rows = ROWS - rank * rows_per
                        if recv_rows > rows_per:
                            recv_rows = Int32(rows_per)
                    for i in cutlass.range_constexpr(C):
                        cute.arch.mbarrier_arrive_and_expect_tx(recv_mbar + i, recv_rows * ROW_B)
                    cute.arch.mbarrier_arrive_and_expect_tx(stats_mbar, (C - 1) * recv_rows * 4)
                cute.arch.cluster_arrive_relaxed()
            cute.arch.barrier_arrive(barrier_id=1, number_of_threads=n_threads)
            npages = (seqlen + (PAGE - 1)) // PAGE
            p_begin = (rank * npages + (C - 1)) // C
            p_end = ((rank + 1) * npages + (C - 1)) // C
            n_tiles = p_end - p_begin

            gK = cute.local_tile(tma_k.tma_tensor, tiler=(mma_tile_n, mma_tile_k), coord=(0, 0, None))
            tBgK = thr_mma_qk.partition_B(gK)
            tBsK_tma, tBgK_tma = cute.nvgpu.cpasync.tma_partition(
                tma_k.atom, mcast_coord_vmnk, mcast_layout_vmnk,
                cute.group_modes(tBsK, 0, 3), cute.group_modes(tBgK, 0, 3),
            )
            gV = cute.local_tile(tma_v.tma_tensor, tiler=(mma_tile_n, mma_tile_k), coord=(0, 0, None))
            tBgV = thr_mma_pv.partition_B(gV)
            tBsV_tma, tBgV_tma = cute.nvgpu.cpasync.tma_partition(
                tma_v.atom, mcast_coord_vmnk, mcast_layout_vmnk,
                cute.group_modes(tBsV, 0, 3), cute.group_modes(tBgV, 0, 3),
            )
            # Put Q and the first K page in flight before merge-receive setup;
            # the score warp will need exactly these two transfers first.
            if n_tiles > 0:
                idx = p_begin
                src = pid0
                if idx >= 32:
                    src = pid1
                page = cute.arch.shuffle_sync(src, idx % 32)
                k_handle = k_producer.acquire_and_advance()
                cute.copy(tma_k.atom, tBgK_tma[None, page], tBsK_tma[None, k_handle.index],
                          tma_bar_ptr=k_handle.barrier)

            # Keep the score stream k_lead complete pages ahead of V. Exact
            # softmax leaves enough slack before each corresponding V consume.
            LEAD = self.k_lead
            for t in cutlass.range(1, n_tiles + LEAD):
                if t < n_tiles:
                    idx = p_begin + t
                    src = pid0
                    if idx >= 32:
                        src = pid1
                    page = cute.arch.shuffle_sync(src, idx % 32)
                    k_handle = k_producer.acquire_and_advance()
                    cute.copy(tma_k.atom, tBgK_tma[None, page], tBsK_tma[None, k_handle.index],
                              tma_bar_ptr=k_handle.barrier)
                if t >= LEAD:
                    idx = p_begin + t - LEAD
                    src = pid0
                    if idx >= 32:
                        src = pid1
                    page = cute.arch.shuffle_sync(src, idx % 32)
                    v_handle = v_producer.acquire_and_advance()
                    cute.copy(tma_v.atom, tBgV_tma[None, page], tBsV_tma[None, v_handle.index],
                              tma_bar_ptr=v_handle.barrier)
        else:
            if warp_idx == mma_warp_qk_id:
                cute.arch.alloc_tmem(tmem_alloc_cols, tmem_ptr_smem_ptr)
            if warp_idx >= 4:
                ones_i32 = cute.make_tensor(cute.recast_ptr(ones_raw, dtype=Int32), cute.make_layout(512))
                ones_i32[tidx - 128] = Int32(0x38383838)
                ones_i32[tidx + 128] = Int32(0x38383838)
                cute.arch.fence_proxy("async.shared", space="cta")
            if cutlass.const_expr(C > 1):
                cute.arch.cluster_arrive_relaxed()
            cute.arch.barrier(barrier_id=1, number_of_threads=n_threads)
            npages = (seqlen + (PAGE - 1)) // PAGE
            p_begin = (rank * npages + (C - 1)) // C
            p_end = ((rank + 1) * npages + (C - 1)) // C
            n_tiles = p_end - p_begin
        tmem_ptr = cute.arch.retrieve_tmem_ptr(Int32, 16, tmem_ptr_smem_ptr)
        tmem = cutlass.utils.TmemBufferPool(tmem_ptr, tmem_alloc_cols)

        tCtS_shape = tiled_mma_qk.partition_shape_C((mma_tile_m, mma_tile_n, s_stages))
        tCtS = tiled_mma_qk.make_fragment_C(tCtS_shape)
        tCtS = tmem.allocate_tensor(tCtS.layout, acc_dtype)

        p_col = tmem.current_offset
        tAtP_shape = tiled_mma_pv.partition_shape_A((mma_tile_m, mma_tile_k, p_stages))
        tAtP = tiled_mma_pv.make_fragment_A(tAtP_shape)
        tAtP = tmem.allocate_tensor(tAtP.layout, mma_dtype)
        p_cols_per_stage = (tmem.current_offset - p_col) // p_stages  # 32: 4 e4m3 per 32-bit column

        tCtO_shape = tiled_mma_pv.partition_shape_C((mma_tile_m, mma_tile_n))
        tCtO = tiled_mma_pv.make_fragment_C(tCtO_shape)
        tCtO = tmem.allocate_tensor(tCtO.layout, acc_dtype)

        tCtL_shape = tiled_mma_pl.partition_shape_C((mma_tile_m, 16))
        tCtL = tiled_mma_pl.make_fragment_C(tCtL_shape)
        tCtL = tmem.allocate_tensor(tCtL.layout, acc_dtype)

        # ================= MMA S = Q K^T =================
        if warp_idx == mma_warp_qk_id:
            tAsQ_desc = thr_mma_qk.make_fragment_A(tAsQ)
            tBsK_desc = thr_mma_qk.make_fragment_B(tBsK)
            cute.arch.mbarrier_wait(q_mbar_ptr, phase=0)
            for t in cutlass.range(n_tiles):
                tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, False)
                k_handle = k_consumer.wait_and_advance()
                s_handle = s_producer.acquire_and_advance()
                tc_fence_after()
                tAsQ_s = tAsQ_desc[mma_dice + (0,)]
                tBsK_s = tBsK_desc[mma_dice + (k_handle.index,)]
                tCtS_s = tCtS[mma_dice + (s_handle.index,)]
                num_mma_k = cute.size(tBsK.shape[2])
                for mma_k in cutlass.range_constexpr(num_mma_k):
                    cute.gemm(tiled_mma_qk, tCtS_s, tAsQ_s[None, None, mma_k], tBsK_s[None, None, mma_k], tCtS_s)
                    if mma_k == 0:
                        tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, True)
                k_handle.release()
                s_handle.commit()

        # ================= MMA O += P V,  L += P . ones =================
        elif warp_idx == mma_warp_pv_id:
            tBsV_desc = thr_mma_pv.make_fragment_B(tBsV)
            tBsOnes_desc = thr_mma_pl.make_fragment_B(tBsOnes)
            tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, False)
            tiled_mma_pl.set(tcgen05.Field.ACCUMULATE, False)
            for t in cutlass.range(n_tiles):
                v_handle = v_consumer.wait_and_advance()
                p_handle = p_consumer.wait_and_advance()
                o_handle = o_producer.acquire_and_advance()
                tc_fence_after()
                tAtP_s = tAtP[mma_dice + (p_handle.index,)]
                tBsV_s = tBsV_desc[mma_dice + (v_handle.index,)]
                num_mma_k = cute.size(tBsV.shape[2])
                for mma_k in cutlass.range_constexpr(num_mma_k):
                    cute.gemm(tiled_mma_pv, tCtO, tAtP_s[None, None, mma_k], tBsV_s[None, None, mma_k], tCtO)
                    cute.gemm(tiled_mma_pl, tCtL, tAtP_s[None, None, mma_k], tBsOnes_desc[None, None, mma_k, 0], tCtL)
                    if mma_k == 0:
                        tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, True)
                        tiled_mma_pl.set(tcgen05.Field.ACCUMULATE, True)
                v_handle.release()
                p_handle.release()
                o_handle.commit()
            if n_tiles > 0:
                o_producer.tail()
            tc_fence_after()
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

        # ================= softmax / correction / epilogue (2 warpgroups: key halves) =================
        elif warp_idx >= 4:
            row = tidx % warpgroup_threads  # TMEM lane = query row
            half = (tidx - warpgroup_threads) // warpgroup_threads  # 0: keys 0..63, 1: keys 64..127
            sm_tid = tidx - warpgroup_threads

            tS0 = tCtS[mma_dice + (0,)]
            lay64 = cute.composition(tS0.layout, cute.make_layout((128, 64)))
            lay16 = cute.composition(tS0.layout, cute.make_layout((128, 16)))
            vS = cute.make_tensor(tS0.iterator, lay64)
            ld64 = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(64)), acc_dtype)
            st64 = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(64)), acc_dtype)
            st16 = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(16)), acc_dtype)
            # SM103 fused TMEM load + row reduction: the row max (or min, for a negative scale) of each 64-key half
            red_op = tcgen05.TmemLoadRedOp.MAX if self.pos_scale else tcgen05.TmemLoadRedOp.MIN
            ldr64 = cute.make_copy_atom(tcgen05.copy.LdRed32x32bOp(tcgen05.Repetition(64), redOp=red_op), acc_dtype)
            thr_ldr = tcgen05.make_tmem_copy(ldr64, vS).get_slice(row)
            thr_ld = tcgen05.make_tmem_copy(ld64, vS).get_slice(row)
            thr_st = tcgen05.make_tmem_copy(st64, vS).get_slice(row)
            rS_shape = thr_ld.partition_D(vS).shape
            red_layout = cute.make_layout((1,) + tuple(rS_shape)[1:])
            p_base = cute.recast_ptr(tmem_ptr + p_col, dtype=acc_dtype)
            vP = cute.make_tensor(p_base, lay16)
            thr_stp = tcgen05.make_tmem_copy(st16, vP).get_slice(row)
            rP_shape = thr_stp.partition_S(vP).shape
            cS = cute.make_identity_tensor((128, 64))
            tScS = thr_ld.partition_D(cS)

            vO = cute.make_tensor(tCtO.iterator + half * 64, cute.composition(tCtO.layout, cute.make_layout((128, 64))))
            tOtO = thr_ld.partition_S(vO)
            tOtO_st = thr_st.partition_D(vO)

            tmem_l_rep = tcgen05.Repetition(16)
            tmem_load_atom_l = cute.make_copy_atom(tcgen05.Ld32x32bOp(tmem_l_rep), acc_dtype)
            tmem_store_atom_l = cute.make_copy_atom(tcgen05.St32x32bOp(tmem_l_rep), acc_dtype)
            thr_tmem_load_l = tcgen05.make_tmem_copy(tmem_load_atom_l, tCtL).get_slice(row)
            tLtL = thr_tmem_load_l.partition_S(tCtL)
            tLrL_shape = thr_tmem_load_l.partition_D(tCtL).shape

            smax_base = smem_addr(sMax.iterator)
            prev_max = -Float32.inf
            row_sum = Float32(0)  # sum of the e4m3-quantized P (x448), from the tensor-core row-sum MMA
            for t in cutlass.range(n_tiles):
                s_handle = s_consumer.wait_and_advance()
                tc_fence_after()
                s_base = tCtS[mma_dice + (s_handle.index,)].iterator
                tSrS = cute.make_rmem_tensor(rS_shape, acc_dtype)
                red_own = cute.make_rmem_tensor(red_layout, acc_dtype)
                cute.copy(ldr64, thr_ldr.partition_S(cute.make_tensor(s_base + half * 64, lay64)), (tSrS, red_own))
                # the request's last page has keys past seq_len: masked, row max exchanged through smem
                masked = (p_begin + t + 1) * PAGE > seqlen
                red_oth = cute.make_rmem_tensor(red_layout, acc_dtype)
                if not masked:
                    # other half's row max straight from TMEM (values discarded): no cross-warpgroup barrier
                    junk = cute.make_rmem_tensor(rS_shape, acc_dtype)
                    cute.copy(ldr64, thr_ldr.partition_S(cute.make_tensor(s_base + (1 - half) * 64, lay64)),
                              (junk, red_oth))
                cute.arch.fence_view_async_tmem_load()
                tc_fence_before()
                s_handle.release()

                a_mul = scale_log2  # P = exp2(s * a_mul + (log2(448) - row_max))
                tile_max = Float32(0)
                if masked:
                    tSrS.store(tSrS.load() * scale_log2)
                    valid = seqlen - (p_begin + t) * PAGE - half * 64
                    for j in cutlass.range_constexpr(cute.size(tSrS)):
                        tSrS[j] = Float32(cutlass.select_(tScS[j][1] < valid, tSrS[j], -Float32.inf))
                    local_max = tree_max([tSrS[j] for j in range(cute.size(tSrS))])
                    slot = smax_base + row * 4
                    st_smem_f32(slot + half * 512, local_max)
                    cute.arch.barrier(barrier_id=2, number_of_threads=sm_threads)
                    other_max = ld_smem_f32(slot + (1 - half) * 512)
                    tile_max = cute.arch.fmax(local_max, other_max)
                    a_mul = Float32(1.0)
                else:
                    if cutlass.const_expr(self.pos_scale):
                        tile_max = cute.arch.fmax(red_own[0], red_oth[0]) * scale_log2
                    else:
                        tile_max = cute.arch.fmin(red_own[0], red_oth[0]) * scale_log2
                row_max = cute.arch.fmax(prev_max, tile_max)
                nb = LOG2_P_SCALE - row_max
                # packed f32x2 FMAs, MUFU.EX2 (the same as the exact path; see ex2_p for why the e4m3 P is
                # bit-identical) and paired e4m3 conversion, 4 keys per 32-bit word
                tPrP8 = cute.make_rmem_tensor(rS_shape, mma_dtype)
                tPrPw = cute.make_tensor(cute.recast_ptr(tPrP8.iterator, dtype=Int32),
                                         cute.make_layout(cute.size(tSrS) // 4))
                for i in cutlass.range_constexpr(cute.size(tSrS) // 4):
                    tPrPw[i] = p4_from_s(tSrS[4 * i], tSrS[4 * i + 1], tSrS[4 * i + 2], tSrS[4 * i + 3], a_mul, nb)
                tPrP = cute.make_tensor(cute.recast_ptr(tPrP8.iterator, dtype=acc_dtype), cute.make_layout(rP_shape))

                p_handle = p_producer.acquire_and_advance()
                tc_fence_after()
                vP_t = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + (p_col + p_handle.index * p_cols_per_stage + half * 16), dtype=acc_dtype),
                    lay16,
                )
                cute.copy(st16, tPrP, thr_stp.partition_D(vP_t))
                cute.arch.fence_view_async_tmem_store()
                tc_fence_before()
                p_handle.commit()

                if t > 0:
                    correction = Float32(1.0)
                    if row_max != prev_max:
                        correction = cute.math.exp2(prev_max - row_max, fastmath=False)
                    o_handle = o_consumer.wait_and_advance()
                    tc_fence_after()
                    if not cute.arch.vote_all_sync(correction == Float32(1.0)):
                        tOrO = cute.make_rmem_tensor(rS_shape, acc_dtype)
                        cute.copy(ld64, tOtO, tOrO)
                        tOrO.store(correction * tOrO.load())
                        cute.copy(st64, tOrO, tOtO_st)
                        if half == 0:
                            tLrL = cute.make_rmem_tensor(tLrL_shape, acc_dtype)
                            cute.copy(tmem_load_atom_l, tLtL, tLrL)
                            tLrL.store(correction * tLrL.load())
                            cute.copy(tmem_store_atom_l, tLrL, tLtL)
                        cute.arch.fence_view_async_tmem_store()
                    tc_fence_before()
                    o_handle.release()
                prev_max = row_max

            if cutlass.const_expr(C > 1):
                # this row's max goes to its owner while the last P.V runs, so the owner's exact split weights are
                # ready before the partial rows land (peers may be targeted after the cluster barrier)
                cute.arch.cluster_wait()
                if half == 0:
                    if row < ROWS:
                        o_s = row // rows_per
                        st_off = (rank * rows_per + row - o_s * rows_per) * 4
                        if o_s == rank:
                            st_smem_f32(smem_addr(sStats.iterator) + st_off, prev_max)
                        else:
                            st_async_f32(dsmem_addr(sStats.iterator, o_s) + st_off, prev_max,
                                         dsmem_addr(stats_mbar, o_s))
            tOrO_acc = cute.make_rmem_tensor(rS_shape, acc_dtype)
            if n_tiles > 0:
                o_handle = o_consumer.wait_and_advance()
                tc_fence_after()
                cute.copy(ld64, tOtO, tOrO_acc)
                tLrL_f = cute.make_rmem_tensor(tLrL_shape, acc_dtype)
                cute.copy(tmem_load_atom_l, tLtL, tLrL_f)
                cute.arch.fence_view_async_tmem_load()
                tc_fence_before()
                o_handle.release()
                row_sum = tLrL_f[0]
            else:
                tOrO_acc.fill(0.0)
            if cutlass.const_expr(C == 1):
                if row < ROWS:
                    # a request with no keys (row_sum 0): zeros, or with empty_rows="skip" its rows are left untouched
                    if seqlen > 0:
                        inv = scale_o / row_sum
                        tOrO_out = cute.make_rmem_tensor(rS_shape, mO.element_type)
                        tOrO_out.store((tOrO_acc.load() * inv).to(mO.element_type))
                        gO_row = cute.local_tile(mO, (64,), ((coord_b * ROWS + row) * 2 + half,))
                        cute.autovec_copy(tOrO_out, cute.make_tensor(gO_row.iterator, cute.make_layout(rS_shape)))
                    if cutlass.const_expr(self.zero_empty):
                        if seqlen <= 0:
                            tOrO_zero = cute.make_rmem_tensor(rS_shape, mO.element_type)
                            tOrO_zero.fill(0.0)
                            gO_row_z = cute.local_tile(mO, (64,), ((coord_b * ROWS + row) * 2 + half,))
                            cute.autovec_copy(tOrO_zero,
                                              cute.make_tensor(gO_row_z.iterator, cute.make_layout(rS_shape)))
            else:
                if cutlass.const_expr(C <= 5):
                    # few, large copies: the first one is worth starting before all rows are staged
                    # Stage the partial row half (fp32, unnormalised) + (m, l) in the idle K/V buffers, then one bulk
                    # copy per owner CTA.  The rows of the first copy's owner (rank + 1) are staged first, so that copy
                    # (the copy engine is the merge's bottleneck) starts while the other rows are still being staged.
                    first_o = (rank + 1) % C
                    in_first = (row < ROWS) and (row // rows_per == first_o)
                    local_stage = smem_addr(tBsK.iterator) + row * ROW_B + half * 256
                    if in_first:
                        for j in cutlass.range_constexpr(16):
                            st_smem_v4f32(local_stage + j * 16, tOrO_acc[4 * j], tOrO_acc[4 * j + 1],
                                          tOrO_acc[4 * j + 2], tOrO_acc[4 * j + 3])
                        if half == 0:
                            st_smem_v4f32(local_stage + 512, prev_max, row_sum, Float32(0), Float32(0))
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.barrier(barrier_id=3, number_of_threads=sm_threads)
                    for ph in cutlass.range_constexpr(2):
                        if cutlass.const_expr(ph == 1):
                            if row < ROWS and not in_first:
                                recv_stage = smem_addr(sRecvO.iterator) + row * ROW_B + half * 256
                                r0_stage = rank * rows_per
                                owned = (row >= r0_stage) and (row < r0_stage + rows_per)
                                stage_base = Int32(cutlass.select_(owned, recv_stage, local_stage))
                                for j in cutlass.range_constexpr(16):
                                    st_smem_v4f32(stage_base + j * 16, tOrO_acc[4 * j], tOrO_acc[4 * j + 1],
                                                  tOrO_acc[4 * j + 2], tOrO_acc[4 * j + 3])
                                if half == 0:
                                    st_smem_v4f32(stage_base + 512, prev_max, row_sum, Float32(0), Float32(0))
                            cute.arch.fence_proxy("async.shared", space="cta")
                            cute.arch.barrier(barrier_id=3, number_of_threads=sm_threads)
                        cp_ok = (sm_tid == 1) if ph == 0 else (sm_tid > 1 and sm_tid < C)
                        if cp_ok:
                            o = (rank + sm_tid) % C  # rotated so that owners receive their pieces evenly
                            copy_rows = Int32(rows_per)
                            if cutlass.const_expr(ROWS % C != 0):
                                copy_rows = ROWS - o * rows_per
                                if copy_rows > rows_per:
                                    copy_rows = Int32(rows_per)
                            bulk_s2dsmem(dsmem_addr(sRecvO.iterator, o) + rank * (rows_per * ROW_B),
                                         smem_addr(tBsK.iterator) + o * (rows_per * ROW_B), copy_rows * ROW_B,
                                         dsmem_addr(recv_mbar + rank, o))
                            bulk_commit()
                else:
                    # Stage the partial row half (fp32, unnormalised) + (m, l) in the idle K/V buffers,
                    # then one bulk copy per owner CTA
                    if row < ROWS:
                        local_stage = smem_addr(tBsK.iterator) + row * ROW_B + half * 256
                        recv_stage = smem_addr(sRecvO.iterator) + row * ROW_B + half * 256
                        r0_stage = rank * rows_per
                        owned = (row >= r0_stage) and (row < r0_stage + rows_per)
                        stage_base = Int32(cutlass.select_(owned, recv_stage, local_stage))
                        for j in cutlass.range_constexpr(16):
                            st_smem_v4f32(stage_base + j * 16, tOrO_acc[4 * j], tOrO_acc[4 * j + 1],
                                          tOrO_acc[4 * j + 2], tOrO_acc[4 * j + 3])
                        if half == 0:
                            st_smem_v4f32(stage_base + 512, prev_max, row_sum, Float32(0), Float32(0))
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.barrier(barrier_id=3, number_of_threads=sm_threads)
                    if sm_tid > 0 and sm_tid < C:
                        o = (rank + sm_tid) % C  # rotated so that owners receive their pieces evenly
                        copy_rows = Int32(rows_per)
                        if cutlass.const_expr(ROWS % C != 0):
                            copy_rows = ROWS - o * rows_per
                            if copy_rows > rows_per:
                                copy_rows = Int32(rows_per)
                        bulk_s2dsmem(dsmem_addr(sRecvO.iterator, o) + rank * (rows_per * ROW_B),
                                     smem_addr(tBsK.iterator) + o * (rows_per * ROW_B), copy_rows * ROW_B,
                                     dsmem_addr(recv_mbar + rank, o))
                        bulk_commit()

        # ================= cluster merge through DSMEM =================
        if cutlass.const_expr(C > 1):
            cute.arch.mbarrier_wait(stats_mbar, phase=0)
            cute.arch.sync_threads()  # own staged rows / row maxima (written by the softmax warps) are read below
            r0 = rank * rows_per
            nrows = ROWS - r0
            if nrows > rows_per:
                nrows = Int32(rows_per)
            recv_base = smem_addr(sRecvO.iterator)
            st_base = smem_addr(sStats.iterator)
            if tidx < nrows:
                ms = [ld_smem_f32(st_base + (c * rows_per + tidx) * 4) for c in range(C)]
                m_all = ms[0]
                for c in cutlass.range_constexpr(1, C):
                    m_all = cute.arch.fmax(m_all, ms[c])
                for c in cutlass.range_constexpr(C):
                    sW[c * rows_per + tidx] = cute.math.exp2(ms[c] - m_all, fastmath=False)
            cute.arch.sync_threads()
            out_base = gmem_addr(mO.iterator) + Int64(coord_b * ROWS + r0) * (D * 2)
            # one thread per (row, CH channels) with CH chosen so every thread has at most one item; a thread's 4-channel
            # groups are ch, ch + NIR, ... so adjacent threads read adjacent 16 bytes.  The partials are accumulated
            # in their arrival order (own first; split s copies to s + 1, s + 2, ... so owner o gets o - 1 first),
            # each behind its own barrier, so only the last one's work trails the last copy.
            CH = 4 if rows_per * (D // 4) <= n_threads else (8 if rows_per * (D // 8) <= n_threads else 16)
            NIR = D // CH
            srcs = [(rank + (C - j)) % C for j in range(C)]
            for it in cutlass.range(tidx, nrows * NIR, n_threads):
                lr = it // NIR
                ch = it % NIR
                acc = [Float32(0)] * CH
                l_all = Float32(0)
                for j in cutlass.range_constexpr(C):
                    if cutlass.const_expr(j > 0):
                        cute.arch.mbarrier_wait(recv_mbar + srcs[j], phase=0)
                    base = recv_base + (srcs[j] * rows_per + lr) * ROW_B
                    wc = sW[srcs[j] * rows_per + lr]
                    vs = [ld_smem_v4f32(base + (ch + NIR * q) * 16) for q in range(CH // 4)]
                    lc = ld_smem_f32(base + 516)
                    acc = [acc[4 * q + e] + wc * vs[q][e] for q in range(CH // 4) for e in range(4)]
                    l_all = l_all + wc * lc
                inv = scale_o / l_all
                # a request with no keys (every split's m is -inf: NaN weights) gets zeros or, with
                # empty_rows="skip", no store; its merge waits ran as for any request (every partial has landed)
                for q in cutlass.range_constexpr(CH // 4):
                    o_addr = out_base + Int64(lr * (D * 2) + (ch + NIR * q) * 8)
                    if seqlen > 0:
                        st_global_f16x4(o_addr, acc[4 * q] * inv, acc[4 * q + 1] * inv, acc[4 * q + 2] * inv,
                                        acc[4 * q + 3] * inv)
                    if cutlass.const_expr(self.zero_empty):
                        if seqlen <= 0:
                            st_global_f16x4(o_addr, Float32(0), Float32(0), Float32(0), Float32(0))
            if tidx > warpgroup_threads and tidx < warpgroup_threads + C:
                bulk_wait_read0()  # the copy-issuing softmax threads: their staged rows were read out
        return


# ---------------------------------------------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------------------------------------------
# Batch sizes up to DUAL_MAX_B and those in DUAL_EXTRA run PagedDecodeQ7Dual, the others PagedDecodeQ7Online.
DUAL_MAX_B = 11
# Batch sizes above DUAL_MAX_B served by PagedDecodeQ7Dual, with their cluster size: B12-B14 as 2 x 5 splits of up to
# 14 tiles, B16-B18 as 2 x 4 splits of up to 17 tiles.
DUAL_EXTRA = {12: 5, 13: 5, 14: 5, 16: 4, 17: 4, 18: 4}
# Early pairs of PagedDecodeQ7Dual per batch size (none for the others).
DUAL_NE_B = {7: 1, 8: 2, 10: 2, 12: 2, 13: 2, 14: 2, 16: 2, 17: 2, 18: 2}
# Max co-resident clusters of PagedDecodeQ7Online (1 CTA/SM) per cluster size on a 152-SM part.
_MAX_CLUSTERS_152 = {1: 152, 2: 76, 3: 46, 4: 36, 5: 28, 6: 23, 7: 15, 8: 15, 9: 15, 10: 13, 11: 8, 12: 8,
                     13: 7, 14: 7, 15: 7, 16: 7}
# Split counts tried for PagedDecodeQ7Online, best first.
_CLUSTER_PREF = (16, 12, 10, 9, 8, 6, 5, 4, 3, 2)


class KernelConfig(NamedTuple):
    """A compiled kernel configuration (see kernel_config)."""

    variant: str  # "dual" (PagedDecodeQ7Dual) or "online" (PagedDecodeQ7Online)
    cluster_size: int  # KV splits per request (row group): the CTAs of one cluster
    # "dual" only: query-row groups per request, progressive split merge, early pairs
    groups: int | None = None
    prog: bool | None = None
    ne: int | None = None

    @property
    def name(self) -> str:
        if self.variant == "dual":
            return f"dual-c{self.cluster_size}-g{self.groups}" + ("-prog" if self.prog else "") + f"-ne{self.ne}"
        return f"online-c{self.cluster_size}"


@functools.cache
def _num_sms(device_index: int) -> int:
    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def _dual_row_groups(batch_size: int) -> int:
    # more, smaller row groups where SMs are idle (smaller DSMEM partials): B1 8 x 14 rows, B2 4 x 28 rows
    if batch_size == 1:
        return 8
    if batch_size == 2:
        return 4
    return 2


def _dual_cluster_size(batch_size: int) -> int:
    if batch_size in DUAL_EXTRA:
        return DUAL_EXTRA[batch_size]
    if batch_size <= 2:
        return 12  # B1: 8 row groups x C12, B2: 4 x C12 (see _dual_row_groups)
    # 2 row groups x C splits per request, one wave of co-resident clusters
    if batch_size <= 3:
        return 14
    if batch_size == 4:
        return 12
    if batch_size <= 6:
        return 10
    if batch_size <= 7:
        return 9
    return 6  # B8-11: 12 tiles per CTA, the sixth S pair in the P / l columns


def _online_cluster_size(batch_size: int, device_index: int) -> int:
    sms = _num_sms(device_index)
    for c in _CLUSTER_PREF:
        cap = _MAX_CLUSTERS_152[c] if sms == 152 else (sms // 20) * (19 // c)
        if batch_size <= cap:
            return c
    return 1


def kernel_config(batch_size: int, device_index: int) -> KernelConfig:
    """The kernel configuration launched for ``batch_size`` requests on ``device_index``."""
    if batch_size <= DUAL_MAX_B or batch_size in DUAL_EXTRA:
        # progressive split merge from B4 on; B1-B3 merge their small partials behind one barrier
        return KernelConfig("dual", _dual_cluster_size(batch_size), groups=_dual_row_groups(batch_size),
                            prog=batch_size >= 4, ne=DUAL_NE_B.get(batch_size, 0))
    return KernelConfig("online", _online_cluster_size(batch_size, device_index))


def _views(q, k_cache, v_cache, seq_lens, block_tables, out):
    """DLPack views in the kernel's operand order.

    q [B, 7, 16, 128] compact -> (112, 128, B); K/V [pages, 1, 128, 128] with unit inner stride ->
    K (token, d, page) and V (d, token, page); out [B, 7, 16, 128] compact -> flat.
    """
    b = q.shape[0]
    q3 = q.view(b, ROWS, D).permute(1, 2, 0)
    k3 = k_cache[:, 0].permute(1, 2, 0)
    v3 = v_cache[:, 0].permute(2, 1, 0)
    o2 = out.view(-1)
    tq = from_dlpack(q3, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    tk = from_dlpack(k3, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    tv = from_dlpack(v3, assumed_align=16).mark_layout_dynamic(leading_dim=0)
    to = from_dlpack(o2, assumed_align=16).mark_layout_dynamic()
    ts = from_dlpack(seq_lens, assumed_align=4).mark_layout_dynamic()
    tb = from_dlpack(block_tables, assumed_align=4).mark_layout_dynamic(leading_dim=1)
    return tq, tk, tv, to, ts, tb


@functools.cache
def _compile(device_index: int, config: KernelConfig, empty_rows: str):
    with torch.cuda.device(device_index):
        dev = torch.device("cuda", device_index)
        q = torch.zeros((1, SEQ_Q, HEADS, D), dtype=torch.float8_e4m3fn, device=dev)
        kv = torch.zeros((2, 1, PAGE, 2, D), dtype=torch.float8_e4m3fn, device=dev)
        seq_lens = torch.ones((1,), dtype=torch.int32, device=dev)
        block_tables = torch.zeros((1, MAX_PAGES), dtype=torch.int32, device=dev)
        out = torch.empty((1, SEQ_Q, HEADS, D), dtype=torch.float16, device=dev)
        args = _views(q, kv[:, :, :, 0, :], kv[:, :, :, 1, :], seq_lens, block_tables, out)
        stream = cuda.CUstream(torch.cuda.current_stream(dev).cuda_stream)
        if config.variant == "dual":
            kernel = PagedDecodeQ7Dual(config.cluster_size, pos_scale=True, empty_rows=empty_rows,
                                       groups=config.groups, prog=config.prog, ne=config.ne)
        else:
            kernel = PagedDecodeQ7Online(config.cluster_size, pos_scale=True, empty_rows=empty_rows)
        return cute.compile(kernel, *args, Float32(LOG2_E), Float32(1.0), stream)


@functools.cache
def get_compiled(device_index: int, batch_size: int, empty_rows: str):
    """Compile the configuration :func:`kernel_config` picks for ``batch_size`` on ``device_index``.

    Positive ``bmm1_scale`` only (the row max is scale * max(s)); ``empty_rows`` is "zero" or "skip" (see
    EMPTY_ROWS_MODES). Compiled once per (device, configuration, empty_rows) on example tensors in the served
    layout; shapes and strides are dynamic, the examples only fix dtypes, alignment and unit strides.
    Call outside CUDA-graph capture (plan time).
    """
    return _compile(device_index, kernel_config(batch_size, device_index), empty_rows)


def launch(compiled, q, k_cache, v_cache, seq_lens, block_tables, out, bmm1_scale: float, bmm2_scale: float):
    """Launch on the current stream (capture-safe: no compilation, allocation or host sync).

    ``bmm1_scale`` must be positive (the compiled row-max form).
    """
    if not bmm1_scale > 0.0:
        raise ValueError("the specialized decode kernel is compiled for a positive bmm1_scale")
    args = _views(q, k_cache, v_cache, seq_lens, block_tables, out)
    stream = cuda.CUstream(torch.cuda.current_stream(q.device).cuda_stream)
    compiled(*args, Float32(float(bmm1_scale) * LOG2_E), Float32(float(bmm2_scale)), stream)
    return out
