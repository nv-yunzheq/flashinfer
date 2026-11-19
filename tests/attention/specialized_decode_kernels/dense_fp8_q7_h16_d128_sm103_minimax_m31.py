"""Test spec of the specialized decode kernel ``dense_fp8_q7_h16_d128_sm103_minimax_m31``:
7 dense FP8 query tokens, 16 query heads over 1 K/V head, head_dim 128, page 128, SM103.

Builders of the served inputs (shuffled pages over a pool with loud stale slots and pages,
one interleaved ``[pages, 1, 128, 2, 128]`` K/V buffer as vLLM allocates it, or separate
pools), the exact fp32 reference of the dense 7-token block, the FP8-attention tolerances,
the case grids and the kernel-internal checks (page-split view of the stock path, length
edges and scale folding, run-time refusals).
"""

from __future__ import annotations

import importlib.util
import math
from typing import Any, Dict, Optional

import pytest
import torch

from flashinfer.attention.specialized_decode import kernel as framework
from flashinfer.attention.specialized_decode.kernel import DecodePlanCall
from flashinfer.attention.specialized_decode.kernels.dense_fp8_q7_h16_d128_sm103_minimax_m31 import (
    KERNEL,
    kernel as kmod,
)
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer.utils import is_sm100a_supported

from . import Check, KernelSpec

FP8 = torch.float8_e4m3fn
HEADS, D, PAGE, Q, TABLE = 16, 128, 128, 7, 34
MAX_KV = 4232  # the served maximum
SCALE = 1.0 / math.sqrt(D)
STOCK_TOL = dict(
    row_tol=0.15, min_cos=0.998
)  # the stock cute-dsl decode's own FP8-P error
AUTO_MAIN_MSG = "only supports window_right != 0"  # how auto refuses a dense plan


def _cute_dsl() -> bool:
    return importlib.util.find_spec("cutlass") is not None


def _sm103() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (10, 3)
        and is_sm100a_supported(torch.device("cuda"))
        and _cute_dsl()
    )


# ----------------------------------------------------------------------------- inputs


def case(kv_lens, *, seed, interleaved=True, table_width=TABLE):
    """Served-layout inputs: shuffled pages over a pool with loud stale slots and pages."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    batch = len(kv_lens)
    need = sum((n + PAGE - 1) // PAGE for n in kv_lens)
    pages = 1 + need + max(4, need // 4)

    def e4m3(shape, scale):
        x = torch.randn(shape, generator=gen, device="cuda") * scale
        return x.clamp(-448, 448).to(FP8)

    q = e4m3((batch * Q, HEADS, D), 2.0)
    if interleaved:  # one [pages, 1, 128, 2, 128] buffer, as vLLM allocates it
        kv = e4m3((pages, 1, PAGE, 2, D), 1.0)
        k_cache, v_cache = kv[:, :, :, 0, :], kv[:, :, :, 1, :]
    else:
        k_cache = e4m3((pages, 1, PAGE, D), 1.0)
        v_cache = e4m3((pages, 1, PAGE, D), 1.0)
    order = torch.randperm(pages - 1, generator=gen, device="cuda").to(torch.int32) + 1
    block_tables = torch.zeros((batch, table_width), dtype=torch.int32, device="cuda")
    pos = 0
    for b, n in enumerate(kv_lens):
        used = (n + PAGE - 1) // PAGE
        block_tables[b, :used] = order[pos : pos + used]
        pos += used
        if used < table_width:  # stale (valid) ids past the request
            block_tables[b, used:] = order[
                (pos + torch.arange(table_width - used, device="cuda")) % (pages - 1)
            ]
        if n % PAGE:  # loud stale slots after the request's last key
            last = int(block_tables[b, used - 1])
            k_cache[last, :, n % PAGE :] = torch.tensor(448.0).to(FP8)
            v_cache[last, :, n % PAGE :] = torch.tensor(-448.0).to(FP8)
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    return q, k_cache, v_cache, seq_lens, block_tables


def csr(seq_lens, block_tables):
    """The wrapper's CSR page metadata for the same requests."""
    used = [(int(n) + PAGE - 1) // PAGE for n in seq_lens.tolist()]
    indptr = torch.tensor([0] + used, dtype=torch.int32).cumsum(0).to(torch.int32)
    indices = torch.cat([block_tables[b, :u] for b, u in enumerate(used)])
    last = torch.tensor(
        [int(n) - (u - 1) * PAGE for n, u in zip(seq_lens.tolist(), used, strict=True)],
        dtype=torch.int32,
    )
    return indptr.cuda(), indices.contiguous(), last.cuda()


def reference(q, k_cache, v_cache, seq_lens, block_tables, sm_scale=SCALE, v_scale=1.0):
    """Exact fp32 dense attention over each request's 7-token block."""
    batch = seq_lens.shape[0]
    q4 = q.view(batch, Q, HEADS, D)
    out = torch.zeros(q4.shape, dtype=torch.float32, device=q.device)
    for b in range(batch):
        n = int(seq_lens[b])
        used = (n + PAGE - 1) // PAGE
        ids = block_tables[b, :used].long()
        k = k_cache[ids, 0].reshape(-1, D)[:n].float()
        v = v_cache[ids, 0].reshape(-1, D)[:n].float()
        s = torch.einsum("thd,nd->thn", q4[b].float(), k) * sm_scale
        out[b] = torch.einsum("thn,nd->thd", torch.softmax(s, dim=-1), v) * v_scale
    return out.view(q.shape)


def row_error(out, ref):
    """Worst per-row (token, head) max error relative to the row's max."""
    diff = (out.float() - ref).abs().amax(dim=-1)
    return float((diff / ref.abs().amax(dim=-1).clamp_min(1e-2)).max())


def assert_close(out, ref, row_tol=0.08, min_cos=0.999):
    """FP8-attention class (P quantized to e4m3): per-row error and cosine."""
    assert torch.isfinite(out).all()
    assert row_error(out, ref) <= row_tol
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.flatten(), dim=0
    )
    assert float(cos) >= min_cos


def plan(wrapper, case_, *, table=True, **overrides):
    _, _, _, seq_lens, block_tables = case_
    indptr, indices, last = csr(seq_lens, block_tables)
    kwargs = dict(
        q_data_type=FP8,
        kv_data_type=FP8,
        o_data_type=torch.float16,
        sm_scale=SCALE,
        q_len_per_req=Q,
        window_right=-1,
        seq_lens=seq_lens,
        block_tables=block_tables if table else None,
    )
    kwargs.update(overrides)
    wrapper.plan(indptr, indices, last, HEADS, 1, D, PAGE, **kwargs)


def wrapper(
    case_,
    *,
    graph=False,
    table=True,
    plan_kwargs=None,
    backend="specialized",
    use_tensor_cores=False,
):
    _, _, _, seq_lens, block_tables = case_
    batch = seq_lens.shape[0]
    indptr, _, last = csr(seq_lens, block_tables)
    workspace = torch.zeros(128 << 20, dtype=torch.uint8, device="cuda")
    extra = (
        dict(
            paged_kv_indptr_buffer=torch.empty_like(indptr),
            paged_kv_indices_buffer=torch.empty(
                batch * TABLE, dtype=torch.int32, device="cuda"
            ),
            paged_kv_last_page_len_buffer=torch.empty_like(last),
        )
        if graph
        else {}
    )
    w = BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        "HND",
        use_cuda_graph=graph,
        use_tensor_cores=use_tensor_cores,
        backend=backend,
        **extra,
    )
    plan(w, case_, table=table, **(plan_kwargs or {}))
    return w


def run(w, case_, out=None, **kwargs):
    q, k_cache, v_cache, _, _ = case_
    return w.run(q, (k_cache, v_cache), out=out, **kwargs)


# ----------------------------------------------------------------- auto-route builders


def _plan_run(kv_lens, *, table_width=TABLE, **plan_kwargs):
    def go():
        c = case(kv_lens, seed=61, table_width=table_width)
        w = wrapper(c, backend="auto", use_tensor_cores=True, plan_kwargs=plan_kwargs)
        return w, run(w, c)

    return go


# ----------------------------------------------------------------------------- spec


class Spec(KernelSpec):
    def runs_here(self) -> bool:
        return _sm103()

    def auto_here(self) -> bool:
        return (
            _sm103()
            and torch.cuda.get_device_properties(0).multi_processor_count
            in kmod.envelope()["sm_counts"]
        )

    def call(self, **overrides) -> DecodePlanCall:
        kw: Dict[str, Any] = dict(
            device=torch.device("cuda", 0),
            kv_layout="HND",
            batch_size=8,
            num_qo_heads=HEADS,
            num_kv_heads=1,
            head_dim=D,
            page_size=PAGE,
            q_len_per_req=Q,
            q_data_type=FP8,
            kv_data_type=FP8,
            o_data_type=torch.float16,
            pos_encoding_mode="NONE",
            window_left=-1,
            window_right=-1,
            logits_soft_cap=0.0,
            sm_scale=SCALE,
            fixed_split_size=-1,
            disable_split_kv=False,
            use_token_head_sf=False,
            custom_jit=False,
        )
        max_kv = overrides.pop("max_kv_len", MAX_KV)
        kv_lens = overrides.pop("kv_lens", None)
        kw.update(overrides)
        if kv_lens is None:
            kv_lens = [max_kv] * max(kw["batch_size"], 1)
        kw["kv_lens"] = torch.tensor(kv_lens, dtype=torch.int32)
        return DecodePlanCall(**kw)

    def pin_device(self, monkeypatch, *, sms: Optional[int] = None) -> None:
        monkeypatch.setattr(framework, "device_capability", lambda index: (10, 3))
        monkeypatch.setattr(
            framework,
            "device_sm_count",
            lambda index: kmod.envelope()["sm_counts"][0] if sms is None else sms,
        )

    def case(self, kv_lens, *, seed, **kwargs):
        return case(kv_lens, seed=seed, **kwargs)

    def wrapper(self, case_, **kwargs):
        return wrapper(case_, **kwargs)

    def plan(self, w, case_, **kwargs):
        plan(w, case_, **kwargs)

    def run(self, w, case_, out=None, **kwargs):
        return run(w, case_, out, **kwargs)

    def reference(self, case_, **kwargs):
        return reference(*case_, **kwargs)

    def check_close(self, out, ref):
        assert_close(out, ref)

    def check_stock(self, out, ref, kernel_out=None):
        assert_close(out, ref, **STOCK_TOL)
        if kernel_out is not None:  # no coarser than the stock path's own FP8-P error
            assert row_error(kernel_out, ref) <= row_error(out, ref) + 0.02

    def lens(self, batch_size, *, seed):
        gen = torch.Generator().manual_seed(seed)
        lens = torch.randint(Q, MAX_KV + 1, (batch_size,), generator=gen).tolist()
        lens[0] = MAX_KV
        return lens


# ------------------------------------------------------------------- kernel checks


def _check_split_pages_view(spec, monkeypatch):
    """The stock path re-describes a 128-token page as two 64-token pages, no copy:
    for one interleaved K/V buffer and for a compact pool."""
    pages = 5
    buf = torch.randn(pages, 1, PAGE, 2, D)
    for cache in (buf[:, :, :, 0, :], torch.randn(pages, 1, PAGE, D)):
        half = kmod._split_pages(cache)
        assert half.shape == (2 * pages, 1, PAGE // 2, D)
        assert half.data_ptr() == cache.data_ptr()
        for p in range(pages):
            torch.testing.assert_close(half[2 * p, 0], cache[p, 0, : PAGE // 2])
            torch.testing.assert_close(half[2 * p + 1, 0], cache[p, 0, PAGE // 2 :])
    with pytest.raises(ValueError, match="page stride"):
        kmod._split_pages(torch.randn(pages, 1, PAGE + 8, D)[:, :, :PAGE])


def _check_run_refusals_without_a_plan(spec, monkeypatch):
    """Tensor scales and a malformed q are refused by the kernel's run()."""
    state = KERNEL(
        torch.empty(16, dtype=torch.uint8), use_cuda_graph=False, fixed_batch_size=0
    )
    q = torch.zeros((Q, HEADS, D), dtype=FP8)
    kv = torch.zeros((2, 1, PAGE, D), dtype=FP8)
    base = dict(
        stock=False, out=None, q_scale=None, k_scale=None, v_scale=None, enable_pdl=None
    )
    with pytest.raises(RuntimeError, match="plan"):
        state.run(q, kv, kv, **base)
    state._batch_size = 1
    with pytest.raises(NotImplementedError, match="k_scale"):
        state.run(q, kv, kv, **dict(base, q_scale=1.0, k_scale=torch.ones(1)))
    with pytest.raises(ValueError, match="contiguous float8_e4m3fn q"):
        state.run(q[:, :8], kv, kv, **base)


def _check_length_edges_and_scales(spec, monkeypatch):
    """Page edges and the shortest lengths the wrapper admits (q_len_per_req), with
    q / k / v scales folded into the softmax and output scales."""
    monkeypatch.delenv(framework.KILL_SWITCH_ENV, raising=False)
    kv_lens = [7, 8, 127, 128, 129, 255, 256, 257, 4095, 4096, 4097, 4224, 4229, 4232]
    c = case(kv_lens, seed=11)
    w = wrapper(c)
    out = run(w, c, q_scale=0.5, k_scale=1.5, v_scale=0.25)
    assert_close(out, reference(*c, sm_scale=SCALE * 0.75, v_scale=0.25))


SPEC = Spec(
    name=kmod.NAME,
    kernel_class=KERNEL,
    compute_capability=(10, 3),
    refusal_cases={
        "q_len_8": ({"q_len_per_req": 8}, "q_len"),
        "q_len_1": ({"q_len_per_req": 1}, "q_len"),
        "causal": ({"window_right": 0}, "mask"),  # the wrapper's multi-token default
        "window_left": ({"window_left": 4095}, "mask"),
        "rope": ({"pos_encoding_mode": "ROPE_LLAMA"}, "rope"),
        "soft_cap": ({"logits_soft_cap": 30.0}, "soft_cap"),
        "token_head_sf": ({"use_token_head_sf": True}, "kv_scales"),
        "fixed_split": ({"fixed_split_size": 8}, "split"),
        "disable_split": ({"disable_split_kv": True}, "split"),
        "jit": ({"custom_jit": True}, "jit"),
        "scale_0": ({"sm_scale": 0.0}, "scale"),
        "scale_neg": ({"sm_scale": -SCALE}, "scale"),
        "bf16_out": ({"o_data_type": torch.bfloat16}, "dtype"),
        "bf16_qkv": (
            {"q_data_type": torch.bfloat16, "kv_data_type": torch.bfloat16},
            "dtype",
        ),
        "heads_32": ({"num_qo_heads": 32}, "geometry"),
        "kv_heads_2": ({"num_kv_heads": 2}, "geometry"),
        "head_dim_64": ({"head_dim": 64}, "geometry"),
        "nhd": ({"kv_layout": "NHD"}, "kv_layout"),
        "page_64": ({"page_size": 64}, "page"),
        "kv_4353": ({"max_kv_len": 4353}, "max_kv_len"),
        "batch_65": ({"batch_size": 65}, "batch_size"),
        "batch_0": ({"batch_size": 0}, "batch_size"),
        "cpu": ({"device": torch.device("cpu")}, "arch"),
    },
    auto_refusal_cases={
        "inside": ({}, None),
        "padded_row": ({"kv_lens": [0, 300, 4232], "batch_size": 3}, "kv_len"),
        "short_row": ({"kv_lens": [6, 4232], "batch_size": 2}, "kv_len"),
        "lengths_vs_batch": ({"kv_lens": [300, 4232], "batch_size": 3}, "kv_len"),
        "tables_rows": (
            {"block_tables": torch.zeros(1, 34, dtype=torch.int32)},
            "block_tables",
        ),
        "tables_1d": (
            {"block_tables": torch.zeros(68, dtype=torch.int32)},
            "block_tables",
        ),
        "sm_count": ({"_sms": 148}, "sm_count"),
        "kv_4353": ({"kv_lens": [4353, 300]}, "max_kv_len"),
        "batch_65": ({"kv_lens": [300] * 65, "batch_size": 65}, "batch_size"),
        "q_len_8": ({"q_len_per_req": 8}, "q_len"),
        "window_left": ({"window_left": 1024}, "mask"),
    },
    # every configuration the 152-SM selection returns: Dual C12 x 8 / 4 row groups (B1, B2),
    # C14 (B3), C12, C10, C9, C6, C5 (B12-14), C4 (B16-18); Online C9 (B15), C6, C5, C4, C3, C2
    numerics_cases=tuple(
        (
            f"B{b}-{'block_tables' if t else 'csr'}-{'interleaved' if i else 'separate'}",
            dict(batch_size=b, table=t, interleaved=i),
        )
        for b in (
            1,
            2,
            3,
            4,
            5,
            7,
            8,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            20,
            26,
            32,
            39,
            64,
        )
        for t in (True, False)
        for i in (True, False)
    ),
    graph_batches=(4, 26),
    auto_cases=tuple(
        (
            f"B{b}-{'block_tables' if t else 'csr'}",
            dict(batch_size=b, use_tensor_cores=tc, table=t),
        )
        for b, tc in (
            (1, False),
            (2, True),
            (7, False),
            (8, True),
            (15, False),
            (18, True),
            (26, False),
            (39, True),
            (64, False),
        )
        for t in (True, False)
    ),
    auto_graph_batches=(4, 26),
    short_rows=(0, 3),
    off_case_plans={
        "causal": ({"window_right": 0}, "window_right=0"),
        "q_len_8": ({"q_len_per_req": 8}, "q_len_per_req=8"),
        "bf16_out": ({"o_data_type": torch.bfloat16}, "dtypes"),
        "soft_cap": ({"logits_soft_cap": 30.0}, "logits_soft_cap"),
    },
    off_case_runs={
        "return_lse": ({"return_lse": True}, "return_lse"),
        "sinks": ({"sinks": torch.zeros(HEADS)}, "sinks"),
        "tensor_q_scale": ({"q_scale": torch.ones(1)}, "q_scale"),
    },
    auto_route_cases={
        # declined dense candidates: auto refuses them, as before
        "dense_b65": _plan_run([300] * 65),
        "dense_kv4353": _plan_run([4353, 300], table_width=35),
        "dense_padded_row": _plan_run([0, 300, 4232]),
        "dense_short_row": _plan_run([6, 4232]),
        "dense_bf16_out": _plan_run([300, 4232], o_data_type=torch.bfloat16),
        "dense_q8": _plan_run([300, 4232], q_len_per_req=8),
        "dense_soft_cap": _plan_run([300, 4232], logits_soft_cap=30.0),
        # a causal plan of the same geometry: not a candidate
        "causal_q7_fp8": _plan_run([300, 4232], window_right=0),
        # the case itself: unchanged off the measured envelope (e.g. B200)
        "in_case": _plan_run([4232] * 8),
    },
    checks=(
        Check("split_pages_view", _check_split_pages_view, needs_kernel=False),
        Check(
            "run_refusals_without_a_plan",
            _check_run_refusals_without_a_plan,
            needs_kernel=False,
        ),
        Check("length_edges_and_scales", _check_length_edges_and_scales),
    ),
)
