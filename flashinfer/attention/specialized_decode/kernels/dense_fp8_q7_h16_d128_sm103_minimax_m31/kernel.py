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

"""Specialized decode kernel ``dense_fp8_q7_h16_d128_sm103_minimax_m31``.

Not a public API: :class:`flashinfer.BatchDecodeWithPagedKVCacheWrapper` reaches it through
``backend="specialized"`` (and ``backend="auto"`` inside the measured envelope).

One decode case, served by an SM103 CuTe DSL kernel written for it
(:mod:`flashinfer.cute_dsl.attention.gqa_decode_cluster_q7`), the draft-model attention of
MiniMax-M3.1's speculative decoding:

* SM103 device (compute capability 10.3);
* FP8-E4M3 query and K/V, FP16 output, 16 query heads over 1 K/V head, head dim 128;
* ``kv_layout="HND"``, page size 128, KV lengths up to 4352 (34 pages);
* 7 query tokens per request (``q_len_per_req=7``), dense over the block:
  ``window_left=-1`` and ``window_right=-1`` (the wrapper's multi-token default,
  ``window_right=0``, is causal);
* no RoPE, logits soft-cap, sinks, LSE output, NVFP4 / token-head K/V scales, fixed KV
  split or custom JIT module; host-float ``q_scale`` / ``k_scale`` / ``v_scale``;
* batch size 1 to 64.

``backend="auto"`` takes the kernel for such a call when the call is also inside the
measured envelope (``workloads.json``: the SM counts and batch sizes it was benchmarked
at against every functionally equivalent backend; every KV length >= ``q_len_per_req``,
which the wrapper's multi-token plan requires, so a padded CUDA-graph row with
``seq_len`` 0 is declined).

:meth:`plan` compiles the kernel configuration for the plan's batch size (outside any
CUDA-graph capture) and stages the KV lengths and a 34-page block table (from the caller's
``block_tables`` or from the CSR page indices) in buffers the kernel owns; :meth:`run`
launches the kernel on the current stream. The kernel needs no workspace. The stock path
(:meth:`plan_stock`, the kill switch) is the wrapper's ``cute-dsl`` decode on 64-token
half pages of the same K/V pool (the pool is re-described and each block-table entry split
in two; nothing is copied).

Nothing here imports the CuTe DSL at module scope: the kernel module is loaded by the
first plan.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import math
import numbers
import threading
from importlib import resources
from typing import Any, Dict, Optional, Tuple

import torch

from ... import kernel as _framework
from ...kernel import API_NAME, DecodePlanCall, SpecializedDecodeKernel

NAME = "dense_fp8_q7_h16_d128_sm103_minimax_m31"

NUM_QO_HEADS = 16
NUM_KV_HEADS = 1
HEAD_DIM = 128
PAGE_SIZE = 128
Q_LEN_PER_REQ = 7
TABLE_PAGES = 34
MAX_KV_LEN = TABLE_PAGES * PAGE_SIZE
MIN_BATCH = 1
MAX_BATCH = 64
COMPUTE_CAPABILITY = (10, 3)
# The stock path: the cute-dsl decode plans pages of 8..64 tokens.
FALLBACK_PAGE_SIZE = 64
# Rows of a request with no keys are left as they are. The wrapper's multi-token plan
# requires every KV length >= q_len_per_req, so the kernel never meets such a request here.
_EMPTY_ROWS = "skip"
# The measured envelope of backend="auto" (package data next to this module).
_ENVELOPE_FILE = "workloads.json"

_FP8 = torch.float8_e4m3fn

_LOCK = threading.Lock()
_STATS: Dict[str, Any] = {"compiled_variants": []}


def stats() -> Dict[str, Any]:
    """Compiled kernel configurations (``cuda:<device>/<configuration>``)."""
    with _LOCK:
        return {"compiled_variants": list(_STATS["compiled_variants"])}


@functools.cache
def envelope() -> Dict[str, Any]:
    """The measured envelope ``auto`` admits (``workloads.json``): ``sm_counts`` and
    ``batch_sizes``."""
    text = resources.files(__package__).joinpath(_ENVELOPE_FILE).read_text()
    doc = json.loads(text)
    return {
        "sm_counts": tuple(int(x) for x in doc["sm_counts"]),
        "batch_sizes": frozenset(int(x) for x in doc["batch_sizes"]),
    }


def _split_pages(cache: torch.Tensor) -> torch.Tensor:
    """``[pages, 1, 128, d]`` HND view -> ``[2 * pages, 1, 64, d]`` (64-token halves), no copy."""
    p, h, t, d = cache.shape
    s_page, s_head, s_tok, s_dim = cache.stride()
    if s_page != t * s_tok:
        raise ValueError(
            "backend='specialized' with the kill switch needs each K/V page to be its "
            f"{t} tokens at the token stride (page stride {s_page}, token stride {s_tok})"
        )
    half = t // 2
    return cache.as_strided(
        (2 * p, h, half, d),
        (half * s_tok, s_head, s_tok, s_dim),
        cache.storage_offset(),
    )


def _is_host_scalar(x: Any) -> bool:
    return x is None or (isinstance(x, numbers.Real) and not isinstance(x, bool))


class DenseFp8Q7Sm103MiniMaxM31Kernel(SpecializedDecodeKernel):
    """The 7-token dense FP8 GQA-16 decode on SM103 (see the module docstring)."""

    name = NAME
    compute_capabilities = (COMPUTE_CAPABILITY,)

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        *,
        use_cuda_graph: bool,
        fixed_batch_size: int,
    ) -> None:
        super().__init__(
            float_workspace_buffer,
            use_cuda_graph=use_cuda_graph,
            fixed_batch_size=fixed_batch_size,
        )
        self._seq_lens: Optional[torch.Tensor] = None
        self._block_tables: Optional[torch.Tensor] = None
        self._compiled: Any = None
        self._config_name: Optional[str] = None
        self._batch_size = 0
        self._sm_scale = 1.0 / math.sqrt(HEAD_DIM)
        self._fallback: Any = (
            None  # BatchDecodeWithPagedKVCacheWrapper(backend="cute-dsl")
        )
        self._fallback_meta: Optional[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None

    # ------------------------------------------------------------------ guards

    @classmethod
    def plan_refusal(cls, call: DecodePlanCall) -> Optional[Tuple[str, str]]:
        """Semantics first, then geometry and sizes, the device last (so the checks run
        without a GPU; the KV lengths are read only once everything before them holds)."""
        if call.q_len_per_req != Q_LEN_PER_REQ:
            return (
                "q_len",
                f"q_len_per_req={call.q_len_per_req} (needs {Q_LEN_PER_REQ})",
            )
        if call.window_right != -1 or call.window_left != -1:
            return "mask", (
                f"window_left={call.window_left}, window_right={call.window_right} "
                "(needs -1 and -1: a dense block with no sliding window)"
            )
        if call.pos_encoding_mode != "NONE":
            return (
                "rope",
                f"pos_encoding_mode={call.pos_encoding_mode!r} (needs 'NONE')",
            )
        if call.logits_soft_cap:
            return "soft_cap", f"logits_soft_cap={call.logits_soft_cap} (needs none)"
        if call.use_token_head_sf:
            return "kv_scales", "use_token_head_sf=True (needs host-float scales)"
        if call.fixed_split_size > 0 or call.disable_split_kv:
            return (
                "split",
                "fixed_split_size / disable_split_kv (the kernel picks its split)",
            )
        if call.custom_jit:
            return "jit", "a custom JIT module (jit_args)"
        if call.sm_scale is not None and not call.sm_scale > 0.0:
            return "scale", f"sm_scale={call.sm_scale} (needs > 0)"
        if not (
            call.q_data_type == _FP8
            and call.kv_data_type == _FP8
            and call.o_data_type == torch.float16
        ):
            return "dtype", (
                f"q/kv/o dtypes {call.q_data_type}/{call.kv_data_type}/{call.o_data_type} "
                "(needs float8_e4m3fn/float8_e4m3fn/float16)"
            )
        geometry = (call.num_qo_heads, call.num_kv_heads, call.head_dim)
        if geometry != (NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM):
            return "geometry", (
                f"num_qo_heads={call.num_qo_heads}, num_kv_heads={call.num_kv_heads}, "
                f"head_dim={call.head_dim} (needs 16, 1, 128)"
            )
        if call.kv_layout != "HND":
            return "kv_layout", f"kv_layout={call.kv_layout!r} (needs 'HND')"
        if call.page_size != PAGE_SIZE:
            return "page", f"page_size={call.page_size} (needs {PAGE_SIZE})"
        max_kv_len = call.max_kv_len
        if not 1 <= max_kv_len <= MAX_KV_LEN:
            return "max_kv_len", f"max KV length {max_kv_len} (needs 1..{MAX_KV_LEN})"
        if not MIN_BATCH <= call.batch_size <= MAX_BATCH:
            return (
                "batch_size",
                f"batch_size={call.batch_size} (needs {MIN_BATCH}..{MAX_BATCH})",
            )
        if call.compute_capability != COMPUTE_CAPABILITY:
            return "arch", "a device other than SM103"
        return None

    @classmethod
    def refusal_message(cls, detail: str) -> str:
        return (
            f"{API_NAME}(backend='specialized') serves one decode case (SM103; "
            "e4m3 Q/KV, fp16 out; 16 query heads over 1 K/V head, head_dim 128; HND, "
            f"page 128, KV length <= {MAX_KV_LEN}; q_len_per_req=7, window_left=-1, "
            "window_right=-1; batch 1..64); this plan has "
            f"{detail}. Use another backend for it."
        )

    @classmethod
    def auto_candidate(cls, call: DecodePlanCall) -> bool:
        # a dense multi-token plan: one backend="auto" refused before this kernel existed
        return call.q_len_per_req > 1 and call.window_right == -1

    @classmethod
    def auto_refusal(cls, call: DecodePlanCall) -> Optional[Tuple[str, str]]:
        if importlib.util.find_spec("cutlass") is None:
            return "import", "the CuTe DSL is not installed"
        lens = call.kv_lens_host
        if lens.numel() != call.batch_size:
            return (
                "kv_len",
                f"{lens.numel()} KV lengths for batch_size={call.batch_size}",
            )
        min_kv_len = int(lens.min())
        if min_kv_len < Q_LEN_PER_REQ:
            return "kv_len", (
                f"a request with {min_kv_len} keys (needs >= q_len_per_req={Q_LEN_PER_REQ}, "
                "as every multi-token plan of the wrapper; a padded CUDA-graph row has 0)"
            )
        tables = call.block_tables
        if tables is not None and (
            tables.dim() != 2 or tables.shape[0] < call.batch_size
        ):
            return "block_tables", f"block_tables of shape {tuple(tables.shape)}"
        env = envelope()
        if call.batch_size not in env["batch_sizes"]:
            return "batch_size", f"batch_size={call.batch_size} (not measured)"
        index = call.device.index if call.device.index is not None else 0
        sms = _framework.device_sm_count(index)
        if sms not in env["sm_counts"]:
            return "sm_count", (
                f"{sms} SMs (measured on {', '.join(map(str, env['sm_counts']))})"
            )
        return None

    # ------------------------------------------------------------------ plan

    def _buffers(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """The owned length / table buffers, sized once under CUDA graphs."""
        cap = self.fixed_batch_size if self.use_cuda_graph else batch_size
        if self._seq_lens is None or self._seq_lens.shape[0] < cap:
            self._seq_lens = torch.zeros((cap,), dtype=torch.int32, device=self.device)
            self._block_tables = torch.zeros(
                (cap, TABLE_PAGES), dtype=torch.int32, device=self.device
            )
        assert self._block_tables is not None
        return self._seq_lens, self._block_tables

    def _stage(self, call: DecodePlanCall, non_blocking: bool) -> torch.Tensor:
        """Scale, KV lengths and the 34-page block table into the owned buffers."""
        if call.sm_scale is not None:
            self._sm_scale = float(call.sm_scale)
        else:
            self._sm_scale = 1.0 / math.sqrt(call.head_dim)
        batch_size = call.batch_size
        kv_lens_host = call.kv_lens_host
        seq_lens, table = self._buffers(batch_size)
        seq_lens[:batch_size].copy_(
            kv_lens_host.to(torch.int32), non_blocking=non_blocking
        )
        tbl = table[:batch_size]
        block_tables = call.block_tables
        if block_tables is not None:
            if block_tables.dim() != 2 or block_tables.shape[0] < batch_size:
                raise ValueError(
                    f"block_tables must be [batch_size, max_pages], got {tuple(block_tables.shape)}"
                )
            width = min(int(block_tables.shape[1]), TABLE_PAGES)
            tbl[:, :width].copy_(
                block_tables[:batch_size, :width], non_blocking=non_blocking
            )
            if width < TABLE_PAGES:
                tbl[:, width:].zero_()
        else:
            # pages [indptr[b], indptr[b + 1]) of the CSR, on the device (no host readback)
            indptr, indices = call.indptr, call.indices
            assert indptr is not None and indices is not None
            starts = indptr[:batch_size].to(torch.int64)
            counts = indptr[1 : batch_size + 1].to(torch.int64) - starts
            cols = torch.arange(TABLE_PAGES, dtype=torch.int64, device=indices.device)
            valid = cols[None, :] < counts[:, None]
            pos = (starts[:, None] + cols[None, :]).clamp(
                max=max(indices.numel() - 1, 0)
            )
            tbl.copy_(indices[pos].to(torch.int32).masked_fill_(~valid, 0))
        self._batch_size = batch_size
        return tbl

    def plan(self, call: DecodePlanCall, *, non_blocking: bool) -> None:
        self._stage(call, non_blocking)
        from .....cute_dsl.attention.gqa_decode_cluster_q7 import (
            get_compiled,
            kernel_config,
        )

        device_index = self.device.index if self.device.index is not None else 0
        config = kernel_config(call.batch_size, device_index)
        self._compiled = get_compiled(device_index, call.batch_size, _EMPTY_ROWS)
        self._config_name = config.name
        with _LOCK:
            key = f"cuda:{device_index}/{config.name}"
            if key not in _STATS["compiled_variants"]:
                _STATS["compiled_variants"].append(key)

    def plan_stock(self, call: DecodePlanCall, *, non_blocking: bool) -> None:
        """The stock cute-dsl decode of the same call on 64-token half pages."""
        table = self._stage(call, non_blocking)
        self._compiled = None
        self._config_name = None
        from .....decode import BatchDecodeWithPagedKVCacheWrapper

        batch_size = call.batch_size
        width = 2 * TABLE_PAGES
        cap = self.fixed_batch_size if self.use_cuda_graph else batch_size
        meta = self._fallback_meta
        if meta is None or meta[2].shape[0] < cap:
            indptr = (
                torch.arange(cap + 1, dtype=torch.int32, device=self.device) * width
            )
            indices = torch.zeros((cap * width,), dtype=torch.int32, device=self.device)
            last = torch.full(
                (cap,), FALLBACK_PAGE_SIZE, dtype=torch.int32, device=self.device
            )
            meta = self._fallback_meta = (indptr, indices, last)
            self._fallback = None
        indptr, indices, last = meta
        # entry p of the 128-token table -> 64-token pages 2p and 2p + 1
        halves = torch.stack((2 * table, 2 * table + 1), dim=-1).reshape(
            batch_size, width
        )
        indices[: batch_size * width].copy_(halves.reshape(-1))
        if self._fallback is None:
            graph = (
                dict(
                    paged_kv_indptr_buffer=indptr,
                    paged_kv_indices_buffer=indices,
                    paged_kv_last_page_len_buffer=last,
                )
                if self.use_cuda_graph
                else {}
            )
            self._fallback = BatchDecodeWithPagedKVCacheWrapper(
                self.workspace,
                "HND",
                use_cuda_graph=self.use_cuda_graph,
                backend="cute-dsl",
                **graph,
            )
        self._fallback.plan(
            indptr[: batch_size + 1],
            indices[: batch_size * width],
            last[:batch_size],
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            FALLBACK_PAGE_SIZE,
            q_data_type=_FP8,
            kv_data_type=_FP8,
            o_data_type=torch.float16,
            sm_scale=self._sm_scale,
            q_len_per_req=Q_LEN_PER_REQ,
            window_left=-1,
            window_right=-1,
            seq_lens=call.kv_lens_host,
            non_blocking=non_blocking,
        )

    # ------------------------------------------------------------------ run

    def run(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        stock: bool,
        out: Optional[torch.Tensor],
        q_scale: Any,
        k_scale: Any,
        v_scale: Any,
        enable_pdl: Optional[bool],
        positional_args: tuple = (),
        return_lse: bool = False,
        lse: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
        kv_cache_sf: Any = None,
        skip_softmax_threshold_scale_factor: Optional[float] = None,
        window_left: Optional[int] = None,
        q_len_per_req: Optional[int] = None,
    ) -> torch.Tensor:
        """Launch on the current stream (capture-safe: no compilation, allocation or host sync
        when ``out`` is given). The other :meth:`BatchDecodeWithPagedKVCacheWrapper.run`
        options are refused, never ignored."""
        unsupported = {
            "positional arguments": bool(positional_args),
            "return_lse": return_lse,
            "lse": lse is not None,
            "sinks": sinks is not None,
            "kv_cache_sf": kv_cache_sf is not None,
            "skip_softmax_threshold_scale_factor": (
                skip_softmax_threshold_scale_factor is not None
            ),
            f"window_left={window_left}": window_left not in (None, -1),
            f"q_len_per_req={q_len_per_req}": q_len_per_req
            not in (None, Q_LEN_PER_REQ),
        }
        refused = [name for name, hit in unsupported.items() if hit]
        if refused:
            raise NotImplementedError(
                f"{API_NAME}(backend='specialized') does not take {', '.join(refused)}"
            )
        b = self._batch_size
        if b == 0:
            raise RuntimeError("plan() must complete before run().")
        for name, s in (
            ("q_scale", q_scale),
            ("k_scale", k_scale),
            ("v_scale", v_scale),
        ):
            if not _is_host_scalar(s):
                raise NotImplementedError(
                    f"backend='specialized' takes host-float scales; {name} is {type(s).__name__}"
                )
        rows = (b * Q_LEN_PER_REQ, NUM_QO_HEADS, HEAD_DIM)
        if q.dtype != _FP8 or tuple(q.shape) != rows or not q.is_contiguous():
            raise ValueError(
                f"backend='specialized' needs a contiguous float8_e4m3fn q of shape {rows} "
                f"(the planned batch x 7 rows), got {q.dtype} {tuple(q.shape)}"
            )
        for name, t in (("k_cache", k_cache), ("v_cache", v_cache)):
            if (
                t.dtype != _FP8
                or t.dim() != 4
                or tuple(t.shape[1:]) != (NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM)
                or t.stride(3) != 1
                or t.stride(2) < HEAD_DIM
                or t.stride(2) % 16
                or t.stride(0) % 16
                or t.data_ptr() % 16
                or t.device != q.device
            ):
                raise ValueError(
                    f"backend='specialized' needs {name} as an HND float8_e4m3fn "
                    f"[pages, 1, 128, 128] view with unit inner stride and 16-byte aligned "
                    f"token and page strides, got {t.dtype} {tuple(t.shape)} stride {t.stride()}"
                )
        if out is None:
            out = torch.empty(rows, dtype=torch.float16, device=q.device)
        elif (
            out.dtype != torch.float16
            or tuple(out.shape) != rows
            or not out.is_contiguous()
            or out.device != q.device
            or out.data_ptr() % 16
        ):
            raise ValueError(
                f"backend='specialized' needs a contiguous float16 out of shape {rows}, "
                f"got {out.dtype} {tuple(out.shape)}"
            )
        if q.data_ptr() % 16 or q.device != self.device:
            raise ValueError("q must be 16-byte aligned on the planned device")

        if stock:
            assert self._fallback is not None
            # the cute-dsl decode may reduce its KV splits by atomic adds into out
            out.zero_()
            return self._fallback.run(
                q,
                (_split_pages(k_cache), _split_pages(v_cache)),
                out=out,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                enable_pdl=enable_pdl,
            )

        bmm1 = self._sm_scale
        if q_scale is not None:
            bmm1 *= q_scale
        if k_scale is not None:
            bmm1 *= k_scale
        if not bmm1 > 0.0:
            raise ValueError(
                f"backend='specialized' needs a positive softmax scale, got {bmm1}"
            )
        bmm2 = 1.0 if v_scale is None else float(v_scale)
        from .....cute_dsl.attention.gqa_decode_cluster_q7 import launch

        assert self._seq_lens is not None and self._block_tables is not None
        launch(
            self._compiled,
            q.view(b, Q_LEN_PER_REQ, NUM_QO_HEADS, HEAD_DIM),
            k_cache,
            v_cache,
            self._seq_lens[:b],
            self._block_tables[:b],
            out.view(b, Q_LEN_PER_REQ, NUM_QO_HEADS, HEAD_DIM),
            bmm1,
            bmm2,
        )
        return out

    @classmethod
    def stats(cls) -> Dict[str, Any]:
        return stats()
