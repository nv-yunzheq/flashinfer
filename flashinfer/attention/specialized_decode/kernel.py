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

"""Interface of one kernel of ``BatchDecodeWithPagedKVCacheWrapper(backend="specialized")``,
the description of a ``plan()`` call the kernels judge, and the kill switch.

A kernel serves one decode case: a model's attention geometry, dtypes, mask, page size,
batch and KV-length ranges on one GPU architecture. The contract, which the backend
(:class:`~.backend.SpecializedDecode`) relies on and the shared tests check:

* :meth:`SpecializedDecodeKernel.plan_refusal` is the case: ``None`` when the kernel
  computes the call, else ``(reason, detail)``. An explicit ``backend="specialized"``
  plan that no kernel serves raises ``NotImplementedError`` with the kernel's
  :meth:`~SpecializedDecodeKernel.refusal_message`. Never raises;
* :meth:`SpecializedDecodeKernel.auto_candidate` and
  :meth:`SpecializedDecodeKernel.auto_refusal` decide ``backend="auto"``: a cheap screen
  (no device query, no KV-length read), then the envelope the kernel was measured in, on
  top of the case. ``auto`` takes the kernel only inside it. Never raises;
* :meth:`SpecializedDecodeKernel.plan` stages the call's metadata in buffers the kernel
  owns and compiles what it launches, outside any CUDA-graph capture, so
  :meth:`SpecializedDecodeKernel.run` is capture-safe and a captured ``run()`` replays
  against the latest plan; :meth:`SpecializedDecodeKernel.plan_stock` plans the
  wrapper's stock path for the same call instead (the kill switch);
* :meth:`SpecializedDecodeKernel.stats` reports kernel-specific diagnostics without side
  effects.

The kill switch ``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1`` is read by the backend, once
per plan, never by a kernel.
"""

from __future__ import annotations

import abc
import functools
import os
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, Optional, Tuple

import torch

# ``1``: ``auto`` never takes the specialized backend, and an explicit
# ``backend="specialized"`` plan runs the wrapper's stock path for the call. Read at
# ``plan()``. The same variable switches FlashInfer's other specialized kernels off.
KILL_SWITCH_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"

API_NAME = "BatchDecodeWithPagedKVCacheWrapper"


def kill_switch_set() -> bool:
    """Whether ``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1`` is set now."""
    return os.environ.get(KILL_SWITCH_ENV, "") == "1"


@functools.cache
def device_capability(device_index: int) -> Tuple[int, int]:
    """Compute capability of a CUDA device (cached)."""
    major, minor = torch.cuda.get_device_capability(device_index)
    return int(major), int(minor)


@functools.cache
def device_sm_count(device_index: int) -> int:
    """Streaming multiprocessors of a CUDA device (cached)."""
    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


@dataclass(slots=True)
class DecodePlanCall:
    """One ``BatchDecodeWithPagedKVCacheWrapper.plan()`` call as the kernels judge it.

    ``indptr`` / ``indices`` / ``last_page_len`` are the CSR page metadata the wrapper
    planned with (device copies when the kernel plans), ``seq_lens`` / ``block_tables``
    the optional caller tensors. The per-request KV lengths (:attr:`kv_lens_host`) are
    read on first use only: from ``seq_lens`` when given, else from the CSR metadata.
    """

    device: torch.device
    kv_layout: str
    batch_size: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    q_len_per_req: int
    q_data_type: torch.dtype
    kv_data_type: torch.dtype
    o_data_type: torch.dtype
    pos_encoding_mode: str
    window_left: int
    window_right: int
    logits_soft_cap: Optional[float]
    sm_scale: Optional[float]
    fixed_split_size: int  # -1: none
    disable_split_kv: bool
    use_token_head_sf: bool
    custom_jit: bool
    indptr: Optional[torch.Tensor] = None
    indices: Optional[torch.Tensor] = None
    last_page_len: Optional[torch.Tensor] = None
    seq_lens: Optional[torch.Tensor] = None
    block_tables: Optional[torch.Tensor] = None
    kv_lens: Optional[torch.Tensor] = field(default=None, repr=False)

    @property
    def kv_lens_host(self) -> torch.Tensor:
        """Per-request KV lengths on the host (int32), read once."""
        if self.kv_lens is None:
            if self.seq_lens is not None:
                self.kv_lens = self.seq_lens.cpu().to(torch.int32)
            else:
                from ...page import get_seq_lens

                assert self.indptr is not None and self.last_page_len is not None
                self.kv_lens = get_seq_lens(
                    self.indptr.to("cpu"),
                    self.last_page_len.to("cpu"),
                    self.page_size,
                )
        return self.kv_lens

    @property
    def max_kv_len(self) -> int:
        lens = self.kv_lens_host
        return int(lens.max()) if lens.numel() else 0

    @property
    def min_kv_len(self) -> int:
        lens = self.kv_lens_host
        return int(lens.min()) if lens.numel() else 0

    @property
    def compute_capability(self) -> Optional[Tuple[int, int]]:
        """The device's compute capability, ``None`` off CUDA."""
        if self.device.type != "cuda":
            return None
        return device_capability(
            self.device.index if self.device.index is not None else 0
        )


class SpecializedDecodeKernel(abc.ABC):
    """One kernel of the specialized decode backend, with its per-wrapper state.

    A subclass declares its registry name and architectures and is exported as
    ``KERNEL`` by its subpackage of ``flashinfer/attention/specialized_decode/kernels/``,
    whose name equals :attr:`name`. The backend creates one instance per wrapper the
    first time a plan picks the kernel; process-wide state (compiled programs,
    diagnostics) belongs to the kernel's module.
    """

    #: Registry name: the kernel's subpackage name.
    name: ClassVar[str] = ""
    #: Compute capabilities ``(major, minor)`` the kernel runs on; ``auto`` asks a
    #: kernel only on these.
    compute_capabilities: ClassVar[Tuple[Tuple[int, int], ...]] = ()

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        *,
        use_cuda_graph: bool,
        fixed_batch_size: int,
    ) -> None:
        self.workspace = float_workspace_buffer
        self.device = float_workspace_buffer.device
        self.use_cuda_graph = use_cuda_graph
        self.fixed_batch_size = fixed_batch_size

    @classmethod
    @abc.abstractmethod
    def plan_refusal(cls, call: DecodePlanCall) -> Optional[Tuple[str, str]]:
        """``(reason, detail)`` when ``call`` is outside the kernel's case, else ``None``.
        Never raises."""

    @classmethod
    @abc.abstractmethod
    def refusal_message(cls, detail: str) -> str:
        """The ``NotImplementedError`` text for an explicit plan outside the case."""

    @classmethod
    @abc.abstractmethod
    def auto_candidate(cls, call: DecodePlanCall) -> bool:
        """Whether ``backend="auto"`` asks this kernel about ``call`` at all (cheap: no
        device query, no KV-length read). Declines of candidates are counted."""

    @classmethod
    def auto_refusal(cls, call: DecodePlanCall) -> Optional[Tuple[str, str]]:
        """``(reason, detail)`` when ``auto`` does not take the kernel for a call inside
        its case (outside the measured envelope), else ``None``. Never raises."""
        return None

    @abc.abstractmethod
    def plan(self, call: DecodePlanCall, *, non_blocking: bool) -> None:
        """Stage the call's metadata and compile; outside any CUDA-graph capture."""

    @abc.abstractmethod
    def plan_stock(self, call: DecodePlanCall, *, non_blocking: bool) -> None:
        """Plan the wrapper's stock path for the same call (the kill switch)."""

    @abc.abstractmethod
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
        """Launch the planned call (the stock path when ``stock``) on the current
        stream. Options the kernel does not take are refused, never ignored."""

    @classmethod
    def stats(cls) -> Dict[str, Any]:
        """Kernel-specific diagnostics (e.g. compiled configurations); side-effect free."""
        return {}
