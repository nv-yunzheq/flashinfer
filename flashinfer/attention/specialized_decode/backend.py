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

"""``BatchDecodeWithPagedKVCacheWrapper(backend="specialized")``: the kernel chosen at
``plan()``, its per-wrapper state, the ``auto`` decision and the counters.

Dispatch, at ``plan()`` only: the registered kernels are asked in name order and the
first one whose case admits the call (explicit backend) or whose case and measured
envelope admit it (``auto``) serves it. ``run()`` launches what the plan chose. Under
``use_cuda_graph=True`` the kernel and the kill-switch state are part of the frozen plan.
The kill switch is read here, once per plan, before any kernel plans: an explicit plan
then runs the kernel's stock path for the call, and ``auto`` takes no kernel.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple, Type

import torch

from .kernel import (
    API_NAME,
    KILL_SWITCH_ENV,
    DecodePlanCall,
    SpecializedDecodeKernel,
    kill_switch_set,
)
from .registry import discovery_errors, registered_kernels

_LOCK = threading.Lock()
# Counters per kernel name; "" holds the backend's own (kill-switch declines of auto).
_COUNTERS: Dict[str, Dict[str, Any]] = {}


def _new_counters() -> Dict[str, Any]:
    return {
        "plans": 0,
        "fallback_plans": 0,
        "plan_refusals": {},
        "auto_plans": 0,
        "auto_declines": {},
        "dispatches": 0,
        "fallback_runs": 0,
    }


def _count(kernel: str, key: str, reason: Optional[str] = None) -> None:
    with _LOCK:
        c = _COUNTERS.setdefault(kernel, _new_counters())
        if reason is None:
            c[key] += 1
        else:
            c[key][reason] = c[key].get(reason, 0) + 1


def specialized_decode_stats() -> Dict[str, Any]:
    """Counters of the specialized decode backend in the process.

    Totals over the kernels: plans (kernel / kill-switch stock path), refused explicit
    plans by reason, plans ``backend="auto"`` took (``auto_plans``, also counted in
    ``plans``), candidate ``auto`` plans declined by reason (``auto_declines``, the kill
    switch included), compiled kernel configurations and launches (kernel / stock path).
    ``kernels`` holds the same counters per kernel name, with each kernel's own
    diagnostics."""
    totals = _new_counters()
    totals["compiled_variants"] = []
    per_kernel: Dict[str, Dict[str, Any]] = {}
    with _LOCK:
        snapshot = {
            k: {key: dict(v) if isinstance(v, dict) else v for key, v in c.items()}
            for k, c in _COUNTERS.items()
        }
    for name, kernel in registered_kernels().items():
        counters = snapshot.get(name, _new_counters())
        own = kernel.stats()
        per_kernel[name] = dict(counters, **own)
        totals["compiled_variants"].extend(own.get("compiled_variants", []))
    for counters in snapshot.values():
        for key, value in counters.items():
            if isinstance(value, dict):
                for reason, n in value.items():
                    totals[key][reason] = totals[key].get(reason, 0) + n
            else:
                totals[key] += value
    order = (
        "plans",
        "fallback_plans",
        "plan_refusals",
        "auto_plans",
        "auto_declines",
        "compiled_variants",
        "dispatches",
        "fallback_runs",
    )
    out: Dict[str, Any] = {key: totals[key] for key in order}
    out["kernels"] = per_kernel
    return out


def _refusal_error(
    refusals: List[Tuple[Type[SpecializedDecodeKernel], Tuple[str, str]]],
) -> NotImplementedError:
    if not refusals:
        errors = discovery_errors()
        detail = (
            f" ({'; '.join(f'{k}: {v}' for k, v in errors.items())})" if errors else ""
        )
        return NotImplementedError(
            f"{API_NAME}(backend='specialized') has no registered kernel{detail}."
        )
    if len(refusals) == 1:
        kernel, (_, detail) = refusals[0]
        return NotImplementedError(kernel.refusal_message(detail))
    parts = "; ".join(f"[{k.name}] {k.refusal_message(d)}" for k, (_, d) in refusals)
    return NotImplementedError(
        f"{API_NAME}(backend='specialized') serves {len(refusals)} decode cases and "
        f"this plan is in none of them: {parts}"
    )


def plan_kernel(call: DecodePlanCall) -> Type[SpecializedDecodeKernel]:
    """The kernel an explicit ``backend="specialized"`` plan runs on: the first
    registered kernel whose case admits ``call``. Raises ``NotImplementedError`` (each
    kernel's refusal counted) when none does."""
    refusals: List[Tuple[Type[SpecializedDecodeKernel], Tuple[str, str]]] = []
    for kernel in registered_kernels().values():
        refusal = kernel.plan_refusal(call)
        if refusal is None:
            return kernel
        refusals.append((kernel, refusal))
    for kernel, (reason, _) in refusals:
        _count(kernel.name, "plan_refusals", reason)
    raise _refusal_error(refusals)


def auto_kernel(call: DecodePlanCall) -> Optional[Type[SpecializedDecodeKernel]]:
    """The kernel ``backend="auto"`` plans ``call`` on, or ``None``. Never raises.

    The candidates are the registered kernels that run on the device's compute capability
    and screen the call in (:meth:`SpecializedDecodeKernel.auto_candidate`); with none,
    nothing is read or counted. With the kill switch set every candidate is declined.
    Otherwise the first candidate (name order) whose case and measured envelope admit the
    call is taken; each declining candidate is counted by reason."""
    try:
        cc = call.compute_capability
        candidates = [
            k
            for k in registered_kernels().values()
            if cc in k.compute_capabilities and k.auto_candidate(call)
        ]
    except Exception:  # noqa: BLE001  (auto keeps its route on any surprise)
        return None
    if not candidates:
        return None
    if kill_switch_set():
        _count("", "auto_declines", "kill_switch")
        return None
    for kernel in candidates:
        try:
            refusal = kernel.plan_refusal(call) or kernel.auto_refusal(call)
        except Exception as exc:  # noqa: BLE001
            refusal = ("error", f"{type(exc).__name__}: {exc}")
        if refusal is None:
            return kernel
        _count(kernel.name, "auto_declines", refusal[0])
    return None


class SpecializedDecode:
    """Plan and run state of ``backend="specialized"`` for one wrapper: the kernel the
    last plan chose, one state object per kernel used, and the CUDA-graph freezes."""

    #: The plan description :meth:`plan` takes (for callers that hold only this object).
    call_type = DecodePlanCall

    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        *,
        use_cuda_graph: bool,
        fixed_batch_size: int,
    ) -> None:
        self._workspace = float_workspace_buffer
        self._use_cuda_graph = use_cuda_graph
        self._fixed_batch_size = fixed_batch_size
        self._states: Dict[str, SpecializedDecodeKernel] = {}
        self._active: Optional[SpecializedDecodeKernel] = None
        self._stock = False
        self._frozen_fallback: Optional[bool] = None
        self._frozen_kernel: Optional[str] = None

    @property
    def kernel_name(self) -> Optional[str]:
        """The kernel the last plan chose."""
        return None if self._active is None else self._active.name

    @property
    def uses_fallback(self) -> bool:
        """Whether the last plan took the kill-switch (stock) path."""
        return self._active is not None and self._stock

    def plan(
        self,
        call: DecodePlanCall,
        *,
        non_blocking: bool,
        kernel: Optional[Type[SpecializedDecodeKernel]] = None,
        selected_by_auto: bool = False,
    ) -> None:
        """Choose the kernel (unless ``auto`` already did: ``kernel``), read the kill
        switch, check the CUDA-graph freezes, and plan the kernel or its stock path."""
        if kernel is None:
            kernel = plan_kernel(call)
        fallback = kill_switch_set()
        if self._use_cuda_graph:
            if self._frozen_fallback is not None and self._frozen_fallback != fallback:
                raise ValueError(
                    f"{KILL_SWITCH_ENV} is part of the frozen cudagraph plan: this wrapper "
                    f"was first planned {'with' if self._frozen_fallback else 'without'} "
                    "it; use a separate wrapper."
                )
            if self._frozen_kernel is not None and self._frozen_kernel != kernel.name:
                raise ValueError(
                    "The specialized kernel is part of the frozen cudagraph plan: this "
                    f"wrapper was first planned on {self._frozen_kernel!r} and this plan "
                    f"is served by {kernel.name!r}; use a separate wrapper."
                )
            self._frozen_fallback = fallback
            self._frozen_kernel = kernel.name
        state = self._states.get(kernel.name)
        if state is None:
            state = kernel(
                self._workspace,
                use_cuda_graph=self._use_cuda_graph,
                fixed_batch_size=self._fixed_batch_size,
            )
            self._states[kernel.name] = state
        if fallback:
            state.plan_stock(call, non_blocking=non_blocking)
            _count(kernel.name, "fallback_plans")
        else:
            state.plan(call, non_blocking=non_blocking)
            _count(kernel.name, "plans")
            if selected_by_auto:
                _count(kernel.name, "auto_plans")
        self._active = state
        self._stock = fallback

    def run(
        self, q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Launch what the last plan chose, on the current stream."""
        if self._active is None:
            raise RuntimeError("plan() must complete before run().")
        out = self._active.run(q, k_cache, v_cache, stock=self._stock, **kwargs)
        _count(self._active.name, "fallback_runs" if self._stock else "dispatches")
        return out
