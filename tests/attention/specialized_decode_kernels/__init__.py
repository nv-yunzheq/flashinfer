"""Test specs of the specialized decode backend's kernels (data, not test modules).

Every kernel registered under
``flashinfer/attention/specialized_decode/kernels/<name>/`` has a module ``<name>.py``
here that exports ``SPEC``, a :class:`KernelSpec` subclass instance.
``tests/attention/test_decode_specialized.py`` discovers the specs with :func:`load_specs`
(as the registry discovers the kernel subpackages) and runs every contract test of the
backend on each kernel; a spec's ``checks`` adds the kernel's internal checks. A new
kernel adds its spec module here and no line to the shared test file.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple


@dataclass(frozen=True)
class Check:
    """A kernel-internal check, run by ``test_kernel_check``.

    ``fn(spec, monkeypatch, **params)`` runs one ``params`` item. ``needs_kernel`` skips
    the check where the kernel does not run (``spec.runs_here()``).
    """

    name: str
    fn: Callable[..., None]
    params: Tuple[Tuple[str, Dict[str, Any]], ...] = (("", {}),)
    needs_kernel: bool = True


@dataclass
class KernelSpec:
    """What the shared tests need to know about one kernel.

    Host checks: ``refusal_cases`` (``plan_refusal`` reason per override of the served
    call, device facts pinned), ``auto_refusal_cases`` (the ``auto`` envelope). On the
    kernel's device: the served inputs (``case`` / ``wrapper`` / ``plan`` / ``run``), the
    reference and tolerances (``reference`` / ``check_close`` / ``check_stock``), the
    numerics grid, the batch sizes of the graph and ``auto`` tests, the off-case plans and
    runs, the padded rows, and the plans outside the ``auto`` envelope
    (``auto_route_cases``: builders whose outcome must equal ``auto``'s own route).
    """

    name: str
    kernel_class: type
    compute_capability: Tuple[int, int]
    refusal_cases: Dict[str, Tuple[Dict[str, Any], str]] = field(default_factory=dict)
    auto_refusal_cases: Dict[str, Tuple[Dict[str, Any], Optional[str]]] = field(
        default_factory=dict
    )
    numerics_cases: Tuple[Tuple[str, Dict[str, Any]], ...] = ()
    graph_batches: Tuple[int, ...] = ()
    auto_cases: Tuple[Tuple[str, Dict[str, Any]], ...] = ()
    auto_graph_batches: Tuple[int, ...] = ()
    short_rows: Tuple[int, ...] = ()
    off_case_plans: Dict[str, Tuple[Dict[str, Any], str]] = field(default_factory=dict)
    off_case_runs: Dict[str, Tuple[Dict[str, Any], str]] = field(default_factory=dict)
    auto_route_cases: Dict[str, Callable[[], Tuple[Any, Any]]] = field(
        default_factory=dict
    )
    # the auto-route case that IS the kernel's case (auto takes it inside the envelope)
    in_case: str = "in_case"
    checks: Tuple[Check, ...] = ()

    # -- hooks every spec implements -------------------------------------
    def runs_here(self) -> bool:
        """Whether the kernel runs on this host's GPU (explicit backend tests)."""
        raise NotImplementedError

    def auto_here(self) -> bool:
        """Whether this GPU is inside the kernel's ``auto`` envelope (device facts)."""
        raise NotImplementedError

    def call(self, **overrides) -> Any:
        """The served ``DecodePlanCall`` (host KV lengths, no GPU) with overrides."""
        raise NotImplementedError

    def pin_device(self, monkeypatch, *, sms: Optional[int] = None) -> None:
        """Pin the device facts the guards read to the kernel's device."""
        raise NotImplementedError

    def case(self, kv_lens, *, seed: int, **kwargs) -> Tuple[Any, ...]:
        """Served-layout inputs ``(q, k_cache, v_cache, seq_lens, block_tables)``."""
        raise NotImplementedError

    def wrapper(self, case, **kwargs) -> Any:
        """A planned ``BatchDecodeWithPagedKVCacheWrapper`` for ``case``."""
        raise NotImplementedError

    def plan(self, wrapper, case, **kwargs) -> None:
        raise NotImplementedError

    def run(self, wrapper, case, out=None, **kwargs) -> Any:
        raise NotImplementedError

    def reference(self, case, **kwargs) -> Any:
        raise NotImplementedError

    def check_close(self, out, ref) -> None:
        raise NotImplementedError

    def check_stock(self, out, ref) -> None:
        """The stock path (kill switch) against the reference, and the kernel no coarser."""
        raise NotImplementedError

    def lens(self, batch_size: int, *, seed: int) -> list:
        """Served-like KV lengths for ``batch_size`` requests."""
        raise NotImplementedError


def load_specs() -> Dict[str, KernelSpec]:
    """Every kernel spec of this package, by kernel name."""
    specs: Dict[str, KernelSpec] = {}
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda m: m.name):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        spec = module.SPEC
        assert spec.name == info.name, (spec.name, info.name)
        specs[spec.name] = spec
    return specs
