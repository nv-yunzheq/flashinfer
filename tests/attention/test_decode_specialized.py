"""Tests of ``BatchDecodeWithPagedKVCacheWrapper(backend="specialized")`` and of
``backend="auto"`` taking it: the framework (registry, dispatch order, kill switch, CUDA-graph
freezes, counters) and every registered kernel, parametrized over the kernel specs of
``tests/attention/specialized_decode_kernels/`` (one module per kernel: builders, cases,
tolerances, kernel-internal checks).

Host checks run everywhere; the ``auto``-route comparison runs on every CUDA GPU (outside a
kernel's envelope ``auto`` must end exactly as with the specialized route switched off); the
kernel tests run where the kernel runs, the ``auto`` tests inside its measured envelope.
"""

from __future__ import annotations

import contextlib
import os
from types import MappingProxyType
from typing import Optional, Tuple

import pytest
import torch

from flashinfer.attention import specialized_decode as backend_pkg
from flashinfer.attention.specialized_decode import backend as backend_mod
from flashinfer.attention.specialized_decode import (
    KILL_SWITCH_ENV,
    DecodePlanCall,
    SpecializedDecode,
    SpecializedDecodeKernel,
    auto_kernel,
    plan_kernel,
    registered_kernels,
    specialized_decode_stats,
)
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

from tests.attention.specialized_decode_kernels import load_specs

SPECS = load_specs()
NAMES = sorted(SPECS)
AUTO_MAIN_MSG = (
    "only supports window_right != 0"  # how the auto route refuses a dense plan
)


@pytest.fixture
def spec(name):
    return SPECS[name]


def _cases(attr: str):
    return [
        pytest.param(n, c, id=f"{n}-{c}")
        for n in NAMES
        for c in getattr(SPECS[n], attr)
    ]


def _items(attr: str):
    out = []
    for n in NAMES:
        for c in getattr(SPECS[n], attr):
            cid = c[0] if isinstance(c, tuple) else c
            out.append(pytest.param(n, c, id=f"{n}-{cid}"))
    return out


def _needs_kernel(spec):
    if not spec.runs_here():
        pytest.skip(f"{spec.name} runs on compute capability {spec.compute_capability}")


def _needs_envelope(spec):
    if not spec.auto_here():
        pytest.skip(f"this GPU is outside {spec.name}'s auto envelope")


# ===================================================================== registry and host


def test_every_registered_kernel_has_a_spec():
    assert set(registered_kernels()) == set(SPECS)
    assert backend_pkg.discovery_errors() == {}


@pytest.mark.parametrize("name", NAMES)
def test_registration(spec):
    kernel = registered_kernels()[spec.name]
    assert kernel is spec.kernel_class
    assert issubclass(kernel, SpecializedDecodeKernel)
    assert kernel.name == spec.name
    assert spec.compute_capability in kernel.compute_capabilities


def test_registry_is_in_name_order():
    names = list(registered_kernels())
    assert names == sorted(names)


@pytest.mark.parametrize("name,case", _cases("refusal_cases"))
def test_plan_refusal(spec, case, monkeypatch):
    """Every condition of the kernel's case is named; semantics before the device."""
    overrides, reason = spec.refusal_cases[case]
    spec.pin_device(monkeypatch)
    refusal = spec.kernel_class.plan_refusal(spec.call(**overrides))
    assert refusal is not None and refusal[0] == reason


@pytest.mark.parametrize("name,case", _cases("auto_refusal_cases"))
def test_auto_refusal(spec, case, monkeypatch):
    """The measured envelope on top of the case, and what auto_kernel() makes of it."""
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    overrides, reason = (
        dict(spec.auto_refusal_cases[case][0]),
        spec.auto_refusal_cases[case][1],
    )
    spec.pin_device(monkeypatch, sms=overrides.pop("_sms", None))
    call = spec.call(**overrides)
    kernel = spec.kernel_class
    refusal = kernel.plan_refusal(call) or kernel.auto_refusal(call)
    if reason is None and refusal is not None and refusal[0] == "import":
        pytest.skip("the kernel's software stack is not installed here")
    assert (refusal and refusal[0]) == reason
    before = specialized_decode_stats()["kernels"][spec.name]["auto_declines"]
    chosen = auto_kernel(call)
    after = specialized_decode_stats()["kernels"][spec.name]["auto_declines"]
    assert chosen is (kernel if reason is None else None)
    if reason is not None and kernel.auto_candidate(call):
        assert after.get(reason, 0) == before.get(reason, 0) + 1


@pytest.mark.parametrize("name", NAMES)
def test_explicit_off_case_plan_raises_before_any_device_work(spec, monkeypatch):
    """An explicit plan no kernel serves raises the kernel's NotImplementedError (no silent
    fallback), counted by reason per kernel and in the totals."""
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    spec.pin_device(monkeypatch)
    overrides, reason = next(iter(spec.refusal_cases.values()))
    call = spec.call(**overrides)
    detail = spec.kernel_class.plan_refusal(call)[1]
    before = specialized_decode_stats()
    backend = SpecializedDecode(
        torch.empty(16, dtype=torch.uint8), use_cuda_graph=False, fixed_batch_size=0
    )
    if len(registered_kernels()) == 1:
        expected = spec.kernel_class.refusal_message(detail)
    else:
        expected = None
    with pytest.raises(NotImplementedError) as info:
        backend.plan(call, non_blocking=False)
    if expected is not None:
        assert str(info.value) == expected
    after = specialized_decode_stats()
    assert after["plan_refusals"][reason] == before["plan_refusals"].get(reason, 0) + 1
    per = after["kernels"][spec.name]["plan_refusals"]
    assert (
        per[reason] == before["kernels"][spec.name]["plan_refusals"].get(reason, 0) + 1
    )


def test_run_before_plan():
    backend = SpecializedDecode(
        torch.empty(16, dtype=torch.uint8), use_cuda_graph=False, fixed_batch_size=0
    )
    with pytest.raises(RuntimeError, match=r"plan\(\) must complete before run\(\)"):
        backend.run(torch.zeros(1), torch.zeros(1), torch.zeros(1), out=None)


@pytest.mark.parametrize("name", NAMES)
def test_kill_switch_is_read_once_at_the_backend(spec, monkeypatch):
    """With the kill switch set auto takes no kernel and asks none: the switch is the
    backend's, counted once as the backend's own decline, never a kernel's."""
    spec.pin_device(monkeypatch)
    call = spec.call()
    if not (spec.kernel_class.auto_candidate(call)):
        pytest.skip("the served call is not an auto candidate of this kernel")

    def never(*args, **kwargs):
        raise AssertionError("a kernel was asked under the kill switch")

    monkeypatch.setattr(spec.kernel_class, "plan_refusal", classmethod(never))
    monkeypatch.setattr(spec.kernel_class, "auto_refusal", classmethod(never))
    monkeypatch.setenv(KILL_SWITCH_ENV, "1")
    before = specialized_decode_stats()
    assert auto_kernel(call) is None
    after = specialized_decode_stats()
    assert (
        after["auto_declines"]["kill_switch"]
        == before["auto_declines"].get("kill_switch", 0) + 1
    )
    assert "kill_switch" not in after["kernels"][spec.name]["auto_declines"]


@pytest.mark.parametrize("name", NAMES)
def test_stats(spec):
    stats = specialized_decode_stats()
    assert list(stats) == [
        "plans",
        "fallback_plans",
        "plan_refusals",
        "auto_plans",
        "auto_declines",
        "compiled_variants",
        "dispatches",
        "fallback_runs",
        "kernels",
    ]
    per = stats["kernels"][spec.name]
    for key in ("plans", "fallback_plans", "auto_plans", "dispatches", "fallback_runs"):
        assert isinstance(per[key], int) and per[key] <= stats[key]
    for key in ("plan_refusals", "auto_declines"):
        assert all(stats[key].get(r, 0) >= n for r, n in per[key].items())


# --------------------------------------------------------- dispatch order (fake kernels)


def _fake_kernel(kernel_name: str, admits, *, log: list):
    class Fake(SpecializedDecodeKernel):
        name = kernel_name
        compute_capabilities = ((10, 3),)

        @classmethod
        def plan_refusal(cls, call):
            return None if admits(call) else ("batch_size", f"batch {call.batch_size}")

        @classmethod
        def refusal_message(cls, detail):
            return f"{kernel_name} serves other calls; this plan has {detail}."

        @classmethod
        def auto_candidate(cls, call):
            return True

        def plan(self, call, *, non_blocking):
            log.append((kernel_name, "plan"))

        def plan_stock(self, call, *, non_blocking):
            log.append((kernel_name, "stock"))

        def run(self, q, k_cache, v_cache, *, stock, **kwargs):
            log.append((kernel_name, "run", stock))
            return q

    return Fake


def _host_call(batch_size: int) -> DecodePlanCall:
    return DecodePlanCall(
        device=torch.device("cuda", 0),
        kv_layout="HND",
        batch_size=batch_size,
        num_qo_heads=16,
        num_kv_heads=1,
        head_dim=128,
        page_size=128,
        q_len_per_req=7,
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
        o_data_type=torch.float16,
        pos_encoding_mode="NONE",
        window_left=-1,
        window_right=-1,
        logits_soft_cap=0.0,
        sm_scale=None,
        fixed_split_size=-1,
        disable_split_kv=False,
        use_token_head_sf=False,
        custom_jit=False,
        kv_lens=torch.full((batch_size,), 300, dtype=torch.int32),
    )


def test_dispatch_order_first_admitting_kernel_wins(monkeypatch):
    """Kernels are asked in name order; the first that admits a call serves it (explicit
    and auto); a plan none admits lists every kernel's refusal; under CUDA graphs the kernel
    is part of the frozen plan, as the kill switch is."""
    from flashinfer.attention.specialized_decode import kernel as framework

    monkeypatch.setattr(framework, "device_capability", lambda index: (10, 3))
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    log: list = []
    a = _fake_kernel("a_small", lambda c: c.batch_size <= 4, log=log)
    b = _fake_kernel("b_any", lambda c: c.batch_size <= 8, log=log)
    monkeypatch.setattr(
        backend_mod,
        "registered_kernels",
        lambda: MappingProxyType({"a_small": a, "b_any": b}),
    )
    assert plan_kernel(_host_call(2)) is a and plan_kernel(_host_call(6)) is b
    assert auto_kernel(_host_call(2)) is a and auto_kernel(_host_call(6)) is b
    assert auto_kernel(_host_call(9)) is None
    with pytest.raises(NotImplementedError, match="serves 2 decode cases") as info:
        plan_kernel(_host_call(9))
    assert "[a_small]" in str(info.value) and "[b_any]" in str(info.value)

    ws = torch.empty(16, dtype=torch.uint8)
    backend = SpecializedDecode(ws, use_cuda_graph=True, fixed_batch_size=4)
    backend.plan(_host_call(2), non_blocking=False)
    assert backend.kernel_name == "a_small" and not backend.uses_fallback
    backend.run(torch.zeros(1), torch.zeros(1), torch.zeros(1))
    monkeypatch.setattr(
        backend_mod, "registered_kernels", lambda: MappingProxyType({"b_any": b})
    )
    with pytest.raises(ValueError, match="specialized kernel is part of the frozen"):
        backend.plan(_host_call(2), non_blocking=False)
    monkeypatch.setattr(
        backend_mod,
        "registered_kernels",
        lambda: MappingProxyType({"a_small": a, "b_any": b}),
    )
    monkeypatch.setenv(KILL_SWITCH_ENV, "1")
    with pytest.raises(ValueError, match=f"{KILL_SWITCH_ENV} is part of the frozen"):
        backend.plan(_host_call(2), non_blocking=False)
    eager = SpecializedDecode(ws, use_cuda_graph=False, fixed_batch_size=0)
    eager.plan(_host_call(6), non_blocking=False)  # kill switch: b's stock path
    assert eager.uses_fallback and eager.kernel_name == "b_any"
    eager.run(torch.zeros(1), torch.zeros(1), torch.zeros(1))
    assert log == [
        ("a_small", "plan"),
        ("a_small", "run", False),
        ("b_any", "stock"),
        ("b_any", "run", True),
    ]


# ============================================================ auto route, every CUDA GPU


def _fp16_plan_run(q_len: int, *, use_tensor_cores: bool):
    """An ordinary fp16 decode on auto's own route (16 q heads over 1 K/V head, page 128)."""

    def go():
        gen = torch.Generator(device="cuda").manual_seed(7)
        batch, kv_len, page = 4, 1000, 128
        per = (kv_len + page - 1) // page
        q = torch.randn((batch * q_len, 16, 128), generator=gen, device="cuda").half()
        k = torch.randn(
            (batch * per + 1, 1, page, 128), generator=gen, device="cuda"
        ).half()
        v = torch.randn(k.shape, generator=gen, device="cuda").half()
        indptr = torch.arange(batch + 1, dtype=torch.int32, device="cuda") * per
        indices = torch.arange(1, batch * per + 1, dtype=torch.int32, device="cuda")
        last = torch.full(
            (batch,), kv_len - (per - 1) * page, dtype=torch.int32, device="cuda"
        )
        w = BatchDecodeWithPagedKVCacheWrapper(
            torch.zeros(128 << 20, dtype=torch.uint8, device="cuda"),
            "HND",
            use_tensor_cores=use_tensor_cores,
        )
        w.plan(
            indptr,
            indices,
            last,
            16,
            1,
            128,
            page,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
            q_len_per_req=q_len,
        )
        return w, w.run(q, (k, v))

    return go


_ORDINARY = {
    "single_token_fp16": _fp16_plan_run(1, use_tensor_cores=False),
    "causal_q7_fp16": _fp16_plan_run(7, use_tensor_cores=True),
}


def _outcome(go) -> Tuple[tuple, Optional[torch.Tensor]]:
    torch.cuda.synchronize()
    try:
        w, out = go()
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001  (the outcome is compared, whatever it is)
        return ("raises", type(exc).__name__, str(exc)), None
    return ("ok", w.resolved_backend), out


def _route_params():
    out = [pytest.param(None, c, id=f"ordinary-{c}") for c in sorted(_ORDINARY)]
    out += [
        pytest.param(n, c, id=f"{n}-{c}")
        for n in NAMES
        for c in sorted(SPECS[n].auto_route_cases)
    ]
    return out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
@pytest.mark.parametrize("kill_switch", (False, True), ids=("", "kill_switch"))
@pytest.mark.parametrize("name,case", _route_params())
def test_auto_route_unchanged_outside_envelope(name, case, kill_switch, monkeypatch):
    """Outside every kernel's envelope (and on GPUs no kernel runs on), a plan + run of
    backend="auto" ends exactly as with the specialized route switched off: the same
    exception and message, or the same resolved backend and output. Nothing reaches the
    specialized backend."""
    spec_ = SPECS[name] if name else None
    if (
        spec_ is not None
        and case == spec_.in_case
        and spec_.auto_here()
        and not kill_switch
    ):
        pytest.skip(
            "inside the envelope: auto takes the kernel (test_auto_selects_kernel)"
        )
    if kill_switch:
        monkeypatch.setenv(KILL_SWITCH_ENV, "1")
    else:
        monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    go = spec_.auto_route_cases[case] if spec_ else _ORDINARY[case]
    with monkeypatch.context() as m:
        m.setattr(
            BatchDecodeWithPagedKVCacheWrapper,
            "_auto_takes_specialized",
            lambda self, *args, **kwargs: None,
        )
        expected, expected_out = _outcome(go)
    before = specialized_decode_stats()
    got, out = _outcome(go)
    after = specialized_decode_stats()
    assert got == expected
    if out is not None:
        torch.testing.assert_close(out, expected_out, rtol=1e-3, atol=1e-3)
    for key in ("plans", "auto_plans", "fallback_plans", "dispatches", "fallback_runs"):
        assert after[key] == before[key], key
    if spec_ is not None and (case.startswith("dense") or case == spec_.in_case):
        assert got[0] == "raises" and AUTO_MAIN_MSG in got[2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
@pytest.mark.parametrize("name", NAMES)
def test_refuses_other_architectures(spec, monkeypatch):
    """Explicit backend: a plan on a GPU the kernel does not run on raises instead of running
    something else."""
    if torch.cuda.get_device_capability() in spec.kernel_class.compute_capabilities:
        pytest.skip("covers GPUs other than the kernel's")
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    detail = spec.kernel_class.plan_refusal(spec.call())
    assert detail is not None and detail[0] == "arch"
    c = spec.case([300, 4232], seed=5)
    with pytest.raises(NotImplementedError) as info:
        spec.wrapper(c)
    assert str(info.value) == spec.kernel_class.refusal_message(detail[1])


# ======================================================= on the kernel's device: explicit


@pytest.mark.parametrize("name", NAMES)
def test_explicit_backend_reports(spec, monkeypatch):
    _needs_kernel(spec)
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    w = spec.wrapper(spec.case(spec.lens(2, seed=3), seed=3))
    assert w.resolved_backend == "specialized"
    assert w._specialized_decode.kernel_name == spec.name
    assert not w._specialized_decode.uses_fallback


@pytest.mark.parametrize("name,case", _items("numerics_cases"))
def test_matches_reference_and_stock(spec, case, monkeypatch):
    """The kernel vs an fp32 reference and vs its stock path (kill switch), counted as one
    dispatch / one stock run; an internally allocated output gives the same bits."""
    _needs_kernel(spec)
    params = dict(case[1])
    batch_size, table = params.pop("batch_size"), params.pop("table")
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    c = spec.case(
        spec.lens(batch_size, seed=batch_size), seed=100 + batch_size, **params
    )
    ref = spec.reference(c)
    w = spec.wrapper(c, table=table)
    before = specialized_decode_stats()
    out = torch.full(c[0].shape, float("nan"), dtype=torch.float16, device="cuda")
    result = spec.run(w, c, out)
    assert result.data_ptr() == out.data_ptr()
    after = specialized_decode_stats()
    assert after["dispatches"] == before["dispatches"] + 1
    assert (
        after["kernels"][spec.name]["dispatches"]
        == before["kernels"][spec.name]["dispatches"] + 1
    )
    spec.check_close(out, ref)
    torch.testing.assert_close(spec.run(w, c), out, rtol=0, atol=0)

    monkeypatch.setenv(KILL_SWITCH_ENV, "1")
    stock = spec.wrapper(c, table=table)
    assert stock._specialized_decode.uses_fallback
    runs = specialized_decode_stats()["fallback_runs"]
    stock_out = spec.run(stock, c, torch.full_like(out, float("nan")))
    assert specialized_decode_stats()["fallback_runs"] == runs + 1
    spec.check_stock(stock_out, ref, kernel_out=out)


def _refresh(spec, static, seed):
    """New lengths, pages and block tables written into the static capture's buffers."""
    q, k_cache, v_cache, _, _ = static
    fresh = spec.case(spec.lens(static[3].shape[0], seed=seed), seed=seed)
    n_pages = min(fresh[1].shape[0], k_cache.shape[0])
    q.copy_(fresh[0])
    k_cache[:n_pages].copy_(fresh[1][:n_pages])
    v_cache[:n_pages].copy_(fresh[2][:n_pages])
    return (q, k_cache, v_cache, fresh[3], fresh[4].clamp(max=n_pages - 1))


def _capture(spec, w, case_, out):
    spec.run(w, case_, out)  # warm-up, outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        spec.run(w, case_, out)
    return graph


@pytest.mark.parametrize("name,batch_size", _cases("graph_batches"))
def test_explicit_cuda_graph_replay(spec, batch_size, monkeypatch):
    """Captured after plan; re-planned (outside the capture) with new lengths, pages and
    block tables, then replayed: the owned metadata buffers follow the latest plan. The
    kill switch at plan time bakes the stock path into its own capture and is frozen."""
    _needs_kernel(spec)
    static = spec.case([max(spec.lens(1, seed=0))] * batch_size, seed=21)
    out = torch.empty(static[0].shape, dtype=torch.float16, device="cuda")
    wrappers, graphs = {}, {}
    for arm in ("kernel", "stock"):
        if arm == "stock":
            monkeypatch.setenv(KILL_SWITCH_ENV, "1")
        else:
            monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
        wrappers[arm] = spec.wrapper(static, graph=True, table=(arm == "kernel"))
        graphs[arm] = _capture(spec, wrappers[arm], static, out)
    for seed in (31, 32):
        c = _refresh(spec, static, seed)
        ref = spec.reference(c)
        results = {}
        for arm, graph in graphs.items():
            if arm == "stock":
                monkeypatch.setenv(KILL_SWITCH_ENV, "1")
            else:
                monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
            spec.plan(wrappers[arm], c, table=(arm == "kernel"))
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            results[arm] = out.clone()
        spec.check_close(results["kernel"], ref)
        spec.check_stock(results["stock"], ref)
    monkeypatch.setenv(KILL_SWITCH_ENV, "1")
    with pytest.raises(ValueError, match="frozen cudagraph plan"):
        spec.plan(wrappers["kernel"], static)


@pytest.mark.parametrize("name,case", _cases("off_case_plans"))
def test_explicit_off_case_plans_raise(spec, case, monkeypatch):
    _needs_kernel(spec)
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    plan_kwargs, match = spec.off_case_plans[case]
    with pytest.raises(NotImplementedError, match=match):
        spec.wrapper(spec.case([300, 4232, 8, 1000], seed=41), plan_kwargs=plan_kwargs)


@pytest.mark.parametrize("name,case", _cases("off_case_runs"))
def test_off_case_runs_raise(spec, case, monkeypatch):
    _needs_kernel(spec)
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    run_kwargs, match = spec.off_case_runs[case]
    c = spec.case([300, 4232], seed=43)
    w = spec.wrapper(c)
    run_kwargs = {
        k: (v.cuda() if torch.is_tensor(v) else v) for k, v in run_kwargs.items()
    }
    with pytest.raises(NotImplementedError, match=match):
        spec.run(w, c, **run_kwargs)


# =========================================== inside the measured envelope: backend="auto"


@pytest.mark.parametrize("name,case", _items("auto_cases"))
def test_auto_selects_kernel(spec, case, monkeypatch):
    """Inside the envelope auto plans the call on the kernel: the same bits as the explicit
    backend, counted as an auto plan and a dispatch, per kernel and in the totals."""
    _needs_envelope(spec)
    params = dict(case[1])
    batch_size = params.pop("batch_size")
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    c = spec.case(spec.lens(batch_size, seed=500 + batch_size), seed=600 + batch_size)
    before = specialized_decode_stats()
    w = spec.wrapper(c, backend="auto", **params)
    mid = specialized_decode_stats()
    assert w.resolved_backend == "specialized"
    assert w._specialized_decode.kernel_name == spec.name
    assert mid["auto_plans"] == before["auto_plans"] + 1
    assert mid["plans"] == before["plans"] + 1
    k_before, k_mid = before["kernels"][spec.name], mid["kernels"][spec.name]
    assert k_mid["auto_plans"] == k_before["auto_plans"] + 1
    out = spec.run(w, c)
    assert specialized_decode_stats()["dispatches"] == mid["dispatches"] + 1
    explicit = spec.run(spec.wrapper(c, table=params.get("table", True)), c)
    torch.testing.assert_close(out, explicit, rtol=0, atol=0)
    spec.check_close(out, spec.reference(c))


@pytest.mark.parametrize("name", NAMES)
def test_auto_kill_switch(spec, monkeypatch):
    """Under the kill switch auto keeps its own route (which refuses the kernel's dense
    case, as before); an explicit backend="specialized" runs the stock path."""
    _needs_envelope(spec)
    c = spec.case([300, 4232, 8, 1000], seed=71)
    monkeypatch.setenv(KILL_SWITCH_ENV, "1")
    before = specialized_decode_stats()
    with pytest.raises(NotImplementedError, match=AUTO_MAIN_MSG):
        spec.wrapper(c, backend="auto")
    after = specialized_decode_stats()
    assert after["auto_plans"] == before["auto_plans"]
    assert (
        after["auto_declines"]["kill_switch"]
        == before["auto_declines"].get("kill_switch", 0) + 1
    )
    assert spec.wrapper(c)._specialized_decode.uses_fallback
    monkeypatch.delenv(KILL_SWITCH_ENV)
    assert spec.wrapper(c, backend="auto").resolved_backend == "specialized"


@pytest.mark.parametrize("name,short", _cases("short_rows"))
def test_auto_padded_rows(spec, short, monkeypatch):
    """A row shorter than q_len_per_req (a padded CUDA-graph row has seq_len 0): auto
    declines it and keeps its own route, refusing the dense call as before; the explicit
    backend raises like every multi-token plan of the wrapper. On a CUDA-graph wrapper
    frozen on the kernel the padded plan is refused the same way and leaves the capture
    intact: a valid re-plan replays correctly."""
    _needs_envelope(spec)
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    padded = spec.case([short, 300, 4232, 1000], seed=81)
    before = specialized_decode_stats()["kernels"][spec.name]["auto_declines"].get(
        "kv_len", 0
    )
    with pytest.raises(NotImplementedError, match=AUTO_MAIN_MSG):
        spec.wrapper(padded, backend="auto")
    assert (
        specialized_decode_stats()["kernels"][spec.name]["auto_declines"]["kv_len"]
        == before + 1
    )
    with pytest.raises(ValueError, match="kv_len >= q_len_per_req"):
        spec.wrapper(padded)

    static = spec.case([max(spec.lens(1, seed=0))] * 4, seed=82)
    out = torch.empty(static[0].shape, dtype=torch.float16, device="cuda")
    w = spec.wrapper(static, graph=True, backend="auto")
    assert w.resolved_backend == "specialized"
    graph = _capture(spec, w, static, out)
    with pytest.raises(NotImplementedError, match=AUTO_MAIN_MSG):
        spec.plan(w, static[:3] + padded[3:])
    fresh = spec.case([7, 300, 4232, 129], seed=83)
    n_pages = min(fresh[1].shape[0], static[1].shape[0])
    static[1][:n_pages].copy_(fresh[1][:n_pages])
    static[2][:n_pages].copy_(fresh[2][:n_pages])
    c = (static[0], static[1], static[2], fresh[3], fresh[4].clamp(max=n_pages - 1))
    spec.plan(w, c)
    assert w.resolved_backend == "specialized"
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    spec.check_close(out, spec.reference(c))


@pytest.mark.parametrize("table", (True, False), ids=("block_tables", "csr"))
@pytest.mark.parametrize("name,batch_size", _cases("auto_graph_batches"))
def test_auto_cuda_graph_replay(spec, batch_size, table, monkeypatch):
    """auto on a CUDA-graph wrapper: planned on the kernel, captured, re-planned with new
    lengths, pages and tables, replayed: the same bits as the explicit backend's capture.
    The route is frozen at plan time: a causal plan on it raises, and a wrapper first
    planned on auto's own route stays there for the kernel's case."""
    _needs_envelope(spec)
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    static = spec.case([max(spec.lens(1, seed=0))] * batch_size, seed=91)
    outs, wrappers, graphs = {}, {}, {}
    for backend in ("auto", "specialized"):
        outs[backend] = torch.empty(static[0].shape, dtype=torch.float16, device="cuda")
        wrappers[backend] = spec.wrapper(
            static, graph=True, table=table, backend=backend
        )
        assert wrappers[backend].resolved_backend == "specialized"
        graphs[backend] = _capture(spec, wrappers[backend], static, outs[backend])
    for seed in (92, 93):
        c = _refresh(spec, static, seed)
        dispatches = specialized_decode_stats()["dispatches"]
        for backend in ("auto", "specialized"):
            spec.plan(wrappers[backend], c, table=table)
            outs[backend].fill_(float("nan"))
            graphs[backend].replay()
        torch.cuda.synchronize()
        # replays are not counted: no launch went through run()
        assert specialized_decode_stats()["dispatches"] == dispatches
        torch.testing.assert_close(outs["auto"], outs["specialized"], rtol=0, atol=0)
        spec.check_close(outs["auto"], spec.reference(c))
    with pytest.raises(ValueError, match="frozen cudagraph shape"):
        spec.plan(wrappers["auto"], static, window_right=0)

    rows = static[3].shape[0]
    stock_first = BatchDecodeWithPagedKVCacheWrapper(
        torch.zeros(128 << 20, dtype=torch.uint8, device="cuda"),
        "HND",
        use_cuda_graph=True,
        use_tensor_cores=True,
        paged_kv_indptr_buffer=torch.empty(rows + 1, dtype=torch.int32, device="cuda"),
        paged_kv_indices_buffer=torch.empty(
            rows * static[4].shape[1], dtype=torch.int32, device="cuda"
        ),
        paged_kv_last_page_len_buffer=torch.empty(
            rows, dtype=torch.int32, device="cuda"
        ),
    )
    # auto's own route for the causal call (whether it serves it or not)
    with contextlib.suppress(Exception):
        spec.plan(stock_first, static, window_right=0)
    with pytest.raises(NotImplementedError, match=AUTO_MAIN_MSG):
        spec.plan(stock_first, static)


# ================================================================ kernel-internal checks


def _check_params():
    out = []
    for n in NAMES:
        for check in SPECS[n].checks:
            for pid, params in check.params:
                cid = f"{n}-{check.name}" + (f"-{pid}" if pid else "")
                out.append(pytest.param(n, check, params, id=cid))
    return out


@pytest.mark.parametrize("name,check,params", _check_params())
def test_kernel_check(spec, check, params, monkeypatch):
    if check.needs_kernel:
        _needs_kernel(spec)
    check.fn(spec, monkeypatch, **params)


def test_kill_switch_env_name():
    assert KILL_SWITCH_ENV == "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"
    assert os.environ.get("FLASHINFER_SPECIALIZED_KERNEL_DISABLE", "") in ("", "0", "1")
