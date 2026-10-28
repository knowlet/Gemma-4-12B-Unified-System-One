"""CPU fakes verify whole-call MPS cleanup, fault latching, and runner wiring."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

import s1.evaluation.mps_maintenance as maintenance_module
from s1.evaluation.mps_maintenance import (
    MPS_CALL_MAINTENANCE_POLICY,
    MPSCallMaintenance,
    MPSMaintenanceBackend,
    MPSMaintenanceFault,
)

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "_run_ultrafast_mps_maintenance", ROOT / "scripts/run_ultrafast.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class Clock:
    now = 0

    def __call__(self):
        return self.now

    def advance(self):
        self.now += 1


class FakeMPS:
    def __init__(self, *, fail_at=None, clock=None):
        self.events = []
        self.fail_at = fail_at
        self.clock = clock
        self.trimmed = False

    def event(self, name):
        index = len(self.events)
        self.events.append(name)
        if self.clock is not None:
            self.clock.advance()
        if index == self.fail_at:
            raise RuntimeError("cleanup failed")

    def synchronize(self):
        self.event("synchronize")

    def empty_cache(self):
        self.event("empty_cache")
        self.trimmed = True

    def current_allocated_memory(self):
        self.event("current_allocated_memory")
        return 20

    def driver_allocated_memory(self):
        self.event("driver_allocated_memory")
        return 24 if self.trimmed else 80

    def recommended_max_memory(self):
        self.event("recommended_max_memory")
        return 100


def controller(*, mps=None, **kwargs):
    return MPSCallMaintenance(SimpleNamespace(mps=mps or FakeMPS()), **kwargs)


def backend(*, result=None):
    calls = []
    raw = result if result is not None else {"answers": {"q": {"raw": "unchanged"}}}

    def predict(request):
        calls.append(request)
        return raw

    return (
        SimpleNamespace(
            name="original backend name",
            model=SimpleNamespace(device="mps"),
            metadata={"device": "mps", "nested": {"identity": "original"}},
            question_mode="independent",
            predict=predict,
        ),
        calls,
        raw,
    )


def test_cleanup_runs_only_after_complete_call_and_keeps_response_identity():
    clock, mps = Clock(), FakeMPS()
    mps.clock = clock
    records, saved = [], []
    maintain = controller(
        mps=mps, records=records, save=lambda rows: saved.append(copy.deepcopy(rows))
    )
    original = {"answers": {"q": {"value": "unchanged"}}}

    def operation():
        assert not mps.events
        clock.advance()
        return original

    old_clock = maintenance_module.time.perf_counter
    maintenance_module.time.perf_counter = clock
    try:
        result = maintain.call(operation, profile="independent", kind="decision")
    finally:
        maintenance_module.time.perf_counter = old_clock
    assert result is original
    assert mps.events == [
        "synchronize",
        "current_allocated_memory",
        "driver_allocated_memory",
        "recommended_max_memory",
        "empty_cache",
        "synchronize",
        "current_allocated_memory",
        "driver_allocated_memory",
        "recommended_max_memory",
    ]
    assert records[0]["cleanup"] == {
        "before": {
            "current_allocated_bytes": 20,
            "driver_allocated_bytes": 80,
            "recommended_max_bytes": 100,
        },
        "after": {
            "current_allocated_bytes": 20,
            "driver_allocated_bytes": 24,
            "recommended_max_bytes": 100,
        },
        "error": None,
        "wall_ms": 9000,
    }
    assert records[0]["wall_ms_including_cleanup"] == 10000
    assert records[0]["profile"] == "independent" and records[0]["kind"] == "decision"
    assert saved == [records] and maintain.fault is None


@pytest.mark.parametrize("fail_at", range(9))
def test_each_cleanup_failure_preserves_success_and_blocks_every_later_mps_access(fail_at):
    mps = FakeMPS(fail_at=fail_at)
    maintain = controller(mps=mps)
    original, dispatches = {"raw": "answer"}, []
    assert (
        maintain.call(lambda: dispatches.append(1) or original, profile="causal", kind="decision")
        is original
    )
    events_after_failure = list(mps.events)
    assert maintain.records[0]["cleanup"]["error"]["message"] == "cleanup failed"
    assert len(maintain.operational_errors()) == 1
    with pytest.raises(MPSMaintenanceFault, match="no model call dispatched"):
        maintain.call(lambda: dispatches.append(2), profile="other", kind="native_helper")
    assert dispatches == [1] and mps.events == events_after_failure
    assert maintain.records[1]["dispatched"] is False
    assert maintain.records[1]["cleanup"] is None
    assert maintain.records[1]["prior_fault"]["stage"] == "cleanup"


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_original_inference_exception_survives_cleanup_in_finally(cleanup_fails):
    original = ValueError("original inference error")
    maintain = controller(mps=FakeMPS(fail_at=4 if cleanup_fails else None))

    def operation():
        raise original

    with pytest.raises(ValueError) as actual:
        maintain.call(operation, profile="independent", kind="decision")
    assert actual.value is original
    assert maintain.records[0]["inference_error"] == {
        "type": "ValueError",
        "message": "original inference error",
    }
    assert (maintain.records[0]["cleanup"]["error"] is not None) is cleanup_fails


def test_trace_write_failure_preserves_answer_and_latches_before_next_dispatch():
    def fail_save(records):
        raise OSError("trace storage unavailable")

    mps, raw = FakeMPS(), {"raw": "unchanged"}
    maintain = controller(mps=mps, save=fail_save)
    assert maintain.call(lambda: raw, profile="p", kind="decision") is raw
    events = list(mps.events)
    assert maintain.fault["stage"] == "trace_write"
    with pytest.raises(MPSMaintenanceFault):
        maintain.call(lambda: pytest.fail("no dispatch"), profile="p", kind="decision")
    assert mps.events == events


def test_transparent_backend_wrapper_preserves_identity_metadata_and_actual_raw_answer():
    original, calls, raw = backend()
    metadata = copy.deepcopy(original.metadata)
    maintain = controller()
    wrapped = MPSMaintenanceBackend(original, maintain, profile="independent")
    request = object()
    assert wrapped.predict(request) is raw and calls == [request]
    assert wrapped.name == original.name and wrapped.model is original.model
    assert wrapped.question_mode == original.question_mode
    assert wrapped.metadata == {**metadata, "mps_call_maintenance": MPS_CALL_MAINTENANCE_POLICY}
    assert original.metadata == metadata
    assert "raw" not in json.dumps(maintain.records)


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:0"])
def test_controller_and_wrapper_reject_non_mps_models_without_cleanup(device):
    mps = FakeMPS()
    with pytest.raises(ValueError, match="MPS model"):
        MPSCallMaintenance(SimpleNamespace(mps=mps), device=device)
    original, _, _ = backend()
    original.model.device = device
    with pytest.raises(ValueError, match="MPS model"):
        MPSMaintenanceBackend(original, controller(mps=mps), profile="p")
    assert not mps.events


def fake_helper_model(raw, torch, monkeypatch, clock):
    class Input:
        shape = (1, 2)

        def to(self, device):
            assert device == "mps"
            return self

    class Output:
        def __getitem__(self, key):
            assert key == (0, slice(2, None))
            return [90, 91]

    def generate(**kwargs):
        assert kwargs["max_new_tokens"] == 128 and kwargs["do_sample"] is False
        assert not torch.mps.events
        clock.advance()
        return Output()

    monkeypatch.setitem(sys.modules, "torch", torch)
    return SimpleNamespace(
        name="original model",
        device="mps",
        max_context=16384,
        processor=SimpleNamespace(
            apply_chat_template=lambda *args, **kwargs: "original prompt",
            __call__=lambda **kwargs: {"input_ids": Input()},
        ),
        lm=SimpleNamespace(generate=generate),
        tok=SimpleNamespace(decode=lambda *args, **kwargs: raw),
    ), Input


@pytest.mark.parametrize("valid", [False, True])
def test_native_helper_cleanup_is_whole_call_and_included_in_latency(valid, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(maintenance_module.time, "perf_counter", clock)
    mps = FakeMPS(clock=clock)
    torch = SimpleNamespace(mps=mps, inference_mode=nullcontext)
    raw = '```json\n{"text":"Graph theory"}\n```' if valid else "_" * 128
    model, Input = fake_helper_model(raw, torch, monkeypatch, clock)

    class Processor:
        def apply_chat_template(self, *args, **kwargs):
            return "original prompt"

        def __call__(self, **kwargs):
            return {"input_ids": Input()}

    model.processor = Processor()
    maintain, records = MPSCallMaintenance(torch), []
    helper = RUNNER.native_field_text(
        model, "original instruction", records, maintenance=maintain, profile="independent"
    )
    if valid:
        value, info = helper({"goal": "original context"})
        assert value == "Graph theory" and info["latency_ms"] == 10000
    else:
        with pytest.raises(ValueError):
            helper({"goal": "original context"})
    assert len(maintain.records) == 1 and maintain.records[0]["kind"] == "native_helper"
    assert maintain.records[0]["cleanup"]["wall_ms"] == 9000
    assert records[0]["raw_text"] == raw and records[0]["latency_ms"] == 10000
    assert records[0]["max_new_tokens"] == 128 and records[0]["do_sample"] is False
    assert records[0]["mps_call_maintenance"] == MPS_CALL_MAINTENANCE_POLICY


def question_payload():
    return {"q": {"type": "choice", "instructions": "choose", "criteria": {"a": "A", "b": "B"}}}


def test_both_jevbench_callbacks_dispatch_through_the_same_maintenance_wrapper(
    tmp_path, monkeypatch
):
    original, calls, raw = backend(result={"answers": {"q": {"raw": "original answer"}}})
    maintain = controller()
    wrapped = MPSMaintenanceBackend(original, maintain, profile="independent")
    seen = []

    def public(predict, root, output, identity):
        assert identity["runtime"]["mps_call_maintenance"] == MPS_CALL_MAINTENANCE_POLICY
        assert predict({"observed": "same state"}, question_payload()) is raw
        seen.append("public")

    def evaluate(callback, **kwargs):
        assert callback({"observed": "same state"}, question_payload()) is raw["answers"]
        seen.append("coherence")
        return SimpleNamespace(save=lambda path: seen.append("saved"))

    monkeypatch.setitem(sys.modules, "run_jevbench_public", SimpleNamespace(run_public=public))
    monkeypatch.setitem(
        sys.modules,
        "jevbench",
        SimpleNamespace(from_callable=lambda callback, **kwargs: callback, evaluate=evaluate),
    )
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(
        RUNNER.subprocess,
        "check_output",
        lambda argv, **kwargs: (
            "e18733694623aa93058e279c3534f8b8e2edefa3" if "rev-parse" in argv else ""
        ),
    )
    RUNNER.run_benchmarks(
        wrapped,
        "independent",
        tmp_path,
        SimpleNamespace(heaven_root=tmp_path / "heaven", coherence_root=tmp_path / "coherence"),
        {"manifest_sha256": "original hash"},
    )
    assert seen == ["public", "coherence", "saved"] and len(calls) == 2
    assert [(r["profile"], r["kind"]) for r in maintain.records] == [
        ("independent", "decision"),
        ("independent", "decision"),
    ]


def test_faulted_jevbench_callback_cannot_synchronize_before_the_dispatch_guard(
    tmp_path, monkeypatch
):
    mps = FakeMPS(fail_at=4)
    maintain = controller(mps=mps)
    original, calls, _ = backend()
    wrapped = MPSMaintenanceBackend(original, maintain, profile="p")
    wrapped.predict(object())
    previous = list(mps.events)

    def public(predict, *args):
        with pytest.raises(MPSMaintenanceFault):
            predict({}, question_payload())

    monkeypatch.setitem(sys.modules, "run_jevbench_public", SimpleNamespace(run_public=public))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(mps=mps))
    RUNNER.run_benchmarks(
        wrapped,
        "p",
        tmp_path,
        SimpleNamespace(heaven_root=tmp_path, coherence_root=None),
        {"manifest_sha256": "hash"},
    )
    assert len(calls) == 1 and mps.events == previous


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:0"])
def test_cli_rejects_maintenance_on_non_mps_before_upstream_or_model_loading(device, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ultrafast.py",
            "--upstream",
            "unused",
            "--output",
            "unused",
            "--mps-call-maintenance",
            "--device",
            device,
        ],
    )
    with pytest.raises(SystemExit) as actual:
        RUNNER.main()
    assert actual.value.code == 2


@pytest.mark.parametrize("mode", ["default_off", "enabled", "cleanup_fault"])
@pytest.mark.parametrize("oversized_policy", ["reject", "hierarchical"])
def test_campaign_preserves_goal_result_but_reports_resource_gate_and_archives_sources(
    tmp_path, monkeypatch, mode, oversized_policy
):
    output, manifest = tmp_path / "output", tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "scope": "fixture",
                "split_policy": "frozen",
                "tasks": [{"id": "one", "scenario": "test", "split": "development"}],
            }
        )
    )
    argv = [
        "run_ultrafast.py",
        "--upstream",
        str(tmp_path),
        "--output",
        str(output),
        "--manifest",
        str(manifest),
        "--profiles",
        "independent",
        "--repeats",
        "1",
    ]
    if mode != "default_off":
        argv.append("--mps-call-maintenance")
    if oversized_policy != "reject":
        argv.extend(["--oversized-head-policy", oversized_policy])
    monkeypatch.setattr(sys, "argv", argv)
    mps, calls = FakeMPS(fail_at=4 if mode == "cleanup_fault" else None), []
    torch = SimpleNamespace(
        mps=mps,
        get_num_threads=lambda: 1,
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    raw = {"answers": {"q": {"label": "a", "probabilities": {"a": 0.6, "b": 0.4}}}}

    class Backend:
        def __init__(self, *args, **kwargs):
            self.name = "same model name"
            self.metadata = {"device": "mps", "model": "same model"}
            self.model = SimpleNamespace(
                device="mps",
                backbone=SimpleNamespace(register_forward_hook=lambda *args, **kwargs: None),
            )

        def predict(self, request):
            calls.append(request)
            return raw

    class Transport:
        browser_version = "fake browser"

        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def set_output(self, path):
            path.mkdir(parents=True)

        def close_task(self):
            pass

    @contextmanager
    def fixture(*args):
        yield "http://local"

    modules = {
        "model": SimpleNamespace(),
        "browser": SimpleNamespace(),
        "agent": SimpleNamespace(field_text=object()),
    }

    def task(agent, selected, url, destination, **kwargs):
        result = modules["model"].post_json(
            "https://api.typesafe.ai/v1/systemone",
            None,
            {"model": "same", "state": {}, "questions": question_payload()},
        )
        assert result["answers"]["q"]["choice"] == "a"
        return {
            "task_id": selected["id"],
            "split": "development",
            "verified_success": True,
            "wall_ms": 10,
            "failure_class": None,
        }

    monkeypatch.setitem(
        sys.modules,
        "ultrafast_browser",
        SimpleNamespace(
            IsolatedCDP=Transport,
            fixture_server=fixture,
            load_upstream=lambda path: (modules, {}),
            run_task=task,
        ),
    )
    monkeypatch.setattr(RUNNER, "GemmaBackend", Backend)
    monkeypatch.setattr(RUNNER, "verify_checkpoint", lambda path: {"manifest_sha256": "verified"})
    monkeypatch.setattr(RUNNER, "run_benchmarks", lambda *args: None)
    monkeypatch.setattr(RUNNER.importlib.metadata, "version", lambda name: "test-version")
    monkeypatch.setattr(
        RUNNER.subprocess,
        "check_output",
        lambda argv, **kwargs: "64000" if argv[-1] == "hw.memsize" else "fake identity",
    )
    environment = [
        "PYTORCH_ENABLE_MPS_FALLBACK",
        "PYTORCH_MPS_FAST_MATH",
        "PYTORCH_MPS_PREFER_METAL",
        "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
        "PYTORCH_MPS_LOW_WATERMARK_RATIO",
    ]
    for name in environment:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PYTORCH_MPS_FAST_MATH", "0")
    assert RUNNER.main() == (1 if mode == "cleanup_fault" else 0)
    assert len(calls) == 1
    identity = json.loads((output / "identity.json").read_text())
    assert identity["oversized_head_policy"] == oversized_policy
    assert identity["hierarchical_head_policy"] == (
        RUNNER.HIERARCHICAL_HEAD_POLICY if oversized_policy == "hierarchical" else None
    )
    assert identity["mps_environment"] == {
        name: "0" if name == "PYTORCH_MPS_FAST_MATH" else None for name in environment
    }
    result = json.loads((output / "results.json").read_text())[0]
    result_keys = list(result)
    assert result_keys.index("backbone_forwards") == result_keys.index("backend_calls") + 1
    assert result_keys.index("text_calls") == result_keys.index("backbone_forwards") + 1
    assert "observed_completed_backbone_forwards" not in result
    assert result["verified_success"] is True
    original = json.loads((output / "independent/one-1/decision-transport.json").read_text())[0]
    assert original["raw_response"] == raw
    for relative in [
        "scripts/run_jevbench_public.py",
        "scripts/summarize_jevbench_campaign.py",
        "configs/benchmarks/jevbench-public.json",
    ]:
        assert (output / "sources" / relative).read_bytes() == (ROOT / relative).read_bytes()
        assert relative in identity["source_sha256"]
    if mode == "default_off":
        assert identity["mps_call_maintenance"] == {"enabled": False, "policy": None}
        assert not (output / "mps-call-maintenance.json").exists() and not mps.events
    else:
        failed = mode == "cleanup_fault"
        assert identity["mps_call_maintenance_resource_gate"]["passed"] is not failed
        assert bool(identity["operational_maintenance_errors"]) is failed
        assert identity["status"] == (
            "completed_with_operational_maintenance_errors" if failed else "completed"
        )
        assert identity["runtime"]["mps_call_maintenance"] == MPS_CALL_MAINTENANCE_POLICY
        assert len(json.loads((output / "mps-call-maintenance.json").read_text())) == 1
