"""Pure CPU orchestration fakes; no Torch, real models, processes or GPU calls."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from s1.backends import normalize_response
from s1.contracts import DecisionRequest, answer_from_probabilities

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_mps_maintenance", ROOT / "scripts/benchmark_mps_maintenance.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
IDENTITY = "Wed Oct 28 20:34:39 2026 S original-command"


@pytest.fixture
def requests():
    return [
        DecisionRequest.model_validate(
            {
                "state": {"decision": index},
                "questions": [
                    {"id": name, "instructions": "Choose", "criteria": {"a": "A", "b": "B"}}
                    for name in ("operation", "target")
                ],
            }
        )
        for index in RUNNER.DECISIONS
    ]


def response(request, p=0.75):
    return {
        "answers": {
            question.id: answer_from_probabilities(question, [p, 1 - p])
            for question in request.questions
        }
    }


class Backbone:
    def __init__(self):
        self.hook = None
        self.forwards = 0

    def register_forward_pre_hook(self, callback, *, with_kwargs):
        assert with_kwargs and self.hook is None
        self.hook = callback
        return SimpleNamespace(remove=lambda: setattr(self, "hook", None))

    def forward(self, length, *, cache=False):
        kwargs = {"input_ids": SimpleNamespace(shape=(2, length)), "use_cache": cache}
        self.hook(self, (), kwargs)
        self.forwards += 1


def fake_control(*, fail_trim=(), driver=40):
    state = {"events": [], "trims": 0, "driver": driver, "tick": 0}

    def clock():
        state["tick"] += 1
        return state["tick"] / 1000

    def snapshot():
        state["events"].append("snapshot")
        return {
            "current_allocated_bytes": 20,
            "driver_allocated_bytes": state["driver"],
            "recommended_max_bytes": 100,
        }

    def trim():
        state["trims"] += 1
        state["events"].append("trim")
        if state["trims"] in fail_trim:
            raise RuntimeError("cleanup failed")

    control = RUNNER.Controller(lambda: state["events"].append("sync"), snapshot, trim, clock)
    return control, state


def execute(requests, *, fail_call=None, fail_trim=(), driver=40, mutate=None, save=None):
    control, state = fake_control(fail_trim=fail_trim, driver=driver)
    backbone = Backbone()
    model, report = SimpleNamespace(backbone=backbone), {}
    calls = []

    def operation(request):
        calls.append(request.state["decision"])
        if fail_call == len(calls):
            raise RuntimeError("original inference failure")
        backbone.forward(10 + request.state["decision"])
        result = response(request)
        return mutate(result, len(calls)) if mutate else result

    return SimpleNamespace(
        control=control,
        state=state,
        backbone=backbone,
        report=report,
        calls=calls,
        run=lambda: RUNNER.run_schedule(
            model, requests, operation, normalize_response, control, report, save or (lambda: None)
        ),
    )


def test_abba_keeps_every_cold_chronological_sample_and_all_cleanup_in_timing(requests):
    job = execute(requests)
    job.run()
    assert job.calls == [1, 2, 3] * 4
    assert [row["policy"] for row in job.report["blocks"]] == ["A", "B", "B", "A"]
    assert len(job.report["samples"]) == 12 and job.state["trims"] == 15
    assert job.report["summary"]["all_numerical_and_shape_gates_passed"]
    assert job.report["summary"]["cold_samples_discarded"] == 0
    assert all(row["pairwise_comparisons"] == 6 for row in job.report["summary"]["requests"])
    for row in job.report["samples"]:
        assert row["pre_dispatch_guard"]["passed"]
        assert row["forwards"][0]["guard"]["passed"]
        assert row["forwards"][0]["use_cache"] is False
        assert row["wall_ms_including_call_cleanup"] >= row["cleanup"]["wall_ms"]
        assert row["raw_response"] == response(requests[row["decision_index"] - 1])
        assert row["cold_after_clear"] == (row["policy"] == "A" or row["decision_index"] == 1)
    for block in job.report["blocks"]:
        samples = [job.report["samples"][i] for i in block["samples"]]
        assert (
            block["wall_ms_including_boundary_cleanup_and_trace_io"]
            >= sum(row["wall_ms_including_call_cleanup"] for row in samples)
            + block["start_cleanup"]["wall_ms"]
            + block["end_cleanup"]["wall_ms"]
        )
    assert (
        job.report["schedule_wall_ms_including_all_cleanup_and_trace_io"]
        >= sum(
            row["wall_ms_including_boundary_cleanup_and_trace_io"] for row in job.report["blocks"]
        )
        + job.report["final_cleanup"]["wall_ms"]
    )
    assert job.backbone.hook is None


def test_gate_uses_all_repeats_and_fixed_point_zero_one_with_argmax(requests):
    job = execute(requests)
    job.run()
    rows = job.report["samples"]
    rows[-1]["raw_response"] = response(requests[2], 0.735)
    summary = RUNNER.summarize(requests, rows, normalize_response)
    assert summary["requests"][2]["max_probability_delta"] == pytest.approx(0.015)
    assert summary["requests"][2]["all_argmax_agree"]
    assert not summary["all_numerical_and_shape_gates_passed"]
    rows[-1]["raw_response"] = response(requests[2], 0.25)
    assert not RUNNER.summarize(requests, rows, normalize_response)["requests"][2][
        "all_argmax_agree"
    ]
    assert RUNNER.MAX_PROBABILITY_DELTA == 0.01 and RUNNER.MAX_DRIVER_FRACTION == 0.8
    with pytest.raises(ValueError, match="all twelve"):
        RUNNER.summarize(requests, rows[:-1], normalize_response)
    rows[-1]["error"] = {"message": "failure must stay in population"}
    with pytest.raises(ValueError, match="failed/invalid"):
        RUNNER.summarize(requests, rows, normalize_response)


def test_shape_difference_cannot_pass_even_when_probabilities_match(requests):
    job = execute(requests)
    job.run()
    job.report["samples"][-1]["forwards"][0]["input_shape"] = [1, 99]
    summary = RUNNER.summarize(requests, job.report["samples"], normalize_response)
    assert summary["requests"][2]["max_probability_delta"] == 0
    assert not summary["all_numerical_and_shape_gates_passed"]


@pytest.mark.parametrize("failed_call", [1, 2, 4])
def test_inference_failure_latches_no_next_call_and_always_final_cleanup(requests, failed_call):
    job = execute(requests, fail_call=failed_call)
    with pytest.raises(RUNNER.MaintenanceStop, match="original inference failure"):
        job.run()
    assert len(job.calls) == len(job.report["samples"]) == failed_call
    assert job.report["samples"][-1]["error"]["error"]["message"] == "original inference failure"
    assert job.report["final_cleanup"]["scope"] == "experiment_final"
    assert job.report["blocks"][-1]["end_cleanup"]["scope"] == "block_end"
    assert job.backbone.hook is None and "summary" not in job.report
    before = list(job.state["events"])
    with pytest.raises(RUNNER.MaintenanceStop):
        job.control.guard({}, "queued")
    assert job.state["events"] == before


def test_cleanup_fault_preserves_raw_success_and_blocks_next_dispatch(requests):
    job = execute(requests, fail_trim={2})
    with pytest.raises(RUNNER.MaintenanceStop, match="cleanup failed"):
        job.run()
    assert job.calls == [1] and job.state["trims"] == 4
    row = job.report["samples"][0]
    assert row["raw_response"] == response(requests[0])
    assert row["cleanup"]["errors"][0]["operation"] == "empty_cache"
    assert job.report["fault"]["scope"] == "after_call"


def test_cleanup_errors_cannot_replace_first_inference_failure(requests):
    job = execute(requests, fail_call=1, fail_trim={2, 3, 4})
    with pytest.raises(RUNNER.MaintenanceStop, match="original inference failure"):
        job.run()
    assert job.report["fault"]["error"]["message"] == "original inference failure"
    assert job.report["samples"][0]["cleanup"]["errors"]
    assert job.report["final_cleanup"]["errors"]
    assert job.state["trims"] == 4


def test_final_cleanup_failure_keeps_all_twelve_results_and_blocks_success(requests):
    job = execute(requests, fail_trim={15})
    with pytest.raises(RUNNER.MaintenanceStop, match="cleanup failed"):
        job.run()
    assert len(job.report["samples"]) == 12 and len(job.calls) == 12
    assert job.report["fault"]["scope"] == "experiment_final"
    assert job.report["final_cleanup"]["errors"]


def test_receipt_write_failure_latches_and_still_attempts_final_cleanup(requests):
    job = execute(requests)

    def save():
        if job.report.get("samples"):
            raise OSError("receipt write failed")

    with pytest.raises(RUNNER.MaintenanceStop, match="receipt write failed"):
        RUNNER.run_schedule(
            SimpleNamespace(backbone=job.backbone),
            requests,
            lambda request: (job.backbone.forward(11), response(request))[1],
            normalize_response,
            job.control,
            job.report,
            save,
        )
    assert len(job.report["samples"]) == 1
    assert job.report["fault"]["scope"] == "receipt_write"
    assert job.report["final_cleanup"] and job.backbone.hook is None


def test_inference_must_not_mutate_original_request(requests):
    job = execute(requests)

    def operation(request):
        job.backbone.forward(11)
        request.state["decision"] = 99
        return response(request)

    with pytest.raises(RUNNER.MaintenanceStop, match="mutated its original request"):
        RUNNER.run_schedule(
            SimpleNamespace(backbone=job.backbone),
            requests,
            operation,
            normalize_response,
            job.control,
            job.report,
            lambda: None,
        )
    assert len(job.report["samples"]) == 1


def test_driver_guard_blocks_prediction_and_latches_before_future_observations(requests):
    job = execute(requests, driver=80)
    with pytest.raises(RUNNER.MaintenanceStop) as stop:
        job.run()
    assert stop.value.evidence["kind"] == "resource_guard"
    assert job.calls == [] and job.backbone.forwards == 0
    assert len(job.report["samples"]) == 1
    assert job.report["samples"][0]["pre_dispatch_guard"]["memory"]["driver_allocated_bytes"] == 80
    assert job.report["final_cleanup"]


def test_driver_guard_is_rechecked_before_each_native_forward(requests):
    job = execute(requests)
    original = job.control.snapshot

    def snapshot():
        value = original()
        if job.calls:
            value["driver_allocated_bytes"] = 80
        return value

    job.control.snapshot = snapshot
    with pytest.raises(RUNNER.MaintenanceStop) as stop:
        job.run()
    assert stop.value.evidence["scope"] == "pre_forward"
    assert job.calls == [1] and job.backbone.forwards == 0


def test_real_cleanup_error_promotes_resource_stop_to_error(requests):
    job = execute(requests, driver=80, fail_trim={2})
    with pytest.raises(RUNNER.MaintenanceStop, match="cleanup failed") as stop:
        job.run()
    assert stop.value.evidence["kind"] == "error"
    assert stop.value.evidence["prior_resource_guard"]["kind"] == "resource_guard"
    assert job.calls == [] and job.report["final_cleanup"]


def test_observation_failure_still_attempts_trim_and_never_predicts(requests):
    job = execute(requests)
    job.control.snapshot = lambda: {
        "current_allocated_bytes": 20,
        "driver_allocated_bytes": None,
        "recommended_max_bytes": 100,
    }
    with pytest.raises(RUNNER.MaintenanceStop, match="complete nonnegative"):
        job.run()
    assert job.calls == [] and job.state["trims"] == 3
    assert len(job.report["final_cleanup"]["errors"]) == 2


def test_invalid_probabilities_preserve_raw_output_without_retry(requests):
    def invalid(raw, _index):
        raw["answers"]["operation"]["probabilities"] = {"a": 0.2, "b": 0.2}
        return raw

    job = execute(requests, mutate=invalid)
    with pytest.raises(RUNNER.MaintenanceStop, match="invalid backend probability"):
        job.run()
    assert job.calls == [1]
    assert job.report["samples"][0]["raw_response"]["answers"]["operation"]["probabilities"] == {
        "a": 0.2,
        "b": 0.2,
    }


def wait_setup(tmp_path, monkeypatch):
    path = tmp_path / "supervisor.json"
    initial = {
        "pid": 22,
        "created_at_utc": "original epoch",
        "status": "running",
        "active_child": {"pid": 33, "identity": IDENTITY},
        "epochs": [{"pid": 33, "identity": IDENTITY, "exit_code": None}],
    }
    RUNNER.write_json(path, initial)
    final = copy.deepcopy(initial)
    final.update(status="completed", active_child=None)
    final["epochs"][0].update(exit_code=0, finished_at_utc="terminal", status="completed")
    events = []

    def capture(pid):
        events.append(("capture", pid))
        return {
            "pid": pid,
            "identity": f"Wed Oct 28 20:34:39 2026 S python supervisor.py --output {tmp_path}"
            if pid == 22
            else IDENTITY,
        }

    def wait(handle, save, seconds):
        events.append(("wait", handle["pid"]))
        if handle["pid"] == 22:
            RUNNER.write_json(path, final)
        handle["status"] = "original_exited"

    observer = SimpleNamespace(
        capture_handle=capture,
        wait_handle=wait,
        stable_identity=lambda value: (
            (value[:24], value[24:].split(maxsplit=1)[1])
            if isinstance(value, str) and value
            else (_ for _ in ()).throw(ValueError("invalid identity"))
        ),
    )
    monkeypatch.setattr(RUNNER, "helper", lambda *_args: observer)
    args = SimpleNamespace(
        output=tmp_path / "output",
        wait_for_pid=[11],
        wait_for_supervisor_receipt=[path],
        poll_seconds=1,
    )
    return args, final, events


def test_all_originals_captured_before_wait_and_actual_supervisor_child_checked(
    tmp_path, monkeypatch
):
    args, final, events = wait_setup(tmp_path, monkeypatch)
    report = {}
    RUNNER.wait_predecessors(args, report, lambda: None)
    assert events[:2] == [("capture", 11), ("capture", 22)]
    assert ("wait", 33) in events
    assert events[-3:] == [("wait", 11), ("wait", 22), ("wait", 33)]
    assert report["supervisors"][0]["prior_status"] == "completed"
    assert report["predecessor_gate"]


@pytest.mark.parametrize(
    "mutation",
    ["pid", "created", "running", "active", "missing_exit", "missing_identity", "dropped_child"],
)
def test_stale_or_unknown_supervisor_children_fail_closed(tmp_path, monkeypatch, mutation):
    args, final, _events = wait_setup(tmp_path, monkeypatch)
    if mutation == "pid":
        final["pid"] = 99
    elif mutation == "created":
        final["created_at_utc"] = "other epoch"
    elif mutation == "running":
        final["status"] = "running"
    elif mutation == "active":
        final["active_child"] = {"pid": 33, "identity": IDENTITY}
    elif mutation == "missing_exit":
        final["epochs"][0]["exit_code"] = None
    elif mutation == "missing_identity":
        final["epochs"][0]["identity"] = None
    else:
        final["epochs"][0]["pid"] = 44
    with pytest.raises(ValueError):
        RUNNER.wait_predecessors(args, {}, lambda: None)


def test_failed_but_terminal_predecessor_is_preserved_without_claiming_success(
    tmp_path, monkeypatch
):
    args, final, _events = wait_setup(tmp_path, monkeypatch)
    final.update(status="error", error={"type": "RuntimeError", "message": "old backend failure"})
    final["epochs"][0].update(status="error", exit_code=1)
    report = {}
    RUNNER.wait_predecessors(args, report, lambda: None)
    assert report["supervisors"][0]["prior_error"]["message"] == "old backend failure"
    assert report["supervisors"][0]["prior_status"] == "error"


def test_frozen_loader_uses_hashed_bytes_and_rejects_omitted_modules_or_mutation(tmp_path):
    package = tmp_path / "sources/src/s1"
    package.mkdir(parents=True)
    source = package / "__init__.py"
    source.write_text("VALUE = 'archived'\n")
    finder = RUNNER.FrozenS1(tmp_path, {"src/s1/__init__.py": RUNNER.digest(source)})
    spec = finder.find_spec("s1")
    module = importlib.util.module_from_spec(spec)
    finder.exec_module(module)
    assert module.VALUE == "archived" and module.__file__ == str(source)
    with pytest.raises(ImportError, match="absent"):
        finder.find_spec("s1.omitted")
    source.write_text("VALUE = 'mutated'\n")
    with pytest.raises(ValueError, match="changed"):
        finder.exec_module(module)


def test_supervision_does_not_import_or_load_runtime_before_wait_and_validation(
    tmp_path, monkeypatch
):
    args = SimpleNamespace(output=tmp_path / "out")
    events = []
    monkeypatch.setattr(RUNNER, "freeze", lambda *args: events.append("freeze"))
    monkeypatch.setattr(
        RUNNER,
        "wait_predecessors",
        lambda *args: events.extend(["capture_all", "wait_supervisor", "wait_actual_children"]),
    )
    monkeypatch.setattr(RUNNER, "validate_frozen", lambda *args: events.append("validate"))

    def real(_args, report, _save):
        assert events == [
            "freeze",
            "capture_all",
            "wait_supervisor",
            "wait_actual_children",
            "validate",
        ]
        events.append("import_torch_and_load")
        report["summary"] = {"all_numerical_and_shape_gates_passed": True}

    monkeypatch.setattr(RUNNER, "run_real", real)
    original = RUNNER.importlib.metadata.version

    def version(name):
        if name == "numpy":
            raise RUNNER.importlib.metadata.PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(RUNNER.importlib.metadata, "version", version)
    assert RUNNER.run(args) == 0
    report = json.loads((args.output / "report.json").read_text())
    assert report["status"] == "completed"
    assert report["environment"]["packages"]["numpy"] is None
    assert events[-1] == "import_torch_and_load"


def test_changed_source_stops_before_any_runtime_import(tmp_path, monkeypatch):
    args = SimpleNamespace(output=tmp_path / "out")
    monkeypatch.setattr(RUNNER, "freeze", lambda *args: None)
    monkeypatch.setattr(RUNNER, "wait_predecessors", lambda *args: None)
    monkeypatch.setattr(
        RUNNER, "validate_frozen", lambda *args: (_ for _ in ()).throw(ValueError("changed source"))
    )
    monkeypatch.setattr(
        RUNNER, "run_real", lambda *args: pytest.fail("runtime must not import/load")
    )
    with pytest.raises(ValueError, match="changed source"):
        RUNNER.run(args)
    assert json.loads((args.output / "report.json").read_text())["status"] == "error"


@pytest.mark.parametrize("changed", RUNNER.PACKAGES)
def test_changed_queue_environment_rejects_before_any_runtime_import(
    tmp_path, monkeypatch, changed
):
    queued = {name: "original" for name in RUNNER.PACKAGES}
    actual = {**queued, changed: "different"}
    monkeypatch.setattr(RUNNER, "package_versions", lambda: actual)
    monkeypatch.setattr(
        RUNNER, "helper", lambda *args: pytest.fail("no source/runtime load allowed")
    )
    with pytest.raises(ValueError, match="environment changed"):
        RUNNER.load_runtime(SimpleNamespace(output=tmp_path), {"environment": {"packages": queued}})


def test_model_load_failure_runs_final_cleanup_and_preserves_original_error(tmp_path, monkeypatch):
    calls = []

    def constructor(*args, **kwargs):
        calls.append("load")
        assert kwargs == {"device": "mps", "dtype": "bfloat16", "attn_implementation": "sdpa"}
        raise RuntimeError("original model load failure")

    torch = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        mps=SimpleNamespace(
            synchronize=lambda: calls.append("sync"),
            empty_cache=lambda: calls.append("trim"),
            current_allocated_memory=lambda: 20,
            driver_allocated_memory=lambda: 40,
            recommended_max_memory=lambda: 100,
        ),
    )
    runtime = SimpleNamespace(
        torch=torch, unified=SimpleNamespace(UnifiedDecisionModel=constructor)
    )
    monkeypatch.setattr(RUNNER, "load_runtime", lambda *args: runtime)
    report = {}
    with pytest.raises(RuntimeError, match="original model load failure"):
        RUNNER.run_real(SimpleNamespace(model=tmp_path), report, lambda: None)
    assert calls.count("load") == calls.count("trim") == 1
    assert report["fault"]["error"]["message"] == "original model load failure"
    assert report["final_cleanup"]["scope"] == "experiment_final"


@pytest.mark.parametrize("mutation", ["source", "extra_source", "input", "manifest"])
def test_actual_freeze_covers_helper_dependencies_inputs_and_manifest(
    tmp_path, monkeypatch, mutation
):
    root = tmp_path / "source"
    for name in [*RUNNER.HELPERS, "src/s1/__init__.py", "src/s1/unified.py", "src/s1/backends.py"]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# original source\n")
    monkeypatch.setattr(RUNNER, "ROOT", root)
    trace = tmp_path / "trace/independent/dev-design-free-1"
    RUNNER.write_json(
        trace / "task.json",
        {"id": "dev-design-free", "split": "development", "expected": "never used"},
    )
    RUNNER.write_json(
        trace / "decision-transport.json", [{"mapped_request": {"unused": "copied verbatim"}}] * 3
    )
    model = tmp_path / "model"
    RUNNER.write_json(model / "artifact-manifest.json", {"files": []})
    args = SimpleNamespace(output=tmp_path / "output", trace_root=tmp_path / "trace", model=model)
    report = {}
    RUNNER.freeze(args, report)
    RUNNER.validate_frozen(args, report)
    assert set(RUNNER.HELPERS) <= set(report["source_sha256"])
    (root / "src/s1/unified.py").write_text("# later live edit\n")
    RUNNER.validate_frozen(args, report)
    if mutation == "source":
        (args.output / "sources/src/s1/unified.py").write_text("# changed archive\n")
    elif mutation == "extra_source":
        (args.output / "sources/src/s1/omitted.py").write_text("# injected module\n")
    elif mutation == "input":
        (
            args.output / "inputs/trace/independent/dev-design-free-1/decision-transport.json"
        ).write_text("[]")
    else:
        RUNNER.write_json(model / "artifact-manifest.json", {"files": ["mutated"]})
    with pytest.raises(ValueError, match="changed"):
        RUNNER.validate_frozen(args, report)


def test_supervisor_receipt_cannot_bind_an_unrelated_live_process(tmp_path, monkeypatch):
    args, _final, events = wait_setup(tmp_path, monkeypatch)
    observer = RUNNER.helper(None, None, None)
    observer.capture_handle = lambda pid: {"pid": pid, "identity": IDENTITY}
    with pytest.raises(ValueError, match="does not own"):
        RUNNER.wait_predecessors(args, {}, lambda: None)
    assert not any(kind == "wait" for kind, _pid in events)


@pytest.mark.parametrize("first", ["inference_stop", "load_error", "success"])
def test_outer_final_write_fault_never_masks_first_error_or_returns_clean_success(
    tmp_path, monkeypatch, first
):
    args = SimpleNamespace(output=tmp_path / "output")
    report_ref = []
    monkeypatch.setattr(RUNNER, "freeze", lambda *args: None)
    monkeypatch.setattr(RUNNER, "wait_predecessors", lambda *args: None)
    monkeypatch.setattr(RUNNER, "validate_frozen", lambda *args: None)

    def real(_args, report, _save):
        report_ref.append(report)
        if first == "inference_stop":
            raise RUNNER.MaintenanceStop(
                {
                    "kind": "error",
                    "scope": "prediction",
                    "error": {"type": "RuntimeError", "message": "FIRST_INFERENCE_ERROR"},
                }
            )
        if first == "load_error":
            raise RuntimeError("FIRST_LOAD_ERROR")
        report["summary"] = {"all_numerical_and_shape_gates_passed": True}

    write = RUNNER.write_json

    def failing_final(path, report):
        if "finished_at_utc" in report:
            raise OSError("SECONDARY_FINAL_SAVE_ERROR")
        write(path, report)

    monkeypatch.setattr(RUNNER, "run_real", real)
    monkeypatch.setattr(RUNNER, "write_json", failing_final)
    if first == "load_error":
        with pytest.raises(RuntimeError, match="FIRST_LOAD_ERROR"):
            RUNNER.run(args)
    else:
        assert RUNNER.run(args) == 1
    report = report_ref[0]
    assert report["status"] == "error"
    assert report["persistence_errors"][0]["message"] == "SECONDARY_FINAL_SAVE_ERROR"
    if first == "inference_stop":
        assert report["error"]["error"]["message"] == "FIRST_INFERENCE_ERROR"
    elif first == "load_error":
        assert report["error"]["message"] == "FIRST_LOAD_ERROR"


def test_model_load_error_survives_fallback_final_receipt_failure(tmp_path, monkeypatch):
    calls = []

    def constructor(*args, **kwargs):
        raise RuntimeError("FIRST_MODEL_LOAD_ERROR")

    torch = SimpleNamespace(
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        mps=SimpleNamespace(
            synchronize=lambda: None,
            empty_cache=lambda: calls.append("trim"),
            current_allocated_memory=lambda: 20,
            driver_allocated_memory=lambda: 40,
            recommended_max_memory=lambda: 100,
        ),
    )
    monkeypatch.setattr(
        RUNNER,
        "load_runtime",
        lambda *args: SimpleNamespace(
            torch=torch, unified=SimpleNamespace(UnifiedDecisionModel=constructor)
        ),
    )

    def save():
        raise OSError("SECONDARY_FALLBACK_SAVE_ERROR")

    report = {}
    with pytest.raises(RuntimeError, match="FIRST_MODEL_LOAD_ERROR"):
        RUNNER.run_real(SimpleNamespace(model=tmp_path), report, save)
    assert calls == ["trim"]
    assert report["fault"]["error"]["message"] == "FIRST_MODEL_LOAD_ERROR"
    assert report["persistence_errors"][0]["message"] == "SECONDARY_FALLBACK_SAVE_ERROR"


def test_module_top_level_did_not_import_torch():
    # The maintained module's globals contain no runtime module or model object.
    assert "torch" not in RUNNER.__dict__
    assert not any(
        name.startswith("s1")
        for name, value in RUNNER.__dict__.items()
        if getattr(value, "__name__", "").startswith("s1")
    )
    assert (
        "benchmark_mps_maintenance" not in sys.modules
        or sys.modules["benchmark_mps_maintenance"] is RUNNER
    )
