"""CPU fakes only: no Torch import, model, browser, or accelerator execution."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_mps_memory", ROOT / "scripts/benchmark_mps_memory.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def request():
    return {
        "state": "unmodified public observation",
        "questions": {"q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}},
    }


def response(p=0.75):
    return {
        "answers": {
            "q": {"id": "q", "type": "choice", "probabilities": {"a": p, "b": 1 - p}, "choice": "a"}
        }
    }


def memory(driver=100, recommended=1000):
    return {
        "current_allocated_bytes": 80,
        "driver_allocated_bytes": driver,
        "recommended_max_bytes": recommended,
    }


def selected(n=20):
    return [
        {
            "source_line": i,
            "original": {
                "request": request(),
                "response": response(),
                "error": None,
                "latency_ms": i,
            },
        }
        for i in range(1, n + 1)
    ]


def successful_row(driver=100):
    return {
        "error": None,
        "forwards": 1,
        "comparison": RUNNER.compare_responses(request(), response(), response()),
        "allocator_maintenance": {"after": memory(driver), "error": None},
    }


def test_select_first20_successful_callbacks_keeps_original_order_and_full_rows(tmp_path):
    rows = [item["original"] for item in selected(22)]
    rows.insert(1, {"error": {"type": "historical"}, "response": None})
    rows.insert(2, {**rows[0], "allocator_maintenance": {"error": {"type": "trim"}}})
    path = tmp_path / "trace.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    actual = RUNNER.select_records(path)
    assert len(actual) == 20
    assert actual[0] == {"source_line": 1, "original": rows[0]}
    assert actual[1]["source_line"] == 4
    assert [row["original"]["latency_ms"] for row in actual] == list(range(1, 21))
    assert "latency_ms" in actual[0]["original"]


def test_select_rejects_short_or_incomplete_success_trace(tmp_path):
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps({"error": None, "request": {}, "response": response()}) + "\n")
    with pytest.raises(ValueError, match="complete request"):
        RUNNER.select_records(path)
    path.write_text(json.dumps(selected(1)[0]["original"]) + "\n")
    with pytest.raises(ValueError, match="20 successful"):
        RUNNER.select_records(path)


def test_numerical_gate_fixed_delta_and_argmax_are_separate():
    assert RUNNER.compare_responses(request(), response(), response())["passed"]
    assert RUNNER.compare_responses(request(), response(), response(0.759))["passed"]
    drift = RUNNER.compare_responses(request(), response(), response(0.761))
    assert drift["all_argmax_agree"] and not drift["passed"]
    changed = RUNNER.compare_responses(request(), response(0.501), response(0.499))
    assert changed["max_probability_delta"] < 1e-2
    assert not changed["all_argmax_agree"] and not changed["passed"]
    assert changed["max_allowed_probability_delta"] == 1e-2


def test_choice_list_uses_original_strings_while_score_list_uses_ordinal_indices():
    choice = request()
    choice["questions"]["q"]["criteria"] = ["a", "b"]
    assert RUNNER.compare_responses(choice, response(), response())["passed"]
    score = {"state": "same", "questions": {"q": {"type": "score", "criteria": ["low", "high"]}}}
    value = {"answers": {"q": {"type": "score", "probabilities": {"0": 0.25, "1": 0.75}}}}
    assert RUNNER.compare_responses(score, value, value)["passed"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["answers"].update(extra=value["answers"]["q"]),
        lambda value: value["answers"]["q"]["probabilities"].update(c=0),
        lambda value: value["answers"]["q"]["probabilities"].update(a=float("nan")),
        lambda value: value["answers"]["q"]["probabilities"].update(a=-1),
    ],
)
def test_comparison_rejects_schema_and_nonfinite_probability_errors(mutate):
    candidate = response()
    mutate(candidate)
    with pytest.raises(ValueError):
        RUNNER.compare_responses(request(), response(), candidate)


def test_memory_gate_exact_boundary_includes_every_trim_and_first_sample():
    baseline = memory()
    rows = [successful_row(100 + RUNNER.MAX_DRIVER_GROWTH_BYTES) for _ in range(20)]
    summary = RUNNER.summarize(rows, baseline)
    assert summary["memory_gate_passed"] and summary["numerical_gate_passed"]
    assert summary["first_sample_retained"] and summary["warmups"] == 0
    rows[0]["allocator_maintenance"]["after"]["driver_allocated_bytes"] += 1
    summary = RUNNER.summarize(rows, baseline)
    assert not summary["memory_gate_passed"] and summary["numerical_gate_passed"]
    assert summary["post_trim_driver_growth_bytes"][0] == RUNNER.MAX_DRIVER_GROWTH_BYTES + 1


@pytest.mark.parametrize("mutation", ["short", "callback_error", "trim_error", "no_forward"])
def test_incomplete_or_any_error_fails_memory_gate(mutation):
    rows = [successful_row() for _ in range(20)]
    if mutation == "short":
        rows.pop()
    elif mutation == "callback_error":
        rows[0]["error"] = {"type": "ValueError"}
    elif mutation == "trim_error":
        rows[0]["allocator_maintenance"]["error"] = {"type": "RuntimeError"}
    else:
        rows[0]["forwards"] = 0
    assert not RUNNER.summarize(rows, memory())["memory_gate_passed"]


def test_driver_limit_uses_greater_equal_and_valid_recommended_memory():
    RUNNER.check_driver(memory(799, 1000))
    with pytest.raises(RUNNER.DriverLimit):
        RUNNER.check_driver(memory(800, 1000))
    torch = SimpleNamespace(
        mps=SimpleNamespace(
            synchronize=lambda: None,
            current_allocated_memory=lambda: 1,
            driver_allocated_memory=lambda: 2,
            recommended_max_memory=lambda: 0,
        )
    )
    with pytest.raises(RuntimeError, match="valid MPS"):
        RUNNER.memory_snapshot(torch)


class FakeMPS:
    def __init__(self):
        self.driver = 100
        self.trims = 0

    def synchronize(self):
        pass

    def current_allocated_memory(self):
        return 80

    def driver_allocated_memory(self):
        return self.driver

    def recommended_max_memory(self):
        return 1000

    def empty_cache(self):
        self.trims += 1
        self.driver = 100


class FakeBackbone:
    def __init__(self):
        self.before, self.after = [], []

    def register_forward_pre_hook(self, fn, **kwargs):
        assert kwargs == {"with_kwargs": True}
        self.before.append(fn)
        return SimpleNamespace(remove=lambda: self.before.remove(fn))

    def register_forward_hook(self, fn, **kwargs):
        assert kwargs == {"always_call": True}
        self.after.append(fn)
        return SimpleNamespace(remove=lambda: self.after.remove(fn))

    def __call__(self):
        for fn in self.before:
            fn(self, (), {"input_ids": SimpleNamespace(shape=(1, 10))})
        for fn in self.after:
            fn(self, (), object())


def fake_runtime():
    return SimpleNamespace(
        torch=SimpleNamespace(mps=FakeMPS()),
        contracts=SimpleNamespace(
            DecisionRequest=SimpleNamespace(model_validate=lambda value: value)
        ),
        backends=SimpleNamespace(normalize_response=lambda request, result: result),
    )


def test_replay_records_all20_actual_forwards_no_warmup_and_every_raw_response(tmp_path):
    runtime = fake_runtime()
    backbone = FakeBackbone()
    backend = SimpleNamespace(
        model=SimpleNamespace(backbone=backbone), predict=lambda value: (backbone(), response())[1]
    )
    summary = RUNNER.execute_replay(backend, runtime, selected(), tmp_path, memory())
    assert summary["memory_gate_passed"] and summary["numerical_gate_passed"]
    assert summary["actual_forwards"] == runtime.torch.mps.trims == 20
    rows = [json.loads(line) for line in (tmp_path / "callbacks.jsonl").read_text().splitlines()]
    assert len(rows) == 20 and rows[0]["index"] == 1
    assert rows[0]["response"] == rows[0]["old_response"] == response()
    assert rows[0]["inference_wall_ms"] >= 0
    events = [
        json.loads(line) for line in (tmp_path / "memory-events.jsonl").read_text().splitlines()
    ]
    assert len(events) == 40
    assert events[0]["event"] == "before_forward" and events[1]["event"] == "after_forward"
    assert not backbone.before and not backbone.after


def test_before_next_forward_driver_cutoff_dispatches_no_callback(tmp_path):
    runtime = fake_runtime()
    runtime.torch.mps.driver = 800
    calls = []
    backend = SimpleNamespace(predict=lambda value: calls.append(value))
    result = RUNNER.execute_replay(backend, runtime, selected(), tmp_path, memory())
    assert not calls and result["completed_callbacks"] == 0
    assert result["stop_reason"] == "driver_limit" and not result["memory_gate_passed"]


def test_forward_hook_checks_driver_again_after_cpu_preparation(tmp_path):
    runtime = fake_runtime()
    backbone = FakeBackbone()

    def predict(value):
        runtime.torch.mps.driver = 800
        backbone()
        raise AssertionError("real forward must not be reached")

    backend = SimpleNamespace(model=SimpleNamespace(backbone=backbone), predict=predict)
    result = RUNNER.execute_replay(backend, runtime, selected(), tmp_path, memory())
    row = json.loads((tmp_path / "callbacks.jsonl").read_text())
    assert row["forwards"] == 0 and row["error"]["type"] == "DriverLimit"
    assert result["actual_forwards"] == 0 and result["stop_reason"] == "driver_limit"
    assert runtime.torch.mps.trims == 1


def test_inference_exception_and_trim_exception_both_survive(tmp_path, monkeypatch):
    runtime = fake_runtime()
    backbone = FakeBackbone()

    def fail(value):
        backbone()
        raise ValueError("primary inference error")

    def trim_failure():
        raise RuntimeError("secondary trim error")

    runtime.torch.mps.empty_cache = trim_failure
    backend = SimpleNamespace(model=SimpleNamespace(backbone=backbone), predict=fail)
    result = RUNNER.execute_replay(backend, runtime, selected(), tmp_path, memory())
    row = json.loads((tmp_path / "callbacks.jsonl").read_text())
    assert row["error"]["message"] == "primary inference error"
    assert row["allocator_maintenance"]["error"]["message"] == "secondary trim error"
    assert result["error"]["message"] == "primary inference error"
    assert result["completed_callbacks"] == 1 and not result["memory_gate_passed"]


def test_run_rejects_output_overwrite_and_bad_wait_before_model_load(tmp_path):
    args = SimpleNamespace(output=tmp_path, wait_for_pid=[], poll_seconds=15)
    with pytest.raises(FileExistsError):
        RUNNER.run(args)
    args.output = tmp_path / "new"
    args.wait_for_pid = [4, 4]
    with pytest.raises(ValueError, match="unique positive"):
        RUNNER.run(args)
    assert not args.output.exists()


def test_load_runtime_rejects_preimported_s1_before_torch(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "s1", SimpleNamespace(__file__="/another/s1/__init__.py"))
    imported = []
    monkeypatch.setattr(RUNNER.importlib, "import_module", lambda value: imported.append(value))
    with pytest.raises(RuntimeError, match="before frozen"):
        RUNNER.load_runtime(tmp_path)
    assert imported == []


def test_freeze_uses_parent_attested_source_bytes_and_exact_selected_inputs(tmp_path):
    parent = tmp_path / "parent"
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "artifact-manifest.json").write_text("{}")
    source_hashes = {}
    for name in ("src/s1/unified.py", "src/s1/backends.py"):
        path = parent / "sources" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# exact old bytes\n")
        source_hashes[name] = {"sha256": RUNNER.digest(path), "size_bytes": path.stat().st_size}
    receipt = {
        "source_inventory": source_hashes,
        "checkpoint": {
            "path": str(checkpoint),
            "manifest_sha256": RUNNER.digest(checkpoint / "artifact-manifest.json"),
        },
    }
    RUNNER.write_json(parent / "receipt.json", receipt)
    trace = parent / "trace.jsonl"
    trace.write_text("".join(json.dumps(row["original"]) + "\n" for row in selected()))
    args = SimpleNamespace(parent_run=parent, trace=trace, output=tmp_path / "out")
    actual_parent, hashes = RUNNER.freeze_inputs(args, b"# frozen guard\n")
    frozen = args.output / "sources"
    assert actual_parent == receipt and hashes == RUNNER.inventory(frozen)
    assert (frozen / "src/s1/unified.py").read_bytes() == (
        parent / "sources/src/s1/unified.py"
    ).read_bytes()
    assert (frozen / "scripts/resume_mps_benchmarks.py").read_bytes() == b"# frozen guard\n"
    assert (frozen / "inputs/original-callbacks.jsonl").read_bytes() == trace.read_bytes()
    assert len(json.loads((frozen / "inputs/selected-callbacks.json").read_text())) == 20
    (parent / "sources/src/s1/unified.py").write_text("changed")
    args.output = tmp_path / "different"
    with pytest.raises(ValueError, match="differs from parent"):
        RUNNER.freeze_inputs(args, b"guard")


def lifecycle(tmp_path, monkeypatch, mutate=False):
    events = []
    args = SimpleNamespace(
        output=tmp_path / "out",
        wait_for_pid=[41, 42],
        poll_seconds=0.1,
        parent_run=tmp_path / "parent",
    )
    expected = dict.fromkeys(RUNNER.RUNTIME_FIELDS)
    expected.update(
        model="frozen-model",
        device="mps",
        dtype="bfloat16",
        attn_implementation="sdpa",
        question_mode="causal_multislot",
        torch_version="2.8.0",
        transformers_version="5.17.0",
    )
    versions = {
        "torch": "2.14.1",
        "torchvision": "0.29.1",
        "transformers": "5.17.0",
        "numpy": "2.1",
    }
    parent = {
        "checkpoint": {"path": "frozen-model"},
        "runtime_loaded": expected,
        "packages": {"numpy": "2.1"},
    }
    handles = []

    def capture(pid):
        events.append(f"capture{pid}")
        handle = {"pid": pid, "identity": f"original-{pid}"}
        handles.append(handle)
        return handle

    def freeze(args, raw):
        events.append("freeze")
        RUNNER.write_json(args.output / "sources/inputs/selected-callbacks.json", selected())
        return parent, RUNNER.inventory(args.output / "sources")

    def wait(handle, save, poll_seconds):
        assert handle is handles[args.wait_for_pid.index(handle["pid"])]
        assert events.index("freeze") < len(events)
        events.append(f"wait{handle['pid']}")
        if mutate:
            (args.output / "sources/inputs/selected-callbacks.json").write_text("changed")

    guard = SimpleNamespace(
        capture_handle=capture,
        wait_handle=wait,
        verify_checkpoint=lambda value: events.append("verify_checkpoint"),
        package_version=versions.get,
    )
    monkeypatch.setattr(RUNNER, "load_module", lambda *args: guard)
    monkeypatch.setattr(RUNNER, "freeze_inputs", freeze)
    monkeypatch.setattr(RUNNER.importlib.metadata, "version", versions.__getitem__)

    def runtime(output):
        assert events[-1] == "verify_checkpoint"
        events.append("load_runtime")
        result = fake_runtime()
        result.torch.version = SimpleNamespace(git_version="new-runtime-pin")
        metadata = {**expected, "torch_version": "2.14.1"}
        result.backends.GemmaBackend = lambda *args, **kwargs: SimpleNamespace(metadata=metadata)
        return result

    monkeypatch.setattr(RUNNER, "load_runtime", runtime)
    monkeypatch.setattr(
        RUNNER,
        "execute_replay",
        lambda *args: {"memory_gate_passed": True, "numerical_gate_passed": True, "error": None},
    )
    return args, events


def test_all_original_handles_captured_and_sources_frozen_before_wait_then_lazy_load(
    tmp_path, monkeypatch
):
    args, events = lifecycle(tmp_path, monkeypatch)
    assert RUNNER.run(args) == 0
    assert events == [
        "capture41",
        "capture42",
        "freeze",
        "wait41",
        "wait42",
        "verify_checkpoint",
        "load_runtime",
    ]
    receipt = json.loads((args.output / "receipt.json").read_text())
    assert receipt["status"] == "completed_replay"
    assert receipt["torch_git_version"] == "new-runtime-pin"


def test_frozen_request_change_during_wait_fails_before_torch_or_model_load(tmp_path, monkeypatch):
    args, events = lifecycle(tmp_path, monkeypatch, mutate=True)
    assert RUNNER.run(args) == 1
    assert "load_runtime" not in events and "verify_checkpoint" not in events
    receipt = json.loads((args.output / "receipt.json").read_text())
    assert receipt["status"] == "error" and "changed while queued" in receipt["error"]["message"]


def test_missing_optional_torch_metadata_still_freezes_and_waits_before_failure(
    tmp_path, monkeypatch
):
    args, events = lifecycle(tmp_path, monkeypatch)
    guard = RUNNER.load_module(None, None)
    original = guard.package_version
    guard.package_version = lambda name: None if name == "torch" else original(name)
    assert RUNNER.run(args) == 1
    assert events == ["capture41", "capture42", "freeze", "wait41", "wait42", "verify_checkpoint"]
    receipt = json.loads((args.output / "receipt.json").read_text())
    assert receipt["packages"]["torch"] is None
    assert "unavailable after predecessor waits" in receipt["error"]["message"]
