"""Bounded shape-diagnostic contracts; CPU fake forwards, no model/GPU calls."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from s1.contracts import DecisionRequest, answer_from_probabilities

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "diagnose_mps_prefix_shapes", ROOT / "scripts/diagnose_mps_prefix_shapes.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


@pytest.fixture
def decision_request():
    return DecisionRequest.model_validate(
        {
            "state": {"observation": "development page"},
            "questions": [
                {"id": name, "instructions": "Choose", "criteria": {"a": "First", "b": "Second"}}
                for name in ("operation", "target")
            ],
        }
    )


def response(decision_request, p=0.75):
    return {
        "answers": {
            question.id: answer_from_probabilities(question, [p, 1 - p])
            for question in decision_request.questions
        }
    }


@pytest.fixture
def inputs(tmp_path, decision_request):
    folder = tmp_path / "trace/independent/dev-case-1"
    folder.mkdir(parents=True)
    RUNNER.write_json(
        folder / "task.json", {"id": "dev-case", "split": "development", "expected": "unused"}
    )
    rows = [
        {
            "error": None,
            "mapped_request": decision_request.model_dump(),
            "raw_response": "unused gold",
        },
        {
            "error": None,
            "mapped_request": {**decision_request.model_dump(), "state": "second state"},
        },
    ]
    RUNNER.write_json(folder / "decision-transport.json", rows)
    return tmp_path / "trace", folder


def test_snapshot_uses_only_selected_development_request(inputs, decision_request):
    root, folder = inputs
    actual, provenance = RUNNER.load_development_snapshot(root, "dev-case", 2)
    assert actual.state == "second state"
    assert actual.questions == decision_request.questions
    assert provenance["labels_or_outcomes_used"] is False
    assert provenance["request_sha256"] == RUNNER.content_sha256(actual.model_dump())
    assert provenance["trace_sha256"] == RUNNER.sha256(folder / "decision-transport.json")


@pytest.mark.parametrize("index", [0, -1, 3])
def test_snapshot_index_is_positive_and_bounded(inputs, index):
    with pytest.raises(ValueError, match="index"):
        RUNNER.load_development_snapshot(inputs[0], "dev-case", index)


def test_transfer_metadata_cannot_be_disguised_by_development_id(inputs):
    root, folder = inputs
    RUNNER.write_json(folder / "task.json", {"id": "dev-case", "split": "reserved_transfer"})
    with pytest.raises(ValueError, match="reserved/transfer"):
        RUNNER.load_development_snapshot(root, "dev-case", 1)
    with pytest.raises(ValueError, match="development"):
        RUNNER.load_development_snapshot(root, "transfer-case", 1)


@pytest.mark.parametrize("mutation", ["failed", "missing", "singleton", "too_many"])
def test_bad_or_unbounded_snapshot_fails_before_model_load(inputs, mutation):
    root, folder = inputs
    path = folder / "decision-transport.json"
    rows = json.loads(path.read_text())
    if mutation == "failed":
        rows[0]["error"] = {"type": "InferenceError"}
    elif mutation == "missing":
        rows[0]["mapped_request"] = None
    elif mutation == "singleton":
        rows[0]["mapped_request"]["questions"] = rows[0]["mapped_request"]["questions"][:1]
    else:
        question = rows[0]["mapped_request"]["questions"][0]
        rows[0]["mapped_request"]["questions"] = [{**question, "id": f"q{i}"} for i in range(9)]
    RUNNER.write_json(path, rows)
    with pytest.raises(ValueError):
        RUNNER.load_development_snapshot(root, "dev-case", 1)


@pytest.fixture
def layout():
    return {
        "prefix_tokens": 3,
        "heads": [
            {"question_id": "operation", "absolute_slot": 5, "suffix_slot": 2, "full_tokens": 6},
            {"question_id": "target", "absolute_slot": 7, "suffix_slot": 4, "full_tokens": 8},
        ],
    }


def test_slot_selection_tracks_full_singleton_and_cached_coordinates(layout):
    assert RUNNER.selected_slots("A", 0, layout, 2) == [(0, 5, "operation"), (1, 7, "target")]
    assert RUNNER.selected_slots("B", 1, layout, 1) == [(0, 7, "target")]
    assert RUNNER.selected_slots("D", 0, layout, 1) == []
    assert RUNNER.selected_slots("D", 1, layout, 2) == [(0, 2, "operation"), (1, 4, "target")]
    with pytest.raises(ValueError, match="shape"):
        RUNNER.selected_slots("A", 1, layout, 2)


def test_summary_keeps_all_samples_and_fixed_probability_gate(decision_request):
    rows = [
        {"phase": phase, "variant": name, "raw_response": response(decision_request), "error": None}
        for phase, name in RUNNER.SCHEDULE
    ]
    assert RUNNER.summarize(decision_request, rows)["all_pairwise_gates_passed"]
    rows[4]["raw_response"] = response(decision_request, 0.70)
    summary = RUNNER.summarize(decision_request, rows)
    assert not summary["all_pairwise_gates_passed"]
    assert summary["comparisons"]["A_vs_B"]["max_probability_delta"] == pytest.approx(0.05)
    assert summary["repeat_max_probability_delta"]["B"] == pytest.approx(0.05)
    assert RUNNER.MAX_PROBABILITY_DELTA == 0.01
    with pytest.raises(ValueError, match="retained"):
        RUNNER.summarize(decision_request, rows[:-1])
    rows[4]["error"] = {"type": "InvalidOutput"}
    with pytest.raises(ValueError, match="failed attempts"):
        RUNNER.summarize(decision_request, rows)


def fake_operations(decision_request, *, fail_variant=None):
    torch = pytest.importorskip("torch")

    class Backbone(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            assert input_ids.device.type == "cpu"
            # Each output vector is recognizably [row, slot] for coordinate checks.
            batch, length = input_ids.shape
            values = torch.zeros(batch, length, 2, device="cpu")
            values[:, :, 0] = torch.arange(batch)[:, None]
            values[:, :, 1] = torch.arange(length)[None, :]
            return SimpleNamespace(last_hidden_state=values)

    backbone = Backbone()

    def operation(variant):
        def call(_request):
            if variant == "A":
                backbone(input_ids=torch.zeros(2, 8, dtype=torch.long), use_cache=False)
            elif variant == "B":
                for length in (6, 8):
                    backbone(input_ids=torch.zeros(1, length, dtype=torch.long), use_cache=False)
            else:
                backbone(input_ids=torch.zeros(1, 3, dtype=torch.long), use_cache=True)
                backbone(
                    input_ids=torch.zeros(2, 5, dtype=torch.long),
                    attention_mask=torch.ones(2, 8),
                    position_ids=torch.arange(3, 8)[None].expand(2, -1),
                    past_key_values=object(),
                    use_cache=True,
                )
            if variant == fail_variant:
                raise RuntimeError("inference interrupted")
            return response(decision_request)

        return call

    return SimpleNamespace(backbone=backbone), {name: operation(name) for name in ("A", "B", "D")}


def test_schedule_records_all_cpu_fake_forwards_and_exact_slot_vectors(
    decision_request, layout, tmp_path
):
    model, operations = fake_operations(decision_request)
    report, syncs = {}, []
    clock = iter(range(18))
    summary = RUNNER.run_schedule(
        model,
        decision_request,
        layout,
        operations,
        report,
        tmp_path,
        synchronize=lambda: syncs.append(True),
        memory=lambda: {"device": "cpu-fake"},
        clock=lambda: next(clock),
    )
    assert summary["all_pairwise_gates_passed"]
    assert len(syncs) == 18
    assert len(report["samples"]) == 9
    a, b, d = report["samples"][:3]
    assert a["forwards"][0]["selected_slots"][0]["hidden_vector"] == [0.0, 5.0]
    assert b["forwards"][1]["selected_slots"][0]["hidden_vector"] == [0.0, 7.0]
    assert d["forwards"][0]["selected_slots"] == []
    assert d["forwards"][1]["selected_slots"][1]["hidden_vector"] == [1.0, 4.0]
    assert d["forwards"][1]["has_past_key_values"]
    assert d["forwards"][1]["attention_mask_shape"] == [2, 8]
    assert not model.backbone._forward_hooks
    assert json.loads((tmp_path / "report.json").read_text())["samples"] == report["samples"]


def test_failed_attempt_keeps_forward_vectors_and_error_receipt(decision_request, layout, tmp_path):
    model, operations = fake_operations(decision_request, fail_variant="B")
    report = {}
    clock = iter(range(4))
    with pytest.raises(RuntimeError, match="interrupted"):
        RUNNER.run_schedule(
            model,
            decision_request,
            layout,
            operations,
            report,
            tmp_path,
            synchronize=lambda: None,
            memory=lambda: {},
            clock=lambda: next(clock),
        )
    rows = json.loads((tmp_path / "report.json").read_text())["samples"]
    assert len(rows) == 2
    assert rows[-1]["error"]["type"] == "RuntimeError"
    assert len(rows[-1]["forwards"]) == 2
    assert rows[-1]["forwards"][1]["selected_slots"][0]["hidden_vector"] == [0.0, 7.0]
    assert not model.backbone._forward_hooks


def test_freeze_and_validation_reject_queued_source_edits(tmp_path):
    source = tmp_path / "repository"
    paths = [
        "src/s1/unit.py",
        "scripts/diagnose_mps_prefix_shapes.py",
        "scripts/run_ultrafast.py",
        "scripts/resume_mps_benchmarks.py",
    ]
    for name in paths:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# frozen bytes\n")
    output = tmp_path / "run"
    hashes = RUNNER.archive_sources(output, source)
    RUNNER.validate_frozen_sources(output, hashes, source)
    (source / paths[0]).write_text("# changed bytes\n")
    with pytest.raises(ValueError, match="source bytes changed"):
        RUNNER.validate_frozen_sources(output, hashes, source)


def test_all_predecessors_captured_before_waiting(monkeypatch, tmp_path):
    events, report = [], {}

    def capture(pid):
        events.append(("capture", pid))
        return {"pid": pid, "identity": "live"}

    def wait(handle, save, seconds):
        assert seconds == 15
        events.append(("wait", handle["pid"]))
        handle["status"] = "original_exited"
        save()

    monkeypatch.setattr(
        RUNNER,
        "load_wait_tools",
        lambda _output: SimpleNamespace(capture_handle=capture, wait_handle=wait),
    )
    RUNNER.wait_for_predecessors([11, 22], report, tmp_path)
    assert events == [("capture", 11), ("capture", 22), ("wait", 11), ("wait", 22)]
    assert all(row["status"] == "original_exited" for row in report["predecessors"])


def test_main_freezes_inputs_before_fake_wait_and_does_not_load_on_wait_error(
    inputs, tmp_path, monkeypatch
):
    events = []
    output = tmp_path / "run"

    def archive(path):
        events.append("freeze")
        assert (path / "request.json").is_file()
        return {"fake-source": "hash"}

    def wait(pids, report, path):
        assert pids == [11, 22]
        assert report["source_sha256"] == {"fake-source": "hash"}
        events.append("wait")
        raise RuntimeError("observation failed; predecessor still unverified")

    monkeypatch.setattr(RUNNER, "archive_sources", archive)
    monkeypatch.setattr(RUNNER, "wait_for_predecessors", wait)
    monkeypatch.setattr(
        RUNNER, "run_real_diagnostic", lambda *_args: pytest.fail("model must not load")
    )
    with pytest.raises(RuntimeError, match="observation failed"):
        RUNNER.main(
            [
                "--trace-root",
                str(inputs[0]),
                "--development-task",
                "dev-case",
                "--decision-index",
                "1",
                "--output",
                str(output),
                "--wait-for-pid",
                "11",
                "--wait-for-pid",
                "22",
            ]
        )
    assert events == ["freeze", "wait"]
    assert json.loads((output / "report.json").read_text())["status"] == "error"


def test_main_validates_source_after_wait_before_fake_model_boundary(inputs, tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(RUNNER, "archive_sources", lambda _output: events.append("freeze") or {})
    monkeypatch.setattr(RUNNER, "wait_for_predecessors", lambda *_args: events.append("wait"))
    monkeypatch.setattr(RUNNER, "validate_frozen_sources", lambda *_args: events.append("validate"))

    def fake_load(*_args):
        events.append("model_boundary")
        return {"all_pairwise_gates_passed": False}

    monkeypatch.setattr(RUNNER, "run_real_diagnostic", fake_load)
    assert (
        RUNNER.main(
            [
                "--trace-root",
                str(inputs[0]),
                "--development-task",
                "dev-case",
                "--decision-index",
                "1",
                "--output",
                str(tmp_path / "run"),
            ]
        )
        == 1
    )
    assert events == ["freeze", "wait", "validate", "model_boundary"]


@pytest.mark.parametrize("mutation", ["source", "input"])
def test_queued_changes_fail_before_real_model_boundary(inputs, tmp_path, monkeypatch, mutation):
    output = tmp_path / "run"
    monkeypatch.setattr(RUNNER, "archive_sources", lambda _output: {})

    def wait(_pids, _report, path):
        if mutation == "input":
            (path / "request.json").write_text("changed frozen input")

    def validate(*_args):
        if mutation == "source":
            raise ValueError("source bytes changed while queued")

    monkeypatch.setattr(RUNNER, "wait_for_predecessors", wait)
    monkeypatch.setattr(RUNNER, "validate_frozen_sources", validate)
    monkeypatch.setattr(
        RUNNER, "run_real_diagnostic", lambda *_args: pytest.fail("model must not load")
    )
    with pytest.raises(ValueError, match="changed while queued"):
        RUNNER.main(
            [
                "--trace-root",
                str(inputs[0]),
                "--development-task",
                "dev-case",
                "--decision-index",
                "1",
                "--output",
                str(output),
            ]
        )
    assert json.loads((output / "report.json").read_text())["status"] == "error"


def test_nonfinite_selected_vectors_are_durable_errors_not_invalid_json(layout):
    torch = pytest.importorskip("torch")
    recorder = RUNNER.ForwardRecorder("A", layout)
    recorder(
        None,
        (),
        {"input_ids": torch.zeros(2, 8, dtype=torch.long)},
        SimpleNamespace(last_hidden_state=torch.full((2, 8, 2), float("nan"))),
    )
    rows = recorder.finish()
    slot = rows[0]["selected_slots"][0]
    assert slot["hidden_vector"] is None
    assert slot["capture_error"]["type"] == "ValueError"
    assert "nan" in slot["hidden_vector_repr"]
    json.dumps(rows, allow_nan=False)
