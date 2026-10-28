"""Pure prefix-experiment measurement contracts; no accelerators or inference."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from s1.contracts import DecisionRequest, answer_from_probabilities
from s1.errors import BackendResponseError

SPEC = importlib.util.spec_from_file_location(
    "benchmark_mps_prefix", Path(__file__).resolve().parents[1] / "scripts/benchmark_mps_prefix.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


@pytest.fixture
def prefix_request():
    return DecisionRequest.model_validate(
        {
            "state": {"page": "development observation"},
            "questions": {
                "operation": {
                    "instructions": "Choose an observed operation",
                    "type": "choice",
                    "criteria": {"CLICK": "Click a control", "DONE": "Requirements satisfied"},
                }
            },
        }
    )


def response(prefix_request, p):
    return {
        "answers": {
            q.id: answer_from_probabilities(q, [p, 1 - p]) for q in prefix_request.questions
        }
    }


def sample(prefix_request, profile, latency, p, phase="measured"):
    return {
        "profile": profile,
        "phase": phase,
        "latency_ms": latency,
        "response": response(prefix_request, p),
    }


def test_compare_answers_exact_probabilities_and_argmax(prefix_request):
    original = response(prefix_request, 0.75)
    assert RUNNER.compare_answers(prefix_request, original, copy.deepcopy(original)) == {
        "max_probability_delta": 0,
        "all_argmax_agree": True,
        "decisions": 1,
    }
    delta = RUNNER.compare_answers(prefix_request, original, response(prefix_request, 0.625))
    assert delta["max_probability_delta"] == 0.125
    assert delta["all_argmax_agree"]
    disagreement = RUNNER.compare_answers(prefix_request, original, response(prefix_request, 0.25))
    assert disagreement["max_probability_delta"] == 0.5
    assert not disagreement["all_argmax_agree"]


@pytest.mark.parametrize(
    "invalid", ["missing_question", "wrong_labels", "bad_mass", "false_choice"]
)
def test_compare_answers_rejects_schema_or_probability_mismatch(prefix_request, invalid):
    candidate = response(prefix_request, 0.75)
    answer = candidate["answers"]["operation"]
    if invalid == "missing_question":
        candidate["answers"] = {}
    elif invalid == "wrong_labels":
        answer["probabilities"] = {"CLICK": 0.75, "OTHER": 0.25}
    elif invalid == "bad_mass":
        answer["probabilities"] = {"CLICK": 0.75, "DONE": 0.75}
    else:
        answer["choice"] = "DONE"
    with pytest.raises(BackendResponseError):
        RUNNER.compare_answers(prefix_request, response(prefix_request, 0.75), candidate)


def test_compare_answers_counts_every_typed_head(decision_request):
    probabilities = [[0.75, 0.25], [0.25, 0.75], [0.625, 0.375]]
    original = {
        "answers": {
            q.id: answer_from_probabilities(q, p)
            for q, p in zip(decision_request.questions, probabilities, strict=True)
        }
    }
    changed = copy.deepcopy(original)
    urgency = next(q for q in decision_request.questions if q.id == "urgency")
    changed["answers"]["urgency"] = answer_from_probabilities(urgency, [0.375, 0.625])
    comparison = RUNNER.compare_answers(decision_request, original, changed)
    assert comparison["decisions"] == 3
    assert comparison["max_probability_delta"] == 0.25
    assert not comparison["all_argmax_agree"]


def test_summary_keeps_slow_and_drifting_measurements_and_excludes_warmups(prefix_request):
    samples = [
        sample(prefix_request, "baseline", 1_000_000, 0.01, "warmup"),
        sample(prefix_request, "prefix", 1_000_000, 0.99, "warmup"),
        sample(prefix_request, "baseline", 100, 0.625),
        sample(prefix_request, "prefix", 50, 0.625),
        sample(prefix_request, "prefix", 2000, 0.875),
        sample(prefix_request, "baseline", 1000, 0.875),
    ]
    before = copy.deepcopy(samples)
    summary = RUNNER.summarize_case(prefix_request, samples, max_delta=0.01)
    assert samples == before
    assert summary["samples_per_profile"] == 2
    assert summary["latency_ms_p50_p95"]["baseline"] == pytest.approx([550, 955])
    assert summary["latency_ms_p50_p95"]["prefix"] == pytest.approx([1025, 1902.5])
    assert summary["median_latency_ratio"] == pytest.approx(1025 / 550)
    # Matched positions agree exactly. The all-to-all audit must still expose
    # repeated BF16 instability, rather than selecting a favorable pair.
    assert summary["max_probability_delta"] == 0.25
    assert summary["baseline_repeat_max_delta"] == 0.25
    assert summary["all_argmax_agree"]
    assert not summary["parity_gate_passed"]


def test_summary_uses_predeclared_delta_without_epsilon_or_posthoc_relaxation(prefix_request):
    samples = [
        sample(prefix_request, "baseline", 100, 0.75),
        sample(prefix_request, "prefix", 50, 0.625),
    ]
    assert RUNNER.summarize_case(prefix_request, samples, max_delta=0.125)["parity_gate_passed"]
    assert not RUNNER.summarize_case(prefix_request, samples, max_delta=0.124999)[
        "parity_gate_passed"
    ]


def test_summary_argmax_gate_cannot_be_bypassed_with_large_delta_bound(prefix_request):
    samples = [
        sample(prefix_request, "baseline", 100, 0.75),
        sample(prefix_request, "prefix", 50, 0.25),
    ]
    summary = RUNNER.summarize_case(prefix_request, samples, max_delta=1)
    assert not summary["all_argmax_agree"]
    assert not summary["parity_gate_passed"]


@pytest.mark.parametrize(
    "profiles", [[], ["baseline"], ["prefix"], ["baseline", "prefix", "prefix"]]
)
def test_summary_requires_balanced_nonempty_measured_samples(prefix_request, profiles):
    samples = [sample(prefix_request, "baseline", 100, 0.75, "warmup")]
    samples += [sample(prefix_request, profile, 100, 0.75) for profile in profiles]
    with pytest.raises(ValueError, match="balanced nonempty"):
        RUNNER.summarize_case(prefix_request, samples, max_delta=0.01)


@pytest.mark.parametrize("field,value", [("profile", "untracked"), ("phase", "discarded")])
def test_summary_cannot_silently_exclude_unknown_measurements(prefix_request, field, value):
    samples = [
        sample(prefix_request, "baseline", 100, 0.75),
        sample(prefix_request, "prefix", 50, 0.75),
    ]
    unknown = sample(prefix_request, "prefix", 10000, 0.25)
    unknown[field] = value
    samples.append(unknown)
    with pytest.raises(ValueError, match="cannot be excluded"):
        RUNNER.summarize_case(prefix_request, samples, max_delta=0.01)


@pytest.fixture
def case_files(tmp_path, prefix_request):
    folder = tmp_path / "traces/independent/dev-observation-1"
    folder.mkdir(parents=True)
    task = {
        "id": "dev-observation",
        "split": "development",
        "expected": {"never_pass_to_model": "private verifier"},
    }
    (folder / "task.json").write_text(json.dumps(task))
    mapped = prefix_request.model_dump()
    transport = [
        {
            "error": None,
            "mapped_request": mapped,
            "original_body": {"state": {"not_used": "original wire metadata"}},
            "raw_response": {"not_used": "predicted output"},
        },
        {"error": None, "mapped_request": {**mapped, "state": "second mapped observation"}},
    ]
    (folder / "decision-transport.json").write_text(json.dumps(transport))
    native = tmp_path / "native.jsonl"
    native.write_text(
        json.dumps(
            {
                "id": "native-case",
                "split": "test",
                "request": mapped,
                "gold": {"operation": "NEVER_READ_GOLD"},
                "expected": "oracle is not an inference input",
            }
        )
        + "\n"
    )
    return tmp_path / "traces", folder, native, mapped


def test_load_cases_uses_only_mapped_and_native_requests_without_gold(case_files):
    root, folder, native, mapped = case_files
    cases, provenance = RUNNER.load_cases(root, "dev-observation", native)
    assert [case_id for case_id, _ in cases] == [
        "browser-dev-observation-1",
        "browser-dev-observation-2",
        "native-native-case",
    ]
    assert cases[0][1].model_dump() == mapped
    assert cases[1][1].state == "second mapped observation"
    assert cases[2][1].model_dump() == mapped
    assert provenance["labels_used"] is False
    assert (
        provenance["development_trace_sha256"]
        == hashlib.sha256((folder / "decision-transport.json").read_bytes()).hexdigest()
    )
    row = json.loads(native.read_text())
    row["gold"] = {"operation": "DIFFERENT_AND_STILL_IGNORED"}
    row["expected"] = "changed oracle"
    native.write_text(json.dumps(row) + "\n")
    changed, changed_provenance = RUNNER.load_cases(root, "dev-observation", native)
    assert [(case_id, value.model_dump()) for case_id, value in changed] == [
        (case_id, value.model_dump()) for case_id, value in cases
    ]
    assert changed_provenance["native_dataset_sha256"] != provenance["native_dataset_sha256"]


def test_load_cases_rejects_transfer_snapshot_even_with_development_name(case_files):
    root, folder, native, _ = case_files
    task = json.loads((folder / "task.json").read_text())
    task["split"] = "reserved_transfer"
    (folder / "task.json").write_text(json.dumps(task))
    with pytest.raises(ValueError, match="reserved/transfer"):
        RUNNER.load_cases(root, "dev-observation", native)


def test_load_cases_rejects_empty_development_trace_even_with_valid_native_inputs(case_files):
    root, folder, native, _ = case_files
    (folder / "decision-transport.json").write_text("[]")
    with pytest.raises(ValueError, match="development trace cannot be empty"):
        RUNNER.load_cases(root, "dev-observation", native)


def test_load_cases_rejects_non_development_task_id_without_reading_files(tmp_path):
    with pytest.raises(ValueError, match="development tasks"):
        RUNNER.load_cases(tmp_path, "transfer-unseen", tmp_path / "absent.jsonl")


@pytest.mark.parametrize("change", ["error", "missing_request", "native_train_split"])
def test_load_cases_rejects_incomplete_or_non_test_inputs(case_files, change):
    root, folder, native, _ = case_files
    if change == "native_train_split":
        row = json.loads(native.read_text())
        row["split"] = "train"
        native.write_text(json.dumps(row) + "\n")
    else:
        rows = json.loads((folder / "decision-transport.json").read_text())
        if change == "error":
            rows[0]["error"] = {"type": "InferenceFailure"}
        else:
            rows[0]["mapped_request"] = None
        (folder / "decision-transport.json").write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        RUNNER.load_cases(root, "dev-observation", native)


def process_line(state, *, started="Mon Oct 26 20:40:00 2026", command="python benchmark.py"):
    assert len(started) == 24
    return f"{started} {state} {command}"


def mock_process_observations(monkeypatch, observations):
    rows = iter(observations)
    calls, sleeps = [], []

    def identity(pid):
        calls.append(pid)
        return next(rows)

    monkeypatch.setattr(RUNNER, "process_identity", identity)
    monkeypatch.setattr(RUNNER.time, "sleep", sleeps.append)
    return calls, sleeps


def test_wait_tracks_same_process_through_state_changes_until_exit(monkeypatch, tmp_path):
    initial = process_line("S")
    calls, sleeps = mock_process_observations(
        monkeypatch, [initial, process_line("R+"), process_line("S+"), None]
    )
    RUNNER.wait_for_process(1234, tmp_path)
    assert calls == [1234, 1234, 1234, 1234]
    assert sleeps == [15, 15]
    assert json.loads((tmp_path / "predecessor.json").read_text())["identity"] == initial
    assert json.loads((tmp_path / "predecessor-completed.json").read_text()) == {
        "pid": 1234,
        "last_observation": None,
    }


def test_wait_rejects_missing_initial_process(monkeypatch, tmp_path):
    _, sleeps = mock_process_observations(monkeypatch, [None])
    with pytest.raises(ValueError, match="not live"):
        RUNNER.wait_for_process(1234, tmp_path)
    assert sleeps == []
    assert not (tmp_path / "predecessor-completed.json").exists()


@pytest.mark.parametrize(
    "reused",
    [
        process_line("S", started="Tue Oct 27 20:40:00 2026"),
        process_line("S", command="python different-job.py"),
    ],
)
def test_wait_does_not_follow_reused_pid(monkeypatch, tmp_path, reused):
    calls, sleeps = mock_process_observations(monkeypatch, [process_line("S"), reused])
    RUNNER.wait_for_process(1234, tmp_path)
    assert calls == [1234, 1234]
    assert sleeps == []
    assert (
        json.loads((tmp_path / "predecessor-completed.json").read_text())["last_observation"]
        == reused
    )


def test_wait_recognizes_terminal_zombie_without_another_sleep(monkeypatch, tmp_path):
    _, sleeps = mock_process_observations(monkeypatch, [process_line("S"), process_line("Z+")])
    RUNNER.wait_for_process(1234, tmp_path)
    assert sleeps == []


def mock_ps_and_probe(monkeypatch, *, returncode=0, stdout="", stderr="", probe_error=None):
    observed = {"ps": [], "probe": []}

    def ps(args, **kwargs):
        observed["ps"].append((args, kwargs))
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    def probe(pid, signal):
        observed["probe"].append((pid, signal))
        if probe_error is not None:
            raise probe_error

    monkeypatch.setattr(RUNNER.subprocess, "run", ps)
    monkeypatch.setattr(RUNNER.os, "kill", probe)
    return observed


def test_process_identity_reads_successful_ps_without_an_absence_probe(monkeypatch):
    identity = process_line("R+")
    observed = mock_ps_and_probe(monkeypatch, stdout="  " + identity + "\n")
    assert RUNNER.process_identity(424242) == identity
    assert observed["probe"] == []
    assert observed["ps"] == [
        (
            ["ps", "-p", "424242", "-o", "lstart=,stat=,command="],
            {"capture_output": True, "text": True},
        )
    ]


@pytest.mark.parametrize("returncode", [0, 1])
def test_process_identity_requires_independent_missing_pid_confirmation(monkeypatch, returncode):
    observed = mock_ps_and_probe(
        monkeypatch, returncode=returncode, probe_error=ProcessLookupError()
    )
    assert RUNNER.process_identity(424242) is None
    assert observed["probe"] == [(424242, 0)]


@pytest.mark.parametrize(
    "returncode,stdout,stderr",
    [
        (2, "", ""),
        (1, "", "ps: permission denied"),
        (0, process_line("S"), "ps: partial observation"),
        (1, process_line("S"), ""),
    ],
)
def test_process_identity_does_not_treat_ps_observation_errors_as_exit(
    monkeypatch, returncode, stdout, stderr
):
    observed = mock_ps_and_probe(
        monkeypatch,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        probe_error=ProcessLookupError(),
    )
    with pytest.raises(RuntimeError, match="observation failed"):
        RUNNER.process_identity(424242)
    assert observed["probe"] == []


def test_process_identity_refuses_live_pid_with_no_ps_identity(monkeypatch):
    observed = mock_ps_and_probe(monkeypatch, returncode=1)
    with pytest.raises(RuntimeError, match="live but ps returned no identity"):
        RUNNER.process_identity(424242)
    assert observed["probe"] == [(424242, 0)]


def test_process_identity_cannot_convert_probe_permission_error_to_absence(monkeypatch):
    observed = mock_ps_and_probe(monkeypatch, returncode=1, probe_error=PermissionError())
    with pytest.raises(RuntimeError, match="cannot verify"):
        RUNNER.process_identity(424242)
    assert observed["probe"] == [(424242, 0)]


def test_process_identity_preserves_ps_launch_failure(monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise OSError("ps executable unavailable")

    monkeypatch.setattr(RUNNER.subprocess, "run", unavailable)
    monkeypatch.setattr(
        RUNNER.os, "kill", lambda *_: pytest.fail("must not probe after ps failure")
    )
    with pytest.raises(OSError, match="ps executable unavailable"):
        RUNNER.process_identity(424242)


def test_frozen_sources_require_same_bytes_after_queue_wait(monkeypatch, tmp_path):
    repository = tmp_path / "repo"
    source = repository / "src/s1/evaluation/prefix.py"
    runner = repository / "scripts/benchmark_mps_prefix.py"
    source.parent.mkdir(parents=True)
    runner.parent.mkdir(parents=True)
    source.write_bytes(b"# frozen prefix implementation\n")
    runner.write_bytes(b"# frozen experiment implementation\n")
    monkeypatch.setattr(RUNNER, "ROOT", repository)
    files = [source, runner]
    expected = {
        str(path.relative_to(repository)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    assert RUNNER.validate_frozen_sources(files, expected) is None
    source.write_bytes(b"# different implementation while queued\n")
    with pytest.raises(RuntimeError, match="source changed while queued; no model was loaded"):
        RUNNER.validate_frozen_sources(files, expected)
