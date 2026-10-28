"""Resume orchestration uses process evidence and frozen sources, without models."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "resume_mps_benchmarks", ROOT / "scripts/resume_mps_benchmarks.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
IDENTITY = "Wed Oct 28 17:14:25 2026 S original-command"
BASE_RUNTIME = {
    "question_mode": "causal_multislot",
    "device": "mps",
    "dtype": "bfloat16",
    "precision": "bfloat16",
    "attn_implementation": "sdpa",
    "quantization": "none",
    "max_context": 16384,
    "torch_version": "2.8.0",
    "transformers_version": "5.17.0",
    "temperature": 2.7,
    "compile_mode": None,
    "max_forward_batch_size": 8,
    "max_prepared_prompts": 8,
    "adapter_path": None,
}
SUITE = {
    "name": "jevbench-mini",
    "version": "1",
    "files": {"tests.jsonl.gz": "frozen-plan-hash"},
    "plan": "tests.jsonl.gz",
}


def arguments(tmp_path):
    return SimpleNamespace(
        parent=tmp_path / "parent",
        output=tmp_path / "resume",
        coherence_root=tmp_path / "coherence",
        heaven_root=tmp_path / "heaven",
        predecessor_receipt=tmp_path / "external.json",
        original_pid=89007,
        wait_for_pid=124,
        poll_seconds=0.001,
    )


def test_waits_on_original_process_until_terminal(monkeypatch):
    observed = iter([IDENTITY, IDENTITY.replace(" S ", " R "), None])
    monkeypatch.setattr(RUNNER, "process_identity", lambda _pid: next(observed))
    sleeps, saves = [], []
    monkeypatch.setattr(RUNNER.time, "sleep", sleeps.append)
    handle = {"pid": 1, "identity": IDENTITY}
    RUNNER.wait_handle(handle, lambda: saves.append(handle["status"]), 2)
    assert saves == ["waiting_same_process", "waiting_same_process", "original_exited"]
    assert sleeps == [2, 2]


@pytest.mark.parametrize(
    "terminal",
    [None, IDENTITY.replace(" S ", " Z "), IDENTITY.replace("original-command", "new-command")],
)
def test_missing_zombie_or_reused_pid_proves_captured_original_terminal(monkeypatch, terminal):
    monkeypatch.setattr(RUNNER, "process_identity", lambda _pid: terminal)
    monkeypatch.setattr(
        RUNNER.time, "sleep", lambda _seconds: pytest.fail("terminal handle must not sleep")
    )
    handle = {"pid": 1, "identity": IDENTITY}
    RUNNER.wait_handle(handle, lambda: None, 1)
    assert handle["status"] in {
        "original_exited",
        "original_terminal_zombie",
        "original_ended_pid_reused",
    }


def test_ps_transient_errors_cannot_be_treated_as_absence(monkeypatch):
    monkeypatch.setattr(
        RUNNER.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=2, stdout="", stderr="temporary failure"
        ),
    )
    monkeypatch.setattr(
        RUNNER.os, "kill", lambda *_args: pytest.fail("bad ps cannot prove absence")
    )
    with pytest.raises(RuntimeError, match="observation failed"):
        RUNNER.process_identity(1)


def test_empty_ps_still_checks_process_liveness(monkeypatch):
    monkeypatch.setattr(
        RUNNER.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stdout="", stderr=""),
    )
    monkeypatch.setattr(RUNNER.os, "kill", lambda *_args: None)
    with pytest.raises(RuntimeError, match="process is live"):
        RUNNER.process_identity(1)

    def missing(*_args):
        raise ProcessLookupError()

    monkeypatch.setattr(RUNNER.os, "kill", missing)
    assert RUNNER.process_identity(1) is None


@pytest.mark.parametrize("status", ["initializing", "launching", "running", "error"])
def test_incomplete_supervisor_receipt_cannot_prove_child_stopped(tmp_path, status):
    path = tmp_path / "external.json"
    RUNNER.write_json(
        path,
        {
            "status": status,
            "supervisor_pid": 124,
            "child": {"pid": 2, "identity": IDENTITY, "exit_code": 0},
        },
    )
    with pytest.raises(RuntimeError, match="no terminal child receipt"):
        RUNNER.verify_external_child(path, lambda: None, {}, 1, expected_supervisor_pid=124)


def test_completed_supervisor_still_waits_on_actual_child(tmp_path, monkeypatch):
    path = tmp_path / "external.json"
    RUNNER.write_json(
        path,
        {
            "status": "completed",
            "supervisor_pid": 124,
            "child": {"pid": 2, "identity": IDENTITY, "exit_code": 0},
        },
    )
    observed = iter([IDENTITY, None])
    monkeypatch.setattr(RUNNER, "process_identity", lambda _pid: next(observed))
    sleeps = []
    monkeypatch.setattr(RUNNER.time, "sleep", sleeps.append)
    receipt = {}
    RUNNER.verify_external_child(path, lambda: None, receipt, 3, expected_supervisor_pid=124)
    assert sleeps == [3]
    assert receipt["external_child"]["status"] == "original_exited"


def test_terminal_child_receipt_from_different_supervisor_is_rejected(tmp_path):
    path = tmp_path / "wrong-supervisor.json"
    RUNNER.write_json(
        path,
        {
            "status": "completed",
            "supervisor_pid": 999,
            "child": {"pid": 2, "identity": IDENTITY, "exit_code": 0},
        },
    )
    with pytest.raises(RuntimeError, match="different supervisor"):
        RUNNER.verify_external_child(path, lambda: None, {}, 1, expected_supervisor_pid=124)


def test_supervisor_freezes_sources_before_wait_and_never_loads_over_live_jobs(
    tmp_path, monkeypatch
):
    args = arguments(tmp_path)
    RUNNER.write_json(
        args.predecessor_receipt,
        {
            "status": "completed",
            "supervisor_pid": 124,
            "child": {"pid": 125, "identity": IDENTITY, "exit_code": 0},
        },
    )
    observations = {
        89007: iter([IDENTITY, None, None]),
        124: iter([IDENTITY, IDENTITY, None]),
        125: iter([None]),
    }
    events = []
    monkeypatch.setattr(RUNNER, "process_identity", lambda pid: next(observations[pid]))
    monkeypatch.setattr(RUNNER.time, "sleep", lambda _seconds: events.append("wait"))

    def freeze(args, receipt):
        events.append("freeze")
        (args.output / "sources").mkdir()
        receipt["source_inventory"] = {}

    def run(args, receipt, save):
        events.append("model_load")
        assert receipt["original_baseline"]["status"] == "original_exited"
        assert receipt["predecessor"]["status"] == "original_exited"
        assert receipt["external_child"]["status"] == "original_exited"

    monkeypatch.setattr(RUNNER, "freeze_sources", freeze)
    monkeypatch.setattr(RUNNER, "run_stages", run)
    assert RUNNER.supervise(args) == 0
    assert events == ["freeze", "wait", "model_load"]
    receipt = json.loads((args.output / "receipt.json").read_text())
    assert receipt["status"] == "completed"


def test_process_observation_error_preserves_error_receipt_and_prevents_load(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    monkeypatch.setattr(RUNNER, "capture_handle", lambda pid: {"pid": pid, "identity": IDENTITY})

    def missing_package(name):
        raise RUNNER.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(RUNNER.importlib.metadata, "version", missing_package)

    def freeze(args, receipt):
        (args.output / "sources").mkdir()
        receipt["source_inventory"] = {}

    monkeypatch.setattr(RUNNER, "freeze_sources", freeze)

    def broken(_pid):
        raise RuntimeError("transient observation failure")

    monkeypatch.setattr(RUNNER, "process_identity", broken)
    monkeypatch.setattr(RUNNER, "run_stages", lambda *_args: pytest.fail("must not load"))
    with pytest.raises(RuntimeError, match="transient observation"):
        RUNNER.supervise(args)
    receipt = json.loads((args.output / "receipt.json").read_text())
    assert receipt["status"] == "error"
    assert receipt["error"]["message"] == "transient observation failure"
    assert receipt["packages"] == {"torch": None, "transformers": None, "numpy": None}


def test_all_additional_handles_are_captured_before_wait_and_terminal_before_load(
    tmp_path, monkeypatch
):
    args = arguments(tmp_path)
    args.additional_predecessor_pid = [44736, 999]
    RUNNER.write_json(
        args.predecessor_receipt,
        {
            "supervisor_pid": 124,
            "status": "completed",
            "child": {"pid": 125, "identity": IDENTITY, "exit_code": 0},
        },
    )
    events = []

    def capture(pid):
        events.append(f"capture-{pid}")
        return {"pid": pid, "identity": IDENTITY}

    def freeze(args, receipt):
        assert events == ["capture-89007", "capture-124", "capture-44736", "capture-999"]
        events.append("freeze")
        (args.output / "sources").mkdir()
        receipt["source_inventory"] = {}

    def wait(handle, save, seconds):
        assert events[:5] == [
            "capture-89007",
            "capture-124",
            "capture-44736",
            "capture-999",
            "freeze",
        ]
        handle["status"] = "original_exited"
        events.append(f"wait-{handle['pid']}")

    def run(args, receipt, save):
        assert all(
            handle["status"] == "original_exited" for handle in receipt["additional_predecessors"]
        )
        assert all(f"wait-{pid}" in events for pid in [89007, 124, 44736, 999, 125])
        events.append("model-load")

    monkeypatch.setattr(RUNNER, "capture_handle", capture)
    monkeypatch.setattr(RUNNER, "freeze_sources", freeze)
    monkeypatch.setattr(RUNNER, "wait_handle", wait)
    monkeypatch.setattr(RUNNER, "run_stages", run)
    assert RUNNER.supervise(args) == 0
    assert events[-1] == "model-load"


@pytest.mark.parametrize("additional", [[0], [-1], [44736, 44736], [124], [89007]])
def test_duplicate_or_nonpositive_additional_pids_rejected_before_capture(
    tmp_path, monkeypatch, additional
):
    args = arguments(tmp_path)
    args.additional_predecessor_pid = additional
    monkeypatch.setattr(
        RUNNER, "capture_handle", lambda pid: pytest.fail("invalid PID must never be observed")
    )
    with pytest.raises(ValueError, match="unique positive"):
        RUNNER.supervise(args)
    assert not args.output.exists()


def test_additional_pid_cli_is_repeatable_and_defaults_remain_compatible(monkeypatch):
    observed = []
    monkeypatch.setattr(RUNNER, "supervise", lambda args: observed.append(args) or 0)
    base = [
        "--parent",
        "parent",
        "--output",
        "fresh",
        "--coherence-root",
        "coherence",
        "--heaven-root",
        "heaven",
        "--predecessor-receipt",
        "receipt",
        "--wait-for-pid",
        "124",
        "--original-pid",
        "89007",
    ]
    assert RUNNER.main(base) == 0
    assert observed[-1].additional_predecessor_pid == []
    assert (
        RUNNER.main(
            [*base, "--additional-predecessor-pid", "44736", "--additional-predecessor-pid", "999"]
        )
        == 0
    )
    assert observed[-1].additional_predecessor_pid == [44736, 999]


def test_package_metadata_keeps_installed_versions_and_propagates_unexpected_errors(monkeypatch):
    monkeypatch.setattr(RUNNER.importlib.metadata, "version", lambda name: "recorded-" + name)
    assert RUNNER.package_version("torch") == "recorded-torch"

    def broken(_name):
        raise RuntimeError("metadata corruption")

    monkeypatch.setattr(RUNNER.importlib.metadata, "version", broken)
    with pytest.raises(RuntimeError, match="metadata corruption"):
        RUNNER.package_version("torch")


def test_frozen_snapshot_mutation_prevents_resume(tmp_path):
    output = tmp_path / "resume"
    path = output / "sources/src/s1/unified.py"
    path.parent.mkdir(parents=True)
    path.write_text("original")
    expected = RUNNER.inventory(output / "sources")
    path.write_text("mutated")
    with pytest.raises(ValueError, match="snapshot changed"):
        RUNNER.verify_frozen_sources(output, expected)


def test_repository_pin_rejected_before_any_archive(tmp_path, monkeypatch):
    responses = iter(["wrong-pin\n", ""])
    monkeypatch.setattr(
        RUNNER.subprocess, "check_output", lambda *_args, **_kwargs: next(responses)
    )
    target = tmp_path / "archived"
    with pytest.raises(ValueError, match="immutable pin"):
        RUNNER.archive_repository(tmp_path, target, RUNNER.COHERENCE_PIN)
    assert not target.exists()


def test_frozen_source_path_installed_before_any_s1_import_or_model_load(tmp_path, monkeypatch):
    for name in tuple(sys.modules):
        if name == "s1" or name.startswith("s1."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(sys, "path", list(sys.path))
    path = tmp_path / "sources/scripts/run_jevbench_public.py"
    path.parent.mkdir(parents=True)
    path.write_text("FROZEN = True\n")
    (path.parent / "summarize_jevbench_campaign.py").write_text("FROZEN = True\n")
    events = []

    def factory(*_args, **_kwargs):
        assert sys.path[0] == str(tmp_path / "sources/src")
        events.append("model_load")

    def load(name):
        assert sys.path[0] == str(tmp_path / "sources/src")
        events.append(name)
        return SimpleNamespace(GemmaBackend=factory)

    monkeypatch.setattr(RUNNER.importlib, "import_module", load)
    runtime = RUNNER.load_runtime(tmp_path)
    assert runtime.public.FROZEN is True
    runtime.backends.GemmaBackend()
    assert events == ["s1.backends", "s1.contracts", "torch", "jevbench", "model_load"]


def fake_mps(*, fail_trim=False, fail_on_trim=()):
    state = {"driver": 80, "calls": [], "trims": 0}

    def operation(name, value=None):
        state["calls"].append(name)
        return value

    def trim():
        operation("empty_cache")
        state["trims"] += 1
        if fail_trim or state["trims"] in fail_on_trim:
            raise RuntimeError("trim failed")
        state["driver"] = 24

    return SimpleNamespace(
        synchronize=lambda: operation("synchronize"),
        current_allocated_memory=lambda: operation("current", 20),
        driver_allocated_memory=lambda: operation("driver", state["driver"]),
        recommended_max_memory=lambda: operation("recommended", 100),
        empty_cache=trim,
        state=state,
    )


def fake_runtime(tmp_path, events, *, fail_coherence=False, fail_prediction=False, mps=None):
    class Backend:
        name = "frozen-checkpoint"
        metadata = BASE_RUNTIME

        def __init__(self, *args, **kwargs):
            assert kwargs == {
                "device": "mps",
                "dtype": "bfloat16",
                "attn_implementation": "sdpa",
                "question_mode": "causal_multislot",
            }
            events.append("load")
            self.model = SimpleNamespace(device="mps", temperature=2.7)

        def predict(self, request):
            if fail_prediction:
                raise RuntimeError("original inference OOM")
            return {"answers": {"decision": {"choice": "a"}}}

    def evaluate(model, **kwargs):
        events.append(f"{model.name}:coherence-mini")
        assert kwargs["suite"] == "jevbench-mini" and kwargs["concurrency"] == 1
        cache = Path(kwargs["cache"])
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "new.json").write_text("{}")
        assert model.fn("state", {"decision": {"type": "choice"}}) == {"decision": {"choice": "a"}}
        if fail_coherence:
            raise RuntimeError("interrupted evaluation")
        return SimpleNamespace(save=lambda path: RUNNER.write_json(path, {"complete": True}))

    def public(predict, heaven, output, identity):
        events.append("independent:public231")
        assert identity["profile"] == "independent"
        predict("state", {"decision": {"type": "choice"}})
        RUNNER.write_json(output / "manifest.json", {"status": "completed"})

    return SimpleNamespace(
        backends=SimpleNamespace(
            GemmaBackend=Backend, normalize_response=lambda request, response: response
        ),
        contracts=SimpleNamespace(
            DecisionRequest=SimpleNamespace(model_validate=lambda value: value)
        ),
        torch=SimpleNamespace(mps=mps or fake_mps()),
        jb=SimpleNamespace(
            from_callable=lambda fn, name: SimpleNamespace(fn=fn, name=name),
            evaluate=evaluate,
            load_suite=lambda name: SimpleNamespace(
                record=lambda: {key: value for key, value in SUITE.items() if key != "plan"}
            ),
        ),
        public=SimpleNamespace(run_public=public),
        auditor=SimpleNamespace(replay_coherence=lambda *args: {"raw_cache_files": 1248}),
    )


def stage_setup(tmp_path, monkeypatch, *, complete_public=False, fail_coherence=False):
    args = arguments(tmp_path)
    args.output.mkdir()
    events = []
    monkeypatch.setattr(
        RUNNER,
        "load_runtime",
        lambda _output: fake_runtime(tmp_path, events, fail_coherence=fail_coherence),
    )
    monkeypatch.setattr(RUNNER, "verify_checkpoint", lambda _checkpoint: None)
    monkeypatch.setattr(
        RUNNER, "audit_public", lambda root, *_args, **_kwargs: {"path": str(root), "audited": True}
    )
    monkeypatch.setattr(
        RUNNER, "audit_coherence", lambda path, **kwargs: {"path": str(path), "audited": True}
    )
    if complete_public:
        RUNNER.write_json(
            args.parent / "independent/public231/manifest.json", {"status": "completed"}
        )
    cache = args.parent / "causal_multislot/coherence-cache"
    cache.mkdir(parents=True)
    (cache / "old.json").write_text('{"old":true}')
    receipt = {
        "checkpoint": {"path": "/fake-checkpoint", "manifest_sha256": "pinned"},
        "runtime_expected": copy.deepcopy(BASE_RUNTIME),
        "packages": {"torch": "2.8.0", "transformers": "5.17.0"},
        "source_inventory": {},
        "stages": [],
    }
    return args, receipt, events


def test_stage_order_and_original_cache_epochs_are_preserved(tmp_path, monkeypatch):
    args, receipt, events = stage_setup(tmp_path, monkeypatch)
    RUNNER.run_stages(args, receipt, lambda: None)
    assert events == [
        "load",
        "causal_multislot:coherence-mini",
        "independent:public231",
        "independent:coherence-mini",
    ]
    assert [stage["status"] for stage in receipt["stages"]] == ["completed"] * 3
    first = receipt["stages"][0]
    assert list(first["cache_before"]) == ["old.json"]
    assert set(first["cache_after"]) == {"old.json", "new.json"}
    assert first["cache_after"]["old.json"] == first["cache_before"]["old.json"]
    assert first["model_calls"] == 1
    trace = args.output / "causal_multislot/coherence-mini-model-calls.jsonl"
    assert len(trace.read_text().splitlines()) == 1
    assert not (args.parent / "causal_multislot/coherence-mini.json").exists()


def test_completed_public_result_is_audited_and_skipped_without_overwrite(tmp_path, monkeypatch):
    args, receipt, events = stage_setup(tmp_path, monkeypatch, complete_public=True)
    original = args.parent / "independent/public231/manifest.json"
    before = original.read_bytes()
    RUNNER.run_stages(args, receipt, lambda: None)
    assert events == ["load", "causal_multislot:coherence-mini", "independent:coherence-mini"]
    assert receipt["stages"][1]["status"] == "skipped_completed_audited"
    assert original.read_bytes() == before
    assert not (args.output / "independent/public231").exists()


def test_stage_error_preserves_callback_and_cache_receipts_without_score(tmp_path, monkeypatch):
    args, receipt, events = stage_setup(tmp_path, monkeypatch, fail_coherence=True)
    with pytest.raises(RuntimeError, match="interrupted evaluation"):
        RUNNER.run_stages(args, receipt, lambda: None)
    assert events == ["load", "causal_multislot:coherence-mini"]
    stage = receipt["stages"][0]
    assert stage["status"] == "error" and stage["model_calls"] == 1
    assert stage["error"]["message"] == "interrupted evaluation"
    assert "new.json" in stage["cache_after"]
    assert not (args.output / "causal_multislot/coherence-mini.json").exists()


def test_partial_public_without_manifest_is_preserved_and_rerun_fresh(tmp_path, monkeypatch):
    args, receipt, events = stage_setup(tmp_path, monkeypatch)
    partial = args.parent / "independent/public231/records.jsonl"
    partial.parent.mkdir(parents=True)
    partial.write_text("original partial bytes\n")
    RUNNER.run_stages(args, receipt, lambda: None)
    assert "independent:public231" in events
    assert partial.read_text() == "original partial bytes\n"
    assert (args.output / "independent/public231/manifest.json").is_file()
    assert "preserved_partial_original" in receipt["stages"][1]


@pytest.mark.parametrize(
    "change",
    [
        lambda result, scores: result.update(errors=["model error"]),
        lambda result, scores: result.update(truncated_by_budget=True),
        lambda result, scores: result.update(backend_down=True),
        lambda result, scores: scores.update(n_checks=49),
        lambda result, scores: scores.update(n_outcomes=1199),
        lambda result, scores: scores.update(min_coverage=0.99),
        lambda result, scores: scores.update(incomplete=1),
        lambda result, scores: scores["checks"]["relation-0/tv"].update(coverage=23 / 24, n=23),
        lambda result, scores: result["planned_tests"].update({"relation-0": 23}),
    ],
)
def test_coherence_completion_requires_exact_population_and_no_errors(tmp_path, change):
    # Boundary fixture follows the genuine pinned schema: variants/question
    # outcomes can exceed the 1,200 planned/judged target tests.
    result = {
        "backend": "causal_multislot",
        "suite": SUITE,
        "seed": 0,
        "repeats": 1,
        "trials": 1,
        "selection": {
            "domains": None,
            "dimensions": None,
            "groups": None,
            "relations": None,
            "diagnostics": False,
        },
        "errors": [],
        "truncated_by_budget": False,
        "backend_down": False,
        "planned_tests": {f"relation-{index}": 24 for index in range(50)},
        "outcomes": [{} for _ in range(1468)],
    }
    scores = {
        "backend": "causal_multislot",
        "n_checks": 50,
        "n_cases": 240,
        "n_outcomes": 1468,
        "min_coverage": 0.95,
        "incomplete": [],
        "skipped_outcomes": 0,
        "checks": {
            f"relation-{index}/tv": {"n": 24, "planned": 24, "coverage": 1.0} for index in range(50)
        },
    }
    settings = {"tolerance": 0.05, "min_n": 5, "bootstrap": 1000, "min_coverage": 0.95}
    path = tmp_path / "report.json"
    RUNNER.write_json(
        path, {"name": "causal_multislot", "result": result, "scores": scores, "settings": settings}
    )
    audit = RUNNER.audit_coherence(path, expected_profile="causal_multislot", expected_suite=SUITE)
    assert audit["n_outcomes"] == 1468
    assert audit["planned_tests"] == audit["judged_tests"] == 1200
    change(result, scores)
    RUNNER.write_json(
        path, {"name": "causal_multislot", "result": result, "scores": scores, "settings": settings}
    )
    with pytest.raises(ValueError, match="identity|incomplete or contains errors"):
        RUNNER.audit_coherence(path, expected_profile="causal_multislot", expected_suite=SUITE)


@pytest.mark.parametrize(
    "relative",
    [
        "artifacts/jevbench/20261008-public-01/s1-bf16/coherence-mini.json",
        "artifacts/jevbench/20261008-public-02/s1-independent/coherence-mini.json",
    ],
)
def test_genuine_saved_full_mini_report_passes_but_missing_achieved_support_fails(
    tmp_path, relative
):
    source = ROOT / relative
    if not source.is_file():
        pytest.skip("historical raw research artifact unavailable in clean checkout")
    report = json.loads(source.read_text())
    expected = {"expected_profile": report["name"], "expected_suite": report["result"]["suite"]}
    audit = RUNNER.audit_coherence(source, **expected)
    assert audit["n_outcomes"] == 1468
    assert audit["planned_tests"] == audit["judged_tests"] == 1200
    assert audit["configured_min_coverage"] == 0.95
    assert audit["achieved_check_coverage"] == 1.0
    first = next(iter(report["scores"]["checks"].values()))
    first.update(n=23, coverage=23 / 24)
    damaged = tmp_path / "missing-support.json"
    RUNNER.write_json(damaged, report)
    with pytest.raises(ValueError, match="incomplete or contains errors"):
        RUNNER.audit_coherence(damaged, **expected)


def test_checkpoint_requires_manifest_members_to_equal_parent_receipt(tmp_path):
    root = tmp_path / "checkpoint"
    root.mkdir()
    rows = []
    for name in ["model.safetensors", "config.json", "s1_config.json"]:
        path = root / name
        path.write_text("tiny test bytes")
        rows.append(
            {"path": name, "sha256": RUNNER.digest(path), "size_bytes": path.stat().st_size}
        )
    manifest = root / "artifact-manifest.json"
    RUNNER.write_json(manifest, {"files": rows})
    checkpoint = {
        "path": str(root),
        "manifest_sha256": RUNNER.digest(manifest),
        "verified_files": rows,
    }
    RUNNER.verify_checkpoint(checkpoint)
    checkpoint["verified_files"] = rows[:-1]
    with pytest.raises(ValueError, match="member identity"):
        RUNNER.verify_checkpoint(checkpoint)
    checkpoint["verified_files"] = rows
    (root / "model.safetensors").write_text("tiny test BYTEs")
    with pytest.raises(ValueError, match="member differs"):
        RUNNER.verify_checkpoint(checkpoint)


@pytest.mark.parametrize(
    "field,value",
    [
        ("dtype", "float32"),
        ("precision", "float32"),
        ("attn_implementation", "eager"),
        ("quantization", "int8"),
        ("max_context", 8192),
        ("device", "cuda"),
    ],
)
def test_parent_runtime_mismatch_prevents_any_model_load(tmp_path, monkeypatch, field, value):
    args, receipt, events = stage_setup(tmp_path, monkeypatch)
    receipt["runtime_expected"][field] = value
    with pytest.raises(ValueError, match="before model load"):
        RUNNER.run_stages(args, receipt, lambda: None)
    assert events == []


def test_different_torch_version_prevents_model_load_before_cache_reuse(tmp_path, monkeypatch):
    args, receipt, events = stage_setup(tmp_path, monkeypatch)
    receipt["packages"]["torch"] = "2.10.0"
    with pytest.raises(ValueError, match="package versions differ"):
        RUNNER.run_stages(args, receipt, lambda: None)
    assert events == []


def test_loaded_runtime_policy_mismatch_prevents_all_evaluations(tmp_path, monkeypatch):
    args, receipt, events = stage_setup(tmp_path, monkeypatch)
    runtime = fake_runtime(tmp_path, events)
    runtime.backends.GemmaBackend.metadata = {**BASE_RUNTIME, "temperature": 7.0}
    monkeypatch.setattr(RUNNER, "load_runtime", lambda _output: runtime)
    with pytest.raises(ValueError, match="resume runtime differs"):
        RUNNER.run_stages(args, receipt, lambda: None)
    assert events == ["load"]
    assert receipt["stages"] == []


def test_complete_report_cannot_be_relabelled_as_another_profile(tmp_path):
    source = ROOT / "artifacts/jevbench/20261008-public-01/s1-bf16/coherence-mini.json"
    if not source.is_file():
        pytest.skip("historical raw research artifact unavailable in clean checkout")
    report = json.loads(source.read_text())
    expected = {"expected_profile": report["name"], "expected_suite": report["result"]["suite"]}
    for field in ["name", "result", "scores"]:
        altered = copy.deepcopy(report)
        if field == "name":
            altered["name"] = "another-profile"
        else:
            altered[field]["backend"] = "another-profile"
        path = tmp_path / f"wrong-{field}.json"
        RUNNER.write_json(path, altered)
        with pytest.raises(ValueError, match="identity|incomplete or contains errors"):
            RUNNER.audit_coherence(path, **expected)


def test_raw_replay_failure_prevents_accepting_header_complete_report():
    source = ROOT / "artifacts/jevbench/20261008-public-01/s1-bf16/coherence-mini.json"
    if not source.is_file():
        pytest.skip("historical raw research artifact unavailable in clean checkout")
    report = json.loads(source.read_text())

    def mismatch(*_args):
        raise ValueError("scores differ from raw-answer replay")

    with pytest.raises(ValueError, match="scores differ from raw-answer replay"):
        RUNNER.audit_coherence(
            source,
            expected_profile=report["name"],
            expected_suite=report["result"]["suite"],
            replay=mismatch,
        )


def schema_report(*, partial=False):
    """Pinned metadata shape; target tests and raw variant outcomes stay distinct."""
    planned = {f"relation-{index}": 24 for index in range(50)}
    return {
        "name": "causal_multislot",
        "settings": {"tolerance": 0.05, "min_n": 5, "bootstrap": 1000, "min_coverage": 0.95},
        "result": {
            "backend": "causal_multislot",
            "suite": SUITE,
            "seed": 0,
            "repeats": 1,
            "trials": 1,
            "selection": {
                "domains": None,
                "dimensions": None,
                "groups": None,
                "relations": None,
                "diagnostics": False,
            },
            "planned_tests": planned,
            "errors": [{"error": "original MPS OOM"}] if partial else [],
            "backend_down": partial,
            "truncated_by_budget": False,
            "outcomes": [{} for _ in range(236 if partial else 1468)],
        },
        "scores": {
            "backend": "causal_multislot",
            "n_cases": 240,
            "n_checks": 0 if partial else 50,
            "n_outcomes": 236 if partial else 1468,
            "skipped_outcomes": 0,
            "min_coverage": 0.95,
            "incomplete": list(planned) if partial else [],
            "overall": None if partial else 0.4,
            "checks": {
                f"{name}/tv": {
                    "planned": 24,
                    "n": 0 if partial else 24,
                    "coverage": 0.0 if partial else 1.0,
                }
                for name in planned
            },
        },
    }


def test_allocator_maintenance_preserves_active_allocations_and_records_driver_trim():
    mps = fake_mps()
    record = RUNNER.maintain_mps_allocator(SimpleNamespace(mps=mps))
    assert mps.state["calls"] == [
        "synchronize",
        "current",
        "driver",
        "recommended",
        "empty_cache",
        "synchronize",
        "current",
        "driver",
        "recommended",
    ]
    assert record["before"]["driver_allocated_bytes"] == 80
    assert record["after"]["driver_allocated_bytes"] == 24
    assert (
        record["before"]["current_allocated_bytes"]
        == record["after"]["current_allocated_bytes"]
        == 20
    )
    assert (
        record["before"]["recommended_max_bytes"] == record["after"]["recommended_max_bytes"] == 100
    )
    assert record["policy"]["limits_changed"] is False
    assert record["error"] is None


def test_every_callback_and_stage_boundary_records_explicit_allocator_epoch_policy(
    tmp_path, monkeypatch
):
    args, receipt, events = stage_setup(tmp_path, monkeypatch)
    RUNNER.run_stages(args, receipt, lambda: None)
    assert receipt["allocator_maintenance_policy"] == RUNNER.ALLOCATOR_POLICY
    for stage in receipt["stages"]:
        assert stage["allocator_maintenance_before"]["error"] is None
        assert stage["allocator_maintenance_after"]["error"] is None
        trace = args.output / stage["profile"] / f"{stage['kind']}-model-calls.jsonl"
        row = json.loads(trace.read_text().splitlines()[0])
        assert row["allocator_maintenance"]["policy"] == RUNNER.ALLOCATOR_POLICY
        assert row["latency_ms"] >= row["allocator_maintenance"]["wall_ms"]


def test_inference_error_survives_callback_and_boundary_trim_errors(tmp_path, monkeypatch):
    args, receipt, events = stage_setup(tmp_path, monkeypatch)
    runtime = fake_runtime(
        tmp_path, events, fail_prediction=True, mps=fake_mps(fail_on_trim={2, 3})
    )
    monkeypatch.setattr(RUNNER, "load_runtime", lambda output: runtime)
    with pytest.raises(RUNNER.EpochFailure, match="original inference OOM") as failure:
        RUNNER.run_stages(args, receipt, lambda: None)
    assert failure.value.evidence["error"]["type"] == "RuntimeError"
    stage = receipt["stages"][0]
    assert stage["error"]["message"] == "original inference OOM"
    assert stage["allocator_maintenance_after"]["error"]["message"] == "trim failed"
    row = json.loads(
        (args.output / "causal_multislot/coherence-mini-model-calls.jsonl").read_text()
    )
    assert row["error"]["message"] == "original inference OOM"
    assert row["allocator_maintenance"]["error"]["message"] == "trim failed"


def test_partial_coherence_is_preserved_and_resumed_into_new_report_with_existing_cache(
    tmp_path, monkeypatch
):
    args, receipt, events = stage_setup(tmp_path, monkeypatch)
    original = args.parent / "causal_multislot/coherence-mini.json"
    RUNNER.write_json(original, schema_report(partial=True))
    before = original.read_bytes()
    RUNNER.run_stages(args, receipt, lambda: None)
    stage = receipt["stages"][0]
    assert stage["status"] == "completed"
    assert stage["preserved_partial_original"]["errors"] == [{"error": "original MPS OOM"}]
    assert stage["preserved_partial_original"]["n_outcomes"] == 236
    assert original.read_bytes() == before
    assert (args.output / "causal_multislot/coherence-mini.json").is_file()
    assert stage["cache_before"]["old.json"] == stage["cache_after"]["old.json"]
    assert events[1] == "causal_multislot:coherence-mini"


def test_invalid_nominally_full_coherence_fails_closed_instead_of_becoming_partial(tmp_path):
    report = schema_report()
    report["result"]["outcomes"] = []
    report["scores"]["n_outcomes"] = 0
    path = tmp_path / "corrupt-complete.json"
    RUNNER.write_json(path, report)
    with pytest.raises(ValueError, match="incomplete or contains errors"):
        RUNNER.inspect_original_coherence(
            path, expected_profile="causal_multislot", expected_suite=SUITE
        )


def test_partial_coherence_with_wrong_profile_is_not_reused(tmp_path):
    report = schema_report(partial=True)
    report["name"] = report["result"]["backend"] = report["scores"]["backend"] = "independent"
    path = tmp_path / "foreign-partial.json"
    RUNNER.write_json(path, report)
    with pytest.raises(ValueError, match="execution identity"):
        RUNNER.inspect_original_coherence(
            path, expected_profile="causal_multislot", expected_suite=SUITE
        )


def test_epoch_checkpoint_latches_and_threadpool_drains_without_gpu_or_model_errors():
    budget = RUNNER.EpochBudget(max_new_calls=2, max_driver_fraction=0.8)
    mps = fake_mps()
    mps.state["driver"] = 10
    torch = SimpleNamespace(mps=mps)
    model_calls, cached, model_errors = [], [], []

    def callback(index):
        try:
            budget.before_new_call(torch)
            model_calls.append(index)
            cached.append(index)  # Success has returned into the cache before next callback.
            return index
        except Exception as exc:
            model_errors.append(str(exc))
            raise

    with ThreadPoolExecutor(max_workers=1) as pool:
        futures = [pool.submit(callback, index) for index in range(20)]
        checkpoints = []
        for future in futures:
            try:
                future.result()
            except RUNNER.EpochCheckpoint as stop:
                checkpoints.append(stop.evidence)
    assert model_calls == cached == [0, 1]
    assert model_errors == []
    assert len(checkpoints) == 18 and all(
        stop["reason"] == "new_model_call_budget" for stop in checkpoints
    )
    assert mps.state["calls"].count("synchronize") == 2
    assert budget.new_calls == 2


def test_driver_budget_latches_before_any_dispatch_and_never_reobserves_gpu():
    budget = RUNNER.EpochBudget(max_new_calls=20, max_driver_fraction=0.8)
    mps = fake_mps()  # Driver80, recommended100: stop at the declared fraction.
    with pytest.raises(RUNNER.EpochCheckpoint) as stop:
        budget.before_new_call(SimpleNamespace(mps=mps))
    before = list(mps.state["calls"])
    assert stop.value.evidence["reason"] == "driver_memory_fraction"
    for _ in range(3):
        with pytest.raises(RUNNER.EpochCheckpoint):
            budget.before_new_call(SimpleNamespace(mps=mps))
    assert mps.state["calls"] == before and budget.new_calls == 0


@pytest.mark.parametrize("failure", ["synchronize", "recommended"])
def test_pre_dispatch_observation_error_latches_without_retrying_gpu(failure):
    budget = RUNNER.EpochBudget()
    mps = fake_mps()
    if failure == "synchronize":

        def fail_sync():
            mps.state["calls"].append("failed_synchronize")
            raise RuntimeError("memory observation failed")

        mps.synchronize = fail_sync
    else:
        mps.recommended_max_memory = lambda: 0
    with pytest.raises(RUNNER.EpochFailure) as stop:
        budget.before_new_call(SimpleNamespace(mps=mps))
    before = list(mps.state["calls"])
    assert stop.value.evidence["scope"] == "pre_dispatch_allocator_observation"
    for _ in range(3):
        with pytest.raises(RUNNER.EpochFailure) as queued:
            budget.before_new_call(SimpleNamespace(mps=mps))
        assert queued.value.evidence == stop.value.evidence
    assert mps.state["calls"] == before and budget.new_calls == 0


@pytest.mark.parametrize("failure", ["inference", "cleanup"])
def test_real_callback_failure_latches_and_threadpool_drains_without_scored_errors(
    tmp_path, monkeypatch, failure
):
    args, receipt, events = stage_setup(tmp_path, monkeypatch)
    args.max_new_calls = 1
    runtime = fake_runtime(
        tmp_path,
        events,
        fail_prediction=failure == "inference",
        mps=fake_mps(fail_on_trim={2} if failure == "cleanup" else ()),
    )
    model_errors, dispatches, stops = [], [], []
    backend_predict = runtime.backends.GemmaBackend.predict

    def predict(backend, request):
        dispatches.append(request)
        return backend_predict(backend, request)

    monkeypatch.setattr(runtime.backends.GemmaBackend, "predict", predict)

    def evaluate(model, **kwargs):
        def callback(index):
            try:
                return model.fn({"index": index}, {"decision": {"type": "choice"}})
            except Exception as exc:  # The official engine catches this family only.
                model_errors.append(str(exc))
                raise

        with ThreadPoolExecutor(max_workers=1) as pool:
            futures = [pool.submit(callback, index) for index in range(12)]
            for future in futures:
                try:
                    future.result()
                except RUNNER.EpochFailure as stop:
                    stops.append(stop)
        raise stops[0]

    runtime.jb.evaluate = evaluate
    monkeypatch.setattr(RUNNER, "load_runtime", lambda output: runtime)
    expected = "original inference OOM" if failure == "inference" else "trim failed"
    with pytest.raises(RUNNER.EpochFailure, match=expected):
        RUNNER.run_stages(args, receipt, lambda: None)
    stage = receipt["stages"][0]
    assert stage["status"] == "error" and stage["error"]["message"] == expected
    assert len(dispatches) == stage["model_calls"] == receipt["epoch_budget"]["new_calls"] == 1
    assert len(stops) == 12 and all(stop.evidence == stops[0].evidence for stop in stops)
    assert model_errors == []
    rows = (
        (args.output / "causal_multislot/coherence-mini-model-calls.jsonl").read_text().splitlines()
    )
    assert len(rows) == 1
    row = json.loads(rows[0])
    if failure == "cleanup":
        assert row["error"] is None and row["response"] is not None
        assert row["allocator_maintenance"]["error"]["message"] == "trim failed"
    else:
        assert row["error"]["message"] == expected
    assert not (args.output / "causal_multislot/coherence-mini.json").exists()


def test_stage_final_cleanup_failure_promotes_resource_checkpoint_to_error(tmp_path, monkeypatch):
    args, receipt, events = stage_setup(tmp_path, monkeypatch)
    args.max_new_calls = 1
    runtime = fake_runtime(tmp_path, events, mps=fake_mps(fail_on_trim={3}))

    def evaluate(model, **kwargs):
        model.fn("first", {"decision": {"type": "choice"}})
        model.fn("next", {"decision": {"type": "choice"}})
        pytest.fail("second request must checkpoint before dispatch")

    runtime.jb.evaluate = evaluate
    monkeypatch.setattr(RUNNER, "load_runtime", lambda output: runtime)
    with pytest.raises(RUNNER.EpochFailure, match="trim failed") as stop:
        RUNNER.run_stages(args, receipt, lambda: None)
    assert stop.value.evidence["scope"] == "stage_allocator_maintenance"
    stage = receipt["stages"][0]
    assert stage["status"] == "error" and stage["model_calls"] == 1
    assert stage["checkpoint"]["reason"] == "new_model_call_budget"
    assert stage["error"]["message"] == "trim failed"


def test_public_response_cache_keeps_first_success_and_origin_immutable(tmp_path):
    identity = {"source": "fixed", "runtime": {"precision": "bfloat16"}, "profile": "independent"}
    cache = RUNNER.PublicResponseCache(tmp_path, identity)
    request = {"state": "observed state", "questions": {"q": {"criteria": {"a": "A", "b": "B"}}}}
    response = {"answers": {"q": {"probabilities": {"a": 0.75, "b": 0.25}, "choice": "a"}}}

    def validate(value):
        return value["answers"]["q"]

    path = cache.store(
        request, response, {"epoch": "first", "inference_plus_maintenance_ms": 10}, validate
    )
    original = path.read_bytes()
    loaded = cache.read(request, validate)
    assert loaded["response"] == response and loaded["origin"]["epoch"] == "first"
    cache.store(request, response, {"epoch": "second"}, validate)
    assert path.read_bytes() == original
    changed = {"answers": {"q": {"probabilities": {"a": 0.2, "b": 0.8}, "choice": "b"}}}
    with pytest.raises(ValueError, match="cannot be replaced"):
        cache.store(request, changed, {"epoch": "third"}, validate)
    assert path.read_bytes() == original
    foreign = RUNNER.PublicResponseCache(tmp_path, {**identity, "source": "changed-source"})
    assert foreign.read(request, validate) is None


def test_public_cache_rejects_invalid_response_tampering_and_preserves_candidate_order(tmp_path):
    cache = RUNNER.PublicResponseCache(tmp_path, {"source": "fixed"})
    request = {"state": "S", "questions": {"q": {"criteria": {"a": "A", "b": "B"}}}}

    def reject(value):
        raise ValueError("invalid native probability distribution")

    with pytest.raises(ValueError, match="invalid native"):
        cache.store(request, {"bad": True}, {}, reject)
    assert not cache.path(request).exists()
    cache.store(request, {"answers": {}}, {}, lambda value: None)
    row = json.loads(cache.path(request).read_text())
    row["response"] = {"tampered": True}
    RUNNER.write_json(cache.path(request), row)
    with pytest.raises(ValueError, match="identity/content differs"):
        cache.read(request, lambda value: None)
    reordered = copy.deepcopy(request)
    reordered["questions"]["q"]["criteria"] = {"b": "B", "a": "A"}
    assert cache.path(request) != cache.path(reordered)


def test_epoch_checkpoint_is_terminal_control_receipt_not_model_error(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    monkeypatch.setattr(RUNNER, "capture_handle", lambda pid: {"pid": pid, "identity": None})
    monkeypatch.setattr(RUNNER, "wait_handle", lambda *args: None)
    monkeypatch.setattr(RUNNER, "verify_external_child", lambda *args, **kwargs: None)

    def freeze(args, receipt):
        (args.output / "sources").mkdir()
        receipt["source_inventory"] = {}

    monkeypatch.setattr(RUNNER, "freeze_sources", freeze)

    def checkpoint(*args):
        raise RUNNER.EpochCheckpoint({"reason": "new_model_call_budget", "new_calls": 20})

    monkeypatch.setattr(RUNNER, "run_stages", checkpoint)
    assert RUNNER.supervise(args) == 75
    receipt = json.loads((args.output / "receipt.json").read_text())
    assert receipt["status"] == "checkpointed" and "error" not in receipt


def test_official_style_queued_callbacks_checkpoint_with_prior_success_cached(
    tmp_path, monkeypatch
):
    args, receipt, events = stage_setup(tmp_path, monkeypatch)
    args.max_new_calls = 1
    runtime = fake_runtime(tmp_path, events)
    model_errors = []

    def evaluate(model, **kwargs):
        cache = Path(kwargs["cache"])

        def request(index):
            try:
                response = model.fn({"index": index}, {"decision": {"type": "choice"}})
                RUNNER.write_json(cache / f"success-{index}.json", response)
                return response
            except Exception as exc:
                model_errors.append(str(exc))
                raise

        with ThreadPoolExecutor(max_workers=1) as pool:
            futures = [pool.submit(request, index) for index in range(12)]
            for future in futures:
                future.result()
        pytest.fail("checkpoint must escape, not publish a partial scored report")

    runtime.jb.evaluate = evaluate
    monkeypatch.setattr(RUNNER, "load_runtime", lambda output: runtime)
    with pytest.raises(RUNNER.EpochCheckpoint):
        RUNNER.run_stages(args, receipt, lambda: None)
    stage = receipt["stages"][0]
    assert stage["status"] == "checkpointed" and "error" not in stage
    assert stage["model_calls"] == receipt["epoch_budget"]["new_calls"] == 1
    assert "success-0.json" in stage["cache_after"]
    assert not any(key.startswith("success-1") for key in stage["cache_after"])
    assert model_errors == []
    assert not (args.output / "causal_multislot/coherence-mini.json").exists()


def test_public_response_cache_survives_epoch_boundary_without_new_public_inference(
    tmp_path, monkeypatch
):
    args, receipt, events = stage_setup(tmp_path, monkeypatch)
    args.max_new_calls = 2
    original_receipt = copy.deepcopy(receipt)
    with pytest.raises(RUNNER.EpochCheckpoint):
        RUNNER.run_stages(args, receipt, lambda: None)
    assert receipt["stages"][1]["model_calls"] == 1
    cache_files = list((tmp_path / "immutable-public-responses").rglob("*.json"))
    assert len(cache_files) == 1
    cached_bytes = cache_files[0].read_bytes()
    later = copy.copy(args)
    later.output = tmp_path / "second-epoch"
    later.output.mkdir()
    RUNNER.run_stages(later, original_receipt, lambda: None)
    public = original_receipt["stages"][1]
    assert public["model_calls"] == 0 and public["public_cache_hits"] == 1
    assert cache_files[0].read_bytes() == cached_bytes
    assert not (later.output / "independent/public231-model-calls.jsonl").exists()
    event = json.loads((later.output / "independent/public231-cache-events.jsonl").read_text())
    assert event["origin"] == "immutable_first_success_response_cache"
    assert event["original_inference"]["epoch_output"] == str(args.output)
    assert "not fresh inference" in event["latency_scope"]
