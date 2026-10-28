"""Sequential child-process epoch supervision, with no models or real children."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "supervise_epochs", ROOT / "scripts/supervise_mps_benchmark_epochs.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def setup(tmp_path, monkeypatch, outcomes):
    resume = RUNNER.load_resume()
    args = SimpleNamespace(
        parent=tmp_path / "parent",
        output=tmp_path / "supervisor",
        coherence_root=tmp_path / "coherence",
        heaven_root=tmp_path / "heaven",
        predecessor_receipt=tmp_path / "external.json",
        original_pid=89007,
        wait_for_pid=124,
        additional_predecessor_pid=[44736],
        poll_seconds=0.01,
        max_new_calls=2,
        max_driver_fraction=0.8,
        max_epochs=3,
        public_cache_root=None,
    )
    resume.write_json(args.parent / "identity.json", {"original": "immutable"})
    checkpoint = tmp_path / "checkpoint"
    resume.write_json(checkpoint / "artifact-manifest.json", {"files": []})
    context = {"events": [], "live": False, "launched": 0}
    identity = {
        "path": str(checkpoint),
        "manifest_sha256": resume.digest(checkpoint / "artifact-manifest.json"),
        "verified_files": [],
    }

    def capture(pid):
        context["events"].append(f"capture-{pid}")
        return {"pid": pid, "identity": None}

    def wait(handle, save, seconds):
        context["events"].append(f"wait-{handle['pid']}")
        handle["status"] = "original_exited"

    def freeze(args, receipt):
        path = args.output / "sources/scripts/resume_mps_benchmarks.py"
        path.parent.mkdir(parents=True)
        path.write_text("frozen resume executor")
        receipt.update(
            source_inventory=resume.inventory(args.output / "sources"), checkpoint=identity
        )
        context["child_sources"] = copy.deepcopy(receipt["source_inventory"])
        context["events"].append("freeze")

    monkeypatch.setattr(resume, "capture_handle", capture)
    monkeypatch.setattr(resume, "wait_handle", wait)
    monkeypatch.setattr(resume, "freeze_sources", freeze)
    monkeypatch.setattr(
        resume,
        "verify_external_child",
        lambda *args, **kwargs: context["events"].append("external-child-terminal"),
    )
    monkeypatch.setattr(resume, "process_identity", lambda pid: "captured original child identity")
    monkeypatch.setattr(RUNNER, "load_resume", lambda: resume)

    class Child:
        def __init__(self, argv, **kwargs):
            assert not context["live"], "replacement child started before previous terminal wait"
            assert "external-child-terminal" in context["events"]
            context["live"] = True
            context["launched"] += 1
            self.index = context["launched"]
            self.pid = 100 + self.index
            self.argv = argv
            self.code = None
            context["events"].append(f"launch-{self.index}")

        def wait(self):
            context["events"].append(f"wait-child-{self.index}")
            outcome = outcomes[self.index - 1]
            self.code = outcome["code"]
            output = Path(self.argv[self.argv.index("--output") + 1])
            if outcome.get("progress", True):
                resume.write_json(
                    args.parent / "causal_multislot/coherence-cache" / f"epoch-{self.index}.json",
                    {"answer": self.index},
                )
            existing = args.parent / "causal_multislot/coherence-cache/original.json"
            if outcome.get("mutate_existing"):
                resume.write_json(existing, {"answer": "replaced"})
            if outcome.get("delete_existing"):
                existing.unlink()
            stages = []
            for profile, kind in [
                ("causal_multislot", "coherence-mini"),
                ("independent", "public231"),
                ("independent", "coherence-mini"),
            ]:
                audit = (
                    {"raw_replay": {"raw_cache_files": 1248}}
                    if kind == "coherence-mini"
                    else {"n_attempted": 231}
                )
                stages.append(
                    {"profile": profile, "kind": kind, "status": "completed", "audit": audit}
                )
            if outcome.get("stage_maintenance_error"):
                stages[0]["allocator_maintenance_after"] = {
                    "error": {"type": "RuntimeError", "message": "stage cleanup failed"}
                }
            calls = outcome.get("calls", 2)
            trace = output / "causal_multislot/coherence-mini-model-calls.jsonl"
            trace.parent.mkdir(parents=True)
            trace.write_text(
                "".join(
                    json.dumps(
                        {
                            "origin": "new_model_inference",
                            "error": (
                                {"type": "RuntimeError", "message": "real inference failure"}
                                if outcome.get("callback_error")
                                else None
                            ),
                            "allocator_maintenance": {
                                "error": (
                                    {"type": "RuntimeError", "message": "cleanup failed"}
                                    if outcome.get("maintenance_error")
                                    else None
                                )
                            },
                        }
                    )
                    + "\n"
                    for _ in range(calls)
                )
            )
            child = {
                "status": outcome["status"],
                "source_inventory": context["child_sources"],
                "checkpoint": identity,
                "epoch_budget": {"new_calls": calls},
                "stages": stages,
            }
            if child["status"] == "checkpointed":
                child["epoch_checkpoint"] = {"reason": "new_model_call_budget", "new_calls": calls}
            if outcome.get("error"):
                child["error"] = {"type": "RuntimeError", "message": outcome["error"]}
            resume.write_json(output / "receipt.json", child)
            context["live"] = False
            return self.code

        def poll(self):
            return None if context["live"] else self.code

    monkeypatch.setattr(RUNNER.subprocess, "Popen", Child)
    return args, resume, context


def test_sequential_epochs_reap_before_restart_and_preserve_final_audit(tmp_path, monkeypatch):
    args, resume, context = setup(
        tmp_path,
        monkeypatch,
        [
            {"code": 75, "status": "checkpointed", "calls": 2},
            {"code": 0, "status": "completed", "calls": 1},
        ],
    )
    assert RUNNER.supervise(args) == 0
    receipt = json.loads((args.output / "receipt.json").read_text())
    assert receipt["status"] == "completed" and receipt["active_child"] is None
    assert receipt["total_new_model_calls"] == 3
    assert [
        event for event in context["events"] if "child" in event or event.startswith("launch")
    ] == ["external-child-terminal", "launch-1", "wait-child-1", "launch-2", "wait-child-2"]


def test_checkpoint_without_successful_cache_progress_never_restarts(tmp_path, monkeypatch):
    args, resume, context = setup(
        tmp_path, monkeypatch, [{"code": 75, "status": "checkpointed", "progress": False}]
    )
    with pytest.raises(RuntimeError, match="no successful cache progress"):
        RUNNER.supervise(args)
    assert context["launched"] == 1 and not context["live"]


def test_child_failure_preserved_and_not_reclassified_as_checkpoint(tmp_path, monkeypatch):
    args, resume, context = setup(
        tmp_path, monkeypatch, [{"code": 1, "status": "error", "error": "real model OOM"}]
    )
    with pytest.raises(RuntimeError, match="without a valid resource checkpoint"):
        RUNNER.supervise(args)
    receipt = json.loads((args.output / "receipt.json").read_text())
    assert receipt["epochs"][0]["child_error"]["message"] == "real model OOM"
    assert Path(receipt["epochs"][0]["cache_after"]["path"]).is_file()
    assert context["launched"] == 1


@pytest.mark.parametrize("status,code", [("checkpointed", 75), ("completed", 0)])
@pytest.mark.parametrize(
    "failure", ["callback_error", "maintenance_error", "stage_maintenance_error"]
)
def test_real_errors_never_restart_or_complete_even_with_successful_cache_progress(
    tmp_path, monkeypatch, status, code, failure
):
    args, _resume, context = setup(
        tmp_path, monkeypatch, [{"code": code, "status": status, failure: True}]
    )
    with pytest.raises(RuntimeError, match="real callback/maintenance failure"):
        RUNNER.supervise(args)
    receipt = json.loads((args.output / "receipt.json").read_text())
    assert receipt["status"] == "error"
    assert receipt["epochs"][0]["changed_cache_entries"]
    assert context["launched"] == 1 and not context["live"]
    if failure != "stage_maintenance_error":
        counter = (
            "total_failed_callbacks"
            if failure == "callback_error"
            else "total_maintenance_failures"
        )
        assert receipt[counter] == 2


@pytest.mark.parametrize("status,code", [("checkpointed", 75), ("error", 1)])
@pytest.mark.parametrize("mutation", ["mutate_existing", "delete_existing"])
def test_preexisting_coherence_cache_entries_are_immutable_even_on_error_exit(
    tmp_path, monkeypatch, status, code, mutation
):
    args, resume, context = setup(
        tmp_path,
        monkeypatch,
        [{"code": code, "status": status, mutation: True, "error": "original child failure"}],
    )
    original = args.parent / "causal_multislot/coherence-cache/original.json"
    resume.write_json(original, {"answer": "first success"})
    with pytest.raises(ValueError, match="existing shared cache entry was modified or deleted"):
        RUNNER.supervise(args)
    receipt = json.loads((args.output / "receipt.json").read_text())
    epoch = receipt["epochs"][0]
    assert epoch["immutable_cache_violations"] == ["coherence/causal_multislot/original.json"]
    assert epoch["child_error"]["message"] == "original child failure"
    before = json.loads(Path(epoch["cache_before"]["path"]).read_text())
    after = json.loads(Path(epoch["cache_after"]["path"]).read_text())
    assert before["coherence/causal_multislot/original.json"] != after.get(
        "coherence/causal_multislot/original.json"
    )
    assert context["launched"] == 1 and not context["live"]


def test_epoch_limit_does_not_claim_completion(tmp_path, monkeypatch):
    args, resume, context = setup(tmp_path, monkeypatch, [{"code": 75, "status": "checkpointed"}])
    args.max_epochs = 1
    assert RUNNER.supervise(args) == 2
    receipt = json.loads((args.output / "receipt.json").read_text())
    assert receipt["status"] == "epoch_limit_reached"


def test_final_receipt_requires_full_populations_and_cache_replays():
    with pytest.raises(ValueError, match="required benchmark stages"):
        RUNNER.verify_final_receipt({"stages": []})


def test_child_argv_executes_frozen_source_and_forwards_exact_epoch_budget(tmp_path):
    args = SimpleNamespace(
        output=tmp_path,
        parent=tmp_path / "original",
        coherence_root=tmp_path / "coherence",
        heaven_root=tmp_path / "heaven",
        predecessor_receipt=tmp_path / "external.json",
        wait_for_pid=124,
        original_pid=89007,
        poll_seconds=15,
        max_new_calls=20,
        max_driver_fraction=0.8,
        public_cache_root=tmp_path / "public",
        additional_predecessor_pid=[44736],
    )
    argv = RUNNER.child_argv(args, 2)
    assert argv[2] == str(tmp_path / "sources/scripts/resume_mps_benchmarks.py")
    assert argv[argv.index("--max-new-calls") + 1] == "20"
    assert argv[argv.index("--max-driver-fraction") + 1] == "0.8"
    assert argv[argv.index("--additional-predecessor-pid") + 1] == "44736"
