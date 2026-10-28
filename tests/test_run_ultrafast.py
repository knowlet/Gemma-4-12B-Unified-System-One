"""Browser campaign reporting keeps failures and checkpoint identity auditable."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("run_ultrafast", ROOT / "scripts/run_ultrafast.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def row(profile, task, split, *, success, latency, calls, text_calls=0, failure=None):
    return {
        "profile": profile,
        "task_id": task,
        "split": split,
        "verified_success": success,
        "wall_ms": latency,
        "backend_calls": calls,
        "text_calls": text_calls,
        "failure_class": failure,
    }


def test_summary_counts_all_attempts_failures_and_profile_specific_calls():
    rows = [
        row("causal", "a", "development", success=True, latency=10, calls=2, text_calls=1),
        row(
            "independent",
            "a",
            "development",
            success=False,
            latency=500,
            calls=7,
            failure="decision_budget",
        ),
        row(
            "causal",
            "a",
            "development",
            success=False,
            latency=100,
            calls=3,
            text_calls=2,
            failure="model_error",
        ),
        row(
            "causal",
            "b",
            "held_out",
            success=False,
            latency=1000,
            calls=5,
            failure="false_done",
        ),
    ]
    result = RUNNER.summarize(rows)
    assert list(result) == ["causal", "independent"]
    causal = result["causal"]
    assert causal["attempted"] == 3
    assert causal["unique_tasks"] == 2
    assert causal["verified_successes"] == 1
    assert causal["task_success_rate"] == pytest.approx(1 / 3)
    assert causal["task_latency_ms_p50_p95"] == pytest.approx([100, 910])
    assert causal["model_calls"] == 13
    assert causal["decision_calls"] == 10
    assert causal["text_calls"] == 3
    assert causal["failure_classes"] == {"model_error": 1, "false_done": 1}
    assert causal["by_split"] == {
        "development": {"attempted": 2, "verified_successes": 1},
        "held_out": {"attempted": 1, "verified_successes": 0},
    }
    assert causal["reported_cost_usd"] is None
    assert "unknown is not free" in causal["cost_scope"]
    independent = result["independent"]
    assert independent["attempted"] == 1
    assert independent["model_calls"] == 7
    assert independent["decision_calls"] == 7
    assert independent["text_calls"] == 0
    assert independent["task_success_rate"] == 0
    assert independent["task_latency_ms_p50_p95"] == [500, 500]
    assert independent["failure_classes"] == {"decision_budget": 1}


def test_summary_retains_all_failed_tasks_and_does_not_invent_free_cost():
    rows = [
        row("x", "a", "development", success=False, latency=20, calls=0, failure="setup_error"),
        row("x", "b", "held_out", success=False, latency=80, calls=1, failure="setup_error"),
    ]
    summary = RUNNER.summarize(rows)["x"]
    assert summary["attempted"] == summary["unique_tasks"] == 2
    assert summary["verified_successes"] == summary["task_success_rate"] == 0
    assert summary["failure_classes"] == {"setup_error": 2}
    assert summary["model_calls"] == 1
    assert summary["task_latency_ms_p50_p95"] == pytest.approx([50, 77])
    assert summary["reported_cost_usd"] is None
    assert RUNNER.summarize([]) == {}


def checkpoint(tmp_path):
    """Tiny arbitrary bytes exercise hashing; no model loader or weight download."""
    path = tmp_path / "checkpoint"
    path.mkdir()
    files = []
    for name in (
        "model.safetensors",
        "config.json",
        "s1_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "chat_template.jinja",
    ):
        raw = f"fixture bytes for {name}".encode()
        (path / name).write_bytes(raw)
        files.append(
            {"path": name, "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        )
    manifest = path / "artifact-manifest.json"
    manifest.write_text(json.dumps({"files": files}))
    return path, files


def test_verify_checkpoint_binds_every_manifest_member_and_manifest_bytes(tmp_path):
    path, files = checkpoint(tmp_path)
    result = RUNNER.verify_checkpoint(path)
    assert result["path"] == str(path.resolve())
    assert result["verified_files"] == files
    assert (
        result["manifest_sha256"]
        == hashlib.sha256((path / "artifact-manifest.json").read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("mutation", ["missing", "size", "same_size_content"])
def test_verify_checkpoint_rejects_missing_or_mutated_member(tmp_path, mutation):
    path, _ = checkpoint(tmp_path)
    target = path / "model.safetensors"
    if mutation == "missing":
        target.unlink()
    elif mutation == "size":
        target.write_bytes(b"short")
    else:
        target.write_bytes(b"x" * target.stat().st_size)
    with pytest.raises(ValueError, match="checkpoint member"):
        RUNNER.verify_checkpoint(path)


@pytest.mark.parametrize("escape", ["parent", "absolute", "symlink"])
def test_verify_checkpoint_rejects_paths_outside_checkpoint(tmp_path, escape):
    path, files = checkpoint(tmp_path)
    external = tmp_path / "external.bin"
    external.write_bytes(b"external weights")
    if escape == "parent":
        member = "../external.bin"
    elif escape == "absolute":
        member = str(external.resolve())
    else:
        (path / "outside.bin").symlink_to(external)
        member = "outside.bin"
    files.append(
        {
            "path": member,
            "size_bytes": external.stat().st_size,
            "sha256": hashlib.sha256(external.read_bytes()).hexdigest(),
        }
    )
    (path / "artifact-manifest.json").write_text(json.dumps({"files": files}))
    with pytest.raises(ValueError, match="checkpoint member|unsafe"):
        RUNNER.verify_checkpoint(path)


@pytest.mark.parametrize("mutation", ["empty", "missing_weights", "duplicate"])
def test_verify_checkpoint_rejects_incomplete_or_duplicate_manifest(tmp_path, mutation):
    path, files = checkpoint(tmp_path)
    if mutation == "empty":
        files = []
    elif mutation == "missing_weights":
        files = [row for row in files if row["path"] != "model.safetensors"]
    else:
        files.append(files[0].copy())
    (path / "artifact-manifest.json").write_text(json.dumps({"files": files}))
    with pytest.raises(ValueError):
        RUNNER.verify_checkpoint(path)


@pytest.mark.parametrize("alias", ["absolute_inside", "parent_inside"])
def test_verify_checkpoint_rejects_unsafe_paths_even_when_they_resolve_inside(tmp_path, alias):
    path, files = checkpoint(tmp_path)
    (path / "sub").mkdir()
    files[0]["path"] = (
        str((path / "model.safetensors").resolve())
        if alias == "absolute_inside"
        else "sub/../model.safetensors"
    )
    (path / "artifact-manifest.json").write_text(json.dumps({"files": files}))
    with pytest.raises(ValueError):
        RUNNER.verify_checkpoint(path)


def test_archive_sources_accepts_relative_repository_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    relative = Path("configs/benchmarks/ultrafast-external.json")
    destination = tmp_path / "sources"
    RUNNER.archive_sources([relative], destination)
    archived = destination / relative
    assert archived.read_bytes() == relative.read_bytes()


def test_archive_sources_copies_external_manifest_under_stable_name(tmp_path):
    source = tmp_path / "custom-manifest.json"
    source.write_text('{"tasks": [], "description": "external manifest"}\n')
    destination = tmp_path / "new" / "sources"
    RUNNER.archive_sources([source], destination)
    assert (destination / "input-manifest.json").read_bytes() == source.read_bytes()
    assert sorted(path.name for path in destination.iterdir()) == ["input-manifest.json"]


def test_archive_sources_creates_destination_for_empty_inputs(tmp_path):
    destination = tmp_path / "new" / "sources"
    RUNNER.archive_sources([], destination)
    assert destination.is_dir()
    assert list(destination.iterdir()) == []
