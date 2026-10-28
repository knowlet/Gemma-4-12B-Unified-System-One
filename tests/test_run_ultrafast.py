"""Browser campaign reporting keeps failures and checkpoint identity auditable."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize("records", [[], [{"backend_calls": 1}], [{"hierarchy": {"subcalls": []}}]])
def test_flat_and_nondispatched_hierarchy_task_counts_keep_the_existing_shape(records):
    assert RUNNER.task_backbone_counts(records, 7) == {"backbone_forwards": 7}


@pytest.mark.parametrize("counts,actual", [([1, 2], 7), ([1, None], None), ([None], None)])
def test_hierarchy_task_actual_count_is_unknown_if_any_core_call_is_unknown(counts, actual):
    records = [{"hierarchy": {"subcalls": [{"backbone_forwards": value} for value in counts]}}]
    assert RUNNER.task_backbone_counts(records, 7) == {
        "backbone_forwards": actual,
        "observed_completed_backbone_forwards": 7,
    }


def test_summary_keeps_mixed_known_unknown_actual_counts_null_and_sums_completed_lower_bounds():
    rows = [
        {**row("x", "flat", "external", success=True, latency=10, calls=1), "backbone_forwards": 4},
        {
            **row("x", "hierarchy", "external", success=True, latency=20, calls=1),
            "backbone_forwards": 2,
            "observed_completed_backbone_forwards": 2,
        },
        {
            **row(
                "x", "failed", "external", success=False, latency=30, calls=1, failure="model_error"
            ),
            "backbone_forwards": None,
            "observed_completed_backbone_forwards": 0,
        },
    ]
    summary = RUNNER.summarize(rows)["x"]
    assert summary["backbone_forwards"] is None
    assert summary["observed_completed_backbone_forwards"] == 6
    assert summary["attempted"] == 3 and summary["verified_successes"] == 2
    assert summary["model_calls"] == 3


def test_all_known_flat_summary_has_no_hierarchy_counter_field():
    rows = [
        {
            **row("flat", "one", "external", success=True, latency=10, calls=1),
            "backbone_forwards": 4,
        }
    ]
    summary = RUNNER.summarize(rows)["flat"]
    assert summary["backbone_forwards"] == 4
    assert "observed_completed_backbone_forwards" not in summary


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


def fake_native_model(raw, monkeypatch):
    """Exercise transport handling without importing Torch or dispatching a model."""
    calls = []

    class Input:
        shape = (1, 2)

        def to(self, device):
            assert device == "cpu"
            return self

    class Output:
        def __getitem__(self, key):
            assert key == (0, slice(2, None))
            return [90, 91, 92]

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            calls.append(("template", messages, kwargs))
            return "unchanged rendered prompt"

        def __call__(self, **kwargs):
            assert kwargs == {
                "text": "unchanged rendered prompt",
                "return_tensors": "pt",
                "add_special_tokens": False,
            }
            return {"input_ids": Input()}

    def generate(**kwargs):
        assert set(kwargs) == {"input_ids", "max_new_tokens", "do_sample"}
        assert kwargs["max_new_tokens"] == 128 and kwargs["do_sample"] is False
        calls.append(("generate",))
        return Output()

    def decode(tokens, *, skip_special_tokens):
        assert tokens == [90, 91, 92] and skip_special_tokens is True
        return raw

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    return SimpleNamespace(
        name="same specialist",
        device="cpu",
        max_context=16384,
        processor=Processor(),
        lm=SimpleNamespace(generate=generate),
        tok=SimpleNamespace(decode=decode),
    ), calls


@pytest.mark.parametrize(
    "raw, format",
    [
        ('{"text":"  Graph theory  "}', "direct"),
        ('```json\n{"text":"  Graph theory  "}\n```', "fenced"),
        ('```\n{"text":"  Graph theory  "}\n```', "fenced"),
    ],
)
def test_native_helper_logs_format_normalization_and_preserves_original_generation(
    raw, format, monkeypatch
):
    model, calls = fake_native_model(raw, monkeypatch)
    records = []
    instruction = "Original general JSON field instruction"
    context = {"goal": "Original task context", "observed": ["a", "b"]}
    value, metadata = RUNNER.native_field_text(model, instruction, records)(context)
    assert value == "  Graph theory  "
    assert [call[0] for call in calls] == ["template", "generate"]
    assert calls[0][1] == [
        {"role": "system", "content": instruction},
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
    ]
    assert calls[0][2] == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    record = records[0]
    assert record["raw_text"] == raw and record["instruction"] == instruction
    assert record["context"] == context and record["context"] is not context
    assert record["max_new_tokens"] == 128 and record["do_sample"] is False
    assert record["input_tokens"] == 2 and record["output_tokens"] == 3
    assert record["reported_cost_usd"] is None and record["error"] is None
    assert record["transport_normalization"] == metadata["transport_normalization"]
    assert record["transport_normalization"]["format"] == format
    metadata["transport_normalization"]["format"] = "caller mutation"
    assert record["transport_normalization"]["format"] == format


@pytest.mark.parametrize(
    "raw",
    [
        "```json\n" + "_" * 127,
        'Here is JSON: {"text":"Graph theory"}',
        '{"text":"first","text":"second"}',
        '```json\n{"text":NaN}\n```',
    ],
)
def test_native_helper_keeps_raw_invalid_output_and_raises_without_value_repair(raw, monkeypatch):
    model, calls = fake_native_model(raw, monkeypatch)
    records = []
    with pytest.raises(ValueError):
        RUNNER.native_field_text(model, "Original instruction", records)({"goal": "same context"})
    assert [call[0] for call in calls].count("generate") == 1
    assert len(records) == 1 and records[0]["raw_text"] == raw
    assert records[0]["transport_normalization"] is None
    assert records[0]["error"] is not None


def test_runner_rejects_unknown_oversized_head_policy_before_loading_upstream(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ultrafast.py",
            "--upstream",
            "unused",
            "--output",
            "unused",
            "--oversized-head-policy",
            "unknown",
        ],
    )
    with pytest.raises(SystemExit) as actual:
        RUNNER.main()
    assert actual.value.code == 2


@pytest.mark.parametrize("enabled", [False, True])
def test_stability_flag_declares_policy_and_passes_only_the_original_marker(
    tmp_path, monkeypatch, enabled
):
    output, manifest = tmp_path / "run", tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "scope": "CPU fake integration",
                "split_policy": "fixed",
                "tasks": [
                    {
                        "id": "one",
                        "split": "development",
                        "url": "https://example.test/",
                        "goal": "Original goal",
                        "expected": {"title": "Original"},
                    }
                ],
            }
        )
    )
    argv = [
        "run_ultrafast.py",
        "--upstream",
        str(tmp_path / "upstream"),
        "--output",
        str(output),
        "--manifest",
        str(manifest),
        "--profiles",
        "uniform",
        "--repeats",
        "1",
    ] + (["--stable-observation"] if enabled else [])
    monkeypatch.setattr(sys, "argv", argv)
    original_marker = "(() => original upstream MARKER)()"
    policy = {"name": "fixed marker diagnostic", "quiet_ms": 250, "poll_window_ms": 2000}
    modules = {
        "browser": SimpleNamespace(MARKER=original_marker),
        "model": SimpleNamespace(),
        "agent": SimpleNamespace(field_text=object()),
    }
    observed = []

    class Transport:
        browser_version = "fake only"

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

    def run_task(agent, selected, url, destination, **kwargs):
        assert agent is modules["agent"] and url == "https://example.test/"
        observed.append(kwargs)
        return row("uniform", selected["id"], "development", success=True, latency=10, calls=0)

    monkeypatch.setitem(
        sys.modules,
        "ultrafast_browser",
        SimpleNamespace(
            IsolatedCDP=Transport,
            fixture_server=lambda *_args: nullcontext("http://local"),
            load_upstream=lambda *_args: (modules, {}),
            run_task=run_task,
            STABLE_OBSERVATION_POLICY=policy,
        ),
    )
    monkeypatch.setattr(RUNNER.importlib.metadata, "version", lambda _name: "fake-version")
    monkeypatch.setattr(RUNNER.platform, "platform", lambda: "fake-platform")
    monkeypatch.setattr(RUNNER.subprocess, "check_output", lambda *args, **kwargs: "fake head")
    monkeypatch.setattr(
        RUNNER, "GemmaBackend", lambda *args, **kwargs: pytest.fail("no real model")
    )
    assert RUNNER.main() == 0
    identity = json.loads((output / "identity.json").read_text())
    assert len(observed) == 1
    if enabled:
        assert identity["stable_observation"] == {
            "enabled": True,
            "policy": policy,
            "marker_expression_sha256": hashlib.sha256(original_marker.encode()).hexdigest(),
        }
        assert observed[0]["stable_observation_marker"] == original_marker
    else:
        assert "stable_observation" not in identity
        assert "stable_observation_marker" not in observed[0]
