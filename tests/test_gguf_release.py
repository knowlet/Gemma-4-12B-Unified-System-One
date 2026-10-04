"""CPU-only checks for population accounting, native protocol and artifact binding."""

import copy
import hashlib
import json
import runpy
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope="module")
def evaluator():
    return runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/evaluate_gguf_release.py")
    )


@pytest.fixture
def manifests():
    return {
        role: {
            "letters": list(range(52)),
            "cases": [
                {
                    "id": f"{role}-{i}",
                    "split": "calibration" if role == "calibration" else "test",
                    "group_id": f"{role}-{i}",
                    "request_sha256": "a" * 64,
                    "tensor_file_sha256": "b" * 64,
                    "slots": [2],
                    "nopts": [2],
                    "questions": [{"id": "answer", "type": "choice", "labels": ["yes", "no"]}],
                    "gold": {"answer": "yes"},
                }
                for i in range(2)
            ],
        }
        for role in ("calibration", "test", "media")
    }


def payload(role, case):
    return {
        "case_id": case["id"],
        "question_ids": ["answer"],
        "slots": [2],
        "nopts": [2],
        "input_ids": [10, 11, 12],
        "source_input_token_count": 3,
        "letters": list(range(52)),
        "has_media": role == "media",
    }


def response(item, logits=None):
    return {
        "status": "ok",
        "case_id": item["case_id"],
        "question_ids": item["question_ids"],
        "slots": item["slots"],
        "runtime_slots": item["slots"],
        "native_token_count": 3,
        "raw_logits": [[2.0, 0.0] if logits is None else logits],
        "runtime": {
            "engine": "llama.cpp",
            "llama_cpp_revision": "a" * 40,
            "logits_semantics": "post_native_softcap_pre_temperature",
        },
    }


def test_complete_own_calibration_and_full_population(evaluator, manifests):
    from s1.training import fit_temperature

    calls = []

    def backend(item):
        assert "gold" not in item and "split" not in item
        calls.append(item["case_id"])
        return response(item, [3.0, 0.0] if item["case_id"].endswith("-0") else [0.0, 1.0])

    result = evaluator["evaluate_backend"](backend, manifests, payload, warmup=1, repeat=2)
    expected = fit_temperature([[3.0, 0.0], [0.0, 1.0]], [0, 0])
    assert result["population_complete"] is True
    assert result["calibration"]["temperature"] == expected["temperature"]
    assert calls == [f"{role}-{i}" for role in manifests for i in (0, 0, 0, 1, 1)]
    for role in manifests:
        stats = result["summary"][role]
        assert stats["case_coverage"] == stats["decision_coverage"] == 1
        assert stats["backend_call_latency_ms"]["samples"] == 4
        assert "prepared_forward_latency_ms" not in stats
        assert stats["decision_status_counts"] == {"ok": 2}
        answer = result["cases"][role][0]["answers"]["answer"]
        assert sum(answer["probabilities"].values()) == pytest.approx(1)


@pytest.mark.parametrize("status", ["error", "unsupported"])
def test_calibration_failure_prevents_fit_and_heldout_execution(evaluator, manifests, status):
    calls = []

    def backend(item):
        calls.append(item["case_id"])
        if item["case_id"] == "calibration-0":
            return {"status": status, "case_id": item["case_id"], "error": "fixture failure"}
        return response(item)

    result = evaluator["evaluate_backend"](backend, manifests, payload, warmup=0)
    assert calls == ["calibration-0", "calibration-1"]
    assert result["calibration"] is None and not result["population_complete"]
    assert result["summary"]["calibration"]["decision_status_counts"] == {status: 1, "unscored": 1}
    for role in ("test", "media"):
        assert result["summary"][role]["case_status_counts"] == {"not_run": 2}
        assert result["summary"][role]["expected_decisions"] == 2


def test_media_unsupported_retains_denominator(evaluator, manifests):
    def backend(item):
        if item["case_id"] == "media-0":
            return {"status": "unsupported", "case_id": item["case_id"], "error": "fixture codec"}
        return response(item)

    result = evaluator["evaluate_backend"](backend, manifests, payload, warmup=0)
    stats = result["summary"]["media"]
    assert not result["population_complete"]
    assert stats["expected_cases"] == stats["expected_decisions"] == 2
    assert stats["successful_cases"] == stats["scored_decisions"] == 1
    assert stats["case_coverage"] == stats["decision_coverage"] == 0.5
    assert stats["decision_status_counts"] == {"unsupported": 1, "ok": 1}


def test_fatal_transport_preserves_remaining_not_run(evaluator, manifests):
    def backend(item):
        raise evaluator["AdapterUnavailable"]("closed")

    result = evaluator["evaluate_backend"](backend, manifests, payload, warmup=0)
    assert result["summary"]["calibration"]["case_status_counts"] == {"error": 1, "not_run": 1}
    assert result["summary"]["test"]["case_status_counts"] == {"not_run": 2}


def test_runtime_identity_change_stops_population(evaluator, manifests):
    def backend(item):
        value = response(item)
        value["runtime"]["revision"] = item["case_id"]
        return value

    result = evaluator["evaluate_backend"](backend, manifests, payload, warmup=0)
    assert result["cases"]["calibration"][1]["error_type"] == "AdapterUnavailable"
    assert result["summary"]["test"]["case_status_counts"] == {"not_run": 2}


def test_phase_guard_failure_keeps_partial_records(evaluator, manifests):
    result = {}

    def guard():
        raise ValueError("runtime changed")

    with pytest.raises(ValueError, match="runtime changed"):
        evaluator["evaluate_backend"](
            response, manifests, payload, warmup=0, result=result, phase_guard=guard
        )
    assert [r["status"] for r in result["cases"]["calibration"]] == ["ok", "ok"]
    assert [r["status"] for r in result["cases"]["test"]] == ["not_run", "not_run"]
    assert result["calibration"] is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("case_id", "wrong"),
        ("question_ids", ["wrong"]),
        ("slots", [1]),
        ("runtime_slots", [3]),
        ("native_token_count", True),
        ("native_token_count", 100000),
        ("raw_logits", [[True, 0]]),
        ("raw_logits", [[float("nan"), 0]]),
        ("raw_logits", [[1, 2, 3]]),
        ("runtime", {"engine": "llama.cpp", "logits_semantics": "softmax"}),
    ],
)
def test_malformed_native_response_is_rejected(evaluator, manifests, field, value):
    item = payload("test", manifests["test"]["cases"][0])
    result = response(item)
    result[field] = value
    with pytest.raises(ValueError):
        evaluator["validate_response"](result, item)


def test_media_prefix_variance_is_recorded_text_variance_rejected(evaluator, manifests):
    item = payload("media", manifests["media"]["cases"][0])
    result = {**response(item), "runtime_slots": [4], "native_token_count": 5}
    assert evaluator["validate_response"](result, item)["runtime_slots"] == [4]
    with pytest.raises(ValueError, match="text token count"):
        evaluator["validate_response"](result, {**item, "has_media": False})


def test_make_payload_never_sends_gold(evaluator, tmp_path):
    from s1.contracts import DecisionRequest

    request = DecisionRequest.model_validate(
        {
            "state": "hello",
            "questions": [
                {"id": "answer", "type": "choice", "instructions": "pick", "options": ["yes", "no"]}
            ],
        }
    )
    path = tmp_path / "case.npz"
    np.savez(path, input_ids=np.array([[10, 11, 12]]), attention_mask=np.ones((1, 3)))
    case = {
        "id": "case",
        "file": "case.npz",
        "slots": [2],
        "nopts": [2],
        "questions": [{"id": "answer"}],
        "gold": {"answer": "yes"},
        "tensor_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    item = evaluator["make_payload"](
        "test",
        case,
        directories={"test": tmp_path},
        manifests={"test": {"letters": list(range(52))}},
        population={"test": {"case": SimpleNamespace(request=request)}},
    )
    assert "gold" not in item and "gold" not in item["request"]
    assert item["input_ids"] == [10, 11, 12]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError):
        evaluator["make_payload"](
            "test",
            case,
            directories={"test": tmp_path},
            manifests={"test": {"letters": list(range(52))}},
            population={"test": {"case": SimpleNamespace(request=request)}},
        )


def test_persistent_adapter_ndjson_roundtrip(evaluator, tmp_path):
    script = tmp_path / "backend.py"
    script.write_text(
        "import json,sys\nfor line in sys.stdin:\n print(json.dumps(json.loads(line)),flush=True)\n"
    )
    adapter = evaluator["JsonLineAdapter"]([sys.executable, str(script)], timeout=5)
    try:
        pid = adapter.process.pid
        assert adapter({"n": 1}) == {"n": 1}
        assert adapter({"n": 2}) == {"n": 2}
        assert adapter.process.pid == pid
    finally:
        adapter.close()
    assert adapter.process.poll() is not None


@pytest.mark.parametrize(
    "program", ["import time; time.sleep(30)", "print('invalid JSON')", "pass"]
)
def test_bad_transport_stops_and_reaps_process(evaluator, program):
    adapter = evaluator["JsonLineAdapter"]([sys.executable, "-c", program], timeout=0.1)
    with pytest.raises(evaluator["AdapterUnavailable"]):
        adapter({"case_id": "example"})
    assert adapter.closed and adapter.process.poll() is not None


def test_timeout_kills_adapter_child_process_group(evaluator, tmp_path):
    marker = tmp_path / "child-survived"
    child = f"import pathlib,time; time.sleep(0.6); pathlib.Path({str(marker)!r}).write_text('bad')"
    parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(30)"
    adapter = evaluator["JsonLineAdapter"]([sys.executable, "-c", parent], timeout=0.2)
    with pytest.raises(evaluator["AdapterUnavailable"], match="timed out"):
        adapter({"case_id": "example"})
    time.sleep(0.6)
    assert not marker.exists()


@pytest.fixture
def package(evaluator, tmp_path):
    base = {
        "source_model": "fixture/gemma",
        "source_revision": "a" * 40,
        "prompt_version": 1,
        "temperature": 7.0,
        "training_recipe_sha256": "b" * 64,
    }
    config = {
        "model_type": "gemma4_unified",
        "tie_word_embeddings": True,
        "text_config": {"final_logit_softcapping": 30},
        "vision_config": {"x": 1},
        "audio_config": {"x": 1},
    }
    contents = {
        "base_s1_config.json": json.dumps(base),
        "config.json": json.dumps(config),
        "tokenizer.json": "{}",
        "tokenizer_config.json": "{}",
        "processor_config.json": "{}",
        "chat_template.jinja": "template",
        "tensor-inventory.json": "{}",
        "model.gguf": "GGUF fixture",
        "mmproj.gguf": "GGUF projector",
    }
    for name, content in contents.items():
        (tmp_path / name).write_text(content)
    files = [
        {
            "path": name,
            "size_bytes": (tmp_path / name).stat().st_size,
            "sha256": evaluator["file_sha256"](tmp_path / name),
        }
        for name in sorted(contents)
    ]
    source_files = [{"path": "config.json", "size_bytes": 10, "sha256": "c" * 64}]
    identity = {
        "source_model": base["source_model"],
        "source_revision": base["source_revision"],
        "prompt_version": 1,
        "source_checkpoint_sha256": evaluator["digest"](source_files),
        "source_s1_config_sha256": evaluator["file_sha256"](tmp_path / "base_s1_config.json"),
    }
    conversion = {
        **identity,
        "format": "gguf",
        "status": "converted_and_structurally_verified",
        "training_recipe_sha256": "b" * 64,
        "files": files,
        "model_file": "model.gguf",
        "mmproj_file": "mmproj.gguf",
        "source_checkpoint_files": source_files,
    }
    manifests = {
        role: {"source_checkpoint_files": copy.deepcopy(source_files)}
        for role in ("calibration", "test", "media")
    }
    return tmp_path, conversion, identity, manifests


def test_immutable_export_hashing_and_source_temperature_preservation(evaluator, package):
    exported, base = evaluator["validate_conversion"](*package)
    assert exported["sha256"] == evaluator["digest"](package[1]["files"])
    assert base["temperature"] == 7.0
    (package[0] / "model.gguf").write_text("GGUF changed")
    with pytest.raises(ValueError, match="exported file changed"):
        evaluator["validate_conversion"](*package)


@pytest.mark.parametrize(
    "mutation", ["source", "inventory", "training", "extra", "duplicate", "path"]
)
def test_conversion_mismatches_rejected(evaluator, package, mutation):
    directory, conversion, identity, manifests = package
    if mutation == "source":
        conversion["source_checkpoint_sha256"] = "d" * 64
    elif mutation == "inventory":
        manifests["test"]["source_checkpoint_files"][0]["sha256"] = "e" * 64
    elif mutation == "training":
        conversion["training_recipe_sha256"] = "f" * 64
    elif mutation == "extra":
        (directory / "extra.gguf").write_text("unexpected weights")
    elif mutation == "duplicate":
        conversion["files"].append(conversion["files"][0])
    else:
        conversion["files"][0]["path"] = "../outside"
    with pytest.raises(ValueError):
        evaluator["validate_conversion"](directory, conversion, identity, manifests)


def test_runtime_source_pin_and_mutation_guard(evaluator, tmp_path):
    def git(*args):
        return subprocess.check_output(
            [
                "git",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "core.hooksPath=/dev/null",
                "-C",
                str(tmp_path),
                *args,
            ],
            stderr=subprocess.DEVNULL,
        )

    (tmp_path / "scripts").mkdir()
    (tmp_path / "src/s1").mkdir(parents=True)
    names = [
        "evaluate_mlx_release.py",
        "summarize_release.py",
        "prepare_mlx_validation.py",
        "verify_mlx_export.py",
    ]
    for name in names:
        (tmp_path / "scripts" / name).write_text("# fixture\n")
    owned = tmp_path / "src/s1/unified.py"
    owned.write_text("# fixture\n")
    git("init", "-q")
    git("add", ".")
    git(
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    head = git("rev-parse", "HEAD").decode().strip()
    guard = evaluator["RuntimeGuard"](head, [sys.executable], root=tmp_path)
    guard.check()
    assert guard.receipt()["source_revision"] == head
    owned.write_text("# modified\n")
    with pytest.raises(ValueError, match="runtime file changed"):
        guard.check()
    with pytest.raises(ValueError, match="differs from pinned Git HEAD"):
        evaluator["RuntimeGuard"](head, [sys.executable], root=tmp_path)
    with pytest.raises(ValueError, match="current full Git HEAD"):
        evaluator["RuntimeGuard"]("a" * 40, [sys.executable], root=tmp_path)


def test_preflight_failure_does_not_launch_adapter_or_overwrite_sidecar(evaluator, tmp_path):
    marker, output, sidecar = (
        tmp_path / "launched",
        tmp_path / "report.json",
        tmp_path / "s1_config.json",
    )
    sidecar.write_text('{"temperature": 7.0}')
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    argv = [
        "--model",
        str(tmp_path),
        "--output",
        str(output),
        "--runtime-source-revision",
        "a" * 40,
    ]
    for role in ("calibration", "test", "media"):
        argv += [f"--{role}-inputs", str(tmp_path), f"--{role}-manifest-sha256", "0" * 64]
    for role in ("dataset", "conversion"):
        argv += [f"--{role}-manifest", str(manifest), f"--{role}-manifest-sha256", "0" * 64]
    argv += [
        "--adapter-command",
        sys.executable,
        "-c",
        f"import pathlib; pathlib.Path({str(marker)!r}).write_text('bad')",
    ]
    assert evaluator["main"](argv) == 1
    report = json.loads(output.read_text())
    assert report["status"] == "error" and not report["release_validation_complete"]
    assert report["failure"]["stage"] == "preflight"
    assert not marker.exists()
    assert sidecar.read_text() == '{"temperature": 7.0}'
