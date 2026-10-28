"""Base-helper lifecycle contracts use fake files/runtime and never load a model."""

from __future__ import annotations

import builtins
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "diagnose_mps_base_helper", ROOT / "scripts/diagnose_mps_base_helper.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


@pytest.fixture
def files(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    for name in RUNNER.REQUIRED_FILES:
        (base / name).write_bytes(b"local fake bytes")
    RUNNER.write_json(
        base / "config.json",
        {
            "model_type": "gemma4_unified",
            "architectures": ["Gemma4UnifiedForConditionalGeneration"],
            "dtype": "bfloat16",
        },
    )
    trace = tmp_path / "helper.json"
    helper = {
        "instruction": "Return the original field value as JSON.",
        "context": {"goal": "Enter Oslo", "page": "Observed field"},
        "max_new_tokens": 128,
        "do_sample": False,
    }
    RUNNER.write_json(
        trace, [{**helper, "raw_text": "UNUSED OLD OUTPUT", "expected": "UNUSED GOLD"}]
    )
    return SimpleNamespace(
        base_model=base,
        helper_trace=trace,
        helper_record=1,
        wait_for_pid=[11, 22],
        output=tmp_path / "control",
        helper=helper,
    )


def test_select_helper_preserves_only_original_generation_input(files):
    assert RUNNER.select_helper(files.helper_trace, 1) == files.helper


@pytest.mark.parametrize(
    "field,value",
    [
        ("do_sample", True),
        ("max_new_tokens", 129),
        ("max_new_tokens", True),
        ("instruction", ""),
        ("context", "not an object"),
    ],
)
def test_helper_cannot_change_frozen_greedy_settings(files, field, value):
    row = {**files.helper, field: value}
    RUNNER.write_json(files.helper_trace, [row])
    with pytest.raises(ValueError, match="original instruction/context"):
        RUNNER.select_helper(files.helper_trace, 1)


def test_invalid_or_out_of_range_helper_index_fails_closed(files):
    for index in [0, 2]:
        with pytest.raises(ValueError, match="existing one-based"):
            RUNNER.select_helper(files.helper_trace, index)


def test_freeze_copies_metadata_and_sources_without_reading_weight_payload(files, monkeypatch):
    files.output.mkdir()
    original = RUNNER.digest

    def digest(path):
        assert Path(path).name != "model.safetensors", "weight reads must wait for predecessor exit"
        return original(path)

    monkeypatch.setattr(RUNNER, "digest", digest)
    receipt = RUNNER.freeze(files)
    assert receipt["base_files"]["model.safetensors"]["kind"] == "weight_not_copied"
    assert not list((files.output / "sources").rglob("*.safetensors"))
    assert (files.output / "sources/base-metadata/tokenizer.json").is_file()
    assert "scripts/diagnose_mps_numerics.py" in receipt["source_inventory"]
    assert "scripts/resume_mps_benchmarks.py" in receipt["source_inventory"]
    assert (
        receipt["declared_base"]["repository"] is None
    )  # Arbitrary local names prove no publisher.
    assert not receipt["declared_base"]["absence_of_fine_tuning_proven"]
    assert not receipt["declared_base"]["derivative_artifact_manifest_used"]


def test_hashes_every_base_file_including_weights_before_runtime_copy(files):
    files.output.mkdir()
    receipt = RUNNER.freeze(files)
    hashes = RUNNER.verify_base_files(files, receipt)
    assert set(hashes) == RUNNER.REQUIRED_FILES
    assert hashes["model.safetensors"]["sha256"] == hashlib.sha256(b"local fake bytes").hexdigest()
    receipt["verified_base_files"] = hashes
    runtime = RUNNER.prepare_runtime_model(files, receipt)
    assert (runtime / "model.safetensors").is_symlink()
    assert (runtime / "model.safetensors").resolve() == (
        files.base_model / "model.safetensors"
    ).resolve()
    assert (runtime / "config.json").read_bytes() == (files.base_model / "config.json").read_bytes()


def test_metadata_content_hash_mismatch_detected_even_with_same_stat_identity(files):
    files.output.mkdir()
    receipt = RUNNER.freeze(files)
    path = files.base_model / "tokenizer.json"
    stat = path.stat()
    path.write_bytes(b"x" * stat.st_size)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="metadata hash differs"):
        RUNNER.verify_base_files(files, receipt)


@pytest.mark.parametrize("mutation", ["weight", "extra_file", "missing"])
def test_queued_base_mutation_cannot_load(files, mutation):
    files.output.mkdir()
    receipt = RUNNER.freeze(files)
    if mutation == "weight":
        (files.base_model / "model.safetensors").write_bytes(b"changed weights")
    elif mutation == "missing":
        (files.base_model / "tokenizer.json").unlink()
    else:
        (files.base_model / "unexpected.json").write_text("{}")
    with pytest.raises(ValueError, match="changed while queued"):
        RUNNER.verify_base_files(files, receipt)


def fake_modules(monkeypatch, *, wait=None):
    events = []

    def capture(pid):
        events.append(("capture", pid))
        return {"pid": pid, "identity": "captured process"}

    def waiting(handle, save, seconds):
        assert seconds == 15
        events.append(("wait", handle["pid"]))
        if wait:
            wait(handle)
        handle["status"] = "original_exited"
        save()

    def load(path, name):
        if Path(path).name == "resume_mps_benchmarks.py":
            return SimpleNamespace(capture_handle=capture, wait_handle=waiting)
        assert Path(path).name == "diagnose_mps_numerics.py"
        events.append(("generation_source", name))
        return SimpleNamespace(generate_native=lambda *_args: pytest.fail("no real generation"))

    monkeypatch.setattr(RUNNER, "load_module", load)
    return events


def test_freeze_wait_hash_then_fake_execution_without_torch_import(files, monkeypatch):
    events = fake_modules(monkeypatch)
    original_digest = RUNNER.digest
    original_import = builtins.__import__

    def digest(path):
        if Path(path).name == "model.safetensors":
            assert [("wait", 11), ("wait", 22)] == [e for e in events if e[0] == "wait"]
            events.append(("hash_weights", str(path)))
        return original_digest(path)

    def guarded_import(name, *args, **kwargs):
        if name == "torch" or name == "s1" or name.startswith("s1."):
            pytest.fail("Torch/S1 must not import before the fake execution boundary")
        return original_import(name, *args, **kwargs)

    def execute(args, receipt, generator):
        assert "model.safetensors" in receipt["verified_base_files"]
        events.append(("execute", None))
        return True

    monkeypatch.setattr(RUNNER, "digest", digest)
    monkeypatch.setattr(RUNNER, "execute", execute)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert RUNNER.run(files) == 0
    assert events[:4] == [("capture", 11), ("capture", 22), ("wait", 11), ("wait", 22)]
    assert events[-1] == ("execute", None)
    receipt = json.loads((files.output / "receipt.json").read_text())
    assert receipt["status"] == "completed_control"


def test_observation_error_prevents_weight_hash_and_model_import(files, monkeypatch):
    def fail(_handle):
        raise RuntimeError("process observation failed")

    fake_modules(monkeypatch, wait=fail)
    original = RUNNER.digest

    def digest(path):
        assert Path(path).name != "model.safetensors", "blocked wait cannot read weights"
        return original(path)

    monkeypatch.setattr(RUNNER, "digest", digest)
    monkeypatch.setattr(RUNNER, "execute", lambda *_args: pytest.fail("model must not load"))
    with pytest.raises(RuntimeError, match="observation failed"):
        RUNNER.run(files)
    assert json.loads((files.output / "receipt.json").read_text())["status"] == "error"


def test_frozen_source_tampering_prevents_weight_hash_and_load(files, monkeypatch):
    def change(_handle):
        (files.output / "sources/scripts/diagnose_mps_numerics.py").write_text("changed")

    fake_modules(monkeypatch, wait=change)
    monkeypatch.setattr(RUNNER, "execute", lambda *_args: pytest.fail("model must not load"))
    with pytest.raises(ValueError, match="frozen source/input hash differs"):
        RUNNER.run(files)


def test_execute_preserves_invalid_json_result_without_repair_or_fallback(files, monkeypatch):
    files.output.mkdir()
    receipt = RUNNER.freeze(files)
    receipt["verified_base_files"] = RUNNER.verify_base_files(files, receipt)
    model_calls, synchronizations = [], []

    processor = object()
    lm = SimpleNamespace(config=SimpleNamespace(model_type="gemma4_unified"))
    load_calls = []

    def load_processor(path, **kwargs):
        load_calls.append(("processor", path, kwargs))
        assert kwargs == {"local_files_only": True, "trust_remote_code": False}
        return processor

    def load_lm(path, **kwargs):
        load_calls.append(("model", path, kwargs))
        assert kwargs == {
            "local_files_only": True,
            "trust_remote_code": False,
            "dtype": "fake-bfloat16",
            "attn_implementation": "sdpa",
        }
        return lm

    def model(loaded_lm, loaded_processor, **kwargs):
        model_calls.append(kwargs)
        assert loaded_lm is lm and loaded_processor is processor
        assert kwargs == {"device": "mps", "temperature": 1.0, "max_context": 16384}
        return SimpleNamespace(
            device="mps",
            dtype="bfloat16",
            attn_implementation="sdpa",
            temperature=1.0,
            max_context=16384,
            compile_mode=None,
            quantization="none",
        )

    runtime = SimpleNamespace(
        torch=SimpleNamespace(
            bfloat16="fake-bfloat16",
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
            mps=SimpleNamespace(synchronize=lambda: synchronizations.append(True)),
        ),
        transformers=SimpleNamespace(
            AutoProcessor=SimpleNamespace(from_pretrained=load_processor),
            AutoModelForMultimodalLM=SimpleNamespace(from_pretrained=load_lm),
        ),
        unified=SimpleNamespace(UnifiedDecisionModel=SimpleNamespace(from_components=model)),
        profiling=SimpleNamespace(
            memory_snapshot=lambda device: {"device": device, "kind": "fake_not_peak"},
            environment_metadata=lambda: {"scope": "fake runtime"},
        ),
    )
    monkeypatch.setattr(RUNNER, "load_runtime", lambda _output: runtime)
    original = {
        "raw_text": '{"text":',
        "error": {"type": "JSONDecodeError", "message": "invalid JSON"},
    }

    def generate(_model, _torch, helper):
        assert helper == files.helper
        return original

    assert not RUNNER.execute(files, receipt, SimpleNamespace(generate_native=generate))
    assert len(model_calls) == 1
    assert len(load_calls) == 2
    assert len(synchronizations) == 2
    assert json.loads((files.output / "generation.json").read_text()) == original
    assert receipt["generation_calls"] == 1
    assert receipt["generation_error"] == original["error"]
    assert receipt["memory_after_generation"]["kind"] == "fake_not_peak"


def test_helper_hash_mismatch_fails_before_runtime_import(files, monkeypatch):
    files.output.mkdir()
    receipt = RUNNER.freeze(files)
    RUNNER.write_json(files.output / "sources/inputs/helper-record.json", {"changed": "input"})
    monkeypatch.setattr(RUNNER, "load_runtime", lambda *_args: pytest.fail("no runtime import"))
    with pytest.raises(ValueError, match="changed before model load"):
        RUNNER.execute(files, receipt, SimpleNamespace())


def test_source_tampering_during_weight_verification_prevents_load(files, monkeypatch):
    fake_modules(monkeypatch)

    def verify(args, receipt):
        (args.output / "sources/scripts/diagnose_mps_numerics.py").write_text("changed")
        return {}

    monkeypatch.setattr(RUNNER, "verify_base_files", verify)
    monkeypatch.setattr(RUNNER, "execute", lambda *_args: pytest.fail("no model load"))
    with pytest.raises(ValueError, match="during base verification"):
        RUNNER.run(files)


@pytest.mark.parametrize("mutation", ["s1_sidecar", "wrong_architecture", "quantization"])
def test_base_format_cannot_silently_switch_to_derivative_or_quantized_model(files, mutation):
    files.output.mkdir()
    path = files.base_model / "config.json"
    config = json.loads(path.read_text())
    if mutation == "s1_sidecar":
        RUNNER.write_json(files.base_model / "s1_config.json", {"temperature": 2})
    elif mutation == "wrong_architecture":
        config["model_type"] = "gemma4"
        RUNNER.write_json(path, config)
    else:
        config["quantization"] = {"bits": 4}
        RUNNER.write_json(path, config)
    with pytest.raises(ValueError, match="base control requires"):
        RUNNER.freeze(files)
