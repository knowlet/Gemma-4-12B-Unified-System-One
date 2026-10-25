"""Decision3 publisher contracts with fake runtimes only: no Torch/model/network."""

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from s1.contracts import DecisionRequest
from s1.errors import BackendResponseError, BackendTimeoutError, RequestValidationError
from s1.evaluation import decision3 as d3
from s1.evaluation.adapters import execution_blockers, load_adapter
from s1.evaluation.contracts import ProfileSpec
from s1.evaluation.integrity import validate_runtime_telemetry
from s1.evaluation.planning import case_eligibility
from s1.evaluation.registry import Registry


@pytest.mark.parametrize("name", d3.CHECKPOINTS)
def test_all_six_catalog_and_factory_use_immutable_native_pins(name, monkeypatch):
    registry = Registry.load(
        "configs/benchmarks/models.toml",
        "configs/benchmarks/suites.toml",
        "configs/benchmarks/profiles.toml",
    )
    spec = registry.models[f"{name}-local"]
    identity = d3.CHECKPOINTS[name]
    assert not spec.enabled
    assert (spec.model_id, spec.revision) == (identity["model_id"], identity["revision"])
    assert len(spec.revision) == 40 and len(identity["manifest_sha256"]) == 64
    assert spec.context_limit == 16384
    assert spec.capabilities.max_options == 255
    assert spec.capabilities.modalities == ("text", "image")
    assert spec.capabilities.independent_questions is True
    assert execution_blockers(spec, ProfileSpec(id="p", suite="s", models=(spec.id,))) == []
    captured = []
    monkeypatch.setattr(d3, "Decision3Backend", lambda value: captured.append(value) or object())
    load_adapter(spec, {}).close()
    assert captured == [spec]


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"revision": "0" * 40}, "checkpoint_unverified"),
        ({"device": "cpu"}, "requires_cuda_bfloat16"),
        ({"precision": "float32"}, "requires_cuda_bfloat16"),
        ({"context_limit": 16385}, "project_context_limit_exceeded"),
        ({"calibration": "none"}, "policy_mismatch"),
    ],
)
def test_invalid_spec_refused_before_memory_or_import(changes, reason, monkeypatch):
    monkeypatch.setattr(
        d3, "require_memory", lambda *a, **k: pytest.fail("invalid spec reached loader")
    )
    with pytest.raises(ValueError, match=reason):
        d3.Decision3Backend(d3.decision3_spec("d3-edge").model_copy(update=changes))


def mock_memory(monkeypatch, available, total=128 * 1024**3, boot="unit-boot"):
    original = Path.read_text

    def read(path, *args, **kwargs):
        if str(path) == "/proc/meminfo":
            return f"MemTotal: {total // 1024} kB\nMemAvailable: {available // 1024} kB\nSwapTotal: 999999999 kB\nSwapFree: 999999999 kB\n"
        if str(path) == "/proc/sys/kernel/random/boot_id":
            return boot
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)


@pytest.mark.parametrize("name", ["d3-edge", "d3"])
def test_memory_formula_largest_transient_credit_once_and_no_swap(name, monkeypatch):
    mock_memory(monkeypatch, 120 * 1024**3)
    identity = d3.CHECKPOINTS[name]
    resident = identity["resident_file_bytes"]
    before = d3.memory_admission(identity, include_helper=True)
    after = d3.memory_admission(identity, include_helper=True, helper_already_loaded=True)
    expected_transient = max(resident, d3.FIXED_HELPER_BYTES)
    assert before["largest_declared_transient_bytes"] == expected_transient
    assert (
        before["required_available_bytes"]
        == resident
        + d3.FIXED_HELPER_BYTES
        + expected_transient
        + before["os_display_reserve_bytes"]
    )
    assert (
        before["required_available_bytes"] - after["required_available_bytes"]
        == d3.FIXED_HELPER_BYTES
    )
    assert after["largest_declared_transient_bytes"] == expected_transient
    assert before["swap_credit_bytes"] == 0
    assert before["readout_storage_precision"] == "float32"
    assert before["readout_file_bytes"] == before["readout_tensor_bytes"] + 80
    assert before["boot_id"] == "unit-boot"


def test_memory_floor_available_refusal_before_any_optional_import(monkeypatch):
    mock_memory(monkeypatch, 33 * 1024**3)
    monkeypatch.setattr(
        d3.metadata, "version", lambda _: pytest.fail("refusal reached model packages")
    )
    with pytest.raises(ValueError, match="memory admission refused") as caught:
        d3.Decision3Backend(d3.decision3_spec("d3"))
    assert caught.value.decision3_memory["admitted"] is False
    assert caught.value.decision3_memory["required_available_bytes"] > 33 * 1024**3


def test_memory_requires_boot_and_total_identity(monkeypatch):
    mock_memory(monkeypatch, 120 * 1024**3)
    for kwargs, match in [
        ({"expected_boot_id": "other"}, "boot identity"),
        ({"expected_memtotal": 1}, "MemTotal"),
    ]:
        with pytest.raises(ValueError, match=match):
            d3.memory_admission(d3.CHECKPOINTS["d3-edge"], **kwargs)


def test_helper_credit_requires_explicit_declared_residency(monkeypatch):
    mock_memory(monkeypatch, 120 * 1024**3)
    with pytest.raises(ValueError, match="declared helper"):
        d3.memory_admission(d3.CHECKPOINTS["d3-edge"], helper_already_loaded=True)
    with pytest.raises(ValueError, match="explicit booleans"):
        d3.memory_admission(d3.CHECKPOINTS["d3-edge"], include_helper="yes")


def package(tmp_path):
    sources = {
        name: b"raise AssertionError('publisher code must not execute in CPU test')\n"
        for name in ("modeling_d3.py", "d3_runtime.py", "d3_format.py", "d3_fast.py")
    }
    sources.update(
        {
            "config.json": b"{}",
            "tokenizer_config.json": b"{}",
            "chat_template.jinja": b"template",
            "decision_config.json": json.dumps(
                {
                    "temperature": 1.0,
                    "readout_dtype": "float32",
                    "max_length": None,
                    "attention_mode": "noncausal_full_attention",
                    "pooling": "last",
                    "codes": list(range(255)),
                }
            ).encode(),
        }
    )
    for name, data in sources.items():
        (tmp_path / name).write_bytes(data)
    manifest = {
        "schema": "d3-package-manifest/1",
        "files_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()},
    }
    raw = json.dumps(manifest).encode()
    (tmp_path / "MODEL_MANIFEST.json").write_bytes(raw)
    return {"manifest_sha256": hashlib.sha256(raw).hexdigest()}


def test_manifest_and_all_python_sources_authenticated_without_execution(tmp_path):
    identity = package(tmp_path)
    assert d3.verify_publisher_sources(tmp_path, identity)["schema"] == "d3-package-manifest/1"
    (tmp_path / "d3_fast.py").write_text("raise AssertionError('changed')\n")
    with pytest.raises(ValueError, match="source changed"):
        d3.verify_publisher_sources(tmp_path, identity)


def test_extra_source_and_manifest_tampering_refused(tmp_path):
    identity = package(tmp_path)
    (tmp_path / "shadow.py").write_text("raise AssertionError('must not run')\n")
    with pytest.raises(ValueError, match="unexpected executable"):
        d3.verify_publisher_sources(tmp_path, identity)
    (tmp_path / "MODEL_MANIFEST.json").write_text("{}")
    with pytest.raises(ValueError, match="manifest differs"):
        d3.verify_publisher_sources(tmp_path, identity)


def request():
    return DecisionRequest.model_validate(
        {
            "state": {"second": 2, "first": [1]},
            "questions": {
                "choice": {"instructions": "choose", "criteria": {"b": "same", "a": "same"}},
                "noul": {"type": "noul", "instructions": "yes?"},
                "score": {
                    "type": "score",
                    "instructions": "rate",
                    "criteria": {"10": "low", "20": "high"},
                },
            },
        }
    )


def native_response():
    return {
        "model": "d3-edge",
        "usage": {"input_tokens": 18, "output_tokens": 0},
        "answers": {
            "choice": {
                "type": "choice",
                "choice": "b",
                "probabilities": {"b": 0.7, "a": 0.3},
                "confidence": 0.01,
            },
            "noul": {"type": "noul", "noul": 0.8},
            "score": {"type": "score", "probabilities": {"0": 0.25, "1": 0.75}, "score": 0.75},
        },
    }


def bare_backend(length=16384, raw=None):
    backend = d3.Decision3Backend.__new__(d3.Decision3Backend)
    backend.spec = d3.decision3_spec("d3-edge")
    backend.identity = d3.CHECKPOINTS["d3-edge"]
    backend.calls = []

    def prepare(**payload):
        backend.calls.append(("prepare", copy.deepcopy(payload)))
        return SimpleNamespace(
            sequences={key: [0] * length for key in payload["questions"]}, lengths={}, errors={}
        )

    def system_one(**payload):
        backend.calls.append(("system_one", copy.deepcopy(payload)))
        return copy.deepcopy(native_response() if raw is None else raw)

    backend.runtime = SimpleNamespace(prepare=prepare)
    backend.model = SimpleNamespace(system_one=system_one)
    return backend


def test_native_all_question_order_calibration_raw_confidence_and_score_mapping():
    backend = bare_backend()
    result = backend.predict(request())
    assert [kind for kind, _ in backend.calls] == ["prepare", "system_one"]
    payload = backend.calls[0][1]
    assert list(payload["state"]) == ["second", "first"]
    assert list(payload["questions"]) == ["choice", "noul", "score"]
    assert list(payload["questions"]["choice"]["criteria"]) == ["b", "a"]
    assert payload["questions"]["score"]["criteria"] == ["low", "high"]
    assert result["answers"]["score"]["score"] == 17.5
    assert result["answers"]["choice"]["confidence"] == 0.7
    assert (
        result["decision3_provider_evidence"]["raw_response"]["answers"]["choice"]["confidence"]
        == 0.01
    )
    assert backend.calls[0][1] == backend.calls[1][1]


def test_context_cap_rejects_full_request_before_native_forward_and_keeps_raw_proof():
    backend = bare_backend(length=16385)
    with pytest.raises(RequestValidationError, match="nothing truncated") as caught:
        backend.predict(request())
    assert [name for name, _ in backend.calls] == ["prepare"]
    assert caught.value.decision3_provider_evidence["preflight"]["lengths"]["choice"] == 16385


def test_deadline_after_preflight_prevents_forward_and_late_raw_is_retained():
    backend = bare_backend()
    ticks = []

    def deadline():
        ticks.append(1)
        if len(ticks) == 2:
            raise BackendTimeoutError("spent")

    with pytest.raises(BackendTimeoutError):
        backend.predict(request(), deadline=deadline)
    assert [name for name, _ in backend.calls] == ["prepare"]
    backend = bare_backend()
    ticks.clear()

    def late():
        ticks.append(1)
        if len(ticks) == 3:
            raise BackendTimeoutError("late")

    with pytest.raises(BackendTimeoutError) as caught:
        backend.predict(request(), deadline=late)
    assert caught.value.decision3_provider_evidence["raw_response"] == native_response()


@pytest.mark.parametrize(
    "change",
    [
        "missing_head",
        "wrong_labels",
        "nan",
        "bool",
        "bad_mass",
        "wrong_model",
        "bad_usage",
        "native_error",
        "bad_confidence",
    ],
)
def test_invalid_native_output_never_becomes_success(change):
    raw = native_response()
    if change == "missing_head":
        raw["answers"].pop("noul")
    elif change == "wrong_labels":
        raw["answers"]["choice"]["probabilities"] = {"x": 0.7, "a": 0.3}
    elif change == "nan":
        raw["answers"]["choice"]["probabilities"]["b"] = float("nan")
    elif change == "bool":
        raw["answers"]["choice"]["probabilities"]["b"] = True
    elif change == "bad_mass":
        raw["answers"]["choice"]["probabilities"]["b"] = 0.69
    elif change == "wrong_model":
        raw["model"] = "d3"
    elif change == "bad_usage":
        raw["usage"]["input_tokens"] = True
    elif change == "native_error":
        raw["answers"]["noul"] = {"error": "max_length_exceeded"}
    else:
        raw["answers"]["choice"]["confidence"] = float("inf")
    with pytest.raises(BackendResponseError) as caught:
        bare_backend(raw=raw).predict(request())
    assert caught.value.decision3_provider_evidence["raw_response"] == raw


@pytest.mark.parametrize("count,eligible", [(2, True), (10, True), (11, False)])
def test_score_native_limit_is_explicit(count, eligible):
    req = DecisionRequest.model_validate(
        {
            "state": "x",
            "questions": {
                "q": {
                    "type": "score",
                    "instructions": "rate",
                    "criteria": [f"level{i}" for i in range(count)],
                }
            },
        }
    )
    result = case_eligibility(d3.decision3_spec("d3-edge"), SimpleNamespace(id="x", request=req))
    assert (result["eligibility"] == "eligible") is eligible


def test_duplicate_score_descriptions_and_audio_timestamp_media_refused_before_prepare():
    req = DecisionRequest.model_validate(
        {
            "state": "x",
            "questions": {
                "q": {
                    "type": "score",
                    "instructions": "rate",
                    "criteria": {"a": "same", "b": "same"},
                }
            },
        }
    )
    assert "distinct_nonempty" in d3.request_blockers(req)[0]
    for media, expected in [
        ({"type": "audio", "samples": [0.0]}, "audio"),
        ({"type": "image", "data": "unread", "timestamp_seconds": 1}, "timestamped"),
    ]:
        req = DecisionRequest.model_validate({**request().model_dump(), "media": [media]})
        backend = bare_backend()
        with pytest.raises(RequestValidationError, match=expected):
            backend.predict(req)
        assert backend.calls == []


def test_loader_requires_cached_snapshot_and_native_dtype_map_only(monkeypatch, tmp_path):
    calls = []
    runtime = SimpleNamespace(
        backbone=SimpleNamespace(
            parameters=lambda: iter([SimpleNamespace(dtype="torch.bfloat16")])
        ),
        readout_dtype="float32",
        temperature=1.0,
        max_length=None,
        attention_mode="noncausal_full_attention",
        permutation_average=False,
        batch_size=8,
    )
    monkeypatch.setattr(d3, "require_memory", lambda *a, **k: {"admitted": True})
    monkeypatch.setattr(d3.metadata, "version", lambda _: "5.17.0")
    monkeypatch.setattr(d3, "verify_publisher_sources", lambda *a: {})

    def snapshot(*args, **kwargs):
        calls.append(("snapshot", args, kwargs))
        return str(tmp_path)

    def load(*args, **kwargs):
        calls.append(("model", args, kwargs))
        return SimpleNamespace(runtime=runtime)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModel=SimpleNamespace(from_pretrained=load)),
    )
    d3.Decision3Backend(d3.decision3_spec("d3-edge"))
    assert calls[0][2] == {
        "revision": d3.CHECKPOINTS["d3-edge"]["revision"],
        "local_files_only": True,
    }
    assert calls[1][1] == (str(tmp_path),)
    assert calls[1][2] == {
        "trust_remote_code": True,
        "local_files_only": True,
        "device_map": {"": "cuda"},
        "dtype": "bfloat16",
        "batch_size": 8,
        "verify": "full",
    }


def test_telemetry_distinguishes_native_context_architecture_and_project_cap():
    backend = bare_backend()
    backend.runtime_precision = "bfloat16"
    backend.memory_preflight = {"admitted": True}
    backend.runtime.temperature = 1.0
    backend.runtime.runtime_info = lambda: {
        "fast_path": {"active": False, "reason": "not a ROCm GPU"}
    }
    backend.runtime.provenance = lambda: {"max_length": None}
    telemetry = validate_runtime_telemetry(backend, backend.spec)
    options = telemetry["runtime_options"]
    assert options["native_candidate_cap"] == 255
    assert options["native_hard_context_cap"] is None
    assert options["architectural_context_metadata"] == 262144
    assert options["project_context_cap"] == 16384
    assert options["efficacy_numerical_equivalence_and_peak_memory_verified"] is False
