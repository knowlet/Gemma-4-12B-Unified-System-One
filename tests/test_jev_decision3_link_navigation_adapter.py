"""Synthetic dual-head adapter controls; no GPU, model or provider is started."""

import copy
import hashlib
import struct
from types import SimpleNamespace

import jev_decision3_link_navigation_adapter as adapter
import pytest
from test_jev_decision3_operation_gpu import (
    Provider,
    RequestDTO,
    Tensor,
    Torch,
    dto_payload,
    flatten,
    normalizer,
    request_fixture,
)

REAL_RUNTIME_VALIDATOR = adapter.DualHeadDecision3Backend._validate_runtime


def fake_sha(tensor):
    return hashlib.sha256(
        b"".join(struct.pack("<f", value) for value in flatten(tensor.values))
    ).hexdigest()


@pytest.fixture
def synthetic(monkeypatch):
    def fake_where(mask, when_true, when_false):
        values = [
            [a if m else b for m, a, b in zip(ms, xs, ys, strict=True)]
            for ms, xs, ys in zip(mask.values, when_true.values, when_false.values, strict=True)
        ]
        return Tensor(values, device=str(when_true.device), dtype=when_true.dtype)

    monkeypatch.setattr(Torch, "where", staticmethod(fake_where), raising=False)
    monkeypatch.setattr(adapter, "CODES", 3)
    monkeypatch.setattr(adapter, "WIDTH", 2)
    monkeypatch.setattr(
        adapter,
        "MODEL_IDENTITY",
        {
            "name": "d3-edge",
            "model_sha256": "fake",
            "model_id": "vllm-sr/d3-edge",
            "revision": "fake",
            "manifest_sha256": "fake",
        },
        raising=False,
    )
    monkeypatch.setattr(adapter, "CONTEXT_LIMIT", 16384, raising=False)
    monkeypatch.setattr(adapter, "Decision3NativeRequest", RequestDTO)
    monkeypatch.setattr(adapter, "native_payload", dto_payload)
    monkeypatch.setattr(adapter, "normalize_native_response", normalizer)
    monkeypatch.setattr(adapter, "_tensor_sha", fake_sha)
    monkeypatch.setattr(adapter.DualHeadDecision3Backend, "_validate_runtime", lambda _: None)
    monkeypatch.setattr(adapter.features_module, "_validate_provider_spec", lambda _: None)
    verify_contract = adapter.features_module._verify_call_contract

    def verify_tiny(events, native, lengths, first, norm, result):
        assert first == norm == (len(lengths), max(lengths.values()), 2)
        full_shape = (len(lengths), max(lengths.values()), 1024)
        return verify_contract(events, native, lengths, full_shape, full_shape, result)

    monkeypatch.setattr(adapter.features_module, "_verify_call_contract", verify_tiny)
    provider = Provider(adapter)
    provider.model = SimpleNamespace(runtime=provider.runtime)
    provider.name = "fake-d3"
    monkeypatch.setattr(adapter.features_module, "W0_SHA256", fake_sha(provider.runtime.readout))
    deltas = {
        "operation": Tensor([[4.0, 0.0], [0.0, 0.0], [0.0, 0.0]], device="cpu"),
        "click_target": Tensor([[0.0, 0.0], [4.0, 0.0], [100.0, 100.0]], device="cpu"),
    }
    hashes = {name: fake_sha(value) for name, value in deltas.items()}
    witnesses = []

    def guard(_base, _deltas):
        return {"invented": "guard"}

    def witness(**values):
        witnesses.append(values)

    instance = adapter.DualHeadDecision3Backend(
        provider, deltas, hashes, Torch, identity_guard=guard, witness_sink=witness
    )
    return instance, provider, deltas, hashes, witnesses


def test_dual_head_single_forward_preserves_other_heads_and_zero_control(synthetic):
    instance, provider, _, hashes, witnesses = synthetic
    request = request_fixture(names=["click_target", "other", "operation"], counts=[2, 2, 3])
    before = copy.deepcopy(request.model_dump())
    events = []
    result = instance.predict(
        request,
        deadline=lambda: None,
        evidence_sink=lambda event, value: events.append((event, value)),
    )
    assert provider.predict_calls == provider.runtime.forward_calls == 1
    assert request.model_dump() == before
    assert [event for event, _ in events] == ["native_preflight", "native_raw_returned"]
    assert len(witnesses) == 1
    w = witnesses[0]
    assert w["delta_sha256"] == hashes
    assert w["ordered_head_ids"] == ["click_target", "other", "operation"]
    assert w["projection"]["applied_head_ids"] == ["click_target", "operation"]
    assert w["projection"]["absent_head_ids"] == []
    assert w["projection"]["legal_counts"] == [2, 2, 3]
    assert w["baseline_raw"]["answers"]["other"] == w["composed_raw"]["answers"]["other"]
    for head in w["ordered_head_ids"]:
        assert list(w["baseline_raw"]["answers"][head]["probabilities"]) == list(
            w["composed_raw"]["answers"][head]["probabilities"]
        )
        assert w["zero_raw"]["answers"][head]["probabilities"] == pytest.approx(
            w["baseline_raw"]["answers"][head]["probabilities"]
        )
    assert w["composed_raw"]["answers"]["click_target"]["choice"] == "invented-1"
    assert w["composed_raw"]["answers"]["operation"]["choice"] == "invented-0"
    assert result["decision3_provider_evidence"]["raw_response"] == w["composed_raw"]
    assert not provider.runtime.backbone.language_model.layers[0].pre_hooks
    assert not provider.runtime.backbone.language_model.norm.hooks


@pytest.mark.parametrize("names", [["operation"], ["click_target"], ["other"]])
def test_absent_heads_are_neither_created_nor_supervised(synthetic, names):
    instance, provider, _, _, witnesses = synthetic
    request = request_fixture(names=names, counts=[2] * len(names))
    instance.predict(request, deadline=lambda: None)
    w = witnesses[0]
    assert list(w["composed_raw"]["answers"]) == names
    assert w["projection"]["absent_head_ids"] == [
        head for head in adapter.HEADS if head not in names
    ]
    assert provider.runtime.forward_calls == 1


def test_rejects_wrong_delta_digest_and_witness_mutation(synthetic):
    instance, provider, deltas, hashes, _ = synthetic
    bad_hashes = dict(hashes, operation="f" * 64)
    with pytest.raises(ValueError, match="SHA256"):
        adapter.DualHeadDecision3Backend(
            provider,
            deltas,
            bad_hashes,
            Torch,
            identity_guard=lambda *_: {"ok": True},
            witness_sink=lambda **_: None,
        )
    instance.witness_sink = lambda **values: values["features"].values[0].__setitem__(0, 999.0)
    with pytest.raises(ValueError, match="witness mutated"):
        instance.predict(request_fixture(names=["operation"], counts=[2]), deadline=lambda: None)
    assert not provider.runtime.backbone.language_model.layers[0].pre_hooks
    assert not provider.runtime.backbone.language_model.norm.hooks


def test_provider_failure_cleans_both_hooks_and_keeps_primary(synthetic):
    instance, provider, *_ = synthetic
    provider.failure = RuntimeError("provider failure")
    with pytest.raises(RuntimeError, match="provider failure"):
        instance.predict(request_fixture(names=["operation"], counts=[2]), deadline=lambda: None)
    assert not provider.runtime.backbone.language_model.layers[0].pre_hooks
    assert not provider.runtime.backbone.language_model.norm.hooks


def test_real_native_http_adapter_accepts_backend_interface(synthetic, monkeypatch):
    import s1.jev_decision3_native as native

    instance, provider, *_ = synthetic
    instance.spec = SimpleNamespace(adapter="decision3", model_id="vllm-sr/d3-edge")
    monkeypatch.setattr(adapter.features_module, "_validate_provider_spec", lambda _: None)
    monkeypatch.setattr(native, "configuration_blockers", lambda _: [])
    monkeypatch.setattr(native, "Decision3NativeRequest", RequestDTO)
    monkeypatch.setattr(native, "normalize_native_response", normalizer)
    request = request_fixture(names=["click_target", "operation"], counts=[2, 2])
    monkeypatch.setattr(
        native, "parse_native_payload", lambda payload, **_: (request, payload, b"{}", {})
    )
    native_adapter = native.JevDecision3NativeAdapter(instance, model_identity="invented")
    result = native_adapter.predict({"invented": True})
    assert result["model"] == "invented"
    assert len(result["answers"]) == 2
    assert provider.predict_calls == provider.runtime.forward_calls == 1


def test_same_call_witness_connects_real_writer_and_fp64_audit(synthetic, monkeypatch, tmp_path):
    import jev_decision3_link_navigation_adapter_audit as audit
    import numpy as np

    instance, provider, _, hashes, _ = synthetic
    monkeypatch.setattr(
        Tensor, "numpy", lambda self: np.array(self.values, dtype=np.float32), raising=False
    )
    monkeypatch.setattr(audit, "WIDTH", 2)
    monkeypatch.setattr(audit, "CODES", 3)
    monkeypatch.setattr(audit, "W0_SHA256", fake_sha(provider.runtime.readout))
    sink, receipts = audit.make_witness_writer(tmp_path / "private-witness", maximum_calls=1)
    instance.witness_sink = sink
    instance.predict(
        request_fixture(names=["click_target", "other", "operation"], counts=[2, 2, 3]),
        deadline=lambda: None,
    )
    assert len(receipts) == 1
    receipt = receipts[0]
    result = audit.audit_directory(
        receipt["directory"],
        expected_file_sha256=receipt["manifest_sha256"],
        expected_witness_sha256=receipt["witness_sha256"],
        expected_delta_sha256=hashes,
    )
    assert result["fit_eligible"] is False
    assert provider.runtime.forward_calls == 1


def test_zero_deltas_match_native_probabilities_with_singleton(synthetic):
    instance, provider, _, _, witnesses = synthetic
    instance.deltas = {key: Torch.zeros_like(value) for key, value in instance.deltas.items()}
    instance.delta_sha256 = {key: fake_sha(value) for key, value in instance.deltas.items()}
    instance.predict(
        request_fixture(names=["operation", "click_target", "other"], counts=[1, 2, 2]),
        deadline=lambda: None,
    )
    w = witnesses[0]
    assert w["baseline_raw"]["answers"]["operation"]["probabilities"] == {"invented-0": 1.0}
    for head in w["ordered_head_ids"]:
        assert w["composed_raw"]["answers"][head]["probabilities"] == pytest.approx(
            w["baseline_raw"]["answers"][head]["probabilities"]
        )
    assert provider.runtime.forward_calls == 1


def test_bad_cpu_delta_precision_device_and_mutation_are_rejected(synthetic):
    instance, provider, deltas, hashes, _ = synthetic
    for value in (
        deltas["operation"].to(dtype=Torch.bfloat16),
        deltas["operation"].to(device="cuda:0"),
    ):
        invalid = dict(deltas, operation=value)
        with pytest.raises(ValueError, match="CPU FP32"):
            adapter.DualHeadDecision3Backend(
                provider,
                invalid,
                hashes,
                Torch,
                identity_guard=lambda *_: {"ok": True},
                witness_sink=lambda **_: None,
            )
    instance.deltas["operation"].values[0][0] += 1.0
    with pytest.raises(ValueError, match="delta bytes changed"):
        instance.predict(request_fixture(names=["operation"], counts=[2]), deadline=lambda: None)
    assert provider.predict_calls == 0


def test_candidate_device_and_count_validation_never_prunes(synthetic):
    instance, _, _, _, _ = synthetic
    base = Tensor([[0.0, 1.0, 2.0]])
    hidden = Tensor([[1.0, 0.0]])
    before = copy.deepcopy(base.values)
    for counts in ([0], [4], [True]):
        with pytest.raises(ValueError, match="geometry"):
            adapter.project(Torch, base, hidden, instance.deltas, counts, ["operation"])
    with pytest.raises(ValueError, match="geometry"):
        adapter.project(Torch, base.to(device="cpu"), hidden, instance.deltas, [2], ["operation"])
    assert base.values == before


def test_closed_adapter_rejects_calls_without_closing_shared_provider(synthetic):
    instance, provider, *_ = synthetic
    instance.close()
    assert provider.close_calls == 0
    with pytest.raises(ValueError, match="open"):
        instance.predict(request_fixture(names=["operation"], counts=[2]), deadline=lambda: None)


def test_real_dto_and_native_adapter_contract_with_synthetic_provider(synthetic, monkeypatch):
    import test_jev_decision3_operation_gpu as fake_support

    import s1.jev_decision3_native as native
    from s1.evaluation.decision3 import native_payload, normalize_native_response

    instance, provider, *_ = synthetic
    monkeypatch.setattr(adapter, "Decision3NativeRequest", native.Decision3NativeRequest)
    monkeypatch.setattr(adapter, "native_payload", native_payload)
    monkeypatch.setattr(adapter, "normalize_native_response", normalize_native_response)
    monkeypatch.setattr(fake_support, "RequestDTO", native.Decision3NativeRequest)
    monkeypatch.setattr(native, "configuration_blockers", lambda _: [])
    instance.spec = SimpleNamespace(adapter="decision3", model_id="vllm-sr/d3-edge")
    wire = {
        "model": "wire-alias",
        "state": {"page": {"title": "synthetic"}},
        "questions": {
            "click_target": {
                "type": "choice",
                "instructions": "Choose",
                "criteria": {"first": "first", "second": "second"},
            },
            "operation": {
                "type": "choice",
                "instructions": "Choose",
                "criteria": {"CLICK": "click", "DONE": "done"},
            },
        },
    }
    native_adapter = native.JevDecision3NativeAdapter(instance, model_identity="wire-alias")
    result = native_adapter.predict(wire)
    assert result["model"] == "wire-alias"
    assert len(result["answers"]) == 2
    assert provider.predict_calls == provider.runtime.forward_calls == 1
    assert list(result["answers"][0]["probabilities"]) == ["first", "second"]


def test_runtime_validator_checks_original_identity_and_frozen_policy(synthetic):
    instance, provider, *_ = synthetic
    provider.runtime.encoder = SimpleNamespace(
        language_model=provider.runtime.backbone.language_model
    )
    provider.runtime.parameter.numel = lambda: 594582720 - provider.runtime.readout.numel()
    provider.identity = {
        "model_id": adapter.features_module.MODEL_ID,
        "manifest_sha256": adapter.features_module.MANIFEST_SHA256,
    }
    provider.runtime.manifest["identity"]["model_sha256"] = adapter.features_module.MODEL_SHA256
    REAL_RUNTIME_VALIDATOR(instance)
    for attribute, value in (
        ("temperature", 2.0),
        ("max_length", 1024),
        ("attention_mode", "causal"),
        ("fast", True),
    ):
        original = getattr(provider.runtime, attribute)
        setattr(provider.runtime, attribute, value)
        with pytest.raises(ValueError, match="reference runtime"):
            REAL_RUNTIME_VALIDATOR(instance)
        setattr(provider.runtime, attribute, original)
    provider.runtime.parameter.requires_grad = True
    with pytest.raises(ValueError, match="frozen CUDA"):
        REAL_RUNTIME_VALIDATOR(instance)


def test_predict_and_close_are_atomically_serialized(synthetic, monkeypatch):
    import threading

    instance, provider, *_ = synthetic
    started, release = threading.Event(), threading.Event()
    outcomes = []

    def blocked(_request, *, deadline, evidence_sink):
        started.set()
        assert release.wait(3)
        return "completed"

    monkeypatch.setattr(instance, "_predict", blocked)

    def worker():
        try:
            outcomes.append(instance.predict(object(), deadline=lambda: None))
        except BaseException as error:
            outcomes.append(error)

    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert started.wait(3)
        with pytest.raises(ValueError, match="serialized"):
            instance.predict(object(), deadline=lambda: None)
        with pytest.raises(ValueError, match="active"):
            instance.close()
    finally:
        release.set()
        thread.join(timeout=3)
    assert not thread.is_alive()
    assert outcomes == ["completed"]
    assert provider.predict_calls == 0
    instance.close()
    with pytest.raises(ValueError, match="serialized"):
        instance.predict(object(), deadline=lambda: None)
