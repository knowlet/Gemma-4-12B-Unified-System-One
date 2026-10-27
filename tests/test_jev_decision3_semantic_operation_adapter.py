"""Invented synthetic semantic native adapter tests; no model or GPU."""

import copy
import hashlib
import math
import struct
from types import SimpleNamespace

import jev_decision3_semantic_operation_adapter as adapter
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


def fake_sha(tensor):
    return hashlib.sha256(
        b"".join(struct.pack("<f", value) for value in flatten(tensor.values))
    ).hexdigest()


@pytest.fixture
def synthetic(monkeypatch):
    monkeypatch.setattr(adapter, "SOURCE_ONLY", False)
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
    monkeypatch.setattr(
        adapter.SemanticDualHeadDecision3Backend, "_validate_runtime", lambda _: None
    )
    monkeypatch.setattr(adapter.features_module, "_validate_provider_spec", lambda _: None)

    def index_select(tensor, axis, indexes):
        assert axis == 1
        return Tensor(
            [[row[index] for index in indexes.values] for row in tensor.values],
            device=str(tensor.device),
            dtype=tensor.dtype,
        )

    def where(mask, true, false):
        return Tensor(
            [
                [a if flag else b for flag, a, b in zip(flags, left, right, strict=True)]
                for flags, left, right in zip(mask.values, true.values, false.values, strict=True)
            ],
            device=str(true.device),
            dtype=true.dtype,
        )

    monkeypatch.setattr(Torch, "index_select", staticmethod(index_select), raising=False)
    monkeypatch.setattr(Torch, "where", staticmethod(where), raising=False)
    verify_contract = adapter.features_module._verify_call_contract

    def verify_tiny(events, native, lengths, first, norm, result):
        assert first == norm == (len(lengths), max(lengths.values()), 2)
        return verify_contract(
            events,
            native,
            lengths,
            (len(lengths), max(lengths.values()), 1024),
            (len(lengths), max(lengths.values()), 1024),
            result,
        )

    monkeypatch.setattr(adapter.features_module, "_verify_call_contract", verify_tiny)
    provider = Provider(adapter)
    provider.model = SimpleNamespace(runtime=provider.runtime)
    provider.name = "fake-d3"
    monkeypatch.setattr(adapter.features_module, "W0_SHA256", fake_sha(provider.runtime.readout))
    deltas = {
        "operation": Tensor([[4.0, 0.0], [0.0, 0.0]], device="cpu"),
        "click_target": Tensor([[0.0, 0.0], [4.0, 0.0], [100.0, 100.0]], device="cpu"),
    }
    names = ["invented-0", "invented-1"]
    names_sha = hashlib.sha256(adapter.features_module._canonical(names)).hexdigest()
    hashes = {name: fake_sha(value) for name, value in deltas.items()}
    witnesses = []
    instance = adapter.SemanticDualHeadDecision3Backend(
        provider,
        deltas,
        hashes,
        Torch,
        operation_names=names,
        operation_names_sha256=names_sha,
        identity_guard=lambda *_: {"invented": "guard"},
        witness_sink=lambda **w: witnesses.append(w),
    )
    return instance, provider, deltas, hashes, names, names_sha, witnesses


def request(order):
    value = request_fixture(names=["click_target", "operation"], counts=[2, 2])
    question = value.questions[1]
    question.criteria = {name: {"invented": name} for name in order}
    return value


def test_same_semantic_choice_survives_operation_order_and_preserves_click(synthetic):
    instance, provider, _, hashes, names, names_sha, witnesses = synthetic
    for order in (names, list(reversed(names))):
        value = request(order)
        before = copy.deepcopy(value.model_dump())
        result = instance.predict(value, deadline=lambda: None)
        assert value.model_dump() == before
        assert result["answers"]["operation"]["choice"] == "invented-0"
        assert result["answers"]["click_target"]["choice"] == "invented-1"
        witness = witnesses[-1]
        assert witness["operation_names"] == names
        assert witness["operation_names_sha256"] == names_sha
        assert (
            result["decision3_provider_evidence"]["composition"]["operation_names_sha256"]
            == names_sha
        )
        assert witness["delta_sha256"] == hashes
        assert list(witness["composed_raw"]["answers"]["operation"]["probabilities"]) == order
        assert witness["zero_raw"]["answers"]["operation"]["probabilities"] == pytest.approx(
            witness["baseline_raw"]["answers"]["operation"]["probabilities"]
        )
    assert provider.predict_calls == provider.runtime.forward_calls == 2


def test_unknown_operation_refused_before_forward_or_witness(synthetic):
    instance, provider, *_rest, witnesses = synthetic
    with pytest.raises(ValueError, match="unknown operation"):
        instance.predict(request(["invented-0", "unknown"]), deadline=lambda: None)
    assert provider.predict_calls == 0
    assert witnesses == []


def test_semantic_zero_delta_and_invalid_tail_preserve_native(synthetic):
    _, provider, originals, _, names, names_sha, witnesses = synthetic
    zero_deltas = {key: Torch.zeros_like(value) for key, value in originals.items()}
    zero_pins = {key: fake_sha(value) for key, value in zero_deltas.items()}
    instance = adapter.SemanticDualHeadDecision3Backend(
        provider,
        zero_deltas,
        zero_pins,
        Torch,
        operation_names=names,
        operation_names_sha256=names_sha,
        identity_guard=lambda *_: {"invented": "guard"},
        witness_sink=lambda **value: witnesses.append(value),
    )
    instance.predict(request(["invented-1", "invented-0"]), deadline=lambda: None)
    for head in ("operation", "click_target"):
        assert witnesses[0]["composed_raw"]["answers"][head]["probabilities"] == pytest.approx(
            witnesses[0]["baseline_raw"]["answers"][head]["probabilities"]
        )
        assert len(witnesses[0]["composed_raw"]["answers"][head]["probabilities"]) == 2
    assert provider.runtime.forward_calls == 1


def test_fp64_projection_reference_ignores_large_illegal_tail(synthetic):
    instance, *_ = synthetic
    hidden = Tensor([[2.0, 1.0]])
    base = Tensor([[0.0, 1.0, 10000.0]])
    values, _ = adapter.project(
        Torch,
        base,
        hidden,
        instance.deltas,
        [2],
        ["operation"],
        [["invented-1", "invented-0"]],
        instance.operation_names,
    )
    # Name mapping adds 0 to the first legal score and 8 to the second;
    # the giant third score is masked before softmax.
    expected_second = 1.0 / (1.0 + math.exp(-9.0))
    assert values[0] == pytest.approx([1.0 - expected_second, expected_second], abs=1e-12)


def test_bad_vocabulary_pin_and_mutation_rejected(synthetic):
    instance, provider, deltas, hashes, names, _, _ = synthetic
    with pytest.raises(ValueError, match="vocabulary"):
        adapter.SemanticDualHeadDecision3Backend(
            provider,
            deltas,
            hashes,
            Torch,
            operation_names=names,
            operation_names_sha256="0" * 64,
            identity_guard=lambda *_: {"ok": True},
            witness_sink=lambda **_: None,
        )
    instance.operation_names = tuple(reversed(names))
    with pytest.raises(ValueError, match="semantic vocabulary changed"):
        instance.predict(request(names), deadline=lambda: None)
    assert provider.predict_calls == 0


def test_replacing_delta_and_rewriting_public_hash_cannot_bypass_external_pins(synthetic):
    instance, provider, *_ = synthetic
    instance.deltas["operation"].values[0][0] += 1.0
    instance.delta_sha256["operation"] = fake_sha(instance.deltas["operation"])
    with pytest.raises(ValueError, match="external delta pins changed"):
        instance.predict(request(["invented-0", "invented-1"]), deadline=lambda: None)
    assert provider.predict_calls == 0


@pytest.mark.parametrize("changed", [True, 1.0, {"nested": 2}])
def test_mutable_nested_owner_identity_snapshot_is_type_sensitive(synthetic, changed):
    _, provider, deltas, hashes, names, names_sha, _ = synthetic
    state = {"value": 1} if not isinstance(changed, dict) else {"value": {"nested": 1}}
    instance = adapter.SemanticDualHeadDecision3Backend(
        provider,
        deltas,
        hashes,
        Torch,
        operation_names=names,
        operation_names_sha256=names_sha,
        identity_guard=lambda *_: state,
        witness_sink=lambda **_: None,
    )
    state["value"] = changed
    with pytest.raises(ValueError, match="owner identity changed"):
        instance.predict(request(names), deadline=lambda: None)
    assert provider.predict_calls == 0


def test_gate_prevents_entry(synthetic, monkeypatch):
    instance, *_ = synthetic
    monkeypatch.setattr(adapter, "SOURCE_ONLY", True)
    with pytest.raises(ValueError, match="independent review"):
        instance.predict(request(["invented-0", "invented-1"]), deadline=lambda: None)
