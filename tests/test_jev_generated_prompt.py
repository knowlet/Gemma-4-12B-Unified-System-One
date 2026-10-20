"""Opt-in prompt representation preserves the native action validation boundary."""

import copy
import hashlib
import json

import pytest
from test_jev_generated import backend as backend
from test_jev_generated import payload as payload

from s1 import jev_generated
from s1.api import create_app
from s1.errors import BackendResponseError, RequestValidationError
from s1.jev_generated import JevGeneratedActionAdapter, format_action_request


def wire(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def test_default_production_text_is_exactly_the_original_native_text(payload):
    encoded, original, labels = jev_generated._request(payload, "experiment")
    text, source, actual_labels, evidence = format_action_request(payload, "experiment")
    assert text == jev_generated.INSTRUCTIONS + "\n" + encoded
    assert source == original == wire(payload) and actual_labels == labels
    assert evidence["format"] == "raw-json-v1"


@pytest.mark.parametrize("mode", ["raw-json-v1", "shared-elements-v1", "shared-elements-v2"])
def test_original_400_labels_and_schema401_boundary_survive(mode, backend, payload):
    payload = copy.deepcopy(payload)
    payload["questions"]["click_target"]["criteria"] = {
        str(n): {"element": f"[{n}] Choice{n}", "current_value": "", "role": "button"}
        for n in range(1, 401)
    }
    payload["state"]["elements"] = [
        {"role": "button", "index": str(n), "label": f"Choice{n}", "operations": ["CLICK"]}
        for n in range(1, 401)
    ]
    original = wire(payload)
    adapter = JevGeneratedActionAdapter(backend, model_identity="experiment", prompt_format=mode)
    result = adapter.predict(payload)
    assert wire(payload) == original
    assert result["generation"]["candidate_labels"]["click_target"] == list(map(str, range(1, 401)))
    assert len(result["answers"]["click_target"]["probabilities"]) == 400
    assert result["generation"]["original_request_sha256"] == hashlib.sha256(original).hexdigest()
    assert result["generation"]["prompt_representation"]["format"] == mode
    before_calls = len(backend.model.calls)
    payload["questions"]["click_target"]["criteria"]["401"] = "Still offered; never prune"
    with pytest.raises(RequestValidationError, match="candidate limit"):
        adapter.predict(payload)
    assert len(backend.model.calls) == before_calls


@pytest.mark.parametrize("mode", ["shared-elements-v1", "shared-elements-v2"])
def test_shared_roundtrip_failure_stops_before_native_forward(monkeypatch, backend, payload, mode):
    from s1 import jev_prompt

    monkeypatch.setattr(jev_prompt, "unpack_payload", lambda _encoded: {})
    with pytest.raises(RequestValidationError, match="complete ordered request"):
        JevGeneratedActionAdapter(backend, model_identity="experiment", prompt_format=mode).predict(
            payload
        )
    assert not backend.model.calls and not backend.model.forwards


@pytest.mark.parametrize("length,accepted", [(8160, True), (8161, False)])
@pytest.mark.parametrize("mode", ["shared-elements-v1", "shared-elements-v2"])
def test_shared_prefix_keeps_full_8192_budget_and_32_output_reserve(
    backend, payload, length, accepted, mode
):
    backend.model.max_context = 16384
    backend.model.processor.length = length
    adapter = JevGeneratedActionAdapter(backend, model_identity="experiment", prompt_format=mode)
    if accepted:
        assert adapter.predict(payload)["usage"]["input_tokens"] == length
    else:
        with pytest.raises(RequestValidationError, match="context exceeds budget"):
            adapter.predict(payload)
        assert not backend.model.calls and not backend.model.forwards


def test_shared_output_still_requires_original_key_and_not_description(backend, payload):
    backend.model.generated = '{"operation":"CLICK","target":"not an offered key"}'
    with pytest.raises(BackendResponseError, match="does not belong"):
        JevGeneratedActionAdapter(
            backend, model_identity="experiment", prompt_format="shared-elements-v1"
        ).predict(payload)


def test_shared_mode_keeps_grammar_preflight(backend, payload):
    with pytest.raises(RequestValidationError, match="grammar preflight"):
        JevGeneratedActionAdapter(
            backend,
            model_identity="experiment",
            constrain_actions=True,
            prompt_format="shared-elements-v1",
        ).predict(payload)
    assert not backend.model.calls and not backend.model.forwards


@pytest.mark.parametrize("value", [None, True, 1, "automatic", "shared-elements"])
def test_prompt_mode_is_explicit_and_rejected_before_serving(backend, value):
    with pytest.raises(ValueError, match="prompt format"):
        create_app(backend, jev_generated_prompt_format=value)
    with pytest.raises(ValueError, match="prompt format"):
        JevGeneratedActionAdapter(backend, prompt_format=value)


def test_shared_mode_requires_generated_actions(backend):
    with pytest.raises(ValueError, match="requires generated actions"):
        create_app(backend, jev_compatible=True, jev_generated_prompt_format="shared-elements-v1")


def test_v2_fallback_keeps_raw_text_and_discloses_both_candidate_costs(payload):
    raw, source, labels, _ = format_action_request(payload, "experiment")
    chosen, original, offered, evidence = format_action_request(
        payload, "experiment", prompt_format="shared-elements-v2"
    )
    assert chosen == raw and source == original and wire(labels) == wire(offered)
    assert evidence["requested_format"] == "shared-elements-v2"
    assert evidence["format"] == "raw-json-v1"
    assert evidence["selection_basis"] == "complete_user_text_utf8_bytes"
    costs = evidence["candidate_user_text_bytes"]
    assert costs["raw-json-v1"] == len(raw.encode())
    assert costs["shared-elements-v2"] >= costs["raw-json-v1"]
    assert evidence["encoded_sha256"] == hashlib.sha256(original).hexdigest()
    assert evidence["element_rows"] == evidence["criterion_refs"] == 0
    assert evidence["candidate_shared_representation"]["format"] == "shared-elements-v2"
    assert evidence["fallback_reason"] == "shared_user_text_is_not_smaller"


def test_v1_remains_explicit_even_when_its_prompt_is_larger(payload):
    from s1.jev_prompt import SHARED_ELEMENT_INSTRUCTIONS, format_payload

    raw, *_ = format_action_request(payload, "experiment")
    text, _, _, evidence = format_action_request(
        payload, "experiment", prompt_format="shared-elements-v1"
    )
    packed, _ = format_payload(payload, prompt_format="shared-elements-v1")
    assert text == jev_generated.INSTRUCTIONS + "\n" + SHARED_ELEMENT_INSTRUCTIONS + "\n" + packed
    assert len(text.encode()) > len(raw.encode())
    assert evidence["format"] == "shared-elements-v1"
    assert "fallback_reason" not in evidence and "requested_format" not in evidence


@pytest.mark.parametrize("mode", ["shared-elements-v1", "shared-elements-v2"])
def test_http_opt_in_uses_the_same_production_renderer(backend, payload, mode):
    from fastapi.testclient import TestClient

    expected_text, original, labels, _ = format_action_request(
        payload, "experiment", prompt_format=mode
    )
    app = create_app(
        backend,
        jev_compatible=True,
        jev_generated_actions=True,
        jev_generated_prompt_format=mode,
        model_identity="experiment",
        api_key="test-secret",
    )
    with TestClient(app) as client:
        assert client.post("/v1/systemone", json=payload).status_code == 401
        response = client.post(
            "/v1/systemone", json=payload, headers={"Authorization": "Bearer test-secret"}
        )
    assert response.status_code == 200, response.text
    result = response.json()
    assert backend.model.processor.messages[0][0]["content"][0]["text"] == expected_text
    assert result["generation"]["candidate_labels"] == labels
    assert result["generation"]["original_request_sha256"] == hashlib.sha256(original).hexdigest()
    assert result["generation"]["prompt_representation"]["ordered_roundtrip_verified"] is True
    assert len(backend.model.calls) == 1 and not backend.native_calls
