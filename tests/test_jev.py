import copy
import json

import pytest
from pydantic import ValidationError

from s1.backends import UniformBackend
from s1.contracts import DecisionRequest
from s1.errors import BackendResponseError, RequestValidationError
from s1.jev import JevCompatibilityAdapter


@pytest.fixture
def jev_payload():
    # Same named heads, rich candidates and instruction layout as upstream
    # jev_ultrafast/model.py at 1231850a0bf1a0c0341fe408ef1668dbbfdfac46.
    return {
        "model": "gemma-browser",
        "state": {
            "page": {
                "url": "http://fixture.invalid",
                "title": "Hotel search",
                "text": "Casa Flora",
            },
            "elements": [
                {
                    "index": "1",
                    "label": "Destination",
                    "value": "Lisbon",
                    "operations": ["TYPE_TEXT"],
                },
                {"index": "2", "label": "Find stays", "operations": ["CLICK"]},
            ],
            "recent_actions": [{"action": "Destination", "kind": "fill", "text": "Lisbon"}],
        },
        "questions": {
            "operation": {
                "type": "choice",
                "criteria": {
                    "CLICK": "Click",
                    "TYPE_TEXT": "Fill",
                    "DONE": "All requirements satisfied",
                },
                "instructions": {
                    "goal": "Find Design stays in Lisbon",
                    "rules": "Submit before DONE",
                },
            },
            "click_target": {
                "type": "choice",
                "criteria": {
                    "2": {"element": "[2] Find stays", "current_value": "", "role": "button"}
                },
                "instructions": {
                    "goal": "Find Design stays in Lisbon",
                    "operation": "CLICK",
                    "rules": ["Advance", "Pick target"],
                },
            },
            "type_text_target": {
                "type": "choice",
                "criteria": {
                    "1": {
                        "element": "[1] Destination",
                        "current_value": "Lisbon",
                        "role": "searchbox",
                    }
                },
                "instructions": {
                    "goal": "Find Design stays in Lisbon",
                    "operation": "TYPE_TEXT",
                    "rules": ["Advance", "Pick target"],
                },
            },
        },
    }


class CapturingBackend(UniformBackend):
    name = "gemma-native"

    def __init__(self):
        self.calls = []

    def predict(self, request):
        self.calls.append(request)
        return super().predict(request)


def test_upstream_request_preserves_goal_rules_state_and_singletons(jev_payload):
    original = copy.deepcopy(jev_payload)
    backend = CapturingBackend()
    result = JevCompatibilityAdapter(backend, model_identity="gemma-browser").predict(jev_payload)
    assert jev_payload == original
    assert len(backend.calls) == 1
    request = backend.calls[0]
    assert isinstance(request, DecisionRequest)
    assert request.state == original["state"]
    assert [q.id for q in request.questions] == ["operation"]
    assert (
        json.loads(request.questions[0].instructions)
        == original["questions"]["operation"]["instructions"]
    )
    assert result["model"] == "gemma-native"
    assert list(result["answers"]) == list(original["questions"])
    for head, label in (("click_target", "2"), ("type_text_target", "1")):
        assert result["answers"][head]["choice"] == label
        assert result["answers"][head]["probabilities"] == {label: 1.0}
    assert result["jev_compatibility"] == {
        "model_identity": "gemma-browser",
        "singleton_heads": ["click_target", "type_text_target"],
    }


def test_rich_criteria_and_instructions_have_deterministic_lossless_serialization(jev_payload):
    question = jev_payload["questions"]["click_target"]
    question["criteria"]["9"] = {
        "selected": False,
        "checked": True,
        "element": "[9] 中文",
        "current_value": "Design",
        "expanded": None,
    }
    backend = CapturingBackend()
    adapter = JevCompatibilityAdapter(backend, model_identity="gemma-browser")
    adapter.predict(jev_payload)
    native = next(q for q in backend.calls[-1].questions if q.id == "click_target")
    for label, value in question["criteria"].items():
        assert json.loads(native.criteria[label]) == value
    assert json.loads(native.instructions) == question["instructions"]
    reversed_payload = copy.deepcopy(jev_payload)
    reversed_question = reversed_payload["questions"]["click_target"]
    reversed_question["instructions"] = dict(reversed(list(question["instructions"].items())))
    reversed_question["criteria"]["9"] = dict(reversed(list(question["criteria"]["9"].items())))
    adapter.predict(reversed_payload)
    repeated = next(q for q in backend.calls[-1].questions if q.id == "click_target")
    assert repeated.instructions == native.instructions
    assert repeated.criteria == native.criteria


def test_all_singleton_heads_need_no_inference(jev_payload):
    del jev_payload["questions"]["operation"]
    backend = CapturingBackend()
    result = JevCompatibilityAdapter(backend, model_identity="gemma-browser").predict(jev_payload)
    assert not backend.calls
    assert result["usage"] == {"input_tokens": 0, "output_tokens": 0}
    assert result["model"] == backend.name


def test_requested_model_must_match_explicit_configuration(jev_payload):
    backend = CapturingBackend()
    adapter = JevCompatibilityAdapter(backend)
    with pytest.raises(RequestValidationError, match="does not switch models"):
        adapter.predict(jev_payload)
    assert not backend.calls
    del jev_payload["model"]
    assert adapter.predict(jev_payload)["model"] == backend.name


def test_oversized_target_table_is_rejected_without_truncation(jev_payload):
    jev_payload["questions"]["click_target"]["criteria"] = {
        str(i): {"element": f"[{i}] button"} for i in range(1, 54)
    }
    backend = CapturingBackend()
    with pytest.raises(
        RequestValidationError, match="click_target.*53 candidates.*52.*No targets were truncated"
    ):
        JevCompatibilityAdapter(backend, model_identity="gemma-browser").predict(jev_payload)
    assert not backend.calls


@pytest.mark.parametrize("invalid", [{}, {" ": "target"}])
def test_empty_or_blank_singleton_targets_are_rejected(jev_payload, invalid):
    jev_payload["questions"]["click_target"]["criteria"] = invalid
    with pytest.raises(RequestValidationError):
        JevCompatibilityAdapter(CapturingBackend(), model_identity="gemma-browser").predict(
            jev_payload
        )


def test_singleton_exemption_does_not_apply_to_operation_or_ordinary_contract(jev_payload):
    jev_payload["questions"]["operation"]["criteria"] = {"DONE": "Complete"}
    with pytest.raises(RequestValidationError):
        JevCompatibilityAdapter(CapturingBackend(), model_identity="gemma-browser").predict(
            jev_payload
        )
    with pytest.raises(ValidationError):
        DecisionRequest(
            state="page",
            questions={"click_target": {"instructions": "pick", "criteria": {"1": "only"}}},
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(extra=True),
        lambda p: p["questions"]["operation"].update(id="different"),
        lambda p: p["questions"]["operation"].update(extra=True),
        lambda p: p["questions"]["operation"].update(options={"A": "a", "B": "b"}),
        lambda p: p["questions"]["operation"].update(instructions={"value": float("nan")}),
        lambda p: p["questions"]["operation"]["criteria"].update(CLICK=3),
    ],
)
def test_invalid_upstream_data_retains_strict_validation(jev_payload, mutation):
    mutation(jev_payload)
    backend = CapturingBackend()
    with pytest.raises(RequestValidationError):
        JevCompatibilityAdapter(backend, model_identity="gemma-browser").predict(jev_payload)
    assert not backend.calls


def test_backend_model_and_distribution_errors_remain_gateway_errors(jev_payload):
    class Broken(CapturingBackend):
        def predict(self, request):
            result = super().predict(request)
            result["answers"]["operation"]["choice"] = "unknown"
            return result

    with pytest.raises(BackendResponseError):
        JevCompatibilityAdapter(Broken(), model_identity="gemma-browser").predict(jev_payload)

    class MissingIdentity(CapturingBackend):
        def predict(self, request):
            return {**super().predict(request), "model": None}

    with pytest.raises(BackendResponseError, match="model identity"):
        JevCompatibilityAdapter(MissingIdentity(), model_identity="gemma-browser").predict(
            jev_payload
        )
