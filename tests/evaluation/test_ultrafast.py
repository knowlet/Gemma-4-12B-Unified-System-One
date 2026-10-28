import copy
import json
import math

import pytest

from s1.backends import UniformBackend, UnsupportedRequest
from s1.errors import BackendResponseError, RequestValidationError
from s1.evaluation.ultrafast import (
    TYPESAFE_DECISION_URL,
    UltrafastDecisionBridge,
    map_decision_request,
)


def payload():
    return {
        "model": "jev-latest",
        "state": {
            "page": {"url": "http://127.0.0.1/", "title": "旅館", "text": "Design stays"},
            "elements": [{"index": "1", "label": "Destination", "operations": ["TYPE_TEXT"]}],
            "recent_actions": [{"action": "Click hotels", "page_changed": True}],
        },
        "questions": {
            "operation": {
                "type": "choice",
                "instructions": {"goal": "Find a stay", "rules": ["Use observed controls"]},
                "criteria": {"TYPE_TEXT": "Write a value", "DONE": "Requirements visible"},
            },
            "type_text_target": {
                "type": "choice",
                "instructions": {"goal": "Find a stay", "operation": "TYPE_TEXT"},
                "criteria": {"1": {"element": "[1] Destination", "current_value": ""}},
            },
        },
    }


class RecordingBackend(UniformBackend):
    def __init__(self):
        self.requests = []

    def predict(self, request):
        self.requests.append(copy.deepcopy(request))
        return super().predict(request)


def test_maps_original_objects_and_state_without_semantic_rewrite():
    body = payload()
    body["questions"]["operation"]["criteria"]["TYPE_TEXT"] = {
        "element": "Destination",
        "role": "textbox",
        "checked": False,
    }
    original = copy.deepcopy(body)
    request, singletons = map_decision_request(body)
    assert request.state == body["state"]
    assert request.state is not body["state"]
    question = request.questions[0]
    assert question.labels() == ["TYPE_TEXT", "DONE"]
    assert json.loads(question.instructions) == body["questions"]["operation"]["instructions"]
    assert (
        json.loads(question.criteria["TYPE_TEXT"])
        == body["questions"]["operation"]["criteria"]["TYPE_TEXT"]
    )
    assert question.criteria["DONE"] == "Requirements visible"
    assert list(singletons) == ["type_text_target"]
    assert body == original


def test_mixed_heads_use_one_backend_call_and_preserve_question_order():
    backend = RecordingBackend()
    bridge = UltrafastDecisionBridge(backend)
    body = payload()
    body["questions"] = dict(reversed(list(body["questions"].items())))
    body["questions"]["click_target"] = {
        "type": "choice",
        "instructions": {"goal": "Find a stay", "operation": "CLICK"},
        "criteria": {"2": {"element": "[2] Go"}, "3": {"element": "[3] Cancel"}},
    }
    result = bridge(TYPESAFE_DECISION_URL, "never-record-this-key", body)
    assert bridge.backend_calls == len(backend.requests) == 1
    assert [q.id for q in backend.requests[0].questions] == ["operation", "click_target"]
    assert list(result["answers"]) == list(body["questions"])
    assert result["answers"]["type_text_target"]["choice"] == "1"
    assert result["answers"]["type_text_target"]["probabilities"] == {"1": 1.0}
    record = bridge.records[0]
    assert record["body"] == body
    assert record["mapped_request"] == backend.requests[0].model_dump()
    assert record["raw_response"]["answers"]["operation"]["probabilities"] == {
        "TYPE_TEXT": 0.5,
        "DONE": 0.5,
    }
    assert record["normalized_response"] == result
    assert record["singleton_heads"] == ["type_text_target"]
    assert record["reported_cost_usd"] is None
    assert record["backend_calls"] == 1
    assert record["error"] is None
    assert record["latency_ms"] >= 0
    assert result["usage"] is None
    assert "never-record-this-key" not in json.dumps(record)


def test_only_singletons_are_explicit_and_require_no_model_call():
    backend = RecordingBackend()
    bridge = UltrafastDecisionBridge(backend)
    body = payload()
    del body["questions"]["operation"]
    result = bridge(TYPESAFE_DECISION_URL, None, body)
    assert result["answers"]["type_text_target"]["confidence"] == 1.0
    assert bridge.backend_calls == 0
    assert backend.requests == []
    assert bridge.records[0]["mapped_request"] is None
    assert bridge.records[0]["singleton_heads"] == ["type_text_target"]


def test_all_candidates_at_contract_limit_are_kept_and_oversize_is_rejected():
    bridge = UltrafastDecisionBridge(RecordingBackend())
    body = payload()
    criteria = {str(i): {"element": f"[{i}] Button"} for i in range(52)}
    body["questions"]["type_text_target"]["criteria"] = criteria
    result = bridge(TYPESAFE_DECISION_URL, None, body)
    assert list(result["answers"]["type_text_target"]["probabilities"]) == list(criteria)
    criteria["52"] = {"element": "[52] Last button"}
    with pytest.raises(UnsupportedRequest, match="nothing was truncated"):
        bridge(TYPESAFE_DECISION_URL, None, body)
    assert bridge.backend_calls == 1
    assert bridge.records[-1]["body"] == body
    assert bridge.records[-1]["backend_calls"] == 0
    assert bridge.records[-1]["error"]["type"] == "UnsupportedRequest"


@pytest.mark.parametrize(
    "change",
    [
        lambda b: b.update(extra="unreviewed instruction"),
        lambda b: b.update(model=""),
        lambda b: b.update(state=[]),
        lambda b: b.update(questions={}),
        lambda b: b["questions"]["operation"].update(extra="unreviewed instruction"),
        lambda b: b["questions"]["operation"].update(instructions=""),
        lambda b: b["questions"]["operation"].update(instructions=None),
        lambda b: b["questions"]["operation"].update(criteria={}),
        lambda b: b["questions"]["operation"].update(criteria={" ": "Blank"}),
        lambda b: b["questions"]["operation"].update(criteria={"1": 1}),
        lambda b: b["state"].update(nonfinite=math.nan),
    ],
)
def test_malformed_request_is_recorded_before_any_backend_call(change):
    bridge = UltrafastDecisionBridge(RecordingBackend())
    body = payload()
    change(body)
    with pytest.raises(RequestValidationError):
        bridge(TYPESAFE_DECISION_URL, None, body)
    assert bridge.backend_calls == 0
    assert len(bridge.records) == 1
    assert bridge.records[0]["error"] is not None
    json.dumps(bridge.records, allow_nan=False)


def test_nonchoice_questions_and_text_transport_are_not_silently_adapted():
    bridge = UltrafastDecisionBridge(RecordingBackend())
    with pytest.raises(UnsupportedRequest, match="decision URL"):
        bridge("https://example.com/chat/completions", None, payload())
    body = payload()
    body["questions"]["operation"]["type"] = "noul"
    with pytest.raises(UnsupportedRequest, match="choice heads"):
        bridge(TYPESAFE_DECISION_URL, None, body)
    assert bridge.backend_calls == 0


@pytest.mark.parametrize(
    "answer",
    [
        {"choice": "DONE", "probabilities": {"TYPE_TEXT": 0.9, "DONE": 0.1}},
        {"choice": "unknown", "probabilities": {"TYPE_TEXT": 0.5, "DONE": 0.5}},
        {"choice": "DONE", "probabilities": {"DONE": 1.0}},
        {"choice": "DONE", "probabilities": {"TYPE_TEXT": False, "DONE": 1.0}},
        {"choice": "DONE", "probabilities": {"TYPE_TEXT": math.nan, "DONE": 1.0}},
    ],
)
def test_invalid_backend_probabilities_or_choice_cannot_become_browser_actions(answer):
    class InvalidBackend:
        def predict(self, request):
            return {"answers": {"operation": answer}}

    bridge = UltrafastDecisionBridge(InvalidBackend())
    with pytest.raises(BackendResponseError):
        bridge(TYPESAFE_DECISION_URL, None, payload())
    record = bridge.records[0]
    assert record["backend_calls"] == bridge.backend_calls == 1
    assert record["raw_response"] is not None
    assert record["normalized_response"] is None
    assert record["error"]["type"] == "BackendResponseError"
    json.dumps(record, allow_nan=False)


def test_backend_failure_does_not_invent_a_response_or_hide_a_model_call():
    class BrokenBackend:
        def predict(self, request):
            raise RuntimeError("inference failed")

    bridge = UltrafastDecisionBridge(BrokenBackend())
    with pytest.raises(RuntimeError, match="inference failed"):
        bridge(TYPESAFE_DECISION_URL, None, payload())
    assert bridge.backend_calls == 1
    assert bridge.records[0]["raw_response"] is None
    assert bridge.records[0]["error"] == {"type": "RuntimeError", "message": "inference failed"}
