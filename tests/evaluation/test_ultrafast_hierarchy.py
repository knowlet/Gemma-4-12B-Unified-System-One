"""The opt-in structural hierarchy scores every candidate without changing the core contract."""

import copy
import json
import math

import pytest

import s1.evaluation.ultrafast as bridge_module
from s1.backends import UniformBackend, UnsupportedRequest
from s1.contracts import MAX_OPTIONS, DecisionRequest
from s1.errors import BackendResponseError, RequestValidationError
from s1.evaluation.ultrafast import TYPESAFE_DECISION_URL, UltrafastDecisionBridge


def payload(count=57):
    return {
        "model": "original model",
        "state": {"page": {"text": "complete observed state"}, "gold": "unconsumed raw data"},
        "questions": {
            "operation": {
                "type": "choice",
                "instructions": {"rule": "original operation"},
                "criteria": {"CLICK": "Click", "DONE": "Done"},
            },
            "click_target": {
                "type": "choice",
                "instructions": {
                    "goal": "original task",
                    "rules": ["all original rules"],
                    "nested": {"keep": True},
                },
                "criteria": {
                    f"id-{i}": {"element": f"[{i}] Original", "nested": [i, False]}
                    for i in range(count)
                },
            },
            "type_text_target": {
                "type": "choice",
                "instructions": "original singleton",
                "criteria": {"field": {"role": "textbox"}},
            },
        },
    }


class RecordingBackend(UniformBackend):
    def __init__(self):
        self.requests, self.responses = [], []

    def predict(self, request):
        self.requests.append(copy.deepcopy(request))
        response = super().predict(request)
        self.responses.append(copy.deepcopy(response))
        return response


def execute(body, backend=None):
    backend = backend or RecordingBackend()
    bridge = UltrafastDecisionBridge(backend, oversized_head_policy="hierarchical")
    result = bridge(TYPESAFE_DECISION_URL, "never-log-this", body)
    return bridge, backend, result


@pytest.mark.parametrize(
    "count,sizes", [(53, [27, 26]), (57, [29, 28]), (104, [52, 52]), (105, [35, 35, 35])]
)
def test_joint_request_preserves_all_candidates_state_and_instructions(count, sizes):
    body = payload(count)
    original = copy.deepcopy(body)
    bridge, backend, result = execute(body)
    request = backend.requests[0]
    assert len(backend.requests) == bridge.backend_calls == 1
    assert len(request.questions) == 2 + len(sizes)
    assert request.state == body["state"] and request.state is not body["state"]
    parent = request.questions[1]
    original_instruction = body["questions"]["click_target"]["instructions"]
    assert json.loads(parent.instructions)["original_instructions"] == original_instruction
    members = {}
    for index, conditional in enumerate(request.questions[2:]):
        parent_members = json.loads(parent.descriptions()[index])["members"]
        members.update(parent_members)
        assert conditional.labels() == list(parent_members)
        assert len(parent_members) == sizes[index]
        assert {
            key: json.loads(value) for key, value in conditional.criteria.items()
        } == parent_members
        assert json.loads(conditional.instructions)["original_instructions"] == original_instruction
    assert members == body["questions"]["click_target"]["criteria"]
    assert list(members) == list(body["questions"]["click_target"]["criteria"])
    assert all(2 <= len(q.labels()) <= MAX_OPTIONS for q in request.questions)
    assert list(result["answers"]) == list(body["questions"])
    answer = result["answers"]["click_target"]
    assert list(answer["probabilities"]) == list(members)
    assert all(value > 0 for value in answer["probabilities"].values())
    assert math.fsum(answer["probabilities"].values()) == pytest.approx(1)
    assert answer["choice"] == max(answer["probabilities"], key=answer["probabilities"].get)
    assert answer["probability_semantics"] == "uncalibrated_hierarchical_approximation"
    assert result["answers"]["type_text_target"]["probabilities"] == {"field": 1.0}
    assert body == original and MAX_OPTIONS == 52
    record = bridge.records[0]
    assert record["body"] == original
    assert record["mapped_request"] == request.model_dump()
    assert record["raw_response"] == backend.responses[0]
    assert record["reported_cost_usd"] is result["usage"] is None
    assert record["hierarchy"]["internal_head_count"] == len(request.questions)
    calls = record["hierarchy"]["subcalls"]
    assert len(calls) == 1 and calls[0]["kind"] == "joint"
    assert calls[0]["backend_calls"] == 1 and calls[0]["latency_ms"] >= 0
    assert calls[0]["raw_response"] == backend.responses[0]
    assert calls[0]["reported_cost_usd"] is None and calls[0]["error"] is None
    assert "never-log-this" not in json.dumps(record)


class ConsistentWeightedBackend(RecordingBackend):
    def predict(self, request):
        self.requests.append(copy.deepcopy(request))
        answers = {}
        for question in request.questions:
            instructions = json.loads(question.instructions)
            stage = instructions.get("hierarchy", {}).get("stage")
            if stage == "parent":
                weights = [
                    sum(int(label.split("-")[1]) + 1 for label in json.loads(value)["members"])
                    for value in question.descriptions()
                ]
            elif stage == "conditional":
                weights = [int(label.split("-")[1]) + 1 for label in question.labels()]
            else:
                weights = [1] * len(question.labels())
            answers[question.id] = {
                "probabilities": {
                    label: value / sum(weights) for label, value in zip(question.labels(), weights)
                }
            }
        response = {"answers": answers}
        self.responses.append(copy.deepcopy(response))
        return response


def test_candidate_permutation_preserves_exact_label_description_identity_and_derived_mass():
    body = payload()
    first_bridge, _, first = execute(body, ConsistentWeightedBackend())
    body["questions"]["click_target"]["criteria"] = dict(
        reversed(list(body["questions"]["click_target"]["criteria"].items()))
    )
    second_bridge, _, second = execute(body, ConsistentWeightedBackend())
    a, b = first["answers"]["click_target"], second["answers"]["click_target"]
    assert list(b["probabilities"]) == list(reversed(a["probabilities"]))
    assert b["probabilities"] == pytest.approx(a["probabilities"])
    assert a["choice"] == b["choice"] == "id-56"
    assert first_bridge.records[0]["hierarchy"]["heads"][0]["groups"][0]["candidate_ids"] == [
        f"id-{i}" for i in range(29)
    ]
    assert second_bridge.records[0]["hierarchy"]["heads"][0]["groups"][0]["candidate_ids"] == [
        f"id-{i}" for i in range(56, 27, -1)
    ]


def test_global_argmax_can_come_from_parent_nonwinning_group_without_pruning():
    class Backend(RecordingBackend):
        def predict(self, request):
            response = super().predict(request)
            parent, first, second = request.questions[1:]
            response["answers"][parent.id] = {"probabilities": {"group_1": 0.6, "group_2": 0.4}}
            weights = [0.6] + [0.4 / 27] * 27
            response["answers"][second.id] = {"probabilities": dict(zip(second.labels(), weights))}
            return response

    _, _, result = execute(payload(), Backend())
    answer = result["answers"]["click_target"]
    assert answer["choice"] == "id-29"
    assert answer["probabilities"]["id-29"] == pytest.approx(0.24)
    assert all(value > 0 for value in answer["probabilities"].values())


def test_generated_head_ids_avoid_original_id_collisions_and_multiple_large_heads_remain_distinct():
    body = payload()
    body["questions"]["__s1_hierarchy_1"] = body["questions"].pop("operation")
    body["questions"]["other_target"] = copy.deepcopy(body["questions"]["click_target"])
    bridge, backend, result = execute(body)
    ids = [q.id for q in backend.requests[0].questions]
    assert len(ids) == len(set(ids)) == 7
    assert "__s1_hierarchy_1" in ids and "__s1_hierarchy_2" in ids
    assert list(result["answers"]) == list(body["questions"])
    assert len(bridge.records[0]["hierarchy"]["heads"]) == 2
    assert bridge.backend_calls == 1


def test_maximum_52_groups_score_every_original_candidate_in_one_core_call():
    body = payload(52 * 52)
    bridge, backend, result = execute(body)
    assert len(backend.requests) == 1
    assert len(backend.requests[0].questions) == 54
    assert len(bridge.records[0]["hierarchy"]["heads"][0]["groups"]) == 52
    probabilities = result["answers"]["click_target"]["probabilities"]
    assert len(probabilities) == 2704 and all(p > 0 for p in probabilities.values())
    assert math.fsum(probabilities.values()) == pytest.approx(1)


def test_opt_in_singleton_only_request_still_has_zero_backend_calls():
    body = payload(1)
    body["questions"] = {"type_text_target": body["questions"]["type_text_target"]}
    bridge, backend, result = execute(body)
    assert not backend.requests and bridge.backend_calls == 0
    assert result["answers"]["type_text_target"]["choice"] == "field"
    assert "hierarchy" not in bridge.records[0]


def test_partition_and_aggregation_do_not_inspect_gold_like_raw_state():
    body = payload()
    _, _, first = execute(body)
    body["state"]["gold"] = {"click_target": "id-56", "label": "id-0"}
    bridge, _, second = execute(body)
    assert second == first
    assert bridge.records[0]["mapped_request"]["state"] == body["state"]


@pytest.mark.parametrize("count", [1, 2, 52])
def test_opt_in_flat_path_is_exactly_the_existing_reject_path(count, monkeypatch):
    monkeypatch.setattr(bridge_module.time, "perf_counter", lambda: 1)
    body = payload(count)
    baseline = UltrafastDecisionBridge(RecordingBackend())
    hierarchical = UltrafastDecisionBridge(RecordingBackend(), oversized_head_policy="hierarchical")
    assert baseline(TYPESAFE_DECISION_URL, None, body) == hierarchical(
        TYPESAFE_DECISION_URL, None, body
    )
    assert baseline.records == hierarchical.records
    assert [r.model_dump() for r in baseline.backend.requests] == [
        r.model_dump() for r in hierarchical.backend.requests
    ]
    assert "hierarchy" not in hierarchical.records[0]


@pytest.mark.parametrize("policy", ["reject", "unknown"])
def test_default_rejection_or_unknown_policy_never_silently_adapts(policy):
    backend = RecordingBackend()
    if policy == "unknown":
        with pytest.raises(ValueError, match="policy"):
            UltrafastDecisionBridge(backend, oversized_head_policy=policy)
    else:
        bridge = UltrafastDecisionBridge(backend)
        with pytest.raises(UnsupportedRequest, match="nothing was truncated"):
            bridge(TYPESAFE_DECISION_URL, None, payload())
    assert not backend.requests


@pytest.mark.parametrize(
    "change",
    [
        lambda b: b.update(extra="invalid"),
        lambda b: b["questions"]["click_target"].update(extra="invalid"),
        lambda b: b["questions"]["click_target"].update(type="noul"),
        lambda b: b["questions"]["click_target"].update(instructions=""),
        lambda b: b["questions"]["click_target"].update(instructions="x" * 16000),
        lambda b: b["questions"]["click_target"]["criteria"].update({" ": "invalid label"}),
        lambda b: b["questions"]["click_target"]["criteria"].update({"last": 1}),
        lambda b: b["questions"]["operation"].update(criteria={"invalid": 1}),
    ],
)
def test_all_original_heads_are_validated_before_the_joint_dispatch(change):
    body, backend = payload(), RecordingBackend()
    change(body)
    bridge = UltrafastDecisionBridge(backend, oversized_head_policy="hierarchical")
    with pytest.raises((RequestValidationError, UnsupportedRequest)):
        bridge(TYPESAFE_DECISION_URL, None, body)
    assert not backend.requests and bridge.backend_calls == 0
    assert bridge.records[0]["normalized_response"] is None


def test_more_than_52_groups_or_64_joint_heads_fails_whole_without_dispatch():
    for body in [payload(52 * 52 + 1), payload()]:
        if len(body["questions"]["click_target"]["criteria"]) == 57:
            body["questions"].update(
                {f"ordinary-{i}": copy.deepcopy(body["questions"]["operation"]) for i in range(61)}
            )
        backend = RecordingBackend()
        bridge = UltrafastDecisionBridge(backend, oversized_head_policy="hierarchical")
        with pytest.raises((UnsupportedRequest, RequestValidationError)):
            bridge(TYPESAFE_DECISION_URL, None, body)
        assert not backend.requests and bridge.backend_calls == 0


@pytest.mark.parametrize("failure", ["unknown", "missing", "nonfinite", "context"])
def test_malformed_internal_head_or_context_failure_cannot_produce_partial_browser_answer(failure):
    class Backend(RecordingBackend):
        def predict(self, request):
            if failure == "context":
                self.requests.append(copy.deepcopy(request))
                raise RequestValidationError(
                    "request exceeds native context; nothing was truncated"
                )
            response = super().predict(request)
            last = request.questions[-1]
            if failure == "unknown":
                response["answers"][last.id]["choice"] = "unobserved"
            elif failure == "missing":
                response["answers"].pop(last.id)
            else:
                response["answers"][last.id]["probabilities"][last.labels()[0]] = math.nan
            return response

    backend = Backend()
    bridge = UltrafastDecisionBridge(backend, oversized_head_policy="hierarchical")
    with pytest.raises((BackendResponseError, RequestValidationError)):
        bridge(TYPESAFE_DECISION_URL, None, payload())
    record = bridge.records[0]
    assert bridge.backend_calls == len(backend.requests) == record["backend_calls"] == 1
    assert record["normalized_response"] is None
    assert record["hierarchy"]["subcalls"][0]["error"] is not None
    json.dumps(record, allow_nan=False)


def test_hierarchy_does_not_expand_the_core_question_candidate_limit():
    with pytest.raises(ValueError, match="2..52"):
        DecisionRequest.model_validate(
            {
                "state": {},
                "questions": {
                    "q": {
                        "type": "choice",
                        "instructions": "original",
                        "criteria": {str(i): str(i) for i in range(57)},
                    }
                },
            }
        )


@pytest.mark.parametrize("fails", [False, True])
def test_observed_backbone_forwards_are_counted_instead_of_inferred_from_head_count(fails):
    forwards = []

    class Backend(RecordingBackend):
        def predict(self, request):
            forwards.extend(["observed forward", "observed forward"])
            if fails:
                raise RuntimeError("partial inference failure")
            return super().predict(request)

    bridge = UltrafastDecisionBridge(
        Backend(),
        oversized_head_policy="hierarchical",
        backbone_forward_count=lambda: len(forwards),
    )
    if fails:
        with pytest.raises(RuntimeError, match="partial inference"):
            bridge(TYPESAFE_DECISION_URL, None, payload())
    else:
        bridge(TYPESAFE_DECISION_URL, None, payload())
    record = bridge.records[0]
    assert record["hierarchy"]["internal_head_count"] == 4
    assert record["hierarchy"]["backbone_forwards"] == (None if fails else 2)
    call = record["hierarchy"]["subcalls"][0]
    assert call["backbone_forwards"] == (None if fails else 2)
    assert call["observed_completed_forward_delta"] == 2
    assert call["forward_count_basis"] == "successful_backbone_returns"
    assert call["forward_count_interpretation"] == (
        "lower_bound" if fails else "exact_successful_forwards"
    )
    assert record["hierarchy"]["observed_completed_forward_delta"] == 2
    assert record["backend_calls"] == 1


def test_failure_before_a_forward_return_is_unknown_actual_count_with_zero_lower_bound():
    original = RuntimeError("backbone dispatched but no successful return")

    class Backend:
        def predict(self, request):
            raise original

    bridge = UltrafastDecisionBridge(
        Backend(), oversized_head_policy="hierarchical", backbone_forward_count=lambda: 0
    )
    with pytest.raises(RuntimeError) as actual:
        bridge(TYPESAFE_DECISION_URL, None, payload())
    assert actual.value is original
    trace = bridge.records[0]["hierarchy"]
    assert trace["backbone_forwards"] is None
    assert trace["observed_completed_forward_delta"] == 0
    assert trace["forward_count_basis"] == "successful_backbone_returns"
    assert trace["forward_count_interpretation"] == "lower_bound"


@pytest.mark.parametrize("fail_on", [1, 2])
@pytest.mark.parametrize("backend_fails", [False, True])
def test_counter_observer_failure_is_unknown_and_cannot_replace_backend_result_or_error(
    fail_on, backend_fails
):
    counter_calls = []
    original = RuntimeError("original model failure")

    def counter():
        counter_calls.append(1)
        if len(counter_calls) == fail_on:
            raise OSError("counter observation failed")
        return 0

    class Backend(RecordingBackend):
        def predict(self, request):
            if backend_fails:
                raise original
            return super().predict(request)

    bridge = UltrafastDecisionBridge(
        Backend(), oversized_head_policy="hierarchical", backbone_forward_count=counter
    )
    if backend_fails:
        with pytest.raises(RuntimeError) as actual:
            bridge(TYPESAFE_DECISION_URL, None, payload())
        assert actual.value is original
    else:
        result = bridge(TYPESAFE_DECISION_URL, None, payload())
        assert len(result["answers"]["click_target"]["probabilities"]) == 57
    trace = bridge.records[0]["hierarchy"]
    assert trace["backbone_forwards"] is trace["observed_completed_forward_delta"] is None
    assert trace["forward_count_interpretation"] is None
    assert trace["forward_observation_error"] == {
        "type": "OSError",
        "message": "counter observation failed",
    }
