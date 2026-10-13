"""Offline contract tests for the opt-in hierarchical browser decision policy."""

import copy
import json
from types import SimpleNamespace

import pytest

import s1.jev_hierarchy as hierarchy_module
from s1.contracts import DecisionRequest, answer_from_probabilities
from s1.errors import BackendResponseError, BackendTimeoutError, RequestValidationError
from s1.jev import JevCompatibilityAdapter
from s1.jev_hierarchy import JevHierarchicalAdapter

SEMANTICS = "hierarchical_policy_distribution_not_flat_calibration"


def browser_payload(count=53, head="click_target"):
    return {
        "model": "browser-alias",
        "state": {
            "page": {
                "url": "http://fixture.invalid/controls",
                "title": "Observed controls",
                "text": "Choose an observed control",
            },
            "recent_actions": [{"operation": "WAIT"}],
        },
        "questions": {
            "operation": {
                "type": "choice",
                "instructions": {"goal": "Choose an observed control", "rules": ["Use IDs"]},
                "criteria": {"CLICK": "Click an observed control", "DONE": "Task complete"},
            },
            head: {
                "type": "choice",
                "instructions": {
                    "goal": "Choose an observed control",
                    "operation": head.removesuffix("_target").upper(),
                    "rules": ["Preserve every visible target", {"nested": [True, None, "中文"]}],
                },
                "criteria": {
                    f"node-{i:03d}": {
                        "element": f"[{i}] Control 中文 {i}",
                        "current_value": f"value-{i}",
                        "role": "button",
                        "checked": False,
                        "nested": {"index": i, "optional": None},
                    }
                    for i in range(1, count + 1)
                },
            },
        },
    }


def stage_info(question):
    try:
        instructions = json.loads(question.instructions)
    except json.JSONDecodeError:
        return {}
    return instructions.get("hierarchy", {}) if isinstance(instructions, dict) else {}


class RecordingBackend:
    """Return actual, inspectable stage distributions without loading a model."""

    name = "offline-policy"

    def __init__(self, probabilities=None, *, model=None):
        self.calls = []
        self.responses = []
        self.probabilities = probabilities
        self.model = self.name if model is None else model

    def predict(self, request):
        self.calls.append(copy.deepcopy(request))
        answers = {}
        for question in request.questions:
            values = (
                self.probabilities(question)
                if self.probabilities is not None
                else [1 / len(question.labels())] * len(question.labels())
            )
            answers[question.id] = answer_from_probabilities(question, values)
        response = {
            "model": self.model,
            "answers": answers,
            "usage": {"input_tokens": 11, "output_tokens": 2},
            "provider_metadata": {"backend_call": len(self.calls)},
        }
        self.responses.append(copy.deepcopy(response))
        return response


def adapter(backend, **kwargs):
    return JevHierarchicalAdapter(backend, model_identity="browser-alias", **kwargs)


def hierarchy_questions(backend, stage):
    return [
        question
        for request in backend.calls
        for question in request.questions
        if stage_info(question).get("stage") == stage
    ]


@pytest.mark.parametrize("head", ["click_target", "type_text_target", "select_target"])
@pytest.mark.parametrize("count", [53, 74, 80])
def test_all_original_targets_have_real_group_and_conditional_probabilities(head, count):
    payload = browser_payload(count, head)
    original = copy.deepcopy(payload)
    backend = RecordingBackend()
    result = adapter(backend).predict(payload)

    assert payload == original
    labels = list(original["questions"][head]["criteria"])
    answer = result["answers"][head]
    assert list(answer["probabilities"]) == labels
    assert sum(answer["probabilities"].values()) == pytest.approx(1)
    assert answer["choice"] in labels
    assert answer["confidence"] == pytest.approx(max(answer["probabilities"].values()))
    ordered_probabilities = sorted(answer["probabilities"].values(), reverse=True)
    assert answer["margin"] == pytest.approx(ordered_probabilities[0] - ordered_probabilities[1])

    roots = hierarchy_questions(backend, "group")
    leaves = hierarchy_questions(backend, "conditional")
    assert len(roots) == 1
    root = roots[0]
    group_count = (count + 19) // 20
    assert root.labels() == [f"group{i}" for i in range(group_count)]
    assert len(leaves) == group_count
    memberships = {
        group_id: json.loads(description)["members"]
        for group_id, description in root.criteria.items()
    }
    assert [member["target_id"] for members in memberships.values() for member in members] == labels
    sizes = [len(members) for members in memberships.values()]
    assert max(sizes) <= 20
    assert max(sizes) - min(sizes) <= 1
    for members in memberships.values():
        for member in members:
            assert (
                member["criteria"] == original["questions"][head]["criteria"][member["target_id"]]
            )

    for leaf in leaves:
        context = json.loads(leaf.instructions)
        info = context["hierarchy"]
        expected_members = [member["target_id"] for member in memberships[info["group"]]]
        assert leaf.id == head
        assert leaf.labels() == expected_members == info["members"]
        assert context["original_instructions"] == original["questions"][head]["instructions"]
        assert info["head"] == head
        for label in leaf.labels():
            assert (
                json.loads(leaf.criteria[label]) == original["questions"][head]["criteria"][label]
            )
            assert answer["probabilities"][label] == pytest.approx(
                (1 / group_count) * (1 / len(leaf.labels()))
            )
    root_context = json.loads(root.instructions)
    assert root_context["original_instructions"] == original["questions"][head]["instructions"]
    assert root_context["hierarchy"]["head"] == head
    assert all(isinstance(request, DecisionRequest) for request in backend.calls)
    assert all(request.state == original["state"] for request in backend.calls)
    assert len(backend.calls) == 2 + group_count
    assert [question.id for question in backend.calls[0].questions] == ["operation"]
    assert result["model"] == backend.model
    assert result["usage"] == {
        "input_tokens": 11 * len(backend.calls),
        "output_tokens": 2 * len(backend.calls),
    }

    trace = result["hierarchy"]
    assert trace["version"] == "jev_hierarchy_v1"
    assert trace["status"] == "completed"
    assert trace["partition"] == "balanced_observation_order"
    assert trace["branch_width"] == 20
    assert trace["probabilities_semantics"] == SEMANTICS
    assert trace["backend_predict_calls"] == len(backend.calls)
    assert len(trace["original_request_sha256"]) == 64
    assert trace["elapsed_ms"] >= 0
    head_trace = trace["heads"][head]
    assert head_trace["original_labels"] == labels
    assert head_trace["groups"] == [
        {"id": group_id, "members": [member["target_id"] for member in members]}
        for group_id, members in memberships.items()
    ]
    assert head_trace["joint_probabilities"] == answer["probabilities"]
    assert head_trace["group_probabilities"] == {
        group_id: pytest.approx(1 / group_count) for group_id in memberships
    }
    for stage, request, response in zip(
        trace["stages"], backend.calls, backend.responses, strict=True
    ):
        assert stage["request"] == request.model_dump(exclude_none=True)
        assert stage["response"]["model"] == response["model"]
        assert stage["response"]["provider_metadata"] == response["provider_metadata"]
        assert stage["status"] == "completed"
        assert stage["elapsed_ms"] >= 0


def test_global_winner_can_be_outside_the_highest_probability_group():
    def probabilities(question):
        info = stage_info(question)
        if info.get("stage") == "group":
            return [0.5, 0.3, 0.2]
        if info.get("stage") == "conditional" and info["group"] == "group1":
            return [0.03 / (len(question.labels()) - 1)] * (len(question.labels()) - 1) + [0.97]
        return [1 / len(question.labels())] * len(question.labels())

    backend = RecordingBackend(probabilities)
    result = adapter(backend).predict(browser_payload())
    leaves = hierarchy_questions(backend, "conditional")
    winning_leaf = next(leaf for leaf in leaves if stage_info(leaf)["group"] == "group1")
    winner = winning_leaf.labels()[-1]
    assert result["answers"]["click_target"]["choice"] == winner
    assert result["answers"]["click_target"]["probabilities"][winner] == pytest.approx(0.3 * 0.97)
    trace = result["hierarchy"]["heads"]["click_target"]
    assert max(trace["group_probabilities"], key=trace["group_probabilities"].get) == "group0"
    assert winner in trace["groups"][1]["members"]
    assert len(leaves) == 3


def test_even_zero_probability_groups_are_forwarded_and_have_real_leaf_evidence():
    def probabilities(question):
        if stage_info(question).get("stage") == "group":
            return [1.0, 0.0, 0.0]
        return [1 / len(question.labels())] * len(question.labels())

    backend = RecordingBackend(probabilities)
    result = adapter(backend).predict(browser_payload())
    leaves = hierarchy_questions(backend, "conditional")
    assert [stage_info(leaf)["group"] for leaf in leaves] == ["group0", "group1", "group2"]
    for leaf in leaves[1:]:
        for label in leaf.labels():
            assert result["answers"]["click_target"]["probabilities"][label] == 0
    assert (
        len([stage for stage in result["hierarchy"]["stages"] if stage["scope"] == "conditional"])
        == 3
    )


@pytest.mark.parametrize("count", [2, 20])
def test_small_requests_preserve_the_existing_adapter_result_and_exact_backend_request(count):
    payload = browser_payload(count)
    payload["questions"]["select_target"] = {
        "type": "choice",
        "instructions": {"goal": "Select existing option"},
        "criteria": {"node-900": {"element": "Only select", "role": "combobox"}},
    }
    hierarchy_backend = RecordingBackend()
    ordinary_backend = RecordingBackend()
    expected = JevCompatibilityAdapter(ordinary_backend, model_identity="browser-alias").predict(
        payload
    )
    actual = adapter(hierarchy_backend).predict(payload)
    assert actual == expected
    assert hierarchy_backend.calls == ordinary_backend.calls
    assert "hierarchy" not in actual
    assert actual["answers"]["select_target"]["probabilities"] == {"node-900": 1.0}
    assert [q.id for q in hierarchy_backend.calls[0].questions] == ["operation", "click_target"]


def test_all_singleton_targets_need_no_backend_calls_and_keep_existing_contract():
    payload = browser_payload(1)
    del payload["questions"]["operation"]
    backend = RecordingBackend()
    expected = JevCompatibilityAdapter(RecordingBackend(), model_identity="browser-alias").predict(
        payload
    )
    assert adapter(backend).predict(payload) == expected
    assert not backend.calls


def test_hierarchical_request_keeps_original_singletons_out_of_backend_calls():
    payload = browser_payload()
    payload["questions"]["select_target"] = {
        "type": "choice",
        "instructions": "Pick the observed select",
        "criteria": {"select-original-id": "Only select"},
    }
    backend = RecordingBackend()
    result = adapter(backend).predict(payload)
    assert result["answers"]["select_target"]["choice"] == "select-original-id"
    assert result["answers"]["select_target"]["probabilities"] == {"select-original-id": 1.0}
    assert result["hierarchy"]["deterministic_heads"] == ["select_target"]
    assert all(q.id != "select_target" for request in backend.calls for q in request.questions)
    assert list(result["answers"]) == list(payload["questions"])


def test_twenty_one_targets_crosses_the_common_branching_boundary_without_pruning():
    backend = RecordingBackend()
    result = adapter(backend).predict(browser_payload(21))
    leaves = hierarchy_questions(backend, "conditional")
    assert [len(leaf.labels()) for leaf in leaves] == [11, 10]
    assert len(result["answers"]["click_target"]["probabilities"]) == 21
    assert len(backend.calls) == 4


def test_partition_depends_on_observation_order_and_not_goal_or_backend_distribution():
    first = browser_payload(74)
    second = copy.deepcopy(first)
    second["questions"]["click_target"]["instructions"]["goal"] = "A completely different goal"
    first_backend = RecordingBackend(model="first-model")

    def concentrate_on_last(question):
        return [0.0] * (len(question.labels()) - 1) + [1.0]

    second_backend = RecordingBackend(concentrate_on_last, model="second-model")
    first_result = adapter(first_backend).predict(first)
    second_result = adapter(second_backend).predict(second)
    assert (
        first_result["hierarchy"]["heads"]["click_target"]["groups"]
        == second_result["hierarchy"]["heads"]["click_target"]["groups"]
    )
    assert (
        hierarchy_questions(first_backend, "group")[0].criteria
        == hierarchy_questions(second_backend, "group")[0].criteria
    )
    assert (
        first_result["hierarchy"]["original_request_sha256"]
        != second_result["hierarchy"]["original_request_sha256"]
    )


def test_leaf_serialization_is_deterministic_and_retains_structured_original_descriptions():
    payload = browser_payload()
    reordered = copy.deepcopy(payload)
    question = reordered["questions"]["click_target"]
    question["instructions"] = dict(reversed(list(question["instructions"].items())))
    question["criteria"] = {
        label: dict(reversed(list(description.items())))
        for label, description in question["criteria"].items()
    }
    first_backend, second_backend = RecordingBackend(), RecordingBackend()
    adapter(first_backend).predict(payload)
    adapter(second_backend).predict(reordered)
    for first, second in zip(first_backend.calls, second_backend.calls, strict=True):
        assert first == second


@pytest.mark.parametrize(
    ("instructions", "expected_goal"),
    [
        ({"goal": "  Pick 中文 exactly!  ", "rules": ["Keep details"]}, "  Pick 中文 exactly!  "),
        ('{"goal":"  Pick 中文 exactly!  ","rules":["Keep details"]}', "  Pick 中文 exactly!  "),
        ("Pick a visible control", None),
        ('{"goal": broken JSON', None),
        ({"rules": ["No supplied goal"]}, None),
        ('{"rules":["No supplied goal"]}', None),
        ({"goal": False, "rules": ["No string goal"]}, None),
        ('{"goal":7,"rules":["No string goal"]}', None),
    ],
    ids=[
        "object-goal",
        "json-string-goal",
        "plaintext",
        "invalid-json",
        "object-missing-goal",
        "json-string-missing-goal",
        "object-nonstring-goal",
        "json-string-nonstring-goal",
    ],
)
def test_hierarchy_exposes_only_an_exact_supplied_goal_and_keeps_full_original_instructions(
    instructions, expected_goal
):
    payload = browser_payload()
    payload["questions"]["click_target"]["instructions"] = copy.deepcopy(instructions)
    original = copy.deepcopy(payload)
    backend = RecordingBackend()
    adapter(backend).predict(payload)
    questions = hierarchy_questions(backend, "group") + hierarchy_questions(backend, "conditional")
    assert len(questions) == 4
    for question in questions:
        body = json.loads(question.instructions)
        assert body["original_instructions"] == instructions
        if expected_goal is None:
            assert "goal" not in body
        else:
            assert body["goal"] == expected_goal
    assert payload == original


def test_original_request_hash_changes_with_candidate_observation_order():
    payload = browser_payload()
    reordered = copy.deepcopy(payload)
    criteria = reordered["questions"]["click_target"]["criteria"]
    reordered["questions"]["click_target"]["criteria"] = dict(reversed(list(criteria.items())))
    assert payload == reordered  # Equal mappings still have different observed order.
    first_result = adapter(RecordingBackend()).predict(payload)
    second_result = adapter(RecordingBackend()).predict(reordered)
    assert (
        first_result["hierarchy"]["original_request_sha256"]
        != second_result["hierarchy"]["original_request_sha256"]
    )
    first_members = first_result["hierarchy"]["heads"]["click_target"]["original_labels"]
    second_members = second_result["hierarchy"]["heads"]["click_target"]["original_labels"]
    assert first_members == list(reversed(second_members))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["questions"]["click_target"]["criteria"].update(
            {" ": "Invalid last observed label"}
        ),
        lambda payload: payload["questions"]["click_target"]["criteria"].update({"node-053": 7}),
        lambda payload: payload["questions"]["click_target"]["criteria"]["node-053"].update(
            checked=float("nan")
        ),
        lambda payload: payload["questions"]["click_target"].update(extra="hidden extra field"),
        lambda payload: payload["questions"]["click_target"].update(id="different-id"),
    ],
    ids=[
        "blank-last-label",
        "invalid-last-description",
        "nonfinite-last-description",
        "extra",
        "id",
    ],
)
def test_invalid_original_data_is_not_hidden_by_partitioning(mutation):
    payload = browser_payload()
    mutation(payload)
    backend = RecordingBackend()
    with pytest.raises(RequestValidationError):
        adapter(backend).predict(payload)
    assert not backend.calls


def test_model_alias_mismatch_is_rejected_before_any_hierarchical_inference():
    payload = browser_payload()
    payload["model"] = "another-model"
    backend = RecordingBackend()
    with pytest.raises(RequestValidationError):
        adapter(backend).predict(payload)
    assert not backend.calls


@pytest.mark.parametrize(
    "kwargs",
    [{"max_targets": 40}, {"max_stage_calls": 4}, {"max_payload_bytes": 1024}],
    ids=["candidate-bound", "work-bound", "serialized-payload-bound"],
)
def test_resource_bounds_are_checked_before_predict(kwargs):
    backend = RecordingBackend()
    with pytest.raises(RequestValidationError):
        adapter(backend, **kwargs).predict(browser_payload())
    assert not backend.calls


def test_four_hundred_and_one_targets_are_rejected_without_silently_truncating():
    backend = RecordingBackend()
    with pytest.raises(RequestValidationError):
        adapter(backend).predict(browser_payload(401))
    assert not backend.calls


def test_three_full_capacity_target_heads_use_exactly_the_bounded_sixty_four_calls():
    payload = browser_payload(400)
    for head in ("type_text_target", "select_target"):
        payload["questions"][head] = copy.deepcopy(payload["questions"]["click_target"])
        payload["questions"][head]["instructions"]["operation"] = head
    backend = RecordingBackend()
    result = adapter(backend).predict(payload)
    assert len(backend.calls) == result["hierarchy"]["backend_predict_calls"] == 64
    assert result["hierarchy"]["planned_backend_calls"] == 64
    for head in ("click_target", "type_text_target", "select_target"):
        assert len(result["hierarchy"]["heads"][head]["groups"]) == 20
        assert list(result["answers"][head]["probabilities"]) == list(
            payload["questions"][head]["criteria"]
        )
        assert all(
            probability == pytest.approx(1 / 400)
            for probability in result["answers"][head]["probabilities"].values()
        )
    rejected_backend = RecordingBackend()
    with pytest.raises(RequestValidationError):
        adapter(rejected_backend, max_stage_calls=63).predict(payload)
    assert not rejected_backend.calls


@pytest.mark.parametrize("field", ["label", "description"])
def test_expanded_head_lengths_are_bounded_even_for_the_last_original_candidate(field):
    payload = browser_payload()
    criteria = payload["questions"]["click_target"]["criteria"]
    if field == "label":
        criteria["x" * 129] = criteria.pop("node-053")
    else:
        criteria["node-053"] = "x" * 16_001
    backend = RecordingBackend()
    with pytest.raises(RequestValidationError):
        adapter(backend).predict(payload)
    assert not backend.calls


@pytest.mark.parametrize("kind", ["unknown-head", "score"])
def test_only_recognized_browser_choice_target_heads_gain_extended_capacity(kind):
    payload = browser_payload()
    question = payload["questions"]["click_target"]
    if kind == "unknown-head":
        payload["questions"]["unrecognized_target"] = payload["questions"].pop("click_target")
    elif kind == "score":
        question["type"] = "score"
    backend = RecordingBackend()
    with pytest.raises(RequestValidationError):
        adapter(backend).predict(payload)
    assert not backend.calls


def test_supported_list_criteria_keep_every_original_label_and_description():
    payload = browser_payload()
    labels = list(payload["questions"]["click_target"]["criteria"])
    payload["questions"]["click_target"]["criteria"] = labels
    backend = RecordingBackend()
    result = adapter(backend).predict(payload)
    assert list(result["answers"]["click_target"]["probabilities"]) == labels
    for root in hierarchy_questions(backend, "group"):
        for description in root.criteria.values():
            assert all(
                member["criteria"] == member["target_id"]
                for member in json.loads(description)["members"]
            )
    for leaf in hierarchy_questions(backend, "conditional"):
        assert leaf.criteria == {label: label for label in leaf.labels()}


@pytest.mark.parametrize("wide_target", [False, True])
def test_valid_ordinary_heads_above_branch_width_keep_the_core_fifty_two_option_contract(
    wide_target,
):
    payload = browser_payload(53 if wide_target else 20)
    payload["questions"]["operation"]["criteria"] = {
        f"OP-{index}": f"Ordinary choice {index}" for index in range(52)
    }
    backend = RecordingBackend()
    result = adapter(backend).predict(payload)
    ordinary = backend.calls[0].questions[0]
    assert ordinary.id == "operation"
    assert ordinary.criteria == payload["questions"]["operation"]["criteria"]
    assert len(result["answers"]["operation"]["probabilities"]) == 52
    assert not stage_info(ordinary)
    if not wide_target:
        expected_backend = RecordingBackend()
        expected = JevCompatibilityAdapter(
            expected_backend, model_identity="browser-alias"
        ).predict(payload)
        assert result == expected
        assert backend.calls == expected_backend.calls


def test_duplicate_original_question_ids_are_rejected_before_predict():
    payload = browser_payload()
    target = {"id": "click_target", **payload["questions"]["click_target"]}
    payload["questions"] = [target, copy.deepcopy(target)]
    backend = RecordingBackend()
    with pytest.raises(RequestValidationError):
        adapter(backend).predict(payload)
    assert not backend.calls


class LateFailureBackend(RecordingBackend):
    def __init__(self, failure):
        super().__init__()
        self.failure = failure

    def predict(self, request):
        question = request.questions[0]
        info = stage_info(question)
        if info.get("stage") == "conditional" and info["group"] == "group2":
            self.calls.append(copy.deepcopy(request))
            raise self.failure
        return super().predict(request)


@pytest.mark.parametrize("error_type", [RequestValidationError, BackendTimeoutError])
def test_late_typed_failure_preserves_prior_stages_and_original_generation(error_type):
    failure = error_type("late conditional failure")
    failure.generation = {"raw": "original failure evidence", "model": "offline-policy"}
    backend = LateFailureBackend(failure)
    with pytest.raises(error_type) as caught:
        adapter(backend).predict(browser_payload())
    assert caught.value is failure
    assert caught.value.generation["raw"] == "original failure evidence"
    trace = caught.value.hierarchy
    assert trace["status"] == "failed"
    assert trace["backend_predict_calls"] == len(backend.calls) == 5
    assert [stage["status"] for stage in trace["stages"]] == [
        "completed",
        "completed",
        "completed",
        "completed",
        "failed",
    ]
    failed = trace["stages"][-1]
    assert failed["scope"] == "conditional"
    assert failed["group"] == "group2"
    assert failed["request"] == backend.calls[-1].model_dump(exclude_none=True)
    assert failed["error"]
    assert trace["stages"][0]["response"]["answers"]["operation"]
    assert "joint_probabilities" not in trace["heads"]["click_target"]


def test_invalid_late_distribution_preserves_actual_response_and_returns_no_joint_decision():
    class InvalidLastLeaf(RecordingBackend):
        def predict(self, request):
            response = super().predict(request)
            info = stage_info(request.questions[0])
            if info.get("stage") == "conditional" and info["group"] == "group2":
                del response["answers"]["click_target"]["probabilities"]["node-053"]
                self.invalid_response = copy.deepcopy(response)
            return response

    backend = InvalidLastLeaf()
    with pytest.raises(BackendResponseError) as caught:
        adapter(backend).predict(browser_payload())
    trace = caught.value.hierarchy
    assert trace["status"] == "failed"
    failed = trace["stages"][-1]
    assert failed["status"] == "failed"
    assert failed["response"] == backend.invalid_response
    assert failed["request"] == backend.calls[-1].model_dump(exclude_none=True)
    assert "node-053" not in failed["response"]["answers"]["click_target"]["probabilities"]
    assert len(trace["stages"]) == 5
    assert all(stage["status"] == "completed" for stage in trace["stages"][:-1])
    assert "joint_probabilities" not in trace["heads"]["click_target"]


def test_nonfinite_provider_failure_evidence_remains_strictly_json_serializable():
    class NonfiniteLastLeaf(RecordingBackend):
        def predict(self, request):
            response = super().predict(request)
            info = stage_info(request.questions[0])
            if info.get("stage") == "conditional" and info["group"] == "group2":
                response["answers"]["click_target"]["probabilities"]["node-053"] = float("nan")
                response["provider_metadata"]["invalid_numbers"] = [float("inf"), float("-inf")]
                self.invalid_response = copy.deepcopy(response)
            return response

    backend = NonfiniteLastLeaf()
    with pytest.raises(BackendResponseError) as caught:
        adapter(backend).predict(browser_payload())
    trace = caught.value.hierarchy
    assert trace["status"] == "failed"
    failed = trace["stages"][-1]
    assert failed["status"] == "failed"
    assert failed["scope"] == "conditional"
    assert failed["group"] == "group2"
    for key in ("raw_response", "response"):
        response = failed[key]
        assert response["answers"]["click_target"]["probabilities"]["node-053"] == {
            "nonfinite_float": "nan"
        }
        assert response["provider_metadata"]["invalid_numbers"] == [
            {"nonfinite_float": "inf"},
            {"nonfinite_float": "-inf"},
        ]
    assert json.loads(json.dumps(trace, allow_nan=False)) == trace
    assert json.loads(json.dumps(caught.value.generation, allow_nan=False))["status"] == "failed"
    original_nan = backend.invalid_response["answers"]["click_target"]["probabilities"]["node-053"]
    assert original_nan != original_nan
    assert "joint_probabilities" not in trace["heads"]["click_target"]


def test_available_preflight_checks_every_expanded_context_before_any_predict():
    failure = RequestValidationError("final leaf exceeds context capacity")

    class PreflightBackend(RecordingBackend):
        def __init__(self):
            super().__init__()
            self.preflight_calls = []

        def preflight(self, request):
            self.preflight_calls.append(copy.deepcopy(request))
            info = stage_info(request.questions[0])
            if info.get("stage") == "conditional" and info["group"] == "group2":
                raise failure
            return {"prepared_context_tokens": 123}

    backend = PreflightBackend()
    with pytest.raises(RequestValidationError) as caught:
        adapter(backend).predict(browser_payload())
    assert caught.value is failure
    assert not backend.calls
    assert len(backend.preflight_calls) == 5
    assert stage_info(backend.preflight_calls[-1].questions[0])["group"] == "group2"
    trace = caught.value.hierarchy
    assert trace["status"] == "failed"
    assert trace["backend_predict_calls"] == 0
    failed = trace["stages"][-1]
    assert failed["status"] == "failed"
    assert failed["error_stage"] == "preflight"
    assert failed["error"]
    assert failed["request"] == backend.preflight_calls[-1].model_dump(exclude_none=True)


def test_native_model_prepare_rejects_late_context_before_any_prediction():
    failure = RequestValidationError("prepared model context too large")

    class PreparedModel:
        def __init__(self):
            self.requests = []

        def prepare(self, request):
            self.requests.append(copy.deepcopy(request))
            info = stage_info(request.questions[0])
            if info.get("stage") == "conditional" and info["group"] == "group2":
                raise failure
            return {"input_ids": SimpleNamespace(shape=(1, 123))}, None, None

    backend = RecordingBackend()
    backend.model = PreparedModel()
    with pytest.raises(RequestValidationError) as caught:
        adapter(backend).predict(browser_payload())
    assert caught.value is failure
    assert not backend.calls
    assert len(backend.model.requests) == 5
    stages = caught.value.hierarchy["stages"]
    for stage in stages[:-1]:
        assert stage["preflight"] == {"scope": "full_native_context", "input_tokens": 123}
        assert stage["preflight_elapsed_ms"] >= 0
    assert stages[-1]["error_stage"] == "preflight"
    assert stages[-1]["group"] == "group2"
    assert caught.value.hierarchy["backend_predict_calls"] == 0


@pytest.mark.parametrize(
    ("mode", "expected_scope"),
    [
        ("user", "full_native_context"),
        ("reasoned", "base_context_only"),
        ("grounded", "full_native_context"),
    ],
)
def test_browser_native_preparation_reports_its_scope_and_rejects_late_before_predict(
    mode, expected_scope
):
    failure = RequestValidationError("browser prepared context too large")

    class FallbackModel:
        calls = 0

        def prepare(self, request):
            self.calls += 1
            raise AssertionError("native browser preparation must take priority over model.prepare")

    class PreparedBrowserBackend(RecordingBackend):
        prompt_mode = mode

        def __init__(self):
            super().__init__()
            self.prepared = []
            self.model = FallbackModel()

        def _prepare_question(self, request, question):
            self.prepared.append((copy.deepcopy(request), copy.deepcopy(question)))
            info = stage_info(question)
            if info.get("stage") == "conditional" and info["group"] == "group2":
                raise failure
            return "prepared input", 321

    backend = PreparedBrowserBackend()
    with pytest.raises(RequestValidationError) as caught:
        adapter(backend).predict(browser_payload())
    assert caught.value is failure
    assert not backend.calls
    assert backend.model.calls == 0
    assert len(backend.prepared) == 5
    stages = caught.value.hierarchy["stages"]
    for stage in stages[:-1]:
        assert stage["preflight"] == {"scope": expected_scope, "input_tokens": [321]}
    assert stages[-1]["error_stage"] == "preflight"
    assert stages[-1]["group"] == "group2"
    assert caught.value.hierarchy["backend_predict_calls"] == 0


def test_decider_native_preparation_measures_full_context_and_rejects_late_capacity():
    class PreparedEngine:
        def __init__(self):
            self.payloads = []

        def _system_one_items(self, **payload):
            self.payloads.append(copy.deepcopy(payload))
            question = next(iter(payload["questions"].values()))
            info = json.loads(question["instructions"]).get("hierarchy", {})
            over_limit = info.get("stage") == "conditional" and info["group"] == "group2"
            return None, None, [{"ids": list(range(124 if over_limit else 123))}]

    backend = RecordingBackend()
    backend.spec = SimpleNamespace(
        adapter="decider", context_limit=123, capabilities=SimpleNamespace(max_options=52)
    )
    backend.engine = PreparedEngine()
    payload = browser_payload()
    with pytest.raises(RequestValidationError, match="Decider context limit") as caught:
        adapter(backend).predict(payload)
    assert not backend.calls
    assert len(backend.engine.payloads) == 5
    for prepared in backend.engine.payloads:
        assert prepared["state"] == payload["state"]
        assert prepared["independent"] is True
        assert prepared["max_state_tokens"] == 2**31 - 1
    stages = caught.value.hierarchy["stages"]
    for stage in stages[:-1]:
        assert stage["preflight"] == {"scope": "full_native_context", "input_tokens": [123]}
    assert stages[-1]["error_stage"] == "preflight"
    assert stages[-1]["group"] == "group2"
    assert caught.value.hierarchy["backend_predict_calls"] == 0


def test_successful_preflight_finishes_all_stages_before_first_prediction():
    class PreflightBackend(RecordingBackend):
        def __init__(self):
            super().__init__()
            self.events = []

        def preflight(self, request):
            self.events.append(("preflight", copy.deepcopy(request)))
            return {"context_tokens": len(request.questions[0].instructions)}

        def predict(self, request):
            self.events.append(("predict", copy.deepcopy(request)))
            return super().predict(request)

    backend = PreflightBackend()
    result = adapter(backend).predict(browser_payload())
    assert [event for event, _ in backend.events] == ["preflight"] * 5 + ["predict"] * 5
    for index, stage in enumerate(result["hierarchy"]["stages"]):
        assert backend.events[index][1] == backend.calls[index]
        assert stage["preflight"]["metadata"]["context_tokens"] == len(
            backend.calls[index].questions[0].instructions
        )


def test_different_reported_models_across_stages_fail_with_both_responses_preserved():
    class ChangedModelBackend(RecordingBackend):
        def predict(self, request):
            response = super().predict(request)
            if len(self.calls) == 2:
                response["model"] = "different-checkpoint"
            return response

    backend = ChangedModelBackend()
    with pytest.raises(BackendResponseError) as caught:
        adapter(backend).predict(browser_payload())
    assert len(backend.calls) == 2
    stages = caught.value.hierarchy["stages"]
    assert stages[0]["response"]["model"] == "offline-policy"
    assert stages[1]["response"]["model"] == "different-checkpoint"
    assert stages[1]["status"] == "failed"
    assert caught.value.hierarchy["status"] == "failed"


def test_missing_reported_usage_stays_unknown_instead_of_fabricating_token_counts():
    class MissingUsageBackend(RecordingBackend):
        def predict(self, request):
            response = super().predict(request)
            if stage_info(request.questions[0]).get("stage") == "conditional":
                response.pop("usage")
            return response

    backend = MissingUsageBackend()
    result = adapter(backend).predict(browser_payload())
    assert result["usage"] == {"input_tokens": None, "output_tokens": None}
    assert result["hierarchy"]["backend_predict_calls"] == len(backend.calls) == 5
    assert all(
        "usage" not in stage["response"]
        for stage in result["hierarchy"]["stages"]
        if stage["scope"] == "conditional"
    )


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now


def test_cooperative_timeout_keeps_completed_blocking_call_evidence_and_stops_next_call(
    monkeypatch,
):
    clock = FakeClock()
    monkeypatch.setattr(hierarchy_module.time, "monotonic", clock.monotonic)

    class SlowBackend(RecordingBackend):
        def predict(self, request):
            response = super().predict(request)
            clock.now += 2.0
            self.blocking_call_completed = True
            return response

    backend = SlowBackend()
    with pytest.raises(BackendTimeoutError) as caught:
        adapter(backend, budget_seconds=1).predict(browser_payload())
    assert backend.blocking_call_completed
    assert len(backend.calls) == 1
    trace = caught.value.hierarchy
    assert trace["status"] == "failed"
    assert trace["timeout_policy"] == "cooperative_before_and_after_blocking_calls_no_hard_cancel"
    assert trace["backend_predict_calls"] == 1
    assert trace["elapsed_ms"] == pytest.approx(2000)
    first = trace["stages"][0]
    assert first["response"] == backend.responses[0]
    assert first["elapsed_ms"] == pytest.approx(2000)
    assert first["status"] == "failed"


def test_preflight_time_consumes_cooperative_budget_before_predict(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(hierarchy_module.time, "monotonic", clock.monotonic)

    class SlowPreflightBackend(RecordingBackend):
        def preflight(self, request):
            clock.now += 2.0
            return {"prepared_context_tokens": 123}

    backend = SlowPreflightBackend()
    with pytest.raises(BackendTimeoutError) as caught:
        adapter(backend, budget_seconds=1).predict(browser_payload())
    assert not backend.calls
    assert caught.value.hierarchy["backend_predict_calls"] == 0
    assert caught.value.hierarchy["status"] == "failed"


class RecordingBatchBackend(RecordingBackend):
    """Return ordered fake responses with shared-forward accounting evidence."""

    supports_stage_batching = True

    def __init__(self, probabilities=None):
        super().__init__(probabilities)
        self.batch_calls = []
        self.batch_responses = []
        self.events = []

    def preflight(self, request):
        self.events.append(("preflight", copy.deepcopy(request)))
        return {"prepared_context_tokens": 123}

    def _batch_response(self, request, index, count):
        answers = {}
        for question in request.questions:
            values = (
                self.probabilities(question)
                if self.probabilities is not None
                else [1 / len(question.labels())] * len(question.labels())
            )
            answers[question.id] = answer_from_probabilities(question, values)
        forward_count = (count + 1) // 2
        return {
            "model": self.name,
            "answers": answers,
            "usage": {"input_tokens": 11 + index, "output_tokens": 2},
            "execution": {
                "batch_id": "fake-stage-batch",
                "forward_calls_scope": "allocated_once_per_shared_browser_batch",
                "forward_calls": forward_count if index == 0 else 0,
                "batch_forward_calls": forward_count,
                "forward_refs": [{"chunk_id": f"chunk-{index // 2}", "forward_index": index // 2}],
                "batch_chunk_ids": [f"chunk-{chunk}" for chunk in range(forward_count)],
                "max_forward_batch_size": 2,
                "input_tokens": 11 + index,
                "latency_scope": "whole_browser_batch",
                "batch_elapsed_ms": 7.5,
            },
        }

    def predict_batch(self, requests):
        self.batch_calls.append(copy.deepcopy(requests))
        self.events.append(("batch", copy.deepcopy(requests)))
        responses = [
            self._batch_response(request, index, len(requests))
            for index, request in enumerate(requests)
        ]
        self.batch_responses = copy.deepcopy(responses)
        return responses


def test_optional_batch_executes_every_stage_once_with_real_joint_probabilities_and_shared_evidence():
    def probabilities(question):
        info = stage_info(question)
        if info.get("stage") == "group":
            return [0.4, 0.3, 0.2, 0.1]
        if info.get("stage") == "conditional" and info["group"] == "group1":
            return [0.03 / (len(question.labels()) - 1)] * (len(question.labels()) - 1) + [0.97]
        return [1 / len(question.labels())] * len(question.labels())

    payload = browser_payload(80)
    original = copy.deepcopy(payload)
    backend = RecordingBatchBackend(probabilities)
    result = adapter(backend, batch_stages=True).predict(payload)
    assert payload == original
    assert not backend.calls
    assert len(backend.batch_calls) == 1
    requests = backend.batch_calls[0]
    assert len(requests) == 6
    assert [event for event, _ in backend.events] == ["preflight"] * 6 + ["batch"]
    assert [stage_info(request.questions[0]).get("stage") for request in requests] == [
        None,
        "group",
        "conditional",
        "conditional",
        "conditional",
        "conditional",
    ]
    assert [stage_info(request.questions[0])["group"] for request in requests[2:]] == [
        "group0",
        "group1",
        "group2",
        "group3",
    ]
    answer = result["answers"]["click_target"]
    assert list(answer["probabilities"]) == list(payload["questions"]["click_target"]["criteria"])
    assert answer["choice"] == "node-040"
    assert answer["probabilities"]["node-040"] == pytest.approx(0.3 * 0.97)
    assert sum(answer["probabilities"].values()) == pytest.approx(1)
    trace = result["hierarchy"]
    assert trace["probabilities_semantics"] == SEMANTICS
    assert trace["heads"]["click_target"]["group_probabilities"]["group0"] == 0.4
    assert trace["backend_predict_batch_calls"] == 1
    assert trace["backend_predict_calls"] == 0
    assert trace["planned_stage_predictions"] == 6
    assert trace["stage_batching"] == {
        "requested": True,
        "method_available": True,
        "supported": True,
        "available": True,
        "used": True,
        "fallback_reason": None,
    }
    execution = trace["batch_execution"]
    assert execution["status"] == "completed"
    assert execution["stage_indices"] == list(range(6))
    assert execution["expected_response_count"] == execution["response_count"] == 6
    assert execution["raw_responses"] == backend.batch_responses
    assert execution["latency_scope"] == "whole_hierarchy_backend_batch"
    assert execution["backend_execution"]["allocated_forward_calls"] == 3
    assert len(execution["backend_execution"]["batches"]) == 1
    shared = execution["backend_execution"]["batches"][0]
    assert shared["batch_id"] == "fake-stage-batch"
    assert shared["stage_indices"] == list(range(6))
    for index, stage in enumerate(trace["stages"]):
        assert stage["batch_stage_index"] == index
        assert stage["elapsed_ms"] is None
        assert stage["latency_scope"] == "shared_backend_batch"
        assert stage["raw_response"] == backend.batch_responses[index]
        assert stage["response"]["execution"] == backend.batch_responses[index]["execution"]
        assert stage["request"] == requests[index].model_dump(exclude_none=True)
    assert result["usage"] == {"input_tokens": sum(range(11, 17)), "output_tokens": 12}


@pytest.mark.parametrize("fallback", ["default-disabled", "missing-method", "declared-unsupported"])
def test_stage_batching_defaults_and_supported_fallbacks_keep_sequential_execution(fallback):
    backend = RecordingBatchBackend()
    requested = fallback != "default-disabled"
    if fallback == "missing-method":
        backend.predict_batch = None
    elif fallback == "declared-unsupported":
        backend.supports_stage_batching = False
    kwargs = {"batch_stages": True} if requested else {}
    result = adapter(backend, **kwargs).predict(browser_payload(80))
    assert not backend.batch_calls
    assert len(backend.calls) == 6
    trace = result["hierarchy"]
    assert trace["backend_predict_batch_calls"] == 0
    assert trace["backend_predict_calls"] == 6
    flags = trace["stage_batching"]
    assert flags["requested"] is requested
    assert flags["used"] is False
    assert flags["method_available"] is (fallback != "missing-method")
    assert flags["supported"] is (fallback != "declared-unsupported")
    assert flags["available"] is (fallback == "default-disabled")
    if requested:
        assert flags["fallback_reason"]


def test_batch_flag_does_not_change_exact_small_request_passthrough():
    payload = browser_payload(20)
    backend = RecordingBatchBackend()
    expected_backend = RecordingBatchBackend()
    result = adapter(backend, batch_stages=True).predict(payload)
    expected = JevCompatibilityAdapter(expected_backend, model_identity="browser-alias").predict(
        payload
    )
    assert result == expected
    assert backend.calls == expected_backend.calls
    assert not backend.batch_calls
    assert not backend.events
    assert "hierarchy" not in result


@pytest.mark.parametrize("flag", [1, "true", None])
def test_batch_flag_must_be_an_explicit_boolean(flag):
    with pytest.raises(ValueError):
        adapter(RecordingBatchBackend(), batch_stages=flag)


@pytest.mark.parametrize("failure_kind", ["first-distribution", "second-model"])
def test_batch_saves_every_raw_result_before_normalizing_any_response(failure_kind):
    class MalformedBatch(RecordingBatchBackend):
        def predict_batch(self, requests):
            responses = super().predict_batch(requests)
            if failure_kind == "first-distribution":
                del responses[0]["answers"]["operation"]["probabilities"]["DONE"]
            else:
                responses[1]["model"] = "different-checkpoint"
            self.batch_responses = copy.deepcopy(responses)
            return responses

    backend = MalformedBatch()
    with pytest.raises(BackendResponseError) as caught:
        adapter(backend, batch_stages=True).predict(browser_payload(80))
    assert not backend.calls
    assert len(backend.batch_calls) == 1
    trace = caught.value.hierarchy
    assert trace["status"] == "failed"
    assert trace["batch_execution"]["raw_responses"] == backend.batch_responses
    assert [stage["raw_response"] for stage in trace["stages"]] == backend.batch_responses
    assert trace["backend_predict_batch_calls"] == 1
    assert trace["backend_predict_calls"] == 0
    assert "joint_probabilities" not in trace["heads"]["click_target"]


@pytest.mark.parametrize("difference", [-1, 1])
def test_batch_response_count_failure_retains_all_returns_including_unmatched_extra(difference):
    class WrongCountBatch(RecordingBatchBackend):
        def predict_batch(self, requests):
            responses = super().predict_batch(requests)
            if difference < 0:
                responses.pop()
            else:
                responses.append(copy.deepcopy(responses[-1]))
            self.batch_responses = copy.deepcopy(responses)
            return responses

    backend = WrongCountBatch()
    with pytest.raises(BackendResponseError) as caught:
        adapter(backend, batch_stages=True).predict(browser_payload(80))
    trace = caught.value.hierarchy
    execution = trace["batch_execution"]
    assert execution["expected_response_count"] == 6
    assert execution["response_count"] == 6 + difference
    assert execution["raw_responses"] == backend.batch_responses
    for index in range(min(6, len(backend.batch_responses))):
        assert trace["stages"][index]["raw_response"] == backend.batch_responses[index]
    assert not backend.calls
    assert len(backend.batch_calls) == 1
    assert "joint_probabilities" not in trace["heads"]["click_target"]


def test_batch_contract_rejects_tuple_return_even_with_correct_order_and_count():
    class TupleBatch(RecordingBatchBackend):
        def predict_batch(self, requests):
            return tuple(super().predict_batch(requests))

    backend = TupleBatch()
    with pytest.raises(BackendResponseError):
        adapter(backend, batch_stages=True).predict(browser_payload(80))
    assert len(backend.batch_calls) == 1
    assert not backend.calls


def test_failed_backend_batch_preserves_partial_receipt_and_original_exception_evidence():
    failure = BackendTimeoutError("backend batch stopped after partial completion")
    original_generation = {"raw": "original provider generation"}
    original_hierarchy = {"original_backend_trace": {"batch": "fake-stage-batch"}}
    failure.generation = copy.deepcopy(original_generation)
    failure.hierarchy = copy.deepcopy(original_hierarchy)

    class FailedBatch(RecordingBatchBackend):
        def predict_batch(self, requests):
            responses = super().predict_batch(requests)
            failure.batch_evidence = {
                "batch_id": "fake-stage-batch",
                "completed_response_indices": [0, 2],
                "completed_responses": [responses[0], responses[2]],
                "elapsed_ms": 7.5,
            }
            self.failure_receipt = copy.deepcopy(failure.batch_evidence)
            raise failure

    backend = FailedBatch()
    with pytest.raises(BackendTimeoutError) as caught:
        adapter(backend, batch_stages=True).predict(browser_payload(80))
    assert caught.value is failure
    assert caught.value.generation == original_generation
    assert caught.value.batch_evidence == backend.failure_receipt
    trace = caught.value.hierarchy
    assert trace["batch_error_evidence"] == backend.failure_receipt
    assert trace["backend_error_hierarchy"] == original_hierarchy
    assert trace["stages"][0]["raw_response"] == backend.failure_receipt["completed_responses"][0]
    assert trace["stages"][2]["raw_response"] == backend.failure_receipt["completed_responses"][1]
    assert trace["status"] == "failed"
    assert trace["backend_predict_batch_calls"] == 1
    assert trace["backend_predict_calls"] == 0
    assert not backend.calls
    assert "joint_probabilities" not in trace["heads"]["click_target"]


def test_batch_cooperative_deadline_keeps_all_raw_responses_after_blocking_call(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(hierarchy_module.time, "monotonic", clock.monotonic)

    class SlowBatch(RecordingBatchBackend):
        def predict_batch(self, requests):
            responses = super().predict_batch(requests)
            clock.now += 2.0
            self.blocking_call_completed = True
            return responses

    backend = SlowBatch()
    with pytest.raises(BackendTimeoutError) as caught:
        adapter(backend, batch_stages=True, budget_seconds=1).predict(browser_payload(80))
    assert backend.blocking_call_completed
    assert not backend.calls
    assert len(backend.batch_calls) == 1
    trace = caught.value.hierarchy
    assert trace["status"] == "failed"
    assert trace["elapsed_ms"] == pytest.approx(2000)
    assert trace["batch_execution"]["status"] == "failed"
    assert trace["batch_execution"]["elapsed_ms"] == pytest.approx(2000)
    assert trace["batch_execution"]["raw_responses"] == backend.batch_responses
    for index, stage in enumerate(trace["stages"]):
        assert stage["raw_response"] == backend.batch_responses[index]
        assert stage["status"] == "returned_unvalidated"
        assert stage["elapsed_ms"] is None
        assert stage["latency_scope"] == "shared_backend_batch"


def test_late_preflight_failure_prevents_both_batch_and_sequential_prediction():
    failure = RequestValidationError("final prepared context too large")

    class RejectedPreflightBatch(RecordingBatchBackend):
        def preflight(self, request):
            metadata = super().preflight(request)
            info = stage_info(request.questions[0])
            if info.get("stage") == "conditional" and info["group"] == "group3":
                raise failure
            return metadata

    backend = RejectedPreflightBatch()
    with pytest.raises(RequestValidationError) as caught:
        adapter(backend, batch_stages=True).predict(browser_payload(80))
    assert caught.value is failure
    assert [event for event, _ in backend.events] == ["preflight"] * 6
    assert not backend.calls
    assert not backend.batch_calls
    assert caught.value.hierarchy["backend_predict_batch_calls"] == 0
    assert caught.value.hierarchy["backend_predict_calls"] == 0
    assert caught.value.hierarchy["stage_batching"]["used"] is False


def test_identical_stage_distributions_preserve_exact_answers_and_usage_between_execution_modes():
    def probabilities(question):
        weights = list(range(1, len(question.labels()) + 1))
        return [weight / sum(weights) for weight in weights]

    class EquivalentSequentialBackend(RecordingBatchBackend):
        def predict(self, request):
            self.calls.append(copy.deepcopy(request))
            response = self._batch_response(request, len(self.calls) - 1, 6)
            self.responses.append(copy.deepcopy(response))
            return response

    payload = browser_payload(80)
    sequential_backend = EquivalentSequentialBackend(probabilities)
    batched_backend = RecordingBatchBackend(probabilities)
    sequential = adapter(sequential_backend).predict(payload)
    batched = adapter(batched_backend, batch_stages=True).predict(payload)
    assert batched["answers"] == sequential["answers"]
    assert (
        batched["usage"]
        == sequential["usage"]
        == {"input_tokens": sum(range(11, 17)), "output_tokens": 12}
    )
    assert batched["hierarchy"]["heads"] == sequential["hierarchy"]["heads"]
    assert sequential_backend.calls == batched_backend.batch_calls[0]
    assert sequential["hierarchy"]["backend_predict_calls"] == 6
    assert batched["hierarchy"]["backend_predict_calls"] == 0
    assert sequential["hierarchy"]["backend_predict_batch_calls"] == 0
    assert batched["hierarchy"]["backend_predict_batch_calls"] == 1
    for result in (sequential, batched):
        assert result["hierarchy"]["planned_backend_calls"] == 6
        assert (
            result["hierarchy"]["planned_backend_calls_scope"]
            == "native_stage_requests_not_batch_method_calls"
        )


def test_batch_skips_mathematically_forced_singleton_leaf_and_preserves_its_distinct_audit():
    backend = RecordingBatchBackend()
    payload = browser_payload(3)
    result = adapter(backend, batch_stages=True, branch_width=2).predict(payload)
    assert not backend.calls
    assert len(backend.batch_calls) == 1
    requests = backend.batch_calls[0]
    assert len(requests) == 3
    assert [stage_info(request.questions[0]).get("stage") for request in requests] == [
        None,
        "group",
        "conditional",
    ]
    assert requests[-1].questions[0].labels() == ["node-001", "node-002"]
    trace = result["hierarchy"]
    assert trace["batch_execution"]["stage_indices"] == [0, 1, 2]
    assert trace["batch_execution"]["expected_response_count"] == 3
    assert trace["planned_stage_predictions"] == 3
    assert len(trace["stages"]) == 4
    assert [stage["batch_stage_index"] for stage in trace["stages"][:3]] == [0, 1, 2]
    singleton = trace["stages"][3]
    assert singleton["scope"] == "conditional"
    assert singleton["group"] == "group1"
    assert singleton["status"] == "deterministic"
    assert singleton["request"]["mathematically_forced_singleton"] is True
    assert list(singleton["request"]["questions"][0]["criteria"]) == ["node-003"]
    assert (
        json.loads(singleton["request"]["questions"][0]["criteria"]["node-003"])
        == payload["questions"]["click_target"]["criteria"]["node-003"]
    )
    assert singleton["response"]["answers"]["click_target"]["probabilities"] == {"node-003": 1.0}
    assert "batch_stage_index" not in singleton
    assert result["answers"]["click_target"]["probabilities"] == {
        "node-001": 0.25,
        "node-002": 0.25,
        "node-003": 0.5,
    }
    assert result["answers"]["click_target"]["choice"] == "node-003"


@pytest.mark.parametrize("failure_kind", ["count", "distribution"])
def test_adapter_batch_validation_errors_expose_full_genuine_receipt_and_keep_generation(
    failure_kind, monkeypatch
):
    original_generation = {"raw": "original normalization evidence"}
    normalize = hierarchy_module.normalize_response

    def preserve_normalization_generation(request, response):
        try:
            return normalize(request, response)
        except BackendResponseError as exc:
            exc.generation = copy.deepcopy(original_generation)
            raise

    monkeypatch.setattr(hierarchy_module, "normalize_response", preserve_normalization_generation)

    class InvalidBatch(RecordingBatchBackend):
        def predict_batch(self, requests):
            responses = super().predict_batch(requests)
            if failure_kind == "count":
                responses.append(copy.deepcopy(responses[-1]))
            else:
                del responses[0]["answers"]["operation"]["probabilities"]["DONE"]
            self.batch_responses = copy.deepcopy(responses)
            return responses

    backend = InvalidBatch()
    with pytest.raises(BackendResponseError) as caught:
        adapter(backend, batch_stages=True).predict(browser_payload(80))
    receipt = caught.value.batch_evidence
    assert receipt["raw_responses"] == backend.batch_responses
    assert receipt["completed_response_indices"] == list(range(6))
    assert receipt["completed_responses"] == backend.batch_responses[:6]
    assert receipt["validation_scope"] == "returned_native_rows_not_all_validated"
    assert caught.value.hierarchy["batch_error_evidence"] == receipt
    assert len(receipt["raw_responses"]) == (7 if failure_kind == "count" else 6)
    if failure_kind == "distribution":
        assert caught.value.generation == original_generation
    assert not backend.calls
    assert len(backend.batch_calls) == 1


def test_request_owned_batch_chunks_are_unioned_without_shared_metadata_conflicts():
    class RequestOwnedChunks(RecordingBatchBackend):
        def predict_batch(self, requests):
            responses = super().predict_batch(requests)
            for index, response in enumerate(responses):
                response["execution"]["batch_chunk_ids"] = [f"chunk-{index // 2}"]
            self.batch_responses = copy.deepcopy(responses)
            return responses

    backend = RequestOwnedChunks()
    result = adapter(backend, batch_stages=True).predict(browser_payload(80))
    metadata = result["hierarchy"]["batch_execution"]["backend_execution"]
    assert metadata["allocated_forward_calls"] == 3
    assert len(metadata["batches"]) == 1
    batch = metadata["batches"][0]
    assert batch["batch_id"] == "fake-stage-batch"
    assert batch["stage_indices"] == list(range(6))
    assert batch["batch_chunk_ids"] == ["chunk-0", "chunk-1", "chunk-2"]
    assert batch["forward_refs"] == [
        {"chunk_id": f"chunk-{index}", "forward_index": index} for index in range(3)
    ]
    assert batch["batch_forward_calls"] == 3
    assert batch["batch_elapsed_ms"] == 7.5
    assert batch["latency_scope"] == "whole_browser_batch"
    assert batch["forward_calls_scope"] == "allocated_once_per_shared_browser_batch"
    assert batch["max_forward_batch_size"] == 2
    assert not batch.get("inconsistent_shared_metadata")
    for index, stage in enumerate(result["hierarchy"]["stages"]):
        assert stage["raw_response"]["execution"]["batch_chunk_ids"] == [f"chunk-{index // 2}"]


def test_gemma_whole_adapter_batch_forward_count_is_preserved_once_without_allocation_sum():
    shared_execution = {
        "scope": "whole_adapter_batch",
        "forward_calls": 3,
        "batch_sizes": [2, 2, 2],
        "sequence_tokens": [192, 176, 161],
    }

    class WholeAdapterBatch(RecordingBatchBackend):
        def predict_batch(self, requests):
            responses = super().predict_batch(requests)
            for response in responses:
                response["execution"] = copy.deepcopy(shared_execution)
            self.batch_responses = copy.deepcopy(responses)
            return responses

    backend = WholeAdapterBatch()
    result = adapter(backend, batch_stages=True).predict(browser_payload(80))
    metadata = result["hierarchy"]["batch_execution"]["backend_execution"]
    assert metadata["allocated_forward_calls"] is None
    assert metadata["batches"] == []
    whole_batch = metadata["whole_adapter_batch"]
    assert whole_batch["stage_indices"] == list(range(6))
    assert whole_batch["execution"] == shared_execution
    assert whole_batch["forward_calls"] == 3
    for stage in result["hierarchy"]["stages"]:
        assert stage["raw_response"]["execution"] == shared_execution
        assert stage["response"]["execution"] == shared_execution


def test_unscoped_forward_counts_remain_unknown_in_aggregate_and_survive_as_raw_evidence():
    class UnscopedBatch(RecordingBatchBackend):
        def predict_batch(self, requests):
            responses = super().predict_batch(requests)
            for response in responses:
                response["execution"] = {"forward_calls": 3}
            self.batch_responses = copy.deepcopy(responses)
            return responses

    backend = UnscopedBatch()
    result = adapter(backend, batch_stages=True).predict(browser_payload(80))
    metadata = result["hierarchy"]["batch_execution"]["backend_execution"]
    assert metadata["allocated_forward_calls"] is None
    assert metadata["batches"] == []
    assert metadata.get("whole_adapter_batch") is None
    for stage in result["hierarchy"]["stages"]:
        assert stage["raw_response"]["execution"] == {"forward_calls": 3}
        assert stage["response"]["execution"] == {"forward_calls": 3}


def test_inconsistent_whole_adapter_metadata_marks_shared_count_unknown_and_preserves_variants():
    shared_execution = {
        "scope": "whole_adapter_batch",
        "forward_calls": 3,
        "batch_sizes": [2, 2, 2],
        "sequence_tokens": [192, 176, 161],
    }
    different_execution = {**shared_execution, "forward_calls": 4}

    class InconsistentWholeBatch(RecordingBatchBackend):
        def predict_batch(self, requests):
            responses = super().predict_batch(requests)
            for response in responses:
                response["execution"] = copy.deepcopy(shared_execution)
            responses[-1]["execution"] = copy.deepcopy(different_execution)
            self.batch_responses = copy.deepcopy(responses)
            return responses

    backend = InconsistentWholeBatch()
    result = adapter(backend, batch_stages=True).predict(browser_payload(80))
    metadata = result["hierarchy"]["batch_execution"]["backend_execution"]
    assert metadata["allocated_forward_calls"] is None
    whole_batch = metadata["whole_adapter_batch"]
    assert whole_batch["stage_indices"] == list(range(6))
    assert whole_batch["execution"] == shared_execution
    assert whole_batch["forward_calls"] is None
    assert whole_batch["inconsistent_shared_metadata"] == [shared_execution, different_execution]
    assert result["hierarchy"]["stages"][-1]["raw_response"]["execution"] == different_execution
