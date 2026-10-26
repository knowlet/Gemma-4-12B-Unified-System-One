"""Selected canonical Agent/choose with fake native review and browser I/O."""

import copy
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
ROOT = HERE.parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


shared_path = (
    ROOT / "artifacts/jev-runtime/browser-find-text-profile-20261022-01/test_find_profile.py"
)
assert hashlib.sha256(shared_path.read_bytes()).hexdigest() == (
    "498c05819cb153c2d01717675847eb43899eab33cbc0f095942ad779e5c009ac"
)
shared = load("completion_shared_fixture", shared_path)
fixture = shared.fixture
profile = load("completion_component", HERE / "completion_review.py")


@pytest.fixture
def reviewed(fixture):
    now = [100.0]
    calls, options = [], {"choice": "INCOMPLETE", "hook": lambda: None}

    def review(body, deadline):
        calls.append((copy.deepcopy(body), deadline))
        options["hook"]()
        choice = options["choice"]
        return {
            "model": "fixture-native",
            "answers": {
                profile.QUESTION: {
                    "choice": choice,
                    "probabilities": {
                        "COMPLETE": float(choice == "COMPLETE"),
                        "INCOMPLETE": float(choice == "INCOMPLETE"),
                    },
                    "confidence": 1.0,
                }
            },
            "usage": {"fixture_only": True},
        }

    cls = profile.make_agent_class(fixture.module, fixture.model, review, clock=lambda: now[0])
    instance = cls(
        "https://fixture.invalid/", "Show the requested method signature and description"
    )
    fixture.options["operation"] = "DONE"
    return instance, fixture, now, calls, options


def act(instance):
    return instance.command("act", {"fingerprint": instance.state["page"]["fingerprint"]})


def test_veto_keeps_done_and_resumes_real_canonical_navigation(reviewed):
    instance, f, _, calls, _ = reviewed
    proposal = instance.command("predict")["decision"]
    snapshot = act(instance)
    assert snapshot["status"] == "ready" and snapshot["plan_index"] == 0
    assert instance.state["decisions"][-1]["operation"] == "DONE"
    assert instance.state["completion_reviews"][-1]["proposal"] == proposal
    assert instance.state["history"][-1]["operation"] == "DONE"
    assert len(calls) == 1 and not instance.browser.inputs
    f.options["operation"] = "CLICK"
    instance.command("tick")
    assert instance.browser.inputs and instance.state["history"][-1]["operation"] == "CLICK"
    assert f.sent[-1]["state"]["recent_actions"][0]["action"] == "Review proposed DONE: vetoed"


def test_approved_has_native_answer_but_no_external_task_success_claim(reviewed):
    instance, _, _, calls, options = reviewed
    options["choice"] = "COMPLETE"
    snapshot = instance.command("tick")
    assert snapshot["status"] == "done" and snapshot["plan_index"] == 1
    assert snapshot["completion_reviews"][-1]["answer"]["choice"] == "COMPLETE"
    assert snapshot["completion_reviews"][-1]["task_success"] == "not_evaluated"
    assert len(calls) == 1 and len(snapshot["history"]) == 1


def test_stale_before_review_consumes_once_and_preserves_typed_record(reviewed):
    instance, fixture, _, calls, _ = reviewed
    instance.command("predict")
    instance.browser.fresh_values = [False]
    with pytest.raises(fixture.module.StalePage):
        act(instance)
    assert not calls and instance.state["decision"] is None
    assert instance.state["completion_reviews"][-1]["outcome"] == "stale"
    with pytest.raises(ValueError):
        act(instance)
    assert len(instance.state["history"]) == 1


def test_stale_after_review_keeps_answer_but_not_approval(reviewed):
    instance, fixture, _, calls, options = reviewed
    options["choice"] = "COMPLETE"
    instance.command("predict")
    instance.browser.fresh_values = [True, False]
    with pytest.raises(fixture.module.StalePage):
        act(instance)
    record = instance.state["completion_reviews"][-1]
    assert record["answer"]["choice"] == "COMPLETE" and record["outcome"] == "stale"
    assert instance.state["status"] == "ready" and len(calls) == 1


def test_parent_tick_catches_stale_without_erasing_review(reviewed):
    instance, _, _, calls, options = reviewed
    options["choice"] = "COMPLETE"
    instance.browser.fresh_values = [True, True, False]
    snapshot = instance.command("tick")
    assert snapshot["status"] == "ready" and len(snapshot["history"]) == 1
    assert snapshot["completion_reviews"][-1]["outcome"] == "stale"
    assert len(calls) == 1


def test_whole_policy_and_review_deadline_not_reset_after_prediction(reviewed):
    instance, _, now, calls, options = reviewed
    options["choice"] = "COMPLETE"
    instance.command("predict")
    now[0] = 119.0
    options["hook"] = lambda: now.__setitem__(0, 120.0)
    assert act(instance)["status"] == "ready"
    assert calls[0][1] == 120.0
    assert instance.state["completion_reviews"][-1]["outcome"] == "deadline"
    assert instance.state["completion_reviews"][-1]["answer"]["choice"] == "COMPLETE"


def test_expired_prediction_makes_no_review_call(reviewed):
    instance, _, now, calls, _ = reviewed
    instance.command("predict")
    now[0] = 120.0
    assert act(instance)["status"] == "ready" and not calls
    assert instance.state["history"][-1]["completion_review"]["outcome"] == "deadline"


def test_final_snapshot_work_cannot_publish_late_approval(reviewed):
    instance, _, now, _, options = reviewed
    options["choice"] = "COMPLETE"
    instance.command("predict")
    original = instance.snapshot

    def snapshot():
        result = original()
        now[0] = 120.0
        return result

    instance.snapshot = snapshot
    result = act(instance)
    assert result["status"] == "ready" and result["plan_index"] == 0
    assert result["completion_reviews"][-1]["outcome"] == "deadline"
    assert result["history"][-1]["completion_review"]["outcome"] == "deadline"
    assert result["completion_feedback"][-1]["review"] == "deadline"


def test_slow_freshness_cannot_publish_late_approval(reviewed):
    instance, _, now, _, options = reviewed
    options["choice"] = "COMPLETE"
    instance.command("predict")
    checks = [0]

    def fresh(_page):
        checks[0] += 1
        if checks[0] == 2:
            now[0] = 120.0
        return True

    instance.browser.fresh = fresh
    assert act(instance)["status"] == "ready"
    assert instance.state["completion_reviews"][-1]["outcome"] == "deadline"


def test_callback_error_is_recorded_and_propagated_not_quality_failure(reviewed):
    instance, _, _, _, options = reviewed
    error = RuntimeError("fixture transport failure")

    def fail():
        raise error

    options["hook"] = fail
    instance.command("predict")
    with pytest.raises(RuntimeError) as caught:
        act(instance)
    assert caught.value is error
    record = instance.state["completion_reviews"][-1]
    assert record["outcome"] == "error" and record["native_call_attempted"]
    assert record["task_success"] == "not_evaluated" and instance.state["status"] == "ready"


def test_invalid_native_choice_cannot_approve(reviewed):
    instance, _, _, _, options = reviewed
    options["choice"] = "invented"
    instance.command("predict")
    with pytest.raises(ValueError):
        act(instance)
    assert instance.state["completion_reviews"][-1]["outcome"] == "error"
    assert instance.state["status"] == "ready"


def test_sixteen_vetoes_and_thirty_two_logical_attempts_exhaust_budget(reviewed):
    instance, _, _, calls, _ = reviewed
    for _ in range(16):
        instance.command("tick")
    assert len(instance.state["history"]) == len(instance.state["decisions"]) == len(calls) == 16
    with pytest.raises(ValueError, match="thirty-two"):
        instance.command("predict")
    assert instance.state["status"] == "blocked" and len(calls) == 16


def test_prior_navigation_consumes_same_action_budget(reviewed):
    instance, _, _, calls, _ = reviewed
    instance.state["history"] = [{"action": "earlier actual action"}] * 16
    instance.command("predict")
    with pytest.raises(ValueError, match="sixteen"):
        act(instance)
    assert not calls and len(instance.state["history"]) == 16


def test_thirty_second_policy_call_has_no_thirty_third_review(reviewed):
    instance, _, _, calls, _ = reviewed
    instance.state["decisions"] = [{"fixture_prior_policy": True}] * 31
    instance.command("predict")
    snapshot = act(instance)
    assert snapshot["status"] == "blocked" and not calls
    assert snapshot["completion_reviews"][-1]["outcome"] == "request_attempt_budget"


def test_request_excludes_unobserved_and_judge_data_and_mutation_isolated(reviewed):
    instance, _, _, calls, _ = reviewed
    instance.state["page"].update(
        hidden_dom="hidden secret", expected_answer="judge secret", judge="private"
    )
    instance.state["expected_route"] = "route secret"
    instance.command("tick")
    body = calls[-1][0]
    assert set(body) == {"model", "state", "questions"}
    assert set(body["state"]["page"]) == {"url", "title", "text"}
    assert list(body["questions"]) == [profile.QUESTION]
    assert list(body["questions"][profile.QUESTION]["criteria"]) == ["COMPLETE", "INCOMPLETE"]
    assert body["questions"][profile.QUESTION]["instructions"]["goal"] == instance.state["goal"]
    assert "secret" not in str(body)
    body["state"]["page"]["text"] = "changed fixture"
    assert (
        instance.state["completion_reviews"][-1]["request"]["state"]["page"]["text"]
        != "changed fixture"
    )


def test_composes_with_find_capability_without_reviewing_navigation(fixture):
    calls = []
    cls = profile.make_agent_class(
        fixture.module, fixture.model, lambda *args: calls.append(args), base_class=fixture.cls
    )
    instance = cls("https://fixture.invalid/", "Find the requested method")
    instance.command("tick")
    assert instance.state["history"][-1]["operation"] == "FIND_TEXT"
    assert instance.browser.navigations and not calls
    assert fixture.module.Agent is fixture.original["Agent"]
    assert fixture.model.choose is fixture.original["model_choose"]


def test_wrong_native_model_rejected_without_argmax_substitution(reviewed):
    _, fixture, _, _, _ = reviewed
    with pytest.raises(ValueError, match="wrong model"):
        profile.validate_native_result(
            {"model": "wrong", "answers": {}}, fixture.model, "fixture-native"
        )


def test_missing_prediction_deadline_is_incomplete_and_no_call(reviewed):
    instance, _, _, calls, _ = reviewed
    instance.command("predict")
    del instance._completion_deadline
    assert act(instance)["status"] == "ready" and not calls
    assert instance.state["completion_reviews"][-1]["outcome"] == "deadline"


def test_failed_real_canonical_choose_attempts_cannot_escape_budget(reviewed):
    instance, fixture, _, review_calls, _ = reviewed
    actual_posts = []
    error = RuntimeError("fixture policy transport error")

    def fail(*args):
        actual_posts.append(args)
        raise error

    fixture.model.post_json = fail
    for _ in range(32):
        with pytest.raises(RuntimeError) as caught:
            instance.command("predict")
        assert caught.value is error
    assert not instance.state["decisions"] and not review_calls
    assert instance.state["completion_policy_attempts"] == len(actual_posts) == 32
    assert len(instance.state["completion_policy_failures"]) == 32
    with pytest.raises(ValueError, match="thirty-two-request-attempt"):
        instance.command("predict")
    assert len(actual_posts) == 32
    assert instance.state["status"] == "blocked"


def test_failed_policy_then_successful_reviews_share_same_attempt_ceiling(reviewed):
    instance, fixture, _, review_calls, _ = reviewed
    original = fixture.model.post_json

    def fail(*_args):
        raise RuntimeError("fixture failed first policy")

    fixture.model.post_json = fail
    with pytest.raises(RuntimeError):
        instance.command("predict")
    fixture.model.post_json = original
    for _ in range(15):
        instance.command("tick")
    assert instance.state["completion_policy_attempts"] == 16 and len(review_calls) == 15
    snapshot = instance.command("tick")
    assert snapshot["status"] == "blocked"
    assert snapshot["completion_policy_attempts"] == 17 and len(review_calls) == 15
    assert snapshot["completion_reviews"][-1]["outcome"] == "request_attempt_budget"
    assert len(snapshot["decisions"]) == 16 and len(snapshot["history"]) == 16
