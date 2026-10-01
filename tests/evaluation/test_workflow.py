import pytest

from s1.evaluation.contracts import ModelSpec
from s1.evaluation.workflow import Episode, compare_policies, run_episode, summarize_episodes


def episode():
    question = {
        "id": "action",
        "type": "choice",
        "instructions": "Choose next action",
        "criteria": {
            "retry": "Retry the transient operation",
            "complete": "Declare completion",
            "escalate": "Ask worker",
        },
    }
    return Episode.model_validate(
        {
            "id": "ci-1",
            "group_id": "ci-source",
            "family": "ci",
            "seed": 0,
            "initial_state": "failing",
            "max_steps": 3,
            "states": {
                "failing": {
                    "observation": {
                        "state": {"tests_passed": False, "retryable": True},
                        "questions": [question],
                    },
                    "transitions": {"retry": {"next_state": "passing", "tool_cost_usd": 0.1}},
                },
                "passing": {
                    "observation": {"state": {"tests_passed": True}, "questions": [question]},
                    "goal_satisfied": True,
                    "transitions": {"retry": {"next_state": "passing", "tool_cost_usd": 0.1}},
                },
            },
        }
    )


class Small:
    def predict(self, request):
        return {"answers": {"action": {"label": "complete"}}, "cost_usd": 0.01}


class Strong:
    def predict(self, request):
        label = "complete" if request.state["tests_passed"] else "retry"
        return {"answers": {"action": {"label": label}}, "cost_usd": 0.2}


def specs():
    return [
        ModelSpec(
            id=name,
            adapter="http",
            cost_per_request_usd=cost,
            capabilities={"probabilities": "none"},
        )
        for name, cost in [("small", 0.01), ("strong", 0.2)]
    ]


def test_workflow_checks_environment_not_model_claim_and_resets():
    scenario = episode()
    failed = run_episode(scenario, "always_small", Small(), Strong(), specs(), 0.8)
    assert failed["false_completion"] and not failed["success"]
    passed = run_episode(scenario, "always_strong", Small(), Strong(), specs(), 0.8)
    assert passed["success"] and passed["retry_steps"] == 1
    assert passed["reported_total_cost_usd"] == pytest.approx(0.5)
    cascade = run_episode(scenario, "cascade", Small(), Strong(), specs(), 0.8)
    assert cascade["success"] and cascade["fallback_count"] == 2
    assert cascade["model_switches"] == 3
    assert cascade["reported_total_cost_usd"] == pytest.approx(0.52)
    assert run_episode(scenario, "always_small", Small(), Strong(), specs(), 0.8)[
        "false_completion"
    ]
    assert scenario.initial_state == "failing"


def test_workflow_cost_per_success_includes_failed_episodes():
    success = run_episode(episode(), "always_strong", Small(), Strong(), specs(), 0.8)
    failure = run_episode(episode(), "always_small", Small(), Strong(), specs(), 0.8)
    failure.update(policy="always_strong", episode_id="ci-2")
    report = summarize_episodes([success, failure])["always_strong"]
    assert report["completion_rate"] == 0.5
    assert report["reported_cost_per_success_usd"] == pytest.approx(0.51)
    assert report["false_completions"] == 1


def test_workflow_pairing_uses_source_groups():
    left = run_episode(episode(), "always_strong", Small(), Strong(), specs(), 0.8)
    right = run_episode(episode(), "always_small", Small(), Strong(), specs(), 0.8)
    comparison = compare_policies([left, right])
    assert comparison["comparisons"]["always_small"]["completion_delta"]["status"] == "inconclusive"
