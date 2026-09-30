import httpx
import pytest

from s1.backends import HTTPBackend
from s1.errors import BackendResponseError
from s1.evaluation.contracts import ModelSpec
from s1.evaluation.multimodal import PipelineAdapter
from s1.evaluation.workflow import Episode, run_episode, summarize_episodes


def episode(*, with_tool=False):
    observation = {
        "state": {"ready": True},
        "media": [{"type": "audio", "samples": [0.1]}],
        "questions": {
            "action": {
                "type": "choice",
                "instructions": "Choose the next action",
                "criteria": {
                    "retry": "Retry the operation",
                    "complete": "Declare completion",
                    "escalate": "Ask a worker",
                },
            }
        },
    }
    states = {
        "ready": {
            "observation": observation,
            "goal_satisfied": True,
            "transitions": {"retry": {"next_state": "ready"}},
        }
    }
    if with_tool:
        states["waiting"] = {
            "observation": {**observation, "state": {"ready": False}},
            "transitions": {"retry": {"next_state": "ready", "tool_cost_usd": 0.1}},
        }
    return Episode.model_validate(
        {
            "id": "media-workflow",
            "group_id": "recording-source",
            "family": "ui_document",
            "seed": 0,
            "initial_state": "waiting" if with_tool else "ready",
            "states": states,
        }
    )


def specs():
    return [
        ModelSpec(
            id=name,
            adapter="http",
            preprocessing="http_asr",
            cost_per_request_usd=1.0,
            capabilities={"probabilities": "none"},
        )
        for name in ("small", "strong")
    ]


class Decision:
    def __init__(self, cost=0.2, *, malformed=False, error=None):
        self.cost, self.malformed, self.error = cost, malformed, error
        self.calls = 0

    def predict(self, request):
        self.calls += 1
        assert not request.media
        assert request.state["media_observation"] == "Observed task state"
        if self.error:
            raise self.error
        label = "complete" if request.state["state"]["ready"] else "retry"
        return {
            "answers": {} if self.malformed else {"action": {"label": label}},
            "cost_usd": self.cost,
        }


@pytest.fixture
def pipeline_factory():
    clients = []

    def create(decision, response):
        def handler(request):
            if isinstance(response, Exception):
                raise response
            return httpx.Response(200, json=response)

        client = httpx.Client(transport=httpx.MockTransport(handler))
        clients.append(client)
        return PipelineAdapter(decision, HTTPBackend("https://example.invalid/asr", client=client))

    yield create
    for client in clients:
        client.close()


@pytest.mark.parametrize("with_tool, calls, total", [(False, 1, 0.7), (True, 2, 1.5)])
def test_workflow_bills_each_pipeline_stage_and_tool_once(
    pipeline_factory, with_tool, calls, total
):
    model = Decision()
    adapter = pipeline_factory(model, {"text": "Observed task state", "cost_usd": 0.5})
    row = run_episode(episode(with_tool=with_tool), "always_small", adapter, None, specs(), 0.8)
    assert row["success"] and model.calls == calls
    assert row["tool_replay_cost_usd"] == (0.1 if with_tool else 0)
    for call in row["calls"]:
        assert call["reported_cost_by_stage_usd"] == {"preprocessing": 0.5, "decision": 0.2}
        assert call["reported_cost_usd"] == pytest.approx(0.7)
        assert call["pipeline"]["preprocess_reported_cost_usd"] == 0.5
    assert row["reported_total_cost_usd"] == pytest.approx(total)
    summary = summarize_episodes([row])["always_small"]
    assert summary["reported_cost_per_success_usd"] == pytest.approx(total)


def test_cascade_includes_both_models_pipeline_bills(pipeline_factory):
    small, strong = Decision(0.2), Decision(0.6)
    small_adapter = pipeline_factory(small, {"text": "Observed task state", "cost_usd": 0.5})
    strong_adapter = pipeline_factory(strong, {"text": "Observed task state", "cost_usd": 0.3})
    row = run_episode(episode(), "cascade", small_adapter, strong_adapter, specs(), 0.8)
    assert row["success"] and row["fallback_count"] == 1
    assert small.calls == strong.calls == 1
    assert [c["model_id"] for c in row["calls"]] == ["small", "strong"]
    assert row["reported_total_cost_usd"] == pytest.approx(1.6)


@pytest.mark.parametrize("missing_stage", ["preprocessing", "decision"])
def test_missing_stage_bill_makes_total_unknown(pipeline_factory, missing_stage):
    response = {"text": "Observed task state", "cost_usd": 0.5}
    model = Decision(None if missing_stage == "decision" else 0.2)
    if missing_stage == "preprocessing":
        response.pop("cost_usd")
    adapter = pipeline_factory(model, response)
    row = run_episode(episode(), "always_small", adapter, None, specs(), 0.8)
    assert row["success"]
    assert row["calls"][0]["reported_cost_by_stage_usd"][missing_stage] is None
    assert row["calls"][0]["reported_cost_usd"] is None
    assert row["reported_total_cost_usd"] is None
    assert summarize_episodes([row])["always_small"]["reported_cost_per_success_usd"] is None


@pytest.mark.parametrize("cost", [-0.1, True, "0.5"])
def test_invalid_preprocessing_bill_is_unknown_and_blocks_decision(pipeline_factory, cost):
    model = Decision()
    adapter = pipeline_factory(model, {"text": "Observed task state", "cost_usd": cost})
    row = run_episode(episode(), "always_small", adapter, None, specs(), 0.8)
    assert row["outcome"] == "model_error" and model.calls == 0
    assert row["calls"][0]["reported_cost_by_stage_usd"] == {"preprocessing": None}
    assert row["reported_total_cost_usd"] is None


@pytest.mark.parametrize("cost", [-0.1, True, "0.2", float("nan"), float("inf")])
def test_invalid_decision_bill_preserves_preprocessing_and_unknown_total(pipeline_factory, cost):
    adapter = pipeline_factory(Decision(cost), {"text": "Observed task state", "cost_usd": 0.5})
    row = run_episode(episode(), "always_small", adapter, None, specs(), 0.8)
    assert row["outcome"] == "model_error"
    assert row["calls"][0]["reported_cost_by_stage_usd"] == {
        "preprocessing": 0.5,
        "decision": None,
    }
    assert row["reported_total_cost_usd"] is None


def test_failed_preprocessing_has_unknown_bill_without_decision_dispatch(pipeline_factory):
    model = Decision()
    adapter = pipeline_factory(model, RuntimeError("service unavailable"))
    row = run_episode(episode(), "always_small", adapter, None, specs(), 0.8)
    assert row["outcome"] == "model_error" and model.calls == 0
    call = row["calls"][0]
    assert call["reported_cost_by_stage_usd"] == {"preprocessing": None}
    assert call["pipeline"]["preprocess_ms"] >= 0
    assert "decision_ms" not in call["pipeline"]
    assert row["reported_total_cost_usd"] is None


def test_failed_decision_keeps_known_preprocessing_bill_and_unknown_total(pipeline_factory):
    model = Decision(error=BackendResponseError("model failed"))
    adapter = pipeline_factory(model, {"text": "Observed task state", "cost_usd": 0.5})
    row = run_episode(episode(), "always_small", adapter, None, specs(), 0.8)
    assert row["outcome"] == "model_error" and model.calls == 1
    call = row["calls"][0]
    assert call["reported_cost_by_stage_usd"] == {"preprocessing": 0.5, "decision": None}
    assert call["pipeline"]["decision_ms"] >= 0
    assert row["reported_total_cost_usd"] is None


def test_invalid_preprocessing_text_keeps_bill_and_never_calls_decision(pipeline_factory):
    model = Decision()
    adapter = pipeline_factory(model, {"cost_usd": 0.5})
    row = run_episode(episode(), "always_small", adapter, None, specs(), 0.8)
    assert row["outcome"] == "model_error" and model.calls == 0
    assert row["calls"][0]["reported_cost_by_stage_usd"] == {"preprocessing": 0.5}
    assert row["reported_total_cost_usd"] == 0.5


def test_malformed_decision_answer_preserves_both_reported_bills(pipeline_factory):
    adapter = pipeline_factory(
        Decision(malformed=True), {"text": "Observed task state", "cost_usd": 0.5}
    )
    row = run_episode(episode(), "always_small", adapter, None, specs(), 0.8)
    assert row["outcome"] == "model_error"
    assert row["calls"][0]["reported_cost_by_stage_usd"] == {"preprocessing": 0.5, "decision": 0.2}
    assert row["reported_total_cost_usd"] == pytest.approx(0.7)
