"""Synthetic generic callback tests; no model, CUDA, external data or network."""

import copy
import json
import threading
from types import SimpleNamespace

import jev_decision3_jevbench_generic as generic
import pytest
from run_jevbench_public import _native_probs

from s1.evaluation.decision3 import Decision3Backend


class FakeModel:
    def system_one(self, *, state, questions):
        answers = {}
        for name, question in questions.items():
            if question["type"] == "noul":
                answers[name] = {"type": "noul", "noul": 0.7}
            else:
                labels = (
                    [str(i) for i in range(len(question["criteria"]))]
                    if question["type"] == "score"
                    else list(question["criteria"])
                )
                answers[name] = {
                    "type": question["type"],
                    "probabilities": {label: 1 / len(labels) for label in labels},
                }
                if question["type"] == "choice":
                    answers[name]["choice"] = labels[0]
        return {
            "model": "edge",
            "answers": answers,
            "usage": {"input_tokens": 10, "output_tokens": 0},
        }


class FakeRuntime:
    def prepare(self, **payload):
        return SimpleNamespace(
            sequences={name: [1] for name in payload["questions"]}, lengths={}, errors={}
        )


class FakeSerial:
    def __init__(self, backend, lease, *, torch):
        self.backend, self.lease = backend, lease

    def predict(self, request, **kwargs):
        self.lease.append("acquired")
        response = self.backend.predict(request, **kwargs)
        self.lease.append("fenced")
        return response


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(generic, "SOURCE_ONLY", False)
    base = Decision3Backend.__new__(Decision3Backend)
    base.spec = SimpleNamespace(adapter="decision3", context_limit=16384)
    base.identity = {"model_id": "model/edge"}
    base.runtime, base.model = FakeRuntime(), FakeModel()
    lease, events = [], []
    module = SimpleNamespace(SOURCE_ONLY=False, SerialBackend=FakeSerial)

    def build(**kwargs):
        return generic.GenericJevBenchAdapter(
            base,
            lease_module=module,
            lease=lease,
            torch=object(),
            deadline_factory=kwargs.get("deadline_factory", lambda: lambda: None),
            identity_guard=kwargs.get("identity_guard", lambda *_: {"provider": "original"}),
            event_sink=kwargs.get("event_sink", events.append),
        )

    return SimpleNamespace(base=base, lease=lease, events=events, build=build, module=module)


def request():
    return {
        "state": {"b": 2, "a": 1},
        "questions": {
            "operation": {"type": "noul", "instructions": "Is it true?"},
            "click_target": {
                "type": "choice",
                "instructions": "Pick",
                "criteria": {"b": "B", "a": "A"},
            },
            "score": {"type": "score", "instructions": "Rate", "criteria": ["low", "high"]},
        },
    }


def test_native_three_types_ids_order_source_and_lease_fence(setup):
    adapter = setup.build()
    source = request()
    original = copy.deepcopy(source)
    raw = json.dumps(source, ensure_ascii=False, separators=(",", ":"))
    result = adapter.predict(**source)
    assert source == original
    assert json.dumps(source, ensure_ascii=False, separators=(",", ":")) == raw
    assert list(result["answers"]) == list(source["questions"])
    assert result["answers"]["operation"]["noul"] == 0.7
    assert result["answers"]["click_target"]["choice"] == "b"
    assert result["answers"]["score"]["probabilities"] == {"0": 0.5, "1": 0.5}
    assert setup.lease == ["acquired", "fenced"]
    assert [event["event"] for event in setup.events] == [
        "planned",
        "native_preflight",
        "native_raw_returned",
        "prepared_not_delivered",
    ]
    assert all(event["delta_applied"] == [] for event in setup.events)
    assert all(event["delta_bypassed"] == ["operation", "click_target"] for event in setup.events)
    assert setup.events[0]["request"] == original


def test_coherence_named_answer_shape_and_repeated_calls(setup):
    adapter = setup.build()
    source = request()
    assert list(adapter.answers(**source)) == list(source["questions"])
    assert list(adapter.answers(**source)) == list(source["questions"])
    assert setup.lease == ["acquired", "fenced", "acquired", "fenced"]
    assert [event["call"] for event in setup.events if event["event"] == "planned"] == [1, 2]


@pytest.mark.parametrize("stage", ["provider", "prepared"])
def test_overlapping_call_latches_failure_and_blocks_inflight_delivery(setup, stage):
    entered, resume = threading.Event(), threading.Event()
    original_forward = setup.base.model.system_one

    def wait_for_overlap():
        entered.set()
        if not resume.wait(timeout=5):
            raise TimeoutError("synthetic overlap timed out")

    def forward(**kwargs):
        if stage == "provider":
            wait_for_overlap()
        return original_forward(**kwargs)

    def sink(event):
        setup.events.append(event)
        if stage == "prepared" and event["event"] == "prepared_not_delivered":
            wait_for_overlap()

    setup.base.model.system_one = forward
    adapter = setup.build(event_sink=sink)
    outcomes = []

    def first_call():
        try:
            outcomes.append(adapter.predict(**request()))
        except BaseException as error:
            outcomes.append(error)

    worker = threading.Thread(target=first_call)
    worker.start()
    try:
        assert entered.wait(timeout=5)
        with pytest.raises(ValueError, match="cannot run concurrently"):
            adapter.predict(**request())
    finally:
        resume.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert len(outcomes) == 1
    assert isinstance(outcomes[0], ValueError)
    assert "failed during call" in str(outcomes[0])
    assert setup.events[-1]["event"] == "failed"
    assert len([event for event in setup.events if event["event"] == "native_preflight"]) == 1
    with pytest.raises(ValueError, match="already failed"):
        adapter.predict(**request())


@pytest.mark.parametrize("kind", ["choice", "noul", "score"])
def test_heaven_scorer_mapping_accepts_original_decision_id(setup, kind):
    questions = {
        "choice": {"type": "choice", "instructions": "Pick", "criteria": {"a": "A", "b": "B"}},
        "noul": {"type": "noul", "instructions": "Is it true?"},
        "score": {"type": "score", "instructions": "Rate", "criteria": ["low", "high"]},
    }
    response = setup.build().predict("state", {"decision": questions[kind]})
    task = SimpleNamespace(question={"type": kind}, labels=["a", "b"])
    probabilities = _native_probs(response, task)
    assert probabilities == (
        {"yes": 0.7, "no": pytest.approx(0.3)}
        if kind == "noul"
        else {"a": 0.5, "b": 0.5}
        if kind == "choice"
        else {"0": 0.5, "1": 0.5}
    )


def test_gate_and_non_native_wrapper_refused(setup, monkeypatch):
    monkeypatch.setattr(generic, "SOURCE_ONLY", True)
    with pytest.raises(ValueError, match="review"):
        setup.build()
    monkeypatch.setattr(generic, "SOURCE_ONLY", False)
    with pytest.raises(ValueError, match="original native"):
        generic.GenericJevBenchAdapter(
            SimpleNamespace(spec=setup.base.spec),
            lease_module=setup.module,
            lease=setup.lease,
            torch=object(),
            deadline_factory=lambda: lambda: None,
            identity_guard=lambda *_: {},
            event_sink=lambda _: None,
        )


def test_malformed_input_records_failure_without_forward(setup):
    adapter = setup.build()
    source = request()
    source["questions"]["score"]["criteria"] = ["one"]
    with pytest.raises(ValueError):
        adapter.predict(**source)
    assert not setup.lease
    assert setup.events[-1]["event"] == "failed"
    assert setup.events[-1]["delta_applied"] == []
    with pytest.raises(ValueError, match="already failed"):
        adapter.predict(**request())


@pytest.mark.parametrize(
    ("state", "questions"),
    [
        (None, {"q": {"type": "noul", "instructions": "allowed?"}}),
        (["one", "two"], {"q": {"type": "noul", "instructions": "allowed?"}}),
        ("state", {"q": {"type": "noul", "instructions": "allowed?", "criteria": {}}}),
        (
            "state",
            {"": {"type": "choice", "instructions": "choose", "criteria": {"a": "A", "b": "B"}}},
        ),
        ("state", {"q": {"type": "score", "instructions": "rate", "criteria": ["", "high"]}}),
    ],
)
def test_wider_official_schema_cases_fail_without_forward(setup, state, questions):
    original = copy.deepcopy((state, questions))
    with pytest.raises(ValueError):
        setup.build().predict(state, questions)
    assert (state, questions) == original
    assert not setup.lease
    assert setup.events[-1]["event"] == "failed"


def test_original_criteria_preserved_separately_from_native_serialization(setup):
    questions = {
        "operation": {"type": "choice", "instructions": "choose", "criteria": ["A", "B"]},
        "score": {"type": "score", "instructions": "rate", "criteria": {"2": "low", "4": "high"}},
    }
    original = copy.deepcopy(questions)
    result = setup.build().predict("state", questions)
    native = setup.events[1]["evidence"]["native_payload"]["questions"]
    assert questions == original == setup.events[0]["request"]["questions"]
    assert native["operation"]["criteria"] == {"A": "A", "B": "B"}
    assert native["score"]["criteria"] == ["low", "high"]
    assert result["answers"]["score"]["probabilities"] == {"2": 0.5, "4": 0.5}


def test_mutated_identity_is_rejected_after_fence(setup):
    identity = {"nested": {"pid": 1}}
    calls = []

    def guard(*_args):
        calls.append(1)
        if len(calls) == 2:
            identity["nested"]["pid"] = True
        return identity

    with pytest.raises(ValueError, match="identity changed"):
        setup.build(identity_guard=guard).predict(**request())
    assert setup.lease == ["acquired", "fenced"]
    assert setup.events[-1]["event"] == "failed"
    assert not any(e["event"] == "prepared_not_delivered" for e in setup.events)


def test_sink_failure_latches_and_preserves_primary(setup):
    error = RuntimeError("sink failed")

    def fail(_event):
        raise error

    adapter = setup.build(event_sink=fail)
    with pytest.raises(RuntimeError) as caught:
        adapter.predict(**request())
    assert caught.value is error
    assert not setup.lease


def test_native_provider_failure_latches_and_does_not_claim_fence(setup):
    error = RuntimeError("provider failed")

    def fail(**_kwargs):
        raise error

    setup.base.model.system_one = fail
    adapter = setup.build()
    with pytest.raises(RuntimeError) as caught:
        adapter.predict(**request())
    assert caught.value is error
    assert setup.lease == ["acquired"]
    assert setup.events[-1]["event"] == "failed"
    assert not any(event["event"] == "prepared_not_delivered" for event in setup.events)


def test_mismatched_native_response_evidence_refused(setup, monkeypatch):
    original = setup.base.predict

    def tamper(*args, **kwargs):
        result = original(*args, **kwargs)
        result["decision3_provider_evidence"]["raw_response"]["model"] = "other"
        return result

    monkeypatch.setattr(setup.base, "predict", tamper)
    with pytest.raises(ValueError, match="evidence differs"):
        setup.build().predict(**request())
    assert setup.lease == ["acquired", "fenced"]
    assert setup.events[-1]["event"] == "failed"


def test_modified_normalized_answer_rejected(setup, monkeypatch):
    original = setup.base.predict

    def tamper(*args, **kwargs):
        result = original(*args, **kwargs)
        result["answers"]["operation"]["noul"] = 0.1
        return result

    monkeypatch.setattr(setup.base, "predict", tamper)
    with pytest.raises(ValueError, match="normalized response differs"):
        setup.build().predict(**request())
    assert setup.lease == ["acquired", "fenced"]
    assert setup.events[-1]["event"] == "failed"


def test_deadline_after_preparation_is_failure_not_delivery(setup):
    expired = [False]

    def sink(event):
        setup.events.append(event)
        if event["event"] == "prepared_not_delivered":
            expired[0] = True

    def deadline():
        if expired[0]:
            raise TimeoutError("expired")

    with pytest.raises(TimeoutError, match="expired"):
        setup.build(event_sink=sink, deadline_factory=lambda: deadline).predict(**request())
    assert setup.events[-1]["event"] == "failed"
