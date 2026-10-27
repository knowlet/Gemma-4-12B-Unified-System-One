import copy
import math
import threading
from types import SimpleNamespace

import jev_decision3_cpu_browser_adapter as adapter
import pytest


def request(singleton=False):
    questions = {
        "operation": {
            "type": "choice",
            "instructions": "Choose an action.",
            "criteria": {"DONE": "Finish."} if singleton else {"CLICK": "Open.", "DONE": "Finish."},
        }
    }
    if not singleton:
        questions["click_target"] = {
            "type": "choice",
            "instructions": "Choose a target.",
            "criteria": {"left": "Left link.", "right": "Right link."},
        }
    return {"model": "cpu-test", "state": {"message": "fictional"}, "questions": questions}


class FakeRuntime:
    device = "cpu"
    model_name = "d3-edge"
    batch_size = 8
    fast = None
    permutation_average = False

    def __init__(self):
        self.prepares = []
        self.forwards = []
        self.fault = None

    def prepare(self, **payload):
        self.prepares.append(copy.deepcopy(payload))
        ids = list(payload["questions"])
        if self.fault == "prepare_mutation":
            payload["state"]["injected"] = True
        questions = {
            key: SimpleNamespace(kind="choice", keys=list(payload["questions"][key]["criteria"]))
            for key in ids
        }
        prepared = SimpleNamespace(
            keys=ids,
            questions=questions,
            sequences={key: [1, 2, 3] for key in ids},
            errors={},
            images=[],
            videos=[],
            reversed_sequences={},
        )
        if self.fault == "preflight_missing":
            prepared.sequences.pop(ids[0])
        elif self.fault == "preflight_long":
            prepared.sequences[ids[0]] = [1] * (adapter.CONTEXT_LIMIT + 1)
        elif self.fault == "preflight_error":
            prepared.errors[ids[0]] = {"error": "invalid_question"}
        elif self.fault == "preflight_options":
            prepared.questions[ids[0]].keys.reverse()
        return prepared

    def system_one(self, **payload):
        self.forwards.append(copy.deepcopy(payload))
        if self.fault == "forward_mutation":
            payload["state"]["injected"] = True
        answers = {}
        for key, question in payload["questions"].items():
            labels = list(question["criteria"])
            values = [1.0] if len(labels) == 1 else [0.75, 0.25]
            confidence = (
                1.0
                if len(values) == 1
                else 1 + sum(x * math.log(x) for x in values) / math.log(len(values))
            )
            answers[key] = {
                "type": "choice",
                "choice": labels[0],
                "probabilities": dict(zip(labels, values, strict=True)),
                "confidence": confidence,
            }
        result = {
            "model": self.model_name,
            "answers": answers,
            "usage": {"input_tokens": len(answers) * 3, "output_tokens": 0},
        }
        if self.fault == "missing_head":
            result["answers"].pop(next(iter(result["answers"])))
        elif self.fault == "reorder_head":
            result["answers"] = dict(reversed(list(result["answers"].items())))
        elif self.fault == "wrong_model":
            result["model"] = "different"
        elif self.fault == "bad_usage":
            result["usage"]["input_tokens"] += 1
        elif self.fault == "reorder_options":
            answer = next(iter(result["answers"].values()))
            answer["probabilities"] = dict(reversed(list(answer["probabilities"].items())))
        elif self.fault == "bad_choice":
            result["answers"]["operation"]["choice"] = "DONE"
        elif self.fault == "bad_confidence":
            result["answers"]["operation"]["confidence"] = 1.0
        elif self.fault == "model_mutation":
            self.model_name = "different"
        elif self.fault == "native_error":
            result["answers"]["operation"] = {"type": "choice", "error": "invalid_model_output"}
        return result


@pytest.fixture
def subject(monkeypatch):
    monkeypatch.setattr(adapter, "SOURCE_ONLY", False)
    runtime = FakeRuntime()
    return runtime, adapter.CpuBrowserAdapter(runtime, model_identity="cpu-test", enabled=True)


def test_source_only_rejects_before_runtime_access(monkeypatch):
    monkeypatch.setattr(adapter, "SOURCE_ONLY", True)
    with pytest.raises(ValueError, match="opt-in"):
        adapter.CpuBrowserAdapter(None, model_identity="cpu-test", enabled=True)


def test_complete_response_preserves_every_original_head_and_option(subject):
    runtime, endpoint = subject
    payload = request()
    original = copy.deepcopy(payload)
    response = endpoint.predict(payload)
    assert payload == original
    assert list(response["answers"]) == ["operation", "click_target"]
    assert list(response["answers"]["click_target"]["probabilities"]) == ["left", "right"]
    assert response["answers"]["operation"]["choice"] == "CLICK"
    assert response["answers"]["click_target"]["choice"] == "left"
    assert response["decision3_native"]["original_request"] == original
    assert response["decision3_native"]["raw_response"]["model"] == "d3-edge"
    assert response["model"] == "cpu-test"
    assert response["decision3_cpu"]["device"] == "cpu"
    assert "no hard cancellation" in response["decision3_cpu"]["timeout"]
    assert len(runtime.prepares) == len(runtime.forwards) == 1
    assert runtime.prepares[0] == runtime.forwards[0]
    assert runtime.forwards[0]["questions"] == {
        key: value for key, value in payload["questions"].items()
    }


def test_singleton_is_really_forwarded(subject):
    runtime, endpoint = subject
    response = endpoint.predict(request(singleton=True))
    assert len(runtime.forwards) == 1
    assert response["answers"]["operation"]["choice"] == "DONE"
    assert response["answers"]["operation"]["probabilities"] == {"DONE": 1.0}


@pytest.mark.parametrize(
    "fault",
    [
        "preflight_missing",
        "preflight_long",
        "preflight_error",
        "preflight_options",
        "prepare_mutation",
        "forward_mutation",
        "missing_head",
        "reorder_head",
        "wrong_model",
        "bad_usage",
        "reorder_options",
        "bad_choice",
        "bad_confidence",
        "model_mutation",
        "native_error",
    ],
)
def test_refuses_invalid_native_data_and_does_not_fabricate_response(subject, fault):
    runtime, endpoint = subject
    runtime.fault = fault
    with pytest.raises(ValueError):
        endpoint.predict(request())
    if fault.startswith("preflight") or fault == "prepare_mutation":
        assert not runtime.forwards


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("device", "cuda:0"),
        ("batch_size", 4),
        ("fast", object()),
        ("permutation_average", True),
        ("model_name", "wrong"),
    ],
)
def test_rejects_wrong_runtime_before_request(monkeypatch, attribute, value):
    monkeypatch.setattr(adapter, "SOURCE_ONLY", False)
    runtime = FakeRuntime()
    setattr(runtime, attribute, value)
    with pytest.raises(ValueError, match="CPU runtime"):
        adapter.CpuBrowserAdapter(runtime, model_identity="cpu-test", enabled=True)
    assert not runtime.prepares and not runtime.forwards


def test_exceeding_deadline_before_forward_is_rejected(monkeypatch):
    monkeypatch.setattr(adapter, "SOURCE_ONLY", False)
    runtime = FakeRuntime()
    clock = iter([0.0, 0.0, 0.0, 21.0])
    endpoint = adapter.CpuBrowserAdapter(
        runtime, model_identity="cpu-test", enabled=True, clock=lambda: next(clock)
    )
    with pytest.raises(adapter.BackendTimeoutError, match="not hard-cancelled"):
        endpoint.predict(request())
    assert len(runtime.prepares) == 1 and not runtime.forwards


def test_overlong_forward_returns_no_response_but_is_not_hard_cancelled(monkeypatch):
    monkeypatch.setattr(adapter, "SOURCE_ONLY", False)
    runtime = FakeRuntime()
    clock = iter([0.0, 0.0, 0.0, 0.0, 21.0])
    endpoint = adapter.CpuBrowserAdapter(
        runtime, model_identity="cpu-test", enabled=True, clock=lambda: next(clock)
    )
    with pytest.raises(adapter.BackendTimeoutError, match="not hard-cancelled"):
        endpoint.predict(request())
    assert len(runtime.forwards) == 1
    with pytest.raises(ValueError, match="adapter failed"):
        endpoint.predict(request())


def test_rejects_concurrent_use(subject):
    runtime, endpoint = subject
    assert endpoint.lock.acquire(blocking=False)
    try:
        with pytest.raises(ValueError, match="exclusive adapter"):
            endpoint.predict(request())
        assert not runtime.prepares
    finally:
        endpoint.lock.release()


def test_singleton_target_is_forwarded_without_placeholder(subject):
    runtime, endpoint = subject
    payload = request()
    payload["questions"]["click_target"]["criteria"] = {"only": "Sole target."}
    response = endpoint.predict(payload)
    assert runtime.forwards[0]["questions"] == payload["questions"]
    assert response["answers"]["click_target"]["choice"] == "only"
    assert response["answers"]["click_target"]["probabilities"] == {"only": 1.0}


def test_singleton_representative_still_rejects_inconsistent_heads(subject):
    runtime, endpoint = subject
    payload = request(singleton=True)
    payload["questions"]["click_target"] = {
        "type": "choice",
        "instructions": "target",
        "criteria": {"a": "one"},
    }
    with pytest.raises(ValueError, match="operation and target"):
        endpoint.predict(payload)
    assert not runtime.prepares and not runtime.forwards


def test_singleton_and_large_head_rejects_above_255_without_pruning(subject):
    runtime, endpoint = subject
    payload = request()
    payload["questions"]["click_target"]["criteria"] = {str(i): "candidate" for i in range(256)}
    payload["questions"]["operation"]["criteria"] = {"CLICK": "open"}
    with pytest.raises(ValueError, match="bounded original options"):
        endpoint.predict(payload)
    assert not runtime.prepares and not runtime.forwards


def test_rejects_invalid_response_and_latches_adapter_failed(subject):
    runtime, endpoint = subject
    runtime.fault = "missing_head"
    with pytest.raises(ValueError):
        endpoint.predict(request())
    runtime.fault = None
    with pytest.raises(ValueError, match="adapter failed"):
        endpoint.predict(request())
    assert len(runtime.forwards) == 1


def test_maximum_original_candidates_remain_complete_without_pruning():
    payload = request()
    payload["questions"]["operation"]["criteria"] = {"CLICK": "open"}
    payload["questions"]["click_target"]["criteria"] = {str(i): "candidate" for i in range(255)}
    parsed, original, _source, _labels = adapter._parse_request(payload, "cpu-test")
    assert original == payload
    assert parsed.questions[1].labels() == [str(i) for i in range(255)]


def test_singleton_label_equal_to_placeholder_is_preserved(subject):
    runtime, endpoint = subject
    payload = request()
    payload["questions"]["click_target"]["criteria"] = {"__cpu_validation_placeholder__": "real"}
    response = endpoint.predict(payload)
    assert runtime.forwards[0]["questions"] == payload["questions"]
    assert response["answers"]["click_target"]["choice"] == "__cpu_validation_placeholder__"


@pytest.mark.parametrize("fault", ["wrong_model", "unknown_field", "media", "nonfinite"])
def test_invalid_singleton_latches_before_forward(subject, fault):
    runtime, endpoint = subject
    payload = request(singleton=True)
    if fault == "wrong_model":
        payload["model"] = "other"
    elif fault == "unknown_field":
        payload["unknown"] = True
    elif fault == "media":
        payload["media"] = ["not-supported"]
    else:
        payload["state"]["invalid"] = float("nan")
    with pytest.raises(ValueError):
        endpoint.predict(payload)
    assert not runtime.forwards
    with pytest.raises(ValueError, match="adapter failed"):
        endpoint.predict(request())
    assert not runtime.forwards


def test_concurrent_call_latches_and_prevents_inflight_success(subject):
    runtime, endpoint = subject
    entered, resume = threading.Event(), threading.Event()
    old = runtime.system_one

    def blocking(**payload):
        entered.set()
        assert resume.wait(timeout=5)
        return old(**payload)

    runtime.system_one = blocking
    outcome = []

    def first():
        try:
            outcome.append(endpoint.predict(request()))
        except BaseException as error:
            outcome.append(error)

    thread = threading.Thread(target=first)
    thread.start()
    try:
        assert entered.wait(timeout=5)
        with pytest.raises(ValueError, match="exclusive adapter"):
            endpoint.predict(request())
    finally:
        resume.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert len(outcome) == 1 and isinstance(outcome[0], ValueError)
    assert "adapter failed" in str(outcome[0])
    assert len(runtime.forwards) == 1
    with pytest.raises(ValueError, match="adapter failed"):
        endpoint.predict(request())
