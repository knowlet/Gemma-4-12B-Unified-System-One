"""Request evidence survives ASGI caller cancellation without model or network access."""

import asyncio
import copy
import hashlib
import json
import re
import threading

import httpx
import pytest

import s1.jev_hierarchy as hierarchy_module
from s1.api import create_app
from s1.contracts import answer_from_probabilities
from s1.errors import (
    BackendResponseError,
    BackendTimeoutError,
    BackendTransportError,
    RequestValidationError,
)

TestClient = pytest.importorskip("fastapi.testclient").TestClient


def wire_payload(target_count=1):
    return {
        "model": "browser-alias",
        "state": {
            "page": {"title": "Observed page", "text": "Complete the visible browser task."},
            "recent_actions": [],
        },
        "questions": {
            "operation": {
                "type": "choice",
                "instructions": {"goal": "Open the observed requested record"},
                "criteria": {"CLICK": "Open record", "DONE": "All requirements confirmed"},
            },
            "click_target": {
                "type": "choice",
                "instructions": {
                    "goal": "Open the observed requested record",
                    "operation": "CLICK",
                },
                "criteria": {
                    f"node-{index:03d}": {"element": f"Observed record {index}", "role": "button"}
                    for index in range(target_count)
                },
            },
        },
    }


class Backend:
    name = "evidence-test-policy"

    def __init__(self):
        self.calls, self.responses = [], []

    def predict(self, request):
        self.calls.append(request.model_copy(deep=True))
        response = {
            "model": self.name,
            "answers": {
                question.id: answer_from_probabilities(
                    question, [1 / len(question.labels())] * len(question.labels())
                )
                for question in request.questions
            },
            "usage": {"input_tokens": 11, "output_tokens": 0},
            "provider_metadata": {"call": len(self.calls)},
        }
        self.responses.append(response)
        return response


class EventLog:
    def __init__(self):
        self.events = []
        self.lock = threading.Lock()
        self.second_started = threading.Event()
        self.terminal = threading.Event()

    def __call__(self, event):
        # Retain the supplied object itself: a later mutation must not rewrite
        # evidence already handed to the sink.
        with self.lock:
            self.events.append(event)
            if sum(item["event"] == "request_started" for item in self.events) == 2:
                self.second_started.set()
            if event["event"] in {"server_returned", "server_failed"}:
                self.terminal.set()

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.events)


def application(backend, sink=None, *, hierarchy=False, batch=False, **kwargs):
    return create_app(
        backend,
        jev_compatible=True,
        jev_hierarchical=hierarchy,
        jev_hierarchical_batch=batch,
        model_identity="browser-alias",
        request_evidence_sink=sink,
        **kwargs,
    )


def events_for(log, request_id):
    return [event for event in log.snapshot() if event["request_id"] == request_id]


def assert_event_envelopes(events):
    assert events
    for event in events:
        assert re.fullmatch(r"[0-9a-f]{32}", event["request_id"])
        assert type(event["sequence"]) is int and event["sequence"] >= 0
        assert type(event["recorded_unix"]) in (int, float) and event["recorded_unix"] > 0
        assert type(event["elapsed_ms"]) in (int, float) and event["elapsed_ms"] >= 0
        assert isinstance(event["event"], str) and isinstance(event["data"], dict)
    sequences = [event["sequence"] for event in events]
    assert sequences == list(range(sequences[0], sequences[0] + len(events)))
    json.dumps(events, allow_nan=False)


def raw_stage_checkpoints(log, call):
    return [
        event
        for event in log.snapshot()
        if event["event"] == "hierarchy_checkpoint"
        and any(
            stage.get("raw_response", {}).get("provider_metadata", {}).get("call") == call
            for stage in event["data"]["audit"].get("stages", [])
        )
    ]


def test_optional_sink_preserves_response_and_emits_detached_ordered_request_evidence():
    payload = wire_payload()
    with TestClient(application(Backend())) as client:
        expected = client.post("/v1/systemone", json=payload)
    log, backend = EventLog(), Backend()
    with TestClient(application(backend, log)) as client:
        response = client.post("/v1/systemone", json=payload)
    assert response.status_code == expected.status_code == 200
    assert response.json() == expected.json()
    events = log.snapshot()
    assert_event_envelopes(events)
    assert len({event["request_id"] for event in events}) == 1
    assert [event["event"] for event in events if event["event"] != "hierarchy_checkpoint"] == [
        "request_started",
        "inference_lock_acquired",
        "server_returned",
    ]
    started = events[0]["data"]
    assert started["payload"] == payload
    assert started["model_identity"] == "browser-alias"
    ordered = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    assert started["original_request_sha256"] == hashlib.sha256(ordered.encode()).hexdigest()
    assert events[-1]["data"]["response"] == response.json()
    backend.responses[0]["provider_metadata"]["call"] = 999
    payload["state"]["page"]["text"] = "Changed after request completion"
    retained = log.snapshot()
    assert (
        retained[0]["data"]["payload"]["state"]["page"]["text"] != payload["state"]["page"]["text"]
    )
    assert retained[-1]["data"]["response"]["provider_metadata"]["call"] == 1


def test_cancelled_async_asgi_caller_does_not_erase_real_worker_return():
    started, release, returned = threading.Event(), threading.Event(), threading.Event()
    log = EventLog()

    class Blocked(Backend):
        def predict(self, request):
            started.set()
            assert release.wait(5), "Test must release the blocked CPU worker"
            result = super().predict(request)
            returned.set()
            return result

    backend = Blocked()
    app = application(backend, log)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://asgi.test") as client:
            task = asyncio.create_task(client.post("/v1/systemone", json=wire_payload()))
            try:
                assert await asyncio.to_thread(started.wait, 5)
                task.cancel()
                await asyncio.sleep(0)
                assert not returned.is_set()
                assert not any(event["event"] == "server_returned" for event in log.snapshot())
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=5)
                assert await asyncio.to_thread(log.terminal.wait, 5)
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    assert returned.is_set() and len(backend.calls) == 1
    events = log.snapshot()
    assert_event_envelopes(events)
    assert events[-1]["event"] == "server_returned"
    assert events[-1]["data"]["response"]["provider_metadata"] == {"call": 1}
    assert events[-1]["data"]["response"]["model"] == backend.name


def test_duplicate_queued_asgi_requests_have_distinct_ids_and_start_before_lock():
    started, release = threading.Event(), threading.Event()
    log = EventLog()

    class BlockFirst(Backend):
        def predict(self, request):
            if not self.calls:
                started.set()
                assert release.wait(5), "Test must release the first worker"
            return super().predict(request)

    backend = BlockFirst()
    app = application(backend, log)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://asgi.test") as client:
            tasks = [asyncio.create_task(client.post("/v1/systemone", json=wire_payload()))]
            try:
                assert await asyncio.to_thread(started.wait, 5)
                tasks.append(asyncio.create_task(client.post("/v1/systemone", json=wire_payload())))
                assert await asyncio.to_thread(log.second_started.wait, 5)
                pending = log.snapshot()
                starts = [event for event in pending if event["event"] == "request_started"]
                assert len(starts) == 2
                assert len({event["request_id"] for event in starts}) == 2
                assert len({event["data"]["original_request_sha256"] for event in starts}) == 1
                assert sum(event["event"] == "inference_lock_acquired" for event in pending) == 1
                release.set()
                responses = await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)
                assert all(response.status_code == 200 for response in responses)
            finally:
                release.set()
                await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(scenario())
    assert len(backend.calls) == 2
    for request_id in {event["request_id"] for event in log.snapshot()}:
        events = events_for(log, request_id)
        assert_event_envelopes(events)
        assert [event["event"] for event in events if event["event"] != "hierarchy_checkpoint"] == [
            "request_started",
            "inference_lock_acquired",
            "server_returned",
        ]


def test_hierarchy_raw_checkpoint_precedes_normalization_and_retains_invalid_output(monkeypatch):
    log = EventLog()

    class Malformed(Backend):
        def predict(self, request):
            result = super().predict(request)
            if len(self.calls) == 3:
                question = request.questions[0]
                result["answers"][question.id]["probabilities"][question.labels()[0]] = "bad"
            return result

    backend = Malformed()
    normalize = hierarchy_module.normalize_response

    def inspected(request, result):
        call = result["provider_metadata"]["call"]
        assert raw_stage_checkpoints(log, call), "Raw stage must reach sink before normalization"
        return normalize(request, result)

    monkeypatch.setattr(hierarchy_module, "normalize_response", inspected)
    with TestClient(application(backend, log, hierarchy=True)) as client:
        response = client.post("/v1/systemone", json=wire_payload(53))
    assert response.status_code == 502
    assert len(backend.calls) == 3
    events = log.snapshot()
    assert_event_envelopes(events)
    failure = events[-1]
    assert failure["event"] == "server_failed"
    audit = failure["data"]["hierarchy"]
    assert audit["status"] == "failed"
    raw = audit["stages"][2]["raw_response"]
    question_id = next(iter(raw["answers"]))
    assert "bad" in raw["answers"][question_id]["probabilities"].values()
    assert audit["stages"][0]["raw_response"]["provider_metadata"] == {"call": 1}
    assert all(
        isinstance(event["data"]["event"], str)
        for event in events
        if event["event"] == "hierarchy_checkpoint"
    )


def test_hierarchy_raw_checkpoint_precedes_cooperative_deadline_error(monkeypatch):
    log, backend = EventLog(), Backend()

    def deadline(_self, _started):
        if backend.responses:
            assert raw_stage_checkpoints(log, 1), "Deadline must not erase returned raw output"
            raise BackendTimeoutError("cooperative deadline reached after real stage return")

    monkeypatch.setattr(hierarchy_module.JevHierarchicalAdapter, "_deadline", deadline)
    with TestClient(application(backend, log, hierarchy=True)) as client:
        response = client.post("/v1/systemone", json=wire_payload(53))
    assert response.status_code == 504
    assert len(backend.calls) == 1
    failed = log.snapshot()[-1]
    assert failed["event"] == "server_failed"
    assert failed["data"]["hierarchy"]["stages"][0]["raw_response"]["provider_metadata"] == {
        "call": 1
    }
    assert failed["data"]["error"]["type"] == "BackendTimeoutError"


def test_batch_raw_rows_are_checkpointed_before_invalid_count_is_rejected():
    log = EventLog()

    class ShortBatch(Backend):
        supports_stage_batching = True

        def predict_batch(self, requests):
            return [self.predict(request) for request in requests][:-1]

    backend = ShortBatch()
    with TestClient(application(backend, log, hierarchy=True, batch=True)) as client:
        response = client.post("/v1/systemone", json=wire_payload(53))
    assert response.status_code == 502
    checkpoints = [
        event
        for event in log.snapshot()
        if event["event"] == "hierarchy_checkpoint"
        and event["data"]["audit"].get("batch_execution", {}).get("raw_responses")
    ]
    assert checkpoints
    raw = checkpoints[0]["data"]["audit"]["batch_execution"]["raw_responses"]
    assert len(raw) == len(backend.calls) - 1
    assert raw[0]["provider_metadata"] == {"call": 1}
    assert log.snapshot()[-1]["event"] == "server_failed"


@pytest.mark.parametrize(
    ("error_type", "status"),
    [
        (RequestValidationError, 422),
        (BackendTimeoutError, 504),
        (BackendResponseError, 502),
        (BackendTransportError, 502),
    ],
)
def test_sink_failures_preserve_typed_http_errors_and_all_attached_evidence(error_type, status):
    primary = error_type("original backend failure")
    primary.generation = {"raw_output": "actual native output"}
    primary.hierarchy = {"status": "failed", "stages": [{"raw_response": {"observed": True}}]}
    primary.batch_evidence = {
        "completed_response_indices": [0],
        "completed_responses": [{"actual": "row"}],
    }
    primary.prefix_evidence = {"status": "failed", "forwards": [{"status": "returned"}]}
    events = []

    class Broken(Backend):
        def predict(self, _request):
            raise primary

    def broken_sink(event):
        events.append(event)
        raise OSError("journal unavailable")

    app = application(Broken(), broken_sink)
    with TestClient(app) as client:
        response = client.post("/v1/systemone", json=wire_payload())
    assert response.status_code == status
    assert events[-1]["event"] == "server_failed"
    data = events[-1]["data"]
    assert data["error"] == {"type": error_type.__name__, "message": str(primary)}
    for key in ("generation", "hierarchy", "batch_evidence", "prefix_evidence"):
        assert data[key] == getattr(primary, key)
    assert app.state.request_evidence_errors
    assert "OSError" in json.dumps(app.state.request_evidence_errors)
    assert "original backend failure" in json.dumps(response.json())


def test_sink_failure_does_not_replace_untyped_internal_primary_exception():
    primary = RuntimeError("actual internal failure")

    class Broken(Backend):
        def predict(self, _request):
            raise primary

    def broken_sink(_event):
        raise OSError("secondary evidence failure")

    app = application(Broken(), broken_sink)
    with TestClient(app) as client:
        with pytest.raises(RuntimeError) as caught:
            client.post("/v1/systemone", json=wire_payload())
    assert caught.value is primary
    assert app.state.request_evidence_errors
    assert "secondary evidence failure" in json.dumps(app.state.request_evidence_errors)


def test_authentication_failure_does_not_expose_payload_to_sink_or_backend():
    backend, log = Backend(), EventLog()
    with TestClient(application(backend, log, api_key="required")) as client:
        response = client.post("/v1/systemone", json=wire_payload())
    assert response.status_code == 401
    assert backend.calls == []
    assert log.snapshot() == []


def test_sink_cannot_mutate_live_request_or_returned_policy_response():
    payload, backend = wire_payload(), Backend()

    def mutating_sink(event):
        if event["event"] == "request_started":
            event["data"]["payload"]["state"]["page"]["text"] = "Sink mutation"
        elif event["event"] == "server_returned":
            event["data"]["response"]["model"] = "Sink replacement model"

    with TestClient(application(backend, mutating_sink)) as client:
        response = client.post("/v1/systemone", json=payload)
    assert response.status_code == 200
    assert backend.calls[0].state == payload["state"]
    assert response.json()["model"] == backend.name


def test_cancelled_asgi_caller_keeps_late_real_hierarchy_failure_and_raw_stage():
    started, release = threading.Event(), threading.Event()
    log = EventLog()
    primary = BackendTransportError("actual failure after caller cancellation")
    primary.prefix_evidence = {"forwards": [{"status": "returned", "row_indices": [0]}]}

    class FailLater(Backend):
        def predict(self, request):
            if not self.calls:
                return super().predict(request)
            started.set()
            assert release.wait(5), "Test must release the blocked failing CPU worker"
            raise primary

    backend = FailLater()
    app = application(backend, log, hierarchy=True)

    async def scenario():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://asgi.test") as client:
            task = asyncio.create_task(client.post("/v1/systemone", json=wire_payload(53)))
            try:
                assert await asyncio.to_thread(started.wait, 5)
                assert raw_stage_checkpoints(log, 1)
                task.cancel()
                await asyncio.sleep(0)
                assert not log.terminal.is_set()
                release.set()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, timeout=5)
                assert await asyncio.to_thread(log.terminal.wait, 5)
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    events = log.snapshot()
    assert_event_envelopes(events)
    failed = events[-1]
    assert failed["event"] == "server_failed"
    assert failed["data"]["error"] == {"type": "BackendTransportError", "message": str(primary)}
    assert failed["data"]["prefix_evidence"] == primary.prefix_evidence
    assert failed["data"]["hierarchy"]["status"] == "failed"
    assert failed["data"]["hierarchy"]["stages"][0]["raw_response"]["provider_metadata"] == {
        "call": 1
    }


@pytest.mark.parametrize("hierarchy", [False, True])
def test_small_invalid_raw_is_journaled_before_compatibility_normalization(monkeypatch, hierarchy):
    import s1.jev as jev_module

    log = EventLog()

    class Invalid(Backend):
        def predict(self, request):
            response = super().predict(request)
            response["answers"]["operation"]["probabilities"]["CLICK"] = "invalid native value"
            return response

    actual = jev_module.normalize_response

    def normalize(request, response):
        raw_events = [
            event
            for event in log.snapshot()
            if event["event"] == "hierarchy_checkpoint"
            and event["data"]["event"] == "compatible_raw_returned"
        ]
        assert len(raw_events) == 1
        assert raw_events[0]["data"]["data"]["raw_response"] == response
        return actual(request, response)

    monkeypatch.setattr(jev_module, "normalize_response", normalize)
    with TestClient(application(Invalid(), log, hierarchy=hierarchy)) as client:
        response = client.post("/v1/systemone", json=wire_payload(2))
    assert response.status_code == 502
    assert log.snapshot()[-1]["event"] == "server_failed"


@pytest.mark.parametrize("observer_error", [OSError, KeyboardInterrupt])
def test_direct_compatible_observer_failure_preserves_primary_and_none_skips_observer(
    monkeypatch, observer_error
):
    from s1.jev import JevCompatibilityAdapter

    primary = RuntimeError("actual native inference failure")

    class Broken(Backend):
        def predict(self, request):
            raise primary

    def broken_sink(_event):
        raise observer_error("actual observer failure")

    adapter = JevCompatibilityAdapter(
        Broken(), model_identity="browser-alias", evidence_sink=broken_sink
    )
    with pytest.raises(RuntimeError) as error:
        adapter.predict(wire_payload(2))
    assert error.value is primary
    assert len(adapter.evidence_sink_errors) == 2

    def unexpected(*_args, **_kwargs):
        raise AssertionError("default adapter invoked an observer")

    monkeypatch.setattr(JevCompatibilityAdapter, "_emit", unexpected)
    result = JevCompatibilityAdapter(Backend(), model_identity="browser-alias").predict(
        wire_payload(2)
    )
    assert result["answers"]["operation"]["choice"] == "CLICK"


def test_batched_checkpoints_keep_ordered_inputs_and_raw_once_without_growing_snapshots():
    log = EventLog()

    class Batch(Backend):
        def predict_batch(self, requests):
            return [self.predict(request) for request in requests]

    backend = Batch()
    app = application(backend, log, hierarchy=True, batch=True)
    with TestClient(app) as client:
        response = client.post("/v1/systemone", json=wire_payload(80))
    assert response.status_code == 200
    checkpoints = [
        event["data"] for event in log.snapshot() if event["event"] == "hierarchy_checkpoint"
    ]
    raw = [event for event in checkpoints if event["event"] == "batch_raw_returned"]
    assert len(raw) == 1
    assert raw[0]["audit"]["batch_execution"]["raw_responses"] == backend.responses
    assert not any(event["event"] == "stage_raw_returned" for event in checkpoints)
    started = next(event for event in checkpoints if event["event"] == "batch_started")
    assert [stage["request"] for stage in started["audit"]["stages"]] == [
        request.model_dump(exclude_none=True) for request in backend.calls
    ]
    for event in checkpoints:
        if event["event"] != "batch_raw_returned":
            assert "raw_responses" not in event["audit"].get("batch_execution", {})
        if event["event"] == "stage_finished":
            assert all(
                "raw_response" not in stage and "response" not in stage and "request" not in stage
                for stage in event["audit"]["stages"]
            )
    final = log.snapshot()[-1]
    assert final["event"] == "server_returned" and final["data"]["response"] == response.json()
    assert len(final["data"]["response"]["answers"]["click_target"]["probabilities"]) == 80
    assert len(app.state.request_evidence_timings) == len(log.snapshot())
    assert sum(row["elapsed_ms"] for row in app.state.request_evidence_timings) > 0


def test_request_identity_hash_preserves_candidate_insertion_order():
    log, original = EventLog(), wire_payload()
    reordered = copy.deepcopy(original)
    reordered["questions"]["operation"]["criteria"] = dict(
        reversed(list(original["questions"]["operation"]["criteria"].items()))
    )
    assert original == reordered  # Mapping equality alone hides answer-letter changes.
    with TestClient(application(Backend(), log)) as client:
        assert client.post("/v1/systemone", json=original).status_code == 200
        assert client.post("/v1/systemone", json=reordered).status_code == 200
    starts = [event for event in log.snapshot() if event["event"] == "request_started"]
    assert len(starts) == 2
    assert starts[0]["request_id"] != starts[1]["request_id"]
    assert (
        starts[0]["data"]["original_request_sha256"] != starts[1]["data"]["original_request_sha256"]
    )


def test_base_exception_is_recorded_before_same_primary_leaves_sync_endpoint():
    class Interrupted(BaseException):
        pass

    primary, log = Interrupted("actual interrupted computation"), EventLog()

    class Broken(Backend):
        def predict(self, _request):
            raise primary

    app = application(Broken(), log)
    endpoint = next(route.endpoint for route in app.routes if route.path == "/v1/systemone")
    with pytest.raises(Interrupted) as caught:
        endpoint(wire_payload())
    assert caught.value is primary
    failed = log.snapshot()[-1]
    assert failed["event"] == "server_failed"
    assert failed["data"]["error"] == {"type": "Interrupted", "message": str(primary)}
