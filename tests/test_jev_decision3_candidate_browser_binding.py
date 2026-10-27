"""Synthetic candidate browser binding tests; no model or GPU is started."""

import copy
import hashlib
import http.client as http_client
import json
import threading
import time
from types import SimpleNamespace

import jev_decision3_candidate_browser_binding as binding
import jev_decision3_candidate_http as http
import jev_decision3_link_navigation_adapter_audit as audit
import pytest
from test_jev_decision3_native import FakeNative, payload

from s1.evaluation.decision3 import native_payload


@pytest.fixture
def candidate_binding_context(monkeypatch):
    monkeypatch.setattr(binding, "SOURCE_ONLY", False)
    monkeypatch.setattr(http, "SOURCE_ONLY", False)
    pins = {"operation": "a" * 64, "click_target": "b" * 64}
    artifacts, receipts, events, bindings = {}, [], [], []
    behavior = {"mismatch": None, "listener_closed": True, "adapter_closed": False}
    base = FakeNative()

    class Runtime:
        pass

    base.runtime = Runtime()

    class Lease:
        def __init__(self):
            self.fd = None
            self.bad = False

        def check(self):
            if self.bad:
                raise RuntimeError("lease failed")

    class Serial:
        def __init__(self, backend, lease, *, torch):
            self.backend, self.lease, self.torch = backend, lease, torch
            self.spec, self.name = backend.spec, backend.name

        def predict(self, request, **kwargs):
            self.lease.check()
            return self.backend.predict(request, **kwargs)

    class Candidate:
        def __init__(self, provider, _deltas, sha, _torch, *, identity_guard, witness_sink):
            assert identity_guard(provider, {})
            self.base, self.spec, self.name = provider, provider.spec, provider.name
            self.runtime = provider.runtime
            self.pins, self.witness_sink = dict(sha), witness_sink

        def predict(self, request, *, deadline, evidence_sink=None):
            raw = self.base.predict(request, deadline=deadline, evidence_sink=evidence_sink)
            current_native = native_payload(request)
            self.witness_sink(
                native_payload=current_native,
                composed_raw=copy.deepcopy(raw),
                delta_sha256=self.pins,
            )
            return {
                **raw,
                "decision3_provider_evidence": {
                    "native_payload": current_native,
                    "raw_response": copy.deepcopy(raw),
                },
            }

        def close(self):
            behavior["adapter_closed"] = True

    monkeypatch.setattr(binding.adapter, "DualHeadDecision3Backend", Candidate)
    module = SimpleNamespace(SOURCE_ONLY=False, ForwardLease=Lease, SerialBackend=Serial)
    lease = Lease()

    def sink(**metadata):
        metadata = copy.deepcopy(metadata)
        if behavior["mismatch"] == "native":
            metadata["native_payload"]["questions"] = dict(
                reversed(list(metadata["native_payload"]["questions"].items()))
            )
        if behavior["mismatch"] == "raw":
            metadata["composed_raw"]["model"] = "other"
        if behavior["mismatch"] == "delta":
            metadata["delta_sha256"]["operation"] = "0" * 64
        witness_sha = "c" * 64
        document = {
            "schema": audit.FILE_SCHEMA,
            "witness_sha256": witness_sha,
            "metadata": metadata,
            "arrays": {},
        }
        raw = json.dumps(
            document, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
        directory = f"/synthetic/call-{len(receipts):04d}"
        artifacts[directory] = raw
        receipts.append(
            {
                "directory": directory,
                "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                "witness_sha256": witness_sha,
                "delta_sha256": dict(pins),
            }
        )

    def read(receipt):
        raw = artifacts[receipt["directory"]]
        return raw + b" " if behavior["mismatch"] == "bytes" else raw

    def make(max_requests=16):
        return binding.CandidateBrowserBinding(
            base,
            {"operation": object(), "click_target": object()},
            pins,
            object(),
            lease_module=module,
            lease=lease,
            http_module=http,
            model_identity="native-test",
            token="t" * 32,
            max_requests=max_requests,
            identity_guard=lambda provider, _deltas: {"base": provider is base},
            owner_guard=lambda backend, _phase: {"serial": backend.lease is lease},
            witness_sink=sink,
            witness_snapshot=lambda: copy.deepcopy(receipts),
            read_witness=read,
            private_artifact_sink=bindings.append,
            event_sink=events.append,
            closure_guard=lambda: behavior["listener_closed"],
        )

    return SimpleNamespace(
        make=make,
        lease=lease,
        module=module,
        base=base,
        receipts=receipts,
        events=events,
        bindings=bindings,
        artifacts=artifacts,
        behavior=behavior,
        pins=pins,
    )


def test_source_gate_and_released_lease_required(candidate_binding_context, monkeypatch):
    monkeypatch.setattr(binding, "SOURCE_ONLY", True)
    with pytest.raises(ValueError, match="independent review"):
        candidate_binding_context.make()
    monkeypatch.setattr(binding, "SOURCE_ONLY", False)
    candidate_binding_context.module.SOURCE_ONLY = True
    with pytest.raises(ValueError, match="released lease"):
        candidate_binding_context.make()


def test_sixteen_calls_bind_witness_raw_request_and_response(candidate_binding_context):
    instance = candidate_binding_context.make()
    for index in range(16):
        request = payload(2)
        request["state"] = {"sequence": index}
        body = json.dumps(request, ensure_ascii=False, indent=2).encode()
        response, row = instance.service.handle(body, started=time.monotonic())
        parsed = json.loads(response)
        assert list(parsed["answers"]) == list(request["questions"])
        assert row["http_body_sha256"] == hashlib.sha256(body).hexdigest()
        assert row["response_sha256"] == hashlib.sha256(response).hexdigest()
        assert row["witness_receipt"] == candidate_binding_context.receipts[index]
        assert candidate_binding_context.bindings[index]["original_request"] == request
        assert candidate_binding_context.bindings[index]["call"] == index + 1
        assert candidate_binding_context.bindings[index]["named_response"] == parsed
        assert (
            candidate_binding_context.bindings[index]["witness_receipt"]
            == candidate_binding_context.receipts[index]
        )
        assert (
            candidate_binding_context.bindings[index]["http_body_sha256"] == row["http_body_sha256"]
        )
        assert (
            candidate_binding_context.bindings[index]["response_sha256"] == row["response_sha256"]
        )
        assert (
            binding.verify_client_exchange(
                candidate_binding_context.bindings[index], body, response
            )["status"]
            == "provided_bytes_match"
        )
        if index == 0:
            with pytest.raises(ValueError, match="client byte hashes"):
                binding.verify_client_exchange(
                    candidate_binding_context.bindings[index],
                    json.dumps(request, separators=(",", ":")).encode(),
                    response,
                )
            with pytest.raises(ValueError, match="client byte hashes"):
                binding.verify_client_exchange(
                    candidate_binding_context.bindings[index], body, response + b" "
                )
        instance.service.delivery(row, delivered=True)
    assert (
        len(candidate_binding_context.base.calls) == len(candidate_binding_context.receipts) == 16
    )
    assert len({item["directory"] for item in candidate_binding_context.receipts}) == 16
    with pytest.raises(ValueError, match="request limit"):
        instance.service.handle(json.dumps(payload(2)).encode(), started=time.monotonic())
    instance.close()
    assert candidate_binding_context.behavior["adapter_closed"] and instance.closed


@pytest.mark.parametrize("mismatch", ["native", "raw", "delta", "bytes"])
def test_inconsistent_witness_never_prepares_response(candidate_binding_context, mismatch):
    instance = candidate_binding_context.make()
    candidate_binding_context.behavior["mismatch"] = mismatch
    with pytest.raises(ValueError):
        instance.service.handle(json.dumps(payload(2)).encode(), started=time.monotonic())
    assert instance.service.failed
    assert len(candidate_binding_context.receipts) == 1
    assert not any(event["status"] == "prepared" for event in candidate_binding_context.events)
    instance.close()


def test_close_refuses_active_or_unconfirmed_resources(candidate_binding_context):
    instance = candidate_binding_context.make()
    assert instance.service.lock.acquire(blocking=False)
    try:
        with pytest.raises(ValueError, match="request or delivery still active"):
            instance.close()
    finally:
        instance.service.lock.release()
    candidate_binding_context.behavior["listener_closed"] = False
    with pytest.raises(ValueError, match="listener closure"):
        instance.close()
    assert not candidate_binding_context.behavior["adapter_closed"]
    candidate_binding_context.behavior["listener_closed"] = True
    candidate_binding_context.lease.fd = 42
    with pytest.raises(ValueError, match="lease still held"):
        instance.close()
    candidate_binding_context.lease.fd = None
    _response, row = instance.service.handle(
        json.dumps(payload(2)).encode(), started=time.monotonic()
    )
    with pytest.raises(ValueError, match="delivery still pending"):
        instance.close()
    instance.service.delivery(row, delivered=True)
    instance.close()
    assert candidate_binding_context.behavior["adapter_closed"]
    with pytest.raises(ValueError, match="open browser binding"):
        instance.close()


def test_close_releases_binding_owned_runtime_reference(candidate_binding_context):
    import gc
    import weakref

    runtime = weakref.ref(candidate_binding_context.base.runtime)
    instance = candidate_binding_context.make()
    instance.close()
    candidate_binding_context.base.runtime = None
    gc.collect()
    assert runtime() is None
    assert instance.service.backend is None and instance.wrapper is None


def test_private_artifact_sink_failure_refuses_delivery(candidate_binding_context):
    instance = candidate_binding_context.make()

    def failed_sink(_record):
        raise OSError("private sink unavailable")

    instance.private_artifact_sink = failed_sink
    with pytest.raises(OSError, match="private sink unavailable"):
        instance.service.handle(json.dumps(payload(2)).encode(), started=time.monotonic())
    assert instance.service.failed and instance.service.pending is None
    assert candidate_binding_context.events[-1]["status"] == "failed"
    instance.close()


def test_real_loopback_two_calls_bind_supplied_client_bytes(candidate_binding_context):
    instance = candidate_binding_context.make(max_requests=2)
    server = http.loopback_server(instance.service)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        for index in range(2):
            request = payload(2)
            request["state"] = {"sequence": index}
            body = json.dumps(request, ensure_ascii=False, indent=2).encode()
            conn = http_client.HTTPConnection(*server.server_address, timeout=3)
            try:
                conn.request(
                    "POST",
                    "/v1/systemone",
                    body=body,
                    headers={"Authorization": "Bearer " + "t" * 32},
                )
                response = conn.getresponse()
                raw = response.read()
                assert response.status == 200
                assert (
                    binding.verify_client_exchange(
                        candidate_binding_context.bindings[index], body, raw
                    )["status"]
                    == "provided_bytes_match"
                )
            finally:
                conn.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    assert not thread.is_alive()
    assert len(candidate_binding_context.receipts) == len(candidate_binding_context.bindings) == 2
    assert [event["status"] for event in candidate_binding_context.events] == [
        "prepared",
        "socket_write_returned",
        "prepared",
        "socket_write_returned",
    ]
    instance.close()
