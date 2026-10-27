"""Synthetic candidate browser binding tests; no model or GPU is started."""

import copy
import hashlib
import http.client as http_client
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import jev_decision3_candidate_http as http
import jev_decision3_semantic_browser_binding as binding
import jev_decision3_semantic_operation_audit as audit
import pytest
from test_jev_decision3_native import FakeNative, payload

from s1.evaluation.decision3 import native_payload


@pytest.fixture
def context(monkeypatch):
    monkeypatch.setattr(binding, "SOURCE_ONLY", False)
    monkeypatch.setattr(http, "SOURCE_ONLY", False)
    monkeypatch.setattr(audit, "SOURCE_ONLY", False)
    pins = {"operation": "a" * 64, "click_target": "b" * 64}
    names = ["CLICK", "DONE", "SELECT", "TYPE_TEXT"]
    names_sha = hashlib.sha256(binding._encode(names)).hexdigest()
    artifacts, receipts, events, bindings = {}, [], [], []
    behavior = {"mismatch": None, "listener_closed": True, "adapter_closed": False}
    base = FakeNative()

    class Runtime:
        pass

    base.runtime = Runtime()

    class Lease:
        def __init__(self):
            self.fd = None
            self.held = None
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
        def __init__(
            self,
            provider,
            _deltas,
            sha,
            _torch,
            *,
            operation_names,
            operation_names_sha256,
            identity_guard,
            witness_sink,
        ):
            assert identity_guard(provider, {})
            self.base, self.spec, self.name = provider, provider.spec, provider.name
            self.runtime = provider.runtime
            self.pins, self.witness_sink = dict(sha), witness_sink
            self.operation_names = tuple(operation_names)
            self.operation_names_sha256 = operation_names_sha256

        def predict(self, request, *, deadline, evidence_sink=None):
            raw = self.base.predict(request, deadline=deadline, evidence_sink=evidence_sink)
            current_native = native_payload(request)
            self.witness_sink(
                native_payload=current_native,
                composed_raw=copy.deepcopy(raw),
                delta_sha256=self.pins,
                operation_names=list(self.operation_names),
                operation_names_sha256=self.operation_names_sha256,
                projection={"operation_names": list(self.operation_names)},
            )
            return {
                **raw,
                "decision3_provider_evidence": {
                    "native_payload": current_native,
                    "raw_response": copy.deepcopy(raw),
                    "composition": {
                        "operation_names_sha256": self.operation_names_sha256,
                        "delta_sha256": self.pins,
                        "projection": {"operation_names": list(self.operation_names)},
                    },
                },
            }

        def close(self):
            behavior["adapter_closed"] = True

    module = SimpleNamespace(SOURCE_ONLY=False, ForwardLease=Lease, SerialBackend=Serial)
    semantic_module = SimpleNamespace(SOURCE_ONLY=False, SemanticDualHeadDecision3Backend=Candidate)
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
        if behavior["mismatch"] == "vocab":
            metadata["operation_names"].reverse()
        if behavior["mismatch"] == "vocab_sha":
            metadata["operation_names_sha256"] = "0" * 64
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
        return binding.SemanticBrowserBinding(
            base,
            {"operation": object(), "click_target": object()},
            pins,
            object(),
            lease_module=module,
            lease=lease,
            http_module=http,
            adapter_module=semantic_module,
            audit_module=audit,
            operation_names=names,
            operation_names_sha256=names_sha,
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


def test_source_gate_and_released_lease_required(context, monkeypatch):
    monkeypatch.setattr(binding, "SOURCE_ONLY", True)
    with pytest.raises(ValueError, match="independent review"):
        context.make()
    monkeypatch.setattr(binding, "SOURCE_ONLY", False)
    context.module.SOURCE_ONLY = True
    with pytest.raises(ValueError, match="released lease"):
        context.make()


def test_sixteen_calls_bind_witness_raw_request_and_response(context):
    instance = context.make()
    for index in range(16):
        request = payload(2)
        request["state"] = {"sequence": index}
        body = json.dumps(request, ensure_ascii=False, indent=2).encode()
        response, row = instance.service.handle(body, started=time.monotonic())
        parsed = json.loads(response)
        assert list(parsed["answers"]) == list(request["questions"])
        assert row["http_body_sha256"] == hashlib.sha256(body).hexdigest()
        assert row["response_sha256"] == hashlib.sha256(response).hexdigest()
        assert row["witness_receipt"] == context.receipts[index]
        assert context.bindings[index]["original_request"] == request
        assert context.bindings[index]["call"] == index + 1
        assert context.bindings[index]["named_response"] == parsed
        assert context.bindings[index]["witness_receipt"] == context.receipts[index]
        assert context.bindings[index]["http_body_sha256"] == row["http_body_sha256"]
        assert context.bindings[index]["response_sha256"] == row["response_sha256"]
        assert (
            binding.verify_client_exchange(context.bindings[index], body, response)["status"]
            == "provided_bytes_match"
        )
        if index == 0:
            with pytest.raises(ValueError, match="client byte hashes"):
                binding.verify_client_exchange(
                    context.bindings[index],
                    json.dumps(request, separators=(",", ":")).encode(),
                    response,
                )
            with pytest.raises(ValueError, match="client byte hashes"):
                binding.verify_client_exchange(context.bindings[index], body, response + b" ")
        instance.service.delivery(row, delivered=True)
    assert len(context.base.calls) == len(context.receipts) == 16
    assert len({item["directory"] for item in context.receipts}) == 16
    with pytest.raises(ValueError, match="request limit"):
        instance.service.handle(json.dumps(payload(2)).encode(), started=time.monotonic())
    instance.close()
    assert context.behavior["adapter_closed"] and instance.closed


@pytest.mark.parametrize("mismatch", ["native", "raw", "delta", "bytes", "vocab", "vocab_sha"])
def test_inconsistent_witness_never_prepares_response(context, mismatch):
    instance = context.make()
    context.behavior["mismatch"] = mismatch
    with pytest.raises(ValueError):
        instance.service.handle(json.dumps(payload(2)).encode(), started=time.monotonic())
    assert instance.service.failed
    assert len(context.receipts) == 1
    assert not any(event["status"] == "prepared" for event in context.events)
    instance.close()


def test_unknown_operation_is_never_delivered_even_if_injected_provider_is_wrong(context):
    instance = context.make()
    request = payload(2)
    request["questions"]["operation"]["criteria"]["SCROLL_DOWN"] = "scroll"
    with pytest.raises(ValueError, match="unknown semantic operation"):
        instance.service.handle(json.dumps(request).encode(), started=time.monotonic())
    assert instance.service.failed
    assert not context.bindings
    instance.close()


def test_close_refuses_a_held_lease_even_when_fd_is_absent(context):
    instance = context.make()
    context.lease.held = object()
    with pytest.raises(ValueError, match="lease still held"):
        instance.close()
    assert not context.behavior["adapter_closed"]
    context.lease.held = None
    instance.close()


def test_close_refuses_active_or_unconfirmed_resources(context):
    instance = context.make()
    assert instance.service.lock.acquire(blocking=False)
    try:
        with pytest.raises(ValueError, match="request or delivery still active"):
            instance.close()
    finally:
        instance.service.lock.release()
    context.behavior["listener_closed"] = False
    with pytest.raises(ValueError, match="listener closure"):
        instance.close()
    assert not context.behavior["adapter_closed"]
    context.behavior["listener_closed"] = True
    context.lease.fd = 42
    with pytest.raises(ValueError, match="lease still held"):
        instance.close()
    context.lease.fd = None
    _response, row = instance.service.handle(
        json.dumps(payload(2)).encode(), started=time.monotonic()
    )
    with pytest.raises(ValueError, match="delivery still pending"):
        instance.close()
    instance.service.delivery(row, delivered=True)
    instance.close()
    assert context.behavior["adapter_closed"]
    with pytest.raises(ValueError, match="open browser binding"):
        instance.close()


def test_close_releases_binding_owned_runtime_reference(context):
    import gc
    import weakref

    runtime = weakref.ref(context.base.runtime)
    instance = context.make()
    instance.close()
    context.base.runtime = None
    gc.collect()
    assert runtime() is None
    assert instance.service.backend is None and instance.wrapper is None


def test_private_artifact_sink_failure_refuses_delivery(context):
    instance = context.make()

    def failed_sink(_record):
        raise OSError("private sink unavailable")

    instance.private_artifact_sink = failed_sink
    with pytest.raises(OSError, match="private sink unavailable"):
        instance.service.handle(json.dumps(payload(2)).encode(), started=time.monotonic())
    assert instance.service.failed and instance.service.pending is None
    assert context.events[-1]["status"] == "failed"
    instance.close()


def test_real_loopback_two_calls_bind_supplied_client_bytes(context):
    instance = context.make(max_requests=2)
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
                    binding.verify_client_exchange(context.bindings[index], body, raw)["status"]
                    == "provided_bytes_match"
                )
            finally:
                conn.close()
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
    assert not thread.is_alive()
    assert len(context.receipts) == len(context.bindings) == 2
    assert [event["status"] for event in context.events] == [
        "prepared",
        "socket_write_returned",
        "prepared",
        "socket_write_returned",
    ]
    instance.close()


pytest_plugins = ("test_jev_decision3_semantic_operation_adapter",)


def test_real_semantic_adapter_and_audit_through_http_binding(
    context, synthetic, monkeypatch, tmp_path
):
    """Exercise the complete invented CPU path; no source or GPU provenance is implied."""
    import jev_decision3_semantic_operation_adapter as semantic
    import numpy as np
    from test_jev_decision3_operation_gpu import RequestDTO, Tensor, Torch
    from test_jev_decision3_semantic_operation_adapter import fake_sha

    import s1.jev_decision3_native as native

    _prior, provider, deltas, hashes, _names, _names_sha, _witnesses = synthetic
    monkeypatch.setattr(semantic, "Decision3NativeRequest", native.Decision3NativeRequest)
    monkeypatch.setattr(semantic, "native_payload", native_payload)
    monkeypatch.setattr(native, "configuration_blockers", lambda _spec: [])
    monkeypatch.setattr(audit, "WIDTH", 2)
    monkeypatch.setattr(audit, "CODES", 3)
    monkeypatch.setattr(audit, "W0_SHA256", fake_sha(provider.runtime.readout))
    monkeypatch.setattr(
        Tensor, "numpy", lambda self: np.asarray(self.values, dtype=np.float32), raising=False
    )
    original_predict = provider.predict

    def predict(dto, **kwargs):
        return original_predict(RequestDTO.model_validate(dto.model_dump()), **kwargs)

    provider.predict = predict
    names = ["CLICK", "TYPE_TEXT"]
    names_sha = hashlib.sha256(binding._encode(names)).hexdigest()
    sink, receipts = audit.make_witness_writer(tmp_path / "invented-witness", maximum_calls=1)
    records, events = [], []
    instance = binding.SemanticBrowserBinding(
        provider,
        deltas,
        hashes,
        Torch,
        lease_module=context.module,
        lease=context.lease,
        http_module=http,
        adapter_module=semantic,
        audit_module=audit,
        operation_names=names,
        operation_names_sha256=names_sha,
        model_identity="native-test",
        token="t" * 32,
        max_requests=1,
        identity_guard=lambda *_: {"invented": "stable"},
        owner_guard=lambda *_: {"invented": "owner"},
        witness_sink=sink,
        witness_snapshot=lambda: copy.deepcopy(receipts),
        read_witness=lambda receipt: (Path(receipt["directory"]) / "witness.json").read_bytes(),
        private_artifact_sink=records.append,
        event_sink=events.append,
        closure_guard=lambda: True,
    )
    request = payload(2)
    request["questions"]["operation"]["criteria"] = {
        key: value
        for key, value in request["questions"]["operation"]["criteria"].items()
        if key in names
    }
    request["questions"] = {
        k: v
        for k, v in request["questions"].items()
        if k in {"operation", "click_target", "type_text_target"}
    }
    body = json.dumps(request).encode()
    response, row = instance.service.handle(body, started=time.monotonic())
    assert (
        binding.verify_client_exchange(records[0], body, response)["status"]
        == "provided_bytes_match"
    )
    receipt = receipts[0]
    result = audit.audit_directory(
        receipt["directory"],
        expected_file_sha256=receipt["manifest_sha256"],
        expected_witness_sha256=receipt["witness_sha256"],
        expected_delta_sha256=hashes,
        expected_names_sha256=names_sha,
    )
    assert result["status"] == "numeric_consistency_only"
    assert provider.runtime.forward_calls == 1
    instance.service.delivery(row, delivered=True)
    instance.close()
