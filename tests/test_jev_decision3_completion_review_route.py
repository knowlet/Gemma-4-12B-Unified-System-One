"""Synthetic real parser, candidate HTTP and semantic binding; no model or browser."""

import copy
import hashlib
import http.client as http_client
import json
import threading
import time
import types
from pathlib import Path

import jev_browser_done_review_owner as review
import jev_decision3_candidate_http as http
import jev_decision3_completion_review_route as route
import jev_decision3_semantic_browser_binding as binding
import pytest
from test_jev_decision3_native import payload

import s1.jev_decision3_native as native
from s1.errors import RequestValidationError
from s1.jev_decision3_native import parse_native_payload
from s1.jev_generated import _request

pytest_plugins = ("test_jev_decision3_semantic_browser_binding",)


def review_payload():
    return {
        "model": "native-test",
        "state": {
            "page": {"url": "https://example.test", "title": "Example", "text": "Visible"},
            "elements": [],
            "recent_actions": [],
        },
        "questions": {
            route.HEAD: {
                "type": "choice",
                "criteria": dict(route.OPTIONS),
                "instructions": {"goal": "Complete the requested task", "rules": route.RULES},
            }
        },
    }


@pytest.fixture(autouse=True)
def reviewed(monkeypatch):
    monkeypatch.setattr(route, "SOURCE_ONLY", False)


def test_real_parser_reproduces_original_failure_and_disabled_is_exact_passthrough():
    request = review_payload()
    assert route.OPTIONS == review.REVIEW_OPTIONS and route.RULES == review.REVIEW_RULES
    with pytest.raises(RequestValidationError, match="operation and named target"):
        _request(request, "native-test")
    original = native.parse_native_payload
    with route.attach(native, binding, enabled=False):
        assert native.parse_native_payload is binding.parse_native_payload is original
        with pytest.raises(RequestValidationError, match="operation and named target"):
            native.parse_native_payload(request, model_identity="native-test")


def test_enabled_review_uses_original_full_dto_and_preserves_action_parser():
    request = review_payload()
    original = native.parse_native_payload
    action = payload(2)
    expected = original(action, model_identity="native-test")
    with route.attach(native, binding, enabled=True):
        result, raw, source, labels = native.parse_native_payload(
            request, model_identity="native-test"
        )
        assert raw == request and labels == {route.HEAD: list(route.OPTIONS)}
        assert source == json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode()
        assert result.questions[0].id == route.HEAD
        assert native.parse_native_payload(action, model_identity="native-test") == expected
        assert binding.parse_native_payload is native.parse_native_payload
    assert native.parse_native_payload is binding.parse_native_payload is original


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["questions"][route.HEAD]["criteria"].pop("unknown"),
        lambda p: p["questions"][route.HEAD]["instructions"].update(rules="changed"),
        lambda p: p["state"].update(oracle="not allowed"),
        lambda p: p["state"]["elements"].append(float("nan")),
        lambda p: p.update(model="another-model"),
        lambda p: p["questions"].update(operation={"type": "choice", "criteria": {"DONE": "done"}}),
    ],
)
def test_invalid_review_and_mixed_heads_fail_closed(mutate):
    request = review_payload()
    mutate(request)
    with route.attach(native, binding, enabled=True):
        with pytest.raises((RequestValidationError, ValueError)):
            native.parse_native_payload(request, model_identity="native-test")


def test_service_http_review_binds_full_request_witness_and_response(context):
    instance = context.make()
    wrapper = instance.wrapper
    original_sink, original_predict = wrapper.witness_sink, wrapper.predict

    # The fixture's deterministic fake candidate records the heads that a real
    # semantic projection would apply; this is synthetic, not a CUDA proof.
    def projection(metadata):
        heads = list(metadata["native_payload"]["questions"])
        return {
            "applied_head_ids": [h for h in heads if h in ("operation", "click_target")],
            "absent_head_ids": [h for h in ("operation", "click_target") if h not in heads],
        }

    def sink(**metadata):
        metadata["projection"].update(projection(metadata))
        original_sink(**metadata)

    def predict(request, **kwargs):
        result = original_predict(request, **kwargs)
        meta = result["decision3_provider_evidence"]
        meta["composition"]["projection"].update(projection(meta))
        return result

    wrapper.witness_sink, wrapper.predict = sink, predict
    request = review_payload()
    body = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode()
    server = http.loopback_server(instance.service)
    thread = threading.Thread(target=server.serve_forever)
    original = native.parse_native_payload
    try:
        with route.attach(native, binding, enabled=True):
            thread.start()
            conn = http_client.HTTPConnection(*server.server_address, timeout=5)
            conn.request(
                "POST",
                "/v1/systemone",
                body=body,
                headers={"Authorization": "Bearer " + "t" * 32, "Content-Type": "application/json"},
            )
            response = conn.getresponse()
            raw = response.read()
            conn.close()
            assert response.status == 200
            assert list(json.loads(raw)["answers"]) == [route.HEAD]
            assert len(context.bindings) == len(context.receipts) == len(context.base.calls) == 1
            record = context.bindings[0]
            assert record["original_request"] == request
            assert record["http_body_sha256"] == hashlib.sha256(body).hexdigest()
            assert record["response_sha256"] == hashlib.sha256(raw).hexdigest()
            assert (
                binding.verify_client_exchange(record, body, raw)["status"]
                == "provided_bytes_match"
            )
            manifest = json.loads(context.artifacts[context.receipts[0]["directory"]])
            assert manifest["metadata"]["projection"]["applied_head_ids"] == []
            assert manifest["metadata"]["projection"]["absent_head_ids"] == [
                "operation",
                "click_target",
            ]
            action = payload(2)
            body2 = json.dumps(action, ensure_ascii=False, separators=(",", ":")).encode()
            conn = http_client.HTTPConnection(*server.server_address, timeout=5)
            conn.request(
                "POST",
                "/v1/systemone",
                body=body2,
                headers={"Authorization": "Bearer " + "t" * 32, "Content-Type": "application/json"},
            )
            response2 = conn.getresponse()
            assert response2.status == 200
            assert list(json.loads(response2.read())["answers"]) == list(action["questions"])
            conn.close()
            assert context.bindings[1]["original_request"] == action
            server.shutdown()
            thread.join(timeout=5)
            assert not thread.is_alive()
    finally:
        if thread.is_alive():
            server.shutdown()
            thread.join(timeout=5)
        server.server_close()
        instance.close()
    assert native.parse_native_payload is binding.parse_native_payload is original


def test_child_bridge_wraps_only_pinned_owned_service_and_restores():
    child = types.ModuleType("synthetic_child")
    child.__file__ = __file__
    checks = []
    child.common = types.SimpleNamespace(_verify_source_pins=lambda pins: checks.append(dict(pins)))
    child.native = native
    child.binding = binding
    child.request = review_payload()
    exec(
        "def _serve(plan, plan_sha, invocation, backend, torch, modules, output, check):\n"
        "    a = native.parse_native_payload(request, model_identity='native-test')\n"
        "    b = binding.parse_native_payload(request, model_identity='native-test')\n"
        "    return a == b\n",
        child.__dict__,
    )
    original = child._serve
    pins = {route.__file__: "a" * 64, child.__file__: "b" * 64}
    plan = {"source_pins": pins}
    modules = (None, None, binding, None, None, None)
    with route.attach_to_child(child, native, enabled=True):
        assert child._serve(plan, "sha", "invocation", None, None, modules, None, None)
        assert checks == [pins]
        with pytest.raises(ValueError, match="explicit pinned"):
            child._serve({"source_pins": {}}, "sha", "invocation", None, None, modules, None, None)
    assert child._serve is original and native.parse_native_payload is parse_native_payload


@pytest.mark.parametrize("mode", ["raw", "delta", "bytes", "vocab_sha"])
def test_review_witness_tampering_fails_before_response(context, mode):
    instance = context.make()
    context.behavior["mismatch"] = mode
    try:
        with route.attach(native, binding, enabled=True):
            with pytest.raises(ValueError):
                instance.service.handle(
                    json.dumps(review_payload()).encode(), started=time.monotonic()
                )
        assert instance.service.failed and not context.bindings
        assert not any(row["status"] == "prepared" for row in context.events)
    finally:
        instance.close()


def test_review_preserves_original_service_deadline(context):
    instance = context.make()
    instance.service.clock = lambda: 21.0
    try:
        with route.attach(native, binding, enabled=True):
            with pytest.raises(ValueError, match="deadline"):
                instance.service.handle(json.dumps(review_payload()).encode(), started=0.0)
        assert instance.service.failed and not context.bindings and not context.base.calls
    finally:
        instance.close()


def test_review_real_adapter_http_witness_and_fp64_audit(context, synthetic, monkeypatch, tmp_path):
    """One invented review passes both real parsers and the real FP64 auditor."""
    import jev_decision3_semantic_operation_adapter as semantic
    import jev_decision3_semantic_operation_audit as audit
    import numpy as np
    from test_jev_decision3_operation_gpu import RequestDTO, Tensor, Torch
    from test_jev_decision3_semantic_operation_adapter import fake_sha

    from s1.evaluation.decision3 import native_payload

    _, provider, deltas, hashes, _, _, _ = synthetic
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
    sink, receipts = audit.make_witness_writer(tmp_path / "review-witness", maximum_calls=1)
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
    request = review_payload()
    body = json.dumps(request, ensure_ascii=False, separators=(",", ":")).encode()
    server = http.loopback_server(instance.service)
    thread = threading.Thread(target=server.serve_forever)
    try:
        with route.attach(native, binding, enabled=True):
            thread.start()
            conn = http_client.HTTPConnection(*server.server_address, timeout=5)
            conn.request(
                "POST",
                "/v1/systemone",
                body=body,
                headers={"Authorization": "Bearer " + "t" * 32},
            )
            response = conn.getresponse()
            raw = response.read()
            conn.close()
            assert response.status == 200
            assert list(json.loads(raw)["answers"]) == [route.HEAD]
            server.shutdown()
            thread.join(timeout=5)
        assert not thread.is_alive()
        assert (
            binding.verify_client_exchange(records[0], body, raw)["status"]
            == "provided_bytes_match"
        )
        receipt = receipts[0]
        kwargs = {
            "expected_file_sha256": receipt["manifest_sha256"],
            "expected_witness_sha256": receipt["witness_sha256"],
            "expected_delta_sha256": hashes,
            "expected_names_sha256": names_sha,
        }
        result = audit.audit_directory(receipt["directory"], **kwargs)
        assert result["status"] == "numeric_consistency_only"
        assert result["heads"] == [route.HEAD]
        assert result["applied_head_ids"] == []
        assert result["absent_head_ids"] == ["operation", "click_target"]
        assert provider.runtime.forward_calls == 1
        witness_file = Path(receipt["directory"]) / "witness.json"
        witness_file.write_bytes(witness_file.read_bytes() + b" ")
        with pytest.raises(ValueError):
            audit.audit_directory(receipt["directory"], **kwargs)
    finally:
        if thread.is_alive():
            server.shutdown()
            thread.join(timeout=5)
        server.server_close()
        instance.close()
