"""Source-only multi-request binding for an externally owned Decision3 worker.

This wires a loaded publisher backend, pinned deltas, an existing shared GPU
lease, the native candidate adapter and a caller-owned private witness store
into the existing loopback HTTP interface. It does not load a model, own or
monitor a process/GPU, authorize a campaign, authenticate callback origins,
perform a CPU numerical audit, or establish browser task success.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re

import jev_decision3_link_navigation_adapter as adapter
import jev_decision3_link_navigation_adapter_audit as audit

from s1.evaluation.decision3 import native_payload
from s1.jev_decision3_native import parse_native_payload

SOURCE_ONLY = True
HEADS = ("operation", "click_target")
MAX_MANIFEST = 8 * 1024**2


def _need(value, message):
    if not value:
        raise ValueError(message)


def _gate():
    _need(SOURCE_ONLY is False, "held browser binding requires independent review")


def _encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def _parse(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            _need(key not in value, "duplicate witness JSON key")
            value[key] = item
        return value

    def finite(value):
        number = float(value)
        _need(math.isfinite(number), "nonfinite JSON number")
        return number

    return json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_float=finite,
        parse_constant=lambda _: _need(False, "nonfinite JSON"),
    )


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _discard(_value):
    """An already closed service never delivers additional event evidence."""


def verify_client_exchange(record, request_bytes, response_bytes):
    """Verify supplied client bytes; their capture origin remains external."""
    _gate()
    _need(
        type(record) is dict
        and record.get("schema") == "decision3-candidate-browser-binding-v1"
        and record.get("status") == "prepared_not_delivered"
        and type(request_bytes) is bytes
        and type(response_bytes) is bytes
        and 0 < len(request_bytes) <= 1_000_000
        and 0 < len(response_bytes) <= 16 * 1024**2,
        "bounded caller-supplied client bytes and prepared binding required",
    )
    request_sha = hashlib.sha256(request_bytes).hexdigest()
    response_sha = hashlib.sha256(response_bytes).hexdigest()
    _need(
        request_sha == record.get("http_body_sha256")
        and response_sha == record.get("response_sha256")
        and _encode(_parse(request_bytes)) == _encode(record["original_request"])
        and _encode(_parse(response_bytes)) == _encode(record["named_response"]),
        "client byte hashes and ordered request/response must match binding",
    )
    return {
        "status": "provided_bytes_match",
        "http_body_sha256": request_sha,
        "response_sha256": response_sha,
        "scope": "Matching supplied bytes only; client capture origin and numerical audit are external",
    }


class CandidateBrowserBinding:
    """Bind 1–16 HTTP calls; ownership and resource supervision remain external.

    ``read_witness`` returns the exact private witness.json bytes for a receipt;
    the caller must ensure ordinary-path, inode and source/lease provenance. The
    SHA pins below establish binding to these provided bytes, not their origin.
    ``closure_guard`` must prove the external listener/request thread is closed
    before this object can release its adapter. Base backend and lease remain
    owned by the caller and are never closed here.
    """

    def __init__(
        self,
        base,
        cpu_deltas,
        delta_sha256,
        torch,
        *,
        lease_module,
        lease,
        http_module,
        model_identity,
        token,
        max_requests,
        identity_guard,
        owner_guard,
        witness_sink,
        witness_snapshot,
        read_witness,
        private_artifact_sink,
        event_sink,
        closure_guard,
    ):
        _gate()
        _need(
            getattr(lease_module, "SOURCE_ONLY", None) is False
            and isinstance(lease, lease_module.ForwardLease)
            and callable(getattr(lease_module, "SerialBackend", None))
            and getattr(http_module, "SOURCE_ONLY", None) is False
            and callable(getattr(http_module, "CandidateHTTP", None)),
            "explicit released lease and HTTP modules required",
        )
        _need(
            type(delta_sha256) is dict
            and set(delta_sha256) == set(HEADS)
            and all(_digest(value) for value in delta_sha256.values()),
            "two independent delta SHA pins required",
        )
        _need(
            type(max_requests) is int and 1 <= max_requests <= 16, "bounded request count required"
        )
        _need(
            all(
                callable(callback)
                for callback in (
                    identity_guard,
                    owner_guard,
                    witness_sink,
                    witness_snapshot,
                    read_witness,
                    private_artifact_sink,
                    event_sink,
                    closure_guard,
                )
            ),
            "external owner, witness, event and closure callbacks required",
        )
        self.base, self.lease, self.torch = base, lease, torch
        self.delta_sha256 = copy.deepcopy(delta_sha256)
        self.model_identity = model_identity
        self.read_witness, self.closure_guard = read_witness, closure_guard
        self.private_artifact_sink, self.event_sink = private_artifact_sink, event_sink
        self.http_module, self._verified_call = http_module, None
        self.wrapper = self.serial = self.service = None
        self.closed = False
        try:
            self.wrapper = adapter.DualHeadDecision3Backend(
                base,
                cpu_deltas,
                self.delta_sha256,
                torch,
                identity_guard=identity_guard,
                witness_sink=witness_sink,
            )
            self.serial = lease_module.SerialBackend(self.wrapper, lease, torch=torch)
            _need(
                self.serial.backend is self.wrapper and self.serial.lease is lease,
                "owned complete candidate and shared lease required",
            )
            self.service = http_module.CandidateHTTP(
                self.serial,
                model_identity=model_identity,
                token=token,
                max_requests=max_requests,
                owner_guard=owner_guard,
                witness_snapshot=witness_snapshot,
                witness_verify=self._verify_witness,
                event_sink=self._event,
            )
        except BaseException as error:
            if self.wrapper is not None:
                try:
                    self.wrapper.close()
                except BaseException as secondary:
                    error.add_note(
                        "Candidate initialization cleanup also failed: " + repr(secondary)
                    )
            self.wrapper = self.serial = self.service = None
            raise

    def _verify_witness(self, receipt, payload, legacy):
        """Bind the current native/HTTP call to pinned private manifest bytes."""
        _gate()
        _need(
            type(receipt) is dict
            and _digest(receipt.get("manifest_sha256"))
            and _digest(receipt.get("witness_sha256"))
            and receipt.get("delta_sha256") == self.delta_sha256,
            "externally pinned two-head witness receipt required",
        )
        raw = self.read_witness(copy.deepcopy(receipt))
        _need(
            type(raw) is bytes
            and 0 < len(raw) <= MAX_MANIFEST
            and hashlib.sha256(raw).hexdigest() == receipt["manifest_sha256"],
            "exact bounded private witness manifest required",
        )
        witness = _parse(raw)
        _need(
            type(witness) is dict
            and witness.get("schema") == audit.FILE_SCHEMA
            and witness.get("witness_sha256") == receipt["witness_sha256"]
            and type(witness.get("metadata")) is dict
            and witness["metadata"].get("delta_sha256") == self.delta_sha256,
            "matching witness commitment and delta identity required",
        )
        parsed, original, _source, _labels = parse_native_payload(
            copy.deepcopy(payload), model_identity=self.model_identity
        )
        native = native_payload(parsed)
        _need(
            type(legacy) is dict
            and type(legacy.get("decision3_native")) is dict
            and _encode(legacy["decision3_native"]["original_request"]) == _encode(original),
            "unchanged ordered original HTTP request required",
        )
        provider = legacy["decision3_native"]["raw_response"]["decision3_provider_evidence"]
        _need(
            _encode(provider["native_payload"]) == _encode(native)
            and _encode(witness["metadata"]["native_payload"]) == _encode(native)
            and _encode(witness["metadata"]["composed_raw"]) == _encode(provider["raw_response"]),
            "same-call native request and composed raw response required",
        )
        named = self.http_module.wire.named_browser_response(payload, legacy)
        self._verified_call = {
            "original_request": copy.deepcopy(original),
            "native_payload": copy.deepcopy(native),
            "native_raw_response": copy.deepcopy(provider["raw_response"]),
            "witness_receipt": copy.deepcopy(receipt),
            "named_response": copy.deepcopy(named),
        }
        return True

    def _event(self, event):
        """Persist request/native/witness/response binding before HTTP delivery."""
        _gate()
        _need(type(event) is dict, "HTTP event required")
        if event.get("status") == "prepared":
            call = self._verified_call
            _need(type(call) is dict, "verified same-call witness required")
            _need(
                type(event.get("call")) is int
                and 1 <= event["call"] <= self.service.max_requests
                and _digest(event.get("http_body_sha256"))
                and event.get("canonical_request_sha256")
                == hashlib.sha256(_encode(call["original_request"])).hexdigest()
                and event.get("response_sha256")
                == hashlib.sha256(self.http_module._encode(call["named_response"])).hexdigest()
                and _encode(event.get("witness_receipt")) == _encode(call["witness_receipt"]),
                "HTTP request, response and same-call witness must agree",
            )
            self.private_artifact_sink(
                {
                    "schema": "decision3-candidate-browser-binding-v1",
                    "status": "prepared_not_delivered",
                    "call": event["call"],
                    "http_body_sha256": event["http_body_sha256"],
                    "canonical_request_sha256": event["canonical_request_sha256"],
                    "response_sha256": event["response_sha256"],
                    **copy.deepcopy(call),
                }
            )
            self._verified_call = None
        elif event.get("status") == "failed":
            self._verified_call = None
        self.event_sink(copy.deepcopy(event))

    def close(self):
        """Release only the adapter, once the external listener is confirmed closed."""
        _gate()
        _need(not self.closed and self.service is not None, "open browser binding required")
        _need(
            self.service.lock.acquire(blocking=False), "candidate request or delivery still active"
        )
        try:
            _need(self.closure_guard() is True, "external listener closure required")
            self.lease.check()
            _need(getattr(self.lease, "fd", None) is None, "GPU lease still held")
            _need(self.service.pending is None, "HTTP delivery still pending")
            self.service.failed = True
            self.wrapper.close()
            self.closed = True
            self._verified_call = None
            # The adapter retains its original runtime even after close().
            # Drop every binding-owned reference so the external owner can
            # close its backend and observe genuine GPU allocator cleanup.
            self.service.backend = None
            self.service.owner_guard = None
            self.service.witness_snapshot = None
            self.service.witness_verify = None
            self.service.event_sink = _discard
            self.wrapper = self.serial = self.base = self.lease = self.torch = None
            self.read_witness = self.private_artifact_sink = self.event_sink = None
            self.closure_guard = self.http_module = None
        finally:
            self.service.lock.release()
