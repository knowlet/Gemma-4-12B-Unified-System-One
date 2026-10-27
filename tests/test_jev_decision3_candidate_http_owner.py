"""Synthetic HTTP-owner checks; no model, GPU or systemd unit is started."""

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import jev_decision3_candidate_http as http
import jev_decision3_candidate_http_owner as owner
import pytest
from test_jev_decision3_link_navigation_candidate_owner import candidate_plan
from test_jev_decision3_native import FakeNative, payload

pytest_plugins = ("test_jev_decision3_candidate_http",)


def transport(tmp_path):
    lock = tmp_path / "lock"
    journal = tmp_path / "journal"
    lock.touch()
    journal.touch()

    def identity(path):
        info = path.stat()
        return {"device": info.st_dev, "inode": info.st_ino}

    return {
        "http_release": str(tmp_path / "http-release.py"),
        "lease_release": str(tmp_path / "lease-release.py"),
        "lock": str(lock),
        "journal": str(journal),
        "nonce": "a" * 32,
        "stat": {"lock": identity(lock), "journal": identity(journal)},
    }


def test_source_gate_plan_and_preserved_resources(tmp_path):
    plan = candidate_plan(tmp_path)
    plan.update(schema=owner.SCHEMA, shard_size=1, transport=transport(tmp_path))
    owner._validate_plan(plan)
    with pytest.raises(ValueError, match="independent review"):
        owner.supervise(tmp_path / "missing", "a" * 64)
    args = owner._unit_args(
        "candidate-http-test.service", tmp_path / "plan", "a" * 64, tmp_path, plan
    )
    for value in (
        "--property=MemoryMax=8589934592",
        "--property=MemorySwapMax=0",
        "--property=CPUQuota=500%",
        "--property=AllowedCPUs=15-19",
        "--property=RuntimeMaxSec=120",
    ):
        assert value in args
    plan["shard_size"] = 2
    with pytest.raises(ValueError, match="bounded shard"):
        owner._validate_plan(plan)


def test_reviewed_release_must_be_exact_single_gate_replacement(tmp_path):
    held = Path(http.__file__).absolute()
    source = held.read_bytes()
    release = tmp_path / "reviewed-http.py"
    release.write_bytes(source.replace(b"SOURCE_ONLY = True", b"SOURCE_ONLY = False", 1))
    pins = {
        str(held): hashlib.sha256(source).hexdigest(),
        str(release): hashlib.sha256(release.read_bytes()).hexdigest(),
    }
    name = "_test_candidate_http_release"
    try:
        module = owner._load_reviewed_release(str(release), held.name, pins, name)
        assert module.SOURCE_ONLY is False
    finally:
        sys.modules.pop(name, None)
    release.write_bytes(release.read_bytes() + b"\n# unauthorized modification\n")
    pins[str(release)] = hashlib.sha256(release.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="only by the explicit source gate"):
        owner._load_reviewed_release(str(release), held.name, pins, name)


def test_real_loopback_client_reads_complete_response_and_one_witness(setup):
    service = setup.make()
    original = json.dumps(payload(2), ensure_ascii=False, separators=(",", ":")).encode()
    received = owner._run_one_http(http, service, original, lambda: None)
    assert json.loads(received)["decision3_browser_wire"]["answer_values_changed"] is False
    assert len(setup.backend.calls) == 1 and len(setup.receipts) == 1
    assert [event["status"] for event in setup.events] == ["prepared", "socket_write_returned"]
    assert setup.events[-1]["response_sha256"] == hashlib.sha256(received).hexdigest()
    assert setup.events[-1]["http_body_sha256"] == hashlib.sha256(original).hexdigest()


def test_no_http_acceptance_if_same_call_witness_fails(setup):
    service = setup.make()
    setup.behavior["invalid_witness"] = True
    original = json.dumps(payload(2), separators=(",", ":")).encode()
    with pytest.raises(ValueError, match="complete timely HTTP client receipt"):
        owner._run_one_http(http, service, original, lambda: None)
    assert service.failed and not any(row["status"] == "prepared" for row in setup.events)


def test_shared_lease_identity_and_complete_pair_required(tmp_path):
    config = transport(tmp_path)
    pid = os.getpid()
    rows = [
        {
            "event": event,
            "kind": "decision3-gpu",
            "pid": pid,
            "thread": 42,
            "monotonic_ns": index,
            "nonce": config["nonce"],
        }
        for index, event in enumerate(("acquired", "completed"), 1)
    ]
    Path(config["journal"]).write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert owner._lease_events(config, pid) == rows
    rows[-1]["thread"] = 43
    Path(config["journal"]).write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="same-thread"):
        owner._lease_events(config, pid)
    replacement = tmp_path / "replacement"
    replacement.touch()
    replacement.replace(config["lock"])
    with pytest.raises(ValueError, match="identity changed"):
        owner._shared_identity(config)


@pytest.mark.parametrize("reorder", [None, "heads", "criteria"])
def test_admitted_wire_http_witness_and_parent_reject_reordering(tmp_path, monkeypatch, reorder):
    import jev_decision3_link_navigation_adapter as adapter
    import jev_decision3_link_navigation_adapter_audit as audit
    import jev_decision3_link_navigation_features as features

    from s1.evaluation.decision3 import native_payload

    monkeypatch.setattr(owner, "SOURCE_ONLY", False)
    monkeypatch.setattr(http, "SOURCE_ONLY", False)
    plan = candidate_plan(tmp_path)
    plan.update(schema=owner.SCHEMA, shard_size=1, transport=transport(tmp_path))
    plan["cache_root"] = str(tmp_path)
    wire = payload(2)
    wire["model"] = features.ALIAS
    shard = {
        "index": 0,
        "ordered_identity_sha256": "1" * 64,
        "receipt_sha256": "2" * 64,
        "plan_sha256": "3" * 64,
        "wires": [{"payload": wire, "wire_sha256": features._sha(features._canonical(wire))}],
    }
    monkeypatch.setattr(features, "feature_only_shards", lambda *_a, **_k: [shard])
    monkeypatch.setattr(features, "_check_sources", lambda *_: None)
    monkeypatch.setattr(features, "_validate_manifest", lambda *_: None)
    monkeypatch.setattr(owner, "_load_deltas", lambda *_: {})
    monkeypatch.setattr(owner.inspect, "getfile", lambda _obj: __file__)
    plan["source_pins"] = {
        str(Path(p).absolute()): "0" * 64
        for p in (
            adapter.__file__,
            adapter.reference.__file__,
            audit.__file__,
            http.wire.__file__,
            __file__,
        )
    }
    expected = {key: value["fp32_sha256"] for key, value in plan["deltas"].items()}

    def writer(root, *, maximum_calls):
        assert maximum_calls == 1
        root.mkdir()
        receipts = []

        def sink(native, raw):
            directory = root / "call-0000"
            directory.mkdir()
            document = {
                "metadata": {"native_payload": native, "composed_raw": raw},
                "witness_sha256": "5" * 64,
            }
            encoded = owner._json(document)
            (directory / "witness.json").write_bytes(encoded)
            receipts.append(
                {
                    "directory": str(directory),
                    "manifest_sha256": owner._sha(encoded),
                    "witness_sha256": "5" * 64,
                    "delta_sha256": expected,
                }
            )

        return sink, receipts

    monkeypatch.setattr(audit, "make_witness_writer", writer)

    class Candidate(FakeNative):
        def __init__(self, provider, _deltas, _expected, _torch, *, identity_guard, witness_sink):
            super().__init__()
            assert identity_guard(provider, {})
            self.sink = witness_sink

        def predict(self, request, *, deadline=None, evidence_sink=None):
            raw = super().predict(request, deadline=deadline, evidence_sink=evidence_sink)
            native = native_payload(request)
            self.sink(native, raw)
            return {
                **raw,
                "decision3_provider_evidence": {"native_payload": native, "raw_response": raw},
            }

        def close(self):
            pass

    monkeypatch.setattr(adapter, "DualHeadDecision3Backend", Candidate)

    class Lease:
        def __init__(self, lock, journal, stat, *, nonce):
            self.journal, self.nonce = journal, nonce
            self.fd = self.held = self.fatal = None

        def check(self):
            assert self.fatal is None

        def emit(self, event):
            row = {
                "event": event,
                "kind": "decision3-gpu",
                "pid": os.getpid(),
                "thread": 42,
                "monotonic_ns": time.monotonic_ns(),
                "nonce": self.nonce,
            }
            with open(self.journal, "a") as stream:
                stream.write(json.dumps(row) + "\n")

    class Serial:
        def __init__(self, backend, lease, *, torch):
            self.backend, self.lease = backend, lease
            self.spec, self.name = backend.spec, backend.name

        def predict(self, request, **kwargs):
            self.lease.emit("acquired")
            result = self.backend.predict(request, **kwargs)
            self.lease.emit("completed")
            return result

    fake_lease = SimpleNamespace(ForwardLease=Lease, SerialBackend=Serial)
    monkeypatch.setattr(
        owner,
        "_load_reviewed_release",
        lambda _path, name, _pins, _module: (
            http if name == "jev_decision3_candidate_http.py" else fake_lease
        ),
    )
    backend = SimpleNamespace(model=SimpleNamespace(), runtime=SimpleNamespace())
    capture = owner._capture_candidate(plan, object(), backend, object(), tmp_path, lambda: None)
    (tmp_path / "ready.json").write_bytes(owner._json({"pid": os.getpid()}))
    worker = {"capture": capture}
    owner._verify_candidate_result(worker, shard, plan, tmp_path)
    if reorder:
        response_file = tmp_path / "response-0000.json"
        response = json.loads(response_file.read_bytes())
        original = response["decision3_native"]["original_request"]["questions"]
        if reorder == "heads":
            response["decision3_native"]["original_request"]["questions"] = dict(
                reversed(list(original.items()))
            )
        else:
            original["click_target"]["criteria"] = dict(
                reversed(list(original["click_target"]["criteria"].items()))
            )
        encoded = owner._json(response)
        response_file.write_bytes(encoded)
        capture["completed"][0]["response_sha256"] = owner._sha(encoded)
        evidence_path = tmp_path / "http-evidence.json"
        evidence = json.loads(evidence_path.read_bytes())
        for event in evidence["http_events"]:
            event["response_sha256"] = owner._sha(encoded)
        evidence_encoded = owner._json(evidence)
        evidence_path.write_bytes(evidence_encoded)
        capture["http_evidence_sha256"] = owner._sha(evidence_encoded)
        with pytest.raises(ValueError, match="same-call witness"):
            owner._verify_candidate_result(worker, shard, plan, tmp_path)
