"""Invented source-only owner checks; never start a model, systemd or GPU."""

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import jev_decision3_candidate_http as http
import jev_decision3_semantic_smoke_owner as owner
import numpy as np
import pytest
from test_jev_decision3_candidate_http_owner import transport
from test_jev_decision3_link_navigation_candidate_owner import candidate_plan
from test_jev_decision3_native import FakeNative, payload


def make_plan(tmp_path):
    plan = candidate_plan(tmp_path)
    names = ["CLICK", "TYPE_TEXT"]
    data = owner._json(names)
    path = tmp_path / "names.json"
    path.write_bytes(data)
    plan.update(
        schema=owner.SCHEMA,
        shard_size=1,
        transport=transport(tmp_path),
        vocabulary={
            "path": str(path),
            "file_sha256": owner._sha(data),
            "names_sha256": owner._sha(data),
        },
        semantic_adapter_release=str(tmp_path / "semantic_adapter_release.py"),
        semantic_audit_release=str(tmp_path / "semantic_audit_release.py"),
    )
    return plan


def test_gate_plan_and_resource_contract(tmp_path):
    plan = make_plan(tmp_path)
    owner._validate_plan(plan)
    with pytest.raises(ValueError, match="independent review"):
        owner.supervise(tmp_path / "absent", "a" * 64)
    args = owner._unit_args("invented.service", tmp_path / "plan", "a" * 64, tmp_path, plan)
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


def test_vocabulary_exact_file_and_order_pins(tmp_path):
    plan = make_plan(tmp_path)
    assert owner._load_vocabulary(plan) == ["CLICK", "TYPE_TEXT"]
    path = Path(plan["vocabulary"]["path"])
    path.write_bytes(b'["TYPE_TEXT","CLICK"]')
    with pytest.raises(ValueError, match="vocabulary file changed"):
        owner._load_vocabulary(plan)
    plan["vocabulary"]["file_sha256"] = owner._sha(path.read_bytes())
    with pytest.raises(ValueError, match="canonical pinned vocabulary"):
        owner._load_vocabulary(plan)


def test_delta_shape_and_raw_pin_bind_to_vocabulary(tmp_path):
    plan = make_plan(tmp_path)
    for name, n in (("operation", 2), ("click_target", 255)):
        array = np.zeros((n, 1024), dtype="<f4")
        path = tmp_path / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        plan["deltas"][name] = {
            "path": str(path),
            "file_sha256": owner._sha(path.read_bytes()),
            "fp32_sha256": owner._sha(array.tobytes()),
        }

    class Torch:
        @staticmethod
        def from_numpy(array):
            return array

    result = owner._load_deltas(plan, Torch)
    assert result["operation"].shape == (2, 1024)
    assert result["click_target"].shape == (255, 1024)
    plan["deltas"]["operation"]["fp32_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="complete pinned"):
        owner._load_deltas(plan, Torch)


def test_control_identity_binds_plan_invocation_process_and_unit(monkeypatch):
    pid, invocation, unit, group = (
        123,
        "a" * 32,
        "invented.service",
        "/user.slice/app.slice/invented.service",
    )
    monkeypatch.setattr(owner.systemd, "boot_id", lambda: "boot")
    monkeypatch.setattr(owner.systemd, "process_identity", lambda _pid: {"startticks": 50})
    monkeypatch.setattr(owner.systemd, "cgroup_membership", lambda _pid: group)
    properties = {
        "InvocationID": invocation,
        "ControlGroup": group,
        "LoadState": "loaded",
        "ActiveState": "active",
        "Type": "exec",
        "MainPID": str(pid),
    }
    ready = {
        "pid": pid,
        "startticks": 50,
        "boot_id": "boot",
        "invocation": invocation,
        "unit": unit,
        "plan_sha256": "b" * 64,
    }
    assert owner._identity(properties, ready, unit, "b" * 64) == (invocation, group)
    for key, value in (("pid", True), ("unit", "other.service"), ("plan_sha256", "c" * 64)):
        altered = {**ready, key: value}
        with pytest.raises(ValueError, match="exact owned"):
            owner._identity(properties, altered, unit, "b" * 64)


def test_source_owner_has_single_explicit_gate():
    raw = Path(owner.__file__).read_bytes()
    assert raw.count(b"SOURCE_ONLY" + b" = True") == 1


@pytest.mark.parametrize("reorder", [None, "heads", "criteria", "projection", "delta"])
def test_admitted_wire_http_witness_and_parent_reject_reordering(tmp_path, monkeypatch, reorder):
    import jev_decision3_link_navigation_features as features
    import jev_decision3_semantic_operation_adapter as adapter
    import jev_decision3_semantic_operation_audit as audit

    from s1.evaluation.decision3 import native_payload

    monkeypatch.setattr(owner, "SOURCE_ONLY", False)
    monkeypatch.setattr(http, "SOURCE_ONLY", False)
    plan = make_plan(tmp_path)
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

        def sink(native, raw, projection):
            directory = root / "call-0000"
            directory.mkdir()
            document = {
                "metadata": {
                    "native_payload": native,
                    "composed_raw": raw,
                    "operation_names": ["CLICK", "TYPE_TEXT"],
                    "operation_names_sha256": plan["vocabulary"]["names_sha256"],
                    "projection": projection,
                },
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
        def __init__(
            self,
            provider,
            _deltas,
            _expected,
            _torch,
            *,
            identity_guard,
            witness_sink,
            operation_names,
            operation_names_sha256,
        ):
            super().__init__()
            assert identity_guard(provider, {})
            self.sink = witness_sink

        def predict(self, request, *, deadline=None, evidence_sink=None):
            raw = super().predict(request, deadline=deadline, evidence_sink=evidence_sink)
            native = native_payload(request)
            projection = {"operation_names": ["CLICK", "TYPE_TEXT"]}
            self.sink(native, raw, projection)
            return {
                **raw,
                "decision3_provider_evidence": {
                    "native_payload": native,
                    "raw_response": raw,
                    "composition": {
                        "operation_names_sha256": plan["vocabulary"]["names_sha256"],
                        "delta_sha256": expected,
                        "projection": projection,
                    },
                },
            }

        def close(self):
            pass

    monkeypatch.setattr(adapter, "SemanticDualHeadDecision3Backend", Candidate)

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
            http
            if name == "jev_decision3_candidate_http.py"
            else adapter
            if name == "jev_decision3_semantic_operation_adapter.py"
            else audit
            if name == "jev_decision3_semantic_operation_audit.py"
            else fake_lease
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
        elif reorder == "criteria":
            original["click_target"]["criteria"] = dict(
                reversed(list(original["click_target"]["criteria"].items()))
            )
        else:
            composition = response["decision3_native"]["raw_response"][
                "decision3_provider_evidence"
            ]["composition"]
            if reorder == "projection":
                composition["projection"]["operation_names"].reverse()
            else:
                composition["delta_sha256"]["operation"] = "0" * 64
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
        with pytest.raises(ValueError, match="witness"):
            owner._verify_candidate_result(worker, shard, plan, tmp_path)
