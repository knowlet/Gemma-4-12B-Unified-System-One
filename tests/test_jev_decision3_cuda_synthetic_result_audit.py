"""Synthetic records only: no live plan, model, GPU, or benchmark data."""

import copy
import json
import math

import jev_decision3_cuda_synthetic_bootstrap as bootstrap
import jev_decision3_cuda_synthetic_result_audit as audit
import pytest


def encode(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode()


@pytest.fixture
def records():
    inv, plan_sha, pid = "a" * 32, "b" * 64, 42
    unit = "jev-cuda-synthetic-" + "c" * 32 + ".service"
    group = "/user.slice/app.slice/" + unit
    plan = {
        "schema": "decision3-cuda-synthetic-plan-v1",
        "unit": unit,
        "wall_seconds": 120,
        "memory_floor_bytes": 34709107508,
        "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
    }
    hello = {
        "schema": "decision3-cuda-synthetic-hello-v1",
        "unit": unit,
        "invocation": inv,
        "plan_sha256": plan_sha,
        "pid": pid,
        "pgid": pid,
        "startticks": 99,
        "boot_id": "synthetic",
        "cgroup": group,
    }
    go = {"schema": "decision3-cuda-synthetic-go-v1", "invocation": inv, "plan_sha256": plan_sha}
    probs = [0.7, 0.3]
    confidence = 1 + sum(x * math.log(x) for x in probs) / math.log(2)
    usage = {"input_tokens": 7, "output_tokens": 0}
    raw = {
        "model": "d3-edge",
        "usage": usage,
        "answers": {
            "destination": {
                "type": "choice",
                "choice": "north",
                "probabilities": {"north": 0.7, "south": 0.3},
                "confidence": confidence,
            },
            "fragile": {"type": "noul", "noul": 0.4},
        },
    }
    normalized = {
        "destination": {
            "id": "destination",
            "type": "choice",
            "choice": "north",
            "probabilities": {"north": 0.7, "south": 0.3},
            "confidence": 0.7,
            "margin": 0.4,
        },
        "fragile": {
            "id": "fragile",
            "type": "noul",
            "noul": 0.4,
            "probabilities": {"false": 0.6, "true": 0.4},
            "confidence": 0.6,
            "margin": 0.2,
        },
    }
    result = {
        "schema": "decision3-cuda-synthetic-worker-v1",
        "status": "completed_pending_external_audit",
        "device": "cuda:0",
        "model_id": "vllm-sr/d3-edge",
        "revision": audit.REVISION,
        "publisher_manifest_sha256": audit.MANIFEST_SHA,
        "calls": 1,
        "score": None,
        "initial_allocated_bytes": 0,
        "final_allocated_bytes": 0,
        "final_reserved_bytes": 100,
        "cleanup": "current_process_allocator_zero_pending_external_owner_audit",
        "elapsed_seconds": 2.1,
        "evidence": {
            "raw_response": raw,
            "normalized_answers": normalized,
            "head_lengths": {"destination": 4, "fragile": 3},
            "usage": usage,
        },
    }
    terminal = {
        "schema": "decision3-cuda-synthetic-terminal-v1",
        "status": "completed_pending_external_audit",
        "invocation": inv,
        "plan_sha256": plan_sha,
        "result_sha256": audit.digest(encode(result)),
    }
    events = [
        {
            "event": "prelaunch",
            "monotonic": 1.0,
            "available_bytes": 40000000000,
            "running": 0,
            "pending": 0,
        },
        {
            "event": "host",
            "monotonic": 1.5,
            "available_bytes": 40000000000,
            "running": 0,
            "pending": 0,
        },
        {"event": "binding_verified", "invocation": inv, "pid": pid, "pgid": pid, "cgroup": group},
        {
            "event": "host",
            "monotonic": 2.0,
            "available_bytes": 40000000000,
            "running": 0,
            "pending": 0,
        },
    ]
    closure = {
        "schema": "decision3-cuda-synthetic-supervisor-closure-v1",
        "status": "completed_pending_external_audit",
        "invocation": inv,
        "plan_sha256": plan_sha,
        "unit": "retained-unit",
        "cgroup": "absent",
        "process_group": "absent",
        "machine_exclusive": False,
    }
    return plan, plan_sha, hello, go, terminal, result, events, closure


def run(records):
    plan, sha, hello, go, terminal, result, events, closure = records
    return audit.audit_records(plan, sha, hello, go, terminal, encode(result), events, closure)


def test_public_source_allowlist_matches_reviewed_bootstrap():
    assert len(audit.source_set()) == 27
    assert audit.source_set() == bootstrap._source_set()
    assert all(
        "heldout" not in x and "root-done-feedback" not in x and "/cases" not in x
        for x in audit.source_set()
    )


def test_synthetic_success_distinguishes_assertions_and_observations(records):
    result = run(records)
    assert result["worker_asserted"]["elapsed_seconds"] == 2.1
    assert result["supervisor_observed"]["host_samples"] == 3
    assert result["supervisor_observed"]["max_sample_gap_seconds"] == 0.5
    assert any("UUID" in x for x in result["limitations"])
    assert any("coverage" in x for x in result["limitations"])


@pytest.mark.parametrize(
    "fault",
    [
        "missing_head",
        "probabilities",
        "entropy",
        "normalized",
        "length",
        "usage",
        "allocated",
        "timeout",
        "nonfinite",
        "model",
    ],
)
def test_worker_evidence_tamper_fails_even_with_updated_terminal_digest(records, fault):
    plan, sha, hello, go, terminal, result, events, closure = copy.deepcopy(records)
    if fault == "missing_head":
        result["evidence"]["raw_response"]["answers"].pop("fragile")
    elif fault == "probabilities":
        result["evidence"]["raw_response"]["answers"]["destination"]["probabilities"]["south"] = 0.2
    elif fault == "entropy":
        result["evidence"]["raw_response"]["answers"]["destination"]["confidence"] = 0.9
    elif fault == "normalized":
        result["evidence"]["normalized_answers"]["fragile"]["margin"] = 0.3
    elif fault == "length":
        result["evidence"]["head_lengths"]["fragile"] = True
    elif fault == "usage":
        result["evidence"]["usage"]["output_tokens"] = 1
    elif fault == "allocated":
        result["final_allocated_bytes"] = 1
    elif fault == "timeout":
        result["elapsed_seconds"] = 20
    elif fault == "nonfinite":
        result["elapsed_seconds"] = -1
    else:
        result["model_id"] = "other"
    raw = encode(result)
    terminal["result_sha256"] = audit.digest(raw)
    with pytest.raises(ValueError):
        audit.audit_records(plan, sha, hello, go, terminal, raw, events, closure)


@pytest.mark.parametrize(
    "fault",
    [
        "digest",
        "go",
        "pid",
        "invocation",
        "host",
        "memory",
        "order",
        "binding",
        "closure",
        "exclusive",
    ],
)
def test_binding_and_supervision_fail_closed(records, fault):
    plan, sha, hello, go, terminal, result, events, closure = copy.deepcopy(records)
    if fault == "digest":
        terminal["result_sha256"] = "0" * 64
    elif fault == "go":
        go["invocation"] = "f" * 32
    elif fault == "pid":
        hello["pgid"] = 43
    elif fault == "invocation":
        closure["invocation"] = "f" * 32
    elif fault == "host":
        events[1]["running"] = 1
    elif fault == "memory":
        events[0]["available_bytes"] = 1
    elif fault == "order":
        events[-1]["monotonic"] = 0.5
    elif fault == "binding":
        events[2]["cgroup"] = "/different"
    elif fault == "closure":
        closure["cgroup"] = "unconfirmed"
    else:
        closure["machine_exclusive"] = True
    with pytest.raises(ValueError):
        run((plan, sha, hello, go, terminal, result, events, closure))


def test_json_rejects_duplicate_keys_and_nonfinite_values():
    for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":1e999}'):
        with pytest.raises(ValueError):
            audit.decode(raw)


def test_full_read_only_audit_uses_only_fixed_synthetic_sources(tmp_path, monkeypatch, records):
    root = tmp_path / "workspace"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    base = root / "experiments"
    run_dir = base / ("run-" + "c" * 32)
    worker, supervisor = run_dir / "worker", run_dir / "supervisor"
    worker.mkdir(parents=True)
    supervisor.mkdir()
    monkeypatch.setattr(audit, "ROOT", root)
    monkeypatch.setattr(audit, "BASE", base)
    pins = {}
    for name in audit.HELD:
        source = scripts / name
        release = scripts / (name[:-3] + "_release.py")
        source.write_bytes(b"SOURCE_ONLY = True\n# synthetic\n")
        release.write_bytes(b"SOURCE_ONLY = False\n# synthetic\n")
        for path in (source, release):
            pins[str(path)] = audit.digest(path.read_bytes())
    for index in range(19):
        path = root / ("source-" + str(index))
        path.write_bytes(("synthetic-" + str(index)).encode())
        pins[str(path)] = audit.digest(path.read_bytes())
    monkeypatch.setattr(audit, "source_set", lambda: set(pins))
    manifest = (
        root
        / "artifacts/jev-local-hf/hub/models--vllm-sr--d3-edge/snapshots"
        / audit.REVISION
        / "MODEL_MANIFEST.json"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b"synthetic manifest")
    monkeypatch.setattr(audit, "MANIFEST_SHA", audit.digest(manifest.read_bytes()))
    plan, _sha, hello, go, terminal, result, events, closure = copy.deepcopy(records)
    plan.update(source_pins=pins, output_dir=str(worker))
    plan_raw = encode(plan)
    plan_sha = audit.digest(plan_raw)
    for item in (hello, go, terminal, closure):
        item["plan_sha256"] = plan_sha
    result["publisher_manifest_sha256"] = audit.MANIFEST_SHA
    result_raw = encode(result)
    terminal["result_sha256"] = audit.digest(result_raw)
    (run_dir / "plan.json").write_bytes(plan_raw)
    for name, value in (("hello.json", hello), ("go.json", go), ("terminal.json", terminal)):
        (worker / name).write_bytes(encode(value))
    (worker / "result.json").write_bytes(result_raw)
    (supervisor / "events.jsonl").write_bytes(b"\n".join(encode(x) for x in events) + b"\n")
    (supervisor / "closure.json").write_bytes(encode(closure))
    report = audit.audit(str(run_dir / "plan.json"), plan_sha)
    assert report["independent_checks"]["selected_source_hashes"] == 27
    assert report["independent_checks"]["gate_only_release_pairs"] == 4
    assert report["independent_checks"]["result_sha256"] == audit.digest(result_raw)
    next(iter(root.glob("source-*"))).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="source_changed"):
        audit.audit(str(run_dir / "plan.json"), plan_sha)


def test_unexpected_pin_is_rejected_before_read(monkeypatch):
    allowed = {"/synthetic/" + str(x) for x in range(27)}
    monkeypatch.setattr(audit, "source_set", lambda: allowed)
    monkeypatch.setattr(audit, "_read", lambda *_: pytest.fail("unexpected file read"))
    pins = {name: "a" * 64 for name in allowed}
    pins["/unselected/extra.py"] = "b" * 64
    with pytest.raises(ValueError, match="source_inventory"):
        audit.verify_sources({"source_pins": pins})
