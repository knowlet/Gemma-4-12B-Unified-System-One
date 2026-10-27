"""Synthetic owner artifacts and independent replay, with no browser or model."""

import copy
import hashlib
import io
import json
from pathlib import Path

import jev_decision3_link_navigation_feature_audit as audit
import jev_decision3_link_navigation_features as features
import jev_decision3_link_navigation_training as training
import numpy as np
import pytest
from test_jev_decision3_link_navigation_training import (
    old_dataset,
    synthetic_collection,
    synthetic_shards,
)


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def pin(data):
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def save_npy(value):
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


def synthetic_owner(root, source, admitted, captured):
    root.mkdir()
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    source_pins = {
        str(source): source_sha,
        audit.owner.GPU_PYTHON_REAL: audit.owner.GPU_PYTHON_SHA256,
    }
    for module in (audit.owner, features, audit.pipeline, audit.owner.systemd):
        path = Path(module.__file__).absolute()
        source_pins[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    plan = {
        "schema": "decision3-link-navigation-gpu-owner-plan-v1",
        "collection_dir": str(root.parent / "collection"),
        "receipt_sha256": admitted.receipt_sha256,
        "collection_plan_sha256": admitted.plan_sha256,
        "cache_root": str(root.parent / "cache"),
        "modules_cache": str(root.parent / "modules"),
        "overlay": str(
            Path(audit.owner.__file__).absolute().parent.parent / "artifacts/jev-runtime/overlay"
        ),
        "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
        "source_pins": source_pins,
        "python": audit.owner.GPU_PYTHON,
        "output_dir": str(root),
        "shard_size": 40,
        "shard_index": captured["index"],
        "memory_floor_bytes": audit.owner.MEMORY_FLOOR,
    }
    plan_bytes = compact(plan)
    plan_sha = hashlib.sha256(plan_bytes).hexdigest()
    invocation = "a" * 32
    unit = "jev-link-feature-" + "b" * 32 + ".service"
    group = "/user.slice/user-1001.slice/user@1001.service/app.slice/" + unit
    ready = {"pid": 1234, "startticks": 5678, "boot_id": "synthetic-boot", "invocation": invocation}
    events = [
        {
            "event": "prelaunch_host",
            "monotonic": 1.0,
            "available_bytes": plan["memory_floor_bytes"],
            "running": 0,
            "pending": 0,
        },
        {"event": "unit_requested", "unit": unit, "plan_sha256": plan_sha},
        {
            "event": "host",
            "monotonic": 2.0,
            "available_bytes": plan["memory_floor_bytes"],
            "running": 0,
            "pending": 0,
        },
        {
            "event": "binding_verified",
            "invocation": invocation,
            "pid": 1234,
            "startticks": 5678,
            "boot_id": "synthetic-boot",
            "cgroup": group,
        },
    ]
    events.append(
        {
            "event": "host",
            "monotonic": 3.0,
            "available_bytes": plan["memory_floor_bytes"],
            "running": 0,
            "pending": 0,
        }
    )
    terminal = {
        "InvocationID": invocation,
        "ActiveState": "active",
        "SubState": "exited",
        "MainPID": "0",
        "ExecMainCode": "exited",
        "ExecMainStatus": "0",
        "Result": "success",
        "ControlGroup": group,
    }
    closure = {
        "schema": "decision3-link-navigation-gpu-owner-closure-v1",
        "unit": unit,
        "invocation": invocation,
        "launch_attempted": True,
        "terminal": terminal,
        "stop_returned": False,
        "closure": "retained-unit",
        "cgroup_closure": "confirmed_unpopulated",
        "fit_eligible": False,
        "model_score": None,
        "final_unit": {
            "InvocationID": invocation,
            "LoadState": "loaded",
            "ActiveState": "active",
            "MainPID": "0",
            "ControlGroup": group,
            "Result": "success",
        },
    }
    capture = copy.deepcopy(captured)
    capture["source_pins"] = source_pins
    blobs = {
        "ready.json": compact(ready),
        "go.json": compact({"invocation": invocation, "plan_sha256": plan_sha}),
        "owner-events.jsonl": b"".join(compact(event) + b"\n" for event in events),
        "owner-closure.json": compact(closure),
        "worker.log": b"",
        "baseline.npy": save_npy(capture["baseline"]),
    }
    capture["baseline"] = {"path": "baseline.npy", **pin(blobs["baseline.npy"])}
    for index, case in enumerate(capture["cases"]):
        name = f"features-{index:04d}.npy"
        blobs[name] = save_npy(case["features"])
        case["features"] = {"path": name, **pin(blobs[name])}
        shape = [len(case["head_order"]), 10, 1024]
        case["first_layer_shape"] = case["final_norm_shape"] = shape
        case["raw_response"]["usage"] = {
            "input_tokens": len(case["head_order"]) * 10,
            "output_tokens": 0,
        }
        raw = compact(case["raw_response"])
        case["raw_response_sha256"] = hashlib.sha256(raw).hexdigest()
        name = f"raw-{index:04d}.json"
        blobs[name] = raw
        case["raw_response"] = {"path": name, **pin(raw)}
    worker = {
        "schema": "decision3-link-navigation-gpu-owner-worker-v1",
        "status": "captured_pending_independent_fp64_audit",
        "fit_eligible": False,
        "model_score": None,
        "invocation": invocation,
        "plan_sha256": plan_sha,
        "gpu": {
            "uuid": plan["gpu_uuid"],
            "name": "synthetic",
            "capability": [12, 1],
            "torch_cuda": "synthetic",
        },
        "initial_cuda_allocated": 0,
        "final_cuda_allocated": 0,
        "capture": capture,
    }
    blobs["worker-result.json"] = compact(worker)
    for name, data in blobs.items():
        (root / name).write_bytes(data)
    return {
        "root": root,
        "owner_plan_bytes": plan_bytes,
        "expected_owner_plan_sha256": plan_sha,
        "expected_files": {name: pin(data) for name, data in blobs.items()},
        "expected_source_pins": source_pins,
        "admitted": admitted,
    }


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    binary = tmp_path / "selected-python-real"
    binary.write_bytes(b"synthetic python executable")
    selected = tmp_path / "selected-python"
    selected.symlink_to(binary)
    monkeypatch.setattr(audit.owner, "GPU_PYTHON", str(selected))
    monkeypatch.setattr(audit.owner, "GPU_PYTHON_REAL", str(binary))
    monkeypatch.setattr(
        audit.owner, "GPU_PYTHON_SHA256", hashlib.sha256(binary.read_bytes()).hexdigest()
    )
    old, accepted = old_dataset(monkeypatch)
    monkeypatch.setattr(features, "W0_SHA256", training.W0_SHA256)
    admitted = synthetic_collection()
    shards = synthetic_shards(admitted, old["operation"]["baseline"])
    source = tmp_path / "source.py"
    source.write_text("# synthetic pinned source\n")
    owners = [
        synthetic_owner(tmp_path / f"owner-{i}", source, admitted, shard)
        for i, shard in enumerate(shards)
    ]
    return old, accepted, admitted, owners


def write_repin(kwargs, name, data):
    (kwargs["root"] / name).write_bytes(data)
    kwargs["expected_files"][name] = pin(data)


def test_audited_shards_feed_real_merge_without_fit_authorization(inputs):
    old, accepted, admitted, owners = inputs
    results = [audit.audit_shard(**kwargs) for kwargs in owners]
    assert all(
        item["audit"]["status"] == "fp64_replayed_pending_external_admission" for item in results
    )
    assert sum(item["audit"]["cases"] for item in results) == 72
    assert sum(item["audit"]["heads"] for item in results) == 144
    assert all(item["shard"]["fit_eligible"] is False for item in results)
    merged = training.merge(old, accepted, admitted, [item["shard"] for item in results])
    assert merged["new_rows"] == {"operation": 72, "click_target": 48}
    assert merged["fit_eligible"] is False and merged["scores"] is None


@pytest.mark.parametrize(
    "kind",
    [
        "extra",
        "modified_file",
        "symlink",
        "plan_sha",
        "source",
        "wrong_invocation",
        "terminal_failure",
        "open_cgroup",
        "monitor_failure",
        "wrong_raw",
        "invalid_probability",
        "bad_usage",
        "head_order",
        "shape",
        "npy_trailing",
    ],
)
def test_corrupt_or_unclosed_artifacts_are_refused(inputs, kind):
    _, _, _, owners = inputs
    kwargs = owners[0]
    root = kwargs["root"]
    if kind == "extra":
        (root / "unexpected").write_bytes(b"x")
    elif kind == "modified_file":
        (root / "baseline.npy").write_bytes(b"changed")
    elif kind == "symlink":
        name = root / "worker.log"
        name.unlink()
        name.symlink_to(root / "ready.json")
    elif kind == "plan_sha":
        kwargs["expected_owner_plan_sha256"] = "0" * 64
    elif kind == "source":
        Path(next(iter(kwargs["expected_source_pins"]))).write_text("changed")
    elif kind in {"wrong_invocation", "terminal_failure", "open_cgroup"}:
        value = json.loads((root / "owner-closure.json").read_bytes())
        if kind == "wrong_invocation":
            value["terminal"]["InvocationID"] = "c" * 32
        elif kind == "terminal_failure":
            value["terminal"]["ExecMainStatus"] = "1"
        else:
            value["cgroup_closure"] = "unknown"
        write_repin(kwargs, "owner-closure.json", compact(value))
    elif kind == "monitor_failure":
        write_repin(
            kwargs,
            "owner-events.jsonl",
            (root / "owner-events.jsonl").read_bytes()
            + compact({"event": "monitor_failed"})
            + b"\n",
        )
    elif kind in {"wrong_raw", "invalid_probability", "bad_usage"}:
        name = "raw-0000.json"
        raw = json.loads((root / name).read_bytes())
        if kind == "bad_usage":
            raw["usage"]["input_tokens"] += 1
        elif kind == "wrong_raw":
            raw["answers"]["operation"]["confidence"] = 0.5
        else:
            raw["answers"]["operation"]["probabilities"]["CLICK"] = 0.9
        raw_bytes = compact(raw)
        write_repin(kwargs, name, raw_bytes)
        worker = json.loads((root / "worker-result.json").read_bytes())
        case = worker["capture"]["cases"][0]
        case["raw_response"] = {"path": name, **pin(raw_bytes)}
        case["raw_response_sha256"] = hashlib.sha256(raw_bytes).hexdigest()
        write_repin(kwargs, "worker-result.json", compact(worker))
    elif kind in {"head_order", "shape"}:
        worker = json.loads((root / "worker-result.json").read_bytes())
        case = worker["capture"]["cases"][0]
        if kind == "head_order":
            case["head_order"].reverse()
        else:
            case["final_norm_shape"][1] += 1
        write_repin(kwargs, "worker-result.json", compact(worker))
    else:
        name = "features-0000.npy"
        data = (root / name).read_bytes() + b"extra"
        write_repin(kwargs, name, data)
        worker = json.loads((root / "worker-result.json").read_bytes())
        worker["capture"]["cases"][0]["features"] = {"path": name, **pin(data)}
        write_repin(kwargs, "worker-result.json", compact(worker))
    with pytest.raises((ValueError, OSError)):
        audit.audit_shard(**kwargs)


@pytest.mark.parametrize("kind", ["nan", "wrong_dtype", "object", "raw_whitespace", "gpu_uuid"])
def test_rehashed_invalid_npy_and_owner_identity_are_refused(inputs, kind):
    _, _, _, owners = inputs
    kwargs = owners[0]
    root = kwargs["root"]
    worker = json.loads((root / "worker-result.json").read_bytes())
    case = worker["capture"]["cases"][0]
    if kind == "gpu_uuid":
        worker["gpu"]["uuid"] = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    elif kind == "raw_whitespace":
        name = "raw-0000.json"
        data = (root / name).read_bytes() + b" "
        write_repin(kwargs, name, data)
        case["raw_response"] = {"path": name, **pin(data)}
    else:
        name = "features-0000.npy"
        if kind == "object":
            value = np.empty((2, 1024), dtype=object)
            value[:] = "untrusted"
            stream = io.BytesIO()
            np.save(stream, value, allow_pickle=True)
            data = stream.getvalue()
        else:
            value = np.zeros((2, 1024), dtype=np.float64 if kind == "wrong_dtype" else np.float32)
            if kind == "nan":
                value[0, 0] = np.nan
            data = save_npy(value)
        write_repin(kwargs, name, data)
        case["features"] = {"path": name, **pin(data)}
    write_repin(kwargs, "worker-result.json", compact(worker))
    with pytest.raises(ValueError):
        audit.audit_shard(**kwargs)


def test_fixed_large_gpu_python_uses_bounded_streaming_and_other_sources_remain_bounded(
    tmp_path, monkeypatch
):
    large = tmp_path / "large-python"
    with large.open("wb") as stream:
        stream.truncate(audit.MAX_FILE + 4096)
    with large.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    monkeypatch.setattr(audit.owner, "GPU_PYTHON_REAL", str(large))
    monkeypatch.setattr(audit.owner, "GPU_PYTHON_SHA256", digest)
    audit._check_sources({str(large): digest})
    ordinary = tmp_path / "ordinary-source"
    with ordinary.open("wb") as stream:
        stream.truncate(audit.MAX_FILE + 1)
    with pytest.raises(ValueError, match="bounded ordinary"):
        audit._check_sources({str(ordinary): digest})
    with pytest.raises(ValueError, match="fixed GPU Python digest"):
        audit._check_sources({str(large): "f" * 64})


def test_wrong_selected_interpreter_plan_is_refused(inputs):
    _, _, _, owners = inputs
    kwargs = owners[0]
    plan = json.loads(kwargs["owner_plan_bytes"])
    plan["python"] = "/usr/bin/python3"
    kwargs["owner_plan_bytes"] = compact(plan)
    kwargs["expected_owner_plan_sha256"] = hashlib.sha256(kwargs["owner_plan_bytes"]).hexdigest()
    with pytest.raises(ValueError, match="fixed pinned GPU Python"):
        audit.audit_shard(**kwargs)


def test_missing_gpu_python_pin_is_refused(inputs):
    _, _, _, owners = inputs
    kwargs = owners[0]
    plan = json.loads(kwargs["owner_plan_bytes"])
    del plan["source_pins"][audit.owner.GPU_PYTHON_REAL]
    kwargs["owner_plan_bytes"] = compact(plan)
    kwargs["expected_owner_plan_sha256"] = hashlib.sha256(kwargs["owner_plan_bytes"]).hexdigest()
    with pytest.raises(ValueError, match="fixed pinned GPU Python"):
        audit.audit_shard(**kwargs)


def test_boolean_usage_is_refused_even_when_numerically_equal(inputs):
    _, _, _, owners = inputs
    kwargs = owners[0]
    root = kwargs["root"]
    name = "raw-0000.json"
    raw = json.loads((root / name).read_bytes())
    raw["usage"]["output_tokens"] = False
    raw_bytes = compact(raw)
    write_repin(kwargs, name, raw_bytes)
    worker = json.loads((root / "worker-result.json").read_bytes())
    case = worker["capture"]["cases"][0]
    case["raw_response"] = {"path": name, **pin(raw_bytes)}
    case["raw_response_sha256"] = hashlib.sha256(raw_bytes).hexdigest()
    write_repin(kwargs, "worker-result.json", compact(worker))
    with pytest.raises(ValueError, match="native response or usage"):
        audit.audit_shard(**kwargs)
