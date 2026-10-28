"""Independently check one fixed synthetic CUDA run; never execute a model.

The caller must authenticate the expected plan SHA independently. File hashes
prove the bytes audited now, not uninterrupted execution-time identity. A
successful audit only checks recorded assertions and observations: the result
does not contain an independently measured GPU UUID or forward timestamps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/jev-runtime/decision3-cuda-synthetic"
MANIFEST_SHA = "3bae8989a6feb36a66bf4d8c44112dc63199d0dc69f564dc895fe368b1e08e59"
REVISION = "cab71e6ad3461b8086d51ddb572c656babd63b0b"
HELD = tuple(
    "jev_decision3_cuda_synthetic_" + part + ".py"
    for part in ("bootstrap", "launcher", "supervisor", "worker")
)


def require(value, reason):
    if not value:
        raise ValueError(reason)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate_json_key")
            result[key] = value
        return result

    def number(value):
        result = float(value)
        require(math.isfinite(result), "nonfinite_json")
        return result

    return json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_float=number,
        parse_constant=lambda _: require(False, "nonfinite_json"),
    )


def _read(path, limit):
    require(path.is_absolute() and path.resolve() == path, "ordinary_path")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_size <= limit, "bounded_file")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        after, current = os.fstat(fd), os.stat(path, follow_symlinks=False)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        require(
            tuple(getattr(before, x) for x in fields)
            == tuple(getattr(after, x) for x in fields)
            == tuple(getattr(current, x) for x in fields)
            and len(raw) == before.st_size,
            "file_changed",
        )
        return raw
    finally:
        os.close(fd)


def source_set():
    scripts = ROOT / "scripts"
    support = (
        "jev_decision3_candidate_browser_child.py",
        "jev_decision3_candidate_http_owner.py",
        "jev_decision3_cpu_experiment_worker.py",
        "jev_stage_memory_profile_v2.py",
    )
    project = (
        "__init__.py",
        "backends.py",
        "contracts.py",
        "errors.py",
        "evaluation/__init__.py",
        "evaluation/contracts.py",
        "evaluation/decision3.py",
    )
    staged = (
        ROOT
        / "artifacts/jev-runtime/decision3-link-navigation-diagnostic-20261027-01/staged-modules/transformers_modules/cab71e6ad3461b8086d51ddb572c656babd63b0b/5e85afd45ec2b404"
    )
    snapshot = ROOT / "artifacts/jev-local-hf/hub/models--vllm-sr--d3-edge/snapshots" / REVISION
    overlay = ROOT / "artifacts/jev-runtime/overlay"
    paths = [scripts / x for x in (*HELD, *support)]
    paths += [scripts / (x[:-3] + "_release.py") for x in HELD]
    paths += [ROOT / "src/s1" / x for x in project]
    paths += [staged / x for x in ("d3_runtime.py", "d3_format.py", "modeling_d3.py")]
    paths += [
        snapshot / "d3_fast.py",
        overlay / "transformers/__init__.py",
        overlay / "transformers/models/qwen3_5/modeling_qwen3_5.py",
        Path("/home/albert_hsieh/ComfyUI/.venv/lib/python3.13/site-packages/torch/__init__.py"),
        Path(
            "/home/albert_hsieh/.local/share/uv/python/cpython-3.13.14-linux-aarch64-gnu/bin/python3.13"
        ),
    ]
    return {str(path) for path in paths}


def verify_sources(plan):
    pins = plan.get("source_pins")
    expected = source_set()
    require(
        type(pins) is dict and len(expected) == 27 and set(pins) == expected, "source_inventory"
    )
    raw_sources = {}
    for name in sorted(expected):
        value = pins[name]
        require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), "source_digest")
        raw = _read(Path(name), 256 * 1024**2)
        require(digest(raw) == value, "source_changed")
        if Path(name).parent == ROOT / "scripts" and Path(name).name.startswith(
            "jev_decision3_cuda_synthetic_"
        ):
            raw_sources[name] = raw
    for name in HELD:
        held = ROOT / "scripts" / name
        release = held.with_name(name[:-3] + "_release.py")
        source, active = raw_sources[str(held)], raw_sources[str(release)]
        marker = b"SOURCE_ONLY" + b" = True"
        require(
            source.count(marker) == 1
            and active == source.replace(marker, b"SOURCE_ONLY = False", 1),
            "release_difference",
        )
    manifest = (
        ROOT
        / "artifacts/jev-local-hf/hub/models--vllm-sr--d3-edge/snapshots"
        / REVISION
        / "MODEL_MANIFEST.json"
    )
    require(digest(_read(manifest, 8 * 1024**2)) == MANIFEST_SHA, "publisher_manifest")
    return len(expected)


def probability(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def validate_result(result):
    require(
        type(result) is dict
        and result.get("schema") == "decision3-cuda-synthetic-worker-v1"
        and result.get("status") == "completed_pending_external_audit",
        "worker_status",
    )
    require(
        result.get("model_id") == "vllm-sr/d3-edge"
        and result.get("revision") == REVISION
        and result.get("publisher_manifest_sha256") == MANIFEST_SHA
        and result.get("device") == "cuda:0"
        and type(result.get("calls")) is int
        and result["calls"] == 1
        and result.get("score") is None,
        "worker_identity",
    )
    require(
        type(result.get("initial_allocated_bytes")) is int
        and result["initial_allocated_bytes"] == 0
        and type(result.get("final_allocated_bytes")) is int
        and result["final_allocated_bytes"] == 0
        and type(result.get("final_reserved_bytes")) is int
        and result["final_reserved_bytes"] >= 0
        and result.get("cleanup") == "current_process_allocator_zero_pending_external_owner_audit",
        "worker_cleanup",
    )
    elapsed = result.get("elapsed_seconds")
    require(
        type(elapsed) in (int, float) and math.isfinite(elapsed) and 0 <= elapsed < 20,
        "worker_elapsed",
    )
    evidence = result.get("evidence")
    require(type(evidence) is dict, "evidence")
    raw, normalized, lengths, usage = (
        evidence.get(x) for x in ("raw_response", "normalized_answers", "head_lengths", "usage")
    )
    require(
        type(lengths) is dict
        and list(lengths) == ["destination", "fragile"]
        and all(type(n) is int and 1 <= n <= 16384 for n in lengths.values()),
        "head_lengths",
    )
    require(
        type(raw) is dict
        and raw.get("model") == "d3-edge"
        and type(raw.get("answers")) is dict
        and list(raw["answers"]) == list(lengths)
        and type(usage) is dict
        and usage == raw.get("usage")
        and type(usage.get("input_tokens")) is int
        and usage["input_tokens"] == sum(lengths.values())
        and type(usage.get("output_tokens")) is int
        and usage["output_tokens"] == 0,
        "native_response",
    )
    choice, truth = raw["answers"]["destination"], raw["answers"]["fragile"]
    require(
        type(choice) is dict
        and set(choice) == {"type", "choice", "probabilities", "confidence"}
        and choice["type"] == "choice"
        and type(choice["probabilities"]) is dict
        and list(choice["probabilities"]) == ["north", "south"],
        "choice",
    )
    probs = list(choice["probabilities"].values())
    require(
        all(probability(x) for x in probs)
        and math.isclose(sum(probs), 1, rel_tol=0, abs_tol=1e-6)
        and choice["choice"] == ["north", "south"][probs.index(max(probs))]
        and probability(choice["confidence"]),
        "choice_probability",
    )
    entropy = 1 + sum(x * math.log(x) for x in probs if x) / math.log(2)
    require(
        math.isclose(choice["confidence"], entropy, rel_tol=1e-12, abs_tol=1e-12),
        "native_confidence",
    )
    require(
        type(truth) is dict
        and set(truth) == {"type", "noul"}
        and truth["type"] == "noul"
        and probability(truth["noul"]),
        "native_noul",
    )
    require(type(normalized) is dict and list(normalized) == list(lengths), "normalized_heads")
    for key, labels, expected, kind in (
        ("destination", ["north", "south"], [x / sum(probs) for x in probs], "choice"),
        ("fragile", ["false", "true"], [1 - truth["noul"], truth["noul"]], "noul"),
    ):
        answer = normalized[key]
        require(
            type(answer) is dict
            and answer.get("id") == key
            and answer.get("type") == kind
            and type(answer.get("probabilities")) is dict
            and list(answer["probabilities"]) == labels
            and all(probability(x) for x in answer["probabilities"].values())
            and all(
                math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                for a, b in zip(answer["probabilities"].values(), expected, strict=True)
            )
            and probability(answer.get("confidence"))
            and math.isclose(answer["confidence"], max(expected), rel_tol=1e-12, abs_tol=1e-12)
            and probability(answer.get("margin"))
            and math.isclose(
                answer["margin"], abs(expected[0] - expected[1]), rel_tol=1e-12, abs_tol=1e-12
            ),
            "normalized_response",
        )
    require(
        normalized["destination"].get("choice") == choice["choice"]
        and normalized["fragile"].get("noul") == truth["noul"],
        "answer_identity",
    )
    return elapsed


def audit_records(plan, plan_sha, hello, go, terminal, result_raw, events, closure):
    require(
        type(plan) is dict
        and plan.get("schema") == "decision3-cuda-synthetic-plan-v1"
        and type(plan.get("wall_seconds")) is int
        and plan["wall_seconds"] == 120
        and type(plan.get("memory_floor_bytes")) is int
        and plan["memory_floor_bytes"] >= 34709107508
        and type(plan.get("gpu_uuid")) is str
        and re.fullmatch(
            r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", plan["gpu_uuid"]
        ),
        "plan",
    )
    require(
        type(hello) is dict
        and hello.get("schema") == "decision3-cuda-synthetic-hello-v1"
        and hello.get("plan_sha256") == plan_sha
        and hello.get("unit") == plan.get("unit")
        and type(hello.get("pid")) is int
        and hello["pid"] > 0
        and hello.get("pgid") == hello["pid"]
        and type(hello.get("startticks")) is int
        and hello["startticks"] > 0
        and type(hello.get("boot_id")) is str
        and bool(hello["boot_id"])
        and type(hello.get("invocation")) is str
        and re.fullmatch(r"[0-9a-f]{32}", hello["invocation"])
        and type(hello.get("cgroup")) is str
        and hello["cgroup"].endswith("/app.slice/" + plan["unit"]),
        "hello",
    )
    inv = hello["invocation"]
    require(
        type(go) is dict
        and go
        == {"schema": "decision3-cuda-synthetic-go-v1", "invocation": inv, "plan_sha256": plan_sha},
        "go",
    )
    require(
        type(terminal) is dict
        and terminal.get("schema") == "decision3-cuda-synthetic-terminal-v1"
        and terminal.get("status") == "completed_pending_external_audit"
        and terminal.get("invocation") == inv
        and terminal.get("plan_sha256") == plan_sha
        and terminal.get("result_sha256") == digest(result_raw),
        "terminal",
    )
    elapsed = validate_result(decode(result_raw))
    require(
        type(events) is list and events and events[0].get("event") == "prelaunch",
        "supervisor_events",
    )
    samples, bindings = [], []
    for item in events:
        require(type(item) is dict, "supervisor_events")
        event = item.get("event")
        if event in {"prelaunch", "host"}:
            stamp, memory = item.get("monotonic"), item.get("available_bytes")
            require(
                type(stamp) in (int, float)
                and math.isfinite(stamp)
                and type(memory) is int
                and memory >= plan["memory_floor_bytes"]
                and type(item.get("running")) is int
                and item["running"] == 0
                and type(item.get("pending")) is int
                and item["pending"] == 0,
                "host_observation",
            )
            samples.append(stamp)
        elif event == "binding_verified":
            require(
                item.get("invocation") == inv
                and item.get("pid") == hello["pid"]
                and item.get("pgid") == hello["pgid"]
                and item.get("cgroup") == hello["cgroup"],
                "binding",
            )
            bindings.append(item)
        else:
            raise ValueError("supervisor_events")
    require(
        sum(x.get("event") == "prelaunch" for x in events) == 1
        and any(x.get("event") == "host" for x in events)
        and len(bindings) == 1
        and samples == sorted(samples),
        "supervisor_events",
    )
    require(
        type(closure) is dict
        and closure.get("schema") == "decision3-cuda-synthetic-supervisor-closure-v1"
        and closure.get("status") == "completed_pending_external_audit"
        and closure.get("invocation") == inv
        and closure.get("plan_sha256") == plan_sha
        and closure.get("unit") == "retained-unit"
        and closure.get("cgroup") in {"absent", "confirmed_unpopulated"}
        and closure.get("process_group") == "absent"
        and closure.get("machine_exclusive") is False,
        "closure",
    )
    return {
        "worker_asserted": {
            "one_call": True,
            "device": "cuda:0",
            "elapsed_seconds": elapsed,
            "initial_allocated_bytes": 0,
            "final_allocated_bytes": 0,
        },
        "supervisor_observed": {
            "host_samples": len(samples),
            "max_sample_gap_seconds": max(
                (b - a for a, b in zip(samples, samples[1:])), default=None
            ),
            "cgroup": closure["cgroup"],
            "process_group": closure["process_group"],
        },
        "limitations": [
            "actual GPU UUID is not recorded in the result",
            "host sampling does not establish complete forward coverage",
            "elapsed excludes model load and cleanup and is not a hard deadline",
            "single synthetic request is not an accuracy benchmark or p95",
            "current file hashes do not prove uninterrupted execution-time identity",
            "dynamic HF modules and external libraries are not an atomically verified supply chain",
        ],
    }


def audit(plan_path, expected_sha):
    path = Path(plan_path)
    require(
        type(expected_sha) is str
        and re.fullmatch(r"[0-9a-f]{64}", expected_sha)
        and path.is_absolute()
        and path.parent.parent == BASE
        and path.name == "plan.json"
        and re.fullmatch(r"run-[0-9a-f]{32}", path.parent.name),
        "plan_path",
    )
    plan_raw = _read(path, 1024**2)
    require(digest(plan_raw) == expected_sha, "plan_digest")
    plan = decode(plan_raw)
    require(
        plan.get("output_dir") == str(path.parent / "worker")
        and plan.get("unit") == "jev-cuda-synthetic-" + path.parent.name[4:] + ".service",
        "plan_identity",
    )
    count = verify_sources(plan)
    worker, supervisor = path.parent / "worker", path.parent / "supervisor"
    result_raw = _read(worker / "result.json", 4 * 1024**2)
    events_raw = _read(supervisor / "events.jsonl", 8 * 1024**2)
    report = audit_records(
        plan,
        expected_sha,
        decode(_read(worker / "hello.json", 4096)),
        decode(_read(worker / "go.json", 4096)),
        decode(_read(worker / "terminal.json", 4096)),
        result_raw,
        [decode(line) for line in events_raw.splitlines()],
        decode(_read(supervisor / "closure.json", 16384)),
    )
    report["independent_checks"] = {
        "plan_sha256": expected_sha,
        "selected_source_hashes": count,
        "gate_only_release_pairs": len(HELD),
        "publisher_manifest_sha256": MANIFEST_SHA,
        "result_sha256": digest(result_raw),
        "native_probability_consistency": True,
        "record_bindings": True,
    }
    return report


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.plan, args.plan_sha256), allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
