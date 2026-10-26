#!/usr/bin/env python3
"""Port only frozen small metadata/source pins; never inspect model/native payloads."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "artifacts/jev-runtime/decision3-publisher-batch4-cpu-20261024-03"
OWNER = ROOT / "artifacts/jev-runtime/llama-cpp-owner-20261024-08"
BASE_PLAN_SHA = "5efcef370b0ae09a78d334d6e00c4a4e39ebbaae9ad18fd822581e8979e1b6e4"
HOST_OBSERVATION_POLICY = "comfy-observed-cpu-coexist-v1"


def small_record(path):
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) >= 100_000 or path.suffix not in {".py", ".json"}:
        raise RuntimeError("only small selected source/JSON metadata may be opened")
    raw.decode("utf-8")
    info = path.stat()
    return {
        "path": str(path),
        "size": len(raw),
        "stat": {key: getattr(info, key) for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns")},
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def build():
    base_record = small_record(BASE / "edge-plan.json")
    if base_record["sha256"] != BASE_PLAN_SHA:
        raise RuntimeError("frozen base native03 plan differs")
    plan = json.loads((BASE / "edge-plan.json").read_text())
    actor = small_record(HERE / "actor.py")
    helper = small_record(HERE / "resource_helper.py")
    audit = small_record(HERE / "source_audit.py")
    controller = small_record(OWNER / "controller.py")
    owner_freeze = small_record(OWNER / "freeze.json")
    tree = ast.parse((HERE / "actor.py").read_text())
    values = {
        target.id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id in {"OWNER_CONTROLLER_SHA", "HELPER_SHA", "HOST_OBSERVATION_POLICY"}
    }
    if values != {
        "OWNER_CONTROLLER_SHA": controller["sha256"],
        "HELPER_SHA": helper["sha256"],
        "HOST_OBSERVATION_POLICY": HOST_OBSERVATION_POLICY,
    }:
        raise RuntimeError("new owner/helper/policy source pins not finalized")
    owner_tree = ast.parse((OWNER / "controller.py").read_text())
    policies = [
        ast.literal_eval(node.value)
        for node in owner_tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "HOST_OBSERVATION_POLICY" for t in node.targets)
    ]
    if policies != [HOST_OBSERVATION_POLICY]:
        raise RuntimeError("selected controller does not implement explicit coexist policy")
    plan.update(
        profile="decision3-edge-publisher-cpu16-batch4-4g-coexist-synthetic-20261024-v1",
        actor_sha256=actor["sha256"],
        helper_sha256=helper["sha256"],
        source_audit_sha256=audit["sha256"],
        owner_unit="jev-d3-llama-cpp-cpu16-4g-coexist-20261024-08.service",
        owner_profile="llama-cpp-cpu16-4g-coexist-command-owned-v1",
        host_observation_policy=HOST_OBSERVATION_POLICY,
        owner_sources=[controller, owner_freeze, actor, helper, audit],
        memory_max_bytes=4294967296,
        startup_available_bytes=30414140212,
        cpu_affinity=list(range(4, 20)),
        cpu_quota_percent=1600,
        threads=16,
        environment={
            **plan["environment"],
            **{
                name: "16"
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
    )
    plan["resource_helper_provenance"] = {
        "path": helper["path"],
        "sha256": helper["sha256"],
        "source": small_record(BASE / "resource_helper.py"),
        "function_bodies": "13 original stdlib APIs unchanged; host_gate retains fullR+entire cap with new4GiB constant and diagnostic",
        "new_policy_constant": HOST_OBSERVATION_POLICY,
        "Comfy_HTTP_observation": "Selected owner08 controller owns strict HTTP idle/busy evidence; child memory helper does not query Comfy",
    }
    plan["preparation"] = {
        "profile_scope": "Distinct owner08 CPU16/4GiB CPU-coexist resource binding; same batch4 model math; no original4core/2GiB or prior fit/quality inheritance",
        "base_plan": base_record,
        "model20_runtime_metadata_copied_exact_from_frozen_native03": True,
        "model_tokenizer_native_payload_reads_or_restats": 0,
        "model_Torch_jobs_or_observer_executed": False,
        "actual_rows_latency_peak_fit": "UNMEASURED",
        "root_exactselection_independent_review_fresh_original_new_profile_admission_required": True,
        "old_original_idle_only_profile_unchanged": True,
    }
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = json.dumps(build(), indent=2, allow_nan=False) + "\n"
    with args.output.open("x") as stream:
        stream.write(raw)


if __name__ == "__main__":
    main()
