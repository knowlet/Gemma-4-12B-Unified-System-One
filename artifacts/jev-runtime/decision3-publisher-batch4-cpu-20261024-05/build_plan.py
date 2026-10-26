#!/usr/bin/env python3
"""Prepare a CPU5 plan from authenticated small metadata; never open payloads.

This source has not been executed. Model/tokenizer/runtime records are copied
from native04's selected plan, without opening or restatting their paths. Final
native05 helper/controller pins and the separate owner09 source freeze are
required. A generated plan is not an actual-run, timing or fit qualification.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import os
import stat
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "artifacts/jev-runtime/decision3-publisher-batch4-cpu-20261024-04"
OWNER = ROOT / "artifacts/jev-runtime/llama-cpp-owner-20261024-09"
BASE_PLAN_SHA = "9a1a48e46663240dd05f45a4558f48cfdfe0c4955ee2c2d7a74565da5015950d"
SOURCE_AUDIT_SHA = "ae299d87cbf7338bb9ff29055efc335e96a44b73a29370e08428aad2fb2b68d3"
OWNER_CONTROLLER_SHA = "81846efe2ee0357da07283c1a77a606283a920cbddef668c8f62def291057dee"
OWNER_FREEZE_SHA = "52ac647a80ca1ef0c05acc45bf90c990f3b753b66df216c87cfb2bdf495020ef"
PROFILE = "decision3-edge-publisher-cpu5-fast-same-cluster-batch4-4g-coexist-synthetic-20261024-v1"
OWNER_PROFILE = "llama-cpp-cpu5-fast-same-cluster-4g-coexist-command-owned-v1"
UNIT = "jev-d3-llama-cpp-cpu5-fast-4g-coexist-20261024-09.service"
HOST_OBSERVATION_POLICY = "comfy-observed-cpu-coexist-v1"
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
MAX_SMALL_BYTES = 100_000
STAT_FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns")


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read_small(path):
    """Read one explicitly selected .py/.json source or metadata file, bounded first."""
    path = Path(path).absolute()
    require(path.suffix in {".py", ".json"}, "only small Python/JSON metadata may be opened")
    require(
        ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "ordinary unambiguous small metadata path required",
    )
    initial = path.stat()
    require(
        stat.S_ISREG(initial.st_mode) and 0 < initial.st_size < MAX_SMALL_BYTES,
        "bounded regular source/metadata file required before opening",
    )
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(before.st_mode) and before.st_size == initial.st_size,
            "selected small file changed before reading",
        )
        raw = stream.read(MAX_SMALL_BYTES)
        after = os.fstat(stream.fileno())
    final = path.stat()
    identities = [
        tuple(getattr(info, key) for key in STAT_FIELDS) for info in (initial, before, after, final)
    ]
    require(
        len(raw) == initial.st_size and identities.count(identities[0]) == 4,
        "selected source/metadata changed while reading",
    )
    raw.decode("utf-8")
    return raw, {
        "path": str(path),
        "size": len(raw),
        "stat": {key: getattr(final, key) for key in STAT_FIELDS},
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def literals(raw, names):
    tree = ast.parse(raw.decode("utf-8"))
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    require(target.id not in result, "duplicate selected source constant")
                    result[target.id] = ast.literal_eval(node.value)
    require(set(result) == set(names), "selected source constants missing")
    return result


def validate_owner_freeze(freeze, controller):
    require(
        freeze.get("schema_id") == "cpu5-fast-coexist-owner-source-freeze-v1"
        and freeze.get("status") == "SOURCE_CHECKS_PASS_NO_ACTUAL"
        and freeze.get("profile") == OWNER_PROFILE,
        "separately checked owner09 source freeze required",
    )
    if "unit" in freeze:
        require(freeze["unit"] == UNIT, "owner09 freeze unit differs")
    matches = []
    for row in freeze.get("files", []):
        path = Path(row["path"])
        selected = path if path.is_absolute() else ROOT / path
        if selected.absolute() == Path(controller["path"]):
            matches.append(row)
    require(
        len(matches) == 1
        and matches[0].get("sha256") == controller["sha256"]
        and matches[0].get("size_bytes") == controller["size"],
        "owner09 freeze does not authenticate the actual controller bytes",
    )


def build():
    base_raw, base_record = read_small(BASE / "edge-plan.json")
    require(base_record["sha256"] == BASE_PLAN_SHA, "frozen native04 metadata plan differs")
    base = json.loads(base_raw)
    require(
        base.get("schema") == "decision3-native-batch4-CPU-probe-plan-v1"
        and base.get("model_key") == "edge"
        and base.get("profile")
        == "decision3-edge-publisher-cpu16-batch4-4g-coexist-synthetic-20261024-v1"
        and base.get("memory_max_bytes") == 4294967296
        and base.get("reserve_bytes") == 26119172916
        and base.get("startup_available_bytes") == 30414140212
        and base.get("owner_runtime_seconds") == 10800
        and (
            base.get("wall_seconds"),
            base.get("call_seconds"),
            base.get("batch_size"),
            base.get("calls"),
        )
        == (120, 20, 4, 2)
        and len(base["model_files"]) == 20
        and len(base["runtime_files"]) == 28,
        "original native04 metadata scope differs",
    )
    actor_raw, actor = read_small(HERE / "actor.py")
    helper_raw, helper = read_small(HERE / "resource_helper.py")
    _, audit = read_small(HERE / "source_audit.py")
    _, parent_helper = read_small(BASE / "resource_helper.py")
    controller_raw, controller = read_small(OWNER / "controller.py")
    freeze_raw, owner_freeze = read_small(OWNER / "freeze.json")
    require(
        controller["sha256"] == OWNER_CONTROLLER_SHA and owner_freeze["sha256"] == OWNER_FREEZE_SHA,
        "independently supplied owner09 controller/freeze pins differ",
    )
    validate_owner_freeze(json.loads(freeze_raw), controller)
    require(
        parent_helper["sha256"] == base["helper_sha256"]
        and audit["sha256"] == base["source_audit_sha256"] == SOURCE_AUDIT_SHA,
        "original helper/audit source identity differs",
    )
    require(
        literals(
            actor_raw,
            {
                "HELPER_SHA",
                "OWNER_CONTROLLER_SHA",
                "SOURCE_AUDIT_SHA",
                "UNIT",
                "OWNER_PROFILE",
                "HOST_OBSERVATION_POLICY",
                "MEMORY_MAX",
                "RESERVE",
                "WALL_SECONDS",
                "CALL_SECONDS",
                "BATCH_SIZE",
                "CALLS",
            },
        )
        == {
            "HELPER_SHA": helper["sha256"],
            "OWNER_CONTROLLER_SHA": controller["sha256"],
            "SOURCE_AUDIT_SHA": SOURCE_AUDIT_SHA,
            "UNIT": UNIT,
            "OWNER_PROFILE": OWNER_PROFILE,
            "HOST_OBSERVATION_POLICY": HOST_OBSERVATION_POLICY,
            "MEMORY_MAX": 4294967296,
            "RESERVE": 26119172916,
            "WALL_SECONDS": 120,
            "CALL_SECONDS": 20,
            "BATCH_SIZE": 4,
            "CALLS": 2,
        },
        "native05 selected helper/controller pins or unchanged budgets not finalized",
    )
    require(
        literals(controller_raw, {"UNIT", "PROFILE", "HOST_OBSERVATION_POLICY", "CPU_ALLOWED_LIST"})
        == {
            "UNIT": UNIT,
            "PROFILE": OWNER_PROFILE,
            "HOST_OBSERVATION_POLICY": HOST_OBSERVATION_POLICY,
            "CPU_ALLOWED_LIST": "15-19",
        },
        "owner09 selected source profile/affinity/policy differs",
    )
    helper_values = literals(
        helper_raw, {"START_ENV", "CPU_ENV", "HOST_OBSERVATION_POLICY", "MEMORY_MAX", "RESERVE"}
    )
    environment = {**base["environment"], **dict.fromkeys(THREAD_VARIABLES, "5")}
    require(
        helper_values
        == {
            "START_ENV": {
                key: value
                for key, value in environment.items()
                if key
                in {
                    "CUDA_VISIBLE_DEVICES",
                    "HF_HUB_OFFLINE",
                    "TRANSFORMERS_OFFLINE",
                    "ACCELERATE_USE_CPU",
                }
            },
            "CPU_ENV": {
                key: value
                for key, value in environment.items()
                if key
                not in {
                    "CUDA_VISIBLE_DEVICES",
                    "HF_HUB_OFFLINE",
                    "TRANSFORMERS_OFFLINE",
                    "ACCELERATE_USE_CPU",
                }
            },
            "HOST_OBSERVATION_POLICY": HOST_OBSERVATION_POLICY,
            "MEMORY_MAX": 4294967296,
            "RESERVE": 26119172916,
        },
        "helper differs beyond the original four CPU thread settings",
    )
    plan = copy.deepcopy(base)
    plan.update(
        profile=PROFILE,
        actor_sha256=actor["sha256"],
        helper_sha256=helper["sha256"],
        source_audit_sha256=audit["sha256"],
        owner_unit=UNIT,
        owner_profile=OWNER_PROFILE,
        host_observation_policy=HOST_OBSERVATION_POLICY,
        owner_sources=[controller, owner_freeze, actor, helper, audit],
        cpu_affinity=[15, 16, 17, 18, 19],
        cpu_quota_percent=500,
        threads=5,
        environment=environment,
    )
    plan["resource_helper_provenance"] = {
        "path": helper["path"],
        "sha256": helper["sha256"],
        "source": parent_helper,
        "function_bodies": "Original native04 stdlib APIs unchanged; only owner label and four thread environment values differ",
        "new_policy_constant": HOST_OBSERVATION_POLICY,
        "Comfy_HTTP_observation": "Selected owner09 controller owns strict known HTTP idle/busy evidence; child memory helper does not query Comfy",
    }
    plan["preparation"] = {
        "profile_scope": "Distinct owner09 CPU15-19/500%/threads5/4GiB resource binding; same native batch4 model math; no inherited CPU16/CPU4 qualification",
        "base_plan": base_record,
        "model20_manifest_runtime_metadata_and_original_request_copied_exact_from_frozen_native04": True,
        "model_tokenizer_native_payload_reads_or_restats": 0,
        "model_Torch_jobs_or_observer_executed": False,
        "actual_rows_latency_peak_fit": "UNMEASURED",
        "root_exactselection_independent_review_fresh_original_new_profile_admission_required": True,
        "old_profiles_unchanged": True,
    }
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.output.suffix == ".json", "metadata plan output must be JSON")
    raw = json.dumps(build(), indent=2, allow_nan=False) + "\n"
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
