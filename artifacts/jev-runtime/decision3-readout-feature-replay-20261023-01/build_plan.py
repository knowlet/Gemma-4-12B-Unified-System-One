#!/usr/bin/env python3
"""Build only small-source metadata; copy frozen model/runtime records, never open payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NATIVE = ROOT / "artifacts/jev-runtime/decision3-publisher-batch4-cpu-20261023-02"
BASE_SHA = "115b7850ac80b35aa0aacb5880644e1ee7ea79deb59beb748c7947c86d0492e8"


def record(path):
    raw = path.read_bytes()
    if len(raw) >= 100_000 or path.suffix not in {".py", ".json"}:
        raise RuntimeError("only small selected source/JSON metadata may be read")
    raw.decode("utf-8")
    return {"path": str(path), "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def build():
    base_record = record(NATIVE / "edge-plan.json")
    if base_record["sha256"] != BASE_SHA:
        raise RuntimeError("frozen base plan changed")
    base = json.loads((NATIVE / "edge-plan.json").read_text())
    sources = [HERE / name for name in ("actor.py", "observer.py", "fixture.json")]
    sources.extend(
        NATIVE / name
        for name in (
            "actor.py",
            "resource_helper.py",
            "source_audit.py",
            "edge-plan.json",
            "freeze.json",
        )
    )
    rows = [record(path) for path in sources]
    own = {Path(row["path"]).name: row for row in rows[:3]}
    return {
        "schema": "decision3-original-readout-feature-replay-plan-v1",
        "profile": "decision3-edge-original-readout-feature-replay-cpu-2g-20261023-v1",
        "actor_sha256": own["actor.py"]["sha256"],
        "observer_sha256": own["observer.py"]["sha256"],
        "fixture_sha256": own["fixture.json"]["sha256"],
        "native_actor_sha256": base["actor_sha256"],
        "native_plan_path": str(NATIVE / "edge-plan.json"),
        "native_plan_sha256": BASE_SHA,
        "source_records": rows,
        "model_key": "edge",
        "model": base["model"],
        **{
            key: base[key]
            for key in (
                "owner_unit",
                "owner_profile",
                "owner_runtime_seconds",
                "reserve_bytes",
                "memory_max_bytes",
                "startup_available_bytes",
                "swap_max_bytes",
                "cpu_affinity",
                "cpu_quota_percent",
                "runtime_python",
            )
        },
        "requests": 1,
        "head_counts": [2, 3, 4],
        "batch_size_maximum": 4,
        "actual_partial_rows_required": 3,
        "temperature": 1.0,
        "wall_seconds": 120,
        "call_seconds": 20,
        "absolute_tolerance": 1e-6,
        "relative_tolerance": 1e-5,
        "comparison": "abs(replay-native) <= 1e-6 + 1e-5*abs(native); exact ordered argmax",
        "timing_scope": {
            "whole": "execute entry through final state write and watchdog cleanup, followed by a literal monotonic refusal",
            "prediction": "prepare, hook registration, original native call, replay, raw/features/state retention, hook and watchdog cleanup; followed by a literal monotonic refusal",
            "startup_is_not_a_20_second_claim": True,
        },
        "payload_authentication": "Unchanged native02 plan fully authenticates all20 model files+manifest+runtime/owner sources before constructor and again after replay; no payload opened by this builder",
        "feature_commit_status": "PENDING",
        "training_eligible": False,
        "external_acceptance_required": [
            "root exit0",
            "accepted unchanged original owner receipt",
            "authenticated original closure",
            "pre/post full model/manifest/runtime/source auth",
            "independent actual proof review",
        ],
        "semantic_expected_answers": None,
        "benchmark_scores": None,
        "http_browser_or_training_qualification": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan = build()
    raw = json.dumps(plan, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(raw, end="")
    else:
        with args.output.open("x") as stream:
            stream.write(raw)


if __name__ == "__main__":
    main()
