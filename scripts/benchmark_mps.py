#!/usr/bin/env python3
"""Profile a real pretrained Unified checkpoint with synchronized device timing."""

from __future__ import annotations

import argparse
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-12B-it")
    parser.add_argument(
        "--revision", help="Hub revision, resolved to an immutable commit before load"
    )
    parser.add_argument("--device", default="mps", choices=("mps", "cpu", "cuda"))
    parser.add_argument(
        "--dtype", default="auto", choices=("auto", "float32", "float16", "bfloat16")
    )
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa"))
    parser.add_argument("--dataset", default="examples/benchmarks/smoke.jsonl")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--limit", type=int, help="Only profile the first N dataset cases")
    parser.add_argument("--max-context", type=int, default=16384)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--output", default="artifacts/mps-profile.json")
    return parser.parse_args(argv)


def main(argv=None):
    # Also support a checkout invocation without an editable package install.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from s1.benchmark import load_cases, write_report
    from s1.profiling import (
        environment_metadata,
        memory_snapshot,
        profile_model,
        synchronize_device,
    )
    from s1.unified import UnifiedDecisionModel

    args = parse_args(argv)
    report = {
        "schema_version": 1,
        "status": "error",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment_metadata(),
        "requested": {
            key: str(value) if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in vars(args).items()
        },
        "load_time_ms": None,
        "cases": [],
    }
    stage = "configuration"
    load_started = None
    try:
        if args.warmup < 0 or args.repeat < 1 or (args.limit is not None and args.limit < 1):
            raise ValueError("warmup must be nonnegative; repeat and limit must be positive")
        if args.max_context < 1:
            raise ValueError("max_context must be positive")
        if args.temperature is not None and (
            not math.isfinite(args.temperature) or args.temperature <= 0
        ):
            raise ValueError("temperature must be positive and finite")
        stage = "dataset"
        cases = load_cases(args.dataset)
        if args.limit is not None:
            cases = cases[: args.limit]
        if any(case.split != "test" for case in cases):
            raise ValueError("profiling requires a test-only dataset")
        stage = "resolve_revision"
        pinned_revision = args.revision
        if not Path(args.model).is_dir():
            from huggingface_hub import HfApi

            pinned_revision = HfApi().model_info(args.model, revision=args.revision).sha
            if not pinned_revision:
                raise ValueError("Hub did not return an immutable model revision")
        report["pinned_revision"] = pinned_revision
        stage = "load_model"
        load_started = time.perf_counter()
        model = UnifiedDecisionModel(
            args.model,
            revision=pinned_revision,
            device=args.device,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
            temperature=args.temperature,
            max_context=args.max_context,
        )
        synchronize_device(model.device)
        report["load_time_ms"] = (time.perf_counter() - load_started) * 1000
        load_started = None
        after_load = memory_snapshot(model.device)
        stage = "profile"
        report.update(
            profile_model(
                model,
                cases,
                warmup=args.warmup,
                repeat=args.repeat,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
        )
        report["memory_snapshots"]["after_load"] = after_load
    except Exception as exc:
        if load_started is not None:
            report["load_time_ms"] = (time.perf_counter() - load_started) * 1000
        report["status"] = "error"
        report["failure"] = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
    write_report(report, args.output)
    print(f"{report['status']}: {args.output}")
    if report["status"] != "ok":
        failure = report["failure"]
        print(f"{failure['stage']}: {failure['type']}: {failure['message']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
