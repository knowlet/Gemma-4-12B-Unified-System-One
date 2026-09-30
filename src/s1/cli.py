"""Small command line interface; optional runtimes import only when selected."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .backends import GemmaBackend, HTTPBackend, LayaBackend, UniformBackend
from .benchmark import compare_reports, evaluate, load_cases, write_report
from .contracts import DecisionRequest


def make_backend(args):
    if args.backend == "uniform":
        return UniformBackend()
    if args.backend == "gemma":
        return GemmaBackend(args.model, revision=args.revision, device=args.device)
    if args.backend == "laya":
        return LayaBackend(
            args.model or "convaiinnovations/laya",
            revision=args.revision,
            device=args.device,
            subfolder=args.subfolder,
            max_len=args.max_len,
        )
    if not args.endpoint:
        raise ValueError("--endpoint is required for the HTTP backend")
    return HTTPBackend(
        args.endpoint,
        name=args.name or "http",
        token=os.environ.get(args.token_env),
        supports_media=args.supports_media,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="s1")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("decide", "benchmark", "serve"):
        sub = commands.add_parser(command)
        sub.add_argument("--backend", choices=["gemma", "laya", "http", "uniform"], default="gemma")
        sub.add_argument("--model")
        sub.add_argument("--revision")
        sub.add_argument("--device")
        sub.add_argument("--subfolder")
        sub.add_argument("--max-len", type=int, default=512)
        sub.add_argument("--endpoint")
        sub.add_argument("--name")
        sub.add_argument("--token-env", default="S1_API_KEY")
        sub.add_argument("--supports-media", action="store_true")
        if command == "decide":
            sub.add_argument("request", type=Path)
        elif command == "benchmark":
            sub.add_argument("dataset", type=Path)
            sub.add_argument("--output", type=Path, required=True)
            sub.add_argument("--warmup", type=int, default=0)
        else:
            sub.add_argument("--host", default="127.0.0.1")
            sub.add_argument("--port", type=int, default=8000)
    compare = commands.add_parser("compare")
    compare.add_argument("reports", nargs="+", type=Path)
    evaluation = commands.add_parser("eval", help="offline v2 evaluation planning")
    planning = evaluation.add_subparsers(dest="eval_command", required=True)
    for command in ("preflight", "plan"):
        sub = planning.add_parser(command)
        sub.add_argument("--registry", type=Path, default=Path("configs/benchmarks/models.toml"))
        sub.add_argument("--suites", type=Path, default=Path("configs/benchmarks/suites.toml"))
        sub.add_argument("--profiles", type=Path, default=Path("configs/benchmarks/profiles.toml"))
        sub.add_argument("--profile", required=True)
        sub.add_argument("--lockfile", type=Path, default=Path("uv.lock"))
        sub.add_argument("--enable-model", action="append", default=[])
        sub.add_argument("--output", type=Path)
    train = commands.add_parser("train")
    train.add_argument("--train-data", type=Path, required=True)
    train.add_argument("--calibration-data", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--model", default="google/gemma-4-12B-it")
    train.add_argument("--revision")
    train.add_argument("--device")
    train.add_argument("--steps", type=int, default=100)
    train.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        if args.command == "eval":
            from .evaluation.planning import create_plan
            from .evaluation.registry import Registry

            registry = Registry.load(args.registry, args.suites, args.profiles)
            report = create_plan(
                registry,
                args.profile,
                lockfile=args.lockfile,
                environment=os.environ,
                enable_models=args.enable_model,
                purpose=args.eval_command,
            )
            if args.output:
                write_report(report, args.output)
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0 if report["can_execute"] else 1
        if args.command == "compare":
            print(
                json.dumps(
                    compare_reports([json.loads(p.read_text()) for p in args.reports]), indent=2
                )
            )
            return 0
        if args.command == "train":
            import torch

            from .training import train_model
            from .unified import UnifiedDecisionModel

            train_cases = load_cases(args.train_data)
            calibration_cases = load_cases(args.calibration_data)
            torch.manual_seed(args.seed)
            model = UnifiedDecisionModel(
                args.model,
                revision=args.revision,
                device=args.device,
                lora={
                    "r": 8,
                    "lora_alpha": 16,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                    "bias": "none",
                },
            )
            report = train_model(
                model, train_cases, calibration_cases, steps=args.steps, seed=args.seed
            )
            model.save(args.output)
            write_report(report, args.output / "training.json")
            print(json.dumps(report, indent=2))
            return 0
        # Validate data before loading potentially large checkpoints.
        if args.command == "decide":
            request = DecisionRequest.model_validate_json(args.request.read_text())
        elif args.command == "benchmark":
            cases = load_cases(args.dataset)
            if any(c.split != "test" for c in cases) or args.warmup < 0:
                raise ValueError("benchmark requires a test-only dataset and nonnegative warmup")
        backend = make_backend(args)
        try:
            if args.command == "decide":
                print(json.dumps(backend.predict(request), indent=2))
            elif args.command == "benchmark":
                report = evaluate(backend, cases, warmup=args.warmup)
                write_report(report, args.output)
                print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
                return 1 if report["counts"].get("error") else 0
            else:
                import uvicorn

                from .api import create_app

                uvicorn.run(
                    create_app(backend, api_key=os.environ.get("S1_API_KEY")),
                    host=args.host,
                    port=args.port,
                )
        finally:
            if hasattr(backend, "close"):
                backend.close()
        return 0
    except (ValueError, OSError) as exc:
        parser.exit(2, f"s1: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
