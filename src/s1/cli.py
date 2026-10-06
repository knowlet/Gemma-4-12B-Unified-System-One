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
    if args.backend != "gemma" and (
        args.quantization != "none"
        or args.adapter_path
        or args.precision is not None
        or args.dtype is not None
        or args.attn_implementation is not None
        or getattr(args, "compile_mode", None) not in (None, "none")
    ):
        raise ValueError(
            "--quantization, --adapter-path, --precision, --dtype, --attn-implementation "
            "and --compile-mode require --backend gemma"
        )
    if args.backend == "uniform":
        return UniformBackend()
    if args.backend == "gemma":
        if args.precision is not None and args.dtype not in (None, "auto", args.precision):
            raise ValueError("--precision and --dtype must agree when both are specified")
        if args.quantization != "none" and (
            (args.device not in (None, "auto") and not args.device.startswith("cuda"))
            or args.precision == "float32"
            or args.dtype in ("float32", "float16")
        ):
            raise ValueError("--quantization int8/nf4 requires CUDA and bfloat16 precision")
        runtime_options = {}
        if args.dtype is not None:
            runtime_options["dtype"] = args.dtype
        if args.attn_implementation is not None:
            runtime_options["attn_implementation"] = args.attn_implementation
        compile_mode = getattr(args, "compile_mode", None)
        if compile_mode not in (None, "none"):
            runtime_options["compile_mode"] = compile_mode
        return GemmaBackend(
            args.model,
            revision=args.revision,
            device=args.device,
            precision=args.precision,
            quantization=args.quantization,
            adapter_path=str(args.adapter_path) if args.adapter_path else None,
            **runtime_options,
        )
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
        sub.add_argument("--device", help="Torch device; default: CUDA, MPS, then CPU")
        sub.add_argument(
            "--precision",
            choices=["float32", "bfloat16"],
            help="Legacy Gemma precision selector; use --dtype for MPS or FP16",
        )
        sub.add_argument(
            "--quantization",
            choices=["none", "int8", "nf4"],
            default="none",
            help="Gemma CUDA inference weight format; keeps the candidate head dense",
        )
        sub.add_argument(
            "--adapter-path", type=Path, help="existing Gemma PEFT decision adapter directory"
        )
        sub.add_argument(
            "--dtype",
            choices=["auto", "float32", "float16", "bfloat16"],
            help="Gemma weight dtype; auto selects precision for the device and OS",
        )
        sub.add_argument(
            "--attn-implementation",
            choices=["eager", "sdpa"],
            help="Gemma attention implementation; default: Transformers selection",
        )
        sub.add_argument(
            "--compile-mode",
            choices=["none", "decoder-max-autotune-no-cudagraphs", "decoder-default-dynamic"],
            default="none",
            help="Opt-in torch.compile for text decoder only (research; default none). "
            "Do not use reduce-overhead/CUDA-graphs on 60-716 tok dynamic shapes.",
        )
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
    evaluation = commands.add_parser("eval", help="v2 evaluation planning and execution")
    planning = evaluation.add_subparsers(dest="eval_command", required=True)
    for command in ("preflight", "plan", "run"):
        sub = planning.add_parser(command)
        sub.add_argument("--registry", type=Path, default=Path("configs/benchmarks/models.toml"))
        sub.add_argument("--suites", type=Path, default=Path("configs/benchmarks/suites.toml"))
        sub.add_argument("--profiles", type=Path, default=Path("configs/benchmarks/profiles.toml"))
        sub.add_argument("--profile", required=True)
        sub.add_argument("--lockfile", type=Path, default=Path("uv.lock"))
        sub.add_argument("--enable-model", action="append", default=[])
        sub.add_argument("--output", type=Path, required=command == "run")
    summary = planning.add_parser("summarize", help="recompute summary from saved run records")
    summary.add_argument("directory", type=Path)
    audit = planning.add_parser("audit-splits")
    audit.add_argument("datasets", nargs="+", type=Path)
    comparison = planning.add_parser("compare")
    comparison.add_argument("directories", nargs="+", type=Path)
    comparison.add_argument("--seed", type=int, default=0)
    comparison.add_argument("--resamples", type=int, default=2000)
    comparison.add_argument("--output", type=Path)
    calibration = planning.add_parser("calibrate")
    calibration.add_argument("directory", type=Path)
    calibration.add_argument("--model", required=True)
    calibration.add_argument("--output", required=True, type=Path)
    sweep = planning.add_parser("prepare-sweep")
    sweep.add_argument("--output", required=True, type=Path)
    sweep.add_argument("--cases", type=int, default=8)
    perturb = planning.add_parser("perturb")
    perturb.add_argument("dataset", type=Path)
    perturb.add_argument("--output", required=True, type=Path)
    perturb.add_argument("--seed", type=int, default=0)
    behavior = planning.add_parser("behavior")
    behavior.add_argument("left", type=Path)
    behavior.add_argument("right", type=Path)
    behavior.add_argument("--left-model", required=True)
    behavior.add_argument("--right-model", required=True)
    support = planning.add_parser("prepare-support")
    support.add_argument("train", type=Path)
    support.add_argument("--heldout", nargs="+", required=True, type=Path)
    support.add_argument("--output", required=True, type=Path)
    support.add_argument("--shots", nargs="+", type=int, default=[8, 32, 128])
    fit = planning.add_parser("fit-baseline")
    fit.add_argument("dataset", type=Path)
    fit.add_argument("--output", required=True, type=Path)
    fit.add_argument("--method", choices=["tfidf", "prior", "setfit"], default="tfidf")
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--model")
    fit.add_argument("--revision")
    fit.add_argument("--steps", type=int, default=20)
    importer = planning.add_parser("import-data")
    importer.add_argument("source", type=Path)
    importer.add_argument("--output", required=True, type=Path)
    importer.add_argument("--format", required=True, choices=["boolq", "ocnli", "choice"])
    importer.add_argument("--source-url", required=True)
    importer.add_argument("--revision", required=True)
    importer.add_argument("--license", required=True)
    importer.add_argument("--original-split", required=True)
    importer.add_argument(
        "--split", required=True, choices=["train", "development", "calibration", "test"]
    )
    media = planning.add_parser("prepare-media")
    media.add_argument("bundle", type=Path)
    media.add_argument("--output", required=True, type=Path)
    counterfactual = planning.add_parser("counterfactual")
    counterfactual.add_argument("directory", type=Path)
    counterfactual.add_argument("--model", required=True)
    counterfactual.add_argument("--views", required=True, type=Path)
    report = planning.add_parser("report")
    report.add_argument("directories", nargs="+", type=Path)
    report.add_argument("--output", required=True, type=Path)
    workflow = planning.add_parser("workflow")
    workflow.add_argument("episodes", type=Path)
    workflow.add_argument("--registry", type=Path, default=Path("configs/benchmarks/models.toml"))
    workflow.add_argument("--suites", type=Path, default=Path("configs/benchmarks/suites.toml"))
    workflow.add_argument("--profiles", type=Path, default=Path("configs/benchmarks/profiles.toml"))
    workflow.add_argument("--small", required=True)
    workflow.add_argument("--strong", required=True)
    workflow.add_argument("--enable-model", action="append", default=[])
    workflow.add_argument("--threshold", type=float, default=0.8)
    workflow.add_argument("--max-calls", type=int, default=1000)
    workflow.add_argument("--max-cost-usd", type=float, default=0)
    workflow.add_argument("--lockfile", type=Path, default=Path("uv.lock"))
    workflow.add_argument("--output", required=True, type=Path)
    curve = planning.add_parser("train-curve")
    curve.add_argument("support_manifest", type=Path)
    curve.add_argument("--calibration", required=True, type=Path)
    curve.add_argument("--output", required=True, type=Path)
    curve.add_argument("--model", default="google/gemma-4-12B-it")
    curve.add_argument("--revision", required=True)
    curve.add_argument("--device")
    curve.add_argument("--steps", type=int, default=100)
    curve.add_argument("--max-updates", type=int, default=10000)
    curve.add_argument("--lr", type=float, default=2e-5)
    curve.add_argument("--brier-weight", type=float, default=0.1)
    curve.add_argument("--lockfile", type=Path, default=Path("uv.lock"))
    gate = planning.add_parser("gate")
    gate.add_argument("comparison", type=Path)
    gate.add_argument("--spec", required=True, type=Path)
    train = commands.add_parser("train")
    train.add_argument("--train-data", type=Path, required=True)
    train.add_argument("--calibration-data", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--model", default="google/gemma-4-12B-it")
    train.add_argument("--revision")
    train.add_argument("--device", help="Torch device; default: CUDA, MPS, then CPU")
    train.add_argument(
        "--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto"
    )
    train.add_argument("--attn-implementation", choices=["eager", "sdpa"])
    train.add_argument("--steps", type=int, default=100)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--brier-weight", type=float, default=0.1)
    train.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args(argv)
    try:
        if args.command == "eval":
            if args.eval_command == "gate":
                from .evaluation.gates import assess_gate

                result = assess_gate(args.comparison, args.spec)
                print(json.dumps(result, indent=2))
                return 0 if result["status"] == "passed" else 1
            if args.eval_command == "train-curve":
                from .evaluation.training_curve import train_curve

                result = train_curve(
                    args.support_manifest,
                    args.calibration,
                    args.output,
                    model_id=args.model,
                    revision=args.revision,
                    device=args.device,
                    steps=args.steps,
                    lr=args.lr,
                    brier_weight=args.brier_weight,
                    max_updates=args.max_updates,
                    lockfile=args.lockfile,
                )
                print(json.dumps(result, indent=2))
                return 0
            if args.eval_command == "report":
                from .evaluation.reports import build_report

                print(json.dumps(build_report(args.directories, args.output), indent=2))
                return 0
            if args.eval_command == "workflow":
                from .evaluation.registry import Registry
                from .evaluation.workflow import execute_workflows

                registry = Registry.load(args.registry, args.suites, args.profiles)
                result = execute_workflows(
                    registry,
                    args.episodes,
                    args.small,
                    args.strong,
                    output=args.output,
                    lockfile=args.lockfile,
                    environment=os.environ,
                    enable_models=args.enable_model,
                    threshold=args.threshold,
                    max_calls=args.max_calls,
                    max_cost_usd=args.max_cost_usd,
                )
                print(json.dumps(result, indent=2))
                return 0 if result["status"] == "completed" else 1
            if args.eval_command == "prepare-media":
                from .evaluation.multimodal import prepare_media_views

                print(json.dumps(prepare_media_views(args.bundle, args.output), indent=2))
                return 0
            if args.eval_command == "counterfactual":
                from .evaluation.multimodal import counterfactual_summary

                print(
                    json.dumps(
                        counterfactual_summary(args.directory, args.model, args.views), indent=2
                    )
                )
                return 0
            if args.eval_command == "import-data":
                from .evaluation.importing import import_dataset

                report = import_dataset(
                    args.source,
                    args.output,
                    format=args.format,
                    source_url=args.source_url,
                    revision=args.revision,
                    license=args.license,
                    original_split=args.original_split,
                    split=args.split,
                )
                print(json.dumps(report, indent=2))
                return 0
            if args.eval_command == "prepare-support":
                from .evaluation.supervision import prepare_support

                report = prepare_support(args.train, args.heldout, args.output, shots=args.shots)
                print(json.dumps(report, indent=2))
                return 0
            if args.eval_command == "fit-baseline":
                from .evaluation.supervision import fit_baseline

                report = fit_baseline(
                    args.dataset,
                    args.output,
                    method=args.method,
                    seed=args.seed,
                    model_id=args.model,
                    revision=args.revision,
                    steps=args.steps,
                )
                print(json.dumps(report, indent=2))
                return 0
            if args.eval_command in ("prepare-sweep", "perturb", "behavior"):
                from .evaluation.experiments import (
                    behavior_comparison,
                    perturb_dataset,
                    prepare_sweep,
                )

                if args.eval_command == "prepare-sweep":
                    report = prepare_sweep(args.output, cases_per_cell=args.cases)
                elif args.eval_command == "perturb":
                    report = perturb_dataset(args.dataset, args.output, seed=args.seed)
                else:
                    report = behavior_comparison(
                        args.left, args.right, args.left_model, args.right_model
                    )
                print(json.dumps(report, indent=2, allow_nan=False))
                return 0
            if args.eval_command == "audit-splits":
                from .evaluation.datasets import audit_splits

                report = audit_splits(args.datasets)
                print(json.dumps(report, indent=2))
                return 0 if report["valid"] else 1
            if args.eval_command == "compare":
                from .evaluation.statistics import compare_runs

                report = compare_runs(args.directories, seed=args.seed, resamples=args.resamples)
                if args.output:
                    write_report(report, args.output)
                print(json.dumps(report, indent=2, allow_nan=False))
                return 0
            if args.eval_command == "calibrate":
                from .evaluation.calibration import fit_from_run

                print(
                    json.dumps(
                        fit_from_run(args.directory, args.model, args.output),
                        indent=2,
                        allow_nan=False,
                    )
                )
                return 0
            if args.eval_command == "summarize":
                from .evaluation.reporting import write_summary

                print(json.dumps(write_summary(args.directory), indent=2, allow_nan=False))
                return 0
            from .evaluation.planning import create_plan
            from .evaluation.registry import Registry

            registry = Registry.load(args.registry, args.suites, args.profiles)
            if args.eval_command == "run":
                from .evaluation.runner import execute

                report = execute(
                    registry,
                    args.profile,
                    output=args.output,
                    lockfile=args.lockfile,
                    environment=os.environ,
                    enable_models=args.enable_model,
                )
                print(json.dumps(report, indent=2, allow_nan=False))
                return 0 if report["run_status"] == "completed" else 1
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

            from .evaluation.datasets import load_cases as load_evaluation_cases
            from .training import train_model
            from .unified import UnifiedDecisionModel

            train_cases = load_evaluation_cases(args.train_data)
            calibration_cases = load_evaluation_cases(args.calibration_data)
            torch.manual_seed(args.seed)
            model = UnifiedDecisionModel(
                args.model,
                revision=args.revision,
                device=args.device,
                dtype=args.dtype,
                attn_implementation=args.attn_implementation,
                lora={
                    "r": 8,
                    "lora_alpha": 16,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                    "bias": "none",
                },
            )
            report = train_model(
                model,
                train_cases,
                calibration_cases,
                steps=args.steps,
                seed=args.seed,
                brier_weight=args.brier_weight,
                lr=args.lr,
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
