#!/usr/bin/env python3
"""Run pinned local Gemma decisions in Jev's actual Chromium agent loop."""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from importlib import metadata
from pathlib import Path

from jev_browser_benchmark import TASK_IDS, run_benchmark, upstream_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="knowlet/Gemma-4-12B-Unified-System-One")
    parser.add_argument("--revision", default="a66f836b56605039fe040f330180e336d19b3362")
    parser.add_argument("--quantization", choices=["none", "nf4", "int8"], default="nf4")
    parser.add_argument(
        "--prompt-mode",
        choices=["native", "user", "reasoned", "grounded", "goal-first"],
        default="native",
    )
    parser.add_argument("--analysis-style", choices=["detailed", "compact"], default="detailed")
    parser.add_argument("--hierarchical-candidates", action="store_true")
    parser.add_argument("--batch-hierarchy-stages", action="store_true")
    parser.add_argument("--decision-readout")
    parser.add_argument("--prefix-cache", action="store_true")
    parser.add_argument("--prefix-suffix-batch-size", type=int, choices=[1, 2], default=1)
    parser.add_argument(
        "--competitor",
        choices=["decider", "jev-omni"],
        help="Use the pinned competitor for decisions and the same Gemma checkpoint for text",
    )
    parser.add_argument("--suite-split", choices=["development", "holdout"], default="development")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--travel-challenges-only", action="store_true")
    parser.add_argument("--travel-transfer-only", action="store_true")
    parser.add_argument("--server-evidence", action="store_true")
    parser.add_argument("--chromium-executable")
    parser.add_argument("--live-url")
    parser.add_argument("--live-goal")
    parser.add_argument("--live-expected-title")
    parser.add_argument("--live-only", action="store_true")
    args = parser.parse_args(argv)
    if args.travel_challenges_only and (
        args.suite_split != "development"
        or args.travel_transfer_only
        or args.tasks is not None
        or args.live_only
        or any(
            value is not None for value in (args.live_url, args.live_goal, args.live_expected_title)
        )
    ):
        parser.error(
            "travel-challenges-only requires development and cannot use tasks or live flags"
        )
    if args.travel_transfer_only and (
        args.suite_split != "development"
        or args.travel_challenges_only
        or args.tasks is not None
        or args.live_only
        or any(
            value is not None for value in (args.live_url, args.live_goal, args.live_expected_title)
        )
    ):
        parser.error(
            "travel-transfer-only requires development and cannot mix suites, tasks or live flags"
        )
    if args.repeats < 1 or (args.tasks and set(args.tasks) - set(TASK_IDS)):
        parser.error("repeats must be positive and every task must belong to the frozen suite")
    if args.analysis_style == "compact" and args.prompt_mode != "reasoned":
        parser.error("compact analysis-style requires reasoned prompt-mode")
    if args.batch_hierarchy_stages and not args.hierarchical_candidates:
        parser.error("batch-hierarchy-stages requires hierarchical-candidates")
    if args.decision_readout and (
        args.competitor or args.prompt_mode not in {"user", "grounded", "goal-first"}
    ):
        parser.error("decision-readout requires an owned user/grounded/goal-first decision policy")
    if args.prefix_cache and (
        args.competitor or args.prompt_mode not in {"user", "grounded", "goal-first"}
    ):
        parser.error("prefix-cache requires an owned user/grounded/goal-first decision policy")
    if args.prefix_suffix_batch_size != 1 and not args.prefix_cache:
        parser.error("prefix-suffix-batch-size=2 requires prefix-cache")
    if args.live_only and (not args.live_url or args.tasks):
        parser.error("live-only requires live-url and cannot be combined with fixture tasks")
    if args.live_url and not (args.live_goal and args.live_expected_title):
        parser.error("live-url requires live-goal and live-expected-title")
    upstream = upstream_manifest(args.upstream_path)
    destination = Path(args.output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    receipt = {
        "model_requested": args.model,
        "revision_requested": args.revision,
        "quantization": args.quantization,
        "prompt_mode": args.prompt_mode,
        "analysis_style": args.analysis_style,
        "hierarchical_candidates": args.hierarchical_candidates,
        "batch_hierarchy_stages": args.batch_hierarchy_stages,
        "decision_readout_requested": args.decision_readout,
        "prefix_cache": args.prefix_cache,
        "prefix_suffix_batch_size": args.prefix_suffix_batch_size,
        "seed": args.seed,
        "travel_challenges_only": args.travel_challenges_only,
        "travel_transfer_only": args.travel_transfer_only,
        "competitor": args.competitor,
        "live_url": args.live_url,
        "live_goal": args.live_goal,
        "live_expected_title": args.live_expected_title,
        "text_helper_policy": "same pinned checkpoint, greedy native generation",
        "stage": "dependencies",
        "cost_usd": None,
        "upstream": upstream,
    }
    (destination / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    server, thread, listener, application = None, None, None, None
    thread_started = False
    source_manifest = None
    server_journal = None
    source_files = {
        "scripts/jev_local_model.py": Path(__file__),
        "scripts/jev_browser_benchmark.py": Path(run_benchmark.__code__.co_filename),
    }
    started = time.perf_counter()
    try:
        from s1.browser_evidence import capture_runtime_sources, verify_runtime_sources
        from s1.browser_server_evidence import ServerEvidenceJournal

        receipt["stage"] = "source_capture"
        source_manifest = capture_runtime_sources(destination, additional_files=source_files)
        receipt["source_manifest"] = source_manifest
        if args.server_evidence:
            server_journal = ServerEvidenceJournal(destination / "server-evidence")
            receipt["server_evidence"] = {
                "artifact_path": "server-evidence",
                "scope": "server computation only; HTTP delivery unknown; unfinished requests remain unsealed",
            }
        receipt["stage"] = "dependencies"
        import torch
        import uvicorn

        from s1.api import create_app
        from s1.browser_policy import BrowserDecisionBackend
        from s1.browser_text import generate_field_text
        from s1.unified import UnifiedDecisionModel

        receipt["versions"] = {
            name: metadata.version(name)
            for name in ("torch", "transformers", "bitsandbytes", "playwright")
        }
        receipt["gpu"] = torch.cuda.get_device_name()
        torch.backends.cuda.enable_cudnn_sdp(False)
        receipt["attention_policy"] = {
            "implementation": "sdpa",
            "cudnn_sdp_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
            "scope": "owned benchmark process, identical policy for every comparison arm",
        }
        receipt["stage"] = "model_load"
        (destination / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        load_started = time.perf_counter()
        model = UnifiedDecisionModel(
            args.model,
            revision=args.revision,
            device="cuda",
            precision="bfloat16",
            quantization=args.quantization,
            max_context=8192,
            attn_implementation="sdpa",
        )
        if model.revision != args.revision:
            raise ValueError("loaded checkpoint revision differs from requested immutable revision")
        receipt["model_reported"] = model.name
        receipt["revision_reported"] = model.revision
        receipt["model_load_ms"] = (time.perf_counter() - load_started) * 1000
        receipt["setup_through_model_load_ms"] = (time.perf_counter() - started) * 1000
        receipt["temperature"] = model.temperature
        receipt["quantization_details"] = model.quantization_details
        if args.competitor:
            from s1.evaluation.local_competitors import (
                LocalCompetitorBackend,
                local_competitor_spec,
            )

            receipt["stage"] = "competitor_load"
            (destination / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            if args.competitor == "jev-omni":
                from s1.evaluation.contracts import Capabilities, ModelSpec
                from s1.evaluation.jev_omni import (
                    CONTEXT_LIMIT,
                    MAX_OPTIONS,
                    MODEL_ID,
                    REVISION,
                    JevOmniBackend,
                )

                spec = ModelSpec(
                    id="jev-omni-local",
                    adapter="jev_omni",
                    enabled=True,
                    model_id=MODEL_ID,
                    revision=REVISION,
                    precision="bfloat16",
                    device="cuda",
                    execution_mode="sequential",
                    calibration="checkpoint",
                    context_limit=CONTEXT_LIMIT,
                    capabilities=Capabilities(
                        modalities=["text", "image", "audio"],
                        primitives=["choice", "noul", "score"],
                        max_options=MAX_OPTIONS,
                        max_questions=64,
                        probabilities="complete",
                        batch="loop_emulated",
                        concurrent_requests=False,
                        independent_questions=True,
                    ),
                )
                backend = JevOmniBackend(spec)
            else:
                spec = local_competitor_spec(args.competitor, device="cuda")
                backend = LocalCompetitorBackend(spec)
            backend.name = spec.id
            receipt["decision_model"] = {
                **backend.telemetry(),
                "model": spec.model_id,
                "model_identity": spec.id,
            }
            cell = spec.id
        else:
            decision_readout = None
            if args.decision_readout:
                from s1.browser_readout import load_readout

                prompt_backend = BrowserDecisionBackend(
                    model, prompt_mode=args.prompt_mode, analysis_style=args.analysis_style
                )
                decision_readout = load_readout(
                    args.decision_readout, model, prompt_backend.prompt_variant
                )
            backend = BrowserDecisionBackend(
                model,
                prompt_mode=args.prompt_mode,
                analysis_style=args.analysis_style,
                decision_readout=decision_readout,
                prefix_cache=args.prefix_cache,
                **(
                    {"prefix_suffix_batch_size": args.prefix_suffix_batch_size}
                    if args.prefix_suffix_batch_size != 1
                    else {}
                ),
            )
            receipt["decision_model"] = {
                "model": model.name,
                "revision": model.revision,
                "prompt_mode": args.prompt_mode,
                "analysis_style": args.analysis_style,
                "prompt_variant": backend.prompt_variant,
                "prefix_cache": args.prefix_cache,
                "prefix_suffix_batch_size": args.prefix_suffix_batch_size,
            }
            if decision_readout is not None:
                receipt["decision_model"]["decision_readout"] = decision_readout.telemetry()
            cell = f"{args.prompt_mode}-{args.quantization}"
        application = create_app(
            backend,
            jev_compatible=True,
            jev_hierarchical=args.hierarchical_candidates,
            jev_hierarchical_batch=args.batch_hierarchy_stages,
            model_identity=cell,
            request_evidence_sink=server_journal,
        )
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(application, log_level="error"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        receipt["stage"] = "server_start"
        thread.start()
        thread_started = True
        deadline = time.monotonic() + 15
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        if not server.started:
            raise RuntimeError("local model API failed to start")
        receipt["stage"] = "browser_benchmark"
        (destination / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        receipt["summary"] = run_benchmark(
            upstream_path=args.upstream_path,
            endpoint=f"http://127.0.0.1:{port}/v1/systemone",
            model=cell,
            output_dir=destination / "benchmark",
            repeats=args.repeats,
            seed=args.seed,
            suite_split=args.suite_split,
            task_ids=[] if args.live_only else args.tasks,
            travel_challenges_only=args.travel_challenges_only,
            travel_transfer_only=args.travel_transfer_only,
            chromium_executable=args.chromium_executable,
            live_url=args.live_url,
            live_goal=args.live_goal,
            live_expected_title=args.live_expected_title,
            text_helper=lambda context: generate_field_text(model, context),
        )
        receipt["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        receipt["stage"] = "complete"
    except (Exception, KeyboardInterrupt) as exc:
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        cleanup_errors = []
        if source_manifest is not None:
            try:
                receipt["source_verification"] = verify_runtime_sources(
                    source_manifest, additional_files=source_files
                )
                if not receipt["source_verification"]["success"]:
                    raise RuntimeError("runtime source changed during the model campaign")
            except Exception as exc:
                cleanup_errors.append({"stage": "source_verification", "message": str(exc)})
        if server is not None:
            try:
                server.should_exit = True
            except Exception as exc:
                cleanup_errors.append({"stage": "server_shutdown", "message": str(exc)})
        if thread_started:
            try:
                thread.join(timeout=10)
                if thread.is_alive():
                    raise RuntimeError("owned model API did not stop within 10 seconds")
            except Exception as exc:
                cleanup_errors.append({"stage": "thread_shutdown", "message": str(exc)})
        if listener is not None:
            try:
                listener.close()
            except Exception as exc:
                cleanup_errors.append({"stage": "socket_shutdown", "message": str(exc)})
        if server_journal is not None:
            try:
                index = server_journal.snapshot()
                state = getattr(application, "state", None)
                index["observer_errors"] = list(getattr(state, "request_evidence_errors", []))
                failed_ids = {item.get("request_id") for item in index["observer_errors"]}
                for row in index["requests"]:
                    if row["request_id"] in failed_ids:
                        row.update(observer_failed=True, sealed=False)
                index["all_requests_sealed"] = not index["observer_errors"] and all(
                    row["sealed"] for row in index["requests"]
                )
                timings = list(getattr(state, "request_evidence_timings", []))
                index["observer_calls"] = len(timings)
                index["observer_elapsed_ms"] = sum(item["elapsed_ms"] for item in timings)
                index["observer_timing_scope"] = (
                    "synchronous API observer callbacks; hierarchy checkpoints count toward its unchanged wall-clock budget"
                )
                index_path = destination / "server-evidence" / "index.json"
                index_path.write_text(json.dumps(index, indent=2) + "\n")
                receipt["server_evidence"].update(index_path="server-evidence/index.json", **index)
                if index["errors"] or index["observer_errors"]:
                    cleanup_errors.append(
                        {
                            "stage": "server_evidence",
                            "message": "Server evidence writes failed; see journal index",
                        }
                    )
            except Exception as exc:
                cleanup_errors.append({"stage": "server_evidence", "message": str(exc)})
        if cleanup_errors:
            receipt["cleanup_errors"] = cleanup_errors
        receipt["elapsed_ms"] = (time.perf_counter() - started) * 1000
        (destination / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)
    if "error" in receipt or "cleanup_errors" in receipt:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
