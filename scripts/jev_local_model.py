#!/usr/bin/env python3
"""Run pinned local Gemma decisions in Jev's actual Chromium agent loop."""

from __future__ import annotations

import argparse
import json
import os
import socket
import tempfile
import threading
import time
from importlib import metadata
from pathlib import Path

from jev_browser_benchmark import (
    COMPOSED_OBSERVATION_PROFILE,
    NATIVE_OBSERVATION_PROFILE,
    PRUNED_OBSERVATION_PROFILE,
    TASK_IDS,
    prepare_observation_profile,
    run_benchmark,
    upstream_manifest,
)


def _write_receipt(path, receipt, *, qualified=False):
    payload = json.dumps(receipt, indent=2) + "\n"
    if not qualified:
        path.write_text(payload)
        return
    pending = previous = None
    replaced = False
    primary = None
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        pending = Path(name)
        with os.fdopen(descriptor, "w") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        if path.exists():
            descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.previous.", dir=path.parent)
            os.close(descriptor)
            previous = Path(name)
            previous.unlink()
            os.link(path, previous)
        os.replace(pending, path)
        pending = None
        replaced = True
        os.fsync(directory)
        if previous is not None:
            previous.unlink()
            previous = None
    except BaseException as exc:
        primary = exc
        if replaced:
            try:
                if previous is not None:
                    os.replace(previous, path)
                    previous = None
                else:
                    path.unlink()
                os.fsync(directory)
            except BaseException as rollback:
                exc.add_note("Receipt rollback also failed: " + repr(rollback))
                if previous is not None:
                    exc.add_note("Prior receipt backup retained at " + str(previous))
                    previous = None
        raise
    finally:
        for leftover in (pending, previous):
            if leftover is not None:
                try:
                    leftover.unlink(missing_ok=True)
                except OSError as exc:
                    if primary is None:
                        raise
                    primary.add_note("Receipt temporary-file cleanup also failed: " + repr(exc))
        try:
            os.close(directory)
        except OSError as exc:
            if primary is None:
                raise
            primary.add_note("Receipt directory close also failed: " + repr(exc))


def _finish_qualified_receipt(path, receipt, primary):
    if primary is not None or receipt.get("cleanup_errors"):
        receipt["stage"] = "failed"
    try:
        _write_receipt(path, receipt, qualified=True)
    except (Exception, KeyboardInterrupt) as exc:
        receipt["stage"] = "failed"
        receipt["receipt_write_error"] = {"type": type(exc).__name__, "message": str(exc)}
        notes = getattr(exc, "__notes__", [])
        if notes:
            receipt["receipt_write_error"]["notes"] = list(notes)
        if primary is not None:
            primary.add_note("Final BF16 receipt persistence also failed: " + repr(exc))
            for note in notes:
                primary.add_note("BF16 receipt persistence detail: " + note)
            return primary
        return exc
    return None


def _bf16_evaluation_scope(args):
    return {
        "scope": "actual requested evaluation; historical qualification supplies no quality guarantee",
        "completed_configuration_path": "benchmark/configuration.json",
        "tasks": [] if args.live_only else args.tasks,
        "suite_split": args.suite_split,
        "seed": args.seed,
        "repeats": args.repeats,
        "public_case_ids": args.public_case_ids,
        "travel_challenges_only": args.travel_challenges_only,
        "travel_transfer_only": args.travel_transfer_only,
        "live_only": args.live_only,
        "live_url": args.live_url,
        "live_goal": args.live_goal,
        "live_expected_title": args.live_expected_title,
    }


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
    parser.add_argument("--generated-actions", action="store_true")
    parser.add_argument("--generated-action-grammar", action="store_true")
    parser.add_argument("--generated-action-adapter", type=Path)
    parser.add_argument("--generated-action-qualification", type=Path)
    parser.add_argument("--hierarchical-candidates", action="store_true")
    parser.add_argument("--batch-hierarchy-stages", action="store_true")
    parser.add_argument("--browser-operation-first", action="store_true")
    parser.add_argument("--decision3-native", action="store_true")
    parser.add_argument("--decision-readout")
    parser.add_argument("--prefix-cache", action="store_true")
    parser.add_argument("--prefix-suffix-batch-size", type=int, choices=[1, 2], default=1)
    parser.add_argument(
        "--competitor",
        choices=[
            "decider",
            "jev-omni",
            "d3",
            "d3-flash",
            "d3-mini",
            "d3-nano",
            "d3-lite",
            "d3-edge",
        ],
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
    parser.add_argument(
        "--observation-profile",
        choices=[
            NATIVE_OBSERVATION_PROFILE,
            COMPOSED_OBSERVATION_PROFILE,
            PRUNED_OBSERVATION_PROFILE,
        ],
        default=NATIVE_OBSERVATION_PROFILE,
    )
    parser.add_argument("--observation-profile-helper-sha256")
    parser.add_argument("--live-url")
    parser.add_argument("--live-goal")
    parser.add_argument("--live-expected-title")
    parser.add_argument("--live-only", action="store_true")
    parser.add_argument("--public-cases", nargs="+", dest="public_case_ids")
    args = parser.parse_args(argv)
    try:
        observation_bundle = prepare_observation_profile(
            args.upstream_path, args.observation_profile, args.observation_profile_helper_sha256
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    decision3_names = {"d3", "d3-flash", "d3-mini", "d3-nano", "d3-lite", "d3-edge"}
    if (args.competitor in decision3_names) != args.decision3_native:
        parser.error("Decision3 competitors require the separate --decision3-native profile")
    if args.decision3_native and (
        args.competitor not in decision3_names
        or args.quantization != "none"
        or not args.server_evidence
        or args.hierarchical_candidates
        or args.batch_hierarchy_stages
        or args.browser_operation_first
        or args.generated_actions
        or args.generated_action_grammar
        or args.generated_action_adapter
        or args.generated_action_qualification
        or args.prompt_mode != "native"
        or args.analysis_style != "detailed"
        or args.decision_readout
        or args.prefix_cache
        or args.prefix_suffix_batch_size != 1
        or args.model != "knowlet/Gemma-4-12B-Unified-System-One"
        or args.revision != "a66f836b56605039fe040f330180e336d19b3362"
    ):
        parser.error(
            "decision3-native requires one pinned D3 competitor, fixed BF16 Gemma helper, quantization none and server-evidence without other policies"
        )
    if args.browser_operation_first and (
        args.competitor not in {"decider", "jev-omni"}
        or not args.hierarchical_candidates
        or not args.server_evidence
        or args.batch_hierarchy_stages
        or args.generated_actions
        or args.generated_action_grammar
        or args.generated_action_adapter
        or args.generated_action_qualification
        or args.prompt_mode != "native"
        or args.analysis_style != "detailed"
        or args.decision_readout
        or args.prefix_cache
        or args.prefix_suffix_batch_size != 1
    ):
        parser.error(
            "browser-operation-first requires a native competitor, hierarchical-candidates and server-evidence without other decision policies"
        )
    if args.public_case_ids is not None:
        if (
            args.suite_split != "development"
            or args.travel_challenges_only
            or args.travel_transfer_only
            or args.tasks is not None
            or args.live_only
            or any(
                value is not None
                for value in (args.live_url, args.live_goal, args.live_expected_title)
            )
            or len(set(args.public_case_ids)) != len(args.public_case_ids)
        ):
            parser.error("public-cases requires distinct IDs in a separate development run")
        from jev_public_web_cases import get_public_web_case

        try:
            for case_id in args.public_case_ids:
                get_public_web_case(case_id)
        except ValueError as exc:
            parser.error(str(exc))
    if args.generated_action_adapter and not args.generated_actions:
        parser.error("generated-action-adapter requires --generated-actions")
    if args.generated_action_qualification and (
        not args.generated_action_adapter
        or args.quantization != "none"
        or not args.generated_actions
        or not args.generated_action_grammar
    ):
        parser.error(
            "generated-action-qualification requires a BF16 adapter with generated actions and grammar"
        )
    if args.generated_action_qualification and not args.server_evidence:
        parser.error("BF16 qualification requires --server-evidence")
    if args.generated_action_grammar and not args.generated_actions:
        parser.error("generated-action-grammar requires --generated-actions")
    if args.generated_actions and (
        args.competitor
        or args.prompt_mode != "native"
        or args.analysis_style != "detailed"
        or args.hierarchical_candidates
        or args.batch_hierarchy_stages
        or args.decision_readout
        or args.prefix_cache
        or args.prefix_suffix_batch_size != 1
    ):
        parser.error(
            "generated-actions requires the owned native model without other decision policies"
        )
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
    verified_action_adapter = None
    verified_bf16_qualification = None
    if args.generated_action_qualification:
        # Set before strict verification imports PEFT/Transformers/Hub constants.
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if args.generated_action_adapter:
        from s1.browser_action_adapter import verify_action_adapter

        try:
            if args.generated_action_qualification:
                from s1.browser_action_qualification import verify_bf16_action_qualification

                if (args.model, args.revision) != (
                    "knowlet/Gemma-4-12B-Unified-System-One",
                    "a66f836b56605039fe040f330180e336d19b3362",
                ):
                    parser.error("BF16 qualification requires its pinned checkpoint and revision")
                verified_bf16_qualification = verify_bf16_action_qualification(
                    args.generated_action_adapter, args.generated_action_qualification
                )
                verified_action_adapter = verified_bf16_qualification.artifact
            else:
                verified_action_adapter = verify_action_adapter(
                    args.generated_action_adapter,
                    model_id=args.model,
                    revision=args.revision,
                    quantization=args.quantization,
                )
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
    upstream = upstream_manifest(args.upstream_path)
    destination = Path(args.output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    receipt = {
        "model_requested": args.model,
        "revision_requested": args.revision,
        "quantization": args.quantization,
        "prompt_mode": args.prompt_mode,
        "analysis_style": args.analysis_style,
        "generated_actions": args.generated_actions,
        "generated_action_grammar": args.generated_action_grammar,
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
    observation_options = {}
    if observation_bundle is not None:
        receipt["observation_profile"] = {
            "id": args.observation_profile,
            "expected_materialized_manifest": observation_bundle.materialized_manifest(),
            "scope": "Explicit requested observation profile; no historical score parity claim",
        }
        observation_options = {
            "observation_profile": args.observation_profile,
            "observation_profile_helper_sha256": args.observation_profile_helper_sha256,
        }
    if args.public_case_ids is not None:
        receipt["public_case_ids"] = args.public_case_ids
    if args.browser_operation_first:
        from s1.jev_browser_operation_first import profile_manifest

        receipt["browser_decision_profile"] = profile_manifest()
    if args.decision3_native:
        from s1.jev_decision3_native import profile_manifest

        receipt["browser_decision_profile"] = profile_manifest()
    if verified_action_adapter is not None:
        receipt["generated_action_adapter"] = verified_action_adapter.telemetry()
        receipt["text_helper_policy"] = (
            "same pinned checkpoint; action adapter disabled for field generation"
        )
    if verified_bf16_qualification is not None:
        receipt["bf16_qualification"] = verified_bf16_qualification.telemetry()
        receipt["bf16_evaluation"] = _bf16_evaluation_scope(args)
    receipt_path = destination / "receipt.json"
    qualified = verified_bf16_qualification is not None
    atomic_receipt = qualified or observation_bundle is not None
    _write_receipt(receipt_path, receipt, qualified=atomic_receipt)
    primary_error = receipt_failure = None
    server, thread, listener, application = None, None, None, None
    thread_started = False
    source_manifest = None
    server_journal = None
    source_files = {
        "scripts/jev_local_model.py": Path(__file__),
        "scripts/jev_browser_benchmark.py": Path(run_benchmark.__code__.co_filename),
    }
    if args.public_case_ids is not None:
        source_files["scripts/jev_public_web_cases.py"] = Path(
            get_public_web_case.__code__.co_filename
        )
    started = time.perf_counter()
    try:
        from s1.browser_evidence import capture_runtime_sources, verify_runtime_sources
        from s1.browser_server_evidence import ServerEvidenceJournal

        receipt["stage"] = "source_capture"
        source_manifest = capture_runtime_sources(destination, additional_files=source_files)
        receipt["source_manifest"] = source_manifest
        if verified_bf16_qualification is not None:
            from s1.browser_action_qualification import capture_bf16_qualification

            receipt["bf16_qualification_capture"] = capture_bf16_qualification(
                verified_bf16_qualification, destination / "bf16-qualification"
            )
        if args.server_evidence:
            server_journal = ServerEvidenceJournal(destination / "server-evidence")
            receipt["server_evidence"] = {
                "artifact_path": "server-evidence",
                "scope": "server computation only; HTTP delivery unknown; unfinished requests remain unsealed",
            }
        receipt["stage"] = "dependencies"
        if args.decision3_native:
            from s1.evaluation.decision3 import CHECKPOINTS, require_memory

            receipt["stage"] = "decision3_memory_before_any_weight_read"
            receipt["decision3_memory_before_helper"] = require_memory(
                CHECKPOINTS[args.competitor],
                include_helper=True,
            )
            _write_receipt(receipt_path, receipt, qualified=True)
        import torch
        import uvicorn

        from s1.api import create_app
        from s1.browser_action_adapter import base_field_text
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
        _write_receipt(receipt_path, receipt, qualified=atomic_receipt)
        load_started = time.perf_counter()
        bf16_options = {}
        if verified_bf16_qualification is not None:
            from s1.browser_action_adapter import LORA
            from s1.browser_action_qualification import verify_bf16_base_checkpoint

            os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            receipt["bf16_base_exact_bytes_before_load"] = verify_bf16_base_checkpoint(
                verified_bf16_qualification
            )
            torch.manual_seed(20261014)
            bf16_options["lora"] = {**LORA, "revision": args.revision}
        model = UnifiedDecisionModel(
            args.model,
            revision=args.revision,
            device="cuda",
            precision="bfloat16",
            quantization=args.quantization,
            max_context=8192,
            attn_implementation="sdpa",
            **bf16_options,
        )
        if model.revision != args.revision:
            raise ValueError("loaded checkpoint revision differs from requested immutable revision")
        if verified_action_adapter is not None:
            from s1.browser_action_adapter import attach_action_adapter

            receipt["stage"] = "action_adapter_load"
            if verified_bf16_qualification is not None:
                from s1.browser_action_qualification import attach_qualified_bf16_action_adapter

                receipt["generated_action_adapter"] = attach_qualified_bf16_action_adapter(
                    model, verified_bf16_qualification
                )
                receipt["bf16_exact_initialization"] = model.action_adapter.initialization
                receipt["bf16_initial_invariants"] = model.action_adapter.initial_invariants
            else:
                receipt["generated_action_adapter"] = attach_action_adapter(
                    model, verified_action_adapter
                )
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
            _write_receipt(receipt_path, receipt, qualified=atomic_receipt)
            if args.decision3_native:
                from s1.evaluation.decision3 import Decision3Backend, decision3_spec

                spec = decision3_spec(args.competitor, device="cuda")
                before = receipt["decision3_memory_before_helper"]
                receipt["decision3_memory_before_native_load"] = require_memory(
                    CHECKPOINTS[args.competitor],
                    include_helper=True,
                    helper_already_loaded=True,
                    expected_boot_id=before["boot_id"],
                    expected_memtotal=before["memory_bytes"]["MemTotal"],
                )
                _write_receipt(receipt_path, receipt, qualified=True)
                backend = Decision3Backend(
                    spec,
                    helper_already_loaded=True,
                    expected_boot_id=before["boot_id"],
                    expected_memtotal=before["memory_bytes"]["MemTotal"],
                )
            elif args.competitor == "jev-omni":
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
            if args.generated_actions:
                from s1.jev_generated import POLICY, PROBABILITIES

                cell = f"generated-actions-{args.quantization}"
                receipt["decision_model"].update(
                    policy=POLICY,
                    decoding_constraint=(
                        "legal_action_token_trie_v1" if args.generated_action_grammar else "none"
                    ),
                    probabilities_semantics=PROBABILITIES,
                    max_new_tokens=32,
                    budget_seconds=20,
                )
                if verified_action_adapter is not None:
                    receipt["decision_model"]["action_adapter"] = receipt[
                        "generated_action_adapter"
                    ]
        application = create_app(
            backend,
            jev_compatible=True,
            jev_hierarchical=args.hierarchical_candidates,
            jev_hierarchical_batch=args.batch_hierarchy_stages,
            jev_generated_actions=args.generated_actions,
            jev_generated_action_grammar=args.generated_action_grammar,
            model_identity=cell,
            request_evidence_sink=server_journal,
            **({"jev_browser_operation_first": True} if args.browser_operation_first else {}),
            **({"jev_decision3_native": True} if args.decision3_native else {}),
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
        _write_receipt(receipt_path, receipt, qualified=atomic_receipt)
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
            public_case_ids=args.public_case_ids,
            text_helper=lambda context: (
                base_field_text(model, context)
                if verified_action_adapter is not None
                else generate_field_text(model, context)
            ),
            **observation_options,
        )
        if receipt["summary"].get("observation_profile_failed"):
            raise RuntimeError(
                "Benchmark observation profile capture or lifecycle verification failed"
            )
        receipt["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        receipt["stage"] = "complete"
    except (Exception, KeyboardInterrupt) as exc:
        primary_error = exc
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
        if hasattr(exc, "decision3_memory"):
            receipt["decision3_memory_refusal"] = exc.decision3_memory
    finally:
        cleanup_errors = []

        def cleanup_failure(stage, exc):
            nonlocal primary_error
            cleanup_errors.append({"stage": stage, "message": str(exc)})
            if primary_error is None:
                primary_error = exc

        if source_manifest is not None:
            try:
                receipt["source_verification"] = verify_runtime_sources(
                    source_manifest, additional_files=source_files
                )
                if not receipt["source_verification"]["success"]:
                    raise RuntimeError("runtime source changed during the model campaign")
            except Exception as exc:
                cleanup_failure("source_verification", exc)
        if server is not None:
            try:
                server.should_exit = True
            except Exception as exc:
                cleanup_failure("server_shutdown", exc)
        if thread_started:
            try:
                thread.join(timeout=10)
                if thread.is_alive():
                    raise RuntimeError("owned model API did not stop within 10 seconds")
            except Exception as exc:
                cleanup_failure("thread_shutdown", exc)
        if listener is not None:
            try:
                listener.close()
            except Exception as exc:
                cleanup_failure("socket_shutdown", exc)
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
                    cleanup_failure(
                        "server_evidence",
                        RuntimeError("Server evidence writes failed; see journal index"),
                    )
            except Exception as exc:
                cleanup_failure("server_evidence", exc)
        if verified_bf16_qualification is not None:
            try:
                from s1.browser_action_qualification import (
                    verify_bf16_base_checkpoint,
                    verify_bf16_qualification_capture,
                )

                receipt["bf16_qualification_final_capture"] = verify_bf16_qualification_capture(
                    verified_bf16_qualification, destination / "bf16-qualification"
                )
                evidence = receipt.get("server_evidence", {})
                if not evidence.get("all_requests_sealed") or evidence.get("pending_request_ids"):
                    raise RuntimeError("qualified BF16 server evidence is pending or unsealed")
                if "model" in locals() and hasattr(model, "action_adapter"):
                    receipt["bf16_final_invariants"] = model.action_adapter.check(full=True)
                if "bf16_base_exact_bytes_before_load" in receipt:
                    receipt["bf16_base_exact_bytes_after_run"] = verify_bf16_base_checkpoint(
                        verified_bf16_qualification
                    )
            except Exception as exc:
                cleanup_failure("bf16_qualification_finalization", exc)
        if cleanup_errors:
            receipt["cleanup_errors"] = cleanup_errors
        receipt["elapsed_ms"] = (time.perf_counter() - started) * 1000
        if atomic_receipt or args.decision3_native:
            receipt_failure = _finish_qualified_receipt(receipt_path, receipt, primary_error)
        else:
            _write_receipt(receipt_path, receipt, qualified=args.decision3_native)
    print(json.dumps(receipt, indent=2), flush=True)
    if receipt_failure is not None:
        raise receipt_failure
    if "error" in receipt or "cleanup_errors" in receipt:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
