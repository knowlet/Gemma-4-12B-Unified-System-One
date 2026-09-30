"""Run pinned public text baselines on CPU through the real v2 evaluator.

The caller supplies disjoint public dataset splits and owns the Modal job. Small
matched support sets are screening experiments, not completed learning curves or
production-quality evidence. Failures of one baseline do not skip the others.
"""

from __future__ import annotations

import gc
import os
import re
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from s1.evaluation.adapters import load_adapter
from s1.evaluation.artifacts import read_records, write_json
from s1.evaluation.baselines import state_text
from s1.evaluation.contracts import Budget, ProfileSpec, SuiteSpec
from s1.evaluation.datasets import audit_splits, fingerprint, load_cases
from s1.evaluation.registry import Registry
from s1.evaluation.runner import execute
from s1.evaluation.supervision import fit_baseline, prepare_support

CHECKPOINTS = {
    "embedding": "sentence-transformers/all-MiniLM-L6-v2",
    "nli": "cross-encoder/nli-MiniLM2-L6-H768",
    "cross_encoder": "cross-encoder/ms-marco-MiniLM-L6-v2",
}
REQUEST_COST_CEILING_USD = 0.005


def _error(exc):
    return {"error_type": type(exc).__name__, "error": str(exc)[:1000]}


def _resolve_checkpoint(model_id):
    from huggingface_hub import HfApi
    from transformers import AutoConfig

    revision = HfApi().model_info(model_id).sha
    if not revision or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        raise ValueError("Hugging Face did not return an immutable checkpoint revision")
    config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    return {
        "model_id": model_id,
        "revision": revision,
        "model_type": config.model_type,
        "num_labels": config.num_labels,
        "id2label": config.id2label,
        "max_position_embeddings": getattr(config, "max_position_embeddings", None),
    }


def _fit_setfit(support, directory, checkpoint, *, seed=0, steps=1):
    """Bound the real fitter's training settings and verify both CPU selections."""
    import torch
    from setfit import SetFitModel, Trainer, TrainingArguments

    from_pretrained = SetFitModel.from_pretrained
    original_train = Trainer.train
    training_receipt = {
        "requested_device": "cpu",
        "support_cases": len(load_cases(support)),
        "seed": seed,
        "steps": steps,
        "num_iterations": 1,
        "num_epochs": 1,
        "training_devices": [],
        "optimizer_updates": 0,
        "trainer_global_step": None,
        "trainer_observed_epoch": None,
    }

    def explicit_cpu_model(*args, **kwargs):
        kwargs["device"] = "cpu"
        return from_pretrained(*args, **kwargs)

    def bounded_arguments(*args, **kwargs):
        kwargs.update(num_iterations=1, num_epochs=1, max_steps=steps)
        return TrainingArguments(*args, **kwargs)

    def checked_train(trainer, *args, **kwargs):
        model_device = trainer.model.device.type
        trainer_device = trainer.st_trainer.args.device.type
        training_receipt["training_devices"].append(
            {"model": model_device, "trainer": trainer_device}
        )
        if (model_device, trainer_device) != ("cpu", "cpu"):
            raise ValueError("SetFit model and Trainer must both select CPU")
        original_step = torch.optim.AdamW.step

        def observed_step(optimizer, *step_args, **step_kwargs):
            result = original_step(optimizer, *step_args, **step_kwargs)
            training_receipt["optimizer_updates"] += 1
            return result

        started = time.perf_counter()
        try:
            with patch.object(torch.optim.AdamW, "step", new=observed_step):
                result = original_train(trainer, *args, **kwargs)
            state = trainer.st_trainer.state
            training_receipt.update(
                trainer_global_step=state.global_step,
                trainer_observed_epoch=state.epoch,
            )
            if training_receipt["optimizer_updates"] != steps or state.global_step != steps:
                raise ValueError("observed SetFit optimizer updates differ from requested steps")
            return result
        finally:
            training_receipt["training_wall_seconds"] = time.perf_counter() - started

    try:
        with (
            patch.dict(os.environ, {"ACCELERATE_USE_CPU": "true"}),
            patch.object(torch.cuda, "is_available", return_value=False),
            patch.object(torch.backends.mps, "is_available", return_value=False),
            patch.object(SetFitModel, "from_pretrained", side_effect=explicit_cpu_model),
            patch("setfit.TrainingArguments", side_effect=bounded_arguments),
            patch.object(Trainer, "train", new=checked_train),
        ):
            return fit_baseline(
                support,
                directory,
                method="setfit",
                seed=seed,
                model_id=checkpoint["model_id"],
                revision=checkpoint["revision"],
                steps=steps,
            )
    finally:
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "training-receipt.json", training_receipt)


def _validate_loaded_adapter(adapter, spec):
    """Check actual CPU/FP32 tensors and the resolved public base revision."""
    backend = adapter.backend
    if spec.adapter in ("prior", "tfidf"):
        return {"device": "cpu", "precision": "float64", "kind": "numpy_classifier"}
    model = backend.model
    body = model.model_body if spec.adapter == "setfit" else model
    parameter = next(body.parameters())
    device, precision = parameter.device.type, str(parameter.dtype).removeprefix("torch.")
    if device != "cpu" or precision != "float32":
        raise ValueError("live text baselines require actual CPU float32 parameters")
    encoder = body._first_module().auto_model if spec.adapter in ("embedding", "setfit") else body
    loaded_revision = getattr(encoder.config, "_commit_hash", None)
    if spec.adapter != "setfit" and loaded_revision != spec.revision:
        raise ValueError("loaded baseline checkpoint differs from its pinned revision")
    return {
        "device": device,
        "precision": precision,
        "resolved_base_revision": loaded_revision,
        "local_trained_checkpoint": spec.adapter == "setfit",
    }


def _paired_token_lengths(adapter, spec, cases):
    """Measure exactly the untruncated tokenizer pairs used by BaselineBackend."""
    if spec.adapter not in ("nli", "cross_encoder"):
        return {}
    tokenizer = adapter.backend.tokenizer
    lengths = {}
    for case in cases:
        per_question = {}
        for question in case.request.questions:
            hypotheses = [
                f"{question.instructions}\n{label}: {description}"
                for label, description in zip(question.labels(), question.descriptions())
            ]
            inputs = tokenizer(
                [state_text(case.request.state)] * len(hypotheses),
                hypotheses,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            per_question[question.id] = int(inputs["input_ids"].shape[1])
        lengths[case.id] = {
            "max_paired_input_tokens": max(per_question.values()),
            "questions": per_question,
        }
    return lengths


def _context_outcomes(spec, requests, errors, token_lengths):
    """Expected rejection needs both observed validation error and exact length evidence."""
    by_id = {row["request_id"]: row for row in requests}
    expected, unexpected = [], []
    for error in errors:
        request = by_id.get(error.get("request_id"), {})
        length = token_lengths.get(request.get("case_id"), {}).get("max_paired_input_tokens")
        if (
            spec.adapter in ("nli", "cross_encoder")
            and error.get("error_type") == "RequestValidationError"
            and request.get("status") == "error"
            and request.get("eligibility") == "eligible"
            and length is not None
            and length > spec.context_limit
        ):
            expected.append(
                {
                    **error,
                    "case_id": request["case_id"],
                    "phase": request["phase"],
                    "decisions": request["decisions"],
                    "max_paired_input_tokens": length,
                    "context_limit": spec.context_limit,
                    "outcome": "expected_untruncated_context_rejection",
                }
            )
        else:
            unexpected.append(error)
    expected_ids = {row["request_id"] for row in expected}
    missing_rejections = [
        {
            "request_id": row["request_id"],
            "case_id": row["case_id"],
            "status": row["status"],
            "max_paired_input_tokens": token_lengths[row["case_id"]]["max_paired_input_tokens"],
        }
        for row in requests
        if row["case_id"] in token_lengths
        and token_lengths[row["case_id"]]["max_paired_input_tokens"] > spec.context_limit
        and row["request_id"] not in expected_ids
    ]
    return {
        "model_id": spec.id,
        "checkpoint_revision": spec.revision,
        "context_limit": spec.context_limit,
        "tokenization": "actual_pinned_adapter_tokenizer; paired inputs; truncation=False",
        "input_lengths": token_lengths,
        "expected_rejections": expected,
        "expected_measurement_rejected_decisions": sum(
            row["decisions"] for row in expected if row["phase"] == "measurement"
        ),
        "unexpected_errors": unexpected,
        "missing_expected_rejections": missing_rejections,
        "denominator_policy": "all128 eligible decisions retained in original runner metrics",
    }


def run(root: Path, output: Path, dataset_dir: Path) -> dict:
    import torch

    root, output, dataset_dir = Path(root), Path(output), Path(dataset_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    receipt = {
        "kind": "public_text_pretrained_baseline_screening",
        "status": "running",
        "device": "cpu",
        "models": {},
        "checkpoints": {},
        "failures": [],
        "expected_context_rejections": [],
        "runtime_pass_is_quality_pass": False,
        "limitations": [
            "128-row public test sample is descriptive screening, not production evidence.",
            "Prior, TFIDF and SetFit use the same 8-per-class support at seed 0.",
            "SetFit requests one pairing iteration and one epoch, capped at one actual optimizer update.",
            "This bounded training experiment does not complete a multi-seed learning curve.",
            "Encoder/NLI/cross-encoder rankings do not invent candidate probabilities.",
            "Per-request costs are operator ceilings; cloud billing is recorded by the caller.",
        ],
    }

    def persist():
        receipt["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "receipt.json", receipt)

    def failure(name, phase, exc):
        detail = {"model": name, "phase": phase, **_error(exc)}
        receipt["failures"].append(detail)
        receipt["models"][name] = {"status": "failed", **detail}
        write_json(output / f"{name}-{phase}-error.json", detail)
        persist()

    old_threads = torch.get_num_threads()
    torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
    receipt["torch_cpu_threads"] = torch.get_num_threads()
    try:
        config = root / "configs/benchmarks"
        registry = Registry.load(
            config / "models.toml", config / "suites.toml", config / "profiles.toml"
        )
        test_path, train_path = dataset_dir / "text-test.jsonl", dataset_dir / "text-train.jsonl"
        test_cases = load_cases(test_path)
        train_cases = load_cases(train_path)
        if len(test_cases) != 128:
            raise ValueError("baseline screening requires the prepared 128-row public test split")
        heldout = [test_path]
        calibration = dataset_dir / "text-calibration.jsonl"
        if calibration.exists():
            heldout.append(calibration)
        split_audit = audit_splits([train_path, *heldout])
        write_json(output / "split-audit.json", split_audit)
        if not split_audit["valid"]:
            raise ValueError("public baseline train/test splits overlap")
        receipt["datasets"] = {
            "train_cases": len(train_cases),
            "train_sha256": fingerprint(train_cases),
            "test_cases": len(test_cases),
            "test_sha256": fingerprint(test_cases),
            "split_audit_valid": True,
        }
        support_path = None
        try:
            support_manifest = prepare_support(
                train_path, heldout, output / "support", shots=(8,), seeds=(0, 1, 2)
            )
            selected = support_manifest["cells"][0]
            support_path = output / "support" / selected["dataset"]
            receipt["selected_support"] = selected
            receipt["support_seeds_prepared_but_not_fitted"] = [1, 2]
        except Exception as exc:
            for name in ("prior", "tfidf", "setfit"):
                failure(name, "support_preparation", exc)
        persist()

        for name, model_id in CHECKPOINTS.items():
            try:
                print(f"LIVE_BASELINE resolve {name}: {model_id}", flush=True)
                receipt["checkpoints"][name] = _resolve_checkpoint(model_id)
                write_json(output / f"{name}-checkpoint.json", receipt["checkpoints"][name])
            except Exception as exc:
                failure(name, "revision_resolution", exc)
                if name == "embedding":
                    failure("setfit", "revision_resolution", exc)
            persist()

        for name in ("prior", "tfidf", "embedding", "nli", "cross_encoder", "setfit"):
            try:
                print(f"LIVE_BASELINE evaluate {name}", flush=True)
                spec = registry.models[name].model_copy(
                    update={
                        "enabled": True,
                        "device": "cpu",
                        "context_limit": 512,
                        "cost_per_request_usd": REQUEST_COST_CEILING_USD,
                    }
                )
                if name in ("prior", "tfidf", "setfit"):
                    if support_path is None:
                        continue
                    if name == "setfit":
                        checkpoint = receipt["checkpoints"].get("embedding")
                        if not checkpoint:
                            continue
                        trained = _fit_setfit(support_path, output / "setfit-fit", checkpoint)
                    else:
                        trained = fit_baseline(support_path, output / f"{name}-fit", method=name)
                    spec = spec.model_copy(
                        update={
                            "artifact_file": trained["artifact_file"],
                            "artifact_sha256": trained["artifact_sha256"],
                            "revision": trained["artifact_sha256"],
                            **(
                                {"model_id": str(output / "setfit-fit/model")}
                                if name == "setfit"
                                else {}
                            ),
                        }
                    )
                else:
                    checkpoint = receipt["checkpoints"].get(name)
                    if not checkpoint:
                        continue
                    spec = spec.model_copy(
                        update={
                            "model_id": checkpoint["model_id"],
                            "revision": checkpoint["revision"],
                            "processor_revision": checkpoint["revision"],
                            "context_limit": min(512, checkpoint["max_position_embeddings"] or 512),
                        }
                    )
                suite = SuiteSpec(
                    id="public-text",
                    dataset=str(test_path.resolve()),
                    track="specialist" if name in ("prior", "tfidf", "setfit") else "generalist",
                    information_view="raw_text",
                    dataset_sha256=fingerprint(test_cases),
                )
                profile = ProfileSpec(
                    id=name,
                    suite=suite.id,
                    models=(name,),
                    warmup=1,
                    budget=Budget(
                        max_requests=129,
                        max_estimated_cost_usd=129 * REQUEST_COST_CEILING_USD + 0.01,
                    ),
                )
                configured = replace(
                    registry,
                    models={name: spec},
                    suites={suite.id: suite},
                    profiles={name: profile},
                )
                write_json(
                    output / f"{name}-config.json",
                    {
                        "model": spec.model_dump(mode="json"),
                        "suite": suite.model_dump(mode="json"),
                        "profile": profile.model_dump(mode="json"),
                    },
                )
                token_lengths = {}

                def checked_factory(requested_spec, environment):
                    adapter = None
                    try:
                        adapter = load_adapter(requested_spec, environment)
                        resolved = _validate_loaded_adapter(adapter, requested_spec)
                        write_json(output / f"{name}-resolved-runtime.json", resolved)
                        token_lengths.update(
                            _paired_token_lengths(adapter, requested_spec, test_cases)
                        )
                    except Exception as exc:
                        write_json(output / f"{name}-adapter-load-error.json", _error(exc))
                        if adapter is not None:
                            adapter.close()
                        raise
                    return adapter

                summary = execute(
                    configured,
                    name,
                    output=output / f"{name}-run",
                    lockfile=root / "uv.lock",
                    enable_models=[name],
                    adapter_factory=checked_factory,
                )
                stats = summary["models"][name]
                errors = read_records(output / f"{name}-run", "errors")
                requests = read_records(output / f"{name}-run", "requests")
                context = _context_outcomes(spec, requests, errors, token_lengths)
                write_json(output / f"{name}-context-outcomes.json", context)
                receipt["expected_context_rejections"].extend(context["expected_rejections"])
                contract_pass = (
                    summary["run_status"] in ("completed", "completed_with_errors")
                    and stats["records_complete"]
                    and stats["denominators"]["valid_decisions"]
                    + context["expected_measurement_rejected_decisions"]
                    == stats["denominators"]["eligible_decisions"]
                    and not context["unexpected_errors"]
                    and not context["missing_expected_rejections"]
                    and stats["status"] in ("completed", "completed_with_errors")
                )
                receipt["models"][name] = {
                    "status": summary["run_status"],
                    "response_contract_passed": contract_pass,
                    "probability_contract": spec.capabilities.probabilities,
                    "quality": stats["quality"],
                    "denominators": stats["denominators"],
                    "latency_ms": stats["successful_request_latency_ms"],
                    "systems": stats["systems"],
                    "telemetry": stats["telemetry"],
                    "error_count": len(errors),
                    "expected_context_rejection_count": len(context["expected_rejections"]),
                    "unexpected_error_count": len(context["unexpected_errors"]),
                    "context_outcomes_artifact": f"{name}-context-outcomes.json",
                    "directory": f"{name}-run",
                }
                if not contract_pass:
                    receipt["failures"].append(
                        {
                            "model": name,
                            "phase": "evaluation",
                            "status": summary["run_status"],
                            "execution": stats["status"],
                            "reasons": stats["reasons"],
                            "errors": errors,
                            "missing_expected_rejections": context["missing_expected_rejections"],
                        }
                    )
                persist()
            except Exception as exc:
                failure(name, "fit_or_evaluation", exc)
            finally:
                gc.collect()
        receipt["status"] = "passed" if not receipt["failures"] else "completed_with_failures"
    except Exception as exc:
        receipt["status"] = "setup_failed"
        receipt["failures"].append({"phase": "setup", **_error(exc)})
    finally:
        torch.set_num_threads(old_threads)
        persist()
    return receipt


def run_curve(root: Path, output: Path, dataset_dir: Path) -> dict:
    """Fit the complete matched 8/32/128-support, three-seed baseline curve."""
    import torch
    from huggingface_hub import HfApi

    root, output, dataset_dir = Path(root), Path(output), Path(dataset_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    checkpoint = {
        "model_id": CHECKPOINTS["embedding"],
        "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
    }
    receipt = {
        "kind": "matched_public_text_baseline_learning_curve",
        "status": "running",
        "device": "cpu",
        "checkpoint": checkpoint,
        "shots_per_class": [8, 32, 128],
        "seeds": [0, 1, 2],
        "methods": ["prior", "tfidf", "setfit"],
        "setfit_optimizer_steps_per_cell": 20,
        "expected_fits": 27,
        "fits": [],
        "failures": [],
        "limitations": [
            "The 128-row public held-out sample supports descriptive learning curves only.",
            "Same support IDs/groups/fingerprints are used across all three methods in each cell.",
            "SetFit max_steps=20 governs stopping; configured num_epochs=1 can be exceeded.",
            "Prior and TFIDF are closed-form/sklearn fits; optimizer steps are not comparable FLOPs.",
            "No test case is used for fitting, support selection or hyperparameter selection.",
            "Passing runtime/response contracts does not establish an accuracy threshold.",
        ],
    }

    def persist():
        receipt["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "receipt.json", receipt)

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
    receipt["torch_cpu_threads"] = torch.get_num_threads()
    try:
        config = root / "configs/benchmarks"
        registry = Registry.load(
            config / "models.toml", config / "suites.toml", config / "profiles.toml"
        )
        train_path = dataset_dir / "text-train-full.jsonl"
        test_path = dataset_dir / "text-test.jsonl"
        train_cases, test_cases = load_cases(train_path), load_cases(test_path)
        if len(train_cases) != 256 or len(test_cases) != 128:
            raise ValueError("matched curve requires 256 full training and 128 held-out test cases")
        heldout = [test_path]
        calibration = dataset_dir / "text-calibration.jsonl"
        if calibration.exists():
            heldout.append(calibration)
        audit = audit_splits([train_path, *heldout])
        write_json(output / "split-audit.json", audit)
        if not audit["valid"]:
            raise ValueError("matched baseline curve fitting and held-out sources overlap")
        receipt["datasets"] = {
            "train_cases": len(train_cases),
            "train_sha256": fingerprint(train_cases),
            "test_cases": len(test_cases),
            "test_sha256": fingerprint(test_cases),
            "split_audit_valid": True,
        }
        support = prepare_support(
            train_path, heldout, output / "support", shots=(8, 32, 128), seeds=(0, 1, 2)
        )
        receipt["support_manifest"] = "support/support.json"
        if len(support["cells"]) != 9:
            raise ValueError("matched baseline support manifest must have all nine cells")
        checkpoint_ready = False
        try:
            pinned = HfApi().model_info(checkpoint["model_id"], revision=checkpoint["revision"])
            if pinned.sha != checkpoint["revision"]:
                raise ValueError(
                    "MiniLM source revision differs from its immutable learning-curve pin"
                )
            checkpoint_ready = True
        except Exception as exc:
            receipt["checkpoint_resolution_error"] = _error(exc)
            write_json(output / "checkpoint-resolution-error.json", _error(exc))
        persist()
        for cell in support["cells"]:
            support_path = output / "support" / cell["dataset"]
            actual_support = load_cases(support_path)
            if fingerprint(actual_support) != cell["dataset_sha256"]:
                raise ValueError("matched support changed after preparation")
            for method in ("prior", "tfidf", "setfit"):
                name = f"{cell['id']}-{method}"
                fit_started = time.perf_counter()
                entry = {
                    "id": name,
                    "cell_id": cell["id"],
                    "method": method,
                    "seed": cell["seed"],
                    "shots_per_class": cell["shots_per_class"],
                    "support_sha256": cell["dataset_sha256"],
                    "support_ids": cell["support_ids"],
                    "status": "running",
                    "fit_directory": f"{name}-fit",
                    "evaluation_directory": f"{name}-run",
                }
                receipt["fits"].append(entry)
                persist()
                print(
                    f"LIVE_BASELINE_CURVE {name}: {len(actual_support)} support cases", flush=True
                )
                try:
                    if method == "setfit" and not checkpoint_ready:
                        raise RuntimeError(
                            "SetFit immutable checkpoint could not be resolved; see checkpoint-resolution-error.json"
                        )
                    fit_directory = output / f"{name}-fit"
                    trained = (
                        _fit_setfit(
                            support_path, fit_directory, checkpoint, seed=cell["seed"], steps=20
                        )
                        if method == "setfit"
                        else fit_baseline(
                            support_path, fit_directory, method=method, seed=cell["seed"], steps=20
                        )
                    )
                    entry["fitting_wall_seconds"] = time.perf_counter() - fit_started
                    if trained["support_ids"] != cell["support_ids"]:
                        raise ValueError("fitter used different support IDs from the matched cell")
                    entry.update(
                        artifact_sha256=trained["artifact_sha256"],
                        artifact_file=f"{name}-fit/classifier.json",
                    )
                    if method == "setfit":
                        import json

                        entry["training"] = json.loads(
                            (fit_directory / "training-receipt.json").read_text()
                        )
                    spec = registry.models[method].model_copy(
                        update={
                            "id": name,
                            "enabled": True,
                            "device": "cpu",
                            "context_limit": 512,
                            "revision": trained["artifact_sha256"],
                            "artifact_file": trained["artifact_file"],
                            "artifact_sha256": trained["artifact_sha256"],
                            "cost_per_request_usd": REQUEST_COST_CEILING_USD,
                            **(
                                {"model_id": str(fit_directory / "model")}
                                if method == "setfit"
                                else {}
                            ),
                        }
                    )
                    suite = SuiteSpec(
                        id="public-text",
                        dataset=str(test_path.resolve()),
                        track="specialist",
                        information_view="raw_text",
                        dataset_sha256=fingerprint(test_cases),
                    )
                    profile = ProfileSpec(
                        id=name,
                        suite=suite.id,
                        models=(name,),
                        warmup=1,
                        seed=cell["seed"],
                        budget=Budget(
                            max_requests=129,
                            max_estimated_cost_usd=129 * REQUEST_COST_CEILING_USD + 0.01,
                        ),
                    )
                    configured = replace(
                        registry,
                        models={name: spec},
                        suites={suite.id: suite},
                        profiles={name: profile},
                    )
                    write_json(
                        output / f"{name}-config.json",
                        {
                            "model": spec.model_dump(mode="json"),
                            "suite": suite.model_dump(mode="json"),
                            "profile": profile.model_dump(mode="json"),
                        },
                    )

                    def checked_factory(selected, environment):
                        adapter = None
                        try:
                            adapter = load_adapter(selected, environment)
                            resolved = _validate_loaded_adapter(adapter, selected)
                            write_json(output / f"{name}-resolved-runtime.json", resolved)
                        except Exception as exc:
                            write_json(output / f"{name}-adapter-load-error.json", _error(exc))
                            if adapter is not None:
                                adapter.close()
                            raise
                        return adapter

                    summary = execute(
                        configured,
                        name,
                        output=output / f"{name}-run",
                        lockfile=root / "uv.lock",
                        enable_models=[name],
                        adapter_factory=checked_factory,
                    )
                    stats = summary["models"][name]
                    errors = read_records(output / f"{name}-run", "errors")
                    contract = (
                        summary["run_status"] == "completed"
                        and stats["records_complete"]
                        and stats["denominators"]["valid_decisions"] == 128
                        and stats["denominators"]["eligible_decisions"] == 128
                        and not errors
                    )
                    entry.update(
                        status="passed" if contract else "failed",
                        response_contract_passed=contract,
                        run_status=summary["run_status"],
                        quality=stats["quality"],
                        metrics=stats["metrics"],
                        denominators=stats["denominators"],
                        telemetry=stats["telemetry"],
                        latency_ms=stats["successful_request_latency_ms"],
                        errors=errors,
                        execution_reasons=stats["reasons"],
                    )
                    if not contract:
                        receipt["failures"].append(
                            {
                                "cell": name,
                                "phase": "evaluation",
                                "reasons": stats["reasons"],
                                "errors": errors,
                            }
                        )
                except Exception as exc:
                    entry.update(status="failed", **_error(exc))
                    receipt["failures"].append(
                        {"cell": name, "phase": "fit_or_evaluation", **_error(exc)}
                    )
                    write_json(output / f"{name}-error.json", _error(exc))
                finally:
                    entry["total_wall_seconds"] = time.perf_counter() - fit_started
                    gc.collect()
                    persist()
        if len(receipt["fits"]) != 27 or any(
            entry["status"] != "passed" for entry in receipt["fits"]
        ):
            receipt["status"] = "completed_with_failures"
        else:
            receipt["status"] = "passed"
        receipt["actual_setfit_optimizer_updates"] = sum(
            entry.get("training", {}).get("optimizer_updates", 0) for entry in receipt["fits"]
        )
    except Exception as exc:
        receipt["status"] = "setup_failed"
        receipt["failures"].append({"phase": "setup", **_error(exc)})
        write_json(output / "setup-error.json", _error(exc))
    finally:
        torch.set_num_threads(previous_threads)
        persist()
    return receipt
