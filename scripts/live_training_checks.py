"""Bounded, real-checkpoint LoRA training and adapter-persistence screening.

Defaults use two optimizer updates per matched cell for a quick execution check.
Explicit shots/steps also run a full matched support curve on the pinned backbone.
Held-out observations do not establish broad or production quality.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import time
import traceback
from pathlib import Path


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_records(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in records), encoding="utf-8"
    )


def _weight_sample(model):
    """Hash at most 64 deterministic values per LoRA tensor, never clone the base."""
    import torch

    digest = hashlib.sha256()
    names, parameters = [], 0
    for name, parameter in sorted(model.lm.named_parameters()):
        if "lora_" not in name:
            continue
        flat = parameter.detach().reshape(-1)
        indices = torch.linspace(
            0, flat.numel() - 1, min(64, flat.numel()), device=flat.device
        ).long()
        values = flat.index_select(0, indices).float().cpu().numpy().tobytes()
        digest.update(name.encode())
        digest.update(str(tuple(parameter.shape)).encode())
        digest.update(values)
        names.append(name)
        parameters += parameter.numel()
    if not names:
        raise AssertionError("no LoRA weights were available to sample")
    return {"sha256": digest.hexdigest(), "tensors": len(names), "parameters": parameters}


def _predict_records(model, cases, model_id):
    from s1.backends import normalize_response
    from s1.evaluation.datasets import request_fingerprint
    from s1.evaluation.metrics import quality_metrics

    started = time.perf_counter()
    records, predictions = [], {}
    for case in cases:
        response = normalize_response(case.request, model.predict(case.request))
        predictions[case.id] = response["answers"]
        for question in case.request.questions:
            answer = response["answers"][question.id]
            labels = question.labels()
            distribution = answer["probabilities"]
            argmax = max(labels, key=distribution.__getitem__)
            row = {
                **case.metadata(),
                "model_id": model_id,
                "request_id": case.id,
                "request_sha256": request_fingerprint(case.request),
                "question_id": question.id,
                "type": question.type,
                "labels": labels,
                "gold": case.gold[question.id],
                "actual_label": answer.get("choice", argmax),
                "standardized_argmax": argmax,
                "probabilities": distribution,
                "status": "ok",
                "eligibility": "eligible",
                "task_id": case.task_id,
                "schema_id": case.schema_id,
                "temperature": model.temperature,
            }
            if question.type == "score":
                row.update(
                    score=answer["score"],
                    gold_score=question.score_values()[labels.index(case.gold[question.id])],
                )
            records.append(row)
    metrics = quality_metrics(records)
    return (
        {
            "status": "completed",
            "cases": len(cases),
            "decisions": len(records),
            "accuracy": sum(r["actual_label"] == r["gold"] for r in records) / len(records),
            "nll": metrics["nll"],
            "brier_sum": metrics["brier_sum"],
            "ece_15": metrics["ece_15"],
            "wall_seconds": time.perf_counter() - started,
            "temperature": model.temperature,
        },
        records,
        predictions,
    )


def _compare_answers(expected, actual, *, tolerance=2e-6):
    maximum = 0.0
    if expected.keys() != actual.keys():
        raise AssertionError("fresh adapter reload changed question IDs")
    for question_id, left in expected.items():
        right = actual[question_id]
        if left["probabilities"].keys() != right["probabilities"].keys():
            raise AssertionError("fresh adapter reload changed candidate labels")
        for label, value in left["probabilities"].items():
            difference = abs(value - right["probabilities"][label])
            maximum = max(maximum, difference)
            if difference > tolerance:
                raise AssertionError(f"adapter reload probability difference {difference:g}")
        for field in ("choice", "level"):
            if left.get(field) != right.get(field):
                raise AssertionError(f"fresh adapter reload changed {field}")
    return maximum


def run(
    root: Path,
    output: Path,
    dataset_dir: Path,
    model_id: str,
    revision: str,
    *,
    shots=(2, 8),
    steps=2,
    train_filename="text-train.jsonl",
):
    """Run three seeds x requested support sizes x matched CE/CE+Brier on CUDA.

    The caller must release its inference checkpoint first. A single frozen base
    is shared between cells, with adapters unloaded without merging after every
    persistence check. All CUDA model references are released before returning.
    """
    import torch
    from peft import PeftModel, get_peft_model
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    import s1.training as training_module
    from s1.checkpoint import load_temperature
    from s1.evaluation.checkpoints import directory_digest
    from s1.evaluation.datasets import audit_splits, fingerprint, load_cases, request_fingerprint
    from s1.evaluation.supervision import prepare_support
    from s1.evaluation.training_curve import train_curve
    from s1.unified import UnifiedDecisionModel

    root, output, dataset_dir = Path(root), Path(output), Path(dataset_dir)
    shots = tuple(shots)
    if (
        not shots
        or len(set(shots)) != len(shots)
        or any(type(value) is not int or value < 1 for value in shots)
        or type(steps) is not int
        or steps < 1
        or Path(train_filename).name != train_filename
    ):
        raise ValueError(
            "distinct positive support sizes, positive steps and a dataset filename required"
        )
    expected_cells = len(shots) * 3 * 2
    expected_updates = expected_cells * steps
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "training-checks.json"
    started = time.perf_counter()
    paths = {
        key: dataset_dir / f"{key}.jsonl"
        for key in ("text-train", "text-calibration", "text-test", "media-test")
    }
    paths["text-train"] = dataset_dir / train_filename
    report = {
        "status": "running",
        "scope": "matched full-checkpoint CE/CE+Brier support curve and adapter persistence",
        "base_model": model_id,
        "base_revision": revision,
        "device": "cuda",
        "precision": "bfloat16",
        "shots_per_class": list(shots),
        "seeds": [0, 1, 2],
        "steps_per_cell": steps,
        "planned_cells": expected_cells,
        "planned_optimizer_updates": expected_updates,
        "gradient_checkpointing": {"enabled": True, "use_reentrant": False},
        "reload_scope": "independently deserialized adapter on the same frozen base",
        "base_weights_merged": False,
        "base_retention": {"status": "not_run"},
        "runs": [],
        "limitations": [
            f"{steps} updates per cell on the supplied public support data; no broad quality claim.",
            "Held-out retention results are descriptive screening on the supplied public cases.",
            "The immutable frozen base is shared; each LoRA is freshly seeded and reloaded.",
            "No full-model merge, full-model clone, or provider inference is performed here.",
        ],
    }
    _write_json(report_path, report)
    holder = {}
    original_train_model = training_module.train_model

    def persist():
        report["wall_seconds"] = time.perf_counter() - started
        _write_json(report_path, report)

    def progress(stage, evidence=None, **details):
        if evidence is not None:
            evidence["stage"] = stage
        print(
            json.dumps(
                {
                    "check": "live_training",
                    "stage": stage,
                    "cell": evidence["id"] if evidence is not None else None,
                    "wall_seconds": round(time.perf_counter() - started, 3),
                    **details,
                }
            ),
            flush=True,
        )
        persist()

    def gpu_usage():
        return {
            "allocated_peak_bytes": torch.cuda.max_memory_allocated(),
            "reserved_peak_bytes": torch.cuda.max_memory_reserved(),
        }

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("live training screening requires a CUDA GPU")
        report["gpu"] = torch.cuda.get_device_name()
        progress("dataset_preflight")
        audit = audit_splits(list(paths.values()))
        if not audit["valid"]:
            raise ValueError("live training datasets overlap across fitting/test splits")
        cases = {key: load_cases(path) for key, path in paths.items()}
        if any(case.split != "test" for key in ("text-test", "media-test") for case in cases[key]):
            raise ValueError("retention data must be held-out test cases")
        report["split_audit"] = audit
        report["datasets"] = {
            key: {"cases": len(value), "sha256": fingerprint(value)} for key, value in cases.items()
        }
        support = prepare_support(
            paths["text-train"],
            [paths["text-calibration"], paths["text-test"], paths["media-test"]],
            output / "support",
            shots=shots,
            seeds=[0, 1, 2],
        )
        for cell in support["cells"]:
            for method in ("ce", "ce_brier"):
                report["runs"].append(
                    {
                        "id": f"{cell['id']}-{method}",
                        "cell_id": cell["id"],
                        "seed": cell["seed"],
                        "shots_per_class": cell["shots_per_class"],
                        "method": method,
                        "status": "not_run",
                        "optimizer_updates": 0,
                        "step_losses": [],
                        "heldout_text": {"status": "not_run"},
                        "heldout_media": {"status": "not_run"},
                        "adapter_reload": {"status": "not_run"},
                    }
                )
        persist()
        planned = iter(report["runs"])
        progress("processor_loading")
        processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        base_temperature = load_temperature(model_id, revision=revision)

        class CheckedModel(UnifiedDecisionModel):
            def logits(self, request):
                if (
                    hasattr(self, "train_gold")
                    and not self.lm.training
                    and self.evidence.get("stage") == "training"
                ):
                    progress("calibration", self.evidence, cases=len(cases["text-calibration"]))
                logits = super().logits(request)
                if self.lm.training and torch.is_grad_enabled() and hasattr(self, "train_gold"):
                    from torch.nn import functional as functional

                    gold = self.train_gold[request_fingerprint(request)]
                    targets = torch.tensor(
                        [q.labels().index(gold[q.id]) for q in request.questions],
                        device=logits.device,
                    )
                    detached = logits.detach()
                    probability = detached.softmax(-1)
                    onehot = functional.one_hot(targets, detached.shape[1]).to(probability.dtype)
                    loss = (
                        functional.cross_entropy(detached, targets)
                        + self.brier_weight * (probability - onehot).square().sum(-1).mean()
                    )
                    value = float(loss)
                    if not math.isfinite(value):
                        raise AssertionError("nonfinite observed training loss")
                    self.evidence["step_losses"].append(value)
                return logits

            def save_adapter(self, path):
                evidence = self.evidence
                evidence["status"] = "validating"
                progress("adapter_save", evidence)
                after = _weight_sample(self)
                if after["sha256"] == evidence["lora_before"]["sha256"]:
                    raise AssertionError("optimizer did not change sampled LoRA parameters")
                evidence.update(lora_after=after, trainable_parameters_changed=True)
                super().save_adapter(path)
                progress("heldout_text", evidence, cases=len(cases["text-test"]))
                text, text_records, text_predictions = _predict_records(
                    self, cases["text-test"], evidence["id"]
                )
                evidence["heldout_text"] = text
                _write_records(
                    output / "predictions" / f"{evidence['id']}-text.jsonl", text_records
                )
                probe_cases = [cases["text-test"][0]]
                expected = {probe_cases[0].id: text_predictions[probe_cases[0].id]}
                # Both objectives at every support size retain native media on
                # seed zero; the other seeds have explicit not_run media states.
                selected_media = evidence["seed"] == 0
                if selected_media:
                    progress("heldout_media", evidence, cases=len(cases["media-test"]))
                    media, media_records, media_predictions = _predict_records(
                        self, cases["media-test"], evidence["id"]
                    )
                    evidence["heldout_media"] = media
                    _write_records(
                        output / "predictions" / f"{evidence['id']}-media.jsonl", media_records
                    )
                    probe_cases.append(cases["media-test"][0])
                    expected[probe_cases[-1].id] = media_predictions[probe_cases[-1].id]
                else:
                    evidence["heldout_media"]["reason"] = (
                        "media retention covered by seed0 at every support size"
                    )
                expected_temperature = self.temperature
                progress("adapter_reload", evidence, probes=len(probe_cases))
                base = self.lm.unload()  # PEFT unload does not merge adapter weights.
                del self.lm
                gc.collect()
                torch.cuda.empty_cache()
                fresh = PeftModel.from_pretrained(base, str(path), is_trainable=False)
                reloaded_temperature = load_temperature(path)
                self._initialize(fresh, processor, "cuda", reloaded_temperature, self.max_context)
                if reloaded_temperature != expected_temperature:
                    raise AssertionError("adapter reload changed fitted temperature")
                if _weight_sample(self)["sha256"] != after["sha256"]:
                    raise AssertionError("adapter deserialization changed sampled LoRA weights")
                maximum = 0.0
                for case in probe_cases:
                    actual = self.predict(case.request)["answers"]
                    maximum = max(maximum, _compare_answers(expected[case.id], actual))
                evidence["adapter_reload"] = {
                    "status": "passed",
                    "probes": len(probe_cases),
                    "probability_max_abs_difference": maximum,
                    "probability_abs_tolerance": 2e-6,
                    "temperature_exact_match": True,
                    "sampled_weights_exact_match": True,
                    "frozen_base_shared": True,
                    "merged": False,
                }
                base = self.lm.unload()
                holder["base"] = base
                self._initialize(base, processor, "cuda", base_temperature, self.max_context)
                evidence.update(status="passed", gpu=gpu_usage())
                evidence["cell_wall_seconds"] = time.perf_counter() - self.cell_started
                progress("cell_completed", evidence)

        def model_factory(name, *, revision, device, lora):
            from peft import LoraConfig

            evidence = next(planned)
            evidence["status"] = "loading"
            progress("cell_loading", evidence)
            cell_started = time.perf_counter()
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            if "base" not in holder:
                base = AutoModelForMultimodalLM.from_pretrained(
                    name, revision=revision, dtype=torch.bfloat16
                ).to(device)
                if base.config.model_type != "gemma4_unified":
                    raise ValueError("live training requires the full Gemma 4 Unified checkpoint")
                if getattr(base.config, "_commit_hash", None) != revision:
                    raise AssertionError(
                        "loaded checkpoint revision differs from the pinned revision"
                    )
                holder["base"] = base
                reference = UnifiedDecisionModel.from_components(
                    base, processor, device=device, temperature=base_temperature
                )
                retention = {}
                for key, values in (("text", cases["text-test"]), ("media", cases["media-test"])):
                    metrics, rows, _ = _predict_records(reference, values, "base")
                    retention[key] = metrics
                    _write_records(output / "predictions" / f"base-{key}.jsonl", rows)
                report["base_retention"] = {"status": "completed", **retention}
                del reference
            base = holder["base"]
            for parameter in base.parameters():
                parameter.requires_grad_(False)
            base.config.use_cache = False
            if hasattr(base.config, "text_config"):
                base.config.text_config.use_cache = False
            lm = get_peft_model(base, LoraConfig(**lora))
            lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model = CheckedModel.from_components(
                lm, processor, device=device, temperature=base_temperature
            )
            model.name, model.revision = name, revision
            model.evidence, model.cell_started = evidence, cell_started
            trainable = [(n, p) for n, p in model.lm.named_parameters() if p.requires_grad]
            if not trainable or any("lora_" not in name for name, _ in trainable):
                raise AssertionError("only LoRA parameters may be trainable")
            evidence.update(
                status="ready",
                trainable_parameters=sum(p.numel() for _, p in trainable),
                lora_before=_weight_sample(model),
                resolved_revision=revision,
                load_wall_seconds=time.perf_counter() - cell_started,
            )
            persist()
            return model

        def monitored_train(model, train_cases, calibration_cases, **kwargs):
            evidence = model.evidence
            model.train_gold = {request_fingerprint(c.request): c.gold for c in train_cases}
            model.brier_weight = kwargs["brier_weight"]
            evidence["status"] = "training"
            progress("training", evidence, optimizer_steps=kwargs["steps"])
            before = time.perf_counter()
            original_optimizer = torch.optim.AdamW

            class ObservedAdamW(original_optimizer):
                def step(self, *args, **options):
                    result = super().step(*args, **options)
                    evidence["optimizer_updates"] += 1
                    return result

            try:
                torch.optim.AdamW = ObservedAdamW
                result = original_train_model(model, train_cases, calibration_cases, **kwargs)
            finally:
                torch.optim.AdamW = original_optimizer
                evidence["training_wall_seconds"] = time.perf_counter() - before
            if evidence["optimizer_updates"] != kwargs["steps"]:
                raise AssertionError("observed optimizer update count differs from requested steps")
            if len(evidence["step_losses"]) != kwargs["steps"]:
                raise AssertionError("observed finite loss count differs from requested steps")
            if result["calibration_ids"] != [c.id for c in cases["text-calibration"]]:
                raise AssertionError("calibration used cases outside the locked calibration split")
            calibration = result["calibration"]
            if not all(
                math.isfinite(calibration[k]) for k in ("temperature", "nll_before", "nll_after")
            ):
                raise AssertionError("nonfinite temperature fit")
            evidence.update(status="trained", calibration=calibration, finite_losses=True)
            persist()
            return result

        training_module.train_model = monitored_train
        curve = train_curve(
            output / "support" / "support.json",
            paths["text-calibration"],
            output / "training-curve",
            model_id=model_id,
            revision=revision,
            device="cuda",
            steps=steps,
            lr=2e-5,
            brier_weight=0.1,
            max_updates=expected_updates,
            lockfile=str(root / "uv.lock"),
            model_factory=model_factory,
        )
        by_id = {run["id"]: run for run in report["runs"]}
        for result in curve["runs"]:
            evidence = by_id[f"{result['cell_id']}-{result['method']}"]
            adapter_path = Path(result["adapter_path"])
            provenance_path = adapter_path / "training-provenance.json"
            provenance = json.loads(provenance_path.read_text())
            if set(provenance["train_ids"]) & set(provenance["calibration_ids"]):
                raise AssertionError("adapter fitting provenance contains overlapping case IDs")
            evidence.update(
                adapter_path=str(adapter_path.relative_to(output)),
                adapter_sha256=directory_digest(adapter_path),
                provenance_sha256=hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
                support_sha256=result["support_sha256"],
                fitting_provenance={
                    "train_cases": len(provenance["train_ids"]),
                    "calibration_cases": len(provenance["calibration_ids"]),
                    "test_cases_fitted": 0,
                },
            )
            for key in ("text", "media"):
                metrics = evidence[f"heldout_{key}"]
                if metrics["status"] == "completed":
                    baseline = report["base_retention"][key]
                    metrics["accuracy_delta_from_base"] = metrics["accuracy"] - baseline["accuracy"]
                    metrics["nll_delta_from_base"] = metrics["nll"] - baseline["nll"]
                    metrics["brier_delta_from_base"] = metrics["brier_sum"] - baseline["brier_sum"]
        if len(curve["runs"]) != expected_cells or any(
            r["status"] != "passed" for r in report["runs"]
        ):
            raise AssertionError("not all planned matched real-checkpoint training cells passed")
        report["optimizer_updates"] = sum(r["optimizer_updates"] for r in report["runs"])
        report["status"] = "passed"
        progress("completed", optimizer_updates=report["optimizer_updates"])
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
        (output / "training-error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        for evidence in report["runs"]:
            if evidence["status"] not in ("not_run", "passed"):
                evidence["status"] = "failed"
                evidence["error"] = report["error"]
        if report["base_retention"]["status"] == "not_run":
            report["base_retention"]["reason"] = "training preflight or checkpoint loading failed"
        report["optimizer_updates"] = sum(r["optimizer_updates"] for r in report["runs"])
        progress("failed", error_type=type(exc).__name__, error_message=str(exc)[:300])
    finally:
        training_module.train_model = original_train_model
        holder.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            report["cuda_allocated_bytes_on_return"] = torch.cuda.memory_allocated()
        persist()
    return report
