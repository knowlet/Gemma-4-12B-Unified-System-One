#!/usr/bin/env python3
"""Bounded one-pass LoRA release with durable resume and post-merge calibration.

The pilot is discarded. Training restarts from the pinned base, then traverses
one seeded permutation exactly once. Only calibration labels fit temperature;
test and regression labels are consumed by the separate evaluate phase.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

BASE = "google/gemma-4-12B-it"
REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"


@dataclass(frozen=True)
class Recipe:
    model: str = BASE
    revision: str = REVISION
    seed: int = 42
    learning_rate: float = 2e-5
    brier_weight: float = 0.0
    rank: int = 8
    alpha: int = 16
    max_context: int = 4096
    checkpoint_every: int = 128
    progress_every: int = 32
    pilot_steps: int = 8
    max_training_seconds: int = 18000

    def validate(self):
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", self.revision):
            raise ValueError("base revision must be an immutable commit")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning rate must be positive and finite")
        if not math.isfinite(self.brier_weight) or self.brier_weight < 0:
            raise ValueError("Brier weight must be nonnegative and finite")
        if (
            min(
                self.rank,
                self.alpha,
                self.max_context,
                self.checkpoint_every,
                self.progress_every,
                self.pilot_steps,
                self.max_training_seconds,
            )
            < 1
        ):
            raise ValueError("recipe counts and durations must be positive")
        if self.max_training_seconds > 18000:
            raise ValueError("training budget cannot exceed five hours; Modal hard cap is six")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def event(output, kind, **fields):
    value = {"event": kind, "time_unix": time.time(), **fields}
    with (Path(output) / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(value, allow_nan=False), flush=True)


def load_release_data(directory, *, expected_counts=(2048, 256, 256)):
    from s1.evaluation.datasets import audit_splits, fingerprint, load_cases, request_fingerprint

    root = Path(directory)
    paths = {
        "train": root / "train.jsonl",
        "calibration": root / "calibration.jsonl",
        "test": root / "test.jsonl",
        "media": root / "media-test.jsonl",
    }
    if (root / "regression-text.jsonl").exists():
        paths["regression_text"] = root / "regression-text.jsonl"
    data = {name: load_cases(path) for name, path in paths.items()}
    for name, cases in data.items():
        expected_split = name if name in ("train", "calibration") else "test"
        if any(case.split != expected_split for case in cases):
            raise ValueError(f"incorrect split in {name}")
    if tuple(len(data[name]) for name in ("train", "calibration", "test")) != expected_counts:
        raise ValueError(f"release requires train/calibration/test counts {expected_counts}")
    audit = audit_splits(paths.values())
    if not audit["valid"]:
        raise ValueError("release source groups, ids or requests overlap across splits")
    # Also reject overlap between different test corpora: they are separate
    # populations and must not silently count the same case twice.
    seen = set()
    for name, cases in data.items():
        ids = {case.id for case in cases}
        if seen & ids:
            raise ValueError("case ids overlap between release corpora")
        seen |= ids
    identity = {
        name: {
            "cases": len(cases),
            "dataset_sha256": fingerprint(cases),
            "file_sha256": file_sha256(paths[name]),
        }
        for name, cases in data.items()
    }
    manifest = root / "manifest.json"
    prepared = json.loads(manifest.read_text())
    for name, expected in prepared["files"].items():
        if Path(name).name != name or file_sha256(root / name) != expected:
            raise ValueError("prepared release file checksum mismatch")
    for name, cases in data.items():
        recorded = prepared["datasets"][paths[name].name]
        expected = {
            **identity[name],
            "case_ids": [case.id for case in cases],
            "group_ids": [case.group_id or case.id for case in cases],
            "request_sha256s": [request_fingerprint(case.request) for case in cases],
        }
        if any(recorded.get(key) != value for key, value in expected.items()):
            raise ValueError("prepared release dataset receipt mismatch")
    identity["preparation_manifest_sha256"] = file_sha256(manifest)
    return data, identity


def implementation_identity():
    import s1

    package = Path(s1.__file__).parent
    root = Path(__file__).resolve().parents[1]
    return {
        "trainer_sha256": file_sha256(__file__),
        "uv_lock_sha256": file_sha256(root / "uv.lock"),
        "s1_sources": {
            str(path.relative_to(package)): file_sha256(path)
            for path in sorted(package.rglob("*.py"))
        },
        "packages": _runtime()["packages"],
    }


def _runtime():
    import torch

    return {
        "packages": {
            name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")
        },
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory
        if torch.cuda.is_available()
        else None,
    }


def _memory():
    import torch

    if not torch.cuda.is_available():
        return {}
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def _checkpoint(model, optimizer, rng, output, state, commit):
    import torch

    root = Path(output) / "checkpoints"
    root.mkdir(exist_ok=True)
    name = f"step-{state['step']:06d}-{uuid.uuid4().hex[:8]}"
    temporary = root / ("." + name)
    temporary.mkdir()
    model.save_adapter(temporary / "adapter")
    payload = {
        **state,
        "optimizer": optimizer.state_dict(),
        "python_rng": rng.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "trainable": {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.lm.named_parameters()
            if parameter.requires_grad
        },
    }
    torch.save(payload, temporary / "state.pt")
    meta = {
        "step": state["step"],
        "identity": state["identity"],
        "state_sha256": file_sha256(temporary / "state.pt"),
    }
    write_json(temporary / "checkpoint.json", meta)
    temporary.rename(root / name)
    write_json(Path(output) / "latest.json", {"directory": name, **meta})
    commit()


def _restore(model, optimizer, rng, output, identity):
    import torch

    pointer = json.loads((Path(output) / "latest.json").read_text())
    name = pointer["directory"]
    if not re.fullmatch(r"step-[0-9]{6}-[0-9a-f]{8}", name):
        raise ValueError("invalid checkpoint pointer")
    path = Path(output) / "checkpoints" / name / "state.pt"
    if pointer["identity"] != identity or file_sha256(path) != pointer["state_sha256"]:
        raise ValueError("checkpoint identity or state checksum mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["identity"] != identity or state["step"] != pointer["step"]:
        raise ValueError("checkpoint payload identity mismatch")
    parameters = {
        name: parameter
        for name, parameter in model.lm.named_parameters()
        if parameter.requires_grad
    }
    if parameters.keys() != state["trainable"].keys():
        raise ValueError("checkpoint trainable parameter names differ")
    with torch.no_grad():
        for name, parameter in parameters.items():
            saved = state["trainable"][name]
            if saved.shape != parameter.shape or not torch.isfinite(saved).all():
                raise ValueError("invalid checkpoint trainable tensor")
            parameter.copy_(saved)
    optimizer.load_state_dict(state.pop("optimizer"))
    rng.setstate(state.pop("python_rng"))
    torch.set_rng_state(state.pop("torch_rng"))
    cuda_rng = state.pop("cuda_rng")
    if cuda_rng and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_rng)
    state.pop("trainable")
    return state


def train_epoch(
    model,
    cases,
    output,
    recipe,
    *,
    identity,
    commit=lambda: None,
    max_steps=None,
    resume=False,
    first_case_id=None,
):
    """One example per optimizer update; checkpoint only complete updates."""
    import torch
    from torch.nn import functional as F

    from s1.training import permute_request

    recipe.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = len(cases) if max_steps is None else min(max_steps, len(cases))
    if not cases or target < 1 or any(case.split != "train" for case in cases):
        raise ValueError("a nonempty training-only population is required")
    rng = random.Random(recipe.seed)
    order = list(range(len(cases)))
    rng.shuffle(order)
    if first_case_id is not None:
        first = next(i for i, case in enumerate(cases) if case.id == first_case_id)
        order.remove(first)
        order.insert(0, first)
    parameters = [p for p in model.lm.parameters() if p.requires_grad]
    if not parameters or any(p.dtype != torch.float32 for p in parameters):
        raise ValueError("release LoRA trainable parameters must be nonempty and FP32")
    optimizer = torch.optim.AdamW(parameters, lr=recipe.learning_rate)
    state = {
        "identity": identity,
        "step": 0,
        "order": order,
        "target_steps": target,
        "training_seconds": 0.0,
        "loss_initial": None,
        "loss_final": None,
    }
    latest = output / "latest.json"
    if resume and latest.exists():
        state = _restore(model, optimizer, rng, output, identity)
        if state["order"] != order or state["target_steps"] != target:
            raise ValueError("resume sample order or target steps changed")
        if not 0 <= state["step"] <= target:
            raise ValueError("invalid checkpoint step")
        updates = output / "updates.jsonl"
        if updates.exists():
            # A crash may leave updates newer than the last durable checkpoint.
            # Retain the abandoned attempt for diagnosis, but keep the canonical
            # epoch ledger at exactly the restored update boundary.
            previous = updates.read_text()
            rows = []
            for line in previous.splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    break
                if row["step"] <= state["step"]:
                    rows.append(row)
            if [row["step"] for row in rows] != list(range(1, state["step"] + 1)):
                raise ValueError("checkpoint update ledger is incomplete or duplicated")
            (output / f"attempt-{uuid.uuid4().hex}.jsonl").write_text(previous)
            updates.write_text("".join(json.dumps(row) + "\n" for row in rows))
    elif latest.exists():
        raise ValueError("checkpoint exists; explicitly enable resume")
    model.lm.train()
    start = time.monotonic()
    previous_seconds = state["training_seconds"]
    status = "completed"
    for step in range(state["step"], target):
        if previous_seconds + time.monotonic() - start >= recipe.max_training_seconds:
            status = "paused_budget"
            break
        case = cases[order[step]]
        request = permute_request(case.request, rng)
        logits = model.logits(request)
        golds = torch.tensor(
            [q.labels().index(case.gold[q.id]) for q in request.questions], device=logits.device
        )
        probabilities = logits.softmax(-1)
        onehot = F.one_hot(golds, logits.shape[1]).to(probabilities.dtype)
        loss = F.cross_entropy(logits, golds)
        if recipe.brier_weight:
            loss = loss + recipe.brier_weight * (probabilities - onehot).square().sum(-1).mean()
        if not torch.isfinite(loss):
            raise ValueError(f"nonfinite loss at update {step + 1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        if any(not torch.isfinite(parameter).all() for parameter in parameters):
            raise ValueError(f"nonfinite parameter at update {step + 1}")
        value = float(loss.detach())
        state.update(
            step=step + 1,
            loss_final=value,
            training_seconds=previous_seconds + time.monotonic() - start,
        )
        if state["loss_initial"] is None:
            state["loss_initial"] = value
        # All updates persist locally; concise stdout at requested intervals.
        row = {
            "step": step + 1,
            "case_id": case.id,
            "loss": value,
            "gradient_norm": float(grad_norm),
            "elapsed_seconds": state["training_seconds"],
        }
        with (output / "updates.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if step == 0 or (step + 1) % recipe.progress_every == 0 or step + 1 == target:
            event(output, "update", **row, **_memory())
        if (step + 1) % recipe.checkpoint_every == 0:
            _checkpoint(model, optimizer, rng, output, state, commit)
    state["training_seconds"] = previous_seconds + time.monotonic() - start
    _checkpoint(model, optimizer, rng, output, state, commit)
    result = {
        **state,
        "status": status,
        "memory": _memory(),
        "seconds_per_update": state["training_seconds"] / max(state["step"], 1),
    }
    write_json(output / "training.json", result)
    commit()
    return result


def _new_model(recipe, *, adapter=False, path=None):
    import torch

    from s1.unified import UnifiedDecisionModel

    torch.manual_seed(recipe.seed)
    kwargs = {
        "device": "cuda",
        "precision": "bfloat16",
        "max_context": recipe.max_context,
        "temperature": 1.0,
    }
    if adapter:
        kwargs["lora"] = {
            "r": recipe.rank,
            "lora_alpha": recipe.alpha,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            "bias": "none",
        }
    model = UnifiedDecisionModel(
        path or recipe.model, revision=None if path else recipe.revision, **kwargs
    )
    if adapter:
        # PEFT normally upcasts adapters. Make the actual training contract explicit.
        for parameter in model.lm.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        model.lm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model


def _free_cuda():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def capture_calibration_logits(model, cases):
    import torch

    if not cases or any(case.split != "calibration" for case in cases):
        raise ValueError("temperature fitting requires calibration-only cases")
    model.lm.eval()
    records = []
    with torch.inference_mode():
        for case in cases:
            logits = model.logits(case.request).cpu().tolist()
            for question, row in zip(case.request.questions, logits, strict=True):
                values = row[: len(question.labels())]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("nonfinite calibration candidate logits")
                gold = question.labels().index(case.gold[question.id])
                records.append(
                    {
                        "case_id": case.id,
                        "question_id": question.id,
                        "labels": question.labels(),
                        "logits": values,
                        "gold_index": gold,
                    }
                )
    return records


def calibrate(model, cases):
    from s1.training import fit_temperature

    records = capture_calibration_logits(model, cases)
    rows = [record["logits"] for record in records]
    golds = [record["gold_index"] for record in records]
    report = fit_temperature(rows, golds)
    model.set_temperature(report["temperature"])
    return {**report, "records": records}


def merge_drift(before, after):
    import numpy as np

    lookup = {(row["case_id"], row["question_id"]): row for row in after}
    rows = []
    for original in before:
        observed = lookup[(original["case_id"], original["question_id"])]
        if original["labels"] != observed["labels"]:
            raise ValueError("merge calibration labels changed")
        a, b = (np.asarray(row["logits"], dtype=float) for row in (original, observed))
        pa, pb = (np.exp(x - x.max()) / np.exp(x - x.max()).sum() for x in (a, b))
        rows.append(
            {
                "case_id": original["case_id"],
                "question_id": original["question_id"],
                "logit_max_abs_delta": float(np.abs(a - b).max()),
                "probability_max_abs_delta_at_t1": float(np.abs(pa - pb).max()),
                "argmax_match": bool(a.argmax() == b.argmax()),
            }
        )
    return {
        "decisions": len(rows),
        "coverage": 1.0,
        "max_logit_abs_delta": max(row["logit_max_abs_delta"] for row in rows),
        "max_probability_abs_delta_at_t1": max(
            row["probability_max_abs_delta_at_t1"] for row in rows
        ),
        "argmax_matches": sum(row["argmax_match"] for row in rows),
        "records": rows,
        "scope": "measurement on first eight calibration cases; no bitwise parity claim",
    }


def artifact_hashes(directory):
    root = Path(directory)
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def run_phase(stage, data_dir, output, recipe=None, *, commit=lambda: None, resume=False):
    """Run one phase. Non-complete states never authorize the following phase."""
    recipe = recipe or Recipe()
    recipe.validate()
    if stage not in ("pilot", "train", "evaluate"):
        raise ValueError("stage must be pilot, train or evaluate")
    data, data_identity = load_release_data(data_dir)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    implementation = implementation_identity()
    identity = digest(
        {"recipe": asdict(recipe), "datasets": data_identity, "implementation": implementation}
    )
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())["identity"] != identity:
            raise ValueError("release recipe or datasets differ from the existing run")
    else:
        write_json(
            manifest_path,
            {
                "identity": identity,
                "recipe": asdict(recipe),
                "datasets": data_identity,
                "epochs": 1,
                "pilot_policy": "discard; full pass restarts from pinned base",
                "runtime": _runtime(),
                "implementation": implementation,
            },
        )
        commit()
    phase = output / stage
    phase.mkdir(exist_ok=True)
    event(phase, "phase_started", stage=stage, identity=identity)
    receipt = {"stage": stage, "identity": identity, "status": "running", "runtime": _runtime()}
    write_json(phase / "receipt.json", receipt)
    commit()
    started = time.monotonic()
    try:
        if stage == "pilot":
            model = _new_model(recipe, adapter=True)
            # Exercise the longest training example first. Preflight every train
            # prompt to reject context truncation before paying for a full pass.
            lengths = []
            for case in data["train"]:
                prepared, _, _ = model.prepare(case.request)
                lengths.append((int(prepared["input_ids"].shape[-1]), case.id))
            longest = max(lengths)
            event(
                phase,
                "context_preflight",
                cases=len(lengths),
                longest_tokens=longest[0],
                longest_case_id=longest[1],
            )
            result = train_epoch(
                model,
                data["train"],
                phase,
                recipe,
                identity=identity,
                commit=commit,
                max_steps=recipe.pilot_steps,
                resume=resume,
                first_case_id=longest[1],
            )
            result["longest_training_tokens"] = longest[0]
            del model
            _free_cuda()
        elif stage == "train":
            pilot = json.loads((output / "pilot/receipt.json").read_text())
            if pilot["identity"] != identity or pilot["status"] != "completed":
                raise ValueError("matching successful pilot required before full training")
            if pilot["result"]["step"] != recipe.pilot_steps:
                raise ValueError("pilot did not complete its fixed update budget")
            model = _new_model(recipe, adapter=True)
            result = train_epoch(
                model, data["train"], phase, recipe, identity=identity, commit=commit, resume=resume
            )
            if result["status"] == "completed":
                adapter = output / "adapter"
                if not adapter.exists():
                    temporary_adapter = output / (".adapter-" + uuid.uuid4().hex)
                    model.save_adapter(temporary_adapter)
                    write_json(
                        temporary_adapter / "release-identity.json",
                        {
                            "identity": identity,
                            "step": result["step"],
                            "files": artifact_hashes(temporary_adapter),
                        },
                    )
                    temporary_adapter.rename(adapter)
                    commit()
                adapter_identity = json.loads((adapter / "release-identity.json").read_text())
                actual = artifact_hashes(adapter)
                actual.pop("release-identity.json")
                if (
                    adapter_identity["identity"] != identity
                    or adapter_identity["step"] != result["step"]
                    or adapter_identity["files"] != actual
                ):
                    raise ValueError("saved final adapter is incomplete or changed")
                before_merge = capture_calibration_logits(model, data["calibration"][:8])
                write_json(phase / "premerge-calibration-logits.json", before_merge)
                commit()
                merged = output / "merged"
                if not merged.exists():
                    temporary = output / (".merged-" + uuid.uuid4().hex)
                    temporary.mkdir()
                    lm = model.lm.merge_and_unload(safe_merge=True)
                    lm.save_pretrained(temporary, safe_serialization=True)
                    model.processor.save_pretrained(temporary)
                    write_json(
                        temporary / "s1_config.json",
                        {
                            "temperature": 1.0,
                            "source_model": recipe.model,
                            "source_revision": recipe.revision,
                            "prompt_version": 1,
                            "training_recipe_sha256": identity,
                        },
                    )
                    temporary.rename(merged)
                    del lm
                    commit()
                del model
                _free_cuda()
                # This is a new model loaded from the exact final weight files.
                model = _new_model(recipe, path=str(merged))
                calibration = calibrate(model, data["calibration"])
                write_json(phase / "postmerge-calibration.json", calibration)
                drift = merge_drift(before_merge, calibration["records"])
                write_json(phase / "merge-drift.json", drift)
                result["merge_drift"] = {
                    key: value for key, value in drift.items() if key != "records"
                }
                config = json.loads((merged / "s1_config.json").read_text())
                config.update(
                    temperature=calibration["temperature"],
                    calibration_dataset_sha256=data_identity["calibration"]["dataset_sha256"],
                )
                write_json(merged / "s1_config.json", config)
                write_json(
                    output / "artifacts.json",
                    {
                        "identity": identity,
                        "adapter": artifact_hashes(adapter),
                        "merged": artifact_hashes(merged),
                    },
                )
                result["postmerge_calibration"] = {
                    key: value for key, value in calibration.items() if key != "records"
                }
                del model
                _free_cuda()
                commit()
        else:
            result = evaluate_release(data, data_identity, output, recipe, identity, commit)
        receipt.update(status=result["status"], result=result)
    except BaseException as exc:
        receipt.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        (phase / "failure.txt").write_text(traceback.format_exc())
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["memory"] = _memory()
        write_json(phase / "receipt.json", receipt)
        event(
            phase,
            "phase_finished",
            status=receipt["status"],
            elapsed_seconds=receipt["elapsed_seconds"],
        )
        commit()
    return receipt


def evaluate_release(data, data_identity, output, recipe, identity, commit):
    from s1.benchmark import compare_reports, evaluate

    training = json.loads((output / "train/receipt.json").read_text())
    if training["identity"] != identity or training["status"] != "completed":
        raise ValueError("completed matching training and merged calibration required")
    hashes = json.loads((output / "artifacts.json").read_text())
    if hashes["identity"] != identity or hashes["merged"] != artifact_hashes(output / "merged"):
        raise ValueError("merged checkpoint changed after calibration")
    result = {"status": "completed", "reports": {}, "comparisons": {}}
    for kind in ("base", "trained"):
        model = _new_model(recipe, path=str(output / "merged") if kind == "trained" else None)
        if kind == "base":
            calibration = calibrate(model, data["calibration"])
            write_json(output / "evaluate/base-calibration.json", calibration)
        else:
            # _new_model uses temperature=1 explicitly, so restore the audited
            # final checkpoint sidecar rather than refit on any test data.
            sidecar = json.loads((output / "merged/s1_config.json").read_text())
            model.set_temperature(sidecar["temperature"])
        model.name = kind
        for name in ("test", "media", "regression_text"):
            if name not in data:
                continue
            report = evaluate(model, data[name], warmup=1)
            report["metadata"] = {
                "identity": identity,
                "temperature": model.temperature,
                "base_model": recipe.model,
                "base_revision": recipe.revision,
                "release_dataset_sha256": data_identity[name]["dataset_sha256"],
            }
            write_json(output / "evaluate" / f"{kind}-{name}.json", report)
            result["reports"][f"{kind}-{name}"] = {
                key: report[key] for key in ("counts", "coverage", "metrics", "latency_ms")
            }
            if report["coverage"] != 1.0 or report["warmup_errors"]:
                result["status"] = "completed_with_errors"
            commit()
        del model
        _free_cuda()
    for name in ("test", "media", "regression_text"):
        if name in data:
            reports = [
                json.loads((output / "evaluate" / f"{kind}-{name}.json").read_text())
                for kind in ("base", "trained")
            ]
            result["comparisons"][name] = compare_reports(reports)
    result["scope"] = (
        "descriptive one-seed public-data evaluation; no acceptance or production claim"
    )
    write_json(output / "evaluate/summary.json", result)
    return result
