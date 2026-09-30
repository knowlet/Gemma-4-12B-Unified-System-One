"""Matched CE versus CE+Brier LoRA training with test data locked out."""

from __future__ import annotations

import gc
import json
import math
import re
from pathlib import Path

from .artifacts import write_json
from .checkpoints import directory_digest
from .datasets import audit_splits, fingerprint, load_cases
from .planning import runtime_identity


def train_curve(
    support_manifest,
    calibration_path,
    output,
    *,
    model_id,
    revision,
    device=None,
    steps=100,
    lr=2e-5,
    brier_weight=0.1,
    max_updates=10000,
    lockfile="uv.lock",
    model_factory=None,
):
    from s1.training import train_model

    manifest_path = Path(support_manifest)
    support = json.loads(manifest_path.read_text())
    cells = support["cells"]
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(brier_weight) or brier_weight <= 0:
        raise ValueError("positive finite learning rate and Brier weight required")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", c["id"]) for c in cells):
        raise ValueError("invalid support cell ID")
    if len({c["seed"] for c in cells}) < 3 or steps < 1 or steps * len(cells) * 2 > max_updates:
        raise ValueError("three seeds and an adequate explicit update budget are required")
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        raise ValueError("pin the base model to an immutable revision")
    calibration = load_cases(calibration_path)
    if (
        any(c.split != "calibration" for c in calibration)
        or fingerprint(calibration) not in support["heldout_sha256"]
    ):
        raise ValueError("calibration must match the locked support manifest")
    prepared = []
    for cell in cells:
        path = manifest_path.parent / cell["dataset"]
        cases = load_cases(path)
        if (
            fingerprint(cases) != cell["dataset_sha256"]
            or [c.id for c in cases] != cell["support_ids"]
        ):
            raise ValueError("support data changed after selection")
        if (
            any(c.split != "train" for c in cases)
            or not audit_splits([path, calibration_path])["valid"]
        ):
            raise ValueError("training/calibration overlap")
        prepared.append((cell, cases))
    import torch

    from s1.unified import UnifiedDecisionModel

    factory = model_factory or UnifiedDecisionModel
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "source_support": support,
        "runtime": runtime_identity(lockfile),
        "base_model": model_id,
        "base_revision": revision,
        "calibration_sha256": fingerprint(calibration),
        "steps_per_cell": steps,
        "learning_rate": lr,
        "brier_weight": brier_weight,
        "runs": [],
    }
    write_json(output / "training-curve.json", report)
    try:
        for cell, cases in prepared:
            for name, weight in (("ce", 0.0), ("ce_brier", brier_weight)):
                torch.manual_seed(cell["seed"])
                model = factory(
                    model_id,
                    revision=revision,
                    device=device,
                    lora={
                        "r": 8,
                        "lora_alpha": 16,
                        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                        "bias": "none",
                    },
                )
                result = train_model(
                    model,
                    cases,
                    calibration,
                    steps=steps,
                    lr=lr,
                    seed=cell["seed"],
                    brier_weight=weight,
                )
                path = output / f"{cell['id']}-{name}"
                model.save_adapter(path)
                # Provenance travels with the adapter, so test overlap can be rejected
                # even when it is moved independently of this training report.
                from .datasets import request_fingerprint

                provenance = {
                    "train_ids": [c.id for c in cases],
                    "train_groups": [c.group_id or c.id for c in cases],
                    "train_requests": [request_fingerprint(c.request) for c in cases],
                    "calibration_ids": [c.id for c in calibration],
                    "calibration_groups": [c.group_id or c.id for c in calibration],
                    "calibration_requests": [request_fingerprint(c.request) for c in calibration],
                }
                write_json(path / "training-provenance.json", provenance)
                result.update(
                    cell_id=cell["id"],
                    method=name,
                    adapter_path=str(path),
                    adapter_sha256=directory_digest(path),
                    support_sha256=cell["dataset_sha256"],
                )
                report["runs"].append(result)
                write_json(output / "training-curve.json", report)
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        report["status"] = "completed"
    except BaseException:
        report["status"] = "interrupted"
        raise
    finally:
        write_json(output / "training-curve.json", report)
    return report
