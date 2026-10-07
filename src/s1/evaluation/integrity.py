"""Shared fitting-data and resolved-runtime checks for every evaluation path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from s1.contracts import StrictModel

from .datasets import request_fingerprint


def _provenance_sets(artifact, keys):
    values = [artifact[key] for key in keys]
    if any(
        not isinstance(items, list) or not all(isinstance(value, str) for value in items)
        for items in values
    ):
        raise ValueError("invalid fitting provenance")
    return [set(items) for items in values]


def fitting_provenance_blockers(model, cases):
    """Reject any ID, source group, or request used to fit the selected model."""
    cases = tuple(cases)
    identities = [
        (case.id, getattr(case, "group_id", None) or case.id, request_fingerprint(case.request))
        for case in cases
    ]
    reasons = []

    def overlaps(artifact, keys):
        fitted = _provenance_sets(artifact, keys)
        return any(
            any(value in known for value, known in zip(identity, fitted)) for identity in identities
        )

    if model.decision_adapter_path:
        try:
            provenance = json.loads(
                (Path(model.decision_adapter_path) / "training-provenance.json").read_text()
            )
            for split in ("train", "calibration"):
                if overlaps(
                    provenance, [f"{split}_{key}" for key in ("ids", "groups", "requests")]
                ):
                    reasons.append("evaluation_overlaps_adapter_training")
        except (OSError, ValueError, KeyError, TypeError):
            reasons.append("missing_adapter_training_provenance")
    if model.adapter in ("tfidf", "prior", "setfit") and model.artifact_file:
        try:
            artifact = json.loads(Path(model.artifact_file).read_text())
            if overlaps(artifact, ["train_ids", "train_groups", "train_requests"]):
                reasons.append("evaluation_overlaps_training")
        except (OSError, ValueError, KeyError, TypeError):
            reasons.append("invalid_training_provenance")
    if model.calibration == "domain":
        from .calibration import validate_calibration

        try:
            validate_calibration(model, cases)
        except (ValueError, OSError):
            reasons.append("invalid_or_overlapping_calibration")
    return list(dict.fromkeys(reasons))


class RuntimeTelemetry(StrictModel):
    revision: str | None = None
    device: str | None = None
    precision: Literal["float64", "float32", "float16", "bfloat16", "quantized", "provider"]
    temperature: float | None = Field(default=None, gt=0)
    calibration_temperature: float | None = Field(default=None, gt=0)
    context_limit: int | None = Field(default=None, ge=1)
    context_policy: Literal["reject", "truncate", "provider", "not_applicable"]
    quantization: Literal["none", "int8", "nf4", "nvfp4"] = "none"
    quantization_details: dict | None = None
    runtime_package: str | None = None
    source_revision: str | None = None
    base_model_id: str | None = None
    base_revision: str | None = None
    checkpoint_task: str | None = None
    compute_precision: str | None = None
    compute_backend: str | None = None
    runtime_options: dict | None = None
    temperatures_by_type: dict[str, float] | None = None


def validate_runtime_telemetry(adapter, spec):
    """Validate resolved identity before inference and return persistable telemetry."""
    result = RuntimeTelemetry.model_validate(adapter.telemetry())
    if spec.adapter in ("gemma", "laya", "decider", "kev", "agentjev", "clef", "jev_omni") and (
        result.revision != spec.revision
    ):
        raise ValueError("resolved model revision differs from the plan")
    if result.precision != spec.precision:
        raise ValueError("resolved precision differs from the plan")
    if result.quantization != spec.quantization:
        raise ValueError("resolved quantization differs from the plan")
    return result.model_dump(mode="json")
