"""Deterministic preflight and budget accounting. This module never runs a model."""

from __future__ import annotations

import hashlib
import json
import platform
import re
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.parse import urlsplit

from .contracts import ModelSpec, ProfileSpec
from .datasets import fingerprint, load_cases, request_fingerprint
from .registry import Registry


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def runtime_identity(lockfile) -> dict:
    """Hash the installed S1 source and the supplied lock; no Git/network dependency."""
    root = Path(__file__).resolve().parents[1]
    sources = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.py"))
    }
    packages = {}
    for name in (
        "pydantic",
        "numpy",
        "httpx",
        "torch",
        "transformers",
        "accelerate",
        "laya",
        "peft",
        "scikit-learn",
        "sentence-transformers",
        "setfit",
        "datasets",
    ):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "source_sha256": _digest(sources),
        "lock_sha256": hashlib.sha256(Path(lockfile).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def case_eligibility(model: ModelSpec, case) -> dict:
    caps = model.capabilities
    unsupported, unknown = [], []
    modalities = {"text", *(media.type for media in case.request.media)}
    primitives = {q.type for q in case.request.questions}
    for name, required, supported in (
        ("modality", modalities, caps.modalities),
        ("primitive", primitives, caps.primitives),
    ):
        if supported is None:
            unknown.append(f"unknown_{name}_support")
        else:
            unsupported.extend(
                f"unsupported_{name}:{item}" for item in sorted(required - set(supported))
            )
    for name, actual, limit in (
        ("options", max(len(q.labels()) for q in case.request.questions), caps.max_options),
        ("questions", len(case.request.questions), caps.max_questions),
    ):
        if limit is None:
            unknown.append(f"unknown_max_{name}")
        elif actual > limit:
            unsupported.append(f"max_{name}_exceeded")
    if model.protocol == "agentjev" and any(
        q.type == "score" and len(q.labels()) > 10 for q in case.request.questions
    ):
        unsupported.append("max_score_levels_exceeded")
    if model.adapter in ("tfidf", "prior", "setfit") and model.artifact_file:
        from .baselines import schema_key

        try:
            artifact = json.loads(Path(model.artifact_file).read_text())
            if any(schema_key(q) not in artifact["schemas"] for q in case.request.questions):
                unsupported.append("unseen_supervised_schema")
        except (OSError, ValueError, KeyError, TypeError):
            unknown.append("invalid_baseline_artifact")
    return {
        "case_id": case.id,
        "group_id": getattr(case, "group_id", None) or case.id,
        "request_sha256": request_fingerprint(case.request),
        "decisions": len(case.request.questions),
        "eligibility": "unsupported" if unsupported else "unknown" if unknown else "eligible",
        "reasons": unsupported + unknown,
    }


def _configuration_blockers(
    model: ModelSpec, profile: ProfileSpec, environment
) -> tuple[list, dict]:
    reasons, service = [], {}
    if model.decision_adapter_path:
        from .checkpoints import directory_digest

        if model.adapter != "gemma" or not model.decision_adapter_sha256:
            reasons.append("invalid_decision_adapter_settings")
        else:
            try:
                if directory_digest(model.decision_adapter_path) != model.decision_adapter_sha256:
                    reasons.append("decision_adapter_hash_mismatch")
            except (OSError, ValueError):
                reasons.append("missing_decision_adapter")
    caps = model.capabilities
    if model.adapter == "unimplemented":
        reasons.append("adapter_not_implemented")
    if model.adapter in ("gemma", "laya", "embedding", "nli", "cross_encoder", "setfit"):
        if not model.model_id:
            reasons.append("missing_model_id")
        if not model.revision or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", model.revision):
            reasons.append("unresolved_model_revision")
        processor = model.processor_revision or model.revision
        if not processor or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", processor):
            reasons.append("unresolved_processor_revision")
    if model.adapter in ("tfidf", "prior", "setfit"):
        if not model.artifact_file or not model.artifact_sha256:
            reasons.append("missing_baseline_artifact")
        else:
            try:
                if (
                    hashlib.sha256(Path(model.artifact_file).read_bytes()).hexdigest()
                    != model.artifact_sha256
                ):
                    reasons.append("baseline_artifact_hash_mismatch")
            except OSError:
                reasons.append("missing_baseline_artifact")
    if model.adapter != "uniform":
        if model.precision == "unknown":
            reasons.append("unknown_precision")
        if model.execution_mode == "unknown":
            reasons.append("unknown_execution_mode")
        if model.calibration == "unknown":
            reasons.append("unknown_calibration")
    if model.adapter == "http":
        # Presence and syntax only: no credential values are copied into artifacts.
        endpoint = environment.get(model.endpoint_env, "") if model.endpoint_env else ""
        if not endpoint:
            reasons.append("missing_endpoint")
        else:
            try:
                parsed = urlsplit(endpoint)
                valid = (
                    parsed.scheme in ("https", "http")
                    and parsed.hostname
                    and not (parsed.username or parsed.password or parsed.query or parsed.fragment)
                )
                _ = parsed.port
            except ValueError:
                valid = False
            if not valid:
                reasons.append("invalid_endpoint")
            else:
                service["endpoint_sha256"] = hashlib.sha256(endpoint.encode()).hexdigest()
        if model.token_env and not environment.get(model.token_env):
            reasons.append("missing_credentials")
        if not model.revision:
            reasons.append("missing_provider_version")
    if profile.require_probabilities and caps.probabilities != "complete":
        reasons.append(f"probabilities_{caps.probabilities}")
    if profile.require_independent_questions and caps.independent_questions is not True:
        reasons.append("independent_questions_unavailable")
    if profile.require_native_batch and caps.batch != "native":
        reasons.append("native_batch_unavailable")
    if profile.batch_size > 1 and caps.batch not in ("native", "loop_emulated"):
        reasons.append("batch_unavailable")
    if profile.concurrency > 1 and caps.concurrent_requests is not True:
        reasons.append("concurrent_requests_unavailable")
    if model.preprocessing != "none":
        if not model.preprocessor_revision:
            reasons.append("unresolved_preprocessor_revision")
        endpoint = (
            environment.get(model.preprocessor_endpoint_env, "")
            if model.preprocessor_endpoint_env
            else ""
        )
        try:
            parsed = urlsplit(endpoint)
            valid = (
                parsed.scheme in ("https", "http")
                and parsed.hostname
                and not (parsed.username or parsed.password or parsed.query or parsed.fragment)
            )
            _ = parsed.port
        except ValueError:
            valid = False
        if not valid:
            reasons.append("missing_or_invalid_preprocessor_endpoint")
        else:
            service["preprocessor_endpoint_sha256"] = hashlib.sha256(endpoint.encode()).hexdigest()
        if model.preprocessor_token_env and not environment.get(model.preprocessor_token_env):
            reasons.append("missing_preprocessor_credentials")
    return reasons, service


def create_plan(
    registry: Registry,
    profile_id: str,
    *,
    lockfile,
    environment=None,
    enable_models=(),
    purpose="plan",
) -> dict:
    """Inspect actual dataset shapes and reserve aggregate cost before execution.

    `ready` means declaration-level checks passed, not runtime/GPU compatibility.
    The sequential executor performs additional adapter-specific checks.
    """
    if purpose not in ("plan", "preflight"):
        raise ValueError("purpose must be plan or preflight")
    if profile_id not in registry.profiles:
        raise ValueError(f"unknown profile {profile_id}")
    profile = registry.profiles[profile_id]
    if set(enable_models) - set(profile.models):
        raise ValueError("--enable-model must name a model selected by the profile")
    suite = registry.suites[profile.suite]
    cases = load_cases(registry.dataset_path(suite))
    if any(case.split != profile.split for case in cases):
        raise ValueError(f"evaluation planning requires a {profile.split}-only dataset")
    dataset_hash = fingerprint(cases)
    if suite.dataset_sha256 and suite.dataset_sha256 != dataset_hash:
        raise ValueError("dataset fingerprint differs from the suite's pinned dataset_sha256")
    if suite.information_view not in ("native_media", "pipeline_output") and any(
        case.request.media for case in cases
    ):
        raise ValueError("media payloads require the native_media information view")
    runtime = runtime_identity(lockfile)
    cells = []
    for model_id in profile.models:
        model = registry.models[model_id]
        if model_id in enable_models:
            model = model.model_copy(update={"enabled": True})
        case_rows = [case_eligibility(model, case) for case in cases]
        reasons, service = _configuration_blockers(model, profile, environment or {})
        if model.decision_adapter_path:
            try:
                provenance = json.loads(
                    (Path(model.decision_adapter_path) / "training-provenance.json").read_text()
                )
                for case in cases:
                    for split in ("train", "calibration"):
                        if (
                            case.id in provenance[f"{split}_ids"]
                            or (case.group_id or case.id) in provenance[f"{split}_groups"]
                            or request_fingerprint(case.request) in provenance[f"{split}_requests"]
                        ):
                            reasons.append("evaluation_overlaps_adapter_training")
            except (OSError, ValueError, KeyError, TypeError):
                reasons.append("missing_adapter_training_provenance")
        if model.preprocessing != "none" and suite.track != "pipeline":
            reasons.append("preprocessor_requires_pipeline_track")
        if (
            suite.track == "pipeline"
            and model.preprocessing == "none"
            and any(c.request.media for c in cases)
        ):
            reasons.append("raw_media_pipeline_requires_preprocessor")
        if model.adapter in ("tfidf", "prior", "setfit") and model.artifact_file:
            try:
                artifact = json.loads(Path(model.artifact_file).read_text())
                for case in cases:
                    if (
                        case.id in artifact["train_ids"]
                        or (case.group_id or case.id) in artifact["train_groups"]
                        or request_fingerprint(case.request) in artifact["train_requests"]
                    ):
                        reasons.append("evaluation_overlaps_training")
                        break
            except (OSError, ValueError, KeyError, TypeError):
                reasons.append("invalid_training_provenance")
        if model.calibration == "domain":
            from .calibration import validate_calibration

            try:
                validate_calibration(model, cases)
            except (ValueError, OSError):
                reasons.append("invalid_or_overlapping_calibration")
        counts = Counter(row["eligibility"] for row in case_rows)
        decision_counts = {
            status: sum(row["decisions"] for row in case_rows if row["eligibility"] == status)
            for status in ("eligible", "unsupported", "unknown")
        }
        if counts["unknown"]:
            reasons.append("unknown_case_capabilities")
        if not counts["eligible"]:
            reasons.append("no_eligible_cases")
        measured = counts["eligible"] * profile.repetitions
        warmup = profile.warmup * profile.repetitions if counts["eligible"] else 0
        requests = measured + warmup
        unit_cost = 0 if model.adapter == "uniform" else model.cost_per_request_usd
        cost = None if unit_cost is None else Decimal(str(unit_cost)) * requests
        if unit_cost is None:
            reasons.append("unknown_cost_estimate")
        identity = {
            "baseline_revision": registry.baseline_revision,
            "model": model.model_dump(mode="json"),
            "suite": suite.model_dump(mode="json"),
            "dataset_sha256": dataset_hash,
            "profile": profile.model_dump(mode="json"),
            "runtime": runtime,
            "service": service,
        }
        cells.append(
            {
                "model_id": model.id,
                "experiment_id": _digest(identity),
                "identity": identity,
                "status": "not_run" if not model.enabled else "blocked" if reasons else "ready",
                "reasons": (["disabled"] if not model.enabled else []) + reasons,
                "coverage": {
                    "requests": {
                        key: counts[key] for key in ("eligible", "unsupported", "unknown")
                    },
                    "decisions": decision_counts,
                },
                "cases": case_rows,
                "estimate": {
                    "measured_requests": measured,
                    "warmup_requests": warmup,
                    "total_requests": requests,
                    "measured_decisions": decision_counts["eligible"] * profile.repetitions,
                    "cost_usd": float(cost) if cost is not None else None,
                },
            }
        )
    candidates = [cell for cell in cells if cell["status"] == "ready"]
    total_requests = sum(cell["estimate"]["total_requests"] for cell in candidates)
    # Recompute using decimal unit prices, not rounded binary float totals.
    total_cost = sum(
        (
            Decimal(str(cell["identity"]["model"]["cost_per_request_usd"] or 0))
            * cell["estimate"]["total_requests"]
            for cell in candidates
        ),
        Decimal(0),
    )
    budget_reasons = []
    if total_requests > profile.budget.max_requests:
        budget_reasons.append("request_budget_exceeded")
    if total_cost > Decimal(str(profile.budget.max_estimated_cost_usd)):
        budget_reasons.append("cost_budget_exceeded")
    if budget_reasons:
        for cell in candidates:
            cell["status"] = "blocked"
            cell["reasons"].extend(budget_reasons)
    statuses = Counter(cell["status"] for cell in cells)
    return {
        "schema_version": 2,
        "kind": purpose,
        "dry_run": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plan_id": _digest([cell["experiment_id"] for cell in cells]),
        "profile_id": profile.id,
        "dataset_sha256": dataset_hash,
        "runtime": runtime,
        "models": cells,
        "status_counts": {key: statuses[key] for key in ("ready", "blocked", "not_run")},
        "budget": {
            **profile.budget.model_dump(),
            "candidate_requests": total_requests,
            "candidate_cost_usd": float(total_cost),
            "reasons": budget_reasons,
        },
        "can_execute": bool(candidates) and not statuses["blocked"] and not budget_reasons,
        "limitations": [
            "Offline declaration checks only; no models loaded or endpoints contacted.",
            "Context length, hardware and media decoding remain runtime checks.",
            "Costs are operator estimates, not measured or provider-verified charges.",
            "The request contract allows at most 52 options and 64 questions per request.",
        ],
    }
