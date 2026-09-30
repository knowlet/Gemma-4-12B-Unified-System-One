"""Matched media views, counterfactual checks and measured OCR/ASR service pipelines."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from s1.backends import HTTPBackend
from s1.errors import BackendResponseError

from .artifacts import read_records, write_json
from .datasets import EvaluationCase, fingerprint
from .experiments import write_cases


def prepare_media_views(source, output):
    """Bundle rows contain a native case and separately annotated missing-media gold.

    Groups with differing gold must have identical text/questions and different
    media, making the intervention inspectable before any model run.
    """
    rows = [json.loads(line) for line in Path(source).read_text().splitlines() if line.strip()]
    views = {"M0": [], "M1": [], "M2": [], "M3": []}
    groups = defaultdict(list)
    for row in rows:
        case = EvaluationCase.model_validate(row["case"])
        if not case.request.media or not case.group_id:
            raise ValueError("media cases need actual media and an explicit source group")
        groups[case.group_id].append(case)
        views["M1"].append(case)
        views["M3"].append(case)
        missing = case.model_dump(mode="json")
        missing["request"]["media"] = []
        # No default: missing required information must be deliberately relabeled.
        missing["gold"] = row["missing_media_gold"]
        missing["soft_gold"] = None
        views["M0"].append(EvaluationCase.model_validate(missing))
        if not isinstance(row.get("verified_text"), str) or not row["verified_text"].strip():
            raise ValueError("M2 requires a human-verified transcript/description for every case")
        transcript = case.model_dump(mode="json")
        transcript["request"]["media"] = []
        transcript["request"]["state"] = {
            "state": case.request.state,
            "verified_media_description": row["verified_text"],
        }
        views["M2"].append(EvaluationCase.model_validate(transcript))
    if not rows or len({c.id for c in views["M1"]}) != len(rows):
        raise ValueError("empty bundle or duplicate case IDs")
    pairs = []
    for group, cases in groups.items():
        for a, b in combinations(cases, 2):
            if a.gold == b.gold:
                continue
            if a.request.state != b.request.state or a.request.questions != b.request.questions:
                raise ValueError("counterfactual pairs must keep text and questions fixed")
            if a.request.media == b.request.media:
                raise ValueError("counterfactual pair did not change media")
            pairs.append({"group_id": group, "left": a.id, "right": b.id})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, cases in views.items():
        write_cases(output / f"{name}.jsonl", cases)
    manifest = {
        "schema_version": 2,
        "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "views": {
            name: {"dataset": f"{name}.jsonl", "dataset_sha256": fingerprint(cases)}
            for name, cases in views.items()
        },
        "counterfactual_pairs": pairs,
        "M2_policy": "human-verified diagnostic; not free product capability",
        "M3_policy": "raw media passed through a measured, pinned OCR/ASR service",
    }
    write_json(output / "media-views.json", manifest)
    return manifest


class PipelineAdapter:
    def __init__(self, adapter, preprocessor):
        from threading import local

        self.adapter, self.preprocessor = adapter, preprocessor
        self._local = local()

    def predict(self, request):
        try:
            return self._predict(request)
        except Exception as exc:
            exc.evaluation_pipeline = getattr(self._local, "pipeline", {})
            raise

    def _predict(self, request):
        started = time.perf_counter()
        self._local.pipeline = {}
        try:
            response = self.preprocessor._post(
                {"media": [m.model_dump(mode="json") for m in request.media]}
            )
        finally:
            self._local.pipeline = {"preprocess_ms": (time.perf_counter() - started) * 1000}
        if not isinstance(response, dict):
            raise BackendResponseError("preprocessor must return text")
        cost = response.get("cost_usd")
        if cost is not None and (
            type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0
        ):
            raise BackendResponseError("invalid reported preprocessing cost")
        self._local.pipeline["preprocess_reported_cost_usd"] = cost
        if not isinstance(response.get("text"), str):
            raise BackendResponseError("preprocessor must return text")
        converted = request.model_copy(
            update={
                "media": [],
                "state": {"state": request.state, "media_observation": response["text"]},
            }
        )
        decision_started = time.perf_counter()
        try:
            result = self.adapter.predict(converted)
        finally:
            self._local.pipeline["decision_ms"] = (time.perf_counter() - decision_started) * 1000
        return {
            **result,
            "pipeline": dict(self._local.pipeline),
        }

    def predict_batch(self, requests):
        # Explicit loop-emulated policy; no claim that separate services batch.
        return [self.predict(request) for request in requests]

    def telemetry(self):
        return self.adapter.telemetry()

    def close(self):
        try:
            self.adapter.close()
        finally:
            self.preprocessor.close()


def wrap_pipeline(adapter, spec, environment):
    service = HTTPBackend(
        environment[spec.preprocessor_endpoint_env],
        token=environment.get(spec.preprocessor_token_env) if spec.preprocessor_token_env else None,
        timeout=spec.http_timeout_seconds,
    )
    return PipelineAdapter(adapter, service)


def counterfactual_summary(directory, model, views_manifest):
    manifest = json.loads(Path(views_manifest).read_text())
    run = json.loads((Path(directory) / "run_manifest.json").read_text())
    if run["status"] in ("running", "interrupted") or run["execution"].get(model, {}).get(
        "status"
    ) not in ("completed", "completed_with_errors"):
        raise ValueError("counterfactual scoring requires a finished executed model")
    if run["plan"]["dataset_sha256"] != manifest["views"]["M1"]["dataset_sha256"]:
        raise ValueError("counterfactual run does not match the prepared native media dataset")
    rows = [r for r in read_records(directory, "predictions") if r["model_id"] == model]
    if not rows:
        raise ValueError("missing model predictions")
    cell = next(c for c in run["plan"]["models"] if c["model_id"] == model)
    expected = (
        sum(cell["coverage"]["decisions"].values()) * cell["identity"]["profile"]["repetitions"]
    )
    if len(rows) != expected:
        raise ValueError("counterfactual scoring requires complete decision records")
    lookup = {(r["case_id"], r["question_id"], r["repetition"]): r for r in rows}
    if len(lookup) != len(rows):
        raise ValueError("duplicate prediction records")
    pairs, both_correct, valid_pairs, flips = 0, 0, 0, 0
    for pair in manifest["counterfactual_pairs"]:
        for key, left in lookup.items():
            if key[0] != pair["left"]:
                continue
            right = lookup.get((pair["right"], key[1], key[2]))
            if right is None:
                raise ValueError("missing counterfactual decision")
            if left["gold"] == right["gold"]:
                continue
            if left["eligibility"] != "eligible" or right["eligibility"] != "eligible":
                continue
            pairs += 1
            if left["status"] == right["status"] == "ok":
                valid_pairs += 1
                flips += left["actual_label"] != right["actual_label"]
                both_correct += (
                    left["actual_label"] == left["gold"] and right["actual_label"] == right["gold"]
                )
    return {
        "required_change_pairs": pairs,
        "both_valid": valid_pairs,
        "conditional_flip_rate": flips / valid_pairs if valid_pairs else None,
        "operational_pair_accuracy": both_correct / pairs if pairs else None,
        "interpretation": "both answers must be correct; changing answers alone is insufficient",
    }
