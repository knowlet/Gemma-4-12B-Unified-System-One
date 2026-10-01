"""Temperature fits are tied to calibration-only records and their base model identity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from s1.backends import normalize_response
from s1.contracts import answer_from_probabilities
from s1.training import fit_temperature

from .artifacts import read_records, write_json
from .datasets import request_fingerprint


def base_identity(model):
    config = model.model_dump(mode="json") if hasattr(model, "model_dump") else dict(model)
    policy = (
        config["base_calibration"] if config["calibration"] == "domain" else config["calibration"]
    )
    for field in (
        "id",
        "enabled",
        "notes",
        "cost_per_request_usd",
        "calibration_sha256",
        "calibration_file",
        "base_calibration",
    ):
        config.pop(field, None)
    config["calibration"] = policy
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def fit_from_run(directory, model_id, output):
    path = Path(directory)
    manifest = json.loads((path / "run_manifest.json").read_text())
    cells = [cell for cell in manifest["plan"]["models"] if cell["model_id"] == model_id]
    if len(cells) != 1 or cells[0]["identity"]["profile"].get("split", "test") != "calibration":
        raise ValueError("temperature fitting requires a calibration-only run")
    cell = cells[0]
    if cell["identity"]["model"]["calibration"] == "domain":
        raise ValueError("fit from base probabilities, not an already domain-calibrated run")
    records = [r for r in read_records(path, "predictions") if r["model_id"] == model_id]
    expected = (
        sum(cell["coverage"]["decisions"].values()) * cell["identity"]["profile"]["repetitions"]
    )
    if (
        not records
        or len(records) != expected
        or any(
            r["status"] != "ok" or r.get("split") != "calibration" or r.get("probabilities") is None
            for r in records
        )
    ):
        raise ValueError("calibration requires complete valid probability records")
    logits, golds = [], []
    for row in records:
        labels = row["labels"]
        logits.append(np.log(np.maximum([row["probabilities"][label] for label in labels], 1e-12)))
        golds.append(labels.index(row["gold"]))
    fit = fit_temperature(logits, golds)
    artifact = {
        "schema_version": 2,
        "transform": "epsilon_regularized_probability_power",
        "floor": 1e-12,
        "base_identity": base_identity(cell["identity"]["model"]),
        "source_experiment_id": cell["experiment_id"],
        "dataset_sha256": cell["identity"]["dataset_sha256"],
        "split": "calibration",
        "case_ids": sorted({r["case_id"] for r in records}),
        "group_ids": sorted({r["group_id"] for r in records}),
        "request_sha256": sorted({r["request_sha256"] for r in cell["cases"]}),
        **fit,
    }
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, artifact)
    return {**artifact, "artifact_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def validate_calibration(spec, cases=()):
    data = Path(spec.calibration_file).read_bytes()
    artifact = json.loads(data)
    if hashlib.sha256(data).hexdigest() != spec.calibration_sha256 or artifact.get(
        "base_identity"
    ) != base_identity(spec):
        raise ValueError("calibration identity or hash mismatch")
    if (
        artifact.get("split") != "calibration"
        or artifact.get("transform") != "epsilon_regularized_probability_power"
        or artifact.get("floor") != 1e-12
    ):
        raise ValueError("unsupported calibration artifact")
    for key in ("case_ids", "group_ids", "request_sha256"):
        if not isinstance(artifact.get(key), list) or not all(
            isinstance(value, str) for value in artifact[key]
        ):
            raise ValueError("invalid calibration provenance")
    temperature = artifact.get("temperature")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not np.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("invalid calibration temperature")
    for case in cases:
        if (
            case.id in artifact["case_ids"]
            or (case.group_id or case.id) in artifact["group_ids"]
            or request_fingerprint(case.request) in artifact["request_sha256"]
        ):
            raise ValueError("evaluation overlaps temperature fitting data")
    return artifact


class CalibratedAdapter:
    def __init__(self, adapter, artifact):
        self.adapter, self.temperature = adapter, artifact["temperature"]

    def predict(self, request):
        return self._apply(request, self.adapter.predict(request))

    def predict_batch(self, requests):
        responses = self.adapter.predict_batch(requests)
        if len(responses) != len(requests):
            raise ValueError("batch response count mismatch")
        return [self._apply(request, response) for request, response in zip(requests, responses)]

    def _apply(self, request, response):
        response = normalize_response(request, response)
        answers = {}
        for q in request.questions:
            original = response["answers"][q.id]
            logits = (
                np.log(np.maximum([original["probabilities"][key] for key in q.labels()], 1e-12))
                / self.temperature
            )
            p = np.exp(logits - logits.max())
            answers[q.id] = answer_from_probabilities(q, p / p.sum())
            if q.type == "choice":
                answers[q.id]["choice"] = original["choice"]
        return {**response, "answers": answers}

    def telemetry(self):
        return {**self.adapter.telemetry(), "calibration_temperature": self.temperature}

    def close(self):
        self.adapter.close()
