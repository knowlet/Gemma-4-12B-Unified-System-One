"""Optional, pinned 52-row browser readout; the original LM head stays intact.

Artifacts contain only FP32 linear rows and development provenance. Prediction
uses hidden states, option counts and the pinned softcap, never example IDs,
request fingerprints or gold labels. Deployment temperature remains the base
model's temperature; a training objective may declare its own temperature.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import re
from pathlib import Path

from .schema import LETTERS

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_SCHEMA = "browser_readout_v1"
_MAX_MANIFEST_BYTES = 4_000_000
_PIN_KEYS = {
    "base_model_id",
    "base_revision",
    "quantization",
    "prompt_variant",
    "candidate_ids",
    "base_temperature",
    "temperature",
    "softcap",
    "hidden_size",
}
_MANIFEST_KEYS = _PIN_KEYS | {
    "schema",
    "kind",
    "weights_file",
    "weights_sha256",
    "projection_precision",
    "provenance",
}


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _softcap(value):
    return None if value is None else _positive(value, "softcap")


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ValueError(f"{name} must be nonempty text of at most 512 characters")
    return value


def _json_bytes(value):
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode(
            "utf-8"
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("readout metadata must contain finite JSON data") from exc


def _provenance(value, *, learned):
    required = {"source_scope", "train", "calibration"}
    if not isinstance(value, dict) or not required <= set(value):
        raise ValueError("readout provenance requires source_scope, train and calibration")
    if set(value) - required - {"training_objective"}:
        raise ValueError("unsupported readout provenance fields")
    if value["source_scope"] != "development_only":
        raise ValueError("readout provenance must explicitly use development_only data")
    result = copy.deepcopy(value)
    for split in ("train", "calibration"):
        inventory = value[split]
        keys = {"ids", "groups", "request_fingerprints"}
        if not isinstance(inventory, dict) or set(inventory) != keys:
            raise ValueError(f"{split} provenance needs ids, groups and request_fingerprints")
        for key in keys:
            items = inventory[key]
            if not isinstance(items, list):
                raise ValueError(f"{split} {key} must be a list")
            for item in items:
                _text(item, f"{split} {key}")
                if key == "request_fingerprints" and _SHA256.fullmatch(item) is None:
                    raise ValueError("request fingerprints must be lowercase SHA256 hex")
            if len(set(items)) != len(items):
                raise ValueError(f"{split} {key} must contain unique inventory entries")
        occupied = [bool(inventory[key]) for key in keys]
        if any(occupied) and not all(occupied):
            raise ValueError(f"{split} provenance inventories must all be populated or all empty")
    for key in ("ids", "groups", "request_fingerprints"):
        if set(value["train"][key]) & set(value["calibration"][key]):
            raise ValueError(f"train and calibration {key} must be disjoint")
    if learned and not value["train"]["ids"]:
        raise ValueError("learned readout rows require recorded development training provenance")
    if "training_objective" in value and not isinstance(value["training_objective"], dict):
        raise ValueError("training_objective must be a finite JSON object")
    _json_bytes(result)
    return result


def _model_pins(model, prompt_variant):
    import torch

    name = _text(getattr(model, "name", None), "base model ID")
    revision = getattr(model, "revision", None)
    if not isinstance(revision, str) or _REVISION.fullmatch(revision) is None:
        raise ValueError("base model revision must be an immutable 40-character commit")
    quantization = _text(getattr(model, "quantization", None), "quantization")
    prompt_variant = _text(prompt_variant, "prompt variant")
    head = getattr(model, "head", None)
    weight = getattr(head, "weight", None)
    if (
        not torch.is_tensor(weight)
        or weight.ndim != 2
        or not weight.is_floating_point()
        or weight.shape[1] < 1
    ):
        raise ValueError("original LM head must expose a floating vocabulary-by-hidden matrix")
    letters = getattr(model, "letters", None)
    integer_dtypes = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
    if (
        not torch.is_tensor(letters)
        or letters.ndim != 1
        or len(letters) != len(LETTERS)
        or letters.dtype not in integer_dtypes
    ):
        raise ValueError("model.letters must contain the original 52 integer candidate token IDs")
    ids = letters.detach().cpu().tolist()
    if len(set(ids)) != len(LETTERS) or any(index < 0 or index >= weight.shape[0] for index in ids):
        raise ValueError("candidate token IDs must be distinct rows in the original LM head")
    temperature = _positive(getattr(model, "temperature", None), "base temperature")
    return {
        "base_model_id": name,
        "base_revision": revision,
        "quantization": quantization,
        "prompt_variant": prompt_variant,
        "candidate_ids": ids,
        "base_temperature": temperature,
        "temperature": temperature,
        "softcap": _softcap(getattr(model, "softcap", None)),
        "hidden_size": weight.shape[1],
    }


def _rows(weight, bias, hidden_size):
    import torch

    expected = (len(LETTERS), hidden_size)
    if not torch.is_tensor(weight) or tuple(weight.shape) != expected:
        raise ValueError(f"readout weight must have shape {expected}")
    if not torch.is_tensor(bias) or tuple(bias.shape) != (len(LETTERS),):
        raise ValueError("readout bias must have shape (52,)")
    if not weight.is_floating_point() or not bias.is_floating_point():
        raise ValueError("readout rows and bias must be floating tensors")
    weight = weight.detach().to(device="cpu", dtype=torch.float32).clone().contiguous()
    bias = bias.detach().to(device="cpu", dtype=torch.float32).clone().contiguous()
    if not torch.isfinite(weight).all().item() or not torch.isfinite(bias).all().item():
        raise ValueError("readout rows and bias must be finite in FP32")
    return weight, bias


def write_readout(
    path,
    model,
    prompt_variant,
    *,
    weight=None,
    bias=None,
    provenance,
    temperature=None,
):
    """Write a new immutable artifact, using original rows or supplied learned rows.

    Provenance inventories are unique sets, not parallel per-row label arrays.
    An empty calibration inventory explicitly means no calibration data was used.
    Omitting rows creates the untrained FP32 control from ``model.letters``;
    supplying rows requires both a weight and bias and nonempty training provenance.
    """
    import torch

    pins = _model_pins(model, prompt_variant)
    if temperature is not None and _positive(temperature, "temperature") != pins["temperature"]:
        raise ValueError("readout deployment temperature must equal the pinned base temperature")
    learned = weight is not None or bias is not None
    if learned and (weight is None or bias is None):
        raise ValueError("supply both learned weight and learned bias")
    if not learned:
        ids = torch.tensor(pins["candidate_ids"], device=model.head.weight.device)
        weight = model.head.weight[ids]
        full_bias = getattr(model.head, "bias", None)
        if full_bias is None:
            bias = torch.zeros(len(LETTERS), dtype=torch.float32)
        elif (
            not torch.is_tensor(full_bias)
            or full_bias.ndim != 1
            or len(full_bias) != model.head.weight.shape[0]
        ):
            raise ValueError("original LM head bias must match its vocabulary rows")
        else:
            bias = full_bias[ids.to(full_bias.device)]
    weight, bias = _rows(weight, bias, pins["hidden_size"])
    provenance = _provenance(provenance, learned=learned)
    stream = io.BytesIO()
    torch.save({"weight": weight, "bias": bias}, stream)
    payload = stream.getvalue()
    manifest = {
        "schema": _SCHEMA,
        "kind": "learned_rows" if learned else "untrained_fp32_selected_rows_control",
        **pins,
        "projection_precision": "float32",
        "weights_file": "weights.pt",
        "weights_sha256": hashlib.sha256(payload).hexdigest(),
        "provenance": provenance,
    }
    encoded = _json_bytes(manifest)
    if len(encoded) > _MAX_MANIFEST_BYTES:
        raise ValueError("readout manifest exceeds its byte limit")
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "weights.pt").write_bytes(payload)
    (destination / "manifest.json").write_bytes(encoded + b"\n")
    return copy.deepcopy(manifest)


class BrowserReadout:
    """Frozen FP32 rows; provenance is telemetry and cannot route predictions."""

    def __init__(self, weight, bias, manifest, path):
        self._weight, self._bias = weight, bias
        self._manifest = copy.deepcopy(manifest)
        self._path = str(Path(path).resolve())
        self.temperature = manifest["temperature"]
        self._device_rows = {"cpu": (weight, bias)}

    def project(self, hidden, counts, softcap):
        import torch
        from torch.nn import functional as F

        if _softcap(softcap) != self._manifest["softcap"]:
            raise ValueError("readout softcap differs from its pinned base model")
        if (
            not torch.is_tensor(hidden)
            or hidden.ndim != 2
            or hidden.shape[0] < 1
            or hidden.shape[1] != self._manifest["hidden_size"]
            or not hidden.is_floating_point()
        ):
            raise ValueError(
                "readout hidden states must be a nonempty batch-by-hidden float matrix"
            )
        integer_dtypes = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
        if (
            not torch.is_tensor(counts)
            or counts.ndim != 1
            or counts.shape[0] != hidden.shape[0]
            or counts.dtype not in integer_dtypes
            or not ((counts >= 1) & (counts <= len(LETTERS))).all().item()
        ):
            raise ValueError("readout counts must contain one integer in 1..52 per hidden row")
        hidden = hidden.float()
        if not torch.isfinite(hidden).all().item():
            raise ValueError("readout hidden states must be finite in FP32")
        key = str(hidden.device)
        if key not in self._device_rows:
            self._device_rows[key] = (
                self._weight.to(hidden.device),
                self._bias.to(hidden.device),
            )
        weight, bias = self._device_rows[key]
        # Even an outer BF16 autocast must leave this explicitly FP32 control
        # and learned sidecar in FP32. The original model's full head is untouched.
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            logits = F.linear(hidden, weight, bias)
            if not torch.isfinite(logits).all().item():
                raise ValueError("readout projection produced nonfinite logits")
            if softcap is not None:
                logits = torch.tanh(logits / softcap) * softcap
        if not torch.isfinite(logits).all().item():
            raise ValueError("readout projection produced nonfinite logits")
        mask = (
            torch.arange(len(LETTERS), device=hidden.device)[None]
            >= counts.to(hidden.device)[:, None]
        )
        return logits.masked_fill(mask, float("-inf"))

    def telemetry(self):
        return {**copy.deepcopy(self._manifest), "artifact_path": self._path}


def load_readout(path, model, prompt_variant):
    """Validate pins and the exact hashed bytes, then load tensor-only weights."""
    import torch

    pins = _model_pins(model, prompt_variant)
    directory = Path(path)
    manifest_path = directory / "manifest.json"
    if manifest_path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ValueError("readout manifest exceeds its byte limit")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ValueError("readout manifest must be valid JSON") from exc
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise ValueError("unsupported readout manifest fields")
    if manifest["schema"] != _SCHEMA or manifest["projection_precision"] != "float32":
        raise ValueError("unsupported readout schema or projection precision")
    if manifest["kind"] not in ("learned_rows", "untrained_fp32_selected_rows_control"):
        raise ValueError("unsupported readout artifact kind")
    if not isinstance(manifest["candidate_ids"], list) or any(
        type(value) is not int for value in manifest["candidate_ids"]
    ):
        raise ValueError("readout candidate IDs must be the original integer token IDs")
    for key in _PIN_KEYS:
        if type(manifest[key]) is not type(pins[key]) or manifest[key] != pins[key]:
            raise ValueError(f"readout {key} differs from the configured model or prompt")
    _provenance(manifest["provenance"], learned=manifest["kind"] == "learned_rows")
    if (
        manifest["weights_file"] != "weights.pt"
        or not isinstance(manifest["weights_sha256"], str)
        or _SHA256.fullmatch(manifest["weights_sha256"]) is None
    ):
        raise ValueError("readout weights must name weights.pt and an exact SHA256")
    weights_path = directory / "weights.pt"
    maximum = (len(LETTERS) * pins["hidden_size"] + len(LETTERS)) * 4 + 1_000_000
    if weights_path.stat().st_size > maximum:
        raise ValueError("readout weights exceed the expected tensor byte limit")
    payload = weights_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest["weights_sha256"]:
        raise ValueError("readout weights SHA256 differs from the manifest")
    try:
        data = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("readout weights could not be loaded as safe tensors") from exc
    if not isinstance(data, dict) or set(data) != {"weight", "bias"}:
        raise ValueError("readout weights require only weight and bias tensors")
    if any(not torch.is_tensor(value) or value.dtype != torch.float32 for value in data.values()):
        raise ValueError("readout saved weights must be FP32 tensors")
    weight, bias = _rows(data["weight"], data["bias"], pins["hidden_size"])
    return BrowserReadout(weight, bias, manifest, directory)
