"""Independent CPU FP64 numerical audit of one dual-head GPU witness.

The producer/owner must authenticate the single native forward, witness capture,
source/runtime, exact CPU tensor bytes and external manifest pin separately.
This module performs no model calls and cannot establish browser or HTTP success.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import re
import stat
from pathlib import Path

import numpy as np

HEADS = ("operation", "click_target")
WIDTH, CODES = 1024, 255
W0_SHA256 = "d408d94a0ab2915fa85c511dd9b8d62af49903c4e4ca28e2e2d5aff9ef565e0f"
SCHEMA = "decision3-dual-head-gpu-witness-audit-v1"
FILE_SCHEMA = "decision3-dual-head-gpu-witness-files-v1"
TOL = 1e-5


def _need(value, message):
    if not value:
        raise ValueError(message)


def _encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _digest_string(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _matrix(value, shape):
    _need(
        type(value) is np.ndarray
        and value.shape == shape
        and value.dtype == np.float32
        and value.flags.c_contiguous
        and bool(np.isfinite(value).all()),
        "finite contiguous CPU FP32 witness matrix required",
    )
    return value


def _matrix_sha(value):
    return _sha(value.tobytes(order="C"))


def witness_manifest(witness):
    """Canonical commitment to every evidence value and FP32 tensor byte.

    Externally authenticate its SHA before audit. Arrays must already have been
    loaded from authenticated no-pickle files by an independent owner.
    """
    fields = {
        "native_payload",
        "lengths",
        "ordered_head_ids",
        "ordered_criteria_ids",
        "features",
        "baseline",
        "deltas",
        "delta_sha256",
        "baseline_raw",
        "zero_raw",
        "composed_raw",
        "projection",
    }
    _need(
        type(witness) is dict and set(witness) == fields,
        "exact persisted dual-head witness required",
    )
    heads = witness["ordered_head_ids"]
    _need(
        type(heads) is list
        and 1 <= len(heads) <= 4
        and all(type(key) is str and key for key in heads)
        and len(set(heads)) == len(heads),
        "one to four unique ordered native head IDs required",
    )
    features = _matrix(witness["features"], (len(heads), WIDTH))
    baseline = _matrix(witness["baseline"], (CODES, WIDTH))
    deltas = witness["deltas"]
    _need(type(deltas) is dict and set(deltas) == set(HEADS), "exact two delta matrices required")
    matrices = {key: _matrix(deltas[key], (CODES, WIDTH)) for key in HEADS}
    _need(
        type(witness["delta_sha256"]) is dict
        and set(witness["delta_sha256"]) == set(HEADS)
        and all(witness["delta_sha256"][head] == _matrix_sha(matrices[head]) for head in HEADS),
        "captured delta byte hashes differ",
    )
    return {
        "schema": SCHEMA,
        "features_sha256": _matrix_sha(features),
        "baseline_sha256": _matrix_sha(baseline),
        "delta_sha256": {head: _matrix_sha(matrices[head]) for head in HEADS},
        **{
            key: witness[key]
            for key in (
                "native_payload",
                "lengths",
                "ordered_head_ids",
                "ordered_criteria_ids",
                "baseline_raw",
                "zero_raw",
                "composed_raw",
                "projection",
            )
        },
    }


def _native_answer(raw, payload, lengths):
    _need(
        type(raw) is dict
        and raw.get("model") == "d3-edge"
        and type(raw.get("answers")) is dict
        and list(raw["answers"]) == list(payload["questions"]),
        "complete native raw answer order required",
    )
    usage = raw.get("usage")
    _need(
        type(usage) is dict
        and set(usage) == {"input_tokens", "output_tokens"}
        and all(type(value) is int for value in usage.values())
        and usage == {"input_tokens": sum(lengths.values()), "output_tokens": 0},
        "actual native usage differs",
    )
    _encode(raw)
    for head, question in payload["questions"].items():
        answer = raw["answers"][head]
        _need(
            type(answer) is dict and answer.get("type") == "choice" and not answer.get("error"),
            "native choice response required",
        )
        probabilities = answer.get("probabilities")
        _need(
            type(probabilities) is dict and list(probabilities) == list(question["criteria"]),
            "complete original criterion order required",
        )
        values = list(probabilities.values())
        _need(
            all(
                type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
                for value in values
            )
            and abs(sum(values) - 1) <= 1e-6,
            "finite normalized native probability vector required",
        )
        _need(
            answer.get("choice") == max(probabilities, key=probabilities.__getitem__),
            "actual native argmax differs",
        )
        entropy = -sum(value * math.log(value) for value in values if value > 0)
        confidence = (
            1.0 if len(values) == 1 else max(0.0, min(1.0, 1.0 - entropy / math.log(len(values))))
        )
        reported = answer.get("confidence")
        _need(
            type(reported) in (int, float)
            and math.isfinite(reported)
            and 0 <= reported <= 1
            and abs(reported - confidence) <= 1e-6,
            "native entropy confidence differs",
        )
        if len(values) == 1:
            _need(
                values == [1.0] and reported == 1.0,
                "native singleton requires an exact point mass and confidence",
            )


def _softmax(logits):
    values = logits - np.max(logits)
    exp = np.exp(values)
    return exp / exp.sum()


def _compare(raw, expected, heads, criteria):
    worst = 0.0
    for index, head in enumerate(heads):
        reported = np.asarray(
            list(raw["answers"][head]["probabilities"].values()), dtype=np.float64
        )
        want = expected[index]
        _need(len(reported) == len(want), "all legal probabilities required")
        error = float(np.max(np.abs(reported - want)))
        _need(
            error <= TOL
            and raw["answers"][head]["choice"] == criteria[index][int(np.argmax(want))],
            "full FP64 replay or argmax differs",
        )
        worst = max(worst, error)
    return worst


def audit(witness, *, expected_manifest_sha256, expected_delta_sha256):
    """Replay baseline, GPU zero control and composed response in CPU FP64.

    Passing confirms numerical consistency of *provided* bytes only. It does
    not authenticate producer timing, single-forward provenance, owner cleanup,
    model quality or the integrity of caller-supplied manifest/delta pins.
    """
    _need(_digest_string(expected_manifest_sha256), "external witness SHA required")
    _need(
        type(expected_delta_sha256) is dict
        and set(expected_delta_sha256) == set(HEADS)
        and all(_digest_string(value) for value in expected_delta_sha256.values()),
        "external two-delta SHA pins required",
    )
    manifest = witness_manifest(witness)
    manifest_sha = _sha(_encode(manifest))
    _need(manifest_sha == expected_manifest_sha256, "witness commitment differs")
    _need(
        manifest["baseline_sha256"] == W0_SHA256
        and manifest["delta_sha256"] == expected_delta_sha256,
        "original W0 or external delta bytes differ",
    )
    payload, lengths = witness["native_payload"], witness["lengths"]
    heads, criteria = witness["ordered_head_ids"], witness["ordered_criteria_ids"]
    _need(
        type(payload) is dict
        and set(payload) == {"state", "questions"}
        and type(payload["questions"]) is dict
        and list(payload["questions"]) == heads,
        "original complete native payload and head order required",
    )
    _need(
        type(lengths) is dict
        and list(lengths) == heads
        and all(type(value) is int and 0 < value <= 16384 for value in lengths.values()),
        "complete native prepared lengths required",
    )
    _need(type(criteria) is list and len(criteria) == len(heads), "all ordered criteria required")
    for index, head in enumerate(heads):
        question = payload["questions"][head]
        _need(
            type(question) is dict
            and set(question) == {"type", "instructions", "criteria"}
            and question["type"] == "choice"
            and type(question["criteria"]) is dict
            and 1 <= len(question["criteria"]) <= CODES
            and criteria[index] == list(question["criteria"]),
            "complete original ordered candidate inventory required",
        )
    applied, absent = (
        [head for head in heads if head in HEADS],
        [head for head in HEADS if head not in heads],
    )
    projection = witness["projection"]
    _need(
        type(projection) is dict
        and set(projection)
        == {
            "applied_head_ids",
            "absent_head_ids",
            "legal_counts",
            "ordered_head_ids",
            "same_call_baseline_replayed",
            "unmodified_heads_preserved",
        }
        and projection.get("ordered_head_ids") == heads
        and projection.get("applied_head_ids") == applied
        and projection.get("absent_head_ids") == absent
        and projection.get("legal_counts") == [len(row) for row in criteria]
        and projection.get("same_call_baseline_replayed") is True
        and projection.get("unmodified_heads_preserved") is True,
        "original head mapping and legal counts required",
    )
    for name in ("baseline_raw", "zero_raw", "composed_raw"):
        _native_answer(witness[name], payload, lengths)
    features = witness["features"].astype(np.float64)
    baseline = witness["baseline"].astype(np.float64)
    deltas = {head: witness["deltas"][head].astype(np.float64) for head in HEADS}
    base_logits = features @ baseline.T
    baseline_expected, zero_expected, composed_expected = [], [], []
    for index, head in enumerate(heads):
        legal = len(criteria[index])
        base = base_logits[index, :legal]
        baseline_expected.append(_softmax(base))
        zero_expected.append(_softmax(base + features[index] @ np.zeros_like(baseline[:legal]).T))
        delta = (
            features[index] @ deltas[head][:legal].T
            if head in HEADS
            else np.zeros(legal, dtype=np.float64)
        )
        composed_expected.append(_softmax(base + delta))
        if head not in HEADS:
            _need(
                _encode(witness["baseline_raw"]["answers"][head])
                == _encode(witness["composed_raw"]["answers"][head]),
                "unadapted raw answer bytes changed",
            )
    worst = {
        "baseline": _compare(witness["baseline_raw"], baseline_expected, heads, criteria),
        "gpu_zero": _compare(witness["zero_raw"], zero_expected, heads, criteria),
        "candidate": _compare(witness["composed_raw"], composed_expected, heads, criteria),
    }
    return {
        "schema": SCHEMA,
        "status": "numeric_consistency_only",
        "witness_sha256": manifest_sha,
        "baseline_sha256": manifest["baseline_sha256"],
        "delta_sha256": manifest["delta_sha256"],
        "heads": heads,
        "applied_head_ids": applied,
        "absent_head_ids": absent,
        "legal_counts": [len(row) for row in criteria],
        "legal_probability_values": sum(map(len, criteria)),
        "max_probability_absolute_error": worst,
        "fit_eligible": False,
        "browser_qualified": False,
        "http_qualified": False,
        "model_score": None,
        "scope": "CPU FP64 replay of pinned witness; producer/owner provenance and actual single-forward timing require separate verification",
    }


def _write_new(path, raw):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def make_witness_writer(root, *, maximum_calls):
    """Create a private, bounded witness sink and an in-process receipt list.

    The caller must seal the returned receipts externally after its owner closes.
    A failed callback leaves a quarantined incomplete directory.
    """
    root = Path(root).absolute()
    _need(
        type(maximum_calls) is int and 1 <= maximum_calls <= 32,
        "explicit bounded witness count required",
    )
    _need(
        root.parent.is_dir() and root.parent.resolve() == root.parent,
        "ordinary witness parent required",
    )
    root.mkdir(mode=0o700, exist_ok=False)
    receipts = []

    def sink(*, torch, features, baseline, deltas, **metadata):
        del torch
        _need(len(receipts) < maximum_calls, "witness count exceeded")
        directory = root / f"call-{len(receipts):04d}"
        directory.mkdir(mode=0o700, exist_ok=False)

        def cpu_copy(tensor):
            # A real GPU tensor must be detached and independently copied;
            # dtype/shape are checked by witness_manifest, not coerced.
            return np.ascontiguousarray(tensor.detach().cpu().contiguous().numpy()).copy()

        witness = copy.deepcopy(metadata)
        witness.update(
            features=cpu_copy(features),
            baseline=cpu_copy(baseline),
            deltas={key: cpu_copy(value) for key, value in deltas.items()},
        )
        commitment = witness_manifest(witness)
        commitment_sha = _sha(_encode(commitment))
        arrays = {
            "features.npy": witness["features"],
            "baseline.npy": witness["baseline"],
            "delta-operation.npy": witness["deltas"]["operation"],
            "delta-click-target.npy": witness["deltas"]["click_target"],
        }
        refs = {}
        for name, array in arrays.items():
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            raw = buffer.getvalue()
            _write_new(directory / name, raw)
            refs[name] = {
                "sha256": _sha(raw),
                "size_bytes": len(raw),
                "fp32_sha256": _matrix_sha(array),
            }
        document = {
            "schema": FILE_SCHEMA,
            "witness_sha256": commitment_sha,
            "metadata": {
                key: value
                for key, value in witness.items()
                if key not in {"features", "baseline", "deltas"}
            },
            "arrays": refs,
        }
        raw = _encode(document)
        _need(len(raw) <= 8 * 1024**2, "bounded private witness JSON required")
        _write_new(directory / "witness.json", raw)
        receipts.append(
            {
                "directory": str(directory),
                "manifest_sha256": _sha(raw),
                "witness_sha256": commitment_sha,
                "delta_sha256": dict(witness["delta_sha256"]),
                "scope": "pending external owner and numerical audit",
            }
        )

    return sink, receipts


def _pairs(items):
    result = {}
    for key, value in items:
        _need(key not in result, "duplicate witness JSON key")
        result[key] = value
    return result


def _read_file(directory_fd, name, *, limit):
    _need(
        type(name) is str and name and "/" not in name and name not in {".", ".."},
        "single ordinary witness file name required",
    )
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        _need(
            stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit,
            "bounded ordinary witness file required",
        )
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    _need(
        stat.S_ISREG(current.st_mode)
        and len(raw) == before.st_size
        and all(
            getattr(before, key) == getattr(after, key) == getattr(current, key) for key in identity
        ),
        "witness file changed during read",
    )
    return raw


def _load_npy(raw, expected_shape):
    buffer = io.BytesIO(raw)
    version = np.lib.format.read_magic(buffer)
    _need(version in {(1, 0), (2, 0)}, "supported bounded NPY version required")
    reader = (
        np.lib.format.read_array_header_1_0
        if version == (1, 0)
        else np.lib.format.read_array_header_2_0
    )
    shape, fortran, dtype = reader(buffer, max_header_size=4096)
    _need(
        shape == expected_shape and fortran is False and dtype == np.dtype("<f4"),
        "exact bounded little-endian FP32 NPY header required",
    )
    _need(len(raw) - buffer.tell() == math.prod(shape) * 4, "exact FP32 NPY data length required")
    value = np.load(io.BytesIO(raw), allow_pickle=False, max_header_size=4096)
    return _matrix(value, expected_shape)


def audit_directory(
    directory, *, expected_file_sha256, expected_witness_sha256, expected_delta_sha256
):
    """Load only an externally pinned closed witness directory and replay it."""
    _need(
        _digest_string(expected_file_sha256) and _digest_string(expected_witness_sha256),
        "external file and witness SHA pins required",
    )
    directory = Path(directory).absolute()
    _need(
        directory.is_dir() and directory.resolve() == directory,
        "ordinary witness directory required",
    )
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        names = {
            "witness.json",
            "features.npy",
            "baseline.npy",
            "delta-operation.npy",
            "delta-click-target.npy",
        }
        _need(set(os.listdir(fd)) == names, "closed exact witness inventory required")
        raw = _read_file(fd, "witness.json", limit=8 * 1024**2)
        _need(_sha(raw) == expected_file_sha256, "external witness file SHA differs")
        document = json.loads(
            raw,
            object_pairs_hook=_pairs,
            parse_constant=lambda _: _need(False, "nonfinite witness JSON"),
        )
        _need(
            type(document) is dict
            and set(document) == {"schema", "witness_sha256", "metadata", "arrays"}
            and document["schema"] == FILE_SCHEMA
            and document["witness_sha256"] == expected_witness_sha256,
            "exact externally pinned witness document required",
        )
        meta, refs = document["metadata"], document["arrays"]
        _need(
            type(meta) is dict
            and type(meta.get("ordered_head_ids")) is list
            and 1 <= len(meta["ordered_head_ids"]) <= 4
            and type(refs) is dict
            and set(refs) == names - {"witness.json"},
            "complete witness metadata and tensor inventory required",
        )
        shapes = {
            "features.npy": (len(meta["ordered_head_ids"]), WIDTH),
            "baseline.npy": (CODES, WIDTH),
            "delta-operation.npy": (CODES, WIDTH),
            "delta-click-target.npy": (CODES, WIDTH),
        }
        arrays = {}
        for name, shape in shapes.items():
            ref = refs[name]
            _need(
                type(ref) is dict
                and set(ref) == {"sha256", "size_bytes", "fp32_sha256"}
                and _digest_string(ref["sha256"])
                and _digest_string(ref["fp32_sha256"])
                and type(ref["size_bytes"]) is int
                and ref["size_bytes"] > 0,
                "exact witness tensor reference required",
            )
            encoded = _read_file(fd, name, limit=2 * 1024**2)
            _need(
                len(encoded) == ref["size_bytes"] and _sha(encoded) == ref["sha256"],
                "witness NPY file pin differs",
            )
            value = _load_npy(encoded, shape)
            _need(_matrix_sha(value) == ref["fp32_sha256"], "original witness FP32 bytes differ")
            arrays[name] = value
        witness = copy.deepcopy(meta)
        witness.update(
            features=arrays["features.npy"],
            baseline=arrays["baseline.npy"],
            deltas={
                "operation": arrays["delta-operation.npy"],
                "click_target": arrays["delta-click-target.npy"],
            },
        )
        _need(set(os.listdir(fd)) == names, "witness inventory changed during read")
        return audit(
            witness,
            expected_manifest_sha256=expected_witness_sha256,
            expected_delta_sha256=expected_delta_sha256,
        )
    finally:
        os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--witness-dir", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--witness-sha256", required=True)
    parser.add_argument("--operation-delta-sha256", required=True)
    parser.add_argument("--click-target-delta-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = audit_directory(
        args.witness_dir,
        expected_file_sha256=args.manifest_sha256,
        expected_witness_sha256=args.witness_sha256,
        expected_delta_sha256={
            "operation": args.operation_delta_sha256,
            "click_target": args.click_target_delta_sha256,
        },
    )
    _write_new(Path(args.output), _encode(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
