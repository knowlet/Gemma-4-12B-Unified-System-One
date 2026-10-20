"""Native action data bridge over one immutable historical-data reader.

This authenticates historical bytes and teacher/split joins. It does not admit an
old collection type, qualify execution sources, load a processor, or train a model.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import stat
from pathlib import Path

CODE_ROOT = Path(__file__).absolute().parents[2]
COMPONENT = {
    "path": "artifacts/jev-runtime/v2-action-training-prepare-20261021-01/historical_data_projection.py",
    "sha256": "abefbe19229b557c618dbf90f0722b7d71e92552b94c85a6c2289521968d05b5",
    "size_bytes": 15999,
}
KIND = "jev-native-historical-action-data-v1"
CASE_KIND = "jev-native-action-case-v1"
COUNTS = {"train": 142, "development": 37, "total": 179}
_DATA_KEYS = {"cases", "wires", "refs", "train_ids", "dev_ids", "counts", "provenance", "root"}


def _require(value, message):
    if not value:
        raise ValueError(message)


def _wire(value, *, canonical=False):
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=canonical, allow_nan=False
    ).encode("utf-8")


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _parse(blob):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate native data JSON key")
            result[key] = value
        return result

    return json.loads(
        blob,
        object_pairs_hook=pairs,
        parse_constant=lambda _: _require(False, "Nonfinite native data JSON"),
    )


def _ordinary(path):
    path = Path(path).absolute()
    _require(
        ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "Native data symlink or parent traversal",
    )
    return path


def _read(root, label, pin):
    relative = Path(label)
    _require(
        type(label) is str
        and not relative.is_absolute()
        and ".." not in relative.parts
        and relative.as_posix() == label
        and bool(relative.parts),
        "Native data relative path",
    )
    _require(
        type(pin) is dict
        and set(pin) == {"sha256", "size_bytes"}
        and type(pin["size_bytes"]) is int
        and 0 <= pin["size_bytes"] <= 8 * 1024**2
        and type(pin["sha256"]) is str
        and len(pin["sha256"]) == 64,
        "Bounded native data pin",
    )
    path = _ordinary(root / relative)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        _require(
            stat.S_ISREG(before.st_mode) and before.st_size == pin["size_bytes"],
            "Regular pinned native data file",
        )
        with os.fdopen(fd, "rb", closefd=False) as stream:
            blob = stream.read(pin["size_bytes"] + 1)
        after = os.fstat(fd)
    finally:
        os.close(fd)

    def signature(s):
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns

    _require(
        signature(before) == signature(after) == signature(path.stat())
        and len(blob) == pin["size_bytes"]
        and _sha(blob) == pin["sha256"],
        "Native data bytes changed",
    )
    return blob


def _reader():
    path = CODE_ROOT / COMPONENT["path"]
    blob = _read(CODE_ROOT, COMPONENT["path"], {k: v for k, v in COMPONENT.items() if k != "path"})
    spec = importlib.util.spec_from_file_location("_native_historical_data_component", path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(blob, str(path), "exec"), module.__dict__)
    return module


def load_historical_data(root):
    """Load all179 cases through the byte-pinned reader, without old admission DTOs."""
    root = _ordinary(root)
    _require(root.is_dir(), "Native historical data root required")
    data = _reader().load_historical_data(root)
    _require(
        data["counts"] == COUNTS and len(data["refs"]) == 45, "Complete historical179/45 authority"
    )
    return {**data, "root": root}


def _case(case):
    return {
        "schema_id": CASE_KIND,
        "id": case.id,
        "split": case.split,
        "group_id": case.group_id,
        "request": _parse(case.request.wire_bytes),
        "gold": copy.deepcopy(case.gold),
    }


def _view(data):
    _require(type(data) is dict and set(data) == _DATA_KEYS, "Exact historical data bridge object")
    return {
        "cases": [_case(case) for case in data["cases"]],
        **{
            key: data[key]
            for key in ("wires", "refs", "train_ids", "dev_ids", "counts", "provenance")
        },
    }


def verify_original(data):
    """Recheck original files and every in-memory case, full wire, teacher and split."""
    _require(type(data) is dict and set(data) == _DATA_KEYS, "Exact historical data bridge object")
    current = load_historical_data(data["root"])
    _require(_wire(_view(data)) == _wire(_view(current)), "Historical data object changed")
    identity = _sha(_wire(current["refs"], canonical=True))
    return {
        "schema_id": KIND,
        "success": True,
        "identity_sha256": identity,
        "current_identity_sha256": identity,
        "changed": [],
        "missing": [],
        "added": [],
        "counts": dict(COUNTS),
        "ordered_cases_sha256": _sha(_wire(_view(current)["cases"])),
        "ordered_wires_sha256": _sha(_wire(current["wires"])),
    }


def _descriptor(data):
    return {
        "schema_id": KIND,
        "artifact_path": "inputs/historical",
        "files": copy.deepcopy(data["refs"]),
        "identity_sha256": _sha(_wire(data["refs"], canonical=True)),
        "counts": dict(COUNTS),
        "component": dict(COMPONENT),
    }


def freeze_historical_data(data, destination):
    """Copy exactly45 original files into a fresh root; return no execution qualification."""
    verify_original(data)
    root, destination = _ordinary(data["root"]), _ordinary(destination)
    _require(not destination.exists(), "Fresh frozen historical destination required")
    _require(not root.is_relative_to(destination), "Frozen destination overlaps original root")
    destination.mkdir(parents=True, mode=0o700)
    for label, pin in data["refs"].items():
        blob = _read(root, label, pin)
        target = _ordinary(destination / label)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
    descriptor = _descriptor(data)
    verify_frozen_historical_data(destination, descriptor)
    verify_original(data)
    return descriptor


def verify_frozen_historical_data(destination, descriptor):
    """Replay the same fixed reader on relocated original paths; reject tree extras."""
    _require(
        type(descriptor) is dict
        and set(descriptor)
        == {"schema_id", "artifact_path", "files", "identity_sha256", "counts", "component"},
        "Exact frozen native historical descriptor",
    )
    _require(
        descriptor["schema_id"] == KIND
        and descriptor["artifact_path"] == "inputs/historical"
        and descriptor["counts"] == COUNTS
        and descriptor["component"] == COMPONENT
        and type(descriptor["files"]) is dict
        and len(descriptor["files"]) == 45
        and descriptor["identity_sha256"] == _sha(_wire(descriptor["files"], canonical=True)),
        "Frozen historical component/count/identity differs",
    )
    destination = _ordinary(destination)
    _require(destination.is_dir(), "Frozen historical directory required")
    found = set()
    for folder, dirs, files in os.walk(destination, followlinks=False):
        _require(
            all(not (Path(folder) / name).is_symlink() for name in dirs), "Frozen directory symlink"
        )
        for name in files:
            path = _ordinary(Path(folder) / name)
            _require(stat.S_ISREG(path.stat().st_mode), "Frozen historical ordinary file")
            found.add(path.relative_to(destination).as_posix())
    _require(found == set(descriptor["files"]), "Frozen historical file population differs")
    data = load_historical_data(destination)
    _require(
        data["refs"] == descriptor["files"] and _descriptor(data) == descriptor,
        "Frozen historical files differ from original authority",
    )
    return data


def split_blobs(data, split):
    """New native cases plus unchanged compiler wire projections, in original order."""
    _require(split in {"train", "development"}, "Exact native train/development split")
    verify_original(data)
    selected = [
        (case, wire)
        for case, wire in zip(data["cases"], data["wires"], strict=True)
        if case.split == split
    ]
    _require(len(selected) == COUNTS[split], "Complete native split")
    return {
        "cases.jsonl": b"".join(_wire(_case(case)) + b"\n" for case, _ in selected),
        "wire_requests.jsonl": b"".join(_wire(wire) + b"\n" for _, wire in selected),
    }


def inventory(blob, split):
    """Legacy provenance keys over full native requests; no EvaluationCase conversion."""
    from .browser_public_training import NativeWireRequest
    from .evaluation.datasets import request_fingerprint
    from .jev_generated import TARGETS

    _require(
        split in {"train", "development"} and isinstance(blob, (bytes, str)),
        "Native inventory input",
    )
    _require(len(blob) <= 64 * 1024**2, "Bounded complete native cases")
    cases = [_parse(line) for line in blob.splitlines() if line.strip()]
    _require(bool(cases), "Nonempty native cases required")
    ids, groups, fingerprints = [], [], []
    for case in cases:
        _require(
            type(case) is dict
            and set(case) == {"schema_id", "id", "split", "group_id", "request", "gold"}
            and case["schema_id"] == CASE_KIND
            and case["split"] == split
            and all(type(case[k]) is str and bool(case[k]) for k in ("id", "group_id")),
            "Exact native case identity/split/schema",
        )
        request = NativeWireRequest.model_validate(case["request"])
        labels, gold = request.candidate_labels, case["gold"]
        _require(
            type(gold) is dict
            and type(gold.get("operation")) is str
            and gold["operation"] in labels["operation"],
            "Native full gold operation",
        )
        head = TARGETS.get(gold["operation"])
        _require(
            set(gold) == ({"operation", head} if head else {"operation"})
            and (head is None or type(gold[head]) is str and gold[head] in labels[head]),
            "Native full gold target",
        )
        ids.append(case["id"])
        groups.append(case["group_id"])
        fingerprints.append(request_fingerprint(request))
    _require(len(ids) == len(set(ids)), "Duplicate native case IDs")
    return {
        "ids": sorted(ids),
        "groups": sorted(set(groups)),
        "request_fingerprints": sorted(set(fingerprints)),
    }
