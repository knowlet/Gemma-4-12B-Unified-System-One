"""Native historical-data freezing and full-wire inventories; no ML or actors."""

import builtins
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from local_evidence import require_local_evidence

from s1 import browser_native_action_data as data
from s1.browser_public_training import NativeWireRequest
from s1.evaluation.datasets import request_fingerprint

ROOT = Path(__file__).absolute().parents[1]
HISTORICAL_MANIFEST = (
    ROOT
    / "artifacts/jev-runtime/owned-public-mixed-composition-20261018-01/actual-admit-02/mixed-manifest-A.json"
)


@pytest.fixture(autouse=True)
def no_heavy_import_or_actor(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {
            "torch",
            "transformers",
            "tokenizers",
            "numpy",
            "peft",
            "playwright",
        }:
            pytest.fail("Data bridge imported heavy runtime: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(
        subprocess, "Popen", lambda *_a, **_k: pytest.fail("Data bridge started actor")
    )


@pytest.fixture
def historical(no_heavy_import_or_actor):
    require_local_evidence(ROOT / data.COMPONENT["path"], HISTORICAL_MANIFEST)
    return data.load_historical_data(ROOT)


def wire(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def native_case(width=53):
    return {
        "schema_id": data.CASE_KIND,
        "id": "native-full-head",
        "split": "train",
        "group_id": "native-test-group",
        "request": {
            "model": "original-native-alias",
            "state": {"page": "Reference", "history": ["original"]},
            "questions": {
                "operation": {
                    "type": "choice",
                    "criteria": {"CLICK": "Click", "DONE": "Done"},
                    "instructions": {"goal": "Open the requested current target"},
                },
                "click_target": {
                    "type": "choice",
                    "instructions": {"goal": "Choose the exact offered target"},
                    "criteria": {str(n): "Target " + str(n) for n in range(width)},
                },
            },
        },
        "gold": {"operation": "CLICK", "click_target": "1"},
    }


@pytest.mark.local_evidence
def test_real179_original_teacher_splits_refs_and_verification(historical):
    assert historical["root"] == ROOT
    assert historical["counts"] == {"train": 142, "development": 37, "total": 179}
    assert (
        len(historical["refs"]) == 45
        and len(historical["cases"]) == len(historical["wires"]) == 179
    )
    assert all(type(case).__name__ == "SimpleNamespace" for case in historical["cases"])
    assert all(type(case.request) is NativeWireRequest for case in historical["cases"])
    assert historical["provenance"]["old_admission_types_constructed"] is False
    check = data.verify_original(historical)
    assert check["success"] and check["identity_sha256"] == check["current_identity_sha256"]
    assert check["changed"] == check["missing"] == check["added"] == []


@pytest.mark.parametrize("split,expected", [("train", 142), ("development", 37)])
@pytest.mark.local_evidence
def test_split_blobs_preserve_complete_case_gold_and_wire_projection(historical, split, expected):
    blobs = data.split_blobs(historical, split)
    assert set(blobs) == {"cases.jsonl", "wire_requests.jsonl"}
    cases = [json.loads(line) for line in blobs["cases.jsonl"].splitlines()]
    wires = [json.loads(line) for line in blobs["wire_requests.jsonl"].splitlines()]
    original = [
        (c, w)
        for c, w in zip(historical["cases"], historical["wires"], strict=True)
        if c.split == split
    ]
    assert len(cases) == len(wires) == expected
    for exported, exported_wire, (case, original_wire) in zip(cases, wires, original, strict=True):
        assert exported["schema_id"] == data.CASE_KIND
        assert exported["gold"] == case.gold and exported["group_id"] == case.group_id
        assert wire(exported["request"]) == case.request.wire_bytes
        assert wire(exported_wire) == wire(original_wire)
        assert exported_wire["teacher_choice"]["operation"] == exported["gold"]["operation"]
    inventory = data.inventory(blobs["cases.jsonl"], split)
    assert inventory == {
        "ids": sorted(c.id for c, _ in original),
        "groups": sorted({c.group_id for c, _ in original}),
        "request_fingerprints": sorted({request_fingerprint(c.request) for c, _ in original}),
    }


@pytest.mark.local_evidence
def test_exact45_freeze_relocate_reverify_without_original_root(historical, tmp_path):
    destination = tmp_path / "inputs/historical"
    descriptor = data.freeze_historical_data(historical, destination)
    assert set(descriptor) == {
        "schema_id",
        "artifact_path",
        "files",
        "identity_sha256",
        "counts",
        "component",
    }
    assert (
        descriptor["artifact_path"] == "inputs/historical"
        and descriptor["component"] == data.COMPONENT
    )
    assert descriptor["files"] == historical["refs"] and len(descriptor["files"]) == 45
    moved = tmp_path / "relocated"
    destination.rename(moved)
    restored = data.verify_frozen_historical_data(moved, descriptor)
    assert restored["root"] == moved and restored["refs"] == historical["refs"]
    assert [c.id for c in restored["cases"]] == [c.id for c in historical["cases"]]
    for split in ("train", "development"):
        assert data.split_blobs(restored, split) == data.split_blobs(historical, split)
    with pytest.raises(ValueError, match="Fresh"):
        data.freeze_historical_data(historical, moved)


@pytest.mark.parametrize(
    "mutation", ["teacher", "gold", "group", "order", "counts", "refs", "request"]
)
@pytest.mark.local_evidence
def test_in_memory_original_mutations_refuse_before_freeze(historical, tmp_path, mutation):
    changed = copy.deepcopy(historical)
    if mutation == "teacher":
        changed["wires"][0]["teacher_choice"]["target"] = "invented-target"
    elif mutation == "gold":
        changed["cases"][0].gold["operation"] = "WAIT"
    elif mutation == "group":
        changed["cases"][0].group_id = "changed-group"
    elif mutation == "order":
        changed["cases"].reverse()
        changed["wires"].reverse()
    elif mutation == "counts":
        changed["counts"]["train"] -= 1
    elif mutation == "refs":
        changed["refs"].pop(next(iter(changed["refs"])))
    else:
        changed["cases"][0].request.root["state"] = {"invented": True}
    with pytest.raises(ValueError):
        data.freeze_historical_data(changed, tmp_path / "never-created")
    assert not (tmp_path / "never-created").exists()


@pytest.mark.local_evidence
def test_frozen_corruption_missing_extra_and_symlink_refuse(historical, tmp_path):
    frozen = tmp_path / "frozen"
    descriptor = data.freeze_historical_data(historical, frozen)
    label = next(iter(descriptor["files"]))
    path = frozen / label
    original = path.read_bytes()
    path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(ValueError, match="bytes changed"):
        data.verify_frozen_historical_data(frozen, descriptor)
    path.write_bytes(original)
    path.unlink()
    with pytest.raises(ValueError, match="population"):
        data.verify_frozen_historical_data(frozen, descriptor)
    path.write_bytes(original)
    extra = frozen / "extra.json"
    extra.write_text("{}")
    with pytest.raises(ValueError, match="population"):
        data.verify_frozen_historical_data(frozen, descriptor)
    extra.unlink()
    path.unlink()
    path.symlink_to(ROOT / label)
    with pytest.raises(ValueError, match="symlink"):
        data.verify_frozen_historical_data(frozen, descriptor)


@pytest.mark.parametrize(
    "key", ["component", "counts", "identity_sha256", "files", "artifact_path"]
)
@pytest.mark.local_evidence
def test_descriptor_mutation_refuses(historical, tmp_path, key):
    frozen = tmp_path / "frozen"
    descriptor = data.freeze_historical_data(historical, frozen)
    altered = copy.deepcopy(descriptor)
    if key == "component":
        altered[key]["sha256"] = "0" * 64
    elif key == "counts":
        altered[key]["total"] -= 1
    elif key == "files":
        altered[key].pop(next(iter(altered[key])))
    else:
        altered[key] = "incorrect"
    with pytest.raises(ValueError, match="differs"):
        data.verify_frozen_historical_data(frozen, altered)


@pytest.mark.parametrize("width", [53, 400])
def test_inventory_accepts_native_full_width_and_matches_current_compiler_fingerprint(width):
    case = native_case(width)
    original = copy.deepcopy(case)
    result = data.inventory(wire(case) + b"\n", "train")
    request = NativeWireRequest.model_validate(case["request"])
    assert len(request.candidate_labels["click_target"]) == width
    assert result == {
        "ids": [case["id"]],
        "groups": [case["group_id"]],
        "request_fingerprints": [request_fingerprint(request)],
    }
    assert case == original
    assert result["request_fingerprints"][0] != request.wire_sha256


@pytest.mark.parametrize(
    "mutation", ["width401", "split", "duplicate", "gold", "schema", "jsonduplicate"]
)
def test_inventory_rejects_invalid_native_case_without_pruning(mutation):
    case = native_case(401 if mutation == "width401" else 53)
    if mutation == "split":
        case["split"] = "development"
    elif mutation == "gold":
        case["gold"]["click_target"] = "not-offered"
    elif mutation == "schema":
        case.pop("schema_id")
    blob = wire(case) + b"\n"
    if mutation == "duplicate":
        blob += blob
    elif mutation == "jsonduplicate":
        blob = blob.replace(b'"id":', b'"id":"duplicate-key","id":', 1)
    with pytest.raises(ValueError):
        data.inventory(blob, "train")


@pytest.mark.local_evidence
def test_component_source_must_match_exact_frozen_pin(historical, tmp_path, monkeypatch):
    target = tmp_path / data.COMPONENT["path"]
    target.parent.mkdir(parents=True)
    original = (ROOT / data.COMPONENT["path"]).read_bytes()
    assert hashlib.sha256(original).hexdigest() == data.COMPONENT["sha256"]
    target.write_bytes(original.replace(b"Historical", b"historical", 1))
    monkeypatch.setattr(data, "CODE_ROOT", tmp_path)
    with pytest.raises(ValueError, match="bytes changed"):
        data.load_historical_data(historical["root"])
