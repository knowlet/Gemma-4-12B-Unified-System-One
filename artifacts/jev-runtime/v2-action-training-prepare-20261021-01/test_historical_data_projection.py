"""Historical data tests only; no processors, models, browser or old proof actors."""

import builtins
import copy
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).absolute().parent
SPEC = importlib.util.spec_from_file_location(
    "historical_projection_tested", HERE / "historical_data_projection.py"
)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


@pytest.fixture(autouse=True)
def no_frameworks(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "tokenizers", "peft", "safetensors"}:
            raise AssertionError("Historical data loader attempted framework import: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.fixture(scope="module")
def genuine():
    if not (M.ROOT / next(iter(M.AUTHORITIES))).exists():
        pytest.skip("Ignored historical data is not distributed")
    return M.load_historical_data()


@pytest.fixture
def preserved(tmp_path, monkeypatch, genuine):
    for label in genuine["refs"]:
        target = tmp_path / label
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(M.ROOT / label, target)
    monkeypatch.setattr(M, "AUTHORITIES", dict(M.AUTHORITIES))
    return tmp_path


def read(root, label):
    return json.loads((root / label).read_bytes())


def write(root, label, value):
    blob = M.wire(value)
    (root / label).write_bytes(blob)
    if label in M.AUTHORITIES:
        M.AUTHORITIES[label] = M.sha(blob)


def public_repin(root, label, value):
    """Externally reseal a corrupt test input so semantic joins, not SHA alone, run."""
    blob = b"\n".join(M.wire(row) for row in value) + b"\n"
    (root / label).write_bytes(blob)
    pin = {"sha256": M.sha(blob), "size_bytes": len(blob)}
    sub = label.removeprefix(M.PUB + "/")
    manifest = read(root, M.MIX + "/mixed-manifest-A.json")
    manifest["components"]["public"]["public_inventory"]["files"][sub] = pin
    write(root, M.MIX + "/mixed-manifest-A.json", manifest)
    receipt_label = (
        "artifacts/jev-browser/20261016-public-native-training-07-admission-01/receipt.json"
    )
    receipt = read(root, receipt_label)
    receipt["full_closed_input_inventory"]["files"][sub] = pin
    write(root, receipt_label, receipt)
    receipt = read(root, M.MIX + "/receipt.json")
    receipt["external_manifest_pins"]["A"] = M.AUTHORITIES[M.MIX + "/mixed-manifest-A.json"]
    write(root, M.MIX + "/receipt.json", receipt)


def test_duplicate_nonfinite_and_nonobject_authorities(preserved):
    for value in ('{"x":1,"x":2}', '{"x":NaN}'):
        with pytest.raises(ValueError):
            M.parse(value)
    label = M.MIX + "/receipt.json"
    write(preserved, label, [])
    with pytest.raises(ValueError, match="authority must be an object"):
        M.load_historical_data(preserved)


def test_complete_real179_protocol_and_history_scope(genuine):
    assert genuine["counts"] == {"train": 142, "development": 37, "total": 179}
    assert len(genuine["cases"]) == len(genuine["wires"]) == 179
    assert len(genuine["train_ids"]) == 142 and len(genuine["dev_ids"]) == 37
    assert len({c.id for c in genuine["cases"]}) == 179
    assert [c.split for c in genuine["cases"]] == ["train"] * 142 + ["development"] * 37
    for case, row in zip(genuine["cases"], genuine["wires"], strict=True):
        assert case.id == row["case_id"] and case.request.wire_bytes == M.wire(row["payload"])
        assert case.gold["operation"] == row["teacher_choice"]["operation"]
    assert genuine["provenance"]["schema_id"] == M.KIND
    assert genuine["provenance"]["old_admission_types_constructed"] is False
    assert genuine["provenance"]["new_labels_created"] is False
    assert not {"torch", "transformers", "tokenizers", "peft"} & set(sys.modules)


def test_physical_relocation_is_data_only_and_exact(preserved, genuine):
    moved = M.load_historical_data(preserved)
    assert moved["refs"] == genuine["refs"]
    assert M.wire(moved["wires"]) == M.wire(genuine["wires"])
    assert moved["train_ids"] == genuine["train_ids"]


@pytest.mark.parametrize("mutation", ["bytes", "symlink", "missing"])
def test_original_data_drift_refuses(preserved, mutation):
    path = preserved / M.PUB / "train/cases.jsonl"
    if mutation == "bytes":
        path.write_bytes(path.read_bytes() + b" ")
    elif mutation == "symlink":
        source = path.with_suffix(".copy")
        path.rename(source)
        path.symlink_to(source)
    else:
        path.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        M.load_historical_data(preserved)


@pytest.mark.parametrize("mutation", ["gold", "order", "count", "teacher", "targets"])
def test_resigned_public_data_must_still_join_annotations_and_teacher(preserved, mutation):
    name = "cases.jsonl" if mutation in {"gold", "order", "count"} else "wire_requests.jsonl"
    label = M.PUB + "/train/" + name
    rows = [json.loads(b) for b in (preserved / label).read_bytes().splitlines()]
    if mutation == "gold":
        rows[0]["gold"]["click_target"] = "not the teacher"
    elif mutation == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif mutation == "count":
        rows.pop()
    elif mutation == "teacher":
        rows[0]["teacher_choice"]["target"] = "unoffered"
    else:
        rows[0]["valid_next_targets"] = {"CLICK": []}
    public_repin(preserved, label, rows)
    with pytest.raises(ValueError, match="teacher|Complete public|annotation/wire"):
        M.load_historical_data(preserved)


@pytest.mark.parametrize("mutation", ["wire-sha", "split", "family", "duplicate", "proof-index"])
def test_resigned_form_terminal_must_still_join_closed_data(preserved, mutation):
    label = M.FORM + "/actual-collection-verification-01/terminal.json"
    terminal = read(preserved, label)
    if mutation == "duplicate":
        terminal["all23_rows"][1] = copy.deepcopy(terminal["all23_rows"][0])
    elif mutation == "proof-index":
        terminal["provenance"]["proof_index_sha256"] = "0" * 64
    else:
        field = {"wire-sha": "native_wire_sha256", "split": "split", "family": "family"}[mutation]
        terminal["all23_rows"][0][field] = "corrupt"
    write(preserved, label, terminal)
    exit_label = M.FORM + "/root-collection-verification01-exit-review-01.json"
    closure = read(preserved, exit_label)
    closure["terminal"]["sha256"] = M.AUTHORITIES[label]
    write(preserved, exit_label, closure)
    with pytest.raises(ValueError, match="form|Form|Unique179|Complete historical"):
        M.load_historical_data(preserved)
