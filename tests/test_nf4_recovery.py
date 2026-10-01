"""Recovery rejects changed data/lineage before any GPU model is loaded."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "nf4_recovery", Path(__file__).resolve().parents[1] / "scripts/nf4_recovery.py"
)
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


def test_recovery_recipe_is_fixed_for_all_three_seeds():
    recipes = [recovery.recipe(seed) for seed in range(3)]
    assert {r["additional_optimizer_updates"] for r in recipes} == {100}
    assert {r["learning_rate"] for r in recipes} == {2e-5}
    assert {r["brier_weight"] for r in recipes} == {0.0}
    assert len({r["parent_adapter_sha256"] for r in recipes}) == 3
    assert all(r["no_test_fitting"] and r["recipe_fixed_before_execution"] for r in recipes)
    with pytest.raises(ValueError):
        recovery.recipe(3)
    with pytest.raises(ValueError):
        recovery.recipe(True)


def test_recovery_rejects_changed_training_data_before_loading_model(monkeypatch):
    monkeypatch.setattr(recovery, "verify_datasets", lambda _: {})
    monkeypatch.setattr(recovery, "load_cases", lambda _: [object()] * 256)
    monkeypatch.setattr(recovery, "fingerprint", lambda _: "changed")
    with pytest.raises(ValueError, match="training dataset changed"):
        recovery.prepare_data("unused")


def test_recovery_lineage_checks_all_six_identity_collections(tmp_path, monkeypatch):
    provenance = {
        f"{split}_{identity}": [f"{split}-a", f"{split}-b"]
        for split in ("train", "calibration")
        for identity in ("ids", "groups", "requests")
    }
    path = tmp_path / "training-provenance.json"
    path.write_text(json.dumps({k: list(reversed(v)) for k, v in provenance.items()}))
    monkeypatch.setattr(recovery, "directory_digest", lambda _: recovery.CE_DIGESTS[0])
    recovery.verify_parent(tmp_path, 0, provenance)
    changed = json.loads(path.read_text())
    changed["calibration_requests"][0] = "test-request"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="calibration_requests"):
        recovery.verify_parent(tmp_path, 0, provenance)


def test_recovery_rejects_parent_digest_change_before_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "directory_digest", lambda _: "changed")
    with pytest.raises(ValueError, match="parent adapter digest"):
        recovery.verify_parent(tmp_path, 0, {})


@pytest.mark.parametrize("size", [65, 16777217, 1342177280, 2**40])
def test_frozen_weight_audit_indices_remain_in_bounds_for_huge_embeddings(size):
    indices = recovery.audit_indices(size)
    assert len(indices) == 64
    assert indices[0] == 0
    assert indices[-1] == size - 1
    assert all(type(index) is int and 0 <= index < size for index in indices)
    assert len(set(indices)) == 64
