"""Readiness-gated metadata only; fake observations, no services or models."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P = load("fresh_owner05_preparation", HERE / "prepare_plan.py")
ORIGINAL = load("fresh_owner05_original_host", HERE / "q95_owner.py")


@pytest.fixture
def fake(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "HERE", tmp_path)
    for name in ("q95_owner.py", "launch_q95.py"):
        (tmp_path / name).write_bytes((HERE / name).read_bytes())
    source = tmp_path / "artifacts/jev-runtime/fake-observer.py"
    source.parent.mkdir(parents=True)
    source.write_text("# simulated observation source only\n")

    def ref(path):
        blob = path.read_bytes()
        return {
            "path": str(path),
            "sha256": hashlib.sha256(blob).hexdigest(),
            "size_bytes": len(blob),
        }

    row = {
        "boot_id": "selected-boot",
        "mem_total_bytes": 130595864576,
        "mem_available_bytes": 100_000_000_000,
        "comfy_state": "http_idle",
        "comfy_running": 0,
        "comfy_pending": 0,
    }
    calls, options = [], {"drift": None, "after_observe": lambda: None}

    def observe(*, expected_boot_id):
        calls.append("observe")
        assert expected_boot_id == row["boot_id"]
        options["after_observe"]()
        return copy.deepcopy(row)

    def prepare(output):
        calls.append("prepare")
        assert output == "owner-05"
        components = {
            "observer": ref(source),
            "owner": ref(tmp_path / "q95_owner.py"),
            "entry": ref(tmp_path / "launch_q95.py"),
        }
        if options["drift"]:
            components[options["drift"]]["sha256"] = "0" * 64
        return {
            "unit": "jev-d3-q95-helper-20261023-05.service",
            "boot_id": row["boot_id"],
            "mem_total_bytes": row["mem_total_bytes"],
            "components": components,
        }

    def check_ref(value):
        calls.append("source")
        return ORIGINAL.check_ref(value)

    def observer_load(value, _name):
        calls.append("load_observer")
        assert value == ref(source)
        return SimpleNamespace(observe=observe)

    def validate(path, digest):
        calls.append("validate")
        blob = path.read_bytes()
        assert hashlib.sha256(blob).hexdigest() == digest
        assert json.loads(blob)["unit"] == "jev-d3-q95-helper-20261023-05.service"

    owner = SimpleNamespace(
        ROOT=tmp_path,
        SERVICE=tmp_path / "serve-05",
        FIXED={"observer": ("fake-observer.py", ref(source)["sha256"])},
        prepare=prepare,
        check_ref=check_ref,
        load=observer_load,
        host=ORIGINAL.host,
        encode=ORIGINAL.encode,
        validate=validate,
        error_record=ORIGINAL.error_record,
    )
    return SimpleNamespace(
        owner=owner,
        row=row,
        calls=calls,
        options=options,
        root=tmp_path,
        target=tmp_path / "launch-plan-01.json",
    )


def test_default_dry_never_loads_owner_or_observer(monkeypatch, capsys):
    monkeypatch.setattr(P, "load_owner", lambda: pytest.fail("Dry mode loaded owner"))
    assert P.main([]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "DRY_SOURCE_ONLY"


def test_actual_checked_source_loader_binds_fresh_slots_only():
    owner = P.load_owner()
    assert owner.UNIT == "jev-d3-q95-helper-20261023-05.service"
    assert owner.SERVICE.name == "serve-05"
    assert owner.PROFILE == "decision3-q95-guarded-owner-host-diagnostics-v3"


def test_low_memory_preserves_real_host_refusal_without_sealing(fake):
    fake.row["mem_available_bytes"] = 29_000_000_000
    with pytest.raises(ValueError) as caught:
        P.seal(fake.owner, fake.target)
    assert caught.value.evidence["gate"] == "memory_reserve"
    assert caught.value.evidence["thresholds"]["min_mem_available_bytes"] == 73958271732
    assert not fake.target.exists() and "validate" not in fake.calls
    assert not (fake.root / "owner-05").exists()


def test_busy_comfy_preserves_original_refusal(fake):
    fake.row.update(comfy_state="http_busy", comfy_running=1)
    with pytest.raises(ValueError) as caught:
        P.seal(fake.owner, fake.target)
    assert caught.value.evidence["gate"] == "host_identity_and_comfy_idle"
    assert not fake.target.exists()


def test_ready_host_seals_only_metadata_and_validates(fake):
    result = P.seal(fake.owner, fake.target)
    assert result["status"] == "fresh_metadata_sealed_not_launched"
    assert not result["unit_started"] and not result["model_loaded"]
    assert result["quality_score"] is None
    assert fake.target.stat().st_mode & 0o777 == 0o600
    assert fake.calls.index("observe") < fake.calls.index("validate")
    assert not (fake.root / "owner-05").exists() and not fake.owner.SERVICE.exists()


@pytest.mark.parametrize("used", ["plan", "service", "output"])
def test_old_slot_or_plan_cannot_be_reused(fake, used):
    if used == "plan":
        fake.target.write_text("prior plan")
    elif used == "service":
        fake.owner.SERVICE.mkdir()
    else:
        (fake.root / "owner-05").mkdir()
    with pytest.raises(ValueError):
        P.seal(fake.owner, fake.target)
    assert "prepare" not in fake.calls
    if used == "plan":
        assert fake.target.read_text() == "prior plan"


@pytest.mark.parametrize("component", ["observer", "owner", "entry"])
def test_source_authority_drift_refused_before_observer_executes(fake, component):
    fake.options["drift"] = component
    with pytest.raises(ValueError):
        P.seal(fake.owner, fake.target)
    assert "load_observer" not in fake.calls and not fake.target.exists()


def test_owner_source_drift_during_observation_cannot_seal(fake):
    fake.options["after_observe"] = lambda: (fake.root / "q95_owner.py").write_text(
        "changed source"
    )
    with pytest.raises(ValueError):
        P.seal(fake.owner, fake.target)
    assert "observe" in fake.calls and not fake.target.exists()


def test_cli_retains_typed_low_memory_evidence_and_no_plan(fake, monkeypatch, capsys):
    fake.row["mem_available_bytes"] = 29_000_000_000
    monkeypatch.setattr(P, "load_owner", lambda: fake.owner)
    assert P.main(["--seal"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["error"]["observation_evidence"]["gate"] == "memory_reserve"
    assert not result["unit_started"] and not fake.target.exists()


def test_plan_path_is_scoped_before_preparing_anything(fake):
    with pytest.raises(ValueError):
        P.seal(fake.owner, fake.root.parent / "launch-plan-01.json")
    assert not fake.calls
