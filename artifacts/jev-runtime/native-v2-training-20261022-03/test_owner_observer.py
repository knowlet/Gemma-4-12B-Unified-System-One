"""Owner/probe binding tests; production absence evidence is tested separately."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).absolute().parents[3]
OLD_PROBE = "artifacts/jev-runtime/comfy-observer-vanished-context-20261022-01/comfy_observer.py"
NEW_PROBE = "artifacts/jev-runtime/comfy-observer-resolved-context-20261022-01/comfy_observer.py"
OWNERS = (
    "artifacts/jev-runtime/native-v2-training-20261022-03/launch_training.py",
    "artifacts/jevbench/20261022-local-score-01/launch_score_02b.py",
)


@pytest.fixture(params=OWNERS)
def owner(request, tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("_test_owned_probe_binding", ROOT / request.param)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "HERE", tmp_path / "run")
    return module


def selected_probe(owner):
    # This fake proves byte/boot handoff only, never real Comfy absence.
    probe = NEW_PROBE if Path(owner.__file__).name == "launch_training.py" else OLD_PROBE
    path = owner.ROOT / probe
    path.parent.mkdir(parents=True)
    payload = (
        "def observe(*, expected_boot_id):\n"
        '    return {"boot_id": expected_boot_id, "comfy_state": "fixture_only"}\n'
    ).encode()
    path.write_bytes(payload)
    ref = {"path": probe, "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
    return {"comfy_observer": ref, "sources": [ref], "boot_id": "selected-boot"}, path


def test_default_dry_never_loads_guard_or_probe(owner, monkeypatch, capsys):
    def forbidden(*_args):
        pytest.fail("Default dry imported an operational observer")

    monkeypatch.setattr(owner, "load_guard", forbidden)
    monkeypatch.setattr(owner, "load_observer", forbidden)
    assert owner.main([]) == 0
    assert "DRY_NO_ACTOR_OR_MODEL" in capsys.readouterr().out
    assert not owner.HERE.exists()


def test_selected_probe_preserves_boot_argument(owner):
    plan, _ = selected_probe(owner)
    observe = owner.load_observer(plan)
    assert observe() == {"boot_id": "selected-boot", "comfy_state": "fixture_only"}


def test_changed_probe_bytes_refuse_before_execution(owner):
    plan, path = selected_probe(owner)
    path.write_text('raise RuntimeError("Unselected code executed")\n')
    with pytest.raises(AssertionError):
        owner.load_observer(plan)
    assert not owner.HERE.exists()


def test_probe_requires_full_source_population_binding(owner):
    plan, _ = selected_probe(owner)
    plan["sources"] = []
    with pytest.raises(AssertionError):
        owner.load_observer(plan)


def test_another_ordinary_python_file_cannot_be_selected_as_probe(owner):
    plan, path = selected_probe(owner)
    alternate = owner.ROOT / "unreviewed.py"
    alternate.write_bytes(path.read_bytes())
    plan["comfy_observer"] = {**plan["comfy_observer"], "path": "unreviewed.py"}
    plan["sources"] = [plan["comfy_observer"]]
    with pytest.raises(AssertionError):
        owner.load_observer(plan)


def test_linked_probe_refuses_before_execution(owner):
    plan, path = selected_probe(owner)
    actual = path.with_name("actual.py")
    path.rename(actual)
    path.symlink_to(actual)
    with pytest.raises(AssertionError):
        owner.load_observer(plan)


def test_sibling_observer_profile_refuses_before_execution(owner):
    plan, path = selected_probe(owner)
    alternate = OLD_PROBE if plan["comfy_observer"]["path"] == NEW_PROBE else NEW_PROBE
    other = owner.ROOT / alternate
    other.parent.mkdir(parents=True, exist_ok=True)
    other.write_bytes(path.read_bytes())
    plan["comfy_observer"] = {**plan["comfy_observer"], "path": alternate}
    plan["sources"] = [plan["comfy_observer"]]
    with pytest.raises(AssertionError):
        owner.load_observer(plan)


def test_unconfirmed_comfy_status_refuses_before_any_launch(owner, monkeypatch, tmp_path):
    failure = RuntimeError("Comfy state remains unknown")

    def observe():
        raise failure

    def forbidden(*_args, **_kwargs):
        pytest.fail("Unknown Comfy state reached a process/unit command")

    monkeypatch.setattr(owner.resource, "setrlimit", lambda *_args: None)
    monkeypatch.setattr(owner, "load_guard", lambda: (SimpleNamespace(), {}))
    monkeypatch.setattr(owner, "load_observer", lambda _plan: observe)
    monkeypatch.setattr(owner, "command", forbidden)
    plan = {
        "unit": owner.UNIT,
        "cgroup": owner.CGROUP,
        "output": str(
            (
                owner.HERE / ("actual-02" if "jevbench" in str(owner.__file__) else "fit-01")
            ).relative_to(owner.ROOT)
        ),
        "unit_memory_max_bytes": owner.CAP,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "sources": [],
    }
    path = tmp_path / "input-plan.json"
    payload = json.dumps(plan).encode()
    path.write_bytes(payload)
    with pytest.raises(RuntimeError) as caught:
        owner.main(
            ["--run", "--plan", str(path), "--plan-sha256", hashlib.sha256(payload).hexdigest()]
        )
    assert caught.value is failure
    assert not owner.HERE.exists()


def test_actual_probe_error_evidence_survives_owner_record(owner):
    failure = RuntimeError("unknown")
    failure.evidence = [
        {
            "processes": {
                "uncertain": [{"pid": 123, "classification": "uncertain_process_identity_changed"}]
            }
        }
    ]
    assert owner.error_record(failure)["observation_evidence"] == failure.evidence

    def refuse():
        raise failure

    assert owner.observe_final_host(refuse)["observation_error"]


def test_large_observation_evidence_is_bounded(owner):
    failure = RuntimeError("unknown")
    failure.evidence = {"sample": "x" * 70000}
    record = owner.error_record(failure)["observation_evidence"]
    assert record["truncated"] and record["size_bytes"] > 65536
    assert len(record["sha256"]) == 64


@pytest.mark.parametrize(
    "change,valid",
    [
        ({}, True),
        ({"comfy_state": "http_busy", "comfy_running": 1}, True),
        ({"observation_error": {"type": "ObservationError"}}, False),
        ({"boot_id": "wrong"}, False),
        ({"mem_total_bytes": 129}, False),
        ({"mem_available_bytes": 0}, False),
        ({"comfy_state": "http_idle", "comfy_running": 1}, False),
        ({"comfy_running": False}, False),
    ],
)
def test_native_final_host_must_be_known(owner, change, valid):
    native = owner
    row = {
        "boot_id": "boot",
        "mem_total_bytes": 130,
        "mem_available_bytes": 100,
        "comfy_running": 0,
        "comfy_pending": 0,
        "comfy_state": "absent_confirmed",
    }
    if "observation_error" in change:
        row = change
    else:
        row.update(change)
    assert native.known_final_host(row, {"boot_id": "boot", "expected_memtotal": 130}) is valid


def test_unexecuted_plan02_preserves_recipe_and_exact104_sources():
    directory = ROOT / "artifacts/jev-runtime/native-v2-training-20261022-03"
    old = json.loads((directory / "launch-plan-01.json").read_bytes())
    new = json.loads((directory / "launch-plan-02.json").read_bytes())
    assert len(new["sources"]) == len({r["path"] for r in new["sources"]}) == 104
    for row in new["sources"]:
        blob = (ROOT / row["path"]).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == row["sha256"]
        assert len(blob) == row["size_bytes"]
    for key in (
        "fitter_arguments unit cgroup output unit_memory_max_bytes boot_id expected_memtotal "
        "nonce capacity_qualified model_loads training_updates scope"
    ).split():
        assert new[key] == old[key]
    assert new["comfy_observer"]["path"] == NEW_PROBE
    assert new["comfy_observer"]["sha256"] == (
        "103804b05e3047011aa5daf099c04ee71a60b3e1435c0690dc7b0377bd486df7"
    )
    assert len(new["observer_successor_changes"]) == 2
    old_owner = (directory / "launch_training_522.py").read_bytes()
    new_owner = (directory / "launch_training.py").read_bytes()
    assert new_owner == old_owner.replace(OLD_PROBE.encode(), NEW_PROBE.encode())
    assert hashlib.sha256(old_owner).hexdigest() == (
        "0c31929e0217e9612b6933ea1b806e8230b03f687b466d98797f066ec2d50509"
    )
    active = ROOT / "artifacts/jev-runtime/native-v2-browser-execution-20261022-02/raw-02/selected-plan.json"
    raw = json.loads(active.read_bytes())
    assert len(raw["sources"]) == 621
    assert not any(
        r["path"].startswith("artifacts/jev-runtime/native-v2-training-20261022-03/")
        for r in raw["sources"]
    )
