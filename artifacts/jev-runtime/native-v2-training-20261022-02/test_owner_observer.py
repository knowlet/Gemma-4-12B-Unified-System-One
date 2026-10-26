"""Owner/probe binding tests; production absence evidence is tested separately."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).absolute().parents[3]
PROBE = "artifacts/jev-runtime/gx10-boot-identity-20261022-01/comfy_observer.py"
OWNERS = (
    "artifacts/jev-runtime/native-v2-training-20261022-02/launch_training.py",
    "artifacts/jevbench/20261022-local-score-01/launch_score_with_observer.py",
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
    path = owner.ROOT / PROBE
    path.parent.mkdir(parents=True)
    payload = (
        "def observe(*, expected_boot_id):\n"
        '    return {"boot_id": expected_boot_id, "comfy_state": "fixture_only"}\n'
    ).encode()
    path.write_bytes(payload)
    ref = {"path": PROBE, "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
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
                owner.HERE / ("actual-01" if "jevbench" in str(owner.__file__) else "fit-01")
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
