"""Invented source bytes only: no model, dataset, cases or external execution."""

import hashlib
from types import SimpleNamespace

import jev_decision3_cuda_synthetic_bootstrap as boot
import jev_decision3_cuda_synthetic_launcher as launcher
import pytest


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def sources(tmp_path, monkeypatch):
    root = tmp_path / "root"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    monkeypatch.setattr(boot, "ROOT", root)
    monkeypatch.setattr(boot, "SCRIPTS", scripts)
    monkeypatch.setattr(boot, "SOURCE_ONLY", False)
    monkeypatch.setattr(boot, "sys", SimpleNamespace(modules={}))
    pins = {}
    for name in boot.FILES:
        held, released = scripts / boot.FILES[name], boot._release(name)
        source = ("SOURCE_ONLY = True\nlabel = " + repr(name) + "\n").encode()
        target = source.replace(b"SOURCE_ONLY = True", b"SOURCE_ONLY = False")
        for path, raw in ((held, source), (released, target)):
            path.write_bytes(raw)
            pins[str(path)] = digest(raw)
    for _name, relative in boot.SUPPORT:
        path = scripts / relative
        raw = ("value = " + repr(relative) + "\n").encode()
        path.write_bytes(raw)
        pins[str(path)] = digest(raw)
    for _name, relative, _package in boot.PROJECT:
        path = root / "src/s1" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = ("value = " + repr(relative) + "\n").encode()
        path.write_bytes(raw)
        pins[str(path)] = digest(raw)
    monkeypatch.setattr(boot, "_plan", lambda *_: {"source_pins": pins})
    monkeypatch.setattr(boot, "_source_set", lambda: set(pins))
    return pins


def test_launcher_and_bootstrap_use_the_same_fixed_allowlist(monkeypatch):
    import jev_decision3_candidate_http_owner as common
    import jev_decision3_cpu_experiment_worker as cpu

    monkeypatch.setattr(launcher, "cpu", cpu)
    monkeypatch.setattr(launcher, "common", common)
    assert boot._source_set() == launcher._source_set()


def test_held_bootstrap_rejects_without_reading_plan(monkeypatch):
    monkeypatch.setattr(boot, "SOURCE_ONLY", True)
    monkeypatch.setattr(boot, "_plan", lambda *_: pytest.fail("must not read"))
    with pytest.raises(ValueError, match="bootstrap release"):
        boot.prepare("unused", "unused", mode="launcher")


def test_bootstrap_verifies_all_selected_bytes_before_executing_any(sources, monkeypatch):
    pins = sources
    calls = []
    original = boot._read

    def read(path, expected, **kwargs):
        calls.append(("read", str(path)))
        return original(path, expected, **kwargs)

    def module(name, path, data, **_kwargs):
        assert sum(kind == "read" for kind, _ in calls) == len(pins)
        calls.append(("execute", name))
        if name == "jev_decision3_cuda_synthetic_launcher":
            return SimpleNamespace(_source_set=lambda: set(pins), _plan=lambda *_: None)
        return SimpleNamespace(SOURCE_ONLY=False)

    monkeypatch.setattr(boot, "_read", read)
    monkeypatch.setattr(boot, "_module", module)
    entry = boot.prepare("synthetic", "sha", mode="supervisor")
    assert entry.SOURCE_ONLY is False
    assert calls[0][0] == "read" and calls[-1] == ("execute", "_cuda_synthetic_entry")
    assert set(path for kind, path in calls if kind == "read") == set(pins)


def test_tampered_last_source_prevents_all_workspace_execution(sources, monkeypatch):
    path = boot.ROOT / "src/s1/evaluation/decision3.py"
    path.write_bytes(b"tampered")
    monkeypatch.setattr(boot, "_module", lambda *_args, **_kwargs: pytest.fail("must not execute"))
    with pytest.raises(ValueError, match="source bytes changed"):
        boot.prepare("synthetic", "sha", mode="launcher")


def test_release_gate_difference_cannot_hide_other_changes(sources, monkeypatch):
    path = boot._release("supervisor")
    path.write_bytes(b"SOURCE_ONLY = False\nextra = True\n")
    sources[str(path)] = digest(path.read_bytes())
    monkeypatch.setattr(boot, "_module", lambda *_args, **_kwargs: pytest.fail("must not execute"))
    with pytest.raises(ValueError, match="gate-only"):
        boot.prepare("synthetic", "sha", mode="supervisor")


def test_source_reader_rejects_symlinks_and_hardlinks_before_read(tmp_path):
    source = tmp_path / "plain.py"
    source.write_bytes(b"synthetic")
    alias = tmp_path / "alias.py"
    alias.symlink_to(source)
    with pytest.raises(ValueError, match="ordinary"):
        boot._read(alias, digest(b"synthetic"))
    hard = tmp_path / "hard.py"
    hard.hardlink_to(source)
    with pytest.raises(ValueError, match="bounded regular"):
        boot._read(source, digest(b"synthetic"))


def test_module_executes_verified_supplied_bytes_only(tmp_path, monkeypatch):
    name = "_synthetic_bootstrap_test"
    monkeypatch.setattr(boot, "sys", SimpleNamespace(modules={}))
    path = tmp_path / "source.py"
    path.write_bytes(b"value = 'unselected on disk'\n")
    module = boot._module(name, path, b"value = 'verified bytes'\n")
    assert module.value == "verified bytes" and boot.sys.modules[name] is module
    with pytest.raises(ValueError, match="fresh"):
        boot._module(name, path, b"value = 2\n")


def test_bad_plan_inventory_fails_before_any_source_reads(tmp_path, monkeypatch):
    import json

    base = tmp_path / "root"
    run = base / ("run-" + "a" * 32)
    run.mkdir(parents=True)
    path = run / "plan.json"
    plan = {
        "schema": "decision3-cuda-synthetic-plan-v1",
        "source_pins": {str(tmp_path / "unexpected.py"): "a" * 64},
    }
    raw = json.dumps(plan).encode()
    path.write_bytes(raw)
    monkeypatch.setattr(boot, "BASE", base)
    monkeypatch.setattr(boot, "_source_set", lambda: {str(tmp_path / "approved.py")})
    with pytest.raises(ValueError, match="source inventory"):
        boot._plan(str(path), digest(raw))
    assert not (tmp_path / "unexpected.py").exists()


def test_full_synthetic_bootstrap_loads_verified_entry_and_injects_dependencies(
    sources, monkeypatch
):
    expected = set(sources)
    held = boot.SCRIPTS / boot.FILES["launcher"]
    released = boot._release("launcher")
    raw = (
        "SOURCE_ONLY = True\ndef _source_set(): return "
        + repr(expected)
        + "\ndef _plan(*args): return None\n"
    ).encode()
    held.write_bytes(raw)
    release_raw = raw.replace(b"SOURCE_ONLY = True", b"SOURCE_ONLY = False")
    released.write_bytes(release_raw)
    sources[str(held)], sources[str(released)] = digest(raw), digest(release_raw)
    entry = boot.prepare("synthetic", "sha", mode="launcher")
    assert entry.SOURCE_ONLY is False
    assert entry.common is boot.sys.modules["jev_decision3_candidate_http_owner"]
    assert entry.child is boot.sys.modules["jev_decision3_candidate_browser_child"]
    assert entry.cpu is boot.sys.modules["jev_decision3_cpu_experiment_worker"]
    assert boot.sys.modules["s1.evaluation.decision3"].value == "evaluation/decision3.py"
