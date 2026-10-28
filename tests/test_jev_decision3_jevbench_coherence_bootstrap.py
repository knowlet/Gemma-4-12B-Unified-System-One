"""Synthetic bootstrap files and plans only; no benchmark, model or GPU."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jev_decision3_jevbench_coherence_bootstrap as boot
import pytest


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    base = root / "experiments"
    run = base / ("run-" + "a" * 32)
    run.mkdir(parents=True)
    monkeypatch.setattr(boot, "ROOT", root)
    monkeypatch.setattr(boot, "SCRIPTS", scripts)
    monkeypatch.setattr(boot, "BASE", base)
    monkeypatch.setattr(boot, "SOURCE_ONLY", False)
    monkeypatch.setattr(boot, "sys", SimpleNamespace(modules={}))
    sources = {}
    public = {}
    for relative in boot.PUBLIC:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = ("synthetic = " + repr(relative) + "\n").encode()
        if path.name == boot.OWNER:
            raw = b"\nSOURCE_ONLY = True\ndef _validate_plan(plan): pass\n"
        elif path.name == boot.SUPERVISOR:
            raw = b"\nSOURCE_ONLY = True\ndef _unit_args(p, path, sha, release): return ['systemd-run', '--fixed', p['python'], '-B', '-s', str(release), '--plan', str(path), '--plan-sha256', sha]\n"
        elif path.name in ("jev_decision3_jevbench_generic.py", "jev_gpu_forward_lease.py"):
            raw = b"\nSOURCE_ONLY = True\n# synthetic\n"
        path.write_bytes(raw)
        sources[str(path)] = sha(raw)
        public[relative] = sha(raw)
    for held, released in (
        ("jev_decision3_jevbench_generic.py", "jev_decision3_jevbench_generic_release_20261027.py"),
        ("jev_gpu_forward_lease.py", "jev_gpu_forward_lease_release_20261027.py"),
    ):
        target = scripts / released
        target.write_bytes(
            (scripts / held).read_bytes().replace(b"SOURCE_ONLY = True", b"SOURCE_ONLY = False")
        )
        sources[str(target)] = public["scripts/" + released] = sha(target.read_bytes())
    for held, released in (
        (boot.BOOTSTRAP, boot.BOOT_RELEASE),
        (boot.OWNER, boot.OWNER_RELEASE),
        (boot.SUPERVISOR, boot.SUPERVISOR_RELEASE),
    ):
        source, target = scripts / held, scripts / released
        if not source.exists():
            source.write_bytes(b"\nSOURCE_ONLY = True\n# synthetic\n")
        target.write_bytes(
            source.read_bytes().replace(b"SOURCE_ONLY = True", b"SOURCE_ONLY = False")
        )
        sources[str(source)], sources[str(target)] = (
            sha(source.read_bytes()),
            sha(target.read_bytes()),
        )
    runtime = {}
    for index in range(8):
        path = root / "runtime" / str(index)
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(("runtime-" + str(index)).encode())
        sources[str(path)] = runtime[str(path)] = sha(path.read_bytes())
    monkeypatch.setattr(boot, "PUBLIC", public)
    monkeypatch.setattr(boot, "runtime_pins", lambda: runtime)
    manifest = (
        root
        / "artifacts/jev-local-hf/hub/models--vllm-sr--d3-edge/snapshots"
        / boot.REVISION
        / "MODEL_MANIFEST.json"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b"synthetic manifest")
    monkeypatch.setattr(boot, "MANIFEST_SHA", sha(manifest.read_bytes()))
    plan = {
        "schema": "decision3-jevbench-coherence-owner-plan-v1",
        "unit": "jev-coherence-" + "a" * 32 + ".service",
        "output_dir": str(run / "worker"),
        "source_pins": sources,
        "python": "/home/albert_hsieh/ComfyUI/.venv/bin/python",
        "cache_root": str(root / "artifacts/jev-local-hf/hub"),
        "modules_cache": str(
            root
            / "artifacts/jev-runtime/decision3-link-navigation-diagnostic-20261027-01/staged-modules"
        ),
        "overlay": str(root / "artifacts/jev-runtime/overlay"),
        "tasks_root": "/tmp/jev-coherence-mini-controlled-snapshot",
        "generic_release": str(scripts / "jev_decision3_jevbench_generic_release_20261027.py"),
        "deltas": boot.DELTAS,
        "transport": {
            "lease_release": str(scripts / "jev_gpu_forward_lease_release_20261027.py"),
            "lock": str(run / "lock"),
            "journal": str(run / "journal"),
        },
    }
    path = run / "plan.json"
    raw = json.dumps(plan).encode()
    path.write_bytes(raw)
    return path, plan, sha(raw)


def test_gate_rejects_before_plan_read(monkeypatch):
    monkeypatch.setattr(boot, "SOURCE_ONLY", True)
    monkeypatch.setattr(boot, "_plan", lambda *_: pytest.fail("unexpected read"))
    with pytest.raises(ValueError, match="bootstrap release"):
        boot.prepare("unused", "unused", mode="owner")


def test_main_rejects_stdin_before_plan_or_workspace_read(monkeypatch):
    monkeypatch.setattr(boot, "SOURCE_ONLY", False)
    monkeypatch.setattr(boot, "__file__", "<stdin>")
    monkeypatch.setattr(boot, "sys", SimpleNamespace(flags=SimpleNamespace(isolated=1)))
    monkeypatch.setattr(boot, "prepare", lambda *_args, **_kwargs: pytest.fail("unexpected read"))
    with pytest.raises(ValueError, match="isolated reviewed entry"):
        boot.main()


@pytest.mark.parametrize("mode", ["owner", "supervisor"])
def test_all_selected_sources_are_checked_before_any_exec(setup, monkeypatch, mode):
    path, _plan, digest = setup
    read = boot._read
    seen = []
    original = boot._module

    def tracked_read(target, *args, **kwargs):
        seen.append(str(target))
        return read(target, *args, **kwargs)

    def checked_module(name, target, data, **kwargs):
        assert boot.source_set().issubset(seen)
        return original(name, target, data, **kwargs)

    monkeypatch.setattr(boot, "_read", tracked_read)
    monkeypatch.setattr(boot, "_module", checked_module)
    entry = boot.prepare(str(path), digest, mode=mode)
    assert entry.SOURCE_ONLY is False
    if mode == "supervisor":
        args = entry._unit_args({"python": "/synthetic/python"}, path, digest, scripts_release())
        assert args[:2] == ["systemd-run", "--fixed"]
        assert args[2:6] == ["/synthetic/python", "-I", "-B", "-s"]
        assert args[6] == str(boot.SCRIPTS / boot.BOOT_RELEASE)
        assert args[7:9] == ["--mode", "owner"]


def scripts_release():
    return boot.SCRIPTS / boot.OWNER_RELEASE


def test_unknown_pin_refused_before_pin_content_read(setup, monkeypatch):
    path, plan, _digest = setup
    plan["source_pins"][str(path.parent / "unselected.py")] = "b" * 64
    raw = json.dumps(plan).encode()
    path.write_bytes(raw)
    original = boot._read

    def read(target, *args, **kwargs):
        assert target == path
        return original(target, *args, **kwargs)

    monkeypatch.setattr(boot, "_read", read)
    with pytest.raises(ValueError, match="allowlist"):
        boot.prepare(str(path), sha(raw), mode="owner")


@pytest.mark.parametrize("fault", ["tasks", "delta", "lease"])
def test_unapproved_input_path_is_rejected_before_source_read(setup, monkeypatch, fault):
    path, plan, _digest = setup
    if fault == "tasks":
        plan["tasks_root"] = "/synthetic/unapproved"
    elif fault == "delta":
        plan["deltas"] = {
            **plan["deltas"],
            "operation": {**plan["deltas"]["operation"], "path": "/synthetic/unapproved"},
        }
    else:
        plan["transport"]["journal"] = "/synthetic/unapproved"
    raw = json.dumps(plan).encode()
    path.write_bytes(raw)
    original = boot._read

    def read(target, *args, **kwargs):
        assert target == path
        return original(target, *args, **kwargs)

    monkeypatch.setattr(boot, "_read", read)
    with pytest.raises(ValueError, match="fixed coherence MINI|fixed private"):
        boot.prepare(str(path), sha(raw), mode="owner")


def test_bad_gate_or_tampered_source_prevents_module_execution(setup, monkeypatch):
    path, plan, digest = setup
    target = boot.SCRIPTS / boot.OWNER_RELEASE
    target.write_bytes(b"SOURCE_ONLY = False\nmodified = True\n")
    plan["source_pins"][str(target)] = sha(target.read_bytes())
    raw = json.dumps(plan).encode()
    path.write_bytes(raw)
    monkeypatch.setattr(boot, "_module", lambda *_args, **_kwargs: pytest.fail("unexpected exec"))
    with pytest.raises(ValueError, match="gate-only"):
        boot.prepare(str(path), sha(raw), mode="owner")


def test_launch_wrapper_rejects_modified_contract():
    def bad(*_args):
        return ["systemd-run", "--unreviewed"]

    with pytest.raises(ValueError, match="launch contract"):
        boot._unit_args_wrapper(bad, {"python": "/synthetic/python"}, "/plan", "a" * 64, "/release")


def test_unknown_plan_path_and_duplicate_keys_fail_closed(setup):
    path, _plan, digest = setup
    with pytest.raises(ValueError, match="plan identity"):
        boot._plan(str(path.parent / "other.json"), digest)
    with pytest.raises(ValueError, match="duplicate"):
        boot._decode(b'{"x":1,"x":2}')


def test_fixed_inventory_never_includes_browser_heldout_paths():
    names = boot.source_set()
    assert len(names) == 32
    assert all(
        "heldout" not in x and "root-done-feedback" not in x and "/cases" not in x for x in names
    )


def test_mini_loader_only_reads_exact_synthetic_sdk_and_six_assets(tmp_path, monkeypatch):
    import jevbench_public_request_inventory as inventory

    expected_assets = {
        "src/jevbench/suites/jevbench-mini/manifest.json",
        "src/jevbench/suites/jevbench-mini/tests.jsonl.gz",
        "src/jevbench/suites/jevbench-240/manifest.json",
        "src/jevbench/suites/jevbench-240/cases.jsonl",
        "src/jevbench/suites/jevbench-240/rewrites.jsonl",
        "src/jevbench/suites/jevbench-240/tests.jsonl.gz",
    }
    assert inventory._MINI_ASSETS == expected_assets
    selected = inventory._MINI_SDK | inventory._MINI_ASSETS
    assert all("/examples/" not in name for name in selected)
    pins = {}
    for name in selected:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = ("synthetic " + name).encode()
        path.write_bytes(raw)
        pins[name] = sha(raw)
    excluded = tmp_path / "src/jevbench/examples/not_selected.jsonl"
    excluded.parent.mkdir(parents=True)
    excluded.write_bytes(b"synthetic unselected")
    pins["src/jevbench/examples/not_selected.jsonl"] = sha(excluded.read_bytes())
    original = Path.read_bytes

    def guarded_read(path):
        assert path != excluded, "unselected example must not be read"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    sources = inventory._check_coherence_files(
        tmp_path, {"files": pins}, suite_name="jevbench-mini"
    )
    assert set(sources) == inventory._MINI_SDK


def test_mini_loader_rejects_missing_asset_before_payload_read(tmp_path, monkeypatch):
    import jevbench_public_request_inventory as inventory

    names = inventory._MINI_SDK | inventory._MINI_ASSETS
    missing = next(iter(inventory._MINI_ASSETS))
    monkeypatch.setattr(Path, "read_bytes", lambda *_: pytest.fail("unexpected payload read"))
    with pytest.raises(ValueError, match="missing required MINI"):
        inventory._check_coherence_files(
            tmp_path,
            {"files": {name: "a" * 64 for name in names if name != missing}},
            suite_name="jevbench-mini",
        )
