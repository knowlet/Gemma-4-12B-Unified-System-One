"""Synthetic temporary sources and benchmark bytes only; never a real plan or GPU."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jev_decision3_jevbench_heaven_bootstrap as real_boot
import jev_decision3_jevbench_heaven_prepare as prep
import pytest


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def npy():
    header = b"{'descr': '<f4', 'fortran_order': False, 'shape': (255, 1024), }\n"
    body = b"\0" * (255 * 1024 * 4)
    return b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header + body, body


@pytest.fixture
def fake(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    base = root / "artifacts/jev-runtime/decision3-heaven-v2"
    base.parent.mkdir(parents=True)
    monkeypatch.setattr(prep, "ROOT", root)
    monkeypatch.setattr(prep, "SCRIPTS", scripts)
    monkeypatch.setattr(prep, "BASE", base)
    monkeypatch.setattr(prep, "SOURCE_ONLY", False)
    monkeypatch.setattr(prep, "_gpu_uuid", lambda: "GPU-12345678-1234-1234-1234-123456789abc")
    monkeypatch.setattr(prep.secrets, "token_hex", lambda _: "a" * 32)
    boot = SimpleNamespace(
        ROOT=root, SCRIPTS=scripts, BASE=base, SOURCE_ONLY=True, REVISION="synthetic"
    )
    for name in (
        "BOOTSTRAP",
        "BOOT_RELEASE",
        "OWNER",
        "OWNER_RELEASE",
        "SUPERVISOR",
        "SUPERVISOR_RELEASE",
    ):
        setattr(boot, name, getattr(real_boot, name))
    public = {}
    for relative in real_boot.PUBLIC:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = ("# synthetic " + relative + "\n").encode()
        if path.name in (
            boot.OWNER,
            boot.SUPERVISOR,
            "jev_decision3_jevbench_generic.py",
            "jev_gpu_forward_lease.py",
        ):
            raw = b"SOURCE_ONLY = True\n# synthetic\n"
        path.write_bytes(raw)
        public[relative] = sha(raw)
    for held, release in (
        ("jev_decision3_jevbench_generic.py", "jev_decision3_jevbench_generic_release_20261027.py"),
        ("jev_gpu_forward_lease.py", "jev_gpu_forward_lease_release_20261027.py"),
    ):
        path = scripts / release
        path.write_bytes(
            (scripts / held).read_bytes().replace(b"SOURCE_ONLY = True", b"SOURCE_ONLY = False")
        )
        public["scripts/" + release] = sha(path.read_bytes())
    boot.PUBLIC = public
    runtime = {}
    for index in range(8):
        path = root / "runtime" / str(index)
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(("runtime-" + str(index)).encode())
        runtime[str(path)] = sha(path.read_bytes())
    boot.runtime_pins = lambda: runtime
    bootstrap = scripts / boot.BOOTSTRAP
    bootstrap.write_bytes(b"SOURCE_ONLY = True\n# synthetic bootstrap\n")
    monkeypatch.setattr(prep, "BOOT_SHA", sha(bootstrap.read_bytes()))
    payload, body = npy()
    delta_dir = root / "deltas"
    delta_dir.mkdir()
    boot.DELTAS = {}
    for head in ("operation", "click_target"):
        path = delta_dir / (head + ".npy")
        path.write_bytes(payload)
        boot.DELTAS[head] = {
            "path": str(path),
            "file_sha256": sha(payload),
            "fp32_sha256": sha(body),
        }
    snapshot = (
        root / "artifacts/jev-local-hf/hub/models--vllm-sr--d3-edge/snapshots" / boot.REVISION
    )
    snapshot.mkdir(parents=True)
    (snapshot / "MODEL_MANIFEST.json").write_bytes(b"synthetic manifest")
    boot.MANIFEST_SHA = sha(b"synthetic manifest")
    boot.source_set = lambda: (
        {str(root / x) for x in public}
        | set(runtime)
        | {
            str(scripts / x)
            for x in (
                boot.BOOTSTRAP,
                boot.BOOT_RELEASE,
                boot.OWNER_RELEASE,
                boot.SUPERVISOR_RELEASE,
            )
        }
    )
    assert len(boot.source_set()) == 31
    monkeypatch.setattr(prep, "_bootstrap", lambda: (boot, bootstrap.read_bytes()))
    monkeypatch.setattr(prep, "_official", lambda _config: None)
    return boot, root


def test_held_gate_refuses_before_filesystem(monkeypatch):
    monkeypatch.setattr(prep, "SOURCE_ONLY", True)
    monkeypatch.setattr(prep, "_bootstrap", lambda: pytest.fail("unexpected read"))
    with pytest.raises(ValueError, match="source-only"):
        prep.prepare()


def test_fresh_private_plan_and_gate_only_v2_releases(fake):
    boot, _root = fake
    result = prep.prepare()
    plan_path = Path(result["plan"])
    raw = plan_path.read_bytes()
    plan = json.loads(raw)
    assert result["sha256"] == sha(raw) and result["pin_count"] == 31
    assert set(plan["source_pins"]) == boot.source_set()
    assert (
        plan["deltas"] == boot.DELTAS
        and plan["wall_seconds"] == 600
        and plan["request_seconds"] == 60
    )
    assert plan["unit"] == "jev-heaven-" + "a" * 32 + ".service"
    assert plan_path.stat().st_mode & 0o077 == 0 and plan_path.parent.stat().st_mode & 0o077 == 0
    assert Path(plan["output_dir"]).is_dir() and not any(Path(plan["output_dir"]).iterdir())
    assert not Path(result["report"]).exists()
    for name in ("lock", "journal"):
        path = plan_path.parent / name
        st = path.stat()
        assert plan["transport"][name] == str(path)
        assert plan["transport"]["stat"][name] == {"device": st.st_dev, "inode": st.st_ino}
    for held, released in (
        (boot.BOOTSTRAP, boot.BOOT_RELEASE),
        (boot.OWNER, boot.OWNER_RELEASE),
        (boot.SUPERVISOR, boot.SUPERVISOR_RELEASE),
    ):
        original, release = prep.SCRIPTS / held, prep.SCRIPTS / released
        assert release.read_bytes() == original.read_bytes().replace(
            prep.MARKER, b"SOURCE_ONLY = False", 1
        )
        assert plan["source_pins"][str(release)] == sha(release.read_bytes())


def test_synthetic_plan_is_accepted_by_actual_bootstrap_validator(fake, monkeypatch):
    boot, root = fake
    result = prep.prepare()
    for name, value in (
        ("ROOT", root),
        ("SCRIPTS", prep.SCRIPTS),
        ("BASE", prep.BASE),
        ("PUBLIC", boot.PUBLIC),
        ("DELTAS", boot.DELTAS),
        ("REVISION", boot.REVISION),
        ("MANIFEST_SHA", boot.MANIFEST_SHA),
    ):
        monkeypatch.setattr(real_boot, name, value)
    monkeypatch.setattr(real_boot, "runtime_pins", boot.runtime_pins)
    plan = real_boot._plan(result["plan"], result["sha256"])
    verified = real_boot._verified_bytes(plan)
    assert set(verified) == set(plan["source_pins"]) == boot.source_set()


def test_existing_release_refuses_without_overwrite(fake):
    boot, _root = fake
    target = prep.SCRIPTS / boot.OWNER_RELEASE
    target.write_bytes(b"existing user data")
    with pytest.raises(FileExistsError):
        prep.prepare()
    assert target.read_bytes() == b"existing user data" and not prep.BASE.exists()


def test_changed_delta_fails_before_publication(fake):
    boot, _root = fake
    Path(boot.DELTAS["operation"]["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="source changed"):
        prep.prepare()
    assert not prep.BASE.exists() and not (prep.SCRIPTS / boot.BOOT_RELEASE).exists()


def test_changed_fixed_source_fails_before_publication(fake):
    boot, _root = fake
    (prep.SCRIPTS / boot.OWNER).write_bytes(b"changed")
    with pytest.raises(ValueError, match="source changed"):
        prep.prepare()
    assert not prep.BASE.exists() and not (prep.SCRIPTS / boot.BOOT_RELEASE).exists()


def test_competing_release_preserved_and_owned_files_rolled_back(fake, monkeypatch):
    boot, _root = fake
    first, contested = prep.SCRIPTS / boot.BOOT_RELEASE, prep.SCRIPTS / boot.OWNER_RELEASE
    original_link = prep.os.link

    def link(src, dst, **kwargs):
        if Path(dst) == contested:
            first.unlink()
            first.write_bytes(b"replacement")
            contested.write_bytes(b"other")
            raise FileExistsError("synthetic competitor")
        return original_link(src, dst, **kwargs)

    monkeypatch.setattr(prep.os, "link", link)
    with pytest.raises(FileExistsError, match="competitor"):
        prep.prepare()
    assert first.read_bytes() == b"replacement" and contested.read_bytes() == b"other"
    assert not prep.BASE.exists()
    assert not any(path.name.endswith(".tmp") for path in prep.SCRIPTS.iterdir())


def test_npy_requires_exact_finite_complete_bytes():
    raw, body = npy()
    prep._delta(raw, sha(body))
    with pytest.raises(ValueError, match="FP32"):
        prep._delta(raw[:-1], sha(body))
    nonfinite = bytearray(raw)
    nonfinite[-4:] = bytes.fromhex("0000807f")
    with pytest.raises(ValueError, match="FP32"):
        prep._delta(bytes(nonfinite), sha(bytes(nonfinite[-len(body) :])))


def test_official_inventory_exact_and_checked_before_return(tmp_path, monkeypatch):
    root = tmp_path / "official"
    root.mkdir()
    names = [
        "datasets/public/original.jsonl",
        "datasets/public/easy.jsonl",
        "datasets/public/hard.jsonl",
        "jevbench/__init__.py",
        "jevbench/tasks.py",
        "jevbench/scoring.py",
        "jevbench/metrics.py",
        "jevbench/summarize.py",
        "jevbench/adapters/base.py",
    ]
    pins = {}
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("synthetic " + name).encode())
        pins[name] = sha(path.read_bytes())
    config = {
        "expected_tasks": 231,
        "source_revision": "bb05a335bc809e61b20c0f745d25499a82b326fc",
        "task_files": [{"path": name, "sha256": pins[name]} for name in names[:3]],
        "source_files": {name: pins[name] for name in names[3:]},
    }
    monkeypatch.setattr(prep, "OFFICIAL", root)
    monkeypatch.setattr(prep, "OFFICIAL_FILES", pins)
    prep._official(json.dumps(config).encode())
    changed = copy.deepcopy(config)
    changed["source_files"][names[-1]] = "0" * 64
    with pytest.raises(ValueError, match="official inventory"):
        prep._official(json.dumps(changed).encode())
