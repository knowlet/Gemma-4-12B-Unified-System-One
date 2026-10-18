"""Pure boundary checks; no Modal/ML imports, network, worker or GPU execution."""

import ast
import hashlib
import json
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Literal, get_args, get_type_hints

import pytest

APP = Path(__file__).resolve().parents[1] / "apps/modal/public_action_fit.py"
GIB = 1024**3
MEM_TOTAL = 130595860480
TRAIN_REQUIRED = 106438961088


def extracted_helpers(tmp_path):
    """Compile only reviewed function definitions, never the application module."""
    names = {
        "require",
        "training_steps_choice",
        "encode",
        "parse",
        "ordinary",
        "capture_mount",
        "mount_root",
        "remote_cache",
        "named_input",
        "relative",
        "read",
        "pin",
        "create",
        "stream_copy",
        "stage_weights",
        "complete_cgroup",
        "cgroup_memory",
        "host",
        "clear_baked_processor_metadata",
        "admit",
        "group_alive",
        "stop_owned",
        "tree_inventory",
        "copy_output",
        "owned",
    }
    body = [
        node
        for node in ast.parse(APP.read_bytes()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in body} == names
    assert all(not node.decorator_list for node in body)
    # AST-only selection excludes imports, image definitions, volume creation,
    # entrypoint dispatch and every function that imports the fitter/framework.
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        for definition in body
        for node in ast.walk(definition)
    )
    mounts = {}
    for name in ("export", "weights", "input"):
        root = tmp_path / ("provider-" + name)
        root.mkdir()
        alias = tmp_path / name
        alias.symlink_to(root, target_is_directory=True)
        mounts["/" + name] = alias
    export_root = mounts["/export"].resolve()

    def task_path(value, *parts):
        path = Path(value, *parts)
        return mounts.get(str(path), path)

    namespace = {
        "hashlib": hashlib,
        "json": json,
        "re": re,
        "stat": stat,
        "signal": signal,
        "Path": task_path,
        "PurePosixPath": PurePosixPath,
        "GIB": GIB,
        "MAX_FILES": 10000,
        "MAX_LOG": 16 * 1024**2,
        "BASE": {"size_bytes": 23919549408},
        "DECLARED_MOUNTS": {"/weights", "/input", "/export"},
        "ROOT": tmp_path / "source",
        "RUNTIME": tmp_path / "source/artifacts/jev-runtime",
        "SCRIPT": tmp_path / "source/apps/modal/public_action_fit.py",
        "OVERLAY": tmp_path / "source/overlay",
        "HUB": tmp_path / "cache/hub",
        "INTERPRETER": "/synthetic-only/python",
        "time": SimpleNamespace(monotonic=lambda: 100.0),
        "os": SimpleNamespace(
            fsync=os.fsync, fstat=os.fstat, scandir=os.scandir, replace=os.replace, environ={}
        ),
        "tempfile": SimpleNamespace(TemporaryFile=lambda: tempfile.TemporaryFile(dir=tmp_path)),
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(APP), "exec"), namespace)
    return namespace, export_root


@pytest.mark.parametrize(
    "fault",
    [None, "unexpected_weight", "unexpected_empty_directory", "wrong_pin", "missing", "symlink"],
)
def test_image_mount_cleanup_requires_exact_metadata_before_removing_any_file(tmp_path, fault):
    namespace, _export = extracted_helpers(tmp_path)
    hub = tmp_path / "cache/hub"
    cache = hub / "models--public/snapshots/fixed"
    cache.mkdir(parents=True)
    metadata = {}
    for name, blob in (("config.json", b"{}"), ("tokenizer.json", b"public tokenizer")):
        (cache / name).write_bytes(blob)
        metadata[name] = {"sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)}
    namespace.update(HUB=hub, CACHE=cache, BASE_METADATA=metadata)
    if fault == "unexpected_weight":
        (cache / "model.safetensors").write_bytes(b"must survive")
    elif fault == "unexpected_empty_directory":
        (hub / "unrelated").mkdir()
    elif fault == "wrong_pin":
        (cache / "tokenizer.json").write_bytes(b"changed")
    elif fault == "missing":
        (cache / "tokenizer.json").unlink()
    elif fault == "symlink":
        (cache / "tokenizer.json").unlink()
        (cache / "tokenizer.json").symlink_to(cache / "config.json")
    before = {str(p.relative_to(hub)): p.read_bytes() for p in hub.rglob("*") if p.is_file()}
    before_dirs = {str(p.relative_to(hub)) for p in hub.rglob("*") if p.is_dir()}
    if fault is None:
        namespace["clear_baked_processor_metadata"]()
        assert hub.is_dir() and not list(hub.iterdir())
    else:
        with pytest.raises(ValueError):
            namespace["clear_baked_processor_metadata"]()
        after = {str(p.relative_to(hub)): p.read_bytes() for p in hub.rglob("*") if p.is_file()}
        assert before == after
        assert before_dirs == {str(p.relative_to(hub)) for p in hub.rglob("*") if p.is_dir()}


@pytest.mark.parametrize(
    "mode", ["unlimited", "ancestor_limit", "child_limit", "partial", "missing", "root_partial"]
)
def test_host_observes_membership_and_all_controlling_ancestors(tmp_path, mode):
    namespace, _export = extracted_helpers(tmp_path)
    mount = tmp_path / "cgroup"
    parent, leaf = mount / "user.slice", mount / "user.slice/session.scope"
    leaf.mkdir(parents=True)
    for directory, current in ((parent, 20 * GIB), (leaf, GIB)):
        (directory / "memory.max").write_text("max")
        (directory / "memory.current").write_text(str(current))
    if mode == "ancestor_limit":
        (parent / "memory.max").write_text(str(110 * GIB))
    elif mode == "child_limit":
        (parent / "memory.max").write_text(str(160 * GIB))
        (leaf / "memory.max").write_text(str(100 * GIB))
    elif mode == "partial":
        (parent / "memory.current").unlink()
    elif mode == "root_partial":
        (mount / "memory.current").write_text(str(20 * GIB))
    member = "/missing.scope" if mode == "missing" else "/user.slice/session.scope"
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "sys/kernel/random").mkdir(parents=True)
    (proc / "meminfo").write_text(
        f"MemTotal: {160 * GIB // 1024} kB\nMemAvailable: {150 * GIB // 1024} kB\n"
    )
    (proc / "sys/kernel/random/boot_id").write_text("fixture-boot")
    (proc / "self/cgroup").write_text("0::" + member + "\n")
    (proc / "self/mountinfo").write_text(f"39 29 0:33 / {mount} rw - cgroup2 cgroup2 rw\n")

    def fake_path(value, *parts):
        path = Path(value, *parts)
        return proc / path.relative_to("/proc") if path.is_relative_to("/proc") else path

    namespace["Path"] = fake_path
    snapshot = namespace["host"]()
    assert snapshot["cgroup_membership"] == "0::" + member + "\n"
    if mode in {"partial", "missing", "root_partial"}:
        assert snapshot["cgroup"] == {}
        assert any("error" in row for row in snapshot["cgroup_ancestors"])
    else:
        assert [row["path"] for row in snapshot["cgroup_ancestors"]] == [str(leaf), str(parent)]
        expected = {"memory.max": None, "memory.current": GIB}
        if mode == "ancestor_limit":
            expected = {"memory.max": 110 * GIB, "memory.current": 20 * GIB}
        elif mode == "child_limit":
            expected = {"memory.max": 100 * GIB, "memory.current": GIB}
        assert snapshot["cgroup"] == expected
    journal = tmp_path / "admission.jsonl"
    if mode == "unlimited":
        assert namespace["admit"](journal, "train", "fixture-boot")["admitted"]
    else:
        with pytest.raises(ValueError, match="admission refused"):
            namespace["admit"](journal, "train", "fixture-boot")
    saved = json.loads(journal.read_bytes())
    assert saved["reserve_bytes"] == 32 * GIB
    assert saved["required_bytes"] == 32 * GIB + 2 * 23919549408 + 30 * GIB + 256 * 1024**2
    assert saved["swap_or_resident_credit_bytes"] == 0


@pytest.mark.parametrize("member", ["/tenant/job", "/job", "/"])
def test_cgroup_namespace_mount_root_is_a_controlling_group(tmp_path, member):
    namespace, _export = extracted_helpers(tmp_path)
    mount = tmp_path / "mounted-subtree"
    current = mount if member == "/" else mount / "job"
    current.mkdir(parents=True)
    for directory in {mount, current}:
        (directory / "memory.max").write_text(str(110 * GIB))
        (directory / "memory.current").write_text(str(20 * GIB))
    observed = namespace["cgroup_memory"](
        "0::" + member + "\n", f"39 29 0:33 /tenant {mount} rw - cgroup2 cgroup2 rw\n"
    )
    assert observed[-1]["path"] == str(mount)
    assert all(row["memory"]["memory.max"] == 110 * GIB for row in observed)
    (mount / "memory.current").unlink()
    incomplete = namespace["cgroup_memory"](
        "0::" + member + "\n", f"39 29 0:33 /tenant {mount} rw - cgroup2 cgroup2 rw\n"
    )
    assert incomplete[-1]["error"] == "FileNotFoundError"


@pytest.mark.parametrize(
    "mode",
    [
        "namespace_current",
        "finite_ancestor",
        "partial",
        "negative_usage",
        "nonmemory",
        "unlimited_integer",
    ],
)
def test_v1_memory_namespace_ancestors_and_incomplete_observations(tmp_path, mode):
    namespace, _ = extracted_helpers(tmp_path)
    mount = tmp_path / "v1-memory"
    mount.mkdir()
    leaf = mount / "job" if mode == "finite_ancestor" else mount
    leaf.mkdir(exist_ok=True)
    limit = 9223372036854771712 if mode == "unlimited_integer" else 512 * GIB
    for directory in {mount, leaf}:
        (directory / "memory.limit_in_bytes").write_text(str(limit))
        (directory / "memory.usage_in_bytes").write_text(str(GIB))
    if mode == "finite_ancestor":
        (mount / "memory.usage_in_bytes").write_text(str(350 * GIB))
    elif mode == "partial":
        (mount / "memory.usage_in_bytes").unlink()
    elif mode == "negative_usage":
        (mount / "memory.usage_in_bytes").write_text("-1")
    member = "/ta-real/job" if mode == "finite_ancestor" else "/ta-real"
    controllers = "cpuacct" if mode == "nonmemory" else "memory"
    memberships = "7:pids:" + member + "\n6:" + controllers + ":" + member + "\n"
    mountinfo = f"31 25 0:15 /ta-real {mount} rw - cgroup none rw,{controllers}\n"
    cpu = tmp_path / "v1-cpu"
    cpu.mkdir()
    mountinfo += f"32 25 0:16 /ta-real {cpu} rw - cgroup none rw,cpu\n"
    memberships += "1:cpu:" + member + "\n"
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "sys/kernel/random").mkdir(parents=True)
    (proc / "meminfo").write_text("MemTotal: 831668580 kB\nMemAvailable: 831599080 kB\n")
    (proc / "sys/kernel/random/boot_id").write_text("v1-boot")
    (proc / "self/cgroup").write_text(memberships)
    (proc / "self/mountinfo").write_text(mountinfo)

    def fake_path(value, *parts):
        path = Path(value, *parts)
        return proc / path.relative_to("/proc") if path.is_relative_to("/proc") else path

    namespace["Path"] = fake_path
    snapshot = namespace["host"]()
    observed = snapshot["cgroup_ancestors"]
    if mode == "nonmemory":
        assert observed == [] and snapshot["cgroup"] == {}
    else:
        assert all(row["version"] == 1 and row["path"] != str(cpu) for row in observed)
        assert observed[-1]["path"] == str(mount)
        assert observed[-1]["paths"] == {
            "memory.max": str(mount / "memory.limit_in_bytes"),
            "memory.current": str(mount / "memory.usage_in_bytes"),
        }
        assert observed[-1]["raw"]["memory.limit_in_bytes"] == str(limit)
        if mode in {"partial", "negative_usage"}:
            assert snapshot["cgroup"] == {}
        else:
            assert snapshot["cgroup"] == {
                "memory.max": limit,
                "memory.current": (350 if mode == "finite_ancestor" else 1) * GIB,
            }
    journal = tmp_path / "v1-gate.jsonl"
    if mode in {"namespace_current", "unlimited_integer"}:
        namespace["admit"](journal, "train", "v1-boot")
    else:
        with pytest.raises(ValueError, match="admission refused"):
            namespace["admit"](journal, "train", "v1-boot")
    saved = json.loads(journal.read_bytes())
    assert saved["reserve_bytes"] == 170325725184
    assert saved["required_bytes"] == 170325725184 + 2 * 23919549408 + 30 * GIB + 256 * 1024**2
    assert saved["swap_or_resident_credit_bytes"] == 0
    if mode == "finite_ancestor":
        assert [row["path"] for row in observed] == [str(leaf), str(mount)]
    if mode == "unlimited_integer":
        assert type(saved["observation"]["cgroup"]["memory.max"]) is int
        (proc / "meminfo").write_text("MemTotal: 831668580 kB\nMemAvailable: 1000 kB\n")
        with pytest.raises(ValueError, match="admission refused"):
            namespace["admit"](tmp_path / "host-low.jsonl", "cpu", "v1-boot")


def test_modal_ram_request_increases_without_changing_cpu_gpu_disk_or_recipe():
    tree = ast.parse(APP.read_bytes())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in ("download", "fit"):
        decorator = functions[name].decorator_list[0]
        options = {keyword.arg: keyword.value for keyword in decorator.keywords}
        assert ast.literal_eval(options["memory"]) == 344064
        assert ast.literal_eval(options["cpu"]) == 2
        assert ast.literal_eval(options["ephemeral_disk"]) == 524288
        if name == "fit":
            assert ast.literal_eval(options["gpu"]) == "H200"
    calls = [
        node
        for node in ast.walk(functions["fit_child"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "fit_public_actions"
    ]
    assert len(calls) == 1
    assert {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in calls[0].keywords
        if keyword.arg in {"learning_rate", "seed"}
    } == {
        "learning_rate": 5e-5,
        "seed": 42,
    }
    selected = next(keyword.value for keyword in calls[0].keywords if keyword.arg == "steps")
    assert isinstance(selected, ast.Name) and selected.id == "steps"


@pytest.mark.parametrize(
    "group,accepted",
    [
        ({}, False),
        ({"memory.max": None}, False),
        ({"memory.max": None, "memory.current": GIB}, True),
        ({"memory.max": 160 * GIB, "memory.current": 10 * GIB}, True),
        ({"memory.max": 110 * GIB, "memory.current": 20 * GIB}, False),
    ],
)
def test_admission_distinguishes_unknown_cgroup_from_observed_unlimited_and_real_headroom(
    tmp_path, group, accepted
):
    namespace, _export = extracted_helpers(tmp_path)
    namespace["host"] = lambda: {
        "MemTotal": MEM_TOTAL,
        "MemAvailable": 120000000000,
        "boot_id": "synthetic-boot",
        "cgroup": group,
    }
    journal = tmp_path / "gates.jsonl"
    if accepted:
        assert namespace["admit"](journal, "train", "synthetic-boot")["admitted"] is True
    else:
        with pytest.raises(ValueError, match="admission refused"):
            namespace["admit"](journal, "train", "synthetic-boot")
    rows = [json.loads(line) for line in journal.read_bytes().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["admitted"] is accepted
    assert row["required_bytes"] == TRAIN_REQUIRED
    assert row["reserve_bytes"] == 26119172096 and row["swap_or_resident_credit_bytes"] == 0
    assert row["observation"]["cgroup"] == group
    assert row["cgroup_observation_complete"] is (set(group) == {"memory.max", "memory.current"})


@pytest.mark.parametrize("fault", [None, "inventory", "commit"])
@pytest.mark.parametrize("training_steps", [120, 240])
def test_owned_exit_receipt_never_claims_overall_success_before_export_and_volume_publication(
    tmp_path, fault, training_steps
):
    namespace, export_root = extracted_helpers(tmp_path)
    launches, probes, commits = [], [], []

    class FakeProcess:
        pid, returncode = 31337, 0

        def __init__(self, command, **kwargs):
            launches.append((command, kwargs))
            assert command[0] == "/synthetic-only/python" and command[-2] == "--fit-child"
            assert kwargs["start_new_session"] is True and kwargs["stdin"] == -3
            assert kwargs["env"]["HF_HUB_OFFLINE"] == kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
            if fault is None:
                target = namespace["ROOT"] / "src/s1/browser_public_training_validation.py"
                fit = next(
                    node
                    for node in ast.parse(APP.read_bytes()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "fit_child"
                )
                refresh = next(
                    node
                    for node in fit.body
                    if isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and isinstance(node.value.func.value, ast.Name)
                    and node.value.func.value.id == "importlib"
                    and node.value.func.attr == "invalidate_caches"
                )
                code = (
                    """import importlib.util, sys
from pathlib import Path
target = Path(sys.argv[1])
source = str(target.parents[1])
# The local test venv has an editable project .pth; the Modal image does not.
sys.path[:] = [p for p in sys.path if p == source or not (Path(p) / "s1").is_dir()]
assert not target.parent.exists()
assert importlib.util.find_spec('jev_fixture_missing_module') is None
assert source in sys.path_importer_cache and sys.path_importer_cache[source] is None
target.parent.mkdir(parents=True)
(target.parent / '__init__.py').write_text('')
target.write_text('raise AssertionError("target must only be located")')
try:
    importlib.util.find_spec('s1.browser_public_training_validation')
    raise AssertionError('baseline must fail after late source creation')
except ModuleNotFoundError as exc:
    assert exc.name == 's1'
"""
                    + ast.unparse(refresh)
                    + """
print(importlib.util.find_spec('s1.browser_public_training_validation').origin)
assert 's1.browser_public_training_validation' not in sys.modules
"""
                )
                probe = subprocess.run(
                    [sys.executable, "-B", "-s", "-c", code, str(target)],
                    env=kwargs["env"],
                    cwd=kwargs["cwd"],
                    capture_output=True,
                    check=True,
                    timeout=10,
                )
                assert probe.stdout.decode().strip() == str(target)
            kwargs["stdout"].write(b"Synthetic child already exited; no worker was launched.\n")
            kwargs["stdout"].flush()

        def poll(self):
            return self.returncode

    def absent_group(pid, sig):
        assert pid == FakeProcess.pid and sig == 0
        probes.append(pid)
        raise ProcessLookupError("Synthetic owned group absent")

    def commit():
        commits.append(True)
        if fault == "commit":
            raise OSError("synthetic volume commit failure")

    namespace["os"].killpg = absent_group
    namespace["subprocess"] = SimpleNamespace(Popen=FakeProcess, DEVNULL=-3)
    namespace["host"] = lambda: {"boot_id": "synthetic-boot"}
    namespace["app"] = SimpleNamespace(app_id="synthetic-app")
    namespace["modal"] = SimpleNamespace(current_function_call_id=lambda: "synthetic-call")
    namespace["results"] = SimpleNamespace(commit=commit)
    if fault == "inventory":

        def unavailable_inventory(_root):
            raise OSError("synthetic export inventory failure")

        namespace["tree_inventory"] = unavailable_inventory
    result = namespace["owned"]("fit", "synthetic-run", "a" * 64, training_steps=training_steps)
    root = export_root / "synthetic-run-fit"
    ephemeral = namespace["RUNTIME"] / "owned-public07/synthetic-run-fit"
    saved = json.loads((ephemeral / "controller-receipt.json").read_bytes())
    config = json.loads((ephemeral / "child-config.json").read_bytes())
    assert (
        config["training_steps"]
        == saved["training_steps"]
        == result["training_steps"]
        == training_steps
    )
    assert len(launches) == 1 and probes
    assert len(commits) == {None: 1, "inventory": 0, "commit": 2}[fault]
    assert (
        saved["child_success"] is True
        and saved["owned_exit_confirmed"] is True
        and saved["returncode"] == 0
    )
    assert saved["scope"] == "child exit only; export/volume publication pending"
    assert "success" not in saved and saved.get("stage") != "complete"
    assert result["success"] is (fault is None)
    if fault is None:
        assert result["error"] is None and result["export_index"] is not None
        index = json.loads((root / "export-index.json").read_bytes())
        assert index["complete_owned_exit"] is True
        assert set(index["files"]) == {
            "child-config.json",
            "stdout.log",
            "stderr.log",
            "controller-receipt.json",
        }
    else:
        assert result["error"]["type"] == "OSError"
        assert "synthetic" in result["error"]["message"]
        assert result["export_index"] is None
        assert not (root / "export-index.json").exists()
        if fault == "commit":
            assert (root / "export-index.invalid.json").is_file()


def test_only_declared_mount_boundaries_resolve_and_retargeting_is_rejected(tmp_path):
    namespace, _ = extracted_helpers(tmp_path)
    with pytest.raises(ValueError, match="Ordinary path"):
        namespace["ordinary"]("/weights")
    with pytest.raises(ValueError, match="Undeclared"):
        namespace["capture_mount"]("/unrelated")
    binding = namespace["capture_mount"]("/weights")
    assert namespace["mount_root"](binding) == tmp_path / "provider-weights"
    alias = namespace["Path"]("/weights")
    alias.unlink()
    replacement = tmp_path / "other-provider-directory"
    replacement.mkdir()
    alias.symlink_to(replacement, target_is_directory=True)
    with pytest.raises(ValueError, match="alias/identity changed"):
        namespace["mount_root"](binding)


def test_canonical_input_projection_preserves_flat_named_bytes_and_rejects_descendant_links(
    tmp_path,
):
    namespace, _ = extracted_helpers(tmp_path)
    config = {"run_id": "case-01", "mounts": {"/input": namespace["capture_mount"]("/input")}}
    manifest = namespace["named_input"](config)
    named = namespace["named_input"](config, 7)
    namespace["create"](manifest, b'{"files":1}')
    namespace["create"](named, b"exact uploaded evidence")
    assert named.relative_to(tmp_path / "provider-input").as_posix() == "case-01/files/00007"
    copied = tmp_path / "ordinary-root/evidence.bin"
    namespace["stream_copy"](named, copied, namespace["pin"](named))
    assert copied.read_bytes() == named.read_bytes()
    for value in (-1, True, "../7"):
        with pytest.raises(ValueError, match="index"):
            namespace["named_input"](config, value)
    with pytest.raises(ValueError, match="run"):
        namespace["named_input"]({**config, "run_id": "../escape"})
    named.unlink()
    named.symlink_to(copied)
    with pytest.raises(ValueError, match="Ordinary path"):
        namespace["named_input"](config, 7)


def tiny_provider_weights(namespace, tmp_path):
    namespace.update(MODEL="public/fixture", REVISION="fixed")
    config = {
        "boot_id": "fixture-boot",
        "mounts": {"/weights": namespace["capture_mount"]("/weights")},
    }
    hub, cache = namespace["remote_cache"](config)
    cache.mkdir(parents=True)
    local = tmp_path / "ordinary-cache"
    namespace.update(HUB=local, CACHE=local / cache.relative_to(hub))
    metadata = {}
    for index in range(8):
        name, blob = f"metadata-{index}.json", json.dumps({"index": index}).encode()
        (cache / name).write_bytes(blob)
        metadata[name] = namespace["pin"](cache / name)
    blob = hub / "models--public--fixture/blobs/base"
    blob.parent.mkdir()
    blob.write_bytes(b"genuine synthetic checkpoint bytes" * 1024)
    (cache / "model.safetensors").symlink_to(blob)
    namespace.update(BASE_METADATA=metadata, BASE=namespace["pin"](blob))
    return config, cache, blob


def test_remote_stage_copies_all_eight_metadata_and_full_base_with_fresh_load_gates(tmp_path):
    namespace, _ = extracted_helpers(tmp_path)
    config, cache, base = tiny_provider_weights(namespace, tmp_path)
    observations = []

    def admit(path, scope, boot):
        assert boot == "fixture-boot" and scope == "load"
        observations.append((scope, (namespace["CACHE"] / "model.safetensors").exists()))

    namespace["admit"] = admit
    output = tmp_path / "ordinary-output"
    namespace["stage_weights"](config, output)
    assert observations == [("load", False), ("load", True)]
    staged = namespace["CACHE"]
    assert {p.name for p in staged.iterdir()} == {*namespace["BASE_METADATA"], "model.safetensors"}
    assert all(not p.is_symlink() for p in (staged, *staged.parents, *staged.iterdir()))
    assert (staged / "model.safetensors").read_bytes() == base.read_bytes()
    assert (cache / "model.safetensors").is_symlink()
    assert (
        json.loads((output / "cache-staging.json").read_bytes())["checkpoint"] == namespace["BASE"]
    )
    # Exercise the unchanged production resolver at both arbitrary canonical
    # provider HF layout and the ordinary fixed snapshot; no module/ML import.
    artifact = APP.parents[2] / "scripts/jev_public_action_artifact.py"
    definitions = [
        node
        for node in ast.parse(artifact.read_bytes()).body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"ordinary", "require", "resolve_base_checkpoint"}
    ]
    resolver = {"Path": Path, "MODEL_ID": namespace["MODEL"], "REVISION": namespace["REVISION"]}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(artifact), "exec"), resolver)
    assert resolver["resolve_base_checkpoint"](cache) == base
    assert resolver["resolve_base_checkpoint"](staged) == staged / "model.safetensors"


def test_remote_stage_refuses_corrupt_full_base_and_keeps_failure_bytes(tmp_path):
    namespace, _ = extracted_helpers(tmp_path)
    config, _, base = tiny_provider_weights(namespace, tmp_path)
    base.write_bytes(b"x" * namespace["BASE"]["size_bytes"])
    gates = []
    namespace["admit"] = lambda *args: gates.append(args[1])
    output = tmp_path / "ordinary-output"
    with pytest.raises(ValueError, match="source/content changed"):
        namespace["stage_weights"](config, output)
    assert gates == ["load"]
    assert not (output / "cache-staging.json").exists()
    assert (namespace["CACHE"] / "model.safetensors").read_bytes() == base.read_bytes()


def test_remote_stage_never_deletes_or_overwrites_an_existing_local_cache(tmp_path):
    namespace, _ = extracted_helpers(tmp_path)
    config, _, _ = tiny_provider_weights(namespace, tmp_path)
    local = namespace["HUB"]
    local.mkdir()
    (local / "existing-local-model").write_bytes(b"must survive")
    namespace["admit"] = lambda *args: pytest.fail("must refuse before any checkpoint gate/read")
    with pytest.raises(ValueError, match="must be empty"):
        namespace["stage_weights"](config, tmp_path / "output")
    assert list(local.iterdir()) == [local / "existing-local-model"]
    assert (local / "existing-local-model").read_bytes() == b"must survive"


def test_streamed_copy_refuses_existing_destination_and_source_mutation(tmp_path):
    namespace, _ = extracted_helpers(tmp_path)
    source, target = tmp_path / "source.bin", tmp_path / "target.bin"
    source.write_bytes(b"real source bytes")
    expected = namespace["pin"](source)
    target.write_bytes(b"existing destination")
    with pytest.raises(FileExistsError):
        namespace["stream_copy"](source, target, expected)
    assert target.read_bytes() == b"existing destination"
    target.unlink()

    def changed_after_read(fd):
        observed = os.fstat(fd)
        source.write_bytes(b"changed after original read")
        return observed

    namespace["os"].fstat = changed_after_read
    with pytest.raises(ValueError, match="source/content changed"):
        namespace["stream_copy"](source, target, expected)
    assert target.read_bytes() == b"real source bytes"


@pytest.mark.parametrize("weights_fail", [False, True])
def test_download_publication_commits_weights_before_marker_and_results(tmp_path, weights_fail):
    namespace, export_root = extracted_helpers(tmp_path)
    events = []

    class Process:
        pid, returncode = 52, 0

        def __init__(self, command, **kwargs):
            assert command[-2] == "--download-child"

        def poll(self):
            return self.returncode

    def absent_group(*args):
        raise ProcessLookupError()

    marker = export_root / "download-01-download/export-index.json"

    def weights_commit():
        assert not marker.exists()
        events.append("weights")
        if weights_fail:
            raise OSError("weights publication failed")

    def results_commit():
        assert marker.exists()
        events.append("results")

    namespace.update(
        subprocess=SimpleNamespace(Popen=Process, DEVNULL=-3),
        host=lambda: {"boot_id": "fixture-boot"},
        app=SimpleNamespace(app_id="fixture-app"),
        modal=SimpleNamespace(current_function_call_id=lambda: "fixture-call"),
        weights=SimpleNamespace(commit=weights_commit),
        results=SimpleNamespace(commit=results_commit),
    )
    namespace["os"].killpg = absent_group
    result = namespace["owned"]("download", "download-01")
    assert result["success"] is (not weights_fail)
    assert events == (["weights"] if weights_fail else ["weights", "results"])
    assert marker.exists() is (not weights_fail)
    if weights_fail:
        assert result["export_index"] is None
        assert result["error"]["message"] == "weights publication failed"


def test_failed_child_primary_survives_failed_full_export_copy(tmp_path):
    namespace, export_root = extracted_helpers(tmp_path)

    class Process:
        pid, returncode = 53, 17

        def __init__(self, command, **kwargs):
            pass

        def poll(self):
            return self.returncode

    def absent_group(*args):
        raise ProcessLookupError()

    def broken_copy(*args):
        raise OSError("full export copy failed")

    namespace.update(
        subprocess=SimpleNamespace(Popen=Process, DEVNULL=-3),
        host=lambda: {"boot_id": "fixture-boot"},
        app=SimpleNamespace(app_id="fixture-app"),
        modal=SimpleNamespace(current_function_call_id=lambda: "fixture-call"),
        results=SimpleNamespace(commit=lambda: None),
        copy_output=broken_copy,
    )
    namespace["os"].killpg = absent_group
    result = namespace["owned"]("fit", "failed-01", "a" * 64)
    assert result["success"] is False and result["export_index"] is None
    assert result["error"]["type"] == "ValueError"
    assert result["error"]["message"] == "Owned child failed: 17"
    assert any("full export copy failed" in note for note in result["error"]["notes"])
    assert not list(export_root.rglob("export-index.json"))


@pytest.mark.parametrize("value", [360, 336, True, 120.0, "240", None])
def test_training_steps_refused_before_output_or_upload(tmp_path, value):
    namespace, _export = extracted_helpers(tmp_path)
    tree = ast.parse(APP.read_bytes())
    namespace["Literal"] = Literal
    for name in ("fit", "main"):
        node = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
        )
        node.decorator_list = []
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(APP), "exec"), namespace)
        assert get_args(get_type_hints(namespace[name])["training_steps"]) == (120, 240)
        assert namespace[name].__defaults__[-1] == 120
    with pytest.raises(ValueError, match="Training steps must be120 or240"):
        namespace["fit"]("refused", "a" * 64, training_steps=value)
    output = tmp_path / "must-not-upload"
    with pytest.raises(ValueError, match="Training steps must be120 or240"):
        namespace["main"](
            output=str(output),
            manifest="must-not-read",
            run_id="refused",
            run=True,
            training_steps=value,
        )
    assert not output.exists()
    assert not (namespace["RUNTIME"] / "owned-public07").exists()


@pytest.mark.parametrize("steps,completed", [(120, 120), (240, 240), (240, 120)])
def test_selected_steps_reach_real_fitter_boundary_and_completed_count_check(
    tmp_path, steps, completed
):
    """Execute the actual child training tail with a fake fitter, never its setup/model path."""
    namespace, _export = extracted_helpers(tmp_path)
    node = next(
        node
        for node in ast.parse(APP.read_bytes()).body
        if isinstance(node, ast.FunctionDef) and node.name == "fit_child"
    )
    start = next(
        index
        for index, entry in enumerate(node.body)
        if isinstance(entry, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "result" for target in entry.targets)
    )
    function = ast.parse("def child_training_tail():\n    pass\n").body[0]
    function.body = [node.body[0], *node.body[start:]]
    output = tmp_path / "child"
    (output / "fit/adapter").mkdir(parents=True)
    (output / "fit/receipt.json").write_bytes(b"{}")
    (output / "fit/adapter/manifest.json").write_bytes(b"{}")
    calls = []

    def fit_public_actions(**kwargs):
        calls.append(kwargs)
        return {"stage": "complete", "training_updates": completed, "qualified": False}

    namespace.update(
        config={"training_steps": steps, "boot_id": "fixed-boot"},
        output=output,
        HANDOFF=tmp_path / "fixed-handoff.json",
        HANDOFF_SHA="b" * 64,
        fitter=SimpleNamespace(fit_public_actions=fit_public_actions),
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(APP), "exec"), namespace)
    if steps == completed:
        assert namespace["child_training_tail"]() == 0
        receipt = json.loads((output / "fit-child-result.json").read_bytes())
        assert receipt["training_updates"] == completed and receipt["qualified"] is False
    else:
        with pytest.raises(ValueError, match="Complete requested-step"):
            namespace["child_training_tail"]()
        assert not (output / "fit-child-result.json").exists()
    assert calls == [
        {
            "output": output / "fit",
            "source_handoff": namespace["HANDOFF"],
            "expected_source_handoff": "b" * 64,
            "expected_boot_id": "fixed-boot",
            "steps": steps,
            "learning_rate": 5e-5,
            "seed": 42,
        }
    ]


@pytest.mark.parametrize("steps,completed", [(120, 120), (240, 240), (240, 120)])
def test_cli_forwards_selected_steps_and_refuses_mismatched_closed_result(
    tmp_path, steps, completed
):
    namespace, _export = extracted_helpers(tmp_path)
    node = next(
        node
        for node in ast.parse(APP.read_bytes()).body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    start = next(
        index
        for index, entry in enumerate(node.body)
        if isinstance(entry, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "fitted" for target in entry.targets)
    )
    function = ast.parse("def cli_fit_tail():\n    pass\n").body[0]
    function.body = node.body[start:]
    calls, exports = [], []

    def remote(run_id, manifest_sha256, **kwargs):
        calls.append((run_id, manifest_sha256, kwargs))
        return {"success": True, "training_steps": completed}

    namespace.update(
        supplied=b"{}",
        run_id="selected-run",
        training_steps=steps,
        local=tmp_path / "local",
        fit=SimpleNamespace(remote=remote),
        export=lambda *args: exports.append(args),
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(APP), "exec"), namespace)
    if steps == completed:
        namespace["cli_fit_tail"]()
    else:
        with pytest.raises(ValueError, match="different step count"):
            namespace["cli_fit_tail"]()
    assert calls == [("selected-run", hashlib.sha256(b"{}").hexdigest(), {"training_steps": steps})]
    assert exports == [({"success": True, "training_steps": completed}, tmp_path / "local/fit-job")]
