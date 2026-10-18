"""Pure D3 wrapper boundaries; no Modal, model, GPU, network or package installation."""

import ast
import hashlib
import json
import os
import re
import shutil
import signal
import stat
import tempfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

APP = Path(__file__).resolve().parents[1] / "apps/modal/decision3_public_replay.py"


def helpers(tmp_path):
    names = {
        "require",
        "encode",
        "parse",
        "ordinary",
        "capture_mount",
        "mount_root",
        "relative",
        "read",
        "pin",
        "create",
        "stream_copy",
        "dependency_commands",
        "child_environment",
        "replay_output",
        "provider_snapshot",
        "provider_cache_file",
        "runtime_files",
        "stage_cache",
        "group_alive",
        "remember_model_groups",
        "proc_read",
        "proc_record",
        "discover_owned_processes",
        "stop_owned",
        "tree_inventory",
        "preserve_replay",
        "publish_output",
        "owned",
        "receive",
        "export",
    }
    tree = ast.parse(APP.read_bytes())
    body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in body} == names
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        for function in body
        for node in ast.walk(function)
    )
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id
        in {"PACKAGES", "WHEELS", "MODELS", "CORE_SHA", "INTERPRETER", "DECLARED_MOUNTS"}
    }
    export_root = tmp_path / "export"
    export_root.mkdir()
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    input_root, weights_root = tmp_path / "input", tmp_path / "weights"
    input_root.mkdir()
    weights_root.mkdir()

    def task_path(value, *parts):
        path = Path(value, *parts)
        return {
            Path("/export"): export_root,
            Path("/proc"): proc_root,
            Path("/input"): input_root,
            Path("/weights"): weights_root,
        }.get(path, path)

    namespace = {
        **constants,
        "Path": task_path,
        "PurePosixPath": PurePosixPath,
        "hashlib": hashlib,
        "json": json,
        "re": re,
        "signal": signal,
        "stat": stat,
        "ROOT": tmp_path / "source",
        "SCRIPT": tmp_path / "source/apps/modal/decision3_public_replay.py",
        "CORE": tmp_path / "source/scripts/jev_decision3_public_dev22_gpu_replay.py",
        "HUB": tmp_path / "cache",
        "GIB": 1024**3,
        "MAX_FILES": 4096,
        "MAX_LOG": 16 * 1024**2,
        "os": SimpleNamespace(environ={}, fstat=os.fstat, fsync=os.fsync, scandir=os.scandir),
        "tempfile": SimpleNamespace(TemporaryFile=lambda: tempfile.TemporaryFile(dir=tmp_path)),
        "time": SimpleNamespace(monotonic=lambda: 100.0),
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(APP), "exec"), namespace)
    return namespace, export_root


def test_dedicated_dependencies_resolve_normally_and_use_verified_wheels(tmp_path):
    namespace, _ = helpers(tmp_path)
    install, check = namespace["dependency_commands"](tmp_path / "constraints.txt")
    assert install[:3] == ["/opt/decision3/bin/python", "-m", "pip"]
    assert "--no-deps" not in install and "--constraint" in install
    assert check == ["/opt/decision3/bin/python", "-m", "pip", "check"]
    assert all("#sha256=" in value for value in namespace["WHEELS"].values())
    assert namespace["WHEELS"]["causal-conv1d"].endswith(
        "#sha256=e03db18b70006574a8f4206297cdeedd55d2ae1284034cb60ef9d02c4c014696"
    )
    assert {"flash-linear-attention==0.5.2", "fla-core==0.5.2", "transformers==5.17.0"} <= set(
        install
    )
    assert namespace["PACKAGES"]["torch"] == "2.10.0+cu130"


def test_child_environment_excludes_credentials_and_gemma_overlay(tmp_path):
    namespace, _ = helpers(tmp_path)
    namespace["os"].environ = {
        "HF_TOKEN": "private",
        "AWS_SECRET_ACCESS_KEY": "private",
        "PYTHONPATH": "/private/Gemma-overlay",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    replay = namespace["child_environment"]("replay")
    assert replay["PYTHONPATH"] == str(namespace["ROOT"] / "src") + ":" + str(
        namespace["ROOT"] / "scripts"
    )
    assert replay["HF_HUB_OFFLINE"] == replay["TRANSFORMERS_OFFLINE"] == "1"
    assert replay["CUDA_VISIBLE_DEVICES"] == "0"
    assert not {"HF_TOKEN", "AWS_SECRET_ACCESS_KEY"} & set(replay)
    assert "Gemma-overlay" not in json.dumps(replay)
    assert namespace["child_environment"]("download")["CUDA_VISIBLE_DEVICES"] == ""


def test_exact_resource_and_named_upload_declarations_are_not_imported():
    tree = ast.parse(APP.read_bytes())
    for name in ("download", "replay"):
        function = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
        )
        configuration = {item.arg: item.value for item in function.decorator_list[0].keywords}
        for key, value in {
            "cpu": 2,
            "memory": 344064,
            "ephemeral_disk": 524288,
            "max_containers": 1,
            "retries": 0,
            "single_use_containers": True,
            "include_source": False,
        }.items():
            assert ast.literal_eval(configuration[key]) == value
        if name == "replay":
            assert ast.literal_eval(configuration["gpu"]) == "H200"
            assert ast.literal_eval(configuration["block_network"]) is True
            mounts = configuration["volumes"]
            assert [ast.literal_eval(key) for key in mounts.keys] == [
                "/weights",
                "/input",
                "/export",
            ]
            assert (
                sum(
                    isinstance(value, ast.Call)
                    and value.func.attr == "with_mount_options"
                    and ast.literal_eval(value.keywords[0].value) is True
                    for value in mounts.values
                )
                == 2
            )
    assert not any(
        isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("apps.modal")
        for node in ast.walk(tree)
    )
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "add_local_dir" for node in ast.walk(tree)
    )
    assert (
        sum(
            isinstance(node, ast.Attribute) and node.attr == "add_local_file"
            for node in ast.walk(tree)
        )
        == 1
    )


@pytest.mark.parametrize("fault", [None, "discovery", "preserve", "commit"])
def test_owned_nested_groups_and_full_raw_tree_are_accounted_before_success(tmp_path, fault):
    namespace, export_root = helpers(tmp_path)
    events, launches = [], []

    class Process:
        pid, returncode = 31337, 0

        def __init__(self, command, **kwargs):
            launches.append((command, kwargs))
            assert command[0] == "/opt/decision3/bin/python" and command[-2] == "--replay-child"
            assert kwargs["start_new_session"] is True
            kwargs["stdout"].write(b"Full synthetic wrapper log.\n")
            kwargs["stdout"].flush()

        def poll(self):
            return self.returncode

    def discover(run_id, groups, *, force=False):
        events.append("discover")
        if fault == "discovery":
            raise ValueError("synthetic discovery error")
        groups[31338] = {"model": "d3", "command": ["synthetic-model"]}

    def absent(pid, sig):
        assert pid in {31337, 31338} and sig == 0
        events.append("probe-" + str(pid))
        raise ProcessLookupError("Synthetic owned group absent")

    def preserve(run_id, output):
        events.append("preserve")
        namespace["create"](output / "replay/raw.json", b'{"probabilities":{"A":0.2,"B":0.8}}')
        if fault == "preserve":
            raise OSError("synthetic copy failure after partial preservation")

    def commit():
        events.append("commit")
        if fault == "commit":
            raise OSError("synthetic result commit failure")

    namespace.update(
        subprocess=SimpleNamespace(Popen=Process, DEVNULL=-3),
        remember_model_groups=discover,
        discover_owned_processes=lambda run_id, groups: {
            "observed_processes": 0,
            "owned_pids": [],
            "unknown": [],
        },
        preserve_replay=preserve,
        app=SimpleNamespace(app_id="synthetic-app"),
        modal=SimpleNamespace(current_function_call_id=lambda: "synthetic-call"),
        results=SimpleNamespace(commit=commit),
    )
    namespace["os"].killpg = absent
    source = namespace["replay_output"]("synthetic-run")
    source.mkdir(parents=True)
    result = namespace["owned"]("replay", "synthetic-run", "a" * 64)
    output = namespace["ROOT"] / "artifacts/jev-d3-modal-owned/decision3-synthetic-run-replay"
    report = json.loads((output / "controller-receipt.json").read_bytes())
    assert len(launches) == 1 and "probe-31337" in events and events[-1] == "commit"
    assert report["owned_exit_confirmed"] is (fault != "discovery")
    assert "success" not in report
    assert result["success"] is (fault is None)
    assert (
        "synthetic" in result["error"]["message"] if fault is not None else result["error"] is None
    )
    if fault == "discovery":
        assert "preserve" not in events and result["export_index"] is None
    else:
        assert "probe-31338" in events
        index = json.loads((output / "export-index.json").read_bytes())
        assert {"replay/raw.json", "stdout.log", "stderr.log", "controller-receipt.json"} <= set(
            index["files"]
        )
        assert report["replay_preservation_complete"] is (fault != "preserve")
    assert namespace["tree_inventory"](output) == namespace["tree_inventory"](
        export_root / output.name
    )


@pytest.mark.parametrize("unknown", [False, True])
def test_cleanup_fresh_proc_scan_catches_session_created_after_term(tmp_path, unknown):
    namespace, _ = helpers(tmp_path)
    root_pid, late_pid = 31337, 31338
    live, signals = {root_pid}, []
    marker = "JEV_DECISION3_RUN_OUTPUT=" + str(namespace["replay_output"]("late-run"))

    def proc(pid, command, parent):
        path = tmp_path / "proc" / str(pid)
        path.mkdir()
        fields = ["S", str(parent), str(pid), str(pid), *(["0"] * 15), "100"]
        (path / "stat").write_text(str(pid) + " (synthetic owned process) " + " ".join(fields))
        (path / "environ").write_bytes((marker + "\x00UNRELATED_SECRET=never-persist\x00").encode())
        (path / "cmdline").write_bytes(b"\x00".join(item.encode() for item in command) + b"\x00")

    proc(root_pid, ["synthetic-owned-wrapper"], 1)
    command = [
        namespace["INTERPRETER"],
        "-B",
        "-s",
        str(namespace["CORE"]),
        "--source-sha256",
        namespace["CORE_SHA"],
        "--worker",
        "d3",
        "--output-dir",
        str(namespace["replay_output"]("late-run") / "d3"),
        "--expected-boot-id",
        "11111111-2222-3333-4444-555555555555",
        "--expected-memtotal",
        "274877906944",
    ]

    class Process:
        pid, returncode = root_pid, None

        def poll(self):
            if self.pid not in live:
                self.returncode = -15
            return self.returncode

    def killpg(pid, sig):
        if pid not in live:
            raise ProcessLookupError("Synthetic group exited")
        if sig == 0:
            return
        signals.append((pid, sig))
        live.remove(pid)
        shutil.rmtree(tmp_path / "proc" / str(pid))
        if pid == root_pid:
            # No receipt exists: spawn appears only after the first cleanup scan,
            # and its parent has already been reaped/reparented.
            live.add(late_pid)
            proc(late_pid, ["unknown-compiler"] if unknown else command, 1)

    ticks = iter(range(1000))
    namespace["time"] = SimpleNamespace(monotonic=lambda: next(ticks), sleep=lambda _: None)
    namespace["os"].killpg = killpg
    groups = {root_pid: {"model": None, "start_ticks": 100}}
    if unknown:
        with pytest.raises(ValueError, match="exit unconfirmed") as caught:
            namespace["stop_owned"](Process(), groups, "late-run")
        evidence = caught.value.owned_process_observations
        assert evidence["history"][-1]["unknown"] == [
            {"pid": late_pid, "group": late_pid, "reason": "Unknown owned session"}
        ]
        assert live == {late_pid} and all(pid != late_pid for pid, _ in signals)
    else:
        evidence = namespace["stop_owned"](Process(), groups, "late-run")
        assert not live and (late_pid, signal.SIGTERM) in signals
        assert groups[late_pid]["command"] == command
        assert evidence["history"][-1]["owned_pids"] == []
    assert evidence["passes"] >= 3
    assert "never-persist" not in json.dumps(evidence) + json.dumps(groups)


def test_complete_raw_preservation_includes_pending_files_and_rejects_links(tmp_path):
    namespace, _ = helpers(tmp_path)
    source = namespace["replay_output"]("case")
    source.mkdir(parents=True)
    (source / ".pending-row").write_bytes(b"partial raw native response")
    (source / "nested").mkdir()
    (source / "nested/raw.json").write_bytes(b'{"raw":true}')
    output = tmp_path / "result"
    output.mkdir()
    namespace["preserve_replay"]("case", output)
    assert namespace["tree_inventory"](source) == namespace["tree_inventory"](output / "replay")
    (source / "linked").symlink_to(source / ".pending-row")
    with pytest.raises(ValueError, match="links"):
        namespace["tree_inventory"](source)
    with pytest.raises(ValueError, match="Confined"):
        namespace["relative"]("../.aws/credentials")


def test_only_declared_mount_boundary_resolves_and_identity_drift_refuses(tmp_path):
    namespace, _ = helpers(tmp_path)
    alias, storage = tmp_path / "input", tmp_path / "provider-input-storage"
    alias.rename(storage)
    alias.symlink_to(storage, target_is_directory=True)
    binding = namespace["capture_mount"]("/input")
    assert binding["root"] == str(storage)
    assert namespace["mount_root"](binding) == storage
    with pytest.raises(ValueError, match="Declared"):
        namespace["capture_mount"]("/arbitrary-provider-path")
    (storage / "child").symlink_to(tmp_path / "proc", target_is_directory=True)
    with pytest.raises(ValueError, match="Ordinary"):
        namespace["ordinary"](storage / "child/file.json")
    alternate = tmp_path / "replacement-input-storage"
    alternate.mkdir()
    alias.unlink()
    alias.symlink_to(alternate, target_is_directory=True)
    with pytest.raises(ValueError, match="identity changed"):
        namespace["mount_root"](binding)


@pytest.mark.parametrize("corrupt", [False, True])
def test_six_exact_provider_snapshots_stage_as_ordinary_files_without_ml(tmp_path, corrupt):
    namespace, _ = helpers(tmp_path)
    alias, storage = tmp_path / "weights", tmp_path / "provider-weight-storage"
    alias.rename(storage)
    alias.symlink_to(storage, target_is_directory=True)
    binding = namespace["capture_mount"]("/weights")
    identities = {}
    for name in namespace["MODELS"]:
        payloads = {
            "config.json": namespace["encode"]({"model": name}),
            "model.safetensors": b"synthetic-byte-fixture-not-real-weights-" + name.encode(),
            "readout.safetensors": b"synthetic-readout-" + name.encode(),
            "README.md": b"excluded card",
        }
        manifest = {
            "model_name": name,
            "repo_id": "publisher/" + name,
            "files_sha256": {
                label: hashlib.sha256(blob).hexdigest() for label, blob in payloads.items()
            },
            "files_bytes": {label: len(blob) for label, blob in payloads.items()},
        }
        blob = namespace["encode"](manifest)
        identity = {
            "model_id": manifest["repo_id"],
            "revision": "revision-" + name,
            "manifest_sha256": hashlib.sha256(blob).hexdigest(),
        }
        identities[name] = identity
        snapshot = namespace["provider_snapshot"](binding, identity)
        snapshot.mkdir(parents=True)
        (snapshot / "MODEL_MANIFEST.json").write_bytes(blob)
        for label, data in payloads.items():
            if label == "model.safetensors":
                cached = snapshot.parent.parent / "blobs" / manifest["files_sha256"][label]
                cached.parent.mkdir(exist_ok=True)
                cached.write_bytes(b"incorrect" if corrupt and name == "d3-edge" else data)
                (snapshot / label).symlink_to("../../blobs/" + cached.name)
            else:
                (snapshot / label).write_bytes(data)
    output = tmp_path / "stage-output"
    output.mkdir()

    def read_manifest(path, sha):
        assert namespace["pin"](path)["sha256"] == sha
        return json.loads(path.read_bytes())

    core = SimpleNamespace(read=read_manifest, pin=namespace["pin"])
    config = {"mounts": {"/weights": binding}, "output": str(output)}
    if corrupt:
        with pytest.raises(ValueError, match="copy source|SHA/size"):
            namespace["stage_cache"](config, core, identities)
        assert not (output / "cache-staging-result.json").exists()
    else:
        namespace["stage_cache"](config, core, identities)
        report = json.loads((output / "cache-staging-result.json").read_bytes())
        assert tuple(report["models"]) == tuple(sorted(namespace["MODELS"]))
        for name, model in report["models"].items():
            target = Path(model["snapshot"])
            assert target.is_relative_to(namespace["HUB"])
            assert not any(path.is_symlink() for path in (target, *target.parents))
            assert set(model["files"]) == {
                "MODEL_MANIFEST.json",
                "config.json",
                "model.safetensors",
                "readout.safetensors",
            }
            assert all(not (target / label).is_symlink() for label in model["files"])
            assert not (target / "README.md").exists()


def test_provider_publication_copies_full_payload_and_writes_closure_index_last(tmp_path):
    namespace, export_root = helpers(tmp_path)
    storage = tmp_path / "provider-export-storage"
    export_root.rename(storage)
    export_root.symlink_to(storage, target_is_directory=True)
    source = tmp_path / "decision3-publish-run-replay"
    source.mkdir()
    (source / "raw.json").write_bytes(b'{"all_probabilities":[0.2,0.8]}')
    (source / ".pending-row").write_bytes(b"partial evidence")
    namespace["create"](
        source / "export-index.json", namespace["encode"](namespace["tree_inventory"](source))
    )
    copied, original = [], namespace["stream_copy"]

    def copy(source, destination, expected):
        copied.append(destination.name)
        original(source, destination, expected)

    namespace["stream_copy"] = copy
    binding = namespace["capture_mount"]("/export")
    index = namespace["publish_output"](source, binding, True)
    assert copied[-1] == "export-index.json"
    assert index == namespace["pin"](source / "export-index.json")
    assert namespace["tree_inventory"](storage / source.name) == namespace["tree_inventory"](source)


def test_full_export_streams_every_file_and_checks_sha(tmp_path):
    namespace, _ = helpers(tmp_path)
    raw = {
        "controller-receipt.json": b'{"child_success":true}',
        "replay/.pending-row": b"partial",
        "replay/raw.json": b'{"scores":[0.1,0.9]}',
    }
    pins = {
        name: {"sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)}
        for name, blob in raw.items()
    }
    index = namespace["encode"](
        {"root": "decision3-case-replay", "complete_owned_exit": True, "files": pins}
    )
    files = {**raw, "export-index.json": index}
    calls = []

    def stream(remote):
        calls.append(remote)
        blob = files[remote.removeprefix("decision3-case-replay/")]
        yield blob[:3]
        yield blob[3:]

    namespace["results"] = SimpleNamespace(read_file=stream)
    result = {
        "root": "decision3-case-replay",
        "export_index": {"sha256": hashlib.sha256(index).hexdigest(), "size_bytes": len(index)},
    }
    namespace["export"](result, tmp_path / "received")
    assert len(calls) == len(files)
    assert all((tmp_path / "received" / name).read_bytes() == blob for name, blob in files.items())
    with pytest.raises(ValueError, match="SHA/size"):
        namespace["receive"](
            "decision3-case-replay/replay/raw.json",
            tmp_path / "corrupt",
            {"sha256": "0" * 64, "size_bytes": len(raw["replay/raw.json"])},
        )
