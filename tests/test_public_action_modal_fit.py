"""Pure boundary checks; no Modal/ML imports, network, worker or GPU execution."""

import ast
import hashlib
import json
import os
import re
import signal
import stat
import tempfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest

APP = Path(__file__).resolve().parents[1] / "apps/modal/public_action_fit.py"
GIB = 1024**3
MEM_TOTAL = 130595860480
TRAIN_REQUIRED = 106438961088


def extracted_helpers(tmp_path):
    """Compile only reviewed function definitions, never the application module."""
    names = {
        "require",
        "encode",
        "parse",
        "ordinary",
        "relative",
        "read",
        "pin",
        "create",
        "admit",
        "group_alive",
        "stop_owned",
        "tree_inventory",
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
    export_root = tmp_path / "export"
    export_root.mkdir()

    def task_path(value, *parts):
        path = Path(value, *parts)
        return export_root if path == Path("/export") else path

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
        "ROOT": tmp_path / "source",
        "SCRIPT": tmp_path / "source/apps/modal/public_action_fit.py",
        "OVERLAY": tmp_path / "source/overlay",
        "HUB": tmp_path / "cache/hub",
        "INTERPRETER": "/synthetic-only/python",
        "time": SimpleNamespace(monotonic=lambda: 100.0),
        "os": SimpleNamespace(fsync=os.fsync, fstat=os.fstat, scandir=os.scandir, environ={}),
        "tempfile": SimpleNamespace(TemporaryFile=lambda: tempfile.TemporaryFile(dir=tmp_path)),
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(APP), "exec"), namespace)
    return namespace, export_root


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
def test_owned_exit_receipt_never_claims_overall_success_before_export_and_volume_publication(
    tmp_path, fault
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
    result = namespace["owned"]("fit", "synthetic-run", "a" * 64)
    root = export_root / "synthetic-run-fit"
    saved = json.loads((root / "controller-receipt.json").read_bytes())
    assert len(launches) == 1 and probes and commits == [True]
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
        assert (result["export_index"] is None) is (fault == "inventory")
