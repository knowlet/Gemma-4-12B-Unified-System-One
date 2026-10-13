"""Retain worker failure evidence without importing Modal or running a GPU/server."""

import ast
import importlib.metadata
import io
import json
import re
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from s1 import api, browser_policy, unified

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cloud_worker(tmp_path, monkeypatch):
    source = ROOT / "apps/modal/jev_browser.py"
    tree = ast.parse(source.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_record_failure", "benchmark_cell"}
    ]
    for node in functions:
        node.decorator_list = []
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"UPSTREAM_REVISION", "BASE_REVISION", "SPECIALIST_REVISION"}
    }
    state = SimpleNamespace(
        fail_at=None,
        fail_commit=False,
        commits=[],
        model_calls=[],
        cudnn_sdp=True,
        initial_receipts=[],
        sockets=[],
        servers=[],
        threads=[],
        root=tmp_path / "browser-runs",
    )

    def task_path(value):
        return state.root if value == "/vol/browser-runs" else Path(value)

    def output():
        return state.root / "offline-test" / state.cell

    def load_model(name, **kwargs):
        assert state.cudnn_sdp is False
        state.model_calls.append((name, kwargs))
        state.initial_receipts.append(json.loads((output() / "receipt.json").read_text()))
        if state.fail_at == "helper_model_loading" or (
            state.fail_at == "backend_loading" and name.startswith("knowlet/")
        ):
            raise RuntimeError("offline model load failure")
        return SimpleNamespace(name=name, revision=kwargs["revision"], temperature=1.0)

    class Socket:
        def __init__(self):
            self.closed = False
            state.sockets.append(self)

        def bind(self, address):
            assert address == ("127.0.0.1", 0)

        def getsockname(self):
            return ("127.0.0.1", 12345)

        def close(self):
            self.closed = True

    class Server:
        def __init__(self, config):
            if state.fail_at == "server_start":
                raise RuntimeError("offline server creation failure")
            self.started = False
            self.should_exit = False
            state.servers.append(self)

        def run(self, *, sockets):
            assert sockets == state.sockets
            self.started = True

    class Thread:
        def __init__(self, *, target, kwargs, daemon):
            assert daemon is True
            self.target = target
            self.kwargs = kwargs
            self.joined = False
            state.threads.append(self)

        def start(self):
            if state.fail_at == "thread_start":
                raise RuntimeError("offline thread start failure")
            self.target(**self.kwargs)

        def join(self, timeout):
            assert timeout == 10
            self.joined = True
            if state.fail_at == "cleanup_thread":
                raise RuntimeError("offline thread cleanup failure")

        def is_alive(self):
            return False

    def run_benchmark(**kwargs):
        assert kwargs["endpoint"] == "http://127.0.0.1:12345/v1/systemone"
        if state.fail_at == "benchmark":
            directory = kwargs["output_dir"]
            directory.mkdir()
            (directory / "partial.json").write_text('{"status":"partial"}')
            raise RuntimeError("offline inference failure")
        return {"completion_rate": 0.0, "tasks": 1}

    def commit():
        state.commits.append(json.loads((output() / "receipt.json").read_text()))
        if state.fail_commit:
            raise RuntimeError("offline volume commit failure")

    monkeypatch.setattr(unified, "UnifiedDecisionModel", load_model)
    monkeypatch.setattr(
        browser_policy,
        "BrowserDecisionBackend",
        lambda model, prompt_mode: SimpleNamespace(name=model.name),
    )
    monkeypatch.setattr(api, "create_app", lambda *args, **kwargs: object())
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "offline")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            backends=SimpleNamespace(
                cuda=SimpleNamespace(
                    enable_cudnn_sdp=lambda enabled: setattr(state, "cudnn_sdp", enabled),
                    cudnn_sdp_enabled=lambda: state.cudnn_sdp,
                )
            ),
            cuda=SimpleNamespace(
                get_device_name=lambda: "offline-gpu",
                max_memory_allocated=lambda: 123,
            ),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(Config=lambda *args, **kwargs: object(), Server=Server),
    )
    monkeypatch.setitem(sys.modules, "socket", SimpleNamespace(socket=Socket))
    monkeypatch.setitem(sys.modules, "threading", SimpleNamespace(Thread=Thread))
    monkeypatch.setitem(
        sys.modules, "jev_browser_benchmark", SimpleNamespace(run_benchmark=run_benchmark)
    )
    # Restore the worker's script import path after each isolated execution.
    monkeypatch.setattr(sys, "path", list(sys.path))
    namespace = {
        "__file__": str(source),
        "Path": task_path,
        "json": json,
        "re": re,
        "io": io,
        "zipfile": zipfile,
        "app": SimpleNamespace(app_id="offline-app"),
        "modal": SimpleNamespace(current_function_call_id=lambda: "offline-call"),
        "volume": SimpleNamespace(commit=commit),
        "field_text_helper": lambda *args: pytest.fail("no helper generation in offline tests"),
        **constants,
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)

    def execute(cell="native-base", **kwargs):
        state.cell = cell
        return namespace["benchmark_cell"](
            kwargs.get("run", "offline-test"),
            cell,
            kwargs.get("repeats", 1),
            kwargs.get("split", "development"),
            "",
        )

    state.execute = execute
    state.output = output
    state.constants = constants
    return state


def check_archived_receipt(state, result):
    receipt = result["receipt"]
    assert json.loads((state.output() / "receipt.json").read_text()) == receipt
    with zipfile.ZipFile(io.BytesIO(result["archive"])) as archive:
        assert json.loads(archive.read("receipt.json")) == receipt
        assert all(not name.startswith("/") for name in archive.namelist())
    assert receipt["elapsed_ms"] >= receipt["setup_ms"] >= 0
    return receipt


@pytest.mark.parametrize("stage", ["runtime_import", "benchmark_import"])
def test_worker_import_failures_are_committed_and_archived(cloud_worker, monkeypatch, stage):
    module = "s1.unified" if stage == "runtime_import" else "jev_browser_benchmark"
    monkeypatch.setitem(sys.modules, module, None)
    result = cloud_worker.execute()
    receipt = check_archived_receipt(cloud_worker, result)
    assert receipt["status"] == "failed"
    assert receipt["stage"] == stage
    assert receipt["error"]["type"] == "ModuleNotFoundError"
    assert receipt["model"] is None
    assert cloud_worker.model_calls == []
    assert len(cloud_worker.commits) == 1
    with zipfile.ZipFile(io.BytesIO(result["archive"])) as archive:
        assert "ModuleNotFoundError" in archive.read("failure.txt").decode()


@pytest.mark.parametrize(
    ("stage", "cell"),
    [("helper_model_loading", "native-base"), ("backend_loading", "user-specialist")],
)
def test_model_load_failure_retains_early_receipt_and_public_revision(cloud_worker, stage, cell):
    cloud_worker.fail_at = stage
    receipt = check_archived_receipt(cloud_worker, cloud_worker.execute(cell))
    assert receipt["status"] == "failed"
    assert receipt["stage"] == stage
    assert receipt["error"]["message"] == "offline model load failure"
    assert receipt["model"] is None
    assert cloud_worker.initial_receipts[0]["status"] == "initializing"
    assert cloud_worker.initial_receipts[0]["model"] is None
    assert cloud_worker.sockets == []
    assert len(cloud_worker.commits) == 1
    if cell.endswith("specialist"):
        name, settings = cloud_worker.model_calls[-1]
        assert name == "knowlet/Gemma-4-12B-Unified-System-One"
        assert settings["revision"] == cloud_worker.constants["SPECIALIST_REVISION"]
        assert receipt["text_helper"]["revision"] == cloud_worker.constants["BASE_REVISION"]


@pytest.mark.parametrize("stage", ["server_start", "thread_start"])
def test_server_start_failure_closes_socket_without_joining_unstarted_thread(cloud_worker, stage):
    cloud_worker.fail_at = stage
    receipt = check_archived_receipt(cloud_worker, cloud_worker.execute())
    assert receipt["status"] == "failed"
    assert receipt["stage"] == "server_start"
    assert cloud_worker.sockets[0].closed
    assert all(not thread.joined for thread in cloud_worker.threads)
    assert all(server.should_exit for server in cloud_worker.servers)
    assert len(cloud_worker.commits) == 1


def test_inference_failure_retains_partial_artifacts_and_shuts_down_server(cloud_worker):
    cloud_worker.fail_at = "benchmark"
    result = cloud_worker.execute()
    receipt = check_archived_receipt(cloud_worker, result)
    assert receipt["status"] == "failed"
    assert receipt["stage"] == "benchmark"
    assert receipt["model"]["checkpoint"] == "google/gemma-4-12B-it"
    assert cloud_worker.servers[0].should_exit
    assert cloud_worker.threads[0].joined
    assert cloud_worker.sockets[0].closed
    with zipfile.ZipFile(io.BytesIO(result["archive"])) as archive:
        assert json.loads(archive.read("benchmark/partial.json")) == {"status": "partial"}


def test_commit_failure_preserves_original_setup_error_and_returned_archive(cloud_worker):
    cloud_worker.fail_at = "helper_model_loading"
    cloud_worker.fail_commit = True
    receipt = check_archived_receipt(cloud_worker, cloud_worker.execute())
    assert receipt["stage"] == "helper_model_loading"
    assert [error["stage"] for error in receipt["failures"]] == [
        "helper_model_loading",
        "volume_commit",
    ]
    assert receipt["error"] == receipt["failures"][0]


def test_cleanup_failure_retains_successful_benchmark_summary(cloud_worker):
    cloud_worker.fail_at = "cleanup_thread"
    receipt = check_archived_receipt(cloud_worker, cloud_worker.execute())
    assert receipt["status"] == "failed"
    assert receipt["stage"] == "cleanup_thread"
    assert receipt["summary"] == {"completion_rate": 0.0, "tasks": 1}
    assert cloud_worker.sockets[0].closed
    assert len(cloud_worker.commits) == 1


def test_success_receipt_preserves_runtime_telemetry_and_shutdown(cloud_worker):
    receipt = check_archived_receipt(cloud_worker, cloud_worker.execute())
    assert receipt["status"] == receipt["stage"] == "completed"
    assert "error" not in receipt
    assert receipt["peak_cuda_allocated_bytes"] == 123
    assert receipt["modal_app_id"] == "offline-app"
    assert receipt["modal_call_id"] == "offline-call"
    assert len(receipt["versions"]) == 5
    assert cloud_worker.servers[0].should_exit
    assert cloud_worker.threads[0].joined
    assert cloud_worker.sockets[0].closed


@pytest.mark.parametrize(
    "arguments",
    [{"run": "../invalid"}, {"cell": "unknown"}, {"repeats": 11}, {"split": "training"}],
)
def test_invalid_configuration_creates_no_attempt_or_cloud_activity(cloud_worker, arguments):
    with pytest.raises(ValueError):
        cloud_worker.execute(**arguments)
    assert not cloud_worker.root.exists()
    assert cloud_worker.model_calls == []
    assert cloud_worker.commits == []


def test_previous_attempt_is_not_overwritten_or_recommitted(cloud_worker):
    prior = cloud_worker.root / "offline-test" / "native-base"
    prior.mkdir(parents=True)
    (prior / "receipt.json").write_text("previous attempt")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        cloud_worker.execute()
    assert (prior / "receipt.json").read_text() == "previous attempt"
    assert cloud_worker.model_calls == []
    assert cloud_worker.commits == []
