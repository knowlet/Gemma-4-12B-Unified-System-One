"""Lifecycle tests use real small owned children; ML components are explicit fakes."""

import importlib.util
import json
import signal
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

SPEC = importlib.util.spec_from_file_location(
    "gx10_warm_probe_tested", Path(__file__).with_name("gx10_warm_probe.py")
)
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


def test_normal_child_wholly_exits_and_restores_signal_handler(tmp_path):
    handler = signal.getsignal(signal.SIGTERM)
    state, error = P.supervise(
        [sys.executable, "-c", "print('owned-only')"], tmp_path, lambda: None, wall_seconds=5
    )
    assert error is None
    assert state["closure"]["returncode"] == 0
    assert state["whole_worker_exit_confirmed"] is True
    assert P.proc_identity(state["owner"]["pid"]) is None
    assert signal.getsignal(signal.SIGTERM) == handler
    assert (tmp_path / "stdout.log").read_text() == "owned-only\n"


def test_initial_resource_refusal_spawns_nothing(tmp_path):
    def refuse():
        raise ValueError("full original memory floor")

    state, error = P.supervise(
        [sys.executable, "-c", "raise AssertionError('must not run')"], tmp_path, refuse
    )
    assert isinstance(error, ValueError)
    assert "owner" not in state
    assert state["whole_worker_exit_confirmed"] is False


def test_resource_loss_closes_only_owned_group(tmp_path):
    calls = 0

    def observe():
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise ValueError("memory floor lost")

    state, error = P.supervise(
        [sys.executable, "-c", "import time;time.sleep(60)"], tmp_path, observe
    )
    assert isinstance(error, ValueError)
    assert state["whole_worker_exit_confirmed"] is True
    assert state["closure"]["remaining"] == []
    assert state["closure"]["signals"] == [signal.SIGTERM]


def test_wall_timeout_kills_term_resistant_owned_child(tmp_path):
    state, error = P.supervise(
        [
            sys.executable,
            "-c",
            "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)",
        ],
        tmp_path,
        lambda: None,
        wall_seconds=0.5,
    )
    assert isinstance(error, ValueError)
    assert "wall limit" in str(error)
    assert state["whole_worker_exit_confirmed"] is True
    assert state["closure"]["signals"] == [signal.SIGTERM, signal.SIGKILL]
    assert state["closure"]["returncode"] == -signal.SIGKILL


def test_stdout_is_bounded_and_nonzero_exit_preserved(tmp_path):
    state, error = P.supervise(
        [sys.executable, "-c", "import os;os.write(1,b'x'*(8*1024**2));os.write(1,b'x')"],
        tmp_path,
        lambda: None,
        wall_seconds=5,
    )
    assert error is None
    assert (tmp_path / "stdout.log").stat().st_size <= P.LOG_BYTES
    assert state["closure"]["returncode"] != 0
    assert state["whole_worker_exit_confirmed"] is True


def test_warmup_primary_error_survives_backend_close_error(tmp_path, monkeypatch):
    (tmp_path / "worker").mkdir()
    primary = ValueError("publisher warmup primary")
    terminal = {}

    class Gates:
        def __init__(self, *_):
            self.torch = None

        def check(self, *_):
            return {}

    class Backend:
        def __init__(self, *_args, **_kwargs):
            self.runtime = object()

        def close(self):
            raise RuntimeError("secondary backend close")

    def warmup(*_args):
        raise primary

    def persist(path, state, **_kwargs):
        terminal.update(json.loads(json.dumps(state)))

    def error_info(exc):
        return {
            "type": type(exc).__name__,
            "message": str(exc),
            "notes": getattr(exc, "__notes__", []),
        }

    identities = {"d3-edge": {}}
    core = NS(Gates=Gates, authenticate_cache=lambda *_args: {"files": {"d3_runtime.py": {}}})
    worker = NS(
        CORE=core,
        matched_context_spec=lambda spec: spec,
        publisher_warmup=warmup,
        error_info=error_info,
        persist=persist,
    )
    cuda = NS(is_available=lambda: True, device_count=lambda: 1, synchronize=lambda *_: None)
    deps = NS(
        torch=NS(cuda=cuda),
        checkpoints=identities,
        backend=Backend,
        spec=lambda *_a, **_k: object(),
    )
    base = NS(
        UUIDGates=lambda gates, *_: gates,
        offline_environment=lambda *_: [],
        dependencies=lambda _: deps,
        geometry=lambda *_: {},
    )
    plan = {
        "output": str(tmp_path),
        "model": "d3-edge",
        "boot_id": "fake-boot",
        "memtotal": 1,
        "gpu_uuid": "fake-gpu",
        "nonce": "fake-owned",
    }
    monkeypatch.setattr(P, "read_plan", lambda *_: (base, worker, None, identities, plan))
    with pytest.raises(ValueError, match="publisher warmup primary") as caught:
        P.worker_main(Path("synthetic-plan"), "synthetic-sha")
    assert caught.value is primary
    assert terminal["error"]["message"] == "publisher warmup primary"
    assert terminal["secondary_errors"][0]["stage"] == "backend_close"
    assert "secondary backend close" in str(terminal["error"]["notes"])
    assert terminal["warmup_complete"] is False
    assert terminal["matched_quality_qualified"] is False


def test_reused_leader_pid_never_receives_group_signal(monkeypatch):
    owner = {"pid": 12345, "pgid": 12345, "sid": 12345, "start_ticks": 100}
    replacement = {**owner, "start_ticks": 101, "state": "S"}
    monkeypatch.setattr(P, "proc_identity", lambda _: replacement)
    sent = []
    monkeypatch.setattr(P.os, "killpg", lambda *args: sent.append(args))
    with pytest.raises(ValueError, match="leader PID identity changed"):
        P.close_group(NS(), owner)
    assert sent == []


def test_closed_evidence_failure_preserves_original_primary():
    primary = ValueError("original monitor failure")

    def fail(*_args, **_kwargs):
        raise OSError("secondary closed write")

    with pytest.raises(ValueError) as caught:
        P.persist_closed(NS(persist=fail), Path("synthetic-closed"), {}, primary)
    assert caught.value is primary
    assert any("secondary closed write" in note for note in primary.__notes__)


def test_inherited_group_descendant_is_also_closed(tmp_path):
    marker = tmp_path / "grandchild-pid"
    code = (
        "import signal,subprocess,sys,time;from pathlib import Path;"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
        "signal.signal(signal.SIGTERM,lambda *_:(child.wait(timeout=2),sys.exit(0)));"
        "Path(sys.argv[1]).write_text(str(child.pid));time.sleep(60)"
    )

    def observe():
        if marker.exists():
            raise ValueError("close owned inherited group")

    state, error = P.supervise(
        [sys.executable, "-c", code, str(marker)], tmp_path, observe, wall_seconds=5
    )
    assert isinstance(error, ValueError)
    assert state["whole_worker_exit_confirmed"] is True
    assert state["closure"]["remaining"] == []
    assert P.proc_identity(int(marker.read_text())) is None
