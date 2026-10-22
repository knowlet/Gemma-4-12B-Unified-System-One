"""Pure observer regressions: fake proc/HTTP facts, never models or Comfy writes."""

import copy
import errno
import importlib.util
import json
import os
import shutil
import time
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location("_comfy_observer_tests", HERE / "comfy_observer.py")
o = importlib.util.module_from_spec(spec)
spec.loader.exec_module(o)
BOOT = "11111111-2222-4333-8444-555555555555"
MEMORY = {"mem_total_bytes": 130595864576, "mem_available_bytes": 120000000000}


@pytest.fixture
def facts(monkeypatch):
    monkeypatch.setattr(o, "_boot", lambda: BOOT)
    monkeypatch.setattr(o, "_memory", lambda: copy.deepcopy(MEMORY))


def refused():
    raise urllib.error.URLError(ConnectionRefusedError(errno.ECONNREFUSED, "refused"))


def sample(index, *, listener=False, known=False, unknown=False):
    return {
        "started_monotonic": float(index * 2),
        "finished_monotonic": float(index * 2 + 1),
        "boot_id": BOOT,
        **MEMORY,
        "tcp": {"listeners": [{"port": 8188}] if listener else [], "inventories": []},
        "processes": {
            "known_comfy": [{"pid": 42, "classification": "comfy_runtime"}] if known else [],
            "uncertain": [{"pid": 43, "classification": "uncertain_python_cwd"}] if unknown else [],
            "scanned": 20,
        },
    }


def test_proven_absence_requires_two_separate_fresh_scans(facts, monkeypatch):
    monkeypatch.setattr(o, "_queue", refused)
    values = [sample(0), sample(1)]
    calls = []

    def scan(boot):
        calls.append(boot)
        return values[len(calls) - 1]

    monkeypatch.setattr(o, "_sample", scan)
    result = o.observe(BOOT)
    assert calls == [BOOT, BOOT]
    assert result["comfy_state"] == "absent_confirmed"
    assert result["comfy_running"] == result["comfy_pending"] == 0
    assert result["comfy_evidence"]["transport"] == "connection_refused"
    assert len(result["comfy_evidence"]["samples"]) == 2
    assert "not an HTTP" in result["comfy_evidence"]["scope"]


def eligible(pid=42):
    return {
        "pid": pid,
        "start_ticks": 123,
        "proc_owner_uid": 1001,
        "classification": "uncertain_python_cwd",
        "disappearance_resolution_eligible": True,
        "context_diagnostics": [{"cwd_failure_type": "PermissionError", "cwd_sha256": None}],
    }


@pytest.mark.parametrize("unknown_scan", [0, 1])
def test_resolved_generic_unknown_restarts_both_whole_scans(facts, monkeypatch, unknown_scan):
    monkeypatch.setattr(o, "_queue", refused)
    values = [sample(i) for i in range(unknown_scan + 1)]
    values[-1]["processes"]["uncertain"] = [eligible()]
    values += [sample(5), sample(6)]
    calls, waits = [], []

    def fresh(_boot):
        calls.append(True)
        return values.pop(0)

    def disappeared(rows):
        waits.append(rows)
        return {"disappeared": [42]}

    monkeypatch.setattr(o, "_sample", fresh)
    monkeypatch.setattr(o, "_wait_disappeared", disappeared)
    result = o.observe(BOOT)
    assert len(calls) == unknown_scan + 3 and len(waits) == 1
    assert [row["started_monotonic"] for row in result["comfy_evidence"]["samples"]] == [10, 12]
    assert result["comfy_evidence"]["natural_exit_resolution"]["disappeared"] == [42]


def test_unknown_after_one_restart_never_waits_again(facts, monkeypatch):
    monkeypatch.setattr(o, "_queue", refused)
    values = [sample(0), sample(1)]
    for value in values:
        value["processes"]["uncertain"] = [eligible()]
    waits = []
    monkeypatch.setattr(o, "_sample", lambda _boot: values.pop(0))
    monkeypatch.setattr(
        o, "_wait_disappeared", lambda rows: waits.append(rows) or {"disappeared": [42]}
    )
    with pytest.raises(o.ObservationError, match="one fresh restart"):
        o.observe(BOOT)
    assert len(waits) == 1


@pytest.mark.parametrize(
    "classification",
    [
        "uncertain_comfy_script",
        "uncertain_comfy_cwd",
        "uncertain_comfy_runtime",
        "uncertain_python",
    ],
)
def test_intrinsic_or_comfy_unknown_never_opens_pidfd(monkeypatch, classification):
    row = {**eligible(), "classification": classification}
    monkeypatch.setattr(o, "_open_pidfd", lambda _pid: pytest.fail("Intrinsic evidence waited"))
    with pytest.raises(o.ObservationError, match="cannot wait"):
        o._wait_disappeared([row])


@pytest.fixture
def fake_exit_proof(monkeypatch):
    opened, closed = [], []
    monkeypatch.setattr(o, "_open_pidfd", lambda pid: opened.append(pid) or (pid + 100))
    monkeypatch.setattr(o.os, "close", lambda fd: closed.append(fd))
    monkeypatch.setattr(o, "RESOLUTION_SECONDS", 0.01)
    monkeypatch.setattr(
        o, "_process_attempt", lambda _path: ({"classification": "python_external_script"}, None)
    )
    return opened, closed


def test_alive_external_reclassification_cannot_clear_prior_unknown(monkeypatch, fake_exit_proof):
    monkeypatch.setattr(o, "_stat", lambda _path: ("python", "S", 123))
    with pytest.raises(o.ObservationError, match="remains after"):
        o._wait_disappeared([eligible()])
    assert fake_exit_proof == ([42], [142])


def test_alive_reused_identity_refuses_instead_of_reclassifying(monkeypatch, fake_exit_proof):
    monkeypatch.setattr(o, "_stat", lambda _path: ("python", "S", 124))
    with pytest.raises(o.ObservationError, match="identity changed"):
        o._wait_disappeared([eligible()])
    assert fake_exit_proof == ([42], [142])


@pytest.mark.parametrize("comfy_source", ["comm", "context"])
def test_comfy_seen_during_wait_is_sticky_immediately(monkeypatch, fake_exit_proof, comfy_source):
    monkeypatch.setattr(
        o, "_stat", lambda _path: ("comfyui" if comfy_source == "comm" else "python", "S", 123)
    )
    monkeypatch.setattr(
        o, "_process_attempt", lambda _path: ({"classification": "uncertain_comfy_cwd"}, None)
    )
    with pytest.raises(o.ObservationError, match="Comfy"):
        o._wait_disappeared([eligible()])
    assert fake_exit_proof == ([42], [142])


@pytest.mark.parametrize("ready", [False, True])
def test_proc_absence_requires_pidfd_readiness(monkeypatch, fake_exit_proof, ready):
    def absent(_path):
        raise FileNotFoundError("owned fake PID gone")

    class Poll:
        def register(self, *_args):
            pass

        def poll(self, _timeout):
            return [(142, o.select.POLLIN)] if ready else []

    monkeypatch.setattr(o, "_stat", absent)
    monkeypatch.setattr(o.select, "poll", Poll)
    if ready:
        assert o._wait_disappeared([eligible()])["disappeared"] == [42]
    else:
        with pytest.raises(o.ObservationError, match="remains after"):
            o._wait_disappeared([eligible()])
    assert fake_exit_proof == ([42], [142])


def test_pidfd_esrch_also_requires_fresh_proc_absence(monkeypatch):
    def gone(_pid):
        raise ProcessLookupError("owned fake PID gone")

    monkeypatch.setattr(o, "_open_pidfd", gone)
    monkeypatch.setattr(o, "_stat", lambda _path: ("python", "S", 123))
    with pytest.raises(o.ObservationError, match="remains visible"):
        o._wait_disappeared([eligible()])


def test_all_offenders_share_one_resolution_budget(monkeypatch, fake_exit_proof):
    monkeypatch.setattr(o, "_stat", lambda _path: ("python", "S", 123))
    started = time.monotonic()
    with pytest.raises(o.ObservationError, match="remains after"):
        o._wait_disappeared([eligible(42), eligible(43)])
    assert time.monotonic() - started < 0.1
    assert fake_exit_proof == ([42, 43], [142, 143])


def test_whole_observation_deadline_prevents_late_scan_admission(facts, monkeypatch):
    monkeypatch.setattr(o, "_queue", refused)
    monkeypatch.setattr(o, "OBSERVE_SECONDS", 0.01)
    calls = []

    def late(_boot):
        calls.append(True)
        time.sleep(0.02)
        return sample(0)

    monkeypatch.setattr(o, "_sample", late)
    with pytest.raises(o.ObservationError, match="Whole Comfy observation deadline"):
        o.observe(BOOT)
    assert calls == [True]


@pytest.mark.parametrize("kind", ["listener", "known", "unknown"])
@pytest.mark.parametrize("changed_scan", [0, 1])
def test_either_scan_with_listener_runtime_or_uncertainty_refuses(
    facts, monkeypatch, kind, changed_scan
):
    monkeypatch.setattr(o, "_queue", refused)
    values = [sample(i, **{kind: i == changed_scan}) for i in range(2)]
    monkeypatch.setattr(o, "_sample", lambda _boot: values.pop(0))
    with pytest.raises(o.ObservationError) as error:
        o.observe(BOOT)
    assert error.value.evidence


@pytest.mark.parametrize("change", ["total", "freshness", "boot"])
def test_second_scan_identity_or_freshness_drift_refuses(facts, monkeypatch, change):
    monkeypatch.setattr(o, "_queue", refused)
    values = [sample(0), sample(1)]
    if change == "total":
        values[1]["mem_total_bytes"] += 4096
    elif change == "freshness":
        values[1]["started_monotonic"] = values[0]["finished_monotonic"] - 0.1
    else:
        values[1]["boot_id"] = "different"
    monkeypatch.setattr(o, "_sample", lambda _boot: values.pop(0))
    with pytest.raises(o.ObservationError):
        o.observe(BOOT)


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("timeout"),
        urllib.error.URLError(TimeoutError()),
        urllib.error.URLError(OSError(errno.EHOSTUNREACH, "unreachable")),
        urllib.error.HTTPError(o.QUEUE_URL, 403, "denied", {}, None),
        ValueError("malformed JSON"),
        RuntimeError("contains errno111 but is not refusal"),
    ],
)
def test_only_actual_connection_refusal_may_inspect_absence(facts, monkeypatch, error):
    def failed():
        raise error

    monkeypatch.setattr(o, "_queue", failed)
    monkeypatch.setattr(o, "_sample", lambda _boot: pytest.fail("Absence bypass attempted"))
    with pytest.raises(o.ObservationError, match="without proven connection refusal"):
        o.observe(BOOT)


@pytest.mark.parametrize(
    "running,pending,state", [(0, 0, "http_idle"), (1, 0, "http_busy"), (0, 2, "http_busy")]
)
def test_typed_http_queue_preserves_actual_counts(facts, monkeypatch, running, pending, state):
    payload = {"queue_running": [None] * running, "queue_pending": [None] * pending}

    class Reply:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def read(self, _limit):
            return json.dumps(payload).encode()

    monkeypatch.setattr(o.urllib.request, "urlopen", lambda url, timeout: Reply())
    monkeypatch.setattr(o, "_sample", lambda _boot: pytest.fail("HTTP queue did not need absence"))
    result = o.observe(BOOT)
    assert result["comfy_state"] == state
    assert (result["comfy_running"], result["comfy_pending"]) == (running, pending)
    assert result["comfy_evidence"]["transport"] == "HTTP"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"queue_running": False, "queue_pending": []},
        {"queue_running": [], "queue_pending": None},
    ],
)
def test_malformed_http_queue_is_not_absence(facts, monkeypatch, payload):
    reply = SimpleNamespace(status=200, read=lambda _n: json.dumps(payload).encode())

    class Context:
        def __enter__(self):
            return reply

        def __exit__(self, *_args):
            pass

    monkeypatch.setattr(o.urllib.request, "urlopen", lambda *_a, **_k: Context())
    monkeypatch.setattr(o, "_sample", lambda _b: pytest.fail("Malformed queue used absence"))
    with pytest.raises(o.ObservationError):
        o.observe(BOOT)


@pytest.mark.parametrize(
    "argv,cwd,expected",
    [
        (["bash", "-c", "python /home/albert_hsieh/ComfyUI/main.py"], "/tmp", "non_python"),
        (
            [
                "/home/albert_hsieh/ComfyUI/.venv/bin/python",
                "-B",
                "-s",
                "/home/albert_hsieh/Gemma-4-12B-Unified-System-One/scripts/fit.py",
            ],
            "/home/albert_hsieh/Gemma-4-12B-Unified-System-One",
            "python_external_script",
        ),
        (["python", "main.py"], str(o.COMFY_ROOT), "comfy_runtime"),
        (["python3.13", "-u", "main.py"], str(o.COMFY_ROOT), "comfy_runtime"),
        (["python", "-m", "comfy"], "/tmp", "comfy_runtime"),
        (["python", "-m", "comfy.cli"], None, "comfy_runtime"),
        (["python", str(o.COMFY_ROOT / "main.py")], None, "comfy_runtime"),
        (["python", "main.py"], None, "uncertain_python_cwd"),
        (["python", "/tmp/launcher.py"], None, "uncertain_python_cwd"),
        (["python", "/tmp/launcher.py"], str(o.COMFY_ROOT), "uncertain_comfy_cwd"),
        (["python", "-c", "arbitrary code"], str(o.COMFY_ROOT), "uncertain_comfy_cwd"),
        (["python", "-B", "-"], "/tmp/project", "python_external_invocation"),
    ],
)
def test_structural_invocation_classification_ignores_shell_payload_and_venv_alone(
    argv, cwd, expected
):
    assert o.classify_command(argv, cwd) == expected


def proc_entry(root, pid, argv, cwd=None, *, comm="python", ticks=123):
    path = root / str(pid)
    path.mkdir()
    fields = ["S"] + ["0"] * 18 + [str(ticks)] + ["0"] * 4
    (path / "stat").write_text(f"{pid} ({comm}) " + " ".join(fields))
    (path / "cmdline").write_bytes(b"\0".join(v.encode() for v in argv) + b"\0")
    if cwd is not None:
        (path / "cwd").symlink_to(cwd)
    return path


def test_real_proc_file_scan_classifies_comfy_and_excludes_long_shell_payload(tmp_path):
    proc_entry(tmp_path, 1, ["bash", "-c", "ComfyUI main.py " * 5000], "/tmp", comm="bash")
    proc_entry(tmp_path, 2, ["python", str(o.COMFY_ROOT / "main.py")], str(o.COMFY_ROOT))
    proc_entry(tmp_path, 3, ["python", "/tmp/project/fit.py"], "/tmp/project")
    result = o.scan_processes(tmp_path)
    assert result["scanned"] == 3
    assert [x["pid"] for x in result["known_comfy"]] == [2]
    assert result["uncertain"] == []
    assert "argv" not in json.dumps(result) and "arbitrary" not in json.dumps(result)


def test_unreadable_relative_python_cwd_is_uncertain(tmp_path, monkeypatch):
    path = proc_entry(tmp_path, 4, ["python", "main.py"])
    original = o.os.readlink

    def denied(value):
        if Path(value) == path / "cwd":
            raise PermissionError("fake proc denied")
        return original(value)

    monkeypatch.setattr(o.os, "readlink", denied)
    result = o.scan_processes(tmp_path)
    assert result["uncertain"][0]["classification"] == "uncertain_python_cwd"
    row = result["uncertain"][0]
    assert row["proc_owner_uid"] == os.getuid()
    assert row["disappearance_resolution_eligible"] is True
    assert (
        row["context_diagnostics"]
        == [{"cwd_failure_type": "PermissionError", "cwd_sha256": None}] * 2
    )


@pytest.mark.parametrize("deleted", [False, True])
def test_later_comfy_cwd_dominates_prior_generic_uncertainty(tmp_path, monkeypatch, deleted):
    proc_entry(tmp_path, 99, ["python", "/tmp/project/worker.py"], "/tmp/project")
    count = 0

    def changed(_path):
        nonlocal count
        count += 1
        if count == 1:
            raise PermissionError("owned fake first cwd denied")
        return str(o.COMFY_ROOT) + (" (deleted)" if deleted else "")

    monkeypatch.setattr(o.os, "readlink", changed)
    row = o.scan_processes(tmp_path)["uncertain"][0]
    assert row["classification"] == "uncertain_comfy_cwd"
    assert row["disappearance_resolution_eligible"] is False
    assert str(o.COMFY_ROOT) not in json.dumps(row)


def test_missing_cwd_for_still_live_python_is_not_a_vanished_process(tmp_path):
    proc_entry(tmp_path, 6, ["python", "main.py"])
    result = o.scan_processes(tmp_path)
    assert result["uncertain"][0]["classification"] == "uncertain_process_read"
    assert result["vanished_sample"] == []


@pytest.mark.parametrize("cwd", [None, str(o.COMFY_ROOT)])
def test_absolute_external_wrapper_requires_known_external_cwd(tmp_path, cwd):
    proc_entry(tmp_path, 8, ["python", "/tmp/launcher.py"], cwd)
    result = o.scan_processes(tmp_path)
    assert result["known_comfy"] == []
    assert [row["pid"] for row in result["uncertain"]] == [8]
    assert result["vanished_sample"] == []


@pytest.mark.parametrize("context", [1, 2])
@pytest.mark.parametrize("error_type", [FileNotFoundError, ProcessLookupError])
def test_missing_cwd_records_vanished_only_after_fresh_stat_proves_absence(
    tmp_path, monkeypatch, context, error_type
):
    path = proc_entry(tmp_path, 50, ["python", "/tmp/project/worker.py"], "/tmp/project")
    original, count = o.os.readlink, 0

    def vanished(selected):
        nonlocal count
        count += 1
        if count == context:
            shutil.rmtree(path)
            raise error_type("owned fake proc vanished before cwd")
        return original(selected)

    monkeypatch.setattr(o.os, "readlink", vanished)
    result = o.scan_processes(tmp_path)
    assert count == context and result["identity_retries"] == 0
    assert result["uncertain"] == result["known_comfy"] == []
    assert result["vanished_sample"] == [{"pid": 50, "start_ticks": 123}]


@pytest.mark.parametrize("context", [1, 2])
@pytest.mark.parametrize("error_type", [FileNotFoundError, ProcessLookupError])
def test_missing_cwd_on_still_live_pid_remains_unknown(tmp_path, monkeypatch, context, error_type):
    proc_entry(tmp_path, 51, ["python", "/tmp/project/worker.py"], "/tmp/project")
    original, count = o.os.readlink, 0

    def missing(selected):
        nonlocal count
        count += 1
        if count == context:
            raise error_type("owned fake live cwd unavailable")
        return original(selected)

    monkeypatch.setattr(o.os, "readlink", missing)
    result = o.scan_processes(tmp_path)
    assert count == context and result["identity_retries"] == 0
    assert result["uncertain"][0]["classification"] == "uncertain_process_read"
    assert result["vanished_sample"] == []


@pytest.mark.parametrize("kind", ["permission", "deleted", "nonabsolute"])
def test_cwd_uncertainty_is_sticky_even_if_process_vanishes(tmp_path, monkeypatch, kind):
    path = proc_entry(tmp_path, 52, ["python", "/tmp/project/worker.py"], "/tmp/project")

    def uncertain(_selected):
        shutil.rmtree(path)
        if kind == "permission":
            raise PermissionError("owned fake denied cwd")
        return "/tmp/project (deleted)" if kind == "deleted" else "relative/cwd"

    monkeypatch.setattr(o.os, "readlink", uncertain)
    result = o.scan_processes(tmp_path)
    assert result["identity_retries"] == 0
    assert result["uncertain"][0]["classification"] == "uncertain_python_cwd"
    assert result["vanished_sample"] == []


@pytest.mark.parametrize("context", [1, 2])
@pytest.mark.parametrize("kind", ["absolute", "module", "executable"])
def test_definite_comfy_argv_is_latched_before_any_cwd_read_can_vanish(
    tmp_path, monkeypatch, context, kind
):
    known = {
        "absolute": ["python", str(o.COMFY_ROOT / "main.py")],
        "module": ["python", "-m", "comfy.cli"],
        "executable": ["comfyui"],
    }[kind]
    initial = known if context == 1 else ["python", "/tmp/project/worker.py"]
    path = proc_entry(tmp_path, 53, initial, "/tmp/project")
    original_command, original_link, commands, links = o._command, o.os.readlink, 0, 0

    def vanished(selected, comm):
        nonlocal commands
        value = original_command(selected, comm)
        commands += 1
        if commands == context:
            if context == 2:
                value = known, value[1]
            shutil.rmtree(path)
        return value

    def checked(selected):
        nonlocal links
        links += 1
        return original_link(selected)

    monkeypatch.setattr(o, "_command", vanished)
    monkeypatch.setattr(o.os, "readlink", checked)
    result = o.scan_processes(tmp_path)
    assert commands == context and links == context - 1
    assert result["known_comfy"][0]["classification"] == "comfy_runtime"
    assert result["vanished_sample"] == [] and result["identity_retries"] == 0


@pytest.mark.parametrize("context", [1, 2])
def test_comfy_comm_remains_sticky_when_cwd_disappears(tmp_path, monkeypatch, context):
    path = proc_entry(tmp_path, 54, ["python", "/tmp/project/worker.py"], "/tmp/project")
    original_stat, original_link, stats, links = o._stat, o.os.readlink, 0, 0

    def comm_changed(selected):
        nonlocal stats
        value = original_stat(selected)
        stats += 1
        return ("comfyui", value[1], value[2]) if stats == context else value

    def vanished(selected):
        nonlocal links
        links += 1
        if links == context:
            shutil.rmtree(path)
            raise FileNotFoundError("owned fake Comfy comm cwd vanished")
        return original_link(selected)

    monkeypatch.setattr(o, "_stat", comm_changed)
    monkeypatch.setattr(o.os, "readlink", vanished)
    result = o.scan_processes(tmp_path)
    assert links == context and result["identity_retries"] == 0
    assert result["uncertain"][0]["classification"] == "uncertain_comfy_runtime"
    assert result["vanished_sample"] == []


@pytest.mark.parametrize("context", [1, 2])
@pytest.mark.parametrize(
    "argv,expected",
    [
        (["python", str(o.COMFY_ROOT / "other.py")], "uncertain_comfy_script"),
        (["python", "--unknown-launcher-option"], "uncertain_python"),
        (["python", "-m"], "uncertain_python"),
    ],
)
def test_intrinsic_uncertain_argv_survives_disappearance_before_cwd(
    tmp_path, monkeypatch, context, argv, expected
):
    initial = argv if context == 1 else ["python", "/tmp/project/worker.py"]
    path = proc_entry(tmp_path, 55, initial, "/tmp/project")
    original, count = o._command, 0

    def vanished(selected, comm):
        nonlocal count
        value = original(selected, comm)
        count += 1
        if count == context:
            if context == 2:
                value = argv, value[1]
            shutil.rmtree(path)
        return value

    monkeypatch.setattr(o, "_command", vanished)
    result = o.scan_processes(tmp_path)
    assert count == context and result["identity_retries"] == 0
    assert result["uncertain"][0]["classification"] == expected
    assert result["vanished_sample"] == []


def test_missing_cmdline_for_still_live_process_is_not_proven_absent(tmp_path):
    path = proc_entry(tmp_path, 7, ["python", "main.py"])
    (path / "cmdline").unlink()
    result = o.scan_processes(tmp_path)
    assert result["uncertain"][0]["classification"] == "uncertain_process_read"
    assert result["vanished_sample"] == []


@pytest.mark.parametrize("field", ["comm", "ticks"])
def test_transient_process_identity_change_is_rechecked_with_a_stable_same_pid(
    tmp_path, monkeypatch, field
):
    proc_entry(tmp_path, 8, ["bash", "-c", "not a runtime"], "/tmp", comm="bash")
    original, count = o._stat, 0

    def changed(path):
        nonlocal count
        value = original(path)
        count += 1
        if count >= 2:
            return (
                "python" if field == "comm" else value[0],
                value[1],
                value[2] + (field == "ticks"),
            )
        return value

    monkeypatch.setattr(o, "_stat", changed)
    result = o.scan_processes(tmp_path)
    assert result["uncertain"] == result["known_comfy"] == []
    assert count == 6 and result["identity_retries"] == 1
    assert result["retry_samples"][0]["observation_attempts"] == 2
    assert result["retry_samples"][0]["transitions"][0]["reason"] == "identity_changed"


@pytest.mark.parametrize("field", ["comm", "ticks"])
def test_persistent_identity_changes_remain_unknown_after_two_attempts(
    tmp_path, monkeypatch, field
):
    proc_entry(tmp_path, 9, ["bash", "-c", "not a runtime"], "/tmp", comm="bash")
    original, count = o._stat, 0

    def changed(path):
        nonlocal count
        comm, state, ticks = original(path)
        count += 1
        return (
            "python" if field == "comm" and count % 2 == 0 else comm,
            state,
            ticks + count if field == "ticks" else ticks,
        )

    monkeypatch.setattr(o, "_stat", changed)
    result = o.scan_processes(tmp_path)
    assert count == 6 and result["identity_retries"] == 1
    assert result["uncertain"][0]["classification"] == "uncertain_process_identity_changed"
    assert len(result["retry_samples"][0]["transitions"]) == 2


@pytest.mark.parametrize("persistent", [False, True])
def test_argv_change_with_same_pid_comm_and_start_time_is_bounded(
    tmp_path, monkeypatch, persistent
):
    path = proc_entry(tmp_path, 10, ["python", "/tmp/project/old.py"], "/tmp/project")
    original, count = o._command, 0

    def changed(selected, comm):
        nonlocal count
        value = original(selected, comm)
        count += 1
        if count == 1 or persistent:
            target = "new.py" if count % 2 else "old.py"
            (path / "cmdline").write_bytes(f"python\0/tmp/project/{target}\0".encode())
        return value

    monkeypatch.setattr(o, "_command", changed)
    result = o.scan_processes(tmp_path)
    assert count == 4 and result["identity_retries"] == 1
    if persistent:
        assert result["uncertain"][0]["classification"] == "uncertain_process_argv_changed"
    else:
        assert result["uncertain"] == result["known_comfy"] == []
    assert result["retry_samples"][0]["transitions"][0]["reason"] == "argv_changed"
    saved = json.dumps(result)
    assert "old.py" not in saved and "new.py" not in saved


@pytest.mark.parametrize("comfy_at", ["before", "after"])
def test_either_observed_comfy_invocation_is_latched_even_if_it_vanishes(
    tmp_path, monkeypatch, comfy_at
):
    argv = (
        ["python", str(o.COMFY_ROOT / "main.py")]
        if comfy_at == "before"
        else ["python", "/tmp/project/worker.py"]
    )
    path = proc_entry(tmp_path, 11, argv, "/tmp/project")
    original, count = o._command, 0

    def changed(selected, comm):
        nonlocal count
        value = original(selected, comm)
        count += 1
        if count == 1:
            next_script = (
                "/tmp/project/worker.py" if comfy_at == "before" else str(o.COMFY_ROOT / "main.py")
            )
            (path / "cmdline").write_bytes(f"python\0{next_script}\0".encode())
        elif count == 2:
            shutil.rmtree(path)
        return value

    monkeypatch.setattr(o, "_command", changed)
    result = o.scan_processes(tmp_path)
    assert count == 2 and result["identity_retries"] == 0
    assert result["known_comfy"][0]["pid"] == 11
    assert result["vanished_sample"] == []


def test_new_comfy_comm_is_not_retried_until_it_disappears(tmp_path, monkeypatch):
    proc_entry(tmp_path, 12, ["bash", "-c", "not a runtime"], "/tmp", comm="bash")
    original, count = o._stat, 0

    def changed(path):
        nonlocal count
        comm, state, ticks = original(path)
        count += 1
        return ("comfyui" if count == 2 else comm, state, ticks)

    monkeypatch.setattr(o, "_stat", changed)
    result = o.scan_processes(tmp_path)
    assert count == 3 and result["identity_retries"] == 0
    assert result["uncertain"][0]["classification"] == "uncertain_comfy_runtime"


@pytest.mark.parametrize("comfy_stat", [1, 2])
def test_comfy_comm_is_latched_before_cmdline_can_vanish(tmp_path, monkeypatch, comfy_stat):
    path = proc_entry(tmp_path, 15, ["bash", "-c", "not a runtime"], "/tmp", comm="bash")
    original_stat, original_command, count = o._stat, o._command, 0

    def changed(selected):
        nonlocal count
        value = original_stat(selected)
        count += 1
        return ("comfyui", value[1], value[2]) if count == comfy_stat else value

    def vanished(selected, comm):
        if count == comfy_stat:
            shutil.rmtree(path)
            raise FileNotFoundError("owned fake proc vanished")
        return original_command(selected, comm)

    monkeypatch.setattr(o, "_stat", changed)
    monkeypatch.setattr(o, "_command", vanished)
    result = o.scan_processes(tmp_path)
    assert count == comfy_stat and result["identity_retries"] == 0
    assert result["uncertain"][0]["classification"] == "uncertain_comfy_runtime"
    assert result["vanished_sample"] == []


def test_final_stat_comfy_comm_is_latched(tmp_path, monkeypatch):
    proc_entry(tmp_path, 16, ["bash", "-c", "not a runtime"], "/tmp", comm="bash")
    original, count = o._stat, 0

    def changed(path):
        nonlocal count
        value = original(path)
        count += 1
        return ("comfyui", value[1], value[2]) if count == 3 else value

    monkeypatch.setattr(o, "_stat", changed)
    result = o.scan_processes(tmp_path)
    assert count == 3 and result["identity_retries"] == 0
    assert result["uncertain"][0]["classification"] == "uncertain_comfy_runtime"


@pytest.mark.parametrize("field", ["comm", "ticks"])
@pytest.mark.parametrize("persistent", [False, True])
def test_final_stat_bounds_second_context_read(tmp_path, monkeypatch, field, persistent):
    proc_entry(tmp_path, 17, ["bash", "-c", "not a runtime"], "/tmp", comm="bash")
    original, count = o._stat, 0

    def changed(path):
        nonlocal count
        comm, state, ticks = original(path)
        count += 1
        drift = count >= 3 if not persistent else count % 3 == 0
        return (
            "worker" if drift and field == "comm" else comm,
            state,
            ticks + (drift and field == "ticks"),
        )

    monkeypatch.setattr(o, "_stat", changed)
    result = o.scan_processes(tmp_path)
    assert count == 6 and result["identity_retries"] == 1
    if persistent:
        assert result["uncertain"][0]["classification"] == "uncertain_process_identity_changed"
    else:
        assert result["uncertain"] == result["known_comfy"] == []
    transition = result["retry_samples"][0]["transitions"][0]
    assert transition["reason"] == "identity_changed" and "final" in transition


def test_known_comfy_context_survives_second_command_permission_failure(tmp_path, monkeypatch):
    proc_entry(tmp_path, 18, ["python", "-m", "comfy"], "/tmp")
    original, count = o._command, 0

    def denied(selected, comm):
        nonlocal count
        count += 1
        if count == 2:
            raise PermissionError("fake second command denied")
        return original(selected, comm)

    monkeypatch.setattr(o, "_command", denied)
    result = o.scan_processes(tmp_path)
    assert count == 2 and result["identity_retries"] == 0
    assert result["known_comfy"][0]["pid"] == 18


def test_second_attempt_read_failure_is_not_retried_or_erased(tmp_path, monkeypatch):
    proc_entry(tmp_path, 19, ["bash", "-c", "not a runtime"], "/tmp", comm="bash")
    original_stat, original_command, count = o._stat, o._command, 0

    def changed(path):
        nonlocal count
        value = original_stat(path)
        count += 1
        return ("worker", value[1], value[2]) if count >= 3 else value

    def denied(selected, comm):
        if count >= 4:
            raise PermissionError("fake retry command denied")
        return original_command(selected, comm)

    monkeypatch.setattr(o, "_stat", changed)
    monkeypatch.setattr(o, "_command", denied)
    result = o.scan_processes(tmp_path)
    assert count == 4 and result["identity_retries"] == 1
    assert result["uncertain"][0]["classification"] == "uncertain_process_read"


def test_permission_failure_is_not_a_retryable_exec_race(tmp_path, monkeypatch):
    proc_entry(tmp_path, 13, ["python", "/tmp/project/worker.py"], "/tmp/project")
    calls = []

    def denied(*_args):
        calls.append(True)
        raise PermissionError("fake command denied")

    monkeypatch.setattr(o, "_command", denied)
    result = o.scan_processes(tmp_path)
    assert calls == [True] and result["identity_retries"] == 0
    assert result["uncertain"][0]["classification"] == "uncertain_process_read"


def test_unreadable_cwd_in_either_observation_is_not_retried_away(tmp_path, monkeypatch):
    proc_entry(tmp_path, 14, ["python", "/tmp/project/worker.py"], "/tmp/project")
    original, count = o.os.readlink, 0

    def denied_once(path):
        nonlocal count
        count += 1
        if count == 1:
            raise PermissionError("fake first cwd denied")
        return original(path)

    monkeypatch.setattr(o.os, "readlink", denied_once)
    result = o.scan_processes(tmp_path)
    assert count == 2 and result["identity_retries"] == 0
    assert result["uncertain"][0]["classification"] == "uncertain_python_cwd"


def test_process_vanishing_is_recorded_without_claiming_a_live_runtime(tmp_path, monkeypatch):
    proc_entry(tmp_path, 5, ["python", "main.py"])
    original = o._stat

    def vanished(path):
        if Path(path).parent.name == "5":
            raise FileNotFoundError("fake vanished PID")
        return original(path)

    monkeypatch.setattr(o, "_stat", vanished)
    result = o.scan_processes(tmp_path)
    assert result["known_comfy"] == result["uncertain"] == []
    assert result["vanished_sample"] == [{"pid": 5, "start_ticks": None}]


@pytest.mark.parametrize(
    "family,address", [("tcp", "0100007F"), ("tcp6", "00000000000000000000000001000000")]
)
def test_any_ipv4_or_ipv6_8188_listener_prevents_absence(tmp_path, family, address):
    net = tmp_path / "net"
    net.mkdir()
    for name in ("tcp", "tcp6"):
        (net / name).write_text("sl local_address rem_address st\n")
    with (net / family).open("a") as stream:
        stream.write(f"0: {address}:1FFC 00000000:0000 0A\n")
    result = o.scan_listeners(tmp_path)
    assert result["listeners"] == [{"family": family, "row": 0, "port": 8188, "state": "LISTEN"}]
