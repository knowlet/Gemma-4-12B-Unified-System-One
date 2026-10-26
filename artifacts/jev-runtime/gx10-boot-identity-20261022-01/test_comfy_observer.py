"""Pure observer regressions: fake proc/HTTP facts, never models or Comfy writes."""

import copy
import errno
import importlib.util
import json
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


def test_missing_cwd_for_still_live_python_is_not_a_vanished_process(tmp_path):
    proc_entry(tmp_path, 6, ["python", "main.py"])
    result = o.scan_processes(tmp_path)
    assert result["uncertain"][0]["classification"] == "uncertain_python_cwd"
    assert result["vanished_sample"] == []


@pytest.mark.parametrize("cwd", [None, str(o.COMFY_ROOT)])
def test_absolute_external_wrapper_requires_known_external_cwd(tmp_path, cwd):
    proc_entry(tmp_path, 8, ["python", "/tmp/launcher.py"], cwd)
    result = o.scan_processes(tmp_path)
    assert result["known_comfy"] == []
    assert [row["pid"] for row in result["uncertain"]] == [8]
    assert result["vanished_sample"] == []


def test_missing_cmdline_for_still_live_process_is_not_proven_absent(tmp_path):
    path = proc_entry(tmp_path, 7, ["python", "main.py"])
    (path / "cmdline").unlink()
    result = o.scan_processes(tmp_path)
    assert result["uncertain"][0]["classification"] == "uncertain_process_read"
    assert result["vanished_sample"] == []


@pytest.mark.parametrize("field", ["comm", "ticks"])
def test_process_identity_changes_during_a_scan_are_uncertain(tmp_path, monkeypatch, field):
    proc_entry(tmp_path, 8, ["bash", "-c", "not a runtime"], "/tmp", comm="bash")
    original, count = o._stat, 0

    def changed(path):
        nonlocal count
        value = original(path)
        count += 1
        if count == 2:
            return (
                "python" if field == "comm" else value[0],
                value[1],
                value[2] + (field == "ticks"),
            )
        return value

    monkeypatch.setattr(o, "_stat", changed)
    result = o.scan_processes(tmp_path)
    assert result["uncertain"][0]["classification"] == "uncertain_process_identity_changed"


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
