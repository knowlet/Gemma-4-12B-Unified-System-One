"""Fake-only owned-unit closure tests; no systemd, Torch, models or actors."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "_native_training_owner_tests", Path(__file__).with_name("launch_training.py")
)
OWNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OWNER)


@pytest.fixture
def harness(tmp_path, monkeypatch):
    monkeypatch.setattr(OWNER, "ROOT", tmp_path)
    monkeypatch.setattr(OWNER, "HERE", tmp_path / "run")
    OWNER.HERE.mkdir()
    monkeypatch.setattr(OWNER.resource, "setrlimit", lambda *_: None)
    monkeypatch.setattr(OWNER.time, "sleep", lambda *_: None)
    monkeypatch.setattr(OWNER, "scan", lambda _: {"matches": [], "unreadable": []})
    state = {
        "started": False,
        "stopped": False,
        "stop_calls": 0,
        "description": "nonce",
        "queries": 0,
        "first_error": False,
        "busy": False,
        "available": 118_000_000_000,
        "running": False,
        "post_busy": False,
        "post_available": None,
    }
    actor = {"pid": 98765, "pgid": 98765, "sid": 98765, "start_ticks": 123}

    def observe(callback):
        try:
            return callback()
        except BaseException as error:
            return {"observation_error": {"type": type(error).__name__, "message": str(error)}}

    def status(_):
        if not state["started"] or state["stopped"]:
            return {
                "LoadState": "not-found",
                "MainPID": "0",
                "ControlGroup": "",
                "InvocationID": "",
            }
        state["queries"] += 1
        if state["first_error"] and state["queries"] == 1:
            raise RuntimeError("initial status unavailable")
        result = {
            "LoadState": "loaded",
            "MainPID": "0",
            "ExecMainPID": "98765",
            "ControlGroup": "",
            "InvocationID": "a" * 32,
            "SubState": "exited",
            "Result": "success",
            "ExecMainCode": "1",
            "ExecMainStatus": "0",
        }
        if state["running"]:
            result.update(MainPID="98765", ControlGroup=OWNER.CGROUP, SubState="running")
        return result

    def host():
        return {
            "mem_available_bytes": state["post_available"]
            if state["started"] and state["post_available"] is not None
            else state["available"],
            "comfy_running": int(state["busy"] or state["started"] and state["post_busy"]),
            "comfy_pending": 0,
        }

    def command(*args, timeout=15):
        if args[0] == "systemd-run":
            state["started"] = True
            (OWNER.HERE / "actor-intent-01.json").write_text(json.dumps({"actor": actor}))
            fit = OWNER.HERE / "fit-01"
            (fit / "adapter").mkdir(parents=True)
            (fit / "adapter/manifest.json").write_text("{}")
            (fit / "receipt.json").write_text(
                json.dumps(
                    {
                        "stage": "complete",
                        "training": {
                            "completed_updates": 142,
                            "development_examples": 37,
                            "development_updates": 0,
                        },
                    }
                )
            )
            return "fake launch\n"
        if "stop" in args:
            assert args == ("systemctl", "--user", "stop", OWNER.UNIT)
            assert timeout >= 25
            state["stop_calls"] += 1
            state["stopped"] = True
            return ""
        if "--property=Description" in args:
            return state["description"] + "\n"
        raise AssertionError(args)

    guard = SimpleNamespace(status=status, host=host, observe=observe)
    monkeypatch.setattr(OWNER, "load_guard", lambda: (guard, {"fake_guard": True}))
    monkeypatch.setattr(OWNER, "command", command)
    plan = {
        "unit": OWNER.UNIT,
        "cgroup": OWNER.CGROUP,
        "output": "run/fit-01",
        "unit_memory_max_bytes": OWNER.CAP,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "nonce": "nonce",
        "sources": [],
        "fitter_arguments": [],
    }
    path = tmp_path / "plan.json"
    blob = json.dumps(plan).encode()
    path.write_bytes(blob)
    arguments = ["--run", "--plan", str(path), "--plan-sha256", hashlib.sha256(blob).hexdigest()]
    return state, arguments


def test_default_dry_has_no_process_or_gpu(harness, capsys):
    state, _ = harness
    assert OWNER.main([]) == 0
    assert "DRY_NO_ACTOR_OR_MODEL" in capsys.readouterr().out
    assert not state["started"]


def test_success_joins_invocation_actor_and_whole_unit(harness):
    state, arguments = harness
    assert OWNER.main(arguments) == 0
    closure = json.loads((OWNER.HERE / "root-external-closure-01.json").read_bytes())
    assert state["stop_calls"] == 1 and state["stopped"]
    assert closure["invocation_id"] == "a" * 32 and closure["cgroup_absent_after_stop"]
    assert closure["actor"]["pid"] == 98765
    assert closure["two_proc_scans"] == [{"matches": [], "unreadable": []}] * 2


def test_initial_query_failure_authenticates_and_stops_owned_nonce(harness):
    state, arguments = harness
    state["first_error"] = True
    with pytest.raises(RuntimeError, match="initial status unavailable"):
        OWNER.main(arguments)
    closure = json.loads((OWNER.HERE / "root-external-closure-01.json").read_bytes())
    assert state["stop_calls"] == 1 and state["stopped"]
    assert closure["primary_error"]["message"] == "initial status unavailable"


def test_unknown_nonce_is_not_stopped_or_relabelled_closed(harness):
    state, arguments = harness
    state["description"] = "foreign"
    with pytest.raises(AssertionError):
        OWNER.main(arguments)
    closure = json.loads((OWNER.HERE / "root-external-closure-01.json").read_bytes())
    assert state["stop_calls"] == 0
    assert closure["stop_result"]["closure"] == "UNKNOWN_NO_AUTHENTICATED_OWNED_INVOCATION"


@pytest.mark.parametrize("mutation", ["busy", "available"])
def test_prelaunch_resource_or_queue_refusal_never_starts(harness, mutation):
    state, arguments = harness
    state[mutation] = True if mutation == "busy" else 100_000_000_000
    with pytest.raises(AssertionError):
        OWNER.main(arguments)
    assert not state["started"] and state["stop_calls"] == 0


@pytest.mark.parametrize("condition", ["queue", "host"])
def test_running_resource_or_queue_change_stops_only_owned_unit(harness, condition):
    state, arguments = harness
    state["running"] = True
    state["post_busy"] = condition == "queue"
    state["post_available"] = OWNER.RESERVE + 1024**3 if condition == "host" else None
    with pytest.raises(AssertionError):
        OWNER.main(arguments)
    closure = json.loads((OWNER.HERE / "root-external-closure-01.json").read_bytes())
    assert state["stop_calls"] == 1 and state["stopped"]
    expected = (
        "Comfy_became_busy_stop_own_unit_only"
        if condition == "queue"
        else "host_reserve_margin_crossed"
    )
    assert closure["reason"] == expected


@pytest.mark.parametrize("secondary", ["scan", "save"])
def test_secondary_finalization_failure_preserves_primary_and_owned_stop(
    harness, monkeypatch, secondary
):
    state, arguments = harness
    state["first_error"] = True
    if secondary == "scan":

        def failed_scan(_):
            raise OSError("scan unavailable")

        monkeypatch.setattr(OWNER, "scan", failed_scan)
    else:
        original = OWNER.save

        def failed_save(name, value):
            if name == "root-external-closure-01.json":
                raise OSError("closure persistence unavailable")
            return original(name, value)

        monkeypatch.setattr(OWNER, "save", failed_save)
    with pytest.raises(RuntimeError, match="initial status unavailable") as caught:
        OWNER.main(arguments)
    assert state["stop_calls"] == 1 and state["stopped"]
    assert any("Owner finalization also failed" in note for note in caught.value.__notes__)
    proof = json.loads((OWNER.HERE / "root-finalization-error-01.json").read_bytes())
    assert proof["closure"] == "UNKNOWN_INCOMPLETE_FINALIZATION"
    assert proof["primary_error"]["message"] == "initial status unavailable"
