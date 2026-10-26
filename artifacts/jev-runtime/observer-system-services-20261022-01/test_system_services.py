"""Pure authority/race fixtures; never start or signal a foreign system service."""

import copy
import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location("_system_observer_tests", HERE / "comfy_observer.py")
o = importlib.util.module_from_spec(spec)
exec(
    compile((HERE / "comfy_observer.py").read_bytes(), str(HERE / "comfy_observer.py"), "exec"),
    o.__dict__,
)
BOOT = "11111111-2222-4333-8444-555555555555"


@pytest.fixture
def authority(monkeypatch):
    manifest = o._system_services()
    selected = manifest["services"][0]
    process = {
        "start_ticks": 123,
        "comm_sha256": "a" * 64,
        "argv": {"sha256": "b" * 64, "size_bytes": 65},
        "cgroup": "/system.slice/apt-news.service",
        "apparmor": "ubuntu_pro_apt_news (enforce)",
    }
    unit = {
        "Id": "apt-news.service",
        "LoadState": "loaded",
        "ActiveState": "activating",
        "SubState": "start",
        "MainPID": "555555",
        "ExecMainPID": "555555",
        "InvocationID": "c" * 32,
        "ControlGroup": process["cgroup"],
        "FragmentPath": selected["unit_file"]["path"],
        "User": "",
        "Group": "",
        "AppArmorProfile": "-ubuntu_pro_apt_news",
        "WorkingDirectory": "",
        "Environment": "",
        "EnvironmentFiles": "",
        "DropInPaths": "",
        "ExecStart": "{ path=/usr/bin/python3 ; argv[]="
        + " ".join(selected["argv"])
        + " ; ignore_errors=no ; pid=555555 ; code=(null) ; }",
    }
    monkeypatch.setattr(o, "_boot", lambda: BOOT)
    monkeypatch.setattr(o, "_service_process", lambda *_: copy.deepcopy(process))
    monkeypatch.setattr(o, "_service_unit", lambda *_: copy.deepcopy(unit))
    return manifest, selected, process, unit


def test_exact_installed_root_file_pins_and_authenticated_live_sandwich(authority):
    manifest, selected, _, _ = authority
    for ref in (
        manifest["interpreter"],
        manifest["package_files"],
        selected["script"],
        selected["unit_file"],
        selected["apparmor"],
    ):
        assert len(o._os_ref(ref)) == ref["size_bytes"]
    diagnostics = []
    assert o._authenticated_system_service(Path("/proc/555555"), selected["argv"], diagnostics)
    assert diagnostics[-1]["authenticated_system_service"] == "apt-news.service"
    assert diagnostics[-1]["invocation_id"] == "c" * 32


@pytest.mark.parametrize(
    "field,value",
    [
        ("MainPID", "0"),
        ("ExecMainPID", "1"),
        ("InvocationID", ""),
        ("ControlGroup", "/system.slice/other.service"),
        ("Id", "other.service"),
        ("LoadState", "not-found"),
        ("ActiveState", "inactive"),
        ("SubState", "dead"),
        ("FragmentPath", "/tmp/apt-news.service"),
        ("User", "albert_hsieh"),
        ("AppArmorProfile", "unconfined"),
        ("WorkingDirectory", "/home/albert_hsieh/ComfyUI"),
        ("Environment", "PYTHONPATH=/tmp"),
        ("ExecStart", "python /tmp/apt_news.py"),
        ("EnvironmentFiles", "/tmp/environment"),
        ("DropInPaths", "/tmp/override.conf"),
    ],
)
def test_missing_or_different_live_unit_authority_remains_unknown(authority, field, value):
    _, selected, _, unit = authority
    unit[field] = value
    diagnostics = []
    assert not o._authenticated_system_service(Path("/proc/555555"), selected["argv"], diagnostics)
    assert diagnostics[-1]["system_service_authentication"] == "refused"


@pytest.mark.parametrize(
    "field,value",
    [
        ("cgroup", "/user.slice/ComfyUI.service"),
        ("apparmor", "unconfined"),
        ("apparmor", "ubuntu_pro_apt_news (complain)"),
    ],
)
def test_exact_current_process_security_context_required(authority, field, value):
    _, selected, process, _ = authority
    process[field] = value
    assert not o._authenticated_system_service(Path("/proc/555555"), selected["argv"], [])


@pytest.mark.parametrize("kind", ["start", "invocation", "boot", "unreadable", "selected-file"])
def test_sandwich_drift_or_unreadable_authority_refuses(authority, monkeypatch, kind):
    _, selected, process, unit = authority
    if kind == "start":
        values = [copy.deepcopy(process), {**process, "start_ticks": 124}]
        monkeypatch.setattr(o, "_service_process", lambda *_: values.pop(0))
    elif kind == "invocation":
        values = [copy.deepcopy(unit), {**unit, "InvocationID": "d" * 32}]
        monkeypatch.setattr(o, "_service_unit", lambda *_: values.pop(0))
    elif kind == "boot":
        values = [BOOT, "00000000-0000-4000-8000-000000000000"]
        monkeypatch.setattr(o, "_boot", lambda: values.pop(0))
    elif kind == "unreadable":

        def denied(*_):
            raise PermissionError("synthetic authority unreadable")

        monkeypatch.setattr(o, "_service_process", denied)
    else:

        def drift(*_):
            raise o.ObservationError("synthetic fixed file drift")

        monkeypatch.setattr(o, "_os_ref", drift)
    assert not o._authenticated_system_service(Path("/proc/555555"), selected["argv"], [])


@pytest.mark.parametrize(
    "argv",
    [
        ["python", "/usr/lib/ubuntu-advantage/apt_news.py"],
        ["/usr/bin/python3", "/tmp/apt_news.py"],
        ["/usr/bin/python3", "/usr/lib/ubuntu-advantage/apt_news.py", "--extra"],
        ["/usr/bin/python3", "/home/albert_hsieh/ComfyUI/main.py"],
        ["/usr/bin/python3", "-m", "comfy"],
    ],
)
def test_root_or_script_name_alone_never_opens_authority_queries(monkeypatch, argv):
    monkeypatch.setattr(o, "_system_services", lambda: pytest.fail("Unselected command"))
    assert not o._authenticated_system_service(Path("/proc/555555"), argv, [])


def test_owned_user_file_and_changed_selected_bytes_are_not_os_authority(tmp_path):
    path = tmp_path / "apt_news.py"
    path.write_text("# fixture only")
    with pytest.raises(o.ObservationError):
        o._root_locked(path, regular=True)
    selected = o._system_services()["services"][0]["script"]
    with pytest.raises(o.ObservationError, match="changed"):
        o._os_ref({**selected, "sha256": "0" * 64})


def eligible():
    return {
        "pid": 42,
        "start_ticks": 123,
        "proc_owner_uid": 0,
        "classification": "uncertain_python_cwd",
        "disappearance_resolution_eligible": True,
    }


@pytest.fixture
def wait(monkeypatch):
    monkeypatch.setattr(o, "_open_pidfd", lambda _pid: 142)
    monkeypatch.setattr(o.os, "close", lambda _fd: None)
    monkeypatch.setattr(o, "RESOLUTION_SECONDS", 0.01)

    class Poll:
        def register(self, *_):
            pass

        def poll(self, _timeout):
            return [(142, o.select.POLLIN)]

    monkeypatch.setattr(o.select, "poll", Poll)


def test_initial_cwd_unknown_empty_transition_waits_for_two_absences_and_pidfd(wait, monkeypatch):
    count = 0

    def state(_):
        nonlocal count
        count += 1
        if count == 1:
            return "python", "S", 123
        raise FileNotFoundError("owned fake naturally gone")

    monkeypatch.setattr(o, "_stat", state)
    monkeypatch.setattr(
        o,
        "_process_attempt",
        lambda _: (
            {
                "start_ticks": 123,
                "classification": "uncertain_python",
                "transitional_empty_python_command": True,
            },
            None,
        ),
    )
    receipt = o._wait_disappeared([eligible()])
    assert receipt["disappeared"] == [42] and count >= 3
    assert receipt["observed_contexts"][0]["classification"] == "uncertain_python"
    assert receipt["observed_contexts"][0]["transitional_empty_python_command"] is True


def test_alive_empty_command_still_refuses_and_diagnostics_are_bounded(wait, monkeypatch):
    monkeypatch.setattr(o, "_stat", lambda _: ("python", "S", 123))
    monkeypatch.setattr(
        o,
        "_process_attempt",
        lambda _: (
            {
                "start_ticks": 123,
                "classification": "uncertain_python",
                "transitional_empty_python_command": True,
            },
            None,
        ),
    )
    with pytest.raises(o.ObservationError, match="remains after") as caught:
        o._wait_disappeared([eligible()])
    assert 0 < len(caught.value.evidence["observed_contexts"]) <= 32


@pytest.mark.parametrize(
    "classification,empty,ticks",
    [
        ("uncertain_python", False, 123),
        ("uncertain_python", True, 124),
        ("uncertain_comfy_cwd", True, 123),
        ("comfy_runtime", True, 123),
    ],
)
def test_nonempty_intrinsic_changed_identity_and_comfy_are_never_empty_exempted(
    wait, monkeypatch, classification, empty, ticks
):
    monkeypatch.setattr(o, "_stat", lambda _: ("python", "S", 123))
    monkeypatch.setattr(
        o,
        "_process_attempt",
        lambda _: (
            {
                "start_ticks": ticks,
                "classification": classification,
                "transitional_empty_python_command": empty,
            },
            None,
        ),
    )
    with pytest.raises(o.ObservationError) as caught:
        o._wait_disappeared([eligible()])
    assert caught.value.evidence["observed_contexts"][-1]["classification"] == classification


def test_non_cwd_initial_unknown_cannot_use_empty_transition(wait, monkeypatch):
    monkeypatch.setattr(o, "_stat", lambda _: ("python", "S", 123))
    monkeypatch.setattr(
        o,
        "_process_attempt",
        lambda _: (
            {
                "start_ticks": 123,
                "classification": "uncertain_python",
                "transitional_empty_python_command": True,
            },
            None,
        ),
    )
    with pytest.raises(o.ObservationError, match="Non-resolvable"):
        o._wait_disappeared([{**eligible(), "classification": "uncertain_process_read"}])


def test_profile_keeps_original_half_second_and_global_two_second_budgets():
    assert o.RESOLUTION_SECONDS == 0.5 and o.OBSERVE_SECONDS == 2.0
    assert (
        o.SYSTEM_SERVICES_SHA
        == __import__("hashlib").sha256(o.SYSTEM_SERVICES.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("argv,expected", [([], True), (["python", "--unknown-option"], False)])
def test_empty_transition_tag_requires_actual_empty_command_in_both_context_reads(
    monkeypatch, argv, expected
):
    monkeypatch.setattr(o, "_stat", lambda _: ("python", "S", 123))
    monkeypatch.setattr(
        o, "_command", lambda *_: (argv, {"sha256": "a" * 64, "size_bytes": 0 if not argv else 30})
    )
    value, race = o._process_attempt(Path("/proc/42"))
    assert value["classification"] == "uncertain_python" and race is None
    assert value["transitional_empty_python_command"] is expected


def test_known_comfy_cwd_never_invokes_os_service_override(monkeypatch):
    manifest = o._system_services()
    argv = manifest["services"][0]["argv"]
    monkeypatch.setattr(o, "_command", lambda *_: (argv, {"sha256": "a" * 64, "size_bytes": 60}))
    monkeypatch.setattr(o.os, "readlink", lambda _: str(o.COMFY_ROOT))
    monkeypatch.setattr(
        o, "_authenticated_system_service", lambda *_: pytest.fail("Comfy evidence overridden")
    )
    value, _signature = o._process_context(Path("/proc/42"), "python3", "S", [])
    assert value == "uncertain_comfy_cwd"


def test_second_selected_esm_cache_has_its_own_exact_authority(authority, monkeypatch):
    manifest, _selected, process, unit = authority
    selected = manifest["services"][1]
    process["cgroup"] = "/system.slice/" + selected["unit"]
    process["apparmor"] = selected["apparmor_profile"] + " (enforce)"
    unit.update(
        Id=selected["unit"],
        ControlGroup=process["cgroup"],
        FragmentPath=selected["unit_file"]["path"],
        AppArmorProfile="-" + selected["apparmor_profile"],
        ExecStart="{ path=/usr/bin/python3 ; argv[]="
        + " ".join(selected["argv"])
        + " ; ignore_errors=no ; pid=555555 ; code=(null) ; }",
    )
    diagnostics = []
    assert o._authenticated_system_service(Path("/proc/555555"), selected["argv"], diagnostics)
    assert diagnostics[-1]["authenticated_system_service"] == "esm-cache.service"


def test_empty_then_nonempty_intrinsic_unknown_is_not_an_empty_only_transition(monkeypatch):
    monkeypatch.setattr(o, "_stat", lambda _: ("python", "S", 123))
    values = [
        ([], {"sha256": "a" * 64, "size_bytes": 0}),
        (["python", "--unknown"], {"sha256": "b" * 64, "size_bytes": 20}),
    ]
    monkeypatch.setattr(o, "_command", lambda *_: values.pop(0))
    value, _ = o._process_attempt(Path("/proc/42"))
    assert value["classification"] == "uncertain_python"
    assert value["transitional_empty_python_command"] is False


def test_late_changed_failure_context_is_retained_after_ring_is_full(wait, monkeypatch):
    monkeypatch.setattr(o, "RESOLUTION_SECONDS", 0.1)
    monkeypatch.setattr(o, "_stat", lambda _: ("python", "S", 123))
    monkeypatch.setattr(o.time, "sleep", lambda _: None)
    calls = 0

    def context(_):
        nonlocal calls
        calls += 1
        return {
            "start_ticks": 123,
            "classification": "uncertain_python",
            "transitional_empty_python_command": calls < 40,
        }, None

    monkeypatch.setattr(o, "_process_attempt", context)
    with pytest.raises(o.ObservationError, match="Non-resolvable") as caught:
        o._wait_disappeared([eligible()])
    evidence = caught.value.evidence
    assert calls == 40 and len(evidence["observed_contexts"]) == 32
    assert evidence["observed_contexts_dropped"] == 8
    assert evidence["observed_contexts"][-1]["transitional_empty_python_command"] is False
