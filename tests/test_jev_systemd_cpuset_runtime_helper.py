"""Synthetic-only coverage: no systemd, sudo, real /run, or model operations."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "cpuset_helper", Path(__file__).parents[1] / "scripts/jev_systemd_cpuset_runtime_helper.py"
)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
INVOCATION = "a" * 32
PID = 1234


class FakeSystemd:
    def __init__(self, root):
        self.root = root
        self.applied = False
        self.calls = []
        self.overrides = {}
        self.fail_reload = 0

    def show(self):
        self.calls.append("show")
        state = {
            "LoadState": "loaded",
            "ActiveState": "active",
            "InvocationID": INVOCATION,
            "MainPID": str(PID),
            "ControlGroup": "/user.slice/user-1001.slice/user@1001.service",
            "Delegate": "yes",
            "DelegateControllers": "cpu memory pids cpuset" if self.applied else "cpu memory pids",
            "FragmentPath": helper.FRAGMENT,
            "DropInPaths": " ".join(
                sorted(helper.BASELINE_DROPINS | ({helper.TARGET} if self.applied else set()))
            ),
            "NeedDaemonReload": "no",
        }
        state.update(self.overrides)
        return state

    def reload(self):
        self.calls.append("reload")
        if self.fail_reload:
            self.fail_reload -= 1
            raise OSError("fake reload failed")
        self.applied = (self.root / helper.DIRNAME / helper.FILENAME).is_file()


@pytest.fixture
def environment(tmp_path, monkeypatch):
    root = tmp_path / "systemd"
    root.mkdir()
    root.chmod(0o700)
    monkeypatch.setattr(helper, "BASE", root)
    monkeypatch.setattr(helper, "TARGET", str(root / helper.DIRNAME / helper.FILENAME))
    monkeypatch.setattr(helper, "SOURCE_ONLY", False)
    return root, FakeSystemd(root)


def test_gate_fails_before_any_backend_or_filesystem_action(environment, monkeypatch):
    root, backend = environment
    monkeypatch.setattr(helper, "SOURCE_ONLY", True)
    with pytest.raises(ValueError, match="release required"):
        helper.install(INVOCATION, PID, backend=backend)
    assert backend.calls == []
    assert list(root.iterdir()) == []


def test_install_and_rollback_with_exact_receipt(environment):
    root, backend = environment
    result = helper.install(INVOCATION, PID, backend=backend)
    target = root / helper.DIRNAME / helper.FILENAME
    assert target.read_bytes() == helper.CONTENT
    assert result["status"] == "configured_effect_unverified"
    assert "system-wide" in result["daemon_reload_scope"]
    assert "cannot guarantee" in result["rollback_limit"]
    result = helper.rollback(result["receipt"], backend=backend)
    assert result["status"] == "dropin_removed_cgroup_effects_unverified"
    assert not target.exists()
    assert not target.parent.exists()
    assert backend.calls.count("reload") == 2


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("InvocationID", "b" * 32),
        ("MainPID", "1235"),
        ("DelegateControllers", "cpu memory pids cpuset"),
        ("NeedDaemonReload", "yes"),
        ("DropInPaths", ""),
        ("FragmentPath", "/wrong"),
        ("ActiveState", "inactive"),
    ],
)
def test_baseline_mismatch_refuses_before_write(environment, key, value):
    root, backend = environment
    backend.overrides[key] = value
    with pytest.raises(helper.OperationFailed) as failure:
        helper.install(INVOCATION, PID, backend=backend)
    assert failure.value.record["cleanup"] == "no_owned_dropin"
    assert not list(root.iterdir())
    assert "reload" not in backend.calls


def test_existing_file_is_not_overwritten(environment):
    root, backend = environment
    target = root / helper.DIRNAME / helper.FILENAME
    target.parent.mkdir()
    target.write_text("other")
    with pytest.raises(helper.OperationFailed):
        helper.install(INVOCATION, PID, backend=backend)
    assert target.read_text() == "other"
    assert "reload" not in backend.calls


def test_existing_directory_and_unrelated_file_are_preserved(environment):
    root, backend = environment
    folder = root / helper.DIRNAME
    folder.mkdir()
    folder.chmod(0o700)
    unrelated = folder / "unrelated.conf"
    unrelated.write_text("existing")
    installed = helper.install(INVOCATION, PID, backend=backend)
    helper.rollback(installed["receipt"], backend=backend)
    assert unrelated.read_text() == "existing"


def test_reload_failure_attempts_owned_cleanup_and_second_reload(environment):
    root, backend = environment
    backend.fail_reload = 1
    with pytest.raises(helper.OperationFailed) as failure:
        helper.install(INVOCATION, PID, backend=backend)
    assert failure.value.record["events"][-2:] == [
        "owned_dropin_removed",
        "cleanup_reload_returned",
    ]
    assert backend.calls.count("reload") == 2
    assert not list(root.iterdir())


def test_identity_drift_after_create_cleans_up_without_reload(environment):
    root, backend = environment
    original_show = backend.show

    def show():
        result = original_show()
        if len(backend.calls) == 2:
            result["InvocationID"] = "b" * 32
        return result

    backend.show = show
    with pytest.raises(helper.OperationFailed) as failure:
        helper.install(INVOCATION, PID, backend=backend)
    assert failure.value.record["cleanup"] == "dropin_removed_cgroup_effects_unverified"
    assert not list(root.iterdir())
    assert "reload" not in backend.calls


def test_cleanup_reload_failure_is_reported(environment):
    root, backend = environment
    backend.fail_reload = 2
    with pytest.raises(helper.OperationFailed) as failure:
        helper.install(INVOCATION, PID, backend=backend)
    record = failure.value.record
    assert record["cleanup"] == "unconfirmed"
    assert "owned_dropin_removed" in record["events"]
    assert not (root / helper.DIRNAME / helper.FILENAME).exists()
    assert backend.calls.count("reload") == 2


def test_install_failure_exposes_receipt_to_retry_uncertain_reload(environment):
    root, backend = environment
    calls = 0
    original_reload = backend.reload

    def fail_after_possible_reload():
        nonlocal calls
        calls += 1
        if calls == 1:
            original_reload()  # reload may have succeeded before the caller saw failure
            raise OSError("synthetic lost reload response")
        if calls == 2:
            backend.calls.append("reload")
            raise OSError("synthetic cleanup reload failure")
        original_reload()

    backend.reload = fail_after_possible_reload
    with pytest.raises(helper.OperationFailed) as failure:
        helper.install(INVOCATION, PID, backend=backend)
    record = failure.value.record
    assert record["reload_attempted"] and record["cleanup_reload_attempted"]
    assert record["cleanup"] == "unconfirmed"
    assert "owned_dropin_removed" in record["events"]
    assert not (root / helper.DIRNAME / helper.FILENAME).exists()
    recovered = helper.rollback(record["recovery_receipt"], backend=backend)
    assert recovered["events"][0] == "dropin_absent_reload_retry"
    assert recovered["status"] == "dropin_removed_cgroup_effects_unverified"
    assert not list(root.iterdir())


def test_install_failure_recovery_receipt_rejects_replaced_file(environment):
    root, backend = environment
    backend.fail_reload = 2
    with pytest.raises(helper.OperationFailed) as failure:
        helper.install(INVOCATION, PID, backend=backend)
    target = root / helper.DIRNAME / helper.FILENAME
    target.write_text("someone else's file")
    with pytest.raises(helper.OperationFailed):
        helper.rollback(failure.value.record["recovery_receipt"], backend=backend)
    assert target.read_text() == "someone else's file"
    assert backend.calls.count("reload") == 2


def test_partial_write_is_removed_by_inode(environment, monkeypatch):
    root, backend = environment
    original_write = helper.os.write
    calls = 0

    def fail_after_prefix(fd, data):
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_write(fd, data[:7])
        raise OSError("synthetic short write")

    monkeypatch.setattr(helper.os, "write", fail_after_prefix)
    with pytest.raises(helper.OperationFailed) as failure:
        helper.install(INVOCATION, PID, backend=backend)
    assert failure.value.record["cleanup"] == "dropin_removed_cgroup_effects_unverified"
    assert not list(root.iterdir())
    assert "reload" not in backend.calls


def test_partial_write_replaced_file_is_preserved(environment, monkeypatch):
    root, backend = environment
    original_write = helper.os.write
    calls = 0
    target = root / helper.DIRNAME / helper.FILENAME

    def replace_after_prefix(fd, data):
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_write(fd, data[:7])
        target.unlink()
        target.write_text("someone else's file")
        raise OSError("synthetic write failure")

    monkeypatch.setattr(helper.os, "write", replace_after_prefix)
    with pytest.raises(helper.OperationFailed) as failure:
        helper.install(INVOCATION, PID, backend=backend)
    assert failure.value.record["cleanup"] == "unconfirmed"
    assert target.read_text() == "someone else's file"
    assert "reload" not in backend.calls


def test_rollback_reload_can_be_retried_after_file_removal(environment):
    root, backend = environment
    receipt = helper.install(INVOCATION, PID, backend=backend)["receipt"]
    backend.fail_reload = 1
    with pytest.raises(helper.OperationFailed) as failure:
        helper.rollback(receipt, backend=backend)
    assert "owned_dropin_removed" in failure.value.record["events"]
    assert not (root / helper.DIRNAME / helper.FILENAME).exists()
    backend.overrides["NeedDaemonReload"] = "yes"
    original_reload = backend.reload

    def reload():
        original_reload()
        backend.overrides["NeedDaemonReload"] = "no"

    backend.reload = reload
    result = helper.rollback(receipt, backend=backend)
    assert result["events"][0] == "dropin_absent_reload_retry"
    assert result["status"] == "dropin_removed_cgroup_effects_unverified"
    assert not list(root.iterdir())


def test_reload_needed_only_after_own_write_is_accepted(environment):
    _root, backend = environment
    original_show = backend.show

    def show():
        result = original_show()
        if len(backend.calls) == 2:
            result["NeedDaemonReload"] = "yes"
        return result

    backend.show = show
    result = helper.install(INVOCATION, PID, backend=backend)
    assert result["status"] == "configured_effect_unverified"


def test_changed_file_is_never_deleted_on_rollback(environment):
    root, backend = environment
    result = helper.install(INVOCATION, PID, backend=backend)
    target = root / helper.DIRNAME / helper.FILENAME
    target.write_text("modified by someone else")
    with pytest.raises(helper.OperationFailed) as failure:
        helper.rollback(result["receipt"], backend=backend)
    assert failure.value.record["cleanup"] == "unconfirmed"
    assert target.read_text() == "modified by someone else"
    assert backend.calls.count("reload") == 1


def test_changed_unit_identity_prevents_rollback(environment):
    root, backend = environment
    result = helper.install(INVOCATION, PID, backend=backend)
    backend.overrides["InvocationID"] = "b" * 32
    with pytest.raises(helper.OperationFailed):
        helper.rollback(result["receipt"], backend=backend)
    assert (root / helper.DIRNAME / helper.FILENAME).is_file()
    assert backend.calls.count("reload") == 1


def test_symlink_directory_refused(environment, tmp_path):
    root, backend = environment
    alternate = tmp_path / "elsewhere"
    alternate.mkdir()
    (root / helper.DIRNAME).symlink_to(alternate, target_is_directory=True)
    with pytest.raises(helper.OperationFailed):
        helper.install(INVOCATION, PID, backend=backend)
    assert not list(alternate.iterdir())
    assert "reload" not in backend.calls


def test_reload_is_fixed_and_never_uses_shell(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return type("Completed", (), {"stdout": ""})()

    monkeypatch.setattr(helper.subprocess, "run", fake_run)
    helper.SystemCtl().reload()
    assert calls[0][0] == ["/usr/bin/systemctl", "daemon-reload"]
    assert not calls[0][1].get("shell", False)
