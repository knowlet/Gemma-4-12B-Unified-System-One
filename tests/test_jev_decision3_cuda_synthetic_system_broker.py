"""Pure synthetic files and fake systemd; no host service, sudo, or model."""

import hashlib
import json
import subprocess
from types import SimpleNamespace

import jev_decision3_cuda_synthetic_system_broker as broker
import pytest


def encode(value):
    return json.dumps(value, separators=(",", ":")).encode()


@pytest.fixture
def fake(tmp_path, monkeypatch):
    nonce = "a" * 32
    run_root = tmp_path / "runs"
    run = run_root / ("run-" + nonce)
    run.mkdir(parents=True)
    output = run / "worker"
    output.mkdir(mode=0o700)
    receipt_root = tmp_path / "receipts"
    receipt_root.mkdir()
    bootstrap = tmp_path / "bootstrap_release.py"
    bootstrap.write_bytes(b"synthetic reviewed bootstrap")
    bootsha = hashlib.sha256(bootstrap.read_bytes()).hexdigest()
    unit = "jev-cuda-synthetic-system-" + nonce + ".service"
    plan = {
        "schema": "decision3-cuda-synthetic-system-plan-v1",
        "unit": unit,
        "output_dir": str(output),
        "python": broker.PYTHON,
        "source_pins": {str(bootstrap): bootsha},
        "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
        "wall_seconds": 120,
        "memory_floor_bytes": 34709107508,
    }
    raw = encode(plan)
    (run / "plan.json").write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    for name, value in {
        "SOURCE_ONLY": False,
        "RUN_NONCE": nonce,
        "PLAN_SHA256": digest,
        "BOOTSTRAP_SHA256": bootsha,
        "RUN_ROOT": run_root,
        "RECEIPT_ROOT": receipt_root,
        "BOOTSTRAP": bootstrap,
    }.items():
        monkeypatch.setattr(broker, name, value)
    monkeypatch.setattr(broker.os, "geteuid", lambda: 0)
    monkeypatch.setattr(broker, "_boot_id", lambda: "synthetic-boot")
    group = "/system.slice/" + unit
    monkeypatch.setattr(broker, "_process", lambda pid: {"startticks": 55, "cgroup": group})
    running = {k: "" for k in broker.PROPERTIES}
    running.update(
        LoadState="loaded",
        ActiveState="active",
        Type="exec",
        User="1001",
        InvocationID="b" * 32,
        MainPID="123",
        ControlGroup=group,
    )
    absent = {**running, "LoadState": "not-found", "ActiveState": "inactive", "MainPID": "0"}
    calls = []
    monkeypatch.setattr(broker, "_call", lambda args, **kwargs: calls.append((args, kwargs)) or b"")
    monkeypatch.setattr(broker, "_show", lambda _: absent if not calls else running)
    return SimpleNamespace(
        run=run,
        output=output,
        unit=unit,
        digest=digest,
        running=running,
        calls=calls,
        receipt=receipt_root / ("jev-cuda-synthetic-system-" + nonce),
    )


def test_source_gate_and_unbound_release_refuse_before_io(monkeypatch):
    monkeypatch.setattr(broker, "SOURCE_ONLY", True)
    monkeypatch.setattr(broker, "_plan", lambda *_: pytest.fail("unexpected plan read"))
    with pytest.raises(ValueError, match="reviewed"):
        broker.start()
    monkeypatch.setattr(broker, "SOURCE_ONLY", False)
    monkeypatch.setattr(broker.os, "geteuid", lambda: 0)
    monkeypatch.setattr(broker, "RUN_NONCE", None)
    with pytest.raises(ValueError, match="nonce"):
        broker.start()


def test_fixed_start_binds_only_system_service_and_writes_receipt(fake):
    result = broker.start()
    assert result["unit"] == fake.unit and result["plan_sha256"] == fake.digest
    args, kwargs = fake.calls[0]
    assert args[0] == "/usr/bin/systemd-run" and "--system" in args and "--user" not in args
    for item in (
        "--slice=system.slice",
        "--unit=" + fake.unit,
        "--property=User=1001",
        "--property=AllowedCPUs=15-19",
        "--property=CPUQuota=500%",
        "--property=MemoryMax=8589934592",
        "--property=MemorySwapMax=0",
        "--property=RuntimeMaxSec=120",
        "--property=KillMode=control-group",
    ):
        assert item in args
    assert kwargs["timeout"] == 10 and "-I" in args
    assert json.loads((fake.receipt / "binding.json").read_bytes())["invocation"] == "b" * 32
    with pytest.raises((FileExistsError, ValueError)):
        broker.start()
    assert len(fake.calls) == 1


def test_invalid_plan_and_bootstrap_rejected_before_system_call(fake):
    (fake.run / "plan.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest"):
        broker.start()
    assert not fake.calls and not fake.receipt.exists()


def test_ambiguous_launch_preserves_intent_without_claiming_unit(fake, monkeypatch):
    def timed_out(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("synthetic", 10)

    monkeypatch.setattr(broker, "_call", timed_out)
    with pytest.raises(subprocess.TimeoutExpired):
        broker.start()
    assert (fake.receipt / "intent.json").exists()
    assert not (fake.receipt / "binding.json").exists()


def _hello(fake, binding):
    (fake.output / "hello.json").write_bytes(
        encode({"schema": "decision3-cuda-synthetic-hello-v1", **binding, "pgid": binding["pid"]})
    )


def test_stop_requires_matching_root_receipt_and_worker(fake, monkeypatch):
    result = broker.start()
    binding = json.loads((fake.receipt / "binding.json").read_bytes())
    _hello(fake, binding)
    # Actual synthetic files are owned by the test UID, not root.
    with pytest.raises(ValueError, match="root-owned"):
        broker.stop()
    monkeypatch.setattr(broker, "_root_owned", lambda _: True)
    stopped = broker.stop()
    assert stopped["invocation"] == result["invocation"]
    assert fake.calls[-1][0] == ["/usr/bin/systemctl", "--system", "stop", fake.unit]
    assert stopped["status"] == "stop_returned_closure_unconfirmed"


@pytest.mark.parametrize("fault", ["invocation", "group", "pid", "hello", "uid"])
def test_stop_rejects_replacement_or_bad_identity(fake, monkeypatch, fault):
    broker.start()
    binding = json.loads((fake.receipt / "binding.json").read_bytes())
    _hello(fake, binding)
    monkeypatch.setattr(broker, "_root_owned", lambda _: True)
    if fault == "invocation":
        fake.running["InvocationID"] = "c" * 32
    elif fault == "group":
        fake.running["ControlGroup"] = "/user.slice/other"
    elif fault == "pid":
        fake.running["MainPID"] = "124"
    elif fault == "hello":
        (fake.output / "hello.json").write_bytes(b"{}")
    else:
        fake.running["User"] = "0"
    with pytest.raises(ValueError):
        broker.stop()
    assert len(fake.calls) == 1


def test_show_uses_only_fixed_system_manager_and_unit(fake, monkeypatch):
    calls = []
    monkeypatch.undo()
    monkeypatch.setattr(
        broker,
        "_call",
        lambda args, **kwargs: (
            calls.append((args, kwargs))
            or b"".join((key + "=\n").encode() for key in broker.PROPERTIES)
        ),
    )
    broker._show(fake.unit)
    assert calls[0][0][:4] == ["/usr/bin/systemctl", "--system", "show", fake.unit]
    assert "--user" not in calls[0][0]
