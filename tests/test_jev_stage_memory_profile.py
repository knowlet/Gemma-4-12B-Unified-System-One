"""Pure capsule parity/failure tests: no framework/native/browser/unit launch.

Selected frozen H12 pure tests are derived here as ordinary test source, never
runtime imported. Mechanical adaptations are recorded in the derivation proof.
"""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "scripts/jev_stage_memory_profile.py"
FROZEN = (
    SOURCE.parents[1]
    / "artifacts/jev-runtime/public-native-fit-memory-peak-probe-20261018-01/source-HOLD-12/native_peak_probe_01.py"
)


def optional_frozen_source():
    assert not FROZEN.is_symlink(), "Pinned H12 source must be an ordinary file"
    if not FROZEN.exists():
        pytest.skip("Optional frozen H12 research source is absent in this checkout")
    blob = FROZEN.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == (
        "ffbe30212bc2c34cd0f01a19599646d7f740ffbf1de1ef24496a7f3676af9c86"
    ), "Pinned H12 research source changed"
    return blob.decode()


@pytest.fixture
def probe():
    spec = importlib.util.spec_from_file_location("isolated_stage_capsule", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_floor_uses_full_caps_without_resident_credit_and_rejects_bool(probe):
    observed = {"MemTotal": 130595860480, "MemAvailable": 120000000000}
    assert probe.research_floor(observed, startup=True) == {
        "reserve_bytes": 26119172096,
        "required_bytes": 116313485312,
        "old_train_required_bytes": 106438961088,
    }
    assert probe.research_floor(observed, startup=False)["required_bytes"] == 77658779648
    with pytest.raises(ValueError, match="physical"):
        probe.research_floor({**observed, "MemAvailable": True}, startup=True)
    assert 2 * probe.ARM_CAP + 4 * probe.MIB == probe.TOTAL_CAP


def fake_cuda(probe, *, backend="native", uuid="GPU-test"):
    calls = []
    state = {"fraction": 1.0}
    cuda = SimpleNamespace(
        device_count=lambda: 1,
        current_device=lambda: 0,
        get_device_properties=lambda _: SimpleNamespace(
            uuid=uuid, name="NVIDIA GB10", total_memory=128 * probe.GIB
        ),
        memory=SimpleNamespace(get_allocator_backend=lambda: backend),
        set_per_process_memory_fraction=lambda f, device: (
            calls.append(("fraction", f, device)),
            state.update(fraction=f),
        ),
        get_per_process_memory_fraction=lambda _: state["fraction"],
    )
    return SimpleNamespace(cuda=cuda), calls


def test_allocator_cap_readback_is_native_uuid_bound_and_conservative(probe):
    torch, calls = fake_cuda(probe)
    actual = probe.allocator_cap(torch, "GPU-test")
    assert calls and calls[0][0] == "fraction"
    assert 0 < actual["allowed_bytes_upper"] < probe.CAP
    torch, calls = fake_cuda(probe, backend="cudaMallocAsync")
    with pytest.raises(ValueError, match="Native"):
        probe.allocator_cap(torch, "GPU-test")
    assert calls == []
    torch, calls = fake_cuda(probe, uuid="GPU-other")
    with pytest.raises(ValueError, match="UUID"):
        probe.allocator_cap(torch, "GPU-test")
    assert calls == []


def test_phase_gate_precedes_call_and_preserves_native_error_with_fence_failure(
    probe, tmp_path, monkeypatch
):
    calls = []
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            synchronize=lambda _: calls.append("fence"),
            reset_peak_memory_stats=lambda _: calls.append("reset"),
        )
    )
    gate = probe.StageGate(torch, tmp_path, {"nonce": "n"}, {"ControlGroup": "g"}, {})
    monkeypatch.setattr(probe, "host_memory", lambda: {})
    monkeypatch.setattr(probe, "phase_cpu", lambda: {})
    monkeypatch.setattr(probe, "cgroup_snapshot", lambda _: {})
    monkeypatch.setattr(gate, "cuda", lambda: {"peak_allocated": 7, "peak_reserved": 9})
    monkeypatch.setattr(gate, "check", lambda *_: calls.append("gate"))
    assert gate.phase("native", lambda: calls.append("actual") or 17) == 17
    assert calls[:4] == ["gate", "fence", "reset", "actual"]
    rows = [json.loads(b) for b in (tmp_path / "phases.jsonl").read_bytes().splitlines()]
    assert [r["event"] for r in rows] == ["intent", "complete"]
    assert gate.peak_reserved == 9
    error = ValueError("actual-native-failure")

    def actual_failure():
        torch.cuda.synchronize = lambda _: (_ for _ in ()).throw(RuntimeError("cleanup-fence"))
        raise error

    with pytest.raises(ValueError) as caught:
        gate.phase("partial", actual_failure)
    assert caught.value is error
    assert any("cleanup-fence" in note for note in error.__notes__)
    final = json.loads((tmp_path / "phases.jsonl").read_bytes().splitlines()[-1])
    assert final["event"] == "failed" and final["error"]["message"] == str(error)
    assert final["secondary_errors"][0]["stage"] == "CUDA-fence"
    monkeypatch.setattr(gate, "check", lambda *_: probe.fail("refused-before-call"))
    before = list(calls)
    with pytest.raises(ValueError, match="refused"):
        gate.phase("not-entered", lambda: calls.append("must-not-enter"))
    assert calls == before


def properties(unit, *, pid="0", invocation="a" * 32):
    return {
        "LoadState": "loaded",
        "InvocationID": invocation,
        "ControlGroup": "/user.slice/app.slice/" + unit,
        "MainPID": pid,
        "SubState": "dead" if pid == "0" else "running",
    }


def test_safe_stop_never_signals_replacement_invocation(probe, monkeypatch):
    calls = []
    monkeypatch.setattr(probe, "service_show", lambda u: properties(u, invocation="b" * 32))
    monkeypatch.setattr(probe.subprocess, "run", lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError, match="replacement"):
        probe.safe_stop("owned.service", "a" * 32)
    assert calls == []


def test_pressure_limits_and_final_receipt_reservation_fail_closed(probe, tmp_path, monkeypatch):
    row = {
        "available": 100 * probe.GIB,
        "current": probe.GIB,
        "swap": 0,
        "events": {},
        "max": probe.CAP,
        "high": None,
        "swap_max": 0,
        "oom_group": 1,
    }
    assert not probe.pressure_stop(row, 24 * probe.GIB)
    for change in (
        {"current": probe.STOP},
        {"max": None},
        {"high": 3},
        {"swap": 1},
        {"oom_group": 0},
        {"events": {"max": 1}},
    ):
        assert probe.pressure_stop({**row, **change}, 24 * probe.GIB)
    monkeypatch.setattr(
        probe, "tree_inventory", lambda _: {"data": {"size_bytes": probe.TOTAL_CAP - 1}}
    )
    with pytest.raises(ValueError, match="capacity"):
        probe.final_inventory(tmp_path, reserved=4 * probe.MIB)


def test_unit_readback_checks_real_pid_startticks_invocation_and_properties(probe, monkeypatch):
    unit = "owned.service"
    ready = {"pid": 12, "startticks": 34, "boot_id": "boot"}
    props = {
        **properties(unit, pid="12"),
        "Type": "exec",
        "MemoryAccounting": "yes",
        "MemoryMax": str(probe.CAP),
        "MemoryHigh": "infinity",
        "MemorySwapMax": "0",
        "OOMPolicy": "kill",
        "KillMode": "control-group",
        "RuntimeMaxUSec": "30min",
        "TimeoutStopUSec": "10s",
    }
    monkeypatch.setattr(probe, "process_identity", lambda _: {"startticks": 34})
    monkeypatch.setattr(probe, "boot_id", lambda: "boot")
    monkeypatch.setattr(probe, "cgroup_membership", lambda _: props["ControlGroup"])
    monkeypatch.setattr(probe, "verify_unit_limits", lambda _: {"own": {"procs": [12]}})
    assert probe.unit_binding(props, ready, unit)["InvocationID"] == "a" * 32
    for change in (
        {"Type": "simple"},
        {"MemoryMax": "infinity"},
        {"OOMPolicy": "continue"},
        {"ControlGroup": "/wrong/" + unit},
        {"MainPID": "13"},
        {"InvocationID": "not-an-invocation"},
    ):
        with pytest.raises(ValueError):
            probe.unit_binding({**props, **change}, ready, unit)


def test_heartbeat_clock_after_read_reproduces_and_fixes_fresh_ticker_race(probe, monkeypatch):
    observed = {"clock": 10.0}
    pre_sample_clock = observed["clock"]

    def read_while_ticker_advances(_):
        observed["clock"] = 10.03
        return probe.encode({"nonce": "n", "monotonic": 10.02})

    monkeypatch.setattr(probe, "read", read_while_ticker_advances)
    row = probe.heartbeat_observation("fake", "n", clock=lambda: observed["clock"])
    assert pre_sample_clock - row["heartbeat_reported_monotonic"] < 0
    assert row["heartbeat_checked_monotonic"] == 10.03
    assert row["heartbeat_nonce_matches"] and 0 <= row["heartbeat_age_seconds"] < 10


@pytest.mark.parametrize(
    "reported,nonce,accepted",
    [(11, "n", False), (0, "n", False), (0.001, "n", True), (9.5, "alien", False)],
)
def test_heartbeat_future_stale_and_nonce_thresholds_remain_unchanged(
    probe, monkeypatch, reported, nonce, accepted
):
    monkeypatch.setattr(
        probe, "read", lambda _: probe.encode({"nonce": nonce, "monotonic": reported})
    )
    row = probe.heartbeat_observation("fake", "n", clock=lambda: 10.0)
    assert (row["heartbeat_nonce_matches"] and 0 <= row["heartbeat_age_seconds"] < 10) is accepted


def native_work_fixture(probe, monkeypatch):
    ready = {
        "nonce": "n",
        "pid": 4294967285,
        "startticks": 1,
        "boot_id": "boot",
        "source_sha256": "source",
        "profile_sha256": "profile",
    }
    props = properties("unique.service", pid=str(ready["pid"]))
    binding = {"ControlGroup": props["ControlGroup"], "InvocationID": props["InvocationID"]}
    cg = {
        "group": binding["ControlGroup"],
        "current": 100,
        "peak": 200,
        "max": probe.CAP,
        "high": None,
        "swap": 0,
        "swap_max": 0,
        "oom_group": 1,
        "events": {"max": 0, "oom": 0, "oom_kill": 0},
        "stat": {"anon": 100},
        "procs": [ready["pid"]],
    }
    monkeypatch.setattr(probe.os, "getpid", lambda: ready["pid"])
    monkeypatch.setenv("INVOCATION_ID", binding["InvocationID"])
    monkeypatch.setattr(probe, "process_identity", lambda _: {"startticks": 1})
    monkeypatch.setattr(probe, "cgroup_membership", lambda _: binding["ControlGroup"])
    monkeypatch.setattr(probe, "boot_id", lambda: "boot")
    monkeypatch.setattr(probe, "cgroup_snapshot", lambda _: cg)
    monkeypatch.setattr(probe, "service_show", lambda *_a, **_k: props)
    capture = {
        "schema_id": "native-work-end-cgroup-v1",
        "identity": probe.native_work_identity(ready, binding),
        "checked_monotonic": 10.0,
        "properties": props,
        "cgroup": cg,
    }
    return ready, binding, capture


@pytest.mark.parametrize("tamper", [None, "nonce", "snapshot_sha256", "extra", "missing"])
def test_success_only_barrier_binds_real_saved_capture_and_exact_ack(
    probe, tmp_path, monkeypatch, tamper
):
    ready, binding, expected = native_work_fixture(probe, monkeypatch)

    def parent_capture(_):
        probe.capture_native_work_end(tmp_path, ready, binding, "unique.service")
        if tamper:
            ack = json.loads((tmp_path / "native-work-ack.json").read_bytes())
            if tamper == "missing":
                del ack["source_sha256"]
            else:
                ack[tamper] = "forged"
            probe.replace(tmp_path / "native-work-ack.json", ack)

    monkeypatch.setattr(probe.time, "sleep", parent_capture)
    if tamper:
        with pytest.raises(ValueError, match="ACK"):
            probe.wait_native_work_capture(tmp_path, ready, binding, [])
    else:
        actual = probe.wait_native_work_capture(tmp_path, ready, binding, [])
        assert (
            actual["snapshot_sha256"]
            == probe.file_pin(tmp_path / "native-work-final-cgroup.json")["sha256"]
        )
        saved = json.loads((tmp_path / "native-work-final-cgroup.json").read_bytes())
        assert saved["cgroup"] == expected["cgroup"]
        assert "not post-ACK" in actual["scope"]


@pytest.mark.parametrize(
    "tamper", ["events_missing", "peak_cap", "bool_current", "wrong_invocation", "wrong_pid"]
)
def test_native_work_end_capture_rejects_incomplete_or_wrong_scope(probe, monkeypatch, tamper):
    _ready, _binding, capture = native_work_fixture(probe, monkeypatch)
    import copy

    altered = copy.deepcopy(capture)
    if tamper == "events_missing":
        del altered["cgroup"]["events"]["oom"]
    elif tamper == "peak_cap":
        altered["cgroup"]["peak"] = probe.STOP
    elif tamper == "bool_current":
        altered["cgroup"]["current"] = True
    elif tamper == "wrong_invocation":
        altered["properties"]["InvocationID"] = "b" * 32
    else:
        altered["cgroup"]["procs"] = [1]
    with pytest.raises(ValueError):
        probe.validate_native_work_capture(altered, capture["identity"])


@pytest.mark.parametrize(
    "variant",
    [
        "valid",
        "missing",
        "not_set",
        "bool",
        "below_snapshot",
        "at_stop",
        "swap",
        "wrong_invocation",
    ],
)
def test_retained_systemd_lifetime_peaks_are_separate_typed_and_consistent(
    probe, monkeypatch, variant
):
    _ready, binding, capture = native_work_fixture(probe, monkeypatch)
    props = {**properties("unique.service"), "MemoryPeak": "250", "MemorySwapPeak": "0"}
    if variant == "missing":
        del props["MemoryPeak"]
    elif variant == "not_set":
        props["MemoryPeak"] = "[not set]"
    elif variant == "bool":
        props["MemoryPeak"] = True
    elif variant == "below_snapshot":
        props["MemoryPeak"] = "199"
    elif variant == "at_stop":
        props["MemoryPeak"] = str(probe.STOP)
    elif variant == "swap":
        props["MemorySwapPeak"] = "1"
    elif variant == "wrong_invocation":
        props["InvocationID"] = "b" * 32
    if variant == "valid":
        actual = probe.retained_lifetime_peaks(props, capture["identity"], capture)
        assert actual["MemoryPeak"] == 250 and "no inference" in actual["scope"]
    else:
        with pytest.raises(ValueError):
            probe.retained_lifetime_peaks(props, capture["identity"], capture)


@pytest.mark.parametrize("outcome", ["success", "worker_failed", "missing_lifetime"])
def test_full_owned_fake_natural_exit_keeps_worker_failure_and_never_reads_empty_cgroup(
    probe, tmp_path, monkeypatch, outcome
):
    ready, _binding, capture = native_work_fixture(probe, monkeypatch)
    config = {
        "mode": "A",
        "nonce": "n",
        "device_uuid": "GPU-test",
        "boot_id": "boot",
        "source_pin": {"sha256": "source"},
        "profile_pin": {"sha256": "profile"},
        "capsule_pin": {"sha256": "capsule"},
        "source_inventory": {"sha256": "union"},
        "worker_entry": str(SOURCE),
        "artifact_kind": probe.PUBLIC_KIND,
    }
    ready.update(capsule_sha256="capsule", source_inventory_sha256="union")
    unit = "jev-stage-memory-a-n.service"
    group = "/user.slice/app.slice/" + unit
    binding = {"ControlGroup": group, "InvocationID": "a" * 32}
    identity = probe.native_work_identity(ready, binding)
    capture["cgroup"]["group"] = group
    monkeypatch.setattr(probe, "cgroup_membership", lambda _: group)
    # Admission is tested separately; this fixture isolates unchanged monitor/closure.
    monkeypatch.setattr(probe, "validate_stage_config", lambda *_: None)
    gone = []
    stopped = []
    snapshot_groups = []

    def show(*_a, **_k):
        terminal = bool(gone or (tmp_path / "native-work-ack.json").exists())
        props = {
            **properties(unit, pid="0" if terminal else str(ready["pid"])),
            "Result": "exit-code" if outcome == "worker_failed" else "success",
            "ExecMainCode": "1",
            "ExecMainStatus": "1" if outcome == "worker_failed" else "0",
            "MemoryPeak": "[not set]" if outcome == "missing_lifetime" else "250",
            "MemorySwapPeak": "0",
        }
        if terminal:
            props["ControlGroup"] = ""
            if not (tmp_path / "worker-receipt.json").exists():
                receipt = {
                    "schema_id": probe.RECEIPT_SCHEMA,
                    "profile_id": probe.PROFILE,
                    "allocator_policy": probe.CACHE_POLICY,
                    "kind": probe.PUBLIC_KIND,
                    "actual_native_forward_calls": 142,
                    "development_updates": 0,
                    "stage": "failed_stage_memory"
                    if outcome == "worker_failed"
                    else "complete_stage_fit_unqualified",
                    "completed_updates": 120,
                    "synchronized_updates": 120,
                    "optimizer_step_attempts": 120,
                    "model_loads": 1,
                    "verified_model_loads": 1,
                }
                if outcome == "worker_failed":
                    receipt["error"] = {
                        "type": "ValueError",
                        "message": "original-native-worker-failure",
                    }
                else:
                    receipt["native_work_end_capture"] = {
                        "snapshot_sha256": probe.file_pin(
                            tmp_path / "native-work-final-cgroup.json"
                        )["sha256"]
                    }
                probe.write(tmp_path / "worker-receipt.json", receipt)
        return props

    def snapshot(requested):
        snapshot_groups.append(requested)
        assert requested == group
        if gone or (tmp_path / "native-work-ack.json").exists():
            raise FileNotFoundError("exact owned cgroup naturally removed")
        return capture["cgroup"]

    def sample(*_):
        if outcome == "worker_failed":
            gone.append(True)
            raise ProcessLookupError("worker exited")
        probe.write(
            tmp_path / "native-work-ready.json", {"schema_id": "native-work-ready-v1", **identity}
        )
        return {
            "t": 10.0,
            "total": 130595860480,
            "available": 100 * probe.GIB,
            "current": 100,
            "swap": 0,
            "events": {},
            "max": probe.CAP,
            "high": None,
            "swap_max": 0,
            "oom_group": 1,
        }

    ticks = {"t": 1.0}

    def clock():
        ticks["t"] += 0.1
        return ticks["t"]

    monkeypatch.setattr(probe.time, "monotonic", clock)
    monkeypatch.setattr(probe.time, "sleep", lambda _: None)

    def fake_launch(*_a, **_k):
        probe.write(tmp_path / "ready-for-readback.json", ready)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(probe.subprocess, "run", fake_launch)
    monkeypatch.setattr(probe, "service_show", show)
    monkeypatch.setattr(probe, "safe_stop", lambda *_: stopped.append(True))
    monkeypatch.setattr(probe, "unit_binding", lambda *_: binding)
    monkeypatch.setattr(probe, "cg_path", lambda _: tmp_path / "absent-cgroup")
    monkeypatch.setattr(probe, "cgroup_snapshot", snapshot)
    monkeypatch.setattr(probe, "sample", sample)
    monkeypatch.setattr(
        probe,
        "heartbeat_observation",
        lambda *_: {"heartbeat_nonce_matches": True, "heartbeat_age_seconds": 0.01},
    )
    if outcome == "success":
        result = probe.arm_run(tmp_path, config)
        assert result["success"] and result["retained_lifetime_peaks"]["MemoryPeak"] == 250
    else:
        with pytest.raises(ValueError) as caught:
            probe.arm_run(tmp_path, config)
        if outcome == "worker_failed":
            assert (
                str(caught.value)
                == "Recorded worker failure (ValueError): original-native-worker-failure"
            )
        else:
            assert "Measured retained systemd lifetime peak" in str(caught.value)
    receipt = json.loads((tmp_path / "arm-receipt.json").read_bytes())
    assert receipt["closure_confirmed"] and stopped
    assert receipt["success"] is (outcome == "success")
    assert snapshot_groups and all(value == group for value in snapshot_groups)
    evidence = json.loads((tmp_path / "pre-stop-unit-evidence.json").read_bytes())
    assert evidence["bound_ControlGroup"] == group and evidence["properties"]["ControlGroup"] == ""
    assert "cgroup_read_error" in evidence and "cleanup_errors" not in receipt


def test_native_work_wait_deadline_and_heartbeat_failure_do_not_fake_ack(
    probe, tmp_path, monkeypatch
):
    ready, binding, _capture = native_work_fixture(probe, monkeypatch)
    ticks = {"t": 0.0}

    def clock():
        ticks["t"] += 5.0
        return ticks["t"]

    monkeypatch.setattr(probe.time, "monotonic", clock)
    monkeypatch.setattr(probe.time, "sleep", lambda _: None)
    with pytest.raises(ValueError, match="acknowledgement"):
        probe.wait_native_work_capture(tmp_path, ready, binding, [])
    assert not (tmp_path / "native-work-ack.json").exists()


@pytest.fixture
def native_tools(probe, tmp_path, monkeypatch):
    environment = probe.native_cache_environment(tmp_path)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    for key in list(probe.os.environ):
        if key.startswith(("TRITON_", "TORCHINDUCTOR_")) and key not in environment:
            monkeypatch.delenv(key)
    for key in (
        "LD_PRELOAD",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "CPATH",
        "C_INCLUDE_PATH",
        "CPLUS_INCLUDE_PATH",
        "LIBRARY_PATH",
        "GCC_EXEC_PREFIX",
        "COMPILER_PATH",
        "GCC_SPECS",
        "CFLAGS",
        "CPPFLAGS",
        "LDFLAGS",
        "PTXAS_OPTIONS",
        "DISABLE_PTXAS_OPT",
        "NVPTX_ENABLE_DUMP",
        "USE_IR_LOC",
    ):
        monkeypatch.delenv(key, raising=False)
    native_tmp = Path(environment["TMPDIR"])
    native_tmp.mkdir(parents=True)
    sources = {
        key: str(tmp_path / (key + ".py"))
        for key in ("platform", "ctypes", "build", "driver", "compiler", "knobs")
    }
    driver = tmp_path / "driver.c"
    driver.write_bytes(b"original public utility C\n")
    sources["cuda_utils_source"] = str(driver)
    canonical = tmp_path / "python-versioned/bin/python3.13"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"public fake Python identity\n")
    alias = tmp_path / "python-unversioned"
    alias.symlink_to(canonical.parent.parent, target_is_directory=True)
    lexical = alias / "bin/python3.13"
    entrypoint = tmp_path / "venv/bin/python"
    entrypoint.parent.mkdir(parents=True)
    entrypoint.symlink_to(lexical)
    binding = {
        "schema_id": "native-python-entrypoint-alias-binding-v1",
        "entrypoint": str(entrypoint),
        "native_file_target": str(lexical),
        "canonical_executable": str(canonical),
    }
    binding["components"] = probe.python_path_components(
        [binding[k] for k in ("entrypoint", "native_file_target", "canonical_executable")]
    )
    monkeypatch.setattr(
        probe,
        "sys",
        SimpleNamespace(
            executable=str(entrypoint), modules=probe.sys.modules, _getframe=probe.sys._getframe
        ),
    )
    paths = {
        "file": "/usr/bin/file",
        "cc": environment["CC"],
        "ldconfig": "/sbin/ldconfig",
        "python": str(canonical),
        "ptxas": str(tmp_path / "ptxas"),
        "ptxas_blackwell": str(tmp_path / "ptxas-blackwell"),
    }
    fake_pin = {"sha256": "a" * 64, "size_bytes": 12}
    runtime = {
        "python_entrypoint": binding,
        "tools": {
            name: {"path": path, "resolved": path, "pin": fake_pin} for name, path in paths.items()
        },
        "sources": sources,
        "files": {str(driver): probe.file_pin(driver)},
        "triton_root": str(tmp_path / "triton"),
        "python_include": str(tmp_path / "python-include"),
        "extension_suffix": ".cpython-313-aarch64-linux-gnu.so",
        "cuda_library_directories": ["/lib/aarch64-linux-gnu", "/usr/lib/aarch64-linux-gnu"],
    }
    runtime["tools"]["python"]["pin"] = probe.file_pin(canonical)
    original_pin = probe.file_pin
    monkeypatch.setattr(
        probe,
        "file_pin",
        lambda path: (
            fake_pin
            if str(path) in paths.values() and str(path) != paths["python"]
            else original_pin(path)
        ),
    )
    tools = probe.NativeTools(tmp_path, {"ControlGroup": "/owned"})
    tools.runtime, tools.environment = runtime, environment
    source = native_tmp / "one/cuda_utils.c"
    source.parent.mkdir()
    source.write_bytes(driver.read_bytes())
    target = source.parent / ("cuda_utils" + runtime["extension_suffix"])
    gcc = [
        paths["cc"],
        str(source),
        "-O3",
        "-shared",
        "-fPIC",
        "-Wno-psabi",
        "-o",
        str(target),
        "-l:libcuda.so.1",
        "-L" + str(Path(runtime["triton_root"]) / "backends/nvidia/lib"),
        "-L/lib/aarch64-linux-gnu",
        "-I" + str(Path(runtime["triton_root"]) / "backends/nvidia/include"),
        "-I" + str(source.parent),
        "-I" + runtime["python_include"],
    ]
    ptx = native_tmp / "tiny.ptx"
    ptx.write_text(".version 9.0\n.target sm_121a\n")
    assembler = [
        paths["ptxas_blackwell"],
        "-lineinfo",
        "-v",
        "--regAllocOptLevel=2",
        "--gpu-name=sm_121a",
        str(ptx),
        "-o",
        str(ptx) + ".o",
    ]
    return tools, paths, sources, gcc, assembler


@pytest.mark.parametrize(
    "kind",
    [
        "ldconfig",
        "ldconfig_ctypes",
        "file",
        "cc",
        "ptxas_version",
        "ptxas_blackwell_version",
        "ptxas",
        "ptxas_fmad",
    ],
)
def test_native_tools_original_source_commands_are_offered_without_execution(
    native_tools, probe, kind
):
    tools, paths, sources, gcc, assembler = native_tools
    env, callers = None, set()
    if kind.startswith("ldconfig"):
        argv = [paths["ldconfig"], "-p"]
        if kind.endswith("ctypes"):
            env, callers = (
                {"LC_ALL": "C", "LANG": "C"},
                {(sources["ctypes"], "_findSoname_ldconfig")},
            )
        else:
            callers = {(sources["driver"], "libcuda_dirs")}
    elif kind == "file":
        argv = ["file", "-b", tools.runtime["python_entrypoint"]["native_file_target"]]
        env = dict(probe.os.environ, LC_ALL="C")
        callers = {(sources["platform"], "_syscmd_file"), (sources["build"], "platform_key")}
    elif kind == "cc":
        argv, callers = gcc, {(sources["build"], "_build")}
    elif kind.endswith("version"):
        argv = [paths[kind.removesuffix("_version")], "--version"]
        callers = {(sources["knobs"], "from_path")}
    else:
        argv = list(assembler)
        if kind.endswith("fmad"):
            argv.insert(2, "--fmad=false")
        callers = {(sources["compiler"], "make_cubin")}
    result = tools.classify((argv[0], argv, None, env), callers)
    assert result["argv"] == argv and result["environment_values_recorded"] is False
    assert tools.authorizations == 0


@pytest.mark.parametrize(
    "mutation",
    [
        "shell",
        "caller",
        "cwd",
        "env",
        "plugin",
        "outside",
        "changed_C",
        "existing_output",
        "symlink",
        "cache_drift",
        "custom_options",
        "foreign_ptxas",
        "ptx_output",
        "tool_drift",
        "override",
    ],
)
def test_native_tools_reject_unreviewed_process_shapes_without_execution(
    native_tools, probe, tmp_path, monkeypatch, mutation
):
    tools, paths, sources, gcc, assembler = native_tools
    argv, cwd, env = list(gcc), None, None
    callers = {(sources["build"], "_build")}
    if mutation == "shell":
        argv = ["/bin/sh", "-c", "true"]
    elif mutation == "caller":
        callers = set()
    elif mutation == "cwd":
        cwd = str(tmp_path)
    elif mutation == "env":
        env = dict(probe.os.environ)
    elif mutation == "plugin":
        argv.append("-fplugin=/tmp/foreign.so")
    elif mutation == "outside":
        argv[1] = str(tmp_path / "outside.c")
    elif mutation == "changed_C":
        Path(argv[1]).write_text("changed code")
    elif mutation == "existing_output":
        Path(argv[7]).write_text("old output")
    elif mutation == "symlink":
        link = Path(argv[1]).with_name("link.c")
        link.symlink_to(Path(argv[1]))
        argv[1] = str(link)
    elif mutation == "cache_drift":
        monkeypatch.setenv("TMPDIR", "/tmp")
    elif mutation in {"custom_options", "foreign_ptxas", "ptx_output"}:
        argv, callers = list(assembler), {(sources["compiler"], "make_cubin")}
        if mutation == "custom_options":
            argv.insert(2, "--options-file=/tmp/extra")
        elif mutation == "foreign_ptxas":
            argv[0] = "/tmp/ptxas-blackwell"
        else:
            argv[-1] = str(tmp_path / "other.o")
    elif mutation == "tool_drift":
        previous = probe.file_pin
        monkeypatch.setattr(
            probe,
            "file_pin",
            lambda path: (
                {"sha256": "b" * 64, "size_bytes": 12}
                if str(path) == paths["cc"]
                else previous(path)
            ),
        )
    else:
        monkeypatch.setenv("TRITON_BUILD_IMPL", "foreign:build")
    with pytest.raises(ValueError):
        tools.classify((argv[0], argv, cwd, env), callers)
    assert tools.authorizations == 0


def test_native_tools_pending_runtime_and_capacity_refuse_before_entry(
    probe, native_tools, monkeypatch
):
    tools, _paths, sources, gcc, _assembler = native_tools
    tools.runtime = None
    with pytest.raises(ValueError, match="verified"):
        tools.classify((gcc[0], gcc, None, None), set())
    tools.runtime = {"sources": sources}
    tools.authorizations = 1024
    with pytest.raises(ValueError, match="count"):
        tools.audit((gcc[0], gcc, None, None))


def test_native_tools_arm_capacity_refusal_preserves_original_error_before_Popen(
    probe, native_tools, monkeypatch
):
    tools, _paths, sources, gcc, _assembler = native_tools
    monkeypatch.setattr(probe, "cgroup_membership", lambda _: "/owned")
    monkeypatch.setattr(tools, "callers", lambda: {(sources["build"], "_build")})
    error = ValueError("Bounded arm evidence allowance")

    def refused(_):
        raise error

    monkeypatch.setattr(probe, "arm_bound", refused)
    with pytest.raises(ValueError) as raised:
        tools.audit((gcc[0], gcc, None, None))
    assert raised.value is error and tools.authorizations == 0


@pytest.mark.parametrize(
    "mutation", [None, "foreign_root", "unbound_child", "wrong_type", "unbound_nested"]
)
def test_native_tools_pybind_submodules_bind_pinned_root_and_actual_parent_attributes(
    probe, native_tools, monkeypatch, mutation
):
    from types import ModuleType

    tools, _paths, _sources, _gcc, _assembler = native_tools
    root_name = "triton._C.libtriton"
    root = ModuleType(root_name)
    root.__file__ = str(Path(tools.runtime["triton_root"]) / "_C/libtriton.so")
    tools.runtime["files"][root.__file__] = {"sha256": "a" * 64, "size_bytes": 1}
    root.ir = ModuleType(root_name + ".ir")
    root.ir.passes = ModuleType(root_name + ".ir.passes")
    modules = {
        root_name: root,
        root_name + ".ir": root.ir,
        root_name + ".ir.passes": root.ir.passes,
    }
    if mutation == "foreign_root":
        root.__file__ = "/tmp/foreign/libtriton.so"
    elif mutation == "unbound_child":
        modules[root_name + ".ir"] = ModuleType(root_name + ".ir")
    elif mutation == "wrong_type":
        root.ir = SimpleNamespace(passes=root.ir.passes)
        modules[root_name + ".ir"] = root.ir
    elif mutation == "unbound_nested":
        modules[root_name + ".ir.passes"] = ModuleType(root_name + ".ir.passes")
    monkeypatch.setattr(probe, "sys", SimpleNamespace(modules=modules))
    if mutation:
        with pytest.raises(ValueError, match="origin|owner|identity"):
            tools.origins()
    else:
        result = tools.origins()
        assert result[root_name] == root.__file__
        assert result[root_name + ".ir"] == {
            "native_root": root.__file__,
            "module_attribute": root_name + ".ir",
        }
        assert result[root_name + ".ir.passes"] == {
            "native_root": root.__file__,
            "module_attribute": root_name + ".ir.passes",
        }


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "canonical_argument",
        "other_alias",
        "entrypoint_spelling",
        "parent_retarget",
        "same_target_new_link",
        "entrypoint_retarget",
        "directory_to_alias",
        "binary_changed",
    ],
)
def test_native_python_exact_lexical_entrypoint_and_parent_alias_binding(
    probe, native_tools, tmp_path, monkeypatch, mutation
):
    tools, paths, sources, _gcc, _assembler = native_tools
    binding = tools.runtime["python_entrypoint"]
    target = binding["native_file_target"]
    alias = Path(target).parent.parent
    canonical = Path(paths["python"])
    entrypoint = Path(binding["entrypoint"])
    if mutation == "canonical_argument":
        target = str(canonical)
    elif mutation == "other_alias":
        other = tmp_path / "other-alias"
        other.symlink_to(canonical.parent.parent, target_is_directory=True)
        target = str(other / "bin/python3.13")
    elif mutation == "entrypoint_spelling":
        probe.sys.executable = str(canonical)
    elif mutation == "parent_retarget":
        foreign = tmp_path / "foreign/bin/python3.13"
        foreign.parent.mkdir(parents=True)
        foreign.write_bytes(canonical.read_bytes())
        alias.unlink()
        alias.symlink_to(foreign.parent.parent, target_is_directory=True)
    elif mutation == "same_target_new_link":
        # Allocate a distinct inode while the original still exists.
        replacement = alias.with_name("replacement")
        replacement.symlink_to(canonical.parent.parent, target_is_directory=True)
        replacement.replace(alias)
    elif mutation == "entrypoint_retarget":
        entrypoint.unlink()
        entrypoint.symlink_to(canonical)
    elif mutation == "directory_to_alias":
        directory = entrypoint.parent
        saved = directory.with_name("saved-bin")
        directory.rename(saved)
        directory.symlink_to(saved, target_is_directory=True)
    elif mutation == "binary_changed":
        original = canonical.stat()
        canonical.write_bytes(b"X" * original.st_size)
        probe.os.utime(canonical, ns=(original.st_atime_ns, original.st_mtime_ns))
    argv = ["file", "-b", target]
    args = ("file", argv, None, dict(probe.os.environ, LC_ALL="C"))
    callers = {(sources["platform"], "_syscmd_file"), (sources["build"], "platform_key")}
    if mutation is None:
        row = tools.classify(args, callers)
        assert row["argv"] == argv and row["kind"] == "file"
        # The original stdlib follows only terminal symlinks, retaining parent alias.
        import platform

        assert platform._follow_symlinks(str(entrypoint)) == target
        assert str(Path(target).resolve()) == str(canonical)
    else:
        with pytest.raises(ValueError, match="Python|pinned canonical"):
            tools.classify(args, callers)
    assert tools.authorizations == 0


def test_native_python_alias_second_snapshot_refuses_change_during_hash(
    probe, native_tools, monkeypatch
):
    tools, paths, _sources, _gcc, _assembler = native_tools
    binding = tools.runtime["python_entrypoint"]
    alias = Path(binding["native_file_target"]).parent.parent
    original_pin = probe.file_pin

    def changed_after_hash(path):
        result = original_pin(path)
        if str(path) == paths["python"]:
            replacement = alias.with_name("changed-during-hash")
            replacement.symlink_to(Path(path).parent.parent, target_is_directory=True)
            replacement.replace(alias)
        return result

    monkeypatch.setattr(probe, "file_pin", changed_after_hash)
    with pytest.raises(ValueError, match="changed during"):
        probe.verify_python_entrypoint(tools.runtime, binding["native_file_target"])


def test_native_python_directory_mtime_is_not_unrelated_file_admission(probe, native_tools):
    tools, _paths, _sources, _gcc, _assembler = native_tools
    binding = tools.runtime["python_entrypoint"]
    sibling = Path(binding["canonical_executable"]).with_name("unrelated-public.txt")
    sibling.write_text("ordinary unrelated directory entry\n")
    probe.verify_python_entrypoint(tools.runtime, binding["native_file_target"])
    assert all(
        "mtime_ns" not in row
        for row in binding["components"].values()
        if probe.stat.S_ISDIR(row["mode"])
    )


@pytest.fixture
def cache_case(probe, tmp_path, monkeypatch):
    def tensor(pointer, *, grad=None):
        return SimpleNamespace(
            device="cuda:0",
            dtype="torch.float32",
            shape=(2, 3),
            stride=lambda: (3, 1),
            storage_offset=lambda: 0,
            requires_grad=True,
            _version=7,
            grad=grad,
            untyped_storage=lambda: SimpleNamespace(data_ptr=lambda: pointer, nbytes=lambda: 24),
        )

    gradient = tensor(200)
    parameter = tensor(100, grad=gradient)
    model = SimpleNamespace(lm=SimpleNamespace(named_parameters=lambda: [("q.lora", parameter)]))
    calls, control = [], {"recover": True, "mutate": None}
    cuda = {
        "allocated": 24 * probe.GIB,
        "reserved": 29 * probe.GIB,
        "peak_allocated": 28 * probe.GIB,
        "peak_reserved": 29 * probe.GIB,
        "free": 42 * probe.GIB,
        "total": 130595860480,
        "backend": "native",
        "fraction": 0.3,
        "stats": {"active_bytes.all.current": 25 * probe.GIB},
    }

    def release():
        calls.append("empty-cache")
        if control["recover"]:
            cuda.update(reserved=26 * probe.GIB, free=42 * probe.GIB)
        # Allocator active blocks can decrease without live tensor bytes changing.
        cuda["stats"] = {"active_bytes.all.current": 24 * probe.GIB}
        if control["mutate"] is not None:
            control["mutate"]()

    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            synchronize=lambda _: calls.append("fence"),
            empty_cache=release,
            reset_peak_memory_stats=lambda _: calls.append("original-phase-reset"),
        )
    )
    gate = probe.StageGate(
        torch, tmp_path, {"nonce": "n", "boot_id": "b"}, {"ControlGroup": "g"}, {"fraction": 0.3}
    )
    gate.loaded = True
    monkeypatch.setattr(gate, "cuda", lambda: {**cuda, "stats": dict(cuda["stats"])})
    monkeypatch.setattr(
        probe,
        "host_memory",
        lambda: {"MemTotal": 130595860480, "MemAvailable": 120000000000, "boot_id": "b"},
    )
    monkeypatch.setattr(probe, "phase_cpu", lambda: {})
    monkeypatch.setattr(
        probe,
        "cgroup_snapshot",
        lambda _: {
            "current": probe.GIB,
            "peak": probe.GIB,
            "swap": 0,
            "events": {},
            "max": probe.CAP,
            "high": None,
            "swap_max": 0,
            "oom_group": 1,
        },
    )

    def backward():
        calls.append("backward")
        cuda["free"] = 39 * probe.GIB
        return "original-result"

    def run(mode="A", fn=backward):
        return probe.backward_cache_release(
            torch, model, gate, tmp_path, fn, mode=mode, case_id="original-row", step=1
        )

    return SimpleNamespace(
        torch=torch,
        model=model,
        parameter=parameter,
        gradient=gradient,
        gate=gate,
        calls=calls,
        control=control,
        cuda=cuda,
        run=run,
    )


@pytest.mark.parametrize("mode", ["A", "C"])
@pytest.mark.parametrize("recover", [False, True])
def test_H12_fixed_release_preserves_original_next_gate_and_peaks(
    probe, cache_case, tmp_path, mode, recover
):
    c = cache_case
    c.control["recover"] = recover
    assert c.gate.phase("native-backward", lambda: c.run(mode)) == "original-result"
    assert c.calls.count("backward") == c.calls.count("empty-cache") == 1
    assert c.calls.count("original-phase-reset") == 1
    assert c.calls.index("backward") < c.calls.index("empty-cache")
    assert c.calls[c.calls.index("backward") + 1] == "fence"
    if recover:
        c.gate.phase("native-grad-clip", lambda: c.calls.append("clip"))
        assert c.calls.count("clip") == 1
    else:
        with pytest.raises(ValueError, match="Native allocator/device guard refused"):
            c.gate.phase("native-grad-clip", lambda: c.calls.append("clip"))
        assert "clip" not in c.calls
    assert c.gate.peak_allocated == 28 * probe.GIB
    assert c.gate.peak_reserved == 29 * probe.GIB
    rows = [
        json.loads(line) for line in (tmp_path / "cache-releases.jsonl").read_bytes().splitlines()
    ]
    assert [r["event"] for r in rows] == ["intent", "complete"]
    assert rows[-1]["mode"] == mode and rows[-1]["success"]
    assert rows[-1]["before"]["storage_identity"] == rows[-1]["after"]["storage_identity"]
    assert rows[-1]["before"]["cuda"]["allocated"] == rows[-1]["after"]["cuda"]["allocated"]


@pytest.mark.parametrize(
    "mutation", ["allocated", "parameter", "gradient", "peak_allocated", "peak_reserved"]
)
def test_H12_release_refuses_live_identity_or_high_water_loss(
    probe, cache_case, tmp_path, mutation
):
    c = cache_case

    def mutate():
        if mutation == "allocated":
            c.cuda["allocated"] -= 1
        elif mutation == "parameter":
            c.parameter.storage_offset = lambda: 1
        elif mutation == "gradient":
            c.gradient._version += 1
        else:
            c.cuda[mutation] -= 1

    c.control["mutate"] = mutate
    with pytest.raises(ValueError, match="Cache release"):
        c.run()
    assert c.calls == ["backward", "fence", "empty-cache"]
    assert c.gate.peak_allocated == 28 * probe.GIB and c.gate.peak_reserved == 29 * probe.GIB
    final = json.loads((tmp_path / "cache-releases.jsonl").read_bytes().splitlines()[-1])
    assert final["event"] == "failed" and final["after"] and not final["success"]


@pytest.mark.parametrize("stage", ["backward", "fence", "API"])
def test_H12_native_failure_never_retries_or_replaces_primary(probe, cache_case, tmp_path, stage):
    c = cache_case
    primary = RuntimeError("actual-" + stage)

    def failure(*_args):
        raise primary

    if stage == "fence":
        c.torch.cuda.synchronize = failure
    elif stage == "API":
        c.torch.cuda.empty_cache = failure
    with pytest.raises(RuntimeError) as caught:
        c.run(fn=failure) if stage == "backward" else c.run()
    assert caught.value is primary
    if stage != "API":
        assert not (tmp_path / "cache-releases.jsonl").exists()
    else:
        final = json.loads((tmp_path / "cache-releases.jsonl").read_bytes().splitlines()[-1])
        assert final["event"] == "failed" and not final["API_returned"]


def test_H12_evidence_failure_preserves_cache_error_and_before_peaks(
    probe, cache_case, tmp_path, monkeypatch
):
    c = cache_case
    primary = RuntimeError("native-empty-cache-error")
    c.torch.cuda.empty_cache = lambda: (_ for _ in ()).throw(primary)
    original = probe.append

    def append(path, row, limit):
        if row.get("event") == "failed":
            raise OSError("cache-journal-fsync")
        original(path, row, limit)

    monkeypatch.setattr(probe, "append", append)
    with pytest.raises(RuntimeError) as caught:
        c.run()
    assert caught.value is primary and "cache-journal-fsync" in primary.__notes__[0]
    assert c.gate.peak_allocated == 28 * probe.GIB and c.gate.peak_reserved == 29 * probe.GIB
    assert len((tmp_path / "cache-releases.jsonl").read_bytes().splitlines()) == 1


def test_capsule_has_only_stdlib_imports_and_no_driver_or_CLI(probe):
    tree = ast.parse(SOURCE.read_text())
    imports = [
        n.module if isinstance(n, ast.ImportFrom) else a.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.Import, ast.ImportFrom))
        for a in (n.names if isinstance(n, ast.Import) else [None])
    ]
    assert not any(
        name
        and name.split(".")[0]
        in {"torch", "triton", "transformers", "modal", "playwright", "httpx"}
        for name in imports
    )
    assert not any(
        isinstance(n, ast.If)
        and isinstance(n.test, ast.Compare)
        and "__name__" in ast.unparse(n.test)
        for n in tree.body
    )
    assert not hasattr(probe, "worker") and not hasattr(probe, "diagnostic_steps")
    assert probe.PROFILE != "gb10-public-native-stage-peak-research-post-backward-cache-v4"


def test_capsule_retains_entire_native_tools_cache_and_gate_predicates(probe):
    optional_frozen_source()

    def blocks(path):
        source = path.read_text()
        return {
            n.name: ast.get_source_segment(source, n)
            for n in ast.parse(source).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        }

    old, new = blocks(FROZEN), blocks(SOURCE)
    for name in [
        "NativeTools",
        "allocator_cap",
        "allocator_origins",
        "research_floor",
        "pressure_stop",
        "backward_cache_release",
        "cache_storage_identity",
        "capture_native_work_end",
        "wait_native_work_capture",
        "retained_lifetime_peaks",
        "closed_unit_metadata",
        "terminal_after_proc_disappearance",
        "safe_stop",
    ]:
        assert new[name] == old[name], name
    old_gate = next(n for n in ast.parse(old["Research"]).body if isinstance(n, ast.ClassDef))
    new_gate = next(n for n in ast.parse(new["StageGate"]).body if isinstance(n, ast.ClassDef))
    for name in ["__init__", "cuda", "check", "phase"]:
        a = next(n for n in old_gate.body if isinstance(n, ast.FunctionDef) and n.name == name)
        z = next(n for n in new_gate.body if isinstance(n, ast.FunctionDef) and n.name == name)
        if name == "check":
            # Only stage journal schema/profile/event metadata changes.
            a = ast.get_source_segment(old["Research"], a)
            z = ast.get_source_segment(new["StageGate"], z)
            z = z.replace('"schema_id": PHASE_SCHEMA,', "").replace('"profile_id": PROFILE,', "")
            z = z.replace('"event": "stage-gate"', '"event": "research-gate"')
            assert ast.dump(ast.parse(z), include_attributes=False) == ast.dump(
                ast.parse(a), include_attributes=False
            )
    assert (probe.CAP, probe.STOP, probe.FILE_CAP) == (
        40 * 1024**3,
        int(39.5 * 1024**3),
        64 * 1024**2,
    )


def test_optional_frozen_source_skips_missing_but_refuses_corruption(tmp_path, monkeypatch):
    path = tmp_path / "optional_frozen_H12.py"
    monkeypatch.setitem(globals(), "FROZEN", path)
    with pytest.raises(pytest.skip.Exception):
        optional_frozen_source()
    path.write_text("# corrupt source\n")
    with pytest.raises(AssertionError, match="Pinned H12 research source changed"):
        optional_frozen_source()


def test_unit_argv_uses_explicit_producer_with_original_properties(probe, tmp_path):
    worker = probe.ROOT / "scripts/future_reviewed_stage_producer.py"
    argv = probe.unit_argv(
        "owned.service",
        tmp_path / "config.json",
        "GPU-test",
        tmp_path / "native.log",
        worker_entry=worker,
    )
    assert argv[-6:] == [
        "/home/albert_hsieh/ComfyUI/.venv/bin/python",
        "-B",
        "-s",
        str(worker),
        "--worker",
        str(tmp_path / "config.json"),
    ]
    for item in [
        "MemoryMax=42949672960",
        "MemorySwapMax=0",
        "RuntimeMaxSec=1800s",
        "OOMPolicy=kill",
    ]:
        assert "--property=" + item in argv


@pytest.mark.parametrize("kind,updates,forwards", [("public", 120, 142), ("mixed", 129, 156)])
def test_completion_contract_requires_new_exact_regime(probe, kind, updates, forwards):
    chosen = probe.PUBLIC_KIND if kind == "public" else probe.MIXED_KIND
    assert probe.completion_contract(chosen) == {
        "kind": chosen,
        "updates": updates,
        "forwards": forwards,
    }
    with pytest.raises(ValueError):
        probe.completion_contract("memory_probe_only")


@pytest.fixture
def stage_config(probe, tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    capsule = scripts / "jev_stage_memory_profile.py"
    capsule.write_text("# public pure capsule fixture\n")
    monkeypatch.setattr(probe, "__file__", str(capsule))
    base = scripts / "base.py"
    base.write_text("# public base source fixture\n")
    worker = scripts / "future_stage_producer.py"
    worker.write_text("# public producer fixture\n")
    validator = scripts / "future_artifact_validator.py"
    validator.write_text("# public validator fixture\n")
    area = tmp_path / "artifacts/jev-runtime/test"
    area.mkdir(parents=True)

    def ref(path):
        return {"path": path.relative_to(tmp_path).as_posix(), **probe.file_pin(path)}

    original = area / "base.json"
    original.write_bytes(probe.encode({"files": {"scripts/base.py": probe.file_pin(base)}}))
    extensions = {"scripts/" + p.name: probe.file_pin(p) for p in [capsule, worker, validator]}
    files = {"scripts/base.py": probe.file_pin(base), **extensions}
    union = {
        "schema_id": probe.SOURCE_SCHEMA,
        "producer_ready": True,
        "base_inventory": ref(original),
        "extensions": extensions,
        "files": files,
        "identity_sha256": probe.sha(probe.encode(files)),
    }
    union_path = area / "union.json"
    union_path.write_bytes(probe.encode(union))
    profile = {
        "schema_id": "jev-stage-memory-profile-v1",
        "profile_id": probe.PROFILE,
        "cache_policy": probe.CACHE_POLICY,
        "source_inventory": ref(union_path),
        "caps": {
            "unit": probe.CAP,
            "allocator": probe.CAP,
            "device_free": probe.CAP + probe.GIB,
            "stop": probe.STOP,
            "swap": 0,
            "wall_seconds": 1800,
            "campaign": probe.TOTAL_CAP,
            "arm": probe.ARM_CAP,
            "file_and_log": probe.FILE_CAP,
        },
    }
    profile_path = area / "profile.json"
    profile_path.write_bytes(probe.encode(profile))
    campaign = area / "fresh"
    campaign.mkdir()
    output = campaign / "A"
    output.mkdir()
    config = {
        "schema_id": "jev-stage-memory-config-v1",
        "profile_id": probe.PROFILE,
        "artifact_kind": probe.PUBLIC_KIND,
        "mode": "A",
        "nonce": "a" * 32,
        "boot_id": "public-boot",
        "source_pin": probe.file_pin(worker),
        "profile_pin": probe.file_pin(profile_path),
        "capsule_pin": probe.file_pin(capsule),
        "worker_entry": str(worker),
        "profile_path": str(profile_path),
        "source_inventory": ref(union_path),
        "campaign_output": str(campaign),
        "campaign_root": str(area),
        "device_uuid": "GPU-public-fixture",
    }
    return SimpleNamespace(
        config=config,
        output=output,
        profile=profile,
        profile_path=profile_path,
        union=union,
        union_path=union_path,
        capsule=capsule,
        worker=worker,
    )


@pytest.mark.parametrize(
    "tamper", [None, "old_schema", "cap", "policy", "worker", "capsule", "pending", "extra_source"]
)
def test_stage_admission_authenticates_all_new_identities_before_launch(
    probe, stage_config, tamper
):
    c = stage_config
    if tamper == "old_schema":
        c.config["schema_id"] = "gb10-memory-research-source-inventory-v1"
    elif tamper in {"cap", "policy"}:
        if tamper == "cap":
            c.profile["caps"]["device_free"] -= 1
        else:
            c.profile["cache_policy"] = "no-release"
        c.profile_path.write_bytes(probe.encode(c.profile))
        c.config["profile_pin"] = probe.file_pin(c.profile_path)
    elif tamper in {"worker", "capsule"}:
        getattr(c, tamper).write_text("# drift\n")
    elif tamper in {"pending", "extra_source"}:
        if tamper == "pending":
            c.union["producer_ready"] = False
        else:
            c.union["files"]["scripts/invented.py"] = {"sha256": "x", "size_bytes": 1}
            c.union["identity_sha256"] = probe.sha(probe.encode(c.union["files"]))
        c.union_path.write_bytes(probe.encode(c.union))
        c.config["source_inventory"].update(probe.file_pin(c.union_path))
        c.profile["source_inventory"] = c.config["source_inventory"]
        c.profile_path.write_bytes(probe.encode(c.profile))
        c.config["profile_pin"] = probe.file_pin(c.profile_path)
    if tamper is None:
        assert probe.validate_stage_config(c.config, c.output) == c.profile
    else:
        with pytest.raises(ValueError):
            probe.validate_stage_config(c.config, c.output)


@pytest.mark.parametrize("tamper", [False, True])
def test_allocator_file_authentication_precedes_optional_dependency_admission(
    probe, tmp_path, monkeypatch, tamper
):
    source = tmp_path / "public_runtime_source.py"
    source.write_text("# small public pure runtime fixture\n")
    runtime = {"files": [{"path": str(source), "pin": probe.file_pin(source)}]}
    calls = []
    optional = {"public": "fixture"}
    monkeypatch.setattr(
        probe, "verify_optional_dependency_files", lambda value: calls.append(value)
    )
    if tamper:
        source.write_text("# changed source\n")
        with pytest.raises(ValueError, match="Exact native allocator runtime"):
            probe.verify_allocator_files(runtime, optional)
        assert calls == []
    else:
        probe.verify_allocator_files(runtime, optional)
        assert calls == [optional]
