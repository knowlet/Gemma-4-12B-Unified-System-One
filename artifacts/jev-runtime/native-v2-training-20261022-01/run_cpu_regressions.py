"""Own CPU-only regressions with a separate host floor; default dry.

The 16GiB host floor is for this at-most3GiB CPU unit only. It never changes
the12B training gate or permits CUDA use. Preserve earlier26.119GB refusals.
"""

import argparse
import hashlib
import importlib.util
import json
import re
import resource
import secrets
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[2]
PREVIOUS = ROOT / "artifacts/jev-runtime/v2-action-training-prepare-20261021-01"
GIB = 1024**3
CAP, CPU_RESERVE = 3 * GIB, 16 * GIB


def pin(path):
    blob = path.read_bytes()
    return {"sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)}


def save(output, name, value):
    with (output / name).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def command(*argv, timeout=15):
    return subprocess.run(argv, capture_output=True, text=True, check=True, timeout=timeout).stdout


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--attempt", default="tests-03")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        print("DRY_NO_CPU_TEST_OR_MODEL; explicit --run required")
        return
    assert re.fullmatch(r"tests-[0-9]{2}", args.attempt)
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
    output = HERE / args.attempt
    unit = "jev-native-v2-check-" + args.attempt + "-20261022.service"
    cgroup = "/user.slice/user-1001.slice/user@1001.service/app.slice/" + unit
    nonce = "CPU-only3GiB-16GiB-host-" + secrets.token_hex(16)
    index = json.loads((PREVIOUS / "selected-evidence-index-01.json").read_bytes())
    ref = next(r for r in index["sources"] if r["path"].endswith("/run_compiler_tests.py"))
    path = ROOT / ref["path"]
    assert not any(p.is_symlink() for p in (path, *path.parents))
    assert pin(path) == {k: ref[k] for k in ("sha256", "size_bytes")}
    spec = importlib.util.spec_from_file_location("_cpu_status_and_host_only", path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    # Use only status/host/observe from the prior authority, never change its
    # globals or call its26.119GB watch controller under this new profile.
    before = guard.host()
    assert before["mem_available_bytes"] >= CPU_RESERVE + CAP + 64 * 1024**2, before
    assert guard.status(unit)["LoadState"] == "not-found"
    output.mkdir(mode=0o700)
    sources = {
        str(p.relative_to(ROOT)): pin(p) for p in (Path(__file__), HERE / "integration_tests.py")
    }
    env = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "HF_HUB_CACHE": str(ROOT / "artifacts/jev-local-hf/hub"),
        "PYTHONPATH": ":".join(
            str(ROOT / p) for p in ("artifacts/jev-runtime/overlay", "src", "scripts")
        ),
        "CUDA_VISIBLE_DEVICES": "",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "JEV_NATIVE_V2_CPU_TESTS": "tiny-regression-only-v1",
        "JEV_NATIVE_V2_TEST_ATTEMPT": args.attempt,
    }
    argv = [
        "systemd-run",
        "--user",
        "--unit=" + unit,
        "--working-directory=" + str(ROOT),
        "--property=Type=exec",
        "--property=RemainAfterExit=yes",
        "--property=MemoryAccounting=yes",
        "--property=MemoryMax=" + str(CAP),
        "--property=MemorySwapMax=0",
        "--property=OOMPolicy=kill",
        "--property=KillMode=control-group",
        "--property=RuntimeMaxSec=600",
        "--property=TimeoutStopSec=10",
        "--property=Description=" + nonce,
        "--property=StandardOutput=append:" + str(output / "integration-tests-service.log"),
        "--property=StandardError=append:" + str(output / "integration-tests-service.log"),
        *["--setenv=" + k + "=" + v for k, v in env.items()],
        "/home/albert_hsieh/ComfyUI/.venv/bin/python",
        "-B",
        "-s",
        str(HERE / "integration_tests.py"),
    ]
    save(
        output,
        "integration-tests-launch.json",
        {
            "profile": "CPU-only3GiB-16GiB-host-v1",
            "host_before": before,
            "CPU_host_floor_bytes": CPU_RESERVE,
            "unit_memory_max_bytes": CAP,
            "argv": argv,
            "status_host_authority": ref,
            "sources": sources,
            "scope": "CPU regressions only; CUDA hidden. No large-model resource admission, GPU training or browser quality qualification.",
        },
    )
    primary, invocation, reason, attempted = None, None, "terminal", False
    observations, started = [], time.monotonic()

    def description():
        return command(
            "systemctl", "--user", "show", unit, "--value", "--property=Description"
        ).strip()

    try:
        attempted = True
        print(command(*argv), flush=True)
        first = guard.status(unit)
        invocation = first["InvocationID"]
        assert len(invocation) == 32 and description() == nonce
        assert first["ControlGroup"] == cgroup
        wanted = {
            "Type": "exec",
            "MemoryAccounting": "yes",
            "MemoryMax": str(CAP),
            "MemorySwapMax": "0",
            "OOMPolicy": "kill",
            "KillMode": "control-group",
            "RuntimeMaxUSec": "10min",
            "TimeoutStopUSec": "10s",
            "RemainAfterExit": "yes",
            "InvocationID": invocation,
        }
        properties = dict(
            line.split("=", 1)
            for line in command(
                "systemctl",
                "--user",
                "show",
                unit,
                *["--property=" + name for name in wanted],
            ).splitlines()
        )
        group = Path("/sys/fs/cgroup") / cgroup.lstrip("/")
        limits = {
            name: (group / name).read_text().strip() for name in ("memory.max", "memory.swap.max")
        }
        save(
            output,
            "integration-tests-unit-readback.json",
            {
                "properties": properties,
                "limits": limits,
                "first_status": first,
            },
        )
        assert properties == wanted
        assert limits == {"memory.max": str(CAP), "memory.swap.max": "0"}
        while True:
            state, host = guard.status(unit), guard.host()
            assert state["InvocationID"] == invocation
            row = {"elapsed_seconds": time.monotonic() - started, "unit": state, "host": host}
            observations.append(row)
            if len(observations) % 6 == 1 or state["SubState"] in ("exited", "dead", "failed"):
                print(json.dumps(row), flush=True)
            if state["SubState"] in ("exited", "dead", "failed"):
                break
            if host["mem_available_bytes"] < CPU_RESERVE:
                reason = "CPU_host_floor_crossed"
                break
            if time.monotonic() - started > 580:
                reason = "CPU_owner_deadline"
                break
            time.sleep(5)
    except BaseException as exc:
        primary, reason = exc, "owner_error"
    finally:
        try:
            terminal = guard.observe(lambda: guard.status(unit))
            current = terminal.get("InvocationID")
            owned = (
                attempted
                and guard.observe(description) == nonce
                and isinstance(current, str)
                and len(current) == 32
                and (invocation is None or invocation == current)
            )
            stop = (
                guard.observe(lambda: command("systemctl", "--user", "stop", unit, timeout=20))
                if owned
                else {"closure": "UNKNOWN_NOT_AUTHENTICATED"}
            )
            after = guard.observe(lambda: guard.status(unit))
            changed = [
                p
                for p, expected in sources.items()
                if guard.observe(lambda p=p: pin(ROOT / p)) != expected
            ]
            closure = {
                "profile": "CPU-only3GiB-16GiB-host-v1",
                "unit": unit,
                "invocation_id": current if owned else invocation,
                "reason": reason,
                "primary_error": None if primary is None else repr(primary),
                "observations": observations,
                "terminal_before_stop": terminal,
                "stop_result": stop,
                "after_stop": after,
                "owned_invocation_authenticated": owned,
                "cgroup_absent_after_stop": not (
                    Path("/sys/fs/cgroup") / cgroup.lstrip("/")
                ).exists(),
                "changed_controller_worker_sources": changed,
                "host_after": guard.observe(guard.host),
                "counter_scope": "Recorded live systemd counters only; no final zero-OOM claim after cgroup removal.",
            }
            save(output, "integration-tests-root-closure.json", closure)
        except BaseException as secondary:
            if primary is not None:
                primary.add_note("CPU owner finalization also failed: " + repr(secondary))
            else:
                primary = secondary
            try:
                save(
                    output,
                    "integration-tests-finalization-error.json",
                    {
                        "closure": "UNKNOWN_INCOMPLETE_FINALIZATION",
                        "reason": reason,
                        "error": repr(primary),
                        "finalization_error": repr(secondary),
                    },
                )
            except BaseException as persistence_error:
                primary.add_note(
                    "Could not preserve finalization error: " + repr(persistence_error)
                )
    if primary is not None:
        raise primary
    assert reason == "terminal" and owned and not changed
    assert closure["cgroup_absent_after_stop"] and after["MainPID"] == "0"
    assert all(
        terminal[k] == v
        for k, v in {
            "Result": "success",
            "ExecMainCode": "1",
            "ExecMainStatus": "0",
            "MainPID": "0",
        }.items()
    )
    test = json.loads((output / "integration-tests-terminal.json").read_bytes())
    assert (
        test["pytest_returncode"] == 0
        and test["sources_unchanged"]
        and not test["CUDA_initialized"]
    )
    print(
        "Selected CPU regressions passed under their separateCPU profile; owned unit closed.",
        flush=True,
    )


if __name__ == "__main__":
    main()
