"""Own one BF16 JevBench scoring run; default dry, no cloud or Comfy mutation."""

import argparse
import hashlib
import importlib.util
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[2]
PREVIOUS = ROOT / "artifacts/jev-runtime/v2-action-training-prepare-20261021-01"
UNIT = "jevbench-gx10-20261022-02.service"
CGROUP = "/user.slice/user-1001.slice/user@1001.service/app.slice/" + UNIT
CAP = 40 * 1024**3


def pin(path):
    blob = path.read_bytes()
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "size_bytes": len(blob),
    }


def authenticate(ref):
    path = ROOT / ref["path"]
    assert not any(p.is_symlink() for p in (path, *path.parents))
    assert pin(path) == ref
    return path


def save(name, value):
    with (HERE / name).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def error_record(error):
    result = {"type": type(error).__name__, "message": str(error)}
    evidence = getattr(error, "evidence", None)
    if evidence is not None:
        try:
            blob = json.dumps(
                evidence, ensure_ascii=False, allow_nan=False, sort_keys=True
            ).encode()
            result["observation_evidence"] = (
                json.loads(blob)
                if len(blob) <= 65536
                else {
                    "truncated": True,
                    "sha256": hashlib.sha256(blob).hexdigest(),
                    "size_bytes": len(blob),
                }
            )
        except (TypeError, ValueError, RecursionError) as secondary:
            result["observation_evidence_error"] = type(secondary).__name__
    return result


def observe_final_host(observe_host):
    try:
        return observe_host()
    except Exception as error:
        return {"observation_error": error_record(error)}


def known_final_host(row, plan):
    if not (
        type(row) is dict
        and row.get("boot_id") == plan["boot_id"]
        and type(row.get("mem_total_bytes")) is int
        and row["mem_total_bytes"] == plan["expected_memtotal"]
        and type(row.get("mem_available_bytes")) is int
        and 0 < row["mem_available_bytes"] <= row["mem_total_bytes"]
        and all(type(row.get(k)) is int and row[k] >= 0 for k in ("comfy_running", "comfy_pending"))
    ):
        return False
    busy = row["comfy_running"] + row["comfy_pending"] > 0
    return (
        row.get("comfy_state") == "http_busy"
        if busy
        else row.get("comfy_state") in {"http_idle", "absent_confirmed"}
    )


def scan(actor):
    matches, unreadable = [], []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            raw = (path / "stat").read_text()
            fields = raw[raw.rfind(")") + 2 :].split()
            pid, pgid, sid = int(path.name), int(fields[2]), int(fields[3])
            if pid == actor["pid"] or pgid == actor["pgid"] or sid == actor["sid"]:
                matches.append(
                    {"pid": pid, "pgid": pgid, "sid": sid, "start_ticks": int(fields[19])}
                )
        except FileNotFoundError:
            continue
        except (PermissionError, OSError, ValueError, IndexError) as error:
            unreadable.append({"pid": path.name, "error": type(error).__name__})
    return {"matches": matches, "unreadable": unreadable}


def command(*args, timeout=15):
    return subprocess.run(args, capture_output=True, text=True, check=True, timeout=timeout).stdout


def description():
    return command("systemctl", "--user", "show", UNIT, "--value", "--property=Description").strip()


def progress():
    try:
        data = json.loads((HERE / "actual-02/progress.json").read_bytes())
        return {
            k: data.get(k)
            for k in ("stage", "completed_predictions", "public_completed", "coherence_completed")
        }
    except (FileNotFoundError, ValueError):
        return None


def load_guard():
    index = json.loads((PREVIOUS / "selected-evidence-index-01.json").read_bytes())
    ref = next(r for r in index["sources"] if r["path"].endswith("/run_compiler_tests.py"))
    path = authenticate(ref)
    spec = importlib.util.spec_from_file_location("_owned_jevbench_score_status", path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    return guard, ref


def load_observer(plan):
    ref = plan["comfy_observer"]
    assert ref in plan["sources"]
    assert (
        ref["path"]
        == "artifacts/jev-runtime/comfy-observer-vanished-context-20261022-01/comfy_observer.py"
    )
    path = authenticate(ref)
    spec = importlib.util.spec_from_file_location("_owned_jevbench_comfy_observer", path)
    observer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = observer
    spec.loader.exec_module(observer)
    return lambda: observer.observe(expected_boot_id=plan["boot_id"])


FRESH_OUTPUTS = (
    "actual-02",
    "actor-intent-02.json",
    "root-launch-02b.json",
    "root-unit-readback-02b.json",
    "root-external-closure-02b.json",
    "root-finalization-error-02b.json",
    "score-service-02b.log",
)


def require_fresh_outputs():
    assert all(not (HERE / name).exists() for name in FRESH_OUTPUTS), (
        "Fresh owned output/evidence required"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if not args.run:
        print("DRY_NO_ACTOR_OR_MODEL; explicit --run and sealed plan required")
        return 0
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
    assert args.plan and args.plan_sha256
    blob = args.plan.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == args.plan_sha256
    plan = json.loads(blob)
    assert plan["unit"] == UNIT and plan["cgroup"] == CGROUP
    assert plan["output"] == (HERE / "actual-02").relative_to(ROOT).as_posix()
    assert plan["unit_memory_max_bytes"] == CAP
    assert plan["boot_id"] == Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    for ref in plan["sources"]:
        authenticate(ref)
    guard, ref = load_guard()
    observe_host = load_observer(plan)
    before = observe_host()
    assert before["mem_total_bytes"] == plan["expected_memtotal"]
    reserve = max(24 * 1024**3, (before["mem_total_bytes"] + 4) // 5)
    assert before["mem_available_bytes"] >= reserve + 2 * 23_919_549_408 + 64 * 1024**2, before
    assert before["comfy_running"] == before["comfy_pending"] == 0, before
    assert guard.status(UNIT)["LoadState"] == "not-found"
    require_fresh_outputs()
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
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTORCH_ALLOC_CONF": "backend:native",
        "JEV_JEVBENCH_EXECUTE": "jevbench-gx10-bf16-score-v1",
        "JEV_JEVBENCH_UNIT": UNIT,
    }
    launch = [
        "systemd-run",
        "--user",
        "--unit=" + UNIT,
        "--working-directory=" + str(ROOT),
        "--property=Type=exec",
        "--property=MemoryAccounting=yes",
        "--property=MemoryMax=" + str(CAP),
        "--property=MemoryHigh=infinity",
        "--property=MemorySwapMax=0",
        "--property=OOMPolicy=kill",
        "--property=KillMode=control-group",
        "--property=RuntimeMaxSec=7200",
        "--property=KillSignal=SIGINT",
        "--property=TimeoutStopSec=20",
        "--property=RemainAfterExit=yes",
        "--property=Description=" + plan["nonce"],
        "--property=StandardOutput=append:" + str(HERE / "score-service-02b.log"),
        "--property=StandardError=append:" + str(HERE / "score-service-02b.log"),
        *["--setenv=" + k + "=" + v for k, v in env.items()],
        "/home/albert_hsieh/ComfyUI/.venv/bin/python",
        "-B",
        "-s",
        str(HERE / "score_worker_02.py"),
        *plan["worker_arguments"],
    ]
    save(
        "root-launch-02b.json",
        {
            "plan_sha256": args.plan_sha256,
            "host_before": before,
            "argv": launch,
            "guardian": ref,
            "scope": "Actual JevBench inference. memcg and native allocator are separate40GiB bounds; no combinedCUDA/hosthard-bound claim.",
        },
    )
    observations, first, invocation, primary = [], None, None, None
    reason, terminal, stopped, stop_result = "terminal", None, None, None
    started, attempted, last_print = time.monotonic(), False, -30
    try:
        attempted = True
        print(command(*launch), flush=True)
        first = guard.status(UNIT)
        invocation = first["InvocationID"]
        assert len(invocation) == 32 and description() == plan["nonce"]
        wanted = {
            "Type": "exec",
            "MemoryAccounting": "yes",
            "MemoryMax": str(CAP),
            "MemoryHigh": "infinity",
            "MemorySwapMax": "0",
            "OOMPolicy": "kill",
            "KillMode": "control-group",
            "RuntimeMaxUSec": "2h",
            "KillSignal": "2",
            "TimeoutStopUSec": "20s",
            "RemainAfterExit": "yes",
            "InvocationID": invocation,
        }
        properties = dict(
            line.split("=", 1)
            for line in command(
                "systemctl",
                "--user",
                "show",
                UNIT,
                *["--property=" + name for name in wanted],
            ).splitlines()
        )
        group = Path("/sys/fs/cgroup") / CGROUP.lstrip("/")
        limits = {
            name: (group / name).read_text().strip() for name in ("memory.max", "memory.swap.max")
        }
        save("root-unit-readback-02b.json", {"properties": properties, "limits": limits})
        assert first["ControlGroup"] == CGROUP and properties == wanted
        assert limits == {"memory.max": str(CAP), "memory.swap.max": "0"}
        while True:
            state, host = guard.status(UNIT), observe_host()
            assert state["InvocationID"] == invocation
            assert host["mem_total_bytes"] == before["mem_total_bytes"]
            elapsed = time.monotonic() - started
            row = {"elapsed_seconds": elapsed, "unit": state, "host": host, "progress": progress()}
            observations.append(row)
            if elapsed - last_print >= 30 or state["SubState"] in ("exited", "dead", "failed"):
                print(json.dumps(row), flush=True)
                last_print = elapsed
            if state["SubState"] in ("exited", "dead", "failed"):
                break
            if host["mem_available_bytes"] < reserve + 2 * 1024**3:
                reason = "host_reserve_margin_crossed"
                break
            if host["comfy_running"] or host["comfy_pending"]:
                reason = "Comfy_became_busy_stop_own_unit_only"
                break
            if elapsed > 7180:
                reason = "owner_deadline"
                break
            time.sleep(2)
    except BaseException as error:
        primary, reason = error, "owner_error"
    finally:
        try:
            if attempted:
                terminal = guard.observe(lambda: guard.status(UNIT))
                # Recover ownership even if the initial query failed; unknown identity
                # retains UNKNOWN and the independent systemd7200-second bound.
                same_nonce = guard.observe(description) == plan["nonce"]
                current = terminal.get("InvocationID")
                owned = (
                    same_nonce
                    and isinstance(current, str)
                    and len(current) == 32
                    and (invocation is None or current == invocation)
                )
                if owned:
                    invocation = current
                    stop_result = guard.observe(
                        lambda: command("systemctl", "--user", "stop", UNIT, timeout=30)
                    )
                else:
                    stop_result = {"closure": "UNKNOWN_NO_AUTHENTICATED_OWNED_INVOCATION"}
                stopped = guard.observe(lambda: guard.status(UNIT))
            cgroup_absent = not (Path("/sys/fs/cgroup") / CGROUP.lstrip("/")).exists()
            actor_path = HERE / "actor-intent-02.json"
            actor_read = guard.observe(lambda: json.loads(actor_path.read_bytes())["actor"])
            actor = actor_read if type(actor_read) is dict and "pid" in actor_read else None
            scans = []
            if actor is not None:
                scans.append(scan(actor))
                time.sleep(0.2)
                scans.append(scan(actor))
            changed = [
                r["path"]
                for r in plan["sources"]
                if guard.observe(lambda r=r: pin(ROOT / r["path"])) != r
            ]
            result = {
                "plan_sha256": args.plan_sha256,
                "invocation_id": invocation,
                "reason": reason,
                "primary_error": None if primary is None else error_record(primary),
                "observations": observations,
                "terminal_before_stop": terminal,
                "stop_result": stop_result,
                "after_stop": stopped,
                "cgroup_absent_after_stop": cgroup_absent,
                "actor": actor,
                "actor_read": actor_read,
                "two_proc_scans": scans,
                "changed_sources": changed,
                "host_after": observe_final_host(observe_host),
                "final_postexit_cgroup_counters": None,
                "counter_scope": "Recorded live counters only; no final zero-OOM claim after cgroup removal.",
            }
            save("root-external-closure-02b.json", result)
        except BaseException as secondary:
            if primary is not None:
                primary.add_note("Owner finalization also failed: " + repr(secondary))
            else:
                primary = secondary
            try:
                save(
                    "root-finalization-error-02b.json",
                    {
                        "invocation_id": invocation,
                        "reason": reason,
                        "terminal_before_stop": terminal,
                        "stop_result": stop_result,
                        "after_stop": stopped,
                        "closure": "UNKNOWN_INCOMPLETE_FINALIZATION",
                        "primary_error": error_record(primary),
                        "finalization_error": {
                            "type": type(secondary).__name__,
                            "message": str(secondary),
                        },
                    },
                )
            except BaseException as persistence_error:
                primary.add_note(
                    "Could not preserve finalization error: " + repr(persistence_error)
                )
    if primary is not None:
        raise primary
    assert reason == "terminal" and cgroup_absent and stopped["MainPID"] == "0"
    assert actor and str(actor["pid"]) == terminal["ExecMainPID"]
    assert all(not r["matches"] and not r["unreadable"] for r in scans) and len(scans) == 2
    assert not changed
    assert known_final_host(result["host_after"], plan), "Final host/Comfy observation is unknown"
    assert all(
        terminal[k] == v
        for k, v in {
            "Result": "success",
            "ExecMainCode": "1",
            "ExecMainStatus": "0",
            "MainPID": "0",
        }.items()
    )
    receipt = json.loads((HERE / "actual-02/receipt.json").read_bytes())
    assert receipt["stage"] == "completed"
    assert (HERE / "actual-02/public231/summary.json").is_file()
    assert (HERE / "actual-02/coherence-mini.json").is_file()
    print(
        json.dumps(
            {
                "score_worker_complete": True,
                "whole_owned_closed": True,
                "InvocationID": invocation,
                "receipt": "actual-02/receipt.json",
                "scope": "Public231 accuracy and JevBench-mini coherence; no formal sealed or composite rank.",
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
