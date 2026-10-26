"""Bounded stdlib owner for explicitly selected CPU command sequences."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import selectors
import subprocess
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[2]
UNIT = "jev-d3-llama-cpp-cpu4-2g-coexist-20261024-07.service"
PROFILE = "llama-cpp-cpu4-2g-coexist-command-owned-v1"
HOST_OBSERVATION_POLICY = "comfy-observed-cpu-coexist-v1"
CPU_CORES = frozenset({16, 17, 18, 19})
CPU_ALLOWED_LIST = "16-19"
SPEC_PROFILE = "llama-cpp-command-spec-v1"
FILE_SIZE_MAX = 256 * 1024**2
LOG_LIMIT = 4 * 1024**2
REPO_PYTHON = ROOT / ".venv/bin/python"
GIB, MAX_SOURCE = 1024**3, 1024**2
MEMORY_MAX_BYTES = 2 * GIB
INITIAL_HOST_EXTRA_BYTES = 2 * GIB
FIXED = {
    "boundary": (
        "artifacts/jev-runtime/decision3-local-owner-terminal-20261022-01/owner_boundary.py",
        "7a50c47d6d7f696973c666cb7a40ce47dd0a7624667a18422c8c60c0db72e4e5",
    ),
    "observer": (
        "artifacts/jev-runtime/observer-system-services-20261022-01/comfy_observer.py",
        "fd977040d659770e26a8fd6cc2b94e6c5440a6a3d3f05099e07d7597d8309e45",
    ),
    "authority": (
        "artifacts/jev-runtime/observer-system-services-20261022-01/system-services.json",
        "a2446af344f084dcf556256fd04c9754fcba940f33b3866990de85a7374c0727",
    ),
}


def need(value, message):
    if not value:
        raise ValueError(message)


def ordinary(path, *, local=True):
    path = Path(path).absolute()
    need(
        ".." not in path.parts
        and not any(p.is_symlink() for p in (path, *path.parents))
        and (not local or path.is_relative_to(ROOT)),
        "Ordinary selected path",
    )
    return path


def read(path):
    with ordinary(path, local=False).open("rb") as stream:
        blob = stream.read(MAX_SOURCE + 1)
    need(len(blob) <= MAX_SOURCE, "Small selected metadata/source only")
    return blob


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def ref(path):
    path = ordinary(path)
    blob = read(path)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(blob), "size_bytes": len(blob)}


def encoded(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"


def load_fixed(name):
    label, digest = FIXED[name]
    path, blob = ROOT / label, read(ROOT / label)
    need(sha(blob) == digest, "Exact unchanged " + name + " authority")
    module = types.ModuleType("_llama_command_" + name)
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(blob, str(path), "exec", dont_inherit=True), vars(module))
    return module


def sources_match(rows):
    need(type(rows) is list and 1 <= len(rows) <= 64, "At most64 small source pins")
    labels = set()
    for row in rows:
        need(
            set(row) == {"path", "sha256", "size_bytes"}
            and not Path(row["path"]).is_absolute()
            and row["path"] not in labels
            and 0 <= row["size_bytes"] <= MAX_SOURCE,
            "Distinct typed small source references",
        )
        labels.add(row["path"])
        need(ref(ROOT / row["path"]) == row, "Selected small source bytes unchanged")
    return True


def reserve(total):
    need(type(total) is int and total > 0, "Actual positive host total")
    return max(24 * GIB, (total + 4) // 5)


def fixed_memory():
    need(
        type(MEMORY_MAX_BYTES) is int
        and type(INITIAL_HOST_EXTRA_BYTES) is int
        and MEMORY_MAX_BYTES == INITIAL_HOST_EXTRA_BYTES == 2 * GIB,
        "Exact fixed2GiB cap and full-cap initial host extra",
    )
    return MEMORY_MAX_BYTES


def comfy_observation(row):
    running, pending = row["comfy_running"], row["comfy_pending"]
    need(
        type(running) is int and running >= 0 and type(pending) is int and pending >= 0,
        "Typed nonnegative observed Comfy counts",
    )
    state, evidence = row["comfy_state"], row.get("comfy_evidence")
    need(type(evidence) is dict, "Explicit original Comfy observation evidence")
    if state in {"http_idle", "http_busy"}:
        need(
            evidence.get("transport") == "HTTP"
            and type(evidence.get("http_status")) is int
            and evidence["http_status"] == 200
            and type(evidence.get("response_sha256")) is str
            and re.fullmatch(r"[0-9a-f]{64}", evidence["response_sha256"]) is not None
            and type(evidence.get("comfy_running")) is int
            and type(evidence.get("comfy_pending")) is int
            and evidence["comfy_running"] == running
            and evidence["comfy_pending"] == pending,
            "Exact typed HTTP200 Comfy count evidence",
        )
        need(
            (state == "http_busy") == (running > 0 or pending > 0),
            "Consistent observed Comfy HTTP state/counts",
        )
    else:
        need(
            state == "absent_confirmed"
            and running == pending == 0
            and evidence.get("transport") == "connection_refused",
            "Original confirmed absence or explicit observed HTTP state required",
        )


def host(observer, boot_id, total=None, *, initial=False):
    fixed_memory()
    row = observer.observe(boot_id)
    required = None
    try:
        need(
            row["boot_id"] == boot_id and (total is None or row["mem_total_bytes"] == total),
            "Same actual boot/host",
        )
        comfy_observation(row)
        required = reserve(row["mem_total_bytes"]) + (INITIAL_HOST_EXTRA_BYTES if initial else 0)
        need(
            type(row["mem_available_bytes"]) is int and row["mem_available_bytes"] >= required,
            "Original whole-host reserve" + (" plus full selected2GiB" if initial else ""),
        )
    except BaseException as error:
        error.evidence = {
            "host_observation_policy": HOST_OBSERVATION_POLICY,
            "observation": row,
            "required_available_bytes": required,
            "initial": initial,
        }
        raise
    return {
        "host_observation_policy": HOST_OBSERVATION_POLICY,
        "observation": row,
        "reserve_bytes": reserve(row["mem_total_bytes"]),
        "required_available_bytes": required,
        "initial": initial,
    }


def failure(error):
    def captured(value):
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        return value[:32768] if isinstance(value, str) else value

    return {
        "type": type(error).__name__,
        "message": str(error),
        "evidence": getattr(error, "evidence", None),
        "returncode": getattr(error, "returncode", None),
        "stdout": captured(getattr(error, "stdout", None)),
        "stderr": captured(getattr(error, "stderr", None)),
        "command_cleanup_errors": getattr(error, "command_cleanup_errors", None),
    }


def keep(primary, errors, phase, error):
    row = {"phase": phase, **failure(error)}
    if primary is None:
        return row
    errors.append(row)
    return primary


def stable_process(row):
    return {key: row[key] for key in ("pid", "start_ticks", "pgid", "sid", "uid", "cgroup")}


def same_identity(left, right):
    return stable_process(left["process"]) == stable_process(right["process"]) and all(
        left[key] == right[key] for key in ("exe", "argv", "cwd")
    )


def process_identity(owner, pid):
    base = Path("/proc") / str(pid)
    before = owner.process(pid)
    value = {
        "process": before,
        "exe": str((base / "exe").resolve(strict=True)),
        "argv": [part.decode() for part in (base / "cmdline").read_bytes().split(b"\0") if part],
        "cwd": str((base / "cwd").resolve(strict=True)),
    }
    need(
        stable_process(owner.process(pid)) == stable_process(before),
        "Stable actual process across exec identity observation",
    )
    return value


def memory_events(group):
    with (group / "memory.events").open("rb") as stream:
        blob = stream.read(8193)
    need(len(blob) <= 8192, "Bounded own cgroup memory.events observation")
    pairs = [line.split() for line in blob.decode().splitlines()]
    need(
        1 <= len(pairs) <= 32
        and all(len(row) == 2 and row[1].isdigit() for row in pairs)
        and len({row[0] for row in pairs}) == len(pairs),
        "Distinct numeric own cgroup memory event counters",
    )
    counters = {key: int(value) for key, value in pairs}
    need({"oom", "oom_kill"} <= counters.keys(), "Own cgroup OOM event counters present")
    return {
        "counters": counters,
        "observed_monotonic": time.monotonic(),
        "scope": "Selected own live cgroup cumulative memory.events snapshot; OOM attribution requires the actual counter evidence and no conclusion is inherited from earlier runs",
    }


def properties(owner, selected, pid):
    keys = (
        "CPUQuotaPerSecUSec",
        "CPUAffinity",
        "LimitCORE",
        "LimitNOFILE",
        "LimitFSIZE",
        "WorkingDirectory",
        "MemoryPeak",
    )
    text = owner.command(
        "systemctl", "--user", "show", UNIT, *("--property=" + k for k in keys), timeout=1
    )
    row = dict(line.split("=", 1) for line in text.splitlines())
    need(
        all(
            row.get(k) == v
            for k, v in {
                "CPUQuotaPerSecUSec": "4s",
                "LimitCORE": "0",
                "LimitNOFILE": "512",
                "LimitFSIZE": str(FILE_SIZE_MAX),
                "WorkingDirectory": str(ROOT),
            }.items()
        ),
        "Actual extra unit resource properties",
    )
    group = owner.CGROUP_ROOT / selected.cgroup.lstrip("/")
    quota = (group / "cpu.max").read_text().split()
    need(
        len(quota) == 2
        and quota[0].isdigit()
        and quota[1].isdigit()
        and int(quota[1]) > 0
        and int(quota[0]) == 4 * int(quota[1]),
        "Actual CPU400% quota kernel readback",
    )
    limits = (Path("/proc") / str(pid) / "limits").read_text()
    for name, number, unit in (
        ("Max core file size", 0, "bytes"),
        ("Max file size", FILE_SIZE_MAX, "bytes"),
        ("Max open files", 512, "files"),
    ):
        need(
            re.search(
                r"(?m)^"
                + name
                + r"\s+"
                + str(number)
                + r"\s+"
                + str(number)
                + r"\s+"
                + unit
                + r"\s*$",
                limits,
            ),
            "Actual process limit: " + name,
        )
    peak = (group / "memory.peak").read_text().strip()
    need(peak.isdigit(), "Actual kernel memory.peak")
    affinity = main_affinity(owner, pid)
    cpuset = group / "cpuset.cpus.effective"
    return {
        "unit": row,
        "kernel_memory_peak_bytes": int(peak),
        "kernel_memory_events": memory_events(group),
        "kernel_cpu_max": quota,
        "cpuset_cpus_effective_raw": cpuset.read_text().strip() if cpuset.exists() else None,
        "main_thread_affinity": affinity,
        "affinity_scope": "Observed main-thread affinity only; no delegated cpuset prevention of later widening or all-thread lifetime claim",
    }


def main_affinity(owner, pid):
    before = stable_process(owner.process(pid))
    first = os.sched_getaffinity(pid)
    status = (Path("/proc") / str(pid) / "status").read_text()
    second = os.sched_getaffinity(pid)
    need(
        first == second == CPU_CORES
        and re.search(r"(?m)^Cpus_allowed_list:\s*16-19$", status)
        and stable_process(owner.process(pid)) == before,
        "Stable actual main-thread affinity sched/proc dual read is CPU16-19",
    )
    return {
        "sched_before": sorted(first),
        "sched_after": sorted(second),
        "proc_Cpus_allowed_list": CPU_ALLOWED_LIST,
        "process": before,
    }


def ensure_absent(owner, selected):
    need(
        owner.status(selected).get("LoadState") == "not-found"
        and not (owner.CGROUP_ROOT / selected.cgroup.lstrip("/")).exists(),
        "Fresh exact unit and cgroup absent; never reuse",
    )


def finalize(owner, boundary, attempted, cfg, observer, out, primary, state):
    errors, closure, final_host, source_ok = [], None, None, False
    if attempted:
        try:
            keys = (
                "Description",
                "InvocationID",
                "MainPID",
                "ExecMainPID",
                "ExecMainCode",
                "ExecMainStatus",
                "Result",
                "MemoryPeak",
            )
            raw = owner.command(
                "systemctl",
                "--user",
                "show",
                UNIT,
                *("--property=" + key for key in keys),
                timeout=1,
            )
            row = dict(line.split("=", 1) for line in raw.splitlines())
            state["preclose_exit_memory_raw"] = row
            need(
                row.get("Description") == cfg["nonce"]
                and boundary.invocation is not None
                and row.get("InvocationID") == boundary.invocation,
                "Memory/exit observation belongs to exact admitted invocation",
            )
            state["preclose_exit_memory_authenticated"] = True
            need(
                boundary.actor is not None and row.get("MainPID") == str(boundary.actor["pid"]),
                "Postreceipt hold must remain live until authenticated close; expiry125 is owner failure",
            )
            state["preclose_main_live"] = True
        except BaseException as error:
            primary = keep(primary, errors, "preclose_exit_memory", error)
        try:
            closure = boundary.close(primary_error=primary)
            state["closure_finished_monotonic"] = time.monotonic()
            if state.get("actor_success"):
                need(
                    closure.get("closure_mode") == "authenticated_stop"
                    and closure.get("before_stop", {}).get("MainPID") == str(boundary.actor["pid"])
                    and state["closure_finished_monotonic"]
                    < state["actor_exit"]["hold_deadline_monotonic"],
                    "Successful owner closure must authenticate live main and finish before fallback125 deadline",
                )
        except BaseException as error:
            primary = keep(primary, errors, "boundary_close", error)
    try:
        sources_match(cfg["sources"])
        command_selection(cfg["selection"], cfg["selection_sha256"], cfg["base_sources"])
        source_ok = True
    except BaseException as error:
        primary = keep(primary, errors, "final_source", error)
    try:
        final_host = host(observer, cfg["boot_id"], cfg.get("mem_total_bytes"))
    except BaseException as error:
        primary = keep(primary, errors, "final_host", error)
    completed = bool(
        primary is None
        and state.get("actor_success")
        and source_ok
        and final_host
        and closure
        and closure.get("whole_unit_exit_confirmed")
        and closure.get("observed_cgroup_cohort_exit_confirmed")
    )
    owner.save(
        out,
        "receipt.json",
        {
            "profile": PROFILE,
            "selected_config": cfg,
            "completed": completed,
            "primary_error": primary,
            "secondary_errors": errors,
            "closure": closure,
            "final_host": final_host,
            "sources_unchanged": source_ok,
            "state": state,
            "unit_attempted": attempted,
            "command_scope": "Selected CPU command exits/resource observations only; no conversion feasibility,12B,GPU,8192context,20s,N3,Q95 or model-quality qualification",
            "large_model_GPU_20s_or_quality_qualified": False,
            "whole_tree_exit_confirmed": False,
            "comfy_service_modified": False,
        },
    )
    return 0 if completed else 1


def record_runtime_host(state, row, phase):
    need(phase in {"readiness", "preexec", "loop"}, "Explicit parent postlaunch host-gate phase")
    summary = state.setdefault(
        "successful_parent_postlaunch_host_gate_summary",
        {
            "runtime_host_gate_count": 0,
            "phase_counts": {"readiness": 0, "preexec": 0, "loop": 0},
            "min_MemAvailable_bytes": None,
            "min_margin_vsR_bytes": None,
            "scope": "All successful executed post-launch readiness/preexec/loop host-gate samples, including discarded raw samples; failing gates stay in primary evidence; initial/final separate; not a continuous-time minimum",
        },
    )
    available = row["observation"]["mem_available_bytes"]
    margin = available - row["reserve_bytes"]
    summary["runtime_host_gate_count"] += 1
    summary["phase_counts"][phase] += 1
    for key, value in (("min_MemAvailable_bytes", available), ("min_margin_vsR_bytes", margin)):
        previous = summary[key]
        summary[key] = value if previous is None else min(previous, value)
    return row


def command_selection(path, digest, local_sources):
    blob = read(ordinary(path))
    need(sha(blob) == digest, "Independently selected command spec SHA")
    value = json.loads(blob)
    need(
        type(value) is dict
        and set(value)
        == {
            "profile",
            "runtime_seconds",
            "file_size_max_bytes",
            "commands",
            "sources",
            "binary_stats",
        }
        and value["profile"] == SPEC_PROFILE
        and type(value["runtime_seconds"]) is int
        and value["runtime_seconds"] in {30, 7200, 10800}
        and type(value["file_size_max_bytes"]) is int
        and value["file_size_max_bytes"] == FILE_SIZE_MAX,
        "Exact selected CPU command profile/runtime/file-size cap",
    )
    need(
        type(value["commands"]) is list and 1 <= len(value["commands"]) <= 8,
        "One to eight explicit sequential commands",
    )
    for row in value["commands"]:
        need(type(row) is dict and set(row) == {"argv", "cwd"}, "Explicit argv/cwd only")
        need(type(row["argv"]) is list and 1 <= len(row["argv"]) <= 256, "Bounded no-shell argv")
        for item in [row["cwd"], *row["argv"]]:
            need(
                type(item) is str and 0 < len(item) <= 4096 and "\0" not in item,
                "Bounded literal command argument",
            )
            rest = item.replace("{OUTPUT}", "")
            need("{" not in rest and "}" not in rest, "Only literal OUTPUT substitution")
        need(
            row["argv"][0].startswith("/") or row["argv"][0].startswith("{OUTPUT}/"),
            "Explicit absolute executable, never PATH lookup",
        )
        need(
            row["cwd"].startswith("/") or row["cwd"].startswith("{OUTPUT}"),
            "Explicit selected absolute cwd",
        )
    need(
        type(value["sources"]) is list and len(value["sources"]) <= 58,
        "Bounded explicit small source pool",
    )
    sources_match([*local_sources, *value["sources"]])
    need(
        type(value["binary_stats"]) is list and len(value["binary_stats"]) <= 32,
        "Bounded tool stat identities, not tool byte closure",
    )
    labels = set()
    for row in value["binary_stats"]:
        need(
            type(row) is dict
            and set(row) == {"path", "stat"}
            and type(row["path"]) is str
            and Path(row["path"]).is_absolute()
            and row["path"] not in labels
            and set(row["stat"]) == {"st_dev", "st_ino", "st_size", "st_mtime_ns"}
            and all(type(n) is int and n >= 0 for n in row["stat"].values()),
            "Explicit distinct binary stat-only identity",
        )
        labels.add(row["path"])
        actual = ordinary(row["path"], local=False).stat()
        need(
            {key: getattr(actual, key) for key in row["stat"]} == row["stat"],
            "Selected executable/tool stat unchanged",
        )
    return value


def environment(cfg):
    return {
        "PATH": os.defpath,
        "LANG": "C.UTF-8",
        "PYTHONPATH": str(ROOT / "artifacts/jev-runtime/overlay"),
        "CUDA_VISIBLE_DEVICES": "",
        "ACCELERATE_USE_CPU": "true",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "USE_TF": "0",
        "USE_FLAX": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
        "XDG_CACHE_HOME": str(Path(cfg["output"]) / "cache"),
        "TMPDIR": str(Path(cfg["output"]) / "tmp"),
        "JEV_COMMAND_NONCE": cfg["nonce"],
    }


def selection(cfg, owner):
    return owner.Selection(
        UNIT, cfg["nonce"], cfg["boot_id"], fixed_memory(), cfg["runtime_seconds"], 3
    )


def bootstrap_argv(cfg, digest):
    return [
        str(REPO_PYTHON),
        "-B",
        "-s",
        str(HERE / "controller.py"),
        "--internal-stage",
        "actor",
        "--config",
        str(Path(cfg["output"]) / "sealed-config.json"),
        "--config-sha256",
        digest,
    ]


class BoundedOutput:
    """Drain all observed bytes, persist only the first4MiB per stream."""

    def __init__(self, path):
        self.path = path
        self.stream = path.open("xb")
        self.digest = hashlib.sha256()
        self.observed = self.retained = 0

    def feed(self, blob):
        self.digest.update(blob)
        self.observed += len(blob)
        retained = blob[: max(0, LOG_LIMIT - self.retained)]
        self.stream.write(retained)
        self.stream.flush()
        self.retained += len(retained)

    def close(self):
        self.stream.close()
        return {
            "path": str(self.path),
            "observed_bytes": self.observed,
            "observed_sha256": self.digest.hexdigest(),
            "retained_bytes": self.retained,
            "truncated": self.observed > self.retained,
            "retention": "First4MiB per stream; all observed bytes drained/hashed; interrupted actor may lack final counts/hash",
        }


def selected_executable(logical, binary_stats, *, generated_under=None):
    resolved = ordinary(Path(logical).resolve(strict=True), local=False)
    need(resolved.is_file(), "Selected executable exists at execution time")
    pinned = next((row for row in binary_stats if row["path"] == str(resolved)), None)
    if pinned is None:
        need(
            generated_under is not None and resolved.is_relative_to(ordinary(generated_under)),
            "Resolved executable must have explicit selected tool stat or fresh owned build origin",
        )
    else:
        actual = resolved.stat()
        need(
            {key: getattr(actual, key) for key in pinned["stat"]} == pinned["stat"],
            "Resolved executable still has selected stat identity",
        )
    return resolved


def execute_command(cfg, index, row, owner):
    out = ordinary(cfg["output"])
    argv = [s.replace("{OUTPUT}", str(out)) for s in row["argv"]]
    cwd = ordinary(row["cwd"].replace("{OUTPUT}", str(out)))
    need(cwd.is_dir(), "Selected command cwd already exists")
    executable = selected_executable(
        argv[0],
        cfg["binary_stats"],
        generated_under=out if row["argv"][0].startswith("{OUTPUT}/") else None,
    )
    executable_stat = executable.stat()
    executable_stat = {
        key: getattr(executable_stat, key) for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    }
    prefix = "command-" + str(index).zfill(2)
    owner.save(
        out,
        prefix + "-intent.json",
        {
            "argv": argv,
            "cwd": str(cwd),
            "shell": False,
            "resolved_executable": str(executable),
            "logical_executable_preserved": argv[0],
            "executable_stat_at_start": executable_stat,
            "executable_scope": "Selected tool stat, or generated executable confined to fresh owned output; generated executable has no prior SHA/quality qualification",
        },
    )
    started = time.monotonic()
    streams, child, selector, output = [], None, selectors.DefaultSelector(), {}
    primary, notes, logs = None, [], []
    try:
        for name in ("stdout", "stderr"):
            streams.append(BoundedOutput(out / (prefix + "." + name)))
        child = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=environment(cfg),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            close_fds=True,
        )
        try:
            identity = process_identity(owner, child.pid)
        except (FileNotFoundError, ProcessLookupError):
            identity = None  # Very short commands can exit before an actual /proc observation.
        if identity is not None:
            need(
                identity["exe"] == str(executable),
                "Observed child executable matches selected resolved target",
            )
        owner.save(
            out,
            prefix + "-started.json",
            {
                "pid": child.pid,
                "observed_identity": identity,
                "identity_scope": "Observed child only; wrapper/cgroup admission is separate",
                "started_monotonic": started,
            },
        )
        for pipe, sink in zip((child.stdout, child.stderr), streams):
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, sink)
        while selector.get_map():
            need(
                time.monotonic() < cfg["command_deadline_monotonic"],
                "Selected command deadline; parent must close authenticated unit",
            )
            for event, _ in selector.select(timeout=0.2):
                blob = os.read(event.fileobj.fileno(), 65536)
                if blob:
                    event.data.feed(blob)
                else:
                    selector.unregister(event.fileobj)
                    event.fileobj.close()
        remaining = max(0.001, cfg["command_deadline_monotonic"] - time.monotonic())
        code = child.wait(timeout=remaining)
        output = {
            "argv": argv,
            "cwd": str(cwd),
            "resolved_executable": str(executable),
            "pid": child.pid,
            "observed_identity": identity,
            "executable_stat_at_start": executable_stat,
            "returncode": code,
            "elapsed_seconds": time.monotonic() - started,
        }
    except BaseException as error:
        primary = error
    finally:

        def cleanup(phase, operation):
            nonlocal primary
            try:
                return operation()
            except BaseException as error:
                if primary is None:
                    primary = error
                else:
                    notes.append({"phase": phase, **failure(error)})
                return None

        cleanup("selector_close", selector.close)
        # No direct process-tree kill: parent uses the exact original boundary close.
        if child is not None:
            for pipe in (child.stdout, child.stderr):
                if pipe is not None and not pipe.closed:
                    cleanup("pipe_close", pipe.close)
        for stream in streams:
            row = cleanup("log_close", stream.close)
            if row is not None:
                logs.append(row)
        cleanup(
            "command_output_receipt",
            lambda: owner.save(
                out,
                prefix + "-output.json",
                {
                    "logs": logs,
                    "result": output,
                    "primary_error": failure(primary) if primary is not None else None,
                    "cleanup_errors": notes,
                    "child_poll_at_finalization": child.poll() if child is not None else None,
                },
            ),
        )
    if primary is not None:
        primary.command_cleanup_errors = notes
        raise primary
    return output


def bootstrap(cfg, digest, owner):
    selected = selection(cfg, owner)
    need(
        os.environ.get("JEV_COMMAND_NONCE") == cfg["nonce"] and owner.boot() == cfg["boot_id"],
        "Exact command bootstrap nonce/boot",
    )
    sources_match(cfg["sources"])
    spec = command_selection(cfg["selection"], cfg["selection_sha256"], cfg["base_sources"])
    me = process_identity(owner, os.getpid())
    p = me["process"]
    need(
        p["pid"] == p["pgid"] == p["sid"]
        and p["cgroup"] == selected.cgroup
        and p["uid"] == os.getuid()
        and me["cwd"] == str(ROOT),
        "Actual selected bootstrap session/cgroup",
    )
    affinity = main_affinity(owner, os.getpid())
    out = ordinary(cfg["output"])
    owner.save(
        out,
        "bootstrap-identity.json",
        {
            "config_sha256": digest,
            "nonce": cfg["nonce"],
            "identity": me,
            "torch_not_imported": "torch" not in sys.modules,
            "observed_main_thread_affinity": affinity["sched_after"],
            "bootstrap_main_thread_affinity": affinity,
            "sys_executable": sys.executable,
            "python_flags": {
                "no_user_site": sys.flags.no_user_site,
                "dont_write_bytecode": sys.dont_write_bytecode,
            },
        },
    )
    until = time.monotonic() + 10
    while not (out / "admission.json").exists():
        need(time.monotonic() < until, "Bounded parent admission wait")
        time.sleep(0.01)
    admitted = json.loads(read(out / "admission.json"))
    need(
        admitted["config_sha256"] == digest
        and admitted["nonce"] == cfg["nonce"]
        and stable_process(admitted["actor"]) == stable_process(p)
        and re.fullmatch(r"[0-9a-f]{32}", admitted["invocation_id"]),
        "Exact parent admission identity",
    )
    code, error, completed = 0, None, []
    try:
        sources_match(cfg["sources"])
        for index, row in enumerate(spec["commands"], 1):
            result = execute_command(cfg, index, row, owner)
            completed.append(result)
            if result["returncode"] != 0:
                raise subprocess.CalledProcessError(result["returncode"], result["argv"])
        sources_match(cfg["sources"])
        command_selection(cfg["selection"], cfg["selection_sha256"], cfg["base_sources"])
    except BaseException as exc:
        code, error = 1, failure(exc)
    hold = time.monotonic()
    owner.save(
        out,
        "actor-exit.json",
        {
            "config_sha256": digest,
            "actor": p,
            "entry_exit_code": code,
            "error": error,
            "command_results": completed,
            "planned_command_count": len(spec["commands"]),
            "postreceipt_hold_seconds": 5,
            "hold_started_monotonic": hold,
            "hold_deadline_monotonic": hold + 5,
            "reason": "Keep admitted wrapper live for original authenticated unit close",
        },
    )
    time.sleep(max(0, hold + 5 - time.monotonic()))
    return 125


def retain_and_validate_result(cfg, actor, result, out, state):
    state["actor_exit"] = result
    need(
        result["config_sha256"] == cfg["config_sha256"]
        and stable_process(result["actor"]) == stable_process(actor),
        "Command result belongs to admitted wrapper identity",
    )
    if result["entry_exit_code"] != 0 or result["error"] is not None:
        error = ValueError("Selected child command failed; original error retained")
        error.evidence = result
        raise error
    rows = result["command_results"]
    need(
        result["planned_command_count"] == cfg["command_count"]
        and len(rows) == cfg["command_count"]
        and all(row["returncode"] == 0 for row in rows),
        "All explicitly selected commands completed with exit0",
    )
    state["actor_success"] = True


def run(args):
    need(
        args.expected_entry_sha256 == sha(read(HERE / "controller.py")),
        "Explicit independently reviewed controller entry SHA",
    )
    need(
        args.selection is not None and re.fullmatch(r"[0-9a-f]{64}", args.selection_sha256 or ""),
        "Explicit selected command spec path and SHA",
    )
    path = ordinary(args.selection)
    base_refs = [
        ref(HERE / "controller.py"),
        ref(path),
        *(ref(ROOT / label) for label, _ in FIXED.values()),
    ]
    for name, (label, expected) in FIXED.items():
        need(sha(read(ROOT / label)) == expected, "Exact initial " + name + " authority")
    spec = command_selection(path, args.selection_sha256, base_refs)
    refs = [*base_refs, *spec["sources"]]
    owner, observer = load_fixed("boundary"), load_fixed("observer")
    out = ordinary(args.output)
    need(
        out.parent == HERE and re.fullmatch(r"actual-[0-9]{2}", out.name) and not out.exists(),
        "Fresh scoped controller output",
    )
    cfg = {
        "profile": PROFILE,
        "unit": UNIT,
        "nonce": "jev-d3-boundary-" + secrets.token_hex(16),
        "boot_id": owner.boot(),
        "output": str(out),
        "sources": refs,
        "base_sources": base_refs,
        "selection": str(path),
        "selection_sha256": args.selection_sha256,
        "runtime_seconds": spec["runtime_seconds"],
        "command_count": len(spec["commands"]),
        "binary_stats": spec["binary_stats"],
    }
    selected = selection(cfg, owner)
    ensure_absent(owner, selected)
    out.mkdir(mode=0o700)
    for label in ("tmp", "cache"):
        (out / label).mkdir(mode=0o700)
    boundary = owner.UnitBoundary(selected, out, refs)
    attempted, primary, state = False, None, {}
    try:
        initial = host(observer, cfg["boot_id"], initial=True)
        cfg["mem_total_bytes"] = initial["observation"]["mem_total_bytes"]
        cfg["command_deadline_monotonic"] = time.monotonic() + cfg["runtime_seconds"] - 7
        state["initial_host"] = initial
        owner.save(out, "sealed-config.json", cfg)
        digest = sha(read(out / "sealed-config.json"))
        props = {
            **selected.properties(),
            "RuntimeMaxSec": str(cfg["runtime_seconds"]),
            "TimeoutStopSec": "3",
            "WorkingDirectory": str(ROOT),
            "CPUQuota": "400%",
            "CPUAffinity": CPU_ALLOWED_LIST,
            "LimitCORE": "0",
            "LimitNOFILE": "512",
            "LimitFSIZE": str(FILE_SIZE_MAX),
            "StandardOutput": "append:" + str(out / "actor.stdout"),
            "StandardError": "append:" + str(out / "actor.stderr"),
        }
        del props["RuntimeMaxUSec"], props["TimeoutStopUSec"]
        command = [
            "systemd-run",
            "--user",
            "--unit=" + UNIT,
            *("--property=" + k + "=" + v for k, v in props.items()),
            "/usr/bin/env",
            "-i",
            *(k + "=" + v for k, v in environment(cfg).items()),
            *bootstrap_argv(cfg, digest),
        ]
        owner.save(
            out,
            "launch-intent.json",
            {
                "argv": command,
                "source_count": len(refs),
                "runtime_seconds": cfg["runtime_seconds"],
                "memory_max_bytes": fixed_memory(),
                "initial_host_extra_bytes": INITIAL_HOST_EXTRA_BYTES,
                "memory_swap_max_bytes": 0,
                "per_file_size_max_bytes": FILE_SIZE_MAX,
                "command_log_limit_bytes_per_stream": LOG_LIMIT,
            },
        )
        attempted = True
        state["systemd_run_stdout"] = owner.command(*command, timeout=5)
        until = time.monotonic() + 10
        while not (out / "bootstrap-identity.json").exists():
            need(time.monotonic() < until, "Bounded bootstrap readiness")
            record_runtime_host(
                state, host(observer, cfg["boot_id"], cfg["mem_total_bytes"]), "readiness"
            )
            time.sleep(0.05)
        identity = json.loads(read(out / "bootstrap-identity.json"))
        rb = boundary.verify()
        actor = rb["actor"]
        need(
            identity["config_sha256"] == digest
            and identity["nonce"] == cfg["nonce"]
            and stable_process(identity["identity"]["process"]) == stable_process(actor)
            and identity["torch_not_imported"] is True
            and identity["observed_main_thread_affinity"] == sorted(CPU_CORES)
            and identity["bootstrap_main_thread_affinity"]
            == {
                "sched_before": sorted(CPU_CORES),
                "sched_after": sorted(CPU_CORES),
                "proc_Cpus_allowed_list": CPU_ALLOWED_LIST,
                "process": stable_process(actor),
            }
            and identity["identity"]["exe"] == str(REPO_PYTHON.resolve(strict=True))
            and identity["identity"]["argv"] == bootstrap_argv(cfg, digest)
            and identity["sys_executable"] == str(REPO_PYTHON)
            and identity["python_flags"] == {"no_user_site": 1, "dont_write_bytecode": True},
            "Actual pre-command PID/start/stdlib identity",
        )
        need(
            same_identity(identity["identity"], process_identity(owner, actor["pid"])),
            "Fresh exact actual wrapper exec identity",
        )
        state["bootstrap_admission"] = {
            "identity": identity,
            "properties": properties(owner, selected, actor["pid"]),
        }
        record_runtime_host(
            state, host(observer, cfg["boot_id"], cfg["mem_total_bytes"]), "preexec"
        )
        sources_match(refs)
        owner.save(
            out,
            "admission.json",
            {
                "config_sha256": digest,
                "nonce": cfg["nonce"],
                "invocation_id": boundary.invocation,
                "actor": actor,
            },
        )
        state["runtime_observations"], state["runtime_observation_count"] = [], 0
        while not (out / "actor-exit.json").exists():
            need(
                time.monotonic() < cfg["command_deadline_monotonic"] + 1,
                "Bounded selected command deadline",
            )
            row = boundary.owned_status()
            need(
                row.get("MainPID") == str(actor["pid"])
                and row.get("ActiveState") == "active"
                and row.get("SubState") == "running",
                "Selected wrapper remains live",
            )
            boundary.members()
            observation = {
                "host": record_runtime_host(
                    state, host(observer, cfg["boot_id"], cfg["mem_total_bytes"]), "loop"
                ),
                "properties": properties(owner, selected, actor["pid"]),
            }
            state["runtime_observation_count"] += 1
            if len(state["runtime_observations"]) < 64:
                state["runtime_observations"].append(observation)
            else:
                state["runtime_observations"][-1] = observation
            state["runtime_observation_retention"] = (
                "First63 plus latest,64 total; all successful samples enter scalar summary; sampled not continuous"
            )
            time.sleep(1)
        result = json.loads(read(out / "actor-exit.json"))
        state["last_live_properties"] = properties(owner, selected, actor["pid"])
        retain_and_validate_result(
            {"config_sha256": digest, "command_count": cfg["command_count"]},
            actor,
            result,
            out,
            state,
        )
    except BaseException as error:
        primary = {"phase": "controller", **failure(error)}
    finally:
        result = finalize(owner, boundary, attempted, cfg, observer, out, primary, state)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--internal-stage", choices=("actor",))
    parser.add_argument("--expected-entry-sha256")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--selection-sha256")
    parser.add_argument("--output", type=Path, default=HERE / "actual-01")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--config-sha256")
    args = parser.parse_args(argv)
    if args.internal_stage:
        blob = read(args.config)
        need(sha(blob) == args.config_sha256, "Exact sealed bootstrap config SHA")
        cfg = json.loads(blob)
        need(
            cfg["profile"] == PROFILE
            and cfg["unit"] == UNIT
            and ordinary(cfg["output"]).parent == HERE,
            "Fixed bootstrap profile/output",
        )
        return bootstrap(cfg, args.config_sha256, load_fixed("boundary"))
    if not args.run:
        print(
            json.dumps(
                {
                    "mode": "DRY_SOURCE_ONLY",
                    "unit_started": False,
                    "torch_imported": False,
                    "unit": UNIT,
                    "host_observation_policy": HOST_OBSERVATION_POLICY,
                    "memory_max_bytes": fixed_memory(),
                    "initial_host_extra_bytes": INITIAL_HOST_EXTRA_BYTES,
                    "runtime_choices_seconds": [30, 7200, 10800],
                    "cpu_quota_percent": 400,
                    "main_thread_cpu_affinity": sorted(CPU_CORES),
                    "model_or_quality_qualified": False,
                }
            )
        )
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
