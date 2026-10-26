"""Read-only Comfy admission: typed HTTP queue or twice-confirmed absence."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

COMFY_ROOT = Path("/home/albert_hsieh/ComfyUI")
QUEUE_URL = "http://127.0.0.1:8188/queue"
PROC = Path("/proc")
PORT = 8188
MAX_PROCESSES, MAX_ROWS = 4096, 8192
SCAN_SECONDS = 2.0
MAX_IDENTITY_ATTEMPTS = 2
PYTHON = re.compile(r"(?:python(?:[0-9]+(?:\.[0-9]+)*[a-z]?)?|pypy[0-9]*)$", re.I)


class ObservationError(RuntimeError):
    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = evidence


def _read(path, limit):
    with Path(path).open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ObservationError("Observation byte bound exceeded")
    return raw


def _boot(proc=PROC):
    value = _read(proc / "sys/kernel/random/boot_id", 128).decode().strip()
    if not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", value):
        raise ObservationError("Malformed boot identity")
    return value


def _memory(proc=PROC):
    values = {}
    for line in _read(proc / "meminfo", 65536).decode().splitlines():
        key, _, raw = line.partition(":")
        if key in {"MemTotal", "MemAvailable"}:
            parts = raw.split()
            if key in values or len(parts) != 2 or parts[1] != "kB" or not parts[0].isdigit():
                raise ObservationError("Malformed host memory")
            values[key] = int(parts[0]) * 1024
    if set(values) != {"MemTotal", "MemAvailable"} or not (
        0 < values["MemAvailable"] <= values["MemTotal"]
    ):
        raise ObservationError("Incomplete positive host memory")
    return {"mem_total_bytes": values["MemTotal"], "mem_available_bytes": values["MemAvailable"]}


def _queue():
    with urllib.request.urlopen(QUEUE_URL, timeout=2) as reply:
        if reply.status != 200:
            raise ObservationError("Comfy queue returned non-200 HTTP status")
        raw = reply.read(1024**2 + 1)
    if len(raw) > 1024**2:
        raise ObservationError("Comfy queue response exceeded bound")
    value = json.loads(raw)
    if type(value) is not dict or any(
        type(value.get(k)) is not list for k in ("queue_running", "queue_pending")
    ):
        raise ObservationError("Malformed typed Comfy queue")
    return {
        "comfy_running": len(value["queue_running"]),
        "comfy_pending": len(value["queue_pending"]),
        "http_status": 200,
        "response_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _refused(error):
    if isinstance(error, urllib.error.HTTPError):
        return False
    reason = error.reason if isinstance(error, urllib.error.URLError) else error
    return isinstance(reason, OSError) and reason.errno == errno.ECONNREFUSED


def scan_listeners(proc=PROC):
    started, matches, inventories = time.monotonic(), [], []
    for family, width in (("tcp", 8), ("tcp6", 32)):
        raw = _read(proc / "net" / family, 2 * 1024**2)
        lines = raw.decode().splitlines()
        if not lines or "local_address" not in lines[0] or len(lines) - 1 > MAX_ROWS:
            raise ObservationError("Unknown TCP listener inventory")
        for index, line in enumerate(lines[1:]):
            if time.monotonic() - started > SCAN_SECONDS:
                raise ObservationError("TCP observation deadline exceeded")
            fields = line.split()
            if len(fields) < 4:
                raise ObservationError("Malformed TCP observation")
            address, separator, port = fields[1].rpartition(":")
            if not (
                separator
                and len(address) == width
                and re.fullmatch(r"[0-9A-Fa-f]+", address)
                and re.fullmatch(r"[0-9A-Fa-f]{4}", port)
                and re.fullmatch(r"[0-9A-Fa-f]{2}", fields[3])
            ):
                raise ObservationError("Malformed TCP endpoint/state")
            if int(port, 16) == PORT and fields[3].upper() == "0A":
                matches.append({"family": family, "row": index, "port": PORT, "state": "LISTEN"})
        inventories.append(
            {"family": family, "rows": len(lines) - 1, "sha256": hashlib.sha256(raw).hexdigest()}
        )
    return {"listeners": matches, "inventories": inventories}


def _under_comfy(path):
    return Path(os.path.abspath(path)).is_relative_to(COMFY_ROOT)


def classify_command(argv, cwd, comm=""):
    """Inspect executable/actual Python invocation; never search shell payloads."""
    if not argv:
        if comm.lower() in {"comfy", "comfyui", "comfy-cli"}:
            return "uncertain_comfy_runtime"
        return "uncertain_python" if PYTHON.fullmatch(comm) else "non_python_empty"
    executable = Path(argv[0]).name
    if executable.lower() in {"comfy", "comfyui", "comfy-cli"}:
        return "comfy_runtime"
    if not (PYTHON.fullmatch(executable) or PYTHON.fullmatch(comm)):
        return "non_python"
    args, i, mode, target = argv[1:], 0, "interactive", None
    while i < len(args):
        arg = args[i]
        if arg in {"-h", "--help", "-V", "--version"}:
            return "python_information_only"
        if arg == "--":
            i += 1
            break
        if arg in {"-m", "-c"}:
            if i + 1 >= len(args):
                return "uncertain_python"
            mode, target = arg, args[i + 1]
            break
        if arg in {"-W", "-X", "--check-hash-based-pycs"}:
            if i + 1 >= len(args):
                return "uncertain_python"
            i += 2
            continue
        if arg.startswith(("-W", "-X")) and len(arg) > 2:
            i += 1
            continue
        if arg.startswith("-") and arg != "-":
            if not re.fullmatch(r"-[BsuSEIqvOibdRP]+", arg):
                return "uncertain_python"
            i += 1
            continue
        break
    if mode == "interactive" and i < len(args):
        mode, target = "script", args[i]
    if mode == "-m" and (target.split(".")[0].lower() in {"comfy", "comfyui"}):
        return "comfy_runtime"
    if mode == "script" and target != "-":
        path = Path(target)
        if path.is_absolute():
            if _under_comfy(path):
                return "comfy_runtime" if path.name == "main.py" else "uncertain_comfy_script"
            if cwd is None:
                return "uncertain_python_cwd"
            if _under_comfy(cwd):
                return "uncertain_comfy_cwd"
            return "python_external_script"
        if cwd is None:
            return "uncertain_python_cwd"
        if _under_comfy(Path(cwd) / path):
            return "comfy_runtime" if path.name == "main.py" else "uncertain_comfy_script"
    if cwd is None:
        return "uncertain_python_cwd"
    if _under_comfy(cwd):
        return "uncertain_comfy_cwd"
    return "python_external_invocation"


def _stat(path):
    raw = _read(path, 65536).decode()
    left, right = raw.find("("), raw.rfind(")")
    fields = raw[right + 2 :].split()
    if left < 0 or right <= left or len(fields) < 20:
        raise ObservationError("Malformed process identity")
    return raw[left + 1 : right], fields[0], int(fields[19])


def _command(path, comm):
    with (path / "cmdline").open("rb") as stream:
        raw = stream.read(32769)
    first = raw.split(b"\0", 1)[0].decode(errors="surrogateescape")
    if len(raw) > 32768:
        if PYTHON.fullmatch(Path(first).name) or PYTHON.fullmatch(comm):
            raise ObservationError("Possible Python command exceeded bound")
        argv = [first]
    else:
        argv = [v.decode(errors="surrogateescape") for v in raw.split(b"\0") if v]
    return argv, {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def _process_context(path, comm, state):
    if state == "Z":
        return "non_runtime_zombie", None
    argv, signature = _command(path, comm)
    # Intrinsic Comfy/uncertain argv must become sticky before cwd disappears.
    # Only cwd-dependent uncertainty may use a fresh stat disappearance proof.
    intrinsic = classify_command(argv, None, comm)
    if intrinsic == "comfy_runtime" or (
        intrinsic.startswith("uncertain") and intrinsic != "uncertain_python_cwd"
    ):
        return intrinsic, signature
    cwd = None
    if argv and (PYTHON.fullmatch(Path(argv[0]).name) or PYTHON.fullmatch(comm)):
        try:
            cwd = os.readlink(path / "cwd")
            if not Path(cwd).is_absolute() or cwd.endswith(" (deleted)"):
                cwd = None
        except PermissionError:
            pass
    classification = classify_command(argv, cwd, comm)
    if classification != "comfy_runtime" and comm.lower() in {"comfy", "comfyui", "comfy-cli"}:
        classification = "uncertain_comfy_runtime"
    return classification, signature


def _process_attempt(path):
    """One complete observation; only demonstrable identity/argv races may retry."""
    ticks, latched = None, []

    def remember(classification):
        if classification == "comfy_runtime" or classification.startswith("uncertain"):
            latched.append(classification)

    def remember_comm(comm, state):
        if state != "Z" and comm.lower() in {"comfy", "comfyui", "comfy-cli"}:
            remember("uncertain_comfy_runtime")

    def blocking():
        return "comfy_runtime" if "comfy_runtime" in latched else (latched[0] if latched else None)

    try:
        comm, state, ticks = _stat(path / "stat")
        remember_comm(comm, state)
        classification, signature = _process_context(path, comm, state)
        remember(classification)
        after_comm, after_state, after_ticks = _stat(path / "stat")
        remember_comm(after_comm, after_state)
        after_classification, after_signature = _process_context(path, after_comm, after_state)
        remember(after_classification)
        final_comm, final_state, final_ticks = _stat(path / "stat")
        remember_comm(final_comm, final_state)
        identity_changed = (comm, ticks) != (after_comm, after_ticks) or (comm, ticks) != (
            final_comm,
            final_ticks,
        )
        argv_changed = signature != after_signature
        # Latch every actual context and comm before retrying. Stat brackets both
        # argv/cwd reads; Comfy or read uncertainty must survive later disappearance.
        if selected := blocking():
            return {"start_ticks": ticks, "classification": selected}, None
        if identity_changed or argv_changed:
            reason = "identity_changed" if identity_changed else "argv_changed"
            evidence = {
                "reason": reason,
                "before": {
                    "start_ticks": ticks,
                    "comm_sha256": hashlib.sha256(comm.encode()).hexdigest(),
                    "argv": signature,
                },
                "after": {
                    "start_ticks": after_ticks,
                    "comm_sha256": hashlib.sha256(after_comm.encode()).hexdigest(),
                    "argv": after_signature,
                },
                "final": {
                    "start_ticks": final_ticks,
                    "comm_sha256": hashlib.sha256(final_comm.encode()).hexdigest(),
                },
            }
            return {
                "start_ticks": final_ticks,
                "classification": "uncertain_process_" + reason,
            }, evidence
        return {"start_ticks": final_ticks, "classification": after_classification}, None
    except (FileNotFoundError, ProcessLookupError):
        if selected := blocking():
            return {"start_ticks": ticks, "classification": selected}, None
        try:
            _stat(path / "stat")
        except (FileNotFoundError, ProcessLookupError):
            return {"start_ticks": ticks, "classification": "vanished"}, None
        except (PermissionError, OSError, ValueError, ObservationError):
            pass
        return {"start_ticks": ticks, "classification": "uncertain_process_read"}, None
    except (PermissionError, OSError, ValueError, ObservationError):
        # Do not retry a known Comfy observation or read/cwd uncertainty away.
        if selected := blocking():
            return {"start_ticks": ticks, "classification": selected}, None
        return {"start_ticks": ticks, "classification": "uncertain_process_read"}, None


def scan_processes(proc=PROC):
    started, known, uncertain, vanished = time.monotonic(), [], [], []
    retry_count, retry_samples = 0, []
    pids = []
    for path in proc.iterdir():
        if path.name.isdecimal():
            pids.append(int(path.name))
            if len(pids) > MAX_PROCESSES:
                raise ObservationError("Process inventory exceeded bound")
    inventory = hashlib.sha256()
    for pid in sorted(pids):
        if time.monotonic() - started > SCAN_SECONDS:
            raise ObservationError("Process observation deadline exceeded")
        path, attempts = proc / str(pid), []
        for attempt in range(MAX_IDENTITY_ATTEMPTS):
            if time.monotonic() - started > SCAN_SECONDS:
                raise ObservationError("Process observation deadline exceeded")
            observed, race = _process_attempt(path)
            if race is None:
                break
            attempts.append(race)
            if attempt + 1 < MAX_IDENTITY_ATTEMPTS:
                retry_count += 1
        classification = observed["classification"]
        row = {"pid": pid, **observed}
        if attempts:
            row["observation_attempts"] = attempt + 1
            if len(retry_samples) < 32:
                retry_samples.append({**row, "transitions": attempts})
        if classification == "vanished" and len(vanished) < 32:
            vanished.append({"pid": pid, "start_ticks": observed["start_ticks"]})
        inventory.update(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        if classification == "comfy_runtime":
            known.append(row)
        elif classification.startswith("uncertain"):
            uncertain.append(row)
        if len(known) + len(uncertain) > 64:
            raise ObservationError("Possible Comfy process inventory exceeded bound")
    return {
        "scanned": len(pids),
        "inventory_sha256": inventory.hexdigest(),
        "known_comfy": known,
        "uncertain": uncertain,
        "vanished_sample": vanished,
        "identity_retries": retry_count,
        "retry_samples": retry_samples,
    }


def _sample(expected_boot_id):
    started = time.monotonic()
    boot, memory = _boot(), _memory()
    if boot != expected_boot_id:
        raise ObservationError("Comfy observation boot changed")
    listeners, processes = scan_listeners(), scan_processes()
    after = _memory()
    if _boot() != boot or after["mem_total_bytes"] != memory["mem_total_bytes"]:
        raise ObservationError("Comfy observation host identity changed")
    return {
        "started_monotonic": started,
        "finished_monotonic": time.monotonic(),
        "boot_id": boot,
        **after,
        "tcp": listeners,
        "processes": processes,
    }


def observe(expected_boot_id):
    """Raise on ambiguity; absence is a distinct proven observation, never HTTP idle."""
    boot, memory = _boot(), _memory()
    if boot != expected_boot_id:
        raise ObservationError("Comfy observation boot changed")
    try:
        queue = _queue()
    except Exception as error:
        if not _refused(error):
            raise ObservationError(
                "Comfy queue unavailable without proven connection refusal"
            ) from error
        samples = []
        for _ in range(2):
            sample = _sample(expected_boot_id)
            samples.append(sample)
            if sample["boot_id"] != expected_boot_id:
                raise ObservationError("Comfy absence scan boot changed", samples)
            if sample["mem_total_bytes"] != memory["mem_total_bytes"]:
                raise ObservationError("Comfy observation total changed", samples)
            if sample["tcp"]["listeners"]:
                raise ObservationError("Connection refused but port8188 listener remains", samples)
            if sample["processes"]["known_comfy"] or sample["processes"]["uncertain"]:
                raise ObservationError(
                    "Connection refused but Comfy process absence is unproven", samples
                )
        if samples[1]["started_monotonic"] < samples[0]["finished_monotonic"]:
            raise ObservationError("Comfy absence scans are not fresh/sequential", samples)
        return {
            "boot_id": boot,
            "mem_total_bytes": samples[-1]["mem_total_bytes"],
            "mem_available_bytes": samples[-1]["mem_available_bytes"],
            "comfy_running": 0,
            "comfy_pending": 0,
            "comfy_state": "absent_confirmed",
            "comfy_evidence": {
                "transport": "connection_refused",
                "errno": errno.ECONNREFUSED,
                "samples": samples,
                "scope": "Two fresh no-listener/no-runtime process scans; not an HTTP queue response or guarantee of future absence.",
            },
        }
    after = _memory()
    if _boot() != boot or after["mem_total_bytes"] != memory["mem_total_bytes"]:
        raise ObservationError("Comfy HTTP observation host identity changed")
    return {
        "boot_id": boot,
        **after,
        "comfy_running": queue["comfy_running"],
        "comfy_pending": queue["comfy_pending"],
        "comfy_state": "http_busy"
        if queue["comfy_running"] or queue["comfy_pending"]
        else "http_idle",
        "comfy_evidence": {"transport": "HTTP", **queue},
    }
