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


def scan_processes(proc=PROC):
    started, known, uncertain, vanished = time.monotonic(), [], [], []
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
        path, ticks, classification = proc / str(pid), None, None
        try:
            comm, state, ticks = _stat(path / "stat")
            if state == "Z":
                classification = "non_runtime_zombie"
            else:
                try:
                    with (path / "cmdline").open("rb") as stream:
                        raw = stream.read(32769)
                    first = raw.split(b"\0", 1)[0].decode(errors="surrogateescape")
                    if len(raw) > 32768:
                        if PYTHON.fullmatch(Path(first).name) or PYTHON.fullmatch(comm):
                            raise ObservationError("Possible Python command exceeded bound")
                        argv = [first]
                    else:
                        argv = [v.decode(errors="surrogateescape") for v in raw.split(b"\0") if v]
                except PermissionError:
                    argv = []
                cwd = None
                if argv and (PYTHON.fullmatch(Path(argv[0]).name) or PYTHON.fullmatch(comm)):
                    try:
                        cwd = os.readlink(path / "cwd")
                        if not Path(cwd).is_absolute() or cwd.endswith(" (deleted)"):
                            cwd = None
                    except (PermissionError, FileNotFoundError, ProcessLookupError):
                        pass
                classification = classify_command(argv, cwd, comm)
            after_comm, _, after_ticks = _stat(path / "stat")
            if after_ticks != ticks or after_comm != comm:
                classification = "uncertain_process_identity_changed"
        except (FileNotFoundError, ProcessLookupError):
            try:
                _stat(path / "stat")
            except (FileNotFoundError, ProcessLookupError):
                if len(vanished) < 32:
                    vanished.append({"pid": pid, "start_ticks": ticks})
                classification = "vanished"
            except (PermissionError, OSError, ValueError, ObservationError):
                classification = "uncertain_process_read"
            else:
                classification = "uncertain_process_read"
        except (PermissionError, OSError, ValueError, ObservationError):
            classification = "uncertain_process_read"
        row = {"pid": pid, "start_ticks": ticks, "classification": classification}
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
