"""Read-only Comfy admission: typed HTTP queue or twice-confirmed absence."""

from __future__ import annotations

import contextvars
import errno
import hashlib
import json
import os
import re
import select
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
OBSERVE_SECONDS, RESOLUTION_SECONDS = 2.0, 0.5
RESOLVABLE = frozenset(
    {
        "uncertain_python_cwd",
        "uncertain_process_read",
        "uncertain_process_identity_changed",
        "uncertain_process_argv_changed",
    }
)
_DEADLINE = contextvars.ContextVar("comfy_observation_deadline", default=None)
PYTHON = re.compile(r"(?:python(?:[0-9]+(?:\.[0-9]+)*[a-z]?)?|pypy[0-9]*)$", re.I)


class ObservationError(RuntimeError):
    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = evidence


def _remaining():
    deadline = _DEADLINE.get()
    remaining = OBSERVE_SECONDS if deadline is None else deadline - time.monotonic()
    if remaining <= 0:
        raise ObservationError("Whole Comfy observation deadline exceeded")
    return remaining


def _read(path, limit):
    _remaining()
    with Path(path).open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ObservationError("Observation byte bound exceeded")
    _remaining()
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
    with urllib.request.urlopen(QUEUE_URL, timeout=min(2, _remaining())) as reply:
        if reply.status != 200:
            raise ObservationError("Comfy queue returned non-200 HTTP status")
        raw = reply.read(1024**2 + 1)
    _remaining()
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


def _process_context(path, comm, state, diagnostics):
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
            diagnostic = {
                "cwd_failure_type": None,
                "cwd_sha256": hashlib.sha256(cwd.encode()).hexdigest(),
            }
            # A deleted Comfy cwd is still an observed Comfy path, never generic.
            candidate = cwd.removesuffix(" (deleted)")
            if Path(candidate).is_absolute() and _under_comfy(candidate):
                diagnostics.append(diagnostic)
                return "uncertain_comfy_cwd", signature
            if not Path(cwd).is_absolute() or cwd.endswith(" (deleted)"):
                diagnostic["cwd_failure_type"] = (
                    "DeletedCwd" if cwd.endswith(" (deleted)") else "NonabsoluteCwd"
                )
                cwd = None
            diagnostics.append(diagnostic)
        except (PermissionError, FileNotFoundError, ProcessLookupError) as error:
            diagnostics.append({"cwd_failure_type": type(error).__name__, "cwd_sha256": None})
            if not isinstance(error, PermissionError):
                raise
    classification = classify_command(argv, cwd, comm)
    if classification != "comfy_runtime" and comm.lower() in {"comfy", "comfyui", "comfy-cli"}:
        classification = "uncertain_comfy_runtime"
    return classification, signature


def _process_attempt_core(path, diagnostics):
    """One complete observation; only demonstrable identity/argv races may retry."""
    ticks, latched = None, []

    def remember(classification):
        if classification == "comfy_runtime" or classification.startswith("uncertain"):
            latched.append(classification)

    def remember_comm(comm, state):
        if state != "Z" and comm.lower() in {"comfy", "comfyui", "comfy-cli"}:
            remember("uncertain_comfy_runtime")

    def blocking():
        if "comfy_runtime" in latched:
            return "comfy_runtime"
        # Every intrinsic/Comfy latch dominates generic provisional uncertainty.
        return next(
            (value for value in latched if value not in RESOLVABLE), latched[0] if latched else None
        )

    try:
        comm, state, ticks = _stat(path / "stat")
        remember_comm(comm, state)
        classification, signature = _process_context(path, comm, state, diagnostics)
        remember(classification)
        after_comm, after_state, after_ticks = _stat(path / "stat")
        remember_comm(after_comm, after_state)
        after_classification, after_signature = _process_context(
            path, after_comm, after_state, diagnostics
        )
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


def _process_attempt(path):
    diagnostics = []
    observed, race = _process_attempt_core(path, diagnostics)
    if diagnostics:
        observed["context_diagnostics"] = diagnostics
    return observed, race


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
        try:
            owner_uid = path.stat().st_uid
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            owner_uid = None
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
        if classification.startswith("uncertain"):
            row["proc_owner_uid"] = owner_uid
            row["disappearance_resolution_eligible"] = (
                classification in RESOLVABLE
                and type(owner_uid) is int
                and type(observed["start_ticks"]) is int
                and observed["start_ticks"] > 0
            )
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
    _remaining()
    started = time.monotonic()
    boot, memory = _boot(), _memory()
    if boot != expected_boot_id:
        raise ObservationError("Comfy observation boot changed")
    listeners, processes = scan_listeners(), scan_processes()
    after = _memory()
    if _boot() != boot or after["mem_total_bytes"] != memory["mem_total_bytes"]:
        raise ObservationError("Comfy observation host identity changed")
    _remaining()
    return {
        "started_monotonic": started,
        "finished_monotonic": time.monotonic(),
        "boot_id": boot,
        **after,
        "tcp": listeners,
        "processes": processes,
    }


def _open_pidfd(pid):
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid, 0)
    import ctypes

    library = ctypes.CDLL(None, use_errno=True)
    if not hasattr(library, "pidfd_open"):
        raise ObservationError("Read-only process exit proof is unavailable")
    function = library.pidfd_open
    function.argtypes, function.restype = [ctypes.c_int, ctypes.c_uint], ctypes.c_int
    descriptor = function(pid, 0)
    if descriptor < 0:
        raise OSError(ctypes.get_errno(), "Read-only pidfd observation failed")
    return descriptor


def _wait_disappeared(rows):
    """Observe natural exit only. Never signal, or clear an alive classification."""
    started = time.monotonic()
    until = min(started + RESOLUTION_SECONDS, started + _remaining())
    evidence = {
        "started_monotonic": started,
        "offenders": rows,
        "disappeared": [],
        "exited_zombies": [],
    }
    if any(
        row.get("disappearance_resolution_eligible") is not True
        or row.get("classification") not in RESOLVABLE
        or type(row.get("pid")) is not int
        or row["pid"] <= 0
        or type(row.get("start_ticks")) is not int
        or row["start_ticks"] <= 0
        or type(row.get("proc_owner_uid")) is not int
        for row in rows
    ):
        raise ObservationError("Intrinsic/unknown Comfy evidence cannot wait", evidence)
    descriptors, pending = [], {}
    try:
        for row in rows:
            _remaining()
            pid = row["pid"]
            path = PROC / str(pid)
            try:
                descriptor = _open_pidfd(pid)
            except ProcessLookupError:
                try:
                    _stat(path / "stat")
                except (FileNotFoundError, ProcessLookupError):
                    evidence["disappeared"].append(pid)
                    continue
                raise ObservationError("Offender PID remains visible", evidence) from None
            descriptors.append(descriptor)
            pending[pid] = (descriptor, row)
        while pending:
            _remaining()
            for pid, (descriptor, row) in list(pending.items()):
                path = PROC / str(pid)
                try:
                    comm, state, ticks = _stat(path / "stat")
                    if ticks != row["start_ticks"]:
                        raise ObservationError("Offender identity changed while alive", evidence)
                    if state == "Z":
                        poller = select.poll()
                        poller.register(descriptor, select.POLLIN)
                        events = poller.poll(0)
                        if any(mask & (select.POLLERR | select.POLLNVAL) for _, mask in events):
                            raise ObservationError("Unknown pidfd exit observation", evidence)
                        if events and all(
                            mask & (select.POLLIN | select.POLLHUP) for _, mask in events
                        ):
                            # A pidfd opened without PIDFD_THREAD is ready only
                            # after the last thread exits. Recheck exact Z identity.
                            _, final_state, final_ticks = _stat(path / "stat")
                            if final_state == "Z" and final_ticks == ticks:
                                evidence["exited_zombies"].append(
                                    {"pid": pid, "start_ticks": ticks}
                                )
                                del pending[pid]
                                continue
                    if state != "Z" and comm.lower() in {"comfy", "comfyui", "comfy-cli"}:
                        raise ObservationError("Observed Comfy during natural-exit check", evidence)
                    observed, _ = _process_attempt(path)
                    classification = observed["classification"]
                    if classification == "comfy_runtime" or (
                        classification.startswith("uncertain") and classification not in RESOLVABLE
                    ):
                        raise ObservationError(
                            "Intrinsic/Comfy context observed during natural-exit check", evidence
                        )
                except (FileNotFoundError, ProcessLookupError):
                    poller = select.poll()
                    poller.register(descriptor, select.POLLIN)
                    events = poller.poll(0)
                    if any(mask & (select.POLLERR | select.POLLNVAL) for _, mask in events):
                        raise ObservationError("Unknown pidfd exit observation", evidence)
                    if events and all(
                        mask & (select.POLLIN | select.POLLHUP) for _, mask in events
                    ):
                        evidence["disappeared"].append(pid)
                        del pending[pid]
            if not pending:
                break
            remaining = min(until - time.monotonic(), _remaining())
            if remaining <= 0:
                raise ObservationError(
                    "Unknown process remains after natural-exit budget", evidence
                )
            time.sleep(min(0.005, remaining))
        if time.monotonic() > until:
            raise ObservationError("Natural-exit resolution exceeded its shared budget", evidence)
        evidence["finished_monotonic"] = time.monotonic()
        evidence["mechanism"] = (
            "read-only pidfd exit and fresh proc absence or same-identity zombie; no signals"
        )
        return evidence
    except ObservationError as error:
        evidence["finished_monotonic"] = time.monotonic()
        evidence["failure_reason"] = str(error)
        raise
    except (PermissionError, OSError, ValueError) as error:
        evidence["failure_type"] = type(error).__name__
        raise ObservationError("Natural-exit proof is unknown", evidence) from error
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


def _observe(expected_boot_id):
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
        resolution = None
        for attempt in range(2):
            samples, restart = [], False
            for _ in range(2):
                _remaining()
                sample = _sample(expected_boot_id)
                samples.append(sample)
                if sample["boot_id"] != expected_boot_id:
                    raise ObservationError("Comfy absence scan boot changed", samples)
                if sample["mem_total_bytes"] != memory["mem_total_bytes"]:
                    raise ObservationError("Comfy observation total changed", samples)
                if sample["tcp"]["listeners"] or sample["processes"]["known_comfy"]:
                    raise ObservationError("Listener/known Comfy evidence cannot wait", samples)
                unknown = sample["processes"]["uncertain"]
                if unknown:
                    if attempt:
                        raise ObservationError(
                            "Unknown evidence remains after one fresh restart",
                            {"samples": samples, "resolution": resolution},
                        )
                    try:
                        resolution = _wait_disappeared(unknown)
                    except ObservationError as refusal:
                        raise ObservationError(
                            "Connection refused but Comfy process absence is unproven",
                            {"samples": samples, "resolution": refusal.evidence},
                        ) from refusal
                    resolution["prior_samples_sha256"] = hashlib.sha256(
                        json.dumps(samples, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest()
                    restart = True
                    break
            if not restart:
                break
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
                "natural_exit_resolution": resolution,
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


def observe(expected_boot_id):
    """One shared admission deadline; a single natural-exit restart at most."""
    token = _DEADLINE.set(time.monotonic() + OBSERVE_SECONDS)
    try:
        result = _observe(expected_boot_id)
        _remaining()
        return result
    finally:
        _DEADLINE.reset(token)
