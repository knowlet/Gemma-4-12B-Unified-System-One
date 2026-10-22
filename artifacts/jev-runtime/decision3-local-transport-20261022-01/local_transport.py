"""Owned local Decision3 transport; a new profile, never model qualification.

Uses the frozen worker's unchanged request envelope and Session/Gateway lifecycle
design. No project, model, CUDA, Q95, Modal or Comfy module is imported here.
The caller must authenticate its selected sources, runtime, cache and helper.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import secrets
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROFILE = "decision3-gx10-local-owned-transport-v1"
MODELS = ("d3", "d3-flash", "d3-mini", "d3-nano", "d3-lite", "d3-edge")
MAX_BODY, MAX_ENVELOPE, MAX_FRAME, MAX_LOG = 1_000_000, 4 * 1024**2, 16 * 1024**2, 16 * 1024**2
MAX_REQUESTS, MAX_FILES, MAX_BYTES = 192, 512, 512 * 1024**2
MARKER = "JEV_D3_LOCAL_OWNER"
SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")


class TransportError(ValueError):
    """The arm is blocked; raw evidence does not authorize another action."""


def require(condition, message):
    if not condition:
        raise TransportError(message)


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def parse(blob):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def finite(value):
        result = float(value)
        require(math.isfinite(result), "Finite JSON numbers required")
        return result

    return json.loads(
        blob, object_pairs_hook=pairs, parse_float=finite,
        parse_constant=lambda value: require(False, "Nonfinite JSON: " + value),
    )


def read_bound(path, limit):
    with open(path, "rb", buffering=0) as stream:
        blob = stream.read(limit + 1)
    require(len(blob) <= limit, "Read bound exceeded: " + str(path))
    return blob


def host_observation():
    values = {}
    for line in read_bound("/proc/meminfo", 65536).decode().splitlines():
        name, _, raw = line.partition(":")
        if name in {"MemTotal", "MemAvailable"}:
            number, unit = raw.split()
            require(unit == "kB", "Host memory units")
            values[name] = int(number) * 1024
    values["boot_id"] = read_bound("/proc/sys/kernel/random/boot_id", 128).decode().strip()
    return values


@dataclass(frozen=True)
class Selection:
    nonce: str
    model: str
    expected_boot_id: str
    expected_memtotal: int
    expected_gpu_uuid: str

    def __post_init__(self):
        require(type(self.nonce) is str and SAFE.fullmatch(self.nonce), "Safe selected nonce")
        require(self.model in MODELS, "Selected native Decision3 model")
        require(type(self.expected_boot_id) is str and bool(self.expected_boot_id), "Selected boot")
        require(type(self.expected_memtotal) is int and self.expected_memtotal > 0, "Selected total")
        require(type(self.expected_gpu_uuid) is str and bool(self.expected_gpu_uuid), "Selected UUID")

    def check_host(self):
        row = host_observation()
        require(
            row["boot_id"] == self.expected_boot_id
            and row.get("MemTotal") == self.expected_memtotal
            and type(row.get("MemAvailable")) is int
            and 0 < row["MemAvailable"] <= self.expected_memtotal,
            "Fresh exact selected boot/host total changed",
        )
        return row


def proc_record(pid):
    path = Path("/proc") / str(pid)
    raw = read_bound(path / "stat", 8192).decode()
    fields = raw.rsplit(")", 1)[1].split()
    return {
        "pid": pid, "state": fields[0], "parent": int(fields[1]),
        "group": int(fields[2]), "session": int(fields[3]), "start_ticks": int(fields[19]),
        "uid": path.stat().st_uid,
    }


def open_pidfd(pid):
    if hasattr(os, "pidfd_open"):
        return os.pidfd_open(pid)
    import ctypes

    library = ctypes.CDLL(None, use_errno=True)
    require(hasattr(library, "pidfd_open"), "Authenticated PID-fd support required")
    function = library.pidfd_open
    function.argtypes, function.restype = [ctypes.c_int, ctypes.c_uint], ctypes.c_int
    result = function(pid, 0)
    if result < 0:
        raise OSError(ctypes.get_errno(), "pidfd_open failed")
    return result


def signal_pidfd(descriptor, signum):
    if hasattr(signal, "pidfd_send_signal"):
        signal.pidfd_send_signal(descriptor, signum)
        return
    import ctypes

    library = ctypes.CDLL(None, use_errno=True)
    require(hasattr(library, "pidfd_send_signal"), "Authenticated PID-fd signaling required")
    function = library.pidfd_send_signal
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(descriptor, signum, None, 0) < 0:
        raise OSError(ctypes.get_errno(), "pidfd_send_signal failed")


def binding(selection, pid, start_ticks):
    return {
        "profile": PROFILE, "nonce": selection.nonce, "model": selection.model,
        "gpu_uuid": selection.expected_gpu_uuid, "pid": pid, "start_ticks": start_ticks,
        "boot_id": selection.expected_boot_id, "mem_total_bytes": selection.expected_memtotal,
    }


def parse_envelope(line):
    require(type(line) is bytes and len(line) <= MAX_ENVELOPE, "Bounded request envelope")
    value = parse(line)
    require(
        type(value) is dict
        and set(value) == {"request_id", "payload_json", "http_sha256", "remaining_seconds"},
        "Exact original request envelope",
    )
    require(type(value["request_id"]) is str and SAFE.fullmatch(value["request_id"]), "Request ID")
    require(type(value["payload_json"]) is str, "Original payload string")
    raw = value["payload_json"].encode()
    require(len(raw) <= MAX_BODY, "Original request body bound")
    require(value["http_sha256"] == hashlib.sha256(raw).hexdigest(), "Original body SHA")
    budget = value["remaining_seconds"]
    require(type(budget) in (int, float) and math.isfinite(budget) and 0 < budget <= 20, "Budget")
    return value


def serve_worker(selection, *, initialize, process_request, finalize, control=None, source=None):
    """Concrete child loop: callbacks own real bootstrap/inference/final checks.

    process_request receives exactly the frozen JSONL envelope bytes. It must
    return the original worker response object, including all original bindings.
    finalize must return a receipt with a typed success boolean. READY says only
    that initialize returned; the future owner must verify its model evidence.
    """
    owned_control = control is None
    if owned_control:
        control = os.fdopen(os.dup(1), "wb", buffering=0)
        os.dup2(2, 1)
        sys.stdout = sys.stderr
    source = sys.stdin.buffer if source is None else source
    identity = binding(selection, os.getpid(), proc_record(os.getpid())["start_ticks"])
    failure, count = None, 0

    def send(value):
        require(type(value) is dict, "Control object")
        for key, expected in identity.items():
            require(key not in value or value[key] == expected, "Worker binding differs: " + key)
        blob = encode({**value, **identity}) + b"\n"
        require(len(blob) <= MAX_FRAME, "Control frame bound")
        pending = memoryview(blob)
        while pending:
            size = control.write(pending)
            require(type(size) is int and size > 0, "Control write failed")
            pending = pending[size:]
        control.flush()

    try:
        selection.check_host()
        details = initialize()
        require(type(details) is dict, "Initialization evidence object")
        send({"event": "ready", "details": details})
        while True:
            line = source.readline(MAX_ENVELOPE + 1)
            require(line and len(line) <= MAX_ENVELOPE and line.endswith(b"\n"), "Complete RPC line")
            if parse(line) == {"command": "close"}:
                break
            envelope = parse_envelope(line)
            count += 1
            require(count <= MAX_REQUESTS, "Worker envelope limit")
            selection.check_host()
            result = process_request(line)
            require(
                type(result) is dict and result.get("event") == "response"
                and result.get("request_id") == envelope["request_id"]
                and result.get("http_sha256") == envelope["http_sha256"]
                and result.get("nonce") == selection.nonce
                and result.get("model") == selection.model
                and result.get("gpu_uuid") == selection.expected_gpu_uuid
                and type(result.get("success")) is bool
                and type(result.get("campaign_stop")) is bool,
                "Original native response binding",
            )
            send(result)
            if result["success"] is not True or result["campaign_stop"] is not False:
                raise TransportError("Native request refused or stopped campaign")
    except BaseException as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)[:2048]}
    finally:
        receipt = None
        try:
            receipt = finalize()
            require(type(receipt) is dict and type(receipt.get("success")) is bool, "Final receipt")
            selection.check_host()
        except BaseException as exc:
            failure = failure or {"type": type(exc).__name__, "message": str(exc)[:2048]}
        success = failure is None and receipt is not None and receipt["success"] is True
        try:
            send({"event": "closed", "success": success, "campaign_stop": not success,
                  "receipt": receipt, "error": failure, "envelopes_seen": count})
        finally:
            if owned_control:
                control.close()
    return 0 if success else 1


class OwnedSession:
    """One fresh owned child. A failed session cannot be restarted or reused."""

    def __init__(self, selection, output_dir):
        self.selection, self.output = selection, Path(output_dir).absolute()
        require(
            self.output.is_dir() and not list(self.output.iterdir())
            and not any(p.is_symlink() for p in (self.output, *self.output.parents)),
            "Fresh ordinary owned output directory",
        )
        self.process, self.identity, self.ready, self.closed = None, None, None, None
        self.stderr_eof = False
        self.blocked, self.buffer, self.stderr_bytes = False, b"", 0
        self.marker, self.tracked = secrets.token_hex(32), {}
        self.lock, self.evidence_lock = threading.RLock(), threading.Lock()
        self.requests, self.last_frame, self.closure = 0, None, None
        self.record("selection.json", {"profile": PROFILE, "selection": asdict(selection)})

    def record(self, name, value):
        require(re.fullmatch(r"[a-z0-9][a-z0-9-]{0,95}\.json", name), "Evidence name")
        blob = encode(value)
        require(len(blob) <= 2 * MAX_FRAME, "Evidence row bound")
        with self.evidence_lock:
            paths = list(self.output.iterdir())
            require(len(paths) < MAX_FILES, "Evidence file bound")
            require(sum(p.stat().st_size for p in paths) + len(blob) <= MAX_BYTES, "Evidence bytes")
            descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=self.output)
            try:
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(blob)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.output / name)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return {"sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)}

    def _pump(self, deadline, *, writable=False):
        require(time.monotonic() < deadline, "Owned transport deadline exceeded")
        readers = [self.process.stdout]
        if not self.stderr_eof:
            readers.append(self.process.stderr)
        ready, writes, _ = select.select(
            readers, [self.process.stdin] if writable else [], [],
            min(0.02, max(0, deadline - time.monotonic())),
        )
        stdout_eof = False
        for stream in ready:
            chunk = os.read(stream.fileno(), 65536)
            if stream is self.process.stdout:
                stdout_eof = not chunk
                self.buffer += chunk
                require(len(self.buffer) <= MAX_FRAME, "Owned stdout frame bound")
            elif chunk:
                self.stderr_bytes += len(chunk)
                require(self.stderr_bytes <= MAX_LOG, "Owned stderr bound")
                with open(self.output / "stderr.log", "ab") as log:
                    log.write(chunk)
            else:
                self.stderr_eof = True
        return bool(writes), stdout_eof

    def _send(self, value, deadline):
        pending = memoryview(encode(value) + b"\n")
        require(len(pending) <= MAX_ENVELOPE, "Bounded outgoing envelope")
        while pending:
            writable, _ = self._pump(deadline, writable=True)
            if writable:
                count = os.write(self.process.stdin.fileno(), pending[:65536])
                require(count > 0, "Owned stdin write failed")
                pending = pending[count:]

    def _read(self, deadline):
        while b"\n" not in self.buffer:
            _, eof = self._pump(deadline)
            require(not eof, "Owned worker ended before complete control frame")
        require(time.monotonic() < deadline, "Owned frame returned after deadline")
        line, self.buffer = self.buffer.split(b"\n", 1)
        value = parse(line)
        self.last_frame = value
        require(type(value) is dict, "Control frame object")
        for key, expected in self.identity.items():
            require(value.get(key) == expected and type(value.get(key)) is type(expected),
                    "Control identity differs: " + key)
        try:
            current = proc_record(self.process.pid)
            require(current["start_ticks"] == self.identity["start_ticks"], "Worker PID reused")
        except (FileNotFoundError, ProcessLookupError):
            require(value.get("event") == "closed" and self.process.poll() is not None,
                    "Live worker process identity missing")
        self.selection.check_host()
        return value

    def start(self, command, *, env=None, cwd=None, timeout=30):
        with self.lock:
            require(self.process is None and not self.blocked, "One fresh child per session")
            require(type(command) in (list, tuple) and command and all(type(x) is str for x in command),
                    "Explicit argv, never shell")
            require(Path(command[0]).is_absolute(), "Absolute selected interpreter")
            self.selection.check_host()
            child_env = dict(os.environ if env is None else env)
            child_env[MARKER] = self.marker
            self.process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                env=child_env, cwd=cwd, start_new_session=True, bufsize=0,
            )
            try:
                row = proc_record(self.process.pid)
                require(row["group"] == row["session"] == self.process.pid, "Fresh owned session")
                self.identity = binding(self.selection, self.process.pid, row["start_ticks"])
                self.tracked[self.process.pid] = row["start_ticks"]
                os.set_blocking(self.process.stdin.fileno(), False)
                ready = self._read(time.monotonic() + timeout)
                if ready.get("event") == "closed":
                    self.closed = ready
                require(ready.get("event") == "ready" and type(ready.get("details")) is dict
                        and self.process.poll() is None,
                        "Authenticated READY required")
                self.ready = ready
                self.record("ready.json", ready)
                return ready
            except BaseException as exc:
                self.blocked = True
                exc.transport_receipt = self.close(grace=0.1, term=0.2, kill=0.2)
                raise

    def request(self, body, *, request_id, deadline):
        with self.lock:
            require(self.ready is not None and not self.blocked and self.closed is None,
                    "No ready unblocked owned arm")
            self.selection.check_host()
            remaining = deadline - time.monotonic()
            envelope = {"request_id": request_id, "payload_json": body.decode(),
                        "http_sha256": hashlib.sha256(body).hexdigest(),
                        "remaining_seconds": min(20, remaining)}
            parse_envelope(encode(envelope))
            self.requests += 1
            require(self.requests <= MAX_REQUESTS, "Owned envelope limit")
            try:
                self._send(envelope, deadline)
                result = self._read(deadline)
                require(
                    result.get("event") == "response" and result.get("request_id") == request_id
                    and result.get("http_sha256") == envelope["http_sha256"]
                    and type(result.get("success")) is bool
                    and type(result.get("campaign_stop")) is bool,
                    "Bound response request identity",
                )
                return result
            except BaseException:
                self.blocked = True
                raise

    def observe_owned(self):
        self.selection.check_host()
        records, vanished, unknown = {}, [], []
        entries = list(Path("/proc").iterdir())
        require(len(entries) <= 65536, "Process observation bound")
        for path in entries:
            if path.name.isdecimal():
                try:
                    row = proc_record(int(path.name))
                    if row["uid"] == os.getuid():
                        records[row["pid"]] = row
                except (FileNotFoundError, ProcessLookupError):
                    vanished.append(int(path.name))
                except PermissionError:
                    unknown.append({"pid": int(path.name), "reason": "Unreadable process identity"})
        owned = {pid for pid, start in self.tracked.items()
                 if pid in records and records[pid]["start_ticks"] == start}
        leader = self.process.pid
        if leader in records and self.identity is not None:
            require(records[leader]["start_ticks"] == self.identity["start_ticks"], "Leader PID reused")
        changed = True
        while changed:
            descendants = {pid for pid, row in records.items() if row["parent"] in owned} - owned
            changed = bool(descendants)
            owned.update(descendants)
        start = self.identity["start_ticks"] if self.identity else 0
        for pid, row in records.items():
            if row["start_ticks"] < start:
                continue
            try:
                environment = read_bound(Path("/proc") / str(pid) / "environ", 1024**2)
                if (MARKER + "=" + self.marker).encode() in environment.split(b"\x00"):
                    owned.add(pid)
            except (FileNotFoundError, ProcessLookupError):
                vanished.append(pid)
            except PermissionError:
                unknown.append({"pid": pid, "reason": "Recent ownership unreadable"})
        unknown.extend({"pid": pid, "reason": "Unattributed owned process group member"}
                       for pid, row in records.items() if row["group"] == leader and pid not in owned)
        require(len(owned) + len(unknown) <= 4096, "Owned process count bound")
        for pid in owned:
            self.tracked[pid] = records[pid]["start_ticks"]
        return {"owned": [records[pid] for pid in sorted(owned)], "unknown": unknown,
                "vanished": sorted(set(vanished))}

    def _signal_owned(self, observed, signum):
        for row in observed["owned"]:
            try:
                descriptor = open_pidfd(row["pid"])
            except ProcessLookupError:
                continue
            try:
                current = proc_record(row["pid"])
                require(current["start_ticks"] == row["start_ticks"], "Signal PID identity changed")
                signal_pidfd(descriptor, signum)
            except (FileNotFoundError, ProcessLookupError):
                pass
            finally:
                os.close(descriptor)

    def _wait_absent(self, seconds):
        until = time.monotonic() + seconds
        while True:
            self.process.poll()
            observed = self.observe_owned()
            if not observed["owned"] or time.monotonic() >= until:
                return observed
            time.sleep(0.01)

    def close(self, *, grace=2, term=2, kill=2):
        with self.lock:
            if self.closure is not None:
                return self.closure
            errors, forced, observed = [], False, {"owned": [], "unknown": []}
            try:
                if self.process is not None:
                    deadline = time.monotonic() + grace
                    try:
                        if self.process.poll() is None:
                            self._send({"command": "close"}, deadline)
                        while self.closed is None:
                            frame = self._read(deadline)
                            if frame.get("event") == "closed":
                                self.closed = frame
                            else:
                                self.record("late-control.json", frame)
                    except Exception as exc:
                        errors.append({"stage": "closed_frame", "message": str(exc)[:2048]})
                    observed = self._wait_absent(grace)
                    for signum, seconds in ((signal.SIGTERM, term), (signal.SIGKILL, kill)):
                        if not observed["owned"]:
                            break
                        forced = True
                        self._signal_owned(observed, signum)
                        observed = self._wait_absent(seconds)
                    # Two fresh observations are necessary to confirm closure.
                    first = self.observe_owned()
                    second = self.observe_owned()
                    observed = second
                    absent = not first["owned"] and not second["owned"]
                    known = not first["unknown"] and not second["unknown"]
                    exited = self.process.poll() is not None
                else:
                    absent = known = exited = True
            except Exception as exc:
                absent = known = exited = False
                errors.append({"stage": "owned_cleanup", "message": str(exc)[:2048]})
            finally:
                if self.process is not None:
                    for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                        stream.close()
            closed_ok = (
                type(self.closed) is dict and self.closed.get("success") is True
                and self.closed.get("campaign_stop") is False
                and self.closed.get("error") is None
                and type(self.closed.get("receipt")) is dict
                and self.closed["receipt"].get("success") is True
                and type(self.closed.get("envelopes_seen")) is int
                and self.closed["envelopes_seen"] == self.requests
            )
            complete = absent and known and exited
            success = (complete and closed_ok and not errors and not forced and not self.blocked
                       and self.process is not None and self.process.returncode == 0)
            self.closure = {
                "profile": PROFILE, "selection": asdict(self.selection), "identity": self.identity,
                "closed": self.closed, "direct_child_exit_confirmed": exited,
                "observed_process_exit_confirmed": complete,
                "observation_scope": "current_lineage_and_inherited_marker_only",
                "owned_exit_confirmed": False, "whole_tree_exit_confirmed": False,
                "external_owner_boundary_required": True,
                "external_owner_boundary_status": "not_verified",
                "continuation_authorized": False,
                "continuation_refusal_reason": "external_kernel_ownership_boundary_required",
                "success": bool(success), "success_scope": "transport_and_observed_processes_only",
                "forced_cleanup": forced, "blocked": self.blocked, "errors": errors,
                "final_observation": observed,
                "returncode": self.process.returncode if self.process is not None else None,
                "model_qualified": False,
            }
            self.record("closure.json", self.closure)
            return self.closure


class LocalGateway:
    """Preserve HTTP bytes; serialize queue/RPC/evidence under one local deadline."""

    def __init__(self, session, *, seconds=20):
        require(type(seconds) in (float, int) and 0 < seconds <= 20, "Gateway budget")
        self.session, self.seconds = session, seconds
        self.lock, self.blocked, self.counter = threading.Lock(), False, 0
        self.state_lock = threading.RLock()

    def block(self):
        with self.state_lock:
            self.blocked = self.session.blocked = True

    def request(self, body, *, started=None, publish=None):
        started = time.monotonic() if started is None else started
        deadline, request_id = started + self.seconds, secrets.token_hex(16)
        row = {"profile": PROFILE, "request_id": request_id,
               "http_sha256": hashlib.sha256(body).hexdigest(), "raw_http_hex": body.hex(),
               "identity": self.session.identity, "gateway_action_authorized": False}
        status, response = 504, encode({"detail": "Decision request unavailable or late; no action"})
        acquired = self.lock.acquire(timeout=max(0, deadline - time.monotonic()))
        try:
            require(acquired, "Observable local queue deadline exceeded")
            require(not self.blocked and not self.session.blocked, "Gateway arm blocked")
            require(type(body) is bytes and len(body) <= MAX_BODY, "Original bounded HTTP body")
            payload = parse(body)
            require(type(payload) is dict and payload.get("model") == self.session.selection.model,
                    "Original HTTP model must match active arm")
            self.counter += 1
            require(self.counter <= MAX_REQUESTS, "Gateway envelope bound")
            result = self.session.request(body, request_id=request_id, deadline=deadline)
            row["worker"] = result
            require(result["success"] is True and result["campaign_stop"] is False
                    and type(result.get("wire_body_utf8")) is str, "Worker refused or returned late")
            response = result["wire_body_utf8"].encode()
            require(len(response) <= MAX_FRAME, "Native wire response bound")
            parse(response)
            require(time.monotonic() < deadline, "Full gateway deadline exceeded")
            status = 200
            row["gateway_action_authorized"] = True
        except Exception as exc:
            self.block()
            row["error"] = {"type": type(exc).__name__, "message": str(exc)[:2048]}
            status, response = 504, encode({"detail": "Decision request failed or late; no action"})
        try:
            row.update(status=status, response_sha256=hashlib.sha256(response).hexdigest(),
                       gateway_elapsed_seconds=time.monotonic() - started)
            pin = self.session.record("request-" + request_id + ".json", row)
            if status == 200 and (time.monotonic() >= deadline or self.blocked):
                self.block()
                status, response = 504, encode({"detail": "Final evidence exceeded deadline; no action"})
                row.update(status=status, gateway_action_authorized=False, late_after_publication=True,
                           response_sha256=hashlib.sha256(response).hexdigest())
                pin = self.session.record("request-" + request_id + ".json", row)
            evidence = {"request_id": request_id, "request_pin": pin, "status": status,
                        "response_sha256": hashlib.sha256(response).hexdigest(), "deadline": deadline}
            if publish is not None:
                return publish(status, response, evidence)
            return status, response, evidence
        except BaseException:
            self.block()
            raise
        finally:
            if acquired:
                self.lock.release()

    def before_delivery(self, evidence, response):
        """Bind the final response to its durable row immediately before headers."""
        path = self.session.output / ("request-" + evidence["request_id"] + ".json")
        try:
            saved = read_bound(path, 2 * MAX_FRAME)
            require(hashlib.sha256(saved).hexdigest() == evidence["request_pin"]["sha256"]
                    and hashlib.sha256(response).hexdigest() == evidence["response_sha256"],
                    "Final delivery response/receipt binding differs")
        except BaseException:
            self.block()
            raise
        if evidence["status"] == 200 and (time.monotonic() >= evidence["deadline"]
                                          or self.blocked or self.session.blocked):
            self.block()
            row = parse(saved)
            response = encode({"detail": "Delivery preparation exceeded deadline; no action"})
            row.update(status=504, gateway_action_authorized=False, late_before_delivery=True,
                       response_sha256=hashlib.sha256(response).hexdigest())
            evidence = {**evidence, "status": 504, "response_sha256": row["response_sha256"],
                        "request_pin": self.session.record(path.name, row)}
        return evidence["status"], response, evidence

    def delivery_failed(self, evidence, response, error):
        self.block()
        require(evidence["response_sha256"] == hashlib.sha256(response).hexdigest(), "Delivery SHA")
        path = self.session.output / ("request-" + evidence["request_id"] + ".json")
        saved = read_bound(path, 2 * MAX_FRAME)
        require(hashlib.sha256(saved).hexdigest() == evidence["request_pin"]["sha256"],
                "Durable delivery request changed")
        self.session.record("delivery-" + evidence["request_id"] + ".json", {
            "profile": PROFILE, "request": evidence,
            "error": {"type": type(error).__name__, "message": str(error)[:2048]},
            "delivered": False,
        })


class _BoundedServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler):
        self.slots = threading.BoundedSemaphore(8)
        self.condition, self.inflight = threading.Condition(), 0
        super().__init__(address, handler)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        request.settimeout(20)
        with self.condition:
            self.inflight += 1
        try:
            super().process_request(request, client_address)
        except BaseException:
            with self.condition:
                self.inflight -= 1
                self.condition.notify_all()
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self.condition:
                self.inflight -= 1
                self.condition.notify_all()
            self.slots.release()


class LoopbackHTTP:
    """Explicit authenticated loopback listener; construction starts no thread."""

    def __init__(self, gateway, *, token, port=0):
        require(type(token) is str and 32 <= len(token) <= 256 and token.isascii(), "Private bearer")
        self.gateway, self.token, self.port = gateway, token, port
        self.server, self.thread = None, None

    def start(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def publish(self, status, response, evidence):
                """Request lock stays held; prepare all bytes before the final state gate."""
                self.publication_started = True
                try:
                    while True:
                        self._headers_buffer = []
                        self.send_response(status)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(response)))
                        self.send_header("Connection", "close")
                        prepared_headers = b"".join(self._headers_buffer) + b"\r\n"
                        prepared_status = status
                        with owner.gateway.state_lock:
                            status, response, evidence = owner.gateway.before_delivery(evidence, response)
                            if status != prepared_status:
                                continue
                            self.wfile.write(prepared_headers)
                            self._headers_buffer = []
                        break
                    self.wfile.write(response)
                    self.wfile.flush()
                    return status, response, evidence
                except Exception as exc:
                    owner.gateway.delivery_failed(evidence, response, exc)
                    raise

            def do_POST(self):
                started = time.monotonic()
                self.close_connection = True
                try:
                    require(self.path == "/predict", "Exact local predict route")
                    auth = self.headers.get_all("Authorization", [])
                    if len(auth) != 1 or not hmac.compare_digest(auth[0], "Bearer " + owner.token):
                        self.send_error(401)
                        return
                    lengths = self.headers.get_all("Content-Length", [])
                    require(len(lengths) == 1 and lengths[0].isdecimal()
                            and self.headers.get("Transfer-Encoding") is None, "HTTP body framing")
                    length = int(lengths[0])
                    require(0 < length <= MAX_BODY, "HTTP body byte bound")
                    self.connection.settimeout(max(0.001, owner.gateway.seconds -
                                                   (time.monotonic() - started)))
                    body = self.rfile.read(length)
                    require(len(body) == length, "Incomplete HTTP body")
                    owner.gateway.request(body, started=started, publish=self.publish)
                except Exception:
                    if not getattr(self, "publication_started", False):
                        try:
                            self.send_error(400)
                        except OSError:
                            pass

        require(self.server is None, "Listener starts once")
        self.server = _BoundedServer(("127.0.0.1", self.port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.02},
                                       daemon=True)
        self.thread.start()
        return "http://127.0.0.1:" + str(self.server.server_address[1])

    def close(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=1)
            require(not self.thread.is_alive(), "Loopback listener did not close")
            with self.server.condition:
                complete = self.server.condition.wait_for(
                    lambda: self.server.inflight == 0, timeout=self.gateway.seconds + 0.1,
                )
            require(complete, "Loopback request handlers remain active")
