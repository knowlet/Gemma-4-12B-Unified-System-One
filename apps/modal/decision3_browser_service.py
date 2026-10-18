"""Prospective loopback HTTP to authenticated Modal RPC Decision3 service.

Inventory/prepare never imports Modal. Actual use requires explicit --run,
reviewed seventeen-source inventory and an externally selected complete public
weight-staging profile. No browser runner or text helper is embedded here.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib
import json
import math
import os
import re
import secrets
import select
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
SCRIPT = ROOT / "apps/modal/decision3_browser_service.py"
WORKER = ROOT / "scripts/jev_decision3_browser_worker.py"
CORE = ROOT / "scripts/jev_decision3_public_dev22_gpu_replay.py"
WRAPPER = ROOT / "apps/modal/decision3_public_replay.py"
WIRE = ROOT / "scripts/jev_decision3_browser_wire.py"
CORE_SHA = "69baedec7efeb7b361ef0b9f8d3c7fdcc70aac30cc19ffba3a74b2a88f54a029"
WRAPPER_SHA = "cf23bc6c4ff0068c622e2c6bb1a3ba7122a6093d5afda96a66dced0816826eac"
WIRE_SHA = "c048d7779e60fce5d1a56b38e26b9ddf5ed9e1ff9c6adc478786d0267cca3027"
MODELS = ("d3", "d3-flash", "d3-mini", "d3-nano", "d3-lite", "d3-edge")
INTERPRETER = "/opt/decision3/bin/python"
MAX_BODY, MAX_FRAME, MAX_LOG = 1_000_000, 16 * 1024**2, 16 * 1024**2
SECONDS, WATCHDOG = 20.0, 1800.0
SOURCE_SCHEMA = "decision3-browser-source-inventory-v1"
WEIGHT_SCHEMA = "decision3-browser-complete-weight-profile-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encode(value):
    # Never sort native questions, criteria or instruction objects.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def parse(blob):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    def finite_float(value):
        result = float(value)
        require(math.isfinite(result), "Nonfinite JSON number")
        return result

    return json.loads(
        blob,
        object_pairs_hook=pairs,
        parse_float=finite_float,
        parse_constant=lambda value: require(False, "Nonfinite JSON " + value),
    )


def ordinary(path):
    path = Path(path).absolute()
    require(
        ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "Ordinary path required",
    )
    return path


def pin(path):
    path = ordinary(path)
    before = path.stat()
    require(
        stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= 64 * 1024**2,
        "Bounded ordinary source required",
    )
    blob = path.read_bytes()
    after = path.stat()
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        and len(blob) == before.st_size,
        "Source changed during read",
    )
    return {"sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)}


def literal(path, name):
    values = [
        ast.literal_eval(node.value)
        for node in ast.parse(ordinary(path).read_bytes()).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    ]
    require(len(values) == 1, "One fixed literal required: " + name)
    return values[0]


def source_inventory():
    require(
        pin(CORE)["sha256"] == CORE_SHA
        and pin(WRAPPER)["sha256"] == WRAPPER_SHA
        and pin(WIRE)["sha256"] == WIRE_SHA,
        "Reviewed core/wrapper/wire source differs",
    )
    frozen = literal(CORE, "SOURCE_PINS")
    require(len(frozen) == 12, "Exactly twelve frozen production sources required")
    rows = []
    for label, sha in frozen.items():
        path = ordinary(ROOT / label)
        actual = pin(path)
        require(actual["sha256"] == sha, "Frozen source12 differs")
        rows.append(
            {"path": str(path), "pin": actual, "classification": "reviewed_public_source12"}
        )
    rows.extend(
        {
            "path": str(path),
            "pin": pin(path),
            "classification": "reviewed_service_composition_source",
        }
        for path in (CORE, WRAPPER, WIRE, WORKER, SCRIPT)
    )
    rows.sort(key=lambda row: row["path"])
    require(
        len(rows) == len({row["path"] for row in rows}) == 17,
        "Exactly seventeen named sources required",
    )
    return {
        "schema_id": SOURCE_SCHEMA,
        "files": 17,
        "rows": rows,
        "weights_uploaded": False,
        "credentials_uploaded": False,
        "collection_or_parent_uploaded": False,
    }


def validate_inventory(blob):
    supplied = parse(blob)
    require(
        supplied == source_inventory() and blob == encode(supplied),
        "Externally reviewed exact seventeen-source inventory required",
    )
    return supplied


def validate_weight_profile(blob, expected_sha, identities):
    require(
        len(blob) <= 8 * 1024**2 and hashlib.sha256(blob).hexdigest() == expected_sha,
        "Externally selected bounded complete weight profile required",
    )
    profile = parse(blob)
    require(
        profile.get("schema_id") == WEIGHT_SCHEMA and profile.get("wrapper_sha256") == WRAPPER_SHA,
        "Reviewed downloader weight profile required",
    )
    # These are the original file bytes from a future closed download export,
    # never guessed model receipts or rewritten historical success records.
    closed = profile["closed_download"]
    require(
        set(closed) == {"index_json", "controller_json", "result_json"}
        and all(type(value) is str for value in closed.values()),
        "Original closed download JSON bytes required",
    )
    index, controller, result = (
        parse(closed[key].encode()) for key in ("index_json", "controller_json", "result_json")
    )
    require(
        index.get("complete_owned_exit") is True
        and controller.get("owned_exit_confirmed") is True
        and controller.get("child_success") is True,
        "Complete successful owned public download required",
    )
    for label, key in (
        ("controller-receipt.json", "controller_json"),
        ("download-result.json", "result_json"),
    ):
        raw = closed[key].encode()
        require(
            index["files"][label]
            == {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)},
            "Original download bytes do not match closed index",
        )
    require(
        result.get("token") is False
        and result.get("model_loads") == 0
        and result.get("GPU_requested") is False
        and set(result["models"]) == set(MODELS),
        "Public CPU staging profile scope differs",
    )
    for name, identity in identities.items():
        row = result["models"][name]
        require(
            row["revision"] == identity["revision"]
            and row["manifest_sha256"] == identity["manifest_sha256"]
            and row.get("files"),
            "Six immutable staged model identities required",
        )
    return profile


def load_staging():
    require(pin(WRAPPER)["sha256"] == WRAPPER_SHA, "Reviewed staging source required")
    tree = ast.parse(WRAPPER.read_bytes())
    boundary = next(
        index
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    )
    namespace = {"__name__": "reviewed_d3_browser_staging", "__file__": str(WRAPPER)}
    exec(
        compile(ast.Module(body=tree.body[:boundary], type_ignores=[]), str(WRAPPER), "exec"),
        namespace,
    )
    return type("ReviewedStaging", (), namespace)


def save(path, value):
    path = ordinary(path)
    blob = encode(value)
    require(len(blob) <= MAX_FRAME, "Single local evidence byte bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def gpu_identity():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        timeout=30,
    )
    require(
        len(result.stdout) <= 65536 and len(result.stdout.decode().strip().splitlines()) == 1,
        "Actual single GPU required",
    )
    name, uuid = (value.strip() for value in result.stdout.decode().strip().split(","))
    require("H200" in name and uuid.startswith("GPU-"), "Actual H200 UUID required")
    return {"name": name, "uuid": uuid}


class Session:
    """One SDK parent allocation, six ordered fresh owned model subprocesses."""

    def __init__(self, run_id, manifest_blob, profile_blob, profile_sha):
        require(
            re.fullmatch(r"[a-z0-9][a-z0-9-]{0,47}", run_id), "Fresh bounded campaign ID required"
        )
        self.manifest = validate_inventory(manifest_blob)
        self.staging = load_staging()
        # All seventeen files are already baked by add_local_file(copy=True).
        # Clear any cached absent scripts directory before the reviewed loader.
        importlib.invalidate_caches()
        self.core = self.staging.load_core()
        self.identities = self.core.authenticate_source(CORE_SHA)
        self.profile = validate_weight_profile(profile_blob, profile_sha, self.identities)
        self.output = ordinary(self.core.OUTPUT_ROOT / ("browser-" + run_id))
        require(not self.output.exists(), "Fresh cloud browser output required")
        self.output.mkdir(parents=True)
        self.nonce, self.gpu = secrets.token_hex(24), gpu_identity()
        self.mounts = {
            alias: self.staging.capture_mount(alias) for alias in ("/weights", "/export")
        }
        try:
            self.core.write(self.output / "source-inventory.json", self.manifest)
            self.core.write(self.output / "weight-profile.json", self.profile)
            self.staging.stage_cache(
                {"mounts": self.mounts, "output": str(self.output)}, self.core, self.identities
            )
        except BaseException as primary:
            # No child can exist before staging completes. Preserve its actual
            # partial failure tree without constructing a usable session.
            failed = {
                "stage": "failed_before_model_child",
                "success": False,
                "owned_exit_confirmed": True,
                "model_loads": 0,
                "arms": [],
                "root": self.output.name,
                "export_index": None,
                "gpu": self.gpu,
                "error": {"type": type(primary).__name__, "message": str(primary)},
            }
            try:
                self.core.write(
                    self.output / "initialization-failure-receipt.json", failed, terminal=True
                )
                self.core.write(
                    self.output / "export-index.json",
                    {
                        "root": self.output.name,
                        "files": self.staging.tree_inventory(self.output),
                        "complete_owned_exit": True,
                    },
                    terminal=True,
                )
                failed["export_index"] = self.staging.publish_output(
                    self.output, self.mounts["/export"], True
                )
            except BaseException as secondary:
                primary.add_note(
                    "Pre-model failure evidence publication also failed: " + repr(secondary)
                )
            failed["error"]["notes"] = list(getattr(primary, "__notes__", []))
            primary.decision3_initialization_failure = failed
            raise
        self.sequence, self.next_arm, self.active, self.blocked = 0, 0, None, False
        self.process, self.stderr, self.buffer, self.marker = None, None, b"", None
        self.start_ticks, self.closures, self.observations = None, [], []
        self.last_control, self.identity_errors, self.protocol_history = None, [], []

    def check(self, nonce, *, cleanup=False):
        require(
            nonce == self.nonce, "Persistent allocation nonce differs; no automatic replacement"
        )
        try:
            require(gpu_identity() == self.gpu, "Persistent H200 UUID changed")
        except BaseException as exc:
            self.blocked = True
            self.identity_errors.append({"type": type(exc).__name__, "message": str(exc)})
            if not cleanup or not isinstance(exc, Exception):
                raise

    def record(self, name, value):
        self.core.write(self.output / name, value)

    def read_control(self, timeout=WATCHDOG):
        started = time.monotonic()
        while b"\n" not in self.buffer:
            require(
                len(self.buffer) <= MAX_FRAME and self.stderr.stat().st_size <= MAX_LOG,
                "Owned worker protocol/log bound exceeded",
            )
            require(
                time.monotonic() - started < timeout,
                "Owned worker infrastructure watchdog exceeded",
            )
            ready, _, _ = select.select([self.process.stdout], [], [], 0.05)
            if ready:
                chunk = os.read(self.process.stdout.fileno(), 65536)
                require(chunk, "Owned worker ended without expected control response")
                self.buffer += chunk
            elif self.process.poll() is not None:
                raise RuntimeError("Owned worker exited before control response")
        line, self.buffer = self.buffer.split(b"\n", 1)
        require(len(line) <= MAX_FRAME, "Owned worker control frame bound exceeded")
        value = parse(line)
        self.last_control = (
            value  # Preserve an available raw response even if its checkpoint fails.
        )
        # Every raw request/native result remains in its worker journal and local
        # RPC row. Avoid another file per response under the frozen2048-file cap.
        summary = {
            key: value.get(key)
            for key in (
                "event",
                "request_id",
                "http_sha256",
                "nonce",
                "model",
                "gpu_uuid",
                "success",
                "campaign_stop",
            )
        }
        if value.get("event") != "response":
            summary["control"] = value
        else:
            summary["journal_path"] = value.get("journal", {}).get("journal_path")
        self.protocol_history.append(
            {
                "sequence": self.sequence,
                "sha256": hashlib.sha256(line).hexdigest(),
                "size_bytes": len(line),
                **summary,
            }
        )
        require(
            len(self.protocol_history) <= 6 * (192 + 2),
            "Bounded campaign protocol history required",
        )
        self.record("protocol-history.json", self.protocol_history)
        self.sequence += 1
        return value

    def observe(self):
        """Fresh process/marker observation; unknown sessions never authorize an arm."""
        marker = b"JEV_D3_BROWSER_OWNER=" + self.marker.encode()
        records, uncertain = {}, []
        paths = list(Path("/proc").iterdir())
        require(len(paths) <= 65536, "Bounded process observation required")
        for path in paths:
            if path.name.isdecimal():
                try:
                    records[int(path.name)] = self.staging.proc_record(path)
                except (FileNotFoundError, ProcessLookupError):
                    continue
        leader = records.get(self.process.pid)
        owned = (
            {self.process.pid}
            if leader is not None
            and leader["group"] == leader["session"] == self.process.pid
            and leader["start_ticks"] == self.start_ticks
            else set()
        )
        changed = True
        while changed:
            children = {pid for pid, row in records.items() if row["parent"] in owned} - owned
            changed = bool(children)
            owned.update(children)
        for pid, row in records.items():
            if (
                pid not in owned
                and self.start_ticks is not None
                and row["start_ticks"] < self.start_ticks
            ):
                continue
            try:
                environment = self.staging.proc_read(Path("/proc") / str(pid) / "environ", 1024**2)
            except (FileNotFoundError, ProcessLookupError):
                continue
            except PermissionError:
                uncertain.append({"pid": pid, "reason": "Recent process ownership unreadable"})
                continue
            if marker in environment.split(b"\x00"):
                owned.add(pid)
        unknown = uncertain + [
            {
                "pid": pid,
                "group": records[pid]["group"],
                "reason": "Owned process outside the tracked worker group",
            }
            for pid in sorted(owned)
            if records[pid]["group"] != self.process.pid
        ]
        group_members = {pid for pid, row in records.items() if row["group"] == self.process.pid}
        unknown.extend(
            {"pid": pid, "reason": "Tracked group identity/ownership cannot be proved"}
            for pid in sorted(group_members - owned)
        )
        require(len(owned) + len(unknown) <= 4096, "Bounded owned process set required")
        row = {
            "owned_pids": sorted(owned),
            "unknown": unknown,
            "group_safe_to_signal": bool(group_members) and group_members <= owned,
        }
        if unknown:
            self.blocked = True
        self.observations.append(row)
        self.observations = self.observations[-16:]
        return row

    def reap(self):
        if self.process is None:
            return True
        until = time.monotonic() + 2
        while self.process.poll() is None and time.monotonic() < until:
            self.observe()
            time.sleep(0.02)
        # Repeat discovery during cleanup; a compiler/session may appear after TERM.
        for signum in (signal.SIGTERM, signal.SIGKILL):
            observed = self.observe()
            alive = self.staging.group_alive(self.process.pid)
            if not alive and self.process.poll() is not None:
                break
            require(
                observed["group_safe_to_signal"],
                "Live process group cannot be authenticated; no global kill",
            )
            try:
                os.killpg(self.process.pid, signum)
            except ProcessLookupError:
                pass
            until = time.monotonic() + 10
            while time.monotonic() < until:
                self.observe()
                if self.process.poll() is not None and not self.staging.group_alive(
                    self.process.pid
                ):
                    break
                time.sleep(0.05)
        final = self.observe()
        confirmed = (
            self.process.poll() is not None
            and not self.staging.group_alive(self.process.pid)
            and not final["owned_pids"]
            and not final["unknown"]
        )
        self.record(
            "owned-process-observations.json",
            {"observations": self.observations, "owned_exit_confirmed": confirmed},
        )
        require(confirmed, "Whole owned worker exit unconfirmed; campaign halted")
        self.last_returncode = self.process.returncode
        self.process.stdin.close()
        self.process.stdout.close()
        self.process = None
        return True

    def begin(self, model, nonce=None):
        if nonce is not None:
            self.check(nonce)
        require(
            not self.blocked
            and self.active is None
            and self.next_arm < len(MODELS)
            and model == MODELS[self.next_arm],
            "Only the next ordered arm may begin after confirmed exit",
        )
        self.check(self.nonce)
        arm = self.output / model
        arm.mkdir()
        snapshot = self.core.resource_snapshot()
        config = {
            "model": model,
            "alias": model,
            "output_dir": str(arm),
            "nonce": self.nonce,
            "expected_boot_id": snapshot["boot_id"],
            "expected_memtotal": snapshot["proc_meminfo"]["MemTotal"],
            "expected_gpu_uuid": self.gpu["uuid"],
            "source_manifest": self.manifest,
        }
        self.core.write(arm / "config.json", config)
        self.stderr = arm / "stderr.log"
        self.marker = self.nonce + ":" + model
        environment = self.staging.child_environment("replay")
        environment["JEV_D3_BROWSER_OWNER"] = self.marker
        try:
            with self.stderr.open("xb") as log:
                self.process = subprocess.Popen(
                    [INTERPRETER, "-B", "-s", str(WORKER), "--worker", str(arm / "config.json")],
                    cwd=ROOT,
                    env=environment,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=log,
                    start_new_session=True,
                    bufsize=0,
                )
            self.start_ticks = self.staging.proc_record(Path("/proc") / str(self.process.pid))[
                "start_ticks"
            ]
            ready = self.read_control()
            require(
                ready["event"] == "ready"
                and ready["model"] == model
                and ready["alias"] == model
                and ready["nonce"] == self.nonce
                and ready["pid"] == self.process.pid
                and ready["gpu_uuid"].removeprefix("GPU-").lower()
                == self.gpu["uuid"].removeprefix("GPU-").lower(),
                "Actual worker identity/UUID readiness differs",
            )
            self.active = model
            return {
                "success": True,
                "nonce": self.nonce,
                "model": model,
                "gpu": self.gpu,
                "ready": ready,
            }
        except BaseException as primary:
            self.blocked = True
            try:
                self.reap()
            except BaseException as secondary:
                primary.add_note("Owned cleanup also failed: " + repr(secondary))
            raise

    def predict(self, nonce, request_id, body, remaining):
        started = time.monotonic()
        self.check(nonce)
        require(
            not self.blocked
            and self.active is not None
            and type(body) is bytes
            and len(body) <= MAX_BODY
            and 0 < remaining <= SECONDS,
            "Active bounded request required",
        )
        digest = hashlib.sha256(body).hexdigest()
        self.last_control = None
        try:
            envelope = {
                "request_id": request_id,
                "payload_json": body.decode("utf-8"),
                "http_sha256": digest,
                "remaining_seconds": remaining - (time.monotonic() - started),
            }
            frame = encode(envelope) + b"\n"
            require(
                len(frame) <= 4 * 1024**2 and envelope["remaining_seconds"] > 0,
                "IPC budget/frame refused",
            )
            require(self.process.stdin.write(frame) == len(frame), "Short owned request write")
            self.process.stdin.flush()
            result = self.read_control()
            require(
                result["event"] == "response"
                and result["request_id"] == request_id
                and result["http_sha256"] == digest
                and result["nonce"] == self.nonce
                and result["model"] == self.active
                and result["gpu_uuid"].removeprefix("GPU-").lower()
                == self.gpu["uuid"].removeprefix("GPU-").lower(),
                "Owned response binding differs",
            )
            self.check(nonce)
            if (
                not result["success"]
                or result["campaign_stop"]
                or time.monotonic() - started >= remaining
            ):
                self.blocked = True
                result["late_wire_body_utf8"] = result.get("wire_body_utf8")
                result["success"], result["campaign_stop"], result["wire_body_utf8"] = (
                    False,
                    True,
                    None,
                )
            return result
        except BaseException as primary:
            self.blocked = True
            raw = self.last_control
            cleanup_error = None
            try:
                self.reap()
            except BaseException as secondary:
                primary.add_note("Owned cleanup also failed: " + repr(secondary))
                cleanup_error = {"type": type(secondary).__name__, "message": str(secondary)}
            if not isinstance(primary, Exception):
                raise
            return {
                "event": "response",
                "request_id": request_id,
                "http_sha256": digest,
                "nonce": self.nonce,
                "model": self.active,
                "gpu_uuid": self.gpu["uuid"],
                "success": False,
                "campaign_stop": True,
                "wire_body_utf8": None,
                "controller_error": {
                    "type": type(primary).__name__,
                    "message": str(primary),
                    "notes": list(getattr(primary, "__notes__", [])),
                },
                "cleanup_error": cleanup_error,
                "available_worker_response": raw,
            }

    def end(self, nonce):
        self.check(nonce, cleanup=True)
        require(self.process is not None, "No active owned arm")
        model, closed, confirmed = self.active, None, False
        try:
            if self.process.poll() is None:
                try:
                    self.process.stdin.write(encode({"command": "close"}) + b"\n")
                    self.process.stdin.flush()
                except BrokenPipeError:
                    pass  # A failed request may already be finalizing its CLOSED frame.
            closed = self.read_control()
            require(closed["event"] == "closed", "Actual final worker receipt required")
        except BaseException as exc:
            self.blocked = True
            closed = {
                "event": "closed",
                "success": False,
                "campaign_stop": True,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        finally:
            try:
                confirmed = self.reap()
            except BaseException as exc:
                self.blocked = True
                closed = closed or {"event": "closed", "success": False, "campaign_stop": True}
                closed.setdefault("cleanup_errors", []).append(
                    {"type": type(exc).__name__, "message": str(exc)}
                )
        self.active = None
        success = (
            confirmed
            and self.last_returncode == 0
            and closed.get("success") is True
            and closed.get("campaign_stop") is False
            and not self.blocked
        )
        self.blocked = not success
        record = {
            "model": model,
            "owned_exit_confirmed": confirmed,
            "returncode": getattr(self, "last_returncode", None),
            "worker": closed,
            "success": success,
            "gpu": self.gpu,
        }
        self.closures.append(record)
        self.record("arm-closures.json", self.closures)
        if success:
            self.next_arm += 1
        return record

    def close(self, nonce):
        self.check(nonce, cleanup=True)
        if self.process is not None:
            self.end(nonce)
        confirmed = self.process is None and all(
            row["owned_exit_confirmed"] for row in self.closures
        )
        self.blocked = self.blocked or not confirmed
        validate_inventory(encode(self.manifest))
        receipt = {
            "stage": "complete" if not self.blocked and self.next_arm == 6 else "stopped",
            "success": not self.blocked and self.next_arm == 6,
            "owned_exit_confirmed": confirmed,
            "gpu": self.gpu,
            "nonce": self.nonce,
            "identity_errors": self.identity_errors,
            "arms": self.closures,
            "unrun": list(MODELS[self.next_arm :]),
            "helper_calls": 0,
            "optimizer_updates": 0,
            "task_score": None,
        }
        self.record("campaign-receipt.json", receipt)
        if confirmed:
            self.record(
                "export-index.json",
                {
                    "root": self.output.name,
                    "files": self.staging.tree_inventory(self.output),
                    "complete_owned_exit": True,
                },
            )
        index = self.staging.publish_output(self.output, self.mounts["/export"], confirmed)
        return {**receipt, "root": self.output.name, "export_index": index}


class Gateway:
    """Local handler-to-durable-evidence timing; client RTT remains external."""

    def __init__(self, rpc, output, helper, *, clock=time.monotonic):
        self.rpc, self.output, self.helper, self.clock = (
            rpc,
            Path(output),
            copy.deepcopy(helper),
            clock,
        )
        self.lock, self.blocked, self.active, self.nonce = threading.Lock(), False, None, None
        self.counter, self.counter_lock = 0, threading.Lock()

    def request(self, body, *, started=None):
        started = self.clock() if started is None else started
        require(type(body) is bytes and len(body) <= MAX_BODY, "Bounded original request required")
        with self.counter_lock:
            if self.counter >= 6 * 192:
                self.blocked = True
                return 503, encode(
                    {"detail": "Bounded campaign request inventory exhausted; no action"}
                )
            self.counter += 1
        request_id = secrets.token_hex(16)
        row = {
            "request_id": request_id,
            "raw_http_hex": body.hex(),
            "http_sha256": hashlib.sha256(body).hexdigest(),
            "model": self.active,
            "helper": self.helper,
            "timing_scope": "local handler entry through observable queue, RPC, conversion, body encoding and durable evidence; excludes socket accept queue, response delivery, client RTT and upstream retries",
            "gateway_action_authorized": False,
        }
        status, response = 503, {"detail": "Decision service unavailable; no action"}
        acquired = self.lock.acquire(timeout=max(0, SECONDS - (self.clock() - started)))
        if not acquired:
            self.blocked = True
            row.update(
                error={
                    "type": "TimeoutError",
                    "message": "Observable local queue exceeded gateway20s",
                },
                gateway_elapsed_seconds=self.clock() - started,
            )
            save(self.output / "requests" / (request_id + ".json"), row)
            return 504, encode({"detail": "Local queue deadline exceeded; no action"})
        try:
            try:
                require(
                    not self.blocked and self.active is not None and self.nonce is not None,
                    "No ready active arm",
                )
                require(
                    type(body) is bytes and len(body) <= MAX_BODY,
                    "Bounded original request required",
                )
                payload = parse(body)
                require(
                    type(payload) is dict and payload.get("model") == self.active,
                    "Original model alias must match the active arm",
                )
                remaining = SECONDS - (self.clock() - started)
                require(remaining > 0, "Local decode/queue deadline exceeded")
                result = self.rpc(self.nonce, request_id, body, remaining)
                row["rpc"] = result
                require(
                    result["request_id"] == request_id
                    and result["http_sha256"] == row["http_sha256"]
                    and result["nonce"] == self.nonce
                    and result["model"] == self.active,
                    "Authenticated RPC response identity differs",
                )
                require(
                    result.get("success") is True
                    and result.get("campaign_stop") is False
                    and type(result.get("wire_body_utf8")) is str,
                    "Worker refused or returned late; no action",
                )
                candidate = result["wire_body_utf8"].encode()
                require(len(candidate) <= MAX_FRAME, "Bounded native wire response required")
                row["gateway_elapsed_seconds"] = self.clock() - started
                save(self.output / "requests" / (request_id + ".json"), row)
                require(
                    self.clock() - started < SECONDS,
                    "Full gateway20s exceeded; raw response retained, no action",
                )
                status, response = 200, candidate
                row["gateway_action_authorized"] = True
            except BaseException as exc:
                self.blocked = True
                row["error"] = {"type": type(exc).__name__, "message": str(exc)}
                status, response = (
                    504,
                    {"detail": "Decision request failed or exceeded gateway20s; no action"},
                )
                if not isinstance(exc, Exception):
                    save(self.output / "requests" / (request_id + ".json"), row)
                    raise
            row["gateway_elapsed_seconds"] = self.clock() - started
            save(self.output / "requests" / (request_id + ".json"), row)
            if status == 200 and self.clock() - started >= SECONDS:
                self.blocked = True
                row.update(gateway_action_authorized=False, late_after_final_evidence=True)
                save(self.output / "requests" / (request_id + ".json"), row)
                status, response = 504, {"detail": "Final evidence exceeded gateway20s; no action"}
            return status, response if type(response) is bytes else encode(response)
        finally:
            self.lock.release()


def http_server(gateway, token):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass  # Bearer headers and page content never enter access logs.

        def do_POST(self):
            started = gateway.clock()
            if self.path != "/v1/systemone":
                self.send_error(404)
                return
            if not secrets.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self.send_error(401)
                return
            body = b""
            try:
                length = int(self.headers.get("Content-Length", "-1"))
                require(
                    0 <= length <= MAX_BODY and not self.headers.get("Transfer-Encoding"),
                    "One bounded Content-Length request required",
                )
                while len(body) < length:
                    remaining = SECONDS - (gateway.clock() - started)
                    require(remaining > 0, "Local body decode deadline exceeded")
                    self.connection.settimeout(remaining)
                    chunk = self.rfile.read(min(65536, length - len(body)))
                    require(chunk, "Incomplete HTTP body")
                    body += chunk
                require(len(body) == length, "Incomplete HTTP body")
                status, response = gateway.request(body, started=started)
            except Exception as exc:
                gateway.blocked = True
                save(
                    gateway.output / "http-errors" / (secrets.token_hex(16) + ".json"),
                    {
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                        "raw_partial_hex": body.hex(),
                        "gateway_elapsed_seconds": gateway.clock() - started,
                        "gateway_action_authorized": False,
                    },
                )
                status, response = 400, encode({"detail": "Malformed request; no action"})
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            try:
                self.wfile.write(response)
            except OSError:
                gateway.blocked = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = False
    return server


def prepare(output):
    output = ordinary(output)
    require(not output.exists(), "Fresh source-only preparation directory required")
    inventory = source_inventory()
    output.mkdir(parents=True)
    save(output / "source-inventory.json", inventory)
    save(
        output / "plan.json",
        {
            "stage": "prospective_not_executed",
            "source_files": 17,
            "models": list(MODELS),
            "weight_profile_schema": WEIGHT_SCHEMA,
            "weight_profile_required_external_sha": True,
            "helper": "Separate fixed qualified loopback endpoint supplied by the caller; never loaded in the D3 image",
            "control_commands": ["begin", "end", "close"],
            "policy_route": "/v1/systemone",
            "public_cloud_endpoint": False,
            "automatic_benchmark_runner": False,
        },
    )


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--inventory-only":
        target = ordinary(sys.argv[2])
        require(not target.exists(), "Fresh source inventory output required")
        with target.open("xb") as stream:
            stream.write(encode(source_inventory()))
        raise SystemExit(0)
    if sys.argv[1:2] == ["prepare"]:
        parser = argparse.ArgumentParser()
        parser.add_argument("prepare")
        parser.add_argument("--output", required=True)
        prepare(parser.parse_args().output)
        raise SystemExit(0)

import modal  # noqa: E402 -- source-only preparation exits before SDK import.

from apps.modal import decision3_public_replay as reviewed  # noqa: E402

app = modal.App("decision3-public-browser-rpc", include_source=False)
image = reviewed.image.env({"PYTHONPATH": str(SCRIPT.parent) + os.pathsep + str(ROOT)})
for row in source_inventory()["rows"]:
    image = image.add_local_file(row["path"], row["path"], copy=True)


@app.cls(
    image=image,
    gpu="H200",
    cpu=2,
    memory=344064,
    ephemeral_disk=524288,
    timeout=25200,
    min_containers=1,
    max_containers=1,
    scaledown_window=1200,
    retries=0,
    include_source=False,
    block_network=True,
    volumes={
        "/weights": reviewed.weights.with_mount_options(read_only=True),
        "/export": reviewed.results,
    },
)
class PersistentWorker:
    @modal.enter()
    def enter(self):
        self.session = None
        self.initialization_failure = None
        self.lock = threading.RLock()

    @modal.method()
    def begin(self, run_id, model, manifest_blob, profile_blob, profile_sha, nonce=None):
        with self.lock:
            require(
                self.initialization_failure is None,
                "Failed allocation initialization cannot resume",
            )
            if self.session is None:
                require(
                    nonce is None and model == MODELS[0],
                    "Restarted allocation cannot continue a campaign",
                )
                try:
                    self.session = Session(run_id, manifest_blob, profile_blob, profile_sha)
                except BaseException as primary:
                    self.initialization_failure = getattr(
                        primary,
                        "decision3_initialization_failure",
                        {
                            "stage": "initialization_refused",
                            "success": False,
                            "owned_exit_confirmed": True,
                            "model_loads": 0,
                            "arms": [],
                            "root": None,
                            "export_index": None,
                            "error": {"type": type(primary).__name__, "message": str(primary)},
                        },
                    )
                    try:
                        reviewed.results.commit()
                    except BaseException as secondary:
                        primary.add_note(
                            "Initialization failure volume commit also failed: " + repr(secondary)
                        )
                    raise
            return self.session.begin(model, nonce)

    @modal.method()
    def predict(self, nonce, request_id, body, remaining):
        with self.lock:
            require(self.session is not None, "Original persistent allocation is unavailable")
            return self.session.predict(nonce, request_id, body, remaining)

    @modal.method()
    def end(self, nonce):
        with self.lock:
            require(self.session is not None, "Original persistent allocation is unavailable")
            return self.session.end(nonce)

    @modal.method()
    def close(self, nonce=None):
        with self.lock:
            if self.initialization_failure is not None:
                require(nonce is None, "Uninitialized allocation has no observed nonce")
                reviewed.results.commit()
                return self.initialization_failure
            require(
                self.session is not None,
                "Original persistent allocation is unavailable; closure unknown",
            )
            # An authenticated owner can close a lost first begin response;
            # this fallback can only stop work, never predict/start another arm.
            result = self.session.close(self.session.nonce if nonce is None else nonce)
            reviewed.results.commit()
            return result


@app.local_entrypoint()
def main(
    output: str,
    manifest: str,
    weight_profile: str,
    weight_profile_sha256: str,
    helper_endpoint: str,
    helper_qualification_sha256: str,
    token_file: str,
    run_id: str,
    run: bool = False,
):
    require(run, "Explicit --run after source/inventory/weight-profile review required")
    destination = ordinary(output)
    require(not destination.exists(), "Fresh local campaign output required")
    manifest_blob = ordinary(manifest).read_bytes()
    validate_inventory(manifest_blob)
    profile_blob = ordinary(weight_profile).read_bytes()
    require(
        hashlib.sha256(profile_blob).hexdigest() == weight_profile_sha256,
        "Externally selected actual complete weight profile required",
    )
    helper_url = urlsplit(helper_endpoint)
    require(
        helper_url.scheme == "http"
        and helper_url.hostname in {"127.0.0.1", "localhost"}
        and helper_url.port
        and not helper_url.username
        and not helper_url.password
        and re.fullmatch(r"[0-9a-f]{64}", helper_qualification_sha256),
        "Fixed separately qualified loopback helper required",
    )
    private = ordinary(token_file)
    require(
        not private.exists() and not private.is_relative_to(destination),
        "Fresh private token file outside exported logs required",
    )
    token = secrets.token_urlsafe(32)
    descriptor = os.open(private, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(token)
    destination.mkdir(parents=True)
    helper = {"endpoint": helper_endpoint, "qualification_sha256": helper_qualification_sha256}
    remote = PersistentWorker()
    gateway = Gateway(lambda *args: remote.predict.remote(*args), destination, helper)
    server = http_server(gateway, token)
    sequence, closed, begin_attempted = 1, False, False
    primary, thread, thread_started = None, None, False
    try:
        thread = threading.Thread(target=server.serve_forever, daemon=False)
        thread.start()
        thread_started = True
        save(
            destination / "endpoint.json",
            {
                "policy": f"http://127.0.0.1:{server.server_port}/v1/systemone",
                "token_file": str(private),
                "helper": helper,
                "control": "Write control-0001.json, control-0002.json... with action begin/model, end, or close. No policy request changes an arm.",
            },
        )
        while not closed:
            command = destination / f"control-{sequence:04d}.json"
            if not command.exists():
                time.sleep(0.1)
                continue
            payload = parse(command.read_bytes())
            with gateway.lock:
                action = payload["action"]
                if action == "begin":
                    require(
                        not gateway.blocked and gateway.active is None,
                        "Previous arm must close successfully",
                    )
                    begin_attempted = True
                    reply = remote.begin.remote(
                        run_id,
                        payload["model"],
                        manifest_blob,
                        profile_blob,
                        weight_profile_sha256,
                        gateway.nonce,
                    )
                    gateway.nonce, gateway.active = reply["nonce"], reply["model"]
                elif action == "end":
                    reply = remote.end.remote(gateway.nonce)
                    gateway.active = None
                    gateway.blocked = gateway.blocked or not reply["success"]
                elif action == "close":
                    gateway.active = None
                    reply = remote.close.remote(gateway.nonce)
                    closed = True
                    save(destination / f"control-{sequence:04d}-receipt.json", reply)
                    require(
                        reply["owned_exit_confirmed"] and reply["export_index"] is not None,
                        "Unknown worker exit: partial provider evidence retained; campaign halted",
                    )
                    reviewed.export(reply, destination / "cloud-export")
                else:
                    raise ValueError("Unknown explicit control command")
                save(destination / f"control-{sequence:04d}-receipt.json", reply)
                sequence += 1
    except BaseException as exc:
        primary = exc
        gateway.blocked = True
        save(
            destination / "controller-error.json",
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "notes": list(getattr(exc, "__notes__", [])),
                "campaign_stop": True,
            },
        )
        raise
    finally:
        if thread_started:
            server.shutdown()
        server.server_close()
        if thread_started:
            thread.join()
        if not closed and begin_attempted:
            try:
                # Authenticated stop-only recovery also covers a lost first
                # begin response, before its allocation nonce reached localhost.
                save(destination / "shutdown-receipt.json", remote.close.remote(gateway.nonce))
            except BaseException as secondary:
                save(
                    destination / "shutdown-error.json",
                    {
                        "type": type(secondary).__name__,
                        "message": str(secondary),
                        "campaign_stop": True,
                        "owned_exit_confirmed": False,
                    },
                )
                if primary is None:
                    raise
                primary.add_note("Authenticated stop-only cleanup also failed: " + repr(secondary))
