"""Fresh Q95 identity/credential/host and unchanged D3 co-resident admission.

Stdlib only. This module starts no process, listener, unit, framework or model.
The actual helper launch/source-root assembly is a separate owner obligation.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import http.client
import json
import os
import re
import secrets
import stat
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).absolute().parents[3]
PROC, CGROOT = Path("/proc"), Path("/sys/fs/cgroup")
SCHEMA = "decision3-q95-live-helper-selection-v1"
PROFILE = "field-syntax-preserving-base-helper-v1"
Q_SHA = "d33bfcac49f90c47b626c3915baf3d5fcbfb2e32522bfaccbbfec59dce35d8db"
E_SHA = "95f2eab0be9646da18694b18c872e044db09ba2fb1c09708077b01b24d58ae44"
R_SHA = "702f7ff3cf194d1db56f86474e48ffc88cdfa8c9c6d3930467e3c12ea6492016"
REG_SHA = "aaa21ab6f2ecf5ae0e00694b43755541dc75fc0310ba10d945ade2a4511add52"
BASE_BYTES, GIB = 23919549408, 1024**3
MODELS = ("d3", "d3-flash", "d3-mini", "d3-nano", "d3-lite", "d3-edge")
CALL_SECONDS, MAX_CHECKS = 3.0, 512


class BindingError(RuntimeError):
    """Stable safe error codes only; never private config/header/body contents."""


def need(value, code):
    if not value:
        raise BindingError(code)


def exact_fields(value, names):
    need(type(value) is dict and set(value) == set(names.split()), "EXACT_SCHEMA_REQUIRED")


def parse(blob):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "DUPLICATE_JSON_KEY")
            result[key] = value
        return result

    return json.loads(
        blob, object_pairs_hook=pairs, parse_constant=lambda _value: need(False, "NONFINITE_JSON")
    )


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def ordinary(path):
    path = Path(path).absolute()
    need(
        ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "ORDINARY_PATH_REQUIRED",
    )
    return path


def read(path, limit=1024**2, *, private=False):
    path = ordinary(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= limit, "BOUNDED_REGULAR_FILE")
        if private:
            need(
                before.st_uid == os.getuid()
                and stat.S_IMODE(before.st_mode) == 0o600
                and before.st_nlink == 1,
                "PRIVATE_CONFIG_OWNER_MODE",
            )
        blob = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    need(
        len(blob) <= limit
        and tuple(getattr(before, n) for n in fields)
        == tuple(getattr(after, n) for n in fields)
        == tuple(getattr(path.stat(), n) for n in fields),
        "FILE_CHANGED_DURING_READ",
    )
    return blob


def pin(path):
    # Only selected small source/runtime files, never model payloads.
    blob = read(path)
    return {"sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)}


def ref_blob(ref, expected_sha):
    exact_fields(ref, "path sha256 size_bytes")
    need(
        type(ref["path"]) is str
        and not Path(ref["path"]).is_absolute()
        and ".." not in Path(ref["path"]).parts,
        "REPOSITORY_RELATIVE_REF",
    )
    need(
        ref["sha256"] == expected_sha and type(ref["size_bytes"]) is int,
        "EXACT_SELECTED_METADATA_IDENTITY",
    )
    blob = read(ROOT / ref["path"])
    need(
        {"sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)}
        == {k: ref[k] for k in ("sha256", "size_bytes")},
        "SELECTED_METADATA_DRIFT",
    )
    return blob


def boot():
    value = (PROC / "sys/kernel/random/boot_id").read_text().strip()
    need(re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", value), "BOOT_IDENTITY")
    return value


def process(pid):
    need(type(pid) is int and pid > 0, "POSITIVE_PROCESS_ID")
    path = PROC / str(pid)

    def identity():
        raw = (path / "stat").read_text()
        fields = raw[raw.rfind(")") + 2 :].split()
        need(len(fields) >= 20 and fields[0] not in {"Z", "X"}, "LIVE_PROCESS_REQUIRED")
        return {
            "pid": pid,
            "pgid": int(fields[2]),
            "sid": int(fields[3]),
            "start_ticks": int(fields[19]),
        }

    first = identity()
    raw = (path / "cmdline").read_bytes()
    need(len(raw) <= 65536, "BOUNDED_PROCESS_COMMAND")
    argv = [part.decode() for part in raw.split(b"\0") if part]
    group = (path / "cgroup").read_text().splitlines()
    need(len(group) == 1 and group[0].startswith("0::/"), "UNIFIED_PROCESS_CGROUP")
    row = {
        **first,
        "boot_id": boot(),
        "exe": str((path / "exe").resolve(strict=True)),
        "argv": argv,
        "cgroup": group[0][3:],
        "uid": path.stat().st_uid,
    }
    need(first == identity() and row["uid"] == os.getuid(), "PROCESS_IDENTITY_SANDWICH")
    return row


def listener(pid, port):
    need(type(port) is int and 0 < port < 65536, "LOOPBACK_PORT")
    path = PROC / str(pid)
    descriptors = list((path / "fd").iterdir())
    need(len(descriptors) <= 4096, "BOUNDED_FD_INVENTORY")
    inodes = set()
    for fd in descriptors:
        try:
            target = os.readlink(fd)
        except FileNotFoundError:
            continue
        match = re.fullmatch(r"socket:\[([0-9]+)\]", target)
        if match:
            inodes.add(match[1])
    matches = []
    for family in ("tcp", "tcp6"):
        lines = (path / "net" / family).read_text().splitlines()
        need(len(lines) <= 8193, "BOUNDED_LISTENER_INVENTORY")
        for line in lines[1:]:
            fields = line.split()
            need(len(fields) > 9, "COMPLETE_TCP_ROW")
            address, _, selected_port = fields[1].rpartition(":")
            if int(selected_port, 16) == port and fields[3] == "0A":
                need(
                    family == "tcp" and address == "0100007F" and fields[9] in inodes,
                    "EXACT_OWNED_IPV4_LOOPBACK_LISTENER",
                )
                matches.append(fields[9])
    need(len(matches) == 1, "UNIQUE_OWNED_LISTENER")
    return {"pid": pid, "address": "127.0.0.1", "port": port, "inode": matches[0]}


def host_memory():
    values = {}
    for line in (PROC / "meminfo").read_text().splitlines():
        key, _, raw = line.partition(":")
        if key in {"MemTotal", "MemAvailable"}:
            parts = raw.split()
            need(
                key not in values and len(parts) == 2 and parts[1] == "kB" and parts[0].isdecimal(),
                "TYPED_HOST_MEMORY",
            )
            values[key] = int(parts[0]) * 1024
    need(
        set(values) == {"MemTotal", "MemAvailable"}
        and 0 < values["MemAvailable"] <= values["MemTotal"],
        "POSITIVE_HOST_MEMORY",
    )
    return {"boot_id": boot(), **values}


def cgroup_memory(group):
    need(
        type(group) is str and group.startswith("/") and ".." not in Path(group).parts,
        "EXACT_UNIFIED_CGROUP",
    )
    current = CGROOT / group.lstrip("/")
    need(current.is_dir(), "SELECTED_CGROUP_EXISTS")
    rows = []
    while True:
        need(len(rows) < 64 and current.is_relative_to(CGROOT), "BOUNDED_CGROUP_ANCESTORS")
        if current == CGROOT and not (current / "memory.max").exists():
            break  # cgroup v2 root has no workload memory.max controller file.
        maximum = (current / "memory.max").read_text().strip()
        used = (current / "memory.current").read_text().strip()
        need(
            (maximum == "max" or maximum.isdecimal() and int(maximum) > 0) and used.isdecimal(),
            "TYPED_CGROUP_MEMORY",
        )
        events = {}
        for line in (current / "memory.events").read_text().splitlines():
            key, value = line.split()
            need(key not in events and value.isdecimal(), "TYPED_CGROUP_EVENTS")
            events[key] = int(value)
        need(bool(events), "KNOWN_CGROUP_EVENTS")
        rows.append(
            {
                "path": str(current),
                "limit": None if maximum == "max" else int(maximum),
                "current": int(used),
                "events": events,
            }
        )
        if current == CGROOT:
            break
        current = current.parent
    need(bool(rows), "COMPLETE_SELECTED_CGROUP_MEMORY")
    return rows


def memory_decisions(host, helper_groups, co_groups, resident):
    reserve = max(24 * GIB, (host["MemTotal"] + 4) // 5)
    helper_required = reserve + 2 * BASE_BYTES
    co_required = reserve + 2 * resident + 2 * GIB
    available = host["MemAvailable"]
    for row in co_groups:
        if row["limit"] is not None:
            available = min(available, max(0, row["limit"] - row["current"]))
    return {
        "host": host,
        "reserve_bytes": reserve,
        "helper": {
            "admitted": host["MemAvailable"] >= helper_required,
            "required_host_available_bytes": helper_required,
            "cgroups_recorded_only": helper_groups,
            "basis": "Original Q95 host-only R+2BASE; no invented memcg reserve floor",
        },
        "co_resident": {
            "admitted": available >= co_required,
            "available_bytes": available,
            "required_available_bytes": co_required,
            "resident_file_bytes": resident,
            "cgroups": co_groups,
            "basis": "Unchanged CORE min(host,cgroup headroom)>=R+2D+2GiB",
        },
        "swap_or_resident_credit_bytes": 0,
        "cuda_observed": False,
    }


class LiveHelperChecker:
    """Externally selected descriptor; fresh facts, never a trusted admitted flag."""

    def __init__(self, *, selection_blob, selection_sha, receipt_directory):
        need(
            type(selection_blob) is bytes
            and len(selection_blob) <= 1024**2
            and type(selection_sha) is str
            and re.fullmatch(r"[0-9a-f]{64}", selection_sha)
            and hashlib.sha256(selection_blob).hexdigest() == selection_sha,
            "EXTERNALLY_SELECTED_DESCRIPTOR",
        )
        value = parse(selection_blob)
        exact_fields(
            value,
            "schema_id binding worker listener service_directory private_config interpreter source_root qualification execution runtime registry expected_memtotal co_resident",
        )
        need(value["schema_id"] == SCHEMA, "EXPLICIT_Q95_SELECTION")
        binding = value["binding"]
        exact_fields(
            binding,
            "profile qualification_sha256 source_count boot_id pid start_ticks pgid sid endpoint seconds tokens context",
        )
        need(
            all(
                binding[k] == v and type(binding[k]) is type(v)
                for k, v in {
                    "profile": PROFILE,
                    "qualification_sha256": Q_SHA,
                    "source_count": 95,
                    "seconds": 20,
                    "tokens": 128,
                    "context": 8192,
                }.items()
            ),
            "EXACT_Q95_BINDING",
        )
        need(
            all(
                type(binding[k]) is int and binding[k] > 0
                for k in ("pid", "start_ticks", "pgid", "sid")
            )
            and binding["pid"] == binding["pgid"] == binding["sid"],
            "OWNED_HELPER_SESSION",
        )
        endpoint = urlsplit(binding["endpoint"])
        need(
            endpoint.scheme == "http"
            and endpoint.hostname == "127.0.0.1"
            and endpoint.port
            and endpoint.path == ""
            and not endpoint.query
            and not endpoint.fragment
            and endpoint.username is None,
            "EXACT_PUBLIC_ENDPOINT",
        )
        exact_fields(value["listener"], "pid address port inode")
        need(
            value["listener"]["pid"] == binding["pid"]
            and value["listener"]["address"] == "127.0.0.1"
            and type(value["listener"]["port"]) is int
            and value["listener"]["port"] == endpoint.port
            and type(value["listener"]["inode"]) is str
            and value["listener"]["inode"].isdecimal(),
            "SELECTED_LISTENER_SCHEMA",
        )
        exact_fields(value["worker"], "pid start_ticks pgid sid boot_id exe argv cgroup")
        need(
            all(
                value["worker"][k] == binding[k]
                for k in ("pid", "start_ticks", "pgid", "sid", "boot_id")
            ),
            "WORKER_BINDING_DIFFERS",
        )
        service = ordinary(value["service_directory"])
        source = ordinary(value["source_root"])
        need(
            service.is_relative_to(ROOT / "artifacts/jev-runtime") and source.is_relative_to(ROOT),
            "CONFINED_SELECTED_SERVICE_AND_SOURCE",
        )
        need(
            ordinary(value["private_config"]) == service / "controller-config.json",
            "EXACT_PRIVATE_CONFIG_PATH",
        )
        need(
            value["worker"]["argv"]
            == [
                value["interpreter"],
                str(source / "scripts/jev_qualify_public_field_syntax_bf16.py"),
                "--internal-child",
                value["private_config"],
            ]
            and value["worker"]["exe"] == str(Path(value["interpreter"]).resolve(strict=True)),
            "EXACT_SELECTED_HELPER_COMMAND",
        )
        need(
            type(value["expected_memtotal"]) is int and value["expected_memtotal"] > 0,
            "EXPECTED_POSITIVE_MEMTOTAL",
        )
        exact_fields(value["co_resident"], "model resident_file_bytes cgroup")
        need(value["co_resident"]["model"] in MODELS, "ALL_SIX_MODEL_POPULATION")
        self._value, self._sha = value, selection_sha
        self._directory = ordinary(receipt_directory)
        need(
            self._directory.is_dir() and not list(self._directory.iterdir()),
            "FRESH_RECEIPT_DIRECTORY",
        )
        self._sequence, self._lock = 0, threading.Lock()
        self._qualification = self._static()

    @property
    def binding(self):
        return copy.deepcopy(self._value["binding"])

    @property
    def model(self):
        return self._value["co_resident"]["model"]

    @property
    def resident_bytes(self):
        return self._value["co_resident"]["resident_file_bytes"]

    @property
    def selection_sha(self):
        return self._sha

    def _static(self):
        value = self._value
        qualification = parse(ref_blob(value["qualification"], Q_SHA))
        need(
            qualification["qualified"] is True
            and qualification["task_performance_attested"] is False
            and qualification["kind"] == "jev_public_action_field_syntax_bf16_qualification_v1"
            and qualification["execution_sha256"] == E_SHA
            and qualification["runtime_sha256"] == R_SHA,
            "EXACT_SAVED_Q95_METADATA",
        )
        execution = parse(ref_blob(value["execution"], E_SHA))
        need(len(execution["files"]) == 95, "EXACT_SOURCE95_COUNT")
        for label, expected in execution["files"].items():
            need(
                not Path(label).is_absolute() and ".." not in Path(label).parts,
                "RELATIVE_SOURCE95_PATH",
            )
            need(pin(Path(value["source_root"]) / label) == expected, "FROZEN_SOURCE95_DRIFT")
        runtime = parse(ref_blob(value["runtime"], R_SHA))
        need(len(runtime["files"]) == 348, "EXACT_SELECTED_RUNTIME348")
        for path, expected in runtime["files"].items():
            need(Path(path).is_absolute() and pin(path) == expected, "SELECTED_RUNTIME_DRIFT")
        registry = ast.parse(ref_blob(value["registry"], REG_SHA))
        candidates = [
            ast.literal_eval(n.value)
            for n in registry.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "CHECKPOINTS" for t in n.targets)
        ]
        need(len(candidates) == 1 and tuple(candidates[0]) == MODELS, "EXACT_SIX_MODEL_REGISTRY")
        need(
            type(self.resident_bytes) is int
            and self.resident_bytes > 0
            and self.resident_bytes == candidates[0][self.model]["resident_file_bytes"],
            "UNCHANGED_SELECTED_D3_RESIDENT_FLOOR",
        )
        return qualification

    def _live(self):
        value, binding = self._value, self.binding
        need(boot() == binding["boot_id"], "FRESH_SELECTED_BOOT")
        actual = process(binding["pid"])
        need(all(actual[k] == v for k, v in value["worker"].items()), "LIVE_WORKER_IDENTITY_DRIFT")
        need(
            listener(binding["pid"], urlsplit(binding["endpoint"]).port) == value["listener"],
            "LIVE_LISTENER_IDENTITY_DRIFT",
        )
        service = Path(value["service_directory"])
        need(
            not any(
                (service / name).exists()
                for name in (
                    "shutdown.request",
                    "owned-exit.json",
                    "controller-receipt.json",
                    "controller-failure.json",
                )
            ),
            "HELPER_STOPPING_OR_CLOSED",
        )
        config_blob = read(value["private_config"], private=True)
        config = parse(config_blob)
        expected = {
            "command": "serve",
            "output": str(service),
            "boot_id": binding["boot_id"],
            "mem_total": value["expected_memtotal"],
            "qualification_sha256": Q_SHA,
            "execution_sha256": E_SHA,
            "runtime_sha256": R_SHA,
            "plan_sha256": self._qualification["plan_sha256"],
            "inputs_sha256": self._qualification["inputs_sha256"],
            "public": self._qualification["public"],
            "qualification_path": str((ROOT / value["qualification"]["path"]).parent),
            "execution_path": str(ROOT / value["execution"]["path"]),
            "runtime_path": str(ROOT / value["runtime"]["path"]),
        }
        need(all(config.get(k) == v for k, v in expected.items()), "PRIVATE_PUBLIC_SELECTION_DRIFT")
        token = config.get("token")
        need(type(token) is str and re.fullmatch(r"[0-9a-f]{64}", token), "PRIVATE_TOKEN_SHAPE")
        # The old shared Q93 policy+text env-key equality is inapplicable here:
        # D3 policy and Q95 text credentials are deliberately different. This
        # independently selected owner-only private file is the authority; live
        # authenticated HTTP below proves its token rather than trusting a flag.
        need(
            token.encode() not in encode(value) and token not in "\0".join(actual["argv"]),
            "PRIVATE_CREDENTIAL_IN_PUBLIC_SELECTION",
        )
        endpoint = parse(read(service / "endpoint.json"))
        need(
            endpoint
            == {
                "policy": binding["endpoint"] + "/v1/systemone",
                "helper": binding["endpoint"],
                "model": "generated-actions-none",
                "qualified": True,
                "TEXT_MODEL_REASONING": "none",
            },
            "LIVE_ENDPOINT_PROFILE",
        )
        return actual, config_blob, token

    def _memory(self):
        host = host_memory()
        need(
            host["boot_id"] == self.binding["boot_id"]
            and host["MemTotal"] == self._value["expected_memtotal"],
            "FRESH_HOST_IDENTITY",
        )
        need(
            process(os.getpid())["cgroup"] == self._value["co_resident"]["cgroup"],
            "ACTUAL_CO_RESIDENT_OWNER_CGROUP",
        )
        result = memory_decisions(
            host,
            cgroup_memory(self._value["worker"]["cgroup"]),
            cgroup_memory(self._value["co_resident"]["cgroup"]),
            self.resident_bytes,
        )
        need(
            result["helper"]["admitted"] and result["co_resident"]["admitted"],
            "FRESH_UNCHANGED_MEMORY_ADMISSION_REFUSED",
        )
        return result

    def _auth(self, token, deadline):
        endpoint = urlsplit(self.binding["endpoint"])
        rows = []
        request_root = ordinary(Path(self._value["service_directory"]) / "requests")

        def request_ids():
            if not request_root.exists():
                return set()
            paths = list(request_root.iterdir())
            need(
                len(paths) <= 5000
                and all(
                    p.is_dir() and not p.is_symlink() and re.fullmatch(r"[0-9a-f]{32}", p.name)
                    for p in paths
                ),
                "BOUNDED_Q95_REQUEST_INVENTORY",
            )
            return {p.name for p in paths}

        initial_requests = request_ids()
        for selected in (False, True):
            remaining = deadline - time.monotonic()
            need(remaining > 0, "CHECKER_DEADLINE")
            connection = http.client.HTTPConnection("127.0.0.1", endpoint.port, timeout=remaining)
            try:
                key = token if selected else secrets.token_hex(32)
                connection.request(
                    "POST",
                    "/chat/completions",
                    body=b"{}",
                    headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
                )
                reply = connection.getresponse()
                body = reply.read(65537)
                need(
                    len(body) <= 65536 and token.encode() not in body and key.encode() not in body,
                    "BOUNDED_REDACTED_AUTH_RESPONSE",
                )
                need(reply.status == (422 if selected else 500), "AUTHENTICATED_SCHEMA_STATUS")
                request_id = reply.getheader("X-Public-Request-ID")
                if selected:
                    need(
                        parse(body).get("error", {}).get("type") == "helper_schema",
                        "NO_INFERENCE_SCHEMA_REFUSAL_REQUIRED",
                    )
                    need(
                        type(request_id) is str
                        and re.fullmatch(r"[0-9a-f]{32}", request_id)
                        and request_id not in initial_requests,
                        "FRESH_AUTHENTICATED_HANDLER_REQUEST_ID",
                    )
                    raw_path = request_root / request_id / "raw.json"
                    raw_blob = read(raw_path)
                    raw = parse(raw_blob)
                    need(
                        all(
                            raw.get(k) == v
                            for k, v in {
                                "schema_id": "jev-public-field-syntax-raw-candidate-v1",
                                "request_id": request_id,
                                "api_request_id": None,
                                "payload_sha256": hashlib.sha256(b"{}").hexdigest(),
                                "prefix": None,
                                "prepared_prompt": None,
                                "helper_phases": [],
                                "generation": {},
                                "forward_events": [],
                                "guard_restored": True,
                                "server_terminal": True,
                                "error": None,
                            }.items()
                        )
                        and raw.get("candidate_response", {}).get("error", {}).get("type")
                        == "helper_schema",
                        "EXACT_AUTHENTICATED_ZERO_FORWARD_HANDLER_EVIDENCE",
                    )
                    handler_evidence = {
                        "path": str(raw_path),
                        "sha256": hashlib.sha256(raw_blob).hexdigest(),
                        "size_bytes": len(raw_blob),
                    }
                else:
                    # The pinned qualifier rejects auth before allocating its
                    # request identity. Generic500 alone is never authentication.
                    need(
                        body == b"Internal Server Error"
                        and request_id is None
                        and request_ids() == initial_requests,
                        "UNAUTHORIZED_PRE_HANDLER_REJECTION_EVIDENCE",
                    )
                    handler_evidence = None
                rows.append(
                    {
                        "selected_credential": selected,
                        "status": reply.status,
                        "body_sha256": hashlib.sha256(body).hexdigest(),
                        "body_bytes": len(body),
                        "handler_evidence": handler_evidence,
                    }
                )
            finally:
                connection.close()
        return {
            "requests": rows,
            "inference_requested": False,
            "scope": "Paired source-bound credential test: generic500 with no handler ID/trace is only negative context; selected-key422 joins fresh persisted helper_schema with zero forward/prefix/generation. No health-only or arbitrary500 authentication claim.",
        }

    def __call__(self, binding, phase):
        need(
            binding == self.binding
            and type(phase) is str
            and re.fullmatch(r"[a-z0-9][a-z0-9:_-]{0,95}", phase),
            "CALLBACK_BINDING_AND_PHASE",
        )
        need(self._lock.acquire(timeout=CALL_SECONDS), "CHECKER_SERIAL_DEADLINE")
        if self._sequence >= MAX_CHECKS:
            self._lock.release()
            raise BindingError("BOUNDED_CHECKER_RECEIPTS")
        started, receipt = (
            time.monotonic(),
            {
                "binding": self.binding,
                "phase": phase,
                "selection_sha256": self._sha,
                "helper_admitted": False,
                "co_resident_admitted": False,
                "qualified_model": False,
                "whole_tree_exit_confirmed": False,
                "continuation_authorized": False,
                "error_type": None,
                "error_code": None,
            },
        )
        primary = None
        try:
            self._static()
            before, config_blob, token = self._live()
            receipt["memory_before"] = self._memory()
            receipt["authentication"] = self._auth(token, started + CALL_SECONDS)
            after, final_config_blob, _ = self._live()
            need(
                before == after and config_blob == final_config_blob,
                "LIVE_BINDING_CHANGED_DURING_CHECK",
            )
            receipt["memory_after"] = self._memory()
            need(time.monotonic() - started <= CALL_SECONDS, "CHECKER_DEADLINE")
            receipt.update(
                helper_admitted=True,
                co_resident_admitted=True,
                listener=self._value["listener"],
                worker={k: v for k, v in before.items() if k != "argv"},
                argv_sha256=hashlib.sha256(encode(before["argv"])).hexdigest(),
            )
        except BaseException as error:
            primary = error
            receipt["error_type"] = type(error).__name__
            receipt["error_code"] = (
                str(error) if isinstance(error, BindingError) else "OBSERVATION_FAILED"
            )
        finally:
            try:
                receipt["elapsed_seconds"] = time.monotonic() - started
                blob = encode(receipt)
                need(len(blob) <= 1024**2, "BOUNDED_SANITIZED_RECEIPT")
                path = ordinary(self._directory / f"helper-check-{self._sequence:04d}.json")
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(blob)
                    stream.flush()
                    os.fsync(stream.fileno())
                self._sequence += 1
            except BaseException as secondary:
                if primary is None:
                    primary = secondary
                else:
                    primary.add_note("Sanitized helper receipt persistence also failed")
            finally:
                self._lock.release()
        if primary is not None:
            raise BindingError(
                "LIVE_HELPER_REFUSED:" + receipt["error_code"]
                if receipt["error_code"]
                else "LIVE_HELPER_RECEIPT_FAILED"
            ) from None
        return {
            "binding": self.binding,
            "phase": phase,
            "helper_admitted": True,
            "co_resident_admitted": True,
            "receipt_sha256": hashlib.sha256(blob).hexdigest(),
        }

    def text_connection(self):
        """In-memory parent browser environment only. NEVER serialize this return."""
        self(self.binding, "browser-text-connection")
        _, _, token = self._live()
        return {
            "endpoint": self.binding["endpoint"],
            "model": "generated-actions-none",
            "api_key": token,
        }
