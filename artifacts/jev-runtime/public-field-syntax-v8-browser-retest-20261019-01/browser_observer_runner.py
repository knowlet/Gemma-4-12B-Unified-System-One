"""Artifact-only exact HTTP observer around explicitly isolated v8 Jev runners.

No model is loaded here. The prepare command does not execute a browser or HTTP.
The explicit child command runs an owned, externally bound benchmark process.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import re
import runpy
import stat
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
PLAN_SHA = "f909d8174e349b4e7facf3ef7866f05d37ed3f1f17eafe26a04c299fe089794c"
Q_SHA = "d33bfcac49f90c47b626c3915baf3d5fcbfb2e32522bfaccbbfec59dce35d8db"
SOURCE_SHA = "cbfb7bdd8301174a97ac18389a998ff3256ac32d6f7a770978f2f9b722c3b43b"
SENSITIVE_HEADERS = frozenset(
    {
        b"authorization",
        b"proxy-authorization",
        b"cookie",
        b"set-cookie",
        b"api-key",
        b"x-api-key",
        b"apikey",
        b"api_key",
        b"access-token",
        b"access_token",
        b"x-auth-token",
        b"password",
    }
)


class EvidenceError(RuntimeError):
    pass


def canonical(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def pin(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise EvidenceError("ordinary pinned file required")
    blob = path.read_bytes()
    return {"sha256": sha(blob), "size_bytes": len(blob)}


def read_json(path):
    return json.loads(Path(path).read_bytes())


def write(path, blob):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("xb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o600)


def endpoints(policy, helper):
    parsed = urlsplit(helper)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.port is None
        or not 0 < parsed.port < 65536
        or parsed.path
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or helper != f"http://127.0.0.1:{parsed.port}"
    ):
        raise EvidenceError("exact explicit owned IPv4 loopback base required")
    if policy != helper + "/v1/systemone":
        raise EvidenceError("policy and helper must share the exact selected service")
    return frozenset((policy, helper + "/chat/completions")), parsed.port


@dataclass(frozen=True)
class Row:
    sequence: int
    endpoint: str
    started_monotonic: float
    elapsed_ms: float
    request_body: bytes | None
    request_body_sha256: str | None
    request_body_size: int | None
    request_headers: tuple[tuple[bytes, bytes], ...]
    response_body: bytes | None
    response_body_sha256: str | None
    response_body_size: int | None
    response_headers: tuple[tuple[bytes, bytes], ...]
    http_status: int | None
    request_id: str | None
    transport_error: tuple[str, str] | None
    capture_errors: tuple[str, ...]


class TransportObserver:
    def __init__(
        self,
        policy_endpoint,
        helper_base,
        *,
        secrets=(),
        max_rows=4096,
        max_body_bytes=16 * 1024**2,
        max_total_bytes=256 * 1024**2,
    ):
        self.selected, self.port = endpoints(policy_endpoint, helper_base)
        self.secrets = tuple(x.encode() if isinstance(x, str) else bytes(x) for x in secrets if x)
        if any(type(x) is not int or x <= 0 for x in (max_rows, max_body_bytes, max_total_bytes)):
            raise EvidenceError("positive observer storage bounds required")
        self.max_rows, self.max_body_bytes, self.max_total_bytes = (
            max_rows,
            max_body_bytes,
            max_total_bytes,
        )
        self.lock = threading.RLock()
        self._rows = {}
        self._pending = set()
        self.selected_attempt_total = self.omitted_count = self.total_bytes = 0
        self.first_error = None

    def _message(self, value):
        text = str(value)
        for secret in self.secrets:
            text = text.replace(secret.decode(errors="replace"), "[REDACTED]")
        return text[:1024]

    def _latch(self, error):
        with self.lock:
            if self.first_error is None:
                self.first_error = error

    def _headers(self, headers):
        result = []
        for name, value in headers.raw:
            name, value = bytes(name), bytes(value)
            if name.lower() in SENSITIVE_HEADERS or any(s in value for s in self.secrets):
                continue
            result.append((name, value))
        if sum(len(n) + len(v) for n, v in result) > 65536:
            raise EvidenceError("nonsecret header storage bound exceeded")
        return tuple(result)

    def _body(self, blob, label, errors):
        if type(blob) is not bytes:
            raise EvidenceError("buffered exact bytes required")
        digest, size = sha(blob), len(blob)
        if size > self.max_body_bytes or any(secret in blob for secret in self.secrets):
            error = EvidenceError(label + " body storage bound or known-secret refusal")
            self._latch(error)
            errors.append(str(error))
            return None, digest, size
        return blob, digest, size

    def snapshot(self):
        with self.lock:
            return tuple(self._rows[i] for i in sorted(self._rows))

    def _capture(self, sequence, request, response, primary, started):
        errors = []
        request_body = response_body = None
        request_sha = response_sha = request_size = response_size = None
        request_headers = response_headers = ()
        rid = status = None
        try:
            request_body, request_sha, request_size = self._body(request.content, "request", errors)
            request_headers = self._headers(request.headers)
            if response is not None:
                status = response.status_code
                response_body, response_sha, response_size = self._body(
                    response.content, "response", errors
                )
                response_headers = self._headers(response.headers)
                ids = [v for n, v in response_headers if n.lower() == b"x-public-request-id"]
                if len(ids) != 1 or re.fullmatch(rb"[0-9a-f]{32}", ids[0]) is None:
                    raise EvidenceError("exact single public request ID absent or invalid")
                rid = ids[0].decode("ascii")
        except BaseException as exc:
            self._latch(exc)
            errors.append(self._message(exc))
            if primary is not None:
                primary.add_note("HTTP observer capture secondary: " + self._message(exc))
        row = Row(
            sequence,
            str(request.url),
            started,
            (time.monotonic() - started) * 1000,
            request_body,
            request_sha,
            request_size,
            request_headers,
            response_body,
            response_sha,
            response_size,
            response_headers,
            status,
            rid,
            None if primary is None else (type(primary).__name__, self._message(primary)),
            tuple(errors),
        )
        cost = sum(len(b) for b in (request_body, response_body) if b is not None)
        cost += sum(len(n) + len(v) for n, v in request_headers + response_headers) + len(
            canonical(
                {
                    "endpoint": row.endpoint,
                    "errors": row.capture_errors,
                    "transport": row.transport_error,
                }
            )
        )
        with self.lock:
            if len(self._rows) >= self.max_rows or self.total_bytes + cost > self.max_total_bytes:
                self.omitted_count += 1
                self._latch(
                    EvidenceError(
                        "complete observer row storage bound exceeded; prior rows retained"
                    )
                )
            else:
                self._rows[sequence] = row
                self.total_bytes += cost

    @contextlib.contextmanager
    def install(self, httpx_module):
        original = httpx_module.Client.send
        if getattr(original, "_jev_exact_observer", False):
            raise EvidenceError("nested HTTP observer installation forbidden")
        observer = self

        def observed_send(client, request, *args, **kwargs):
            if request.method != "POST" or str(request.url) not in observer.selected:
                return original(client, request, *args, **kwargs)
            with observer.lock:
                sequence = observer.selected_attempt_total
                observer.selected_attempt_total += 1
                observer._pending.add(sequence)
            started = time.monotonic()
            response = primary = None
            try:
                response = original(client, request, *args, **kwargs)
                return response
            except BaseException as exc:
                primary = exc
                raise
            finally:
                try:
                    observer._capture(sequence, request, response, primary, started)
                except BaseException as secondary:
                    observer._latch(secondary)
                    with observer.lock:
                        observer.omitted_count += 1
                    if primary is not None:
                        primary.add_note("HTTP observer secondary: " + observer._message(secondary))
                finally:
                    with observer.lock:
                        observer._pending.discard(sequence)

        observed_send._jev_exact_observer = True
        httpx_module.Client.send = observed_send
        try:
            yield self
        finally:
            if httpx_module.Client.send is observed_send:
                httpx_module.Client.send = original
            else:
                self._latch(EvidenceError("HTTP observer installation identity drift"))

    def drain(self, output_dir, primary=None):
        """Only owned-finally I/O; candidate rows never imply actual delivery."""
        output = Path(output_dir)
        rows = self.snapshot()
        with self.lock:
            pending = tuple(sorted(self._pending))
        if pending:
            self._latch(EvidenceError("owned observer drain has inflight HTTP sends"))
        refs = []
        try:
            output.mkdir(parents=True, exist_ok=False, mode=0o700)
            for row in rows:
                prefix = f"{row.sequence:04d}"
                bodies = {}
                for label in ("request", "response"):
                    blob = getattr(row, label + "_body")
                    name = prefix + "/" + label + ".bin"
                    if blob is not None:
                        write(output / name, blob)
                    bodies[label] = {
                        "path": name if blob is not None else None,
                        "sha256": getattr(row, label + "_body_sha256"),
                        "size_bytes": getattr(row, label + "_body_size"),
                    }
                value = {
                    "sequence": row.sequence,
                    "endpoint": row.endpoint,
                    "started_monotonic": row.started_monotonic,
                    "elapsed_ms": row.elapsed_ms,
                    "bodies": bodies,
                    "http_status": row.http_status,
                    "request_id": row.request_id,
                    "request_headers_hex": [[n.hex(), v.hex()] for n, v in row.request_headers],
                    "response_headers_hex": [[n.hex(), v.hex()] for n, v in row.response_headers],
                    "transport_error": row.transport_error,
                    "capture_errors": row.capture_errors,
                }
                name = prefix + "/exchange.json"
                write(output / name, canonical(value))
                refs.append({"path": name, **pin(output / name)})
            receipt = {
                "schema_id": "jev-exact-client-transport-observer-v1",
                "complete": self.first_error is None and not pending and self.omitted_count == 0,
                "selected_attempt_total": self.selected_attempt_total,
                "stored_rows": len(rows),
                "omitted_count": self.omitted_count,
                "pending_sequences": pending,
                "unknown_delivery_count": sum(
                    r.transport_error is not None or r.response_body is None or r.request_id is None
                    for r in rows
                ),
                "all_deliveries_observed": all(
                    r.transport_error is None
                    and r.response_body is not None
                    and r.request_id is not None
                    for r in rows
                ),
                "first_observer_error": None
                if self.first_error is None
                else {
                    "type": type(self.first_error).__name__,
                    "message": self._message(self.first_error),
                },
                "rows": refs,
                "bounds": {
                    "rows": self.max_rows,
                    "body_bytes": self.max_body_bytes,
                    "total_bytes": self.max_total_bytes,
                },
                "timing": "Immutable in-memory capture and hashing occur inside original send; fsync happens only after benchmark returns in owned finally and is outside per-attempt timing. No retry, timeout, response or request option is changed.",
                "authority": "Complete response bytes/status/header ID prove SDK client consumption, not server intent. Transport exceptions may expose no response/partial bytes; unknown delivery remains explicit.",
            }
            write(output / "receipt.json", canonical(receipt))
        except BaseException as exc:
            self._latch(exc)
            if primary is not None:
                primary.add_note("HTTP observer drain secondary: " + self._message(exc))
            else:
                raise
            return None
        if self.first_error is not None:
            if primary is None:
                raise self.first_error
            primary.add_note("HTTP observer failed: " + self._message(self.first_error))
        return receipt


def process_identity(pid, *, proc_root=Path("/proc")):
    if type(pid) is not int or pid <= 0:
        raise EvidenceError("positive owned PID required")
    proc = Path(proc_root)
    root = proc / str(pid)
    text = (root / "stat").read_text()
    left, rest = text[: text.rfind(")") + 1], text[text.rfind(")") + 2 :].split()
    if int(left.split(" ", 1)[0]) != pid or rest[0] in {"Z", "X"} or len(rest) < 20:
        raise EvidenceError("owned worker is absent or not live")
    return {
        "pid": pid,
        "ppid": int(rest[1]),
        "pgid": int(rest[2]),
        "session": int(rest[3]),
        "start_ticks": int(rest[19]),
        "boot_id": (proc / "sys/kernel/random/boot_id").read_text().strip(),
        "exe": str((root / "exe").resolve(strict=True)),
        "argv": tuple(x.decode() for x in (root / "cmdline").read_bytes().split(b"\0") if x),
    }


def assert_process_binding(expected, *, proc_root=Path("/proc")):
    actual = process_identity(expected["pid"], proc_root=proc_root)
    for key in ("pid", "pgid", "session", "start_ticks", "boot_id", "exe"):
        if actual[key] != expected[key]:
            raise EvidenceError("owned process identity mismatch: " + key)
    if list(actual["argv"]) != list(expected["argv"]):
        raise EvidenceError("owned process command binding mismatch")
    return actual


def listener_identity(pid, port, *, proc_root=Path("/proc")):
    if type(port) is not int or not 0 < port < 65536:
        raise EvidenceError("explicit valid owned listener port required")
    root = Path(proc_root) / str(pid)
    inodes = set()
    for fd in (root / "fd").iterdir():
        target = os.readlink(fd)
        match = re.fullmatch(r"socket:\[([0-9]+)\]", target)
        if match:
            inodes.add(match[1])
    selected = []
    for line in (root / "net/tcp").read_text().splitlines()[1:]:
        fields = line.split()
        if (
            len(fields) > 9
            and fields[1] == f"0100007F:{port:04X}"
            and fields[3] == "0A"
            and fields[9] in inodes
        ):
            selected.append(fields[9])
    if len(selected) != 1:
        raise EvidenceError("exact loopback LISTEN inode is not uniquely owned by selected worker")
    return {"pid": pid, "address": "127.0.0.1", "port": port, "inode": selected[0]}


def checked_ref(value):
    path = ROOT / value["path"]
    if pin(path) != {k: value[k] for k in ("sha256", "size_bytes")}:
        raise EvidenceError("selected public reference changed")
    return path


def checked_plan():
    if (
        pin(BASE / "plan-01.json")["sha256"] != PLAN_SHA
        or pin(ROOT / "scripts/jev_qualify_public_field_syntax_bf16.py")["sha256"] != SOURCE_SHA
    ):
        raise EvidenceError("prepared plan or qualified source changed")
    plan = read_json(BASE / "plan-01.json")
    for ref in plan["syntax_existing_source_refs"] + plan["benchmark_source_refs"]:
        checked_ref(ref)
    for name, expected in plan["native_upstream"]["source_sha256"].items():
        if pin(ROOT / "artifacts/jev-runtime/upstream" / name)["sha256"] != expected:
            raise EvidenceError("pinned upstream bytes changed")
    revision = subprocess.run(
        ["git", "-C", str(ROOT / "artifacts/jev-runtime/upstream"), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != plan["native_upstream"]["revision"]:
        raise EvidenceError("pinned upstream revision changed")
    return plan


def capacity_state(binding):
    """Pure saved CPU inputs/closure check; separate from technical model Q."""
    plan = checked_plan()
    expected = plan["v8_capacity"]
    path = checked_ref(expected["binding_source"])
    blob = path.read_bytes()
    if sha(blob) != expected["binding_source"]["sha256"]:
        raise EvidenceError("selected v8 capacity binding source changed")
    spec = importlib.util.spec_from_file_location("_v8_capacity_binding", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(blob, str(path), "exec"), module.__dict__)
    return module.validate_selection(binding.get("v8_capacity"), expected, root=ROOT)


def binding_state(binding):
    """Read only; private credentials are compared, never returned or serialized."""
    capacity = capacity_state(binding)
    if binding.get("schema_id") != "jev-field-syntax-v8-browser-owned-binding-v1":
        raise EvidenceError("concrete independently selected service binding required")
    qpath = checked_ref(binding["qualification"])
    q = read_json(qpath)
    if (
        binding["qualification"]["sha256"] != Q_SHA
        or q["kind"] != "jev_public_action_field_syntax_bf16_qualification_v1"
        or q["qualified"] is not True
        or q["task_performance_attested"] is not False
    ):
        raise EvidenceError("exact newly qualified field-syntax technical condition required")
    verification = read_json(checked_ref(binding["verification"]))
    if (
        set(verification) != {"qualification_sha256", "scope", "success"}
        or verification.get("success") is not True
        or verification.get("qualification_sha256") != Q_SHA
        or verification.get("scope")
        != "Saved-byte/source/tensor verification; no new runtime or browser task score"
    ):
        raise EvidenceError("independent field-syntax verification required")
    service = (ROOT / binding["service_directory"]).resolve(strict=True)
    for name in (
        "shutdown.request",
        "owned-exit.json",
        "controller-receipt.json",
        "controller-failure.json",
    ):
        if (service / name).exists():
            raise EvidenceError("selected service already stopping or closed")
    config_path = service / "controller-config.json"
    if config_path.is_symlink() or stat.S_IMODE(config_path.stat().st_mode) != 0o600:
        raise EvidenceError("owned live private config must be ordinary mode0600")
    config = read_json(config_path)
    if (
        config["command"] != "serve"
        or config["qualification_sha256"] != Q_SHA
        or Path(config["qualification_path"]).resolve() != qpath.parent.resolve()
        or config["execution_sha256"] != q["execution_sha256"]
        or config["runtime_sha256"] != q["runtime_sha256"]
        or config["plan_sha256"] != q["plan_sha256"]
    ):
        raise EvidenceError("owned live config does not select the exact new qualification")
    endpoint_path = service / "endpoint.json"
    endpoint = read_json(endpoint_path)
    _, port = endpoints(endpoint["policy"], endpoint["helper"])
    if (
        endpoint["qualified"] is not True
        or endpoint["model"] != "generated-actions-none"
        or endpoint["TEXT_MODEL_REASONING"] != "none"
    ):
        raise EvidenceError("owned ready endpoint model/profile identity differs")
    worker = assert_process_binding(binding["worker"])
    if worker["pgid"] != worker["pid"] or worker["session"] != worker["pid"]:
        raise EvidenceError("owned fresh-session model worker required")
    expected_argv = [
        str(Path(binding["interpreter"])),
        str(ROOT / "scripts/jev_qualify_public_field_syntax_bf16.py"),
        "--internal-child",
        str(config_path),
    ]
    if (
        list(worker["argv"]) != expected_argv
        or Path(worker["exe"]) != Path(binding["interpreter"]).resolve()
    ):
        raise EvidenceError("owned listener command/executable source binding differs")
    listener = listener_identity(worker["pid"], port)
    if listener != binding["listener"]:
        raise EvidenceError("owned listener inode changed")
    secrets = tuple(os.environ.get(name, "") for name in ("S1_POLICY_API_KEY", "S1_TEXT_API_KEY"))
    if not all(secrets) or secrets[0] != secrets[1] or secrets[0] != config["token"]:
        raise EvidenceError("two env-only credentials must select this exact owned service")
    if any(secret.encode() in canonical(binding) for secret in secrets):
        raise EvidenceError("private credential must not occur in public binding")
    if (
        assert_process_binding(binding["worker"]) != worker
        or listener_identity(worker["pid"], port) != listener
    ):
        raise EvidenceError("owned listener process changed during binding readback")
    return {
        "worker": worker,
        "listener": listener,
        "endpoint": endpoint,
        "endpoint_ref": {"path": str(endpoint_path.relative_to(ROOT)), **pin(endpoint_path)},
        "qualification_sha256": Q_SHA,
        "v8_capacity": capacity,
    }, secrets


def launcher_module():
    old_path = list(sys.path)
    case_names = ("jev_heldout_public_cases", "jev_public_web_cases")
    saved = {name: sys.modules.get(name) for name in case_names}
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        spec = importlib.util.spec_from_file_location(
            "syntax_v8_isolated_launcher", ROOT / "scripts/jev_browser_profile_v8_benchmark.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path
        # Original wrapper preparation is in a different process from its
        # benchmark child. Keep the same fresh bundle-case import behavior.
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def prepare(group_name, binding_path, output):
    """Root may execute this CPU/read-only-service preparation after source review."""
    plan = checked_plan()
    binding = read_json(binding_path)
    public_binding, _ = binding_state(binding)
    group = next(g for g in plan["groups_in_execution_order"] if g["group"] == group_name)
    output = Path(output).resolve()
    if output.exists():
        raise EvidenceError("fresh owned browser group directory required")
    output.mkdir(parents=True, mode=0o700)
    argv = list(group["argv_template"])
    for flag, value in (
        ("--endpoint", public_binding["endpoint"]["policy"]),
        ("--text-endpoint", public_binding["endpoint"]["helper"]),
        ("--output-dir", str(output / "results")),
    ):
        argv[argv.index(flag) + 1] = value
    bundle = manifest = None
    module = launcher_module()
    case_module = "original" if group_name == "known4" else "heldout"
    arguments = argv[argv.index("--") + 1 :]
    forwarded = module._arguments(
        arguments, case_module, plan["browser_profile"]["helper_source_sha256"]
    )
    bundle, manifest = module.prepare_bundle(output / "runtime-bundle", case_module=case_module)
    module.verify_bundle(bundle, manifest)
    target = bundle / "runner.py"
    launch = {
        "schema_id": "jev-field-syntax-v8-observed-browser-launch-v1",
        "group": group_name,
        "case_ids": group["selected_case_ids_in_order"],
        "denominator": group["denominator"],
        "target": str(target),
        "forwarded_argv": forwarded,
        "binding": {
            "path": str(Path(binding_path).resolve().relative_to(ROOT)),
            **pin(binding_path),
        },
        "public_service_identity": public_binding,
        "observer_source": {"path": str(Path(__file__).relative_to(ROOT)), **pin(__file__)},
        "target_pin": pin(target),
        "bundle_manifest": manifest,
        "conditions": "Explicit v8 isolated runner/profile and diagnostic sidecar, with unchanged native client request/retry semantics. Transparent in-memory transport capture and hashing occur within original send; final fsync occurs outside attempt timing. Cooperative five-second observe targets and instrumentation are part of this measured condition.",
        "environment": {
            "updates": plan["environment"]["updates"],
            "secret_names_only": ["S1_POLICY_API_KEY", "S1_TEXT_API_KEY"],
        },
    }
    write(output / "launch.json", canonical(launch))
    command = [
        sys.executable,
        "-B",
        "-s",
        str(Path(__file__).resolve()),
        "child",
        "--run",
        "--launch",
        str(output / "launch.json"),
        "--launch-sha256",
        pin(output / "launch.json")["sha256"],
    ]
    write(
        output / "child-command.json",
        canonical(
            {
                "argv": command,
                "owned_fresh_session_required": True,
                "root_controller_obligation": "Start this exact child with owned PGID, preserve stdout/stderr and exit, bounded owned TERM/KILL/reap on cancellation; do not stop the selected model service during a group.",
            }
        ),
    )
    return command


def child(launch_path, expected_launch_sha):
    launch_path = Path(launch_path).resolve()
    if pin(launch_path)["sha256"] != expected_launch_sha:
        raise EvidenceError("externally selected concrete launch bytes changed")
    launch = read_json(launch_path)
    output = launch_path.parent
    checked_plan()
    checked_ref(launch["observer_source"])
    binding_path = checked_ref(launch["binding"])
    state, secrets = binding_state(read_json(binding_path))
    if canonical(state) != canonical(launch["public_service_identity"]):
        raise EvidenceError("owned service binding changed before fresh child")
    target = Path(launch["target"])
    if pin(target) != launch["target_pin"]:
        raise EvidenceError("selected v8 isolated benchmark runner changed")
    module = launcher_module()
    if module is not None:
        module.verify_bundle(target.parent, launch["bundle_manifest"])
    import httpx

    observer = TransportObserver(
        state["endpoint"]["policy"], state["endpoint"]["helper"], secrets=secrets
    )
    primary = None
    old_argv = sys.argv
    old_path = list(sys.path)
    started = time.monotonic()
    try:
        sys.path.insert(0, str(target.parent))
        sys.argv = [str(target), *launch["forwarded_argv"]]
        with observer.install(httpx):
            runpy.run_path(str(target), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (0, None):
            primary = exc
    except BaseException as exc:
        primary = exc
    finally:
        sys.argv, sys.path = old_argv, old_path
        try:
            observer.drain(output / "http-transport", primary=primary)
        except BaseException as exc:
            if primary is None:
                primary = exc
            else:
                primary.add_note("Observer finalization secondary: " + observer._message(exc))
        try:
            checked_plan()
            current, _ = binding_state(read_json(binding_path))
            if current != state:
                raise EvidenceError("owned service binding changed during group")
            if module is not None:
                module.verify_bundle(target.parent, launch["bundle_manifest"])
        except BaseException as exc:
            if primary is None:
                primary = exc
            else:
                primary.add_note("Post-group identity secondary: " + observer._message(exc))
        receipt = {
            "schema_id": "jev-field-syntax-v8-observed-browser-child-exit-v1",
            "complete": primary is None,
            "group": launch["group"],
            "declared_denominator": launch["denominator"],
            "selected_HTTP_attempts": observer.selected_attempt_total,
            "observer_rows": len(observer.snapshot()),
            "observer_omitted": observer.omitted_count,
            "first_observer_error": None
            if observer.first_error is None
            else observer._message(observer.first_error),
            "error": None
            if primary is None
            else {
                "type": type(primary).__name__,
                "message": observer._message(primary),
                "notes": [observer._message(n) for n in getattr(primary, "__notes__", [])],
            },
            "elapsed_ms_including_final_drain": (time.monotonic() - started) * 1000,
            "process_exit_is_task_success": False,
            "server_selected_row_is_delivered_claim": False,
        }
        try:
            write(output / "child-exit.json", canonical(receipt))
        except BaseException as exc:
            if primary is None:
                primary = exc
            else:
                primary.add_note("Child exit receipt secondary: " + observer._message(exc))
    if primary is not None:
        raise primary
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", allow_abbrev=False)
    prep.add_argument("--group", choices=("known4", "public6"), required=True)
    prep.add_argument("--binding", required=True)
    prep.add_argument("--output", required=True)
    execute = commands.add_parser("child", allow_abbrev=False)
    execute.add_argument("--run", action="store_true", required=True)
    execute.add_argument("--launch", required=True)
    execute.add_argument("--launch-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(
            json.dumps(
                {
                    "prepared_child_argv": prepare(args.group, args.binding, args.output),
                    "browser_executed": False,
                    "HTTP": 0,
                }
            )
        )
        return 0
    return child(args.launch, args.launch_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
