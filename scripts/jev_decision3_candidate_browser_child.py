"""Source-only multi-request candidate browser child, supervised externally.

CLI: --plan ABSOLUTE_JSON --plan-sha256 SHA256. The parent creates a private
output directory, starts this exact child in the specified cgroup and verifies
hello.json before atomically publishing go.json. The child then publishes a
private ready.json containing the loopback endpoint and bearer token. The parent
atomically publishes stop.json after the actor finishes. A terminal.json report
records only this child's evidence; host supervision and client capture remain
external, and no task score or numerical qualification is claimed.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import inspect
import json
import os
import re
import secrets
import stat
import sys
import time
from pathlib import Path

import jev_decision3_candidate_http_owner as common

SOURCE_ONLY = True
SCHEMA = "decision3-candidate-browser-child-plan-v1"
UNIT_PATTERN = r"jev-browser-candidate-[0-9a-f]{32}\.service"
MAX_CONTROL = 8192
MAX_PRIVATE_FILE = 64 * 1024**2


def _need(value, message):
    if not value:
        raise ValueError(message)


def _gate():
    _need(SOURCE_ONLY is False, "held browser child requires independent review")


def _encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _ordinary_absolute(value):
    if type(value) is not str or not value or value.startswith("//"):
        return False
    path = Path(value)
    return path.is_absolute() and str(path) == value and ".." not in path.parts


def _validate_plan(plan):
    keys = {
        "schema",
        "unit",
        "output_dir",
        "stop_path",
        "python",
        "cache_root",
        "modules_cache",
        "overlay",
        "source_pins",
        "gpu_uuid",
        "deltas",
        "transport",
        "model_alias",
        "max_requests",
        "wall_seconds",
    }
    _need(
        type(plan) is dict and set(plan) == keys and plan["schema"] == SCHEMA,
        "exact child plan required",
    )
    _need(
        type(plan["unit"]) is str and re.fullmatch(UNIT_PATTERN, plan["unit"]),
        "fresh explicit candidate unit required",
    )
    _need(
        all(
            _ordinary_absolute(plan[key])
            for key in (
                "output_dir",
                "stop_path",
                "python",
                "cache_root",
                "modules_cache",
                "overlay",
            )
        ),
        "canonical absolute child paths required",
    )
    _need(
        Path(plan["stop_path"]) == Path(plan["output_dir"]) / "stop.json",
        "stop path must be the owned output stop.json",
    )
    _need(
        type(plan["model_alias"]) is str
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", plan["model_alias"]),
        "explicit browser model alias required",
    )
    _need(
        type(plan["max_requests"]) is int and 1 <= plan["max_requests"] <= 16,
        "one to sixteen requests required",
    )
    _need(
        type(plan["wall_seconds"]) is int and 30 <= plan["wall_seconds"] <= 600,
        "bounded child wall time required",
    )
    _need(
        type(plan["gpu_uuid"]) is str
        and re.fullmatch(
            r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", plan["gpu_uuid"]
        ),
        "selected GPU UUID required",
    )
    _need(
        type(plan["source_pins"]) is dict
        and bool(plan["source_pins"])
        and all(
            _ordinary_absolute(path)
            and type(digest) is str
            and re.fullmatch(r"[0-9a-f]{64}", digest)
            for path, digest in plan["source_pins"].items()
        ),
        "explicit source SHA inventory required",
    )
    _need(
        type(plan["deltas"]) is dict and set(plan["deltas"]) == {"operation", "click_target"},
        "both candidate deltas required",
    )
    for value in plan["deltas"].values():
        _need(
            type(value) is dict
            and set(value) == {"path", "file_sha256", "fp32_sha256"}
            and _ordinary_absolute(value["path"])
            and all(
                type(value[key]) is str and re.fullmatch(r"[0-9a-f]{64}", value[key])
                for key in ("file_sha256", "fp32_sha256")
            ),
            "absolute two-digest candidate deltas required",
        )
    transport = plan["transport"]
    releases = ("http_release", "lease_release", "binding_release", "lifecycle_release")
    _need(
        type(transport) is dict
        and set(transport) == {*releases, "lock", "journal", "nonce", "stat"}
        and all(_ordinary_absolute(transport[key]) for key in (*releases, "lock", "journal"))
        and type(transport["nonce"]) is str
        and re.fullmatch(r"[0-9a-f]{32}", transport["nonce"])
        and type(transport["stat"]) is dict
        and set(transport["stat"]) == {"lock", "journal"}
        and all(
            type(row) is dict
            and set(row) == {"device", "inode"}
            and all(type(number) is int and number >= 0 for number in row.values())
            for row in transport["stat"].values()
        ),
        "reviewed releases and same-inode GPU lease required",
    )
    workspace = Path(__file__).absolute().parent.parent
    _need(
        Path(plan["overlay"]) == workspace / "artifacts/jev-runtime/overlay",
        "fixed workspace overlay required",
    )


def _read_bounded(path, limit):
    path = Path(path)
    _need(
        path.is_absolute() and path.resolve() == path and path.is_file(),
        "ordinary private file required",
    )
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        _need(
            stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit,
            "bounded regular private file required",
        )
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        after = os.fstat(fd)
        current = path.stat()
        _need(
            len(raw) == before.st_size
            and (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            == (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
            )
            and path.resolve() == path,
            "private file changed while reading",
        )
        return raw
    finally:
        os.close(fd)


def _load_plan(path, expected_sha):
    _need(
        type(expected_sha) is str and re.fullmatch(r"[0-9a-f]{64}", expected_sha),
        "explicit plan SHA required",
    )
    raw = _read_bounded(path, 1024**2)
    _need(_sha(raw) == expected_sha, "exact child plan digest required")
    plan = common._decode(raw)
    _validate_plan(plan)
    return plan


def _control(path, *, schema, invocation, plan_sha):
    value = common._decode(_read_bounded(path, MAX_CONTROL))
    _need(
        value == {"schema": schema, "invocation": invocation, "plan_sha256": plan_sha},
        "exact bound parent control required",
    )
    return value


def _append(path, value):
    raw = _encode(value) + b"\n"
    _need(len(raw) <= MAX_PRIVATE_FILE, "bounded private event required")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        _need(
            stat.S_ISREG(info.st_mode) and info.st_size + len(raw) <= MAX_PRIVATE_FILE,
            "bounded private journal required",
        )
        _need(os.write(fd, raw) == len(raw), "complete private journal write required")
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_control_bytes(stream, data):
    stream.write(data)
    stream.flush()
    os.fsync(stream.fileno())


def _publish_control(path, data):
    """Publish a complete private handshake without replacing an existing file."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(16)}.tmp")
    created = False
    try:
        fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        created = True
        with os.fdopen(fd, "wb") as stream:
            _write_control_bytes(stream, data)
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if created:
            temporary.unlink()


def _runtime_modules(plan):
    _gate()
    pins = plan["source_pins"]
    specs = (
        ("http_release", "jev_decision3_candidate_http.py", "_candidate_child_http"),
        ("lease_release", "jev_gpu_forward_lease.py", "_candidate_child_lease"),
        (
            "binding_release",
            "jev_decision3_candidate_browser_binding.py",
            "_candidate_child_binding",
        ),
        (
            "lifecycle_release",
            "jev_decision3_candidate_service_lifecycle.py",
            "_candidate_child_lifecycle",
        ),
    )
    return tuple(
        common._load_reviewed_release(plan["transport"][key], held, pins, name)
        for key, held, name in specs
    )


def _require_actual_sources(plan, backend, modules):
    """Ensure the actual publisher and original conversion dependencies are pinned."""
    import jev_decision3_link_navigation_adapter as adapter
    import jev_decision3_link_navigation_adapter_audit as audit
    import jev_decision3_link_navigation_features as features

    from s1.backends import normalize_response
    from s1.contracts import validate_distribution
    from s1.evaluation.decision3 import native_payload
    from s1.jev_decision3_native import parse_native_payload
    from s1.jev_generated import _request
    from s1.jev_hierarchy import _check_json

    http, _lease_module, _binding_module, _lifecycle_module = modules
    files = (
        __file__,
        common.__file__,
        adapter.__file__,
        adapter.reference.__file__,
        audit.__file__,
        features.__file__,
        http.wire.__file__,
        inspect.getfile(http.DeadlineBackend),
        inspect.getfile(type(backend)),
        inspect.getfile(type(backend.model)),
        inspect.getfile(type(backend.runtime)),
        inspect.getfile(native_payload),
        inspect.getfile(parse_native_payload),
        inspect.getfile(_request),
        inspect.getfile(_check_json),
        inspect.getfile(normalize_response),
        inspect.getfile(validate_distribution),
    )
    _need(
        all(str(Path(file).absolute()) in plan["source_pins"] for file in files),
        "actual owner, candidate, publisher and serializer sources must be pinned",
    )
    common._verify_source_pins(plan["source_pins"])


def _lease_summary(transport, pid, count):
    """Check this child has exactly one complete lease pair per HTTP call."""
    common._shared_identity(transport)
    fd = os.open(transport["journal"], os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        _need(
            stat.S_ISREG(before.st_mode) and before.st_size <= 16 * 1024**2,
            "bounded lease journal required",
        )
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(16 * 1024**2 + 1)
        after = os.fstat(fd)
        _need(
            len(raw) == before.st_size
            and (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            "lease journal changed during read",
        )
    finally:
        os.close(fd)
    common._shared_identity(transport)
    rows = [common._decode(line) for line in raw.splitlines()]
    own = [row for row in rows if row.get("nonce") == transport["nonce"] and row.get("pid") == pid]
    _need(
        type(count) is int and len(own) == count * 2, "one lease pair per completed call required"
    )
    for acquired, completed in zip(own[::2], own[1::2], strict=True):
        _need(
            acquired.get("event") == "acquired"
            and completed.get("event") == "completed"
            and acquired.get("kind") == completed.get("kind") == "decision3-gpu"
            and type(acquired.get("thread")) is int
            and acquired["thread"] == completed.get("thread")
            and type(acquired.get("monotonic_ns")) is int
            and type(completed.get("monotonic_ns")) is int
            and acquired["monotonic_ns"] <= completed["monotonic_ns"],
            "matching completed same-thread lease records required",
        )
    return {
        "sha256": _sha(raw),
        "own_completed_pairs": count,
        "scope": "per-child lease pairs; cross-process overlap requires independent audit",
    }


def _serve(plan, plan_sha, invocation, backend, torch, modules, output, check):
    """Run the private candidate listener until exact parent stop or request cap."""
    _gate()
    import jev_decision3_link_navigation_adapter_audit as audit
    import jev_decision3_link_navigation_features as features

    http, lease_module, binding_module, lifecycle_module = modules
    transport = plan["transport"]
    common._shared_identity(transport)
    lease = lease_module.ForwardLease(
        transport["lock"], transport["journal"], transport["stat"], nonce=transport["nonce"]
    )
    sink, receipts = audit.make_witness_writer(
        output / "witnesses", maximum_calls=plan["max_requests"]
    )
    cpu_deltas = common._load_deltas(plan, torch)
    expected = {head: ref["fp32_sha256"] for head, ref in plan["deltas"].items()}
    http_journal, life_journal = output / "http-events.jsonl", output / "lifecycle-events.jsonl"
    common._write_new(http_journal, b"")
    common._write_new(life_journal, b"")
    private_records = []
    pinned_identity = {
        "unit": plan["unit"],
        "invocation": invocation,
        "gpu_uuid": plan["gpu_uuid"],
        "plan_sha256": plan_sha,
        "nonce": transport["nonce"],
        "delta_sha256": expected,
    }
    lifecycle = None
    binding = None

    def authenticate():
        check()
        common._verify_source_pins(plan["source_pins"])
        common._shared_identity(transport)
        features._validate_manifest(backend.runtime, plan["cache_root"])

    def identity_guard(actual, _deltas):
        _need(actual is backend, "original owned publisher backend required")
        authenticate()
        return copy.deepcopy(pinned_identity)

    def owner_guard(actual, _phase):
        _need(
            binding is not None
            and actual is binding.serial
            and actual.backend is binding.wrapper
            and actual.lease is lease,
            "owned serial, candidate and shared lease required",
        )
        authenticate()
        return copy.deepcopy(pinned_identity)

    def read_witness(receipt):
        index = len(private_records)
        directory = output / "witnesses" / f"call-{index:04d}"
        _need(
            receipt.get("directory") == str(directory), "ordered owned witness directory required"
        )
        return _read_bounded(directory / "witness.json", 8 * 1024**2)

    def private_sink(record):
        number = len(private_records) + 1
        _need(
            record.get("call") == number and number <= plan["max_requests"],
            "ordered private binding required",
        )
        raw = _encode(record)
        _need(0 < len(raw) <= MAX_PRIVATE_FILE, "bounded private binding required")
        name = f"binding-{number:04d}.json"
        common._write_new(output / name, raw)
        private_records.append({"path": name, "sha256": _sha(raw), "size_bytes": len(raw)})

    def guard_closed():
        return bool(
            lifecycle is not None
            and lifecycle.done.is_set()
            and lifecycle.thread is not None
            and not lifecycle.thread.is_alive()
            and lifecycle.server.fileno() == -1
        )

    primary = None
    try:
        # CandidateHTTP authenticates during its constructor before binding is
        # assigned, so the callback verifies the explicit actual object then.
        def initial_owner_guard(actual, phase):
            _need(
                actual.lease is lease and actual.backend.base is backend,
                "owned initial serial backend required",
            )
            if binding is not None:
                return owner_guard(actual, phase)
            authenticate()
            return copy.deepcopy(pinned_identity)

        binding = binding_module.CandidateBrowserBinding(
            backend,
            cpu_deltas,
            expected,
            torch,
            lease_module=lease_module,
            lease=lease,
            http_module=http,
            model_identity=plan["model_alias"],
            token=secrets.token_urlsafe(32),
            max_requests=plan["max_requests"],
            identity_guard=identity_guard,
            owner_guard=initial_owner_guard,
            witness_sink=sink,
            witness_snapshot=lambda: copy.deepcopy(receipts),
            read_witness=read_witness,
            private_artifact_sink=private_sink,
            event_sink=lambda row: _append(http_journal, row),
            closure_guard=guard_closed,
        )

        def closure_guard():
            _need(
                guard_closed() and lease.fd is None and lease.held is None and lease.fatal is None,
                "listener and completed lease closure required",
            )

        def cleanup():
            binding.close()
            return {"candidate_binding_closed": True, "lease_unheld": True}

        lifecycle = lifecycle_module.CandidateServiceLifecycle(
            http,
            binding.service,
            supervisor=check,
            closure_guard=closure_guard,
            cleanup=cleanup,
            event_sink=lambda row: _append(life_journal, row),
            poll_seconds=0.1,
        )
        host, port = lifecycle.start()
        _publish_control(
            output / "ready.json",
            _encode(
                {
                    "schema": "decision3-candidate-browser-ready-v1",
                    "invocation": invocation,
                    "plan_sha256": plan_sha,
                    "pid": os.getpid(),
                    "host": host,
                    "port": port,
                    "model_alias": plan["model_alias"],
                    "token": binding.service.token,
                    "max_requests": plan["max_requests"],
                    "scope": "private endpoint; host supervision external",
                }
            ),
        )
        while not lifecycle.done.wait(0.1):
            check()
            if Path(plan["stop_path"]).exists():
                _control(
                    plan["stop_path"],
                    schema="decision3-candidate-browser-stop-v1",
                    invocation=invocation,
                    plan_sha=plan_sha,
                )
                break
        result = lifecycle.stop(timeout_seconds=3.0)
        _need(
            result["status"] == "stopped" and result["reason"] in {"requested", "request_limit"},
            "candidate listener failed",
        )
        _need(
            result["request_calls"] == len(receipts) == len(private_records),
            "completed request, witness and binding counts differ",
        )
        lease_summary = _lease_summary(transport, os.getpid(), result["request_calls"])
        authenticate()
        common._load_deltas(plan, torch)
        return {
            "request_calls": result["request_calls"],
            "stop_reason": result["reason"],
            "private_bindings": private_records,
            "witness_receipts": copy.deepcopy(receipts),
            "lease_summary": lease_summary,
            "model_score": None,
            "fit_eligible": False,
        }
    except BaseException as error:
        primary = error
        raise
    finally:
        # A failed service or binding cleanup leaves closure unconfirmed.
        # Leave it for the external owner to terminate; do not claim closure.
        try:
            if lifecycle is not None and not lifecycle.cleanup_finished:
                lifecycle.stop(timeout_seconds=3.0)
            elif lifecycle is None and binding is not None and not binding.closed:
                binding.close()
        except BaseException as cleanup_error:
            if primary is None:
                cleanup_error.candidate_cleanup_unconfirmed = True
                raise
            primary.candidate_cleanup_unconfirmed = True
            primary.add_note("candidate lifecycle cleanup also failed: " + repr(cleanup_error))


def main(argv=None):
    _gate()
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args(argv)
    return run(args.plan, args.plan_sha256)


def run(plan_path, plan_sha):
    """Execute only inside the explicitly bound owner-provided user unit."""
    _gate()
    plan = _load_plan(plan_path, plan_sha)
    output = Path(plan["output_dir"])
    _need(
        output.is_dir() and output.resolve() == output, "existing ordinary private output required"
    )
    _need((output.stat().st_mode & 0o077) == 0, "private output directory required")
    _need(
        not any(
            (output / name).exists()
            for name in ("hello.json", "go.json", "ready.json", "stop.json", "terminal.json")
        ),
        "fresh handshake files required",
    )
    common._own_cgroup(plan["unit"])
    invocation = os.environ.get("INVOCATION_ID", "")
    _need(re.fullmatch(r"[0-9a-f]{32}", invocation), "systemd invocation required")
    common._verify_source_pins(plan["source_pins"])
    _need(
        str(Path(__file__).absolute()) in plan["source_pins"]
        and str(Path(common.__file__).absolute()) in plan["source_pins"],
        "child and shared owner source pins required",
    )
    common._validate_interpreter(plan)
    hello = {
        "schema": "decision3-candidate-browser-hello-v1",
        "pid": os.getpid(),
        "startticks": common.systemd.process_identity(os.getpid())["startticks"],
        "boot_id": common.systemd.boot_id(),
        "unit": plan["unit"],
        "invocation": invocation,
        "plan_sha256": plan_sha,
    }
    _publish_control(output / "hello.json", _encode(hello))
    started = time.monotonic()
    while not (output / "go.json").exists():
        _need(time.monotonic() - started < 20, "parent handshake timed out")
        time.sleep(0.1)
    _control(
        output / "go.json",
        schema="decision3-candidate-browser-go-v1",
        invocation=invocation,
        plan_sha=plan_sha,
    )
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_CACHE": plan["cache_root"],
            "HF_MODULES_CACHE": plan["modules_cache"],
            "CUDA_VISIBLE_DEVICES": plan["gpu_uuid"],
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "5",
            "MKL_NUM_THREADS": "5",
            "OPENBLAS_NUM_THREADS": "5",
        }
    )
    workspace = Path(__file__).absolute().parent.parent
    common._bind_project_source(workspace)
    common._verify_source_pins(plan["source_pins"])
    torch = common._bind_overlay(workspace, plan["source_pins"])
    import jev_decision3_link_navigation_features as features

    from s1.evaluation.decision3 import Decision3Backend, decision3_spec

    _need(
        all(
            getattr(module, "__file__", None)
            and Path(module.__file__).resolve().is_relative_to((workspace / "src").resolve())
            for name, module in tuple(sys.modules.items())
            if name == "s1" or name.startswith("s1.")
        ),
        "actual project modules must come from workspace",
    )
    started = time.monotonic()

    def check():
        _need(
            time.monotonic() - started < plan["wall_seconds"], "child cooperative deadline exceeded"
        )
        common._own_cgroup(plan["unit"])
        _need(os.environ.get("INVOCATION_ID") == invocation, "child invocation changed")

    backend = None
    primary = None
    cleanup_unconfirmed = False
    status = {
        "schema": "decision3-candidate-browser-terminal-v1",
        "status": "running",
        "invocation": invocation,
        "plan_sha256": plan_sha,
        "model_score": None,
        "fit_eligible": False,
        "host_supervision": "external",
    }
    try:
        check()
        common._shared_identity(plan["transport"])
        torch.set_num_threads(5)
        torch.set_num_interop_threads(1)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        _need(
            torch.cuda.is_available() and torch.cuda.device_count() == 1,
            "exactly one selected GPU required",
        )
        gpu = torch.cuda.get_device_properties(0)
        _need(
            str(gpu.uuid).lower().removeprefix("gpu-")
            == plan["gpu_uuid"].lower().removeprefix("gpu-")
            and (gpu.major, gpu.minor) == (12, 1)
            and torch.cuda.is_bf16_supported(including_emulation=False),
            "selected GB10 BF16 GPU required",
        )
        torch.cuda.synchronize(0)
        _need(torch.cuda.memory_allocated(0) == 0, "fresh CUDA allocator required")
        modules = _runtime_modules(plan)
        backend = Decision3Backend(decision3_spec("d3-edge", device="cuda:0"))
        _require_actual_sources(plan, backend, modules)
        features._validate_manifest(backend.runtime, plan["cache_root"])
        status["service"] = _serve(
            plan, plan_sha, invocation, backend, torch, modules, output, check
        )
        status["status"] = "stopped_pending_external_audit"
    except BaseException as error:
        primary = error
        cleanup_unconfirmed = bool(getattr(error, "candidate_cleanup_unconfirmed", False))
        status.update(
            status="failed_no_task_result",
            error_type=type(error).__name__,
            error_stack=common._drop_exception_frames(error),
        )
    finally:
        if cleanup_unconfirmed:
            status.update(
                status="failed_cleanup_unconfirmed",
                cleanup_skipped="service or lease closure unconfirmed; external owner must terminate this invocation",
            )
            _publish_control(output / "terminal.json", _encode(status))
            raise primary
        try:
            if backend is not None:
                backend.close()
        except BaseException as error:
            status.update(
                status="failed_cleanup_unconfirmed", close_error_type=type(error).__name__
            )
            common._drop_exception_frames(error)
            if primary is None:
                primary = error
            else:
                primary.add_note("backend close also failed: " + repr(error))
        backend = None
        gc.collect()
        try:
            torch.cuda.synchronize(0)
            clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
            if callable(clear):
                clear()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(0)
            status["final_cuda_allocated"] = torch.cuda.memory_allocated(0)
            _need(status["final_cuda_allocated"] == 0, "worker CUDA allocation not cleared")
        except BaseException as error:
            status.update(
                status="failed_cleanup_unconfirmed", cleanup_error_type=type(error).__name__
            )
            common._drop_exception_frames(error)
            if primary is None:
                primary = error
            else:
                primary.add_note("CUDA cleanup also failed: " + repr(error))
        _publish_control(output / "terminal.json", _encode(status))
    if primary is not None:
        raise primary
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
