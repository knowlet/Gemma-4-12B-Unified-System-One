"""Held supervised Decision3 generic Benchmark Heaven worker.

The parent authenticates and supervises this exact invocation and its cgroup;
this process cannot hard-cancel a stalled forward. The original scorer runs
unchanged. No result is accepted as a candidate score without external audit.
CLI: --plan ABSOLUTE_JSON --plan-sha256 SHA256.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import os
import re
import stat
import sys
import time
from pathlib import Path

import jev_decision3_candidate_browser_child as child
import jev_decision3_candidate_http_owner as common

SOURCE_ONLY = True
SCHEMA = "decision3-jevbench-heaven-owner-plan-v1"
CONFIG_SHA256 = "206ca6745290cc4bc5c577d64addad7814381cdc6d336b5b7c09cb8992af8754"


def _need(value, message):
    if not value:
        raise ValueError(message)


def _gate():
    _need(SOURCE_ONLY is False, "held Heaven owner requires independent review")


def _validate_plan(plan):
    keys = {
        "schema",
        "unit",
        "output_dir",
        "python",
        "cache_root",
        "modules_cache",
        "overlay",
        "source_pins",
        "gpu_uuid",
        "deltas",
        "transport",
        "generic_release",
        "tasks_root",
        "wall_seconds",
        "request_seconds",
    }
    _need(
        type(plan) is dict and set(plan) == keys and plan["schema"] == SCHEMA,
        "exact Heaven plan required",
    )
    _need(
        type(plan["unit"]) is str
        and re.fullmatch(r"jev-heaven-[0-9a-f]{32}\.service", plan["unit"]),
        "fresh Heaven unit required",
    )
    _need(
        all(
            child._ordinary_absolute(plan[key])
            for key in (
                "output_dir",
                "python",
                "cache_root",
                "modules_cache",
                "overlay",
                "generic_release",
                "tasks_root",
            )
        ),
        "canonical absolute paths required",
    )
    _need(
        type(plan["wall_seconds"]) is int and 30 <= plan["wall_seconds"] <= 7200,
        "bounded wall time required",
    )
    _need(
        type(plan["request_seconds"]) is int and 1 <= plan["request_seconds"] <= 180,
        "bounded per-call time required",
    )
    _need(
        type(plan["gpu_uuid"]) is str
        and re.fullmatch(
            r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", plan["gpu_uuid"]
        ),
        "explicit GPU UUID required",
    )
    pins = plan["source_pins"]
    _need(
        type(pins) is dict
        and bool(pins)
        and all(
            child._ordinary_absolute(path)
            and type(digest) is str
            and re.fullmatch(r"[0-9a-f]{64}", digest)
            for path, digest in pins.items()
        ),
        "explicit source SHA inventory required",
    )
    _need(
        type(plan["deltas"]) is dict and set(plan["deltas"]) == {"operation", "click_target"},
        "both candidate deltas required",
    )
    for ref in plan["deltas"].values():
        _need(
            type(ref) is dict
            and set(ref) == {"path", "file_sha256", "fp32_sha256"}
            and child._ordinary_absolute(ref["path"])
            and all(
                type(ref[key]) is str and re.fullmatch(r"[0-9a-f]{64}", ref[key])
                for key in ("file_sha256", "fp32_sha256")
            ),
            "two pinned deltas required",
        )
    transport = plan["transport"]
    _need(
        type(transport) is dict
        and set(transport) == {"lease_release", "lock", "journal", "nonce", "stat"}
        and all(
            child._ordinary_absolute(transport[key]) for key in ("lease_release", "lock", "journal")
        )
        and type(transport["nonce"]) is str
        and re.fullmatch(r"[0-9a-f]{32}", transport["nonce"])
        and type(transport["stat"]) is dict
        and set(transport["stat"]) == {"lock", "journal"}
        and all(
            type(row) is dict
            and set(row) == {"device", "inode"}
            and all(type(v) is int and v >= 0 for v in row.values())
            for row in transport["stat"].values()
        ),
        "reviewed same-inode lease required",
    )
    workspace = Path(__file__).absolute().parent.parent
    _need(
        Path(plan["overlay"]) == workspace / "artifacts/jev-runtime/overlay",
        "fixed overlay required",
    )


def _load_plan(path, digest):
    _need(
        type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest), "explicit plan SHA required"
    )
    raw = child._read_bounded(path, 1024**2)
    _need(child._sha(raw) == digest, "exact plan digest required")
    plan = common._decode(raw)
    _validate_plan(plan)
    return plan


def _actual_sources(plan, backend):
    import jev_decision3_runtime_integrity as features
    import run_jevbench_public as heaven

    from s1.backends import normalize_response
    from s1.contracts import DecisionRequest, validate_distribution
    from s1.evaluation.decision3 import native_payload, normalize_native_response

    files = [
        __file__,
        child.__file__,
        common.__file__,
        features.__file__,
        heaven.__file__,
        heaven.CONFIG,
        inspect.getfile(DecisionRequest),
        inspect.getfile(normalize_response),
        inspect.getfile(validate_distribution),
        inspect.getfile(native_payload),
        inspect.getfile(normalize_native_response),
        inspect.getfile(type(backend)),
        inspect.getfile(type(backend.model)),
        inspect.getfile(type(backend.runtime)),
    ]
    _need(
        all(str(Path(path).absolute()) in plan["source_pins"] for path in files),
        "actual owner, scorer bridge, native and model sources must be pinned",
    )
    _need(
        plan["source_pins"].get(str(heaven.CONFIG.absolute())) == CONFIG_SHA256,
        "fixed official public231 configuration required",
    )
    common._verify_source_pins(plan["source_pins"])


class EvaluationAbort(BaseException):
    """Stop the official runner after a model failure rather than scoring a latch."""


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_publish(path, raw):
    """Durably publish complete bytes with exclusive, same-directory link."""
    child._publish_control(path, raw)


def _quarantine_scorer_output(scorer_output):
    """Label the original, unmodified scorer trace as unaccepted evidence."""
    if not scorer_output.exists():
        return
    _atomic_publish(
        scorer_output / "STATUS_NOT_ACCEPTED.json",
        child._encode(
            {
                "status": "not_accepted_pending_external_audit",
                "score_accepted": False,
                "scope": "entire original scorer directory; original manifest status is not owner acceptance",
            }
        ),
    )
    _fsync_directory(scorer_output)


def _runtime_guard(backend, torch):
    import jev_decision3_runtime_integrity as features

    runtime = backend.runtime
    features._validate_provider_spec(backend.spec)
    parameters = list(runtime.backbone.parameters())
    _need(
        getattr(backend.model, "runtime", None) is runtime
        and runtime.model_name == "d3-edge"
        and runtime.batch_size == 8
        and runtime.temperature == 1.0
        and runtime.max_length is None
        and runtime.attention_mode == "noncausal_full_attention"
        and runtime.permutation_average is False
        and runtime.fast is None
        and runtime.readout_dtype == "float32"
        and runtime.encoder.language_model is runtime.backbone.language_model
        and runtime.backbone.training is False
        and torch.is_tensor(runtime.readout)
        and tuple(runtime.readout.shape) == (255, 1024)
        and runtime.readout.dtype == torch.float32
        and str(runtime.readout.device) == "cuda:0"
        and runtime.readout.requires_grad is False
        and parameters
        and all(
            str(p.device) == "cuda:0" and p.dtype == torch.bfloat16 and p.requires_grad is False
            for p in parameters
        )
        and sum(p.numel() for p in parameters) + runtime.readout.numel() == 594582720
        and features._tensor_sha(runtime.readout) == features.W0_SHA256
        and torch.get_float32_matmul_precision() == "highest"
        and torch.backends.cuda.matmul.allow_tf32 is False
        and not torch.is_autocast_enabled("cuda"),
        "pinned native runtime, W0 and numerical policy required",
    )


def _model_stat_snapshot(runtime, cache_root):
    """Bind checked model-file metadata between full pre/post SHA checks."""
    root = Path(runtime.root)
    cache = Path(cache_root).resolve(strict=True)
    _need(
        root.is_absolute() and root.resolve(strict=True).is_relative_to(cache),
        "selected model cache required",
    )
    inventory = runtime.manifest.get("files_sha256")
    _need(type(inventory) is dict and bool(inventory), "complete publisher inventory required")
    names = ["MODEL_MANIFEST.json"]
    for name in sorted(inventory):
        if (
            name in ("LICENSE", "NOTICE", "requirements.txt", ".gitattributes")
            or name.endswith(".md")
            or name.startswith(("assets/", "LICENSES/"))
        ):
            continue
        relative = Path(name)
        _need(
            not relative.is_absolute() and ".." not in relative.parts,
            "safe model inventory required",
        )
        names.append(name)
    snapshot = []
    for name in names:
        path = (root / name).resolve(strict=True)
        _need(path.is_relative_to(cache), "model file escapes selected cache")
        info = path.stat(follow_symlinks=False)
        _need(stat.S_ISREG(info.st_mode), "regular model file required")
        snapshot.append(
            (
                name,
                str(path),
                info.st_dev,
                info.st_ino,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
            )
        )
    return tuple(snapshot)


def _run_heaven(plan, backend, torch, generic, lease_module, output, check):
    _gate()
    import jev_decision3_runtime_integrity as features
    import run_jevbench_public as heaven

    transport = plan["transport"]
    common._shared_identity(transport)
    features._validate_manifest(backend.runtime, plan["cache_root"])
    model_snapshot = _model_stat_snapshot(backend.runtime, plan["cache_root"])
    lease = lease_module.ForwardLease(
        transport["lock"], transport["journal"], transport["stat"], nonce=transport["nonce"]
    )
    deltas = common._load_deltas(plan, torch)
    events = output / "generic-events.jsonl"
    common._write_new(events, b"")
    event_counts = {
        "planned": 0,
        "native_preflight": 0,
        "native_raw_returned": 0,
        "prepared_not_delivered": 0,
        "failed": 0,
    }
    prepared = []
    stage = [0]
    expected_events = (
        "planned",
        "native_preflight",
        "native_raw_returned",
        "prepared_not_delivered",
    )

    def identity_guard(actual, actual_lease):
        _need(
            actual is backend and actual_lease is lease, "owned native provider and lease required"
        )
        check()
        common._shared_identity(transport)
        common._verify_source_pins(plan["source_pins"])
        _need(
            _model_stat_snapshot(backend.runtime, plan["cache_root"]) == model_snapshot,
            "model file identity changed during evaluation",
        )
        _runtime_guard(backend, torch)
        return {
            "unit": plan["unit"],
            "nonce": transport["nonce"],
            "gpu_uuid": plan["gpu_uuid"],
            "delta_sha256": {key: value["fp32_sha256"] for key, value in plan["deltas"].items()},
        }

    def sink(row):
        _need(row.get("event") in event_counts, "known generic event required")
        allowed_calls = {len(prepared) + 1}
        if row["event"] == "failed" and stage[0] == 0 and prepared:
            allowed_calls.add(len(prepared))
        _need(
            row.get("route") == "jevbench-generic-native-v1"
            and row.get("call") in allowed_calls
            and row.get("delta_applied") == []
            and row.get("delta_bypassed") == ["operation", "click_target"],
            "exact generic route, call and bypass evidence required",
        )
        if row["event"] != "failed":
            _need(row["event"] == expected_events[stage[0]], "ordered generic events required")
            if stage[0] == 0:
                _need(
                    row.get("request_sha256") == child._sha(child._encode(row.get("request"))),
                    "exact original request evidence required",
                )
                stage.append(row["request_sha256"])
            else:
                _need(row.get("request_sha256") == stage[1], "same original request required")
            if stage[0] == 3:
                _need(
                    row.get("response_sha256") == child._sha(child._encode(row.get("response"))),
                    "exact prepared response evidence required",
                )
                prepared.append({"request": stage[1], "response": row["response_sha256"]})
                stage[:] = [0]
            else:
                stage[0] += 1
        child._append(events, row)
        event_counts[row["event"]] += 1

    def deadline_factory():
        started = time.monotonic()

        def deadline():
            check()
            _need(
                time.monotonic() - started < plan["request_seconds"],
                "cooperative request deadline exceeded",
            )

        return deadline

    adapter = generic.GenericJevBenchAdapter(
        backend,
        lease_module=lease_module,
        lease=lease,
        torch=torch,
        deadline_factory=deadline_factory,
        identity_guard=identity_guard,
        event_sink=sink,
    )
    attempted = []

    def predict(state, questions):
        try:
            result = adapter.predict(state, questions)
            _need(
                prepared
                and prepared[-1]
                == {
                    "request": child._sha(child._encode({"state": state, "questions": questions})),
                    "response": child._sha(child._encode(result)),
                },
                "same-call request and prepared response required",
            )
            attempted.append(
                {
                    "request": heaven._hash(
                        heaven._bytes({"state": state, "questions": questions})
                    ),
                    "response": heaven._hash(heaven._bytes(result)),
                }
            )
            return result
        except BaseException as error:
            raise EvaluationAbort("generic call failed; entire evaluation invalid") from error

    identity = {
        "model_id": backend.spec.model_id,
        "revision": backend.spec.revision,
        "route": "generic_native_delta_bypassed",
        "delta_sha256": {key: value["fp32_sha256"] for key, value in plan["deltas"].items()},
        "cost_usd": None,
    }
    staging = output / "staging"
    staging.mkdir(mode=0o700)
    _atomic_publish(
        staging / "status.json",
        child._encode(
            {
                "status": "not_accepted_pending_external_audit",
                "score_accepted": False,
                "scope": "all staging files including original scorer output",
            }
        ),
    )
    _fsync_directory(staging)
    _fsync_directory(output)
    scorer_output = staging / "public231"
    primary = None
    try:
        summary = heaven.run_public(predict, plan["tasks_root"], scorer_output, identity)
        check()
        _need(
            len(attempted) == 231
            and event_counts
            == {
                "planned": 231,
                "native_preflight": 231,
                "native_raw_returned": 231,
                "prepared_not_delivered": 231,
                "failed": 0,
            },
            "all 231 generic calls and evidence required",
        )
        manifest = common._decode(
            child._read_bounded(scorer_output / "manifest.json", 16 * 1024**2)
        )
        records_path = scorer_output / "records.jsonl"
        records = [
            common._decode(line)
            for line in child._read_bounded(records_path, 64 * 1024**2).splitlines()
        ]
        _need(
            manifest.get("status") == "completed"
            and manifest.get("n_attempted") == 231
            and len(records) == 231
            and all(row.get("ok") is True and row.get("valid") is True for row in records),
            "official full scorer completion required",
        )
        tasks, _, _ = heaven.load_public(plan["tasks_root"])
        _need(
            [row.get("task_id") for row in records] == [task.id for task in tasks],
            "original ordered official task IDs required",
        )
        for row, observed in zip(records, attempted, strict=True):
            raw_name = row.get("raw_path")
            _need(
                type(raw_name) is str
                and re.fullmatch(r"raw/[0-9]{4}-[0-9a-f]{64}\.json", raw_name),
                "official raw response path required",
            )
            raw_bytes = child._read_bounded(scorer_output / raw_name, 16 * 1024**2)
            raw = common._decode(raw_bytes)
            _need(
                row.get("raw_sha256") == child._sha(raw_bytes)
                and row.get("request_sha256")
                == observed["request"]
                == heaven._hash(heaven._bytes(raw.get("request")))
                and observed["response"] == heaven._hash(heaven._bytes(raw.get("response"))),
                "official raw callback request and returned response must match",
            )
        _need(
            lease.fd is None and lease.held is None and lease.fatal is None,
            "lease closure required",
        )
        lease_result = child._lease_summary(transport, os.getpid(), 231)
        features._validate_manifest(backend.runtime, plan["cache_root"])
        _need(
            _model_stat_snapshot(backend.runtime, plan["cache_root"]) == model_snapshot,
            "model file identity changed after evaluation",
        )
        _need(
            all(
                common._sha(tensor.detach().contiguous().numpy().tobytes())
                == plan["deltas"][head]["fp32_sha256"]
                for head, tensor in deltas.items()
            ),
            "candidate delta CPU bytes changed",
        )
        identity_guard(backend, lease)
        return {
            "status": "completed_pending_external_audit",
            "official_summary": summary,
            "calls": 231,
            "event_counts": event_counts,
            "lease": lease_result,
            "delta_route": "both_bypassed",
            "cost_usd": None,
            "cost_basis": "unknown_local_compute",
            "latency_scope": "full callback wall time including file metadata checks, source guard, evidence fsync and CUDA fence; not model-forward-only",
            "model_forward_latency": None,
            "score_accepted": False,
        }
    except BaseException as error:
        primary = error
        if lease.fd is not None or lease.held is not None or lease.fatal is not None:
            error.owner_cleanup_unconfirmed = True
        raise
    finally:
        deltas.clear()
        try:
            _quarantine_scorer_output(scorer_output)
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note("scorer quarantine also failed: " + repr(error))


def main(argv=None):
    _gate()
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args(argv)
    return run(args.plan, args.plan_sha256)


def run(plan_path, plan_sha):
    _gate()
    plan = _load_plan(plan_path, plan_sha)
    output = Path(plan["output_dir"])
    _need(
        output.is_dir() and output.resolve() == output and output.stat().st_mode & 0o077 == 0,
        "ordinary private output directory required",
    )
    _need(not any(output.iterdir()), "fresh empty private output required")
    common._own_cgroup(plan["unit"])
    invocation = os.environ.get("INVOCATION_ID", "")
    _need(re.fullmatch(r"[0-9a-f]{32}", invocation), "systemd invocation required")
    common._verify_source_pins(plan["source_pins"])
    for path in (__file__, child.__file__, common.__file__):
        _need(
            str(Path(path).absolute()) in plan["source_pins"],
            "worker and shared owner source pins required",
        )
    common._validate_interpreter(plan)
    hello = {
        "schema": "decision3-jevbench-heaven-hello-v1",
        "pid": os.getpid(),
        "startticks": common.systemd.process_identity(os.getpid())["startticks"],
        "boot_id": common.systemd.boot_id(),
        "unit": plan["unit"],
        "invocation": invocation,
        "plan_sha256": plan_sha,
    }
    _atomic_publish(output / "hello.json", child._encode(hello))
    started = time.monotonic()
    while not (output / "go.json").exists():
        _need(time.monotonic() - started < 20, "parent handshake timed out")
        time.sleep(0.1)
    child._control(
        output / "go.json",
        schema="decision3-jevbench-heaven-go-v1",
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
            time.monotonic() - started < plan["wall_seconds"],
            "cooperative evaluation deadline exceeded",
        )
        common._own_cgroup(plan["unit"])
        _need(os.environ.get("INVOCATION_ID") == invocation, "invocation changed")

    backend = None
    primary = None
    status = {
        "schema": "decision3-jevbench-heaven-terminal-v1",
        "status": "running",
        "invocation": invocation,
        "plan_sha256": plan_sha,
        "model_score": None,
        "host_supervision": "external",
        "official_rank": None,
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
            "one selected GPU required",
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
        generic = common._load_reviewed_release(
            plan["generic_release"],
            "jev_decision3_jevbench_generic.py",
            plan["source_pins"],
            "_jevbench_generic_release",
        )
        lease_module = common._load_reviewed_release(
            plan["transport"]["lease_release"],
            "jev_gpu_forward_lease.py",
            plan["source_pins"],
            "_jevbench_lease_release",
        )
        backend = Decision3Backend(decision3_spec("d3-edge", device="cuda:0"))
        _actual_sources(plan, backend)
        status["evaluation"] = _run_heaven(
            plan, backend, torch, generic, lease_module, output, check
        )
        status["status"] = "completed_pending_external_audit"
    except BaseException as error:
        primary = error
        status.update(
            status="failed_no_score",
            error_type=type(error).__name__,
            error_stack=common._drop_exception_frames(error),
        )
    finally:
        if primary is not None and getattr(primary, "owner_cleanup_unconfirmed", False):
            status.update(
                status="failed_cleanup_unconfirmed",
                cleanup_skipped="lease closure unconfirmed; external owner must terminate this invocation",
            )
            _atomic_publish(output / "terminal.json", child._encode(status))
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
        if status["status"] != "completed_pending_external_audit":
            status.pop("evaluation", None)
        _atomic_publish(output / "terminal.json", child._encode(status))
    if primary is not None:
        raise primary
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
