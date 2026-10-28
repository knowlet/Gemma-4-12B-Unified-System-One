"""Fixed development-only ABBA control of whole-call MPS cache clearing.

All twelve native full-independent predictions are retained. This changes only
experimental cleanup placement; no labels, KV reuse, fitting or defaults change.
Torch and model imports follow captured predecessor and actual-child termination.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.abc
import importlib.metadata
import importlib.util
import json
import os
import platform
import shlex
import shutil
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("torch", "transformers", "numpy", "torchvision")
TASK = "dev-design-free"
DECISIONS = (1, 2, 3)
BLOCKS = ("A", "B", "B", "A")
MAX_PROBABILITY_DELTA = 0.01
MAX_DRIVER_FRACTION = 0.8
HELPERS = (
    "scripts/diagnose_mps_prefix_shapes.py",
    "scripts/resume_mps_benchmarks.py",
    "scripts/run_ultrafast.py",
    "scripts/benchmark_mps_maintenance.py",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def package_versions():
    result = {}
    for name in PACKAGES:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def validate_environment(report):
    actual = package_versions()
    if actual != report["environment"]["packages"]:
        raise ValueError("package environment changed while queued; refusing runtime import/load")
    report["actual_package_versions_after_wait"] = actual
    return actual


def error(exc):
    return {"type": type(exc).__name__, "message": str(exc)}


def persistence_fault(report, exc, scope):
    report.setdefault("persistence_errors", []).append({"scope": scope, **error(exc)})
    try:
        sys.stderr.write(
            json.dumps(
                {
                    "primary_error": report.get("error", report.get("fault")),
                    "persistence_errors": report["persistence_errors"],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    except BaseException:
        pass  # No logging failure may replace the first operational exception.


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_hash(value):
    # Object ordering is part of the native state/question prompt.
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def capture(value):
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False)), "json"
    except (TypeError, ValueError, OverflowError):
        return repr(value), "python_repr_not_json"


def load_verified(path, expected, name):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("frozen Python source changed")
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


class FrozenS1(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Compile only archived, hashed package bytes; never consult cached pyc."""

    def __init__(self, output, hashes):
        self.root, self.hashes = Path(output) / "sources", hashes

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "s1" and not fullname.startswith("s1."):
            return None
        base = "src/" + fullname.replace(".", "/")
        name = base + "/__init__.py" if base + "/__init__.py" in self.hashes else base + ".py"
        if name not in self.hashes:
            raise ImportError("S1 module is absent from frozen source inventory")
        spec = importlib.util.spec_from_loader(
            fullname, self, is_package=name.endswith("/__init__.py")
        )
        spec.loader_state = name
        return spec

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        name = module.__spec__.loader_state
        path = self.root / name
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.hashes[name]:
            raise ValueError("frozen S1 source changed before import")
        module.__file__ = str(path)
        if module.__spec__.submodule_search_locations is not None:
            module.__path__ = [str(path.parent)]
        exec(compile(raw, str(path), "exec"), module.__dict__)


def freeze(args, report):
    paths = [*ROOT.glob("src/s1/**/*.py"), *(ROOT / name for name in HELPERS)]
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError("execution sources must be regular files")
        target = args.output / "sources" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    report["source_sha256"] = {
        str(path.relative_to(args.output / "sources")): digest(path)
        for path in sorted((args.output / "sources").rglob("*.py"))
    }
    folder = args.trace_root / "independent" / f"{TASK}-1"
    for name in ("task.json", "decision-transport.json"):
        target = args.output / "inputs/trace/independent" / f"{TASK}-1" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(folder / name, target)
    target = args.output / "inputs/checkpoint-manifest.json"
    shutil.copyfile(args.model / "artifact-manifest.json", target)
    report["input_sha256"] = {
        str(path.relative_to(args.output)): digest(path)
        for path in sorted((args.output / "inputs").rglob("*"))
        if path.is_file()
    }
    frozen_folder = args.output / "inputs/trace/independent" / f"{TASK}-1"
    task = json.loads((frozen_folder / "task.json").read_text())
    if task.get("id") != TASK or task.get("split") != "development":
        raise ValueError("only the fixed original development task is allowed")
    report["inputs"] = {
        "task": TASK,
        "decisions": list(DECISIONS),
        "labels_or_outcomes_used": False,
    }


def validate_frozen(args, report):
    actual = {
        str(path.relative_to(args.output / "sources")): digest(path)
        for path in sorted((args.output / "sources").rglob("*.py"))
    }
    if actual != report["source_sha256"]:
        raise ValueError("frozen source inventory changed")
    if any(digest(args.output / name) != value for name, value in report["input_sha256"].items()):
        raise ValueError("frozen inputs changed")
    if (
        digest(args.model / "artifact-manifest.json")
        != report["input_sha256"]["inputs/checkpoint-manifest.json"]
    ):
        raise ValueError("checkpoint manifest changed while queued")


def helper(args, report, name):
    return load_verified(
        args.output / "sources" / name,
        report["source_sha256"][name],
        "_maintenance_" + Path(name).stem,
    )


def wait_predecessors(args, report, save):
    observer = helper(args, report, "scripts/resume_mps_benchmarks.py")
    if any(type(pid) is not int or pid <= 0 for pid in args.wait_for_pid) or len(
        set(args.wait_for_pid)
    ) != len(args.wait_for_pid):
        raise ValueError("wait PIDs must be unique positive integers")
    paths = [Path(path).resolve() for path in args.wait_for_supervisor_receipt]
    if len(set(paths)) != len(paths):
        raise ValueError("supervisor receipt paths must be unique")
    supervisors, pids = [], list(args.wait_for_pid)
    for index, path in enumerate(paths):
        raw = path.read_bytes()
        source = json.loads(raw)
        pid, created = source.get("pid"), source.get("created_at_utc")
        if type(pid) is not int or pid <= 0 or not isinstance(created, str) or not created:
            raise ValueError("epoch supervisor receipt lacks pid/creation identity")
        if any(row["pid"] == pid for row in supervisors):
            raise ValueError("multiple receipts claim the same supervisor")
        initial = args.output / "supervisors" / f"{index + 1}-initial.json"
        initial.parent.mkdir(parents=True, exist_ok=True)
        initial.write_bytes(raw)
        initial_children = {}
        for child in [*source.get("epochs", []), source.get("active_child")]:
            if child is None or child.get("pid") is None:
                continue
            child_pid = child["pid"]
            if type(child_pid) is not int or child_pid <= 0:
                raise ValueError("initial supervisor child PID is invalid")
            observer.stable_identity(child.get("identity"))
            if child_pid in initial_children and observer.stable_identity(
                initial_children[child_pid]
            ) != observer.stable_identity(child["identity"]):
                raise ValueError("initial supervisor child identity is inconsistent")
            initial_children[child_pid] = child["identity"]
        supervisors.append(
            {
                "path": str(path),
                "pid": pid,
                "created_at_utc": created,
                "initial_sha256": hashlib.sha256(raw).hexdigest(),
                "initial_snapshot": str(initial),
                "initial_children": initial_children,
            }
        )
        if pid not in pids:
            pids.append(pid)
    # Every original identity is captured before waiting on any predecessor.
    report["predecessors"] = [observer.capture_handle(pid) for pid in pids]
    report["supervisors"] = supervisors
    for row in supervisors:
        row["captured_supervisor"] = next(
            handle for handle in report["predecessors"] if handle["pid"] == row["pid"]
        )
        identity = row["captured_supervisor"]["identity"]
        if identity is not None:
            command = observer.stable_identity(identity)[1]
            words = shlex.split(command)
            outputs = [word.split("=", 1)[1] for word in words if word.startswith("--output=")]
            outputs.extend(
                words[index + 1] for index, word in enumerate(words[:-1]) if word == "--output"
            )
            if len(outputs) != 1 or Path(outputs[0]).resolve() != Path(row["path"]).parent:
                raise ValueError("captured supervisor command does not own this output receipt")
            row["captured_command_output_bound"] = True
        else:
            row["captured_command_output_bound"] = False
            row["identity_limit"] = (
                "Supervisor already absent at capture; terminal receipt PID/creation binding and actual recorded child handles still required."
            )
    report["status"] = "waiting"
    save()
    for handle in report["predecessors"]:
        observer.wait_handle(handle, save, args.poll_seconds)
    for index, row in enumerate(supervisors):
        source = json.loads(Path(row["path"]).read_text())
        if source.get("pid") != row["pid"] or source.get("created_at_utc") != row["created_at_utc"]:
            raise ValueError("supervisor receipt no longer belongs to the captured original")
        if (
            source.get("status") not in {"completed", "error", "epoch_limit_reached"}
            or source.get("active_child", "missing") is not None
        ):
            raise ValueError("supervisor lacks a terminal receipt with no active child")
        epochs = source.get("epochs")
        if not isinstance(epochs, list) or not epochs:
            raise ValueError("supervisor has no complete launched-child history")
        row.update(prior_status=source["status"], prior_error=source.get("error"), children=[])
        for epoch in epochs:
            pid = epoch.get("pid")
            if (
                type(pid) is not int
                or pid <= 0
                or type(epoch.get("exit_code")) is not int
                or not epoch.get("finished_at_utc")
            ):
                raise ValueError("launched supervisor epoch lacks terminal child evidence")
            observer.stable_identity(epoch.get("identity"))
            handle = {
                "pid": pid,
                "identity": epoch["identity"],
                "captured_at_utc": now(),
                "exit_code": epoch["exit_code"],
                "receipt_status": epoch.get("status"),
            }
            row["children"].append(handle)
        final_children = {
            child["pid"]: observer.stable_identity(child["identity"]) for child in row["children"]
        }
        if len(final_children) != len(row["children"]) or any(
            final_children.get(pid) != observer.stable_identity(identity)
            for pid, identity in row["initial_children"].items()
        ):
            raise ValueError("supervisor dropped or changed an initially recorded child")
        save()
        for handle in row["children"]:
            observer.wait_handle(handle, save, args.poll_seconds)
        raw = Path(row["path"]).read_bytes()
        if json.loads(raw) != source:
            raise ValueError("terminal supervisor receipt changed during child verification")
        final = args.output / "supervisors" / f"{index + 1}-terminal.json"
        final.write_bytes(raw)
        row.update(
            terminal_sha256=hashlib.sha256(raw).hexdigest(),
            terminal_snapshot=str(final),
            interpretation="Original processes are terminal; predecessor completion and errors are preserved, not reinterpreted as benchmark success.",
        )
    # Re-observe all original handles immediately before allowing runtime import.
    for handle in [
        *report["predecessors"],
        *(child for row in supervisors for child in row["children"]),
    ]:
        observer.wait_handle(handle, save, args.poll_seconds)
    for row in supervisors:
        if digest(row["path"]) != row["terminal_sha256"]:
            raise ValueError("terminal supervisor receipt changed before runtime import")
    report["predecessor_gate"] = (
        "all captured originals and every recorded launched child proven terminal"
    )
    save()


class MaintenanceStop(BaseException):
    def __init__(self, evidence):
        self.evidence = evidence
        super().__init__(evidence["error"]["message"])


class Controller:
    """A first-fault latch; only mandatory cleanup can touch MPS afterward."""

    def __init__(self, synchronize, snapshot, empty_cache, clock=time.perf_counter):
        self.synchronize, self.snapshot, self.empty_cache, self.clock = (
            synchronize,
            snapshot,
            empty_cache,
            clock,
        )
        self.fault = None

    def latch(self, failure, scope, kind="error"):
        evidence = {"error": copy.deepcopy(failure), "scope": scope, "kind": kind}
        if self.fault is None:
            self.fault = evidence
        elif self.fault["kind"] == "resource_guard" and kind == "error":
            evidence["prior_resource_guard"] = copy.deepcopy(self.fault)
            self.fault = evidence

    def ensure(self):
        if self.fault is not None:
            raise MaintenanceStop(copy.deepcopy(self.fault))

    def observe(self):
        observed = self.snapshot()
        values = [
            observed[key]
            for key in (
                "current_allocated_bytes",
                "driver_allocated_bytes",
                "recommended_max_bytes",
            )
        ]
        if any(type(value) is not int or value < 0 for value in values) or values[2] <= 0:
            raise ValueError("complete nonnegative MPS memory observations required")
        return observed

    def guard(self, target, scope):
        self.ensure()
        try:
            self.synchronize()
            observed = self.observe()
            target["memory"] = observed
            values = [
                observed[key]
                for key in (
                    "current_allocated_bytes",
                    "driver_allocated_bytes",
                    "recommended_max_bytes",
                )
            ]
            if values[1] >= MAX_DRIVER_FRACTION * values[2]:
                self.latch(
                    {
                        "type": "DriverMemoryGuard",
                        "message": "driver allocation reached 0.8 of recommended memory; no next inference",
                    },
                    scope,
                    "resource_guard",
                )
                self.ensure()
            target["passed"] = True
        except Exception as exc:
            self.latch(error(exc), scope)
            self.ensure()

    def cleanup(self, scope):
        started = self.clock()
        result = {"scope": scope, "before": None, "after": None, "errors": []}
        # Even a synchronization/observation fault must not skip the final trim.
        for name, operation in (
            ("synchronize_before", self.synchronize),
            ("before", self.observe),
            ("empty_cache", self.empty_cache),
            ("synchronize_after", self.synchronize),
            ("after", self.observe),
        ):
            try:
                value = operation()
                if name in ("before", "after"):
                    result[name] = value
            except BaseException as exc:
                result["errors"].append({"operation": name, **error(exc)})
                self.latch(error(exc), scope)
        result["wall_ms"] = (self.clock() - started) * 1000
        return result


def summarize(requests, samples, normalize):
    expected = [
        (block, policy, decision)
        for block, policy in enumerate(BLOCKS, 1)
        for decision in DECISIONS
    ]
    if [(row["block"], row["policy"], row["decision_index"]) for row in samples] != expected:
        raise ValueError("all twelve chronological samples, including cold attempts, are required")
    if any(row.get("error") or row.get("raw_response_format") != "json" for row in samples):
        raise ValueError("failed/invalid samples cannot be omitted or scored")
    comparisons = []
    for index, request in enumerate(requests, 1):
        rows = [row for row in samples if row["decision_index"] == index]
        normalized = [normalize(request, row["raw_response"]) for row in rows]
        differences, agreements = [], []
        for left in range(len(rows)):
            for right in range(left + 1, len(rows)):
                for question in request.questions:
                    a = normalized[left]["answers"][question.id]["probabilities"]
                    b = normalized[right]["answers"][question.id]["probabilities"]
                    labels = question.labels()
                    differences.extend(abs(a[label] - b[label]) for label in labels)
                    agreements.append(max(labels, key=a.get) == max(labels, key=b.get))
        maximum, agree = max(differences), all(agreements)
        shapes = [[forward["input_shape"] for forward in row["forwards"]] for row in rows]
        same_shapes = all(shape == shapes[0] for shape in shapes)
        comparisons.append(
            {
                "decision_index": index,
                "samples": len(rows),
                "pairwise_comparisons": 6,
                "max_probability_delta": maximum,
                "all_argmax_agree": agree,
                "same_native_forward_shapes": same_shapes,
                "gate_passed": maximum <= MAX_PROBABILITY_DELTA and agree and same_shapes,
            }
        )
    return {
        "requests": comparisons,
        "all_numerical_and_shape_gates_passed": all(row["gate_passed"] for row in comparisons),
        "retained_samples": len(samples),
        "cold_samples_discarded": 0,
    }


def run_schedule(model, requests, operation, normalize, control, report, save):
    report.update(blocks=[], samples=[])
    expected_requests = [content_hash(request.model_dump()) for request in requests]
    started = control.clock()

    def publish():
        try:
            save()
        except BaseException as exc:
            control.latch(error(exc), "receipt_write")
            control.ensure()

    try:
        for index, policy in enumerate(BLOCKS, 1):
            control.ensure()
            block = {"block": index, "policy": policy, "status": "running", "samples": []}
            report["blocks"].append(block)
            block_started = control.clock()
            try:
                block["start_cleanup"] = control.cleanup("block_start")
                control.ensure()
                publish()
                for decision, request in zip(DECISIONS, requests, strict=True):
                    control.ensure()
                    sample = {
                        "block": index,
                        "policy": policy,
                        "decision_index": decision,
                        "request_sha256": content_hash(request.model_dump()),
                        "raw_response": None,
                        "raw_response_format": None,
                        "error": None,
                        "forwards": [],
                        "pre_dispatch_guard": {},
                        "cold_after_clear": policy == "A" or decision == 1,
                    }
                    report["samples"].append(sample)
                    block["samples"].append(len(report["samples"]) - 1)
                    call_started = control.clock()
                    hook = None

                    def before_forward(_module, _args, kwargs):
                        control.ensure()
                        row = {
                            "input_shape": list(kwargs["input_ids"].shape),
                            "use_cache": kwargs.get("use_cache"),
                            "has_past_key_values": kwargs.get("past_key_values") is not None,
                            "guard": {},
                        }
                        sample["forwards"].append(row)
                        try:
                            if row["use_cache"] is not False or row["has_past_key_values"]:
                                raise ValueError(
                                    "control requires unchanged full prompts and no KV cache"
                                )
                            control.guard(row["guard"], "pre_forward")
                        except Exception as exc:
                            control.latch(error(exc), "pre_forward")
                            control.ensure()

                    try:
                        if sample["request_sha256"] != expected_requests[decision - 1]:
                            raise ValueError("native request changed between repeated observations")
                        control.guard(sample["pre_dispatch_guard"], "pre_dispatch")
                        hook = model.backbone.register_forward_pre_hook(
                            before_forward, with_kwargs=True
                        )
                        prediction_started = control.clock()
                        raw = operation(request)
                        sample["raw_response"], sample["raw_response_format"] = capture(raw)
                        control.synchronize()
                        sample["prediction_ms_including_forward_guards"] = (
                            control.clock() - prediction_started
                        ) * 1000
                        normalize(request, raw)  # Validate without replacing raw output.
                        if content_hash(request.model_dump()) != expected_requests[decision - 1]:
                            raise ValueError("native prediction mutated its original request")
                        if not sample["forwards"]:
                            raise ValueError("native prediction produced no observed forward")
                        sample["memory_after_prediction"] = control.observe()
                    except MaintenanceStop:
                        raise
                    except BaseException as exc:
                        control.latch(error(exc), "prediction")
                    finally:
                        if hook is not None:
                            hook.remove()
                        if policy == "A":
                            sample["cleanup"] = control.cleanup("after_call")
                        else:
                            sample["cleanup"] = {
                                "scope": "retained_until_block_end",
                                "errors": [],
                                "wall_ms": 0,
                            }
                        if control.fault is not None:
                            sample["error"] = copy.deepcopy(control.fault)
                        sample["wall_ms_including_call_cleanup"] = (
                            control.clock() - call_started
                        ) * 1000
                        publish()
                    control.ensure()
                block["status"] = "completed"
            finally:
                block["end_cleanup"] = control.cleanup("block_end")
                if control.fault is not None:
                    block.update(status="stopped", error=copy.deepcopy(control.fault))
                block["wall_ms_including_boundary_cleanup_and_trace_io"] = (
                    control.clock() - block_started
                ) * 1000
                publish()
            control.ensure()
        report["summary"] = summarize(requests, report["samples"], normalize)
        report["timing_summary"] = {
            policy: {
                "block_ms": [
                    row["wall_ms_including_boundary_cleanup_and_trace_io"]
                    for row in report["blocks"]
                    if row["policy"] == policy
                ],
                "median_block_ms": statistics.median(
                    row["wall_ms_including_boundary_cleanup_and_trace_io"]
                    for row in report["blocks"]
                    if row["policy"] == policy
                ),
            }
            for policy in ("A", "B")
        }
    finally:
        report["final_cleanup"] = control.cleanup("experiment_final")
        report["schedule_wall_ms_including_all_cleanup_and_trace_io"] = (
            control.clock() - started
        ) * 1000
        report["fault"] = copy.deepcopy(control.fault)
        publish()
        # A real final-cleanup fault supersedes resource-only control flow.
        # The controller still preserves the first real inference/IO error.
        control.ensure()


def load_runtime(args, report):
    versions = validate_environment(report)
    if any(name == "s1" or name.startswith("s1.") for name in sys.modules):
        raise RuntimeError("S1 was imported before archived source installation")
    sys.meta_path.insert(0, FrozenS1(args.output, report["source_sha256"]))
    snapshots = helper(args, report, "scripts/diagnose_mps_prefix_shapes.py")
    runner = helper(args, report, "scripts/run_ultrafast.py")
    requests, provenance = [], []
    for decision in DECISIONS:
        request, record = snapshots.load_development_snapshot(
            args.output / "inputs/trace", TASK, decision
        )
        requests.append(request)
        record["order_preserving_request_sha256"] = content_hash(request.model_dump())
        provenance.append(record)
    report["input_requests"] = [request.model_dump() for request in requests]
    report["input_provenance"] = provenance
    report["checkpoint"] = runner.verify_checkpoint(args.model)
    torch = importlib.import_module("torch")
    imported = {}
    for name, expected in versions.items():
        if expected is not None:
            actual = str(getattr(importlib.import_module(name), "__version__", "unavailable"))
            if actual != expected:
                raise ValueError("imported package differs from queued distribution version")
            imported[name] = actual
    report["actual_imported_package_versions"] = imported
    validate_environment(report)
    return SimpleNamespace(
        torch=torch,
        requests=requests,
        unified=importlib.import_module("s1.unified"),
        gemma=importlib.import_module("s1.evaluation.gemma"),
        backends=importlib.import_module("s1.backends"),
    )


def run_real(args, report, save):
    runtime = load_runtime(args, report)
    torch = runtime.torch
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; no alternative device allowed")
    report["actual_torch_build"] = {
        "version": getattr(torch, "__version__", None),
        "git_version": getattr(getattr(torch, "version", None), "git_version", None),
    }

    def snapshot():
        return {
            "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
            "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
            "recommended_max_bytes": int(torch.mps.recommended_max_memory()),
            "kind": "snapshot_not_peak",
            "device": "mps",
        }

    control = Controller(torch.mps.synchronize, snapshot, torch.mps.empty_cache)
    try:
        started = time.perf_counter()
        model = runtime.unified.UnifiedDecisionModel(
            str(args.model), device="mps", dtype="bfloat16", attn_implementation="sdpa"
        )
        validate_environment(report)
        torch.mps.synchronize()
        report["load_seconds"] = time.perf_counter() - started
        report["model"] = {
            key: getattr(model, key, None)
            for key in (
                "name",
                "revision",
                "device",
                "dtype",
                "temperature",
                "attn_implementation",
                "max_context",
                "quantization",
                "compile_mode",
                "softcap",
            )
        }
        if (
            model.device != "mps"
            or model.dtype != "bfloat16"
            or model.attn_implementation != "sdpa"
            or model.quantization != "none"
            or getattr(model, "compile_mode", None) is not None
        ):
            raise ValueError("control requires unchanged MPS BF16 SDPA dense uncompiled runtime")
        report.update(status="running", memory_after_load=snapshot())
        save()
        run_schedule(
            model,
            runtime.requests,
            lambda request: runtime.gemma.predict_batch(model, [request], independent=True)[0],
            runtime.backends.normalize_response,
            control,
            report,
            save,
        )
    except BaseException as exc:
        if isinstance(exc, MaintenanceStop):
            control.fault = control.fault or copy.deepcopy(exc.evidence)
        else:
            control.latch(error(exc), "model_load_or_schedule")
        raise
    finally:
        if "final_cleanup" not in report:
            primary = sys.exception()
            report["final_cleanup"] = control.cleanup("experiment_final")
            report["fault"] = copy.deepcopy(control.fault)
            try:
                save()
            except BaseException as exc:
                control.latch(error(exc), "fallback_final_receipt_write")
                report["fault"] = copy.deepcopy(control.fault)
                persistence_fault(report, exc, "fallback_final_receipt_write")
                if primary is None or isinstance(primary, MaintenanceStop):
                    control.ensure()


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "initializing",
        "created_at_utc": now(),
        "pid": os.getpid(),
        "scope": "Fixed development-only maintenance control; no task score, production adoption or performance claim.",
        "schedule": [
            {"block": i, "policy": p, "decisions": list(DECISIONS)} for i, p in enumerate(BLOCKS, 1)
        ],
        "policies": {
            "A": "Clear at block start/end and after each complete native prediction.",
            "B": "Clear only at block start/end; retain caches within each three-call block.",
        },
        "native_path": "Full independent native prompts, same checkpoint/BF16/SDPA; no cross-request KV cache.",
        "gate": {
            "maximum_probability_delta": MAX_PROBABILITY_DELTA,
            "all_argmax_agree": True,
            "driver_fraction_before_dispatch_and_each_forward": MAX_DRIVER_FRACTION,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(),
            "mps_environment": {
                name: os.environ.get(name)
                for name in (
                    "PYTORCH_ENABLE_MPS_FALLBACK",
                    "PYTORCH_MPS_FAST_MATH",
                    "PYTORCH_MPS_PREFER_METAL",
                    "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
                    "PYTORCH_MPS_LOW_WATERMARK_RATIO",
                )
            },
        },
        "timing_scope": "Every cold sample and failure retained. Call timing includes pre-dispatch/forward guards and A per-call cleanup; primary block timing includes start/end clears and trace IO. Schedule timing additionally includes final cleanup. Loading is separate.",
        "defaults_changed": False,
        "labels_or_outcomes_used": False,
        "reported_cost_usd": None,
    }

    def save():
        write_json(args.output / "report.json", report)

    save()
    exit_code, primary_exception = 1, None
    try:
        freeze(args, report)
        save()
        wait_predecessors(args, report, save)
        validate_frozen(args, report)
        report["execution_sources"] = (
            "compile_verified_archived_bytes_no_pyc; live source changes never mixed into execution"
        )
        save()
        run_real(args, report, save)
        report["status"] = "completed"
        exit_code = 0 if report["summary"]["all_numerical_and_shape_gates_passed"] else 1
    except MaintenanceStop as stop:
        report.update(
            status="resource_stopped" if stop.evidence["kind"] == "resource_guard" else "error",
            error=stop.evidence,
        )
        exit_code = 2 if stop.evidence["kind"] == "resource_guard" else 1
    except BaseException as exc:
        report.update(status="error", error=error(exc))
        primary_exception = exc
    finally:
        report["finished_at_utc"] = now()
        try:
            save()
        except BaseException as exc:
            if report.get("error", {}).get("kind") == "resource_guard":
                report["prior_resource_guard"] = copy.deepcopy(report["error"])
                report["error"] = error(exc)
            elif not report.get("error"):
                report["error"] = error(exc)
            report["status"] = "error"
            exit_code = 1
            persistence_fault(report, exc, "outer_final_receipt_write")
    if primary_exception is not None:
        raise primary_exception
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/release/checkpoint")
    parser.add_argument("--wait-for-pid", type=int, action="append", default=[])
    parser.add_argument("--wait-for-supervisor-receipt", type=Path, action="append", default=[])
    parser.add_argument("--poll-seconds", type=float, default=15)
    args = parser.parse_args(argv)
    if not 0 < args.poll_seconds <= 60:
        parser.error("poll seconds must be in (0,60]")
    for name in ("trace_root", "output", "model"):
        setattr(args, name, getattr(args, name).resolve())
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
