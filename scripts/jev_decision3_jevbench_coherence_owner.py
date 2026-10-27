"""Held supervised Decision3 generic JevBench coherence mini worker.

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
from collections import Counter
from pathlib import Path

import jev_decision3_candidate_browser_child as child
import jev_decision3_candidate_http_owner as common

SOURCE_ONLY = True
SCHEMA = "decision3-jevbench-coherence-owner-plan-v1"
CONFIG_SHA256 = "ebacca34e3d4193419e43629f73fa5fed6617834415ff87ff6d12a09b6d3904a"


def _need(value, message):
    if not value:
        raise ValueError(message)


def _gate():
    _need(SOURCE_ONLY is False, "held coherence owner requires independent review")


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
        "exact coherence plan required",
    )
    _need(
        type(plan["unit"]) is str
        and re.fullmatch(r"jev-coherence-[0-9a-f]{32}\.service", plan["unit"]),
        "fresh coherence unit required",
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
    import jevbench_public_request_inventory as inventory

    from s1.backends import normalize_response
    from s1.contracts import DecisionRequest, validate_distribution
    from s1.evaluation.decision3 import native_payload, normalize_native_response

    files = [
        __file__,
        child.__file__,
        common.__file__,
        features.__file__,
        inventory.__file__,
        inventory.COHERENCE_CONFIG,
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
        "actual worker, publisher and official inventory sources must be pinned",
    )
    _need(
        plan["source_pins"].get(str(inventory.COHERENCE_CONFIG.absolute())) == CONFIG_SHA256,
        "fixed official coherence source inventory required",
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


def _run_coherence(plan, backend, torch, generic, lease_module, output, check):
    """Run the pinned official mini plan once, with a fresh in-memory SDK cache."""
    _gate()
    import jev_decision3_runtime_integrity as features
    import jevbench_public_request_inventory as inventory

    config_raw = inventory.COHERENCE_CONFIG.read_bytes()
    _need(child._sha(config_raw) == CONFIG_SHA256, "fixed official coherence inventory required")
    config = common._decode(config_raw)
    finder = inventory._verified_coherence(plan["tasks_root"], config)
    sys.meta_path.insert(0, finder)
    primary = None
    deltas = {}
    scorer_output = output / "staging" / "coherence-mini"
    try:
        from jevbench import evaluate
        from jevbench.backends import CallableBackend
        from jevbench.plan import decode_requests
        from jevbench.schema import validate_question
        from jevbench.suites import load_suite

        suite = load_suite("jevbench-mini")
        manifest_path = Path(plan["tasks_root"]) / "src/jevbench/suites/jevbench-mini/manifest.json"
        manifest = common._decode(manifest_path.read_bytes())
        counts = inventory.coherence_inventory(suite, manifest, decode_requests, validate_question)
        _need(
            len(suite.cases) == 240
            and len(suite.plan) == 1200
            and counts["case_question_unique_requests"] == 1248
            and counts["state_question_unique_requests"] == 1248
            and counts["summary"]["totals"].get("contract_accepted") == 1248
            and counts["summary"]["totals"].get("decision3_static_accepted") == 1248,
            "complete official mini population and native input compatibility required",
        )
        cases = {case.id: case for case in suite.cases}
        requests = [(case.state, case.questions) for case in suite.cases]
        trial_ids = set()
        expected_outcomes = Counter()
        for rows in suite.plan.by.values():
            for row in rows:
                case = cases[row["case"]]
                requests.extend(
                    (case.state, q) for q in decode_requests(case, row["requests"]).values()
                )
                suffix = f":v{row['v']}" if row["v"] else ""
                trial_id = f"{row['relation']}:{row['case']}:{row['key']}:{row['t']}" + suffix
                trial_ids.add(trial_id)
                # The three pinned SDK relations below emit one row per target;
                # every other scored mini relation emits one row per trial.
                trial = suite.plan.trial(row, case)
                if row["relation"] in {"batch_order", "batch_size_sweep"}:
                    target_sets = [[target] for target in trial.targets]
                elif row["relation"] == "key_instruction_conflict":
                    target_sets = [[target] for target in trial.params["keys"]]
                else:
                    target_sets = [trial.targets]
                for targets in target_sets:
                    expected_outcomes[(trial_id, row["relation"], row["case"], tuple(targets))] += 1
        expected_requests = {
            child._sha(child._encode({"state": s, "questions": q})) for s, q in requests
        }
        _need(
            len(expected_requests) == 1248 and len(trial_ids) == 1200,
            "all distinct official request and trial identities required",
        )
        features._validate_manifest(backend.runtime, plan["cache_root"])
        model_snapshot = _model_stat_snapshot(backend.runtime, plan["cache_root"])
        transport = plan["transport"]
        common._shared_identity(transport)
        lease = lease_module.ForwardLease(
            transport["lock"], transport["journal"], transport["stat"], nonce=transport["nonce"]
        )
        deltas = common._load_deltas(plan, torch)
        events_path = output / "generic-events.jsonl"
        common._write_new(events_path, b"")
        timing_path = output / "callback-timing.jsonl"
        common._write_new(timing_path, b"")
        event_counts = {
            name: 0
            for name in (
                "planned",
                "native_preflight",
                "native_raw_returned",
                "prepared_not_delivered",
                "failed",
            )
        }
        stages = ("planned", "native_preflight", "native_raw_returned", "prepared_not_delivered")
        state = {"stage": 0, "request": None}
        prepared = []
        actual_requests = set()

        def identity_guard(actual, actual_lease):
            _need(
                actual is backend and actual_lease is lease,
                "owned native provider and lease required",
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
                "delta_sha256": {k: v["fp32_sha256"] for k, v in plan["deltas"].items()},
            }

        def sink(row):
            event = row.get("event")
            _need(
                event in event_counts
                and row.get("route") == "jevbench-generic-native-v1"
                and row.get("call") == len(prepared) + 1
                and row.get("delta_applied") == []
                and row.get("delta_bypassed") == ["operation", "click_target"],
                "original generic route and complete delta bypass required",
            )
            if event != "failed":
                _need(event == stages[state["stage"]], "ordered generic events required")
                if state["stage"] == 0:
                    _need(
                        row.get("request_sha256") == child._sha(child._encode(row.get("request"))),
                        "exact original request evidence required",
                    )
                    state["request"] = row["request_sha256"]
                else:
                    _need(row.get("request_sha256") == state["request"], "same request required")
                if state["stage"] == 3:
                    _need(
                        row.get("response_sha256")
                        == child._sha(child._encode(row.get("response"))),
                        "exact prepared response evidence required",
                    )
                    prepared.append((state["request"], row["response_sha256"]))
                    state.update(stage=0, request=None)
                else:
                    state["stage"] += 1
            child._append(events_path, row)
            event_counts[event] += 1

        def deadline_factory():
            started = time.monotonic()

            def deadline():
                check()
                _need(
                    time.monotonic() - started < plan["request_seconds"],
                    "request deadline exceeded",
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

        def predict(state_value, questions):
            started_ns = time.monotonic_ns()
            try:
                request_sha = child._sha(
                    child._encode({"state": state_value, "questions": questions})
                )
                _need(
                    request_sha in expected_requests and request_sha not in actual_requests,
                    "official unique request without prior cache required",
                )
                result = adapter.predict(state_value, questions)
                _need(
                    prepared and prepared[-1] == (request_sha, child._sha(child._encode(result))),
                    "same-call original request and prepared response required",
                )
                completed_ns = time.monotonic_ns()
                _need(completed_ns >= started_ns, "monotonic callback interval required")
                child._append(
                    timing_path,
                    {
                        "call": len(prepared),
                        "request_sha256": request_sha,
                        "started_monotonic_ns": started_ns,
                        "completed_monotonic_ns": completed_ns,
                    },
                )
                actual_requests.add(request_sha)
                return result["answers"]
            except BaseException as error:
                raise EvaluationAbort(
                    "generic coherence call failed; entire evaluation invalid"
                ) from error

        staging = output / "staging"
        staging.mkdir(mode=0o700)
        _atomic_publish(
            staging / "status.json",
            child._encode(
                {
                    "status": "not_accepted_pending_external_audit",
                    "score_accepted": False,
                    "scope": "all staging files including original official report",
                }
            ),
        )
        _fsync_directory(staging)
        _fsync_directory(output)
        scorer_output.mkdir(mode=0o700)
        report = evaluate(
            CallableBackend(predict, name="d3-edge-native-generic"),
            suite=suite,
            cache=":memory:",
            concurrency=1,
            seed=0,
            repeats=1,
            trials=1,
            minimize=False,
            progress=False,
        )
        check()
        result = report.result
        outcome_ids = {row["trial_id"] for row in result["outcomes"]}
        actual_outcomes = Counter(
            (row["trial_id"], row["relation"], row["case_id"], tuple(row["targets"]))
            for row in result["outcomes"]
        )
        _need(
            result["requests"] == len(actual_requests) == len(prepared) == 1248
            and actual_requests == expected_requests
            and result["cache_hits"] == 193
            and len(result["bases"]) == 240
            and sum(result["planned_tests"].values()) == 1200
            and outcome_ids == trial_ids
            and len(outcome_ids) == 1200
            and actual_outcomes == expected_outcomes
            and not result["errors"]
            and not result["truncated_by_budget"]
            and not result["backend_down"]
            and event_counts
            == {
                "planned": 1248,
                "native_preflight": 1248,
                "native_raw_returned": 1248,
                "prepared_not_delivered": 1248,
                "failed": 0,
            }
            and report.overall is not None,
            "complete official mini scorer and all 1200 trial outcomes required",
        )
        timing_raw = child._read_bounded(timing_path, 4 * 1024**2)
        timing_rows = [common._decode(line) for line in timing_raw.splitlines()]
        _need(len(timing_rows) == 1248, "complete callback timing evidence required")
        previous_end = None
        for index, row in enumerate(timing_rows):
            started_ns = row.get("started_monotonic_ns")
            completed_ns = row.get("completed_monotonic_ns")
            _need(
                type(row.get("call")) is int
                and row["call"] == index + 1
                and row.get("request_sha256") == prepared[index][0]
                and type(started_ns) is int
                and type(completed_ns) is int
                and 0 <= started_ns <= completed_ns
                and (previous_end is None or previous_end <= started_ns),
                "ordered same-call monotonic timing evidence required",
            )
            previous_end = completed_ns
        _need(
            lease.fd is None and lease.held is None and lease.fatal is None,
            "lease closure required",
        )
        lease_result = child._lease_summary(transport, os.getpid(), 1248)
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
        inventory._check_coherence_files(plan["tasks_root"], config)
        # The original SDK report is retained only in the pre-labelled private staging directory.
        report_path = scorer_output / "report.json"
        report.save(report_path)
        os.chmod(report_path, 0o600)
        report_fd = os.open(report_path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            os.fsync(report_fd)
        finally:
            os.close(report_fd)
        _fsync_directory(scorer_output)
        return {
            "status": "completed_pending_external_audit",
            "calls": 1248,
            "planned_trials": 1200,
            "outcome_rows": len(result["outcomes"]),
            "event_counts": event_counts,
            "lease": lease_result,
            "delta_route": "both_bypassed",
            "official_report_sha256": child._sha((scorer_output / "report.json").read_bytes()),
            "score_accepted": False,
            "model_score": None,
            "callback_timing": {
                "count": len(timing_rows),
                "sha256": child._sha(timing_raw),
                "scope": "callback entry through verified response; includes guard and generic evidence fsync, excludes timing journal write and SDK cache hits",
            },
            "cost_usd": None,
            "cost_basis": "unknown_local_compute",
            "latency_scope": "guarded callback; not model-forward-only",
        }
    except BaseException as error:
        primary = error
        if "lease" in locals() and (
            lease.fd is not None or lease.held is not None or lease.fatal is not None
        ):
            error.owner_cleanup_unconfirmed = True
        raise
    finally:
        deltas.clear()
        try:
            _quarantine_scorer_output(scorer_output)
        except BaseException as error:
            if primary is None:
                raise
            primary.add_note("scorer quarantine also failed: " + type(error).__name__)
        finally:
            sys.meta_path.remove(finder)
            for name in list(sys.modules):
                if name == "jevbench" or name.startswith("jevbench."):
                    del sys.modules[name]


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
        "schema": "decision3-jevbench-coherence-hello-v1",
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
        schema="decision3-jevbench-coherence-go-v1",
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
        "schema": "decision3-jevbench-coherence-terminal-v1",
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
        status["evaluation"] = _run_coherence(
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
