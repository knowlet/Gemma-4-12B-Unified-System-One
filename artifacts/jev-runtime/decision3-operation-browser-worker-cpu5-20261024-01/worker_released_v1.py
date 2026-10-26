"""SOURCE-ONLY owned CPU5 candidate browser wiring, reusing original authorities.

No new guardian, training/data loader, API route or caller is introduced. The
default hold precedes arguments, IO, source imports, models, HTTP and Chromium.
All new sources are unexecuted/unfrozen/unselected at authoring time. Only the
operation head uses the distinct accepted candidate. Dev informed selection;
the original exposed browser toy remains a repeated diagnostic, not training
material, unseen evaluation, an official score or a quality claim.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import os
import re
import socket
import sys
import threading
import time
import types
from pathlib import Path

SOURCE_ONLY = False
ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
HERE = ROOT / "artifacts/jev-runtime/decision3-operation-browser-worker-cpu5-20261024-01"
BASE = ROOT / "artifacts/jev-runtime/decision3-cpu5-browser-worker-candidate-20261024-01"
DELTA = ROOT / "artifacts/jev-runtime/decision3-operation-head-backend-cpu5-20261024-01"
TRAIN = ROOT / "artifacts/jev-runtime/decision3-operation-readout-cpu5-20261024-01"
NUMERIC = ROOT / "artifacts/jev-runtime/decision3-operation-numeric-qualification-cpu5-20261024-01"
OBSERVER = (
    ROOT
    / "artifacts/jev-runtime/decision3-readout-numpy-cpu5-qualification-20261024-02/observer.py"
)
BASE_WORKER_SHA256 = "b317cf48300bfbf3c2c4d54d72dae1589278a28ebec0eeb20eac683a931eb15d"
BASE_BRIDGE_SHA256 = "1c8f689bd40d5073f44f7e567c32c5944bf0df6a33957ba571760072e7632b00"
CAPABILITY_SHA256 = "4dbbb9470d011931854f6473c0840ef7f0746c4194c719cde6ab454c3e79d463"
NUMERIC_ACCEPTANCE_SHA256 = "404828850c2e3b3a46f8e20e712603c31b1d7156728b17bab740937207734c21"
PROFILE = "decision3-dsl01-operation-cpu5-owned-two-control-browser-worker-v1"
ALIAS = "d3-edge-dsl01-operation-delta-cpu5-candidate"
CANDIDATE_PROFILE = "decision3-dsl01-operation-readout-cpu5-4g-candidate-v1"
HYPERPARAMETERS = {
    "steps": 512,
    "learning_rate": 0.003,
    "l2": 0.0001,
    "seed": 20261023,
    "batch_size": 64,
}
WHOLE_SECONDS = 120


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def released():
    require(
        SOURCE_ONLY is False,
        "SOURCE_ONLY: no arguments, IO, imports, model, HTTP or browser authorized",
    )
    require(
        all(
            isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha)
            for sha in (
                BASE_WORKER_SHA256,
                BASE_BRIDGE_SHA256,
                CAPABILITY_SHA256,
                NUMERIC_ACCEPTANCE_SHA256,
            )
        ),
        "original source and new operation capability/numeric pins remain pending",
    )


def bootstrap():
    released()
    path = BASE / "worker_v2.py"
    require(
        not any(p.is_symlink() for p in (path, *path.parents)),
        "ordinary fixed helper source required",
    )
    with path.open("rb") as stream:
        raw = stream.read(1024**2 + 1)
    require(
        len(raw) <= 1024**2 and hashlib.sha256(raw).hexdigest() == BASE_WORKER_SHA256,
        "original worker source SHA differs",
    )
    module = types.ModuleType("_delta_browser_original_worker_v2")
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(raw, str(path), "exec", dont_inherit=True), vars(module))
    return module


def validate_plan(base, plan, config, path, sha):
    require(
        plan.get("schema") == "owned-decision3-operation-cpu5-browser-worker-plan-v1"
        and plan.get("profile") == PROFILE
        and plan.get("status") == "selected_after_separate_review"
        and plan.get("owner_profile") == base.OWNER_PROFILE
        and plan.get("owner_unit") == base.UNIT
        and plan.get("model_alias") == ALIAS
        and plan.get("only_head") == "operation"
        and plan.get("diagnostic_only") is True
        and plan.get("max_ticks") == 16
        and plan.get("http_caller_timeout_seconds") == 25
        and plan.get("http_cooperative_budget_seconds") == 20
        and plan.get("whole_budget_seconds") == WHOLE_SECONDS
        and plan.get("helper_model") is None,
        "pending/foreign resource, candidate identity or caller budgets refused",
    )
    expected = {
        "base_worker": (BASE / "worker_v2.py", BASE_WORKER_SHA256),
        "base_bridge": (BASE / "bridge_successor.py", BASE_BRIDGE_SHA256),
        "native_actor": (base.NATIVE / "actor.py", base.NATIVE_SHA),
        "native_plan": (base.NATIVE / "edge-plan.json", base.NATIVE_PLAN_SHA),
        "observer": (OBSERVER, None),
        "worker": (Path(__file__).absolute(), None),
        "caller": (base.CALLER, None),
    }
    pool = {str(ROOT / row["path"]): row["sha256"] for row in config["sources"]}
    require(pool.get(str(Path(path).absolute())) == sha, "plan lacks original command source pin")
    for key, (required_path, required_sha) in expected.items():
        record = plan[key]
        require(
            record["path"] == str(required_path)
            and (required_sha is None or record["sha256"] == required_sha),
            "fixed dependency identity differs: " + key,
        )
    for key in (*expected, "adapter", "delta_backend"):
        record = plan[key]
        require(
            isinstance(record.get("path"), str) and isinstance(record.get("sha256"), str),
            "source dependency pins remain pending: " + key,
        )
        selected = base._ordinary(record["path"])
        require(
            isinstance(record["sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
            and pool.get(str(selected)) == record["sha256"]
            and base._sha(base._read(selected)) == record["sha256"],
            "dependency not authenticated by original selected union: " + key,
        )
    require(
        Path(plan["adapter"]["path"]).parent == HERE
        and Path(plan["delta_backend"]["path"]) == DELTA / "backend_released_v1.py",
        "only reviewed local candidate source successors allowed",
    )
    require(
        type(plan.get("import_sources")) is list and plan["import_sources"],
        "original explicit import closure remains pending",
    )
    for record in plan["import_sources"]:
        require(
            set(record) == {"module", "path", "sha256"}
            and pool.get(record["path"]) == record["sha256"]
            and base._sha(base._read(record["path"])) == record["sha256"],
            "imported source lacks original selected identity",
        )
    # No invented qualification record is supplied by this source. Root must
    # select and semantically review the genuine closed acceptance first.
    for key in ("candidate_capability", "numeric_acceptance"):
        record = plan.get(key)
        require(
            type(record) is dict
            and set(record) == {"path", "sha256", "size_bytes"}
            and isinstance(record["sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
            and type(record["size_bytes"]) is int
            and 0 < record["size_bytes"] <= 1024**2
            and pool.get(record["path"]) == record["sha256"],
            "genuine root-selected capability/numeric acceptance remains pending",
        )
    require(
        plan["candidate_capability"]["path"]
        == str(TRAIN / "root-frozen-operation-candidate-capability-01.json")
        and plan["candidate_capability"]["sha256"] == CAPABILITY_SHA256
        and plan["numeric_acceptance"]["path"]
        == str(NUMERIC / "root-accepted-operation-numeric-01.json")
        and plan["numeric_acceptance"]["sha256"] == NUMERIC_ACCEPTANCE_SHA256,
        "exact accepted operation capability and closed numeric acceptance required",
    )


def read_ref(base, record):
    require(
        type(record["size_bytes"]) is int and 0 < record["size_bytes"] <= 2 * 1024**2,
        "bounded artifact required",
    )
    raw = base._read(record["path"], maximum=2 * 1024**2)
    require(
        len(raw) == record["size_bytes"] and base._sha(raw) == record["sha256"],
        "selected artifact bytes changed",
    )
    return raw


def fp32_payload(raw):
    """Only a bounded NPY v1 little-endian contiguous FP32[255,1024] record.

    This parses a literal metadata dict, never pickle or Python code. File SHA
    authentication happens before this function; exact copied bytes after it.
    """
    require(
        sys.byteorder == "little" and raw[:8] == b"\x93NUMPY\x01\x00",
        "fixed little-endian NPYv1 required",
    )
    length = int.from_bytes(raw[8:10], "little")
    require(
        0 < length <= 4096 and len(raw) == 10 + length + 255 * 1024 * 4,
        "exact bounded NPY header/data length required",
    )
    header = raw[10 : 10 + length]
    require(header.endswith(b"\n"), "native NPY header terminator required")
    metadata = ast.literal_eval(header.decode("ascii").strip())
    require(
        metadata == {"descr": "<f4", "fortran_order": False, "shape": (255, 1024)},
        "exact plain C-order FP32 shape required",
    )
    return raw[10 + length :]


def tensor_sha(torch, tensor):
    require(
        tensor.device.type == "cpu"
        and tensor.dtype == torch.float32
        and tuple(tensor.shape) == (255, 1024),
        "actual CPU FP32 readout required",
    )
    # A bounded 1MiB byte copy, without importing NumPy or exporting values.
    return hashlib.sha256(
        bytes(tensor.detach().contiguous().view(torch.uint8).reshape(-1).tolist())
    ).hexdigest()


def validate_numeric_acceptance(numeric, capability, model, plan):
    require(
        numeric.get("schema") == "root-accepted-operation-cpu5-numeric-v1"
        and numeric.get("authority") == "root"
        and numeric.get("status") == "accepted"
        and numeric.get("root_exit") == 0
        and numeric.get("owner_completed") is True
        and numeric.get("current_original_scoped_closure_positive") is True
        and numeric.get("candidate_alias") == ALIAS
        and numeric.get("candidate_profile") == CANDIDATE_PROFILE
        and numeric.get("only_adapted_head_id") == "operation"
        and numeric.get("dev_used_for_model_selection") is True
        and numeric.get("optimizer_input_dev") is False
        and numeric.get("optimizer_input_sealed") is False
        and numeric.get("candidate_selection_before_dev_or_sealed") is False
        and numeric.get("candidate_capability") == plan["candidate_capability"]
        and numeric.get("training_acceptance") == capability["root_acceptance_ref"]
        and numeric.get("delta_ref") == capability["delta_ref"]
        and numeric.get("baseline_fp32_bytes_sha256") == capability["baseline_fp32_bytes_sha256"]
        and numeric.get("model") == model
        and numeric.get("browser_qualified") is False
        and numeric.get("quality_or_win_claim") is False
        and all(
            numeric.get("backend_source", {}).get(key) == plan["delta_backend"][key]
            for key in ("path", "sha256")
        ),
        "actual root numeric acceptance/candidate/backend/model/baseline binding differs",
    )


def load_candidate(base, plan, model, torch, runtime):
    capability = base._decode(read_ref(base, plan["candidate_capability"]))
    accepted = base._decode(read_ref(base, capability["root_acceptance_ref"]))
    require(
        accepted["schema"] == "root-accepted-dsl01-operation-trained-readout-v1"
        and accepted["status"] == "accepted"
        and accepted["authority"] == "root"
        and accepted["root_exit"] == 0
        and accepted["owner_completed"] is True
        and accepted["current_original_scoped_closure_positive"] is True
        and capability["schema"] == "root-accepted-dsl01-operation-train-delta-v1"
        and capability["status"] == "ROOT_ACCEPTED_OPERATION_TRAIN_ONLY_READOUT"
        and capability["split"] == "train"
        and capability["nontraining"] is False
        and capability["training_eligible"] is True
        and capability["model"] == accepted["model"] == model
        and capability["alias"] == ALIAS
        and capability["profile"] == CANDIDATE_PROFILE
        and capability["only_head"] == "operation"
        and capability["hyperparameters"] == accepted["hyperparameters"] == HYPERPARAMETERS
        and capability["dev_used_for_model_selection"] is True
        and capability["optimizer_input_dev"] is False
        and capability["optimizer_input_sealed"] is False
        and capability["candidate_selection_before_dev_or_sealed"] is False
        and capability["W0_merge_allowed"] is False
        and capability["delta_ref"] == accepted["delta_ref"]
        and capability["fit_capability"] == accepted["capability"]
        and capability["baseline_fp32_bytes_sha256"]
        == accepted["baseline_ref"]["FP32_bytes_sha256"],
        "frozen artifact/model/train fit/baseline acceptance differs",
    )
    numeric = base._decode(read_ref(base, plan["numeric_acceptance"]))
    validate_numeric_acceptance(numeric, capability, model, plan)
    pool = {
        str(ROOT / row["path"]): row["sha256"]
        for row in base._json(Path(plan["owner_output"]) / "sealed-config.json")["sources"]
    }
    for record in (capability["root_acceptance_ref"], capability["delta_ref"]):
        require(
            pool.get(record["path"]) == record["sha256"],
            "accepted artifact lacks original selected byte pin",
        )
    raw = read_ref(base, capability["delta_ref"])
    payload = fp32_payload(raw)
    payload_sha = hashlib.sha256(payload).hexdigest()
    require(
        payload_sha == accepted["delta_statistics"]["FP32_raw_bytes_sha256"],
        "accepted delta value-byte hash differs",
    )
    delta = (
        torch.frombuffer(bytearray(payload), dtype=torch.float32)
        .reshape(255, 1024)
        .clone()
        .contiguous()
    )
    require(
        not delta.requires_grad and bool(torch.isfinite(delta).all().item()),
        "finite frozen copied delta required",
    )
    require(
        tensor_sha(torch, delta) == payload_sha,
        "actual copied tensor differs from accepted FP32 bytes",
    )
    baseline_sha = tensor_sha(torch, runtime.readout)
    require(
        baseline_sha == capability["baseline_fp32_bytes_sha256"],
        "actual W0 bits differ from accepted original baseline",
    )
    return (
        capability,
        delta,
        {
            "baseline_fp32_bytes_sha256": baseline_sha,
            "delta_fp32_bytes_sha256": payload_sha,
            "numeric_acceptance": copy.deepcopy(plan["numeric_acceptance"]),
        },
    )


def dependencies():
    from s1.errors import BackendResponseError, BackendTimeoutError, RequestValidationError
    from s1.evaluation.decision3 import (
        configuration_blockers,
        decision3_spec,
        native_payload,
        normalize_native_response,
    )
    from s1.jev_decision3_native import Decision3NativeRequest, parse_native_payload
    from scripts.jev_decision3_browser_wire import named_browser_response

    return {
        "configuration_blockers": configuration_blockers,
        "spec_factory": decision3_spec,
        "native_payload": native_payload,
        "normalize_native_response": normalize_native_response,
        "request_type": Decision3NativeRequest,
        "parse_native_payload": parse_native_payload,
        "named_browser_response": named_browser_response,
        "timeout_type": BackendTimeoutError,
        "validation_type": RequestValidationError,
        "response_error_type": BackendResponseError,
    }


def call_original(base, caller, endpoint, plan, output, fatal):
    """Only environment alias changes; actual caller source/fixture/logic is reused."""
    saved = {key: os.environ.get(key) for key in base.FIXED_ENVIRONMENT}
    try:
        os.environ.update(
            {
                "JEV_NATIVE_HTTP_BROWSER_TESTS": "1",
                "JEV_NATIVE_HTTP_ENDPOINT": endpoint,
                "JEV_NATIVE_HTTP_MODEL": ALIAS,
                "JEV_UPSTREAM_PATH": plan["upstream_path"],
                "JEV_CHROMIUM_EXECUTABLE": plan["chromium_executable"],
                "JEV_NATIVE_HTTP_BROWSER_OUTPUT": str(output),
                "JEV_NATIVE_HTTP_API_KEY": "",
            }
        )
        fatal.raise_if_set()
        caller.test_real_native_http_choose_agent_and_guarded_click()
        fatal.raise_if_set()
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run(plan_path, plan_sha, owner_output):
    released()
    began = time.monotonic()

    def deadline():
        require(
            time.monotonic() - began < WHOLE_SECONDS, "whole owned candidate-browser120 exceeded"
        )

    base = bootstrap()
    raw_plan = base._read(plan_path)
    require(base._sha(raw_plan) == plan_sha, "selected candidate plan SHA differs")
    plan = base._decode(raw_plan)
    require(
        plan.get("owner_output") == str(Path(owner_output).absolute()),
        "exact new original owner output required",
    )
    owner, boundary, config, admission, process = base._owner_binding(owner_output)
    validate_plan(base, plan, config, plan_path, plan_sha)
    bridge = base._import_source(
        "_delta_browser_original_gateway", BASE / "bridge_successor.py", BASE_BRIDGE_SHA256
    )
    native = base._import_source(
        "_delta_browser_original_native05", base.NATIVE / "actor.py", base.NATIVE_SHA
    )
    helper = native.load_helper()
    native_plan = native.load_plan(helper, base.NATIVE / "edge-plan.json", base.NATIVE_PLAN_SHA)
    output = base._new_output(Path(owner_output) / "operation-browser-candidate")
    fatal, primary = base.FatalLatch(), None
    state = {"model_loads": 0}
    listener = server = thread = runtime = torch = report = None
    capability = delta = artifact_proof = None
    receipt = {
        "profile": PROFILE,
        "candidate_alias": ALIAS,
        "only_adapted_head_id": "operation",
        "diagnostic_only": True,
        "dev_used_for_model_selection": True,
        "optimizer_input_dev": False,
        "optimizer_input_sealed": False,
        "candidate_selection_before_dev_or_sealed": False,
        "new_unseen_evaluation_claim": False,
        "official_score_claim": False,
        "publisher_native_numerics": False,
        "stage": "bound_before_model",
        "worker": process,
        "plan_sha256": plan_sha,
        "native_state": state,
        "helper_model_loaded": False,
        "helper_model_calls": 0,
        "cleanup_errors": [],
        "success": False,
        "pending_native_or_server": False,
        "whole_tree_exit_confirmed": False,
        "external_current_owner_closure_required": True,
        "no_escape_enforced": False,
        "browser_qualified": False,
    }
    save = base._ReceiptWriter(boundary, helper, output)

    def cleanup_failure(phase, error):
        nonlocal primary
        receipt["cleanup_errors"].append({"stage": phase, **base._failure(error)})
        if primary is None:
            primary = error
        elif primary is not error:
            primary.add_note(phase + " also failed: " + repr(error))

    def verify_owned():
        deadline()
        try:
            _, _, current, current_admission, current_process = base._owner_binding(
                owner_output, loaded=(owner, boundary)
            )
            require(
                current == config
                and current_admission == admission
                and owner.same_identity(current_process, process),
                "original worker/admission identity changed",
            )
            validate_plan(base, plan, current, plan_path, plan_sha)
            require(base._sha(base._read(plan_path)) == plan_sha, "selected plan changed")
            cpu = helper.process_identity()
            native.validate_owner(
                cpu, helper.owned_limits(cpu), base._phase_environment(helper, state)
            )
            deadline()
            return {
                "schema": base.BRIDGE_BINDING,
                "plan_path": str(Path(plan_path).absolute()),
                "plan_sha256": plan_sha,
                "config_sha256": base._sha(base._read(Path(owner_output) / "sealed-config.json")),
                "selection_sha256": config["selection_sha256"],
                "owner_unit": base.UNIT,
                "owner_profile": base.OWNER_PROFILE,
                "nonce": config["nonce"],
                "invocation_id": admission["invocation_id"],
                "worker_pid": os.getpid(),
                "worker_start_ticks": process["process"]["start_ticks"],
                "device": "cpu",
                "precision": "bfloat16",
                "context_limit": 16384,
                "memory_max_bytes": 4294967296,
                "swap_max_bytes": 0,
                "cpu_affinity": [15, 16, 17, 18, 19],
                "cpu_quota_percent": 500,
                "expected_threads": 5,
                "expected_interop_threads": 1,
                "native_model": native_plan["model"],
            }
        except BaseException as error:
            fatal.capture(error)
            raise

    try:
        save(receipt)
        release = bridge.bind_owned_selection(verify_owned)
        owner.selected_executable(plan["chromium_executable"], config["binary_stats"])
        receipt["stage"] = "loading_one_original_native_runtime"
        save(receipt)
        runtime, torch = native.load_runtime(helper, native_plan, output, state)
        require(state["model_loads"] == 1, "exactly one original native model load required")
        deadline()
        capability, delta, artifact_proof = load_candidate(
            base, plan, native_plan["model"], torch, runtime
        )
        receipt.update(
            stage="accepted_artifact_loaded_before_browser_imports", artifact_proof=artifact_proof
        )
        save(receipt)
        adapter = base._import_source(
            "_delta_browser_candidate_adapter", plan["adapter"]["path"], plan["adapter"]["sha256"]
        )
        delta_backend = base._import_source(
            "_delta_browser_released_readout",
            plan["delta_backend"]["path"],
            plan["delta_backend"]["sha256"],
        )
        observer = base._import_source(
            "_delta_browser_same_forward_observer", OBSERVER, plan["observer"]["sha256"]
        )
        require(
            adapter.SOURCE_ONLY is False and delta_backend.SOURCE_ONLY is False,
            "candidate sources remain held",
        )
        sys.path.insert(0, str(ROOT / "src"))
        sys.path.insert(0, str(ROOT))
        deps = dependencies()
        provider = adapter.CandidateProvider(
            spec=deps["spec_factory"]("d3-edge", device="cpu"),
            native=native,
            observer=observer,
            delta_backend=delta_backend,
            runtime=runtime,
            torch=torch,
            model=native_plan["model"],
            delta=delta,
            capability=capability,
            capability_sha256=plan["candidate_capability"]["sha256"],
            native_payload=deps["native_payload"],
            verify_owned=release.verify,
        )
        identity = {
            "device": "cpu",
            "precision": "bfloat16",
            "context_limit": 16384,
            "profile": PROFILE,
            "candidate_alias": ALIAS,
            "only_adapted_head_id": "operation",
            "diagnostic_only": True,
            "dev_used_for_model_selection": True,
            "optimizer_input_dev": False,
            "optimizer_input_sealed": False,
            "candidate_selection_before_dev_or_sealed": False,
            "new_unseen_evaluation_claim": False,
            "candidate_capability": plan["candidate_capability"],
            "numeric_acceptance": plan["numeric_acceptance"],
            "publisher_native_numerics": False,
            "readout_adaptation_applied": True,
            "wire_copies_candidate_probabilities_without_change": True,
            "owner_unit": base.UNIT,
            "owner_profile": base.OWNER_PROFILE,
            "nonce": config["nonce"],
            "invocation_id": admission["invocation_id"],
            "pid": os.getpid(),
            "start_ticks": process["process"]["start_ticks"],
            "native_model": native_plan["model"],
            "legacy_v1_scope": "DTO/wire ABI only; adapted CPU candidate, no CUDA qualification",
        }
        wrapped = []

        def guard(backend):
            result = base.FatalBackend(backend, fatal)
            wrapped.append(result)
            return result

        server_output = base._new_output(output / "server-evidence")
        application = adapter.create_application(
            bridge=bridge,
            owned_release=release,
            provider=provider,
            dependencies=deps,
            output=server_output,
            cpu_identity=identity,
            backend_wrapper=guard,
        )
        require(
            len(wrapped) == 1 and application.backend is wrapped[0],
            "every candidate forward must pass original fatal guard",
        )
        application = base.GuardedASGI(application, fatal)
        caller = base._import_source(
            "_delta_browser_unchanged_original_caller", base.CALLER, plan["caller"]["sha256"]
        )
        import uvicorn

        base._verify_imports(plan)
        deadline()
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        endpoint = "http://127.0.0.1:" + str(listener.getsockname()[1]) + "/v1/systemone"
        server = uvicorn.Server(
            uvicorn.Config(
                application, host="127.0.0.1", workers=1, lifespan="off", log_level="error"
            )
        )

        def serve():
            try:
                require(
                    os.sched_getaffinity(0) == {15, 16, 17, 18, 19}
                    and torch.get_num_threads() == 5
                    and torch.get_num_interop_threads() == 1
                    and not torch.cuda.is_initialized(),
                    "actual ASGI thread CPU5 policy differs",
                )
                native.validate_runtime(runtime, torch, native_plan["model"])
                server.run(sockets=[listener])
                require(server.should_exit, "ASGI exited without owned shutdown")
            except BaseException as error:
                fatal.capture(error)

        thread = threading.Thread(target=serve, name="owned-delta-candidate-asgi", daemon=False)
        receipt.update(stage="starting_original_loopback_gateway", endpoint=endpoint)
        save(receipt)
        thread.start()
        until = time.monotonic() + 15
        while not server.started:
            fatal.raise_if_set()
            deadline()
            require(thread.is_alive() and time.monotonic() < until, "owned ASGI readiness failed")
            time.sleep(0.02)
        call_original(base, caller, endpoint, plan, output / "browser", fatal)
        deadline()
    except BaseException as error:
        primary = fatal.first if fatal.event.is_set() else error
        if primary is not error:
            primary.add_note("Caller/worker also failed: " + repr(error))
    finally:
        # Original caller finally closes Agent/Chromium before this shutdown.
        try:
            if server is not None:
                server.should_exit = True
        except BaseException as error:
            cleanup_failure("server_shutdown_request", error)
        drained = thread is None or thread.ident is None
        try:
            if thread is not None and thread.ident is not None:
                thread.join(timeout=10)
                drained = not thread.is_alive()
            receipt["server_thread_drained"] = drained
            if not drained:
                receipt["pending_native_or_server"] = True
                raise RuntimeError(
                    "ASGI/NN live after original10s drain; original owner must close cohort"
                )
        except BaseException as error:
            drained = False
            receipt["pending_native_or_server"] = True
            cleanup_failure("server_drain", error)
        if listener is not None and drained:
            try:
                listener.close()
            except BaseException as error:
                cleanup_failure("listener_close", error)
        if drained:
            try:
                path = output / "browser/report.json"
                if path.exists():
                    report = base._json(path, maximum=256 * 1024**2)
                    receipt["browser_report_sha256"] = base._sha(
                        base._read(path, maximum=256 * 1024**2)
                    )
                    receipt["http_evidence_closure"] = base._http_closure(
                        report, output / "server-evidence"
                    )
                    require(
                        report.get("success") is True
                        and all(row["kind"] == "policy" for row in report["records"])
                        and report.get("final_snapshot", {}).get("text_calls") == [],
                        "unchanged strict caller/helper0 did not succeed",
                    )
            except BaseException as error:
                cleanup_failure("unchanged_browser_report_and_http_closure", error)
            if runtime is not None:
                try:
                    fatal.raise_if_set()
                    native.validate_runtime(runtime, torch, native_plan["model"])
                    native.authenticate(helper, native_plan)
                    require(
                        artifact_proof is not None
                        and tensor_sha(torch, runtime.readout)
                        == artifact_proof["baseline_fp32_bytes_sha256"],
                        "final original W0 bits changed",
                    )
                    require(
                        tensor_sha(torch, delta) == artifact_proof["delta_fp32_bytes_sha256"],
                        "final frozen delta bits changed",
                    )
                    for record in (
                        plan["candidate_capability"],
                        plan["numeric_acceptance"],
                        capability["root_acceptance_ref"],
                        capability["delta_ref"],
                    ):
                        read_ref(base, record)
                    receipt["final_native_and_artifact_authentication"] = True
                except BaseException as error:
                    cleanup_failure("final_native_and_artifact_authentication", error)
        try:
            owner.sources_match(config["sources"])
            base._verify_imports(plan)
            deadline()
            receipt["sources_unchanged"] = True
        except BaseException as error:
            cleanup_failure("source_and_whole120_retention", error)
        if fatal.event.is_set() and primary is None:
            primary = fatal.first
        receipt["success"] = bool(
            primary is None
            and drained
            and report
            and report.get("success") is True
            and receipt.get("final_native_and_artifact_authentication") is True
            and receipt.get("sources_unchanged") is True
        )
        receipt["stage"] = (
            "complete_pending_original_external_closure"
            if receipt["success"]
            else "failed_or_pending_no_wiring_success"
        )
        receipt["error"] = None if primary is None else base._failure(primary)
        receipt["elapsed_before_final_write_seconds"] = time.monotonic() - began
        try:
            save(receipt)
            deadline()  # all retention, drain and terminal writes are charged
        except BaseException as error:
            cleanup_failure("final_receipt_or_whole120", error)
            receipt.update(
                success=False,
                stage="failed_or_pending_no_wiring_success",
                error=base._failure(primary),
            )
            try:
                save(receipt)  # Bounded failure-only publication; never authorizes success.
            except BaseException as secondary:
                primary.add_note("Failure receipt also failed: " + repr(secondary))
    if primary is not None:
        raise primary
    return receipt


def main(argv=None):
    released()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--owner-output", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.plan, args.plan_sha256, args.owner_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
