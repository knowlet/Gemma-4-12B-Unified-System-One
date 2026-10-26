"""HELD HTTP-only Decision3 service for one original V7 case.

This source has not been executed or qualified. The original owner09 remains
the resource and final closure authority. No benchmark or browser is imported.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import socket
import stat
import sys
import threading
import time
import types
from pathlib import Path

SOURCE_ONLY = True
ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
HERE = ROOT / "artifacts/jev-runtime/decision3-v7-full-choice-service-cpu5-20261024-01"
OPERATION = ROOT / "artifacts/jev-runtime/decision3-operation-browser-worker-cpu5-20261024-01"
MATCHED = ROOT / "artifacts/jev-runtime/decision3-v7-matched-browser-cpu5-20261024-01"
OPERATION_WORKER_SHA256 = None
ADAPTER_SHA256 = None
BACKEND_SHA256 = None
BRIDGE_SHA256 = None
SERIAL_SHA256 = None
CAPABILITY_SHA256 = None
NUMERIC_ACCEPTANCE_SHA256 = None
PROFILE = "decision3-dsl01-operation-cpu5-v7-full-choice-http-service-cooperative1200-v1"
SCHEMA = "owned-decision3-v7-full-choice-service-plan-v1"
ALIAS = "d3-edge-dsl01-operation-delta-cpu5-candidate"
COOPERATIVE_SECONDS = 1200
OWNER_HARD_SECONDS = 7200
READY_SCHEMA = "jev-v7-full-choice-service-ready-v1"
STOP_SCHEMA = "jev-v7-full-choice-service-stop-v1"
CLOSED_SCHEMA = "jev-v7-full-choice-service-closed-v1"
CASE_IDS = (
    "public-arxiv-title-filter-results-v1",
    "public-arxiv-original-paper-open-v1",
    "public-python-counter-most-common-v1",
    "public-nasa-mars-facts-moons-v1",
    *(f"heldout-public-{index:03d}" for index in range(6)),
)


def require(value, message):
    if not value:
        raise RuntimeError(message)


def released():
    require(SOURCE_ONLY is False, "SOURCE_ONLY: no arguments, IO, imports, model, or HTTP")
    for value in (
        OPERATION_WORKER_SHA256,
        ADAPTER_SHA256,
        BACKEND_SHA256,
        BRIDGE_SHA256,
        SERIAL_SHA256,
        CAPABILITY_SHA256,
        NUMERIC_ACCEPTANCE_SHA256,
    ):
        require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "pending pin")


def ordinary(path):
    path = Path(path).absolute()
    require(path.is_relative_to(ROOT) and ".." not in path.parts, "ordinary repository path")
    require(not any(item.is_symlink() for item in (path, *path.parents)), "no symlink")
    return path


def bootstrap():
    """Import only selected source definitions, never the old worker's run()."""
    released()
    path = ordinary(OPERATION / "worker_released_v1.py")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= 1024**2, "small source")
        raw = stream.read(1024**2 + 1)
        after = os.fstat(stream.fileno())
    require(
        len(raw) == before.st_size
        and all(
            getattr(before, field) == getattr(after, field) == getattr(path.stat(), field)
            for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        )
        and hashlib.sha256(raw).hexdigest() == OPERATION_WORKER_SHA256,
        "selected original operation worker bytes differ",
    )
    name = "_v7_http_original_operation_source"
    require(name not in sys.modules, "fresh source module required")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
        return module, module.bootstrap()
    except BaseException:
        sys.modules.pop(name, None)
        raise


def binding(plan, plan_sha):
    return {
        "case_index": plan["case_index"],
        "case_id": plan["case_id"],
        "root_plan_sha256": plan["root_plan_sha256"],
        "plan_sha256": plan_sha,
        "nonce": plan["handoff"]["nonce"],
    }


def stop_record(plan, plan_sha, record):
    require(
        record == {"schema": STOP_SCHEMA, **binding(plan, plan_sha)},
        "exact selected stop schema/case/root-plan/service-plan/nonce required",
    )
    return record


def publish(path, value):
    """Publish a new private handoff, charging file and parent durability."""
    path = ordinary(path)
    require(path.parent.is_dir() and not path.exists(), "precreated private handoff directory")
    parent = path.parent.stat()
    require(
        parent.st_uid == os.getuid() and stat.S_IMODE(parent.st_mode) == 0o700, "private parent"
    )
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    temporary = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    primary = None
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            if temporary.exists():
                temporary.unlink()
        except BaseException as secondary:
            if primary is None:
                raise
            if secondary is not primary:
                primary.add_note("Temporary handoff cleanup also failed: " + repr(secondary))


def read_stop(base, plan, plan_sha):
    path = ordinary(plan["handoff"]["stop_path"])
    info = path.stat()
    require(
        stat.S_ISREG(info.st_mode)
        and info.st_uid == os.getuid()
        and stat.S_IMODE(info.st_mode) == 0o600,
        "private regular selected stop required",
    )
    return stop_record(plan, plan_sha, base._json(path))


def validate_plan(operation, base, plan, config, path, plan_sha):
    require(
        plan.get("schema") == SCHEMA
        and plan.get("profile") == PROFILE
        and plan.get("status") == "selected_after_separate_review"
        and plan.get("owner_profile") == base.OWNER_PROFILE
        and plan.get("owner_unit") == base.UNIT
        and plan.get("model_alias") == ALIAS
        and plan.get("only_head") == "operation"
        and plan.get("diagnostic_only") is True
        and plan.get("cooperative_budget_seconds") == COOPERATIVE_SECONDS
        and plan.get("owner_hard_runtime_seconds") == OWNER_HARD_SECONDS
        and plan.get("http_cooperative_budget_seconds") == 20
        and plan.get("http_caller_timeout_seconds") == 25
        and plan.get("helper_model") is None
        and config.get("runtime_seconds") == OWNER_HARD_SECONDS
        and len(config["sources"]) <= 63,
        "exact selected HTTP-only service/resource profile required",
    )
    index = plan.get("case_index")
    require(
        type(index) is int and 1 <= index <= 10 and plan.get("case_id") == CASE_IDS[index - 1],
        "original exact-ten case order",
    )
    require(
        isinstance(plan.get("root_plan_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", plan["root_plan_sha256"]),
        "selected root plan SHA",
    )
    handoff = plan.get("handoff", {})
    require(
        set(handoff) == {"ready_path", "stop_path", "closed_path", "nonce"}, "exact handoff fields"
    )
    require(
        isinstance(handoff["nonce"], str) and re.fullmatch(r"[0-9a-f]{32}", handoff["nonce"]),
        "selected case nonce",
    )
    parents = set()
    for key, filename in (
        ("ready_path", "service-ready.json"),
        ("stop_path", "service-stop.json"),
        ("closed_path", "service-closed.json"),
    ):
        item = ordinary(handoff[key])
        require(
            str(item) == handoff[key]
            and item.name == filename
            and item.parent.name == f"case-{index:02d}",
            "fixed case service handoff paths",
        )
        parents.add(item.parent)
    require(len(parents) == 1, "one selected case handoff directory")
    selected = {str(ROOT / row["path"]): row["sha256"] for row in config["sources"]}
    require(
        selected.get(str(Path(path).absolute())) == plan_sha,
        "service plan lacks original owner pin",
    )
    expected = {
        "operation_worker": (OPERATION / "worker_released_v1.py", OPERATION_WORKER_SHA256),
        "base_worker": (base.HERE / "worker_v2.py", operation.BASE_WORKER_SHA256),
        "native_actor": (base.NATIVE / "actor.py", base.NATIVE_SHA),
        "native_plan": (base.NATIVE / "edge-plan.json", base.NATIVE_PLAN_SHA),
        "observer": (operation.OBSERVER, None),
        "worker": (Path(__file__).absolute(), None),
        "bridge": (HERE / "bridge_full_choice_released_v1.py", BRIDGE_SHA256),
        "adapter": (OPERATION / "adapter_released_v1.py", ADAPTER_SHA256),
        "delta_backend": (operation.DELTA / "backend_released_v1.py", BACKEND_SHA256),
        "serial_forward": (MATCHED / "serial_forward_released_v1.py", SERIAL_SHA256),
    }
    for key, (expected_path, expected_sha) in expected.items():
        record = plan.get(key, {})
        require(
            set(record) == {"path", "sha256"}
            and record["path"] == str(expected_path)
            and isinstance(record["sha256"], str)
            and re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
            and (expected_sha is None or record["sha256"] == expected_sha)
            and selected.get(record["path"]) == record["sha256"]
            and base._sha(base._read(record["path"])) == record["sha256"],
            "selected source path/pin differs: " + key,
        )
    for key, expected_path, expected_sha in (
        (
            "candidate_capability",
            operation.TRAIN / "root-frozen-operation-candidate-capability-01.json",
            CAPABILITY_SHA256,
        ),
        (
            "numeric_acceptance",
            operation.NUMERIC / "root-accepted-operation-numeric-01.json",
            NUMERIC_ACCEPTANCE_SHA256,
        ),
    ):
        record = plan.get(key, {})
        require(
            set(record) == {"path", "sha256", "size_bytes"}
            and record["path"] == str(expected_path)
            and record["sha256"] == expected_sha
            and type(record["size_bytes"]) is int
            and 0 < record["size_bytes"] <= 1024**2
            and selected.get(record["path"]) == expected_sha,
            "selected accepted artifact differs: " + key,
        )
    require(
        type(plan.get("import_sources")) is list and bool(plan["import_sources"]),
        "selected imported source closure",
    )
    for record in plan["import_sources"]:
        require(
            set(record) == {"module", "path", "sha256"}
            and selected.get(record["path"]) == record["sha256"]
            and base._sha(base._read(record["path"])) == record["sha256"],
            "selected import differs",
        )
    serialization = plan.get("serialization", {})
    require(
        set(serialization) == {"lock_path", "journal_path", "expected_stat", "nonce"},
        "selected serialization contract",
    )
    require(
        isinstance(serialization["nonce"], str)
        and re.fullmatch(r"[0-9a-f]{32}", serialization["nonce"]),
        "campaign serialization nonce",
    )
    for key in ("lock_path", "journal_path"):
        require(
            str(ordinary(serialization[key])) == serialization[key],
            "absolute shared serialization path",
        )
    descriptor = plan.get("serialization_ref", {})
    require(
        set(descriptor) == {"path", "sha256", "size_bytes"}
        and selected.get(descriptor["path"]) == descriptor["sha256"]
        and base._decode(operation.read_ref(base, descriptor)) == serialization,
        "shared campaign descriptor requires an original selected byte pin",
    )


def wait_for_stop(base, plan, plan_sha, *, fatal, server, thread, deadline, sleep=time.sleep):
    while True:
        fatal.raise_if_set()
        deadline()
        require(
            thread.is_alive() and not server.should_exit, "service exited before authenticated stop"
        )
        if ordinary(plan["handoff"]["stop_path"]).exists():
            return read_stop(base, plan, plan_sha)
        sleep(0.02)


def serialized_provider(serial, provider, descriptor):
    """The existing serializer surrounds the actual provider, without a new answer path."""
    require(serial.SOURCE_ONLY is False, "serialization dependency remains held")
    lease = serial.ForwardLease(**descriptor)
    return serial.SerialBackend(provider, lease), lease


class ServiceAdmission:
    """Refuse new model admissions once shutdown starts; drain prior work."""

    def __init__(self, backend, stopping):
        self.backend, self.stopping = backend, stopping
        self.name, self.spec = backend.name, backend.spec

    def predict(self, request, **kwargs):
        require(not self.stopping.is_set(), "service stopping; no new native admission")
        return self.backend.predict(request, **kwargs)


def run(plan_path, plan_sha, owner_output):
    released()
    began = time.monotonic()

    def deadline():
        require(
            time.monotonic() - began < COOPERATIVE_SECONDS,
            "cooperative1200 service deadline; original owner hard7200 remains responsible for closure",
        )

    operation, base = bootstrap()
    raw_plan = base._read(plan_path)
    require(base._sha(raw_plan) == plan_sha, "selected service plan bytes differ")
    plan = base._decode(raw_plan)
    require(
        plan.get("owner_output") == str(Path(owner_output).absolute()),
        "selected original owner output",
    )
    owner, boundary, config, admission, process = base._owner_binding(owner_output)
    validate_plan(operation, base, plan, config, plan_path, plan_sha)
    require(
        not any(
            ordinary(plan["handoff"][key]).exists()
            for key in ("ready_path", "stop_path", "closed_path")
        ),
        "fresh service handoffs required before model load",
    )
    bridge = base._import_source(
        "_v7_http_full_choice_gateway", plan["bridge"]["path"], plan["bridge"]["sha256"]
    )
    require(bridge.SOURCE_ONLY is False, "full-choice bridge remains held")
    native = base._import_source(
        "_v7_http_original_native05", base.NATIVE / "actor.py", base.NATIVE_SHA
    )
    helper = native.load_helper()
    native_plan = native.load_plan(helper, base.NATIVE / "edge-plan.json", base.NATIVE_PLAN_SHA)
    output = base._new_output(Path(owner_output) / "v7-full-choice-service")
    fatal, primary = base.FatalLatch(), None
    stopping = threading.Event()
    state = {"model_loads": 0}
    listener = server = thread = runtime = torch = lease = None
    capability = delta = artifact_proof = None
    stopped = False
    receipt = {
        "schema": "jev-v7-full-choice-service-receipt-v1",
        "profile": PROFILE,
        **binding(plan, plan_sha),
        "owner_output": str(Path(owner_output).absolute()),
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
        "browser_qualified": False,
        "task_success": None,
        "service_success": False,
        "stage": "bound_before_model",
        "worker": process,
        "native_state": state,
        "cooperative_budget_seconds": COOPERATIVE_SECONDS,
        "owner_hard_runtime_seconds": OWNER_HARD_SECONDS,
        "helper_model_loaded": False,
        "helper_model_calls": 0,
        "cleanup_errors": [],
        "pending_native_or_server": False,
        "whole_tree_exit_confirmed": False,
        "external_current_owner_closure_required": True,
        "no_escape_enforced": False,
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
        try:
            deadline()
            _, _, current, current_admission, current_process = base._owner_binding(
                owner_output, loaded=(owner, boundary)
            )
            require(
                current == config
                and current_admission == admission
                and owner.same_identity(current_process, process),
                "original worker/admission changed",
            )
            validate_plan(operation, base, plan, current, plan_path, plan_sha)
            require(base._sha(base._read(plan_path)) == plan_sha, "service plan changed")
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
        receipt["stage"] = "loading_one_original_native_runtime"
        save(receipt)
        runtime, torch = native.load_runtime(helper, native_plan, output, state)
        require(state["model_loads"] == 1, "one original model load per service case")
        deadline()
        capability, delta, artifact_proof = operation.load_candidate(
            base, plan, native_plan["model"], torch, runtime
        )
        receipt["artifact_proof"] = artifact_proof
        adapter = base._import_source(
            "_v7_http_operation_adapter", plan["adapter"]["path"], plan["adapter"]["sha256"]
        )
        backend = base._import_source(
            "_v7_http_operation_backend",
            plan["delta_backend"]["path"],
            plan["delta_backend"]["sha256"],
        )
        observer = base._import_source(
            "_v7_http_same_forward_observer", plan["observer"]["path"], plan["observer"]["sha256"]
        )
        serial = base._import_source(
            "_v7_http_shared_serialization",
            plan["serial_forward"]["path"],
            plan["serial_forward"]["sha256"],
        )
        require(
            adapter.SOURCE_ONLY is False
            and backend.SOURCE_ONLY is False
            and serial.SOURCE_ONLY is False,
            "held dependency",
        )
        sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
        deps = operation.dependencies()
        provider = adapter.CandidateProvider(
            spec=deps["spec_factory"]("d3-edge", device="cpu"),
            native=native,
            observer=observer,
            delta_backend=backend,
            runtime=runtime,
            torch=torch,
            model=native_plan["model"],
            delta=delta,
            capability=capability,
            capability_sha256=plan["candidate_capability"]["sha256"],
            native_payload=deps["native_payload"],
            verify_owned=release.verify,
        )
        # Composition wraps the actual CandidateProvider.predict call. Its lock
        # wait, forward, retention, CPU completion and journal fsync use the
        # unchanged adapter's original whole-call deadline callback.
        provider, lease = serialized_provider(serial, provider, plan["serialization"])
        identity = {
            "profile": PROFILE,
            "candidate_alias": ALIAS,
            "only_adapted_head_id": "operation",
            "publisher_native_numerics": False,
            "diagnostic_only": True,
            "dev_used_for_model_selection": True,
            "optimizer_input_dev": False,
            "optimizer_input_sealed": False,
            "candidate_selection_before_dev_or_sealed": False,
            "new_unseen_evaluation_claim": False,
            "readout_adaptation_applied": True,
            "wire_copies_candidate_probabilities_without_change": True,
            "candidate_capability": copy.deepcopy(plan["candidate_capability"]),
            "numeric_acceptance": copy.deepcopy(plan["numeric_acceptance"]),
            "native_model": native_plan["model"],
            "device": "cpu",
            "precision": "bfloat16",
            "context_limit": 16384,
            "cpu_affinity": [15, 16, 17, 18, 19],
            "memory_max_bytes": 4294967296,
            "swap_max_bytes": 0,
            "cpu_quota_percent": 500,
            "expected_threads": 5,
            "expected_interop_threads": 1,
            "owner_unit": base.UNIT,
            "owner_profile": base.OWNER_PROFILE,
            "nonce": config["nonce"],
            "invocation_id": admission["invocation_id"],
            "pid": os.getpid(),
            "start_ticks": process["process"]["start_ticks"],
            "legacy_v1_scope": "DTO/wire ABI only; adapted CPU candidate, no CUDA qualification",
        }
        wrapped = []

        def guard(value):
            result = base.FatalBackend(ServiceAdmission(value, stopping), fatal)
            wrapped.append(result)
            return result

        server_output = base._new_output(output / "server-evidence")
        app = adapter.create_application(
            bridge=bridge,
            owned_release=release,
            provider=provider,
            dependencies=deps,
            output=server_output,
            cpu_identity=identity,
            backend_wrapper=guard,
        )
        require(
            len(wrapped) == 1 and app.backend is wrapped[0],
            "actual gateway uses original fatal backend guard",
        )
        app = base.GuardedASGI(app, fatal)
        import uvicorn

        base._verify_imports(plan)
        deadline()
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        endpoint = "http://127.0.0.1:" + str(listener.getsockname()[1]) + "/v1/systemone"
        server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", workers=1, lifespan="off", log_level="error")
        )

        def serve():
            try:
                require(
                    sorted(os.sched_getaffinity(0)) == [15, 16, 17, 18, 19]
                    and torch.get_num_threads() == 5
                    and torch.get_num_interop_threads() == 1
                    and not torch.cuda.is_initialized(),
                    "actual ASGI thread CPU binding",
                )
                native.validate_runtime(runtime, torch, native_plan["model"])
                server.run(sockets=[listener])
                require(server.should_exit, "unexpected HTTP server exit")
            except BaseException as error:
                fatal.capture(error)

        thread = threading.Thread(target=serve, name="owned-v7-full-choice-http", daemon=False)
        thread.start()
        ready_until = time.monotonic() + 15
        while not server.started:
            fatal.raise_if_set()
            deadline()
            require(thread.is_alive() and time.monotonic() < ready_until, "bounded HTTP readiness")
            time.sleep(0.02)
        current_binding = release.verify()
        receipt["endpoint"] = endpoint
        receipt["stage"] = "http_ready_waiting_for_selected_stop"
        save(receipt)
        publish(
            plan["handoff"]["ready_path"],
            {
                "schema": READY_SCHEMA,
                **binding(plan, plan_sha),
                "endpoint": endpoint,
                "model": ALIAS,
                "owner_output": str(Path(owner_output).absolute()),
                "worker_identity": process,
                "verify_owned": current_binding,
            },
        )
        deadline()
        receipt["authenticated_stop"] = wait_for_stop(
            base, plan, plan_sha, fatal=fatal, server=server, thread=thread, deadline=deadline
        )
        stopped = True
    except BaseException as error:
        fatal.capture(error)
        primary = fatal.first
    finally:
        stopping.set()
        if server is not None:
            try:
                server.should_exit = True
            except BaseException as error:
                cleanup_failure("request_server_stop", error)
        drained = thread is None
        if thread is not None:
            try:
                thread.join(timeout=10)
                drained = not thread.is_alive()
                require(drained, "server/native thread live; original owner must close cohort")
            except BaseException as error:
                receipt["pending_native_or_server"] = True
                cleanup_failure("drain_server", error)
        receipt["server_thread_drained"] = drained
        if listener is not None and drained:
            try:
                listener.close()
            except BaseException as error:
                cleanup_failure("close_listener", error)
        if runtime is not None and drained:
            try:
                fatal.raise_if_set()
                if lease is not None:
                    lease.check()
                    require(lease.fd is None, "no unfinished actual forward lease")
                native.validate_runtime(runtime, torch, native_plan["model"])
                native.authenticate(helper, native_plan)
                require(artifact_proof is not None, "accepted artifact initialization completed")
                require(
                    operation.tensor_sha(torch, runtime.readout)
                    == artifact_proof["baseline_fp32_bytes_sha256"]
                    and operation.tensor_sha(torch, delta)
                    == artifact_proof["delta_fp32_bytes_sha256"],
                    "W0/delta bytes unchanged",
                )
                for record in (
                    plan["candidate_capability"],
                    plan["numeric_acceptance"],
                    capability["root_acceptance_ref"],
                    capability["delta_ref"],
                ):
                    operation.read_ref(base, record)
                verify_owned()
                receipt["final_native_artifact_authentication"] = True
            except BaseException as error:
                cleanup_failure("final_native_artifact_authentication", error)
        try:
            owner.sources_match(config["sources"])
            base._verify_imports(plan)
            deadline()
            receipt["sources_unchanged"] = True
        except BaseException as error:
            cleanup_failure("final_sources", error)
        if fatal.first is not None and primary is None:
            primary = fatal.first
        receipt["service_success"] = bool(
            primary is None
            and stopped
            and drained
            and receipt.get("final_native_artifact_authentication")
            and receipt.get("sources_unchanged")
        )
        receipt["stage"] = (
            "service_complete_pending_external_owner_closure"
            if receipt["service_success"]
            else "service_failed_pending_external_owner_closure"
        )
        receipt["elapsed_seconds_before_final_write"] = time.monotonic() - began
        if primary is not None:
            receipt["error"] = base._failure(primary)
        try:
            save(receipt)
            deadline()
        except BaseException as error:
            cleanup_failure("final_receipt", error)
            receipt["service_success"] = False
        # This is local termination evidence only. The root pump alone can
        # publish its distinct closed.json after original owner closure passes.
        try:
            publish(
                plan["handoff"]["closed_path"],
                {
                    "schema": CLOSED_SCHEMA,
                    **binding(plan, plan_sha),
                    "owner_output": str(Path(owner_output).absolute()),
                    "service_success": receipt["service_success"],
                    "server_thread_drained": drained,
                    "final_native_artifact_authentication": receipt.get(
                        "final_native_artifact_authentication", False
                    ),
                    "sources_unchanged": receipt.get("sources_unchanged", False),
                    "worker_receipt": str(output / "worker-receipt.json"),
                    "external_current_owner_closure_required": True,
                    "whole_tree_exit_confirmed": False,
                    "task_success": None,
                },
            )
            deadline()
        except BaseException as error:
            cleanup_failure("service_closed_handoff", error)
            receipt["service_success"] = False
        if primary is not None:
            receipt["error"] = base._failure(primary)
            receipt["stage"] = "service_failed_pending_external_owner_closure"
            try:
                save(receipt)
            except BaseException as error:
                cleanup_failure("bounded_failure_receipt", error)
    if primary is not None:
        raise primary
    return receipt


def main(argv=None):
    released()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--owner-output", required=True, type=Path)
    args = parser.parse_args(argv)
    run(args.plan, args.plan_sha256, args.owner_output)


if __name__ == "__main__":
    main()
