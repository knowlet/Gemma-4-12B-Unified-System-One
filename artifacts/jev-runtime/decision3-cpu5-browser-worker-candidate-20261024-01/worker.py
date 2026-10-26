"""HELD source-only owned CPU/native/HTTP/Chromium wiring worker candidate.

No code in this candidate has been imported, checked, tested, or executed.
The default gate refuses before arguments/plan/dependencies/output/model work.
Existing owner09 remains the ONLY resource/cleanup authority. A separately
reviewed bridge successor is still required; held b4cf00a is never bypassed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import socket
import stat
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE_ONLY = False
PROFILE = "decision3-cpu5-fast-same-cluster-4g-owned-browser-worker-source-candidate-v1"
BRIDGE_BINDING = "owned-decision3-cpu5-browser-binding-v1"
OWNER = ROOT / "artifacts/jev-runtime/llama-cpp-owner-20261024-09"
OWNER_SHA = "81846efe2ee0357da07283c1a77a606283a920cbddef668c8f62def291057dee"
OWNER_PROFILE = "llama-cpp-cpu5-fast-same-cluster-4g-coexist-command-owned-v1"
UNIT = "jev-d3-llama-cpp-cpu5-fast-4g-coexist-20261024-09.service"
NATIVE = ROOT / "artifacts/jev-runtime/decision3-publisher-batch4-cpu-20261024-05"
NATIVE_SHA = "9a7e5af918c17f6824a720fb6d112ffa5fdac03493dde3efa176c7d50470a365"
NATIVE_PLAN_SHA = "095db88b78687d90fbed51aafd736c1b06fb1a582312de8dff97bf6109131e96"
PYTHON = Path("/home/albert_hsieh/ComfyUI/.venv/bin/python")
CALLER = ROOT / "tests/test_jev_native_http_browser.py"
MODEL_ALIAS = "d3-edge-cpu-toy"
FIXED_ENVIRONMENT = (
    "JEV_NATIVE_HTTP_BROWSER_TESTS",
    "JEV_NATIVE_HTTP_ENDPOINT",
    "JEV_NATIVE_HTTP_MODEL",
    "JEV_UPSTREAM_PATH",
    "JEV_CHROMIUM_EXECUTABLE",
    "JEV_NATIVE_HTTP_BROWSER_OUTPUT",
    "JEV_NATIVE_HTTP_API_KEY",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _release_gate():
    if SOURCE_ONLY:
        raise RuntimeError(
            "SOURCE_ONLY: no arguments, plan, imports, model, HTTP, or browser execution"
        )
    # A future review changes this source flag THEN pins those exact new bytes
    # in the original owner command selection. The subsequent live binding is
    # mandatory; no environment override or embedded circular plan hash exists.


def _ordinary(path):
    path = Path(path).absolute()
    require(
        ".." not in path.parts and path.is_relative_to(ROOT),
        "Owned ordinary repository path required",
    )
    for parent in (path, *path.parents):
        require(not parent.is_symlink(), "No symlink in selected source/output path")
    return path


def _read(path, maximum=1024 * 1024):
    path = _ordinary(path)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(before.st_mode) and 0 < before.st_size <= maximum,
            "Bounded regular metadata/source required",
        )
        data = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(
        len(data) == before.st_size
        and all(
            getattr(before, key) == getattr(after, key) == getattr(path.stat(), key)
            for key in fields
        ),
        "Selected bytes changed during read",
    )
    return data


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def _decode(data):
    return json.loads(
        data,
        object_pairs_hook=_pairs,
        parse_constant=lambda _value: require(False, "Nonfinite JSON refused"),
    )


def _json(path, maximum=1024 * 1024):
    return _decode(_read(path, maximum))


def _import_source(name, path, sha):
    require(
        name not in sys.modules and isinstance(sha, str) and len(sha) == 64,
        "Fresh explicit source SHA required",
    )
    path, raw = _ordinary(path), _read(path)
    require(_sha(raw) == sha and path.suffix == ".py", "Selected Python source differs")
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "Selected source loader unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        # Execute the exact bytes already authenticated, not a second path read.
        exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _failure(exc):
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "notes": list(getattr(exc, "__notes__", [])),
    }


def _phase_environment(helper, native_state):
    """Expected identity follows the original loader's explicit state transition."""
    count = native_state.get("model_loads")
    require(
        type(count) is int and count in (0, 1),
        "Only original pre-load0 or loaded1 environment phase is supported",
    )
    if count == 0:
        return dict(helper.START_ENV)
    return {**helper.START_ENV, **helper.CPU_ENV}


class FatalLatch:
    """Keep the original object/traceback across ASGI -> caller/main threads."""

    def __init__(self):
        self.lock = threading.Lock()
        self.event = threading.Event()
        self.first = None

    def capture(self, error):
        with self.lock:
            if self.first is None:
                self.first = error
            elif self.first is not error:
                self.first.add_note("Additional owned worker error: " + repr(error))
            self.event.set()

    def raise_if_set(self):
        with self.lock:
            error = self.first
        if error is not None:
            raise error


class FatalBackend:
    """Guard only genuine backend invocation; never rewrite a request/answer."""

    def __init__(self, backend, fatal):
        self.backend, self.fatal = backend, fatal
        self.spec, self.name = backend.spec, backend.name

    def predict(self, request, **kwargs):
        self.fatal.raise_if_set()  # No subsequent native forward after an abort.
        try:
            return self.backend.predict(request, **kwargs)
        except BaseException as error:
            self.fatal.capture(error)
            raise


class GuardedASGI:
    """Retain a BaseException even if Uvicorn converts it to an HTTP error."""

    def __init__(self, application, fatal):
        self.application, self.fatal = application, fatal

    async def __call__(self, scope, receive, send):
        # Do not reject ahead of the bridge's raw request recorder. Its wrapped
        # backend blocks the next NN, while subsequent attempts remain evidence.
        try:
            return await self.application(scope, receive, send)
        except BaseException as error:
            self.fatal.capture(error)
            raise


def _wait_json(path, seconds, fatal=None):
    until = time.monotonic() + seconds
    while not path.exists():
        if fatal is not None:
            fatal.raise_if_set()
        require(time.monotonic() < until, "Bounded original command-started handoff timed out")
        time.sleep(0.01)
    return _json(path)


def _owner_binding(owner_output, *, loaded=None):
    """Consume the ORIGINAL owner artifacts; never create replacement admission."""
    owner_output = _ordinary(owner_output)
    require(owner_output.parent == OWNER, "Only selected owner09 output accepted")
    if loaded is None:
        owner = _import_source(
            "_owned_cpu_browser_original_controller", OWNER / "controller.py", OWNER_SHA
        )
        boundary = owner.load_fixed("boundary")
    else:
        owner, boundary = loaded
    config_path = owner_output / "sealed-config.json"
    config_bytes = _read(config_path)
    config, config_sha = _decode(config_bytes), _sha(config_bytes)
    admission = _json(owner_output / "admission.json")
    require(
        config["profile"] == OWNER_PROFILE
        and config["unit"] == UNIT
        and config["output"] == str(owner_output)
        and config["command_count"] == 1
        and config["boot_id"] == boundary.boot()
        and os.environ.get("JEV_COMMAND_NONCE") == config["nonce"]
        and admission["config_sha256"] == config_sha
        and admission["nonce"] == config["nonce"],
        "Original owner config/admission/nonce/boot binding differs",
    )
    owner.sources_match(config["sources"])
    selection = owner.command_selection(
        config["selection"], config["selection_sha256"], config["base_sources"]
    )
    require(len(selection["commands"]) == 1, "Exactly one owned worker command required")
    actual = owner.process_identity(boundary, os.getpid())
    wrapper = admission["actor"]
    require(
        owner.stable_process(boundary.process(wrapper["pid"])) == owner.stable_process(wrapper),
        "Original admitted wrapper no longer matches",
    )
    require(
        all(actual["process"][key] == wrapper[key] for key in ("pgid", "sid", "uid", "cgroup")),
        "Worker must belong to admitted wrapper cohort",
    )
    require(
        actual["process"]["pid"] != wrapper["pid"],
        "Command worker is not the admission wrapper PID",
    )
    started = _wait_json(owner_output / "command-01-started.json", 10)
    intent = _json(owner_output / "command-01-intent.json")
    expected = selection["commands"][0]
    expected_argv = [value.replace("{OUTPUT}", str(owner_output)) for value in expected["argv"]]
    require(
        started["pid"] == os.getpid() and owner.same_identity(started["observed_identity"], actual),
        "Worker PID/start/executable differs from original command-started observation",
    )
    require(
        actual["argv"] == intent["argv"] == expected_argv
        and actual["cwd"]
        == intent["cwd"]
        == expected["cwd"].replace("{OUTPUT}", str(owner_output))
        == str(ROOT)
        and actual["exe"] == intent["resolved_executable"] == str(PYTHON.resolve(strict=True))
        and intent["shell"] is False,
        "Original exact worker argv/cwd/ComfyPython join differs",
    )
    require(
        sys.flags.no_user_site and sys.dont_write_bytecode and sys.flags.optimize == 0,
        "Fresh -B -s without -O is required; fixture assertions must remain active",
    )
    return owner, boundary, config, admission, actual


def _new_output(path):
    path = _ordinary(path)
    path.mkdir(mode=0o700, exist_ok=False)
    fd = os.open(path / ".gitignore", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(b"*\n")
        stream.flush()
        os.fsync(stream.fileno())
    return path


def _source_contract(plan, config, plan_path, plan_sha):
    require(
        plan.get("profile") == PROFILE and plan.get("status") == "selected_after_separate_review",
        "Pending/other worker plan is not a selection",
    )
    require(
        plan.get("native_actor") == {"path": str(NATIVE / "actor.py"), "sha256": NATIVE_SHA}
        and plan.get("native_plan")
        == {"path": str(NATIVE / "edge-plan.json"), "sha256": NATIVE_PLAN_SHA},
        "Selected native05 actor/plan required; no CPU4/CPU16 relabeling",
    )
    require(
        plan.get("owner_profile") == OWNER_PROFILE and plan.get("owner_unit") == UNIT,
        "Worker plan must bind the original owner resource profile",
    )
    require(
        plan.get("model_alias") == MODEL_ALIAS
        and plan.get("max_ticks") == 16
        and plan.get("http_caller_timeout_seconds") == 25
        and plan.get("http_cooperative_budget_seconds") == 20
        and plan.get("helper_model") is None,
        "Original toy/caller budgets and helper0 scope required",
    )
    # Every import/payload-producing source must be a member of the original
    # selected command union; new pins cannot be invented after model loading.
    selected = {str(ROOT / row["path"]): row["sha256"] for row in config["sources"]}
    require(
        selected.get(str(Path(plan_path).absolute())) == plan_sha,
        "Plan itself must have the original command source pin",
    )
    for key in ("worker", "bridge_successor", "caller"):
        record = plan[key]
        require(
            type(record) is dict
            and set(record) == {"path", "sha256"}
            and selected.get(record["path"]) == record["sha256"]
            and _sha(_read(record["path"])) == record["sha256"],
            "Selected worker/bridge/caller source pin missing: " + key,
        )
    require(
        plan["worker"]["path"] == str(Path(__file__).absolute())
        and plan["caller"]["path"] == str(CALLER)
        and plan["bridge_successor"]["path"] == str(HERE / "bridge_successor.py"),
        "Only this worker, adjacent explicit successor and original actual caller accepted",
    )
    owner_source = next(
        row for row in config["sources"] if str(ROOT / row["path"]) == str(OWNER / "controller.py")
    )
    require(owner_source["sha256"] == OWNER_SHA, "Original owner controller pin required")
    require(
        type(plan.get("import_sources")) is list and bool(plan["import_sources"]),
        "Full explicit imported source closure remains pending",
    )
    for record in plan["import_sources"]:
        require(
            set(record) == {"module", "path", "sha256"}
            and selected.get(record["path"]) == record["sha256"]
            and _sha(_read(record["path"])) == record["sha256"],
            "Imported source must have the original command source pin",
        )


def _verify_imports(plan):
    listed = {record["module"] for record in plan["import_sources"]}
    require(
        all(name in listed for name in sys.modules if name == "s1" or name.startswith("s1.")),
        "Every actually imported s1 module requires an explicit selected source record",
    )
    for record in plan["import_sources"]:
        module = sys.modules.get(record["module"])
        if module is not None:
            require(
                Path(module.__file__).resolve(strict=True)
                == Path(record["path"]).resolve(strict=True),
                "Actual imported module origin differs: " + record["module"],
            )
        require(
            _sha(_read(record["path"])) == record["sha256"],
            "Imported source changed during worker lifetime",
        )


def _bind_bridge(
    bridge, release, plan_path, plan, native, native_plan, runtime, torch, output, fatal, identity
):
    wrapped = []

    def backend_wrapper(backend):
        result = FatalBackend(backend, fatal)
        wrapped.append(result)
        return result

    # Implemented in the adjacent explicit successor, not in held b4cf00a.
    # No private __new__, gate monkeypatch, SOURCE_ONLY mutation or fake lease.
    application = bridge.create_application(
        plan_path,
        owned_release=release,
        actor=native,
        runtime=runtime,
        torch=torch,
        model=native_plan["model"],
        model_identity=MODEL_ALIAS,
        output=output,
        cpu_identity=identity,
        backend_wrapper=backend_wrapper,
    )
    require(
        len(wrapped) == 1 and application.backend is wrapped[0],
        "Every native invocation must pass the fatal guard",
    )
    require(
        application.backend.spec.device == "cpu"
        and application.backend.spec.precision == "bfloat16"
        and application.backend.spec.context_limit == 16384,
        "Honest CPU/BF16/context16384 spec required",
    )
    from s1.jev_decision3_native import JevDecision3NativeAdapter

    require(
        application.adapter_type.predict is JevDecision3NativeAdapter.predict,
        "Original native predict must be inherited verbatim",
    )
    return GuardedASGI(application, fatal)


def _caller(caller, endpoint, plan, browser_output, fatal):
    """Call the unchanged real HTTP -> choose -> tick -> guarded DOM test path.

    Importing pytest happens only with this source AFTER the one native load;
    pytest.main/plugins/fixtures/subprocesses are never started by the worker.
    No chooser/Agent/HTTP timeout/action/policy is patched here.
    """
    saved = {key: os.environ.get(key) for key in FIXED_ENVIRONMENT}
    values = {
        "JEV_NATIVE_HTTP_BROWSER_TESTS": "1",
        "JEV_NATIVE_HTTP_ENDPOINT": endpoint,
        "JEV_NATIVE_HTTP_MODEL": MODEL_ALIAS,
        "JEV_UPSTREAM_PATH": plan["upstream_path"],
        "JEV_CHROMIUM_EXECUTABLE": plan["chromium_executable"],
        "JEV_NATIVE_HTTP_BROWSER_OUTPUT": str(browser_output),
        "JEV_NATIVE_HTTP_API_KEY": "",
    }
    try:
        os.environ.update(values)
        fatal.raise_if_set()
        try:
            caller.test_real_native_http_choose_agent_and_guarded_click()
        except BaseException as error:
            if fatal.event.is_set():
                fatal.capture(error)
                fatal.raise_if_set()
            raise
        fatal.raise_if_set()
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _http_closure(report, server_output):
    """Join actual wire bytes/IDs after drain; never use these data as policy."""
    expected, joined = {}, []
    for row in report["http"]:
        request_id = row.get("server_request_id")
        require(
            type(request_id) is str
            and len(request_id) == 32
            and all(value in "0123456789abcdef" for value in request_id)
            and request_id not in expected,
            "Every actual HTTP response requires its generated unique server ID",
        )
        expected[request_id] = row
    require(expected, "No actual native HTTP response evidence")
    directories = {path.name: path for path in server_output.iterdir() if path.is_dir()}
    require(
        set(directories) == set(expected),
        "Unexpected/unreturned server request; evidence remains unclosed",
    )
    for request_id, row in expected.items():
        directory = directories[request_id]
        exchange = _json(directory / "exchange.json")
        raw_request = _read(directory / "request.bin")
        raw_response = _read(directory / "response.bin", maximum=256 * 1024 * 1024)
        require(
            exchange["request_id"] == request_id
            and exchange["request_body_complete"] is True
            and exchange["status"] == row["status"]
            and exchange["body_sha256"] == _sha(raw_request) == row["request_sha256"]
            and exchange["response_sha256"] == _sha(raw_response) == row["response_sha256"]
            and exchange["observer_errors"] == [],
            "HTTP/client/server byte or terminal evidence differs",
        )
        joined.append(
            {
                "request_id": request_id,
                "status": exchange["status"],
                "request_sha256": row["request_sha256"],
                "response_sha256": row["response_sha256"],
            }
        )
    return {
        "status": "all_actual_client_server_ids_and_bytes_joined",
        "requests": joined,
        "scope": "Original HTTP responses only; not a network delivery timing guarantee",
    }


def run(plan_path, expected_plan_sha, owner_output):
    _release_gate()  # BEFORE plan read, import, output, resource or model work.
    raw_plan = _read(plan_path)
    require(
        isinstance(expected_plan_sha, str)
        and len(expected_plan_sha) == 64
        and _sha(raw_plan) == expected_plan_sha,
        "Exact reviewed worker plan required",
    )
    plan = _decode(raw_plan)
    owner, boundary, config, admission, process = _owner_binding(owner_output)
    _source_contract(plan, config, plan_path, expected_plan_sha)
    bridge = _import_source(
        "_owned_cpu_browser_bridge",
        plan["bridge_successor"]["path"],
        plan["bridge_successor"]["sha256"],
    )
    require(
        getattr(bridge, "OWNED_WORKER_BINDING_VERSION", None) == BRIDGE_BINDING,
        "Only an explicit reviewed owned-binding successor is supported; held b4cf00a cannot be bypassed",
    )
    native = _import_source("_owned_cpu_browser_native05", NATIVE / "actor.py", NATIVE_SHA)
    helper = native.load_helper()
    native_plan = native.load_plan(helper, NATIVE / "edge-plan.json", NATIVE_PLAN_SHA)
    output = _new_output(Path(owner_output) / "cpu-browser-wiring")
    fatal, primary = FatalLatch(), None
    listener = thread = server = runtime = torch = report = None
    native_state = {"model_loads": 0}
    receipt = {
        "profile": PROFILE,
        "stage": "bound_before_model_load",
        "source_only": False,
        "owner": {
            "unit": UNIT,
            "profile": OWNER_PROFILE,
            "nonce": config["nonce"],
            "boot_id": config["boot_id"],
            "invocation_id": admission["invocation_id"],
        },
        "worker": process,
        "worker_plan_sha256": expected_plan_sha,
        "native_constructor_attempts": 0,
        "native_state": native_state,
        "helper_model_loaded": False,
        "helper_model_calls": 0,
        "browser_call_finished": False,
        "browser_cleanup_scope": "Original caller finally; external cohort closure is mandatory",
        "server_thread_drained": False,
        "pending_native_or_server": False,
        "cleanup_errors": [],
        "success": False,
        "external_current_owner_closure_required": True,
        "whole_tree_exit_confirmed": False,
        "no_escape_enforced": False,
        "model_qualified": False,
    }

    def save():
        boundary.save(output, "worker-receipt.json", receipt)

    def cleanup_failure(stage, error):
        nonlocal primary
        receipt["cleanup_errors"].append({"stage": stage, **_failure(error)})
        if primary is None:
            primary = error
        elif primary is not error:
            primary.add_note(stage + " also failed: " + repr(error))

    def verify_owned_selection():
        try:
            _, _, current_config, current_admission, current_process = _owner_binding(
                owner_output, loaded=(owner, boundary)
            )
            require(
                current_config == config
                and current_admission == admission
                and owner.same_identity(current_process, process),
                "Original worker/admission binding changed",
            )
            _source_contract(plan, current_config, plan_path, expected_plan_sha)
            require(_sha(_read(plan_path)) == expected_plan_sha, "Selected plan changed")
            current_cpu = helper.process_identity()
            native.validate_owner(
                current_cpu,
                helper.owned_limits(current_cpu),
                _phase_environment(helper, native_state),
            )
            return {
                "schema": BRIDGE_BINDING,
                "plan_path": str(Path(plan_path).absolute()),
                "plan_sha256": expected_plan_sha,
                "config_sha256": _sha(_read(Path(owner_output) / "sealed-config.json")),
                "selection_sha256": config["selection_sha256"],
                "owner_unit": UNIT,
                "owner_profile": OWNER_PROFILE,
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
        save()
        release = bridge.bind_owned_selection(verify_owned_selection)
        # Original selected_executable authenticates stats, not a new PATH or
        # external CDP browser. All children are launched by this owned worker.
        owner.selected_executable(plan["chromium_executable"], config["binary_stats"])
        receipt["stage"] = "loading_one_original_native_runtime"
        receipt["native_constructor_attempts"] = 1
        save()
        runtime, torch = native.load_runtime(helper, native_plan, output, native_state)
        require(
            native_state["model_loads"] == 1, "Exactly one successful original constructor required"
        )
        receipt["stage"] = "native_loaded_before_browser_imports"
        save()
        # The original loader has already authenticated source/kernel/runtime
        # identity and set CPU threads/env before any service/browser threads.
        sys.path.insert(0, str(ROOT / "src"))
        sys.path.insert(0, str(ROOT))
        identity = {
            "device": "cpu",
            "precision": "bfloat16",
            "context_limit": 16384,
            "profile": PROFILE,
            "owner_unit": UNIT,
            "owner_profile": OWNER_PROFILE,
            "nonce": config["nonce"],
            "invocation_id": admission["invocation_id"],
            "pid": os.getpid(),
            "start_ticks": process["process"]["start_ticks"],
            "native_model": native_plan["model"],
            "legacy_v1_scope": "ABI only; no CUDA qualification",
        }
        server_output = _new_output(output / "server-evidence")
        application = _bind_bridge(
            bridge,
            release,
            plan_path,
            plan,
            native,
            native_plan,
            runtime,
            torch,
            server_output,
            fatal,
            identity,
        )
        caller = _import_source(
            "_owned_cpu_browser_original_caller", CALLER, plan["caller"]["sha256"]
        )
        import uvicorn

        _verify_imports(plan)
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))  # Retain this socket; never attach to an external endpoint.
        endpoint = "http://127.0.0.1:" + str(listener.getsockname()[1]) + "/v1/systemone"
        server = uvicorn.Server(
            uvicorn.Config(
                application, host="127.0.0.1", workers=1, lifespan="off", log_level="error"
            )
        )
        thread_identity = {}

        def serve():
            try:
                require(
                    os.sched_getaffinity(0) == {15, 16, 17, 18, 19}
                    and torch.get_num_threads() == 5
                    and torch.get_num_interop_threads() == 1
                    and not torch.cuda.is_initialized(),
                    "Actual ASGI thread must preserve selected CPU5 policy",
                )
                native.validate_runtime(runtime, torch, native_plan["model"])
                thread_identity.update(
                    pid=os.getpid(),
                    native_thread_id=threading.get_native_id(),
                    affinity=sorted(os.sched_getaffinity(0)),
                    intraop=torch.get_num_threads(),
                    interop=torch.get_num_interop_threads(),
                    cuda_initialized=False,
                )
                server.run(sockets=[listener])
                if not server.should_exit:
                    raise RuntimeError("Owned ASGI thread exited without an authorized shutdown")
            except BaseException as error:
                fatal.capture(error)

        # A stuck non-daemon NN cannot silently disappear on a normal return.
        # Failed drain stays pending until the ORIGINAL owner closes the unit.
        thread = threading.Thread(target=serve, name="owned-cpu-native-asgi", daemon=False)
        receipt["stage"] = "starting_one_loopback_asgi_thread"
        save()
        thread.start()
        until = time.monotonic() + 15
        while not server.started:
            fatal.raise_if_set()
            require(thread.is_alive() and time.monotonic() < until, "Owned ASGI readiness failed")
            time.sleep(0.02)
        receipt.update(
            stage="original_two_control_caller", endpoint=endpoint, server_thread=thread_identity
        )
        save()
        _caller(caller, endpoint, plan, output / "browser", fatal)
        receipt["browser_call_finished"] = True
        report = _json(output / "browser/report.json", maximum=256 * 1024 * 1024)
        require(
            report.get("success") is True
            and all(row.get("kind") == "policy" for row in report["records"])
            and report.get("final_snapshot", {}).get("text_calls") == [],
            "Original strict judge/helper0 caller did not succeed",
        )
        receipt["browser_report_sha256"] = _sha(
            _read(output / "browser/report.json", maximum=256 * 1024 * 1024)
        )
        receipt["stage"] = "browser_finished_pending_service_and_source_cleanup"
    except BaseException as error:
        primary = fatal.first if fatal.event.is_set() else error
        if primary is not error:
            primary.add_note("Caller/worker also failed: " + repr(error))
    finally:
        # The unchanged caller has already completed its browser/Agent finally
        # before it returns/raises. Do not shut down service ahead of that path.
        try:
            if server is not None:
                server.should_exit = True
        except BaseException as error:
            cleanup_failure("server_shutdown_request", error)
        drained = thread is None
        try:
            if thread is not None and thread.ident is not None:
                thread.join(timeout=10)
                drained = not thread.is_alive()
            elif thread is not None:
                drained = True  # start failed before a server thread existed.
            receipt["server_thread_drained"] = drained
            if not drained:
                receipt["pending_native_or_server"] = True
                raise RuntimeError(
                    "ASGI/NN thread still live after10s; original owned close required"
                )
        except BaseException as error:
            drained = False
            cleanup_failure("server_thread_drain", error)
        if listener is not None and drained:
            try:
                listener.close()
            except BaseException as error:
                cleanup_failure("listener_close", error)
        if report is not None and drained:
            try:
                receipt["http_evidence_closure"] = _http_closure(report, output / "server-evidence")
            except BaseException as error:
                cleanup_failure("http_evidence_closure", error)
        # Bridge removes each per-request row hook in its existing finally.
        # Never remove an in-use hook or release a model from a live NN thread.
        if runtime is not None and drained:
            try:
                fatal.raise_if_set()
                native.validate_runtime(runtime, torch, native_plan["model"])
                native.authenticate(helper, native_plan)
                receipt["final_native_authentication"] = True
            except BaseException as error:
                cleanup_failure("final_native_authentication", error)
        try:
            owner.sources_match(config["sources"])
            _verify_imports(plan)
            receipt["sources_unchanged"] = True
        except BaseException as error:
            cleanup_failure("source_closure", error)
            receipt["sources_unchanged"] = False
        if fatal.event.is_set() and primary is None:
            primary = fatal.first
        receipt["success"] = bool(
            primary is None
            and receipt["browser_call_finished"]
            and drained
            and receipt.get("final_native_authentication") is True
            and receipt.get("sources_unchanged") is True
            and receipt.get("http_evidence_closure", {}).get("status")
            == "all_actual_client_server_ids_and_bytes_joined"
        )
        receipt["stage"] = (
            "complete_pending_original_external_owner_closure"
            if receipt["success"]
            else "failed_or_pending_no_wiring_success"
        )
        receipt["error"] = None if primary is None else _failure(primary)
        try:
            save()
        except BaseException as error:
            if primary is None:
                primary = error
            else:
                primary.add_note("Final worker receipt also failed: " + repr(error))
    if primary is not None:
        raise primary
    return receipt


def main(argv=None):
    _release_gate()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--owner-output", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.plan, args.plan_sha256, args.owner_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
