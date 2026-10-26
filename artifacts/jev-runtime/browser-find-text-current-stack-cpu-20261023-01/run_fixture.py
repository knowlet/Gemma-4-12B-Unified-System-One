"""Owned localhost CPU current-stack proof; every model number is simulated."""

from __future__ import annotations

import argparse
import ast
import copy
import errno
import hashlib
import importlib.util
import json
import math
import os
import re
import resource
import socket
import sys
import threading
import time
import types
from contextlib import ExitStack, contextmanager, nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).absolute().parents[3]
HERE = Path(__file__).absolute().parent
UPSTREAM = ROOT / "artifacts/jev-runtime/upstream"
PINS = {
    "artifacts/jev-runtime/browser-find-text-v7-client-budget-20261023-02/find_client_selector.py": "f56f0800a981370f5a0e74d8722677721cb2043b03596c7779227502d4565ee1",
    "artifacts/jev-runtime/browser-find-text-s1-nativev2-integration-20261022-01/selection.py": "c3712f8d29851a521a3e9b5a1691d3fb8cf7ae9f82aeb1051c27aa8c64b04c1a",
    "artifacts/jev-runtime/browser-find-text-s1-nativev2-integration-20261022-01/selection.json": "b54a709873c6c3a6ed41894c5fb12aca0ba806791afa718820578346e0ee5d14",
    "artifacts/jev-runtime/browser-find-text-s1-nativev2-integration-20261022-01/test_selection.py": "3d4a18a9399cfcc98bba3e76efab377af32c7d41bcfac25490b734d31806cc14",
    "artifacts/jev-runtime/browser-find-text-cpu-browser-20261022-01/run_fixture.py": "67bfd8a83068c3d0fa6ea8dc3c5f4edd22f3834eb2a1dab21fbb6dcbd2d01bbc",
    "scripts/jev_qualify_public_field_syntax_bf16.py": "cbfb7bdd8301174a97ac18389a998ff3256ac32d6f7a770978f2f9b722c3b43b",
    "scripts/jev_browser_benchmark.py": "fe1410675e4decb06a8491a5a83d2ec965629c99f8912b797446c4cb2c155f6c",
    "scripts/jev_observation_profile_v7.py": "d9f4236d1d46c9472c3d6c1551342639ecb870e6c0dd419246927645410ea868",
    "scripts/jev_browser_profile_v7_benchmark.py": "31450f9002ad216c32b77751997eb27748c491d7bc757c4410294d3704fdfb9e",
    "scripts/jev_public_web_cases.py": "0ec9826d334cb2fbed24ade8780fa785291017d904b18233419ced760750bca3",
    "src/s1/jev_observation_profile.py": "8d199ac1564af05892e9b21e225d1fdf500f1953840855c902160ed8d56df89d",
    "src/s1/jev_observation_events.py": "ee6d3ea04d948fd8947d12934b63564494aab6a67cb8d8063ea7e12fb84a0664",
    "artifacts/jev-runtime/browser-find-text-v7-client-integration-20261023-01/find_client_selector.py": "0c663f03f9900c5b7c1d0d3e7c64ad51908133eedfea53bd2baf4b5ba3253150",
}
CLIENT_SELECTOR, SERVICE_SELECTOR, SERVICE_SELECTION, FAKE_SOURCE, OLD_FIXTURE, QUALIFIER = tuple(
    PINS
)[:6]
TOKEN = "owned-cpu-fixture-only"
MAX_JSON = 8 * 1024**2
MAX_BODY = 128 * 1024


def need(value, reason):
    if not value:
        raise ValueError(reason)


def encoded(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode()


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def ordinary(path):
    path = Path(path).absolute()
    need(
        ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "Ordinary path",
    )
    return path


def checked(label):
    path = ordinary(ROOT / label)
    with path.open("rb") as stream:
        blob = stream.read(1024**2 + 1)
    need(len(blob) <= 1024**2 and sha(blob) == PINS[label], "Exact selected source: " + label)
    return blob


def ref(path):
    path = ordinary(path)
    with path.open("rb") as stream:
        blob = stream.read(1024**2 + 1)
    need(len(blob) <= 1024**2, "Bounded source reference")
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(blob), "size_bytes": len(blob)}


def source_module(label, name):
    path, blob = ROOT / label, checked(label)
    selected = types.ModuleType(name)
    selected.__file__ = str(path)
    exec(compile(blob, str(path), "exec", dont_inherit=True), vars(selected))
    return selected


def definitions(label, names, namespace):
    nodes = [
        n
        for n in ast.parse(checked(label)).body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
    ]
    need({n.name for n in nodes} == set(names), "Exact selected CPU definitions")
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]), str(ROOT / label), "exec", dont_inherit=True
        ),
        namespace,
    )


def save(output, name, value):
    path = output / name
    blob = json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
    need(len(blob) <= MAX_JSON and path.parent == output, "Bounded flat fixture evidence")
    with path.open("xb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())


def persist_evidence(output, result, ledger, source_evidence, primary):
    """A source failure must never suppress the final cleanup/error receipt."""
    for name, value in (("join-ledger.json", ledger), ("source-evidence.json", source_evidence)):
        try:
            save(output, name, value)
        except BaseException as error:
            result["completed"] = False
            result.setdefault("persistence_errors", []).append(
                {"file": name, "type": type(error).__name__, "message": str(error)}
            )
            if primary is None:
                primary = error
            else:
                primary.add_note(
                    "Additional fixture evidence persistence failed: " + name + ": " + repr(error)
                )
    if primary is not None:
        result["error"] = {
            "type": type(primary).__name__,
            "message": str(primary),
            "notes": getattr(primary, "__notes__", []),
        }
    try:
        save(output, "receipt.json", result)
    except BaseException as error:
        if primary is not None:
            primary.add_note("Final fixture receipt persistence failed: " + repr(error))
            raise primary from error
        raise
    return primary


def terminal_flow_matches_original_agent(final):
    """Original DONE consumes the decision and returns before history append."""
    return (
        final["status"] == "done"
        and [h["operation"] for h in final["history"]] == ["FIND_TEXT"]
        and [d["operation"] for d in final["decisions"]] == ["FIND_TEXT", "DONE"]
        and final["decision"] is None
        and final["plan_index"] == 1
        and len(final["text_calls"]) == 1
    )


def loopback_url(url, allowed):
    p = urlsplit(str(url))
    need(
        p.scheme == "http"
        and p.hostname == "127.0.0.1"
        and p.port is not None
        and p.username is None
        and not p.query
        and not p.fragment,
        "Only exact owned loopback HTTP",
    )
    need(str(url) in allowed, "Only selected owned endpoint")
    return str(url)


def process(pid):
    try:
        base = Path("/proc") / str(pid)
        fields = (base / "stat").read_text().rsplit(")", 1)[1].split()
        status = (base / "status").read_text().splitlines()
        rss = next((int(line.split()[1]) * 1024 for line in status if line.startswith("VmRSS:")), 0)
        return {"pid": pid, "start_ticks": int(fields[19]), "state": fields[0], "rss_bytes": rss}
    except (FileNotFoundError, ProcessLookupError):
        return None


def socket_closed(port):
    with socket.socket() as probe:
        probe.settimeout(0.3)
        code = probe.connect_ex(("127.0.0.1", port))
    return {"connect_errno": code, "refused": code == errno.ECONNREFUSED}


def server_record(name, sock):
    return {
        "name": name,
        "host": "127.0.0.1",
        "port": sock.getsockname()[1],
        "socket_fd_at_start": sock.fileno(),
        "thread_started": False,
        "thread_joined": False,
        "socket_fd_after_close": None,
        "after_close_probe": None,
    }


@contextmanager
def owned_http(name, get, post, result):
    class Server(ThreadingHTTPServer):
        daemon_threads = False
        block_on_close = True

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(3)

        def do_GET(self):
            self.dispatch("GET")

        def do_POST(self):
            self.dispatch("POST")

        def dispatch(self, method):
            try:
                count = int(self.headers.get("Content-Length", "0"))
                need(0 <= count <= MAX_BODY, "Bounded owned request")
                body = self.rfile.read(count)
                callback = get if method == "GET" else post
                need(callback is not None, "Declared fixture method")
                status, content_type, response = callback(self.path, body, self.headers)
            except Exception as error:
                status, content_type, response = (
                    400,
                    "application/json",
                    encoded({"fixture_error": type(error).__name__, "detail": str(error)}),
                )
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *_args):
            pass

    # TCPServer closes a partially bound socket if its constructor raises. From
    # successful construction onward, cleanup covers even thread.start failure.
    http = Server(("127.0.0.1", 0), Handler)
    row = server_record(name, http.socket)
    result["servers"].append(row)
    thread, primary = None, None
    try:
        thread = threading.Thread(
            target=http.serve_forever,
            kwargs={"poll_interval": 0.05},
            name="owned-find-" + name,
            daemon=False,
        )
        thread.start()
        row["thread_started"] = True
        yield "http://127.0.0.1:" + str(row["port"])
    except BaseException as error:
        primary = error
        raise
    finally:
        errors = []
        for phase, action in (
            ("shutdown", lambda: http.shutdown() if row["thread_started"] else None),
            ("server_close", http.server_close),
            ("join", lambda: thread.join(timeout=3) if row["thread_started"] else None),
        ):
            try:
                action()
            except BaseException as error:
                errors.append({"phase": phase, "type": type(error).__name__, "message": str(error)})
        row.update(
            thread_joined=bool(row["thread_started"] and not thread.is_alive()),
            socket_fd_after_close=http.socket.fileno(),
            after_close_probe=socket_closed(row["port"]),
            errors=errors,
        )
        if (
            errors
            or not row["thread_joined"]
            or row["socket_fd_after_close"] != -1
            or not row["after_close_probe"]["refused"]
        ):
            if primary is not None:
                primary.add_note("Owned HTTP thread/socket closure incomplete: " + repr(row))
            else:
                raise ValueError("Owned HTTP thread/socket closure incomplete")


@contextmanager
def owned_asgi(app, result):
    import uvicorn

    sock = socket.socket()
    row = {
        "name": "actual-current-policy-asgi",
        "host": "127.0.0.1",
        "port": None,
        "socket_fd_at_start": sock.fileno(),
        "thread_started": False,
        "thread_joined": False,
        "socket_fd_after_close": None,
        "after_close_probe": None,
    }
    result["servers"].append(row)
    thread, server, primary, errors = None, None, None, []
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        sock.listen(16)
        row["port"] = sock.getsockname()[1]
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=row["port"],
            log_config=None,
            log_level="critical",
            access_log=False,
            proxy_headers=False,
            lifespan="off",
            loop="asyncio",
            timeout_keep_alive=1,
            timeout_graceful_shutdown=3,
        )
        server = uvicorn.Server(config)

        def run():
            try:
                server.run(sockets=[sock])
            except BaseException as error:
                errors.append(
                    {"phase": "serve", "type": type(error).__name__, "message": str(error)}
                )

        thread = threading.Thread(target=run, name="owned-find-current-policy", daemon=False)
        thread.start()
        row["thread_started"] = True
        end = time.monotonic() + 5
        while not server.started and thread.is_alive() and time.monotonic() < end:
            time.sleep(0.01)
        need(
            server.started and thread.is_alive() and not errors,
            "Actual owned ASGI listener started",
        )
        yield "http://127.0.0.1:" + str(row["port"])
    except BaseException as error:
        primary = error
        raise
    finally:
        if row["thread_started"]:
            server.should_exit = True
            thread.join(timeout=5)
            if thread.is_alive():
                server.force_exit = True
                thread.join(timeout=1)
        sock.close()
        row.update(
            thread_joined=bool(row["thread_started"] and not thread.is_alive()),
            socket_fd_after_close=sock.fileno(),
            after_close_probe=socket_closed(row["port"]) if row["port"] else None,
            errors=errors,
        )
        closed = (
            row["thread_joined"]
            and not errors
            and row["socket_fd_after_close"] == -1
            and row["after_close_probe"]
            and row["after_close_probe"]["refused"]
        )
        if not closed:
            if primary is not None:
                primary.add_note("Owned ASGI thread/socket closure incomplete: " + repr(row))
            else:
                raise ValueError("Owned ASGI thread/socket closure incomplete")


def fake_model(ledger):
    # The real torch package is never imported. The exact selected fixture only
    # needs inference_mode; numpy Tensor and fake LM implement numerical output.
    import numpy as np

    ns = {
        "np": np,
        "copy": copy,
        "json": json,
        "SimpleNamespace": types.SimpleNamespace,
        "contextmanager": contextmanager,
    }
    definitions(FAKE_SOURCE, {"Tensor", "Tokenizer", "Processor", "model_fixture"}, ns)
    model = ns["model_fixture"]()
    tensor = ns["Tensor"]
    model.allow_done = False

    def generate(**kwargs):
        index = len(model.calls)
        need(index < 2, "Exactly two fake policy generations")
        operation = ("FIND_TEXT", "DONE")[index]
        need(
            operation != "DONE" or model.allow_done,
            "Actual viewport observation must precede fake DONE",
        )
        model.calls.append(kwargs)
        grammar = kwargs["prefix_allowed_tokens_fn"]
        selected = next(
            i
            for i, action in enumerate(grammar.canonical_actions)
            if json.loads(action) == {"operation": operation, "target": None}
        )
        tokens = list(grammar.encoded_actions[selected])
        prefix = kwargs["input_ids"][0].tolist()
        kwargs["streamer"].put(kwargs["input_ids"])
        checks = []
        for i, token in enumerate(tokens):
            allowed = grammar(0, tensor(prefix))
            need(token in allowed, "Real original prefix-allowed-token grammar")
            checks.append({"token": int(token), "allowed_count": len(allowed), "accepted": True})
            model.backbone(input_ids=kwargs["input_ids"] if i == 0 else tensor([[token]]))
            kwargs["streamer"].put(tensor([token]))
            prefix.append(token)
        kwargs["streamer"].end()
        ledger["fake_generations"].append(
            {
                "index": index,
                "operation": operation,
                "target": None,
                "numerical_output_fake": True,
                "real_weights_loaded": False,
                "grammar_legal_actions": len(grammar.canonical_actions),
                "canonical_actions_sha256": sha(encoded(grammar.canonical_actions)),
                "allowed_token_checks": checks,
                "done_observed_visible_gate": model.allow_done,
            }
        )
        return tensor([prefix])

    model.lm.generate = generate
    return model


def source_check():
    for label in PINS:
        checked(label)
    old = source_module(OLD_FIXTURE, "_owned_find_selected_prior_fixture")
    for label, expected in old.PINS.items():
        selected = ref(ROOT / label)
        need(selected["sha256"] == expected, "Exact bounded reused source")
    selection = json.loads(checked(SERVICE_SELECTION))
    expected = selection["project"]["descriptor"]
    need(len(expected["project_sources"]) == 22, "All selected original project22")
    service = source_module(SERVICE_SELECTOR, "_owned_find_service_selector")
    for value in [
        selection["source"],
        selection["integration"],
        selection["driver"],
        *selection["scripts"].values(),
        *expected["components"].values(),
        *expected["project_sources"],
    ]:
        service.checked(value)
    return old, service, selection


def source_refs(old, selection):
    labels = set(PINS) | set(old.PINS)
    descriptor = selection["project"]["descriptor"]
    labels.update(
        row["path"]
        for row in [
            selection["source"],
            selection["integration"],
            selection["driver"],
            *selection["scripts"].values(),
            *descriptor["components"].values(),
            *descriptor["project_sources"],
        ]
    )
    return [ref(ROOT / label) for label in sorted(labels)]


def thread_snapshot():
    return [
        {"name": t.name, "ident": t.ident, "native_id": t.native_id, "daemon": t.daemon}
        for t in threading.enumerate()
    ]


def resource_snapshot():
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            fields[key] = int(value.split()[0]) * 1024
    return {
        "controller": process(os.getpid()),
        "host": fields,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "scope": "Read-only measurements; not admission and not a global hard cap",
    }


def execute(output, reviewed_source_sha, reviewed_self_ref):
    need(
        not any(
            name == "s1" or name.startswith("s1.") or name in {"torch", "transformers"}
            for name in sys.modules
        ),
        "Fresh worker without frameworks/project modules",
    )
    before_imports_resources = resource_snapshot()
    before_imports_threads = thread_snapshot()
    torch_stub = types.SimpleNamespace(
        inference_mode=nullcontext, fixture_only=True, real_torch_imported=False
    )
    sys.modules["torch"] = torch_stub
    # Installed before any possible project/framework import. This process is
    # disposable and never imports true torch, transformers, models or weights.
    import httpx
    from playwright.sync_api import sync_playwright

    result = {
        "schema_id": "find-current-stack-owned-cpu-v1",
        "fixture_only": True,
        "completed": False,
        "quality_score": None,
        "model_qualified": False,
        "live_q95_qualified": False,
        "real_model_calls": 0,
        "comfy_calls": 0,
        "production_plan_created": False,
        "true_torch_imported": False,
        "torch_stub": "explicit inference_mode-only installed before project imports",
        "servers": [],
        "observed_browser_processes": [],
        "closed_observed_browser_processes": False,
        "whole_tree_exit_claimed": False,
        "browser_close_returned": False,
        "owned_thread_baseline": before_imports_threads,
        "thread_closure_scope": "Three owned server accept threads are explicitly joined; other nonbaseline threads are observed only, with no universal thread-closure claim.",
        "resources": {
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "rlimit_cpu": list(resource.getrlimit(resource.RLIMIT_CPU)),
            "rlimit_core": list(resource.getrlimit(resource.RLIMIT_CORE)),
            "rlimit_nofile": list(resource.getrlimit(resource.RLIMIT_NOFILE)),
            "rlimit_fsize": list(resource.getrlimit(resource.RLIMIT_FSIZE)),
            "rlimit_as": list(resource.getrlimit(resource.RLIMIT_AS)),
            "process_rss_measurements": [],
            "memory_global_hard_cap_claimed": False,
            "before_imports": before_imports_resources,
        },
    }
    ledger = {
        "model_adapter": [],
        "helper": [],
        "documents": [],
        "fake_generations": [],
        "server_events": [],
        "resource_gate": [],
        "cdp": [],
        "network": [],
        "blocked_requests": [],
        "viewport": [],
        "real_http": [],
    }
    primary, browser, browser_session, runner, client_selector, manifest, bundle = (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    baseline_env = {
        key: os.environ.get(key)
        for key in (
            "TYPESAFE_MODEL",
            "TYPESAFE_API_KEY",
            "TEXT_MODEL_BASE_URL",
            "TEXT_MODEL",
            "TEXT_MODEL_API_KEY",
            "TEXT_MODEL_REASONING",
        )
    }
    os.environ["TEXT_MODEL_REASONING"] = "none"
    old, original_source_refs, active_records, model, final_self_ref = None, [], None, None, None
    try:
        need(
            ref(Path(__file__))["sha256"] == reviewed_source_sha, "Reviewed entry source unchanged"
        )
        result["reviewed_source_sha256"] = reviewed_source_sha
        old, service_selector, service_selection = source_check()
        original_source_refs = source_refs(old, service_selection)
        result["chromium"], result["launch_flags"] = old.binary_pin(), old.FLAGS
        client_selector = source_module(CLIENT_SELECTOR, "_owned_find_client_selector")
        bundle, manifest = client_selector.prepare_bundle(
            output / "client-bundle", profile=client_selector.PROFILE
        )
        sys.path.insert(0, str(bundle))
        try:
            spec = importlib.util.spec_from_file_location(
                "_owned_find_current_runner", bundle / "runner.py"
            )
            runner = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = runner
            exec(
                compile(
                    (bundle / "runner.py").read_bytes(),
                    str(bundle / "runner.py"),
                    "exec",
                    dont_inherit=True,
                ),
                vars(runner),
            )
            profile = importlib.import_module("jev_observation_profile_v7")
            runtime_data = profile.materialize_profile(
                UPSTREAM,
                output / "v7-observation",
                expected_source_sha256=client_selector.V7_SHA,
                profile=client_selector.V7,
            )
            runtime = profile.verify_materialized_profile(
                output / "v7-observation",
                expected_manifest_sha256=runtime_data["manifest_sha256"],
                expected_source_sha256=client_selector.V7_SHA,
                profile=client_selector.V7,
            )
            result["v7_materialized_manifest_sha256"] = runtime_data["manifest_sha256"]
            result["v7_runtime_manifest"] = runtime.manifest()
            result["client_bundle_manifest_sha256"] = sha(
                (bundle / "bundle-manifest.json").read_bytes()
            )
            selection_blob = checked(SERVICE_SELECTION)
            with ExitStack() as stack:
                selected_bundle, driver, inverse = stack.enter_context(
                    service_selector.selected(
                        selection_blob, PINS[SERVICE_SELECTION], profile=service_selector.PROFILE
                    )
                )
                result["service_inverse"], result["integration_inverse"] = (
                    inverse,
                    selected_bundle.proofs,
                )
                model = fake_model(ledger)
                gate = types.SimpleNamespace(
                    check=lambda phase: ledger["resource_gate"].append(
                        {
                            "phase": phase,
                            "fixture_gate_only": True,
                            "own_process": process(os.getpid()),
                        }
                    )
                )
                app = driver.make_application(
                    model,
                    None,
                    ledger["server_events"].append,
                    gate,
                    output / "policy-service",
                    TOKEN,
                    "shared-elements-v2",
                    "unused-separate-strict-helper",
                )
                policy_base = stack.enter_context(owned_asgi(app, result))
                needle = "current-owned-" + os.urandom(12).hex() + " ' \\ literal"
                goal = "Make this exact literal text visible: " + needle
                document = old.document([needle])
                save(
                    output,
                    "owned-document.json",
                    {
                        "fixture_only": True,
                        "html_utf8": document.decode(),
                        "needle": needle,
                        "goal": goal,
                    },
                )

                def get_document(path, body, _headers):
                    need(path == "/owned" and body == b"", "Only self-generated owned document")
                    ledger["documents"].append({"path": path, "sha256": sha(document)})
                    return 200, "text/html; charset=utf-8", document

                document_base = stack.enter_context(
                    owned_http("document", get_document, None, result)
                )
                helper_schema = {"json": json, "math": math, "IDENTITY": "generated-actions-none"}
                definitions(
                    QUALIFIER,
                    {
                        "require",
                        "canonical",
                        "strict_json",
                        "exact",
                        "keys",
                        "parse_helper_request",
                    },
                    helper_schema,
                )
                # TEXT_VALUE is selected from the actual canonical questions module
                # after load_upstream; closure resolves the value at request time.
                helper_system = None

                def helper_post(path, body, headers):
                    need(
                        path == "/chat/completions"
                        and headers.get("Authorization") == "Bearer " + TOKEN,
                        "Owned strict helper route/auth",
                    )
                    request = json.loads(body)
                    context = helper_schema["parse_helper_request"](
                        request, system_prompt=helper_system
                    )
                    need(
                        context["field"]["role"] == "browser_find_query",
                        "Strict literal query field",
                    )
                    prefix = "Make this exact literal text visible: "
                    need(
                        context["goal"].startswith(prefix),
                        "Query derived only from generated fixture goal",
                    )
                    query = context["goal"][len(prefix) :]
                    need(query == needle, "Exact generated literal query")
                    response = {
                        "choices": [{"message": {"content": encoded({"text": query}).decode()}}],
                        "usage": {"fixture_numbers_fake": True},
                    }
                    ledger["helper"].append(
                        {
                            "request": request,
                            "request_bytes_sha256": sha(body),
                            "context": context,
                            "query": query,
                            "schema_valid": True,
                            "response": response,
                            "model_numbers_fake": True,
                            "HTTP_real": True,
                        }
                    )
                    return 200, "application/json", encoded(response)

                helper_base = stack.enter_context(
                    owned_http("strict-text-helper", None, helper_post, result)
                )
                endpoints = {policy_base + "/v1/systemone", helper_base + "/chat/completions"}

                class LoopbackTransport(httpx.BaseTransport):
                    def __init__(self):
                        self.actual = httpx.HTTPTransport(retries=0, trust_env=False)

                    def handle_request(self, request):
                        loopback_url(request.url, endpoints)
                        need(request.method == "POST", "Only owned model POST")
                        response = self.actual.handle_request(request)
                        response_blob = response.read()
                        ledger["real_http"].append(
                            {
                                "url": str(request.url),
                                "method": request.method,
                                "request_bytes_sha256": sha(request.content),
                                "response_bytes_sha256": sha(response_blob),
                                "status": response.status_code,
                                "x_request_id": response.headers.get("x-request-id"),
                                "real_socket_transport": True,
                            }
                        )
                        return response

                    def close(self):
                        self.actual.close()

                result["resources"]["before_browser"] = resource_snapshot()
                playwright = stack.enter_context(sync_playwright())
                browser = playwright.chromium.launch(
                    executable_path=str(old.CHROMIUM),
                    headless=True,
                    args=old.FLAGS,
                    env={"PATH": os.defpath, "LANG": "C.UTF-8"},
                    timeout=15000,
                )
                observe = None

                def close_browser():
                    nonlocal browser
                    if browser is None:
                        return
                    try:
                        if observe is not None:
                            observe()
                    except Exception as error:
                        result.setdefault("cleanup_errors", []).append(
                            {
                                "phase": "final_observed_process_read",
                                "type": type(error).__name__,
                                "message": str(error),
                            }
                        )
                    browser.close()
                    result["browser_close_returned"] = True
                    browser = None

                stack.callback(close_browser)
                browser_session = browser.new_browser_cdp_session()

                def observe_processes():
                    seen = {
                        (r["pid"], r["start_ticks"]) for r in result["observed_browser_processes"]
                    }
                    for item in browser_session.send("SystemInfo.getProcessInfo")["processInfo"]:
                        row = process(int(item["id"]))
                        if row and (row["pid"], row["start_ticks"]) not in seen:
                            result["observed_browser_processes"].append(row)
                    result["resources"]["process_rss_measurements"].append(
                        {
                            "controller": process(os.getpid()),
                            "observed_chromium": [
                                process(r["pid"]) for r in result["observed_browser_processes"]
                            ],
                            "scope": "Observed processes only, not a global hard cap",
                        }
                    )

                observe = observe_processes
                observe_processes()

                class LocalOnly:
                    def new_page(self):
                        page = browser.new_page(service_workers="block")

                        def route(route):
                            req = route.request
                            if req.url == document_base + "/owned" and req.method in {
                                "GET",
                                "HEAD",
                            }:
                                route.continue_()
                            else:
                                ledger["blocked_requests"].append(
                                    {"url": req.url, "method": req.method}
                                )
                                route.abort()

                        page.route("**/*", route)
                        page.on(
                            "request",
                            lambda request: ledger["network"].append(
                                {
                                    "kind": "playwright-request",
                                    "url": request.url,
                                    "method": request.method,
                                }
                            ),
                        )
                        return page

                class Transport(runner.PlaywrightCDP):
                    def cdp(self, method, session_id=None, **params):
                        row = {
                            "method": method,
                            "session": session_id,
                            "params": copy.deepcopy(params),
                        }
                        ledger["cdp"].append(row)
                        try:
                            response = super().cdp(method, session_id=session_id, **params)
                            row["response"] = copy.deepcopy(response)
                            if method == "Target.attachToTarget":
                                session = self.sessions[response["sessionId"]]
                                session.on(
                                    "Network.requestWillBeSent",
                                    lambda data: ledger["network"].append(
                                        {"kind": "cdp-request", "data": data}
                                    ),
                                )
                                session.send("Network.enable")
                            return response
                        except BaseException as error:
                            row["error"] = {"type": type(error).__name__, "message": str(error)}
                            raise

                transport = Transport(LocalOnly())
                # Browser closure must happen before Playwright's context exits.
                try:
                    with runner.load_upstream(UPSTREAM, transport, observation_runtime=runtime) as (
                        agent_module,
                        model_module,
                    ):
                        helper_system = model_module.TEXT_VALUE
                        original_client = model_module.CLIENT
                        original_client.close()
                        with httpx.Client(
                            transport=LoopbackTransport(),
                            timeout=25,
                            trust_env=False,
                            follow_redirects=False,
                        ) as actual_client:
                            model_module.CLIENT = actual_client
                            try:
                                with runner.model_adapter(
                                    agent_module,
                                    model_module,
                                    endpoint=policy_base + "/v1/systemone",
                                    model=driver.IDENTITY,
                                    api_key=TOKEN,
                                    text_endpoint=helper_base,
                                    text_model="generated-actions-none",
                                    text_api_key=TOKEN,
                                ) as (records, secrets):
                                    active_records = records
                                    canonical_agent, canonical_choose = (
                                        agent_module.Agent,
                                        agent_module.choose,
                                    )
                                    canonical_cap = agent_module.MAX_STEPS
                                    selected_class = runner._selected_find_agent_class(
                                        agent_module, model_module
                                    )
                                    need(
                                        canonical_cap == agent_module.MAX_STEPS == 60
                                        and agent_module.Agent is canonical_agent
                                        and agent_module.choose is canonical_choose,
                                        "Formal successor selects private16 without canonical-global mutation",
                                    )
                                    result["client_budget_selection"] = {
                                        "selector_source_sha256": PINS[CLIENT_SELECTOR],
                                        "profile": client_selector.PROFILE,
                                        "canonical_cap_before_after": [
                                            canonical_cap,
                                            agent_module.MAX_STEPS,
                                        ],
                                        "canonical_Agent_and_choose_unchanged": True,
                                        "harness_cap_override": False,
                                        "private_selected_cap": 16,
                                    }
                                    with selected_class(
                                        document_base + "/owned", goal, screenshots=False
                                    ) as agent:
                                        before = old.visible(agent.browser)
                                        ledger["viewport"].append(
                                            {"phase": "before_find", "value": before}
                                        )
                                        save(output, "before-state.json", agent.snapshot())
                                        need(
                                            before["in_view"] is False
                                            and needle not in agent.state["page"]["text"],
                                            "Needle initially offscreen and absent from actual V7 visible observation",
                                        )
                                        first = copy.deepcopy(agent.command("tick"))
                                        after = old.visible(agent.browser)
                                        ledger["viewport"].append(
                                            {"phase": "after_find", "value": after}
                                        )
                                        save(output, "after-find-state.json", first)
                                        need(
                                            first["status"] == "ready"
                                            and len(first["history"]) == 1
                                            and first["history"][0]["operation"] == "FIND_TEXT",
                                            "Exactly one actual FIND action before completion",
                                        )
                                        feedback = first["find_text_feedback"][-1]
                                        need(
                                            feedback["status"] == "found"
                                            and feedback["scroll_requested"] is True
                                            and feedback["scroll_changed"] is True
                                            and after["in_view"] is True
                                            and after["y"] > before["y"]
                                            and needle in first["page"]["text"],
                                            "Actual CDP FIND scroll makes owned literal visible",
                                        )
                                        model.allow_done = True
                                        final = copy.deepcopy(agent.command("tick"))
                                        save(output, "final-state.json", final)
                                        save(
                                            output,
                                            "final-dom.json",
                                            {
                                                "fixture_only": True,
                                                "outer_html": agent.browser.evaluate(
                                                    "document.documentElement.outerHTML"
                                                ),
                                            },
                                        )
                                        need(
                                            terminal_flow_matches_original_agent(final),
                                            "Fake legal DONE follows actual successful FIND scroll",
                                        )
                                        need(
                                            [r["kind"] for r in records]
                                            == ["policy", "text", "policy"],
                                            "Original model_adapter sees both real endpoints",
                                        )
                                        need(
                                            all(
                                                len(r.get("transport_attempts", [])) == 1
                                                and r["transport_attempts"][0]["http_status"] == 200
                                                for r in records
                                            ),
                                            "Three actual HTTP exchanges succeeded",
                                        )
                                        need(
                                            len(ledger["helper"]) == 1
                                            and len(model.calls) == 2
                                            and model.contexts == ["enter", "exit", "enter", "exit"]
                                            and not model.hooks,
                                            "Fake LM N3 context/hooks complete",
                                        )
                                        need(
                                            not ledger["blocked_requests"]
                                            and all(
                                                r["url"] == document_base + "/owned"
                                                for r in ledger["network"]
                                                if r["kind"] == "playwright-request"
                                            ),
                                            "Owned document network scope only",
                                        )
                                        ledger["model_adapter"] = copy.deepcopy(records)
                                        for record in ledger["model_adapter"]:
                                            if record["kind"] == "policy":
                                                record["runtime_question_ids"] = list(
                                                    record["request"]["questions"]
                                                )
                                                record["runtime_candidate_labels"] = {
                                                    key: list(value["criteria"])
                                                    for key, value in record["request"][
                                                        "questions"
                                                    ].items()
                                                }
                                        need(
                                            TOKEN not in json.dumps(ledger["model_adapter"]),
                                            "Original adapter redacts the configured synthetic token",
                                        )
                                        result.update(
                                            functional_path_complete=True,
                                            policy_http_calls=2,
                                            strict_helper_http_calls=1,
                                            fake_model_generations=2,
                                            actions=["FIND_TEXT", "DONE"],
                                            viewport_before_after=[
                                                before["in_view"],
                                                after["in_view"],
                                            ],
                                            n3_fake_contexts=model.contexts,
                                            cdp_calls=transport.calls,
                                            observation_profile_errors=transport.observation_profile_errors,
                                            model_adapter_records_redacted=True,
                                        )
                            finally:
                                model_module.CLIENT = original_client
                finally:
                    close_browser()
                need(
                    sys.modules.get("torch") is torch_stub and "transformers" not in sys.modules,
                    "Only explicit fake torch module used",
                )
                client_selector.verify_bundle(bundle, manifest)
                selected_bundle.verify()
        finally:
            sys.path.remove(str(bundle))
            sys.modules.pop("_owned_find_current_runner", None)
            sys.modules.pop("find_client_selector", None)
    except BaseException as error:
        primary = error
        result["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "notes": getattr(error, "__notes__", []),
        }
    finally:
        # All servers and the selected project context have exited before these
        # observations. Failure evidence is saved; original sources are untouched.
        if active_records is not None:
            ledger["model_adapter"] = copy.deepcopy(active_records)
            for record in ledger["model_adapter"]:
                if record["kind"] == "policy":
                    record["runtime_question_ids"] = list(record["request"]["questions"])
                    record["runtime_candidate_labels"] = {
                        key: list(value["criteria"])
                        for key, value in record["request"]["questions"].items()
                    }
        if model is not None:
            result["fake_model_final"] = {
                "calls": len(model.calls),
                "contexts": model.contexts,
                "hook_count": len(model.hooks),
                "all_numbers_fake": True,
            }
        for key, value in baseline_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        result["environment_restored"] = {
            k: os.environ.get(k) for k in baseline_env
        } == baseline_env
        closure = []
        for row in result["observed_browser_processes"]:
            current = process(row["pid"])
            closure.append(
                {
                    "before": row,
                    "after": current,
                    "same_process_absent": current is None
                    or current["start_ticks"] != row["start_ticks"],
                }
            )
        result["process_closure"] = closure
        result["closed_observed_browser_processes"] = bool(closure) and all(
            r["same_process_absent"] for r in closure
        )
        result["final_threads"] = thread_snapshot()
        baseline_thread_ids = {row["ident"] for row in result["owned_thread_baseline"]}
        result["nonbaseline_final_threads_observed_only"] = [
            row for row in result["final_threads"] if row["ident"] not in baseline_thread_ids
        ]
        result["owned_threads_remaining"] = [
            t.name for t in threading.enumerate() if t.name.startswith("owned-find-")
        ]
        result["all_owned_server_threads_sockets_closed"] = len(result["servers"]) == 3 and all(
            r["thread_joined"]
            and r["socket_fd_after_close"] == -1
            and r["after_close_probe"]["refused"]
            for r in result["servers"]
        )
        result["true_torch_imported"] = sys.modules.get("torch") is not torch_stub
        try:
            final_self_ref = ref(Path(__file__))
            need(
                final_self_ref["sha256"] == reviewed_source_sha,
                "Reviewed fixture self source still exact",
            )
            old_final, _, final_selection = source_check()
            final_source_refs = source_refs(old_final, final_selection)
            need(
                original_source_refs == final_source_refs, "Complete consumed source refs unchanged"
            )
            need(
                old_final.binary_pin() == result["chromium"],
                "Exact consumed Chromium binary unchanged",
            )
            if client_selector is not None and manifest is not None:
                client_selector.verify_bundle(bundle, manifest)
            result["all_selected_sources_unchanged"] = True
        except BaseException as error:
            result["all_selected_sources_unchanged"] = False
            result["source_recheck_error"] = {"type": type(error).__name__, "message": str(error)}
            if primary is None:
                primary = error
        result["resources"]["after_closure"] = resource_snapshot()
        result["resources"]["controller_peak_rss_bytes"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        result["fixture_controller"] = process(os.getpid())
        result["completed"] = (
            primary is None
            and result.get("functional_path_complete", False)
            and result["closed_observed_browser_processes"]
            and result["all_owned_server_threads_sockets_closed"]
            and not result["owned_threads_remaining"]
            and not result.get("cleanup_errors")
            and result["environment_restored"]
            and not result["true_torch_imported"]
            and result["all_selected_sources_unchanged"]
        )
        source_evidence = {
            "fixture_source": reviewed_self_ref,
            "final_fixture_source": final_self_ref,
            "selected_sources": original_source_refs,
            "reviewed_fixture_sha256": reviewed_source_sha,
            "final_fixture_sha256": final_self_ref["sha256"]
            if final_self_ref is not None
            else None,
            "source_recheck_error": result.get("source_recheck_error"),
            "service_selection_sha256": PINS[SERVICE_SELECTION],
            "all_selected_sources_unchanged": result["all_selected_sources_unchanged"],
            "model_numbers_fake": True,
        }
        primary = persist_evidence(output, result, ledger, source_evidence, primary)
        sys.modules.pop("torch", None)
    if primary:
        raise primary
    need(result["completed"], "Functional/current-stack or owned cleanup evidence incomplete")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "actual-01")
    parser.add_argument("--source-sha256")
    args = parser.parse_args(argv)
    source_check()
    if not args.execute:
        print(
            json.dumps(
                {
                    "fixture_only": True,
                    "execute_requested": False,
                    "actors_started": False,
                    "real_model_calls": 0,
                    "quality_score": None,
                }
            )
        )
        return 0
    reviewed_self_ref = ref(Path(__file__))
    need(
        args.source_sha256 == reviewed_self_ref["sha256"],
        "Explicit independently reviewed fixture source SHA",
    )
    output = ordinary(args.output)
    need(
        output.parent == HERE
        and re.fullmatch(r"actual-[0-9]{2}", output.name)
        and not output.exists(),
        "Fresh owned attempt output",
    )
    need(os.sched_getaffinity(0) == {0}, "Explicit CPU0 worker affinity")
    for limit, maximum in (
        (resource.RLIMIT_CPU, 60),
        (resource.RLIMIT_CORE, 0),
        (resource.RLIMIT_NOFILE, 512),
        (resource.RLIMIT_FSIZE, MAX_JSON),
    ):
        soft, hard = resource.getrlimit(limit)
        need(0 <= soft <= hard <= maximum, "Finite inherited CPU/core/FD/file limit")
    need(
        all(
            os.environ.get(key) == "1"
            for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        ),
        "Single BLAS thread environment",
    )
    output.mkdir()
    result = execute(output, args.source_sha256, reviewed_self_ref)
    print(
        json.dumps(
            {
                "completed": result["completed"],
                "policy_http_calls": result["policy_http_calls"],
                "strict_helper_http_calls": result["strict_helper_http_calls"],
                "viewport_before_after": result["viewport_before_after"],
                "observed_processes_closed": result["closed_observed_browser_processes"],
                "server_threads_sockets_closed": result["all_owned_server_threads_sockets_closed"],
                "quality_score": None,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
