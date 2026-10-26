"""Explicit CPU-only owned Chromium capability proof, never a model benchmark."""

from __future__ import annotations

import argparse
import ast
import base64
import copy
import hashlib
import html
import json
import math
import os
import secrets
import sys
import threading
import time
import types
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
COMPONENT = ROOT / "artifacts/jev-runtime/browser-find-text-profile-20261022-01"
UPSTREAM = ROOT / "artifacts/jev-runtime/upstream/jev_ultrafast"
CHROMIUM = Path(
    "/home/albert_hsieh/.cache/ms-playwright/chromium_headless_shell-1208/chrome-linux/headless_shell"
)
CHROMIUM_SHA = "3d7ae069d9c2e7f049998f6871be2381fb6967f7a06c26ea0b346321e3dc496f"
PINS = {
    "artifacts/jev-runtime/browser-find-text-profile-20261022-01/find_profile.py": "98c7fa29265e2c0c25bdcae0825522cebcd35dc86c16ed53aa7609bd52e6a08d",
    "artifacts/jev-runtime/browser-find-text-profile-20261022-01/private_protocol.py": "7f4dab379ac145818fad73babcb4ba0633520124fd862c365c8429489a352f1f",
    "artifacts/jev-runtime/upstream/jev_ultrafast/agent.py": "aaf8b271cb0ee6643fe7beb525328ab80de8568c7ac044d02b9c6721161e7c8b",
    "artifacts/jev-runtime/upstream/jev_ultrafast/model.py": "a85ada8458d23642c5c8a4095acf0565e6268f81dcda26f6308942b653e9c9f3",
    "artifacts/jev-runtime/upstream/jev_ultrafast/browser.py": "ee08dafa93d731bd292aa0263a578a29c8001cf22eef3c59fe44a159649155ee",
    "artifacts/jev-runtime/upstream/jev_ultrafast/questions.py": "66a7054b85040b949e1757311257ea0c5a843c635f9a40c99f6d15ad660fb422",
    "artifacts/jev-runtime/upstream/jev_ultrafast/snapshot.js": "e50473501c8fb8e70f3b21866d987393e3f2315c639d638bd477d170e81ed78d",
    "scripts/jev_browser_benchmark.py": "fe1410675e4decb06a8491a5a83d2ec965629c99f8912b797446c4cb2c155f6c",
    "scripts/jev_qualify_public_field_syntax_bf16.py": "cbfb7bdd8301174a97ac18389a998ff3256ac32d6f7a770978f2f9b722c3b43b",
}
FLAGS = ["--disable-gpu", "--disable-software-rasterizer", "--disable-background-networking"]


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def wire_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def checked(path):
    blob = path.read_bytes()
    if sha(blob) != PINS[path.relative_to(ROOT).as_posix()]:
        raise ValueError("Selected fixture source changed")
    return blob


def load(path, name, *, selected=False):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    blob = checked(path) if selected else path.read_bytes()
    exec(compile(blob, str(path), "exec", dont_inherit=True), vars(module))
    return module


def selected_nodes(path, namespace, names=None, *, remove_client=False):
    nodes = ast.parse(checked(path)).body
    nodes = [n for n in nodes if not isinstance(n, (ast.Import, ast.ImportFrom))]
    if names is not None:
        nodes = [
            n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
        ]
        if {n.name for n in nodes} != set(names):
            raise ValueError("Exact selected source definitions")
    if remove_client:
        nodes = [
            n
            for n in nodes
            if not (
                isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "CLIENT" for t in n.targets)
            )
        ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)


def binary_pin():
    digest, count = hashlib.sha256(), 0
    before = CHROMIUM.stat()
    with CHROMIUM.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
            count += len(block)
    after = CHROMIUM.stat()
    if (
        digest.hexdigest() != CHROMIUM_SHA
        or count != 287892096
        or (before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_ino, after.st_size, after.st_mtime_ns)
    ):
        raise ValueError("Exact selected Chromium binary")
    return {"path": str(CHROMIUM), "sha256": digest.hexdigest(), "size_bytes": count}


@contextmanager
def server(documents):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            blob = documents.get(self.path)
            self.send_response(200 if blob is not None else 404)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(blob or b"")))
            self.end_headers()
            self.wfile.write(blob or b"")

        def log_message(self, *_args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:" + str(http.server_port), requests
    finally:
        http.shutdown()
        http.server_close()
        thread.join(timeout=2)
        if thread.is_alive():
            raise ValueError("Owned fixture server did not stop")


def document(needles, *, hidden_only=False):
    controls = '<h1>Owned generated document</h1><button>Generic control</button><a href="#bottom">Generic link</a><label>Generic field<input></label><select><option>One</option></select>'
    parts = [
        "<!doctype html><title>Owned random navigation fixture</title><style>body{font:16px monospace}</style>",
        controls,
    ]
    parts.append('<p hidden id="hidden">' + html.escape(needles[0]) + "</p>")
    if not hidden_only:
        for index, needle in enumerate(needles):
            parts.append(
                '<div style="height:'
                + str(30000 if index == 0 else 1400)
                + 'px">Owned filler</div>'
            )
            parts.append('<p id="needle-' + str(index) + '">' + html.escape(needle) + "</p>")
        parts.append(
            '<div style="height:4000px"></div><p id="duplicate">' + html.escape(needles[0]) + "</p>"
        )
    return ("".join(parts) + '<p id="bottom">Owned end</p>').encode()


def proc_identity(pid):
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "start_ticks": int(fields[19]), "state": fields[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def private_upstream(transport, fixture_io):
    import httpx

    questions = load(UPSTREAM / "questions.py", "_find_fixture_questions", selected=True)
    model = types.ModuleType("_find_fixture_model")
    model.__dict__.update(
        __file__=str(UPSTREAM / "model.py"),
        json=json,
        math=math,
        os=os,
        time=time,
        httpx=httpx,
        CLIENT=fixture_io,
    )
    model.__dict__.update(
        {k: getattr(questions, k) for k in ("NEXT_ACTION", "TARGET", "TEXT_VALUE")}
    )
    selected_nodes(UPSTREAM / "model.py", vars(model), remove_client=True)
    browser = types.ModuleType("_find_fixture_browser")
    browser.__dict__.update(
        __file__=str(UPSTREAM / "browser.py"),
        hashlib=hashlib,
        json=json,
        sys=sys,
        time=time,
        Path=Path,
        ensure_daemon=lambda: None,
        cdp=transport.cdp,
    )
    selected_nodes(UPSTREAM / "browser.py", vars(browser))
    agent = types.ModuleType("_find_fixture_agent")
    agent.__dict__.update(
        __file__=str(UPSTREAM / "agent.py"),
        base64=base64,
        time=time,
        Path=Path,
        Browser=browser.Browser,
        StalePage=browser.StalePage,
        MAX_STEPS=16,
    )
    agent.__dict__.update(
        {k: getattr(model, k) for k in ("action_space", "choose", "field_context", "field_text")}
    )
    selected_nodes(UPSTREAM / "agent.py", vars(agent))
    return agent, model, browser


class FixtureIO:
    """No sockets: original post_json receives explicit CPU-only substitutions."""

    def __init__(self, native, wire):
        from s1.evaluation import decision3

        self.d3, self.wire = decision3, wire
        self.name = "fixture-find-only"
        self.spec = decision3.decision3_spec("d3-edge")
        self.adapter = native.JevDecision3NativeAdapter(self, model_identity=self.name)
        self.query, self.hook = None, lambda: None
        self.policy, self.text, self.native_calls = [], [], []
        self.schema = {"json": json, "math": math, "IDENTITY": "generated-actions-none"}
        selected_nodes(
            ROOT / "scripts/jev_qualify_public_field_syntax_bf16.py",
            self.schema,
            ["require", "canonical", "strict_json", "exact", "keys", "parse_helper_request"],
        )

    def predict(self, request, *, deadline, evidence_sink):
        deadline()
        payload = self.d3.native_payload(request)
        self.native_calls.append(payload)
        answers = {}
        for question in request.questions:
            labels = question.labels()
            choice = "FIND_TEXT" if question.id == "operation" else labels[0]
            answers[question.id] = {
                "type": "choice",
                "choice": choice,
                "confidence": 1.0,
                "probabilities": {label: float(label == choice) for label in labels},
            }
        evidence_sink(
            "fixture_native_forward",
            {"fixture_only": True, "native_payload_sha256": sha(wire_json(payload))},
        )
        return {
            "model": self.spec.model_id.rsplit("/", 1)[-1],
            "answers": answers,
            "usage": {"input_tokens": 1, "output_tokens": 0},
        }

    def post(self, url, *, json: dict, headers):
        body = json
        if "questions" in body:
            assert url == "https://api.typesafe.ai/v1/systemone"
            response = self.wire.named_browser_response(body, self.adapter.predict(body))
            assert list(response["answers"]) == list(body["questions"])
            self.policy.append(
                {
                    "request_sha256": sha(wire_json(body)),
                    "question_ids": list(body["questions"]),
                    "candidate_labels": {
                        k: list(v["criteria"]) for k, v in body["questions"].items()
                    },
                    "native_payload_sha256": sha(wire_json(self.native_calls[-1])),
                    "answers": response["answers"],
                    "wire": response["decision3_browser_wire"],
                }
            )
        else:
            assert url == "http://fixture.invalid/chat/completions"
            context = self.schema["parse_helper_request"](body, system_prompt=self.system_prompt)
            assert context["field"]["role"] == "browser_find_query"
            self.hook()
            self.text.append(
                {
                    "request_sha256": sha(wire_json(body)),
                    "context_sha256": sha(wire_json(context)),
                    "goal": context["goal"],
                    "query": self.query,
                    "schema_valid": True,
                }
            )
            response = {
                "choices": [{"message": {"content": wire_json({"text": self.query}).decode()}}],
                "usage": {},
            }
        return types.SimpleNamespace(status_code=200, is_error=False, json=lambda: response)


def visible(browser, index=0):
    return browser.evaluate(
        "(() => {const e=document.getElementById('needle-"
        + str(index)
        + "');const r=e?.getBoundingClientRect();return {y:scrollY,top:r?.top,bottom:r?.bottom,in_view:!!r&&r.top>=0&&r.bottom<=innerHeight};})()"
    )


def execute(output):
    from playwright.sync_api import sync_playwright

    sys.path.insert(0, str(ROOT / "src"))
    profile = load(COMPONENT / "find_profile.py", "_fixture_profile", selected=True)
    protocol = load(COMPONENT / "private_protocol.py", "_fixture_protocol", selected=True)
    wire, wire_proof = load(HERE / "experimental_wire.py", "_fixture_wire_source").load_wire()
    bridge_namespace = {"copy": copy, "time": time, "urlsplit": urlsplit}
    selected_nodes(ROOT / "scripts/jev_browser_benchmark.py", bridge_namespace, ["PlaywrightCDP"])
    needle = "owned-" + secrets.token_hex(12) + " ' \\ literal"
    series = ["owned-" + secrets.token_hex(12) for _ in range(16)]
    documents = {
        "/long": document([needle]),
        "/hidden": document([needle], hidden_only=True),
        "/series": document(series),
    }
    result = {
        "schema_id": "jev-find-text-owned-cpu-browser-v1",
        "fixture_only": True,
        "quality_score": None,
        "model_qualified": False,
        "server_bootstrap_qualified": False,
        "chromium": binary_pin(),
        "launch_flags": FLAGS,
        "cases": [],
        "owned_processes": [],
        "closed_observed_processes": False,
        "comfy_calls": 0,
        "real_model_calls": 0,
    }
    variables = {
        "TYPESAFE_MODEL": "fixture-find-only",
        "TYPESAFE_API_KEY": "fixture-only",
        "TEXT_MODEL": "generated-actions-none",
        "TEXT_MODEL_API_KEY": "fixture-only",
        "TEXT_MODEL_BASE_URL": "http://fixture.invalid",
        "TEXT_MODEL_REASONING": "none",
    }
    prior = {k: os.environ.get(k) for k in variables}
    os.environ.update(variables)
    primary = None
    try:
        with (
            protocol.load_private_protocol() as (modules, inverse),
            server(documents) as (base, requests),
            sync_playwright() as playwright,
        ):
            browser = playwright.chromium.launch(
                executable_path=str(CHROMIUM),
                headless=True,
                args=FLAGS,
                env={"PATH": os.defpath, "LANG": "C.UTF-8"},
                timeout=15000,
            )
            try:
                session = browser.new_browser_cdp_session()
                info = session.send("SystemInfo.getProcessInfo")
                result["owned_processes"] = [
                    proc_identity(int(row["id"])) for row in info["processInfo"]
                ]
                result["owned_processes"] = [row for row in result["owned_processes"] if row]
                blocked = []

                class LocalOnly:
                    def new_page(self):
                        page = browser.new_page()

                        def route(request):
                            if request.request.url.startswith(base + "/"):
                                request.continue_()
                            else:
                                blocked.append(request.request.url)
                                request.abort()

                        page.route("**/*", route)
                        return page

                transport = bridge_namespace["PlaywrightCDP"](LocalOnly())
                io = FixtureIO(modules["jev_decision3_native"], wire)
                agent_module, model_module, browser_module = private_upstream(transport, io)
                io.system_prompt = model_module.TEXT_VALUE

                def one(
                    name, *, path="/long", query=needle, hook=None, navigator=None, expect="found"
                ):
                    io.query, io.hook = query, hook or (lambda: None)
                    options = {"marker_expression": browser_module.MARKER}
                    if navigator:
                        options["navigator"] = navigator
                    cls = profile.make_agent_class(agent_module, model_module, **options)
                    with cls(base + path, "Find this exact literal text: " + query) as agent:
                        before = visible(agent.browser)
                        original_page = copy.deepcopy(agent.state["page"])
                        policy_index, text_index = len(io.policy), len(io.text)
                        outcome = agent.command("tick")
                        fact = outcome["find_text_feedback"][-1]
                        after = visible(agent.browser)
                        assert (
                            len(outcome["history"]) == 1
                            and fact["status"] == expect
                            and outcome["status"] != "done"
                        )
                        assert agent.state["decisions"][0]["request"]["state"]["page"] == {
                            k: original_page[k] for k in ("url", "title", "text")
                        }
                        assert len(io.policy) == policy_index + 1 and len(io.text) == text_index + 1
                        assert io.text[-1]["query"] == fact["query"] == query
                        if expect == "found":
                            assert before["in_view"] is False and after["in_view"] is True
                            assert fact["scroll_requested"] is True
                        result["cases"].append(
                            {
                                "name": name,
                                "before": before,
                                "after": after,
                                "actions": len(outcome["history"]),
                                "decisions": len(outcome["decisions"]),
                                "feedback": fact,
                                "policy_index": policy_index,
                                "helper_index": text_index,
                            }
                        )
                        return agent, outcome

                one("long_literal_hidden_duplicate")
                one("hidden_only_not_found", path="/hidden", expect="not_found")
                one(
                    "random_absent_not_found",
                    query="absent-" + secrets.token_hex(12),
                    expect="not_found",
                )

                def stale(browser, page, query, **kwargs):
                    browser.evaluate("document.querySelector('h1').textContent+=' changed'")
                    return profile.navigate(browser, page, query, **kwargs)

                one("stale_before_navigation", navigator=stale, expect="stale")

                def unknown(browser, page, query, **kwargs):
                    profile.navigate(browser, page, query, **kwargs)
                    raise browser_module.StalePage(
                        "Owned injected exception after actual CDP scroll"
                    )

                _, state = one("post_dispatch_stale_unknown", navigator=unknown, expect="stale")
                assert (
                    state["find_text_feedback"][-1]["scroll_requested"] is None
                    and result["cases"][-1]["after"]["in_view"] is True
                )
                start = time.monotonic()
                _, state = one(
                    "real_late_helper_no_navigation",
                    hook=lambda: time.sleep(20.02),
                    expect="deadline",
                )
                assert (
                    time.monotonic() - start >= 20
                    and state["find_text_feedback"][-1]["scroll_requested"] is False
                )
                cls = profile.make_agent_class(
                    agent_module, model_module, marker_expression=browser_module.MARKER
                )
                io.hook = lambda: None
                with cls(
                    base + "/series", "Exercise sixteen owned literal navigation actions"
                ) as agent:
                    for index, query in enumerate(series):
                        io.query = query
                        state = agent.command("tick")
                        assert len(state["history"]) == index + 1 and state["status"] == "ready"
                        assert visible(agent.browser, index)["in_view"] is True
                    before_counts = (len(io.text), len(io.policy), transport.calls)
                    agent.command("predict")
                    try:
                        agent.command("act", {"fingerprint": agent.state["page"]["fingerprint"]})
                    except ValueError as error:
                        assert "sixteen-action" in str(error)
                    else:
                        raise AssertionError("Seventeenth action was admitted")
                    assert len(io.text) == before_counts[0] and len(agent.state["history"]) == 16
                    result["cases"].append(
                        {
                            "name": "sixteen_actions_then_refusal",
                            "actions": 16,
                            "decisions": len(agent.state["decisions"]),
                            "seventeenth_action_refused": True,
                            "helper_calls_after_budget": 0,
                        }
                    )
                assert not blocked and all(path in documents for path in requests)
                result.update(
                    wire_inverse=wire_proof,
                    protocol_inverse=inverse,
                    native_calls=len(io.native_calls),
                    policy_calls=len(io.policy),
                    helper_calls=len(io.text),
                    local_document_requests=len(requests),
                    blocked_nonlocal_requests=len(blocked),
                    cdp_calls=transport.calls,
                    completed=True,
                )
                (output / "join-ledger.json").write_text(
                    json.dumps({"policy": io.policy, "helper": io.text}, indent=2) + "\n"
                )
                assert "torch" not in sys.modules and "transformers" not in sys.modules
            finally:
                # Scope is the owned browser's actual process population observed
                # at admission and closure, not a universal no-escape claim.
                cleanup_errors = []
                try:
                    if "session" in locals():
                        seen = {
                            (row["pid"], row["start_ticks"]) for row in result["owned_processes"]
                        }
                        for process in session.send("SystemInfo.getProcessInfo")["processInfo"]:
                            row = proc_identity(int(process["id"]))
                            if row and (row["pid"], row["start_ticks"]) not in seen:
                                result["owned_processes"].append(row)
                except Exception as error:
                    cleanup_errors.append(
                        {"phase": "final_process_read", "type": type(error).__name__}
                    )
                try:
                    browser.close()
                except Exception as error:
                    cleanup_errors.append({"phase": "browser_close", "type": type(error).__name__})
                result["cleanup_errors"] = cleanup_errors
                if cleanup_errors and sys.exc_info()[0] is None:
                    raise ValueError("Owned browser cleanup evidence incomplete")
    except BaseException as error:
        primary = error
        result.update(completed=False, error={"type": type(error).__name__, "message": str(error)})
    finally:
        for key, value in prior.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        rows = []
        for row in result["owned_processes"]:
            after = proc_identity(row["pid"])
            rows.append(
                {
                    "before": row,
                    "after": after,
                    "same_process_absent": after is None
                    or after["start_ticks"] != row["start_ticks"],
                }
            )
        result["process_closure"] = rows
        result["closed_observed_processes"] = bool(rows) and all(
            row["same_process_absent"] for row in rows
        )
        (output / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    if primary:
        raise primary
    if not result["closed_observed_processes"]:
        raise ValueError("Observed owned Chromium process closure unproven")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, default=HERE / "actual-01")
    args = parser.parse_args(argv)
    for label in PINS:
        checked(ROOT / label)
    if not args.execute:
        print(
            json.dumps(
                {
                    "fixture_only": True,
                    "execute_requested": False,
                    "actual_browser_started": False,
                    "cases": 7,
                    "gpu_disabled": True,
                    "quality_score": None,
                }
            )
        )
        return 0
    if args.output.exists():
        raise ValueError("Fresh fixture output required")
    args.output.mkdir(parents=True)
    result = execute(args.output)
    print(
        json.dumps(
            {
                "completed": result["completed"],
                "cases": len(result["cases"]),
                "closed_observed_processes": result["closed_observed_processes"],
                "quality_score": None,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
