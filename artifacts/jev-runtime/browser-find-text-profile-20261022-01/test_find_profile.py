"""Actual selected Python bodies and fake CDP/DOM/I/O; no browser or model."""

import ast
import copy
import json
import math
import os
import shutil
import subprocess
import sys
import time
import types
import uuid
from pathlib import Path

import pytest

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))


def load(name, path):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), vars(module))
    return module


profile = load("find_profile_component", HERE / "find_profile.py")
protocol = load("find_protocol_component", HERE / "private_protocol.py")


def functions(path, names, namespace):
    tree = ast.parse(path.read_bytes())
    selected = [
        n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    assert {n.name for n in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def page():
    return {
        "url": "https://fixture.invalid/document",
        "title": "Fixture document",
        "text": "Visible opening",
        "fingerprint": "before",
        "marker": "marker-before",
        "scroll": {"y": 0, "height": 50000},
        "actions": [
            {"id": "click1", "kind": "click", "node": 1, "label": "Generic link", "role": "link"},
            {
                "id": "fill2",
                "kind": "fill",
                "node": 2,
                "label": "Generic input",
                "role": "textbox",
                "value": "",
            },
            {
                "id": "select3",
                "kind": "select",
                "node": 3,
                "label": "Generic option",
                "role": "combobox",
                "value": "value",
            },
            {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 560},
            {"id": "scroll_up", "kind": "scroll", "label": "Scroll up", "delta": -560},
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ],
    }


@pytest.fixture
def fixture(monkeypatch):
    prefix = "find_fixture_" + uuid.uuid4().hex
    package = types.ModuleType(prefix)
    package.__path__ = []
    monkeypatch.setitem(sys.modules, prefix, package)
    questions = load(prefix + ".questions", profile.UPSTREAM / "questions.py")
    browser = types.ModuleType(prefix + ".browser")
    functions(profile.UPSTREAM / "browser.py", ["StalePage"], vars(browser))

    class FakeBrowser:
        def __init__(self, _url):
            self.page, self.fresh_values, self.navigations = page(), [], []
            self.inputs = []

        def observe(self, **_kwargs):
            return copy.deepcopy(self.page)

        def fresh(self, _page, _action=None):
            return self.fresh_values.pop(0) if self.fresh_values else True

        def act(self, action, _page, text=None):
            self.inputs.append((action, text))

        def close(self):
            pass

    browser.Browser = FakeBrowser
    monkeypatch.setitem(sys.modules, prefix + ".browser", browser)
    model = types.ModuleType(prefix + ".model")
    model.__file__ = str(profile.UPSTREAM / "model.py")
    model.__dict__.update(json=json, math=math, os=os, time=time)
    for name in ("NEXT_ACTION", "TARGET", "TEXT_VALUE"):
        setattr(model, name, getattr(questions, name))
    sent, helper_requests = [], []
    options = {
        "operation": "FIND_TEXT",
        "query": "a generic section",
        "helper_hook": lambda: None,
        "result": "found",
    }

    def post(_url, _key, body):
        if "questions" not in body:
            helper_requests.append(copy.deepcopy(body))
            options["helper_hook"]()
            return {
                "choices": [{"message": {"content": json.dumps({"text": options["query"]})}}],
                "usage": {},
            }
        sent.append(copy.deepcopy(body))
        answers = {}
        for key, question in body["questions"].items():
            labels = list(question["criteria"])
            chosen = options["operation"] if key == "operation" else labels[0]
            answers[key] = {
                "choice": chosen,
                "probabilities": {label: float(label == chosen) for label in labels},
                "confidence": 1.0,
            }
        return {"model": "fixture-native", "answers": answers}

    model.post_json = post
    functions(
        profile.UPSTREAM / "model.py",
        ["validate_choice", "action_space", "choose", "field_context", "field_text"],
        vars(model),
    )
    monkeypatch.setitem(sys.modules, prefix + ".model", model)
    agent = load(prefix + ".agent", profile.UPSTREAM / "agent.py")
    agent.MAX_STEPS = 16
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-no-credential")
    monkeypatch.setenv("TYPESAFE_MODEL", "fixture-native")
    monkeypatch.setenv("TEXT_MODEL_API_KEY", "fake-no-credential")
    monkeypatch.setenv("TEXT_MODEL", "generated-actions-none")
    monkeypatch.setenv("TEXT_MODEL_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("TEXT_MODEL_REASONING", "none")

    def navigate(browser, _page, query, **_kwargs):
        browser.navigations.append(query)
        status = options["result"]
        if status == "found":
            browser.page.update(fingerprint="after", marker="marker-after")
        return {
            "status": status,
            "scroll_requested": status == "found",
            **({"scroll_changed": True} if status == "found" else {}),
        }

    original = {
        "Agent": agent.Agent,
        "command": agent.Agent.command,
        "choose": agent.choose,
        "model_choose": model.choose,
        "space": model.action_space,
        "post": model.post_json,
    }
    cls = profile.make_agent_class(
        agent, model, marker_expression="'marker-before'", navigator=navigate
    )
    instance = cls("https://fixture.invalid/document", "Find the relevant section for my task")
    yield types.SimpleNamespace(
        agent=instance,
        cls=cls,
        module=agent,
        model=model,
        original=original,
        sent=sent,
        helper_requests=helper_requests,
        options=options,
        browser=browser,
        navigate=navigate,
    )
    for name in tuple(sys.modules):
        if name == prefix or name.startswith(prefix + "."):
            del sys.modules[name]


def test_actual_choose_preserves_all_existing_heads_and_targets(fixture):
    before = copy.deepcopy(fixture.agent.state["page"])
    fixture.agent.command("predict")
    body = fixture.sent[-1]
    assert list(body["questions"]) == [
        "operation",
        "click_target",
        "type_text_target",
        "select_target",
    ]
    assert len(body["questions"]["operation"]["criteria"]) == 9
    assert fixture.agent.state["decision"]["target"] is None
    assert fixture.agent.state["decision"]["request"] == body
    assert fixture.agent.state["page"] == before
    original_elements, targets, _ = fixture.model.action_space(before["actions"])
    assert body["state"]["elements"] == original_elements
    for operation, candidates in targets.items():
        assert list(body["questions"][operation.lower() + "_target"]["criteria"]) == list(
            candidates
        )


def test_original_bindings_and_agent_instances_are_unchanged(fixture):
    assert fixture.module.Agent is fixture.original["Agent"]
    assert fixture.module.Agent.command is fixture.original["command"]
    assert fixture.module.choose is fixture.original["choose"]
    assert fixture.model.choose is fixture.original["model_choose"]
    assert fixture.model.post_json is fixture.original["post"]
    old = fixture.module.Agent("https://fixture.invalid/", "same goal")
    fixture.options["operation"] = "WAIT"
    old.command("predict")
    assert "FIND_TEXT" not in fixture.sent[-1]["questions"]["operation"]["criteria"]
    assert "agent_profile" not in fixture.sent[-1]["state"]


def test_find_query_uses_original_q95_schema_and_counts_one_action(fixture):
    result = fixture.agent.command("tick")
    assert result["status"] == "ready"
    assert len(result["history"]) == 1 and len(result["decisions"]) == 1
    assert fixture.agent.browser.navigations == [fixture.options["query"]]
    assert fixture.agent.browser.inputs == []
    body = fixture.helper_requests[0]
    namespace = {"IDENTITY": "generated-actions-none", "json": json}
    namespace.update(
        keys=lambda obj, allowed: require(type(obj) is dict and set(obj) == allowed, "keys"),
        require=require,
        exact=lambda a, b: type(a) is type(b) and a == b,
        canonical=lambda x: json.dumps(x, separators=(",", ":")).encode(),
        strict_json=json.loads,
    )
    functions(
        ROOT / "scripts/jev_qualify_public_field_syntax_bf16.py",
        ["parse_helper_request"],
        namespace,
    )
    context = namespace["parse_helper_request"](body, system_prompt=fixture.model.TEXT_VALUE)
    assert context["field"]["role"] == "browser_find_query"
    assert context["goal"] == fixture.agent.state["goal"]
    assert context["page"]["text"] == "Visible opening"
    assert result["history"][0]["target"] is None
    assert result["find_text_feedback"][0]["task_success"] == "not_evaluated"


def require(value, message):
    if not value:
        raise ValueError(message)


@pytest.mark.parametrize("status", ["not_found", "scan_limit"])
def test_failed_find_is_action_and_typed_feedback_not_success(fixture, status):
    fixture.options["result"] = status
    result = fixture.agent.command("tick")
    assert len(result["history"]) == 1 and result["status"] == "ready"
    assert result["find_text_feedback"][0]["scroll_requested"] is False
    fixture.agent.command("predict")
    assert fixture.sent[-1]["state"]["find_text_feedback"][0]["status"] == status
    assert fixture.sent[-1]["state"]["recent_actions"][0]["action"].endswith(status)


@pytest.mark.parametrize("fresh", [[False], [True, False]])
def test_stale_before_or_after_helper_records_no_navigation(fixture, fresh):
    fixture.agent.command("predict")
    fixture.agent.browser.fresh_values = fresh
    with pytest.raises(fixture.browser.StalePage):
        fixture.agent.command("act", {"fingerprint": "before"})
    assert fixture.agent.state["decision"] is None
    assert len(fixture.agent.state["history"]) == 1
    assert fixture.agent.state["find_text_feedback"][0]["status"] == "stale"
    assert fixture.agent.browser.navigations == []
    assert fixture.agent.browser.inputs == []


def test_late_helper_does_not_navigate(fixture):
    now = [0.0]
    fixture.options["helper_hook"] = lambda: now.__setitem__(0, 21.0)
    cls = profile.make_agent_class(
        fixture.module,
        fixture.model,
        marker_expression="'marker-before'",
        navigator=fixture.navigate,
        clock=lambda: now[0],
    )
    agent = cls("https://fixture.invalid/", "same task")
    result = agent.command("tick")
    assert result["find_text_feedback"][0]["status"] == "deadline"
    assert result["find_text_feedback"][0]["late"] is True
    assert agent.browser.navigations == [] and len(result["history"]) == 1


def test_helper_invalid_query_does_not_navigate_and_preserves_error(fixture):
    fixture.options["query"] = "  "
    with pytest.raises(ValueError, match="valid field value"):
        fixture.agent.command("tick")
    row = fixture.agent.state["find_text_feedback"][0]
    assert row["status"] == "error" and row["scroll_requested"] is False
    assert fixture.agent.browser.navigations == []
    assert len(fixture.agent.state["history"]) == 1


def test_original_sixteen_action_and_prediction_budgets(fixture):
    fixture.agent.state["history"] = [{} for _ in range(16)]
    fixture.agent.command("predict")
    with pytest.raises(ValueError, match="sixteen-action"):
        fixture.agent.command("act", {"fingerprint": "before"})
    assert fixture.agent.browser.navigations == [] and fixture.helper_requests == []
    fixture.agent.state.update(status="ready", decisions=[{} for _ in range(32)])
    with pytest.raises(ValueError, match="model-call budget"):
        fixture.agent.command("predict")


def test_feedback_is_per_agent_and_bounded(fixture):
    fixture.agent.command("tick")
    second = fixture.cls("https://fixture.invalid/", "another task")
    second.command("predict")
    assert fixture.sent[-1]["state"]["find_text_feedback"] == []
    for index in range(5):
        fixture.agent.browser.page.update(fingerprint=str(index), marker=str(index))
        fixture.agent.state["page"] = fixture.agent.browser.observe()
        fixture.agent.command("tick")
    assert len(fixture.agent.state["find_text_feedback"]) == 4


def test_original_operations_still_delegate_unchanged(fixture):
    fixture.options["operation"] = "CLICK"
    result = fixture.agent.command("tick")
    assert len(fixture.agent.browser.inputs) == 1
    assert fixture.agent.browser.navigations == [] and fixture.helper_requests == []
    assert result["history"][0]["operation"] == "CLICK"
    assert "find_text_feedback" not in result


def test_stale_navigation_gate_counts_attempt_without_claiming_scroll(fixture):
    fixture.options["result"] = "stale"
    result = fixture.agent.command("tick")  # original tick catches selected StalePage
    assert len(result["history"]) == 1
    assert result["find_text_feedback"][0]["status"] == "stale"
    assert result["find_text_feedback"][0]["scroll_requested"] is False
    assert result["status"] == "ready"


def test_post_dispatch_stale_exception_preserves_unknown_navigation(fixture):
    def uncertain(browser, _page, query, **_kwargs):
        browser.navigations.append(query)
        raise fixture.browser.StalePage("fixture document changed after dispatch")

    cls = profile.make_agent_class(
        fixture.module, fixture.model, marker_expression="'marker-before'", navigator=uncertain
    )
    agent = cls("https://fixture.invalid/", "same task")
    result = agent.command("tick")
    assert len(result["history"]) == 1
    assert result["find_text_feedback"][0]["status"] == "stale"
    assert result["find_text_feedback"][0]["scroll_requested"] is None
    assert agent.browser.navigations == [fixture.options["query"]]


def test_late_navigation_retains_actual_result_and_late_fact(fixture):
    now = [0.0]

    def late(browser, page, query, **kwargs):
        result = fixture.navigate(browser, page, query, **kwargs)
        now[0] = 21.0
        return result

    cls = profile.make_agent_class(
        fixture.module,
        fixture.model,
        marker_expression="'marker-before'",
        navigator=late,
        clock=lambda: now[0],
    )
    agent = cls("https://fixture.invalid/", "same task")
    result = agent.command("tick")
    fact = result["find_text_feedback"][0]
    assert fact["status"] == "found" and fact["scroll_requested"] is True and fact["late"] is True
    assert fact["physical_control_input"] == "none" and result["status"] != "done"


def test_invalid_probability_choice_is_rejected_before_helper(fixture):
    fixture.options["operation"] = "UNSUPPORTED_OPERATION"
    with pytest.raises(ValueError, match="Invalid TypeSafe response"):
        fixture.agent.command("predict")
    assert not fixture.helper_requests and not fixture.agent.browser.navigations


def native_payload(fixture):
    fixture.agent.command("predict")
    return fixture.sent[-1]


def test_private_original_inverse_native_full_payload_argmax(fixture):
    from s1.errors import BackendResponseError, RequestValidationError
    from s1.evaluation.decision3 import MAX_OPTIONS, normalize_native_response
    from s1.jev_generated import OPERATIONS, _request

    payload = native_payload(fixture)
    with pytest.raises(RequestValidationError):
        _request(payload, "fixture-native")
    with protocol.load_private_protocol() as (modules, evidence):
        assert all(item["inverse_exact"] for item in evidence.values())
        native = modules["jev_decision3_native"]
        request, original, source, labels = native.parse_native_payload(
            payload, model_identity="fixture-native"
        )
        assert original == payload and json.loads(source) == payload
        assert list(labels) == [q.id for q in request.questions] == list(payload["questions"])
        assert len(request.questions) == 4 and MAX_OPTIONS == 255
        assert request.questions[0].labels() == list(payload["questions"]["operation"]["criteria"])
        raw = {"answers": {}, "usage": {"input_tokens": 1, "output_tokens": 0}}
        for q in request.questions:
            chosen = "FIND_TEXT" if q.id == "operation" else q.labels()[0]
            raw["answers"][q.id] = {
                "type": "choice",
                "choice": chosen,
                "probabilities": {label: float(label == chosen) for label in q.labels()},
                "confidence": 1.0,
            }
        result = normalize_native_response(request, raw)
        assert result["answers"]["operation"]["choice"] == "FIND_TEXT"
        assert (
            result["answers"]["operation"]["probabilities"]
            == raw["answers"]["operation"]["probabilities"]
        )
        assert list(result["answers"]) == list(payload["questions"])
        broken = copy.deepcopy(raw)
        broken["answers"]["operation"]["choice"] = "BLOCKED"
        with pytest.raises(BackendResponseError):
            normalize_native_response(request, broken)
        broken = copy.deepcopy(raw)
        del broken["answers"]["select_target"]
        with pytest.raises(BackendResponseError):
            normalize_native_response(request, broken)
    assert len(OPERATIONS) == 8 and "FIND_TEXT" not in OPERATIONS
    assert not set(protocol.NAMES.values()) & sys.modules.keys()


def test_private_grammar_targetless_find_and_unchanged_limits():
    from s1.jev_action_grammar import _labels

    labels = {"operation": ["FIND_TEXT", "WAIT", "DONE", "BLOCKED"]}
    with pytest.raises(ValueError):
        _labels(labels)
    with protocol.load_private_protocol() as (modules, _evidence):
        grammar = modules["jev_action_grammar"]
        assert list(grammar._labels(labels)["operation"]) == labels["operation"]

        class Tokenizer:
            all_special_ids = []

            def encode(self, text, **_kwargs):
                return [ord(c) + 1000 for c in text[14:22]]

        # Long/ambiguous artificial encoding must refuse; no token budget increase.
        with pytest.raises(ValueError):
            grammar.LegalActionTokenTrie(labels, Tokenizer(), prompt_length=1, max_new_tokens=2)
        wide = {
            "operation": [
                "CLICK",
                "TYPE_TEXT",
                "SELECT",
                "SCROLL_UP",
                "SCROLL_DOWN",
                "WAIT",
                "DONE",
                "BLOCKED",
                "FIND_TEXT",
            ],
            **{
                name: [str(i) for i in range(400)]
                for name in ("click_target", "type_text_target", "select_target")
            },
        }
        with pytest.raises(ValueError, match="max_legal_actions"):
            grammar.LegalActionTokenTrie(wide, Tokenizer(), prompt_length=1)


def test_native255_candidate_limit_rejects_without_pruning(fixture):
    from s1.errors import RequestValidationError

    payload = native_payload(fixture)
    payload["questions"]["click_target"]["criteria"] = {
        str(i): {"element": str(i)} for i in range(255)
    }
    with protocol.load_private_protocol() as (modules, _evidence):
        native = modules["jev_decision3_native"]
        parsed, original, _, _ = native.parse_native_payload(
            payload, model_identity="fixture-native"
        )
        assert len(parsed.questions[1].criteria) == 255 and original == payload
        payload["questions"]["click_target"]["criteria"]["255"] = {"element": "extra"}
        with pytest.raises(RequestValidationError, match="exceeds255"):
            native.parse_native_payload(payload, model_identity="fixture-native")
        assert len(payload["questions"]["click_target"]["criteria"]) == 256


def test_private_grammar_can_emit_exact_targetless_find():
    with protocol.load_private_protocol() as (modules, _evidence):

        class Tokenizer:
            all_special_ids = []

            def __init__(self):
                self.encoded = {}

            def encode(self, text, **_kwargs):
                return [self.encoded.setdefault(text, 1000 + len(self.encoded))]

        tokenizer = Tokenizer()
        grammar = modules["jev_action_grammar"].LegalActionTokenTrie(
            {"operation": ["FIND_TEXT", "WAIT"]}, tokenizer, prompt_length=1
        )
        text = '{"operation":"FIND_TEXT","target":null}'
        # The original grammar validates the model token suffix, including native EOS.
        grammar.assert_complete([tokenizer.encoded[text], 106])


def test_fake_dom_literal_search_long_page_and_typed_failures():
    node = shutil.which("node")
    if node is None:
        pytest.skip("CPU JavaScript interpreter unavailable")
    query = "literal ' \" \\ text </script>"
    expression = profile.search_expression("globalThis.marker", "same", query, 1.0)
    harness = """
const input=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
globalThis.marker='same'; globalThis.scrollX=0; globalThis.scrollY=0;
globalThis.NodeFilter={SHOW_TEXT:4}; let requests=0;
const parent={closest:()=>false,getClientRects:()=>[{}],checkVisibility:()=>true,
scrollIntoView:()=>{requests++;globalThis.scrollY=40000}};
globalThis.getComputedStyle=()=>({display:'block',visibility:'visible',opacity:'1'});
const nodes=Array.from({length:100},()=>({nodeValue:'unrelated text',parentElement:parent}));
nodes.push({nodeValue:input.query,parentElement:parent});
globalThis.document={body:{},createTreeWalker:()=>{let i=0;return{nextNode:()=>nodes[i++]||null}}};
const found=eval(input.expression); nodes.pop();const missing=eval(input.expression);
globalThis.marker='other';const stale=eval(input.expression);
process.stdout.write(JSON.stringify({found,missing,stale,requests}));
"""
    result = subprocess.run(
        [node, "-e", harness],
        input=json.dumps({"expression": expression, "query": query}),
        text=True,
        capture_output=True,
        timeout=5,
        check=True,
    )
    observed = json.loads(result.stdout)
    assert observed["found"] == {
        "status": "found",
        "scroll_requested": True,
        "scroll_changed": True,
    }
    assert observed["missing"]["status"] == "not_found"
    assert observed["stale"]["status"] == "stale" and observed["requests"] == 1


@pytest.mark.parametrize("scenario", ["scan_limit", "final_stale", "hidden"])
def test_fake_dom_limit_final_freshness_and_hidden_matches(scenario):
    node = shutil.which("node")
    if node is None:
        pytest.skip("CPU JavaScript interpreter unavailable")
    expression = profile.search_expression("globalThis.marker", "same", "generic match", 1.0)
    harness = """
const input=JSON.parse(require('node:fs').readFileSync(0,'utf8'));
globalThis.marker='same';globalThis.scrollX=0;globalThis.scrollY=0;globalThis.NodeFilter={SHOW_TEXT:4};
let requests=0, i=0;
const parent={closest:()=>false,getClientRects:()=>[{}],checkVisibility:()=>input.scenario!=='hidden',scrollIntoView:()=>requests++};
globalThis.getComputedStyle=()=>({display:'block',visibility:'visible',opacity:'1'});
globalThis.document={body:{},createTreeWalker:()=>({nextNode:()=>{
 if(input.scenario==='scan_limit') return i++<50001?{nodeValue:'other',parentElement:parent}:null;
 if(i++) return null;
 if(input.scenario==='final_stale') globalThis.marker='changed';
 return {nodeValue:'generic match',parentElement:parent};
}})};
process.stdout.write(JSON.stringify({result:eval(input.expression),requests}));
"""
    result = subprocess.run(
        [node, "-e", harness],
        input=json.dumps({"expression": expression, "scenario": scenario}),
        text=True,
        capture_output=True,
        timeout=5,
        check=True,
    )
    observed = json.loads(result.stdout)
    assert (
        observed["result"]["status"]
        == {"scan_limit": "scan_limit", "final_stale": "stale", "hidden": "not_found"}[scenario]
    )
    assert observed["result"]["scroll_requested"] is False and observed["requests"] == 0


@pytest.mark.parametrize(
    "response",
    [
        {"exceptionDetails": {}},
        {"result": {"value": {"status": "not_found", "scroll_requested": True}}},
    ],
)
def test_cdp_exception_or_invalid_result_never_claims_success(response):
    class Fake:
        def call(self, method, **params):
            assert method == "Runtime.evaluate" and params["timeout"] <= 500
            return response

    if "exceptionDetails" in response:
        # Real CDP exceptionDetails is nonempty.
        response["exceptionDetails"] = {"text": "timeout"}
        assert (
            profile.navigate(Fake(), page(), "query", marker_expression="null", remaining=1)[
                "scroll_requested"
            ]
            is None
        )
    else:
        with pytest.raises(ValueError):
            profile.navigate(Fake(), page(), "query", marker_expression="null", remaining=1)
