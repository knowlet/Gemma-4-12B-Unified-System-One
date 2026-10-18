"""Actual legacy producer to pinned upstream choose(), with no network or ML."""

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from s1.evaluation.decision3 import decision3_spec
from s1.jev_decision3_native import JevDecision3NativeAdapter

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/jev_decision3_browser_wire.py"
SPEC = importlib.util.spec_from_file_location("decision3_browser_wire_test_subject", SCRIPT)
WIRE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WIRE)
UPSTREAM = ROOT / "artifacts/jev-runtime/upstream/jev_ultrafast"
PINS = {
    "model.py": "a85ada8458d23642c5c8a4095acf0565e6268f81dcda26f6308942b653e9c9f3",
    "questions.py": "66a7054b85040b949e1757311257ea0c5a843c635f9a40c99f6d15ad660fb422",
}
OPERATIONS = ("CLICK", "TYPE_TEXT", "SELECT", "SCROLL_DOWN", "SCROLL_UP", "WAIT", "DONE", "BLOCKED")
TARGETS = {"CLICK": "click_target", "TYPE_TEXT": "type_text_target", "SELECT": "select_target"}


def upstream():
    import math

    blobs = {name: (UPSTREAM / name).read_bytes() for name in PINS}
    assert all(hashlib.sha256(blobs[name]).hexdigest() == sha for name, sha in PINS.items())
    functions = {"validate_choice", "action_space", "choose"}
    tree = ast.parse(blobs["model.py"])
    body = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in functions
    ]
    assert {node.name for node in body} == functions
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(blobs["questions.py"]).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in {"NEXT_ACTION", "TARGET"}
    }
    namespace = {
        **constants,
        "math": math,
        "os": SimpleNamespace(
            environ={"TYPESAFE_MODEL": "wire-test", "TYPESAFE_API_KEY": "test-only"}
        ),
        "time": SimpleNamespace(perf_counter=lambda: 1.0),
    }
    # Only these already-read pure functions run; no httpx client, module import,
    # request, field generation, browser, agent executor or publisher code runs.
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(UPSTREAM / "model.py"), "exec"),
        namespace,
    )
    return namespace


def state():
    actions = []
    for kind in ("click", "fill"):
        for suffix in ("a", "b"):
            actions.append(
                {
                    "kind": kind,
                    "id": kind + "-" + suffix,
                    "node": kind + "-node-" + suffix,
                    "role": "button" if kind == "click" else "textbox",
                    "label": "Synthetic " + kind + " " + suffix,
                    "value": "",
                }
            )
    for suffix in ("a", "b"):
        actions.append(
            {
                "kind": "select",
                "id": "select-" + suffix,
                "node": "select-node",
                "role": "combobox",
                "label": "Synthetic mode → " + suffix,
                "value": suffix,
                "current_value": "unselected-synthetic-value",
            }
        )
    actions.extend(
        {"kind": "wait" if name == "wait" else "scroll", "id": name, "label": name}
        for name in ("scroll_down", "scroll_up", "wait")
    )
    return {
        "url": "https://example.invalid/synthetic-controls",
        "title": "Pure wire fixture",
        "text": "Synthetic observations, not a benchmark or model prediction.",
        "actions": actions,
    }


class NativeWithoutML:
    def __init__(self, operation):
        self.spec = decision3_spec("d3-edge")
        self.name, self.operation, self.calls = self.spec.id, operation, []

    def predict(self, request, *, deadline=None, evidence_sink=None):
        self.calls.append(request)
        answers = {}
        for question in request.questions:
            labels = question.labels()
            chosen = self.operation if question.id == "operation" else labels[-1]
            values = {
                label: 1.0
                if len(labels) == 1
                else 0.75
                if label == chosen
                else 0.25 / (len(labels) - 1)
                for label in labels
            }
            answers[question.id] = {
                "type": "choice",
                "choice": chosen,
                "probabilities": values,
                "confidence": 0.2,
            }
        return {
            "model": "d3-edge",
            "answers": answers,
            "usage": {"input_tokens": 400, "output_tokens": 0},
            "cost": {"usd": None},
            "future_metadata": {"ordered": [3, 1, 2], "source": "synthetic test only"},
        }


def consumer(operation, *, convert):
    native = NativeWithoutML(operation)
    adapter = JevDecision3NativeAdapter(native, model_identity="wire-test")
    namespace, captures = upstream(), []

    def network_stub(url, key, body):
        assert url == "https://api.typesafe.ai/v1/systemone" and key == "test-only"
        before = copy.deepcopy(body)
        legacy = adapter.predict(body)
        assert body == before
        converted = WIRE.named_browser_response(body, legacy) if convert else legacy
        captures.append((copy.deepcopy(body), legacy, converted))
        # Match the real JSON HTTP boundary without opening a connection.
        return json.loads(json.dumps(converted, allow_nan=False))

    namespace["post_json"] = network_stub
    return namespace["choose"], captures, native


def legacy_pair():
    choose, captures, _ = consumer("CLICK", convert=False)
    with pytest.raises(AttributeError, match="has no attribute 'get'"):
        choose(state(), "Advance the synthetic controls.", [])
    request, response, _ = captures[0]
    return request, response


def test_unmodified_legacy_response_fails_the_actual_pinned_upstream_consumer():
    request, response = legacy_pair()
    assert type(response["answers"]) is list
    assert [answer["id"] for answer in response["answers"]] == list(request["questions"])


@pytest.mark.parametrize("operation", OPERATIONS)
def test_actual_producer_to_upstream_choose_preserves_all_heads_and_selected_actions(operation):
    choose, captures, native = consumer(operation, convert=True)
    result = choose(state(), "Advance the synthetic controls.", [])
    assert len(native.calls) == len(captures) == 1
    request, legacy, converted = captures[0]
    assert result["operation"] == operation and result["model"] == legacy["model"] == "wire-test"
    assert list(converted["answers"]) == list(request["questions"])
    for answer in legacy["answers"]:
        assert converted["answers"][answer["id"]] == answer
        assert json.dumps(converted["answers"][answer["id"]]) == json.dumps(answer)
    assert {
        key: value for key, value in converted.items() if key not in {"answers", WIRE.METADATA_KEY}
    } == {key: value for key, value in legacy.items() if key != "answers"}
    assert converted[WIRE.METADATA_KEY]["profile"] == WIRE.PROFILE
    assert converted["decision3_native"] == legacy["decision3_native"]
    assert result["raw_answers"] == converted["answers"]
    if operation in TARGETS:
        head = TARGETS[operation]
        answer = converted["answers"][head]
        assert result["target"] == answer["choice"]
        assert result["target_probabilities"] == answer["probabilities"]
        assert (
            result["choice"]
            == {"CLICK": "click-b", "TYPE_TEXT": "fill-b", "SELECT": "select-b"}[operation]
        )
    else:
        assert result["target"] is None
        expected = operation if operation in {"DONE", "BLOCKED"} else operation.lower()
        assert result["choice"] == expected


def test_conversion_deep_copies_request_bound_values_without_changing_inputs():
    request, response = legacy_pair()
    original_request, original_response = copy.deepcopy(request), copy.deepcopy(response)
    result = WIRE.named_browser_response(request, response)
    result["answers"]["operation"]["probabilities"]["CLICK"] = 0.0
    result["usage"]["input_tokens"] = 0
    result["cost"]["usd"] = 1
    result["decision3_native"]["original_request"]["state"]["page"]["text"] = "changed copy"
    result["future_metadata"]["ordered"].append(9)
    result[WIRE.METADATA_KEY]["question_ids"].append("new")
    assert request == original_request and response == original_response


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "unknown",
        "reordered",
        "missing_id",
        "nonobject",
        "named",
        "metadata",
        "profile",
        "request",
        "candidate_order",
        "nonfinite",
    ],
)
def test_invalid_formats_are_refused_without_repair_or_input_mutation(fault):
    request, response = legacy_pair()
    if fault == "missing":
        response["answers"].pop()
    elif fault == "duplicate":
        response["answers"][-1] = copy.deepcopy(response["answers"][0])
    elif fault == "unknown":
        response["answers"][-1]["id"] = "unrequested"
    elif fault == "reordered":
        response["answers"].reverse()
    elif fault == "missing_id":
        response["answers"][0].pop("id")
    elif fault == "nonobject":
        response["answers"][0] = []
    elif fault == "named":
        response["answers"] = {answer["id"]: answer for answer in response["answers"]}
    elif fault == "metadata":
        response[WIRE.METADATA_KEY] = {"profile": "existing"}
    elif fault == "profile":
        response["decision3_native"]["version"] = "another-profile"
    elif fault == "request":
        request["state"]["page"]["title"] = "Different request with the same head IDs"
    elif fault == "candidate_order":
        question = request["questions"]["click_target"]
        question["criteria"] = dict(reversed(list(question["criteria"].items())))
    else:
        response["cost"]["usd"] = float("inf")
    before = copy.deepcopy(response)
    with pytest.raises(ValueError):
        WIRE.named_browser_response(request, response)
    assert response == before


def test_helper_imports_only_standard_library_and_existing_api_does_not_enable_it():
    imports = {
        alias.name
        for node in ast.parse(SCRIPT.read_bytes()).body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert imports == {"copy", "hashlib", "json"}
    assert "jev_decision3_browser_wire" not in (ROOT / "src/s1/api.py").read_text()
