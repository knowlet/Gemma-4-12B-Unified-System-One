"""Selected source→real API→native DTO/fake numbers→wire→actual Agent/Chromium."""

import copy
import hashlib
import json
import os
import secrets
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SELECTION_SHA = "063e5afde12bc408f18be446abedac3e66ff064a6e81858c6b7ce4c995719cb0"


def load(path, name, expected=None):
    blob = path.read_bytes()
    if expected:
        assert hashlib.sha256(blob).hexdigest() == expected
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(blob, str(path), "exec", dont_inherit=True), vars(module))
    return module


integration = load(HERE / "integration.py", "_integration_test_source")


@pytest.fixture
def bundle():
    with integration.bootstrap_selected(
        (HERE / "selection.json").read_bytes(), SELECTION_SHA, profile=integration.PROFILE
    ) as selected:
        yield selected
    assert not any(name == "s1" or name.startswith("s1.") for name in sys.modules)


def payload(width=2):
    return {
        "model": "fixture-integration-only",
        "state": {"page": {"url": "http://owned.invalid/", "text": "owned fixture"}},
        "questions": {
            "operation": {
                "type": "choice",
                "instructions": {"goal": "Owned fixture"},
                "criteria": {
                    "FIND_TEXT": "Find literal text",
                    "CLICK": "Click",
                    "TYPE_TEXT": "Type",
                    "SELECT": "Select",
                    "WAIT": "Wait",
                    "DONE": "Done",
                    "BLOCKED": "Blocked",
                    "SCROLL_UP": "Up",
                    "SCROLL_DOWN": "Down",
                },
            },
            "click_target": {
                "type": "choice",
                "instructions": "All original labels",
                "criteria": {str(i): {"text": "candidate", "row": i} for i in range(width)},
            },
            "type_text_target": {
                "type": "choice",
                "instructions": "Singleton still evaluated",
                "criteria": {"field": {"label": "Owned field"}},
            },
            "select_target": {
                "type": "choice",
                "instructions": "Original order",
                "criteria": {"option:2": "Two", "option:1": "One"},
            },
        },
    }


class NumericalFixture:
    def __init__(self):
        from s1.evaluation.decision3 import decision3_spec

        self.spec = decision3_spec("d3-edge")
        self.name, self.calls, self.corrupt = "fixture-integration-only", [], None

    def predict(self, request, *, deadline, evidence_sink):
        deadline()
        self.calls.append(request)
        answers = {}
        for question in request.questions:
            choice = "FIND_TEXT" if question.id == "operation" else question.labels()[0]
            answers[question.id] = {
                "type": "choice",
                "choice": choice,
                "confidence": 1.0,
                "probabilities": {label: float(label == choice) for label in question.labels()},
            }
        if self.corrupt == "argmax":
            answers["operation"]["choice"] = "BLOCKED"
        if self.corrupt == "missing_head":
            del answers["select_target"]
        return {
            "model": self.spec.model_id.rsplit("/", 1)[-1],
            "answers": answers,
            "usage": {"input_tokens": 1, "output_tokens": 0},
        }


def app(bundle, backend, events):
    return bundle.create_app(
        backend,
        provider="decision3-native",
        model_identity=backend.name,
        api_key="fixture-only",
        evidence_sink=events.append,
    )


def test_default_dry_does_not_import_project_or_construct_backend(capsys):
    assert integration.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert (
        result["source_only"] and not result["backend_constructed"] and not result["server_started"]
    )
    assert not any(name == "s1" or name.startswith("s1.") for name in sys.modules)


@pytest.mark.parametrize("kind", ["profile", "sha", "component", "missing_api"])
def test_descriptor_refuses_before_import_or_backend(kind):
    value = json.loads((HERE / "selection.json").read_bytes())
    profile = integration.PROFILE
    if kind == "profile":
        profile = None
    elif kind == "component":
        value["components"]["wire"]["sha256"] = "0" * 64
    elif kind == "missing_api":
        value["project_sources"] = [
            r for r in value["project_sources"] if r["path"] != "src/s1/api.py"
        ]
    blob = json.dumps(value).encode()
    selected_sha = "0" * 64 if kind == "sha" else hashlib.sha256(blob).hexdigest()
    with pytest.raises(ValueError):
        with integration.bootstrap_selected(blob, selected_sha, profile=profile):
            pytest.fail("Invalid source selection entered")
    assert not any(name == "s1" or name.startswith("s1.") for name in sys.modules)


def test_preloaded_original_is_never_replaced(monkeypatch):
    sentinel = types.ModuleType("s1")
    monkeypatch.setitem(sys.modules, "s1", sentinel)
    with pytest.raises(ValueError, match="Fresh selected"):
        with integration.bootstrap_selected(
            (HERE / "selection.json").read_bytes(), SELECTION_SHA, profile=integration.PROFILE
        ):
            pytest.fail("Preloaded source was replaced")
    assert sys.modules["s1"] is sentinel


def test_real_server_native_all_heads_and_original_parser_refusal(bundle):
    from s1.errors import RequestValidationError
    from s1.jev_generated import OPERATIONS, _request

    backend, events = NumericalFixture(), []
    request = payload()
    with pytest.raises(RequestValidationError):
        _request(request, backend.name)
    with TestClient(app(bundle, backend, events)) as client:
        raw = client.post(
            "/v1/systemone", json=request, headers={"Authorization": "Bearer fixture-only"}
        )
    assert raw.status_code == 200
    response = bundle.named_response(request, raw.json())
    assert list(response["answers"]) == list(request["questions"])
    assert response["answers"]["operation"]["choice"] == "FIND_TEXT"
    assert [q.id for q in backend.calls[0].questions] == list(request["questions"])
    assert [q.labels() for q in backend.calls[0].questions] == [
        list(q["criteria"]) for q in request["questions"].values()
    ]
    assert all(e["data"]["payload"] == request for e in events if e["event"] == "request_started")
    assert any(e["event"] == "server_returned" for e in events)
    assert len(OPERATIONS) == 8 and "FIND_TEXT" not in OPERATIONS
    assert all(p["inverse_exact"] for p in bundle.proofs["protocol"].values())
    assert bundle.proofs["api"]["inverse_exact"] and bundle.proofs["wire"]["inverse_exact"]
    original = bundle.selection
    original["components"].clear()
    bundle.verify()  # public returned selection cannot alter the private authority


@pytest.mark.parametrize(
    "kind,status", [("auth", 401), ("identity", 422), ("wide", 422), ("unknown_op", 422)]
)
def test_real_server_refusals_preserve_full_request_before_forward(bundle, kind, status):
    backend, events, request = NumericalFixture(), [], payload()
    token = "fixture-only"
    if kind == "auth":
        token = "wrong-fixture-key"
    elif kind == "identity":
        request["model"] = "different-model"
    elif kind == "wide":
        request = payload(256)
    else:
        request["questions"]["operation"]["criteria"]["GOLD_ROUTE"] = "unsupported fixture label"
    original = copy.deepcopy(request)
    with TestClient(app(bundle, backend, events)) as client:
        result = client.post(
            "/v1/systemone", json=request, headers={"Authorization": "Bearer " + token}
        )
    assert result.status_code == status and backend.calls == [] and request == original


def test_real_native255_server_and_singletons(bundle):
    backend, events = NumericalFixture(), []
    request = payload(255)
    with TestClient(app(bundle, backend, events)) as client:
        response = client.post(
            "/v1/systemone", json=request, headers={"Authorization": "Bearer fixture-only"}
        )
    assert response.status_code == 200
    converted = bundle.named_response(request, response.json())
    assert len(converted["answers"]["click_target"]["probabilities"]) == 255
    assert converted["answers"]["type_text_target"]["probabilities"] == {"field": 1.0}


@pytest.mark.parametrize("kind", ["argmax", "missing_head"])
def test_real_server_rejects_invalid_native_output_without_repair(bundle, kind):
    backend, events = NumericalFixture(), []
    backend.corrupt = kind
    with TestClient(app(bundle, backend, events)) as client:
        response = client.post(
            "/v1/systemone", json=payload(), headers={"Authorization": "Bearer fixture-only"}
        )
    assert response.status_code == 502 and len(backend.calls) == 1


@pytest.mark.parametrize("kind", ["order", "version", "request"])
def test_explicit_wire_refuses_repair_or_relabel(bundle, kind):
    backend, events, request = NumericalFixture(), [], payload()
    with TestClient(app(bundle, backend, events)) as client:
        response = client.post(
            "/v1/systemone", json=request, headers={"Authorization": "Bearer fixture-only"}
        ).json()
    if kind == "order":
        response["answers"].reverse()
    elif kind == "version":
        response["decision3_native"]["version"] = "jev_decision3_native_v1"
    else:
        request["state"]["extra"] = "not the original"
    with pytest.raises(ValueError):
        bundle.named_response(request, response)


def test_checked_project_loader_and_closed_bundle():
    with integration.bootstrap_selected(
        (HERE / "selection.json").read_bytes(), SELECTION_SHA, profile=integration.PROFILE
    ) as bundle:
        import s1.contracts

        assert type(s1.contracts.__loader__).__name__ == "ProjectSources"
        assert s1.contracts.__file__ == str(ROOT / "src/s1/contracts.py")
    with pytest.raises(ValueError, match="closed"):
        bundle.verify()


def test_actual_owned_chromium_agent_uses_selected_server_wire_native(
    bundle, monkeypatch, tmp_path
):
    from playwright.sync_api import sync_playwright

    fixture = load(
        ROOT / "artifacts/jev-runtime/browser-find-text-cpu-browser-20261022-01/run_fixture.py",
        "_selected_browser_fixture",
        "67bfd8a83068c3d0fa6ea8dc3c5f4edd22f3834eb2a1dab21fbb6dcbd2d01bbc",
    )
    fixture.binary_pin()
    namespace = {
        "copy": copy,
        "time": __import__("time"),
        "urlsplit": __import__("urllib.parse", fromlist=["urlsplit"]).urlsplit,
    }
    fixture.selected_nodes(ROOT / "scripts/jev_browser_benchmark.py", namespace, ["PlaywrightCDP"])
    needle = "integration-owned-" + secrets.token_hex(12)
    variables = {
        "TYPESAFE_MODEL": "fixture-find-only",
        "TYPESAFE_API_KEY": "fixture-only",
        "TEXT_MODEL": "generated-actions-none",
        "TEXT_MODEL_API_KEY": "fixture-only",
        "TEXT_MODEL_BASE_URL": "http://fixture.invalid",
        "TEXT_MODEL_REASONING": "none",
    }
    for key, value in variables.items():
        monkeypatch.setenv(key, value)
    io = fixture.FixtureIO(bundle.protocol["jev_decision3_native"], bundle.wire)
    events, exchanges = [], []
    selected_app = bundle.create_app(
        io,
        provider="decision3-native",
        model_identity=io.name,
        api_key="fixture-only",
        evidence_sink=events.append,
    )
    original_post = io.post
    with (
        TestClient(selected_app) as server_client,
        fixture.server({"/owned": fixture.document([needle])}) as (base, _requests),
        sync_playwright() as playwright,
    ):
        browser = playwright.chromium.launch(
            executable_path=str(fixture.CHROMIUM),
            headless=True,
            args=fixture.FLAGS,
            env={"PATH": os.defpath, "LANG": "C.UTF-8"},
            timeout=15000,
        )
        try:

            def post(url, *, json, headers):
                if "questions" not in json:
                    return original_post(url, json=json, headers=headers)
                raw = server_client.post("/v1/systemone", json=json, headers=headers)
                assert raw.status_code == 200
                converted = bundle.named_response(json, raw.json())
                exchanges.append(
                    {"request": copy.deepcopy(json), "raw": raw.json(), "converted": converted}
                )
                return types.SimpleNamespace(
                    status_code=200, is_error=False, json=lambda: converted
                )

            io.post = post
            transport = namespace["PlaywrightCDP"](browser)
            agent_module, model_module, browser_module = fixture.private_upstream(transport, io)
            io.system_prompt, io.query = model_module.TEXT_VALUE, needle
            cls = bundle.agent_class(
                agent_module, model_module, marker_expression=browser_module.MARKER
            )
            with cls(base + "/owned", "Find literal text " + needle) as agent:
                before = fixture.visible(agent.browser)
                result = agent.command("tick")
                after = fixture.visible(agent.browser)
                assert before["in_view"] is False and after["in_view"] is True
                assert len(result["history"]) == 1 and result["status"] != "done"
                assert result["find_text_feedback"][-1]["query"] == io.text[0]["query"] == needle
                assert agent.state["decisions"][0]["request"] == exchanges[0]["request"]
                assert (
                    exchanges[0]["raw"]["decision3_native"]["version"]
                    == "jev_decision3_native_find_text_experimental_v1"
                )
                assert list(exchanges[0]["converted"]["answers"]) == list(
                    exchanges[0]["request"]["questions"]
                )
                assert len(io.native_calls) == len(exchanges) == len(io.text) == 1
                assert any(e["event"] == "server_returned" for e in events)
                proof = {
                    "fixture_only": True,
                    "selection_sha256": SELECTION_SHA,
                    "server_status": 200,
                    "native_profile": exchanges[0]["raw"]["decision3_native"]["version"],
                    "wire_profile": exchanges[0]["converted"]["decision3_browser_wire"]["profile"],
                    "request_sha256": exchanges[0]["converted"]["decision3_browser_wire"][
                        "original_request_sha256"
                    ],
                    "question_ids": list(exchanges[0]["request"]["questions"]),
                    "native_calls": 1,
                    "helper_schema_valid": io.text[0]["schema_valid"],
                    "helper_query_sha256": hashlib.sha256(needle.encode()).hexdigest(),
                    "helper_query_equals_executed_query": True,
                    "actions": len(result["history"]),
                    "before_in_view": before["in_view"],
                    "after_in_view": after["in_view"],
                    "task_success_claimed": False,
                    "quality_score": None,
                }
        finally:
            browser.close()
    proof["owned_browser_close_returned"] = True
    (tmp_path / "source-integration-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    assert "torch" not in sys.modules and "transformers" not in sys.modules
