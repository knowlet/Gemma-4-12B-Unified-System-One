"""Real browser transport diagnostics, never measurements of model quality."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import pytest

from s1.contracts import answer_from_probabilities

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "jev_hierarchy_browser_harness", ROOT / "scripts/jev_browser_benchmark.py"
)
HARNESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HARNESS
SPEC.loader.exec_module(HARNESS)


@pytest.fixture
def hierarchical_browser_environment():
    if os.environ.get("JEV_BROWSER_TESTS") != "1":
        pytest.skip("Set JEV_BROWSER_TESTS=1 and JEV_UPSTREAM_PATH for real Chromium checks")
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    playwright = pytest.importorskip("playwright.sync_api")
    upstream = Path(os.environ.get("JEV_UPSTREAM_PATH", "/tmp/s1-jev-ultrafast"))
    HARNESS.upstream_manifest(upstream)
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        try:
            yield upstream, browser
        finally:
            browser.close()


class ObservedControlDiagnostic:
    """Follow this test's natural goal and observed candidates without a model."""

    name = "scripted-browser-diagnostic-not-a-model-score"

    def __init__(self):
        self.calls = []
        self.responses = []

    def predict(self, request):
        self.calls.append(request.model_copy(deep=True))
        answers = {}
        for question in request.questions:
            instructions = json.loads(question.instructions)
            goal = instructions["goal"]
            desired = re.search(r"Activate item (\d+)", goal)
            assert desired is not None
            item = desired.group(1)
            label_text = "Activate item " + item
            labels = question.labels()
            info = instructions.get("hierarchy", {})
            selected = None
            if question.id == "operation":
                confirmed = "Activation confirmed: item " + item in request.state["page"]["text"]
                selected = "DONE" if confirmed else "CLICK"
            elif info.get("stage") == "group":
                for label, description in question.criteria.items():
                    members = json.loads(description)["members"]
                    if any(
                        member["criteria"]["element"].endswith("] " + label_text)
                        for member in members
                    ):
                        selected = label
            elif info.get("stage") == "conditional":
                for label, description in question.criteria.items():
                    if json.loads(description)["element"].endswith("] " + label_text):
                        selected = label
            else:
                raise AssertionError("Unexpected diagnostic question")
            # Every stage has an actual complete distribution, including groups
            # that cannot contain the requested control. No leaf is skipped.
            values = (
                [0.9 if label == selected else 0.1 / (len(labels) - 1) for label in labels]
                if selected is not None
                else [1 / len(labels)] * len(labels)
            )
            answers[question.id] = answer_from_probabilities(question, values)
        response = {
            "model": self.name,
            "answers": answers,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        self.responses.append(copy.deepcopy(response))
        return response


def eighty_control_fixture():
    buttons = "".join(
        f'<button type="button" data-item="{item}">Activate item {item}</button>'
        for item in range(1, 81)
    )
    source = f"""<!doctype html><html><head><title>Visible control grid</title>
      <style>
        body{{margin:12px;font:14px sans-serif}}h1{{font-size:20px;margin:0 0 10px}}
        #confirmation{{min-height:24px;margin:0 0 12px}}
        #grid{{display:grid;grid-template-columns:repeat(10,minmax(0,1fr));gap:8px}}
        button{{height:56px;font-size:12px}}
      </style></head><body><h1>Visible control grid</h1>
      <p id="confirmation" role="status">Activation not yet confirmed</p>
      <section id="grid">{buttons}</section>
      <script>
        for(const button of document.querySelectorAll('button')){{
          button.onclick=()=>{{
            document.getElementById('confirmation').textContent=
              'Activation confirmed: item '+button.dataset.item;
            button.textContent='Activated item '+button.dataset.item;
            button.disabled=true;
          }};
        }}
      </script></body></html>"""
    return HARNESS.FixtureCase(
        case_id="diagnostic-visible-80-controls",
        task_id="diagnostic_80_controls",
        goal="Activate item 80. Stop when its activation is visibly confirmed.",
        html=source,
        expected={},
        needs_text=False,
    )


@pytest.mark.integration
def test_eighty_visible_targets_reach_upstream_guarded_executor_without_pruning(
    hierarchical_browser_environment, monkeypatch, tmp_path
):
    from fastapi.testclient import TestClient

    from s1.api import create_app

    upstream, chromium = hierarchical_browser_environment
    case = eighty_control_fixture()
    backend = ObservedControlDiagnostic()
    app = create_app(
        backend,
        jev_compatible=True,
        jev_hierarchical=True,
        model_identity="diagnostic-browser-policy",
    )
    transport = HARNESS.PlaywrightCDP(chromium)
    evidence = {"scope": "scripted diagnostic; no model quality measurement", "wire": []}
    with TestClient(app) as client, HARNESS.fixture_server([case]) as base:
        with HARNESS.load_upstream(upstream, transport) as (agent_module, model_module):
            agent = agent_module.Agent(base + case.path, case.goal)
            try:
                page = agent.state["page"]
                clicks = [action for action in page["actions"] if action["kind"] == "click"]
                assert (page["w"], page["h"]) == (1120, 780)
                assert page["omitted_actions"] == 0
                assert len(clicks) == 80
                assert [action["label"] for action in clicks] == [
                    "Activate item " + str(item) for item in range(1, 81)
                ]
                assert all(
                    0 <= action["rect"]["x"]
                    and 0 <= action["rect"]["y"]
                    and action["rect"]["x"] + action["rect"]["w"] <= page["w"]
                    and action["rect"]["y"] + action["rect"]["h"] <= page["h"]
                    for action in clicks
                )
                action80 = clicks[-1]
                assert action80["node"] == 80

                def local_http_post(url, **kwargs):
                    # Keep upstream post_json retries, choose, validation and
                    # execution intact; only its HTTP destination is local.
                    assert url == "http://testserver/v1/systemone"
                    response = client.post("/v1/systemone", **kwargs)
                    evidence["wire"].append(
                        {
                            "request": copy.deepcopy(kwargs["json"]),
                            "status": response.status_code,
                            "response": response.json(),
                        }
                    )
                    return response

                with monkeypatch.context() as patch:
                    patch.setattr(model_module.CLIENT, "post", local_http_post)
                    with HARNESS.model_adapter(
                        agent_module,
                        model_module,
                        endpoint="http://testserver/v1/systemone",
                        model="diagnostic-browser-policy",
                        api_key=None,
                    ) as (records, _secrets):
                        predicted = agent.command("predict")
                        decision = predicted["decision"]
                        assert decision["operation"] == "CLICK"
                        assert decision["target"] == "80"
                        assert decision["choice"] == action80["id"]
                        assert len(decision["target_probabilities"]) == 80
                        assert set(decision["probabilities"]) == {a["id"] for a in clicks}

                        first = evidence["wire"][0]
                        assert first["status"] == 200
                        raw = first["request"]
                        assert len(raw["state"]["elements"]) == 80
                        labels = list(raw["questions"]["click_target"]["criteria"])
                        assert labels == [str(item) for item in range(1, 81)]
                        answer = first["response"]["answers"]["click_target"]
                        assert list(answer["probabilities"]) == labels
                        assert all(value > 0 for value in answer["probabilities"].values())
                        assert sum(answer["probabilities"].values()) == pytest.approx(1)
                        assert answer["choice"] == "80"
                        hierarchy = first["response"]["hierarchy"]
                        assert hierarchy["status"] == "completed"
                        assert hierarchy["backend_predict_calls"] == 6
                        assert [stage["scope"] for stage in hierarchy["stages"]] == [
                            "ordinary",
                            "group",
                            "conditional",
                            "conditional",
                            "conditional",
                            "conditional",
                        ]
                        assert all(stage["status"] == "completed" for stage in hierarchy["stages"])
                        assert len(backend.calls) == 6
                        assert all(request.state == raw["state"] for request in backend.calls)
                        root_question = backend.calls[1].questions[0]
                        members = [
                            member
                            for description in root_question.criteria.values()
                            for member in json.loads(description)["members"]
                        ]
                        assert [member["target_id"] for member in members] == labels
                        for member in members:
                            assert (
                                member["criteria"]
                                == raw["questions"]["click_target"]["criteria"][member["target_id"]]
                            )
                        leaf_labels = []
                        for request in backend.calls[2:]:
                            question = request.questions[0]
                            assert (
                                json.loads(question.instructions)["hierarchy"]["stage"]
                                == "conditional"
                            )
                            assert len(question.labels()) == 20
                            leaf_labels.extend(question.labels())
                            for label, description in question.criteria.items():
                                assert (
                                    json.loads(description)
                                    == raw["questions"]["click_target"]["criteria"][label]
                                )
                        assert leaf_labels == labels

                        acted = agent.command("act", {"fingerprint": page["fingerprint"]})
                        assert acted["history"][-1]["action"] == "Activate item 80"
                        assert acted["history"][-1]["target"] == "80"
                        observed = agent.browser.evaluate("""(() => ({
                          text:document.getElementById('confirmation').textContent,
                          disabled:[...document.querySelectorAll('button:disabled')].map(e=>e.dataset.item),
                          target:document.querySelector('[data-item="80"]').textContent
                        }))()""")
                        assert observed == {
                            "text": "Activation confirmed: item 80",
                            "disabled": ["80"],
                            "target": "Activated item 80",
                        }
                        with pytest.raises(agent_module.StalePage):
                            agent.browser.act(action80, page)
                        finished = agent.command("tick")
                        assert finished["status"] == "done"
                        assert len(finished["history"]) == 1
                        assert len(evidence["wire"]) == len(records) == 2
                        assert records[0]["transport_attempts"][0]["http_status"] == 200
                        assert records[1]["response"]["answers"]["operation"]["choice"] == "DONE"
                        evidence["adapter_records"] = copy.deepcopy(records)
            finally:
                evidence["backend_requests"] = [request.model_dump() for request in backend.calls]
                evidence["backend_responses"] = backend.responses
                (tmp_path / "hierarchy-browser-diagnostic.json").write_text(
                    json.dumps(evidence, ensure_ascii=False, indent=2) + "\n"
                )
                agent.close()
