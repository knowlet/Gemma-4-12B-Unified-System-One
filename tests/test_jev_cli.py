"""Serving the actual Jev wire request through the public CLI boundary."""

import sys
from types import SimpleNamespace

import pytest
from test_browser_policy import browser_model as browser_model

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from s1.cli import main


@pytest.mark.parametrize("compatible", [False, True])
def test_cli_serves_structured_jev_request_only_when_enabled(monkeypatch, compatible):
    payload = {
        "model": "browser-policy",
        "state": {"page": {"text": "Task awaiting confirmation"}},
        "questions": {
            "operation": {
                "type": "choice",
                "instructions": {"goal": "Confirm task", "rules": "Stop only once confirmed"},
                "criteria": {"CLICK": "Click a control", "DONE": "Every requirement is visible"},
            },
            "click_target": {
                "type": "choice",
                "instructions": {"goal": "Confirm task", "operation": "CLICK"},
                "criteria": {"1": {"element": "[1] Confirm", "current_value": ""}},
            },
        },
    }
    observed = []

    def run(application, *, host, port):
        assert host == "127.0.0.1"
        with TestClient(application) as client:
            response = client.post("/v1/systemone", json=payload)
            observed.append(response)
            assert client.post("/decide", json=payload).status_code == 422

    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=run))
    monkeypatch.delenv("S1_API_KEY", raising=False)
    arguments = ["serve", "--backend", "uniform"]
    if compatible:
        arguments += ["--jev-compatible", "--jev-model-identity", "browser-policy"]
    assert main(arguments) == 0
    assert observed[0].status_code == (200 if compatible else 422)
    if compatible:
        assert observed[0].json()["answers"]["click_target"]["choice"] == "1"
        assert observed[0].json()["model"] == "uniform"


@pytest.mark.parametrize("option", ["--jev-hierarchical", "--jev-batch-stages"])
def test_cli_rejects_missing_hierarchy_prerequisites_before_backend_loading(option):
    with pytest.raises(SystemExit) as error:
        main(["serve", "--backend", "uniform", option])
    assert error.value.code == 2


@pytest.mark.parametrize(
    "flags",
    [
        ["--jev-prompt-mode", "user"],
        ["--jev-decision-readout", "artifact"],
        ["--jev-prefix-cache"],
        ["--jev-compatible", "--backend", "uniform", "--jev-prompt-mode", "user"],
        ["--jev-compatible", "--backend", "laya", "--jev-decision-readout", "artifact"],
        ["--jev-compatible", "--backend", "http", "--jev-prefix-cache"],
        ["--jev-compatible", "--jev-decision-readout", "artifact"],
        ["--jev-compatible", "--jev-prefix-cache"],
        ["--jev-compatible", "--jev-prompt-mode", "reasoned", "--jev-prefix-cache"],
        ["--jev-compatible", "--jev-prompt-mode", "reasoned", "--jev-decision-readout", "artifact"],
        ["--jev-compatible", "--jev-prefix-suffix-batch-size", "2"],
    ],
)
def test_cli_rejects_browser_configurations_before_any_backend_or_weight_loading(
    monkeypatch, flags
):
    import s1.cli as cli

    monkeypatch.setattr(cli, "make_backend", lambda *_: pytest.fail("invalid flags loaded weights"))
    with pytest.raises(SystemExit) as error:
        main(["serve", *flags])
    assert error.value.code == 2


@pytest.mark.parametrize("mode", ["user", "grounded", "goal-first", "reasoned"])
def test_cli_explicit_browser_mode_wraps_only_loaded_gemma_model_and_closes_original(
    monkeypatch, mode
):
    import s1.browser_policy as browser_policy
    import s1.cli as cli

    loaded_model = SimpleNamespace(name="loaded-gemma-model")
    closed, constructors, served = [], [], []
    original = SimpleNamespace(
        name="original-gemma-sdk",
        model=loaded_model,
        close=lambda: closed.append("original"),
    )

    class BrowserWrapper:
        def __init__(self, supplied, **options):
            assert supplied is loaded_model
            constructors.append(options)
            self.name = "explicit-browser-policy"
            self.prompt_variant = f"selected-{mode}-variant"

    monkeypatch.setattr(cli, "make_backend", lambda args: original)
    monkeypatch.setattr(browser_policy, "BrowserDecisionBackend", BrowserWrapper)
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(run=lambda application, **options: served.append((application, options))),
    )
    assert main(["serve", "--backend", "gemma", "--jev-compatible", "--jev-prompt-mode", mode]) == 0
    assert constructors == [{"prompt_mode": mode, "prefix_cache": False}]
    assert served[0][1] == {"host": "127.0.0.1", "port": 8000}
    assert closed == ["original"]


@pytest.mark.parametrize("prefix_batch_size", [1, 2])
def test_cli_readout_load_uses_selected_prompt_variant_and_explicit_prefix_option(
    monkeypatch, tmp_path, prefix_batch_size
):
    import s1.browser_policy as browser_policy
    import s1.browser_readout as browser_readout
    import s1.cli as cli

    loaded_model = SimpleNamespace(name="loaded-gemma-model")
    closed, constructors, loads = [], [], []
    original = SimpleNamespace(
        name="original-gemma-sdk", model=loaded_model, close=lambda: closed.append("original")
    )
    readout = object()
    path = tmp_path / "readout"

    class BrowserWrapper:
        def __init__(self, supplied, **options):
            assert supplied is loaded_model
            constructors.append(options)
            self.name = "explicit-browser-policy"
            self.prompt_variant = "grounded_user_question_v1"

    def load(supplied_path, supplied_model, variant):
        loads.append((supplied_path, supplied_model, variant))
        return readout

    monkeypatch.setattr(cli, "make_backend", lambda args: original)
    monkeypatch.setattr(browser_policy, "BrowserDecisionBackend", BrowserWrapper)
    monkeypatch.setattr(browser_readout, "load_readout", load)
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=lambda *args, **kwargs: None))
    assert (
        main(
            [
                "serve",
                "--backend",
                "gemma",
                "--jev-compatible",
                "--jev-prompt-mode",
                "grounded",
                "--jev-decision-readout",
                str(path),
                "--jev-prefix-cache",
                "--jev-prefix-suffix-batch-size",
                str(prefix_batch_size),
            ]
        )
        == 0
    )
    assert loads == [(path, loaded_model, "grounded_user_question_v1")]
    options = {"prompt_mode": "grounded", "prefix_cache": True}
    if prefix_batch_size == 2:
        options["prefix_suffix_batch_size"] = 2
    assert constructors == [options, {**options, "decision_readout": readout}]
    assert closed == ["original"]


def test_cli_implicit_native_backend_is_not_wrapped_or_given_browser_defaults(monkeypatch):
    import s1.browser_policy as browser_policy
    import s1.cli as cli
    from s1.backends import UniformBackend

    backend = UniformBackend()
    closes = []
    backend.close = lambda: closes.append("original")
    monkeypatch.setattr(cli, "make_backend", lambda args: backend)
    monkeypatch.setattr(
        browser_policy,
        "BrowserDecisionBackend",
        lambda *args, **kwargs: pytest.fail("native was wrapped"),
    )
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=lambda *args, **kwargs: None))
    assert main(["serve", "--backend", "gemma", "--jev-compatible"]) == 0
    assert closes == ["original"]


def test_cli_readout_loading_failure_still_closes_original_gemma_backend(monkeypatch, tmp_path):
    import s1.browser_policy as browser_policy
    import s1.browser_readout as browser_readout
    import s1.cli as cli

    closes = []
    original = SimpleNamespace(
        name="original-gemma-sdk", model=object(), close=lambda: closes.append("original")
    )
    monkeypatch.setattr(cli, "make_backend", lambda args: original)
    monkeypatch.setattr(
        browser_policy,
        "BrowserDecisionBackend",
        lambda *args, **kwargs: SimpleNamespace(prompt_variant="grounded_user_question_v1"),
    )

    def rejected(*args, **kwargs):
        raise ValueError("readout immutable model pins differ")

    monkeypatch.setattr(browser_readout, "load_readout", rejected)
    monkeypatch.setitem(
        sys.modules,
        "uvicorn",
        SimpleNamespace(run=lambda *args, **kwargs: pytest.fail("bad readout served")),
    )
    with pytest.raises(SystemExit) as error:
        main(
            [
                "serve",
                "--jev-compatible",
                "--jev-prompt-mode",
                "grounded",
                "--jev-decision-readout",
                str(tmp_path),
            ]
        )
    assert error.value.code == 2
    assert closes == ["original"]


def test_cli_real_browser_wrapper_preserves_full_goal_and_rich_candidates(
    monkeypatch, browser_model
):
    import s1.cli as cli

    payload = {
        "model": "explicit-browser",
        "state": {"page": {"title": "Observed page", "text": "Search changes remain unapplied"}},
        "questions": {
            "operation": {
                "instructions": {
                    "goal": "Apply every requested filter; open 台北 record",
                    "rules": ["Check current status"],
                },
                "criteria": {
                    "CLICK": "Click an observed control",
                    "DONE": "Every goal requirement is visible",
                },
            },
            "click_target": {
                "instructions": {"goal": "Apply every requested filter; open 台北 record"},
                "criteria": {
                    "first": {"element": "[1] Apply", "current_value": "", "checked": False},
                    "second": {
                        "element": "[2] Open record",
                        "current_value": "台北",
                        "role": "link",
                    },
                },
            },
        },
    }
    closes = []
    original = SimpleNamespace(
        name="original-gemma-sdk", model=browser_model, close=lambda: closes.append("original")
    )
    monkeypatch.setattr(cli, "make_backend", lambda args: original)
    monkeypatch.delenv("S1_API_KEY", raising=False)

    def run(application, **options):
        assert options == {"host": "127.0.0.1", "port": 8000}
        with TestClient(application) as client:
            result = client.post("/v1/systemone", json=payload)
            assert result.status_code == 200
            assert result.json()["model"] == browser_model.name
            assert list(result.json()["answers"]["click_target"]["probabilities"]) == [
                "first",
                "second",
            ]

    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=run))
    assert (
        main(
            [
                "serve",
                "--jev-compatible",
                "--jev-model-identity",
                "explicit-browser",
                "--jev-prompt-mode",
                "user",
            ]
        )
        == 0
    )
    rendered = [messages[0]["content"][0]["text"] for messages in browser_model.processor.messages]
    assert len(rendered) == 2
    assert all("Apply every requested filter; open 台北 record" in text for text in rendered)
    assert all("Search changes remain unapplied" in text for text in rendered)
    assert "Every goal requirement is visible" in rendered[0]
    assert "[1] Apply" in rendered[1] and "[2] Open record" in rendered[1]
    assert '"checked":false' in rendered[1]
    assert '"current_value":"台北"' in rendered[1]
    assert closes == ["original"]
