"""Synthetic upstream-style callback bridge; no real browser, tasks or GPU."""

import copy
import hashlib
import inspect
import types
from pathlib import Path

import jev_browser_benchmark as runner
import jev_browser_progress_integration as integration
import jev_browser_progress_owner as owner
import pytest
from test_jev_browser_progress_integration import body, page


class FakeModel(types.ModuleType):
    pass


class FakeBrowser:
    def __init__(self):
        self.fresh_value = True
        self.result = {"executed": "e1"}
        self.failure = None

    def fresh(self, _page, action=None):
        return self.fresh_value

    def act(self, action, _page, text=None):
        if self.failure:
            raise self.failure
        return copy.deepcopy(self.result)

    def observe(self):
        return page()


class FakeAgent:
    def __init__(self, model):
        self.model = model
        self.state = {"browser": FakeBrowser(), "page": page(), "history": [], "decision": None}

    def command(self, name, body=None):
        state = self.state
        body = body or {}
        if name == "tick":
            self.command("predict")
            return self.command("act", {"fingerprint": state["page"]["fingerprint"]})
        if name == "predict":
            if not state["browser"].fresh(state["page"]):
                state["page"] = state["browser"].observe()
            state["decision"] = self.model.choose(state["page"], "generic goal", state["history"])
        elif name == "act":
            previous = state["page"]
            choice = state["decision"]["choice"]
            state["decision"] = None
            action = next(row for row in previous["actions"] if row["id"] == choice)
            state["browser"].act(action, previous, text=None)
            state["history"].append(
                {
                    "step": len(state["history"]) + 1,
                    "choice": choice,
                    "kind": action["kind"],
                    "action": action["label"],
                    "text": None,
                    "page_changed": None,
                    "url": previous["url"],
                }
            )
            state["page"] = state["browser"].observe()
            state["history"][-1].update(
                page_changed=state["page"]["fingerprint"] != previous["fingerprint"],
                url=state["page"]["url"],
            )
        return state


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setattr(owner, "SOURCE_ONLY", False)
    monkeypatch.setattr(integration, "SOURCE_ONLY", False)
    sent = []

    def network(_url, _key, request):
        sent.append(copy.deepcopy(request))
        return {"model": "synthetic", "choice": "e1"}

    model = FakeModel("synthetic_model")
    model.__file__ = __file__
    model.post_json = network
    model.make_body = body
    source = "def choose(page, goal, history):\n return post_json('https://api.typesafe.ai/v1/systemone', 'unused', make_body(page, history[-1] if history else None))\n"
    exec(compile(source, __file__, "exec"), model.__dict__)
    agent_module = types.SimpleNamespace(
        Agent=FakeAgent, Browser=FakeBrowser, choose=model.choose, field_text=lambda _context: None
    )
    agent = FakeAgent(model)

    def pins():
        objects = (
            FakeAgent.command,
            FakeBrowser.act,
            FakeBrowser.fresh,
            model.choose,
            model.post_json,
            integration.BrowserProgressIntegration,
            integration.ProgressLedger,
            integration._metadata_valid,
            owner.attach,
        )
        paths = {str(Path(inspect.getsourcefile(item)).absolute()) for item in objects}
        return {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths}

    return types.SimpleNamespace(
        agent=agent, agent_module=agent_module, model=model, sent=sent, pins=pins
    )


def runner_context(session):
    return runner.model_adapter(
        session.agent_module,
        session.model,
        endpoint="http://127.0.0.1:9000",
        model="synthetic",
        api_key="unused",
    )


def test_original_runner_callback_receives_one_scoped_advisory(session):
    agent, model = session.agent, session.model
    with runner_context(session) as (runner_records, _secrets):
        original_post = model.post_json
        with owner.attach(
            agent, session.agent_module, model, integration, pins=session.pins(), enabled=True
        ):
            agent.command("tick")
            agent.command("tick")
            agent.command("predict")
            assert integration.KEY not in session.sent[0]["state"]
            assert integration.KEY not in session.sent[1]["state"]
            assert session.sent[2]["state"][integration.KEY]["scope"] == integration.SCOPE
            assert session.sent[2]["questions"] == session.sent[1]["questions"]
            assert runner_records[-1]["request"]["state"][integration.KEY]["repeated_transition"]
        assert model.post_json is original_post
        assert "command" not in agent.__dict__
        assert "act" not in agent.state["browser"].__dict__


@pytest.mark.parametrize("failure", ["stale", "unconfirmed", "exception", "changed_document"])
def test_unverified_execution_or_freshness_does_not_reach_policy(session, failure):
    agent, browser = session.agent, session.agent.state["browser"]
    with runner_context(session):
        with owner.attach(
            agent,
            session.agent_module,
            session.model,
            integration,
            pins=session.pins(),
            enabled=True,
        ):
            agent.command("tick")
            if failure == "unconfirmed":
                browser.result = {"executed": "different"}
            elif failure == "exception":
                browser.failure = RuntimeError("synthetic failure")
            if failure == "exception":
                with pytest.raises(RuntimeError, match="synthetic failure"):
                    agent.command("tick")
                browser.failure = None
            else:
                agent.command("tick")
            if failure == "stale":
                browser.fresh_value = False
            elif failure == "changed_document":
                agent.state["page"]["page_key"][0] = 999
                agent.state["page"]["marker"][0] = 999
            agent.command("predict")
            assert integration.KEY not in session.sent[-1]["state"]


def test_disabled_bridge_is_noop_and_release_gate(session, monkeypatch):
    with owner.attach(
        session.agent, session.agent_module, session.model, integration, pins={}, enabled=False
    ):
        assert "command" not in session.agent.__dict__
    monkeypatch.setattr(owner, "SOURCE_ONLY", True)
    with pytest.raises(ValueError, match="independent review"):
        with owner.attach(
            session.agent, session.agent_module, session.model, integration, pins={}, enabled=True
        ):
            pass


def test_source_mutation_is_refused_before_install(session):
    with runner_context(session):
        pins = session.pins()
        pins[str(Path(owner.__file__).absolute())] = "0" * 64
        with pytest.raises(ValueError, match="source identity"):
            with owner.attach(
                session.agent,
                session.agent_module,
                session.model,
                integration,
                pins=pins,
                enabled=True,
            ):
                pass
        assert "command" not in session.agent.__dict__


def test_original_exception_preserved_and_every_callback_restored(session):
    with runner_context(session):
        prior = session.model.post_json
        error = RuntimeError("synthetic primary")
        with pytest.raises(RuntimeError) as caught:
            with owner.attach(
                session.agent,
                session.agent_module,
                session.model,
                integration,
                pins=session.pins(),
                enabled=True,
            ):
                raise error
        assert caught.value is error
        assert session.model.post_json is prior
        assert "command" not in session.agent.__dict__
        assert "act" not in session.agent.state["browser"].__dict__
        assert "fresh" not in session.agent.state["browser"].__dict__


def test_preexisting_instance_callback_is_not_replaced(session):
    with runner_context(session):
        browser = session.agent.state["browser"]
        browser.act = browser.act
        original = browser.act
        with pytest.raises(ValueError, match="unwrapped original"):
            with owner.attach(
                session.agent,
                session.agent_module,
                session.model,
                integration,
                pins=session.pins(),
                enabled=True,
            ):
                pass
        assert browser.act is original


@pytest.mark.parametrize("attribute", ["act", "fresh", "command", "post_json"])
@pytest.mark.parametrize("after_assignment", [False, True])
def test_partial_install_failure_rolls_back_owned_callbacks(
    session, monkeypatch, attribute, after_assignment
):
    with runner_context(session):
        agent, model = session.agent, session.model
        browser = agent.state["browser"]
        original_post = model.post_json
        objects = {"act": browser, "fresh": browser, "command": agent, "post_json": model}
        target = objects[attribute]
        cls = type(target)
        original_setattr = cls.__setattr__
        error = RuntimeError("synthetic setter failure")

        def failing_setattr(obj, name, value):
            if obj is target and name == attribute:
                if after_assignment:
                    original_setattr(obj, name, value)
                raise error
            return original_setattr(obj, name, value)

        monkeypatch.setattr(cls, "__setattr__", failing_setattr)
        with pytest.raises(RuntimeError) as caught:
            with owner.attach(
                agent, session.agent_module, model, integration, pins=session.pins(), enabled=True
            ):
                pytest.fail("partial callback installation must not yield")
        assert caught.value is error
        assert "act" not in browser.__dict__ and "fresh" not in browser.__dict__
        assert "command" not in agent.__dict__ and model.post_json is original_post
        monkeypatch.setattr(cls, "__setattr__", original_setattr)


def test_class_callback_change_is_detected_without_overwriting(session, monkeypatch):
    with runner_context(session):
        agent, model = session.agent, session.model

        def replacement(*_args, **_kwargs):
            return None

        with pytest.raises(RuntimeError, match="cleanup unconfirmed"):
            with owner.attach(
                agent, session.agent_module, model, integration, pins=session.pins(), enabled=True
            ):
                monkeypatch.setattr(FakeBrowser, "act", replacement)
        assert FakeBrowser.act is replacement
        assert "act" not in agent.state["browser"].__dict__
