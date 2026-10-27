"""Synthetic real-runner callback tests; no benchmark tasks, browser or GPU."""

import copy
import math
import time
import types

import jev_browser_benchmark as runner
import jev_browser_done_review_owner as owner
import pytest


class StalePage(Exception):
    pass


class Browser:
    def __init__(self):
        self.fresh_values = []
        self.actions = []

    def fresh(self, page, action=None):
        return self.fresh_values.pop(0) if self.fresh_values else True

    def act(self, action, page, text=None):
        if not self.fresh(page):
            raise StalePage()
        self.actions.append(action["id"])
        return {"executed": action["id"]}


class Agent:
    def __init__(self, model):
        self.model = model
        self.state = {
            "browser": Browser(),
            "page": {
                "fingerprint": "fresh",
                "url": "https://example.test",
                "title": "Example",
                "text": "Visible text",
                "actions": [{"id": "item", "kind": "click", "label": "Continue"}],
            },
            "goal": "Complete the requested task",
            "history": [],
            "decisions": [],
            "decision": None,
            "status": "ready",
            "started_at": time.perf_counter(),
            "plan_index": 0,
        }

    def snapshot(self):
        return copy.deepcopy({k: v for k, v in self.state.items() if k != "browser"})

    def command(self, name, body=None):
        body = body or {}
        state = self.state
        if name == "tick":
            self.command("predict")
            return self.command("act", {"fingerprint": state["page"]["fingerprint"]})
        if name == "predict":
            if not state["browser"].fresh(state["page"]):
                raise StalePage()
            state["decision"] = self.model.choose(state["page"], state["goal"], state["history"])
            state["decisions"].append(copy.deepcopy(state["decision"]))
            state["status"] = "predicted"
        elif name == "act":
            decision, page = state["decision"], state["page"]
            if not decision or body.get("fingerprint") != page["fingerprint"]:
                raise ValueError("Observe and choose before acting")
            state["decision"] = None
            choice = decision["choice"]
            if choice in {"DONE", "BLOCKED"}:
                if not state["browser"].fresh(page):
                    state["status"] = "ready"
                    raise StalePage()
                state["status"] = "done" if choice == "DONE" else "blocked"
                state["plan_index"] = int(choice == "DONE")
            else:
                action = next(x for x in page["actions"] if x["id"] == choice)
                state["browser"].act(action, page)
                state["history"].append(
                    {
                        "action": action["label"],
                        "kind": action["kind"],
                        "text": None,
                        "page_changed": False,
                    }
                )
                state["status"] = "ready"
        return self.snapshot()


CHOOSE = """\
def choose(page, goal, history):
    elements, targets, _ = action_space(page['actions'])
    operations = {'CLICK': 'click', 'DONE': 'done', 'BLOCKED': 'blocked'}
    body = {'model': 'synthetic', 'state': {'page': {k: page[k] for k in ('url','title','text')}, 'elements': elements, 'recent_actions': [{k: h.get(k) for k in ('action','kind','text','page_changed')} for h in history[-10:]]}, 'questions': {'operation': {'type':'choice', 'criteria':operations, 'instructions':{'goal':goal}}, 'click_target': {'type':'choice', 'criteria': {'1': 'continue'}}}}
    result = post_json('https://api.typesafe.ai/v1/systemone', 'key', body)
    operation = validate_choice(result['answers']['operation'], operations)
    choice = targets['CLICK']['1']['id'] if operation['choice'] == 'CLICK' else operation['choice']
    return {'choice': choice, 'operation': operation['choice'], 'request': body, 'raw_answers': result['answers']}
"""


def valid_answer(choice, options):
    return {
        "choice": choice,
        "probabilities": {key: float(key == choice) for key in options},
        "confidence": 1.0,
    }


def validate_choice(answer, options):
    probs = answer["probabilities"]
    if (
        set(probs) != set(options)
        or answer["choice"] not in options
        or any(type(x) not in (int, float) or not math.isfinite(x) for x in probs.values())
        or abs(sum(probs.values()) - 1) > 0.01
        or probs[answer["choice"]] != max(probs.values())
    ):
        raise ValueError("invalid response")
    return answer


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(owner, "SOURCE_ONLY", False)
    sent = []
    options = {"review": "supported", "initial": "DONE", "failure": None}

    def network(_url, _key, body):
        sent.append(copy.deepcopy(body))
        if owner.REVIEW_ID in body["questions"]:
            if options["failure"]:
                raise RuntimeError("review unavailable")
            return {
                "model": "synthetic",
                "answers": {owner.REVIEW_ID: valid_answer(options["review"], owner.REVIEW_OPTIONS)},
            }
        criteria = body["questions"]["operation"]["criteria"]
        choice = options["initial"] if "DONE" in criteria else "CLICK"
        return {
            "model": "synthetic",
            "answers": {
                "operation": valid_answer(choice, criteria),
                "click_target": valid_answer("1", ["1"]),
            },
        }

    model = types.ModuleType("synthetic_model")
    model.post_json = network
    model.validate_choice = validate_choice
    model.action_space = lambda actions: (
        [{"index": "1", "label": actions[0]["label"]}],
        {"CLICK": {"1": actions[0]}},
        {},
    )
    exec(compile(CHOOSE, __file__, "exec"), model.__dict__)
    agent_module = types.SimpleNamespace(
        Agent=Agent,
        Browser=Browser,
        StalePage=StalePage,
        choose=model.choose,
        field_text=lambda *_: None,
        MAX_STEPS=60,
    )
    return types.SimpleNamespace(model=model, agent_module=agent_module, sent=sent, options=options)


def context(case):
    return runner.model_adapter(
        case.agent_module,
        case.model,
        endpoint="http://127.0.0.1:9000",
        model="synthetic",
        api_key="key",
    )


def test_disabled_is_exact_pass_through_even_when_source_held(setup, monkeypatch):
    monkeypatch.setattr(owner, "SOURCE_ONLY", True)
    original = runner.model_adapter
    with owner.attach_to_runner(runner, enabled=False):
        assert runner.model_adapter is original
        with context(setup):
            assert Agent.command.__name__ == "command"
            assert Agent(setup.model).command("tick")["status"] == "done"
    assert len(setup.sent) == 1


def test_supported_preserves_original_done_and_same_visible_context(setup):
    original = Agent.command
    events = []
    with owner.attach_to_runner(runner, enabled=True, event_sink=events.append):
        with context(setup) as (records, _):
            a = Agent(setup.model)
            assert a.command("tick")["status"] == "done"
            assert a.state["decision"] is None and a.state["browser"].actions == []
            assert len(records) == 2 and len(a.state["decisions"]) == 1
            review = setup.sent[1]
            assert review["state"] == setup.sent[0]["state"]
            assert review["questions"][owner.REVIEW_ID]["instructions"]["goal"] == a.state["goal"]
            assert set(review["questions"][owner.REVIEW_ID]["criteria"]) == {
                "supported",
                "unsupported",
                "unknown",
            }
    assert [event["status"] for event in events] == ["review_call", "done_supported"]
    assert events[0]["request_sha256"] == owner._body_sha256(setup.sent[1])
    assert events[-1] == {"status": "done_supported", "reviews": 1}
    assert Agent.command is original


@pytest.mark.parametrize("verdict", ["unsupported", "unknown"])
def test_unconfirmed_done_replans_only_offered_action_and_records_actual_wire(setup, verdict):
    setup.options["review"] = verdict
    with owner.attach_to_runner(runner, enabled=True):
        with context(setup) as (records, _):
            a = Agent(setup.model)
            assert a.command("tick")["status"] == "ready"
            assert a.state["browser"].actions == ["item"] and a.state["decision"] is None
            assert len(records) == 3 and len(a.state["decisions"]) == 2
            wire = setup.sent[-1]
            assert "DONE" not in wire["questions"]["operation"]["criteria"]
            assert wire["state"] == setup.sent[0]["state"]
            assert a.state["decisions"][-1]["request"] == wire
            assert a.state["decisions"][-1]["completion_review"]["verdict"] == verdict
            with pytest.raises(ValueError, match="Observe"):
                a.command("act", {"fingerprint": "fresh"})
            assert a.state["browser"].actions == ["item"]


@pytest.mark.parametrize("verdict", ["unsupported", "unknown"])
def test_opt_in_feedback_changes_only_replan_copy_and_is_hash_bound(setup, verdict):
    setup.options["review"] = verdict
    events = []
    with owner.attach_to_runner(
        runner, enabled=True, feedback_enabled=True, event_sink=events.append
    ):
        with context(setup) as (records, _):
            a = Agent(setup.model)
            assert a.command("tick")["status"] == "ready"
            original, review_body, replan = setup.sent
            assert original["state"] == review_body["state"]
            assert owner.FEEDBACK_KEY not in original["state"]
            assert replan["state"] == {
                **original["state"],
                owner.FEEDBACK_KEY: {
                    "verdict": verdict,
                    "guidance": owner.FEEDBACK_GUIDANCE[verdict],
                },
            }
            assert "DONE" in original["questions"]["operation"]["criteria"]
            assert "DONE" not in replan["questions"]["operation"]["criteria"]
            assert a.state["decisions"][-1]["request"] == replan
            assert a.state["decisions"][-1]["completion_review"]["feedback_enabled"] is True
            replan_event = next(event for event in events if event["status"] == "replan_call")
            assert replan_event["request_sha256"] == owner._body_sha256(replan)
            assert records[-1]["request"] == replan
            assert a.state["decision"] is None and a.state["browser"].actions == ["item"]


def test_feedback_requires_explicit_boolean_and_is_not_sent_for_supported(setup):
    with pytest.raises(ValueError, match="feedback opt-in"):
        with owner.attach_to_runner(runner, enabled=True, feedback_enabled="yes"):
            pass
    with owner.attach_to_runner(runner, enabled=True, feedback_enabled=True):
        with context(setup):
            a = Agent(setup.model)
            assert a.command("tick")["status"] == "done"
            assert len(setup.sent) == 2
            assert all(owner.FEEDBACK_KEY not in row["state"] for row in setup.sent)


@pytest.mark.parametrize(
    "verdict,expected_status,expected_actions",
    [("unsupported", "ready", ["item"]), ("unknown", "blocked", [])],
)
def test_feedback_can_inform_a_synthetic_policy_choice(
    setup, verdict, expected_status, expected_actions
):
    setup.options["review"] = verdict
    network = setup.model.post_json

    def choose_from_feedback(url, key, body):
        result = network(url, key, body)
        advisory = body["state"].get(owner.FEEDBACK_KEY)
        if advisory:
            operation = "CLICK" if advisory["verdict"] == "unsupported" else "BLOCKED"
            result["answers"]["operation"] = valid_answer(
                operation, body["questions"]["operation"]["criteria"]
            )
        return result

    setup.model.post_json = choose_from_feedback
    with owner.attach_to_runner(runner, enabled=True, feedback_enabled=True):
        with context(setup):
            agent = Agent(setup.model)
            assert agent.command("tick")["status"] == expected_status
            assert agent.state["browser"].actions == expected_actions


def test_feedback_text_is_bound_before_event_callbacks(setup, monkeypatch):
    setup.options["review"] = "unknown"
    expected = owner.FEEDBACK_GUIDANCE["unknown"]
    key = owner.FEEDBACK_KEY

    def event(_event):
        monkeypatch.setitem(owner.FEEDBACK_GUIDANCE, "unknown", "changed by callback")
        monkeypatch.setattr(owner, "FEEDBACK_KEY", "changed_by_callback")

    with owner.attach_to_runner(runner, enabled=True, feedback_enabled=True, event_sink=event):
        with context(setup):
            assert Agent(setup.model).command("tick")["status"] == "ready"
    assert setup.sent[-1]["state"][key]["guidance"] == expected
    assert "changed_by_callback" not in setup.sent[-1]["state"]


def test_feedback_rejects_a_key_altered_before_opt_in(setup, monkeypatch):
    monkeypatch.setattr(owner, "FEEDBACK_KEY", "changed_before_attach")
    with pytest.raises(ValueError, match="fixed feedback key"):
        with owner.attach_to_runner(runner, enabled=True, feedback_enabled=True):
            pass


def test_feedback_rejects_guidance_altered_before_opt_in(setup, monkeypatch):
    monkeypatch.setitem(owner.FEEDBACK_GUIDANCE, "unknown", "changed before attach")
    with pytest.raises(ValueError, match="fixed feedback guidance"):
        with owner.attach_to_runner(runner, enabled=True, feedback_enabled=True):
            pass


def test_feedback_off_does_not_depend_on_feedback_globals(setup, monkeypatch):
    monkeypatch.setattr(owner, "FEEDBACK_KEY", None)
    monkeypatch.setattr(owner, "FEEDBACK_GUIDANCE", None)
    setup.options["review"] = "unknown"
    with owner.attach_to_runner(runner, enabled=True, feedback_enabled=False):
        with context(setup):
            assert Agent(setup.model).command("tick")["status"] == "ready"
    assert setup.sent[-1]["state"] == setup.sent[0]["state"]


def test_non_done_is_unmodified_and_extra_calls_zero(setup):
    setup.options["initial"] = "CLICK"
    with owner.attach_to_runner(runner, enabled=True):
        with context(setup):
            a = Agent(setup.model)
            a.command("tick")
            assert a.state["browser"].actions == ["item"]
            assert len(setup.sent) == 1


def test_review_failure_consumes_decision_without_done_or_action(setup):
    setup.options["failure"] = True
    with owner.attach_to_runner(runner, enabled=True):
        with context(setup):
            a = Agent(setup.model)
            with pytest.raises(RuntimeError, match="review unavailable"):
                a.command("tick")
            assert a.state["decision"] is None and a.state["status"] == "ready"
            assert a.state["browser"].actions == [] and a.state["plan_index"] == 0


def test_stale_after_review_never_reuses_done(setup):
    with owner.attach_to_runner(runner, enabled=True):
        with context(setup):
            a = Agent(setup.model)
            a.state["browser"].fresh_values = [True, True, False]
            with pytest.raises(StalePage):
                a.command("tick")
            assert a.state["decision"] is None and a.state["status"] == "ready"
            assert a.state["browser"].actions == []


def test_review_limit_blocks_without_unreviewed_done(setup):
    setup.options["review"] = "unsupported"
    with owner.attach_to_runner(runner, enabled=True, max_reviews=1):
        with context(setup):
            a = Agent(setup.model)
            a.command("tick")
            assert a.command("tick")["status"] == "blocked"
            assert a.state["browser"].actions == ["item"] and a.state["decision"] is None
            assert len(setup.sent) == 4


def test_extra_policy_calls_respect_upstream_budget(setup):
    setup.agent_module.MAX_STEPS = 1
    setup.options["review"] = "unknown"
    with owner.attach_to_runner(runner, enabled=True):
        with context(setup):
            a = Agent(setup.model)
            with pytest.raises(ValueError, match="budget exhausted before replan"):
                a.command("tick")
            assert len(setup.sent) == 2
            assert a.state["decision"] is None and a.state["status"] == "ready"
            assert a.state["browser"].actions == []


@pytest.mark.parametrize("verdict", ["supported", "unsupported"])
def test_event_sink_failure_does_not_leave_replayable_decision(setup, verdict):
    setup.options["review"] = verdict

    def fail(_event):
        raise RuntimeError("event sink failed")

    with owner.attach_to_runner(runner, enabled=True, event_sink=fail):
        with context(setup):
            a = Agent(setup.model)
            with pytest.raises(RuntimeError, match="event sink failed"):
                a.command("tick")
            assert a.state["decision"] is None and a.state["status"] == "ready"
            assert a.state["browser"].actions == [] and a.state["plan_index"] == 0


def test_page_mutation_after_review_cannot_execute(setup):
    selected = {}

    def mutate(_event):
        selected["agent"].state["page"]["text"] = "changed"

    with owner.attach_to_runner(runner, enabled=True, event_sink=mutate):
        with context(setup):
            a = Agent(setup.model)
            selected["agent"] = a
            with pytest.raises(ValueError, match="visible state changed"):
                a.command("tick")
            assert a.state["decision"] is None and a.state["browser"].actions == []


def test_event_callback_cannot_force_done_or_restore_consumed_decision(setup):
    selected = {}

    def mutate(_event):
        agent = selected["agent"]
        agent.state["status"] = "done"
        agent.state["decision"] = {"choice": "DONE"}

    with owner.attach_to_runner(runner, enabled=True, event_sink=mutate):
        with context(setup):
            a = Agent(setup.model)
            selected["agent"] = a
            with pytest.raises(ValueError, match="visible state changed"):
                a.command("tick")
            assert a.state["status"] == "ready" and a.state["decision"] is None
            assert a.state["browser"].actions == []


def test_unexpected_browser_fresh_wrapper_fails_closed(setup):
    with owner.attach_to_runner(runner, enabled=True):
        with context(setup):
            a = Agent(setup.model)
            a.command("predict")
            a.state["browser"].fresh = lambda *_args, **_kwargs: True
            with pytest.raises(ValueError, match="original browser freshness"):
                a.command("act", {"fingerprint": "fresh"})
            assert a.state["status"] == "ready" and a.state["decision"] is None
            assert a.state["browser"].actions == []


@pytest.mark.parametrize("feedback_enabled", [False, True])
@pytest.mark.parametrize("callback", ["event", "policy"])
def test_callback_cannot_reenter_review_or_execute_and_guard_clears(
    setup, callback, feedback_enabled
):
    selected = {}
    original_network = setup.model.post_json

    def reenter():
        setup.options["initial"] = "CLICK"
        selected["agent"].command("tick")

    def network(url, key, body):
        if callback == "policy" and owner.REVIEW_ID in body["questions"]:
            reenter()
        return original_network(url, key, body)

    def event(_event):
        if callback == "event":
            reenter()

    setup.model.post_json = network
    with owner.attach_to_runner(
        runner, enabled=True, feedback_enabled=feedback_enabled, event_sink=event
    ):
        with context(setup):
            a = Agent(setup.model)
            selected["agent"] = a
            with pytest.raises(ValueError, match="reentrant command"):
                a.command("tick")
            assert a.state["decision"] is None and a.state["status"] == "ready"
            assert a.state["browser"].actions == []
            # An exception in either callback must release the guard.
            assert a.command("tick")["status"] == "ready"
            assert a.state["browser"].actions == ["item"]


def test_feedback_replan_event_page_mutation_cannot_execute(setup):
    setup.options["review"] = "unknown"
    selected = {}

    def event(row):
        if row["status"] == "replan_call":
            selected["agent"].state["page"]["text"] = "changed"

    with owner.attach_to_runner(runner, enabled=True, feedback_enabled=True, event_sink=event):
        with context(setup):
            agent = Agent(setup.model)
            selected["agent"] = agent
            with pytest.raises(ValueError, match="visible state changed"):
                agent.command("tick")
            assert agent.state["decision"] is None and agent.state["status"] == "ready"
            assert agent.state["browser"].actions == []


def test_progress_advisory_state_is_explicitly_unsupported(setup):
    with owner.attach_to_runner(runner, enabled=True):
        with context(setup):
            a = Agent(setup.model)
            a.command("predict")
            a.state["decision"]["request"]["state"]["observed_progress_advisory"] = {
                "scope": "bounded_visible_projection_not_full_dom_or_task_outcome"
            }
            with pytest.raises(ValueError, match="visible state differs"):
                a.command("act", {"fingerprint": "fresh"})
            assert a.state["decision"] is None and a.state["status"] == "ready"
            assert a.state["browser"].actions == [] and len(setup.sent) == 1
