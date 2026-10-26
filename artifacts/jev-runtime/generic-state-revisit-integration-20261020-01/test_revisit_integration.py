"""In-memory generated modules with fake Browser/HTTP only; no live actors."""

import builtins
import copy
import hashlib
import importlib.util
import json
import sys
from collections import deque
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

BASE = Path(__file__).parent
ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
PARENT = (
    ROOT
    / "artifacts/jev-runtime/public-native-hover-settled-v8-20261019-01/profile-01/runtime/jev_ultrafast"
)
CORE = (
    ROOT / "artifacts/jev-runtime/generic-state-revisit-progress-20261020-01/observation_revisit.py"
)
SPEC = importlib.util.spec_from_file_location(
    "revisit_integration_subject", BASE / "revisit_integration.py"
)
T = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(T)


@pytest.fixture(autouse=True)
def no_framework_or_browser_imports(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in {
            "torch",
            "transformers",
            "modal",
            "ctypes",
            "playwright",
            "selenium",
        }:
            raise AssertionError("Forbidden actor/framework import: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def inputs():
    return {
        "agent.py": (PARENT / "agent.py").read_bytes(),
        "model.py": (PARENT / "model.py").read_bytes(),
        "observation_revisit.py": CORE.read_bytes(),
    }


@pytest.fixture
def built():
    return T.build_runtime_sources(inputs())


def module(monkeypatch, name, blob=None, **attributes):
    value = ModuleType(name)
    value.__package__ = name.rpartition(".")[0]
    value.__file__ = "/in-memory-fake-runtime/" + name.replace(".", "/") + ".py"
    value.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, value)
    if blob is not None:
        exec(compile(blob, value.__file__, "exec"), value.__dict__)
    return value


def page(text="A", *, document=1000, scroll=0):
    return {
        "observation_profile": {"id": "composed-visible-hover-entries-settled-v8"},
        "url": "https://example.test/page",
        "title": "Example",
        "text": text,
        "w": 1200,
        "h": 800,
        "scroll": {"y": scroll, "height": 3000},
        "actions": [
            {
                "id": "e1",
                "kind": "click",
                "node": 1,
                "label": "Open",
                "role": "button",
                "rect": [0, 0, 20, 20],
            },
            {"id": "wait", "kind": "wait", "label": "Wait for the page to update"},
        ],
        "page_key": [
            document,
            "https://example.test/page",
            0,
            scroll,
            1200,
            800,
            [[2, "query", False, -1, False, False]],
        ],
        "guards": {"e1": [1, "Open", "native guard"]},
        "marker": [1, "native marker"],
        "omitted_actions": 0,
        "fingerprint": str((text, document, scroll)),
    }


class StalePage(RuntimeError):
    pass


class FakeBrowser:
    def __init__(self, observations):
        self.observations = deque(observations)
        self.fresh_values = deque()
        self.observed = []
        self.acted = []
        self.closed = False
        self.act_error = None

    def observe(self, *, screenshot):
        self.observed.append(screenshot)
        value = self.observations.popleft()
        if isinstance(value, BaseException):
            raise value
        return value

    def fresh(self, _page):
        return self.fresh_values.popleft() if self.fresh_values else True

    def act(self, action, original_page, *, text=None):
        self.acted.append((action, original_page, text))
        if self.act_error is not None:
            error, self.act_error = self.act_error, None
            raise error

    def close(self):
        self.closed = True


def agent_fixture(monkeypatch, outputs, observations):
    package = "revisit_fake_agent"
    module(monkeypatch, package, __path__=[])
    browser = FakeBrowser(observations)
    module(monkeypatch, package + ".browser", Browser=lambda _url: browser, StalePage=StalePage)
    module(monkeypatch, package + ".questions", MAX_STEPS=16)
    module(monkeypatch, package + ".observation_revisit", outputs["observation_revisit.py"])
    calls = []

    def choose(state, goal, history, *, observation_revisit):
        calls.append(
            {
                "page": copy.deepcopy(state),
                "goal": goal,
                "history": copy.deepcopy(history),
                "metadata": copy.deepcopy(observation_revisit),
            }
        )
        return {
            "choice": "e1",
            "operation": "CLICK",
            "target": "1",
            "confidence": 1.0,
            "probabilities": {"e1": 1.0},
            "latency_ms": 0,
            "model": "fake",
            "usage": {},
        }

    module(
        monkeypatch,
        package + ".model",
        choose=choose,
        action_space=lambda _actions: ([], {}, {}),
        field_context=lambda *_args: None,
        field_text=lambda *_args: pytest.fail("No field writer expected"),
    )
    generated = module(monkeypatch, package + ".agent", outputs["agent.py"])
    agent = generated.Agent("https://example.test/page", "Keep the complete original goal")
    return agent, browser, calls


def execute_action(agent):
    agent.command("predict")
    return agent.command("act", {"fingerprint": agent.state["page"]["fingerprint"]})


def test_fixed_parent_inverse_AST_and_adjacent_core_contract(built):
    outputs, report = built
    assert outputs["observation_revisit.py"] == CORE.read_bytes()
    assert set(outputs) == {"agent.py", "model.py", "observation_revisit.py"}
    assert report["core_bytes_unchanged"] is True
    assert report["runtime_imported"] is False and report["materialized_profile"] is False
    assert report["capacity_or_browser_qualified"] is False
    for row in report["transformations"]:
        assert row["byte_inverse_exact"] and row["whole_inverse_AST_equal"]
    assert report["transformations"][0]["unchanged_definitions"] == [
        "Agent.__enter__",
        "Agent.__exit__",
        "Agent.close",
        "Agent.run",
    ]
    assert set(report["transformations"][1]["unchanged_definitions"]) == {
        "action_space",
        "field_context",
        "field_text",
        "post_json",
        "validate_choice",
    }


@pytest.mark.parametrize("name", list(T.INPUT_PINS))
def test_changed_parent_or_core_is_refused_before_transformation(name):
    values = inputs()
    values[name] += b"\n"
    with pytest.raises(ValueError, match="Exact parent bytes"):
        T.build_runtime_sources(values)


def test_initial_and_repeated_predict_keep_single_accepted_observation(monkeypatch, built):
    p = page()
    original = copy.deepcopy(p)
    agent, browser, calls = agent_fixture(monkeypatch, built[0], [p])
    for _ in range(3):
        agent.command("predict")
    assert len(browser.observed) == 1
    assert agent._observation_sequence == 0 and agent._revisit_tracker.retained_count == 1
    assert [c["metadata"]["observation_sequence"] for c in calls] == [0, 0, 0]
    assert all(c["metadata"]["observation_notes"] == [] for c in calls)
    assert agent.state["page"] is p and p == original
    assert calls[-1]["goal"] == "Keep the complete original goal"


def test_post_action_A_B_A_and_snapshot_metadata_are_separate_owned_copies(monkeypatch, built):
    observations = [page("A"), page("B"), page("A")]
    before = copy.deepcopy(observations)
    agent, browser, calls = agent_fixture(monkeypatch, built[0], observations)
    execute_action(agent)
    result = execute_action(agent)
    assert len(browser.observed) == 3 and agent._observation_sequence == 2
    assert result["observation_revisit"]["exact"]["previous_observation_sequences"] == [0]
    result["observation_revisit"]["observation_notes"].clear()
    result["observation_revisit"]["exact"]["previous_observation_sequences"].append(99)
    agent.command("predict")
    assert calls[-1]["metadata"]["exact"]["previous_observation_sequences"] == [0]
    assert calls[-1]["metadata"]["observation_notes"]
    assert observations == before
    assert agent.state["page"] is observations[2]


def test_predict_freshness_refresh_is_a_distinct_observation(monkeypatch, built):
    observations = [page("A"), page("A", document=2000)]
    before = copy.deepcopy(observations)
    agent, browser, calls = agent_fixture(monkeypatch, built[0], observations)
    browser.fresh_values.append(False)
    agent.command("predict")
    metadata = calls[-1]["metadata"]
    assert len(browser.observed) == 2 and metadata["observation_sequence"] == 1
    assert metadata["exact"]["previous_observation_sequences"] == []
    assert metadata["projection"]["previous_observation_sequences"] == [0]
    assert "not complete state or action identity" in metadata["projection"]["scope"]
    assert observations == before


def test_tick_stale_refresh_uses_wrapper_without_recording_unexecuted_action(monkeypatch, built):
    observations = [page("A"), page("B")]
    agent, browser, _calls = agent_fixture(monkeypatch, built[0], observations)
    browser.act_error = StalePage("fake document changed before input")
    result = agent.command("tick")
    assert result["status"] == "ready" and result["decision"] is None
    assert result["history"] == []
    assert result["observation_revisit"]["observation_sequence"] == 1
    assert len(browser.observed) == 2 and agent._revisit_tracker.retained_count == 2


@pytest.mark.parametrize("failure", ["observe", "rejected-observation"])
def test_failed_observation_does_not_increment_or_replace_hint(monkeypatch, built, failure):
    invalid = page("not accepted")
    del invalid["guards"]
    failed = RuntimeError("fake observe failed") if failure == "observe" else invalid
    p = page("A")
    agent, browser, _calls = agent_fixture(monkeypatch, built[0], [p, failed, page("B")])
    original_metadata = copy.deepcopy(agent._observation_revisit)
    browser.fresh_values.append(False)
    with pytest.raises((ValueError, RuntimeError)):
        agent.command("predict")
    assert agent._observation_sequence == 0 and agent._revisit_tracker.retained_count == 1
    assert agent._observation_revisit == original_metadata and agent.state["page"] is p
    browser.fresh_values.append(False)
    agent.command("predict")
    assert agent._observation_sequence == 1 and agent._revisit_tracker.retained_count == 2


def test_initial_observation_failure_still_closes_owned_fake_browser(monkeypatch, built):
    closed = []
    original = FakeBrowser.close

    def close(browser):
        closed.append(True)
        original(browser)

    monkeypatch.setattr(FakeBrowser, "close", close)
    with pytest.raises(RuntimeError, match="initial fake observe"):
        agent_fixture(monkeypatch, built[0], [RuntimeError("initial fake observe")])
    assert closed == [True]


def test_failed_post_action_observation_retains_executed_history_without_new_sequence(
    monkeypatch, built
):
    p = page()
    agent, browser, _calls = agent_fixture(
        monkeypatch, built[0], [p, RuntimeError("fake post-action observe"), page("B")]
    )
    agent.command("predict")
    with pytest.raises(RuntimeError, match="fake post-action observe"):
        agent.command("act", {"fingerprint": p["fingerprint"]})
    assert len(agent.state["history"]) == 1 and len(browser.acted) == 1
    assert agent.state["history"][0]["page_changed"] is None
    assert agent._observation_sequence == 0 and agent.state["page"] is p
    browser.fresh_values.append(False)
    agent.command("predict")
    assert agent._observation_sequence == 1


def test_trackers_are_per_agent_and_do_not_carry_task_history(monkeypatch, built):
    first, _browser, _calls = agent_fixture(monkeypatch, built[0], [page(), page()])
    execute_action(first)
    assert first._observation_revisit["observation_notes"]
    second, _browser, calls = agent_fixture(monkeypatch, built[0], [page()])
    second.command("predict")
    assert second._revisit_tracker is not first._revisit_tracker
    assert second._observation_sequence == 0
    assert calls[-1]["metadata"]["observation_notes"] == []


def model_fixture(monkeypatch, blob, core_blob):
    package = "revisit_fake_policy"
    module(monkeypatch, package, __path__=[])
    module(monkeypatch, "httpx", Client=lambda **_kwargs: SimpleNamespace(), HTTPError=RuntimeError)
    module(monkeypatch, package + ".questions", (PARENT / "questions.py").read_bytes())
    core = module(monkeypatch, package + ".observation_revisit", core_blob)
    model = module(monkeypatch, package + ".model", blob)
    bodies = []

    def post(_url, _key, body):
        bodies.append(body)
        answers = {}
        for name, question in body["questions"].items():
            labels = list(question["criteria"])
            answers[name] = {
                "choice": labels[0],
                "probabilities": {label: 1 / len(labels) for label in labels},
                "confidence": 1 / len(labels),
            }
        return {"answers": answers, "model": "fake-policy", "usage": {}}

    model.post_json = post
    monkeypatch.setenv("TYPESAFE_API_KEY", "fake-never-sent")
    return model, core, bodies


def test_no_notes_new_request_is_byte_exact_original_input(monkeypatch, built):
    p, goal = page(), "Every word of the original goal"
    history = [
        {"action": str(i), "kind": "click", "text": "query", "page_changed": True, "unused": i}
        for i in range(14)
    ]
    original, _core, original_bodies = model_fixture(
        monkeypatch, inputs()["model.py"], built[0]["observation_revisit.py"]
    )
    original.choose(p, goal, history)
    new, core, new_bodies = model_fixture(
        monkeypatch, built[0]["model.py"], built[0]["observation_revisit.py"]
    )
    metadata = core.ObservationRevisitTracker(include_projection=True).observe(
        p, observation_sequence=0
    )
    new.choose(p, goal, history, observation_revisit=metadata)
    assert json.dumps(new_bodies[0], ensure_ascii=False, separators=(",", ":")) == json.dumps(
        original_bodies[0], ensure_ascii=False, separators=(",", ":")
    )
    assert "observation_revisit" not in new_bodies[0]["state"]
    new.choose(p, goal, history)
    assert new_bodies[1] == new_bodies[0]


@pytest.mark.parametrize("cross_document", [False, True])
def test_only_matched_scope_notes_enter_state_and_inputs_are_not_mutated(
    monkeypatch, built, cross_document
):
    model, core, bodies = model_fixture(
        monkeypatch, built[0]["model.py"], built[0]["observation_revisit.py"]
    )
    tracker = core.ObservationRevisitTracker(include_projection=True)
    tracker.observe(page(), observation_sequence=0)
    p = page(document=2000 if cross_document else 1000)
    metadata = tracker.observe(p, observation_sequence=1)
    history = [{"action": "unchanged", "kind": "wait", "text": None, "page_changed": False}]
    before = copy.deepcopy((p, metadata, history))
    model.choose(p, "unchanged goal", history, observation_revisit=metadata)
    body = bodies[-1]
    assert list(body) == ["model", "state", "questions"]
    hint = body["state"]["observation_revisit"]
    assert "sha256" not in json.dumps(hint)
    assert hint["observation_notes"] == metadata["observation_notes"]
    assert hint["projection"]["previous_observation_sequences"] == [0]
    assert "not complete state or action identity" in hint["projection"]["scope"]
    assert ("exact" not in hint) is cross_document
    if not cross_document:
        assert "screenshot" in hint["exact"]["scope"]
    assert (p, metadata, history) == before
    hint["observation_notes"].clear()
    hint["projection"]["previous_observation_sequences"].append(999)
    assert metadata == before[1]


def test_signal_preserves_full_questions_candidate_order_and_original_state(monkeypatch, built):
    p = page()
    p["actions"] = [
        {
            "id": f"e{i}",
            "kind": "click",
            "node": i,
            "label": f"Label {i}",
            "role": "button",
            "rect": [i, 0, 10, 10],
        }
        for i in range(1, 301)
    ] + [p["actions"][-1]]
    history = [
        {"action": str(i), "kind": "click", "text": "value", "page_changed": bool(i % 2)}
        for i in range(14)
    ]
    original, _core, old_bodies = model_fixture(
        monkeypatch, inputs()["model.py"], built[0]["observation_revisit.py"]
    )
    original.choose(p, "complete untrimmed goal", history)
    new, core, new_bodies = model_fixture(
        monkeypatch, built[0]["model.py"], built[0]["observation_revisit.py"]
    )
    tracker = core.ObservationRevisitTracker(include_projection=True)
    tracker.observe(p, observation_sequence=0)
    metadata = tracker.observe(p, observation_sequence=1)
    new.choose(p, "complete untrimmed goal", history, observation_revisit=metadata)
    actual = copy.deepcopy(new_bodies[-1])
    del actual["state"]["observation_revisit"]
    assert json.dumps(actual, ensure_ascii=False, separators=(",", ":")) == json.dumps(
        old_bodies[-1], ensure_ascii=False, separators=(",", ":")
    )
    assert len(actual["questions"]["click_target"]["criteria"]) == 300
    assert list(actual["questions"]["click_target"]["criteria"]) == [str(i) for i in range(1, 301)]
    assert len(actual["state"]["recent_actions"]) == 10


def test_input_pins_remain_current_after_all_in_memory_composition():
    for name, blob in inputs().items():
        sha, size = T.INPUT_PINS[name]
        assert len(blob) == size and hashlib.sha256(blob).hexdigest() == sha
