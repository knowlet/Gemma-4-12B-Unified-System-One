"""Collector CPU contracts; fake transport is not genuine public collection."""

import copy
import hashlib
import importlib.util
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "public_native_collector", ROOT / "scripts/jev_browser_collect_public_training.py"
)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
from s1.browser_public_training_tasks import get_public_training_task, tasks  # noqa: E402

UPSTREAM = ROOT / "artifacts/jev-runtime/upstream"
COMMON = (
    "trusted_expected_ownership",
    "owned_context",
    "same_native_document",
    "viewport",
    "document_ready",
    "no_visible_access_blocker",
    "no_access_error_title",
)


def task():
    return get_public_training_task("git-restore-description")


def transport_for(t):
    return SimpleNamespace(
        targets={"target-1": object()},
        sessions={"session-1": object()},
        public_root_frames={"session-1": "root-1"},
        public_document_commits=[
            {
                "session": "session-1",
                "frame_id": "root-1",
                "source": "cdp_frame_commit",
                "setup_blank": False,
                "loader_id": "loader-0",
                "url": t["start_url"],
            }
        ],
        public_navigation_violations=[],
        public_document_requests=[],
        public_observer_errors=[],
        document_responses=[],
        calls=0,
        cdp=lambda *a, **k: None,
    )


def synthetic_page(url, width=58, scroll_y=0):
    actions = [
        {
            "id": f"click-{n}",
            "node": n,
            "kind": "click",
            "label": f"Link {n}",
            "role": "link",
            "value": "",
        }
        for n in range(1, width + 1)
    ]
    actions.extend(
        [
            {"id": "scroll_down", "kind": "scroll", "label": "Scroll down"},
            {"id": "scroll_up", "kind": "scroll", "label": "Scroll up"},
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ]
    )
    page = {
        "url": url,
        "title": "Reference",
        "text": "Visible public index and references",
        "actions": actions,
        "scroll": {"y": scroll_y, "height": 2000},
        "omitted_actions": 0,
        "page_key": [1000, url, 0, scroll_y, 1120, 780, []],
        "marker": ["stable"],
        "guards": {},
    }
    page["fingerprint"] = hashlib.sha256(
        json.dumps(
            {k: page[k] for k in ("url", "text", "actions", "scroll")}, sort_keys=True
        ).encode()
    ).hexdigest()
    return page


class FakeCompiler:
    def compile(self, payload):
        return {
            "wire_sha256": m.sha(m.encode(payload)),
            "prompt_tokens": 100,
            "labels": {k: list(v["criteria"]) for k, v in payload["questions"].items()},
        }


class FakeEvidence:
    def __init__(self, *, failed_check=None, HTML_error=False):
        self.failed_check = failed_check

    def observe_public_training(self, browser, t):
        return {
            "url": browser.url,
            "actual_scroll_y": browser.scroll_y,
            "actual_private_fact": "not in native payload",
            "target": {"heading": {"rect": {"y": 300}}},
        }

    def validate_public_training_observation(self, t, observed, *, expected_ownership):
        checks = {key: True for key in COMMON}
        if self.failed_check:
            checks[self.failed_check] = False
        return {
            "success": observed["url"].split("#")[0] == "https://git-scm.com" + t["target_path"]
            and all(checks.values()),
            "checks": checks,
            "failed_checks": [k for k, v in checks.items() if not v],
        }


@contextmanager
def fake_native_episode(tmp_path, kind="ordinary", *, pre_refresh=False, reject_mutation=False):
    """Real pinned Agent/choose with an explicit fake Browser only for CPU tests."""
    t = task()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    folder = bundle / "attempt"
    pages = m.Pages(bundle / "pages")
    ep = m.Episode(
        bundle,
        folder,
        t,
        kind,
        {"collection_id": "cpu-test"},
        pages,
        FakeCompiler(),
        FakeEvidence(),
    )
    tr = transport_for(t)
    ep.transport = tr
    with m.load_upstream(UPSTREAM, tr) as (agent_module, model_module):
        stale = agent_module.StalePage

        class Browser:
            target = "target-1"
            session = "session-1"

            def __init__(self, url):
                self.url = url
                self.counter = 0
                self.fresh_count = 0
                self.scroll_y = 0

            def observe(self, screenshot=False):
                page = synthetic_page(self.url, scroll_y=self.scroll_y)
                raw = {k: copy.deepcopy(v) for k, v in page.items() if k != "fingerprint"}
                ep.raw_sink(
                    {
                        "protocol_call": len(ep.raw_rows),
                        "session": self.session,
                        "response": {"result": {"value": raw}},
                    }
                )
                return page

            def fresh(self, page, action=None):
                self.fresh_count += 1
                if pre_refresh and self.fresh_count == 1:
                    return False
                return (
                    page["url"] == self.url
                    and page["page_key"][3] == self.scroll_y
                    and not (reject_mutation and action is not None)
                )

            def act(self, action, page, text=None):
                if not self.fresh(page, action):
                    raise stale("Actual fake mutation guard rejected")
                if action["kind"] == "scroll":
                    self.scroll_y += action.get("delta", 560)
                    return {"executed": action["id"]}
                self.scroll_y = 0
                self.url = "https://git-scm.com" + (
                    t["target_path"] if action["node"] == 1 else t["bootstrap_target_path"]
                )
                self.counter += 1
                tr.public_document_commits.append(
                    {
                        "session": self.session,
                        "frame_id": "root-1",
                        "source": "cdp_frame_commit",
                        "setup_blank": False,
                        "loader_id": "loader-" + str(self.counter),
                        "url": self.url,
                    }
                )
                return {"executed": action["id"]}

            def evaluate(self, expression):
                if expression == m.PASSIVE_STATE:
                    return {
                        "ready_state": "complete",
                        "time_origin": 1000,
                        "url": self.url,
                        "scroll_x": 0,
                        "scroll_y": self.scroll_y,
                        "width": 1120,
                        "height": 780,
                    }
                if expression == "document.documentElement.outerHTML":
                    return "<html>" + self.url + "</html>"
                return [
                    {
                        "node": n,
                        "href": "https://git-scm.com"
                        + (
                            t["target_path"]
                            if n == 1
                            else t["bootstrap_target_path"]
                            if n == 2
                            else "/docs/irrelevant-" + str(n)
                        ),
                        "label": f"Link {n}",
                    }
                    for n in range(1, 59)
                ]

            def close(self):
                pass

        agent_module.Browser = Browser
        ep.model = model_module
        old = model_module.post_json
        model_module.post_json = ep.post
        prior_env = {key: os.environ.get(key) for key in ("TYPESAFE_API_KEY", "TYPESAFE_MODEL")}
        os.environ.update(TYPESAFE_API_KEY="synthetic-cpu-only", TYPESAFE_MODEL=m.MODEL_ALIAS)
        try:
            ep.agent = agent_module.Agent(t["start_url"], t["goal"], screenshots=False)
            ep.install_guards()
            yield ep, agent_module, model_module
        finally:
            model_module.post_json = old
            for key, value in prior_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def test_real_native_agent_stops_first_true_without_forced_extra_states(tmp_path):
    with fake_native_episode(tmp_path) as (ep, _agent, _model):
        result = ep.execute()
        assert result["success"] and result["supervised_decisions"] == 2
        assert [r["selected"]["operation"] for r in ep.pending] == ["CLICK", "DONE"]
        assert len(ep.agent.state["history"]) == 1
        requests = [r for r in ep.journals["all-native-calls"].rows if r["phase"] == "request"]
        assert [r["native_policy_call_index"] for r in requests] == [0, 1]
        assert all(
            len(r["payload"]["questions"]["click_target"]["criteria"]) == 58 for r in requests
        )
        assert all(set(r["payload"]) == {"model", "state", "questions"} for r in requests)
        assert all("actual_private_fact" not in m.encode(r["payload"]).decode() for r in requests)


def test_recovery_bootstrap_history_genuine_but_not_positive_labels(tmp_path):
    with fake_native_episode(tmp_path, "recovery") as (ep, _agent, _model):
        result = ep.execute()
        assert result["bootstrap_reached"]
        assert result["bootstrap_decisions"] == 1 and result["supervised_decisions"] == 2
        assert [r["bootstrap"] for r in ep.pending] == [True, False, False]
        assert len(ep.pending[1]["payload"]["state"]["recent_actions"]) == 1
        assert all(
            q["instructions"]["goal"] == ep.task["goal"]
            for r in ep.pending
            for q in r["payload"]["questions"].values()
        )
        m.write(ep.folder / "public-request-ledger.json", {"fake_CPU_ledger": True})
        cases, wires = m.serialize_episode(ep, "episodes/recovery")
        assert len(cases) == len(wires) == 2
        assert [row["state_role"] for row in cases] == ["recovery_incomplete", "recovered_complete"]
        assert all(row["gold"] != {"operation": "CLICK", "click_target": "2"} for row in cases)
        for row in cases:
            for name, ref in row["proof_refs"].items():
                if name == "public_ledger":
                    path = ep.folder / "public-request-ledger.json"
                else:
                    path = ep.folder / Path(ref["artifact_path"]).name
                assert m.record(path)["sha256"] == ref["sha256"]
        assert len(cases[0]["request"]["questions"]["click_target"]["criteria"]) == 58


def test_native_internal_pre_predict_refresh_is_retained_not_rejected(tmp_path):
    with fake_native_episode(tmp_path, pre_refresh=True) as (ep, _agent, _model):
        assert ep.execute()["success"]
        assert any(r.get("accepted") is False for r in ep.journals["guarded-actions"].rows)
        assert all(r["accepted"] for r in ep.pending)
        assert any(r.get("postproof_deferred") for r in ep.journals["transitions"].rows)
        assert ep.raw.count >= 3


def test_nonterminal_completion_appends_at_next_actual_native_pre(tmp_path):
    with fake_native_episode(tmp_path) as (ep, _agent, _model):
        first = ep.tick()
        assert not first["accepted"] and "post" not in first
        staged = copy.deepcopy(ep.journals["transitions"].rows[-1])
        assert staged["postproof_deferred"] and staged["execution_accepted"]
        terminal = ep.tick()
        assert terminal["selected"]["operation"] == "DONE" and terminal["accepted"]
        assert ep.journals["transitions"].rows[staged["sequence"]] == staged
        completion = ep.journals["transitions"].rows[first["transition_sequence"]]
        assert completion["phase"] == "completed-at-next-native-pre" and completion["accepted"]
        assert completion["postcondition_native_request_sequence"] == terminal["call_sequence"]
        assert first["post"] == terminal["pre"] and ep.awaiting_post is None


def test_transient_scroll_diagnostic_defers_to_upstreams_own_refresh(tmp_path):
    with fake_native_episode(tmp_path) as (ep, _agent, _model):
        browser = ep.agent.browser
        observe = browser.observe
        drifted = False

        def drifting_observe(*args, **kwargs):
            nonlocal drifted
            page = observe(*args, **kwargs)
            if browser.counter == 1 and not drifted:
                drifted = True
                browser.scroll_y = 88
            return page

        browser.observe = drifting_observe
        validate = ep.evidence.validate_public_training_observation

        def strict_scroll(t, observed, *, expected_ownership):
            result = validate(t, observed, expected_ownership=expected_ownership)
            result["checks"]["same_native_document"] = (
                observed["actual_scroll_y"] == expected_ownership["native_page_key"][3]
            )
            result["success"] = result["success"] and result["checks"]["same_native_document"]
            return result

        ep.evidence.validate_public_training_observation = strict_scroll
        first = ep.tick()
        assert not first["immediate_diagnostic"]["verdict"]["checks"]["same_native_document"]
        raw_count = ep.raw.count
        terminal = ep.tick()
        assert terminal["accepted"] and first["accepted"]
        assert ep.raw.count == raw_count + 1  # only native predict's own refresh
        assert terminal["pre"]["binding"]["native_page_key"][3] == 88
        assert any(
            r.get("accepted") is False and r.get("native_method") == "fresh"
            for r in ep.journals["guarded-actions"].rows
        )
        samples = [r for r in ep.journals["passive-settling"].rows if r["phase"] == "sample"]
        assert len(samples) == 3 and all(r["observed"]["scroll_y"] == 88 for r in samples)


def test_native_post_action_navigation_read_failure_keeps_real_action_once(tmp_path):
    with fake_native_episode(tmp_path) as (ep, agent, _model):
        browser = ep.agent.browser
        observe = browser.observe
        failed = 0

        def navigating(*args, **kwargs):
            nonlocal failed
            if browser.counter == 1 and failed < 2:
                failed += 1
                raise agent.StalePage("Document is navigating")
            return observe(*args, **kwargs)

        browser.observe = navigating
        first = ep.tick()
        assert len(ep.agent.state["history"]) == 1 and not first["accepted"]
        assert first["execution_record"]["error"]["type"] == "StalePage"
        assert ep.tick()["accepted"] and first["accepted"]
        assert len(ep.agent.state["history"]) == 1 and browser.counter == 1


@pytest.mark.parametrize("transient", [True, False])
def test_immediate_diagnostic_navigation_error_is_retained_without_accepting_it(
    tmp_path, transient
):
    with fake_native_episode(tmp_path) as (ep, agent, _model):
        original = ep.proof
        observe = ep.evidence.observe_public_training

        def proof(phase, **kwargs):
            if phase != "after-tick-diagnostic":
                return original(phase, **kwargs)

            def failed(*args):
                raise (
                    agent.StalePage("Document changed during evaluation")
                    if transient
                    else ValueError("Observer programming failure")
                )

            ep.evidence.observe_public_training = failed
            try:
                return original(phase, **kwargs)
            finally:
                ep.evidence.observe_public_training = observe

        ep.proof = proof
        if transient:
            first = ep.tick()
            assert not first["accepted"] and first["immediate_diagnostic"]["observed"] is None
            assert (
                first["immediate_diagnostic"]["errors"]["observation_error"]["type"] == "StalePage"
            )
            assert ep.tick()["accepted"] and first["accepted"]
        else:
            with pytest.raises(ValueError, match="programming failure"):
                ep.tick()
            assert not ep.pending[-1]["accepted"]
        assert any(r.get("observation_error") for r in ep.journals["private-dom"].rows)


def test_diagnostic_navigation_error_cannot_hide_HTML_write_failure(tmp_path, monkeypatch):
    with fake_native_episode(tmp_path) as (ep, agent, _model):

        def navigating(*args):
            raise agent.StalePage("Document changed during evaluation")

        monkeypatch.setattr(ep.evidence, "observe_public_training", navigating)
        monkeypatch.setattr(
            ep.pages, "store", lambda html: (_ for _ in ()).throw(OSError("Evidence write failed"))
        )
        with pytest.raises(OSError, match="Evidence write failed"):
            ep.proof("diagnostic", strict=False)
        row = ep.journals["private-dom"].rows[-1]
        assert (
            row["observation_error"]["type"] == "StalePage"
            and row["HTML_error"]["type"] == "OSError"
        )


def test_DONE_still_requires_immediate_strict_document_readiness(tmp_path):
    with fake_native_episode(tmp_path) as (ep, _agent, _model):
        ep.tick()
        proof = ep.proof

        def unready_after_done(phase, **kwargs):
            if phase == "after-DONE-strict":
                ep.evidence.failed_check = "document_ready"
            return proof(phase, **kwargs)

        ep.proof = unready_after_done
        with pytest.raises(ValueError, match="binding failed"):
            ep.tick()
        assert ep.pending[-1]["selected"]["operation"] == "DONE"
        assert not ep.pending[-1]["accepted"]
        assert ep.journals["private-dom"].rows[-1]["strict"] is True


def test_recovery_boundary_resolves_actual_request_without_phase_rewrite(tmp_path):
    with fake_native_episode(tmp_path, "recovery") as (ep, _agent, _model):
        assert ep.execute()["success"]
        rows = ep.journals["all-native-calls"].rows
        requests = [r for r in rows if r["phase"] == "request"]
        resolved = [r for r in rows if r["phase"] == "decision-resolved"]
        assert [r["phase_at_entry"] for r in requests] == ["bootstrap", "bootstrap", "supervised"]
        assert [r["resolved_phase"] for r in resolved] == ["bootstrap", "supervised", "supervised"]
        boundary = next(r for r in rows if r["phase"] == "bootstrap-boundary")
        assert boundary["request_sequence"] == requests[1]["sequence"]
        assert boundary["private_precondition_sequence"] == ep.pending[1]["pre"]["sequence"]
        assert ep.resolved_counts() == {"bootstrap": 1, "supervised": 2}


def test_sixteenth_bootstrap_action_resolves_next_request_as_supervised(tmp_path, monkeypatch):
    with fake_native_episode(tmp_path, "recovery") as (ep, _agent, _model):
        teacher = m.scripted_teacher

        def route(*args, **kwargs):
            if kwargs.get("bootstrap") and ep.resolved_counts()["bootstrap"] < 16:
                return {"operation": "SCROLL_DOWN", "target": None}
            return teacher(*args, **kwargs)

        monkeypatch.setattr(m, "scripted_teacher", route)
        result = ep.execute()
        assert result["bootstrap_decisions"] == 16 and result["supervised_decisions"] == 2
        requests = [r for r in ep.journals["all-native-calls"].rows if r["phase"] == "request"]
        resolved = [
            r for r in ep.journals["all-native-calls"].rows if r["phase"] == "decision-resolved"
        ]
        assert len(requests) == len(resolved) == 18
        assert (
            requests[16]["phase_at_entry"] == "bootstrap"
            and resolved[16]["resolved_phase"] == "supervised"
        )
        assert ep.pending[15]["bootstrap"] and not ep.pending[16]["bootstrap"]


@pytest.mark.parametrize("kind", ["ordinary", "recovery"])
def test_sixteen_resolved_decisions_have_no_seventeenth_phantom(tmp_path, monkeypatch, kind):
    with fake_native_episode(tmp_path, kind) as (ep, _agent, _model):
        monkeypatch.setattr(
            m, "scripted_teacher", lambda *a, **k: {"operation": "SCROLL_DOWN", "target": None}
        )
        with pytest.raises(ValueError, match="exceeded"):
            ep.execute()
        requests = [r for r in ep.journals["all-native-calls"].rows if r["phase"] == "request"]
        assert len(requests) == 16 and len(ep.agent.state["history"]) == 16
        assert ep.resolved_counts() == {
            "bootstrap": 16 if kind == "recovery" else 0,
            "supervised": 16 if kind == "ordinary" else 0,
        }
        assert not ep.pending[-1]["accepted"] and ep.awaiting_post is ep.pending[-1]


@pytest.mark.parametrize("timeout", [False, True])
def test_passive_settle_is_bounded_literal_read_only_and_keeps_every_sample(tmp_path, timeout):
    now = [0.0]
    calls = []
    journal = m.Journal(tmp_path / "passive.jsonl")

    def evaluate(expression):
        calls.append(expression)
        return {
            "ready_state": "loading" if timeout or len(calls) == 1 else "complete",
            "time_origin": 123,
            "url": "https://example.invalid/reference",
            "scroll_x": 0,
            "scroll_y": 88,
            "width": 1120,
            "height": 780,
        }

    browser = SimpleNamespace(evaluate=evaluate)

    def run():
        m.passive_settle(
            browser, journal, clock=lambda: now[0], sleep=lambda n: now.__setitem__(0, now[0] + n)
        )

    if timeout:
        with pytest.raises(ValueError, match="settling exceeded2s"):
            run()
        assert journal.rows[-1]["phase"] == "timeout" and now[0] == 2.0
    else:
        run()
        assert journal.rows[-1]["phase"] == "settled" and len(calls) == 4
        assert now[0] == pytest.approx(0.15)
    assert calls and set(calls) == {m.PASSIVE_STATE}
    assert len([r for r in journal.rows if r["phase"] == "sample"]) == len(calls)


def test_passive_late_third_stable_read_cannot_pass_declared_deadline(tmp_path):
    now = [0.0]
    count = [0]
    journal = m.Journal(tmp_path / "late.jsonl")

    def evaluate(expression):
        assert expression == m.PASSIVE_STATE
        count[0] += 1
        if count[0] == 3:
            now[0] = 2.05
        return {
            "ready_state": "complete",
            "time_origin": 123,
            "url": "https://example.invalid/reference",
            "scroll_x": 0,
            "scroll_y": 0,
            "width": 1120,
            "height": 780,
        }

    with pytest.raises(ValueError, match="settling exceeded2s"):
        m.passive_settle(
            SimpleNamespace(evaluate=evaluate),
            journal,
            clock=lambda: now[0],
            sleep=lambda n: now.__setitem__(0, now[0] + n),
        )
    assert count[0] == 3 and journal.rows[-1]["phase"] == "timeout"
    assert journal.rows[-1]["elapsed_ms"] == pytest.approx(2050)
    assert not any(row["phase"] == "settled" for row in journal.rows)


def test_rejected_execution_guard_quarantines_without_mutation_retry(tmp_path):
    with fake_native_episode(tmp_path, reject_mutation=True) as (ep, _agent, _model):
        with pytest.raises(ValueError, match="guard rejected"):
            ep.execute()
        assert len(ep.pending) == 1 and not ep.pending[0]["accepted"]
        assert ep.agent.state["history"] == []
        assert ep.raw.count >= 2
        assert ep.journals["transitions"].rows[-1]["accepted"] is False


def test_DONE_requires_identical_pre_post_owned_document(tmp_path):
    with fake_native_episode(tmp_path) as (ep, _agent, _model):
        ep.tick()
        original = ep.agent.command

        def command(name, *args, **kwargs):
            result = original(name, *args, **kwargs)
            if name == "tick" and result["status"] == "done":
                ep.transport.public_document_commits.append(
                    {
                        "session": "session-1",
                        "frame_id": "root-1",
                        "source": "cdp_frame_commit",
                        "setup_blank": False,
                        "loader_id": "new-loader",
                        "url": result["page"]["url"],
                    }
                )
            return result

        ep.agent.command = command
        with pytest.raises(ValueError, match="guard rejected"):
            ep.tick()
        assert ep.pending[-1]["selected"]["operation"] == "DONE"
        assert ep.pending[-1]["accepted"] is False


def test_failed_private_predicates_are_durable_before_refusal(tmp_path):
    with fake_native_episode(tmp_path) as (ep, _agent, _model):
        ep.evidence = FakeEvidence(failed_check="same_native_document")
        with pytest.raises(ValueError, match="binding failed"):
            ep.proof("failing-read")
        saved = json.loads((ep.folder / "private-dom.jsonl").read_text().splitlines()[-1])
        assert saved["observed"]["actual_private_fact"] == "not in native payload"
        assert saved["verdict"]["checks"]["same_native_document"] is False
        assert saved["expected_ownership"]["loader_id"] == "loader-0"
        assert (ep.bundle / saved["HTML"]["artifact_path"]).is_file()


def test_private_HTML_read_error_keeps_observation_and_verdict(tmp_path, monkeypatch):
    with fake_native_episode(tmp_path) as (ep, _agent, _model):
        monkeypatch.setattr(
            ep.pages, "store", lambda _html: (_ for _ in ()).throw(OSError("disk read/write error"))
        )
        with pytest.raises(OSError, match="disk read/write"):
            ep.proof("failed-HTML")
        row = ep.journals["private-dom"].rows[-1]
        assert row["observed"] and row["verdict"] and row["HTML_error"]["type"] == "OSError"


def test_targetless_rows_keep_operation_only_and_first_proof_DONE(tmp_path):
    with fake_native_episode(tmp_path) as (ep, _agent, model):
        ep.execute()
        m.write(ep.folder / "public-request-ledger.json", {})
        cases, _ = m.serialize_episode(ep, "episodes/ordinary")
        assert cases[-1]["gold"] == {"operation": "DONE"}
        payload = ep.pending[0]["payload"]
        selected = {"operation": "SCROLL_DOWN", "target": None}
        response = m.teacher_response(payload, selected)
        assert set(response["answers"]) == {"operation"}
        response = m.teacher_response(payload, {"operation": "CLICK", "target": "1"})
        assert set(response["answers"]) == {"operation", "click_target"}
        assert "select_target" not in response["answers"]


def test_unoffered_path_cannot_become_direct_navigation(tmp_path):
    with fake_native_episode(tmp_path) as (ep, _agent, model):
        p = ep.agent.state["page"]
        payload = ep.model.choose
        del payload
        body = {
            "questions": {
                "operation": {
                    "criteria": {"CLICK": "Click", "SCROLL_DOWN": "Scroll", "DONE": "Done"}
                }
            }
        }
        selected = m.scripted_teacher(ep.task, p, body, model, [], {"success": False}, {})
        assert selected == {"operation": "SCROLL_DOWN", "target": None}
        p["actions"] = [a for a in p["actions"] if a["id"].upper() != "SCROLL_DOWN"]
        with pytest.raises(ValueError, match="No offered native route"):
            m.scripted_teacher(ep.task, p, body, model, [], {"success": False}, {})


def test_MDN_interfaces_anchor_not_reclicked_after_it_is_applied(tmp_path):
    with fake_native_episode(tmp_path) as (ep, _agent, model):
        t = get_public_training_task("mdn-abortcontroller-abort")
        p = synthetic_page(t["start_url"] + "#interfaces")
        links = [{"node": 1, "href": t["start_url"] + "#interfaces", "label": "Interfaces"}]
        body = {
            "questions": {"operation": {"criteria": {"CLICK": "Click", "SCROLL_DOWN": "Scroll"}}}
        }
        assert (
            m.scripted_teacher(t, p, body, model, links, {"success": False}, {})["operation"]
            == "SCROLL_DOWN"
        )


def test_MDN_offered_interfaces_precedes_same_document_specifications(tmp_path):
    with fake_native_episode(tmp_path) as (_ep, _agent, model):
        t = get_public_training_task("mdn-abortcontroller-abort")
        p = synthetic_page(t["start_url"])
        links = [
            {"node": 1, "href": t["start_url"] + "#specifications", "label": "Specifications"},
            {"node": 2, "href": t["start_url"] + "#interfaces", "label": "Interfaces"},
        ]
        body = {
            "questions": {"operation": {"criteria": {"CLICK": "Click", "SCROLL_DOWN": "Scroll"}}}
        }
        assert m.scripted_teacher(t, p, body, model, links, {"success": False}, {}) == {
            "operation": "CLICK",
            "target": "2",
        }


def test_MDN_real_empty_fragment_breadcrumb_cannot_reset_interfaces(tmp_path):
    """Exact MDN breadcrumb/TOC URLs seen in sealed02, portable CPU reproduction."""
    with fake_native_episode(tmp_path) as (_ep, _agent, model):
        t = get_public_training_task("mdn-abortcontroller-abort")
        page = synthetic_page(t["start_url"] + "#interfaces", scroll_y=18000)
        links = [
            {"node": 1, "href": t["start_url"], "label": "Web APIs"},
            {"node": 2, "href": t["start_url"] + "#specifications", "label": "Specifications"},
            {"node": 3, "href": t["start_url"] + "#interfaces", "label": "Interfaces"},
        ]
        for action, link in zip(page["actions"], links):
            action["label"] = link["label"]
        body = {
            "questions": {"operation": {"criteria": {"CLICK": "Click", "SCROLL_DOWN": "Scroll"}}}
        }
        selected = m.scripted_teacher(t, page, body, model, links, {"success": False}, {})
        assert selected == {"operation": "SCROLL_DOWN", "target": None}


def test_MDN_saved02_interfaces_views_progress_without_same_path_breadcrumb_when_present(tmp_path):
    """Uses immutable failed pilot states only; no rerun/rescore or live site read."""
    root = ROOT / "artifacts/jev-browser/20261015-public-native-training-collection-02"
    if not (root / "receipt.json").exists():
        pytest.skip("Optional sealed real MDN route diagnostic absent")
    with fake_native_episode(tmp_path) as (_ep, _agent, model):
        checked = 0
        for result in root.rglob("attempt-result.json"):
            if not result.parent.name.startswith("mdn-"):
                continue
            folder = result.parent
            rows = [
                json.loads(line)
                for line in (folder / "all-native-calls.jsonl").read_text().splitlines()
            ]
            raw = [
                json.loads(line)
                for line in (folder / "observations.jsonl").read_text().splitlines()
            ]
            t = get_public_training_task(json.loads(result.read_text())["goal_id"])
            for returned in (row for row in rows if row["phase"] == "teacher-returned"):
                request = rows[returned["request_sequence"]]
                page = raw[returned["raw_observation_sequence"]]["response"]["result"]["value"]
                if page["url"] != t["start_url"] + "#interfaces":
                    continue
                old_operation = returned["selected"]["operation"]
                if old_operation != "CLICK":
                    continue
                _elements, targets, _controls = model.action_space(page["actions"])
                old = targets[old_operation][returned["selected"]["target"]]
                href = next(
                    (
                        link["href"]
                        for link in returned["offered_link_metadata"]
                        if link["node"] == old["node"]
                    ),
                    None,
                )
                if href != t["start_url"]:
                    continue
                selected = m.scripted_teacher(
                    t,
                    page,
                    request["payload"],
                    model,
                    returned["offered_link_metadata"],
                    {"success": False},
                    {},
                    bootstrap=returned["bootstrap"],
                )
                if selected["operation"] == "CLICK":
                    chosen = targets["CLICK"][selected["target"]]
                    chosen_href = next(
                        link["href"]
                        for link in returned["offered_link_metadata"]
                        if link["node"] == chosen["node"]
                    )
                    fragment = t[
                        "bootstrap_index_section_fragment"
                        if returned["bootstrap"]
                        else "index_section_fragment"
                    ]
                    assert chosen_href == t["start_url"] + "#" + fragment
                else:
                    assert selected == {"operation": "SCROLL_DOWN", "target": None}
                checked += 1
        assert checked > 0


@pytest.mark.parametrize("bootstrap", [False, True])
@pytest.mark.parametrize(
    "goal_id",
    [t["goal_id"] for t in m.private_modules()[0].tasks() if t["judge_kind"] == "mdn_method"],
)
def test_MDN_index_hint_uses_current_trajectory_and_stops_reclicking(tmp_path, goal_id, bootstrap):
    with fake_native_episode(tmp_path) as (_ep, _agent, model):
        t = get_public_training_task(goal_id)
        fragment = t["bootstrap_index_section_fragment" if bootstrap else "index_section_fragment"]
        page = synthetic_page(t["start_url"])
        links = [
            {"node": 1, "href": t["start_url"], "label": "Web APIs"},
            {"node": 2, "href": t["start_url"] + "#specifications", "label": "Specifications"},
            {"node": 3, "href": t["start_url"] + "#interfaces", "label": "Interfaces"},
        ]
        body = {
            "questions": {"operation": {"criteria": {"CLICK": "Click", "SCROLL_DOWN": "Scroll"}}}
        }
        assert m.scripted_teacher(
            t, page, body, model, links, {"success": False}, {}, bootstrap=bootstrap
        ) == {"operation": "CLICK", "target": "2" if fragment == "specifications" else "3"}
        page["url"] += "#" + fragment
        assert m.scripted_teacher(
            t, page, body, model, links, {"success": False}, {}, bootstrap=bootstrap
        ) == {"operation": "SCROLL_DOWN", "target": None}


@pytest.mark.parametrize(
    "goal_id",
    ["mdn-intersectionobserver-observe", "mdn-resizeobserver-observe", "mdn-urlsearchparams-get"],
)
def test_MDN_category_waypoint_requires_actual_offered_link(tmp_path, goal_id):
    with fake_native_episode(tmp_path) as (_ep, _agent, model):
        t = get_public_training_task(goal_id)
        page = synthetic_page(t["start_url"])
        body = {
            "questions": {"operation": {"criteria": {"CLICK": "Click", "SCROLL_DOWN": "Scroll"}}}
        }
        assert m.scripted_teacher(t, page, body, model, [], {"success": False}, {}) == {
            "operation": "SCROLL_DOWN",
            "target": None,
        }
        links = [
            {
                "node": 2,
                "href": "https://developer.mozilla.org" + t["route_paths"][0],
                "label": "API category",
            }
        ]
        assert m.scripted_teacher(t, page, body, model, links, {"success": False}, {}) == {
            "operation": "CLICK",
            "target": "2",
        }


def test_ownership_carries_full_document_loader_over_fragment_commit():
    t = task()
    tr = transport_for(t)
    browser = SimpleNamespace(target="target-1", session="session-1")
    page = synthetic_page(t["start_url"])
    tr.public_document_commits.append(
        {
            "session": "session-1",
            "frame_id": "root-1",
            "source": "cdp_within_document",
            "setup_blank": False,
            "loader_id": None,
            "url": t["start_url"],
        }
    )
    assert m.ownership(browser, tr, page)["loader_id"] == "loader-0"
    tr.public_root_frames = {}
    with pytest.raises(ValueError, match="root frame"):
        m.ownership(browser, tr, page)


def test_same_URL_reload_ownership_keeps_exact_ledger_prefix_and_context_snapshot(tmp_path):
    """Synthetic commit events; proves temporal joins, not a live reload."""
    with fake_native_episode(tmp_path) as (ep, _agent, _model):
        first = ep.proof("before-same-url-reload")
        row = copy.deepcopy(ep.journals["ownership"].rows[first["ownership_sequence"]])
        assert row["document_commit_count"] == 1 and row["document_request_count"] == 0
        ep.transport.public_document_requests.append(
            {"session": "session-1", "loader_id": "reloaded", "url": task()["start_url"]}
        )
        ep.transport.public_document_commits.append(
            {
                "session": "session-1",
                "frame_id": "root-1",
                "source": "cdp_frame_commit",
                "setup_blank": False,
                "loader_id": "reloaded",
                "url": task()["start_url"],
            }
        )
        later = ep.proof("after-same-url-reload")
        new = ep.journals["ownership"].rows[later["ownership_sequence"]]
        assert new["document_commit_count"] == 2 and new["document_request_count"] == 1
        prefix = ep.transport.public_document_commits[: row["document_commit_count"]]
        assert prefix[-1]["loader_id"] == row["binding"]["loader_id"] == "loader-0"
        assert (
            ep.transport.public_document_commits[: new["document_commit_count"]][-1]["loader_id"]
            == new["binding"]["loader_id"]
            == "reloaded"
        )
        assert row["binding"]["url"] == new["binding"]["url"]
        assert set(row["binding"]) == {
            "target_id",
            "session_id",
            "root_frame_id",
            "loader_id",
            "native_page_key",
            "url",
        }
        assert new["owned_target_ids"] == ["target-1"] and new["owned_session_ids"] == ["session-1"]
        assert new["root_frames"] == {"session-1": "root-1"}
        ep.transport.public_root_frames["session-1"] = "changed-later"
        assert row == ep.journals["ownership"].rows[first["ownership_sequence"]]
        assert new["root_frames"] == {"session-1": "root-1"}


@pytest.mark.parametrize("primary_type", [RuntimeError, KeyboardInterrupt])
def test_final_snapshot_failure_restores_hooks_and_preserves_primary(
    tmp_path, monkeypatch, primary_type
):
    t = task()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "attempts").mkdir()
    pages = m.Pages(bundle / "pages")
    tr = transport_for(t)
    post_original = object()
    field_original = object()
    agent_field_original = object()
    model = SimpleNamespace(post_json=post_original, field_text=field_original)
    browser = SimpleNamespace(close=lambda: None)

    class Agent:
        def __init__(self, *args, **kwargs):
            self.state = {"goal": t["goal"], "history": []}
            self.browser = browser

        def snapshot(self):
            raise OSError("snapshot failed")

        def close(self):
            pass

    agent_module = SimpleNamespace(Agent=Agent, field_text=agent_field_original)

    @contextmanager
    def load(*args):
        yield agent_module, model

    monkeypatch.setattr(m, "load_upstream", load)
    monkeypatch.setattr(m, "StatusTransport", lambda _chrome: tr)
    monkeypatch.setattr(m.Episode, "install_guards", lambda self: None)
    monkeypatch.setattr(
        m.Episode,
        "execute",
        lambda self: (_ for _ in ()).throw(primary_type("primary teacher failure")),
    )
    monkeypatch.setattr(
        m,
        "_finish_public_browser",
        lambda chrome, transport, folder, secrets: m.write(
            folder / "public-request-ledger.json", {}
        ),
    )
    upstream = tmp_path / "upstream"
    (upstream / "jev_ultrafast").mkdir(parents=True)
    (upstream / "jev_ultrafast/snapshot.js").write_text("pinned synthetic source")

    def run():
        return m.run_episode(
            bundle,
            t,
            "ordinary",
            {"collection_id": "test"},
            pages,
            FakeCompiler(),
            FakeEvidence(),
            SimpleNamespace(chromium=SimpleNamespace(launch=lambda **kwargs: object())),
            upstream,
        )

    if primary_type is KeyboardInterrupt:
        with pytest.raises(m.EpisodeInterrupted) as captured:
            run()
        ep, result = captured.value.episode, captured.value.result
        assert isinstance(captured.value.primary, KeyboardInterrupt)
        assert (
            json.loads((ep.folder / "attempt-result.json").read_text())["error"]["type"]
            == "KeyboardInterrupt"
        )
    else:
        ep, result = run()
    assert result["error"]["message"] == "primary teacher failure"
    assert any(
        r["stage"] == "final_native_snapshot_write" and r["error"]["message"] == "snapshot failed"
        for r in result["cleanup_errors"]
    )
    assert (
        model.post_json is post_original
        and model.field_text is field_original
        and agent_module.field_text is agent_field_original
    )
    assert result["success"] is False
    assert result["history_actions"] == result["actual_native_policy_calls"] == 0
    assert result["bootstrap_decisions"] == result["supervised_decisions"] == 0
    assert result["unresolved_native_calls"] == 0
    assert result["actual_native_requests_by_entry_phase"] == {"bootstrap": 0, "supervised": 0}


def test_native_compiler_real_cached_Rust_full58_and_no_pruning():
    pytest.importorskip("tokenizers")
    pytest.importorskip("jinja2")
    path = (
        ROOT
        / "artifacts/jev-browser/20261015-public-native-training-start-readiness-01/postgres18-limit-offset/native-original-wire.json"
    )
    wire = json.loads(path.read_text())
    wire["model"] = m.MODEL_ALIAS
    compiled = m.NativeCompiler().compile(wire)
    assert len(compiled["labels"]["click_target"]) == 58
    assert compiled["prompt_tokens"] + 32 <= 8192 and compiled["grammar"]["eos_token_id"] == 106
    assert compiled["wire_sha256"] == m.sha(m.encode(wire))


def test_projected_future_fit_caps_count_parent_and_both_full_bundles(tmp_path):
    bundles = {s: tmp_path / s for s in ("train", "development")}
    for root in bundles.values():
        root.mkdir()
        (root / "raw.json").write_bytes(b"proof")
    result = m.projected_fit_bound(
        bundles, {"train": [], "development": []}, {"size_bytes": 1024**3, "file_count": 9999}
    )
    assert result["success"] is False
    assert result["both_full_bundle_files"] == 2 and result["unchanged_limit_bytes"] == 1024**3


def test_setup_failure_retains_all24_empty_disjoint_bundles(tmp_path, monkeypatch):
    goals = tasks()
    plan = {
        "collection_id": "cpu-failed",
        "tasks": goals,
        "scope": "CPU failure test",
        "ordered_episode_ids": [
            "cpu-failed-" + t["goal_id"] + "-" + k for t in goals for k in ("ordinary", "recovery")
        ],
        "upstream": {"source_sha256": {"jev_ultrafast/snapshot.js": "a" * 64}},
        "prospective_existing_parent_metadata": {"size_bytes": 0, "file_count": 0},
    }
    monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
    monkeypatch.setattr(m, "upstream_manifest", lambda _path: plan["upstream"])
    monkeypatch.setattr(
        m, "parent_tree_metadata", lambda: plan["prospective_existing_parent_metadata"]
    )
    output = tmp_path / "failed"
    with pytest.raises(ModuleNotFoundError):
        m.collect(output, tmp_path, plan)
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["stage"] == "partial" and receipt["all24_attempt_records_retained"]
    assert len(receipt["outcomes"]) == 24 and all(
        r["attempted"] is False for r in receipt["outcomes"]
    )
    assert (
        (output / "train/cases.jsonl").read_bytes()
        == (output / "development/cases.jsonl").read_bytes()
        == b""
    )
    assert json.loads((output / "train/receipt.json").read_text())["split"] == "train"
    assert json.loads((output / "development/receipt.json").read_text())["split"] == "development"


@pytest.mark.parametrize(
    "refusal", ["late_serializer", "final_cap", "native_post_proof", "interrupted"]
)
def test_late_population_refusal_quarantines_every_pair(tmp_path, monkeypatch, refusal):
    """Publication protocol only; synthetic episodes are not public evidence."""
    goals = tasks()
    parent = {"size_bytes": 0, "file_count": 0}
    plan = {
        "collection_id": "cpu-publication",
        "tasks": goals,
        "scope": "Synthetic publication failure test",
        "ordered_episode_ids": [
            "cpu-publication-" + t["goal_id"] + "-" + k
            for t in goals
            for k in ("ordinary", "recovery")
        ],
        "runtime_source_pins": {},
        "upstream": {"source_sha256": {"jev_ultrafast/snapshot.js": "a" * 64}},
        "prospective_existing_parent_metadata": parent,
    }
    native = tmp_path / "native/jev_ultrafast"
    native.mkdir(parents=True)
    (native / "snapshot.js").write_text("synthetic test source")

    @contextmanager
    def fake_playwright():
        yield object()

    monkeypatch.setitem(
        sys.modules, "playwright.sync_api", SimpleNamespace(sync_playwright=fake_playwright)
    )
    monkeypatch.setattr(m, "NativeCompiler", FakeCompiler)
    monkeypatch.setattr(m, "capture_runtime_sources", lambda *a, **k: {"files": {}})
    monkeypatch.setattr(m, "verify_runtime_sources", lambda *a, **k: {"success": True})
    monkeypatch.setattr(m, "verify_source_copies", lambda *a, **k: {"success": True})
    monkeypatch.setattr(m, "upstream_manifest", lambda _path: plan["upstream"])
    monkeypatch.setattr(m, "parent_tree_metadata", lambda: parent)
    actual_pair_coverage = m.pair_coverage

    def coverage(ordinary, recovery):
        if any(not row.get("accepted") for ep in (ordinary, recovery) for row in ep.pending):
            return actual_pair_coverage(ordinary, recovery)
        return {
            "required_complete": True,
            "required": {"synthetic": True},
            "observed_position_incomplete": False,
        }

    monkeypatch.setattr(m, "pair_coverage", coverage)

    def fake_episode(bundle, task, kind, *args):
        if (
            refusal == "native_post_proof"
            and task["goal_id"] == "git-restore-description"
            and kind == "ordinary"
        ):
            probe = tmp_path / "real-native-partial"
            probe.mkdir()
            with fake_native_episode(probe) as (ep, _agent, _model):
                original = ep.proof

                def proof(phase, **kwargs):
                    if phase == "before-native-call-1":
                        ep.evidence.failed_check = "same_native_document"
                    return original(phase, **kwargs)

                ep.proof = proof
                ep.tick()
                with pytest.raises(ValueError, match="binding failed"):
                    ep.tick()
                assert not ep.pending[-1]["accepted"]
                assert actual_pair_coverage(ep, ep)["required_complete"] is False
                return ep, {
                    "episode_id": ep.id,
                    "goal_id": task["goal_id"],
                    "split": task["split"],
                    "trajectory_kind": kind,
                    "success": False,
                    "error": {
                        "type": "ValueError",
                        "message": "Actual next-native-pre proof refusal",
                    },
                }
        folder = bundle / "attempts" / (task["goal_id"] + "-" + kind)
        folder.mkdir()
        journal = m.Journal(folder / "all-native-calls.jsonl")
        ep = SimpleNamespace(
            id=plan["collection_id"] + "-" + folder.name,
            folder=folder,
            pending=[],
            bootstrap_reached=False,
            journals={"all-native-calls": journal},
        )
        result = {
            "episode_id": ep.id,
            "goal_id": task["goal_id"],
            "split": task["split"],
            "trajectory_kind": kind,
            "success": True,
        }
        if refusal == "interrupted":
            result.update(
                success=False,
                error={"type": "KeyboardInterrupt", "message": "Actual interrupted attempt"},
            )
            raise m.EpisodeInterrupted(KeyboardInterrupt("Actual interrupted attempt"), ep, result)
        return ep, result

    monkeypatch.setattr(m, "run_episode", fake_episode)
    serialized_count = 0

    def fake_serialize(ep, relative):
        nonlocal serialized_count
        serialized_count += 1
        if refusal == "late_serializer" and serialized_count == 3:
            raise ValueError("Injected late serializer refusal after first good pair")
        return [{"request": {"synthetic": ep.id}}], [{"synthetic": ep.id}]

    monkeypatch.setattr(m, "serialize_episode", fake_serialize)
    original_projection = m.projected_fit_bound

    def projection(*args, **kwargs):
        result = original_projection(*args, **kwargs)
        if refusal == "final_cap" and kwargs.get("publication_files"):
            result["success"] = False
        return result

    monkeypatch.setattr(m, "projected_fit_bound", projection)
    output = tmp_path / "output"
    expected_error = KeyboardInterrupt if refusal == "interrupted" else ValueError
    message = (
        "Actual interrupted attempt" if refusal == "interrupted" else "Public collection partial"
    )
    with pytest.raises(expected_error, match=message):
        m.collect(output, native.parent, plan)
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["stage"] == "partial" and not receipt["all24_mandatory_pair_gate"]
    assert len(receipt["outcomes"]) == 24
    assert (
        serialized_count == 0
        if refusal in {"native_post_proof", "interrupted"}
        else serialized_count >= 3
    )
    assert all(not r["fit_eligible"] for r in receipt["outcomes"])
    assert sum(r["attempted"] for r in receipt["outcomes"]) == (
        1 if refusal == "interrupted" else 24
    )
    assert sum(r["success"] for r in receipt["outcomes"]) == (
        0 if refusal == "interrupted" else 23 if refusal == "native_post_proof" else 24
    )
    assert all(not pair["fit_eligible_pair"] for pair in receipt["coverage"])
    for split, count in (("train", 16), ("development", 8)):
        bundle = output / split
        assert not list((bundle / "episodes").iterdir())
        expected_count = (1 if split == "train" else 0) if refusal == "interrupted" else count
        assert len(list((bundle / "quarantined").iterdir())) == expected_count
        assert (bundle / "cases.jsonl").read_bytes() == b""
        assert (bundle / "wire_requests.jsonl").read_bytes() == b""
        assert json.loads((bundle / "receipt.json").read_text())["fit_eligible_row_count"] == 0


def test_browser_evidence_allows_new_collector_and_no_arbitrary_callers(tmp_path):
    manifest = m.capture_runtime_sources(tmp_path, additional_files=m.source_files())
    assert (
        manifest["files"]["scripts/jev_browser_collect_public_training.py"]["sha256"]
        == m.record(m.__file__)["sha256"]
    )
    with pytest.raises(ValueError, match="approved"):
        m.capture_runtime_sources(
            tmp_path / "bad", additional_files={"scripts/not-approved.py": Path(m.__file__)}
        )
