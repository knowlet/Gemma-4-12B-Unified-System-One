"""Opt-in scroll recovery through pinned native Agent/Browser, fake CDP leaves only."""

import copy
import os
import sys
from contextlib import contextmanager

import pytest
import test_jev_browser_collect_public_training as native

from s1 import jev_observation_profile as profile

m = native.m


@contextmanager
def actor(
    tmp_path,
    *,
    kind="ordinary",
    direction="SCROLL_DOWN",
    failures=1,
    opt_in=True,
    fault=None,
    pre_refresh=False,
):
    t = native.task()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    helper_sha = m.record(profile.__file__)["sha256"]
    binding = m.observation_profile_binding(native.UPSTREAM, m.COMPOSED_PROFILE, helper_sha)
    plan = {
        "collection_id": "synthetic-scroll",
        "schema_id": m.SCROLL_SCHEMA if opt_in else m.PROFILE_SCHEMA,
        "observation_profile": binding,
        "upstream": m.upstream_manifest(native.UPSTREAM),
    }
    if opt_in:
        plan["no_input_scroll_recovery"] = copy.deepcopy(m.NO_INPUT_SCROLL_RULE)
    profile.materialize_profile(
        native.UPSTREAM,
        bundle / "observation-profile",
        expected_source_sha256=helper_sha,
        profile=m.COMPOSED_PROFILE,
    )
    handle = m.verify_observation_profile(bundle, plan)
    ep = m.Episode(
        bundle,
        bundle / "attempt",
        t,
        kind,
        plan,
        m.Pages(bundle / "pages"),
        native.FakeCompiler(),
        native.FakeEvidence(),
        observation_runtime=handle,
    )
    tr = native.transport_for(t)
    tr.observation_profile_errors = []
    ep.transport = tr
    inputs = []
    with m.load_upstream(native.UPSTREAM, tr, observation_runtime=handle) as (agent, model):
        browser_module = sys.modules["jev_ultrafast.browser"]
        native_browser = browser_module.Browser
        source_fingerprint = browser_module.fingerprint
        old_operation = browser_module.browser_operation
        original_teacher = m.scripted_teacher
        calls = {"count": 0}

        class Browser(native_browser):
            def __init__(self, url):
                self.target, self.session, self.url = "target-1", "session-1", url
                self.scroll_y = 560 if direction == "SCROLL_UP" else 0
                self.text = "Synthetic visible reference"
                self.remaining = failures
                self.refresh_pending = pre_refresh
                self.counter = 0

            def page(self):
                page = native.synthetic_page(self.url, scroll_y=self.scroll_y)
                page["actions"][-3]["delta"] = 560
                page["actions"][-2]["delta"] = -560
                page.update(
                    text=self.text,
                    w=1120,
                    h=780,
                    observation_profile={
                        "id": m.COMPOSED_PROFILE,
                        "visited_nodes": 42,
                        "node_limit": 20000,
                        "depth_limit": 128,
                        "node_limit_reached": False,
                        "depth_limit_reached": False,
                    },
                )
                page["marker"] = [
                    *page["page_key"][:6],
                    page["title"],
                    page["text"],
                    [
                        {k: v for k, v in a.items() if k not in {"id", "rect"}}
                        for a in page["actions"]
                        if "node" in a
                    ],
                    page["page_key"][6],
                ]
                page.pop("fingerprint", None)
                page["fingerprint"] = source_fingerprint(page)
                return page

            def emit(self, expression, value):
                tr.calls += 1
                event = {
                    "protocol_call": tr.calls,
                    "session": self.session,
                    "expression": expression,
                    "response": {"result": {"value": copy.deepcopy(value)}},
                }
                ep.profile_tracker.capture(event)
                if expression == handle.read_state:
                    ep.raw_sink({k: v for k, v in event.items() if k != "expression"})

            def evaluate(self, expression):
                if expression == browser_module.MARKER:
                    guard = (
                        ep.journals["guarded-actions"].rows[-1]
                        if ep.journals["guarded-actions"].rows
                        else {}
                    )
                    args = guard.get("arguments", [])
                    if self.refresh_pending:
                        self.text += " initial native refresh"
                        self.refresh_pending = False
                    if self.remaining and len(args) == 2 and args[1].get("kind") == "scroll":
                        self.remaining -= 1
                        if fault == "fresh_exception":
                            raise browser_module.StalePage("Document changed during evaluation")
                        self.text += " newly visible public text"
                        if fault == "geometry":
                            self.scroll_y += 1
                    marker = self.page()["marker"]
                    self.emit(expression, marker)
                    return marker
                if "c.pageKey(),c.guard" in expression:
                    return [self.page()["page_key"], None]
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
                    return "<html>synthetic public reference</html>"
                return [
                    {
                        "node": n,
                        "href": "https://git-scm.com"
                        + (
                            t["target_path"]
                            if n == 1
                            else t["bootstrap_target_path"]
                            if n == 2
                            else "/docs/unrelated"
                        ),
                        "label": f"Link {n}",
                    }
                    for n in range(1, 59)
                ]

            def call(self, *args, **kwargs):
                return {"result": {"value": None}}  # Native post-input passive helper leaf.

            def close(self):
                pass

        def operation(request):
            browser = ep.agent.browser if ep.agent is not None else active[0]
            if request["operation"] == "observe":
                page = browser.page()
                raw = {k: copy.deepcopy(v) for k, v in page.items() if k != "fingerprint"}
                browser.emit(handle.read_state, raw)
                return page
            assert request["operation"] == "act"
            inputs.append(copy.deepcopy(request))
            action = request["action"]
            if action["kind"] == "scroll":
                browser.scroll_y = max(0, browser.scroll_y + action["delta"])
            else:
                browser.url = "https://git-scm.com" + (
                    t["target_path"] if action["node"] == 1 else t["bootstrap_target_path"]
                )
                browser.counter += 1
                browser.scroll_y = 0
                tr.public_document_commits.append(
                    {
                        "session": "session-1",
                        "frame_id": "root-1",
                        "source": "cdp_frame_commit",
                        "setup_blank": False,
                        "loader_id": f"loader-{browser.counter}",
                        "url": browser.url,
                    }
                )
            return {"executed": action["id"]}

        active = []
        original_init = Browser.__init__

        def init(self, url):
            original_init(self, url)
            active[:] = [self]

        Browser.__init__ = init
        agent.Browser = Browser
        browser_module.browser_operation = operation
        old_post = model.post_json
        model.post_json = ep.post

        def teacher(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] <= failures + 1:
                return {"operation": direction, "target": None}
            return original_teacher(*args, **kwargs)

        m.scripted_teacher = teacher
        old_env = {key: os.environ.get(key) for key in ("TYPESAFE_MODEL", "TYPESAFE_API_KEY")}
        os.environ.update(TYPESAFE_MODEL=m.MODEL_ALIAS, TYPESAFE_API_KEY="synthetic-only")
        try:
            ep.model = model
            ep.agent = agent.Agent(t["start_url"], t["goal"], screenshots=False)
            ep.install_guards()
            yield ep, inputs
        finally:
            browser_module.browser_operation = old_operation
            model.post_json = old_post
            m.scripted_teacher = original_teacher
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


@pytest.mark.parametrize("direction", ["SCROLL_UP", "SCROLL_DOWN"])
@pytest.mark.parametrize("kind", ["ordinary", "recovery"])
def test_native_false_scroll_reobserves_then_new_tick_can_complete(tmp_path, direction, kind):
    with actor(tmp_path, direction=direction, kind=kind, pre_refresh=True) as (ep, inputs):
        row = ep.tick()
        assert row["accepted"] is False and row["excluded_no_input"] is True
        assert inputs == [] and ep.agent.state["history"] == []
        assert ep.agent.state["status"] == "ready" and ep.agent.state["decision"] is None
        evidence = m._require_no_input_scroll_exclusion(ep, row)
        assert evidence["rule_id"] == m.NO_INPUT_SCROLL_RULE["rule_id"]
        prefix = {p.name: p.read_bytes() for p in ep.folder.glob("*.jsonl")}
        result = ep.execute()
        assert result["success"] and inputs
        assert all(
            (ep.folder / name).read_bytes().startswith(blob) for name, blob in prefix.items()
        )
        fields = m.scroll_accounting(ep.plan, ep.journals["all-native-calls"].rows)
        assert fields["excluded_no_input_scroll_count"] == 1
        assert fields["excluded_no_input_scroll_native_request_sequences"] == [row["call_sequence"]]
        m.write(ep.folder / "public-request-ledger.json", {"synthetic_CPU_ledger": True})
        cases, wires = m.serialize_episode(ep, "episodes/synthetic")
        assert cases and wires and cases[-1]["gold"]["operation"] == "DONE"
        assert row["call_sequence"] not in [w["native_policy_request_sequence"] for w in wires]
        assert all(ep.pending[int(w["compiled_input_sequence"])]["accepted"] for w in wires)


def test_default_v4_still_refuses_same_stale_scroll(tmp_path):
    with actor(tmp_path, opt_in=False) as (ep, inputs):
        with pytest.raises(ValueError, match="Native freshness/action guard rejected"):
            ep.tick()
        assert inputs == [] and not ep.pending[-1].get("excluded_no_input")


@pytest.mark.parametrize("fault", ["geometry", "fresh_exception"])
def test_other_native_guard_failures_still_refuse(tmp_path, fault):
    with actor(tmp_path, fault=fault) as (ep, inputs):
        with pytest.raises(ValueError, match="Native freshness/action guard rejected"):
            ep.tick()
        assert inputs == [] and not ep.pending[-1].get("excluded_no_input")


def test_excluded_scrolls_consume_original16_budget_without_inputs(tmp_path):
    with actor(tmp_path, failures=50) as (ep, inputs):
        with pytest.raises(ValueError, match="16 supervised"):
            ep.execute()
        assert ep.resolved_counts() == {"bootstrap": 0, "supervised": 16}
        assert len(ep.pending) == 16 and inputs == []
        assert (
            m.scroll_accounting(ep.plan, ep.journals["all-native-calls"].rows)[
                "excluded_no_input_scroll_count"
            ]
            == 16
        )
        assert not m.pair_coverage(ep, ep)["required"]["ordinary_DONE"]


@pytest.mark.parametrize(
    "mutation",
    [
        "history",
        "accepted_act",
        "wrong_args",
        "extra_guard",
        "new_loader",
        "raw_missing",
        "marker_geometry",
        "forged_flag",
        "duplicate_event",
        "DONE",
        "CLICK",
        "SELECT",
        "TYPE_TEXT",
        "WAIT",
        "source",
        "profile_failure",
        "uncompleted_previous",
    ],
)
def test_exclusion_rederived_and_cannot_hide_other_failures(tmp_path, mutation):
    with actor(tmp_path) as (ep, inputs):
        row = ep.tick()
        transition = ep.journals["transitions"].rows[row["transition_sequence"]]
        guards = ep.journals["guarded-actions"].rows
        start = transition["execution_guard_sequence_start"]
        if mutation == "history":
            ep.journals["states"].rows[-1]["state"]["history"].append({"invented": "mutation"})
        elif mutation == "accepted_act":
            guards[start + 3]["accepted"] = True
        elif mutation == "wrong_args":
            guards[start]["keywords"]["text"] = "unexpected"
        elif mutation == "extra_guard":
            transition["guard_sequence_end"] += 1
        elif mutation == "new_loader":
            ep.journals["private-dom"].rows[transition["immediate_private_diagnostic_sequence"]][
                "expected_ownership"
            ]["loader_id"] = "new-loader"
        elif mutation == "raw_missing":
            transition["raw_observation_count_after_native_tick"] = transition[
                "execution_raw_observation_start"
            ]
        elif mutation == "marker_geometry":
            next(
                e
                for e in ep.profile_tracker._rows
                if e["sequence"] == transition["exclusion_evidence"]["marker_event_sequence"]
            )["response"]["result"]["value"][3] += 1
        elif mutation == "forged_flag":
            transition["exclusion_evidence"]["unchanged_full_history_sha256"] = "0" * 64
        elif mutation == "duplicate_event":
            ep.journals["all-native-calls"].append(
                copy.deepcopy(ep.journals["all-native-calls"].rows[row["exclusion_event_sequence"]])
            )
        elif mutation in {"DONE", "CLICK", "SELECT", "TYPE_TEXT", "WAIT"}:
            row["selected"]["operation"] = mutation
        elif mutation == "source":
            ep.plan["upstream"]["source_sha256"]["jev_ultrafast/browser.py"] = "0" * 64
        elif mutation == "uncompleted_previous":
            ep.awaiting_post = {"uncompleted": True}
        else:
            ep.transport.observation_profile_errors.append(
                {"kind": "integrity", "reason": "synthetic sink failure"}
            )
        with pytest.raises(ValueError, match="Unverified excluded-no-input SCROLL"):
            m._require_no_input_scroll_exclusion(ep, row)
        assert inputs == []


def test_flag_is_explicit_composed_only_and_old_plans_remain_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "ASSET_SHA", {})
    monkeypatch.setattr(m, "parent_tree_metadata", lambda: {"size_bytes": 0, "file_count": 0})
    helper_sha = m.record(profile.__file__)["sha256"]
    plain = m.collection_plan(tmp_path / "native", native.UPSTREAM, "native-default")
    composed = m.collection_plan(
        tmp_path / "v4",
        native.UPSTREAM,
        "composed-default",
        observation_profile=m.COMPOSED_PROFILE,
        observation_profile_helper_sha256=helper_sha,
    )
    opted = m.collection_plan(
        tmp_path / "v5",
        native.UPSTREAM,
        "scroll-opted",
        observation_profile=m.COMPOSED_PROFILE,
        observation_profile_helper_sha256=helper_sha,
        allow_no_input_stale_scroll=True,
    )
    assert plain["schema_id"] == m.SCHEMA and composed["schema_id"] == m.PROFILE_SCHEMA
    assert "no_input_scroll_recovery" not in plain and "no_input_scroll_recovery" not in composed
    assert (
        opted["schema_id"] == m.SCROLL_SCHEMA
        and opted["no_input_scroll_recovery"] == m.NO_INPUT_SCROLL_RULE
    )
    assert opted["tasks"] == composed["tasks"] == plain["tasks"]
    assert (
        opted["no_input_done_recovery"]
        == composed["no_input_done_recovery"]
        == plain["no_input_done_recovery"]
    )
    assert opted["limits"] == composed["limits"] == plain["limits"]
    with pytest.raises(ValueError, match="requires explicit composed"):
        m.collection_plan(
            tmp_path / "refused", native.UPSTREAM, "refused", allow_no_input_stale_scroll=True
        )


def test_scroll_accounting_never_counts_done_exclusions(tmp_path):
    with actor(tmp_path) as (ep, _inputs):
        ep.tick()
        calls = copy.deepcopy(ep.journals["all-native-calls"].rows)
        calls.append(
            {
                "phase": "excluded-no-input",
                "request_sequence": 123,
                "rule_id": m.NO_INPUT_DONE_RULE["rule_id"],
            }
        )
        result = m.scroll_accounting(ep.plan, calls)
        assert result["excluded_no_input_scroll_count"] == 1
        assert result["excluded_no_input_scroll_native_request_sequences"] == [0]
        old_plan = copy.deepcopy(ep.plan)
        old_plan["schema_id"] = m.PROFILE_SCHEMA
        old_plan.pop("no_input_scroll_recovery")
        assert m.scroll_accounting(old_plan, calls) == {}
