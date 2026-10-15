"""CPU-only collector profile contracts; fake episodes are not public collection."""

import copy
import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import test_jev_browser_collect_public_training as native

from s1 import jev_observation_profile as helper
from s1.jev_observation_events import ObservationProfileTracker

m = native.m
UPSTREAM = native.UPSTREAM
HELPER_SHA = m.record(helper.__file__)["sha256"]


def binding(profile=m.COMPOSED_PROFILE):
    return m.observation_profile_binding(UPSTREAM, profile, HELPER_SHA)


def materialized(tmp_path, profile=m.COMPOSED_PROFILE):
    plan = {"schema_id": m.PROFILE_SCHEMA, "observation_profile": binding(profile)}
    if profile == m.PRUNED_PROFILE:
        plan.update(
            schema_id=m.SCROLL_SCHEMA,
            no_input_scroll_recovery=copy.deepcopy(m.NO_INPUT_SCROLL_RULE),
        )
    helper.materialize_profile(
        UPSTREAM,
        tmp_path / "observation-profile",
        expected_source_sha256=HELPER_SHA,
        profile=profile,
    )
    return plan, m.verify_observation_profile(tmp_path, plan)


def enable_profile(ep, profile=m.COMPOSED_PROFILE):
    """Inject explicit fake protocol events into an otherwise real native Agent."""
    ep.plan.update(schema_id=m.PROFILE_SCHEMA, observation_profile=binding(profile))
    if profile == m.PRUNED_PROFILE:
        ep.plan.update(
            schema_id=m.SCROLL_SCHEMA,
            no_input_scroll_recovery=copy.deepcopy(m.NO_INPUT_SCROLL_RULE),
        )
    ep.journals[m.PROFILE_JOURNAL] = m.ProfileEventJournal(
        ep.folder / (m.PROFILE_JOURNAL + ".jsonl")
    )
    ep.profile_tracker = ObservationProfileTracker(
        read_state="synthetic effective read",
        marker_expression="synthetic effective marker",
        manifest_sha256=ep.plan["observation_profile"]["manifest_sha256"],
        profile=profile,
        sink=ep.journals[m.PROFILE_JOURNAL].append,
    )
    ep.transport.observation_profile_errors = []


def event(ep, *, kind="READ_STATE", cap=False, transient=False, malformed=False):
    metadata = {
        "id": ep.plan["observation_profile"]["profile"],
        "visited_nodes": 20000 if cap else 14,
        "node_limit": 20000,
        "depth_limit": 128,
        "node_limit_reached": cap,
        "depth_limit_reached": False,
    }
    marker = [1000, native.task()["start_url"], 0, 0, 1120, 780, "Title", "Text", [], []]
    if cap:
        response = {
            "exceptionDetails": {
                "exception": {
                    "description": "ComposedVisibleTextLimitError: COMPOSED_VISIBLE_TEXT_LIMIT "
                    + json.dumps(metadata)
                }
            }
        }
    elif transient:
        response = {"exceptionDetails": {"text": "Execution context was destroyed"}}
    elif malformed:
        response = {"result": {"value": {"observation_profile": {}}}}
    else:
        value = (
            marker
            if kind == "MARKER"
            else {
                "observation_profile": metadata,
                "text": "Text",
                "marker": marker,
            }
        )
        response = {"result": {"value": value}}
    return ep.profile_tracker.capture(
        {
            "protocol_call": len(ep.profile_tracker.rows) + 1,
            "session": "session-1",
            "expression": "synthetic effective read"
            if kind == "READ_STATE"
            else "synthetic effective marker",
            "response": response,
        }
    )


def test_native_binding_never_imports_profile_helper(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Native default must not build or materialize a profile")

    monkeypatch.setattr(helper, "build_profile", forbidden)
    assert m.observation_profile_binding(UPSTREAM, m.NATIVE_PROFILE, None) is None
    with pytest.raises(ValueError, match="Native default"):
        m.observation_profile_binding(UPSTREAM, m.NATIVE_PROFILE, HELPER_SHA)


@pytest.mark.parametrize("profile", ["unknown", "composed-visible-text-v1", None, []])
def test_unknown_profile_fails_before_plan_or_materialization(profile):
    with pytest.raises(ValueError, match="Unknown observation profile"):
        m.observation_profile_binding(UPSTREAM, profile, HELPER_SHA)


def test_pruned_profile_requires_scroll_opt_in_before_plan_dependencies(tmp_path, monkeypatch):
    def forbidden():
        raise AssertionError("Rejected profile mode must not read tasks or private evidence")

    monkeypatch.setattr(m, "private_modules", forbidden)
    with pytest.raises(ValueError, match="v3 requires explicit no-input stale SCROLL opt-in"):
        m.collection_plan(
            tmp_path / "refused",
            UPSTREAM,
            "pruned-without-opt-in",
            observation_profile=m.PRUNED_PROFILE,
            observation_profile_helper_sha256=HELPER_SHA,
        )
    assert not (tmp_path / "refused").exists()


def test_v4_cannot_verify_pruned_profile_even_with_valid_scroll_rule(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Wrong schema must fail before profile verification")

    monkeypatch.setattr(helper, "verify_materialized_profile", forbidden)
    plan = {
        "schema_id": m.PROFILE_SCHEMA,
        "observation_profile": {
            "profile": m.PRUNED_PROFILE,
            "manifest_path": "observation-profile/profile-manifest.json",
        },
        "no_input_scroll_recovery": copy.deepcopy(m.NO_INPUT_SCROLL_RULE),
    }
    with pytest.raises(ValueError, match="Explicit composed profile"):
        m.verify_observation_profile(tmp_path, plan)


def test_explicit_plan_has_exact_complete_manifest_and_unchanged_population(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "ASSET_SHA", {})
    monkeypatch.setattr(m, "parent_tree_metadata", lambda: {"size_bytes": 0, "file_count": 0})
    native_plan = m.collection_plan(tmp_path / "native", UPSTREAM, "native-cpu")
    plan = m.collection_plan(
        tmp_path / "composed",
        UPSTREAM,
        "composed-cpu",
        observation_profile=m.COMPOSED_PROFILE,
        observation_profile_helper_sha256=HELPER_SHA,
    )
    assert native_plan["schema_id"] == m.SCHEMA and "observation_profile" not in native_plan
    assert plan["schema_id"] == m.PROFILE_SCHEMA and plan["schema_version"] == 1
    assert plan["tasks"] == native_plan["tasks"] and len(plan["ordered_episode_ids"]) == 24
    for key in (
        "supervised_decisions_per_episode",
        "recovery_bootstrap_decisions_separate",
        "upstream_action_budget_unchanged",
        "limits",
        "passive_between_ticks",
        "coverage",
    ):
        assert plan[key] == native_plan[key]
    assert plan["runtime_source_pins"]["s1/jev_observation_profile.py"]["sha256"] == HELPER_SHA
    bundle = helper.build_profile(
        helper.read_canonical_sources(UPSTREAM),
        expected_source_sha256=HELPER_SHA,
        profile=m.COMPOSED_PROFILE,
    )
    blob = (
        json.dumps(
            bundle.materialized_manifest(),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        + b"\n"
    )
    block = plan["observation_profile"]
    assert block == binding()
    assert block["manifest_sha256"] == m.sha(blob) and block["manifest_size_bytes"] == len(blob)
    assert (
        block["canonical_snapshot_js_sha256"]
        == native_plan["upstream"]["source_sha256"][helper.SNAPSHOT]
    )
    assert block["effective_snapshot_js_sha256"] != block["canonical_snapshot_js_sha256"]
    assert not (tmp_path / "composed").exists()


@pytest.mark.parametrize("profile", [m.COMPOSED_PROFILE, m.PRUNED_PROFILE])
def test_both_split_profiles_are_complete_identical_and_self_contained(tmp_path, profile):
    manifests = []
    for split in ("train", "development"):
        root = tmp_path / split
        root.mkdir()
        plan, handle = materialized(root, profile)
        files = m.inventory(root / "observation-profile")
        assert len(files) == 16
        assert handle.runtime_root == root / "observation-profile/runtime"
        assert m.profile_verification(root, plan) == {
            "success": True,
            "manifest_sha256": binding(profile)["manifest_sha256"],
            "errors": [],
        }
        manifests.append(files)
    assert manifests[0] == manifests[1]


def test_pruned_and_v2_bindings_cannot_be_interchanged(tmp_path):
    plan, handle = materialized(tmp_path, m.PRUNED_PROFILE)
    old = binding(m.COMPOSED_PROFILE)
    current = plan["observation_profile"]
    assert current["canonical_snapshot_js_sha256"] == old["canonical_snapshot_js_sha256"]
    for key in ("manifest_sha256", "effective_snapshot_js_sha256", "marker_expression_sha256"):
        assert current[key] != old[key]
    assert handle.bundle.manifest()["profile"] == m.PRUNED_PROFILE
    plan["observation_profile"]["profile"] = m.COMPOSED_PROFILE
    with pytest.raises(ValueError):
        m.verify_observation_profile(tmp_path, plan)


def test_v4_episode_rejects_pruned_handle_before_tracker_creation(tmp_path):
    plan, handle = materialized(tmp_path, m.PRUNED_PROFILE)
    plan.update(collection_id="forged-v4", schema_id=m.PROFILE_SCHEMA)
    folder = tmp_path / "refused"
    with pytest.raises(ValueError, match="explicit verified runtime"):
        m.Episode(
            tmp_path,
            folder,
            native.task(),
            "ordinary",
            plan,
            m.Pages(tmp_path / "pages"),
            native.FakeCompiler(),
            native.FakeEvidence(),
            observation_runtime=handle,
        )
    assert not (folder / (m.PROFILE_JOURNAL + ".jsonl")).exists()


@pytest.mark.parametrize(
    "damage", ["runtime", "manifest", "extra", "symlink", "plan", "canonical_pin", "manifest_path"]
)
def test_split_profile_integrity_refuses_before_browser(tmp_path, damage):
    plan, _handle = materialized(tmp_path)
    profile_root = tmp_path / "observation-profile"
    path = profile_root / "runtime" / helper.SNAPSHOT
    if damage == "runtime":
        path.write_bytes(path.read_bytes() + b"\n")
    elif damage == "manifest":
        (profile_root / "profile-manifest.json").write_text("{}\n")
    elif damage == "extra":
        (profile_root / "unused.txt").write_text("unapproved")
    elif damage == "symlink":
        path.unlink()
        path.symlink_to(UPSTREAM / helper.SNAPSHOT)
    elif damage == "canonical_pin":
        plan["observation_profile"]["canonical_snapshot_js_sha256"] = "0" * 64
    elif damage == "manifest_path":
        plan["observation_profile"]["manifest_path"] = "../unapproved-manifest.json"
    else:
        plan["observation_profile"]["effective_snapshot_js_sha256"] = "0" * 64
    result = m.profile_verification(tmp_path, plan)
    assert not result["success"] and result["errors"]


def test_episode_constructor_requires_explicit_handle_and_creates_only_v4_journal(tmp_path):
    t = native.task()
    plan, handle = materialized(tmp_path)
    plan["collection_id"] = "synthetic-profile"
    ep = m.Episode(
        tmp_path,
        tmp_path / "v4",
        t,
        "ordinary",
        plan,
        m.Pages(tmp_path / "pages"),
        native.FakeCompiler(),
        native.FakeEvidence(),
        observation_runtime=handle,
    )
    assert ep.profile_tracker is not None and (ep.folder / (m.PROFILE_JOURNAL + ".jsonl")).exists()
    plain = m.Episode(
        tmp_path,
        tmp_path / "v3",
        t,
        "ordinary",
        {"collection_id": "plain"},
        ep.pages,
        native.FakeCompiler(),
        native.FakeEvidence(),
    )
    assert plain.profile_tracker is None and plain.profile_result_fields() == {}
    assert not (plain.folder / (m.PROFILE_JOURNAL + ".jsonl")).exists()
    with pytest.raises(ValueError, match="explicit verified runtime"):
        m.Episode(
            tmp_path,
            tmp_path / "refused",
            t,
            "ordinary",
            plan,
            ep.pages,
            native.FakeCompiler(),
            native.FakeEvidence(),
        )


@pytest.mark.parametrize("profile", [m.COMPOSED_PROFILE, m.PRUNED_PROFILE])
def test_transient_native_navigation_error_is_journaled_without_poison(tmp_path, profile):
    with native.fake_native_episode(tmp_path) as (ep, _agent, _model):
        enable_profile(ep, profile)
        event(ep, transient=True)
        event(ep, kind="MARKER")
        event(ep)
        assert ep.execute()["success"]
        fields = ep.profile_result_fields()
        assert fields["observation_profile_event_count"] == 3
        assert fields["observation_profile_failures"] == []
        assert fields["observation_profile_failure_count"] == 0
        assert ep.journals[m.PROFILE_JOURNAL].rows == ep.profile_tracker.rows
        assert ep.raw.count > 0  # The original READ_STATE journal remains separate.


@pytest.mark.parametrize("failure", ["cap", "integrity", "transport"])
@pytest.mark.parametrize("profile", [m.COMPOSED_PROFILE, m.PRUNED_PROFILE])
def test_persistent_failure_cannot_be_cleared_by_good_read_or_make_fit_row(
    tmp_path, failure, profile
):
    with native.fake_native_episode(tmp_path) as (ep, _agent, _model):
        enable_profile(ep, profile)
        assert ep.execute()["success"]
        if failure == "transport":
            ep.transport.observation_profile_errors.append(
                {"kind": "integrity", "reason": "Sink failed"}
            )
        else:
            event(ep, cap=failure == "cap", malformed=failure == "integrity")
        event(ep)
        assert ep.profile_result_fields()["observation_profile_failure_count"] == 1
        with pytest.raises(ValueError, match="Persistent observation profile failure"):
            m.serialize_episode(ep, "episodes/synthetic")
        with pytest.raises(ValueError, match="Persistent observation profile failure"):
            ep.execute()


def test_profile_failure_during_native_tick_retains_action_and_excludes_label(tmp_path):
    with native.fake_native_episode(tmp_path) as (ep, _agent, _model):
        enable_profile(ep)
        act = ep.agent.browser.act

        def cap_after_real_action(*args, **kwargs):
            result = act(*args, **kwargs)
            event(ep, cap=True)
            return result

        ep.agent.browser.act = cap_after_real_action
        with pytest.raises(ValueError, match="Persistent observation profile failure"):
            ep.tick()
        assert len(ep.agent.state["history"]) == 1
        assert ep.pending[-1]["accepted"] is False
        assert ep.journals["transitions"].rows
        assert ep.profile_result_fields()["observation_profile_failure_count"] == 1


def test_profile_journal_tamper_refuses_serialization(tmp_path):
    with native.fake_native_episode(tmp_path) as (ep, _agent, _model):
        enable_profile(ep)
        event(ep)
        assert ep.execute()["success"]
        path = ep.folder / (m.PROFILE_JOURNAL + ".jsonl")
        path.write_bytes(path.read_bytes() + b"{}\n")
        with pytest.raises(ValueError, match="Profile event journal differs"):
            m.serialize_episode(ep, "episodes/synthetic")


def test_v4_setup_failure_preserves_all24_and_profile_verification_errors(tmp_path, monkeypatch):
    goals = native.tasks()
    plan = {
        "schema_id": m.PROFILE_SCHEMA,
        "collection_id": "profile-failed",
        "tasks": goals,
        "scope": "CPU failure test",
        "observation_profile": binding(),
        "ordered_episode_ids": [
            "profile-failed-" + t["goal_id"] + "-" + k
            for t in goals
            for k in ("ordinary", "recovery")
        ],
        "upstream": m.upstream_manifest(UPSTREAM),
        "prospective_existing_parent_metadata": {"size_bytes": 0, "file_count": 0},
    }
    monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
    monkeypatch.setattr(
        m, "parent_tree_metadata", lambda: plan["prospective_existing_parent_metadata"]
    )
    output = tmp_path / "failed"
    with pytest.raises(ModuleNotFoundError):
        m.collect(output, UPSTREAM, plan)
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["stage"] == "partial" and not receipt["all24_mandatory_pair_gate"]
    assert len(receipt["outcomes"]) == 24 and all(not r["attempted"] for r in receipt["outcomes"])
    assert receipt["schema_id"] == m.PROFILE_SCHEMA
    assert set(receipt["observation_profile_verification"]) == {"train", "development"}
    for split in ("train", "development"):
        row = json.loads((output / split / "receipt.json").read_text())
        assert (
            row["schema_id"] == m.PROFILE_SCHEMA
            and not row["observation_profile_verification"]["success"]
        )
        assert "native_snapshot_js_sha256" not in row
        assert row["effective_snapshot_js_sha256"] == binding()["effective_snapshot_js_sha256"]
        assert (output / split / "cases.jsonl").read_bytes() == b""
        assert (output / split / "wire_requests.jsonl").read_bytes() == b""
        assert all(
            r["observation_profile_event_count"] == 0 and r["observation_profile_failures"] == []
            for r in receipt["outcomes"]
            if r["split"] == split
        )


def test_run_episode_installs_effective_expression_and_sink_before_agent(tmp_path, monkeypatch):
    plan, handle = materialized(tmp_path)
    plan["collection_id"] = "profile-lifecycle"
    (tmp_path / "attempts").mkdir()
    t = native.task()
    tr = native.transport_for(t)
    tr.observation_profile_errors = []
    browser_module = SimpleNamespace(READ_STATE=handle.read_state, MARKER=handle.marker_expression)
    model = SimpleNamespace(post_json=object(), field_text=object())
    calls = []

    class Agent:
        def __init__(self, url, goal, **kwargs):
            calls.append("agent")
            assert tr.observation_expression == handle.read_state
            assert callable(tr.observation_profile_sink) and callable(tr.observation_sink)
            self.state = {"goal": goal, "history": [], "status": "done"}
            self.browser = SimpleNamespace(close=lambda: None)
            marker = [1000, url, 0, 0, 1120, 780, "Title", "Text", [], []]
            metadata = {
                "id": m.COMPOSED_PROFILE,
                "visited_nodes": 5,
                "node_limit": 20000,
                "depth_limit": 128,
                "node_limit_reached": False,
                "depth_limit_reached": False,
            }
            snapshot = {"observation_profile": metadata, "text": "Text", "marker": marker}
            for sequence, expression, value in (
                (1, handle.read_state, snapshot),
                (2, handle.marker_expression, marker),
            ):
                event = {
                    "protocol_call": sequence,
                    "session": "session-1",
                    "expression": expression,
                    "response": {"result": {"value": value}},
                }
                tr.observation_profile_sink(event)
                if sequence == 1:
                    tr.observation_sink({k: v for k, v in event.items() if k != "expression"})

        def snapshot(self):
            return copy.deepcopy(self.state)

        def close(self):
            pass

    agent = SimpleNamespace(Agent=Agent, field_text=object())

    @contextmanager
    def loader(upstream, transport, *, observation_runtime):
        assert transport is tr and observation_runtime.manifest_sha256 == handle.manifest_sha256
        assert observation_runtime.read_state == handle.read_state
        calls.append("loader")
        monkeypatch.setitem(sys.modules, "jev_ultrafast.browser", browser_module)
        yield agent, model

    def launch(**kwargs):
        calls.append("chrome")
        return object()

    monkeypatch.setattr(m, "StatusTransport", lambda chrome: tr)
    monkeypatch.setattr(m, "load_upstream", loader)
    monkeypatch.setattr(m.Episode, "install_guards", lambda ep: None)
    monkeypatch.setattr(m.Episode, "execute", lambda ep: {"success": True})
    monkeypatch.setattr(
        m,
        "_finish_public_browser",
        lambda chrome, transport, folder, secrets: m.write(
            folder / "public-request-ledger.json", {"synthetic": True}
        ),
    )
    ep, result = m.run_episode(
        tmp_path,
        t,
        "ordinary",
        plan,
        m.Pages(tmp_path / "pages"),
        native.FakeCompiler(),
        native.FakeEvidence(),
        SimpleNamespace(chromium=SimpleNamespace(launch=launch)),
        UPSTREAM,
        observation_runtime=handle,
    )
    assert calls == ["chrome", "loader", "agent"]
    assert result["success"] and result["observation_profile_event_count"] == 2
    assert result["observation_profile_failure_count"] == 0 and ep.raw.count == 1
    assert [row["expression_kind"] for row in ep.profile_tracker.rows] == ["READ_STATE", "MARKER"]


def test_profile_failure_quarantines_all24_with_both_complete_profile_trees(tmp_path, monkeypatch):
    """Whole publication gate with synthetic outcomes; no browser or model is invoked."""
    goals = native.tasks()
    parent = {"size_bytes": 0, "file_count": 0}
    plan = {
        "schema_id": m.PROFILE_SCHEMA,
        "collection_id": "profile-population",
        "tasks": goals,
        "scope": "Synthetic profile refusal test",
        "observation_profile": binding(),
        "ordered_episode_ids": [
            "profile-population-" + t["goal_id"] + "-" + k
            for t in goals
            for k in ("ordinary", "recovery")
        ],
        "runtime_source_pins": {},
        "upstream": m.upstream_manifest(UPSTREAM),
        "prospective_existing_parent_metadata": parent,
    }

    @contextmanager
    def fake_playwright():
        yield object()

    monkeypatch.setitem(
        sys.modules, "playwright.sync_api", SimpleNamespace(sync_playwright=fake_playwright)
    )
    monkeypatch.setattr(m, "NativeCompiler", native.FakeCompiler)
    monkeypatch.setattr(m, "capture_runtime_sources", lambda *a, **k: {"files": {}})
    monkeypatch.setattr(m, "verify_runtime_sources", lambda *a, **k: {"success": True})
    monkeypatch.setattr(m, "verify_source_copies", lambda *a, **k: {"success": True})
    monkeypatch.setattr(m, "parent_tree_metadata", lambda: parent)
    monkeypatch.setattr(
        m,
        "pair_coverage",
        lambda *a: {
            "required_complete": True,
            "required": {"synthetic": True},
            "observed_position_incomplete": False,
        },
    )
    seen = []

    def fake_episode(bundle, task, kind, *args, observation_runtime):
        assert len(m.inventory(bundle / "observation-profile")) == 16
        assert (
            m.sha(observation_runtime.read_state.encode())
            == binding()["effective_snapshot_js_sha256"]
        )
        folder = bundle / "attempts" / (task["goal_id"] + "-" + kind)
        folder.mkdir()
        journal = m.Journal(folder / "all-native-calls.jsonl")
        failed = len(seen) == 7
        seen.append((task["goal_id"], kind))
        failures = (
            [{"sequence": 0, "kind": "cap", "reason": "Confirmed synthetic limit"}]
            if failed
            else []
        )
        ep = SimpleNamespace(
            id=plan["collection_id"] + "-" + folder.name,
            folder=folder,
            pending=[],
            bootstrap_reached=False,
            journals={"all-native-calls": journal},
        )
        return ep, {
            "episode_id": ep.id,
            "goal_id": task["goal_id"],
            "split": task["split"],
            "trajectory_kind": kind,
            "success": not failed,
            "observation_profile": copy.deepcopy(plan["observation_profile"]),
            "observation_profile_manifest_sha256": binding()["manifest_sha256"],
            "observation_profile_event_count": 1,
            "observation_profile_failure_count": len(failures),
            "observation_profile_failures": failures,
        }

    def forbidden_serialization(*args):
        raise AssertionError("An all24 failed population must never serialize fit rows")

    monkeypatch.setattr(m, "run_episode", fake_episode)
    monkeypatch.setattr(m, "serialize_episode", forbidden_serialization)
    output = tmp_path / "collection"
    with pytest.raises(ValueError, match="Public collection partial"):
        m.collect(output, UPSTREAM, plan)
    receipt = json.loads((output / "receipt.json").read_text())
    assert len(seen) == 24 and len(receipt["outcomes"]) == 24
    assert sum(row["success"] for row in receipt["outcomes"]) == 23
    assert sum(row["observation_profile_failure_count"] for row in receipt["outcomes"]) == 1
    assert receipt["stage"] == "partial" and not receipt["all24_mandatory_pair_gate"]
    assert all(not row["fit_eligible"] for row in receipt["outcomes"])
    for split, count in (("train", 16), ("development", 8)):
        bundle = output / split
        row = json.loads((bundle / "receipt.json").read_text())
        assert (
            row["schema_id"] == m.PROFILE_SCHEMA
            and row["observation_profile_verification"]["success"]
        )
        assert row["fit_eligible_row_count"] == 0 and "native_snapshot_js_sha256" not in row
        assert len(m.inventory(bundle / "observation-profile")) == 16
        assert len(list((bundle / "quarantined").iterdir())) == count
        assert (
            (bundle / "cases.jsonl").read_bytes()
            == (bundle / "wire_requests.jsonl").read_bytes()
            == b""
        )


def test_nonempty_tracker_journal_preserves_exact_bytes_through_serialization(tmp_path):
    """Real tracker and pinned native Agent with a fake Browser, no public run."""
    with native.fake_native_episode(tmp_path) as (ep, _agent, _model):
        enable_profile(ep)
        event(ep)
        event(ep, kind="MARKER")
        assert ep.execute()["success"]
        path = ep.folder / (m.PROFILE_JOURNAL + ".jsonl")
        expected = b"".join(m.encode(row) + b"\n" for row in ep.profile_tracker.rows)
        assert path.read_bytes() == expected
        assert list(json.loads(path.read_bytes().splitlines()[0]))[:2] == ["schema_id", "sequence"]
        m.write(ep.folder / "public-request-ledger.json", {"fake_CPU_ledger": True})
        cases, wires = m.serialize_episode(ep, "episodes/synthetic")
        assert len(cases) == len(wires) == 2
        assert [row["gold"]["operation"] for row in cases] == ["CLICK", "DONE"]
        assert path.read_bytes() == expected
        # Equal parsed dictionaries do not authorize different physical field order.
        rows = ep.profile_tracker.rows
        reordered = [{"sequence": row["sequence"], **row} for row in rows]
        assert reordered == rows
        path.write_bytes(b"".join(m.encode(row) + b"\n" for row in reordered))
        assert path.read_bytes() != expected
        with pytest.raises(ValueError, match="Profile event journal differs"):
            m.serialize_episode(ep, "episodes/synthetic")


@pytest.mark.parametrize("sequence", [True, -1, 0, 2])
def test_profile_journal_rejects_wrong_sequence_without_writing(tmp_path, sequence):
    journal = m.ProfileEventJournal(tmp_path / "events.jsonl")
    with pytest.raises(ValueError, match="sequence differs"):
        journal.append({"schema_id": "synthetic", "sequence": sequence})
    assert journal.path.read_bytes() == b"" and journal.rows == []
