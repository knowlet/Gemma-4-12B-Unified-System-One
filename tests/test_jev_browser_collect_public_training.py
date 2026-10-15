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
        public_observer_errors=[],
        document_responses=[],
        calls=0,
        cdp=lambda *a, **k: None,
    )


def synthetic_page(url, width=58):
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
        "scroll": {"y": 0, "height": 2000},
        "omitted_actions": 0,
        "page_key": [1000, url, 0, 0, 1120, 780, []],
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

            def observe(self, screenshot=False):
                page = synthetic_page(self.url)
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
                return not (reject_mutation and action is not None)

            def act(self, action, page, text=None):
                if not self.fresh(page, action):
                    raise stale("Actual fake mutation guard rejected")
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
        assert all(r["accepted"] for r in ep.journals["transitions"].rows)
        assert ep.raw.count >= 3


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
        t = get_public_training_task("mdn-intersectionobserver-observe")
        p = synthetic_page(t["start_url"] + "#interfaces")
        links = [{"node": 1, "href": t["start_url"] + "#interfaces", "label": "Interfaces"}]
        body = {
            "questions": {"operation": {"criteria": {"CLICK": "Click", "SCROLL_DOWN": "Scroll"}}}
        }
        assert (
            m.scripted_teacher(t, p, body, model, links, {"success": False}, {})["operation"]
            == "SCROLL_DOWN"
        )


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
        if any("accepted" not in row for ep in (ordinary, recovery) for row in ep.pending):
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

                def proof(phase):
                    if phase == "after-tick":
                        ep.evidence.failed_check = "same_native_document"
                    return original(phase)

                ep.proof = proof
                with pytest.raises(ValueError, match="binding failed"):
                    ep.tick()
                assert "accepted" not in ep.pending[-1]
                assert actual_pair_coverage(ep, ep)["required_complete"] is False
                return ep, {
                    "episode_id": ep.id,
                    "goal_id": task["goal_id"],
                    "split": task["split"],
                    "trajectory_kind": kind,
                    "success": False,
                    "error": {"type": "ValueError", "message": "Actual after-tick proof refusal"},
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
