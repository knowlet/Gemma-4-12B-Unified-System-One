"""Explicit synthetic v5 whole24 native evidence; never public task/model/fit scores."""

import copy
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
import test_browser_public_training_evidence as facts
import test_jev_browser_collect_public_training as native
from test_browser_public_training_profile_validation import encoded, jsonl, publish, reseal, rows

from s1 import browser_public_training_evidence as evidence
from s1 import browser_public_training_validation as v
from s1 import jev_observation_profile as profile

m = native.m


def run_synthetic_episode(
    bundle,
    task,
    kind,
    plan,
    pages,
    compiler,
    _evidence,
    _playwright,
    upstream,
    *,
    observation_runtime,
):
    """Real pinned Agent, Browser.observe/fresh/act; fake CDP/read-only DOM leaves."""
    folder = bundle / "attempts" / (task["goal_id"] + "-" + kind)

    class Evidence:
        validate_public_training_observation = staticmethod(
            evidence.validate_public_training_observation
        )

        @staticmethod
        def observe_public_training(browser, current_task):
            observed, _binding = facts.observation(current_task)
            observed.pop("ownership")
            observed.update(
                url=browser.url,
                document_time_origin=1000,
                scroll_x=0,
                scroll_y=browser.scroll_y,
                viewport={"width": 1120, "height": 780},
            )
            observed["document_read"] = {
                "url_start": browser.url,
                "url_end": browser.url,
                "time_origin_start": 1000,
                "time_origin_end": 1000,
            }
            return observed

    ep = m.Episode(
        bundle,
        folder,
        task,
        kind,
        plan,
        pages,
        compiler,
        Evidence(),
        observation_runtime=observation_runtime,
    )
    transport = native.transport_for(task)
    transport.observation_profile_errors = []
    transport.public_request_blocks = []
    transport.public_navigation_events = []
    ep.transport = transport
    origin = "https://" + urlsplit(task["start_url"]).hostname
    direction = "SCROLL_UP" if plan["tasks"].index(task) % 2 else "SCROLL_DOWN"
    actual_inputs = []
    injected_phases = set()
    with m.load_upstream(upstream, transport, observation_runtime=observation_runtime) as (
        agent,
        model,
    ):
        browser_module = sys.modules["jev_ultrafast.browser"]
        original_operation, original_post, original_teacher = (
            browser_module.browser_operation,
            model.post_json,
            m.scripted_teacher,
        )
        active = []

        class Browser(browser_module.Browser):
            def __init__(self, url):
                self.target, self.session, self.url = "target-1", "session-1", url
                self.scroll_y = 560 if direction == "SCROLL_UP" else 0
                self.text, self.counter = "Synthetic source-shaped public text", 0
                self.refresh_pending, self.scroll_pending, self.done_pending = True, False, True
                active[:] = [self]

            def page(self):
                page = native.synthetic_page(self.url, scroll_y=self.scroll_y)
                page.pop("fingerprint")  # Real Browser.observe appends this after full raw fields.
                page["actions"][-3]["delta"], page["actions"][-2]["delta"] = 560, -560
                page.update(
                    text=self.text,
                    w=1120,
                    h=780,
                    observation_profile={
                        "id": plan["observation_profile"]["profile"],
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
                        {k: value for k, value in action.items() if k not in {"id", "rect"}}
                        for action in page["actions"]
                        if "node" in action
                    ],
                    page["page_key"][6],
                ]
                page["fingerprint"] = browser_module.fingerprint(page)
                return page

            def emit(self, expression, value):
                transport.calls += 1
                event = {
                    "protocol_call": transport.calls,
                    "session": self.session,
                    "expression": expression,
                    "response": {"result": {"value": copy.deepcopy(value)}},
                }
                ep.profile_tracker.capture(event)
                if expression == observation_runtime.read_state:
                    ep.raw_sink({key: value for key, value in event.items() if key != "expression"})

            def evaluate(self, expression):
                if expression == browser_module.MARKER:
                    guard = (
                        ep.journals["guarded-actions"].rows[-1]
                        if ep.journals["guarded-actions"].rows
                        else {}
                    )
                    arguments = guard.get("arguments", [])
                    if self.refresh_pending:
                        self.text += " genuine pre-policy refresh"
                        self.refresh_pending = False
                    if (
                        self.scroll_pending
                        and len(arguments) == 2
                        and arguments[1].get("kind") == "scroll"
                    ):
                        self.text += " newly visible synthetic text"
                        self.scroll_pending = False
                    if (
                        self.done_pending
                        and len(arguments) == 1
                        and ep.pending
                        and ep.pending[-1]["selected"]["operation"] == "DONE"
                        and "transition_sequence" not in ep.pending[-1]
                    ):
                        self.text += " transient completion text"
                        self.done_pending = False
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
                    return (
                        "<html><body>Explicit synthetic CPU document " + self.url + "</body></html>"
                    )
                return [
                    {
                        "node": number,
                        "href": origin
                        + (
                            task["target_path"]
                            if number == 1
                            else task["bootstrap_target_path"]
                            if number == 2
                            else "/unused-" + str(number)
                        ),
                        "label": "Link " + str(number),
                    }
                    for number in range(1, 59)
                ]

            def call(self, *_args, **_kwargs):
                return {"result": {"value": None}}  # Source-native post-input passive leaf.

            def close(self):
                pass

        def operation(request):
            browser = active[0]
            if request["operation"] == "observe":
                page = browser.page()
                browser.emit(
                    observation_runtime.read_state,
                    {
                        key: copy.deepcopy(value)
                        for key, value in page.items()
                        if key != "fingerprint"
                    },
                )
                return page
            assert request["operation"] == "act"
            actual_inputs.append(copy.deepcopy(request))
            action = request["action"]
            if action["kind"] == "scroll":
                browser.scroll_y = max(0, browser.scroll_y + action["delta"])
            else:
                browser.url = origin + (
                    task["target_path"] if action["node"] == 1 else task["bootstrap_target_path"]
                )
                browser.counter += 1
                browser.scroll_y = 560 if direction == "SCROLL_UP" else 0
                transport.public_document_commits.append(
                    {
                        "session": browser.session,
                        "frame_id": "root-1",
                        "source": "cdp_frame_commit",
                        "setup_blank": False,
                        "loader_id": "loader-" + str(browser.counter),
                        "url": browser.url,
                        "scope_violation": None,
                    }
                )
            return {"executed": action["id"]}

        def teacher(*args, **kwargs):
            phase = "bootstrap" if ep.bootstrap else "supervised"
            if phase not in injected_phases:
                injected_phases.add(phase)
                active[0].scroll_pending = True
                return {"operation": direction, "target": None}
            return original_teacher(*args, **kwargs)

        agent.Browser = Browser
        browser_module.browser_operation, model.post_json, m.scripted_teacher = (
            operation,
            ep.post,
            teacher,
        )
        try:
            ep.model = model
            ep.agent = agent.Agent(task["start_url"], task["goal"], screenshots=False)
            ep.install_guards()
            result = {
                "episode_id": ep.id,
                "goal_id": task["goal_id"],
                "split": task["split"],
                "trajectory_kind": kind,
                "fit_eligible": False,
                "scope": "Synthetic CPU native graph only; no real public site or model performance",
            }
            result.update(ep.execute())
            m.write(folder / "final_state.json", ep.agent.snapshot())
        finally:
            browser_module.browser_operation, model.post_json, m.scripted_teacher = (
                original_operation,
                original_post,
                original_teacher,
            )
    # Real Browser.act's first fresh-False branch must never reach the input spy.
    assert all(request["action"]["kind"] == "click" for request in actual_inputs)
    assert len(injected_phases) == (2 if kind == "recovery" else 1)
    m.write(
        folder / "public-request-ledger.json",
        {
            "scope": "Synthetic CPU graph; no network",
            "blocks": [],
            "navigation_events": [],
            "navigation_violations": [],
            "observer_errors": [],
            "document_requests": [],
            "document_commits": transport.public_document_commits,
        },
    )
    m.write(
        folder / "document-response-statuses.json",
        [{"url": task["start_url"], "method": "GET", "status": 200}],
    )
    counts, calls = ep.resolved_counts(), ep.journals["all-native-calls"].rows
    done = [
        row
        for row in calls
        if row["phase"] == "excluded-no-input" and row["rule_id"] == m.NO_INPUT_DONE_RULE["rule_id"]
    ]
    result.update(
        public_scope_violations=[],
        public_observer_errors=[],
        HTTP_document_failures=[],
        cleanup_errors=[],
        browser_protocol_calls=transport.calls,
        elapsed_ms=1.0,
        native_observations=ep.raw.count,
        native_observation_write_ms=ep.raw.write_ms,
        actual_native_policy_calls=sum(counts.values()),
        bootstrap_calls=counts["bootstrap"],
        supervised_calls=counts["supervised"],
        unresolved_native_calls=0,
        actual_call_phase_accounting="Synthetic actual canonical calls; rejected scrolls consume original phase budgets",
        actual_native_requests_by_entry_phase={
            phase: sum(
                row["phase"] == "request" and row["phase_at_entry"] == phase for row in calls
            )
            for phase in ("bootstrap", "supervised")
        },
        scripted_teacher_calls=sum(row["phase"] == "teacher-returned" for row in calls),
        excluded_no_input_done_count=len(done),
        excluded_no_input_native_request_sequences=[row["request_sequence"] for row in done],
        model_inference_calls=0,
        **ep.profile_result_fields(),
        **m.scroll_accounting(plan, calls),
    )
    m.write(folder / "attempt-result.json", result)
    return ep, result


def build_scroll_collection(tmp_path_factory, profile_id):
    pytest.importorskip("tokenizers")
    if (
        not (m.CACHE / "tokenizer.json").exists()
        or not (native.UPSTREAM / "jev_ultrafast/agent.py").exists()
    ):
        pytest.skip("Pinned cached CPU native assets absent")
    patch = pytest.MonkeyPatch()

    @contextmanager
    def fake_playwright():
        yield object()

    patch.setitem(
        sys.modules, "playwright.sync_api", SimpleNamespace(sync_playwright=fake_playwright)
    )
    patch.setattr(m, "run_episode", run_synthetic_episode)
    patch.setattr(
        m,
        "parent_tree_metadata",
        lambda: {
            "root": "/synthetic-parent",
            "file_count": 1,
            "size_bytes": 1,
            "size_inventory_sha256": v._sha(b"synthetic"),
            "scope": "Synthetic size-only projection",
        },
    )
    output = tmp_path_factory.mktemp("synthetic-scroll-whole24") / "collection"
    plan = m.collection_plan(
        output,
        native.UPSTREAM,
        "synthetic-scroll24",
        observation_profile=profile_id,
        observation_profile_helper_sha256=m.record(profile.__file__)["sha256"],
        allow_no_input_stale_scroll=True,
    )
    try:
        m.collect(output, native.UPSTREAM, plan)
    finally:
        patch.undo()
    return output, plan["runtime_source_pins"], m.CACHE


@pytest.fixture(scope="module")
def scroll_collection(tmp_path_factory):
    return build_scroll_collection(tmp_path_factory, profile.COMPOSED_PROFILE)


@pytest.fixture(scope="module")
def pruned_scroll_collection(tmp_path_factory):
    return build_scroll_collection(tmp_path_factory, profile.PRUNED_PROFILE)


@pytest.mark.parametrize("collection_fixture", ["scroll_collection", "pruned_scroll_collection"])
def test_v5_complete_actual_native_synthetic_graph_keeps_separate_exclusions(
    request, collection_fixture
):
    output, pins, cache = request.getfixturevalue(collection_fixture)
    admitted = v.validate_public_training_collection(
        output, reviewed_source_pins=pins, tokenizer_cache=cache
    )
    provenance = admitted.provenance
    assert provenance["collection_schema_id"] == v.SCROLL_COLLECTION_SCHEMA
    assert provenance["observation_profile"]["profile"] == (
        profile.PRUNED_PROFILE
        if collection_fixture == "pruned_scroll_collection"
        else profile.COMPOSED_PROFILE
    )
    assert len(admitted.train_cases) == 32 and len(admitted.development_cases) == 16
    assert sum(map(len, provenance["excluded_no_input_scroll_mapping"].values())) == 36
    assert sum(map(len, provenance["excluded_no_input_done_mapping"].values())) == 24
    all_calls = [row for split in provenance["all_native_call_mapping"].values() for row in split]
    assert len(all_calls) == 120
    assert sum(row["disposition"] == "excluded-no-input-SCROLL" for row in all_calls) == 36
    assert {
        row["resolved_phase"]
        for row in all_calls
        if row["disposition"] == "excluded-no-input-SCROLL"
    } == {"bootstrap", "supervised"}
    assert all(
        case.gold["operation"] in {"CLICK", "DONE"}
        for case in (*admitted.train_cases, *admitted.development_cases)
    )
    assert all(
        len(case.request.candidate_labels["click_target"]) == 58 for case in admitted.train_cases
    )
    assert (
        sum(
            len(provenance["accepted_native_call_mapping"][split])
            for split in ("train", "development")
        )
        == 48
    )
    detached = admitted.provenance
    detached["excluded_no_input_scroll_mapping"]["train"].clear()
    assert admitted.provenance == provenance


def test_excluded_scroll_cannot_supply_pair_incomplete_coverage():
    incomplete = {
        "bootstrap": False,
        "selected": {"operation": "SCROLL_DOWN"},
        "pre": {"success": False},
        "excluded": True,
    }
    done = {
        "bootstrap": False,
        "selected": {"operation": "DONE"},
        "pre": {"success": True},
        "post": {"success": True},
    }
    audit = {"proofs": {0: incomplete, 1: done}, "counts": {"bootstrap": 1}, "cases": []}
    with pytest.raises(v.PublicCollectionError, match="coverage"):
        v._coverage({"goal_id": "synthetic", "split": "train"}, audit, audit)


def change_episode_fields(blobs, episode, fields):
    split = episode["split"]
    episode.update(copy.deepcopy(fields))
    prefix = split + "/" + episode["artifact_path"] + "/"
    blobs[prefix + "episode.json"] = encoded(episode)
    attempt = v._json(blobs[prefix + "attempt-result.json"])
    attempt.update(copy.deepcopy(fields))
    blobs[prefix + "attempt-result.json"] = encoded(attempt)
    episodes = rows(blobs, split + "/episodes.jsonl")
    next(row for row in episodes if row["episode_id"] == episode["episode_id"]).update(
        copy.deepcopy(fields)
    )
    blobs[split + "/episodes.jsonl"] = jsonl(episodes)
    outer = v._json(blobs["receipt.json"])
    next(row for row in outer["outcomes"] if row["episode_id"] == episode["episode_id"]).update(
        copy.deepcopy(fields)
    )
    blobs["receipt.json"] = encoded(outer)


@pytest.mark.parametrize(
    "mutation",
    [
        "default-v4",
        "wrong-rule",
        "non-scroll",
        "targeted-scroll",
        "fresh-true",
        "fresh-exception",
        "act-accepted",
        "act-text-kwargs",
        "extra-guard",
        "nontext-marker",
        "no-newraw",
        "changed-history",
        "bad-after-status",
        "foreign-session",
        "sameURL-newloader",
        "duplicate-exclusion",
        "missing-exclusion",
        "done-scroll-count-conflated",
        "smuggled-fit-row",
        "profile-failure",
        "budget-extension",
    ],
)
def test_whole24_rehashed_v5_corruptions_refuse(scroll_collection, tmp_path, mutation):
    output, pins, cache = scroll_collection
    blobs = dict(v.capture_closed_public_snapshot(output).blobs)
    episode = rows(blobs, "train/episodes.jsonl")[0]
    prefix = "train/" + episode["artifact_path"] + "/"
    transitions = rows(blobs, prefix + "transitions.jsonl")
    execution = next(row for row in transitions if row["phase"] == "excluded-no-input-SCROLL")
    calls = rows(blobs, prefix + "all-native-calls.jsonl")
    event = next(
        row
        for row in calls
        if row["phase"] == "excluded-no-input"
        and row["rule_id"] == v.NO_INPUT_SCROLL_RULE["rule_id"]
    )
    ret = next(
        row
        for row in calls
        if row["phase"] == "teacher-returned"
        and row["request_sequence"] == event["request_sequence"]
    )
    guards = rows(blobs, prefix + "guarded-actions.jsonl")
    start = execution["execution_guard_sequence_start"]
    profile_events = rows(blobs, prefix + "observation-profile-events.jsonl")
    marker = next(
        row
        for row in profile_events
        if row["sequence"] == event["evidence"]["marker_event_sequence"]
    )
    states = rows(blobs, prefix + "states.jsonl")
    before, after = (
        states[execution[key]]["state"] for key in ("before_state_sequence", "after_state_sequence")
    )
    if mutation == "default-v4":
        for label in (
            "collection_plan.json",
            "receipt.json",
            "train/receipt.json",
            "development/receipt.json",
        ):
            value = v._json(blobs[label])
            value["schema_id"] = v.COMPOSED_COLLECTION_SCHEMA
            blobs[label] = encoded(value)
    elif mutation == "wrong-rule":
        event["rule_id"] = "invented-scroll-rule"
    elif mutation in {"non-scroll", "targeted-scroll"}:
        selected = {
            "operation": "WAIT" if mutation == "non-scroll" else "SCROLL_DOWN",
            "target": None if mutation == "non-scroll" else "1",
        }
        ret["selected"] = execution["selected"] = selected
    elif mutation == "fresh-true":
        guards[start + 2].update(accepted=True, result=True)
        marker["response"]["result"]["value"] = copy.deepcopy(
            guards[start + 1]["arguments"][0]["marker"]
        )
    elif mutation == "fresh-exception":
        guards[start + 2].pop("result")
        guards[start + 2]["error"] = {
            "type": "StalePage",
            "message": "Document changed during evaluation",
            "notes": [],
        }
        marker["response"] = {
            "result": {"type": "object", "subtype": "error", "description": "Context changed"},
            "exceptionDetails": {"text": "Uncaught"},
        }
    elif mutation == "act-accepted":
        guards[start + 3].pop("error")
        guards[start + 3].update(accepted=True, result={"executed": "scroll_down"})
    elif mutation == "act-text-kwargs":
        guards[start]["keywords"] = {"text": "invented helper value"}
    elif mutation == "extra-guard":
        extra = copy.deepcopy(guards[start : start + 2])
        for index, row in enumerate(extra):
            row["sequence"] = len(guards) + index
        guards.extend(extra)
    elif mutation == "nontext-marker":
        marker["response"]["result"]["value"][6] += " changed title"
    elif mutation == "no-newraw":
        execution["raw_observation_count_after_native_tick"] = execution[
            "execution_raw_observation_start"
        ]
        event["evidence"]["raw_observation_count_after_native_tick"] = execution[
            "raw_observation_count_after_native_tick"
        ]
    elif mutation == "changed-history":
        after["history"].append({"invented": "input"})
    elif mutation == "bad-after-status":
        after["status"] = "predicted"
    elif mutation == "foreign-session":
        guards[start]["session"] = "foreign"
    elif mutation == "sameURL-newloader":
        ledger = v._json(blobs[prefix + "public-request-ledger.json"])
        ledger["document_commits"].insert(
            1, {**copy.deepcopy(ledger["document_commits"][0]), "loader_id": "new-sameURL-loader"}
        )
        owners, private = (
            rows(blobs, prefix + "ownership.jsonl"),
            rows(blobs, prefix + "private-dom.jsonl"),
        )
        for index in range(execution["immediate_private_diagnostic_sequence"], len(owners)):
            owners[index]["document_commit_count"] += 1
            if owners[index]["document_commit_count"] == 2:
                owners[index]["binding"]["loader_id"] = "new-sameURL-loader"
                private[index]["expected_ownership"]["loader_id"] = "new-sameURL-loader"
                private[index]["observed"]["ownership"]["loader_id"] = "new-sameURL-loader"
        blobs[prefix + "public-request-ledger.json"] = encoded(ledger)
        blobs[prefix + "ownership.jsonl"], blobs[prefix + "private-dom.jsonl"] = (
            jsonl(owners),
            jsonl(private),
        )
    elif mutation == "duplicate-exclusion":
        calls.append({**copy.deepcopy(event), "sequence": len(calls)})
    elif mutation == "missing-exclusion":
        event["phase"] = "teacher-returned"
    elif mutation == "done-scroll-count-conflated":
        change_episode_fields(
            blobs,
            episode,
            {
                "excluded_no_input_scroll_count": 0,
                "excluded_no_input_done_count": episode["excluded_no_input_done_count"] + 1,
            },
        )
    elif mutation == "smuggled-fit-row":
        for name in ("cases.jsonl", "wire_requests.jsonl"):
            declared = rows(blobs, "train/" + name)
            forged = copy.deepcopy(declared[0])
            forged["id" if name == "cases.jsonl" else "case_id"] += "-excluded-scroll"
            forged["wire_row_id"] += "-excluded-scroll"
            if name == "wire_requests.jsonl":
                forged["native_policy_request_sequence"] = event["request_sequence"]
                forged["compiled_input_sequence"] = ret["compiled_sequence"]
            declared.append(forged)
            blobs["train/" + name] = jsonl(declared)
    elif mutation == "profile-failure":
        change_episode_fields(
            blobs,
            episode,
            {
                "observation_profile_failure_count": 1,
                "observation_profile_failures": [
                    {"sequence": 1, "kind": "integrity", "reason": "synthetic sink failure"}
                ],
            },
        )
    else:
        plan = v._json(blobs["collection_plan.json"])
        plan["supervised_decisions_per_episode"] = 17
        blobs["collection_plan.json"] = encoded(plan)
    blobs[prefix + "all-native-calls.jsonl"] = jsonl(calls)
    blobs[prefix + "transitions.jsonl"] = jsonl(transitions)
    blobs[prefix + "guarded-actions.jsonl"] = jsonl(guards)
    blobs[prefix + "observation-profile-events.jsonl"] = jsonl(profile_events)
    blobs[prefix + "states.jsonl"] = jsonl(states)
    reseal(blobs)
    changed = publish(tmp_path / "corrupted-whole24", blobs)
    with pytest.raises(v.PublicCollectionError):
        v.validate_public_training_collection(
            changed, reviewed_source_pins=pins, tokenizer_cache=cache
        )


@pytest.mark.parametrize(
    "forgery", ["v4-with-v3", "v2-name-v3-identity", "v2-tree-v3-binding", "v2-identity-v3-raw"]
)
def test_whole24_crossprofile_forgeries_refuse(
    pruned_scroll_collection, scroll_collection, tmp_path, forgery
):
    output, pins, cache = pruned_scroll_collection
    blobs = dict(v.capture_closed_public_snapshot(output).blobs)
    legacy = v.capture_closed_public_snapshot(scroll_collection[0])
    v2_binding = v._json(legacy.blobs["collection_plan.json"])["observation_profile"]
    if forgery == "v4-with-v3":
        for label in (
            "collection_plan.json",
            "receipt.json",
            "train/receipt.json",
            "development/receipt.json",
        ):
            value = v._json(blobs[label])
            value["schema_id"] = v.COMPOSED_COLLECTION_SCHEMA
            value.pop("no_input_scroll_recovery", None)
            value.pop("excluded_no_input_scroll_count", None)
            blobs[label] = encoded(value)
    elif forgery == "v2-name-v3-identity":
        for label in ("collection_plan.json", "receipt.json"):
            value = v._json(blobs[label])
            value["observation_profile"]["profile"] = profile.COMPOSED_PROFILE
            blobs[label] = encoded(value)
    if forgery in {"v2-tree-v3-binding", "v2-identity-v3-raw"}:
        for split in ("train", "development"):
            prefix = split + "/observation-profile/"
            for label in list(blobs):
                if label.startswith(prefix):
                    del blobs[label]
            blobs.update(
                {label: blob for label, blob in legacy.blobs.items() if label.startswith(prefix)}
            )
    if forgery == "v2-identity-v3-raw":
        verification = {
            "success": True,
            "manifest_sha256": v2_binding["manifest_sha256"],
            "errors": [],
        }
        for label in ("collection_plan.json", "receipt.json"):
            value = v._json(blobs[label])
            value["observation_profile"] = copy.deepcopy(v2_binding)
            if label == "receipt.json":
                value["observation_profile_verification"] = {
                    split: verification for split in ("train", "development")
                }
            blobs[label] = encoded(value)
        for split in ("train", "development"):
            receipt = v._json(blobs[split + "/receipt.json"])
            receipt.update(
                observation_profile=copy.deepcopy(v2_binding),
                observation_profile_verification=verification,
                effective_snapshot_js_sha256=v2_binding["effective_snapshot_js_sha256"],
            )
            blobs[split + "/receipt.json"] = encoded(receipt)
            for episode in rows(blobs, split + "/episodes.jsonl"):
                change_episode_fields(
                    blobs,
                    episode,
                    {
                        "observation_profile": copy.deepcopy(v2_binding),
                        "observation_profile_manifest_sha256": v2_binding["manifest_sha256"],
                    },
                )
                journal = (
                    split + "/" + episode["artifact_path"] + "/observation-profile-events.jsonl"
                )
                events = rows(blobs, journal)
                for event in events:
                    event["manifest_sha256"] = v2_binding["manifest_sha256"]
                    event["expression_sha256"] = (
                        v2_binding["effective_snapshot_js_sha256"]
                        if event["expression_kind"] == "READ_STATE"
                        else v2_binding["marker_expression_sha256"]
                    )
                blobs[journal] = jsonl(events)
    reseal(blobs)
    changed = publish(tmp_path / ("forged-" + forgery), blobs)
    with pytest.raises(v.PublicCollectionError, match="profile|composed|classification"):
        v.validate_public_training_collection(
            changed, reviewed_source_pins=pins, tokenizer_cache=cache
        )
