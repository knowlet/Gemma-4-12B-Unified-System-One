"""Synthetic whole-corpus v4 admission; no browser/model/network or fit claims.

The cached CPU lane derives a separate, explicitly synthetic profile-aware copy
of the existing real-native-Agent/fake-Browser 24-episode fixture. Nothing here
rewrites or relabels an actual public collection.
"""

import copy

import pytest
from test_browser_public_training_validation import (
    synthetic_complete_public_collection as synthetic_complete_public_collection,
)

from s1 import browser_public_training_validation as v
from s1 import jev_observation_profile as p
from s1.jev_observation_events import ObservationProfileTracker


def encoded(value):
    return v._encode(value) + b"\n"


def jsonl(rows):
    return b"".join(encoded(row) for row in rows)


def rows(blobs, label):
    return [v._json(line) for line in blobs[label].splitlines()]


def upgrade_pages(value, *, profile_id=None):
    """Only test data: replace fake ['stable'] markers with native-shaped facts."""
    if type(value) is dict:
        for child in list(value.values()):
            upgrade_pages(child, profile_id=profile_id)
        if {"actions", "page_key", "marker", "guards", "scroll"} <= set(value):
            value.update(w=1120, h=780)
            value["observation_profile"] = {
                "id": p.COMPOSED_PROFILE if profile_id is None else profile_id,
                "visited_nodes": 42,
                "node_limit": 20000,
                "depth_limit": 128,
                "node_limit_reached": False,
                "depth_limit_reached": False,
            }
            value["marker"] = [
                *value["page_key"][:6],
                value["title"],
                value["text"],
                [
                    {key: item for key, item in action.items() if key not in {"id", "rect"}}
                    for action in value["actions"]
                    if "node" in action
                ],
                value["page_key"][6],
            ]
            ordered = {
                key: value[key]
                for key in (
                    "observation_profile",
                    "url",
                    "title",
                    "w",
                    "h",
                    "text",
                    "scroll",
                    "actions",
                    "marker",
                    "page_key",
                    "guards",
                    "omitted_actions",
                )
            }
            ordered.update({key: item for key, item in value.items() if key not in ordered})
            if "fingerprint" in ordered:
                fingerprint = ordered.pop("fingerprint")
                ordered["fingerprint"] = fingerprint
            value.clear()
            value.update(ordered)
    elif type(value) is list:
        for child in value:
            upgrade_pages(child, profile_id=profile_id)


def profile_binding(canonical_root, helper_sha, *, profile_id=None):
    bundle = p.build_profile(
        p.read_canonical_sources(canonical_root),
        expected_source_sha256=helper_sha,
        profile=p.COMPOSED_PROFILE if profile_id is None else profile_id,
    )
    manifest = p._encode(bundle.materialized_manifest()) + b"\n"
    return bundle, {
        "profile": p.COMPOSED_PROFILE if profile_id is None else profile_id,
        "helper_source_sha256": helper_sha,
        "manifest_path": "observation-profile/profile-manifest.json",
        "manifest_sha256": v._sha(manifest),
        "manifest_size_bytes": len(manifest),
        "canonical_snapshot_js_sha256": v.NATIVE_SOURCE_SHA256[p.SNAPSHOT],
        "effective_snapshot_js_sha256": v._sha(bundle.read_state.encode()),
        "marker_expression_sha256": v._sha(bundle.marker_expression.encode()),
    }


def episode_events(blobs, prefix, bundle, binding):
    """Source-shaped synthetic protocol chronology, with real installed classifier."""
    raw = rows(blobs, prefix + "observations.jsonl")
    guards = rows(blobs, prefix + "guarded-actions.jsonl")
    calls = rows(blobs, prefix + "all-native-calls.jsonl")
    transitions = rows(blobs, prefix + "transitions.jsonl")
    requests = [row for row in calls if row["phase"] == "request"]
    returned = {row["request_sequence"]: row for row in calls if row["phase"] == "teacher-returned"}
    executed = {
        row["request_sequence"]: row
        for row in transitions
        if row["phase"] != "completed-at-next-native-pre"
    }
    before, after = v._guard_pairs(guards)
    timeline = [(index * 10 + 5, "READ_STATE", row) for index, row in enumerate(raw)]
    for index, request in enumerate(requests):
        execution = executed[request["sequence"]]
        raw_before = (
            1
            if index == 0
            else executed[requests[index - 1]["sequence"]][
                "raw_observation_count_after_native_tick"
            ]
        )
        for guard_sequence in range(
            execution["guard_sequence_start"], execution["guard_sequence_end"]
        ):
            if guard_sequence not in before or before[guard_sequence]["native_method"] != "fresh":
                continue
            guard = before[guard_sequence]
            args = guard["arguments"]
            if len(args) == 2 and args[1]["kind"] in {"click", "select"}:
                continue
            value = copy.deepcopy(args[0]["marker"])
            if after[guard_sequence]["result"] is False:
                value[7] += " synthetic transient stale marker"
            position = (
                raw_before * 10
                if guard_sequence == execution["guard_sequence_start"]
                else returned[request["sequence"]]["execution_raw_observation_start"] * 10 + 1
            )
            timeline.append(
                (
                    position,
                    "MARKER",
                    {
                        "session": guard["session"],
                        "response": {"result": {"value": value}},
                    },
                )
            )
    tracker = ObservationProfileTracker(
        read_state=bundle.read_state,
        marker_expression=bundle.marker_expression,
        manifest_sha256=binding["manifest_sha256"],
        profile=binding["profile"],
    )
    for protocol, (_position, kind, row) in enumerate(
        sorted(timeline, key=lambda item: item[0]), 1
    ):
        if kind == "READ_STATE":
            row["protocol_call"] = protocol
        tracker.capture(
            {
                "protocol_call": protocol,
                "session": row["session"],
                "expression": bundle.read_state
                if kind == "READ_STATE"
                else bundle.marker_expression,
                "response": copy.deepcopy(row["response"]),
            }
        )
    assert not tracker.failed
    blobs[prefix + "observations.jsonl"] = jsonl(raw)
    blobs[prefix + "observation-profile-events.jsonl"] = jsonl(tracker.rows)
    return len(tracker.rows)


def reseal(blobs):
    """Repair physical refs/inventories, preserving semantic corruptions under test."""
    plan = v._json(blobs["collection_plan.json"])
    plan_sha = v._sha(v._encode(plan))
    outer = v._json(blobs["receipt.json"])
    outer["collection_plan_sha256"] = plan_sha
    projection = copy.deepcopy(outer["projected_future_fit"])
    bundle_blobs = [
        blob for label, blob in blobs.items() if label.startswith(("train/", "development/"))
    ]
    projection["both_full_bundle_bytes"] = sum(map(len, bundle_blobs)) * 2 + 1024**2
    projection["both_full_bundle_files"] = len(bundle_blobs) * 2 + 32
    projection["staged_publication_bytes"] = projection["staged_publication_files"] = 0
    projection["projected_complete_tree_bytes"] = (
        projection["parent_metadata"]["size_bytes"]
        + projection["both_full_bundle_bytes"]
        + projection["compiled_rows_upper_estimate_bytes"]
        + 64 * 1024**2
    )
    projection["projected_complete_tree_files"] = (
        projection["parent_metadata"]["file_count"] + projection["both_full_bundle_files"] + 256
    )
    outer["projected_future_fit"] = projection
    for split in ("train", "development"):
        blobs[split + "/collection_plan.json"] = blobs["collection_plan.json"]
        for name in ("cases.jsonl", "wire_requests.jsonl"):
            declared = rows(blobs, split + "/" + name)
            for row in declared:
                for ref in row["proof_refs"].values():
                    ref["sha256"] = v._sha(blobs[split + "/" + ref["artifact_path"]])
            blobs[split + "/" + name] = jsonl(declared)
        call_index = rows(blobs, split + "/all_native_calls.jsonl")
        for row in call_index:
            row["sha256"] = v._sha(blobs[split + "/" + row["artifact_path"]])
        blobs[split + "/all_native_calls.jsonl"] = jsonl(call_index)
        receipt = v._json(blobs[split + "/receipt.json"])
        receipt.update(collection_plan_sha256=plan_sha, projected_future_fit=projection)
        receipt["evidence_files"] = {
            label[len(split) + 1 :]: {"sha256": v._sha(blob), "size_bytes": len(blob)}
            for label, blob in sorted(blobs.items())
            if label.startswith(split + "/") and label != split + "/receipt.json"
        }
        for field, name in {
            "cases_sha256": "cases.jsonl",
            "wire_sha256": "wire_requests.jsonl",
            "episode_inventory_sha256": "episodes.jsonl",
            "all_actual_calls_index_sha256": "all_native_calls.jsonl",
        }.items():
            receipt[field] = v._sha(blobs[split + "/" + name])
        blobs[split + "/receipt.json"] = encoded(receipt)
        outer["bundles"][split].update(
            sha256=v._sha(blobs[split + "/receipt.json"]),
            size_bytes=len(blobs[split + "/receipt.json"]),
        )
    blobs["receipt.json"] = encoded(outer)


def publish(destination, blobs):
    destination.mkdir()
    for label, blob in blobs.items():
        path = destination / label
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    return destination


@pytest.fixture(scope="module")
def composed_collection(synthetic_complete_public_collection, tmp_path_factory):
    original, pins, cache = synthetic_complete_public_collection
    snapshot = v.capture_closed_public_snapshot(original)
    blobs = dict(snapshot.blobs)
    for label, blob in list(blobs.items()):
        if label.endswith(".json"):
            value = v._json(blob)
            upgrade_pages(value)
            blobs[label] = encoded(value)
        elif label.endswith(".jsonl"):
            value = rows(blobs, label)
            upgrade_pages(value)
            blobs[label] = jsonl(value)
    bundle, binding = profile_binding(
        original / "train/upstream", pins["s1/jev_observation_profile.py"]["sha256"]
    )
    plan = v._json(blobs["collection_plan.json"])
    plan.update(schema_id=v.COMPOSED_COLLECTION_SCHEMA, observation_profile=binding)
    blobs["collection_plan.json"] = encoded(plan)
    verification = {"success": True, "manifest_sha256": binding["manifest_sha256"], "errors": []}
    outer = v._json(blobs["receipt.json"])
    outer.update(
        schema_id=v.COMPOSED_COLLECTION_SCHEMA,
        observation_profile=binding,
        observation_profile_verification={
            split: verification for split in ("train", "development")
        },
    )
    outcomes = []
    for split in ("train", "development"):
        materialized = tmp_path_factory.mktemp("synthetic-v4-profile") / "profile"
        p.materialize_profile(
            original / split / "upstream",
            materialized,
            expected_source_sha256=binding["helper_source_sha256"],
            profile=p.COMPOSED_PROFILE,
        )
        for path in materialized.rglob("*"):
            if path.is_file():
                blobs[
                    split + "/observation-profile/" + path.relative_to(materialized).as_posix()
                ] = path.read_bytes()
        episodes = rows(blobs, split + "/episodes.jsonl")
        for episode in episodes:
            prefix = split + "/" + episode["artifact_path"] + "/"
            count = episode_events(blobs, prefix, bundle, binding)
            fields = {
                "observation_profile": binding,
                "observation_profile_manifest_sha256": binding["manifest_sha256"],
                "observation_profile_event_count": count,
                "observation_profile_failure_count": 0,
                "observation_profile_failures": [],
            }
            episode.update(fields)
            blobs[prefix + "episode.json"] = encoded(episode)
            attempt = v._json(blobs[prefix + "attempt-result.json"])
            attempt.update(fields)
            blobs[prefix + "attempt-result.json"] = encoded(attempt)
        blobs[split + "/episodes.jsonl"] = jsonl(episodes)
        outcomes.extend(episodes)
        receipt = v._json(blobs[split + "/receipt.json"])
        receipt.pop("native_snapshot_js_sha256")
        receipt.update(
            schema_id=v.COMPOSED_COLLECTION_SCHEMA,
            observation_profile=binding,
            observation_profile_verification=verification,
            canonical_snapshot_js_sha256=binding["canonical_snapshot_js_sha256"],
            effective_snapshot_js_sha256=binding["effective_snapshot_js_sha256"],
        )
        blobs[split + "/receipt.json"] = encoded(receipt)
    outer["outcomes"] = outcomes
    blobs["receipt.json"] = encoded(outer)
    reseal(blobs)
    output = publish(tmp_path_factory.mktemp("synthetic-v4-corpus") / "collection", blobs)
    return output, pins, cache


@pytest.mark.parametrize("schema", [None, "v4", "jev-public-native-training-collection-v2", {}, []])
def test_dispatch_is_explicit_known_schema_only(schema):
    with pytest.raises(v.PublicCollectionError, match="collection-v3 or collection-v4"):
        v._collection_contract(schema)


def test_schema_dispatch_contract_is_frozen_and_does_not_swap_globals():
    from dataclasses import FrozenInstanceError

    canonical = v._collection_contract(v.COLLECTION_SCHEMA)
    composed = v._collection_contract(v.COMPOSED_COLLECTION_SCHEMA)
    assert not canonical.composed and composed.composed
    assert v.COLLECTION_SCHEMA == "jev-public-native-training-collection-v3"
    with pytest.raises(FrozenInstanceError):
        canonical.schema_id = composed.schema_id


def test_complete_synthetic_v4_preserves_native_wires_splits_and_compiler(composed_collection):
    output, pins, cache = composed_collection
    admitted = v.validate_public_training_collection(
        output, reviewed_source_pins=pins, tokenizer_cache=cache
    )
    provenance = admitted.provenance
    assert provenance["collection_schema_id"] == v.COMPOSED_COLLECTION_SCHEMA
    assert provenance["observation_profile"]["profile"] == p.COMPOSED_PROFILE
    assert set(provenance["observation_profile_verification"]) == {"train", "development"}
    assert len(provenance["counts"]["train"]["ordered_episode_ids"]) == 16
    assert len(provenance["counts"]["development"]["ordered_episode_ids"]) == 8
    assert admitted.train_cases and admitted.development_cases
    projected = admitted.compile_wire_projection("train")
    assert any(case.gold["operation"] == "DONE" for case in admitted.train_cases)
    assert all(case.gold["operation"] in {"DONE", "CLICK"} for case in admitted.train_cases)
    assert len(projected) == len(admitted.train_cases)
    detached = admitted.provenance
    detached["observation_profile"]["manifest_sha256"] = "changed"
    assert admitted.provenance == provenance


def modify_episode_fields(blobs, split, episode_id, fields):
    episodes = rows(blobs, split + "/episodes.jsonl")
    episode = next(row for row in episodes if row["episode_id"] == episode_id)
    episode.update(copy.deepcopy(fields))
    prefix = split + "/" + episode["artifact_path"] + "/"
    blobs[prefix + "episode.json"] = encoded(episode)
    attempted = v._json(blobs[prefix + "attempt-result.json"])
    attempted.update(copy.deepcopy(fields))
    blobs[prefix + "attempt-result.json"] = encoded(attempted)
    blobs[split + "/episodes.jsonl"] = jsonl(episodes)
    outer = v._json(blobs["receipt.json"])
    next(row for row in outer["outcomes"] if row["episode_id"] == episode_id).update(
        copy.deepcopy(fields)
    )
    blobs["receipt.json"] = encoded(outer)


@pytest.mark.parametrize(
    "mutation",
    [
        "mixed-schema",
        "wrong-plan-profile",
        "swapped-effective-source",
        "foreign-helper",
        "missing-profile-tree",
        "extra-profile-file",
        "extra-empty-profile-directory",
        "changed-runtime-source",
        "foreign-development-profile",
        "old-canonical-raw",
        "missing-READ_STATE",
        "missing-MARKER",
        "wrong-event-manifest",
        "wrong-event-session",
        "wrong-event-expression",
        "reordered-event",
        "marker-result-disagrees",
        "persistent-cap-then-good",
        "transport-failure-claim",
        "omitted-installed-source",
    ],
)
def test_whole24_v4_rejects_rehashed_profile_or_event_corruption(
    composed_collection, tmp_path, mutation
):
    output, pins, cache = composed_collection
    snapshot = v.capture_closed_public_snapshot(output)
    blobs = dict(snapshot.blobs)
    first = rows(blobs, "train/episodes.jsonl")[0]
    prefix = "train/" + first["artifact_path"] + "/"
    plan = v._json(blobs["collection_plan.json"])
    event_rows = rows(blobs, prefix + "observation-profile-events.jsonl")
    binding = plan["observation_profile"]
    if mutation == "mixed-schema":
        outer = v._json(blobs["receipt.json"])
        outer["schema_id"] = v.COLLECTION_SCHEMA
        blobs["receipt.json"] = encoded(outer)
    elif mutation in {"wrong-plan-profile", "swapped-effective-source", "foreign-helper"}:
        key, value = {
            "wrong-plan-profile": ("profile", p.NATIVE_PROFILE),
            "swapped-effective-source": (
                "effective_snapshot_js_sha256",
                binding["canonical_snapshot_js_sha256"],
            ),
            "foreign-helper": ("helper_source_sha256", "0" * 64),
        }[mutation]
        plan["observation_profile"][key] = value
        blobs["collection_plan.json"] = encoded(plan)
    elif mutation == "missing-profile-tree":
        blobs.pop("train/observation-profile/runtime/jev_ultrafast/snapshot.js")
    elif mutation == "extra-profile-file":
        blobs["train/observation-profile/runtime/jev_ultrafast/unreviewed.py"] = (
            b"raise RuntimeError('never execute')\n"
        )
    elif mutation == "changed-runtime-source":
        label = "train/observation-profile/runtime/jev_ultrafast/model.py"
        blobs[label] += b"\n# changed code remains data\n"
    elif mutation == "foreign-development-profile":
        receipt = v._json(blobs["development/receipt.json"])
        receipt["observation_profile"]["profile"] = p.NATIVE_PROFILE
        blobs["development/receipt.json"] = encoded(receipt)
    elif mutation == "transport-failure-claim":
        modify_episode_fields(
            blobs,
            "train",
            first["episode_id"],
            {
                "observation_profile_failure_count": 1,
                "observation_profile_failures": [{"kind": "integrity", "reason": "sink failed"}],
            },
        )
    elif mutation == "omitted-installed-source":
        pins = {key: value for key, value in pins.items() if key != "s1/jev_observation_events.py"}
        plan["runtime_source_pins"] = pins
        blobs["collection_plan.json"] = encoded(plan)
    elif mutation not in {"extra-empty-profile-directory"}:
        marker = next(row for row in event_rows if row["expression_kind"] == "MARKER")
        read = next(row for row in event_rows if row["expression_kind"] == "READ_STATE")
        if mutation == "old-canonical-raw":
            read["response"]["result"]["value"].pop("observation_profile")
            raw = rows(blobs, prefix + "observations.jsonl")
            next(row for row in raw if row["protocol_call"] == read["protocol_call"])[
                "response"
            ] = copy.deepcopy(read["response"])
            blobs[prefix + "observations.jsonl"] = jsonl(raw)
        elif mutation in {"missing-READ_STATE", "missing-MARKER"}:
            event_rows.remove(read if mutation == "missing-READ_STATE" else marker)
            for index, row in enumerate(event_rows, 1):
                row["sequence"] = index
            modify_episode_fields(
                blobs,
                "train",
                first["episode_id"],
                {"observation_profile_event_count": len(event_rows)},
            )
        elif mutation == "wrong-event-manifest":
            marker["manifest_sha256"] = "0" * 64
        elif mutation == "wrong-event-session":
            marker["session"] = "foreign-owned-session"
        elif mutation == "wrong-event-expression":
            marker["expression_sha256"] = binding["canonical_snapshot_js_sha256"]
        elif mutation == "reordered-event":
            event_rows[0], event_rows[1] = event_rows[1], event_rows[0]
        elif mutation == "marker-result-disagrees":
            marker["response"]["result"]["value"][7] += " changed without stale return"
        elif mutation == "persistent-cap-then-good":
            marker["response"] = {
                "exceptionDetails": {
                    "exception": {
                        "description": 'ComposedVisibleTextLimitError: COMPOSED_VISIBLE_TEXT_LIMIT {"id":"composed-visible-text-v2","visited_nodes":20000,"node_limit":20000,"depth_limit":128,"node_limit_reached":true,"depth_limit_reached":false}'
                    }
                }
            }
            bundle, _ = profile_binding(output / "train/upstream", binding["helper_source_sha256"])
            tracker = ObservationProfileTracker(
                read_state=bundle.read_state,
                marker_expression=bundle.marker_expression,
                manifest_sha256=binding["manifest_sha256"],
            )
            for row in event_rows:
                tracker.capture(
                    {
                        "protocol_call": row["protocol_call"],
                        "session": row["session"],
                        "expression": bundle.read_state
                        if row["expression_kind"] == "READ_STATE"
                        else bundle.marker_expression,
                        "response": row["response"],
                    }
                )
            event_rows = tracker.rows
            assert tracker.failed and event_rows[-1]["profile_failure_count"] > 0
        blobs[prefix + "observation-profile-events.jsonl"] = jsonl(event_rows)
    reseal(blobs)
    copied = publish(tmp_path / mutation, blobs)
    if mutation == "extra-empty-profile-directory":
        (copied / "train/observation-profile/unused-empty").mkdir()
    # Physical ordinary files, complete inventories and outer hashes are repaired.
    closed = v.capture_closed_public_snapshot(copied)
    outer = closed.read_json("receipt.json")
    for split in ("train", "development"):
        receipt = closed.read_json(split + "/receipt.json")
        v._inventory_matches(
            closed, split + "/", receipt["evidence_files"], excluded=("receipt.json",)
        )
        assert v._sha(closed.blobs[split + "/receipt.json"]) == outer["bundles"][split]["sha256"]
    with pytest.raises(v.PublicCollectionError):
        v.validate_public_training_collection(
            copied, reviewed_source_pins=pins, tokenizer_cache=cache
        )
