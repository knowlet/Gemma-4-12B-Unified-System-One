"""Admission primitives use synthetic CPU evidence, never claim a real collection."""

import copy
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from s1 import browser_public_training_validation as v
from s1.browser_public_training import PROOF_FIELDS, public_training_case


def payload(width=400):
    return {
        "model": "generated-actions-none",
        "state": {
            "page": {
                "url": "https://example.invalid/reference",
                "title": "Index",
                "text": "Reference",
            },
            "elements": [],
            "recent_actions": [],
        },
        "questions": {
            "operation": {
                "type": "choice",
                "criteria": {"CLICK": "Click", "DONE": "Done", "BLOCKED": "Blocked"},
                "instructions": {"goal": "Open the reference"},
            },
            "click_target": {
                "type": "choice",
                "criteria": {
                    str(i): {"element": "Reference " + str(i)} for i in range(1, width + 1)
                },
                "instructions": {"goal": "Open the reference", "operation": "CLICK"},
            },
        },
    }


def case_and_wire(operation="DONE"):
    native = payload()
    selected = {"operation": operation, "target": "399" if operation == "CLICK" else None}
    refs = {
        key: {"artifact_path": "episodes/cpu/" + key + ".jsonl", "sha256": "a" * 64, "sequence": 0}
        for key in PROOF_FIELDS
    }
    case = public_training_case(
        native,
        case_id="cpu-case",
        group_id="public-native-site:cpu-site",
        task_id="cpu-task",
        split="train",
        selected=selected,
        valid_targets={operation: [selected["target"]] if selected["target"] else []},
        wire_row_id="cpu-wire",
        episode_id="cpu-episode",
        pair_id="cpu-pair",
        state_role="ordinary_complete" if operation == "DONE" else "ordinary_incomplete",
        proof_refs=refs,
    )
    wire = {
        "case_id": case.id,
        "wire_row_id": case.wire_row_id,
        "episode_id": case.episode_id,
        "pair_id": case.pair_id,
        "native_payload": native,
        "native_wire_sha256": hashlib.sha256(v._encode(native)).hexdigest(),
        "teacher_choice": selected,
        "valid_next_targets": copy.deepcopy(case.valid_next_targets),
        "proof_refs": refs,
    }
    return case, wire


def test_closed_snapshot_seals_every_byte_and_detaches_JSON(tmp_path):
    (tmp_path / "data.json").write_text('{"scope":"synthetic"}\n')
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/raw.bin").write_bytes(b"native bytes")
    snapshot = v.capture_closed_public_snapshot(tmp_path)
    assert list(snapshot.blobs) == ["data.json", "nested/raw.bin"]
    with pytest.raises(TypeError):
        snapshot.blobs["added"] = b"changed"
    snapshot.read_json("data.json")["scope"] = "changed"
    manifest = snapshot.manifest()
    manifest["files"].clear()
    assert snapshot.read_json("data.json") == {"scope": "synthetic"}
    assert len(snapshot.manifest()["files"]) == 2 and snapshot.verify_unchanged()
    (tmp_path / "added.json").write_text("{}")
    with pytest.raises(v.PublicCollectionError, match="changed after"):
        snapshot.verify_unchanged()


@pytest.mark.parametrize("number", ["1e999", "-1e999", "1.7976931348623159e308"])
@pytest.mark.parametrize("container", ["{}", '{"nested":[{}]}'])
def test_finite_JSON_rejects_exponent_overflow_in_any_tree_position(number, container):
    encoded = container.replace("{}", '{"elapsed_ms":' + number + "}")
    with pytest.raises(v.PublicCollectionError, match="finite"):
        v._json(encoded.encode())


def test_finite_JSON_accepts_finite_extremes_and_exponent_text():
    assert v._json(b'{"maximum":1.7976931348623157e308,"small":1e-999,"text":"1e999"}') == {
        "maximum": 1.7976931348623157e308,
        "small": 0.0,
        "text": "1e999",
    }


@pytest.mark.parametrize("kind", ["file", "directory", "fifo", "root"])
def test_nonordinary_or_symlinked_evidence_never_opens_or_passes(tmp_path, kind):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "raw").write_text("proof")
    root = tmp_path / "collection"
    root.mkdir()
    if kind == "fifo":
        os.mkfifo(root / "pipe")
    elif kind == "root":
        root.rmdir()
        root.symlink_to(outside, target_is_directory=True)
    else:
        (root / "link").symlink_to(
            outside / "raw" if kind == "file" else outside, target_is_directory=kind == "directory"
        )
    with pytest.raises(v.PublicCollectionError, match="ordinary|symlink"):
        v.capture_closed_public_snapshot(root)


def test_unchanged_caps_reject_before_reading_oversized_population(tmp_path, monkeypatch):
    assert v.MAX_BYTES == 1_073_741_824 and v.MAX_FILES == 10_000
    (tmp_path / "raw").write_bytes(b"123456")
    monkeypatch.setattr(v, "MAX_BYTES", 5)
    with pytest.raises(v.PublicCollectionError, match="bound"):
        v.capture_closed_public_snapshot(tmp_path)
    monkeypatch.setattr(v, "MAX_BYTES", 100)
    monkeypatch.setattr(v, "MAX_FILES", 1)
    (tmp_path / "second").write_bytes(b"")
    with pytest.raises(v.PublicCollectionError, match="bound"):
        v.capture_closed_public_snapshot(tmp_path)


@pytest.mark.parametrize("blob", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b"\xff"])
def test_JSON_rejects_duplicate_nonfinite_or_nonUTF8(blob):
    with pytest.raises(v.PublicCollectionError):
        v._json(blob)


def test_JSONL_rejects_unsealed_tail_and_foreign_path(tmp_path):
    (tmp_path / "raw.jsonl").write_bytes(b'{"sequence":0}')
    snapshot = v.capture_closed_public_snapshot(tmp_path)
    with pytest.raises(v.PublicCollectionError, match="Unsealed"):
        snapshot.read_jsonl("raw.jsonl")
    with pytest.raises(v.PublicCollectionError, match="confined"):
        snapshot.read_json("../foreign")


@pytest.mark.parametrize("operation", ["CLICK", "DONE"])
def test_compile_projection_retains_full400_heads_and_only_alias_may_change(operation):
    case, wire = case_and_wire(operation)
    original = copy.deepcopy(wire)
    projected = v.public_compile_wire_projection(
        [case], [wire], model_alias="generated-actions-bf16"
    )
    assert wire == original and "native_payload" not in projected[0]
    assert projected[0]["payload"]["model"] == "generated-actions-bf16"
    restored = copy.deepcopy(projected[0]["payload"])
    restored["model"] = original["native_payload"]["model"]
    assert v._encode(restored) == v._encode(original["native_payload"])
    assert len(restored["questions"]["click_target"]["criteria"]) == 400
    assert case.gold == (
        {"operation": "DONE"}
        if operation == "DONE"
        else {"operation": "CLICK", "click_target": "399"}
    )
    projected[0]["payload"]["state"]["page"]["title"] = "changed"
    assert wire == original


@pytest.mark.parametrize("mutation", ["digest", "choice", "target", "order", "case_id", "proof"])
def test_projection_cannot_smuggle_changed_native_bytes_or_annotations(mutation):
    case, wire = case_and_wire()
    if mutation == "digest":
        wire["native_wire_sha256"] = "f" * 64
    elif mutation == "choice":
        wire["teacher_choice"]["operation"] = "CLICK"
    elif mutation == "target":
        wire["teacher_choice"]["target"] = "1"
    elif mutation == "order":
        criteria = wire["native_payload"]["questions"]["click_target"]["criteria"]
        wire["native_payload"]["questions"]["click_target"]["criteria"] = dict(
            reversed(list(criteria.items()))
        )
        wire["native_wire_sha256"] = v._sha(v._encode(wire["native_payload"]))
    elif mutation == "case_id":
        wire["case_id"] = "foreign"
    elif mutation == "proof":
        wire["proof_refs"]["ownership"]["sequence"] = 99
    with pytest.raises(v.PublicCollectionError):
        v.public_compile_wire_projection([case], [wire])


def test_native_ordered_wire_digest_is_distinct_from_canonical_request_fingerprint():
    _case, wire = case_and_wire()
    changed = copy.deepcopy(wire)
    criteria = changed["native_payload"]["questions"]["click_target"]["criteria"]
    changed["native_payload"]["questions"]["click_target"]["criteria"] = dict(
        reversed(list(criteria.items()))
    )
    changed["native_wire_sha256"] = v._sha(v._encode(changed["native_payload"]))
    one, two = v.public_wire_identities([wire]), v.public_wire_identities([changed])
    assert one["canonical_request_fingerprints"] == two["canonical_request_fingerprints"]
    assert one["native_ordered_wire_sha256"] != two["native_ordered_wire_sha256"]


def test_canonical_fingerprint_matches_existing_helper_for_unicode_without_fixture_coercion():
    from s1.evaluation.datasets import request_fingerprint

    _case, wire = case_and_wire()
    wire["native_payload"]["state"]["page"]["title"] = "Référence · 參考"
    wire["native_wire_sha256"] = v._sha(v._encode(wire["native_payload"]))
    identities = v.public_wire_identities([wire])
    assert identities["canonical_request_fingerprints"] == [
        request_fingerprint(v.NativeWireRequest(wire["native_payload"]))
    ]


@pytest.mark.parametrize(
    "schema,stage",
    [
        ("jev-public-native-training-collection-v1", "complete"),
        ("jev-public-native-training-collection-v2", "complete"),
        (v.COLLECTION_SCHEMA, "partial"),
    ],
)
def test_v1_and_partial_are_not_admitted_even_with_asserted_success(tmp_path, schema, stage):
    (tmp_path / "receipt.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "schema_id": schema,
                "stage": stage,
                "success": True,
                "fit_eligible": True,
            }
        )
    )
    with pytest.raises(v.PublicCollectionError):
        v._outer_contract(v.capture_closed_public_snapshot(tmp_path))


def test_caller_pins_cannot_turn_copied_artifact_code_into_installed_code(tmp_path):
    blobs = {}
    pins = {}
    malicious = b"raise RuntimeError('Collection code must never execute')\n"
    for label in v._INSTALLED_REVIEWED_SOURCES:
        pins[label] = {"sha256": v._sha(malicious), "size_bytes": len(malicious)}
        for split in ("train", "development"):
            blobs[split + "/sources/" + label] = malicious
    snapshot = v.ClosedPublicSnapshot(tmp_path, MappingProxyType(blobs))
    with pytest.raises(v.PublicCollectionError, match="Installed reviewed code differs"):
        v.authenticate_reviewed_public_sources(snapshot, pins)


def test_every_captured_runtime_source_is_anchored_to_installed_reviewed_bytes(tmp_path):
    blobs, pins = {}, {}
    for label, path in v._INSTALLED_REVIEWED_SOURCES.items():
        blob = path.read_bytes()
        pins[label] = {"sha256": v._sha(blob), "size_bytes": len(blob)}
        for split in ("train", "development"):
            blobs[split + "/sources/" + label] = blob
    malicious = b"raise RuntimeError('Never import dataset code')\n"
    pins["s1/api.py"] = {"sha256": v._sha(malicious), "size_bytes": len(malicious)}
    for split in ("train", "development"):
        blobs[split + "/sources/s1/api.py"] = malicious
    snapshot = v.ClosedPublicSnapshot(tmp_path, MappingProxyType(blobs))
    with pytest.raises(v.PublicCollectionError, match="Installed reviewed code differs: s1/api.py"):
        v.authenticate_reviewed_public_sources(snapshot, pins)
    pins["outside/execute.py"] = pins.pop("s1/api.py")
    with pytest.raises(v.PublicCollectionError, match="Unapproved"):
        v.authenticate_reviewed_public_sources(snapshot, pins)


def test_pure_native_builder_keeps_full400_unused_head_without_executing_it():
    page = {
        "url": "https://example.invalid/reference",
        "title": "Index",
        "text": "Reference",
        "actions": [
            {
                "id": "e" + str(i),
                "kind": "click",
                "node": i,
                "role": "link",
                "value": "",
                "label": "Reference " + str(i),
            }
            for i in range(1, 401)
        ],
    }
    page["actions"].append({"id": "scroll_down", "kind": "scroll", "label": "Scroll down"})
    wire = v._native_request(page, "Open the reference", [])
    assert len(wire["questions"]["click_target"]["criteria"]) == 400
    assert list(wire["questions"]["click_target"]["criteria"])[-1] == "400"
    assert wire["state"]["elements"][-1]["index"] == "400"
    assert v.NativeWireRequest(wire).candidate_labels["operation"][-2:] == ["DONE", "BLOCKED"]
    assert wire["state"]["recent_actions"] == []


def test_pure_builder_matches_all_actual_failed_collection_requests_when_present():
    """Reconstruction check only; immutable v1 remains categorically inadmissible."""
    root = (
        Path(__file__).resolve().parents[1]
        / "artifacts/jev-browser/20261015-public-native-training-collection-01"
    )
    if not (root / "receipt.json").exists():
        pytest.skip("Optional immutable actual v1 diagnostic is not in this checkout")
    checked = 0
    for episode in sorted(root.rglob("attempt-result.json")):
        folder = episode.parent
        calls = [
            json.loads(line)
            for line in (folder / "all-native-calls.jsonl").read_text().splitlines()
        ]
        raw = [
            json.loads(line) for line in (folder / "observations.jsonl").read_text().splitlines()
        ]
        before = [
            json.loads(line)["state"]
            for line in (folder / "states.jsonl").read_text().splitlines()
            if json.loads(line)["phase"] == "before-tick"
        ]
        requests = [row for row in calls if row["phase"] == "request"]
        for returned in (row for row in calls if row["phase"] == "teacher-returned"):
            request = calls[returned["request_sequence"]]
            page = raw[returned["raw_observation_sequence"]]["response"]["result"]["value"]
            actual = request["payload"]
            goal = actual["questions"]["operation"]["instructions"]["goal"]
            rebuilt = v._native_request(
                page,
                goal,
                before[request["native_policy_call_index"]]["history"],
                model_alias=actual["model"],
            )
            assert v._encode(rebuilt) == v._encode(actual)
            checked += 1
        assert len(requests) == len(before)
    assert checked == 34


def owned_prefix():
    url = "https://example.invalid/reference"
    ledger = {
        "document_commits": [
            {
                "source": "cdp_frame_commit",
                "session": "session",
                "frame_id": "root",
                "loader_id": loader,
                "url": url,
                "setup_blank": False,
            }
            for loader in ("old-loader", "new-loader")
        ],
        "document_requests": [{}, {}],
    }
    page = {"url": url, "page_key": [123, url, 0, 0, 1120, 780, []]}
    binding = {
        "target_id": "target",
        "session_id": "session",
        "root_frame_id": "root",
        "loader_id": "old-loader",
        "native_page_key": page["page_key"],
        "url": url,
    }
    row = {
        "document_commit_count": 1,
        "document_request_count": 1,
        "owned_target_ids": ["target"],
        "owned_session_ids": ["session"],
        "root_frames": {"session": "root"},
        "binding": binding,
    }
    return ledger, page, row


def test_blocked_nonGET_subrequest_is_retained_but_scope_violation_refuses():
    task = {"allowed_top_level_hosts": ["example.invalid"]}
    ledger = {
        "scope": "synthetic",
        "blocks": [
            {
                "url": "https://analytics.invalid/event",
                "method": "POST",
                "reason": "public task permits only GET and HEAD requests",
            }
        ],
        "navigation_events": [],
        "navigation_violations": [],
        "observer_errors": [],
        "document_requests": [
            {
                "method": "GET",
                "url": "https://example.invalid/",
                "redirected_from": None,
                "scope_violation": None,
            }
        ],
        "document_commits": [
            {
                "source": "cdp_frame_commit",
                "url": "https://example.invalid/",
                "setup_blank": False,
                "scope_violation": None,
            }
        ],
    }
    original = copy.deepcopy(ledger)
    v._check_document_ledger(ledger, task)
    assert ledger == original
    for mutation in ("reason", "GET", "navigation", "document_POST"):
        bad = copy.deepcopy(ledger)
        if mutation == "reason":
            bad["blocks"][0]["reason"] = "public navigation host outside allowlist"
        elif mutation == "GET":
            bad["blocks"][0]["method"] = "GET"
        elif mutation == "navigation":
            bad["navigation_violations"].append({"url": "https://foreign.invalid/"})
        else:
            bad["document_requests"][0]["method"] = "POST"
        with pytest.raises(v.PublicCollectionError):
            v._check_document_ledger(bad, task)


def test_owned_loader_is_rederived_from_exact_proof_prefix_not_final_same_URL_commit():
    ledger, page, row = owned_prefix()
    assert ledger["document_commits"][-1]["loader_id"] == "new-loader"
    assert v._expected_owned_binding(row, page, "session", ledger)["loader_id"] == "old-loader"
    row["binding"]["loader_id"] = "new-loader"
    with pytest.raises(v.PublicCollectionError, match="differs"):
        v._expected_owned_binding(row, page, "session", ledger)


def test_owned_loader_refuses_historical_URL_paired_with_a_new_document_loader():
    ledger, page, row = owned_prefix()
    ledger["document_commits"][-1]["url"] = "https://example.invalid/new-document"
    row.update(document_commit_count=2)
    row["binding"]["loader_id"] = "new-loader"
    with pytest.raises(v.PublicCollectionError, match="latest owned document"):
        v._expected_owned_binding(row, page, "session", ledger)


def test_owned_latest_document_allows_real_fragment_commit_and_ignores_foreign_frames():
    ledger, page, row = owned_prefix()
    base = ledger["document_commits"][-1]["url"]
    fragment = base + "#syntax"
    ledger["document_commits"].extend(
        [
            {
                "source": "cdp_within_document",
                "session": "session",
                "frame_id": "root",
                "loader_id": None,
                "url": fragment,
                "setup_blank": False,
            },
            {
                "source": "cdp_frame_commit",
                "session": "session",
                "frame_id": "iframe",
                "loader_id": "iframe-loader",
                "url": base + "/iframe",
                "setup_blank": False,
            },
        ]
    )
    row["document_commit_count"] = 4
    page["url"] = page["page_key"][1] = fragment
    row["binding"].update(url=fragment, loader_id="new-loader")
    assert v._expected_owned_binding(row, page, "session", ledger)["url"] == fragment
    # A fragment commits a new active URL, so its earlier document URL cannot
    # be reused even when it shares the very same full-document loader.
    page["url"] = page["page_key"][1] = base
    row["binding"]["url"] = base
    with pytest.raises(v.PublicCollectionError, match="latest owned document"):
        v._expected_owned_binding(row, page, "session", ledger)


def test_owned_within_document_cannot_change_the_last_full_document_loader():
    ledger, page, row = owned_prefix()
    ledger["document_commits"].append(
        {
            "source": "cdp_within_document",
            "session": "session",
            "frame_id": "root",
            "loader_id": "foreign-loader",
            "url": page["url"],
            "setup_blank": False,
        }
    )
    row["document_commit_count"] = 3
    row["binding"]["loader_id"] = "new-loader"
    with pytest.raises(v.PublicCollectionError, match="latest owned document"):
        v._expected_owned_binding(row, page, "session", ledger)


@pytest.mark.parametrize(
    "mutation",
    ["prefix", "negative", "request_prefix", "extra_target", "foreign_session", "root", "key"],
)
def test_owned_prefix_rejects_unbound_or_out_of_range_context(mutation):
    ledger, page, row = owned_prefix()
    if mutation == "prefix":
        row["document_commit_count"] = 3
    elif mutation == "negative":
        row["document_commit_count"] = -1
    elif mutation == "request_prefix":
        row["document_request_count"] = 3
    elif mutation == "extra_target":
        row["owned_target_ids"].append("foreign")
    elif mutation == "foreign_session":
        row["owned_session_ids"] = ["foreign"]
    elif mutation == "root":
        row["root_frames"]["session"] = "foreign"
    elif mutation == "key":
        page["page_key"][1] = "https://example.invalid/foreign"
    with pytest.raises(v.PublicCollectionError):
        v._expected_owned_binding(row, page, "session", ledger)


def test_whole_file_reference_rejects_foreign_episode_or_changed_digest(tmp_path):
    (tmp_path / "train/episodes/cpu").mkdir(parents=True)
    blob = b'{"sequence":0,"fact":"synthetic"}\n'
    (tmp_path / "train/episodes/cpu/raw.jsonl").write_bytes(blob)
    snapshot = v.capture_closed_public_snapshot(tmp_path)
    reference = {"artifact_path": "episodes/cpu/raw.jsonl", "sha256": v._sha(blob), "sequence": 0}
    rows = v._sequential(snapshot.read_jsonl("train/episodes/cpu/raw.jsonl"), "raw")
    assert (
        v._checked_reference(snapshot, "train", "episodes/cpu", reference, "raw.jsonl", rows=rows)[
            "fact"
        ]
        == "synthetic"
    )
    for changed in (
        {**reference, "artifact_path": "episodes/foreign/raw.jsonl"},
        {**reference, "sha256": "f" * 64},
        {**reference, "sequence": 1},
    ):
        with pytest.raises(v.PublicCollectionError):
            v._checked_reference(snapshot, "train", "episodes/cpu", changed, "raw.jsonl", rows=rows)


def test_strict_private_recomputes_unsatisfied_facts_not_asserted_success(tmp_path):
    from s1.browser_public_training_evidence import (
        OBSERVATION_SCHEMA,
        validate_public_training_observation,
    )
    from s1.browser_public_training_tasks import tasks

    task = tasks()[0]
    ledger, page, owner = owned_prefix()
    url = task["start_url"]
    page["url"] = owner["binding"]["url"] = url
    page["page_key"][1] = url
    for commit in ledger["document_commits"]:
        commit["url"] = url
    owner["phase"] = "before-native-call-0"
    expected = v._expected_owned_binding(owner, page, "session", ledger)
    raw = {
        "schema_id": OBSERVATION_SCHEMA,
        "url": url,
        "document_title": "Index",
        "document_time_origin": 123,
        "document_read": {
            "url_start": url,
            "url_end": url,
            "time_origin_start": 123,
            "time_origin_end": 123,
        },
        "ready_state": "complete",
        "scroll_x": 0,
        "scroll_y": 0,
        "viewport": {"width": 1120, "height": 780},
        "main": None,
        "headings": [],
        "target": {},
        "blockers": [],
        "ownership": expected,
    }
    html = b"<html><title>Index</title></html>"
    relative = "pages/" + v._sha(html) + ".html"
    proof = {
        "phase": owner["phase"],
        "strict": True,
        "expected_ownership": expected,
        "observed": raw,
        "verdict": validate_public_training_observation(task, raw, expected_ownership=expected),
        "public_scope_violations": [],
        "public_observer_errors": [],
        "HTTP_document_failures": [],
        "HTML": {"artifact_path": relative, "sha256": v._sha(html), "size_bytes": len(html)},
    }
    snapshot = v.ClosedPublicSnapshot(tmp_path, MappingProxyType({"train/" + relative: html}))
    _bound, verdict = v._strict_private(
        snapshot, "train", task, proof, owner, page, "session", ledger
    )
    assert verdict["success"] is False
    proof["verdict"]["success"] = True
    with pytest.raises(v.PublicCollectionError, match="trusted recomputation"):
        v._strict_private(snapshot, "train", task, proof, owner, page, "session", ledger)


def execution_fixture(done=False, pre_refresh=False):
    action = {
        "id": "e1",
        "node": 1,
        "kind": "click",
        "role": "link",
        "value": "",
        "label": "Reference",
    }
    page = {
        "url": "https://example.invalid/reference",
        "title": "Index",
        "text": "Reference",
        "actions": [action],
        "scroll": {"y": 0, "height": 1200},
        "page_key": [123, "https://example.invalid/reference", 0, 0, 1120, 780, []],
        "marker": ["native"],
        "guards": {},
    }
    actual_page = {**copy.deepcopy(page), "fingerprint": v._native_fingerprint(page)}
    rows = []

    def before(method, args, keywords):
        index = len(rows)
        rows.append(
            {
                "sequence": index,
                "phase": "before",
                "native_method": method,
                "target": "target",
                "session": "session",
                "arguments": args,
                "keywords": keywords,
            }
        )
        return index

    def after(method, index, result, accepted=True):
        rows.append(
            {
                "sequence": len(rows),
                "phase": "after",
                "native_method": method,
                "before_sequence": index,
                "accepted": accepted,
                "result": result,
            }
        )

    if pre_refresh:
        index = before("fresh", [copy.deepcopy(actual_page)], {})
        after("fresh", index, False, False)
    start = len(rows)
    if done:
        index = before("fresh", [copy.deepcopy(actual_page)], {})
        after("fresh", index, True)
    else:
        act = before("act", [copy.deepcopy(action), copy.deepcopy(actual_page)], {"text": None})
        fresh = before("fresh", [copy.deepcopy(actual_page), copy.deepcopy(action)], {})
        after("fresh", fresh, True)
        after("act", act, {"executed": "e1"})
    selected = {"operation": "DONE" if done else "CLICK", "target": None if done else "1"}
    return (
        page,
        selected,
        {"target_id": "target", "session_id": "session"},
        {"execution_guard_sequence_start": start, "guard_sequence_end": len(rows)},
        rows,
    )


@pytest.mark.parametrize("fresh", [False, True])
def test_predict_refresh_accepts_same_page_only_with_source_correct_raw_chronology(fresh):
    page = execution_fixture()[0]
    before = {**copy.deepcopy(page), "fingerprint": v._native_fingerprint(page)}
    raw = [{"response": {"result": {"value": copy.deepcopy(page)}}}]
    if not fresh:
        raw.append(copy.deepcopy(raw[0]))
    ret = {"execution_raw_observation_start": len(raw), "raw_observation_sequence": len(raw) - 1}
    # A legitimate false freshness branch may observe byte-identical content;
    # the evidence required is a new native observation, not different text.
    v._check_predict_observation(before, page, ret, raw, 1, fresh)


@pytest.mark.parametrize(
    "mutation", ["false_no_observe", "true_changed_page", "true_extra_observe"]
)
def test_predict_refresh_refuses_impossible_native_branch(mutation):
    page = execution_fixture()[0]
    before = {**copy.deepcopy(page), "fingerprint": v._native_fingerprint(page)}
    if mutation == "true_changed_page":
        before["title"] = "A different native state"
    raw = [{"response": {"result": {"value": copy.deepcopy(page)}}}]
    if mutation == "true_extra_observe":
        raw.append(copy.deepcopy(raw[0]))
    ret = {"execution_raw_observation_start": len(raw), "raw_observation_sequence": len(raw) - 1}
    with pytest.raises(v.PublicCollectionError, match="pre-policy freshness"):
        v._check_predict_observation(before, page, ret, raw, 1, mutation != "false_no_observe")


@pytest.mark.parametrize("done,pre_refresh", [(False, False), (False, True), (True, True)])
def test_guard_join_accepts_current_native_action_and_legitimate_pre_policy_refresh(
    done, pre_refresh
):
    arguments = execution_fixture(done, pre_refresh)
    actual = v._check_execution_guards(*arguments)
    assert actual is None if done else actual["id"] == "e1"


@pytest.mark.parametrize(
    "mutation",
    [
        "fresh_false",
        "foreign_session",
        "changed_page_key",
        "changed_action",
        "duplicate_completion",
        "targetless_nonnull",
    ],
)
def test_guard_join_rejects_stale_foreign_or_different_actual_mutation(mutation):
    page, selected, owner, transition, rows = execution_fixture()
    if mutation == "fresh_false":
        rows[2].update(accepted=False, result=False)
    elif mutation == "foreign_session":
        rows[0]["session"] = "foreign"
    elif mutation == "changed_page_key":
        rows[1]["arguments"][0]["page_key"][3] = 560
    elif mutation == "changed_action":
        rows[0]["arguments"][0]["node"] = 999
    elif mutation == "duplicate_completion":
        rows.append({**rows[-1], "sequence": len(rows)})
        transition["guard_sequence_end"] = len(rows)
    elif mutation == "targetless_nonnull":
        selected = {"operation": "DONE", "target": "1"}
    with pytest.raises(v.PublicCollectionError):
        v._check_execution_guards(page, selected, owner, transition, rows)


def test_all_sealed02_completed_episode_joins_when_present_but_partial_corpus_still_rejected():
    """Actual recorded-fact checks only; sixteen completions do not admit partial02."""
    from s1.browser_public_training_tasks import get_public_training_task

    root = (
        Path(__file__).resolve().parents[1]
        / "artifacts/jev-browser/20261015-public-native-training-collection-02"
    )
    if not (root / "receipt.json").exists():
        pytest.skip("Optional sealed actual v2 pilot is absent")
    snapshot = v.capture_closed_public_snapshot(root)
    with pytest.raises(v.PublicCollectionError, match="partial|collection-v3"):
        v._outer_contract(snapshot)
    judge = Path(v.__file__).with_name("browser_public_training_evidence.py").read_bytes()
    if snapshot.blobs["train/sources/s1/browser_public_training_evidence.py"] != judge:
        pytest.skip("Installed prospective judge differs; old v2 facts are not reinterpreted")
    completed = 0
    accepted = 0
    for label in snapshot.blobs:
        if not label.endswith("/attempt-result.json"):
            continue
        result = snapshot.read_json(label)
        if not result["success"]:
            continue
        completed += 1
        prefix = label.removesuffix("attempt-result.json")
        split = label.split("/")[0]
        task = get_public_training_task(result["goal_id"])
        episode_result = snapshot.read_json(prefix + "episode.json")
        plan = snapshot.read_json("collection_plan.json")
        joined = v._audit_episode(
            snapshot, split, task, result["trajectory_kind"], plan, episode_result
        )
        assert len(joined["cases"]) == result["supervised_calls"]
        assert len(joined["call_index"]) == len(joined["rows"]["all-native-calls"])
        ledger = snapshot.read_json(prefix + "public-request-ledger.json")
        v._check_document_ledger(ledger, task)
        raw = v._sequential(snapshot.read_jsonl(prefix + "observations.jsonl"), "raw")
        private = v._sequential(snapshot.read_jsonl(prefix + "private-dom.jsonl"), "private")
        ownership = v._sequential(snapshot.read_jsonl(prefix + "ownership.jsonl"), "ownership")
        guards = v._sequential(snapshot.read_jsonl(prefix + "guarded-actions.jsonl"), "guards")
        calls = v._sequential(snapshot.read_jsonl(prefix + "all-native-calls.jsonl"), "calls")
        returned = {
            row["request_sequence"]: row for row in calls if row["phase"] == "teacher-returned"
        }
        transitions = v._sequential(
            snapshot.read_jsonl(prefix + "transitions.jsonl"), "transitions"
        )
        for transition in transitions:
            if not transition["accepted"]:
                continue
            returned_row = returned[transition["request_sequence"]]
            pre_index = transition["private_precondition_sequence"]
            pre_raw = raw[returned_row["raw_observation_sequence"]]
            page = pre_raw["response"]["result"]["value"]
            binding, _verdict = v._strict_private(
                snapshot,
                split,
                task,
                private[pre_index],
                ownership[pre_index],
                page,
                pre_raw["session"],
                ledger,
            )
            v._check_execution_guards(page, transition["selected"], binding, transition, guards)
            post_index = transition["private_postcondition_sequence"]
            if transition["selected"]["operation"] == "DONE":
                post_raw = pre_raw
            else:
                next_request = transition["postcondition_native_request_sequence"]
                post_raw = raw[returned[next_request]["raw_observation_sequence"]]
                assert (
                    transition["postcondition_raw_observation_sequence"]
                    == returned[next_request]["raw_observation_sequence"]
                )
            post_binding, verdict = v._strict_private(
                snapshot,
                split,
                task,
                private[post_index],
                ownership[post_index],
                post_raw["response"]["result"]["value"],
                post_raw["session"],
                ledger,
            )
            if transition["selected"]["operation"] == "DONE":
                assert binding == post_binding and verdict["success"]
            accepted += 1
    assert completed == 16 and accepted >= 32


def test_cached_CPU_compiler_rederives_actual_prompt_tokens_full_grammar_when_present():
    pytest.importorskip("tokenizers")
    root = Path(__file__).resolve().parents[1]
    pilot = root / "artifacts/jev-browser/20261015-public-native-training-collection-02"
    cache = (
        root
        / "artifacts/jev-local-hf/hub/models--knowlet--Gemma-4-12B-Unified-System-One/snapshots/a66f836b56605039fe040f330180e336d19b3362"
    )
    if not (pilot / "receipt.json").exists() or not (cache / "tokenizer.json").exists():
        pytest.skip("Optional immutable native wire and pinned cached CPU tokenizer absent")
    folder = pilot / "development/quarantined/rust-stack-heap-ordinary"
    calls = [
        json.loads(line) for line in (folder / "all-native-calls.jsonl").read_text().splitlines()
    ]
    compiled = [
        json.loads(line) for line in (folder / "compiled-inputs.jsonl").read_text().splitlines()
    ]
    native = next(row["payload"] for row in calls if row["phase"] == "request")
    plan = json.loads((pilot / "collection_plan.json").read_text())
    compiler = v._CachedNativeCompiler(cache, plan["tokenizer_metadata"])
    v._compiled_structure(native, compiled[0])
    compiler.check(native, compiled[0])
    for field in ("prompt_sha256", "prompt_token_ids_sha256"):
        bad = copy.deepcopy(compiled[0])
        bad[field] = "0" * 64
        with pytest.raises(v.PublicCollectionError, match="CPU prompt/token"):
            compiler.check(native, bad)
    bad = copy.deepcopy(compiled[0])
    bad["grammar"]["encoded_token_lengths"].pop()
    with pytest.raises(v.PublicCollectionError, match="pruned"):
        v._compiled_structure(native, bad)


@pytest.mark.parametrize("pilot", ["01", "02", "03", "04"])
def test_real_partial_pilots_fail_before_cached_tokenizer_or_any_compile(pilot, monkeypatch):
    root = Path(__file__).resolve().parents[1] / (
        "artifacts/jev-browser/20261015-public-native-training-collection-" + pilot
    )
    if not (root / "receipt.json").exists():
        pytest.skip("Optional immutable partial pilot absent")

    def forbidden(*args, **kwargs):
        pytest.fail("Partial corpus reached CPU compiler")

    monkeypatch.setattr(v, "_CachedNativeCompiler", forbidden)
    with pytest.raises(v.PublicCollectionError, match="partial|collection-v3"):
        v.validate_public_training_collection(
            root, reviewed_source_pins={}, tokenizer_cache="absent"
        )


def test_closed_inventory_never_accepts_omitted_or_orphaned_costs(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "train/raw.jsonl").write_bytes(b'{"sequence":0}\n')
    (tmp_path / "train/cost.json").write_bytes(b'{"blocked_POST":1}\n')
    snapshot = v.capture_closed_public_snapshot(tmp_path)
    expected = {name: v._file_pin(snapshot, "train/" + name) for name in ("raw.jsonl", "cost.json")}
    v._inventory_matches(snapshot, "train/", expected)
    missing = copy.deepcopy(expected)
    del missing["cost.json"]
    with pytest.raises(v.PublicCollectionError, match="omitted"):
        v._inventory_matches(snapshot, "train/", missing)
    changed = copy.deepcopy(expected)
    changed["raw.jsonl"]["sha256"] = "0" * 64
    with pytest.raises(v.PublicCollectionError):
        v._inventory_matches(snapshot, "train/", changed)


def test_passive_journal_checks_final_read_deadline_not_only_stable_count():
    sample = {
        "ready_state": "complete",
        "time_origin": 10,
        "url": "https://example.invalid/",
        "scroll_x": 0,
        "scroll_y": 0,
        "width": 100,
        "height": 100,
    }
    rows = [
        {
            "sequence": i,
            "phase": "sample",
            "sample_index": i,
            "elapsed_ms": 10 + i * 50,
            "observed": copy.deepcopy(sample),
        }
        for i in range(3)
    ]
    rows.append({"sequence": 3, "phase": "settled", "elapsed_ms": 111, "stable_samples": 3})
    v._passive_journal(rows, 1)
    bad = copy.deepcopy(rows)
    bad[-2]["elapsed_ms"] = 2050
    bad[-1]["elapsed_ms"] = 2051
    with pytest.raises(v.PublicCollectionError, match="2s"):
        v._passive_journal(bad, 1)
    with pytest.raises(v.PublicCollectionError, match="orphaned"):
        v._passive_journal(rows, 2)


@pytest.mark.parametrize(
    "message",
    ["Document changed during evaluation", "Document is navigating", "Page did not settle"],
)
def test_passive_preserves_only_exact_source_defined_native_read_errors(message):
    sample = {
        "ready_state": "complete",
        "time_origin": 10,
        "url": "https://example.invalid/",
        "scroll_x": 0,
        "scroll_y": 0,
        "width": 100,
        "height": 100,
    }
    rows = [
        {
            "sequence": 0,
            "phase": "sample-error",
            "sample_index": 0,
            "elapsed_ms": 0,
            "error": {"type": "StalePage", "message": message, "notes": []},
        }
    ]
    rows.extend(
        {
            "sequence": index,
            "phase": "sample",
            "sample_index": index,
            "elapsed_ms": index * 50,
            "observed": copy.deepcopy(sample),
        }
        for index in range(1, 4)
    )
    rows.append({"sequence": 4, "phase": "settled", "elapsed_ms": 151, "stable_samples": 3})
    v._passive_journal(rows, 1)
    rows[0]["error"]["message"] = "Unrecognized private read failure"
    with pytest.raises(v.PublicCollectionError, match="Non-navigation"):
        v._passive_journal(rows, 1)


@pytest.fixture(scope="module")
def synthetic_complete_public_collection(tmp_path_factory):
    """CPU test evidence, NOT a genuine public-site collection or task score.

    Real pinned native Agent/choose, scripted teacher, pure private validator,
    publication and cached Rust compiler run against an explicit fake Browser.
    """
    pytest.importorskip("tokenizers")
    import test_browser_public_training_evidence as facts
    import test_jev_browser_collect_public_training as collector_tests

    from s1 import browser_public_training_evidence as evidence

    m = collector_tests.m
    root = tmp_path_factory.mktemp("synthetic-public-admission")
    cache = m.CACHE
    if (
        not (cache / "tokenizer.json").exists()
        or not (collector_tests.UPSTREAM / "jev_ultrafast/agent.py").exists()
    ):
        pytest.skip("Pinned offline CPU assets absent")
    patch = pytest.MonkeyPatch()
    parent = {
        "root": "/synthetic-test-parent",
        "file_count": 1,
        "size_bytes": 1,
        "size_inventory_sha256": v._sha(v._encode({"synthetic": 1})),
        "scope": "Synthetic file-size projection only, no parent lineage claim",
    }
    patch.setattr(m, "parent_tree_metadata", lambda: copy.deepcopy(parent))

    @contextmanager
    def fake_playwright():
        yield object()

    patch.setitem(
        sys.modules, "playwright.sync_api", SimpleNamespace(sync_playwright=fake_playwright)
    )

    def run_fake(bundle, task, kind, plan, pages, compiler, _evidence, _playwright, upstream):
        folder = bundle / "attempts" / (task["goal_id"] + "-" + kind)

        class Evidence:
            validate_public_training_observation = staticmethod(
                evidence.validate_public_training_observation
            )

            @staticmethod
            def observe_public_training(browser, current_task):
                observed, _unused = facts.observation(current_task)
                observed.pop("ownership")
                observed.update(
                    url=browser.url,
                    document_time_origin=1000,
                    scroll_x=0,
                    scroll_y=0,
                    viewport={"width": 1120, "height": 780},
                )
                observed["document_read"] = {
                    "url_start": browser.url,
                    "url_end": browser.url,
                    "time_origin_start": 1000,
                    "time_origin_end": 1000,
                }
                return observed

        ep = m.Episode(bundle, folder, task, kind, plan, pages, compiler, Evidence())
        transport = collector_tests.transport_for(task)
        transport.public_request_blocks = []
        transport.public_navigation_events = []
        ep.transport = transport
        base = task["start_url"].split("/", 3)[:3]
        origin = "/".join(base)

        class Browser:
            target = "target-1"
            session = "session-1"

            def __init__(self, url):
                self.url, self.counter = url, 0
                self.reject_DONE_once = True

            def observe(self, screenshot=False):
                page = collector_tests.synthetic_page(self.url)
                raw = {
                    key: copy.deepcopy(value) for key, value in page.items() if key != "fingerprint"
                }
                ep.raw_sink(
                    {
                        "protocol_call": len(ep.raw_rows),
                        "session": self.session,
                        "response": {"result": {"value": raw}},
                    }
                )
                return page

            def fresh(self, page, action=None):
                # The real pinned Agent consumes DONE, catches its own StalePage
                # and re-observes without any input or invented native history.
                # Pre-policy freshness runs before the new pending row exists.
                if (
                    action is None
                    and self.reject_DONE_once
                    and ep.pending
                    and ep.pending[-1]["selected"]["operation"] == "DONE"
                    and "transition_sequence" not in ep.pending[-1]
                ):
                    self.reject_DONE_once = False
                    return False
                return page["url"] == self.url

            def act(self, action, page, text=None):
                assert self.fresh(page, action) and text is None and action["kind"] == "click"
                self.url = origin + (
                    task["target_path"] if action["node"] == 1 else task["bootstrap_target_path"]
                )
                self.counter += 1
                transport.public_document_commits.append(
                    {
                        "session": self.session,
                        "frame_id": "root-1",
                        "source": "cdp_frame_commit",
                        "setup_blank": False,
                        "loader_id": "loader-" + str(self.counter),
                        "url": self.url,
                        "scope_violation": None,
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
                        "scroll_y": 0,
                        "width": 1120,
                        "height": 780,
                    }
                if expression == "document.documentElement.outerHTML":
                    return "<html><body>Synthetic CPU document " + self.url + "</body></html>"
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

            def close(self):
                pass

        with m.load_upstream(upstream, transport) as (agent_module, model_module):
            agent_module.Browser = Browser
            old_post = model_module.post_json
            model_module.post_json = ep.post
            try:
                ep.model = model_module
                ep.agent = agent_module.Agent(task["start_url"], task["goal"], screenshots=False)
                ep.install_guards()
                result = {
                    "episode_id": ep.id,
                    "goal_id": task["goal_id"],
                    "split": task["split"],
                    "trajectory_kind": kind,
                    "fit_eligible": False,
                    "scope": "Synthetic CPU evidence only, no real public task completion",
                }
                result.update(ep.execute())
                m.write(folder / "final_state.json", ep.agent.snapshot())
            finally:
                model_module.post_json = old_post
        m.write(
            folder / "public-request-ledger.json",
            {
                "scope": "Synthetic CPU request ledger, no actual network",
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
        counts = ep.resolved_counts()
        calls = ep.journals["all-native-calls"].rows
        result.update(
            public_scope_violations=[],
            public_observer_errors=[],
            HTTP_document_failures=[],
            cleanup_errors=[],
            browser_protocol_calls=0,
            elapsed_ms=1.0,
            native_observations=ep.raw.count,
            native_observation_write_ms=ep.raw.write_ms,
            actual_native_policy_calls=counts["bootstrap"] + counts["supervised"],
            bootstrap_calls=counts["bootstrap"],
            supervised_calls=counts["supervised"],
            unresolved_native_calls=0,
            actual_call_phase_accounting="Synthetic CPU fixture retains all native calls",
            actual_native_requests_by_entry_phase={
                phase: sum(
                    row["phase"] == "request" and row["phase_at_entry"] == phase for row in calls
                )
                for phase in ("bootstrap", "supervised")
            },
            scripted_teacher_calls=sum(row["phase"] == "teacher-returned" for row in calls),
            excluded_no_input_done_count=sum(row["phase"] == "excluded-no-input" for row in calls),
            excluded_no_input_native_request_sequences=[
                row["request_sequence"] for row in calls if row["phase"] == "excluded-no-input"
            ],
            model_inference_calls=0,
        )
        m.write(folder / "attempt-result.json", result)
        return ep, result

    patch.setattr(m, "run_episode", run_fake)
    output = root / "synthetic24"
    plan = m.collection_plan(output, collector_tests.UPSTREAM, "synthetic24")
    try:
        m.collect(output, collector_tests.UPSTREAM, plan)
    finally:
        patch.undo()
    return output, plan["runtime_source_pins"], cache


def test_synthetic24_real_native_Agent_full_graph_and_cached_CPU_admission(
    synthetic_complete_public_collection,
):
    output, pins, cache = synthetic_complete_public_collection
    admitted = v.validate_public_training_collection(
        output, reviewed_source_pins=pins, tokenizer_cache=cache
    )
    assert len(admitted.train_cases) == 32 and len(admitted.development_cases) == 16
    assert len(admitted.provenance["counts"]["train"]["ordered_episode_ids"]) == 16
    assert len(admitted.provenance["counts"]["development"]["ordered_episode_ids"]) == 8
    assert all(
        len(case.request.candidate_labels["click_target"]) == 58
        for case in (*admitted.train_cases, *admitted.development_cases)
    )
    original = admitted.wire_rows("train")
    projected = admitted.compile_wire_projection("train", model_alias="synthetic-compiler-alias")
    assert all(
        "native_payload" not in row and row["payload"]["model"] == "synthetic-compiler-alias"
        for row in projected
    )
    assert admitted.wire_rows("train") == original
    assert admitted.provenance["counts"]["train"]["observed_position_incomplete"] is False
    assert admitted.provenance["counts"]["train"]["missing_observed_only_coverage"]
    provenance = admitted.provenance
    assert provenance["collection_schema_id"] == v.COLLECTION_SCHEMA
    for split, episodes, actual_calls, accepted in (
        ("train", 16, 56, 32),
        ("development", 8, 28, 16),
    ):
        exclusions = provenance["excluded_no_input_done_mapping"][split]
        mapping = provenance["accepted_native_call_mapping"][split]
        assert len(exclusions) == episodes
        assert provenance["counts"][split]["excluded_no_input_done_count"] == episodes
        assert len(mapping) == accepted
        physical = provenance["all_native_call_mapping"][split]
        assert len(physical) == actual_calls
        assert sum(row["disposition"] == "accepted-fit-row" for row in physical) == accepted
        assert sum(row["disposition"] == "excluded-no-input-DONE" for row in physical) == episodes
        assert sum(row["disposition"] == "bootstrap-non-fit" for row in physical) == episodes // 2
        assert all(
            row["resolved_phase"] == "supervised"
            for row in physical
            if row["disposition"] == "excluded-no-input-DONE"
        )
        assert all(
            row["compiled_input_sequence"] == row["native_policy_call_index"] for row in physical
        )
        assert (
            len(
                provenance["all_native_call_wire_identities"][split][
                    "canonical_request_fingerprints"
                ]
            )
            == actual_calls
        )
        excluded_keys = {
            (row["episode_id"], row["native_policy_request_sequence"]) for row in exclusions
        }
        assert not excluded_keys & {
            (row["episode_id"], row["native_policy_request_sequence"]) for row in mapping
        }
        # Dense accepted IDs are separate from physical compiled-call indices:
        # ordinary CLICK/excluded-DONE/DONE; recovery adds one bootstrap CLICK.
        for index in range(0, episodes, 2):
            ordinary = mapping[index * 2 : index * 2 + 2]
            recovery = mapping[index * 2 + 2 : index * 2 + 4]
            assert [row["compiled_input_sequence"] for row in ordinary] == [0, 2]
            assert [row["compiled_input_sequence"] for row in recovery] == [1, 3]
            assert [row["case_id"].rsplit("-step-", 1)[1] for row in ordinary] == ["000", "001"]
            assert [row["case_id"].rsplit("-step-", 1)[1] for row in recovery] == ["000", "001"]


@pytest.mark.parametrize(
    "mutation",
    [
        "history",
        "unused_head",
        "private_in_payload",
        "dropped_return",
        "dropped_boundary",
        "wrong_target",
        "early_postproof",
        "unaccepted",
        "wrong_owned_loader",
        "extra_guard",
        "forged_private_success",
        "cost_count",
        "native_decision",
        "bootstrap_classification",
        "rewritten_between_ticks",
        "excluded_FLAG_only",
        "excluded_rule",
        "excluded_selected",
        "excluded_preproof",
        "excluded_postproof",
        "excluded_history",
        "excluded_status",
        "excluded_guard_exception",
        "excluded_guard_true",
        "excluded_guard_arguments",
        "excluded_guard_act",
        "excluded_no_reobserve",
        "excluded_raw_end",
        "excluded_forged_evidence",
        "excluded_duplicate_event",
        "excluded_orphan_event",
        "excluded_event_after_next",
        "excluded_count",
        "excluded_request_list",
        "excluded_compiled_gap",
    ],
)
def test_synthetic_real_native_episode_join_rejects_semantic_corruption(
    synthetic_complete_public_collection, mutation
):
    from s1.browser_public_training_tasks import get_public_training_task

    output, _pins, _cache = synthetic_complete_public_collection
    snapshot = v.capture_closed_public_snapshot(output)
    plan = snapshot.read_json("collection_plan.json")
    result = copy.deepcopy(snapshot.read_jsonl("train/episodes.jsonl")[1])
    task = get_public_training_task(result["goal_id"])
    prefix = "train/" + result["artifact_path"] + "/"
    blobs = dict(snapshot.blobs)
    calls = snapshot.read_jsonl(prefix + "all-native-calls.jsonl")
    exclusion = next(row for row in calls if row["phase"] == "excluded-no-input")
    excluded_sequence = exclusion["request_sequence"]
    excluded_request = next(row for row in calls if row["sequence"] == excluded_sequence)
    excluded_ret = next(
        row
        for row in calls
        if row["phase"] == "teacher-returned" and row["request_sequence"] == excluded_sequence
    )
    transitions = snapshot.read_jsonl(prefix + "transitions.jsonl")
    excluded_transition = next(
        row for row in transitions if row["request_sequence"] == excluded_sequence
    )

    def alter(name, callback):
        rows = snapshot.read_jsonl(prefix + name + ".jsonl")
        callback(rows)
        blobs[prefix + name + ".jsonl"] = b"".join(v._encode(row) + b"\n" for row in rows)

    if mutation in {"history", "unused_head", "private_in_payload"}:

        def payload_change(rows):
            request = next(row for row in rows if row["phase"] == "request")
            if mutation == "history":
                request["payload"]["state"]["recent_actions"].append({"action": "invented"})
            elif mutation == "unused_head":
                request["payload"]["questions"]["click_target"]["criteria"].pop("58")
            else:
                request["payload"]["state"]["private_goal_profile"] = {"expected": "DONE"}
            request["wire_sha256"] = v._sha(v._encode(request["payload"]))

        alter("all-native-calls", payload_change)
    elif mutation in {
        "dropped_return",
        "dropped_boundary",
        "wrong_target",
        "bootstrap_classification",
    }:

        def call_change(rows):
            if mutation.startswith("dropped"):
                phase = "teacher-returned" if mutation == "dropped_return" else "bootstrap-boundary"
                rows.pop(next(i for i, row in enumerate(rows) if row["phase"] == phase))
                for index, row in enumerate(rows):
                    row["sequence"] = index
            elif mutation == "wrong_target":
                next(row for row in rows if row["phase"] == "teacher-returned")["selected"][
                    "target"
                ] = "58"
            else:
                next(row for row in rows if row["phase"] == "decision-resolved")["bootstrap"] = (
                    False
                )

        alter("all-native-calls", call_change)
    elif mutation in {"early_postproof", "unaccepted"}:

        def transition_change(rows):
            accepted = next(row for row in rows if row.get("accepted"))
            if mutation == "unaccepted":
                accepted["accepted"] = False
            else:
                accepted["private_postcondition_sequence"] = accepted[
                    "immediate_private_diagnostic_sequence"
                ]

        alter("transitions", transition_change)
    elif mutation == "wrong_owned_loader":
        alter("ownership", lambda rows: rows[1]["binding"].update(loader_id="foreign-loader"))
    elif mutation == "extra_guard":

        def extra(rows):
            before = copy.deepcopy(next(row for row in rows if row.get("phase") == "before"))
            before["sequence"] = len(rows)
            rows.append(before)
            rows.append(
                {
                    "sequence": len(rows),
                    "phase": "after",
                    "native_method": "fresh",
                    "before_sequence": before["sequence"],
                    "accepted": True,
                    "result": True,
                }
            )

        alter("guarded-actions", extra)
    elif mutation == "forged_private_success":
        alter("private-dom", lambda rows: rows[1]["verdict"].update(success=True))
    elif mutation == "native_decision":
        alter("states", lambda rows: rows[1]["state"]["decisions"][-1].update(target="58"))
    elif mutation == "rewritten_between_ticks":
        alter("states", lambda rows: rows[2]["state"].update(elapsed_ms=99999))
    elif mutation.startswith("excluded_"):
        if mutation in {"excluded_count", "excluded_request_list"}:
            field = (
                "excluded_no_input_done_count"
                if mutation == "excluded_count"
                else "excluded_no_input_native_request_sequences"
            )
            result[field] = 0 if mutation == "excluded_count" else []
        elif mutation in {
            "excluded_rule",
            "excluded_duplicate_event",
            "excluded_orphan_event",
            "excluded_event_after_next",
        }:

            def change_call(rows):
                row = next(row for row in rows if row["phase"] == "excluded-no-input")
                if mutation == "excluded_rule":
                    row["rule_id"] = "forged-rule"
                elif mutation == "excluded_orphan_event":
                    row["request_sequence"] = 99999
                elif mutation == "excluded_duplicate_event":
                    rows.append({**copy.deepcopy(row), "sequence": len(rows)})
                else:
                    # Preserve dense event sequence while moving the original
                    # exclusion behind the following actual request.
                    old = row["sequence"]
                    following = next(
                        item
                        for item in rows
                        if item["phase"] == "request" and item["sequence"] > old
                    )
                    pos = following["sequence"]
                    rows[old], rows[pos] = rows[pos], rows[old]
                    for index, item in enumerate(rows):
                        item["sequence"] = index

            alter("all-native-calls", change_call)
        elif mutation in {
            "excluded_guard_exception",
            "excluded_guard_true",
            "excluded_guard_arguments",
            "excluded_guard_act",
        }:

            def change_guard(rows):
                start = excluded_transition["execution_guard_sequence_start"]
                if mutation == "excluded_guard_exception":
                    rows[start + 1].pop("result")
                    rows[start + 1]["error"] = {"type": "RuntimeError", "message": "failed"}
                elif mutation == "excluded_guard_true":
                    rows[start + 1].update(result=True, accepted=True)
                elif mutation == "excluded_guard_arguments":
                    rows[start]["arguments"][0]["page_key"][3] = 12345
                else:
                    rows[start]["native_method"] = rows[start + 1]["native_method"] = "act"

            alter("guarded-actions", change_guard)
        elif mutation == "excluded_compiled_gap":

            def wrong_compiled(rows):
                ret = next(
                    row
                    for row in rows
                    if row["phase"] == "teacher-returned"
                    and row["request_sequence"] == excluded_sequence
                )
                ret["compiled_sequence"] -= 1

            alter("all-native-calls", wrong_compiled)
        elif mutation in {"excluded_history", "excluded_status"}:

            def change_state(rows):
                after = rows[excluded_request["native_policy_call_index"] * 2 + 1]["state"]
                if mutation == "excluded_history":
                    after["history"][0]["action"] = "Same length, different action"
                else:
                    after["status"] = "done"

            alter("states", change_state)
        else:

            def change_transition(rows):
                row = next(row for row in rows if row["request_sequence"] == excluded_sequence)
                if mutation == "excluded_FLAG_only":
                    row["phase"] = "terminal-or-rejected"
                elif mutation == "excluded_selected":
                    row["selected"] = {"operation": "CLICK", "target": "1"}
                elif mutation == "excluded_preproof":
                    row["private_precondition_sequence"] = 0
                elif mutation == "excluded_postproof":
                    row["private_postcondition_sequence"] = excluded_ret[
                        "private_precondition_sequence"
                    ]
                elif mutation == "excluded_no_reobserve":
                    row["raw_observation_count_after_native_tick"] = row[
                        "execution_raw_observation_start"
                    ]
                elif mutation == "excluded_raw_end":
                    row["raw_observation_count_after_native_tick"] = 999999
                else:
                    row["exclusion_evidence"]["unchanged_full_history_sha256"] = "f" * 64

            alter("transitions", change_transition)
    else:
        result["actual_native_policy_calls"] = 0
    changed = v.ClosedPublicSnapshot(snapshot.root, MappingProxyType(blobs))
    with pytest.raises(v.PublicCollectionError):
        v._audit_episode(changed, "train", task, "recovery", plan, result)


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("forged_gold", "Public cases/wires"),
        ("dropped_bootstrap_index", "Actual call index"),
        ("orphan_HTML", "Unindexed quarantined"),
        ("changed_source", "Copied reviewed source differs"),
        ("excluded_fit_row", "Public cases/wires"),
        ("excluded_missing_event", "accepted transition or one explicit"),
        ("excluded_forged_rule", "Excluded DONE flag/event"),
        ("excluded_receipt_count", "Split actual population/coverage"),
        ("excluded_outer_count", "Outer excluded no-input"),
    ],
)
def test_whole_admission_rejects_rehashed_population_and_label_corruption(
    synthetic_complete_public_collection, monkeypatch, mutation, reason
):
    output, pins, cache = synthetic_complete_public_collection
    original = v.capture_closed_public_snapshot(output)
    blobs = dict(original.blobs)

    def put(name, value):
        blobs[name] = v._encode(value) + b"\n"

    def put_rows(name, rows):
        blobs[name] = b"".join(v._encode(row) + b"\n" for row in rows)

    if mutation == "forged_gold":
        cases = original.read_jsonl("train/cases.jsonl")
        wires = original.read_jsonl("train/wire_requests.jsonl")
        cases[0]["gold"]["click_target"] = "58"
        cases[0]["valid_next_targets"] = {"CLICK": ["58"]}
        wires[0]["teacher_choice"]["target"] = "58"
        wires[0]["valid_next_targets"] = {"CLICK": ["58"]}
        put_rows("train/cases.jsonl", cases)
        put_rows("train/wire_requests.jsonl", wires)
    elif mutation == "dropped_bootstrap_index":
        rows = original.read_jsonl("train/all_native_calls.jsonl")
        rows.pop(next(i for i, row in enumerate(rows) if row["phase"] == "bootstrap-boundary"))
        put_rows("train/all_native_calls.jsonl", rows)
    elif mutation == "orphan_HTML":
        html = b"<html>Unreferenced synthetic cost/evidence</html>"
        blobs["train/pages/" + v._sha(html) + ".html"] = html
    elif mutation == "excluded_fit_row":
        cases = original.read_jsonl("train/cases.jsonl")
        wires = original.read_jsonl("train/wire_requests.jsonl")
        extra_case, extra_wire = copy.deepcopy(cases[1]), copy.deepcopy(wires[1])
        extra_case["id"] += "-excluded"
        extra_case["wire_row_id"] += "-excluded"
        extra_wire.update(case_id=extra_case["id"], wire_row_id=extra_case["wire_row_id"])
        result = original.read_jsonl("train/episodes.jsonl")[0]
        calls = original.read_jsonl("train/" + result["artifact_path"] + "/all-native-calls.jsonl")
        event = next(row for row in calls if row["phase"] == "excluded-no-input")
        request = next(row for row in calls if row["sequence"] == event["request_sequence"])
        extra_wire.update(
            native_policy_request_sequence=request["sequence"],
            compiled_input_sequence=request["native_policy_call_index"],
        )
        # Same current DONE payload is still not permission to turn a rejected
        # no-input request into another positive row.
        assert v._encode(extra_wire["native_payload"]) == v._encode(request["payload"])
        cases.insert(1, extra_case)
        wires.insert(1, extra_wire)
        put_rows("train/cases.jsonl", cases)
        put_rows("train/wire_requests.jsonl", wires)
    elif mutation in {"excluded_missing_event", "excluded_forged_rule"}:
        result = original.read_jsonl("train/episodes.jsonl")[0]
        label = "train/" + result["artifact_path"] + "/all-native-calls.jsonl"
        calls = original.read_jsonl(label)
        event = next(row for row in calls if row["phase"] == "excluded-no-input")
        if mutation == "excluded_missing_event":
            calls.remove(event)
            old_to_new = {row["sequence"]: index for index, row in enumerate(calls)}
            for index, row in enumerate(calls):
                row["sequence"] = index
                if "request_sequence" in row:
                    row["request_sequence"] = old_to_new[row["request_sequence"]]
            transitions_label = "train/" + result["artifact_path"] + "/transitions.jsonl"
            transitions = original.read_jsonl(transitions_label)
            for row in transitions:
                row["request_sequence"] = old_to_new[row["request_sequence"]]
                if "postcondition_native_request_sequence" in row:
                    row["postcondition_native_request_sequence"] = old_to_new[
                        row["postcondition_native_request_sequence"]
                    ]
            put_rows(transitions_label, transitions)
        else:
            event["rule_id"] = "not-reviewed"
        put_rows(label, calls)
    elif mutation in {"excluded_receipt_count", "excluded_outer_count"}:
        pass  # Deliberately alter these after the ordinary inventory repairs.
    else:
        blobs["train/sources/s1/browser_public_training_evidence.py"] += b"\n# altered\n"
    # Repair all ordinary file-inventory and receipt hashes, so refusal must
    # reach semantic graph/source checks rather than merely detect stale pins.
    outer = original.read_json("receipt.json")
    for split in ("train", "development"):
        receipt = original.read_json(split + "/receipt.json")
        if mutation == "excluded_receipt_count" and split == "train":
            receipt["excluded_no_input_done_count"] = 0
        receipt["evidence_files"] = {
            label[len(split) + 1 :]: {"sha256": v._sha(blob), "size_bytes": len(blob)}
            for label, blob in sorted(blobs.items())
            if label.startswith(split + "/") and label != split + "/receipt.json"
        }
        for field, name in {
            "cases_sha256": "cases.jsonl",
            "wire_sha256": "wire_requests.jsonl",
            "all_actual_calls_index_sha256": "all_native_calls.jsonl",
        }.items():
            receipt[field] = v._sha(blobs[split + "/" + name])
        put(split + "/receipt.json", receipt)
        outer["bundles"][split].update(
            sha256=v._sha(blobs[split + "/receipt.json"]),
            size_bytes=len(blobs[split + "/receipt.json"]),
        )
    put("receipt.json", outer)
    if mutation == "excluded_outer_count":
        outer["excluded_no_input_done_count"] = 0
        put("receipt.json", outer)
    changed = v.ClosedPublicSnapshot(original.root, MappingProxyType(dict(sorted(blobs.items()))))
    monkeypatch.setattr(v, "capture_closed_public_snapshot", lambda _path: changed)

    def closure_unreached(_snapshot):
        pytest.fail("Semantic corruption reached the final unchanged-input closure")

    monkeypatch.setattr(v.ClosedPublicSnapshot, "verify_unchanged", closure_unreached)
    with pytest.raises(v.PublicCollectionError, match=reason):
        v.validate_public_training_collection(
            output, reviewed_source_pins=pins, tokenizer_cache=cache
        )


def _write_rehashed_synthetic_copy(destination, original, blobs, outer):
    """Materialize a complete test-only copy; retain every original artifact.

    The overflow marker lets a negative fixture retain valid ordinary byte
    inventories while intentionally containing invalid finite-JSON evidence.
    It is never used by the collector or production admission code.
    """

    def encoded(value):
        return (v._encode(value) + b"\n").replace(b'"OVERFLOW_TIMING"', b"1e999")

    for split in ("train", "development"):
        for name in ("cases.jsonl", "wire_requests.jsonl"):
            rows = [json.loads(line) for line in blobs[split + "/" + name].splitlines()]
            for row in rows:
                for ref in row["proof_refs"].values():
                    ref["sha256"] = v._sha(blobs[split + "/" + ref["artifact_path"]])
            blobs[split + "/" + name] = b"".join(encoded(row) for row in rows)
        receipt = original.read_json(split + "/receipt.json")
        receipt["evidence_files"] = {
            label[len(split) + 1 :]: {"sha256": v._sha(blob), "size_bytes": len(blob)}
            for label, blob in sorted(blobs.items())
            if label.startswith(split + "/") and label != split + "/receipt.json"
        }
        for field, label in {
            "cases_sha256": "cases.jsonl",
            "wire_sha256": "wire_requests.jsonl",
            "episode_inventory_sha256": "episodes.jsonl",
            "all_actual_calls_index_sha256": "all_native_calls.jsonl",
        }.items():
            receipt[field] = v._sha(blobs[split + "/" + label])
        blobs[split + "/receipt.json"] = encoded(receipt)
        outer["bundles"][split].update(
            sha256=v._sha(blobs[split + "/receipt.json"]),
            size_bytes=len(blobs[split + "/receipt.json"]),
        )
    blobs["receipt.json"] = encoded(outer)
    destination.mkdir()
    for label, blob in blobs.items():
        path = destination / label
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    # These checks run on the actual complete copied tree, not a monkeypatched
    # closure or inventory. The mutation cannot fail merely on stale file pins.
    copied = v.capture_closed_public_snapshot(destination)
    assert set(copied.blobs) == set(original.blobs)
    for split in ("train", "development"):
        receipt = copied.read_json(split + "/receipt.json")
        v._inventory_matches(
            copied, split + "/", receipt["evidence_files"], excluded=("receipt.json",)
        )
        assert v._sha(copied.blobs[split + "/receipt.json"]) == outer["bundles"][split]["sha256"]
    return destination


def test_whole24_rehashed_historical_URL_and_latest_loader_pair_is_refused(
    synthetic_complete_public_collection,
    tmp_path,
):
    output, pins, cache = synthetic_complete_public_collection
    original = v.capture_closed_public_snapshot(output)
    blobs = dict(original.blobs)
    result = original.read_jsonl("train/episodes.jsonl")[0]
    prefix = "train/" + result["artifact_path"] + "/"
    ledger = original.read_json(prefix + "public-request-ledger.json")
    old = ledger["document_commits"][0]
    new = {**old, "url": old["url"] + "/different-document", "loader_id": "phantom-new-loader"}
    ledger["document_commits"].insert(1, new)
    blobs[prefix + "public-request-ledger.json"] = v._encode(ledger) + b"\n"
    owners = original.read_jsonl(prefix + "ownership.jsonl")
    private = original.read_jsonl(prefix + "private-dom.jsonl")
    stale_indices = []
    for owner in owners:
        if owner["document_commit_count"] == 1:
            stale_indices.append(owner["sequence"])
            owner["binding"]["loader_id"] = "phantom-new-loader"
            proof = private[owner["sequence"]]
            proof["expected_ownership"]["loader_id"] = "phantom-new-loader"
            proof["observed"]["ownership"]["loader_id"] = "phantom-new-loader"
        owner["document_commit_count"] += 1
    assert stale_indices
    for name, rows in (("ownership", owners), ("private-dom", private)):
        blobs[prefix + name + ".jsonl"] = b"".join(v._encode(row) + b"\n" for row in rows)
    # Recompute every affected whole-journal proof reference, in both typed
    # cases and untouched-source native-wire annotations.
    for name in ("cases.jsonl", "wire_requests.jsonl"):
        rows = original.read_jsonl("train/" + name)
        for row in rows:
            for ref in row["proof_refs"].values():
                ref["sha256"] = v._sha(blobs["train/" + ref["artifact_path"]])
        blobs["train/" + name] = b"".join(v._encode(row) + b"\n" for row in rows)
    copied = _write_rehashed_synthetic_copy(
        tmp_path / "stale-document-copy", original, blobs, original.read_json("receipt.json")
    )
    with pytest.raises(v.PublicCollectionError, match="latest owned document"):
        v.validate_public_training_collection(
            copied, reviewed_source_pins=pins, tokenizer_cache=cache
        )


def test_whole24_rehashed_nonfinite_elapsed_cost_is_refused(
    synthetic_complete_public_collection,
    tmp_path,
):
    output, pins, cache = synthetic_complete_public_collection
    original = v.capture_closed_public_snapshot(output)
    blobs = dict(original.blobs)
    results = original.read_jsonl("train/episodes.jsonl")
    result = results[0]
    prefix = "train/" + result["artifact_path"] + "/"

    def cost_blob(value):
        return (v._encode(value) + b"\n").replace(b'"OVERFLOW_TIMING"', b"1e999")

    for name in ("episode.json", "attempt-result.json"):
        value = original.read_json(prefix + name)
        value["elapsed_ms"] = "OVERFLOW_TIMING"
        blobs[prefix + name] = cost_blob(value)
    result["elapsed_ms"] = "OVERFLOW_TIMING"
    blobs["train/episodes.jsonl"] = b"".join(cost_blob(row) for row in results)
    outer = original.read_json("receipt.json")
    outer["outcomes"][0]["elapsed_ms"] = "OVERFLOW_TIMING"
    copied = _write_rehashed_synthetic_copy(tmp_path / "overflow-copy", original, blobs, outer)
    # The declared costs agree under the old permissive float decoder. A
    # whole-corpus gate must reject them rather than ignore an unused cost.
    assert json.loads((copied / "receipt.json").read_bytes())["outcomes"][0]["elapsed_ms"] == float(
        "inf"
    )
    with pytest.raises(v.PublicCollectionError, match="finite"):
        v.validate_public_training_collection(
            copied, reviewed_source_pins=pins, tokenizer_cache=cache
        )
    snapshot = v.capture_closed_public_snapshot(copied)
    for label in (prefix + "episode.json", prefix + "attempt-result.json"):
        with pytest.raises(v.PublicCollectionError, match="finite"):
            snapshot.read_json(label)
    with pytest.raises(v.PublicCollectionError, match="finite"):
        snapshot.read_jsonl("train/episodes.jsonl")


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("predict_false_no_observe", "False pre-policy freshness"),
        ("DONE_changed_page", "DONE mutated its current native page"),
        ("final_changed_snapshot", "Final native DONE"),
    ],
)
def test_whole24_rehashed_impossible_native_flow_is_refused(
    synthetic_complete_public_collection,
    tmp_path,
    mutation,
    reason,
):
    output, pins, cache = synthetic_complete_public_collection
    original = v.capture_closed_public_snapshot(output)
    blobs = dict(original.blobs)
    result = original.read_jsonl("train/episodes.jsonl")[0]
    prefix = "train/" + result["artifact_path"] + "/"
    if mutation == "predict_false_no_observe":
        rows = original.read_jsonl(prefix + "guarded-actions.jsonl")
        assert rows[1]["result"] is True
        rows[1].update(result=False, accepted=False)
        blobs[prefix + "guarded-actions.jsonl"] = b"".join(v._encode(row) + b"\n" for row in rows)
    else:
        final = original.read_json(prefix + "final_state.json")
        if mutation == "DONE_changed_page":
            states = original.read_jsonl(prefix + "states.jsonl")
            initial = states[0]["state"]
            for key in ("page", "elements"):
                final[key] = copy.deepcopy(initial[key])
                states[-1]["state"][key] = copy.deepcopy(initial[key])
            blobs[prefix + "states.jsonl"] = b"".join(v._encode(row) + b"\n" for row in states)
        else:
            final["elapsed_ms"] += 1
        blobs[prefix + "final_state.json"] = v._encode(final) + b"\n"
    copied = _write_rehashed_synthetic_copy(
        tmp_path / mutation, original, blobs, original.read_json("receipt.json")
    )
    with pytest.raises(v.PublicCollectionError, match=reason):
        v.validate_public_training_collection(
            copied, reviewed_source_pins=pins, tokenizer_cache=cache
        )
