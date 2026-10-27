"""Synthetic callable accounting; never imports models or benchmark data."""

import asyncio
import copy
import hashlib
import json

import jev_decision3_head_coverage as coverage
import pytest


@pytest.fixture
def typed():
    return {
        "state": {"sample": [1, 2]},
        "questions": {
            "decision": {"type": "noul", "instructions": "Allowed?"},
            "operation": {
                "type": "choice",
                "instructions": "literal unrelated ID",
                "criteria": {"b": "B", "a": "A"},
            },
            "measure": {"type": "score", "instructions": "Rate", "levels": ["low", "high"]},
            "click_target": {
                "type": "score",
                "instructions": "also a generic ID",
                "criteria": {"1": "one", "2": "two"},
            },
        },
    }


@pytest.fixture
def browser():
    return {
        "model": "synthetic",
        "state": {"page": "original"},
        "questions": {
            "operation": {
                "type": "choice",
                "instructions": "choose",
                "criteria": {"CLICK": "click", "DONE": "done"},
            },
            "type_text_target": {
                "type": "choice",
                "instructions": "text",
                "criteria": {"one": "field"},
            },
            "click_target": {
                "type": "choice",
                "instructions": "target",
                "criteria": {"2": "second", "1": "first"},
            },
        },
    }


def receipt(view, *, operation=True, click=True):
    ids = view["question_ids"]
    return {
        "question_ids": copy.deepcopy(ids),
        "ordered_questions": copy.deepcopy(view["ordered_questions"]),
        "adapted_rows": {
            "operation": [ids.index("operation")] if operation and "operation" in ids else [],
            "click_target": [ids.index("click_target")] if click and "click_target" in ids else [],
        },
    }


def test_heaven_and_coherence_generic_do_not_infer_heads_from_names(typed):
    tracker = coverage.HeadCoverage()
    calls = []
    result = {"answers": {"decision": {"type": "noul", "noul": 0.4}}}

    def predict(state, questions):
        calls.append((state, questions))
        return result

    wrapped = tracker.wrap_sync(predict, route="generic")
    original = copy.deepcopy(typed)
    assert wrapped(typed["state"], typed["questions"]) is result
    assert typed == original and calls == [(original["state"], original["questions"])]
    stats = tracker.snapshot()
    assert stats["generic_bypass_requests"] == stats["generic_bypass_completed"] == 1
    assert stats["requested_heads"] == {"operation": 1, "click_target": 1}
    assert stats["applied_heads"] == {"operation": 0, "click_target": 0}
    assert stats["model_score"] is None
    view = coverage.inspect_request(typed["state"], typed["questions"])
    assert view["question_ids"] == ["decision", "operation", "measure", "click_target"]
    assert view["ordered_questions"][1]["criteria"]["items"] == [["b", "B"], ["a", "A"]]
    assert view["ordered_questions"][2]["criteria"]["items"] == ["low", "high"]


def test_browser_counts_only_actual_evidenced_heads_and_returns_original_result(browser):
    tracker = coverage.HeadCoverage()
    before = copy.deepcopy(browser)
    view = coverage.inspect_packet(browser)
    answer = {"answers": {"operation": {"choice": "CLICK"}}}
    wrapped = tracker.wrap_sync(
        lambda packet: answer,
        route="browser",
        packet=True,
        evidence=lambda _: receipt(view, operation=True, click=False),
    )
    assert wrapped(browser) is answer
    assert browser == before
    assert tracker.snapshot()["applied_heads"] == {"operation": 1, "click_target": 0}
    assert tracker.snapshot()["requested_heads"] == {"operation": 1, "click_target": 1}
    both = tracker.wrap_sync(
        lambda packet: answer, route="browser", packet=True, evidence=lambda _: receipt(view)
    )
    assert both(browser) is answer
    assert tracker.snapshot()["applied_heads"] == {"operation": 2, "click_target": 1}


@pytest.mark.parametrize(
    "kind", ["head_order", "criterion_order", "wrong_row", "wrong_type", "extra", "wrong_keys"]
)
def test_browser_bad_or_misaligned_receipt_is_refused(browser, kind):
    tracker = coverage.HeadCoverage()
    view = coverage.inspect_packet(browser)
    proof = receipt(view)
    if kind == "head_order":
        proof["question_ids"].reverse()
    elif kind == "criterion_order":
        proof["ordered_questions"][2]["criteria"]["items"].reverse()
    elif kind == "wrong_row":
        proof["adapted_rows"]["click_target"] = [1]
    elif kind == "wrong_type":
        proof["adapted_rows"]["operation"] = [False]
    elif kind == "extra":
        proof["adapted_rows"]["other"] = [1]
    else:
        proof["adapted_rows"] = dict(reversed(list(proof["adapted_rows"].items())))
    wrapped = tracker.wrap_sync(
        lambda packet: {"ok": True}, route="browser", packet=True, evidence=lambda _: proof
    )
    with pytest.raises(ValueError):
        wrapped(browser)
    stats = tracker.snapshot()
    assert stats["routes"]["browser"] == {"started": 1, "completed": 0, "failed": 1}
    assert stats["applied_heads"] == {"operation": 0, "click_target": 0}


def test_browser_cannot_adapt_a_score_with_a_matching_name(typed):
    tracker = coverage.HeadCoverage()
    view = coverage.inspect_request(typed["state"], typed["questions"])
    wrapped = tracker.wrap_sync(lambda *_: {}, route="browser", evidence=lambda _: receipt(view))
    with pytest.raises(ValueError, match="Choice head"):
        wrapped(typed["state"], typed["questions"])


def test_packet_json_duplicates_nan_and_ambiguous_criteria_are_rejected(browser):
    tracker = coverage.HeadCoverage()
    wrapped = tracker.wrap_sync(
        lambda packet: pytest.fail("invalid packet reached model"), route="generic", packet=True
    )
    for value in (
        '{"state": {}, "questions": {"x":{"type":"noul","instructions":"a"},"x":{"type":"noul","instructions":"b"}}}',
        '{"state": NaN, "questions": {}}',
        {
            "state": {},
            "questions": {
                "q": {"instructions": "x", "criteria": {"a": "A"}, "options": {"b": "B"}}
            },
        },
        {"state": {1: "integer key"}, "questions": browser["questions"]},
    ):
        with pytest.raises(ValueError):
            wrapped(value)
    assert tracker.snapshot()["rejected_before_prediction"] == 4
    assert tracker.snapshot()["routes"]["generic"]["started"] == 0


def test_model_and_evidence_mutation_are_detected_without_credit(browser):
    tracker = coverage.HeadCoverage()

    def mutate(packet):
        packet["model"] = "other"
        return {}

    wrapped = tracker.wrap_sync(mutate, route="generic", packet=True)
    with pytest.raises(ValueError, match="changed original request"):
        wrapped(browser)
    assert tracker.snapshot()["generic_bypass_requests"] == 1
    assert tracker.snapshot()["generic_bypass_completed"] == 0
    view = coverage.inspect_packet(browser)
    result = {"answers": {"unchanged": True}}

    def evidence(value):
        value["answers"]["unchanged"] = False
        return receipt(view)

    wrapped = tracker.wrap_sync(lambda _: result, route="browser", packet=True, evidence=evidence)
    with pytest.raises(ValueError, match="changed model response"):
        wrapped(browser)
    assert tracker.snapshot()["applied_heads"] == {"operation": 0, "click_target": 0}


def test_async_concurrency_counts_each_call_and_preserves_original_order(browser):
    tracker = coverage.HeadCoverage()

    async def predict(packet):
        await asyncio.sleep(0)
        return {"view": coverage.inspect_packet(packet)}

    async def evidence(result):
        await asyncio.sleep(0)
        return receipt(result["view"], click=False)

    wrapped = tracker.wrap_async(predict, route="browser", evidence=evidence, packet=True)

    async def run():
        packets = [copy.deepcopy(browser) for _ in range(6)]
        results = await asyncio.gather(*(wrapped(packet) for packet in packets))
        assert all(
            result["view"]["question_ids"] == list(browser["questions"]) for result in results
        )
        assert all(packet == browser for packet in packets)

    asyncio.run(run())
    assert tracker.snapshot()["routes"]["browser"] == {"started": 6, "completed": 6, "failed": 0}
    assert tracker.snapshot()["applied_heads"] == {"operation": 6, "click_target": 0}


def test_original_failures_are_recorded_and_propagated(typed):
    tracker = coverage.HeadCoverage()

    def failed(*_):
        raise RuntimeError("predict failed")

    wrapped = tracker.wrap_sync(failed, route="generic")
    with pytest.raises(RuntimeError, match="predict failed"):
        wrapped(typed["state"], typed["questions"])
    assert tracker.snapshot()["routes"]["generic"] == {"started": 1, "completed": 0, "failed": 1}


def test_raw_packet_and_no_delta_browser_evidence(browser):
    tracker = coverage.HeadCoverage()
    body = json.dumps(browser, ensure_ascii=False).encode()
    original = bytes(body)
    view = coverage.inspect_packet(body)
    wrapped = tracker.wrap_sync(
        lambda packet: {"ok": True},
        route="browser",
        packet=True,
        evidence=lambda _: receipt(view, operation=False, click=False),
    )
    assert wrapped(body) == {"ok": True}
    assert body == original
    assert tracker.snapshot()["browser_completed_without_adapted_head"] == 1
    assert tracker.snapshot()["applied_heads"] == {"operation": 0, "click_target": 0}


@pytest.mark.parametrize("as_text", [False, True], ids=["bytes", "utf8-string"])
def test_wire_pin_distinguishes_equivalent_json_encodings(browser, as_text):
    compact = json.dumps(browser, ensure_ascii=False, separators=(",", ":"))
    spaced = json.dumps(browser, ensure_ascii=False, indent=2)
    original = compact if as_text else compact.encode("utf-8")
    changed = spaced if as_text else spaced.encode("utf-8")
    original_view = coverage.inspect_packet(original)
    changed_view = coverage.inspect_packet(changed)
    assert original_view["packet_sha256"] == original_view["packet_canonical_sha256"]
    assert original_view["packet_canonical_sha256"] == changed_view["packet_canonical_sha256"]
    assert (
        original_view["packet_wire_sha256"] == hashlib.sha256(compact.encode("utf-8")).hexdigest()
    )
    assert changed_view["packet_wire_sha256"] == hashlib.sha256(spaced.encode("utf-8")).hexdigest()
    assert original_view["packet_wire_sha256"] != changed_view["packet_wire_sha256"]
    seen = []
    tracker = coverage.HeadCoverage()
    wrapped = tracker.wrap_sync(
        lambda packet: seen.append(packet) or {},
        route="generic",
        packet=True,
        expected_views={original_view["packet_sha256"]: original_view},
    )
    with pytest.raises(ValueError, match="pinned packet"):
        wrapped(changed)
    assert seen == []
    assert wrapped(original) == {}
    assert seen == [original]
    assert tracker.snapshot()["rejected_before_prediction"] == 1


def test_dict_packet_has_canonical_hash_but_no_claimed_wire_hash(browser):
    view = coverage.inspect_packet(browser)
    assert view["packet_sha256"] == view["packet_canonical_sha256"]
    assert view["packet_wire_sha256"] is None
    serialized = json.dumps(browser, ensure_ascii=False, separators=(",", ":"))
    raw_view = coverage.inspect_packet(serialized)
    assert raw_view["packet_canonical_sha256"] == view["packet_canonical_sha256"]
    tracker = coverage.HeadCoverage()
    wrapped = tracker.wrap_sync(
        lambda packet: {},
        route="generic",
        packet=True,
        expected_views={view["packet_sha256"]: view},
    )
    with pytest.raises(ValueError, match="pinned packet"):
        wrapped(serialized)
    assert wrapped(browser) == {}


def test_cyclic_input_refused_and_shared_noncyclic_values_allowed(browser):
    tracker = coverage.HeadCoverage()
    wrapped = tracker.wrap_sync(lambda _: {}, route="generic", packet=True)
    loop = []
    loop.append(loop)
    invalid = copy.deepcopy(browser)
    invalid["state"] = {"recursive": loop}
    with pytest.raises(ValueError, match="cyclic"):
        wrapped(invalid)
    shared = ["same", 3]
    valid = copy.deepcopy(browser)
    valid["state"] = {"first": shared, "second": shared}
    assert wrapped(valid) == {}
    assert tracker.snapshot()["generic_bypass_completed"] == 1


def test_pinned_original_view_rejects_reordered_generic_request_before_predict(browser):
    tracker = coverage.HeadCoverage()
    original_view = coverage.inspect_packet(browser)
    calls = []
    wrapped = tracker.wrap_sync(
        lambda packet: calls.append(packet) or {"answer": "ok"},
        route="generic",
        packet=True,
        expected_views={original_view["packet_sha256"]: original_view},
    )
    reordered = copy.deepcopy(browser)
    reordered["questions"]["click_target"]["criteria"] = dict(
        reversed(list(reordered["questions"]["click_target"]["criteria"].items()))
    )
    with pytest.raises(ValueError, match="pinned packet"):
        wrapped(reordered)
    assert calls == []
    assert tracker.snapshot()["rejected_before_prediction"] == 1
    assert wrapped(browser) == {"answer": "ok"}
    assert tracker.snapshot()["generic_bypass_completed"] == 1


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            {"type": "noul", "instructions": "allowed?", "criteria": None},
            {"field": "criteria", "kind": "null", "items": None},
        ),
        (
            {
                "type": "noul",
                "instructions": "allowed?",
                "criteria": {"true": "yes", "false": "no"},
            },
            {"field": "criteria", "kind": "mapping", "items": [["true", "yes"], ["false", "no"]]},
        ),
        (
            {"type": "choice", "instructions": "choose", "criteria": ["second", "first"]},
            {"field": "criteria", "kind": "list", "items": ["second", "first"]},
        ),
        (
            {"type": "choice", "instructions": "choose", "options": ["second", "first"]},
            {"field": "options", "kind": "list", "items": ["second", "first"]},
        ),
    ],
    ids=["noul-null", "noul-reversed-mapping", "choice-list-criteria", "choice-list-options"],
)
def test_accepted_typed_formats_are_passed_through_unchanged(question, expected):
    from s1.contracts import DecisionRequest

    state = {"document": "synthetic"}
    questions = {"decision": copy.deepcopy(question)}
    DecisionRequest.model_validate({"state": state, "questions": questions})
    tracker = coverage.HeadCoverage()
    original = copy.deepcopy(questions)
    result = {"decision": {"example": "unchanged"}}
    received = []

    def predictor(original_state, original_questions):
        received.append((copy.deepcopy(original_state), copy.deepcopy(original_questions)))
        return result

    wrapped = tracker.wrap_sync(predictor, route="generic")
    assert wrapped(state, questions) is result
    assert questions == original and received == [(state, original)]
    view = coverage.inspect_request(state, questions)
    assert view["ordered_questions"][0]["criteria"] == expected
    assert tracker.snapshot()["generic_bypass_completed"] == 1
    assert tracker.snapshot()["applied_heads"] == {"operation": 0, "click_target": 0}


def test_coherence_noul_empty_criteria_reaches_predictor_unchanged():
    # JevBench's public schema accepts an empty Noul criteria mapping. Some
    # providers may still refuse it; counting must not preempt their decision.
    question = {"type": "noul", "instructions": "allowed?", "criteria": {}}
    questions = {"decision": question}
    original = copy.deepcopy(questions)
    seen = []
    tracker = coverage.HeadCoverage()
    wrapped = tracker.wrap_sync(
        lambda state, original_questions: seen.append(copy.deepcopy(original_questions)) or {},
        route="generic",
    )
    assert wrapped("state", questions) == {}
    assert questions == original and seen == [original]
    assert coverage.inspect_request("state", questions)["ordered_questions"][0]["criteria"] == {
        "field": "criteria",
        "kind": "mapping",
        "items": [],
    }
    assert tracker.snapshot()["generic_bypass_completed"] == 1


@pytest.mark.parametrize(
    ("state", "questions"),
    [
        (None, {"q": {"type": "noul", "instructions": "allowed?"}}),
        (["one", "two"], {"q": {"type": "noul", "instructions": "allowed?"}}),
        (
            "state",
            {"": {"type": "choice", "instructions": "choose", "criteria": {"": "empty", "a": "A"}}},
        ),
        ("state", {"q": {"type": "score", "instructions": "rate", "criteria": ["", "high"]}}),
    ],
    ids=["null-state", "list-state", "empty-choice-key", "empty-score-description"],
)
def test_coherence_public_schema_inputs_are_not_preemptively_rejected(state, questions):
    tracker = coverage.HeadCoverage()
    original = copy.deepcopy((state, questions))
    seen = []
    wrapped = tracker.wrap_sync(
        lambda actual_state, actual_questions: (
            seen.append(copy.deepcopy((actual_state, actual_questions))) or {}
        ),
        route="generic",
    )
    assert wrapped(state, questions) == {}
    assert (state, questions) == original and seen == [original]
    assert tracker.snapshot()["generic_bypass_completed"] == 1


@pytest.mark.parametrize(
    "question",
    [
        {"type": "noul", "instructions": "allowed?", "criteria": {"true": "yes"}},
        {"type": "noul", "instructions": "allowed?", "criteria": ["false", "true"]},
        {"type": "choice", "instructions": "choose", "options": ["duplicate", "duplicate"]},
        {"type": "choice", "instructions": "choose", "criteria": ["valid", " "]},
        {"type": "choice", "instructions": "choose", "criteria": None},
    ],
)
def test_invalid_typed_criteria_still_fail_before_prediction(question):
    tracker = coverage.HeadCoverage()
    wrapped = tracker.wrap_sync(
        lambda *_: pytest.fail("invalid question reached model"), route="generic"
    )
    with pytest.raises(ValueError):
        wrapped("state", {"decision": question})
    assert tracker.snapshot()["rejected_before_prediction"] == 1
    assert tracker.snapshot()["generic_bypass_requests"] == 0


def test_empty_media_is_preserved_and_covered_by_packet_hash(browser):
    from s1.jev_decision3_native import parse_native_payload

    packet = copy.deepcopy(browser)
    packet["questions"]["operation"]["criteria"]["TYPE_TEXT"] = "type"
    without_media = copy.deepcopy(packet)
    packet["media"] = []
    parsed, original, _, _ = parse_native_payload(packet, model_identity=packet["model"])
    assert parsed.media == [] and original == packet
    expected = coverage.inspect_packet(packet)
    assert expected["packet_sha256"] != coverage.inspect_packet(without_media)["packet_sha256"]
    tracker = coverage.HeadCoverage()
    observed = []
    answer = {"answers": {"operation": {"choice": "CLICK"}}}
    wrapped = tracker.wrap_sync(
        lambda item: observed.append(copy.deepcopy(item)) or answer,
        route="browser",
        packet=True,
        evidence=lambda _: receipt(expected),
        expected_views={expected["packet_sha256"]: expected},
    )
    assert wrapped(packet) is answer
    assert observed == [packet] and packet["media"] == []
    assert tracker.snapshot()["applied_heads"] == {"operation": 1, "click_target": 1}
    with pytest.raises(ValueError, match="pinned packet"):
        wrapped(without_media)


@pytest.mark.parametrize("media", [None, {}, "", [{"type": "image"}]])
def test_nonempty_or_nonlist_media_is_refused_before_prediction(browser, media):
    packet = copy.deepcopy(browser)
    packet["media"] = media
    tracker = coverage.HeadCoverage()
    wrapped = tracker.wrap_sync(
        lambda _: pytest.fail("invalid media reached model"), route="generic", packet=True
    )
    with pytest.raises(ValueError, match="empty media"):
        wrapped(packet)
    assert tracker.snapshot()["rejected_before_prediction"] == 1
    assert tracker.snapshot()["generic_bypass_requests"] == 0
