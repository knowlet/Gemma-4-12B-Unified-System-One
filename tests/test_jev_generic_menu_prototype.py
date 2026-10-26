"""Source-only contracts for the owned generic menu correction prototype.

Synthetic records below exercise DTO conversion and byte comparisons only.
They are not browser proofs or admitted training data. No test in this module
starts Chromium, a model, a tokenizer, an external client, collection, or fitting.
The independent one-shot owned browser proof belongs to the prototype driver.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import jev_generic_menu_prototype as prototype
import pytest

from s1.browser_public_training import PROOF_FIELDS, PublicNativeActionCase
from s1.browser_public_training_validation import public_compile_wire_projection

ROOT = Path(__file__).resolve().parents[1]


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def payload(goal="Open the distant section through the local menu.", *, width=3):
    questions = {
        "operation": {
            "type": "choice",
            "criteria": {
                "CLICK": "Click an observed control",
                "TYPE_TEXT": "Fill an observed field",
                "SELECT": "Select an observed option",
                "SCROLL_DOWN": "Scroll down",
                "SCROLL_UP": "Scroll up",
                "WAIT": "Wait",
                "DONE": "Every requirement is visible",
                "BLOCKED": "No supported progress",
            },
            "instructions": {"goal": goal, "rules": ["Keep unrelated fields unchanged"]},
        }
    }
    for head, operation, criteria in (
        (
            "click_target",
            "CLICK",
            {str(i): {"label": "Entry " + str(i)} for i in range(1, width + 1)},
        ),
        ("type_text_target", "TYPE_TEXT", {"field": {"label": "Unused editable field"}}),
        ("select_target", "SELECT", {"choice:a": {"label": "Unused native option"}}),
    ):
        questions[head] = {
            "type": "choice",
            "criteria": criteria,
            "instructions": {"goal": goal, "operation": operation, "rules": ["Use current state"]},
        }
    return {
        "model": "source-only-native-alias",
        "state": {
            "page": {
                "url": "http://127.0.0.1:43210/owned-fixture",
                "title": "Owned generic menu",
                "text": goal + " · 選單",
            },
            "elements": [
                {"index": "1", "role": "button", "label": "Menu", "operations": ["CLICK"]}
            ],
            "recent_actions": [{"action": "Scroll down", "kind": "scroll", "page_changed": True}],
        },
        "questions": questions,
    }


def refs():
    # Syntactically valid, nonexistent references deliberately confer no admission.
    return {
        name: {"artifact_path": f"owned-proof/{name}.jsonl", "sha256": "a" * 64, "sequence": 0}
        for name in sorted(PROOF_FIELDS)
    }


def records(*, width=3):
    spec = next(s for s in prototype.scenarios() if s["menu_kind"] != "none")
    html_sha = sha(prototype.render_fixture(spec).encode())
    native = payload(width=width)
    left = {
        "scenario_id": spec["id"],
        "split": spec["split"],
        "episode_id": spec["id"] + "-menu",
        "branch": "menu",
        "html_sha256": html_sha,
        "history": copy.deepcopy(native["state"]["recent_actions"]),
        "payload": native,
        "selected": {"operation": "CLICK", "target": "1"},
        "reachability_verified": True,
    }
    right = copy.deepcopy(left)
    right.update(
        episode_id=spec["id"] + "-scroll",
        branch="scroll",
        selected={"operation": "SCROLL_DOWN", "target": None},
    )
    # Keep page text, candidate order and every other field byte-identical.
    for question in right["payload"]["questions"].values():
        question["instructions"]["goal"] = "Bring the nearby lower section into view."
    return left, right


def test_scenario_grains_and_split_are_explicit():
    scenes = prototype.scenarios()
    assert isinstance(scenes, list) and len(scenes) == 8
    assert len({s["id"] for s in scenes}) == 8
    fields = {
        "id",
        "split",
        "layout",
        "menu_kind",
        "prior_scrolls",
        "menu_label",
        "link_label",
        "near_label",
        "far_label",
    }
    for scene in scenes:
        assert fields <= scene.keys()
        assert scene["menu_kind"] in {"hover", "button", "none"}
        assert type(scene["prior_scrolls"]) is int and scene["prior_scrolls"] in {0, 1, 3}
        assert scene["split"] in {"train", "development"}
    menus = [s for s in scenes if s["menu_kind"] != "none"]
    assert Counter((s["menu_kind"], s["prior_scrolls"]) for s in menus) == Counter(
        (kind, n) for kind in ("hover", "button") for n in (0, 1, 3)
    )
    assert Counter(s["split"] for s in scenes) == {"train": 5, "development": 3}
    assert Counter(s["split"] for s in menus) == {"train": 4, "development": 2}
    assert Counter(s["split"] for s in scenes if s["menu_kind"] == "none") == {
        "train": 1,
        "development": 1,
    }
    assert sum(2 if s["menu_kind"] != "none" else 1 for s in scenes) == 14


class FixtureMarkup(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.external = [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        for attribute in ("src", "href", "action"):
            value = attrs.get(attribute)
            if value and (urlsplit(value).scheme or value.startswith("//")):
                self.external.append((tag, attribute, value))


@pytest.mark.parametrize("index", range(8))
def test_fixture_render_is_deterministic_and_has_no_external_resource(index):
    scene = prototype.scenarios()[index]
    original = copy.deepcopy(scene)
    first = prototype.render_fixture(scene)
    assert isinstance(first, str) and first == prototype.render_fixture(scene)
    assert scene == original
    parsed = FixtureMarkup()
    parsed.feed(first)
    assert parsed.external == []
    assert len(set(parsed.ids)) == len(parsed.ids)
    assert scene["near_label"] in first and scene["far_label"] in first


def test_normalization_changes_only_each_question_goal_and_keeps_order():
    original = payload(width=53)
    before = encoded(original)
    normalized = prototype.normalized_pair_payload(original)
    assert isinstance(normalized, bytes)
    result = json.loads(normalized)
    replacement = result["questions"]["operation"]["instructions"]["goal"]
    expected = copy.deepcopy(original)
    for question in expected["questions"].values():
        question["instructions"]["goal"] = replacement
    assert normalized == encoded(expected)
    assert result["state"] == original["state"]
    assert list(result["questions"]) == list(original["questions"])
    assert list(result["questions"]["click_target"]["criteria"]) == [str(i) for i in range(1, 54)]
    assert before == encoded(original)


@pytest.mark.parametrize(
    "mutation", ["absent", "empty", "whitespace", "non_string", "inconsistent"]
)
def test_missing_or_inconsistent_unused_head_goal_cannot_be_hidden(mutation):
    native = payload()
    instructions = native["questions"]["select_target"]["instructions"]
    if mutation == "absent":
        del instructions["goal"]
    elif mutation == "empty":
        instructions["goal"] = ""
    elif mutation == "whitespace":
        instructions["goal"] = "   "
    elif mutation == "non_string":
        instructions["goal"] = ["goal"]
    else:
        instructions["goal"] = "A different unused-head goal"
    with pytest.raises(ValueError):
        prototype.normalized_pair_payload(native)


def test_valid_pair_has_only_goal_difference_and_does_not_mutate_records():
    left, right = records()
    before = encoded([left, right])
    verified = prototype.validate_pair(left, right)
    assert verified == {
        "scenario_id": left["scenario_id"],
        "normalized_payload_sha256": sha(prototype.normalized_pair_payload(left["payload"])),
        "goals_differ": True,
    }
    assert before == encoded([left, right])


PAIR_MUTATIONS = (
    "same_goal",
    "scenario",
    "split",
    "html",
    "history",
    "state",
    "candidate_order",
    "unused_head",
    "rules",
    "model",
    "menu_branch",
    "control_branch",
    "click_target",
    "scroll_target",
    "menu_not_click",
    "scroll_not_scroll",
    "left_unverified",
    "right_unverified",
    "scroll_not_offered",
    "record_history_drift",
)


@pytest.mark.parametrize("mutation", PAIR_MUTATIONS)
def test_pair_rejects_observation_history_head_or_action_mismatch(mutation):
    left, right = records()
    if mutation == "same_goal":
        right["payload"] = copy.deepcopy(left["payload"])
    elif mutation == "scenario":
        right["scenario_id"] += "-other"
    elif mutation == "split":
        right["split"] = "development" if left["split"] == "train" else "train"
    elif mutation == "html":
        right["html_sha256"] = "b" * 64
    elif mutation == "history":
        right["history"][0]["page_changed"] = False
    elif mutation == "state":
        right["payload"]["state"]["page"]["url"] += "?different-episode"
    elif mutation == "candidate_order":
        criteria = right["payload"]["questions"]["click_target"]["criteria"]
        right["payload"]["questions"]["click_target"]["criteria"] = dict(reversed(criteria.items()))
    elif mutation == "unused_head":
        right["payload"]["questions"]["select_target"]["criteria"]["choice:a"]["label"] = "Changed"
    elif mutation == "rules":
        right["payload"]["questions"]["operation"]["instructions"]["rules"] = ["Different rule"]
    elif mutation == "model":
        right["payload"]["model"] = "rewritten-alias"
    elif mutation == "menu_branch":
        left["branch"] = "scroll"
    elif mutation == "control_branch":
        right["branch"] = "control"
    elif mutation == "click_target":
        left["selected"]["target"] = "not-offered"
    elif mutation == "scroll_target":
        right["selected"]["target"] = "1"
    elif mutation == "menu_not_click":
        left["selected"] = {"operation": "SCROLL_DOWN", "target": None}
    elif mutation == "scroll_not_scroll":
        right["selected"] = {"operation": "CLICK", "target": "1"}
    elif mutation == "left_unverified":
        left["reachability_verified"] = False
    elif mutation == "right_unverified":
        right["reachability_verified"] = False
    elif mutation == "scroll_not_offered":
        for record in (left, right):
            del record["payload"]["questions"]["operation"]["criteria"]["SCROLL_DOWN"]
    else:
        # Both records agree with each other, so pair comparison alone cannot
        # reveal that they no longer match the actual captured native history.
        for record in (left, right):
            record["history"][0]["page_changed"] = False
    with pytest.raises(ValueError):
        prototype.validate_pair(left, right)


@pytest.mark.parametrize("branch", ["menu", "scroll"])
@pytest.mark.parametrize("width", [53, 400])
def test_dto_and_real_compile_projection_keep_full_native_wire(branch, width):
    pair = records(width=width)
    record = pair[0 if branch == "menu" else 1]
    case, wire = prototype.build_case_and_wire(record, refs())
    assert isinstance(case, PublicNativeActionCase)
    assert case.request.model_dump(mode="json") == record["payload"] == wire["native_payload"]
    assert case.id == record["episode_id"] + "-initial" == wire["case_id"]
    assert case.wire_row_id == case.id + "-wire" == wire["wire_row_id"]
    assert case.pair_id == "owned-generic-menu-" + record["scenario_id"] == wire["pair_id"]
    assert case.state_role == "ordinary_incomplete"
    assert case.group_id == "public-native-site:owned-generic-menu-" + record["scenario_id"]
    assert wire["schema_id"] == "jev-public-native-action-wire-v1"
    assert wire["teacher_choice"] == record["selected"]
    assert wire["native_wire_sha256"] == sha(encoded(record["payload"]))
    assert list(wire["native_payload"]["questions"]) == list(record["payload"]["questions"])
    assert len(wire["native_payload"]["questions"]["click_target"]["criteria"]) == width
    assert set(case.gold) == ({"operation", "click_target"} if branch == "menu" else {"operation"})
    projected = public_compile_wire_projection([case], [wire])
    assert len(projected) == 1 and projected[0]["payload"] == record["payload"]
    assert "native_payload" not in projected[0]
    projected[0]["payload"]["questions"]["select_target"]["criteria"].clear()
    assert wire["native_payload"]["questions"]["select_target"]["criteria"]
    assert case.request.model_dump()["questions"]["select_target"]["criteria"]


def test_dto_and_wire_are_detached_from_record_and_external_references():
    record, _ = records()
    proof = refs()
    case, wire = prototype.build_case_and_wire(record, proof)
    before_case, before_wire = case.model_dump(mode="json"), copy.deepcopy(wire)
    record["payload"]["state"]["page"]["text"] = "MUTATED"
    record["selected"]["target"] = "2"
    proof["ownership"]["sequence"] = 99
    assert case.model_dump(mode="json") == before_case
    assert wire == before_wire
    wire["native_payload"]["questions"]["click_target"]["criteria"].clear()
    assert case.model_dump(mode="json") == before_case


def test_no_menu_control_serializes_scroll_without_inventing_a_click_pair():
    spec = next(s for s in prototype.scenarios() if s["menu_kind"] == "none")
    _, record = records()
    record.update(
        scenario_id=spec["id"],
        split=spec["split"],
        episode_id=spec["id"] + "-control",
        branch="control",
        html_sha256=sha(prototype.render_fixture(spec).encode()),
    )
    case, wire = prototype.build_case_and_wire(record, refs())
    assert case.gold == {"operation": "SCROLL_DOWN"}
    assert case.valid_next_targets == {"SCROLL_DOWN": []}
    assert public_compile_wire_projection([case], [wire])[0]["payload"] == record["payload"]
    with pytest.raises(ValueError):
        prototype.validate_pair(record, copy.deepcopy(record))


@pytest.mark.parametrize("flag", [False, None, 0, 1, "true"])
def test_unverified_or_non_boolean_reachability_never_serializes(flag):
    record, _ = records()
    record["reachability_verified"] = flag
    with pytest.raises(ValueError):
        prototype.build_case_and_wire(record, refs())


@pytest.mark.parametrize(
    "selected",
    [
        {"operation": "CLICK", "target": None},
        {"operation": "CLICK", "target": "not-offered"},
        {"operation": "SCROLL_DOWN", "target": "1"},
        {"operation": "DONE", "target": None},
        {"operation": "BLOCKED", "target": None},
    ],
)
def test_builder_rejects_illegal_active_target_or_complete_label(selected):
    record, _ = records()
    record["selected"] = selected
    with pytest.raises(ValueError):
        prototype.build_case_and_wire(record, refs())


@pytest.mark.parametrize("mutation", ["missing", "path_escape", "bad_digest", "bad_sequence"])
def test_builder_keeps_existing_strict_external_reference_contract(mutation):
    record, _ = records()
    proof = refs()
    if mutation == "missing":
        del proof["ownership"]
    elif mutation == "path_escape":
        proof["ownership"]["artifact_path"] = "../outside.json"
    elif mutation == "bad_digest":
        proof["ownership"]["sha256"] = "not-a-sha256"
    else:
        proof["ownership"]["sequence"] = -1
    with pytest.raises(ValueError):
        prototype.build_case_and_wire(record, proof)


def test_rehashed_candidate_reordering_fails_real_dto_wire_projection():
    record, _ = records()
    case, wire = prototype.build_case_and_wire(record, refs())
    criteria = wire["native_payload"]["questions"]["click_target"]["criteria"]
    wire["native_payload"]["questions"]["click_target"]["criteria"] = dict(
        reversed(criteria.items())
    )
    wire["native_wire_sha256"] = sha(encoded(wire["native_payload"]))
    with pytest.raises(ValueError, match="order/bytes"):
        public_compile_wire_projection([case], [wire])


@pytest.mark.parametrize("existing_environment", [False, True])
@pytest.mark.parametrize("unexpected_error", [False, True])
def test_capture_uses_no_real_post_and_restores_environment_and_callable(
    monkeypatch, existing_environment, unexpected_error
):
    if existing_environment:
        monkeypatch.setenv("TYPESAFE_API_KEY", "owned-test-original-key")
        monkeypatch.setenv("TYPESAFE_MODEL", "owned-test-original-model")
    else:
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        monkeypatch.delenv("TYPESAFE_MODEL", raising=False)
    before = {name: os.environ.get(name) for name in ("TYPESAFE_API_KEY", "TYPESAFE_MODEL")}
    attempted_original_posts, seen = [], []

    def original_post(*_args, **_kwargs):
        attempted_original_posts.append(True)
        raise AssertionError("Original post callable must never execute")

    model = SimpleNamespace(post_json=original_post)
    sentinel = RuntimeError("owned unexpected request construction error")
    goal, page, history = "Read the local menu destination.", {"source-only": True}, []

    def choose(current_page, current_goal, current_history):
        assert (current_page, current_goal, current_history) == (page, goal, history)
        assert os.environ["TYPESAFE_API_KEY"] == "owned-local-capture-no-http"
        assert os.environ["TYPESAFE_MODEL"] == prototype.MODEL_ALIAS
        if unexpected_error:
            raise sentinel
        body = payload(current_goal)
        body["model"] = os.environ["TYPESAFE_MODEL"]
        seen.append(copy.deepcopy(body))
        # This is an in-memory intercepted callable. No HTTP library or URL is used.
        model.post_json(None, os.environ["TYPESAFE_API_KEY"], body)

    model.choose = choose
    if unexpected_error:
        with pytest.raises(RuntimeError) as caught:
            prototype.capture_request(model, page, goal, history)
        assert caught.value is sentinel
    else:
        actual = prototype.capture_request(model, page, goal, history)
        assert len(seen) == 1 and actual == seen[0]
    assert model.post_json is original_post
    assert {name: os.environ.get(name) for name in before} == before
    assert attempted_original_posts == []


def test_pure_api_import_does_not_start_runtime_or_import_model_frameworks():
    code = r"""
import importlib.abc, pathlib, sys
blocked = {'torch','transformers','peft','bitsandbytes','tensorflow','jax','modal','playwright'}
class Reject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('runtime import during pure API: ' + fullname)
sys.meta_path.insert(0, Reject())
def audit(event, args):
    if event in {'socket.connect', 'subprocess.Popen'}:
        raise AssertionError('runtime side effect during pure API: ' + event)
sys.addaudithook(audit)
sys.path.insert(0, str(pathlib.Path(sys.argv[1]) / 'scripts'))
import jev_generic_menu_prototype as p
scenes = p.scenarios()
assert len(scenes) == 8
for spec in scenes:
    assert isinstance(p.render_fixture(spec), str)
assert not blocked.intersection(sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-s", "-c", code, str(ROOT)],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
