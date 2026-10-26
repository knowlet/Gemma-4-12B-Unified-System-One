"""Source/contract checks only; synthetic rows never count as browser proof."""

import ast
import copy
import importlib.util
import json
import subprocess
import sys
from collections import Counter
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

import pytest
from local_evidence import require_local_evidence

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/jev_generic_menu_semantics_prototype.py"
SPEC = (
    ROOT
    / "artifacts/jev-runtime/generic-menu-semantics-prototype-20261018-01/specification-02.json"
)
module_spec = importlib.util.spec_from_file_location("semantic_menu_tested_source", SOURCE)
producer = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(producer)


class Document(HTMLParser):
    def __init__(self, value):
        super().__init__()
        self.nodes = []
        self.stack = []
        self.feed(value)

    def handle_starttag(self, tag, attrs):
        row = {"tag": tag, "attrs": dict(attrs), "ancestors": list(self.stack)}
        self.nodes.append(row)
        if tag not in {"meta", "input", "br", "hr", "img", "link"}:
            self.stack.append(row)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i]["tag"] == tag:
                del self.stack[i:]
                return


def shown(text="example"):
    return {
        "text": text,
        "visible": True,
        "in_view": True,
        "unoccluded_center": True,
        "complete_text_in_view": True,
        "text_unoccluded": True,
    }


def initial_pair(scene):
    evidence = producer.observable_contract(scene)
    page = {
        "text": "\n".join(evidence["expected_initial_text"]),
        "scroll": {"y": scene["prior_scrolls"] * 560},
    }
    elements = {
        "near-heading": shown(scene["near"]),
        "near-paragraph": {**shown(), "in_view": False},
        "far-paragraph": {**shown(), "in_view": False},
        "entry-local": shown(),
        "entry-global": shown(),
        "leaf-local": {"visible": False},
        "leaf-global": {"visible": False},
    }
    private = {
        "width": 1120,
        "height": 780,
        "document_scope": "record",
        "title": scene["title"],
        "note": "unchanged",
        "elements": elements,
    }
    return page, private


def test_fixed_population_and_whole_scene_family_split():
    scenes = producer.scenarios()
    assert len(scenes) == len({row["id"] for row in scenes}) == 8
    assert sum(len(row["goals"]) for row in scenes) == 14
    assert Counter(row["target_family"] for row in scenes) == {
        "local": 3,
        "global": 3,
        "control": 2,
    }
    assert {
        split: sum(len(row["goals"]) for row in scenes if row["split"] == split)
        for split in ["train", "development"]
    } == {"train": 9, "development": 5}
    for field in ["id", "title", "remote_title", "family", "layout_family"]:
        assert not {row[field] for row in scenes if row["split"] == "train"} & {
            row[field] for row in scenes if row["split"] == "development"
        }
    for family in {row["family"] for row in scenes}:
        assert Counter(
            row["target_family"]
            for row in scenes
            if row["family"] == family and row["target_family"] != "control"
        ) == {"local": 1, "global": 1}


def test_target_kind_order_and_scroll_history_balance():
    scenes = [row for row in producer.scenarios() if row["target_family"] != "control"]
    assert Counter(row[row["target_family"] + "_kind"] for row in scenes) == {
        "button": 3,
        "hover": 3,
    }
    assert Counter(row["order"].index(row["target_family"]) for row in scenes) == {0: 3, 1: 3}
    for family in ["local", "global"]:
        assert sorted(row["prior_scrolls"] for row in scenes if row["target_family"] == family) == [
            0,
            1,
            3,
        ]


@pytest.mark.local_evidence
def test_reviewed_specification_and_goals_remain_exact():
    require_local_evidence(SPEC)
    assert producer.specification() == json.loads(SPEC.read_bytes())


def test_specification_goals_contract():
    spec = producer.specification()
    assert len(spec["per_goal_review"]) == 14
    for row in spec["per_goal_review"]:
        scene = next(scene for scene in producer.scenarios() if scene["id"] == row["scene_id"])
        assert row["goal"] == scene["goals"][row["branch"]]
        assert (
            row["required_passage"] == scene["remote_detail"]
            if row["branch"] == "menu" and scene["target_family"] == "global"
            else row["required_passage"] in {scene["local_detail"], scene["near_detail"]}
        )
    assert "not used to assign labels" in spec["teacher_policy"]["sixteen_steps"]
    assert "shortest" in spec["teacher_policy"]["meaning"]


@pytest.mark.parametrize("scene", producer.scenarios(), ids=lambda row: row["id"])
def test_owned_html_has_real_scoped_routes_and_no_teacher_metadata(scene):
    source = producer.render_fixture(scene)
    document = Document(source)
    links = [node["attrs"] for node in document.nodes if node["tag"] == "a"]
    assert {row["href"] for row in links} == {"#chapter", "/" + scene["id"] + "/library"}
    assert {row["id"] for row in links} == {"leaf-local", "leaf-global"}
    assert not any(
        node["tag"] in {"iframe", "img", "form"} or "src" in node["attrs"]
        for node in document.nodes
    )
    assert all(
        word not in source
        for word in ["teacher_operation", "target_family", "expected_action", "goal", "gold"]
    )
    for node in document.nodes:
        if node["attrs"].get("class") == "caption":
            assert not any(
                "navbox" in parent["attrs"].get("class", "") for parent in node["ancestors"]
            )
    remote = producer.render_fixture(scene, remote=True)
    assert scene["remote_title"] in unescape(remote) and scene["remote_detail"] in unescape(remote)
    assert scene["near_detail"] not in remote and scene["local_detail"] not in remote
    assert not any(node["tag"] in {"a", "input"} for node in Document(remote).nodes)
    assert f"#near{{top:{scene['prior_scrolls'] * 560 + 688}px}}" in source
    assert "#near p{margin-top:110px}" in source
    assert (
        "sessionStorage" in source
    )  # Trusted pre-navigation event survives the real route change.


def test_controls_have_working_irrelevant_menus_without_a_fake_click_pair():
    for row in producer.scenarios():
        if row["target_family"] == "control":
            assert list(row["goals"]) == ["control"]
            assert row["local_heading"] != row["near"] == row["shared"]
            assert row["local_detail"] != row["near_detail"] != row["remote_detail"]
            assert "entry-local" in producer.render_fixture(
                row
            ) and "entry-global" in producer.render_fixture(row)


def test_lexical_audit_reports_limits_instead_of_claiming_shortcut_freedom():
    audit = producer.lexical_and_split_audit()
    assert audit["goal_count"] == 14
    assert "does not prove" in audit["scope"]
    assert audit["residual_risks"]
    for word in ["separate", "reference", "this", "view"]:
        counts = audit["cue_occurrence_by_teacher_operation"][word]
        assert counts["CLICK"] > 0 and counts["SCROLL_DOWN"] > 0


@pytest.mark.parametrize("scene", producer.scenarios(), ids=lambda row: row["id"])
def test_initial_contract_requires_actual_observable_text_before_annotation(scene):
    page, private = initial_pair(scene)
    producer.assert_initial(scene, page, private)
    for clue in set(producer.observable_contract(scene)["expected_initial_text"]):
        broken = copy.deepcopy(page)
        broken["text"] = broken["text"].replace(clue, "")
        with pytest.raises(ValueError, match="observable clue"):
            producer.assert_initial(scene, broken, private)


@pytest.mark.parametrize(
    "fault",
    [
        "paragraph_already_visible",
        "near_heading_hidden",
        "wrong_document",
        "wrong_viewport",
        "wrong_prior_history",
        "popup_preopened",
        "scope_line_absent",
    ],
)
def test_initial_negative_evidence_cannot_be_assumed_from_fixture_layout(fault):
    scene = producer.scenarios()[0]
    page, private = initial_pair(scene)
    if fault == "paragraph_already_visible":
        page["text"] += scene["near_detail"]
    elif fault == "near_heading_hidden":
        private["elements"]["near-heading"]["complete_text_in_view"] = False
    elif fault == "wrong_document":
        private["document_scope"] = "library"
    elif fault == "wrong_viewport":
        private["width"] = 1920
    elif fault == "wrong_prior_history":
        page["scroll"]["y"] = 560
    elif fault == "popup_preopened":
        private["elements"]["leaf-global"]["visible"] = True
    else:
        page["text"] = scene["near"]
    with pytest.raises(ValueError):
        producer.assert_initial(scene, page, private)


@pytest.mark.parametrize("scope", ["local", "global"])
def test_same_word_destination_requires_document_scope_and_full_passage(scope):
    scene = producer.scenarios()[0]
    origin = "http://127.0.0.1:31234"
    result = {
        "url": origin + "/" + scene["id"] + ("/library" if scope == "global" else "#chapter"),
        "document_scope": "library" if scope == "global" else "record",
        "title": scene["remote_title" if scope == "global" else "title"],
        "elements": {
            "far-heading": shown(scene["shared"]),
            "far-paragraph": shown(scene["remote_detail" if scope == "global" else "local_detail"]),
        },
    }
    producer.assert_destination(scene, scope, result, origin)
    for fault in ["scope", "paragraph", "heading_only", "occluded", "route"]:
        wrong = copy.deepcopy(result)
        if fault == "scope":
            wrong["document_scope"] = "record" if scope == "global" else "library"
        elif fault == "paragraph":
            wrong["elements"]["far-paragraph"]["text"] = scene[
                "local_detail" if scope == "global" else "remote_detail"
            ]
        elif fault == "heading_only":
            wrong["elements"]["far-paragraph"]["in_view"] = False
        elif fault == "occluded":
            wrong["elements"]["far-heading"]["text_unoccluded"] = False
        else:
            wrong["url"] = origin + "/different-document"
        with pytest.raises(ValueError):
            producer.assert_destination(scene, scope, wrong, origin)


def synthetic_coverage_rows():
    """Validator unit fixture, deliberately not native CDP/browser evidence."""
    proofs = []
    for scene in producer.scenarios():
        for task in producer.auxiliary_tasks():
            stage, probe = task["stage"], task["probe"]
            click = {"1": {}, "2": {}, "3": {}}
            if stage != "initial":
                click["4"] = {}
            payload = {
                "model": "unit-only",
                "state": {},
                "questions": {
                    "operation": {"criteria": {"CLICK": "click", "TYPE_TEXT": "fill"}},
                    "click_target": {"criteria": click},
                    "type_text_target": {"criteria": {"1": {}}},
                },
            }
            ids = {"local": "1", "global": "2", "focus": "3", "fill": "1", "leaf": "4"}
            proof = {
                "mode": "auxiliary",
                "stage": stage,
                "probe": probe,
                "positive_supervision": False,
                "stage_native_payload": payload,
                "stage_private": {"html": "synthetic-identical-" + stage},
                "stage_candidate_heads": {"click_target": list(click), "type_text_target": ["1"]},
                "record": {
                    "scenario_id": scene["id"],
                    "episode_id": scene["id"] + "-" + stage + "-" + probe,
                    "reachability_verified": True,
                    "selected": {
                        "operation": "TYPE_TEXT" if probe == "fill" else "CLICK",
                        "target": ids[probe],
                    },
                },
                "transitions": [{"fresh": True, "inputs": [{"unit_only": True}]}],
            }
            proofs.append(proof)
    return proofs


def test_every_declared_offered_target_has_separate_fresh_clone_certificate():
    tasks = producer.auxiliary_tasks()
    assert len(tasks) == 14
    assert Counter(row["stage"] for row in tasks) == {
        "initial": 4,
        "local_open": 5,
        "global_open": 5,
    }
    rows = synthetic_coverage_rows()
    assert len(rows) == 112
    cert = producer.coverage_certificate(rows)
    assert len(cert) == 24 and sum(len(row["covered_targets"]) for row in cert) == 112
    assert all(row["positive_supervision"] is False for row in cert)


@pytest.mark.parametrize(
    "fault",
    [
        "missing_proof",
        "duplicate_task",
        "extra_uncovered_target",
        "unexpected_head",
        "payload_order_changed",
        "candidate_order_changed",
        "DOM_changed",
        "unoffered_target",
        "not_reached",
        "fake_supervision",
        "not_fresh",
        "no_input",
    ],
)
def test_coverage_fails_closed_without_pruning_or_fake_success(fault):
    rows = synthetic_coverage_rows()
    first = rows[0]
    if fault == "missing_proof":
        rows.pop()
    elif fault == "duplicate_task":
        rows[1] = copy.deepcopy(first)
    elif fault == "extra_uncovered_target":
        for row in rows[:4]:
            row["stage_native_payload"]["questions"]["click_target"]["criteria"]["99"] = {}
            row["stage_candidate_heads"]["click_target"].append("99")
    elif fault == "unexpected_head":
        first["stage_native_payload"]["questions"]["select_target"] = {"criteria": {"1:x": {}}}
    elif fault == "payload_order_changed":
        questions = first["stage_native_payload"]["questions"]
        questions["operation"] = questions.pop("operation")
    elif fault == "candidate_order_changed":
        first["stage_candidate_heads"]["click_target"].reverse()
    elif fault == "DOM_changed":
        first["stage_private"]["html"] += "changed"
    elif fault == "unoffered_target":
        first["record"]["selected"]["target"] = "77"
    elif fault == "not_reached":
        first["record"]["reachability_verified"] = False
    elif fault == "fake_supervision":
        first["positive_supervision"] = True
    elif fault == "not_fresh":
        first["transitions"][-1]["fresh"] = False
    else:
        first["transitions"][-1]["inputs"] = []
    with pytest.raises(ValueError):
        producer.coverage_certificate(rows)


def test_read_only_private_expression_and_original_native_input_source_contract():
    assert all(
        forbidden not in producer.PRIVATE
        for forbidden in [".click(", "dispatchEvent", "location.href=", "scrollTo(", ".checked="]
    )
    tree = ast.parse(SOURCE.read_bytes())
    assert not any(
        isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "node"
            for target in node.targets
        )
        for node in ast.walk(tree)
    )
    act = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "act"
    ]
    assert (
        len(act) == 1
        and isinstance(act[0].func.value, ast.Name)
        and act[0].func.value.id == "browser"
    )
    assert [arg.id for arg in act[0].args] == ["action", "page"]
    assert {arg.arg: arg.value.id for arg in act[0].keywords} == {"text": "text"}
    assert "transport.observation_expression = verified.read_state" in SOURCE.read_text()
    assert "parent.capture_request(model, page, goal, history)" in SOURCE.read_text()


def test_default_cli_is_pure_spec_with_network_frameworks_and_browser_denied(tmp_path):
    program = r"""
import http.server, importlib.abc, importlib.util, pathlib, socket, sys
class Reject(importlib.abc.MetaPathFinder):
 def find_spec(self, name, path=None, target=None):
  if name.split('.')[0] in {'torch','transformers','tokenizers','playwright','peft','httpx'}:raise AssertionError(name)
sys.meta_path.insert(0,Reject())
def denied(*a,**k):raise AssertionError('network denied')
socket.socket.connect=socket.socket.bind=socket.socket.listen=denied
p=pathlib.Path(sys.argv[1]);s=importlib.util.spec_from_file_location('pure_semantic',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
m.resources=m.run_prototype=denied
raise SystemExit(m.main(['--output',sys.argv[2],'--source-sha256',m.sha(p.read_bytes())]))
"""
    result = subprocess.run(
        [sys.executable, "-B", "-s", "-c", program, str(SOURCE), str(tmp_path / "pure")],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "specification_only_no_browser"
    assert (
        json.loads((tmp_path / "pure/specification.json").read_bytes()) == producer.specification()
    )


def test_declared_fixed_header_does_not_make_wheel_reference_impossible_by_step_geometry():
    # Source geometry only, not a browser timing/visibility result. The earlier
    # 4000px gap aligned the far heading under the fixed header at wheel8.
    for scene in producer.scenarios():
        if scene["target_family"] == "local":
            source = producer.render_fixture(scene)
            far_y = scene["prior_scrolls"] * 560 + 688 + 4200
            assert f"#chapter{{top:{far_y}px}}" in source
            relative_heading_top = far_y + 22 - scene["prior_scrolls"] * 560 - 8 * 560
            assert relative_heading_top > 300
            assert relative_heading_top + 28 + 16 + 72 < 780
