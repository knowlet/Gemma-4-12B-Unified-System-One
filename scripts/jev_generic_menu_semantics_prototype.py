"""Owned semantic navigation proof; separate from public admission or training.

The frozen v5 observer, original native request constructor and executor remain
unchanged. A teacher is a demonstrated useful route, not the only correct
action or a proof of global shortest paths. Default CLI only writes the spec.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import html
import importlib.util
import json
import re
import sys
import threading
from collections import Counter
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

PARENT_SHA = "97230a6b741d82a54236191edd8236083ab46fa1d7612dd0934b94abdcde34ee"
PROFILE_SHA = "7847b2f9eecb3f9d492ccfdf8415c52632b99ac6ead6df560c49c3caf004e141"
PROFILE = "composed-visible-hover-entries-v5"
WRAPPER_SHA = "c84c722d080b1aba8c4225f6e47e42baf33cb19f15545c25b167c0105c89f5d6"
COLLECTOR_SHA = "5cdfa3ffb8b1077eb5cd67c4910fb8b1759005258faf11b0c2012ed73fac0f4b"


def require(value, message):
    if not value:
        raise ValueError(message)


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(encode(value) + b"\n")


def write_rows(path, rows):
    with Path(path).open("xb") as stream:
        for row in rows:
            stream.write(encode(row) + b"\n")


def scenarios():
    # Each goal is authored here; no automatic operation-dependent template.
    rows = [
        {
            "id": "semantic-01",
            "split": "train",
            "family": "A-question",
            "target_family": "local",
            "prior_scrolls": 0,
            "title": "Glassmaking notebook",
            "remote_title": "Workshop glass reference",
            "shared": "Heat rest",
            "near": "Batch marks",
            "local_detail": "A batch cools beside the north rack while its surface loses the last bright reflection.",
            "remote_detail": "The stock-rack dimensions are listed for a separate workshop reference table.",
            "near_detail": "Pencilled symbols on sample cards distinguish the morning batches.",
            "local_trigger": "Notebook panels",
            "global_trigger": "Workshop shelf",
            "local_kind": "hover",
            "global_kind": "button",
            "order": ["local", "global"],
            "goals": {
                "menu": "In the glassmaking notebook, find what the Heat rest paragraph says about a batch cooling beside the north rack.",
                "scroll": "In the glassmaking notebook, find the Batch marks paragraph about pencilled symbols on sample cards.",
            },
        },
        {
            "id": "semantic-02",
            "split": "train",
            "family": "B-reader-context",
            "target_family": "global",
            "prior_scrolls": 1,
            "title": "Irrigation workbook",
            "remote_title": "Separate parts guide",
            "shared": "Drip balance",
            "near": "Watering log",
            "local_detail": "This workbook records an uneven drip measured beside the western planting bed.",
            "remote_detail": "The parts guide compares brass junction widths across three fittings.",
            "near_detail": "Morning meter readings were copied beside the sketch of the hose connection.",
            "local_trigger": "Workbook leaves",
            "global_trigger": "Parts collection",
            "local_kind": "button",
            "global_kind": "hover",
            "order": ["global", "local"],
            "goals": {
                "menu": "The separate parts guide is needed for its Drip balance account of brass junction widths; leave that passage in view.",
                "scroll": "The parts guide is a separate reference. My view should contain this irrigation workbook's Watering log account of morning meter readings.",
            },
        },
        {
            "id": "semantic-03",
            "split": "train",
            "family": "A-question",
            "target_family": "global",
            "prior_scrolls": 3,
            "title": "Acoustic journal",
            "remote_title": "Hall design reference",
            "shared": "Echo tails",
            "near": "Microphone notes",
            "local_detail": "The long hall held a fading tone after the padded door closed.",
            "remote_detail": "A separate hall-design reference tabulates ceiling panels rather than the journal session.",
            "near_detail": "The taped stand position was measured from the left edge of the stage.",
            "local_trigger": "Session pages",
            "global_trigger": "Building records",
            "local_kind": "hover",
            "global_kind": "button",
            "order": ["local", "global"],
            "goals": {
                "menu": "What does the separate hall-design reference say under Echo tails about ceiling panels? Bring that passage itself into view.",
                "scroll": "The hall-design reference is separate. What does this acoustic journal record under Microphone notes about the taped stand position? Bring that passage into view.",
            },
        },
        {
            "id": "semantic-04",
            "split": "train",
            "family": "B-reader-context",
            "target_family": "local",
            "prior_scrolls": 3,
            "title": "Textile handbook",
            "remote_title": "Collection material reference",
            "shared": "Fold memory",
            "near": "Storage tags",
            "local_detail": "The displayed handbook describes a loose crease in the undyed demonstration cloth.",
            "remote_detail": "Packed linen samples retained narrow ridges after a week in a wooden case.",
            "near_detail": "Cotton ties carry handwritten tags with the shelf date and the fabric count.",
            "local_trigger": "Handbook tabs",
            "global_trigger": "Material archive",
            "local_kind": "button",
            "global_kind": "hover",
            "order": ["global", "local"],
            "goals": {
                "menu": "I am comparing this handbook with the collection's material reference. I need this handbook's Fold memory paragraph about an undyed demonstration cloth visible.",
                "scroll": "I am checking the displayed textile handbook alongside the collection reference. I need this handbook's Storage tags paragraph about cotton ties visible.",
            },
        },
        {
            "id": "semantic-05",
            "split": "development",
            "family": "C-required-evidence",
            "target_family": "local",
            "prior_scrolls": 1,
            "title": "Glazing record",
            "remote_title": "Kiln equipment notes",
            "shared": "Cooling bands",
            "near": "Test tiles",
            "local_detail": "The blue test cup showed a narrow pale band beside its handle after cooling.",
            "remote_detail": "The equipment notes describe the outer metal bands around a kiln cabinet.",
            "near_detail": "Numbered clay squares carry two glaze strokes and a small punched mark.",
            "local_trigger": "Record folios",
            "global_trigger": "Equipment cabinet",
            "local_kind": "hover",
            "global_kind": "button",
            "order": ["global", "local"],
            "goals": {
                "menu": "A finished view contains the Cooling bands passage from this glazing record, including its observation of the blue test cup.",
                "scroll": "A finished view contains this glazing record's Test tiles passage, including the note about numbered clay squares.",
            },
        },
        {
            "id": "semantic-06",
            "split": "development",
            "family": "C-required-evidence",
            "target_family": "global",
            "prior_scrolls": 0,
            "title": "Canal maintenance ledger",
            "remote_title": "Separate seal catalogue",
            "shared": "Gate seals",
            "near": "Inspection marks",
            "local_detail": "The ledger records a damp streak seen beneath the lower gate after a short closure.",
            "remote_detail": "The catalogue's cross-section shows the braided packing inside the round seal housing.",
            "near_detail": "Three chalk ticks beside the hinge identify the latest inspection visit.",
            "local_trigger": "Ledger sheets",
            "global_trigger": "Supply catalogue",
            "local_kind": "hover",
            "global_kind": "button",
            "order": ["global", "local"],
            "goals": {
                "menu": "For the required evidence, the separate seal catalogue's Gate seals passage and its description of braided packing must be visible.",
                "scroll": "The separate seal catalogue uses similar headings. The required evidence here is this maintenance ledger's Inspection marks passage about three chalk ticks.",
            },
        },
        {
            "id": "semantic-07",
            "split": "train",
            "family": "A-question",
            "target_family": "control",
            "prior_scrolls": 1,
            "title": "Soundscape field notes",
            "remote_title": "Recorder equipment reference",
            "shared": "Recorder settings",
            "near": "Recorder settings",
            "local_heading": "Wind layers",
            "local_detail": "The appendix compares gusts passing above the orchard's upper branches.",
            "remote_detail": "The equipment reference explains generic calibration settings for a bench recorder.",
            "near_detail": "For the courtyard session, the recorder used a low gain setting and a felt cover.",
            "local_trigger": "Appendix leaves",
            "global_trigger": "Recorder manuals",
            "local_kind": "button",
            "global_kind": "hover",
            "order": ["local", "global"],
            "goals": {
                "control": "Which recorder settings were recorded for the courtyard? Find that passage in the field notes currently open."
            },
        },
        {
            "id": "semantic-08",
            "split": "development",
            "family": "C-required-evidence",
            "target_family": "control",
            "prior_scrolls": 0,
            "title": "Paper restoration log",
            "remote_title": "Paste equipment guide",
            "shared": "Paste weights",
            "near": "Paste weights",
            "local_heading": "Fiber shadows",
            "local_detail": "The appendix describes faint fiber shadows photographed through a thin blue sheet.",
            "remote_detail": "The separate equipment guide lists the mass of rectangular bench stones.",
            "near_detail": "The small cotton-bound parcel held one light wooden weight while its paste dried.",
            "local_trigger": "Log attachments",
            "global_trigger": "Equipment volumes",
            "local_kind": "hover",
            "global_kind": "button",
            "order": ["global", "local"],
            "goals": {
                "control": "The required view is the restoration log's Paste weights entry about the small cotton-bound parcel, rather than the equipment guide's bench stones."
            },
        },
    ]
    for index, row in enumerate(rows):
        row["layout_family"] = (
            "train-wide-bars"
            if row["family"] == "A-question"
            else "train-staggered"
            if row["family"] == "B-reader-context"
            else "development-side-rails"
        )
        row["layout_variant"] = index
    return rows


def observable_contract(spec):
    return {
        "source_intended_not_yet_observed": True,
        "expected_initial_text": [
            spec["title"],
            spec["remote_title"],
            spec["local_trigger"],
            spec["global_trigger"],
            spec.get("local_heading", spec["shared"]),
            spec["shared"],
            spec["near"],
        ],
        "absent_initial_passages": [
            spec["near_detail"],
            spec["local_detail"],
            spec["remote_detail"],
        ],
        "near_heading_top_after_true_prior_scroll": 710,
        "near_paragraph_top_after_true_prior_scroll": 848,
        "viewport": {"width": 1120, "height": 780},
        "evidence": "Fixed visible document/scope intro and menu captions; near heading is visible at the bottom, its required paragraph remains below. Runtime must assert exact original native snapshot text and private geometry before any annotation.",
        "postcondition": "Intended document URL/title/scope plus complete actual heading AND paragraph, each visible/in_view/unoccluded; heading alone is insufficient.",
    }


def specification():
    rows = scenarios()
    return {
        "schema": "owned-semantic-menu-specification-v1",
        "status": "authored_source_specification_not_browser_proof",
        "specification_revision": 2,
        "previous_specification_sha256": "a339c3dbcc1b1bb4f16eb2a0f36ebaafef4072622d7a7821a2368b02fb9321fd",
        "revision_delta": "Original14goals retained. Add explicit visible intro/near-heading initial evidence, absent paragraphs, separate heading+paragraph postcondition, layout family groups; no browser evidence is inferred.",
        "population": {
            "scenes": 8,
            "goal_only_pairs": 6,
            "controls_without_relevant_menu": 2,
            "initial_requests": 14,
            "initial_teacher_click": 6,
            "initial_teacher_scroll": 8,
            "click_target_families": {"local": 3, "global": 3},
            "split_initial_requests": {"train": 9, "development": 5},
        },
        "profile": {
            "name": PROFILE,
            "source_sha256": PROFILE_SHA,
            "native_request_fresh_executor": "unchanged",
        },
        "teacher_policy": {
            "meaning": "A useful reachable demonstration; no unique-correct-action or globally shortest-path assertion.",
            "menu": "Native trigger then genuinely offered visible link; local chapter or separate document as the goal requires.",
            "scroll": "One original native wheel reveals the requested current-document near passage; unrelated working menus stay available.",
            "efficiency": "For the three local CLICK cases, independently replay a fresh-clone wheel-only reference to the same far passage and record actual input counts. Two input menu route versus that observed reference only; never generalize to all possible routes.",
            "global": "Two demonstrated native clicks reach the specified separate document. No scrolling-impossibility or shortest-path claim.",
            "sixteen_steps": "Canonical16 unchanged and not used to assign labels, exclude alternatives, define reachability or admit this collection.",
        },
        "scene_and_goal_records": rows,
        "observable_evidence": {row["id"]: observable_contract(row) for row in rows},
        "per_goal_review": [
            {
                "scene_id": row["id"],
                "branch": branch,
                "goal": goal,
                "teacher_operation": "CLICK" if branch == "menu" else "SCROLL_DOWN",
                "intended_document": row["remote_title"]
                if branch == "menu" and row["target_family"] == "global"
                else row["title"],
                "intended_heading": row["shared"] if branch == "menu" else row["near"],
                "required_passage": row["remote_detail"]
                if branch == "menu" and row["target_family"] == "global"
                else row["local_detail"]
                if branch == "menu"
                else row["near_detail"],
                "counterexamples": [
                    "Matching heading on the other document is insufficient: document title and the goal's specific passage must both match.",
                    "Opening an unrelated working menu is reachable but does not expose the requested passage.",
                    "A menu demonstration does not make later scrolling or other useful actions incorrect.",
                ],
                "planned_proof": "Fresh observe/offered input/native executor/visible unoccluded postcondition; no proof has run.",
            }
            for row in rows
            for branch, goal in row["goals"].items()
        ],
        "split_rule": "Whole scene, content, layout and goal family stay in one split. A-question/B-reader-context only train; C-required-evidence only development. This small authored dev is not public-site or broad semantic generalization.",
        "review_limits": [
            "Goals remain authored and target nouns necessarily correlate with their scene; absence of click/scroll directives does not prove shortcut freedom.",
            "Scope words, goal lengths, family phrasing, target nouns, menu kind/order and task branch must be reviewed together; residual correlations are reported, not erased by a lexical test.",
            "Every offered initial/popup navigation distractor is to be exercised on a fresh clone, including same-word other-page routes. Input focus/edit evidence is auxiliary only, not positive supervision.",
            "These14new initial requests must not be mixed with, replace or directly fit the previous14; syntax compatibility is separate from a future independently versioned local admission.",
        ],
        "browser_model_fit_calls": 0,
    }


def render_fixture(spec, *, remote=False):
    require(spec in scenarios(), "Only the eight frozen authored scenes")
    esc = html.escape
    if remote:
        return f"""<!doctype html><html><head><meta charset="utf-8"><title>{esc(spec["remote_title"])}</title>
<style>body{{font:18px sans-serif;margin:40px;min-height:1300px}}section{{margin-top:120px;width:750px;padding:20px;border:2px solid #567}}h2{{margin:0;line-height:28px}}p{{line-height:24px}}</style></head>
<body data-owned-document="library"><h1>{esc(spec["remote_title"])}</h1><p>A separate collection document.</p>
<section id="chapter"><h2 id="far-heading">{esc(spec["shared"])}</h2><p id="far-paragraph">{esc(spec["remote_detail"])}</p></section>
{event_script()}</body></html>"""
    positions = {
        "train-wide-bars": [(30, 22), (690, 22)],
        "train-staggered": [(36, 30), (700, 92)],
        "development-side-rails": [(26, 70), (758, 18)],
    }[spec["layout_family"]]
    boxes = []
    for slot, scope in enumerate(spec["order"]):
        left, top = positions[slot]
        kind, label = spec[scope + "_kind"], spec[scope + "_trigger"]
        caption = spec["title"] if scope == "local" else spec["remote_title"]
        leaf = spec.get("local_heading", spec["shared"]) if scope == "local" else spec["shared"]
        href = "#chapter" if scope == "local" else "/" + spec["id"] + "/library"
        trigger = (
            f'<span id="entry-{scope}">{esc(label)}</span>'
            if kind == "hover"
            else f"<button id=\"entry-{scope}\" aria-expanded=\"false\" onclick=\"document.getElementById('popup-{scope}').hidden=false;this.setAttribute('aria-expanded','true')\">{esc(label)}</button>"
        )
        hidden = "" if kind == "hover" else " hidden"
        boxes.append(
            f'<div class="scope" style="left:{left}px;top:{top}px"><div class="caption">{esc(caption)}</div><div class="navbox kind-{kind}">{trigger}<div class="popup" id="popup-{scope}"{hidden}><a id="leaf-{scope}" href="{href}">{esc(leaf)}</a></div></div></div>'
        )
    near_y = spec["prior_scrolls"] * 560 + 688
    far_y = near_y + 4200
    local_heading = spec.get("local_heading", spec["shared"])
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{esc(spec["title"])}</title>
<style>html{{scroll-behavior:auto}}body{{margin:0;font:18px sans-serif;background:#edf1e8}}
.scope{{position:fixed;z-index:5;width:290px}}.caption{{font:12px/16px sans-serif;background:#e1e7ef;padding:4px;min-height:16px}}
.navbox{{position:relative;background:white;border:2px solid #42546b}}.navbox>span,.navbox>button{{display:inline-block;padding:12px;font:18px/22px sans-serif;cursor:pointer}}
.popup{{position:absolute;top:46px;left:0;width:270px;background:white;border:2px solid #42546b;padding:10px}}.popup a{{display:block;padding:10px;color:#153b67}}
.kind-hover .popup{{display:none}}.kind-hover:hover .popup{{display:block}}
header{{position:fixed;z-index:1;left:54px;top:210px;width:900px;background:#edf1e8}}header h1{{font-size:23px;line-height:28px;margin:0}}header p{{font-size:14px;line-height:20px;margin:8px 0}}
main{{position:relative;height:{far_y + 1500}px}}section{{position:absolute;left:55px;width:750px;padding:20px;border:2px solid #56616c;background:white;scroll-margin-top:330px}}
section h2{{font:22px/28px sans-serif;margin:0}}section p{{font:18px/24px sans-serif;margin:16px 0 0}}#near{{top:{near_y}px}}#near p{{margin-top:110px}}#chapter{{top:{far_y}px}}
aside{{position:fixed;z-index:4;right:12px;bottom:12px;width:235px;background:white;padding:8px}}aside input{{width:210px}}
</style></head><body data-owned-document="record">{"".join(boxes)}
<header><h1>{esc(spec["title"])}</h1><p>Chapter listing: {esc(local_heading)} in {esc(spec["local_trigger"])}. Related document: {esc(spec["remote_title"])} from {esc(spec["global_trigger"])}.</p></header>
<main><section id="near"><h2 id="near-heading">{esc(spec["near"])}</h2><p id="near-paragraph">{esc(spec["near_detail"])}</p></section>
<section id="chapter"><h2 id="far-heading">{esc(local_heading)}</h2><p id="far-paragraph">{esc(spec["local_detail"])}</p></section></main>
<aside><label for="note">Reading scratchpad {spec["layout_variant"] + 1}</label><input id="note" value="unchanged"></aside>
{event_script()}</body></html>"""


def event_script():
    return """<script>window.ownedEvents=JSON.parse(sessionStorage.getItem('ownedSemanticEvents')||'[]');for(const type of ['click','input','change','wheel'])document.addEventListener(type,e=>{ownedEvents.push({type,target:e.target.id,trusted:e.isTrusted,hash:location.hash,scrollY});sessionStorage.setItem('ownedSemanticEvents',JSON.stringify(ownedEvents));});</script>"""


PRIVATE = """(() => {
 const info=id=>{const e=document.getElementById(id);if(!e)return null;const r=e.getBoundingClientRect(),range=document.createRange();range.selectNodeContents(e);const rects=[...range.getClientRects()].filter(q=>q.width>0&&q.height>0);return {id,text:e.innerText,visible:e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true}),rect:{x:r.x,y:r.y,w:r.width,h:r.height},unoccluded_center:e.contains(document.elementFromPoint(r.x+r.width/2,r.y+r.height/2)),in_view:r.top>=0&&r.bottom<=innerHeight&&r.left>=0&&r.right<=innerWidth,complete_text_in_view:rects.length>0&&rects.every(q=>q.top>=0&&q.bottom<=innerHeight&&q.left>=0&&q.right<=innerWidth),text_unoccluded:rects.length>0&&rects.every(q=>[q.left+1,(q.left+q.right)/2,q.right-1].every(x=>e.contains(document.elementFromPoint(x,(q.top+q.bottom)/2))))};};
 const ids=['entry-local','entry-global','leaf-local','leaf-global','near-heading','near-paragraph','far-heading','far-paragraph'];
 return {url:location.href,title:document.title,document_scope:document.body.dataset.ownedDocument,ready:document.readyState,scroll_y:scrollY,width:innerWidth,height:innerHeight,elements:Object.fromEntries(ids.map(id=>[id,info(id)])),note:document.getElementById('note')?.value??null,focus:document.activeElement?.id,events:structuredClone(window.ownedEvents),html:document.documentElement.outerHTML};})()"""


def load_parent():
    path = ROOT / "scripts/jev_generic_menu_prototype.py"
    require(sha(path.read_bytes()) == PARENT_SHA, "Frozen parent source changed")
    name = "owned_semantic_menu_parent"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def auxiliary_tasks():
    """Closed decision frontier: initial and each singly opened menu.

    Four initial targets and five targets per open-menu state are exercised on
    separate fresh contexts. Recursive arbitrary UI state exploration is not
    claimed. Any additional offered target makes the certificate fail closed.
    """
    return [
        {"stage": stage, "probe": probe}
        for stage in ("initial", "local_open", "global_open")
        for probe in (
            ["local", "global", "focus", "fill"] + ([] if stage == "initial" else ["leaf"])
        )
    ]


def complete(element):
    return bool(
        element
        and element["visible"]
        and element["in_view"]
        and element["unoccluded_center"]
        and element["complete_text_in_view"]
        and element["text_unoccluded"]
    )


def assert_initial(spec, page, private):
    evidence = observable_contract(spec)
    require(private["width"] == 1120 and private["height"] == 780, "Exact viewport")
    require(
        private["document_scope"] == "record" and private["title"] == spec["title"],
        "Initial document identity",
    )
    require(private["note"] == "unchanged", "Initial unused input value")
    require(page["scroll"]["y"] == spec["prior_scrolls"] * 560, "True initial scroll history")
    require(
        all(text in page["text"] for text in evidence["expected_initial_text"]),
        "Native snapshot lacks declared observable clue",
    )
    require(
        all(text not in page["text"] for text in evidence["absent_initial_passages"]),
        "Initial required paragraph already visible",
    )
    require(
        complete(private["elements"]["near-heading"]), "Near heading must be completely visible"
    )
    require(
        not complete(private["elements"]["near-paragraph"]),
        "Near paragraph must remain below viewport",
    )
    require(
        not complete(private["elements"]["far-paragraph"]),
        "Far passage must not already be reached",
    )
    require(
        all(complete(private["elements"]["entry-" + scope]) for scope in ["local", "global"]),
        "Both native navigation entries must be visible",
    )
    require(
        not private["elements"]["leaf-local"]["visible"]
        and not private["elements"]["leaf-global"]["visible"],
        "Both popup leaves initially hidden",
    )


def assert_destination(spec, scope, private, origin):
    expected_path = "/" + spec["id"] + ("/library" if scope == "global" else "")
    parsed = urlsplit(private["url"])
    require(
        parsed.netloc == urlsplit(origin).netloc and parsed.path == expected_path,
        "Actual intended route",
    )
    require(
        private["document_scope"] == ("library" if scope == "global" else "record"),
        "Actual document scope",
    )
    require(
        private["title"] == spec["remote_title" if scope == "global" else "title"],
        "Actual document title",
    )
    heading = spec["shared"] if scope == "global" else spec.get("local_heading", spec["shared"])
    detail = spec["remote_detail" if scope == "global" else "local_detail"]
    require(
        private["elements"]["far-heading"]["text"] == heading
        and private["elements"]["far-paragraph"]["text"] == detail,
        "Exact independent destination passage",
    )
    require(
        complete(private["elements"]["far-heading"])
        and complete(private["elements"]["far-paragraph"]),
        "Complete destination heading and paragraph visible/unoccluded",
    )
    require(scope == "global" or parsed.fragment == "chapter", "Real local anchor navigation")


def native_step(browser, transport, page, action, history, *, text=None):
    before = browser.evaluate(PRIVATE)
    require(browser.fresh(page, action) is True, "Original fresh guard before input")
    first = len(transport.inputs)
    result = browser.act(action, page, text=text)
    require(result == {"executed": action["id"]}, "Original native executor confirmation")
    observed = browser.observe(screenshot=False)
    after = browser.evaluate(PRIVATE)
    require(after["width"] == 1120 and after["height"] == 780, "Native input changed viewport")
    inputs = copy.deepcopy(transport.inputs[first:])
    shape = [(row["method"], row["params"].get("type")) for row in inputs]
    expected = (
        [("Input.dispatchMouseEvent", "mouseWheel")]
        if action["kind"] == "scroll"
        else [
            ("Input.dispatchMouseEvent", "mousePressed"),
            ("Input.dispatchMouseEvent", "mouseReleased"),
        ]
    )
    if action["kind"] == "fill":
        expected += [
            ("Input.dispatchKeyEvent", "keyDown"),
            ("Input.dispatchKeyEvent", "keyUp"),
            ("Input.insertText", None),
        ]
    require(shape == expected, "Unexpected input or input replay")
    if action["kind"] == "fill":
        require(inputs[-1]["params"]["text"] == text, "Original inserted proof text")
    history.append(
        {
            "action": action["label"],
            "kind": action["kind"],
            "text": text,
            "page_changed": observed["fingerprint"] != page["fingerprint"],
        }
    )
    return observed, {
        "action": copy.deepcopy(action),
        "before": before,
        "after": after,
        "fresh": True,
        "executor_result": result,
        "inputs": inputs,
        "text": text,
    }


def probe_action(model, page, spec, probe, stage):
    _, targets, _ = model.action_space(page["actions"])
    if probe == "fill":
        operation = "TYPE_TEXT"
        matches = list(targets.get(operation, {}).items())
    else:
        operation = "CLICK"
        if probe in {"local", "global"}:
            matches = [
                (key, action)
                for key, action in targets.get(operation, {}).items()
                if action["label"] == spec[probe + "_trigger"]
            ]
        elif probe == "focus":
            matches = [
                (key, action)
                for key, action in targets.get(operation, {}).items()
                if action["role"] == "textbox"
            ]
        else:
            require(probe == "leaf" and stage != "initial", "Only actual opened-menu leaf")
            scope = stage.removesuffix("_open")
            label = (
                spec.get("local_heading", spec["shared"]) if scope == "local" else spec["shared"]
            )
            matches = [
                (key, action)
                for key, action in targets.get(operation, {}).items()
                if action["role"] == "link" and action["label"] == label
            ]
    require(len(matches) == 1, "One actual offered target for probe")
    target, action = matches[0]
    return (
        action,
        {"operation": operation, "target": target},
        {op.lower() + "_target": list(values) for op, values in targets.items()},
    )


class FixtureServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), FixtureHandler)
        self.documents = {}
        for spec in scenarios():
            self.documents["/" + spec["id"]] = render_fixture(spec).encode()
            self.documents["/" + spec["id"] + "/library"] = render_fixture(
                spec, remote=True
            ).encode()
        self.ledger = []


class FixtureHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.respond()

    def do_HEAD(self):
        self.respond(head_only=True)

    def respond(self, head_only=False):
        parsed = urlsplit(self.path)
        blob = self.server.documents.get(parsed.path)
        status = 200 if blob is not None and not parsed.query else 404
        self.server.ledger.append({"method": self.command, "path": self.path, "status": status})
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(blob) if status == 200 else 0))
        self.end_headers()
        if status == 200 and not head_only:
            self.wfile.write(blob)

    def log_message(self, *_args):
        pass


def run_episode(
    spec,
    mode,
    stage,
    probe,
    *,
    origin,
    chromium,
    harness,
    verified,
    upstream,
    compiler,
    checker,
    folder,
):
    parent = load_parent()
    folder.mkdir()
    calls, raw, inputs, requests, transitions, network, blocked, posts = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    proof = None
    primary_branch = (
        mode
        if mode in spec["goals"]
        else "menu"
        if spec["target_family"] != "control"
        else "control"
    )
    goal = spec["goals"][primary_branch]
    try:
        with chromium.new_context(
            viewport={"width": 1120, "height": 780}, service_workers="block"
        ) as context:

            def route_owned(route):
                request = route.request
                url = urlsplit(request.url)
                if (
                    request.method in {"GET", "HEAD"}
                    and url.scheme == "http"
                    and url.netloc == urlsplit(origin).netloc
                    and not url.username
                    and not url.password
                ):
                    route.continue_()
                else:
                    blocked.append({"method": request.method, "url": request.url})
                    route.abort("blockedbyclient")

            context.route("**/*", route_owned)
            context.on(
                "response",
                lambda response: network.append(
                    {
                        "url": response.url,
                        "method": response.request.method,
                        "status": response.status,
                    }
                ),
            )

            class Transport(harness.PlaywrightCDP):
                def cdp(self, method, session_id=None, **params):
                    row = {
                        "sequence": len(calls),
                        "method": method,
                        "session": session_id,
                        "params": copy.deepcopy(params),
                    }
                    calls.append(row)
                    try:
                        result = super().cdp(method, session_id=session_id, **params)
                        row["response"] = copy.deepcopy(result)
                        if method.startswith("Input."):
                            inputs.append(copy.deepcopy(row))
                        return result
                    except BaseException as exc:
                        row["error"] = {"type": type(exc).__name__, "message": str(exc)}
                        raise

            transport = Transport(context)
            transport.inputs = inputs
            transport.observation_expression = verified.read_state
            transport.observation_sink = lambda event: raw.append(copy.deepcopy(event))
            with harness.load_upstream(upstream, transport, observation_runtime=verified) as (
                _,
                model,
            ):

                def deny(*_args, **_kwargs):
                    posts.append(True)
                    raise AssertionError("Policy/helper POST and field helper forbidden")

                model.CLIENT.post = deny
                model.field_text = deny
                browser = sys.modules["jev_ultrafast.browser"].Browser(origin + "/" + spec["id"])
                history = []
                try:
                    page = browser.observe(screenshot=False)
                    for _ in range(spec["prior_scrolls"]):
                        page, step = native_step(
                            browser,
                            transport,
                            page,
                            parent.find_action(page, action_id="scroll_down"),
                            history,
                        )
                        transitions.append({"phase": "real_prior_scroll", **step})
                    initial = browser.evaluate(PRIVATE)
                    initial_raw_sequence = len(raw) - 1
                    require(initial_raw_sequence >= 0, "Actual raw initial observer required")
                    require(
                        all(
                            page.get(key) == value
                            for key, value in raw[initial_raw_sequence]["response"]["result"][
                                "value"
                            ].items()
                        ),
                        "Original raw/native page equality",
                    )
                    assert_initial(spec, page, initial)
                    initial_page, initial_history = copy.deepcopy(page), copy.deepcopy(history)

                    def request(phase):
                        payload = parent.capture_request(model, page, goal, history)
                        compiled = {"sequence": len(requests), **compiler.compile(payload)}
                        checker.check(payload, compiled)
                        requests.append(
                            {
                                "sequence": len(requests),
                                "phase": phase,
                                "native_payload": copy.deepcopy(payload),
                                "native_wire_sha256": sha(encode(payload)),
                                "compiled": compiled,
                            }
                        )
                        return payload

                    initial_payload = request("initial")
                    require(
                        len(initial_payload["questions"]["click_target"]["criteria"]) >= 3,
                        "Multiple actual initial CLICK candidates",
                    )
                    selected = None
                    stage_payload, stage_private, stage_raw_sequence = (
                        initial_payload,
                        initial,
                        initial_raw_sequence,
                    )
                    if mode == "auxiliary" and stage != "initial":
                        scope = stage.removesuffix("_open")
                        action, _, _ = probe_action(model, page, spec, scope, "initial")
                        page, step = native_step(browser, transport, page, action, history)
                        transitions.append({"phase": "fresh_clone_stage_setup", **step})
                        require(
                            complete(step["after"]["elements"]["leaf-" + scope]),
                            "Real setup reveals offered leaf",
                        )
                        stage_payload = request("single_menu_open")
                        stage_private, stage_raw_sequence = browser.evaluate(PRIVATE), len(raw) - 1
                    if mode in {"menu", "scroll", "control"}:
                        if mode == "menu":
                            scope = spec["target_family"]
                            require(
                                scope in {"local", "global"},
                                "Control cannot receive fabricated CLICK label",
                            )
                            action, selected, _ = probe_action(model, page, spec, scope, "initial")
                            page, step = native_step(browser, transport, page, action, history)
                            transitions.append({"phase": "initial_selected_input", **step})
                            require(
                                complete(step["after"]["elements"]["leaf-" + scope]),
                                "Actual selected entry reveals real leaf",
                            )
                            request("reachability_visible_leaf")
                            link, _, _ = probe_action(model, page, spec, "leaf", scope + "_open")
                            page, step = native_step(browser, transport, page, link, history)
                            transitions.append({"phase": "visible_leaf_input", **step})
                            assert_destination(spec, scope, step["after"], origin)
                        else:
                            action = parent.find_action(page, action_id="scroll_down")
                            selected = parent.selected_for(model, page, action)
                            page, step = native_step(browser, transport, page, action, history)
                            transitions.append({"phase": "initial_selected_input", **step})
                            require(
                                complete(step["after"]["elements"]["near-heading"])
                                and complete(step["after"]["elements"]["near-paragraph"]),
                                "One native wheel reveals complete near passage",
                            )
                            require(
                                step["after"]["title"] == spec["title"]
                                and step["after"]["document_scope"] == "record",
                                "Scroll preserves required document",
                            )
                            require(
                                step["after"]["elements"]["near-heading"]["text"] == spec["near"]
                                and step["after"]["elements"]["near-paragraph"]["text"]
                                == spec["near_detail"],
                                "Exact current-document requested near passage",
                            )
                    elif mode == "auxiliary":
                        action, selected, candidate_heads = probe_action(
                            model, page, spec, probe, stage
                        )
                        proof_text = "owned auxiliary focus proof" if probe == "fill" else None
                        page, step = native_step(
                            browser, transport, page, action, history, text=proof_text
                        )
                        transitions.append({"phase": "offered_target_fresh_clone", **step})
                        if probe in {"local", "global"}:
                            require(
                                complete(step["after"]["elements"]["leaf-" + probe]),
                                "Offered trigger's visible leaf",
                            )
                        elif probe == "leaf":
                            assert_destination(
                                spec, stage.removesuffix("_open"), step["after"], origin
                            )
                        elif probe == "focus":
                            require(
                                step["after"]["focus"] == "note"
                                and step["after"]["note"] == "unchanged",
                                "Offered input native focus",
                            )
                        else:
                            require(
                                step["after"]["focus"] == "note"
                                and step["after"]["note"] == proof_text,
                                "Offered input native fill",
                            )
                            require(
                                any(
                                    event["type"] == "input"
                                    and event["target"] == "note"
                                    and event["trusted"]
                                    for event in step["after"]["events"]
                                ),
                                "Genuine trusted native input event",
                            )
                    else:
                        require(
                            mode == "wheel_reference" and spec["target_family"] == "local",
                            "Only independent local wheel route reference",
                        )
                        count = 0
                        while not (
                            complete(browser.evaluate(PRIVATE)["elements"]["far-heading"])
                            and complete(browser.evaluate(PRIVATE)["elements"]["far-paragraph"])
                        ):
                            require(
                                count < 20,
                                "Finite owned-fixture wheel reference safety cap; not canonical16 or a label criterion",
                            )
                            if count:
                                request("wheel_only_reference")
                            page, step = native_step(
                                browser,
                                transport,
                                page,
                                parent.find_action(page, action_id="scroll_down"),
                                history,
                            )
                            transitions.append({"phase": "wheel_only_reference", **step})
                            count += 1
                        require(
                            count > 2,
                            "Claimed two-click/reference efficiency must be independently observed",
                        )
                        private = browser.evaluate(PRIVATE)
                        require(
                            private["title"] == spec["title"]
                            and private["elements"]["far-paragraph"]["text"]
                            == spec["local_detail"],
                            "Same actual local passage reached by reference",
                        )
                        require(
                            private["document_scope"] == "record"
                            and urlsplit(private["url"]).path == "/" + spec["id"]
                            and not urlsplit(private["url"]).fragment,
                            "Wheel-only reference must not jump or change document",
                        )
                    require(browser.fresh(page) is True, "Final native snapshot fresh")
                    final = browser.evaluate(PRIVATE)
                    require(
                        all(event["trusted"] for event in final["events"]),
                        "Only trusted native fixture events",
                    )
                    if mode == "menu":
                        require(
                            all(
                                any(
                                    event["type"] == "click"
                                    and event["target"] == target
                                    and event["trusted"]
                                    for event in final["events"]
                                )
                                for target in [
                                    "entry-" + spec["target_family"],
                                    "leaf-" + spec["target_family"],
                                ]
                            ),
                            "Trusted trigger and leaf navigation events",
                        )
                    elif mode == "auxiliary":
                        event_target = (
                            "entry-" + probe
                            if probe in {"local", "global"}
                            else "leaf-" + stage.removesuffix("_open")
                            if probe == "leaf"
                            else "note"
                        )
                        require(
                            any(
                                event["target"] == event_target and event["trusted"]
                                for event in final["events"]
                            ),
                            "Offered target receives genuine trusted event",
                        )
                    if not (mode == "auxiliary" and probe == "fill"):
                        require(final["note"] in {"unchanged", None}, "Unused input changed")
                    if mode != "wheel_reference":
                        require(
                            any(
                                row["params"].get("type") in {"mousePressed", "mouseWheel"}
                                for row in transitions[-1]["inputs"]
                            ),
                            "Actual native selected input required",
                        )
                    record = {
                        "scenario_id": spec["id"],
                        "split": spec["split"],
                        "episode_id": folder.name,
                        "branch": mode,
                        "html_sha256": sha(render_fixture(spec).encode()),
                        "history": initial_history,
                        "payload": initial_payload,
                        "selected": selected,
                        "reachability_verified": True,
                    }
                    proof = {
                        "schema": "owned-semantic-native-episode-proof-v1",
                        "mode": mode,
                        "stage": stage,
                        "probe": probe,
                        "record": record,
                        "initial_private": initial,
                        "initial_native_page": initial_page,
                        "initial_raw_sequence": initial_raw_sequence,
                        "stage_private": stage_private,
                        "stage_native_payload": stage_payload,
                        "stage_raw_sequence": stage_raw_sequence,
                        "stage_candidate_heads": candidate_heads if mode == "auxiliary" else None,
                        "final_native_page": page,
                        "final_private": final,
                        "transitions": transitions,
                        "requests": requests,
                        "inputs": inputs,
                        "raw_observations": raw,
                        "zero_policy_helper_POSTs": not posts,
                        "blocked_requests": blocked,
                        "ownership": {
                            "target": browser.target,
                            "session": browser.session,
                            "native_page_key": initial_page["page_key"],
                            "url": initial_page["url"],
                        },
                        "positive_supervision": mode in {"menu", "scroll", "control"},
                    }
                finally:
                    browser.close()
                    model.CLIENT.close()
                require(
                    not posts and not blocked and all(row["status"] == 200 for row in network),
                    "Only successful owned GET/HEAD, no model/helper POST",
                )
                require(not transport.observation_profile_errors, "Frozen profile integrity")
        write(folder / "proof.json", proof)
        write_rows(folder / "native-cdp.jsonl", calls)
        write_rows(folder / "observations.jsonl", raw)
        write_rows(folder / "private-dom.jsonl", [initial, final])
        write_rows(folder / "transitions.jsonl", transitions)
        write_rows(folder / "ownership.jsonl", [proof["ownership"]])
        write_rows(folder / "native-requests.jsonl", requests)
        write(
            folder / "local-request-ledger.json",
            {"scope": "owned loopback only", "responses": network, "blocked": blocked},
        )
        return proof
    except BaseException as exc:
        write(
            folder / "failure.json",
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "positive_annotation_published": False,
            },
        )
        write_rows(folder / "failed-native-cdp.jsonl", calls)
        write_rows(folder / "failed-observations.jsonl", raw)
        write_rows(folder / "failed-native-requests.jsonl", requests)
        write_rows(folder / "failed-transitions.jsonl", transitions)
        write(
            folder / "failed-local-request-ledger.json",
            {"responses": network, "blocked": blocked, "policy_helper_POST_attempts": len(posts)},
        )
        raise


def coverage_certificate(proofs):
    expected_tasks = {(task["stage"], task["probe"]) for task in auxiliary_tasks()}
    certificates = []
    for spec in scenarios():
        selected = [proof for proof in proofs if proof["record"]["scenario_id"] == spec["id"]]
        require(
            len(selected) == len(expected_tasks)
            and {(p["stage"], p["probe"]) for p in selected} == expected_tasks,
            "Complete fresh-clone task population",
        )
        for stage in ["initial", "local_open", "global_open"]:
            current = [proof for proof in selected if proof["stage"] == stage]
            require(
                all(
                    proof["mode"] == "auxiliary"
                    and proof["positive_supervision"] is False
                    and proof["record"]["reachability_verified"] is True
                    for proof in current
                ),
                "Auxiliary proof is not supervision",
            )
            first = current[0]
            payload = first["stage_native_payload"]
            offered = {
                name: list(question["criteria"])
                for name, question in payload["questions"].items()
                if name != "operation"
            }
            require(
                set(offered) == {"click_target", "type_text_target"},
                "Unexpected target head cannot be omitted",
            )
            expected = {(name, target) for name, targets in offered.items() for target in targets}
            actual = set()
            for proof in current:
                require(
                    encode(proof["stage_native_payload"]) == encode(payload),
                    "Identical full ordered stage payload on every fresh clone",
                )
                require(
                    proof["stage_private"]["html"].encode()
                    == first["stage_private"]["html"].encode(),
                    "Identical actual stage DOM on fresh clones",
                )
                require(
                    encode(proof["stage_candidate_heads"]) == encode(offered),
                    "All actual native offered target ids/order",
                )
                action = proof["record"]["selected"]
                head = {"CLICK": "click_target", "TYPE_TEXT": "type_text_target"}[
                    action["operation"]
                ]
                key = (head, action["target"])
                require(
                    key not in actual and key in expected, "No duplicated/unoffered coverage target"
                )
                actual.add(key)
                require(
                    proof["transitions"][-1]["fresh"] is True
                    and proof["transitions"][-1]["inputs"],
                    "Fresh actual target input evidence",
                )
            require(
                actual == expected,
                "Every offered decision-frontier target requires its own fresh-clone proof",
            )
            certificates.append(
                {
                    "scene_id": spec["id"],
                    "stage": stage,
                    "native_payload_sha256": sha(encode(payload)),
                    "actual_stage_DOM_sha256": sha(first["stage_private"]["html"].encode()),
                    "offered_target_heads": offered,
                    "covered_targets": [
                        [head, target] for head, targets in offered.items() for target in targets
                    ],
                    "fresh_clone_proofs": [proof["record"]["episode_id"] for proof in current],
                    "positive_supervision": False,
                }
            )
    return certificates


def lexical_and_split_audit():
    rows = scenarios()
    goals = [(goal, branch) for row in rows for branch, goal in row["goals"].items()]
    cues = [
        "separate",
        "reference",
        "this",
        "current",
        "required",
        "find",
        "visible",
        "passage",
        "paragraph",
        "view",
    ]
    return {
        "scope": "Descriptive source audit; this does not prove absence of linguistic shortcuts.",
        "goal_count": len(goals),
        "cue_occurrence_by_teacher_operation": {
            cue: {
                operation: sum(
                    bool(re.search(r"\b" + cue + r"\b", goal.lower()))
                    and (branch == "menu") == (operation == "CLICK")
                    for goal, branch in goals
                )
                for operation in ["CLICK", "SCROLL_DOWN"]
            }
            for cue in cues
        },
        "family_target_balance": {
            family: dict(
                Counter(
                    row["target_family"]
                    for row in rows
                    if row["family"] == family and row["target_family"] != "control"
                )
            )
            for family in sorted({row["family"] for row in rows})
        },
        "split_groups": {
            split: {
                "scene_ids": [row["id"] for row in rows if row["split"] == split],
                "goal_families": sorted({row["family"] for row in rows if row["split"] == split}),
                "layout_families": sorted(
                    {row["layout_family"] for row in rows if row["split"] == split}
                ),
            }
            for split in ["train", "development"]
        },
        "residual_risks": [
            "Small authored population, correlated content names and sentence length.",
            "Only three goal discourse families and three layout families; unseen family does not establish broad semantics.",
            "A visible chapter listing is intended task evidence; teacher-specific operation ids never enter that listing.",
            "Human independent review of every goal, visible scope and counterexample remains necessary.",
        ],
    }


@contextmanager
def resources(output, upstream, chromium_executable):
    parent = load_parent()
    require(
        not any(name.split(".")[0] in parent.FORBIDDEN for name in sys.modules),
        "No model package resident in proof interpreter",
    )
    parent.memory_admission()
    reject = parent.RejectModels()
    sys.meta_path.insert(0, reject)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    server = thread = owned = None
    try:
        import jev_browser_collect_public_training as collector
        import jev_browser_profile_v5_benchmark as wrapper
        import jev_observation_profile_v5 as profile
        from playwright.sync_api import sync_playwright

        from s1.browser_public_training_validation import _CachedNativeCompiler

        require(
            sha(Path(collector.__file__).read_bytes()) == COLLECTOR_SHA
            and sha(Path(profile.__file__).read_bytes()) == PROFILE_SHA
            and sha(Path(wrapper.__file__).read_bytes()) == WRAPPER_SHA,
            "Frozen collector/profile/wrapper source identity",
        )
        metadata = {
            name: {
                "sha256": sha((collector.CACHE / name).read_bytes()),
                "size_bytes": (collector.CACHE / name).stat().st_size,
            }
            for name in collector.ASSET_SHA
        }
        require(
            {name: value["sha256"] for name, value in metadata.items()} == collector.ASSET_SHA,
            "Existing offline metadata closure",
        )
        compiler, checker = (
            collector.NativeCompiler(),
            _CachedNativeCompiler(collector.CACHE, metadata),
        )
        bundle, manifest = wrapper.prepare_bundle(output / "runner", case_module="original")
        prepared = profile.materialize_profile(
            upstream, output / "profile", expected_source_sha256=PROFILE_SHA
        )
        verified = profile.verify_materialized_profile(
            output / "profile",
            expected_manifest_sha256=prepared["manifest_sha256"],
            expected_source_sha256=PROFILE_SHA,
        )
        spec = importlib.util.spec_from_file_location(
            "owned_semantic_v5_runner", bundle / "runner.py"
        )
        harness = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = harness
        spec.loader.exec_module(harness)
        require(
            harness.upstream_manifest(upstream)["source_sha256"] == dict(profile.CANONICAL_SHA256),
            "Exact pinned canonical Git source",
        )
        server = FixtureServer()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        origin = f"http://127.0.0.1:{server.server_port}"
        with sync_playwright() as runtime:
            owned = runtime.chromium.launch(
                executable_path=str(chromium_executable),
                headless=True,
                args=[
                    "--disable-gpu",
                    "--disable-software-rasterizer",
                    "--disable-dev-shm-usage",
                    "--disable-background-networking",
                ],
            )
            try:
                yield {
                    "origin": origin,
                    "chromium": owned,
                    "harness": harness,
                    "verified": verified,
                    "upstream": upstream,
                    "compiler": compiler,
                    "checker": checker,
                    "metadata": metadata,
                    "profile_manifest_sha256": prepared["manifest_sha256"],
                }
                wrapper.verify_bundle(bundle, manifest)
                profile.verify_materialized_profile(
                    output / "profile",
                    expected_manifest_sha256=prepared["manifest_sha256"],
                    expected_source_sha256=PROFILE_SHA,
                )
            finally:
                owned.close()
                owned = None
        write(output / "local-request-ledger.json", server.ledger)
        require(
            all(row["method"] in {"GET", "HEAD"} and row["status"] == 200 for row in server.ledger),
            "Successful exact owned fixture routes only",
        )
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
            if not (output / "local-request-ledger.json").exists():
                write(output / "local-request-ledger.json", server.ledger)
        if thread is not None:
            thread.join(timeout=2)
            require(not thread.is_alive(), "Owned fixture server must exit")
        sys.dont_write_bytecode = previous
        sys.meta_path.remove(reject)


def run_prototype(output, *, expected_source_sha256, upstream=None, chromium_executable=None):
    require(
        sha(Path(__file__).read_bytes()) == expected_source_sha256,
        "Reviewed semantic producer source",
    )
    parent = load_parent()
    output = Path(output).absolute()
    require(
        not output.exists() and not any(path.is_symlink() for path in (output, *output.parents)),
        "Fresh ordinary proof output",
    )
    parent.memory_admission()
    output.mkdir(parents=True)
    upstream = Path(upstream or ROOT / "artifacts/jev-runtime/upstream")
    chromium_executable = Path(chromium_executable or parent.CHROMIUM)
    write(
        output / "started.json",
        {
            "schema": "owned-semantic-menu-started-v1",
            "source_sha256": expected_source_sha256,
            "specification": specification(),
            "memory": parent.memory_admission(),
            "scope": "No model/fit/public admission",
        },
    )
    primary, auxiliary, references, pairs, cases, wires = [], [], [], [], [], []
    try:
        with resources(output, upstream, chromium_executable) as environment:
            kwargs = {
                key: value
                for key, value in environment.items()
                if key not in {"metadata", "profile_manifest_sha256"}
            }
            for scene in scenarios():
                parent.memory_admission()
                current = []
                for branch in scene["goals"]:
                    proof = run_episode(
                        scene,
                        branch,
                        "initial",
                        None,
                        **kwargs,
                        folder=output / (scene["id"] + "-" + branch),
                    )
                    primary.append(proof)
                    current.append(proof)
                if len(current) == 2:
                    pair = parent.validate_pair(*(proof["record"] for proof in current))
                    require(
                        current[0]["initial_private"]["html"].encode()
                        == current[1]["initial_private"]["html"].encode(),
                        "Actual goal-only initial DOM bytes",
                    )
                    pair.update(
                        actual_initial_DOM_bytes_equal=True,
                        actual_initial_DOM_sha256=sha(
                            current[0]["initial_private"]["html"].encode()
                        ),
                    )
                    pairs.append(pair)
                for task in auxiliary_tasks():
                    proof = run_episode(
                        scene,
                        "auxiliary",
                        **task,
                        **kwargs,
                        folder=output
                        / (scene["id"] + "-aux-" + task["stage"] + "-" + task["probe"]),
                    )
                    require(
                        parent.normalized_pair_payload(proof["record"]["payload"])
                        == parent.normalized_pair_payload(current[0]["record"]["payload"]),
                        "Auxiliary clone preserves original initial DOM/history/native payload",
                    )
                    auxiliary.append(proof)
                if scene["target_family"] == "local":
                    reference = run_episode(
                        scene,
                        "wheel_reference",
                        "initial",
                        None,
                        **kwargs,
                        folder=output / (scene["id"] + "-wheel-reference"),
                    )
                    require(
                        parent.normalized_pair_payload(reference["record"]["payload"])
                        == parent.normalized_pair_payload(current[0]["record"]["payload"]),
                        "Independent wheel reference identical initialization",
                    )
                    references.append(reference)
            certificate = coverage_certificate(auxiliary)
            require(
                len(primary) == 14
                and len(auxiliary) == 112
                and len(references) == 3
                and len(pairs) == 6,
                "Complete predeclared population; no selection",
            )
            # No DTO/teacher rows are emitted until all selected routes and every
            # declared decision-frontier distractor proof have succeeded.
            for proof in primary:
                record = proof["record"]
                folder = output / record["episode_id"]

                def reference(name, sequence):
                    return {
                        "artifact_path": folder.name + "/" + name,
                        "sha256": sha((folder / name).read_bytes()),
                        "sequence": sequence,
                    }

                refs = {
                    "raw_observation": reference(
                        "observations.jsonl", proof["initial_raw_sequence"]
                    ),
                    "private_precondition": reference("private-dom.jsonl", 0),
                    "guarded_transition": reference(
                        "transitions.jsonl",
                        next(
                            i
                            for i, transition in enumerate(proof["transitions"])
                            if transition["phase"] == "initial_selected_input"
                        ),
                    ),
                    "private_postcondition": reference("private-dom.jsonl", 1),
                    "ownership": reference("ownership.jsonl", 0),
                    "public_ledger": reference("local-request-ledger.json", 0),
                }
                case, wire = parent.build_case_and_wire(record, refs)
                cases.append(case)
                wires.append(wire)
            from s1.browser_public_training_validation import public_compile_wire_projection

            projection = public_compile_wire_projection(cases, wires)
            require(len(projection) == 14, "Syntax bridge complete14, not public admission")
            for case, wire, projected in zip(cases, wires, projection, strict=True):
                require(
                    encode(case.request.model_dump(mode="json"))
                    == encode(wire["native_payload"])
                    == encode(projected["payload"]),
                    "Exact native wire through DTO/compiler projection",
                )
            write_rows(
                output / "compatibility-cases.jsonl",
                [case.model_dump(mode="json") for case in cases],
            )
            write_rows(output / "compatibility-wire-requests.jsonl", wires)
            write(output / "coverage-certificate.json", certificate)
            write(output / "pair-proof.json", pairs)
            write(output / "lexical-and-split-audit.json", lexical_and_split_audit())
            grains = {}
            for split in ["train", "development"]:
                chosen = [case for case in cases if case.split == split]
                proof_rows = [
                    proof
                    for proof in primary + auxiliary + references
                    if proof["record"]["split"] == split
                ]
                requests = [request for proof in proof_rows for request in proof["requests"]]
                grains[split] = {
                    "initial_supervision_requests": len(chosen),
                    "initial_offered_heads": sum(
                        len(case.request.candidate_labels) for case in chosen
                    ),
                    "initial_supervised_heads": sum(len(case.gold) for case in chosen),
                    "all_evidence_episodes": len(proof_rows),
                    "all_native_wire_captures": len(requests),
                    "all_offered_heads": sum(
                        len(request["native_payload"]["questions"]) for request in requests
                    ),
                }
            receipt = {
                "schema": "owned-semantic-menu-prototype-receipt-v1",
                "status": "complete_local_proof_not_public_admission",
                "source_sha256": expected_source_sha256,
                "parent_sha256": PARENT_SHA,
                "profile": PROFILE,
                "profile_source_sha256": PROFILE_SHA,
                "profile_manifest_sha256": environment["profile_manifest_sha256"],
                "compiler_metadata": environment["metadata"],
                "scene_count": 8,
                "primary_episode_count": 14,
                "auxiliary_fresh_clone_count": 112,
                "wheel_reference_count": 3,
                "all_episode_count": 129,
                "pair_count": 6,
                "candidate_frontier": "All target heads in initial and singly-opened local/global menu decision requests; no recursive arbitrary UI-state coverage claim",
                "grains": grains,
                "zero_policy_helper_POSTs": True,
                "model_or_fit_calls": 0,
                "external_network_calls": 0,
                "public_admitted": False,
                "fit_bundle_created": False,
                "teacher_semantics": specification()["teacher_policy"],
                "efficiency_comparisons": [
                    {
                        "scene_id": proof["record"]["scenario_id"],
                        "demonstrated_menu_inputs": 2,
                        "observed_wheel_reference_inputs": sum(
                            step["phase"] == "wheel_only_reference" for step in proof["transitions"]
                        ),
                        "shortest_path_claim": False,
                    }
                    for proof in references
                ],
            }
        require(
            sha(Path(__file__).read_bytes()) == expected_source_sha256,
            "Producer source stable throughout proof",
        )
        write(output / "receipt.json", receipt)
        return receipt
    except BaseException as exc:
        write(
            output / "failure.json",
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "positive_fit_rows_published": False,
                "complete_population_required": True,
            },
        )
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--chromium", type=Path)
    args = parser.parse_args(argv)
    require(sha(Path(__file__).read_bytes()) == args.source_sha256, "Reviewed producer source SHA")
    if args.run:
        run_prototype(
            args.output,
            expected_source_sha256=args.source_sha256,
            upstream=args.upstream,
            chromium_executable=args.chromium,
        )
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        write(args.output / "specification.json", specification())
        write(args.output / "source-only-audit.json", lexical_and_split_audit())
        print(
            json.dumps(
                {
                    "status": "specification_only_no_browser",
                    "goals": 14,
                    "browser_calls": 0,
                    "model_calls": 0,
                    "output": str(args.output),
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
