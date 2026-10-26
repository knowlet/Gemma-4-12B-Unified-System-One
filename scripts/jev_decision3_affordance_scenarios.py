"""New owned, train-only affordance scenes; no model or benchmark data.

The unchanged publisher observes the HTML and constructs every policy request.
Scenario/ancestor IDs and teacher annotations belong outside those requests.
This module supplies authored routes and DOM predicates, not measured answers.
"""

from __future__ import annotations

import copy
import hashlib
import html
import json
import math
import unicodedata

SCHEMA = "decision3-owned-affordance-scenarios-v1"
TRANSITION_SCHEMA = "decision3-owned-affordance-transition-v1"
NAMESPACE = "decision3-owned-affordance-training-v1"
GOAL_NORMALIZATION = "NFKC-collapse-whitespace-v1"
FAMILIES = ("compact-one", "compact-two", "form-top", "form-middle")


def _require(value, message):
    if not value:
        raise ValueError(message)


def _goal_sha256(goal):
    """Same NFKC/whitespace algorithm as browser_training_counterfactuals."""
    _require(type(goal) is str and bool(goal.strip()), "Nonempty authored goal required")
    normalized = " ".join(unicodedata.normalize("NFKC", goal).split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def _exclusions(values):
    _require(
        isinstance(values, (list, tuple, set, frozenset)) and bool(values),
        "A nonempty exclusion-only goal-hash inventory is required",
    )
    _require(
        all(
            type(value) is str
            and len(value) == 64
            and all(char in "0123456789abcdef" for char in value)
            for value in values
        ),
        "Exclusions must be lowercase SHA-256 goal hashes",
    )
    _require(len(set(values)) == len(values), "Distinct goal-hash exclusions required")
    return frozenset(values)


def _check_spec(spec):
    _require(type(spec) is dict, "Detached authored scenario dictionary required")
    _require(
        spec.get("schema_id") == SCHEMA
        and spec.get("namespace") == NAMESPACE
        and spec.get("split") == "train"
        and spec.get("family") in FAMILIES,
        "Only this new owned train-only scenario namespace is supported",
    )
    _require(
        type(spec.get("seed")) is int and spec["seed"] >= 0,
        "Nonnegative authored seed required",
    )
    _require(
        spec.get("id") == f"d3-affordance-train-{spec['seed']}-{spec['family']}"
        and spec.get("ancestor_id") == f"d3-affordance-owned-{spec['seed']}-{spec['family']}",
        "New owned scenario and ancestor identities required",
    )
    _require(
        spec.get("goal_sha256") == _goal_sha256(spec.get("goal")),
        "Authored goal normalization/hash mismatch",
    )
    expected = [4] if spec["family"].startswith("compact-") else [7]
    if spec["family"] == "form-middle":
        expected = [7, 8]
    _require(spec.get("expected_menu_counts") == expected, "Natural menu coverage changed")
    _require(
        type(spec.get("title")) is str and bool(spec["title"].strip()),
        "Visible authored title required",
    )
    if spec["family"].startswith("compact-"):
        buttons = spec.get("buttons")
        _require(
            type(buttons) is list
            and len(buttons) == (1 if spec["family"] == "compact-one" else 2)
            and [button.get("id") for button in buttons]
            == ["primary-control", "secondary-control"][: len(buttons)]
            and all(type(button.get("label")) is str and button["label"] for button in buttons),
            "Exact compact observed controls required",
        )
        _require(
            spec.get("desired_button_label") in [button["label"] for button in buttons]
            and spec["desired_button_label"] in spec["goal"],
            "Compact goal must name its visible control",
        )
    else:
        options = spec.get("options")
        _require(
            type(options) is list
            and len(options) == 3
            and len({option["value"] for option in options}) == 3
            and all(
                type(option.get("value")) is str
                and bool(option["value"])
                and type(option.get("label")) is str
                and bool(option["label"])
                for option in options
            ),
            "Three real alternative select values required",
        )
        _require(
            {
                "value": spec.get("desired_option_value"),
                "label": spec.get("desired_option_label"),
            }
            in options
            and type(spec.get("desired_text")) is str
            and bool(spec["desired_text"])
            and spec["desired_text"] in spec["goal"]
            and spec["desired_option_label"] in spec["goal"],
            "Field values must be supplied by the plain authored goal",
        )
        _require(
            spec.get("paragraph_required") is (spec["family"] == "form-middle"),
            "Authored paragraph requirement changed",
        )


def scenarios(seed, excluded_goal_hashes):
    """Return four detached train-only scene specifications or refuse a collision."""
    _require(type(seed) is int and seed >= 0, "Nonnegative integer scenario seed required")
    excluded = _exclusions(excluded_goal_hashes)
    option_pool = [
        {"value": "ribbon-wrap", "label": "Ribbon wrap"},
        {"value": "cord-wrap", "label": "Cord wrap"},
        {"value": "sleeve-wrap", "label": "Sleeve wrap"},
    ]
    rotated = option_pool[seed % 3 :] + option_pool[: seed % 3]
    result = []
    for index, family in enumerate(FAMILIES):
        spec = {
            "schema_id": SCHEMA,
            "namespace": NAMESPACE,
            "id": f"d3-affordance-train-{seed}-{family}",
            "ancestor_id": f"d3-affordance-owned-{seed}-{family}",
            "seed": seed,
            "split": "train",
            "family": family,
            "title": ("Quill signal", "Pebble signal", "Envelope docket", "Parcel docket")[index],
            "expected_menu_counts": [4] if index < 2 else [7] if index == 2 else [7, 8],
            "buttons": [],
            "desired_button_label": None,
            "field_label": "Delivery note",
            "select_label": "Handling mode",
            "submit_label": "Record entry",
            "options": copy.deepcopy(rotated),
            "desired_text": None,
            "desired_option_value": None,
            "desired_option_label": None,
            "paragraph_required": family == "form-middle",
            "paragraph_heading": "Packing paragraph",
            "paragraph_text": (
                "The storage sleeve rests flat beside the docket. Its folded edge remains "
                "visible, and the packing record stays with the parcel until dispatch."
            ),
        }
        if index < 2:
            labels = ("Raise pennant", "Ring marker")
            spec["buttons"] = [
                {"id": "primary-control", "label": labels[0]},
                *([{"id": "secondary-control", "label": labels[1]}] if index else []),
            ]
            spec["desired_button_label"] = labels[(seed % 2) if index else 0]
            spec["goal"] = (
                f"On the {spec['title']} page, activate the {spec['desired_button_label']} "
                "control. Finish when the page "
                "reports that control as activated."
            )
        else:
            chosen = rotated[(seed + index) % len(rotated)]
            spec["desired_text"] = f"parcel memo {seed % 101 + 11}-{index}"
            spec["desired_option_value"] = chosen["value"]
            spec["desired_option_label"] = chosen["label"]
            prefix = (
                "First scroll down until the whole Packing paragraph is visible, and keep "
                "that paragraph visible while completing the entry. "
                if spec["paragraph_required"]
                else ""
            )
            spec["goal"] = (
                prefix + f"Enter {spec['desired_text']} in Delivery note, choose {chosen['label']} "
                "in Handling mode, then click Record entry. Finish when the submitted "
                "entry shows that note and handling mode."
            )
        spec["goal_sha256"] = _goal_sha256(spec["goal"])
        _require(spec["goal_sha256"] not in excluded, "Authored goal collides with an exclusion")
        _check_spec(spec)
        result.append(spec)
    _require(len({item["goal_sha256"] for item in result}) == 4, "Four distinct authored goals")
    return tuple(copy.deepcopy(result))


_EVENT_SCRIPT = """<script>
window.__d3AffordanceEvents=[];
for(const type of ['click','input','change','wheel'])document.addEventListener(type,e=>{
 const t=e.target;
 window.__d3AffordanceEvents.push({sequence:window.__d3AffordanceEvents.length,
  type,target:t?.id||'',trusted:e.isTrusted,label:t?.getAttribute?.('aria-label')||t?.textContent?.trim()||'',
  value:t&&'value'in t?String(t.value):null,scroll_y:scrollY,
  delta_y:type==='wheel'?e.deltaY:null});
},true);
for(const b of document.querySelectorAll('[data-activation-control]'))b.addEventListener('click',()=>{
 const s=document.getElementById('activation-status');
 s.dataset.activatedLabel=b.textContent.trim();s.textContent='Activated: '+b.textContent.trim()+'.';
});
const note=document.getElementById('entry-text'),mode=document.getElementById('entry-select');
const summary=document.getElementById('submission-summary'),status=document.getElementById('progress-status');
if(note&&mode&&summary){
 const edited=()=>{summary.dataset.submitted='false';status.textContent='Edited; submit the entry.';};
 note.addEventListener('input',edited);mode.addEventListener('input',edited);mode.addEventListener('change',edited);
 document.getElementById('commit-control').addEventListener('click',()=>{
  summary.dataset.submitted='true';summary.dataset.count=String(Number(summary.dataset.count)+1);
  document.getElementById('submitted-note').textContent=note.value;
  const selected=mode.selectedOptions[0];
  const recorded=document.getElementById('submitted-mode');
  recorded.textContent=selected?.label||'';recorded.dataset.value=mode.value;
  status.textContent='Submitted entry '+summary.dataset.count+'.';
 });
}
</script>"""


def render_fixture(spec):
    """Render app content without scenario IDs, hidden goals or desired field text."""
    _check_spec(spec)
    esc = html.escape
    title = esc(spec["title"])
    head = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{title}</title>"
    )
    if spec["family"].startswith("compact-"):
        buttons = "".join(
            f"<button type='button' id='{esc(button['id'], quote=True)}' data-activation-control>"
            f"{esc(button['label'])}</button>"
            for button in spec["buttons"]
        )
        body = (
            "<style>*{box-sizing:border-box}html,body{margin:0;width:100%;height:100%;"
            "overflow:hidden}body{font:20px sans-serif;display:grid;place-items:center;"
            "background:#f0f3f5}.panel{padding:30px;background:white;border:2px solid #425464;"
            "max-width:850px}button{font:inherit;margin:10px;padding:14px}output{display:block;"
            "margin-top:24px}</style></head><body><main class='panel'>"
            f"<h1>{title}</h1><p>Controls remain available after activation.</p>{buttons}"
            "<output id='activation-status' data-activated-label=''>No control activated.</output>"
            "</main>"
        )
    else:
        options = "".join(
            f"<option value='{esc(option['value'], quote=True)}'>{esc(option['label'])}</option>"
            for option in spec["options"]
        )
        body = (
            "<style>*{box-sizing:border-box}html{scroll-behavior:auto}body{margin:0;"
            "font:18px sans-serif;background:#f1f4ed}#toolbar{position:fixed;top:0;left:0;"
            "right:0;height:126px;padding:12px 18px;background:white;border-bottom:2px solid "
            "#405565;z-index:10}h1{font-size:20px;line-height:24px;margin:0}.controls{display:"
            "flex;gap:18px;align-items:end;margin-top:8px}label{display:flex;flex-direction:"
            "column;font-size:14px;line-height:18px}input,select,button{font:16px sans-serif;"
            "height:28px}input{width:300px}select{width:230px}button{padding:0 18px}"
            "#submission-summary{font-size:13px;line-height:16px;margin-top:6px}"
            "#page-track{position:relative;min-height:2300px}#required-passage{position:absolute;"
            "top:720px;left:60px;width:680px;height:180px;padding:20px;border:2px solid #405565;"
            "background:white}#required-passage h2{font-size:22px;margin:0 0 12px}"
            "#required-passage p{margin:0;line-height:26px}</style></head><body>"
            f"<header id='toolbar'><h1>{title}</h1><div class='controls'>"
            f"<label>{esc(spec['field_label'])}<input id='entry-text' type='text' "
            f"aria-label='{esc(spec['field_label'], quote=True)}' value=''></label>"
            f"<label>{esc(spec['select_label'])}<select id='entry-select' "
            f"aria-label='{esc(spec['select_label'], quote=True)}'>"
            "<option value='' selected disabled>Choose a handling mode</option>"
            f"{options}</select></label><button type='button' id='commit-control'>"
            f"{esc(spec['submit_label'])}</button></div>"
            "<div id='submission-summary' data-submitted='false' data-count='0'>"
            "<span id='progress-status'>No entry submitted.</span> "
            "<span>Recorded note: <output id='submitted-note'></output></span> "
            "<span>Recorded mode: <output id='submitted-mode' data-value=''></output></span>"
            "</div></header><main id='page-track'>"
            f"<section id='required-passage'><h2>{esc(spec['paragraph_heading'])}</h2>"
            f"<p>{esc(spec['paragraph_text'])}</p></section></main>"
        )
    return head + body + _EVENT_SCRIPT + "</body></html>"


PRIVATE = """(() => {
 const el=id=>document.getElementById(id),text=id=>el(id)?.textContent??null;
 const p=el('required-passage');let paragraphVisible=false;
 if(p){const r=p.getBoundingClientRect(),range=document.createRange();range.selectNodeContents(p);
  const rects=[...range.getClientRects()].filter(q=>q.width>0&&q.height>0);
  paragraphVisible=p.checkVisibility({checkOpacity:true,checkVisibilityCSS:true})&&r.top>=0&&
   r.bottom<=innerHeight&&r.left>=0&&r.right<=innerWidth&&rects.length>0&&
   rects.every(q=>q.top>=0&&q.bottom<=innerHeight&&q.left>=0&&q.right<=innerWidth&&
    [q.left+1,(q.left+q.right)/2,q.right-1].every(x=>p.contains(document.elementFromPoint(x,(q.top+q.bottom)/2))));
 }
 const note=el('entry-text'),mode=el('entry-select'),summary=el('submission-summary');
 const activation=el('activation-status'),selected=mode?.selectedOptions?.[0];
 return {title:document.title,ready:document.readyState,url:location.href,
  viewport_width:innerWidth,viewport_height:innerHeight,document_height:document.documentElement.scrollHeight,
  scroll_y:scrollY,status:text('progress-status')??text('activation-status'),
  activated_label:activation?.dataset.activatedLabel||null,
  note_value:note?.value??null,mode_value:mode?.value??null,mode_label:selected?.label??null,
  submitted:summary?.dataset.submitted==='true',
  submitted_note:text('submitted-note'),submitted_mode_label:text('submitted-mode'),
  submitted_mode_value:el('submitted-mode')?.dataset.value??null,
  submission_count:summary?Number(summary.dataset.count):0,
  paragraph_visible:paragraphVisible,paragraph_text:p?.innerText??null,
  events:structuredClone(window.__d3AffordanceEvents||[]),html:document.documentElement.outerHTML};
})()"""


def private_state_expression():
    return PRIVATE


def _descriptor(operation, kind, label=None, action_id=None, text=None):
    return {
        "operation": operation,
        "kind": kind,
        "label": label,
        "action_id": action_id,
        "text": text,
    }


def _paragraph_complete(spec, state):
    text = state.get("paragraph_text")
    expected = spec["paragraph_heading"] + " " + spec["paragraph_text"]
    return (
        state.get("paragraph_visible") is True
        and type(text) is str
        and " ".join(text.split()) == " ".join(expected.split())
    )


def verify_goal(spec, private_state):
    """Evaluate actual DOM state against the authored goal, without model predictions."""
    _check_spec(spec)
    state = private_state
    if type(state) is not dict or state.get("title") != spec["title"]:
        return False
    if spec["family"].startswith("compact-"):
        label = spec["desired_button_label"]
        return (
            state.get("activated_label") == label and state.get("status") == f"Activated: {label}."
        )
    count = state.get("submission_count")
    return (
        state.get("submitted") is True
        and type(count) is int
        and count > 0
        and state.get("status") == f"Submitted entry {count}."
        and state.get("note_value") == state.get("submitted_note") == spec["desired_text"]
        and state.get("mode_value")
        == state.get("submitted_mode_value")
        == spec["desired_option_value"]
        and state.get("mode_label")
        == state.get("submitted_mode_label")
        == spec["desired_option_label"]
        and (not spec["paragraph_required"] or _paragraph_complete(spec, state))
    )


def teacher_step(spec, private_state):
    """Choose one authored useful route; no unique/global-shortest claim is made."""
    _check_spec(spec)
    _require(
        type(private_state) is dict and private_state.get("title") == spec["title"],
        "Fresh private state belongs to a different fixture",
    )
    if verify_goal(spec, private_state):
        return _descriptor("DONE", "terminal")
    if spec["family"].startswith("compact-"):
        return _descriptor("CLICK", "click", spec["desired_button_label"])
    if spec["paragraph_required"] and not _paragraph_complete(spec, private_state):
        y = private_state.get("scroll_y")
        _require(
            type(y) in (int, float) and math.isfinite(y) and abs(y) <= 1,
            "Expected first native scroll did not reveal the authored passage",
        )
        return _descriptor("SCROLL_DOWN", "scroll", "Scroll down", "scroll_down")
    if private_state.get("note_value") != spec["desired_text"]:
        return _descriptor("TYPE_TEXT", "fill", spec["field_label"], text=spec["desired_text"])
    if private_state.get("mode_value") != spec["desired_option_value"]:
        return _descriptor(
            "SELECT", "select", f"{spec['select_label']} → {spec['desired_option_label']}"
        )
    return _descriptor("CLICK", "click", spec["submit_label"])


def _state_digest(state):
    return hashlib.sha256(
        json.dumps(
            state, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False
        ).encode()
    ).hexdigest()


def verify_transition(spec, before, after, step, events):
    """Check real effects/events; collector must separately join canonical native execution."""
    _check_spec(spec)
    _require(type(before) is dict and type(after) is dict, "Actual private DOM states required")
    _require(step == teacher_step(spec, before), "Step differs from the explicit authored route")
    _require(after.get("title") == spec["title"], "Transition changed fixture identity")
    _require(
        type(before.get("url")) is str
        and bool(before["url"])
        and after.get("url") == before["url"],
        "Authored fixture transition must retain its owned document URL",
    )
    _require(type(events) is list, "Actual appended event list required")
    old, new = before.get("events"), after.get("events")
    _require(type(old) is list and type(new) is list, "Complete private event journals required")
    _require(
        new[: len(old)] == old and new[len(old) :] == events,
        "Private event prefix or appended events differ",
    )
    _require(
        all(
            type(e) is dict
            and type(e.get("sequence")) is int
            and e["sequence"] == i
            and type(e.get("trusted")) is bool
            and e.get("type") in {"click", "input", "change", "wheel"}
            and type(e.get("target")) is str
            for i, e in enumerate(new)
        ),
        "Event identity, order or native isTrusted value differs",
    )
    operation = step["operation"]
    proof = {
        "schema_id": TRANSITION_SCHEMA,
        "operation": operation,
        "native_kind": step["kind"],
        "before_sha256": _state_digest(before),
        "after_sha256": _state_digest(after),
        "appended_event_count": len(events),
        "positive_supervision": True,
        "requires_canonical_native_executor_join": operation != "DONE",
        "requires_native_select_executor_join": False,
    }

    def matched(kind, target, trusted, value=None):
        return any(
            e.get("type") == kind
            and e.get("target") == target
            and e.get("trusted") is trusted
            and (value is None or e.get("value") == value)
            for e in events
        )

    if operation == "DONE":
        _require(
            verify_goal(spec, before) and verify_goal(spec, after),
            "DONE needs fresh goal completion",
        )
        _require(not events and before == after, "DONE must not execute or mutate browser input")
        proof["input_executed"] = False
    elif operation == "SCROLL_DOWN":
        y0, y1 = before.get("scroll_y"), after.get("scroll_y")
        _require(
            type(y0) in (int, float)
            and type(y1) in (int, float)
            and math.isfinite(y0)
            and math.isfinite(y1)
            and abs((y1 - y0) - 560) <= 1
            and any(
                e.get("type") == "wheel" and e.get("trusted") is True and e.get("delta_y") == 560
                for e in events
            )
            and all(e.get("type") == "wheel" for e in events)
            and after.get("note_value") == before.get("note_value")
            and after.get("mode_value") == before.get("mode_value")
            and after.get("submission_count") == before.get("submission_count")
            and _paragraph_complete(spec, after),
            "Original trusted native wheel must reveal the full authored paragraph",
        )
        proof.update(input_executed=True, scroll_before=y0, scroll_after=y1, native_delta=560)
    elif operation == "TYPE_TEXT":
        _require(
            after.get("note_value") == spec["desired_text"]
            and after.get("mode_value") == before.get("mode_value")
            and after.get("submission_count") == before.get("submission_count")
            and after.get("submitted") is False
            and matched("input", "entry-text", True, spec["desired_text"]),
            "Trusted input did not produce the actual authored field value",
        )
        proof["input_executed"] = True
    elif operation == "SELECT":
        value = spec["desired_option_value"]
        _require(
            after.get("mode_value") == value
            and after.get("mode_label") == spec["desired_option_label"]
            and after.get("note_value") == before.get("note_value")
            and after.get("submission_count") == before.get("submission_count")
            and after.get("submitted") is False
            and matched("input", "entry-select", False, value)
            and matched("change", "entry-select", False, value),
            "Canonical SELECT synthetic events and actual selected DOM value required",
        )
        proof.update(
            input_executed=True,
            requires_native_select_executor_join=True,
            select_event_isTrusted=False,
            select_proof_scope="DOM/event effect only; guarded native Runtime.evaluate join required",
        )
    elif operation == "CLICK":
        if spec["family"].startswith("compact-"):
            target = next(
                button["id"]
                for button in spec["buttons"]
                if button["label"] == spec["desired_button_label"]
            )
            _require(
                matched("click", target, True) and verify_goal(spec, after),
                "Trusted compact click did not satisfy the authored goal",
            )
        else:
            count = before.get("submission_count")
            _require(
                type(count) is int
                and after.get("submission_count") == count + 1
                and matched("click", "commit-control", True)
                and verify_goal(spec, after),
                "Trusted submit did not record the actual requested values and goal state",
            )
        proof["input_executed"] = True
    else:
        raise ValueError("No positive BLOCKED, WAIT or other unauthored route is admitted")
    proof["goal_verified_after"] = verify_goal(spec, after)
    return proof
