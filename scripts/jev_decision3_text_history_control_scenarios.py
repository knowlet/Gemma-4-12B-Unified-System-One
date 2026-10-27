"""Independent TRAIN drafts: current values and remaining work after text input.

Only canonical browser actions can create policy history. This module never
manufactures a history or publisher menu. Prefilled correct text, empty/wrong
text and a native control change that refreshes text form counterexamples to
both blindly repeating input and blacklisting input after a previous attempt.
Teacher metadata stays outside the full original native wires. No ancestor is
imported; fixture identities, visible content and element IDs are independent.
"""

from __future__ import annotations

import copy
import hashlib
import html
import json
import unicodedata

SCHEMA = "decision3-owned-text-history-control-scenarios-v1"
TRANSITION_SCHEMA = "decision3-owned-text-history-control-transition-v1"
NAMESPACE = "decision3-owned-text-history-control-training-v1"
GOAL_NORMALIZATION = "NFKC-collapse-whitespace-v1"
EXPECTED_MENU_COUNTS = (7,)
RADIO_GROUP = "draft-dispatch"
SELECT_PLACEHOLDER = "Choose a calibration tier"
AUTHORED_OPTION_COUNTS = (12, 16, 24)
AUTHORED_CLICK_COUNTS = (10, 16, 22)
_BASE_FAMILIES = tuple(
    (f"{text}-{remaining}", 3, text, remaining, None)
    for text in ("empty", "wrong", "desired")
    for remaining in ("submit", "radio", "select")
) + (
    ("recheck-radio-clears", 3, "empty", "radio", "empty"),
    ("recheck-select-replaces", 3, "wrong", "select", "wrong"),
)
_FAMILY_DEFINITIONS = tuple(
    (f"{family}-options{size}", radios, text, remaining, refresh, size)
    for size in AUTHORED_OPTION_COUNTS
    for family, radios, text, remaining, refresh in _BASE_FAMILIES
)
FAMILIES = tuple(definition[0] for definition in _FAMILY_DEFINITIONS)
_NAVIGATION_LABELS = (
    "Sensor glossary",
    "Calibration calendar",
    "Bench capacity",
    "Instrument ledger",
    "Measurement units",
    "Precision guide",
    "Probe compatibility",
    "Reference standards",
    "Lab opening times",
    "Shipment tracking",
    "Loan agreements",
    "Training sessions",
    "Station availability",
    "Certificate archive",
    "Environmental readings",
    "Repair log",
    "Metrology notes",
    "Accessory register",
    "Warranty terms",
    "Contact the laboratory",
)


def _require(value, message):
    if not value:
        raise ValueError(message)


def _goal_sha256(goal):
    _require(type(goal) is str and bool(goal.strip()), "Nonempty authored goal required")
    return hashlib.sha256(
        " ".join(unicodedata.normalize("NFKC", goal).split()).encode()
    ).hexdigest()


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
        )
        and len(set(values)) == len(values),
        "Distinct lowercase SHA256 goal exclusions required",
    )
    return frozenset(values)


_SPEC_FIELDS = {
    "schema_id",
    "namespace",
    "id",
    "ancestor_id",
    "seed",
    "split",
    "family",
    "title",
    "expected_menu_counts",
    "radio_group_label",
    "radios",
    "desired_radio_value",
    "desired_radio_label",
    "field_label",
    "select_label",
    "submit_label",
    "options",
    "desired_text",
    "desired_option_value",
    "desired_option_label",
    "goal",
    "goal_sha256",
    "required_fields",
    "initial",
    "goal_order",
    "layout_order",
    "navigation_links",
    "authored_click_candidates",
    "commit_kind",
    "authored_option_count",
    "text_refresh",
}


def _axis(seed, family, name):
    return hashlib.sha256(f"{NAMESPACE}:{seed}:{family}:{name}".encode()).hexdigest()


def _pick(seed, family, name, values):
    return values[int(_axis(seed, family, name), 16) % len(values)]


def _permutation(seed, family, name, values):
    return sorted(
        values,
        key=lambda value: _axis(seed, family, name + ":" + json.dumps(value, sort_keys=True)),
    )


def _make_spec(seed, definition):
    family, radio_count, text_mode, remaining, refresh_mode, option_count = definition
    radios = [
        {"id": f"dispatch-choice-{index + 1}", "value": value, "label": label}
        for index, (value, label) in enumerate(
            _permutation(
                seed,
                family,
                "radio-order",
                (
                    ("amber-bench", "Amber bench"),
                    ("violet-bench", "Violet bench"),
                    ("cobalt-bench", "Cobalt bench"),
                    ("silver-bench", "Silver bench"),
                    ("indigo-bench", "Indigo bench"),
                    ("coral-bench", "Coral bench"),
                ),
            )[:radio_count]
        )
    ]
    wanted_radio = _pick(seed, family, "radio-target", radios)
    options = [
        {"value": value, "label": label}
        for value, label in _permutation(
            seed,
            family,
            "option-order",
            tuple((f"tier-{index:02d}", f"Tier {index:02d}") for index in range(1, 25)),
        )[:option_count]
    ]
    wanted_option = _pick(seed, family, "select-target", options)
    desired_text = (
        _pick(seed, family, "text-prefix", ("probe", "sensor", "gauge", "meter", "optic"))
        + "-"
        + _axis(seed, family, "text-target")[:10]
    )
    modes = ["", *(option["value"] for option in options)]
    if remaining == "select":
        modes.remove(wanted_option["value"])
    initial_mode = (
        _pick(seed, family, "initial-select", modes)
        if remaining == "select"
        else wanted_option["value"]
    )
    initial_radio = (
        _pick(
            seed,
            family,
            "initial-radio",
            [None, *(r["value"] for r in radios if r != wanted_radio)],
        )
        if remaining == "radio"
        else wanted_radio["value"]
    )
    initial_text = (
        desired_text
        if text_mode == "desired"
        else ""
        if text_mode == "empty"
        else "draft-" + _axis(seed, family, "initial-text")[:12]
    )
    order = (
        ["text", "radio", "select"]
        if refresh_mode is not None
        else _permutation(seed, family, "requested-order", ["radio", "text", "select"])
    )
    opaque = _axis(seed, family, "identity")[:24]
    click_count = _pick(seed, family, "click-count", AUTHORED_CLICK_COUNTS)
    spec = {
        "schema_id": SCHEMA,
        "namespace": NAMESPACE,
        "id": f"d3-text-history-train-{opaque}",
        "ancestor_id": f"d3-text-history-owned-{_axis(seed, family, 'ancestor')[:24]}",
        "seed": seed,
        "split": "train",
        "family": family,
        "title": _pick(
            seed,
            family,
            "title",
            (
                "Sensor calibration",
                "Gauge inspection",
                "Optics certification",
                "Probe servicing",
                "Meter verification",
                "Instrument check-in",
            ),
        )
        + f" {_axis(seed, family, 'card-name')[:8]}",
        "expected_menu_counts": list(EXPECTED_MENU_COUNTS),
        "radio_group_label": _pick(
            seed,
            family,
            "radio-label",
            (
                "Calibration bench",
                "Inspection bench",
                "Verification bench",
            ),
        ),
        "radios": radios,
        "desired_radio_value": wanted_radio["value"],
        "desired_radio_label": wanted_radio["label"],
        "field_label": _pick(
            seed,
            family,
            "text-label",
            (
                "Instrument annotation",
                "Measurement reference",
                "Calibration note",
                "Probe identifier",
            ),
        ),
        "select_label": _pick(
            seed,
            family,
            "select-label",
            (
                "Calibration tier",
                "Inspection tier",
                "Verification tier",
            ),
        ),
        "submit_label": _pick(
            seed,
            family,
            "submit-label",
            (
                "Record calibration",
                "Confirm inspection",
                "Save instrument draft",
                "Register verification",
            ),
        ),
        "options": options,
        "authored_option_count": option_count,
        "desired_text": desired_text,
        "desired_option_value": wanted_option["value"],
        "desired_option_label": wanted_option["label"],
        "required_fields": {name: True for name in ("radio", "text", "select")},
        "initial": {
            "radio_value": initial_radio,
            "note_value": initial_text,
            "mode_value": initial_mode,
        },
        "text_refresh": {
            "control": remaining if refresh_mode is not None else None,
            "value": (
                ""
                if refresh_mode == "empty"
                else "refreshed-" + _axis(seed, family, "refresh-text")[:12]
            )
            if refresh_mode is not None
            else None,
        },
        "goal_order": list(order),
        "layout_order": [],
        "navigation_links": [],
        "authored_click_candidates": click_count,
        "commit_kind": _pick(seed, family, "commit-kind", ("link", "button")),
    }
    # Link labels, order and density are independent of requested control values.
    spec["navigation_links"] = [
        {
            "id": f"lab-reference-{index + 1}",
            "label": label,
            "href": f"/owned-calibration-reference/{opaque}/topic-{index + 1}",
        }
        for index, label in enumerate(
            _permutation(
                seed,
                family,
                "navigation-labels",
                _NAVIGATION_LABELS,
            )[: click_count - radio_count - 2]
        )
    ]
    spec["layout_order"] = _permutation(
        seed,
        family,
        "layout-order",
        ["radio", "text", "select", "submit"] + [link["id"] for link in spec["navigation_links"]],
    )
    links = {link["id"]: link for link in spec["navigation_links"]}
    spec["navigation_links"] = [links[token] for token in spec["layout_order"] if token in links]
    clauses = {
        "radio": f"{spec['radio_group_label']} = {spec['desired_radio_label']}",
        "text": f"{spec['field_label']} = {spec['desired_text']}",
        "select": f"{spec['select_label']} = {spec['desired_option_label']}",
    }
    spec["goal"] = (
        f"For {spec['title']}, record a calibration draft with these requested values: "
        + "; ".join(clauses[name] for name in spec["goal_order"])
        + f". Leave all other controls as they are. Use {spec['submit_label']} after "
        "the requested values are set, then finish when the visible saved record matches."
    )
    spec["goal_sha256"] = _goal_sha256(spec["goal"])
    return spec


def _check_spec(spec):
    _require(type(spec) is dict and set(spec) == _SPEC_FIELDS, "Exact authored spec required")
    _require(
        type(spec["seed"]) is int and 0 <= spec["seed"] < 2**63,
        "Nonnegative bounded integer scenario seed required",
    )
    _require(
        type(spec["family"]) is str and spec["family"] in FAMILIES,
        "Only this owned train-only namespace is supported",
    )
    _require(
        type(spec["required_fields"]) is dict
        and set(spec["required_fields"]) == {"radio", "text", "select"}
        and all(type(value) is bool for value in spec["required_fields"].values()),
        "Strict authored required-field booleans required",
    )
    _require(
        spec == _make_spec(spec["seed"], _FAMILY_DEFINITIONS[FAMILIES.index(spec["family"])]),
        "Authored namespace, identities, controls, initial values, goal or hash differ",
    )


def scenarios(seed, excluded_goal_hashes):
    """Return33 detached TRAIN scenes; real history requires native execution."""
    _require(type(seed) is int and 0 <= seed < 2**63, "Seed must be an integer in [0, 2**63)")
    excluded = _exclusions(excluded_goal_hashes)
    result = []
    for definition in _FAMILY_DEFINITIONS:
        spec = _make_spec(seed, definition)
        _require(spec["goal_sha256"] not in excluded, "Authored goal collides with an exclusion")
        _check_spec(spec)
        result.append(spec)
    _require(
        len({spec["goal_sha256"] for spec in result}) == len(FAMILIES),
        "Distinct authored goals required",
    )
    return tuple(copy.deepcopy(result))


_EVENT_SCRIPT = """<script>
window.__d3TextHistoryEvents=[];
for(const type of ['click','input','change'])document.addEventListener(type,e=>{
 const t=e.target;
 window.__d3TextHistoryEvents.push({sequence:window.__d3TextHistoryEvents.length,
  type,target:t?.id||'',trusted:e.isTrusted,value:t&&'value'in t?String(t.value):null,
  checked:t?.type==='radio'?t.checked:null});
},true);
const note=document.getElementById('draft-annotation'),mode=document.getElementById('draft-tier');
const radios=[...document.querySelectorAll('input[type="radio"][name="draft-dispatch"]')];
const summary=document.getElementById('submission-summary'),status=document.getElementById('progress-status');
const edited=()=>{summary.dataset.submitted='false';status.textContent='Edited; submit the entry.';};
for(const control of [note,mode,...radios]){
 control.addEventListener('input',edited);control.addEventListener('change',edited);
}
// AUTHORED_REFRESH_HOOK
document.getElementById('record-draft').addEventListener('click',event=>{
 event.preventDefault();
 const checked=radios.find(r=>r.checked),selected=mode.selectedOptions[0];
 const route=document.getElementById('submitted-route'),recorded=document.getElementById('submitted-mode');
 route.textContent=checked?.getAttribute('aria-label')||'';route.dataset.value=checked?.value||'';
 document.getElementById('submitted-note').textContent=note.value;
 recorded.textContent=selected?.label||'';recorded.dataset.value=mode.value;
 summary.dataset.count=String(Number(summary.dataset.count)+1);summary.dataset.submitted='true';
 status.textContent='Submitted entry '+summary.dataset.count+'.';
});
</script>"""


def _event_script(spec):
    refresh = json.dumps(spec["text_refresh"], ensure_ascii=True, separators=(",", ":"))
    script = (
        "const refresh="
        + refresh
        + ";\n"
        + """
const refreshDraft=()=>{
 if(Number(summary.dataset.textRefreshCount)!==0)return;
 note.value=refresh.value;summary.dataset.textRefreshCount='1';edited();
};
if(refresh.control==='radio')for(const radio of radios)radio.addEventListener('change',refreshDraft);
if(refresh.control==='select')mode.addEventListener('change',refreshDraft);
"""
    )
    return _EVENT_SCRIPT.replace("// AUTHORED_REFRESH_HOOK", script)


def render_fixture(spec):
    """Render ordinary editable controls with independently authored visible defaults."""
    _check_spec(spec)
    esc = html.escape
    radios = "".join(
        "<label class='route-choice'>"
        f"<input type='radio' id='{esc(choice['id'], quote=True)}' name='{RADIO_GROUP}' "
        f"value='{esc(choice['value'], quote=True)}' aria-label='{esc(choice['label'], quote=True)}'"
        + (" checked" if choice["value"] == spec["initial"]["radio_value"] else "")
        + f"><span>{esc(choice['label'])}</span></label>"
        for choice in spec["radios"]
    )
    options = "".join(
        f"<option value='{esc(option['value'], quote=True)}'"
        + (" selected" if option["value"] == spec["initial"]["mode_value"] else "")
        + f">{esc(option['label'])}</option>"
        for option in spec["options"]
    )
    placeholder = " selected" if spec["initial"]["mode_value"] == "" else ""
    blocks = {
        "radio": f"<fieldset><legend>{esc(spec['radio_group_label'])}</legend>{radios}</fieldset>",
        "text": (
            f"<label class='field'>{esc(spec['field_label'])}"
            f"<input id='draft-annotation' type='text' value='{esc(spec['initial']['note_value'], quote=True)}' "
            f"aria-label='{esc(spec['field_label'], quote=True)}'></label>"
        ),
        "select": (
            f"<label class='field'>{esc(spec['select_label'])}<select id='draft-tier' "
            f"aria-label='{esc(spec['select_label'], quote=True)}'>"
            f"<option value=''{placeholder} disabled>{SELECT_PLACEHOLDER}</option>{options}"
            "</select></label>"
        ),
        "submit": (
            f"<a id='record-draft' href='#saved-record'>{esc(spec['submit_label'])}</a>"
            if spec["commit_kind"] == "link"
            else f"<button type='button' id='record-draft'>{esc(spec['submit_label'])}</button>"
        ),
    }
    blocks.update(
        {
            link["id"]: (
                f"<a class='navigation-link' id='{esc(link['id'], quote=True)}' "
                f"href='{esc(link['href'], quote=True)}'>{esc(link['label'])}</a>"
            )
            for link in spec["navigation_links"]
        }
    )
    controls = "".join(blocks[name] for name in spec["layout_order"])
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{esc(spec['title'])}</title>"
        "<style>*{box-sizing:border-box}html{min-height:100%;overflow-y:auto}"
        "body{margin:0;width:100%;min-height:100%;font:16px sans-serif;background:#edf1f4}"
        "main{width:min(1100px,calc(100% - 32px));margin:16px auto;padding:16px;background:white;"
        "border:2px solid #405565}h1{font-size:22px;margin:0 0 12px}"
        ".controls{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));"
        "gap:8px;align-items:center}fieldset{grid-column:span 2;margin:0;"
        "padding:8px;border:1px solid #405565}legend{padding:0 4px}"
        ".route-choice{display:inline-flex;align-items:center;gap:4px;margin:3px 10px 3px 0}"
        ".route-choice input{width:16px;height:16px;margin:0}"
        ".field{grid-column:span 2;display:flex;flex-direction:column;gap:3px}"
        ".field input,.field select{font:16px sans-serif;width:100%;height:32px}"
        "button,#record-draft{font:16px sans-serif;padding:8px 10px}"
        ".navigation-link{font:14px sans-serif;padding:6px;border:1px solid #b3c1ce;"
        "overflow-wrap:anywhere}#submission-summary{font-size:14px;line-height:20px;margin-top:12px}"
        "#submission-summary span{display:block}output{overflow-wrap:anywhere}"
        ".reference-notes{width:min(1100px,calc(100% - 32px));margin:24px auto;"
        "padding:24px;min-height:1800px;background:#f8fafb}"
        ".reference-notes p{max-width:70ch;line-height:1.6}</style>"
        "</head><body><main>"
        f"<h1>{esc(spec['title'])}</h1><div class='controls'>{controls}</div>"
        "<div id='submission-summary' data-submitted='false' data-count='0' data-text-refresh-count='0'>"
        "<span id='progress-status'>No entry submitted.</span>"
        "<span>Recorded route: <output id='submitted-route' data-value=''></output></span>"
        "<span>Recorded note: <output id='submitted-note'></output></span>"
        "<span>Recorded mode: <output id='submitted-mode' data-value=''></output></span>"
        "</div></main><section class='reference-notes' aria-label='Reference notes'>"
        "<h2>Reference notes</h2>"
        "<p>Calibration records accompany instruments through routine inspection. "
        "Bench schedules and precision standards are available in the laboratory reference pages.</p>"
        "<p>Each instrument carries a readable annotation and an assigned calibration tier. "
        "The recorded draft remains visible after confirmation.</p>"
        "<p>Laboratory staff review measurements before issuing a certificate.</p>"
        "</section>" + _event_script(spec) + "</body></html>"
    )


PRIVATE = """(() => {
 const el=id=>document.getElementById(id),text=id=>el(id)?.textContent??null;
 const note=el('draft-annotation'),mode=el('draft-tier'),summary=el('submission-summary');
 const selected=mode?.selectedOptions?.[0],recorded=el('submitted-mode'),route=el('submitted-route');
 const radios=[...document.querySelectorAll('input[type="radio"]')].map(r=>({
  id:r.id,name:r.name,value:r.value,label:r.getAttribute('aria-label'),checked:r.checked}));
 const boxes=[...document.querySelectorAll('input,select,#record-draft,.navigation-link')].map(e=>{
  const r=e.getBoundingClientRect(),s=getComputedStyle(e);
  return {id:e.id,left:r.left,top:r.top,right:r.right,bottom:r.bottom,
   visible:s.display!=='none'&&s.visibility!=='hidden'&&s.visibility!=='collapse'&&Number(s.opacity)!==0};
 });
 return {title:document.title,ready:document.readyState,url:location.href,
  viewport_width:innerWidth,viewport_height:innerHeight,document_height:document.documentElement.scrollHeight,
  scroll_y:scrollY,control_boxes:boxes,radios,
  navigation:[...document.querySelectorAll('a.navigation-link')].map(a=>({
   id:a.id,label:a.textContent,href:a.getAttribute('href')})),
  commit_tag:el('record-draft')?.tagName.toLowerCase()??null,
  note_value:note?.value??null,mode_value:mode?.value??null,
  mode_label:selected?.label??null,submitted:summary?.dataset.submitted==='true',
  text_refresh_count:summary?Number(summary.dataset.textRefreshCount):null,
  submission_count:summary?Number(summary.dataset.count):null,status:text('progress-status'),
  submitted_radio_value:route?.dataset.value??null,submitted_radio_label:route?.textContent??null,
  submitted_note:text('submitted-note'),submitted_mode_value:recorded?.dataset.value??null,
  submitted_mode_label:recorded?.textContent??null,
  events:structuredClone(window.__d3TextHistoryEvents||[]),html:document.documentElement.outerHTML};
})()"""


def private_state_expression():
    return PRIVATE


def _radio_selection(spec, state):
    radios = state.get("radios")
    _require(
        type(radios) is list and len(radios) == len(spec["radios"]),
        "Complete actual radio group required",
    )
    selected = []
    for actual, expected in zip(radios, spec["radios"], strict=True):
        _require(
            type(actual) is dict
            and set(actual) == {"id", "name", "value", "label", "checked"}
            and actual["name"] == RADIO_GROUP
            and all(actual[name] == expected[name] for name in ("id", "value", "label"))
            and type(actual["checked"]) is bool,
            "Actual radio identity and strict DOM checked booleans required",
        )
        if actual["checked"]:
            selected.append(actual)
    _require(len(selected) <= 1, "Actual radio group must remain mutually exclusive")
    return selected[0] if selected else None


def _top_geometry(spec, state):
    """Require actual tall-document, top-position and full visible-control evidence."""
    width, height = state.get("viewport_width"), state.get("viewport_height")
    document_height, scroll = state.get("document_height"), state.get("scroll_y")
    _require(
        type(width) is int
        and width > 0
        and type(height) is int
        and height > 0
        and type(document_height) is int
        and document_height > height + 1000
        and type(scroll) in (int, float)
        and scroll == 0,
        "Fresh tall-document geometry and unchanged top scroll position required",
    )
    block_ids = {
        "radio": [choice["id"] for choice in spec["radios"]],
        "text": ["draft-annotation"],
        "select": ["draft-tier"],
        "submit": ["record-draft"],
        **{link["id"]: [link["id"]] for link in spec["navigation_links"]},
    }
    expected = [identity for token in spec["layout_order"] for identity in block_ids[token]]
    boxes = state.get("control_boxes")
    _require(
        type(boxes) is list and len(boxes) == len(expected),
        "Complete actual ordered control geometry required",
    )
    for box, identity in zip(boxes, expected, strict=True):
        _require(
            type(box) is dict
            and set(box) == {"id", "left", "top", "right", "bottom", "visible"}
            and box["id"] == identity
            and box["visible"] is True
            and all(type(box[name]) in (int, float) for name in ("left", "top", "right", "bottom"))
            and 0 <= box["left"] < box["right"] <= width
            and 0 <= box["top"] < box["bottom"] <= height,
            "Every actual useful control and distractor must remain visible at the top",
        )


def _state_selection(spec, state):
    _require(
        type(state) is dict and state.get("title") == spec["title"],
        "Fresh private state belongs to a different fixture",
    )
    selection = _radio_selection(spec, state)
    _top_geometry(spec, state)
    _require(
        state.get("navigation") == spec["navigation_links"]
        and state.get("commit_tag") == ("a" if spec["commit_kind"] == "link" else "button"),
        "Complete ordinary navigation links and useful commit control identity required",
    )
    _require(
        type(state.get("note_value")) is str
        and type(state.get("mode_value")) is str
        and type(state.get("submitted")) is bool
        and type(state.get("submission_count")) is int
        and state["submission_count"] >= 0
        and type(state.get("text_refresh_count")) is int
        and state["text_refresh_count"] in (0, 1)
        and (spec["text_refresh"]["control"] is not None or state["text_refresh_count"] == 0),
        "Actual input values, submitted boolean and nonnegative counter required",
    )
    modes = {"": SELECT_PLACEHOLDER, **{o["value"]: o["label"] for o in spec["options"]}}
    _require(
        state["mode_value"] in modes and state.get("mode_label") == modes[state["mode_value"]],
        "Actual selected option label and value required",
    )
    return selection


def verify_goal(spec, private_state):
    """Check current native DOM and visible recorded values, not a saved success flag."""
    _check_spec(spec)
    state = private_state
    try:
        selection = _state_selection(spec, state)
        _preserved(spec, state, selection)
    except ValueError:
        return False
    count = state["submission_count"]
    return (
        _requirements_satisfied(spec, state, selection)
        and state.get("submitted_radio_value") == (selection["value"] if selection else "")
        and state.get("submitted_radio_label") == (selection["label"] if selection else "")
        and state["submitted"] is True
        and count > 0
        and state.get("status") == f"Submitted entry {count}."
        and state["note_value"] == state.get("submitted_note")
        and state["mode_value"] == state.get("submitted_mode_value")
        and state["mode_label"] == state.get("submitted_mode_label")
    )


def _preserved(spec, state, selection):
    initial, required = spec["initial"], spec["required_fields"]
    actual = {
        "radio": selection["value"] if selection else None,
        "text": state["note_value"],
        "select": state["mode_value"],
    }
    originals = {
        "radio": initial["radio_value"],
        "text": initial["note_value"],
        "select": initial["mode_value"],
    }
    _require(
        all(required[name] or actual[name] == originals[name] for name in required),
        "Controls outside the requested values must retain their initial actual values",
    )


def _requirements_satisfied(spec, state, selection):
    satisfied = {
        "radio": selection is not None and selection["value"] == spec["desired_radio_value"],
        "text": state["note_value"] == spec["desired_text"],
        "select": state["mode_value"] == spec["desired_option_value"],
    }
    return all(
        not required or satisfied[name] for name, required in spec["required_fields"].items()
    )


def _descriptor(operation, kind, label=None, action_id=None, text=None):
    return {
        "operation": operation,
        "kind": kind,
        "label": label,
        "action_id": action_id,
        "text": text,
    }


def teacher_step(spec, private_state):
    """Choose a genuinely unsatisfied requested field from current actual controls."""
    _check_spec(spec)
    selection = _state_selection(spec, private_state)
    _preserved(spec, private_state, selection)
    if verify_goal(spec, private_state):
        return _descriptor("DONE", "terminal")
    for name in spec["goal_order"]:
        if name == "radio" and (
            selection is None or selection["value"] != spec["desired_radio_value"]
        ):
            return _descriptor("CLICK", "click", spec["desired_radio_label"])
        if name == "text" and private_state["note_value"] != spec["desired_text"]:
            return _descriptor("TYPE_TEXT", "fill", spec["field_label"], text=spec["desired_text"])
        if name == "select" and private_state["mode_value"] != spec["desired_option_value"]:
            return _descriptor(
                "SELECT", "select", f"{spec['select_label']} → {spec['desired_option_label']}"
            )
    return _descriptor("CLICK", "click", spec["submit_label"])


def verify_native_wire_unchanged(original, observed):
    """Check externally supplied full wires without constructing a menu/history.

    Dict insertion order is part of the comparison at every depth, including
    state/history, questions, instructions and criteria values. Counts come
    solely from the supplied original wire, never from prospective fixture K.
    """
    _require(
        type(original) is dict and type(observed) is dict, "Two complete native wires required"
    )
    questions = original.get("questions")
    _require(
        type(questions) is dict and 1 <= len(questions) <= 4, "Original native questions required"
    )
    for question in questions.values():
        _require(
            type(question) is dict
            and question.get("type") == "choice"
            and type(question.get("criteria")) is dict
            and 1 <= len(question["criteria"]) <= 255,
            "Original complete ordered Choice criteria required",
        )

    def encode(value):
        return json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()

    _require(encode(original) == encode(observed), "Full original native wire or order changed")
    return {
        "ordered_head_ids": list(questions),
        "candidate_counts": [len(question["criteria"]) for question in questions.values()],
        "wire_sha256": hashlib.sha256(encode(original)).hexdigest(),
    }


def _state_digest(state):
    return hashlib.sha256(
        json.dumps(
            state, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False
        ).encode()
    ).hexdigest()


_RECORDED_FIELDS = (
    "submitted_radio_value",
    "submitted_radio_label",
    "submitted_note",
    "submitted_mode_value",
    "submitted_mode_label",
)


def verify_transition(spec, before, after, step, events):
    """Check native DOM/events; admission must also join the canonical native executor."""
    _check_spec(spec)
    _require(type(before) is dict and type(after) is dict, "Actual private DOM states required")
    _require(step == teacher_step(spec, before), "Step differs from the explicit authored route")
    after_selection = _state_selection(spec, after)
    _preserved(spec, after, after_selection)
    _require(
        type(before.get("url")) is str
        and bool(before["url"])
        and after.get("url") == before["url"],
        "Authored fixture transition must retain its owned document URL",
    )
    old, new = before.get("events"), after.get("events")
    _require(
        type(events) is list and type(old) is list and type(new) is list,
        "Complete actual private event journals required",
    )
    _require(
        new[: len(old)] == old and new[len(old) :] == events,
        "Private event prefix or appended events differ",
    )
    _require(
        all(
            type(event) is dict
            and set(event) == {"sequence", "type", "target", "trusted", "value", "checked"}
            and type(event.get("sequence")) is int
            and event["sequence"] == index
            and type(event.get("trusted")) is bool
            and event.get("type") in {"click", "input", "change"}
            and type(event.get("target")) is str
            and (event.get("value") is None or type(event["value"]) is str)
            and (event.get("checked") is None or type(event["checked"]) is bool)
            for index, event in enumerate(new)
        ),
        "Event identity, order, native isTrusted or checked value differs",
    )
    operation = step["operation"]
    refresh = spec["text_refresh"]
    refreshed = before["text_refresh_count"] == 0 and (
        (
            refresh["control"] == "radio"
            and operation == "CLICK"
            and step["label"] == spec["desired_radio_label"]
        )
        or (refresh["control"] == "select" and operation == "SELECT")
    )
    _require(
        after["text_refresh_count"] == before["text_refresh_count"] + int(refreshed),
        "Only the authored first native control change may refresh text",
    )
    expected_note = refresh["value"] if refreshed else before["note_value"]
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

    def matched(kind, target, trusted, value=None, checked=None):
        return any(
            event["type"] == kind
            and event["target"] == target
            and event["trusted"] is trusted
            and (value is None or event.get("value") == value)
            and event.get("checked") is checked
            for event in events
        )

    def unchanged(*names):
        return all(after.get(name) == before.get(name) for name in names)

    if operation == "DONE":
        _require(
            verify_goal(spec, before) and verify_goal(spec, after),
            "DONE needs fresh goal completion",
        )
        _require(not events and before == after, "DONE must not execute or mutate browser input")
        proof["input_executed"] = False
    elif operation == "TYPE_TEXT":
        _require(
            after["note_value"] == spec["desired_text"]
            and unchanged(
                "radios", "mode_value", "mode_label", "submission_count", *_RECORDED_FIELDS
            )
            and after["submitted"] is False
            and matched("input", "draft-annotation", True, spec["desired_text"])
            and all(event["target"] == "draft-annotation" for event in events),
            "Trusted text input must preserve the completed radio and unrelated fields",
        )
        proof["input_executed"] = True
    elif operation == "SELECT":
        value = spec["desired_option_value"]
        _require(
            after["mode_value"] == value
            and after["mode_label"] == spec["desired_option_label"]
            and after["note_value"] == expected_note
            and unchanged("radios", "submission_count", *_RECORDED_FIELDS)
            and after["submitted"] is False
            and matched("input", "draft-tier", False, value)
            and matched("change", "draft-tier", False, value)
            and all(event["target"] == "draft-tier" for event in events),
            "Canonical SELECT synthetic events and actual selected DOM value required",
        )
        proof.update(
            input_executed=True,
            requires_native_select_executor_join=True,
            select_event_isTrusted=False,
            select_proof_scope="DOM/event effect only; guarded native Runtime.evaluate join required",
        )
    elif operation == "CLICK" and step["label"] == spec["desired_radio_label"]:
        radio = next(
            choice for choice in spec["radios"] if choice["value"] == spec["desired_radio_value"]
        )
        target, value = radio["id"], radio["value"]
        expected = [dict(choice, checked=choice["value"] == value) for choice in before["radios"]]
        signatures = [
            (event["type"], event["target"], event["trusted"], event["value"], event["checked"])
            for event in events
        ]
        radio_events = [(kind, target, True, value, True) for kind in ("click", "input", "change")]
        text_blur = ("change", "draft-annotation", True, before["note_value"], None)
        _require(
            after_selection is not None
            and after_selection["value"] == value
            and after["radios"] == expected
            and after["note_value"] == expected_note
            and unchanged("mode_value", "mode_label", "submission_count", *_RECORDED_FIELDS)
            and after["submitted"] is False
            and signatures in (radio_events, [text_blur, *radio_events]),
            "Trusted radio click must change selection with native group exclusion",
        )
        proof["input_executed"] = True
    elif operation == "CLICK":
        # Clicking submit can commit the focused text input's unchanged value
        # through a native change event before the button's click event.
        signatures = [
            (event["type"], event["target"], event["trusted"], event["value"], event["checked"])
            for event in events
        ]
        submit_click = (
            "click",
            "record-draft",
            True,
            "" if spec["commit_kind"] == "button" else None,
            None,
        )
        text_blur = ("change", "draft-annotation", True, before["note_value"], None)
        _require(
            step["label"] == spec["submit_label"]
            and after["submission_count"] == before["submission_count"] + 1
            and unchanged("radios", "note_value", "mode_value", "mode_label")
            and signatures in ([submit_click], [text_blur, submit_click])
            and verify_goal(spec, after),
            "Trusted submit must record the actual requested route, note and option",
        )
        proof["input_executed"] = True
    else:
        raise ValueError("No positive BLOCKED, WAIT or other unauthored route is admitted")
    proof["goal_verified_after"] = verify_goal(spec, after)
    return proof
