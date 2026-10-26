"""Independent TRAIN edit routes on ordinary tall documents with large selects.

The publisher discovers every operation and target from the actual browser.
Useful controls remain at the top, with ordinary reference content below.
Prospective K7 is an admission expectation, never a constructed menu or evidence.
Teacher annotations remain outside publisher wires. Native event/state/judge
machinery is retained from the separately versioned navigation-distractor scenes.
This module imports no ancestor and has its own strict generator and identities.
"""

from __future__ import annotations

import copy
import hashlib
import html
import json
import unicodedata

SCHEMA = "decision3-owned-scroll-state-control-scenarios-v1"
TRANSITION_SCHEMA = "decision3-owned-scroll-state-control-transition-v1"
NAMESPACE = "decision3-owned-scroll-state-control-training-v1"
GOAL_NORMALIZATION = "NFKC-collapse-whitespace-v1"
EXPECTED_MENU_COUNTS = (7,)
RADIO_GROUP = "entry-route"
SELECT_PLACEHOLDER = "Choose a storage tray"
AUTHORED_OPTION_COUNTS = (12, 16, 24)
AUTHORED_CLICK_COUNTS = (10, 16, 22)
_BASE_FAMILIES = (
    ("checked-text", 2, ("text",), "desired", ("radio", "text")),
    ("checked-select", 3, ("select",), "desired", ("radio", "select")),
    ("radio-text-select", 2, ("radio", "text", "select"), "unchecked", ("radio", "text", "select")),
    ("radio-select-text", 3, ("radio", "select", "text"), "unchecked", ("radio", "select", "text")),
)
_FAMILY_DEFINITIONS = tuple(
    (f"{family}-options{size}", radios, missing, initial, order, size)
    for size in AUTHORED_OPTION_COUNTS
    for family, radios, missing, initial, order in _BASE_FAMILIES
)
FAMILIES = tuple(definition[0] for definition in _FAMILY_DEFINITIONS)
_NAVIGATION_LABELS = (
    "Work queue",
    "Saved worksheets",
    "Supply catalog",
    "Team contacts",
    "Receiving hours",
    "Delivery notes",
    "Maintenance schedule",
    "Safety handbook",
    "Equipment list",
    "Room directory",
    "Dispatch calendar",
    "Packing guide",
    "Collection points",
    "Reference library",
    "Service updates",
    "Storage policy",
    "Staff roster",
    "Previous requests",
    "Account overview",
    "Help center",
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
    "click_band",
    "commit_kind",
    "authored_option_count",
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
    family, radio_count, missing, radio_mode, order, option_count = definition
    radios = [
        {"id": f"route-choice-{index + 1}", "value": value, "label": label}
        for index, (value, label) in enumerate(
            _permutation(
                seed,
                family,
                "radio-order",
                (
                    ("birch-counter", "Birch counter"),
                    ("cedar-counter", "Cedar counter"),
                    ("maple-counter", "Maple counter"),
                    ("elm-counter", "Elm counter"),
                    ("pine-counter", "Pine counter"),
                    ("ash-counter", "Ash counter"),
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
            tuple((f"tray-{index:02d}", f"Tray {index:02d}") for index in range(1, 25)),
        )[:option_count]
    ]
    wanted_option = _pick(seed, family, "select-target", options)
    desired_text = (
        _pick(seed, family, "text-prefix", ("crate", "folio", "kit", "bundle", "carton"))
        + "-"
        + _axis(seed, family, "text-target")[:10]
    )
    modes = ["", *(option["value"] for option in options)]
    if "select" in missing:
        modes.remove(wanted_option["value"])
    initial_mode = _pick(seed, family, "initial-select", modes)
    opaque = _axis(seed, family, "identity")[:24]
    click_count = _pick(seed, family, "click-count", AUTHORED_CLICK_COUNTS)
    spec = {
        "schema_id": SCHEMA,
        "namespace": NAMESPACE,
        "id": f"d3-scroll-state-train-{opaque}",
        "ancestor_id": f"d3-scroll-state-owned-{_axis(seed, family, 'ancestor')[:24]}",
        "seed": seed,
        "split": "train",
        "family": family,
        "title": _pick(
            seed,
            family,
            "title",
            (
                "Supply allocation",
                "Workshop pickup",
                "Archive intake",
                "Container booking",
                "Tool reservation",
                "Material transfer",
            ),
        )
        + f" {_axis(seed, family, 'card-name')[:8]}",
        "expected_menu_counts": list(EXPECTED_MENU_COUNTS),
        "radio_group_label": _pick(
            seed,
            family,
            "radio-label",
            (
                "Receiving counter",
                "Collection counter",
                "Issuing counter",
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
                "Crate reference",
                "Folio reference",
                "Kit reference",
                "Bundle note",
            ),
        ),
        "select_label": _pick(
            seed,
            family,
            "select-label",
            (
                "Storage tray",
                "Sorting tray",
                "Receiving tray",
                "Packing tray",
            ),
        ),
        "submit_label": _pick(
            seed,
            family,
            "submit-label",
            (
                "Save allocation",
                "Record this request",
                "Confirm allocation",
                "File this entry",
            ),
        ),
        "options": options,
        "authored_option_count": option_count,
        "desired_text": desired_text,
        "desired_option_value": wanted_option["value"],
        "desired_option_label": wanted_option["label"],
        "required_fields": {name: name in order for name in ("radio", "text", "select")},
        "initial": {
            "radio_value": wanted_radio["value"] if radio_mode == "desired" else None,
            "note_value": f"memo-{_axis(seed, family, 'initial-text')[:12]}",
            "mode_value": initial_mode,
        },
        "goal_order": list(order),
        "layout_order": [],
        "navigation_links": [],
        "authored_click_candidates": click_count,
        "click_band": _pick(seed, family, "click-band", ("early", "middle", "late")),
        "commit_kind": _pick(seed, family, "commit-kind", ("link", "button")),
    }
    # Link labels, order and density are independent of requested control values.
    spec["navigation_links"] = [
        {
            "id": f"navigation-{index + 1}",
            "label": label,
            "href": f"/owned-scroll-reference/{opaque}/section-{index + 1}",
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
        f"For {spec['title']}, record an allocation with these requested values: "
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
    """Return twelve detached TRAIN scenes, without browser or model work."""
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
window.__d3CheckedControlEvents=[];
for(const type of ['click','input','change'])document.addEventListener(type,e=>{
 const t=e.target;
 window.__d3CheckedControlEvents.push({sequence:window.__d3CheckedControlEvents.length,
  type,target:t?.id||'',trusted:e.isTrusted,value:t&&'value'in t?String(t.value):null,
  checked:t?.type==='radio'?t.checked:null});
},true);
const note=document.getElementById('entry-text'),mode=document.getElementById('entry-select');
const radios=[...document.querySelectorAll('input[type="radio"][name="entry-route"]')];
const summary=document.getElementById('submission-summary'),status=document.getElementById('progress-status');
const edited=()=>{summary.dataset.submitted='false';status.textContent='Edited; submit the entry.';};
for(const control of [note,mode,...radios]){
 control.addEventListener('input',edited);control.addEventListener('change',edited);
}
document.getElementById('commit-control').addEventListener('click',event=>{
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
            f"<input id='entry-text' type='text' value='{esc(spec['initial']['note_value'], quote=True)}' "
            f"aria-label='{esc(spec['field_label'], quote=True)}'></label>"
        ),
        "select": (
            f"<label class='field'>{esc(spec['select_label'])}<select id='entry-select' "
            f"aria-label='{esc(spec['select_label'], quote=True)}'>"
            f"<option value=''{placeholder} disabled>{SELECT_PLACEHOLDER}</option>{options}"
            "</select></label>"
        ),
        "submit": (
            f"<a id='commit-control' href='#saved-record'>{esc(spec['submit_label'])}</a>"
            if spec["commit_kind"] == "link"
            else f"<button type='button' id='commit-control'>{esc(spec['submit_label'])}</button>"
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
        "button,#commit-control{font:16px sans-serif;padding:8px 10px}"
        ".navigation-link{font:14px sans-serif;padding:6px;border:1px solid #b3c1ce;"
        "overflow-wrap:anywhere}#submission-summary{font-size:14px;line-height:20px;margin-top:12px}"
        "#submission-summary span{display:block}output{overflow-wrap:anywhere}"
        ".reference-notes{width:min(1100px,calc(100% - 32px));margin:24px auto;"
        "padding:24px;min-height:1800px;background:#f8fafb}"
        ".reference-notes p{max-width:70ch;line-height:1.6}</style>"
        "</head><body><main>"
        f"<h1>{esc(spec['title'])}</h1><div class='controls'>{controls}</div>"
        "<div id='submission-summary' data-submitted='false' data-count='0'>"
        "<span id='progress-status'>No entry submitted.</span>"
        "<span>Recorded route: <output id='submitted-route' data-value=''></output></span>"
        "<span>Recorded note: <output id='submitted-note'></output></span>"
        "<span>Recorded mode: <output id='submitted-mode' data-value=''></output></span>"
        "</div></main><section class='reference-notes' aria-label='Reference notes'>"
        "<h2>Reference notes</h2>"
        "<p>Keep allocation references with their accompanying materials. Check the "
        "receiving counter and tray names before filing a request.</p>"
        "<p>Containers should carry a readable reference. Staff can consult the "
        "packing guide and supply catalog for ordinary handling information.</p>"
        "<p>Leave room for the receiving team to add handling notes after collection.</p>"
        "</section>" + _EVENT_SCRIPT + "</body></html>"
    )


PRIVATE = """(() => {
 const el=id=>document.getElementById(id),text=id=>el(id)?.textContent??null;
 const note=el('entry-text'),mode=el('entry-select'),summary=el('submission-summary');
 const selected=mode?.selectedOptions?.[0],recorded=el('submitted-mode'),route=el('submitted-route');
 const radios=[...document.querySelectorAll('input[type="radio"]')].map(r=>({
  id:r.id,name:r.name,value:r.value,label:r.getAttribute('aria-label'),checked:r.checked}));
 const boxes=[...document.querySelectorAll('input,select,#commit-control,.navigation-link')].map(e=>{
  const r=e.getBoundingClientRect(),s=getComputedStyle(e);
  return {id:e.id,left:r.left,top:r.top,right:r.right,bottom:r.bottom,
   visible:s.display!=='none'&&s.visibility!=='hidden'&&s.visibility!=='collapse'&&Number(s.opacity)!==0};
 });
 return {title:document.title,ready:document.readyState,url:location.href,
  viewport_width:innerWidth,viewport_height:innerHeight,document_height:document.documentElement.scrollHeight,
  scroll_y:scrollY,control_boxes:boxes,radios,
  navigation:[...document.querySelectorAll('a.navigation-link')].map(a=>({
   id:a.id,label:a.textContent,href:a.getAttribute('href')})),
  commit_tag:el('commit-control')?.tagName.toLowerCase()??null,
  note_value:note?.value??null,mode_value:mode?.value??null,
  mode_label:selected?.label??null,submitted:summary?.dataset.submitted==='true',
  submission_count:summary?Number(summary.dataset.count):null,status:text('progress-status'),
  submitted_radio_value:route?.dataset.value??null,submitted_radio_label:route?.textContent??null,
  submitted_note:text('submitted-note'),submitted_mode_value:recorded?.dataset.value??null,
  submitted_mode_label:recorded?.textContent??null,
  events:structuredClone(window.__d3CheckedControlEvents||[]),html:document.documentElement.outerHTML};
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
        "text": ["entry-text"],
        "select": ["entry-select"],
        "submit": ["commit-control"],
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
        and state["submission_count"] >= 0,
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
            and matched("input", "entry-text", True, spec["desired_text"])
            and all(event["target"] == "entry-text" for event in events),
            "Trusted text input must preserve the completed radio and unrelated fields",
        )
        proof["input_executed"] = True
    elif operation == "SELECT":
        value = spec["desired_option_value"]
        _require(
            after["mode_value"] == value
            and after["mode_label"] == spec["desired_option_label"]
            and unchanged("radios", "note_value", "submission_count", *_RECORDED_FIELDS)
            and after["submitted"] is False
            and matched("input", "entry-select", False, value)
            and matched("change", "entry-select", False, value)
            and all(event["target"] == "entry-select" for event in events),
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
        text_blur = ("change", "entry-text", True, before["note_value"], None)
        _require(
            after_selection is not None
            and after_selection["value"] == value
            and after["radios"] == expected
            and unchanged(
                "note_value", "mode_value", "mode_label", "submission_count", *_RECORDED_FIELDS
            )
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
            "commit-control",
            True,
            "" if spec["commit_kind"] == "button" else None,
            None,
        )
        text_blur = ("change", "entry-text", True, before["note_value"], None)
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
