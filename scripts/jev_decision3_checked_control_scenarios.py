"""Owned train-only scenes for progressing beyond an already checked control.

The pinned publisher constructs every observed menu and policy wire. These
authored fixtures provide an application and an independent DOM teacher only;
scenario identities and proof annotations stay outside wires. The plain goal
supplies the requested visible values.
No collected browser evidence or model quality result is supplied here.
"""

from __future__ import annotations

import copy
import hashlib
import html
import json
import unicodedata

SCHEMA = "decision3-owned-checked-control-scenarios-v1"
TRANSITION_SCHEMA = "decision3-owned-checked-control-transition-v1"
NAMESPACE = "decision3-owned-checked-control-training-v1"
GOAL_NORMALIZATION = "NFKC-collapse-whitespace-v1"
FAMILIES = ("radio-two", "radio-three")
EXPECTED_MENU_COUNTS = (6,)
RADIO_GROUP = "entry-route"
SELECT_PLACEHOLDER = "Choose a wrapping option"


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
}


def _check_spec(spec):
    _require(type(spec) is dict and set(spec) == _SPEC_FIELDS, "Exact authored spec required")
    _require(
        spec["schema_id"] == SCHEMA
        and spec["namespace"] == NAMESPACE
        and spec["split"] == "train"
        and spec["family"] in FAMILIES,
        "Only this owned train-only checked-control namespace is supported",
    )
    _require(
        type(spec["seed"]) is int and 0 <= spec["seed"] < 2**63,
        "Nonnegative bounded integer scenario seed required",
    )
    _require(
        spec["id"] == f"d3-checked-train-{spec['seed']}-{spec['family']}"
        and spec["ancestor_id"] == f"d3-checked-owned-{spec['seed']}-{spec['family']}",
        "New checked-control scenario and ancestor identities required",
    )
    _require(
        spec["expected_menu_counts"] == list(EXPECTED_MENU_COUNTS),
        "Natural checked-control menu coverage changed",
    )
    for name in (
        "title",
        "radio_group_label",
        "field_label",
        "select_label",
        "submit_label",
        "desired_text",
    ):
        _require(type(spec[name]) is str and bool(spec[name].strip()), "Visible text required")
    radios, options = spec["radios"], spec["options"]
    count = 2 if spec["family"] == "radio-two" else 3
    _require(type(radios) is list and len(radios) == count, "Exact authored radio count required")
    _require(
        all(
            type(choice) is dict
            and set(choice) == {"id", "value", "label"}
            and choice["id"] == f"route-choice-{index + 1}"
            and all(type(choice[name]) is str and bool(choice[name]) for name in ("value", "label"))
            for index, choice in enumerate(radios)
        )
        and len({choice["value"] for choice in radios}) == count
        and len({choice["label"] for choice in radios}) == count,
        "Distinct visible radio values, labels and DOM identities required",
    )
    _require(
        type(options) is list
        and len(options) == 3
        and all(
            type(option) is dict
            and set(option) == {"value", "label"}
            and all(type(option[name]) is str and bool(option[name]) for name in ("value", "label"))
            for option in options
        )
        and len({option["value"] for option in options}) == 3
        and len({option["label"] for option in options}) == 3,
        "Three distinct actual select options required",
    )
    _require(
        any(
            choice["value"] == spec["desired_radio_value"]
            and choice["label"] == spec["desired_radio_label"]
            for choice in radios
        )
        and {"value": spec["desired_option_value"], "label": spec["desired_option_label"]}
        in options,
        "Desired values must be actual authored controls",
    )
    _require(
        spec["goal_sha256"] == _goal_sha256(spec["goal"])
        and all(
            spec[name] in spec["goal"]
            for name in (
                "desired_radio_label",
                "desired_text",
                "desired_option_label",
                "radio_group_label",
                "field_label",
                "select_label",
                "submit_label",
            )
        ),
        "The plain authored goal must supply the requested visible field values",
    )


def scenarios(seed, excluded_goal_hashes):
    """Return two detached new training scenes; no browser or model is started."""
    _require(
        type(seed) is int and 0 <= seed < 2**63,
        "Seed must be an integer in [0, 2**63)",
    )
    excluded = _exclusions(excluded_goal_hashes)
    radio_pool = (
        ("amber-route", "Amber route"),
        ("cobalt-route", "Cobalt route"),
        ("moss-route", "Moss route"),
        ("ivory-route", "Ivory route"),
    )
    option_pool = (
        ("cotton-sleeve", "Cotton sleeve"),
        ("paper-wrap", "Paper wrap"),
        ("mesh-pouch", "Mesh pouch"),
    )
    titles = ("Dispatch card", "Courier card", "Parcel card", "Freight card")
    result = []
    for index, family in enumerate(FAMILIES):
        count = index + 2
        offset = (seed + index) % len(radio_pool)
        rotated = radio_pool[offset:] + radio_pool[:offset]
        radios = [
            {"id": f"route-choice-{number + 1}", "value": value, "label": label}
            for number, (value, label) in enumerate(rotated[:count])
        ]
        wanted_radio = radios[(seed // len(radio_pool) + index) % count]
        offset = (seed + index) % len(option_pool)
        options = [
            {"value": value, "label": label}
            for value, label in option_pool[offset:] + option_pool[:offset]
        ]
        wanted_option = options[(seed // len(option_pool) + index) % len(options)]
        spec = {
            "schema_id": SCHEMA,
            "namespace": NAMESPACE,
            "id": f"d3-checked-train-{seed}-{family}",
            "ancestor_id": f"d3-checked-owned-{seed}-{family}",
            "seed": seed,
            "split": "train",
            "family": family,
            "title": titles[(seed + index) % len(titles)],
            "expected_menu_counts": list(EXPECTED_MENU_COUNTS),
            "radio_group_label": "Dispatch route",
            "radios": radios,
            "desired_radio_value": wanted_radio["value"],
            "desired_radio_label": wanted_radio["label"],
            "field_label": "Dispatch note",
            "select_label": "Wrapping mode",
            "submit_label": "Record dispatch",
            "options": options,
            "desired_text": f"dispatch memo {seed}-{index + 1}",
            "desired_option_value": wanted_option["value"],
            "desired_option_label": wanted_option["label"],
        }
        spec["goal"] = (
            f"Choose {spec['desired_radio_label']} in {spec['radio_group_label']}, enter "
            f"{spec['desired_text']} in {spec['field_label']}, choose "
            f"{spec['desired_option_label']} in {spec['select_label']}, then click "
            f"{spec['submit_label']}. Finish when the submitted record shows that route, "
            "note and wrapping mode."
        )
        spec["goal_sha256"] = _goal_sha256(spec["goal"])
        _require(spec["goal_sha256"] not in excluded, "Authored goal collides with an exclusion")
        _check_spec(spec)
        result.append(spec)
    _require(len({spec["goal_sha256"] for spec in result}) == 2, "Distinct authored goals required")
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
document.getElementById('commit-control').addEventListener('click',()=>{
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
    """Render ordinary app controls without goal annotations or prefilled answers."""
    _check_spec(spec)
    esc = html.escape
    radios = "".join(
        "<label class='route-choice'>"
        f"<input type='radio' id='{esc(choice['id'], quote=True)}' name='{RADIO_GROUP}' "
        f"value='{esc(choice['value'], quote=True)}' aria-label='{esc(choice['label'], quote=True)}'>"
        f"<span>{esc(choice['label'])}</span></label>"
        for choice in spec["radios"]
    )
    options = "".join(
        f"<option value='{esc(option['value'], quote=True)}'>{esc(option['label'])}</option>"
        for option in spec["options"]
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{esc(spec['title'])}</title>"
        "<style>*{box-sizing:border-box}html,body{margin:0;width:100%;height:100%;overflow:hidden}"
        "body{font:18px sans-serif;background:#f1f4ed;display:grid;place-items:center}"
        "main{width:min(880px,calc(100% - 40px));padding:24px;background:white;"
        "border:2px solid #405565}h1{font-size:24px;margin:0 0 18px}"
        "fieldset{margin:0 0 18px;padding:14px;border:1px solid #405565}"
        "legend{padding:0 6px}.route-choice{display:inline-flex;align-items:center;gap:6px;"
        "margin:4px 18px 4px 0}.route-choice input{width:20px;height:20px;margin:0}"
        ".fields{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:18px}"
        ".fields label{display:flex;flex-direction:column;gap:6px}"
        ".fields input,.fields select{font:18px sans-serif;width:100%;height:38px}"
        "button{font:18px sans-serif;padding:10px 18px}#submission-summary{font-size:16px;"
        "line-height:24px;margin-top:18px}#submission-summary span{display:block}"
        "output{overflow-wrap:anywhere}</style></head><body><main>"
        f"<h1>{esc(spec['title'])}</h1><fieldset><legend>{esc(spec['radio_group_label'])}</legend>"
        f"{radios}</fieldset><div class='fields'>"
        f"<label>{esc(spec['field_label'])}<input id='entry-text' type='text' value='' "
        f"aria-label='{esc(spec['field_label'], quote=True)}'></label>"
        f"<label>{esc(spec['select_label'])}<select id='entry-select' "
        f"aria-label='{esc(spec['select_label'], quote=True)}'>"
        f"<option value='' selected disabled>{SELECT_PLACEHOLDER}</option>{options}</select></label>"
        f"</div><button type='button' id='commit-control'>{esc(spec['submit_label'])}</button>"
        "<div id='submission-summary' data-submitted='false' data-count='0'>"
        "<span id='progress-status'>No entry submitted.</span>"
        "<span>Recorded route: <output id='submitted-route' data-value=''></output></span>"
        "<span>Recorded note: <output id='submitted-note'></output></span>"
        "<span>Recorded mode: <output id='submitted-mode' data-value=''></output></span>"
        "</div></main>" + _EVENT_SCRIPT + "</body></html>"
    )


PRIVATE = """(() => {
 const el=id=>document.getElementById(id),text=id=>el(id)?.textContent??null;
 const note=el('entry-text'),mode=el('entry-select'),summary=el('submission-summary');
 const selected=mode?.selectedOptions?.[0],recorded=el('submitted-mode'),route=el('submitted-route');
 const radios=[...document.querySelectorAll('input[type="radio"]')].map(r=>({
  id:r.id,name:r.name,value:r.value,label:r.getAttribute('aria-label'),checked:r.checked}));
 return {title:document.title,ready:document.readyState,url:location.href,
  viewport_width:innerWidth,viewport_height:innerHeight,document_height:document.documentElement.scrollHeight,
  scroll_y:scrollY,radios,note_value:note?.value??null,mode_value:mode?.value??null,
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


def _state_selection(spec, state):
    _require(
        type(state) is dict and state.get("title") == spec["title"],
        "Fresh private state belongs to a different fixture",
    )
    selection = _radio_selection(spec, state)
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
    except ValueError:
        return False
    count = state["submission_count"]
    return (
        selection is not None
        and selection["value"] == state.get("submitted_radio_value") == spec["desired_radio_value"]
        and selection["label"] == state.get("submitted_radio_label") == spec["desired_radio_label"]
        and state["submitted"] is True
        and count > 0
        and state.get("status") == f"Submitted entry {count}."
        and state["note_value"] == state.get("submitted_note") == spec["desired_text"]
        and state["mode_value"] == state.get("submitted_mode_value") == spec["desired_option_value"]
        and state["mode_label"] == state.get("submitted_mode_label") == spec["desired_option_label"]
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
    """Choose an authored useful route, including when a prior field is already complete."""
    _check_spec(spec)
    selection = _state_selection(spec, private_state)
    if verify_goal(spec, private_state):
        return _descriptor("DONE", "terminal")
    if selection is None or selection["value"] != spec["desired_radio_value"]:
        return _descriptor("CLICK", "click", spec["desired_radio_label"])
    if private_state["note_value"] != spec["desired_text"]:
        return _descriptor("TYPE_TEXT", "fill", spec["field_label"], text=spec["desired_text"])
    if private_state["mode_value"] != spec["desired_option_value"]:
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
        _require(
            after_selection is not None
            and after_selection["value"] == value
            and after["radios"] == expected
            and unchanged(
                "note_value", "mode_value", "mode_label", "submission_count", *_RECORDED_FIELDS
            )
            and after["submitted"] is False
            and all(
                matched(kind, target, True, value, True) for kind in ("click", "input", "change")
            )
            and all(event["target"] == target for event in events),
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
        submit_click = ("click", "commit-control", True, "", None)
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
