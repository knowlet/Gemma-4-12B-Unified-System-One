"""Independently authored, deterministic workflow branch snapshots; no model IO."""

import copy
import hashlib
import json
import random

from request_builder import OPERATIONS, capture_request

DSL_VERSION = "independent-browser-workflow-dsl/1"
SEED = 271828
STAGES = (
    "acknowledge",
    "details",
    "classification",
    "below",
    "above",
    "pending",
    "complete",
    "denied",
)
SPLITS = ("train", "dev", "sealed")
# Split assignment is a family-level authoring decision, preceding every entity,
# phrasing, stage and permutation. All three ancestors of a family stay together.
FAMILIES = (
    (
        "appointment_edit",
        "train",
        "appointment",
        "visit reference",
        "appointment type",
        "Clinical review",
        "Verify contact permission",
        "Save appointment",
    ),
    (
        "document_upload",
        "train",
        "document submission",
        "document reference",
        "document category",
        "Compliance evidence",
        "Confirm attachment matches reference",
        "Submit document",
    ),
    (
        "booking_inventory",
        "train",
        "stock reservation",
        "reservation reference",
        "warehouse",
        "North depot",
        "Confirm requested quantity is reserved",
        "Confirm reservation",
    ),
    (
        "admin_filters",
        "train",
        "saved audit view",
        "view name",
        "account status filter",
        "Suspended accounts",
        "Include archived audit entries",
        "Apply and save view",
    ),
    (
        "expense_claim",
        "dev",
        "expense claim",
        "claim reference",
        "expense class",
        "Rail travel",
        "Attest receipt belongs to this claim",
        "Send claim for review",
    ),
    (
        "training_enrollment",
        "sealed",
        "course enrollment",
        "enrollment reference",
        "delivery session",
        "Morning workshop",
        "Accept prerequisite declaration",
        "Submit enrollment",
    ),
    (
        "equipment_checkout",
        "sealed",
        "equipment checkout",
        "checkout reference",
        "pickup desk",
        "Annex reception",
        "Confirm safety briefing completed",
        "Authorize checkout",
    ),
)
FAMILY_ORDERS = {
    "appointment_edit": ("details", "classification", "acknowledgement"),
    "document_upload": ("classification", "details", "acknowledgement"),
    "booking_inventory": ("acknowledgement", "details", "classification"),
    "admin_filters": ("classification", "acknowledgement", "details"),
    "expense_claim": ("details", "acknowledgement", "classification"),
    "training_enrollment": ("acknowledgement", "classification", "details"),
    "equipment_checkout": ("details", "classification", "acknowledgement"),
}
FAMILY_FACTS = {
    "appointment_edit": "The existing appointment and patient link are already selected; the calendar slot must stay unchanged.",
    "document_upload": "The supplied document packet is already staged for upload and previewed; no file content or local filename must be invented.",
    "booking_inventory": "The requested six storage units are available and held for this reservation; unrelated inventory must not be modified.",
    "admin_filters": "This is a saved administrative audit view; applying its requested filters must also persist the view configuration.",
    "expense_claim": "The itemized receipt packet is already attached; changing the travel class alone does not submit the claim.",
    "training_enrollment": "The requested workshop has an available seat and prerequisite evidence; accepting the declaration alone does not enroll the participant.",
    "equipment_checkout": "The requested equipment identifier is already selected and available; a pickup preference alone does not authorize the checkout.",
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _rng(*parts):
    return random.Random(int(hashlib.sha256(canonical([SEED, *parts]).encode()).hexdigest(), 16))


def solve(state):
    """Return every valid next (operation, semantic control) pair.

    Precedence follows visible preconditions and the explicit user ordering. In
    parallel mode no arbitrary lexical/first-target tie break is introduced.
    Labels are authored DSL semantics, never inferred from model predictions.
    """
    for key in ("acknowledged", "saved_evidence", "permanent_denial", "pending"):
        if type(state[key]) is not bool:
            raise ValueError("Semantic boolean must be exact")
    if state["ordering"] not in {
        "explicit_requirements",
        "parallel_details_and_classification",
    } or state["viewport"] not in {
        "required_section_visible",
        "above_required_section",
        "below_submit_section",
    }:
        raise ValueError("Unknown semantic ordering/viewport")
    if sorted(state["requirement_order"]) != ["acknowledgement", "classification", "details"]:
        raise ValueError("Invalid explicit requirement order")
    if state["permanent_denial"]:
        return {("BLOCKED", None)}
    if state["pending"]:
        return {("WAIT", None)}
    missing_by_step = {}
    if state["current_text"] != state["desired_text"]:
        missing_by_step["details"] = ("TYPE_TEXT", "details")
    if state["current_selection"] != state["desired_selection"]:
        missing_by_step["classification"] = ("SELECT", "classification")
    if not state["acknowledged"]:
        missing_by_step["acknowledgement"] = ("CLICK", "acknowledgement")
    missing = [
        missing_by_step[step] for step in state["requirement_order"] if step in missing_by_step
    ]
    if state["saved_evidence"] and missing:
        raise ValueError("Saved receipt contradicts required values")
    if not missing and state["saved_evidence"]:
        return {("DONE", None)}
    if state["viewport"] == "above_required_section":
        return {("SCROLL_DOWN", None)}
    if state["viewport"] == "below_submit_section":
        return {("SCROLL_UP", None)}
    if missing:
        if state["ordering"] == "parallel_details_and_classification":
            first = [v for v in missing if v[0] in {"TYPE_TEXT", "SELECT"}]
            if first:
                return set(first)
        return {missing[0]}
    return {("CLICK", "submit")}


def _state(family, ancestor, stage):
    desired = f"{family[0].replace('_', ' ')} record {ancestor + 1}"
    state = {
        "ordering": "explicit_requirements",
        "requirement_order": list(FAMILY_ORDERS[family[0]]),
        "desired_text": desired,
        "desired_selection": family[5],
        "current_text": desired,
        "current_selection": family[5],
        "acknowledged": True,
        "saved_evidence": False,
        "permanent_denial": False,
        "pending": False,
        "viewport": "required_section_visible",
    }
    if stage in {"details", "classification", "acknowledge"}:
        step = "acknowledgement" if stage == "acknowledge" else stage
        if not (stage == "acknowledge" and ancestor == 1):
            # All later requirements remain unfinished: completing any of them
            # first would violate this family's explicitly ordered goal.
            remaining = state["requirement_order"][state["requirement_order"].index(step) :]
            if "details" in remaining:
                state["current_text"] = ""
            if "classification" in remaining:
                state["current_selection"] = "Unassigned"
            if "acknowledgement" in remaining:
                state["acknowledged"] = False
    elif stage == "below":
        state.update(
            current_text="",
            current_selection="Unassigned",
            acknowledged=False,
            viewport="above_required_section",
        )
    elif stage == "above":
        state["viewport"] = "below_submit_section"
    elif stage == "pending":
        state["pending"] = True
    elif stage == "complete":
        state["saved_evidence"] = True
    elif stage == "denied":
        state.update(current_text="", permanent_denial=True)
    else:
        raise ValueError("Unknown authored stage")
    return state


def _page(family, ancestor, stage, state):
    family_id, _, noun, field, category, desired_option, check, submit = family
    record = f"{noun} {ancestor + 1}"
    prefix = ("Update", "Revise", "Prepare")[ancestor]
    steps = {
        "details": f"enter {field} exactly as '{state['desired_text']}'",
        "classification": f"select '{desired_option}' in {category}",
        "acknowledgement": f"check '{check}'",
    }
    ordered = [steps[key] for key in state["requirement_order"]]
    goal = (
        f"{prefix} {record}. Follow this required order: first {ordered[0]}; "
        f"then {ordered[1]}; then {ordered[2]}. Finally use '{submit}'. "
        "Skip requirements already visibly satisfied, preserving the order of those still missing. "
        "Finish only when the saved receipt visibly confirms all three requirements. "
        "Do not alter unrelated records or preferences."
    )
    text = [
        f"Editing {record}.",
        FAMILY_FACTS[family_id],
        f"{field}: {state['current_text'] or '(empty)'}. "
        f"{category}: {state['current_selection']}. {check}: "
        f"{'checked' if state['acknowledged'] else 'unchecked'}.",
    ]
    if state["pending"]:
        text += [
            "Submission is still processing. Required controls are temporarily disabled; "
            "the page will update automatically. No action can advance it yet."
        ]
    elif state["permanent_denial"]:
        text += [
            "Permanent permission denial: this account cannot edit or submit this record. "
            "No retry, alternate form, permission editor or supported recovery exists."
        ]
    elif state["saved_evidence"]:
        text += [
            f"Saved receipt for {record}: reference '{state['desired_text']}', "
            f"{category} '{desired_option}', acknowledgement checked. Submission complete."
        ]
    elif stage == "below":
        text += [
            "Required details are below the current viewport. Scroll down to reach them; "
            "the visible controls belong only to unrelated account preferences."
        ]
    elif stage == "above":
        text += [
            f"All requested fields are correct but unsaved. '{submit}' is above the current "
            "viewport. Scroll up to reach it. Visible controls are unrelated preferences."
        ]
    else:
        text += [
            f"'{submit}' is available only after details, classification and acknowledgement "
            "are complete. The required section is visible. No loading is in progress."
        ]
    relevant = stage not in {"below", "above", "pending", "denied"}
    actions = []
    semantic_ids = {}

    def add(kind, key, label, value="", **extra):
        identifier = f"{family_id}-{ancestor}-{key}"
        action = dict(kind=kind, id=identifier, node=identifier, label=label, value=value, **extra)
        actions.append(action)
        semantic_ids[key] = identifier

    add(
        "fill",
        "details",
        field if relevant else "Unrelated account nickname",
        state["current_text"] if relevant else "Guest",
        role="textbox",
    )
    for k, label in enumerate(
        ("Unrelated personal notes", "Unrelated directory query", "Unrelated mailing label")
    ):
        add("fill", f"field-distractor-{k}", label, "", role="textbox")
    add(
        "click",
        "acknowledgement",
        check if relevant else "Unrelated news subscription",
        role="checkbox",
        checked=state["acknowledged"] if relevant else False,
    )
    ready_to_submit = (
        state["current_text"] == state["desired_text"]
        and state["current_selection"] == state["desired_selection"]
        and state["acknowledged"]
    )
    add(
        "click",
        "submit",
        submit if relevant and ready_to_submit else "Open unrelated preferences",
        role="button",
    )
    add("click", "click-distractor-0", "Open keyboard help", role="button")
    add("click", "click-distractor-1", "Open unrelated record archive", role="button")
    for k, option in enumerate((desired_option, "Unassigned", "Other category", "Review queue")):
        add(
            "select",
            f"classification-option-{k}",
            f"{category if relevant else 'Unrelated digest category'} → {option}",
            option,
            role="combobox",
            current_value=state["current_selection"] if relevant else "Unassigned",
        )
        # The original upstream construction groups dropdown choices by node.
        actions[-1]["node"] = f"{family_id}-{ancestor}-classification"
        if k == 0:
            semantic_ids["classification"] = actions[-1]["id"]
    add("scroll", "SCROLL_DOWN", "Scroll down")
    add("scroll", "SCROLL_UP", "Scroll up")
    add("wait", "WAIT", "Wait for a pending page update")
    for action in actions:
        if action["kind"] in {"scroll", "wait"}:
            action["id"] = action["id"].rsplit("-", 1)[1]
    _rng(family_id, ancestor, stage, "observed-control-order").shuffle(actions)
    return (
        {
            "url": f"https://workflow-dsl.invalid/{family_id}/{ancestor + 1}",
            "title": f"{record} workspace",
            "text": " ".join(text),
            "actions": actions,
        },
        goal,
        semantic_ids,
    )


def permute_criteria(request, ancestor_ordinal):
    """Eight predeclared cyclic operation orders; target rotation applied below."""
    result = copy.deepcopy(request)
    original = result["questions"]["operation"]["criteria"]
    order = list(OPERATIONS)
    shift = ancestor_ordinal % len(order)
    order = order[-shift:] + order[:-shift] if shift else order
    result["questions"]["operation"]["criteria"] = {k: original[k] for k in order}
    return result


def labels_for(request, valid, targets, semantic_ids, target_position):
    """Bind complete valid sets to actual permuted criteria, never fixed codes."""
    questions = request["questions"]
    ops = {op for op, _ in valid}
    if not ops or not ops <= set(questions["operation"]["criteria"]):
        raise ValueError("Semantic operations are not offered")
    valid_heads = {"operation": ops}
    for op, semantic in valid:
        if semantic is None:
            continue
        matches = {k for k, a in targets[op].items() if a["id"] == semantic_ids[semantic]}
        if len(matches) != 1:
            raise ValueError("Semantic control must map to exactly one offered target")
        head = op.lower() + "_target"
        valid_heads.setdefault(head, set()).update(matches)
    # Permute all offered target heads, including unselected heads. Never remove
    # a distractor or fabricate labels for an unused target question.
    for head, question in questions.items():
        if head == "operation":
            continue
        criteria = question["criteria"]
        order = sorted(criteria)
        selected = sorted(valid_heads.get(head, ()))
        if selected:
            # Current released cases are singleton; generic labels below retain
            # all valid targets if an authored parallel case is introduced.
            anchor = selected[0]
            order.remove(anchor)
            order.insert(target_position % len(criteria), anchor)
        else:
            shift = target_position % len(order)
            order = order[shift:] + order[:shift]
        question["criteria"] = {k: criteria[k] for k in order}
    labels = {}
    for head, valid_ids in valid_heads.items():
        criteria = list(questions[head]["criteria"])
        labels[head] = {
            "criteria_ids": criteria,
            "valid_ids": [k for k in criteria if k in valid_ids],
            "valid_positions": [i for i, k in enumerate(criteria) if k in valid_ids],
            "soft_target": [1 / len(valid_ids) if k in valid_ids else 0.0 for k in criteria],
        }
    return labels


def authored_snapshot(family_id, ancestor, stage):
    """Canonical family/ancestor/branch definition, before request/label export."""
    matches = [family for family in FAMILIES if family[0] == family_id]
    if (
        len(matches) != 1
        or type(ancestor) is not int
        or ancestor not in (0, 1, 2)
        or stage not in STAGES
    ):
        raise ValueError("Unknown authored family/ancestor/stage")
    state = _state(matches[0], ancestor, stage)
    page, goal, semantic_ids = _page(matches[0], ancestor, stage, state)
    return state, page, goal, semantic_ids


def generate_cases():
    result = []
    for family_index, family in enumerate(FAMILIES):
        for ancestor in range(3):
            ancestor_id = f"{family[0]}/workflow-{ancestor + 1}"
            ordinal = family_index * 3 + ancestor
            for stage_index, stage in enumerate(STAGES):
                state, page, goal, semantic_ids = authored_snapshot(family[0], ancestor, stage)
                original, targets = capture_request(page, goal)
                request = permute_criteria(original, ordinal)
                valid = solve(state)
                labels = labels_for(request, valid, targets, semantic_ids, ordinal % 4)
                identifier = hashlib.sha256(
                    (DSL_VERSION + ancestor_id + stage).encode()
                ).hexdigest()[:24]
                result.append(
                    {
                        "case_id": identifier,
                        "dsl_version": DSL_VERSION,
                        "split": family[1],
                        "family_id": family[0],
                        "ancestor_id": ancestor_id,
                        "branch_stage": stage,
                        "state": state,
                        "page": page,
                        "goal": goal,
                        "semantic_control_ids": semantic_ids,
                        "request": request,
                        "request_sha256": digest(request),
                        "native_payload_sha256": digest(
                            {k: request[k] for k in ("state", "questions")}
                        ),
                        "labels": labels,
                        "variant_provenance": {
                            "ancestor_ordinal": ordinal,
                            "stage_ordinal": stage_index,
                            "operation_rotation": ordinal % 8,
                            "target_position": ordinal % 4,
                        },
                    }
                )
    return result
