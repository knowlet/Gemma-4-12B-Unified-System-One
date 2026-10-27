"""Independent TRAIN fixtures for link selection across actual documents.

No benchmark goals, traces or labels enter this module. Each scenario exposes
ordinary same-origin links to separately served documents. A collector must
serve *all and only* the returned paths, verify every browser navigation and
retain the publisher's original ordered action/target IDs. Rendering or unit
testing this module does not constitute a browser collection or model score.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata

SCHEMA = "decision3-owned-link-navigation-scenarios-v1"
TRANSITION_SCHEMA = "decision3-owned-link-navigation-transition-v1"
NAMESPACE = "decision3-owned-link-navigation-training-v1"
GOAL_NORMALIZATION = "NFKC-collapse-whitespace-v1"
DENSITIES = (8, 16, 24)
START_MODES = ("landing", "workspace", "complete")

# Authored TRAIN-only vocabulary. Similar links point to different documents;
# neither position nor an internal scenario family name identifies the target.
FAMILIES = (
    {
        "id": "maintenance",
        "title": "Maintenance desk",
        "target_link": "Open inspection queue",
        "distractors": (
            "Inspection history",
            "Inspection settings",
            "Repair queue",
            "Equipment list",
            "Service notices",
            "Maintenance guide",
            "Parts catalog",
            "Shift calendar",
            "Operator directory",
            "Audit exports",
            "Alert settings",
            "Work summaries",
            "Repair history",
            "Asset tags",
            "Supply requests",
            "Service contacts",
            "Inspection templates",
            "Team schedule",
            "Access settings",
            "Device inventory",
            "Training resources",
            "Support tickets",
            "Reports archive",
        ),
        "record_prefix": "Inspection record",
        "action": "Mark inspected",
        "pending": "Inspection is pending.",
        "completed": "Inspection recorded.",
    },
    {
        "id": "booking",
        "title": "Reservation office",
        "target_link": "Open reservation queue",
        "distractors": (
            "Reservation history",
            "Reservation settings",
            "Booking calendar",
            "Room directory",
            "Service notices",
            "Visitor guide",
            "Request templates",
            "Office calendar",
            "Staff directory",
            "Archive exports",
            "Alert settings",
            "Booking summaries",
            "Cancellation history",
            "Room tags",
            "Supply requests",
            "Office contacts",
            "Reservation templates",
            "Team schedule",
            "Access settings",
            "Room inventory",
            "Training resources",
            "Support tickets",
            "Reports archive",
        ),
        "record_prefix": "Reservation record",
        "action": "Confirm reservation",
        "pending": "Reservation is pending.",
        "completed": "Reservation confirmed.",
    },
    {
        "id": "catalog",
        "title": "Collection catalog",
        "target_link": "Open catalog review queue",
        "distractors": (
            "Catalog review history",
            "Catalog settings",
            "Collection index",
            "Archive directory",
            "Service notices",
            "Catalog guide",
            "Record templates",
            "Archive calendar",
            "Curator directory",
            "Archive exports",
            "Alert settings",
            "Catalog summaries",
            "Record history",
            "Collection tags",
            "Supply requests",
            "Archive contacts",
            "Review templates",
            "Team schedule",
            "Access settings",
            "Collection inventory",
            "Training resources",
            "Support tickets",
            "Reports archive",
        ),
        "record_prefix": "Catalog record",
        "action": "Mark catalog reviewed",
        "pending": "Catalog review is pending.",
        "completed": "Catalog review recorded.",
    },
)


def _require(value, message):
    if not value:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _goal_hash(goal):
    _require(
        type(goal) is str and bool(goal.strip()), "Nonempty independent training goal required"
    )
    return hashlib.sha256(
        " ".join(unicodedata.normalize("NFKC", goal).split()).encode()
    ).hexdigest()


def _axis(seed, family, mode, density, name):
    return hashlib.sha256(
        f"{NAMESPACE}:{seed}:{family}:{mode}:{density}:{name}".encode()
    ).hexdigest()


def _ordered(seed, family, mode, density, name, values):
    return sorted(
        values, key=lambda value: _axis(seed, family, mode, density, name + ":" + str(value))
    )


def _validate_exclusions(values):
    _require(
        isinstance(values, (set, frozenset, list, tuple)) and bool(values),
        "Nonempty hash-only exclusions required",
    )
    _require(
        all(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) for value in values)
        and len(set(values)) == len(values),
        "Distinct lowercase SHA256 exclusions required",
    )
    return frozenset(values)


def _check_spec(spec):
    _require(
        type(spec) is dict
        and spec.get("schema_id") == SCHEMA
        and spec.get("namespace") == NAMESPACE
        and spec.get("split") == "train",
        "Exact independent TRAIN scenario required",
    )
    _require(
        type(spec.get("seed")) is int
        and spec["seed"] >= 0
        and spec.get("density") in DENSITIES
        and spec.get("start_mode") in START_MODES,
        "Explicit seed, density and start mode required",
    )
    _require(
        re.fullmatch(r"d3-link-train-[0-9a-f]{20}", spec.get("id", "")) is not None
        and re.fullmatch(r"d3-link-owned-[0-9a-f]{20}", spec.get("ancestor_id", "")) is not None,
        "Distinct opaque scenario and ancestor identities required",
    )
    _require(
        spec.get("goal_sha256") == _goal_hash(spec.get("goal")),
        "Original normalized goal digest required",
    )
    base = "/link-training/" + spec["id"]
    expected = {
        "landing": base + "/index",
        "workspace": base + "/workspace",
        "detail": base + "/detail",
    }
    _require(spec.get("paths") == expected, "Exact independent same-origin document paths required")
    _require(
        spec.get("start_path")
        == expected[
            {"landing": "landing", "workspace": "workspace", "complete": "detail"}[
                spec["start_mode"]
            ]
        ],
        "Start mode and document path differ",
    )
    for key in ("landing_links", "workspace_links"):
        links = spec.get(key)
        _require(
            type(links) is list
            and len(links) == spec["density"]
            and all(
                type(item) is dict
                and set(item) == {"label", "path"}
                and type(item["label"]) is str
                and item["label"].strip()
                and type(item["path"]) is str
                and item["path"].startswith(base + "/")
                for item in links
            )
            and len({item["label"] for item in links}) == len(links)
            and len({item["path"] for item in links}) == len(links),
            "Complete unique labeled links and local document paths required",
        )
    _require(
        sum(
            x["label"] == spec["target_link"] and x["path"] == expected["workspace"]
            for x in spec["landing_links"]
        )
        == 1
        and sum(
            x["label"] == spec["record_label"] and x["path"] == expected["detail"]
            for x in spec["workspace_links"]
        )
        == 1,
        "One unique expected real link in each document required",
    )
    decoys = spec.get("decoy_paths")
    expected_decoys = {
        x["path"] for name in ("landing_links", "workspace_links") for x in spec[name]
    } - set(expected.values())
    _require(
        type(decoys) is list and len(decoys) == len(set(decoys)) and set(decoys) == expected_decoys,
        "All and only independent local distractor documents required",
    )
    for key in (
        "title",
        "target_link",
        "record_label",
        "action_label",
        "pending_text",
        "completed_text",
    ):
        _require(
            type(spec.get(key)) is str and bool(spec[key].strip()),
            "Complete natural-language fixture values required",
        )
    _require(
        spec["pending_text"] != spec["completed_text"], "Pending and completed evidence must differ"
    )


def scenarios(seed, excluded_goal_hashes):
    """Return deterministic, goal-disjoint authored scenarios with 3 real documents.

    `density` is the number of authored links per navigation document. It is
    not an assertion about the native publisher's observed candidate count.
    """
    _require(type(seed) is int and 0 <= seed < 2**32, "Explicit bounded integer seed required")
    exclusions = _validate_exclusions(excluded_goal_hashes)
    result = []
    for family in FAMILIES:
        for density in DENSITIES:
            for mode in START_MODES:

                def axis(name):
                    return _axis(seed, family["id"], mode, density, name)

                identity = "d3-link-train-" + axis("scenario")[:20]
                base = "/link-training/" + identity
                paths = {
                    "landing": base + "/index",
                    "workspace": base + "/workspace",
                    "detail": base + "/detail",
                }
                record_token = axis("target-record")[:8]
                record_label = family["record_prefix"] + " " + record_token
                landing_decoys = _ordered(
                    seed, family["id"], mode, density, "landing-label", family["distractors"]
                )[: density - 1]
                landing = [{"label": family["target_link"], "path": paths["workspace"]}]
                landing += [
                    {"label": label, "path": base + "/landing-decoy-" + str(index)}
                    for index, label in enumerate(landing_decoys, 1)
                ]
                workspace = [{"label": record_label, "path": paths["detail"]}]
                workspace += [
                    {
                        "label": family["record_prefix"]
                        + " "
                        + axis("decoy-record-" + str(index))[:8],
                        "path": base + "/record-decoy-" + str(index),
                    }
                    for index in range(1, density)
                ]
                landing = _ordered(seed, family["id"], mode, density, "landing-order", landing)
                workspace = _ordered(seed, family["id"], mode, density, "record-order", workspace)
                goal = (
                    f"In {family['title']}, find {record_label} and use {family['action']}. "
                    "Finish only after the page shows the requested action is complete."
                )
                spec = {
                    "schema_id": SCHEMA,
                    "namespace": NAMESPACE,
                    "split": "train",
                    "seed": seed,
                    "id": identity,
                    "ancestor_id": "d3-link-owned-" + axis("ancestor")[:20],
                    "density": density,
                    "start_mode": mode,
                    "title": family["title"],
                    "goal": goal,
                    "goal_sha256": _goal_hash(goal),
                    "paths": paths,
                    "start_path": paths[
                        {"landing": "landing", "workspace": "workspace", "complete": "detail"}[mode]
                    ],
                    "target_link": family["target_link"],
                    "record_label": record_label,
                    "action_label": family["action"],
                    "pending_text": family["pending"],
                    "completed_text": family["completed"],
                    "landing_links": landing,
                    "workspace_links": workspace,
                    "decoy_paths": [
                        x["path"]
                        for links in (landing, workspace)
                        for x in links
                        if x["path"] not in (paths["workspace"], paths["detail"])
                    ],
                }
                _check_spec(spec)
                _require(
                    spec["goal_sha256"] not in exclusions, "TRAIN goal overlaps excluded inventory"
                )
                result.append(spec)
    _require(
        len({x["id"] for x in result}) == len(result)
        and len({x["ancestor_id"] for x in result}) == len(result)
        and len({x["goal_sha256"] for x in result}) == len(result),
        "Distinct authored identities and goals required",
    )
    return result


def render_documents(spec):
    """Return independently served same-origin HTML keyed by exact path."""
    _check_spec(spec)
    escape = html.escape

    def document(view, heading, body):
        return (
            "<!doctype html><html><head><meta charset='utf-8'><title>"
            + escape(spec["title"])
            + "</title><meta name='viewport' content='width=device-width, initial-scale=1'>"
            + "<style>body{margin:24px;font-family:system-ui,sans-serif;line-height:1.5}"
            + "main{max-width:72rem}ul{display:grid;grid-template-columns:repeat(auto-fit,minmax(16rem,1fr));"
            + "gap:8px 16px;list-style:none;padding:0}li a{display:block;padding:8px}</style>"
            + "</head><body><main data-view='"
            + view
            + "'><h1>"
            + escape(heading)
            + "</h1>"
            + body
            + "</main></body></html>"
        )

    def links(values):
        return "".join(
            "<li><a href='"
            + escape(x["path"], quote=True)
            + "'>"
            + escape(x["label"])
            + "</a></li>"
            for x in values
        )

    documents = {
        spec["paths"]["landing"]: document(
            "landing",
            spec["title"],
            "<nav aria-label='Sections'><ul>" + links(spec["landing_links"]) + "</ul></nav>",
        ),
        spec["paths"]["workspace"]: document(
            "workspace", spec["target_link"], "<ul>" + links(spec["workspace_links"]) + "</ul>"
        ),
    }
    initial = spec["start_mode"] == "complete"
    status = spec["completed_text"] if initial else spec["pending_text"]
    count = 1 if initial else 0
    action = (
        "<button type='button' id='complete-record'>"
        + escape(spec["action_label"])
        + "</button>"
        + "<p id='record-status' data-count='"
        + str(count)
        + "'>"
        + escape(status)
        + "</p>"
        + "<script>window.__linkTrainingEvents=[];document.getElementById('complete-record').addEventListener('click',function(e){"
        + "const s=document.getElementById('record-status');if(Number(s.dataset.count)===0){s.dataset.count='1';"
        + "s.textContent="
        + json.dumps(spec["completed_text"], ensure_ascii=False).replace("<", "\\u003c")
        + ";}"
        + "window.__linkTrainingEvents.push({type:'click',id:'complete-record',trusted:e.isTrusted});});</script>"
    )
    documents[spec["paths"]["detail"]] = document("detail", spec["record_label"], action)
    for path in spec["decoy_paths"]:
        documents[path] = document(
            "distractor",
            "Other destination",
            "<p>This destination is not the requested record.</p>",
        )
    _require(
        len(documents) == 3 + len(spec["decoy_paths"]), "All unique authored documents required"
    )
    return documents


def private_state_expression():
    """DOM evidence; the collector owns and authenticates each actual evaluation."""
    return """(() => {
      const main=document.querySelector('main'), status=document.getElementById('record-status');
      return {pathname:location.pathname, view:main?.getAttribute('data-view')??null,
        ready:document.readyState, heading:main?.querySelector('h1')?.textContent??null,
        status:status?.textContent??null, count:status?Number(status.dataset.count):null,
        links:[...document.querySelectorAll('main a')].map(a=>({label:a.textContent, path:new URL(a.href).pathname})),
        events:structuredClone(window.__linkTrainingEvents||[])};
    })()"""


def _state(spec, state):
    _check_spec(spec)
    _require(
        type(state) is dict
        and state.get("ready") == "complete"
        and type(state.get("links")) is list
        and type(state.get("events")) is list,
        "Complete current document evidence required",
    )
    path, view = state.get("pathname"), state.get("view")
    _require(
        path in spec["paths"].values()
        and view in ("landing", "workspace", "detail")
        and spec["paths"][view] == path,
        "Actual current document must belong to scenario",
    )
    expected_heading = {
        "landing": spec["title"],
        "workspace": spec["target_link"],
        "detail": spec["record_label"],
    }[view]
    _require(state.get("heading") == expected_heading, "Visible current document heading differs")
    if view in ("landing", "workspace"):
        _require(
            state["links"] == spec[view + "_links"]
            and state.get("status") is None
            and state.get("count") is None,
            "Exact visible link order and no false completion required",
        )
    else:
        _require(
            not state["links"]
            and type(state.get("count")) is int
            and state["count"] in (0, 1)
            and state.get("status")
            == (spec["completed_text"] if state["count"] else spec["pending_text"]),
            "Visible selected-record completion evidence differs",
        )
    return view


def verify_goal(spec, private_state):
    try:
        view = _state(spec, private_state)
    except (ValueError, TypeError, KeyError):
        return False
    return (
        view == "detail"
        and private_state["count"] == 1
        and private_state["status"] == spec["completed_text"]
    )


def _descriptor(operation, kind, label=None):
    return {"operation": operation, "kind": kind, "label": label, "action_id": None, "text": None}


def teacher_step(spec, private_state):
    view = _state(spec, private_state)
    if verify_goal(spec, private_state):
        return _descriptor("DONE", "terminal")
    if view == "landing":
        return _descriptor("CLICK", "click", spec["target_link"])
    if view == "workspace":
        return _descriptor("CLICK", "click", spec["record_label"])
    return _descriptor("CLICK", "click", spec["action_label"])


def verify_transition(spec, before, after, step, events):
    """Check one actual navigation or visible completion, not a model prediction."""
    before_view, after_view = _state(spec, before), _state(spec, after)
    expected = teacher_step(spec, before)
    _require(
        type(step) is dict and step == expected and type(events) is list,
        "Actual teacher descriptor required",
    )
    if expected["operation"] == "DONE":
        _require(
            before == after and verify_goal(spec, after) and not events,
            "DONE requires unchanged visible completion",
        )
        kind = "terminal"
    elif before_view == "landing":
        _require(
            after_view == "workspace" and not verify_goal(spec, after),
            "Actual target workspace document required",
        )
        kind = "document_navigation"
    elif before_view == "workspace":
        _require(
            after_view == "detail" and not verify_goal(spec, after),
            "Actual target detail document required",
        )
        kind = "document_navigation"
    else:
        _require(
            after_view == "detail" and before["count"] == 0 and verify_goal(spec, after),
            "Actual pending-to-complete visible transition required",
        )
        _require(
            len(events) == 1
            and events[0] == {"type": "click", "id": "complete-record", "trusted": True}
            and after["events"] == before["events"] + events,
            "One genuine completion click required",
        )
        kind = "visible_completion"
    return {
        "schema_id": TRANSITION_SCHEMA,
        "kind": kind,
        "source_path": before["pathname"],
        "destination_path": after["pathname"],
        "teacher_operation": expected["operation"],
        "teacher_label": expected["label"],
        # This expresses intent only. The collector must separately authenticate
        # the native action and map its target before creating supervised rows.
        "supervision_intent": {"operation": True, "click_target": expected["operation"] == "CLICK"},
        "before_sha256": hashlib.sha256(_canonical(before)).hexdigest(),
        "after_sha256": hashlib.sha256(_canonical(after)).hexdigest(),
        "teacher_sha256": hashlib.sha256(_canonical(step)).hexdigest(),
        "model_score": None,
    }
