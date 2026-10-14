#!/usr/bin/env python3
"""Collect development-only browser supervision, never model evaluation scores.

The scripted teacher supplies training labels and field values. The pinned Jev
Agent still constructs every wire request and executes guarded browser actions.
Only operation and the executed operation's target are supervised. Gold data
stays outside the preserved inference payload; holdout fixtures are never read.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

from jev_browser_benchmark import (
    TASK_IDS,
    FixtureCase,
    PlaywrightCDP,
    fixture_server,
    load_upstream,
    make_cases,
    make_upstream_travel_case,
    upstream_manifest,
    verify_fixture,
)


@dataclass(frozen=True)
class TeachingAction:
    label: str
    kind: str = "click"
    text: str | None = None

    @property
    def operation(self):
        return {"click": "CLICK", "fill": "TYPE_TEXT", "select": "SELECT"}[self.kind]


def make_training_travel_cases(upstream_path, repeat=0):
    """Training-only compositions on exposed HTML; no evaluation goal factory."""
    if type(repeat) is not int or repeat < 0:
        raise ValueError("Training travel repeat must be a nonnegative integer")
    original = make_upstream_travel_case(upstream_path, repeat)
    properties = (
        ("Casa Flora", "Lisbon", "Design", "#casa-flora"),
        ("The Glasshouse", "Copenhagen", "Design", "#the-glasshouse"),
        ("Serra Lodge", "Lisbon", "Nature", "#serra-lodge"),
    )
    cases = []
    for property_index, (name, city, category, anchor) in enumerate(properties):
        for kind in ("open", "query", "category", "query_category", "full"):
            expected = {"hash": anchor, "name": name}
            requirements = []
            if kind in {"query", "query_category", "full"}:
                expected.update(query=city, searched=True)
                requirements.append(f"enter {city} in Destination and submit Find stays")
            if kind in {"category", "query_category", "full"}:
                expected["category"] = category
                requirements.append(f"set Stay category to {category}")
            if kind == "full":
                enabled = property_index != 2 and (repeat + property_index) % 2 == 0
                expected["free"] = enabled
                requirements.append(
                    "enable Free cancellation" if enabled else "keep Free cancellation disabled"
                )
            prefix = (
                "Before viewing the requested property, ",
                "Complete these controls before opening the property: ",
                "Prepare the stay listing by completing the following: ",
                "Use the listing controls first: ",
            )[repeat % 4]
            if requirements:
                goal = (
                    prefix
                    + "; ".join(requirements)
                    + f". Then view {name} and finish when its details are visible."
                )
            else:
                goal = (
                    f"Show the detail page for {name}; no search or filter changes are requested.",
                    f"View {name} and finish on its property details, without extra search requirements.",
                    f"The only requested action is viewing {name}'s details. Finish there.",
                    f"Navigate directly to the property details of {name}, then stop.",
                )[repeat % 4]
            cases.append(
                FixtureCase(
                    case_id=f"training-travel-{repeat:03d}-{len(cases):02d}",
                    task_id="upstream_travel",
                    goal=goal,
                    html=original.html,
                    expected=expected,
                    needs_text="query" in expected,
                    query_string=original.query_string,
                    source={
                        **original.source,
                        "scope": "development-only diversified training goals on exposed unchanged upstream HTML",
                    },
                )
            )
    return cases


def development_phases(case):
    """Explicit development teacher; each phase contains independent prerequisites."""
    action = TeachingAction
    expected, task = case.expected, case.task_id
    if task in {"filter_open", "search_filter_open"}:
        prerequisites = [
            action("Region → " + expected["region"], "select"),
            action("Verified only"),
        ]
        if case.needs_text:
            prerequisites.append(action("Search records", "fill", expected["name"]))
        return [prerequisites, [action("Apply filters")], [action("Open " + expected["name"])]]
    if task == "already_checked":
        return [[action("Product tips")], [action("Save preferences")]]
    if task == "native_select":
        return [
            [action("Review mode → " + expected["mode"], "select")],
            [action("Save review mode")],
        ]
    if task == "autocomplete":
        return [
            [action("Destination", "fill", expected["city"])],
            [action(expected["city"])],
            [action("Confirm destination")],
        ]
    if task == "submit_populated":
        return [[action("Submit enrollment")]]
    if task == "premature_done":
        return [[action("Activate report")]]
    if task == "completion_toggle":
        return [[action("Mark complete")]]
    if task == "unapplied_search":
        return [
            [
                action("Search area", "fill", expected["query"]),
                action("Collection → " + expected["category"], "select"),
                action("Verified only"),
            ],
            [action("Apply search")],
            [action("Open " + expected["name"])],
        ]
    if task == "upstream_travel":
        prerequisites = []
        if "query" in expected:
            prerequisites.append(action("Destination", "fill", expected["query"]))
        if "category" in expected:
            prerequisites.append(action("Stay category → " + expected["category"], "select"))
        if expected.get("free") is True:
            prerequisites.append(action("Free cancellation"))
        return (
            ([prerequisites] if prerequisites else [])
            + ([[action("Find stays")]] if "query" in expected else [])
            + [[action("View " + expected["name"])]]
        )
    raise ValueError("Unknown development teaching task")


def _available(page, teaching_action):
    matches = [
        item
        for item in page["actions"]
        if item["kind"] == teaching_action.kind and item["label"] == teaching_action.label
    ]
    if len(matches) != 1:
        raise ValueError(f"Training action is not uniquely observed: {teaching_action.label}")
    return matches[0]


def _remaining_travel_actions(case, verification):
    """Use fresh real application state, outside every inference payload."""
    if verification["success"]:
        return []
    observed = verification["observed"]
    actual, expected = observed["result"], case.expected
    if actual.get("hash"):
        return [TeachingAction("← All stays")]
    actions = []
    query_pending = "query" in expected and (
        actual.get("query") != expected["query"] or actual.get("searched") is not True
    )
    if query_pending and observed.get("typed_destination") != expected["query"]:
        actions.append(TeachingAction("Destination", "fill", expected["query"]))
    if "category" in expected and actual.get("category") != expected["category"]:
        actions.append(TeachingAction("Stay category → " + expected["category"], "select"))
    if "free" in expected and actual.get("free") is not expected["free"]:
        actions.append(TeachingAction("Free cancellation"))
    if actions:
        return actions
    if query_pending:
        return [TeachingAction("Find stays")]
    return [TeachingAction("View " + expected["name"])]


def _training_recovery_actions(case, repeat):
    """Bounded genuine initial actions, including observed repeated navigation."""
    expected = case.expected
    mode = repeat % 4
    correct_filters = [TeachingAction("Stay category → " + expected["category"], "select")]
    if expected["free"]:
        correct_filters.append(TeachingAction("Free cancellation"))
    if mode == 0:
        return (
            *correct_filters,
            TeachingAction("Find stays"),
            TeachingAction("Find a stay"),
            TeachingAction("Find a stay"),
        )
    if mode == 1:
        return (*correct_filters, TeachingAction("Destination", "fill", expected["query"]))
    wrong_free = [TeachingAction("Free cancellation")] if not expected["free"] else []
    if mode == 2:
        return (*wrong_free, TeachingAction("Find stays"), TeachingAction("Find stays"))
    return (
        TeachingAction("Destination", "fill", expected["query"]),
        TeachingAction("Find stays"),
        TeachingAction("Stay category → " + expected["category"], "select"),
        *wrong_free,
        TeachingAction("Find a stay"),
    )


def _distribution(labels, accepted):
    accepted = set(accepted)
    if not accepted or not accepted.issubset(labels):
        raise ValueError("Training labels must be observed candidates")
    return {label: 1.0 / len(accepted) if label in accepted else 0.0 for label in labels}


def training_case(payload, *, case_id, group_id, task_id, split, selected, valid_targets):
    """Retain exact native head wording while excluding unused target supervision."""
    from s1.evaluation.datasets import EvaluationCase
    from s1.jev import _parse_payload

    parsed = _parse_payload(payload, model_identity=payload["model"])
    questions = {question.id: question for question in parsed.questions}
    operation = selected["operation"]
    used = ["operation"]
    gold = {"operation": operation}
    soft = {"operation": _distribution(questions["operation"].labels(), valid_targets)}
    if operation != "DONE":
        head = operation.lower() + "_target"
        # A singleton is mathematically forced by Jev's compatibility contract,
        # not a learned classifier sample. Its original wire evidence remains.
        if len(questions[head].labels()) > 1:
            used.append(head)
            gold[head] = selected["target"]
            soft[head] = _distribution(questions[head].labels(), valid_targets[operation])
    request = parsed.model_copy(update={"questions": [questions[head] for head in used]})
    return EvaluationCase(
        id=case_id,
        group_id=group_id,
        task_id=task_id,
        split=split,
        request=request,
        gold=gold,
        soft_gold=soft,
        critical_questions=used,
        schema_id="jev-development-supervision-v1",
    )


def collect_episode(
    agent,
    agent_module,
    model_module,
    case,
    *,
    rng,
    split,
    episode_id,
    rows,
    wires,
    recover_early_open=False,
    recovery_prerequisites=(),
    recovery_submitted=False,
    dynamic_travel=False,
    recovery_initialization=None,
    counterfactual_initialization=None,
):
    phases = development_phases(case)
    if type(dynamic_travel) is not bool or (
        dynamic_travel and (split != "train" or case.task_id != "upstream_travel")
    ):
        raise ValueError("Dynamic travel supervision is an explicit train-only option")
    if recovery_initialization is not None and (
        not dynamic_travel
        or not recover_early_open
        or not isinstance(recovery_initialization, tuple)
        or any(not isinstance(action, TeachingAction) for action in recovery_initialization)
    ):
        raise ValueError("Dynamic recovery requires explicit guarded initialization actions")
    if counterfactual_initialization is not None and (
        not dynamic_travel
        or recover_early_open
        or recovery_initialization is not None
        or recovery_prerequisites
        or recovery_submitted
        or not isinstance(counterfactual_initialization, tuple)
        or not counterfactual_initialization
        or any(not isinstance(action, TeachingAction) for action in counterfactual_initialization)
    ):
        raise ValueError("Counterfactual initialization requires an explicit train-only schedule")
    if recovery_prerequisites or recovery_submitted:
        if not recover_early_open:
            raise ValueError("Partial initialization requires exposed travel recovery")
    if counterfactual_initialization is not None:
        initialization_history = []
        for teaching_action in counterfactual_initialization:
            before = agent.state["page"]
            action = _available(before, teaching_action)
            agent.browser.act(action, before, text=teaching_action.text)
            agent.state["page"] = agent.browser.observe(screenshot=False)
            history = {
                "action": action["label"],
                "kind": action["kind"],
                "text": teaching_action.text,
                "page_changed": before["fingerprint"] != agent.state["page"]["fingerprint"],
            }
            agent.state["history"].append(history)
            initialization_history.append(copy.deepcopy(history))
        # A no-query goal may already be satisfied after the same early open
        # that is incomplete for its query-required partner.
        initialization_dom = verify_fixture(agent.browser, case)
        initialization_snapshot = copy.deepcopy(agent.state["page"])
        initialization_html_sha256 = hashlib.sha256(
            agent.browser.evaluate("document.documentElement.outerHTML").encode()
        ).hexdigest()
    elif recover_early_open and dynamic_travel:
        if (
            recovery_initialization is None
            or not {"query", "category", "free"} <= case.expected.keys()
        ):
            raise ValueError(
                "Broad recovery requires a full travel goal and explicit initialization"
            )
        initialization_history = []
        initialize = list(recovery_initialization)
        # A wrong cancellation filter can hide Serra Lodge. Preserve that real
        # unsatisfied listing rather than inventing an unavailable View action.
        view = TeachingAction("View " + case.expected["name"])
        for teaching_action in [*initialize, view]:
            matches = [
                item
                for item in agent.state["page"]["actions"]
                if item["kind"] == teaching_action.kind and item["label"] == teaching_action.label
            ]
            if teaching_action is view and not matches:
                continue
            before = agent.state["page"]
            action = _available(before, teaching_action)
            agent.browser.act(action, before, text=teaching_action.text)
            agent.state["page"] = agent.browser.observe(screenshot=False)
            history = {
                "action": action["label"],
                "kind": action["kind"],
                "text": teaching_action.text,
                "page_changed": before["fingerprint"] != agent.state["page"]["fingerprint"],
            }
            agent.state["history"].append(history)
            initialization_history.append(copy.deepcopy(history))
        initialization_dom = verify_fixture(agent.browser, case)
        if initialization_dom["success"]:
            raise ValueError("Broad recovery must begin with an unsatisfied actual DOM")
    elif recover_early_open:
        if split != "train" or case.task_id != "upstream_travel":
            raise ValueError("Exposed travel recovery is training only")
        if (
            type(recovery_submitted) is not bool
            or not isinstance(recovery_prerequisites, (tuple, list))
            or len(set(recovery_prerequisites)) != len(recovery_prerequisites)
            or any(
                type(index) is not int or index not in range(3) for index in recovery_prerequisites
            )
            or (len(recovery_prerequisites) == 3 and recovery_submitted)
        ):
            raise ValueError("Recovery must initialize an incomplete subset of travel requirements")
        initialize = [phases[0][index] for index in recovery_prerequisites]
        if recovery_submitted:
            initialize.append(TeachingAction("Find stays"))
        initialize.append(TeachingAction("View Casa Flora"))
        initialization_history = []
        for teaching_action in initialize:
            before = agent.state["page"]
            action = _available(before, teaching_action)
            agent.browser.act(action, before, text=teaching_action.text)
            agent.state["page"] = agent.browser.observe(screenshot=False)
            history = {
                "action": action["label"],
                "kind": action["kind"],
                "text": teaching_action.text,
                "page_changed": before["fingerprint"] != agent.state["page"]["fingerprint"],
            }
            agent.state["history"].append(history)
            initialization_history.append(copy.deepcopy(history))
        initialization_dom = verify_fixture(agent.browser, case)
        if initialization_dom["success"]:
            raise ValueError("Early-open recovery must begin with an unsatisfied actual DOM")
        phases[0] = [
            action
            for index, action in enumerate(phases[0])
            if index not in recovery_prerequisites or (index == 0 and not recovery_submitted)
        ]
        phases.insert(0, [TeachingAction("← All stays")])
    current = []
    selected_action = None
    selected_page_action = None
    pending_targets = None
    step = 0
    group = "browser-development:" + hashlib.sha256(case.html.encode()).hexdigest()

    def post_json(url, _key, payload):
        nonlocal step
        if url != "https://api.typesafe.ai/v1/systemone":
            raise ValueError("Training collector permits only the intercepted policy call")
        elements, targets, _controls = model_module.action_space(agent.state["page"]["actions"])
        del elements
        if selected_action is None:
            operation, target = "DONE", None
            if not verify_fixture(agent.browser, case)["success"]:
                raise ValueError("Positive DONE training label requires a fresh satisfied DOM")
            accepted = {"DONE": []}
        else:
            operation = selected_action.operation
            target = next(
                index
                for index, item in targets[operation].items()
                if item["id"] == selected_page_action["id"]
            )
            accepted = pending_targets
        chosen = {"operation": operation, "target": target}
        row = training_case(
            payload,
            case_id=f"{episode_id}-step-{step:03d}",
            group_id=group,
            task_id=case.task_id,
            split=split,
            selected=chosen,
            valid_targets=accepted,
        )
        rows.append(row)
        wires.append(
            {
                "case_id": row.id,
                "payload": copy.deepcopy(payload),
                "label_source": "scripted development teacher, fresh observed candidate IDs",
                "teacher_choice": chosen,
                "valid_next_targets": accepted,
            }
        )
        step += 1
        # These probabilities belong to a scripted teacher, never a measured model.
        answers = {}
        for head, question in payload["questions"].items():
            labels = list(question["criteria"])
            label = (
                operation
                if head == "operation"
                else target
                if head == operation.lower() + "_target"
                else labels[0]
            )
            answers[head] = {
                "choice": label,
                "probabilities": {item: float(item == label) for item in labels},
                "confidence": 1.0,
            }
        return {"answers": answers, "model": "scripted-development-teacher", "usage": {}}

    def field_text(_context):
        if (
            selected_action is None
            or selected_action.kind != "fill"
            or selected_action.text is None
        ):
            raise ValueError("Training teacher has no field value for this action")
        return selected_action.text, {
            "model": "scripted-development-teacher",
            "latency_ms": 0,
            "usage": {},
        }

    model_module.post_json = post_json
    model_module.field_text = field_text
    agent_module.field_text = field_text

    def teach_step(candidates):
        nonlocal selected_action, selected_page_action, pending_targets
        page = agent.state["page"]
        _elements, targets, _controls = model_module.action_space(page["actions"])
        pending_targets = {}
        for candidate in candidates:
            item = _available(page, candidate)
            label = next(
                index
                for index, action in targets[candidate.operation].items()
                if action["id"] == item["id"]
            )
            pending_targets.setdefault(candidate.operation, []).append(label)
        selected_action = rng.choice(candidates)
        selected_page_action = _available(page, selected_action)
        before = len(wires)
        snapshot = agent.command("tick")
        if len(wires) != before + 1 or snapshot["status"] != "ready":
            raise ValueError("Training trajectory did not perform one guarded action")
        return selected_action

    if dynamic_travel:
        for _ in range(24):
            current = _remaining_travel_actions(case, verify_fixture(agent.browser, case))
            if not current:
                break
            teach_step(current)
        else:
            raise ValueError("Dynamic travel supervision exceeded its guarded action budget")
    else:
        for phase in phases:
            current = list(phase)
            while current:
                current.remove(teach_step(current))
    selected_action = None
    before = len(wires)
    result = agent.command("tick")
    verified = verify_fixture(agent.browser, case)
    if len(wires) != before + 1 or result["status"] != "done" or not verified["success"]:
        raise ValueError("Development teaching trajectory did not complete its actual DOM goal")
    result = {
        "episode_id": episode_id,
        "task_id": case.task_id,
        "group_id": group,
        "steps": step,
        "verified_dom": verified,
        "scope": "scripted training collection, no model score",
        "initialization": "guarded early-open recovery" if recover_early_open else "fresh page",
    }
    if counterfactual_initialization is not None:
        result["initialization"] = "guarded counterfactual training initialization"
        result["initialization_history"] = initialization_history
        result["initial_verified_dom"] = initialization_dom
        result["initialization_snapshot"] = initialization_snapshot
        result["initialization_html_sha256"] = initialization_html_sha256
    elif recover_early_open and dynamic_travel:
        result["initialization"] = "guarded diversified travel-goal recovery"
        result["initialization_history"] = initialization_history
        result["initial_verified_dom"] = initialization_dom
    elif recover_early_open and (recovery_prerequisites or recovery_submitted):
        result["initialization"] = "guarded partial-progress early-open recovery"
        result["initialization_history"] = initialization_history
        result["initial_prerequisite_indices"] = list(recovery_prerequisites)
        result["initial_search_submitted"] = recovery_submitted
        result["initial_verified_dom"] = initialization_dom
    return result


def _counterfactual_comparison_bytes(payload, goal):
    """External pair comparison only; never return this copy to an inference path.

    Replace exactly the native per-question user goal. Preserve URL, every
    offered candidate and its order, state, history, rules and model identity.
    """
    compared = copy.deepcopy(payload)
    for question in compared["questions"].values():
        instructions = question["instructions"]
        if instructions.get("goal") != goal:
            raise ValueError("Counterfactual wire does not retain its entire user goal")
        instructions["goal"] = "<same user goal for external pair comparison>"
    return json.dumps(compared, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def collect(
    *,
    upstream_path,
    output_dir,
    seed=20261001,
    repeats=4,
    split="train",
    chromium_executable=None,
    task_ids=None,
    travel_recovery_mode="empty",
    travel_goal_variants=False,
    travel_counterfactual_goals=False,
    counterfactual_seed=None,
    counterfactual_exclusions=None,
):
    if (
        type(repeats) is not int
        or repeats < 1
        or split not in {"train", "calibration", "development"}
    ):
        raise ValueError("positive repeat count and an explicit fitting/development split required")
    if task_ids is not None and (
        not isinstance(task_ids, (list, tuple))
        or not task_ids
        or len(set(task_ids)) != len(task_ids)
        or set(task_ids) - set(TASK_IDS)
        or (split != "train" and "upstream_travel" in task_ids)
    ):
        raise ValueError("Training task selection must name distinct available development tasks")
    if travel_recovery_mode not in ("empty", "partial") or (
        travel_recovery_mode == "partial" and split != "train"
    ):
        raise ValueError("Partial travel recovery is an explicit train-only collection option")
    if type(travel_goal_variants) is not bool or (
        travel_goal_variants
        and (
            split != "train"
            or repeats > 4
            or travel_recovery_mode != "empty"
            or (task_ids is not None and "upstream_travel" not in task_ids)
        )
    ):
        raise ValueError(
            "Travel goal variants require train, at most four repeats, and their own recovery schedule"
        )
    if (
        type(travel_counterfactual_goals) is not bool
        or (
            travel_counterfactual_goals
            and (
                split != "train"
                or repeats != 1
                or travel_goal_variants
                or travel_recovery_mode != "empty"
                or (task_ids is not None and list(task_ids) != ["upstream_travel"])
                or type(counterfactual_seed) is not int
                or counterfactual_seed < 0
                or counterfactual_seed == seed
                or counterfactual_exclusions is None
            )
        )
        or (
            not travel_counterfactual_goals
            and (counterfactual_seed is not None or counterfactual_exclusions is not None)
        )
    ):
        raise ValueError(
            "Counterfactual-only collection requires train, repeats=1, its own explicit seed "
            "distinct from seed, and an exclusion-hash file; other travel modes cannot be combined"
        )
    effective_seed = counterfactual_seed if travel_counterfactual_goals else seed
    if travel_counterfactual_goals:
        from s1.browser_training_counterfactuals import (
            GOAL_NORMALIZATION,
            make_counterfactual_travel_episodes,
            validate_excluded_goal_hashes,
        )

        exclusion_path = Path(counterfactual_exclusions).resolve()
        exclusion_blob = exclusion_path.read_bytes()
        exclusions = json.loads(exclusion_blob)
        if (
            not isinstance(exclusions, dict)
            or set(exclusions) != {"schema_version", "normalization", "goal_sha256"}
            or type(exclusions["schema_version"]) is not int
            or exclusions["schema_version"] != 1
            or exclusions["normalization"] != GOAL_NORMALIZATION
        ):
            raise ValueError(
                "Counterfactual exclusions need schema 1 and declared goal normalization"
            )
        excluded_hashes = validate_excluded_goal_hashes(exclusions["goal_sha256"])
    manifest = upstream_manifest(upstream_path)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema_version": 1,
        "scope": "development-only scripted training collection; no model evaluation",
        "stage": "collection",
        "seed": effective_seed,
        "split": split,
        "repeats": repeats,
        "upstream": manifest,
        "episodes": [],
        "training_cases": 0,
        "text_source": "scripted development labels only, never evaluation text helper",
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "task_ids": None if task_ids is None else list(task_ids),
        "travel_recovery_mode": travel_recovery_mode,
    }
    if travel_goal_variants:
        receipt["travel_goal_variants"] = True
    if travel_counterfactual_goals:
        receipt.update(
            travel_counterfactual_goals=True,
            ordinary_seed=seed,
            counterfactual_seed=counterfactual_seed,
            effective_seed=effective_seed,
            task_ids=["upstream_travel"],
            counterfactual_exclusions={
                "source_path": str(exclusion_path),
                "artifact_path": "counterfactual_exclusions.json",
                "sha256": hashlib.sha256(exclusion_blob).hexdigest(),
                "normalization": GOAL_NORMALIZATION,
                "goal_count": len(excluded_hashes),
                "role": "exclusion-only hashes, never goals, teacher labels or policy inputs",
                "preservation_timing": "before goal generation and browser collection",
            },
        )
    rows, wires = [], []
    source_manifest = None
    primary_error = None
    verify_sources = None
    source_files = {
        "scripts/jev_browser_training_data.py": Path(__file__),
        "scripts/jev_browser_benchmark.py": Path(make_cases.__code__.co_filename),
    }
    key_before = os.environ.get("TYPESAFE_API_KEY")
    os.environ["TYPESAFE_API_KEY"] = "training-oracle-placeholder"
    try:
        from s1.browser_evidence import capture_runtime_sources, verify_runtime_sources

        verify_sources = verify_runtime_sources
        source_manifest = capture_runtime_sources(output, additional_files=source_files)
        receipt["source_manifest"] = source_manifest
        receipt["source_preservation_timing"] = "before fixture generation and browser collection"
        receipt["collector_sha256"] = source_manifest["files"][
            "scripts/jev_browser_training_data.py"
        ]["sha256"]
        if travel_counterfactual_goals:
            (output / "counterfactual_exclusions.json").write_bytes(exclusion_blob)
        from playwright.sync_api import sync_playwright

        all_cases = []
        if travel_counterfactual_goals:
            counterfactual_specs = make_counterfactual_travel_episodes(
                seed=effective_seed, excluded_goal_hashes=excluded_hashes
            )
            original = make_upstream_travel_case(upstream_path)
            # All goals use exactly the same real browser URL. Internal profile
            # IDs belong only to the external ledgers, never a served route.
            public_fixture = FixtureCase(
                case_id=hashlib.sha256(original.html.encode()).hexdigest()[:24],
                task_id="upstream_travel",
                goal="",
                html=original.html,
                expected={},
                query_string=original.query_string,
            )
            receipt["counterfactual_public_fixture"] = {
                "url_suffix": public_fixture.url_suffix,
                "html_sha256": hashlib.sha256(original.html.encode()).hexdigest(),
                "identity_source": "pinned HTML bytes only; independent of goal, scope and seed",
                "served_pages": 1,
            }
            for spec in counterfactual_specs:
                all_cases.append(
                    FixtureCase(
                        case_id=spec.case_id,
                        task_id="upstream_travel",
                        goal=spec.goal,
                        html=original.html,
                        expected=spec.expected,
                        needs_text=spec.query_required,
                        query_string=original.query_string,
                        source={
                            **original.source,
                            "scope": "training-only counterfactual compositions on exposed HTML",
                            "counterfactual_seed": effective_seed,
                        },
                    )
                )
        else:
            for repeat in range(repeats):
                generated = make_cases(seed, repeat, "development", upstream_path)
                if travel_goal_variants:
                    generated = [case for case in generated if case.task_id != "upstream_travel"]
                    generated.extend(make_training_travel_cases(upstream_path, repeat))
                all_cases.extend(
                    case
                    for case in generated
                    if (split == "train" or case.task_id != "upstream_travel")
                    and (task_ids is None or case.task_id in task_ids)
                )
        fixtures = output / "fixtures"
        fixtures.mkdir()
        for index, case in enumerate(all_cases):
            (fixtures / f"{index:04d}.html").write_text(case.html)
        if travel_goal_variants or travel_counterfactual_goals:
            inventory = {
                "scope": "development-only diversified travel supervision, no model evaluation",
                "cases": [
                    {
                        "case_id": case.case_id,
                        "task_id": case.task_id,
                        "goal": case.goal,
                        "expected": case.expected,
                        "html_sha256": hashlib.sha256(case.html.encode()).hexdigest(),
                    }
                    for case in all_cases
                ],
            }
            inventory_blob = (
                json.dumps(inventory, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
            ).encode()
            if travel_counterfactual_goals:
                inventory.update(
                    scope="training-only counterfactual goals on exposed training HTML; no transfer",
                    seed=effective_seed,
                    episode_count=len(counterfactual_specs),
                    initialization_preservation_timing="before browser collection",
                )
                for entry, spec in zip(inventory["cases"], counterfactual_specs, strict=True):
                    entry.update(
                        pair_id=spec.pair_id,
                        family=spec.family,
                        start=spec.start,
                        query_required=spec.query_required,
                        initialization=[
                            {"label": label, "kind": kind, "text": text}
                            for label, kind, text in spec.initialization
                        ],
                        public_url_suffix=public_fixture.url_suffix,
                    )
                inventory_blob = (
                    json.dumps(inventory, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
                ).encode()
            (output / "training_goal_inventory.json").write_bytes(inventory_blob)
            receipt["training_goal_inventory"] = {
                "artifact_path": "training_goal_inventory.json",
                "sha256": hashlib.sha256(inventory_blob).hexdigest(),
                "preservation_timing": "before browser collection",
            }
        episode_specs = [
            (index, case, recovery)
            for index, case in enumerate(all_cases)
            for recovery in (
                [False, True]
                if case.task_id == "upstream_travel"
                and not travel_counterfactual_goals
                and (
                    not travel_goal_variants
                    or {"query", "category", "free"} <= case.expected.keys()
                )
                else [False]
            )
        ]
        served_cases = [public_fixture] if travel_counterfactual_goals else all_cases
        with fixture_server(served_cases) as base, sync_playwright() as runtime:
            options = {"headless": True}
            if chromium_executable:
                options["executable_path"] = chromium_executable
            browser = runtime.chromium.launch(**options)
            try:
                with load_upstream(upstream_path, PlaywrightCDP(browser)) as (
                    agent_module,
                    model_module,
                ):
                    recovery_index = 0
                    counterfactual_pairs = {}
                    for index, case, recovery in episode_specs:
                        variant = "recovery" if recovery else "fresh"
                        if travel_counterfactual_goals:
                            spec = counterfactual_specs[index]
                            variant = "counterfactual-" + spec.start
                        episode_id = (
                            f"{split}-{effective_seed}-{index:04d}-{case.task_id}-{variant}"
                        )
                        agent = agent_module.Agent(
                            base
                            + (
                                public_fixture.url_suffix
                                if travel_counterfactual_goals
                                else case.url_suffix
                            ),
                            case.goal,
                            screenshots=False,
                        )
                        try:
                            recovery_options = {}
                            if travel_counterfactual_goals:
                                recovery_options = {
                                    "dynamic_travel": True,
                                    "counterfactual_initialization": tuple(
                                        TeachingAction(label, kind, text)
                                        for label, kind, text in spec.initialization
                                    ),
                                }
                            elif travel_goal_variants and case.task_id == "upstream_travel":
                                recovery_options["dynamic_travel"] = True
                                if recovery:
                                    repeat_index = int(case.case_id.split("-")[2])
                                    recovery_options["recovery_initialization"] = (
                                        _training_recovery_actions(case, repeat_index)
                                    )
                            elif recovery and travel_recovery_mode == "partial":
                                # Cover all incomplete combinations. A fully configured,
                                # submitted search is not a negative recovery example.
                                configurations = [
                                    (mask, submitted)
                                    for submitted in (False, True)
                                    for mask in range(8 if not submitted else 7)
                                ]
                                mask, submitted = configurations[recovery_index % 15]
                                recovery_index += 1
                                recovery_options = {
                                    "recovery_prerequisites": tuple(
                                        bit for bit in range(3) if mask & (1 << bit)
                                    ),
                                    "recovery_submitted": submitted,
                                }
                            first_wire = len(wires)
                            episode = collect_episode(
                                agent,
                                agent_module,
                                model_module,
                                case,
                                rng=random.Random(f"{effective_seed}:{index}:{variant}"),
                                split=split,
                                episode_id=episode_id,
                                rows=rows,
                                wires=wires,
                                recover_early_open=recovery,
                                **recovery_options,
                            )
                            episode.update(
                                fixture_path=f"fixtures/{index:04d}.html",
                                fixture_sha256=hashlib.sha256(case.html.encode()).hexdigest(),
                                goal_sha256=hashlib.sha256(case.goal.encode()).hexdigest(),
                            )
                            receipt["episodes"].append(episode)
                            if travel_counterfactual_goals:
                                episode.update(
                                    counterfactual_case_id=spec.case_id,
                                    pair_id=spec.pair_id,
                                    family=spec.family,
                                    start=spec.start,
                                    query_required=spec.query_required,
                                )
                                actual = episode["initial_verified_dom"]["observed"]
                                compared_wire = _counterfactual_comparison_bytes(
                                    wires[first_wire]["payload"], case.goal
                                )
                                episode["first_wire_goal_comparison_sha256"] = hashlib.sha256(
                                    compared_wire
                                ).hexdigest()
                                signature = {
                                    "html_sha256": episode["initialization_html_sha256"],
                                    "history": episode["initialization_history"],
                                    "result": actual["result"],
                                    "typed_destination": actual.get("typed_destination"),
                                    "native_payload_goal_only_replaced": compared_wire,
                                }
                                pair = counterfactual_pairs.setdefault(spec.pair_id, [])
                                if pair and pair[0] != signature:
                                    raise ValueError("Counterfactual pair initialization differs")
                                pair.append(signature)
                        finally:
                            agent.close()
            finally:
                browser.close()
        if travel_counterfactual_goals:
            if len(receipt["episodes"]) != 32 or any(
                len(pair) != 2 for pair in counterfactual_pairs.values()
            ):
                raise ValueError("Counterfactual collection must preserve all 32 paired episodes")
            receipt["counterfactual_pair_verification"] = {
                "success": True,
                "pairs": len(counterfactual_pairs),
                "compared": (
                    "initial HTML, guarded history, actual application state, typed text, "
                    "complete first native payload bytes with only questions.*.instructions.goal "
                    "replaced by a common placeholder; no field or candidate-order exclusions"
                ),
            }
        receipt["stage"] = "complete"
    except BaseException as exc:
        primary_error = exc
        receipt.update(stage="failed", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        if key_before is None:
            os.environ.pop("TYPESAFE_API_KEY", None)
        else:
            os.environ["TYPESAFE_API_KEY"] = key_before
        cases_path, wire_path = output / "cases.jsonl", output / "wire_requests.jsonl"
        cases_path.write_text("".join(row.model_dump_json() + "\n" for row in rows))
        wire_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in wires)
        )
        receipt["training_cases"] = len(rows)
        receipt["dataset_sha256"] = hashlib.sha256(cases_path.read_bytes()).hexdigest()
        receipt["wire_sha256"] = hashlib.sha256(wire_path.read_bytes()).hexdigest()
        integrity_errors = []
        if (
            travel_goal_variants or travel_counterfactual_goals
        ) and "training_goal_inventory" in receipt:
            try:
                frozen_inputs_valid = (
                    output / "training_goal_inventory.json"
                ).read_bytes() == inventory_blob and all(
                    (output / "fixtures" / f"{index:04d}.html").read_bytes() == case.html.encode()
                    for index, case in enumerate(all_cases)
                )
            except OSError:
                frozen_inputs_valid = False
            receipt["training_goal_input_verification"] = {"success": frozen_inputs_valid}
            if not frozen_inputs_valid:
                integrity_errors.append(
                    "Diversified training inventory or fixture bytes changed during collection"
                )
        if travel_counterfactual_goals:
            try:
                exclusion_valid = (
                    exclusion_path.read_bytes() == exclusion_blob
                    and (output / "counterfactual_exclusions.json").read_bytes() == exclusion_blob
                )
            except OSError:
                exclusion_valid = False
            receipt["counterfactual_exclusion_verification"] = {"success": exclusion_valid}
            if not exclusion_valid:
                integrity_errors.append("Counterfactual exclusion input changed during collection")
        if source_manifest is not None:
            try:
                receipt["source_verification"] = verify_sources(
                    source_manifest, additional_files=source_files
                )
            except Exception as exc:
                receipt["source_verification"] = {
                    "success": False,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            if not receipt["source_verification"].get("success"):
                integrity_errors.append(
                    "Collection runtime source changed or could not be verified"
                )
        try:
            current_upstream = upstream_manifest(upstream_path)
            receipt["upstream_verification"] = {"success": current_upstream == manifest}
        except Exception as exc:
            receipt["upstream_verification"] = {
                "success": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        if not receipt["upstream_verification"]["success"]:
            integrity_errors.append("Collection upstream source changed or could not be verified")
        if integrity_errors:
            receipt.update(stage="failed", integrity_errors=integrity_errors)
        try:
            (output / "receipt.json").write_text(
                json.dumps(receipt, indent=2, allow_nan=False) + "\n"
            )
        except Exception as exc:
            if primary_error is None:
                raise
            primary_error.add_note(
                "Failed to write final collection receipt: " + type(exc).__name__
            )
        if integrity_errors and primary_error is None:
            raise RuntimeError("; ".join(integrity_errors))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--split", choices=["train", "calibration", "development"], default="train")
    parser.add_argument("--chromium-executable")
    parser.add_argument("--tasks", dest="task_ids", nargs="+", choices=sorted(TASK_IDS))
    parser.add_argument("--travel-recovery-mode", choices=["empty", "partial"], default="empty")
    parser.add_argument("--travel-goal-variants", action="store_true")
    parser.add_argument("--travel-counterfactual-goals", action="store_true")
    parser.add_argument("--counterfactual-seed", type=int)
    parser.add_argument("--counterfactual-exclusions")
    receipt = collect(**vars(parser.parse_args()))
    print(
        json.dumps(
            {key: receipt[key] for key in ["stage", "scope", "training_cases", "dataset_sha256"]}
        )
    )


if __name__ == "__main__":
    main()
