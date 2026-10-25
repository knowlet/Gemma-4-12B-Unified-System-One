"""Explicit, model-free collection of new owned, train-only native browser wires.

Default execution only describes a prospective collection. ``--collect`` requires
the reviewed plan and collector hashes. The original pinned Agent, choose,
action_space, freshness checks and executor perform every episode. A scripted
teacher replaces the HTTP boundary and field writer, never the browser policy
publisher or an observed menu. Its point masses are execution instructions, not
model probabilities or a browser quality score.

Raw DOM, requests and execution journals belong in the caller's private output
directory. Admission is a separate, explicit evidence replay. Even an admitted
collection is not automatically eligible for fitting or qualification.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import re
import stat
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import jev_browser_benchmark as benchmark
import jev_decision3_affordance_scenarios as fixtures

SCHEMA = "decision3-owned-affordance-training-collection-v1"
ROW_SCHEMA = "decision3-owned-affordance-native-row-v1"
EXCLUSION_SCHEMA = "decision3-affordance-exclusions-v1"
MODEL_ALIAS = "generated-actions-none"
NORMALIZATION = fixtures.GOAL_NORMALIZATION
ROOT = Path(__file__).resolve().parents[1]
MAX_TICKS = 16
SETTLED_V7_PROFILE = "composed-visible-text-settled-v7"
DEFAULT_SCENARIO_PROFILE = "affordance-v1"
CHECKED_CONTROL_PROFILE = "checked-control-progress-v1"
GOAL_ROUTED_CONTROL_PROFILE = "goal-routed-control-v1"
_SCENARIO_REGISTRY = {
    DEFAULT_SCENARIO_PROFILE: (
        "jev_decision3_affordance_scenarios",
        "decision3-owned-affordance-scenarios-v1",
        "decision3-owned-affordance-training-v1",
        (4, 7, 8),
    ),
    CHECKED_CONTROL_PROFILE: (
        "jev_decision3_checked_control_scenarios",
        "decision3-owned-checked-control-scenarios-v1",
        "decision3-owned-checked-control-training-v1",
        (6,),
    ),
    GOAL_ROUTED_CONTROL_PROFILE: (
        "jev_decision3_goal_routed_control_scenarios",
        "decision3-owned-goal-routed-control-scenarios-v1",
        "decision3-owned-goal-routed-control-training-v1",
        (6,),
    ),
}
SCENARIO_PROFILES = tuple(_SCENARIO_REGISTRY)
_ADMISSION_PROOF = object()
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_IDENTITY = re.compile(r"[a-z0-9][a-z0-9._-]{0,95}\Z")


def _scenario_definition(scenario_profile):
    if type(scenario_profile) is not str or scenario_profile not in _SCENARIO_REGISTRY:
        raise ValueError("Unknown explicit scenario profile")
    return _SCENARIO_REGISTRY[scenario_profile]


def _scenario_module(scenario_profile=DEFAULT_SCENARIO_PROFILE):
    name, schema, namespace, counts = _scenario_definition(scenario_profile)
    selected = (
        fixtures if scenario_profile == DEFAULT_SCENARIO_PROFILE else importlib.import_module(name)
    )
    if (
        Path(selected.__file__).resolve() != ROOT / "scripts" / (name + ".py")
        or selected.SCHEMA != schema
        or selected.NAMESPACE != namespace
        or selected.GOAL_NORMALIZATION != NORMALIZATION
        or (
            scenario_profile != DEFAULT_SCENARIO_PROFILE and selected.EXPECTED_MENU_COUNTS != counts
        )
    ):
        raise ValueError("Scenario module origin, namespace, schema or menu profile changed")
    return selected


def _validate_scenario_spec(spec, scenario_profile):
    _, schema, namespace, counts = _scenario_definition(scenario_profile)
    if (
        type(spec) is not dict
        or spec.get("schema_id") != schema
        or spec.get("namespace") != namespace
        or spec.get("split") != "train"
    ):
        raise ValueError("Scenario profile, namespace, schema or train split mismatch")
    intended = spec.get("expected_menu_counts")
    if (
        type(intended) is not list
        or not intended
        or any(type(value) is not int or value not in counts for value in intended)
        or len(set(intended)) != len(intended)
    ):
        raise ValueError("Scenario menu counts differ from the selected profile")


def _plan_scenarios(plan):
    """Legacy missing bindings mean only the original scenario population."""
    profile = plan.get("scenario_profile", DEFAULT_SCENARIO_PROFILE)
    selected = _scenario_module(profile)
    _, schema, namespace, counts = _scenario_definition(profile)
    expected = {
        "scenario_schema_id": schema,
        "scenario_namespace": namespace,
        "expected_menu_counts": list(counts),
    }
    if "scenario_profile" in plan:
        if any(
            type(plan.get(key)) is not type(value) or plan.get(key) != value
            for key, value in expected.items()
        ):
            raise ValueError("Explicit scenario profile binding changed")
    elif any(key in plan for key in expected):
        raise ValueError("Scenario binding requires its explicit scenario profile")
    for episode in plan.get("episodes", []):
        _validate_scenario_spec(episode["scenario"], profile)
    return profile, selected


def _assert_menu_coverage(observed, scenario_profile=DEFAULT_SCENARIO_PROFILE):
    counts = _scenario_definition(scenario_profile)[3]
    if observed != set(counts):
        bins = "/".join("K" + str(count) for count in counts)
        raise ValueError("Complete genuine " + bins + " population is required")


def _encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _digest(value):
    return _sha(_encode(value))


def _plain_json(path, *, limit=32_000_000):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        raise ValueError("Evidence must be a bounded ordinary file: " + str(path))

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key in evidence")
            result[key] = value
        return result

    def constant(_):
        raise ValueError("Nonfinite JSON is forbidden")

    raw = path.read_bytes()
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant), _sha(raw)


def _write_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write(_encode(value))


def _hash_pin(value):
    if type(value) is not str or not _HEX.fullmatch(value):
        raise ValueError("An explicit SHA256 pin is required")
    return value


def _goal_hash(goal):
    import unicodedata

    return _sha(" ".join(unicodedata.normalize("NFKC", goal).split()).encode())


def _prospective_output(value):
    path = Path(value).expanduser().absolute()
    if ".." in path.parts:
        raise ValueError("Use an ordinary explicit output path without parent traversal")
    if path.exists() or path.is_symlink():
        raise ValueError("Collection output must be fresh")
    if not path.parent.is_dir():
        raise ValueError("Create the output parent separately before reviewing a plan")
    for parent in (path.parent, *path.parents):
        if parent.is_symlink():
            raise ValueError("Collection output cannot traverse symlinks")
    return path


def exclusion_manifest(path):
    """Read hashes and identities only; this format never accepts held-out content."""
    data, source_sha = _plain_json(path, limit=2_000_000)
    expected = {"schema_id", "normalization", "goal_sha256", "scenario_ids", "ancestor_ids"}
    if type(data) is not dict or set(data) != expected:
        raise ValueError("Use the separate, hash-only affordance exclusion schema")
    if data["schema_id"] != EXCLUSION_SCHEMA or data["normalization"] != NORMALIZATION:
        raise ValueError("Unsupported exclusion identity or goal normalization")
    hashes = data["goal_sha256"]
    if (
        not isinstance(hashes, list)
        or not hashes
        or any(type(item) is not str for item in hashes)
        or len(set(hashes)) != len(hashes)
    ):
        raise ValueError("A nonempty, unique reviewed exclusion hash inventory is required")
    for item in hashes:
        _hash_pin(item)
    for name in ("scenario_ids", "ancestor_ids"):
        values = data[name]
        if (
            not isinstance(values, list)
            or any(type(item) is not str or not item for item in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError("Excluded identities must be unique nonempty strings")
    return data, source_sha


def _source_inventory(scenario_profile=DEFAULT_SCENARIO_PROFILE):
    """A NEW inventory, without changing any older frozen collector inventory."""
    paths = [
        Path(__file__).resolve(),
        Path(fixtures.__file__).resolve(),
        Path(_scenario_module(scenario_profile).__file__).resolve(),
        Path(benchmark.__file__).resolve(),
        ROOT / "scripts" / "jev_observation_profile_v7.py",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    ]
    paths.extend(sorted((ROOT / "src" / "s1").rglob("*.py")))
    result = {}
    for path in sorted(set(paths)):
        if path.is_symlink() or not path.is_file():
            raise ValueError("Collector source must be an ordinary file")
        raw = path.read_bytes()
        result[path.relative_to(ROOT).as_posix()] = {"sha256": _sha(raw), "bytes": len(raw)}
    return result


def _assert_sources(expected, scenario_profile=DEFAULT_SCENARIO_PROFILE):
    if _source_inventory(scenario_profile) != expected:
        raise ValueError("Collector source inventory changed")


def _executable_pin(path):
    if path is None:
        return None
    given = Path(path).expanduser().absolute()
    resolved = given.resolve(strict=True)
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError("Chromium must be an explicit executable")
    return {
        "path": str(given),
        "resolved_path": str(resolved),
        "sha256": _sha(resolved.read_bytes()),
    }


def _profile_module(name):
    if name == SETTLED_V7_PROFILE:
        import jev_observation_profile_v7 as profile
    else:
        from s1 import jev_observation_profile as profile
    return profile


def _prepare_observation_profile(upstream, name, helper_sha256):
    if name != SETTLED_V7_PROFILE:
        return benchmark.prepare_observation_profile(upstream, name, helper_sha256)
    profile = _profile_module(name)
    if helper_sha256 != _sha(Path(profile.__file__).read_bytes()):
        raise ValueError("An explicit reviewed settled-V7 helper SHA256 is required")
    bundle = profile.build_profile(
        profile.read_canonical_sources(upstream),
        expected_source_sha256=helper_sha256,
        profile=name,
    )
    if (
        benchmark.upstream_manifest(upstream)["source_sha256"]
        != bundle.manifest()["canonical_source_sha256"]
    ):
        raise ValueError("Settled-V7 canonical Git identity differs")
    return bundle


def collection_plan(
    *,
    output_dir,
    upstream,
    collection_id,
    seed,
    exclusion_manifest_path,
    chromium=None,
    observation_profile=benchmark.NATIVE_OBSERVATION_PROFILE,
    observation_helper_sha256=None,
    scenario_profile=DEFAULT_SCENARIO_PROFILE,
):
    """Return an unwritten, deterministic prospective plan; launch nothing."""
    selected = _scenario_module(scenario_profile)
    _, scenario_schema, scenario_namespace, expected_counts = _scenario_definition(scenario_profile)
    if type(collection_id) is not str or not _IDENTITY.fullmatch(collection_id):
        raise ValueError("Use a bounded new collection identity")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("Seed must be an integer in [0, 2**63)")
    output = _prospective_output(output_dir)
    exclusions, exclusions_sha = exclusion_manifest(exclusion_manifest_path)
    upstream_path = Path(upstream).expanduser().resolve(strict=True)
    canonical = benchmark.upstream_manifest(upstream_path)
    bundle = _prepare_observation_profile(
        upstream_path, observation_profile, observation_helper_sha256
    )
    source = _source_inventory(scenario_profile)
    specs = selected.scenarios(seed, set(exclusions["goal_sha256"]))
    if not specs or len({spec["id"] for spec in specs}) != len(specs):
        raise ValueError("Fresh authored scenario identities must be unique")
    planned = []
    for spec in specs:
        _validate_scenario_spec(spec, scenario_profile)
        if spec["split"] != "train" or spec["id"] in exclusions["scenario_ids"]:
            raise ValueError("A held-out or nontraining scenario cannot enter this collection")
        if spec["ancestor_id"] in exclusions["ancestor_ids"]:
            raise ValueError("A held-out ancestor cannot enter this collection")
        goal_sha = _goal_hash(spec["goal"])
        if goal_sha in exclusions["goal_sha256"]:
            raise ValueError("A held-out goal cannot enter this collection")
        planned.append(
            {
                "episode_id": collection_id + "." + spec["id"],
                "scenario": copy.deepcopy(spec),
                "scenario_sha256": _digest(spec),
                "goal_sha256": goal_sha,
                "fixture_sha256": _sha(selected.render_fixture(spec).encode()),
                "intended_menu_counts": list(spec["expected_menu_counts"]),
            }
        )
    return {
        "schema_id": SCHEMA,
        "schema_version": 1,
        "scenario_profile": scenario_profile,
        "scenario_schema_id": scenario_schema,
        "scenario_namespace": scenario_namespace,
        "expected_menu_counts": list(expected_counts),
        "purpose": "new owned train-only operation supervision; no model evaluation",
        "collection_id": collection_id,
        "seed": seed,
        "split": "train",
        "output_dir": str(output),
        "upstream_path": str(upstream_path),
        "upstream": canonical,
        "collector_sha256": source[Path(__file__).resolve().relative_to(ROOT).as_posix()]["sha256"],
        "sources": source,
        "exclusion_manifest_path": str(Path(exclusion_manifest_path).resolve()),
        "exclusion_manifest_sha256": exclusions_sha,
        "exclusions": exclusions,
        "observation_profile": observation_profile,
        "observation_helper_sha256": observation_helper_sha256,
        "observation_profile_manifest": None if bundle is None else bundle.manifest(),
        "chromium": _executable_pin(chromium),
        "model_alias": MODEL_ALIAS,
        "max_ticks": MAX_TICKS,
        "episodes": planned,
        "annotation": "scripted teacher point masses; never model confidence",
        "loss_policy": "operation-only; target/text proofs are execution evidence",
        "model_score": None,
        "fit_eligible": False,
        "admission_status": "not_collected",
    }


def build_native_training_row(
    payload,
    *,
    row_id,
    scenario_id,
    ancestor_id,
    episode_id,
    step,
    operation,
    target,
    proof_refs,
    valid_next_targets=None,
):
    """Seal the FULL original wire and keep all supervision outside it."""
    from s1.browser_public_training import NativeWireRequest
    from s1.jev_decision3_native import parse_native_payload
    from s1.jev_generated import TARGETS

    if any(
        type(value) is not str or not value or len(value) > 256
        for value in (row_id, scenario_id, ancestor_id, episode_id)
    ):
        raise ValueError("New row, scenario, ancestor and episode identities are required")
    if type(step) is not int or not 0 <= step < MAX_TICKS:
        raise ValueError("A row must belong to an original bounded episode step")
    request = NativeWireRequest.model_validate(payload)
    _, original, source, labels = parse_native_payload(payload, model_identity=MODEL_ALIAS)
    if payload.get("model") != MODEL_ALIAS or source != request.wire_bytes:
        raise ValueError("The collected native alias and full ordered wire must remain unchanged")
    if operation == "BLOCKED" or operation not in labels["operation"]:
        raise ValueError("Positive BLOCKED or an unoffered operation is not admitted")
    head = TARGETS.get(operation)
    if head:
        if type(target) is not str or target not in labels[head]:
            raise ValueError("A targeted operation requires its actual offered target")
    elif target is not None:
        raise ValueError("A targetless operation requires target=null")
    accepted = (
        {operation: [] if head is None else [target]}
        if valid_next_targets is None
        else copy.deepcopy(valid_next_targets)
    )
    if type(accepted) is not dict or operation not in accepted:
        raise ValueError("The actual teacher operation must be accepted")
    for op, targets in accepted.items():
        if op == "BLOCKED" or op not in labels["operation"] or type(targets) is not list:
            raise ValueError("Only offered, independently justified actions may be accepted")
        active = TARGETS.get(op)
        if active:
            if (
                not targets
                or any(type(t) is not str for t in targets)
                or len(set(targets)) != len(targets)
                or any(t not in labels[active] for t in targets)
            ):
                raise ValueError("Accepted targets must be unique actual offered keys")
        elif targets:
            raise ValueError("A targetless accepted operation has an empty target list")
    if head and target not in accepted[operation]:
        raise ValueError("The actual teacher target must be accepted")
    if "DONE" in accepted and set(accepted) != {"DONE"}:
        raise ValueError("Completion must not simultaneously supervise further mutation")
    if not proof_refs or type(proof_refs) is not dict:
        raise ValueError("External proof references are required, and are not authentication")
    if any(
        type(name) is not str or type(value) is not str or not value
        for name, value in proof_refs.items()
    ):
        raise ValueError("Proof references must be nonempty external string identities")
    gold = {"operation": operation}
    if head:
        gold[head] = target
    return {
        "schema_id": ROW_SCHEMA,
        "schema_version": 1,
        "id": row_id,
        "split": "train",
        "scenario_id": scenario_id,
        "ancestor_id": ancestor_id,
        "episode_id": episode_id,
        "step": step,
        "payload": original,
        "wire_sha256": request.wire_sha256,
        "candidate_labels": request.candidate_labels,
        "teacher_choice": {"operation": operation, "target": target},
        "gold": gold,
        "valid_next_targets": accepted,
        "loss_mask": {name: name == "operation" for name in labels},
        "proof_refs": copy.deepcopy(proof_refs),
        "fit_eligible": False,
        "admission_status": "pending_independent_admission",
    }


def _action_space(actions):
    """Independent replay of the pinned publisher's ordered observed-node mapping."""
    elements, indices, targets, controls = [], {}, {}, {}
    operations = {"click": "CLICK", "fill": "TYPE_TEXT", "select": "SELECT"}
    for action in actions:
        kind = action["kind"]
        if kind not in operations:
            controls[action["id"].upper()] = action
            continue
        node = action["node"]
        if node not in indices:
            index = str(len(elements) + 1)
            indices[node] = index
            element = {
                k: action[k]
                for k in ("role", "value", "checked", "selected", "expanded")
                if k in action
            }
            element.update(index=index, label=action["label"].split(" → ")[0], operations=[])
            if kind == "select":
                element["value"] = action.get("current_value", "")
                element["options"] = []
            elements.append(element)
        index = indices[node]
        operation = operations[kind]
        group = targets.setdefault(operation, {})
        element = elements[int(index) - 1]
        if operation not in element["operations"]:
            element["operations"].append(operation)
        target = index
        if kind == "select":
            target = f"{index}:{len(element['options']) + 1}"
            element["options"].append(
                {"index": target, "label": action["label"], "value": action["value"]}
            )
        group[target] = action
    return elements, targets, controls


def _publisher_rules(upstream):
    # The pinned questions file contains string constants only. Read literals;
    # never execute an arbitrary source file to recover instructions.
    source = (Path(upstream) / "jev_ultrafast" / "questions.py").read_text()
    rules = {}
    for name in ("NEXT_ACTION", "TARGET"):
        marker = name + ' = """'
        if source.count(marker) != 1:
            raise ValueError("Pinned publisher instruction literal changed")
        rules[name] = source.split(marker, 1)[1].split('"""', 1)[0]
    return rules


def _publisher_wire(page, goal, history, rules):
    elements, targets, controls = _action_space(page["actions"])
    descriptions = {
        "CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
        "TYPE_TEXT": "Enter or replace text in an editable field. A small LLM will supply the value from the goal.",
        "SELECT": "Select an observed dropdown value.",
    }
    operations = {op: descriptions[op] for op in targets}
    operations.update({key: value["label"] for key, value in controls.items()})
    operations.update(
        DONE="Every requirement is visibly satisfied.",
        BLOCKED="No supported operation can progress.",
    )
    questions = {
        "operation": {
            "type": "choice",
            "criteria": operations,
            "instructions": {"goal": goal, "rules": rules["NEXT_ACTION"]},
        }
    }
    for operation, candidates in targets.items():
        questions[operation.lower() + "_target"] = {
            "type": "choice",
            "criteria": {
                index: {
                    "element": f"[{index}] {a['label']}",
                    "current_value": a.get("current_value", a.get("value", "")),
                    **{k: a[k] for k in ("role", "checked", "selected", "expanded") if k in a},
                }
                for index, a in candidates.items()
            },
            "instructions": {
                "goal": goal,
                "operation": operation,
                "rules": [rules["NEXT_ACTION"], rules["TARGET"]],
            },
        }
    return {
        "model": MODEL_ALIAS,
        "state": {
            "page": {key: page[key] for key in ("url", "title", "text")},
            "elements": elements,
            "recent_actions": [
                {key: h.get(key) for key in ("action", "kind", "text", "page_changed")}
                for h in history[-10:]
            ],
        },
        "questions": questions,
    }


def _selected_action(page, descriptor):
    from s1.jev_generated import TARGETS

    _, targets, controls = _action_space(page["actions"])
    operation = descriptor["operation"]
    if TARGETS.get(operation):
        candidates = [
            (target, action)
            for target, action in targets.get(operation, {}).items()
            if action["kind"] == descriptor["kind"]
            and action["label"] == descriptor["label"]
            and (descriptor["action_id"] is None or action["id"] == descriptor["action_id"])
        ]
        if len(candidates) != 1:
            raise ValueError("Teacher must resolve one actual observed native action")
        return candidates[0]
    if operation in {"DONE", "BLOCKED"}:
        return None, None
    action = controls.get(operation)
    if action is None or action["kind"] != descriptor["kind"]:
        raise ValueError("Teacher control was not genuinely offered")
    if descriptor["action_id"] is not None and action["id"] != descriptor["action_id"]:
        raise ValueError("Teacher control identity differs from the native action")
    return None, action


def _teacher_response(payload, operation, target):
    from s1.jev_generated import TARGETS

    selected = {"operation": operation}
    if TARGETS.get(operation):
        selected[TARGETS[operation]] = target
    answers = {}
    for name, choice in selected.items():
        offered = payload["questions"][name]["criteria"]
        if choice not in offered:
            raise ValueError("The teacher choice is absent from the original wire")
        answers[name] = {
            "choice": choice,
            "confidence": 1.0,
            "probabilities": {label: float(label == choice) for label in offered},
        }
    return {
        "model": "scripted-owned-native-affordance-teacher",
        "answers": answers,
        "usage": {},
        "annotation": "scripted execution instruction; not a model probability",
    }


class _NoHTTP:
    def post(self, *_args, **_kwargs):
        raise ValueError("Collection forbids model/provider HTTP")


@contextmanager
def _publisher_environment():
    keys = {
        "TYPESAFE_MODEL": MODEL_ALIAS,
        "TYPESAFE_API_KEY": "owned-scripted-collection-no-provider",
    }
    saved = {key: os.environ.get(key) for key in keys}
    os.environ.update(keys)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class RecordingCDP(benchmark.PlaywrightCDP):
    """Record real protocol responses, including original READ_STATE evaluations."""

    def __init__(self, browser):
        super().__init__(browser)
        self.journal = []

    def cdp(self, method, session_id=None, **params):
        row = {
            "sequence": len(self.journal),
            "method": method,
            "session": session_id,
            "params": copy.deepcopy(params),
        }
        self.journal.append(row)
        try:
            result = super().cdp(method, session_id=session_id, **params)
            row["response"] = copy.deepcopy(result)
            return result
        except BaseException as exc:
            row["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise


def _runtime_value(journal, sequence, expression):
    if type(sequence) is not int or not 0 <= sequence < len(journal):
        raise ValueError("Missing raw protocol proof")
    call = journal[sequence]
    if call.get("sequence") != sequence or call.get("method") != "Runtime.evaluate":
        raise ValueError("Proof does not refer to a raw Runtime.evaluate")
    if call.get("params", {}).get("expression") != expression or "error" in call:
        raise ValueError("Proof expression changed or evaluation failed")
    response = call.get("response", {})
    if response.get("exceptionDetails") or "value" not in response.get("result", {}):
        raise ValueError("The original evaluation did not return a by-value proof")
    return response["result"]["value"]


def _latest_observation(journal, expression, page):
    for call in reversed(journal):
        if call["method"] == "Runtime.evaluate" and call["params"].get("expression") == expression:
            raw = _runtime_value(journal, call["sequence"], expression)
            if type(raw) is not dict or any(
                raw.get(key) != value for key, value in page.items() if key != "fingerprint"
            ):
                raise ValueError("Agent page differs from the latest actual native observation")
            content = {key: raw[key] for key in ("url", "text", "actions", "scroll")}
            fingerprint = _sha(json.dumps(content, sort_keys=True).encode())
            if page.get("fingerprint") != fingerprint:
                raise ValueError("Native observation fingerprint changed")
            return call["sequence"]
    raise ValueError("No original native READ_STATE response exists")


def _native_action_expression(upstream, action):
    source = (Path(upstream) / "jev_ultrafast" / "browser.py").read_text()
    start = 'target = evaluate("""'
    end = '""" + json.dumps(action) + ")")'
    if source.count(start) != 1 or end not in source:
        raise ValueError("Pinned native executor expression changed")
    prefix = source.split(start, 1)[1].split(end, 1)[0]
    return prefix + json.dumps(action) + ")"


def _verify_freshness(entry, journal, read_state):
    """Recompute original fresh() from actual responses, never its saved boolean."""
    start, end = entry["cdp_start"], entry["cdp_end"]
    if end != start + 1:
        raise ValueError("Original freshness requires exactly one protocol evaluation")
    action, page = entry["action"], entry["page"]
    if action is not None and action["kind"] in {"click", "select"}:
        node = action["node"]
        if type(node) is not int:
            raise ValueError("Native freshness requires an actual observed node")
        expression = (
            "(() => { const c=window.__jevFast; "
            f"return c ? [c.pageKey(),c.guard(c.nodes.get({node}))] : null; }})()"
        )
        expected = [page["page_key"], page["guards"].get(str(node))]
    else:
        expression = f"(() => {{ const state={read_state}; return state?.marker ?? null; }})()"
        expected = page["marker"]
    if (
        _runtime_value(journal, start, expression) != expected
        or entry.get("result") is not True
        or "error" in entry
    ):
        raise ValueError("Actual canonical freshness response does not match the observed page")


def _field_context(goal, action, page, history):
    return {
        "goal": goal,
        "field": {key: action.get(key) for key in ("label", "role", "value")},
        "page": {"title": page["title"], "text": page["text"][:6000]},
        "recent_actions": [
            {key: item.get(key) for key in ("action", "text")} for item in history[-6:]
        ],
    }


def _verify_executor(
    step, journal, upstream, read_state, *, scenario_profile=DEFAULT_SCENARIO_PROFILE
):
    selected = _scenario_module(scenario_profile)
    descriptor, action = step["teacher"], step["action"]
    operation = descriptor["operation"]
    calls = journal[step["tick_start"] : step["tick_end"]]
    if any(
        call.get("sequence") != index for index, call in enumerate(calls, start=step["tick_start"])
    ):
        raise ValueError("Native tick protocol sequence changed")
    if any("error" in call for call in calls):
        raise ValueError("A native tick with a failed protocol call is quarantined")
    fresh = step["fresh_calls"]
    if not fresh:
        raise ValueError("Every original freshness guard must have succeeded")
    for entry in fresh:
        if not step["tick_start"] <= entry["cdp_start"] < entry["cdp_end"] <= step["tick_end"]:
            raise ValueError("Freshness proof lies outside its original tick")
        _verify_freshness(entry, journal, read_state)
    inputs = [call for call in calls if call["method"].startswith("Input.")]
    decision = step["decision"]
    if (
        _encode(decision["request"]) != _encode(step["payload"])
        or decision["operation"] != operation
        or decision["target"] != step["target"]
        or decision["raw_answers"]
        != _teacher_response(step["payload"], operation, step["target"])["answers"]
    ):
        raise ValueError("Original choose returned a different full request or choice")
    if operation == "DONE":
        if (
            action is not None
            or step["acts"]
            or inputs
            or step["status"] != "done"
            or step["field_calls"]
        ):
            raise ValueError("DONE requires original terminal handling without browser input")
        if decision["choice"] != "DONE" or not selected.verify_goal(
            step["scenario"], step["after"]
        ):
            raise ValueError("DONE lacks independently checked goal completion")
        terminal_guards = [
            item
            for item in fresh
            if item.get("action") is None
            and item.get("page") == step["page"]
            and step["private_before_call"] < item["cdp_start"]
            and item["cdp_end"] == step["private_after_call"]
        ]
        if len(terminal_guards) != 1:
            raise ValueError("DONE requires its original post-decision terminal freshness guard")
        return {"native_terminal": True, "input_calls": 0}
    if operation == "BLOCKED" or action is None or len(step["acts"]) != 1:
        raise ValueError("One actual guarded original action is required")
    execution = step["acts"][0]
    if (
        execution.get("action") != action
        or execution.get("page") != step["page"]
        or execution.get("result") != {"executed": action["id"]}
        or decision["choice"] != action["id"]
        or "error" in execution
    ):
        raise ValueError("Actual original Browser.act did not execute the selected observed ID")
    if not (
        step["tick_start"]
        <= step["private_before_call"]
        < execution["cdp_start"]
        < execution["cdp_end"]
        <= step["observation_after_call"]
        < step["private_after_call"]
        < step["tick_end"]
    ):
        raise ValueError("Execution must follow its precondition and precede post-observation")
    matching_guards = [
        item
        for item in fresh
        if item.get("action") == action
        and item.get("page") == step["page"]
        and item["cdp_start"] == execution["cdp_start"]
        and item["cdp_end"] == execution["cdp_start"] + 1
    ]
    if len(matching_guards) != 1:
        raise ValueError("The actual action lacks its original immediate freshness guard")
    relevant = journal[execution["cdp_start"] : execution["cdp_end"]]
    if inputs != [call for call in relevant if call["method"].startswith("Input.")]:
        raise ValueError("Execution input must lie inside the actual Browser.act span")
    canonical_methods = {
        "wait": ["Runtime.evaluate"],
        "scroll": ["Runtime.evaluate", "Input.dispatchMouseEvent"],
        "select": ["Runtime.evaluate", "Runtime.evaluate"],
        "click": [
            "Runtime.evaluate",
            "Runtime.evaluate",
            "Input.dispatchMouseEvent",
            "Input.dispatchMouseEvent",
        ],
        "fill": [
            "Runtime.evaluate",
            "Runtime.evaluate",
            "Input.dispatchMouseEvent",
            "Input.dispatchMouseEvent",
            "Input.dispatchKeyEvent",
            "Input.dispatchKeyEvent",
            "Input.insertText",
        ],
    }
    if [call["method"] for call in relevant] != canonical_methods.get(action["kind"]):
        raise ValueError("Execution differs from the canonical guard/executor/input sequence")
    target_point = None
    if action["kind"] in {"click", "fill", "select"}:
        expression = _native_action_expression(upstream, action)
        evaluations = [
            call
            for call in relevant
            if call["method"] == "Runtime.evaluate"
            and call["params"].get("expression") == expression
        ]
        if len(evaluations) != 1 or evaluations[0]["sequence"] != execution["cdp_start"] + 1:
            raise ValueError("Action must join the exact canonical native DOM hit/executor")
        target_point = _runtime_value(journal, evaluations[0]["sequence"], expression)
        if (
            type(target_point) is not dict
            or set(target_point) != {"x", "y"}
            or any(type(value) not in (int, float) for value in target_point.values())
        ):
            raise ValueError("The original hit test must return its actual target coordinates")
    if action["kind"] != "fill" and step["field_calls"]:
        raise ValueError("Only the actual native fill action can invoke the field writer")
    if action["kind"] == "select":
        if inputs:
            raise ValueError("Pinned SELECT does not generate trusted CDP input")
        return {
            "native_select_executor_join": True,
            "select_input_change_events_trusted": False,
            "input_calls": 0,
        }
    if action["kind"] == "scroll":
        expected = {
            "type": "mouseWheel",
            "x": 550,
            "y": 650,
            "deltaX": 0,
            "deltaY": action["delta"],
        }
        if (
            len(inputs) != 1
            or inputs[0]["method"] != "Input.dispatchMouseEvent"
            or inputs[0]["params"] != expected
        ):
            raise ValueError("Scroll proof must retain the original native wheel delta")
    elif action["kind"] in {"click", "fill"}:
        mouse = [call for call in inputs if call["method"] == "Input.dispatchMouseEvent"]
        expected_mouse = [
            {"type": event, **target_point, "button": "left", "clickCount": 1}
            for event in ("mousePressed", "mouseReleased")
        ]
        if [call["params"] for call in mouse] != expected_mouse:
            raise ValueError("A trusted click must join the original CDP press and release")
        inserts = [call for call in inputs if call["method"] == "Input.insertText"]
        if action["kind"] == "fill":
            if (
                len(inserts) != 1
                or inserts[0]["params"] != {"text": descriptor["text"]}
                or execution["text"] != descriptor["text"]
            ):
                raise ValueError("Field proof must join the literal actually typed by native input")
            expected_field = {
                "context": _field_context(
                    step["scenario"]["goal"], action, step["page"], step["history_before"]
                ),
                "text": descriptor["text"],
            }
            if step["field_calls"] != [expected_field]:
                raise ValueError("Actual field literal/context evidence is incomplete")
            keys = [call["params"] for call in inputs if call["method"] == "Input.dispatchKeyEvent"]
            if len(keys) != 2 or keys[0].get("modifiers") not in {2, 4}:
                raise ValueError("Native fill must perform its original select-all keys")
            modifiers = keys[0]["modifiers"]
            if (
                keys
                != [
                    {
                        "type": "keyDown",
                        "key": "a",
                        "code": "KeyA",
                        "modifiers": modifiers,
                        "commands": ["selectAll"],
                    },
                    {"type": "keyUp", "key": "a", "code": "KeyA", "modifiers": modifiers},
                ]
                or len(inputs) != 5
            ):
                raise ValueError("Unexpected input outside the native fill executor")
        elif inserts:
            raise ValueError("CLICK cannot synthesize field input")
        elif len(inputs) != 2:
            raise ValueError("Unexpected input outside the native click executor")
    elif action["kind"] == "wait":
        if inputs:
            raise ValueError("WAIT cannot issue input")
    else:
        raise ValueError("Unsupported native executor kind")
    return {"native_guarded_action": True, "input_calls": len(inputs)}


def _verify_step(step, journal, upstream, read_state, *, scenario_profile=DEFAULT_SCENARIO_PROFILE):
    """Replay raw observation, native publisher, executor and independent DOM joins."""
    spec = step["scenario"]
    selected = _scenario_module(scenario_profile)
    _validate_scenario_spec(spec, scenario_profile)
    if spec["split"] != "train":
        raise ValueError("Only new train-only scenarios are allowed")
    if (
        _runtime_value(journal, step["private_before_call"], selected.private_state_expression())
        != step["before"]
    ):
        raise ValueError("Private precondition lacks its actual raw evaluation")
    if (
        _runtime_value(journal, step["private_after_call"], selected.private_state_expression())
        != step["after"]
    ):
        raise ValueError("Private postcondition lacks its actual raw evaluation")
    raw_page = _runtime_value(journal, step["observation_call"], read_state)
    if any(
        raw_page.get(key) != value for key, value in step["page"].items() if key != "fingerprint"
    ):
        raise ValueError("Original wire page lacks its original READ_STATE join")
    if (
        _latest_observation(journal[: step["private_before_call"]], read_state, step["page"])
        != step["observation_call"]
    ):
        raise ValueError("Wire observation is not the latest original observation")
    session = journal[step["observation_call"]]["session"]
    if not session or any(
        call["session"] != session for call in journal[step["tick_start"] : step["tick_end"]]
    ):
        raise ValueError("Native tick/private evidence crossed the owned browser session")
    if (
        not step["observation_call"]
        < step["private_before_call"]
        < step["private_after_call"]
        < step["tick_end"]
    ):
        raise ValueError("Observation and independent pre/postconditions have invalid ordering")
    if _encode(
        _publisher_wire(
            step["page"], spec["goal"], step["history_before"], _publisher_rules(upstream)
        )
    ) != _encode(step["payload"]):
        raise ValueError(
            "The full native publisher wire changed, including ordering or instructions"
        )
    descriptor = selected.teacher_step(spec, step["before"])
    if descriptor != step["teacher"]:
        raise ValueError("Teacher annotation differs from the authored independent route")
    target, action = _selected_action(step["page"], descriptor)
    if target != step["target"] or action != step["action"]:
        raise ValueError("Teacher did not choose the actual offered native target/control")
    before_events, after_events = step["before"]["events"], step["after"]["events"]
    if after_events[: len(before_events)] != before_events:
        raise ValueError("Owned event history was replaced")
    effect = selected.verify_transition(
        spec, step["before"], step["after"], descriptor, after_events[len(before_events) :]
    )
    execution = _verify_executor(
        step, journal, upstream, read_state, scenario_profile=scenario_profile
    )
    if effect.get("requires_native_select_executor_join") and not execution.get(
        "native_select_executor_join"
    ):
        raise ValueError("Synthetic SELECT events require their canonical executor proof")
    before_history, after_history = step["history_before"], step["history_after"]
    if descriptor["operation"] == "DONE":
        if after_history != before_history:
            raise ValueError("Native DONE must not append or mutate action history")
    else:
        if len(after_history) != len(before_history) + 1 or after_history[:-1] != before_history:
            raise ValueError("Original Agent action history was not appended exactly once")
        if (
            _latest_observation(
                journal[: step["private_after_call"]], read_state, step["page_after"]
            )
            != step["observation_after_call"]
        ):
            raise ValueError("Post-action page lacks its actual original native observation")
        expected_history = {
            "step": len(after_history),
            "action": action["label"],
            "kind": action["kind"],
            "choice": action["id"],
            "text": descriptor["text"],
            "operation": descriptor["operation"],
            "target": target,
            "page_changed": step["page_after"]["fingerprint"] != step["page"]["fingerprint"],
        }
        if any(after_history[-1].get(key) != value for key, value in expected_history.items()):
            raise ValueError(
                "Original Agent history differs from the actual selected action/effect"
            )
    return {"effect": effect, "execution": execution}


def collect_episode(
    *,
    spec,
    episode_id,
    origin,
    transport,
    agent_module,
    model_module,
    upstream,
    read_state,
    scenario_profile=DEFAULT_SCENARIO_PROFILE,
):
    """Run original Agent ticks; retain failed attempts and never retry a stale action."""
    selected = _scenario_module(scenario_profile)
    _validate_scenario_spec(spec, scenario_profile)
    trace = {
        "schema_id": SCHEMA,
        "scenario_profile": scenario_profile,
        "episode_id": episode_id,
        "scenario": copy.deepcopy(spec),
        "steps": [],
        "rows": [],
        "completed": False,
        "status": "pending",
        "errors": [],
        "cdp_start": len(transport.journal),
    }
    agent = None
    saved_post, saved_model_field = model_module.post_json, model_module.field_text
    saved_agent_field, saved_client = agent_module.field_text, model_module.CLIENT
    pending = {}
    raw_fresh, raw_act = None, None
    settling_start = len(getattr(agent_module.Browser, "observation_settling_records", ()))
    try:
        model_module.CLIENT = _NoHTTP()
        agent = agent_module.Agent(origin + "/tasks/" + spec["id"], spec["goal"], screenshots=False)
        raw_fresh, raw_act = agent.browser.fresh, agent.browser.act

        def private():
            start = len(transport.journal)
            value = agent.browser.evaluate(selected.private_state_expression())
            if len(transport.journal) != start + 1:
                raise ValueError("A private proof must have one actual protocol evaluation")
            return value, start

        def fresh(page, action=None):
            entry = {
                "page": copy.deepcopy(page),
                "action": copy.deepcopy(action),
                "cdp_start": len(transport.journal),
            }
            pending.setdefault("fresh_calls", []).append(entry)
            try:
                result = raw_fresh(page, action)
                entry["result"] = result
                return result
            except BaseException as exc:
                entry["error"] = {"type": type(exc).__name__, "message": str(exc)}
                raise
            finally:
                entry["cdp_end"] = len(transport.journal)

        def act(action, page, text=None):
            entry = {
                "action": copy.deepcopy(action),
                "page": copy.deepcopy(page),
                "text": text,
                "cdp_start": len(transport.journal),
            }
            pending.setdefault("acts", []).append(entry)
            try:
                result = raw_act(action, page, text=text)
                entry["result"] = copy.deepcopy(result)
                return result
            except BaseException as exc:
                entry["error"] = {"type": type(exc).__name__, "message": str(exc)}
                raise
            finally:
                entry["cdp_end"] = len(transport.journal)

        def teacher_post(url, _key, payload):
            if url != "https://api.typesafe.ai/v1/systemone" or pending.get("policy_calls", 0):
                raise ValueError("Exactly one original policy publication is allowed per tick")
            pending["policy_calls"] = 1
            page, history = (
                copy.deepcopy(agent.state["page"]),
                copy.deepcopy(agent.state["history"]),
            )
            if model_module.action_space(page["actions"]) != _action_space(page["actions"]):
                raise ValueError("Pinned publisher action_space differs from independent replay")
            expected = _publisher_wire(page, spec["goal"], history, _publisher_rules(upstream))
            if _encode(payload) != _encode(expected):
                raise ValueError("The original publisher request was altered")
            before, before_call = private()
            descriptor = selected.teacher_step(spec, before)
            target, action = _selected_action(page, descriptor)
            pending.update(
                {
                    "payload": copy.deepcopy(payload),
                    "page": page,
                    "history_before": history,
                    "before": before,
                    "private_before_call": before_call,
                    "observation_call": _latest_observation(transport.journal, read_state, page),
                    "teacher": descriptor,
                    "target": target,
                    "action": copy.deepcopy(action),
                }
            )
            # Validate all heads, including unused and singleton heads, before act.
            build_native_training_row(
                payload,
                row_id="pending",
                scenario_id=spec["id"],
                ancestor_id=spec["ancestor_id"],
                episode_id=episode_id,
                step=len(trace["steps"]),
                operation=descriptor["operation"],
                target=target,
                proof_refs={"transition": "pending"},
            )
            return _teacher_response(payload, descriptor["operation"], target)

        def field_text(context):
            action, descriptor = pending.get("action"), pending.get("teacher", {})
            if not action or action["kind"] != "fill" or descriptor.get("operation") != "TYPE_TEXT":
                raise ValueError("A field writer requires the original selected fill action")
            expected = model_module.field_context(
                spec["goal"], action, agent.state["page"], agent.state["history"]
            )
            if context != expected or type(descriptor.get("text")) is not str:
                raise ValueError("Field literal must join the original native field context")
            pending.setdefault("field_calls", []).append(
                {"context": copy.deepcopy(context), "text": descriptor["text"]}
            )
            return descriptor["text"], {
                "model": "scripted-owned-field-literal",
                "latency_ms": 0,
                "usage": {},
            }

        agent.browser.fresh, agent.browser.act = fresh, act
        model_module.post_json = teacher_post
        model_module.field_text = agent_module.field_text = field_text
        for index in range(MAX_TICKS):
            pending.clear()
            pending.update(
                {
                    "tick_start": len(transport.journal),
                    "fresh_calls": [],
                    "acts": [],
                    "field_calls": [],
                }
            )
            try:
                agent.command("tick")
                if pending.get("policy_calls") != 1:
                    raise ValueError("Original Agent did not publish exactly one policy request")
                after, after_call = private()
                step = {
                    **copy.deepcopy(pending),
                    "index": index,
                    "scenario": copy.deepcopy(spec),
                    "after": after,
                    "private_after_call": after_call,
                    "tick_end": len(transport.journal),
                    "decision": copy.deepcopy(agent.state["decisions"][-1]),
                    "history_after": copy.deepcopy(agent.state["history"]),
                    "page_after": copy.deepcopy(agent.state["page"]),
                    "observation_after_call": _latest_observation(
                        transport.journal, read_state, agent.state["page"]
                    ),
                    "status": agent.state["status"],
                }
                step["proof"] = _verify_step(
                    step,
                    transport.journal,
                    upstream,
                    read_state,
                    scenario_profile=scenario_profile,
                )
                trace["steps"].append(step)
                trace["rows"].append(
                    build_native_training_row(
                        step["payload"],
                        row_id=episode_id + f".step-{index:02d}",
                        scenario_id=spec["id"],
                        ancestor_id=spec["ancestor_id"],
                        episode_id=episode_id,
                        step=index,
                        operation=step["teacher"]["operation"],
                        target=step["target"],
                        proof_refs={"transition": episode_id + f"#step-{index}"},
                    )
                )
                if agent.state["status"] == "done":
                    trace.update(completed=True, status="completed")
                    break
                if agent.state["status"] == "blocked":
                    raise ValueError(
                        "Original Agent blocked; do not convert failure to a teacher label"
                    )
            except BaseException:
                trace["failed_attempt"] = copy.deepcopy(pending)
                raise
        if not trace["completed"]:
            raise ValueError("Original 16-tick collection budget exhausted")
    except BaseException as exc:
        trace["status"] = "failed"
        trace["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    finally:
        model_module.post_json, model_module.field_text = saved_post, saved_model_field
        agent_module.field_text, model_module.CLIENT = saved_agent_field, saved_client
        trace["agent_closed"] = agent is None
        if agent is not None:
            if raw_fresh is not None:
                agent.browser.fresh, agent.browser.act = raw_fresh, raw_act
            try:
                agent.close()
                trace["agent_closed"] = agent.browser.target is None
            except BaseException as exc:
                trace["errors"].append(
                    {"type": type(exc).__name__, "message": str(exc), "phase": "agent_close"}
                )
                trace.update(completed=False, status="failed")
        trace["observed_menu_counts"] = sorted(
            {len(row["payload"]["questions"]["operation"]["criteria"]) for row in trace["rows"]}
        )
        if trace["completed"] and trace["observed_menu_counts"] != sorted(
            spec["expected_menu_counts"]
        ):
            trace["errors"].append(
                {
                    "type": "ValueError",
                    "message": "Planned menu bins were not naturally observed; no options were removed",
                }
            )
            trace.update(completed=False, status="failed")
        trace["cdp_end"] = len(transport.journal)
        if hasattr(agent_module.Browser, "observation_settling_records"):
            trace["observation_settling"] = {
                "records": copy.deepcopy(
                    agent_module.Browser.observation_settling_records[settling_start:]
                ),
                "fault": bool(getattr(agent_module.Browser, "_observation_settling_fault", False)),
            }
            if trace["observation_settling"]["fault"]:
                trace["errors"].append(
                    {"type": "ValueError", "message": "Original V7 settling diagnostics faulted"}
                )
                trace.update(completed=False, status="failed")
        trace["model_score"] = None
        trace["fit_eligible"] = False
    return trace


def _materialize_profile(plan, output):
    if plan["observation_profile"] == benchmark.NATIVE_OBSERVATION_PROFILE:
        return None
    profile = _profile_module(plan["observation_profile"])

    target = output / "observation-profile"
    receipt = profile.materialize_profile(
        plan["upstream_path"],
        target,
        expected_source_sha256=plan["observation_helper_sha256"],
        profile=plan["observation_profile"],
    )
    return profile.verify_materialized_profile(
        target,
        expected_manifest_sha256=receipt["manifest_sha256"],
        expected_source_sha256=plan["observation_helper_sha256"],
        profile=plan["observation_profile"],
    )


@contextmanager
def _load_runtime(plan, transport, runtime):
    """Reuse existing loader, or its same shim with the reviewed V7 handle type.

    The V7 library owns its sole Browser.observe derivation. Agent/model/choose/
    action_space/executor bytes remain its authenticated canonical parent bytes.
    No benchmark cases, held-out goals or benchmark runner are imported here.
    """
    if plan["observation_profile"] != SETTLED_V7_PROFILE:
        with benchmark.load_upstream(
            plan["upstream_path"], transport, observation_runtime=runtime
        ) as modules:
            yield modules
        return
    profile = _profile_module(SETTLED_V7_PROFILE)

    def verify():
        if type(runtime) is not profile.VerifiedMaterializedProfile:
            raise ValueError("A verified settled-V7 runtime handle is required")
        checked = profile.verify_materialized_profile(
            runtime.path,
            expected_manifest_sha256=runtime.manifest_sha256,
            expected_source_sha256=plan["observation_helper_sha256"],
            profile=SETTLED_V7_PROFILE,
        )
        if (
            checked.manifest()["canonical_source_sha256"]
            != benchmark.upstream_manifest(plan["upstream_path"])["source_sha256"]
        ):
            raise ValueError("Settled-V7 canonical source changed")
        return checked

    runtime = verify()
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == "browser_harness"
        or name.startswith("browser_harness.")
        or name == "jev_ultrafast"
        or name.startswith("jev_ultrafast.")
    }
    previous_path = list(sys.path)
    previous_bytecode = sys.dont_write_bytecode
    previous_expressions = transport.observation_profile_expressions
    for name in saved:
        del sys.modules[name]
    package = types.ModuleType("browser_harness")
    package.__path__ = []
    admin = types.ModuleType("browser_harness.admin")
    admin.ensure_daemon = lambda: None
    helpers = types.ModuleType("browser_harness.helpers")
    helpers.cdp = transport.cdp
    sys.modules.update(
        {
            "browser_harness": package,
            "browser_harness.admin": admin,
            "browser_harness.helpers": helpers,
        }
    )
    sys.path.insert(0, str(runtime.runtime_root))
    sys.dont_write_bytecode = True
    transport.observation_profile_expressions = {
        runtime.read_state: "READ_STATE",
        runtime.marker_expression: "MARKER",
    }
    primary = None

    def verify_imports():
        native = sys.modules["jev_ultrafast.browser"]
        if native.READ_STATE != runtime.read_state or native.MARKER != runtime.marker_expression:
            raise ValueError("Imported settled-V7 observer expressions changed")
        for label, expected in runtime.manifest()["runtime_source_sha256"].items():
            path = runtime.runtime_root / label
            if _sha(path.read_bytes()) != expected:
                raise ValueError("Settled-V7 runtime source changed")
            if label.endswith(".py"):
                name = label.removesuffix(".py").replace("/", ".").removesuffix(".__init__")
                if Path(sys.modules[name].__file__).resolve() != path:
                    raise ValueError("Imported settled-V7 source escaped its reviewed runtime")

    try:
        modules = (
            importlib.import_module("jev_ultrafast.agent"),
            importlib.import_module("jev_ultrafast.model"),
        )
        verify_imports()
        yield modules
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            verify_imports()
            verify()
        except BaseException as exc:
            transport.observation_profile_errors.append(
                {"phase": "settled_v7_closure", "type": type(exc).__name__, "message": str(exc)}
            )
            if primary is None:
                raise
            primary.add_note("Settled-V7 source closure failed: " + repr(exc))
        finally:
            for name in tuple(sys.modules):
                if (
                    name == "browser_harness"
                    or name.startswith("browser_harness.")
                    or name == "jev_ultrafast"
                    or name.startswith("jev_ultrafast.")
                ):
                    del sys.modules[name]
            sys.modules.update(saved)
            sys.path[:] = previous_path
            sys.dont_write_bytecode = previous_bytecode
            transport.observation_profile_expressions = previous_expressions


def _replan(plan):
    scenario_profile, _ = _plan_scenarios(plan)
    return collection_plan(
        output_dir=plan["output_dir"],
        upstream=plan["upstream_path"],
        collection_id=plan["collection_id"],
        seed=plan["seed"],
        exclusion_manifest_path=plan["exclusion_manifest_path"],
        chromium=None if plan["chromium"] is None else plan["chromium"]["path"],
        observation_profile=plan["observation_profile"],
        observation_helper_sha256=plan["observation_helper_sha256"],
        scenario_profile=scenario_profile,
    )


def collect(plan, *, collector_sha256, plan_sha256):
    """Explicit execution boundary; pins and executable are checked before writes."""
    if _hash_pin(collector_sha256) != _sha(Path(__file__).read_bytes()):
        raise ValueError("Reviewed collector SHA256 does not match")
    if _hash_pin(plan_sha256) != _digest(plan) or _encode(_replan(plan)) != _encode(plan):
        raise ValueError("Reviewed prospective plan changed")
    scenario_profile, selected = _plan_scenarios(plan)
    if plan["chromium"] is None:
        raise ValueError("Explicit collection requires a pinned --chromium executable")
    output = _prospective_output(plan["output_dir"])
    output.mkdir(mode=0o700)
    _write_json(output / "plan.json", plan)
    _write_json(
        output / "receipt.pending.json",
        {
            "schema_id": SCHEMA,
            "scenario_profile": scenario_profile,
            "plan_sha256": plan_sha256,
            "status": "pending",
            "completed": False,
            "model_score": None,
            "fit_eligible": False,
        },
    )
    receipt = {
        "schema_id": SCHEMA,
        "schema_version": 1,
        "scenario_profile": scenario_profile,
        "plan_sha256": plan_sha256,
        "status": "pending",
        "completed": False,
        "episodes": [],
        "errors": [],
        "model_calls": 0,
        "model_score": None,
        "fit_eligible": False,
        "admission_status": "pending_independent_admission",
        "browser_closed": False,
        "context_closed": False,
        "sources_unchanged": False,
        "upstream_unchanged": False,
    }
    browser, context, transport = None, None, None
    network = []
    try:
        for name, pin in plan["sources"].items():
            target = output / "sources" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            raw = (ROOT / name).read_bytes()
            if _sha(raw) != pin["sha256"] or len(raw) != pin["bytes"]:
                raise ValueError("Source changed before capture")
            with target.open("xb") as stream:
                stream.write(raw)
        runtime = _materialize_profile(plan, output)
        from playwright.sync_api import sync_playwright

        cases = [
            benchmark.FixtureCase(
                case_id=item["scenario"]["id"],
                task_id=item["scenario"]["id"],
                goal=item["scenario"]["goal"],
                html=selected.render_fixture(item["scenario"]),
                expected={},
            )
            for item in plan["episodes"]
        ]
        with benchmark.fixture_server(cases) as origin, sync_playwright() as playwright:
            receipt["origin"] = origin
            browser = playwright.chromium.launch(
                headless=True, executable_path=plan["chromium"]["resolved_path"]
            )
            receipt["browser_version"] = browser.version
            context = browser.new_context(
                viewport={"width": 1120, "height": 780}, service_workers="block"
            )
            allowed_paths = {case.path for case in cases} | {"/favicon.ico"}

            def route(request_route):
                request = request_route.request
                parsed = urlsplit(request.url)
                allowed = (
                    request.method in {"GET", "HEAD"}
                    and parsed.scheme == "http"
                    and parsed.netloc == urlsplit(origin).netloc
                    and parsed.path in allowed_paths
                    and not parsed.query
                )
                network.append({"url": request.url, "method": request.method, "allowed": allowed})
                request_route.continue_() if allowed else request_route.abort("blockedbyclient")

            context.route("**/*", route)
            transport = RecordingCDP(context)
            with (
                _publisher_environment(),
                _load_runtime(plan, transport, runtime) as (agent_module, model_module),
            ):
                read_state = sys.modules["jev_ultrafast.browser"].READ_STATE
                receipt["read_state_sha256"] = _sha(read_state.encode())
                _write_json(
                    output / "runtime-observation.json",
                    {
                        "read_state": read_state,
                        "profile_manifest_sha256": None
                        if runtime is None
                        else runtime.manifest_sha256,
                    },
                )
                for item in plan["episodes"]:
                    trace = collect_episode(
                        spec=item["scenario"],
                        episode_id=item["episode_id"],
                        origin=origin,
                        transport=transport,
                        agent_module=agent_module,
                        model_module=model_module,
                        upstream=plan["upstream_path"],
                        read_state=read_state,
                        scenario_profile=scenario_profile,
                    )
                    relative = "episodes/" + item["episode_id"] + ".json"
                    target = output / relative
                    target.parent.mkdir(exist_ok=True)
                    _write_json(target, trace)
                    receipt["episodes"].append(
                        {
                            "episode_id": item["episode_id"],
                            "path": relative,
                            "sha256": _sha(target.read_bytes()),
                            "completed": trace["completed"],
                            "status": trace["status"],
                            "rows": len(trace["rows"]),
                            "observed_menu_counts": trace["observed_menu_counts"],
                        }
                    )
            receipt["observation_profile_errors"] = copy.deepcopy(
                transport.observation_profile_errors
            )
            receipt["remaining_targets"] = len(transport.targets)
            if any(not item["allowed"] for item in network):
                raise ValueError("Loopback-only collection encountered a forbidden browser request")
            context.close()
            context = None
            receipt["context_closed"] = True
            browser.close()
            browser = None
            receipt["browser_closed"] = True
    except BaseException as exc:
        receipt["errors"].append({"type": type(exc).__name__, "message": str(exc)})
    finally:
        for resource, label in ((context, "context"), (browser, "browser")):
            if resource is not None:
                try:
                    resource.close()
                    receipt[label + "_closed"] = True
                except BaseException as exc:
                    receipt["errors"].append(
                        {"phase": label + "_close", "type": type(exc).__name__, "message": str(exc)}
                    )
        if transport is not None:
            _write_json(output / "cdp-journal.json", transport.journal)
            receipt["journal_sha256"] = _sha((output / "cdp-journal.json").read_bytes())
            receipt["observation_profile_errors"] = copy.deepcopy(
                transport.observation_profile_errors
            )
        _write_json(output / "network.json", network)
        receipt["network_sha256"] = _sha((output / "network.json").read_bytes())
        try:
            _assert_sources(plan["sources"], scenario_profile)
            receipt["sources_unchanged"] = True
            if benchmark.upstream_manifest(plan["upstream_path"]) != plan["upstream"]:
                raise ValueError("Pinned canonical upstream changed")
            receipt["upstream_unchanged"] = True
            if _executable_pin(plan["chromium"]["path"]) != plan["chromium"]:
                raise ValueError("Pinned Chromium entry executable changed")
            receipt["chromium_entry_unchanged"] = True
        except BaseException as exc:
            receipt["errors"].append(
                {"phase": "source_closure", "type": type(exc).__name__, "message": str(exc)}
            )
        complete_population = len(receipt["episodes"]) == len(plan["episodes"]) and all(
            item["completed"] for item in receipt["episodes"]
        )
        receipt["completed"] = bool(
            complete_population
            and not receipt["errors"]
            and receipt["browser_closed"]
            and receipt["context_closed"]
            and receipt.get("remaining_targets") == 0
            and not receipt.get("observation_profile_errors", [])
            and receipt["sources_unchanged"]
            and receipt["upstream_unchanged"]
            and receipt.get("chromium_entry_unchanged") is True
        )
        receipt["status"] = (
            "completed_pending_admission" if receipt["completed"] else "failed_quarantined"
        )
        _write_json(output / "receipt.json", receipt)
    return receipt


@dataclass(frozen=True)
class AdmittedAffordanceCollection:
    """New local proof handle; it does not grant optimizer or qualification access."""

    _rows: tuple[bytes, ...]
    receipt_sha256: str
    plan_sha256: str
    _seal: str = field(repr=False)
    _proof: object = field(repr=False, compare=False)


def _confined_file(output, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or str(path) != relative or "\\" in relative:
        raise ValueError("Evidence reference escapes collection output")
    full = output / path
    if any(parent.is_symlink() for parent in (full, *full.parents)):
        raise ValueError("Evidence references cannot traverse symlinks")
    return full


def _verify_episode_ownership(trace, journal, origin, read_state):
    start, end = trace["cdp_start"], trace["cdp_end"]
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(journal):
        raise ValueError("Owned episode protocol span is invalid")
    calls = journal[start:end]
    if any("error" in call for call in calls):
        raise ValueError("An episode with protocol errors cannot be admitted")
    created, attached, closed = calls[0], calls[1], calls[-1]
    target = created.get("response", {}).get("targetId")
    session = attached.get("response", {}).get("sessionId")
    if (
        created["method"] != "Target.createTarget"
        or created["params"] != {"url": "about:blank", "background": True}
        or not target
        or attached["method"] != "Target.attachToTarget"
        or attached["params"] != {"targetId": target, "flatten": True}
        or not session
        or closed["method"] != "Target.closeTarget"
        or closed["params"] != {"targetId": target}
        or closed.get("response") != {"success": True}
    ):
        raise ValueError("Actual native target creation, ownership and closure do not join")
    if any(call["session"] != session for call in calls[2:-1]):
        raise ValueError("Episode commands crossed the actual attached owned session")
    expected_url = origin + "/tasks/" + trace["scenario"]["id"]
    navigations = [call for call in calls if call["method"] == "Page.navigate"]
    if len(navigations) != 1 or navigations[0]["params"] != {"url": expected_url}:
        raise ValueError("Original native navigation is not the authored loopback fixture")
    for step in trace["steps"]:
        if not start < step["tick_start"] < step["tick_end"] < end:
            raise ValueError("A native step escaped its actual owned episode")
        if step["before"]["url"] != expected_url or step["after"]["url"] != expected_url:
            raise ValueError("Private DOM evidence does not belong to the owned authored URL")
    steps = trace["steps"]
    if not steps or steps[-1]["tick_end"] != end - 1:
        raise ValueError("Native terminal handling must be immediately followed by original close")
    for previous, current in zip(steps, steps[1:]):
        if (
            previous["tick_end"] != current["tick_start"]
            or previous["teacher"]["operation"] == "DONE"
        ):
            raise ValueError(
                "Native ticks must advance once with no reused or post-terminal evidence"
            )
    setup = journal[start + 2 : steps[0]["tick_start"]]
    expected_setup = [
        (
            "Emulation.setDeviceMetricsOverride",
            {"width": 1120, "height": 780, "deviceScaleFactor": 1, "mobile": False},
        ),
        ("Emulation.setFocusEmulationEnabled", {"enabled": True}),
        ("Page.navigate", {"url": expected_url}),
    ]
    if [(call["method"], call["params"]) for call in setup[:3]] != expected_setup:
        raise ValueError("Original browser setup changed")
    reads = setup[3:]
    if not reads or any(
        call["method"] != "Runtime.evaluate"
        or call["params"].get("expression") not in {"document.readyState", read_state}
        for call in reads
    ):
        raise ValueError("Unassigned execution exists before the first native tick")
    return start, end


def admit_training_collection(output_dir, *, expected_receipt_sha256, expected_plan_sha256):
    """Explicit independent replay of every planned episode, not a boolean receipt."""
    output = Path(output_dir).absolute()
    plan, actual_plan_sha = _plain_json(output / "plan.json", limit=8_000_000)
    receipt, receipt_sha = _plain_json(output / "receipt.json", limit=8_000_000)
    if actual_plan_sha != _hash_pin(expected_plan_sha256) or receipt_sha != _hash_pin(
        expected_receipt_sha256
    ):
        raise ValueError("Explicit independent plan/receipt pins differ")
    if (
        plan.get("schema_id") != SCHEMA
        or receipt.get("schema_id") != SCHEMA
        or plan.get("split") != "train"
    ):
        raise ValueError("Foreign collection schema or nontraining split")
    if (
        receipt.get("plan_sha256") != actual_plan_sha
        or not receipt.get("completed")
        or receipt.get("status") != "completed_pending_admission"
    ):
        raise ValueError("Incomplete or failed collections are quarantined")
    scenario_profile, selected = _plan_scenarios(plan)
    if receipt.get("scenario_profile", DEFAULT_SCENARIO_PROFILE) != scenario_profile:
        raise ValueError("Receipt scenario profile differs from the selected plan")
    if (
        receipt.get("model_calls") != 0
        or receipt.get("model_score") is not None
        or receipt.get("fit_eligible") is not False
    ):
        raise ValueError("Collection cannot claim model quality or automatic fit eligibility")
    if (
        receipt.get("errors")
        or not all(
            receipt.get(key) is True
            for key in (
                "browser_closed",
                "context_closed",
                "sources_unchanged",
                "upstream_unchanged",
                "chromium_entry_unchanged",
            )
        )
        or receipt.get("remaining_targets") != 0
        or receipt.get("observation_profile_errors")
    ):
        raise ValueError("Collection closure is incomplete")
    _assert_sources(plan["sources"], scenario_profile)
    if benchmark.upstream_manifest(plan["upstream_path"]) != plan["upstream"]:
        raise ValueError("Canonical upstream source changed")
    if _executable_pin(plan["chromium"]["path"]) != plan["chromium"]:
        raise ValueError("Pinned Chromium entry executable changed")
    exclusions, exclusions_sha = exclusion_manifest(plan["exclusion_manifest_path"])
    if exclusions_sha != plan["exclusion_manifest_sha256"] or exclusions != plan["exclusions"]:
        raise ValueError("Reviewed exclusion inventory changed")
    for name, pin in plan["sources"].items():
        raw = _confined_file(output, "sources/" + name).read_bytes()
        if _sha(raw) != pin["sha256"] or len(raw) != pin["bytes"]:
            raise ValueError("Captured source changed")
    expected_specs = selected.scenarios(plan["seed"], set(exclusions["goal_sha256"]))
    if [item["scenario"] for item in plan["episodes"]] != list(expected_specs):
        raise ValueError("Planned scenario population differs from the new authored inventory")
    journal, journal_sha = _plain_json(output / "cdp-journal.json")
    if journal_sha != receipt.get("journal_sha256") or any(
        call.get("sequence") != index for index, call in enumerate(journal)
    ):
        raise ValueError("Raw protocol journal changed")
    observation, _ = _plain_json(output / "runtime-observation.json", limit=2_000_000)
    read_state = observation["read_state"]
    if _sha(read_state.encode()) != receipt["read_state_sha256"]:
        raise ValueError("Observation expression changed")
    canonical_read_state = (
        Path(plan["upstream_path"]) / "jev_ultrafast" / "snapshot.js"
    ).read_text()
    if plan["observation_profile"] == benchmark.NATIVE_OBSERVATION_PROFILE:
        # Browser.py loads the UTF-8 snapshot source directly.
        if read_state != canonical_read_state:
            raise ValueError("Native collection changed the canonical observer")
    else:
        profile = _profile_module(plan["observation_profile"])

        runtime = profile.verify_materialized_profile(
            output / "observation-profile",
            expected_manifest_sha256=observation["profile_manifest_sha256"],
            expected_source_sha256=plan["observation_helper_sha256"],
            profile=plan["observation_profile"],
        )
        if runtime.read_state != read_state:
            raise ValueError("Verified observation profile changed")
    network, network_sha = _plain_json(output / "network.json", limit=2_000_000)
    origin = receipt.get("origin")
    parsed_origin = urlsplit(origin) if type(origin) is str else None
    if (
        parsed_origin is None
        or parsed_origin.scheme != "http"
        or parsed_origin.hostname != "127.0.0.1"
        or not parsed_origin.port
        or origin != "http://127.0.0.1:" + str(parsed_origin.port)
    ):
        raise ValueError("Collection origin is not the actual bounded loopback server")
    allowed_paths = {"/tasks/" + item["scenario"]["id"] for item in plan["episodes"]} | {
        "/favicon.ico"
    }
    if network_sha != receipt.get("network_sha256") or not network:
        raise ValueError("Loopback network evidence is incomplete")
    for item in network:
        parsed = urlsplit(item["url"])
        if (
            item.get("allowed") is not True
            or item.get("method") not in {"GET", "HEAD"}
            or parsed.scheme != "http"
            or parsed.netloc != parsed_origin.netloc
            or parsed.path not in allowed_paths
            or parsed.query
        ):
            raise ValueError("Actual browser request violates the authored loopback-only scope")
    if len(receipt["episodes"]) != len(plan["episodes"]):
        raise ValueError(
            "Every planned episode, including failures, must remain in the denominator"
        )
    encoded_rows = []
    observed = set()
    menu_roles = {}
    previous_cdp_end = 0
    for planned, summary in zip(plan["episodes"], receipt["episodes"], strict=True):
        spec = planned["scenario"]
        if (
            _digest(spec) != planned["scenario_sha256"]
            or _sha(selected.render_fixture(spec).encode()) != planned["fixture_sha256"]
        ):
            raise ValueError("Authored fixture or scenario changed")
        if (
            _goal_hash(spec["goal"]) in exclusions["goal_sha256"]
            or spec["id"] in exclusions["scenario_ids"]
            or spec["ancestor_id"] in exclusions["ancestor_ids"]
        ):
            raise ValueError("A held-out identity or goal entered the training collection")
        if (
            summary["episode_id"] != planned["episode_id"]
            or summary["path"] != "episodes/" + planned["episode_id"] + ".json"
        ):
            raise ValueError("Episode identity changed")
        trace, trace_sha = _plain_json(_confined_file(output, summary["path"]))
        if (
            trace_sha != summary["sha256"]
            or trace.get("schema_id") != SCHEMA
            or trace.get("scenario") != spec
            or trace.get("scenario_profile", DEFAULT_SCENARIO_PROFILE) != scenario_profile
        ):
            raise ValueError("Episode evidence changed or belongs to a foreign schema")
        if (
            not trace.get("completed")
            or trace.get("status") != "completed"
            or trace.get("errors")
            or not trace.get("agent_closed")
        ):
            raise ValueError("Failed or unclosed episodes are not admitted")
        if plan["observation_profile"] == SETTLED_V7_PROFILE:
            settling = trace.get("observation_settling", {})
            if settling.get("fault") is not False or not settling.get("records"):
                raise ValueError(
                    "Original V7 observation-settling diagnostics are missing or faulted"
                )
        start, end = _verify_episode_ownership(trace, journal, origin, read_state)
        if start != previous_cdp_end:
            raise ValueError("Unassigned or overlapping native browser commands exist")
        previous_cdp_end = end
        steps, rows = trace["steps"], trace["rows"]
        if (
            not steps
            or len(steps) != len(rows)
            or len(steps) > MAX_TICKS
            or len(rows) != summary["rows"]
        ):
            raise ValueError("Episode row/step population is incomplete")
        previous = None
        episode_counts = set()
        for index, (step, row) in enumerate(zip(steps, rows, strict=True)):
            if (
                step["index"] != index
                or step["scenario"] != spec
                or (previous is not None and step["before"] != previous["after"])
            ):
                raise ValueError("Episode transition sequence is discontinuous")
            if index == 0 and step["history_before"]:
                raise ValueError("Each new episode must start with empty native history")
            if previous is not None and step["history_before"] != previous["history_after"]:
                raise ValueError("Original Agent history was replaced")
            proof = _verify_step(
                step,
                journal,
                plan["upstream_path"],
                read_state,
                scenario_profile=scenario_profile,
            )
            if proof != step.get("proof"):
                raise ValueError("Stored proof differs from independent evidence replay")
            rebuilt = build_native_training_row(
                step["payload"],
                row_id=planned["episode_id"] + f".step-{index:02d}",
                scenario_id=spec["id"],
                ancestor_id=spec["ancestor_id"],
                episode_id=planned["episode_id"],
                step=index,
                operation=step["teacher"]["operation"],
                target=step["target"],
                proof_refs={"transition": planned["episode_id"] + f"#step-{index}"},
            )
            if _encode(row) != _encode(rebuilt):
                raise ValueError(
                    "Native wire, ordering, alias, annotation, loss mask or proof identity changed"
                )
            count = len(row["payload"]["questions"]["operation"]["criteria"])
            observed.add(count)
            episode_counts.add(count)
            menu_roles.setdefault(count, set()).add(
                "completed" if step["teacher"]["operation"] == "DONE" else "continuing"
            )
            encoded_rows.append(_encode(row))
            previous = step
        if steps[-1]["teacher"]["operation"] != "DONE" or not selected.verify_goal(
            spec, steps[-1]["after"]
        ):
            raise ValueError("Every completed episode requires actual independently verified DONE")
        if (
            episode_counts != set(planned["intended_menu_counts"])
            or sorted(episode_counts) != trace["observed_menu_counts"]
            or trace["observed_menu_counts"] != summary["observed_menu_counts"]
        ):
            raise ValueError("Intended K bins were not genuinely observed in this complete episode")
        if not any(step["teacher"]["operation"] != "DONE" for step in steps):
            raise ValueError("A menu bin must include continuing and completed states")
    _assert_menu_coverage(observed, scenario_profile)
    if any(roles != {"continuing", "completed"} for roles in menu_roles.values()):
        raise ValueError("Each observed K must have continuing and verified completed states")
    if previous_cdp_end != len(journal):
        raise ValueError("Unassigned native browser commands remain after closure")
    seal = _digest(
        {
            "rows": [blob.decode() for blob in encoded_rows],
            "receipt": receipt_sha,
            "plan": actual_plan_sha,
        }
    )
    return AdmittedAffordanceCollection(
        tuple(encoded_rows), receipt_sha, actual_plan_sha, seal, _ADMISSION_PROOF
    )


def compile_wire_projection(admitted, *, model_alias=None):
    """Full wire projection only after explicit evidence replay; no alias rewriting."""
    if (
        type(admitted) is not AdmittedAffordanceCollection
        or admitted._proof is not _ADMISSION_PROOF
    ):
        raise ValueError("A new independently admitted affordance collection handle is required")
    if (
        _digest(
            {
                "rows": [blob.decode() for blob in admitted._rows],
                "receipt": admitted.receipt_sha256,
                "plan": admitted.plan_sha256,
            }
        )
        != admitted._seal
    ):
        raise ValueError("Admitted row bytes or collection identity changed")
    if model_alias not in (None, MODEL_ALIAS):
        raise ValueError("The original native model alias cannot be rewritten")
    result = []
    for encoded in admitted._rows:
        row = json.loads(encoded)
        if row["split"] != "train" or row["schema_id"] != ROW_SCHEMA:
            raise ValueError("Foreign or nontraining rows are not projected")
        result.append(
            {
                "case_id": row["id"],
                "payload": row["payload"],
                "teacher_choice": row["teacher_choice"],
                "valid_next_targets": row["valid_next_targets"],
                "loss_mask": row["loss_mask"],
                "wire_sha256": row["wire_sha256"],
                "fit_eligible": False,
            }
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--exclusion-manifest", required=True)
    parser.add_argument("--chromium")
    parser.add_argument(
        "--scenario-profile", choices=SCENARIO_PROFILES, default=DEFAULT_SCENARIO_PROFILE
    )
    parser.add_argument(
        "--observation-profile",
        required=True,
        choices=[
            benchmark.NATIVE_OBSERVATION_PROFILE,
            benchmark.COMPOSED_OBSERVATION_PROFILE,
            benchmark.PRUNED_OBSERVATION_PROFILE,
            SETTLED_V7_PROFILE,
        ],
    )
    parser.add_argument(
        "--observation-helper-sha256",
        "--observation-profile-helper-sha256",
        dest="observation_helper_sha256",
    )
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--collector-sha256")
    parser.add_argument("--plan-sha256")
    args = parser.parse_args(argv)
    plan = collection_plan(
        output_dir=args.output_dir,
        upstream=args.upstream,
        collection_id=args.collection_id,
        seed=args.seed,
        exclusion_manifest_path=args.exclusion_manifest,
        chromium=args.chromium,
        observation_profile=args.observation_profile,
        observation_helper_sha256=args.observation_helper_sha256,
        scenario_profile=args.scenario_profile,
    )
    if not args.collect:
        if args.collector_sha256 or args.plan_sha256:
            parser.error("Execution pins are used only with explicit --collect")
        print(
            json.dumps(
                {"dry_run": True, "plan": plan, "plan_sha256": _digest(plan)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    receipt = collect(plan, collector_sha256=args.collector_sha256, plan_sha256=args.plan_sha256)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
