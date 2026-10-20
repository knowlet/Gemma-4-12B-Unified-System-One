"""Pure closed-byte capacity selection; no processor, model, service or browser."""

import importlib.util
import sys
from pathlib import Path

KIND = "jev_v8_actual_payload_capacity_admission_v1"
PROFILE = "composed-visible-hover-entries-settled-v8"
VALIDATOR_SHA = "1b8c621c77e68fb42e8e00b1c37b29f4847746f6214f020d1a54281f41464b4e"
PLAN_SHA = "d6cbc3b14948e2f15fb0e5a98e528c928d3ad3101ed4dae12a6019f20c7058fb"
ROOT_SELECTION_SHA = "724a74e741db5f7ad1b88a0bc91a0b9ed778931e344983b581f5ba7cfd386995"


def validator(root, expected):
    import hashlib

    path = Path(root) / expected["path"]
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Ordinary selected capacity validator required")
    blob = path.read_bytes()
    if (
        expected["sha256"] != VALIDATOR_SHA
        or len(blob) != expected["size_bytes"]
        or hashlib.sha256(blob).hexdigest() != VALIDATOR_SHA
    ):
        raise ValueError("Exact corrected H02 typed validator source required")
    spec = importlib.util.spec_from_file_location("_selected_v8_closed_capacity_validator", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(blob, str(path), "exec"), module.__dict__)
    return module


def validate_selection(selection, expected, *, root):
    """Bind genuine inputs and confirmed outer closure independently of technical Q."""
    v = validator(root, expected["validator_source"])
    selected = v.strict(v.checked(expected["root_selection"], root))
    v.require(
        expected["root_selection"]["sha256"] == ROOT_SELECTION_SHA
        and selected["schema"] == "root_v8_capacity_inputs_selection_v1"
        and selected["status"] == "SELECTED_EXACT_CLOSED_CPU_INPUTS_FIXED_23_ROWS"
        and selected["inputs"] == expected["selected_inputs"]
        and selected["outer_receipt"] == expected["selected_outer"],
        "Exact independent-CLEAR root capacity selection required",
    )
    v.keys(selection, {"schema_id", "kind", "profile", "plan_sha256", "inputs", "outer"})
    v.require(
        selection["schema_id"] == "jev-v8-closed-capacity-selection-v1"
        and selection["kind"] == KIND
        and selection["profile"] == PROFILE
        and selection["plan_sha256"] == expected["plan"]["sha256"] == PLAN_SHA,
        "Fresh explicit v8 capacity selection required; technical Q is insufficient",
    )
    for name in ("inputs", "outer"):
        v.require(
            type(selection[name]) is dict, "Pending capacity reference cannot authorize a run"
        )
        approved = expected["selected_" + name]
        v.require(
            approved is None or selection[name] == approved, "Root-selected capacity pin changed"
        )
    plan = v.strict(v.checked(expected["plan"], root))
    v.require(plan["kind"] == KIND and plan["profile"] == PROFILE, "Capacity plan identity")
    inputs_path = v.ordinary(Path(root) / selection["inputs"]["path"])
    outer_path = v.ordinary(Path(root) / selection["outer"]["path"])
    directory = v.ordinary(Path(root) / expected["prepare_directory"])
    v.require(
        inputs_path.parent == outer_path.parent == directory
        and inputs_path.name == "capacity-inputs.json"
        and outer_path.name == "controller-terminal.json",
        "One exact closed CPU prepare tree",
    )
    value = v.strict(v.checked(selection["inputs"], root))
    outer = v.strict(v.checked(selection["outer"], root))
    v.validate_admission(value, plan_sha256=PLAN_SHA, output=directory)
    v.require(
        value["counts"] == selected["counts"]
        and all(row["outcome"] == "ACCEPTED" for row in value["rows"][:16]),
        "Selected genuine actual16 accepted; fixed23 retains edge refusals",
    )
    wanted = {
        "cohort": plan["cohort"],
        "model_authority_only": plan["model_authority_only"],
        "observation": plan["observation"],
        "execution95": plan["technical"]["execution"],
        "runtime": plan["technical"]["runtime"],
    }
    v.require(
        v.canonical(value["bindings"]) == v.canonical(wanted)
        and v.canonical(value["bindings"]["observation"]) == v.canonical(expected["observation"]),
        "Exact new profile and original technical model authority bindings",
    )
    v.require(
        outer["plan_sha256"] == PLAN_SHA
        and outer["qualified"] is False
        and outer["status"] == "CLOSED_CPU_PREPARED_REQUIRES_EXTERNAL_REVIEW"
        and outer["admission_complete"] is True
        and outer["primary_error"] is None
        and type(outer["child_returncode"]) is int
        and outer["child_returncode"] == 0,
        "Successful whole owned CPU closure required",
    )
    v.require(
        outer["inputs"]
        == {
            "path": "capacity-inputs.json",
            **{key: selection["inputs"][key] for key in ("sha256", "size_bytes")},
        },
        "Outer receipt selects these exact genuine input bytes",
    )
    cleanup = outer["owned_cleanup"]
    v.require(
        cleanup["confirmed"] is True
        and cleanup["remaining"] == []
        and type(cleanup["tracked_groups"]) is list
        and bool(cleanup["tracked_groups"])
        and all(type(x) is int and x > 0 for x in cleanup["tracked_groups"])
        and type(cleanup["tracked_PID_start_ticks"]) is dict
        and bool(cleanup["tracked_PID_start_ticks"])
        and all(
            type(k) is str and k.isdigit() and int(k) > 0 and type(ticks) is int and ticks > 0
            for k, ticks in cleanup["tracked_PID_start_ticks"].items()
        ),
        "Tracked PID/start/group closure required; no reused or unknown group",
    )
    slots = outer["all23_terminal_or_explicit_unknown"]
    v.require(
        type(slots) is list and len(slots) == len(value["rows"]) == 23,
        "Full16+7 closure denominator",
    )
    for ordinal, (slot, row) in enumerate(zip(slots, value["rows"], strict=True)):
        v.require(
            type(slot["ordinal"]) is int
            and slot["ordinal"] == ordinal
            and slot["id"] == row["id"] == v.IDS[ordinal]
            and slot["outcome"] == row["outcome"] != "UNKNOWN"
            and slot["expectation_met"] is row["expectation_met"] is True
            and slot["error"] == row["error"],
            "All23 exact closed outcome joins",
        )
    v.require(
        outer["full_denominator"] == 23
        and outer["actual_denominator"] == 16
        and outer["edge_denominator"] == 7,
        "Exact actual16 plus7 edge scope",
    )
    return {
        "kind": KIND,
        "profile": PROFILE,
        "plan_sha256": PLAN_SHA,
        "inputs": selection["inputs"],
        "outer": selection["outer"],
        "counts": value["counts"],
        "qualified": False,
        "universal_profile_capacity_attested": False,
        "task_quality_attested": False,
    }
