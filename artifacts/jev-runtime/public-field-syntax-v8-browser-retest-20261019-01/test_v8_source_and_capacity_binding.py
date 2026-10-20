"""Pure inverse/source and synthetic closed-evidence tests; no live service."""

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import v8_capacity_binding as capacity

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
OLD = ROOT / "artifacts/jev-runtime/public-field-syntax-v7-browser-retest-20261018-01"
PLAN = json.loads((BASE / "plan-01.json").read_bytes())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def test_observer_inverse_bytes_and_original_transport_logic():
    old = (OLD / "browser_observer_runner.py").read_text()
    new = (BASE / "browser_observer_runner.py").read_text()
    start, end = new.index("def capacity_state(binding):"), new.index("def binding_state(binding):")
    inverse = new[:start] + new[end:]
    inverse = inverse.replace("    capacity = capacity_state(binding)\n", "", 1)
    inverse = inverse.replace('        "v8_capacity": capacity,\n', "", 1)
    replacements = json.loads((BASE / "observer-literal-replacements.json").read_bytes())
    for before, after in reversed(list(replacements.items())):
        inverse = inverse.replace(after, before)
    assert inverse == old
    assert (
        new[new.index("class TransportObserver:") : new.index("def process_identity(")]
        == old[old.index("class TransportObserver:") : old.index("def process_identity(")]
    )


def test_controller_only_observer_SHA_delta_and_default_dry_no_private_read(monkeypatch):
    old = (OLD / "root_owned_group_controller.py").read_text()
    new = (BASE / "root_owned_group_controller.py").read_text()
    assert (
        new.replace(
            hashlib.sha256((BASE / "browser_observer_runner.py").read_bytes()).hexdigest(),
            "0aa07134ea272c2a62b6534e3a3506b9e2006e9886bc506a210b349e4b03c960",
            1,
        )
        == old
    )
    module = load("test_v8_owned_controller", BASE / "root_owned_group_controller.py")
    monkeypatch.setattr(module, "prepare", lambda *args: (None, {}, []))

    def refuse(*args, **kwargs):
        raise AssertionError("No private read or process")

    monkeypatch.setattr(module, "bound_environment", refuse)
    monkeypatch.setattr(module.subprocess, "Popen", refuse)
    assert module.execute("unused", "0" * 64) == 0


def test_profile_checker_inverse_literal_only_and_native_helper_checks_unchanged():
    text = (BASE / "review_v8_profile_evidence.py").read_text()
    for before, after in (
        ("v8 checks", "v7 checks"),
        (capacity.PROFILE, "composed-visible-text-settled-v7"),
        (
            "jev-v8-native-observation-settling-sidecar-v1",
            "jev-v7-native-observation-settling-sidecar-v1",
        ),
        ("jev_browser_profile_v8_benchmark.py", "jev_browser_profile_v7_benchmark.py"),
        ("jev_observation_profile_v8.py", "jev_observation_profile_v7.py"),
        ("_saved_v8_audit_", "_saved_v7_audit_"),
    ):
        text = text.replace(before, after)
    assert text == (OLD / "review_v7_profile_evidence.py").read_text()
    assert "module.verify_materialized_profile(" in text


def test_fixed10_budgets_and_original_Q_remain_separate_from_capacity():
    old = json.loads((OLD / "plan-01.json").read_bytes())
    assert PLAN["full_denominator"] == old["full_denominator"]
    assert PLAN["fixed_case_manifest"] == old["fixed_case_manifest"]
    for new, before in zip(
        PLAN["groups_in_execution_order"], old["groups_in_execution_order"], strict=True
    ):
        for key in (
            "group",
            "denominator",
            "expected_attempt_ids",
            "selected_case_ids_in_order",
            "repeats",
        ):
            assert new[key] == before[key]
    for key in (
        "decision_limit",
        "helper_HTTP_client_seconds",
        "model_deadline_seconds",
        "native_helper_generated_token_cap_including_allEOS",
        "prepared_maximum_context_tokens",
        "reasoning",
        "repeats",
        "seed",
        "strict_success",
    ):
        assert PLAN["judge_and_budget"][key] == old["judge_and_budget"][key]
    assert PLAN["existing_qualification"] == old["existing_qualification"]
    assert (
        PLAN["v8_capacity"]["selected_inputs"]["sha256"]
        == "dca384b048118e15b134ee1201e2ec3602745e349ff025fa39a63d1a514c4063"
    )
    assert (
        PLAN["v8_capacity"]["selected_outer"]["sha256"]
        == "580e966d02958d653d586fb1f0134075a4ed003c48cea73563eaf51a025f8ef6"
    )
    assert PLAN["v8_capacity"]["root_selection"]["sha256"] == capacity.ROOT_SELECTION_SHA
    assert PLAN["service_start_proposal"]["execute"] is False


@pytest.fixture
def closed(tmp_path, monkeypatch):
    """Synthetic bytes/IDs only; production source/SHA constants are locally patched."""
    v = capacity.validator(ROOT, PLAN["v8_capacity"]["validator_source"])
    directory = tmp_path / "prepare"
    directory.mkdir()
    real = v.strict(v.checked(PLAN["v8_capacity"]["plan"], ROOT))
    plan = {
        k: real[k]
        for k in ("kind", "profile", "cohort", "model_authority_only", "observation", "technical")
    }

    def put(path, value, *, raw=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = value if raw else v.wire(value)
        path.write_bytes(blob)
        return {
            "path": str(path.relative_to(tmp_path)),
            "sha256": v.sha(blob),
            "size_bytes": len(blob),
        }

    plan_ref = put(tmp_path / "plan.json", plan)
    monkeypatch.setattr(capacity, "PLAN_SHA", plan_ref["sha256"])
    rows = []
    for ordinal, case_id in enumerate(v.IDS):
        kind = "helper" if ordinal >= 21 else "policy"
        outcome = (
            "REFUSED_SCHEMA"
            if ordinal == 17
            else "REFUSED_CONTEXT"
            if ordinal in (20, 22)
            else "ACCEPTED"
        )
        original = {
            "model": "fixture-no-LLM",
            "state": {"text": "Synthetic typed fixture"},
            "questions": {"operation": {"criteria": {"WAIT": "Wait", "DONE": "Done"}}},
        }
        payload = copy.deepcopy(original)
        payload["model"] = v.IDENTITY
        count = 8161 if ordinal == 20 else 8065 if ordinal == 22 else 2
        prompt = b"Synthetic prefix; not a real processor result"
        row = {
            "ordinal": ordinal,
            "id": case_id,
            "kind": kind,
            "outcome": outcome,
            "native_forward_calls": 0,
            "expectation_met": True,
            "payload": put(directory / f"body-{ordinal}.json", v.wire(payload), raw=True),
            "prefix": None,
            "prompt": None,
            "provenance": None,
            "error": {"type": "RequestValidationError", "message": "Synthetic schema refusal"}
            if ordinal == 17
            else None,
        }
        if ordinal != 17:
            row["prefix"] = {
                "ids": [7] * count,
                "attention_mask": [1] * count,
                "token_count": count,
                "prompt_sha256": v.sha(prompt),
            }
            prompt_ref = put(directory / f"prompt-{ordinal}.txt", prompt, raw=True)
            row["prompt"] = {**prompt_ref, "path": Path(prompt_ref["path"]).name}
        row["payload"]["path"] = Path(row["payload"]["path"]).name
        if ordinal < 16:
            original_ref = put(directory / f"original-{ordinal}.json", v.wire(original), raw=True)
            row["provenance"] = {
                "saved_canonical_payload_sha256": v.sha(v.matrix_payload_authority(original)),
                "native_original_ordered_body": {
                    **original_ref,
                    "path": Path(original_ref["path"]).name,
                },
            }
        rows.append(row)
    total = 128 * 1024**3
    reserve = max(24 * 1024**3, (total + 4) // 5)
    gate = {
        "sequence": 0,
        "phase": "startup",
        "scope": "cpu",
        "monotonic": 1.0,
        "expected_boot_id": "fixture-boot",
        "observation": {
            "boot_id": "fixture-boot",
            "MemTotal": total,
            "MemAvailable": reserve + 4 * 1024**3,
        },
        "reserve_bytes": reserve,
        "required_bytes": reserve + 4 * 1024**3,
        "swap_or_resident_credit_bytes": 0,
        "admitted": True,
        "error": None,
    }
    journal = b"\n".join(
        v.canonical(x)
        for x in [gate, {**gate, "sequence": 1, "phase": "final-cpu", "monotonic": 2.0}]
    )
    journal_ref = put(directory / "gates.jsonl", journal, raw=True)
    value = {
        "schema_id": v.KIND,
        "profile": v.PROFILE,
        "plan_sha256": plan_ref["sha256"],
        "qualified": False,
        "rows": rows,
        "counts": {name: sum(r["outcome"] == name for r in rows) for name in sorted(v.OUTCOMES)},
        "universal_profile_capacity_attested": False,
        "task_quality_attested": False,
        "gate_journal": {**journal_ref, "path": "gates.jsonl"},
        "CPU_resource_identity": {"boot_id": "fixture-boot", "MemTotal": total},
        "bindings": {
            "cohort": plan["cohort"],
            "model_authority_only": plan["model_authority_only"],
            "observation": plan["observation"],
            "execution95": plan["technical"]["execution"],
            "runtime": plan["technical"]["runtime"],
        },
    }
    outer = {
        "plan_sha256": plan_ref["sha256"],
        "qualified": False,
        "status": "CLOSED_CPU_PREPARED_REQUIRES_EXTERNAL_REVIEW",
        "admission_complete": True,
        "primary_error": None,
        "child_returncode": 0,
        "full_denominator": 23,
        "actual_denominator": 16,
        "edge_denominator": 7,
        "owned_cleanup": {
            "confirmed": True,
            "remaining": [],
            "tracked_groups": [123],
            "tracked_PID_start_ticks": {"123": 456},
        },
        "all23_terminal_or_explicit_unknown": [
            {k: r[k] for k in ("ordinal", "id", "outcome", "expectation_met", "error")}
            for r in rows
        ],
    }
    validator_ref = PLAN["v8_capacity"]["validator_source"]
    expected = {
        "validator_source": {**validator_ref, "path": str(ROOT / validator_ref["path"])},
        "plan": plan_ref,
        "prepare_directory": "prepare",
        "observation": plan["observation"],
    }

    def refresh():
        inputs = put(directory / "capacity-inputs.json", value)
        outer["inputs"] = {**inputs, "path": "capacity-inputs.json"}
        outer_ref = put(directory / "controller-terminal.json", outer)
        selected = {
            "schema": "root_v8_capacity_inputs_selection_v1",
            "status": "SELECTED_EXACT_CLOSED_CPU_INPUTS_FIXED_23_ROWS",
            "inputs": inputs,
            "outer_receipt": outer_ref,
            "counts": value["counts"],
        }
        authority = put(tmp_path / "selection.json", selected)
        monkeypatch.setattr(capacity, "ROOT_SELECTION_SHA", authority["sha256"])
        expected.update(root_selection=authority, selected_inputs=inputs, selected_outer=outer_ref)
        return {
            "schema_id": "jev-v8-closed-capacity-selection-v1",
            "kind": v.KIND,
            "profile": v.PROFILE,
            "plan_sha256": plan_ref["sha256"],
            "inputs": inputs,
            "outer": outer_ref,
        }

    return value, outer, expected, refresh, tmp_path


def test_closed_capacity_binding_keeps_refusals_distinct_from_quality(closed):
    value, outer, expected, refresh, root = closed
    result = capacity.validate_selection(refresh(), expected, root=root)
    assert result["counts"] == {
        "ACCEPTED": 20,
        "REFUSED_SCHEMA": 1,
        "REFUSED_CONTEXT": 2,
        "UNKNOWN": 0,
    }
    assert result["qualified"] is result["task_quality_attested"] is False


@pytest.mark.parametrize(
    "change", ["unknown", "unclosed", "old_qualification", "wrong_profile", "partial_slots"]
)
def test_capacity_failures_cannot_be_relabelled_as_admission(closed, change):
    value, outer, expected, refresh, root = closed
    if change == "unknown":
        outer["all23_terminal_or_explicit_unknown"][0]["outcome"] = "UNKNOWN"
    elif change == "unclosed":
        outer["owned_cleanup"]["confirmed"] = False
    elif change == "old_qualification":
        value["schema_id"] = "jev_public_action_field_syntax_bf16_qualification_v1"
    elif change == "wrong_profile":
        value["profile"] = "composed-visible-text-settled-v7"
    else:
        outer["all23_terminal_or_explicit_unknown"].pop()
    with pytest.raises(ValueError):
        capacity.validate_selection(refresh(), expected, root=root)


def test_pending_and_changed_root_selected_capacity_pins_refused(closed):
    value, outer, expected, refresh, root = closed
    selection = refresh()
    selection["outer"] = None
    with pytest.raises(ValueError):
        capacity.validate_selection(selection, expected, root=root)
    selection = refresh()
    selection["inputs"] = {**selection["inputs"], "sha256": "0" * 64}
    with pytest.raises(ValueError, match="pin changed"):
        capacity.validate_selection(selection, expected, root=root)


def test_capacity_refusal_precedes_original_live_config_and_Q_reads(monkeypatch):
    observer = load("test_v8_observer_precondition", BASE / "browser_observer_runner.py")

    def refuse_capacity(binding):
        raise observer.EvidenceError("No selected closed capacity")

    def forbidden_read(*args, **kwargs):
        raise AssertionError("No private/service/Q read before capacity admission")

    monkeypatch.setattr(observer, "capacity_state", refuse_capacity)
    monkeypatch.setattr(observer, "read_json", forbidden_read)
    with pytest.raises(observer.EvidenceError, match="capacity"):
        observer.binding_state({"schema_id": "jev-field-syntax-browser-owned-binding-v1"})


def test_capacity_validator_source_drift_is_refused():
    ref = PLAN["v8_capacity"]["validator_source"]
    with pytest.raises(ValueError, match="validator source"):
        capacity.validator(ROOT, {**ref, "sha256": "0" * 64})
