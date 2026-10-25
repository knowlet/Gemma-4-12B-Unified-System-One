"""Public action DTO contracts, without a claim of genuine collection proofs."""

import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from s1.browser_public_training import (
    PROOF_FIELDS,
    NativeWireRequest,
    PublicNativeActionCase,
    public_training_case,
)
from s1.evaluation.datasets import request_fingerprint

ROOT = Path(__file__).resolve().parents[1]


def payload(width=3):
    goal = "Open the requested reference and bring its Syntax section into view."
    questions = {
        "operation": {
            "type": "choice",
            "criteria": {
                "CLICK": "Click",
                "TYPE_TEXT": "Fill",
                "SELECT": "Select",
                "SCROLL_DOWN": "Scroll down",
                "SCROLL_UP": "Scroll up",
                "WAIT": "Wait",
                "DONE": "Goal satisfied",
                "BLOCKED": "No supported progress",
            },
            "instructions": {"goal": goal, "rules": ["Use the current page"]},
        }
    }
    for head, operation, labels in (
        ("click_target", "CLICK", [str(i) for i in range(width)]),
        ("type_text_target", "TYPE_TEXT", ["field"]),
        ("select_target", "SELECT", ["4:0", "4:1"]),
    ):
        questions[head] = {
            "type": "choice",
            "criteria": {label: {"label": "Reference " + label} for label in labels},
            "instructions": {"goal": goal, "operation": operation},
        }
    return {
        "model": "scripted-public-native",
        "state": {"page": {"title": "Reference", "text": "Index · Syntax"}, "history": []},
        "questions": questions,
    }


def proof_refs():
    # These files do not exist: DTO validation must not claim to authenticate
    # them, execute a judge, open a browser or make a case eligible for fitting.
    return {
        name: {"artifact_path": f"episode/{name}.jsonl", "sha256": "a" * 64, "sequence": 0}
        for name in sorted(PROOF_FIELDS)
    }


def case(wire=None, *, operation="CLICK", target="1", **updates):
    arguments = {
        "case_id": "train-public-episode-step-000",
        "group_id": "public-native-site:official-docs",
        "task_id": "reference-syntax",
        "split": "train",
        "selected": {"operation": operation, "target": target},
        "valid_targets": {operation: [target] if target is not None else []},
        "wire_row_id": "wire-000",
        "episode_id": "episode-000",
        "pair_id": "pair-000",
        "state_role": "ordinary_complete" if operation == "DONE" else "ordinary_incomplete",
        "proof_refs": proof_refs(),
    }
    arguments.update(updates)
    return public_training_case(payload() if wire is None else wire, **arguments)


@pytest.mark.parametrize("width", [53, 400])
def test_full_public_width_and_unused_heads_preserved(width):
    wire = payload(width)
    row = case(wire)
    assert row.request.model_dump(mode="json") == wire
    assert list(row.request.model_dump()["questions"]["click_target"]["criteria"]) == [
        str(i) for i in range(width)
    ]
    assert set(row.request.model_dump()["questions"]) == set(wire["questions"])
    assert row.gold == {"operation": "CLICK", "click_target": "1"}
    assert isinstance(request_fingerprint(row.request), str)


def test_401_candidates_fail_instead_of_pruning():
    with pytest.raises(ValidationError, match="candidate limit"):
        case(payload(401))


def test_original_order_and_wire_seal_protect_candidate_assignment():
    wire = payload(53)
    ordinary = NativeWireRequest.model_validate(wire)
    reversed_wire = copy.deepcopy(wire)
    criteria = reversed_wire["questions"]["click_target"]["criteria"]
    reversed_wire["questions"]["click_target"]["criteria"] = dict(reversed(list(criteria.items())))
    reordered = NativeWireRequest.model_validate(reversed_wire)
    assert ordinary.wire_sha256 != reordered.wire_sha256
    assert (
        ordinary.wire_bytes
        == json.dumps(wire, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    )
    ordinary.root["questions"]["click_target"]["criteria"] = dict(reversed(criteria.items()))
    with pytest.raises(ValueError, match="changed"):
        ordinary.candidate_labels
    with pytest.raises(ValueError, match="changed"):
        ordinary.model_dump()


def test_input_and_serialization_are_deeply_detached():
    wire, refs, accepted = payload(), proof_refs(), {"CLICK": ["1", "2"]}
    row = case(wire, proof_refs=refs, valid_targets=accepted)
    original = row.model_dump(mode="json")
    wire["state"]["page"]["title"] = "MUTATED"
    refs["raw_observation"]["artifact_path"] = "changed.json"
    accepted["CLICK"].append("not-offered")
    dumped = row.model_dump(mode="json")
    dumped["request"]["questions"]["click_target"]["criteria"]["1"]["label"] = "MUTATED"
    dumped["valid_next_targets"]["CLICK"].append("not-offered")
    assert row.model_dump(mode="json") == original
    labels = row.request.candidate_labels
    labels["click_target"].clear()
    assert row.request.candidate_labels["click_target"] == ["0", "1", "2"]


def test_nested_annotation_mutation_is_detected_on_serialization():
    row = case()
    row.gold["click_target"] = "2"
    with pytest.raises(ValueError, match="changed"):
        row.model_dump_json()


def test_mutated_request_instance_cannot_be_revalidated_or_embedded():
    wire = NativeWireRequest.model_validate(payload())
    wire.root["state"]["page"]["title"] = "MUTATED"
    with pytest.raises(ValueError, match="changed"):
        NativeWireRequest.model_validate(wire)
    with pytest.raises(ValueError, match="changed"):
        case(wire)


@pytest.mark.parametrize("operation", ["SCROLL_DOWN", "SCROLL_UP", "WAIT", "DONE"])
def test_targetless_operations_have_only_operation_gold(operation):
    row = case(operation=operation, target=None)
    assert row.gold == {"operation": operation}
    assert row.valid_next_targets == {operation: []}
    assert set(row.request.model_dump()["questions"]) == set(payload()["questions"])


@pytest.mark.parametrize(
    ("operation", "target", "head"),
    [
        ("CLICK", "1", "click_target"),
        ("TYPE_TEXT", "field", "type_text_target"),
        ("SELECT", "4:1", "select_target"),
    ],
)
def test_active_target_is_the_exact_observed_key(operation, target, head):
    assert case(operation=operation, target=target).gold == {"operation": operation, head: target}


@pytest.mark.parametrize(
    ("operation", "target"),
    [
        ("SCROLL_DOWN", "1"),
        ("DONE", "1"),
        ("CLICK", 1),
        ("CLICK", None),
        ("SELECT", "4:99"),
        ("ENTER", None),
        ("BACK", None),
        ("BLOCKED", None),
    ],
)
def test_invalid_or_unoffered_selected_actions_fail(operation, target):
    with pytest.raises(ValueError):
        case(operation=operation, target=target)


@pytest.mark.parametrize(
    "accepted",
    [
        {},
        {"TYPE_TEXT": ["field"]},
        {"CLICK": []},
        {"CLICK": ["1", "1"]},
        {"CLICK": ["not-offered"]},
        {"CLICK": [1]},
        {"CLICK": ["1"], "WAIT": ["x"]},
        {"CLICK": ["1"], "BLOCKED": []},
        {"CLICK": ["1"], "DONE": []},
    ],
)
def test_invalid_accepted_action_sets_fail(accepted):
    with pytest.raises(ValueError):
        case(valid_targets=accepted)


def test_complete_role_and_done_annotations_are_consistent_only():
    with pytest.raises(ValueError, match="complete role"):
        case(operation="DONE", target=None, state_role="ordinary_incomplete")
    with pytest.raises(ValueError, match="complete role"):
        case(state_role="ordinary_complete")
    with pytest.raises(ValueError, match="continuing action"):
        case(operation="DONE", target=None, valid_targets={"DONE": [], "CLICK": ["1"]})
    # A valid reference shape proves no file content or DOM fact at this stage.
    assert case(operation="DONE", target=None).state_role == "ordinary_complete"


def test_no_arbitrary_unused_gold_head():
    data = case().model_dump(mode="json")
    data["gold"]["select_target"] = "4:1"
    with pytest.raises(ValidationError, match="active target"):
        PublicNativeActionCase.model_validate(data)


@pytest.mark.parametrize("field", ["private_dom", "teacher_choice", "fit_eligible", "fixture_path"])
def test_unknown_metadata_and_private_payload_outer_fields_fail(field):
    data = case().model_dump(mode="json")
    data[field] = "PRIVATE"
    with pytest.raises(ValidationError):
        PublicNativeActionCase.model_validate(data)
    wire = payload()
    wire[field] = "PRIVATE"
    with pytest.raises(ValueError):
        NativeWireRequest.model_validate(wire)


def test_private_question_outer_field_fails_but_page_text_is_preserved():
    wire = payload()
    wire["state"]["page"]["text"] = "Page content may literally say gold, teacher or private_dom."
    assert NativeWireRequest.model_validate(wire).model_dump() == wire
    wire["questions"]["operation"]["private_dom"] = {"success": True}
    with pytest.raises(ValueError):
        NativeWireRequest.model_validate(wire)


@pytest.mark.parametrize("value", [True, "1", 2])
def test_schema_version_is_not_coerced(value):
    data = case().model_dump(mode="json")
    data["schema_version"] = value
    with pytest.raises(ValueError):
        PublicNativeActionCase.model_validate(data)


def test_wrong_schema_split_and_fixture_group_fail():
    for key, value in [
        ("schema_id", "jev-development-supervision-v1"),
        ("split", "test"),
        ("group_id", "browser-development:" + "a" * 64),
    ]:
        data = case().model_dump(mode="json")
        data[key] = value
        with pytest.raises(ValueError):
            PublicNativeActionCase.model_validate(data)


@pytest.mark.parametrize(
    "path",
    [
        "../private.json",
        "/tmp/proof.json",
        "x\\proof.json",
        "x//proof.json",
        "./proof.json",
        "x\nproof.json",
    ],
)
def test_external_proof_paths_are_confined_but_not_authenticated(path):
    refs = proof_refs()
    refs["raw_observation"]["artifact_path"] = path
    with pytest.raises(ValueError):
        case(proof_refs=refs)


def test_proof_inventory_and_fields_are_exact():
    refs = proof_refs()
    refs.pop("ownership")
    with pytest.raises(ValueError):
        case(proof_refs=refs)
    refs = proof_refs()
    refs["ownership"]["verified"] = True
    with pytest.raises(ValueError):
        case(proof_refs=refs)


def test_selected_action_object_is_exact_and_not_mutated():
    with pytest.raises(ValueError):
        case(selected={"operation": "CLICK", "target": "1", "thought": "teacher"})


def test_roundtrip_retains_full_wire_order_without_oracle_fields():
    row = case(payload(53))
    restored = PublicNativeActionCase.model_validate_json(row.model_dump_json())
    assert restored.request.wire_bytes == row.request.wire_bytes
    assert list(restored.request.model_dump()["questions"]["click_target"]["criteria"]) == [
        str(i) for i in range(53)
    ]
    assert "proof_refs" not in restored.request.model_dump()
    assert "gold" not in restored.request.model_dump()
    assert "valid_next_targets" not in restored.request.model_dump()


@pytest.mark.parametrize("duplicate", ["model", "criteria"])
def test_native_json_duplicate_keys_fail_without_losing_the_original_order(duplicate):
    raw = json.dumps(payload(), ensure_ascii=False, separators=(",", ":"))
    needle = '"model":"scripted-public-native"' if duplicate == "model" else '"criteria":{'
    replacement = (
        '"model":"different","model":"scripted-public-native"'
        if duplicate == "model"
        else '"criteria":{},"criteria":{'
    )
    with pytest.raises(ValueError, match="duplicate"):
        NativeWireRequest.model_validate_json(raw.replace(needle, replacement, 1))


def test_case_json_duplicate_metadata_and_nonfinite_native_values_fail():
    raw = case().model_dump_json()
    with pytest.raises(ValueError, match="duplicate"):
        PublicNativeActionCase.model_validate_json(
            raw.replace('"split":', '"split":"train","split":', 1)
        )
    with pytest.raises(ValueError, match="finite"):
        NativeWireRequest.model_validate_json(
            json.dumps(payload()).replace('"history": []', '"history": [NaN]')
        )


def test_existing_case_and_model_copy_do_not_reseal_mutated_annotations():
    row = case()
    changed = row.model_copy(update={"gold": {"operation": "CLICK", "click_target": "2"}})
    with pytest.raises(ValueError, match="changed"):
        changed.model_dump()
    row.valid_next_targets["CLICK"].append("2")
    with pytest.raises(ValueError, match="changed"):
        PublicNativeActionCase.model_validate(row)


@pytest.mark.inference
def test_unchanged_compile_actions_with_genuine_cached_processor_cpu_only():
    if os.environ.get("JEV_PUBLIC_DTO_NATIVE_CPU") != "1":
        pytest.skip("explicit CPU-only cached native processor check")
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    assert not torch.cuda.is_initialized()
    snapshot = (
        ROOT
        / "artifacts/jev-local-hf/hub/models--knowlet--Gemma-4-12B-Unified-System-One/snapshots/a66f836b56605039fe040f330180e336d19b3362"
    )
    if not snapshot.is_dir():
        pytest.skip("pinned local native processor is unavailable")
    processor = transformers.AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    script = ROOT / "scripts/jev_browser_fit_actions.py"
    before = script.read_bytes()
    spec = importlib.util.spec_from_file_location("public_dto_actual_action_compiler", script)
    fit = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        spec.loader.exec_module(fit)
    finally:
        sys.path.pop(0)
    model = SimpleNamespace(
        tok=processor.tokenizer,
        processor=processor,
        lm=SimpleNamespace(
            generation_config=SimpleNamespace(eos_token_id=[106]),
            config=SimpleNamespace(),
        ),
        max_context=8192,
    )
    rows, wires = [], []
    for operation, target, width in [
        ("CLICK", "1", 53),
        ("CLICK", "399", 400),
        ("SCROLL_DOWN", None, 3),
        ("SCROLL_UP", None, 3),
        ("WAIT", None, 3),
        ("DONE", None, 3),
    ]:
        wire = payload(width)
        # 400 verbose full targets exceed8192; compact genuine-shaped criteria
        # remain complete and ordered so this is a compiler width check.
        wire["questions"]["click_target"]["criteria"] = {str(i): "x" for i in range(width)}
        row = case(wire, operation=operation, target=target, case_id=f"dto-{operation}-{width}")
        rows.append(row)
        wires.append(
            {
                "case_id": row.id,
                "payload": wire,
                "teacher_choice": {"operation": operation, "target": target},
                "valid_next_targets": row.valid_next_targets,
            }
        )
    captured = {
        "blobs": {
            "wire_requests.jsonl": b"\n".join(
                json.dumps(wire, ensure_ascii=False, separators=(",", ":")).encode()
                for wire in wires
            )
        }
    }
    compiled = fit.compile_actions(model, rows, captured)
    assert len(compiled) == 6
    for row, native, wire in zip(compiled, rows, wires, strict=True):
        record = row["record"]
        assert record["native_stop_token_id"] == 106
        assert record["target_token_ids"][-1] == 106
        assert 1 <= len(record["target_token_ids"]) <= 32
        assert record["prompt_tokens"] + 32 <= 8192
        assert json.loads(record["target_json"]) == wire["teacher_choice"]
        assert "private_precondition" not in record["prompt"]
        assert "proof_refs" not in record["prompt"]
        assert "valid_next_targets" not in record["prompt"]
        assert record["request_fingerprint"] == request_fingerprint(native.request)
    assert script.read_bytes() == before
    assert not torch.cuda.is_initialized()
