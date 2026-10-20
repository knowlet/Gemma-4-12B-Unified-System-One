"""Real native grammar + tiny CPU tensors; no model/GPU/download/training."""

import copy
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from jev_cfg_action_loss import (
    _decision_indices,
    action_loss,
    compile_action_plan,
    decision_token_loss,
)

from s1.jev_generated import INSTRUCTIONS, POLICY, format_action_request


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


class TinyTokenizer:
    all_special_ids = [97, 98, 99]
    bos_token_id, eos_token_id, pad_token_id = 98, 99, 97
    operations = {
        "CLICK": [20],
        "DONE": [19],
        "BLOCKED": [21],
        "WAIT": [22],
        "SCROLL_DOWN": [23, 24],
        "SCROLL_UP": [23, 25],
        "TYPE_TEXT": [26],
        "SELECT": [27],
    }

    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        if text == "<turn|>":
            return [99]
        if text.startswith(self.prompt):
            remaining = text[len(self.prompt) :]
            return [90, 91, 92] + (self.encode(remaining) if remaining else [])
        if text.endswith("<turn|>"):
            return self.encode(text.removesuffix("<turn|>")) + [99]
        action = json.loads(text)
        target = [12] if action["target"] is None else [40 + int(c) for c in action["target"]]
        return [10] + self.operations[action["operation"]] + [11] + target + [13]


def compiled(
    *, zero_decisions=False, prompt_format="raw-json-v1", compressible=False, tensor_inputs=True
):
    torch = pytest.importorskip("torch") if tensor_inputs else None
    payload = {
        "model": "generated-actions-none",
        "state": {"goal": "Open a record", "history": []},
        "questions": {
            "operation": {
                "type": "choice",
                "criteria": {"DONE": "Done"}
                if zero_decisions
                else {"CLICK": "Click", "DONE": "Done"},
            }
        },
    }
    if not zero_decisions:
        payload["questions"]["click_target"] = {
            "type": "choice",
            "criteria": {"1": "One", "10": "Ten"},
        }
    if compressible:
        payload["state"]["elements"] = [
            {
                "index": str(index),
                "label": "Open the complete original catalog entry " + str(index),
                "value": "",
                "role": "button",
                "operations": ["CLICK"],
            }
            for index in range(1, 65)
        ]
        payload["questions"]["click_target"]["criteria"] = {
            element["index"]: {
                "element": "[" + element["index"] + "] " + element["label"],
                "current_value": element["value"],
                "role": element["role"],
            }
            for element in payload["state"]["elements"]
        }
    for question in payload["questions"].values():
        question["instructions"] = "Choose one supplied original criterion key."
    target = {
        "operation": "DONE" if zero_decisions else "CLICK",
        "target": None if zero_decisions else "1",
    }
    tok = TinyTokenizer()
    if zero_decisions:
        assert prompt_format == "raw-json-v1"
        user_text = INSTRUCTIONS + "\n" + encode(payload).decode()
    else:
        user_text, _, _, representation = format_action_request(
            payload, payload["model"], prompt_format=prompt_format
        )
    tok.prompt = "<native-user>" + user_text + "<native-model>"
    target_json = encode(target).decode()
    ids = tok.encode(tok.prompt + target_json + "<turn|>")
    targets = tok.encode(target_json) + [99]
    positions = list(range(2, len(ids) - 1))
    tensors = {
        "input_ids": [ids],
        "attention_mask": [[1] * len(ids)],
        "mm_token_type_ids": [[0] * len(ids)],
    }
    record = {
        "case_id": "tiny-approved-case",
        "inference_model_identity": payload["model"],
        "normalized_wire_sha256": hashlib.sha256(encode(payload)).hexdigest(),
        "policy": POLICY,
        "prompt": tok.prompt,
        "prompt_tokens": 3,
        "target_json": target_json,
        "target_token_ids": targets,
        "native_stop_token_id": 99,
        "input_tokens": len(ids),
        "logical_input_tokens": len(ids),
        "logits_positions": positions,
        "processor_tensors": copy.deepcopy(tensors),
    }
    if prompt_format != "raw-json-v1":
        record.update(prompt_format=prompt_format, prompt_representation=representation)
    row = {
        "record": record,
        "inputs": {k: torch.tensor(v) if torch else v for k, v in tensors.items()},
        "positions": torch.tensor(positions) if torch else positions,
        "targets": torch.tensor(targets) if torch else targets,
    }
    return row, payload, tok


def test_plan_uses_complete_original_inventory_and_includes_syntax_looking_branch():
    row, payload, tok = compiled()
    plan = compile_action_plan(row, payload, tok)
    assert plan.legal_action_count == 3 and plan.target_ids == tuple(row["targets"].tolist())
    assert plan.positions == tuple(row["positions"].tolist())
    assert plan.decision_indices == (1, 4)
    assert plan.allowed_ids[1] == (20, 19)  # Preserve offered order, not sorted token IDs.
    assert plan.allowed_ids[4] == (13, 40)  # Complete target1 versus continue to target10.
    assert plan.allowed_ids[-1] == (99,)
    assert plan.request_sha256 == hashlib.sha256(encode(payload)).hexdigest()
    assert plan.evidence()["allowed_ids"] == plan.allowed_ids


@pytest.mark.parametrize("prompt_format", ["shared-elements-v1", "shared-elements-v2"])
def test_shared_compilation_keeps_complete_grammar_and_teacher_path(prompt_format):
    raw, raw_payload, raw_tok = compiled(compressible=True, tensor_inputs=False)
    row, payload, tok = compiled(
        prompt_format=prompt_format, compressible=True, tensor_inputs=False
    )
    raw_plan = compile_action_plan(raw, raw_payload, raw_tok)
    plan = compile_action_plan(row, payload, tok, prompt_format=prompt_format)
    assert row["record"]["prompt_representation"]["format"] == prompt_format
    assert plan.legal_action_count == raw_plan.legal_action_count == 65
    assert plan.canonical_inventory_sha256 == raw_plan.canonical_inventory_sha256
    assert plan.encoded_inventory_sha256 == raw_plan.encoded_inventory_sha256
    assert plan.target_ids == raw_plan.target_ids
    assert plan.allowed_ids == raw_plan.allowed_ids
    assert plan.decision_indices == raw_plan.decision_indices
    assert plan.request_sha256 == raw_plan.request_sha256
    assert plan.compiled_row_sha256 != raw_plan.compiled_row_sha256


def test_explicit_v2_raw_fallback_retains_requested_mode_and_exact_metadata():
    row, payload, tok = compiled(prompt_format="shared-elements-v2", tensor_inputs=False)
    metadata = row["record"]["prompt_representation"]
    assert metadata["format"] == "raw-json-v1"
    assert metadata["requested_format"] == "shared-elements-v2"
    plan = compile_action_plan(row, payload, tok, prompt_format="shared-elements-v2")
    assert plan.legal_action_count == 3
    with pytest.raises(ValueError, match="Historical raw"):
        compile_action_plan(row, payload, tok)


def test_metadata_object_order_is_irrelevant_but_values_are_authenticated():
    row, payload, tok = compiled(prompt_format="shared-elements-v2", tensor_inputs=False)

    def reversed_objects(value):
        if type(value) is dict:
            return {key: reversed_objects(value[key]) for key in reversed(value)}
        if type(value) is list:
            return [reversed_objects(item) for item in value]
        return value

    row["record"]["prompt_representation"] = reversed_objects(
        row["record"]["prompt_representation"]
    )
    assert (
        compile_action_plan(
            row, payload, tok, prompt_format="shared-elements-v2"
        ).legal_action_count
        == 3
    )


@pytest.mark.parametrize("prompt_format", [None, True, 1, [], "unknown", ""])
def test_unknown_or_untyped_trusted_mode_rejects_before_row_work(prompt_format):
    with pytest.raises(ValueError, match="supported native prompt format"):
        compile_action_plan(None, None, None, prompt_format=prompt_format)


@pytest.mark.parametrize(
    "change",
    [
        "missing_mode",
        "wrong_mode",
        "missing_metadata",
        "effective_mode",
        "boolean_as_integer",
        "extra_field",
        "nonfinite",
        "raw_prompt",
    ],
)
def test_shared_forged_mode_metadata_or_prompt_rejects_before_grammar(monkeypatch, change):
    row, payload, tok = compiled(
        prompt_format="shared-elements-v2", compressible=True, tensor_inputs=False
    )
    record = row["record"]
    metadata = record["prompt_representation"]
    if change == "missing_mode":
        del record["prompt_format"]
    elif change == "wrong_mode":
        record["prompt_format"] = "shared-elements-v1"
    elif change == "missing_metadata":
        del record["prompt_representation"]
    elif change == "effective_mode":
        metadata["format"] = "raw-json-v1"
    elif change == "boolean_as_integer":
        metadata["ordered_roundtrip_verified"] = 1
    elif change == "extra_field":
        metadata["unverified"] = True
    elif change == "nonfinite":
        metadata["original_bytes"] = float("nan")
    else:
        raw_text, _, _, _ = format_action_request(payload, payload["model"])
        record["prompt"] = "<native-user>" + raw_text + "<native-model>"

    def forbidden_grammar(*args, **kwargs):
        pytest.fail("Invalid representation reached grammar construction")

    monkeypatch.setattr("jev_cfg_action_loss.LegalActionTokenTrie", forbidden_grammar)
    with pytest.raises(ValueError):
        compile_action_plan(row, payload, tok, prompt_format="shared-elements-v2")


def test_explicit_trusted_mode_cannot_be_inferred_or_replaced_from_record():
    row, payload, tok = compiled(
        prompt_format="shared-elements-v2", compressible=True, tensor_inputs=False
    )
    with pytest.raises(ValueError, match="Historical raw"):
        compile_action_plan(row, payload, tok)
    with pytest.raises(ValueError, match="explicit trusted mode"):
        compile_action_plan(row, payload, tok, prompt_format="shared-elements-v1")
    raw, payload, tok = compiled(tensor_inputs=False)
    with pytest.raises(ValueError, match="explicit trusted mode"):
        compile_action_plan(raw, payload, tok, prompt_format="shared-elements-v2")


@pytest.mark.parametrize("field", ["prompt_format", "prompt_representation"])
def test_legacy_raw_record_rejects_new_shared_fields_even_if_null(field):
    row, payload, tok = compiled(tensor_inputs=False)
    row["record"][field] = None
    with pytest.raises(ValueError, match="Historical raw"):
        compile_action_plan(row, payload, tok)


@pytest.mark.parametrize("mode", ["B", "C"])
def test_actual_cpu_gradients_keep_full_forward_and_every_original_position(mode):
    torch = pytest.importorskip("torch")
    row, payload, tok = compiled()
    plan = compile_action_plan(row, payload, tok)
    logits = torch.linspace(-1, 1, len(plan.target_ids) * 128).reshape(1, -1, 128).requires_grad_()
    calls = []

    def native(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(logits=logits)

    model = SimpleNamespace(device="cpu", lm=native)
    loss, metrics = action_loss(model, row, plan=plan, mode=mode)
    loss.backward()
    assert len(calls) == 1 and calls[0]["use_cache"] is False and calls[0]["return_dict"] is True
    assert torch.equal(calls[0]["logits_to_keep"], row["positions"])
    for key, value in row["inputs"].items():
        assert torch.equal(calls[0][key], value)
    assert calls[0]["input_ids"].shape[-1] == row["record"]["input_tokens"]
    gradient = logits.grad[0]
    forced = set(range(len(plan.target_ids))) - set(plan.decision_indices)
    assert all(torch.count_nonzero(gradient[index]).item() == 0 for index in forced)
    for index in plan.decision_indices:
        allowed = set(plan.allowed_ids[index])
        assert gradient[index, plan.target_ids[index]] < 0
        forbidden = [token for token in range(128) if token not in allowed]
        if mode == "C":
            assert torch.count_nonzero(gradient[index, forbidden]).item() == 0
        else:
            assert torch.all(gradient[index, forbidden] > 0)
    assert (
        metrics["target_tokens"] == len(plan.target_ids) and metrics["decision_target_tokens"] == 2
    )
    assert metrics["forced_singleton_target_tokens"] == len(forced)


def test_c_loss_matches_direct_stable_allowed_ce_and_ignores_forbidden_values():
    torch = pytest.importorskip("torch")
    row, payload, tok = compiled()
    plan = compile_action_plan(row, payload, tok)
    logits = torch.zeros(len(plan.target_ids), 128, requires_grad=True)
    with torch.no_grad():
        logits[1, 20], logits[1, 19] = 2, 3
        logits[4, 13], logits[4, 40] = 4, 2
        logits[:, 100] = 1000
    c, _ = decision_token_loss(logits, plan, mode="C")
    b, _ = decision_token_loss(logits, plan, mode="B")
    expected = (
        torch.logsumexp(torch.tensor([2.0, 3.0]), 0)
        - 2
        + torch.logsumexp(torch.tensor([4.0, 2.0]), 0)
        - 4
    ) / 2
    assert torch.allclose(c, expected) and b > 900
    changed = logits.detach().clone()
    changed[:, 100] = -1000
    assert torch.equal(decision_token_loss(changed, plan, mode="C")[0], c.detach())
    assert decision_token_loss(changed, plan, mode="B")[0] < b


def test_allowed_argmax_tie_matches_native_global_token_id_order():
    torch = pytest.importorskip("torch")
    row, payload, tok = compiled()
    plan = compile_action_plan(row, payload, tok)
    logits = torch.zeros(len(plan.target_ids), 128)
    _, metrics = decision_token_loss(logits, plan, mode="C")
    # Native full-vocabulary masking chooses19 (DONE), although callback order is20,19.
    assert metrics["branch_allowed_correct"] == 1
    assert metrics["teacher_forced_constrained_exact_branches"] is False


@pytest.mark.parametrize("where", [(0, 100), (1, 100), (1, 20)])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("mode", ["B", "C"])
def test_all_logits_nonfinite_refusal_includes_forbidden_and_forced_positions(where, value, mode):
    torch = pytest.importorskip("torch")
    row, payload, tok = compiled()
    plan = compile_action_plan(row, payload, tok)
    logits = torch.zeros(len(plan.target_ids), 128)
    logits[where] = value
    with pytest.raises(ValueError, match="finite"):
        decision_token_loss(logits, plan, mode=mode)


def test_zero_decision_row_refuses_before_any_forward():
    row, payload, tok = compiled(zero_decisions=True, tensor_inputs=False)
    # The existing strict request API refuses a sole operation before grammar construction.
    with pytest.raises(ValueError, match="choice needs 2"):
        compile_action_plan(row, payload, tok)
    # The loss compiler also refuses an all-singleton legal path if admitted by a future API.
    with pytest.raises(ValueError, match="Zero-decision"):
        _decision_indices(((10,), (99,)))


@pytest.mark.parametrize(
    "change",
    ["request", "missing_candidate", "candidate_order", "target", "mask", "positions", "prompt"],
)
def test_full_original_request_and_compiled_boundary_refuse_changed_inputs(change):
    row, payload, tok = compiled()
    if change == "request":
        payload["state"]["goal"] = "Changed goal"
    elif change == "missing_candidate":
        del payload["questions"]["click_target"]["criteria"]["10"]
    elif change == "candidate_order":
        payload["questions"]["click_target"]["criteria"] = {"10": "Ten", "1": "One"}
    elif change == "target":
        row["targets"][1] = 19
    elif change == "mask":
        row["inputs"]["attention_mask"][0, 4] = 0
    elif change == "positions":
        row["positions"][1] += 1
    else:
        row["record"]["prompt"] = "Incomplete request"
    with pytest.raises(ValueError):
        compile_action_plan(row, payload, tok)


def test_illegal_gold_is_not_added_to_candidates_and_mutated_plan_is_rejected():
    torch = pytest.importorskip("torch")
    row, payload, tok = compiled()
    plan = compile_action_plan(row, payload, tok)
    forged = replace(plan, allowed_ids=tuple((target,) for target in plan.target_ids))
    with pytest.raises(ValueError, match="request-bound"):
        decision_token_loss(torch.zeros(len(plan.target_ids), 128), forged, mode="C")
    row["record"]["target_json"] = '{"operation":"CLICK","target":"9"}'
    targets = tok.encode(row["record"]["target_json"]) + [99]
    ids = tok.encode(tok.prompt) + targets
    row["record"]["target_token_ids"] = targets
    row["record"]["processor_tensors"]["input_ids"] = [ids]
    row["inputs"]["input_ids"] = torch.tensor([ids])
    row["targets"] = torch.tensor(targets)
    with pytest.raises(ValueError, match="complete legal"):
        compile_action_plan(row, payload, tok)


def test_plan_reuse_after_row_mutation_refuses_before_forward():
    row, payload, tok = compiled()
    plan = compile_action_plan(row, payload, tok)
    calls = []
    row["inputs"]["input_ids"][0, 0] += 1
    model = SimpleNamespace(device="cpu", lm=lambda **kwargs: calls.append(kwargs))
    with pytest.raises(ValueError, match="changed after"):
        action_loss(model, row, plan=plan, mode="C")
    assert calls == []


@pytest.mark.parametrize("mode", ["B", "C"])
def test_actual_full_candidate_inventory_above_legacy52_limit_is_not_pruned(mode):
    torch = pytest.importorskip("torch")
    row, payload, tok = compiled()
    payload["questions"]["click_target"]["criteria"] = {
        str(i): "Original label " + str(i) for i in range(64)
    }
    tok.prompt = "<native-user>" + INSTRUCTIONS + "\n" + encode(payload).decode() + "<native-model>"
    row["record"]["prompt"] = tok.prompt
    row["record"]["normalized_wire_sha256"] = hashlib.sha256(encode(payload)).hexdigest()
    plan = compile_action_plan(row, payload, tok)
    assert plan.legal_action_count == 65
    loss, metrics = decision_token_loss(torch.zeros(len(plan.target_ids), 128), plan, mode=mode)
    assert torch.isfinite(loss) and metrics["decision_target_tokens"] == 3


def test_explicit_mode_and_complete_vocabulary_bounds_are_required():
    torch = pytest.importorskip("torch")
    row, payload, tok = compiled()
    plan = compile_action_plan(row, payload, tok)
    with pytest.raises(ValueError, match="mode"):
        decision_token_loss(torch.zeros(len(plan.target_ids), 128), plan, mode="A")
    with pytest.raises(ValueError, match="actual vocabulary"):
        decision_token_loss(torch.zeros(len(plan.target_ids), 64), plan, mode="C")


def test_allowed_normalization_remains_stable_under_large_common_logit_offset():
    torch = pytest.importorskip("torch")
    row, payload, tok = compiled()
    plan = compile_action_plan(row, payload, tok)
    plain = torch.zeros(len(plan.target_ids), 128)
    shifted = plain + 100_000_000
    first = decision_token_loss(plain, plan, mode="C")[0]
    second = decision_token_loss(shifted, plan, mode="C")[0]
    assert torch.equal(first, second) and first > 0
