"""Explicit experimental B/C losses for native legal-token branches.

No model is imported or loaded here. This module does not alter the historical
fitter, grammar, deployment policy or qualification contract. Plans must be
compiled from a complete original request and an unchanged compiled row.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from s1.jev_action_grammar import LegalActionTokenTrie

_PLAN_PROOF = object()
_SEQUENCE_KEYS = {
    "input_ids",
    "attention_mask",
    "mm_token_type_ids",
    "token_type_ids",
    "position_ids",
}


def _require(value, message):
    if not value:
        raise ValueError(message)


def _encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def _list(value):
    if callable(getattr(value, "tolist", None)):
        _require(str(getattr(value, "device", "cpu")) == "cpu", "Compiled tensors must remain CPU")
        value = value.tolist()
    _require(type(value) is list, "Compiled sequence must be a list or CPU tensor")
    return value


def _integers(value, name):
    value = _list(value)
    _require(all(type(x) is int and x >= 0 for x in value), name + " must contain integer IDs")
    return value


def _snapshot(row):
    _require(
        type(row) is dict and set(row) == {"record", "inputs", "positions", "targets"},
        "Exact original compiled row fields required",
    )
    _require(
        type(row["record"]) is dict and type(row["inputs"]) is dict, "Compiled row objects required"
    )
    inputs = {key: _list(value) for key, value in row["inputs"].items()}
    _require(
        {"input_ids", "attention_mask"} <= set(inputs) <= _SEQUENCE_KEYS,
        "Unchanged native text sequence inputs required",
    )
    return {
        "record": row["record"],
        "inputs": inputs,
        "positions": _integers(row["positions"], "Positions"),
        "targets": _integers(row["targets"], "Targets"),
    }


@dataclass(frozen=True)
class CFGActionLossPlan:
    """An in-memory verified plan; persisted evidence must be recompiled on load."""

    compiled_row_sha256: str
    request_sha256: str
    target_ids: tuple[int, ...]
    positions: tuple[int, ...]
    allowed_ids: tuple[tuple[int, ...], ...]
    decision_indices: tuple[int, ...]
    canonical_inventory_sha256: str
    encoded_inventory_sha256: str
    legal_action_count: int
    _proof: object = field(repr=False, compare=False)
    _plan_digest: str = field(init=False, default="", repr=False, compare=False)

    def evidence(self):
        return {
            "schema_id": "native-cfg-action-loss-plan-v1",
            "compiled_row_sha256": self.compiled_row_sha256,
            "request_sha256": self.request_sha256,
            "target_ids": self.target_ids,
            "positions": self.positions,
            "allowed_ids": self.allowed_ids,
            "decision_indices": self.decision_indices,
            "canonical_inventory_sha256": self.canonical_inventory_sha256,
            "encoded_inventory_sha256": self.encoded_inventory_sha256,
            "legal_action_count": self.legal_action_count,
            "scope": "all original legal candidates; teacher-forced native token branches",
        }


def _decision_indices(allowed):
    decisions = tuple(index for index, choices in enumerate(allowed) if len(choices) > 1)
    _require(bool(decisions), "Zero-decision row has no legal-branch training objective")
    return decisions


def compile_action_plan(row, payload, tokenizer):
    """Use the actual inference request validator and public grammar callback."""
    from s1.jev_generated import INSTRUCTIONS, POLICY, _request

    snapshot = _snapshot(row)
    record, inputs = snapshot["record"], snapshot["inputs"]
    identity = record.get("inference_model_identity")
    _require(type(identity) is str and bool(identity), "Compiled native model identity required")
    encoded, source, labels = _request(payload, identity)
    _require(
        record.get("normalized_wire_sha256") == hashlib.sha256(source).hexdigest(),
        "Complete original request differs from compiled row",
    )
    _require(record.get("policy") == POLICY, "Native generated-action policy differs")
    prompt, target_json = record.get("prompt"), record.get("target_json")
    _require(
        type(prompt) is str and prompt.count(INSTRUCTIONS + "\n" + encoded) == 1,
        "Complete original request/instructions must remain in the native prompt",
    )
    _require(type(target_json) is str, "Compiled canonical target required")
    p, length = record.get("prompt_tokens"), record.get("input_tokens")
    _require(
        type(p) is int and p >= 1 and type(length) is int and p < length,
        "Native prompt/sequence lengths required",
    )
    _require(1 <= length - p <= 32 and p + 32 <= 8192, "Unchanged native token/context bounds")
    for key, values in inputs.items():
        _require(
            len(values) == 1
            and type(values[0]) is list
            and len(values[0]) == length
            and all(type(value) in (int, bool) for value in values[0]),
            "Native sequence tensor shape/type differs: " + key,
        )
    ids = _integers(inputs["input_ids"][0], "Input IDs")
    targets, positions = snapshot["targets"], snapshot["positions"]
    _require(
        targets == record.get("target_token_ids") == ids[p:]
        and positions == record.get("logits_positions") == list(range(p - 1, length - 1)),
        "Every target token/readout position must remain unchanged",
    )
    _require(
        _encode(inputs) == _encode(record.get("processor_tensors")),
        "Complete native processor tensors differ",
    )
    mask = inputs["attention_mask"][0]
    _require(
        all(value in (0, 1) for value in mask)
        and all(mask[p - 1 :])
        and record.get("logical_input_tokens") == sum(mask),
        "Native attention mask differs",
    )
    _require(not any(inputs.get("mm_token_type_ids", [[0]])[0]), "Text-only compiled row required")
    stop = tokenizer.encode("<turn|>", add_special_tokens=False)
    _require(
        len(stop) == 1 and type(stop[0]) is int and stop[0] == record.get("native_stop_token_id"),
        "Native compiled EOS differs",
    )
    _require(
        tokenizer.encode(prompt, add_special_tokens=False) == ids[:p]
        and tokenizer.encode(target_json, add_special_tokens=False) + stop == targets
        and tokenizer.encode(prompt + target_json + "<turn|>", add_special_tokens=False) == ids,
        "Exact native prefix/target/combined tokenization differs",
    )
    grammar = LegalActionTokenTrie(
        labels, tokenizer, prompt_length=p, eos_token_id=stop[0], max_new_tokens=32
    )
    _require(
        grammar.assert_complete(targets) == target_json, "Gold path is not an original legal action"
    )
    allowed = tuple(tuple(grammar(0, ids[:p] + targets[:index])) for index in range(len(targets)))
    _require(
        all(
            target in choices and len(choices) == len(set(choices))
            for target, choices in zip(targets, allowed, strict=True)
        ),
        "Original teacher path must remain in every actual allowed set",
    )
    decisions = _decision_indices(allowed)
    telemetry = grammar.telemetry()
    plan = CFGActionLossPlan(
        hashlib.sha256(_encode(snapshot)).hexdigest(),
        hashlib.sha256(source).hexdigest(),
        tuple(targets),
        tuple(positions),
        allowed,
        decisions,
        telemetry["canonical_inventory_sha256"],
        hashlib.sha256(_encode(grammar.encoded_actions)).hexdigest(),
        telemetry["legal_action_count"],
        _PLAN_PROOF,
    )

    object.__setattr__(plan, "_plan_digest", hashlib.sha256(_encode(plan.evidence())).hexdigest())
    return plan


def _verified(plan):
    _require(
        type(plan) is CFGActionLossPlan
        and plan._proof is _PLAN_PROOF
        and plan._plan_digest == hashlib.sha256(_encode(plan.evidence())).hexdigest(),
        "An actual request-bound compiled CFG plan is required",
    )


def decision_token_loss(logits, plan, *, mode):
    """B: full vocabulary on branches. C: actual allowed vocabulary on branches."""
    import torch
    import torch.nn.functional as functional

    _verified(plan)
    _require(mode in ("B", "C"), "Explicit experimental loss mode B or C required")
    _require(
        torch.is_tensor(logits)
        and logits.is_floating_point()
        and logits.ndim == 2
        and logits.shape[0] == len(plan.target_ids)
        and logits.shape[1] > 0
        and bool(torch.isfinite(logits).all()),
        "All native target logits must be finite with unchanged shape",
    )
    float_logits = logits.float()
    _require(bool(torch.isfinite(float_logits).all()), "All FP32 target logits must remain finite")
    vocabulary = logits.shape[1]
    _require(
        all(0 <= token < vocabulary for choices in plan.allowed_ids for token in choices),
        "Original allowed token ID exceeds actual vocabulary",
    )
    targets = torch.tensor(plan.target_ids, dtype=torch.long, device=logits.device)
    decisions = torch.tensor(plan.decision_indices, dtype=torch.long, device=logits.device)
    selected = float_logits.index_select(0, decisions)
    chosen = targets.index_select(0, decisions)
    conditional, correct = [], []
    for index in plan.decision_indices:
        choices = plan.allowed_ids[index]
        allowed = torch.tensor(choices, dtype=torch.long, device=logits.device)
        values = float_logits[index].index_select(0, allowed)
        local_target = choices.index(plan.target_ids[index])
        local_label = torch.tensor([local_target], dtype=torch.long, device=logits.device)
        conditional.append(functional.cross_entropy(values.unsqueeze(0), local_label))
        # Native masked full-vocabulary argmax breaks ties by the lowest global ID.
        winners = allowed[values.detach() == values.detach().max()]
        correct.append(winners.min().item() == plan.target_ids[index])
    cfg_loss = torch.stack(conditional).mean()
    loss = functional.cross_entropy(selected, chosen) if mode == "B" else cfg_loss
    _require(bool(torch.isfinite(loss)), "Native CFG branch loss is nonfinite")
    with torch.no_grad():
        full_correct = logits.argmax(-1).eq(targets)
        metrics = {
            "training_objective": "native-cfg-branch-" + mode + "-mean-v1",
            "target_tokens": len(plan.target_ids),
            "correct_target_tokens": int(full_correct.sum()),
            "teacher_forced_exact_tokens": bool(full_correct.all()),
            "decision_target_tokens": len(plan.decision_indices),
            "forced_singleton_target_tokens": len(plan.target_ids) - len(plan.decision_indices),
            "branch_full_vocab_correct": int(selected.argmax(-1).eq(chosen).sum()),
            "branch_allowed_correct": sum(correct),
            "teacher_forced_constrained_exact_branches": all(correct),
            "full_vocab_ce_mean": float(functional.cross_entropy(float_logits, targets)),
            "allowed_branch_ce_mean": float(cfg_loss.detach()),
            "metrics_scope": "teacher-forced diagnostics; not actual generation/browser success",
        }
    return loss, metrics


def action_loss(model, row, *, plan, mode):
    """One original full causal forward and every original target readout position."""
    _verified(plan)
    snapshot = _snapshot(row)
    _require(
        hashlib.sha256(_encode(snapshot)).hexdigest() == plan.compiled_row_sha256,
        "Compiled request/inputs/target/readout positions changed after plan verification",
    )
    _require(mode in ("B", "C"), "Explicit experimental loss mode B or C required")
    inputs = {key: value.to(model.device) for key, value in row["inputs"].items()}
    output = model.lm(
        **inputs,
        logits_to_keep=row["positions"].to(model.device),
        use_cache=False,
        return_dict=True,
    )
    _require(
        output.logits.ndim == 3 and output.logits.shape[0] == 1,
        "Exactly one unchanged native teacher-forced forward required",
    )
    return decision_token_loss(output.logits[0], plan, mode=mode)
