"""AST-only publisher batch contract checks; never execute publisher source."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

PUBLISHER_SHA = "adf7f7aa085674b5682e492d8b2e3cc4f11c67b0d153515c742e907e9b550917"


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def classes(tree):
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


def methods(cls):
    return {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}


def verify_tree(tree):
    cls = classes(tree)
    d3 = methods(cls["D3"])
    encoder = methods(cls["PromptEncoder"])
    default = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "DEFAULT_BATCH_SIZE" for t in node.targets)
    ]
    need(
        len(default) == 1 and ast.literal_eval(default[0].value) == 8,
        "publisher shipped default must stay eight",
    )
    constructors = [
        node
        for node in ast.walk(d3["from_pretrained"])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "cls"
    ]
    need(
        len(constructors) == 1
        and any(
            kw.arg == "batch_size"
            and isinstance(kw.value, ast.Name)
            and kw.value.id == "batch_size"
            for kw in constructors[0].keywords
        ),
        "one publisher constructor must receive native batch argument",
    )
    backbone = [
        node
        for node in ast.walk(d3["__init__"])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_pretrained"
        and isinstance(node.func.value, ast.Call)
        and isinstance(node.func.value.func, ast.Name)
        and node.func.value.func.id == "backbone_class"
    ]
    need(len(backbone) == 1, "one original backbone loader required")
    kwargs = {kw.arg: ast.unparse(kw.value) for kw in backbone[0].keywords}
    need(
        kwargs.get("dtype") == "torch.bfloat16"
        and kwargs.get("attn_implementation") == "'sdpa'"
        and kwargs.get("device_map") == "{'': str(self.device)}",
        "original BF16/SDPA/single-device loader changed",
    )
    loops = [
        node
        for node in ast.walk(d3["run"])
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name)
        and node.iter.func.id == "range"
    ]
    need(
        len(loops) == 1 and ast.unparse(loops[0].iter) == "range(0, len(keys), self.batch_size)",
        "native batch grouping loop changed",
    )
    run = ast.unparse(d3["run"])
    need(
        "keys[start:start + self.batch_size]" in run
        and "prepared.sequences[k] for k in chunk" in run
        and "out.update(zip(chunk, probs))" in run,
        "original question order/full sequences must be preserved",
    )
    logits = ast.unparse(d3["logits"])
    need(
        "len(sequences), width" in logits
        and "width - len(sequence):" in logits
        and "self.encoder(embeds, mask)[:, -1]" in logits
        and "hidden.float() @ self.readout.T" in logits,
        "original batched left padding/last/FP32 readout changed",
    )
    probabilities = ast.unparse(d3["probabilities"])
    need(
        "self.logits(sequences, counts) / self.temperature" in probabilities
        and "self.torch.softmax(scaled, dim=-1)" in probabilities,
        "original legal logits/temperature/softmax changed",
    )
    need(
        "attention_mask=keys if softmax else scan" in ast.unparse(encoder["__call__"]),
        "original full-attention/DeltaNet masks changed",
    )
    respond = ast.unparse(d3["respond"])
    need("for key in prepared.keys:" in respond, "original answer order changed")
    symbols = {
        name: {
            "line": node.lineno,
            "ast_sha256": hashlib.sha256(ast.dump(node).encode()).hexdigest(),
        }
        for name, node in d3.items()
        if name in ("from_pretrained", "__init__", "run", "logits", "probabilities", "respond")
    }
    return {
        "one_constructor": True,
        "one_backbone_loader": True,
        "native_batch_argument": True,
        "shipped_default_batch_unchanged": 8,
        "original_request_question_order": True,
        "original_sequences_context_options": True,
        "left_padding_and_original_masks": True,
        "BF16_backbone_FP32_readout_temperature_softmax_unchanged": True,
        "symbols": symbols,
        "runtime_executed": False,
    }


def verify_file(path):
    path = Path(path)
    need(
        path.suffix == ".py" and path.stat().st_size <= 200_000,
        "small Python publisher source only",
    )
    raw = path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == PUBLISHER_SHA, "publisher source SHA differs")
    return {
        "path": str(path),
        "sha256": PUBLISHER_SHA,
        "size": len(raw),
        **verify_tree(ast.parse(raw, filename=str(path))),
    }
