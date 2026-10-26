"""Production compiler→CFG contract with lightweight CPU sequence stand-ins.

These tests exercise formatting and token boundaries, not a real processor,
model, admission receipt, optimizer, or browser task.
"""

import builtins
import copy
import json
import re
import sys
from types import SimpleNamespace

import jev_browser_fit_actions as FIT
import numpy as np
import pytest
from jev_cfg_action_loss import compile_action_plan

from s1.browser_public_training import NativeWireRequest
from s1.jev_generated import INSTRUCTIONS, format_action_request

RAW = "raw-json-v1"
V2 = "shared-elements-v2"


def wire(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class CPUDevice:
    type = "cpu"

    def __str__(self):
        return "cpu"


class Tensor:
    device = CPUDevice()

    def __init__(self, values, *, dtype=np.int64):
        self.values = np.asarray(values, dtype=dtype)

    @property
    def shape(self):
        return self.values.shape

    @property
    def ndim(self):
        return self.values.ndim

    @property
    def dtype(self):
        return self.values.dtype

    def __getitem__(self, index):
        value = self.values[index]
        return Tensor(value, dtype=value.dtype)

    def __eq__(self, other):
        return Tensor(self.values == other, dtype=np.bool_)

    def __or__(self, other):
        return Tensor(self.values | other.values, dtype=np.bool_)

    def clone(self):
        return Tensor(self.values.copy(), dtype=self.dtype)

    def tolist(self):
        return self.values.tolist()

    def item(self):
        return self.values.item()

    def all(self):
        return bool(self.values.all())

    def any(self):
        return bool(self.values.any())

    def sum(self):
        return self.values.sum()


@pytest.fixture
def cpu_sequences(monkeypatch):
    fake = SimpleNamespace(
        int64=np.int64,
        int32=np.int32,
        bool=np.bool_,
        long=np.int64,
        is_tensor=lambda value: isinstance(value, Tensor),
        equal=lambda left, right: np.array_equal(left.values, right.values),
        all=lambda value: value.all(),
        arange=lambda start, end, dtype: Tensor(np.arange(start, end, dtype=dtype)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)


class Tokenizer:
    pad_token_id, bos_token_id, eos_token_id = 0, 1, 2
    all_special_ids = [0, 1, 2]
    unk_token_id = -1
    pattern = re.compile(r"<turn\|>|<assistant>|<user>|\w+|\s+|[^\w\s]")

    def __init__(self):
        self.vocab = {"<turn|>": self.eos_token_id}
        # The same tokenizer must assign action tokens identically regardless
        # of which representation is encoded first.
        for operation in ("CLICK", "DONE"):
            targets = [str(index) for index in range(1, 65)] if operation == "CLICK" else [None]
            for target in targets:
                self.encode(wire({"operation": operation, "target": target}))

    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return [
            self.vocab.setdefault(piece, len(self.vocab) + 3)
            for piece in self.pattern.findall(text)
        ]

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)


class Processor:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.messages = []
        self.encoded = []
        self.mutate_prefix = None
        self.mask_target = False

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        self.messages.append(copy.deepcopy(messages))
        return "<user>" + messages[0]["content"][0]["text"] + "<assistant>"

    def __call__(self, *, text, return_tensors, add_special_tokens):
        assert return_tensors == "pt" and add_special_tokens is False
        ids = self.tokenizer.encode(text)
        values = {
            "input_ids": Tensor([ids]),
            "attention_mask": Tensor([[0] + [1] * (len(ids) - 1)]),
            "mm_token_type_ids": Tensor([[0] * len(ids)]),
        }
        if text.endswith("<turn|>"):
            if self.mutate_prefix:
                values[self.mutate_prefix].values[0, 1] += 1
            if self.mask_target:
                values["attention_mask"].values[0, -1] = 0
        self.encoded.append((text, values))
        return values


def model():
    processor = Processor()
    return SimpleNamespace(
        tok=processor.tokenizer,
        processor=processor,
        max_context=8192,
        lm=SimpleNamespace(
            config=SimpleNamespace(), generation_config=SimpleNamespace(eos_token_id=[2])
        ),
    )


def example(*, targets=0):
    elements = [
        {
            "index": str(index),
            "label": "Open the complete original catalog entry " + str(index),
            "value": "",
            "role": "button",
            "operations": ["CLICK"],
        }
        for index in range(1, targets + 1)
    ]
    criteria = (
        {
            element["index"]: {
                "element": "[" + element["index"] + "] " + element["label"],
                "current_value": element["value"],
                "role": element["role"],
            }
            for element in elements
        }
        if elements
        else {"1": "Open record"}
    )
    payload = {
        "model": "original-training-alias",
        "state": {"page": {"text": "Catalog records"}, "elements": elements, "recent_actions": []},
        "questions": {
            "operation": {
                "instructions": "Choose one useful operation.",
                "criteria": {"CLICK": "Open an entry", "DONE": "Complete"},
            },
            "click_target": {
                "instructions": "Choose the requested observed entry.",
                "criteria": criteria,
            },
        },
    }
    case = SimpleNamespace(
        id="synthetic-prompt-contract",
        split="train",
        group_id="synthetic-prompt-contract-family",
        gold={"operation": "CLICK", "click_target": "1"},
        request=NativeWireRequest.model_validate(payload),
    )
    entry = {
        "case_id": case.id,
        "payload": payload,
        "teacher_choice": {"operation": "CLICK", "target": "1"},
        "valid_next_targets": {"CLICK": ["1"]},
    }
    return case, entry


def compile_one(native, case, entry, *, prompt_format=RAW):
    return FIT.compile_actions(
        native,
        [case],
        {"blobs": {"wire_requests.jsonl": wire(entry).encode() + b"\n"}},
        identity=entry["payload"]["model"],
        prompt_format=prompt_format,
    )[0]


def test_default_raw_keeps_literal_prefix_record_and_every_sequence_boundary(cpu_sequences):
    native = model()
    case, entry = example()
    row = compile_one(native, case, entry)
    record = row["record"]
    text = native.processor.messages[0][0]["content"][0]["text"]
    assert text == INSTRUCTIONS + "\n" + wire(entry["payload"])
    assert "teacher_choice" not in text and "valid_next_targets" not in text
    assert "prompt_format" not in record and "prompt_representation" not in record
    p, length = record["prompt_tokens"], record["input_tokens"]
    for name, tensor in row["inputs"].items():
        prefix = native.processor.encoded[0][1][name]
        assert tensor.shape == (1, length)
        assert tensor[:, :p].tolist() == prefix.tolist()
        assert record["processor_tensors"][name] == tensor.tolist()
    assert row["inputs"]["attention_mask"][0, 0].item() == 0
    assert row["positions"].tolist() == list(range(p - 1, length - 1))
    assert row["targets"].tolist() == record["target_token_ids"]
    assert compile_action_plan(row, entry["payload"], native.tok).legal_action_count == 2


@pytest.mark.parametrize("targets", [24, 64])
def test_selected_v2_compiler_cfg_inventory_targets_and_shifted_positions(cpu_sequences, targets):
    native = model()
    case, entry = example(targets=targets)
    original_wire = wire(entry)
    raw = compile_one(native, case, entry)
    shared = compile_one(native, case, entry, prompt_format=V2)
    raw_plan = compile_action_plan(raw, entry["payload"], native.tok)
    plan = compile_action_plan(shared, entry["payload"], native.tok, prompt_format=V2)
    assert wire(entry) == original_wire
    assert shared["record"]["prompt_representation"]["format"] == V2
    assert shared["record"]["prompt_format"] == V2
    assert plan.legal_action_count == raw_plan.legal_action_count == targets + 1
    assert plan.canonical_inventory_sha256 == raw_plan.canonical_inventory_sha256
    assert plan.encoded_inventory_sha256 == raw_plan.encoded_inventory_sha256
    assert plan.request_sha256 == raw_plan.request_sha256
    assert plan.target_ids == raw_plan.target_ids
    assert plan.allowed_ids == raw_plan.allowed_ids
    assert plan.decision_indices == raw_plan.decision_indices
    difference = raw["record"]["prompt_tokens"] - shared["record"]["prompt_tokens"]
    assert difference > 0
    assert all(
        old - new == difference for old, new in zip(raw_plan.positions, plan.positions, strict=True)
    )
    expected, _, _, metadata = format_action_request(
        entry["payload"], entry["payload"]["model"], prompt_format=V2
    )
    assert native.processor.messages[1][0]["content"][0]["text"] == expected
    assert shared["record"]["prompt_representation"] == metadata
    with pytest.raises(ValueError, match="Historical raw"):
        compile_action_plan(shared, entry["payload"], native.tok)
    with pytest.raises(ValueError, match="explicit trusted mode"):
        compile_action_plan(
            shared, entry["payload"], native.tok, prompt_format="shared-elements-v1"
        )


def test_v2_raw_fallback_keeps_full_raw_prefix_and_requested_mode(cpu_sequences):
    native = model()
    case, entry = example()
    raw = compile_one(native, case, entry)
    shared = compile_one(native, case, entry, prompt_format=V2)
    assert shared["record"]["prompt_format"] == V2
    assert shared["record"]["prompt_representation"]["format"] == RAW
    assert shared["record"]["prompt_representation"]["requested_format"] == V2
    assert shared["record"]["prompt"] == raw["record"]["prompt"]
    assert shared["record"]["processor_tensors"] == raw["record"]["processor_tensors"]
    assert shared["targets"].tolist() == raw["targets"].tolist()
    assert shared["positions"].tolist() == raw["positions"].tolist()
    assert (
        compile_action_plan(
            shared, entry["payload"], native.tok, prompt_format=V2
        ).legal_action_count
        == 2
    )
    with pytest.raises(ValueError, match="Historical raw"):
        compile_action_plan(shared, entry["payload"], native.tok)


@pytest.mark.parametrize("prompt_format", [None, True, 1, [], "unknown"])
def test_unknown_mode_refuses_before_importing_tensor_library(monkeypatch, prompt_format):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch":
            pytest.fail("Invalid prompt mode imported the tensor library")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(ValueError, match="supported action-training prompt format"):
        FIT.compile_actions(None, [], None, prompt_format=prompt_format)


@pytest.mark.parametrize("name", ["input_ids", "attention_mask", "mm_token_type_ids"])
def test_v2_still_rejects_any_changed_prefix_tensor(cpu_sequences, name):
    native = model()
    native.processor.mutate_prefix = name
    case, entry = example(targets=24)
    with pytest.raises(ValueError, match="prefix changed"):
        compile_one(native, case, entry, prompt_format=V2)


def test_v2_still_rejects_masked_target(cpu_sequences):
    native = model()
    native.processor.mask_target = True
    case, entry = example(targets=24)
    with pytest.raises(ValueError, match="must be unmasked"):
        compile_one(native, case, entry, prompt_format=V2)


@pytest.mark.parametrize("prompt_format", [RAW, V2])
def test_both_formats_preserve_full_completion_reserve_without_truncation(
    cpu_sequences, prompt_format
):
    native = model()
    case, entry = example(targets=24)
    row = compile_one(native, case, entry, prompt_format=prompt_format)
    native.max_context = row["record"]["prompt_tokens"] + 31
    with pytest.raises(ValueError, match="nothing truncated"):
        compile_one(native, case, entry, prompt_format=prompt_format)
