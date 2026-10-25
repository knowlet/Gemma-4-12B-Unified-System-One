"""Invented stdlib controls for the held GPU operation-only composition.

These source fakes import no Torch/NumPy/model and read no
training, request, plan, array, qualification, or owner-output data. Tiny fake
arithmetic proves control contracts, never real CUDA numerical qualification.
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import json
import math
import struct
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parents[1] / "scripts"


def source():
    name = "_held_gpu_operation_backend_controls_v3"
    spec = importlib.util.spec_from_file_location(name, HERE / "jev_decision3_operation_gpu.py")
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def nested_shape(values):
    if not isinstance(values, list):
        return ()
    if not values:
        return (0,)
    suffix = nested_shape(values[0])
    if any(nested_shape(value) != suffix for value in values):
        raise AssertionError("ragged invented tensor")
    return (len(values), *suffix)


def flatten(values):
    if isinstance(values, list):
        return [item for value in values for item in flatten(value)]
    return [values]


def indexed(values, indices):
    if not indices:
        return values
    first, *remaining = indices
    if first is None:
        return [indexed(values, remaining)]
    if isinstance(first, slice):
        return [indexed(value, remaining) for value in values[first]]
    return indexed(values[first], remaining)


class Device:
    def __init__(self, value):
        self.value = str(value)
        self.type = self.value.partition(":")[0]
        self.index = int(self.value.partition(":")[2]) if ":" in self.value else None

    def __str__(self):
        return self.value

    def __eq__(self, other):
        return self.value == str(other)


class Flag:
    def __init__(self, value):
        self.value = bool(value)

    def all(self):
        return self

    def item(self):
        return self.value

    def __bool__(self):
        return self.value


class Tensor:
    """Small eager fake, plus declared shapes for non-numerical wrapper controls."""

    matmul_calls = []

    def __init__(self, values, *, device="cuda:0", dtype="float32", shape=None):
        self.values = copy.deepcopy(values)
        self.shape = tuple(shape) if shape is not None else nested_shape(self.values)
        self.device, self.dtype = Device(device), dtype
        self.requires_grad, self._version = False, 0

    def data_ptr(self):
        return id(self.values)

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return {"float32": 4, "bfloat16": 2, "uint8": 1}[self.dtype]

    def is_contiguous(self):
        return True

    def contiguous(self):
        return self

    def detach(self):
        result = Tensor.__new__(Tensor)
        result.__dict__ = dict(self.__dict__)
        return result

    def clone(self):
        return Tensor(self.values, device=str(self.device), dtype=self.dtype, shape=self.shape)

    def to(self, *, device=None, dtype=None):
        selected_device = str(self.device) if device is None else str(device)
        selected_dtype = self.dtype if dtype is None else dtype
        if selected_device == str(self.device) and selected_dtype == self.dtype:
            return self
        return Tensor(self.values, device=selected_device, dtype=selected_dtype, shape=self.shape)

    def float(self):
        return self.to(dtype="float32")

    def cpu(self):
        return self.to(device="cpu")

    def tolist(self):
        return copy.deepcopy(self.values)

    @property
    def T(self):
        if len(self.shape) != 2:
            raise AssertionError("invented transpose requires matrix")
        return Tensor(
            [list(row) for row in zip(*self.values)], device=str(self.device), dtype=self.dtype
        )

    def __getitem__(self, key):
        indices = key if isinstance(key, tuple) else (key,)
        return Tensor(indexed(self.values, indices), device=str(self.device), dtype=self.dtype)

    def __setitem__(self, key, value):
        if not isinstance(key, slice):
            raise AssertionError("only invented row slices are assigned")
        self.values[key] = copy.deepcopy(value.values)
        self._version += 1

    def __matmul__(self, other):
        if str(self.device) != str(other.device) or self.shape[1] != other.shape[0]:
            raise AssertionError("invented matmul device/shape mismatch")
        self.matmul_calls.append((self.shape, other.shape))
        columns = list(zip(*other.values))
        values = [
            [sum(a * b for a, b in zip(row, column, strict=True)) for column in columns]
            for row in self.values
        ]
        return Tensor(values, device=str(self.device), dtype=self.dtype)

    def __add__(self, other):
        if self.shape != other.shape or self.device != other.device:
            raise AssertionError("invented addition mismatch")
        return Tensor(
            [
                [a + b for a, b in zip(left, right, strict=True)]
                for left, right in zip(self.values, other.values, strict=True)
            ],
            device=str(self.device),
            dtype=self.dtype,
        )

    def __truediv__(self, value):
        return Tensor(
            [[number / value for number in row] for row in self.values],
            device=str(self.device),
            dtype=self.dtype,
        )

    def __ge__(self, other):
        rows = max(self.shape[0], other.shape[0])
        columns = max(self.shape[1], other.shape[1])
        return Tensor(
            [
                [
                    self.values[min(i, self.shape[0] - 1)][min(j, self.shape[1] - 1)]
                    >= other.values[min(i, other.shape[0] - 1)][min(j, other.shape[1] - 1)]
                    for j in range(columns)
                ]
                for i in range(rows)
            ],
            device=str(self.device),
            dtype="bool",
        )

    def masked_fill(self, mask, value):
        if self.shape != mask.shape or self.device != mask.device:
            raise AssertionError("invented mask device/shape mismatch")
        return Tensor(
            [
                [value if blocked else number for number, blocked in zip(row, flags, strict=True)]
                for row, flags in zip(self.values, mask.values, strict=True)
            ],
            device=str(self.device),
            dtype=self.dtype,
        )

    def view(self, dtype):
        if dtype == "uint8":
            if len(self.shape) != 2:
                raise AssertionError("invented byte view requires matrix")
            return Tensor(
                [
                    list(b"".join(struct.pack("<f", number) for number in row))
                    for row in self.values
                ],
                device=str(self.device),
                dtype="uint8",
            )
        if dtype == -1:
            return Tensor(flatten(self.values), device=str(self.device), dtype=self.dtype)
        raise AssertionError("unsupported invented view")


class Torch:
    float32, bfloat16, uint8 = "float32", "bfloat16", "uint8"
    fence_calls = []
    precision = "highest"

    @staticmethod
    def get_float32_matmul_precision():
        return Torch.precision

    @staticmethod
    def is_autocast_enabled(device):
        if device != "cuda":
            raise AssertionError("unexpected invented autocast device")
        return False

    @staticmethod
    def is_tensor(value):
        return isinstance(value, Tensor)

    @staticmethod
    def isfinite(tensor):
        return Flag(all(math.isfinite(number) for number in flatten(tensor.values)))

    @staticmethod
    def inference_mode():
        return contextlib.nullcontext()

    @staticmethod
    def device(value):
        return Device(value)

    @staticmethod
    def zeros_like(tensor):
        return Tensor(
            [[0.0] * tensor.shape[1] for _ in range(tensor.shape[0])],
            device=str(tensor.device),
            dtype=tensor.dtype,
        )

    @staticmethod
    def as_tensor(values, *, device):
        return Tensor(list(values), device=str(device), dtype="int64")

    @staticmethod
    def arange(count, *, device):
        return Tensor(list(range(count)), device=str(device), dtype="int64")

    @staticmethod
    def equal(left, right):
        return (
            left.shape == right.shape
            and left.dtype == right.dtype
            and left.device == right.device
            and left.values == right.values
        )

    @staticmethod
    def softmax(tensor, *, dim):
        if dim != -1:
            raise AssertionError("invented softmax requires final dimension")
        result = []
        for row in tensor.values:
            maximum = max(row)
            weights = [math.exp(value - maximum) for value in row]
            total = sum(weights)
            result.append([value / total for value in weights])
        return Tensor(result, device=str(tensor.device), dtype=tensor.dtype)


def synchronize(index=0):
    if index != 0:
        raise AssertionError("invented fence selected another device")
    Torch.fence_calls.append(index)


Torch.cuda = types.SimpleNamespace(synchronize=synchronize)
Torch.backends = types.SimpleNamespace(
    cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)),
    cudnn=types.SimpleNamespace(allow_tf32=False),
)


def capability(module, owner_profile="invented-owned-gpu-profile"):
    value = {
        "schema": module.CAPABILITY_SCHEMA,
        "status": module.CAPABILITY_STATUS,
        "profile": module.PROFILE,
        "alias": module.ALIAS,
        "owner_profile": owner_profile,
        "purpose": module.PURPOSE,
        "only_head": "operation",
        "model": copy.deepcopy(module.MODEL_IDENTITY),
        "population": copy.deepcopy(module.POPULATION),
        "baseline_fp32_bytes_sha256": "a" * 64,
        "delta_fp32_bytes_sha256": "b" * 64,
        "criteria_mapping": module.CRITERIA_MAPPING,
        "head_mapping": module.HEAD_MAPPING,
        "dev_used_for_model_selection": True,
        "optimizer_input_dev": False,
        "optimizer_input_sealed": False,
        "browser_authorized": False,
        "benchmark_authorized": False,
        "training_authorized": False,
    }
    for key in module.REF_KEYS:
        value[key] = {"path": "/tmp/invented-" + key, "sha256": "c" * 64, "size_bytes": 42}
    return value


class RequestDTO:
    validations = []

    @classmethod
    def model_validate(cls, value):
        cls.validations.append(copy.deepcopy(value))
        result = cls()
        result.state = value["state"]
        result.questions = [types.SimpleNamespace(**entry) for entry in value["questions"]]
        result.media = value.get("media", [])
        result.model = value.get("model")
        return result

    def model_dump(self):
        return copy.deepcopy(
            {
                "state": self.state,
                "questions": [vars(question) for question in self.questions],
                "media": self.media,
                "model": self.model,
            }
        )


def request_fixture(names=None, counts=None):
    names = names or ["click_target", "singleton", "operation", "text"]
    counts = counts or [2, 1, 3, 2]
    return RequestDTO.model_validate(
        {
            "state": {"invented_view": {"items": ["alpha", None, 7]}, "second": []},
            "questions": [
                {
                    "id": head,
                    "type": "choice",
                    # Outer DTO fields intentionally differ from canonical native
                    # field order; inner JSON and candidate order must stay exact.
                    "criteria": {
                        f"invented-{index}": {"description": [head, index], "optional": None}
                        for index in range(count)
                    },
                    "instructions": {"last": [head, {"z": 1, "a": 2}], "first": "invented choice"},
                }
                for head, count in zip(names, counts, strict=True)
            ],
            "media": [],
            "model": "invented-transport-alias",
        }
    )


def dto_payload(request):
    return {
        "state": copy.deepcopy(request.state),
        "questions": {
            question.id: {
                "type": question.type,
                "instructions": copy.deepcopy(question.instructions),
                "criteria": copy.deepcopy(question.criteria),
            }
            for question in request.questions
        },
    }


class HookHandle:
    def __init__(self, owner, collection, hook):
        self.owner, self.collection, self.hook = owner, collection, hook

    def remove(self):
        self.collection.remove(self.hook)
        if self.owner.remove_failure is not None:
            raise self.owner.remove_failure


class Hookable:
    def __init__(self):
        self.pre_hooks, self.hooks, self.remove_failure = [], [], None

    def register_forward_pre_hook(self, hook):
        self.pre_hooks.append(hook)
        return HookHandle(self, self.pre_hooks, hook)

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return HookHandle(self, self.hooks, hook)


class Runtime:
    def __init__(self, module):
        self.module = module
        self.readout = Tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        self.encoder, self.prepare_calls, self.respond_calls, self.forward_calls = object(), 0, 0, 0
        self.parameter = Tensor([0.0], dtype=Torch.bfloat16)
        self.backbone = types.SimpleNamespace(
            language_model=types.SimpleNamespace(layers=[Hookable()], norm=Hookable()),
            parameters=lambda: iter([self.parameter]),
            training=False,
        )
        self.device, self.model_name = Device("cuda:0"), module.MODEL_IDENTITY["name"]
        self.batch_size, self.temperature, self.max_length = 8, 1.0, None
        self.readout_dtype, self.attention_mode = "float32", "noncausal_full_attention"
        self.permutation_average, self.fast = False, None
        self.manifest = {"identity": {"model_sha256": module.MODEL_IDENTITY["model_sha256"]}}

    def prepare(self, *, state, questions):
        self.prepare_calls += 1
        keys = list(questions)
        return types.SimpleNamespace(
            keys=keys,
            runnable=keys,
            errors=[],
            lengths={},
            sequences={key: [1] * (index + 1) for index, key in enumerate(keys)},
            questions={
                key: types.SimpleNamespace(keys=list(question["criteria"]))
                for key, question in questions.items()
            },
            images=[],
            videos=[],
            media=[],
            reversed_sequences=[],
        )

    def respond(self, prepared, probabilities, tokens):
        self.respond_calls += 1
        answers = {}
        for key in prepared.keys:
            labels = prepared.questions[key].keys
            values = probabilities[key]
            distribution = dict(zip(labels, values, strict=True))
            entropy = -sum(value * math.log(value) for value in values if value > 0)
            confidence = (
                1.0 if len(values) == 1 else max(0.0, min(1.0, 1 - entropy / math.log(len(values))))
            )
            answers[key] = {
                "type": "choice",
                "choice": max(distribution, key=distribution.__getitem__),
                "probabilities": distribution,
                "confidence": confidence,
                "publisher_extra": {
                    "response_serial": self.respond_calls,
                    "untouched": [None, key],
                },
            }
        return {
            "model": self.model_name,
            "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        }


def normalizer(_request, raw):
    result = copy.deepcopy(raw)
    for answer in result["answers"].values():
        answer["confidence"] = max(answer["probabilities"].values())
    return result


class Provider:
    def __init__(self, module):
        self.runtime = Runtime(module)
        self.spec = types.SimpleNamespace(
            adapter="decision3",
            model_id=module.MODEL_IDENTITY["model_id"],
            revision=module.MODEL_IDENTITY["revision"],
            device="cuda:0",
            precision="bfloat16",
            context_limit=module.CONTEXT_LIMIT,
        )
        self.identity = {"manifest_sha256": module.MODEL_IDENTITY["manifest_sha256"]}
        self.predict_calls, self.close_calls, self.payloads = 0, 0, []
        self.failure, self.failure_seen = None, False

    def predict(self, request, *, deadline, evidence_sink):
        if not isinstance(request, RequestDTO):
            raise AssertionError("invented normal DTO was replaced")
        self.predict_calls += 1
        payload = dto_payload(request)
        self.payloads.append(copy.deepcopy(payload))
        prepared = self.runtime.prepare(**payload)
        lengths = {key: len(sequence) for key, sequence in prepared.sequences.items()}
        evidence = {
            "native_payload": payload,
            "preflight": {
                "lengths": lengths,
                "errors": [],
                "project_context_limit": self.spec.context_limit,
            },
            "raw_response": None,
        }
        evidence_sink("native_preflight", copy.deepcopy(evidence))
        deadline()
        if self.failure is not None:
            self.failure_seen = True
            raise self.failure
        rows, width = len(prepared.keys), max(lengths.values())
        hidden = Tensor(
            [[[float(index + 1), 0.0] for _ in range(width)] for index in range(rows)],
            dtype=Torch.bfloat16,
        )
        language_model = self.runtime.backbone.language_model
        for hook in list(language_model.layers[0].pre_hooks):
            if hook(None, (hidden,)) is not None:
                raise AssertionError("first-layer hook changed the fake forward")
        self.runtime.forward_calls += 1
        for hook in list(language_model.norm.hooks):
            if hook(None, (), hidden) is not None:
                raise AssertionError("norm hook changed the fake forward")
        features = hidden[:, -1, :].float()
        counts = [len(prepared.questions[key].keys) for key in prepared.keys]
        logits = features @ self.runtime.readout.T
        invalid = (
            Torch.arange(self.runtime.module.CODES, device="cuda:0")[None]
            >= Torch.as_tensor(counts, device="cuda:0")[:, None]
        )
        full = Torch.softmax(logits.masked_fill(invalid, float("-inf")), dim=-1).tolist()
        probabilities = {
            key: row[:count] for key, row, count in zip(prepared.keys, full, counts, strict=True)
        }
        raw = self.runtime.respond(prepared, probabilities, sum(lengths.values()))
        evidence["raw_response"] = raw
        evidence_sink("native_raw_returned", copy.deepcopy(evidence))
        deadline()
        result = normalizer(request, raw)
        result["decision3_provider_evidence"] = copy.deepcopy(evidence)
        return result

    def resources(self):
        return {"invented_base_resource": True}

    def close(self):
        self.close_calls += 1


class Bomb:
    def __getattr__(self, name):
        raise AssertionError("dependency accessed before release/capability: " + name)

    def __len__(self):
        raise AssertionError("dependency sized before release/capability")

    def __call__(self, *args, **kwargs):
        raise AssertionError("dependency called before release/capability")


@contextlib.contextmanager
def tiny_math(module):
    with (
        mock.patch.object(module, "SOURCE_ONLY", False),
        mock.patch.object(module, "CAPABILITY_SHA256", "a" * 64, create=True),
        mock.patch.object(module, "CODES", 3),
        mock.patch.object(module, "WIDTH", 2),
    ):
        yield


class ProjectionControls(unittest.TestCase):
    def setUp(self):
        self.module = source()
        Tensor.matmul_calls = []
        Torch.fence_calls = []

    def test_source_hold_precedes_any_projection_or_constructor_dependency(self):
        self.assertIs(self.module.SOURCE_ONLY, True)
        with self.assertRaisesRegex(RuntimeError, "SOURCE_ONLY"):
            self.module.project(Bomb(), Bomb(), Bomb(), Bomb(), Bomb(), Bomb())
        with self.assertRaisesRegex(RuntimeError, "SOURCE_ONLY"):
            self.module.OperationDeltaGPUBackend(
                Bomb(),
                Bomb(),
                Bomb(),
                capability=Bomb(),
                capability_sha256=Bomb(),
                owner_profile=Bomb(),
                request_type=Bomb(),
                normalize_native_response=Bomb(),
                identity_guard=Bomb(),
                transfer_witness=Bomb(),
                witness_sink=Bomb(),
            )

    def test_actual_nonzero_operation_row_changes_only_that_ordinal_distribution(self):
        base = Tensor([[-0.0, 1.0, float("-inf")], [2.0, 1.0, 0.0], [0.0, 1.0, float("-inf")]])
        features = Tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 2.0]])
        delta = Tensor([[3.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        original_base = base.view(Torch.uint8).tolist()
        original_features, original_delta = features.tolist(), delta.tolist()
        with tiny_math(self.module):
            probabilities, metadata = self.module.project(
                Torch, base, features, delta, [2, 3, 2], ["click_target", "text", "operation"]
            )
        self.assertEqual(Tensor.matmul_calls, [((1, 2), (2, 3))])
        self.assertEqual(base.view(Torch.uint8).tolist(), original_base)
        self.assertEqual(features.tolist(), original_features)
        self.assertEqual(delta.tolist(), original_delta)
        expected_first = [1.0 / (1.0 + math.e), math.e / (1.0 + math.e)]
        for actual, expected in zip(probabilities[0], expected_first, strict=True):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(len(probabilities[1]), 3)
        self.assertGreater(probabilities[1][0], probabilities[1][1])
        self.assertGreater(probabilities[2][0], probabilities[2][1])
        self.assertEqual([len(row) for row in probabilities], [2, 3, 2])
        self.assertTrue(all(abs(sum(row) - 1.0) < 1e-12 for row in probabilities))
        json.dumps(metadata, allow_nan=False)

    def test_no_operation_does_not_project_or_add_a_head(self):
        base = Tensor([[-0.0, 1.0, float("-inf")], [0.0, float("-inf"), float("-inf")]])
        features = Tensor([[1.0, 0.0], [1.0, 2.0]])
        delta = Tensor([[100.0, 100.0], [-100.0, -100.0], [3.0, 4.0]])
        with tiny_math(self.module):
            probabilities, metadata = self.module.project(
                Torch, base, features, delta, [2, 1], ["target", "singleton"]
            )
        self.assertEqual(Tensor.matmul_calls, [])
        self.assertEqual([len(row) for row in probabilities], [2, 1])
        self.assertEqual(probabilities[1], [1.0])
        self.assertLess(probabilities[0][0], probabilities[0][1])
        json.dumps(metadata, allow_nan=False)

    def test_singleton_operation_keeps_one_legal_code_without_skipping_projection(self):
        base = Tensor([[0.0, float("-inf"), float("-inf")]])
        features = Tensor([[1.0, 2.0]])
        delta = Tensor([[2.0, 1.0], [7.0, 8.0], [9.0, 10.0]])
        with tiny_math(self.module):
            probabilities, _ = self.module.project(Torch, base, features, delta, [1], ["operation"])
        self.assertEqual(probabilities, [[1.0]])
        self.assertEqual(Tensor.matmul_calls, [((1, 2), (2, 3))])

    def test_original_real_geometry_is_not_replaced_by_the_small_fake_profile(self):
        self.assertEqual((self.module.CODES, self.module.WIDTH), (255, 1024))
        with (
            mock.patch.object(self.module, "SOURCE_ONLY", False),
            mock.patch.object(self.module, "CAPABILITY_SHA256", "a" * 64, create=True),
            self.assertRaises(RuntimeError),
        ):
            self.module.project(
                Torch,
                Tensor([[0.0] * 3]),
                Tensor([[1.0, 2.0]]),
                Tensor([[0.0, 0.0]] * 3),
                [2],
                ["operation"],
            )
        self.assertEqual(Tensor.matmul_calls, [])

    def test_wrong_device_precision_counts_or_head_identity_are_refused(self):
        valid_base = Tensor([[0.0, 1.0, float("-inf")]])
        valid_features, valid_delta = Tensor([[1.0, 0.0]]), Tensor([[0.0, 0.0]] * 3)
        attempts = (
            (valid_base.to(device="cpu"), valid_features, valid_delta, [2], ["operation"]),
            (valid_base, valid_features.to(device="cuda:1"), valid_delta, [2], ["operation"]),
            (valid_base, valid_features, valid_delta.to(dtype=Torch.bfloat16), [2], ["operation"]),
            (valid_base, valid_features, valid_delta, [0], ["operation"]),
            (valid_base, valid_features, valid_delta, [4], ["operation"]),
            (valid_base, valid_features, valid_delta, [True], ["operation"]),
            (valid_base, valid_features, valid_delta, [2], [""]),
            (valid_base, valid_features, valid_delta, [2], []),
        )
        with tiny_math(self.module):
            for base, features, delta, counts, heads in attempts:
                with self.subTest(counts=counts, heads=heads), self.assertRaises(RuntimeError):
                    self.module.project(Torch, base, features, delta, counts, heads)

    def test_duplicate_operation_rows_and_nonfinite_legal_values_are_refused(self):
        with tiny_math(self.module):
            with self.assertRaises(RuntimeError):
                self.module.project(
                    Torch,
                    Tensor([[0.0, float("-inf"), float("-inf")]] * 2),
                    Tensor([[1.0, 0.0]] * 2),
                    Tensor([[0.0, 0.0]] * 3),
                    [1, 1],
                    ["operation", "operation"],
                )
            with self.assertRaises(RuntimeError):
                self.module.project(
                    Torch,
                    Tensor([[float("nan"), 0.0, float("-inf")]]),
                    Tensor([[1.0, 0.0]]),
                    Tensor([[0.0, 0.0]] * 3),
                    [2],
                    ["operation"],
                )


class ObserverControls(unittest.TestCase):
    def setUp(self):
        self.module = source()

    def test_single_same_call_observation_keeps_only_last_position_features(self):
        rows, width = 2, 3
        hidden = Tensor(
            [
                [[float(row * 10 + position)] * 1024 for position in range(width)]
                for row in range(rows)
            ],
            dtype=Torch.bfloat16,
        )
        with (
            mock.patch.object(self.module, "SOURCE_ONLY", False),
            mock.patch.object(self.module, "CAPABILITY_SHA256", "a" * 64, create=True),
        ):
            observer = self.module.GPUObserver(Torch, rows, width)
            self.assertIsNone(observer.first_layer(None, (hidden,)))
            self.assertIsNone(observer(None, (), hidden))
            observer.require_single_pass()
            self.assertEqual(observer.features.shape, (rows, 1024))
            self.assertEqual(str(observer.features.device), "cuda:0")
            self.assertEqual(observer.features.dtype, Torch.float32)
            self.assertEqual(observer.features.tolist(), [[2.0] * 1024, [12.0] * 1024])
            self.assertEqual(len(observer.first_layer_events), 1)
            self.assertEqual(len(observer.norm_events), 1)
            hidden.values[0][-1][0] = 999.0
            self.assertEqual(observer.features.tolist()[0][0], 2.0)
            with self.assertRaises(RuntimeError):
                observer.first_layer(None, (hidden,))
            with self.assertRaises(RuntimeError):
                observer(None, (), hidden)

    def test_observer_refuses_wrong_cuda_geometry_precision_and_missing_pass(self):
        valid = Tensor([[[0.0] * 1024] * 2], dtype=Torch.bfloat16)
        with (
            mock.patch.object(self.module, "SOURCE_ONLY", False),
            mock.patch.object(self.module, "CAPABILITY_SHA256", "a" * 64, create=True),
        ):
            observer = self.module.GPUObserver(Torch, 1, 2)
            with self.assertRaises(RuntimeError):
                observer.require_single_pass()
            for hidden in (
                valid.to(device="cpu"),
                valid.to(device="cuda:1"),
                valid.to(dtype=Torch.float32),
                Tensor([[[0.0] * 1023] * 2], dtype=Torch.bfloat16),
            ):
                with self.subTest(device=str(hidden.device), shape=hidden.shape):
                    observer = self.module.GPUObserver(Torch, 1, 2)
                    with self.assertRaises(RuntimeError):
                        observer.first_layer(None, (hidden,))


class ProviderCompositionControls(unittest.TestCase):
    def setUp(self):
        self.module = source()
        self.module.SOURCE_ONLY = False
        self.module.CODES, self.module.WIDTH = 3, 2
        # This fixture covers composition control flow with tiny arithmetic;
        # runtime/model authentication remains root's real-GPU responsibility.
        self.real_runtime_validator = self.module.OperationDeltaGPUBackend._validate_runtime
        runtime_patch = mock.patch.object(
            self.module.OperationDeltaGPUBackend, "_validate_runtime", return_value=None
        )
        runtime_patch.start()
        self.addCleanup(runtime_patch.stop)
        Torch.precision = "highest"
        Torch.backends.cuda.matmul.allow_tf32 = False
        Torch.backends.cudnn.allow_tf32 = False
        Torch.fence_calls, Tensor.matmul_calls, RequestDTO.validations = [], [], []
        self.now = types.SimpleNamespace(value=0.0)
        self.base = Provider(self.module)
        self.cpu_delta = Tensor([[4.0, 0.0], [0.0, 0.0], [0.0, 0.0]], device="cpu")
        self.capability = capability(self.module)
        self.witness_calls = []

    def witness(self, **values):
        self.assertEqual(
            set(values),
            {
                "torch",
                "features",
                "baseline",
                "delta",
                "native_payload",
                "lengths",
                "ordered_head_ids",
                "ordered_criteria_ids",
                "baseline_raw",
                "composed_raw",
                "projection",
            },
        )
        self.witness_calls.append(values)

    def backend(self, **overrides):
        def transfer(_source, _destination):
            return {
                "bytewise_equal": True,
                "source_device": "cpu",
                "destination_device": "cuda:0",
                "source_fp32_bytes_sha256": self.capability["delta_fp32_bytes_sha256"],
                "destination_fp32_bytes_sha256": self.capability["delta_fp32_bytes_sha256"],
            }

        values = {
            "capability": self.capability,
            "capability_sha256": "d" * 64,
            "owner_profile": self.capability["owner_profile"],
            "request_type": RequestDTO,
            "normalize_native_response": normalizer,
            "identity_guard": lambda _base, _delta: {"invented_guard": True},
            "transfer_witness": transfer,
            "witness_sink": self.witness,
            "clock": lambda: self.now.value,
        }
        values.update(overrides)
        return self.module.OperationDeltaGPUBackend(self.base, self.cpu_delta, Torch, **values)

    def assert_no_hooks(self):
        language_model = self.base.runtime.backbone.language_model
        self.assertEqual(language_model.layers[0].pre_hooks, [])
        self.assertEqual(language_model.norm.hooks, [])

    def test_new_capability_rejects_old_fit_authority_and_authorization_grants_before_dependencies(
        self,
    ):
        self.assertEqual(set(self.capability), self.module.CAPABILITY_KEYS)
        self.module.validate_capability(self.capability, self.capability["owner_profile"])
        for key, value in (
            ("schema", "root-accepted-dsl01-operation-train-delta-v1"),
            ("only_head", "all"),
            ("dev_used_for_model_selection", False),
            ("optimizer_input_dev", True),
            ("optimizer_input_sealed", True),
            ("browser_authorized", True),
            ("benchmark_authorized", True),
            ("training_authorized", True),
            ("head_mapping", "operation_is_always_row0"),
        ):
            changed = {**self.capability, key: value}
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                self.module.OperationDeltaGPUBackend(
                    Bomb(),
                    Bomb(),
                    Bomb(),
                    capability=changed,
                    capability_sha256="d" * 64,
                    owner_profile=self.capability["owner_profile"],
                    request_type=Bomb(),
                    normalize_native_response=Bomb(),
                    identity_guard=Bomb(),
                    transfer_witness=Bomb(),
                    witness_sink=Bomb(),
                )
        changed = copy.deepcopy(self.capability)
        changed["population"]["rows"] = 96
        with self.assertRaises(RuntimeError):
            self.module.validate_capability(changed, changed["owner_profile"])

    def test_real_runtime_validator_rejects_the_tiny_fake_geometry_unpatched(self):
        value = object.__new__(self.module.OperationDeltaGPUBackend)
        value.base, value.runtime, value.spec, value.torch = (
            self.base,
            self.base.runtime,
            self.base.spec,
            Torch,
        )
        with (
            mock.patch.object(self.module, "CODES", 255),
            mock.patch.object(self.module, "WIDTH", 1024),
            self.assertRaisesRegex(RuntimeError, "full255 W0"),
        ):
            self.real_runtime_validator(value)

    def test_one_normal_provider_call_preserves_every_raw_target_and_ordinal(self):
        events = []

        def sink(event, evidence):
            events.append((event, copy.deepcopy(evidence)))

        backend = self.backend(evidence_sink=sink)
        request = request_fixture()
        original = request.model_dump()
        validation_count = len(RequestDTO.validations)
        result = backend.predict(request, deadline=lambda: None, evidence_sink=sink)
        self.assertIs(backend.spec, self.base.spec)
        self.assertEqual(result["model"], self.module.MODEL_IDENTITY["name"])
        self.assertEqual(self.base.predict_calls, 1)
        self.assertEqual(self.base.runtime.forward_calls, 1)
        self.assertEqual(self.base.runtime.prepare_calls, 2)
        self.assertEqual(len(RequestDTO.validations), validation_count + 1)
        self.assertEqual(request.model_dump(), original)
        self.assertEqual(
            [name for name, _ in events],
            [
                "native_preflight",
                "baseline_native_raw_returned",
                "operation_delta_projected",
                "native_raw_returned",
                "completed",
            ],
        )
        self.assertEqual(len(self.witness_calls), 1)
        witness = self.witness_calls[0]
        expected_heads = [question.id for question in request.questions]
        self.assertEqual(witness["ordered_head_ids"], expected_heads)
        self.assertEqual(
            witness["ordered_criteria_ids"],
            [list(question.criteria) for question in request.questions],
        )
        self.assertEqual(self.base.payloads[0]["state"], request.state)
        self.assertEqual(list(self.base.payloads[0]["questions"]), expected_heads)
        for question in request.questions:
            native_question = self.base.payloads[0]["questions"][question.id]
            self.assertEqual(list(native_question["criteria"]), list(question.criteria))
            self.assertEqual(
                json.dumps(native_question["instructions"]), json.dumps(question.instructions)
            )
        for head in expected_heads:
            if head != "operation":
                self.assertEqual(
                    json.dumps(witness["baseline_raw"]["answers"][head]),
                    json.dumps(witness["composed_raw"]["answers"][head]),
                )
        operation = witness["composed_raw"]["answers"]["operation"]
        self.assertEqual(operation["choice"], "invented-0")
        self.assertAlmostEqual(
            result["answers"]["operation"]["confidence"], max(operation["probabilities"].values())
        )
        self.assertNotAlmostEqual(
            operation["confidence"], result["answers"]["operation"]["confidence"]
        )
        self.assertTrue(backend.last_evidence["final_immutability_guard_passed"])
        self.assertIs(backend._in_call, False)
        self.assert_no_hooks()
        backend.close()
        self.assertTrue(backend.closed)
        self.assertEqual(self.base.close_calls, 0)

    def test_completed_and_failed_sinks_cannot_reenter_predict_or_close(self):
        for terminal in ("completed", "failed"):
            self.base.failure = (
                KeyboardInterrupt("invented original failure") if terminal == "failed" else None
            )
            self.base.failure_seen = False
            request = request_fixture()
            observed = []

            def sink(event, _evidence):
                if event != terminal:
                    return
                self.assertIs(backend._in_call, True)
                with self.assertRaises(RuntimeError):
                    backend.predict(request, deadline=lambda: None)
                with self.assertRaises(RuntimeError):
                    backend.close()
                self.assertIs(backend._in_call, True)
                observed.append(event)

            backend = self.backend(evidence_sink=sink)
            if terminal == "failed":
                with self.assertRaises(KeyboardInterrupt) as caught:
                    backend.predict(request, deadline=lambda: None)
                self.assertIs(caught.exception, self.base.failure)
            else:
                backend.predict(request, deadline=lambda: None)
            self.assertEqual(observed, [terminal])
            self.assertIs(backend._in_call, False)
            self.assert_no_hooks()

    def test_terminal_sink_changes_to_baseline_or_delta_cannot_return_success(self):
        for which in ("baseline", "delta"):

            def sink(event, _evidence):
                if event == "completed":
                    tensor = self.base.runtime.readout if which == "baseline" else backend.delta
                    tensor.values[0][0] += 1.0
                    tensor._version += 1

            backend = self.backend(evidence_sink=sink)
            with self.assertRaises(RuntimeError):
                backend.predict(request_fixture(), deadline=lambda: None)
            self.assertIs(backend._in_call, False)
            self.assertEqual(backend.last_evidence["status"], "failed")
            self.assertFalse(backend.last_evidence["completed"])
            self.assert_no_hooks()
            # A fresh base is required for the next independent invented case.
            self.base = Provider(self.module)

    def test_original_forward_error_survives_failed_sink_guard_hook_and_fence_failures(self):
        primary = KeyboardInterrupt("invented original forward failure")
        self.base.failure = primary
        self.base.runtime.backbone.language_model.norm.remove_failure = OSError(
            "invented hook cleanup"
        )

        def guard(_base, _delta):
            if self.base.failure_seen:
                raise OSError("invented terminal guard cleanup")
            return {"invented_guard": True}

        def sink(event, _evidence):
            if event == "failed":
                self.assertIs(backend._in_call, True)
                raise ValueError("invented failed sink cleanup")

        def fence(index=0):
            if self.base.failure_seen:
                raise OSError("invented terminal fence cleanup")
            synchronize(index)

        backend = self.backend(identity_guard=guard, evidence_sink=sink)
        with mock.patch.object(Torch.cuda, "synchronize", side_effect=fence):
            with self.assertRaises(KeyboardInterrupt) as caught:
                backend.predict(request_fixture(), deadline=lambda: None)
        self.assertIs(caught.exception, primary)
        notes = "\n".join(primary.__notes__)
        for expected in (
            "invented hook cleanup",
            "invented terminal guard cleanup",
            "invented failed sink cleanup",
            "invented terminal fence cleanup",
        ):
            self.assertIn(expected, notes)
        self.assertIs(backend._in_call, False)
        self.assertEqual(backend.last_evidence["status"], "failed")
        self.assert_no_hooks()

    def test_required_private_witness_failure_or_mutation_has_no_success(self):
        for behavior in ("raise", "mutate"):
            primary = OSError("invented required witness failure")

            def witness(**values):
                if behavior == "raise":
                    raise primary
                values["features"].values[0][0] += 1.0

            backend = self.backend(witness_sink=witness)
            with self.assertRaises((OSError, RuntimeError)) as caught:
                backend.predict(request_fixture(), deadline=lambda: None)
            if behavior == "raise":
                self.assertIs(caught.exception, primary)
            self.assertEqual(backend.last_evidence["status"], "failed")
            self.assertIs(backend._in_call, False)
            self.assert_no_hooks()

    def test_secondary_note_refusal_does_not_replace_the_original_forward_error(self):
        class NoteRefusingInterrupt(KeyboardInterrupt):
            def add_note(self, _note):
                raise OSError("invented refused exception note")

        primary = NoteRefusingInterrupt("invented original forward failure")
        self.base.failure = primary
        self.base.runtime.backbone.language_model.norm.remove_failure = OSError(
            "invented hook cleanup"
        )
        backend = self.backend()
        with self.assertRaises(NoteRefusingInterrupt) as caught:
            backend.predict(request_fixture(), deadline=lambda: None)
        self.assertIs(caught.exception, primary)
        self.assertFalse(backend._in_call)
        self.assertFalse(backend.last_evidence["completed"])
        self.assertIn(
            "observation_hook_removal",
            [row["stage"] for row in backend.last_evidence["secondary_errors"]],
        )
        self.assertIn("pre_terminal_elapsed_seconds", backend.last_evidence)
        self.assert_no_hooks()

    def test_exact20_after_completed_retention_does_not_authorize_result(self):
        def sink(event, _evidence):
            if event == "completed":
                self.now.value = 20.0

        backend = self.backend(evidence_sink=sink)
        with self.assertRaises(RuntimeError):
            backend.predict(request_fixture(), deadline=lambda: None)
        self.assertIs(backend._in_call, False)
        self.assertEqual(backend.last_evidence["status"], "failed")
        self.assertFalse(backend.last_evidence["completed"])
        self.assert_no_hooks()


if __name__ == "__main__":
    unittest.main()
