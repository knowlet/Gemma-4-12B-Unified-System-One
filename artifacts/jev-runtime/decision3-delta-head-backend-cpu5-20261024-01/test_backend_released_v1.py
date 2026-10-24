"""Unexecuted stdlib fakes; no model, framework, data or actual artifact access.

The fake matrix arithmetic exercises composition/order/masking only. It does not
establish Torch numerical parity, CPU resource closure or runtime qualification.
"""

import contextlib
import copy
import importlib.util
import math
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def load_candidate():
    path = Path(__file__).with_name("backend_released_v1.py")
    spec = importlib.util.spec_from_file_location("held_delta_backend_fake", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Matrix:
    def __init__(self, values, *, name="tensor", dtype="float32", shape=None):
        self.values = copy.deepcopy(values)
        self.shape = shape or (len(values), len(values[0]))
        self.dtype, self.name = dtype, name
        self.device = types.SimpleNamespace(type="cpu")
        self.requires_grad = False
        self._version = 0

    def data_ptr(self):
        return id(self.values)

    def is_contiguous(self):
        return True

    def detach(self):
        return self

    def clone(self):
        return Matrix(self.values, name=self.name, dtype=self.dtype, shape=self.shape)

    def cpu(self):
        return self

    def tolist(self):
        return copy.deepcopy(self.values)

    @property
    def T(self):
        return Matrix(list(map(list, zip(*self.values))), name=self.name + ".T")

    def __matmul__(self, other):
        columns = list(zip(*other.values))
        return Matrix(
            [[sum(a * b for a, b in zip(row, col)) for col in columns] for row in self.values]
        )

    def __add__(self, other):
        return Matrix(
            [[a + b for a, b in zip(left, right)] for left, right in zip(self.values, other.values)]
        )

    def __truediv__(self, divisor):
        return Matrix([[value / divisor for value in row] for row in self.values])

    def __getitem__(self, key):
        if key is None:
            return Matrix([self.values], shape=(1, len(self.values)))
        if isinstance(key, tuple) and key[1] is None:
            return Matrix([[value] for value in self.values], shape=(len(self.values), 1))
        raise AssertionError("unexpected fake tensor indexing")

    def __ge__(self, other):
        rows = max(len(self.values), len(other.values))
        width = max(len(self.values[0]), len(other.values[0]))
        return Matrix(
            [
                [
                    self.values[i % len(self.values)][j % len(self.values[0])]
                    >= other.values[i % len(other.values)][j % len(other.values[0])]
                    for j in range(width)
                ]
                for i in range(rows)
            ]
        )

    def masked_fill(self, mask, fill):
        return Matrix(
            [
                [fill if bad else value for value, bad in zip(row, invalid)]
                for row, invalid in zip(self.values, mask.values)
            ]
        )

    def all(self):
        value = all(all(row) for row in self.values)
        return types.SimpleNamespace(item=lambda: value)


class FakeTorch:
    float32 = "float32"
    bfloat16 = "bfloat16"

    @staticmethod
    def is_tensor(value):
        return isinstance(value, Matrix)

    @staticmethod
    def inference_mode():
        return contextlib.nullcontext()

    @staticmethod
    def isfinite(tensor):
        return Matrix([[math.isfinite(value) for value in row] for row in tensor.values])

    @staticmethod
    def as_tensor(values, **_kwargs):
        return Matrix(values, shape=(len(values),))

    @staticmethod
    def arange(count, **_kwargs):
        return Matrix(list(range(count)), shape=(count,))

    @staticmethod
    def softmax(tensor, dim):
        assert dim == -1
        result = []
        for row in tensor.values:
            maximum = max(row)
            exps = [math.exp(value - maximum) for value in row]
            denominator = sum(exps)
            result.append([value / denominator for value in exps])
        return Matrix(result)


class Handle:
    def __init__(self, hooks, hook, failure=None):
        self.hooks, self.hook, self.failure = hooks, hook, failure

    def remove(self):
        self.hooks.remove(self.hook)
        if self.failure is not None:
            raise self.failure


class HookModule:
    def __init__(self):
        self.hooks = []
        self.remove_failure = None

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return Handle(self.hooks, hook, self.remove_failure)

    register_forward_pre_hook = register_forward_hook


class Watch:
    def __init__(self, _torch, rows, padded_width):
        self.rows, self.width = rows, padded_width
        self.first_layer_events, self.norm_events = [], []
        self.features = None

    def first_layer(self, _module, args):
        assert not self.first_layer_events
        assert args[0].shape == (self.rows, self.width, 1024)
        self.first_layer_events.append(list(args[0].shape))

    def __call__(self, _module, _args, output):
        assert not self.norm_events
        assert output.shape == (self.rows, self.width, 1024)
        self.norm_events.append(list(output.shape))
        self.features = output.features.clone()

    def require_single_pass(self):
        assert len(self.first_layer_events) == len(self.norm_events) == 1
        assert self.features is not None


class FakeObserver:
    FinalNormObserver = Watch

    @staticmethod
    def tensor_identity(tensor):
        return id(tensor), tensor.shape, tensor.dtype, tensor.data_ptr(), tensor._version

    @staticmethod
    def baseline_identity(runtime):
        if runtime.identity_callback is not None:
            runtime.identity_callback()
        return (
            FakeObserver.tensor_identity(runtime.readout),
            id(runtime.backbone),
            id(runtime.encoder),
        )


class FakeRuntime:
    def __init__(self):
        self.model_name, self.batch_size, self.temperature = "d3-edge", 4, 1.0
        self.encoder = object()
        self.readout = Matrix([[float(i)] + [0.0] * 1023 for i in range(255)], name="W0")
        self.backbone = types.SimpleNamespace(
            language_model=types.SimpleNamespace(layers=[HookModule()], norm=HookModule())
        )
        self.forward_calls, self.respond_calls, self.prepare_calls = [], 0, []
        self.failure, self.identity_callback = None, None

    def logits(self, sequences, counts):
        width = max(map(len, sequences))
        self.forward_calls.append((copy.deepcopy(sequences), list(counts)))
        features = Matrix(
            [[float(sequence[0])] + [0.0] * 1023 for sequence in sequences], name="features"
        )
        output = types.SimpleNamespace(shape=(len(sequences), width, 1024), features=features)
        first, norm = self.backbone.language_model.layers[0], self.backbone.language_model.norm
        for hook in first.hooks:
            hook(first, (output,))
        for hook in norm.hooks:
            hook(norm, (), output)
        if self.failure is not None:
            raise self.failure
        logits = features @ self.readout.T
        mask = Matrix([[i >= count for i in range(255)] for count in counts])
        return logits.masked_fill(mask, float("-inf"))

    def respond(self, prepared, probabilities, tokens):
        self.respond_calls += 1
        answers = {}
        for key in prepared.keys:
            labels, row = prepared.questions[key].keys, probabilities[key]
            distribution = dict(zip(labels, row))
            entropy = -sum(p * math.log(p) for p in row if p)
            confidence = (
                1.0 if len(row) == 1 else max(0.0, min(1.0, 1 - entropy / math.log(len(row))))
            )
            answers[key] = {
                "type": "choice",
                "choice": max(distribution, key=distribution.__getitem__),
                "probabilities": distribution,
                "confidence": confidence,
            }
        return {
            "model": self.model_name,
            "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        }


class FakeNative:
    class NativeBatch4Backend:
        def __init__(self, runtime, _torch, model, context_limit):
            assert context_limit == 16384
            self.runtime, self.model = runtime, model
            self.check_runtime()

        def check_runtime(self):
            assert self.runtime.model_name == self.model["name"]
            assert self.runtime.batch_size == 4 and self.runtime.temperature == 1.0

        def prepare(self, state, questions):
            self.runtime.prepare_calls.append((copy.deepcopy(state), copy.deepcopy(questions)))
            sequences = {key: value["tokens"] for key, value in questions.items()}
            lengths = {key: len(value) for key, value in sequences.items()}
            if any(length < 1 or length > 16384 for length in lengths.values()):
                raise ValueError("context cap exceeded; nothing truncated")
            prepared = types.SimpleNamespace(
                keys=list(questions),
                runnable=list(questions),
                sequences=sequences,
                questions={
                    key: types.SimpleNamespace(keys=list(value["criteria"]))
                    for key, value in questions.items()
                },
                images=[],
                videos=[],
                media={},
                reversed_sequences={},
            )
            return prepared, lengths

    @staticmethod
    def validate_native(raw, questions, lengths, model):
        assert raw["model"] == model["name"]
        assert list(raw["answers"]) == list(questions)
        assert raw["usage"] == {"input_tokens": sum(lengths.values()), "output_tokens": 0}


def capability(module, model):
    ref = {"path": "/independently/selected/train-only", "sha256": "a" * 64, "size_bytes": 7}
    return {
        "schema": module.CAPABILITY_SCHEMA,
        "status": "ROOT_ACCEPTED_TRAIN_ONLY_READOUT",
        "split": "train",
        "training_eligible": True,
        "nontraining": False,
        "profile": module.PROFILE,
        "alias": module.ALIAS,
        "owner_profile": module.OWNER_PROFILE,
        "model": copy.deepcopy(model),
        "delta_ref": dict(ref),
        "root_acceptance_ref": dict(ref),
        "baseline_fp32_bytes_sha256": "b" * 64,
    }


def questions(count=1):
    return {
        "head-" + str(i): {
            "criteria": {"second-offered": "A", "first-named": "B"},
            "tokens": [1] * (i + 1),
        }
        for i in range(count)
    }


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.module = load_candidate()
        self.module.SOURCE_ONLY = False
        self.runtime, self.clock = FakeRuntime(), types.SimpleNamespace(now=0.0)
        self.model = {"name": "d3-edge", "model_sha256": "c" * 64}
        self.delta = Matrix([[0.0] * 1024 for _ in range(255)], name="delta")

    def make(self, **kwargs):
        return self.module.DeltaHeadBackend(
            FakeNative,
            FakeObserver,
            self.runtime,
            FakeTorch,
            self.model,
            self.delta,
            root_capability=kwargs.pop("root_capability", capability(self.module, self.model)),
            deadline=kwargs.pop("deadline", lambda: None),
            clock=lambda: self.clock.now,
            **kwargs,
        )

    def test_source_hold_precedes_all_dependency_access(self):
        self.module.SOURCE_ONLY = True

        class Bomb:
            def __getattr__(self, _name):
                raise AssertionError("dependency accessed before hold")

        with self.assertRaisesRegex(RuntimeError, "SOURCE_ONLY"):
            self.module.DeltaHeadBackend(
                Bomb(),
                Bomb(),
                Bomb(),
                Bomb(),
                Bomb(),
                Bomb(),
                root_capability=Bomb(),
                deadline=Bomb(),
            )

    def test_nontraining_or_foreign_model_cannot_become_training_capability(self):
        for key, value in (
            ("nontraining", True),
            ("split", "dev"),
            ("training_eligible", False),
            ("model", {"name": "foreign"}),
            ("root_acceptance_ref", None),
        ):
            cap = capability(self.module, self.model)
            cap[key] = value
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(RuntimeError, "root-accepted train-only"),
            ):
                self.make(root_capability=cap)
        self.assertEqual(self.runtime.forward_calls, [])

    def test_zero_delta_preserves_order_mask_entropy_and_baseline_storage(self):
        backend = self.make()
        before = FakeObserver.baseline_identity(self.runtime)
        result = backend({"context": "unchanged"}, questions())
        row = result["answers"]["head-0"]
        expected = 1 / (1 + math.exp(1))
        self.assertEqual(list(row["probabilities"]), ["second-offered", "first-named"])
        self.assertAlmostEqual(row["probabilities"]["second-offered"], expected)
        self.assertEqual(row["choice"], "first-named")
        self.assertEqual(FakeObserver.baseline_identity(self.runtime), before)
        self.assertEqual(result["model"], self.module.ALIAS)
        self.assertFalse(backend.last_evidence["publisher_native_numerics"])
        self.assertEqual(
            backend.last_evidence["respond_before_candidate_alias"]["model"], "d3-edge"
        )
        self.assertEqual(self.runtime.respond_calls, 1)

    def test_nonzero_delta_changes_real_readout_without_merging_W0(self):
        self.delta.values[0][0] = 2.0
        backend = self.make()
        self.delta.values[0][0] = 90.0  # External tensor is not the owned copy.
        result = backend({}, questions())
        self.assertEqual(result["answers"]["head-0"]["choice"], "second-offered")
        self.assertEqual(self.runtime.readout.values[0][0], 0.0)
        self.assertEqual(backend.delta.values[0][0], 2.0)
        self.assertEqual(
            backend.last_evidence["chunks"][0]["projection"]["adapted_legal_logits"], [[2.0, 1.0]]
        )

    def test_true_four_plus_singleton_tail_no_fake_or_omitted_questions(self):
        offered = questions(5)
        offered["head-4"]["criteria"] = {"only-option": "Only"}
        backend = self.make()
        result = backend({"context": "unchanged"}, offered)
        self.assertEqual(list(result["answers"]), list(offered))
        self.assertEqual([len(call[0]) for call in self.runtime.forward_calls], [4, 1])
        chunks = backend.last_evidence["chunks"]
        self.assertEqual(
            [chunk["first_layer_events"] for chunk in chunks], [[[4, 4, 1024]], [[1, 5, 1024]]]
        )
        self.assertEqual(
            [chunk["final_norm_events"] for chunk in chunks], [[[4, 4, 1024]], [[1, 5, 1024]]]
        )
        self.assertEqual([chunk["physical_padded_input_tokens"] for chunk in chunks], [16, 5])
        self.assertEqual(result["usage"]["input_tokens"], 15)
        self.assertEqual(result["answers"]["head-4"]["confidence"], 1.0)
        self.assertEqual(result["answers"]["head-4"]["probabilities"], {"only-option": 1.0})

    def test_native_context_and_unsupported_kwargs_fail_before_any_forward(self):
        backend = self.make()
        oversized = questions()
        oversized["head-0"]["tokens"] = [1] * 16385
        with self.assertRaisesRegex(ValueError, "nothing truncated"):
            backend({}, oversized)
        with self.assertRaises(TypeError):
            backend({}, questions(), images=["unsupported"])
        self.assertEqual(self.runtime.forward_calls, [])

    def test_forward_primary_survives_hook_cleanup_and_sink_failure(self):
        original = KeyboardInterrupt("original forward interrupted")
        self.runtime.failure = original
        self.runtime.backbone.language_model.norm.remove_failure = RuntimeError("hook cleanup")

        def sink(event, _data):
            if event == "failed":
                raise OSError("secondary failure retention")

        backend = self.make(evidence_sink=sink)
        with self.assertRaises(KeyboardInterrupt) as caught:
            backend({}, questions())
        self.assertIs(caught.exception, original)
        self.assertTrue(any("hook removal" in note for note in original.__notes__))
        self.assertEqual(backend.last_evidence["status"], "failed")
        self.assertEqual(self.runtime.backbone.language_model.norm.hooks, [])
        self.assertEqual(self.runtime.backbone.language_model.layers[0].hooks, [])
        self.assertTrue(backend.last_evidence["observer_errors"])

    def test_audit_snapshot_failure_cannot_skip_hooks_or_replace_forward_primary(self):
        for failed_forward in (False, True):
            with self.subTest(failed_forward=failed_forward):
                snapshot_error = RuntimeError("audit snapshot failed")
                original = KeyboardInterrupt("original forward")

                class BadEvents(list):
                    def __deepcopy__(self, _memo):
                        raise snapshot_error

                class BadWatch(Watch):
                    def __init__(self, *args):
                        super().__init__(*args)
                        self.first_layer_events = BadEvents()

                self.runtime.failure = original if failed_forward else None
                backend = self.make()
                expected = original if failed_forward else snapshot_error
                with patch.object(FakeObserver, "FinalNormObserver", BadWatch):
                    with self.assertRaises(type(expected)) as caught:
                        backend({}, questions())
                self.assertIs(caught.exception, expected)
                self.assertEqual(self.runtime.backbone.language_model.layers[0].hooks, [])
                self.assertEqual(self.runtime.backbone.language_model.norm.hooks, [])
                self.assertTrue(
                    any("audit snapshot after hook cleanup" in note for note in expected.__notes__)
                )

    def test_required_retention_failure_never_returns_candidate(self):
        for event in ("prepared", "chunk_returned", "raw_returned_before_validation", "completed"):
            for error_type in (OSError, KeyboardInterrupt):
                with self.subTest(event=event, error=error_type.__name__):
                    original = error_type("required retention failed")

                    def sink(current, _data):
                        if current == event:
                            raise original
                        if current == "failed":
                            raise RuntimeError("secondary retention failure")

                    backend = self.make(evidence_sink=sink)
                    with self.assertRaises(error_type) as caught:
                        backend({}, questions())
                    self.assertIs(caught.exception, original)
                    self.assertEqual(backend.last_evidence["status"], "failed")
                    self.assertTrue(
                        any("failure evidence retention" in note for note in original.__notes__)
                    )

    def test_exact_twenty_after_publication_or_final_identity_never_returns_candidate(self):
        for delay in ("completed_sink", "final_identity"):
            with self.subTest(delay=delay):
                self.clock.now = 0.0
                pending_identity_delay = {"active": False}

                def sink(event, _data):
                    if event == "completed":
                        if delay == "completed_sink":
                            self.clock.now = 20.0
                        else:
                            pending_identity_delay["active"] = True

                def identity():
                    if pending_identity_delay["active"]:
                        self.clock.now = 20.0

                self.runtime.identity_callback = identity
                backend = self.make(evidence_sink=sink)
                with self.assertRaisesRegex(TimeoutError, "cooperative20s") as caught:
                    backend({}, questions())
                self.assertEqual(caught.exception.delta_head_evidence["status"], "failed")
                self.assertIsNotNone(backend.last_evidence["raw_response"])
                self.runtime.identity_callback = None

    def test_changed_owned_delta_version_is_rejected_without_encoder(self):
        backend = self.make()
        backend.delta._version += 1
        with self.assertRaisesRegex(RuntimeError, "identity/version changed"):
            backend({}, questions())
        self.assertEqual(self.runtime.forward_calls, [])

    def test_original_outer_deadline_error_remains_primary(self):
        original = RuntimeError("owned full-call deadline")

        def deadline():
            raise original

        backend = self.make(deadline=deadline)
        with self.assertRaises(RuntimeError) as caught:
            backend({}, questions())
        self.assertIs(caught.exception, original)
        self.assertEqual(self.runtime.forward_calls, [])


if __name__ == "__main__":
    unittest.main()
