"""Unexecuted stdlib source fakes; no Torch/model/data or artifact loading.

Reuse only the previously authored fake runtime source. These tests cannot
qualify genuine BF16 CPU numerics, memory, latency, artifacts or model quality.
"""

import copy
import importlib.util
import struct
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


legacy = source(
    "_operation_only_fake_runtime_source",
    ROOT / "artifacts/jev-runtime/decision3-delta-head-backend-cpu5-20261024-01/test_backend.py",
)


class Matrix(legacy.Matrix):
    matmul_calls = []

    def clone(self):
        return Matrix(self.values, name=self.name, dtype=self.dtype, shape=self.shape)

    def contiguous(self):
        return self

    def view(self, dtype):
        assert dtype == "uint8"
        return Matrix(
            [list(b"".join(struct.pack("<f", value) for value in row)) for row in self.values],
            dtype=dtype,
        )

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Matrix(self.values[key], name=self.name, dtype=self.dtype)
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        assert isinstance(key, slice)
        self.values[key] = copy.deepcopy(value.values)
        self._version += 1

    def __matmul__(self, other):
        self.matmul_calls.append((self.shape, other.shape))
        value = super().__matmul__(other)
        return Matrix(value.values)

    def __add__(self, other):
        return Matrix(super().__add__(other).values)

    def masked_fill(self, mask, value):
        return Matrix(super().masked_fill(mask, value).values)


class Torch(legacy.FakeTorch):
    uint8 = "uint8"

    @staticmethod
    def zeros_like(tensor):
        return Matrix([[0.0] * tensor.shape[1] for _ in range(tensor.shape[0])])

    @staticmethod
    def equal(left, right):
        return (
            left.dtype == right.dtype and left.shape == right.shape and left.values == right.values
        )


class Watch(legacy.Watch):
    def __call__(self, *args):
        super().__call__(*args)
        self.features = Matrix(self.features.values)


class Observer(legacy.FakeObserver):
    FinalNormObserver = Watch


class Runtime(legacy.FakeRuntime):
    def logits(self, *args):
        return Matrix(super().logits(*args).values)


def capability(module, model):
    record = {
        "path": "/fake/root-accepted-operation-train-only",
        "sha256": "b" * 64,
        "size_bytes": 42,
    }
    return {
        "schema": module.CAPABILITY_SCHEMA,
        "status": "ROOT_ACCEPTED_OPERATION_TRAIN_ONLY_READOUT",
        "split": "train",
        "training_eligible": True,
        "nontraining": False,
        "profile": module.PROFILE,
        "alias": module.ALIAS,
        "owner_profile": module.OWNER_PROFILE,
        "only_head": "operation",
        "model": copy.deepcopy(model),
        "delta_ref": dict(record),
        "root_acceptance_ref": dict(record),
        "fit_capability": dict(record),
        "baseline_fp32_bytes_sha256": "c" * 64,
        "dev_used_for_model_selection": True,
        "optimizer_input_dev": False,
        "optimizer_input_sealed": False,
        "candidate_selection_before_dev_or_sealed": False,
    }


def offered(names, counts):
    return {
        name: {
            "tokens": [1] * (i + 1),
            "criteria": {"offered-" + str(j): "description" for j in range(count)},
        }
        for i, (name, count) in enumerate(zip(names, counts))
    }


class OperationBackendTests(unittest.TestCase):
    def setUp(self):
        self.module = source(
            "_held_operation_backend_fake", Path(__file__).with_name("backend_held.py")
        )
        self.module.SOURCE_ONLY = False
        self.module.CAPABILITY_SHA256 = "a" * 64
        self.runtime, self.now = Runtime(), types.SimpleNamespace(value=0.0)
        self.model = {"name": "d3-edge", "model_sha256": "d" * 64}
        self.delta = Matrix([[0.0] * 1024 for _ in range(255)])
        self.delta.values[0][0] = 100.0
        Matrix.matmul_calls = []

    def backend(self, **kwargs):
        return self.module.DeltaHeadBackend(
            legacy.FakeNative,
            Observer,
            self.runtime,
            Torch,
            self.model,
            self.delta,
            root_capability=kwargs.pop("root_capability", capability(self.module, self.model)),
            root_capability_sha256=kwargs.pop("root_capability_sha256", "a" * 64),
            deadline=lambda: None,
            clock=lambda: self.now.value,
            **kwargs,
        )

    def test_hold_and_pending_capability_precede_dependency_access(self):
        self.module.SOURCE_ONLY = True

        class Bomb:
            def __getattr__(self, name):
                raise AssertionError("dependency touched before hold: " + name)

        with self.assertRaisesRegex(RuntimeError, "SOURCE_ONLY"):
            self.module.DeltaHeadBackend(
                Bomb(),
                Bomb(),
                Bomb(),
                Bomb(),
                Bomb(),
                Bomb(),
                root_capability=Bomb(),
                root_capability_sha256=Bomb(),
                deadline=Bomb(),
            )
        self.module.SOURCE_ONLY = False
        self.module.CAPABILITY_SHA256 = None
        with self.assertRaisesRegex(RuntimeError, "SHA pending"):
            self.backend()

    def test_new_acceptance_rejects_shared_fit_cap_or_false_selection_provenance(self):
        for key, value in (
            ("schema", "root-committed-train-operation-fit-v1"),
            ("status", "ROOT_COMMITTED_TRAIN_OPERATION_FIT"),
            ("schema", "root-accepted-dsl01-train-delta-v1"),
            ("only_head", "all"),
            ("dev_used_for_model_selection", False),
            ("optimizer_input_dev", True),
            ("optimizer_input_sealed", True),
            ("candidate_selection_before_dev_or_sealed", True),
            ("fit_capability", None),
        ):
            cap = capability(self.module, self.model)
            cap[key] = value
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(RuntimeError, "new root-accepted operation-only"),
            ):
                self.backend(root_capability=cap)
        with self.assertRaisesRegex(RuntimeError, "capability SHA differs"):
            self.backend(root_capability_sha256="f" * 64)

    def test_reordered_operation_at_nonzero_row_with_non8_count_changes_only_it(self):
        names, counts = (
            ["click_target", "select_target", "operation", "type_text_target"],
            [3, 1, 5, 4],
        )
        result = self.backend()({}, offered(names, counts))
        self.assertEqual(list(result["answers"]), names)
        self.assertEqual(result["answers"]["operation"]["choice"], "offered-0")
        for name, count in zip(names, counts):
            if name != "operation":
                self.assertEqual(result["answers"][name]["choice"], "offered-" + str(count - 1))
        proof = self.runtime.forward_calls
        self.assertEqual([len(call[0]) for call in proof], [4])
        self.assertEqual(Matrix.matmul_calls, [((1, 1024), (1024, 255))])

    def test_target_logits_are_bitwise_original_including_negative_zero(self):
        counts, names = [2, 3], ["click_target", "operation"]
        base = Matrix(
            [[-0.0, 1.0] + [float("-inf")] * 253, [0.0, 1.0, 2.0] + [float("-inf")] * 252]
        )
        features = Matrix([[1.0] + [0.0] * 1023 for _ in names])
        before = base.view(Torch.uint8).tolist()
        _values, proof = self.module.project(Torch, base, features, self.delta, counts, names)
        self.assertEqual(base.view(Torch.uint8).tolist(), before)
        self.assertTrue(proof["target_logits_bitwise_unchanged"])
        self.assertEqual(
            struct.pack("<f", proof["adapted_legal_logits"][0][0]), struct.pack("<f", -0.0)
        )
        self.assertEqual(proof["base_legal_logits"][0], proof["adapted_legal_logits"][0])
        self.assertEqual(proof["delta_legal_logits"][0], [0.0, 0.0])

    def test_absent_operation_does_not_project_or_add_a_fake_question(self):
        backend = self.backend()
        names = ["select_target", "click_target"]
        result = backend({}, offered(names, [2, 1]))
        self.assertEqual(list(result["answers"]), names)
        self.assertEqual(Matrix.matmul_calls, [])
        proof = backend.last_evidence["chunks"][0]["projection"]
        self.assertEqual(proof["adapted_head_ids"], [])
        self.assertEqual(proof["base_legal_logits"], proof["adapted_legal_logits"])

    def test_real1_to4_rows_mixed_counts_and_genuine_four_plus_one_tail(self):
        for rows in range(1, 5):
            self.runtime = Runtime()
            backend = self.backend()
            names = (
                ["operation"]
                if rows == 1
                else ["click_target", "operation", "select_target", "type_text_target"][:rows]
            )
            counts = list(range(1, rows + 1))
            backend({}, offered(names, counts))
            chunk = backend.last_evidence["chunks"][0]
            self.assertEqual(chunk["rows"], rows)
            self.assertEqual(chunk["first_layer_events"], [[rows, rows, 1024]])
            self.assertEqual(chunk["final_norm_events"], [[rows, rows, 1024]])
            self.assertEqual(chunk["projection"]["ordered_head_ids"], names)
        self.runtime = Runtime()
        backend = self.backend()
        names = ["target-a", "target-b", "target-c", "target-d", "operation"]
        backend({}, offered(names, [1, 2, 3, 4, 1]))
        self.assertEqual([row["rows"] for row in backend.last_evidence["chunks"]], [4, 1])
        self.assertEqual(backend.last_evidence["chunks"][0]["projection"]["adapted_head_ids"], [])
        self.assertEqual(
            backend.last_evidence["chunks"][1]["projection"]["adapted_head_ids"], ["operation"]
        )

    def test_required_sink_failure_rethrows_exact_primary_and_removes_hooks(self):
        for event in ("prepared", "chunk_returned", "raw_returned_before_validation", "completed"):
            primary = KeyboardInterrupt("required " + event)

            def sink(name, data):
                if name == event:
                    raise primary
                if name == "failed":
                    raise OSError("secondary failed retention")

            backend = self.backend(evidence_sink=sink)
            with self.assertRaises(KeyboardInterrupt) as caught:
                backend({}, offered(["operation", "click_target"], [3, 2]))
            self.assertIs(caught.exception, primary)
            self.assertEqual(self.runtime.backbone.language_model.norm.hooks, [])
            self.assertEqual(self.runtime.backbone.language_model.layers[0].hooks, [])
            self.assertTrue(any("failure evidence retention" in note for note in primary.__notes__))

    def test_exact20_in_final_retention_cannot_authorize_a_result(self):
        def sink(event, data):
            if event == "completed":
                self.now.value = 20.0

        backend = self.backend(evidence_sink=sink)
        with self.assertRaises(TimeoutError):
            backend({}, offered(["click_target", "operation"], [1, 3]))
        self.assertEqual(backend.last_evidence["status"], "failed")


if __name__ == "__main__":
    unittest.main()
