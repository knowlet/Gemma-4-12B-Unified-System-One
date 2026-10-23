"""Bounded stdlib fakes: no Torch/model/source payload imports or actual jobs."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import math
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), module.__dict__)
    return module


actor = load("_test_feature_replay_actor", HERE / "actor.py")
observer = load("_test_feature_replay_observer", HERE / "observer.py")
native = load("_test_feature_replay_native", actor.NATIVE / "actor.py")


class Scalar:
    def __init__(self, value):
        self.value = value

    def all(self):
        return self

    def item(self):
        return self.value


class Tensor:
    """Small list tensor fake implementing only the observed FP32 readout path."""

    def __init__(self, data, dtype="float32", shape=None):
        self.data = copy.deepcopy(data)
        self.dtype, self.device = dtype, types.SimpleNamespace(type="cpu")
        self._version = 0
        self.shape = shape or self._shape(data)

    @staticmethod
    def _shape(data):
        return (len(data),) + Tensor._shape(data[0]) if isinstance(data, list) and data else ()

    def __getitem__(self, key):
        if key == (slice(None), -1, slice(None)):
            return Tensor([row[-1] for row in self.data], self.dtype)
        if key == (slice(None), None):
            return Tensor([[item] for item in self.data], self.dtype)
        if key is None:
            return Tensor([self.data], self.dtype)
        raise AssertionError("unexpected slice: " + repr(key))

    def detach(self):
        return self

    def to(self, *, device, dtype):
        if device != "cpu":
            raise AssertionError("nonCPU replay")
        return Tensor(self.data, dtype)

    def clone(self):
        return Tensor(self.data, self.dtype)

    def float(self):
        return Tensor(self.data)

    def cpu(self):
        return self

    def tolist(self):
        return copy.deepcopy(self.data)

    def data_ptr(self):
        return id(self.data)

    @property
    def T(self):
        return Tensor([list(row) for row in zip(*self.data)])

    def __matmul__(self, other):
        return Tensor(
            [
                [sum(a * b for a, b in zip(row, column)) for column in zip(*other.data)]
                for row in self.data
            ]
        )

    def __ge__(self, other):
        return Tensor([[code >= limit[0] for code in self.data[0]] for limit in other.data])

    def masked_fill(self, mask, value):
        return Tensor(
            [
                [value if masked else v for v, masked in zip(row, flags)]
                for row, flags in zip(self.data, mask.data)
            ]
        )

    def __truediv__(self, value):
        return Tensor([[item / value for item in row] for row in self.data])


class Scope:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class Torch:
    bfloat16, float32 = "bfloat16", "float32"
    cuda = types.SimpleNamespace(is_initialized=lambda: False)

    @staticmethod
    def isfinite(tensor):
        def flat(data):
            for item in data:
                if isinstance(item, list):
                    yield from flat(item)
                else:
                    yield item

        return Scalar(all(math.isfinite(v) for v in flat(tensor.data)))

    @staticmethod
    def inference_mode():
        return Scope()

    @staticmethod
    def as_tensor(data, *, device):
        return Tensor(data)

    @staticmethod
    def arange(count, *, device):
        return Tensor(list(range(count)))

    @staticmethod
    def softmax(tensor, *, dim):
        rows = []
        for row in tensor.data:
            top = max(row)
            exps = [math.exp(v - top) for v in row]
            total = sum(exps)
            rows.append([v / total for v in exps])
        return Tensor(rows)


def full_hidden(rows=3, width=2):
    # Last position differs from the preceding position and between rows.
    return Tensor(
        [[[float(i + j)] + [0.0] * 1023 for j in range(width)] for i in range(rows)], "bfloat16"
    )


def fixture_questions():
    return json.loads((HERE / "fixture.json").read_text())["request"]["questions"]


def raw_for(questions, rows):
    return {
        "answers": {
            key: {
                "probabilities": dict(zip(q["criteria"], row)),
                "choice": list(q["criteria"])[max(range(len(row)), key=row.__getitem__)],
            }
            for (key, q), row in zip(questions.items(), rows)
        }
    }


class ObservationTests(unittest.TestCase):
    def witness(self):
        witness = observer.FinalNormObserver(Torch, 3, 2)
        hidden = full_hidden()
        self.assertIsNone(witness.first_layer(None, (hidden,)))
        self.assertIsNone(witness(None, (), hidden))
        return witness, hidden

    def test_last_position_copy_only_and_none_hooks(self):
        witness, hidden = self.witness()
        original = copy.deepcopy(hidden.data)
        self.assertEqual(witness.features.shape, (3, 1024))
        self.assertEqual([row[0] for row in witness.features.data], [1, 2, 3])
        witness.features.data[0][0] = 999
        self.assertEqual(hidden.data, original)

    def test_wrong_partial_rows_refused(self):
        with self.assertRaisesRegex(RuntimeError, "shape differs"):
            observer.FinalNormObserver(Torch, 3, 2)(None, (), full_hidden(4))

    def test_wrong_dtype_and_gpu_refused(self):
        for which in ("dtype", "device"):
            hidden = full_hidden()
            if which == "dtype":
                hidden.dtype = "float32"
            else:
                hidden.device.type = "cuda"
            with self.assertRaises(RuntimeError):
                observer.FinalNormObserver(Torch, 3, 2)(None, (), hidden)

    def test_nonfinite_feature_refused(self):
        hidden = full_hidden()
        hidden.data[0][-1][0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "finite"):
            observer.FinalNormObserver(Torch, 3, 2)(None, (), hidden)

    def test_missing_and_extra_encoder_events_refused(self):
        witness = observer.FinalNormObserver(Torch, 3, 2)
        with self.assertRaises(RuntimeError):
            witness.require_single_pass()
        witness, hidden = self.witness()
        with self.assertRaises(RuntimeError):
            witness.first_layer(None, (hidden,))
        with self.assertRaises(RuntimeError):
            witness(None, (), hidden)

    def test_replay_mask_excludes_huge_illegal_logits_without_extra_pass(self):
        witness, _ = self.witness()
        # Illegal codes would dominate without the original legal mask.
        weights = [[float(code) if code < 4 else 1000.0] + [0.0] * 1023 for code in range(255)]
        runtime = types.SimpleNamespace(readout=Tensor(weights), temperature=1.0, device="cpu")
        before = copy.deepcopy(runtime.readout.data)
        rows = observer.replay_probabilities(runtime, Torch, witness, (2, 3, 4))
        expected_first = [1 / (1 + math.e), math.e / (1 + math.e)]
        self.assertAlmostEqual(rows[0][0], expected_first[0])
        self.assertAlmostEqual(rows[0][1], expected_first[1])
        self.assertEqual(list(map(len, rows)), [2, 3, 4])
        self.assertEqual(runtime.readout.data, before)
        self.assertEqual(len(witness.first_layer_events), 1)
        self.assertEqual(len(witness.norm_events), 1)

    def test_nonunit_temperature_refused(self):
        witness, _ = self.witness()
        runtime = types.SimpleNamespace(readout=Tensor([], shape=(255, 1024)), temperature=0.5)
        with self.assertRaises(RuntimeError):
            observer.replay_probabilities(runtime, Torch, witness, (2, 3, 4))

    def test_baseline_version_change_detectable(self):
        weight, parameter = Tensor([[0.0]]), Tensor([[1.0]])
        runtime = types.SimpleNamespace(
            readout=weight,
            encoder=object(),
            backbone=types.SimpleNamespace(
                parameters=lambda: [parameter], language_model=types.SimpleNamespace(norm=object())
            ),
        )
        before = observer.baseline_identity(runtime)
        parameter._version += 1
        self.assertNotEqual(before, observer.baseline_identity(runtime))


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.questions = fixture_questions()
        self.rows = [[0.4, 0.6], [0.2, 0.5, 0.3], [0.1, 0.2, 0.3, 0.4]]
        self.raw = raw_for(self.questions, self.rows)

    def test_all_legal_probabilities_and_no_semantic_gold(self):
        result = observer.compare_distributions(self.raw, self.questions, self.rows)
        self.assertTrue(result["all_legal_probabilities_matched"])
        self.assertFalse(result["semantic_correctness_evaluated"])

    def test_declared_tolerance_boundary(self):
        for delta, accepted in ((2e-6, True), (2e-5, False)):
            rows = copy.deepcopy(self.rows)
            rows[0][0] += delta
            rows[0][1] -= delta
            if accepted:
                observer.compare_distributions(self.raw, self.questions, rows)
            else:
                with self.assertRaisesRegex(RuntimeError, "tolerance"):
                    observer.compare_distributions(self.raw, self.questions, rows)

    def test_missing_probability_or_partial_rows_refused(self):
        for rows in (self.rows[:2], [[1.0], *self.rows[1:]]):
            with self.assertRaises(RuntimeError):
                observer.compare_distributions(self.raw, self.questions, rows)

    def test_native_argmax_mismatch_refused(self):
        self.raw["answers"]["envelope_record"]["choice"] = "memo"
        with self.assertRaisesRegex(RuntimeError, "argmax"):
            observer.compare_distributions(self.raw, self.questions, self.rows)

    def test_original_label_order_refused(self):
        probabilities = self.raw["answers"]["drawer_record"]["probabilities"]
        self.raw["answers"]["drawer_record"]["probabilities"] = dict(
            reversed(list(probabilities.items()))
        )
        with self.assertRaisesRegex(RuntimeError, "order"):
            observer.compare_distributions(self.raw, self.questions, self.rows)

    def test_tie_uses_first_original_label(self):
        rows = [[0.5, 0.5], *self.rows[1:]]
        observer.compare_distributions(raw_for(self.questions, rows), self.questions, rows)


class TimingTests(unittest.TestCase):
    def run_case(
        self,
        *,
        budget=20,
        initial=0,
        operation_error=None,
        cleanup_advance=0,
        cleanup_error=None,
        exit_advance=0,
        exit_error=None,
        write_advance=0,
        write_error=None,
    ):
        clock = [initial]
        record = {"completed": False}
        writes = []

        class Deadline:
            def __init__(self, remaining, *_):
                record["remaining"] = remaining

            def __enter__(self):
                return self

            def __exit__(self, *_):
                clock[0] += exit_advance
                if exit_error is not None:
                    raise exit_error
                return False

        def elapsed(began, limit, stage):
            value = clock[0] - began
            if not 0 <= value <= limit:
                raise RuntimeError(stage + " late")
            return value

        def operation():
            if operation_error is not None:
                raise operation_error
            record["completed"] = True
            return "native-result"

        def cleanup():
            clock[0] += cleanup_advance
            if cleanup_error is not None:
                raise cleanup_error

        def finalize(primary):
            writes.append(record["completed"])
            clock[0] += write_advance
            if write_error is not None:
                if primary is not None:
                    primary.add_note("failure evidence: " + repr(write_error))
                else:
                    raise write_error

        def demote(error):
            record["completed"] = False
            record["error"] = error

        helper = types.SimpleNamespace(Deadline=Deadline, ensure_elapsed=elapsed)
        try:
            with patch.object(actor.time, "monotonic", side_effect=lambda: clock[0]):
                result = actor.transaction(
                    native,
                    helper,
                    Path("/unused"),
                    "test",
                    0,
                    budget,
                    operation,
                    cleanup,
                    finalize,
                    demote,
                )
            return record, writes, result, None
        except BaseException as error:
            return record, writes, None, error

    def test_complete_call_includes_hook_cleanup_write_and_watchdog_exit(self):
        record, writes, result, error = self.run_case(
            cleanup_advance=2, write_advance=3, exit_advance=4
        )
        self.assertIsNone(error)
        self.assertEqual(result, ("native-result", 9))
        self.assertTrue(record["completed"])
        self.assertEqual(writes, [True])

    def test_call_cleanup_expiry_demotes_after_candidate_write(self):
        record, writes, _, error = self.run_case(exit_advance=21)
        self.assertIsInstance(error, RuntimeError)
        self.assertFalse(record["completed"])
        self.assertEqual(writes, [True, False])

    def test_whole120_last_write_expiry_demotes(self):
        record, writes, _, error = self.run_case(budget=120, write_advance=121)
        self.assertIsInstance(error, RuntimeError)
        self.assertFalse(record["completed"])
        self.assertEqual(writes, [True, False])

    def test_execute_entry_preflight_is_not_reset(self):
        record, _, _, error = self.run_case(budget=120, initial=119, exit_advance=2)
        self.assertEqual(record["remaining"], 1)
        self.assertIsNotNone(error)
        self.assertFalse(record["completed"])

    def test_cleanup_raise_demotes(self):
        secondary = RuntimeError("hook remove")
        record, _, _, error = self.run_case(cleanup_error=secondary)
        self.assertIs(error, secondary)
        self.assertFalse(record["completed"])

    def test_watchdog_cleanup_raise_demotes(self):
        secondary = RuntimeError("timer cancel")
        record, _, _, error = self.run_case(exit_error=secondary)
        self.assertIs(error, secondary)
        self.assertFalse(record["completed"])

    def test_original_primary_survives_hook_timer_and_failure_write_errors(self):
        primary = KeyboardInterrupt("original native failure")
        record, _, _, error = self.run_case(
            operation_error=primary,
            cleanup_error=RuntimeError("hook"),
            exit_error=RuntimeError("timer"),
            write_error=OSError("disk"),
        )
        self.assertIs(error, primary)
        self.assertFalse(record["completed"])
        notes = " ".join(primary.__notes__)
        self.assertIn("hook", notes)
        self.assertIn("timer", notes)
        self.assertIn("disk", notes)

    def test_last_write_error_demotes_and_propagates(self):
        failure = OSError("last write")
        record, _, _, error = self.run_case(write_error=failure)
        self.assertIs(error, failure)
        self.assertFalse(record["completed"])


class StaticContractTests(unittest.TestCase):
    def test_fixture_one_unscored_request_and_three_original_heads(self):
        fixture = json.loads((HERE / "fixture.json").read_text())
        self.assertIsNone(fixture["semantic_expected_answers"])
        self.assertEqual(fixture["fixture_request_count"], 1)
        self.assertEqual(
            [len(q["criteria"]) for q in fixture["request"]["questions"].values()], [2, 3, 4]
        )

    def test_observer_has_no_framework_import_or_encoder_call(self):
        tree = ast.parse((HERE / "observer.py").read_text())
        imports = [
            node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        self.assertEqual(len(imports), 2)  # future annotations + math
        calls = [
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        self.assertNotIn("system_one", calls)
        self.assertNotIn("encoder", calls)
        self.assertNotIn("backbone", calls)

    def test_model_frameworks_not_imported(self):
        self.assertFalse(actor.MODEL_MODULES.intersection(sys.modules))


if __name__ == "__main__":
    unittest.main()
