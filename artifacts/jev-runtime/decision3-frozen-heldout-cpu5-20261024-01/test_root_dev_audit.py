"""Bounded stdlib corruption checks using invented values; no actual IO."""

import array
import copy
import importlib.util
import math
import unittest
from pathlib import Path
from unittest.mock import patch


def load():
    p = Path(__file__).with_name("root-closed-dev-chunk-audit-held.py")
    spec = importlib.util.spec_from_file_location("_root_dev_audit_fakes", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class AuditCorruptionTests(unittest.TestCase):
    def setUp(self):
        self.m = load()

    def test_hold_precedes_argument_parsing(self):
        with patch.object(self.m.argparse, "ArgumentParser", side_effect=AssertionError("parser")):
            with self.assertRaisesRegex(ValueError, "SOURCE_ONLY"):
                self.m.main([])

    def test_all_path_guards_refuse_labels_train_sealed_and_trained_delta(self):
        for p in [
            self.m.DSL / "labels.jsonl",
            self.m.DSL_ROOT / "corpus/data/train/requests.jsonl",
            self.m.DSL_ROOT / "corpus/data/sealed/requests.jsonl",
            self.m.OWNER / "actual-19/fit-head/delta-pending.npy",
        ]:
            with self.subTest(path=str(p)), self.assertRaises(ValueError):
                self.m.local(p)

    def test_dev_taint_is_never_training_or_labels(self):
        value = {
            "split": "dev",
            "role": "dev_feature_collector",
            "labels": None,
            "training_eligible": False,
        }
        self.m.dev_taint(value)
        for change in (
            {"split": "train"},
            {"role": "train_collector"},
            {"labels": {}},
            {"training_eligible": True},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.m.dev_taint({**value, **change})

    def fixture(self):
        h = array.array("f", [0.0]) * 4096
        w = array.array("f", [0.0]) * (255 * 1024)
        h[0] = 1.0
        w[1024] = 1.0
        p = math.exp(1) / (1 + math.exp(1))
        request = {
            "questions": {
                "operation": {
                    "type": "choice",
                    "criteria": {"second-named": "A", "first-named": "B"},
                }
            }
        }
        ps = {"second-named": 1 - p, "first-named": p}
        conf = 1 + sum(v * math.log(v) for v in ps.values()) / math.log(2)
        raw = {
            "model": "d3-edge",
            "answers": {
                "operation": {
                    "type": "choice",
                    "choice": "first-named",
                    "probabilities": ps,
                    "confidence": conf,
                }
            },
            "usage": {"input_tokens": 5, "output_tokens": 0},
        }
        return h, w, request, raw, {"operation": 5}

    def test_full_legal_witness_rejects_feature_distribution_and_entropy_corruption(self):
        h, w, request, raw, lengths = self.fixture()
        self.assertEqual(len(self.m.witness(h, w, request, raw, lengths)), 1)
        altered = h[:]
        altered[0] = 2.0
        with self.assertRaises(ValueError):
            self.m.witness(altered, w, request, raw, lengths)
        for kind in ["distribution", "entropy", "order", "argmax"]:
            changed = copy.deepcopy(raw)
            answer = changed["answers"]["operation"]
            if kind == "distribution":
                answer["probabilities"] = {"second-named": 0.5, "first-named": 0.5}
            elif kind == "entropy":
                answer["confidence"] = 0.0
            elif kind == "order":
                answer["probabilities"] = dict(reversed(list(answer["probabilities"].items())))
            else:
                answer["choice"] = "second-named"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.m.witness(h, w, request, changed, lengths)


if __name__ == "__main__":
    unittest.main()
