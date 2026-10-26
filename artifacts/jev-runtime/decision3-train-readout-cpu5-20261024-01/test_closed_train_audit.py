"""Invented stdlib corruption cases for root acceptance; no actual data/models."""

import array
import copy
import hashlib
import importlib.util
import math
import struct
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "_closed_train_audit_fake", HERE / "root-closed-train-chunk-audit-01.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def npy_bytes(*, descr="<f4", shape=(4, 1024), fortran=False, nonfinite=False):
    header = (
        repr({"descr": descr, "fortran_order": fortran, "shape": shape}).encode("latin1") + b"\n"
    )
    values = array.array("f", [0.0]) * math.prod(shape)
    if nonfinite:
        values[-1] = float("nan")
    payload = values.tobytes()
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + payload, payload


def uniform_fixture():
    features = array.array("f", [0.0]) * (4 * 1024)
    baseline = array.array("f", [0.0]) * (255 * 1024)
    questions = {}
    answers = {}
    lengths = {}
    for i, (head, count) in enumerate(
        zip(("operation", "click_target", "type_text_target", "select_target"), (8, 1, 2, 3))
    ):
        criteria = {f"original-{k}": "invented" for k in range(count)}
        questions[head] = {"type": "choice", "criteria": criteria}
        probabilities = {key: 1 / count for key in criteria}
        answers[head] = {
            "type": "choice",
            "choice": next(iter(criteria)),
            "probabilities": probabilities,
            "confidence": 1.0 if count == 1 else 0.0,
        }
        lengths[head] = 10 + i
    raw = {
        "model": "d3-edge",
        "answers": answers,
        "usage": {"input_tokens": sum(lengths.values()), "output_tokens": 0},
    }
    return features, baseline, {"questions": questions}, raw, lengths


class RootAcceptanceCorruptionTests(unittest.TestCase):
    def test_source_hold_precedes_args_and_all_actual_reads(self):
        with (
            patch.object(AUDIT, "SOURCE_ONLY", True),
            patch.object(AUDIT, "fetch", side_effect=AssertionError("must not read")),
        ):
            with self.assertRaisesRegex(ValueError, "SOURCE_ONLY"):
                AUDIT.main(["--unknown"])

    def test_npy_checks_actual_header_and_verified_raw_digest(self):
        blob, payload = npy_bytes()
        ref = {
            "format": "npy_no_pickle",
            "dtype": "float32",
            "shape": [4, 1024],
            "FP32_bytes_sha256": hashlib.sha256(payload).hexdigest(),
        }
        with patch.object(AUDIT, "verify", return_value=(blob, ref)):
            values, _, digest = AUDIT.npy(ref, None, (4, 1024))
            self.assertEqual(len(values), 4096)
            self.assertEqual(digest, ref["FP32_bytes_sha256"])
        for mutation in ("dtype", "fortran", "shape", "nan", "digest", "truncated"):
            raw = blob
            changed = copy.deepcopy(ref)
            if mutation == "dtype":
                raw, _ = npy_bytes(descr="<f8")
            elif mutation == "fortran":
                raw, _ = npy_bytes(fortran=True)
            elif mutation == "shape":
                raw, _ = npy_bytes(shape=(4, 1023))
            elif mutation == "nan":
                raw, _ = npy_bytes(nonfinite=True)
                changed.pop("FP32_bytes_sha256")
            elif mutation == "digest":
                changed["FP32_bytes_sha256"] = "0" * 64
            else:
                raw = blob[:-4]
            with (
                self.subTest(mutation=mutation),
                patch.object(AUDIT, "verify", return_value=(raw, changed)),
                self.assertRaises(ValueError),
            ):
                AUDIT.npy(changed, None, (4, 1024))

    def test_all_legal_slots_including_singleton_and_order_are_checked(self):
        args = uniform_fixture()
        result = AUDIT.witness(*args)
        self.assertEqual([row["legal_count"] for row in result], [8, 1, 2, 3])
        self.assertTrue(all(row["max_absolute_error"] < 1e-12 for row in result))
        for mutation in ("slot", "argmax", "order", "singleton_confidence", "usage"):
            features, baseline, request, raw, lengths = uniform_fixture()
            if mutation == "slot":
                raw["answers"]["operation"]["probabilities"]["original-0"] += 1e-3
                raw["answers"]["operation"]["probabilities"]["original-1"] -= 1e-3
            elif mutation == "argmax":
                raw["answers"]["operation"]["choice"] = "original-1"
            elif mutation == "order":
                raw["answers"]["operation"]["probabilities"] = dict(
                    reversed(list(raw["answers"]["operation"]["probabilities"].items()))
                )
            elif mutation == "singleton_confidence":
                raw["answers"]["click_target"]["confidence"] = 0.0
            else:
                raw["usage"]["input_tokens"] += 1
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                AUDIT.witness(features, baseline, request, raw, lengths)

    def test_fixed_relative_term_is_retained(self):
        # A valid mixed-tolerance result can exceed the absolute1e-6 term.
        for epsilon, passes in ((2e-6, True), (4e-6, False)):
            args = list(uniform_fixture())
            raw = args[3]
            answer = raw["answers"]["operation"]
            p = answer["probabilities"]
            p["original-0"] += epsilon
            p["original-1"] -= epsilon
            answer["confidence"] = 1 + sum(v * math.log(v) for v in p.values()) / math.log(len(p))
            if passes:
                AUDIT.witness(*args)
            else:
                with self.assertRaisesRegex(ValueError, "legal-slot witness"):
                    AUDIT.witness(*args)

    def test_role_boundary_refuses_labels_authoring_dev_and_sealed_before_open(self):
        for path in (
            AUDIT.TRAIN / "labels.jsonl",
            AUDIT.DSL_ROOT / "dsl.py",
            AUDIT.TRAIN.parent / "dev/requests.jsonl",
            AUDIT.TRAIN.parent / "sealed/requests.jsonl",
        ):
            with patch.object(Path, "open", side_effect=AssertionError("must not open")):
                with self.assertRaisesRegex(ValueError, "audit refuses"):
                    AUDIT.fetch(path)


if __name__ == "__main__":
    unittest.main()
