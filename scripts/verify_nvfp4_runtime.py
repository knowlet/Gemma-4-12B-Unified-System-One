"""Replay the released populations using only the exact bundled S1 wheel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.wheel.resolve()))

    from release_training import file_sha256, load_release_data, write_json

    import s1
    from s1.benchmark import evaluate
    from s1.unified import UnifiedDecisionModel

    if not s1.__file__.startswith(str(args.wheel.resolve()) + "/"):
        raise RuntimeError("verification must import S1 exclusively from the packaged wheel")
    package = args.model
    evaluation = json.loads((package / "evaluation.json").read_text())
    data, identities = load_release_data(args.data)
    if identities != evaluation["datasets"]:
        raise ValueError("runtime replay datasets differ from the evaluated artifact")
    model = UnifiedDecisionModel(
        str(package),
        device="cuda",
        precision="bfloat16",
        quantization="nvfp4",
        attn_implementation="sdpa",
        max_context=4096,
    )
    model.name = "packaged-runtime-nvfp4"
    result = {
        "status": "running",
        "population_complete": False,
        "conversion_manifest_sha256": file_sha256(package / "conversion-manifest.json"),
        "evaluation_sha256": file_sha256(package / "evaluation.json"),
        "runtime_wheel": {"filename": args.wheel.name, "sha256": file_sha256(args.wheel)},
        "s1_import_path": s1.__file__,
        "reports": {},
        "probability_absolute_tolerance": 1e-6,
    }
    for split in ("test", "media", "regression_text"):
        print(f"Bundled wheel verification: {split}", flush=True)
        report = evaluate(model, data[split], warmup=1)
        reference = evaluation["reports"][split]
        if report["coverage"] != 1.0 or report["warmup_errors"]:
            raise RuntimeError("bundled runtime did not complete every case")
        if report["case_ids"] != reference["case_ids"]:
            raise ValueError("runtime replay case order differs from evaluation")
        max_difference = 0.0
        for actual, expected in zip(report["records"], reference["records"], strict=True):
            if (actual["case_id"], actual["question_id"]) != (
                expected["case_id"],
                expected["question_id"],
            ):
                raise ValueError("runtime replay decision identity mismatch")
            left, right = actual["probabilities"], expected["probabilities"]
            if list(left) != list(right) or max(left, key=left.get) != max(right, key=right.get):
                raise ValueError("bundled runtime changed candidate ordering or prediction")
            max_difference = max(max_difference, max(abs(left[key] - right[key]) for key in left))
        if max_difference > result["probability_absolute_tolerance"]:
            raise ValueError("bundled runtime candidate probabilities drifted from evaluation")
        result["reports"][split] = {key: report[key] for key in ("coverage", "counts", "metrics")}
        result["reports"][split]["max_absolute_probability_difference"] = max_difference
    result.update(status="ok", population_complete=True)
    write_json(package / "runtime-verification.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
