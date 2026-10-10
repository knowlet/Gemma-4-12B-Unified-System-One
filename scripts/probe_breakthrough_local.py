"""Offline MPS native prompt probe; separate from the frozen A100 experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    # This probe cannot download checkpoints or contact an external service.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["USE_TF"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from breakthrough_training import capture, write_json
    from prepare_mlx_validation import checkpoint_identity

    import s1
    from s1.contracts import answer_from_probabilities
    from s1.evaluation.datasets import load_cases
    from s1.unified import UnifiedDecisionModel

    if not torch.backends.mps.is_available():
        raise RuntimeError("local MPS required; no cloud or CPU-model fallback")
    root = Path(__file__).resolve().parents[1]
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    sources = [
        Path(__file__),
        root / "scripts/breakthrough_prompt.py",
        root / "scripts/breakthrough_training.py",
        root / "scripts/prepare_mlx_validation.py",
        *Path(s1.__file__).parent.rglob("*.py"),
    ]
    hashes = {}
    for path in sources:
        name = str(path.relative_to(root))
        data = path.read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        saved = destination / "sources" / f"{name}.txt"
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(data)
    checkpoint = root / "artifacts/release/checkpoint"
    cases_path = root / "examples/benchmarks/mps.jsonl"
    cases = load_cases(cases_path)
    if len(cases) != 8 or sum(len(c.request.questions) for c in cases) != 29:
        raise ValueError("complete native fixture population required")
    receipt = {
        "status": "running",
        "scope": "Local native prompt smoke only; not frozen A100 profiles, fresh384, public231, complete release regression, coherence or official ranking.",
        "device": "mps",
        "precision": "bfloat16",
        "attention": "sdpa",
        "execution": "Sequential independent native forward per question; no fitting, cache, compile or probability constraints.",
        "temperature": 2.7830344470383452,
        "profiles_declared": ["current", "user_question"],
        "cases": 8,
        "questions": 29,
        "data_sha256": hashlib.sha256(cases_path.read_bytes()).hexdigest(),
        "source_files": hashes,
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers")},
        "profiles": [],
        "official_rank": None,
        "cloud_calls": 0,
        "training_executed": False,
    }
    write_json(destination / "receipt.json", receipt)
    try:
        receipt["checkpoint_files"] = checkpoint_identity(checkpoint)
        receipt["processor_files"] = {
            p.name: {
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "size_bytes": p.stat().st_size,
            }
            for p in sorted(checkpoint.iterdir())
            if p.is_file() and p.suffix in (".json", ".model", ".jinja", ".txt")
        }
        model = UnifiedDecisionModel(
            str(checkpoint),
            device="mps",
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            temperature=1.0,
        )
        model.lm.eval()
        for prompt in receipt["profiles_declared"]:
            saved, correct = [], 0
            for case in cases:
                torch.mps.synchronize()
                start = time.perf_counter()
                with torch.inference_mode():
                    rows = capture(model, case.request, prompt)
                torch.mps.synchronize()
                answers, raw = {}, []
                for row in rows:
                    question = row["question"]
                    logits = row["logits"].float().cpu()
                    if not torch.isfinite(logits).all():
                        raise ValueError("nonfinite native logits")
                    answers[question.id] = answer_from_probabilities(
                        question, (logits / receipt["temperature"]).softmax(-1).tolist()
                    )
                    probabilities = answers[question.id]["probabilities"]
                    predicted = max(probabilities, key=probabilities.__getitem__)
                    correct += int(predicted == case.gold[question.id])
                    raw.append(
                        {
                            "question_id": question.id,
                            "labels": question.labels(),
                            "logits": logits.tolist(),
                            "sequence_tokens": row["sequence_tokens"],
                        }
                    )
                saved.append(
                    {
                        "case_id": case.id,
                        "modalities": [m.type for m in case.request.media],
                        "gold": case.gold,
                        "answers": answers,
                        "raw_logits": raw,
                        "seconds": time.perf_counter() - start,
                    }
                )
                write_json(destination / f"{prompt}.json", saved)
                print(
                    json.dumps(
                        {
                            "prompt": prompt,
                            "case": case.id,
                            "completed_questions": sum(len(r["answers"]) for r in saved),
                        }
                    ),
                    flush=True,
                )
            entry = {
                "name": prompt,
                "status": "completed",
                "cases": len(saved),
                "questions": sum(len(r["answers"]) for r in saved),
                "n_correct": correct,
            }
            receipt["profiles"].append(entry)
            write_json(destination / "receipt.json", receipt)
            print(json.dumps(entry), flush=True)
        receipt["status"] = "completed"
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(destination / "receipt.json", receipt)


if __name__ == "__main__":
    main()
