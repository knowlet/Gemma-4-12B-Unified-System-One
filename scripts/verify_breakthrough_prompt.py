"""Local-only native processor evidence; never load model weights or run inference.

Compare the existing and USER-question preparation on each independent native
fixture question. Persist tensor identities and sequence/mask contents, not
predicted model quality or performance.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from collections import Counter
from pathlib import Path

SEQUENCE_KEYS = {"input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids"}
ROOT = Path(__file__).resolve().parents[1]


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def json_bytes(value):
    return (json.dumps(value, allow_nan=False, sort_keys=True, ensure_ascii=False) + "\n").encode()


def tensor_evidence(value, *, values=False):
    """Retain exact native bytes without serializing large image feature arrays."""
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError("native processor output must contain tensors")
    value = value.detach().cpu().contiguous()
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError("nonfinite native processor tensor")
    data = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    result = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "bytes": len(data),
        "sha256": sha256(data),
    }
    if values:
        result["values"] = value.tolist()
    return result


class ObservedProcessor:
    """Delegate all work to the loaded real processor; retain its native prefix."""

    def __init__(self, processor):
        self.processor, self.tokenizer = processor, processor.tokenizer
        self.prefix = self.messages = self.rendered = None

    def apply_chat_template(self, messages, **options):
        self.messages = messages
        self.rendered = self.processor.apply_chat_template(messages, **options)
        return self.rendered

    def __call__(self, **options):
        result = self.processor(**options)
        self.prefix = {key: value.detach().clone() for key, value in result.items()}
        return result


def _require(condition, error):
    if not condition:
        raise ValueError(error)


def capture_preparation(model, request, mode):
    """Call the actual preparation and fail on dropped/changed media or slots."""
    import torch
    from breakthrough_prompt import wrap_model

    processor = model.processor
    _require(len(request.questions) == 1, "one independent question required")
    before = request.model_dump_json()
    inputs, slots, counts = wrap_model(model, mode).prepare(request)
    _require(request.model_dump_json() == before, "preparation mutated the native request")
    prefix, question = processor.prefix, request.questions[0]
    _require(prefix is not None and set(inputs) == set(prefix), "native processor keys changed")
    prefix_length = prefix["input_ids"].shape[1]
    sequence_length = inputs["input_ids"].shape[1]
    extension = sequence_length - prefix_length
    _require(extension > 0, "answer suffix missing")
    _require(
        slots.tolist() == [sequence_length - 1],
        "answer slot is not after the full processor prefix",
    )
    _require(counts.tolist() == [len(question.labels())], "candidate count changed")
    tensor_checks = {}
    for key, value in prefix.items():
        if key in SEQUENCE_KEYS:
            _require(
                torch.equal(inputs[key][:, :prefix_length], value), f"native prefix changed: {key}"
            )
            _require(inputs[key].shape[1] == sequence_length, f"sequence mask not extended: {key}")
            if key != "input_ids":
                fill = 1 if key == "attention_mask" else 0
                _require(
                    bool((inputs[key][:, prefix_length:] == fill).all()),
                    f"answer suffix mask changed: {key}",
                )
            tensor_checks[key] = "exact_prefix_retained_and_suffix_appended"
        else:
            _require(torch.equal(inputs[key], value), f"native media tensor changed: {key}")
            tensor_checks[key] = "exact_native_tensor_retained"
    _require(bool((inputs["attention_mask"][:, slots] == 1).all()), "answer slot is masked")
    suffix = model.tok.decode(inputs["input_ids"][0, prefix_length:], skip_special_tokens=False)
    _require(suffix.endswith("Answer: ("), "answer boundary differs under the actual tokenizer")
    _require(
        [message["role"] for message in processor.messages] == ["user"], "request role changed"
    )
    rendered = processor.rendered
    _require(isinstance(rendered, str), "chat template did not render text")
    user_start = rendered.find("<|turn>user\n")
    user_end = rendered.find("<turn|>", user_start)
    assistant = rendered.find("<|turn>model\n", user_end)
    _require(
        user_start >= 0 and user_end > user_start and assistant > user_end,
        "actual native role boundaries missing",
    )
    # The native chat template trims each text block's outer whitespace.
    marker = f"Question ({question.type}): {question.instructions}"
    question_position = rendered.find(marker)
    if mode == "user_question":
        _require(user_start < question_position < user_end, "question is outside the USER role")
        _require("Question (" not in suffix, "question duplicated into the ASSISTANT suffix")
    else:
        _require(
            question_position == -1 and question.instructions in suffix,
            "current question placement changed",
        )
    nontext_positions = []
    if "mm_token_type_ids" in inputs:
        nontext_positions = torch.where(inputs["mm_token_type_ids"][0] != 0)[0].tolist()
        _require(
            not nontext_positions or slots[0].item() > nontext_positions[-1],
            "answer slot precedes native media",
        )
    return {
        "mode": mode,
        "status": "verified",
        "prefix_tokens_after_native_expansion": prefix_length,
        "sequence_tokens": sequence_length,
        "appended_tokens": extension,
        "slots": slots.tolist(),
        "candidate_counts": counts.tolist(),
        "slot_token_id": int(inputs["input_ids"][0, slots[0]].item()),
        "suffix_decoded": suffix,
        "rendered_prompt": rendered,
        "role_boundaries": {
            "user_start": user_start,
            "user_end": user_end,
            "assistant_start": assistant,
            "question_start": question_position,
        },
        "question_inside_user": mode == "user_question",
        "nontext_token_positions": nontext_positions,
        "tensor_checks": tensor_checks,
        "native_prefix_tensors": {
            key: tensor_evidence(value, values=key in SEQUENCE_KEYS or "mask" in key)
            for key, value in prefix.items()
        },
        "prepared_tensors": {
            key: tensor_evidence(value, values=key in SEQUENCE_KEYS or "mask" in key)
            for key, value in inputs.items()
        },
    }


def verify(checkpoint, fixture, output, *, max_context=16384):
    """Persist complete evidence or the exact partial failure; never overwrite a run."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from s1.evaluation.datasets import load_cases, request_fingerprint
    from s1.unified import UnifiedDecisionModel, candidate_ids

    checkpoint, fixture, output = Path(checkpoint).resolve(), Path(fixture).resolve(), Path(output)
    _require(
        checkpoint.is_dir() and fixture.is_file(), "local processor checkpoint and fixture required"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    source_paths = [
        Path(__file__),
        ROOT / "scripts/breakthrough_prompt.py",
        ROOT / "src/s1/unified.py",
    ]
    report = {
        "schema_version": 1,
        "status": "running",
        "scope": "Local native tokenizer/processor and prompt/tensor preparation only. No model weights loaded, model inference, GPU run, quality or speed evidence.",
        "offline": {"local_files_only": True, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        "checkpoint": str(checkpoint),
        "fixture": {"path": str(fixture), "file_sha256": sha256(fixture.read_bytes())},
        "source_files": {
            str(path.relative_to(ROOT)): sha256(path.read_bytes()) for path in source_paths
        },
        "processor_files": {
            path.name: {"sha256": sha256(path.read_bytes()), "size_bytes": path.stat().st_size}
            for path in sorted(checkpoint.iterdir())
            if path.is_file() and path.suffix in (".json", ".model", ".jinja", ".txt")
        },
        "versions": {
            key: importlib.metadata.version(key)
            for key in ("torch", "transformers", "numpy", "pillow")
        },
        "platform": platform.platform(),
        "max_context": max_context,
        "records": [],
        "errors": [],
        "inference": "not_run",
        "model_quality": "not_measured",
        "gpu_performance": "not_measured",
    }
    with output.open("xb") as stream:
        stream.write(json_bytes(report))

    def checkpoint_report():
        temporary = output.with_suffix(output.suffix + ".partial")
        temporary.write_bytes(json_bytes(report))
        temporary.replace(output)

    try:
        from transformers import AutoProcessor

        raw_processor = AutoProcessor.from_pretrained(str(checkpoint), local_files_only=True)
        processor = ObservedProcessor(raw_processor)
        # prepare() needs only these four attributes; no backbone/head is created.
        model = UnifiedDecisionModel.__new__(UnifiedDecisionModel)
        model.processor, model.tok, model.device, model.max_context = (
            processor,
            processor.tokenizer,
            "cpu",
            max_context,
        )
        report["processor_class"] = type(raw_processor).__name__
        report["tokenizer_class"] = type(raw_processor.tokenizer).__name__
        report["stable_candidate_token_ids"] = candidate_ids(model.tok)
        cases = load_cases(fixture)
        report["fixture"].update(
            cases=len(cases),
            questions=sum(len(case.request.questions) for case in cases),
            ordered_case_ids=[case.id for case in cases],
        )
        case_types, question_types = Counter(), Counter()
        for case in cases:
            media_types = {media.type for media in case.request.media}
            modality = "mixed" if len(media_types) > 1 else next(iter(media_types), "text")
            case_types[modality] += 1
            for question in case.request.questions:
                question_types[modality] += 1
                request = case.request.model_copy(update={"questions": [question]})
                record = {
                    "case_id": case.id,
                    "question_id": question.id,
                    "type": question.type,
                    "labels": question.labels(),
                    "modality": modality,
                    "request_sha256": request_fingerprint(request),
                    "modes": {},
                }
                report["records"].append(record)
                for mode in ("current", "user_question"):
                    record["modes"][mode] = capture_preparation(model, request, mode)
                    checkpoint_report()
                current, proposed = [
                    record["modes"][mode]["native_prefix_tensors"]
                    for mode in ("current", "user_question")
                ]
                native_keys = set(current) - SEQUENCE_KEYS
                _require(
                    native_keys == set(proposed) - SEQUENCE_KEYS,
                    "cross-prompt native media keys differ",
                )
                _require(
                    all(current[key] == proposed[key] for key in native_keys),
                    "cross-prompt native media feature bytes differ",
                )
                record["native_media_identical_across_prompts"] = True
        _require(
            len(cases) == 8 and len(report["records"]) == 29,
            "complete 8-case/29-question fixture required",
        )
        _require(
            set(case_types) == {"text", "image", "audio", "mixed"},
            "four native fixture modalities required",
        )
        report["coverage"] = {
            "cases_by_modality": dict(case_types),
            "questions_by_modality": dict(question_types),
            "prepared_requests": 2 * len(report["records"]),
            "expected_independent_questions": 29,
        }
        report["status"] = "verified_local_processor_only"
    except Exception as error:
        report["status"] = "failed_local_processor_check"
        report["errors"].append(
            {
                "type": type(error).__name__,
                "message": str(error),
                "completed_questions": sum(len(row["modes"]) == 2 for row in report["records"]),
                "prepared_requests": sum(len(row["modes"]) for row in report["records"]),
            }
        )
        checkpoint_report()
        raise
    checkpoint_report()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/release/checkpoint")
    parser.add_argument("--fixture", type=Path, default=ROOT / "examples/benchmarks/mps.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/jevbench/breakthrough-20261008/local-processor-check.json",
    )
    args = parser.parse_args(argv)
    report = verify(args.checkpoint, args.fixture, args.output)
    print(
        json.dumps(
            {"status": report["status"], "coverage": report["coverage"], "output": str(args.output)}
        )
    )
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    raise SystemExit(main())
