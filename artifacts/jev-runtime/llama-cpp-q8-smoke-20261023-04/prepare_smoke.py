"""Owned CPU preprocessing only: original S1 prepare and native payload adapter."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from smoke_common import (
    HERE,
    PROCESSOR,
    ROOT,
    encoded,
    error_row,
    need,
    ordinary,
    payload_check,
    read,
    ref,
    save,
    selection,
    sha,
)

PROCESSOR_NAMES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "processor_config.json",
)


def processor_identity(cfg):
    manifest = json.loads(read(ROOT / cfg["processor"]["artifact_manifest"]))
    files = {row["path"]: row for row in manifest["files"]}
    result = []
    for name in PROCESSOR_NAMES:
        expected = files[name]
        path = PROCESSOR / name
        actual = path.resolve(strict=True)
        need(
            actual.is_relative_to(ROOT) and actual.is_file(),
            "Local selected processor metadata only",
        )
        size = actual.stat().st_size
        need(
            size == expected["size_bytes"] <= 64 * 1024**2,
            "Bounded selected processor/tokenizer file",
        )
        digest = hashlib.sha256()
        with actual.open("rb") as stream:
            while block := stream.read(1024**2):
                digest.update(block)
        need(digest.hexdigest() == expected["sha256"], "Same published processor file bytes")
        result.append(
            {
                "name": name,
                "logical_path": str(path),
                "resolved_path": str(actual),
                "size_bytes": size,
                "sha256": digest.hexdigest(),
            }
        )
    return result


def request_data(*, long, padding=0):
    criteria = {"cedar": "A synthetic cedar label", "stone": "A synthetic stone label"}
    early = {
        "id": "synthetic-early" if long else "synthetic-warm",
        "type": "choice",
        "instructions": "Consider this generated sheet and select one label.",
        "criteria": criteria,
    }
    questions = [early]
    if long:
        neutral = "".join(f"Record {n:04d}: cedar stone cloud meadow.\n" for n in range(padding))
        questions.append(
            {
                "id": "synthetic-late",
                "type": "choice",
                "criteria": criteria,
                "instructions": "Read these neutral generated records before selecting a label.\n"
                + neutral
                + "End of synthetic records. Select one label for this sheet.",
            }
        )
    return {
        "state": "A synthetic sheet contains neutral cedar and stone labels. This is a cache arithmetic probe.",
        "questions": questions,
        "media": [],
    }


def build_payload(prepare, context, validate_request, adapter, letters, *, long, directory):
    attempts, low, high, chosen = [], 0, 350 if long else 0, None
    while low <= high:
        count = (low + high) // 2
        request = validate_request(request_data(long=long, padding=count))
        try:
            inputs, slots, nopts = prepare(context, request)
        except ValueError as error:
            if "context tokens; nothing was truncated" not in str(error):
                raise
            attempts.append({"padding_records": count, "context_overflow_refused": True})
            high = count - 1
            continue
        positions, sizes = slots.tolist(), nopts.tolist()
        length = inputs["input_ids"].shape[1]
        attempts.append({"padding_records": count, "slots": positions, "input_token_count": length})
        if long and positions[1] < 1664:
            low = count + 1
            continue
        chosen = (request, inputs, positions, sizes, count)
        high = count - 1
    need(chosen is not None, "Find bounded authoritative long payload without truncation")
    request, inputs, slots, nopts, count = chosen
    ids = inputs["input_ids"][0].tolist()
    envelope = {
        "case_id": "synthetic-cache-long-v1" if long else "synthetic-cache-warm-v1",
        "question_ids": [q.id for q in request.questions],
        "slots": slots,
        "nopts": nopts,
        "letters": letters,
        "input_ids": ids,
        "has_media": False,
        "source_input_token_count": len(ids),
    }
    payload = adapter(envelope, request, inputs, slots, nopts, letters, directory)
    payload_check(payload, long=long)
    need(
        all(
            int(inputs["input_ids"][0, slot])
            == context.tok.encode("Answer: (", add_special_tokens=False)[-1]
            for slot in slots
        ),
        "Each observed slot remains an unfilled original answer boundary",
    )
    return (
        payload,
        request.model_dump(),
        {
            "attempts": attempts,
            "selected_padding_records": count,
            "input_ids_sha256": sha(encoded(ids)),
            "all_inputs_on_CPU": all(
                str(t.device) == "cpu" for t in inputs.values() if hasattr(t, "device")
            ),
            "original_prepare_used": True,
            "original_native_payload_adapter_used": True,
            "answer_labels_injected": False,
            "gold_or_benchmark_data": False,
        },
    )


def run(args):
    need(
        args.expected_entry_sha256 == sha(read(HERE / "prepare_smoke.py")),
        "Explicit prepare entry SHA",
    )
    cfg = selection(args.selection, args.selection_sha256, HERE / "prepare_smoke.py")
    need(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and os.environ.get("ACCELERATE_USE_CPU") == "true",
        "Explicit CPU-only environment",
    )
    out = ordinary(args.output)
    need(not out.exists(), "Fresh preprocessing output")
    out.mkdir(mode=0o700)
    started = time.monotonic()
    state, primary, notes = {}, None, []
    try:
        state["processor_before"] = processor_identity(cfg)
        config = json.loads((PROCESSOR / "config.json").read_text())
        need(
            config["model_type"] == "gemma4_unified"
            and config["text_config"].get("hidden_size_per_layer_input", 0) == 0,
            "Original adapter-compatible processor config",
        )
        sys.path.insert(0, str(ROOT / "src"))
        import torch
        from transformers import AutoProcessor

        from s1.contracts import DecisionRequest
        from s1.unified import UnifiedDecisionModel, candidate_ids

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        need(not torch.cuda.is_initialized(), "No CUDA context in CPU preprocessing")
        adapter_spec = importlib.util.spec_from_file_location(
            "_original_smoke_gguf_adapter", ROOT / "scripts/gguf_adapter.py"
        )
        adapter = importlib.util.module_from_spec(adapter_spec)
        adapter_spec.loader.exec_module(adapter)
        processor = AutoProcessor.from_pretrained(
            str(PROCESSOR), local_files_only=True, trust_remote_code=False
        )
        context = SimpleNamespace(
            processor=processor, tok=processor.tokenizer, device="cpu", max_context=2048
        )
        letters = candidate_ids(processor.tokenizer)
        generated = {}
        for name, is_long in (("warm", False), ("long", True)):
            payload, request, evidence = build_payload(
                UnifiedDecisionModel.prepare,
                context,
                DecisionRequest.model_validate,
                adapter.make_native_payload,
                letters,
                long=is_long,
                directory=out,
            )
            save(out / (name + "-request.json"), request)
            save(out / (name + "-native.json"), payload)
            generated[name] = {
                "request": ref(out / (name + "-request.json")),
                "payload": ref(out / (name + "-native.json")),
                "question_ids": payload["question_ids"],
                "slots": payload["slots"],
                "source_input_token_count": payload["source_input_token_count"],
                "evidence": evidence,
            }
        state["generated"] = generated
        state["torch_version"] = torch.__version__
        state["CUDA_initialized_final"] = torch.cuda.is_initialized()
        need(
            state["CUDA_initialized_final"] is False, "CPU preparation remains without CUDA context"
        )
    except BaseException as error:
        primary = error_row(error)
    finally:
        for phase, operation in (
            (
                "final_source",
                lambda: selection(args.selection, args.selection_sha256, HERE / "prepare_smoke.py"),
            ),
            ("final_processor", lambda: processor_identity(cfg)),
        ):
            try:
                result = operation()
                if phase == "final_processor":
                    state["processor_after"] = result
                    need(
                        result == state.get("processor_before"),
                        "Processor identity unchanged across preparation",
                    )
            except BaseException as error:
                row = {"phase": phase, **error_row(error)}
                if primary is None:
                    primary = row
                else:
                    notes.append(row)
        save(
            out / "prepare-receipt.json",
            {
                "profile": "llama-cpp-q8-smoke-preparation-v1",
                "completed": primary is None,
                "selection_sha256": args.selection_sha256,
                "entry_sha256": args.expected_entry_sha256,
                "elapsed_seconds": time.monotonic() - started,
                "state": state,
                "primary_error": primary,
                "secondary_errors": notes,
                "scope": "Original CPU processor/answer slots only; no language-model loader, weights, generation or native execution invoked",
                "model_loader_called": False,
                "model_quality_or_20s_qualified": False,
            },
        )
    return 0 if primary is None else 1


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--expected-entry-sha256")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--selection-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.run:
        print(
            json.dumps({"mode": "DRY_SOURCE_ONLY", "Torch_imported": False, "model_loaded": False})
        )
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
