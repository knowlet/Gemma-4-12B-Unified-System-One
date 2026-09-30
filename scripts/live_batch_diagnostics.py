"""Inspect live single-versus-batch divergence without changing core inference.

The caller supplies the already loaded pinned model. This script starts no cloud
job and never loosens the application's parity tolerance.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import time
from pathlib import Path


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _input_info(inputs):
    from s1.evaluation.gemma import SEQUENCE_KEYS

    return {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            **({"values": value.detach().cpu().tolist()} if key in SEQUENCE_KEYS else {}),
        }
        for key, value in inputs.items()
    }


def _measure(model, inputs, prepared, row, batch_prepared=None):
    import torch

    from s1.unified import project_candidates

    _, slots, counts = prepared
    length = prepared[0]["input_ids"].shape[1]
    torch.cuda.synchronize(model.device)
    before = time.perf_counter()
    with torch.inference_mode():
        output = model.backbone(**inputs, use_cache=False, return_dict=True)
        valid_hidden = output.last_hidden_state[row, :length].float().cpu()
        hidden = output.last_hidden_state[row, slots]
        single_row_logits = project_candidates(
            hidden, model.head, model.letters, counts, model.softcap
        )
        single_candidates = single_row_logits[0, : int(counts[0])]
        batch_prepared = batch_prepared or [prepared]
        batch_hidden = torch.cat(
            [output.last_hidden_state[index, item[1]] for index, item in enumerate(batch_prepared)]
        )
        batch_counts = torch.cat([item[2] for item in batch_prepared])
        # Reproduce G4's joint candidate projection as well as projecting the
        # identical target hidden state alone, isolating readout kernel drift.
        logits = project_candidates(
            batch_hidden, model.head, model.letters, batch_counts, model.softcap
        )
        candidates = logits[0, : int(counts[0])]
        probabilities = (candidates / model.temperature).softmax(-1)
        result = {
            "candidate_logits": candidates.cpu().tolist(),
            "probabilities": probabilities.cpu().tolist(),
            "single_row_candidate_logits": single_candidates.cpu().tolist(),
            "readout_batch_logit_delta": float((candidates - single_candidates).abs().max()),
            "readout_batch_probability_delta": float(
                (probabilities - (single_candidates / model.temperature).softmax(-1)).abs().max()
            ),
            "argmax": int(probabilities.argmax()),
            "slot_hidden": hidden.float().cpu(),
            "valid_hidden": valid_hidden,
        }
    torch.cuda.synchronize(model.device)
    result["wall_seconds"] = time.perf_counter() - before
    return result


def _difference(reference, actual):
    import torch

    logits_a, logits_b = (
        torch.tensor(reference["candidate_logits"]),
        torch.tensor(actual["candidate_logits"]),
    )
    probabilities_a, probabilities_b = (
        torch.tensor(reference["probabilities"]),
        torch.tensor(actual["probabilities"]),
    )
    hidden_difference = (reference["valid_hidden"] - actual["valid_hidden"]).abs()
    return {
        "candidate_logit_max_abs_delta": float((logits_a - logits_b).abs().max()),
        "probability_max_abs_delta": float((probabilities_a - probabilities_b).abs().max()),
        "valid_hidden_max_abs_delta": float(hidden_difference.max()),
        "valid_hidden_mean_abs_delta": float(hidden_difference.mean()),
        "slot_hidden_max_abs_delta": float(
            (reference["slot_hidden"] - actual["slot_hidden"]).abs().max()
        ),
        "argmax_match": reference["argmax"] == actual["argmax"],
    }


def _configs(model):
    found = {}
    for module in model.lm.modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "_attn_implementation"):
            found[id(config)] = config
    return list(found.values())


def run(model, root: Path, output: Path, dataset_dir: Path):
    """Save actual inputs and isolate padding, attention backend, and precision."""
    import torch

    from s1.contracts import Question
    from s1.evaluation.datasets import load_cases
    from s1.evaluation.gemma import _collate

    root, output, dataset_dir = Path(root), Path(output), Path(dataset_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    receipt = {
        "status": "running",
        "checkpoint": model.name,
        "revision": model.revision,
        "original_dtype": str(model.head.weight.dtype),
        "original_temperature": model.temperature,
        "cases": [],
        "experiments": [],
        "errors": [],
        "hypotheses": [
            "missing or incorrect attention masks",
            "sequence padding versus batch shape",
            "implicit versus explicit position IDs",
            "SDPA versus eager attention",
            "BF16 matmul reduction versus FP32 computation",
        ],
    }
    receipt_path = output / "batch-diagnostics.json"
    original_dtype = model.head.weight.dtype
    configurations = _configs(model)
    original_attention = [(config, config._attn_implementation) for config in configurations]
    original_matmul_precision = torch.get_float32_matmul_precision()
    original_bf16_reduction = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    # BF16 parameters round-trip through FP32 exactly. Preserve the few original
    # FP32 floating buffers independently, without cloning any model parameter.
    original_buffer_dtypes = {name: buffer.dtype for name, buffer in model.lm.named_buffers()}
    preserved_buffers = {
        name: buffer.detach().cpu().clone()
        for name, buffer in model.lm.named_buffers()
        if buffer.is_floating_point() and buffer.dtype != original_dtype
    }

    def persist(stage):
        receipt["stage"] = stage
        receipt["wall_seconds"] = time.perf_counter() - started
        _write(receipt_path, receipt)
        print(
            json.dumps(
                {
                    "check": "batch_diagnostics",
                    "stage": stage,
                    "wall_seconds": round(receipt["wall_seconds"], 3),
                }
            ),
            flush=True,
        )

    def force_masks(prepared):
        return [
            (
                {
                    **inputs,
                    "attention_mask": inputs.get(
                        "attention_mask", torch.ones_like(inputs["input_ids"])
                    ),
                },
                slots,
                counts,
            )
            for inputs, slots, counts in prepared
        ]

    def backend(implementation):
        # The official setter validates supported nested model implementations.
        model.lm.set_attn_implementation(implementation)
        return [
            {"model_type": config.model_type, "implementation": config._attn_implementation}
            for config in configurations
        ]

    def experiment(
        case_name,
        condition,
        prepared,
        *,
        explicit_positions=False,
        pad_single=False,
        duplicate=False,
    ):
        try:
            items = force_masks(prepared) if condition.startswith("forced_mask") else prepared
            inputs = _collate(items, getattr(model.tok, "pad_token_id", None) or 0)
            if pad_single:
                inputs = {key: value[:1] for key, value in inputs.items()}
            if duplicate:
                inputs = {
                    key: value[:1].repeat((2,) + (1,) * (value.ndim - 1))
                    for key, value in inputs.items()
                }
            if explicit_positions:
                ids = inputs["input_ids"]
                inputs["position_ids"] = torch.arange(ids.shape[1], device=ids.device).expand(
                    ids.shape[0], -1
                )
            batch_prepared = (
                [prepared[0], prepared[0]]
                if duplicate
                else prepared[:1]
                if pad_single
                else prepared
            )
            result = _measure(model, inputs, prepared[0], 0, batch_prepared)
            row = {
                "case": case_name,
                "condition": condition,
                "precision": str(model.head.weight.dtype),
                "attention_implementations": [
                    config._attn_implementation for config in configurations
                ],
                "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
                "batch_size": int(inputs["input_ids"].shape[0]),
                "padded_sequence_tokens": int(inputs["input_ids"].shape[1]),
                "candidate_logits": result["candidate_logits"],
                "probabilities": result["probabilities"],
                "single_row_candidate_logits": result["single_row_candidate_logits"],
                "readout_batch_logit_delta": result["readout_batch_logit_delta"],
                "readout_batch_probability_delta": result["readout_batch_probability_delta"],
                "argmax": result["argmax"],
                "wall_seconds": result["wall_seconds"],
                "inputs": _input_info(inputs),
            }
            reference = references.get(case_name)
            if reference is not None:
                row["difference_from_original_single"] = _difference(reference, result)
            receipt["experiments"].append(row)
            persist(f"{case_name}:{condition}")
            return result
        except Exception as exc:
            receipt["errors"].append(
                {
                    "case": case_name,
                    "condition": condition,
                    "type": type(exc).__name__,
                    "message": str(exc)[:1000],
                }
            )
            persist(f"{case_name}:{condition}:failed")
            gc.collect()
            torch.cuda.empty_cache()
            return None

    references = {}
    prepared_cases = {}
    try:
        model.lm.eval()
        contract = load_cases(root / "examples/benchmarks/text-contract.jsonl")
        login = next(case for case in contract if case.id == "login-en")
        target = next(question for question in login.request.questions if question.id == "refund")
        peer = next(question for question in login.request.questions if question.id != "refund")
        target_request = login.request.model_copy(update={"questions": [target]})
        peer_request = login.request.model_copy(update={"questions": [peer]})
        long_request = target_request.model_copy(
            update={"state": "Neutral context. " * 128 + str(target_request.state)}
        )
        prepared_cases["refund-question-batch"] = [
            model.prepare(target_request),
            model.prepare(peer_request),
        ]
        prepared_cases["refund-unequal-context"] = [
            model.prepare(target_request),
            model.prepare(long_request),
        ]
        unrelated = Question(
            id="unrelated-control",
            instructions="Choose the first option.",
            criteria={"first": "The first option", "second": "The second"},
        )
        prepared_cases["refund-with-unrelated"] = [
            *prepared_cases["refund-question-batch"],
            model.prepare(login.request.model_copy(update={"questions": [unrelated]})),
        ]
        spec = importlib.util.spec_from_file_location(
            "live_checkpoint_checks_diag", root / "scripts/live_checkpoint_checks.py"
        )
        checks = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checks)
        multimodal = checks._parity_requests(contract, load_cases(dataset_dir / "media-test.jsonl"))
        observed = multimodal["mixed"].questions[0]
        prepared_cases["mixed-unequal-context"] = [
            model.prepare(multimodal["mixed"].model_copy(update={"questions": [observed]})),
            model.prepare(
                multimodal["mixed-unequal-length"].model_copy(update={"questions": [observed]})
            ),
        ]
        prepared_cases["mixed-native-question-batch"] = [
            model.prepare(request.model_copy(update={"questions": [question]}))
            for request in (multimodal["mixed"], multimodal["mixed-unequal-length"])
            for question in request.questions
        ]
        for case_name, prepared in prepared_cases.items():
            receipt["cases"].append(
                {
                    "case": case_name,
                    "prepared_inputs": [_input_info(item[0]) for item in prepared],
                    "answer_slots": [item[1].cpu().tolist() for item in prepared],
                    "attention_mask_present": ["attention_mask" in item[0] for item in prepared],
                    "position_ids_present": ["position_ids" in item[0] for item in prepared],
                }
            )
            single = experiment(case_name, "original_single", prepared[:1])
            if single is None:
                continue
            references[case_name] = single
            experiment(case_name, "original_batch", prepared)
            experiment(case_name, "forced_mask_batch", prepared)
            experiment(case_name, "padded_single", prepared, pad_single=True)
            experiment(case_name, "batch_explicit_positions", prepared, explicit_positions=True)
            if case_name == "refund-question-batch":
                experiment(case_name, "identical_unpadded_batch", prepared[:1], duplicate=True)
                experiment(case_name, "identical_padded_batch", prepared, duplicate=True)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        for case_name, prepared in prepared_cases.items():
            experiment(case_name, "bf16_full_reduction_single", prepared[:1])
            experiment(case_name, "bf16_full_reduction_batch", prepared)
        receipt["attention_backend_switches"] = {}
        for implementation in ("eager", "sdpa"):
            try:
                receipt["attention_backend_switches"][implementation] = backend(implementation)
            except Exception as exc:
                receipt["errors"].append(
                    {
                        "condition": f"switch_{implementation}",
                        "type": type(exc).__name__,
                        "message": str(exc)[:1000],
                    }
                )
                continue
            for case_name, prepared in prepared_cases.items():
                single = experiment(case_name, f"{implementation}_single", prepared[:1])
                batched = experiment(case_name, f"{implementation}_batch", prepared)
                if single is not None and batched is not None:
                    receipt["experiments"][-1]["difference_from_same_backend_single"] = _difference(
                        single, batched
                    )
        # FP32 is warranted only if existing BF16 mask/position/backend variants
        # leave material batch drift. No additional checkpoint is resident.
        material_drift = any(
            row.get("difference_from_original_single", {}).get("probability_max_abs_delta", 0)
            > 0.01
            for row in receipt["experiments"]
            if row["condition"] in ("original_batch", "forced_mask_batch")
        )
        receipt["fp32_exercised"] = material_drift
        if material_drift:
            persist("casting_single_checkpoint_to_fp32")
            model.lm.to(dtype=torch.float32)
            for case_name, prepared in prepared_cases.items():
                single = experiment(case_name, "fp32_single", prepared[:1])
                batched = experiment(case_name, "fp32_batch", prepared)
                if single is not None and batched is not None:
                    receipt["experiments"][-1]["difference_from_same_precision_single"] = (
                        _difference(single, batched)
                    )
        receipt["status"] = "completed" if not receipt["errors"] else "completed_with_errors"
    except Exception as exc:
        receipt["status"] = "failed"
        receipt["errors"].append({"type": type(exc).__name__, "message": str(exc)[:1000]})
    finally:
        if model.head.weight.dtype != original_dtype:
            model.lm.to(dtype=original_dtype)
        for name, original in preserved_buffers.items():
            module_name, _, field = name.rpartition(".")
            module = model.lm.get_submodule(module_name) if module_name else model.lm
            module._buffers[field] = original.to(
                device=model.device, dtype=original_buffer_dtypes[name]
            )
        for config, implementation in original_attention:
            config._attn_implementation = implementation
        torch.set_float32_matmul_precision(original_matmul_precision)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = original_bf16_reduction
        torch.backends.cuda.matmul.allow_tf32 = original_tf32
        torch.backends.cudnn.allow_tf32 = original_cudnn_tf32
        references.clear()
        prepared_cases.clear()
        gc.collect()
        torch.cuda.empty_cache()
        receipt["restored_dtype"] = str(model.head.weight.dtype)
        receipt["cuda_allocated_bytes_on_return"] = torch.cuda.memory_allocated()
        persist("completed")
    return receipt
