"""Fixed, exploratory NF4 LoRA continuation; test inputs never enter fitting.

The recipe is intentionally not configurable: all three existing CE parents,
100 batch-one CE updates at 2e-5, followed by the original 16-case calibration.
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path

from s1.evaluation.checkpoints import directory_digest
from s1.evaluation.comparison import CE_DIGESTS, GEMMA_REVISION, verify_datasets
from s1.evaluation.datasets import fingerprint, load_cases, request_fingerprint

TRAIN_SHA = "4bc27d96d998ef02137e58e2c8b5d4f928dafc9db03476d436a13acddf35decb"
CALIBRATION_SHA = "2d919497859b47f55aaaf6e67a63e8f2141877f77cacc0625dd59d3e67add9b8"
STEPS = 100
LEARNING_RATE = 2e-5


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def recipe(seed):
    if type(seed) is not int or seed not in range(3):
        raise ValueError("all three predeclared seeds are required: 0, 1, 2")
    return {
        "method": "fixed_nf4_qlora_continuation",
        "seed": seed,
        "parent_adapter_sha256": CE_DIGESTS[seed],
        "additional_optimizer_updates": STEPS,
        "learning_rate": LEARNING_RATE,
        "brier_weight": 0.0,
        "batch_size": 1,
        "train_dataset_sha256": TRAIN_SHA,
        "calibration_dataset_sha256": CALIBRATION_SHA,
        "supplied_training_cases": 256,
        "observed_training_cases": STEPS,
        "no_test_fitting": True,
        "exploratory": True,
        "recipe_fixed_before_execution": True,
        "selection_policy": "report all three seeds; no test-selected recipe, checkpoint, or seed",
        "optimizer": "AdamW default betas/eps/weight_decay; gradient norm clipped to 1.0",
        "training_order": "existing train_model seeded shuffle; 100 updates, less than one epoch",
        "calibration": "original 16 cases; scalar temperature; cannot change argmax accuracy",
    }


def prepare_data(directory):
    """Check all historical inputs, then return only train and calibration cases."""
    directory = Path(directory)
    dataset = verify_datasets(directory)
    train = load_cases(directory / "text-train-full.jsonl")
    calibration = load_cases(directory / "text-calibration.jsonl")
    if len(train) != 256 or fingerprint(train) != TRAIN_SHA:
        raise ValueError("historical 256-case training dataset changed")
    if len(calibration) != 16 or fingerprint(calibration) != CALIBRATION_SHA:
        raise ValueError("historical 16-case calibration dataset changed")
    provenance = {}
    for prefix, cases in (("train", train), ("calibration", calibration)):
        provenance.update(
            {
                f"{prefix}_ids": [c.id for c in cases],
                f"{prefix}_groups": [c.group_id or c.id for c in cases],
                f"{prefix}_requests": [request_fingerprint(c.request) for c in cases],
            }
        )
    return train, calibration, provenance, dataset


def verify_parent(parent, seed, provenance):
    if directory_digest(parent) != CE_DIGESTS[seed]:
        raise ValueError("parent adapter digest differs from historical CE adapter")
    previous = json.loads((Path(parent) / "training-provenance.json").read_text())
    # The original support manifest may store the same 256 examples in a
    # seed-specific order. Identity equality is set equality, including requests.
    for key, values in provenance.items():
        if len(previous.get(key, [])) != len(values) or set(previous[key]) != set(values):
            raise ValueError(f"parent adapter training lineage differs: {key}")


def audit_indices(size):
    """Exact integer indices: float32 linspace rounds large final indices up."""
    if size <= 64:
        return list(range(size))
    return [index * (size - 1) // 63 for index in range(64)]


def parameter_digest(model, *, lora):
    """Full LoRA digest; sampled frozen-weight audit without copying 7GB to CPU."""
    import torch

    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if ("lora_" in name) != lora:
            continue
        tensor = parameter.detach().flatten()
        if not lora and tensor.numel() > 64:
            indices = torch.tensor(
                audit_indices(tensor.numel()), device=tensor.device, dtype=torch.int64
            )
            tensor = tensor[indices]
        digest.update(f"{name}:{parameter.dtype}:{tuple(parameter.shape)}".encode())
        digest.update(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()


def assert_training_state(model):
    import torch

    parameters = dict(model.lm.named_parameters())
    trainable = {name: p for name, p in parameters.items() if p.requires_grad}
    if not trainable or any("lora_" not in name for name in trainable):
        raise ValueError("only existing LoRA parameters may train")
    if any(p.dtype != torch.float32 for p in trainable.values()):
        raise ValueError("LoRA optimization requires the declared FP32 adapter tensors")
    embeddings = model.lm.get_base_model().get_input_embeddings().weight
    if embeddings.requires_grad or embeddings.dtype != torch.bfloat16:
        raise ValueError("input embeddings must remain frozen BF16")
    if model.head.weight.requires_grad or model.head.weight.dtype != torch.bfloat16:
        raise ValueError("output head must remain frozen BF16")
    return {
        "trainable_parameters": sum(p.numel() for p in trainable.values()),
        "trainable_names": sorted(trainable),
        "trainable_dtype": "float32",
        "embeddings_dtype": "bfloat16",
        "embeddings_frozen": True,
        "head_dtype": "bfloat16",
        "head_frozen": True,
        "only_lora_trainable": True,
    }


def calibration_reference(model, cases):
    return [
        {
            "case_id": case.id,
            "request_sha256": request_fingerprint(case.request),
            "answers": model.predict(case.request)["answers"],
        }
        for case in cases
    ]


def train_recovery(data, parent, adapter, receipt_path, *, seed, lockfile):
    import importlib.metadata

    import torch
    from peft import PeftModel
    from torch.nn import functional as F

    from s1.quantization import inspect_quantization
    from s1.training import train_model
    from s1.unified import DEFAULT_MODEL, UnifiedDecisionModel

    config = recipe(seed)
    train, calibration, provenance, dataset = prepare_data(data)
    verify_parent(parent, seed, provenance)
    adapter, receipt_path = Path(adapter), Path(receipt_path)
    if adapter.exists() or receipt_path.exists():
        raise ValueError("recovery outputs must be new; existing results cannot be overwritten")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running",
        "recovery": config,
        "dataset": dataset,
        "base_model": DEFAULT_MODEL,
        "base_revision": GEMMA_REVISION,
        "parent_adapter_path": str(parent),
        "adapter_path": str(adapter),
        "lock_sha256": hashlib.sha256(Path(lockfile).read_bytes()).hexdigest(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "peft", "bitsandbytes")
        },
    }
    write_json(receipt_path, report)
    started = time.perf_counter()
    model = None
    original_adamw = torch.optim.AdamW
    try:
        torch.manual_seed(seed)
        model = UnifiedDecisionModel(
            DEFAULT_MODEL,
            revision=GEMMA_REVISION,
            device="cuda",
            precision="bfloat16",
            quantization="nf4",
            temperature=1.0,
        )
        model.lm.requires_grad_(False)
        # Do not call prepare_model_for_kbit_training: it casts the large frozen
        # embedding/head to FP32. Non-reentrant checkpointing works with frozen inputs.
        model.lm = PeftModel.from_pretrained(
            model.lm, str(parent), is_trainable=True, autocast_adapter_dtype=True
        )
        base = model.lm.get_base_model()
        model.backbone, model.head = base.model, base.get_output_embeddings()
        model.quantization_details = inspect_quantization(model.lm, "nf4")
        base.config.use_cache = False
        if hasattr(base.config, "text_config"):
            base.config.text_config.use_cache = False
        model.lm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        report["training_state"] = assert_training_state(model)
        report["quantization_details"] = model.quantization_details
        report["lora_sha256_before"] = parameter_digest(model.lm, lora=True)
        report["frozen_parameter_sample_sha256_before"] = parameter_digest(model.lm, lora=False)
        model.reset_memory_peak()
        # Meaningful real-NF4 smoke: one backward pass and zero optimizer updates.
        model.lm.train()
        logits = model.logits(train[0].request)
        golds = torch.tensor(
            [q.labels().index(train[0].gold[q.id]) for q in train[0].request.questions],
            device=logits.device,
        )
        smoke_loss = F.cross_entropy(logits, golds)
        smoke_loss.backward()
        gradients = [
            p.grad for p in model.lm.parameters() if p.requires_grad and p.grad is not None
        ]
        if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
            raise ValueError("NF4 backward smoke produced missing or nonfinite LoRA gradients")
        if not any(torch.count_nonzero(g).item() for g in gradients):
            raise ValueError("NF4 backward smoke produced only zero gradients")
        if any(p.grad is not None for p in model.lm.parameters() if not p.requires_grad):
            raise ValueError("frozen base received a gradient")
        report["smoke"] = {
            "status": "passed",
            "optimizer_updates": 0,
            "train_case_id": train[0].id,
            "loss": float(smoke_loss.detach()),
            "nonzero_finite_lora_gradients": True,
        }
        del logits, golds, smoke_loss, gradients
        model.lm.zero_grad(set_to_none=True)
        if parameter_digest(model.lm, lora=True) != report["lora_sha256_before"]:
            raise ValueError("smoke check changed adapter weights")
        print(f"seed {seed}: NF4 backward smoke passed; starting fixed 100 updates", flush=True)
        observed, updates = [], []
        by_request = {request_fingerprint(c.request): c.id for c in train}
        original_logits = model.logits

        def tracked_logits(request):
            if model.lm.training:
                key = request_fingerprint(request)
                if key not in by_request:
                    raise ValueError("optimizer received a request outside the frozen train split")
                observed.append(by_request[key])
            return original_logits(request)

        class CountedAdamW(original_adamw):
            def step(self, closure=None):
                result = super().step(closure)
                updates.append(len(updates) + 1)
                if len(updates) % 10 == 0:
                    print(f"seed {seed}: optimizer update {len(updates)}/100", flush=True)
                return result

        model.logits = tracked_logits
        torch.optim.AdamW = CountedAdamW
        try:
            report["training"] = train_model(
                model,
                train,
                calibration,
                steps=STEPS,
                lr=LEARNING_RATE,
                brier_weight=0.0,
                seed=seed,
            )
        finally:
            torch.optim.AdamW = original_adamw
            model.logits = original_logits
        if len(updates) != STEPS or len(observed) != STEPS or len(set(observed)) != STEPS:
            raise ValueError("fixed 100-update training trace differs from declaration")
        report["actual_optimizer_updates"] = len(updates)
        report["observed_training_ids"] = observed
        report["observed_training_request_sha256"] = [
            request_fingerprint(next(c.request for c in train if c.id == case_id))
            for case_id in observed
        ]
        report["training_state_after"] = assert_training_state(model)
        report["lora_sha256_after"] = parameter_digest(model.lm, lora=True)
        report["frozen_parameter_sample_sha256_after"] = parameter_digest(model.lm, lora=False)
        if report["lora_sha256_before"] == report["lora_sha256_after"]:
            raise ValueError("100 optimizer updates did not change LoRA weights")
        if (
            report["frozen_parameter_sample_sha256_before"]
            != report["frozen_parameter_sample_sha256_after"]
        ):
            raise ValueError("frozen base parameter audit changed")
        model.lm.gradient_checkpointing_disable()
        report["calibration_reference"] = calibration_reference(model, calibration)
        report["training_memory"] = model.memory_snapshot()
        model.save_adapter(adapter)
        write_json(adapter / "training-provenance.json", {**provenance, "recovery": config})
        report["adapter_sha256"] = directory_digest(adapter)
        report["status"] = "completed"
        print(f"seed {seed}: saved recovered adapter {report['adapter_sha256']}", flush=True)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        torch.optim.AdamW = original_adamw
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(receipt_path, report)
        model = None
        gc.collect()
        # Keep the original failure and receipt if CUDA itself has become invalid.
        if report["status"] == "completed":
            torch.cuda.empty_cache()
    return report


def check_fresh_calibration(data, report):
    """Reload from disk in an inference container and compare only calibration."""
    import torch

    from s1.unified import UnifiedDecisionModel

    _, cases, _, _ = prepare_data(data)
    if directory_digest(report["adapter_path"]) != report["adapter_sha256"]:
        raise ValueError("saved adapter digest changed before fresh inference")
    model = UnifiedDecisionModel(
        report["base_model"],
        revision=report["base_revision"],
        adapter_path=report["adapter_path"],
        device="cuda",
        precision="bfloat16",
        quantization="nf4",
    )
    try:
        actual = calibration_reference(model, cases)
        maximum = 0.0
        for expected, observed in zip(report["calibration_reference"], actual, strict=True):
            if (
                expected["case_id"] != observed["case_id"]
                or expected["request_sha256"] != observed["request_sha256"]
            ):
                raise ValueError("calibration identity changed during fresh reload")
            for question, answer in expected["answers"].items():
                for label, probability in answer["probabilities"].items():
                    maximum = max(
                        maximum,
                        abs(probability - observed["answers"][question]["probabilities"][label]),
                    )
        passed = maximum <= 2e-6
        result = {
            "status": "passed" if passed else "failed",
            "cases": len(actual),
            "maximum_probability_absolute_difference": maximum,
            "tolerance": 2e-6,
            "calibration_dataset_sha256": CALIBRATION_SHA,
            "adapter_sha256": report["adapter_sha256"],
            "temperature": model.temperature,
            "frozen_adapter_on_reload": all(not p.requires_grad for p in model.lm.parameters()),
            "predictions": actual,
        }
        if not passed or not result["frozen_adapter_on_reload"]:
            raise ValueError(
                f"fresh calibration parity failed: maximum probability drift {maximum}"
            )
        return result
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
