"""Export, restore and evaluate the trained release using real NVFP4 buffers."""

from __future__ import annotations

import gc
import importlib.metadata
import json
import time
from pathlib import Path

from release_training import calibrate, file_sha256, load_release_data, write_json

SOURCE_MODEL = "knowlet/Gemma-4-12B-Unified-System-One"
SOURCE_REVISION = "a66f836b56605039fe040f330180e336d19b3362"


def _free():
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def _runtime():
    import torch

    import s1

    return {
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "cuda": torch.version.cuda,
        "s1_sources": {
            str(path.relative_to(Path(s1.__file__).parent)): file_sha256(path)
            for path in sorted(Path(s1.__file__).parent.rglob("*.py"))
        },
        "exporter_sha256": file_sha256(__file__),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "kernels", "huggingface-hub", "safetensors")
        },
    }


def _evaluate(model, data, root, prefix, commit):
    from s1.benchmark import evaluate

    reports = {}
    model.reset_memory_peak()
    for split in ("test", "media", "regression_text"):
        print(f"Evaluating {prefix}: {split} ({len(data[split])} cases)", flush=True)
        report = evaluate(model, data[split], warmup=1)
        report["memory"] = model.memory_snapshot()
        write_json(root / f"{prefix}-{split}.json", report)
        if report["coverage"] != 1.0 or report["warmup_errors"]:
            raise RuntimeError(f"{prefix}/{split} did not complete every request")
        reports[split] = report
        commit()
    return reports


def run_release(source, data_dir, output, source_receipt, *, commit=lambda: None):
    import torch

    from s1.nvfp4 import save_nvfp4, validate_nvfp4_artifact
    from s1.unified import UnifiedDecisionModel

    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    package = output / "package"
    evidence = output / "evidence"
    evidence.mkdir()
    started = time.monotonic()
    receipt = {"status": "running", "runtime": _runtime()}
    write_json(output / "receipt.json", receipt)
    commit()
    try:
        data, identities = load_release_data(data_dir)
        expected = json.loads(Path(source_receipt).read_text())["merged"]
        for name, digest in expected.items():
            if file_sha256(source / name) != digest:
                raise ValueError(f"trained BF16 source checksum mismatch: {name}")
        print("Verified trained BF16 source and frozen datasets", flush=True)
        model = UnifiedDecisionModel(
            str(source),
            device="cuda",
            precision="bfloat16",
            attn_implementation="sdpa",
            max_context=4096,
        )
        model.name = "trained-bf16-b200"
        baseline = _evaluate(model, data, evidence, "bf16", commit)
        baseline_temperature = model.temperature
        del model
        _free()

        print("Quantizing language linear layers to packed NVFP4", flush=True)
        model = UnifiedDecisionModel(
            str(source),
            device="cuda",
            precision="bfloat16",
            attn_implementation="sdpa",
            quantization="nvfp4",
            temperature=1.0,
            max_context=4096,
        )
        parity_cases = [data["test"][0], data["media"][0], data["media"][-1]]
        with torch.inference_mode():
            before = [model.logits(case.request).cpu() for case in parity_cases]
        print("Saving packed checkpoint", flush=True)
        save_nvfp4(model.lm, package)
        model.processor.save_pretrained(package)
        del model
        _free()
        print("Restoring the exported packed checkpoint", flush=True)
        model = UnifiedDecisionModel(
            str(package),
            device="cuda",
            precision="bfloat16",
            attn_implementation="sdpa",
            quantization="nvfp4",
            temperature=1.0,
            max_context=4096,
        )
        with torch.inference_mode():
            after = [model.logits(case.request).cpu() for case in parity_cases]
        parity = []
        for case, left, right in zip(parity_cases, before, after, strict=True):
            width = len(case.request.questions[0].labels())
            left, right = left[:, :width], right[:, :width]
            equal = torch.equal(left, right)
            parity.append(
                {
                    "case_id": case.id,
                    "exact_logits_equal": equal,
                    "max_absolute_logit_difference": (left - right).abs().max().item(),
                }
            )
            if not equal:
                raise RuntimeError("packed NVFP4 save/reload changed candidate logits")
        write_json(evidence / "reload-parity.json", parity)
        calibration = calibrate(model, data["calibration"])
        write_json(evidence / "calibration.json", calibration)
        sidecar = json.loads((source / "s1_config.json").read_text())
        sidecar.update(
            temperature=model.temperature,
            source_model=SOURCE_MODEL,
            source_revision=SOURCE_REVISION,
            quantization="nvfp4",
            calibration_dataset_sha256=identities["calibration"]["dataset_sha256"],
        )
        write_json(package / "s1_config.json", sidecar)
        structural = validate_nvfp4_artifact(package)
        conversion = {
            "schema_version": 1,
            "format": "nvfp4",
            "status": "ok",
            "structural_verification": {"status": "ok", "details": structural},
            "source_model": SOURCE_MODEL,
            "source_revision": SOURCE_REVISION,
            "source_files": expected,
            "runtime": _runtime(),
            "datasets": identities,
            "quantization": model.quantization_details,
            "quantization_data": "none; native dynamic NVFP4 quantization",
            "temperature_fitting": "256 disjoint calibration cases only",
            "files": {
                path.name: {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
                for path in sorted(package.iterdir())
                if path.is_file()
            },
        }
        write_json(package / "conversion-manifest.json", conversion)
        commit()
        model.name = "trained-nvfp4-b200"
        reports = _evaluate(model, data, evidence, "nvfp4", commit)
        evaluation = {
            "schema_version": 1,
            "status": "ok",
            "release_validation_complete": True,
            "population_complete": True,
            "conversion_manifest_sha256": file_sha256(package / "conversion-manifest.json"),
            "runtime": _runtime(),
            "datasets": identities,
            "reload_parity": parity,
            "temperature": model.temperature,
            "bf16_temperature": baseline_temperature,
            "reports": reports,
            "bf16_reports": baseline,
            "scope": "Fixed public data; same B200 and PyTorch runtime; no training or test fitting.",
        }
        write_json(package / "evaluation.json", evaluation)
        receipt.update(
            status="ok",
            elapsed_seconds=time.monotonic() - started,
            package=str(package),
            reports={key: value["metrics"] for key, value in reports.items()},
        )
        return receipt
    except BaseException as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(output / "receipt.json", receipt)
        commit()
