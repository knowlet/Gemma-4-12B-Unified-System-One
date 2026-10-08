"""Bounded Blackwell NVFP4 export and validation of the trained S1 release."""

import re
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/workspace")
app = modal.App("gemma-s1-nvfp4-release")
volume = modal.Volume.from_name("gemma-unified-system-one", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.10.0",
        "torchvision==0.25.0",
        "transformers==5.17.0",
        "accelerate==1.15.0",
        "kernels==0.16.0",
        "huggingface-hub==1.33.0",
        "safetensors==0.8.0",
        "pillow==12.3.0",
        "soundfile==0.14.0",
        "librosa==0.11.0",
        "sentencepiece==0.2.2",
        "protobuf==6.33.6",
        "numpy==2.5.3",
        "pydantic==2.13.5",
        "httpx==0.28.1",
        "peft==0.21.0",
    )
    .env({"HF_HOME": "/vol/hf", "TOKENIZERS_PARALLELISM": "false", "USE_TF": "0"})
    .add_local_python_source("s1")
    .add_local_dir(ROOT / "scripts", "/workspace/scripts")
    .add_local_file(
        ROOT / "artifacts/nvfp4/runtime/gemma_system_one-0.1.0-py3-none-any.whl",
        "/workspace/gemma_system_one-0.1.0-py3-none-any.whl",
    )
    .add_local_dir(ROOT / "artifacts/release/datasets", "/workspace/datasets")
    .add_local_file(
        ROOT / "docs/validation/2026-10-02/release/artifacts.json",
        "/workspace/source-artifacts.json",
    )
)


@app.function(
    image=image,
    gpu="B200",
    cpu=4,
    memory=49152,
    volumes={"/vol": volume},
    timeout=600,
    max_containers=1,
)
def probe():
    import importlib.metadata

    import torch

    from s1.nvfp4 import KERNEL_REVISION, configure_nvfp4_kernel

    kernel = configure_nvfp4_kernel()
    weight = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    inputs = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
    packed = kernel.pack(weight, device=torch.device("cuda"))
    result = kernel.gemm(packed, inputs)
    torch.cuda.synchronize()
    assert result.shape == (32, 128) and torch.isfinite(result).all()
    return {
        "status": "ok",
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "weight_dtype": str(packed.qweight.dtype),
        "weight_shape": list(packed.qweight.shape),
        "kernel_revision": KERNEL_REVISION,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "kernels", "huggingface-hub")
        },
    }


@app.function(
    image=image,
    gpu="B200",
    cpu=4,
    memory=49152,
    volumes={"/vol": volume},
    timeout=7200,
    max_containers=1,
)
def release(run: str):
    import sys

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", run):
        raise ValueError("run must be a unique lowercase identifier")
    sys.path.insert(0, "/workspace/scripts")
    from export_nvfp4_release import run_release

    volume.reload()
    return run_release(
        "/vol/releases/20261002-boolq-release-01/merged",
        "/workspace/datasets",
        f"/vol/nvfp4/{run}",
        "/workspace/source-artifacts.json",
        commit=volume.commit,
    )


@app.local_entrypoint()
def main(stage: str = "probe", run: str = "", revision: str = ""):
    import json

    if stage not in ("probe", "release", "verify", "hub-smoke"):
        raise ValueError("stage must be probe, release, verify or hub-smoke")
    if stage != "probe" and not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", run):
        raise ValueError("release requires a unique lowercase run")
    if stage == "hub-smoke" and not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Hub verification requires the immutable published revision")
    result = (
        probe.remote()
        if stage == "probe"
        else hub_smoke.remote(run, revision)
        if stage == "hub-smoke"
        else verify.remote(run)
        if stage == "verify"
        else release.remote(run)
    )
    output = ROOT / "artifacts/nvfp4"
    output.mkdir(parents=True, exist_ok=True)
    (output / ("probe.json" if stage == "probe" else f"{run}-{stage}-receipt.json")).write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


@app.function(
    image=image,
    gpu="B200",
    cpu=4,
    memory=49152,
    volumes={"/vol": volume},
    timeout=3600,
    max_containers=1,
)
def verify(run: str):
    import json
    import subprocess
    import sys

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", run):
        raise ValueError("invalid run identifier")
    volume.reload()
    package = Path(f"/vol/nvfp4/{run}/package")
    subprocess.run(
        [
            sys.executable,
            "/workspace/scripts/verify_nvfp4_runtime.py",
            "--wheel",
            "/workspace/gemma_system_one-0.1.0-py3-none-any.whl",
            "--model",
            str(package),
            "--data",
            "/workspace/datasets",
        ],
        check=True,
    )
    volume.commit()
    return json.loads((package / "runtime-verification.json").read_text())


@app.function(
    image=image,
    gpu="B200",
    cpu=4,
    memory=49152,
    volumes={"/vol": volume},
    timeout=1800,
    max_containers=1,
)
def hub_smoke(run: str, revision: str):
    import json
    import sys

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", run) or not re.fullmatch(
        r"[a-f0-9]{40}", revision
    ):
        raise ValueError("invalid immutable publication identity")
    sys.path.insert(0, "/workspace/scripts")
    from release_training import load_release_data

    from s1.unified import UnifiedDecisionModel

    volume.reload()
    data, _ = load_release_data("/workspace/datasets")
    package = Path(f"/vol/nvfp4/{run}/package")
    evaluation = json.loads((package / "evaluation.json").read_text())
    repo = "knowlet/Gemma-4-12B-Unified-System-One-NVFP4"
    model = UnifiedDecisionModel(
        repo,
        revision=revision,
        quantization="nvfp4",
        device="cuda",
        precision="bfloat16",
        attn_implementation="sdpa",
        max_context=4096,
    )
    if model.revision != revision:
        raise ValueError("Hub loader did not preserve the pinned publication revision")
    records = []
    for split, index in (("test", 0), ("media", 0), ("media", -1)):
        case = data[split][index]
        answer = model.predict(case.request)
        expected = next(
            record
            for record in evaluation["reports"][split]["records"]
            if record["case_id"] == case.id
        )
        observed = answer["answers"][expected["question_id"]]["probabilities"]
        difference = max(
            abs(observed[key] - value) for key, value in expected["probabilities"].items()
        )
        if difference > 1e-6:
            raise ValueError("published Hub inference differs from the validated artifact")
        records.append({"case_id": case.id, "max_absolute_probability_difference": difference})
    result = {
        "status": "ok",
        "repository": repo,
        "revision": revision,
        "quantization": model.quantization_details,
        "records": records,
        "scope": "anonymous pinned Hub load with native text/image/audio forwards",
    }
    volume.commit()
    return result
