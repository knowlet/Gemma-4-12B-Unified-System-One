"""Ephemeral matched A100 comparison, one model per fresh container.

uv run --no-sync modal run apps/modal/compare.py --run <unique-name>
"""

import io
import json
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/workspace")
DATA = ROOT / "artifacts/live-validation/datasets"
ADAPTER_ROOT = "/vol/runs/20261001-v2-live-04/training-full/checks/training-curve"
app = modal.App("gemma-matched-comparison")
volume = modal.Volume.from_name("gemma-unified-system-one", create_if_missing=False)
base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_sync(
        uv_project_dir=ROOT,
        extras=["inference", "train", "api", "laya", "quantization"],
        frozen=True,
    )
    .env({"HF_HOME": "/vol/hf", "TOKENIZERS_PARALLELISM": "false", "USE_TF": "0"})
)
competitor_image = (
    base_image.uv_pip_install(
        "decider-ai==1.4.0",
        "git+https://github.com/jaredpalmer/kev.git@1d77363be5769ad8c64486a51f731f940e92a59b",
        extra_options="--no-deps",
    )
    .run_commands(
        "git clone https://github.com/malevrigns/agent-jev.git /opt/agentjev",
        "git -C /opt/agentjev checkout a965ca8ff06ccabc0c796dca5447b55cc2069cee",
    )
    .env({"PYTHONPATH": "/opt/agentjev"})
)


def sources(image):
    return (
        image.add_local_python_source("s1")
        .add_local_dir(ROOT / "scripts", "/workspace/scripts")
        .add_local_dir(ROOT / "configs", "/workspace/configs")
        .add_local_dir(DATA, "/workspace/artifacts/live-validation/datasets")
        .add_local_file(ROOT / "uv.lock", "/workspace/uv.lock")
    )


def registry():
    from s1.evaluation.registry import Registry

    config = ROOT / "configs/benchmarks"
    return Registry.load(config / "models.toml", config / "suites.toml", config / "profiles.toml")


def evaluate(run, spec_data, count, arrival_rate):
    from s1.evaluation.comparison import run_cell
    from s1.evaluation.contracts import ModelSpec

    sys.path.insert(0, str(ROOT / "scripts"))
    from live_services_checks import _serve

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", run):
        raise ValueError("run must be a unique lowercase identifier")
    spec = ModelSpec.model_validate(spec_data)
    output = Path("/vol/comparisons") / run / spec.id
    try:
        receipt = run_cell(
            registry(),
            spec,
            DATA,
            output,
            ROOT / "uv.lock",
            _serve,
            count=count,
            arrival_rate=arrival_rate,
        )
    finally:
        volume.commit()
    receipt["modal_call_id"] = modal.current_function_call_id()
    receipt["modal_app_id"] = app.app_id
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    volume.commit()
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.suffix in (".json", ".jsonl", ".txt"):
                archive.write(path, path.relative_to(output))
    return {"receipt": receipt, "archive": bundle.getvalue()}


@app.function(
    image=sources(base_image),
    gpu="A100-80GB",
    cpu=4,
    memory=49152,
    volumes={"/vol": volume},
    timeout=1800,
    max_containers=1,
    single_use_containers=True,
    scaledown_window=2,
)
def gemma_cell(run, spec_data, count, arrival_rate):
    return evaluate(run, spec_data, count, arrival_rate)


@app.function(
    image=sources(competitor_image),
    gpu="A100-80GB",
    cpu=4,
    memory=49152,
    volumes={"/vol": volume},
    timeout=1800,
    max_containers=1,
    single_use_containers=True,
    scaledown_window=2,
)
def competitor_cell(run, spec_data, count, arrival_rate):
    return evaluate(run, spec_data, count, arrival_rate)


@app.local_entrypoint()
def main(
    run: str,
    models: str = "all",
    count: int = 128,
    arrival_rate: float = 5.0,
    output: str = "artifacts/comparison",
    adapter_root: str = ADAPTER_ROOT,
):
    from s1.evaluation.comparison import campaign_specs, planned_row, verify_datasets

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", run) or not 128 <= count <= 10000:
        raise ValueError("unique lowercase run and 128..10000 load requests required")
    if arrival_rate <= 0:
        raise ValueError("positive arrival rate required")
    specs = campaign_specs(registry(), adapter_root)
    selected = {s.id for s in specs} if models == "all" else set(models.split(","))
    if not selected or selected - {s.id for s in specs}:
        raise ValueError("unknown or empty model selection")
    specs = [s for s in specs if s.id in selected]
    destination = Path(output) / run
    destination.mkdir(parents=True, exist_ok=False)
    campaign = {
        "schema_version": 1,
        "campaign_id": run,
        "generated_at": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
        "dataset": verify_datasets(DATA),
        "status": "running",
        "runs": [planned_row(s) for s in specs],
        "policy": {
            "load_requests_per_cell": count,
            "arrival_rate": arrival_rate,
            "gpu": "A100-80GB",
            "concurrency": [1, 4, 16, 64],
            "quantization_targets": "accuracy>=90%; INT8<=12GB/NF4<=8GB peak allocated",
            "training": "reuse frozen CE-128 seeds0,1,2; no test fitting or seed selection",
        },
    }

    def persist():
        (destination / "campaign.json").write_text(
            json.dumps(campaign, indent=2, allow_nan=False) + "\n"
        )

    persist()
    for index, spec in enumerate(specs):
        print(f"Starting {spec.id}", flush=True)
        fn = gemma_cell if spec.adapter in ("gemma", "laya") else competitor_cell
        try:
            result = fn.remote(run, spec.model_dump(mode="json"), count, arrival_rate)
            target = destination / spec.id
            target.mkdir(exist_ok=False)
            with zipfile.ZipFile(io.BytesIO(result["archive"])) as archive:
                for member in archive.infolist():
                    if not (target / member.filename).resolve().is_relative_to(target.resolve()):
                        raise ValueError("invalid artifact archive path")
                archive.extractall(target)
            campaign["runs"][index] = result["receipt"]
            print(
                json.dumps(
                    {
                        "model": spec.id,
                        "status": result["receipt"]["status"],
                        "boolq": result["receipt"].get("boolq"),
                        "memory": result["receipt"].get("memory"),
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            campaign["runs"][index] = planned_row(spec, "failed", f"{type(exc).__name__}: {exc}")
            print(f"{spec.id}: {type(exc).__name__}: {exc}", flush=True)
        persist()
    campaign["status"] = (
        "completed"
        if all(r["status"] == "completed" for r in campaign["runs"])
        else "completed_with_errors"
    )
    persist()
    if campaign["status"] != "completed":
        raise RuntimeError(f"Campaign has failed cells; see {destination / 'campaign.json'}")
