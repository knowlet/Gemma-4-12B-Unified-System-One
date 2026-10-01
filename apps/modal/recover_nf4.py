"""Run the predeclared three-seed NF4 recovery, then fresh matched inference.

uv run --no-sync modal run apps/modal/recover_nf4.py --run <unique-name>
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
app = modal.App("gemma-nf4-recovery")
volume = modal.Volume.from_name("gemma-unified-system-one", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_sync(
        uv_project_dir=ROOT,
        extras=["inference", "train", "api", "laya", "quantization"],
        frozen=True,
    )
    .env({"HF_HOME": "/vol/hf", "TOKENIZERS_PARALLELISM": "false", "USE_TF": "0"})
    .add_local_python_source("s1")
    .add_local_dir(ROOT / "scripts", "/workspace/scripts")
    .add_local_dir(ROOT / "configs", "/workspace/configs")
    .add_local_dir(DATA, "/workspace/artifacts/live-validation/datasets")
    .add_local_file(ROOT / "uv.lock", "/workspace/uv.lock")
)


def helpers():
    sys.path.insert(0, str(ROOT / "scripts"))
    import nf4_recovery

    return nf4_recovery


def registry():
    from s1.evaluation.registry import Registry

    config = ROOT / "configs/benchmarks"
    return Registry.load(config / "models.toml", config / "suites.toml", config / "profiles.toml")


def paths(run, seed):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", run) or seed not in range(3):
        raise ValueError("unique lowercase run and predeclared seed required")
    root = Path("/vol/comparisons") / run
    return root / "adapters" / f"seed-{seed}", root / "training" / f"seed-{seed}.json"


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4,
    memory=49152,
    volumes={"/vol": volume},
    timeout=1800,
    max_containers=3,
    single_use_containers=True,
    scaledown_window=2,
)
def train_seed(run, seed):
    adapter, receipt_path = paths(run, seed)
    helper = helpers()
    try:
        try:
            report = helper.train_recovery(
                DATA,
                Path(ADAPTER_ROOT) / f"shots-128-seed-{seed}-ce",
                adapter,
                receipt_path,
                seed=seed,
                lockfile=ROOT / "uv.lock",
            )
        except Exception as exc:
            report = (
                json.loads(receipt_path.read_text())
                if receipt_path.exists()
                else {
                    "status": "failed",
                    "recovery": helper.recipe(seed),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        report.update(modal_call_id=modal.current_function_call_id(), modal_app_id=app.app_id)
        helper.write_json(receipt_path, report)
        return report
    finally:
        volume.commit()


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4,
    memory=49152,
    volumes={"/vol": volume},
    timeout=1800,
    max_containers=3,
    single_use_containers=True,
    scaledown_window=2,
)
def evaluate_seed(run, spec_data, training_report, count, arrival_rate):
    import hashlib

    from s1.evaluation.comparison import run_cell
    from s1.evaluation.contracts import ModelSpec

    helper = helpers()
    from live_services_checks import _serve

    volume.reload()
    spec = ModelSpec.model_validate(spec_data)
    output = Path("/vol/comparisons") / run / spec.id
    try:
        parity = helper.check_fresh_calibration(DATA, training_report)
        print(f"{spec.id}: fresh calibration reload parity passed", flush=True)
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
        helper.write_json(output / "training-receipt.json", training_report)
        helper.write_json(output / "calibration-parity.json", parity)
        receipt.update(
            recovery=training_report["recovery"],
            modal_call_id=modal.current_function_call_id(),
            modal_app_id=app.app_id,
        )
        receipt["artifacts"].update(
            training="training-receipt.json", calibration_parity="calibration-parity.json"
        )
        for name in ("training-receipt.json", "calibration-parity.json"):
            receipt["artifact_sha256"][name] = hashlib.sha256(
                (output / name).read_bytes()
            ).hexdigest()
        helper.write_json(output / "receipt.json", receipt)
        archive_bytes = io.BytesIO()
        with zipfile.ZipFile(archive_bytes, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output.rglob("*")):
                if path.is_file() and path.suffix in (".json", ".jsonl", ".txt"):
                    archive.write(path, path.relative_to(output))
        return {"receipt": receipt, "archive": archive_bytes.getvalue()}
    finally:
        volume.commit()


@app.local_entrypoint()
def main(
    run: str, count: int = 128, arrival_rate: float = 5.0, output: str = "artifacts/comparison"
):
    from s1.evaluation.comparison import campaign_specs, planned_row

    helper = helpers()
    paths(run, 0)
    if not 128 <= count <= 10000 or arrival_rate <= 0:
        raise ValueError("128..10000 requests and positive arrival rate required")
    _, _, _, dataset = helper.prepare_data(DATA)
    parents = {s.id: s for s in campaign_specs(registry(), ADAPTER_ROOT)}
    specs = [
        parents[f"gemma-ce128-s{seed}-nf4"].model_copy(
            update={
                "id": f"gemma-ce128-s{seed}-nf4-recovered",
                "decision_adapter_path": str(paths(run, seed)[0]),
                "decision_adapter_sha256": None,
            }
        )
        for seed in range(3)
    ]
    destination = Path(output) / run
    destination.mkdir(parents=True, exist_ok=False)
    campaign = {
        "schema_version": 1,
        "campaign_id": run,
        "generated_at": datetime.now(ZoneInfo("Asia/Taipei")).isoformat(),
        "dataset": dataset,
        "status": "running",
        "runs": [planned_row(s) for s in specs],
        "training": [{"status": "not_run", "recovery": helper.recipe(seed)} for seed in range(3)],
        "policy": {
            "load_requests_per_cell": count,
            "arrival_rate": arrival_rate,
            "gpu": "A100-80GB",
            "concurrency": [1, 4, 16, 64],
            "training": "fixed exploratory NF4 LoRA continuation; all 3 seeds; no test fitting or selection",
            "recipes": [helper.recipe(seed) for seed in range(3)],
            "memory": "fresh inference container; calibration parity unloaded before standard harness",
        },
    }

    def persist():
        helper.write_json(destination / "campaign.json", campaign)

    persist()
    print(f"Starting all three fixed recovery seeds; Modal app {app.app_id}", flush=True)
    calls = [train_seed.spawn(run, seed) for seed in range(3)]
    for seed, call in enumerate(calls):
        try:
            report = call.get()
            campaign["training"][seed] = report
            helper.write_json(destination / f"training-seed-{seed}.json", report)
            if report["status"] != "completed":
                raise RuntimeError(report.get("error", "training did not complete"))
            specs[seed] = specs[seed].model_copy(
                update={"decision_adapter_sha256": report["adapter_sha256"]}
            )
            campaign["runs"][seed] = {**planned_row(specs[seed]), "recovery": report["recovery"]}
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            campaign["training"][seed].update(status="failed", error=message)
            campaign["runs"][seed] = {
                **planned_row(specs[seed], "failed", message),
                "recovery": helper.recipe(seed),
            }
            print(f"seed {seed} training failed: {message}", flush=True)
        persist()
    # Phase barrier bounds the campaign to three GPU containers at a time and
    # ensures every evaluation uses a fresh, single-use inference process.
    evaluations = [
        (
            seed,
            evaluate_seed.spawn(
                run,
                specs[seed].model_dump(mode="json"),
                campaign["training"][seed],
                count,
                arrival_rate,
            ),
        )
        for seed in range(3)
        if campaign["training"][seed]["status"] == "completed"
    ]
    for seed, call in evaluations:
        try:
            result = call.get()
            target = destination / specs[seed].id
            target.mkdir(exist_ok=False)
            with zipfile.ZipFile(io.BytesIO(result["archive"])) as archive:
                for member in archive.infolist():
                    if not (target / member.filename).resolve().is_relative_to(target.resolve()):
                        raise ValueError("invalid artifact archive path")
                archive.extractall(target)
            campaign["runs"][seed] = result["receipt"]
            print(
                json.dumps(
                    {
                        "model": specs[seed].id,
                        "status": result["receipt"]["status"],
                        "boolq": result["receipt"].get("boolq"),
                        "memory": result["receipt"].get("memory"),
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            campaign["runs"][seed] = {
                **planned_row(specs[seed], "failed", message),
                "recovery": helper.recipe(seed),
            }
            print(f"seed {seed} evaluation failed: {message}", flush=True)
        persist()
    campaign["status"] = (
        "completed"
        if all(r["status"] == "completed" for r in campaign["runs"])
        else "completed_with_errors"
    )
    persist()
    if campaign["status"] != "completed":
        raise RuntimeError(f"Recovery has failed cells; see {destination / 'campaign.json'}")
