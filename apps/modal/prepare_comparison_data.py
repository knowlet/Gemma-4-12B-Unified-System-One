"""Recreate historical media on x86 and accept only the archived dataset pins."""

import io
import json
import shutil
import zipfile
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/workspace")
DATA = ROOT / "artifacts/live-validation/datasets"
app = modal.App("gemma-comparison-data-recovery")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "numpy==2.4.6",
        "scipy==1.17.1",
        "pillow==12.3.0",
        "soundfile==0.14.0",
        "pyarrow==25.0.1",
        "pydantic>=2.11,<3",
        "httpx>=0.28,<1",
    )
    .add_local_python_source("s1")
    .add_local_dir(ROOT / "scripts", "/workspace/scripts")
    .add_local_dir(DATA / "raw", "/workspace/raw")
    .add_local_file(
        ROOT / "docs/validation/2026-10-01/live-summary.json", "/workspace/historical.json"
    )
)


@app.function(image=image, cpu=2, memory=4096, timeout=300, max_containers=1)
def prepare():
    import importlib.metadata
    import platform
    import sys

    sys.path.insert(0, "/workspace/scripts")
    from prepare_live_validation_data import prepare as prepare_data

    destination = Path("/tmp/recovered-datasets")
    destination.mkdir()
    shutil.copytree("/workspace/raw", destination / "raw")
    result = prepare_data(destination)
    historical = json.loads(Path("/workspace/historical.json").read_text())["data"]
    matches = {
        name: result["datasets"][name]["dataset_sha256"] == expected["dataset_sha256"]
        for name, expected in historical["datasets"].items()
    }
    diagnostics = {
        "architecture": platform.machine(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "pillow", "soundfile", "pyarrow")
        },
        "matches": matches,
        "datasets": result["datasets"],
        "media_bundle_sha256": result["media_bundle_sha256"],
    }
    (destination / "recovery.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(destination.glob("*.json*")):
            archive.write(path, path.name)
    return {"diagnostics": diagnostics, "archive": bundle.getvalue()}


@app.local_entrypoint()
def main():
    result = prepare.remote()
    output = ROOT / "artifacts/comparison-data-x86"
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(result["archive"])) as archive:
        archive.extractall(output)
    print(json.dumps(result["diagnostics"], indent=2))
    diagnostics = result["diagnostics"]
    if not all(diagnostics["matches"].values()):
        raise ValueError("reconstructed data differs from historical pins; files not replaced")
    historical = json.loads((ROOT / "docs/validation/2026-10-01/live-summary.json").read_text())[
        "data"
    ]
    if diagnostics["media_bundle_sha256"] != historical["media_bundle_sha256"]:
        raise ValueError("reconstructed media bundle differs; files not replaced")
    backup = ROOT / "artifacts/comparison-data-arm"
    backup.mkdir(parents=True, exist_ok=True)
    for path in DATA.glob("*.json*"):
        if not (backup / path.name).exists():
            shutil.copy2(path, backup / path.name)
    for path in output.glob("*.json*"):
        if path.name != "recovery.json":
            shutil.copy2(path, DATA / path.name)
    print("Verified historical dataset and media bundle pins; original inputs restored.")
