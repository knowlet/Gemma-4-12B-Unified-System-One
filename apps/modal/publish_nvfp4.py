"""Prepare and publish only the verified NVFP4 package already on the Volume."""

import json
import re
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/workspace")
app = modal.App("gemma-s1-publish-nvfp4")
volume = modal.Volume.from_name("gemma-unified-system-one", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface-hub==1.33.0", "numpy==2.5.3", "pydantic==2.13.5")
    .add_local_python_source("s1")
    .add_local_dir(ROOT / "scripts", "/workspace/scripts")
    .add_local_dir(ROOT / "docs/validation/2026-10-07", "/workspace/docs/validation/2026-10-07")
    .add_local_dir(
        ROOT / "artifacts/nvfp4/publication-assets", "/workspace/artifacts/nvfp4/publication-assets"
    )
    .add_local_dir(ROOT / "artifacts/nvfp4/runtime", "/workspace/artifacts/nvfp4/runtime")
    .add_local_file(ROOT / "requirements-nvfp4.txt", "/workspace/requirements-nvfp4.txt")
    .add_local_file(
        ROOT / "docs/validation/2026-10-02/release-datasets.tar.gz",
        "/workspace/docs/validation/2026-10-02/release-datasets.tar.gz",
    )
)


def _paths(run):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", run):
        raise ValueError("invalid run identifier")
    root = Path("/vol/nvfp4") / run
    return root, root / "package"


def _secret():
    from huggingface_hub import get_token

    token = get_token()
    # Keep one dependency in both local and container imports. The remote
    # function receives its credential through Modal's ephemeral secret.
    return [modal.Secret.from_dict({"HF_TOKEN": token or ""})]


@app.function(
    image=image, cpu=4, memory=16384, volumes={"/vol": volume}, timeout=1800, max_containers=1
)
def prepare(run: str):
    import subprocess
    import sys

    volume.reload()
    root, package = _paths(run)
    subprocess.run(
        [sys.executable, "/workspace/scripts/package_nvfp4_release.py", "--package", str(package)],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "/workspace/scripts/publish_nvfp4.py",
            "--model",
            str(package),
            "--file-list",
            str(root / "upload-files.json"),
            "--receipt",
            str(root / "preupload-verification.json"),
            "--validate-only",
        ],
        check=True,
    )
    volume.commit()
    names = [
        "README.md",
        "artifact-manifest.json",
        "runtime-verification.json",
        "calibration.json",
        "reload-parity.json",
        "NOTICE",
    ]
    return {
        "status": "prepared",
        "files": {name: (package / name).read_text() for name in names},
        "allowlist": json.loads((root / "upload-files.json").read_text()),
        "validation": json.loads((root / "preupload-verification.json").read_text()),
    }


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    volumes={"/vol": volume},
    secrets=_secret(),
    timeout=3600,
    max_containers=1,
)
def publish(run: str):
    import subprocess
    import sys

    volume.reload()
    root, package = _paths(run)
    try:
        subprocess.run(
            [
                sys.executable,
                "/workspace/scripts/publish_nvfp4.py",
                "--model",
                str(package),
                "--file-list",
                str(root / "upload-files.json"),
                "--receipt",
                str(root / "publication.json"),
            ],
            check=True,
        )
    finally:
        volume.commit()
    return json.loads((root / "publication.json").read_text())


@app.local_entrypoint()
def main(stage: str, run: str):
    _paths(run)
    if stage not in ("prepare", "publish"):
        raise ValueError("stage must be prepare or publish")
    result = prepare.remote(run) if stage == "prepare" else publish.remote(run)
    local = ROOT / "artifacts/nvfp4" / run
    local.mkdir(parents=True, exist_ok=True)
    if stage == "prepare":
        card = local / "publication-review"
        card.mkdir(exist_ok=True)
        for name, content in result["files"].items():
            (card / name).write_text(content)
        (local / "upload-files.json").write_text(json.dumps(result["allowlist"], indent=2) + "\n")
        (local / "preupload-verification.json").write_text(
            json.dumps(result["validation"], indent=2) + "\n"
        )
    else:
        (local / "publication.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: result[key] for key in ("status", "url", "published_revision") if key in result}
        )
    )
