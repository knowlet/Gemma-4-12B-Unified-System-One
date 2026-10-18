"""Read only the actual provider mount aliases using the reviewed cached image."""

import hashlib
import json
from pathlib import Path

import modal

from apps.modal.public_action_fit import HUB, ROOT, image, inputs, results, weights

POINTS = ("/", "/home", str(ROOT), str(HUB), "/input", "/export")
app = modal.App("public07-readonly-mount-probe", include_source=False)
probe_image = image.add_local_file(
    str(Path(__file__).resolve()), str(Path(__file__).resolve()), copy=True
)


@app.function(
    image=probe_image,
    cpu=1,
    memory=4096,
    ephemeral_disk=524288,
    timeout=120,
    max_containers=1,
    retries=0,
    single_use_containers=True,
    include_source=False,
    volumes={
        str(HUB): weights.with_mount_options(read_only=True),
        "/input": inputs.with_mount_options(read_only=True),
        "/export": results.with_mount_options(read_only=True),
    },
)
def inspect_mounts():
    import os
    import stat

    rows = []
    for label in POINTS:
        path = Path(label)
        row = {"path": label, "resolved": str(path.resolve(strict=True)), "ancestors": []}
        for candidate in (path, *path.parents):
            item = {"path": str(candidate), "mode": oct(candidate.lstat().st_mode)}
            if stat.S_ISLNK(candidate.lstat().st_mode):
                item["link_target"] = os.readlink(candidate)
            row["ancestors"].append(item)
        rows.append(row)
    return {"mounts": rows, "model_loads": 0, "training_updates": 0, "volume_writes": 0}


@app.local_entrypoint()
def main(output: str, run: bool = False):
    if not run:
        raise ValueError("Explicit --run for read-only CPU mount inspection required")
    destination = Path(output)
    if destination.exists():
        raise ValueError("Fresh local diagnostic output required")
    report = inspect_mounts.remote()
    report["probe_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write("\n")
