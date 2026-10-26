"""Task-owned Colab CPU metadata probe; no model, payload or credential access."""

import datetime
import importlib.metadata
import json
import pathlib
import platform
import shutil
import sys

memory = {}
for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
    key, value = line.split(":", 1)
    if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
        memory[key] = int(value.split()[0]) * 1024

packages = {}
for name in (
    "torch",
    "transformers",
    "tokenizers",
    "huggingface-hub",
    "safetensors",
    "peft",
    "numpy",
    "Pillow",
    "jinja2",
    "google-colab",
):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None

limits = {}
for name in ("memory.max", "memory.current", "memory.swap.max"):
    path = pathlib.Path("/sys/fs/cgroup") / name
    if path.is_file():
        limits[name] = path.read_text().strip()

disk = shutil.disk_usage("/content")
result = {
    "schema": "jev-colab-runtime-metadata-preflight-v1",
    "at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "requested_hardware": "CPU",
    "python": sys.version,
    "system": platform.system(),
    "machine": platform.machine(),
    "memory_bytes": memory,
    "cgroup_memory_values": limits,
    "content_disk_bytes": {"total": disk.total, "free": disk.free},
    "installed_package_versions_metadata_only": packages,
    "nvidia_smi_binary_present": shutil.which("nvidia-smi") is not None,
    "scope": "Stdlib metadata only; no framework import, GPU call, model construction, training, decoding or benchmark.",
}
with pathlib.Path("/content/jev-colab-runtime-preflight-01.json").open("x") as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print(json.dumps(result, ensure_ascii=False))
