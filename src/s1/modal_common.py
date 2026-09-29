"""Shared, locked environment for the retained upstream Modal applications."""

import os

import modal

vol = modal.Volume.from_name(os.environ.get("S1_VOLUME", "jev-replica"), create_if_missing=True)
base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(extras=["inference", "train", "api"], frozen=True, extra_options="--no-dev")
    .env({"HF_HOME": "/vol/hf", "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"})
)
image = base_image.add_local_python_source("s1")


def gpu_kwargs(default="H100"):
    g = os.environ.get("S1_GPU", default)
    if g.lower() == "none":
        return dict(
            cpu=float(os.environ.get("S1_CPU", "8")), memory=int(os.environ.get("S1_MEM", "32768"))
        )
    return dict(gpu=g)
