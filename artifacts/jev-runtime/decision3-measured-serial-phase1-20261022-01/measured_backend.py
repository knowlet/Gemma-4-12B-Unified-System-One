"""Explicit serial measured constructor; original inference methods are inherited."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

ORIGINAL_SHA = "aaa21ab6f2ecf5ae0e00694b43755541dc75fc0310ba10d945ade2a4511add52"


def require(value, message):
    if not value:
        raise ValueError(message)


def backend_type(original, gates):
    """Called only after the checked bootstrap ProjectSources dependency loader.

    The constructor's declared difference is the named measured gate in place
    of original require_memory. No original module/class/global is modified.
    """
    path = Path(original.__file__)
    require(
        hashlib.sha256(path.read_bytes()).hexdigest() == ORIGINAL_SHA,
        "Selected original backend source differs",
    )
    return _backend_type(original, gates)


def _backend_type(original, gates):
    class MeasuredSerialD3Backend(original.Decision3Backend):
        def __init__(
            self,
            spec,
            *,
            helper_already_loaded=False,
            expected_boot_id=None,
            expected_memtotal=None,
        ):
            require(
                os.environ.get("HF_DEACTIVATE_ASYNC_LOAD") == "1"
                and os.environ.get("HF_ENABLE_PARALLEL_LOADING") == "false"
                and os.environ.get("HF_PARALLEL_LOADING_WORKERS") == "1"
                and helper_already_loaded is True
                and expected_boot_id == gates.expected["boot_id"]
                and expected_memtotal == gates.expected["mem_total_bytes"],
                "Explicit measured serial/helper/host constructor binding",
            )
            reasons = original.configuration_blockers(spec)
            if reasons:
                raise ValueError("invalid Decision3 configuration: " + ", ".join(reasons))
            self.spec, self.name = spec, spec.id
            self.identity = next(
                i for i in original.CHECKPOINTS.values() if i["model_id"] == spec.model_id
            )
            require(
                self.identity == original.CHECKPOINTS["d3"]
                and self.identity["resident_file_bytes"] == 52176025120,
                "This measured phase1 selects full d3 only; registry remains all six",
            )
            self.memory_preflight = gates.check("before-model-constructor")
            if original.metadata.version("transformers") != "5.17.0":
                raise ValueError("Decision3 requires publisher-pinned transformers==5.17.0")
            from huggingface_hub import snapshot_download

            # Same pinned cache, publisher verifier and exact original load kwargs.
            directory = snapshot_download(
                spec.model_id, revision=spec.revision, local_files_only=True
            )
            self.manifest = original.verify_publisher_sources(directory, self.identity)
            from transformers import AutoModel

            self.model = AutoModel.from_pretrained(
                directory,
                trust_remote_code=True,
                local_files_only=True,
                device_map={"": spec.device},
                dtype="bfloat16",
                batch_size=original.NATIVE_BATCH_SIZE,
                verify="full",
            )
            self.runtime = self.model.runtime
            self.runtime_precision = str(
                next(self.runtime.backbone.parameters()).dtype
            ).removeprefix("torch.")
            if (
                self.runtime_precision != "bfloat16"
                or self.runtime.readout_dtype != "float32"
                or self.runtime.temperature != 1.0
                or self.runtime.max_length is not None
                or self.runtime.attention_mode != "noncausal_full_attention"
                or self.runtime.permutation_average
                or self.runtime.batch_size != original.NATIVE_BATCH_SIZE
            ):
                raise ValueError(
                    "loaded Decision3 runtime differs from the pinned native numerical policy"
                )

    return MeasuredSerialD3Backend
