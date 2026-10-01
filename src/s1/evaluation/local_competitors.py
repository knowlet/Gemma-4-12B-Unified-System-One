"""Pinned, lazy native runtimes for the public decision-model checkpoints.

These adapters use each publisher's encoder, readout and checkpoint calibration.
They never generate surrogate answers for unsupported image or audio inputs.
"""

from __future__ import annotations

import json
import math
import subprocess
import threading
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

from s1.backends import normalize_response
from s1.errors import RequestValidationError

from .contracts import Capabilities, ModelSpec
from .providers import (
    agentjev_payload,
    normalize_agentjev_response,
    normalize_typesafe_rounding,
    typesafe_payload,
)


@dataclass(frozen=True)
class CheckpointIdentity:
    model_id: str
    revision: str
    runtime_package: str
    source_revision: str | None = None
    base_model_id: str | None = None
    base_revision: str | None = None
    precision: str = "bfloat16"
    context_limit: int = 8192
    checkpoint_task: str = "typed_decisions"


CHECKPOINTS = {
    "decider": CheckpointIdentity(
        "Mapika/decider-2b",
        "533964dae8be954c5b5e19fa4948e48408094c1e",
        "decider-ai==1.4.0",
        context_limit=32768,
    ),
    "kev": CheckpointIdentity(
        "jaredpalmer/kev-0.8b",
        "9a45d25eb2ab761841196625383fa1dff0e56c1e",
        "kev",
        source_revision="1d77363be5769ad8c64486a51f731f940e92a59b",
        base_model_id="Qwen/Qwen3.5-0.8B-Base",
        base_revision="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
    ),
    "agentjev": CheckpointIdentity(
        "aimeigaoshou/agent-jev",
        "0e2593e6e6c0eade0700712ac13c4389aa7654cc",
        "agentjev-source",
        source_revision="a965ca8ff06ccabc0c796dca5447b55cc2069cee",
        base_model_id="Qwen/Qwen3-0.6B",
        base_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        precision="float32",
        context_limit=2048,
        checkpoint_task="coding_completion",
    ),
}


def local_competitor_spec(name, *, device="cuda", cost_per_request_usd=None):
    """Construct the same ModelSpec used by planning, quality and load runners."""
    identity = CHECKPOINTS[name]
    return ModelSpec(
        id=f"{name}-local",
        adapter=name,
        enabled=True,
        model_id=identity.model_id,
        revision=identity.revision,
        precision=identity.precision,
        device=device,
        execution_mode="sequential",
        calibration="checkpoint",
        context_limit=identity.context_limit,
        cost_per_request_usd=cost_per_request_usd,
        capabilities=Capabilities(
            modalities=("text",),
            primitives=("choice", "noul", "score"),
            max_options=52,
            max_questions=64,
            probabilities="complete",
            batch="loop_emulated",
            concurrent_requests=False,
            independent_questions=True,
        ),
        notes=(
            "Native eager inference with checkpoint calibration; text-only. "
            + (
                "FP32 coding-completion weights with upstream BF16 CUDA autocast; "
                "not the historical typed-decisions specialist checkpoint."
                if name == "agentjev"
                else "No persistent prefix cache, CUDA graphs or compilation."
            )
        ),
    )


def _verify_source(package, revision, module):
    """Accept a pinned VCS wheel or an exact checked-out source tree."""
    try:
        direct = metadata.distribution(package).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        direct = None
    if direct:
        recorded = json.loads(direct).get("vcs_info", {}).get("commit_id")
        if recorded == revision:
            return
    root = Path(module.__file__).resolve().parent.parent
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode or result.stdout.strip() != revision:
        raise ValueError(f"{package} runtime must use pinned source revision {revision}")


def _snapshot(model_id, revision, *, patterns=None):
    from huggingface_hub import snapshot_download

    return snapshot_download(model_id, revision=revision, allow_patterns=patterns)


def _load_agentjev(spec, identity):
    import agentjev
    import torch
    from agentjev.model import AgentJevModel
    from jev_service.engine import DecisionEngine
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    _verify_source("agentjev", identity.source_revision, agentjev)
    checkpoint = Path(_snapshot(spec.model_id, spec.revision))
    base = _snapshot(identity.base_model_id, identity.base_revision)
    temperatures = json.loads((checkpoint / "temperatures.json").read_text())
    parsed = {}
    for kind, entry in temperatures.items():
        value = float(entry["temperature"])
        if kind not in ("boolean", "choice", "score") or not math.isfinite(value):
            raise ValueError("invalid AgentJev checkpoint calibration")
        if not 0.05 <= value <= 20:
            raise ValueError("invalid AgentJev checkpoint temperature")
        parsed[kind] = value
    # Reuse the publisher's evaluate method, while loading public safetensors
    # directly instead of converting them into an unrestricted pickle checkpoint.
    engine = DecisionEngine.__new__(DecisionEngine)
    engine.torch, engine.device = torch, spec.device
    engine.lock = threading.Lock()
    engine.max_tokens, engine.path_batch, engine.encoder = spec.context_limit, 16, "auto"
    engine.temperatures = parsed
    engine.tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    if engine.tokenizer.pad_token_id is None:
        engine.tokenizer.pad_token = engine.tokenizer.eos_token
    engine.model = AgentJevModel(base, dtype=torch.float32)
    weights = load_file(str(checkpoint / "model.safetensors"))
    engine.model.load_state_dict(weights, strict=True)
    del weights
    engine.model.eval().to(spec.device)
    engine.checkpoint_name = identity.checkpoint_task
    return engine


class LocalCompetitorBackend:
    def __init__(self, spec):
        self.spec = spec
        self.identity = CHECKPOINTS[spec.adapter]
        identity = self.identity
        if (spec.model_id, spec.revision) != (identity.model_id, identity.revision):
            raise ValueError("unsupported checkpoint identity; add and verify an immutable pin")
        if not spec.device or spec.precision != identity.precision:
            raise ValueError("native competitor requires its declared device and precision")
        self.engine = self.model = self.tokenizer = None
        if spec.adapter == "decider":
            import torch
            from decider.infer import Decider

            if metadata.version("decider-ai") != "1.4.0":
                raise ValueError("Decider v11 requires pinned decider-ai==1.4.0")
            path = _snapshot(spec.model_id, spec.revision)
            self.engine = Decider(path, device=spec.device, dtype=torch.bfloat16, use_graphs=False)
            self.model, self.tokenizer = self.engine.m, self.engine.m.tok
        elif spec.adapter == "kev":
            import kev
            import torch
            from kev.checkpoint import Checkpoint, LoadOptions

            _verify_source("kev", identity.source_revision, kev)
            checkpoint = Checkpoint(_snapshot(spec.model_id, spec.revision))
            saved = checkpoint.meta.base_revision
            if (
                checkpoint.meta.base != identity.base_model_id
                or not saved
                or not identity.base_revision.startswith(saved)
            ):
                raise ValueError("Kev checkpoint base differs from the pinned base")
            # Upstream head.pt stores a short Git SHA; resolve to our full pin.
            checkpoint.meta.base_revision = identity.base_revision
            self.tokenizer, self.model = checkpoint.load(
                spec.device,
                LoadOptions(
                    dtype=torch.bfloat16,
                    backend="torch",
                    cuda_graphs=False,
                    fused=False,
                    merge=True,
                ),
            )
        else:
            self.engine = _load_agentjev(spec, identity)
            self.model, self.tokenizer = self.engine.model, self.engine.tokenizer

    def predict(self, request):
        if request.media:
            raise RequestValidationError("native competitor checkpoint is text-only")
        if self.spec.adapter == "agentjev":
            return normalize_agentjev_response(
                request, self.engine.evaluate(agentjev_payload(request))
            )
        payload = typesafe_payload(request)
        if self.spec.adapter == "decider":
            # Upstream truncates overlong states. Preflight its exact rows with an
            # unlimited state budget so we can reject instead of losing evidence.
            _, _, rows = self.engine._system_one_items(
                **payload, independent=True, max_state_tokens=2**31 - 1
            )
            if any(len(row["ids"]) > self.spec.context_limit for row in rows):
                raise RequestValidationError("Decider context limit exceeded")
            raw = self.engine.system_one(
                **payload, independent=True, max_state_tokens=self.spec.context_limit
            )
        else:
            from kev.api import SystemOneRequest, to_answers, to_record

            record, mapping = to_record(SystemOneRequest.model_validate(payload))
            encoded = self.model.encode(
                self.tokenizer,
                record,
                max_state=self.spec.context_limit,
                max_branch=self.spec.context_limit,
                strict=True,
            )
            probabilities = [p.tolist() for p in self.model.probs(encoded)]
            raw = {"answers": to_answers(probabilities, mapping)}
        return normalize_response(request, normalize_typesafe_rounding(raw), ordinal_scores=True)

    def telemetry(self):
        installed = {}
        for package in ("causal-conv1d", "flash-linear-attention"):
            try:
                installed[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                installed[package] = None
        result = {
            "revision": self.spec.revision,
            "device": self.spec.device,
            "precision": self.spec.precision,
            "context_limit": self.spec.context_limit,
            "context_policy": "reject",
            "runtime_package": self.identity.runtime_package,
            "source_revision": self.identity.source_revision,
            "base_model_id": self.identity.base_model_id,
            "base_revision": self.identity.base_revision,
            "checkpoint_task": self.identity.checkpoint_task,
            "compute_precision": self.spec.precision,
            "compute_backend": "torch_reference",
            "runtime_options": {
                "cuda_graphs": False,
                "torch_compile": False,
                "persistent_prefix_cache": False,
                "optional_kernel_packages": installed,
            },
        }
        if self.spec.adapter == "decider":
            result.update(temperature=self.engine.T, temperatures_by_type=self.engine.T_by_type)
        elif self.spec.adapter == "kev":
            result["temperature"] = self.model.head.temperature
        else:
            result["temperatures_by_type"] = self.engine.temperatures
            if self.spec.device.startswith("cuda"):
                result["compute_precision"] = "bfloat16_autocast"
        return result

    def resources(self):
        from s1.resources import memory_snapshot

        return memory_snapshot(self.model, device=self.spec.device)

    def close(self):
        self.engine = self.model = self.tokenizer = None
