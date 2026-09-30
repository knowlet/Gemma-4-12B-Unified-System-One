"""Bridge the existing runtimes without silently changing experiment settings."""

from __future__ import annotations

from typing import Protocol

from s1.contracts import DecisionRequest

from .contracts import Capabilities, ModelSpec, ProfileSpec


class Adapter(Protocol):
    def predict(self, request: DecisionRequest) -> dict: ...
    def telemetry(self) -> dict: ...
    def close(self) -> None: ...


def execution_blockers(spec: ModelSpec, profile: ProfileSpec) -> list[str]:
    if spec.calibration == "domain":
        spec = spec.model_copy(update={"calibration": spec.base_calibration})
    reasons = []
    if profile.batch_size != 1 or profile.concurrency != 1 or profile.require_native_batch:
        reasons.append("executor_requires_b1_c1")
    if spec.runtime != "reference":
        reasons.append("executor_runtime_unavailable")
    if spec.capabilities.probabilities != "complete":
        reasons.append("executor_requires_complete_probabilities")
    if spec.adapter == "uniform":
        if spec.execution_mode != "sequential" or spec.calibration != "none":
            reasons.append("uniform_policy_mismatch")
        if spec.precision != "float64":
            reasons.append("uniform_precision_mismatch")
    elif spec.adapter == "gemma":
        if spec.execution_mode != "causal_multislot":
            reasons.append("gemma_requires_causal_multislot")
        if spec.calibration not in ("none", "checkpoint"):
            reasons.append("gemma_calibration_unavailable")
        if spec.subfolder:
            reasons.append("gemma_subfolder_unavailable")
        if spec.processor_revision and spec.processor_revision != spec.revision:
            reasons.append("separate_processor_revision_unavailable")
        # The existing runtime chooses dtype from device; do not mislabel the run.
        if spec.device is None:
            reasons.append("gemma_requires_explicit_device")
        else:
            expected = "bfloat16" if spec.device.startswith("cuda") else "float32"
            if spec.precision != expected:
                reasons.append("gemma_device_precision_mismatch")
    elif spec.adapter == "laya":
        if spec.execution_mode != "provider" or spec.calibration != "checkpoint":
            reasons.append("laya_policy_mismatch")
        # The existing Laya wrapper does not expose precision control.
        if spec.precision != "provider":
            reasons.append("laya_precision_control_unavailable")
        if spec.processor_revision and spec.processor_revision != spec.revision:
            reasons.append("separate_processor_revision_unavailable")
    elif spec.adapter == "http":
        if (spec.execution_mode, spec.calibration, spec.precision) != (
            "provider",
            "provider",
            "provider",
        ):
            reasons.append("http_policy_mismatch")
    else:
        reasons.append("adapter_not_implemented")
    return reasons


class ReferenceAdapter:
    def __init__(self, spec: ModelSpec, backend):
        self.spec, self.backend = spec, backend

    def capabilities(self) -> Capabilities:
        return self.spec.capabilities

    def predict(self, request: DecisionRequest) -> dict:
        return self.backend.predict(request)

    def telemetry(self) -> dict:
        # Never persist arbitrary backend metadata (e.g. endpoint URLs or tokens).
        if self.spec.adapter == "gemma":
            model = self.backend.model
            return {
                "revision": model.revision,
                "device": model.device,
                "precision": str(model.head.weight.dtype).removeprefix("torch."),
                "temperature": model.temperature,
                "context_limit": model.max_context,
                "context_policy": "reject",
            }
        if self.spec.adapter == "laya":
            return {
                "revision": self.backend.metadata.get("revision"),
                "device": self.backend.metadata.get("device"),
                "precision": "provider",
                "context_limit": self.backend.max_len,
                "context_policy": "truncate",
            }
        return {
            "precision": self.spec.precision,
            "context_policy": "provider" if self.spec.adapter == "http" else "not_applicable",
        }

    def close(self) -> None:
        if hasattr(self.backend, "close"):
            self.backend.close()
        self.backend = None


def load_adapter(spec: ModelSpec, environment) -> Adapter:
    if spec.calibration == "domain":
        from .calibration import CalibratedAdapter, validate_calibration

        artifact = validate_calibration(spec)
        return CalibratedAdapter(
            load_adapter(
                spec.model_copy(update={"calibration": spec.base_calibration}), environment
            ),
            artifact,
        )
    # Optional model libraries are still imported only inside their constructors.
    from s1.backends import GemmaBackend, HTTPBackend, LayaBackend, UniformBackend

    if spec.adapter == "uniform":
        backend = UniformBackend()
    elif spec.adapter == "gemma":
        backend = GemmaBackend(
            spec.model_id,
            revision=spec.revision,
            device=spec.device,
            max_context=spec.context_limit,
            temperature=1.0 if spec.calibration == "none" else None,
        )
    elif spec.adapter == "laya":
        backend = LayaBackend(
            spec.model_id,
            revision=spec.revision,
            device=spec.device,
            subfolder=spec.subfolder,
            max_len=spec.context_limit,
        )
    elif spec.adapter == "http":
        backend = HTTPBackend(
            environment[spec.endpoint_env],
            name=spec.id,
            token=environment.get(spec.token_env) if spec.token_env else None,
            timeout=spec.http_timeout_seconds,
            supports_media=bool(set(spec.capabilities.modalities or ()) & {"image", "audio"}),
        )
    else:
        raise ValueError("adapter not implemented")
    return ReferenceAdapter(spec, backend)
