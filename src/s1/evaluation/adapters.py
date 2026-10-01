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
    if spec.preprocessing != "none" and spec.capabilities.batch == "native":
        reasons.append("pipeline_batch_is_loop_emulated")
    if profile.batch_size > 1 and not (
        (
            spec.adapter == "gemma"
            and spec.execution_mode == "independent_batch"
            and spec.readout != "generate"
        )
        or spec.capabilities.batch == "loop_emulated"
    ):
        reasons.append("executor_batch_unavailable")
    if profile.concurrency > 1 and spec.adapter in ("gemma", "laya", "decider", "kev", "agentjev"):
        reasons.append("local_model_concurrency_unavailable")
    if spec.runtime != "reference":
        reasons.append("executor_runtime_unavailable")
    if spec.capabilities.probabilities == "unknown":
        reasons.append("executor_requires_known_probability_contract")
    if spec.adapter != "gemma" and spec.readout != "candidate":
        reasons.append("readout_only_supported_by_gemma")
    if spec.adapter == "uniform":
        if spec.execution_mode != "sequential" or spec.calibration != "none":
            reasons.append("uniform_policy_mismatch")
        if spec.precision != "float64":
            reasons.append("uniform_precision_mismatch")
    elif spec.adapter == "gemma":
        if spec.quantization != "none" and (
            not (spec.device or "").startswith("cuda") or spec.precision != "bfloat16"
        ):
            reasons.append("gemma_quantization_requires_cuda_bfloat16")
        if spec.execution_mode not in ("causal_multislot", "independent_batch", "sequential"):
            reasons.append("gemma_execution_mode_unavailable")
        if spec.readout == "generate" and (
            spec.execution_mode != "sequential" or spec.capabilities.probabilities != "none"
        ):
            reasons.append("generation_requires_sequential_hard_labels")
        if spec.readout != "generate" and spec.capabilities.probabilities != "complete":
            reasons.append("logit_readout_requires_complete_probabilities")
        if spec.calibration not in ("none", "checkpoint"):
            reasons.append("gemma_calibration_unavailable")
        if spec.subfolder:
            reasons.append("gemma_subfolder_unavailable")
        if spec.processor_revision and spec.processor_revision != spec.revision:
            reasons.append("separate_processor_revision_unavailable")
        # Precision is explicit: CUDA may use FP32 for strict batch parity;
        # the default CUDA BF16 path retains its distinct runtime identity.
        if spec.device is None:
            reasons.append("gemma_requires_explicit_device")
        else:
            supported = ("bfloat16", "float32") if spec.device.startswith("cuda") else ("float32",)
            if spec.precision not in supported:
                reasons.append("gemma_device_precision_mismatch")
    elif spec.adapter == "laya":
        if spec.execution_mode != "provider" or spec.calibration != "checkpoint":
            reasons.append("laya_policy_mismatch")
        # The existing Laya wrapper does not expose precision control.
        if spec.precision != "provider":
            reasons.append("laya_precision_control_unavailable")
        if spec.processor_revision and spec.processor_revision != spec.revision:
            reasons.append("separate_processor_revision_unavailable")
    elif spec.adapter in ("decider", "kev", "agentjev"):
        from .local_competitors import CHECKPOINTS

        identity = CHECKPOINTS[spec.adapter]
        if spec.execution_mode != "sequential" or spec.calibration != "checkpoint":
            reasons.append("local_competitor_policy_mismatch")
        if not spec.device or spec.precision != identity.precision:
            reasons.append("local_competitor_device_precision_mismatch")
        if (spec.model_id, spec.revision) != (identity.model_id, identity.revision):
            reasons.append("local_competitor_checkpoint_unverified")
        if spec.context_limit > identity.context_limit:
            reasons.append("local_competitor_context_limit_exceeded")
        if spec.capabilities.probabilities != "complete":
            reasons.append("local_competitor_requires_complete_probabilities")
        if spec.subfolder or (spec.processor_revision and spec.processor_revision != spec.revision):
            reasons.append("local_competitor_processor_override_unavailable")
    elif spec.adapter == "http":
        if (spec.execution_mode, spec.calibration, spec.precision) != (
            "provider",
            "provider",
            "provider",
        ):
            reasons.append("http_policy_mismatch")
    elif spec.adapter in ("tfidf", "prior", "embedding", "nli", "cross_encoder", "setfit"):
        if spec.execution_mode != "sequential" or spec.calibration not in ("none", "checkpoint"):
            reasons.append("baseline_policy_mismatch")
        expected = "float64" if spec.adapter in ("tfidf", "prior") else "float32"
        if spec.precision != expected:
            reasons.append("baseline_precision_mismatch")
        expected_probabilities = (
            "none" if spec.adapter in ("embedding", "nli", "cross_encoder") else "complete"
        )
        if spec.capabilities.probabilities != expected_probabilities:
            reasons.append("baseline_probability_contract_mismatch")
    else:
        reasons.append("adapter_not_implemented")
    return reasons


class ReferenceAdapter:
    def __init__(self, spec: ModelSpec, backend):
        self.spec, self.backend = spec, backend

    def capabilities(self) -> Capabilities:
        return self.spec.capabilities

    def predict(self, request: DecisionRequest) -> dict:
        if self.spec.adapter == "gemma":
            from .gemma import predict_batch, predict_generated, predict_sequential

            if self.spec.readout == "generate":
                return predict_generated(self.backend.model, request)
            if self.spec.execution_mode == "sequential":
                return predict_sequential(self.backend.model, request, readout=self.spec.readout)
            return predict_batch(
                self.backend.model,
                [request],
                readout=self.spec.readout,
                independent=self.spec.execution_mode == "independent_batch",
            )[0]
        return self.backend.predict(request)

    def predict_batch(self, requests):
        if self.spec.adapter == "gemma" and self.spec.execution_mode == "independent_batch":
            from .gemma import predict_batch

            return predict_batch(self.backend.model, requests, readout=self.spec.readout)
        if self.spec.capabilities.batch == "loop_emulated":
            return [self.predict(request) for request in requests]
        raise ValueError("native batching is unavailable for this adapter")

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
                "quantization": model.quantization,
                "quantization_details": model.quantization_details,
            }
        if self.spec.adapter in ("decider", "kev", "agentjev"):
            return self.backend.telemetry()
        if self.spec.adapter == "laya":
            return {
                "revision": self.backend.metadata.get("revision"),
                "device": self.backend.metadata.get("device"),
                "precision": "provider",
                "context_limit": self.backend.max_len,
                "context_policy": "truncate",
            }
        if self.spec.adapter in ("embedding", "setfit"):
            return {"precision": "float32", "context_policy": "truncate"}
        if self.spec.adapter in ("nli", "cross_encoder"):
            return {
                "precision": "float32",
                "context_policy": "reject",
                "context_limit": self.spec.context_limit,
            }
        return {
            "precision": self.spec.precision,
            "context_policy": "provider" if self.spec.adapter == "http" else "not_applicable",
        }

    def resources(self):
        if self.spec.adapter == "gemma":
            return self.backend.model.memory_snapshot()
        if self.spec.adapter in ("decider", "kev", "agentjev"):
            return self.backend.resources()
        import resource
        import sys

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result = {"process_peak_rss_kib": rss / 1024 if sys.platform == "darwin" else rss}
        if self.spec.adapter == "gemma" and self.backend.model.device.startswith("cuda"):
            import torch

            device = self.backend.model.device
            result.update(
                gpu_name=torch.cuda.get_device_name(device),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            )
        return result

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
    if spec.preprocessing != "none":
        from .multimodal import wrap_pipeline

        return wrap_pipeline(
            load_adapter(spec.model_copy(update={"preprocessing": "none"}), environment),
            spec,
            environment,
        )
    # Optional model libraries are still imported only inside their constructors.
    if spec.decision_adapter_path:
        from .checkpoints import directory_digest

        if directory_digest(spec.decision_adapter_path) != spec.decision_adapter_sha256:
            raise ValueError("decision adapter changed after planning")
    from s1.backends import GemmaBackend, LayaBackend, UniformBackend

    if spec.adapter == "uniform":
        backend = UniformBackend()
    elif spec.adapter == "gemma":
        backend = GemmaBackend(
            spec.model_id,
            revision=spec.revision,
            device=spec.device,
            precision=spec.precision,
            quantization=spec.quantization,
            max_context=spec.context_limit,
            temperature=1.0 if spec.calibration == "none" else None,
            **({"adapter_path": spec.decision_adapter_path} if spec.decision_adapter_path else {}),
        )
    elif spec.adapter == "laya":
        backend = LayaBackend(
            spec.model_id,
            revision=spec.revision,
            device=spec.device,
            subfolder=spec.subfolder,
            max_len=spec.context_limit,
        )
    elif spec.adapter in ("decider", "kev", "agentjev"):
        from .local_competitors import LocalCompetitorBackend

        backend = LocalCompetitorBackend(spec)
    elif spec.adapter == "http":
        from .providers import ProviderHTTPBackend

        backend = ProviderHTTPBackend(
            environment[spec.endpoint_env],
            name=spec.id,
            token=environment.get(spec.token_env) if spec.token_env else None,
            timeout=spec.http_timeout_seconds,
            supports_media=bool(set(spec.capabilities.modalities or ()) & {"image", "audio"}),
            protocol=spec.protocol,
        )
    elif spec.adapter in ("tfidf", "prior", "embedding", "nli", "cross_encoder", "setfit"):
        from .baselines import BaselineBackend

        backend = BaselineBackend(spec)
    else:
        raise ValueError("adapter not implemented")
    return ReferenceAdapter(spec, backend)
