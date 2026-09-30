"""Declarative capabilities: unknown is distinct from unsupported."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, StrictBool, model_validator

from s1.contracts import StrictModel

BASELINE_REVISION = "f8390676ae942699a71a85ae9a99cfcd7b9b3806"
Identifier = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")]
EnvironmentName = Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Primitive = Literal["choice", "noul", "score"]
Modality = Literal["text", "image", "audio"]


class ConfigModel(StrictModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True)


class Capabilities(ConfigModel):
    # None means unverified. An empty tuple explicitly supports no members.
    modalities: tuple[Modality, ...] | None = None
    primitives: tuple[Primitive, ...] | None = None
    max_options: int | None = Field(default=None, ge=2, strict=True)
    max_questions: int | None = Field(default=None, ge=1, strict=True)
    probabilities: Literal["complete", "none", "unknown"] = "unknown"
    batch: Literal["native", "loop_emulated", "none", "unknown"] = "unknown"
    concurrent_requests: StrictBool | None = None
    independent_questions: StrictBool | None = None


class ModelSpec(ConfigModel):
    id: Identifier
    adapter: Literal["uniform", "gemma", "laya", "http", "unimplemented"]
    enabled: StrictBool = False
    model_id: str | None = Field(default=None, min_length=1)
    revision: str | None = Field(default=None, min_length=1)
    processor_revision: str | None = Field(default=None, min_length=1)
    precision: Literal[
        "float64", "float32", "bfloat16", "float16", "quantized", "provider", "unknown"
    ] = "unknown"
    execution_mode: Literal[
        "sequential", "causal_multislot", "independent_batch", "provider", "unknown"
    ] = "unknown"
    calibration: Literal["none", "checkpoint", "domain", "provider", "unknown"] = "unknown"
    calibration_sha256: Digest | None = None
    runtime: str = Field(default="reference", min_length=1)
    device: str | None = Field(default=None, min_length=1)
    subfolder: str | None = Field(default=None, min_length=1)
    context_limit: int = Field(default=16384, ge=1, strict=True)
    http_timeout_seconds: float = Field(default=120, gt=0, strict=True)
    endpoint_env: EnvironmentName | None = None
    token_env: EnvironmentName | None = None
    # An operator-supplied conservative ceiling, including local compute costs.
    # None means unknown, never free. Uniform is the only intrinsically free adapter.
    cost_per_request_usd: float | None = Field(default=None, ge=0, strict=True)
    capabilities: Capabilities = Field(default_factory=Capabilities)
    notes: str = ""

    @model_validator(mode="after")
    def consistent_settings(self):
        if self.calibration == "domain" and self.calibration_sha256 is None:
            raise ValueError("domain calibration requires calibration_sha256")
        if self.adapter != "http" and (self.endpoint_env or self.token_env):
            raise ValueError("endpoint_env and token_env are only valid for HTTP adapters")
        if self.adapter == "uniform" and self.cost_per_request_usd not in (None, 0):
            raise ValueError("uniform has zero compute/API cost")
        if self.execution_mode == "causal_multislot" and self.capabilities.independent_questions:
            raise ValueError("causal_multislot cannot claim independent questions")
        return self


class SuiteSpec(ConfigModel):
    id: Identifier
    dataset: str = Field(min_length=1)
    track: Literal[
        "contract",
        "generalist",
        "specialist",
        "native_multimodal",
        "pipeline",
        "systems",
        "workflow",
    ]
    information_view: Literal["raw_text", "native_media", "verified_transcript", "pipeline_output"]
    dataset_sha256: Digest | None = None
    notes: str = ""

    @model_validator(mode="after")
    def consistent_view(self):
        if self.track == "native_multimodal" and self.information_view != "native_media":
            raise ValueError("native_multimodal track requires the native_media information view")
        if self.track == "pipeline" and self.information_view != "pipeline_output":
            raise ValueError("pipeline track requires the pipeline_output information view")
        return self


class Budget(ConfigModel):
    max_requests: int = Field(default=1000, ge=0, strict=True)
    max_estimated_cost_usd: float = Field(default=0, ge=0, strict=True)


class ProfileSpec(ConfigModel):
    id: Identifier
    suite: Identifier
    models: tuple[Identifier, ...] = Field(min_length=1)
    repetitions: int = Field(default=1, ge=1, strict=True)
    warmup: int = Field(default=0, ge=0, strict=True)
    batch_size: int = Field(default=1, ge=1, strict=True)
    concurrency: int = Field(default=1, ge=1, strict=True)
    require_native_batch: StrictBool = False
    require_independent_questions: StrictBool = False
    require_probabilities: StrictBool = False
    seed: int = Field(default=0, ge=0, strict=True)
    budget: Budget = Field(default_factory=Budget)

    @model_validator(mode="after")
    def unique_models(self):
        if len(set(self.models)) != len(self.models):
            raise ValueError("profile model ids must be unique")
        return self


class ModelRegistry(ConfigModel):
    schema_version: Literal[2]
    baseline_revision: Literal[BASELINE_REVISION]
    models: tuple[ModelSpec, ...] = Field(min_length=1)


class SuiteRegistry(ConfigModel):
    schema_version: Literal[2]
    suites: tuple[SuiteSpec, ...] = Field(min_length=1)


class ProfileRegistry(ConfigModel):
    schema_version: Literal[2]
    profiles: tuple[ProfileSpec, ...] = Field(min_length=1)
