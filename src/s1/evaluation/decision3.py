"""Publisher-native Decision 3.0: pinned code, BF16 backbone and FP32 readout.

The 16,384-token refusal policy is this project's prospective resource cap.
The publisher's decision max_length is None; 262,144 is architecture/tokenizer
metadata, not measured context support. No weights are fetched by this adapter:
an explicitly provisioned complete immutable local HF snapshot is required.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from importlib import metadata
from pathlib import Path

from s1.backends import normalize_response
from s1.contracts import AudioInput, validate_distribution
from s1.errors import BackendResponseError, RequestValidationError

from .contracts import Capabilities, ModelSpec

MAX_OPTIONS = 255
CONTEXT_LIMIT = 16384
ARCHITECTURAL_CONTEXT = 262144
FIXED_HELPER_BYTES = 23919549408
NATIVE_BATCH_SIZE = 8

# Authenticated official HF API/model manifests, fetched without tensor payloads.
# Each manifest pins every source, tokenizer, backbone shard and FP32 readout.
CHECKPOINTS = {
    "d3": {
        "model_id": "vllm-sr/d3",
        "revision": "225be8b1e2270becab69c5fe05b26f99cbbdd66b",
        "manifest_sha256": "4959df7ee964c2fad24a8e853df8e3cda9bae61636fa5650cc1a0fed78539677",
        "base_model_id": "Qwen/Qwen3.8-27B",
        "base_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        "resident_file_bytes": 52176025120,
        "loaded_parameters": 26086635760,
        "vision_parameters": 460730096,
        "readout_file_bytes": 5222480,
        "readout_tensor_bytes": 5222400,
    },
    "d3-flash": {
        "model_id": "vllm-sr/d3-flash",
        "revision": "8b35c853b90e777595ec36bacae820fe08ea077a",
        "manifest_sha256": "abdafb9f4a3e39595a54b7ca977e6bd7a3279c57745231319e3a171a43976e65",
        "base_model_id": "Qwen/Qwen3.5-9B",
        "base_revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "resident_file_bytes": 16789656976,
        "loaded_parameters": 8393739504,
        "vision_parameters": 456010480,
        "readout_file_bytes": 4178000,
        "readout_tensor_bytes": 4177920,
    },
    "d3-mini": {
        "model_id": "vllm-sr/d3-mini",
        "revision": "97dd054df4b3e0b3beb6af23ab8c82b7c0052e0e",
        "manifest_sha256": "bc638e17982411f6ad0f524d33d0dc1f0db81ad7995c1bfdf93dba7674df5e48",
        "base_model_id": "Qwen/Qwen3.5-4B",
        "base_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "resident_file_bytes": 9081227360,
        "loaded_parameters": 4539918336,
        "vision_parameters": 333514240,
        "readout_file_bytes": 2611280,
        "readout_tensor_bytes": 2611200,
    },
    "d3-nano": {
        "model_id": "vllm-sr/d3-nano",
        "revision": "aeb70d4dd53ed25b855c3b0a66394b8fe265db7f",
        "manifest_sha256": "db2139c5b5b7060f8f95150ec70ca2fe7e69e6bf4249e5766df5388bf35977f2",
        "base_model_id": "Qwen/Qwen3.5-2B",
        "base_revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
        "resident_file_bytes": 4428644200,
        "loaded_parameters": 2213763904,
        "vision_parameters": 331416576,
        "readout_file_bytes": 2089040,
        "readout_tensor_bytes": 2088960,
    },
    "d3-lite": {
        "model_id": "vllm-sr/d3-lite",
        "revision": "9c10703193afc7c7d28c88201c0364bfbbf58320",
        "manifest_sha256": "31f0ff94ff26619c0a063ccac803edcfe34c56c92e5c9c50b703b62893f125dd",
        "base_model_id": "Qwen/Qwen3.5-0.8B",
        "base_revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "resident_file_bytes": 1707072248,
        "loaded_parameters": 853247040,
        "vision_parameters": 100592896,
        "readout_file_bytes": 1044560,
        "readout_tensor_bytes": 1044480,
    },
    "d3-edge": {
        "model_id": "vllm-sr/d3-edge",
        "revision": "cab71e6ad3461b8086d51ddb572c656babd63b0b",
        "manifest_sha256": "3bae8989a6feb36a66bf4d8c44112dc63199d0dc69f564dc895fe368b1e08e59",
        "base_model_id": "Qwen/Qwen3.5-0.8B",
        "base_revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "resident_file_bytes": 1189722528,
        "loaded_parameters": 594582720,
        "vision_parameters": 100592896,
        "readout_file_bytes": 1044560,
        "readout_tensor_bytes": 1044480,
    },
}


def decision3_spec(name, *, device="cuda", cost_per_request_usd=None):
    identity = CHECKPOINTS[name]
    return ModelSpec(
        id=f"{name}-local",
        adapter="decision3",
        enabled=True,
        model_id=identity["model_id"],
        revision=identity["revision"],
        precision="bfloat16",
        device=device,
        execution_mode="sequential",
        calibration="checkpoint",
        context_limit=CONTEXT_LIMIT,
        cost_per_request_usd=cost_per_request_usd,
        capabilities=Capabilities(
            modalities=("text", "image"),
            primitives=("choice", "noul", "score"),
            max_options=MAX_OPTIONS,
            max_questions=64,
            probabilities="complete",
            batch="loop_emulated",
            concurrent_requests=False,
            independent_questions=True,
        ),
        notes="Publisher AutoModel.system_one; native255, BF16/FP32, T=1.0. Project context16384; native hard cap None. No efficacy/peak-memory claim.",
    )


def configuration_blockers(spec):
    identity = next(
        (
            i
            for i in CHECKPOINTS.values()
            if (i["model_id"], i["revision"]) == (spec.model_id, spec.revision)
        ),
        None,
    )
    reasons = []
    if identity is None:
        reasons.append("decision3_checkpoint_unverified")
    if spec.execution_mode != "sequential" or spec.calibration != "checkpoint":
        reasons.append("decision3_policy_mismatch")
    if not (spec.device or "").startswith("cuda") or spec.precision != "bfloat16":
        reasons.append("decision3_requires_cuda_bfloat16")
    if spec.context_limit > CONTEXT_LIMIT:
        reasons.append("decision3_project_context_limit_exceeded")
    if spec.capabilities.probabilities != "complete":
        reasons.append("decision3_requires_complete_probabilities")
    if spec.capabilities.independent_questions is not True:
        reasons.append("decision3_requires_independent_questions")
    if spec.capabilities.max_options is None or spec.capabilities.max_options > MAX_OPTIONS:
        reasons.append("decision3_option_limit_exceeded")
    if spec.subfolder or (spec.processor_revision and spec.processor_revision != spec.revision):
        reasons.append("decision3_processor_override_unavailable")
    return reasons


def request_blockers(request):
    reasons = []
    if any(isinstance(media, AudioInput) for media in request.media):
        reasons.append("decision3_audio_unsupported")
    if any(getattr(media, "timestamp_seconds", None) is not None for media in request.media):
        reasons.append("decision3_timestamped_frames_unsupported")
    for question in request.questions:
        if len(question.labels()) > MAX_OPTIONS:
            reasons.append("max_options_exceeded")
        if question.type == "score":
            descriptions = question.descriptions()
            if not 2 <= len(descriptions) <= 10:
                reasons.append("decision3_score_levels_exceeded")
            if len(set(descriptions)) != len(descriptions) or any(not x for x in descriptions):
                reasons.append("decision3_score_levels_must_be_distinct_nonempty")
    return list(dict.fromkeys(reasons))


def memory_admission(
    identity,
    *,
    helper_already_loaded=False,
    include_helper=False,
    expected_boot_id=None,
    expected_memtotal=None,
):
    """Conservative admission, not a measured loader bound; swap gets no credit.

    Include the fixed Gemma helper before any CLI weight read. Once that helper
    is actually resident, MemAvailable already reflects it, so do not count it
    again. Keep a complete D3 transient copy and OS/display reserve in both gates.
    """
    if type(helper_already_loaded) is not bool or type(include_helper) is not bool:
        raise ValueError("memory helper states must be explicit booleans")
    if helper_already_loaded and not include_helper:
        raise ValueError("loaded helper credit requires a declared helper")
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            pieces = value.split()
            if len(pieces) != 2 or pieces[1] != "kB":
                raise ValueError("unsupported memory snapshot units")
            values[key] = int(pieces[0]) * 1024
    if not 0 < values.get("MemAvailable", 0) <= values.get("MemTotal", 0):
        raise ValueError("invalid live memory snapshot")
    resident = identity["resident_file_bytes"]
    reserve = max(24 * 1024**3, (values["MemTotal"] + 4) // 5)
    additional_helper = FIXED_HELPER_BYTES if include_helper and not helper_already_loaded else 0
    transient = max(resident, FIXED_HELPER_BYTES if include_helper else 0)
    required = resident + transient + additional_helper + reserve
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if not boot_id:
        raise ValueError("missing Decision3 host boot identity")
    if expected_boot_id is not None and boot_id != expected_boot_id:
        raise ValueError("Decision3 host boot identity changed")
    if expected_memtotal is not None and values["MemTotal"] != expected_memtotal:
        raise ValueError("Decision3 host MemTotal changed")
    return {
        "boot_id": boot_id,
        "memory_bytes": values,
        "model_resident_file_bytes": resident,
        "backbone_file_bytes": resident - identity["readout_file_bytes"],
        "backbone_storage_precision": "bfloat16",
        "readout_file_bytes": identity["readout_file_bytes"],
        "readout_tensor_bytes": identity["readout_tensor_bytes"],
        "readout_storage_precision": "float32",
        "largest_declared_transient_bytes": transient,
        "os_display_reserve_bytes": reserve,
        "d3_full_transient_file_bytes": resident,
        "helper_full_transient_file_bytes": FIXED_HELPER_BYTES if include_helper else 0,
        "fixed_helper_file_bytes": FIXED_HELPER_BYTES if include_helper else 0,
        "helper_already_loaded": helper_already_loaded,
        "additional_helper_bytes": additional_helper,
        "swap_credit_bytes": 0,
        "required_available_bytes": required,
        "admitted": values["MemAvailable"] >= required,
        "basis": "pinned publisher file bytes; conservative full transient; not a measured peak",
    }


def require_memory(identity, **kwargs):
    report = memory_admission(identity, **kwargs)
    if not report["admitted"]:
        error = ValueError("Decision3 conservative memory admission refused; no weights read")
        error.decision3_memory = report
        raise error
    return report


def verify_publisher_sources(directory, identity):
    """Authenticate the manifest and every executable Python source before import."""
    root = Path(directory)
    raw = (root / "MODEL_MANIFEST.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != identity["manifest_sha256"]:
        raise ValueError("Decision3 publisher manifest differs from reviewed immutable revision")
    manifest = json.loads(raw)
    if manifest.get("schema") != "d3-package-manifest/1":
        raise ValueError("invalid Decision3 publisher manifest")
    listed = manifest["files_sha256"]
    required = {
        "config.json",
        "decision_config.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    }
    required.update(name for name in listed if name.endswith(".py"))
    if not {"modeling_d3.py", "d3_runtime.py", "d3_format.py", "d3_fast.py"} <= required:
        raise ValueError("incomplete Decision3 publisher source inventory")
    if {p.name for p in root.glob("*.py")} - required:
        raise ValueError("unexpected executable Decision3 package source")
    for name in sorted(required):
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("unsafe Decision3 manifest source path")
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != listed[name]:
            raise ValueError(f"Decision3 publisher source changed: {name}")
    config = json.loads((root / "decision_config.json").read_text())
    if (
        config.get("temperature"),
        config.get("readout_dtype"),
        config.get("max_length"),
        config.get("attention_mode"),
        config.get("pooling"),
        len(config.get("codes", [])),
    ) != (1.0, "float32", None, "noncausal_full_attention", "last", MAX_OPTIONS):
        raise ValueError("Decision3 native numerical policy changed")
    return manifest


def native_payload(request):
    reasons = request_blockers(request)
    if reasons:
        raise RequestValidationError("unsupported Decision3 request: " + ", ".join(reasons))
    questions = {}
    for q in request.questions:
        criteria = (
            q.descriptions()
            if q.type == "score"
            else copy.deepcopy(q.criteria)
            if q.criteria is not None
            else None
        )
        if q.type == "choice" and isinstance(criteria, list):
            criteria = dict(zip(q.labels(), criteria))
        entry = {"type": q.type, "instructions": copy.deepcopy(q.instructions)}
        if criteria is not None:
            entry["criteria"] = criteria
        questions[q.id] = entry
    payload = {"state": copy.deepcopy(request.state), "questions": questions}
    if request.media:
        # Embedded bytes only: never pass URLs/paths that let native decoding fetch data.
        payload["images"] = ["data:image/png;base64," + media.data for media in request.media]
    return payload


def normalize_native_response(request, raw):
    """Validate complete native probabilities; expose project's existing confidence.

    Native entropy confidence remains in the separately retained raw response.
    A native singleton is still evaluated and must return its exact point mass.
    """
    if not isinstance(raw, dict) or not isinstance(raw.get("answers"), dict):
        raise BackendResponseError("Decision3 native response must have named answers")
    if set(raw["answers"]) != {q.id for q in request.questions}:
        raise BackendResponseError("Decision3 native response must answer every requested head")
    usage = raw.get("usage")
    if (
        not isinstance(usage, dict)
        or any(
            type(usage.get(key)) is not int or usage[key] < 0
            for key in ("input_tokens", "output_tokens")
        )
        or usage["output_tokens"] != 0
    ):
        raise BackendResponseError("invalid Decision3 native token usage")
    singletons, ordinary = {}, []
    for q in request.questions:
        answer = raw["answers"][q.id]
        if not isinstance(answer, dict) or answer.get("error"):
            raise BackendResponseError("Decision3 returned a native per-question error")
        if "confidence" in answer:
            confidence = answer["confidence"]
            if (
                type(confidence) not in (int, float)
                or not math.isfinite(confidence)
                or not 0 <= confidence <= 1
            ):
                raise BackendResponseError("invalid Decision3 native entropy confidence")
        if q.type != "noul":
            labels = [str(i) for i in range(len(q.labels()))] if q.type == "score" else q.labels()
            probabilities = answer.get("probabilities")
            if not isinstance(probabilities, dict) or set(probabilities) != set(labels):
                raise BackendResponseError(
                    "Decision3 probability labels differ from native contract"
                )
            try:
                values = [probabilities[label] for label in labels]
                if any(type(value) not in (int, float) for value in values):
                    raise ValueError("probabilities must be numbers")
                validate_distribution(values, len(labels))
            except (ValueError, TypeError, OverflowError) as exc:
                raise BackendResponseError(
                    "invalid Decision3 complete native probabilities"
                ) from exc
        if len(q.labels()) == 1:
            label = q.labels()[0]
            if (
                answer.get("type", q.type) != "choice"
                or answer.get("choice") != label
                or values != [1.0]
            ):
                raise BackendResponseError("invalid Decision3 native singleton answer")
            singletons[q.id] = {
                "id": q.id,
                "type": "choice",
                "choice": label,
                "probabilities": {label: 1.0},
                "confidence": 1.0,
                "margin": 1.0,
            }
        else:
            ordinary.append(q)
    if ordinary:
        # Normal validation has already run; this projection changes no candidate.
        from types import SimpleNamespace

        projection = SimpleNamespace(questions=ordinary)
        projected_raw = {**raw, "answers": {q.id: raw["answers"][q.id] for q in ordinary}}
        normalized = normalize_response(projection, projected_raw, ordinal_scores=True)["answers"]
    else:
        normalized = {}
    return {
        **raw,
        "answers": {q.id: singletons.get(q.id, normalized.get(q.id)) for q in request.questions},
    }


class Decision3Backend:
    def __init__(
        self, spec, *, helper_already_loaded=False, expected_boot_id=None, expected_memtotal=None
    ):
        reasons = configuration_blockers(spec)
        if reasons:
            raise ValueError("invalid Decision3 configuration: " + ", ".join(reasons))
        self.spec, self.name = spec, spec.id
        self.identity = next(i for i in CHECKPOINTS.values() if i["model_id"] == spec.model_id)
        self.memory_preflight = require_memory(
            self.identity,
            include_helper=helper_already_loaded,
            helper_already_loaded=helper_already_loaded,
            expected_boot_id=expected_boot_id,
            expected_memtotal=expected_memtotal,
        )
        if metadata.version("transformers") != "5.17.0":
            raise ValueError("Decision3 requires publisher-pinned transformers==5.17.0")
        from huggingface_hub import snapshot_download

        # An absent cache is a refusal, never a hidden weight/network download.
        directory = snapshot_download(spec.model_id, revision=spec.revision, local_files_only=True)
        self.manifest = verify_publisher_sources(directory, self.identity)
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(
            directory,
            trust_remote_code=True,
            local_files_only=True,
            device_map={"": spec.device},
            dtype="bfloat16",
            batch_size=NATIVE_BATCH_SIZE,
            verify="full",
        )
        self.runtime = self.model.runtime
        self.runtime_precision = str(next(self.runtime.backbone.parameters()).dtype).removeprefix(
            "torch."
        )
        if (
            self.runtime_precision != "bfloat16"
            or self.runtime.readout_dtype != "float32"
            or self.runtime.temperature != 1.0
            or self.runtime.max_length is not None
            or self.runtime.attention_mode != "noncausal_full_attention"
            or self.runtime.permutation_average
            or self.runtime.batch_size != NATIVE_BATCH_SIZE
        ):
            raise ValueError(
                "loaded Decision3 runtime differs from the pinned native numerical policy"
            )

    def predict(self, request, *, deadline=None, evidence_sink=None):
        evidence = {
            "native_payload": native_payload(request),
            "preflight": None,
            "raw_response": None,
        }
        try:
            if deadline is not None:
                deadline()
            prepared = self.runtime.prepare(**copy.deepcopy(evidence["native_payload"]))
            lengths = {key: len(value) for key, value in prepared.sequences.items()}
            lengths.update(prepared.lengths)
            evidence["preflight"] = {
                "lengths": lengths,
                "errors": copy.deepcopy(prepared.errors),
                "project_context_limit": self.spec.context_limit,
            }
            if prepared.errors or set(lengths) != {q.id for q in request.questions}:
                raise RequestValidationError(
                    "Decision3 native preparation rejected the full request"
                )
            if any(
                type(n) is not int or n < 1 or n > self.spec.context_limit for n in lengths.values()
            ):
                raise RequestValidationError(
                    "Decision3 project context cap exceeded; nothing truncated"
                )
            if evidence_sink is not None:
                evidence_sink("native_preflight", copy.deepcopy(evidence))
            if deadline is not None:
                deadline()
            raw = self.model.system_one(**copy.deepcopy(evidence["native_payload"]))
            evidence["raw_response"] = copy.deepcopy(raw)
            if evidence_sink is not None:
                evidence_sink("native_raw_returned", copy.deepcopy(evidence))
            if deadline is not None:
                deadline()
            if (
                not isinstance(raw, dict)
                or raw.get("model") != self.identity["model_id"].rsplit("/", 1)[-1]
            ):
                raise BackendResponseError("Decision3 returned a different native model identity")
            result = normalize_native_response(request, raw)
            result["decision3_provider_evidence"] = evidence
            return result
        except BaseException as exc:
            exc.decision3_provider_evidence = evidence
            raise

    def telemetry(self):
        return {
            "revision": self.spec.revision,
            "device": self.spec.device,
            "precision": self.runtime_precision,
            "compute_precision": "bfloat16_backbone_float32_readout",
            "temperature": self.runtime.temperature,
            "context_limit": self.spec.context_limit,
            "context_policy": "reject",
            "runtime_package": "vllm-sr/Decision3 AutoModel.system_one",
            "source_revision": self.spec.revision,
            "base_model_id": self.identity["base_model_id"],
            "base_revision": self.identity["base_revision"],
            "checkpoint_task": "multimodal_typed_decisions",
            "compute_backend": "publisher_native_sdpa",
            "runtime_options": {
                "publisher_manifest_sha256": self.identity["manifest_sha256"],
                "native_candidate_cap": MAX_OPTIONS,
                "native_hard_context_cap": None,
                "architectural_context_metadata": ARCHITECTURAL_CONTEXT,
                "project_context_cap": self.spec.context_limit,
                "head_storage_precision": "float32",
                "native_question_batch_size": NATIVE_BATCH_SIZE,
                "permutation_average": False,
                "all_requested_heads_evaluated": True,
                "singleton_policy": "native_forward",
                "probability_postprocessing": "complete native distribution; project confidence=p(max), native entropy confidence retained in raw evidence",
                "context_preflight": "publisher prepare before public system_one; tokenization repeats, no silent truncation",
                "package_verification": "reviewed manifest/source pins before import, publisher full verification",
                "publisher_runtime_info": self.runtime.runtime_info(),
                "publisher_provenance": self.runtime.provenance(),
                "memory_admission": self.memory_preflight,
                "efficacy_numerical_equivalence_and_peak_memory_verified": False,
            },
        }

    def resources(self):
        from s1.resources import memory_snapshot

        # The readout is a separate tensor, not registered in backbone.parameters.
        result = memory_snapshot(self.runtime.backbone, device=self.spec.device)
        result["native_readout_bytes"] = (
            self.runtime.readout.numel() * self.runtime.readout.element_size()
        )
        return result

    def close(self):
        self.model = self.runtime = None
