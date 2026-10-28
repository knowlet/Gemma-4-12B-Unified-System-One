"""Held single-call synthetic Decision3 CUDA worker; the supervisor owns hard limits.

No collection, benchmark, browser, service, or input file is loaded here. The
caller must authenticate the selected source/runtime/model and own the process,
GPU, memory, and cleanup independently. The deadline is cooperative, not a
hard cancellation. This module alone is not a launcher or a provenance proof.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import math
import re
import time

SOURCE_ONLY = True
SCHEMA = "decision3-cuda-synthetic-worker-v1"
CALL_SECONDS = 20.0
MODEL_ID = "vllm-sr/d3-edge"
REVISION = "cab71e6ad3461b8086d51ddb572c656babd63b0b"
MANIFEST_SHA = "3bae8989a6feb36a66bf4d8c44112dc63199d0dc69f564dc895fe368b1e08e59"
READOUT_SHA = "d408d94a0ab2915fa85c511dd9b8d62af49903c4e4ca28e2e2d5aff9ef565e0f"


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _input():
    # Invented locally: no input path or caller-selected examples are accepted.
    return {
        "state": {"setting": "fictional parcel sorting demonstration"},
        "questions": {
            "destination": {
                "type": "choice",
                "instructions": "Choose a destination for the fictional parcel.",
                "criteria": {"north": "northern shelf", "south": "southern shelf"},
            },
            "fragile": {"type": "noul", "instructions": "Is this fictional parcel fragile?"},
        },
    }


def _probability(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def _response(value, payload):
    _need(type(value) is dict and value.get("model") == "d3-edge", "native model response required")
    evidence = value.get("decision3_provider_evidence")
    _need(
        type(evidence) is dict
        and evidence.get("native_payload") == payload
        and type(evidence.get("preflight")) is dict,
        "matching native preparation required",
    )
    preflight = evidence["preflight"]
    lengths = preflight.get("lengths")
    _need(
        type(lengths) is dict
        and list(lengths) == ["destination", "fragile"]
        and all(type(n) is int and 1 <= n <= 16384 for n in lengths.values())
        and not preflight.get("errors"),
        "complete bounded native head lengths required",
    )
    raw = evidence.get("raw_response")
    _need(
        type(raw) is dict
        and raw.get("model") == "d3-edge"
        and type(raw.get("answers")) is dict
        and list(raw["answers"]) == ["destination", "fragile"]
        and type(raw.get("usage")) is dict
        and type(raw["usage"].get("input_tokens")) is int
        and raw["usage"]["input_tokens"] == sum(lengths.values())
        and type(raw["usage"].get("output_tokens")) is int
        and raw["usage"]["output_tokens"] == 0
        and value.get("usage") == raw["usage"],
        "complete native response and usage required",
    )
    answers = value.get("answers")
    _need(type(answers) is dict and list(answers) == list(raw["answers"]), "all heads required")
    choice, truth = raw["answers"]["destination"], raw["answers"]["fragile"]
    _need(
        type(choice) is dict
        and set(choice) == {"type", "choice", "probabilities", "confidence"}
        and choice["type"] == "choice"
        and type(choice["probabilities"]) is dict
        and list(choice["probabilities"]) == ["north", "south"],
        "complete original choice distribution required",
    )
    probabilities = list(choice["probabilities"].values())
    _need(
        all(_probability(p) for p in probabilities)
        and math.isclose(sum(probabilities), 1.0, rel_tol=0, abs_tol=1e-6)
        and choice["choice"] == ["north", "south"][probabilities.index(max(probabilities))]
        and _probability(choice["confidence"]),
        "valid original choice required",
    )
    entropy_confidence = 1 + sum(p * math.log(p) for p in probabilities if p) / math.log(2)
    _need(
        math.isclose(choice["confidence"], entropy_confidence, rel_tol=1e-12, abs_tol=1e-12),
        "original entropy confidence differs",
    )
    _need(
        type(truth) is dict
        and set(truth) == {"type", "noul"}
        and truth["type"] == "noul"
        and _probability(truth["noul"]),
        "complete original noul required",
    )
    named_choice, named_truth = answers["destination"], answers["fragile"]
    total = sum(probabilities)
    expected_choice = [p / total for p in probabilities]
    expected_truth = [1 - truth["noul"], truth["noul"]]
    for answer, labels, expected, kind in (
        (named_choice, ["north", "south"], expected_choice, "choice"),
        (named_truth, ["false", "true"], expected_truth, "noul"),
    ):
        _need(
            type(answer) is dict
            and answer.get("type") == kind
            and type(answer.get("probabilities")) is dict
            and list(answer["probabilities"]) == labels
            and all(_probability(p) for p in answer["probabilities"].values())
            and all(
                math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                for a, b in zip(answer["probabilities"].values(), expected, strict=True)
            )
            and _probability(answer.get("confidence"))
            and math.isclose(answer["confidence"], max(expected), rel_tol=1e-12, abs_tol=1e-12)
            and _probability(answer.get("margin"))
            and math.isclose(
                answer["margin"], abs(expected[0] - expected[1]), rel_tol=1e-12, abs_tol=1e-12
            ),
            "normalized complete probabilities, confidence and margin required",
        )
    _need(
        named_choice.get("id") == "destination"
        and named_choice.get("choice") == choice["choice"]
        and named_truth.get("id") == "fragile"
        and named_truth.get("noul") == truth["noul"],
        "normalized choices must retain native answers",
    )
    return {
        "raw_response": copy.deepcopy(raw),
        "normalized_answers": copy.deepcopy(answers),
        "head_lengths": copy.deepcopy(lengths),
        "usage": copy.deepcopy(raw["usage"]),
    }


def _runtime(backend, torch):
    spec, identity, runtime = backend.spec, backend.identity, backend.runtime
    _need(
        spec.model_id == MODEL_ID
        and spec.revision == REVISION
        and spec.device == "cuda:0"
        and spec.precision == "bfloat16"
        and identity.get("manifest_sha256") == MANIFEST_SHA
        and backend.model.runtime is runtime
        and runtime.model_name == "d3-edge"
        and runtime.batch_size == 8
        and runtime.temperature == 1.0
        and runtime.max_length is None
        and runtime.attention_mode == "noncausal_full_attention"
        and runtime.permutation_average is False
        and runtime.fast is None
        and runtime.readout_dtype == "float32"
        and runtime.backbone.training is False,
        "fixed Decision3 CUDA runtime required",
    )
    params = list(runtime.backbone.parameters())
    _need(
        params
        and all(
            str(p.device) == "cuda:0" and p.dtype == torch.bfloat16 and p.requires_grad is False
            for p in params
        )
        and torch.is_tensor(runtime.readout)
        and str(runtime.readout.device) == "cuda:0"
        and runtime.readout.dtype == torch.float32
        and tuple(runtime.readout.shape) == (255, 1024)
        and runtime.readout.requires_grad is False
        and sum(p.numel() for p in params) + runtime.readout.numel() == 594582720,
        "fixed BF16 backbone and FP32 readout required",
    )
    raw = runtime.readout.detach().to(device="cpu").contiguous().numpy().tobytes()
    _need(
        hashlib.sha256(raw).hexdigest() == READOUT_SHA
        and torch.get_float32_matmul_precision() == "highest"
        and torch.backends.cuda.matmul.allow_tf32 is False
        and not torch.is_autocast_enabled("cuda"),
        "fixed original readout and CUDA numerical policy required",
    )


def run(*, torch, gpu_uuid, source_guard, ownership_guard, clock=time.monotonic):
    """Execute once after review; supervisor must authenticate guards and enforce hard wall.

    `source_guard` and `ownership_guard` are supplied and authenticated by the
    independently supervised owner, not proven by this function. All evidence is
    caller-owned and remains pending external audit; no benchmark score is made.
    """
    _need(SOURCE_ONLY is False, "reviewed synthetic CUDA worker release required")
    _need(
        callable(source_guard) and callable(ownership_guard) and callable(clock),
        "owner guards required",
    )
    _need(
        type(gpu_uuid) is str
        and re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", gpu_uuid),
        "explicit GPU UUID required",
    )
    backend = None
    error = None
    cuda_started = False
    result = {
        "schema": SCHEMA,
        "status": "failed_no_score",
        "device": "cuda:0",
        "deadline_scope": "cooperative 20 seconds from DTO validation through synchronized response and guards; excludes construction and cleanup; external hard limit required",
        "score": None,
    }
    try:
        source_guard()
        ownership_guard()
        from s1.contracts import DecisionRequest
        from s1.evaluation.decision3 import Decision3Backend, decision3_spec, native_payload

        _need(
            torch.cuda.is_available() is True
            and torch.cuda.device_count() == 1
            and torch.cuda.current_device() == 0,
            "one selected CUDA device required",
        )
        properties = torch.cuda.get_device_properties(0)
        _need(
            str(properties.uuid).lower().removeprefix("gpu-")
            == gpu_uuid.lower().removeprefix("gpu-")
            and (properties.major, properties.minor) == (12, 1)
            and torch.cuda.is_bf16_supported(including_emulation=False) is True,
            "selected GB10 BF16 GPU required",
        )
        cuda_started = True
        torch.cuda.synchronize(0)
        result["initial_allocated_bytes"] = torch.cuda.memory_allocated(0)
        _need(result["initial_allocated_bytes"] == 0, "fresh CUDA allocator required")
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        backend = Decision3Backend(decision3_spec("d3-edge", device="cuda:0"))
        _runtime(backend, torch)
        source_guard()
        ownership_guard()
        started = clock()

        def deadline():
            _need(clock() - started < CALL_SECONDS, "cooperative request deadline exceeded")
            ownership_guard()

        deadline()
        request = DecisionRequest.model_validate(_input())
        payload = native_payload(request)
        response = backend.predict(request, deadline=deadline)
        torch.cuda.synchronize(0)
        deadline()
        evidence = _response(response, payload)
        _runtime(backend, torch)
        source_guard()
        deadline()
        elapsed = clock() - started
        _need(0 <= elapsed < CALL_SECONDS, "cooperative request deadline exceeded")
        result.update(
            evidence=evidence,
            elapsed_seconds=elapsed,
            model_id=MODEL_ID,
            revision=REVISION,
            publisher_manifest_sha256=MANIFEST_SHA,
            calls=1,
            status="completed_pending_cleanup",
        )
    except BaseException as exc:
        error = exc
    finally:
        try:
            if backend is not None:
                backend.close()
            backend = None
            gc.collect()
            if cuda_started:
                torch.cuda.synchronize(0)
                clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
                if callable(clear):
                    clear()
                torch.cuda.empty_cache()
                torch.cuda.synchronize(0)
                result["final_allocated_bytes"] = torch.cuda.memory_allocated(0)
                result["final_reserved_bytes"] = torch.cuda.memory_reserved(0)
                _need(result["final_allocated_bytes"] == 0, "CUDA allocator cleanup unconfirmed")
                result["cleanup"] = "current_process_allocator_zero_pending_external_owner_audit"
            else:
                result["cleanup"] = "CUDA synchronization was not started by worker"
        except BaseException as cleanup_error:
            result["cleanup"] = "unconfirmed"
            result["cleanup_error_type"] = type(cleanup_error).__name__
            if error is None:
                error = cleanup_error
            else:
                error.add_note("CUDA cleanup also failed: " + type(cleanup_error).__name__)
    if error is not None:
        result["status"] = "failed_no_score"
        result["error_type"] = type(error).__name__
        error.cuda_synthetic_evidence = result
        raise error
    result["status"] = "completed_pending_external_audit"
    return result
