"""HELD operation-only GPU composition around one genuine native provider call.

This module loads no files/framework/model and owns no HTTP/browser service.
The caller authenticates capability bytes, data artifacts, source/runtime
origins, resource ownership and later numerical/browser acceptance separately.
Only a normal native DTO reaches the supplied genuine CUDA provider.  Every
original head is evaluated once; only the actual operation row receives delta.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
import time

SOURCE_ONLY = True
PROFILE = "decision3-affordance-operation-readout-gpu-candidate-v1"
ALIAS = "d3-edge-dsl01-affordance-operation-delta-gpu-candidate"
CAPABILITY_SCHEMA = "root-selected-affordance-operation-gpu-composition-v1"
CAPABILITY_STATUS = "ROOT_SELECTED_OPERATION_GPU_COMPOSITION_ONLY"
PURPOSE = "readout_composition_only"
CRITERIA_MAPPING = "actual_ordered_criteria_ordinal"
HEAD_MAPPING = "actual_question_id_and_row"
CONTEXT_LIMIT, BATCH_SIZE, MAX_ROWS, CODES, WIDTH, CALL_SECONDS = 16384, 8, 4, 255, 1024, 20
ABS_TOL, REL_TOL = 1e-6, 1e-5
MODEL_IDENTITY = {
    "name": "d3-edge",
    "model_id": "vllm-sr/d3-edge",
    "revision": "cab71e6ad3461b8086d51ddb572c656babd63b0b",
    "manifest_sha256": "3bae8989a6feb36a66bf4d8c44112dc63199d0dc69f564dc895fe368b1e08e59",
    "model_sha256": "6e00d04abbdbdf6d35c9e12065769d9edaacc3bfe9404583891797718d239dc8",
    "resident_file_bytes": 1189722528,
    "loaded_parameters": 594582720,
    "readout_shape": [255, 1024],
}
POPULATION = {
    "rows": 109,
    "old_rows": 96,
    "new_rows": 13,
    "K_counts": {"4": 4, "7": 5, "8": 100},
}
CAPABILITY_KEYS = {
    "schema",
    "status",
    "profile",
    "alias",
    "owner_profile",
    "purpose",
    "only_head",
    "model",
    "population",
    "baseline_fp32_bytes_sha256",
    "delta_fp32_bytes_sha256",
    "delta_ref",
    "fit_integrity_ref",
    "fit_capability_ref",
    "gpu_native_prerequisite_ref",
    "criteria_mapping",
    "head_mapping",
    "dev_used_for_model_selection",
    "optimizer_input_dev",
    "optimizer_input_sealed",
    "browser_authorized",
    "benchmark_authorized",
    "training_authorized",
}
REF_KEYS = {"delta_ref", "fit_integrity_ref", "fit_capability_ref", "gpu_native_prerequisite_ref"}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def release_gate():
    require(SOURCE_ONLY is False, "SOURCE_ONLY: capability/dependency/runtime/delta access refused")


def _sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _ref(value):
    return (
        type(value) is dict
        and set(value) == {"path", "sha256", "size_bytes"}
        and type(value["path"]) is str
        and value["path"].startswith("/")
        and "\x00" not in value["path"]
        and ".." not in value["path"].split("/")
        and _sha(value["sha256"])
        and type(value["size_bytes"]) is int
        and value["size_bytes"] > 0
    )


def _json(value):
    # Importing stdlib JSON here is harmless; no tensor or framework is loaded.
    import json

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def validate_capability(capability, owner_profile):
    """Validate a caller-authenticated new composition-only contract.

    The supplied SHA is an external owner-plan binding, not a compiled constant
    and not a claim that this module authenticates file/producer/closure bytes.
    The109 integrity reference is data provenance, not old96 fit authority.
    """
    release_gate()
    require(
        type(capability) is dict and set(capability) == CAPABILITY_KEYS,
        "exact new109 GPU composition capability required",
    )
    wanted = {
        "schema": CAPABILITY_SCHEMA,
        "status": CAPABILITY_STATUS,
        "profile": PROFILE,
        "alias": ALIAS,
        "purpose": PURPOSE,
        "only_head": "operation",
        "model": MODEL_IDENTITY,
        "population": POPULATION,
        "criteria_mapping": CRITERIA_MAPPING,
        "head_mapping": HEAD_MAPPING,
    }
    require(
        all(capability[key] == value for key, value in wanted.items()),
        "wrong model/population/mapping/composition contract",
    )
    require(
        type(owner_profile) is str
        and owner_profile
        and capability["owner_profile"] == owner_profile,
        "caller-selected owner profile differs",
    )
    require(
        capability["dev_used_for_model_selection"] is True
        and capability["optimizer_input_dev"] is False
        and capability["optimizer_input_sealed"] is False
        and capability["browser_authorized"] is False
        and capability["benchmark_authorized"] is False
        and capability["training_authorized"] is False,
        "data provenance cannot grant browser/benchmark/training authorization",
    )
    require(
        all(_ref(capability[key]) for key in REF_KEYS),
        "complete separately pinned data/prerequisite references required",
    )
    require(
        _sha(capability["baseline_fp32_bytes_sha256"])
        and _sha(capability["delta_fp32_bytes_sha256"]),
        "raw FP32 body hashes required independently of NPY file hashes",
    )
    _json(capability)


def _tensor_identity(tensor):
    try:
        version = tensor._version
    except RuntimeError:
        version = None  # Captured inference tensors may have no version counter.
    return (
        id(tensor),
        tuple(tensor.shape),
        str(tensor.dtype),
        str(tensor.device),
        tensor.data_ptr(),
        version,
    )


def _baseline_identity(runtime):
    return {
        "readout": _tensor_identity(runtime.readout),
        "backbone": tuple(_tensor_identity(p) for p in runtime.backbone.parameters()),
        "encoder": id(runtime.encoder),
        "norm": id(runtime.backbone.language_model.norm),
    }


def _policy(torch):
    return {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_autocast_enabled": torch.is_autocast_enabled("cuda"),
    }


def _bytes_equal(torch, left, right):
    return bool(
        torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))
    )


class GPUObserver:
    """Read-only native partial-batch hooks; no retained full-sequence tensor."""

    def __init__(self, torch, rows, padded_width):
        release_gate()
        require(
            type(rows) is int
            and 1 <= rows <= MAX_ROWS
            and type(padded_width) is int
            and padded_width > 0,
            "actual1..4 GPU rows and positive width required",
        )
        self.torch, self.rows, self.padded_width = torch, rows, padded_width
        self.first_layer_events, self.norm_events, self.features = [], [], None

    def _check(self, tensor):
        require(
            tuple(tensor.shape) == (self.rows, self.padded_width, WIDTH)
            and str(tensor.device) == "cuda:0"
            and tensor.dtype == self.torch.bfloat16,
            "actual complete CUDA0 BF16 native partial-batch shape differs",
        )

    def first_layer(self, _module, args):
        require(
            args and not self.first_layer_events, "exactly one original first-layer pass required"
        )
        self._check(args[0])
        self.first_layer_events.append(
            {
                "shape": list(args[0].shape),
                "dtype": str(args[0].dtype),
                "device": str(args[0].device),
                "observed_monotonic": time.monotonic(),
                "modified": False,
            }
        )
        return None

    def __call__(self, _module, _args, output):
        require(not self.norm_events, "exactly one original final-norm pass required")
        self._check(output)
        self.features = output[:, -1, :].detach().to(dtype=self.torch.float32).clone()
        require(
            tuple(self.features.shape) == (self.rows, WIDTH)
            and str(self.features.device) == "cuda:0"
            and bool(self.torch.isfinite(self.features).all().item()),
            "finite complete same-call CUDA FP32 features required",
        )
        self.norm_events.append(
            {
                "shape": list(output.shape),
                "dtype": str(output.dtype),
                "device": str(output.device),
                "observed_monotonic": time.monotonic(),
                "modified": False,
            }
        )
        return None

    def require_single_pass(self):
        require(
            len(self.first_layer_events) == len(self.norm_events) == 1
            and self.features is not None,
            "one complete original forward/final-norm witness required",
        )


def project(torch, base_logits, features, delta, counts, ordered_head_ids):
    """Native full255 mask/softmax; separate delta only at actual operation row.

    The target byte assertion concerns the provided replay-logit operands.
    Original native target answers are preserved separately by raw copying.
    """
    release_gate()
    rows = len(counts)
    require(
        type(counts) is list
        and 1 <= rows <= MAX_ROWS
        and type(ordered_head_ids) is list
        and len(ordered_head_ids) == rows
        and all(type(key) is str and key for key in ordered_head_ids)
        and len(set(ordered_head_ids)) == rows
        and all(type(n) is int and 1 <= n <= CODES for n in counts)
        and tuple(base_logits.shape) == (rows, CODES)
        and tuple(features.shape) == (rows, WIDTH)
        and tuple(delta.shape) == (CODES, WIDTH)
        and all(
            t.dtype == torch.float32 and str(t.device) == "cuda:0"
            for t in (base_logits, features, delta)
        ),
        "complete original CUDA FP32 logits/features/delta/head/count geometry required",
    )
    with torch.inference_mode():
        snapshots = [tensor.detach().clone() for tensor in (base_logits, features, delta)]
        combined = base_logits.clone()
        limit = torch.as_tensor(list(counts), device="cuda:0")[:, None]
        invalid = torch.arange(CODES, device="cuda:0")[None] >= limit
        require(
            bool(torch.isfinite(base_logits.masked_fill(invalid, 0)).all().item()),
            "finite legal original logits required",
        )
        require(
            bool(torch.isfinite(features).all().item())
            and bool(torch.isfinite(delta).all().item()),
            "finite same-call feature/delta values required",
        )
        operation_rows = [index for index, key in enumerate(ordered_head_ids) if key == "operation"]
        for index in operation_rows:
            added = features[index : index + 1] @ delta.T
            require(
                bool(torch.isfinite(added).all().item()), "nonfinite operation delta projection"
            )
            combined[index : index + 1] = base_logits[index : index + 1] + added
        combined = combined.masked_fill(invalid, float("-inf"))
        preserved = [index for index, key in enumerate(ordered_head_ids) if key != "operation"]
        require(
            all(
                _bytes_equal(torch, combined[index : index + 1], base_logits[index : index + 1])
                for index in preserved
            ),
            "non-operation replay-logit bytes changed",
        )
        require(
            bool(torch.isfinite(combined.masked_fill(invalid, 0)).all().item()),
            "finite legal adapted logits required",
        )
        values = torch.softmax(combined / 1.0, dim=-1).cpu().tolist()
        require(
            all(
                _bytes_equal(torch, tensor, snapshot)
                for tensor, snapshot in zip((base_logits, features, delta), snapshots, strict=True)
            ),
            "projection mutated an original operand",
        )
    require(
        len(values) == rows
        and all(
            len(row) == CODES
            and all(type(v) in (float, int) and math.isfinite(v) and 0 <= v <= 1 for v in row)
            and abs(sum(row) - 1) <= ABS_TOL
            and all(value == 0 for value in row[count:])
            for row, count in zip(values, counts, strict=True)
        ),
        "complete finite normalized adapted distributions and zero illegal mass required",
    )
    return [row[:count] for row, count in zip(values, counts, strict=True)], {
        "ordered_head_ids": list(ordered_head_ids),
        "legal_counts": list(counts),
        "operation_rows": operation_rows,
        "adapted_head_ids": [ordered_head_ids[index] for index in operation_rows],
        "preserved_head_ids": [ordered_head_ids[index] for index in preserved],
        "target_replay_logits_bitwise_unchanged": True,
        "projection_operands_bitwise_unchanged": True,
        "W0_merged_or_mutated": False,
        "sum_order": "h.float()@W0.T then actual-operation-row h.float()@delta.T",
        "temperature": 1.0,
    }


def _validate_raw(raw, questions, lengths):
    require(
        type(raw) is dict and raw.get("model") == MODEL_IDENTITY["name"],
        "original internal model identity required",
    )
    require(
        type(raw.get("answers")) is dict and list(raw["answers"]) == list(questions),
        "complete original raw question order required",
    )
    usage = raw.get("usage")
    require(
        type(usage) is dict
        and set(usage) == {"input_tokens", "output_tokens"}
        and all(type(v) is int for v in usage.values())
        and usage == {"input_tokens": sum(lengths.values()), "output_tokens": 0},
        "complete original native usage required",
    )
    _json(raw)
    for key, question in questions.items():
        labels, answer = list(question["criteria"]), raw["answers"][key]
        require(
            type(answer) is dict and answer.get("type") == "choice" and not answer.get("error"),
            "valid native choice answer required",
        )
        probabilities = answer.get("probabilities")
        require(
            type(probabilities) is dict and list(probabilities) == labels,
            "every original criterion in original order required",
        )
        values = list(probabilities.values())
        require(
            all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in values)
            and abs(sum(values) - 1) <= ABS_TOL,
            "finite full native choice distribution required",
        )
        require(
            answer.get("choice") == max(probabilities, key=probabilities.__getitem__),
            "native ordered argmax required",
        )
        entropy = -sum(v * math.log(v) for v in values if v > 0)
        expected = (
            1.0 if len(values) == 1 else max(0.0, min(1.0, 1 - entropy / math.log(len(values))))
        )
        confidence = answer.get("confidence")
        require(
            type(confidence) in (int, float)
            and math.isfinite(confidence)
            and 0 <= confidence <= 1
            and abs(confidence - expected) <= ABS_TOL,
            "native raw entropy confidence required",
        )
        if len(values) == 1:
            require(
                values == [1.0] and confidence == 1.0,
                "native singleton point mass/confidence1 required",
            )


def _compare_baseline(raw, questions, values):
    require(len(values) == len(questions), "complete same-call baseline replay required")
    maximum = 0.0
    for (key, question), predicted in zip(questions.items(), values, strict=True):
        labels = list(question["criteria"])
        actual = raw["answers"][key]["probabilities"]
        require(
            len(predicted) == len(labels) and abs(sum(predicted) - 1) <= ABS_TOL,
            "complete normalized baseline replay required",
        )
        for label, probability in zip(labels, predicted, strict=True):
            error = abs(probability - actual[label])
            require(
                error <= ABS_TOL + REL_TOL * abs(actual[label]),
                "same-call baseline replay exceeds declared mixed tolerance",
            )
            maximum = max(maximum, error)
        require(
            raw["answers"][key]["choice"]
            == labels[max(range(len(predicted)), key=predicted.__getitem__)],
            "same-call baseline replay ordered argmax differs",
        )
    return {
        "heads": len(questions),
        "legal_outputs": sum(len(question["criteria"]) for question in questions.values()),
        "all_legal_probabilities_matched": True,
        "all_ordered_argmaxes_matched": True,
        "max_absolute_error": maximum,
        "absolute_tolerance": ABS_TOL,
        "relative_tolerance": REL_TOL,
    }


def _note(primary, other, stage):
    # Structured secondary evidence is already retained by the caller.
    # An exception subclass cannot replace the primary by refusing a note.
    try:
        primary.add_note(stage + ": " + type(other).__name__ + ": " + str(other))
    except BaseException:
        pass


class OperationDeltaGPUBackend:
    """Own only an independent delta; shared genuine base remains caller-owned."""

    name = ALIAS

    def __init__(
        self,
        base_provider,
        cpu_delta,
        torch,
        *,
        capability,
        capability_sha256,
        owner_profile,
        request_type,
        normalize_native_response,
        identity_guard,
        transfer_witness,
        witness_sink,
        evidence_sink=None,
        clock=time.monotonic,
    ):
        release_gate()
        validate_capability(capability, owner_profile)
        require(_sha(capability_sha256), "externally owner-pinned capability SHA required")
        require(
            all(
                callable(value)
                for value in (
                    request_type.model_validate,
                    normalize_native_response,
                    identity_guard,
                    transfer_witness,
                    witness_sink,
                    clock,
                )
            ),
            "required normal DTO/normalizer/guard/transfer/private-witness callbacks required",
        )
        require(
            evidence_sink is None or callable(evidence_sink),
            "optional serializable sink must be callable",
        )
        self.base, self.runtime, self.spec = (
            base_provider,
            base_provider.runtime,
            base_provider.spec,
        )
        self.torch, self.request_type = torch, request_type
        self.normalize, self.identity_guard = normalize_native_response, identity_guard
        self.witness_sink, self.evidence_sink, self.clock = witness_sink, evidence_sink, clock
        self.capability, self.capability_sha256 = copy.deepcopy(capability), capability_sha256
        self.closed, self._in_call, self.last_evidence = False, False, None
        self._validate_runtime()
        require(
            torch.is_tensor(cpu_delta)
            and tuple(cpu_delta.shape) == (CODES, WIDTH)
            and cpu_delta.dtype == torch.float32
            and str(cpu_delta.device) == "cpu"
            and cpu_delta.requires_grad is False
            and cpu_delta.is_contiguous()
            and bool(torch.isfinite(cpu_delta).all().item()),
            "authenticated finite frozen contiguous CPU FP32 delta required",
        )
        source_identity = _tensor_identity(cpu_delta)
        self.delta = None
        try:
            self.delta = (
                cpu_delta.detach().to(device="cuda:0", dtype=torch.float32).contiguous().clone()
            )
            require(
                str(self.delta.device) == "cuda:0"
                and self.delta.dtype == torch.float32
                and self.delta.requires_grad is False
                and self.delta.data_ptr() != cpu_delta.data_ptr(),
                "owned independent frozen CUDA FP32 delta copy required",
            )
            self.transfer = transfer_witness(cpu_delta, self.delta)
            expected_transfer = {
                "bytewise_equal": True,
                "source_device": "cpu",
                "destination_device": "cuda:0",
                "source_fp32_bytes_sha256": capability["delta_fp32_bytes_sha256"],
                "destination_fp32_bytes_sha256": capability["delta_fp32_bytes_sha256"],
            }
            require(
                type(self.transfer) is dict and self.transfer == expected_transfer,
                "caller-authenticated raw FP32 CPU-to-GPU transfer differs",
            )
            require(
                _tensor_identity(cpu_delta) == source_identity,
                "original CPU delta metadata changed during transfer",
            )
            self._baseline = _baseline_identity(self.runtime)
            self._delta_identity = _tensor_identity(self.delta)
            self._precision_policy = _policy(torch)
            require(
                self._precision_policy["float32_matmul_precision"] == "highest"
                and self._precision_policy["cuda_matmul_allow_tf32"] is False
                and self._precision_policy["cuda_autocast_enabled"] is False,
                "selected FP32 readout requires highest matmul precision without TF32/autocast",
            )
            self.initial_guard = self._guard()
        except BaseException:
            self.delta = self.runtime = self.base = None
            self.closed = True
            raise

    def _validate_runtime(self):
        require(
            getattr(self.spec, "adapter", None) == "decision3"
            and getattr(self.spec, "model_id", None) == MODEL_IDENTITY["model_id"]
            and getattr(self.spec, "revision", None) == MODEL_IDENTITY["revision"]
            and str(getattr(self.spec, "device", None)) == "cuda:0"
            and getattr(self.spec, "precision", None) == "bfloat16"
            and getattr(self.spec, "context_limit", None) == CONTEXT_LIMIT,
            "genuine pinned CUDA provider specification required",
        )
        runtime = self.runtime
        require(
            runtime is self.base.runtime
            and str(runtime.device) == "cuda:0"
            and runtime.model_name == MODEL_IDENTITY["name"]
            and runtime.batch_size == BATCH_SIZE
            and runtime.temperature == 1.0
            and runtime.max_length is None
            and runtime.readout_dtype == "float32"
            and runtime.attention_mode == "noncausal_full_attention"
            and runtime.permutation_average is False
            and runtime.fast is None,
            "unchanged genuine reference-CUDA nativebatch8 runtime required",
        )
        require(
            tuple(runtime.readout.shape) == (CODES, WIDTH)
            and runtime.readout.dtype == self.torch.float32
            and str(runtime.readout.device) == "cuda:0",
            "original CUDA FP32 full255 W0 required",
        )
        parameters = list(runtime.backbone.parameters())
        require(
            parameters
            and all(
                str(p.device) == "cuda:0"
                and p.dtype == self.torch.bfloat16
                and p.requires_grad is False
                for p in parameters
            )
            and runtime.backbone.training is False,
            "original frozen CUDA BF16 evaluation backbone required",
        )
        require(
            runtime.manifest["identity"]["model_sha256"] == MODEL_IDENTITY["model_sha256"]
            and self.base.identity["manifest_sha256"] == MODEL_IDENTITY["manifest_sha256"]
            and sum(parameter.numel() for parameter in parameters) + runtime.readout.numel()
            == MODEL_IDENTITY["loaded_parameters"],
            "actual pinned publisher model/manifest/parameter identity differs",
        )

    def _guard(self):
        self._validate_runtime()
        require(
            _baseline_identity(self.runtime) == self._baseline
            and _tensor_identity(self.delta) == self._delta_identity
            and _policy(self.torch) == self._precision_policy,
            "baseline/delta metadata or FP32 policy changed",
        )
        value = self.identity_guard(self.base, self.delta)
        require(
            type(value) is dict and bool(value),
            "owned caller source/runtime/byte guard snapshot required",
        )
        _json(value)
        self._validate_runtime()
        require(
            _baseline_identity(self.runtime) == self._baseline
            and _tensor_identity(self.delta) == self._delta_identity
            and _policy(self.torch) == self._precision_policy,
            "owned guard callback mutated baseline/delta metadata or precision policy",
        )
        return copy.deepcopy(value)

    def _payload(self, request):
        require(
            isinstance(request, self.request_type), "normally validated native request DTO required"
        )
        parsed = self.request_type.model_validate(copy.deepcopy(request.model_dump()))
        require(
            not parsed.media and 1 <= len(parsed.questions) <= MAX_ROWS,
            "complete text-only1..4 head deployment scope required",
        )
        questions = {}
        for question in parsed.questions:
            require(
                question.id not in questions
                and question.type == "choice"
                and type(question.criteria) is dict
                and 1 <= len(question.criteria) <= CODES,
                "unique complete native choice heads/K1..255 required",
            )
            questions[question.id] = {
                "type": "choice",
                "instructions": copy.deepcopy(question.instructions),
                "criteria": copy.deepcopy(question.criteria),
            }
        payload = {"state": copy.deepcopy(parsed.state), "questions": questions}
        _json(payload)
        return parsed, payload

    def _emit(self, local_sink, event, evidence):
        _json(evidence)
        seen = set()
        for sink in (self.evidence_sink, local_sink):
            if sink is not None and id(sink) not in seen:
                seen.add(id(sink))
                sink(event, copy.deepcopy(evidence))

    def predict(self, request, *, deadline, evidence_sink=None):
        release_gate()
        require(not self.closed and not self._in_call, "open serialized GPU composition required")
        require(
            callable(deadline) and (evidence_sink is None or callable(evidence_sink)),
            "required owned full-call deadline and callable evidence sink",
        )
        self._in_call = True
        try:
            return self._predict_owned(request, deadline=deadline, evidence_sink=evidence_sink)
        finally:
            # The serialization lease covers every required terminal sink,
            # source/tensor guard, CUDA fence and final budget authorization.
            self._in_call = False

    def _predict_owned(self, request, *, deadline, evidence_sink=None):
        release_gate()
        require(
            self._in_call is True and not self.closed, "active outer GPU callback lease required"
        )
        started = self.clock()
        audit = {
            "profile": PROFILE,
            "alias": ALIAS,
            "capability_sha256": self.capability_sha256,
            "kind": "operation_delta_composition",
            "purpose": PURPOSE,
            "base_forward_native": True,
            "publisher_native_numerics": False,
            "W0_merged_or_mutated": False,
            "status": "preparing",
            "completed": False,
            "only_adapted_head_id": "operation",
            "provider_events": [],
            "guard_snapshots": [],
            "raw_target_answers_bitwise_preserved": False,
            "secondary_errors": [],
            "call_budget_seconds": CALL_SECONDS,
            "browser_authorized": False,
            "benchmark_authorized": False,
            "training_authorized": False,
            "private_tensor_witnesses_retained_by_caller": False,
            "final_immutability_guard_passed": False,
        }
        self.last_evidence = audit
        primary = result = observer = features_snapshot = base_logits = None
        handles = []

        def check(phase):
            deadline()
            elapsed = self.clock() - started
            require(
                math.isfinite(elapsed) and 0 <= elapsed < CALL_SECONDS,
                "full GPU composition20s deadline exceeded",
            )
            audit["last_checked_phase"] = phase

        def cleanup_error(error, phase):
            nonlocal primary
            audit["secondary_errors"].append(
                {"stage": phase, "type": type(error).__name__, "message": str(error)}
            )
            if primary is None:
                primary = error
            else:
                _note(primary, error, phase)

        try:
            check("entry_before_DTO_validation")
            audit["guard_snapshots"].append(self._guard())
            check("after_initial_source_runtime_byte_guard")
            parsed, payload = self._payload(request)
            expected_payload = _json(payload)
            audit["native_payload_sha256"] = hashlib.sha256(expected_payload).hexdigest()
            audit["original_transport_model"] = getattr(parsed, "model", None)
            keys = list(payload["questions"])
            audit["original_question_order"] = keys
            prepared = self.runtime.prepare(**copy.deepcopy(payload))
            lengths = {key: len(sequence) for key, sequence in prepared.sequences.items()}
            lengths.update(prepared.lengths)
            require(
                not prepared.errors
                and prepared.keys == prepared.runnable == keys
                and list(lengths) == keys
                and list(prepared.sequences) == keys
                and not any(
                    (prepared.images, prepared.videos, prepared.media, prepared.reversed_sequences)
                ),
                "unchanged complete text-only native preparation required",
            )
            require(
                all(
                    type(value) is int
                    and 0 < value <= CONTEXT_LIMIT
                    and value == len(prepared.sequences[key])
                    for key, value in lengths.items()
                ),
                "complete native lengths/context cap required; nothing truncated",
            )
            criteria = [list(payload["questions"][key]["criteria"]) for key in keys]
            require(
                all(
                    list(prepared.questions[key].keys) == offered
                    for key, offered in zip(keys, criteria, strict=True)
                ),
                "prepared full criterion order changed",
            )
            counts = [len(offered) for offered in criteria]
            observer = GPUObserver(self.torch, len(keys), max(lengths.values()))
            language_model = self.runtime.backbone.language_model
            handles.append(language_model.layers[0].register_forward_pre_hook(observer.first_layer))
            handles.append(language_model.norm.register_forward_hook(observer))
            baseline_events = []
            baseline_evidence = None

            def baseline_sink(event, evidence):
                nonlocal baseline_evidence
                require(
                    event in {"native_preflight", "native_raw_returned"}
                    and type(evidence) is dict
                    and set(evidence) == {"native_payload", "preflight", "raw_response"},
                    "exact original native provider event required",
                )
                require(
                    _json(evidence["native_payload"]) == expected_payload,
                    "baseline provider changed state/head/criterion order",
                )
                preflight = evidence["preflight"]
                require(
                    type(preflight) is dict
                    and set(preflight) == {"lengths", "errors", "project_context_limit"}
                    and list(preflight["lengths"]) == keys
                    and preflight["lengths"] == lengths
                    and not preflight["errors"]
                    and preflight["project_context_limit"] == CONTEXT_LIMIT,
                    "baseline provider full preflight binding differs",
                )
                if event == "native_preflight":
                    require(
                        not baseline_events and evidence["raw_response"] is None,
                        "exactly one original preflight required",
                    )
                    forwarded_event = "native_preflight"
                else:
                    require(
                        baseline_events == ["native_preflight"]
                        and evidence["raw_response"] is not None,
                        "one original baseline raw response after preflight required",
                    )
                    baseline_evidence = copy.deepcopy(evidence)
                    forwarded_event = "baseline_native_raw_returned"
                baseline_events.append(event)
                audit["provider_events"].append(forwarded_event)
                self._emit(evidence_sink, forwarded_event, evidence)
                check("after_baseline_provider_evidence")

            self.torch.cuda.synchronize(0)
            check("before_one_genuine_provider_predict")
            baseline_result = self.base.predict(
                parsed,
                deadline=lambda: check("inside_genuine_provider"),
                evidence_sink=baseline_sink,
            )
            self.torch.cuda.synchronize(0)
            require(
                baseline_events == ["native_preflight", "native_raw_returned"]
                and baseline_evidence is not None,
                "one complete original native provider call required",
            )
            returned_evidence = baseline_result.get("decision3_provider_evidence")
            require(
                type(returned_evidence) is dict
                and _json(returned_evidence) == _json(baseline_evidence),
                "baseline provider returned different original evidence",
            )
            observer.require_single_pass()
            baseline_raw = copy.deepcopy(baseline_evidence["raw_response"])
            _validate_raw(baseline_raw, payload["questions"], lengths)
            features_snapshot = observer.features.detach().clone()
            with self.torch.inference_mode():
                base_logits = observer.features.float() @ self.runtime.readout.T
                limit = self.torch.as_tensor(counts, device="cuda:0")[:, None]
                invalid = self.torch.arange(CODES, device="cuda:0")[None] >= limit
                base_logits = base_logits.masked_fill(invalid, float("-inf"))
                baseline_values_full = self.torch.softmax(base_logits / 1.0, dim=-1).cpu().tolist()
            baseline_values = [
                row[:count] for row, count in zip(baseline_values_full, counts, strict=True)
            ]
            audit["baseline_replay"] = _compare_baseline(
                baseline_raw, payload["questions"], baseline_values
            )
            check("after_same_call_full_baseline_replay")
            values, projection = project(
                self.torch, base_logits, observer.features, self.delta, counts, keys
            )
            self.torch.cuda.synchronize(0)
            audit["projection"] = copy.deepcopy(projection)
            self._emit(evidence_sink, "operation_delta_projected", projection)
            audit["provider_events"].append("operation_delta_projected")
            check("after_operation_only_projection_and_evidence")
            composed_raw = copy.deepcopy(baseline_raw)
            if "operation" in keys:
                probabilities = {
                    key: values[index]
                    if key == "operation"
                    else list(baseline_raw["answers"][key]["probabilities"].values())
                    for index, key in enumerate(keys)
                }
                responded = self.runtime.respond(prepared, probabilities, sum(lengths.values()))
                _validate_raw(responded, payload["questions"], lengths)
                composed_raw["answers"]["operation"] = copy.deepcopy(
                    responded["answers"]["operation"]
                )
            _validate_raw(composed_raw, payload["questions"], lengths)
            require(
                all(
                    _json(composed_raw["answers"][key]) == _json(baseline_raw["answers"][key])
                    for key in keys
                    if key != "operation"
                ),
                "original native raw target answers changed",
            )
            audit["raw_target_answers_bitwise_preserved"] = True
            require(
                _bytes_equal(self.torch, observer.features, features_snapshot),
                "same-call features changed before private witness",
            )
            check("before_required_private_tensor_witness")
            self.witness_sink(
                torch=self.torch,
                features=observer.features,
                baseline=self.runtime.readout,
                delta=self.delta,
                native_payload=copy.deepcopy(payload),
                lengths=copy.deepcopy(lengths),
                ordered_head_ids=list(keys),
                ordered_criteria_ids=copy.deepcopy(criteria),
                baseline_raw=copy.deepcopy(baseline_raw),
                composed_raw=copy.deepcopy(composed_raw),
                projection=copy.deepcopy(projection),
            )
            self.torch.cuda.synchronize(0)
            require(
                _bytes_equal(self.torch, observer.features, features_snapshot),
                "private witness callback mutated same-call features",
            )
            audit["private_tensor_witnesses_retained_by_caller"] = True
            audit["guard_snapshots"].append(self._guard())
            check("after_required_private_witness_and_immutability_guard")
            provider_evidence = {
                "kind": "operation_delta_composition",
                "native_payload": copy.deepcopy(payload),
                "preflight": copy.deepcopy(baseline_evidence["preflight"]),
                "raw_response": copy.deepcopy(composed_raw),
                "baseline_provider_evidence": copy.deepcopy(baseline_evidence),
                "composition": {
                    "profile": PROFILE,
                    "alias": ALIAS,
                    "capability_sha256": self.capability_sha256,
                    "publisher_native_numerics": False,
                    "only_head": "operation",
                    "projection": copy.deepcopy(projection),
                    "raw_target_answers_bitwise_preserved": True,
                },
            }
            self._emit(evidence_sink, "native_raw_returned", provider_evidence)
            audit["provider_events"].append("native_raw_returned")
            audit["baseline_provider_evidence"] = copy.deepcopy(baseline_evidence)
            audit["raw_response"] = copy.deepcopy(composed_raw)
            check("after_composed_raw_retention")
            result = self.normalize(parsed, composed_raw)
            require(
                type(result) is dict and result.get("model") == MODEL_IDENTITY["name"],
                "normal project response must retain internal publisher model name",
            )
            result["decision3_provider_evidence"] = provider_evidence
            check("after_normal_project_normalization")
        except BaseException as error:
            primary = error
        finally:
            for handle in reversed(handles):
                try:
                    handle.remove()
                except BaseException as error:
                    cleanup_error(error, "observation_hook_removal")
            handles.clear()
            if observer is not None:
                try:
                    audit["first_layer_events"] = copy.deepcopy(observer.first_layer_events)
                    audit["final_norm_events"] = copy.deepcopy(observer.norm_events)
                except BaseException as error:
                    cleanup_error(error, "observation_metadata_retention")
                observer.features = None
            observer = features_snapshot = base_logits = None
            try:
                self.torch.cuda.synchronize(0)
            except BaseException as error:
                cleanup_error(error, "final_CUDA_fence")
            try:
                audit["guard_snapshots"].append(self._guard())
            except BaseException as error:
                cleanup_error(error, "final_source_runtime_tensor_guard")
            try:
                check("final_authorization_after_hooks_fences_copies_and_guards")
            except BaseException as error:
                cleanup_error(error, "final_full_call_deadline")
        audit["pre_terminal_elapsed_seconds"] = self.clock() - started
        if primary is not None:
            audit.update(
                status="failed",
                completed=False,
                error={"type": type(primary).__name__, "message": str(primary)},
            )
            try:
                self._emit(evidence_sink, "failed", audit)
            except BaseException as error:
                cleanup_error(error, "failed evidence retention")
            try:
                self.torch.cuda.synchronize(0)
            except BaseException as error:
                cleanup_error(error, "post_failed_sink_CUDA_fence")
            try:
                audit["guard_snapshots"].append(self._guard())
            except BaseException as error:
                cleanup_error(error, "post_failed_sink_source_runtime_tensor_guard")
            try:
                check("final_failed_callback_authorization_after_required_sink")
            except BaseException as error:
                cleanup_error(error, "post_failed_sink_deadline")
            primary.operation_delta_evidence = copy.deepcopy(audit)
            raise primary
        audit.update(status="completed", completed=True)
        try:
            self._emit(
                evidence_sink,
                "completed",
                {
                    "profile": PROFILE,
                    "capability_sha256": self.capability_sha256,
                    "completed": True,
                },
            )
            self.torch.cuda.synchronize(0)
            audit["guard_snapshots"].append(self._guard())
            check("final_authorization_after_complete_evidence")
            audit["final_immutability_guard_passed"] = True
        except BaseException as error:
            primary = error
            audit.update(
                status="failed",
                completed=False,
                error={"type": type(error).__name__, "message": str(error)},
            )
            try:
                self._emit(evidence_sink, "failed", audit)
            except BaseException as failure:
                cleanup_error(failure, "terminal_failure_evidence_retention")
            try:
                self.torch.cuda.synchronize(0)
            except BaseException as failure:
                cleanup_error(failure, "post_terminal_failure_CUDA_fence")
            try:
                audit["guard_snapshots"].append(self._guard())
            except BaseException as failure:
                cleanup_error(failure, "post_terminal_failure_source_runtime_tensor_guard")
            try:
                check("final_failed_completed_sink_authorization")
            except BaseException as failure:
                cleanup_error(failure, "post_terminal_failure_deadline")
            primary.operation_delta_evidence = copy.deepcopy(audit)
            raise primary
        return result

    def telemetry(self):
        release_gate()
        return {
            "profile": PROFILE,
            "alias": ALIAS,
            "base_model": MODEL_IDENTITY["model_id"],
            "native_batch_size": BATCH_SIZE,
            "max_actual_rows": MAX_ROWS,
            "precision": "BF16 backbone; FP32 separate W0 and operation delta",
            "capability_sha256": self.capability_sha256,
            "composition_only": True,
            "browser_authorized": False,
            "benchmark_authorized": False,
            "training_authorized": False,
            "closed": self.closed,
        }

    def resources(self):
        release_gate()
        require(not self.closed, "closed composition has no live tensor resources")
        values = copy.deepcopy(self.base.resources())
        values["owned_operation_delta_bytes"] = self.delta.numel() * self.delta.element_size()
        return values

    def close(self):
        release_gate()
        require(not self._in_call, "cannot close during an active owned callback")
        self.delta = None
        self.runtime = None
        self.base = None
        self.closed = True
        # The shared genuine base is deliberately not closed or mutated here.
