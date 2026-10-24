"""Held CPU5 operation-only delta composition; no framework imports or file IO.

The owned caller supplies authenticated dependencies, the already loaded original
runtime and a separately accepted train-only delta. This module is no loader,
guardian, publisher-native qualification or HTTP deadline authority.
"""

from __future__ import annotations

import copy
import math
import re
import time

SOURCE_ONLY = False
CAPABILITY_SHA256 = "4dbbb9470d011931854f6473c0840ef7f0746c4194c719cde6ab454c3e79d463"
PROFILE = "decision3-dsl01-operation-readout-cpu5-4g-candidate-v1"
ALIAS = "d3-edge-dsl01-operation-delta-cpu5-candidate"
OWNER_PROFILE = "llama-cpp-cpu5-fast-same-cluster-4g-coexist-command-owned-v1"
CAPABILITY_SCHEMA = "root-accepted-dsl01-operation-train-delta-v1"
CONTEXT_LIMIT, BATCH_SIZE, CODES, WIDTH, CALL_SECONDS = 16384, 4, 255, 1024, 20


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def release_gate():
    require(
        SOURCE_ONLY is False,
        "SOURCE_ONLY: no dependency, runtime, delta, capability or prediction access authorized",
    )
    require(_sha(CAPABILITY_SHA256), "new root-accepted operation-only capability SHA pending")


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _ref(value):
    return (
        isinstance(value, dict)
        and isinstance(value.get("path"), str)
        and bool(value["path"])
        and _sha(value.get("sha256"))
        and type(value.get("size_bytes")) is int
        and value["size_bytes"] > 0
    )


def validate_capability(capability, model):
    """Check an explicit caller-authenticated contract, never infer from a filename.

    This does not authenticate file bytes, a producer's terminal/closure, or the
    correspondence of an in-memory tensor to an artifact. Those remain required
    responsibilities of the independently selected owned caller.
    """
    require(
        isinstance(capability, dict)
        and capability.get("schema") == CAPABILITY_SCHEMA
        and capability.get("status") == "ROOT_ACCEPTED_OPERATION_TRAIN_ONLY_READOUT"
        and capability.get("split") == "train"
        and capability.get("training_eligible") is True
        and capability.get("nontraining") is False
        and capability.get("profile") == PROFILE
        and capability.get("alias") == ALIAS
        and capability.get("owner_profile") == OWNER_PROFILE
        and capability.get("only_head") == "operation"
        and capability.get("dev_used_for_model_selection") is True
        and capability.get("optimizer_input_dev") is False
        and capability.get("optimizer_input_sealed") is False
        and capability.get("candidate_selection_before_dev_or_sealed") is False
        and capability.get("model") == model
        and _ref(capability.get("delta_ref"))
        and _ref(capability.get("root_acceptance_ref"))
        and _ref(capability.get("fit_capability"))
        and _sha(capability.get("baseline_fp32_bytes_sha256")),
        "explicit new root-accepted operation-only train delta/model/baseline capability required",
    )


def project(torch, base_logits, features, delta, counts, ordered_head_ids):
    """Add h@delta.T only for the actual head ID 'operation', at any row/count.

    W0 is never merged with delta. Base logits already contain the original
    masked -inf columns; applying the same count mask again is deliberate.
    """
    release_gate()
    rows = len(counts)
    require(
        1 <= rows <= BATCH_SIZE
        and type(ordered_head_ids) is list
        and len(ordered_head_ids) == rows
        and all(isinstance(key, str) and key for key in ordered_head_ids)
        and len(set(ordered_head_ids)) == rows
        and all(type(n) is int and 1 <= n <= CODES for n in counts)
        and tuple(base_logits.shape) == (rows, CODES)
        and tuple(features.shape) == (rows, WIDTH)
        and tuple(delta.shape) == (CODES, WIDTH)
        and all(
            t.dtype == torch.float32 and t.device.type == "cpu"
            for t in (base_logits, features, delta)
        ),
        "complete CPU FP32 original logits/features/delta/counts required",
    )
    with torch.inference_mode():
        updates = torch.zeros_like(base_logits)
        combined = base_logits.clone()
        operation_rows = [i for i, head in enumerate(ordered_head_ids) if head == "operation"]
        for i in operation_rows:
            added = features[i : i + 1] @ delta.T
            require(
                bool(torch.isfinite(added).all().item()),
                "nonfinite operation-only delta projection",
            )
            updates[i : i + 1] = added
            combined[i : i + 1] = base_logits[i : i + 1] + added
        limit = torch.as_tensor(list(counts), device="cpu")[:, None]
        invalid = torch.arange(CODES, device="cpu")[None] >= limit
        require(
            bool(torch.isfinite(base_logits.masked_fill(invalid, 0)).all().item()),
            "nonfinite legal original logits",
        )
        combined = combined.masked_fill(invalid, float("-inf"))
        preserved = [i for i, head in enumerate(ordered_head_ids) if head != "operation"]
        require(
            all(
                torch.equal(
                    combined[i : i + 1].contiguous().view(torch.uint8),
                    base_logits[i : i + 1].contiguous().view(torch.uint8),
                )
                for i in preserved
            ),
            "non-operation logits changed at the byte level",
        )
        require(
            bool(torch.isfinite(combined.masked_fill(invalid, 0)).all().item()),
            "nonfinite legal adapted logits",
        )
        # Native validate_runtime enforces temperature=1.0. No calibration or
        # extra normalization is introduced between the original two operations.
        probabilities = torch.softmax(combined / 1.0, dim=-1).cpu().tolist()
        original = base_logits.cpu().tolist()
        added = updates.cpu().tolist()
        adapted = combined.cpu().tolist()
    require(
        len(probabilities) == rows
        and all(
            len(row) == CODES
            and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in row)
            and abs(sum(row) - 1) <= 1e-6
            and all(value == 0 for value in row[count:])
            for row, count in zip(probabilities, counts)
        ),
        "complete finite normalized legal adapted distribution required",
    )
    return [row[:n] for row, n in zip(probabilities, counts)], {
        "ordered_head_ids": list(ordered_head_ids),
        "adapted_head_ids": [ordered_head_ids[i] for i in operation_rows],
        "preserved_head_ids": [ordered_head_ids[i] for i in preserved],
        "target_logits_bitwise_unchanged": True,
        "base_legal_logits": [row[:n] for row, n in zip(original, counts)],
        "delta_legal_logits": [row[:n] for row, n in zip(added, counts)],
        "adapted_legal_logits": [row[:n] for row, n in zip(adapted, counts)],
    }


def _note(primary, secondary, phase):
    if hasattr(primary, "add_note"):
        primary.add_note(phase + ": " + type(secondary).__name__ + ": " + str(secondary))


class DeltaHeadBackend:
    """Composition around genuine NativeBatch4Backend.prepare and D3.logits/respond.

    No .spec or fake CUDA provider is fabricated. A future CPU candidate caller
    must bind the distinct alias/profile explicitly, outside the original native
    adapter's hardcoded publisher-name check.
    """

    name = ALIAS

    def __init__(
        self,
        native,
        observer,
        runtime,
        torch,
        model,
        delta,
        *,
        root_capability,
        root_capability_sha256,
        deadline,
        evidence_sink=None,
        clock=time.monotonic,
    ):
        release_gate()
        require(
            root_capability_sha256 == CAPABILITY_SHA256,
            "caller-authenticated new operation capability SHA differs",
        )
        validate_capability(root_capability, model)
        require(
            callable(deadline) and callable(clock), "owned full-call deadline callback required"
        )
        self.native, self.observer = native, observer
        self.runtime, self.torch = runtime, torch
        self.model = copy.deepcopy(model)
        self.capability = copy.deepcopy(root_capability)
        self.deadline, self.evidence_sink, self.clock = deadline, evidence_sink, clock
        self.base = native.NativeBatch4Backend(
            runtime, torch, self.model, context_limit=CONTEXT_LIMIT
        )
        require(
            torch.is_tensor(delta)
            and tuple(delta.shape) == (CODES, WIDTH)
            and delta.dtype == torch.float32
            and delta.device.type == "cpu"
            and delta.requires_grad is False
            and delta.is_contiguous()
            and bool(torch.isfinite(delta).all().item()),
            "accepted delta requires finite contiguous frozen CPU FP32[255,1024]",
        )
        self.delta = delta.detach().clone()
        require(self.delta.data_ptr() != delta.data_ptr(), "owned independent delta copy required")
        self.baseline_identity = observer.baseline_identity(runtime)
        self.delta_identity = observer.tensor_identity(self.delta)
        self.last_evidence = None

    def prepare(self, *args, **kwargs):
        release_gate()
        # Preserve native signature errors and every original argument. Never
        # remove kwargs, truncate context, reorder heads, or add forced questions.
        return self.base.prepare(*args, **kwargs)

    def _unchanged(self):
        self.base.check_runtime()
        require(
            self.observer.baseline_identity(self.runtime) == self.baseline_identity
            and self.observer.tensor_identity(self.delta) == self.delta_identity,
            "baseline/encoder or owned delta identity/version changed",
        )

    def __call__(self, state, questions, **kwargs):
        release_gate()
        started = self.clock()
        audit = {
            "profile": PROFILE,
            "alias": ALIAS,
            "base_model": self.model["name"],
            "publisher_native_numerics": False,
            "W0_merged_or_mutated": False,
            "identity_guard_scope": "object/storage/version metadata, not a live tensor-byte hash",
            "capability": copy.deepcopy(self.capability),
            "capability_sha256": CAPABILITY_SHA256,
            "only_adapted_head_id": "operation",
            "status": "preparing",
            "call_budget_seconds": CALL_SECONDS,
            "original_question_order": list(questions),
            "chunks": [],
            "raw_response": None,
            "deadline_scope": "cooperative backend20 plus caller full20/final HTTP authorization",
        }
        self.last_evidence = audit

        def check(phase):
            self.deadline()
            elapsed = self.clock() - started
            require(math.isfinite(elapsed) and elapsed >= 0, "invalid monotonic elapsed time")
            if elapsed >= CALL_SECONDS:
                raise TimeoutError("delta-head cooperative20s exceeded; no hard cancel")
            audit["last_checked_phase"] = phase

        def emit(event, data):
            if self.evidence_sink is not None:
                try:
                    self.evidence_sink(event, copy.deepcopy(data))
                except BaseException as error:
                    audit.setdefault("observer_errors", []).append(
                        {"type": type(error).__name__, "message": str(error)}
                    )
                    # A supplied sink is required retention, not best-effort observation.
                    raise

        primary = None
        try:
            check("before_prepare")
            self._unchanged()
            prepared, lengths = self.prepare(state, questions, **kwargs)
            keys = prepared.runnable
            require(
                keys == prepared.keys == list(questions)
                and set(prepared.sequences) == set(keys)
                and not any(
                    (prepared.images, prepared.videos, prepared.media, prepared.reversed_sequences)
                ),
                "complete original text-only non-permuted native rows required",
            )
            audit["lengths"] = dict(lengths)
            emit("prepared", {"lengths": lengths, "original_question_order": keys})
            check("after_prepare")
            probabilities = {}
            for start in range(0, len(keys), BATCH_SIZE):
                chunk = keys[start : start + BATCH_SIZE]
                sequences = [prepared.sequences[key] for key in chunk]
                counts = [len(prepared.questions[key].keys) for key in chunk]
                width = max(map(len, sequences))
                record = {
                    "head_order": list(chunk),
                    "ordered_criteria_ids": [list(prepared.questions[key].keys) for key in chunk],
                    "legal_counts": counts,
                    "rows": len(chunk),
                    "padded_width": width,
                    "logical_input_tokens": sum(map(len, sequences)),
                    "physical_padded_input_tokens": len(chunk) * width,
                    "status": "before_forward",
                }
                audit["chunks"].append(record)
                watch = self.observer.FinalNormObserver(self.torch, len(chunk), width)
                handles, forward_primary = [], None
                try:
                    handles.append(
                        self.runtime.backbone.language_model.layers[0].register_forward_pre_hook(
                            watch.first_layer
                        )
                    )
                    handles.append(
                        self.runtime.backbone.language_model.norm.register_forward_hook(watch)
                    )
                    check("before_forward")
                    record["status"] = "forward_started"
                    forward_started = self.clock()
                    base_logits = self.runtime.logits(sequences, counts)
                    record.update(
                        status="forward_returned",
                        forward_elapsed_seconds=self.clock() - forward_started,
                    )
                    watch.require_single_pass()
                    record["first_layer_events"] = copy.deepcopy(watch.first_layer_events)
                    record["final_norm_events"] = copy.deepcopy(watch.norm_events)
                    check("after_original_forward")
                    values, projection = project(
                        self.torch, base_logits, watch.features, self.delta, counts, list(chunk)
                    )
                    record.update(projection=projection, status="projected")
                    probabilities.update(zip(chunk, values))
                except BaseException as error:
                    forward_primary = error
                    raise
                finally:
                    # Untrusted audit copies must never prevent hook removal.
                    cleanup_error = None
                    for handle in reversed(handles):
                        try:
                            handle.remove()
                        except BaseException as error:
                            if cleanup_error is None:
                                cleanup_error = error
                            elif forward_primary is None:
                                _note(cleanup_error, error, "additional hook removal")
                            if forward_primary is not None:
                                _note(forward_primary, error, "hook removal")
                    try:
                        record["first_layer_events"] = copy.deepcopy(watch.first_layer_events)
                        record["final_norm_events"] = copy.deepcopy(watch.norm_events)
                    except BaseException as snapshot_error:
                        if forward_primary is not None:
                            _note(
                                forward_primary, snapshot_error, "audit snapshot after hook cleanup"
                            )
                        elif cleanup_error is not None:
                            _note(
                                cleanup_error, snapshot_error, "audit snapshot after hook cleanup"
                            )
                        else:
                            cleanup_error = snapshot_error
                    if cleanup_error is not None and forward_primary is None:
                        raise cleanup_error
                self._unchanged()
                emit("chunk_returned", record)
                check("after_projection_hook_cleanup_and_evidence")
            tokens = sum(len(prepared.sequences[key]) for key in keys)
            # Original respond supplies every head, entropy confidence, ordered
            # criteria and usage. Its returned base-name object is retained.
            raw = self.runtime.respond(prepared, probabilities, tokens)
            audit["respond_before_candidate_alias"] = copy.deepcopy(raw)
            audit["respond_input_profile"] = (
                "operation-only adapted probabilities; other heads native; not a baseline control response"
            )
            emit("raw_returned_before_validation", raw)
            check("after_original_respond_and_evidence")
            self.native.validate_native(raw, questions, lengths, self.model)
            result = copy.deepcopy(raw)
            result["model"] = ALIAS
            audit.update(raw_response=copy.deepcopy(result), status="completed")
            self._unchanged()
            emit("completed", {"profile": PROFILE, "raw_response": result})
            return result
        except BaseException as error:
            primary = error
            audit.update(
                status="failed", error={"type": type(error).__name__, "message": str(error)}
            )
            error.delta_head_evidence = audit
            try:
                emit("failed", audit)
            except BaseException as secondary:
                _note(error, secondary, "failure evidence retention")
            raise
        finally:
            # Never replace an original forward/deadline/response error with a
            # secondary identity failure. The owned caller must latch failures.
            try:
                self._unchanged()
                if primary is None:
                    audit["elapsed_seconds_before_final_gate"] = self.clock() - started
                    check("final_backend_authorization_after_identity_and_evidence")
            except BaseException as error:
                if primary is None:
                    audit.update(
                        status="failed", error={"type": type(error).__name__, "message": str(error)}
                    )
                    error.delta_head_evidence = audit
                    try:
                        emit("failed", audit)
                    except BaseException as secondary:
                        _note(error, secondary, "final failure evidence retention")
                    raise
                _note(primary, error, "final identity check")

    def answers(self, state, questions, **kwargs):
        return self(state, questions, **kwargs)["answers"]
