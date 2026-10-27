"""Candidate native Decision3 two-head readout adapter.

Only an already loaded, caller-owned Decision3Backend is wrapped. No file,
model, service or GPU is opened here. Raw FP32 delta hashes and a private
same-forward witness are mandatory; the caller owns resource supervision and
must independently audit the witness before making any qualification claim.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
import threading
import time

import jev_decision3_link_navigation_features as features_module
import jev_decision3_operation_gpu as reference

from s1.evaluation.decision3 import native_payload, normalize_native_response
from s1.jev_decision3_native import Decision3NativeRequest

HEADS = ("operation", "click_target")
CODES, WIDTH, MAX_HEADS = 255, 1024, 4
PROFILE = "decision3-link-navigation-dual-head-candidate-v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _tensor_sha(tensor):
    raw = tensor.detach().to(device="cpu", dtype=tensor.dtype).contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _valid_tensor(torch, tensor, *, device, shape):
    return bool(
        torch.is_tensor(tensor)
        and tuple(tensor.shape) == shape
        and tensor.dtype == torch.float32
        and str(tensor.device) == device
        and not tensor.requires_grad
        and tensor.is_contiguous()
        and bool(torch.isfinite(tensor).all().item())
    )


def project(torch, base_logits, hidden, deltas, counts, heads):
    """Alter only legal ordinals of actually offered adapted question IDs."""
    n = len(heads)
    _require(
        1 <= n <= MAX_HEADS
        and len(set(heads)) == n
        and len(counts) == n
        and all(type(h) is str and h for h in heads)
        and all(type(k) is int and 1 <= k <= CODES for k in counts)
        and set(deltas) == set(HEADS)
        and _valid_tensor(torch, hidden, device="cuda:0", shape=(n, WIDTH))
        and _valid_tensor(torch, base_logits, device="cuda:0", shape=(n, CODES))
        and all(
            _valid_tensor(torch, deltas[h], device="cuda:0", shape=(CODES, WIDTH)) for h in HEADS
        ),
        "complete finite CUDA FP32 projection geometry required",
    )
    with torch.inference_mode():
        combined = base_logits.clone()
        for index, head in enumerate(heads):
            if head in HEADS:
                addition = hidden[index : index + 1] @ deltas[head].T
                invalid = (
                    torch.arange(CODES, device="cuda:0")[None, :]
                    >= torch.as_tensor([counts[index]], device="cuda:0")[:, None]
                )
                # Addition outside the offered candidate range is exactly zero.
                addition = addition.masked_fill(invalid, 0.0)
                original = base_logits[index : index + 1]
                combined[index : index + 1] = torch.where(invalid, original, original + addition)
        invalid = (
            torch.arange(CODES, device="cuda:0")[None, :]
            >= torch.as_tensor(counts, device="cuda:0")[:, None]
        )
        values = torch.softmax(combined.masked_fill(invalid, float("-inf")), dim=-1).cpu().tolist()
    _require(
        len(values) == n
        and all(
            len(row) == CODES
            and all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in row)
            and math.isclose(sum(row), 1.0, abs_tol=1e-6)
            and all(x == 0 for x in row[count:])
            for row, count in zip(values, counts, strict=True)
        ),
        "complete finite legal probability distribution required",
    )
    return [row[:count] for row, count in zip(values, counts, strict=True)], {
        "applied_head_ids": [head for head in heads if head in HEADS],
        "absent_head_ids": [head for head in HEADS if head not in heads],
        "legal_counts": list(counts),
        "ordered_head_ids": list(heads),
        "same_call_baseline_replayed": True,
        "unmodified_heads_preserved": True,
    }


class DualHeadDecision3Backend:
    """Wrap a native provider for the explicit text-only, 1–4 choice-head profile."""

    def __init__(
        self, base_provider, cpu_deltas, delta_sha256, torch, *, identity_guard, witness_sink
    ):
        _require(
            callable(identity_guard) and callable(witness_sink),
            "owner guard and witness sink required",
        )
        _require(
            type(cpu_deltas) is dict and set(cpu_deltas) == set(HEADS),
            "two explicit deltas required",
        )
        _require(
            type(delta_sha256) is dict
            and set(delta_sha256) == set(HEADS)
            and all(
                type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value)
                for value in delta_sha256.values()
            ),
            "two externally reviewed FP32 SHA256 pins required",
        )
        self.base = base_provider
        self.spec = base_provider.spec
        self.name = base_provider.name
        self.runtime = base_provider.runtime
        self.torch = torch
        self.identity_guard, self.witness_sink = identity_guard, witness_sink
        self.delta_sha256 = copy.deepcopy(delta_sha256)
        self.deltas = {}
        self.closed, self._in_call = False, False
        self._lock = threading.Lock()
        features_module._validate_provider_spec(self.spec)
        _require(
            getattr(base_provider.model, "runtime", None) is self.runtime,
            "original model/runtime required",
        )
        _require(
            _valid_tensor(torch, self.runtime.readout, device="cuda:0", shape=(CODES, WIDTH))
            and _tensor_sha(self.runtime.readout) == features_module.W0_SHA256,
            "original verified CUDA FP32 W0 required",
        )
        try:
            for head in HEADS:
                source = cpu_deltas[head]
                _require(
                    _valid_tensor(torch, source, device="cpu", shape=(CODES, WIDTH)),
                    "frozen CPU FP32 delta required",
                )
                _require(_tensor_sha(source) == self.delta_sha256[head], "CPU delta SHA256 differs")
                target = (
                    source.detach().to(device="cuda:0", dtype=torch.float32).contiguous().clone()
                )
                _require(
                    _valid_tensor(torch, target, device="cuda:0", shape=(CODES, WIDTH)),
                    "owned CUDA FP32 delta required",
                )
                _require(_tensor_sha(target) == self.delta_sha256[head], "CUDA delta bytes differ")
                self.deltas[head] = target
            self._guard()
        except BaseException as error:
            target = None
            self.deltas.clear()
            self.closed = True
            try:
                torch.cuda.synchronize(0)
            except BaseException as cleanup_error:
                error.add_note(
                    "Delta initialization CUDA fence also failed: " + repr(cleanup_error)
                )
            raise

    def _validate_runtime(self):
        runtime = self.runtime
        features_module._validate_provider_spec(self.spec)
        _require(
            runtime is self.base.runtime
            and getattr(self.base.model, "runtime", None) is runtime
            and runtime.model_name == "d3-edge"
            and runtime.batch_size == 8
            and runtime.temperature == 1.0
            and runtime.max_length is None
            and runtime.attention_mode == "noncausal_full_attention"
            and runtime.permutation_average is False
            and runtime.fast is None
            and runtime.readout_dtype == "float32"
            and runtime.encoder.language_model is runtime.backbone.language_model
            and runtime.backbone.training is False,
            "original frozen reference runtime required",
        )
        params = list(runtime.backbone.parameters())
        _require(
            params
            and all(
                str(p.device) == "cuda:0"
                and p.dtype == self.torch.bfloat16
                and p.requires_grad is False
                for p in params
            )
            and sum(p.numel() for p in params) + runtime.readout.numel() == 594582720,
            "original frozen CUDA BF16 parameter identity required",
        )
        _require(
            self.base.identity.get("model_id") == features_module.MODEL_ID
            and self.base.identity.get("manifest_sha256") == features_module.MANIFEST_SHA256
            and runtime.manifest.get("identity", {}).get("model_sha256")
            == features_module.MODEL_SHA256,
            "pinned publisher identity required",
        )

    def _guard(self):
        self._validate_runtime()
        _require(
            self.runtime is self.base.runtime
            and getattr(self.base.model, "runtime", None) is self.runtime,
            "provider runtime changed",
        )
        _require(
            _valid_tensor(self.torch, self.runtime.readout, device="cuda:0", shape=(CODES, WIDTH))
            and _tensor_sha(self.runtime.readout) == features_module.W0_SHA256,
            "original W0 changed",
        )
        _require(
            all(
                _valid_tensor(self.torch, self.deltas[head], device="cuda:0", shape=(CODES, WIDTH))
                and _tensor_sha(self.deltas[head]) == self.delta_sha256[head]
                for head in HEADS
            ),
            "delta bytes changed",
        )
        _require(
            self.torch.get_float32_matmul_precision() == "highest"
            and self.torch.backends.cuda.matmul.allow_tf32 is False
            and not self.torch.is_autocast_enabled("cuda"),
            "original FP32 numerical policy required",
        )
        evidence = self.identity_guard(self.base, self.deltas)
        _require(
            type(evidence) is dict and bool(evidence), "external owner identity guard required"
        )
        reference._json(evidence)
        self._validate_runtime()
        _require(
            _tensor_sha(self.runtime.readout) == features_module.W0_SHA256
            and all(_tensor_sha(self.deltas[head]) == self.delta_sha256[head] for head in HEADS),
            "owner guard mutated model or delta",
        )
        return copy.deepcopy(evidence)

    def predict(self, request, *, deadline, evidence_sink=None):
        _require(
            callable(deadline) and (evidence_sink is None or callable(evidence_sink)),
            "deadline and callable sink required",
        )
        _require(self._lock.acquire(blocking=False), "open serialized adapter required")
        primary = None
        try:
            _require(not self.closed, "open serialized adapter required")
            self._in_call = True
            return self._predict(request, deadline=deadline, evidence_sink=evidence_sink)
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                if self._in_call:
                    self.torch.cuda.synchronize(0)
            except BaseException as cleanup_error:
                if primary is None:
                    raise
                primary.add_note("Final CUDA fence also failed: " + repr(cleanup_error))
            finally:
                self._in_call = False
                self._lock.release()

    def _predict(self, request, *, deadline, evidence_sink):
        started = time.monotonic()

        def check():
            deadline()
            _require(time.monotonic() - started < 20, "full adapter deadline exceeded")

        check()
        _require(
            isinstance(request, Decision3NativeRequest),
            "validated original native request required",
        )
        parsed = Decision3NativeRequest.model_validate(copy.deepcopy(request.model_dump()))
        payload = native_payload(parsed)
        heads = list(payload["questions"])
        _require(
            1 <= len(heads) <= MAX_HEADS
            and all(q["type"] == "choice" for q in payload["questions"].values()),
            "complete original choice heads required",
        )
        self._guard()
        prepared = self.runtime.prepare(**copy.deepcopy(payload))
        lengths = features_module._prepared_contract(prepared, payload["questions"])
        counts = [len(payload["questions"][head]["criteria"]) for head in heads]
        criteria = [list(payload["questions"][head]["criteria"]) for head in heads]
        shape = (len(heads), max(lengths.values()), WIDTH)
        observed, first, events = [], [], []

        def first_hook(_module, args):
            _require(
                not first
                and args
                and tuple(args[0].shape) == shape
                and args[0].dtype == self.torch.bfloat16
                and str(args[0].device) == "cuda:0",
                "one original first-layer pass required",
            )
            first.append(tuple(args[0].shape))

        def norm_hook(_module, _args, output):
            _require(
                not observed
                and tuple(output.shape) == shape
                and output.dtype == self.torch.bfloat16
                and str(output.device) == "cuda:0",
                "one original final-norm pass required",
            )
            observed.append(output[:, -1, :].detach().to(dtype=self.torch.float32).clone())

        def sink(event, value):
            _require(
                len(events) < 2
                and event == ("native_preflight", "native_raw_returned")[len(events)],
                "ordered native events required",
            )
            events.append(copy.deepcopy(value))
            if event == "native_preflight" and evidence_sink is not None:
                evidence_sink(event, copy.deepcopy(value))

        language = self.runtime.backbone.language_model
        check()
        result = features_module._call_with_hooks(
            language,
            self.torch,
            first_hook,
            norm_hook,
            lambda: self.base.predict(parsed, deadline=check, evidence_sink=sink),
        )
        _require(len(first) == len(observed) == 1, "one same-call feature witness required")
        baseline_raw = features_module._verify_call_contract(
            events, payload, lengths, first[0], shape, result
        )
        reference._validate_raw(baseline_raw, payload["questions"], lengths)
        hidden = observed[0]
        _require(
            _valid_tensor(self.torch, hidden, device="cuda:0", shape=(len(heads), WIDTH)),
            "finite same-call features required",
        )
        with self.torch.inference_mode():
            logits = hidden @ self.runtime.readout.T
        zeros = {head: self.torch.zeros_like(self.deltas[head]) for head in HEADS}
        baseline_values, _ = project(self.torch, logits, hidden, zeros, counts, heads)
        reference._compare_baseline(baseline_raw, payload["questions"], baseline_values)
        adapted_values, projection = project(self.torch, logits, hidden, self.deltas, counts, heads)

        def respond(values):
            probabilities = {
                head: values[index]
                if head in HEADS
                else list(baseline_raw["answers"][head]["probabilities"].values())
                for index, head in enumerate(heads)
            }
            candidate = self.runtime.respond(prepared, probabilities, sum(lengths.values()))
            reference._validate_raw(candidate, payload["questions"], lengths)
            result = copy.deepcopy(baseline_raw)
            for head in heads:
                if head in HEADS:
                    result["answers"][head] = copy.deepcopy(candidate["answers"][head])
            reference._validate_raw(result, payload["questions"], lengths)
            return result

        zero_raw = respond(baseline_values)
        reference._compare_baseline(zero_raw, payload["questions"], baseline_values)
        composed_raw = respond(adapted_values)
        _require(
            all(
                reference._json(composed_raw["answers"][head])
                == reference._json(baseline_raw["answers"][head])
                for head in heads
                if head not in HEADS
            ),
            "unadapted native answers changed",
        )
        check()
        feature_sha = _tensor_sha(hidden)
        self.witness_sink(
            torch=self.torch,
            features=hidden,
            baseline=self.runtime.readout,
            deltas=dict(self.deltas),
            delta_sha256=copy.deepcopy(self.delta_sha256),
            native_payload=copy.deepcopy(payload),
            lengths=copy.deepcopy(lengths),
            ordered_head_ids=list(heads),
            ordered_criteria_ids=copy.deepcopy(criteria),
            baseline_raw=copy.deepcopy(baseline_raw),
            zero_raw=copy.deepcopy(zero_raw),
            composed_raw=copy.deepcopy(composed_raw),
            projection=copy.deepcopy(projection),
        )
        self.torch.cuda.synchronize(0)
        _require(_tensor_sha(hidden) == feature_sha, "witness mutated same-call features")
        self._guard()
        check()
        provider_evidence = {
            "native_payload": copy.deepcopy(payload),
            "preflight": copy.deepcopy(events[0]["preflight"]),
            "raw_response": copy.deepcopy(composed_raw),
            "baseline_provider_evidence": copy.deepcopy(events[1]),
            "composition": {
                "profile": PROFILE,
                "delta_sha256": copy.deepcopy(self.delta_sha256),
                "projection": projection,
            },
        }
        if evidence_sink is not None:
            evidence_sink("native_raw_returned", copy.deepcopy(provider_evidence))
        normalized = normalize_native_response(parsed, composed_raw)
        normalized["decision3_provider_evidence"] = provider_evidence
        check()
        return normalized

    def close(self):
        _require(self._lock.acquire(blocking=False), "cannot close an active adapter")
        try:
            self.deltas.clear()
            self.closed = True
        finally:
            self._lock.release()
