"""Invented inputs and doubles only: no model, GPU, cases, or file access."""

import hashlib
import math
from types import SimpleNamespace

import jev_decision3_cuda_synthetic_worker as worker
import pytest

import s1.evaluation.decision3 as native

UUID = "GPU-12345678-1234-1234-1234-123456789abc"


class Tensor:
    def __init__(self, dtype, count, shape=()):
        self.device = "cuda:0"
        self.dtype = dtype
        self.requires_grad = False
        self.shape = shape
        self.count = count

    def numel(self):
        return self.count

    def detach(self):
        return self

    def to(self, *, device):
        assert device == "cpu"
        return self

    def contiguous(self):
        return self

    def numpy(self):
        return SimpleNamespace(tobytes=lambda: b"synthetic readout")


class FakeCuda:
    def __init__(self):
        self.allocated = 0
        self.syncs = 0
        self.clears = 0
        self.reserved = 0

    def is_available(self):
        return True

    def device_count(self):
        return 1

    def current_device(self):
        return 0

    def get_device_properties(self, _index):
        return SimpleNamespace(uuid=UUID, major=12, minor=1)

    def is_bf16_supported(self, *, including_emulation):
        assert including_emulation is False
        return True

    def synchronize(self, _index):
        self.syncs += 1

    def memory_allocated(self, _index):
        return self.allocated

    def memory_reserved(self, _index):
        return self.reserved

    def empty_cache(self):
        self.clears += 1
        self.reserved = 0


class FakeTorch:
    bfloat16 = "bfloat16"
    float32 = "float32"

    def __init__(self):
        self.cuda = FakeCuda()
        self._C = SimpleNamespace()
        self.backends = SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True))
        )

    def is_tensor(self, value):
        return isinstance(value, Tensor)

    def set_float32_matmul_precision(self, precision):
        assert precision == "highest"

    def get_float32_matmul_precision(self):
        return "highest"

    def is_autocast_enabled(self, device):
        assert device == "cuda"
        return False


class FakeBackend:
    def __init__(self, spec, torch):
        self.spec = spec
        self.identity = {"manifest_sha256": worker.MANIFEST_SHA}
        self.torch = torch
        self.calls = 0
        self.closed = False
        self.mutate_response = lambda x: x
        readout = Tensor(torch.float32, 255 * 1024, (255, 1024))
        param = Tensor(torch.bfloat16, 594582720 - readout.numel())
        backbone = SimpleNamespace(training=False, parameters=lambda: iter([param]))
        self.runtime = SimpleNamespace(
            model_name="d3-edge",
            batch_size=8,
            temperature=1.0,
            max_length=None,
            attention_mode="noncausal_full_attention",
            permutation_average=False,
            fast=None,
            readout_dtype="float32",
            backbone=backbone,
            readout=readout,
        )
        self.model = SimpleNamespace(runtime=self.runtime)
        self.torch.cuda.allocated = 10
        self.torch.cuda.reserved = 12

    def predict(self, request, *, deadline):
        self.calls += 1
        deadline()
        payload = native.native_payload(request)
        confidence = 1 + (0.7 * math.log(0.7) + 0.3 * math.log(0.3)) / math.log(2)
        raw = {
            "model": "d3-edge",
            "answers": {
                "destination": {
                    "type": "choice",
                    "choice": "north",
                    "probabilities": {"north": 0.7, "south": 0.3},
                    "confidence": confidence,
                },
                "fragile": {"type": "noul", "noul": 0.4},
            },
            "usage": {"input_tokens": 7, "output_tokens": 0},
        }
        result = native.normalize_native_response(request, raw)
        result["decision3_provider_evidence"] = {
            "native_payload": payload,
            "preflight": {"lengths": {"destination": 4, "fragile": 3}, "errors": {}},
            "raw_response": raw,
        }
        return self.mutate_response(result)

    def close(self):
        self.closed = True
        self.torch.cuda.allocated = 0


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(worker, "SOURCE_ONLY", False)
    monkeypatch.setattr(worker, "READOUT_SHA", hashlib.sha256(b"synthetic readout").hexdigest())
    torch = FakeTorch()
    holders = []

    def create(spec):
        value = FakeBackend(spec, torch)
        holders.append(value)
        return value

    monkeypatch.setattr(native, "Decision3Backend", create)
    checks = []
    return torch, holders, checks


def execute(setup, **kwargs):
    torch, _holders, checks = setup
    options = {
        "torch": torch,
        "gpu_uuid": UUID,
        "source_guard": lambda: checks.append("source"),
        "ownership_guard": lambda: checks.append("ownership"),
        "clock": lambda: 1.0,
    }
    options.update(kwargs)
    return worker.run(**options)


def test_source_gate_fails_before_cuda_or_guard(setup, monkeypatch):
    monkeypatch.setattr(worker, "SOURCE_ONLY", True)
    with pytest.raises(ValueError, match="release required"):
        execute(setup)
    assert setup[0].cuda.syncs == 0 and setup[2] == []


def test_one_synthetic_call_retains_full_response_and_cleans_up(setup):
    result = execute(setup)
    torch, holders, checks = setup
    assert result["status"] == "completed_pending_external_audit"
    assert result["calls"] == 1 and result["score"] is None and result["device"] == "cuda:0"
    assert holders[0].calls == 1 and holders[0].closed
    assert result["evidence"]["usage"] == {"input_tokens": 7, "output_tokens": 0}
    assert result["evidence"]["raw_response"]["answers"]["fragile"]["noul"] == 0.4
    assert result["evidence"]["normalized_answers"]["fragile"]["probabilities"] == {
        "false": 0.6,
        "true": 0.4,
    }
    assert result["final_allocated_bytes"] == 0 and result["final_reserved_bytes"] == 0
    assert torch.cuda.syncs == 4 and torch.cuda.clears == 1
    assert checks.count("source") == 3 and checks.count("ownership") >= 3
    assert (
        "cooperative" in result["deadline_scope"]
        and "external hard limit" in result["deadline_scope"]
    )


def test_source_guard_failure_does_not_touch_cuda(setup):
    def reject():
        raise RuntimeError("guard rejected")

    with pytest.raises(RuntimeError) as error:
        execute(setup, source_guard=reject)
    assert setup[0].cuda.syncs == 0 and not setup[1]
    assert error.value.cuda_synthetic_evidence["status"] == "failed_no_score"
    assert "not started" in error.value.cuda_synthetic_evidence["cleanup"]


def test_bad_gpu_uuid_rejects_before_synchronization(setup):
    with pytest.raises(ValueError, match="selected GB10"):
        execute(setup, gpu_uuid="GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    assert setup[0].cuda.syncs == 0 and not setup[1]


def test_late_response_aborts_and_preserves_cleanup(setup):
    calls = iter([0.0, 0.0, 0.0, 21.0])
    with pytest.raises(ValueError, match="deadline") as error:
        execute(setup, clock=lambda: next(calls))
    assert setup[1][0].closed and setup[0].cuda.allocated == 0
    assert error.value.cuda_synthetic_evidence["status"] == "failed_no_score"


def test_cleanup_failure_never_returns_success(setup):
    torch, holders, _checks = setup
    original = torch.cuda.empty_cache

    def broken():
        original()
        raise RuntimeError("allocator failure")

    torch.cuda.empty_cache = broken
    with pytest.raises(RuntimeError) as error:
        execute(setup)
    assert holders[0].closed
    assert error.value.cuda_synthetic_evidence["cleanup"] == "unconfirmed"
    assert error.value.cuda_synthetic_evidence["status"] == "failed_no_score"


@pytest.mark.parametrize(
    "fault", ["missing", "probability", "confidence", "noul", "usage", "length", "payload"]
)
def test_incomplete_or_inconsistent_native_evidence_is_rejected(setup, monkeypatch, fault):
    torch, holders, _checks = setup
    original = native.Decision3Backend

    def create(spec):
        backend = original(spec)

        def mutate(value):
            raw = value["decision3_provider_evidence"]["raw_response"]
            if fault == "missing":
                raw["answers"].pop("fragile")
            elif fault == "probability":
                raw["answers"]["destination"]["probabilities"]["south"] = 0.1
            elif fault == "confidence":
                raw["answers"]["destination"]["confidence"] = 0.9
            elif fault == "noul":
                raw["answers"]["fragile"]["noul"] = 0.8
            elif fault == "usage":
                raw["usage"]["output_tokens"] = 1
            elif fault == "length":
                value["decision3_provider_evidence"]["preflight"]["lengths"]["fragile"] = 4
            else:
                value["decision3_provider_evidence"]["native_payload"]["state"] = "changed"
            return value

        backend.mutate_response = mutate
        return backend

    monkeypatch.setattr(native, "Decision3Backend", create)
    with pytest.raises(ValueError) as error:
        execute(setup)
    assert holders[0].closed and torch.cuda.allocated == 0
    assert error.value.cuda_synthetic_evidence["status"] == "failed_no_score"


def test_changed_readout_after_forward_fails_closed(setup, monkeypatch):
    original = native.Decision3Backend

    def create(spec):
        backend = original(spec)

        def mutate(result):
            backend.runtime.readout.numpy = lambda: SimpleNamespace(tobytes=lambda: b"changed")
            return result

        backend.mutate_response = mutate
        return backend

    monkeypatch.setattr(native, "Decision3Backend", create)
    with pytest.raises(ValueError, match="original readout") as error:
        execute(setup)
    assert setup[1][0].closed
    assert error.value.cuda_synthetic_evidence["status"] == "failed_no_score"
