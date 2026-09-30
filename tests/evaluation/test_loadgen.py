import time
from dataclasses import replace
from pathlib import Path
from threading import Lock

import pytest

from s1.backends import UniformBackend
from s1.errors import BackendTimeoutError
from s1.evaluation.adapters import ReferenceAdapter
from s1.evaluation.artifacts import read_records
from s1.evaluation.contracts import ProfileSpec
from s1.evaluation.loadgen import arrival_offsets
from s1.evaluation.registry import Registry
from s1.evaluation.runner import execute

ROOT = Path(__file__).resolve().parents[2]


def test_arrival_schedules_are_seeded_and_independent_of_service_time():
    assert arrival_offsets(3, "fixed", 4, 0) == [0, 0.25, 0.5]
    assert arrival_offsets(10, "poisson", 10, 42) == arrival_offsets(10, "poisson", 10, 42)
    assert arrival_offsets(10, "poisson", 10, 42) != arrival_offsets(10, "poisson", 10, 43)
    with pytest.raises(ValueError):
        ProfileSpec(id="x", suite="x", models=("x",), load_mode="fixed")


@pytest.mark.parametrize("mode", ["closed_loop", "fixed", "poisson"])
def test_concurrency_deadlines_and_censored_timeouts(tmp_path, mode):
    config = ROOT / "configs" / "benchmarks"
    registry = Registry.load(
        config / "models.toml", config / "suites.toml", config / "profiles.toml"
    )
    profile = registry.profiles["text-smoke"].model_copy(
        update={
            "concurrency": 2,
            "load_mode": mode,
            "arrival_rate": 10000.0,
            "slo_ms": 1.0,
            "warmup": 0,
        }
    )
    registry = replace(registry, profiles={profile.id: profile})

    class Slow(UniformBackend):
        active = peak = count = 0
        lock = Lock()

        def predict(self, request):
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
                self.count += 1
                count = self.count
            try:
                time.sleep(0.005)
                if count == 3:
                    raise BackendTimeoutError("private transport details")
                return super().predict(request)
            finally:
                with self.lock:
                    self.active -= 1

    backend = Slow()
    result = execute(
        registry,
        profile.id,
        output=tmp_path / "run",
        lockfile=ROOT / "uv.lock",
        enable_models=["uniform"],
        adapter_factory=lambda spec, _: ReferenceAdapter(spec, backend),
    )
    assert backend.peak == 2
    stats = result["models"]["uniform"]
    assert stats["request_counts"] == {"ok": 9, "timeout": 1}
    assert stats["records_complete"]
    assert stats["systems"]["correct_within_slo_per_second"] == 0
    assert stats["systems"]["deadline_misses"] == 10
    requests = read_records(tmp_path / "run", "requests")
    timeout = next(row for row in requests if row["status"] == "timeout")
    assert timeout["latency_ms"] is timeout["all_required_answers_ready_at_s"] is None
    if mode != "closed_loop":
        assert max(row["dispatch_lag_ms"] for row in requests) > 5


def test_native_batch_calls_once_per_batch_and_validates_each_member(tmp_path):
    config = ROOT / "configs" / "benchmarks"
    registry = Registry.load(
        config / "models.toml", config / "suites.toml", config / "profiles.toml"
    )
    model = registry.models["uniform"].model_copy(
        update={
            "capabilities": registry.models["uniform"].capabilities.model_copy(
                update={"batch": "loop_emulated"}
            )
        }
    )
    profile = registry.profiles["smoke"].model_copy(
        update={"batch_size": 2, "warmup": 0, "repetitions": 1}
    )
    registry = replace(registry, models={"uniform": model}, profiles={"smoke": profile})
    batches = []

    class Batch(ReferenceAdapter):
        def predict_batch(self, requests):
            batches.append(len(requests))
            return [self.backend.predict(request) for request in requests]

    result = execute(
        registry,
        "smoke",
        output=tmp_path / "run",
        lockfile=ROOT / "uv.lock",
        enable_models=["uniform"],
        adapter_factory=lambda spec, _: Batch(spec, UniformBackend()),
    )
    assert batches == [2, 1]
    assert result["models"]["uniform"]["decision_counts"] == {"ok": 5}
