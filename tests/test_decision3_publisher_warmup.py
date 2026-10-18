"""Pinned publisher warmup with fake numerical components; no ML/network/SDK."""

import ast
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "warm_worker_pure", ROOT / "scripts/jev_decision3_browser_worker.py"
)
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)
PUBLISHER = ROOT / "artifacts/jev-d3-native/20261015-native-source-01/d3/d3_runtime.py"
PUBLISHER_SHA = "2d90e93ac06b9ca4a84e0610ab9acfa2d6793ea4b40eec1b4a3da47577bdea37"


def components(
    tmp_path,
    monkeypatch,
    *,
    fail_gate=None,
    fail_native=None,
    fail_fence=None,
    delay=0,
    owned_attribute=False,
):
    if not PUBLISHER.is_file():
        pytest.skip("Existing pinned publisher source required; never downloads")
    blob = PUBLISHER.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == PUBLISHER_SHA
    tree = ast.parse(blob)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "D3")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "warmup")
    now, events, calls = [100.0], [], []
    namespace = {"Sequence": Sequence, "time": SimpleNamespace(perf_counter=lambda: now[0])}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(PUBLISHER), "exec"), namespace)

    class Runtime:
        warmup = namespace["warmup"]
        fast, batch_size, permutation_average = None, 8, False
        token_ids = list(range(255))

        def probabilities(self, sequences, counts):
            index = len(calls)
            events.append(("native", index))
            calls.append((copy.deepcopy(sequences), copy.deepcopy(counts)))
            now[0] += delay
            if index == fail_native:
                raise ValueError("original-native-failure")
            return [[0.125, 0.875] for _ in sequences]

        def synchronize(self):
            events.append(("publisher-final-fence", None))

    runtime = Runtime()
    if owned_attribute:
        runtime.probabilities = runtime.probabilities
    previous = runtime.__dict__.copy()
    fence_count = [0]

    def fence(_index):
        fence_count[0] += 1
        if fence_count[0] == fail_fence:
            raise RuntimeError("secondary-fence-failure")

    def gate(stage):
        index = int(stage.rsplit("-", 1)[1])
        events.append(("gate", index))
        if index == fail_gate:
            raise ValueError("original-gate-refusal")
        return {"snapshot": {"cuda": {"uuid": "GPU-fake"}, "actual_fake_fixture": True}}

    real_pin = WORKER.pin
    source_pin = real_pin(PUBLISHER)

    def fixture_pin(path):
        # Only the explicitly fake probabilities implementation has a test-file
        # origin; journal hashes and the actual publisher source remain real.
        if Path(path) == Path(__file__):
            return source_pin
        return real_pin(path)

    monkeypatch.setattr(WORKER, "pin", fixture_pin)
    state = {}
    args = (
        runtime,
        SimpleNamespace(version=SimpleNamespace(hip=None), cuda=SimpleNamespace(synchronize=fence)),
        SimpleNamespace(check=gate),
        tmp_path,
        state,
        "GPU-fake",
        source_pin,
    )
    return args, state, runtime, calls, events, previous, lambda: now[0]


@pytest.mark.parametrize("owned_attribute", [False, True])
def test_real_pinned_publisher_loop_has48_gated_raw_calls_and_exact_restoration(
    tmp_path, monkeypatch, owned_attribute
):
    args, state, runtime, calls, events, previous, clock = components(
        tmp_path, monkeypatch, owned_attribute=owned_attribute, delay=22
    )
    summary = WORKER.publisher_warmup(*args, clock=clock)
    assert len(calls) == 48 and runtime.__dict__ == previous
    assert [v for v in events if v[0] in {"gate", "native"}] == [
        v for i in range(48) for v in (("gate", i), ("native", i))
    ]
    assert sum(len(seqs) for seqs, _ in calls) == 216
    assert sum(sum(map(len, seqs)) for seqs, _ in calls) == 96544
    assert state["warmup_calls"] == 48 and state["publisher_warmup_invocations"] == 1
    assert summary["elapsed_seconds"] == 48 * 22  # Startup is distinct from task20s.
    raw = json.loads((tmp_path / "warmup.json").read_bytes())
    assert raw["stage"] == "complete" and len(raw["records"]) == 48
    assert raw["records"][0]["raw_probabilities"] == [[0.125, 0.875]]
    assert raw["records"][-1]["sequences"] == calls[-1][0]
    assert raw["records"][-1]["counts"] == [2] * 8
    assert all(r["cuda_synchronized_elapsed_seconds"] == 22 for r in raw["records"])
    assert summary["journal_pin"] == WORKER.pin(tmp_path / "warmup.json")


@pytest.mark.parametrize("fault", ["gate", "native", "post-return-fence"])
def test_partial_warmup_retains_original_failure_raw_and_real_partial_counters(
    tmp_path, monkeypatch, fault
):
    args, state, runtime, calls, _, previous, clock = components(
        tmp_path,
        monkeypatch,
        fail_gate=2 if fault == "gate" else None,
        fail_native=2 if fault == "native" else None,
        fail_fence=2 if fault == "post-return-fence" else None,
    )
    with pytest.raises((ValueError, RuntimeError)):
        WORKER.publisher_warmup(*args, clock=clock)
    assert runtime.__dict__ == previous
    raw = json.loads((tmp_path / "warmup.json").read_bytes())
    assert raw["stage"] == "failed" and raw["error"] is not None
    if fault == "gate":
        assert len(calls) == 2 and raw["attempted_probability_calls"] == 3
        assert raw["entered_probability_calls"] == raw["completed_probability_calls"] == 2
    elif fault == "native":
        assert len(calls) == 3 and raw["entered_probability_calls"] == 3
        assert raw["completed_probability_calls"] == 2
    else:
        assert len(calls) == raw["completed_probability_calls"] == 1
        assert raw["records"][0]["raw_returned"] is True
        assert raw["records"][0]["raw_probabilities"] == [[0.125, 0.875]]


def test_actual_publisher_origin_mismatch_refuses_before_any_call(tmp_path, monkeypatch):
    args, state, _, calls, events, _, clock = components(tmp_path, monkeypatch)
    bad = list(args)
    bad[-1] = {"sha256": "0" * 64, "size_bytes": 1}
    with pytest.raises(ValueError, match="source differs"):
        WORKER.publisher_warmup(*bad, clock=clock)
    assert calls == events == [] and state == {}


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_browser_envelopes", True),
        ("native_batch_size", 8.0),
        ("images", 0),
        ("max_browser_envelopes", 160),
        ("lengths", [37.0, 64, 320, 333, 1000, 1024]),
    ],
)
def test_startup_config_is_exact_typed_and_cannot_raise_workload_budget(field, value):
    config = WORKER.startup_configuration("publisher-text48-v1", "public-six")
    config[field] = value
    with pytest.raises(ValueError):
        WORKER.validate_startup(config)


@pytest.mark.parametrize("workload,limit", [("public-six", 96), ("known-four", 64)])
def test_worker_gate_counts_every_nonclose_frame_and_keeps_raw_excess_before_decode(
    tmp_path, monkeypatch, workload, limit
):
    startup = WORKER.startup_configuration("publisher-text48-v1", workload)
    sent, predictions, persisted = [], [], []
    # The extra input is deliberately invalid JSON. The envelope limit must run
    # before decoding it and retain its exact bytes without another prediction.
    extra = b"not-json-budget-refusal\n"
    source = io.BytesIO(b"{}\n" * limit + extra)
    torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda _index: None))
    backend = SimpleNamespace(close=lambda: None)

    def initialize(_config, state, _output):
        state.update(
            resources_after_load={},
            resources_after_warmup={},
            runtime={},
            startup_timing={},
            warmup={},
            cache_before={},
            cache_post_warmup={},
            post_warmup_checks={},
        )
        return SimpleNamespace(backend=backend), None, torch, None, None, []

    def process(*_args, **kwargs):
        predictions.append(kwargs)
        return {
            "success": True,
            "campaign_stop": False,
            "journal": {
                "request_id": str(len(predictions)),
                "http_sha256": "a" * 64,
                "elapsed_seconds": 0,
                "journal_path": str(tmp_path / "one.json"),
            },
        }

    monkeypatch.setattr(WORKER, "validate_output", lambda _path: tmp_path)
    monkeypatch.setattr(WORKER, "initialize", initialize)
    monkeypatch.setattr(WORKER, "process_request", process)
    monkeypatch.setattr(
        WORKER, "persist", lambda path, value, **_kw: persisted.append(copy.deepcopy(value))
    )
    monkeypatch.setattr(WORKER, "send", lambda _control, value: sent.append(copy.deepcopy(value)))
    monkeypatch.setattr(WORKER, "authenticate_sources", lambda _manifest: {})
    monkeypatch.setattr(WORKER, "sys", SimpleNamespace(stdin=SimpleNamespace(buffer=source)))
    monkeypatch.setattr(
        WORKER,
        "CORE",
        SimpleNamespace(
            pin=lambda _p: {"sha256": "b" * 64, "size_bytes": 1},
            resource_snapshot=lambda _t: {"cuda": {"uuid": "GPU-fake"}},
        ),
    )
    config = {
        "model": "d3",
        "alias": "d3",
        "output_dir": str(tmp_path),
        "nonce": "nonce",
        "expected_boot_id": "boot",
        "expected_memtotal": 1,
        "expected_gpu_uuid": "GPU-fake",
        "source_manifest": {},
        "startup": startup,
    }
    assert WORKER.run(config, None) == 1 and len(predictions) == limit
    terminal = sent[-1]["receipt"]
    assert terminal["browser_envelopes_seen"] == limit + 1
    assert terminal["envelope_limit_refusal"]["raw_envelope_hex"] == extra.hex()
    assert terminal["envelope_limit_refusal"]["sha256"] == hashlib.sha256(extra).hexdigest()
    assert terminal["stage"] == "failed" and sent[-1]["success"] is False
    assert all(p["startup"] == startup and type(p["started"]) is float for p in predictions)
    assert any("envelope_limit_refusal" in record for record in persisted)


@pytest.mark.parametrize("late", [False, True])
def test_warm_request_counts_actual_api_entries_and_restores_methods_even_when_late(tmp_path, late):
    now, fences = [100.0], []

    class Runtime:
        def probabilities(self, sequences, counts):
            return [[0.125, 0.875]]

    runtime = Runtime()

    class Model:
        def system_one(self):
            result = runtime.probabilities([[1, 2]], [2])
            if late:
                now[0] += 21
            return {"actual_fake_return": result}

    model = Model()

    class Backend:
        def predict(self, _request, *, deadline, evidence_sink):
            deadline()
            raw = self.model.system_one()
            evidence_sink("native_raw_returned", {"raw_response": raw})
            deadline()
            return raw

    backend = Backend()
    backend.model, backend.runtime = model, runtime

    class Adapter:
        evidence_sink = None

        def predict(self, _payload):
            return self.backend.predict(
                None,
                deadline=lambda: None,
                evidence_sink=lambda kind, data: self.evidence_sink({"event": kind, "data": data}),
            )

    adapter = Adapter()
    adapter.backend = backend
    before = (model.__dict__.copy(), runtime.__dict__.copy())
    body = WORKER.encode({"model": "d3"})
    result = WORKER.process_request(
        WORKER.encode(
            {
                "request_id": "real-api-counter",
                "payload_json": body.decode(),
                "http_sha256": hashlib.sha256(body).hexdigest(),
                "remaining_seconds": 20,
            }
        ),
        adapter=adapter,
        wire=SimpleNamespace(named_browser_response=lambda _payload, raw: raw),
        torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=fences.append)),
        gates=SimpleNamespace(check=lambda _stage: None),
        output=tmp_path,
        model="d3",
        nonce="nonce",
        gpu_uuid="GPU-fake",
        startup=WORKER.startup_configuration("publisher-text48-v1", "public-six"),
        clock=lambda: now[0],
    )
    assert result["success"] is (not late) and adapter.backend is backend
    assert (model.__dict__, runtime.__dict__) == before
    row = result["journal"]
    assert row["task_native_calls"] == {
        "backend_predict_entries": 1,
        "system_one_entries": 1,
        "probabilities_entries": 1,
        "sequence_rows": 1,
        "input_tokens": 2,
        "scope": "actual publisher API entries; not GPU kernel launch counts",
    }
    assert row["raw_response"] == {"actual_fake_return": [[0.125, 0.875]]}
    if late:
        assert result["wire_body_utf8"] is None


def test_failed_real_warmup_never_emits_ready_or_falls_back_to_cold(tmp_path, monkeypatch):
    args, _, _, _, _, _, clock = components(tmp_path, monkeypatch, fail_native=2)
    sent, closes = [], []
    torch = args[1]
    backend = SimpleNamespace(close=lambda: closes.append(True))

    def initialize(config, state, output):
        state.update(_torch=torch, _backend=backend, stage="publisher_warmup")
        changed = list(args)
        changed[3], changed[4] = output, state
        WORKER.publisher_warmup(*changed, clock=clock)
        raise AssertionError("Failed warmup must not return an adapter")

    monkeypatch.setattr(WORKER, "validate_output", lambda _path: tmp_path)
    monkeypatch.setattr(WORKER, "initialize", initialize)
    monkeypatch.setattr(WORKER, "authenticate_sources", lambda _manifest: {})
    monkeypatch.setattr(WORKER, "send", lambda _control, value: sent.append(copy.deepcopy(value)))
    monkeypatch.setattr(
        WORKER,
        "CORE",
        SimpleNamespace(
            resource_snapshot=lambda _t: {"cuda": {"uuid": "GPU-fake"}},
            write=lambda path, value, **_kw: Path(path).write_bytes(WORKER.encode(value)),
        ),
    )
    config = {
        "model": "d3",
        "alias": "d3",
        "output_dir": str(tmp_path),
        "nonce": "nonce",
        "expected_boot_id": "boot",
        "expected_memtotal": 1,
        "expected_gpu_uuid": "GPU-fake",
        "source_manifest": {},
        "startup": WORKER.startup_configuration("publisher-text48-v1", "public-six"),
    }
    assert WORKER.run(config, None) == 1 and closes == [True]
    assert [row["event"] for row in sent] == ["closed"]
    receipt = sent[0]["receipt"]
    assert (
        receipt["profile"] == WORKER.WARM_PROFILE
        and receipt["error"]["message"] == "original-native-failure"
    )
    assert receipt["warmup_calls"] == 3 and receipt["warmup"]["completed_probability_calls"] == 2
