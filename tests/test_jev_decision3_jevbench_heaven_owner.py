"""Synthetic Heaven-owner contract tests; no GPU, private data or network."""

import copy
import hashlib
import inspect
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import jev_decision3_jevbench_heaven_owner as owner
import jev_decision3_link_navigation_features as features
import pytest
import run_jevbench_public as heaven


class FakeTensor:
    def __init__(self, raw):
        self.raw = raw

    def detach(self):
        return self

    def contiguous(self):
        return self

    def numpy(self):
        return self

    def tobytes(self):
        return self.raw


class FakeLease:
    def __init__(self, *_args, **_kwargs):
        self.fd = self.held = self.fatal = None


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.setattr(owner, "SOURCE_ONLY", False)
    monkeypatch.setattr(owner.common, "_shared_identity", lambda *_: None)
    monkeypatch.setattr(owner.common, "_verify_source_pins", lambda *_: None)
    monkeypatch.setattr(features, "_validate_manifest", lambda *_: None)
    monkeypatch.setattr(owner, "_runtime_guard", lambda *_: None)
    monkeypatch.setattr(owner, "_model_stat_snapshot", lambda *_: ("synthetic",))
    monkeypatch.setattr(
        owner.child, "_lease_summary", lambda _transport, _pid, n: {"own_completed_pairs": n}
    )
    deltas = {key: FakeTensor(key.encode()) for key in ("operation", "click_target")}
    monkeypatch.setattr(owner.common, "_load_deltas", lambda *_: dict(deltas))
    plan = {
        "unit": "jev-heaven-" + "a" * 32 + ".service",
        "source_pins": {},
        "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
        "cache_root": str(tmp_path / "model"),
        "tasks_root": str(tmp_path),
        "request_seconds": 180,
        "deltas": {
            key: {"fp32_sha256": hashlib.sha256(key.encode()).hexdigest()} for key in deltas
        },
        "transport": {"lock": "x", "journal": "y", "stat": {}, "nonce": "a" * 32},
    }
    backend = SimpleNamespace(
        spec=SimpleNamespace(model_id="model/edge", revision="a" * 40), runtime=object()
    )
    tasks = [
        SimpleNamespace(
            id=f"task-{i}",
            split="public",
            family="fake",
            group=None,
            state=f"state-{i}",
            question={"type": "choice"},
            labels=["a", "b"],
        )
        for i in range(231)
    ]
    scoring = SimpleNamespace(
        score_task=lambda *_: {
            "valid": True,
            "strict_valid": True,
            "renormalized": False,
            "correct": True,
            "predicted": "a",
            "probs": {"a": 0.6, "b": 0.4},
        }
    )
    official = SimpleNamespace(
        base=SimpleNamespace(
            build_question=lambda _task: {
                "type": "choice",
                "instructions": "select",
                "criteria": {"a": "A", "b": "B"},
            }
        ),
        scoring=scoring,
        summarize=SimpleNamespace(
            summarize=lambda _tasks, records: {"n_valid": sum(row["valid"] for row in records)}
        ),
    )
    monkeypatch.setattr(
        heaven,
        "load_public",
        lambda *_: (
            tasks,
            official,
            {
                "source_revision": "test",
                "dataset_label": "synthetic",
                "public_exposure": "synthetic",
            },
        ),
    )
    settings = SimpleNamespace(
        fail_at=None, invalid_at=None, alter_response_at=None, leave_held=False
    )

    class Generic:
        def __init__(self, backend, *, lease, identity_guard, event_sink, **_kwargs):
            self.backend, self.lease, self.guard, self.sink, self.calls = (
                backend,
                lease,
                identity_guard,
                event_sink,
                0,
            )

        def predict(self, state, questions):
            self.calls += 1
            req = {"state": state, "questions": questions}
            sha = owner.child._sha(owner.child._encode(req))
            row = {
                "route": "jevbench-generic-native-v1",
                "call": self.calls,
                "request_sha256": sha,
                "delta_applied": [],
                "delta_bypassed": ["operation", "click_target"],
            }
            self.guard(self.backend, self.lease)
            self.sink({**row, "event": "planned", "request": copy.deepcopy(req)})
            if self.calls == settings.fail_at:
                self.sink({**row, "event": "failed"})
                if settings.leave_held:
                    self.lease.held = object()
                raise RuntimeError("synthetic provider failure")
            self.sink({**row, "event": "native_preflight"})
            self.sink({**row, "event": "native_raw_returned"})
            result = {
                "answers": {
                    "decision": {
                        "type": "choice",
                        "choice": "a",
                        "probabilities": {"a": 0.6, "b": 0.4},
                    }
                }
            }
            self.sink(
                {
                    **row,
                    "event": "prepared_not_delivered",
                    "response": copy.deepcopy(result),
                    "response_sha256": owner.child._sha(owner.child._encode(result)),
                }
            )
            if self.calls == settings.alter_response_at:
                result["answers"]["decision"]["choice"] = "b"
            return result

    return SimpleNamespace(
        plan=plan,
        backend=backend,
        generic=SimpleNamespace(GenericJevBenchAdapter=Generic),
        lease=SimpleNamespace(ForwardLease=FakeLease),
        output=tmp_path,
        settings=settings,
        scoring=scoring,
    )


def run(env):
    return owner._run_heaven(
        env.plan, env.backend, object(), env.generic, env.lease, env.output, lambda: None
    )


def test_full_231_calls_are_bound_to_official_records(environment):
    result = run(environment)
    assert result["status"] == "completed_pending_external_audit"
    assert result["score_accepted"] is False
    assert result["calls"] == result["lease"]["own_completed_pairs"] == 231
    assert result["official_summary"]["n_valid"] == 231
    assert result["delta_route"] == "both_bypassed"
    assert (
        len(
            (environment.output / "staging" / "public231" / "records.jsonl")
            .read_bytes()
            .splitlines()
        )
        == 231
    )
    assert len((environment.output / "generic-events.jsonl").read_bytes().splitlines()) == 924
    with pytest.raises(FileExistsError):
        run(environment)


def test_actual_reviewed_generic_release_with_heaven_runner(environment, tmp_path):
    from test_jev_decision3_jevbench_generic import FakeModel, FakeRuntime

    from s1.evaluation.decision3 import Decision3Backend

    held = Path(owner.__file__).parent / "jev_decision3_jevbench_generic.py"
    held_bytes = held.read_bytes()
    assert held_bytes.count(b"SOURCE_ONLY = True") == 1
    release = tmp_path / "synthetic-generic-release.py"
    release.write_bytes(held_bytes.replace(b"SOURCE_ONLY = True", b"SOURCE_ONLY = False", 1))
    pins = {
        str(held): hashlib.sha256(held_bytes).hexdigest(),
        str(release): hashlib.sha256(release.read_bytes()).hexdigest(),
    }
    module_name = "_synthetic_heaven_generic_release"
    generic = owner.common._load_reviewed_release(str(release), held.name, pins, module_name)

    class Serial:
        def __init__(self, backend, lease, *, torch):
            self.backend, self.lease = backend, lease

        def predict(self, request, **kwargs):
            return self.backend.predict(request, **kwargs)

    backend = Decision3Backend.__new__(Decision3Backend)
    backend.spec = SimpleNamespace(
        adapter="decision3", context_limit=16384, model_id="model/edge", revision="a" * 40
    )
    backend.identity = {"model_id": "model/edge"}
    backend.runtime, backend.model = FakeRuntime(), FakeModel()
    environment.backend = backend
    environment.generic = generic
    environment.lease = SimpleNamespace(
        SOURCE_ONLY=False, ForwardLease=FakeLease, SerialBackend=Serial
    )
    try:
        result = run(environment)
        assert result["calls"] == 231
        assert result["score_accepted"] is False
        assert result["event_counts"]["failed"] == 0
    finally:
        del sys.modules[module_name]


def test_provider_failure_stops_whole_run(environment):
    environment.settings.fail_at = 3
    with pytest.raises(owner.EvaluationAbort, match="entire evaluation invalid"):
        run(environment)
    assert not (environment.output / "staging" / "public231" / "summary.json").exists()
    assert (
        len(
            (environment.output / "staging" / "public231" / "records.jsonl")
            .read_bytes()
            .splitlines()
        )
        == 2
    )


def test_scorer_invalidates_whole_run_even_if_runner_continues(environment):
    environment.scoring.score_task = lambda *_: {
        "valid": False,
        "strict_valid": False,
        "renormalized": False,
        "correct": False,
        "predicted": None,
    }
    with pytest.raises(ValueError, match="full scorer completion"):
        run(environment)
    _assert_scorer_quarantined(environment, "completed_with_errors")


def _assert_scorer_quarantined(environment, original_status):
    stage = environment.output / "staging"
    assert json.loads((stage / "status.json").read_text())["score_accepted"] is False
    scorer = stage / "public231"
    assert json.loads((scorer / "STATUS_NOT_ACCEPTED.json").read_text())["score_accepted"] is False
    assert json.loads((scorer / "manifest.json").read_text())["status"] == original_status
    assert (scorer / "summary.json").is_file()
    assert len((scorer / "records.jsonl").read_bytes().splitlines()) == 231


def test_post_scorer_evidence_failure_leaves_trace_unaccepted(environment, monkeypatch):
    def fail(*_args):
        raise ValueError("synthetic lease audit failure")

    monkeypatch.setattr(owner.child, "_lease_summary", fail)
    with pytest.raises(ValueError, match="lease audit failure"):
        run(environment)
    _assert_scorer_quarantined(environment, "completed")


def test_full_model_hash_only_before_and_after_evaluation(environment, monkeypatch):
    calls = []
    monkeypatch.setattr(features, "_validate_manifest", lambda *_: calls.append("full_hash"))
    run(environment)
    assert calls == ["full_hash", "full_hash"]


def test_model_file_metadata_snapshot_detects_identity_change(tmp_path):
    model = tmp_path / "snapshot"
    model.mkdir()
    (model / "MODEL_MANIFEST.json").write_text("manifest")
    weights = model / "weights.bin"
    weights.write_bytes(b"weights")
    runtime = SimpleNamespace(
        root=str(model), manifest={"files_sha256": {"weights.bin": "ignored", "NOTICE": "ignored"}}
    )
    before = owner._model_stat_snapshot(runtime, str(tmp_path))
    weights.unlink()
    weights.write_bytes(b"weights")
    after = owner._model_stat_snapshot(runtime, str(tmp_path))
    assert before != after


@pytest.mark.parametrize("failure", ["close", "cuda"])
def test_cleanup_failure_terminal_omits_score_and_keeps_unaccepted_trace(
    environment, monkeypatch, failure
):
    import s1.evaluation.decision3 as d3

    output = environment.output
    plan = {
        **environment.plan,
        "wall_seconds": 30,
        "output_dir": str(output),
        "modules_cache": str(output / "modules"),
        "generic_release": str(output / "generic"),
        "transport": {**environment.plan["transport"], "lease_release": str(output / "lease")},
    }
    plan["source_pins"] = {
        str(Path(path).absolute()): "a" * 64
        for path in (owner.__file__, owner.child.__file__, owner.common.__file__)
    }
    monkeypatch.setenv("INVOCATION_ID", "a" * 32)
    monkeypatch.setattr(owner, "_load_plan", lambda *_: plan)
    monkeypatch.setattr(owner.common, "_own_cgroup", lambda *_: None)
    monkeypatch.setattr(owner.common, "_validate_interpreter", lambda *_: None)
    monkeypatch.setattr(owner.common, "_bind_project_source", lambda *_: None)
    monkeypatch.setattr(owner.common, "_load_reviewed_release", lambda *_: object())
    monkeypatch.setattr(owner, "_actual_sources", lambda *_: None)
    monkeypatch.setattr(owner.child, "_control", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(owner.common.systemd, "boot_id", lambda: "synthetic")
    monkeypatch.setattr(owner.common.systemd, "process_identity", lambda _pid: {"startticks": 1})
    original_write = owner.common._write_new
    original_publish = owner._atomic_publish

    def publish(path, data):
        original_publish(path, data)
        if path.name == "hello.json":
            original_write(output / "go.json", b"{}")

    monkeypatch.setattr(owner, "_atomic_publish", publish)
    calls = []

    def synchronize(_device):
        calls.append("synchronize")
        if failure == "cuda" and len(calls) > 1:
            raise RuntimeError("synthetic CUDA cleanup failure")

    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_properties=lambda _device: SimpleNamespace(
            uuid=plan["gpu_uuid"], major=12, minor=1
        ),
        is_bf16_supported=lambda **_kwargs: True,
        synchronize=synchronize,
        memory_allocated=lambda _device: 0,
        empty_cache=lambda: None,
    )
    torch = SimpleNamespace(
        cuda=cuda,
        set_num_threads=lambda *_: None,
        set_num_interop_threads=lambda *_: None,
        set_float32_matmul_precision=lambda *_: None,
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False))),
        _C=SimpleNamespace(),
    )
    monkeypatch.setattr(owner.common, "_bind_overlay", lambda *_: torch)

    class Provider:
        runtime = object()

        def close(self):
            if failure == "close":
                raise RuntimeError("synthetic close failure")

    monkeypatch.setattr(d3, "Decision3Backend", lambda *_: Provider())
    monkeypatch.setattr(d3, "decision3_spec", lambda *_args, **_kwargs: object())

    def completed(*_args):
        stage = output / "staging"
        stage.mkdir()
        original_write(stage / "status.json", b'{"score_accepted":false}')
        scorer = stage / "public231"
        scorer.mkdir()
        original_write(scorer / "summary.json", b'{"score":1}')
        original_write(scorer / "STATUS_NOT_ACCEPTED.json", b'{"score_accepted":false}')
        return {
            "status": "completed_pending_external_audit",
            "official_summary": {"score": 1},
            "score_accepted": False,
        }

    monkeypatch.setattr(owner, "_run_heaven", completed)
    with pytest.raises(RuntimeError, match="synthetic"):
        owner.run("unused", "b" * 64)
    terminal = json.loads((output / "terminal.json").read_text())
    assert terminal["status"] == "failed_cleanup_unconfirmed"
    assert "evaluation" not in terminal
    assert terminal["model_score"] is None
    assert json.loads((output / "staging" / "status.json").read_text())["score_accepted"] is False
    assert (output / "staging" / "public231" / "summary.json").exists()


def test_atomic_control_never_exposes_partial_bytes(tmp_path, monkeypatch):
    started, resume = threading.Event(), threading.Event()
    original = owner.child._write_control_bytes

    def delayed(stream, raw):
        original(stream, raw)
        started.set()
        assert resume.wait(timeout=5)

    monkeypatch.setattr(owner.child, "_write_control_bytes", delayed)
    target = tmp_path / "hello.json"
    errors = []

    def publish():
        try:
            owner._atomic_publish(target, b'{"complete":true}')
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=publish)
    worker.start()
    try:
        assert started.wait(timeout=5)
        assert not target.exists()
    finally:
        resume.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert not errors
    assert json.loads(target.read_bytes()) == {"complete": True}
    with pytest.raises(FileExistsError):
        owner._atomic_publish(target, b"replacement")
    assert json.loads(target.read_bytes()) == {"complete": True}


def test_response_changed_after_prepared_event_aborts(environment):
    environment.settings.alter_response_at = 1
    with pytest.raises(owner.EvaluationAbort, match="entire evaluation invalid"):
        run(environment)


def test_failure_with_held_lease_marks_cleanup_unconfirmed(environment):
    environment.settings.fail_at = 1
    environment.settings.leave_held = True
    with pytest.raises(owner.EvaluationAbort) as caught:
        run(environment)
    assert caught.value.owner_cleanup_unconfirmed is True


def test_gate_and_exact_plan_refuse_bad_controls(environment, monkeypatch):
    monkeypatch.setattr(owner, "SOURCE_ONLY", True)
    with pytest.raises(ValueError, match="independent review"):
        run(environment)
    with pytest.raises(ValueError, match="exact Heaven plan"):
        owner._validate_plan({"schema": owner.SCHEMA})


def test_owner_rejects_any_preexisting_output_before_resource_use(environment, monkeypatch):
    (environment.output / "unrelated-old-file").write_text("old")
    monkeypatch.setattr(owner, "_load_plan", lambda *_: {"output_dir": str(environment.output)})
    monkeypatch.setattr(
        owner.common, "_own_cgroup", lambda *_: pytest.fail("resource check reached")
    )
    with pytest.raises(ValueError, match="fresh empty private output"):
        owner.run("unused", "unused")


def test_runtime_guard_checks_w0_device_and_frozen_parameters(monkeypatch):
    class Param:
        dtype = "bf16"
        device = "cuda:0"
        requires_grad = False

        def numel(self):
            return 594321600

    class Readout(Param):
        dtype = "fp32"
        shape = (255, 1024)

        def numel(self):
            return 255 * 1024

    language = object()
    backbone = SimpleNamespace(
        language_model=language, training=False, parameters=lambda: [Param()]
    )
    runtime = SimpleNamespace(
        model_name="d3-edge",
        batch_size=8,
        temperature=1.0,
        max_length=None,
        attention_mode="noncausal_full_attention",
        permutation_average=False,
        fast=None,
        readout_dtype="float32",
        encoder=SimpleNamespace(language_model=language),
        backbone=backbone,
        readout=Readout(),
    )
    backend = SimpleNamespace(
        spec=object(), runtime=runtime, model=SimpleNamespace(runtime=runtime)
    )
    torch = SimpleNamespace(
        float32="fp32",
        bfloat16="bf16",
        is_tensor=lambda value: isinstance(value, Readout),
        get_float32_matmul_precision=lambda: "highest",
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False))),
        is_autocast_enabled=lambda _device: False,
    )
    monkeypatch.setattr(features, "_validate_provider_spec", lambda _spec: None)
    monkeypatch.setattr(features, "_tensor_sha", lambda _tensor: features.W0_SHA256)
    owner._runtime_guard(backend, torch)
    runtime.readout.device = "cpu"
    with pytest.raises(ValueError, match="pinned native runtime"):
        owner._runtime_guard(backend, torch)


def test_actual_critical_source_and_official_config_pins_required(environment):
    from s1.backends import normalize_response
    from s1.contracts import DecisionRequest, validate_distribution
    from s1.evaluation.decision3 import native_payload, normalize_native_response

    class Provider:
        model = FakeTensor(b"fake")
        runtime = FakeTensor(b"fake")

    environment.backend = Provider()
    files = [
        owner.__file__,
        owner.child.__file__,
        owner.common.__file__,
        features.__file__,
        heaven.__file__,
        heaven.CONFIG,
        inspect.getfile(DecisionRequest),
        inspect.getfile(normalize_response),
        inspect.getfile(validate_distribution),
        inspect.getfile(native_payload),
        inspect.getfile(normalize_native_response),
        inspect.getfile(type(environment.backend)),
        inspect.getfile(type(environment.backend.model)),
        inspect.getfile(type(environment.backend.runtime)),
    ]
    pins = {str(Path(path).absolute()): "a" * 64 for path in files}
    pins[str(heaven.CONFIG.absolute())] = owner.CONFIG_SHA256
    environment.plan["source_pins"] = pins
    owner._actual_sources(environment.plan, environment.backend)
    del pins[str(Path(inspect.getfile(normalize_response)).absolute())]
    with pytest.raises(ValueError, match="sources must be pinned"):
        owner._actual_sources(environment.plan, environment.backend)
    pins[str(Path(inspect.getfile(normalize_response)).absolute())] = "a" * 64
    pins[str(heaven.CONFIG.absolute())] = "b" * 64
    with pytest.raises(ValueError, match="fixed official"):
        owner._actual_sources(environment.plan, environment.backend)
