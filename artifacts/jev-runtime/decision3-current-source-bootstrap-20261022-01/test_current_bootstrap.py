"""CPU current-source bootstrap tests; fake numerical dependencies, no model/runtime actors."""

import ast
import builtins
import copy
import hashlib
import importlib.util
import json
import os
import py_compile
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
PATH = (
    REPO / "artifacts/jev-runtime/decision3-current-source-bootstrap-20261022-01/current_bootstrap.py"
)
SPEC = importlib.util.spec_from_file_location("gx10_current_bootstrap_test_subject", PATH)
B = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(B)
OFFLINE_ENVIRONMENT = B.offline_environment
VERIFY_RUNTIME = B.RuntimeSelection.verify
sys.path.insert(0, str(REPO / "src"))
from s1.evaluation.decision3 import decision3_spec  # noqa: E402
from s1.jev_decision3_native import JevDecision3NativeAdapter  # noqa: E402


@pytest.fixture(autouse=True)
def forbid_frameworks(monkeypatch):
    original = builtins.__import__

    def checked(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"torch", "transformers", "modal", "ctypes"}:
            raise AssertionError("Forbidden numerical/actor dependency: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked)


class Clock:
    now = 100.0

    def __call__(self):
        return self.now


class Tensor:
    device, dtype = "cuda:0", "bf16"

    def __init__(self, count, shape=None, dtype="bf16"):
        self.count, self.shape, self.dtype = count, shape, dtype

    def numel(self):
        return self.count


class Runtime:
    """Explicit fake publisher schedule, not the genuine publisher implementation."""

    fast, batch_size, permutation_average = None, 8, False
    readout_dtype, temperature, max_length = "float32", 1.0, None
    attention_mode = "noncausal_full_attention"
    kernels = {"implementation": "synthetic-torch-reference"}

    def runtime_info(self):
        return {"fake_numerical_dependencies": True}

    token_ids = list(range(255))

    def __init__(self, calls, identity, fail_native=None):
        self.calls, self.fail_native = calls, fail_native
        width = identity["readout_tensor_bytes"] // (255 * 4)
        readout_count = 255 * width
        self.backbone = SimpleNamespace(parameters=lambda: iter([Tensor(identity["loaded_parameters"] - readout_count)]))
        self.readout = Tensor(readout_count, [255, width], "fp32")

    def probabilities(self, sequences, counts):
        index = len(self.calls)
        self.calls.append((copy.deepcopy(sequences), copy.deepcopy(counts)))
        if index == self.fail_native:
            raise ValueError("primary-fake-native")
        return [[0.25, 0.75] for _ in sequences]

    def warmup(self, *, lengths, images):
        assert images is False
        for size in range(1, 9):
            for length in lengths:
                sequences = [
                    [self.token_ids[(i + j) % 255] for j in range(max(8, length - 7 * i))]
                    for i in range(size)
                ]
                self.probabilities(sequences, [2] * size)
        return 0.25

    def provenance(self):
        return {"model_name": "d3-edge", "repo_id": "vllm-sr/d3-edge", "pooling": "last"}


def runtime_blob():
    files = {
        p: {"sha256": "a" * 64, "size_bytes": 1}
        for p in ("/selected/source.py", "/selected/python")
    }
    distributions = {
        name: {
            "version": "5.17.0" if name == "transformers" else "test",
            "metadata_path": "/selected/source.py",
        }
        for name in (
            "torch",
            "torchvision",
            "transformers",
            "tokenizers",
            "safetensors",
            "accelerate",
            "huggingface-hub",
            "numpy",
        )
    }
    return B.encode(
        {
            "schema_id": B.RUNTIME_KIND,
            "machine": "aarch64",
            "interpreter": "/selected/python",
            "python": [3, 13],
            "files": files,
            "distributions": distributions,
            "modules": {
                name: {"path": "/selected/source.py", "kind": "source", "size_bytes": 1}
                for name in ("torch", "torch._C", "transformers")
            },
            "kernel_policy": {
                "fast": False,
                "permutation_average": False,
                "native_batch_size": 8,
                "temperature": 1.0,
                "backbone": "bfloat16",
                "readout": "float32",
                "context": 8192,
            },
        }
    )


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    worker = B.load_worker()
    original_identities = worker.authenticate_sources(
        json.loads(B.checked_blob(B.MANIFEST, B.MANIFEST_SHA))
    )
    events, calls, clock = [], [], Clock()
    uuid, failures = ["GPU-fake"], {}
    monkeypatch.setattr(B, "REPLAY_ROOT", tmp_path)
    output = tmp_path / "arm"
    output.mkdir()
    config = {
        "profile": B.PROFILE,
        "model": "d3-edge",
        "workload": "public-six",
        "nonce": "n-01",
        "expected_boot_id": "boot",
        "expected_memtotal": 1000,
        "expected_gpu_uuid": "GPU-fake",
        "output_dir": str(output),
    }
    helper = {
        "profile": B.HELPER_PROFILE,
        "qualification_sha256": B.HELPER_SHA,
        "source_count": 95,
        "boot_id": "boot",
        "pid": 123,
        "start_ticks": 456,
        "pgid": 123,
        "sid": 123,
        "endpoint": "http://127.0.0.1:12345",
        "seconds": 20,
        "tokens": 128,
        "context": 8192,
    }
    identity = original_identities["d3-edge"]
    identities = original_identities
    report_pin = worker.pin(Path(__file__))

    class Gates:
        torch = None

        def __init__(self, *_args):
            pass

        def check(self, stage):
            events.append(stage)
            if failures.get("gate") == stage:
                raise ValueError("primary-gate-refusal")
            return {"snapshot": {"cuda": {"uuid": uuid[0]}}}

    def persist(path, value, **_kwargs):
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(B.encode(value))

    def cache(_name, _identity, _output, _gates, phase):
        events.append("cache-" + phase)
        if failures.get("cache") == phase:
            raise ValueError("primary-cache-refusal")
        return {"files": {"d3_runtime.py": report_pin}, "fake_cache_no_payload": True}

    core = worker.CORE
    monkeypatch.setattr(core, "Gates", Gates)
    monkeypatch.setattr(core, "HUB", tmp_path / "hub")
    monkeypatch.setattr(core, "write", persist)
    monkeypatch.setattr(core, "authenticate_cache", cache)
    monkeypatch.setattr(core, "resource_snapshot", lambda _torch=None: {"cuda": {"uuid": uuid[0]}})
    actual_auth = worker.authenticate_sources

    def auth(manifest):
        events.append("current18-auth")
        return actual_auth(manifest)

    monkeypatch.setattr(worker, "authenticate_sources", auth)
    monkeypatch.setattr(B, "offline_environment", lambda *_args: events.append("offline") or [])
    monkeypatch.setattr(B.RuntimeSelection, "verify", lambda *_args, **_kwargs: {})

    def helper_check(binding, phase):
        events.append("helper-" + phase)
        return {
            "binding": binding,
            "phase": phase,
            "helper_admitted": True,
            "co_resident_admitted": True,
            "receipt_sha256": "b" * 64,
        }

    backend = SimpleNamespace()
    torch = SimpleNamespace(
        device=lambda name: name,
        bfloat16="bf16",
        float32="fp32",
        version=SimpleNamespace(hip=None),
        cuda=SimpleNamespace(
            is_available=lambda: True, device_count=lambda: 1, synchronize=lambda _index: None
        ),
    )

    class Backend:
        def __init__(self, spec, **kwargs):
            events.append("constructor")
            assert kwargs == {"helper_already_loaded": True, "expected_boot_id": "boot", "expected_memtotal": 1000}
            self.spec, self.name, self.closed, self.predict_calls = spec, "d3-edge", False, 0
            self.runtime = Runtime(calls, identity, failures.get("native"))
            self.model = SimpleNamespace(system_one=self.system_one)
            backend.instance = self
            if failures.get("constructor"):
                raise RuntimeError("primary-constructor-partial")
            if failures.get("provenance"):
                self.runtime.provenance = lambda: {
                    "model_name": "wrong",
                    "repo_id": "wrong",
                    "pooling": "last",
                }
            if failures.get("geometry"):
                self.runtime.readout.shape = [254, 1]

        def system_one(self, request):
            self.runtime.probabilities([[1, 2]], [2])
            answers = {}
            for question in request.questions:
                labels = question.labels()
                probabilities = {label: 1 / len(labels) for label in labels}
                answers[question.id] = {
                    "type": "choice",
                    "choice": labels[0],
                    "confidence": 0.5,
                    "probabilities": probabilities,
                }
            return {
                "model": "d3-edge",
                "answers": answers,
                "usage": {"input_tokens": 123, "output_tokens": 0},
            }

        def predict(self, request, *, deadline, evidence_sink):
            self.predict_calls += 1
            deadline()
            raw = self.model.system_one(request)
            evidence_sink("native_raw_returned", {"raw_response": copy.deepcopy(raw)})
            clock.now += failures.get("raw_delay", 0)
            deadline()
            return raw

        def close(self):
            events.append("backend-close")
            self.closed = True
            if failures.get("close"):
                raise RuntimeError("secondary-close")

    def loader(_selection):
        events.append("dependency-loader")
        return SimpleNamespace(
            torch=torch,
            checkpoints=identities,
            backend=Backend,
            spec=decision3_spec,
            adapter=JevDecision3NativeAdapter,
        )

    inventory = B.source_inventory(worker)
    blob = runtime_blob()
    kwargs = dict(
        inventory=inventory,
        inventory_sha=hashlib.sha256(B.encode(inventory)).hexdigest(),
        runtime_blob=blob,
        runtime_sha=hashlib.sha256(blob).hexdigest(),
        helper=helper,
        helper_check=helper_check,
        worker=worker,
        load_dependencies=loader,
    )
    return SimpleNamespace(
        worker=worker,
        config=config,
        helper=helper,
        kwargs=kwargs,
        events=events,
        calls=calls,
        failures=failures,
        uuid=uuid,
        backend=backend,
        torch=torch,
        clock=clock,
        output=output,
        core=core,
    )


def open_components(f):
    return B.bootstrap(f.config, **f.kwargs)


def request_candidate(components, envelope_bytes, clock):
    # Exercise the actual retained helper, not a new transport. The future
    # supervisor still owns ingress/queue/envelope cap/helper freshness/delivery.
    result = components.worker.process_request(
        envelope_bytes,
        adapter=components.adapter,
        wire=components.wire,
        torch=components.torch,
        gates=components.gates,
        output=components.output,
        model=components.config["model"],
        nonce=components.config["nonce"],
        gpu_uuid=components.config["expected_gpu_uuid"],
        startup=components.startup,
        started=clock(),
        clock=clock,
    )
    return {"transport_ready": False, "candidate": result}


def envelope(f, *, remaining=20, width=2, request_id="req-1"):
    payload = {
        "model": "d3-edge",
        "state": {"z": "順序", "a": [3, 1]},
        "questions": {
            "operation": {
                "type": "choice",
                "criteria": {"CLICK": "Click", "DONE": "Done", "BLOCKED": "Blocked"},
                "instructions": {"z": "first", "a": "second"},
            },
            "click_target": {
                "type": "choice",
                "criteria": {str(i): str(i) for i in range(width)},
                "instructions": ["ordered", "full head"],
            },
        },
    }
    body = B.encode(payload)
    return B.encode(
        {
            "request_id": request_id,
            "payload_json": body.decode(),
            "http_sha256": hashlib.sha256(body).hexdigest(),
            "remaining_seconds": remaining,
        }
    )


def test_cli_is_dry_and_run_is_always_refused_before_loading(monkeypatch, capsys):
    monkeypatch.setattr(B, "load_worker", lambda: pytest.fail("CLI must not dynamically load"))
    for argv, code, mode in (
        ([], 0, "DRY_SOURCE_ONLY"),
        (["--run"], 2, "ACTUAL_REFUSED_BEFORE_DYNAMIC_IMPORT"),
    ):
        assert B.main(argv) == code
        report = json.loads(capsys.readouterr().out)
        assert report["mode"] == mode and report["actual_started"] is False
        assert report["transport_ready"] is False and report["qualified"] is False
        assert report["full_scope_models"] == list(B.MODELS)
    with pytest.raises(SystemExit):
        B.main(["--run", "--config", "seemingly-valid.json"])


def test_original_projection_is_exact_and_new_count_is_mechanical(fixture):
    f = fixture
    inventory = f.kwargs["inventory"]
    original = json.loads(B.checked_blob(B.MANIFEST, B.MANIFEST_SHA))
    assert inventory["selected_projection"] == original
    assert inventory["rows"][:-1] == original["rows"]
    assert inventory["files"] == len(inventory["rows"]) == len(original["rows"]) + 1
    assert inventory["rows"][-1]["path"] == str(B.SELF)
    assert B.load_worker().pin(B.WORKER)["sha256"] == B.WORKER_SHA
    historical = json.loads(B.checked_blob(B.HISTORICAL_MANIFEST, B.HISTORICAL_SHA))
    assert inventory["historical_source17_projection"] == historical
    assert historical["files"] == 17 and original["files"] == 18 and inventory["files"] == 19
    assert len(f.worker.CORE.SOURCE_PINS) == 13


@pytest.mark.parametrize("model", B.MODELS)
def test_all_six_have_explicit_current_configuration_without_loading(fixture, model):
    f = fixture
    selected = {**f.config, "model": model}
    assert B.validate_config(selected, f.worker)["max_browser_envelopes"] == 96
    assert not f.events and not f.calls


def test_selected_source19_rejects_drift_even_with_rehashed_claim(fixture):
    f = fixture
    value = copy.deepcopy(f.kwargs["inventory"])
    value["rows"][-1]["pin"]["sha256"] = "0" * 64
    f.kwargs.update(inventory=value, inventory_sha=hashlib.sha256(B.encode(value)).hexdigest())
    with pytest.raises(ValueError, match="source inventory"):
        open_components(f)
    assert not f.events and not f.calls


def test_real_current_source_verifier_rejects_old_helper_pin_and_unselected_paths():
    worker = B.load_worker()
    manifest = json.loads(B.checked_blob(B.MANIFEST, B.MANIFEST_SHA))
    identities = worker.authenticate_sources(manifest)
    assert tuple(identities) == B.MODELS
    row = next(r for r in manifest["rows"] if r["path"].endswith("/src/s1/jev_generated.py"))
    assert row["pin"] == {
        "sha256": "c356e0eb3c046cb0787d7be0c5960803006231e8bdb414de644ae2167e72d849",
        "size_bytes": 23569,
    }
    changed = copy.deepcopy(manifest)
    target = next(r for r in changed["rows"] if r["path"] == row["path"])
    target["pin"]["sha256"] = "582b520615cf123367436622f4e5ca6f5d94abb227449d866ec64cbb8adc9b3c"
    with pytest.raises(ValueError, match="source pin"):
        worker.authenticate_sources(changed)
    target["path"] = "/tmp/unselected-source.py"
    with pytest.raises(ValueError, match="Source path differs"):
        worker.authenticate_sources(changed)


def test_core_fixed_source_policy_rejects_changed_copy(tmp_path, monkeypatch):
    worker = B.load_worker()
    worker.authenticate_sources(json.loads(B.checked_blob(B.MANIFEST, B.MANIFEST_SHA)))
    core = worker.CORE
    for label in core.SOURCE_PINS:
        target = tmp_path / label
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO / label).read_bytes())
    monkeypatch.setattr(core, "ROOT", tmp_path)
    assert tuple(core.authenticate_source(worker.CORE_SHA)) == B.MODELS
    with (tmp_path / "src/s1/jev_generated.py").open("ab") as stream:
        stream.write(b"\n# unselected drift\n")
    with pytest.raises(RuntimeError, match="production source differs"):
        core.authenticate_source(worker.CORE_SHA)


def test_full_d3_unchanged_real_host_gate_refuses_before_cache_framework_or_helper(
    fixture, tmp_path, monkeypatch
):
    f = fixture
    worker = B.load_worker()
    worker.authenticate_sources(json.loads(B.checked_blob(B.MANIFEST, B.MANIFEST_SHA)))
    snapshot = worker.CORE.resource_snapshot()
    total = snapshot["proc_meminfo"]["MemTotal"]
    floor = max(24 * 1024**3, (total + 4) // 5) + 2 * 52176025120 + 2 * 1024**3
    assert floor > total
    output = tmp_path / "full-d3-refusal"
    output.mkdir()
    monkeypatch.setattr(B, "REPLAY_ROOT", tmp_path)
    config = {**f.config, "model": "d3", "expected_boot_id": snapshot["boot_id"],
              "expected_memtotal": total, "output_dir": str(output)}
    helper = {**f.helper, "boot_id": snapshot["boot_id"]}
    inventory = B.source_inventory(worker)
    kwargs = {**f.kwargs, "worker": worker, "inventory": inventory,
              "inventory_sha": hashlib.sha256(B.encode(inventory)).hexdigest(), "helper": helper,
              "helper_check": lambda *_args: pytest.fail("No helper before rejected fullD3 gate"),
              "load_dependencies": lambda *_args: pytest.fail("No framework before fullD3 gate")}
    with pytest.raises(worker.CORE.AdmissionError) as raised:
        B.bootstrap(config, **kwargs)
    decision = raised.value.resource_gate["decision"]
    assert decision["required_available_bytes"] == floor
    assert decision["swap_or_resident_credit_bytes"] == 0 and decision["admitted"] is False
    report = json.loads((output / "bootstrap.json").read_bytes())
    assert report["component_complete"] is False and report["qualified"] is False
    assert not f.calls


@pytest.mark.parametrize("width", [1, 52, 255, 400])
def test_current_and_exact_frozen_request_helper_preserve_same_bytes_and_labels(fixture, width):
    from s1.jev_generated import _request

    path = PATH.parent / "frozen-generated-reference.txt"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "582b520615cf123367436622f4e5ca6f5d94abb227449d866ec64cbb8adc9b3c"
    )
    loader = SourceFileLoader("s1._d3_frozen_generated_reference", str(path))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    historical = importlib.util.module_from_spec(spec)
    loader.exec_module(historical)
    payload = json.loads(json.loads(envelope(fixture, width=width))["payload_json"])
    assert _request(payload, "d3-edge") == historical._request(payload, "d3-edge")
    result = _request(payload, "d3-edge")
    assert result[1] == B.encode(payload)
    assert len(result[2]["click_target"]) == width


def test_complete255_native_roundtrip_keeps_all_labels_and_no_authority(fixture):
    f = fixture
    components = open_components(f)
    result = request_candidate(components, envelope(f, width=255), f.clock)["candidate"]
    assert result["success"] is True
    wire = json.loads(result["wire_body_utf8"])
    assert list(wire["answers"]) == ["operation", "click_target"]
    assert list(wire["answers"]["click_target"]["probabilities"]) == [str(i) for i in range(255)]
    assert components.state["qualified"] is False and components.state["transport_ready"] is False
    assert components.state["continuation_authorized"] is False
    closed = components.close()
    assert closed["continuation_authorized"] is False and closed["whole_tree_exit_confirmed"] is False


def test_core_memory_cache_request_text48_function_bodies_preserved():
    paths = [
        (REPO / "scripts/jev_decision3_public_dev22_gpu_replay.py", PATH.parent / "current_core.py",
         ["memory_decision", "Gates", "authenticate_cache", "tensor_header", "backbone_layout"]),
        (REPO / "artifacts/jev-runtime/decision3-matched-v7-warm-readiness-20261018-01/source-candidate-01/jev_decision3_browser_worker.py",
         PATH.parent / "current_worker.py",
         ["publisher_warmup", "process_request", "DeadlineBackend", "matched_context_spec"]),
    ]
    for original, current, names in paths:
        old = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(original.read_bytes()).body
               if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        new = {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(current.read_bytes()).body
               if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        assert all(old[name] == new[name] for name in names)


def sealed_fixture_inputs(f, tmp_path, monkeypatch):
    selected = tmp_path / "selected"
    selected.mkdir()
    copied_self = selected / "current_bootstrap.py"
    copied_self.write_bytes(PATH.read_bytes())
    monkeypatch.setattr(B, "SELF", copied_self)
    inventory = B.source_inventory(f.worker)
    (selected / "source-inventory19.json").write_bytes(B.encode(inventory))
    (selected / "runtime-selection.json").write_bytes(f.kwargs["runtime_blob"])
    files = {key: {"path": str(path), **f.worker.pin(path)} for key, path in {
        "bootstrap": copied_self, "inventory": selected / "source-inventory19.json",
        "runtime": selected / "runtime-selection.json",
    }.items()}
    blob = B.encode({"schema_id": B.SELECTION_KIND, "profile": B.PROFILE, "files": files})
    return blob, hashlib.sha256(blob).hexdigest(), selected


def test_sealed_entry_runs_concrete_fake_bootstrap_without_live_hash_reselection(
    fixture, tmp_path, monkeypatch
):
    f = fixture
    blob, sha, _ = sealed_fixture_inputs(f, tmp_path, monkeypatch)
    components = B.bootstrap_selected(
        f.config, selection_blob=blob, selection_sha=sha, helper=f.helper,
        helper_check=f.kwargs["helper_check"], worker=f.worker,
        load_dependencies=f.kwargs["load_dependencies"],
    )
    assert len(f.calls) == 48 and components.state["component_complete"] is True
    assert components.state["qualified"] is False and components.state["transport_ready"] is False
    assert components.close()["continuation_authorized"] is False


@pytest.mark.parametrize("which", ["bootstrap", "inventory", "runtime"])
def test_sealed_entry_rejects_actual_selected_file_drift_before_any_framework(
    fixture, tmp_path, monkeypatch, which
):
    f = fixture
    blob, sha, _ = sealed_fixture_inputs(f, tmp_path, monkeypatch)
    path = Path(json.loads(blob)["files"][which]["path"])
    with path.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="Selected source SHA differs"):
        B.bootstrap_selected(
            f.config, selection_blob=blob, selection_sha=sha, helper=f.helper,
            helper_check=f.kwargs["helper_check"], worker=f.worker,
            load_dependencies=f.kwargs["load_dependencies"],
        )
    assert not f.events and not f.calls


def test_bootstrap_gates_order_exact48_and_no_ready(fixture):
    f = fixture
    components = open_components(f)
    assert f.events.index("current18-auth") < f.events.index("before-input-and-framework")
    assert f.events.index("cache-initial") < f.events.index("dependency-loader")
    assert f.events.index("before-model-constructor") < f.events.index("constructor")
    assert len(f.calls) == 48
    assert sum(len(rows) for rows, _ in f.calls) == 216
    assert sum(sum(map(len, rows)) for rows, _ in f.calls) == 96544
    assert "probabilities" not in f.backend.instance.runtime.__dict__
    report = json.loads((f.output / "bootstrap.json").read_bytes())
    assert report["component_complete"] is True and report["transport_ready"] is False
    assert report["qualified"] is False and report["helper"]["qualification_sha256"] == B.HELPER_SHA
    assert report["warmup"]["validated_probability_calls"] == 48
    assert components.close()["whole_worker_exit_confirmed"] is False


def test_context_derivative_uses_normal_validation_and_only_changes_cap():
    worker = B.load_worker()
    original = decision3_spec("d3-edge")
    result = worker.matched_context_spec(original)
    expected = original.model_dump()
    expected["context_limit"] = 8192
    assert result.model_dump() == expected and original.context_limit == 16384
    with pytest.raises(ValueError):
        worker.matched_context_spec(result)


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "unselected"),
        ("model", "D3-edge"),
        ("model", ""),
        ("model", None),
        ("workload", "full-ten"),
        ("expected_memtotal", True),
    ],
)
def test_refused_arms_and_configuration_never_enter_actor(fixture, field, value):
    f = fixture
    f.config[field] = value
    with pytest.raises(ValueError):
        open_components(f)
    assert not f.events and not f.calls


@pytest.mark.parametrize(
    "field,value",
    [
        (
            "qualification_sha256",
            "1dfc2c7ab91bf0144f0547b5b144de969310d84944097ac60ec82b1a8d49012c",
        ),
        ("source_count", 93),
        ("tokens", True),
        ("seconds", 20.0),
        ("endpoint", "http://example.com:80"),
    ],
)
def test_q95_is_a_new_identity_not_q93_retag(fixture, field, value):
    f = fixture
    f.helper[field] = value
    with pytest.raises(ValueError):
        open_components(f)
    assert not f.events


def test_false_live_helper_or_co_resident_admission_stops_before_cache(fixture):
    f = fixture
    f.kwargs["helper_check"] = lambda *_args: {"helper_admitted": False}
    with pytest.raises(ValueError, match="admission evidence"):
        open_components(f)
    assert "cache-initial" not in f.events and "constructor" not in f.events


@pytest.mark.parametrize("failure", ["gate", "native", "geometry", "cache"])
def test_partial_failures_close_backend_without_ready_or_cold_fallback(fixture, failure):
    f = fixture
    values = {
        "gate": "warmup-before-probabilities-02",
        "native": 2,
        "geometry": True,
        "cache": "post-warmup",
    }
    f.failures[failure] = values[failure]
    f.failures["close"] = True
    with pytest.raises((ValueError, RuntimeError)) as captured:
        open_components(f)
    assert "secondary-close" not in str(captured.value)
    assert any("secondary-close" in note for note in captured.value.__notes__)
    assert f.backend.instance.closed
    assert "probabilities" not in f.backend.instance.runtime.__dict__
    report = json.loads((f.output / "bootstrap.json").read_bytes())
    assert report["component_complete"] is False and report["transport_ready"] is False
    if failure in {"gate", "native"}:
        warm = json.loads((f.output / "warmup.json").read_bytes())
        assert warm["stage"] == "failed" and warm["completed_probability_calls"] == 2


def test_output_escape_fails_before_any_auth_or_cache(fixture, tmp_path):
    f = fixture
    f.config["output_dir"] = str(tmp_path.parent)
    with pytest.raises(ValueError, match="bounded replay"):
        open_components(f)
    assert not f.events


def test_uuid_change_is_a_refusal_at_each_cuda_gate(fixture):
    f = fixture
    components = open_components(f)
    f.uuid[0] = "GPU-another"
    result = request_candidate(components, envelope(f), f.clock)["candidate"]
    assert result["success"] is False and result["wire_body_utf8"] is None
    assert result["journal"]["error"]["message"] == "Actual CUDA UUID changed"
    assert f.backend.instance.predict_calls == 0


def test_original_order_and_all_heads_reach_native_wire(fixture):
    f = fixture
    components = open_components(f)
    result = request_candidate(components, envelope(f), f.clock)
    candidate = result["candidate"]
    assert result["transport_ready"] is False and candidate["success"]
    wire = json.loads(candidate["wire_body_utf8"])
    assert list(wire["answers"]) == ["operation", "click_target"]
    assert list(wire["answers"]["click_target"]["probabilities"]) == ["0", "1"]
    assert candidate["journal"]["original_request"]["state"] == {"z": "順序", "a": [3, 1]}
    assert f.backend.instance.predict_calls == 1
    assert (
        components.adapter.backend is f.backend.instance
        and components.adapter.evidence_sink is None
    )
    assert "system_one" not in f.backend.instance.model.__dict__ or callable(
        f.backend.instance.model.system_one
    )


@pytest.mark.parametrize("stage", ["raw", "wire", "final-persist"])
def test_late_raw_conversion_or_final_persist_never_offers_action(fixture, monkeypatch, stage):
    f = fixture
    components = open_components(f)
    previous_model = f.backend.instance.model.__dict__.copy()
    if stage == "raw":
        f.failures["raw_delay"] = 21
    elif stage == "wire":
        original = components.wire.named_browser_response

        def late(*args):
            value = original(*args)
            f.clock.now += 21
            return value

        monkeypatch.setattr(components.wire, "named_browser_response", late)
    else:
        original = f.worker.persist

        def late(path, value, **kwargs):
            original(path, value, **kwargs)
            if value.get("stage") == "complete":
                f.clock.now += 21

        monkeypatch.setattr(f.worker, "persist", late)
    result = request_candidate(components, envelope(f), f.clock)["candidate"]
    assert not result["success"] and result["campaign_stop"] and result["wire_body_utf8"] is None
    assert result["journal"]["raw_returned"] is True
    assert result["journal"]["raw_response"]["model"] == "d3-edge"
    assert result["journal"]["error"]["type"] == "TimeoutError"
    assert (
        components.adapter.backend is f.backend.instance
        and components.adapter.evidence_sink is None
    )
    assert f.backend.instance.model.__dict__ == previous_model
    assert "probabilities" not in f.backend.instance.runtime.__dict__


@pytest.mark.parametrize("remaining,width", [(0, 2), (20.001, 2), (True, 2), (20, 256)])
def test_deadline_or_wide_head_refuses_intact_without_native_call(fixture, remaining, width):
    f = fixture
    components = open_components(f)
    result = request_candidate(components, envelope(f, remaining=remaining, width=width), f.clock)[
        "candidate"
    ]
    assert not result["success"] and result["wire_body_utf8"] is None
    assert f.backend.instance.predict_calls == 0
    if width == 256:
        assert (
            len(result["journal"]["original_request"]["questions"]["click_target"]["criteria"])
            == 256
        )


def test_original_typed_startup_caps_cannot_be_expanded(fixture):
    f = fixture
    f.config["workload"] = "known-four"
    components = open_components(f)
    assert components.startup["max_browser_envelopes"] == 64
    for value in (160, True, 64.0):
        changed = {**components.startup, "max_browser_envelopes": value}
        with pytest.raises(ValueError):
            f.worker.validate_startup(changed)


@pytest.mark.parametrize("change", ["empty", "wrong-sha", "x86", "wrong-transformers", "bool-fast"])
def test_unselected_or_placeholder_runtime_never_authenticates_source(fixture, change):
    f = fixture
    value = json.loads(f.kwargs["runtime_blob"])
    if change == "empty":
        value["files"] = {}
    elif change == "x86":
        value["machine"] = "x86_64"
    elif change == "wrong-transformers":
        value["distributions"]["transformers"]["version"] = "latest"
    elif change == "bool-fast":
        value["kernel_policy"]["fast"] = 0
    blob = B.encode(value)
    f.kwargs.update(
        runtime_blob=blob,
        runtime_sha="a" * 64 if change == "wrong-sha" else hashlib.sha256(blob).hexdigest(),
    )
    with pytest.raises(ValueError):
        open_components(f)
    assert not f.events


def test_offline_environment_requires_fresh_modules_and_denies_network(
    fixture, monkeypatch, tmp_path
):
    f = fixture
    monkeypatch.setattr(B, "ROOT", tmp_path)
    modules = tmp_path / "artifacts/jev-d3-browser-module-cache/n-01/d3-edge"
    modules.mkdir(parents=True)
    required = {
        "HF_HUB_CACHE": str(f.core.HUB),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_TOKEN_PATH": "/nonexistent-d3-public-token",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_ENABLE_PARALLEL_LOADING": "false",
        "HF_PARALLEL_LOADING_WORKERS": "1",
        "HF_MODULES_CACHE": str(modules),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for key, value in required.items():
        monkeypatch.setenv(key, value)
    hooks = []
    monkeypatch.setattr(sys, "addaudithook", hooks.append)
    blocked = OFFLINE_ENVIRONMENT(f.config, f.core)
    assert len(hooks) == 1
    with pytest.raises(RuntimeError, match="Network forbidden"):
        hooks[0]("socket.connect", ("never-connect",))
    assert blocked == ["socket.connect"]
    (modules / "stale.py").write_text("stale")
    with pytest.raises(ValueError, match="Fresh owned"):
        OFFLINE_ENVIRONMENT(f.config, f.core)


def test_source_has_no_actual_process_socket_unit_or_old_initializer():
    tree = ast.parse(PATH.read_bytes())
    forbidden = {
        "Popen",
        "system",
        "fork",
        "socket",
        "systemctl",
        "initialize",
        "setrlimit",
        "CDLL",
    }
    assert not [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and (
            (isinstance(n.func, ast.Attribute) and n.func.attr in forbidden)
            or (isinstance(n.func, ast.Name) and n.func.id in forbidden)
        )
    ]


def test_partial_constructor_is_unknown_until_whole_owned_worker_exit(fixture):
    f = fixture
    f.failures["constructor"] = True
    with pytest.raises(RuntimeError, match="primary-constructor-partial"):
        open_components(f)
    report = json.loads((f.output / "bootstrap.json").read_bytes())
    assert report["component_complete"] is False and report["transport_ready"] is False
    assert report["constructor_partial_effects"].startswith("unknown")
    assert (
        not f.backend.instance.closed
    )  # Controller must close whole worker, not claim library cleanup.


def test_wrong_publisher_provenance_is_not_a_numerical_fallback(fixture):
    f = fixture
    f.failures["provenance"] = True
    with pytest.raises(ValueError, match="Publisher provenance"):
        open_components(f)
    assert not f.calls and f.backend.instance.closed


def test_actual_runtime_verifier_with_only_small_synthetic_files(fixture, monkeypatch, tmp_path):
    f = fixture
    source, metadata = tmp_path / "fake_torch.py", tmp_path / "METADATA"
    source.write_text("# synthetic source, never imported\n")
    metadata.write_text("Name: synthetic\n")
    value = json.loads(runtime_blob())
    value["interpreter"], value["python"] = str(source), list(sys.version_info[:2])
    monkeypatch.setattr(sys, "executable", str(source))
    value["files"] = {str(p): f.worker.pin(p) for p in (source, metadata)}
    value["modules"] = {
        name: {"path": str(source), "kind": "source", "size_bytes": source.stat().st_size}
        for name in ("torch", "torch._C", "transformers")
    }
    for row in value["distributions"].values():
        row["metadata_path"] = str(metadata)
    blob = B.encode(value)
    selection = B.RuntimeSelection(blob, hashlib.sha256(blob).hexdigest(), f.worker)
    monkeypatch.setattr(B.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(
        B.importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(
            version=value["distributions"][name]["version"], _path=tmp_path
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__file__=str(source)))
    assert VERIFY_RUNTIME(selection, loaded=True)["loaded_origins_checked"] is True
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__file__=str(tmp_path / "wrong.py")))
    with pytest.raises(ValueError, match="Selected loaded runtime origin differs"):
        VERIFY_RUNTIME(selection, loaded=True)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__file__=str(source)))
    source.write_text("changed")
    with pytest.raises(ValueError, match="Selected runtime file changed"):
        VERIFY_RUNTIME(selection)


def test_component_close_fence_runs_even_after_backend_failure(fixture, monkeypatch):
    f = fixture
    components = open_components(f)
    f.failures["close"] = True
    fences = []

    def failing_fence(_index):
        fences.append(True)
        raise RuntimeError("secondary-fence")

    monkeypatch.setattr(f.torch.cuda, "synchronize", failing_fence)
    with pytest.raises(RuntimeError, match="secondary-close") as captured:
        components.close()
    assert fences == [True] and any("secondary-fence" in n for n in captured.value.__notes__)


def test_lazy_loader_refuses_unselected_metadata_before_any_framework_import():
    def refuse():
        raise ValueError("synthetic-runtime-refusal")

    with pytest.raises(ValueError, match="synthetic-runtime-refusal"):
        B.dependencies(SimpleNamespace(verify=refuse))


def valid_stale_cache(path):
    """Create a real timestamp/size-valid pyc containing different source code."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'VALUE = "evil"\n')
    stamp = path.stat()
    py_compile.compile(str(path), doraise=True, invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
    path.write_bytes(b'VALUE = "safe"\n')
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    spec = importlib.util.spec_from_file_location("stale_cache_control", path)
    control = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(control)
    assert control.VALUE == "evil"  # Prove the cached code is accepted even under -B.
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_selected_worker_executes_checked_bytes_instead_of_valid_stale_pyc(tmp_path, monkeypatch):
    path = tmp_path / "worker.py"
    expected = valid_stale_cache(path)
    monkeypatch.setattr(B, "WORKER", path)
    monkeypatch.setattr(B, "WORKER_SHA", expected)
    assert B.load_worker().VALUE == "safe"


@pytest.mark.parametrize("kind", ["core", "wire"])
def test_private_selected_source_loader_ignores_valid_stale_pyc(tmp_path, monkeypatch, kind):
    worker = B.load_worker()
    path = tmp_path / (kind + ".py")
    expected = valid_stale_cache(path)
    monkeypatch.setattr(worker, kind.upper() + "_PATH", path)
    monkeypatch.setattr(worker, kind.upper() + "_SHA", expected)
    assert worker.load_file("selected_" + kind, path).VALUE == "safe"
    path.write_bytes(b'VALUE = "drift"\n')
    with pytest.raises(ValueError, match="source SHA differs"):
        worker.load_file("selected_" + kind, path)


def test_selected_project_imports_checked_bytes_ignoring_valid_pyc_and_fails_on_drift(
    tmp_path, monkeypatch
):
    import importlib

    root = tmp_path / "src/s1"
    package, child = root / "__init__.py", root / "fixture.py"
    expected_package, expected_child = valid_stale_cache(package), valid_stale_cache(child)
    monkeypatch.setattr(B, "ROOT", tmp_path)
    for name in list(sys.modules):
        if name == "s1" or name.startswith("s1."):
            monkeypatch.delitem(sys.modules, name)
    pins = {"src/s1/__init__.py": expected_package, "src/s1/fixture.py": expected_child}
    try:
        with B.ProjectSources(pins) as selected:
            module = importlib.import_module("s1.fixture")
            assert module.VALUE == sys.modules["s1"].VALUE == "safe"
            assert module.__loader__ is selected
            with pytest.raises(ValueError, match="Unselected project module"):
                importlib.import_module("s1.unselected")
        with pytest.raises(ValueError, match="Fresh selected project module"):
            with B.ProjectSources(pins):
                pytest.fail("Already loaded module cache was trusted")
        del sys.modules["s1.fixture"]
        del sys.modules["s1"]
        child.write_bytes(b'VALUE = "drift"\n')
        with B.ProjectSources(pins):
            with pytest.raises(ValueError, match="Selected source SHA differs"):
                importlib.import_module("s1.fixture")
    finally:
        for name in list(sys.modules):
            if name == "s1" or name.startswith("s1."):
                del sys.modules[name]


def test_actual_selected13_project_imports_execute_through_checked_byte_loader(monkeypatch):
    import importlib

    worker = B.load_worker()
    worker.authenticate_sources(B.source_inventory(worker)["selected_projection"])
    for name in list(sys.modules):
        if name == "s1" or name.startswith("s1."):
            monkeypatch.delitem(sys.modules, name)
    try:
        with B.ProjectSources(worker.CORE.SOURCE_PINS) as selected:
            importlib.import_module("s1.evaluation.decision3")
            importlib.import_module("s1.jev_decision3_native")
            assert len(selected.sources) == 13
            for name, (path, _expected, _package) in selected.sources.items():
                module = sys.modules[name]
                assert module.__loader__ is selected and Path(module.__file__).resolve() == path
    finally:
        for name in list(sys.modules):
            if name == "s1" or name.startswith("s1."):
                del sys.modules[name]
