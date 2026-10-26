"""Pure fake numerical dependencies; no ML imports, processes, sockets or payloads."""

import ast
import builtins
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
PATH = (
    REPO / "artifacts/jev-runtime/decision3-GX10-warm-successor-20261020-01/gx10_warm_bootstrap.py"
)
SPEC = importlib.util.spec_from_file_location("gx10_bootstrap_test_subject", PATH)
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

    def __init__(self, calls, fail_native=None):
        self.calls, self.fail_native = calls, fail_native
        self.backbone = SimpleNamespace(parameters=lambda: iter([Tensor(100)]))
        self.readout = Tensor(255, [255, 1], "fp32")

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
    identity = {
        **original_identities["d3-edge"],
        "readout_tensor_bytes": 1020,
        "loaded_parameters": 355,
    }
    identities = {**original_identities, "d3-edge": identity}
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

    core = SimpleNamespace(
        Gates=Gates,
        HUB=tmp_path / "hub",
        write=persist,
        authenticate_cache=cache,
        resource_snapshot=lambda _torch=None: {"cuda": {"uuid": uuid[0]}},
    )

    def auth(manifest):
        assert manifest == json.loads(B.checked_blob(B.MANIFEST, B.MANIFEST_SHA))
        events.append("original17-auth")
        worker.CORE = core
        return identities

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
            assert kwargs == {"expected_boot_id": "boot", "expected_memtotal": 1000}
            self.spec, self.name, self.closed, self.predict_calls = spec, "d3-edge", False, 0
            self.runtime = Runtime(calls, failures.get("native"))
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
    assert inventory["original_projection"] == original
    assert inventory["rows"][:-1] == original["rows"]
    assert inventory["files"] == len(inventory["rows"]) == len(original["rows"]) + 1
    assert inventory["rows"][-1]["path"] == str(B.SELF)
    assert B.load_worker().pin(B.WORKER)["sha256"] == B.WORKER_SHA


def test_bootstrap_gates_order_exact48_and_no_ready(fixture):
    f = fixture
    components = open_components(f)
    assert f.events.index("original17-auth") < f.events.index("before-input-and-framework")
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
        ("model", "d3"),
        ("model", "d3-flash"),
        ("model", "d3-mini"),
        ("model", "d3-nano"),
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
