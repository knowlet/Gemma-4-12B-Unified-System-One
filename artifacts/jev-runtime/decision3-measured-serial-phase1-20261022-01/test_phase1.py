"""CPU-only serial constructor parity, gates, scoped helper and monitor tests."""

import ast
import copy
import hashlib
import importlib.util
import json
import sys
import threading
import types
from pathlib import Path

import pytest

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[2]


def module(name):
    path = HERE / (name + ".py")
    spec = importlib.util.spec_from_file_location("_phase1_test_" + name, path)
    value = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), vars(value))
    return value


P, OBS, B, H, S, R = [
    module(name)
    for name in (
        "phase1",
        "observations",
        "measured_backend",
        "measured_helper",
        "owner_sampler",
        "resource_policy",
    )
]


@pytest.fixture
def inputs():
    fixture = module("test_resource_policy")
    return fixture.inputs.__wrapped__()


def descriptor():
    paths = {
        "phase1": "phase1.py",
        "backend": "measured_backend.py",
        "observations": "observations.py",
        "policy": "resource_policy.py",
        "helper": "measured_helper.py",
        "owner_sampler": "owner_sampler.py",
    }
    files = {}
    for key, label in paths.items():
        path, blob = HERE / label, (HERE / label).read_bytes()
        files[key] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": hashlib.sha256(blob).hexdigest(),
            "size_bytes": len(blob),
        }
    blob = P.encode(
        {
            "schema_id": "decision3-measured-phase1-component-selection-v1",
            "profile": P.PROFILE,
            "files": files,
        }
    )
    return blob, hashlib.sha256(blob).hexdigest()


def test_dry_never_imports_frameworks(capsys, monkeypatch):
    def refuse(*_a, **_k):
        pytest.fail("Dry CLI attempted execution")

    monkeypatch.setattr(P, "run_selected", refuse)
    assert P.main([]) == 0
    assert json.loads(capsys.readouterr().out)["actual_model_started"] is False
    with pytest.raises(ValueError, match="No actor entrypoint"):
        P.main(["--run"])


def test_selected_descriptor_authenticates_all_six_sources():
    blob, digest = descriptor()
    assert len(P.selected_components(blob, digest)["files"]) == 6
    with pytest.raises(ValueError, match="Independently selected"):
        P.selected_components(blob, "0" * 64)
    value = P.parse(blob)
    value["files"].pop("owner_sampler")
    changed = P.encode(value)
    with pytest.raises(ValueError, match="Exact measured source set"):
        P.selected_components(changed, hashlib.sha256(changed).hexdigest())


@pytest.mark.parametrize("key,value", [("profile", "old-core"), ("schema_id", "old")])
def test_descriptor_cannot_relabel_original_profile(key, value):
    blob, _ = descriptor()
    parsed = P.parse(blob)
    parsed[key] = value
    changed = P.encode(parsed)
    with pytest.raises(ValueError, match="Explicit measured"):
        P.selected_components(changed, hashlib.sha256(changed).hexdigest())


def test_component_loader_ignores_stale_bytecode(tmp_path, monkeypatch):
    path = tmp_path / "selected.py"
    path.write_text("VALUE = 'checked bytes'\n")
    monkeypatch.setattr(P, "ROOT", tmp_path)
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "selected.cpython-312.pyc").write_bytes(b"untrusted cache")
    value = P.load(path, hashlib.sha256(path.read_bytes()).hexdigest(), "_checked_phase1_fixture")
    assert value.VALUE == "checked bytes"


@pytest.mark.parametrize("previous", [None, "0"])
def test_serial_environment_selected_before_framework_and_restored(previous, monkeypatch):
    if previous is None:
        monkeypatch.delenv("HF_DEACTIVATE_ASYNC_LOAD", raising=False)
    else:
        monkeypatch.setenv("HF_DEACTIVATE_ASYNC_LOAD", previous)
    with pytest.raises(RuntimeError, match="owned-failure"):
        with P.serial_environment():
            assert P.os.environ["HF_DEACTIVATE_ASYNC_LOAD"] == "1"
            raise RuntimeError("owned-failure")
    assert P.os.environ.get("HF_DEACTIVATE_ASYNC_LOAD") == previous


def test_serial_environment_refuses_preimported_framework(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))
    with pytest.raises(ValueError, match="Fresh model process"):
        with P.serial_environment():
            pytest.fail("Late serial selection admitted")


class StripOriginal(ast.NodeTransformer):
    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        return (
            ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
            if (isinstance(node.value, ast.Name) and node.value.id == "original")
            else node
        )


def constructor_nodes(path, name):
    tree = ast.parse(path.read_bytes())
    cls = next(
        node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == name
    )
    constructor = next(
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    return cls, constructor


def test_actual_publisher_loader_and_numerics_ast_parity():
    original_path = ROOT / "src/s1/evaluation/decision3.py"
    assert hashlib.sha256(original_path.read_bytes()).hexdigest() == B.ORIGINAL_SHA
    _, old = constructor_nodes(original_path, "Decision3Backend")
    cls, new = constructor_nodes(HERE / "measured_backend.py", "MeasuredSerialD3Backend")
    assert [node.name for node in cls.body if isinstance(node, ast.FunctionDef)] == ["__init__"]
    # Exact constructor tail from transformers-version gate through all runtime
    # numerical checks. Only the explicit original module qualification differs.
    old_tail = next(
        i
        for i, n in enumerate(old.body)
        if isinstance(n, ast.If) and "metadata.version" in ast.unparse(n.test)
    )
    new_tail = next(
        i
        for i, n in enumerate(new.body)
        if isinstance(n, ast.If) and "metadata.version" in ast.unparse(n.test)
    )
    old_ast = ast.Module(body=old.body[old_tail:], type_ignores=[])
    new_ast = StripOriginal().visit(ast.Module(body=new.body[new_tail:], type_ignores=[]))
    assert ast.dump(old_ast, include_attributes=False) == ast.dump(
        new_ast, include_attributes=False
    )


@pytest.fixture
def constructor_fixture(monkeypatch):
    calls = []
    runtime = types.SimpleNamespace(
        backbone=types.SimpleNamespace(
            parameters=lambda: iter([types.SimpleNamespace(dtype="torch.bfloat16")])
        ),
        readout_dtype="float32",
        temperature=1.0,
        max_length=None,
        attention_mode="noncausal_full_attention",
        permutation_average=False,
        batch_size=8,
    )

    class Original:
        def predict(self, *args):
            return args

        def close(self):
            return "original-close"

    class AutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append((args, kwargs))
            return types.SimpleNamespace(runtime=runtime)

    checkpoint = {"model_id": "vllm-sr/d3", "resident_file_bytes": 52176025120}
    original = types.SimpleNamespace(
        Decision3Backend=Original,
        configuration_blockers=lambda _spec: [],
        CHECKPOINTS={"d3": checkpoint},
        metadata=types.SimpleNamespace(version=lambda _name: "5.17.0"),
        verify_publisher_sources=lambda directory, identity: {
            "directory": directory,
            "identity": identity,
        },
        NATIVE_BATCH_SIZE=8,
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(
            snapshot_download=lambda *args, **kwargs: (
                calls.append((args, kwargs)) or "/cpu/fake-cache"
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoModel=AutoModel))
    for key, value in {
        "HF_DEACTIVATE_ASYNC_LOAD": "1",
        "HF_ENABLE_PARALLEL_LOADING": "false",
        "HF_PARALLEL_LOADING_WORKERS": "1",
    }.items():
        monkeypatch.setenv(key, value)
    gates = types.SimpleNamespace(
        expected={"boot_id": "boot", "mem_total_bytes": 100}, check=lambda stage: {"stage": stage}
    )
    spec = types.SimpleNamespace(
        id="d3-local", model_id="vllm-sr/d3", revision="pinned", device="cuda:0"
    )
    return original, gates, spec, calls, runtime


def test_explicit_constructor_inherits_original_inference(constructor_fixture):
    original, gates, spec, calls, _runtime = constructor_fixture
    constructor = B._backend_type(original, gates)
    backend = constructor(
        spec, helper_already_loaded=True, expected_boot_id="boot", expected_memtotal=100
    )
    assert constructor.predict is original.Decision3Backend.predict
    assert constructor.close is original.Decision3Backend.close
    assert backend.memory_preflight == {"stage": "before-model-constructor"}
    assert calls[0] == (("vllm-sr/d3",), {"revision": "pinned", "local_files_only": True})
    assert calls[1][1] == {
        "trust_remote_code": True,
        "local_files_only": True,
        "device_map": {"": "cuda:0"},
        "dtype": "bfloat16",
        "batch_size": 8,
        "verify": "full",
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("readout_dtype", "bfloat16"),
        ("temperature", 2.0),
        ("max_length", 8192),
        ("permutation_average", True),
        ("batch_size", 1),
    ],
)
def test_constructor_preserves_original_numerical_refusals(constructor_fixture, field, value):
    original, gates, spec, _, runtime = constructor_fixture
    setattr(runtime, field, value)
    with pytest.raises(ValueError, match="native numerical policy"):
        B._backend_type(original, gates)(
            spec, helper_already_loaded=True, expected_boot_id="boot", expected_memtotal=100
        )


def test_serial_environment_refuses_before_loader(constructor_fixture, monkeypatch):
    original, gates, spec, calls, _ = constructor_fixture
    monkeypatch.delenv("HF_DEACTIVATE_ASYNC_LOAD")
    with pytest.raises(ValueError, match="Explicit measured serial"):
        B._backend_type(original, gates)(
            spec, helper_already_loaded=True, expected_boot_id="boot", expected_memtotal=100
        )
    assert calls == []


def test_gate_initial_refusal_is_durable_before_any_driver(inputs, tmp_path):
    snapshot, expected = copy.deepcopy(inputs)
    snapshot["proc_meminfo"]["MemAvailable"] -= 1
    saved = []
    observations = types.SimpleNamespace(
        host_snapshot=lambda _expected: snapshot,
        DriverIdentity=lambda: pytest.fail("Early driver call"),
    )
    gates = P.Gates(
        R,
        observations,
        types.SimpleNamespace(persist=lambda p, v: saved.append((p, v))),
        tmp_path,
        expected,
    )
    with pytest.raises(ValueError, match="host_available_below_phase_floor"):
        gates.check("before-input-and-framework")
    assert gates.baseline is None and saved[0][1]["decision"]["admitted"] is False


def test_repeated_pre_framework_cache_gates_preserve_initial_baseline(inputs, tmp_path):
    snapshot, expected = copy.deepcopy(inputs)
    saved = []
    gates = P.Gates(
        R,
        types.SimpleNamespace(host_snapshot=lambda _expected: snapshot),
        types.SimpleNamespace(persist=lambda p, v: saved.append(v)),
        tmp_path,
        expected,
    )
    gates.check("before-input-and-framework")
    first = copy.deepcopy(gates.baseline)
    for label in ("initial-before-cache", "initial-before-tensor-hash-model.safetensors"):
        assert gates.check(label)["decision"]["admitted"] is True
        assert gates.baseline == first
    snapshot["cgroup_memory"][0]["events"]["oom"] = 1
    with pytest.raises(ValueError, match="selected_pressure_or_oom_activity"):
        gates.check("initial-next-cache-gate")
    assert saved[-1]["decision"]["admitted"] is False


def fake_kernel_tree(tmp_path, monkeypatch):
    proc, groups = tmp_path / "proc", tmp_path / "cgroup"
    (proc / "42").mkdir(parents=True)
    (proc / "sys/kernel/random").mkdir(parents=True)
    (proc / "sys/kernel/random/boot_id").write_text("fixture-boot\n")
    (proc / "meminfo").write_text(
        "MemTotal: 128 kB\nMemAvailable: 64 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n"
    )
    (proc / "42/cgroup").write_text("0::/test.slice/full.service\n")
    (proc / "42/mountinfo").write_text("30 20 0:28 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n")
    leaf = groups / "test.slice/full.service"
    leaf.mkdir(parents=True)
    for path in (leaf, leaf.parent):
        for label, value in {
            "memory.max": "6442450944",
            "memory.current": "100",
            "memory.peak": "200",
            "memory.swap.max": "0",
            "memory.swap.current": "0",
            "memory.events": "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\n",
            "memory.swap.events": "high 0\nmax 0\nfail 0\n",
        }.items():
            (path / label).write_text(value)
    monkeypatch.setattr(OBS, "PROC", proc)
    monkeypatch.setattr(OBS, "CGROOT", groups)
    identity = {"pid": 42, "start_ticks": 77, "pgid": 42, "sid": 42}
    monkeypatch.setattr(OBS, "process_identity", lambda _pid: dict(identity))
    return {"cgroup_path": str(leaf)}, identity, leaf


def test_collector_keeps_every_ancestor_and_kernel_accounting_field(tmp_path, monkeypatch):
    expected, identity, leaf = fake_kernel_tree(tmp_path, monkeypatch)
    snapshot = OBS.host_snapshot(expected, process=identity)
    assert [r["path"] for r in snapshot["cgroup_memory"]] == [str(leaf), str(leaf.parent)]
    assert snapshot["cgroup_memory"][0]["inode"] == leaf.stat().st_ino
    assert snapshot["proc_meminfo"]["MemAvailable"] == 64 * 1024
    assert snapshot["cuda"] is None


def test_collector_missing_swap_limit_fails_without_unlimited_fallback(tmp_path, monkeypatch):
    expected, identity, leaf = fake_kernel_tree(tmp_path, monkeypatch)
    (leaf / "memory.swap.max").unlink()
    with pytest.raises(FileNotFoundError):
        OBS.host_snapshot(expected, process=identity)


def test_collector_reused_pid_is_refused(tmp_path, monkeypatch):
    expected, identity, _ = fake_kernel_tree(tmp_path, monkeypatch)
    calls = iter([dict(identity), {**identity, "start_ticks": 78}])
    monkeypatch.setattr(OBS, "process_identity", lambda _pid: next(calls))
    with pytest.raises(ValueError, match="Selected process changed"):
        OBS.host_snapshot(expected, process=identity)


def cuda_gate_fixture(inputs, tmp_path):
    initial, expected = copy.deepcopy(inputs)
    snapshot, _, _ = module("test_resource_policy").constructor(inputs)
    calls, saved = [], []
    fraction = [1.0]

    class Driver:
        def check(self, _torch, _host):
            return copy.deepcopy(snapshot["raw_driver"])

    observations = types.SimpleNamespace(
        host_snapshot=lambda _expected: copy.deepcopy(initial),
        DriverIdentity=Driver,
        cuda_snapshot=lambda _torch: {
            **snapshot["cuda"],
            "allocator_fraction": fraction[0],
            "allocator_allowed_bytes_upper": int(fraction[0] * snapshot["cuda"]["total_bytes"]),
        },
    )

    def install(value, device):
        calls.append((value, device))
        fraction[0] = value

    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            memory=types.SimpleNamespace(get_allocator_backend=lambda: "native"),
            set_per_process_memory_fraction=install,
        )
    )
    gates = P.Gates(
        R,
        observations,
        types.SimpleNamespace(persist=lambda p, v: saved.append((p, v))),
        tmp_path,
        expected,
    )
    gates.check("before-input-and-framework")
    return gates, torch, snapshot, calls, saved, fraction


def test_allocator_cap_uses_actual_same_denominator_and_installs_once(inputs, tmp_path):
    gates, torch, snapshot, calls, _, _ = cuda_gate_fixture(inputs, tmp_path)
    row = gates.attach_cuda(torch)
    assert calls == [(R.allocator_fraction(snapshot["raw_driver"]["total_bytes"]), 0)]
    assert (
        row["decision"]["cuda"]["fraction_denominator_bytes"]
        == snapshot["raw_driver"]["total_bytes"]
    )
    assert (
        R.CUDA_CAP_BYTES - 1
        <= row["snapshot"]["cuda"]["allocator_allowed_bytes_upper"]
        <= R.CUDA_CAP_BYTES
    )
    with pytest.raises(ValueError, match="One CUDA cap installation"):
        gates.attach_cuda(torch)
    assert len(calls) == 1


def test_wrong_raw_total_refuses_before_cap_installation(inputs, tmp_path):
    gates, torch, snapshot, calls, _, _ = cuda_gate_fixture(inputs, tmp_path)
    snapshot["raw_driver"]["total_bytes"] -= 1
    with pytest.raises(ValueError, match="raw_driver_host_device_mismatch"):
        gates.attach_cuda(torch)
    assert calls == []


def test_allocator_readback_drift_is_durable_and_never_raises_cap(inputs, tmp_path):
    gates, torch, _, calls, saved, fraction = cuda_gate_fixture(inputs, tmp_path)
    gates.attach_cuda(torch)
    fraction[0] = 1.0
    with pytest.raises(ValueError, match="allocator_readback_or_actual_denominator_mismatch"):
        gates.check("warmup-before-probabilities-00")
    assert len(calls) == 1 and saved[-1][1]["decision"]["admitted"] is False


def checker_fixture(inputs):
    snapshot, expected = copy.deepcopy(inputs)
    checker = object.__new__(H.MeasuredHelperChecker)

    def need(value, reason):
        if not value:
            raise ValueError(reason)

    checker.p = types.SimpleNamespace(
        need=need,
        BASE_BYTES=23919549408,
        GIB=1024**3,
        cgroup_memory=lambda _group: [{"recorded_only": True}],
    )
    checker.policy, checker.expected = R, expected
    checker.observations = types.SimpleNamespace(host_snapshot=lambda _expected: snapshot)
    checker._mode, checker._host_baseline = "initial", None
    checker._value = {"worker": {"cgroup": "/helper"}}
    return checker, snapshot


def test_helper_initial_and_steady_are_distinct_named_scopes(inputs):
    checker, snapshot = checker_fixture(inputs)
    initial = checker._memory()
    assert initial["helper"]["required_host_available_bytes"] == 73958271732
    assert initial["co_resident"]["host"]["required_available_bytes"] == 96986133300
    checker._mode = "steady"
    snapshot["proc_meminfo"]["MemAvailable"] = 40 * 1024**3
    steady = checker._memory()
    assert steady["helper"]["required_host_available_bytes"] == 28266656564
    assert steady["original85_full_host_floor_qualification_claimed"] is False
    assert steady["cuda_observed"] is False


def test_helper_steady_still_refuses_selected_oom(inputs):
    checker, snapshot = checker_fixture(inputs)
    checker._memory()
    checker._mode = "steady"
    snapshot["cgroup_memory"][0]["events"]["oom"] = 1
    with pytest.raises(ValueError, match="MEASURED_HOST_CGROUP_REFUSED"):
        checker._memory()


def actual_callback_fixture(inputs, tmp_path, monkeypatch):
    checker, snapshot = checker_fixture(inputs)
    p = H.load(H.PARENT, H.PARENT_SHA, "_actual_85_callback_cpu_fixture")
    monkeypatch.setattr(p, "cgroup_memory", lambda _group: [{"fixture_only": True}])
    checker.p = p
    checker._value["binding"] = {"fixture_only": "no-live-helper"}
    checker._value["listener"] = {"fixture_only": True}
    checker._sha, checker._directory = "0" * 64, tmp_path
    checker._sequence, checker._lock, checker._phase_lock = 0, threading.Lock(), threading.Lock()
    checker._initial_admitted = False
    checker._static = lambda: {"fixture_only": True}
    checker._live = lambda: (
        {"pid": 42, "argv": ["owned-cpu-fixture"]},
        b"not-a-private-config",
        "unused-fixture-key",
    )
    checker._auth = lambda _token, _deadline: {"fixture_only": True, "forward_calls": 0}
    return checker, snapshot


def test_actual85_callback_runs_both_initial_memory_checks_then_distinct_steady(
    inputs, tmp_path, monkeypatch
):
    checker, snapshot = actual_callback_fixture(inputs, tmp_path, monkeypatch)
    result = checker(checker.binding, "before-cache-and-framework")
    receipt = json.loads((tmp_path / "helper-check-0000.json").read_bytes())
    assert result["helper_admitted"] is True
    assert receipt["memory_before"]["co_resident"]["admitted"] is True
    assert receipt["memory_after"]["co_resident"]["admitted"] is True
    assert (
        receipt["memory_before"]["co_resident"]["baseline"]
        == receipt["memory_after"]["co_resident"]["baseline"]
    )
    snapshot["proc_meminfo"]["MemAvailable"] = 40 * 1024**3
    assert checker(checker.binding, "post-warmup")["co_resident_admitted"] is True
    final = json.loads((tmp_path / "helper-check-0001.json").read_bytes())
    assert final["memory_before"]["phase"] == final["memory_after"]["phase"] == "steady"


def test_actual85_callback_keeps_initial_event_history_across_auth(inputs, tmp_path, monkeypatch):
    checker, snapshot = actual_callback_fixture(inputs, tmp_path, monkeypatch)

    def auth(_token, _deadline):
        snapshot["cgroup_memory"][1]["events"]["oom"] -= 1
        return {"fixture_only": True}

    checker._auth = auth
    with pytest.raises(checker.p.BindingError, match="LIVE_HELPER_REFUSED"):
        checker(checker.binding, "before-cache-and-framework")
    receipt = json.loads((tmp_path / "helper-check-0000.json").read_bytes())
    assert receipt["helper_admitted"] is False and checker._initial_admitted is False
    assert receipt["memory_before"]["co_resident"]["admitted"] is True


class StopAfter:
    def __init__(self, count):
        self.remaining = count

    def is_set(self):
        return self.remaining <= 0

    def wait(self, _seconds):
        self.remaining -= 1


def test_external_host_sampler_records_scoped_peaks_without_cuda(inputs, tmp_path):
    snapshot, expected = inputs
    obs = types.SimpleNamespace(host_snapshot=lambda _expected, **_kwargs: snapshot)
    result = S._sample_loop(
        R,
        obs,
        expected,
        {},
        tmp_path,
        StopAfter(2),
        float("inf"),
        lambda _error: pytest.fail("Unexpected close"),
    )
    assert result["samples"] == 2 and result["cuda_observed"] is False
    assert result["min_host_available_bytes"] == 96986133300
    assert json.loads((tmp_path / "sampler.json").read_bytes())["score"] is None


def test_sampler_refusal_closes_once_preserves_primary_and_receipt(inputs, tmp_path):
    snapshot, expected = copy.deepcopy(inputs)
    snapshot["cgroup_memory"][0]["events"]["oom"] = 1
    calls = []

    def close(error):
        calls.append(error)
        raise RuntimeError("cleanup-secondary")

    with pytest.raises(ValueError, match="Host sampler resource refusal"):
        S._sample_loop(
            R,
            types.SimpleNamespace(host_snapshot=lambda *_a, **_k: snapshot),
            expected,
            {},
            tmp_path,
            StopAfter(2),
            float("inf"),
            close,
        )
    result = json.loads((tmp_path / "sampler.json").read_bytes())
    assert len(calls) == 1 and result["error"]["type"] == "ValueError"
    assert result["owned_close_error"]["message"] == "cleanup-secondary"
    assert result["last_samples"][-1]["decision"]["admitted"] is False


def test_sampler_never_overwrites_interrupted_receipt(inputs, tmp_path):
    snapshot, expected = inputs
    target = tmp_path / "sampler.json"
    target.write_text("historical")
    closes = []
    with pytest.raises(FileExistsError):
        S._sample_loop(
            R,
            types.SimpleNamespace(host_snapshot=lambda *_a, **_k: snapshot),
            expected,
            {},
            tmp_path,
            StopAfter(1),
            float("inf"),
            lambda error: closes.append(error),
        )
    assert target.read_text() == "historical"
    assert len(closes) == 1


def test_no_framework_or_driver_library_imported_by_component_import():
    assert "torch" not in sys.modules and "transformers" not in sys.modules
    assert not hasattr(OBS.DriverIdentity, "library")
