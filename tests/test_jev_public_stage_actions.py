"""Pure full-schedule/save/identity/closure tests; never import a framework."""

import ast
import hashlib
import json
import random
import sys
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import jev_browser_fit_public_stage_actions as producer  # noqa: E402
import jev_public_stage_action_artifact as artifact  # noqa: E402
import jev_stage_memory_profile_v2 as memory  # noqa: E402


def refs(regime):
    values = {"train": [], "development": []}
    for split, count in artifact.COUNTS[regime].items():
        public_count = {"train": 120, "development": 22}[split]
        for i in range(count):
            row_id = split + "-" + str(i)
            values[split].append(
                {
                    "component": "public" if i < public_count else "owned",
                    "row_id": row_id,
                    "wire_sha256": hashlib.sha256(row_id.encode()).hexdigest(),
                }
            )
    return values


def inputs_value(regime="public120", mode="A"):
    rows = refs(regime)
    mixed_order = None
    if regime == "mixed129":
        import jev_mixed_action_rehearsal as mixed

        mixed_order = [
            r.value() for r in mixed.prepare_schedule(rows["train"], seed=42, version=mixed.SAMPLER)
        ]
    updates = artifact.schedule(regime, rows, mixed_order=mixed_order)
    return {
        "schema_id": artifact.INPUT_SCHEMA,
        "regime": regime,
        "mode": mode,
        "kind": artifact.KINDS[regime],
        "rows": rows,
        "ordered_updates": updates,
    }


@pytest.mark.parametrize("regime", ["public120", "mixed129"])
def test_complete_schedule_has_exact_train_only_rows_and_regime_order(regime):
    value = inputs_value(regime)
    actual = value["ordered_updates"]
    assert len(actual) == artifact.COUNTS[regime]["train"]
    assert Counter(r["component"] for r in actual) == (
        {"public": 120} if regime == "public120" else {"public": 120, "owned": 9}
    )
    assert {(r["component"], r["row_id"]) for r in actual} == {
        (r["component"], r["row_id"]) for r in value["rows"]["train"]
    }
    assert all(r["weight"]["numerator"] == r["weight"]["denominator"] == 1 for r in actual)
    if regime == "public120":
        expected = list(value["rows"]["train"])
        random.Random(42).shuffle(expected)
        assert [{k: v for k, v in r.items() if k != "weight"} for r in actual] == expected
    else:
        shuffled = list(value["rows"]["train"])
        random.Random(42).shuffle(shuffled)
        assert [r["row_id"] for r in actual] != [r["row_id"] for r in shuffled]
        with pytest.raises(ValueError, match="SHA-ordered"):
            artifact.schedule(regime, value["rows"], mixed_order=list(reversed(actual)))


@pytest.mark.parametrize(
    "tamper", ["short_train", "short_dev", "repeat", "dev_leak", "owned_public"]
)
def test_schedule_rejects_partial_repeat_leak_and_wrong_regime(tamper):
    rows = refs("public120")
    if tamper == "short_train":
        rows["train"].pop()
    elif tamper == "short_dev":
        rows["development"].pop()
    elif tamper == "repeat":
        rows["train"][-1] = rows["train"][0]
    elif tamper == "dev_leak":
        rows["development"][0]["wire_sha256"] = rows["train"][0]["wire_sha256"]
    else:
        rows["train"][0]["component"] = "owned"
    with pytest.raises(ValueError):
        artifact.schedule("public120", rows)


class Parameter:
    def __init__(self, size):
        self.size, self.requires_grad, self.grad, self._version = size, True, None, 0

    def numel(self):
        return self.size


class Module:
    def __init__(self):
        self.hooks = []

    def register_forward_pre_hook(self, hook, **_kwargs):
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.hooks.remove(hook))

    def enter(self, kwargs=None):
        for hook in self.hooks:
            if kwargs is None:
                hook(self, ())
            else:
                hook(self, (), kwargs)


class Loss:
    def __init__(self, layers):
        self.layers = layers

    def backward(self):
        for layer in self.layers:
            layer.enter()

    def detach(self):
        return 1.0

    def __float__(self):
        return 1.0


@pytest.fixture
def full_loop(tmp_path, monkeypatch):
    parameters = [Parameter(5193728 - 175)] + [Parameter(1) for _ in range(175)]
    lm = Module()
    lm.named_parameters = lambda: [("qv-" + str(i), p) for i, p in enumerate(parameters)]
    lm.gradient_checkpointing_enable = lambda **kwargs: calls.append(("GC-enable", kwargs))
    lm.gradient_checkpointing_disable = lambda: calls.append(("GC-disable", None))
    lm.train = lambda: calls.append(("train", None))
    lm.eval = lambda: calls.append(("eval", None))
    layers = [Module() for _ in range(48)]
    model = SimpleNamespace(lm=lm, head=SimpleNamespace(weight=object()))
    calls, phases, releases = [], [], []
    active = {"phase": None, "refuse": None}
    state = {
        "completed_updates": 0,
        "synchronized_updates": 0,
        "optimizer_step_attempts": 0,
        "actual_native_backward_calls": 0,
    }

    class Adam:
        def __init__(self, params, *, lr, weight_decay):
            assert len(params) == 176 and lr == 5e-5 and weight_decay == 0.0
            self.state = {}

        def zero_grad(self, *, set_to_none):
            assert set_to_none is True
            calls.append(("zero", None))

        def step(self):
            assert active["phase"] == "native-Adam-step"
            calls.append(("Adam", None))

    def phase(name, fn):
        phases.append(name)
        active["phase"] = name
        if active["refuse"] == name:
            raise RuntimeError("original-phase-entry-refused")
        result = fn()
        calls.append(("fence", name))
        return result

    gate = SimpleNamespace(phase=phase, check=lambda *_: None)
    torch = SimpleNamespace(
        optim=SimpleNamespace(AdamW=Adam),
        manual_seed=lambda seed: calls.append(("seed", seed)),
        nn=SimpleNamespace(
            utils=SimpleNamespace(clip_grad_norm_=lambda params, norm, error_if_nonfinite: 0.25)
        ),
        isfinite=lambda _: SimpleNamespace(all=lambda: True),
        no_grad=nullcontext,
    )

    def action_loss(_model, row, *, plan, mode):
        assert plan == row["record"]["case_id"] and mode in {"A", "C"}
        lm.enter(
            {
                "input_ids": SimpleNamespace(shape=(1, 80)),
                "logits_to_keep": SimpleNamespace(tolist=lambda: [78, 79]),
                "use_cache": False,
            }
        )
        for layer in layers:
            layer.enter()
        calls.append(("native-loss", row["record"]["case_id"]))
        return Loss(layers), {"mode": mode}

    def release(_torch, _model, _gate, _output, fn, **kwargs):
        result = fn()
        releases.append(kwargs)
        return result

    monkeypatch.setattr(
        producer, "original_decoder_layers", lambda _: (layers, {"decoder_layers": 48})
    )
    monkeypatch.setattr(producer, "storage_ledger", lambda *_: {"metadata-only": True})
    monkeypatch.setattr(producer, "operator_call", lambda _torch, _output, _phase, fn: fn())
    monkeypatch.setattr(producer, "native_cache_release", release)
    monkeypatch.setattr(
        producer,
        "backward_cache_release",
        lambda t, m, g, o, fn, **kwargs: release(t, m, g, o, fn, phase="backward", **kwargs),
    )
    kernel = SimpleNamespace(_tensor_sha=lambda _: artifact.HEAD_SHA)
    fit = SimpleNamespace(action_loss=action_loss)

    def run(value):
        row_map = {
            (r["component"], r["row_id"]): {"record": {"case_id": r["row_id"]}}
            for rows in value["rows"].values()
            for r in rows
        }
        plans = {r["row_id"]: r["row_id"] for rows in value["rows"].values() for r in rows}
        return producer.train_stage(
            torch,
            model,
            row_map,
            plans,
            mode=value["mode"],
            gate=gate,
            kernel=kernel,
            fit_kernel=fit,
            output=tmp_path,
            state=state,
            inputs_value=value,
        )

    return SimpleNamespace(
        run=run,
        calls=calls,
        phases=phases,
        releases=releases,
        state=state,
        lm=lm,
        layers=layers,
        active=active,
        output=tmp_path,
    )


@pytest.mark.parametrize(
    "regime,mode", [("public120", "A"), ("public120", "C"), ("mixed129", "A"), ("mixed129", "C")]
)
def test_entire_native_loop_schedule_and_zero_dev_updates_with_fake_native_objects(
    full_loop, regime, mode
):
    value = inputs_value(regime, mode)
    f = full_loop
    f.run(value)
    expected = artifact.expected_counts(regime)
    assert (
        f.state["completed_updates"]
        == f.state["synchronized_updates"]
        == f.state["optimizer_step_attempts"]
        == expected["updates"]
    )
    assert f.state["actual_update_refs"] == value["ordered_updates"]
    assert f.state["actual_native_forward_calls"] == expected["forwards"]
    assert f.state["actual_native_backward_calls"] == expected["backwards"]
    assert len(f.releases) == expected["cache_releases"]
    assert Counter(r["phase"] for r in f.releases) == {
        "train-forward": expected["updates"],
        "backward": expected["updates"],
        "dev-forward": artifact.COUNTS[regime]["development"],
    }
    assert f.phases.count("native-Adam-step") == expected["updates"]
    assert f.phases[-1] == "final-full-head-hash"
    eval_at = next(i for i, call in enumerate(f.calls) if call[0] == "eval")
    assert not any(call[0] == "Adam" for call in f.calls[eval_at:])
    assert not f.lm.hooks and all(not layer.hooks for layer in f.layers)
    trace = json.loads((f.output / "native-call-trace.json").read_text())
    assert all(trace["layers"]["backward:" + str(i)] == expected["backwards"] for i in range(48))


@pytest.mark.parametrize(
    "phase",
    [
        "native-train-forward",
        "native-backward",
        "native-grad-clip",
        "native-Adam-step",
        "native-development-forward",
    ],
)
def test_original_phase_failure_never_retries_or_updates_development(full_loop, phase):
    f = full_loop
    f.active["refuse"] = phase
    with pytest.raises(RuntimeError, match="original-phase-entry-refused"):
        f.run(inputs_value())
    assert f.phases.count(phase) == 1
    assert not f.lm.hooks and all(not layer.hooks for layer in f.layers)
    if phase == "native-development-forward":
        assert f.state["completed_updates"] == 120
    else:
        assert f.state["completed_updates"] == f.state["optimizer_step_attempts"] == 0


def valid_receipt():
    value = inputs_value()
    config = {"artifact_kind": value["kind"], "mode": "A", "source_inventory": {"sha256": "a" * 64}}
    worker = {
        "schema_id": memory.RECEIPT_SCHEMA,
        "profile_id": memory.PROFILE,
        "allocator_policy": memory.CACHE_POLICY,
        "kind": value["kind"],
        "mode": "A",
        "stage": "complete_stage_fit_unqualified",
        "qualified": False,
        "error": None,
        "completed_updates": 120,
        "synchronized_updates": 120,
        "optimizer_step_attempts": 120,
        "actual_native_backward_calls": 120,
        "actual_native_forward_calls": 142,
        "development_updates": 0,
        "actual_cache_releases": 262,
        "model_loads": 1,
        "verified_model_loads": 1,
        "model_constructor_attempts": 1,
        "actual_update_refs": value["ordered_updates"],
        "admitted_inputs_sha256": artifact.sha(artifact.encode(value)),
        "head_sha256": artifact.HEAD_SHA,
        "tensor_metadata": {"tensor_count": 176, "parameters": 5193728},
        "saved_adapter": {"status": "verified-complete"},
        "source_union": config["source_inventory"],
    }
    return value, config, worker


@pytest.mark.parametrize(
    "mutation",
    [
        "old_schema",
        "probe_kind",
        "partial",
        "attempt",
        "sync",
        "head",
        "dev_update",
        "order",
        "model",
        "source",
        "cache",
        "saved",
    ],
)
def test_completed_receipt_rejects_old_probe_partial_or_drift(mutation):
    value, config, worker = valid_receipt()
    artifact.verify_worker_receipt(worker, inputs_value=value, config=config)
    if mutation == "old_schema":
        worker["schema_id"] = "jev-stage-memory-worker-receipt-v1"
    elif mutation == "probe_kind":
        worker["kind"] = "memory_probe_only"
    elif mutation == "partial":
        worker["completed_updates"] -= 1
    elif mutation == "attempt":
        worker["optimizer_step_attempts"] += 1
    elif mutation == "sync":
        worker["synchronized_updates"] -= 1
    elif mutation == "head":
        worker["head_sha256"] = "b" * 64
    elif mutation == "dev_update":
        worker["development_updates"] = 1
    elif mutation == "order":
        worker["actual_update_refs"] = list(reversed(worker["actual_update_refs"]))
    elif mutation == "model":
        worker["model_loads"] = 2
    elif mutation == "source":
        worker["source_union"] = {}
    elif mutation == "cache":
        worker["actual_cache_releases"] -= 1
    else:
        worker["saved_adapter"]["status"] = "failed-or-partial"
    with pytest.raises(ValueError):
        artifact.verify_worker_receipt(worker, inputs_value=value, config=config)


def test_external_source_or_data_reference_drift_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(artifact, "ROOT", tmp_path)
    path = tmp_path / "data.json"
    path.write_bytes(b'{"public":120}')
    ref = artifact.reference(path)
    assert artifact.read_ref(ref) == path.read_bytes()
    path.write_bytes(b'{"public":129}')
    with pytest.raises(ValueError, match="drift"):
        artifact.read_ref(ref)
    with pytest.raises(ValueError, match="Confined"):
        artifact.read_ref({**ref, "path": "../data.json"})


def test_forged_or_owned_as_public_stage_DTO_is_rejected():
    forged = artifact.StageInputs(b"{}", object(), object(), object(), b"{}", object(), "x")
    with pytest.raises(ValueError, match="Forged"):
        forged.check()
    sealed_wrong_type = artifact.StageInputs(
        b"{}", object(), None, None, b"{}", artifact._PROOF, artifact.sha(b"{}\0{}")
    )
    with pytest.raises(ValueError, match="Genuine original public"):
        sealed_wrong_type.check()


def test_guarded_save_partial_never_creates_valid_marker(tmp_path):
    state = {}
    calls = []
    inputs = SimpleNamespace(check=lambda: calls.append("check"))
    model = SimpleNamespace(lm=SimpleNamespace(named_parameters=lambda: []))

    def partial_save(_model, directory):
        directory.mkdir()
        (directory / "adapter_model.safetensors").write_bytes(b"partial")
        raise RuntimeError("writer failed after partial tensor")

    gate = SimpleNamespace(phase=lambda _name, fn: fn())
    with pytest.raises(RuntimeError, match="partial tensor"):
        artifact.guarded_save_adapter(
            model, inputs, tmp_path, gate, SimpleNamespace(_save_adapter=partial_save), state
        )
    assert state["saved_adapter"]["status"] == "failed-or-partial"
    assert not (tmp_path / "child-manifest.json").exists()
    assert (
        json.loads((tmp_path / "progress.json").read_text())["saved_adapter"]["status"]
        == "failed-or-partial"
    )
    assert calls == ["check"]


@pytest.mark.parametrize("kind", ["source", "data"])
def test_guarded_save_postwrite_source_or_data_drift_never_seals(tmp_path, monkeypatch, kind):
    calls = []

    def check():
        calls.append("check")
        if len(calls) == 2:
            raise ValueError(kind + "-drift-after-write")

    def save_adapter(_model, directory):
        directory.mkdir()
        (directory / "adapter_model.safetensors").write_bytes(b"synthetic-saved-tensor")
        (directory / "adapter_config.json").write_text("{}")
        return {p.name: memory.file_pin(p) for p in directory.iterdir()}

    monkeypatch.setattr(
        artifact,
        "public_api",
        lambda: SimpleNamespace(tensor_metadata=lambda *_args, **_kwargs: {"synthetic-only": True}),
    )
    state = {}
    with pytest.raises(ValueError, match=kind + "-drift-after-write"):
        artifact.guarded_save_adapter(
            SimpleNamespace(lm=SimpleNamespace(named_parameters=lambda: [])),
            SimpleNamespace(check=check),
            tmp_path,
            SimpleNamespace(phase=lambda _name, fn: fn()),
            SimpleNamespace(_save_adapter=save_adapter),
            state,
        )
    assert state["saved_adapter"]["status"] == "failed-or-partial"
    assert not (tmp_path / "child-manifest.json").exists()


def test_old_completed_artifact_and_old_CPU_reference_are_refused_before_admission(
    tmp_path, monkeypatch
):
    old = {
        "schema_id": "jev_public_cfg_action_bf16_qualification_v1",
        "kind": "jev_public_native_action_cfg_lora_v1",
        "qualified": True,
    }
    blob = artifact.encode(old)
    (tmp_path / "child-manifest.json").write_bytes(blob)
    calls = []
    gate = SimpleNamespace(check=lambda phase, scope: calls.append((phase, scope)))
    monkeypatch.setattr(
        artifact, "admit_inputs", lambda *_: pytest.fail("Old artifact never reaches new admission")
    )
    with pytest.raises(ValueError, match="old ART95/Modal90/probes refused"):
        artifact.verify_completed_child(
            tmp_path,
            expected_manifest_sha256=artifact.sha(blob),
            plan={"regime": "public120"},
            gate=gate,
        )
    assert calls == [("before-new-stage-child-tree-or-tensor-payload", "cpu")]
    monkeypatch.setattr(artifact, "read_ref", lambda _: blob)
    with pytest.raises(ValueError, match="genuine CPU reference"):
        artifact.verify_CPU_reference({}, {})


@pytest.mark.parametrize(
    "mutation", ["success", "closure", "invocation", "PID", "lifetime", "ACK", "source"]
)
def test_whole_closure_missing_or_changed_identity_never_passes(mutation):
    value, config, worker = valid_receipt()
    worker["unit_binding"] = {"InvocationID": "a" * 32, "ControlGroup": "/app.slice/test.service"}
    worker["native_work_end_capture"] = {"snapshot_sha256": "b" * 64}
    arm = {
        "schema_id": "jev-stage-memory-arm-receipt-v2",
        "profile_id": memory.PROFILE,
        "success": True,
        "closure_confirmed": True,
        "worker_receipt": worker,
        "invocation": "a" * 32,
        "post_stop": {"MainPID": "0"},
        "closure_kind": "retained-unit",
        "retained_lifetime_peaks": {"MemoryPeak": 1},
        "unit": "test.service",
    }
    artifact.verify_closure(arm, worker, config)
    if mutation == "success":
        arm["success"] = False
    elif mutation == "closure":
        arm["closure_confirmed"] = False
    elif mutation == "invocation":
        arm["invocation"] = "c" * 32
    elif mutation == "PID":
        arm["post_stop"]["MainPID"] = "123"
    elif mutation == "lifetime":
        arm["retained_lifetime_peaks"] = None
    elif mutation == "ACK":
        worker["native_work_end_capture"] = None
    else:
        config["source_inventory"] = {}
    with pytest.raises(ValueError):
        artifact.verify_closure(arm, worker, config)


def test_CPU_gate_preserves_original_floor_and_persists_refusal(tmp_path):
    obs = {"MemTotal": 128 * memory.GIB, "MemAvailable": memory.GIB, "boot_id": "same-boot"}
    gate = artifact.CPUPrepareGate(tmp_path, "same-boot", observe=lambda: obs)
    with pytest.raises(ValueError, match="original CPU floor"):
        gate.check("before-owned-native-target-compile-train-0", "cpu")
    row = json.loads((tmp_path / "CPU-gates.jsonl").read_text())
    assert row["admitted"] is False and row["resident_credit_bytes"] == 0
    assert row["required_bytes"] == (obs["MemTotal"] + 4) // 5 + 4 * memory.GIB


def test_pending_capacity_and_evidence_budget_refuse_before_any_launch(monkeypatch):
    monkeypatch.setattr(memory, "arm_run", lambda *_: pytest.fail("No unit launch permitted"))
    for status in ["blocked-capacity", "reviewed-local-full-fit"]:
        profile = {
            "schema_id": "jev-stage-memory-profile-v2",
            "profile_id": memory.PROFILE,
            "cache_policy": memory.CACHE_POLICY,
            "regime": "public120",
            "launch_status": status,
            "evidence_budget": {"status": "insufficient-observed-extrapolation"},
        }
        with pytest.raises(ValueError, match="still pending"):
            producer.check_launch_profile(profile)


def test_compiled_inventory_refuses_unpinned_side_files(tmp_path):
    (tmp_path / "compiled-train.jsonl").write_bytes(b"{}\n")
    assert set(artifact.compiled_inventory(tmp_path)) == {"compiled-train.jsonl"}
    (tmp_path / "unexpected.txt").write_text("unreviewed")
    with pytest.raises(ValueError, match="Only flat"):
        artifact.compiled_inventory(tmp_path)


def test_only_lazy_framework_imports_and_no_runtime_artifact_driver_or_v1():
    for name in artifact.EXTENSIONS:
        source = (ROOT / name).read_text()
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
                )
                assert all(
                    n.split(".")[0]
                    not in {"torch", "transformers", "peft", "modal", "playwright", "httpx"}
                    for n in names
                )
        assert "import jev_stage_memory_profile as" not in source
        assert "spec_from_file_location" not in source and "native_peak_probe_01" not in source
    assert len(artifact.EXTENSIONS) == len(set(artifact.EXTENSIONS)) == 3


def test_optional_frozen_population_metadata_missing_skips_corrupt_fails(tmp_path):
    path = tmp_path / "manifest.json"

    def load_optional():
        if not path.exists():
            pytest.skip("Optional frozen public metadata absent")
        blob = path.read_bytes()
        assert hashlib.sha256(blob).hexdigest() == "a" * 64, "Frozen metadata corrupted"

    with pytest.raises(pytest.skip.Exception):
        load_optional()
    path.write_bytes(b"changed")
    with pytest.raises(AssertionError, match="corrupted"):
        load_optional()


def test_optional_actual_admitted_population_keeps_94_and_102_without_recounting_sources():
    base = (
        ROOT
        / "artifacts/jev-runtime/owned-public-mixed-composition-20261018-01/actual-admit-02/mixed-manifest-A.json"
    )
    if not base.exists():
        pytest.skip("Optional closed mixed admission absent in this checkout")
    blob = base.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == artifact.MIX_SHA["A"], (
        "Frozen admitted mix corrupted"
    )
    manifest = json.loads(blob)
    assert len(manifest["source_closure"]) == 102
    assert not set(artifact.EXTENSIONS) & set(manifest["source_closure"])
    assert len(manifest["sampler"]["ordered_updates"]) == 129
    assert Counter(r["component"] for r in manifest["sampler"]["ordered_updates"]) == {
        "public": 120,
        "owned": 9,
    }


@pytest.fixture
def native_join(tmp_path):
    """Synthetic saved JSON, not native execution or a CPU reference receipt."""
    value, config, worker = valid_receipt()
    value["objective"] = artifact.public_api().objective("A")
    worker["admitted_inputs_sha256"] = artifact.sha(artifact.encode(value))
    config.update(
        nonce="a" * 32,
        boot_id="boot",
        source_pin={"sha256": "1" * 64},
        profile_pin={"sha256": "2" * 64},
        capsule_pin={"sha256": "3" * 64},
    )
    ready = {
        "nonce": config["nonce"],
        "pid": 22,
        "startticks": 10,
        "boot_id": "boot",
        "source_sha256": "1" * 64,
        "profile_sha256": "2" * 64,
        "capsule_sha256": "3" * 64,
        "source_inventory_sha256": "a" * 64,
    }
    binding = {"InvocationID": "a" * 32, "ControlGroup": "/app.slice/test.service"}
    identity = {**ready, **binding}
    worker.update(
        unit_binding=binding,
        allocator={"fraction": 0.3},
        decoder_architecture={"decoder_layers": 48},
    )
    cg = {
        "group": binding["ControlGroup"],
        "current": memory.GIB,
        "peak": 2 * memory.GIB,
        "max": memory.CAP,
        "high": None,
        "swap": 0,
        "swap_max": 0,
        "oom_group": 1,
        "procs": [22],
        "events": {"max": 0, "oom": 0, "oom_kill": 0},
        "stat": {"anon": 1},
    }
    props = {
        **binding,
        "LoadState": "loaded",
        "MainPID": "22",
        "SubState": "running",
        "ActiveState": "active",
    }

    def save(name, obj):
        (tmp_path / name).write_bytes(artifact.encode(obj) + b"\n")

    def save_rows(name, rows):
        (tmp_path / name).write_bytes(b"".join(artifact.encode(r) + b"\n" for r in rows))

    save("ready-for-readback.json", ready)
    capture = {
        "schema_id": "native-work-end-cgroup-v1",
        "identity": identity,
        "checked_monotonic": 1.0,
        "properties": props,
        "cgroup": cg,
    }
    save("native-work-final-cgroup.json", capture)
    snapshot_sha = memory.file_pin(tmp_path / "native-work-final-cgroup.json")["sha256"]
    save(
        "native-work-ack.json",
        {"schema_id": "native-work-ack-v1", **identity, "snapshot_sha256": snapshot_sha},
    )
    worker["native_work_end_capture"] = {"snapshot_sha256": snapshot_sha}
    terminal = {
        **props,
        "MainPID": "0",
        "SubState": "exited",
        "MemoryPeak": str(3 * memory.GIB),
        "MemorySwapPeak": "0",
    }
    save("unit-terminal.json", {"properties": terminal})
    arm = {
        "invocation": binding["InvocationID"],
        "retained_lifetime_peaks": memory.retained_lifetime_peaks(terminal, identity, capture),
        "post_stop": terminal,
        "closure_kind": "retained-unit",
    }
    obs = {"MemTotal": 128 * memory.GIB, "MemAvailable": 128 * memory.GIB, "boot_id": "boot"}
    reserve = (obs["MemTotal"] + 4) // 5
    load = 2 * 23919549408 + reserve
    startup = []
    for phase, scope in [
        ("before-any-project-framework-or-parent-payload", "cpu"),
        ("before-public-project-admission", "cpu"),
        ("immediately-before-legacy-parent-capture-and-Torch", "cpu"),
        ("before-checkpoint-payload-hash-and-BF16-loader", "load"),
        ("before-any-BF16-model-construction", "train"),
    ]:
        startup.append(
            {
                "sequence": len(startup),
                "phase": phase,
                "scope": scope,
                "admitted": True,
                "error": None,
                "swap_or_resident_credit_bytes": 0,
                "expected_boot_id": "boot",
                "observation": obs,
                "required_bytes": {
                    "cpu": reserve + 4 * memory.GIB,
                    "load": load,
                    "train": load + 30 * memory.GIB + 256 * memory.MIB,
                }[scope],
            }
        )
    save_rows("original-startup-gates.jsonl", startup)
    names = [
        "full-original-checkpoint-hash",
        "full-original-constructor",
        "original-model-head-check",
        "original-parent-injection",
        "complete-original-plus-owned-processor-compile",
        "postcompile-original-model-check",
        "optimizer-create-empty",
    ]
    names += [
        name
        for _ in range(120)
        for name in [
            "zero-grad-before-forward",
            "native-train-forward",
            "native-backward",
            "native-grad-clip",
            "native-Adam-step",
        ]
    ]
    names += ["native-development-forward"] * 22
    names += [
        "final-full-head-hash",
        "final-original-model-check",
        "final-original-parent-check",
        "guarded-save-adapter",
        "guarded-saved-tensor-validation",
        "guarded-final-source-data-validation",
    ]
    cuda = {
        "backend": "native",
        "fraction": 0.3,
        "allocated": 1,
        "reserved": 2,
        "peak_allocated": 3,
        "peak_reserved": 4,
        "stats": {"num_ooms": 0, "num_alloc_retries": 0},
    }
    phases = []
    for i, name in enumerate(names):
        floor = memory.research_floor(
            obs, startup=name in {"full-original-checkpoint-hash", "full-original-constructor"}
        )
        phases.append(
            {
                "schema_id": memory.PHASE_SCHEMA,
                "profile_id": memory.PROFILE,
                "event": "stage-gate",
                "phase": name,
                "observation": obs,
                **floor,
                "admitted": True,
                "resident_credit_bytes": 0,
            }
        )
        base = {
            "schema_id": memory.PHASE_SCHEMA,
            "profile_id": memory.PROFILE,
            "sequence": i,
            "phase": name,
            "cuda": cuda,
            "cgroup": cg,
        }
        phases.extend(
            [
                {**base, "event": "intent"},
                {**base, "event": "complete", "error": None, "secondary_errors": []},
            ]
        )
    save_rows("phases.jsonl", phases)
    expected_calls = [("forward", r["row_id"]) for r in value["ordered_updates"]] + [
        ("development", r["row_id"]) for r in value["rows"]["development"]
    ]
    calls = [
        {
            "phase": p,
            "case_id": row_id,
            "use_cache": False,
            "input_shape": [1, 80],
            "positions": [78, 79],
        }
        for p, row_id in expected_calls
    ]
    trace = {
        "decoder_architecture": worker["decoder_architecture"],
        "native_forward_calls": calls,
        "layers": {
            phase + ":" + str(i): count
            for phase, count in [("forward", 120), ("backward", 120), ("development", 22)]
            for i in range(48)
        },
    }
    save("native-call-trace.json", trace)
    op_phases = [p for _ in range(120) for p in ["forward", "backward"]] + ["development"] * 22
    operators = [
        {
            "phase": phase,
            "operators": [{"name": "aten::scaled_dot_product_attention", "count": 48}]
            + (
                [{"name": "aten::_efficient_attention_backward", "count": 48}]
                if phase == "backward"
                else []
            ),
        }
        for phase in op_phases
    ]
    save_rows("operators.jsonl", operators)
    compiled_root = tmp_path / "compiled"
    compiled_root.mkdir()
    updates, development = [], []
    for split, rows in value["rows"].items():
        compiled, plans = [], []
        for ref in rows:
            compiled.append(
                {"record": {"case_id": ref["row_id"], "input_tokens": 80}, "positions": [78, 79]}
            )
            plan = {
                "compiled_row_sha256": "c" * 64,
                "request_sha256": ref["wire_sha256"],
                "target_ids": [7, 106],
                "decision_indices": [0],
            }
            plans.append({"case_id": ref["row_id"], "plan": plan})
            metrics = {
                "objective_mode": "A",
                "training_objective": value["objective"]["loss"],
                "compiled_row_sha256": "c" * 64,
                "request_sha256": ref["wire_sha256"],
                "target_tokens": 2,
                "decision_target_tokens": 1,
                "forced_singleton_target_tokens": 1,
                "correct_target_tokens": 2,
                "branch_full_vocab_correct": 1,
                "branch_allowed_correct": 1,
                "teacher_forced_exact_tokens": True,
                "teacher_forced_constrained_exact_branches": True,
                "loss": 1.0,
                "full_vocab_ce_mean": 1.0,
                "allowed_branch_ce_mean": 1.0,
            }
            (updates if split == "train" else development).append(
                {"case_id": ref["row_id"], "loss": 1.0, "metrics": metrics}
            )
        save_rows("compiled/compiled-" + split + ".jsonl", compiled)
        save_rows("compiled/cfg-plans-" + split + ".jsonl", plans)
    by_id = {r["case_id"]: r for r in updates}
    updates = [
        {**by_id[ref["row_id"]], "step": i} for i, ref in enumerate(value["ordered_updates"], 1)
    ]
    save_rows("updates.jsonl", updates)
    save_rows("development.jsonl", development)
    releases = []
    expected = [
        (r["row_id"], i, p)
        for i, r in enumerate(value["ordered_updates"], 1)
        for p in ["train-forward", "backward"]
    ] + [(r["row_id"], 120, "dev-forward") for r in value["rows"]["development"]]
    for row_id, step, phase in expected:
        before = {"cuda": cuda, "storage_identity": {"sha256": "d" * 64}}
        base = {
            "case_id": row_id,
            "step": step,
            "native_phase": phase,
            "policy": memory.CACHE_POLICY,
            "mode": "A",
            "before": before,
        }
        releases.extend(
            [
                {**base, "event": "intent"},
                {
                    **base,
                    "event": "complete",
                    "success": True,
                    "API_returned": True,
                    "error": None,
                    "after": before,
                },
            ]
        )
    save_rows("cache-releases.jsonl", releases)
    return SimpleNamespace(
        root=tmp_path,
        value=value,
        config=config,
        worker=worker,
        arm=arm,
        save=save,
        save_rows=save_rows,
        phases=phases,
        operators=operators,
        updates=updates,
        releases=releases,
        trace=trace,
    )


@pytest.mark.parametrize(
    "tamper",
    [
        None,
        "phase_failed",
        "missing_entry_guard",
        "truncated_forward",
        "head_position",
        "operator",
        "loss_objective",
        "dev_update",
        "cache_storage",
        "ACK",
    ],
)
def test_pure_saved_native_full_join_refuses_specific_corruption(native_join, tamper):
    c = native_join
    if tamper == "phase_failed":
        c.phases[-1]["event"] = "failed"
        c.save_rows("phases.jsonl", c.phases)
    elif tamper == "missing_entry_guard":
        c.save_rows("phases.jsonl", c.phases[1:])
    elif tamper == "truncated_forward":
        c.trace["native_forward_calls"][0]["input_shape"] = [1, 40]
        c.save("native-call-trace.json", c.trace)
    elif tamper == "head_position":
        c.trace["native_forward_calls"][0]["positions"] = [79]
        c.save("native-call-trace.json", c.trace)
    elif tamper == "operator":
        c.operators[0]["operators"][0]["count"] = 47
        c.save_rows("operators.jsonl", c.operators)
    elif tamper == "loss_objective":
        c.updates[0]["metrics"]["objective_mode"] = "C"
        c.save_rows("updates.jsonl", c.updates)
    elif tamper == "dev_update":
        c.updates[0]["case_id"] = "development-0"
        c.save_rows("updates.jsonl", c.updates)
    elif tamper == "cache_storage":
        row = dict(c.releases[1])
        row["after"] = {**row["after"], "storage_identity": {"sha256": "changed"}}
        c.releases[1] = row
        c.save_rows("cache-releases.jsonl", c.releases)
    elif tamper == "ACK":
        c.save("native-work-ack.json", {})
    if tamper is None:
        result = artifact.validate_native_evidence(
            c.root, value=c.value, worker=c.worker, arm=c.arm, config=c.config
        )
        assert (
            result["native_updates"] == 120
            and result["native_forwards"] == 142
            and result["cache_releases"] == 262
        )
    else:
        with pytest.raises(ValueError):
            artifact.validate_native_evidence(
                c.root, value=c.value, worker=c.worker, arm=c.arm, config=c.config
            )
