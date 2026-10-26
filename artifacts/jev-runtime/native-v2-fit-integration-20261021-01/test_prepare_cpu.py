"""Focused fake-only single-worker integration; no framework, service or actor."""

import builtins
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location("tested_native179_prepare", HERE / "prepare_cpu.py")
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
BOOT = "11111111-2222-4333-8444-555555555555"
ACTOR = {"pid": 987654, "start_ticks": 12345, "pgid": 987654, "sid": 987654}


@pytest.fixture(autouse=True)
def no_framework_or_actor(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {
            "torch",
            "transformers",
            "tokenizers",
            "numpy",
            "peft",
            "playwright",
        }:
            pytest.fail("Heavy import in fake-only test: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(d.subprocess, "Popen", lambda *_a, **_k: pytest.fail("Actor started"))


def test_default_dry_and_run_requires_selected_plan(capsys, monkeypatch):
    monkeypatch.setattr(d, "execute", lambda *_: pytest.fail("Execution in dry"))
    assert d.main([]) == 0
    assert json.loads(capsys.readouterr().out)["processor_calls"] == 0
    with pytest.raises(ValueError, match="Explicit current"):
        d.main(["--run"])
    assert not (HERE / "prepare-actual-01").exists()


def test_current_plan_covers_all_src_and_rejects_missing_new_source(tmp_path):
    w, r, c, cpu, old = d.libraries()
    plan = d.make_plan(BOOT)
    assert plan["workload_component"] == d.WORKLOAD
    assert set(plan["sources"]) >= {
        p.relative_to(d.ROOT).as_posix() for p in (d.ROOT / "src").rglob("*.py")
    }
    assert set(plan["sources"]) >= d.SCRIPTS
    assert "src/s1/browser_native_action_data.py" in plan["sources"]
    assert plan["resource_limits"]["single_worker_AS_soft"] == 8 * 1024**3
    assert plan["environment"]["CUDA_VISIBLE_DEVICES"] == ""
    # The selected plan remains under the repository for the real confined reader.
    path = HERE / "pure-plan-temporary.json"
    try:
        path.write_bytes(c.wire(plan))
        loaded = d.load_plan(path, c.sha(path.read_bytes()))
        assert loaded[-2] == plan
        assert d.main(["--plan", str(path), "--plan-sha256", c.sha(path.read_bytes())]) == 0
        plan["sources"].pop("src/s1/browser_native_action_data.py")
        path.write_bytes(c.wire(plan))
        with pytest.raises(ValueError, match="current source membership"):
            d.load_plan(path, c.sha(path.read_bytes()))
    finally:
        path.unlink(missing_ok=True)


class Tensor(list):
    def tolist(self):
        return copy.deepcopy(list(self))


def fake_data():
    cases, wires = [], []
    for n in range(179):
        case = SimpleNamespace(id=f"EXPLICIT-FAKE-{n}", split="train" if n < 142 else "development")
        cases.append(case)
        wires.append(
            {
                "case_id": case.id,
                "payload": {
                    "model": "original-alias",
                    "state": {"n": n},
                    "questions": {"operation": ["CLICK", "WAIT"], "click_target": ["a", "b"]},
                },
                "teacher_choice": {"operation": "CLICK", "target": "a"},
                "valid_next_targets": {"CLICK": ["a"]},
            }
        )
    return {
        "cases": cases,
        "wires": wires,
        "counts": d.COUNTS,
        "refs": {"EXPLICIT-FAKE-DATA": {"sha256": "a" * 64, "size_bytes": 1}},
        "train_ids": [x.id for x in cases[:142]],
        "dev_ids": [x.id for x in cases[142:]],
        "provenance": {"scope": "EXPLICIT_FAKE_ONLY"},
    }


def fake_compiler(_model, cases, snapshot, *, prompt_format=d.RAW):
    wire = json.loads(snapshot["blobs"]["wire_requests.jsonl"])
    assert wire["case_id"] == cases[0].id
    record = {
        "case_id": cases[0].id,
        "split": cases[0].split,
        "prompt": "EXPLICIT FAKE COMPLETE PROMPT",
        "prompt_tokens": 2,
        "processor_tensors": {"input_ids": [[2, 3, 4, 50]], "attention_mask": [[1, 1, 1, 1]]},
        "logits_positions": [1, 2],
        "target_token_ids": [4, 50],
        "inference_model_identity": "generated-actions-nf4",
    }
    if prompt_format == d.V2:
        record["prompt_format"] = d.V2
    return [
        {
            "record": record,
            "inputs": {k: Tensor(v) for k, v in record["processor_tensors"].items()},
            "positions": Tensor([1, 2]),
            "targets": Tensor([4, 50]),
        }
    ]


def fake_inventory(*_):
    return {
        "labels": {"operation": ["CLICK", "WAIT"], "click_target": ["a", "b"]},
        "canonical_actions": ["a", "b", "wait"],
        "encoded_actions": [[4, 50], [5, 50], [6, 50]],
        "telemetry": {},
    }


def fake_cfg(c, w):
    def compile_cfg(row, payload, _tok, *, prompt_format):
        assert prompt_format in {d.RAW, d.V2} and payload["model"] == "generated-actions-nf4"
        proof = {
            "compiled_row_sha256": c.sha(c.wire(w.snapshot(row))),
            "target_ids": [4, 50],
            "positions": [1, 2],
            "decision_indices": [0],
            "allowed_ids": [[4, 5], [50]],
            "legal_action_count": 3,
            "canonical_inventory_sha256": c.sha(c.wire(fake_inventory()["canonical_actions"])),
            "encoded_inventory_sha256": c.sha(c.wire(fake_inventory()["encoded_actions"])),
        }
        return SimpleNamespace(evidence=lambda: proof)

    return compile_cfg


def saved_rows(*, current=fake_compiler):
    w, _, c, _, _ = d.libraries()
    blobs, calls = {}, []

    def save(label, blob):
        blobs[label] = blob
        return {"path": label, "sha256": c.sha(blob), "size_bytes": len(blob)}

    def read_ref(ref):
        blob = blobs[ref["path"]]
        assert len(blob) == ref["size_bytes"] and c.sha(blob) == ref["sha256"]
        return blob

    data = fake_data()
    refs = w.prepare_rows(
        c,
        data,
        SimpleNamespace(tok="EXPLICIT_FAKE_TOKENIZER"),
        current,
        fake_compiler,
        fake_cfg(c, w),
        SimpleNamespace(check=lambda *args: calls.append(args)),
        save,
        inventory=fake_inventory,
    )
    return w, c, data, refs, blobs, calls, save, read_ref


def test_full179_workload_and_native_inventory_roundtrip():
    w, c, data, refs, blobs, calls, save, read = saved_rows()
    before = c.wire(data["wires"])
    review = d.validate_rows(c, w, data, refs, read)
    assert len(refs) == len(blobs) == 179 and len(calls) == 179 * 6
    assert review["cases"] == 179 and review["training_ready"] is True
    assert review["counts"][d.RAW] == review["counts"][d.V2] == {"COMPILED/CFG_COMPILED": 179}
    assert c.wire(data["wires"]) == before
    # Reseal both file and CFG inventory digest: direct RAW/V2 parity must still refuse.
    row = json.loads(read(refs[0]))
    row["modes"][d.V2]["cfg"]["inventory"]["encoded_actions"][1][0] = 77
    row["modes"][d.V2]["cfg"]["plan"]["encoded_inventory_sha256"] = c.sha(
        c.wire(row["modes"][d.V2]["cfg"]["inventory"]["encoded_actions"])
    )
    refs[0] = save(refs[0]["path"], c.wire(row))
    with pytest.raises(ValueError, match="full grammar"):
        d.validate_rows(c, w, data, refs, read)


def test_refused_rows_remain_truthful_without_shortening179():
    def current(model, cases, snapshot, *, prompt_format):
        if prompt_format == d.V2 and cases[0].id.endswith("-178"):
            raise ValueError("EXPLICIT_FAKE_CONTEXT_REFUSAL")
        return fake_compiler(model, cases, snapshot, prompt_format=prompt_format)

    w, c, data, refs, _, _, _, read = saved_rows(current=current)
    review = d.validate_rows(c, w, data, refs, read)
    assert review["cases"] == 179 and review["training_ready"] is False
    assert review["counts"][d.V2] == {"COMPILED/CFG_COMPILED": 178, "REFUSED": 1}
    with pytest.raises(ValueError, match="Complete179"):
        d.validate_rows(c, w, data, refs[:-1], read)


def gate_fixture():
    _, _, c, cpu, _ = d.libraries()
    unit = {
        "boot_id": BOOT,
        "cgroup_path": c.CGROUP,
        "cgroup_dev": 1,
        "cgroup_inode": 2,
        "caller": ACTOR,
    }
    obs = {"boot_id": BOOT, "MemTotal": 130595860480, "MemAvailable": 40 * 1024**3}
    phases = [
        "startup",
        "source-before",
        "runtime-before",
        "cpu-reference-before-processor-load",
        "processor-after",
        *(["prepare"] * (179 * 6)),
        "final-cpu",
    ]
    gates = [
        {
            "schema_id": c.KIND + "_resource_gate",
            "sequence": i,
            "phase": phase,
            "monotonic": 1.0 + i / 100,
            "elapsed": i / 100,
            "error": None,
            "admitted": True,
            "unit": copy.deepcopy(unit),
            "observation": obs,
            **cpu.resource_formula(obs, BOOT, prospective=True),
            "limits": {"role": "worker", "soft_bytes": 8 * 1024**3, "hard_bytes": 8 * 1024**3},
            "cuda": {"CUDA_VISIBLE_DEVICES": "", "CUDA_initialized": False, "torch_loaded": True},
        }
        for i, phase in enumerate(phases)
    ]
    fake_cpu = SimpleNamespace(
        resource_formula=cpu.resource_formula,
        unit_identity=cpu.unit_identity,
        validate_unit_evidence=lambda row, expected: None,
    )
    return (
        c,
        fake_cpu,
        {"unit": {"boot_id": BOOT, "cgroup_path": c.CGROUP}},
        {"actor": ACTOR, "unit_before": unit, "observation": obs},
        gates,
    )


@pytest.mark.parametrize("drift", ["caller", "inode", "reserve", "limits", "missing", "CUDA"])
def test_gate_validation_rejects_identity_and_resource_drift(drift):
    c, cpu, plan, intent, gates = gate_fixture()
    assert d.validate_gates(c, cpu, plan, intent, b"\n".join(c.wire(g) for g in gates)) == 1080
    if drift == "caller":
        gates[42]["unit"]["caller"]["start_ticks"] += 1
    elif drift == "inode":
        gates[42]["unit"]["cgroup_inode"] += 1
    elif drift == "reserve":
        gates[42]["required_bytes"] -= 1
    elif drift == "limits":
        gates[42]["limits"]["soft_bytes"] //= 4
    elif drift == "missing":
        gates.pop(42)
    else:
        gates[42]["cuda"]["CUDA_initialized"] = True
    with pytest.raises(ValueError):
        d.validate_gates(c, cpu, plan, intent, b"\n".join(c.wire(g) for g in gates))


def test_unit_properties_bind_actual_mainpid_and_whole_unit(monkeypatch):
    _, _, c, _, _ = d.libraries()
    props = {
        "MainPID": str(ACTOR["pid"]),
        "ControlGroup": c.CGROUP,
        "MemoryMax": str(3 * 1024**3),
        "MemoryHigh": "infinity",
        "MemorySwapMax": "0",
        "OOMPolicy": "kill",
        "KillMode": "control-group",
        "RuntimeMaxUSec": "11min",
        "TimeoutStopUSec": "10s",
        "Type": "exec",
        "InvocationID": "a" * 32,
    }

    def fake_run(argv, **kwargs):
        assert argv[:4] == ["systemctl", "--user", "show", c.UNIT]
        assert kwargs == {"capture_output": True, "timeout": 5, "check": True}
        return SimpleNamespace(
            stdout="\n".join(f"{k}={v}" for k, v in props.items()).encode(), stderr=b""
        )

    monkeypatch.setattr(d.subprocess, "run", fake_run)
    assert d.unit_properties(c, ACTOR["pid"]) == props
    props["MainPID"] = "1"
    with pytest.raises(ValueError, match="MainPID"):
        d.unit_properties(c, ACTOR["pid"])


@pytest.mark.parametrize("terminal_failure", [False, True])
def test_resource_refusal_keeps_partial_terminal_and_primary(
    tmp_path, monkeypatch, terminal_failure
):
    w, r, core, _, _ = d.libraries()
    calls = []
    unit = {"boot_id": BOOT, "cgroup_path": core.CGROUP, "caller": ACTOR}
    plan = {
        "output": "EXPLICIT-FAKE-OUTPUT",
        "unit": unit,
        "nonce": "b" * 32,
        "sources": {},
        "lifecycle": "EXPLICIT_FAKE_SINGLE_WORKER",
    }
    obs = {"MemTotal": 130595860480, "MemAvailable": 40 * 1024**3}
    monkeypatch.setattr(d, "ROOT", tmp_path)
    monkeypatch.setattr(
        d, "entry_checks", lambda *_: (ACTOR, unit, obs, {"InvocationID": "a" * 32})
    )

    def save(root, label, blob, **_):
        if label == "terminal.json" and terminal_failure:
            raise OSError("EXPLICIT_FAKE_TERMINAL_FULL")
        path = root / label
        path.write_bytes(blob)

    def load(*_):
        raise ValueError("EXPLICIT_FAKE_NATIVE_RESOURCE_REFUSAL")

    cpu = SimpleNamespace(
        configure_limits=lambda role: calls.append(role),
        save=save,
        caps=lambda *_: None,
        capture_reference=lambda *_: (),
        load_processor=load,
    )
    old = SimpleNamespace(
        parse=lambda _: {"reference_items": {}},
        checked=lambda *_: b"{}",
        fixed=lambda _: {},
        PROCESSOR="FAKE_PROCESSOR",
        Q="FAKE_Q",
        module=lambda _: SimpleNamespace(
            _configure_environment=lambda _: None, _runtime=lambda _: {"files": {}}
        ),
    )
    c = SimpleNamespace(
        **{k: getattr(core, k) for k in ("wire", "require", "KIND")},
        authenticate=lambda *_a, **_k: None,
        Gate=lambda *_: SimpleNamespace(check=lambda *_: None),
    )
    with pytest.raises(ValueError, match="NATIVE_RESOURCE_REFUSAL") as failure:
        d.execute(w, r, c, cpu, old, plan, {"technical": {"config": {}}}, Path("FAKE"), "sha")
    assert calls == ["worker"]
    if terminal_failure:
        assert any("Terminal also failed" in note for note in failure.value.__notes__)
        assert isinstance(failure.value.__cause__, OSError)
    else:
        result = json.loads((tmp_path / plan["output"] / "terminal.json").read_bytes())
        assert result["rows"] == [] and result["data"] == {}
        assert result["complete_pending_external_unit_exit"] is False
        assert result["primary_error"]["message"] == "EXPLICIT_FAKE_NATIVE_RESOURCE_REFUSAL"
        assert result["old_capacity_qualification_attested"] is False


def test_entry_session_is_owned_without_faking_parent_protocol(monkeypatch):
    _, _, c, _, _ = d.libraries()
    plan = {"environment": {}, "unit": {"boot_id": BOOT}}
    path, expected = Path("/EXPLICIT-FAKE-PLAN"), "a" * 64
    monkeypatch.setattr(d.sys, "executable", c.PYTHON)
    monkeypatch.setattr(d.sys, "version_info", (3, 13, 0))
    monkeypatch.setenv("JEV_CPU3G_BOOT", BOOT)
    monkeypatch.setenv("JEV_CPU3G_CGROUP", c.CGROUP)
    calls = []
    monkeypatch.setattr(d.os, "getpid", lambda: ACTOR["pid"])
    monkeypatch.setattr(d.os, "getpgrp", lambda: 1)
    monkeypatch.setattr(d.os, "getsid", lambda _: 1)
    monkeypatch.setattr(d.os, "setsid", lambda: calls.append("setsid"))
    monkeypatch.setattr(d, "unit_properties", lambda *_: {"InvocationID": "a" * 32})
    argv = [
        c.PYTHON,
        "-B",
        "-s",
        str(HERE / "prepare_cpu.py"),
        "--run",
        "--plan",
        str(path),
        "--plan-sha256",
        expected,
    ]
    cpu = SimpleNamespace(
        child_argv=lambda _: argv,
        assert_current_unit=lambda _: {"caller": ACTOR},
        memory_observation=lambda: {"MemAvailable": 10},
        resource_formula=lambda *_a, **_k: {"required_bytes": 10},
    )
    old = SimpleNamespace(identity=lambda _: ACTOR)
    assert d.entry_checks(c, cpu, old, plan, path, expected)[0] == ACTOR
    assert calls == ["setsid"]
    argv.append("--worker")
    with pytest.raises(ValueError, match="single-worker entry argv"):
        d.entry_checks(c, cpu, old, plan, path, expected)
