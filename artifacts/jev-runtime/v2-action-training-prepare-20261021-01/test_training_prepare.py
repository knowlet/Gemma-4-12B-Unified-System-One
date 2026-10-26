"""Focused new-workload tests with explicit fake compilers; no ML or actors."""

import builtins
import copy
import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location(
    "training_prepare_tested", HERE / "training_prepare.py"
)
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
BOOT = "11111111-2222-4333-8444-555555555555"


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
            pytest.fail("Heavy import in explicit fake test: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(d.subprocess, "Popen", lambda *_a, **_k: pytest.fail("Actor started"))


class Tensor(list):
    def tolist(self):
        return copy.deepcopy(list(self))


class Gate:
    def __init__(self):
        self.calls = []

    def check(self, phase, scope):
        self.calls.append((phase, scope))


def data():
    cases, wires = [], []
    for n in range(179):
        split = "train" if n < 142 else "development"
        case = SimpleNamespace(id=f"fake-{n}", split=split)
        wire = {
            "case_id": case.id,
            "payload": {
                "model": "original-alias",
                "state": {"n": n},
                "questions": {"operation": ["CLICK", "WAIT"], "click_target": ["one", "two"]},
            },
            "teacher_choice": {"operation": "CLICK", "target": "one"},
            "valid_next_targets": {"CLICK": ["one"]},
        }
        cases.append(case)
        wires.append(wire)
    return {
        "cases": cases,
        "wires": wires,
        "refs": {"fake": {"sha256": "a" * 64, "size_bytes": 1}},
        "train_ids": [x.id for x in cases[:142]],
        "dev_ids": [x.id for x in cases[142:]],
        "counts": {"train": 142, "development": 37, "total": 179},
        "provenance": {"scope": "EXPLICIT_FAKE_ONLY"},
    }


def compiler(_model, cases, snapshot, *, prompt_format=d.RAW):
    wire = json.loads(snapshot["blobs"]["wire_requests.jsonl"])
    assert len(cases) == 1 and wire["case_id"] == cases[0].id
    values = [2, 3, 4, 50]
    tensors = {"input_ids": [values], "attention_mask": [[1] * 4], "token_type_ids": [[0] * 4]}
    record = {
        "case_id": cases[0].id,
        "split": cases[0].split,
        "prompt": "full prompt",
        "prompt_tokens": 2,
        "processor_tensors": tensors,
        "target_token_ids": [4, 50],
        "logits_positions": [1, 2],
        "inference_model_identity": "generated-actions-nf4",
    }
    if prompt_format == d.V2:
        record.update(
            prompt_format=d.V2, prompt_representation={"format": d.RAW, "requested_format": d.V2}
        )
    return [
        {
            "record": record,
            "inputs": {k: Tensor(v) for k, v in tensors.items()},
            "positions": Tensor([1, 2]),
            "targets": Tensor([4, 50]),
        }
    ]


def cfg(row, payload, tokenizer, *, prompt_format):
    assert tokenizer == "EXPLICIT_FAKE_TOKENIZER"
    assert payload["model"] == row["record"]["inference_model_identity"]
    assert payload["questions"]["click_target"] == ["one", "two"]
    c = d.libraries()[1]
    proof = {
        "compiled_row_sha256": c.sha(c.wire(d.snapshot(row))),
        "target_ids": [4, 50],
        "positions": [1, 2],
        "allowed_ids": [[4, 5], [50]],
        "decision_indices": [0],
        "canonical_inventory_sha256": c.sha(c.wire(inventory()["canonical_actions"])),
        "encoded_inventory_sha256": c.sha(c.wire(inventory()["encoded_actions"])),
        "legal_action_count": 3,
    }
    return SimpleNamespace(evidence=lambda: proof)


def inventory(*_):
    return {
        "labels": {"operation": ["CLICK", "WAIT"], "click_target": ["one", "two"]},
        "canonical_actions": ["action-one", "action-two", "wait"],
        "encoded_actions": [[4, 50], [5, 50], [6, 50]],
        "telemetry": {"legal_action_count": 3},
    }


def save(root, label, blob, **_):
    path = root / label
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return {"path": label, "sha256": d.libraries()[1].sha(blob), "size_bytes": len(blob)}


def read_ref(ref, root):
    blob = (root / ref["path"]).read_bytes()
    assert len(blob) == ref["size_bytes"] and d.libraries()[1].sha(blob) == ref["sha256"]
    return blob


def test_default_dry_and_run_without_plan_never_create_output(capsys, monkeypatch):
    monkeypatch.setattr(d, "parent", lambda *_: pytest.fail("Parent in dry"))
    monkeypatch.setattr(d, "worker", lambda *_: pytest.fail("Worker in dry"))
    assert d.main([]) == 0
    assert json.loads(capsys.readouterr().out)["processor_calls"] == 0
    with pytest.raises(ValueError, match="selected training preparation plan"):
        d.main(["--run"])
    assert not (HERE / "actual-01").exists()


def test_new_union_mechanically_keeps92_original_and_replaces3_plus_new_cfg_codec():
    r, c = d.libraries()
    plan = d.make_plan(BOOT)
    old = c.module(c.WORKER)
    files, technical = d.source_union(r, c, plan, old)
    prior = old.parse(old.checked(technical["technical"]["execution"]))["files"]
    assert len(prior) == 95 and sum(files[p] == item for p, item in prior.items()) == 92
    assert "scripts/jev_cfg_action_loss.py" not in prior and "src/s1/jev_prompt.py" not in prior
    assert all(files[p] == plan["production_sources"][p] for p in d.PRODUCTION)
    assert files[d.LEGACY["path"]]["sha256"] == d.LEGACY["sha256"]
    c.authenticate(files)
    with tempfile.TemporaryDirectory(prefix="pure-plan-", dir=HERE) as folder:
        path = Path(folder) / "plan.json"
        path.write_bytes(c.wire(plan))
        assert d.load_plan(path, c.sha(path.read_bytes()))[2] == plan
        plan["production_sources"]["scripts/jev_browser_fit_actions.py"]["sha256"] = "0" * 64
        path.write_bytes(c.wire(plan))
        with pytest.raises(ValueError, match="pin differs"):
            d.load_plan(path, c.sha(path.read_bytes()))
    assert not (HERE / "actual-01").exists()


def test_full179_pairs_keep_all_inputs_targets_and_cfg_inventory(tmp_path):
    c = d.libraries()[1]
    inputs, gate, rows = data(), Gate(), []
    original = copy.deepcopy(inputs["wires"])
    assert (
        d.prepare_rows(
            c,
            inputs,
            SimpleNamespace(tok="EXPLICIT_FAKE_TOKENIZER"),
            compiler,
            compiler,
            cfg,
            gate,
            lambda p, b: save(tmp_path, p, b),
            rows=rows,
            inventory=inventory,
        )
        is rows
    )
    assert len(rows) == 179 and len(gate.calls) == 179 * 6
    assert inputs["wires"] == original
    first = json.loads(read_ref(rows[0], tmp_path))
    assert first["raw_byte_compatible"] is True
    assert first["modes"][d.RAW]["compiled"]["inputs"]["token_type_ids"] == [[0] * 4]
    assert first["modes"][d.V2]["cfg"]["plan"]["legal_action_count"] == 3
    assert first["modes"][d.V2]["compiled"]["targets"] == [4, 50]
    assert first["modes"][d.RAW]["compiled"]["record"].get("prompt_format") is None


def test_legacy_incompatibility_is_primary_and_keeps_prior_saved_rows(tmp_path):
    c, inputs, rows = d.libraries()[1], data(), []

    def drift(model, cases, snapshot, **kwargs):
        result = compiler(model, cases, snapshot, **kwargs)
        if cases[0].id == "fake-1" and kwargs["prompt_format"] == d.RAW:
            result[0]["inputs"]["attention_mask"][0][0] = 0
        return result

    with pytest.raises(ValueError, match="RAW compiler byte compatibility"):
        d.prepare_rows(
            c,
            inputs,
            SimpleNamespace(tok="EXPLICIT_FAKE_TOKENIZER"),
            drift,
            compiler,
            cfg,
            Gate(),
            lambda p, b: save(tmp_path, p, b),
            rows=rows,
            inventory=inventory,
        )
    assert len(rows) == 1 and (tmp_path / "rows/000.json").exists()
    assert not (tmp_path / "rows/001.json").exists()


def test_measured_context_and_cfg_refusals_do_not_discard_other_cases(tmp_path):
    c, inputs = d.libraries()[1], data()

    def limited(model, cases, snapshot, *, prompt_format=d.RAW):
        if cases[0].id == "fake-0" and prompt_format == d.RAW:
            raise ValueError("Action context/target exceeds native budget; nothing truncated")
        return compiler(model, cases, snapshot, prompt_format=prompt_format)

    def refuse_cfg(*_args, **_kwargs):
        raise ValueError("Explicit fake zero-decision row")

    rows = d.prepare_rows(
        c,
        inputs,
        SimpleNamespace(tok="EXPLICIT_FAKE_TOKENIZER"),
        limited,
        limited,
        refuse_cfg,
        Gate(),
        lambda p, b: save(tmp_path, p, b),
    )
    first = json.loads(read_ref(rows[0], tmp_path))
    assert len(rows) == 179
    assert first["modes"][d.RAW]["outcome"] == "REFUSED"
    assert first["modes"][d.RAW]["cfg"] is None
    assert first["modes"][d.V2]["outcome"] == "COMPILED"
    assert first["modes"][d.V2]["cfg"]["outcome"] == "REFUSED"


def test_unexpected_compiler_failure_is_not_misreported_as_refusal():
    def broken(*_args, **_kwargs):
        raise RuntimeError("Unexpected failure")

    with pytest.raises(RuntimeError, match="Unexpected"):
        d.compile_one(broken, None, data()["cases"][0], data()["wires"][0], mode=d.V2)


def test_wrong_parent_command_refuses_before_limits_or_processor():
    r, c = d.libraries()
    actor = {"pid": 1, "start_ticks": 2, "pgid": 1, "sid": 1}
    old = SimpleNamespace(identity=lambda _pid: actor)
    cpu = SimpleNamespace(
        child_argv=lambda _pid: ["WRONG"],
        configure_limits=lambda *_: pytest.fail("Limits before binding"),
    )
    with tempfile.TemporaryDirectory(prefix="pure-parent-", dir=HERE) as folder:
        output = Path(folder)
        plan = {"output": str(output), "unit": {}}
        (output / "intent.json").write_bytes(
            c.wire({"nonce": "n", "plan_sha256": "p", "parent": actor, "plan_path": "/plan"})
        )
        with pytest.raises(ValueError, match="New parent argv"):
            d.worker(r, c, plan, "p", "n", cpu, old, {}, {})
        assert not (output / "worker").exists()


def test_all179_required_before_any_compiler_or_write():
    inputs = data()
    inputs["cases"].pop()
    with pytest.raises(ValueError, match="All179"):
        d.prepare_rows(
            d.libraries()[1],
            inputs,
            None,
            lambda *_: pytest.fail("Compiler"),
            lambda *_: pytest.fail("Legacy"),
            None,
            Gate(),
            lambda *_: pytest.fail("Write"),
        )


def test_saved179_closure_validator_accepts_full_inventory_and_rejects_corruption():
    r, original = d.libraries()
    policy = original.module(r.CPU)
    inputs = data()
    fake_unit = {"boot_id": BOOT, "cgroup_path": "/EXPLICIT_FAKE_UNIT"}
    c = SimpleNamespace(
        **{k: getattr(original, k) for k in ("wire", "sha", "read", "require", "KIND")},
        module=lambda _ref: SimpleNamespace(load_historical_data=lambda _root: inputs),
    )
    cpu = SimpleNamespace(
        resource_formula=policy.resource_formula,
        unit_identity=policy.unit_identity,
        validate_unit_evidence=lambda unit, expected: original.require(
            {k: unit[k] for k in fake_unit} == expected == fake_unit, "Explicit fake unit mismatch"
        ),
    )
    old = SimpleNamespace(checked=read_ref)
    child = {"pid": 987654, "start_ticks": 12345, "pgid": 987654, "sid": 987654}
    cleanup = {
        "confirmed": True,
        "remaining": [],
        "tracked_PID_start_ticks": {"987654": 12345},
        "tracked_groups": [987654],
    }
    with tempfile.TemporaryDirectory(prefix="pure-closed-", dir=HERE) as folder:
        output, root = Path(folder), Path(folder) / "worker"
        root.mkdir()
        refs = d.prepare_rows(
            c,
            inputs,
            SimpleNamespace(tok="EXPLICIT_FAKE_TOKENIZER"),
            compiler,
            compiler,
            cfg,
            Gate(),
            lambda p, b: save(root, p, b),
            inventory=inventory,
        )
        terminal = {
            "schema_id": d.KIND + "_worker",
            "plan_sha256": "sha",
            "nonce": "nonce",
            "complete": True,
            "primary_error": None,
            "sources": {},
            "child": child,
            "rows": refs,
            "data": {
                k: inputs[k] for k in ("refs", "train_ids", "dev_ids", "counts", "provenance")
            },
            "processor_only": True,
            "native_model_forward_train_HTTP_browser_calls": 0,
            "task_quality_attested": False,
        }
        save(root, "terminal.json", c.wire(terminal))
        for p in ("processor-before.json", "processor-after.json"):
            save(root, p, b'{"EXPLICIT_FAKE":true}')
        unit_evidence = {**fake_unit, "cgroup_dev": 1, "cgroup_inode": 2, "caller": child}
        save(
            output, "intent.json", c.wire({"unit_before": unit_evidence, "mem_total": 130595860480})
        )
        phases = (
            [
                "startup",
                "source-before",
                "runtime-before",
                "cpu-reference-before-processor-load",
                "processor-after",
            ]
            + ["prepare"] * (179 * 6)
            + ["final-cpu"]
        )
        obs = {"MemTotal": 130595860480, "MemAvailable": 40 * 1024**3, "boot_id": BOOT}
        gates = [
            {
                "schema_id": c.KIND + "_resource_gate",
                "sequence": i,
                "phase": phase,
                "admitted": True,
                "elapsed": 1,
                "observation": obs,
                "unit": unit_evidence,
                "cuda": {
                    "CUDA_VISIBLE_DEVICES": "",
                    "CUDA_initialized": False,
                    "torch_loaded": True,
                },
                "limits": {"role": "worker", "soft_bytes": 8 * 1024**3, "hard_bytes": 8 * 1024**3},
                **policy.resource_formula(obs, BOOT, prospective=True),
            }
            for i, phase in enumerate(phases)
        ]
        save(root, "resource-gates.jsonl", b"\n".join(c.wire(g) for g in gates))
        plan = {"unit": fake_unit, "data_source": {"path": "EXPLICIT_FAKE_DATA"}}
        review = d.validate(c, plan, "sha", "nonce", cpu, old, {}, output, cleanup)
        assert review["training_ready"] is True and review["counts"][d.V2] == {
            "COMPILED/CFG_COMPILED": 179
        }
        for key, value in (("caller", {**child, "start_ticks": 99999}), ("cgroup_inode", 999)):
            wrong = copy.deepcopy(gates)
            wrong[0]["unit"][key] = value
            save(root, "resource-gates.jsonl", b"\n".join(c.wire(g) for g in wrong))
            with pytest.raises(ValueError, match="unit inode, exact worker"):
                d.validate(c, plan, "sha", "nonce", cpu, old, {}, output, cleanup)
        save(root, "resource-gates.jsonl", b"\n".join(c.wire(g) for g in gates))
        first = json.loads(read_ref(refs[0], root))
        first["modes"][d.V2]["cfg"]["inventory"]["encoded_actions"][0][0] = 999
        terminal["rows"][0] = save(root, refs[0]["path"], c.wire(first))
        save(root, "terminal.json", c.wire(terminal))
        with pytest.raises(ValueError, match="grammar inventory"):
            d.validate(c, plan, "sha", "nonce", cpu, old, {}, output, cleanup)
        first["modes"][d.V2]["cfg"]["plan"]["encoded_inventory_sha256"] = c.sha(
            c.wire(first["modes"][d.V2]["cfg"]["inventory"]["encoded_actions"])
        )
        terminal["rows"][0] = save(root, refs[0]["path"], c.wire(first))
        save(root, "terminal.json", c.wire(terminal))
        with pytest.raises(ValueError, match="RAW/V2 full grammar"):
            d.validate(c, plan, "sha", "nonce", cpu, old, {}, output, cleanup)
        terminal["rows"].pop()
        save(root, "terminal.json", c.wire(terminal))
        with pytest.raises(ValueError, match="full179"):
            d.validate(c, plan, "sha", "nonce", cpu, old, {}, output, cleanup)


@pytest.mark.parametrize(
    "failure,drift",
    [
        (None, None),
        ({"type": "TimeoutError", "message": "EXPLICIT_FAKE_TIMEOUT"}, None),
        (None, "monitor"),
        (None, "final"),
    ],
)
def test_parent_reuses_owned_lifecycle_and_preserves_primary(failure, drift, monkeypatch):
    r, core = d.libraries()
    policy = core.module(r.CPU)
    actor = {"pid": 987654, "start_ticks": 12345, "pgid": 987654, "sid": 987654}
    unit = {"boot_id": BOOT, "cgroup_path": core.CGROUP}
    calls = []

    def run_owned(cpu, common, output, argv, env, boot, parent, *, wall):
        assert wall == 600 and boot == BOOT and common == "EXPLICIT_FAKE_COMMON"
        assert argv[:5] == [core.PYTHON, "-B", "-s", str(HERE / "training_prepare.py"), "--run"]
        assert env["CUDA_VISIBLE_DEVICES"] == "" and parent["pid"] == actor["pid"]
        calls.append("owned")
        return {
            "primary_error": failure,
            "returncode": 0 if failure is None else -15,
            "secondary_errors": [],
            "cleanup": {"confirmed": True, "remaining": []},
        }, None

    controller = SimpleNamespace(
        run_owned=run_owned, prospective=lambda *_: calls.append("prospective")
    )
    c = SimpleNamespace(
        **{
            k: getattr(core, k)
            for k in ("wire", "require", "PYTHON", "UNIT", "CGROUP", "CONTROLLER")
        },
        module=lambda _ref: controller,
        read=lambda _path: core.wire(
            {"unit": {**unit, "cgroup_dev": 1, "cgroup_inode": 3 if drift == "monitor" else 2}}
        ),
        authenticate=lambda *_a, **_k: calls.append("sources"),
    )
    old = SimpleNamespace(
        module=lambda _ref: "EXPLICIT_FAKE_COMMON", COMMON="fake", identity=lambda _pid: actor
    )

    def bound_save(root, label, blob, **_):
        path = root / label
        path.write_bytes(blob)
        return {"path": label, "sha256": core.sha(blob), "size_bytes": len(blob)}

    unit_calls = []

    def unit_observation(selected):
        unit_calls.append(selected)
        return {
            **selected,
            "cgroup_dev": 1,
            "cgroup_inode": 3 if drift == "final" and len(unit_calls) > 1 else 2,
        }

    cpu = SimpleNamespace(
        assert_current_unit=unit_observation,
        unit_identity=policy.unit_identity,
        memory_observation=lambda: {"MemTotal": 130595860480},
        configure_limits=lambda role: calls.append("limits-" + role),
        save=bound_save,
        SECRET_FREE_ENV=policy.SECRET_FREE_ENV,
        validate_monitor_evidence=lambda *_a, **_k: calls.append("monitor"),
    )
    monkeypatch.setattr(d, "sys", SimpleNamespace(executable=core.PYTHON, version_info=(3, 13)))
    monkeypatch.setattr(
        d,
        "os",
        SimpleNamespace(
            environ={
                **policy.EXPECTED_ENVIRONMENT,
                "JEV_CPU3G_BOOT": BOOT,
                "JEV_CPU3G_CGROUP": core.CGROUP,
            },
            getpid=lambda: actor["pid"],
            getpgrp=lambda: actor["pid"],
            getsid=lambda _: actor["pid"],
        ),
    )
    props = {
        "MainPID": str(actor["pid"]),
        "ControlGroup": core.CGROUP,
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
    monkeypatch.setattr(
        d,
        "subprocess",
        SimpleNamespace(
            run=lambda *_a, **_k: SimpleNamespace(
                stdout="\n".join(k + "=" + v for k, v in props.items()).encode(), stderr=b""
            )
        ),
    )
    monkeypatch.setattr(d, "validate", lambda *_a: {"cases": 179, "EXPLICIT_FAKE_REVIEW": True})
    with tempfile.TemporaryDirectory(prefix="pure-parent-life-", dir=HERE) as folder:
        output = Path(folder) / "output"
        plan = {"unit": unit, "environment": policy.EXPECTED_ENVIRONMENT, "output": str(output)}
        if failure is None and drift is None:
            d.parent(r, c, Path(folder) / "plan.json", "sha", plan, cpu, old, {})
        else:
            with pytest.raises(ValueError, match="partial facts retained"):
                d.parent(r, c, Path(folder) / "plan.json", "sha", plan, cpu, old, {})
        receipt = json.loads((output / "parent-terminal.json").read_bytes())
        if drift is None:
            assert receipt["primary_error"] == failure
        else:
            assert "unit inode" in receipt["primary_error"]["message"]
        assert receipt["complete_pending_external_unit_exit"] is (failure is None and drift is None)
        assert receipt["InvocationID"] == "a" * 32 and calls.count("owned") == 1
        assert ("monitor" in calls) is (failure is None)
