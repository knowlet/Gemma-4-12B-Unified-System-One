"""Pure/fake regressions for the new unit binding; no processor or actor run."""

import ast
import builtins
import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).absolute().parent
ORIGINAL = HERE.with_name("public-revisit-capacity-execution-20261020-01")
GIB = 1024**3
UNIT = {
    "boot_id": "explicit-fake-boot",
    "cgroup_path": "/user.slice/user-1001.slice/user@1001.service/app.slice/jev-cpu3g-20261020-02.service",
}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


w = load(HERE / "revisit_capacity_worker.py", "revisit_3g_worker_tested")
c = load(HERE / "revisit_capacity_controller.py", "revisit_3g_controller_tested")


@pytest.fixture(autouse=True)
def no_frameworks(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {
            "torch",
            "transformers",
            "numpy",
            "tokenizers",
            "safetensors",
            "peft",
        }:
            raise AssertionError("Pure test attempted framework import: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_full_math_and_closed_validation_source_bodies_unchanged():
    for filename, names in (
        ("revisit_capacity_worker.py", ("slots", "prepare_row", "run_rows")),
        ("revisit_capacity_controller.py", ("validate_worker",)),
    ):
        old = (ORIGINAL / filename).read_text()
        new = (HERE / filename).read_text()
        for name in names:
            before = next(n for n in ast.parse(old).body if getattr(n, "name", None) == name)
            after = next(n for n in ast.parse(new).body if getattr(n, "name", None) == name)
            assert ast.get_source_segment(old, before) == ast.get_source_segment(new, after)
    assert c.KIND.endswith("_3g_v2") and w.KIND.endswith("_3g_v2")
    assert c.PROFILE == w.PROFILE == "composed-visible-hover-entries-settled-revisit-v1"
    assert c.LIMITS["tracked_parent_and_owned_RSS"] == 3 * GIB
    assert c.LIMITS["whole_unit_memory_max"] == 3 * GIB
    assert c.LIMITS["whole_unit_swap_max"] == 0
    assert c.LIMITS["parent_AS_soft"] == 2 * GIB
    assert c.LIMITS["worker_AS_soft_and_hard"] == 8 * GIB


def test_new_fork_all75_full_payloads_original_schema_and_prefix_with_fake_processor(
    tmp_path, monkeypatch
):
    # Reuse the original meaningful regression instead of copying its recipes.
    old = load(ORIGINAL / "test_revisit_capacity_worker.py", "revisit_original_fake_regression")
    monkeypatch.setattr(old, "w", w)
    deps = old.deps.__wrapped__(monkeypatch)
    old.test_all75_actual_closed_payloads_original_schema_and_prefix_with_fake_processor(
        tmp_path, monkeypatch, deps
    )


def fake_cpu(*, available=29_340_397_568, wrong_unit=False):
    old = w.module(w.ORIGINAL_CPU)
    obs = {"MemTotal": 130_595_860_480, "MemAvailable": available, "boot_id": UNIT["boot_id"]}
    evidence = {**UNIT, "memory_max_bytes": 3 * GIB, "memory_swap_max_bytes": 0}
    rows = []

    def formula(observation, boot, *, prospective):
        result = old.resource_formula(observation, boot, prospective=False)
        result["required_bytes"] = result["reserve_bytes"] + (3 * GIB if prospective else 0)
        return result

    def unit(expected=None):
        assert expected is None or expected == UNIT
        if wrong_unit:
            raise ValueError("Explicit fake unit identity mismatch")
        return copy.deepcopy(evidence)

    def validate(value, expected):
        if value != evidence or expected != UNIT:
            raise ValueError("Explicit fake saved unit evidence differs")
        return value

    def append(_output, _label, value, count):
        rows.append(copy.deepcopy(value))
        return count + len(w.canonical(value)) + 1

    return SimpleNamespace(
        memory_observation=lambda: copy.deepcopy(obs),
        resource_formula=formula,
        assert_current_unit=unit,
        validate_unit_evidence=validate,
        verify_limits=lambda role: {"role": role, "soft_bytes": 8 * GIB, "hard_bytes": 8 * GIB},
        cuda_state=lambda: {
            "CUDA_VISIBLE_DEVICES": "",
            "torch_loaded": False,
            "CUDA_initialized": False,
        },
        append_owned=append,
        rows=rows,
    )


@pytest.mark.parametrize("available,admitted", [(29_340_397_568, True), (29_340_397_567, False)])
def test_parent_prospective_keeps_reserve_and_new_whole_unit_cap(available, admitted):
    cpu = fake_cpu(available=available)
    if not admitted:
        with pytest.raises(ValueError, match="unchanged host reserve"):
            c.prospective(cpu, UNIT["boot_id"], 130_595_860_480)
    else:
        result = c.prospective(cpu, UNIT["boot_id"], 130_595_860_480)
        assert result["reserve_bytes"] == 26_119_172_096
        assert result["required_bytes"] == 29_340_397_568
        assert result["swap_or_resident_credit_bytes"] == 0
        assert result["unit"]["memory_max_bytes"] == 3 * GIB


@pytest.mark.parametrize("wrong_unit", [False, True])
def test_each_worker_gate_requires_hard_unit_evidence(monkeypatch, wrong_unit):
    cpu = fake_cpu(wrong_unit=wrong_unit)
    monkeypatch.setattr(w, "save", lambda *a, **k: None)
    monkeypatch.setattr(w, "whole_caps", lambda *a, **k: None)
    config = {"unit": UNIT, "boot_id": UNIT["boot_id"], "mem_total": 130_595_860_480}
    gate = w.Gate(HERE, config, cpu, clock=lambda: 100.0)
    if wrong_unit:
        with pytest.raises(ValueError, match="unit identity mismatch"):
            gate.check("prepare", "cpu")
        assert cpu.rows[-1]["admitted"] is False and cpu.rows[-1]["error"] is not None
    else:
        row = gate.check("prepare", "cpu")
        assert row["admitted"] is True and row["unit"]["memory_max_bytes"] == 3 * GIB


def saved_gates(cpu):
    phases = [
        "startup",
        "source-before",
        "runtime-before",
        "cpu-reference-before-processor-load",
        "processor-after",
    ]
    phases += ["prepare"] * 75 + ["final-cpu"]
    observation = cpu.memory_observation()
    return [
        {
            "schema_id": w.KIND + "_resource_gate",
            "sequence": i,
            "phase": phase,
            "scope": "cpu",
            "monotonic": 100.0 + i,
            "worker_elapsed_seconds": float(i),
            "admitted": True,
            "error": None,
            "observation": observation,
            **cpu.resource_formula(observation, UNIT["boot_id"], prospective=True),
            "limits": cpu.verify_limits("worker"),
            "cuda": cpu.cuda_state(),
            "unit": cpu.assert_current_unit(UNIT),
        }
        for i, phase in enumerate(phases)
    ]


@pytest.mark.parametrize("changed", [False, True])
def test_offline_closed_gate_validator_checks_every_unit_row(monkeypatch, changed):
    cpu = fake_cpu()
    original = w.module
    monkeypatch.setattr(w, "module", lambda name: cpu if name == w.CPU else original(name))
    rows = saved_gates(cpu)
    if changed:
        rows[40]["unit"]["memory_swap_max_bytes"] = 1
    blob = b"\n".join(w.canonical(row) for row in rows) + b"\n"
    config = {"boot_id": UNIT["boot_id"], "mem_total": 130_595_860_480, "unit": UNIT}
    if changed:
        with pytest.raises(ValueError, match="saved unit evidence"):
            w.validate_gates(blob, config)
    else:
        assert w.validate_gates(blob, config)["rows"] == 81


def test_source_union_retains_original95_and_original_cpu_while_adding_new_component():
    cohort = w.parse(w.checked(w.fixed(w.COHORT)))
    technical = w.parse(w.checked(w.fixed(w.TECHNICAL)))
    config = {
        key: {
            "path": str((HERE / name).relative_to(w.ROOT)),
            "sha256": w.sha((HERE / name).read_bytes()),
            "size_bytes": (HERE / name).stat().st_size,
        }
        for key, name in (
            ("worker_source", "revisit_capacity_worker.py"),
            ("parent_source", "revisit_capacity_controller.py"),
        )
    }
    union = w.source_union(config, cohort, technical)
    old95 = w.parse(w.checked(technical["technical"]["execution"]))["files"]
    assert len(old95) == 95 and all(union[path] == ref for path, ref in old95.items())
    assert w.ORIGINAL_CPU in union and w.CPU in union
    assert w.CPU != w.ORIGINAL_CPU


@pytest.mark.parametrize("change", ["none", "extra-key", "wrong-boot"])
def test_new_config_requires_exact_sameboot_unit_metadata(tmp_path, monkeypatch, change):
    config = {
        "schema_id": w.KIND,
        **{
            key: w.fixed(label)
            for key, label in (
                ("cohort", w.COHORT),
                ("technical_plan", w.TECHNICAL),
                ("processor_plan", w.PROCESSOR),
            )
        },
        **{
            key: {
                "path": str((HERE / name).relative_to(w.ROOT)),
                "sha256": w.sha((HERE / name).read_bytes()),
                "size_bytes": (HERE / name).stat().st_size,
            }
            for key, name in (
                ("worker_source", "revisit_capacity_worker.py"),
                ("parent_source", "revisit_capacity_controller.py"),
            )
        },
        "owner": {"pid": 1, "start_ticks": 2, "pgid": 1, "sid": 1},
        "nonce": "1" * 32,
        "boot_id": UNIT["boot_id"],
        "mem_total": 130_595_860_480,
        "output": str(HERE / "actual-fake-not-created"),
        "unit": copy.deepcopy(UNIT),
    }
    if change == "extra-key":
        config["unit"]["unreviewed"] = True
    elif change == "wrong-boot":
        config["unit"]["boot_id"] = "different-boot"
    path = tmp_path / "fake-config.json"
    blob = w.canonical(config)
    original = w.read
    monkeypatch.setattr(
        w, "read", lambda p, maximum=w.MAX_FILE: blob if Path(p) == path else original(p, maximum)
    )
    if change == "none":
        assert w.load_config(path, w.sha(blob))[0]["unit"] == UNIT
    else:
        with pytest.raises(ValueError, match="whole-unit identity"):
            w.load_config(path, w.sha(blob))
