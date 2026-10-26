"""Reuse owned tiny CPU loopback fixture; no real Q95/model/unit calls."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

HERE = Path(__file__).absolute().parent
FIXTURE = HERE.parent / "decision3-local-helper-budget-20261022-01/test_live_helper.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


support = load("_gemma_q95_shared_fixture", FIXTURE)
g = load("_gemma_helper_test", HERE / "gemma_helper.py")
fake = support.fake


@pytest.fixture
def selected(fake, monkeypatch):
    p = support.m
    monkeypatch.setattr(g, "load_parent", lambda: p)
    seed = copy.deepcopy(fake["value"])
    del seed["co_resident"]
    seed_path = p.ROOT / "selected/seed.json"
    support.json_file(seed_path, seed)
    adapter = p.ROOT / "adapter/manifest.json"
    support.json_file(adapter, {"synthetic": "selected native V2 adapter"})
    monkeypatch.setattr(g, "ADAPTER_SHA", p.pin(adapter)["sha256"])
    unit = "jev-current-nf4-v2-q95-20261022-01.service"
    cgroup = f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{unit}"
    co = {
        "profile": g.NATIVE_PROFILE,
        "unit": unit,
        "cgroup": cgroup,
        "memory_max_bytes": 40 * p.GIB,
        "adapter_manifest": {"path": adapter.relative_to(p.ROOT).as_posix(), **p.pin(adapter)},
    }
    value = {
        "schema_id": g.SCHEMA,
        "helper_seed_ref": {"path": seed_path.relative_to(p.ROOT).as_posix(), **p.pin(seed_path)},
        "native_co_resident": co,
    }
    original_process = p.process
    monkeypatch.setattr(
        p,
        "process",
        lambda pid: (
            {**original_process(pid), "cgroup": cgroup}
            if pid == os.getpid()
            else original_process(pid)
        ),
    )
    monkeypatch.setattr(
        p,
        "cgroup_memory",
        lambda group: [
            {
                "path": group,
                "limit": 40 * p.GIB if group == cgroup else 80 * p.GIB,
                "current": p.GIB,
                "events": {"oom": 0, "oom_kill": 0},
            }
        ],
    )
    cgroot = p.ROOT / "cgroups"
    swap = cgroot / cgroup.lstrip("/") / "memory.swap.max"
    swap.parent.mkdir(parents=True)
    swap.write_text("0\n")
    monkeypatch.setattr(p, "CGROOT", cgroot)

    def forbid(*_args, **_kwargs):
        raise AssertionError("D3 CORE must never be called for Gemma")

    monkeypatch.setattr(p, "memory_decisions", forbid)

    def build(candidate=None):
        blob = p.encode(value if candidate is None else candidate)
        return g.GemmaHelperChecker(blob, hashlib.sha256(blob).hexdigest(), fake["receipts"])

    return {
        "build": build,
        "value": value,
        "seed": seed,
        "seed_path": seed_path,
        "swap": swap,
        "adapter": adapter,
        "p": p,
    }


def test_real_owned_loopback_preserves_q95_auth_private_connection_and_separate_gemma_floor(
    fake, selected
):
    checker = selected["build"]()
    result = checker(checker.binding, "before-native-constructor")
    assert result["helper_admitted"] and result["co_resident_admitted"]
    receipt = json.loads((fake["receipts"] / "helper-check-0000.json").read_bytes())
    assert [r["status"] for r in receipt["authentication"]["requests"]] == [500, 422]
    assert receipt["memory_before"]["co_resident"]["profile"] == g.NATIVE_PROFILE
    assert receipt["memory_after"]["co_resident"]["native_forward_gates_remain_external"]
    connection = checker.text_connection()
    assert connection["api_key"] == fake["token"]
    support.no_private_data(fake["token"], selected["p"].encode(receipt))
    assert (
        receipt["whole_tree_exit_confirmed"] is False
        and receipt["continuation_authorized"] is False
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_id", "decision3-q95-live-helper-selection-v1"),
        ("native_co_resident", {"model": "d3-edge"}),
    ],
)
def test_d3_descriptor_cannot_be_relabelled_as_gemma(selected, key, value):
    candidate = copy.deepcopy(selected["value"])
    candidate[key] = value
    with pytest.raises(Exception):
        selected["build"](candidate)


@pytest.mark.parametrize(
    "key,value",
    [
        ("profile", "d3-edge"),
        ("memory_max_bytes", 80 * 1024**3),
        ("unit", "foreign.service"),
        ("cgroup", "/foreign"),
    ],
)
def test_wrong_native_owner_or_resources_refuse_before_callback(selected, key, value):
    candidate = copy.deepcopy(selected["value"])
    candidate["native_co_resident"][key] = value
    with pytest.raises(Exception):
        selected["build"](candidate)


def test_adapter_source_drift_refuses_constructor(selected):
    selected["adapter"].write_text("changed")
    with pytest.raises(Exception):
        selected["build"]()


def test_original_q95_source_drift_refuses_constructor(fake, selected):
    (fake["source"] / "source-000.py").write_text("changed")
    with pytest.raises(Exception):
        selected["build"]()


def test_original_q95_host_floor_is_independent_of_native_memcg_headroom(selected, monkeypatch):
    p = selected["p"]
    checker = selected["build"]()
    monkeypatch.setattr(
        p,
        "cgroup_memory",
        lambda group: [
            {
                "path": group,
                "limit": 40 * p.GIB,
                "current": 40 * p.GIB - 1,
                "events": {"oom": 0, "oom_kill": 0},
            }
        ],
    )
    memory = checker._memory()
    assert memory["helper"]["admitted"] and memory["co_resident"]["admitted"]
    assert memory["cuda_observed"] is False


@pytest.mark.parametrize("problem", ["host", "cap", "swap", "own_cgroup"])
def test_fresh_original_host_and_native_cap_swap_process_gates_refuse(
    selected, monkeypatch, problem
):
    p = selected["p"]
    checker = selected["build"]()
    if problem == "host":
        monkeypatch.setattr(
            p,
            "host_memory",
            lambda: {
                "boot_id": checker.binding["boot_id"],
                "MemTotal": checker._value["expected_memtotal"],
                "MemAvailable": 1,
            },
        )
    elif problem == "cap":
        monkeypatch.setattr(
            p,
            "cgroup_memory",
            lambda group: [{"path": group, "limit": 80 * p.GIB, "current": 0, "events": {}}],
        )
    elif problem == "swap":
        selected["swap"].write_text("1\n")
    else:
        monkeypatch.setattr(p, "process", lambda _pid: {"cgroup": "/foreign"})
    with pytest.raises(Exception):
        checker._memory()
