"""CPU-only metadata preparation; no actual checker callback, actor or GPU."""

import copy
import hashlib
import http.client
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location(
    "_test_remaining_plans", HERE / "prepare_model_plans.py"
)
M = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = M
exec(
    compile(
        (HERE / "prepare_model_plans.py").read_bytes(), str(HERE / "prepare_model_plans.py"), "exec"
    ),
    M.__dict__,
)


@pytest.fixture(scope="module")
def catalog():
    return M.catalogue()


def synthetic_seed(value, base):
    historical = value["historical_closed_helper02_seed"]
    seed = json.loads((M.ROOT / historical["path"]).read_bytes())
    assert (
        hashlib.sha256((M.ROOT / historical["path"]).read_bytes()).hexdigest()
        == historical["sha256"]
    )
    future = value["future_helper_seed"]
    seed["service_directory"] = future["service_directory"]
    seed["private_config"] = future["service_directory"] + "/controller-config.json"
    seed["worker"]["cgroup"] = future["cgroup"]
    seed["worker"]["argv"][-1] = seed["private_config"]
    seed["binding"].update(
        pid=999999, start_ticks=1, pgid=999999, sid=999999, endpoint="http://127.0.0.1:9"
    )
    for key in ("pid", "start_ticks", "pgid", "sid"):
        seed["worker"][key] = seed["binding"][key]
    seed["listener"].update(pid=999999, port=9, inode="1")
    return seed


def test_all_six_exact_identities_and_independent_floors(catalog):
    value, p, base = catalog
    checkpoints = json.loads(p.verify_ref(value["refs"]["checkpoints"])[1])
    assert value["all_six_identities"] == {
        row["model"]: row["registry_identity"] for row in checkpoints["models"]
    }
    assert {m: t["core_floor_bytes"] for m, t in value["templates"].items()} == {
        "d3": 132618706804,
        "d3-flash": 61845970516,
        "d3-mini": 46429111284,
        "d3-nano": 37123944964,
        "d3-lite": 31680801060,
    }
    assert base["components"]["execution"]["sha256"] == M.EXECUTION_SHA
    assert value["future_helper_seed"]["path"].endswith(
        "decision3-q95-owner-20261022-03/owner-03/helper-selection-seed.json"
    )
    assert value["future_helper_seed"]["sha256"] is None


def test_full_model_capacity_refusal_precedes_seed_checker_or_plan(catalog, monkeypatch):
    value, p, base = catalog

    class ForbiddenSeed:
        def __fspath__(self):
            pytest.fail("Full model read a helper seed")

    monkeypatch.setattr(p, "load", lambda *_: pytest.fail("Full model imported execution/checker"))
    result = M.prepare(value, p, base, "d3", ForbiddenSeed(), "0" * 64)
    assert result["score"] is None and result["launch_argv"] is None
    assert result["inference_or_browser_attempted"] is False
    assert result["shortfall_vs_physical_bytes"] == 2022842228
    assert not (M.HERE / "d3").exists()


@pytest.mark.parametrize("model", ["d3-lite", "d3-nano", "d3-mini", "d3-flash"])
def test_model_specific_descriptor_has_exact_registry_and_two_fresh_groups(catalog, model):
    value, p, base = catalog
    selected = M.selected_model(value, base, model)
    seed = synthetic_seed(value, base)
    before = copy.deepcopy(seed)
    descriptors = M.descriptors(p, selected, seed)
    assert seed == before
    for group, descriptor in descriptors.items():
        assert descriptor["co_resident"] == {
            "model": model,
            "resident_file_bytes": value["all_six_identities"][model]["resident_file_bytes"],
            "cgroup": selected["cgroups"][group],
        }
        assert descriptor["binding"] == seed["binding"]
    assert selected["policy"]["known4_envelopes"] == 64
    assert selected["policy"]["public6_envelopes"] == 96
    assert selected["policy"]["max_steps"] == 16
    assert selected["policy"]["fixed_denominator"] == 10
    assert selected["policy"]["native_floor_bytes"] == value["templates"][model]["core_floor_bytes"]


@pytest.mark.parametrize("fault", ["boot", "total", "cgroup", "service", "extra"])
def test_wrong_or_private_seed_refuses(catalog, fault):
    value, p, base = catalog
    selected = M.selected_model(value, base, "d3-lite")
    seed = synthetic_seed(value, base)
    if fault == "boot":
        seed["binding"]["boot_id"] = "wrong-boot"
    elif fault == "total":
        seed["expected_memtotal"] += 4096
    elif fault == "cgroup":
        seed["worker"]["cgroup"] += "-wrong"
    elif fault == "service":
        seed["service_directory"] += "-wrong"
    else:
        seed["api_key"] = "private-forbidden"
    with pytest.raises(ValueError):
        M.descriptors(p, selected, seed)


def test_closed_helper02_path_refuses_before_read(catalog, monkeypatch):
    value, p, base = catalog
    monkeypatch.setattr(p, "checked", lambda *_a, **_kw: pytest.fail("Closed seed read"))
    with pytest.raises(ValueError, match="future helper03"):
        M.prepare(
            value,
            p,
            base,
            "d3-lite",
            M.ROOT / value["historical_closed_helper02_seed"]["path"],
            "0" * 64,
        )


@pytest.mark.parametrize("model", ["d3-lite", "d3-nano", "d3-mini", "d3-flash"])
def test_actual_39_source_roundtrip_keeps_selected_model_and_no_live_calls(
    catalog, monkeypatch, model
):
    value, p, base = catalog
    value = copy.deepcopy(value)
    monkeypatch.setattr(
        http.client, "HTTPConnection", lambda *_a, **_kw: pytest.fail("No live HTTP")
    )
    original_read = Path.read_bytes

    def public_only(path):
        if path.name == "controller-config.json":
            pytest.fail("Private helper config read")
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", public_only)
    with tempfile.TemporaryDirectory(prefix=".cpu-planner-", dir=HERE) as directory:
        folder = Path(directory)
        monkeypatch.setattr(M, "HERE", folder)
        seed = synthetic_seed(value, base)
        seed_path = folder / "synthetic-public-seed.json"
        blob = p.exclusive_json(seed_path, seed)
        value["future_helper_seed"]["path"] = str(seed_path)
        slot = folder / model
        template = value["templates"][model]
        template["plan_path"] = str(slot / "plan.json")
        template["receipt_path"] = str(slot / "prepared.json")
        template["descriptor_paths"] = {g: str(slot / (g + ".json")) for g in ("known4", "public6")}
        result = M.prepare(
            value,
            p,
            base,
            model,
            seed_path,
            hashlib.sha256(blob).hexdigest(),
            expires_after_seconds=90,
        )
        plan = json.loads(Path(template["plan_path"]).read_bytes())
        assert plan["model"] == result["model"] == model
        assert result["registry_identity"] == value["all_six_identities"][model]
        assert result["source_count"] == len(plan["sources"]) == 39
        assert result["selected_plan_validator_passed"] is True
        assert result["actual_live_admission_performed"] is False
        assert result["actual_model_browser_GPU_or_unit_started"] is False
        assert result["score"] is None and result["model_qualified"] is False
        for group, arm in plan["arms"].items():
            descriptor = json.loads((M.ROOT / arm["helper_selection"]["path"]).read_bytes())
            assert descriptor["co_resident"]["model"] == model
            assert (
                descriptor["co_resident"]["resident_file_bytes"]
                == value["all_six_identities"][model]["resident_file_bytes"]
            )
            assert not Path(arm["model_output"]).exists()
        assert not Path(plan["output"]).exists()


def test_default_dry_never_prepares_and_rejects_edge(catalog, monkeypatch, capsys):
    monkeypatch.setattr(M, "prepare", lambda *_a, **_kw: pytest.fail("Dry mode prepared"))
    assert M.main(["--model", "d3-lite"]) == 0
    assert json.loads(capsys.readouterr().out)["actual_started"] is False
    with pytest.raises(SystemExit):
        M.main(["--model", "d3-edge"])
