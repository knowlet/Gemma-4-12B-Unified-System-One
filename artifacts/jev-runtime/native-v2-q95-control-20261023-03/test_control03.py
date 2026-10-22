"""Pure CPU regression and explicitly synthetic, non-live metadata fixtures."""

import copy
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from checked_source import module  # noqa: E402 - explicitly select local audited loader


def local(name):
    path = HERE / name
    spec = importlib.util.spec_from_file_location("_pin_successor_" + path.stem, path)
    selected = importlib.util.module_from_spec(spec)
    exec(compile(path.read_bytes(), str(path), "exec"), selected.__dict__)
    return selected


o = local("launch_browser.py")
native = module(
    "scripts/jev_native_v2_browser.py",
    "c513dc5bcb091de4fd0d4bd7a0776078e82be16106328b6de9d8424e32f2ee11",
    "_pin_test_actual_native",
)


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    output, control_dir = tmp_path / "browser", tmp_path / "q95-control"
    output.mkdir()
    control_dir.mkdir()
    receipt_path = output / "receipt.json"
    receipt_path.write_text('{"stage":"complete_pending_external_closure","attempts":10}')
    # Real checked native implementation, never a mirror of owner.pin.
    actual_native_pin = native.pin(receipt_path)
    assert set(actual_native_pin) == {"sha256", "size_bytes"}
    monkeypatch.setattr(o, "ROOT", tmp_path)
    monkeypatch.setattr(o.owner, "ROOT", tmp_path)
    binding, calls = {"same": "Q95"}, []
    monkeypatch.setattr(o, "original_closed", lambda *_: calls.append("native-whole-closure"))
    monkeypatch.setattr(o, "descriptor", lambda _: ({}, {"binding": binding}))
    monkeypatch.setattr(o, "helper_static", lambda _: calls.append("Q95-static-closure"))
    monkeypatch.setattr(o.owner, "run_paths", lambda _: (tmp_path, output))
    groups = []
    for group in ("known4", "public6"):
        directory = output / group
        directory.mkdir()
        (directory / "external-q95-request-index.json").write_text("{}")
        groups.append({"group_directory": str(directory), "external_q95_requests": 1})
    control = {
        "profile": "jev-current-nf4-v7-q95-browser-v1",
        "stage": "complete_pending_external_closure",
        "primary_error": None,
        "selection_sha256": "a" * 64,
        "helper_binding": binding,
        "native_receipt": actual_native_pin,
        "group_indexes": groups,
        "native_journal_covers_external_helper": False,
        "external_owned_closure_required": True,
    }
    path = control_dir / "receipt.json"
    path.write_text(json.dumps(control))
    return {
        "output": output,
        "control": control,
        "path": path,
        "plan": {"helper_selection": {"sha256": "a" * 64}},
        "calls": calls,
    }


def test_actual_native_pin_two_fields_pass_completed_successor(evidence):
    f = evidence
    assert set(o.owner.pin(f["output"] / "receipt.json")) == {"path", "sha256", "size_bytes"}
    o.verify_closed({}, {}, f["plan"], f["output"])
    assert f["calls"] == ["native-whole-closure", "Q95-static-closure"]


def test_same_actual_native_pin_is_rejected_by_frozen_old_bug(evidence, monkeypatch):
    old = module(
        "artifacts/jev-runtime/native-v2-q95-browser-control-20261022-01/launch_browser.py",
        "d38ea38104bbfc384d108ca917913244af614083f95d7464434f2411d5ce5992",
        "_pin_test_old_bug",
    )
    f = evidence
    monkeypatch.setattr(old, "original_closed", lambda *_: None)
    monkeypatch.setattr(
        old, "descriptor", lambda _: ({}, {"binding": f["control"]["helper_binding"]})
    )
    monkeypatch.setattr(old, "helper_static", lambda _: None)
    monkeypatch.setattr(old.owner, "ROOT", f["output"].parent)
    monkeypatch.setattr(old.owner, "run_paths", lambda _: (f["output"].parent, f["output"]))
    with pytest.raises(ValueError, match="same-helper control receipt"):
        old.verify_closed({}, {}, f["plan"], f["output"])


@pytest.mark.parametrize(
    "mutation", ["hash", "size", "extra_path", "missing", "primary", "incomplete", "binding"]
)
def test_forged_incomplete_or_wrong_binding_receipt_still_refuses(evidence, mutation):
    f, value = evidence, copy.deepcopy(evidence["control"])
    if mutation == "hash":
        value["native_receipt"]["sha256"] = "f" * 64
    elif mutation == "size":
        value["native_receipt"]["size_bytes"] += 1
    elif mutation == "extra_path":
        value["native_receipt"]["path"] = "invented"
    elif mutation == "missing":
        del value["native_receipt"]["size_bytes"]
    elif mutation == "primary":
        value["primary_error"] = {"type": "infrastructure"}
    elif mutation == "incomplete":
        value["group_indexes"] = value["group_indexes"][:1]
    else:
        value["helper_binding"] = {"different": "helper"}
    f["path"].write_text(json.dumps(value))
    with pytest.raises(ValueError):
        o.verify_closed({}, {}, f["plan"], f["output"])


def test_original_unit_closure_refusal_remains_authoritative(evidence, monkeypatch):
    def refused(*_):
        raise ValueError("original closure unknown")

    monkeypatch.setattr(o, "original_closed", refused)
    with pytest.raises(ValueError, match="original closure unknown"):
        o.verify_closed({}, {}, evidence["plan"], evidence["output"])
    assert not evidence["calls"]


def test_private_original_owner_ownership_functions_unchanged():
    assert o.owner.execute.__code__.co_filename == str(o.PARENT)
    assert o.owner.readback.__code__.co_filename == str(o.PARENT)
    assert o.owner.CAP == 40 * 1024**3
    assert o.owner.FORMATS == {"v2-q95": "shared-elements-v2"}


def test_gemma_identity_successor_inverse_is_exact_single_unit_literal():
    old = ROOT / "artifacts/jev-runtime/native-v2-q95-pin-successor-20261022-01/gemma_helper.py"
    assert (HERE / "gemma_helper.py").read_bytes().replace(
        b"jev-current-nf4-v2-q95-20261023-03.service", b"jev-current-nf4-v2-q95-20261022-02.service"
    ) == old.read_bytes()


def test_reused_driver_boundary_and_entry_sources_unchanged_and_dry(capsys):
    driver, boundary, entry = (
        local("control_driver.py"),
        local("policy_boundary.py"),
        local("browser_entry.py"),
    )
    assert driver.selected.HERE == HERE
    assert driver.selected.PROFILE == "jev-current-nf4-v7-q95-browser-v1"
    assert boundary.PolicyBoundary.__module__ == "_native_q95_pin_selected_boundary"
    assert driver.main([]) == entry.main([]) == 0
    assert "DRY_NO_MODEL_OR_BROWSER" in capsys.readouterr().out


def test_no_frameworks_imported_to_exercise_selected_native_pin():
    assert not any(name in sys.modules for name in ("torch", "transformers", "fastapi", "httpx"))


def test_checked_loader_refuses_changed_bytes_before_executing(tmp_path, monkeypatch):
    import checked_source

    path = tmp_path / "tiny.py"
    path.write_text("raise AssertionError('must never run')\n")
    monkeypatch.setattr(checked_source, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="bytes differ"):
        checked_source.module("tiny.py", "0" * 64, "_never_execute_wrong_source")
    assert hashlib.sha256(path.read_bytes()).hexdigest() != "0" * 64


@pytest.fixture
def metadata_factory(tmp_path, monkeypatch):
    """Synthetic impossible-live identity; real small source pins, no actor IO."""
    prepare = local("prepare_control.py")
    real_root, real_here = ROOT, HERE
    new_here = tmp_path / HERE.relative_to(ROOT)
    new_here.mkdir(parents=True)

    def copy_ref(relative):
        source, target = real_root / relative, tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        data = target.read_bytes()
        return {
            "path": relative,
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }

    def tiny_ref(relative, value=b"# synthetic ordinary metadata fixture\n"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value)
        return {
            "path": relative,
            "sha256": hashlib.sha256(value).hexdigest(),
            "size_bytes": len(value),
        }

    for name in prepare.SOURCE_NAMES:
        copy_ref((real_here / name).relative_to(real_root).as_posix())
    parent = o.PARENT.relative_to(real_root).as_posix()
    refs = [copy_ref(parent), copy_ref("scripts/jev_native_v2_browser.py")]
    refs += [tiny_ref(o.owner.GUARD_SOURCE), tiny_ref(o.owner.GUARD_INDEX)]
    for label in (
        *o.REUSED_SOURCES,
        "artifacts/jev-runtime/native-v2-q95-browser-control-20261022-01/launch_browser.py",
        "artifacts/jev-runtime/native-v2-q95-browser-control-20261022-01/gemma_helper.py",
        prepare.OBSERVER,
        prepare.OBSERVER_AUTHORITY,
        prepare.PARENT,
    ):
        copy_ref(label)
    # Never read an old live seed. These zero process IDs and impossible host
    # total explicitly prevent this fixture from representing a live selection.
    source_root = real_root / "artifacts/jev-runtime/decision3-q95-frozen-root-20261022-01"
    seed = {
        "schema_id": "decision3-q95-live-helper-selection-v1",
        "binding": {
            "profile": "field-syntax-preserving-base-helper-v1",
            "qualification_sha256": o.Q95_SHA,
            "source_count": 95,
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "pid": 0,
            "start_ticks": 0,
            "pgid": 0,
            "sid": 0,
            "endpoint": "http://127.0.0.1:1",
            "seconds": 20,
            "tokens": 128,
            "context": 8192,
        },
        "worker": {
            "pid": 0,
            "start_ticks": 0,
            "pgid": 0,
            "sid": 0,
            "boot_id": "SYNTHETIC-NOT-LIVE",
            "exe": "/SYNTHETIC-NOT-LIVE",
            "argv": [],
            "cgroup": "/SYNTHETIC-NOT-LIVE",
        },
        "listener": {"pid": 0, "address": "127.0.0.1", "port": 1, "inode": "0"},
        "service_directory": str(tmp_path / "SYNTHETIC-NOT-LIVE"),
        "private_config": str(tmp_path / "SYNTHETIC-NOT-LIVE/controller-config.json"),
        "interpreter": "/SYNTHETIC-NOT-LIVE",
        "source_root": str(tmp_path / source_root.relative_to(real_root)),
        "qualification": copy_ref(
            "artifacts/jev-runtime/public-field-syntax-helper-20261018-01/qualify-01/qualification.json"
        ),
        "execution": copy_ref(
            (source_root / "support/execution95.json").relative_to(real_root).as_posix()
        ),
        "runtime": copy_ref(
            (source_root / "support/runtime348.json").relative_to(real_root).as_posix()
        ),
        "registry": {"fixture": "SYNTHETIC-NOT-LIVE"},
        "expected_memtotal": 1,
    }
    for key, digest in o.Q95_MANIFESTS.items():
        assert seed[key]["sha256"] == digest
    execution = json.loads((real_root / seed["execution"]["path"]).read_bytes())["files"]
    for label in execution:
        copy_ref((source_root / label).relative_to(real_root).as_posix())
    seed_blob = json.dumps(seed).encode()
    seed_path = tmp_path / "SYNTHETIC-NOT-LIVE/seed.json"
    tiny_ref("SYNTHETIC-NOT-LIVE/seed.json", seed_blob)
    seed_path.chmod(0o444)
    adapter = "artifacts/synthetic-native-adapter"
    manifest = (
        real_root
        / "artifacts/jev-runtime/native-v2-training-20261022-03/fit-01/adapter/manifest.json"
    )
    refs.append(tiny_ref(adapter + "/manifest.json", manifest.read_bytes()))
    upstream = "artifacts/synthetic-upstream"
    (tmp_path / upstream).mkdir()
    chromium = tmp_path / "synthetic-chromium"
    chromium.write_text("not executed\n")
    # Preserve a full 785-row original population in the actual factory.
    for i in range(785 - len(refs)):
        refs.append(tiny_ref(f"SYNTHETIC-ORIGINAL-SOURCE785/{i:04}.py"))
    assert len(refs) == 785
    base = {
        "schema_id": "original-base-fixture",
        "root": str(tmp_path),
        "boot_id": seed["binding"]["boot_id"],
        "expected_memtotal": seed["expected_memtotal"],
        "adapter": adapter,
        "sources": refs,
        "upstream": upstream,
        "chromium": str(chromium),
        "prompt_format": "shared-elements-v2",
        "unit_memory_max_bytes": 40 * 1024**3,
    }
    base_path = tmp_path / "base-plan.json"
    base_blob = json.dumps(base).encode()
    base_path.write_bytes(base_blob)
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    monkeypatch.setattr(prepare, "HERE", new_here)
    monkeypatch.setattr(prepare, "BASE_PLAN", base_path)
    monkeypatch.setattr(prepare, "BASE_SHA", hashlib.sha256(base_blob).hexdigest())
    monkeypatch.setattr(o, "ROOT", tmp_path)
    monkeypatch.setattr(o, "HERE", new_here)
    monkeypatch.setattr(o, "PARENT", tmp_path / parent)
    monkeypatch.setattr(o.owner, "ROOT", tmp_path)
    monkeypatch.setattr(o.owner, "HERE", new_here)
    monkeypatch.setattr(
        o.owner, "DRIVER", (new_here / "control_driver.py").relative_to(tmp_path).as_posix()
    )
    # Those independent contracts are unchanged and already covered; this
    # integration exercises the actual factory, authenticates all small source
    # refs, and executes actual private owner validation/path checks.
    monkeypatch.setattr(o.owner, "verify_cache_aliases", lambda *_: [])
    monkeypatch.setattr(o, "helper_static", lambda *_: None)
    prepare.prepare(seed_path, hashlib.sha256(seed_blob).hexdigest())
    plan = json.loads((new_here / "launch-plan-01.json").read_bytes())
    assert {r["path"]: r for r in refs}.items() <= {r["path"]: r for r in plan["sources"]}.items()
    assert plan["source_population_counts"]["retained_original_sources"] == 785
    return plan, new_here


def test_actual_metadata_factory_passes_original_run_paths_and_private_owner_validation(
    metadata_factory,
):
    plan, directory = metadata_factory
    assert o.owner.run_paths(plan) == (directory / "v2-q95-03", directory / "v2-q95-03/browser")
    assert o.validate_plan(plan) == plan
    assert plan["attempt"] == "03" and plan["unit"].endswith("20261023-03.service")


@pytest.mark.parametrize("old_attempt", ["01", "02"])
def test_actual_metadata_factory_cannot_join_output_to_historical_directory(
    metadata_factory, old_attempt
):
    plan, directory = metadata_factory
    plan["output"] = (directory / f"v2-q95-{old_attempt}/browser").relative_to(o.ROOT).as_posix()
    with pytest.raises(ValueError, match="owned arm child"):
        o.owner.run_paths(plan)
    with pytest.raises(ValueError, match="owned arm child"):
        o.validate_plan(plan)


def test_preparer_default_dry_and_missing_explicit_seed_write_nothing(capsys):
    prepare = local("prepare_control.py")
    before = sorted(p.name for p in HERE.iterdir())
    assert prepare.main([]) == 0
    assert capsys.readouterr().out.strip() == "DRY_NO_METADATA_OR_ACTOR_WRITES"
    with pytest.raises(ValueError, match="Explicit fresh selected helper seed"):
        prepare.main(["--prepare"])
    assert sorted(p.name for p in HERE.iterdir()) == before
    assert not (HERE / "launch-plan-01.json").exists()
    assert not (HERE / "helper-selection-01.json").exists()
    assert not (HERE / "v2-q95-03").exists()


def test_preparer_exact_slot_inverse_to_current7production():
    old = ROOT / "artifacts/jev-runtime/native-v2-q95-pin-successor-20261022-01"
    fresh = (HERE / "prepare_control.py").read_bytes()
    for current, prior, count in (
        (b"v2-q95-03", b"v2-q95-02", 3),
        (
            b"jev-current-nf4-v2-q95-20261023-03.service",
            b"jev-current-nf4-v2-q95-20261022-02.service",
            1,
        ),
        (b'attempt="03"', b'attempt="02"', 1),
    ):
        assert fresh.count(current) == count
        fresh = fresh.replace(current, prior)
    assert fresh == (old / "prepare_control.py").read_bytes()
    for name in (
        "checked_source.py",
        "control_driver.py",
        "policy_boundary.py",
        "browser_entry.py",
    ):
        assert (HERE / name).read_bytes() == (old / name).read_bytes()


def test_checked_parent_unit_date_mapping_exact_inverse_only():
    old_literal = b"f\"jev-current-nf4-{plan['arm']}-20261022-{plan['attempt']}.service\""
    new_literal = b"f\"jev-current-nf4-{plan['arm']}-20261023-{plan['attempt']}.service\""
    original = o.PARENT.read_bytes()
    assert hashlib.sha256(original).hexdigest() == o.PARENT_SHA
    assert original.count(old_literal) == 1 and new_literal not in original
    mapped = original.replace(old_literal, new_literal)
    assert mapped.replace(new_literal, old_literal) == original
    original_namespace = {"__file__": str(o.PARENT), "__name__": "_test_original_unit_date"}
    exec(compile(original, str(o.PARENT), "exec"), original_namespace)
    # Private validation differs by a single date string constant; execute and
    # kernel/cgroup readback implementations have byte-identical bytecode.
    assert (
        o.original_validate.__code__.co_code == original_namespace["validate_plan"].__code__.co_code
    )
    constants = tuple(
        c.replace("20261023", "20261022") if isinstance(c, str) else c
        for c in o.original_validate.__code__.co_consts
    )
    assert constants == original_namespace["validate_plan"].__code__.co_consts
    for name in ("execute", "readback", "run_paths", "verify_closed", "current_boot", "reserve"):
        current = o.original_closed if name == "verify_closed" else getattr(o.owner, name)
        assert current.__code__.co_code == original_namespace[name].__code__.co_code
        assert current.__code__.co_consts == original_namespace[name].__code__.co_consts


def test_actual_metadata_factory_stale_unit_date_refuses(metadata_factory):
    plan, _ = metadata_factory
    plan["unit"] = "jev-current-nf4-v2-q95-20261022-03.service"
    with pytest.raises(ValueError, match="Explicit owned unit"):
        o.validate_plan(plan)


def test_fake_metadata_identity_never_claims_liveness(metadata_factory):
    plan, _ = metadata_factory
    descriptor = json.loads((o.ROOT / plan["helper_selection"]["path"]).read_bytes())
    seed = json.loads((o.ROOT / descriptor["helper_seed_ref"]["path"]).read_bytes())
    assert seed["binding"]["pid"] == seed["binding"]["start_ticks"] == 0
    assert seed["expected_memtotal"] == 1
    assert seed["worker"]["cgroup"] == "/SYNTHETIC-NOT-LIVE"
    assert plan["preparation_status"] == "METADATA_ONLY_NOT_ACTUAL_ADMISSION"
    assert plan["preparation_checks_not_run"]
    assert plan["comparison_contract"]["policy_seconds"] == 20
    assert plan["comparison_contract"]["helper_check_seconds"] == 3
    assert (
        "known4+public6, max16steps, seed20261014, original judges"
        in plan["comparison_contract"]["tasks"]
    )


def test_owner_wrapper_inverse_retains_all_current51d_logic():
    old = ROOT / "artifacts/jev-runtime/native-v2-q95-pin-successor-20261022-01/launch_browser.py"
    original, fresh = old.read_bytes(), (HERE / "launch_browser.py").read_bytes()
    assert (
        hashlib.sha256(original).hexdigest()
        == "51d676e035a7e71a0840700e85d4413f5039d94ca6f4fe567c8e9800ad9b70bd"
    )
    start = fresh.index(
        b"    # This private source copy changes only the declared unit date after SHA proof.\n"
    )
    stop = fresh.index(
        b'    spec = importlib.util.spec_from_file_location("_native_q95_original_owner", PARENT)',
        start,
    )
    assert fresh[:start] + fresh[stop:] == original


def test_caller_selected_seed_sha_and_symlink_refuse_before_read(tmp_path, monkeypatch):
    prepare = local("prepare_control.py")
    monkeypatch.setattr(prepare, "ROOT", tmp_path)
    seed = tmp_path / "SYNTHETIC-NOT-LIVE.json"
    seed.write_text('{"fixture":"SYNTHETIC-NOT-LIVE"}')
    seed.chmod(0o444)
    with pytest.raises(ValueError, match="Independently selected metadata SHA"):
        prepare.selected_blob(seed, "0" * 64)
    link = tmp_path / "seed-link.json"
    link.symlink_to(seed)
    with pytest.raises(ValueError, match="Small ordinary local selected metadata/source only"):
        prepare.selected_blob(link)
