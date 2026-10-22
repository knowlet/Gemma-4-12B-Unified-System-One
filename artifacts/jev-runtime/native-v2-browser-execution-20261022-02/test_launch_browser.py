"""Ownership contracts only: no systemd, network, model or browser operations."""

import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("launch_browser", SOURCE / "launch_browser.py")
owner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(owner)
sys.modules["launch_browser"] = owner
spec = importlib.util.spec_from_file_location("_browser_entry_test", SOURCE / "browser_entry.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    here = root / "artifacts/jev-runtime/native-v2-browser-execution-20261022-01"
    here.mkdir(parents=True)
    monkeypatch.setattr(owner, "ROOT", root)
    monkeypatch.setattr(owner, "HERE", here)
    monkeypatch.setattr(owner, "CGROUP_ROOT", tmp_path / "cgroups")
    monkeypatch.setattr(owner, "current_boot", lambda: "boot")
    monkeypatch.setattr(owner, "memtotal", lambda: 130_595_864_576)
    for label in (
        owner.DRIVER,
        owner.GUARD_INDEX,
        owner.GUARD_SOURCE,
        owner.OBSERVER_SOURCE,
        str((here / "launch_browser.py").relative_to(root)),
        str((here / "browser_entry.py").relative_to(root)),
        "adapter/manifest.json",
        "upstream/marker",
    ):
        p = root / label
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("fake source; never executed")
    chromium = root / "fake-chromium"
    chromium.write_text("not executed")
    unit = "jev-current-nf4-raw-20261022-01.service"
    plan = {
        "schema_id": owner.PROFILE,
        "root": str(root),
        "arm": "raw",
        "attempt": "01",
        "prompt_format": "raw-json-v1",
        "unit": unit,
        "cgroup": f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{unit}",
        "run_directory": str((here / "raw-01").relative_to(root)),
        "output": str((here / "raw-01/browser").relative_to(root)),
        "unit_memory_max_bytes": owner.CAP,
        "boot_id": "boot",
        "nonce": "jev-nf4-browser-" + "a" * 32,
        "adapter": "adapter",
        "upstream": "upstream",
        "chromium": str(chromium),
    }
    plan["sources"] = [
        owner.pin(root / label)
        for label in (
            owner.DRIVER,
            owner.GUARD_INDEX,
            owner.GUARD_SOURCE,
            owner.OBSERVER_SOURCE,
            str((here / "launch_browser.py").relative_to(root)),
            str((here / "browser_entry.py").relative_to(root)),
            "adapter/manifest.json",
        )
    ]
    plan["comfy_observer"] = next(r for r in plan["sources"] if r["path"] == owner.OBSERVER_SOURCE)
    snapshot, blobs = root / owner.MODEL_SNAPSHOT, root / owner.MODEL_HUB / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir(parents=True)
    plan["model_cache_aliases"] = []
    for name in sorted(owner.MODEL_CACHE_NAMES):
        target = blobs / name
        target.write_bytes(("synthetic base artifact:" + name).encode())
        source = owner.pin(target)
        plan["sources"].append(source)
        logical = {**source, "path": (owner.MODEL_SNAPSHOT / name).as_posix()}
        alias = snapshot / name
        alias.symlink_to(os.path.relpath(target, alias.parent))
        plan["model_cache_aliases"].append({"logical": logical, "resolved": source})
        if name == "model.safetensors":
            expected = {k: source[k] for k in ("sha256", "size_bytes")}
            monkeypatch.setattr(owner, "EXPECTED_BASE_CHECKPOINT", expected)
            plan["expected_base_checkpoint"] = expected
    assert owner.validate_plan(plan) == plan
    return root, here, plan


def seal(plan):
    blob = json.dumps(plan, ensure_ascii=False, indent=4).encode() + b"\n"
    return blob, hashlib.sha256(blob).hexdigest()


@pytest.mark.parametrize("which", ["owner", "entry"])
def test_default_dry_never_reads_plan_or_loads_framework(monkeypatch, which):
    def forbidden(*_a, **_k):
        raise AssertionError("Execution boundary crossed")

    monkeypatch.setattr(owner, "load_plan", forbidden)
    monkeypatch.setattr(owner, "load_guard", forbidden)
    monkeypatch.setattr(owner, "load_observer", forbidden)
    monkeypatch.setattr(owner, "verify_cache_aliases", forbidden)
    monkeypatch.setattr(entry.runpy, "run_path", forbidden)
    assert (owner if which == "owner" else entry).main([]) == 0
    assert "torch" not in sys.modules


def test_entry_explicit_run_still_requires_owned_environment(setup, monkeypatch):
    monkeypatch.delenv("JEV_NATIVE_BROWSER_EXECUTE", raising=False)
    with pytest.raises(ValueError, match="environment"):
        entry.main(["--run"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("boot_id", "another-boot"),
        ("unit", "someone-else.service"),
        ("nonce", "unsafe"),
        ("prompt_format", "shared-elements-v2"),
        ("unit_memory_max_bytes", 1),
        ("output", "elsewhere"),
    ],
)
def test_foreign_or_ambiguous_plan_refuses_before_actor(setup, field, value):
    root, here, plan = setup
    plan[field] = value
    with pytest.raises(ValueError):
        owner.validate_plan(plan)
    assert not (here / "raw-01").exists()


def test_source_and_sealed_bytes_must_match(setup):
    root, here, plan = setup
    blob, digest = seal(plan)
    path = here / "plan.json"
    path.write_bytes(blob)
    assert owner.load_plan(path, digest) == plan
    with pytest.raises(ValueError, match="SHA"):
        owner.load_plan(path, "0" * 64)
    (root / owner.DRIVER).write_text("changed")
    with pytest.raises(ValueError, match="source"):
        owner.load_plan(path, digest)


def test_linked_input_cannot_be_pinned(setup):
    root, here, plan = setup
    link = root / "linked"
    link.symlink_to(root / owner.DRIVER)
    with pytest.raises(ValueError, match="Ordinary"):
        owner.pin(link)


def test_multichunk_source_pin_never_uses_whole_file_read(setup, monkeypatch):
    root, _here, _plan = setup
    path = root / "multichunk-source"
    contents = bytes(range(256)) * 8193 + b"terminal-partial-chunk"
    path.write_bytes(contents)
    expected = {
        "path": "multichunk-source",
        "sha256": hashlib.sha256(contents).hexdigest(),
        "size_bytes": len(contents),
    }
    original_open, requests = Path.open, []

    class BoundedReader:
        def __init__(self, source):
            self.source = source

        def __enter__(self):
            self.source.__enter__()
            return self

        def __exit__(self, *args):
            return self.source.__exit__(*args)

        def read(self, count=-1):
            assert 0 < count <= 1024**2
            requests.append(count)
            return self.source.read(count)

    def opened(selected, *args, **kwargs):
        actual = original_open(selected, *args, **kwargs)
        return BoundedReader(actual) if selected == path else actual

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Whole-file source allocation is forbidden")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "open", opened)
    assert owner.pin(path) == expected
    assert len(requests) >= 4


def model_alias(plan):
    return next(
        a
        for a in plan["model_cache_aliases"]
        if a["logical"]["path"].endswith("/model.safetensors")
    )


def test_original_two_hop_model_alias_chain_is_bound(setup):
    root, _here, plan = setup
    alias = model_alias(plan)
    logical, target = root / alias["logical"]["path"], root / alias["resolved"]["path"]
    middle = root / owner.MODEL_REPOSITORY / "blobs/intermediate-weight"
    middle.parent.mkdir(parents=True)
    middle.symlink_to(os.path.relpath(target, middle.parent))
    logical.unlink()
    logical.symlink_to(os.path.relpath(middle, logical.parent))
    assert owner.validate_plan(plan) == plan
    observed = owner.verify_cache_aliases(plan)
    assert len(next(r for r in observed if r["logical"] == alias["logical"])["links"]) == 2
    assert owner.verify_cache_aliases(plan, observed) == observed


@pytest.mark.parametrize(
    "change", ["retarget", "outside", "outside_ref", "field", "missing", "base", "unlisted"]
)
def test_foreign_or_missing_cache_alias_refuses_before_actor(setup, change):
    root, here, plan = setup
    alias = model_alias(plan)
    logical, target = root / alias["logical"]["path"], root / alias["resolved"]["path"]
    if change in {"retarget", "outside"}:
        other = target.with_name("other-target") if change == "retarget" else root / "outside-cache"
        other.write_bytes(target.read_bytes())
        logical.unlink()
        logical.symlink_to(other)
    elif change == "outside_ref":
        other = root / "outside-cache"
        other.write_bytes(target.read_bytes())
        resolved = owner.pin(other)
        plan["sources"].append(resolved)
        alias["resolved"] = resolved
        logical.unlink()
        logical.symlink_to(other)
    elif change == "field":
        del plan["model_cache_aliases"]
    elif change == "missing":
        plan["model_cache_aliases"].pop()
    elif change == "base":
        plan["expected_base_checkpoint"] = {"sha256": "0" * 64, "size_bytes": 1}
    else:
        plan["sources"].remove(alias["resolved"])
    with pytest.raises(ValueError):
        owner.validate_plan(plan)
    assert not (here / "raw-01").exists()


def test_alias_chain_rewrite_to_same_target_is_detected_without_weight_read(setup, monkeypatch):
    root, _here, plan = setup
    before = owner.verify_cache_aliases(plan)
    alias = model_alias(plan)
    logical = root / alias["logical"]["path"]
    logical.unlink()
    logical.symlink_to(root / alias["resolved"]["path"])

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Monitor alias readback must not rehash weights")

    monkeypatch.setattr(owner, "pin", forbidden)
    with pytest.raises(ValueError, match="chain changed"):
        owner.verify_cache_aliases(plan, before)


@pytest.mark.parametrize("missing", ["field", "source", "ref"])
def test_fresh_observer_source_and_exact_ref_are_mandatory(setup, missing):
    _root, here, plan = setup
    if missing == "field":
        del plan["comfy_observer"]
    elif missing == "source":
        plan["sources"] = [r for r in plan["sources"] if r["path"] != owner.OBSERVER_SOURCE]
    else:
        plan["comfy_observer"] = plan["sources"][0]
    with pytest.raises(ValueError):
        owner.validate_plan(plan)
    assert not (here / "raw-01").exists()


@pytest.mark.parametrize(
    "changed",
    [
        {"comfy_state": "connection_refused"},
        {"comfy_state": "absent_confirmed", "comfy_running": 1},
        {"comfy_state": "http_busy"},
        {"boot_id": "another-boot"},
        {"comfy_pending": False},
    ],
)
def test_ambiguous_or_contradictory_observer_state_is_unknown(setup, changed):
    _root, _here, plan = setup
    row = {
        "boot_id": "boot",
        "mem_total_bytes": 130_595_864_576,
        "mem_available_bytes": 125_000_000_000,
        "comfy_running": 0,
        "comfy_pending": 0,
        "comfy_state": "http_idle",
    }
    row.update(changed)
    with pytest.raises(ValueError):
        owner.observe_host(SimpleNamespace(observe=lambda **_kwargs: row), plan)


def fake_runtime(
    setup,
    monkeypatch,
    *,
    foreign_nonce=False,
    readback_bad=False,
    busy=False,
    fail_monitor=False,
    absent=False,
    retarget_alias=False,
):
    root, here, plan = setup
    directory, output = owner.run_paths(plan)
    group = owner.CGROUP_ROOT / plan["cgroup"].lstrip("/")
    calls = []
    state = {"started": False, "stopped": False, "statuses": 0, "hosts": 0}
    invocation = "b" * 32
    actor = {"pid": 77, "pgid": 77, "sid": 77, "start_ticks": 99}
    blob, digest = seal(plan)
    receipt = {
        "stage": "complete_pending_external_closure",
        "attempts": 10,
        "planned_denominator": 10,
        "prompt_format": plan["prompt_format"],
        "successes": 7,
        "groups": [
            {"group": "known4", "denominator": 4, "summary": {"attempts": 4}},
            {"group": "public6", "denominator": 6, "summary": {"attempts": 6}},
        ],
    }

    def command(*args, **_kwargs):
        calls.append(args)
        if args[0] == "systemd-run":
            state["started"] = True
            group.mkdir(parents=True)
            (group / "memory.max").write_text(str(owner.CAP if not readback_bad else 1))
            (group / "memory.swap.max").write_text("0")
            owner.save(
                directory,
                "actor-intent.json",
                {
                    "actor": actor,
                    "unit": plan["unit"],
                    "nonce": plan["nonce"],
                    "plan_sha256": digest,
                },
            )
            output.mkdir()
            (output / "receipt.json").write_text(json.dumps(receipt))
            (output / "output-index.json").write_text("{}")
            if retarget_alias:
                alias = model_alias(plan)
                logical = root / alias["logical"]["path"]
                logical.unlink()
                logical.symlink_to(root / alias["resolved"]["path"])
            return "launched fake child only"
        if "stop" in args:
            assert args[-1] == plan["unit"]
            state["stopped"] = True
            shutil.rmtree(group)
            return ""
        wanted = {
            "Type": "exec",
            "MemoryAccounting": "yes",
            "MemoryMax": str(owner.CAP),
            "MemoryHigh": "infinity",
            "MemorySwapMax": "0",
            "OOMPolicy": "kill",
            "KillMode": "control-group",
            "RuntimeMaxUSec": "2h",
            "KillSignal": "2",
            "TimeoutStopUSec": "20s",
            "RemainAfterExit": "yes",
            "InvocationID": invocation,
        }
        return "\n".join(k + "=" + v for k, v in wanted.items())

    def status(_unit):
        assert _unit == plan["unit"]
        if not state["started"]:
            return {"LoadState": "not-found"}
        state["statuses"] += 1
        running = state["statuses"] == 1 or busy or fail_monitor
        return {
            "InvocationID": invocation,
            "ControlGroup": plan["cgroup"],
            "SubState": "dead" if state["stopped"] else "running" if running else "exited",
            "MainPID": "0" if state["stopped"] or not running else "77",
            "ExecMainPID": "77",
            "Result": "success",
            "ExecMainCode": "1",
            "ExecMainStatus": "0",
        }

    primary = OSError("original monitor error")

    def host():
        state["hosts"] += 1
        if fail_monitor and state["hosts"] == 2:
            raise primary
        return {
            "boot_id": "boot",
            "mem_total_bytes": 130_595_864_576,
            "mem_available_bytes": 125_000_000_000,
            "comfy_running": int(busy and state["hosts"] >= 2),
            "comfy_pending": 0,
            "comfy_state": "absent_confirmed"
            if absent
            else "http_busy"
            if busy and state["hosts"] >= 2
            else "http_idle",
            "comfy_evidence": {"scope": "synthetic pure observer result"},
        }

    def observe(fn):
        try:
            return fn()
        except BaseException as exc:
            return {"observation_error": {"type": type(exc).__name__, "message": str(exc)}}

    monkeypatch.setattr(owner, "command", command)
    monkeypatch.setattr(
        owner, "description", lambda _unit: "foreign" if foreign_nonce else plan["nonce"]
    )
    monkeypatch.setattr(owner, "scan", lambda _actor: {"matches": [], "unreadable": []})
    monkeypatch.setattr(
        owner, "load_observer", lambda _plan: SimpleNamespace(observe=lambda **_kwargs: host())
    )
    monkeypatch.setattr(owner.time, "sleep", lambda _s: None)
    return SimpleNamespace(
        plan=plan,
        blob=blob,
        digest=digest,
        directory=directory,
        output=output,
        calls=calls,
        state=state,
        primary=primary,
        receipt=receipt,
        guard=SimpleNamespace(status=status, host=host, observe=observe),
    )


def test_real_owner_control_flow_preserves_plan_bytes_and_confirms_closure(setup, monkeypatch):
    f = fake_runtime(setup, monkeypatch)
    assert owner.execute(f.plan, f.digest, f.guard, {}, plan_blob=f.blob) == 0
    assert (f.directory / "selected-plan.json").read_bytes() == f.blob
    result = json.loads((f.directory / "root-external-closure.json").read_bytes())
    assert result["unit_readback_verified"] and result["cgroup_absent_after_stop"]
    assert f.state["stopped"] and len(result["two_proc_scans"]) == 2
    assert not result["changed_sources"]


def test_only_explicit_verified_absence_can_replace_http_idle(setup, monkeypatch):
    f = fake_runtime(setup, monkeypatch, absent=True)

    def forbidden():
        raise AssertionError("Old host probe must never be used")

    f.guard.host = forbidden
    assert owner.execute(f.plan, f.digest, f.guard, {}, plan_blob=f.blob) == 0
    result = json.loads((f.directory / "root-external-closure.json").read_bytes())
    assert result["host_after"]["comfy_state"] == "absent_confirmed"


def test_unknown_observer_before_start_never_creates_actor_or_unit(setup, monkeypatch):
    f = fake_runtime(setup, monkeypatch)

    def unknown(**_kwargs):
        raise RuntimeError("unknown observer state")

    monkeypatch.setattr(owner, "load_observer", lambda _plan: SimpleNamespace(observe=unknown))
    with pytest.raises(RuntimeError, match="unknown observer"):
        owner.execute(f.plan, f.digest, f.guard, {}, plan_blob=f.blob)
    assert not f.directory.exists() and not f.calls


def test_unknown_observer_during_monitor_stops_only_owned_actor(setup, monkeypatch):
    f = fake_runtime(setup, monkeypatch, fail_monitor=True)
    f.primary.evidence = {"sample": {"unknown_pid": 19, "argv": ["credential-value"]}}
    with pytest.raises(OSError) as caught:
        owner.execute(f.plan, f.digest, f.guard, {}, plan_blob=f.blob)
    assert caught.value is f.primary and f.state["stopped"]
    assert all(c[-1] == f.plan["unit"] for c in f.calls if "stop" in c)
    result = json.loads((f.directory / "root-external-closure.json").read_bytes())
    assert result["primary_error"]["observation_evidence"] == {
        "sample": {"unknown_pid": 19, "argv": "[REDACTED]"}
    }
    assert "credential-value" not in json.dumps(result)


def test_final_host_error_retains_bounded_redacted_evidence_and_refuses_closure(setup, monkeypatch):
    f = fake_runtime(setup, monkeypatch)
    unknown = RuntimeError("unproven final process sample")
    unknown.evidence = {"sample": {"unknown_pid": 23, "environment": {"token": "private"}}}

    def observed(**_kwargs):
        row = f.guard.host()
        if f.state["hosts"] == 3:
            raise unknown
        return row

    monkeypatch.setattr(owner, "load_observer", lambda _plan: SimpleNamespace(observe=observed))
    with pytest.raises(ValueError, match="final host"):
        owner.execute(f.plan, f.digest, f.guard, {}, plan_blob=f.blob)
    result = json.loads((f.directory / "root-external-closure.json").read_bytes())
    assert result["host_after"]["observation_error"]["observation_evidence"] == {
        "sample": {"unknown_pid": 23, "environment": "[REDACTED]"}
    }
    assert f.state["stopped"] and "private" not in json.dumps(result)


def test_oversized_observer_evidence_keeps_only_redacted_digest_and_size():
    unknown = RuntimeError("bounded observer evidence")
    unknown.evidence = {"note": "x" * (65 * 1024), "headers": {"Authorization": "private"}}
    row = owner.error_record(unknown)
    evidence = row["observation_evidence"]
    assert evidence["truncated"] is True and evidence["size_bytes"] > 64 * 1024
    assert len(evidence["sha256"]) == 64
    assert len(json.dumps(row).encode()) < 1024 and "private" not in json.dumps(row)


def test_prehashed_argv_evidence_is_preserved_without_payload():
    unknown = RuntimeError("exec identity raced")
    digest = {"sha256": "a" * 64, "size_bytes": 97}
    unknown.evidence = {"before": {"argv": digest}, "after": {"argv": ["private payload"]}}
    result = owner.error_record(unknown)["observation_evidence"]
    assert result == {"before": {"argv": digest}, "after": {"argv": "[REDACTED]"}}


def test_unserializable_observer_evidence_stays_explicit_and_small():
    unknown = RuntimeError("invalid observer evidence")
    unknown.evidence = {"invalid": float("nan")}
    assert owner.error_record(unknown)["observation_evidence"] == {"unsupported_type": "dict"}


def test_cache_alias_change_during_monitor_stops_owned_actor_and_preserves_unknown(
    setup, monkeypatch
):
    f = fake_runtime(setup, monkeypatch, retarget_alias=True)
    with pytest.raises(ValueError, match="chain changed"):
        owner.execute(f.plan, f.digest, f.guard, {}, plan_blob=f.blob)
    assert f.state["stopped"]
    result = json.loads((f.directory / "root-external-closure.json").read_bytes())
    assert result["model_cache_aliases_after"]["observation_error"]["type"] == "ValueError"


def test_foreign_nonce_never_stops_unit_or_claims_closure(setup, monkeypatch):
    f = fake_runtime(setup, monkeypatch, foreign_nonce=True)
    with pytest.raises(ValueError, match="invocation"):
        owner.execute(f.plan, f.digest, f.guard, {}, plan_blob=f.blob)
    assert not f.state["stopped"] and not any("stop" in c for c in f.calls)
    result = json.loads((f.directory / "root-external-closure.json").read_bytes())
    assert not result["cgroup_absent_after_stop"]
    assert result["stop_result"]["closure"].startswith("UNKNOWN")


def test_actual_cgroup_readback_mismatch_stops_only_authenticated_unit(setup, monkeypatch):
    f = fake_runtime(setup, monkeypatch, readback_bad=True)
    with pytest.raises(ValueError, match="bounds differ"):
        owner.execute(f.plan, f.digest, f.guard, {}, plan_blob=f.blob)
    assert f.state["stopped"]
    assert not json.loads((f.directory / "root-external-closure.json").read_bytes())[
        "unit_readback_verified"
    ]


def test_comfy_becomes_busy_stops_only_owned_actor_and_does_not_claim_success(setup, monkeypatch):
    f = fake_runtime(setup, monkeypatch, busy=True)
    with pytest.raises(ValueError, match="terminal"):
        owner.execute(f.plan, f.digest, f.guard, {}, plan_blob=f.blob)
    assert f.state["stopped"]
    result = json.loads((f.directory / "root-external-closure.json").read_bytes())
    assert result["reason"] == "Comfy_became_busy_stop_own_unit_only"


def test_monitor_and_finalization_errors_retain_primary(setup, monkeypatch):
    f = fake_runtime(setup, monkeypatch, fail_monitor=True)
    save = owner.save

    def failing_save(directory, name, data):
        if name == "root-external-closure.json":
            raise OSError("secondary closure persistence failure")
        return save(directory, name, data)

    monkeypatch.setattr(owner, "save", failing_save)
    with pytest.raises(OSError) as caught:
        owner.execute(f.plan, f.digest, f.guard, {}, plan_blob=f.blob)
    assert caught.value is f.primary and f.state["stopped"]
    assert "secondary closure" in " ".join(caught.value.__notes__)


@pytest.mark.parametrize(
    "failure",
    [
        "cgroup",
        "pid",
        "scan",
        "readback",
        "sources",
        "denominator",
        "index",
        "ownership",
        "observer",
        "aliases",
    ],
)
def test_missing_external_or_fixed_population_proof_cannot_pass(setup, monkeypatch, failure):
    f = fake_runtime(setup, monkeypatch)
    assert owner.execute(f.plan, f.digest, f.guard, {}, plan_blob=f.blob) == 0
    result = json.loads((f.directory / "root-external-closure.json").read_bytes())
    receipt = copy.deepcopy(f.receipt)
    if failure == "cgroup":
        result["cgroup_absent_after_stop"] = False
    elif failure == "pid":
        result["after_stop"]["MainPID"] = "77"
    elif failure == "scan":
        result["two_proc_scans"][1]["matches"] = [{"pid": 77}]
    elif failure == "readback":
        result["unit_readback_verified"] = False
    elif failure == "sources":
        result["changed_sources"] = ["driver"]
    elif failure == "denominator":
        receipt["groups"][1]["summary"]["attempts"] = 5
    elif failure == "ownership":
        result["stop_result"] = {"closure": "UNKNOWN_NO_AUTHENTICATED_OWNED_INVOCATION"}
    elif failure == "observer":
        result["host_after"] = {"observation_error": {"type": "RuntimeError", "message": "unknown"}}
    elif failure == "aliases":
        result["model_cache_aliases_after"] = {"observation_error": {"type": "ValueError"}}
    else:
        (f.output / "output-index.json").unlink()
    with pytest.raises(ValueError):
        owner.verify_closed(result, receipt, f.plan, f.output)
