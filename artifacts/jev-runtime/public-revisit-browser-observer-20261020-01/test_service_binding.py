"""Explicit fake live service/capacity tests; no model, socket or browser actor."""

import builtins
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location(
    "revisit_service_binding_tested", HERE / "service_binding.py"
)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


@pytest.fixture(autouse=True)
def no_framework_or_actor(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {
            "torch",
            "transformers",
            "numpy",
            "tokenizers",
            "peft",
            "safetensors",
            "playwright",
            "httpx",
        }:
            pytest.fail("Pure test attempted framework/transport/browser import: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def pending():
    selector = b.component("capacity_binding_source")
    return {
        "schema_id": b.SCHEMA,
        "profile": b.PROFILE,
        "legacy_service_binding": {
            "path": "absent-fake-legacy.json",
            "sha256": "0" * 64,
            "size_bytes": 1,
        },
        "revisit_capacity": selector.pending_selection(),
    }


def test_real_pending75_is_refused_before_legacy_read_or_service_check(monkeypatch):
    binding = pending()
    before = copy.deepcopy(binding)
    monkeypatch.setattr(b, "read_legacy_binding", lambda *_: pytest.fail("pending read legacy"))
    monkeypatch.setattr(b, "technical_observer", lambda: pytest.fail("pending checked service"))
    with pytest.raises(ValueError, match="closed75.*all68"):
        b.binding_state(binding)
    assert binding == before


def fake_live(tmp_path, monkeypatch):
    """All live identity and closed capacity objects below are EXPLICIT FAKES."""
    monkeypatch.setattr(b, "ROOT", tmp_path)
    legacy = {
        "schema_id": b.OLD_SCHEMA,
        "service_directory": "EXPLICIT_FAKE_SERVICE_NOT_CREATED",
        "worker": {"pid": 123, "start_ticks": 456},
        "v8_capacity": {"kind": "ORIGINAL_FAKE_OLD23_COMPONENT", "proof": ["old-proof"]},
    }
    raw = b.canonical(legacy)
    path = tmp_path / "genuine-shaped-but-explicit-fake-legacy.json"
    path.write_bytes(raw)
    ref = {"path": path.name, "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
    capacity = {
        "kind": "EXPLICIT_FAKE_NEW75_SELECTOR_KIND",
        "profile": b.PROFILE,
        "closed_evidence": True,
        "benchmark_ready": True,
        "required_policy_denominator": 68,
        "required_policy_ids": [f"EXPLICIT_FAKE_{i}" for i in range(68)],
        "policy_refusals": [],
        "task_quality_attested": False,
    }
    technical = {
        "worker": copy.deepcopy(legacy["worker"]),
        "listener": {"inode": "EXPLICIT_FAKE_SOCKET_NOT_OPENED"},
        "endpoint": {"policy": "http://127.0.0.1:1/v1"},
        "qualification_sha256": "ORIGINAL_FAKE_Q95_NOT_A_REAL_QUALIFICATION",
        "v8_capacity": copy.deepcopy(legacy["v8_capacity"]),
    }
    secrets = ("EXPLICIT_FAKE_PRIVATE_ENV_TOKEN_839518745",) * 2
    events = []

    def select(selection, expected, *, root):
        events.append(("new75", copy.deepcopy(selection)))
        assert expected == {"EXPLICIT_FAKE_EXPECTED": True} and root == tmp_path
        return capacity

    def old_state(value):
        events.append(("old23_Q95_live", copy.deepcopy(value)))
        assert value == legacy
        return technical, secrets

    selector = SimpleNamespace(
        KIND=capacity["kind"],
        expected_sources=lambda: {"EXPLICIT_FAKE_EXPECTED": True},
        validate_selection=select,
    )
    observer = SimpleNamespace(binding_state=old_state)
    real_checked = b.checked
    source_refs = list(b.expected_sources().values())[2:]

    def checked(ref):
        return b"EXPLICIT_FAKE_SOURCE_NEVER_EXECUTED" if ref in source_refs else real_checked(ref)

    monkeypatch.setattr(b, "checked", checked)
    monkeypatch.setattr(
        b, "component", lambda key: selector if key == "capacity_binding_source" else observer
    )
    binding = {
        "schema_id": b.SCHEMA,
        "profile": b.PROFILE,
        "legacy_service_binding": ref,
        "revisit_capacity": {"EXPLICIT_FAKE_NEW75_SELECTION": ["new-proof"]},
    }
    return SimpleNamespace(
        binding=binding,
        legacy=legacy,
        capacity=capacity,
        technical=technical,
        secrets=secrets,
        events=events,
        selector=selector,
        observer=observer,
        path=path,
    )


def test_composes_both_independent_authorities_without_retagging_or_input_mutation(
    tmp_path, monkeypatch
):
    f = fake_live(tmp_path, monkeypatch)
    before = copy.deepcopy((f.binding, f.legacy, f.capacity, f.technical))
    public, secrets = b.binding_state(f.binding)
    assert [e[0] for e in f.events] == ["new75", "old23_Q95_live"]
    assert f.events[1][1] == f.legacy
    assert (f.binding, f.legacy, f.capacity, f.technical) == before
    assert public["schema_id"] == b.PUBLIC_SCHEMA and public["profile"] == b.PROFILE
    assert public["legacy_q95_service"] == f.technical
    assert public["legacy_q95_service"]["v8_capacity"] == f.legacy["v8_capacity"]
    assert public["revisit_capacity"] == f.capacity
    assert "v8_capacity" not in public and public["task_quality_attested"] is False
    assert set(public["client_profile"]) == {
        "profile_manifest",
        "profile_helper",
        "launcher_source",
        "launcher_adapter",
    }
    assert secrets is f.secrets and all(s.encode() not in b.canonical(public) for s in secrets)


@pytest.mark.parametrize(
    "field,value",
    [
        ("closed_evidence", False),
        ("benchmark_ready", False),
        ("benchmark_ready", 1),
        ("profile", "old-v8"),
        ("kind", "old23"),
    ],
)
def test_new_capacity_missing_refused_or_wrong_identity_never_calls_legacy(
    tmp_path, monkeypatch, field, value
):
    f = fake_live(tmp_path, monkeypatch)
    f.capacity[field] = value
    monkeypatch.setattr(b, "technical_observer", lambda: pytest.fail("unready technical call"))
    with pytest.raises(ValueError, match="closed75.*all68"):
        b.binding_state(f.binding)
    assert [e[0] for e in f.events] == ["new75"]


def test_original_service_refusal_is_preserved_even_with_new_capacity_ready(tmp_path, monkeypatch):
    f = fake_live(tmp_path, monkeypatch)
    primary = RuntimeError("EXPLICIT_FAKE_ORIGINAL_Q95_LISTENER_REFUSAL")

    def refused(_):
        raise primary

    f.observer.binding_state = refused
    with pytest.raises(RuntimeError) as caught:
        b.binding_state(f.binding)
    assert caught.value is primary and f.events[0][0] == "new75"


def test_new_selector_failure_is_preserved_and_never_checks_old_service(tmp_path, monkeypatch):
    f = fake_live(tmp_path, monkeypatch)
    primary = ValueError("EXPLICIT_FAKE_NEW75_BROKEN_EXTERNAL_EXIT")

    def refused(*_, **__):
        raise primary

    f.selector.validate_selection = refused
    monkeypatch.setattr(b, "technical_observer", lambda: pytest.fail("selector failure old call"))
    with pytest.raises(ValueError) as caught:
        b.binding_state(f.binding)
    assert caught.value is primary


@pytest.mark.parametrize("location", ["binding", "public"])
def test_private_credential_anywhere_in_composite_is_refused_without_echo(
    tmp_path, monkeypatch, location
):
    f = fake_live(tmp_path, monkeypatch)
    if location == "binding":
        f.binding["revisit_capacity"]["EXPLICIT_FAKE_NEW75_SELECTION"].append(f.secrets[0])
    else:
        f.technical["endpoint"]["accidental_private_field"] = f.secrets[0]
    with pytest.raises(ValueError, match="Private credential") as caught:
        b.binding_state(f.binding)
    assert f.secrets[0] not in str(caught.value)


def test_returned_public_metadata_cannot_mutate_authority_or_later_result(tmp_path, monkeypatch):
    f = fake_live(tmp_path, monkeypatch)
    first, _ = b.binding_state(f.binding)
    first["legacy_q95_service"]["v8_capacity"]["proof"].append("attempted mutation")
    first["revisit_capacity"]["required_policy_ids"].clear()
    first["client_profile"]["profile_helper"]["sha256"] = "0" * 64
    second, _ = b.binding_state(f.binding)
    assert second["legacy_q95_service"] == f.technical
    assert len(second["revisit_capacity"]["required_policy_ids"]) == 68
    assert second["client_profile"]["profile_helper"] == b.expected_sources()["profile_helper"]


def test_legacy_public_read_is_explicitly_not_live_admission(tmp_path, monkeypatch):
    f = fake_live(tmp_path, monkeypatch)
    f.capacity["benchmark_ready"] = False
    assert b.read_legacy_binding(f.binding) == f.legacy
    assert f.events == []
    with pytest.raises(ValueError, match="all68"):
        b.binding_state(f.binding)


def test_legacy_file_drift_during_live_check_is_refused(tmp_path, monkeypatch):
    f = fake_live(tmp_path, monkeypatch)
    real_old = f.observer.binding_state

    def changed(value):
        answer = real_old(value)
        f.path.write_bytes(f.path.read_bytes().replace(b"old-proof", b"bad-proof"))
        return answer

    f.observer.binding_state = changed
    with pytest.raises(ValueError, match="Pinned stable"):
        b.binding_state(f.binding)


@pytest.mark.parametrize("change", ["extra", "oldschema", "oldprofile"])
def test_strict_new_outer_schema_cannot_be_old_binding_or_extra_fields(change):
    binding = pending()
    if change == "extra":
        binding["token"] = "EXPLICIT_FAKE_NEVER_READ"
    elif change == "oldschema":
        binding["schema_id"] = b.OLD_SCHEMA
    else:
        binding["profile"] = "composed-visible-hover-entries-settled-v8"
    with pytest.raises(ValueError, match="Exact new typed"):
        b.binding_shape(binding)


def test_source_byte_pin_failure_precedes_dynamic_execution(monkeypatch):
    ref = b.expected_sources()["technical_observer_source"]
    monkeypatch.setitem(
        b.PINS, "technical_observer_source", (ref["path"], "0" * 64, ref["size_bytes"])
    )
    name = "_revisit_service_technical_observer_source_" + "0" * 16
    assert name not in sys.modules
    with pytest.raises(ValueError, match="Pinned stable"):
        b.technical_observer()
    assert name not in sys.modules


def test_all_six_real_fixed_small_refs_match_without_service_import():
    expected = b.expected_sources()
    for key in b.PINS:
        assert len(b.checked(expected[key])) == expected[key]["size_bytes"]
    assert expected["technical_observer_source"]["sha256"].startswith("8e00743c")


def test_symlink_legacy_and_duplicate_keys_are_refused(tmp_path, monkeypatch):
    f = fake_live(tmp_path, monkeypatch)
    link = tmp_path / "linked.json"
    link.symlink_to(f.path)
    f.binding["legacy_service_binding"]["path"] = link.name
    with pytest.raises(ValueError, match="symlink"):
        b.read_legacy_binding(f.binding)
    with pytest.raises(ValueError, match="Duplicate"):
        b.parse(b'{"schema_id":1,"schema_id":2}')


def test_default_dry_has_no_component_calls_and_no_actual_entry(monkeypatch, capsys):
    monkeypatch.setattr(b, "component", lambda *_: pytest.fail("dry loaded component"))
    assert b.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["benchmark_ready"] is False
    assert result["model_processor_browser_service_calls"] == 0
    with pytest.raises(SystemExit) as caught:
        b.main(["--run"])
    assert caught.value.code == 2
