"""Explicit synthetic closure fixtures; never launch CPU/model/browser actors."""

import builtins
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location(
    "revisit_binding_tested", HERE / "revisit_capacity_binding.py"
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
        }:
            raise AssertionError("Pure binding test attempted framework: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


def test_pending_with_real_fixed_small_refs_does_not_load_components_or_create_output(monkeypatch):
    monkeypatch.setattr(b, "components", lambda *a: pytest.fail("pending dynamic components"))
    expected, selection = b.expected_sources(), b.pending_selection()
    before = copy.deepcopy((expected, selection))
    result = b.validate_selection(selection, expected, root=b.ROOT)
    assert result["closed_evidence"] is False and result["benchmark_ready"] is False
    assert "MISSING" in result["status"] and (expected, selection) == before
    assert not (b.ROOT / b.CPU_BASE / "actual-01").exists()


def test_old_kind_or_changed_expected_source_never_accepted():
    selection, expected = b.pending_selection(), b.expected_sources()
    selection["kind"] = "jev_v8_actual_payload_capacity_admission_v1"
    with pytest.raises(ValueError, match="old23"):
        b.validate_selection(selection, expected, root=b.ROOT)
    expected["worker_source"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Fixed new"):
        b.validate_selection(b.pending_selection(), expected, root=b.ROOT)


def test_real_source_byte_drift_is_refused_before_module_execution():
    ref = {**b.expected_sources()["worker_source"], "sha256": "0" * 64}
    with pytest.raises(ValueError, match="Exact selected evidence SHA"):
        b.module(ref, b.ROOT, "_must_not_execute_revisit_worker")
    assert "_must_not_execute_revisit_worker" not in sys.modules


def synthetic_closed(monkeypatch):
    """Virtual fake output at the selected logical path; no actual tree is made."""
    expected = b.expected_sources()
    real_checked = b.checked
    plan = b.parse(real_checked(expected["plan"], b.ROOT))
    cohort = b.parse(real_checked(expected["cohort"], b.ROOT))
    output = b.ROOT / plan["output"]
    assert not output.exists()
    parent = {"pid": 101, "start_ticks": 202, "pgid": 101, "sid": 101}
    config = {
        "nonce": "a" * 32,
        "boot_id": "EXPLICIT_FAKE_BOOT",
        "owner": parent,
        "output": str(output),
        "cohort": expected["cohort"],
        "technical_plan": plan["technical_plan"],
    }
    config_blob = b.canonical(config)
    vpath = (
        b.ROOT
        / "artifacts/jev-runtime/public-v8-capacity-preflight-20261019-02/v8_capacity_validation.py"
    )
    vref = {
        "path": str(vpath.relative_to(b.ROOT)),
        "sha256": "1b8c621c77e68fb42e8e00b1c37b29f4847746f6214f020d1a54281f41464b4e",
        "size_bytes": 16724,
    }
    validator = b.module(vref, b.ROOT, "_fake_binding_original_prefix_validator")
    rows = []
    ids = [r["case_id"] for r in cohort["rows"]] + list(validator.EDGES)
    for ordinal, label in enumerate(ids):
        kind = "helper" if ordinal >= 73 else "policy"
        count = {71: 8160, 72: 8161, 74: 8065}.get(ordinal, 3)
        prefix = {
            "prompt_sha256": b.sha(b"EXPLICIT_FAKE_PROMPT"),
            "ids": [7] * count,
            "attention_mask": [1] * count,
            "token_count": count,
        }
        rows.append(
            {
                "ordinal": ordinal,
                "id": label,
                "kind": kind,
                "prefix": None if ordinal == 69 else prefix,
                "outcome": "REFUSED_SCHEMA" if ordinal == 69 else validator.outcome(prefix, kind),
                "error": {"type": "RequestValidationError", "message": "EXPLICIT_FAKE_WIDTH401"}
                if ordinal == 69
                else None,
                "native_forward_calls": 0,
                "expectation_met": True,
            }
        )

    def counts():
        return {
            k: sum(r["outcome"] == k for r in rows)
            for k in ("ACCEPTED", "REFUSED_SCHEMA", "REFUSED_CONTEXT", "UNKNOWN")
        }

    selected = {
        "terminal_sha256": "d" * 64,
        "counts": counts(),
        "rows": rows,
        "worker_resource_gates": {"sha256": "e" * 64, "rows": 81},
    }
    cleanup = {
        "confirmed": True,
        "remaining": [],
        "tracked_PID_start_ticks": {"303": 404},
        "tracked_groups": [303],
    }
    outer = {
        "schema_id": "jev_revisit_cpu_processor_capacity_controller_v1",
        "profile": b.PROFILE,
        "plan_sha256": expected["plan"]["sha256"],
        "nonce": config["nonce"],
        "parent": parent,
        "status": "CLOSED_CPU_COHORT_PENDING_EXTERNAL_PARENT_EXIT",
        "primary_error": None,
        "execution": {
            "started": True,
            "returncode": 0,
            "primary_error": None,
            "secondary_errors": [],
            "cleanup": cleanup,
        },
        "selected": selected,
        "task_quality_attested": False,
        "universal_profile_capacity_attested": False,
        "external_parent_exit0_and_PID_start_PG_SID_absence_required": True,
    }
    selection = {
        **b.pending_selection(),
        "controller_terminal": {
            "path": str((output / "controller-terminal.json").relative_to(b.ROOT)),
            "sha256": "1" * 64,
            "size_bytes": 1,
        },
        "external_parent_exit": {
            "path": str((output / "external-parent-exit.json").relative_to(b.ROOT)),
            "sha256": "2" * 64,
            "size_bytes": 1,
        },
    }
    external = {
        "schema_id": b.EXTERNAL,
        "plan_sha256": expected["plan"]["sha256"],
        "nonce": config["nonce"],
        "boot_id": config["boot_id"],
        "parent": parent,
        "argv": [
            plan["python"],
            "-B",
            "-s",
            str(b.ROOT / expected["controller_source"]["path"]),
            "--run",
            "--plan",
            str(b.ROOT / expected["plan"]["path"]),
            "--plan-sha256",
            expected["plan"]["sha256"],
        ],
        "returncode": 0,
        "source_before": {k: expected[k] for k in ("controller_source", "worker_source")},
        "source_after": {k: expected[k] for k in ("controller_source", "worker_source")},
        "controller_terminal": selection["controller_terminal"],
        "absence": {
            "parent_pid_absent": True,
            "matching_parent_start_absent": True,
            "PGID_members": [],
            "SID_members": [],
            "tracked_owned_PID_start_remaining": [],
        },
        "started_unix": 1000,
        "finished_unix": 1100,
        "observed_unix": 1101,
    }
    calls = []

    def fake_checked(ref, root):
        if ref["path"] == selection["controller_terminal"]["path"]:
            return b.canonical(outer)
        if ref["path"] == selection["external_parent_exit"]["path"]:
            return b.canonical(external)
        return real_checked(ref, root)

    def fake_read(path, root, maximum=b.MAX_FILE):
        if Path(path) == output / "worker-config.json":
            return config_blob
        if Path(path) == output / "owned-memory-monitor.jsonl":
            return b"EXPLICIT_FAKE_MONITOR"
        return b.read_original(path, root, maximum)

    def fake_validate(cpu, worker, given_cohort, given_output, config_sha, nonce, given_cleanup):
        calls.append("validate_full75")
        assert given_cohort == cohort and given_output == output
        assert (
            config_sha == b.sha(config_blob)
            and nonce == config["nonce"]
            and given_cleanup == cleanup
        )
        assert len(rows) == 75 and [r["id"] for r in rows] == ids
        for row in rows:
            if row["prefix"] is not None:
                validator.validate_prefix(row["prefix"], "EXPLICIT_FAKE_PROMPT")
                assert row["outcome"] == validator.outcome(row["prefix"], row["kind"])
            if row["ordinal"] >= 68:
                assert row["outcome"] in validator.expected_outcomes(row["id"])
        assert selected["counts"] == counts()
        return copy.deepcopy(selected)

    def fake_monitor(blob, boot, **kwargs):
        calls.append("validate_owned_monitor")
        assert (
            blob == b"EXPLICIT_FAKE_MONITOR"
            and boot == config["boot_id"]
            and kwargs == {"parent": parent, "cleanup": cleanup}
        )
        return 3

    worker = SimpleNamespace(
        source_union=lambda *a: {"EXPLICIT_FAKE_SOURCE_UNION": {}},
        authenticate=lambda files: calls.append("authenticate_source_union"),
    )
    controller = SimpleNamespace(validate_worker=fake_validate)
    cpu = SimpleNamespace(validate_monitor_evidence=fake_monitor)
    real_module = b.module

    def fake_module(ref, root, name):
        if ref == expected["profile_helper"]:

            def verify(path, **kwargs):
                calls.append("verify_new_materialized_profile")
                assert path == b.ROOT / expected["profile_manifest"]["path"].rsplit("/", 1)[0]
                assert kwargs == {
                    "expected_manifest_sha256": expected["profile_manifest"]["sha256"],
                    "expected_source_sha256": expected["profile_helper"]["sha256"],
                }

            return SimpleNamespace(verify_materialized_profile=verify)
        return real_module(ref, root, name)

    monkeypatch.setattr(b, "read_original", b.read, raising=False)
    monkeypatch.setattr(b, "read", fake_read)
    monkeypatch.setattr(b, "checked", fake_checked)
    monkeypatch.setattr(b, "components", lambda *a: (controller, cpu, worker))
    monkeypatch.setattr(b, "module", fake_module)
    return SimpleNamespace(
        expected=expected,
        selection=selection,
        external=external,
        outer=outer,
        rows=rows,
        counts=counts,
        selected=selected,
        calls=calls,
        output=output,
    )


def test_complete_synthetic75_join_separates_negative_edges_and_policy_readiness(monkeypatch):
    case = synthetic_closed(monkeypatch)
    before = copy.deepcopy((case.expected, case.selection, case.outer, case.external))
    result = b.validate_selection(case.selection, case.expected, root=b.ROOT)
    assert result["closed_evidence"] is True and result["benchmark_ready"] is True
    assert result["required_policy_denominator"] == 68 and len(result["required_policy_ids"]) == 68
    assert result["counts"] == {
        "ACCEPTED": 72,
        "REFUSED_SCHEMA": 1,
        "REFUSED_CONTEXT": 2,
        "UNKNOWN": 0,
    }
    assert result["policy_refusals"] == [] and result["task_quality_attested"] is False
    assert case.calls == [
        "validate_full75",
        "validate_owned_monitor",
        "authenticate_source_union",
        "verify_new_materialized_profile",
    ]
    assert (
        case.expected,
        case.selection,
        case.outer,
        case.external,
    ) == before and not case.output.exists()


@pytest.mark.parametrize("ordinal", [16, 50, 67])
def test_any_new_hint_or_technical_window_policy_refusal_keeps_benchmark_not_ready(
    monkeypatch, ordinal
):
    case = synthetic_closed(monkeypatch)
    row = case.rows[ordinal]
    row.update(
        prefix=None,
        outcome="REFUSED_SCHEMA",
        error={"type": "RequestValidationError", "message": "EXPLICIT_FAKE_POLICY_REFUSAL"},
    )
    case.selected["counts"] = case.counts()
    result = b.validate_selection(case.selection, case.expected, root=b.ROOT)
    assert result["closed_evidence"] is True and result["benchmark_ready"] is False
    assert result["policy_refusals"] == [
        {"ordinal": ordinal, "id": row["id"], "outcome": row["outcome"], "error": row["error"]}
    ]


@pytest.mark.parametrize(
    "change",
    [
        "exit",
        "bool-exit",
        "nonce",
        "boot",
        "pid-start",
        "PG-members",
        "SID-members",
        "child-remaining",
        "source-after",
        "argv",
        "time",
        "terminal-pin",
        "partial-parent",
        "wrong-tree",
    ],
)
def test_external_or_controller_mismatch_never_admits(monkeypatch, change):
    case = synthetic_closed(monkeypatch)
    if change == "exit":
        case.external["returncode"] = 1
    elif change == "bool-exit":
        case.external["returncode"] = False
    elif change == "nonce":
        case.external["nonce"] = "b" * 32
    elif change == "boot":
        case.external["boot_id"] = "OTHER_BOOT"
    elif change == "pid-start":
        case.external["parent"] = {**case.external["parent"], "start_ticks": 203}
    elif change == "PG-members":
        case.external["absence"]["PGID_members"] = [{"pid": 505}]
    elif change == "SID-members":
        case.external["absence"]["SID_members"] = [{"pid": 505}]
    elif change == "child-remaining":
        case.external["absence"]["tracked_owned_PID_start_remaining"] = [
            {"pid": 303, "start_ticks": 404}
        ]
    elif change == "source-after":
        case.external["source_after"] = {**case.external["source_after"], "worker_source": {}}
    elif change == "argv":
        case.external["argv"][3] = "/tmp/other.py"
    elif change == "time":
        case.external["observed_unix"] = 1099
    elif change == "terminal-pin":
        case.external["controller_terminal"] = {
            **case.selection["controller_terminal"],
            "sha256": "f" * 64,
        }
    elif change == "partial-parent":
        case.outer["execution"]["secondary_errors"] = [{"type": "Partial"}]
    else:
        case.selection["controller_terminal"] = {
            **case.selection["controller_terminal"],
            "path": str(
                Path(case.selection["controller_terminal"]["path"]).with_name("other.json")
            ),
        }
    with pytest.raises(ValueError):
        b.validate_selection(case.selection, case.expected, root=b.ROOT)
    assert "validate_full75" not in case.calls


def test_complete_prefix_validation_cannot_be_skipped_by_ready_counts(monkeypatch):
    case = synthetic_closed(monkeypatch)
    case.rows[20]["prefix"]["attention_mask"][0] = True
    with pytest.raises(ValueError, match="Genuine complete"):
        b.validate_selection(case.selection, case.expected, root=b.ROOT)
    assert case.calls == ["validate_full75"]


def test_missing_either_external_or_controller_is_not_ready(monkeypatch):
    expected = b.expected_sources()
    monkeypatch.setattr(
        b, "components", lambda *a: pytest.fail("missing closure imported component")
    )
    for field in ("controller_terminal", "external_parent_exit"):
        selection = b.pending_selection()
        selection[field] = {"path": "unused", "sha256": "0" * 64, "size_bytes": 0}
        result = b.validate_selection(selection, expected, root=b.ROOT)
        assert result["benchmark_ready"] is False and result["closed_evidence"] is False


def test_defaultdry_has_no_evidence_read_dynamic_module_or_actor(monkeypatch, capsys):
    monkeypatch.setattr(b, "read", lambda *a, **kw: pytest.fail("defaultdry evidence read"))
    monkeypatch.setattr(b, "module", lambda *a: pytest.fail("defaultdry module"))
    assert b.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert (
        result["benchmark_ready"] is False and result["processor_model_service_browser_calls"] == 0
    )
    assert not any(
        n.split(".")[0] in {"torch", "transformers", "numpy", "tokenizers"} for n in sys.modules
    )
