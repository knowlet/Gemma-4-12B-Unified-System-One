"""Pure post-exit regressions: no processor, service, model, or browser."""

import builtins
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location(
    "tested_native179_postexit", HERE / "validate_prepare_saved.py"
)
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


@pytest.fixture(autouse=True)
def no_framework_or_actor(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "transformers", "tokenizers", "peft", "playwright"}:
            pytest.fail("Framework import in pure postexit verification: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    import subprocess

    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: pytest.fail("Actor started"))


def core():
    return v.actor_module().libraries()[2]


def fresh_data():
    return {
        "refs": {"one": {"sha256": "a" * 64, "size_bytes": 7}},
        "train_ids": ["a"],
        "dev_ids": ["b"],
        "counts": {"total": 2},
        "provenance": {
            "components": {
                "public": ("path", "sha"),
                "owned": ("path2", "sha2"),
                "form": ("path3", "sha3"),
            }
        },
    }


def test_exact_actual_container_bug_uses_complete_ordered_json_bytes():
    c, fresh = core(), fresh_data()
    saved = json.loads(c.wire(fresh))
    assert saved != fresh
    result = v.ordered_data_join(c, saved, fresh)
    assert result["ordered_data_evidence_equal"] is True
    assert result["saved_sha256"] == result["fresh_sha256"]
    assert result["python_container_equal"] is False
    assert saved["provenance"]["components"]["public"] == ["path", "sha"]


@pytest.mark.parametrize("drift", ["value", "key_order", "extra", "missing", "tuple_order"])
def test_ordered_data_join_does_not_drop_normalize_sort_or_waive_fields(drift):
    c, fresh = core(), fresh_data()
    saved = json.loads(c.wire(fresh))
    if drift == "value":
        saved["refs"]["one"]["size_bytes"] += 1
    elif drift == "key_order":
        saved = dict(reversed(list(saved.items())))
    elif drift == "extra":
        saved["extra"] = True
    elif drift == "missing":
        saved.pop("provenance")
    else:
        saved["provenance"]["components"]["public"].reverse()
    with pytest.raises(ValueError, match="ordered native data bytes"):
        v.ordered_data_join(c, saved, fresh)


def closed_fixture():
    c = core()
    actor = {"pid": 987654, "pgid": 987654, "sid": 987654, "start_ticks": 12345}
    unit = {"caller": actor, "cgroup_inode": 7}
    plan = {
        "schema_id": "EXPLICIT_FAKE_ACTOR",
        "nonce": "nonce",
        "sources": {"fake": {}},
        "unit": {"EXPLICIT_FAKE_UNIT": True},
        "lifecycle": "SINGLE_WORKER",
    }
    intent = {
        "schema_id": plan["schema_id"] + "_intent",
        "plan_sha256": "planSHA",
        "nonce": "nonce",
        "sources": plan["sources"],
        "actor": actor,
        "unit_before": unit,
        "properties": {"MainPID": str(actor["pid"]), "InvocationID": "inv"},
    }
    result = {
        "schema_id": plan["schema_id"] + "_terminal",
        "plan_sha256": "planSHA",
        "nonce": "nonce",
        "sources": plan["sources"],
        "actor": actor,
        "InvocationID": "inv",
        "complete_pending_external_unit_exit": True,
        "primary_error": None,
        "runtime_files": 348,
        "processor_reference_files": 58,
        "processor_only": True,
        "native_model_forward_train_HTTP_browser_calls": 0,
        "old_capacity_qualification_attested": False,
        "task_quality_attested": False,
        "lifecycle": "SINGLE_WORKER",
        "unit_after": copy.deepcopy(unit),
    }
    closure = {
        "actors": {"worker": actor},
        "two_proc_scans": [{"matches": [], "unreadable": []}] * 2,
        "source_files_unchanged": 1,
        "worker_primary_error": None,
        "review": None,
        "review_error": {"type": "ValueError", "message": "Same full native data"},
        "task_quality_attested": False,
        "final_postexit_cgroup_counters": None,
        "unit_exit": {
            "reason": "terminal",
            "primary_error": None,
            "invocation_id": "inv",
            "terminal_before_stop": {
                "InvocationID": "inv",
                "ExecMainPID": "987654",
                "ExecMainCode": "1",
                "ExecMainStatus": "0",
                "Result": "success",
                "MainPID": "0",
                "ControlGroup": "",
                "SubState": "exited",
            },
            "cgroup_absent_after_stop": True,
            "after_stop": {"MainPID": "0", "ControlGroup": "", "LoadState": "not-found"},
        },
    }
    cpu = SimpleNamespace(
        validate_unit_evidence=lambda *_: None, unit_identity=lambda unit: unit["cgroup_inode"]
    )
    return c, cpu, plan, intent, result, closure


@pytest.mark.parametrize(
    "drift",
    [
        None,
        "invocation",
        "ExecMainPID",
        "exit1",
        "live_PID",
        "unreadable",
        "cgroup",
        "source_count",
        "inode",
        "primary",
        "unrelated_error",
    ],
)
def test_external_exit_and_saved_unit_join_cannot_mask_real_failures(drift):
    c, cpu, plan, intent, result, closure = closed_fixture()
    if drift == "invocation":
        closure["unit_exit"]["invocation_id"] = "other"
    elif drift == "ExecMainPID":
        closure["unit_exit"]["terminal_before_stop"]["ExecMainPID"] = "1"
    elif drift == "exit1":
        closure["unit_exit"]["terminal_before_stop"]["ExecMainStatus"] = "1"
    elif drift == "live_PID":
        closure["two_proc_scans"] = [{"matches": [987654], "unreadable": []}] * 2
    elif drift == "unreadable":
        closure["two_proc_scans"] = [{"matches": [], "unreadable": [987654]}] * 2
    elif drift == "cgroup":
        closure["unit_exit"]["cgroup_absent_after_stop"] = False
    elif drift == "source_count":
        closure["source_files_unchanged"] = 0
    elif drift == "inode":
        result["unit_after"]["cgroup_inode"] += 1
    elif drift == "primary":
        result["primary_error"] = {"type": "ValueError", "message": "model failure"}
    elif drift == "unrelated_error":
        closure["review_error"]["message"] = "unrelated posthoc failure"
    if drift is None:
        v.closed_join(c, cpu, plan, "planSHA", intent, result, closure)
    else:
        with pytest.raises(ValueError):
            v.closed_join(c, cpu, plan, "planSHA", intent, result, closure)


def test_all179_v2_records_and_cfgs_reach_strict_artifact_boundaries():
    c, calls = core(), []
    cases = [SimpleNamespace(id="FAKE-" + str(i)) for i in range(179)]
    wires = [{"case_id": case.id, "teacher": "complete"} for case in cases]
    rows = [
        {
            "modes": {
                "V2": {
                    "outcome": "COMPILED",
                    "compiled": {"record": {"case_id": case.id}},
                    "cfg": {
                        "outcome": "COMPILED",
                        "plan": {"full": True},
                        "inventory": {"full": True},
                    },
                }
            }
        }
        for case in cases
    ]

    def record_boundary(record, case, wire):
        assert record["case_id"] == case.id == wire["case_id"]
        calls.append((case.id, "record"))
        return {"original_labels": [case.id]}

    def cfg_boundary(cfg, record, labels):
        assert cfg["case_id"] == record["case_id"] == labels["original_labels"][0]
        assert cfg["plan"] == cfg["inventory"] == {"full": True}
        calls.append((cfg["case_id"], "cfg"))

    refs = list(range(179))
    data = {"cases": cases, "wires": wires}

    def read(ref):
        return c.wire(rows[ref])

    result = v.native_records(
        c, data, refs, read, native_record=record_boundary, native_cfg=cfg_boundary, v2="V2"
    )
    assert result["compiled_native_records_verified"] == 179
    assert result["compiled_native_CFG_inventories_verified"] == 179
    assert len(calls) == 358
    assert calls[0] == ("FAKE-0", "record") and calls[-1] == ("FAKE-178", "cfg")
    with pytest.raises(ValueError, match="All179"):
        v.native_records(
            c,
            data,
            refs[:-1],
            read,
            native_record=record_boundary,
            native_cfg=cfg_boundary,
            v2="V2",
        )
    rows[178]["modes"]["V2"]["cfg"]["inventory"]["full"] = False
    with pytest.raises(AssertionError):
        v.native_records(
            c, data, refs, read, native_record=record_boundary, native_cfg=cfg_boundary, v2="V2"
        )
