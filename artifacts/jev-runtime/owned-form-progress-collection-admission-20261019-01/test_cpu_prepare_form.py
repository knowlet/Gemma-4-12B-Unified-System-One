"""Pure CPU driver boundary tests; no process, model or processor is started."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "_form_cpu_driver_pure", Path(__file__).with_name("cpu_prepare_form.py")
)
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


def binding():
    actual = {"pid": 123, "ppid": 100, "pgid": 123, "sid": 123, "start_ticks": 456, "state": "R"}
    released = {**actual, "state": "S"}
    return {"nonce": "owned", "plan_sha256": "selected", "identity": released}, actual


def test_release_accepts_same_owned_identity_across_scheduling_state_change():
    release, actual = binding()
    C.verify_worker_binding(release, actual, "owned", "selected", 123)


@pytest.mark.parametrize("field", ["pid", "ppid", "pgid", "sid", "start_ticks"])
def test_release_rejects_each_identity_drift(field):
    release, actual = binding()
    release["identity"][field] += 1
    with pytest.raises(ValueError, match="Released exact"):
        C.verify_worker_binding(release, actual, "owned", "selected", 123)


@pytest.mark.parametrize("mutation", ["nonce", "plan_sha256", "boolean", "missing", "no_process"])
def test_release_rejects_foreign_or_unknown_binding(mutation):
    release, actual = binding()
    if mutation in {"nonce", "plan_sha256"}:
        release[mutation] = "foreign"
    elif mutation == "boolean":
        release["identity"]["start_ticks"] = actual["start_ticks"] = True
    elif mutation == "missing":
        del release["identity"]["sid"]
    else:
        actual = None
    with pytest.raises(ValueError, match="Released exact"):
        C.verify_worker_binding(release, actual, "owned", "selected", 123)


def test_terminal_failure_keeps_primary_and_records_secondary(tmp_path, monkeypatch):
    primary = {"type": "ValueError", "message": "original native error"}
    secondary = []
    value = {"primary_error": primary, "secondary_errors": secondary}

    def broken(*args, **kwargs):
        raise OSError("journal failed")

    monkeypatch.setattr(C, "save", broken)
    assert C.terminal(tmp_path, "terminal.json", value, secondary) is False
    assert value["primary_error"] is primary
    assert secondary[0]["phase"] == "terminal-publication"
    assert not (tmp_path / "terminal.json").exists()


def test_terminal_reserve_is_independent_and_no_payload_is_trimmed(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "MAX_BYTES", 100)
    monkeypatch.setattr(C, "MAX_FILE", 80)
    monkeypatch.setattr(C, "RESERVE", 20)
    C.save(tmp_path, "payload", b"x" * 75)
    with pytest.raises(ValueError, match="terminal reserve"):
        C.save(tmp_path, "ordinary", b"y" * 10)
    C.save(tmp_path, "terminal", b"z" * 10, terminal=True)
    assert (tmp_path / "payload").read_bytes() == b"x" * 75
    with pytest.raises(ValueError, match="terminal reserve"):
        C.save(tmp_path, "overflow", b"z" * 20, terminal=True)


def test_parent_gate_failure_never_starts_child_and_preserves_all23(tmp_path, monkeypatch):
    plan = {"ordered_cases": [{"id": str(i)} for i in range(23)]}
    monkeypatch.setattr(C, "load_plan", lambda *a: (plan, {}))

    class Gate:
        def check(self, phase, scope):
            assert (phase, scope) == ("before-any-project-framework-or-parent-payload", "cpu")
            raise ValueError("fresh CPU floor refused")

    monkeypatch.setattr(C, "resource_gate", lambda *a: Gate())
    monkeypatch.setattr(C, "authenticate_sources", lambda *a: pytest.fail("past failed gate"))
    monkeypatch.setattr(C.subprocess, "Popen", lambda *a, **k: pytest.fail("no owned child"))
    # The child argv is constructed before the protected stage; it is never executed.
    plan["python"] = "/fake/not-executed"
    output = tmp_path / "fresh"
    assert C.controller(Path("fake-plan"), "selected", output, run=True) == 2
    result = C.parse((output / "controller-terminal.json").read_bytes())
    assert result["primary_error"]["message"] == "fresh CPU floor refused"
    assert result["process_started"] is False and result["reference_ready"] is None
    assert len(result["all23_rows"]) == 23
    assert all(r["status"] == "NOT_EXECUTED" for r in result["all23_rows"])


def fixture_reference(output):
    """Synthetic closed-file joins, explicitly not a genuine processor receipt."""
    ordered = [
        {"id": f"fake-{i}", "split": "train" if i < 13 else "development", "family": "fake"}
        for i in range(23)
    ]
    closure = {"components": {"processor": {"class": "Fake", "path": "/fake"}}, "versions": {}}
    closure_blob = C.encode(closure)
    C.save(output, "reference-environment/processor-source-closure.json", closure_blob)
    evidence = {
        **closure,
        "tokenizer_is_processor_tokenizer": True,
        "max_context": 8192,
        "generation_EOS": [1, 106, 50],
    }
    for side in ["before", "after"]:
        C.save(output, f"actual-processor-{side}.json", C.encode(evidence))
    rows = []
    for i, item in enumerate(ordered):
        raw = C.encode({"synthetic_case": item["id"]})
        wire_sha = hashlib.sha256(raw).hexdigest()
        tensors = {"input_ids": [[11, 12, 10, 106]], "attention_mask": [[1, 1, 1, 1]]}
        record = {
            "case_id": item["id"],
            "split": item["split"],
            "original_wire_sha256": wire_sha,
            "normalized_wire_sha256": wire_sha,
            "prompt_tokens": 2,
            "input_tokens": 4,
            "logits_positions": [1, 2],
            "target_token_ids": [10, 106],
            "native_stop_token_id": 106,
            "processor_tensors": tensors,
        }
        row = {"record": record, "inputs": tensors, "positions": [1, 2], "targets": [10, 106]}
        cfg = {
            "compiled_row_sha256": hashlib.sha256(C.encode(row)).hexdigest(),
            "request_sha256": wire_sha,
            "target_ids": [10, 106],
            "positions": [1, 2],
            "allowed_ids": [[10], [106]],
        }
        body = C.save(output, f"compiled/{i:02d}.native-wire.json", raw)
        ref = C.save(
            output,
            f"compiled/{i:02d}.json",
            C.encode({"case": {"id": item["id"]}, "row": row, "cfg_plan": cfg}),
        )
        rows.append(
            {**item, "status": "COMPLETE_REFERENCE", "native_wire": body, "compiled_reference": ref}
        )
    terminal = {
        "schema_id": "owned-form-progress-native-CPU-reference-v1",
        "status": C.SUCCESS,
        "nonce": "owned",
        "plan_sha256": "selected",
        "primary_error": None,
        "secondary_errors": [],
        "counts": {"saved_rows": 23, "compiled_rows_in_memory": 23, "compile_returned": True},
        "rows": rows,
    }
    C.save(output, "worker-terminal.json", C.encode(terminal))
    return {
        "ordered_cases": ordered,
        "reference_items": {
            "processor-source-closure.json": {
                "sha256": hashlib.sha256(closure_blob).hexdigest(),
                "size_bytes": len(closure_blob),
            }
        },
    }


def test_full_saved23_reference_joins(tmp_path):
    plan = fixture_reference(tmp_path)
    assert C.select_reference(tmp_path, plan, "selected", "owned")["compiled_rows"] == 23


@pytest.mark.parametrize(
    "mutation", ["denominator", "partial", "source", "row_SHA", "wire_SHA", "EOS", "CFG"]
)
def test_saved_reference_refuses_incomplete_or_corrupted_evidence(tmp_path, mutation):
    plan = fixture_reference(tmp_path)
    terminal_path = tmp_path / "worker-terminal.json"
    terminal = C.parse(terminal_path.read_bytes())
    if mutation == "denominator":
        terminal["rows"].pop()
    elif mutation == "partial":
        terminal["counts"]["compile_returned"] = False
    elif mutation == "source":
        terminal["plan_sha256"] = "foreign"
    else:
        item = terminal["rows"][0]
        key = "native_wire" if mutation == "wire_SHA" else "compiled_reference"
        path = tmp_path / item[key]["path"]
        if mutation in {"row_SHA", "wire_SHA"}:
            path.write_bytes(b"changed")
        else:
            data = C.parse(path.read_bytes())
            if mutation == "EOS":
                data["row"]["targets"][-1] = 99
            else:
                data["cfg_plan"]["allowed_ids"][0] = [99]
            blob = C.encode(data)
            path.write_bytes(blob)
            item[key].update(sha256=hashlib.sha256(blob).hexdigest(), size_bytes=len(blob))
    terminal_path.write_bytes(C.encode(terminal))
    with pytest.raises(ValueError):
        C.select_reference(tmp_path, plan, "selected", "owned")
