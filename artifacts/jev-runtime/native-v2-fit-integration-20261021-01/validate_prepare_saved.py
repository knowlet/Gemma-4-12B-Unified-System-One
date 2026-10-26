"""Independent post-exit native179 replay; never rerun or rewrite the actor.

The frozen actor's posthoc Python equality check refused JSON lists versus the
historical reader's tuples. This verifier compares complete ordered JSON bytes,
then independently joins every saved row, resource component and external exit.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[2]
KIND = "jev_native179_postexit_ordered_wire_replay_v1"
ACTOR = {
    "path": (HERE / "prepare_cpu.py").relative_to(ROOT).as_posix(),
    "sha256": "fe3efa783ee9f1227c4ddf6c00144227069a7974b2a8a7deee4637e4bc42dc8c",
    "size_bytes": 23501,
}


def actor_module():
    path = ROOT / ACTOR["path"]
    blob = path.read_bytes()
    if (
        len(blob) != ACTOR["size_bytes"]
        or hashlib.sha256(blob).hexdigest() != ACTOR["sha256"]
        or any(p.is_symlink() for p in (path, *path.parents))
    ):
        raise ValueError("Exact frozen single-worker actor source required")
    spec = importlib.util.spec_from_file_location("_postexit_native179_actor", path)
    d = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = d
    exec(compile(blob, str(path), "exec"), d.__dict__)
    return d


def ordered_data_join(c, saved, fresh):
    """JSON semantics retain every field/value/key order; tuple/list is only storage."""
    expected = {k: fresh[k] for k in ("refs", "train_ids", "dev_ids", "counts", "provenance")}
    saved_bytes, fresh_bytes = c.wire(saved), c.wire(expected)
    c.require(saved_bytes == fresh_bytes, "Complete ordered native data bytes differ")
    return {
        "ordered_data_evidence_equal": True,
        "saved_sha256": c.sha(saved_bytes),
        "fresh_sha256": c.sha(fresh_bytes),
        "python_container_equal": saved == expected,
    }


def closed_join(c, cpu, plan, expected, intent, result, closure):
    actor = result["actor"]
    c.require(
        intent["schema_id"] == plan["schema_id"] + "_intent"
        and result["schema_id"] == plan["schema_id"] + "_terminal"
        and intent["plan_sha256"] == result["plan_sha256"] == expected
        and intent["nonce"] == result["nonce"] == plan["nonce"]
        and intent["sources"] == result["sources"] == plan["sources"]
        and actor == intent["actor"] == intent["unit_before"]["caller"]
        and actor["pid"] == actor["pgid"] == actor["sid"]
        and intent["properties"]["MainPID"] == str(actor["pid"])
        and result["InvocationID"] == intent["properties"]["InvocationID"]
        and result["complete_pending_external_unit_exit"] is True
        and result["primary_error"] is None
        and result["runtime_files"] == 348
        and result["processor_reference_files"] == 58
        and result["processor_only"] is True
        and result["native_model_forward_train_HTTP_browser_calls"] == 0
        and result["old_capacity_qualification_attested"] is False
        and result["task_quality_attested"] is False
        and result["lifecycle"] == plan["lifecycle"],
        "Frozen successful single-worker receipt and exact plan/nonce/identity",
    )
    for unit in (intent["unit_before"], result["unit_after"]):
        cpu.validate_unit_evidence(unit, plan["unit"])
        c.require(
            unit["caller"] == actor
            and cpu.unit_identity(unit) == cpu.unit_identity(intent["unit_before"]),
            "Exact initial/final unit inode and worker",
        )
    exit_proof = closure["unit_exit"]
    state, stopped = exit_proof["terminal_before_stop"], exit_proof["after_stop"]
    c.require(
        closure["actors"] == {"worker": actor}
        and len(closure["two_proc_scans"]) == 2
        and all(row == {"matches": [], "unreadable": []} for row in closure["two_proc_scans"])
        and closure["source_files_unchanged"] == len(plan["sources"])
        and closure["worker_primary_error"] is None
        and closure["review_error"] == {"type": "ValueError", "message": "Same full native data"}
        and closure["review"] is None
        and closure["task_quality_attested"] is False
        and closure["final_postexit_cgroup_counters"] is None
        and exit_proof["reason"] == "terminal"
        and exit_proof["primary_error"] is None
        and exit_proof["invocation_id"] == result["InvocationID"] == state["InvocationID"]
        and state["ExecMainPID"] == str(actor["pid"])
        and state["ExecMainCode"] == "1"
        and state["ExecMainStatus"] == "0"
        and state["Result"] == "success"
        and state["MainPID"] == "0"
        and state["ControlGroup"] == ""
        and state["SubState"] == "exited"
        and exit_proof["cgroup_absent_after_stop"] is True
        and stopped["MainPID"] == "0"
        and stopped["ControlGroup"] == ""
        and stopped["LoadState"] == "not-found",
        "Independent unit exit0, exact invocation/ExecMainPID and whole recorded absence",
    )


def native_records(c, data, refs, read_ref, *, native_record, native_cfg, v2):
    c.require(
        len(refs) == len(data["cases"]) == len(data["wires"]) == 179,
        "All179 strict native artifact replays",
    )
    record_count, cfg_count = 0, 0
    for ref, case, wire in zip(refs, data["cases"], data["wires"], strict=True):
        row = json.loads(read_ref(ref))
        value = row["modes"][v2]
        if value["outcome"] != "COMPILED":
            continue
        record = value["compiled"]["record"]
        labels = native_record(record, case, wire)
        record_count += 1
        if value["cfg"]["outcome"] == "COMPILED":
            cfg = value["cfg"]
            native_cfg(
                {"case_id": case.id, "plan": cfg["plan"], "inventory": cfg["inventory"]},
                record,
                labels,
            )
            cfg_count += 1
    return {
        "compiled_native_records_verified": record_count,
        "compiled_native_CFG_inventories_verified": cfg_count,
        "not_independent_retokenization": True,
    }


def validate(plan_path, expected, closure_path, closure_sha256):
    d = actor_module()
    w, r, c, cpu, old, plan, technical = d.load_plan(plan_path, expected)
    closure_blob = c.read(closure_path)
    c.require(
        type(closure_sha256) is str and c.sha(closure_blob) == closure_sha256,
        "Externally selected immutable root closure SHA",
    )
    closure = json.loads(closure_blob)
    output = ROOT / plan["output"]
    intent, result = (
        json.loads(c.read(output / name)) for name in ("intent.json", "terminal.json")
    )
    closed_join(c, cpu, plan, expected, intent, result, closure)
    c.require(
        result["resource_component"] == c.KIND and result["workload_component"] == d.WORKLOAD,
        "Named components only",
    )
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
    from s1 import browser_action_adapter as adapter
    from s1 import browser_native_action_data as native

    data = native.load_historical_data(ROOT)
    data_join = ordered_data_join(c, result["data"], data)
    native.verify_original(data)
    root = output / "worker"

    def read_ref(ref):
        return old.checked(ref, root)

    review = d.validate_rows(c, w, data, result["rows"], read_ref)
    gate_count = d.validate_gates(c, cpu, plan, intent, c.read(root / "resource-gates.jsonl"))
    c.require(c.wire(review) == c.wire(result["review"]), "Independent full179 measured summary")
    processor_blob = c.read(root / "processor-before.json")
    c.require(
        processor_blob == c.read(root / "processor-after.json"), "Identical processor evidence"
    )
    processor = json.loads(processor_blob)
    c.require(
        processor["tokenizer_is_processor_tokenizer"] is True
        and processor["max_context"] == 8192
        and processor["generation_EOS"] == [1, 106, 50]
        and processor["CPU_threads"] == processor["CPU_interop_threads"] == 2,
        "Original complete processor/context/EOS/thread settings",
    )
    reference = old.parse(old.checked(old.fixed(old.PROCESSOR)))
    c.require(len(reference["reference_items"]) == 58, "Complete original processor58")
    for label, pin in reference["reference_items"].items():
        old.checked({"path": label, **pin}, root / "reference-environment")
    runtime_blob = c.read(root / "runtime-before.json")
    runtime = json.loads(runtime_blob)
    c.require(
        runtime_blob == c.read(root / "runtime-after.json") and len(runtime["files"]) == 348,
        "Original runtime348 recorded before/after equality",
    )
    # Verify the captured original runtime authority, not merely the number348.
    runtime_config = technical["technical"]["config"]
    original_runtime = c.read(Path(runtime_config["runtime_path"]))
    c.require(
        c.sha(original_runtime) == runtime_config["runtime_sha256"]
        and c.wire(json.loads(original_runtime)) == c.wire(runtime),
        "Recorded runtime equals original348 pinned authority",
    )
    strict = native_records(
        c,
        data,
        result["rows"],
        read_ref,
        native_record=adapter._native_record,
        native_cfg=adapter._native_cfg,
        v2=d.V2,
    )
    c.require(
        d.source_files(w, r, c, old) == plan["sources"], "Same complete current source membership"
    )
    c.authenticate(plan["sources"], loaded=True)
    frameworks = sorted(set(sys.modules) & {"torch", "transformers", "tokenizers", "peft"})
    c.require(not frameworks, "Post-exit replay must not load processor/model frameworks")
    return {
        "schema_id": KIND,
        "actor_source": ACTOR,
        "verifier_source": c.pin(Path(__file__)),
        "plan_sha256": expected,
        "external_closure_sha256": closure_sha256,
        "InvocationID": result["InvocationID"],
        "actor": result["actor"],
        "source_files_unchanged": len(plan["sources"]),
        "data_join": data_join,
        "original_179_full_wires_sha256": c.sha(c.wire(data["wires"])),
        "ordered_saved_179_full_wires_verified": True,
        "resource_gate_count": gate_count,
        "processor_references": 58,
        "runtime_files": 348,
        "review": review,
        "strict_native_artifact_replay": strict,
        "unit_exit_code": 0,
        "recorded_whole_closure_verified": True,
        "original_posthoc_error": closure["review_error"],
        "original_wrapper_exit_rewritten": False,
        "final_postexit_cgroup_counters_available": False,
        "new_processor_model_forward_training_browser_calls": 0,
        "framework_modules_loaded": frameworks,
        "numpy_loaded": "numpy" in sys.modules,
        "task_quality_attested": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--external-closure", required=True, type=Path)
    parser.add_argument("--external-closure-sha256", required=True)
    args = parser.parse_args(argv)
    result = validate(
        args.plan.absolute(),
        args.plan_sha256,
        args.external_closure.absolute(),
        args.external_closure_sha256,
    )
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
