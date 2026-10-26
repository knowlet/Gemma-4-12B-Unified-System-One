"""Current native179 RAW/V2 compiler preparation, in one owned CPU unit process.

Default dry. Frozen61de supplies only pure compiler workload functions; frozen03
supplies resource/processor/runtime components, not an old capacity qualification.
The caller must independently attest unit exit, deadline and whole-unit closure.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[2]
KIND = "jev_native_v2_fit_full179_single_cpu_prepare_v1"
RAW, V2 = "raw-json-v1", "shared-elements-v2"
COUNTS = {"train": 142, "development": 37, "total": 179}
WORKLOAD = {
    "path": "artifacts/jev-runtime/v2-action-training-prepare-20261021-01/training_prepare.py",
    "sha256": "61de4a5a037e5cce84fcf91989ef44fbcd13417659ce249d03dace9944756b59",
    "size_bytes": 27669,
}
SCRIPTS = {
    "scripts/jev_browser_fit_native_actions.py",
    "scripts/jev_browser_fit_actions.py",
    "scripts/jev_cfg_action_loss.py",
    "scripts/jev_local_model.py",
    "scripts/jev_browser_benchmark.py",
    "scripts/jev_browser_fit_public_actions.py",
    "scripts/jev_public_action_artifact.py",
    "scripts/jev_browser_fit_readout.py",
    "scripts/jev_browser_training_data.py",
    "scripts/breakthrough_training.py",
}
LIMITS = {
    "unit_memory_max": 3 * 1024**3,
    "unit_swap_max": 0,
    "single_worker_AS_soft": 8 * 1024**3,
    "single_worker_AS_hard": 8 * 1024**3,
    "worker_wall_seconds": 600,
    "external_unit_wall_seconds": 660,
    "file_bytes": 64 * 1024**2,
    "output_bytes": 1024**3,
    "swap_or_own_resident_credit_bytes": 0,
}


def libraries():
    path = ROOT / WORKLOAD["path"]
    blob = path.read_bytes()
    if (
        len(blob) != WORKLOAD["size_bytes"]
        or hashlib.sha256(blob).hexdigest() != WORKLOAD["sha256"]
        or any(p.is_symlink() for p in (path, *path.parents))
    ):
        raise ValueError("Frozen compiler workload component differs")
    spec = importlib.util.spec_from_file_location("_native179_workload", path)
    w = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = w
    exec(compile(blob, str(path), "exec"), w.__dict__)
    r, c = w.libraries()
    return w, r, c, c.module(r.CPU), c.module(c.WORKER)


def source_files(w, r, c, old):
    """Capture current membership; do not seed from the historical95 projection."""
    paths = set((ROOT / "src").rglob("*.py"))
    c.require(paths and len(paths) <= 256, "Bounded complete current src tree")
    c.require(
        not any(p.is_symlink() for p in (ROOT / "src").rglob("*")),
        "No omitted symlink entries in current src tree",
    )
    paths |= {ROOT / label for label in SCRIPTS}
    paths.add(Path(__file__).absolute())
    components = [WORKLOAD, w.REPEAT, r.CORE, r.CPU, c.WORKER, w.LEGACY]
    components += [old.fixed(label) for label in old.PINS]
    from_component = ROOT / (
        "artifacts/jev-runtime/v2-action-training-prepare-20261021-01/historical_data_projection.py"
    )
    paths.add(from_component)
    for ref in components:
        c.checked(ref)
        paths.add(ROOT / ref["path"])
    files = {}
    for path in sorted(paths):
        ref = c.pin(path)
        files[ref["path"]] = {k: v for k, v in ref.items() if k != "path"}
    return files


def make_plan(boot_id, *, nonce=None):
    w, r, c, cpu, old = libraries()
    unit = cpu.unit_binding({"boot_id": boot_id, "cgroup_path": c.CGROUP})
    return {
        "schema_id": KIND,
        "self_source": c.pin(Path(__file__)),
        "workload_component": WORKLOAD,
        "sources": source_files(w, r, c, old),
        "unit": unit,
        "nonce": uuid.uuid4().hex if nonce is None else nonce,
        "output": (HERE / "prepare-actual-01").relative_to(ROOT).as_posix(),
        "python": c.PYTHON,
        "environment": cpu.EXPECTED_ENVIRONMENT,
        "resource_limits": LIMITS,
        "lifecycle": "SINGLE_WORKER_MAINPID_EXTERNAL_UNIT_GUARDIAN_AND_CLOSURE_REQUIRED",
    }


def load_plan(path, expected):
    deps = libraries()
    w, r, c, cpu, old = deps
    blob = c.read(path)
    c.require(type(expected) is str and c.sha(blob) == expected, "External current plan SHA")
    plan = json.loads(blob)
    c.require(
        type(plan) is dict
        and set(plan) == set(make_plan(plan.get("unit", {}).get("boot_id")))
        and type(plan["nonce"]) is str
        and len(plan["nonce"]) == 32
        and all(x in "0123456789abcdef" for x in plan["nonce"]),
        "Strict new typed single-worker plan",
    )
    c.require(
        plan == make_plan(plan["unit"]["boot_id"], nonce=plan["nonce"]),
        "Exact current source membership, workload and independent lifecycle",
    )
    c.authenticate(plan["sources"])
    technical = old.parse(old.checked(old.fixed(old.TECHNICAL)))
    return (*deps, plan, technical)


def unit_properties(c, pid):
    required = {
        "MainPID": str(pid),
        "ControlGroup": c.CGROUP,
        "MemoryMax": str(3 * 1024**3),
        "MemoryHigh": "infinity",
        "MemorySwapMax": "0",
        "OOMPolicy": "kill",
        "KillMode": "control-group",
        "RuntimeMaxUSec": "11min",
        "TimeoutStopUSec": "10s",
        "Type": "exec",
    }
    result = subprocess.run(
        ["systemctl", "--user", "show", c.UNIT]
        + ["--property=" + k for k in [*required, "InvocationID"]],
        capture_output=True,
        timeout=5,
        check=True,
    )
    c.require(len(result.stdout) < 1024**2 and not result.stderr, "Bounded unit properties")
    props = dict(line.split("=", 1) for line in result.stdout.decode().splitlines())
    invocation = props.get("InvocationID", "")
    c.require(
        set(props) == {*required, "InvocationID"}
        and all(props[k] == v for k, v in required.items())
        and len(invocation) == 32
        and all(x in "0123456789abcdef" for x in invocation),
        "Exact MainPID and independently captured invocation",
    )
    return props


def entry_checks(c, cpu, old, plan, path, expected):
    c.require(
        sys.executable == c.PYTHON
        and sys.version_info[:2] == (3, 13)
        and all(os.environ.get(k) == v for k, v in plan["environment"].items())
        and os.environ.get("JEV_CPU3G_BOOT") == plan["unit"]["boot_id"]
        and os.environ.get("JEV_CPU3G_CGROUP") == c.CGROUP,
        "Exact offline CPU execution environment",
    )
    c.require(
        cpu.child_argv(os.getpid())
        == [
            c.PYTHON,
            "-B",
            "-s",
            str(Path(__file__).absolute()),
            "--run",
            "--plan",
            str(path),
            "--plan-sha256",
            expected,
        ],
        "Exact single-worker entry argv",
    )
    if os.getpgrp() != os.getpid() or os.getsid(0) != os.getpid():
        os.setsid()
    actor = old.identity(os.getpid())
    c.require(actor["pid"] == actor["pgid"] == actor["sid"], "Single owned worker session")
    unit = cpu.assert_current_unit(plan["unit"])
    c.require(unit["caller"] == actor, "Exact current unit caller")
    observation = cpu.memory_observation()
    formula = cpu.resource_formula(observation, plan["unit"]["boot_id"], prospective=True)
    c.require(observation["MemAvailable"] >= formula["required_bytes"], "Full reserve+3GiB")
    return actor, unit, observation, unit_properties(c, actor["pid"])


def validate_rows(c, w, data, refs, read_ref):
    c.require(len(refs) == 179 and data["counts"] == COUNTS, "Complete179 preparation")
    counts = {mode: {} for mode in (RAW, V2)}
    for index, (ref, case, wire) in enumerate(zip(refs, data["cases"], data["wires"], strict=True)):
        c.require(ref["path"] == f"rows/{index:03d}.json", "Unique ordered179 row artifacts")
        row = json.loads(read_ref(ref))
        c.require(
            row["ordinal"] == index
            and row["case_id"] == case.id
            and row["split"] == case.split
            and c.wire(row["wire"]) == c.wire(wire)
            and row["raw_byte_compatible"] is True
            and list(row["modes"]) == [RAW, V2],
            "All original ordered wires, teachers, splits and both modes",
        )
        raw = {k: v for k, v in row["modes"][RAW].items() if k != "cfg"}
        c.require(c.sha(c.wire(raw)) == row["legacy_raw_sha256"], "Legacy RAW byte parity")
        for mode, value in row["modes"].items():
            state = value["outcome"]
            c.require(state in {"COMPILED", "REFUSED"}, "Explicit compile result")
            if state == "REFUSED":
                c.require(
                    value["compiled"] is None
                    and value["cfg"] is None
                    and type(value["error"]) is dict,
                    "Retained compiler refusal",
                )
            else:
                item = value["compiled"]
                c.require(
                    item["inputs"] == item["record"]["processor_tensors"]
                    and item["positions"] == item["record"]["logits_positions"]
                    and item["targets"] == item["record"]["target_token_ids"],
                    "All native processor tensors, logits positions and targets",
                )
                cfg = value["cfg"]
                c.require(cfg["outcome"] in {"COMPILED", "REFUSED"}, "Explicit CFG result")
                if cfg["outcome"] == "COMPILED":
                    proof, inventory = cfg["plan"], cfg["inventory"]
                    c.require(
                        proof["compiled_row_sha256"] == c.sha(c.wire(item))
                        and proof["target_ids"] == item["targets"]
                        and proof["positions"] == item["positions"]
                        and all(
                            t in allowed
                            for t, allowed in zip(
                                item["targets"], proof["allowed_ids"], strict=True
                            )
                        )
                        and c.sha(c.wire(inventory["canonical_actions"]))
                        == proof["canonical_inventory_sha256"]
                        and c.sha(c.wire(inventory["encoded_actions"]))
                        == proof["encoded_inventory_sha256"]
                        and len(inventory["canonical_actions"])
                        == len(inventory["encoded_actions"])
                        == proof["legal_action_count"],
                        "Full CFG inventory and every original target",
                    )
                else:
                    c.require(
                        cfg["plan"] is None and type(cfg["error"]) is dict, "Retained CFG refusal"
                    )
                state += "/CFG_" + cfg["outcome"]
            counts[mode][state] = counts[mode].get(state, 0) + 1
        c.require(
            row["both_compiled_grammar_target_parity"]
            == w.pair_parity(c, row["modes"][RAW], row["modes"][V2]),
            "Explicit measured RAW/V2 full grammar/targets parity",
        )
    return {
        "cases": 179,
        "counts": counts,
        "raw_byte_compatible": True,
        "training_ready": counts[V2].get("COMPILED/CFG_COMPILED") == 179,
        "task_quality_attested": False,
    }


def validate_gates(c, cpu, plan, intent, blob):
    gates = [json.loads(line) for line in blob.splitlines()]
    c.require(
        [r["phase"] for r in gates]
        == [
            "startup",
            "source-before",
            "runtime-before",
            "cpu-reference-before-processor-load",
            "processor-after",
            *(["prepare"] * (179 * 6)),
            "final-cpu",
        ],
        "Complete original resource component phases",
    )
    previous = -1
    for index, row in enumerate(gates):
        cpu.validate_unit_evidence(row["unit"], plan["unit"])
        formula = cpu.resource_formula(
            row["observation"], plan["unit"]["boot_id"], prospective=True
        )
        c.require(
            row["schema_id"] == c.KIND + "_resource_gate"
            and row["sequence"] == index
            and row["admitted"] is True
            and row["error"] is None
            and 0 <= row["elapsed"] < 600
            and row["monotonic"] >= previous
            and row["unit"]["caller"] == intent["actor"]
            and cpu.unit_identity(row["unit"]) == cpu.unit_identity(intent["unit_before"])
            and row["observation"]["MemTotal"] == intent["observation"]["MemTotal"]
            and all(row[k] == v for k, v in formula.items())
            and row["observation"]["MemAvailable"] >= formula["required_bytes"]
            and row["limits"]
            == {"role": "worker", "soft_bytes": 8 * 1024**3, "hard_bytes": 8 * 1024**3}
            and row["cuda"]["CUDA_VISIBLE_DEVICES"] == ""
            and row["cuda"]["CUDA_initialized"] is False
            and type(row["cuda"]["torch_loaded"]) is bool,
            "Same unit/actor/full reserve and CPU-only8GiB component gates",
        )
        previous = row["monotonic"]
    return len(gates)


def execute(w, r, c, cpu, old, plan, technical, path, expected):
    actor, unit, observation, props = entry_checks(c, cpu, old, plan, path, expected)
    output, root = ROOT / plan["output"], ROOT / plan["output"] / "worker"
    c.require(not output.exists(), "Fresh independent preparation output")
    cpu.configure_limits("worker")
    output.mkdir(mode=0o700)
    root.mkdir(mode=0o700)
    intent = {
        "schema_id": KIND + "_intent",
        "plan_sha256": expected,
        "plan_path": str(path),
        "nonce": plan["nonce"],
        "actor": actor,
        "unit_before": unit,
        "observation": observation,
        "properties": props,
        "sources": plan["sources"],
    }
    cpu.save(output, "intent.json", c.wire(intent))
    gate = c.Gate(root, plan, cpu, observation["MemTotal"])
    rows, primary, data_evidence, review, after = [], None, {}, None, None
    runtime, processor_plan = None, old.parse(old.checked(old.fixed(old.PROCESSOR)))

    def save(label, blob):
        cpu.caps(output, len(blob))
        ref = cpu.save(root, label, blob)
        cpu.caps(output)
        return ref

    try:
        gate.check("startup", "cpu")
        gate.check("source-before", "cpu")
        c.authenticate(plan["sources"])
        q = old.module(old.Q)
        q._configure_environment("prepare")
        sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
        gate.check("runtime-before", "cpu")
        runtime = q._runtime(technical["technical"]["config"])
        save("runtime-before.json", c.wire(runtime))
        refs = cpu.capture_reference(processor_plan, root)
        model = cpu.load_processor(refs, gate)
        gate.check("processor-after", "cpu")
        before = c.wire(cpu.processor_evidence(model, refs))
        save("processor-before.json", before)
        from s1 import browser_native_action_data as native

        data = native.load_historical_data(ROOT)
        data_evidence = {
            k: data[k] for k in ("refs", "train_ids", "dev_ids", "counts", "provenance")
        }
        current = c.module(
            {
                "path": "scripts/jev_browser_fit_actions.py",
                **plan["sources"]["scripts/jev_browser_fit_actions.py"],
            }
        )
        cfg = c.module(
            {
                "path": "scripts/jev_cfg_action_loss.py",
                **plan["sources"]["scripts/jev_cfg_action_loss.py"],
            }
        )
        legacy = c.module(w.LEGACY)
        w.prepare_rows(
            c,
            data,
            model,
            current.compile_actions,
            legacy.compile_actions,
            cfg.compile_action_plan,
            gate,
            save,
            rows=rows,
            inventory=lambda row, payload, model, mode: w.grammar_inventory(
                row, payload, model, mode, cfg.LegalActionTokenTrie
            ),
        )
        native.verify_original(data)
        c.require(source_files(w, r, c, old) == plan["sources"], "Current source membership stable")
        c.authenticate(plan["sources"], loaded=True)
        runtime_after = q._runtime(technical["technical"]["config"])
        after_processor = c.wire(cpu.processor_evidence(model, refs))
        c.require(
            before == after_processor
            and len(refs) == 58
            and len(runtime["files"]) == 348
            and c.wire(runtime_after) == c.wire(runtime),
            "Unchanged processor58/runtime348 components",
        )
        save("processor-after.json", after_processor)
        save("runtime-after.json", c.wire(runtime_after))
        gate.check("final-cpu", "cpu")
        review = validate_rows(c, w, data, rows, lambda ref: old.checked(ref, root))
        validate_gates(c, cpu, plan, intent, c.read(root / "resource-gates.jsonl"))
        after = cpu.assert_current_unit(plan["unit"])
        c.require(
            after["caller"] == actor and cpu.unit_identity(after) == cpu.unit_identity(unit),
            "Final same single worker and unit inode",
        )
    except BaseException as exc:
        primary = exc
    result = {
        "schema_id": KIND + "_terminal",
        "plan_sha256": expected,
        "nonce": plan["nonce"],
        "actor": actor,
        "InvocationID": props["InvocationID"],
        "rows": rows,
        "data": data_evidence,
        "sources": plan["sources"],
        "runtime_files": None if runtime is None else len(runtime["files"]),
        "processor_reference_files": len(processor_plan["reference_items"]),
        "unit_after": after,
        "review": review,
        "complete_pending_external_unit_exit": primary is None and len(rows) == 179,
        "primary_error": None
        if primary is None
        else {"type": type(primary).__name__, "message": str(primary)[:2048]},
        "resource_component": c.KIND,
        "workload_component": WORKLOAD,
        "lifecycle": plan["lifecycle"],
        "processor_only": True,
        "native_model_forward_train_HTTP_browser_calls": 0,
        "CUDA_initialization_evidence": "RECORDED_RESOURCE_GATES_AND_PROCESSOR_CHECKS",
        "old_capacity_qualification_attested": False,
        "task_quality_attested": False,
    }
    try:
        cpu.save(output, "terminal.json", c.wire(result), terminal=True)
    except BaseException as secondary:
        if primary is not None:
            primary.add_note("Terminal also failed: " + repr(secondary))
            raise primary from secondary
        raise
    if primary is not None:
        raise primary
    return result


def validate_saved(path, expected):
    w, r, c, cpu, old, plan, _ = load_plan(path, expected)
    output = ROOT / plan["output"]
    intent, result = (
        json.loads(c.read(output / name)) for name in ("intent.json", "terminal.json")
    )
    c.require(
        intent["schema_id"] == KIND + "_intent"
        and result["schema_id"] == KIND + "_terminal"
        and intent["plan_sha256"] == result["plan_sha256"] == expected
        and intent["nonce"] == result["nonce"] == plan["nonce"]
        and intent["sources"] == result["sources"] == plan["sources"]
        and result["actor"] == intent["actor"] == intent["unit_before"]["caller"]
        and result["InvocationID"] == intent["properties"]["InvocationID"]
        and result["complete_pending_external_unit_exit"] is True
        and result["primary_error"] is None
        and result["runtime_files"] == 348
        and result["processor_reference_files"] == 58,
        "New single-worker complete receipt; external exit still required",
    )
    cpu.validate_unit_evidence(result["unit_after"], plan["unit"])
    c.require(
        result["unit_after"]["caller"] == intent["actor"]
        and cpu.unit_identity(result["unit_after"]) == cpu.unit_identity(intent["unit_before"]),
        "Same final unit/worker identity",
    )
    c.require(
        result["resource_component"] == c.KIND
        and result["workload_component"] == WORKLOAD
        and result["lifecycle"] == plan["lifecycle"]
        and result["processor_only"] is True
        and result["native_model_forward_train_HTTP_browser_calls"] == 0
        and result["old_capacity_qualification_attested"] is False
        and result["task_quality_attested"] is False
        and intent["properties"]["MainPID"] == str(intent["actor"]["pid"]),
        "New workload scope and actual MainPID binding",
    )
    sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
    from s1 import browser_native_action_data as native

    data = native.load_historical_data(ROOT)
    c.require(
        result["data"]
        == {k: data[k] for k in ("refs", "train_ids", "dev_ids", "counts", "provenance")},
        "Same full native data",
    )
    root = output / "worker"
    review = validate_rows(c, w, data, result["rows"], lambda ref: old.checked(ref, root))
    validate_gates(c, cpu, plan, intent, c.read(root / "resource-gates.jsonl"))
    c.require(
        c.read(root / "processor-before.json") == c.read(root / "processor-after.json")
        and review == result["review"],
        "Unchanged processor and independently derived review",
    )
    processor = json.loads(c.read(root / "processor-before.json"))
    c.require(
        processor["tokenizer_is_processor_tokenizer"] is True
        and processor["max_context"] == 8192
        and processor["generation_EOS"] == [1, 106, 50]
        and processor["CPU_threads"] == processor["CPU_interop_threads"] == 2,
        "Original complete processor/context/EOS/threads contract",
    )
    runtime_blob = c.read(root / "runtime-before.json")
    c.require(
        runtime_blob == c.read(root / "runtime-after.json")
        and len(json.loads(runtime_blob)["files"]) == 348,
        "Original runtime348 before/after byte equality",
    )
    for label, pin in old.parse(old.checked(old.fixed(old.PROCESSOR)))["reference_items"].items():
        old.checked({"path": label, **pin}, root / "reference-environment")
    return review


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--plan")
    parser.add_argument("--plan-sha256")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if args.plan is None:
        if args.run:
            raise ValueError("Explicit current native179 plan required")
        print(json.dumps({"schema_id": KIND, "processor_calls": 0, "status": "PLAN_REQUIRED"}))
        return 0
    path = Path(args.plan).absolute()
    deps = load_plan(path, args.plan_sha256)
    w, r, c, cpu, old, plan, technical = deps
    if not args.run:
        print(
            json.dumps(
                {
                    "schema_id": KIND,
                    "source_files": len(plan["sources"]),
                    "processor_calls": 0,
                    "single_worker": True,
                    "output_exists": (ROOT / plan["output"]).exists(),
                }
            )
        )
        return 0
    previous = signal.signal(
        signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("CPU600s deadline"))
    )
    signal.alarm(600)
    try:
        execute(w, r, c, cpu, old, plan, technical, path, args.plan_sha256)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
