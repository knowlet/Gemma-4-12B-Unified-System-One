"""Separately typed RAW/V2 action compiler preparation; default dry, no model.

The frozen03 resource gate and owned CPU lifecycle remain named components.
Historical data and legacy RAW compilation are references, not old qualification.
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
KIND = "jev_v2_action_training_processor_prepare_v1"
RAW, V2 = "raw-json-v1", "shared-elements-v2"
REPEAT = {
    "path": "artifacts/jev-runtime/shared-elements-capacity-20261021-01/repeat_capacity.py",
    "sha256": "63ced1c0fec638aea11793209f3fe795faa46fccbccf41e988f45a8a6116dd1f",
    "size_bytes": 31295,
}
PRODUCTION = {
    "src/s1/jev_generated.py",
    "src/s1/api.py",
    "src/s1/jev_prompt.py",
    "scripts/jev_browser_fit_actions.py",
    "scripts/jev_cfg_action_loss.py",
}
LEGACY = {
    "path": str((HERE / "legacy_fit_actions.py").relative_to(ROOT)),
    "sha256": "4bf6585008e434a469dafb4a9997dee595b079044e03c9afe00ceffa5e031f82",
    "size_bytes": 35101,
}


def libraries():
    path = ROOT / REPEAT["path"]
    blob = path.read_bytes()
    if (
        len(blob) != REPEAT["size_bytes"]
        or hashlib.sha256(blob).hexdigest() != REPEAT["sha256"]
        or any(p.is_symlink() for p in (path, *path.parents))
    ):
        raise ValueError("Frozen V2 lifecycle library differs")
    spec = importlib.util.spec_from_file_location("_training_prepare_repeat", path)
    r = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = r
    exec(compile(blob, str(path), "exec"), r.__dict__)
    return r, r.core()


def make_plan(boot_id):
    r, c = libraries()
    cpu = c.module(r.CPU)
    return {
        "schema_id": KIND,
        "self_source": c.pin(Path(__file__)),
        "data_source": c.pin(HERE / "historical_data_projection.py"),
        "legacy_source": LEGACY,
        "production_sources": {
            p: {k: v for k, v in c.pin(ROOT / p).items() if k != "path"} for p in sorted(PRODUCTION)
        },
        "unit": {"boot_id": boot_id, "cgroup_path": c.CGROUP},
        "output": str((HERE / "actual-01").relative_to(ROOT)),
        "python": c.PYTHON,
        "environment": cpu.EXPECTED_ENVIRONMENT,
        "resource_limits": c.LIMITS,
    }


def source_union(r, c, plan, old):
    technical = old.parse(old.checked(old.fixed(old.TECHNICAL)))
    execution = old.parse(old.checked(technical["technical"]["execution"]))
    original = execution["files"]
    c.require(
        len(original) == 95
        and PRODUCTION - {"src/s1/jev_prompt.py", "scripts/jev_cfg_action_loss.py"}
        <= set(original),
        "Original95 is an independent prior authority",
    )
    files = dict(original)
    files.update(plan["production_sources"])
    cohort = old.parse(old.checked(old.fixed(old.COHORT)))
    for path, pin in cohort["source_union"].items():
        path = (ROOT / path).absolute()
        c.require(path.is_relative_to(ROOT) and ".." not in path.parts, "Confined source")
        path = path.relative_to(ROOT).as_posix()
        if path not in PRODUCTION:
            c.require(path not in files or files[path] == pin, "One source authority")
            files[path] = pin
    for ref in [
        REPEAT,
        r.CORE,
        r.CPU,
        c.WORKER,
        c.CONTROLLER,
        plan["self_source"],
        plan["data_source"],
        LEGACY,
        *[old.fixed(p) for p in old.PINS],
    ]:
        pin = {k: v for k, v in ref.items() if k != "path"}
        c.require(ref["path"] not in files or files[ref["path"]] == pin, "One current pin")
        files[ref["path"]] = pin
    c.require(
        {p for p, pin in original.items() if files[p] != pin} == PRODUCTION & set(original),
        "Only explicitly declared current production sources replace prior pins",
    )
    return dict(sorted(files.items())), technical


def load_plan(path, expected):
    r, c = libraries()
    blob = c.read(path)
    c.require(type(expected) is str and c.sha(blob) == expected, "External preparation plan SHA")
    plan = json.loads(blob)
    c.require(
        set(plan)
        == {
            "schema_id",
            "self_source",
            "data_source",
            "legacy_source",
            "production_sources",
            "unit",
            "output",
            "python",
            "environment",
            "resource_limits",
        }
        and plan["schema_id"] == KIND,
        "New typed plan",
    )
    c.require(
        ROOT / plan["self_source"]["path"] == Path(__file__).absolute()
        and ROOT / plan["data_source"]["path"] == HERE / "historical_data_projection.py"
        and plan["legacy_source"] == LEGACY,
        "Exact entry/data/legacy origins",
    )
    cpu, old = c.module(r.CPU), c.module(c.WORKER)
    cpu.unit_binding(plan["unit"])
    c.require(
        plan["python"] == c.PYTHON
        and plan["environment"] == cpu.EXPECTED_ENVIRONMENT
        and plan["resource_limits"] == c.LIMITS
        and set(plan["production_sources"]) == PRODUCTION
        and ROOT / plan["output"] == HERE / "actual-01",
        "Same bounded CPU resource contract",
    )
    files, technical = source_union(r, c, plan, old)
    c.authenticate(files)
    return r, c, plan, cpu, old, files, technical


def snapshot(row):
    return {
        "record": row["record"],
        "inputs": {k: v.tolist() for k, v in row["inputs"].items()},
        "positions": row["positions"].tolist(),
        "targets": row["targets"].tolist(),
    }


def compile_one(compiler, model, case, wire, *, mode=None):
    try:
        args = {} if mode is None else {"prompt_format": mode}
        rows = compiler(
            model,
            [case],
            {
                "blobs": {
                    "wire_requests.jsonl": json.dumps(
                        wire, ensure_ascii=False, separators=(",", ":")
                    ).encode()
                }
            },
            **args,
        )
        if len(rows) != 1:
            raise ValueError("One complete compiled case required")
        return {"outcome": "COMPILED", "compiled": snapshot(rows[0]), "error": None}, rows[0]
    except ValueError as exc:
        return {
            "outcome": "REFUSED",
            "compiled": None,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }, None


def grammar_inventory(row, payload, model, mode, grammar):
    from s1.jev_generated import format_action_request

    _, _, labels, _ = format_action_request(
        payload, row["record"]["inference_model_identity"], prompt_format=mode
    )
    tree = grammar(
        labels,
        model.tok,
        prompt_length=row["record"]["prompt_tokens"],
        eos_token_id=row["record"]["native_stop_token_id"],
        max_new_tokens=32,
    )
    return {
        "labels": labels,
        "canonical_actions": tree.canonical_actions,
        "encoded_actions": tree.encoded_actions,
        "telemetry": tree.telemetry(),
    }


def add_cfg(value, row, wire, model, cfg, mode, inventory):
    if row is None:
        value["cfg"] = None
        return
    payload = dict(wire["payload"])
    payload["model"] = row["record"]["inference_model_identity"]
    try:
        plan = cfg(row, payload, model.tok, prompt_format=mode).evidence()
        full = inventory(row, payload, model, mode)
        value["cfg"] = {"outcome": "COMPILED", "plan": plan, "inventory": full, "error": None}
    except ValueError as exc:
        value["cfg"] = {
            "outcome": "REFUSED",
            "plan": None,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }


def pair_parity(c, raw, v2):
    if not all(
        value["outcome"] == "COMPILED" and value["cfg"]["outcome"] == "COMPILED"
        for value in (raw, v2)
    ):
        return None
    c.require(
        all(
            c.wire(raw["cfg"]["inventory"][k]) == c.wire(v2["cfg"]["inventory"][k])
            for k in ("labels", "canonical_actions", "encoded_actions")
        )
        and all(
            c.wire(raw["cfg"]["plan"][k]) == c.wire(v2["cfg"]["plan"][k])
            for k in ("target_ids", "decision_indices", "allowed_ids")
        )
        and c.wire(raw["compiled"]["targets"]) == c.wire(v2["compiled"]["targets"]),
        "RAW/V2 full grammar and native targets parity; positions may shift",
    )
    return True


def prepare_rows(c, data, model, current, legacy, cfg, gate, save, *, rows=None, inventory=None):
    rows = [] if rows is None else rows
    c.require(
        data["counts"] == {"train": 142, "development": 37, "total": 179}
        and len(data["cases"]) == len(data["wires"]) == 179,
        "All179 historical inputs",
    )
    for ordinal, (case, wire) in enumerate(zip(data["cases"], data["wires"], strict=True)):
        original_wire = c.wire(wire)
        gate.check("prepare", "cpu")
        prior, _ = compile_one(legacy, model, case, wire)
        gate.check("prepare", "cpu")
        raw, raw_row = compile_one(current, model, case, wire, mode=RAW)
        c.require(c.wire(raw) == c.wire(prior), "Full historical RAW compiler byte compatibility")
        prior_sha = c.sha(c.wire(prior))
        gate.check("prepare", "cpu")
        add_cfg(raw, raw_row, wire, model, cfg, RAW, inventory)
        gate.check("prepare", "cpu")
        v2, v2_row = compile_one(current, model, case, wire, mode=V2)
        gate.check("prepare", "cpu")
        add_cfg(v2, v2_row, wire, model, cfg, V2, inventory)
        gate.check("prepare", "cpu")
        c.require(c.wire(wire) == original_wire, "Complete historical wire remains unchanged")
        item = {
            "ordinal": ordinal,
            "case_id": case.id,
            "split": case.split,
            "wire": wire,
            "legacy_raw_sha256": prior_sha,
            "raw_byte_compatible": True,
            "both_compiled_grammar_target_parity": pair_parity(c, raw, v2),
            "modes": {RAW: raw, V2: v2},
        }
        rows.append(save(f"rows/{ordinal:03d}.json", c.wire(item)))
    return rows


def worker(r, c, plan, expected, nonce, cpu, old, files, technical):
    output = ROOT / plan["output"]
    intent = json.loads(c.read(output / "intent.json"))
    c.require(
        intent["nonce"] == nonce
        and intent["plan_sha256"] == expected
        and intent["parent"] == old.identity(os.getppid()),
        "Exact owned parent identity",
    )
    c.require(
        cpu.child_argv(os.getppid())
        == [
            c.PYTHON,
            "-B",
            "-s",
            str(Path(__file__).absolute()),
            "--run",
            "--plan",
            intent["plan_path"],
            "--plan-sha256",
            expected,
        ],
        "New parent argv",
    )
    child = old.identity(os.getpid())
    c.require(child["pid"] == child["pgid"] == child["sid"], "Own worker session")
    root = output / "worker"
    c.require(not root.exists(), "Fresh worker output")
    root.mkdir(mode=0o700)
    cpu.configure_limits("worker")
    gate = c.Gate(root, plan, cpu, intent["mem_total"])
    rows, primary, evidence = [], None, {}
    try:
        gate.check("startup", "cpu")
        gate.check("source-before", "cpu")
        c.authenticate(files)
        q = old.module(old.Q)
        q._configure_environment("prepare")
        sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
        gate.check("runtime-before", "cpu")
        runtime = q._runtime(technical["technical"]["config"])
        refs = cpu.capture_reference(old.parse(old.checked(old.fixed(old.PROCESSOR))), root)
        model = cpu.load_processor(refs, gate)
        gate.check("processor-after", "cpu")
        before = c.wire(cpu.processor_evidence(model, refs))
        cpu.save(root, "processor-before.json", before)
        data = c.module(plan["data_source"]).load_historical_data(ROOT)
        current = c.module(
            {
                "path": "scripts/jev_browser_fit_actions.py",
                **plan["production_sources"]["scripts/jev_browser_fit_actions.py"],
            }
        )
        cfg = c.module(
            {
                "path": "scripts/jev_cfg_action_loss.py",
                **plan["production_sources"]["scripts/jev_cfg_action_loss.py"],
            }
        )
        legacy = c.module(LEGACY)
        evidence = {k: data[k] for k in ("refs", "train_ids", "dev_ids", "counts", "provenance")}
        rows = prepare_rows(
            c,
            data,
            model,
            current.compile_actions,
            legacy.compile_actions,
            cfg.compile_action_plan,
            gate,
            lambda label, blob: cpu.save(root, label, blob),
            rows=rows,
            inventory=lambda row, payload, model, mode: grammar_inventory(
                row, payload, model, mode, cfg.LegalActionTokenTrie
            ),
        )
        c.authenticate(files, loaded=True)
        q._runtime(technical["technical"]["config"])
        after = c.wire(cpu.processor_evidence(model, refs))
        c.require(
            before == after and len(refs) == 58 and len(runtime["files"]) == 348,
            "Unchanged processor58/runtime348",
        )
        cpu.save(root, "processor-after.json", after)
        gate.check("final-cpu", "cpu")
    except BaseException as exc:
        primary = exc
    result = {
        "schema_id": KIND + "_worker",
        "plan_sha256": expected,
        "nonce": nonce,
        "child": child,
        "rows": rows,
        "data": evidence,
        "sources": files,
        "complete": primary is None and len(rows) == 179,
        "primary_error": None
        if primary is None
        else {"type": type(primary).__name__, "message": str(primary)[:2048]},
        "resource_component": c.KIND,
        "processor_only": True,
        "native_model_forward_train_HTTP_browser_calls": 0,
        "task_quality_attested": False,
    }
    try:
        cpu.save(root, "terminal.json", c.wire(result), terminal=True)
    except BaseException as secondary:
        if primary is not None:
            primary.add_note("Terminal also failed: " + repr(secondary))
            raise primary from secondary
        raise
    if primary is not None:
        raise primary


def validate(c, plan, expected, nonce, cpu, old, files, output, cleanup):
    c.require(cleanup["confirmed"] is True and cleanup["remaining"] == [], "Whole owned closure")
    root = output / "worker"
    result = json.loads(c.read(root / "terminal.json"))
    c.require(
        result["schema_id"] == KIND + "_worker"
        and result["plan_sha256"] == expected
        and result["nonce"] == nonce
        and result["complete"] is True
        and result["primary_error"] is None
        and result["sources"] == files
        and result["processor_only"] is True
        and result["native_model_forward_train_HTTP_browser_calls"] == 0
        and result["task_quality_attested"] is False
        and len(result["rows"]) == 179,
        "Closed full179 compiler worker",
    )
    child = result["child"]
    c.require(
        cleanup["tracked_PID_start_ticks"].get(str(child["pid"])) == child["start_ticks"]
        and child["pgid"] == child["sid"] == child["pid"]
        and child["pgid"] in cleanup["tracked_groups"],
        "Exact closed worker join",
    )
    data = c.module(plan["data_source"]).load_historical_data(ROOT)
    intent = json.loads(c.read(output / "intent.json"))
    c.require(
        result["data"]
        == {k: data[k] for k in ("refs", "train_ids", "dev_ids", "counts", "provenance")},
        "Same historical data authority",
    )
    counts = {mode: {} for mode in (RAW, V2)}
    for index, (ref, case, wire) in enumerate(
        zip(result["rows"], data["cases"], data["wires"], strict=True)
    ):
        row = json.loads(old.checked(ref, root))
        c.require(
            row["ordinal"] == index
            and row["case_id"] == case.id
            and row["split"] == case.split
            and c.wire(row["wire"]) == c.wire(wire)
            and row["raw_byte_compatible"] is True
            and set(row["modes"]) == {RAW, V2},
            "Full ordered historical row/wire/modes",
        )
        raw = {k: v for k, v in row["modes"][RAW].items() if k != "cfg"}
        c.require(c.sha(c.wire(raw)) == row["legacy_raw_sha256"], "Exact legacy RAW join")
        for mode, value in row["modes"].items():
            state = value["outcome"]
            c.require(state in {"COMPILED", "REFUSED"}, "Explicit compile outcome")
            if state == "COMPILED":
                item = value["compiled"]
                c.require(
                    item["inputs"] == item["record"]["processor_tensors"]
                    and item["positions"] == item["record"]["logits_positions"]
                    and item["targets"] == item["record"]["target_token_ids"],
                    "Complete tensors/targets",
                )
                cfg = value["cfg"]
                c.require(cfg["outcome"] in {"COMPILED", "REFUSED"}, "Explicit CFG outcome")
                if cfg["outcome"] == "COMPILED":
                    proof = cfg["plan"]
                    inventory = cfg["inventory"]
                    c.require(
                        proof["compiled_row_sha256"] == c.sha(c.wire(item))
                        and list(proof["target_ids"]) == item["targets"]
                        and list(proof["positions"]) == item["positions"]
                        and all(
                            t in allowed
                            for t, allowed in zip(
                                item["targets"], proof["allowed_ids"], strict=True
                            )
                        ),
                        "CFG joins every unchanged target and full compiled snapshot",
                    )
                    c.require(
                        c.sha(c.wire(inventory["canonical_actions"]))
                        == proof["canonical_inventory_sha256"]
                        and c.sha(c.wire(inventory["encoded_actions"]))
                        == proof["encoded_inventory_sha256"]
                        and len(inventory["canonical_actions"])
                        == len(inventory["encoded_actions"])
                        == proof["legal_action_count"],
                        "Full unpruned grammar inventory joins CFG",
                    )
                state += "/CFG_" + cfg["outcome"]
            counts[mode][state] = counts[mode].get(state, 0) + 1
        c.require(
            row["both_compiled_grammar_target_parity"]
            == pair_parity(c, row["modes"][RAW], row["modes"][V2]),
            "Saved measured pair parity",
        )
    c.require(
        c.read(root / "processor-before.json") == c.read(root / "processor-after.json"),
        "Processor evidence stable",
    )
    gates = [json.loads(line) for line in c.read(root / "resource-gates.jsonl").splitlines()]
    c.require(
        [row["phase"] for row in gates]
        == [
            "startup",
            "source-before",
            "runtime-before",
            "cpu-reference-before-processor-load",
            "processor-after",
        ]
        + ["prepare"] * (179 * 6)
        + ["final-cpu"],
        "Complete component resource gate sequence",
    )
    for index, row in enumerate(gates):
        c.require(
            row["schema_id"] == c.KIND + "_resource_gate"
            and row["sequence"] == index
            and row["admitted"] is True
            and row["elapsed"] < 600
            and row["cuda"]["CUDA_VISIBLE_DEVICES"] == ""
            and row["cuda"]["CUDA_initialized"] is False
            and type(row["cuda"]["torch_loaded"]) is bool
            and row["limits"]
            == {"role": "worker", "soft_bytes": 8 * 1024**3, "hard_bytes": 8 * 1024**3},
            "All CPU component gates",
        )
        cpu.validate_unit_evidence(row["unit"], plan["unit"])
        c.require(
            row["unit"]["caller"] == child
            and cpu.unit_identity(row["unit"]) == cpu.unit_identity(intent["unit_before"])
            and row["observation"]["MemTotal"] == intent["mem_total"],
            "Same unit inode, exact worker and physical memory",
        )
        formula = cpu.resource_formula(
            row["observation"], plan["unit"]["boot_id"], prospective=True
        )
        c.require(
            all(row[k] == v for k, v in formula.items())
            and row["observation"]["MemAvailable"] >= formula["required_bytes"],
            "Same full reserve+3GiB",
        )
    return {
        "cases": 179,
        "counts": counts,
        "raw_byte_compatible": True,
        "training_ready": counts[V2].get("COMPILED/CFG_COMPILED") == 179,
        "task_quality_attested": False,
    }


def parent(r, c, path, expected, plan, cpu, old, files):
    controller, common = c.module(c.CONTROLLER), old.module(old.COMMON)
    c.require(
        sys.executable == c.PYTHON
        and sys.version_info[:2] == (3, 13)
        and all(os.environ.get(k) == v for k, v in plan["environment"].items())
        and os.environ.get("JEV_CPU3G_BOOT") == plan["unit"]["boot_id"]
        and os.environ.get("JEV_CPU3G_CGROUP") == c.CGROUP,
        "Exact reviewed CPU environment",
    )
    if os.getpgrp() != os.getpid() or os.getsid(0) != os.getpid():
        os.setsid()
    before = cpu.assert_current_unit(plan["unit"])
    props = {
        "MainPID": str(os.getpid()),
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
    got = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            c.UNIT,
            *["--property=" + k for k in [*props, "InvocationID"]],
        ],
        capture_output=True,
        timeout=5,
        check=True,
    )
    c.require(len(got.stdout) < 1024**2 and not got.stderr, "Bounded unit properties")
    observed = dict(line.split("=", 1) for line in got.stdout.decode().splitlines())
    c.require(
        all(observed[k] == v for k, v in props.items()) and len(observed["InvocationID"]) == 32,
        "Exact whole03 unit and new invocation",
    )
    actor = {**old.identity(os.getpid()), "mem_total": cpu.memory_observation()["MemTotal"]}
    c.require(actor["pid"] == actor["pgid"] == actor["sid"], "Parent owned session")
    controller.prospective(cpu, plan["unit"]["boot_id"], actor["mem_total"])
    output = ROOT / plan["output"]
    c.require(not output.exists(), "Fresh training preparation output")
    cpu.configure_limits("parent")
    output.mkdir(mode=0o700)
    nonce = uuid.uuid4().hex
    cpu.save(
        output,
        "intent.json",
        c.wire(
            {
                "parent": {k: actor[k] for k in ("pid", "start_ticks", "pgid", "sid")},
                "mem_total": actor["mem_total"],
                "nonce": nonce,
                "plan_path": str(path),
                "plan_sha256": expected,
                "unit_before": before,
                "properties": observed,
                "sources": files,
            }
        ),
    )
    argv = [
        c.PYTHON,
        "-B",
        "-s",
        str(Path(__file__).absolute()),
        "--run",
        "--plan",
        str(path),
        "--plan-sha256",
        expected,
        "--worker",
        nonce,
    ]
    env = {k: os.environ[k] for k in cpu.SECRET_FREE_ENV if k in os.environ}
    env.update(plan["environment"])
    execution, _ = controller.run_owned(
        cpu, common, output, argv, env, plan["unit"]["boot_id"], actor, wall=600
    )
    primary, review, after = execution["primary_error"], None, None
    try:
        c.require(
            primary is None
            and execution["returncode"] == 0
            and execution["secondary_errors"] == [],
            "Successful owned compiler worker",
        )
        review = validate(c, plan, expected, nonce, cpu, old, files, output, execution["cleanup"])
        monitor_blob = c.read(output / "owned-memory-monitor.jsonl")
        cpu.validate_monitor_evidence(
            monitor_blob,
            plan["unit"]["boot_id"],
            parent=actor,
            cleanup=execution["cleanup"],
            unit=plan["unit"],
        )
        c.require(
            all(
                cpu.unit_identity(json.loads(line)["unit"]) == cpu.unit_identity(before)
                for line in monitor_blob.splitlines()
            ),
            "Parent monitor joins initial unit inode",
        )
        c.authenticate(files, loaded=True)
        controller.prospective(cpu, plan["unit"]["boot_id"], actor["mem_total"])
        after = cpu.assert_current_unit(plan["unit"])
        c.require(cpu.unit_identity(after) == cpu.unit_identity(before), "Final same unit inode")
    except BaseException as exc:
        if primary is None:
            primary = {"type": type(exc).__name__, "message": str(exc)[:2048]}
    result = {
        "schema_id": KIND + "_parent",
        "plan_sha256": expected,
        "nonce": nonce,
        "parent": actor,
        "InvocationID": observed["InvocationID"],
        "execution": execution,
        "review": review,
        "primary_error": primary,
        "unit_after": after,
        "complete_pending_external_unit_exit": primary is None,
        "original95_qualification_attested": False,
        "task_quality_attested": False,
    }
    try:
        cpu.save(output, "parent-terminal.json", c.wire(result), terminal=True)
    except BaseException as secondary:
        if primary is not None:
            raise RuntimeError("Primary retained: " + str(primary)) from secondary
        raise
    c.require(primary is None, "Training preparation failed; partial facts retained")


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--plan")
    parser.add_argument("--plan-sha256")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker")
    args = parser.parse_args(argv)
    if args.plan is None:
        if args.run or args.worker:
            raise ValueError("Explicit selected training preparation plan required")
        print(json.dumps({"schema_id": KIND, "processor_calls": 0, "status": "PLAN_REQUIRED"}))
        return 0
    path = Path(args.plan).absolute()
    r, c, plan, cpu, old, files, technical = load_plan(path, args.plan_sha256)
    if not args.run:
        c.require(args.worker is None, "Dry never authorizes a worker")
        print(
            json.dumps(
                {
                    "schema_id": KIND,
                    "source_files": len(files),
                    "processor_calls": 0,
                    "output_exists": (ROOT / plan["output"]).exists(),
                }
            )
        )
        return 0
    previous = signal.signal(
        signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("CPU deadline"))
    )
    signal.alarm(600 if args.worker else 630)
    try:
        if args.worker:
            worker(r, c, plan, args.plan_sha256, args.worker, cpu, old, files, technical)
        else:
            parent(r, c, path, args.plan_sha256, plan, cpu, old, files)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
