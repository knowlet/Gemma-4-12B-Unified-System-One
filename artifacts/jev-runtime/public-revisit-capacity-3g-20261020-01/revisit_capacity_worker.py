"""Separate3GiB whole-unit68+7 CPU worker; default dry, no model constructor.

The separately owned parent supplies the pinned config, limits its whole tree,
and confirms closure. A worker terminal alone never admits browser execution.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import resource
import signal
import stat
import sys
import time
from pathlib import Path

ROOT = Path(__file__).absolute().parents[3]
HERE = Path(__file__).absolute().parent
KIND = "jev_revisit_cpu_processor_capacity_worker_3g_v1"
PROFILE = "composed-visible-hover-entries-settled-revisit-v1"
GIB, MIB = 1024**3, 1024**2
MAX_FILE, MAX_OUTPUT, RESERVE = 64 * MIB, GIB, MIB
COHORT = "artifacts/jev-runtime/public-revisit-capacity-payloads-20261020-01/prepared-01/payload-cohort.json"
TECHNICAL = "artifacts/jev-runtime/public-v8-capacity-preflight-20261019-02/plan-01.json"
PROCESSOR = "artifacts/jev-runtime/owned-form-progress-collection-admission-20261019-01/CPU-v3-launch-plan-01.json"
Q = "scripts/jev_qualify_public_field_syntax_bf16.py"
V8 = "artifacts/jev-runtime/public-v8-capacity-preflight-20261019-02/v8_capacity_prepare.py"
VALIDATOR = (
    "artifacts/jev-runtime/public-v8-capacity-preflight-20261019-02/v8_capacity_validation.py"
)
CPU = "artifacts/jev-runtime/public-revisit-capacity-3g-20261020-01/cpu_prepare_form_3g.py"
ORIGINAL_CPU = "artifacts/jev-runtime/owned-form-progress-collection-admission-20261019-01/cpu_prepare_form_v3.py"
COMMON = "artifacts/jev-runtime/public-affordance-matrix-harness-20261019-01/matrix_common.py"
PINS = {
    COHORT: ("c39984b1af50d49686a79492f78f618fa37f590aabc423934ee4d8da9d793091", 146149),
    TECHNICAL: ("d6cbc3b14948e2f15fb0e5a98e528c928d3ad3101ed4dae12a6019f20c7058fb", 20983),
    PROCESSOR: ("327e3d2bf6980bceca2d60cfe81ac8b0343bc45149bc1969cb35bc897259b620", 18393),
    Q: ("cbfb7bdd8301174a97ac18389a998ff3256ac32d6f7a770978f2f9b722c3b43b", 204384),
    V8: ("9635abb3ec4ce16c0f05c1a3dc2cebae85e94f993d1db7b568593503a5375893", 24534),
    VALIDATOR: ("1b8c621c77e68fb42e8e00b1c37b29f4847746f6214f020d1a54281f41464b4e", 16724),
    CPU: ("82c7add649276785b64c4fbe24e945e918f555cb2b6bcf96dd61997734bd5334", 67179),
    ORIGINAL_CPU: ("b9b8c7a04464c9e5c41c7a704f6b1a46ae0de7fe1ee80d28adf9f208959bdf55", 57238),
    COMMON: ("71251036c63e8df852ae22814d7234db0daaee90436c265e8a5549de468c6291", 7229),
}
EDGES = (
    "native-click-width400",
    "native-click-width401",
    "native-multihead-retention",
    "policy-near-context",
    "policy-over-context",
    "helper-valid",
    "helper-over-context",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def wire(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def parse(blob):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    return json.loads(blob, object_pairs_hook=pairs, parse_constant=lambda x: require(False, x))


def ordinary(path):
    path = Path(path).absolute()
    require(
        ".." not in path.parts and path.is_relative_to(ROOT), "Repository confined ordinary path"
    )
    require(not any(p.is_symlink() for p in (path, *path.parents)), "No symlink source/output")
    return path


def read(path, maximum=MAX_FILE):
    path = ordinary(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_size <= maximum, "Bounded regular input")
        chunks, remaining = [], before.st_size
        while remaining:
            blob = os.read(fd, min(MIB, remaining))
            require(bool(blob), "Complete bounded read")
            chunks.append(blob)
            remaining -= len(blob)
        after = os.fstat(fd)

        def signature(s):
            return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns

        require(
            signature(before) == signature(after) == signature(path.stat()),
            "Stable input descriptor identity",
        )
        return b"".join(chunks)
    finally:
        os.close(fd)


def checked(ref, base=ROOT):
    require(
        type(ref) is dict and set(ref) == {"path", "sha256", "size_bytes"}, "Typed byte reference"
    )
    require(
        type(ref["path"]) is str
        and type(ref["size_bytes"]) is int
        and ref["size_bytes"] >= 0
        and type(ref["sha256"]) is str
        and re.fullmatch("[0-9a-f]{64}", ref["sha256"]),
        "Typed reference fields",
    )
    path = ordinary(Path(base) / ref["path"])
    require(path.is_relative_to(Path(base).absolute()), "Reference confinement")
    blob = read(path)
    require(
        len(blob) == ref["size_bytes"] and sha(blob) == ref["sha256"], "Selected byte pin differs"
    )
    return blob


def fixed(label):
    digest, size = PINS[label]
    return {"path": label, "sha256": digest, "size_bytes": size}


def whole_caps(output, pending=0, *, terminal=False):
    """Whole owned root observation, not an atomic cross-process disk quota."""
    output = ordinary(output)
    budget_root = output.parent if output.name == "worker" else output
    files = [p for p in budget_root.rglob("*") if p.is_file() or p.is_symlink()]
    require(
        len(files) < 4096
        and all(not p.is_symlink() and p.stat().st_size <= MAX_FILE for p in files),
        "Ordinary bounded output tree",
    )
    require(
        sum(p.stat().st_size for p in files) + pending <= MAX_OUTPUT - (0 if terminal else RESERVE),
        "1GiB output with terminal reserve",
    )


def save(output, label, blob, *, terminal=False):
    require(type(blob) is bytes and len(blob) <= MAX_FILE, "64MiB output file bound")
    output = ordinary(output)
    target = ordinary(output / label)
    require(target.is_relative_to(output), "Output confinement")
    whole_caps(output, len(blob), terminal=terminal)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with target.open("xb") as stream:
        os.chmod(target, 0o600)
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": label, "sha256": sha(blob), "size_bytes": len(blob)}


def identity(pid):
    text = Path(f"/proc/{pid}/stat").read_text()
    parts = text[text.rfind(")") + 2 :].split()
    return {"pid": pid, "start_ticks": int(parts[19]), "pgid": int(parts[2]), "sid": int(parts[3])}


def validate_owner(config, *, lookup=identity, cmdline=None, pid=None, ppid=None):
    pid, ppid = os.getpid() if pid is None else pid, os.getppid() if ppid is None else ppid
    require(config["owner"] == lookup(ppid), "Exact live parent PID/start/PG/SID")
    child = lookup(pid)
    require(child["pgid"] == child["sid"] == pid, "Worker owns fresh PG/SID")
    args = (
        Path(f"/proc/{ppid}/cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        if cmdline is None
        else cmdline
    )
    args = [x.decode() if isinstance(x, bytes) else x for x in args]
    require(
        len(args) == 9
        and args[0] == "/home/albert_hsieh/ComfyUI/.venv/bin/python"
        and args[1:3] == ["-B", "-s"]
        and args[3] == str(HERE / "revisit_capacity_controller.py")
        and args[4:6] == ["--run", "--plan"]
        and Path(args[6]).is_absolute()
        and args[7] == "--plan-sha256"
        and re.fullmatch("[0-9a-f]{64}", args[8]),
        "Exact pinned parent command shape",
    )
    checked(config["parent_source"])
    return child


def load_config(path, expected):
    blob = read(path, 2 * MIB)
    require(
        type(expected) is str and sha(blob) == expected, "Externally selected worker config SHA"
    )
    value = parse(blob)
    require(
        set(value)
        == {
            "schema_id",
            "cohort",
            "technical_plan",
            "processor_plan",
            "worker_source",
            "parent_source",
            "owner",
            "nonce",
            "boot_id",
            "mem_total",
            "output",
            "unit",
        },
        "Strict12 worker config keys",
    )
    require(value["schema_id"] == KIND, "Separate revisit worker kind")
    for key, label in (
        ("cohort", COHORT),
        ("technical_plan", TECHNICAL),
        ("processor_plan", PROCESSOR),
    ):
        require(value[key] == fixed(label), "Original immutable input selection")
    require(
        ordinary(ROOT / value["worker_source"]["path"]) == Path(__file__).absolute(),
        "Exact selected worker origin",
    )
    require(
        ordinary(ROOT / value["parent_source"]["path"]) == HERE / "revisit_capacity_controller.py",
        "Exact new parent source",
    )
    checked(value["worker_source"])
    checked(value["parent_source"])
    require(
        type(value["owner"]) is dict
        and set(value["owner"]) == {"pid", "start_ticks", "pgid", "sid"}
        and all(type(v) is int and v > 0 for v in value["owner"].values()),
        "Typed parent identity",
    )
    require(
        type(value["nonce"]) is str
        and re.fullmatch("[0-9a-f]{32}", value["nonce"])
        and type(value["boot_id"]) is str
        and bool(value["boot_id"])
        and type(value["mem_total"]) is int
        and value["mem_total"] > 0,
        "Typed fresh nonce/boot/total",
    )
    require(
        type(value["output"]) is str and Path(value["output"]).is_absolute(),
        "Absolute owned output",
    )
    ordinary(value["output"])
    require(
        type(value["unit"]) is dict
        and set(value["unit"]) == {"boot_id", "cgroup_path"}
        and value["unit"]["boot_id"] == value["boot_id"]
        and type(value["unit"]["cgroup_path"]) is str
        and value["unit"]["cgroup_path"].startswith("/")
        and ".." not in Path(value["unit"]["cgroup_path"]).parts,
        "Strict sameboot whole-unit identity",
    )
    cohort, technical, processor = (
        parse(checked(value[k])) for k in ("cohort", "technical_plan", "processor_plan")
    )
    require(
        cohort["schema_id"] == "jev_revisit_prospective_ordered_payload_cohort_v1"
        and cohort["profile"] == PROFILE
        and len(cohort["rows"]) == 68
        and cohort["counts"]
        == {
            "genuine_V8_cell_phase_replays": 16,
            "counterfactual_V7_saved_snapshots": 51,
            "technical_repetition": 1,
        },
        "Original68 scoped payload cohort",
    )
    return value, cohort, technical, processor


def source_union(config, cohort, technical):
    execution = parse(checked(technical["technical"]["execution"]))
    result = {}

    def add(path, pin):
        label = ordinary(ROOT / path).relative_to(ROOT).as_posix()
        require(label not in result or result[label] == pin, "Source union conflicting authority")
        result[label] = pin

    for path, pin in execution["files"].items():
        add(path, pin)
    for path, pin in cohort["source_union"].items():
        add(path, pin)
    for label in (Q, V8, VALIDATOR, ORIGINAL_CPU, CPU, COMMON):
        add(label, {k: v for k, v in fixed(label).items() if k != "path"})
    for key in ("worker_source", "parent_source"):
        ref = config[key]
        add(ref["path"], {k: v for k, v in ref.items() if k != "path"})
    return dict(sorted(result.items()))


def authenticate(files):
    for label, pin in files.items():
        checked({"path": label, **pin})


def module(label):
    blob = checked(fixed(label))
    name = "_revisit_cpu_" + Path(label).stem
    spec = importlib.util.spec_from_file_location(name, ROOT / label)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    exec(compile(blob, str(ROOT / label), "exec"), result.__dict__)
    return result


def error_value(exc):
    return {"type": type(exc).__name__, "message": str(exc)[:2048]}


class Gate:
    """New typed journal under a separately enforced whole-unit3GiB cap."""

    def __init__(self, output, config, cpu, *, clock=time.monotonic):
        self.output, self.config, self.cpu, self.clock = output, config, cpu, clock
        self.start, self.sequence, self.bytes = clock(), 0, 0
        self.path = ordinary(output / "resource-gates.jsonl")
        save(output, "resource-gates.jsonl", b"")

    def check(self, phase, scope):
        require(
            scope == "cpu"
            and phase
            in {
                "startup",
                "source-before",
                "runtime-before",
                "cpu-reference-before-processor-load",
                "processor-after",
                "prepare",
                "final-cpu",
            }
            and self.sequence < 2048,
            "Bounded new CPU phase",
        )
        row = {
            "schema_id": KIND + "_resource_gate",
            "sequence": self.sequence,
            "phase": phase,
            "scope": scope,
            "monotonic": self.clock(),
            "admitted": False,
            "error": None,
        }
        row["worker_elapsed_seconds"] = row["monotonic"] - self.start
        primary = None
        try:
            require(row["worker_elapsed_seconds"] < 600, "Worker600s cooperative phase deadline")
            obs = self.cpu.memory_observation()
            require(obs["MemTotal"] == self.config["mem_total"], "Stable physical memory identity")
            row.update(
                observation=obs,
                **self.cpu.resource_formula(obs, self.config["boot_id"], prospective=True),
            )
            row["unit"] = self.cpu.assert_current_unit(self.config["unit"])
            row["limits"], row["cuda"] = self.cpu.verify_limits("worker"), self.cpu.cuda_state()
            require(
                obs["MemAvailable"] >= row["required_bytes"],
                "Prospective3GiB CPU cap plus unchanged host reserve refused",
            )
            row["admitted"] = True
        except BaseException as exc:
            primary, row["error"] = exc, error_value(exc)
        try:
            whole_caps(self.output, 16384)
            self.bytes = self.cpu.append_owned(self.output, "resource-gates.jsonl", row, self.bytes)
            self.sequence += 1
            whole_caps(self.output)
        except BaseException as secondary:
            if primary is not None:
                primary.add_note("Gate persistence also failed: " + repr(secondary))
                raise primary from secondary
            raise
        if primary is not None:
            raise primary
        return row


def validate_gates(blob, config):
    """Close the new journal without pretending it is the old23 reference run."""
    require(type(blob) is bytes and len(blob) <= 32 * MIB, "Bounded new resource journal")
    rows = [parse(line) for line in blob.splitlines()]
    initial = [
        "startup",
        "source-before",
        "runtime-before",
        "cpu-reference-before-processor-load",
        "processor-after",
    ]
    require(
        81 <= len(rows) <= 2048
        and [r["phase"] for r in rows[:5]] == initial
        and rows[-1]["phase"] == "final-cpu"
        and all(r["phase"] == "prepare" for r in rows[5:-1]),
        "Complete new ordered worker gate phases",
    )
    cpu = module(CPU)
    previous, first = -math.inf, None
    for index, row in enumerate(rows):
        cpu.validate_unit_evidence(row["unit"], config["unit"])
        formula = cpu.resource_formula(row["observation"], config["boot_id"], prospective=True)
        monotonic, elapsed = row["monotonic"], row["worker_elapsed_seconds"]
        require(
            row["schema_id"] == KIND + "_resource_gate"
            and row["scope"] == "cpu"
            and type(row["sequence"]) is int
            and row["sequence"] == index
            and row["admitted"] is True
            and row["error"] is None
            and row["observation"]["MemTotal"] == config["mem_total"]
            and row["observation"]["MemAvailable"] >= formula["required_bytes"]
            and all(type(row[k]) is int and row[k] == value for k, value in formula.items()),
            "Exact admitted sameboot worker floor and zero credit",
        )
        require(
            row["limits"] == {"role": "worker", "soft_bytes": 8 * GIB, "hard_bytes": 8 * GIB}
            and all(type(row["limits"][k]) is int for k in ("soft_bytes", "hard_bytes"))
            and set(row["cuda"]) == {"CUDA_VISIBLE_DEVICES", "torch_loaded", "CUDA_initialized"}
            and row["cuda"]["CUDA_VISIBLE_DEVICES"] == ""
            and type(row["cuda"]["torch_loaded"]) is bool
            and row["cuda"]["CUDA_initialized"] is False,
            "Exact AS8 and uninitialized hidden CUDA",
        )
        require(
            type(monotonic) in (int, float)
            and math.isfinite(monotonic)
            and monotonic >= previous
            and type(elapsed) in (int, float)
            and math.isfinite(elapsed)
            and 0 <= elapsed < 600,
            "Finite ordered cooperative worker600s evidence",
        )
        first = monotonic - elapsed if first is None else first
        require(
            math.isclose(monotonic - elapsed, first, rel_tol=0, abs_tol=1e-6),
            "Same worker clock origin",
        )
        previous = monotonic
    return {
        "sha256": sha(blob),
        "rows": len(rows),
        "boot_id": config["boot_id"],
        "mem_total": config["mem_total"],
        "last_worker_elapsed_seconds": rows[-1]["worker_elapsed_seconds"],
    }


def slots(cohort):
    ids = [row["case_id"] for row in cohort["rows"]] + list(EDGES)
    require(len(ids) == len(set(ids)) == 75, "Complete75 unique ordered rows")
    return [
        {
            "ordinal": n,
            "id": label,
            "kind": "helper" if n >= 73 else "policy",
            "outcome": "NOT_EXECUTED",
            "prefix": None,
            "error": None,
            "expectation_met": False,
            "native_forward_calls": 0,
        }
        for n, label in enumerate(ids)
    ]


def prepare_row(
    row,
    payload,
    provenance,
    *,
    q,
    validator,
    model,
    gate,
    output,
    request,
    schema_error,
    prepared=None,
):
    before = wire(payload)
    row["provenance"] = provenance
    row["payload"] = save(output, f"payloads/{row['ordinal']:02d}.json", before)
    stage = "schema"
    try:
        gate.check("prepare", "cpu")
        if row["kind"] == "policy":
            encoded, source, labels = request(payload, "generated-actions-none")
            require(
                encoded.encode() == source == before, "Complete ordered native request unchanged"
            )
            row["offered_head_order"], row["offered_candidate_keys"] = list(labels), labels
        else:
            q.parse_helper_request(payload, system_prompt=q.TEXT_VALUE)
        stage = "processor"
        prefix, prompt = (
            prepared
            if prepared is not None
            else q._prefix(model, payload, row["kind"], include_prompt=True)
        )
        gate.check("prepare", "cpu")
        validator.validate_prefix(prefix, prompt)
        require(wire(payload) == before, "Parser/processor must not mutate complete payload")
        row.update(
            prefix=prefix,
            prompt=save(output, f"prompts/{row['ordinal']:02d}.txt", prompt.encode()),
            outcome=validator.outcome(prefix, row["kind"]),
            error=None,
        )
    except Exception as exc:
        row.update(
            outcome="REFUSED_SCHEMA"
            if stage == "schema" and isinstance(exc, schema_error)
            else "UNKNOWN",
            error=error_value(exc),
        )
    acceptable = (
        {"ACCEPTED", "REFUSED_SCHEMA", "REFUSED_CONTEXT"}
        if row["ordinal"] < 68
        else validator.expected_outcomes(row["id"])
    )
    row["expectation_met"] = row["outcome"] in acceptable
    save(output, f"rows/{row['ordinal']:02d}.json", canonical(row))
    return row


def run_rows(
    cohort, *, payload_root, q, validator, v8, model, gate, output, request, schema_error, rows=None
):
    rows = slots(cohort) if rows is None else rows
    common = dict(
        q=q,
        validator=validator,
        model=model,
        gate=gate,
        output=output,
        request=request,
        schema_error=schema_error,
    )
    for row, selection in zip(rows[:68], cohort["rows"], strict=True):
        gate.check("prepare", "cpu")
        payload = parse(checked(selection["payload_ref"], payload_root))
        prepare_row(
            row, payload, {"scope": selection["scope"], "cohort_selection": selection}, **common
        )
    for ordinal, index in ((68, 32), (69, 32), (70, 33), (73, 36)):
        payload = q.fixture_payload(index)
        if ordinal == 69:
            payload["questions"]["click_target"]["criteria"]["401"] = "Observed control"
        prepare_row(
            rows[ordinal],
            payload,
            {"scope": "SEPARATE_ORIGINAL_TECHNICAL_RECIPE_NOT_GENUINE_QUALITY"},
            **common,
        )
    for helper, targets in ((False, ((71, 8128), (72, 8161))), (True, ((74, 8065),))):
        selected = v8.scan(q, model, gate, output, helper=helper)
        for ordinal, band in targets:
            require(band in selected, "Original bounded context band unreachable")
            payload, prefix_prompt = selected[band]
            prepare_row(
                rows[ordinal],
                payload,
                {"scope": "SEPARATE_ORIGINAL_TECHNICAL_LITERAL_GRID", "band_low": band},
                prepared=prefix_prompt,
                **common,
            )
    return rows


def worker(config_path, expected, *, run=False):
    config, cohort, technical, processor = load_config(config_path, expected)
    files = source_union(config, cohort, technical)
    if not run:
        return {
            "schema_id": KIND,
            "status": "DRY_SOURCE_METADATA_ONLY",
            "rows": 75,
            "new_source_union_files": len(files),
            "processor_calls": 0,
            "output_created": False,
        }
    require(
        not any(
            n.split(".")[0] in {"torch", "transformers", "tokenizers", "peft", "s1"}
            for n in sys.modules
        ),
        "Fresh worker before frameworks",
    )
    child = validate_owner(config)
    cpu = module(CPU)
    cpu.configure_limits("worker")
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_FILE, MAX_FILE))
    output = ordinary(config["output"]) / "worker"
    require(not output.exists(), "Fresh worker output subtree")
    output.mkdir(mode=0o700)
    rows, primary, secondary, gate = slots(cohort), None, [], None
    old_handler = signal.getsignal(signal.SIGALRM)

    def deadline(_number, _frame):
        raise TimeoutError("Worker600s alarm; parent also owns hard deadline")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(600)
    try:
        gate = Gate(output, config, cpu)
        gate.check("startup", "cpu")
        gate.check("source-before", "cpu")
        authenticate(files)
        save(
            output,
            "source-union.json",
            canonical(
                {
                    "schema_id": KIND + "_source_union",
                    "files": files,
                    "count": len(files),
                    "old95_modified": False,
                }
            ),
        )
        q, validator, v8 = module(Q), module(VALIDATOR), module(V8)
        q._configure_environment("prepare")
        sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
        base = technical["technical"]["config"]
        gate.check("runtime-before", "cpu")
        sources = q._sources(base)
        runtime = q._runtime(base)
        whole_caps(output, sum(pin["size_bytes"] for pin in processor["reference_items"].values()))
        refs = cpu.capture_reference(processor, output)
        whole_caps(output)
        model = cpu.load_processor(refs, gate)
        gate.check("processor-after", "cpu")
        save(
            output, "processor-evidence-before.json", canonical(cpu.processor_evidence(model, refs))
        )
        from s1.errors import RequestValidationError
        from s1.jev_generated import _request

        run_rows(
            cohort,
            payload_root=(ROOT / COHORT).parent,
            q=q,
            validator=validator,
            v8=v8,
            model=model,
            gate=gate,
            output=output,
            request=_request,
            schema_error=RequestValidationError,
            rows=rows,
        )
        sources.check()
        authenticate(files)
        q._runtime(base)
        save(
            output, "processor-evidence-after.json", canonical(cpu.processor_evidence(model, refs))
        )
        checked(config["worker_source"])
        checked(config["parent_source"])
        gate.check("final-cpu", "cpu")
        save(
            output,
            "runtime-component.json",
            canonical(
                {
                    "runtime": technical["technical"]["runtime"],
                    "runtime_files": len(runtime["files"]),
                    "original_execution": technical["technical"]["execution"],
                    "original_execution_files": len(sources.manifest()["files"]),
                    "processor_plan": config["processor_plan"],
                    "processor_reference_items": len(refs),
                    "loaded_mandatory_generation_imports_requested": False,
                }
            ),
        )
    except BaseException as exc:
        primary = exc
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        for row in rows:
            if row["outcome"] == "NOT_EXECUTED":
                row.update(
                    outcome="UNKNOWN",
                    error={"type": "NotExecuted", "message": "Worker interrupted before this row"},
                )
        complete = primary is None and all(r["expectation_met"] for r in rows)
        terminal = {
            "schema_id": KIND,
            "profile": PROFILE,
            "config_sha256": expected,
            "nonce": config["nonce"],
            "child": child,
            "status": "CPU_PREFIXES_COMPLETE_AWAIT_WHOLE_OWNED_CLOSURE"
            if complete
            else "FAILED_OR_UNKNOWN",
            "complete": complete,
            "primary_error": None if primary is None else error_value(primary),
            "secondary_errors": secondary,
            "rows": rows,
            "counts": {
                name: sum(r["outcome"] == name for r in rows)
                for name in ("ACCEPTED", "REFUSED_SCHEMA", "REFUSED_CONTEXT", "UNKNOWN")
            },
            "processor_capacity_only": True,
            "task_quality_attested": False,
            "universal_profile_capacity_attested": False,
            "native_model_constructor_forward_HTTP_browser_calls": 0,
            "external_parent_exit0_and_owned_closure_required": True,
        }
        try:
            save(output, "terminal.json", canonical(terminal), terminal=True)
        except BaseException as exc:
            if primary is None:
                primary = exc
            else:
                primary.add_note("Worker terminal persistence also failed: " + repr(exc))
    if primary is not None:
        raise primary
    return 0 if complete else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    result = worker(args.config, args.config_sha256, run=args.run)
    if type(result) is dict:
        print(json.dumps(result))
        return 0
    return result


if __name__ == "__main__":
    raise SystemExit(main())
