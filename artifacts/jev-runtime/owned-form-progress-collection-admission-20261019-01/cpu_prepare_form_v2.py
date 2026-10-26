"""Default-dry independent CPU_REFERENCE_ONLY v2 owned form reference worker.

No browser, native toolchain, base model, forward, HTTP or Modal interface.
Actual execution requires an externally selected plan and explicit --run.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import os
import signal
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GIB, MIB = 1024**3, 1024**2
MAX_BYTES, MAX_FILE, RESERVE = GIB, 64 * MIB, MIB
SCHEMA = "owned-form-progress-CPU-reference-only-launch-v2"
SOURCE_SCHEMA = "owned-form-progress-CPU-reference-only-source-union-v2"
SUCCESS = "COMPLETE_CPU_REFERENCE_ONLY_V2_PENDING_EXTERNAL_CONTROLLER_EXIT"
SECRET_FREE_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR")


PROFILE = "CPU_REFERENCE_ONLY"
PARENT_AS, CHILD_AS, TREE_RSS = 2 * GIB, 16 * GIB, 18 * GIB
GATE_SCHEMA = "owned-form-CPU-reference-resource-journal-v2"
MONITOR_SCHEMA = "owned-form-CPU-reference-owned-memory-monitor-v2"
PARENT_PHASES = (
    "cpu-reference-parent-before-source",
    "cpu-reference-parent-before-child",
    "cpu-reference-parent-after-child",
)
ROW_PHASES = tuple(
    "cpu-reference-compile-" + split + "-" + str(index)
    for split, count in (("train", 13), ("development", 10))
    for index in range(count)
)
WORKER_PHASES = (
    "cpu-reference-worker-before-source",
    # Frozen admission performs exactly this original CPU-only boundary once.
    # It is not a remapping of the original train-scope compile adapter.
    "before-any-project-framework-or-parent-payload",
    "cpu-reference-before-reference-capture",
    "cpu-reference-before-processor-load",
    "cpu-reference-after-processor-load",
    *ROW_PHASES,
    "cpu-reference-after-all-compile",
    "cpu-reference-before-publication",
)
RUNTIME_GUARDS = (
    "after-processor-load",
    *("after-row-" + str(index) for index in range(23)),
    "after-all-compile",
    "before-worker-terminal",
)
LIMITS = {
    "profile": PROFILE,
    "parent_RLIMIT_AS_soft": PARENT_AS,
    "parent_RLIMIT_AS_hard": CHILD_AS,
    "child_RLIMIT_AS_soft": CHILD_AS,
    "child_RLIMIT_AS_hard": CHILD_AS,
    "tracked_parent_plus_owned_RSS_bytes": TREE_RSS,
    "prospective_extra_bytes": TREE_RSS,
    "monitor_interval_seconds": 0.05,
    "CPU_threads": 2,
    "swap_or_resident_credit_bytes": 0,
}
EXPECTED_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "HF_HUB_CACHE": str(ROOT / "artifacts/jev-local-hf/hub"),
    "PYTHONPATH": ":".join(
        str(ROOT / label) for label in ("artifacts/jev-runtime/overlay", "src", "scripts")
    ),
    "CUDA_VISIBLE_DEVICES": "",
}

LIMITATIONS = (
    "RLIMIT_AS is per-process address space, not RSS/cgroup or whole-tree reservation. "
    "50ms tracked PID/start/PG/SID RSS sampling can miss short-lived descendants and "
    "between-sample aggregate spikes; shared RSS pages may be counted repeatedly. "
    "Trusted processor/kernel source creates no model/multiprocessing calls. Other "
    "workloads may change host availability between observations; never stop them. "
    "ENOMEM/import/stack failure is preserved as failed/unknown partial evidence. "
    "No measured capacity, training, task-quality or qualification follows from source."
)


def limit_target(role):
    require(role in {"parent", "worker"}, "Exact CPU-reference limit role")
    return (PARENT_AS if role == "parent" else CHILD_AS, CHILD_AS)


def limit_module(api=None):
    if api is None:
        import resource

        return resource
    return api


def verify_limits(role, *, api=None):
    api = limit_module(api)
    actual = api.getrlimit(api.RLIMIT_AS)
    require(
        type(actual) is tuple
        and len(actual) == 2
        and all(type(x) is int for x in actual)
        and actual == limit_target(role),
        "Exact CPU-reference RLIMIT_AS readback",
    )
    return {"role": role, "soft_bytes": actual[0], "hard_bytes": actual[1]}


def configure_limits(role, *, api=None):
    """Called only by --run roles, never at import/dry/source preparation."""
    require(sys.platform == "linux" and sys.maxsize > 2**32, "64-bit Linux RLIMIT_AS")
    api = limit_module(api)
    before = api.getrlimit(api.RLIMIT_AS)
    require(
        type(before) is tuple
        and len(before) == 2
        and all(type(x) is int for x in before)
        and (before[1] == api.RLIM_INFINITY or before[1] >= CHILD_AS),
        "Inherited hard AS limit cannot be raised or waived",
    )
    api.setrlimit(api.RLIMIT_AS, limit_target(role))
    return {"before": list(before), **verify_limits(role, api=api)}


def cuda_state():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-reference CUDA hidden")
    torch = sys.modules.get("torch")
    initialized = (
        False if torch is None else getattr(getattr(torch, "cuda", None), "_initialized", None)
    )
    require(initialized is False, "CPU-reference CUDA must remain uninitialized")
    return {
        "CUDA_VISIBLE_DEVICES": "",
        "torch_loaded": torch is not None,
        "CUDA_initialized": False,
    }


def runtime_guard(phase):
    require(phase in RUNTIME_GUARDS, "Exact CPU-reference runtime guard")
    return {"phase": phase, "limits": verify_limits("worker"), **cuda_state()}


def memory_observation():
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        if name in {"MemTotal", "MemAvailable"}:
            number, unit = value.split()
            require(unit == "kB", "CPU-reference memory units")
            memory[name] = int(number) * 1024
    memory["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    return memory


def resource_formula(observation, boot, *, prospective):
    require(
        type(observation) is dict
        and set(observation) == {"MemTotal", "MemAvailable", "boot_id"}
        and all(type(observation[k]) is int for k in ("MemTotal", "MemAvailable"))
        and type(boot) is str
        and bool(boot)
        and observation["boot_id"] == boot
        and 0 < observation["MemAvailable"] <= observation["MemTotal"],
        "Fresh sameboot strict CPU-reference observation",
    )
    reserve = max(24 * GIB, (observation["MemTotal"] + 4) // 5)
    original_cpu_floor = reserve + 4 * GIB
    required = reserve + TREE_RSS if prospective else reserve
    return {
        "reserve_bytes": reserve,
        "required_bytes": required,
        "original_CPU_floor_bytes": original_cpu_floor,
        "swap_or_resident_credit_bytes": 0,
    }


def append_owned(output, label, record, expected_bytes):
    blob = encode(record) + b"\n"
    require(len(blob) <= 16384, "CPU-reference journal row cap")
    caps(output, len(blob))
    path = ordinary(output / label)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
    try:
        require(os.fstat(fd).st_size == expected_bytes, "Owned CPU-reference journal changed")
        require(os.write(fd, blob) == len(blob), "Complete CPU-reference journal append")
        os.fsync(fd)
    finally:
        os.close(fd)
    return expected_bytes + len(blob)


def phase_sequence(role):
    require(role in {"parent", "worker"}, "Exact CPU-reference journal role")
    return PARENT_PHASES if role == "parent" else WORKER_PHASES


class CPUReferenceGate:
    """Separate explicit CPU policy, never accepts train/load/forward phases."""

    def __init__(self, output, label, role, boot, *, observe=None, limits=None, cuda=None):
        self.output, self.label, self.role, self.boot = output, label, role, boot
        self.observe = memory_observation if observe is None else observe
        self.limits = verify_limits if limits is None else limits
        self.cuda = cuda_state if cuda is None else cuda
        self.sequence, self.bytes = 0, 0
        self.phases = phase_sequence(role)
        require(type(boot) is str and bool(boot), "Fresh sameboot CPU-reference journal")
        save(output, label, b"")

    def check(self, phase, scope):
        require(
            scope == "cpu"
            and self.sequence < len(self.phases)
            and phase == self.phases[self.sequence],
            "Exact ordered CPU-reference phase/scope",
        )
        record = {
            "schema_id": GATE_SCHEMA,
            "profile": PROFILE,
            "role": self.role,
            "sequence": self.sequence,
            "phase": phase,
            "scope": "cpu",
            "monotonic": time.monotonic(),
            "expected_boot_id": self.boot,
            "admitted": False,
            "error": None,
        }
        primary = None
        try:
            observation = self.observe()
            record.update(
                observation=observation,
                **resource_formula(observation, self.boot, prospective=True),
            )
            record["limits"] = self.limits(self.role)
            record["cuda"] = self.cuda()
            require(
                observation["MemAvailable"] >= record["required_bytes"],
                "CPU-reference prospective parent+child memory refused",
            )
            record["admitted"] = True
        except BaseException as error:
            primary = error
            record["error"] = error_value(error, "CPU-reference-gate")
        try:
            self.bytes = append_owned(self.output, self.label, record, self.bytes)
            self.sequence += 1
        except BaseException as secondary:
            if primary is not None:
                primary.add_note("Resource refusal persistence failed: " + repr(secondary))
                raise primary from secondary
            raise
        if primary is not None:
            raise primary
        return record


def validate_gate_evidence(blob, role):
    rows = [parse(line) for line in blob.splitlines()]
    require(len(rows) == len(phase_sequence(role)), "Complete CPU-reference gate denominator")
    previous, boot = -1, None
    for index, row in enumerate(rows):
        boot = row["expected_boot_id"] if boot is None else boot
        formula = resource_formula(row["observation"], boot, prospective=True)
        require(
            row["schema_id"] == GATE_SCHEMA
            and row["profile"] == PROFILE
            and row["role"] == role
            and row["sequence"] == index
            and type(row["sequence"]) is int
            and row["phase"] == phase_sequence(role)[index]
            and row["scope"] == "cpu"
            and row["admitted"] is True
            and row["error"] is None
            and type(row["monotonic"]) in (int, float)
            and previous < row["monotonic"]
            and row["expected_boot_id"] == boot
            and all(type(row[k]) is int and row[k] == v for k, v in formula.items())
            and row["observation"]["MemAvailable"] >= formula["required_bytes"]
            and row["limits"]
            == {"role": role, "soft_bytes": limit_target(role)[0], "hard_bytes": CHILD_AS}
            and row["cuda"]["CUDA_VISIBLE_DEVICES"] == ""
            and row["cuda"]["CUDA_initialized"] is False,
            "Authenticated sameboot complete CPU-reference gate evidence",
        )
        previous = row["monotonic"]
    return boot


def process_rss(identity, proc=Path("/proc")):
    require(
        type(identity) is dict
        and all(
            type(identity[k]) is int and identity[k] > 0
            for k in ("pid", "start_ticks", "pgid", "sid")
        ),
        "Strict tracked PID/start/PG/SID",
    )
    directory = proc / str(identity["pid"])

    def stat_identity():
        value = (directory / "stat").read_text()
        tail = value[value.rfind(")") + 2 :].split()
        return {
            "pid": int(value.split(" ", 1)[0]),
            "start_ticks": int(tail[19]),
            "pgid": int(tail[2]),
            "sid": int(tail[3]),
        }

    expected = {k: identity[k] for k in ("pid", "start_ticks", "pgid", "sid")}
    require(stat_identity() == expected, "Tracked RSS identity changed before read")
    values = {}
    for line in (directory / "status").read_text().splitlines():
        name, _, value = line.partition(":")
        if name == "VmRSS":
            number, unit = value.split()
            require(unit == "kB" and int(number) >= 0, "Tracked RSS units/value")
            values[name] = int(number) * 1024
    require(
        "VmRSS" in values and stat_identity() == expected, "Unknown/reused tracked RSS identity"
    )
    return {**expected, "rss_bytes": values["VmRSS"]}


class OwnedMemoryMonitor:
    def __init__(self, output, boot, parent, *, observe=None, rss=None):
        self.output, self.boot, self.parent = output, boot, parent
        self.observe = memory_observation if observe is None else observe
        self.rss = process_rss if rss is None else rss
        self.sequence, self.bytes = 0, 0
        save(output, "owned-memory-monitor.jsonl", b"")

    def check(self, rows):
        record = {
            "schema_id": MONITOR_SCHEMA,
            "profile": PROFILE,
            "sequence": self.sequence,
            "monotonic": time.monotonic(),
            "expected_boot_id": self.boot,
            "admitted": False,
            "error": None,
        }
        primary = None
        try:
            require(self.parent["pid"] not in rows, "Controller is not in owned child tree")
            identities = [self.parent, *(rows[k] for k in sorted(rows))]
            record["tracked"] = [self.rss(identity) for identity in identities]
            record["tracked_RSS_bytes"] = sum(r["rss_bytes"] for r in record["tracked"])
            observation = self.observe()
            record.update(
                observation=observation,
                **resource_formula(observation, self.boot, prospective=False),
            )
            record["parent_limits"] = verify_limits("parent")
            require(
                observation["MemAvailable"] >= record["reserve_bytes"]
                and record["tracked_RSS_bytes"] <= TREE_RSS,
                "CPU-reference host reserve or tracked18GiB RSS refused",
            )
            record["admitted"] = True
        except BaseException as error:
            primary = error
            record["error"] = error_value(error, "owned-memory-monitor")
        try:
            self.bytes = append_owned(self.output, "owned-memory-monitor.jsonl", record, self.bytes)
            self.sequence += 1
        except BaseException as secondary:
            if primary is not None:
                primary.add_note("Memory refusal persistence failed: " + repr(secondary))
                raise primary from secondary
            raise
        if primary is not None:
            raise primary
        return record


def validate_monitor_evidence(blob, boot, *, parent, cleanup):
    rows = [parse(line) for line in blob.splitlines()]
    require(0 < len(rows) <= 12010, "Bounded complete CPU-reference monitor")
    require(
        type(parent) is dict
        and type(cleanup) is dict
        and cleanup["confirmed"] is True
        and cleanup["remaining"] == []
        and type(cleanup["tracked_PID_start_ticks"]) is dict
        and type(cleanup["tracked_groups"]) is list
        and all(
            type(k) is str and k.isdigit() and type(v) is int and v > 0
            for k, v in cleanup["tracked_PID_start_ticks"].items()
        )
        and all(type(v) is int and v > 0 for v in cleanup["tracked_groups"]),
        "Closed exact owned memory-monitor identities",
    )
    parent_fields = {k: parent[k] for k in ("pid", "start_ticks", "pgid", "sid")}
    require(
        all(type(v) is int and v > 0 for v in parent_fields.values())
        and str(parent["pid"]) not in cleanup["tracked_PID_start_ticks"],
        "Controller identity must remain outside child cleanup authority",
    )
    previous = -1
    for index, row in enumerate(rows):
        formula = resource_formula(row["observation"], boot, prospective=False)
        identities = [(r["pid"], r["start_ticks"]) for r in row["tracked"]]
        require(
            bool(row["tracked"])
            and {k: row["tracked"][0][k] for k in parent_fields} == parent_fields
            and all(
                cleanup["tracked_PID_start_ticks"].get(str(r["pid"])) == r["start_ticks"]
                and r["pgid"] in cleanup["tracked_groups"]
                for r in row["tracked"][1:]
            ),
            "Memory samples join exact parent and owned closed PID/start/groups",
        )
        require(
            row["schema_id"] == MONITOR_SCHEMA
            and row["profile"] == PROFILE
            and row["sequence"] == index
            and type(row["sequence"]) is int
            and row["expected_boot_id"] == boot
            and row["admitted"] is True
            and row["error"] is None
            and type(row["monotonic"]) in (int, float)
            and previous < row["monotonic"]
            and bool(identities)
            and len(set(identities)) == len(identities)
            and all(
                type(r[k]) is int and r[k] > 0
                for r in row["tracked"]
                for k in ("pid", "start_ticks", "pgid", "sid")
            )
            and all(type(r["rss_bytes"]) is int and r["rss_bytes"] >= 0 for r in row["tracked"])
            and row["tracked_RSS_bytes"] == sum(r["rss_bytes"] for r in row["tracked"])
            and row["tracked_RSS_bytes"] <= TREE_RSS
            and all(type(row[k]) is int and row[k] == v for k, v in formula.items())
            and row["observation"]["MemAvailable"] >= formula["reserve_bytes"]
            and row["parent_limits"]
            == {"role": "parent", "soft_bytes": PARENT_AS, "hard_bytes": CHILD_AS},
            "Complete sameboot CPU-reference tracked-tree memory evidence",
        )
        previous = row["monotonic"]
    return len(rows)


def compile_references_v2(forms, model, collection, reference_items, gate):
    """New CPU-only adapter. Original kernel/CFG implementations are unchanged."""
    require(type(collection) is forms.AdmittedFormProgressCollection, "Exact new form collection")
    collection.check()
    plan = forms.common.parse(collection._plan)
    forms.authenticate_sources(plan, plan["sources"])
    forms.common.verify_reference_environment(reference_items, model=model)
    require(
        type(model) is SimpleNamespace
        and set(vars(model)) == {"processor", "tok", "max_context", "lm"}
        and type(model.lm) is SimpleNamespace
        and set(vars(model.lm)) == {"config", "generation_config"}
        and type(model.lm.config) is SimpleNamespace
        and type(model.lm.generation_config) is SimpleNamespace
        and set(vars(model.lm.generation_config)) == {"eos_token_id"}
        and model.lm.generation_config.eos_token_id == [1, 106, 50]
        and model.max_context == 8192,
        "Processor-only metadata namespace; no LM object or constructor",
    )
    kernel = forms.common.trusted_module(
        "jev_browser_fit_actions", "scripts/jev_browser_fit_actions.py"
    )
    cfg = forms.common.trusted_module("jev_cfg_action_loss", "scripts/jev_cfg_action_loss.py")
    rows, plans, guards = {}, {}, []
    for split in forms.COUNTS:
        cases = collection.train_cases if split == "train" else collection.development_cases
        wires = collection.compile_wire_projection(split)
        for index, (case, wire) in enumerate(zip(cases, wires, strict=True)):
            gate.check("cpu-reference-compile-" + split + "-" + str(index), "cpu")
            actual = kernel.compile_actions(
                model,
                [case],
                {"blobs": {"wire_requests.jsonl": forms.common.encode(wire) + b"\n"}},
                identity=forms.common.IDENTITY,
            )
            require(
                len(actual) == 1
                and actual[0]["record"]["original_wire_sha256"] == case.wire_sha256,
                "Full original ordered form request/processor reference join",
            )
            row = actual[0]
            rows[case.id] = row
            plans[case.id] = cfg.compile_action_plan(row, wire["payload"], model.tok)
            guards.append(runtime_guard("after-row-" + str(len(guards))))
    require(
        len(rows) == len(plans) == 23
        and list(rows) == [c.id for c in collection.train_cases + collection.development_cases],
        "All23 full targets/masks/prefix/EOS/request-bound CFG plans; no selection",
    )
    return rows, plans, guards


def require(value, message):
    if not value:
        raise ValueError(message)


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def parse(blob):
    def pairs(values):
        out = {}
        for key, value in values:
            require(key not in out, "Duplicate CPU preparation JSON key")
            out[key] = value
        return out

    return json.loads(
        blob, object_pairs_hook=pairs, parse_constant=lambda _: require(False, "Nonfinite JSON")
    )


def ordinary(path):
    path = Path(path).absolute()
    require(
        ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "Ordinary CPU preparation path",
    )
    return path


def read(path, maximum=MAX_FILE):
    path = ordinary(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(before.st_mode) and before.st_size <= maximum,
            "Bounded ordinary CPU evidence",
        )
        blob = stream.read(before.st_size + 1)
        after = os.fstat(stream.fileno())
    require(
        len(blob) == before.st_size
        and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
        "CPU evidence changed while read",
    )
    return blob


def checked(ref):
    require(
        type(ref) is dict and set(ref) == {"path", "sha256", "size_bytes"},
        "Typed CPU input reference",
    )
    p = Path(ref["path"])
    require(not p.is_absolute() and ".." not in p.parts, "Confined CPU input reference")
    blob = read(ROOT / p)
    require(
        len(blob) == ref["size_bytes"] and hashlib.sha256(blob).hexdigest() == ref["sha256"],
        "CPU input/source drift: " + ref["path"],
    )
    return blob


def caps(output, pending=0, *, terminal=False):
    files = [p for p in output.rglob("*") if p.is_file()]
    require(
        len(files) < 10000
        and all(not p.is_symlink() and p.stat().st_size <= MAX_FILE for p in files),
        "CPU file count/per-file cap",
    )
    require(
        sum(p.stat().st_size for p in files) + pending <= MAX_BYTES - (0 if terminal else RESERVE),
        "CPU total evidence/terminal reserve",
    )


def save(output, label, blob, *, terminal=False):
    require(type(blob) is bytes and len(blob) <= MAX_FILE, "CPU output file cap")
    path = ordinary(output / label)
    require(path.is_relative_to(output), "Confined CPU output")
    caps(output, len(blob), terminal=terminal)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        require(stream.write(blob) == len(blob), "Complete CPU evidence write")
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {"path": label, "sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)}


def error_value(error, phase):
    return {"type": type(error).__name__, "message": str(error)[:2048], "phase": phase}


def terminal(output, label, value, secondary):
    try:
        save(output, label + ".pending", encode(value) + b"\n", terminal=True)
        require(not (output / label).exists(), "Fresh CPU terminal")
        (output / (label + ".pending")).rename(output / label)
        directory = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return True
    except BaseException as error:
        secondary.append(error_value(error, "terminal-publication"))
        return False


def report_unknown(status, primary, secondary):
    try:
        print(
            encode(
                {"status": status, "primary_error": primary, "secondary_errors": secondary}
            ).decode(),
            file=sys.stderr,
        )
    except BaseException:
        pass


def load_plan(path, expected):
    blob = read(path, 2 * MIB)
    require(hashlib.sha256(blob).hexdigest() == expected, "Externally selected CPU launch plan")
    plan = parse(blob)
    require(
        type(plan) is dict
        and set(plan)
        == {
            "schema_id",
            "source_schema_id",
            "profile",
            "limits",
            "parent_phases",
            "worker_phases",
            "python",
            "caps",
            "ordered_cases",
            "source_handoff",
            "admission_plan",
            "environment",
            "original_compiler_receipt",
            "reference_items",
            "reference_root",
            "form_source_HOLD",
            "previous_runner",
            "limitations",
        },
        "Strict CPU-reference-only launch plan keys",
    )
    require(
        plan["schema_id"] == SCHEMA and plan["source_schema_id"] == SOURCE_SCHEMA,
        "Separate CPU-reference-only v2 execution kind",
    )
    require(
        plan["python"] == "/home/albert_hsieh/ComfyUI/.venv/bin/python"
        and plan["caps"]
        == {
            "seconds": 600,
            "bytes": MAX_BYTES,
            "file_bytes": MAX_FILE,
            "terminal_reserve_bytes": RESERVE,
            "stdout_or_stderr_bytes": MIB,
            "TERM_KILL_grace_each_seconds": 2,
        },
        "Fixed CPU resources/watchdog",
    )
    require(
        plan["profile"] == PROFILE
        and plan["limits"] == LIMITS
        and plan["parent_phases"] == list(PARENT_PHASES)
        and plan["worker_phases"] == list(WORKER_PHASES),
        "Exact independent CPU-reference-only profile",
    )
    require(
        len(plan["ordered_cases"]) == 23
        and len({r["id"] for r in plan["ordered_cases"]}) == 23
        and [r["split"] for r in plan["ordered_cases"]].count("train") == 13
        and [r["split"] for r in plan["ordered_cases"]].count("development") == 10,
        "Fixed complete23 ordered references",
    )
    require(
        plan["form_source_HOLD"]["path"] == str(HERE.relative_to(ROOT) / "source-HOLD-03/HOLD.json")
        and plan["form_source_HOLD"]["sha256"]
        == "f03f6e1b23c0a488325b2457050ce877f2380a697523dcdc9549c2f4b7421d77",
        "Frozen form admission HOLD03",
    )
    checked(plan["form_source_HOLD"])
    require(
        plan["previous_runner"]["sha256"]
        == "11f8cd050afd65c2fc2a6ecb47425a1b1eb91d6840d6ff56c08ba6dedc361c2a",
        "Immutable v1 predecessor",
    )
    checked(plan["previous_runner"])
    handoff = parse(checked(plan["source_handoff"]))
    admission = parse(checked(plan["admission_plan"]))
    require(
        handoff["schema_id"] == SOURCE_SCHEMA
        and len(handoff["files"]) == 105
        and set(handoff["files"])
        == set(admission["sources"]) | {Path(__file__).resolve().relative_to(ROOT).as_posix()},
        "Exact105=original104 plus this CPU driver",
    )
    require(
        all(handoff["files"][k] == v for k, v in admission["sources"].items()),
        "Original104 projection exact",
    )
    driver_label = Path(__file__).resolve().relative_to(ROOT).as_posix()
    checked({"path": driver_label, **handoff["files"][driver_label]})
    require(
        plan["environment"]["CUDA_VISIBLE_DEVICES"] == ""
        and plan["environment"]["HF_HUB_OFFLINE"]
        == plan["environment"]["TRANSFORMERS_OFFLINE"]
        == "1",
        "CPU offline/CUDA hidden",
    )
    require(
        plan["environment"] == EXPECTED_ENVIRONMENT,
        "Exact CPU-reference-only offline/origin/two-thread environment",
    )
    require(
        plan["admission_plan"]["path"] == str(HERE.relative_to(ROOT) / "admission-plan-02.json")
        and plan["admission_plan"]["sha256"]
        == "01518a2859b4483631d19c2592498a3dac29ba4221dc6f908ea6f9c19611cc34"
        and plan["previous_runner"]["path"] == str(HERE.relative_to(ROOT) / "cpu_prepare_form.py")
        and plan["reference_root"]
        == "artifacts/jev-runtime/public-native-action-compiler-07-root-20261016-01/actual-compile-01",
        "Exact immutable admission02/predecessor/original processor namespace",
    )
    return plan, handoff["files"]


def authenticate_sources(files):
    for label, pin in files.items():
        checked({"path": label, **pin})


def module(name, label, files):
    path = ordinary(ROOT / label)
    checked({"path": label, **files[label]})
    if name in sys.modules:
        loaded = sys.modules[name]
        require(Path(loaded.__file__).resolve() == path, "CPU module origin: " + name)
        return loaded
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


# v1 BoundedGate/ResourceGate delegation is deliberately absent in this successor.


def resource_gate(output, label, files):
    role = "parent" if label == "parent-resource-gates.jsonl" else "worker"
    require(
        label in {"parent-resource-gates.jsonl", "worker-resource-gates.jsonl"},
        "Exact CPU journal label",
    )
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    return CPUReferenceGate(output, label, role, boot)


def ownership(files):
    label = "artifacts/jev-runtime/public-affordance-matrix-harness-20261019-01/matrix_common.py"
    return module("_form_CPU_original_owned", label, files)


def child_argv(pid):
    return [
        x.decode()
        for x in (Path("/proc") / str(pid) / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    ]


def fresh_rows(plan, status):
    return [{**r, "status": status, "compiled_reference": None} for r in plan["ordered_cases"]]


def verify_worker_binding(binding, actual, nonce, expected, pid):
    # Scheduling state changes between parent and child observations; PID/start
    # and the independent process group/session are the ownership identity.
    fields = {"pid", "ppid", "pgid", "sid", "start_ticks"}
    released = binding.get("identity")
    require(
        type(released) is type(actual) is dict
        and fields <= set(released)
        and fields <= set(actual)
        and all(type(actual[k]) is type(released[k]) is int for k in fields)
        and all(actual[k] == released[k] for k in fields)
        and actual["start_ticks"] > 0
        and actual["pid"] == actual["pgid"] == actual["sid"] == pid
        and binding.get("nonce") == nonce
        and binding.get("plan_sha256") == expected,
        "Released exact owned PID/start/PG/SID",
    )


def capture_reference(plan, output):
    receipt_blob = checked(plan["original_compiler_receipt"])
    require(
        hashlib.sha256(receipt_blob).hexdigest()
        == "a8a5ebdefccc4ef1be01e673c36d36452efd30368b02921b81aae9cabd096b82",
        "Original genuine CPU compiler authority",
    )
    receipt = parse(receipt_blob)
    expected = {
        k: v
        for k, v in receipt["output_inventory_excluding_receipt"].items()
        if k == "processor-source-closure.json"
        or k.startswith(("processor-runtime-sources/", "processor-metadata/"))
    }
    require(
        expected == plan["reference_items"] and len(expected) == 58,
        "Original51 origins plus6 metadata and closure; complete58 items",
    )
    root = ROOT / plan["reference_root"]
    result = []
    for label, pin in expected.items():
        path = ordinary(root / label)
        require(path.is_relative_to(root), "Original reference confinement")
        blob = read(path)
        require(
            {"sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)} == pin,
            "Original processor reference bytes",
        )
        save(output, "reference-environment/" + label, blob)
        result.append((label, blob))
    return tuple(result)


def load_processor(reference_items, gate):
    import jev_public_action_artifact as public

    public.verify_reference_environment(reference_items)
    gate.check("cpu-reference-before-processor-load", "cpu")
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        public.MODEL_ID, revision=public.REVISION, local_files_only=True
    )
    refs = dict(reference_items)
    model = SimpleNamespace(
        processor=processor,
        tok=processor.tokenizer,
        max_context=8192,
        lm=SimpleNamespace(
            config=SimpleNamespace(**parse(refs["processor-metadata/config.json"])),
            generation_config=SimpleNamespace(eos_token_id=[1, 106, 50]),
        ),
    )
    public.verify_reference_environment(reference_items, model=model)
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    require(
        torch.get_num_threads() == torch.get_num_interop_threads() == 2,
        "Exact two Torch CPU threads/inter-op threads",
    )
    require(
        getattr(torch.cuda, "_initialized", False) is False,
        "No initialized CUDA in CPU processor worker",
    )
    return model


def processor_evidence(model, reference_items):
    import jev_public_action_artifact as public

    versions = public.verify_reference_environment(reference_items, model=model)
    components = {
        "processor": model.processor,
        "tokenizer": model.tok,
        "feature_extractor": model.processor.feature_extractor,
        "image_processor": model.processor.image_processor,
        "video_processor": model.processor.video_processor,
    }
    return {
        "versions": versions,
        "components": {
            k: {"class": type(v).__name__, "path": str(Path(inspect.getfile(type(v))).resolve())}
            for k, v in components.items()
        },
        "tokenizer_is_processor_tokenizer": model.tok is model.processor.tokenizer,
        "max_context": model.max_context,
        "generation_EOS": model.lm.generation_config.eos_token_id,
        "CPU_threads": __import__("torch").get_num_threads(),
        "CPU_interop_threads": __import__("torch").get_num_interop_threads(),
    }


def worker(plan_path, expected, output, nonce):
    plan, files = load_plan(plan_path, expected)
    require(
        sys.version_info[:2] == (3, 13) and sys.executable == plan["python"],
        "Actual reviewed Comfy3.13",
    )
    output = ordinary(output)
    intent = parse(read(output / "controller-intent.json", MIB))
    require(
        intent["nonce"] == nonce and intent["plan_sha256"] == expected, "Owned CPU worker intent"
    )
    rows, primary, secondary = fresh_rows(plan, "NOT_EXECUTED"), None, []
    runtime_checks = []
    state = {
        "admission_started": False,
        "admission_returned": False,
        "processor_load_started": False,
        "processor_load_returned": False,
        "compile_started": False,
        "compile_returned": False,
        "compiled_rows_in_memory": 0,
        "saved_rows": 0,
    }
    model = collection = None
    try:
        require(
            all(os.environ.get(k) == v for k, v in plan["environment"].items()),
            "Exact offline CPU worker environment",
        )
        require(
            not any(
                n.split(".")[0] in {"torch", "transformers", "tokenizers", "peft", "s1"}
                for n in sys.modules
            ),
            "Fresh pre-framework CPU child",
        )
        limit_evidence = configure_limits("worker")
        cuda_state()
        save(output, "worker-address-space-limit.json", encode(limit_evidence) + b"\n")
        deadline = time.monotonic() + 10
        release = output / "controller-release.json"
        while not release.exists():
            require(time.monotonic() < deadline, "Owned CPU release deadline")
            time.sleep(0.02)
        binding = parse(read(release, MIB))
        owned = ownership(files)
        actual = owned.process_table().get(os.getpid())
        verify_worker_binding(binding, actual, nonce, expected, os.getpid())
        gate = resource_gate(output, "worker-resource-gates.jsonl", files)
        gate.check("cpu-reference-worker-before-source", "cpu")
        authenticate_sources(files)
        forms = module("jev_owned_form_progress", "scripts/jev_owned_form_progress.py", files)
        state["admission_started"] = True
        admission_blob = checked(plan["admission_plan"])
        admission_plan = parse(admission_blob)
        collection = forms.validate_form_progress_collection(
            admission_blob,
            expected_plan_sha256=plan["admission_plan"]["sha256"],
            reviewed_source_pins=admission_plan["sources"],
            gate=gate,
        )
        state["admission_returned"] = True
        cases = collection.train_cases + collection.development_cases
        require(
            [{"id": c.id, "split": c.split, "family": c.annotation()["family"]} for c in cases]
            == plan["ordered_cases"],
            "Actual complete typed form case order",
        )
        save(output, "admission-provenance.json", encode(collection.provenance) + b"\n")
        gate.check("cpu-reference-before-reference-capture", "cpu")
        reference_items = capture_reference(plan, output)
        state["processor_load_started"] = True
        model = load_processor(reference_items, gate)
        state["processor_load_returned"] = True
        runtime_checks.append(runtime_guard("after-processor-load"))
        gate.check("cpu-reference-after-processor-load", "cpu")
        save(
            output,
            "actual-processor-before.json",
            encode(processor_evidence(model, reference_items)) + b"\n",
        )
        state["compile_started"] = True
        state["compiled_rows_in_memory"] = None
        rows = fresh_rows(plan, "COMPILE_RETURN_UNKNOWN")
        compiled, cfg, row_guards = compile_references_v2(
            forms, model, collection, reference_items, gate
        )
        runtime_checks.extend(row_guards)
        runtime_checks.append(runtime_guard("after-all-compile"))
        gate.check("cpu-reference-after-all-compile", "cpu")
        state.update(compile_returned=True, compiled_rows_in_memory=len(compiled))
        rows = fresh_rows(plan, "COMPILED_IN_MEMORY_EVIDENCE_INCOMPLETE")
        require(
            len(compiled) == len(cfg) == 23 and set(compiled) == set(cfg) == {c.id for c in cases},
            "Complete23 returned compile/CFG rows",
        )
        save(
            output,
            "actual-processor-after.json",
            encode(processor_evidence(model, reference_items)) + b"\n",
        )
        for ordinal, case in enumerate(cases):
            blob = (
                encode(
                    {
                        "case": case.annotation(),
                        "row": forms.common.serialized_row(compiled[case.id]),
                        "cfg_plan": cfg[case.id].evidence(),
                    }
                )
                + b"\n"
            )
            body = save(output, f"compiled/{ordinal:02d}.native-wire.json", case.wire_bytes)
            saved = save(output, f"compiled/{ordinal:02d}.json", blob)
            rows[ordinal].update(
                status="COMPLETE_REFERENCE", compiled_reference=saved, native_wire=body
            )
            state["saved_rows"] += 1
        for split in forms.COUNTS:
            save(
                output,
                "compile-projection-" + split + ".jsonl",
                b"".join(encode(r) + b"\n" for r in collection.compile_wire_projection(split)),
            )
        authenticate_sources(files)
        runtime_checks.append(runtime_guard("before-worker-terminal"))
        save(output, "runtime-guards.json", encode(runtime_checks) + b"\n")
        gate.check("cpu-reference-before-publication", "cpu")
        validate_gate_evidence(read(output / "worker-resource-gates.jsonl"), "worker")
    except BaseException as error:
        primary = error_value(error, "worker")
    finally:
        model = collection = None
        try:
            gc.collect()
        except BaseException as error:
            secondary.append(error_value(error, "CPU-object-release"))
        complete = (
            primary is None
            and not secondary
            and state["compile_returned"]
            and state["saved_rows"] == 23
            and all(r["status"] == "COMPLETE_REFERENCE" for r in rows)
        )
        result = {
            "schema_id": "owned-form-progress-native-CPU-reference-only-v2",
            "status": SUCCESS if complete else "FAILED_OR_INCOMPLETE_CPU_REFERENCE",
            "nonce": nonce,
            "plan_sha256": expected,
            "rows": rows,
            "counts": state,
            "primary_error": primary,
            "secondary_errors": secondary,
            "model_constructor_calls": 0,
            "model_forward_calls": 0,
            "optimizer_updates": 0,
            "HTTP_browser_Modal_calls": 0,
            "profile": PROFILE,
            "runtime_guard_count": len(runtime_checks),
            "containment_limits": LIMITS,
            "limitations": LIMITATIONS,
            "scope": "Actual processor-only CPU references if complete; not capacity/training/qualification. External owned controller/whole exit mandatory.",
        }
        published = terminal(output, "worker-terminal.json", result, secondary)
        if not published:
            report_unknown("UNKNOWN_WORKER_TERMINAL_EVIDENCE", primary, secondary)
    return 0 if complete and published else 1


def worker_summary(output, plan, expected, nonce):
    terminal_blob = read(output / "worker-terminal.json", MIB)
    value = parse(terminal_blob)
    require(
        value["schema_id"] == "owned-form-progress-native-CPU-reference-only-v2"
        and value["nonce"] == nonce
        and value["plan_sha256"] == expected
        and [{k: r[k] for k in ("id", "split", "family")} for r in value["rows"]]
        == plan["ordered_cases"],
        "Authenticated full23 worker summary, including failures",
    )
    return terminal_blob, value


def select_reference(output, plan, expected, nonce, *, summary=None):
    terminal_blob, value = (
        worker_summary(output, plan, expected, nonce) if summary is None else summary
    )
    require(
        value["status"] == SUCCESS
        and value["primary_error"] is None
        and not value["secondary_errors"]
        and value["counts"]["saved_rows"] == value["counts"]["compiled_rows_in_memory"] == 23
        and value["counts"]["compile_returned"] is True,
        "Complete genuine CPU child terminal",
    )
    for label, pin in plan["reference_items"].items():
        blob = read(output / "reference-environment" / label)
        require(
            {"sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)} == pin,
            "Closed original58 processor reference population",
        )
    closure = parse(read(output / "reference-environment/processor-source-closure.json"))
    for label in ("actual-processor-before.json", "actual-processor-after.json"):
        evidence = parse(read(output / label, MIB))
        require(
            evidence["components"] == closure["components"]
            and evidence["versions"] == closure["versions"]
            and evidence["tokenizer_is_processor_tokenizer"] is True
            and evidence["max_context"] == 8192
            and evidence["generation_EOS"] == [1, 106, 50]
            and evidence["CPU_threads"] == evidence["CPU_interop_threads"] == 2,
            "Genuine actual processor components/origins before/after",
        )
    for item in value["rows"]:
        require(item["status"] == "COMPLETE_REFERENCE", "No success selection from partial run")
        bodies = []
        for key in ("compiled_reference", "native_wire"):
            ref = item[key]
            blob = read(output / ref["path"])
            require(
                len(blob) == ref["size_bytes"]
                and hashlib.sha256(blob).hexdigest() == ref["sha256"],
                "Closed CPU row/body SHA",
            )
            bodies.append(blob)
        data, raw = parse(bodies[0]), bodies[1]
        row, cfg = data["row"], data["cfg_plan"]
        record = row["record"]
        p, length = record["prompt_tokens"], record["input_tokens"]
        require(
            data["case"]["id"] == record["case_id"] == item["id"]
            and record["split"] == item["split"]
            and record["original_wire_sha256"] == hashlib.sha256(raw).hexdigest(),
            "Closed CPU original case/raw request identity",
        )
        require(
            1 <= p < length
            and 1 <= length - p <= 32
            and p + 32 <= 8192
            and row["positions"] == record["logits_positions"] == list(range(p - 1, length - 1))
            and row["targets"] == record["target_token_ids"] == row["inputs"]["input_ids"][0][p:]
            and row["targets"][-1] == record["native_stop_token_id"] == 106
            and not any(t in {1, 106, 50} for t in row["targets"][:-1])
            and row["inputs"] == record["processor_tensors"],
            "Closed full token/mask/readout/firstEOS reference",
        )
        require(
            cfg["compiled_row_sha256"] == hashlib.sha256(encode(row)).hexdigest()
            and cfg["request_sha256"]
            == record["normalized_wire_sha256"]
            == hashlib.sha256(raw).hexdigest()
            and cfg["target_ids"] == row["targets"]
            and cfg["positions"] == row["positions"]
            and len(cfg["allowed_ids"]) == len(row["targets"])
            and all(
                t in allowed for t, allowed in zip(row["targets"], cfg["allowed_ids"], strict=True)
            ),
            "Closed full request-bound CFG path",
        )
    boot = validate_gate_evidence(read(output / "worker-resource-gates.jsonl"), "worker")
    require(
        validate_gate_evidence(read(output / "parent-resource-gates.jsonl"), "parent") == boot,
        "Sameboot parent/worker CPU-reference journals",
    )
    monitor_rows = validate_monitor_evidence(
        read(output / "owned-memory-monitor.jsonl"),
        boot,
        parent=parse(read(output / "parent-monitor-identity.json")),
        cleanup=parse(read(output / "owned-cleanup.json")),
    )
    guards = parse(read(output / "runtime-guards.json"))
    require(
        [r["phase"] for r in guards] == list(RUNTIME_GUARDS)
        and value["profile"] == PROFILE
        and value["runtime_guard_count"] == len(guards)
        and value["containment_limits"] == LIMITS
        and all(
            r["CUDA_VISIBLE_DEVICES"] == ""
            and r["CUDA_initialized"] is False
            and r["limits"] == {"role": "worker", "soft_bytes": CHILD_AS, "hard_bytes": CHILD_AS}
            for r in guards
        ),
        "All processor/23row/end uninitialized CUDA and AS guards",
    )
    return {
        "profile": PROFILE,
        "monitor_rows": monitor_rows,
        "worker_terminal_sha256": hashlib.sha256(terminal_blob).hexdigest(),
        "compiled_rows": 23,
        "counts": {"train": 13, "development": 10},
    }


def controller(plan_path, expected, output, *, run=False):
    plan, files = load_plan(plan_path, expected)
    output = ordinary(output)
    require(not output.exists(), "Fresh CPU output namespace")
    if not run:
        return {"status": "DRY_SOURCE_ONLY_NO_CHILD_FRAMEWORK_OR_PROCESSOR", "fixed_rows": 23}
    output.mkdir(mode=0o700)
    nonce = uuid.uuid4().hex
    argv = [
        plan["python"],
        "-B",
        "-s",
        str(Path(__file__).resolve()),
        "--role",
        "worker",
        "--plan",
        str(Path(plan_path).absolute()),
        "--plan-sha256",
        expected,
        "--output",
        str(output),
        "--nonce",
        nonce,
        "--run",
    ]
    save(
        output,
        "controller-intent.json",
        encode({"nonce": nonce, "plan_sha256": expected, "argv": argv}) + b"\n",
    )
    process = primary = gate = None
    secondary, known, groups, interrupted = [], {}, set(), []
    cleanup, selected, summary, returncode = None, None, None, None
    monitor = None
    old_handlers = {}

    def stop_signal(number, _frame):
        interrupted.append(number)
        raise InterruptedError("Owned form CPU controller signal")

    try:
        require(
            not any(
                n.split(".")[0] in {"torch", "transformers", "tokenizers", "peft", "s1"}
                for n in sys.modules
            ),
            "Fresh CPU controller, no framework/model",
        )
        limit_evidence = configure_limits("parent")
        cuda_state()
        save(output, "parent-address-space-limit.json", encode(limit_evidence) + b"\n")
        gate = resource_gate(output, "parent-resource-gates.jsonl", files)
        gate.check("cpu-reference-parent-before-source", "cpu")
        authenticate_sources(files)
        owned = ownership(files)
        parent_identity = owned.process_table().get(os.getpid())
        require(parent_identity is not None, "Fresh controller identity for memory monitor")
        save(output, "parent-monitor-identity.json", encode(parent_identity) + b"\n")
        monitor = OwnedMemoryMonitor(output, gate.boot, parent_identity)
        monitor.check({})
        old_handlers = {
            sig: signal.signal(sig, stop_signal) for sig in (signal.SIGINT, signal.SIGTERM)
        }
        env = {k: os.environ[k] for k in SECRET_FREE_ENV if k in os.environ}
        env.update(plan["environment"])
        with (
            (output / "child-stdout.txt").open("xb") as stdout,
            (output / "child-stderr.txt").open("xb") as stderr,
        ):
            os.chmod(stdout.name, 0o600)
            os.chmod(stderr.name, 0o600)
            gate.check("cpu-reference-parent-before-child", "cpu")
            process = subprocess.Popen(
                argv,
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            identity = owned.process_table().get(process.pid)
            require(
                identity is not None and identity["pgid"] == identity["sid"] == process.pid,
                "Fresh owned CPU PID/PG/SID",
            )
            require(child_argv(process.pid) == argv, "Exact CPU child argv")
            known[process.pid] = identity["start_ticks"]
            groups.add(process.pid)
            save(
                output,
                "controller-release.json",
                encode({"nonce": nonce, "plan_sha256": expected, "identity": identity}) + b"\n",
            )
            deadline, recorded = time.monotonic() + 600, set()
            while True:
                rows, _ = owned.extend_owned(process.pid, known, groups)
                for pid, row in rows.items():
                    key = (pid, row["start_ticks"])
                    if key not in recorded:
                        save(
                            output,
                            f"processes/{pid}-{row['start_ticks']}.json",
                            encode(row) + b"\n",
                        )
                        recorded.add(key)
                if process.poll() is not None:
                    break
                require(time.monotonic() < deadline, "Owned CPU600s deadline")
                monitor.check(rows)
                caps(output)
                require(stdout.tell() < MIB and stderr.tell() < MIB, "CPU stdout/stderr1MiB bound")
                time.sleep(0.05)
    except BaseException as error:
        primary = error_value(error, "controller")
    finally:
        for sig, handler in old_handlers.items():
            try:
                signal.signal(sig, handler)
            except BaseException as error:
                secondary.append(error_value(error, "signal-handler-restore"))
        if process is not None:
            try:
                owned = ownership(files)
                if not known:
                    identity = owned.process_table().get(process.pid)
                    if identity is not None and identity["pgid"] == identity["sid"] == process.pid:
                        known[process.pid] = identity["start_ticks"]
                        groups.add(process.pid)
                cleanup = owned.stop_owned(process, known, groups)
                save(output, "owned-cleanup.json", encode(cleanup) + b"\n")
            except BaseException as error:
                secondary.append(error_value(error, "owned-cleanup"))
            try:
                returncode = process.poll()
            except BaseException as error:
                secondary.append(error_value(error, "child-final-poll"))
        if cleanup and cleanup["confirmed"]:
            try:
                summary = worker_summary(output, plan, expected, nonce)
            except BaseException as error:
                secondary.append(error_value(error, "closed-worker-summary"))
        if (
            primary is None
            and not secondary
            and returncode == 0
            and cleanup
            and cleanup["confirmed"]
            and summary is not None
        ):
            try:
                authenticate_sources(files)
                monitor.check({})
                gate.check("cpu-reference-parent-after-child", "cpu")
                selected = select_reference(output, plan, expected, nonce, summary=summary)
            except BaseException as error:
                primary = error_value(error, "closed-reference-validation")
        result = {
            "schema_id": SCHEMA,
            "profile": PROFILE,
            "containment_limits": LIMITS,
            "limitations": LIMITATIONS,
            "status": "CLOSED_CPU_REFERENCE_READY_PENDING_EXTERNAL_CONTROLLER_EXIT"
            if selected is not None
            else "FAILED_OR_UNKNOWN_CPU_REFERENCE",
            "nonce": nonce,
            "plan_sha256": expected,
            "process_started": process is not None,
            "actual_returncode": returncode,
            "primary_error": primary,
            "secondary_errors": secondary,
            "worker_primary_error": summary[1]["primary_error"] if summary else None,
            "worker_counts": summary[1]["counts"] if summary else None,
            "cleanup": cleanup,
            "closure_confirmed": bool(cleanup and cleanup["confirmed"]),
            "signals_received": interrupted,
            "reference_ready": selected,
            "selection_requires_external_controller_exit0_and_group_absent": True,
            "all23_rows": summary[1]["rows"]
            if summary is not None
            else fresh_rows(
                plan, "UNKNOWN_AFTER_CHILD_FAILURE" if process is not None else "NOT_EXECUTED"
            ),
            "scope": "Whole exact child closure plus saved references only. External root must confirm this controller exit0/group absent before selecting reference; no capacity/training/model/browser score.",
        }
        published = terminal(output, "controller-terminal.json", result, secondary)
        if not published:
            report_unknown("UNKNOWN_CONTROLLER_TERMINAL_EVIDENCE", primary, secondary)
    return 0 if selected is not None and published else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("controller", "worker"), default="controller")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nonce")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.role == "worker":
        require(args.run and args.nonce, "Owned explicit worker execution")
        result = worker(args.plan, args.plan_sha256, args.output, args.nonce)
    else:
        result = controller(args.plan, args.plan_sha256, args.output, run=args.run)
    print(encode(result).decode())
    raise SystemExit(result if isinstance(result, int) else 0)
