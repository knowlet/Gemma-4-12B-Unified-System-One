#!/usr/bin/env python3
"""Owned, gated, read-only metadata probe for one fixed checkpoint inode.

Linux mincore API and bounded-window design references are in research-report-01.json.
This independent implementation does not copy upstream implementation code.
Default execution is dry. Only --run can open/map the checkpoint, using PROT_NONE.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import resource
import stat
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
PLAN = BASE / "plan-01.json"
GIB = 1024**3
CHUNK = 256 * 1024**2
PAGE = 4096
VECTOR_CAP = 64 * 1024
CHECKPOINT_BYTES = 23919549408
SNAPSHOT = (
    "artifacts/jev-local-hf/hub/models--knowlet--Gemma-4-12B-Unified-System-One/"
    "snapshots/a66f836b56605039fe040f330180e336d19b3362/model.safetensors"
)
BLOB = "artifacts/jev-local-hf/hub/blobs/04/049761b5163d98cd27f2fb2dffb86d20b4dc6894baf3ad711cff387e29b2a09e"
DEPENDENCIES = (
    "scripts/jev_public_action_artifact.py",
    "scripts/jev_browser_fit_public_actions.py",
)
CPU_PHASE = "before-any-project-framework-or-parent-payload"
OUTPUT_CAP = 2 * 1024**2
TERMINAL_RESERVE = 64 * 1024
SOURCE_CAP = 512 * 1024
JOURNAL_CAP = 512 * 1024
WALL_SECONDS = 120  # Cooperative check before each chunk; not a native syscall hard deadline.
IDENTITY_FIELDS = (
    "dev",
    "inode",
    "size_bytes",
    "uid",
    "gid",
    "mode",
    "nlink",
    "mtime_ns",
    "ctime_ns",
)
HOST_FIELDS = (
    "MemTotal",
    "MemFree",
    "MemAvailable",
    "Buffers",
    "Cached",
    "SReclaimable",
    "Shmem",
    "Dirty",
    "Writeback",
    "Active(file)",
    "Inactive(file)",
    "SwapTotal",
    "SwapFree",
    "Unevictable",
    "Mlocked",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def encode(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def parse(blob):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        blob, object_pairs_hook=pairs, parse_constant=lambda _: require(False, "Nonfinite JSON")
    )


def ordinary(path):
    path = Path(path).absolute()
    require(".." not in path.parts, "Ordinary path required")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Symlink path refused")
    return path


def pin_source(path):
    path = ordinary(path)
    size = path.stat().st_size
    require(0 < size <= SOURCE_CAP, "Bounded source metadata only")
    blob = path.read_bytes()
    require(len(blob) == size, "Stable source length")
    return {"sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)}


def identity(value):
    require(stat.S_ISREG(value.st_mode), "Regular checkpoint inode required")
    return {
        "dev": value.st_dev,
        "inode": value.st_ino,
        "size_bytes": value.st_size,
        "uid": value.st_uid,
        "gid": value.st_gid,
        "mode": stat.S_IMODE(value.st_mode),
        "nlink": value.st_nlink,
        "mtime_ns": value.st_mtime_ns,
        "ctime_ns": value.st_ctime_ns,
    }


def fixed_plan(plan):
    require(
        type(plan) is dict and plan.get("schema") == "GX10_cache_residency_plan_v1",
        "Exact plan schema",
    )
    require(
        plan.get("snapshot") == SNAPSHOT and plan.get("canonical_blob") == BLOB,
        "Fixed checkpoint paths",
    )
    require(plan.get("checkpoint_size_bytes") == CHECKPOINT_BYTES, "Fixed full checkpoint size")
    require(
        plan.get("chunk_bytes") == CHUNK and plan.get("page_size") == PAGE,
        "Fixed map window and page size",
    )
    require(plan.get("vector_cap_bytes") == VECTOR_CAP, "Fixed bounded residency vector")
    require(
        plan.get("output_cap_bytes") == OUTPUT_CAP and plan.get("wall_seconds") == WALL_SECONDS,
        "Fixed evidence and wall bounds",
    )
    require(
        plan.get("CPU_phase") == CPU_PHASE
        and type(plan.get("swap_or_resident_credit_bytes")) is int
        and plan["swap_or_resident_credit_bytes"] == 0,
        "Unchanged original CPU scope; zero credit",
    )
    expected = plan.get("checkpoint_metadata")
    require(
        type(expected) is dict and set(expected) == set(IDENTITY_FIELDS), "Exact identity fields"
    )
    require(all(type(v) is int and v >= 0 for v in expected.values()), "Strict integer identity")
    require(
        expected["size_bytes"] == CHECKPOINT_BYTES and expected["mode"] == 0o400,
        "Expected fixed readonly inode",
    )
    require(
        type(plan.get("expected_boot_id")) is str and bool(plan["expected_boot_id"]),
        "Explicit same-boot launch",
    )
    refs = plan.get("source_dependencies")
    require(type(refs) is dict and set(refs) == set(DEPENDENCIES), "Exact original CPU source pair")
    for name in DEPENDENCIES:
        expected_pin = refs[name]
        require(
            type(expected_pin) is dict and set(expected_pin) == {"sha256", "size_bytes"},
            "Exact source pin",
        )
        require(
            type(expected_pin["sha256"]) is str and len(expected_pin["sha256"]) == 64,
            "SHA256 source pin",
        )
        require(
            type(expected_pin["size_bytes"]) is int
            and 0 < expected_pin["size_bytes"] <= SOURCE_CAP,
            "Bounded original source size",
        )
    return plan


def verify_sources(plan, expected_source, expected_plan):
    source = pin_source(Path(__file__))
    plan_ref = pin_source(PLAN)
    require(source["sha256"] == expected_source, "Reviewed probe source SHA mismatch")
    require(plan_ref["sha256"] == expected_plan, "Reviewed plan SHA mismatch")
    require(parse(PLAN.read_bytes()) == plan, "Plan content drift")
    dependencies = {}
    for name in DEPENDENCIES:
        dependencies[name] = pin_source(ROOT / name)
        require(
            dependencies[name] == plan["source_dependencies"][name],
            "Original CPU dependency drift: " + name,
        )
    return {"probe": source, "plan": plan_ref, "dependencies": dependencies}


def original_gate_class(plan):
    modules = []
    for index, name in enumerate(DEPENDENCIES):
        path = ordinary(ROOT / name)
        require(
            pin_source(path) == plan["source_dependencies"][name], "Gate source drift before import"
        )
        module_name = "jev_public_action_artifact" if index == 0 else "_gx10_cache_original_gate"
        if module_name in sys.modules:
            module = sys.modules[module_name]
            require(Path(module.__file__).resolve() == path, "Existing gate module origin mismatch")
        else:
            spec = importlib.util.spec_from_file_location(module_name, path)
            require(spec is not None and spec.loader is not None, "Original pure source loader")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        modules.append(module)
    require(modules[1].artifact is modules[0], "Original ResourceGate dependency identity")
    return modules[1].ResourceGate


def host_metrics():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in HOST_FIELDS:
            number, unit = value.split()
            require(unit == "kB", "Kernel meminfo unit")
            values[key] = int(number) * 1024
    require(set(values) == set(HOST_FIELDS), "Complete host memory snapshot")
    io = {}
    for line in Path("/proc/self/io").read_text().splitlines():
        key, value = line.split(":", 1)
        io[key] = int(value)
    require({"rchar", "syscr", "read_bytes"} <= set(io), "Kernel process IO snapshot")
    return {
        "monotonic": time.monotonic(),
        "host": values,
        "process_io": io,
        "process_major_faults": resource.getrusage(resource.RUSAGE_SELF).ru_majflt,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    }


def process_identity():
    prefix, rest = Path("/proc/self/stat").read_text().rsplit(")", 1)
    fields = rest.split()
    return {
        "pid": int(prefix.split("(", 1)[0]),
        "start_ticks": int(fields[19]),
        "pgid": int(fields[2]),
    }


def output_budget(output, pending=0, *, terminal=False):
    total = 0
    for path in output.iterdir():
        require(path.is_file() and not path.is_symlink(), "Owned ordinary output files only")
        total += path.stat().st_size
    reserve = 0 if terminal else TERMINAL_RESERVE
    require(total + pending + reserve <= OUTPUT_CAP, "Bounded evidence with terminal reserve")


def write_json(output, name, value, *, terminal=False):
    require(name in {"binding.json", "terminal.json", "measurement.json"}, "Fixed output name")
    blob = encode(value)
    require(len(blob) <= TERMINAL_RESERVE, "Small fixed evidence file")
    output_budget(output, len(blob), terminal=terminal)
    temporary = output / (name + ".next")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        require(os.write(fd, blob) == len(blob), "Complete owned evidence write")
        os.fsync(fd)
    finally:
        os.close(fd)
    target = output / name
    require(name == "terminal.json" or not target.exists(), "Fresh evidence destination")
    os.replace(temporary, target)
    fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def append_journal(output, row):
    blob = encode(row)
    path = output / "chunks.jsonl"
    size = path.stat().st_size if path.exists() else 0
    require(size + len(blob) <= JOURNAL_CAP, "Bounded chunk journal")
    output_budget(output, len(blob))
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        require(os.fstat(fd).st_size == size, "Owned chunk journal identity")
        require(os.write(fd, blob) == len(blob), "Complete chunk journal write")
        os.fsync(fd)
    finally:
        os.close(fd)


class LibcMincore:
    """No mapped payload is dereferenced; only kernel-filled vector is read."""

    def __init__(self):
        require(
            sys.platform == "linux" and ctypes.sizeof(ctypes.c_long) == 8,
            "Linux64 mmap ABI required",
        )
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.libc.mmap.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_long,
        ]
        self.libc.mmap.restype = ctypes.c_void_p
        self.libc.mincore.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_ubyte),
        ]
        self.libc.mincore.restype = ctypes.c_int
        self.libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.libc.munmap.restype = ctypes.c_int

    def map(self, fd, offset, length):
        address = self.libc.mmap(None, length, 0, 2, fd, offset)
        if address == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(), "mmap PROT_NONE failed")
        return int(address or 0)

    def residency(self, address, length, pages):
        require(0 < pages <= VECTOR_CAP, "Bounded native vector")
        vector = (ctypes.c_ubyte * pages)()
        if self.libc.mincore(address, length, vector) != 0:
            raise OSError(ctypes.get_errno(), "mincore failed")
        return bytes(vector)

    def unmap(self, address, length):
        if self.libc.munmap(address, length) != 0:
            raise OSError(ctypes.get_errno(), "munmap failed")


def windows():
    for offset in range(0, CHECKPOINT_BYTES, CHUNK):
        length = min(CHUNK, CHECKPOINT_BYTES - offset)
        yield offset, length, (length + PAGE - 1) // PAGE


def scan_fd(fd, *, ops, gate_step, record, deadline, clock=time.monotonic):
    resident_pages = resident_file_bytes = chunks = 0
    for offset, length, pages in windows():
        require(clock() < deadline, "Bounded measurement wall time")
        gate_record = gate_step("before-map", chunks)
        address = ops.map(fd, offset, length)
        primary = None
        try:
            require(
                type(address) is int and address >= 0 and address % PAGE == 0,
                "Aligned native mapping",
            )
            vector = ops.residency(address, length, pages)
            require(type(vector) is bytes and len(vector) == pages, "Complete residency vector")
            count = sum(value & 1 for value in vector)
            file_bytes = count * PAGE
            if length % PAGE and vector[-1] & 1:
                file_bytes -= PAGE - length % PAGE
        except BaseException as exc:
            primary = exc
        try:
            ops.unmap(address, length)
        except BaseException as exc:
            if primary is None:
                primary = exc
            else:
                primary.add_note("Owned mapping cleanup failed: " + repr(exc))
        if primary is not None:
            raise primary
        record(
            {
                "chunk": chunks,
                "offset": offset,
                "length": length,
                "pages": pages,
                "resident_pages": count,
                "resident_file_bytes": file_bytes,
                "gate_sequence": gate_record["sequence"],
                "mapping_unmapped": True,
            }
        )
        resident_pages += count
        resident_file_bytes += file_bytes
        chunks += 1
    return {
        "chunks": chunks,
        "queried_file_bytes": CHECKPOINT_BYTES,
        "pages": (CHECKPOINT_BYTES + PAGE - 1) // PAGE,
        "resident_pages": resident_pages,
        "resident_page_bytes": resident_pages * PAGE,
        "resident_file_bytes": resident_file_bytes,
    }


def inspect_checkpoint(plan, *, gate_step, ops, record, file_api=os, clock=time.monotonic):
    expected = plan["checkpoint_metadata"]
    snapshot, canonical = ROOT / SNAPSHOT, ordinary(ROOT / BLOB)
    require(
        snapshot.resolve(strict=True) == canonical,
        "Pinned snapshot resolves to exact canonical blob",
    )
    path_before = identity(file_api.stat(canonical, follow_symlinks=False))
    require(path_before == expected, "Fresh exact checkpoint path metadata")
    require(
        file_api.geteuid() == path_before["uid"],
        "Inode owner required for genuine mincore visibility",
    )
    require(os.sysconf("SC_PAGE_SIZE") == PAGE, "Exact runtime page size")
    gate_step("before-open", None)
    fd = file_api.open(canonical, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    primary = None
    result = None
    try:
        fd_before = identity(file_api.fstat(fd))
        require(fd_before == path_before, "Opened descriptor exact identity")
        result = scan_fd(
            fd,
            ops=ops,
            gate_step=gate_step,
            record=record,
            deadline=clock() + WALL_SECONDS,
            clock=clock,
        )
        fd_after = identity(file_api.fstat(fd))
        path_after = identity(file_api.stat(canonical, follow_symlinks=False))
        require(fd_after == fd_before == path_after, "Checkpoint path/descriptor metadata stable")
        require(snapshot.resolve(strict=True) == canonical, "Stable pinned snapshot resolution")
        gate_step("after-measurement", None)
        result["checkpoint_identity_before"] = fd_before
        result["checkpoint_identity_after"] = fd_after
        result["inode_count"] = 1
        result["aliases_not_added"] = path_before["nlink"]
    except BaseException as exc:
        primary = exc
    try:
        file_api.close(fd)
    except BaseException as exc:
        if primary is None:
            primary = exc
        else:
            primary.add_note("Owned descriptor cleanup failed: " + repr(exc))
    if primary is not None:
        raise primary
    require(result is not None, "Completed metadata result")
    result["descriptor_closed"] = True
    return result


def run(plan, output, expected_source, expected_plan):
    sources = verify_sources(plan, expected_source, expected_plan)
    output = ordinary(output)
    require(
        output.parent == BASE and output.name.startswith("actual-") and not output.exists(),
        "Fresh owned actual output",
    )
    output.mkdir(mode=0o700)
    terminal = {
        "schema": "GX10_cache_residency_terminal_v1",
        "status": "UNKNOWN",
        "complete": False,
        "owner": process_identity(),
        "source_bindings": sources,
        "error": {"type": "Pending", "message": "Measurement not complete"},
        "scope": "Residency snapshot only; no allocation, reclaimability, model, training or task-quality claim",
    }
    write_json(output, "terminal.json", terminal, terminal=True)
    primary = None
    try:
        before = host_metrics()
        terminal["host_before"] = before
        write_json(output, "terminal.json", terminal, terminal=True)
        require(before["boot_id"] == plan["expected_boot_id"], "Plan boot mismatch")
        Gate = original_gate_class(plan)
        gate = Gate(output / "resource-gates.jsonl", plan["expected_boot_id"])
        write_json(
            output,
            "binding.json",
            {
                "owner": terminal["owner"],
                "sources": sources,
                "expected_boot_id": plan["expected_boot_id"],
                "checkpoint_metadata": plan["checkpoint_metadata"],
            },
        )

        def gate_step(stage, chunk):
            output_budget(output)
            value = gate.check(CPU_PHASE, "cpu")
            require(
                value["admitted"] is True and value["swap_or_resident_credit_bytes"] == 0,
                "Original durable CPU admission",
            )
            append_journal(
                output,
                {
                    "event": "protected-stage",
                    "stage": stage,
                    "chunk": chunk,
                    "gate_sequence": value["sequence"],
                },
            )
            output_budget(output)
            return value

        gate_step("before-native-ABI-binding", None)
        ops = LibcMincore()
        measurement = inspect_checkpoint(
            plan,
            gate_step=gate_step,
            ops=ops,
            record=lambda value: append_journal(output, {"event": "chunk-complete", **value}),
        )
        after = host_metrics()
        terminal["host_after"] = after
        require(after["boot_id"] == before["boot_id"], "Stable observed boot")
        require(
            verify_sources(plan, expected_source, expected_plan) == sources,
            "Protected source closure unchanged",
        )
        measurement.update(
            {
                "host_before": before,
                "host_after": after,
                "process_read_bytes_delta": after["process_io"]["read_bytes"]
                - before["process_io"]["read_bytes"],
                "process_major_faults_delta": after["process_major_faults"]
                - before["process_major_faults"],
                "payload_contract": "Checkpoint fd only mmap(PROT_NONE,MAP_PRIVATE),mincore,fstat,close; mapping never dereferenced. This follows source/syscall contract, not counter-only proof.",
                "limitations": "Sequential per-chunk snapshots can stale and do not establish evictability or future CUDA allocation. Native diagnostics use different metrics; nlink aliases counted once.",
            }
        )
        write_json(output, "measurement.json", measurement)
        terminal.update(
            status="COMPLETE_READ_ONLY_RESIDENCY_SNAPSHOT",
            complete=True,
            error=None,
            resource_gate_count=gate.sequence,
            measurement=pin_source(output / "measurement.json"),
        )
    except BaseException as exc:
        primary = exc
        terminal["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "notes": getattr(exc, "__notes__", []),
        }
    if "host_before" in terminal and "host_after" not in terminal:
        try:
            terminal["host_after"] = host_metrics()
        except BaseException as secondary:
            require(primary is not None, "Failed snapshot retains primary error")
            terminal["error"]["notes"].append("Final kernel metrics failed: " + repr(secondary))
    try:
        terminal["owner_after"] = process_identity()
        terminal["process_identity_stable"] = terminal["owner_after"] == terminal["owner"]
        write_json(output, "terminal.json", terminal, terminal=True)
    except BaseException as secondary:
        if primary is not None:
            primary.add_note("Terminal persistence failed: " + repr(secondary))
            raise primary from secondary
        raise
    return 0 if terminal["complete"] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if not args.run:
        print(
            json.dumps(
                {
                    "status": "DRY_SOURCE_ONLY_NO_CHECKPOINT_OPEN_OR_MAPPING",
                    "plan": str(PLAN),
                    "chunk_bytes": CHUNK,
                    "vector_cap_bytes": VECTOR_CAP,
                    "actual_measurement": False,
                }
            )
        )
        return 0
    require(
        args.output is not None and args.expected_source_sha256 and args.expected_plan_sha256,
        "Explicit owned output and two reviewed SHA256 values required",
    )
    plan = fixed_plan(parse(PLAN.read_bytes()))
    return run(plan, args.output, args.expected_source_sha256, args.expected_plan_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
