"""Bounded read-only host/cgroup collection and explicitly selected Driver API."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import math
import os
import stat
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).absolute().parents[3]
PROC, CGROOT = Path("/proc"), Path("/sys/fs/cgroup")
H15 = (
    ROOT
    / "artifacts/jev-runtime/public-native-fit-memory-uma-probe-20261020-01/native_uma_peak_probe_01.py"
)
H15_SHA = "2ed7cda3054da62e7440bf57e4bfae15e26ee6d22de8b80aede6c8f53973e3cb"
DRIVER_RUNTIME = (
    ROOT
    / "artifacts/jev-runtime/public-native-fit-memory-peak-probe-20261018-01/native-toolchain-runtime-pins-02.json"
)
DRIVER_SHA = "1ad0a1749e1ab69235cf6356939ef55edf904ba8942794f6241ef9580c676582"


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path, maximum=1024**2):
    with Path(path).open("rb") as stream:
        blob = stream.read(maximum + 1)
    require(len(blob) <= maximum, "Bounded observation bytes")
    return blob.decode("utf-8", errors="strict")


def number(path, *, unlimited=False):
    text = read(path, 128).strip()
    if unlimited and text == "max":
        return None
    require(text.isdecimal(), "Typed kernel memory counter")
    return int(text)


def counters(path):
    result = {}
    for line in read(path, 8192).splitlines():
        key, value = line.split()
        require(key not in result and value.isdecimal(), "Typed distinct memory event counter")
        result[key] = int(value)
    require(result, "Complete memory event counters")
    return result


def process_identity(pid):
    require(type(pid) is int and pid > 0, "Selected process PID")
    values = read(PROC / str(pid) / "stat", 8192).rsplit(")", 1)[1].split()
    require(len(values) >= 20, "Complete selected process stat")
    return {
        "pid": pid,
        "start_ticks": int(values[19]),
        "pgid": int(values[2]),
        "sid": int(values[3]),
    }


def host_snapshot(expected, *, process=None):
    """Worker defaults to its actual PID; owner supplies a prior bound PID/start.

    The pure policy validates the complete membership/mount/ancestor population.
    No heuristic missing-ancestor fallback or cgroup-headroom host substitution.
    """
    pid = os.getpid() if process is None else process["pid"]
    before = process_identity(pid)
    require(process is None or before == process, "Selected process identity differs")
    directory = PROC / str(pid)
    membership = read(directory / "cgroup", 8192)
    mountinfo = "\n".join(
        line for line in read(directory / "mountinfo").splitlines() if " - cgroup" in line
    )
    group = Path(expected["cgroup_path"])
    require(
        group.is_relative_to(CGROOT)
        and group != CGROOT
        and ".." not in group.parts
        and not any(p.is_symlink() for p in (group, *group.parents)),
        "Selected ordinary unified cgroup path",
    )
    rows = []
    while group != CGROOT:
        initial = group.stat()
        require(stat.S_ISDIR(initial.st_mode), "Real kernel cgroup directory")
        row = {
            "path": str(group),
            "version": "cgroup2",
            "limit": number(group / "memory.max", unlimited=True),
            "current": number(group / "memory.current"),
            "peak": number(group / "memory.peak"),
            "device": initial.st_dev,
            "inode": initial.st_ino,
            "swap_limit": number(group / "memory.swap.max", unlimited=True),
            "swap_current": number(group / "memory.swap.current"),
            "events": counters(group / "memory.events"),
            "swap_events": counters(group / "memory.swap.events"),
        }
        final = group.stat()
        require(
            (initial.st_dev, initial.st_ino) == (final.st_dev, final.st_ino),
            "Cgroup identity changed during observation",
        )
        rows.append(row)
        group = group.parent
    memory = {}
    for line in read(PROC / "meminfo", 65536).splitlines():
        key, _, value = line.partition(":")
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            fields = value.split()
            require(
                len(fields) == 2 and fields[0].isdecimal() and fields[1] == "kB", "KiB host memory"
            )
            require(key not in memory, "Distinct host memory field")
            memory[key] = int(fields[0]) * 1024
    result = {
        "boot_id": read(PROC / "sys/kernel/random/boot_id", 128).strip(),
        "proc_meminfo": memory,
        "cgroup_membership": membership,
        "cgroup_mountinfo": mountinfo,
        "cgroup_memory": rows,
        "cuda": None,
    }
    require(process_identity(pid) == before, "Selected process changed during host observation")
    return result


def cuda_snapshot(torch):
    free, total = torch.cuda.mem_get_info(0)
    props = torch.cuda.get_device_properties(0)
    fraction = torch.cuda.get_per_process_memory_fraction(0)
    require(type(fraction) is float and math.isfinite(fraction), "Actual allocator fraction")
    return {
        "device": 0,
        "name": props.name,
        "uuid": str(props.uuid),
        "free_bytes": free,
        "total_bytes": total,
        "allocated_bytes": torch.cuda.memory_allocated(0),
        "reserved_bytes": torch.cuda.memory_reserved(0),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
        "allocator_backend": torch.cuda.memory.get_allocator_backend(),
        "allocator_fraction": fraction,
        "allocator_allowed_bytes_upper": math.floor(fraction * total),
    }


def checked(path, expected):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Ordinary selected source")
    blob = read(path).encode()
    require(
        hashlib.sha256(blob).hexdigest() == expected, "Selected Driver mechanism source differs"
    )
    return blob


class DriverIdentity:
    """Same H15 metadata and mapped-symbol authority, plus actual device count.

    Constructor is source-only. check() is an explicit future model-worker call;
    no library is opened until that call. It never loads an absent libcuda.
    """

    def __init__(self):
        blob = checked(H15, H15_SHA)
        spec = importlib.util.spec_from_file_location("_measured_d3_selected_h15", H15)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        exec(compile(blob, str(H15), "exec", dont_inherit=True), vars(module))
        self.h = module
        self.runtime = json.loads(checked(DRIVER_RUNTIME, DRIVER_SHA))

    def check(self, torch, host):
        torch.cuda.synchronize(0)
        require(
            torch.cuda.device_count() == 1 and torch.cuda.current_device() == 0,
            "One actual current Torch CUDA device",
        )
        h = self.h
        metadata = h.uma_driver_metadata(self.runtime)
        library = ctypes.CDLL(h.UMA_LIBCUDA, mode=os.RTLD_NOLOAD | os.RTLD_NOW | os.RTLD_LOCAL)
        symbols = {name: getattr(library, name) for name in h.UMA_SYMBOLS}
        addresses = {name: ctypes.cast(fn, ctypes.c_void_p).value for name, fn in symbols.items()}
        maps = read(PROC / str(os.getpid()) / "maps", 8 * 1024**2)
        h.uma_symbol_mappings(maps, addresses, metadata)
        count = library.cuDeviceGetCount
        count.argtypes, count.restype = [ctypes.POINTER(ctypes.c_int)], ctypes.c_int
        address = ctypes.cast(count, ctypes.c_void_p).value
        mapped = False
        for line in maps.splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and fields[5] == metadata["resolved"] and "x" in fields[1]:
                low, high = [int(x, 16) for x in fields[0].split("-")]
                device = os.makedev(*(int(x, 16) for x in fields[3].split(":")))
                if low <= address < high:
                    mapped = device == metadata["device"] and int(fields[4]) == metadata["inode"]
        require(mapped, "Actual count symbol belongs to selected executable libcuda mapping")
        actual_count = ctypes.c_int(-1)
        require(count(ctypes.byref(actual_count)) == 0, "Raw device count query succeeded")
        raw = {
            **h.uma_driver_read(symbols, ctypes),
            "boot_id": host["boot_id"],
            "device_count": actual_count.value,
        }
        props = torch.cuda.get_device_properties(0)
        require(
            raw["total_bytes"] == props.total_memory == torch.cuda.mem_get_info(0)[1]
            and raw["name"] == props.name,
            "Raw/Torch actual total and device name differ",
        )
        require(h.uma_driver_metadata(self.runtime) == metadata, "Stable selected driver metadata")
        h.uma_symbol_mappings(
            read(PROC / str(os.getpid()) / "maps", 8 * 1024**2), addresses, metadata
        )
        self.evidence = {
            "raw_driver": raw,
            "policy_uuid": str(uuid.UUID(raw["uuid"])),
            "library_metadata": metadata,
            "count_symbol_address": address,
            "scope": "Selected metadata/mapped symbols and actual Driver values; no universal DSO payload attestation",
        }
        # Retain the actual H15 hex value above. The pure policy consumes one
        # explicit canonical UUID representation; no memory value is normalized.
        return {**raw, "uuid": self.evidence["policy_uuid"]}
