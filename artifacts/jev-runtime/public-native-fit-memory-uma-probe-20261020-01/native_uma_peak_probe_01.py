"""Independent two-step memory research point. Default never starts a unit/GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import stat
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
PLAN = BASE / "plan-01.json"
UMA_KIND = "gb10-native-fixed-cache-uma-memavailable-capacity-v1"
UMA_LIBCUDA = "/usr/lib/aarch64-linux-gnu/libcuda.so.580.178.04"
UMA_TOTAL = 130595860480
UMA_SYMBOLS = (
    "cuCtxGetDevice",
    "cuDeviceGetAttribute",
    "cuDeviceGetUuid",
    "cuDeviceGetName",
    "cuDeviceTotalMem_v2",
)
GIB = 1024**3
MIB = 1024**2
CAP = 40 * GIB
STOP = CAP - GIB // 2
TOTAL_CAP = GIB
ARM_CAP = (TOTAL_CAP - 4 * MIB) // 2
FILE_CAP = 64 * MIB
CACHE_POLICY = "fixed-post-forward-backward-unused-cache-release-v1"
LIMITS = {"native.log": 12 * MIB, "samples.jsonl": 16 * MIB, "phases.jsonl": 4 * MIB}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def parse(blob):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        blob, object_pairs_hook=pairs, parse_constant=lambda _: fail("Nonfinite JSON")
    )


def fail(message):
    raise ValueError(message)


def ordinary(path):
    path = Path(path).absolute()
    require(
        ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "Ordinary path required",
    )
    return path


def read(path, maximum=64 * MIB):
    path = ordinary(path)
    require(type(maximum) is int and maximum >= 0, "Nonnegative integer read bound")
    chunks, remaining = [], maximum + 1
    with path.open("rb") as stream:
        while remaining:
            chunk = stream.read(min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    blob = b"".join(chunks)
    require(len(blob) <= maximum, "Bounded file read")
    return blob


def file_pin(path):
    path = ordinary(path)
    before = path.stat()
    require(stat.S_ISREG(before.st_mode) and before.st_size <= 8 * GIB, "Bounded regular source")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * MIB):
            digest.update(chunk)
    after = path.stat()
    require(
        (before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_ino, after.st_size, after.st_mtime_ns),
        "File changed during hash",
    )
    return {"sha256": digest.hexdigest(), "size_bytes": before.st_size}


def ref_read(ref):
    path = ROOT / ref["path"]
    require(
        file_pin(path) == {k: ref[k] for k in ("sha256", "size_bytes")},
        "Externally pinned input drift",
    )
    return read(path)


def write(path, value):
    blob = encode(value) + b"\n"
    require(len(blob) <= 4 * MIB, "Small evidence record")
    fd = os.open(ordinary(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())


def replace(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    write(temporary, value)
    os.replace(temporary, path)


def append(path, value, limit):
    blob = encode(value) + b"\n"
    require(
        len(blob) <= 65536 and (path.stat().st_size if path.exists() else 0) + len(blob) <= limit,
        "Bounded evidence journal",
    )
    with path.open("ab") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())


def tree_inventory(path):
    files = {}
    for item in sorted(path.rglob("*")):
        require(not item.is_symlink(), "No evidence symlink")
        if item.is_file():
            files[item.relative_to(path).as_posix()] = file_pin(item)
    require(
        len(files) <= 10000 and sum(p["size_bytes"] for p in files.values()) <= TOTAL_CAP,
        "Bounded whole research tree",
    )
    return files


def arm_bound(path, pending=0):
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    require(total + pending + 256 * 1024 <= ARM_CAP, "Bounded arm evidence allowance")


def final_inventory(path, *, reserved):
    files = tree_inventory(path)
    require(
        sum(p["size_bytes"] for p in files.values()) + reserved <= TOTAL_CAP,
        "Final receipt capacity reserved before completion",
    )
    return files


def boot_id():
    return read("/proc/sys/kernel/random/boot_id").decode().strip()


def host_memory():
    values = {}
    for line in read("/proc/meminfo").decode().splitlines():
        key, value = line.split(":", 1)
        if key in {
            "MemTotal",
            "MemAvailable",
            "MemFree",
            "Cached",
            "SReclaimable",
            "SwapTotal",
            "SwapFree",
            "Hugepagesize",
            "Hugetlb",
        }:
            number, unit = value.split()
            require(unit == "kB", "Native memory unit")
            values[key] = int(number) * 1024
        elif key in {"HugePages_Total", "HugePages_Free"}:
            fields = value.split()
            require(len(fields) == 1, "Typed hugetlb page count")
            values[key] = int(fields[0])
    values["boot_id"] = boot_id()
    return values


def research_floor(observation, *, startup):
    require(
        all(
            type(observation[k]) is int and observation[k] > 0 for k in ("MemTotal", "MemAvailable")
        ),
        "Observed physical memory",
    )
    reserve = max(24 * GIB, (observation["MemTotal"] + 4) // 5)
    old = reserve + 2 * 23919549408 + 30 * GIB + 256 * MIB
    required = max(old, reserve + 2 * CAP + 4 * GIB) if startup else reserve + CAP + 8 * GIB
    return {"reserve_bytes": reserve, "required_bytes": required, "old_train_required_bytes": old}


def process_identity(pid):
    prefix, rest = read(Path("/proc") / str(pid) / "stat").decode().rsplit(")", 1)
    fields = rest.split()
    return {
        "pid": int(prefix.split("(", 1)[0]),
        "state": fields[0],
        "ppid": int(fields[1]),
        "pgid": int(fields[2]),
        "sid": int(fields[3]),
        "startticks": int(fields[19]),
    }


def cgroup_membership(pid):
    lines = read(Path("/proc") / str(pid) / "cgroup").decode().splitlines()
    values = [line.split(":", 2)[2] for line in lines if line.startswith("0::")]
    require(
        len(values) == 1 and values[0].startswith("/") and ".." not in Path(values[0]).parts,
        "Observed unified cgroup membership",
    )
    return values[0]


def cg_path(group):
    return ordinary(Path("/sys/fs/cgroup") / group.removeprefix("/"))


def scalar(path):
    value = read(path, MIB).decode().strip()
    require(value == "max" or value.isdigit(), "Typed cgroup scalar")
    return None if value == "max" else int(value)


def events(path):
    return {
        key: int(value)
        for key, value in (line.split() for line in read(path).decode().splitlines())
    }


def cgroup_snapshot(group):
    path = cg_path(group)
    return {
        "group": group,
        "current": scalar(path / "memory.current"),
        "peak": scalar(path / "memory.peak"),
        "max": scalar(path / "memory.max"),
        "high": scalar(path / "memory.high"),
        "swap": scalar(path / "memory.swap.current"),
        "swap_max": scalar(path / "memory.swap.max"),
        "oom_group": scalar(path / "memory.oom.group"),
        "events": events(path / "memory.events"),
        "stat": events(path / "memory.stat"),
        "procs": [int(p) for p in read(path / "cgroup.procs").split()],
    }


def verify_unit_limits(group):
    own = cgroup_snapshot(group)
    require(
        own["max"] == CAP
        and own["high"] is None
        and own["swap_max"] == 0
        and own["oom_group"] == 1,
        "Exact own unit hard limits",
    )
    require(
        own["swap"] == 0 and all(own["events"].get(k, 0) == 0 for k in ("max", "oom", "oom_kill")),
        "Fresh unit has no pressure event",
    )
    ancestors = []
    for path in cg_path(group).parents:
        if not path.is_relative_to(Path("/sys/fs/cgroup")):
            break
        if not (path / "memory.max").exists():
            require(path == Path("/sys/fs/cgroup"), "Missing ancestor memory observation")
            continue
        limit, current = scalar(path / "memory.max"), scalar(path / "memory.current")
        require(
            type(current) is int
            and current >= 0
            and (limit is None or limit > 0 and limit - current >= CAP),
            "Effective ancestor full-cap headroom",
        )
        ancestors.append({"path": str(path), "max": limit, "current": current})
    require(bool(ancestors), "Complete observed controller ancestors")
    return {"own": own, "ancestors": ancestors}


def service_show(unit, *, timeout=10):
    require(type(timeout) in {int, float} and 0 < timeout <= 10, "Bounded metadata query timeout")
    keys = [
        "InvocationID",
        "LoadState",
        "ControlGroup",
        "MainPID",
        "ExecMainPID",
        "ExecMainCode",
        "ExecMainStatus",
        "Result",
        "ActiveState",
        "SubState",
        "MemoryMax",
        "MemoryHigh",
        "MemorySwapMax",
        "MemoryAccounting",
        "MemoryPeak",
        "MemorySwapPeak",
        "OOMPolicy",
        "KillMode",
        "Type",
        "RuntimeMaxUSec",
        "TimeoutStopUSec",
    ]
    result = subprocess.run(
        ["systemctl", "--user", "show", unit, *["--property=" + k for k in keys]],
        capture_output=True,
        timeout=timeout,
        check=True,
    )
    require(len(result.stdout) <= MIB and not result.stderr, "Bounded unit metadata")
    values = dict(line.split("=", 1) for line in result.stdout.decode().splitlines())
    require(set(values) == set(keys), "Complete unit metadata")
    return values


def unit_binding(properties, ready, unit):
    group = properties["ControlGroup"]
    require(group.endswith("/app.slice/" + unit), "Explicit app.slice owned unit")
    require(
        properties["LoadState"] == "loaded"
        and properties["Type"] == "exec"
        and properties["MemoryAccounting"] == "yes"
        and properties["MemoryMax"] == str(CAP)
        and properties["MemoryHigh"] == "infinity"
        and properties["MemorySwapMax"] == "0"
        and properties["OOMPolicy"] == "kill"
        and properties["KillMode"] == "control-group",
        "Exact unit property readback",
    )
    require(
        properties["RuntimeMaxUSec"] == "30min" and properties["TimeoutStopUSec"] == "10s",
        "Exact own wall/stop limit readback",
    )
    require(
        int(properties["MainPID"]) == ready["pid"]
        and process_identity(ready["pid"])["startticks"] == ready["startticks"]
        and ready["boot_id"] == boot_id()
        and cgroup_membership(ready["pid"]) == group,
        "PID/startticks/boot/cgroup binding",
    )
    require(
        len(properties["InvocationID"]) == 32
        and all(c in "0123456789abcdef" for c in properties["InvocationID"]),
        "Exact invocation identity",
    )
    limits = verify_unit_limits(group)
    require(ready["pid"] in limits["own"]["procs"], "Worker inside capped unit")
    return {"InvocationID": properties["InvocationID"], "ControlGroup": group, "limits": limits}


def validate_release(config, ready, release):
    invocation = os.environ.get("INVOCATION_ID", "")
    require(
        len(invocation) == 32
        and all(c in "0123456789abcdef" for c in invocation)
        and release.get("InvocationID") == invocation,
        "Own systemd invocation matches release",
    )
    require(
        all(
            release.get(k) == ready[k]
            for k in ("nonce", "pid", "startticks", "boot_id", "source_sha256", "profile_sha256")
        ),
        "Nonce-bound release identity",
    )
    require(
        ready["nonce"] == config["nonce"]
        and release["ControlGroup"] == cgroup_membership(os.getpid()),
        "Own release cgroup",
    )
    require(
        process_identity(os.getpid())["startticks"] == ready["startticks"]
        and boot_id() == config["boot_id"],
        "Fresh same process release",
    )
    verify_unit_limits(release["ControlGroup"])
    return release


def source_auth(plan):
    inventory = parse(ref_read(plan["research_source_inventory"]))
    original = parse(ref_read(plan["source_closure"]["exact_CFG94_ref"]))
    require(
        original["schema_id"] == "jev_public_cfg_action_source_handoff_v1"
        and len(original["files"]) == 94,
        "Externally pinned original CFG94 authority",
    )
    driver = Path(__file__).resolve().relative_to(ROOT).as_posix()
    require(driver not in original["files"], "Distinct research driver extension")
    expected = {**original["files"], driver: file_pin(Path(__file__))}
    require(
        inventory["schema_id"] == "gb10-memory-research-source-inventory-v1"
        and inventory["files"] == expected
        and inventory["original_CFG94_sha256"]
        == plan["source_closure"]["exact_CFG94_ref"]["sha256"]
        and inventory["identity_sha256"] == sha(encode(expected)),
        "Independent research95 source closure",
    )
    for label, pin in inventory["files"].items():
        require(
            not Path(label).is_absolute() and ".." not in Path(label).parts,
            "Root-relative research source",
        )
        require(file_pin(ROOT / label) == pin, "Research source drift: " + label)
    return inventory


def environment(device_uuid):
    return {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "HF_HUB_CACHE": str(ROOT / "artifacts/jev-local-hf/hub"),
        "PYTHONPATH": ":".join(
            str(ROOT / p) for p in ("artifacts/jev-runtime/overlay", "src", "scripts")
        ),
        "CUDA_VISIBLE_DEVICES": device_uuid,
        "PYTORCH_ALLOC_CONF": "backend:native",
        "PYTORCH_CUDA_ALLOC_CONF": "backend:native",
    }


def native_cache_environment(output):
    root = ordinary(output) / "native-tools"
    return {
        "PATH": "/usr/bin:/bin",
        "CC": "/usr/bin/aarch64-linux-gnu-gcc-13",
        "TMPDIR": str(root / "tmp"),
        "TRITON_HOME": str(root / "triton"),
        "TRITON_CACHE_DIR": str(root / "triton/cache"),
        "TRITON_DUMP_DIR": str(root / "triton/dump"),
        "TRITON_OVERRIDE_DIR": str(root / "triton/override"),
        "TORCHINDUCTOR_CACHE_DIR": str(root / "inductor"),
        "CUDA_CACHE_PATH": str(root / "cuda"),
    }


def python_path_components(paths):
    """Bind specific lexical paths and their parent aliases without directory mtimes."""
    result = {}
    for value in paths:
        path = Path(value)
        require(
            path.is_absolute() and ".." not in path.parts, "Explicit absolute Python identity path"
        )
        for item in (*reversed(path.parents), path):
            observed = item.lstat()
            kind = stat.S_IFMT(observed.st_mode)
            require(
                kind in {stat.S_IFDIR, stat.S_IFLNK, stat.S_IFREG}, "Typed Python path component"
            )
            row = {"mode": observed.st_mode, "device": observed.st_dev, "inode": observed.st_ino}
            if kind != stat.S_IFDIR:
                row.update(size_bytes=observed.st_size, mtime_ns=observed.st_mtime_ns)
            if kind == stat.S_IFLNK:
                row["link"] = os.readlink(item)
            result[str(item)] = row
            require(len(result) <= 128, "Bounded Python alias component population")
    return result


def verify_python_entrypoint(runtime, target):
    binding, tool = runtime["python_entrypoint"], runtime["tools"]["python"]
    require(
        binding["schema_id"] == "native-python-entrypoint-alias-binding-v1",
        "Exact Python alias identity schema",
    )
    require(
        sys.executable == binding["entrypoint"]
        and target == binding["native_file_target"]
        and binding["canonical_executable"] == tool["resolved"],
        "Exact original Python entrypoint and native lexical file target",
    )
    paths = [binding[k] for k in ("entrypoint", "native_file_target", "canonical_executable")]
    require(
        python_path_components(paths) == binding["components"],
        "Python entrypoint/parent alias identity drift",
    )
    require(
        all(str(Path(path).resolve(strict=True)) == tool["resolved"] for path in paths)
        and file_pin(tool["resolved"]) == tool["pin"],
        "Same pinned canonical Python executable behind exact aliases",
    )
    require(
        python_path_components(paths) == binding["components"],
        "Python alias identity changed during executable verification",
    )


def unit_argv(unit, config, device_uuid, log):
    properties = {
        "Type": "exec",
        "MemoryAccounting": "yes",
        "MemoryMax": str(CAP),
        "MemorySwapMax": "0",
        "MemoryHigh": "infinity",
        "OOMPolicy": "kill",
        "KillMode": "control-group",
        "TimeoutStopSec": "10s",
        "RuntimeMaxSec": "1800s",
        "RemainAfterExit": "yes",
        "StandardOutput": "append:" + str(log),
        "StandardError": "append:" + str(log),
    }
    return [
        "systemd-run",
        "--user",
        "--slice=app.slice",
        "--unit=" + unit,
        *["--property=" + k + "=" + v for k, v in properties.items()],
        *[
            "--setenv=" + k + "=" + v
            for k, v in (environment(device_uuid) | native_cache_environment(config.parent)).items()
        ],
        "/home/albert_hsieh/ComfyUI/.venv/bin/python",
        "-B",
        "-s",
        str(Path(__file__).resolve()),
        "--worker",
        str(config),
    ]


def safe_stop(unit, invocation):
    current = service_show(unit)
    require(current["InvocationID"] == invocation, "Refuse stopping replacement invocation")
    subprocess.run(
        ["systemctl", "--user", "stop", unit], capture_output=True, check=True, timeout=20
    )


def process_memory(pid):
    smaps = {}
    for line in read(Path("/proc") / str(pid) / "smaps_rollup").decode().splitlines():
        fields = line.split()
        if fields and fields[0] in {
            "Rss:",
            "Pss:",
            "Private_Clean:",
            "Private_Dirty:",
            "Shared_Clean:",
            "Shared_Dirty:",
        }:
            require(len(fields) == 3 and fields[2] == "kB", "Observed smaps units")
            smaps[fields[0][:-1]] = int(fields[1]) * 1024
    require(len(smaps) == 6, "Complete smaps observation")
    return smaps


def memory_pressure():
    return read("/proc/pressure/memory", 4096).decode()


def phase_cpu():
    import resource

    return {
        "process": process_memory(os.getpid()),
        "ru_maxrss_lifetime_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "memory_PSI_raw": memory_pressure(),
    }


def sample(group, pid):
    host, cg = host_memory(), cgroup_snapshot(group)
    smaps = process_memory(pid)
    return {
        "t": time.monotonic(),
        "available": host["MemAvailable"],
        "total": host["MemTotal"],
        "swap_used": host["SwapTotal"] - host["SwapFree"],
        "current": cg["current"],
        "peak": cg["peak"],
        "swap": cg["swap"],
        "max": cg["max"],
        "high": cg["high"],
        "swap_max": cg["swap_max"],
        "oom_group": cg["oom_group"],
        "events": cg["events"],
        "rss": smaps["Rss"],
        "pss": smaps["Pss"],
        "memory_PSI_raw": memory_pressure(),
    }


def pressure_stop(row, reserve):
    return (
        row["available"] < reserve + 2 * GIB
        or row["current"] >= STOP
        or row["swap"] != 0
        or row.get("max", CAP) != CAP
        or row.get("high") is not None
        or row.get("swap_max", 0) != 0
        or row.get("oom_group", 1) != 1
        or any(row["events"].get(k, 0) for k in ("max", "oom", "oom_kill"))
    )


def heartbeat_observation(path, nonce, *, clock=None):
    heartbeat = parse(read(path))
    checked = (clock or time.monotonic)()
    return {
        "heartbeat_checked_monotonic": checked,
        "heartbeat_reported_monotonic": heartbeat["monotonic"],
        "heartbeat_age_seconds": checked - heartbeat["monotonic"],
        "heartbeat_nonce": heartbeat["nonce"],
        "heartbeat_nonce_matches": heartbeat["nonce"] == nonce,
    }


def allocator_cap(torch, expected_uuid):
    require(
        torch.cuda.device_count() == 1 and torch.cuda.current_device() == 0,
        "One exact visible CUDA device",
    )
    props = torch.cuda.get_device_properties(0)
    actual = str(props.uuid)
    require(
        actual.lower().removeprefix("gpu-") == expected_uuid.lower().removeprefix("gpu-"),
        "Exact selected GPU UUID",
    )
    require(props.name == "NVIDIA GB10", "Declared local GB10 profile only")
    require(torch.cuda.memory.get_allocator_backend() == "native", "Native CUDA allocator required")
    total = int(props.total_memory)
    fraction = float((CAP - 4 * MIB) / total)
    require(0 < fraction <= 1, "40GiB device cap must fit visible total")
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    observed = torch.cuda.get_per_process_memory_fraction(0)
    require(
        type(observed) is float
        and math.isfinite(observed)
        and 0 < observed <= 1
        and math.floor(observed * total) <= CAP,
        "Conservative native allocator fraction readback",
    )
    return {
        "uuid": actual,
        "name": props.name,
        "total_bytes": total,
        "fraction": observed,
        "allowed_bytes_upper": math.floor(observed * total),
        "backend": "native",
    }


def native_dso_name(name):
    prefix, marker, version = name.partition(".so.")
    return name.endswith(".so") or (
        bool(prefix and marker and version) and all(part.isdigit() for part in version.split("."))
    )


def allocator_origins(torch, runtime):
    expected = {
        str(ordinary(spec["path"]))
        for spec in runtime["files"]
        if native_dso_name(Path(spec["path"]).name)
    }
    torch_root = ordinary(runtime["torch_root"])
    torch_c = str(ordinary(runtime["torch_c_module_path"]))
    require(
        Path(torch.__file__).absolute().parent == torch_root
        and str(Path(torch._C.__file__).absolute()) == torch_c
        and Path(torch_c).parent == torch_root
        and torch_c in expected,
        "Foreign Torch._C module origin: " + str(torch._C.__file__),
    )
    torch_lib = torch_root / "lib"
    library_paths = {path for path in expected if Path(path).parent == torch_lib}
    require(expected == library_paths | {torch_c}, "Exact Torch wheel DSO namespace")
    library_names = {Path(path).name for path in library_paths}
    required = set(runtime["required_loaded_paths"])
    require(required <= expected and required, "Explicit required native allocator DSOs")
    mapped = set()
    for line in read(Path("/proc") / str(os.getpid()) / "maps", 8 * MIB).decode().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            path = fields[5]
            # A different package may legitimately have its own Python _C DSO.
            # Only Torch library basenames, its exact _C, and its lib namespace
            # identify Torch ownership; every such origin remains exactly pinned.
            if (
                Path(path).is_relative_to(torch_lib)
                or path == torch_c
                or Path(path).name in library_names
            ):
                require(path in expected, "Foreign native Torch allocator DSO origin: " + path)
                mapped.add(path)
    require(required <= mapped and torch_c in mapped, "Complete loaded native allocator origin set")
    return sorted(mapped)


def uma_host_identity(host, config):
    byte_fields = (
        "MemTotal",
        "MemAvailable",
        "MemFree",
        "Cached",
        "SReclaimable",
        "SwapTotal",
        "SwapFree",
        "Hugepagesize",
        "Hugetlb",
    )
    require(
        all(type(host.get(k)) is int and host[k] >= 0 for k in byte_fields)
        and all(type(host.get(k)) is int for k in ("HugePages_Total", "HugePages_Free")),
        "Complete typed UMA host observation",
    )
    require(
        host["boot_id"] == config["boot_id"]
        and host["MemTotal"] == UMA_TOTAL
        and 0 < host["MemAvailable"] <= UMA_TOTAL
        and host["MemFree"] <= UMA_TOTAL
        and host["SwapFree"] <= host["SwapTotal"]
        and host["Hugepagesize"] > 0
        and host["HugePages_Total"] == host["HugePages_Free"] == host["Hugetlb"] == 0,
        "Explicit same-boot GB10 UMA host with no hugetlb pool",
    )


def uma_uuid(value):
    require(type(value) is str, "Typed device UUID")
    return uuid.UUID(value.lower().removeprefix("gpu-")).hex


def uma_validate_identity(driver, allocator, host, config):
    uma_host_identity(host, config)
    require(
        type(driver.get("integrated")) is int
        and driver["integrated"] == 1
        and driver.get("name") == allocator["name"] == "NVIDIA GB10"
        and type(driver.get("total_bytes")) is int
        and driver["total_bytes"] == allocator["total_bytes"] == host["MemTotal"]
        and uma_uuid(driver["uuid"])
        == uma_uuid(allocator["uuid"])
        == uma_uuid(config["device_uuid"])
        and driver.get("boot_id") == host["boot_id"],
        "Driver integrated/UUID/name/total/boot must match selected Torch GB10",
    )


def uma_driver_metadata(runtime):
    origins = runtime["cuda_link_origins"]
    require(
        len(origins) == 2
        and {x["path"] for x in origins}
        == {
            "/lib/aarch64-linux-gnu/libcuda.so.1",
            "/usr/lib/aarch64-linux-gnu/libcuda.so.1",
        },
        "Exact inherited libcuda metadata aliases",
    )
    for spec in origins:
        target = Path(spec["path"]).resolve(strict=True)
        observed = target.stat()
        require(
            str(target) == spec["resolved"] == UMA_LIBCUDA
            and stat.S_ISREG(observed.st_mode)
            and (observed.st_size, observed.st_mtime_ns, observed.st_dev, observed.st_ino)
            == tuple(spec[k] for k in ("size_bytes", "mtime_ns", "device", "inode")),
            "Inherited canonical libcuda metadata drift",
        )
    require(
        all(
            origins[0][k] == origins[1][k]
            for k in (
                "resolved",
                "size_bytes",
                "mtime_ns",
                "device",
                "inode",
            )
        ),
        "libcuda aliases must identify one inode",
    )
    return {
        k: origins[0][k]
        for k in (
            "resolved",
            "size_bytes",
            "mtime_ns",
            "device",
            "inode",
        )
    }


def uma_symbol_mappings(raw_maps, addresses, metadata):
    require(set(addresses) == set(UMA_SYMBOLS), "Exact Driver API symbol set")
    mappings = []
    for line in raw_maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not Path(fields[5]).name.startswith("libcuda.so"):
            continue
        start, end = (int(x, 16) for x in fields[0].split("-"))
        device = os.makedev(*(int(x, 16) for x in fields[3].split(":")))
        require(
            fields[5] == metadata["resolved"] == UMA_LIBCUDA
            and device == metadata["device"]
            and int(fields[4]) == metadata["inode"],
            "Mapped libcuda path/dev/inode drift",
        )
        if "x" in fields[1]:
            mappings.append((start, end))
    require(
        mappings
        and all(
            type(address) is int
            and address > 0
            and any(start <= address < end for start, end in mappings)
            for address in addresses.values()
        ),
        "Every Driver API symbol must belong to exact executable libcuda mapping",
    )
    return {name: addresses[name] for name in UMA_SYMBOLS}


def uma_driver_read(symbols, ctypes):
    signatures = {
        "cuCtxGetDevice": [ctypes.POINTER(ctypes.c_int)],
        "cuDeviceGetAttribute": [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int],
        "cuDeviceGetUuid": [ctypes.c_void_p, ctypes.c_int],
        "cuDeviceGetName": [ctypes.c_void_p, ctypes.c_int, ctypes.c_int],
        "cuDeviceTotalMem_v2": [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int],
    }
    for name, args in signatures.items():
        symbols[name].argtypes, symbols[name].restype = args, ctypes.c_int

    def call(name, *args):
        result = symbols[name](*args)
        require(type(result) is int and result == 0, "Driver identity query failed: " + name)

    device, integrated, total = ctypes.c_int(-1), ctypes.c_int(-1), ctypes.c_size_t(0)
    identifier, name = (ctypes.c_ubyte * 16)(), ctypes.create_string_buffer(128)
    call("cuCtxGetDevice", ctypes.byref(device))
    require(device.value >= 0, "Current CUDA context device")
    call("cuDeviceGetAttribute", ctypes.byref(integrated), 18, device.value)
    call("cuDeviceGetUuid", identifier, device.value)
    call("cuDeviceGetName", name, len(name), device.value)
    call("cuDeviceTotalMem_v2", ctypes.byref(total), device.value)
    return {
        "context_device": device.value,
        "integrated": integrated.value,
        "uuid": uuid.UUID(bytes=bytes(identifier)).hex,
        "name": name.value.decode("ascii", errors="strict"),
        "total_bytes": total.value,
    }


def uma_driver_identity(runtime, allocator, host, config):
    uma_host_identity(host, config)
    require(
        host["MemAvailable"] >= research_floor(host, startup=True)["required_bytes"],
        "Fresh unchanged startup floor before Driver identity query",
    )
    metadata = uma_driver_metadata(runtime)
    require(
        all(hasattr(os, key) for key in ("RTLD_NOLOAD", "RTLD_NOW", "RTLD_LOCAL")),
        "Explicit already-loaded libcuda handle support required",
    )
    import ctypes

    library = ctypes.CDLL(UMA_LIBCUDA, mode=os.RTLD_NOLOAD | os.RTLD_NOW | os.RTLD_LOCAL)
    symbols = {name: getattr(library, name) for name in UMA_SYMBOLS}
    addresses = {name: ctypes.cast(fn, ctypes.c_void_p).value for name, fn in symbols.items()}
    maps = Path("/proc") / str(os.getpid()) / "maps"
    uma_symbol_mappings(read(maps, 8 * MIB).decode(), addresses, metadata)
    driver = uma_driver_read(symbols, ctypes)
    driver["boot_id"] = host["boot_id"]
    uma_validate_identity(driver, allocator, host, config)
    require(uma_driver_metadata(runtime) == metadata, "Stable libcuda metadata after query")
    uma_symbol_mappings(read(maps, 8 * MIB).decode(), addresses, metadata)
    require(boot_id() == host["boot_id"], "Same boot after Driver query")
    return {
        **driver,
        "profile_kind": UMA_KIND,
        "library_metadata": metadata,
        "symbol_addresses": addresses,
        "load_mode": "exact-canonical-RTLD_NOLOAD",
        "pin_scope": "Inherited runtime02 metadata pins and executable symbol mappings; not a libcuda payload SHA attestation",
    }


def uma_memory_decision(host, cuda, identity, allocator, config):
    uma_validate_identity(identity, allocator, host, config)
    require(
        all(type(cuda[k]) is int and cuda[k] >= 0 for k in ("free", "total", "reserved"))
        and cuda["free"] <= cuda["total"] == UMA_TOTAL,
        "Typed stable raw CUDA counters",
    )
    old_free = cuda["free"] >= CAP + GIB
    native = (
        cuda["backend"] == "native"
        and cuda["fraction"] == allocator["fraction"]
        and cuda["reserved"] < STOP
    )
    return {
        "memory_admission_profile_kind": UMA_KIND,
        "original_cuda_free_required_bytes": CAP + GIB,
        "original_cuda_free_predicate_pass": old_free,
        "original_cuda_free_would_refuse": not old_free,
        "original_native_device_guard_would_refuse": not (old_free and native),
        "UMA_estimate_metric": "proc_MemAvailable",
        "UMA_estimate_bytes": host["MemAvailable"],
        "UMA_required_bytes": CAP + GIB,
        "UMA_predicate_pass": host["MemAvailable"] >= CAP + GIB,
        "native_allocator_predicate_pass": native,
        "identity_sha256": sha(encode(identity)),
        "swap_credit_bytes": 0,
        "semantic_delta": "Explicit UMA estimate removes old raw CUDAfree veto;41GiB estimate is redundant under unchanged higher host floors",
    }


class Research:
    """New research gate; never valid old fitter resource evidence."""

    def __init__(self, torch, output, config, binding, allocator, uma_identity):
        self.torch, self.output, self.config, self.binding, self.allocator = (
            torch,
            output,
            config,
            binding,
            allocator,
        )
        self.uma_identity = uma_identity
        self.loaded = False
        self.sequence = 0
        self.peak_allocated = self.peak_reserved = 0

    def cuda(self):
        t = self.torch
        free, total = t.cuda.mem_get_info(0)
        require(
            type(free) is int and type(total) is int and 0 <= free <= total,
            "Typed raw CUDA free/total query required; no fabricated UMA replacement",
        )
        require(int(total) == self.allocator["total_bytes"], "Stable visible device capacity")
        return {
            "allocated": int(t.cuda.memory_allocated(0)),
            "reserved": int(t.cuda.memory_reserved(0)),
            "peak_allocated": int(t.cuda.max_memory_allocated(0)),
            "peak_reserved": int(t.cuda.max_memory_reserved(0)),
            "free": int(free),
            "total": int(total),
            "stats": dict(t.cuda.memory_stats(0)),
            "backend": t.cuda.memory.get_allocator_backend(),
            "fraction": t.cuda.get_per_process_memory_fraction(0),
        }

    def check(self, phase, scope):
        require(scope in {"cpu", "train", "research"}, "Explicit research protection scope")
        row = {
            "event": "research-gate",
            "phase": phase,
            "scope": scope,
            "memory_admission_profile_kind": UMA_KIND,
            "admitted": False,
            "status": "UNKNOWN",
            "resident_credit_bytes": 0,
            "swap_credit_bytes": 0,
            "original_cuda_free_predicate_pass": None,
            "original_cuda_free_would_refuse": None,
            "UMA_predicate_pass": None,
            "UMA_identity_pass": False,
        }
        try:
            obs = host_memory()
            row["observation"] = obs
            floor = research_floor(obs, startup=not self.loaded)
            row.update(floor)
            cg = cgroup_snapshot(self.binding["ControlGroup"])
            row["cgroup"] = cg
            cuda = self.cuda()
            row["cuda"] = cuda
            if (
                type(cuda.get("free")) is int
                and type(cuda.get("total")) is int
                and 0 <= cuda["free"] <= cuda["total"] == UMA_TOTAL
            ):
                row["original_cuda_free_predicate_pass"] = cuda["free"] >= CAP + GIB
                row["original_cuda_free_would_refuse"] = not row[
                    "original_cuda_free_predicate_pass"
                ]
            row.update(
                uma_memory_decision(obs, cuda, self.uma_identity, self.allocator, self.config)
            )
            row["UMA_identity_pass"] = True
            row["unchanged_research_host_floor_pass"] = (
                obs["boot_id"] == self.config["boot_id"]
                and obs["MemAvailable"] >= floor["required_bytes"]
            )
            row["unchanged_cgroup_pressure_pass"] = (
                cg["max"] == CAP
                and cg["high"] is None
                and cg["swap_max"] == 0
                and cg["oom_group"] == 1
                and not pressure_stop(
                    {"available": obs["MemAvailable"], **cg}, floor["reserve_bytes"]
                )
            )
            row["admitted"] = all(
                row[k]
                for k in (
                    "UMA_predicate_pass",
                    "native_allocator_predicate_pass",
                    "unchanged_research_host_floor_pass",
                    "unchanged_cgroup_pressure_pass",
                )
            )
            row["status"] = "ADMITTED" if row["admitted"] else "REFUSED"
        except BaseException as exc:
            row["error"] = {"type": type(exc).__name__, "message": str(exc)}
            try:
                arm_bound(self.output)
                append(self.output / "phases.jsonl", row, LIMITS["phases.jsonl"])
            except BaseException as evidence_error:
                exc.add_note(
                    "Secondary UMA admission evidence failure: "
                    + type(evidence_error).__name__
                    + ": "
                    + str(evidence_error)
                )
            raise
        arm_bound(self.output)
        append(self.output / "phases.jsonl", row, LIMITS["phases.jsonl"])
        require(row["admitted"], "Explicit UMA research/native/cgroup admission refused")
        return True

    def phase(self, name, fn):
        self.check(name, "research")
        t = self.torch
        t.cuda.synchronize(0)
        previous = self.cuda()
        self.peak_allocated = max(self.peak_allocated, previous["peak_allocated"])
        self.peak_reserved = max(self.peak_reserved, previous["peak_reserved"])
        t.cuda.reset_peak_memory_stats(0)
        sequence, self.sequence = self.sequence, self.sequence + 1
        append(
            self.output / "phases.jsonl",
            {
                "event": "intent",
                "sequence": sequence,
                "phase": name,
                "host": host_memory(),
                "cpu": phase_cpu(),
                "cgroup": cgroup_snapshot(self.binding["ControlGroup"]),
                "cuda": self.cuda(),
            },
            LIMITS["phases.jsonl"],
        )
        started, primary = time.monotonic(), None
        try:
            return fn()
        except BaseException as exc:
            primary = exc
            raise
        finally:
            secondary = []
            try:
                t.cuda.synchronize(0)
            except BaseException as exc:
                secondary.append(
                    {"stage": "CUDA-fence", "type": type(exc).__name__, "message": str(exc)}
                )
                if primary is None:
                    primary = exc
                else:
                    primary.add_note("Secondary CUDA fence: " + str(exc))
            try:
                cuda = self.cuda()
                self.peak_allocated = max(self.peak_allocated, cuda["peak_allocated"])
                self.peak_reserved = max(self.peak_reserved, cuda["peak_reserved"])
                append(
                    self.output / "phases.jsonl",
                    {
                        "event": "complete" if primary is None else "failed",
                        "sequence": sequence,
                        "phase": name,
                        "elapsed_seconds": time.monotonic() - started,
                        "error": None
                        if primary is None
                        else {"type": type(primary).__name__, "message": str(primary)},
                        "secondary_errors": secondary,
                        "host": host_memory(),
                        "cpu": phase_cpu(),
                        "cgroup": cgroup_snapshot(self.binding["ControlGroup"]),
                        "cuda": cuda,
                        "peak_scope": "since this explicit stats reset; maxima retained across fixed post-forward/backward cache release",
                    },
                    LIMITS["phases.jsonl"],
                )
                replace(
                    self.output / "heartbeat.json",
                    {"nonce": self.config["nonce"], "monotonic": time.monotonic(), "phase": name},
                )
            except BaseException as exc:
                if primary is None:
                    primary = exc
                else:
                    primary.add_note("Secondary phase evidence: " + str(exc))
            if primary is not None and sys.exc_info()[0] is None:
                raise primary


def storage_ledger(torch, model, optimizer):
    groups = {
        "parameters": list(model.lm.parameters()),
        "gradients": [p.grad for p in model.lm.parameters() if p.grad is not None],
        "optimizer": [
            v
            for state in optimizer.state.values()
            for v in state.values()
            if isinstance(v, torch.Tensor)
        ],
    }
    result = {}
    all_storages = set()
    for name, tensors in groups.items():
        storages = {
            (str(t.device), int(t.untyped_storage().data_ptr()), int(t.untyped_storage().nbytes()))
            for t in tensors
        }
        result[name] = {
            "unique_storages": len(storages),
            "logical_storage_bytes": sum(k[2] for k in storages),
        }
        all_storages |= storages
    result["all"] = {
        "unique_storages": len(all_storages),
        "logical_storage_bytes": sum(k[2] for k in all_storages),
        "scope": "Storage identity ledger only; not RSS/physical/allocator peak",
    }
    return result


def cache_storage_identity(model):
    groups = {"parameters": [], "gradients": []}
    for name, parameter in model.lm.named_parameters():
        for group, tensor in (("parameters", parameter), ("gradients", parameter.grad)):
            if tensor is None:
                continue
            storage = tensor.untyped_storage()
            groups[group].append(
                [
                    name,
                    id(tensor),
                    str(tensor.device),
                    str(tensor.dtype),
                    list(tensor.shape),
                    list(tensor.stride()),
                    int(tensor.storage_offset()),
                    bool(tensor.requires_grad),
                    int(tensor._version),
                    int(storage.data_ptr()),
                    int(storage.nbytes()),
                ]
            )
    return {
        group: {"tensors": len(rows), "identity_sha256": sha(encode(rows))}
        for group, rows in groups.items()
    }


def native_cache_release(torch, model, gate, output, fn, *, mode, case_id, step, phase):
    require(phase in {"train-forward", "dev-forward", "backward"}, "Fixed native cache boundary")
    result = fn()  # Original profiled native call exactly once; failure never enters release.
    torch.cuda.synchronize(0)
    before = gate.cuda()
    storage_before = cache_storage_identity(model)
    gate.peak_allocated = max(gate.peak_allocated, before["peak_allocated"])
    gate.peak_reserved = max(gate.peak_reserved, before["peak_reserved"])
    row = {
        "policy": CACHE_POLICY,
        "native_phase": phase,
        "mode": mode,
        "case_id": case_id,
        "step": step,
        "phase_sequence": gate.sequence - 1,
        "before": {"cuda": before, "storage_identity": storage_before},
        "after": None,
        "API_returned": False,
        "success": False,
        "error": None,
        "scope": "Storage metadata only, not a gradient-value hash; allocator active_bytes is recorded, not equated to live tensor bytes",
    }
    journal = output / "cache-releases.jsonl"
    append(journal, {"event": "intent", **row}, 2 * MIB)
    primary = None
    try:
        torch.cuda.empty_cache()  # Fixed once, never conditional on a gate refusal.
        row["API_returned"] = True
        after = gate.cuda()
        row["after"] = {"cuda": after, "storage_identity": None}
        gate.peak_allocated = max(gate.peak_allocated, after["peak_allocated"])
        gate.peak_reserved = max(gate.peak_reserved, after["peak_reserved"])
        storage_after = cache_storage_identity(model)
        row["after"]["storage_identity"] = storage_after
        require(
            after["allocated"] == before["allocated"] and storage_after == storage_before,
            "Cache release preserved occupied tensor bytes and parameter/gradient identities",
        )
        require(
            after["peak_allocated"] >= before["peak_allocated"]
            and after["peak_reserved"] >= before["peak_reserved"],
            "Cache release must not reset allocator peaks",
        )
        row["success"] = True
    except BaseException as exc:
        primary = exc
        row["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        try:
            append(journal, {"event": "complete" if row["success"] else "failed", **row}, 2 * MIB)
        except BaseException as exc:
            if primary is None:
                raise
            primary.add_note("Cache release evidence: " + str(exc))
    return result


def backward_cache_release(torch, model, gate, output, fn, *, mode, case_id, step):
    return native_cache_release(
        torch, model, gate, output, fn, mode=mode, case_id=case_id, step=step, phase="backward"
    )


def operator_call(torch, output, phase, fn):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        result = fn()
    ops = [
        {"name": event.key, "count": event.count}
        for event in profiler.key_averages()
        if any(
            k in event.key
            for k in (
                "scaled_dot_product",
                "flash_attention",
                "efficient_attention",
                "cudnn_attention",
            )
        )
    ]
    require(0 < len(ops) <= 64, "Bounded actual SDPA operator evidence")
    append(
        output / "operators.jsonl",
        {
            "phase": phase,
            "operators": ops,
            "profiler": "CPU activity, no shapes/stacks/memory, same original native call; instrumentation overhead is part of this research point",
        },
        MIB,
    )
    return result


def counted_optimizer_step(optimizer, state, output):
    state["optimizer_step_attempts"] += 1
    replace(output / "progress.json", state)
    optimizer.step()
    state["completed_updates"] += 1
    replace(output / "progress.json", state)


def original_decoder_layers(model):
    decoder = getattr(model.backbone, "language_model", model.backbone)
    config = getattr(decoder, "config", None)
    layers = getattr(decoder, "layers", ())
    depth = getattr(config, "num_hidden_layers", None)
    types = getattr(config, "layer_types", None)
    expected = ["sliding_attention"] * 5 + ["full_attention"]
    require(
        type(depth) is int
        and depth == 48
        and len(layers) == depth
        and len({id(layer) for layer in layers}) == depth
        and types == expected * 8,
        "Full original48 decoder layers and pinned attention layout",
    )
    return layers, {
        "decoder_layers": depth,
        "attention_layer_types": list(types),
        "scope": "Decoder stack depth, distinct from88 q/v LoRA target modules and176 A/B tensors",
    }


def diagnostic_steps(
    torch, model, rows, plans, *, mode, gate, kernel, fit_kernel, output, state, selected
):
    parameters = [
        (name, value) for name, value in model.lm.named_parameters() if value.requires_grad
    ]
    require(
        len(parameters) == 176 and sum(p.numel() for _, p in parameters) == 5193728,
        "Exact original trainables",
    )
    frozen = {
        name: (value, value._version)
        for name, value in model.lm.named_parameters()
        if not value.requires_grad
    }
    optimizer = gate.phase(
        "optimizer-create-empty",
        lambda: torch.optim.AdamW([p for _, p in parameters], lr=5e-5, weight_decay=0.0),
    )
    require(len(optimizer.state) == 0, "Adam moments must start lazy")
    trace = {"native_forward_calls": [], "layers": {}}
    active = {"phase": "setup", "case_id": None}

    def forward(_module, _args, kwargs):
        gate.check("official-native-forward-entry", "train")
        trace["native_forward_calls"].append(
            {
                **active,
                "input_shape": list(kwargs["input_ids"].shape),
                "positions": kwargs["logits_to_keep"].tolist(),
                "use_cache": kwargs.get("use_cache"),
            }
        )

    def layer(index):
        def entered(_module, _args):
            key = active["phase"] + ":" + str(index)
            trace["layers"][key] = trace["layers"].get(key, 0) + 1

        return entered

    layers, architecture = original_decoder_layers(model)
    state["decoder_architecture"] = trace["decoder_architecture"] = architecture
    replace(output / "progress.json", state)
    hooks = [model.lm.register_forward_pre_hook(forward, with_kwargs=True)]
    hooks += [module.register_forward_pre_hook(layer(i)) for i, module in enumerate(layers)]
    try:
        model.lm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.lm.train()
        train = [
            rows["train"][r["zero_based_line_ordinal"]] for r in reversed(selected["train_rows"])
        ]
        order = list(range(2))
        random.Random(42).shuffle(order)
        torch.manual_seed(42)
        state["actual_train_order"] = [train[i]["record"]["case_id"] for i in order]
        for index in order:
            row = train[index]
            active.update(phase="forward", case_id=row["record"]["case_id"])
            gate.phase("zero-grad-before-forward", lambda: optimizer.zero_grad(set_to_none=True))
            before = len(trace["native_forward_calls"])
            loss, metrics = gate.phase(
                "native-train-forward",
                lambda: native_cache_release(
                    torch,
                    model,
                    gate,
                    output,
                    lambda: operator_call(
                        torch,
                        output,
                        active["phase"],
                        lambda: fit_kernel.action_loss(
                            model, row, plan=plans[row["record"]["case_id"]], mode=mode
                        ),
                    ),
                    mode=mode,
                    case_id=row["record"]["case_id"],
                    step=state["completed_updates"] + 1,
                    phase="train-forward",
                ),
            )
            require(
                len(trace["native_forward_calls"]) == before + 1,
                "One original causal forward per row",
            )
            active["phase"] = "backward"
            backward = loss.backward
            gate.phase(
                "native-backward",
                lambda: backward_cache_release(
                    torch,
                    model,
                    gate,
                    output,
                    lambda: operator_call(torch, output, active["phase"], backward),
                    mode=mode,
                    case_id=row["record"]["case_id"],
                    step=state["completed_updates"] + 1,
                ),
            )
            norm = gate.phase(
                "native-grad-clip",
                lambda: torch.nn.utils.clip_grad_norm_(
                    [p for _, p in parameters], 1.0, error_if_nonfinite=True
                ),
            )

            gate.phase("native-Adam-step", lambda: counted_optimizer_step(optimizer, state, output))
            state["synchronized_updates"] += 1
            replace(output / "progress.json", state)
            require(
                all(torch.isfinite(p).all() for _, p in parameters), "Finite updated original176"
            )
            require(
                all(v.grad is None for v, _ in frozen.values())
                and not any(p in optimizer.state for p, _ in frozen.values()),
                "No frozen gradients or Adam states",
            )
            append(
                output / "updates.jsonl",
                {
                    "step": state["completed_updates"],
                    "case_id": row["record"]["case_id"],
                    "loss": float(loss.detach()),
                    "gradient_norm": float(norm),
                    "metrics": metrics,
                    "storage": storage_ledger(torch, model, optimizer),
                },
                MIB,
            )
            replace(output / "progress.json", state)
            del loss
        model.lm.eval()
        model.lm.gradient_checkpointing_disable()
        with torch.no_grad():
            for descriptor in selected["development_rows"]:
                row = rows["development"][descriptor["zero_based_line_ordinal"]]
                active.update(phase="development", case_id=row["record"]["case_id"])
                loss, metrics = gate.phase(
                    "native-development-forward",
                    lambda: native_cache_release(
                        torch,
                        model,
                        gate,
                        output,
                        lambda: operator_call(
                            torch,
                            output,
                            active["phase"],
                            lambda: fit_kernel.action_loss(
                                model, row, plan=plans[row["record"]["case_id"]], mode=mode
                            ),
                        ),
                        mode=mode,
                        case_id=row["record"]["case_id"],
                        step=state["completed_updates"],
                        phase="dev-forward",
                    ),
                )
                append(
                    output / "development.jsonl",
                    {
                        "case_id": row["record"]["case_id"],
                        "loss": float(loss),
                        "metrics": metrics,
                        "storage": storage_ledger(torch, model, optimizer),
                    },
                    MIB,
                )
                del loss
        require(
            state["completed_updates"] == 2 and len(trace["native_forward_calls"]) == 4,
            "Exactly2 diagnostic steps and2 eval forwards",
        )
        require(
            all(v._version == version and not v.requires_grad for v, version in frozen.values()),
            "Frozen base versions unchanged",
        )
        gate.phase(
            "final-full-head-hash",
            lambda: require(
                kernel._tensor_sha(model.head.weight)
                == "9619fd670a4074367c72f2cf09d777970ad1eafa511bff605a82c8558dcc24d4",
                "Unchanged full head",
            ),
        )
        write(output / "native-call-trace.json", trace)
        return storage_ledger(torch, model, optimizer)
    finally:
        for hook in hooks:
            hook.remove()


def selected_rows(plan):
    selected = parse(ref_read(plan["evidence"]["fixed_inputs"]))["selected"]
    require(
        selected["train"] == plan["fixed_workload"]["train_rows"]
        and selected["development"] == plan["fixed_workload"]["development_rows"],
        "Original selected inputs unchanged",
    )
    for split in ("train", "development"):
        for descriptor in selected[split]:
            blob = ref_read(descriptor["source"])
            raw = blob.splitlines()[descriptor["zero_based_line_ordinal"]]
            value = parse(raw)
            require(
                sha(raw) == descriptor["raw_line_without_LF_sha256"]
                and value["record"]["case_id"] == descriptor["case_id"],
                "Exact predeclared native row bytes",
            )
    return selected


def heartbeat(output, config, state, stop, errors):
    while not stop.is_set():
        try:
            replace(
                output / "heartbeat.json",
                {
                    "nonce": config["nonce"],
                    "monotonic": time.monotonic(),
                    "phase": state.get("stage", "startup"),
                },
            )
        except BaseException as exc:
            errors.append(exc)
            return
        stop.wait(0.5)


def native_work_identity(ready, binding):
    return {
        **ready,
        "InvocationID": binding["InvocationID"],
        "ControlGroup": binding["ControlGroup"],
    }


def native_work_live(identity):
    require(
        process_identity(identity["pid"])["startticks"] == identity["startticks"]
        and cgroup_membership(identity["pid"]) == identity["ControlGroup"]
        and boot_id() == identity["boot_id"],
        "Same live native-work process/cgroup/boot",
    )


def validate_native_work_capture(capture, identity):
    require(
        set(capture) == {"schema_id", "identity", "checked_monotonic", "properties", "cgroup"}
        and capture["schema_id"] == "native-work-end-cgroup-v1"
        and capture["identity"] == identity
        and type(capture["checked_monotonic"]) in {float, int}
        and math.isfinite(capture["checked_monotonic"])
        and capture["checked_monotonic"] >= 0,
        "Exact native-work-end capture identity/schema",
    )
    props, cg = capture["properties"], capture["cgroup"]
    require(
        props["InvocationID"] == identity["InvocationID"]
        and props["ControlGroup"] == identity["ControlGroup"]
        and props["MainPID"] == str(identity["pid"])
        and not terminal(props),
        "Native-work capture belongs to live same invocation",
    )
    require(
        cg["group"] == identity["ControlGroup"]
        and all(
            type(cg[k]) is int and cg[k] >= 0
            for k in ("current", "peak", "max", "swap", "swap_max", "oom_group")
        )
        and 0 < cg["current"] <= cg["peak"] < STOP
        and cg["max"] == CAP
        and cg["high"] is None
        and cg["swap"] == cg["swap_max"] == 0
        and cg["oom_group"] == 1
        and all(type(pid) is int and pid > 0 for pid in cg["procs"])
        and identity["pid"] in cg["procs"]
        and {"max", "oom", "oom_kill"} <= set(cg["events"])
        and all(type(v) is int and v >= 0 for v in cg["events"].values())
        and all(cg["events"][k] == 0 for k in ("max", "oom", "oom_kill"))
        and bool(cg["stat"])
        and all(type(v) is int and v >= 0 for v in cg["stat"].values()),
        "Complete native-work-end kernel counters/caps",
    )


def capture_native_work_end(output, ready, binding, unit):
    identity = native_work_identity(ready, binding)
    marker = parse(read(output / "native-work-ready.json"))
    require(
        marker == {"schema_id": "native-work-ready-v1", **identity},
        "Exact native-work ready identity/schema",
    )
    native_work_live(identity)
    props = service_show(unit)
    capture = {
        "schema_id": "native-work-end-cgroup-v1",
        "identity": identity,
        "checked_monotonic": time.monotonic(),
        "properties": props,
        "cgroup": cgroup_snapshot(identity["ControlGroup"]),
    }
    validate_native_work_capture(capture, identity)
    write(output / "native-work-final-cgroup.json", capture)
    native_work_live(identity)
    after = service_show(unit)
    require(
        after["InvocationID"] == identity["InvocationID"]
        and after["ControlGroup"] == identity["ControlGroup"]
        and after["MainPID"] == str(identity["pid"])
        and not terminal(after),
        "Fresh native-work invocation after snapshot IO",
    )
    digest = file_pin(output / "native-work-final-cgroup.json")["sha256"]
    # The ACK intentionally releases exit. No live-PID claim is made after it.
    write(
        output / "native-work-ack.json",
        {"schema_id": "native-work-ack-v1", **identity, "snapshot_sha256": digest},
    )
    return capture


def wait_native_work_capture(output, ready, binding, errors):
    identity = native_work_identity(ready, binding)
    require(
        os.getpid() == identity["pid"]
        and os.environ.get("INVOCATION_ID") == identity["InvocationID"],
        "Own final-barrier invocation",
    )
    native_work_live(identity)
    write(output / "native-work-ready.json", {"schema_id": "native-work-ready-v1", **identity})
    native_work_live(identity)
    deadline = time.monotonic() + 10
    while not (output / "native-work-ack.json").exists():
        require(
            time.monotonic() < deadline and not errors,
            "Bounded native-work-end acknowledgement/heartbeat",
        )
        time.sleep(0.02)
    require(time.monotonic() < deadline and not errors, "Timely native-work-end acknowledgement")
    native_work_live(identity)
    ack = parse(read(output / "native-work-ack.json"))
    blob = read(output / "native-work-final-cgroup.json")
    require(
        ack == {"schema_id": "native-work-ack-v1", **identity, "snapshot_sha256": sha(blob)},
        "Exact native-work ACK identity/schema/snapshot",
    )
    validate_native_work_capture(parse(blob), identity)
    native_work_live(identity)
    return {
        "snapshot_sha256": sha(blob),
        "scope": "Native-work end before exit ACK; events are not post-ACK/lifetime counters",
    }


def retained_lifetime_peaks(properties, identity, capture):
    validate_native_work_capture(capture, identity)
    require(
        properties["InvocationID"] == identity["InvocationID"] and terminal(properties),
        "Same terminal invocation for retained lifetime peaks",
    )
    values = {}
    for key in ("MemoryPeak", "MemorySwapPeak"):
        raw = properties.get(key)
        require(
            type(raw) is str and raw.isascii() and raw.isdigit(),
            "Measured retained systemd lifetime peak: " + key,
        )
        values[key] = int(raw)
    require(
        0 < capture["cgroup"]["peak"] <= values["MemoryPeak"] < STOP
        and values["MemorySwapPeak"] == 0,
        "Consistent retained lifetime memory/swap peaks",
    )
    return {
        **values,
        "InvocationID": identity["InvocationID"],
        "source": "Same-invocation retained systemd properties after process exit",
        "scope": "Lifetime peaks only; no inference of missing post-ACK kernel events/stat",
    }


def require_unquantized_modules(lm):
    modules, parameters = 0, 0
    for name, module in lm.named_modules():
        modules += 1
        origin = type(module).__module__
        require(
            not origin.startswith(("bitsandbytes.", "peft.tuners.lora.bnb")),
            "Quantized module class: " + name + ":" + origin,
        )
        for flag in ("is_quantized", "is_loaded_in_4bit", "is_loaded_in_8bit"):
            value = getattr(module, flag, None)
            require(value is None or value is False, "Quantization flag: " + name + ":" + flag)
    for name, parameter in lm.named_parameters():
        parameters += 1
        require(
            not type(parameter).__module__.startswith("bitsandbytes."),
            "Quantized parameter class: " + name,
        )
    return {
        "modules_checked": modules,
        "parameters_checked": parameters,
        "quantized_modules": 0,
        "bnb_parameter_instances": 0,
        "flags": ["is_quantized", "is_loaded_in_4bit", "is_loaded_in_8bit"],
    }


def verify_optional_dependency_files(runtime):
    require(
        runtime["schema_id"] == "nonquantized-peft-bnb-runtime-pins-v1",
        "Explicit nonquantized dependency runtime profile",
    )
    require(
        os.environ.get("BNB_CUDA_VERSION") is None and os.environ.get("BNB_ROCM_VERSION") is None,
        "No bitsandbytes version override",
    )
    for spec in runtime["files"]:
        require(
            file_pin(spec["path"]) == spec["pin"],
            "Exact optional dependency runtime: " + spec["path"],
        )


def optional_dependency_origins(runtime):
    origins = {}
    for name, expected in runtime["module_paths"].items():
        module = sys.modules.get(name)
        require(
            module is not None and str(Path(module.__file__).absolute()) == expected,
            "Optional dependency module origin: " + name,
        )
        origins[name] = expected
    cext = sys.modules["bitsandbytes.cextension"]
    library = getattr(cext.lib, "_lib", None)
    selected = getattr(library, "_name", None)
    expected = set(runtime["DSO_paths"])
    require(
        type(selected) is str and selected in expected,
        "Known actual bitsandbytes selected DLL required: " + str(selected),
    )
    mapped = set()
    for line in read(Path("/proc") / str(os.getpid()) / "maps", 8 * MIB).decode().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            path = fields[5]
            if Path(path).name.startswith("libbitsandbytes_"):
                require(path in expected, "Foreign bitsandbytes DLL origin: " + path)
                mapped.add(path)
    require(selected in mapped, "Selected bitsandbytes DLL not actually mapped")
    return {
        "modules": origins,
        "selected_DLL": selected,
        "mapped_DLLs": sorted(mapped),
        "scope": "Dependency provenance only; no quantized model/kernel use authorized",
    }


def require_no_forbidden_modules():
    loaded = sorted(name for name in sys.modules if name.split(".")[0] in {"modal", "playwright"})
    require(not loaded, "Forbidden loaded worker modules: " + ",".join(loaded))


class NativeTools:
    """Exact trusted JIT entry authorization; never a descendant OS sandbox claim."""

    def __init__(self, output, binding, *, preflight_source=None):
        self.output = ordinary(output)
        self.binding = binding
        self.preflight_source = preflight_source
        self.runtime = None
        self.authorizations = 0

    def prepare(self, runtime):
        require(
            runtime["schema_id"] == "gb10-native-jit-toolchain-runtime-pins-v2",
            "Exact native JIT runtime schema",
        )
        require(
            file_pin(runtime["record_authority"]["path"]) == runtime["record_authority"]["pin"],
            "Native JIT wheel RECORD drift",
        )
        for path, expected in runtime["files"].items():
            require(file_pin(path) == expected, "Native JIT source/tool drift: " + path)
        for spec in runtime["tools"].values():
            require(
                str(Path(spec["path"]).resolve(strict=True)) == spec["resolved"]
                and runtime["files"][spec["resolved"]] == spec["pin"],
                "Native JIT executable resolution drift",
            )
        for spec in runtime["cuda_link_origins"]:
            target = Path(spec["path"]).resolve(strict=True)
            observed = target.stat()
            require(
                str(target) == spec["resolved"]
                and (observed.st_size, observed.st_mtime_ns, observed.st_dev, observed.st_ino)
                == tuple(spec[k] for k in ("size_bytes", "mtime_ns", "device", "inode")),
                "Native JIT CUDA link origin drift",
            )
        if self.preflight_source is not None:
            require(
                file_pin(self.preflight_source["path"]) == self.preflight_source["pin"],
                "Reviewed CPU-only toolchain preflight source drift",
            )
        self.check_environment()
        import tempfile

        require(
            tempfile.gettempdir() == self.environment["TMPDIR"],
            "Native tempfile root must be owned before framework imports",
        )
        verify_python_entrypoint(runtime, runtime["python_entrypoint"]["native_file_target"])
        self.runtime = runtime

    def check_environment(self):
        expected = native_cache_environment(self.output)
        require(
            all(os.environ.get(k) == v for k, v in expected.items()),
            "Exact owned JIT cache/compiler environment",
        )
        forbidden = {
            "LD_PRELOAD",
            "LD_AUDIT",
            "LD_LIBRARY_PATH",
            "CPATH",
            "C_INCLUDE_PATH",
            "CPLUS_INCLUDE_PATH",
            "LIBRARY_PATH",
            "GCC_EXEC_PREFIX",
            "COMPILER_PATH",
            "GCC_SPECS",
            "CFLAGS",
            "CPPFLAGS",
            "LDFLAGS",
            "PTXAS_OPTIONS",
            "DISABLE_PTXAS_OPT",
            "NVPTX_ENABLE_DUMP",
            "USE_IR_LOC",
        }
        require(
            not any(os.environ.get(k) for k in forbidden),
            "No native compiler/loader option injection",
        )
        require(
            not any(
                (k.startswith("TRITON_") or k.startswith("TORCHINDUCTOR_")) and k not in expected
                for k in os.environ
            ),
            "No unreviewed JIT implementation/cache/architecture overrides",
        )
        self.environment = expected

    def callers(self):
        sources = set(self.runtime["sources"].values())
        if self.preflight_source:
            sources.add(self.preflight_source["path"])
        result = set()
        frame = sys._getframe(1)
        for _ in range(64):
            if frame is None:
                break
            filename = str(Path(frame.f_code.co_filename).absolute())
            if filename in sources:
                module = sys.modules.get(frame.f_globals.get("__name__"))
                require(
                    module is not None and str(Path(module.__file__).absolute()) == filename,
                    "Native subprocess source/module origin mismatch",
                )
                result.add((filename, frame.f_code.co_name))
            frame = frame.f_back
        return result

    def classify(self, args, callers):
        require(self.runtime is not None, "Native toolchain must be verified before any subprocess")
        require(len(args) == 4, "Exact native Popen audit schema")
        executable, argv, cwd, env = args
        require(
            type(executable) is str
            and type(argv) in (list, tuple)
            and all(type(x) is str for x in argv)
            and cwd is None,
            "Native subprocess requires typed argv and inherited cwd",
        )
        self.check_environment()
        tools, sources = self.runtime["tools"], self.runtime["sources"]

        def caller(source, name, preflight):
            return (sources[source], name) in callers or (
                self.preflight_source is not None
                and (self.preflight_source["path"], "_probe_" + preflight) in callers
            )

        def within(path):
            path = ordinary(path)
            require(
                path.is_relative_to(Path(self.environment["TMPDIR"])),
                "Native tool path must remain in owned tmp",
            )
            return path

        def inherited():
            require(env is None, "Native compiler must inherit the checked worker environment")

        kind = None
        if executable == tools["ldconfig"]["path"] and list(argv) == [executable, "-p"]:
            require(
                caller("driver", "libcuda_dirs", "ldconfig")
                or caller("ctypes", "_findSoname_ldconfig", "ldconfig"),
                "Only original read-only CUDA/library discovery",
            )
            require(
                env is None or env == {"LC_ALL": "C", "LANG": "C"}, "Exact ldconfig environment"
            )
            kind = "ldconfig"
        elif executable == "file" and len(argv) == 3 and list(argv[:2]) == ["file", "-b"]:
            verify_python_entrypoint(self.runtime, argv[2])
            require(
                caller("platform", "_syscmd_file", "file")
                and (
                    (sources["build"], "platform_key") in callers
                    or self.preflight_source is not None
                ),
                "Only original native platform-key file probe",
            )
            require(env == dict(os.environ, LC_ALL="C"), "Exact platform file environment")
            kind = "file"
        elif executable == tools["cc"]["path"]:
            inherited()
            require(caller("build", "_build", "cc"), "Only original native C extension compiler")
            require(len(argv) >= 8, "Complete native C compiler argv")
            src, dst = within(argv[1]), within(argv[7])
            require(
                list(argv[2:7]) == ["-O3", "-shared", "-fPIC", "-Wno-psabi", "-o"],
                "Exact original C compiler flags",
            )
            require(
                src.parent == dst.parent and not dst.exists(),
                "Fresh same-directory native extension output",
            )
            preflight = (
                self.preflight_source is not None
                and (self.preflight_source["path"], "_probe_cc") in callers
            )
            if preflight:
                require(
                    src.name == "public_toolchain_probe.c"
                    and dst.name == "public_toolchain_probe.so"
                    and len(argv) == 8
                    and read(src, MIB) == b"int public_toolchain_probe(void) { return 7; }\n",
                    "Fixed benign CPU preflight C source",
                )
            else:
                require(
                    src.name == "cuda_utils.c"
                    and dst.name == "cuda_utils" + self.runtime["extension_suffix"]
                    and file_pin(src) == self.runtime["files"][sources["cuda_utils_source"]],
                    "Exact original CUDA utility C source/output",
                )
                triton = Path(self.runtime["triton_root"])
                prefix = ["-l:libcuda.so.1", "-L" + str(triton / "backends/nvidia/lib")]
                suffix = [
                    "-I" + str(triton / "backends/nvidia/include"),
                    "-I" + str(src.parent),
                    "-I" + self.runtime["python_include"],
                ]
                remaining = list(argv[8:])
                library_flags = remaining[len(prefix) : -len(suffix)]
                require(
                    remaining[: len(prefix)] == prefix
                    and remaining[-len(suffix) :] == suffix
                    and 1 <= len(library_flags) <= 4
                    and all(
                        flag in {"-L" + path for path in self.runtime["cuda_library_directories"]}
                        for flag in library_flags
                    ),
                    "Exact original CUDA compiler includes/libraries; no plugins or extra flags",
                )
            kind = "cc"
        else:
            names = [
                name for name in ("ptxas", "ptxas_blackwell") if executable == tools[name]["path"]
            ]
            require(len(names) == 1, "Unrecognized native subprocess executable")
            tool_name = names[0]
            inherited()
            if list(argv) == [executable, "--version"]:
                require(
                    caller("knobs", "from_path", "ptxas_version")
                    or caller("compiler", "get_ptxas_version", "ptxas_version"),
                    "Only original ptxas version discovery",
                )
                kind = tool_name + "_version"
            else:
                require(
                    caller("compiler", "make_cubin", "ptxas"), "Only original native PTX assembler"
                )
                require(len(argv) in (8, 9), "Exact native PTX assembler argument count")
                src, dst = within(argv[-3]), within(argv[-1])
                require(
                    src.suffix == ".ptx"
                    and str(dst) == str(src) + ".o"
                    and argv[-2] == "-o"
                    and not dst.exists(),
                    "Native PTX input and same-name fresh cubin output",
                )
                head = [executable, "-lineinfo"]
                if len(argv) == 9:
                    head += ["--fmad=false"]
                arch = argv[-4]
                require(
                    arch.startswith("--gpu-name=sm_")
                    and arch.removeprefix("--gpu-name=sm_").removesuffix("a").isdigit(),
                    "Original native GPU architecture flag",
                )
                capability = int(arch.removeprefix("--gpu-name=sm_").removesuffix("a"))
                require(
                    tool_name == ("ptxas_blackwell" if capability >= 100 else "ptxas"),
                    "Original architecture-dependent ptxas selection",
                )
                require(
                    list(argv[:-4]) == head + ["-v", "--regAllocOptLevel=2"],
                    "Exact original ptxas flags; no custom options",
                )
                require(
                    src.is_file() and src.stat().st_size <= 12 * MIB,
                    "Bounded ordinary native PTX source",
                )
                kind = tool_name
        selected = (
            "file" if kind == "file" else ("cc" if kind == "cc" else kind.removesuffix("_version"))
        )
        require(
            file_pin(tools[selected]["resolved"]) == tools[selected]["pin"],
            "Selected native executable changed",
        )
        return {
            "kind": kind,
            "argv": list(argv),
            "executable": tools[selected]["resolved"],
            "cwd": None,
            "environment_values_recorded": False,
        }

    def audit(self, args):
        require(self.runtime is not None, "Native toolchain not yet admitted")
        require(self.authorizations < 1024, "Bounded native tool invocation count")
        require(
            cgroup_membership(os.getpid()) == self.binding["ControlGroup"],
            "Native tool entry remains in original owned cgroup",
        )
        row = self.classify(args, self.callers())
        origins = self.origins()
        arm_bound(self.output)
        next_sequence = self.authorizations + 1
        append(
            self.output / "native-toolchain-events.jsonl",
            {
                "sequence": next_sequence,
                "status": "authorized_entry_not_completion_or_delivery",
                "scope": "Python Popen entry only; native descendants inherit the original owned unit/caps, not an OS network/escape guarantee",
                "loaded_triton_module_origin_count": len(origins),
                "loaded_triton_module_origins_sha256": sha(encode(origins)),
                **row,
            },
            2 * MIB,
        )
        self.authorizations = next_sequence

    def origins(self):
        from types import ModuleType

        require(self.runtime is not None, "Verified native runtime before loaded-origin check")
        origins = {}
        for name, module in list(sys.modules.items()):
            if name == "triton" or name.startswith("triton."):
                path = getattr(module, "__file__", None)
                if path is None:
                    native_root = "triton._C.libtriton"
                    if name.startswith(native_root + "."):
                        owner = sys.modules.get(native_root)
                        expected = str(Path(self.runtime["triton_root"]) / "_C/libtriton.so")
                        require(
                            isinstance(owner, ModuleType)
                            and getattr(owner, "__file__", None) == expected
                            and expected in self.runtime["files"],
                            "Pinned libtriton owner for native builtin submodule",
                        )
                        for part in name.removeprefix(native_root + ".").split("."):
                            owner = getattr(owner, part, None)
                            require(
                                isinstance(owner, ModuleType),
                                "Typed native builtin submodule owner",
                            )
                        require(
                            owner is module, "Native builtin submodule parent-attribute identity"
                        )
                        origins[name] = {"native_root": expected, "module_attribute": name}
                        continue
                    require(
                        getattr(module, "__path__", None)
                        and all(
                            Path(p).is_relative_to(self.runtime["triton_root"])
                            for p in module.__path__
                        ),
                        "Exact native Triton namespace origin: " + name,
                    )
                    continue
                require(
                    str(Path(path).absolute()) in self.runtime["files"],
                    "Foreign native Triton module origin: " + name,
                )
                origins[name] = str(Path(path).absolute())
        return origins


def worker_audit(event, args, *, tools=None):
    if event == "import" and args[0].split(".")[0] in {"modal", "playwright"}:
        fail("Research worker forbids cloud/browser packages")
    if event == "subprocess.Popen" and tools is not None:
        tools.audit(args)
        return
    if event in {
        "socket.connect",
        "socket.bind",
        "socket.getaddrinfo",
        "socket.sendto",
        "subprocess.Popen",
        "os.system",
    }:
        fail("Research worker forbids network/process work")


def finish_worker(output, state, *, primary, cleanup_errors):
    if cleanup_errors:
        state["secondary_errors"] = [
            {"type": type(e).__name__, "message": str(e)} for e in cleanup_errors
        ]
        state["failed_stage"] = state.get("failed_stage", state["stage"])
        state["stage"] = "failed_memory_point"
        if primary is None:
            state["error"] = state["secondary_errors"][0]
        else:
            for error in cleanup_errors:
                primary.add_note("Cleanup/evidence: " + str(error))
    try:
        replace(output / "worker-receipt.json", state)
    except BaseException as exc:
        if primary is not None:
            primary.add_note("Worker receipt finalization: " + str(exc))
            return
        if cleanup_errors:
            cleanup_errors[0].add_note("Worker receipt finalization: " + str(exc))
            raise cleanup_errors[0]
        raise
    if primary is None and cleanup_errors:
        raise cleanup_errors[0]


def worker(config_path):
    config = parse(read(config_path))
    output = ordinary(config_path).parent
    require(
        set(config)
        == {
            "mode",
            "nonce",
            "boot_id",
            "source_pin",
            "profile_pin",
            "campaign_output",
            "device_uuid",
        },
        "Exact parent config schema",
    )
    require(
        output.parent == ordinary(config["campaign_output"])
        and output.parent.parent == BASE
        and output.name == config["mode"]
        and config["mode"] in ("A", "C"),
        "Parent-owned exact arm output",
    )
    require(
        file_pin(Path(__file__)) == config["source_pin"]
        and file_pin(PLAN) == config["profile_pin"],
        "Pinned worker/profile",
    )
    plan = parse(read(PLAN))
    source_auth(plan)
    selected = selected_rows(plan)
    identity = process_identity(os.getpid())
    ready = {
        "nonce": config["nonce"],
        "pid": os.getpid(),
        "startticks": identity["startticks"],
        "boot_id": boot_id(),
        "source_sha256": config["source_pin"]["sha256"],
        "profile_sha256": config["profile_pin"]["sha256"],
    }
    require(ready["boot_id"] == config["boot_id"], "Same boot before worker release")
    state = {
        "kind": UMA_KIND,
        "qualified": False,
        "stage": "pre-framework-barrier",
        "mode": config["mode"],
        "allocator_policy": CACHE_POLICY,
        "model_constructor_attempts": 0,
        "model_loads": 0,
        "verified_model_loads": 0,
        "optimizer_step_attempts": 0,
        "completed_updates": 0,
        "synchronized_updates": 0,
        "counter_scope": {
            "attempts": "Durable intent before native call; an attempted call may have partial effects",
            "model_loads": "Constructor returned; verified_model_loads additionally passed synchronized full model/head checks",
            "completed_updates": "optimizer.step returned; synchronized_updates additionally passed the enclosing CUDA fence and phase evidence",
            "partial": "On failure or missing final receipt, durable returned/synchronized counts are lower bounds; unrecorded completion or partial native effects remain unknown, never inferred zero",
        },
        "development_updates": 0,
        "new_trained_weight_files": 0,
        "error": None,
    }
    stop, errors = threading.Event(), []
    ticker = threading.Thread(
        target=heartbeat, args=(output, config, state, stop, errors), daemon=True
    )
    ticker.start()
    model = None
    tools = None
    primary = None
    started = time.monotonic()
    try:
        write(output / "ready-for-readback.json", ready)
        deadline = time.monotonic() + 60
        while not (output / "release.json").exists():
            require(
                time.monotonic() < deadline and not errors,
                "Controller release barrier timed out/failed",
            )
            time.sleep(0.02)
        binding = validate_release(config, ready, parse(read(output / "release.json")))
        state["unit_binding"] = binding
        require_no_forbidden_modules()
        cache_root = output / "native-tools"
        require(not cache_root.exists(), "Fresh per-arm native compiler/cache root")
        cache_root.mkdir(mode=0o700)
        for key, value in native_cache_environment(output).items():
            if key not in {"CC", "PATH"}:
                Path(value).mkdir(mode=0o700, parents=True, exist_ok=True)
        tools = NativeTools(output, binding)
        sys.addaudithook(lambda event, arguments: worker_audit(event, arguments, tools=tools))
        sys.path.insert(0, str(ROOT / "scripts"))
        import jev_browser_fit_public_cfg_actions as producer
        import jev_public_cfg_action_artifact as artifact

        producer._fresh_environment()
        old = producer.ResourceGate(output / "original-startup-gates.jsonl", config["boot_id"])
        old.check("before-any-project-framework-or-parent-payload", "cpu")
        state["stage"] = "native-toolchain-verification"
        tools.prepare(parse(ref_read(plan["native_toolchain_runtime_pins"])))
        state["native_toolchain"] = {
            "runtime": plan["native_toolchain_runtime_pins"],
            "fresh_cache_environment": native_cache_environment(output),
            "entry_evidence_scope": "Authorization only, not process completion; native child descendants are trusted normal toolchain under inherited owned unit/caps, not a universal OS isolation claim",
        }
        state["stage"] = "public-admission"
        source94 = ref_read(plan["source_closure"]["exact_CFG94_ref"])
        inputs = artifact.verify_public_fit_inputs(
            **producer.default_paths(),
            source_handoff_bytes=source94,
            expected_source_handoff=sha(source94),
            expected_admission=artifact.FIXED_METADATA["admission-receipt.json"],
            expected_preflight=artifact.FIXED_METADATA["preflight-root-receipt.json"],
            tokenizer_cache=producer.CACHE,
            gate=old,
        )
        state["versions"] = artifact.verify_reference_environment(inputs.reference_items)
        old.check("immediately-before-legacy-parent-capture-and-Torch", "cpu")
        api = artifact.trusted_module(
            "s1.browser_action_adapter", "src/s1/browser_action_adapter.py"
        )
        arm_bound(output, pending=artifact.PARENT_TOTAL_BYTES + 24 * MIB)
        captured = api.capture_initial_action_adapter(
            ROOT / "artifacts/jev-browser/20261015-action-lora-counterfactual-fit-01/adapter",
            output / "initial-parent",
        )
        verified_parent = artifact.validate_parent_capture(
            captured,
            inputs.metadata()["provenance"],
            expected_descriptor=inputs.metadata()["parent_descriptor"],
            gate=old,
        )
        state["parent"] = {
            "files": 496,
            "bytes": artifact.PARENT_TOTAL_BYTES,
            "historical_updates": 521,
            "scope": "Immutable original parent copy only, never diagnostically updated weights",
        }
        # After old parent capture, each file (including native.log) has a64MiB
        # kernel limit. No updated model/adapter/optimizer weights are serialized.
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_CAP, FILE_CAP))
        state["stage"] = "runtime-binary-verification"
        runtime = parse(ref_read(plan["allocator_runtime_pins"]))
        for spec in runtime["files"]:
            require(
                file_pin(spec["path"]) == spec["pin"],
                "Exact native allocator runtime binary/source",
            )
        dependency_runtime = parse(ref_read(plan["optional_dependency_runtime_pins"]))
        verify_optional_dependency_files(dependency_runtime)
        state["optional_dependency_runtime"] = dependency_runtime
        import torch

        state["allocator_loaded_origins"] = allocator_origins(torch, runtime)
        cap = allocator_cap(torch, config["device_uuid"])
        state["allocator"] = cap
        state["runtime"] = {
            "torch": torch.__version__,
            "cuda_compiled": torch.version.cuda,
            "kernel": os.uname().release,
            "driver_version_text": read("/proc/driver/nvidia/version").decode(),
            "native_binary_pins": runtime,
        }
        state["stage"] = "UMA-device-identity"
        replace(output / "progress.json", state)
        old.check("immediately-before-legacy-parent-capture-and-Torch", "cpu")
        uma_pins = parse(ref_read(plan["UMA_driver_identity_metadata_pins"]))
        require(
            uma_pins["schema"] == "gb10-UMA-driver-metadata-pins-v1"
            and uma_pins["inherited_runtime"] == plan["native_toolchain_runtime_pins"]
            and uma_pins["cuda_link_origins"] == tools.runtime["cuda_link_origins"],
            "Separate inherited-metadata UMA driver authority",
        )
        uma = uma_driver_identity(uma_pins, cap, host_memory(), config)
        write(output / "uma-device-identity.json", uma)
        state["uma_device_identity"] = {
            "path": str(output / "uma-device-identity.json"),
            **file_pin(output / "uma-device-identity.json"),
        }
        gate = Research(torch, output, config, binding, cap, uma)
        state["stage"] = "full-checkpoint-hash"
        old.check("before-checkpoint-payload-hash-and-BF16-loader", "load")
        checkpoint = artifact.resolve_base_checkpoint(producer.CACHE)
        gate.phase(
            "full-original-checkpoint-hash",
            lambda: require(
                artifact.pin(checkpoint, limit=producer.BASE_BYTES) == artifact.BASE_CHECKPOINT,
                "Exact full base checkpoint",
            ),
        )
        old.check("before-any-BF16-model-construction", "train")
        kernel = artifact.trusted_module(
            "jev_browser_fit_actions", "scripts/jev_browser_fit_actions.py"
        )
        fit_kernel = artifact.trusted_module(
            "jev_browser_fit_cfg_actions", "scripts/jev_browser_fit_cfg_actions.py"
        )
        state["stage"] = "constructor"

        def construct():
            state["model_constructor_attempts"] += 1
            replace(output / "progress.json", state)
            result = producer._construct_model(42)
            state["model_loads"] += 1
            replace(output / "progress.json", state)
            require_no_forbidden_modules()
            state["nonquantized_constructor"] = require_unquantized_modules(result.lm)
            state["optional_dependency_origins"] = optional_dependency_origins(dependency_runtime)
            return result

        model = gate.phase("full-original-constructor", construct)
        gate.loaded = True
        gate.phase("original-model-head-check", lambda: producer._model_check(model, kernel))
        state["verified_model_loads"] += 1
        replace(output / "progress.json", state)
        require(
            artifact.verify_reference_environment(inputs.reference_items, model=model)
            == state["versions"],
            "Original actual processor51 origins",
        )
        for adapter_config in model.lm.peft_config.values():
            adapter_config.base_model_name_or_path, adapter_config.revision = (
                artifact.MODEL_ID,
                artifact.REVISION,
            )
        state["stage"] = "parent-injection"
        state["parent_injection"] = gate.phase(
            "original-parent-injection",
            lambda: api.initialize_training_action_adapter(model, verified_parent),
        )
        state["stage"] = "original-all142-compile"
        rows, compiled_pins, plans, plan_pins = gate.phase(
            "original-all142-processor-compile",
            lambda: artifact.compile_public_actions(model, inputs, output=output, gate=gate),
        )
        state["compiled_pins"], state["plan_pins"] = compiled_pins, plan_pins
        original_plan_bytes = b"".join(
            encode(
                {
                    "case_id": row["record"]["case_id"],
                    "split": split,
                    "plan": plans[row["record"]["case_id"]].evidence(),
                }
            )
            + b"\n"
            for split in ("train", "development")
            for row in rows[split]
        )
        require(
            original_plan_bytes == ref_read(plan["original_CFG142_plans"]),
            "All142 original native CFG plans byte-identical",
        )
        state["original_CFG142_plans"] = {
            "sha256": sha(original_plan_bytes),
            "size_bytes": len(original_plan_bytes),
        }
        for split in ("train", "development"):
            for descriptor in selected[split]:
                row = rows[split][descriptor["zero_based_line_ordinal"]]
                require(
                    row["record"]["case_id"] == descriptor["case_id"]
                    and len(row["inputs"]["input_ids"][0]) == descriptor["input_tokens"]
                    and row["positions"].numel() == descriptor["target_tokens"],
                    "Exact selected compiled physical shapes",
                )
        gate.phase("postcompile-original-model-check", lambda: producer._model_check(model, kernel))
        state["stage"] = "two-disposable-native-updates"
        state["final_storage"] = diagnostic_steps(
            torch,
            model,
            rows,
            plans,
            mode=config["mode"],
            gate=gate,
            kernel=kernel,
            fit_kernel=fit_kernel,
            output=output,
            state=state,
            selected=plan["fixed_workload"],
        )

        def final_model_check():
            producer._model_check(model, kernel)
            state["nonquantized_final"] = require_unquantized_modules(model.lm)
            state["optional_dependency_origins_final"] = optional_dependency_origins(
                dependency_runtime
            )

        gate.phase("final-original-model-check", final_model_check)
        gate.phase("final-original-parent-check", lambda: producer._parent_checks(captured))
        inputs.check()
        source_auth(plan)
        state.update(
            stage="complete_memory_point_not_fit",
            peak_allocated_bytes=gate.peak_allocated,
            peak_reserved_bytes=gate.peak_reserved,
            actual_native_forward_calls=4,
            source95_unchanged=True,
            allocator_peak_scope="Maximum of preserved phase and pre/post cache-release peaks, not a sum; no cache-release stats reset",
        )
        arm_bound(output)
        require(not errors, "Independent heartbeat evidence failed")
        state["stage"] = "awaiting-native-work-final-capture"
        state["native_work_end_capture"] = wait_native_work_capture(output, ready, binding, errors)
        state["stage"] = "complete_memory_point_not_fit"
    except BaseException as exc:
        primary = exc
        state.update(
            error={"type": type(exc).__name__, "message": str(exc)},
            failed_stage=state["stage"],
            stage="failed_memory_point",
        )
        raise
    finally:
        stop.set()
        ticker.join(timeout=2)
        if ticker.is_alive():
            errors.append(RuntimeError("Heartbeat thread closure unknown"))
        state["elapsed_seconds"] = time.monotonic() - started
        if tools is not None:
            state["native_toolchain_authorized_entries"] = tools.authorizations
            state["native_toolchain_prepared"] = tools.runtime is not None
        finish_worker(output, state, primary=primary, cleanup_errors=errors)


def verify_ready(ready, config):
    require(
        type(ready) is dict
        and set(ready)
        == {"nonce", "pid", "startticks", "boot_id", "source_sha256", "profile_sha256"},
        "Exact preframework ready schema",
    )
    require(
        all(type(ready[k]) is int and ready[k] > 0 for k in ("pid", "startticks")),
        "Typed ready PID identity",
    )
    require(
        ready["nonce"] == config["nonce"]
        and ready["boot_id"] == config["boot_id"]
        and ready["source_sha256"] == config["source_pin"]["sha256"]
        and ready["profile_sha256"] == config["profile_pin"]["sha256"],
        "Reviewed preframework ready identity",
    )


def terminal(properties):
    return properties["SubState"] in {"exited", "dead", "failed"} and properties["MainPID"] == "0"


def closed_unit_metadata(properties, invocation, *, stopped):
    if properties["LoadState"] == "not-found":
        require(
            stopped
            and properties["InvocationID"] == ""
            and properties["ControlGroup"] == ""
            and properties["MainPID"] == "0"
            and properties["ActiveState"] == "inactive"
            and properties["SubState"] == "dead",
            "Typed unit GC only after authenticated successful stop",
        )
        return "confirmed-unit-GC"
    require(
        properties["LoadState"] == "loaded"
        and properties["InvocationID"] == invocation
        and properties["MainPID"] == "0",
        "Unit main process not closed",
    )
    return "retained-unit"


def terminal_after_proc_disappearance(unit, invocation, error, result):
    """Allow at most one second for manager metadata to catch up to proc exit."""
    started = time.monotonic()
    evidence = {
        "error": {"type": type(error).__name__, "message": str(error)},
        "confirmed_same_terminal": False,
        "maximum_seconds": 1.0,
        "started_monotonic": started,
        "observations": [],
    }
    result["terminal_sample_race"] = evidence
    for _ in range(52):
        remaining = 1.0 - (time.monotonic() - started)
        if not 0 < remaining <= 1.0:
            break
        try:
            current = service_show(unit, timeout=remaining)
        except Exception as secondary:
            evidence["query_error"] = {"type": type(secondary).__name__, "message": str(secondary)}
            error.add_note("Terminal confirmation query: " + repr(secondary))
            raise error from secondary
        checked = time.monotonic()
        same = current["InvocationID"] == invocation
        ended = terminal(current)
        evidence["observations"].append(
            {
                "checked_monotonic": checked,
                "query_timeout_seconds": remaining,
                "InvocationID": current["InvocationID"],
                "MainPID": current["MainPID"],
                "SubState": current["SubState"],
                "terminal": ended,
            }
        )
        evidence["elapsed_seconds"] = checked - started
        if not same:
            error.add_note("Terminal confirmation invocation changed")
            raise error
        if ended and 0 <= checked - started <= 1.0:
            evidence["confirmed_same_terminal"] = True
            return current
        if checked - started >= 1.0:
            break
        time.sleep(min(0.02, max(0, 1.0 - (checked - started))))
    error.add_note("No same-invocation terminal confirmation within one second")
    raise error


def arm_run(output, config):
    unit = "jev-native-memory-peak-" + config["mode"].lower() + "-" + config["nonce"] + ".service"
    config_path = output / "config.json"
    write(config_path, config)
    argv = unit_argv(unit, config_path, config["device_uuid"], output / "native.log")
    write(output / "unit-argv.json", argv)
    invocation = None
    result = {
        "unit": unit,
        "invocation": None,
        "success": False,
        "closure_confirmed": False,
        "error": None,
    }
    primary = None
    binding = None
    try:
        launch = subprocess.run(argv, capture_output=True, timeout=20)
        write(
            output / "unit-launch.json",
            {
                "returncode": launch.returncode,
                "stdout": launch.stdout.decode(errors="replace"),
                "stderr": launch.stderr.decode(errors="replace"),
            },
        )
        properties = service_show(unit)
        invocation = properties["InvocationID"]
        require(invocation and len(invocation) == 32, "Known owned invocation before lifecycle")
        result["invocation"] = invocation
        require(launch.returncode == 0, "Transient unit launch failed")
        deadline = time.monotonic() + 60
        while not (output / "ready-for-readback.json").exists():
            require(time.monotonic() < deadline, "Preframework ready timeout")
            current = service_show(unit)
            require(
                current["InvocationID"] == invocation and not terminal(current),
                "Worker exited before unit readback",
            )
            time.sleep(0.05)
        ready = parse(read(output / "ready-for-readback.json"))
        verify_ready(ready, config)
        properties = service_show(unit)
        require(properties["InvocationID"] == invocation, "Invocation drift before release")
        binding = unit_binding(properties, ready, unit)
        write(output / "unit-binding.json", {"properties": properties, **binding})
        write(output / "release.json", {**ready, **binding})
        start = last_metadata = time.monotonic()
        current = properties
        last_sample = None
        native_end = None
        while True:
            now = time.monotonic()
            require(now - start < 1800, "Owned research wall limit")
            if now - last_metadata >= 0.25:
                current = service_show(unit)
                require(current["InvocationID"] == invocation, "Owned invocation changed")
                last_metadata = now
            if terminal(current):
                break
            try:
                row = sample(binding["ControlGroup"], ready["pid"])
            except (FileNotFoundError, ProcessLookupError) as exc:
                current = terminal_after_proc_disappearance(unit, invocation, exc, result)
                break
            row["nonce"] = config["nonce"]
            reserve = research_floor(
                {"MemTotal": row["total"], "MemAvailable": row["available"]}, startup=False
            )["reserve_bytes"]
            pressure = pressure_stop(row, reserve)
            heartbeat_error = None
            if not pressure:
                try:
                    row.update(heartbeat_observation(output / "heartbeat.json", config["nonce"]))
                except BaseException as exc:
                    heartbeat_error = exc
                    row["heartbeat_check_failed_monotonic"] = time.monotonic()
                    row["heartbeat_read_error"] = {"type": type(exc).__name__, "message": str(exc)}
            try:
                append(output / "samples.jsonl", row, LIMITS["samples.jsonl"])
            except BaseException as exc:
                if heartbeat_error is not None:
                    heartbeat_error.add_note("Heartbeat sample persistence: " + repr(exc))
                    raise heartbeat_error from exc
                raise
            last_sample = row
            require(not pressure, "Owned host/cgroup pressure stop")
            if heartbeat_error is not None:
                raise heartbeat_error
            require(
                row["heartbeat_nonce_matches"] and 0 <= row["heartbeat_age_seconds"] < 10,
                "Fresh nonce-bound heartbeat",
            )
            try:
                live_identity = process_identity(ready["pid"])
                live_cgroup = cgroup_membership(ready["pid"])
            except (FileNotFoundError, ProcessLookupError) as exc:
                current = terminal_after_proc_disappearance(unit, invocation, exc, result)
                break
            require(
                live_identity["startticks"] == ready["startticks"]
                and live_cgroup == binding["ControlGroup"],
                "Owned live process identity drift",
            )
            if native_end is None and (output / "native-work-ready.json").exists():
                native_end = capture_native_work_end(output, ready, binding, unit)
            time.sleep(0.05)
        final = {
            "properties": current,
            "last_sample": last_sample,
            "native_work_end_capture": native_end,
            "post_exit_cgroup_scope": "Best-effort separate snapshot; absence is not zero events or a final peak",
        }
        try:
            final["cgroup"] = cgroup_snapshot(binding["ControlGroup"])
        except BaseException as exc:
            final["cgroup_read_error"] = {"type": type(exc).__name__, "message": str(exc)}
        write(output / "unit-terminal.json", final)
        worker_receipt = parse(read(output / "worker-receipt.json"))
        result["worker_receipt"] = worker_receipt
        if worker_receipt["stage"] == "failed_memory_point":
            error = worker_receipt.get("error") or {}
            fail(
                "Recorded worker failure ("
                + str(error.get("type"))
                + "): "
                + str(error.get("message"))[:2048]
            )
        require(
            current["Result"] == "success"
            and current["ExecMainCode"] == "1"
            and current["ExecMainStatus"] == "0",
            "Actual worker nonzero exit",
        )
        require(
            native_end is not None,
            "Native-work-end full cgroup evidence unavailable; point is unknown",
        )
        identity = native_work_identity(ready, binding)
        require(
            worker_receipt.get("native_work_end_capture", {}).get("snapshot_sha256")
            == file_pin(output / "native-work-final-cgroup.json")["sha256"],
            "Worker authenticated the exact native-work-end capture",
        )
        result["retained_lifetime_peaks"] = retained_lifetime_peaks(current, identity, native_end)
        require(
            worker_receipt["stage"] == "complete_memory_point_not_fit"
            and worker_receipt["completed_updates"] == 2
            and worker_receipt["synchronized_updates"] == 2
            and worker_receipt["optimizer_step_attempts"] == 2
            and worker_receipt["model_loads"] == 1
            and worker_receipt["verified_model_loads"] == 1,
            "Complete diagnostic native work",
        )
        result.update(success=True, worker_receipt=worker_receipt)
    except BaseException as exc:
        primary = exc
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        cleanup_errors = []
        before = None
        stopped = False
        try:
            before = service_show(unit)
            if invocation is None:
                # A timed-out launcher can have created this exact unique unit.
                require(
                    before["ControlGroup"].endswith("/app.slice/" + unit)
                    and len(before["InvocationID"]) == 32,
                    "Unknown launch ownership; cannot claim closure",
                )
                invocation = before["InvocationID"]
                result["invocation"] = invocation
            require(before["InvocationID"] == invocation, "Owned invocation drift during cleanup")
        except BaseException as exc:
            cleanup_errors.append(exc)
        if before is not None and before["InvocationID"] == invocation:
            evidence = {"properties": before}
            try:
                bound_group = (
                    binding["ControlGroup"] if binding is not None else before["ControlGroup"]
                )
                require(
                    bound_group.endswith("/app.slice/" + unit), "Pre-stop exact bound cgroup path"
                )
                evidence["bound_ControlGroup"] = bound_group
                evidence["cgroup"] = cgroup_snapshot(bound_group)
            except FileNotFoundError as exc:
                evidence["cgroup_read_error"] = {"type": type(exc).__name__, "message": str(exc)}
                # After authenticated natural exit systemd removes even successful
                # cgroups. This is unavailable evidence, never zero events/peak.
                if not (
                    binding is not None
                    and terminal(before)
                    and not cg_path(binding["ControlGroup"]).exists()
                ):
                    cleanup_errors.append(exc)
            except BaseException as exc:
                evidence["cgroup_read_error"] = {"type": type(exc).__name__, "message": str(exc)}
                cleanup_errors.append(exc)
            try:
                write(output / "pre-stop-unit-evidence.json", evidence)
            except BaseException as exc:
                cleanup_errors.append(exc)
        try:
            require(invocation is not None, "Unknown invocation; refuse guessed stop")
            safe_stop(unit, invocation)
            stopped = True
        except BaseException as exc:
            cleanup_errors.append(exc)
        try:
            require(before is not None and invocation is not None, "Unknown closure identity")
            after = service_show(unit)
            closure_kind = closed_unit_metadata(after, invocation, stopped=stopped)
            group = binding["ControlGroup"] if binding is not None else before["ControlGroup"]
            require(group.endswith("/app.slice/" + unit), "Cleanup exact owned cgroup path")
            path = cg_path(group)
            require(
                not path.exists() or read(path / "cgroup.procs").strip() == b"",
                "Owned cgroup not empty",
            )
            if (output / "ready-for-readback.json").exists():
                old_ready = parse(read(output / "ready-for-readback.json"))
                verify_ready(old_ready, config)
                old_proc = Path("/proc") / str(old_ready["pid"])
                require(
                    not old_proc.exists()
                    or process_identity(old_ready["pid"])["startticks"] != old_ready["startticks"],
                    "Original owned process still alive outside unit",
                )
            else:
                require(
                    closure_kind != "confirmed-unit-GC",
                    "GC without known worker PID/startticks is unknown",
                )
            result["closure_confirmed"] = True
            result["closure_kind"] = closure_kind
            result["post_stop"] = after
        except BaseException as exc:
            cleanup_errors.append(exc)
        if cleanup_errors:
            result["success"] = False
            result["cleanup_errors"] = [
                {"type": type(e).__name__, "message": str(e)} for e in cleanup_errors
            ]
            for exc in cleanup_errors:
                if primary is not None:
                    primary.add_note("Owned unit cleanup: " + str(exc))
        try:
            if (output / "worker-receipt.json").exists():
                result["worker_receipt"] = parse(read(output / "worker-receipt.json"))
            elif (output / "progress.json").exists():
                result["last_durable_progress_only"] = parse(read(output / "progress.json"))
                result["final_update_count"] = (
                    "unknown native effects after last durable progress; recorded returned/synchronized counts are lower bounds, never infer zero from absent completion"
                )
            write(output / "arm-receipt.json", result)
        except BaseException as exc:
            if primary is not None:
                primary.add_note("Arm receipt finalization: " + str(exc))
            else:
                cleanup_errors.append(exc)
        if cleanup_errors and primary is None:
            raise cleanup_errors[0]
    return result


def controller(args):
    require(
        file_pin(Path(__file__))["sha256"] == args.expected_source_sha256
        and file_pin(PLAN)["sha256"] == args.expected_plan_sha256,
        "Externally reviewed driver/profile SHA",
    )
    plan = parse(read(PLAN))
    inventory = source_auth(plan)
    selected_rows(plan)
    if not args.run:
        print(
            json.dumps(
                {
                    "status": "prepared-only-no-unit-no-GPU",
                    "memory_admission_profile_kind": UMA_KIND,
                    "research_sources": len(inventory["files"]),
                    "unit_cap_bytes": CAP,
                    "native_allocator_cap_bytes": CAP,
                    "steps_per_arm": 2,
                    "max_actual_updates": 4,
                }
            )
        )
        return
    require(
        args.expected_gpu_uuid
        and args.expected_gpu_uuid.startswith("GPU-")
        and len(args.expected_gpu_uuid) <= 80,
        "Explicit selected physical GPU UUID required",
    )
    require(args.output is not None, "Fresh explicit controller output")
    output = ordinary(args.output)
    require(
        output.parent == BASE and not output.exists(),
        "New bounded campaign directly under reviewed artifact root",
    )
    output.mkdir(mode=0o700)
    state = {
        "kind": UMA_KIND,
        "qualified": False,
        "success": False,
        "planned_arms": ["A", "C"],
        "arms": [],
        "unrun": ["A", "C"],
        "error": None,
        "original_CPU_host_caps_unchanged": True,
        "memory_admission_profile_kind": UMA_KIND,
        "old_CUDA_free_veto_removed_for_verified_UMA_only": True,
    }
    primary = None
    try:
        obs = host_memory()
        floor = research_floor(obs, startup=True)
        require(
            obs["MemAvailable"] >= floor["required_bytes"],
            "Fresh prospective hard-cap sum admission refused",
        )
        write(
            output / "controller-startup-gate.json",
            {"observation": obs, **floor, "resident_credit_bytes": 0},
        )
        for mode in ("A", "C"):
            arm = output / mode
            arm.mkdir(mode=0o700)
            config = {
                "mode": mode,
                "nonce": uuid.uuid4().hex,
                "boot_id": obs["boot_id"],
                "source_pin": file_pin(Path(__file__)),
                "profile_pin": file_pin(PLAN),
                "campaign_output": str(output),
                "device_uuid": args.expected_gpu_uuid,
            }
            state["unrun"].remove(mode)
            arm_error = None
            try:
                result = arm_run(arm, config)
            except BaseException as exc:
                arm_error = exc
                raise
            finally:
                try:
                    if (arm / "arm-receipt.json").exists():
                        state["arms"].append(parse(read(arm / "arm-receipt.json")))
                except BaseException as exc:
                    if arm_error is None:
                        raise
                    arm_error.add_note("Controller arm receipt: " + str(exc))
            require(
                result["success"] and result["closure_confirmed"],
                "Previous arm failed/closure unknown; no next model",
            )
        state["success"] = True
    except BaseException as exc:
        primary = exc
        state["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        try:
            state["inventory_excluding_controller_receipt"] = final_inventory(
                output, reserved=4 * MIB
            )
            write(output / "controller-receipt.json", state)
        except BaseException as exc:
            if primary is None:
                raise
            primary.add_note("Controller finalization: " + str(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--expected-plan-sha256")
    parser.add_argument("--expected-gpu-uuid")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker is not None:
        require(not args.run, "Worker has no implicit controller execution")
        worker(args.worker)
    else:
        require(
            args.expected_source_sha256 and args.expected_plan_sha256,
            "Explicit reviewed source/profile SHAs",
        )
        controller(args)


if __name__ == "__main__":
    main()
