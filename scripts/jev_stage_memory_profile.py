"""Standalone H12-derived owned-unit/stage-memory capsule (stdlib at import).

No model, worker CLI or artifact-driver import is provided. A separately reviewed
producer must retain original ResourceGate CPU/load/train checks and call the
lazy helpers in their original phases, including guarded save/validation before
the native-work end barrier. This source-only capsule does not attest capacity.
Plan07's historical no-empty_cache condition is superseded only by the explicit
fixed H12 policy below. New source/profile/receipt identities never qualify an
old source94/95 artifact. Full fitter and completed-artifact validator are absent.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE = "gb10-stage-memory-fixed-cache-v1"
PHASE_SCHEMA = "jev-stage-memory-phase-v1"
RECEIPT_SCHEMA = "jev-stage-memory-worker-receipt-v1"
SOURCE_SCHEMA = "jev-stage-memory-source-union-v1"
FROZEN_H12_SOURCE_SHA256 = "ffbe30212bc2c34cd0f01a19599646d7f740ffbf1de1ef24496a7f3676af9c86"
GIB = 1024**3
MIB = 1024**2
CAP = 40 * GIB
STOP = CAP - GIB // 2
TOTAL_CAP = GIB
ARM_CAP = (TOTAL_CAP - 4 * MIB) // 2
FILE_CAP = 64 * MIB
CACHE_POLICY = "fixed-post-backward-unused-cache-release-v1"
LIMITS = {"native.log": 12 * MIB, "samples.jsonl": 16 * MIB, "phases.jsonl": 4 * MIB}
PUBLIC_KIND = "jev_public_cfg_stage_bounded_action_lora_v1"
MIXED_KIND = "jev_mixed_cfg_stage_bounded_action_lora_v1"


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
        if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            number, unit = value.split()
            require(unit == "kB", "Native memory unit")
            values[key] = int(number) * 1024
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
            for k in (
                "nonce",
                "pid",
                "startticks",
                "boot_id",
                "source_sha256",
                "profile_sha256",
                "capsule_sha256",
                "source_inventory_sha256",
            )
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


def unit_argv(unit, config, device_uuid, log, *, worker_entry):
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
        str(ordinary(worker_entry)),
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


class StageGate:
    """Explicit stage profile; never old source94/95 resource evidence."""

    def __init__(self, torch, output, config, binding, allocator):
        self.torch, self.output, self.config, self.binding, self.allocator = (
            torch,
            output,
            config,
            binding,
            allocator,
        )
        self.loaded = False
        self.sequence = 0
        self.peak_allocated = self.peak_reserved = 0

    def cuda(self):
        t = self.torch
        free, total = t.cuda.mem_get_info(0)
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
        obs = host_memory()
        floor = research_floor(obs, startup=not self.loaded)
        cg = cgroup_snapshot(self.binding["ControlGroup"])
        cuda = self.cuda()
        require(
            obs["boot_id"] == self.config["boot_id"]
            and obs["MemAvailable"] >= floor["required_bytes"],
            "Fresh independent research memory admission refused",
        )
        require(
            cuda["backend"] == "native"
            and cuda["fraction"] == self.allocator["fraction"]
            and cuda["free"] >= CAP + GIB
            and cuda["reserved"] < STOP,
            "Native allocator/device guard refused",
        )
        require(
            cg["max"] == CAP
            and cg["high"] is None
            and cg["swap_max"] == 0
            and cg["oom_group"] == 1
            and not pressure_stop({"available": obs["MemAvailable"], **cg}, floor["reserve_bytes"]),
            "Owned cgroup pressure or limit drift",
        )
        arm_bound(self.output)
        append(
            self.output / "phases.jsonl",
            {
                "schema_id": PHASE_SCHEMA,
                "profile_id": PROFILE,
                "event": "stage-gate",
                "phase": phase,
                "scope": scope,
                "observation": obs,
                **floor,
                "resident_credit_bytes": 0,
                "admitted": True,
            },
            LIMITS["phases.jsonl"],
        )
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
                "schema_id": PHASE_SCHEMA,
                "profile_id": PROFILE,
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
                        "schema_id": PHASE_SCHEMA,
                        "profile_id": PROFILE,
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
                        "peak_scope": "since this explicit stats reset; maxima retained across fixed post-backward cache release",
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


def backward_cache_release(torch, model, gate, output, fn, *, mode, case_id, step):
    result = fn()  # Original profiled backward exactly once; failure never enters release.
    torch.cuda.synchronize(0)
    before = gate.cuda()
    storage_before = cache_storage_identity(model)
    gate.peak_allocated = max(gate.peak_allocated, before["peak_allocated"])
    gate.peak_reserved = max(gate.peak_reserved, before["peak_reserved"])
    row = {
        "policy": CACHE_POLICY,
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
        state["stage"] = "failed_stage_memory"
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


def verify_ready(ready, config):
    require(
        type(ready) is dict
        and set(ready)
        == {
            "nonce",
            "pid",
            "startticks",
            "boot_id",
            "source_sha256",
            "profile_sha256",
            "capsule_sha256",
            "source_inventory_sha256",
        },
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
        and ready["profile_sha256"] == config["profile_pin"]["sha256"]
        and ready["capsule_sha256"] == config["capsule_pin"]["sha256"]
        and ready["source_inventory_sha256"] == config["source_inventory"]["sha256"],
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
    validate_stage_config(config, output)
    completion = completion_contract(config["artifact_kind"])
    require(output.is_dir() and not any(output.iterdir()), "Fresh empty owned stage arm")
    unit = "jev-stage-memory-" + config["mode"].lower() + "-" + config["nonce"] + ".service"
    config_path = output / "config.json"
    write(config_path, config)
    argv = unit_argv(
        unit,
        config_path,
        config["device_uuid"],
        output / "native.log",
        worker_entry=config["worker_entry"],
    )
    write(output / "unit-argv.json", argv)
    invocation = None
    result = {
        "schema_id": "jev-stage-memory-arm-receipt-v1",
        "profile_id": PROFILE,
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
        if worker_receipt["stage"] == "failed_stage_memory":
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
            worker_receipt["schema_id"] == RECEIPT_SCHEMA
            and worker_receipt["profile_id"] == PROFILE
            and worker_receipt["allocator_policy"] == CACHE_POLICY
            and worker_receipt["kind"] == completion["kind"]
            and worker_receipt["stage"] == "complete_stage_fit_unqualified"
            and worker_receipt["completed_updates"] == completion["updates"]
            and worker_receipt["synchronized_updates"] == completion["updates"]
            and worker_receipt["optimizer_step_attempts"] == completion["updates"]
            and worker_receipt["actual_native_forward_calls"] == completion["forwards"]
            and worker_receipt["development_updates"] == 0
            and worker_receipt["model_loads"] == 1
            and worker_receipt["verified_model_loads"] == 1,
            "Complete declared stage native work; data/save qualification remains separate",
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


def completion_contract(kind):
    require(kind in {PUBLIC_KIND, MIXED_KIND}, "Explicit new stage artifact kind")
    updates, forwards = (120, 142) if kind == PUBLIC_KIND else (129, 156)
    return {"kind": kind, "updates": updates, "forwards": forwards}


def verify_stage_source_union(ref, *, worker_entry=None):
    """Authenticate explicit new source closure; pending capsule-only unions cannot run."""
    inventory = parse(ref_read(ref))
    require(
        inventory["schema_id"] == SOURCE_SCHEMA and type(inventory["producer_ready"]) is bool,
        "Separate stage source-union schema",
    )
    original = parse(ref_read(inventory["base_inventory"]))
    extensions, files = inventory["extensions"], inventory["files"]
    require(
        type(extensions) is dict
        and type(files) is dict
        and not set(original["files"]) & set(extensions)
        and files == {**original["files"], **extensions}
        and inventory["identity_sha256"] == sha(encode(files))
        and 1 <= len(files) <= 512,
        "Exact base-plus-explicit-stage-extension source population",
    )
    capsule = Path(__file__).resolve().relative_to(ROOT).as_posix()
    require(
        capsule in extensions and files[capsule] == file_pin(Path(__file__)),
        "Exact separately pinned capsule source",
    )
    if worker_entry is not None:
        worker = ordinary(worker_entry).relative_to(ROOT).as_posix()
        require(
            inventory["producer_ready"] and len(extensions) >= 3 and worker in extensions,
            "Full producer source closure pending fitter and artifact validator",
        )
    for label, expected in files.items():
        require(
            type(label) is str
            and not Path(label).is_absolute()
            and ".." not in Path(label).parts
            and label.endswith(".py"),
            "Root-relative Python-only stage source",
        )
        require(file_pin(ROOT / label) == expected, "Stage source drift: " + label)
    return inventory


def validate_stage_config(config, output):
    """Bind a new worker, profile, capsule and source union before launch/release."""
    require(
        type(config) is dict
        and set(config)
        == {
            "schema_id",
            "profile_id",
            "artifact_kind",
            "mode",
            "nonce",
            "boot_id",
            "source_pin",
            "profile_pin",
            "capsule_pin",
            "worker_entry",
            "profile_path",
            "source_inventory",
            "campaign_output",
            "campaign_root",
            "device_uuid",
        },
        "Exact new stage worker config schema",
    )
    require(
        config["schema_id"] == "jev-stage-memory-config-v1" and config["profile_id"] == PROFILE,
        "Explicit new stage profile identity",
    )
    completion_contract(config["artifact_kind"])
    worker = ordinary(config["worker_entry"])
    profile_path = ordinary(config["profile_path"])
    campaign = ordinary(config["campaign_output"])
    campaign_root = ordinary(config["campaign_root"])
    output = ordinary(output)
    require(
        worker.parent == ROOT / "scripts"
        and worker.suffix == ".py"
        and worker != Path(__file__).resolve(),
        "Explicit separately reviewed producer entry",
    )
    require(
        profile_path.is_relative_to(ROOT / "artifacts/jev-runtime")
        and campaign_root.is_relative_to(ROOT / "artifacts/jev-runtime")
        and campaign.parent == campaign_root
        and output.parent == campaign
        and output.name == config["mode"]
        and config["mode"] in {"A", "C"},
        "Bounded explicitly owned stage output/profile paths",
    )
    require(
        type(config["nonce"]) is str
        and len(config["nonce"]) == 32
        and all(c in "0123456789abcdef" for c in config["nonce"])
        and type(config["boot_id"]) is str
        and config["boot_id"]
        and type(config["device_uuid"]) is str
        and config["device_uuid"].startswith("GPU-"),
        "Typed fresh stage identity",
    )
    require(
        file_pin(worker) == config["source_pin"]
        and file_pin(profile_path) == config["profile_pin"]
        and file_pin(Path(__file__)) == config["capsule_pin"],
        "Reviewed producer/profile/capsule pins",
    )
    profile = parse(read(profile_path))
    require(
        profile["schema_id"] == "jev-stage-memory-profile-v1"
        and profile["profile_id"] == PROFILE
        and profile["cache_policy"] == CACHE_POLICY
        and profile["caps"]
        == {
            "unit": CAP,
            "allocator": CAP,
            "device_free": CAP + GIB,
            "stop": STOP,
            "swap": 0,
            "wall_seconds": 1800,
            "campaign": TOTAL_CAP,
            "arm": ARM_CAP,
            "file_and_log": FILE_CAP,
        },
        "Exact reviewed stage policy and unchanged hard limits",
    )
    require(
        profile["source_inventory"] == config["source_inventory"],
        "Profile binds exact externally reviewed source union",
    )
    verify_stage_source_union(config["source_inventory"], worker_entry=worker)
    return profile


def await_worker_release(config_path, errors):
    """Stdlib-only authenticated barrier; caller owns heartbeat and finalization."""
    output = ordinary(config_path).parent
    config = parse(read(config_path))
    profile = validate_stage_config(config, output)
    identity = process_identity(os.getpid())
    ready = {
        "nonce": config["nonce"],
        "pid": os.getpid(),
        "startticks": identity["startticks"],
        "boot_id": boot_id(),
        "source_sha256": config["source_pin"]["sha256"],
        "profile_sha256": config["profile_pin"]["sha256"],
        "capsule_sha256": config["capsule_pin"]["sha256"],
        "source_inventory_sha256": config["source_inventory"]["sha256"],
    }
    verify_ready(ready, config)
    require(ready["boot_id"] == config["boot_id"], "Same boot before worker release")
    write(output / "ready-for-readback.json", ready)
    deadline = time.monotonic() + 60
    while not (output / "release.json").exists():
        require(
            time.monotonic() < deadline and not errors,
            "Controller release barrier timed out/failed",
        )
        time.sleep(0.02)
    binding = validate_release(config, ready, parse(read(output / "release.json")))
    return config, profile, ready, binding


def prepare_native_tools(output, binding, runtime, startup_gate):
    """Original CPU admission precedes all326 payload checks; no framework import."""
    startup_gate.check("before-any-project-framework-or-parent-payload", "cpu")
    require_no_forbidden_modules()
    cache_root = ordinary(output) / "native-tools"
    require(not cache_root.exists(), "Fresh per-arm native compiler/cache root")
    cache_root.mkdir(mode=0o700)
    for key, value in native_cache_environment(output).items():
        if key not in {"CC", "PATH"}:
            Path(value).mkdir(mode=0o700, parents=True, exist_ok=True)
    tools = NativeTools(output, binding)
    sys.addaudithook(lambda event, arguments: worker_audit(event, arguments, tools=tools))
    tools.prepare(runtime)
    return tools


def apply_post_parent_file_limits():
    """Call only after immutable original parent capture, before native work."""
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_CAP, FILE_CAP))


def verify_allocator_files(runtime, dependency_runtime):
    """Original worker byte checks; caller retains the original CPU gate first."""
    for spec in runtime["files"]:
        require(
            file_pin(spec["path"]) == spec["pin"],
            "Exact native allocator runtime binary/source",
        )
    verify_optional_dependency_files(dependency_runtime)
