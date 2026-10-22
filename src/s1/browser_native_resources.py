"""Explicit GB10 UMA training resource profile; the original fitter is unchanged.

Reuses byte-pinned H15 identity/floor functions, never its source qualification
or actor. Allocator and memcg limits are separate, not a combined physical cap.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import math
import os
import stat
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).absolute().parents[2]
PROFILE = "gx10-uma-native-40gib-v1"
SCHEMA = "jev-native-resource-gate-gx10-uma-v1"
GIB, MIB = 1024**3, 1024**2
CAP, STOP = 40 * GIB, 40 * GIB - GIB // 2
TOTAL = 130595860480
EXPECTED_UUID = "e3429caa-b057-cbe0-917d-d3b85b313be7"
EXPECTED_UNIT = "/user.slice/user-1001.slice/user@1001.service/app.slice/jev-native-v2-training-20261022-01.service"
COMPONENTS = {
    "h15": {
        "path": "artifacts/jev-runtime/public-native-fit-memory-uma-probe-20261020-01/native_uma_peak_probe_01.py",
        "sha256": "2ed7cda3054da62e7440bf57e4bfae15e26ee6d22de8b80aede6c8f53973e3cb",
        "size_bytes": 118654,
    },
    "allocator": {
        "path": "artifacts/jev-runtime/public-native-fit-memory-peak-probe-20261018-01/allocator-runtime-pins-02.json",
        "sha256": "1d95d5f9250132299901871433850b0a39b86b18cc2b58eed378dabcf5117a61",
        "size_bytes": 11204,
    },
    "driver": {
        "path": "artifacts/jev-runtime/public-native-fit-memory-peak-probe-20261018-01/native-toolchain-runtime-pins-02.json",
        "sha256": "1ad0a1749e1ab69235cf6356939ef55edf904ba8942794f6241ef9580c676582",
        "size_bytes": 85649,
    },
}


def require(value, message):
    if not value:
        raise ValueError(message)


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def _component(name):
    ref = COMPONENTS[name]
    path = ROOT / ref["path"]
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Ordinary resource component")
    blob = path.read_bytes()
    require(
        len(blob) == ref["size_bytes"] and hashlib.sha256(blob).hexdigest() == ref["sha256"],
        "Pinned resource component differs: " + name,
    )
    return path, blob


def _component_descriptor():
    return {
        "schema_id": "jev-native-resource-components-v1",
        "artifact_path": "inputs/resource-components",
        "files": {
            Path(ref["path"]).name: {k: ref[k] for k in ("sha256", "size_bytes")}
            for ref in COMPONENTS.values()
        },
    }


def verify_components(destination, descriptor):
    """Separate fixed component evidence; never extend the Python source manifest."""
    require(descriptor == _component_descriptor(), "Exact native resource component descriptor")
    destination = Path(destination).absolute()
    require(
        destination.is_dir()
        and not any(p.is_symlink() for p in (destination, *destination.parents)),
        "Ordinary native resource component directory",
    )
    require(
        {p.name for p in destination.iterdir()} == set(descriptor["files"]),
        "Exact three native resource component files",
    )
    for label, pin in descriptor["files"].items():
        path = destination / label
        require(path.is_file() and not path.is_symlink(), "Regular resource component copy")
        blob = path.read_bytes()
        require(
            len(blob) == pin["size_bytes"] and hashlib.sha256(blob).hexdigest() == pin["sha256"],
            "Captured native resource component bytes differ",
        )
    return True


def capture_components(destination):
    destination = Path(destination).absolute()
    require(
        not destination.exists()
        and not any(p.is_symlink() for p in (destination, *destination.parents)),
        "Fresh ordinary native resource component capture",
    )
    destination.mkdir(parents=True, mode=0o700)
    for name in COMPONENTS:
        source, blob = _component(name)
        with (destination / source.name).open("xb") as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
    descriptor = _component_descriptor()
    verify_components(destination, descriptor)
    return descriptor


@lru_cache(maxsize=1)
def mechanism():
    path, blob = _component("h15")
    spec = importlib.util.spec_from_file_location("_native_resource_h15_mechanisms", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(blob, str(path), "exec"), module.__dict__)
    return module


def floor(host, *, loaded):
    require(type(loaded) is bool, "Explicit loaded resource state")
    return mechanism().research_floor(host, startup=not loaded)


def _identity():
    text = Path("/proc/self/stat").read_text()
    fields = text[text.rfind(")") + 2 :].split()
    return {
        "pid": os.getpid(),
        "start_ticks": int(fields[19]),
        "pgid": int(fields[2]),
        "sid": int(fields[3]),
    }


def _read_pairs(path):
    return {k: int(v) for k, v in (line.split() for line in path.read_text().splitlines())}


def unit_snapshot(expected_unit):
    require(expected_unit == EXPECTED_UNIT, "Exact separately owned native training unit")
    require(
        Path("/proc/self/cgroup").read_text().strip() == "0::" + expected_unit,
        "Current process belongs to exact unit",
    )
    group = Path("/sys/fs/cgroup") / expected_unit.lstrip("/")
    metadata = group.stat()
    require(stat.S_ISDIR(metadata.st_mode), "Own cgroup directory")

    def value(name):
        raw = (group / name).read_text().strip()
        return raw if raw == "max" else int(raw)

    members = sorted(int(x) for x in (group / "cgroup.procs").read_text().split())
    return {
        "path": expected_unit,
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "actor": _identity(),
        "InvocationID": os.environ.get("INVOCATION_ID"),
        "memory_max": value("memory.max"),
        "memory_high": value("memory.high"),
        "memory_current": value("memory.current"),
        "memory_peak": value("memory.peak"),
        "swap_max": value("memory.swap.max"),
        "swap_current": value("memory.swap.current"),
        "events": _read_pairs(group / "memory.events"),
        "cgroup_events": _read_pairs(group / "cgroup.events"),
        "cgroup_stat": _read_pairs(group / "cgroup.stat"),
        "members": members,
        "properties": _properties(expected_unit),
    }


def validate_unit(unit, expected_unit):
    require(
        set(unit)
        == {
            "path",
            "device",
            "inode",
            "actor",
            "InvocationID",
            "memory_max",
            "memory_high",
            "memory_current",
            "memory_peak",
            "swap_max",
            "swap_current",
            "events",
            "cgroup_events",
            "cgroup_stat",
            "members",
            "properties",
        },
        "Complete native unit evidence",
    )
    actor = unit["actor"]
    require(
        expected_unit == EXPECTED_UNIT
        and unit["path"] == expected_unit
        and type(unit["device"]) is int
        and unit["device"] >= 0
        and type(unit["inode"]) is int
        and unit["inode"] > 0
        and set(actor) == {"pid", "start_ticks", "pgid", "sid"}
        and all(type(v) is int and v > 0 for v in actor.values())
        and actor["pid"] == actor["pgid"] == actor["sid"]
        and type(unit["InvocationID"]) is str
        and len(unit["InvocationID"]) == 32
        and all(x in "0123456789abcdef" for x in unit["InvocationID"]),
        "Exact actor/unit/invocation identity",
    )
    require(
        unit["properties"]
        == _expected_properties(expected_unit, actor["pid"], unit["InvocationID"]),
        "Recorded exact native unit properties",
    )
    require(
        all(
            type(unit[k]) is int and unit[k] >= 0
            for k in ("memory_max", "memory_current", "memory_peak", "swap_max", "swap_current")
        )
        and unit["memory_max"] == CAP
        and unit["memory_high"] == "max"
        and unit["memory_current"] <= unit["memory_peak"] <= CAP
        and unit["swap_max"] == unit["swap_current"] == 0,
        "Own40GiB memcg/no-swap readback",
    )
    require(
        all(type(v) is int and v >= 0 for v in unit["events"].values())
        and {"max", "oom", "oom_kill", "oom_group_kill"} <= set(unit["events"])
        and all(unit["events"][k] == 0 for k in ("max", "oom", "oom_kill", "oom_group_kill"))
        and unit["cgroup_events"] == {"populated": 1, "frozen": 0}
        and unit["cgroup_stat"].get("nr_descendants") == 0
        and unit["cgroup_stat"].get("nr_dying_descendants") == 0
        and unit["members"] == sorted(set(unit["members"]))
        and all(type(v) is int and v > 0 for v in unit["members"])
        and actor["pid"] in unit["members"],
        "No hidden cgroups or unit pressure/OOM",
    )


def _unit_key(unit):
    return [unit[k] for k in ("path", "device", "inode", "actor", "InvocationID")]


def _expected_properties(unit, pid, invocation):
    return {
        "MainPID": str(pid),
        "ControlGroup": unit,
        "Type": "exec",
        "MemoryAccounting": "yes",
        "MemoryMax": str(CAP),
        "MemoryHigh": "infinity",
        "MemorySwapMax": "0",
        "OOMPolicy": "kill",
        "KillMode": "control-group",
        "RuntimeMaxUSec": "3h",
        "TimeoutStopUSec": "20s",
        "KillSignal": "2",
        "InvocationID": invocation,
    }


def _properties(unit):
    wanted = _expected_properties(unit, os.getpid(), os.environ.get("INVOCATION_ID"))
    result = subprocess.run(
        ["systemctl", "--user", "show", unit.rsplit("/", 1)[1]]
        + ["--property=" + k for k in wanted],
        capture_output=True,
        timeout=5,
        check=True,
    )
    require(len(result.stdout) < MIB and not result.stderr, "Bounded native unit properties")
    observed = dict(line.split("=", 1) for line in result.stdout.decode().splitlines())
    require(observed == wanted, "Exact40GiB native unit/deadline/stop properties")
    return observed


class Runtime:
    """Only called after a fresh startup admission; touches this process's CUDA."""

    def __init__(self, host):
        import torch

        self.t, self.h = torch, mechanism()
        self.allocator_runtime = json.loads(_component("allocator")[1])
        self.driver_runtime = json.loads(_component("driver")[1])
        self.config = {"boot_id": host["boot_id"], "device_uuid": EXPECTED_UUID}
        self.allocator = self.h.allocator_cap(torch, self.config["device_uuid"])
        torch.cuda.synchronize(0)
        self.origins = self.h.allocator_origins(torch, self.allocator_runtime)

    def _driver(self, host, allocator):
        h = self.h
        metadata = h.uma_driver_metadata(self.driver_runtime)
        require(
            all(hasattr(os, k) for k in ("RTLD_NOLOAD", "RTLD_NOW", "RTLD_LOCAL")),
            "Exact already-loaded libcuda support",
        )
        library = ctypes.CDLL(h.UMA_LIBCUDA, mode=os.RTLD_NOLOAD | os.RTLD_NOW | os.RTLD_LOCAL)
        symbols = {name: getattr(library, name) for name in h.UMA_SYMBOLS}
        addresses = {name: ctypes.cast(fn, ctypes.c_void_p).value for name, fn in symbols.items()}
        maps = Path("/proc") / str(os.getpid()) / "maps"
        h.uma_symbol_mappings(h.read(maps, 8 * MIB).decode(), addresses, metadata)
        driver = {**h.uma_driver_read(symbols, ctypes), "boot_id": host["boot_id"]}
        h.uma_validate_identity(driver, allocator, host, self.config)
        require(h.uma_driver_metadata(self.driver_runtime) == metadata, "Stable libcuda metadata")
        h.uma_symbol_mappings(h.read(maps, 8 * MIB).decode(), addresses, metadata)
        return {
            **driver,
            "library_metadata": metadata,
            "symbol_addresses": addresses,
            "load_mode": "exact-canonical-RTLD_NOLOAD",
            "pin_scope": "inherited metadata and executable maps; not DSO payload SHA",
        }

    def snapshot(self, host):
        t = self.t
        require(
            t.cuda.device_count() == 1 and t.cuda.current_device() == 0,
            "One same selected CUDA device",
        )
        props = t.cuda.get_device_properties(0)
        allocator = {
            "uuid": str(props.uuid),
            "name": props.name,
            "total_bytes": props.total_memory,
            "fraction": t.cuda.get_per_process_memory_fraction(0),
            "backend": t.cuda.memory.get_allocator_backend(),
        }
        allocator["allowed_bytes_upper"] = math.floor(allocator["fraction"] * props.total_memory)
        free, total = t.cuda.mem_get_info(0)
        cuda = {
            "free": free,
            "total": total,
            "allocated": t.cuda.memory_allocated(0),
            "reserved": t.cuda.memory_reserved(0),
            "peak_allocated": t.cuda.max_memory_allocated(0),
            "peak_reserved": t.cuda.max_memory_reserved(0),
        }
        return {"identity": self._driver(host, allocator), "allocator": allocator, "cuda": cuda}

    def release(self):
        self.t.cuda.synchronize(0)
        self.t.cuda.empty_cache()


def validate_native(snapshot, host, boot_id):
    h = mechanism()
    require(set(snapshot) == {"allocator", "cuda", "identity"}, "Exact native evidence fields")
    allocator, cuda, identity = (snapshot[k] for k in ("allocator", "cuda", "identity"))
    require(
        set(identity)
        == {
            "context_device",
            "integrated",
            "uuid",
            "name",
            "total_bytes",
            "boot_id",
            "library_metadata",
            "symbol_addresses",
            "load_mode",
            "pin_scope",
        }
        and type(identity["context_device"]) is int
        and identity["context_device"] >= 0,
        "Exact current Driver device evidence",
    )
    h.uma_validate_identity(
        identity, allocator, host, {"boot_id": boot_id, "device_uuid": EXPECTED_UUID}
    )
    require(
        set(allocator)
        == {"uuid", "name", "total_bytes", "fraction", "backend", "allowed_bytes_upper"}
        and allocator["backend"] == "native"
        and type(allocator["total_bytes"]) is int
        and type(allocator["allowed_bytes_upper"]) is int
        and type(allocator["fraction"]) is float
        and math.isfinite(allocator["fraction"])
        and allocator["allowed_bytes_upper"] == math.floor(allocator["fraction"] * TOTAL)
        and CAP - 8 * MIB <= allocator["allowed_bytes_upper"] <= CAP,
        "Conservative40GiB native fraction readback",
    )
    require(
        set(cuda) == {"free", "total", "allocated", "reserved", "peak_allocated", "peak_reserved"}
        and all(type(v) is int and v >= 0 for v in cuda.values())
        and 0 <= cuda["free"] <= cuda["total"] == TOTAL
        and cuda["allocated"] <= cuda["reserved"] < STOP
        and cuda["allocated"] <= cuda["peak_allocated"] <= CAP
        and cuda["reserved"] <= cuda["peak_reserved"] <= CAP,
        "Raw CUDA counters and39.5GiB live soft stop",
    )
    origins = json.loads(_component("driver")[1])["cuda_link_origins"]
    expected_metadata = {
        k: origins[0][k] for k in ("resolved", "size_bytes", "mtime_ns", "device", "inode")
    }
    require(
        identity.get("load_mode") == "exact-canonical-RTLD_NOLOAD"
        and identity.get("library_metadata") == expected_metadata
        and set(identity.get("symbol_addresses", {})) == set(h.UMA_SYMBOLS)
        and all(type(v) is int and v > 0 for v in identity["symbol_addresses"].values()),
        "Pinned driver metadata/executable symbol evidence",
    )


def release_phase(phase):
    return phase.startswith("before-BF16-processor-compile-") or phase in {
        "before-native-train-forward",
        "before-native-development-forward",
        "after-native-training",
    }


def phase_scope(phase):
    if phase in {"model-verified-loaded", "after-native-training"}:
        return "train"
    # Import is stdlib-only and avoids a second hard-coded phase taxonomy.
    from jev_public_action_artifact import resource_phase_scope

    return resource_phase_scope(phase)


class Gate:
    def __init__(
        self,
        path,
        boot_id,
        *,
        expected_unit=EXPECTED_UNIT,
        observer=None,
        unit_reader=unit_snapshot,
        api_factory=Runtime,
    ):
        self.path, self.boot_id, self.expected_unit = Path(path), boot_id, expected_unit
        self.observe = mechanism().host_memory if observer is None else observer
        self.unit_reader, self.api_factory = unit_reader, api_factory
        self.loaded, self.api, self.sequence, self.bytes = False, None, 0, 0
        self.start = time.monotonic()
        require(
            not self.path.exists() and not self.path.is_symlink(), "Fresh native resource evidence"
        )
        if unit_reader is unit_snapshot:
            if os.getpgrp() != os.getpid() or os.getsid(0) != os.getpid():
                os.setsid()
        self.unit = self.unit_reader(expected_unit)
        validate_unit(self.unit, expected_unit)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        os.close(fd)

    def _append(self, row):
        blob = encode(row) + b"\n"
        require(len(blob) <= 32768 and self.bytes + len(blob) <= 16 * MIB, "Bounded native journal")
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
        try:
            require(os.fstat(fd).st_size == self.bytes, "Owned native journal unchanged")
            require(os.write(fd, blob) == len(blob), "Complete native journal write")
            os.fsync(fd)
        finally:
            os.close(fd)
        self.bytes += len(blob)
        self.sequence += 1

    def check(self, phase, scope):
        require(scope == phase_scope(phase) and self.sequence < 512, "Exact bounded native phase")
        row = {
            "schema_id": SCHEMA,
            "profile": PROFILE,
            "sequence": self.sequence,
            "phase": phase,
            "scope": scope,
            "loaded": self.loaded,
            "monotonic": time.monotonic(),
            "admitted": False,
            "error": None,
            "resident_credit_bytes": 0,
            "swap_credit_bytes": 0,
            "own_unused_cache_released": False,
            "native": None,
        }
        primary = None
        try:
            require(row["monotonic"] - self.start < 10800, "Native3hour wall bound")
            host = self.observe()
            mechanism().uma_host_identity(host, {"boot_id": self.boot_id})
            unit = self.unit_reader(self.expected_unit)
            validate_unit(unit, self.expected_unit)
            require(
                _unit_key(unit) == _unit_key(self.unit), "Same native unit inode/process/invocation"
            )
            # Cache release is process-local only; retain the full host reserve even before it.
            require(
                host["MemAvailable"] >= floor(host, loaded=True)["reserve_bytes"], "Host reserve"
            )
            if self.api is not None and release_phase(phase):
                self.api.release()
                row["own_unused_cache_released"] = True
                host = self.observe()
                mechanism().uma_host_identity(host, {"boot_id": self.boot_id})
            policy = floor(host, loaded=self.loaded)
            row.update(
                observation=host,
                unit=unit,
                **policy,
                original_train_would_refuse=host["MemAvailable"]
                < policy["old_train_required_bytes"],
            )
            require(
                host["MemAvailable"] >= policy["required_bytes"], "Explicit native UMA host floor"
            )
            if self.api is None and scope != "cpu":
                self.api = self.api_factory(host)
            if self.api is not None:
                row["native"] = self.api.snapshot(host)
                host = self.observe()
                mechanism().uma_host_identity(host, {"boot_id": self.boot_id})
                unit = self.unit_reader(self.expected_unit)
                validate_unit(unit, self.expected_unit)
                require(_unit_key(unit) == _unit_key(self.unit), "Same unit after native query")
                policy = floor(host, loaded=self.loaded)
                row.update(
                    observation=host,
                    unit=unit,
                    **policy,
                    original_train_would_refuse=host["MemAvailable"]
                    < policy["old_train_required_bytes"],
                )
                require(
                    host["MemAvailable"] >= policy["required_bytes"], "Fresh post-query host floor"
                )
                validate_native(row["native"], host, self.boot_id)
                row["old_cuda_free_would_refuse"] = row["native"]["cuda"]["free"] < CAP + GIB
            else:
                row["old_cuda_free_would_refuse"] = None
            row["admitted"] = True
        except BaseException as error:
            primary, row["error"] = error, {"type": type(error).__name__, "message": str(error)}
        try:
            self._append(row)
        except BaseException as secondary:
            if primary is not None:
                primary.add_note("Native resource evidence also failed: " + repr(secondary))
                raise primary from secondary
            raise
        if primary is not None:
            raise primary
        return row

    def mark_model_loaded(self, model, checker):
        require(not self.loaded and self.api is not None, "One genuine post-constructor transition")
        checker()
        require(
            getattr(model, "quantization", None) == "none"
            and getattr(model, "max_context", None) == 8192,
            "Checked BF16 native model",
        )
        self.loaded = True
        self.check("model-verified-loaded", "train")

    def evidence(self, *, steps, parent_present):
        blob = self.path.read_bytes()
        validate_evidence(
            blob,
            expected_unit=self.expected_unit,
            expected_boot_id=self.boot_id,
            steps=steps,
            parent_present=parent_present,
        )
        return {
            "path": self.path.name,
            "sha256": hashlib.sha256(blob).hexdigest(),
            "size_bytes": len(blob),
        }


def validate_evidence(blob, *, expected_unit, expected_boot_id, steps, parent_present):
    require(
        type(steps) is int
        and 1 <= steps <= 280
        and type(parent_present) is bool
        and len(blob) <= 16 * MIB,
        "Bounded native profile evidence",
    )
    rows = [json.loads(line) for line in blob.splitlines()]
    phases = ["before-any-project-framework-or-parent-payload"]
    if parent_present:
        phases += ["immediately-before-legacy-parent-capture-and-Torch"]
    phases += [
        "before-checkpoint-payload-hash-and-BF16-loader",
        "before-any-BF16-model-construction",
        "model-verified-loaded",
    ]
    if parent_present:
        phases += ["before-parent-injection"]
    phases += [
        f"before-BF16-processor-compile-{split}-{i}"
        for split, count in (("train", 142), ("development", 37))
        for i in range(count)
    ]
    phases += ["immediately-before-unchanged-train_actions-optimizer"]
    phases += ["before-native-train-forward"] * steps
    phases += ["before-native-development-forward"] * 37 + ["after-native-training"]
    require(
        len(rows) <= 512 and [r["phase"] for r in rows] == phases,
        "Complete exact142/37 compile/train/development/transition resource sequence",
    )
    key, native_seen, previous, device_key = None, False, -math.inf, None
    for i, row in enumerate(rows):
        require(
            set(row)
            == {
                "schema_id",
                "profile",
                "sequence",
                "phase",
                "scope",
                "loaded",
                "monotonic",
                "admitted",
                "error",
                "resident_credit_bytes",
                "swap_credit_bytes",
                "own_unused_cache_released",
                "native",
                "observation",
                "unit",
                "reserve_bytes",
                "required_bytes",
                "old_train_required_bytes",
                "original_train_would_refuse",
                "old_cuda_free_would_refuse",
            },
            "Exact native resource journal fields",
        )
        loaded = i >= phases.index("model-verified-loaded")
        host = row["observation"]
        mechanism().uma_host_identity(host, {"boot_id": expected_boot_id})
        policy = floor(host, loaded=loaded)
        validate_unit(row["unit"], expected_unit)
        current = _unit_key(row["unit"])
        key = current if key is None else key
        require(
            row["schema_id"] == SCHEMA
            and row["profile"] == PROFILE
            and row["sequence"] == i
            and row["scope"] == phase_scope(row["phase"])
            and row["loaded"] is loaded
            and row["admitted"] is True
            and row["error"] is None
            and row["resident_credit_bytes"] == row["swap_credit_bytes"] == 0
            and all(row[k] == v for k, v in policy.items())
            and row["original_train_would_refuse"]
            == (host["MemAvailable"] < policy["old_train_required_bytes"])
            and host["MemAvailable"] >= policy["required_bytes"]
            and type(row["monotonic"]) in (int, float)
            and math.isfinite(row["monotonic"])
            and previous <= row["monotonic"] <= rows[0]["monotonic"] + 10800
            and current == key,
            "Admitted sameboot/process/unit/profile/full-floor journal",
        )
        previous = row["monotonic"]
        native_seen = native_seen or row["scope"] != "cpu"
        require(
            (row["native"] is not None) == native_seen,
            "No missing/forged native transition evidence",
        )
        if native_seen:
            validate_native(row["native"], host, expected_boot_id)
            current_device = [row["native"]["identity"], row["native"]["allocator"]]
            device_key = current_device if device_key is None else device_key
            require(
                encode(current_device) == encode(device_key),
                "Same Driver/Torch identity and allocator cap",
            )
            require(
                row["old_cuda_free_would_refuse"] == (row["native"]["cuda"]["free"] < CAP + GIB),
                "Unmodified raw CUDA predicate",
            )
        require(
            row["own_unused_cache_released"] is release_phase(row["phase"]),
            "Exact process-local unused-cache release schedule",
        )
    return {
        "profile": PROFILE,
        "rows": len(rows),
        "loaded_transitions": 1,
        "train_forwards": steps,
        "development_forwards": 37,
        "startup_required_bytes": floor(rows[0]["observation"], loaded=False)["required_bytes"],
        "resident_required_bytes": floor(rows[-1]["observation"], loaded=True)["required_bytes"],
        "resource_only_not_whole_unit_exit_attestation": True,
    }
