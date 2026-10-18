#!/usr/bin/env python3
"""Six pinned Decision3 checkpoints on the same actual07 development22 wires.

Offline saved-state action replay, not model-owned browser success or holdout.
Review this file and pass its SHA before running. The controller is stdlib-only;
each CUDA arm has its own process, which must exit before the next arm starts.
Weights must already exist in the standard, fixed HF hub cache. No downloads,
Gemma helper, warmup, training, hierarchy, pruning or alternate model fallback.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import selectors
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
PUBLIC = ROOT / "artifacts/jev-browser/20261016-public-native-training-collection-07"
ADMISSION = ROOT / "artifacts/jev-browser/20261016-public-native-training-07-admission-01"
HUB = ROOT / "artifacts/jev-local-hf/hub"
OUTPUT_ROOT = ROOT / "artifacts/jev-d3-native"
PROFILE = "decision3-actual07-dev22-native-gpu-replay-v1"
MODEL_ORDER = ("d3", "d3-flash", "d3-mini", "d3-nano", "d3-lite", "d3-edge")
SECONDS, CONTEXT, MAX_BYTES, MAX_FILES = 20, 16384, 512 * 1024**2, 2048
LOG_BYTES, OUTER_SECONDS = 8 * 1024**2, 3600
BUFFER_BYTES = 2 * 1024**3
TARGETS = {"CLICK": "click_target", "SELECT": "select_target", "TYPE_TEXT": "type_text_target"}
INPUT_PINS = {
    "development/cases.jsonl": (
        "649f69ae49b43b000ad3a1338c75760173c1713bd75368e4737fef611bae8c9f",
        293804,
    ),
    "development/wire_requests.jsonl": (
        "055cb23f27f2c06cbe312b99c66704f8ef549363c3fb761f9d5e626cd684becd",
        296484,
    ),
}
ADMISSION_SHA = "bb799a48e8bbf1fb0af5afaf23cb38fb09bb67c3d26c833e96ba33bc34b0fcdb"
PROVENANCE_SHA = "d80e5106ab67375403f1b3b5ab9132ae7f212a134532d0d0d422ac044a648c80"
CLOSED_SHA = "d74f729460c648b8c88cf677bc379d1ae34fe7530a9651810e8af2825d449742"
ORDERED_WIRE_SHA = "37a298f3cc73e4d278edb2461f6f13cdd1c0861a8535893506532ce80ca3a1bc"
SOURCE_PINS = {
    "src/s1/__init__.py": "c9c7ba0ea4cc81609fcf1f0ff2a853243a173a994257e0b9dd19be1a2e6dd52e",
    "src/s1/contracts.py": "24c24c3ded9c35bfd263a4488bf88916a8960fa0b79283ccedbac232983f8a0a",
    "src/s1/errors.py": "e5ad1d9b7e700f07a704d74d3e4ade7156c7a9eaad304d3fe4296bad0dc8c997",
    "src/s1/backends.py": "90c084fa5523dea71b49cf140ecfda1fd2165a9468368624bde6161e2dbf67bf",
    "src/s1/jev.py": "9a2ab7f907c0ed51942d7d5c8a0c872399d5a6766c4afba8117129e6f90aeaa0",
    "src/s1/jev_generated.py": "582b520615cf123367436622f4e5ca6f5d94abb227449d866ec64cbb8adc9b3c",
    "src/s1/jev_hierarchy.py": "c85f8730eaba732c332b268a38e5556304622f3ba8e8f7be3b3a11892e583124",
    "src/s1/browser_text.py": "381299d3961afba4ba2a296b1ce9ee5e4b137f5494b14cb4291b70340462b577",
    "src/s1/jev_decision3_native.py": "67616bb70a8acdbf96b1d2e4c1407cf8cc2a24db51b5ca383e33b4d9a0488de9",
    "src/s1/evaluation/__init__.py": "127cf68d534aa63995896f51d47ff5015c1d8fcb3563914752aa130f5ff25f51",
    "src/s1/evaluation/contracts.py": "ead69fb44ba1a0221b9c73fc0c811774ed15b530cdf1ec56f99e7b98bff254d3",
    "src/s1/evaluation/decision3.py": "aaa21ab6f2ecf5ae0e00694b43755541dc75fc0310ba10d945ade2a4511add52",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def encode(value, *, canonical=False):
    return json.dumps(
        value, ensure_ascii=canonical, sort_keys=canonical, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def parse(blob):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def finite(value):
        number = float(value)
        require(math.isfinite(number), "nonfinite JSON number")
        return number

    return json.loads(
        blob,
        object_pairs_hook=pairs,
        parse_float=finite,
        parse_constant=lambda value: require(False, "nonfinite JSON " + value),
    )


def ordinary(path):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), "symbolic trusted path")
    require(stat.S_ISREG(path.stat().st_mode), "ordinary input required: " + str(path))
    return path


def pin(path):
    path = ordinary(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        result = hashlib.sha256()
        for block in iter(lambda: stream.read(1024**2), b""):
            result.update(block)
        after = os.fstat(stream.fileno())
    current = path.stat()
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        == (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        ),
        "input changed while hashing",
    )
    return {"sha256": result.hexdigest(), "size_bytes": before.st_size}


def read(path, expected=None, *, limit=16 * 1024**2):
    path = ordinary(path)
    require(path.stat().st_size <= limit, "JSON input byte bound")
    blob = path.read_bytes()
    require(
        len(blob) <= limit and (expected is None or digest(blob) == expected),
        "input hash/byte bound differs: " + str(path),
    )
    return parse(blob)


def fsync_dir(path):
    descriptor = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write(path, value, *, terminal=False):
    path = Path(path)
    blob = encode(value)
    require(len(blob) <= 16 * 1024**2, "single evidence JSON byte bound")
    if path.is_relative_to(OUTPUT_ROOT):
        root = OUTPUT_ROOT / path.relative_to(OUTPUT_ROOT).parts[0]
        total, count = 0, 0
        for folder, dirs, files in os.walk(root, followlinks=False):
            require(
                all(not (Path(folder) / name).is_symlink() for name in dirs),
                "linked evidence directory",
            )
            for name in files:
                entry = Path(folder) / name
                require(
                    not entry.is_symlink() and entry.is_file(), "ordinary evidence file required"
                )
                total += entry.stat().st_size
                count += 1
        old_size = path.stat().st_size if path.exists() else 0
        limit = MAX_BYTES if terminal else MAX_BYTES - 32 * 1024**2
        require(
            total - old_size + len(blob) <= limit and count < MAX_FILES - 4,
            "closed output bound; terminal evidence reserve retained",
        )
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fsync_dir(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def inventory(root, *, exclude=()):
    result, total = {}, 0
    for folder, dirs, files in os.walk(root, followlinks=False):
        require(all(not (Path(folder) / name).is_symlink() for name in dirs), "linked directory")
        for name in sorted(files):
            path = Path(folder) / name
            label = path.relative_to(root).as_posix()
            if label in exclude:
                continue
            require(
                path.stat().st_size <= MAX_BYTES - total and len(result) < MAX_FILES,
                "closed evidence inventory bound",
            )
            result[label] = pin(path)
            total += result[label]["size_bytes"]
    return dict(sorted(result.items()))


def authenticate_source(expected):
    require(pin(SCRIPT)["sha256"] == expected, "reviewed replay script differs")
    for label, wanted in SOURCE_PINS.items():
        require(pin(ROOT / label)["sha256"] == wanted, "production source differs: " + label)
    tree = ast.parse((ROOT / "src/s1/evaluation/decision3.py").read_bytes())
    identities = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "CHECKPOINTS" for target in node.targets
        )
    ]
    require(
        len(identities) == 1 and tuple(identities[0]) == MODEL_ORDER,
        "fixed six-model identities differ",
    )
    return identities[0]


def authenticate_inputs():
    receipt = read(ADMISSION / "receipt.json", ADMISSION_SHA)
    provenance = read(ADMISSION / "admission-provenance.json", PROVENANCE_SHA)
    require(
        receipt["stage"] == "complete"
        and receipt["admission_passed"] is True
        and receipt["verification_passed"] is True
        and receipt["closed_inventory_unchanged"] is True,
        "passed sealed admission required",
    )
    closed = inventory(PUBLIC)
    expected = provenance["closed_inventory"]
    require(
        closed == expected["files"] == receipt["full_closed_input_inventory"]["files"]
        and digest(encode(closed, canonical=True)) == CLOSED_SHA == expected["identity_sha256"]
        and len(closed) == 701
        and sum(p["size_bytes"] for p in closed.values()) == 105415940,
        "full701 closed input identity differs",
    )
    for label, (sha, size) in INPUT_PINS.items():
        require(closed[label] == {"sha256": sha, "size_bytes": size}, "fixed dev22 input differs")
    cases = [parse(line) for line in (PUBLIC / "development/cases.jsonl").read_bytes().splitlines()]
    wires = [
        parse(line)
        for line in (PUBLIC / "development/wire_requests.jsonl").read_bytes().splitlines()
    ]
    require(len(cases) == len(wires) == 22, "fixed22 population differs")
    ordered = hashlib.sha256()
    for case, wire in zip(cases, wires):
        source = encode(wire["native_payload"])
        require(
            case["split"] == "development"
            and case["id"] == wire["case_id"]
            and encode(case["request"]) == source
            and case["valid_next_targets"] == wire["valid_next_targets"]
            and digest(source) == wire["native_wire_sha256"],
            "development case/wire join differs",
        )
        ordered.update(source + b"\n")
    require(ordered.hexdigest() == ORDERED_WIRE_SHA, "ordered22 original native digest differs")
    return cases, wires


def cgroup_memory(groups, mountinfo):
    """Observe effective v1/v2 limits, including visible limiting ancestors."""
    controllers, seen = [], set()
    memberships = [line.split(":", 2) for line in groups.splitlines()]
    for line in mountinfo.splitlines():
        before, separator, after = line.partition(" - ")
        if not separator:
            continue
        fields, suffix = before.split(), after.split()
        kind = suffix[0]
        if kind not in {"cgroup", "cgroup2"}:
            continue
        memory_v1 = kind == "cgroup" and "memory" in suffix[-1].split(",")
        if kind == "cgroup" and not memory_v1:
            continue
        root = Path(fields[3].replace("\\040", " "))
        mount = Path(fields[4].replace("\\040", " "))
        for _, subsystems, member in memberships:
            if (kind == "cgroup2" and subsystems) or (
                memory_v1 and "memory" not in subsystems.split(",")
            ):
                continue
            membership = Path(member)
            # Mount roots may expose a subtree; namespace roots may already be
            # relative to that subtree. Try both, then retain visible ancestors.
            candidates = [mount / str(membership).lstrip("/")]
            if membership.is_relative_to(root):
                candidates.insert(0, mount / membership.relative_to(root))
            directory = next((path for path in candidates if path.is_dir()), mount)
            while directory.is_relative_to(mount):
                names = (
                    ("memory.max", "memory.current", "memory.peak")
                    if kind == "cgroup2"
                    else (
                        "memory.limit_in_bytes",
                        "memory.usage_in_bytes",
                        "memory.max_usage_in_bytes",
                    )
                )
                if directory not in seen and (directory / names[0]).is_file():
                    limit = (directory / names[0]).read_text().strip()
                    controllers.append(
                        {
                            "path": str(directory),
                            "version": kind,
                            "limit": None if limit == "max" else int(limit),
                            "current": int((directory / names[1]).read_text()),
                            "peak": int((directory / names[2]).read_text())
                            if (directory / names[2]).exists()
                            else None,
                        }
                    )
                    seen.add(directory)
                if directory == mount:
                    break
                directory = directory.parent
    return controllers


def resource_snapshot(torch=None):
    proc = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, _, value = line.partition(":")
        if name in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            proc[name] = int(value.split()[0]) * 1024
    groups = Path("/proc/self/cgroup").read_text()
    mounts = "\n".join(
        line
        for line in Path("/proc/self/mountinfo").read_text().splitlines()
        if " - cgroup" in line
    )
    # Also preserve the actual membership and mount mapping for independent audit.
    result = {
        "proc_meminfo": proc,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "cgroup_membership": groups,
        "cgroup_mountinfo": mounts,
        "cgroup_memory": cgroup_memory(groups, mounts),
        "cuda": None,
    }
    if torch is not None:
        free, total = torch.cuda.mem_get_info(0)
        properties = torch.cuda.get_device_properties(0)
        result["cuda"] = {
            "device": 0,
            "name": properties.name,
            "uuid": str(getattr(properties, "uuid", "")),
            "free_bytes": free,
            "total_bytes": total,
            "allocated_bytes": torch.cuda.memory_allocated(0),
            "reserved_bytes": torch.cuda.memory_reserved(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
        }
    return result


def memory_decision(snapshot, resident):
    proc = snapshot["proc_meminfo"]
    require(0 < proc["MemAvailable"] <= proc["MemTotal"], "invalid /proc memory snapshot")
    reserve = max(24 * 1024**3, (proc["MemTotal"] + 4) // 5)
    required = reserve + 2 * resident + BUFFER_BYTES
    # Keep the previous replay's full floor at EVERY gate, with no resident/swap credit.
    groups = snapshot.get("cgroup_memory")
    complete = (
        isinstance(groups, list)
        and bool(groups)
        and all(
            isinstance(group, dict)
            and "limit" in group
            and (group["limit"] is None or type(group["limit"]) is int and group["limit"] > 0)
            and type(group.get("current")) is int
            and group["current"] >= 0
            for group in groups
        )
    )
    available = proc["MemAvailable"] if complete else None
    if complete:
        for group in groups:
            if group["limit"] is not None:
                available = min(available, max(0, group["limit"] - group["current"]))
    return {
        "admitted": complete and available >= required,
        "cgroup_observation_complete": complete,
        "refusal_reason": None if complete else "cgroup_memory_observation_incomplete",
        "host_available_bytes": proc["MemAvailable"],
        "available_bytes": available,
        "required_available_bytes": required,
        "reserve_bytes": reserve,
        "resident_file_bytes": resident,
        "full_transient_bytes": resident,
        "framework_runtime_buffer_bytes": BUFFER_BYTES,
        "swap_or_resident_credit_bytes": 0,
        "basis": "unchanged conservative host floor; cgroup further restricts, not measured peak",
    }


class AdmissionError(RuntimeError):
    pass


def fatal_error(exc):
    return (
        isinstance(exc, AdmissionError)
        or isinstance(exc, OSError)
        and not isinstance(exc, TimeoutError)
        or not isinstance(exc, Exception)
    )


def campaign_fatal(exc):
    # An initial resource refusal can leave smaller independent arms runnable.
    # Any secondary fatal error or failed cleanup makes further execution unsafe.
    return (
        not isinstance(exc, Exception)
        or isinstance(exc, OSError)
        and not isinstance(exc, TimeoutError)
        or getattr(exc, "campaign_stop", False)
    )


def error_info(exc):
    result = {"type": type(exc).__name__, "message": str(exc)}
    if getattr(exc, "__notes__", None):
        result["notes"] = list(exc.__notes__)
    if exc.__cause__ is not None:
        result["cause"] = {"type": type(exc.__cause__).__name__, "message": str(exc.__cause__)}
    return result


class Gates:
    def __init__(self, directory, identity, boot, total):
        self.directory, self.identity, self.boot, self.total = directory, identity, boot, total
        self.sequence, self.torch = 0, None
        (directory / "gates").mkdir()

    def check(self, stage):
        snapshot = resource_snapshot(self.torch)
        require(
            snapshot["boot_id"] == self.boot and snapshot["proc_meminfo"]["MemTotal"] == self.total,
            "same host boot/total required",
        )
        decision = memory_decision(snapshot, self.identity["resident_file_bytes"])
        cuda = snapshot["cuda"]
        if cuda is not None:
            # Before load, reserve the real checkpoint plus a declared workspace
            # buffer; after load the unknown activation peak is only observed.
            floor = BUFFER_BYTES + (
                self.identity["resident_file_bytes"] if stage == "before-model-constructor" else 0
            )
            decision["cuda"] = {
                "free_bytes": cuda["free_bytes"],
                "required_free_bytes": floor,
                "admitted": cuda["free_bytes"] >= floor,
                "basis": "checkpoint plus2GiB before load; 2GiB workspace otherwise; no peak guarantee",
            }
            decision["admitted"] = decision["admitted"] and decision["cuda"]["admitted"]
        row = {
            "sequence": self.sequence,
            "stage": stage,
            "monotonic_ns": time.monotonic_ns(),
            "snapshot": snapshot,
            "decision": decision,
        }
        write(self.directory / "gates" / f"{self.sequence:04d}.json", row)
        self.sequence += 1
        if not decision["admitted"]:
            error = AdmissionError("full conservative memory floor refused; capability unmeasured")
            error.resource_gate = row
            raise error
        return row


def cache_file(snapshot, name):
    label = Path(name)
    require(
        not label.is_absolute() and ".." not in label.parts and len(label.parts) == 1,
        "flat immutable runtime file required",
    )
    path = (snapshot / label).resolve(strict=True)
    repo = snapshot.parent.parent
    require(
        path.is_relative_to(repo) or path.is_relative_to(HUB / "blobs"),
        "checkpoint symlink escapes exact repository/shared hub blobs",
    )
    return ordinary(path)


def tensor_header(path):
    with ordinary(path).open("rb") as stream:
        length_bytes = stream.read(8)
        require(len(length_bytes) == 8, "short safetensors length")
        length = struct.unpack("<Q", length_bytes)[0]
        require(2 <= length <= 4 * 1024**2, "bounded safetensors header required")
        header = parse(stream.read(length))
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    end, count = 0, 0
    for value in sorted(tensors.values(), key=lambda entry: entry["data_offsets"][0]):
        require(
            value["dtype"] in {"BF16", "F32"}
            and all(type(n) is int and n > 0 for n in value["shape"]),
            "unsupported tensor dtype/shape",
        )
        size = math.prod(value["shape"]) * {"BF16": 2, "F32": 4}[value["dtype"]]
        require(
            value["data_offsets"] == [end, end + size], "tensor offsets gap/overlap/byte mismatch"
        )
        end += size
        count += math.prod(value["shape"])
    require(path.stat().st_size == 8 + length + end, "tensor payload file length differs")
    return tensors, count


def backbone_layout(files, read_index):
    """Select only a manifest-authenticated indexed or single-file backbone.

    Callers authenticate every runtime file's full SHA/size first. Filesystem
    presence never invents an index or changes the declared checkpoint layout.
    """
    require(
        type(files) is dict
        and all(type(label) is str for label in files)
        and "readout.safetensors" in files,
        "Authenticated runtime file inventory/readout required",
    )
    backbone = {
        label
        for label in files
        if label.endswith(".safetensors") and label != "readout.safetensors"
    }
    index_file = "model.safetensors.index.json"
    if index_file in files:
        index = read_index(index_file)
        weight_map = index.get("weight_map") if type(index) is dict else None
        require(
            type(weight_map) is dict
            and bool(weight_map)
            and all(
                type(key) is str and bool(key) and type(value) is str and bool(value)
                for key, value in weight_map.items()
            ),
            "Nonempty string tensor-to-shard map required",
        )
        shards = sorted(set(weight_map.values()))
        require(set(shards) == backbone, "All backbone shards must equal index/manifest")
        return {
            "layout": "indexed_shards",
            "index_file": index_file,
            "shards": shards,
            "weight_map": weight_map,
        }
    require(
        backbone == {"model.safetensors"},
        "Unindexed checkpoint requires exactly one model.safetensors backbone",
    )
    return {
        "layout": "single_file",
        "index_file": None,
        "shards": ["model.safetensors"],
        "weight_map": None,
    }


def authenticate_cache(name, identity, output, gates, phase):
    snapshot = (
        HUB
        / ("models--" + identity["model_id"].replace("/", "--"))
        / "snapshots"
        / identity["revision"]
    )
    require(
        snapshot.is_dir() and not any(p.is_symlink() for p in (snapshot, *snapshot.parents)),
        "ordinary exact pinned HF snapshot required",
    )
    gates.check(phase + "-before-cache")
    manifest = read(cache_file(snapshot, "MODEL_MANIFEST.json"), identity["manifest_sha256"])
    require(
        manifest["schema"] == "d3-package-manifest/1"
        and manifest["model_name"] == name
        and manifest["repo_id"] == identity["model_id"],
        "publisher manifest identity differs",
    )
    excluded = {"LICENSE", "NOTICE", "requirements.txt", ".gitattributes"}
    wanted = {
        label: sha
        for label, sha in manifest["files_sha256"].items()
        if label not in excluded
        and not label.endswith(".md")
        and not label.startswith(("assets/", "LICENSES/"))
    }
    records = {}
    for label, sha in sorted(wanted.items()):
        if label.endswith(".safetensors"):
            gates.check(phase + "-before-tensor-hash-" + label)
        records[label] = pin(cache_file(snapshot, label))
        require(
            records[label] == {"sha256": sha, "size_bytes": manifest["files_bytes"][label]},
            "full runtime cache authentication failed: " + label,
        )
    layout = backbone_layout(records, lambda label: read(cache_file(snapshot, label)))
    shards, weight_map = layout["shards"], layout["weight_map"]
    union, total_parameters = {}, 0
    for shard in shards:
        entries, count = tensor_header(cache_file(snapshot, shard))
        require(
            bool(entries)
            and not set(entries).intersection(union)
            and all(v["dtype"] == "BF16" for v in entries.values()),
            "empty/duplicate/non-BF16 backbone tensor",
        )
        if weight_map is not None:
            require(
                {key for key, value in weight_map.items() if value == shard} == set(entries),
                "shard header/index keys differ",
            )
        union.update(entries)
        total_parameters += count
    head, head_parameters = tensor_header(cache_file(snapshot, "readout.safetensors"))
    width = identity["readout_tensor_bytes"] // (255 * 4)
    require(
        head
        == {
            "weight": {
                "dtype": "F32",
                "shape": [255, width],
                "data_offsets": [0, identity["readout_tensor_bytes"]],
            }
        },
        "fixed model-specific FP32 readout differs",
    )
    require(
        (weight_map is None or set(union) == set(weight_map))
        and head_parameters == 255 * width
        and total_parameters + head_parameters == identity["loaded_parameters"]
        and sum(records[label]["size_bytes"] for label in [*shards, "readout.safetensors"])
        == identity["resident_file_bytes"],
        "all-shard parameter/file byte count differs",
    )
    report = {
        "snapshot": str(snapshot),
        "revision": identity["revision"],
        "files": records,
        "manifest_sha256": identity["manifest_sha256"],
        "all_shards": shards,
        "backbone_layout": layout["layout"],
        "index_file": layout["index_file"],
        "backbone_parameters": total_parameters,
        "readout_shape": [255, width],
        "full_payload_sha256_authenticated": True,
    }
    write(output / (phase + "-cache.json"), report)
    return report


def project(original, alias, parser, native_payload):
    wire = copy.deepcopy(original)
    wire["model"] = alias
    request, preserved, source, _ = parser(wire, model_identity=alias)
    require(encode(preserved) == source == encode(wire), "native255 DTO changed wire")
    native = native_payload(request)
    expected = {
        "state": copy.deepcopy(original["state"]),
        "questions": {
            key: {
                "type": question["type"],
                "instructions": copy.deepcopy(question["instructions"]),
                "criteria": copy.deepcopy(question["criteria"]),
            }
            for key, question in original["questions"].items()
        },
    }
    require(
        encode(native) == encode(expected), "publisher projection changed state/head/order/value"
    )
    return request, wire, native


def select_action(original, normalized):
    operation = normalized["answers"]["operation"]["choice"]
    require(operation in original["questions"]["operation"]["criteria"], "unoffered operation")
    head = TARGETS.get(operation)
    target = normalized["answers"][head]["choice"] if head else None
    require(
        head is None or target in original["questions"][head]["criteria"], "unoffered active target"
    )
    return {"operation": operation, "target": target, "active_head": head, "action_executed": False}


def score_action(case, action):
    operation, target = action["operation"], action["target"]
    valid = case["valid_next_targets"]
    allowed = operation in valid and (
        target in valid[operation]
        if operation in TARGETS
        else valid[operation] == [] and target is None
    )
    return {
        "valid_next_targets_match": allowed,
        "exact_active_gold_match": operation == case["gold"]["operation"]
        and target == case["gold"].get(TARGETS.get(case["gold"]["operation"])),
    }


def summarize(cases, rows):
    """Keep the original22 denominator and separate unavailable/late raw actions."""
    by_id, groups = {row["case_id"]: row for row in rows}, {}
    require(len(by_id) == len(rows), "duplicate result case")
    for case in cases:
        row = by_id.get(case["id"], {})
        for group in (
            "all22",
            "complete" if case["gold"]["operation"] == "DONE" else "continuing",
            "gold_operation:" + case["gold"]["operation"],
            "state_role:" + case["state_role"],
        ):
            bucket = groups.setdefault(
                group,
                dict.fromkeys(
                    (
                        "planned",
                        "raw_returned",
                        "contract_valid",
                        "raw_action_valid",
                        "within20s_action_valid",
                        "late_valid",
                        "unavailable",
                    ),
                    0,
                ),
            )
            bucket["planned"] += 1
            bucket["raw_returned"] += row.get("raw_returned") is True
            bucket["contract_valid"] += row.get("contract_valid") is True
            scoring = row.get("offline_scoring")
            bucket["unavailable"] += scoring is None
            bucket["raw_action_valid"] += (
                scoring is not None and scoring["valid_next_targets_match"] is True
            )
            bucket["within20s_action_valid"] += row.get("within20s_action_valid") is True
            bucket["late_valid"] += row.get("status") == "late_valid_response"
    return groups


def verify_response(request, raw, name, expected_tokens, normalize):
    encode(raw)
    require(type(raw) is dict and raw.get("model") == name, "raw checkpoint identity differs")
    normalized = normalize(request, raw)
    require(list(raw["answers"]) == [q.id for q in request.questions], "raw head order differs")
    require(raw["usage"]["input_tokens"] == expected_tokens, "full native token usage differs")
    for question in request.questions:
        answer, labels = raw["answers"][question.id], question.labels()
        require(list(answer["probabilities"]) == labels, "raw probability label/order differs")
        values = list(answer["probabilities"].values())
        confidence = (
            1.0
            if len(values) == 1
            else max(
                0.0,
                min(1.0, 1 + sum(p * math.log(p) for p in values if p > 0) / math.log(len(values))),
            )
        )
        require(
            abs(answer["confidence"] - confidence) <= 1e-6
            and answer["choice"] == labels[max(range(len(values)), key=values.__getitem__)],
            "raw entropy confidence/first argmax differs",
        )
    return normalized


def annotate(value):
    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {"invalid_nonfinite_float": repr(value)}
    if isinstance(value, dict) and all(type(key) is str for key in value):
        return {key: annotate(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [annotate(item) for item in value]
    return {"invalid_native_type": type(value).__name__}


def replay_row(case, wire, name, backend, torch, output, gates, functions, *, clock=time.monotonic):
    # A CUDA fence precedes the timer; the completed forward is fenced inside it.
    torch.cuda.synchronize(0)
    started = clock()
    row = {
        "case_id": case["id"],
        "model": name,
        "status": "parsing",
        "raw_response": None,
        "raw_returned": False,
        "contract_valid": False,
        "selected_action": None,
        "offline_scoring": None,
        "within20s_action_valid": None,
        "helper_calls": 0,
        "backend_predict_calls": 0,
        "usd_cost": None,
        "usd_cost_reason": "cloud billing/rate is outside this offline worker; not assumed free",
    }
    path = output / (case["id"] + ".json")
    primary = None

    def secondary(exc, stage):
        nonlocal primary
        row.setdefault("secondary_errors", []).append(
            {"stage": stage, **error_info(exc), "fatal": fatal_error(exc)}
        )
        if fatal_error(exc):
            exc.campaign_stop = True
            if primary is not None:
                primary.campaign_stop = True
        if primary is None:
            primary = exc
        else:
            primary.add_note(stage + " also failed: " + repr(exc))
            if fatal_error(exc) and not fatal_error(primary):
                exc.__cause__ = primary
                primary = exc
        row["campaign_stop"] = campaign_fatal(primary)

    def checkpoint():
        row["elapsed_seconds"] = clock() - started
        write(path, row)

    def deadline():
        if clock() - started >= SECONDS:
            raise TimeoutError("full-call cooperative20s exceeded; no hard cancellation")

    def event(kind, evidence):
        # Save returned native output BEFORE the backend's post-forward deadline.
        if evidence.get("raw_response") is not None:
            row.update(raw_returned=True, raw_response=annotate(evidence["raw_response"]))
            try:
                encode(evidence["raw_response"])
                row["raw_encoding"] = "exact finite ordered JSON"
            except (TypeError, ValueError, OverflowError):
                row["raw_encoding"] = "malformed annotated native response"
        row["provider_evidence"] = annotate(evidence)
        row.setdefault("provider_events", []).append(kind)
        checkpoint()

    try:
        checkpoint()
        deadline()
        request, alias, native = project(wire["native_payload"], backend.name, *functions[:2])
        row.update(
            original_wire=copy.deepcopy(wire["native_payload"]),
            alias_wire=alias,
            publisher_payload=native,
            original_ordered_sha256=digest(encode(wire["native_payload"])),
        )
        gates.check("before-request-" + case["id"])
        deadline()
        prepared = backend.runtime.prepare(**copy.deepcopy(native))
        keys = list(native["questions"])
        require(
            not prepared.errors and prepared.keys == keys and list(prepared.sequences) == keys,
            "full native preflight failed/reordered heads",
        )
        prompts = {}
        for key in keys:
            tokens = list(prepared.sequences[key])
            require(
                tokens
                and len(tokens) <= CONTEXT
                and all(type(n) is int and n >= 0 for n in tokens),
                "native context invalid/exceeds16384; nothing truncated",
            )
            text = backend.runtime.text(native["state"], prepared.questions[key].rendered)
            prompts[key] = {
                "text": text,
                "text_utf8_sha256": digest(text.encode()),
                "token_ids": tokens,
                "token_ids_sha256": digest(encode(tokens)),
                "input_tokens": len(tokens),
            }
        row["native_prompts"] = prompts
        expected_tokens = sum(p["input_tokens"] for p in prompts.values())
        gates.check("before-forward-" + case["id"])
        checkpoint()
        deadline()
        # Decision3Backend uses the exact publisher public system_one call. Its
        # prepare repeats the saved preflight; both are charged to the same SLA.
        row["backend_predict_calls"] += 1
        backend.predict(request, deadline=deadline, evidence_sink=event)
    except BaseException as exc:
        primary = exc
        row["error"] = error_info(exc)
        row["campaign_stop"] = campaign_fatal(exc)
        if hasattr(exc, "decision3_provider_evidence"):
            try:
                event("provider_failed", exc.decision3_provider_evidence)
            except BaseException as secondary_exc:
                secondary(secondary_exc, "provider_failure_evidence")
    finally:
        try:
            torch.cuda.synchronize(0)
            row["cuda_synchronized_return_elapsed_seconds"] = clock() - started
            # A late raw response can still be audited/scored offline; it can
            # never regain within20s status after the deadline exception.
            if row["raw_returned"]:
                normalized = verify_response(
                    request, row["raw_response"], name, expected_tokens, functions[2]
                )
                action = select_action(wire["native_payload"], normalized)
                row.update(
                    normalized_response=normalized,
                    selected_action=action,
                    offline_scoring=score_action(case, action),
                    contract_valid=True,
                )
            checkpoint()
            row["full_call_within20s"] = clock() - started < SECONDS
            if row["contract_valid"]:
                row["within20s_action_valid"] = (
                    row["full_call_within20s"]
                    and primary is None
                    and row["offline_scoring"]["valid_next_targets_match"]
                )
                row["status"] = (
                    "late_valid_response"
                    if not row["full_call_within20s"]
                    else "valid_raw_with_request_error"
                    if primary is not None
                    else "valid_response_within20s"
                )
            else:
                row["status"] = "request_refused_or_incomplete"
            checkpoint()
            if row["full_call_within20s"] and clock() - started >= SECONDS:
                row.update(
                    full_call_within20s=False,
                    within20s_action_valid=False,
                    status="late_valid_response",
                )
                checkpoint()
        except BaseException as exc:
            row["finalization_error"] = error_info(exc)
            secondary(exc, "row_finalization")
            row.update(
                status="request_refused_or_incomplete",
                full_call_within20s=False,
                within20s_action_valid=None,
            )
            try:
                checkpoint()
            except BaseException as persistence_exc:
                secondary(persistence_exc, "failed_row_persistence")
    if primary is not None and fatal_error(primary):
        primary.replay_row = row
        raise primary
    return row


def finish_worker(state, output, backend, torch, started, *, primary=None, cases=(), wires=()):
    """Retain primary errors and cleanup failures before returning a failed arm."""

    def failed(exc, stage):
        state.setdefault("cleanup_errors", []).append({"stage": stage, **error_info(exc)})
        state.update(stage="failed", campaign_stop=True, stop_reason=stage)
        if primary is not None:
            primary.add_note(stage + " also failed: " + repr(exc))

    if torch is not None:
        try:
            torch.cuda.synchronize(0)
            state["final_resources"] = resource_snapshot(torch)
        except BaseException as exc:
            failed(exc, "final_cuda_resources")
    if backend is not None:
        try:
            backend.close()
        except BaseException as exc:
            failed(exc, "backend_close")
    state["elapsed_seconds"] = time.monotonic() - started
    state["unrun_case_ids"] = [
        wire["case_id"]
        for wire in wires
        if wire["case_id"] not in {row["case_id"] for row in state["rows"]}
    ]
    if cases:
        state["summary"] = summarize(cases, state["rows"])
    if primary is not None and state["errors"]:
        state["errors"][0].update(error_info(primary))
    try:
        write(output / "receipt.json", state, terminal=True)
    except BaseException as exc:
        failed(exc, "terminal_receipt_persistence")
        # A failed receipt cannot authorize another model. Preserve the first
        # fatal/control exception object if present; otherwise propagate the I/O.
        if primary is not None and fatal_error(primary):
            primary.worker_receipt = state
            raise primary
        exc.worker_receipt = state
        if primary is not None:
            raise exc from primary
        raise


def worker(args):
    output = args.output_dir
    output.mkdir()
    started, backend, torch, primary = time.monotonic(), None, None, None
    state = {
        "profile": PROFILE,
        "stage": "initializing",
        "model": args.worker,
        "rows": [],
        "errors": [],
        "helper_calls": 0,
        "browser_calls": 0,
        "optimizer_updates": 0,
        "source_sha256": args.source_sha256,
        "task_score": None,
        "warmup_calls": 0,
        "campaign_stop": False,
    }
    try:
        write(output / "receipt.json", state)
        identities = authenticate_source(args.source_sha256)
        identity = identities[args.worker]
        snapshot = resource_snapshot()
        gates = Gates(output, identity, args.expected_boot_id, args.expected_memtotal)
        state["initial_resource_snapshot"] = snapshot
        gates.check("before-input-and-framework")
        cases, wires = authenticate_inputs()
        state["stage"] = "authenticating_cache"
        write(output / "receipt.json", state)
        state["cache_before"] = authenticate_cache(args.worker, identity, output, gates, "initial")
        os.environ.update(
            HF_HUB_CACHE=str(HUB),
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
            HF_ENABLE_PARALLEL_LOADING="false",
            HF_PARALLEL_LOADING_WORKERS="1",
            PYTHONDONTWRITEBYTECODE="1",
            HF_MODULES_CACHE=str(output / "hf-modules"),
        )
        blocked = []

        def deny_network(event, arguments):
            if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
                blocked.append(event)
                raise RuntimeError("network forbidden in offline GPU replay")

        sys.addaudithook(deny_network)
        sys.path.insert(0, str(ROOT / "src"))
        import torch as actual_torch

        from s1.evaluation.decision3 import (
            CHECKPOINTS,
            Decision3Backend,
            decision3_spec,
            native_payload,
            normalize_native_response,
        )
        from s1.jev_decision3_native import parse_native_payload

        torch = actual_torch
        for label in SOURCE_PINS:
            module_name = label.removeprefix("src/").removesuffix(".py").replace("/", ".")
            if module_name.endswith(".__init__"):
                module_name = module_name.removesuffix(".__init__")
            module = sys.modules.get(module_name)
            require(
                module is not None and Path(module.__file__).resolve() == ROOT / label,
                "imported production module origin differs: " + module_name,
            )
        require(
            CHECKPOINTS == identities
            and torch.cuda.is_available()
            and torch.cuda.device_count() == 1,
            "fixed checkpoint identities and exactly one real CUDA GPU required",
        )
        gates.torch = torch
        gates.check("before-model-constructor")
        state["packages"] = {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "accelerate",
                "safetensors",
                "huggingface_hub",
                "torchvision",
                "pillow",
                "einops",
                "pydantic",
                "httpx",
            )
        }
        for name in ("flash-linear-attention", "causal-conv1d", "triton"):
            try:
                state["packages"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                state["packages"][name] = None
        state["cuda_runtime"] = {
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        }
        state["stage"] = "model_loading"
        write(output / "receipt.json", state)
        backend = Decision3Backend(
            decision3_spec(args.worker, device="cuda:0"),
            expected_boot_id=args.expected_boot_id,
            expected_memtotal=args.expected_memtotal,
        )
        runtime, width = backend.runtime, identity["readout_tensor_bytes"] // (255 * 4)
        require(
            all(
                p.device == torch.device("cuda:0") and p.dtype == torch.bfloat16
                for p in runtime.backbone.parameters()
            )
            and runtime.readout.device == torch.device("cuda:0")
            and runtime.readout.dtype == torch.float32
            and list(runtime.readout.shape) == [255, width],
            "actual CUDA placement/BF16 backbone/FP32 readout differs",
        )
        require(
            sum(p.numel() for p in runtime.backbone.parameters()) + runtime.readout.numel()
            == identity["loaded_parameters"],
            "actual loaded parameter count differs",
        )
        provenance = runtime.provenance()
        require(
            provenance["model_name"] == args.worker
            and provenance["repo_id"] == identity["model_id"]
            and provenance["pooling"] == "last",
            "actual publisher provenance identity differs",
        )
        torch.cuda.synchronize(0)
        state["runtime"] = {
            "provenance": provenance,
            "runtime_info": runtime.runtime_info(),
            "kernels": runtime.kernels,
            "fast_enabled": runtime.fast is not None,
            "fast_skipped": runtime.fast_skipped,
            "provider_telemetry": backend.telemetry(),
        }
        state["resource_after_load"] = resource_snapshot(torch)
        state["stage"] = "running"
        write(output / "receipt.json", state)
        functions = (parse_native_payload, native_payload, normalize_native_response)
        for case, wire in zip(cases, wires):
            state["rows"].append(
                replay_row(case, wire, args.worker, backend, torch, output, gates, functions)
            )
            write(output / "receipt.json", state)
        state["cache_after"] = authenticate_cache(args.worker, identity, output, gates, "final")
        authenticate_inputs()
        authenticate_source(args.source_sha256)
        state["network_attempts_blocked"] = blocked
        require(not blocked, "offline worker attempted network access")
        state["stage"] = "complete"
    except BaseException as exc:
        primary = exc
        if hasattr(exc, "replay_row"):
            state["rows"].append(exc.replay_row)
        state["failure_stage"] = state["stage"]
        state["stage"] = "failed"
        state["errors"].append({**error_info(exc), "traceback": traceback.format_exc()[-8192:]})
        if campaign_fatal(exc) or getattr(exc, "replay_row", {}).get("campaign_stop"):
            state.update(campaign_stop=True, stop_reason="fatal_worker_error")
    finally:
        finish_worker(
            state,
            output,
            backend,
            torch,
            started,
            primary=primary,
            cases=locals().get("cases", ()),
            wires=locals().get("wires", ()),
        )
    return 0 if state["stage"] == "complete" else 1


def terminate(child):
    if child is None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            continue
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            return
    raise RuntimeError("owned worker process group exit unconfirmed; next model forbidden")


def drain(child, output):
    """Bound both native logs while waiting; kill is left to the caller's finally."""
    selector, total = selectors.DefaultSelector(), 0
    for stream, name in ((child.stdout, "stdout.log"), (child.stderr, "stderr.log")):
        selector.register(stream, selectors.EVENT_READ, name)
    deadline = time.monotonic() + OUTER_SECONDS
    try:
        with (
            (output / "stdout.log").open("xb") as stdout,
            (output / "stderr.log").open("xb") as stderr,
        ):
            logs = {"stdout.log": stdout, "stderr.log": stderr}
            while selector.get_map():
                require(time.monotonic() < deadline, "outer3600s owned worker deadline")
                for key, _ in selector.select(timeout=1):
                    blob = os.read(key.fileobj.fileno(), min(65536, LOG_BYTES - total + 1))
                    if not blob:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        continue
                    allowed = min(len(blob), LOG_BYTES - total)
                    logs[key.data].write(blob[:allowed])
                    logs[key.data].flush()
                    total += allowed
                    require(allowed == len(blob), "native output log cap exceeded")
        return child.wait(timeout=5)
    finally:
        selector.close()


def controller(args):
    require(
        not {"torch", "transformers", "numpy", "playwright"}.intersection(sys.modules),
        "controller must remain framework/browser free",
    )
    identities = authenticate_source(args.source_sha256)
    output = args.output_dir.absolute()
    require(
        output.parent == OUTPUT_ROOT
        and not output.exists()
        and not any(p.is_symlink() for p in (output, *output.parents)),
        "fresh confined output required",
    )
    output.mkdir()
    fsync_dir(output.parent)
    started, child = time.monotonic(), None
    state = {
        "profile": PROFILE,
        "stage": "initializing",
        "arms": [],
        "errors": [],
        "model_order": list(MODEL_ORDER),
        "identities": identities,
        "source_sha256": args.source_sha256,
        "source_pins": SOURCE_PINS,
        "input_pins": INPUT_PINS,
        "budget_seconds": SECONDS,
        "helper_calls": 0,
        "warmup_calls": 0,
        "browser_calls": 0,
        "optimizer_updates": 0,
        "task_score": None,
        "planned_requests": 132,
        "planned_requests_per_model": 22,
        "scope": "22 known-development teacher states/model; offline actions, not rollout/generalization",
        "deadline_policy": "full-call wall including preparation/evidence/persistence; CUDA synchronized; late raw retained; no hard20s cancellation",
    }
    try:
        write(output / "receipt.json", state)
        cases, _ = authenticate_inputs()
        snapshot = resource_snapshot()
        state["initial_resources"] = snapshot
        previous_uuid = None
        for name in MODEL_ORDER:
            child = None
            arm = {"model": name, "stage": "initializing", "errors": []}
            state["arms"].append(arm)
            control = output / (name + "-controller")
            control.mkdir()
            gates = Gates(
                control, identities[name], snapshot["boot_id"], snapshot["proc_meminfo"]["MemTotal"]
            )
            prelaunch_gate_pending = True
            try:
                gates.check("before-owned-child")
                prelaunch_gate_pending = False
                authenticate_source(args.source_sha256)
                command = [
                    sys.executable,
                    "-B",
                    "-s",
                    str(SCRIPT),
                    "--source-sha256",
                    args.source_sha256,
                    "--worker",
                    name,
                    "--output-dir",
                    str(output / name),
                    "--expected-boot-id",
                    snapshot["boot_id"],
                    "--expected-memtotal",
                    str(snapshot["proc_meminfo"]["MemTotal"]),
                ]
                arm.update(command=command, stage="launching")
                write(output / "receipt.json", state)
                environment = {
                    **os.environ,
                    "HF_HUB_CACHE": str(HUB),
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                }
                child = subprocess.Popen(
                    command,
                    env=environment,
                    cwd=ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                )
                arm.update(pid=child.pid, stage="running", started_monotonic_ns=time.monotonic_ns())
                write(output / "receipt.json", state)
                arm["returncode"] = drain(child, control)
            except Exception as exc:
                arm["errors"].append({"type": type(exc).__name__, "message": str(exc)})
                if isinstance(exc, AdmissionError) and prelaunch_gate_pending and child is None:
                    gate = getattr(exc, "resource_gate", None)
                    require(
                        isinstance(gate, dict)
                        and gate.get("sequence") == 0
                        and gate.get("stage") == "before-owned-child"
                        and gate.get("decision", {}).get("admitted") is False,
                        "typed initial resource refusal requires its durable failed gate",
                    )
                    gate_path = control / "gates/0000.json"
                    require(
                        read(gate_path) == gate, "initial resource refusal gate was not persisted"
                    )
                    arm["initial_resource_refusal"] = {
                        "gate": gate,
                        "gate_file": pin(gate_path),
                        "availability_limit": "full conservative floor refused before spawn; capability unmeasured",
                    }
            finally:
                terminate(child)  # Never launch another arm before confirmed process-group exit.
                arm["worker_spawned"] = child is not None
                arm["owned_process_group_exited"] = True
                arm["ended_monotonic_ns"] = time.monotonic_ns()
            path = output / name / "receipt.json"
            if path.exists():
                arm["worker_receipt"] = read(path)
                worker_state = arm["worker_receipt"]
                arm["summary"] = worker_state.get("summary", summarize(cases, worker_state["rows"]))
                arm["stage"] = (
                    "complete"
                    if worker_state["stage"] == "complete" and arm.get("returncode") == 0
                    else "failed"
                )
                if "resource_after_load" in worker_state:
                    uuid = worker_state["resource_after_load"]["cuda"]["uuid"]
                    require(
                        uuid and (previous_uuid is None or uuid == previous_uuid),
                        "same physical GPU UUID required",
                    )
                    previous_uuid = uuid
            else:
                arm["stage"] = (
                    "unavailable_resource_admission"
                    if arm.get("initial_resource_refusal")
                    else "unavailable_no_worker_receipt"
                )
                arm["summary"] = summarize(cases, [])
            write(output / "receipt.json", state)
            if arm.get("initial_resource_refusal"):
                require(
                    not path.exists() and child is None,
                    "refused prelaunch arm unexpectedly has a worker",
                )
                continue
            worker_state = arm.get("worker_receipt", {})
            if (
                worker_state.get("campaign_stop")
                or worker_state.get("cleanup_errors")
                or worker_state.get("stage") not in {"complete", "failed"}
                or arm.get("returncode") not in {0, 1}
            ):
                state.update(
                    stage="failed",
                    campaign_stop=True,
                    stop_reason=worker_state.get(
                        "stop_reason", "worker_exit_without_trusted_terminal_receipt"
                    ),
                )
                break
        authenticate_inputs()
        authenticate_source(args.source_sha256)
        if not state.get("campaign_stop"):
            state["stage"] = (
                "complete"
                if all(arm["stage"] == "complete" for arm in state["arms"])
                else "incomplete"
            )
    except BaseException as exc:
        state["failure_stage"] = state["stage"]
        state["stage"] = "failed"
        state["errors"].append(
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc()[-8192:],
            }
        )
    finally:
        try:
            terminate(child)
        except BaseException as exc:
            state.setdefault("cleanup_errors", []).append(repr(exc))
            state["stage"] = "failed"
        state["unrun_models"] = [
            name for name in MODEL_ORDER if name not in {arm["model"] for arm in state["arms"]}
        ]
        state["elapsed_seconds"] = time.monotonic() - started
        write(output / "receipt.json", state, terminal=True)
        # This last file intentionally excludes itself; no writes follow sealing.
        write(
            output / "closed-inventory.json",
            {
                "excludes": ["closed-inventory.json"],
                "files": inventory(output, exclude=("closed-inventory.json",)),
            },
            terminal=True,
        )
    return 0 if state["stage"] == "complete" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--worker", choices=MODEL_ORDER, help=argparse.SUPPRESS)
    parser.add_argument("--expected-boot-id", help=argparse.SUPPRESS)
    parser.add_argument("--expected-memtotal", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        require(
            args.expected_boot_id
            and args.expected_memtotal
            and args.output_dir.is_absolute()
            and args.output_dir.name == args.worker
            and args.output_dir.parent.parent == OUTPUT_ROOT
            and not args.output_dir.exists(),
            "fresh owned worker output/host binding required",
        )
        return worker(args)
    return controller(args)


if __name__ == "__main__":
    raise SystemExit(main())
