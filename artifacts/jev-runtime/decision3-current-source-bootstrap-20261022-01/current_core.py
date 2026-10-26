#!/usr/bin/env python3
"""Current-source Decision3 CORE helpers; no controller or executable actor.

Explicit source fork: original cache/layout/resource formulas are retained.
Only source composition identity, root selection and profile differ.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import stat
import struct
import tempfile
import time
from pathlib import Path

ROOT = Path('/home/albert_hsieh/Gemma-4-12B-Unified-System-One')
SCRIPT = Path(__file__).resolve()
PUBLIC = ROOT / "artifacts/jev-browser/20261016-public-native-training-collection-07"
ADMISSION = ROOT / "artifacts/jev-browser/20261016-public-native-training-07-admission-01"
HUB = ROOT / "artifacts/jev-local-hf/hub"
OUTPUT_ROOT = ROOT / "artifacts/jev-d3-native"
PROFILE = "decision3-gx10-current-source-core-v1"
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
    "src/s1/jev_prompt.py": "2891ecbe94e5cbb01b7a2de6d7be9ed35ee55d26f904cfeb6b401416254b5f3c",
    "src/s1/__init__.py": "c9c7ba0ea4cc81609fcf1f0ff2a853243a173a994257e0b9dd19be1a2e6dd52e",
    "src/s1/contracts.py": "24c24c3ded9c35bfd263a4488bf88916a8960fa0b79283ccedbac232983f8a0a",
    "src/s1/errors.py": "e5ad1d9b7e700f07a704d74d3e4ade7156c7a9eaad304d3fe4296bad0dc8c997",
    "src/s1/backends.py": "90c084fa5523dea71b49cf140ecfda1fd2165a9468368624bde6161e2dbf67bf",
    "src/s1/jev.py": "9a2ab7f907c0ed51942d7d5c8a0c872399d5a6766c4afba8117129e6f90aeaa0",
    "src/s1/jev_generated.py": "c356e0eb3c046cb0787d7be0c5960803006231e8bdb414de644ae2167e72d849",
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

