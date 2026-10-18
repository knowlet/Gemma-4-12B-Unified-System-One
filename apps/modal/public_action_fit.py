"""Explicit public07 Modal fit: named inputs, public HF weights, one H200.

First review the inventory without importing Modal or uploading anything:
  .venv/bin/python apps/modal/public_action_fit.py --inventory-only NEW.json
After reviewing that manifest and the source, use the Modal local entrypoint
with --run, --manifest NEW.json and a fresh --output directory. Local derived
training artifacts are classified explicitly; they are not claimed public HF.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shlex
import signal
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath

ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
SCRIPT = ROOT / "apps/modal/public_action_fit.py"
OVERLAY = ROOT / "artifacts/jev-runtime/overlay"
INTERPRETER = "/home/albert_hsieh/ComfyUI/.venv/bin/python"
RUNTIME = ROOT / "artifacts/jev-runtime"
REFERENCE = RUNTIME / "public-native-action-compiler-07-root-20261016-01/actual-compile-01"
PREFLIGHT = RUNTIME / "public-native-action-fit-preflight-07-root-20261016-01"
ADMISSION = ROOT / "artifacts/jev-browser/20261016-public-native-training-07-admission-01"
PUBLIC = ROOT / "artifacts/jev-browser/20261016-public-native-training-collection-07"
PARENT = ROOT / "artifacts/jev-browser/20261015-action-lora-counterfactual-fit-01"
HANDOFF = RUNTIME / "public-native-action-actual07-fit-20261016-01/source-handoff-03.json"
HANDOFF_SHA = "c4601d28519da5563586a84f5d17fc08620a64ce3e80baec529b43f27f5f20d8"
ART_SHA = "dbd1eb660d3bebc8dbc0b3f7e58b7bac66f5ab0a4ea9d687dc463ce830f1672d"
FIT_SHA = "a38bbab599f6f8ec8117d1b912e02913330ea7bb11cd67e4e6793ed80da01ca7"
CLOSURE_SHA = "ff87ceb2563bdee706c4b8cc1064a4682534be2925372c0c02c7ccd2d04e7bbc"
FROZEN_SHA = "a66125c589b7408337685d06daf4d0eb80265b07e7e9a55786442562389b87d3"
COMPILER_SHA = "a8a5ebdefccc4ef1be01e673c36d36452efd30368b02921b81aae9cabd096b82"
MODEL = "knowlet/Gemma-4-12B-Unified-System-One"
REVISION = "a66f836b56605039fe040f330180e336d19b3362"
HUB = ROOT / "artifacts/jev-local-hf/hub"
CACHE = HUB / ("models--" + MODEL.replace("/", "--")) / "snapshots" / REVISION
DECLARED_MOUNTS = {"/weights", "/input", "/export"}
BASE = {
    "sha256": "11144621122123135eb79c9cfd0466a2c51f230109124aefe401cdbec8c71a53",
    "size_bytes": 23919549408,
}
GIB, MAX_FILES, MAX_LOG = 1024**3, 10000, 16 * 1024**2
EXTRA_PACKAGES = {"peft": "0.21.0", "fastapi": "0.115.12", "uvicorn": "0.34.3", "httpx": "0.28.1"}
META = {
    "admission-receipt.json": (
        ADMISSION / "receipt.json",
        "bb799a48e8bbf1fb0af5afaf23cb38fb09bb67c3d26c833e96ba33bc34b0fcdb",
    ),
    "admission-provenance.json": (
        ADMISSION / "admission-provenance.json",
        "d80e5106ab67375403f1b3b5ab9132ae7f212a134532d0d0d422ac044a648c80",
    ),
    "admission-plan.json": (
        ADMISSION / "plan.json",
        "47c9ab6d2607e41598abad7ef7165fcb6577ab5662cfde7c99ba04e1f0f963ca",
    ),
    "preflight-receipt.json": (
        PREFLIGHT / "stage-01/preflight-receipt.json",
        "8e017a703662f64530599657426fe3a5ef7134af3b8df0daca7aeb613057dd7c",
    ),
    "preflight-root-receipt.json": (
        PREFLIGHT / "root-receipt-01.json",
        "2349f04ae8bd8825270603a84e5106b3dccc80914baa804718cae9f0034c3d79",
    ),
    "syntax-audit.json": (
        RUNTIME / "public-native-observation-gap-audit-20261016-01/actual07-H02-01/report.json",
        "e6216d7e4d335c2b76da6942b1c2226fb56eb7d2ba8fa5128bc76f3fb8f0c0bc",
    ),
    "compiler-root-receipt.json": (
        REFERENCE.parent / "root-receipt-01.json",
        "95b061b62c24d2711ab700562603328a95ed656ab5aac0887f2a6d2e8371716a",
    ),
    "compiler-audit.json": (
        RUNTIME / "public-native-action-compiler-07-postrun-audit-20261016-01/report-01.json",
        "099dc2561e7595813c29d1dcb2ed40e40e08958aa18146a9d4dc80dc2476c326",
    ),
}
BASE_METADATA = {
    "artifact-manifest.json": {
        "sha256": "f38d7f50060eabcbed6f73d16404ad42eb6ba1bddc79b68052afc13cab8598eb",
        "size_bytes": 2575,
    },
    "chat_template.jinja": {
        "sha256": "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4",
        "size_bytes": 18683,
    },
    "config.json": {
        "sha256": "b50e760a9107b629e1a140ab7a4392e875c37baefee022f27c98b90b9d947cf0",
        "size_bytes": 4953,
    },
    "generation_config.json": {
        "sha256": "040d0ccb9e10df02df9738424c1ce6d20d5d15b09362da922093ccd97573dbb4",
        "size_bytes": 255,
    },
    "processor_config.json": {
        "sha256": "6b938e76555b3e9946890770e1abcd442a4718f34041a58e8139dc8ad34545c9",
        "size_bytes": 1382,
    },
    "s1_config.json": {
        "sha256": "322ffa1c6bfa7c4069da43265be1e79071eee6fbcff07cd4a74f9ed04774b962",
        "size_bytes": 367,
    },
    "tokenizer.json": {
        "sha256": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
        "size_bytes": 32169626,
    },
    "tokenizer_config.json": {
        "sha256": "794a39f8330ce05020774c70c091225bc5f031b9cacf41fc20e52eb54b4b52d8",
        "size_bytes": 3736,
    },
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encode(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def parse(blob):
    def pairs(items):
        result = {}
        for name, value in items:
            require(name not in result, "Duplicate JSON key")
            result[name] = value
        return result

    return json.loads(
        blob,
        object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")),
    )


def ordinary(path):
    path = Path(path).absolute()
    require(
        ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "Ordinary path required",
    )
    return path


def capture_mount(alias):
    """Resolve only a declared provider boundary; descendants stay ordinary."""
    require(alias in DECLARED_MOUNTS, "Undeclared provider mount")
    root = ordinary(Path(alias).resolve(strict=True))
    require(root.is_dir(), "Provider mount must resolve to a directory")
    before = root.stat()
    result = {
        "alias": alias,
        "root": str(root),
        "device": before.st_dev,
        "inode": before.st_ino,
    }
    require(
        Path(alias).resolve(strict=True) == root
        and (root.stat().st_dev, root.stat().st_ino) == (before.st_dev, before.st_ino),
        "Provider mount changed during capture",
    )
    return result


def mount_root(binding):
    require(capture_mount(binding["alias"]) == binding, "Provider mount alias/identity changed")
    return ordinary(binding["root"])


def remote_cache(config):
    hub = mount_root(config["mounts"]["/weights"])
    return hub, ordinary(hub / ("models--" + MODEL.replace("/", "--")) / "snapshots" / REVISION)


def named_input(config, index=None):
    require(
        re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", config["run_id"]) is not None,
        "Bounded named input run",
    )
    root = mount_root(config["mounts"]["/input"]) / config["run_id"]
    if index is None:
        return ordinary(root / "manifest.json")
    require(type(index) is int and 0 <= index < MAX_FILES, "Bounded flat named input index")
    return ordinary(root / "files" / f"{index:05d}")


def relative(name):
    path = PurePosixPath(name)
    require(
        type(name) is str
        and str(path) == name
        and bool(name)
        and not path.is_absolute()
        and ".." not in path.parts
        and not any(p in {".git", ".aws", ".ssh", ".codex", ".agents"} for p in path.parts),
        "Confined named task file required",
    )
    return path


def read(path, limit=64 * 1024**2):
    path = ordinary(path)
    require(path.is_file() and path.stat().st_size <= limit, "Bounded ordinary file required")
    with path.open("rb") as stream:
        blob = stream.read(limit + 1)
    require(len(blob) <= limit, "Read byte cap")
    return blob


def pin(path, limit=64 * 1024**2):
    path = ordinary(path)
    before = path.stat()
    require(
        stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= limit, "Bounded named file required"
    )
    digest, count = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024**2):
            count += len(chunk)
            require(count <= before.st_size, "File grew during read")
            digest.update(chunk)
        after = os.fstat(stream.fileno())
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        and count == before.st_size,
        "File changed during read",
    )
    return {"sha256": digest.hexdigest(), "size_bytes": count}


def create(path, blob):
    path = ordinary(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())


def stream_copy(source, destination, expected):
    """Copy one pinned ordinary file without holding a checkpoint in RAM."""
    source, destination = ordinary(source), ordinary(destination)
    require(
        set(expected) == {"sha256", "size_bytes"}
        and type(expected["size_bytes"]) is int
        and 0 <= expected["size_bytes"] <= max(GIB, BASE["size_bytes"])
        and type(expected["sha256"]) is str
        and re.fullmatch(r"[0-9a-f]{64}", expected["sha256"]) is not None,
        "Bounded exact streamed file pin",
    )
    before = source.stat()
    require(
        stat.S_ISREG(before.st_mode) and before.st_size == expected["size_bytes"],
        "Pinned copy source size/type differs",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest, count = hashlib.sha256(), 0
    with source.open("rb") as incoming, destination.open("xb") as outgoing:
        while chunk := incoming.read(1024**2):
            count += len(chunk)
            require(count <= expected["size_bytes"], "Streamed copy grew")
            require(outgoing.write(chunk) == len(chunk), "Short streamed copy")
            digest.update(chunk)
        after = os.fstat(incoming.fileno())
        outgoing.flush()
        os.fsync(outgoing.fileno())
    current = source.stat()
    require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        == (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )
        and {"sha256": digest.hexdigest(), "size_bytes": count} == expected
        and pin(destination, expected["size_bytes"]) == expected,
        "Streamed copy source/content changed",
    )


def stage_weights(config, output):
    """Only the remote fit child fills an empty, ordinary fixed HF cache."""
    hub, cache = remote_cache(config)
    ordinary(HUB)
    require(not HUB.exists() or not any(HUB.iterdir()), "Fixed local HF cache must be empty")
    for name, expected in BASE_METADATA.items():
        source = ordinary((cache / name).resolve(strict=True))
        require(source.is_relative_to(hub), "Provider metadata escaped its cache")
        stream_copy(source, CACHE / name, expected)
    admit(output / "preflight-gates.jsonl", "load", config["boot_id"])
    source = ordinary((cache / "model.safetensors").resolve(strict=True))
    require(source.is_relative_to(hub), "Provider checkpoint escaped its cache")
    stream_copy(source, CACHE / "model.safetensors", BASE)
    admit(output / "preflight-gates.jsonl", "load", config["boot_id"])
    require(mount_root(config["mounts"]["/weights"]) == hub, "Provider weight mount drift")
    create(
        output / "cache-staging.json",
        encode({"metadata": BASE_METADATA, "checkpoint": BASE, "fixed_cache": str(CACHE)}),
    )


def complete_cgroup(group):
    return (
        set(group) == {"memory.max", "memory.current"}
        and type(group["memory.current"]) is int
        and group["memory.current"] >= 0
        and (
            group["memory.max"] is None
            or type(group["memory.max"]) is int
            and group["memory.max"] > 0
        )
    )


def cgroup_memory(memberships, mountinfo):
    """Read v1 memory or v2 membership and every visible controlling ancestor."""
    members = {1: [], 2: []}
    for line in memberships.splitlines():
        fields = line.split(":", 2)
        if len(fields) == 3:
            if fields[0] == "0" and fields[1] == "":
                members[2].append(fields[2])
            elif "memory" in fields[1].split(","):
                members[1].append(fields[2])
    observations, seen = [], set()
    for line in mountinfo.splitlines():
        before, separator, after = line.partition(" - ")
        mount_fields = after.split()
        if not separator or len(mount_fields) < 3:
            continue
        if mount_fields[0] == "cgroup2":
            version = 2
            files = {"memory.max": "memory.max", "memory.current": "memory.current"}
        elif mount_fields[0] == "cgroup" and "memory" in mount_fields[2].split(","):
            version = 1
            files = {
                "memory.max": "memory.limit_in_bytes",
                "memory.current": "memory.usage_in_bytes",
            }
        else:
            continue
        fields = before.split()

        def unescape(value):
            return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)

        mount_root, mount = Path(unescape(fields[3])), Path(unescape(fields[4]))
        for member in members[version]:
            member = Path(member)
            if not member.is_absolute() or ".." in member.parts:
                observations.append({"version": version, "error": "invalid memory membership"})
                continue
            candidates = [mount / str(member).lstrip("/")]
            if member.is_relative_to(mount_root):
                candidates.insert(0, mount / member.relative_to(mount_root))
            directory = next((path for path in candidates if path.is_dir()), None)
            if directory is None:
                observations.append({"version": version, "error": "unresolved memory membership"})
                continue
            while directory.is_relative_to(mount):
                if (version, directory) not in seen:
                    seen.add((version, directory))
                    # Only the real v2 hierarchy root lacks memory.max; a
                    # v1 or subtree/namespace root is still controlling.
                    global_root = directory == mount and mount_root == Path("/")
                    if (
                        version == 1
                        or not global_root
                        or any((directory / name).exists() for name in files.values())
                    ):
                        row = {
                            "path": str(directory),
                            "version": version,
                            "paths": {key: str(directory / name) for key, name in files.items()},
                            "raw": {},
                            "memory": {},
                        }
                        try:
                            for key, name in files.items():
                                raw = (directory / name).read_text().strip()
                                row["raw"][name] = raw
                                row["memory"][key] = (
                                    None if version == 2 and raw == "max" else int(raw)
                                )
                        except (OSError, ValueError) as exc:
                            row["error"] = type(exc).__name__
                        observations.append(row)
                if directory == mount:
                    break
                directory = directory.parent
    return observations


def host():
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        if name in {"MemTotal", "MemAvailable"}:
            number, unit = value.split()
            require(unit == "kB", "Memory units")
            memory[name] = int(number) * 1024
    memory["boot_id"] = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    memberships = Path("/proc/self/cgroup").read_text()
    mounts = "\n".join(
        row for row in Path("/proc/self/mountinfo").read_text().splitlines() if " - cgroup" in row
    )
    observations = cgroup_memory(memberships, mounts)
    complete = bool(observations) and all(
        "error" not in row and complete_cgroup(row.get("memory", {})) for row in observations
    )
    effective = {}
    if complete:
        pairs = [row["memory"] for row in observations]
        finite = [pair for pair in pairs if pair["memory.max"] is not None]
        effective = (
            min(finite, key=lambda pair: pair["memory.max"] - pair["memory.current"])
            if finite
            else pairs[0]
        )
    memory.update(
        cgroup=effective,
        cgroup_membership=memberships,
        cgroup_mountinfo=mounts,
        cgroup_ancestors=observations,
    )
    return memory


def admit(path, scope, expected_boot):
    observation = host()
    reserve = max(24 * GIB, (observation["MemTotal"] + 4) // 5)
    required = reserve + (4 * GIB if scope == "cpu" else 2 * BASE["size_bytes"])
    if scope == "train":
        required += 30 * GIB + 256 * 1024**2
    group = observation["cgroup"]
    observed = complete_cgroup(group)
    passed = (
        observed
        and observation["boot_id"] == expected_boot
        and 0 < required <= observation["MemAvailable"] <= observation["MemTotal"]
    )
    if observed and group["memory.max"] is not None:
        passed = passed and group["memory.max"] - group["memory.current"] >= required
    row = {
        "scope": scope,
        "observation": observation,
        "reserve_bytes": reserve,
        "required_bytes": required,
        "admitted": passed,
        "cgroup_observation_complete": observed,
        "swap_or_resident_credit_bytes": 0,
        "monotonic": time.monotonic(),
    }
    with ordinary(path).open("ab") as stream:
        stream.write(encode(row) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    require(passed, "Fresh remote/local CPU/LOAD/TRAIN and cgroup headroom admission refused")
    return row


def authority(path, sha):
    blob = read(path)
    require(
        hashlib.sha256(blob).hexdigest() == sha, "Fixed metadata authority changed: " + str(path)
    )
    return parse(blob)


def expected_upload_manifest():
    """Read sealed metadata only: no parent tensors or base weight reads."""
    handoff = authority(HANDOFF, HANDOFF_SHA)
    stage = authority(*META["preflight-receipt.json"])
    parent_path = PREFLIGHT / "stage-01/parent-capture.json"
    parent = authority(
        parent_path,
        stage["staged_evidence_files_excluding_receipt"]["parent-capture.json"]["sha256"],
    )
    compiler = authority(REFERENCE / "compiler-receipt.json", COMPILER_SHA)
    rows = {}

    def add(path, expected, classification, destinations=None):
        path = ordinary(path)
        require(
            path.is_relative_to(ROOT) and set(expected) == {"sha256", "size_bytes"},
            "Named ROOT upload required",
        )
        require(
            type(expected["size_bytes"]) is int and 0 <= expected["size_bytes"] <= 64 * 1024**2,
            "Upload individual cap; base weights excluded",
        )
        label = path.relative_to(ROOT).as_posix()
        relative(label)
        target = [str(path)] if destinations is None else destinations
        if label in rows:
            require(
                rows[label]["pin"] == expected and rows[label]["destinations"] == target,
                "Duplicate upload path differs",
            )
        else:
            rows[label] = {
                "path": str(path),
                "pin": expected,
                "classification": classification,
                "destinations": target,
            }

    for label, expected in handoff["files"].items():
        add(ROOT / relative(label), expected, "public_github_source_exact90")
    require(
        len(handoff["files"]) == 90
        and handoff["files"]["scripts/jev_public_action_artifact.py"]["sha256"] == ART_SHA
        and handoff["files"]["scripts/jev_browser_fit_public_actions.py"]["sha256"] == FIT_SHA,
        "Exact reviewed90",
    )
    add(HANDOFF, pin(HANDOFF), "local_derived_source_handoff_metadata")
    frozen = ROOT / "artifacts/jev-runtime/frozen-public-observation-36/freeze-manifest.json"
    authority(frozen, FROZEN_SHA)
    add(frozen, pin(frozen), "local_derived_frozen_source_inventory")
    for name, (path, sha) in META.items():
        authority(path, sha)
        add(path, pin(path), "local_derived_training_evidence_metadata:" + name)
    add(
        parent_path,
        stage["staged_evidence_files_excluding_receipt"]["parent-capture.json"],
        "local_derived_parent_descriptor",
    )
    public_files = stage["public_original_inventory"]["files"]
    require(
        len(public_files) == 701
        and sum(p["size_bytes"] for p in public_files.values()) == 105415940,
        "Whole701 scope",
    )
    for label, expected in public_files.items():
        add(
            PUBLIC / relative(label),
            expected,
            "local_derived_public_page_training_collection_not_claimed_public_artifact",
            [str(PUBLIC / label), str(PREFLIGHT / "stage-01/inputs/public_collection" / label)],
        )
    require(
        len(parent["files"]) == 496
        and sum(p["size_bytes"] for p in parent["files"].values()) == 80805458,
        "Whole496 parent scope",
    )
    for label, expected in parent["files"].items():
        add(
            PARENT / relative(label),
            expected,
            "local_derived_trained_adapter_and_full521_lineage_not_public_hf_base",
        )
    references = {
        **compiler["output_inventory_excluding_receipt"],
        "compiler-receipt.json": pin(REFERENCE / "compiler-receipt.json"),
    }
    require(len(references) == 154, "Whole154 CPU reference scope")
    for label, expected in references.items():
        add(
            REFERENCE / relative(label),
            expected,
            "local_derived_CPU_reference_with_public_wheel_and_HF_metadata_copies",
        )
    ordered = [rows[name] for name in sorted(rows)]
    require(
        len(ordered) <= MAX_FILES and sum(row["pin"]["size_bytes"] for row in ordered) <= GIB,
        "Whole upload cap",
    )
    return {
        "schema_id": "public07-modal-fit-named-upload-v1",
        "source_handoff_sha256": HANDOFF_SHA,
        "base_weights_uploaded": False,
        "credentials_uploaded": False,
        "rows": ordered,
        "files": len(ordered),
        "bytes": sum(row["pin"]["size_bytes"] for row in ordered),
        "scope": "Exact named task inputs. Derived collection, trained adapters and evidence require upload authorization; their public source pages do not make these local artifacts public.",
    }


def reference_environment():
    closure = authority(REFERENCE / "processor-source-closure.json", CLOSURE_SHA)
    sys.path.insert(0, str(ROOT / "scripts"))
    import jev_public_action_artifact as artifact

    require(
        pin(ROOT / "scripts/jev_public_action_artifact.py")["sha256"] == ART_SHA, "Reviewed ART"
    )
    items = [("processor-source-closure.json", read(REFERENCE / "processor-source-closure.json"))]
    items += [
        ("processor-runtime-sources/" + name, read(REFERENCE / "processor-runtime-sources" / name))
        for name in closure["files"]
    ]
    for name, expected in BASE_METADATA.items():
        if name not in {"artifact-manifest.json", "s1_config.json"}:
            require(pin(CACHE / name) == expected, "Fixed public processor asset authority")
            blob = read(CACHE / name)
            require(
                {"sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)} == expected,
                "Fixed processor asset readback",
            )
            items.append(("processor-metadata/" + name, blob))
    return artifact.verify_reference_environment(tuple(items), model=None)


def clear_baked_processor_metadata():
    """Empty only the six verified image assets before mounting the HF volume."""
    expected = {
        str((CACHE / name).relative_to(HUB)): value
        for name, value in BASE_METADATA.items()
        if name not in {"artifact-manifest.json", "s1_config.json"}
    }
    allowed_directories = {
        HUB / parent for name in expected for parent in Path(name).parents if parent != Path(".")
    }
    found, directories, pending, entries = {}, [], [ordinary(HUB)], 0
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as children:
            for entry in children:
                entries += 1
                require(entries <= 32 and not entry.is_symlink(), "Bounded ordinary image metadata")
                path = ordinary(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    require(
                        path in allowed_directories,
                        "Unexpected image cache directory; keep all bytes",
                    )
                    directories.append(path)
                    pending.append(path)
                else:
                    name = str(path.relative_to(HUB))
                    require(
                        name in expected, "Image cache contains an unexpected file; keep all bytes"
                    )
                    found[name] = pin(path)
    require(found == expected, "Exact baked processor metadata required before removing any file")
    # Validate the entire set before changing it; a real base checkpoint or any
    # extra cache entry causes refusal. This runs only in the image build step.
    for name in found:
        (HUB / name).unlink()
    for directory in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        directory.rmdir()
    require(not any(HUB.iterdir()), "Empty image volume mount point required")


def install_dependencies():
    closure = authority(REFERENCE / "processor-source-closure.json", CLOSURE_SHA)
    versions = {**closure["versions"], **EXTRA_PACKAGES}
    environment = {**os.environ, "PYTHONPATH": str(OVERLAY), "PYTHONNOUSERSITE": "1"}
    with tempfile.TemporaryDirectory() as temporary:
        constraints = Path(temporary) / "constraints.txt"
        constraints.write_text(
            "".join(name + "==" + version + "\n" for name, version in versions.items())
        )
        command = [
            INTERPRETER,
            "-m",
            "pip",
            "--isolated",
            "install",
            "--disable-pip-version-check",
            "--no-cache-dir",
            "--only-binary=:all:",
            "--timeout",
            "60",
            "--retries",
            "1",
            "--index-url",
            "https://pypi.org/simple",
            "--extra-index-url",
            "https://download.pytorch.org/whl/cu130",
            "--constraint",
            str(constraints),
        ]
        # Keep the four new pure-Python package roots in the same overlay as
        # the local13 profile. Resolve their actual dependencies into the venv
        # afterward, with all thirteen direct versions constrained.
        subprocess.run(
            [
                *command,
                "--no-deps",
                "--target",
                str(OVERLAY),
                *[name + "==" + version for name, version in EXTRA_PACKAGES.items()],
            ],
            env=environment,
            check=True,
            timeout=1800,
        )
        subprocess.run(
            [*command, *[name + "==" + version for name, version in versions.items()]],
            env=environment,
            check=True,
            timeout=1800,
        )
        subprocess.run(
            [INTERPRETER, "-m", "pip", "check"], env=environment, check=True, timeout=120
        )
    reference_environment()
    for name, version in EXTRA_PACKAGES.items():
        distribution = importlib.metadata.distribution(name)
        require(
            distribution.version == version
            and Path(distribution._path).absolute()
            == OVERLAY / (name + "-" + version + ".dist-info"),
            "Pinned added package version and overlay origin",
        )
    base_site = Path(INTERPRETER).parent.parent / "lib/python3.13/site-packages"
    for name in ("transformers", "tokenizers", "huggingface_hub", "safetensors", *EXTRA_PACKAGES):
        require(
            not list(base_site.glob(name + "-*.dist-info")),
            "Do not duplicate fixed overlay distribution into base: " + name,
        )
    clear_baked_processor_metadata()
    return 0


def download_child(config):
    output = ordinary(config["output"])
    admit(output / "preflight-gates.jsonl", "cpu", config["boot_id"])
    hub, cache = remote_cache(config)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MODEL,
        revision=REVISION,
        cache_dir=str(hub),
        token=False,
        allow_patterns=sorted(BASE_METADATA),
        max_workers=1,
    )
    for name, expected in BASE_METADATA.items():
        path = ordinary((cache / name).resolve(strict=True))
        require(
            path.is_relative_to(hub) and pin(path) == expected,
            "Exact public HF metadata bytes",
        )
        target = cache / name
        if target.is_symlink():
            temporary = cache / (name + ".public07-regular.pending")
            create(temporary, read(path))
            os.replace(temporary, target)
        require(
            not target.is_symlink() and pin(target) == expected,
            "Ordinary per-model metadata materialization",
        )
    admit(output / "preflight-gates.jsonl", "load", config["boot_id"])
    snapshot_download(
        repo_id=MODEL,
        revision=REVISION,
        cache_dir=str(hub),
        token=False,
        allow_patterns=["model.safetensors"],
        max_workers=1,
    )
    admit(output / "preflight-gates.jsonl", "load", config["boot_id"])
    sys.path.insert(0, str(ROOT / "scripts"))
    import jev_public_action_artifact as artifact

    require(
        pin(artifact.resolve_base_checkpoint(cache), BASE["size_bytes"]) == BASE,
        "Full fixed public BF16 checkpoint hash/size",
    )
    require(mount_root(config["mounts"]["/weights"]) == hub, "Provider weight mount drift")
    create(
        output / "download-result.json",
        encode(
            {
                "success": True,
                "model": MODEL,
                "revision": REVISION,
                "checkpoint": BASE,
                "metadata": BASE_METADATA,
                "token": False,
                "model_loads": 0,
                "training_updates": 0,
            }
        ),
    )
    return 0


def fit_child(config):
    output = ordinary(config["output"])
    admit(output / "preflight-gates.jsonl", "cpu", config["boot_id"])
    manifest_blob = read(named_input(config))
    require(
        hashlib.sha256(manifest_blob).hexdigest() == config["manifest_sha256"],
        "External named upload manifest",
    )
    manifest = parse(manifest_blob)
    require(
        type(manifest["rows"]) is list
        and len(manifest["rows"]) <= MAX_FILES
        and sum(row["pin"]["size_bytes"] for row in manifest["rows"]) <= GIB,
        "Bounded input declaration",
    )
    # Establish only the fixed metadata anchors first; they define the complete
    # expected file population before any public/parent payload is materialized.
    anchors = {
        str(HANDOFF): HANDOFF_SHA,
        **{str(path): sha for path, sha in META.values()},
        str(
            ROOT / "artifacts/jev-runtime/frozen-public-observation-36/freeze-manifest.json"
        ): FROZEN_SHA,
        str(REFERENCE / "compiler-receipt.json"): COMPILER_SHA,
    }
    for index, row in enumerate(manifest["rows"]):
        if row["path"] in anchors:
            source = named_input(config, index)
            require(
                row["pin"]["sha256"] == anchors[row["path"]] and pin(source) == row["pin"],
                "Fixed uploaded metadata anchor",
            )
            target = ordinary(row["path"])
            if target.exists():
                require(pin(target) == row["pin"], "Existing metadata authority differs")
            else:
                create(target, read(source))
    stage = authority(*META["preflight-receipt.json"])
    parent_anchor = str(PREFLIGHT / "stage-01/parent-capture.json")
    parent_row = next(
        (i, row) for i, row in enumerate(manifest["rows"]) if row["path"] == parent_anchor
    )
    source = named_input(config, parent_row[0])
    require(
        parent_row[1]["pin"]
        == stage["staged_evidence_files_excluding_receipt"]["parent-capture.json"]
        and pin(source) == parent_row[1]["pin"],
        "Fixed parent descriptor anchor",
    )
    if Path(parent_anchor).exists():
        require(pin(parent_anchor) == parent_row[1]["pin"], "Existing parent descriptor differs")
    else:
        create(Path(parent_anchor), read(source))
    require(
        manifest == expected_upload_manifest(),
        "Remote named input declaration differs from fixed authorities",
    )
    for index, row in enumerate(manifest["rows"]):
        source = named_input(config, index)
        require(pin(source) == row["pin"], "Uploaded named file differs")
        for destination in row["destinations"]:
            target = ordinary(destination)
            require(
                target.is_relative_to(ROOT) and relative(target.relative_to(ROOT).as_posix()),
                "Fixed ROOT destination",
            )
            if target.exists():
                require(
                    pin(target) == row["pin"],
                    "Existing image input differs; never overwrite installed packages",
                )
            else:
                create(target, read(source))
            require(pin(target) == row["pin"], "Materialized input differs")
    stage_weights(config, output)
    versions = reference_environment()
    admit(output / "preflight-gates.jsonl", "train", config["boot_id"])
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,memory.total,memory.free,memory.used,driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    require(
        len(gpu.stdout) <= 65536
        and b"H200" in gpu.stdout
        and len(gpu.stdout.decode().strip().splitlines()) == 1,
        "Actual single H200 required",
    )
    create(
        output / "remote-preflight.json",
        encode(
            {
                "versions": versions,
                "host": host(),
                "gpu": gpu.stdout.decode(),
                "source_handoff_sha256": HANDOFF_SHA,
                "manifest_sha256": config["manifest_sha256"],
                "steps": 120,
            }
        ),
    )
    sys.path.insert(0, str(ROOT / "scripts"))
    require(
        pin(ROOT / "scripts/jev_browser_fit_public_actions.py")["sha256"] == FIT_SHA,
        "Reviewed unchanged FIT",
    )
    import jev_browser_fit_public_actions as fitter

    result = fitter.fit_public_actions(
        output=output / "fit",
        source_handoff=HANDOFF,
        expected_source_handoff=HANDOFF_SHA,
        expected_boot_id=config["boot_id"],
        steps=120,
        learning_rate=5e-5,
        seed=42,
    )
    require(
        result["stage"] == "complete"
        and result["training_updates"] == 120
        and result["qualified"] is False,
        "Complete120 unqualified public fit",
    )
    create(
        output / "fit-child-result.json",
        encode(
            {
                "success": True,
                "receipt": pin(output / "fit/receipt.json"),
                "manifest": pin(output / "fit/adapter/manifest.json"),
                "training_updates": 120,
                "qualified": False,
            }
        ),
    )
    return 0


def child(mode, config_path):
    config = parse(read(config_path))
    try:
        return download_child(config) if mode == "download" else fit_child(config)
    except BaseException as exc:
        try:
            create(
                Path(config["output"]) / "child-failure.json",
                encode(
                    {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "notes": list(getattr(exc, "__notes__", [])),
                        "host": host(),
                    }
                ),
            )
        except BaseException as secondary:
            exc.add_note("Child failure persistence: " + repr(secondary))
        raise


if __name__ == "__main__":
    if sys.argv[1:] == ["--install-dependencies"]:
        raise SystemExit(install_dependencies())
    if len(sys.argv) == 3 and sys.argv[1] in {"--download-child", "--fit-child"}:
        raise SystemExit(child(sys.argv[1][2:-6], sys.argv[2]))
    if len(sys.argv) == 3 and sys.argv[1] == "--inventory-only":
        create(Path(sys.argv[2]), encode(expected_upload_manifest()))
        raise SystemExit(0)

import modal  # noqa: E402 -- every stdlib/install/fit child exits before SDK import.

from apps.modal import public_action_env_probe as probe  # noqa: E402

app = modal.App("gemma-public07-fit", include_source=False)
image = (
    probe.image.add_local_file(str(SCRIPT), str(SCRIPT), copy=True)
    .env(
        {
            "PYTHONPATH": ":".join((str(SCRIPT.parent), str(ROOT), str(OVERLAY))),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    .run_commands(shlex.join([INTERPRETER, "-B", "-s", str(SCRIPT), "--install-dependencies"]))
)
inputs = modal.Volume.from_name("jev-public07-fit-named-inputs-01", create_if_missing=True)
weights = modal.Volume.from_name("jev-public07-fit-public-hf-a66-01", create_if_missing=True)
results = modal.Volume.from_name("jev-public07-fit-results-01", create_if_missing=True)


def group_alive(process):
    process.poll()
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    return True


def stop_owned(process):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not group_alive(process):
            break
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        until = time.monotonic() + 10
        while group_alive(process) and time.monotonic() < until:
            time.sleep(0.05)
    require(
        process.poll() is not None and not group_alive(process),
        "Owned child/group exit unknown; no next worker",
    )


def tree_inventory(root):
    found, pending, total, entries = {}, [ordinary(root)], 0, 0
    while pending:
        with os.scandir(pending.pop()) as children:
            for entry in children:
                entries += 1
                require(entries <= MAX_FILES, "Output traversal cap")
                mode = entry.stat(follow_symlinks=False).st_mode
                require(not stat.S_ISLNK(mode), "Output links rejected")
                if stat.S_ISDIR(mode):
                    pending.append(Path(entry.path))
                else:
                    require(stat.S_ISREG(mode), "Output ordinary files only")
                    size = entry.stat(follow_symlinks=False).st_size
                    require(
                        total + size <= GIB and len(found) + 1 <= MAX_FILES,
                        "Whole closed output cap before hashing",
                    )
                    name = Path(entry.path).relative_to(root).as_posix()
                    found[name] = pin(entry.path, GIB)
                    total += size
    return dict(sorted(found.items()))


def copy_output(output, binding):
    """A closed ephemeral tree is copied and rehashed before its export marker."""
    expected = tree_inventory(output)
    root = mount_root(binding)
    destination = ordinary(root / output.name)
    require(not destination.exists(), "Fresh provider export destination")
    destination.mkdir()
    for label, entry in expected.items():
        stream_copy(output / relative(label), destination / relative(label), entry)
    require(
        tree_inventory(output) == expected == tree_inventory(destination)
        and mount_root(binding) == root,
        "Closed full export copy/anchor differs",
    )
    return destination, expected


def owned(mode, run_id, manifest_sha256=None):
    require(re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id) is not None, "Bounded fresh run ID")
    require(mode in {"download", "fit"}, "Declared owned mode")
    output = ordinary(RUNTIME / "owned-public07" / (run_id + "-" + mode))
    require(not output.exists(), "Fresh owned remote output")
    output.mkdir(parents=True)
    config = {
        "run_id": run_id,
        "output": str(output),
        "boot_id": host()["boot_id"],
        "manifest_sha256": manifest_sha256,
        "mounts": {
            alias: capture_mount(alias)
            for alias in (
                ("/weights", "/export", "/input") if mode == "fit" else ("/weights", "/export")
            )
        },
    }
    create(output / "child-config.json", encode(config))
    environment = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "PYTHONPATH": ":".join((str(OVERLAY), str(ROOT / "src"))),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "HF_HUB_CACHE": str(HUB),
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    }
    for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "LD_LIBRARY_PATH"):
        if name in os.environ:
            environment[name] = os.environ[name]
    if mode == "fit":
        environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    else:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    primary, process, confirmed, log_sizes, export_pin = None, None, False, {}, None
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            process = subprocess.Popen(
                [
                    INTERPRETER,
                    "-B",
                    "-s",
                    str(SCRIPT),
                    "--" + mode + "-child",
                    str(output / "child-config.json"),
                ],
                env=environment,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            started = time.monotonic()
            while process.poll() is None:
                require(
                    max(os.fstat(stdout.fileno()).st_size, os.fstat(stderr.fileno()).st_size)
                    <= MAX_LOG,
                    "Owned logs cap exceeded",
                )
                require(time.monotonic() - started < 7200, "Owned7200s watchdog exceeded")
                time.sleep(0.1)
            require(process.returncode == 0, "Owned child failed: " + str(process.returncode))
        except BaseException as exc:
            primary = exc
        finally:
            if process is not None:
                try:
                    stop_owned(process)
                    confirmed = True
                except BaseException as secondary:
                    if primary is None:
                        primary = secondary
                    else:
                        primary.add_note("Owned process cleanup: " + repr(secondary))
            for name, stream in (("stdout.log", stdout), ("stderr.log", stderr)):
                try:
                    actual_size = os.fstat(stream.fileno()).st_size
                    stream.seek(0)
                    create(output / name, stream.read(MAX_LOG))
                    log_sizes[name] = {
                        "actual_bytes": actual_size,
                        "retained_bytes": min(actual_size, MAX_LOG),
                        "truncated": actual_size > MAX_LOG,
                    }
                    require(actual_size <= MAX_LOG, "Owned final log cap exceeded")
                except BaseException as secondary:
                    if primary is None:
                        primary = secondary
                    else:
                        primary.add_note("Bounded log preservation: " + repr(secondary))
    report, destination = None, None
    try:
        report = {
            "child_success": primary is None and confirmed,
            "owned_exit_confirmed": confirmed,
            "scope": "child exit only; export/volume publication pending",
            "returncode": process.returncode if process is not None else None,
            "error": None
            if primary is None
            else {
                "type": type(primary).__name__,
                "message": str(primary),
                "notes": list(getattr(primary, "__notes__", [])),
            },
            "child_failure": parse(read(output / "child-failure.json"))
            if (output / "child-failure.json").exists()
            else None,
            "logs": log_sizes,
            "modal_app_id": app.app_id,
            "modal_call_id": modal.current_function_call_id(),
        }
        create(output / "controller-receipt.json", encode(report))
        if confirmed:
            destination, index = copy_output(output, config["mounts"]["/export"])
            for binding in config["mounts"].values():
                mount_root(binding)
            if mode == "download":
                weights.commit()
            create(
                destination / "export-index.json",
                encode({"root": output.name, "files": index, "complete_owned_exit": True}),
            )
            export_pin = pin(destination / "export-index.json")
        results.commit()
    except BaseException as secondary:
        export_pin = None
        if primary is None:
            primary = secondary
        else:
            primary.add_note("Closed output publication: " + repr(secondary))
        if destination is not None and (destination / "export-index.json").exists():
            try:
                os.replace(
                    destination / "export-index.json", destination / "export-index.invalid.json"
                )
                results.commit()
            except BaseException as cleanup:
                primary.add_note("Failed export marker invalidation: " + repr(cleanup))
    if primary is not None and not isinstance(primary, Exception):
        raise primary
    return {
        "success": primary is None and report is not None and report["child_success"],
        "root": output.name,
        "export_index": export_pin,
        "error": None
        if primary is None
        else {
            "type": type(primary).__name__,
            "message": str(primary),
            "notes": list(getattr(primary, "__notes__", [])),
        },
    }


@app.function(
    image=image,
    cpu=2,
    memory=344064,
    ephemeral_disk=524288,
    timeout=7800,
    max_containers=1,
    retries=0,
    single_use_containers=True,
    include_source=False,
    volumes={"/weights": weights, "/export": results},
)
def download(run_id: str):
    return owned("download", run_id)


@app.function(
    image=image,
    gpu="H200",
    cpu=2,
    memory=344064,
    ephemeral_disk=524288,
    timeout=7800,
    max_containers=1,
    retries=0,
    single_use_containers=True,
    include_source=False,
    block_network=True,
    volumes={
        "/weights": weights.with_mount_options(read_only=True),
        "/input": inputs.with_mount_options(read_only=True),
        "/export": results,
    },
)
def fit(run_id: str, manifest_sha256: str):
    return owned("fit", run_id, manifest_sha256)


def receive(remote, destination, expected, limit=GIB):
    require(
        expected["size_bytes"] <= limit and not destination.exists(),
        "Fresh bounded streamed destination",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    count, digest = 0, hashlib.sha256()
    with destination.open("xb") as stream:
        for chunk in results.read_file(remote):
            count += len(chunk)
            require(count <= expected["size_bytes"], "Stream exceeded declared bytes")
            stream.write(chunk)
            digest.update(chunk)
        stream.flush()
        os.fsync(stream.fileno())
    require(
        {"sha256": digest.hexdigest(), "size_bytes": count} == expected,
        "Streamed output SHA/size differs",
    )


def export(result, destination):
    require(
        result["export_index"] is not None,
        "Remote output not closed; partials remain in results volume, no next worker permitted",
    )
    index_path = destination / "export-index.json"
    receive(result["root"] + "/export-index.json", index_path, result["export_index"], 8 * 1024**2)
    index = parse(read(index_path, 8 * 1024**2))
    require(
        index["root"] == result["root"]
        and len(index["files"]) <= MAX_FILES
        and sum(p["size_bytes"] for p in index["files"].values()) <= GIB,
        "Bounded named export index",
    )
    for name, expected in index["files"].items():
        receive(result["root"] + "/" + str(relative(name)), destination / relative(name), expected)


@app.local_entrypoint()
def main(output: str, manifest: str, run_id: str, run: bool = False):
    require(run, "Explicit --run after source and exact named upload manifest review required")
    require(
        re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", run_id) is not None,
        "Bounded fresh run ID before uploading",
    )
    local = ordinary(output)
    require(not local.exists(), "Fresh local export directory")
    local.mkdir(parents=True)
    supplied = read(manifest, 8 * 1024**2)
    expected = expected_upload_manifest()
    require(
        parse(supplied) == expected and supplied == encode(expected),
        "Reviewed exact input manifest required",
    )
    admit(local / "upload-gates.jsonl", "cpu", host()["boot_id"])
    before = {}
    for index, row in enumerate(expected["rows"]):
        require(pin(row["path"]) == row["pin"], "Named upload bytes differ")
        before[row["path"]] = row["pin"]
    with inputs.batch_upload(force=False) as batch:
        for index, row in enumerate(expected["rows"]):
            batch.put_file(row["path"], f"/{run_id}/files/{index:05d}")
        batch.put_file(manifest, f"/{run_id}/manifest.json")
    require(
        all(pin(path) == value for path, value in before.items()),
        "Local named inputs changed during upload",
    )
    downloaded = download.remote(run_id)
    export(downloaded, local / "download")
    require(downloaded["success"], "Public HF download stage failed; no GPU worker started")
    fitted = fit.remote(run_id, hashlib.sha256(supplied).hexdigest())
    export(fitted, local / "fit-job")
    require(fitted["success"], "Remote fit failed; retained named partial evidence")
