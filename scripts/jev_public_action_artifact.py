"""Separate actual07 public action artifact contract; no deployment attach API.

The original84 collection sources and legacy NF4 verifier remain unchanged.
Tensor and whole public admission checks are lazy and require a fresh durable
CPU gate supplied by the caller. Returned public artifacts remain unqualified.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3
MAX_FILES = 10_000
PUBLIC_FILE_COUNT, PUBLIC_TOTAL_BYTES = 701, 105415940
PARENT_FILE_COUNT, PARENT_TOTAL_BYTES = 496, 80805458
KIND = "jev_public_native_action_lora_v1"
QUALIFICATION_KIND = "jev_public_action_bf16_qualification_v1"
MODEL_ID = "knowlet/Gemma-4-12B-Unified-System-One"
REVISION = "a66f836b56605039fe040f330180e336d19b3362"
HEAD_SHA = "9619fd670a4074367c72f2cf09d777970ad1eafa511bff605a82c8558dcc24d4"
TEMPERATURE = 2.7830344470383452
BASE_CHECKPOINT = {
    "sha256": "11144621122123135eb79c9cfd0466a2c51f230109124aefe401cdbec8c71a53",
    "size_bytes": 23919549408,
}
IDENTITY = "generated-actions-none"
PROFILE = "composed-visible-text-v3"
PROFILE_HELPER_SHA = "8d199ac1564af05892e9b21e225d1fdf500f1953840855c902160ed8d56df89d"
PROFILE_JS_SHA = "c694f308f29ed5000b7bb60e9374afd9398413d2811e07741cf0f0a6321be552"
FROZEN_MANIFEST_SHA = "a66125c589b7408337685d06daf4d0eb80265b07e7e9a55786442562389b87d3"
FROZEN_COMMIT = "e864afa54692d019e21f8b8e234e8f65401703fc"
PUBLIC_IDENTITY = "d74f729460c648b8c88cf677bc379d1ae34fe7530a9651810e8af2825d449742"
PROVENANCE_SHA = "53d8b577a7d875b5d833a6c5fa71d77f033d668d651a12e94c9febf062b3b480"
PARENT_PINS = {
    "manifest_sha256": "7edcf9cd1af3d3d7cb04c9eca14e2b2c67847bc2caeef25c32eebfce6de8ddc3",
    "receipt_sha256": "6d04b5ba6301f0d9cf4d38bd4f335e1447250d556939779ab6a986d2bd4a504a",
    "weights_sha256": "165fbd1db402692ec06f4264a16026eb76abbed711d9ff56aef37f44a442f178",
}
LORA = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "target_modules": r"model\.language_model\.layers\.[0-9]+\.self_attn\.(q_proj|v_proj)",
    "bias": "none",
    "task_type": "CAUSAL_LM",
}
KERNELS = {
    "scripts/jev_browser_fit_actions.py": "4bf6585008e434a469dafb4a9997dee595b079044e03c9afe00ceffa5e031f82",
    "scripts/jev_browser_fit_readout.py": "228ba48f185fd88f4c420c04c1119893ec9e2a18f3d8c69b9c82ab0ae8d9e02f",
    "scripts/jev_browser_training_data.py": "845f81c9ca414bf4c9ea46869381ddf2cbb09cf7611405f832b6671004af1205",
    "scripts/breakthrough_training.py": "72ce7c094f4dfa47000bd3829f06749253bc6283151fbf5af78884d2777b47a6",
}
NEW_SOURCES = {"scripts/jev_public_action_artifact.py", "scripts/jev_browser_fit_public_actions.py"}
COUNTS = {"train": 120, "development": 22}
DISPOSITIONS = {
    "accepted-fit-row": 142,
    "bootstrap-non-fit": 49,
    "excluded-no-input-DONE": 2,
    "excluded-no-input-SCROLL": 1,
}
FIXED_METADATA = {
    "admission-receipt.json": "bb799a48e8bbf1fb0af5afaf23cb38fb09bb67c3d26c833e96ba33bc34b0fcdb",
    "admission-provenance.json": "d80e5106ab67375403f1b3b5ab9132ae7f212a134532d0d0d422ac044a648c80",
    "admission-plan.json": "47c9ab6d2607e41598abad7ef7165fcb6577ab5662cfde7c99ba04e1f0f963ca",
    "preflight-receipt.json": "8e017a703662f64530599657426fe3a5ef7134af3b8df0daca7aeb613057dd7c",
    "preflight-root-receipt.json": "2349f04ae8bd8825270603a84e5106b3dccc80914baa804718cae9f0034c3d79",
    "syntax-audit.json": "e6216d7e4d335c2b76da6942b1c2226fb56eb7d2ba8fa5128bc76f3fb8f0c0bc",
    "compiler-root-receipt.json": "95b061b62c24d2711ab700562603328a95ed656ab5aac0887f2a6d2e8371716a",
    "compiler-audit.json": "099dc2561e7595813c29d1dcb2ed40e40e08958aa18146a9d4dc80dc2476c326",
}
COMPILER_RECEIPT_SHA = "a8a5ebdefccc4ef1be01e673c36d36452efd30368b02921b81aae9cabd096b82"
PROCESSOR_CLOSURE_SHA = "ff87ceb2563bdee706c4b8cc1064a4682534be2925372c0c02c7ccd2d04e7bbc"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact(value, expected):
    """JSON equality with distinct bool/int and scalar types at every depth."""
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return set(value) == set(expected) and all(exact(value[k], v) for k, v in expected.items())
    if type(expected) is list:
        return len(value) == len(expected) and all(
            exact(a, b) for a, b in zip(value, expected, strict=True)
        )
    return value == expected


def encode(value, *, canonical=False):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=canonical, separators=(",", ":"), allow_nan=False
    ).encode()


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def parse(blob):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate public artifact JSON key")
            result[key] = value
        return result

    def constant(_):
        raise ValueError("Nonfinite public artifact JSON")

    def finite(value):
        number = float(value)
        require(math.isfinite(number), "Nonfinite exponent overflow")
        return number

    return json.loads(blob, object_pairs_hook=pairs, parse_constant=constant, parse_float=finite)


def ordinary(path):
    path = Path(path).absolute()
    require(
        ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "Ordinary confined local path required",
    )
    return path


def resolve_base_checkpoint(cache):
    """Resolve only the pinned model's checkpoint within its repo or hub blobs."""
    cache = ordinary(cache)
    repository = cache.parent.parent
    require(
        cache.name == REVISION
        and cache.parent.name == "snapshots"
        and repository.name == "models--" + MODEL_ID.replace("/", "--"),
        "Pinned model snapshot cache required",
    )
    central_blobs = repository.parent / "blobs"
    checkpoint = ordinary((cache / "model.safetensors").resolve(strict=True))
    require(
        checkpoint.is_file()
        and (
            checkpoint.is_relative_to(repository)
            or checkpoint != central_blobs
            and checkpoint.is_relative_to(central_blobs)
        ),
        "Base checkpoint escapes its model repository or fixed hub blob cache",
    )
    return checkpoint


def relative(label):
    require(type(label) is str and bool(label), "Nonempty evidence label required")
    path = PurePosixPath(label)
    require(
        str(path) == label
        and not path.is_absolute()
        and ".." not in path.parts
        and "\\" not in label
        and not any(c in label for c in "\x00\r\n"),
        "Unconfined evidence label",
    )
    return label


def _stat(value):
    return tuple(
        getattr(value, name)
        for name in ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    )


def read(path, limit=16 * 1024**2):
    path = ordinary(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= limit,
            "Evidence byte/type bound",
        )
        blob = stream.read(before.st_size + 1)
        require(
            len(blob) == before.st_size
            and _stat(before) == _stat(os.fstat(stream.fileno())) == _stat(path.stat()),
            "Evidence changed during read",
        )
    return blob


def pin(path, *, limit=GIB):
    path = ordinary(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= limit, "Hash byte/type bound"
        )
        result, size = hashlib.sha256(), 0
        while chunk := stream.read(1024**2):
            result.update(chunk)
            size += len(chunk)
            require(size <= before.st_size, "Evidence grew during hash")
        require(
            size == before.st_size
            and _stat(before) == _stat(os.fstat(stream.fileno())) == _stat(path.stat()),
            "Evidence changed during hash",
        )
    return {"sha256": result.hexdigest(), "size_bytes": size}


def tree_sizes(root):
    root = ordinary(root)
    require(root.is_dir(), "Ordinary evidence tree required")
    result, total, directories = {}, 0, 0

    def walk_error(error):
        raise error

    for directory, names, files in os.walk(
        root, topdown=True, followlinks=False, onerror=walk_error
    ):
        directories += 1
        require(
            directories <= MAX_FILES
            and stat.S_ISDIR(Path(directory).lstat().st_mode)
            and not Path(directory).is_symlink(),
            "Bounded ordinary directory population",
        )
        for name in names:
            require(
                stat.S_ISDIR((Path(directory) / name).lstat().st_mode),
                "Directory symlink or special entry",
            )
        for name in files:
            path = Path(directory) / name
            value = path.lstat()
            require(stat.S_ISREG(value.st_mode), "Evidence symlink or special file")
            result[path.relative_to(root).as_posix()] = value.st_size
            total += value.st_size
            require(len(result) <= MAX_FILES and total <= GIB, "Closed evidence tree cap")
    return result


def inventory(root):
    return {label: pin(Path(root) / label) for label in tree_sizes(root)}


def check_inventory(root, expected):
    require(
        tree_sizes(root) == {relative(k): v["size_bytes"] for k, v in expected.items()},
        "Closed evidence population/size differs",
    )
    for label, expected_pin in expected.items():
        require(pin(Path(root) / label) == expected_pin, "Evidence content differs: " + label)


def identity(files):
    return sha(encode(files, canonical=True))


def trusted_module(name, path):
    expected = ROOT / path
    existing = sys.modules.get(name)
    spec = existing.__spec__ if existing is not None else importlib.util.find_spec(name)
    require(
        spec is not None and spec.origin is not None and Path(spec.origin).resolve() == expected,
        "Foreign trusted module origin: " + name,
    )
    module = importlib.import_module(name)
    require(Path(module.__file__).resolve() == expected, "Changed trusted module origin")
    return module


def verify_source_handoff(blob, *, expected_sha256):
    require(sha(blob) == expected_sha256, "Externally reviewed source handoff SHA differs")
    handoff = parse(blob)
    require(
        set(handoff) == {"schema_id", "collection", "files", "identity_sha256"}
        and handoff["schema_id"] == "jev_public_action_source_handoff_v1",
        "Source handoff schema",
    )
    collection = handoff["collection"]
    require(
        collection
        == {
            "frozen_manifest_sha256": FROZEN_MANIFEST_SHA,
            "git_commit": FROZEN_COMMIT,
            "source_count": 84,
        },
        "Original collection identity",
    )
    manifest = parse(
        read(ROOT / "artifacts/jev-runtime/frozen-public-observation-36/freeze-manifest.json")
    )
    require(
        sha(read(ROOT / "artifacts/jev-runtime/frozen-public-observation-36/freeze-manifest.json"))
        == FROZEN_MANIFEST_SHA
        and manifest["git_commit"] == FROZEN_COMMIT
        and len(manifest["files"]) == manifest["file_count"] == 84,
        "Original frozen84 identity",
    )
    expected = set(manifest["files"]) | set(KERNELS) | NEW_SOURCES
    files = handoff["files"]
    require(
        len(files) == 90
        and set(files) == expected
        and list(files) == sorted(files)
        and handoff["identity_sha256"] == identity(files),
        "Exact sorted84+4+2 source population",
    )
    live = {
        "src/s1/" + p.relative_to(ROOT / "src/s1").as_posix()
        for p in (ROOT / "src/s1").rglob("*.py")
    }
    live.update(
        "scripts/" + n + ".py"
        for n in (
            "jev_browser_benchmark",
            "jev_local_model",
            "jev_public_web_cases",
            "jev_browser_collect_public_training",
        )
    )
    require(live == set(manifest["files"]), "Installed complete84 population changed")
    for label, source_pin in files.items():
        require(
            set(source_pin) == {"sha256", "size_bytes"}
            and pin(ROOT / relative(label)) == source_pin,
            "Installed handoff source differs: " + label,
        )
        if label in manifest["files"]:
            require(source_pin == manifest["files"][label], "Collection source changed")
        if label in KERNELS:
            require(source_pin["sha256"] == KERNELS[label], "Unchanged trusted kernel required")
    return handoff


@dataclass(frozen=True)
class VerifiedPublicFitInputs:
    """Immutable captured bytes; every public getter returns detached values."""

    public_root: Path
    metadata_bytes: bytes
    public_items: tuple[tuple[str, bytes], ...]
    reference_items: tuple[tuple[str, bytes], ...]
    source_handoff_bytes: bytes
    source_handoff_sha256: str

    def metadata(self):
        return parse(self.metadata_bytes)

    def admitted(self):
        public = trusted_module(
            "s1.browser_public_training_validation", "src/s1/browser_public_training_validation.py"
        )
        case_type = trusted_module(
            "s1.browser_public_training", "src/s1/browser_public_training.py"
        ).PublicNativeActionCase
        blobs = dict(self.public_items)
        snapshot = public.ClosedPublicSnapshot(self.public_root, MappingProxyType(blobs))
        cases = {
            s: tuple(
                case_type.model_validate_json(line)
                for line in blobs[s + "/cases.jsonl"].splitlines()
            )
            for s in COUNTS
        }
        wires = {s: tuple(blobs[s + "/wire_requests.jsonl"].splitlines()) for s in COUNTS}
        return public.AdmittedPublicCollection(
            snapshot,
            cases["train"],
            cases["development"],
            MappingProxyType(wires),
            self.metadata()["provenance_bytes"].encode(),
        )

    def check(self):
        meta = self.metadata()
        require(
            {k: {"sha256": sha(b), "size_bytes": len(b)} for k, b in self.public_items}
            == meta["public_inventory"]["files"],
            "Captured public bytes changed",
        )
        check_inventory(self.public_root, meta["public_inventory"]["files"])
        verify_source_handoff(self.source_handoff_bytes, expected_sha256=self.source_handoff_sha256)


def case_provenance(admitted):
    result = {"source_scope": "development_only"}
    for split in COUNTS:
        cases = admitted.train_cases if split == "train" else admitted.development_cases
        wires = admitted.wire_rows(split)
        result[split] = {
            "ids": sorted({c.id for c in cases}),
            "groups": sorted({c.group_id or c.id for c in cases}),
            "request_fingerprints": sorted(
                {
                    sha(
                        json.dumps(
                            w["native_payload"],
                            sort_keys=True,
                            separators=(",", ":"),
                            allow_nan=False,
                        ).encode()
                    )
                    for w in wires
                }
            ),
        }
    for key in ("ids", "groups", "request_fingerprints"):
        require(
            not set(result["train"][key]) & set(result["development"][key]), "Public split leakage"
        )
    return result


def validate_public_scope(admitted, expected_inventory):
    public = trusted_module(
        "s1.browser_public_training_validation", "src/s1/browser_public_training_validation.py"
    )
    require(
        type(admitted) is public.AdmittedPublicCollection,
        "Whole typed AdmittedPublicCollection required",
    )
    provenance = admitted.provenance
    require(
        provenance["collection_schema_id"] == "jev-public-native-training-collection-v5"
        and provenance["provenance_fingerprint"] == PROVENANCE_SHA,
        "Exact actual07 V5 admission",
    )
    profile = provenance["observation_profile"]
    require(
        profile["profile"] == PROFILE
        and profile["helper_source_sha256"] == PROFILE_HELPER_SHA
        and profile["effective_snapshot_js_sha256"] == PROFILE_JS_SHA,
        "Exact V3 observation profile",
    )
    require(
        admitted.snapshot.manifest() == expected_inventory
        and expected_inventory["identity_sha256"] == PUBLIC_IDENTITY
        and len(expected_inventory["files"]) == PUBLIC_FILE_COUNT
        and sum(p["size_bytes"] for p in expected_inventory["files"].values()) == PUBLIC_TOTAL_BYTES
        and len(admitted.train_cases) == 120
        and len(admitted.development_cases) == 22,
        "Exact whole701/142 public scope",
    )
    actual = {}
    for split, total in (("train", 167), ("development", 27)):
        mapping = provenance["all_native_call_mapping"][split]
        require(len(mapping) == total, "Complete194 call mapping")
        for row in mapping:
            disposition = row["disposition"]
            actual[disposition] = actual.get(disposition, 0) + 1
    require(actual == DISPOSITIONS, "Separate accepted/bootstrap/DONE/SCROLL dispositions")
    admitted.snapshot.verify_unchanged()
    return case_provenance(admitted)


def verify_public_fit_inputs(
    public_root,
    preflight_root,
    *,
    admission_root,
    reference_compile,
    metadata_paths,
    source_handoff_bytes,
    expected_source_handoff,
    expected_admission,
    expected_preflight,
    tokenizer_cache,
    gate,
):
    """Fresh gate precedes genuine trusted admission; never accepts fixture DTOs."""
    gate.check("before-public-project-admission", "cpu")
    require(
        expected_admission == FIXED_METADATA["admission-receipt.json"]
        and expected_preflight == FIXED_METADATA["preflight-root-receipt.json"],
        "Externally selected actual07 receipts",
    )
    handoff = verify_source_handoff(source_handoff_bytes, expected_sha256=expected_source_handoff)
    require(set(metadata_paths) == set(FIXED_METADATA), "Exact actual07 metadata population")
    metadata_blobs = {}
    for label, expected in FIXED_METADATA.items():
        blob = read(metadata_paths[label])
        require(sha(blob) == expected, "Fixed actual07 metadata changed: " + label)
        metadata_blobs[label] = blob
    values = {k: parse(v) for k, v in metadata_blobs.items()}
    stage = values["preflight-receipt.json"]
    require(
        stage["stage"] == "complete"
        and stage["test_mode"] is False
        and stage["primary_error"] is None
        and not stage["finalization_errors"]
        and values["syntax-audit.json"]["observation_coverage_passed"] is True
        and values["compiler-audit.json"]["passed"] is True,
        "Complete authentic preflight/syntax/compiler evidence",
    )
    compiler_blob = read(Path(reference_compile) / "compiler-receipt.json")
    require(sha(compiler_blob) == COMPILER_RECEIPT_SHA, "Exact genuine CPU reference compiler")
    compiler = parse(compiler_blob)
    expected_reference = {
        **compiler["output_inventory_excluding_receipt"],
        "compiler-receipt.json": pin(Path(reference_compile) / "compiler-receipt.json"),
    }
    check_inventory(reference_compile, expected_reference)
    expected_public = stage["public_original_inventory"]
    check_inventory(public_root, expected_public["files"])
    check_inventory(
        Path(preflight_root) / "stage-01/inputs/public_collection", expected_public["files"]
    )
    public = trusted_module(
        "s1.browser_public_training_validation", "src/s1/browser_public_training_validation.py"
    )
    reviewed = {
        k.removeprefix("src/"): p
        for k, p in handoff["files"].items()
        if k
        in parse(
            read(ROOT / "artifacts/jev-runtime/frozen-public-observation-36/freeze-manifest.json")
        )["files"]
    }
    admitted = public.validate_public_training_collection(
        public_root, reviewed_source_pins=reviewed, tokenizer_cache=tokenizer_cache
    )
    provenance = validate_public_scope(admitted, expected_public)
    require(
        encode(admitted.provenance) == encode(values["admission-provenance.json"]),
        "Re-admitted provenance differs",
    )
    parent_blob = read(Path(preflight_root) / "stage-01/parent-capture.json")
    expected_parent = stage["staged_evidence_files_excluding_receipt"]["parent-capture.json"]
    require(
        {"sha256": sha(parent_blob), "size_bytes": len(parent_blob)} == expected_parent,
        "Historical parent descriptor metadata differs",
    )
    metadata_blobs["parent-capture.json"] = parent_blob
    meta = {
        "public_inventory": expected_public,
        "provenance": provenance,
        "provenance_bytes": encode(admitted.provenance).decode(),
        "metadata_files": {
            k: {"sha256": sha(v), "size_bytes": len(v)} for k, v in metadata_blobs.items()
        },
        "metadata_blobs": {k: v.decode() for k, v in metadata_blobs.items()},
        "reference_inventory": expected_reference,
        "reference_root": str(ordinary(reference_compile)),
        "parent_descriptor": parse(parent_blob),
        "wire_identities": compiler["compilation"]["wire_identities"],
        "admission_root": str(ordinary(admission_root)),
    }
    result = VerifiedPublicFitInputs(
        ordinary(public_root),
        encode(meta),
        tuple(sorted(admitted.snapshot.blobs.items())),
        tuple((k, read(Path(reference_compile) / k, 64 * 1024**2)) for k in expected_reference),
        bytes(source_handoff_bytes),
        expected_source_handoff,
    )
    result.check()
    return result


def validate_parent_capture(captured, provenance, *, expected_descriptor, gate):
    gate.check("before-parent-recursive-tensor-authentication", "cpu")
    api = trusted_module("s1.browser_action_adapter", "src/s1/browser_action_adapter.py")
    verified, descriptor = captured["verified"], captured["descriptor"]
    require(
        type(verified) is api.VerifiedActionAdapter
        and {
            "manifest_sha256": verified.manifest_sha256,
            "receipt_sha256": verified.receipt_sha256,
            "weights_sha256": verified.manifest["files"]["adapter_model.safetensors"]["sha256"],
        }
        == PARENT_PINS,
        "Exact historical legacy parent required",
    )
    validate_expected_parent_descriptor(descriptor, expected_descriptor)
    require(
        len(descriptor["files"]) == PARENT_FILE_COUNT
        and sum(p["size_bytes"] for p in descriptor["files"].values()) == PARENT_TOTAL_BYTES
        and descriptor["historical_completed_updates"]
        == descriptor["historical_training_cases"]
        == 521
        and verified.tensor_metadata["tensor_count"] == 176
        and verified.tensor_metadata["parameters"] == 5193728
        and verified.tensor_metadata["stored_dtypes"] == ["torch.float32"],
        "Complete496/176FP32/521 parent lineage",
    )
    api.verify_initial_action_adapter_splits(
        captured["frozen_root"], provenance, head_sha256=HEAD_SHA
    )
    for key in ("original_root", "frozen_root"):
        require(
            api.verify_initial_action_adapter_copy(captured[key], descriptor["files"])["success"]
            is True,
            "Historical parent original/copied bytes changed",
        )
    return verified


def validate_expected_parent_descriptor(actual, expected):
    require(
        type(actual) is dict
        and type(expected) is dict
        and set(actual) == set(expected)
        and all(
            exact(actual[key], value)
            for key, value in expected.items()
            if key != "initial_tensor_verification"
        ),
        "Exact authenticated historical496 parent inventory/descriptor differs",
    )


def serialized_row(row):
    return {
        "record": row["record"],
        "inputs": {k: v.tolist() for k, v in row["inputs"].items()},
        "positions": row["positions"].tolist(),
        "targets": row["targets"].tolist(),
    }


def verify_reference_environment(reference_items, *, model=None):
    """Reviewed51 source/metadata files, not every framework binary."""
    reference = dict(reference_items)
    closure_blob = reference["processor-source-closure.json"]
    require(
        sha(closure_blob) == PROCESSOR_CLOSURE_SHA,
        "Reviewed processor/framework source closure differs",
    )
    closure = parse(closure_blob)
    require(
        str(Path(sys.executable).absolute()) == closure["interpreter"]
        and list(sys.version_info[:2]) == closure["python_major_minor"]
        and sys.implementation.name == "cpython",
        "Exact reviewed processor interpreter required",
    )
    for label, spec in closure["files"].items():
        expected = {k: spec[k] for k in ("sha256", "size_bytes")}
        require(
            pin(spec["path"]) == expected
            and {
                "sha256": sha(reference["processor-runtime-sources/" + label]),
                "size_bytes": len(reference["processor-runtime-sources/" + label]),
            }
            == expected,
            "Current/preserved processor source differs",
        )
    versions = {}
    for name, expected in closure["packages"].items():
        distribution = importlib.metadata.distribution(name)
        metadata_path = Path(distribution._path) / "METADATA"
        require(
            str(metadata_path.absolute()) == expected["metadata_path"]
            and pin(metadata_path) == {k: expected[k] for k in ("sha256", "size_bytes")}
            and distribution.version == closure["versions"][name],
            "Actual distribution version/metadata differs: " + name,
        )
        versions[name] = distribution.version
    for name, expected in closure["module_origins"].items():
        loaded = sys.modules.get(name)
        require(
            model is None or loaded is not None,
            "Required actual processor module is absent: " + name,
        )
        if loaded is not None:
            require(
                getattr(loaded, "__file__", None) is not None
                and Path(loaded.__file__).resolve() == Path(expected).resolve(),
                "Foreign loaded processor module: " + name,
            )
    metadata = {
        label.removeprefix("processor-metadata/"): blob
        for label, blob in reference.items()
        if label.startswith("processor-metadata/")
    }
    require(
        set(metadata)
        == {
            "processor_config.json",
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
        },
        "Exact cached six processor metadata files required",
    )
    cache = (
        ROOT
        / "artifacts/jev-local-hf/hub"
        / ("models--" + MODEL_ID.replace("/", "--"))
        / "snapshots"
        / REVISION
    )
    confined = cache.parents[1].resolve(strict=True)
    for label, expected_blob in metadata.items():
        actual_path = (cache / label).resolve(strict=True)
        require(
            actual_path.is_relative_to(confined)
            and read(actual_path, 64 * 1024**2) == expected_blob,
            "Current cached processor metadata differs: " + label,
        )
    require(
        exact(parse(metadata["generation_config.json"])["eos_token_id"], [1, 106, 50]),
        "Exact generation EOS contract",
    )
    if model is not None:
        components = {
            "processor": model.processor,
            "tokenizer": model.tok,
            "feature_extractor": model.processor.feature_extractor,
            "image_processor": model.processor.image_processor,
            "video_processor": model.processor.video_processor,
        }
        require(model.tok is model.processor.tokenizer, "Native tokenizer object differs")
        for name, component in components.items():
            spec = closure["components"][name]
            require(
                type(component).__name__ == spec["class"]
                and str(Path(inspect.getfile(type(component))).resolve()) == spec["path"],
                "Actual native processor component differs",
            )
        require(
            exact(model.lm.generation_config.eos_token_id, [1, 106, 50])
            and model.tok.convert_tokens_to_ids("<turn|>") == 106,
            "Actual generation EOS contract differs",
        )
    return versions


def compile_public_actions(model, inputs, *, output, gate):
    require(type(inputs) is VerifiedPublicFitInputs, "Verified public input type required")
    inputs.check()
    kernel = trusted_module("jev_browser_fit_actions", "scripts/jev_browser_fit_actions.py")
    admitted, rows, compiled_pins = inputs.admitted(), {}, {}
    references = dict(inputs.reference_items)
    for split in COUNTS:
        cases = admitted.train_cases if split == "train" else admitted.development_cases
        projection = b"".join(
            encode(w) + b"\n" for w in admitted.compile_wire_projection(split, model_alias=IDENTITY)
        )
        require(
            projection == references["compile-projection-" + split + ".jsonl"],
            "Exact full400 public projection",
        )
        reference = [parse(b) for b in references["compiled-" + split + ".jsonl"].splitlines()]
        rows[split] = []
        for index, case in enumerate(cases):
            gate.check("before-BF16-processor-compile-" + split + "-" + str(index), "train")
            wire = admitted.compile_wire_projection(split, model_alias=IDENTITY)[index]
            actual = kernel.compile_actions(
                model,
                [case],
                {"blobs": {"wire_requests.jsonl": encode(wire) + b"\n"}},
                identity=IDENTITY,
            )
            require(
                len(actual) == 1 and encode(serialized_row(actual[0])) == encode(reference[index]),
                "Every prompt/ID/target/position/mask must equal genuine CPU reference",
            )
            rows[split].append(actual[0])
        require(
            len(rows[split]) == len(reference) == COUNTS[split], "Complete142 compile population"
        )
        blob = b"".join(encode(serialized_row(row)) + b"\n" for row in rows[split])
        require(
            blob == references["compiled-" + split + ".jsonl"],
            "Full serialized compiled bytes differ",
        )
        create(Path(output) / ("compiled-" + split + ".jsonl"), blob)
        compiled_pins[split] = {"sha256": sha(blob), "size_bytes": len(blob)}
    inputs.check()
    return rows, compiled_pins


def tensor_metadata(blob, expected_shapes, *, gate):
    gate.check("before-public-child-tensor-verification", "cpu")
    import torch
    from safetensors.torch import load

    tensors = load(blob)
    require(
        len(tensors) == 176 and set(tensors) == set(expected_shapes),
        "All176 public q/v tensors required",
    )
    shapes = {}
    for name, tensor in tensors.items():
        match = re.fullmatch(
            r"base_model\.model\.model\.language_model\.layers\.[0-9]+\.self_attn\.(q_proj|v_proj)\.lora_([AB])\.weight",
            name,
        )
        require(
            match is not None
            and tensor.dtype == torch.float32
            and tensor.ndim == 2
            and list(tensor.shape) == expected_shapes[name]
            and torch.isfinite(tensor).all()
            and tensor.shape[0 if match.group(2) == "A" else 1] == 8,
            "Finite exact-shape FP32 q/v tensors required",
        )
        shapes[name] = list(tensor.shape)
    require(
        sum(t.numel() for t in tensors.values()) == 5193728, "Exact original LoRA parameter count"
    )
    return {
        "tensor_count": 176,
        "parameters": 5193728,
        "stored_dtypes": ["torch.float32"],
        "tensor_shapes": shapes,
    }


MANIFEST_KEYS = {
    "schema_version",
    "kind",
    "base_model_id",
    "base_revision",
    "base_checkpoint",
    "training_quantization",
    "training_dtype",
    "head_sha256",
    "temperature",
    "policy",
    "instructions_sha256",
    "inference_model_identity",
    "lora",
    "files",
    "training_receipt",
    "provenance",
    "public_inputs",
    "parent",
    "source_handoff",
    "compiled_files",
    "tensor_metadata",
    "deployment_contract",
    "qualification_required",
}
DEPLOYMENT = {
    "quantization": "none",
    "dtype": "bfloat16",
    "model_identity": IDENTITY,
    "qualified": False,
    "candidate_cap": 400,
    "context": 8192,
    "max_new_tokens": 32,
    "grammar": "legal_action_token_trie_v1",
}


def validate_manifest(manifest):
    require(
        type(manifest) is dict and set(manifest) == MANIFEST_KEYS,
        "Strict separate public manifest schema",
    )
    pins = {
        "schema_version": 1,
        "kind": KIND,
        "base_model_id": MODEL_ID,
        "base_revision": REVISION,
        "base_checkpoint": BASE_CHECKPOINT,
        "training_quantization": "none",
        "training_dtype": "bfloat16",
        "head_sha256": HEAD_SHA,
        "temperature": TEMPERATURE,
        "policy": "greedy_legal_action_v2",
        "instructions_sha256": "63874de795580b95edbcb1f7c6a6857e86f458dabba77e33814d8f08f158d68a",
        "inference_model_identity": IDENTITY,
        "lora": LORA,
        "deployment_contract": DEPLOYMENT,
        "qualification_required": QUALIFICATION_KIND,
    }
    require(
        all(exact(manifest[k], v) for k, v in pins.items()),
        "Public checkpoint/policy/BF16/qualification contract differs",
    )
    require(
        set(manifest["files"]) == {"adapter_config.json", "adapter_model.safetensors"}
        and set(manifest["training_receipt"]) == {"path", "sha256"}
        and manifest["training_receipt"]["path"] == "../receipt.json",
        "Public adapter files/receipt contract",
    )


@dataclass(frozen=True)
class VerifiedPublicActionArtifact:
    """Distinct frozen type, intentionally unusable by legacy attach functions."""

    manifest_bytes: bytes
    receipt_bytes: bytes
    adapter_items: tuple[tuple[str, bytes], ...]
    tensor_metadata_bytes: bytes

    def manifest(self):
        return parse(self.manifest_bytes)

    def telemetry(self):
        manifest = self.manifest()
        return {
            "kind": KIND,
            "manifest_sha256": sha(self.manifest_bytes),
            "receipt_sha256": sha(self.receipt_bytes),
            "qualified": False,
            "qualification_required": QUALIFICATION_KIND,
            "task_score": None,
            "tensor_count": parse(self.tensor_metadata_bytes)["tensor_count"],
            "training_steps": parse(self.receipt_bytes)["training"]["completed_updates"],
            "model_identity": manifest["inference_model_identity"],
        }


def verify_public_action_artifact(
    adapter_path,
    *,
    expected_manifest_sha256,
    expected_receipt_sha256,
    expected_source_handoff,
    gate,
):
    path = ordinary(adapter_path)
    blob = read(path / "manifest.json")
    require(
        sha(blob) == expected_manifest_sha256, "Externally selected public manifest SHA differs"
    )
    return _verify_public_material(
        path,
        blob,
        expected_receipt_sha256=expected_receipt_sha256,
        expected_source_handoff=expected_source_handoff,
        gate=gate,
    )


def _verify_public_material(
    path,
    manifest_blob,
    *,
    expected_receipt_sha256,
    expected_source_handoff,
    gate,
    _producer_journal=False,
):
    """Also validates the candidate BEFORE the final manifest publication rename."""
    gate.check("before-public-artifact-tree-or-parent-payload", "cpu")
    manifest = parse(manifest_blob)
    validate_manifest(manifest)
    receipt_blob = read(path.parent / "receipt.json")
    require(
        sha(receipt_blob) == expected_receipt_sha256 == manifest["training_receipt"]["sha256"],
        "Public completed receipt hash differs",
    )
    receipt = parse(receipt_blob)
    require(
        receipt["kind"] == KIND
        and receipt["stage"] == "complete"
        and receipt["error"] is None
        and receipt["task_score"] is None
        and receipt["qualified"] is False,
        "Completed unqualified public fit required",
    )
    handoff_blob = read(path.parent / "source-handoff.json")
    handoff = verify_source_handoff(handoff_blob, expected_sha256=expected_source_handoff)
    require(
        manifest["source_handoff"]
        == {
            "path": "source-handoff.json",
            "sha256": expected_source_handoff,
            "identity_sha256": handoff["identity_sha256"],
        },
        "Public source handoff binding",
    )
    expected = receipt["output_inventory_excluding_receipt_and_manifest"]
    actual = tree_sizes(path.parent)
    ignored = {"receipt.json", "adapter/manifest.json"}
    # Only the internal producer may still append its own gate journal. Its
    # final immutable inventory/receipt is sealed AFTER this semantic pass.
    mutable = {"resource-gates.jsonl"} if _producer_journal else set()
    require(set(actual) - ignored == set(expected), "Public closed artifact population")
    require(
        {k: n for k, n in actual.items() if k not in ignored | mutable}
        == {k: p["size_bytes"] for k, p in expected.items() if k not in mutable},
        "Public closed artifact sizes",
    )
    for label, entry in expected.items():
        if label not in mutable:
            require(
                pin(path.parent / relative(label)) == entry,
                "Public artifact inventory content changed",
            )
    for label, entry in handoff["files"].items():
        require(pin(path.parent / "sources" / label) == entry, "Preserved90 source differs")
    for name, digest in FIXED_METADATA.items():
        require(
            sha(read(path.parent / "inputs/metadata" / name)) == digest,
            "Bound historical metadata changed",
        )
    stage = parse(read(path.parent / "inputs/metadata/preflight-receipt.json"))
    parent_blob = read(path.parent / "inputs/metadata/parent-capture.json")
    require(
        {"sha256": sha(parent_blob), "size_bytes": len(parent_blob)}
        == stage["staged_evidence_files_excluding_receipt"]["parent-capture.json"],
        "Authenticated historical parent descriptor SHA differs",
    )
    expected_parent_descriptor = parse(parent_blob)
    public_files = stage["public_original_inventory"]["files"]
    check_inventory(path.parent / "inputs/public_collection", public_files)
    parent_descriptor = manifest["parent"]
    validate_expected_parent_descriptor(parent_descriptor, expected_parent_descriptor)
    require(
        parent_descriptor == receipt["initial_action_adapter"]
        and len(parent_descriptor["files"]) == PARENT_FILE_COUNT
        and parent_descriptor["historical_completed_updates"]
        == parent_descriptor["historical_training_cases"]
        == 521,
        "Explicit full legacy parent lineage descriptor",
    )
    check_inventory(path.parent / "inputs/initial_action_adapter", parent_descriptor["files"])
    gate.check("before-legacy-parent-only-verifier", "cpu")
    api = trusted_module("s1.browser_action_adapter", "src/s1/browser_action_adapter.py")
    parent = api.verify_action_adapter(path.parent / "inputs/initial_action_adapter/adapter")
    require(
        {
            "manifest_sha256": parent.manifest_sha256,
            "receipt_sha256": parent.receipt_sha256,
            "weights_sha256": parent.manifest["files"]["adapter_model.safetensors"]["sha256"],
        }
        == PARENT_PINS,
        "Historical parent pins differ",
    )
    require(
        parent.tensor_metadata["tensor_count"] == 176
        and parent.tensor_metadata["parameters"] == 5193728
        and parent.tensor_metadata["stored_dtypes"] == ["torch.float32"]
        and sum(p["size_bytes"] for p in parent_descriptor["files"].values()) == PARENT_TOTAL_BYTES
        and parent_descriptor["identity_sha256"] == identity(parent_descriptor["files"]),
        "Complete496-file parent FP32 inventory identity",
    )
    initial = parent_descriptor["initial_tensor_verification"]
    require(
        type(initial) is dict
        and set(initial)
        == {
            "success",
            "tensor_count",
            "tensor_shapes",
            "weights_sha256",
            "all_tensors_exact",
            "all_trainables_qv_fp32",
            "frozen_base_versions_unchanged",
            "full_head_identity_version_preserved",
        }
        and all(
            initial[key] is True
            for key in (
                "success",
                "all_tensors_exact",
                "all_trainables_qv_fp32",
                "frozen_base_versions_unchanged",
                "full_head_identity_version_preserved",
            )
        )
        and initial["weights_sha256"] == PARENT_PINS["weights_sha256"]
        and initial["tensor_count"] == 176
        and initial["tensor_shapes"] == parent.tensor_metadata["tensor_shapes"],
        "Exact parent initialization evidence",
    )
    api.verify_initial_action_adapter_splits(
        path.parent / "inputs/initial_action_adapter", manifest["provenance"], head_sha256=HEAD_SHA
    )
    admitted = trusted_module(
        "s1.browser_public_training_validation", "src/s1/browser_public_training_validation.py"
    ).validate_public_training_collection(
        path.parent / "inputs/public_collection",
        reviewed_source_pins={
            k.removeprefix("src/"): v
            for k, v in handoff["files"].items()
            if k
            in parse(
                read(
                    ROOT / "artifacts/jev-runtime/frozen-public-observation-36/freeze-manifest.json"
                )
            )["files"]
        },
        tokenizer_cache=Path(receipt["tokenizer_cache"]),
    )
    provenance = validate_public_scope(admitted, stage["public_original_inventory"])
    require(
        provenance == manifest["provenance"] == receipt["provenance"],
        "Public full split provenance changed",
    )
    public_inputs = {
        "inventory": stage["public_original_inventory"],
        "observation_profile": admitted.provenance["observation_profile"],
        "metadata_files": {
            name: pin(path.parent / "inputs/metadata" / name)
            for name in (*FIXED_METADATA, "parent-capture.json")
        },
        "wire_identities": admitted.provenance["wire_identities"],
    }
    require(
        manifest["public_inputs"] == receipt["public_inputs"] == public_inputs,
        "Exact public input/proof/profile/wire bindings",
    )
    reference = parse(read(path.parent / "inputs/reference_compile/compiler-receipt.json"))
    require(
        sha(read(path.parent / "inputs/reference_compile/compiler-receipt.json"))
        == COMPILER_RECEIPT_SHA,
        "Reference compilation receipt changed",
    )
    check_inventory(
        path.parent / "inputs/reference_compile",
        {
            **reference["output_inventory_excluding_receipt"],
            "compiler-receipt.json": pin(
                path.parent / "inputs/reference_compile/compiler-receipt.json"
            ),
        },
    )
    reference_items = tuple(
        (label, read(path.parent / "inputs/reference_compile" / label, 64 * 1024**2))
        for label in {
            **reference["output_inventory_excluding_receipt"],
            "compiler-receipt.json": None,
        }
    )
    require(
        verify_reference_environment(reference_items) == receipt["versions"],
        "Public fitting processor/framework version metadata differs",
    )
    for split in COUNTS:
        compiled = path.parent / ("compiled-" + split + ".jsonl")
        require(
            pin(compiled)
            == manifest["compiled_files"][split]
            == reference["completed_row_files"][split],
            "Public compiled reference identity differs",
        )
    validate_training_receipt(
        receipt, provenance, expected_tensor_shapes=parent.tensor_metadata["tensor_shapes"]
    )
    validate_resource_evidence(
        read(path.parent / "resource-gates.jsonl"),
        steps=receipt["steps"],
        _producer_pending=_producer_journal,
    )
    files = {name: read(path / name, 64 * 1024**2) for name in manifest["files"]}
    for name, blob in files.items():
        require(
            {"sha256": sha(blob), "size_bytes": len(blob)} == manifest["files"][name],
            "Child adapter bytes changed",
        )
    api._config(parse(files["adapter_config.json"]))
    tensors = tensor_metadata(
        files["adapter_model.safetensors"], parent.tensor_metadata["tensor_shapes"], gate=gate
    )
    require(
        tensors == manifest["tensor_metadata"] == receipt["tensor_metadata"],
        "Exact public176FP32 tensor inventory",
    )
    verify_source_handoff(handoff_blob, expected_sha256=expected_source_handoff)
    for label, entry in expected.items():
        if label not in mutable:
            require(
                pin(path.parent / label) == entry, "Public artifact changed during verification"
            )
    return VerifiedPublicActionArtifact(
        bytes(manifest_blob), bytes(receipt_blob), tuple(sorted(files.items())), encode(tensors)
    )


def validate_training_receipt(receipt, provenance, *, expected_tensor_shapes=None):
    training = receipt["training"]
    require(
        type(training["development_loss_mean"]) in (int, float)
        and math.isfinite(training["development_loss_mean"])
        and training["development_loss_mean"] >= 0
        and type(training["elapsed_ms"]) in (int, float)
        and math.isfinite(training["elapsed_ms"])
        and training["elapsed_ms"] >= 0,
        "Finite teacher-forced training diagnostics required",
    )
    require(
        all(
            type(value) is int
            for value in (
                receipt["steps"],
                receipt["training_updates"],
                receipt["model_loads"],
                training["completed_updates"],
                training["development_updates"],
                training["development_examples"],
                training["trainable_parameters"],
            )
        )
        and training["completed_updates"] == receipt["steps"] > 0
        and training["development_examples"] == 22
        and training["development_updates"] == 0
        and training["original_head_sha256_before"]
        == training["original_head_sha256_after"]
        == HEAD_SHA
        and training["frozen_base_versions_unchanged"] is True
        and receipt["temperature_before"] == receipt["temperature_after"] == TEMPERATURE
        and receipt["training_updates"] == receipt["steps"]
        and receipt["model_loads"] == 1
        and exact(receipt["base_checkpoint"], BASE_CHECKPOINT)
        and type(receipt["seed"]) is int
        and type(receipt["learning_rate"]) in (int, float)
        and math.isfinite(receipt["learning_rate"])
        and receipt["learning_rate"] > 0,
        "Public training progress/zero-development/frozen-head evidence",
    )
    trainable = training["trainable_names"]
    require(
        type(trainable) is list
        and len(trainable) == len(set(trainable)) == 176
        and training["trainable_parameters"] == 5193728
        and all(
            type(name) is str
            and re.fullmatch(
                r"base_model\.model\.model\.language_model\.layers\.[0-9]+\.self_attn\.(q_proj|v_proj)\.lora_[AB]\.default\.weight",
                name,
            )
            for name in trainable
        ),
        "Exactly176 named q/v FP32 trainables and5193728 parameters",
    )
    if expected_tensor_shapes is not None:
        require(
            {name.replace(".default.", ".") for name in trainable} == set(expected_tensor_shapes),
            "Trainable keys differ from all176 parent/child matrices",
        )
    calls = training["trace"]["backbone_calls"]
    train = [call for call in calls if call["split"] == "train"]
    dev = [call for call in calls if call["split"] == "development"]
    require(
        len(calls) == len(train) + len(dev) == receipt["steps"] + 22
        and len(train) == receipt["steps"]
        and len(dev) == 22
        and all(type(call["step"]) is int for call in calls)
        and [call["step"] for call in train] == list(range(1, receipt["steps"] + 1))
        and all(
            call["phase"] == "forward" and call["case_id"] in provenance["train"]["ids"]
            for call in train
        )
        and all(call["phase"] == "evaluation" for call in dev)
        and sorted(call["case_id"] for call in dev) == provenance["development"]["ids"]
        and all(
            type(call["input_shape"]) is list
            and len(call["input_shape"]) == 2
            and call["input_shape"][0] == 1
            and 1 < call["input_shape"][1] <= 8192
            for call in calls
        ),
        "Native train/eval full-row trace or batch scope differs",
    )
    require(
        all(type(n) is int for call in calls for n in call["input_shape"]),
        "Strict integer native sequence shapes",
    )
    for start in range(0, len(train), 120):
        epoch = [call["case_id"] for call in train[start : start + 120]]
        require(
            len(set(epoch)) == len(epoch)
            and (len(epoch) < 120 or sorted(epoch) == provenance["train"]["ids"]),
            "Every complete120-step epoch covers each train row exactly once",
        )


def resource_phase_scope(phase):
    require(type(phase) is str, "Resource phase text required")
    cpu = {
        "before-any-project-framework-or-parent-payload",
        "before-public-project-admission",
        "immediately-before-legacy-parent-capture-and-Torch",
        "before-parent-recursive-tensor-authentication",
        "before-public-child-tensor-verification",
        "before-public-artifact-tree-or-parent-payload",
        "before-legacy-parent-only-verifier",
        "before-bounded-partial-adapter-preservation",
    }
    train = {
        "before-any-BF16-model-construction",
        "before-parent-injection",
        "immediately-before-unchanged-train_actions-optimizer",
        "before-native-train-forward",
        "before-native-development-forward",
    }
    if phase in cpu:
        return "cpu"
    if phase in train or re.fullmatch(
        r"before-BF16-processor-compile-(train|development)-[0-9]+", phase
    ):
        return "train"
    require(
        phase == "before-checkpoint-payload-hash-and-BF16-loader",
        "Unknown protected resource phase",
    )
    return "load"


def validate_resource_evidence(blob, *, steps, _producer_pending=False):
    require(type(steps) is int and steps > 0, "Positive integer recorded steps")
    rows = [parse(line) for line in blob.splitlines()]
    require(
        0 < len(rows) <= 512 and all(len(line) <= 16384 for line in blob.splitlines()),
        "Bounded resource evidence",
    )
    boot, previous = rows[0]["expected_boot_id"], -1
    require(type(boot) is str and bool(boot), "Recorded launch boot required")
    phases = []
    for index, row in enumerate(rows):
        observation = row["observation"]
        scope = row["scope"]
        require(
            scope == resource_phase_scope(row["phase"])
            and type(row["sequence"]) is int
            and row["sequence"] == index
            and row["admitted"] is True
            and row["error"] is None
            and row["monotonic"] > previous
            and observation["boot_id"] == row["expected_boot_id"] == boot
            and type(observation["MemTotal"]) is int
            and type(observation["MemAvailable"]) is int,
            "Same-boot successful resource evidence/order",
        )
        require(
            type(row["monotonic"]) in (int, float)
            and math.isfinite(row["monotonic"])
            and all(
                type(row[key]) is int
                for key in ("reserve_bytes", "required_bytes", "swap_or_resident_credit_bytes")
            ),
            "Strict finite resource scalar fields",
        )
        reserve = max(24 * GIB, (observation["MemTotal"] + 4) // 5)
        load = 2 * 23919549408 + reserve
        required = {
            "cpu": reserve + 4 * GIB,
            "load": load,
            "train": load + 30 * GIB + 256 * 1024**2,
        }[scope]
        require(
            row["reserve_bytes"] == reserve
            and row["required_bytes"] == required
            and row["swap_or_resident_credit_bytes"] == 0
            and 0 < required <= observation["MemAvailable"] <= observation["MemTotal"],
            "Recorded prospective resource formula",
        )
        previous = row["monotonic"]
        phases.append(row["phase"])
    ordered = [
        "before-any-project-framework-or-parent-payload",
        "before-public-project-admission",
        "immediately-before-legacy-parent-capture-and-Torch",
        "before-parent-recursive-tensor-authentication",
        "before-checkpoint-payload-hash-and-BF16-loader",
        "before-any-BF16-model-construction",
        "before-parent-injection",
    ]
    ordered += [
        "before-BF16-processor-compile-" + split + "-" + str(index)
        for split, count in COUNTS.items()
        for index in range(count)
    ]
    ordered += ["immediately-before-unchanged-train_actions-optimizer"]
    ordered += ["before-native-train-forward"] * steps + ["before-native-development-forward"] * 22
    ordered += [
        "before-public-child-tensor-verification",
        "before-public-artifact-tree-or-parent-payload",
        "before-legacy-parent-only-verifier",
    ]
    if not _producer_pending:
        ordered += ["before-public-child-tensor-verification"]
    require(phases == ordered, "Exact ordered unique142 compile and whole train/dev gates required")


def create(path, blob):
    path = ordinary(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        require(stream.write(blob) == len(blob), "Short evidence write")
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def copy_items(destination, items):
    for label, blob in items:
        create(Path(destination) / relative(label), blob)
    check_inventory(destination, {k: {"sha256": sha(v), "size_bytes": len(v)} for k, v in items})
