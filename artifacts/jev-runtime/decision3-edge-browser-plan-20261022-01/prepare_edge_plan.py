"""Default metadata-only; explicit preparation saves a plan and launches nothing."""

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SELECTION = HERE / "preparation-selection.json"
SELECTION_SHA = "e73cb364b65c31398fa5d3da0e9e071199e0e42f7e56c74f751474caa57e58e4"
EXECUTION_SHA = "901549c3108d572b04e8402c46d7953c982860d41a22d3f2bc8087a8baa2bf5c"
HELPER_SHA = "b20cdf98a7db4283f75fc8453ce2eafe839b33bfcec35ca2582249fa69b61994"
SEED_FIELDS = set(
    "schema_id binding worker listener service_directory private_config interpreter source_root qualification execution runtime registry expected_memtotal".split()
)


def need(value, message):
    if not value:
        raise ValueError(message)


def ordinary(path):
    path = Path(path).absolute()
    need(
        ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "ordinary selected path",
    )
    return path


def checked(path, expected_sha, limit=2 * 1024**2):
    path = ordinary(path)
    before = path.stat()
    need(0 < before.st_size <= limit, "bounded selected metadata/source")
    blob = path.read_bytes()
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    need(
        all(getattr(before, n) == getattr(after, n) for n in fields),
        "selected bytes changed during read",
    )
    need(hashlib.sha256(blob).hexdigest() == expected_sha, "externally selected SHA differs")
    return blob


def reference(path):
    path = ordinary(path)
    blob = path.read_bytes()
    need(0 < len(blob) <= 2 * 1024**2, "small reference only")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "size_bytes": len(blob),
    }


def verify_ref(ref):
    need(
        type(ref) is dict and set(ref) == {"path", "sha256", "size_bytes"}, "exact source reference"
    )
    path = ordinary(ROOT / ref["path"])
    need(
        not Path(ref["path"]).is_absolute() and path.is_relative_to(ROOT),
        "repository-relative reference",
    )
    blob = checked(path, ref["sha256"])
    need(len(blob) == ref["size_bytes"], "selected size differs")
    return path, blob


def load(ref, name):
    path, blob = verify_ref(ref)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(blob, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


def selection():
    value = json.loads(checked(SELECTION, SELECTION_SHA))
    need(value["status"] == "unsealed-awaiting-actual-helper-seed", "explicit preparation scope")
    need(
        value["components"]["execution"]["sha256"] == EXECUTION_SHA
        and value["components"]["helper_checker"]["sha256"] == HELPER_SHA,
        "fixed reviewed composition",
    )
    for ref in [
        *value["expected_base_source_refs"],
        *value["components"].values(),
        *value["context_refs"].values(),
    ]:
        verify_ref(ref)
    return value


def derive_descriptors(value, seed):
    """Only public seed facts; real callbacks perform fresh live admission later."""
    need(type(seed) is dict and set(seed) == SEED_FIELDS, "exact thirteen-field helper seed")
    need(seed["schema_id"] == "decision3-q95-live-helper-selection-v1", "Q95 helper seed schema")
    need(
        seed["binding"]["boot_id"] == value["boot_id"]
        and seed["expected_memtotal"] == value["mem_total_bytes"],
        "selected same-boot host identity",
    )
    need(seed["worker"]["cgroup"] == value["helper_seed"]["cgroup"], "selected helper owner cgroup")
    need(
        seed["source_root"] == value["helper_seed"]["source_root"]
        and seed["service_directory"] == value["helper_seed"]["service_directory"],
        "exact frozen Q95 source/service root",
    )
    need(
        seed["interpreter"] == value["browser"]["interpreter"],
        "selected Q95/model/browser interpreter",
    )
    result = {}
    for group in ("known4", "public6"):
        descriptor = copy.deepcopy(seed)
        descriptor["co_resident"] = {
            "model": "d3-edge",
            "resident_file_bytes": value["registry_identity"]["resident_file_bytes"],
            "cgroup": value["cgroups"][group],
        }
        result[group] = descriptor
    return result


def exclusive_json(path, value):
    blob = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n"
    )
    with ordinary(path).open("xb") as stream:
        stream.write(blob)
    return blob


def prepare(value, seed_path, seed_sha, *, expires_after_seconds):
    need(
        math.isfinite(expires_after_seconds) and expires_after_seconds > 0,
        "positive finite launch expiry",
    )
    need(
        ordinary(seed_path) == Path(value["helper_seed"]["path"]), "exact planned helper seed path"
    )
    seed_blob = checked(seed_path, seed_sha, limit=1024**2)
    seed = json.loads(seed_blob)
    need(
        Path("/proc/sys/kernel/random/boot_id").read_text().strip() == value["boot_id"],
        "fresh selected host boot",
    )
    memtotal = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemTotal:")
    )
    need(memtotal == value["mem_total_bytes"], "fresh selected host MemTotal")
    descriptors = derive_descriptors(value, seed)
    execution = load(value["components"]["execution"], "_selected_first_edge_preparation")
    checker = load(value["components"]["helper_checker"], "_selected_first_edge_helper_metadata")
    # Constructor checks only schema and saved95/348/registry bytes. __call__ and
    # text_connection (live process/auth checks) are deliberately never invoked.
    with tempfile.TemporaryDirectory(prefix=".static-helper-", dir=HERE) as tmp:
        for group, descriptor in descriptors.items():
            receipt_dir = Path(tmp) / group
            receipt_dir.mkdir()
            blob = execution.encode(descriptor)
            instance = checker.LiveHelperChecker(
                selection_blob=blob,
                selection_sha=hashlib.sha256(blob).hexdigest(),
                receipt_directory=receipt_dir,
            )
            need(
                instance.model == "d3-edge" and instance.resident_bytes == 1189722528,
                "unchanged Edge resident selection",
            )
    new_files = [
        Path(value["plan_path"]),
        HERE / "prepared-launch-01.json",
        *(Path(p) for p in value["descriptor_paths"].values()),
    ]
    fresh_outputs = [Path(value["output"]), *(Path(p) for p in value["model_outputs"].values())]
    need(
        all(not ordinary(p).exists() for p in [*new_files, *fresh_outputs]),
        "fresh unexecuted plan/descriptors/outputs",
    )
    for group, descriptor in descriptors.items():
        exclusive_json(value["descriptor_paths"][group], descriptor)
    plan = execution.prepare_plan(
        model="d3-edge",
        helper_checker=ROOT / value["components"]["helper_checker"]["path"],
        helper_selections=value["descriptor_paths"],
        helper_bindings={group: descriptor["binding"] for group, descriptor in descriptors.items()},
        browser=value["browser"],
        output=value["output"],
        model_outputs=value["model_outputs"],
        boot_id=value["boot_id"],
        mem_total_bytes=value["mem_total_bytes"],
        gpu_uuid=value["gpu_uuid"],
        expires_monotonic=time.monotonic() + expires_after_seconds,
        units=value["units"],
        nonces=value["nonces"],
    )
    need(plan["components"] == value["components"], "fixed component set unchanged")
    expected = sorted(
        [
            *value["expected_base_source_refs"],
            *(reference(Path(p)) for p in value["descriptor_paths"].values()),
        ],
        key=lambda row: row["path"],
    )
    need(
        plan["sources"] == expected and len(expected) == 39,
        "exact37 base plus two selected helper descriptors",
    )
    blob = exclusive_json(value["plan_path"], plan)
    digest = hashlib.sha256(blob).hexdigest()
    execution.selected_plan(value["plan_path"], digest)
    argv = [
        plan["interpreter"],
        "-B",
        "-s",
        str(ROOT / plan["components"]["entry"]["path"]),
        "--execute",
        "--role",
        "controller",
        "--plan",
        value["plan_path"],
        "--plan-sha256",
        digest,
    ]
    result = {
        "schema_id": "decision3-edge-prepared-launch-v1",
        "plan": reference(Path(value["plan_path"])),
        "selected_helper_seed": reference(Path(seed_path)),
        "preparation_selection": reference(SELECTION),
        "metadata_static_helper95_runtime348_verified": True,
        "selected_plan_validator_passed": True,
        "source_count": 39,
        "actual_live_admission_performed": False,
        "actual_model_browser_GPU_or_unit_started": False,
        "model_qualified": False,
        "fixed_denominator": 10,
        "fresh_workers": {"known4": 64, "public6": 96},
        "launch_argv_for_external_owner": argv,
        "scope": "Prepared concrete selected plan only. External owner must independently select its SHA and explicitly launch; fresh actual helper/host/unit/CUDA/kernel/geometry/cache/final closure remain execution facts.",
    }
    exclusive_json(HERE / "prepared-launch-01.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--helper-seed")
    parser.add_argument("--helper-seed-sha256")
    parser.add_argument("--expires-after-seconds", type=float, default=9000)
    args = parser.parse_args(argv)
    value = selection()
    if not args.prepare:
        print(
            json.dumps(
                {
                    "mode": "DRY_PREPARATION_ONLY",
                    "selection": reference(SELECTION),
                    "base_sources_verified": len(value["expected_base_source_refs"]),
                    "status": value["status"],
                    "expected_helper_seed": value["helper_seed"]["path"],
                    "actual_live_helper_bound": False,
                    "actual_actor_started": False,
                    "model_qualified": False,
                }
            )
        )
        return 0
    if not args.helper_seed or not args.helper_seed_sha256:
        parser.error(
            "--prepare requires the actual helper seed path and independently selected SHA"
        )
    result = prepare(
        value,
        args.helper_seed,
        args.helper_seed_sha256,
        expires_after_seconds=args.expires_after_seconds,
    )
    print(
        json.dumps(
            {
                "status": "prepared_unexecuted",
                "plan": result["plan"],
                "prepared_launch": reference(HERE / "prepared-launch-01.json"),
                "actual_actor_started": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
