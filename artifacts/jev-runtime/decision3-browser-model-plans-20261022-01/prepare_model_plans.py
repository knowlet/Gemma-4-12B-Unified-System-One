"""One default-dry preparer for the remaining Decision3 selections; no actors."""

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

ROOT = Path(__file__).absolute().parents[3]
HERE = Path(__file__).absolute().parent
CATALOG = HERE / "model-selections.json"
CATALOG_SHA = "b70a1a142dc7c198c55b93f06b1461bc6ce9a5e610c1b69bead40519487238ed"
PREPARER = (
    ROOT
    / "artifacts/jev-runtime/decision3-edge-browser-retry-plan-20261022-01/prepare_edge_plan.py"
)
PREPARER_SHA = "5696c8fc703546c95dca6e5b959cc1074941a29b68ae606db6d5a9e0520c0293"
EXECUTION_SHA = "248687b7dddc5cf7697482e85b2ec909f1c04cbecc5e258d0f23478ecbf3745f"
MODELS = ("d3", "d3-flash", "d3-mini", "d3-nano", "d3-lite")
GIB = 1024**3


def need(value, message):
    if not value:
        raise ValueError(message)


def preparer():
    """Load the reviewed metadata helpers from checked bytes, never cached code."""
    need(
        not any(p.is_symlink() for p in (PREPARER, *PREPARER.parents)), "Ordinary reviewed preparer"
    )
    blob = PREPARER.read_bytes()
    need(
        len(blob) == 11335 and hashlib.sha256(blob).hexdigest() == PREPARER_SHA,
        "Reviewed preparer source",
    )
    spec = importlib.util.spec_from_file_location("_remaining_selected_preparation", PREPARER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(blob, str(PREPARER), "exec", dont_inherit=True), module.__dict__)
    return module


def catalogue():
    p = preparer()
    value = json.loads(p.checked(CATALOG, CATALOG_SHA))
    need(
        value["status"] == "source_only_unsealed_awaiting_future_helper03",
        "Explicit future-helper preparation",
    )
    for ref in value["refs"].values():
        p.verify_ref(ref)
    checkpoints = json.loads(p.verify_ref(value["refs"]["checkpoints"])[1])
    rows = {r["model"]: r for r in checkpoints["models"]}
    need(
        value["all_six_identities"] == {m: r["registry_identity"] for m, r in rows.items()},
        "Exact all-six checkpoint identities",
    )
    need(set(value["templates"]) == set(MODELS), "All remaining five selections")
    base = p.selection()
    need(
        base["components"]["execution"]["sha256"] == EXECUTION_SHA
        and value["refs"]["final_execution"] == base["components"]["execution"],
        "Unchanged selected execution",
    )
    need(
        value["host"]
        == {
            "boot_id": base["boot_id"],
            "gpu_uuid": base["gpu_uuid"],
            "mem_total_bytes": base["mem_total_bytes"],
        },
        "Same selected host",
    )
    need(
        value["owner"] == {"memory_max_bytes": 80 * GIB, "runtime_seconds": 7200, "swap_bytes": 0},
        "Original approved D3 owner profile",
    )
    reserve = max(24 * GIB, (value["host"]["mem_total_bytes"] + 4) // 5)
    units, nonces, outputs = [], [], []
    for model, template in value["templates"].items():
        floor = reserve + 2 * rows[model]["registry_identity"]["resident_file_bytes"] + 2 * GIB
        need(
            floor == template["core_floor_bytes"] == rows[model]["core_floor_bytes"],
            "Unchanged integer CORE floor",
        )
        need(
            template["score"] is None
            and template["physical_capacity_sufficient"]
            is (floor <= value["host"]["mem_total_bytes"]),
            "Capacity is separate from score",
        )
        if model == "d3":
            need(
                floor > value["host"]["mem_total_bytes"]
                and template["plan_path"] is None
                and template["launch_argv"] is None,
                "Full model refuses before preparation",
            )
        else:
            units.extend(template["units"].values())
            nonces.extend(template["nonces"].values())
            outputs.extend([template["output"], *template["model_outputs"].values()])
    need(
        len(units) == len(set(units)) == len(nonces) == len(set(nonces)) == 8,
        "Independent fresh groups for four admissible models",
    )
    need(len(outputs) == len(set(outputs)) == 12, "Distinct model and campaign outputs")
    need(
        not set(units) & set(base["units"].values())
        and not set(nonces) & set(base["nonces"].values()),
        "Preserve original Edge slots",
    )
    return value, p, base


def capacity_refusal(value, p):
    record = json.loads(p.verify_ref(value["refs"]["capacity_refusal"])[1])
    template = value["templates"]["d3"]
    need(
        record["model"] == "d3"
        and record["registry_identity"] == value["all_six_identities"]["d3"],
        "Exact full-model capacity record",
    )
    need(
        record["core_floor_bytes"] == template["core_floor_bytes"]
        and record["mem_total_bytes"] == value["host"]["mem_total_bytes"]
        and record["score"] is None
        and record["launch_argv"] is None,
        "Capacity refusal has no quality score or launch",
    )
    return record


def selected_model(value, base, model):
    need(model in MODELS and model != "d3", "Only admissible remaining selections have plans")
    result = copy.deepcopy(base)
    result["model"] = model
    result["registry_identity"] = copy.deepcopy(value["all_six_identities"][model])
    result["helper_seed"] = copy.deepcopy(value["future_helper_seed"])
    result["policy"]["native_floor_bytes"] = value["templates"][model]["core_floor_bytes"]
    for key in (
        "units",
        "nonces",
        "cgroups",
        "output",
        "model_outputs",
        "descriptor_paths",
        "plan_path",
        "receipt_path",
    ):
        result[key] = copy.deepcopy(value["templates"][model][key])
    return result


def descriptors(p, value, seed):
    # Reuse the exact thirteen-field host/helper/path/interpreter checks.
    result = p.derive_descriptors(value, seed)
    for group, descriptor in result.items():
        descriptor["co_resident"] = {
            "model": value["model"],
            "resident_file_bytes": value["registry_identity"]["resident_file_bytes"],
            "cgroup": value["cgroups"][group],
        }
    return result


def prepare(value, p, base, model, seed_path=None, seed_sha=None, *, expires_after_seconds=9000):
    need(model in MODELS, "Explicit remaining model selection")
    need(
        Path("/proc/sys/kernel/random/boot_id").read_text().strip() == value["host"]["boot_id"],
        "Fresh same boot",
    )
    total = next(
        int(x.split()[1]) * 1024
        for x in Path("/proc/meminfo").read_text().splitlines()
        if x.startswith("MemTotal:")
    )
    need(total == value["host"]["mem_total_bytes"], "Fresh same host total")
    if model == "d3":
        # This branch precedes seed reads, checker creation and all plan writes.
        return capacity_refusal(value, p)
    need(
        math.isfinite(expires_after_seconds) and expires_after_seconds > 0,
        "Positive finite launch expiry",
    )
    selected = selected_model(value, base, model)
    need(
        seed_path is not None
        and seed_sha is not None
        and p.ordinary(seed_path) == Path(selected["helper_seed"]["path"]),
        "Exact future helper03 seed path and external SHA",
    )
    seed = json.loads(p.checked(seed_path, seed_sha, limit=1024**2))
    derived = descriptors(p, selected, seed)
    execution = p.load(selected["components"]["execution"], "_remaining_selected_execution")
    checker = p.load(
        selected["components"]["helper_checker"], "_remaining_selected_helper_metadata"
    )
    with tempfile.TemporaryDirectory(prefix=".static-helper-", dir=HERE) as directory:
        for group, descriptor in derived.items():
            receipts = Path(directory) / group
            receipts.mkdir()
            blob = execution.encode(descriptor)
            instance = checker.LiveHelperChecker(
                selection_blob=blob,
                selection_sha=hashlib.sha256(blob).hexdigest(),
                receipt_directory=receipts,
            )
            need(
                instance.model == model
                and instance.resident_bytes == selected["registry_identity"]["resident_file_bytes"],
                "Exact selected model and resident bytes",
            )
    metadata = [
        Path(selected["plan_path"]),
        Path(selected["receipt_path"]),
        *(Path(x) for x in selected["descriptor_paths"].values()),
    ]
    outputs = [Path(selected["output"]), *(Path(x) for x in selected["model_outputs"].values())]
    folder = Path(selected["plan_path"]).parent
    need(
        folder.parent == HERE and folder.name == model and not folder.exists(),
        "Fresh model metadata slot",
    )
    need(
        all(not p.ordinary(x).exists() for x in [*metadata, *outputs]),
        "Fresh plan/descriptors/campaign/model outputs",
    )
    folder.mkdir(mode=0o700)
    for group, descriptor in derived.items():
        p.exclusive_json(selected["descriptor_paths"][group], descriptor)
    plan = execution.prepare_plan(
        model=model,
        helper_checker=ROOT / selected["components"]["helper_checker"]["path"],
        helper_selections=selected["descriptor_paths"],
        helper_bindings={g: x["binding"] for g, x in derived.items()},
        browser=selected["browser"],
        output=selected["output"],
        model_outputs=selected["model_outputs"],
        boot_id=selected["boot_id"],
        mem_total_bytes=selected["mem_total_bytes"],
        gpu_uuid=selected["gpu_uuid"],
        expires_monotonic=time.monotonic() + expires_after_seconds,
        units=selected["units"],
        nonces=selected["nonces"],
    )
    need(
        plan["components"] == selected["components"], "Unchanged execution/helper/worker components"
    )
    expected = sorted(
        [
            *selected["expected_base_source_refs"],
            *(p.reference(Path(x)) for x in selected["descriptor_paths"].values()),
        ],
        key=lambda x: x["path"],
    )
    need(
        plan["sources"] == expected and len(expected) == 39, "Exact37 base plus2 helper descriptors"
    )
    blob = p.exclusive_json(selected["plan_path"], plan)
    digest = hashlib.sha256(blob).hexdigest()
    execution.selected_plan(selected["plan_path"], digest)
    result = {
        "schema_id": "decision3-remaining-model-prepared-launch-v1",
        "model": model,
        "registry_identity": selected["registry_identity"],
        "plan": p.reference(Path(selected["plan_path"])),
        "selected_helper_seed": p.reference(Path(seed_path)),
        "preparation_selection": p.reference(CATALOG),
        "source_count": 39,
        "metadata_static_helper95_runtime348_verified": True,
        "selected_plan_validator_passed": True,
        "actual_live_admission_performed": False,
        "actual_model_browser_GPU_or_unit_started": False,
        "model_qualified": False,
        "score": None,
        "fixed_denominator": 10,
        "fresh_workers": {"known4": 64, "public6": 96},
        "launch_argv_for_external_owner": [
            plan["interpreter"],
            "-B",
            "-s",
            str(ROOT / plan["components"]["entry"]["path"]),
            "--execute",
            "--role",
            "controller",
            "--plan",
            selected["plan_path"],
            "--plan-sha256",
            digest,
        ],
        "scope": "Source-only concrete preparation. Independently select plan SHA before an external owner launches; fresh helper/host/ancestor-headroom/CUDA/cache/runtime and task/closure evidence remain execution facts.",
    }
    p.exclusive_json(selected["receipt_path"], result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--helper-seed")
    parser.add_argument("--helper-seed-sha256")
    parser.add_argument("--expires-after-seconds", type=float, default=9000)
    args = parser.parse_args(argv)
    value, p, base = catalogue()
    if not args.prepare:
        print(
            json.dumps(
                {
                    "mode": "DRY_PREPARATION_ONLY",
                    "model": args.model,
                    "capacity_refused": args.model == "d3",
                    "core_floor_bytes": value["templates"][args.model]["core_floor_bytes"],
                    "expected_helper_seed": value["future_helper_seed"]["path"],
                    "actual_started": False,
                    "model_qualified": False,
                    "score": None,
                }
            )
        )
        return 0
    if args.model != "d3" and (not args.helper_seed or not args.helper_seed_sha256):
        parser.error(
            "Preparation requires the actual fresh helper03 seed and independently selected SHA"
        )
    result = prepare(
        value,
        p,
        base,
        args.model,
        args.helper_seed,
        args.helper_seed_sha256,
        expires_after_seconds=args.expires_after_seconds,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
