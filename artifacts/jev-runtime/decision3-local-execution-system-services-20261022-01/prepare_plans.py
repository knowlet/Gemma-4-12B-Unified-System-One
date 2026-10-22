"""Default-dry metadata bridge from reviewed model selections to the new observer."""

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import secrets
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[2]
PARENT = ROOT / "artifacts/jev-runtime/decision3-browser-model-plans-20261022-01/prepare_model_plans.py"
PARENT_SHA = "415252cd53682a8c68b9abbd8f67f51565bedc03408e41e48a6f52cecf4148d4"
EXECUTION_SHA = "b0028af23a6c496ceab5bebf24cc948a6c1e1fc05a5a68b37c6c553220d251a9"
SEED = ROOT / "artifacts/jev-runtime/decision3-q95-owner-20261022-04/owner-04/helper-selection-seed.json"
MODELS = ("d3", "d3-flash", "d3-mini", "d3-nano", "d3-lite", "d3-edge")


def need(value, message):
    if not value:
        raise ValueError(message)


def load(path, digest, name):
    path = Path(path).absolute()
    need(path.is_relative_to(ROOT) and ".." not in path.parts, "Local selected source")
    need(not any(p.is_symlink() for p in (path, *path.parents)), "Ordinary selected source")
    data = path.read_bytes()
    need(len(data) <= 1024**2 and hashlib.sha256(data).hexdigest() == digest, "Selected source bytes")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(data, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


def prepare(model, seed_path, seed_sha):
    parent = load(PARENT, PARENT_SHA, "_d3_os_parent_preparer")
    catalogue, p, base = parent.catalogue()
    need(model in MODELS, "Exact six-model population")
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    total = next(int(x.split()[1]) * 1024 for x in Path("/proc/meminfo").read_text().splitlines()
                 if x.startswith("MemTotal:"))
    need(boot == base["boot_id"] and total == base["mem_total_bytes"], "Fresh selected host")
    if model == "d3":
        # Original conservative capacity refusal precedes seed/constructor/plan access.
        return parent.capacity_refusal(catalogue, p)
    need(p.ordinary(seed_path) == SEED, "Exact new helper04 seed; old03 bindings are dead")
    seed = json.loads(p.checked(seed_path, seed_sha, limit=1024**2))
    execution = load(HERE / "execution.py", EXECUTION_SHA, "_d3_os_selected_execution")
    selected = copy.deepcopy(base)
    selected["model"] = model
    selected["registry_identity"] = copy.deepcopy(catalogue["all_six_identities"][model])
    short = model.removeprefix("d3-")
    folder = HERE / (model + "-01")
    output = HERE / "campaign-01" / model
    model_outputs = {g: ROOT / ("artifacts/jev-d3-native/20261022-os-" + short + "-" + g + "-01")
                     for g in execution.GROUPS}
    units = {g: "jev-d3-" + short + "-" + g + "-os-20261022-01.service"
             for g in execution.GROUPS}
    uid = os.getuid()
    selected["cgroups"] = {g: f"/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/{u}"
                           for g, u in units.items()}
    frozen = ROOT / "artifacts/jev-runtime/decision3-q95-frozen-root-20261022-01"
    selected["helper_seed"] = {"path": str(SEED), "sha256": seed_sha,
        "cgroup": f"/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/jev-d3-q95-helper-20261022-04.service",
        "source_root": str(frozen), "service_directory": str(frozen / "serve-04")}
    derived = parent.descriptors(p, selected, seed)
    checker = p.load(selected["components"]["helper_checker"], "_d3_os_static_helper")
    with tempfile.TemporaryDirectory(prefix=".static-helper-", dir=HERE) as temporary:
        for group, descriptor in derived.items():
            receipts = Path(temporary) / group
            receipts.mkdir()
            data = execution.encode(descriptor)
            instance = checker.LiveHelperChecker(selection_blob=data,
                selection_sha=hashlib.sha256(data).hexdigest(), receipt_directory=receipts)
            need(instance.model == model and instance.resident_bytes ==
                 selected["registry_identity"]["resident_file_bytes"], "Exact selected native model")
    need(not folder.exists() and not output.exists() and
         all(not p.ordinary(x).exists() for x in model_outputs.values()), "Fresh metadata and outputs")
    folder.mkdir(mode=0o700)
    descriptors = {g: folder / ("helper-" + g + "-01.json") for g in derived}
    for group, descriptor in derived.items():
        p.exclusive_json(descriptors[group], descriptor)
    plan = execution.prepare_plan(model=model,
        helper_checker=ROOT / selected["components"]["helper_checker"]["path"],
        helper_selections=descriptors, helper_bindings={g: d["binding"] for g, d in derived.items()},
        browser=selected["browser"], output=output, model_outputs=model_outputs,
        boot_id=boot, mem_total_bytes=total, gpu_uuid=selected["gpu_uuid"],
        expires_monotonic=time.monotonic() + 9000, units=units,
        nonces={g: "jev-d3-boundary-" + secrets.token_hex(16) for g in derived})
    old_refs = {row["path"]: row for row in selected["expected_base_source_refs"]}
    for key in ("execution", "entry", "observer"):
        old_refs.pop(selected["components"][key]["path"])
        replacement = plan["components"][key]
        old_refs[replacement["path"]] = replacement
    authority = plan["components"]["observer_manifest"]
    old_refs[authority["path"]] = authority
    expected = sorted([*old_refs.values(), *(p.reference(x) for x in descriptors.values())],
                      key=lambda x: x["path"])
    need(plan["sources"] == expected and len(expected) == 40,
         "Original37 component bridge plus authority and two fresh descriptors")
    unchanged = set(selected["components"]) - {"execution", "entry", "observer"}
    need(all(plan["components"][k] == selected["components"][k] for k in unchanged),
         "Native bootstrap/helper/transport/resource components remain unchanged")
    plan_path = folder / "plan-01.json"
    data = p.exclusive_json(plan_path, plan)
    digest = hashlib.sha256(data).hexdigest()
    execution.selected_plan(plan_path, digest)
    result = {"schema_id": "decision3-system-services-prepared-launch-v1", "model": model,
        "plan": p.reference(plan_path), "selected_helper_seed": p.reference(SEED),
        "preparer": p.reference(Path(__file__)), "source_count": 40,
        "fixed_denominator": 10, "model_qualified": False, "score": None,
        "actual_live_admission_performed": False, "actual_model_or_browser_started": False,
        "launch_argv_for_external_owner": [plan["interpreter"], "-B", "-s",
            str(HERE / "launch_execution.py"), "--execute", "--role", "controller",
            "--plan", str(plan_path), "--plan-sha256", digest]}
    p.exclusive_json(folder / "prepared-launch-01.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--helper-seed")
    parser.add_argument("--helper-seed-sha256")
    args = parser.parse_args(argv)
    if not args.prepare:
        print(json.dumps({"mode": "DRY_METADATA_ONLY", "model": args.model,
                          "actual_started": False, "score": None}))
        return 0
    if args.model != "d3" and (not args.helper_seed or not args.helper_seed_sha256):
        parser.error("Preparation requires exact fresh helper04 path and independent SHA")
    print(json.dumps(prepare(args.model, args.helper_seed, args.helper_seed_sha256), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
