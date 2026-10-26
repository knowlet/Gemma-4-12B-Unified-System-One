"""Explicit private composition of the selected native40 browser owner."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[2]
PARENT = ROOT / "artifacts/jev-runtime/native-v2-browser-execution-20261022-04/launch_browser.py"
PARENT_SHA = "44eadeb19d907dc6084e31e5e86aa81bb73f514365ee3722c0baaf274a5fd842"
PROFILE = "jev-current-nf4-v7-q95-find-browser-owned-v1"
DRIVER = (HERE / "control_driver.py").relative_to(ROOT).as_posix()
OBSERVER = "artifacts/jev-runtime/observer-system-services-20261022-01/comfy_observer.py"
OBSERVER_AUTHORITY = (
    "artifacts/jev-runtime/observer-system-services-20261022-01/system-services.json"
)
REUSED_SOURCES = {
    "artifacts/jev-runtime/native-v2-q95-browser-control-20261022-01/control_driver.py": "00295d7110522dd346320e598298f923bda56fa4a73dba427035371801c73543",
    "artifacts/jev-runtime/native-v2-q95-browser-control-20261022-01/policy_boundary.py": "4ed323e24cff140caef36c6557fe3286154e1c11b92f39536615eab6f1d3a8ef",
    "artifacts/jev-runtime/native-v2-q95-browser-control-20261022-01/browser_entry.py": "ea0f70680691e1467523fb3a0a8fe34f661d8bb3e2076d00dcecd8a0488aa28d",
}
Q95_SHA = "d33bfcac49f90c47b626c3915baf3d5fcbfb2e32522bfaccbbfec59dce35d8db"
Q95_MANIFESTS = {
    "qualification": Q95_SHA,
    "execution": "95f2eab0be9646da18694b18c872e044db09ba2fb1c09708077b01b24d58ae44",
    "runtime": "702f7ff3cf194d1db56f86474e48ffc88cdfa8c9c6d3930467e3c12ea6492016",
}
HELPER_PARENT = "artifacts/jev-runtime/decision3-local-helper-budget-20261022-01/live_helper.py"


def load_parent():
    blob = PARENT.read_bytes()
    if hashlib.sha256(blob).hexdigest() != PARENT_SHA:
        raise ValueError("Selected original native owner source differs")
    # This private source copy changes only the declared unit date after SHA proof.
    old_unit = b"f\"jev-current-nf4-{plan['arm']}-20261022-{plan['attempt']}.service\""
    new_unit = b"f\"jev-current-nf4-{plan['arm']}-20261023-{plan['attempt']}.service\""
    if blob.count(old_unit) != 1 or new_unit in blob:
        raise ValueError("Exact single original unit-date declaration required")
    original = blob
    blob = blob.replace(old_unit, new_unit)
    if blob.replace(new_unit, old_unit) != original:
        raise ValueError("Unit-date declaration inverse differs")
    spec = importlib.util.spec_from_file_location("_native_q95_original_owner", PARENT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(blob, str(PARENT), "exec", dont_inherit=True), module.__dict__)
    return module


owner = load_parent()
owner.HERE, owner.PROFILE, owner.DRIVER = HERE, PROFILE, DRIVER
owner.FORMATS = {"v2-q95-find": "shared-elements-v2"}
owner.OBSERVER_SOURCE = OBSERVER
original_validate, original_closed = owner.validate_plan, owner.verify_closed


def descriptor(plan):
    ref = plan["helper_selection"]
    owner.require(ref in plan["sources"], "Gemma helper descriptor must be source-pinned")
    blob = owner.ordinary(ROOT / ref["path"]).read_bytes()
    owner.require(
        len(blob) <= 64 * 1024 and owner.pin(ROOT / ref["path"]) == ref,
        "Exact small Gemma helper descriptor",
    )
    value = json.loads(blob)
    owner.require(
        set(value) == {"schema_id", "helper_seed_ref", "native_co_resident"}
        and value["schema_id"] == "gemma-native-nf4-v2-q95-helper-selection-v1",
        "Explicit Gemma helper schema",
    )
    seed_ref, co = value["helper_seed_ref"], value["native_co_resident"]
    owner.require(
        seed_ref in plan["sources"] and owner.pin(ROOT / seed_ref["path"]) == seed_ref,
        "Exact source-pinned fresh Q95 seed",
    )
    seed = json.loads(owner.ordinary(ROOT / seed_ref["path"]).read_bytes())
    binding = seed["binding"]
    owner.require(
        seed["schema_id"] == "decision3-q95-live-helper-selection-v1"
        and binding["qualification_sha256"] == Q95_SHA
        and binding["profile"] == "field-syntax-preserving-base-helper-v1"
        and (binding["source_count"], binding["seconds"], binding["tokens"], binding["context"])
        == (95, 20, 128, 8192)
        and binding["boot_id"] == plan["boot_id"],
        "Same selected original Q95 helper",
    )
    owner.require(
        all(seed[key]["sha256"] == digest for key, digest in Q95_MANIFESTS.items()),
        "Original Q95 exact qualification/source/runtime manifests",
    )
    expected = {
        "profile": "gemma-native-nf4-v2-with-q95-v1",
        "unit": plan["unit"],
        "cgroup": plan["cgroup"],
        "memory_max_bytes": owner.CAP,
        "adapter_manifest": next(
            r for r in plan["sources"] if r["path"] == plan["adapter"] + "/manifest.json"
        ),
    }
    owner.require(co == expected, "Gemma co-resident binding must match actual selected owner")
    return value, seed


def helper_static(seed):
    """Verify the original selected Q95 populations without importing runtime."""
    for name, expected_count in (("execution", 95), ("runtime", 348)):
        ref = seed[name]
        owner.authenticate(ref)
        files = json.loads((ROOT / ref["path"]).read_bytes())["files"]
        owner.require(len(files) == expected_count, "Original Q95 population count")
        for label, expected in files.items():
            owner.require(
                (not Path(label).is_absolute() and ".." not in Path(label).parts)
                if name == "execution"
                else Path(label).is_absolute(),
                "Original Q95 manifest path convention",
            )
            path = Path(seed["source_root"]) / label if name == "execution" else Path(label)
            path = owner.ordinary(path)
            digest, size = hashlib.sha256(), 0
            with path.open("rb") as stream:
                while block := stream.read(1024**2):
                    digest.update(block)
                    size += len(block)
            owner.require(
                {"sha256": digest.hexdigest(), "size_bytes": size} == expected,
                "Original Q95 selected source/runtime bytes",
            )


def validate_plan(plan):
    original_validate(plan)
    labels = {ref["path"] for ref in plan["sources"]}
    required = {
        str(PARENT.relative_to(ROOT)),
        "scripts/jev_native_v2_browser.py",
        HELPER_PARENT,
        *(
            (HERE / name).relative_to(ROOT).as_posix()
            for name in ("gemma_helper.py", "policy_boundary.py", "checked_source.py")
        ),
    }
    owner.require(
        required | set(REUSED_SOURCES) | {OBSERVER_AUTHORITY} <= labels,
        "Complete explicit source composition pins",
    )
    owner.require(
        all(
            any(row["path"] == path and row["sha256"] == sha for row in plan["sources"])
            for path, sha in REUSED_SOURCES.items()
        ),
        "Exact selected reused composition bytes",
    )
    _, seed = descriptor(plan)
    owner.require(
        all(seed[key] in plan["sources"] for key in ("qualification", "execution", "runtime")),
        "Q95 source/runtime manifests must be selected",
    )
    helper_static(seed)
    selected_find(plan).authenticate_plan(plan)
    return plan


def verify_closed(result, receipt, plan, output):
    original_closed(result, receipt, plan, output)
    value, seed = descriptor(plan)
    helper_static(seed)
    directory, _ = owner.run_paths(plan)
    control = json.loads((directory / "q95-control/receipt.json").read_bytes())
    owner.require(
        control["profile"] == "jev-current-nf4-v7-q95-find-browser-v1"
        and control["stage"] == "complete_pending_external_closure"
        and control["primary_error"] is None
        and control["selection_sha256"] == plan["helper_selection"]["sha256"]
        and control["helper_binding"] == seed["binding"]
        # Native pin has sha256,size_bytes; the owner also reports path.
        and control["native_receipt"]
        == {k: v for k, v in owner.pin(output / "receipt.json").items() if k != "path"},
        "Complete source-bound same-helper control receipt",
    )
    owner.require(
        len(control["group_indexes"]) == 2
        and control["native_journal_covers_external_helper"] is False
        and control["external_owned_closure_required"] is True,
        "Both external Q95 group indexes required",
    )
    for row in control["group_indexes"]:
        path = owner.ordinary(row["group_directory"]) / "external-q95-request-index.json"
        owner.require(
            path.is_relative_to(output) and path.is_file(), "Actual external Q95 group index"
        )

    selected_find(plan).verify_outputs(plan, output)


def selected_find(plan):
    """Authenticate the actual local driver before source/receipt verification."""
    from checked_source import module

    path = (HERE / "control_driver.py").relative_to(ROOT).as_posix()
    refs = [row for row in plan["sources"] if row["path"] == path]
    owner.require(len(refs) == 1, "Exact actual FIND driver selection")
    owner.authenticate(refs[0])
    return module(path, refs[0]["sha256"], "_find04_owner_driver")


owner.validate_plan, owner.verify_closed = validate_plan, verify_closed


def __getattr__(name):
    return getattr(owner, name)


def main(argv=None):
    return owner.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
