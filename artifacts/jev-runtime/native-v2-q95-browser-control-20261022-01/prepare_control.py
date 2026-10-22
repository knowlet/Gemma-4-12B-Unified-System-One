"""Metadata-only plan sealing after root selects a fresh live Q95 owner seed."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import secrets
from pathlib import Path

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[2]
BASE_PLAN = (
    ROOT / "artifacts/jev-runtime/native-v2-browser-execution-20261022-04/v2-01/selected-plan.json"
)
BASE_SHA = "73bff770e242b93d5b7b53d24ef933926bd63897a4707d6d2e2a0f06f4438b6d"
OBSERVER = "artifacts/jev-runtime/comfy-observer-zombie-exit-20261022-01/comfy_observer.py"
PARENT = "artifacts/jev-runtime/decision3-local-helper-budget-20261022-01/live_helper.py"
SOURCE_NAMES = (
    "launch_browser.py",
    "browser_entry.py",
    "control_driver.py",
    "gemma_helper.py",
    "policy_boundary.py",
    "prepare_control.py",
)


def need(value, reason):
    if not value:
        raise ValueError(reason)


def selected_blob(path, sha=None):
    path = Path(path).absolute()
    need(
        path.is_relative_to(ROOT)
        and not any(p.is_symlink() for p in (path, *path.parents))
        and path.stat().st_size <= 4 * 1024**2,
        "Small ordinary local selected metadata/source only",
    )
    blob = path.read_bytes()
    need(
        sha is None or hashlib.sha256(blob).hexdigest() == sha,
        "Independently selected metadata SHA",
    )
    return blob


def ref(path):
    path = Path(path).absolute()
    blob = selected_blob(path)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "size_bytes": len(blob),
    }


def prepare(seed_path, seed_sha):
    old = json.loads(selected_blob(BASE_PLAN, BASE_SHA))
    seed_blob = selected_blob(seed_path, seed_sha)
    seed = json.loads(seed_blob)
    need(
        set(seed)
        == set(
            "schema_id binding worker listener service_directory private_config interpreter source_root qualification execution runtime registry expected_memtotal".split()
        ),
        "Exact source-only owner seed13",
    )
    binding = seed["binding"]
    need(
        seed["schema_id"] == "decision3-q95-live-helper-selection-v1"
        and binding["qualification_sha256"]
        == "d33bfcac49f90c47b626c3915baf3d5fcbfb2e32522bfaccbbfec59dce35d8db"
        and binding["profile"] == "field-syntax-preserving-base-helper-v1"
        and (binding["source_count"], binding["seconds"], binding["tokens"], binding["context"])
        == (95, 20, 128, 8192),
        "Exact original Q95 profile",
    )
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    need(
        binding["boot_id"] == boot == old["boot_id"]
        and seed["expected_memtotal"] == old["expected_memtotal"],
        "Same selected boot/physical host",
    )
    output, descriptor_path = HERE / "launch-plan-01.json", HERE / "helper-selection-01.json"
    need(
        not output.exists() and not descriptor_path.exists() and not (HERE / "v2-q95-01").exists(),
        "Fresh plan, descriptor and actor outputs",
    )
    plan = copy.deepcopy(old)
    unit = "jev-current-nf4-v2-q95-20261022-01.service"
    cgroup = f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{unit}"
    adapter = next(r for r in old["sources"] if r["path"] == old["adapter"] + "/manifest.json")
    descriptor = {
        "schema_id": "gemma-native-nf4-v2-q95-helper-selection-v1",
        "helper_seed_ref": ref(seed_path),
        "native_co_resident": {
            "profile": "gemma-native-nf4-v2-with-q95-v1",
            "unit": unit,
            "cgroup": cgroup,
            "memory_max_bytes": 40 * 1024**3,
            "adapter_manifest": adapter,
        },
    }
    descriptor_bytes = (json.dumps(descriptor, sort_keys=True, indent=2) + "\n").encode()
    descriptor_ref = {
        "path": descriptor_path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(descriptor_bytes).hexdigest(),
        "size_bytes": len(descriptor_bytes),
    }
    refs = {row["path"]: row for row in old["sources"]}
    new = [
        *(ref(HERE / name) for name in SOURCE_NAMES),
        ref(ROOT / OBSERVER),
        ref(ROOT / PARENT),
        ref(seed_path),
        descriptor_ref,
        *(seed[k] for k in ("qualification", "execution", "runtime")),
    ]
    execution = json.loads(
        selected_blob(ROOT / seed["execution"]["path"], seed["execution"]["sha256"])
    )["files"]
    need(len(execution) == 95, "Exact selected frozen Q95 source95")
    for label, expected in execution.items():
        path = Path(seed["source_root"]) / label
        need(
            not Path(label).is_absolute()
            and ".." not in Path(label).parts
            and path.absolute().is_relative_to(ROOT),
            "Confined saved Q95 source",
        )
        new.append({"path": path.relative_to(ROOT).as_posix(), **expected})
    for row in new:
        need(
            row["path"] not in refs or refs[row["path"]] == row,
            "No contradictory original source identity",
        )
        refs[row["path"]] = row
    plan.update(
        schema_id="jev-current-nf4-v7-q95-browser-owned-v1",
        arm="v2-q95",
        attempt="01",
        unit=unit,
        cgroup=cgroup,
        nonce="jev-nf4-browser-" + secrets.token_hex(16),
        run_directory=(HERE / "v2-q95-01").relative_to(ROOT).as_posix(),
        output=(HERE / "v2-q95-01/browser").relative_to(ROOT).as_posix(),
        sources=[refs[k] for k in sorted(refs)],
        comfy_observer=ref(ROOT / OBSERVER),
        helper_selection=descriptor_ref,
    )
    plan.update(
        source_successor_of={"path": BASE_PLAN.relative_to(ROOT).as_posix(), "sha256": BASE_SHA},
        matched_control="Same freshly selected Q95 text helper as Decision3; original NF4-helper4/10 is a separate helper condition",
        comparison_contract={
            "policy": "Original Gemma NF4 with N3 full142 V2 adapter",
            "text_helper": binding,
            "tasks": "Unchanged original V7 known4+public6, max16steps, seed20261014, original judges",
            "native_resources": "Original hostR+2BASE/actual-forward rules, native40GiB/swap0; no D3 CORE substitution",
            "policy_seconds": 20,
            "helper_check_seconds": 3,
            "internal_nf4_helper": "Preparation retained unused; local chat route refuses",
        },
        preparation_status="METADATA_ONLY_NOT_ACTUAL_ADMISSION",
        preparation_checks_not_run=[
            "Full23.9GB tensor hash and complete ordinary source validation",
            "Fresh live Q95 identity/authentication/admission",
            "Unit/nonce/Invocation/cgroup/kernel readback",
            "Actual model/browser execution and independent whole-unit closure",
        ],
        source_population_counts={
            "retained_original_sources": len(old["sources"]),
            "selected_ordinary_sources": len(refs),
            "additional_installed_runtime_population": 348,
            "installed_runtime_binding": "Selected original Q95 runtime manifest; independently read-only checked by owner admission/closure and every live helper callback",
        },
    )
    for path, blob in (
        (descriptor_path, descriptor_bytes),
        (output, (json.dumps(plan, sort_keys=True, indent=2) + "\n").encode()),
    ):
        with path.open("xb") as stream:
            stream.write(blob)
    return {
        "plan": ref(output),
        "descriptor": descriptor_ref,
        "launch_ready_claimed": False,
        "mode": "METADATA_ONLY",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--helper-seed", type=Path)
    parser.add_argument("--helper-seed-sha256")
    args = parser.parse_args(argv)
    if not args.prepare:
        print("DRY_NO_METADATA_OR_ACTOR_WRITES")
        return 0
    need(args.helper_seed and args.helper_seed_sha256, "Explicit fresh selected helper seed")
    print(json.dumps(prepare(args.helper_seed, args.helper_seed_sha256), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
