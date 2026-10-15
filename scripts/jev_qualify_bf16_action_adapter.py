#!/usr/bin/env python3
"""Issue a bounded BF16 diagnostic qualification from finalized, audited files.

This command copies ordinary local evidence only. It never imports Torch,
executes audit code, loads a model/base weights, starts a browser, or uses a GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from s1 import browser_action_adapter as artifact
from s1 import browser_action_qualification as qualification


def issue(adapter_path, campaign, audit_path, audit_sha256, audit_source, output):
    forbidden = {"torch", "transformers", "peft", "safetensors", "playwright"}
    qualification.require(
        not forbidden.intersection(name.split(".")[0] for name in sys.modules),
        "issuer must not import model or browser dependencies",
    )
    fit = artifact._ordinary_directory(Path(adapter_path).parent)
    campaign = artifact._ordinary_directory(campaign)
    output = Path(output).absolute()
    artifact._ordinary_directory(output.parent)
    qualification.require(
        not output.exists()
        and not any(
            output.is_relative_to(path) or path.is_relative_to(output) for path in (fit, campaign)
        ),
        "qualification output must be new and outside its evidence inputs",
    )
    audit_path, audit_source = Path(audit_path), Path(audit_source)
    qualification.require(
        qualification.file_record(audit_path)["sha256"] == audit_sha256,
        "reviewed audit SHA256 differs",
    )
    audit = artifact._json(audit_path.read_bytes())
    qualification.require(
        qualification.file_record(audit_source)["sha256"] == audit["script_sha256"],
        "audit source identity differs",
    )
    fit_files, campaign_files = qualification.inventory(fit), qualification.inventory(campaign)
    manifest = artifact._json((fit / "adapter/manifest.json").read_bytes())
    receipt = artifact._json((fit / "receipt.json").read_bytes())
    pins = {
        "manifest_sha256": qualification.file_record(fit / "adapter/manifest.json")["sha256"],
        "training_receipt_sha256": qualification.file_record(fit / "receipt.json")["sha256"],
        "weights_sha256": manifest["files"]["adapter_model.safetensors"]["sha256"],
        "inventory": fit_files,
    }
    facts = qualification.campaign_facts(campaign, audit, pins)
    qualification.require(
        Path(adapter_path).absolute() == fit / "adapter"
        and manifest["deployment_quantization"] == "nf4"
        and manifest["training_receipt"]["sha256"]
        == qualification.file_record(fit / "receipt.json")["sha256"]
        and receipt["stage"] == "complete",
        "qualification must reference the unchanged sealed fit",
    )
    sources, packages = qualification.source_inventory(), qualification.versions()
    # Read only bounded ordinary audit inputs, using labels as data, never code.
    repository = Path(__file__).resolve().parents[1]
    audit_inputs = {}
    for label, expected in audit["read_inputs"].items():
        path = Path(label)
        qualification.require(
            not path.is_absolute()
            and ".." not in path.parts
            and "\\" not in label
            and path.name != "model.safetensors",
            "unsafe or base-weight audit input",
        )
        # HF snapshot metadata may point into the immutable blob cache. Only
        # authenticated source bytes may follow that link; the bundle is ordinary.
        source = (repository / path).resolve(strict=True)
        qualification.require(
            qualification.file_record(source) == expected, "audited input changed: " + label
        )
        audit_inputs[label] = (source, expected)
    output.mkdir(exist_ok=False)

    def copy_files(root, files, destination):
        for label, expected in files.items():
            source, target = root / label, destination / label
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            qualification.require(
                qualification.file_record(target) == expected, "copied qualification input differs"
            )

    copy_files(fit, fit_files, output / "fit")
    copy_files(campaign, campaign_files, output / "evidence/campaign")
    for name, source in (
        ("audit.json", audit_path),
        ("audit.py", audit_source),
        ("qualification-issuer.py", Path(__file__)),
    ):
        shutil.copyfile(source, output / "evidence" / name)
    for label, (source, expected) in audit_inputs.items():
        target = output / "evidence/audit-inputs" / label
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        qualification.require(
            qualification.file_record(target) == expected, "copied audited input differs"
        )
    qualification.verify_audit_inputs(output / "evidence", audit)
    qualification.require(
        qualification.inventory(fit) == fit_files
        and qualification.inventory(campaign) == campaign_files
        and qualification.source_inventory() == sources
        and qualification.versions() == packages,
        "qualification inputs changed during preservation",
    )
    qualification.require(
        qualification.campaign_facts(output / "evidence/campaign", audit, pins) == facts,
        "copied diagnostic evidence differs",
    )
    declaration = {
        "schema_version": 1,
        "kind": "jev_action_bf16_qualification_v1",
        "scope": qualification.SCOPE,
        "runtime": qualification.runtime_profile(),
        "runtime_sources": sources,
        "versions": packages,
        "artifact": {
            "manifest_sha256": qualification.file_record(fit / "adapter/manifest.json")["sha256"],
            "training_receipt_sha256": qualification.file_record(fit / "receipt.json")["sha256"],
            "weights_sha256": manifest["files"]["adapter_model.safetensors"]["sha256"],
            "inventory": fit_files,
        },
        "evidence": {
            "audit_sha256": audit_sha256,
            "inventory": qualification.inventory(output / "evidence"),
        },
        "observed_diagnostics": facts,
    }
    path = output / "qualification.json"
    with path.open("xb") as stream:
        stream.write(
            json.dumps(declaration, ensure_ascii=False, indent=2, allow_nan=False).encode() + b"\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    qualification.require(
        not forbidden.intersection(name.split(".")[0] for name in sys.modules),
        "issuer imported model or browser dependencies",
    )
    return {
        "qualification": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "observed_diagnostics": facts,
        "base_weight_bytes_read": 0,
        "models_loaded": 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--audit-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            issue(
                args.adapter,
                args.campaign,
                args.audit,
                args.audit_sha256,
                args.audit_source,
                args.output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
