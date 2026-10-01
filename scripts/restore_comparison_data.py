"""Restore byte-identical historical media from the retained Modal M1 artifact.

Run prepare_live_validation_data.py first for hash-verified public source files
and text splits. This command downloads a stored artifact; it starts no GPU or
Modal function. An existing local M1 file can be supplied with --source.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
import struct
import tempfile
from pathlib import Path

from s1.evaluation.datasets import fingerprint, load_cases, request_fingerprint

ROOT = Path(__file__).resolve().parents[1]
VOLUME = "gemma-unified-system-one"
ARCHIVED_MEDIA = "/runs/20261001-v2-live-03/services/checks/pipeline/media-views/M1.jsonl"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def restore(dataset_dir, summary_file, *, source=None, backup_dir=None):
    dataset_dir = Path(dataset_dir)
    summary = json.loads(Path(summary_file).read_text())
    historical = summary["data"]
    required = {
        **{name: value["file_sha256"] for name, value in historical["datasets"].items()},
        "media-bundle.jsonl": historical["media_bundle_sha256"],
        "provenance.json": summary["dataset_provenance_sha256"],
    }
    for name in historical["datasets"]:
        if name != "media-test.jsonl" and sha((dataset_dir / name).read_bytes()) != required[name]:
            raise ValueError(f"text split differs from historical pin: {name}")
    if all(sha((dataset_dir / name).read_bytes()) == digest for name, digest in required.items()):
        return {"status": "already_matches_historical", "files_sha256": required}
    if source is None:
        import modal

        volume = modal.Volume.from_name(VOLUME, create_if_missing=False)
        archived = b"".join(volume.read_file(ARCHIVED_MEDIA))
        source_identity = f"modal-volume://{VOLUME}{ARCHIVED_MEDIA}"
    else:
        archived = Path(source).read_bytes()
        source_identity = str(Path(source).resolve())
    with tempfile.TemporaryDirectory(prefix="s1-media-restore-") as directory:
        stage = Path(directory)
        (stage / "M1.jsonl").write_bytes(archived)
        cases = load_cases(stage / "M1.jsonl")
        expected = historical["datasets"]["media-test.jsonl"]
        if fingerprint(cases) != expected["dataset_sha256"]:
            raise ValueError("archived media fingerprint does not match historical pin")
        current = load_cases(dataset_dir / "media-test.jsonl")
        if [(c.id, c.gold) for c in current] != [(c.id, c.gold) for c in cases]:
            raise ValueError("current media selection differs from archived IDs or labels")
        (stage / "media-test.jsonl").write_bytes(
            b"".join(case.model_dump_json().encode() + b"\n" for case in cases)
        )
        rows = [
            json.loads(line)
            for line in (dataset_dir / "media-bundle.jsonl").read_text().splitlines()
        ]
        if [row["case"]["id"] for row in rows] != [case.id for case in cases]:
            raise ValueError("media bundle selection differs from archived case order")
        for row, case in zip(rows, cases):
            row["case"] = case.model_dump(mode="json")
        (stage / "media-bundle.jsonl").write_bytes(
            b"".join(
                json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
                + b"\n"
                for row in rows
            )
        )
        provenance = json.loads((dataset_dir / "provenance.json").read_text())
        provenance["datasets"]["media-test.jsonl"] = expected
        provenance["media_bundle_sha256"] = historical["media_bundle_sha256"]
        by_id = {case.id: case for case in cases}
        for row in provenance["selected_source_records"]:
            if row["case_id"] not in by_id:
                continue
            case = by_id[row["case_id"]]
            row["request_sha256"] = request_fingerprint(case.request)
            media = case.request.media[0]
            if media.type == "image":
                row["converted_image_sha256"] = sha(base64.b64decode(media.data))
            else:
                row["resampled_float32_sha256"] = sha(
                    struct.pack("<" + "f" * len(media.samples), *media.samples)
                )
        (stage / "provenance.json").write_text(
            json.dumps(provenance, indent=2, ensure_ascii=False) + "\n"
        )
        names = ("media-test.jsonl", "media-bundle.jsonl", "provenance.json")
        for name in names:
            if sha((stage / name).read_bytes()) != required[name]:
                raise ValueError(f"restored artifact differs from historical pin: {name}")
        # Check all destination and backup identities before changing any file.
        backup_dir = Path(backup_dir or dataset_dir.with_name("datasets-before-restore"))
        for name in names:
            backup = backup_dir / name
            if backup.exists() and backup.read_bytes() != (dataset_dir / name).read_bytes():
                raise FileExistsError(f"backup already contains different data: {backup}")
        backup_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            shutil.copy2(dataset_dir / name, backup_dir / name)
        for name in names:
            shutil.copy2(stage / name, dataset_dir / name)
    return {
        "status": "restored_historical_inputs",
        "source": source_identity,
        "cases": len(cases),
        "dataset_sha256": expected["dataset_sha256"],
        "files_sha256": required,
        "backup_directory": str(backup_dir),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path, default=ROOT / "artifacts/live-validation/datasets"
    )
    parser.add_argument(
        "--summary", type=Path, default=ROOT / "docs/validation/2026-10-01/live-summary.json"
    )
    parser.add_argument("--source", type=Path, help="Previously downloaded original M1 JSONL")
    parser.add_argument("--backup-dir", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            restore(args.dataset_dir, args.summary, source=args.source, backup_dir=args.backup_dir),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
