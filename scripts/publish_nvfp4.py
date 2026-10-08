#!/usr/bin/env python3
"""Publish an evaluated, packed NVFP4 checkpoint with immutable Hub verification.

No directory glob is uploaded. Supply every intended repository path with
--allow-file, or supply a JSON list of exact paths with --file-list. Validation
runs before authentication or any Hub mutation; --validate-only never contacts
the Hub. Publication receipts are written outside the uploaded package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

DEFAULT_REPOSITORY = "knowlet/Gemma-4-12B-Unified-System-One-NVFP4"
REQUIRED_FILES = frozenset(
    {
        "README.md",
        "LICENSE",
        "NOTICE",
        "config.json",
        "s1_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "conversion-manifest.json",
        "nvfp4_manifest.json",
        "evaluation.json",
        "runtime-verification.json",
    }
)
REQUIRED_POPULATIONS = {"test": 256, "media": 52, "regression_text": 128}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def exact_path(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("upload paths must be nonempty relative POSIX paths")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or any(character in value for character in "*?[]\x00\n\r")
    ):
        raise ValueError(f"upload path must be exact and remain inside the package: {value!r}")
    return value


def local_file(root, relative):
    relative = exact_path(relative)
    path = root / relative
    chain = [root / parent for parent in PurePosixPath(relative).parents]
    if any(part.is_symlink() for part in (path, *chain)):
        raise ValueError(f"package paths must not contain symbolic links: {relative}")
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"package file is missing or outside the package: {relative}")
    return path


def inventory_entries(value):
    if isinstance(value, dict):
        value = [{"path": name, **entry} for name, entry in value.items()]
    if not isinstance(value, list) or not value:
        raise ValueError("conversion manifest must contain a nonempty file inventory")
    result = {}
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueError("file inventory entries must be objects")
        name = exact_path(entry.get("path"))
        if name in result:
            raise ValueError(f"duplicate inventory entry: {name}")
        if (
            not isinstance(entry.get("sha256"), str)
            or not re.fullmatch(r"[a-f0-9]{64}", entry["sha256"])
            or type(entry.get("size_bytes")) is not int
            or entry["size_bytes"] <= 0
        ):
            raise ValueError(f"invalid file identity: {name}")
        result[name] = {"path": name, "sha256": entry["sha256"], "size_bytes": entry["size_bytes"]}
    return result


def validate_evaluation(evaluation, conversion_sha256):
    if (
        evaluation.get("status") != "ok"
        or evaluation.get("release_validation_complete") is not True
        or evaluation.get("population_complete") is not True
    ):
        raise ValueError("NVFP4 release evaluation must have completed successfully")
    if evaluation.get("conversion_manifest_sha256") != conversion_sha256:
        raise ValueError("evaluation is not bound to this conversion manifest")
    validate_populations(evaluation.get("reports", {}), "evaluation")


def validate_populations(reports, label):
    for name, expected in REQUIRED_POPULATIONS.items():
        report = reports.get(name, {})
        counts = report.get("counts", {})
        if (
            report.get("coverage") != 1.0
            or type(counts.get("ok")) is not int
            or counts["ok"] != expected
            or any(count != 0 for status, count in counts.items() if status != "ok")
            or report.get("metrics", {}).get("n") != expected
            or report.get("warmup_errors")
        ):
            raise ValueError(f"{label} must include all {expected} successful {name} cases")


def validate_runtime_verification(verification, observed):
    if verification.get("status") != "ok" or verification.get("population_complete") is not True:
        raise ValueError("published runtime verification must have completed successfully")
    for field, filename in (
        ("conversion_manifest_sha256", "conversion-manifest.json"),
        ("evaluation_sha256", "evaluation.json"),
    ):
        if verification.get(field) != observed[filename]["sha256"]:
            raise ValueError(f"runtime verification is not bound to this {filename}")
    wheel = verification.get("runtime_wheel", {})
    filename = exact_path(wheel.get("filename"))
    if (
        not filename.endswith(".whl")
        or filename not in observed
        or wheel.get("sha256") != observed[filename]["sha256"]
    ):
        raise ValueError("verified runtime wheel must be included with its exact SHA256")
    validate_populations(verification.get("reports", {}), "runtime verification")
    return {"filename": filename, **{k: observed[filename][k] for k in ("sha256", "size_bytes")}}


def validate_package(model, allow_files):
    model = Path(model).absolute()
    names = [exact_path(name) for name in allow_files]
    if len(names) != len(set(names)):
        raise ValueError("the upload whitelist contains duplicate paths")
    allowed = set(names)
    if missing := REQUIRED_FILES - allowed:
        raise ValueError(f"upload whitelist lacks required files: {sorted(missing)}")
    paths = {name: local_file(model, name) for name in names}
    conversion = json.loads(paths["conversion-manifest.json"].read_text())
    if (
        conversion.get("format") != "nvfp4"
        or conversion.get("status") != "ok"
        or conversion.get("structural_verification", {}).get("status") != "ok"
    ):
        raise ValueError("conversion manifest must verify an actual NVFP4 export")
    converted = inventory_entries(conversion.get("files"))
    if missing := converted.keys() - allowed:
        raise ValueError(f"upload whitelist omits converted checkpoint files: {sorted(missing)}")
    bound_sidecars = {"config.json", "s1_config.json", "nvfp4_manifest.json"}
    if not bound_sidecars <= converted.keys() or not any(
        name.endswith(".safetensors") for name in converted
    ):
        raise ValueError(
            "conversion inventory must bind config, S1 calibration and packed NVFP4 files"
        )
    packed_manifest = json.loads(paths["nvfp4_manifest.json"].read_text())
    tensor_files = {
        exact_path(item["filename"]) for item in packed_manifest.get("tensors", {}).values()
    }
    if not tensor_files or not tensor_files <= converted.keys():
        raise ValueError("conversion inventory must bind every packed tensor shard")
    if {name for name in allowed if name.endswith(".safetensors")} - converted.keys():
        raise ValueError("uploaded tensor shards must be bound to the conversion inventory")
    files = [
        {"path": name, "size_bytes": paths[name].stat().st_size, "sha256": file_sha256(paths[name])}
        for name in sorted(names)
    ]
    observed = {entry["path"]: entry for entry in files}
    for name, expected in converted.items():
        if observed[name] != expected:
            raise ValueError(f"converted checkpoint file identity changed: {name}")
    evaluation = json.loads(paths["evaluation.json"].read_text())
    validate_evaluation(evaluation, observed["conversion-manifest.json"]["sha256"])
    verification = json.loads(paths["runtime-verification.json"].read_text())
    runtime_wheel = validate_runtime_verification(verification, observed)
    # This checks the checkpoint's actual packed tensors, not only its labels.
    from s1.nvfp4 import validate_nvfp4_artifact

    packed = validate_nvfp4_artifact(model)
    if packed.get("status") != "ok" or packed.get("format") != "nvfp4":
        raise ValueError("packed NVFP4 structural validation did not succeed")
    return {
        "schema_version": 1,
        "status": "validated_locally",
        "format": "nvfp4",
        "local_path": str(model),
        "validated_at": timestamp(),
        "conversion_manifest_sha256": observed["conversion-manifest.json"]["sha256"],
        "evaluation_sha256": observed["evaluation.json"]["sha256"],
        "runtime_verification_sha256": observed["runtime-verification.json"]["sha256"],
        "runtime_wheel": runtime_wheel,
        "preupload_inventory_sha256": digest(files),
        "packed_checkpoint": packed,
        "files": files,
    }


def verify_remote(api, download, repository, revision, files, cache_dir):
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        raise ValueError("Hub did not return an immutable publication revision")
    info = api.model_info(repository, revision=revision, files_metadata=True, token=False)
    if info.sha != revision or info.private or getattr(info, "gated", False):
        raise ValueError("immutable uploaded revision is not publicly accessible")
    siblings = {item.rfilename: item for item in info.siblings}
    expected_names = {entry["path"] for entry in files}
    if set(siblings) - expected_names - {".gitattributes"}:
        raise ValueError("remote repository contains files outside the upload whitelist")
    verified = []
    for entry in files:
        name = entry["path"]
        remote = siblings.get(name)
        if remote is None or remote.size != entry["size_bytes"]:
            raise ValueError(f"remote file missing or size differs: {name}")
        lfs = remote.lfs
        if lfs:
            sha = lfs.get("sha256") if isinstance(lfs, dict) else lfs.sha256
            size = lfs.get("size") if isinstance(lfs, dict) else lfs.size
            if sha != entry["sha256"] or size != entry["size_bytes"]:
                raise ValueError(f"remote LFS identity differs: {name}")
            method = "remote_lfs_sha256_and_size"
        else:
            downloaded = Path(
                download(
                    repo_id=repository,
                    filename=name,
                    revision=revision,
                    repo_type="model",
                    token=False,
                    cache_dir=str(cache_dir),
                    force_download=True,
                )
            )
            if (
                downloaded.stat().st_size != entry["size_bytes"]
                or file_sha256(downloaded) != entry["sha256"]
            ):
                raise ValueError(f"anonymous immutable file download differs: {name}")
            method = "anonymous_immutable_download_sha256_and_size"
        verified.append({**entry, "verification": method})
    return verified


def publish_package(model, allow_files, repository, receipt, *, validate_only=False):
    model = Path(model).absolute()
    receipt = Path(receipt).absolute()
    if receipt.resolve().is_relative_to(model.resolve()):
        raise ValueError("publication receipt must be outside the uploaded package")
    if not re.fullmatch(r"[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*", repository):
        raise ValueError("repository must be an explicit Hugging Face owner/model name")
    result = validate_package(model, allow_files)
    result.update(repository=repository, url=f"https://huggingface.co/{repository}")
    write_json(receipt, result)
    if validate_only:
        return result

    from huggingface_hub import CommitOperationAdd, HfApi, get_token, hf_hub_download

    if not get_token():
        raise ValueError("Hugging Face authentication is unavailable; use hf auth login")
    api = HfApi()
    result.update(status="upload_started", started_at=timestamp())
    write_json(receipt, result)
    try:
        api.create_repo(repo_id=repository, repo_type="model", private=False, exist_ok=True)
        current = api.model_info(repository, files_metadata=True)
        if current.private or getattr(current, "gated", False):
            raise ValueError("public publication requires an ungated public target repository")
        extra = (
            {item.rfilename for item in current.siblings} - set(allow_files) - {".gitattributes"}
        )
        if extra:
            raise ValueError(f"target repository contains unlisted files: {sorted(extra)}")
        result["parent_commit"] = current.sha
        commit = api.create_commit(
            repo_id=repository,
            repo_type="model",
            revision="main",
            parent_commit=current.sha,
            commit_message="Publish evaluated packed NVFP4 System One checkpoint",
            operations=[
                CommitOperationAdd(
                    path_in_repo=entry["path"], path_or_fileobj=model / entry["path"]
                )
                for entry in result["files"]
            ],
        )
        result.update(
            status="uploaded_verification_pending",
            published_revision=commit.oid,
            commit_url=str(commit.commit_url),
            published_at=timestamp(),
        )
        write_json(receipt, result)
        result["files"] = verify_remote(
            HfApi(token=False),
            hf_hub_download,
            repository,
            commit.oid,
            result["files"],
            receipt.parent / ".hub-publication-cache",
        )
        result.update(
            status="verified_public",
            verified_at=timestamp(),
            verification_scope=(
                "Anonymous immutable Hub inventory; all LFS SHA256/size identities; "
                "fresh anonymous immutable downloads of non-LFS files. "
                "No remote inference or full weight redownload."
            ),
        )
        write_json(receipt, result)
    except Exception as exc:
        result.update(
            status=(
                "verification_failed"
                if result.get("published_revision")
                else "upload_failed_or_unconfirmed"
            ),
            error_type=type(exc).__name__,
            failed_at=timestamp(),
        )
        write_json(receipt, result)
        raise
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--allow-file", action="append", default=[])
    parser.add_argument(
        "--file-list", type=Path, help="JSON array of exact repository-relative paths"
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    names = args.allow_file
    if args.file_list:
        listed = json.loads(args.file_list.read_text())
        if not isinstance(listed, list):
            raise ValueError("--file-list must contain a JSON array of exact paths")
        names.extend(listed)
    result = publish_package(
        args.model, names, args.repository, args.receipt, validate_only=args.validate_only
    )
    print(json.dumps({key: result[key] for key in ("status", "repository", "url")}))
    return result


if __name__ == "__main__":
    main()
