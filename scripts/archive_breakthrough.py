"""Create a deterministic, exclusive archive of breakthrough research evidence.

This is byte archival, not metric certification. Missing execution receipts stay
not_run; a frozen Git snapshot never substitutes for actual executed source.
Research heads/features/adapters are included, but baseline model weights and
redundant cloud export ZIPs are explicitly excluded. No cloud call is made.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "cc20b68b45d08dd1a7e309d46c4d8807c4616f66"
PROTOCOL = "configs/experiments/jevbench-breakthrough-20261008.json"
ARTIFACT_ROOT = ROOT / "artifacts/jevbench/breakthrough-20261008"
RESEARCH_BINARIES = {
    "decision-head.pt",
    "train-features.pt",
    "adapter.safetensors",
    "adapter_model.safetensors",
}
WEIGHT_SUFFIXES = {".safetensors", ".pt", ".pth", ".bin", ".gguf", ".onnx"}
MAX_MEMBER_BYTES = 128 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
ALIASES = {
    "campaign": "apps/modal/breakthrough.py",
    "breakthrough_prompt.py": "scripts/breakthrough_prompt.py",
    "breakthrough_training.py": "scripts/breakthrough_training.py",
    "data_generator": "scripts/prepare_breakthrough_data.py",
    "public_runner": "scripts/run_jevbench_public.py",
    "release_data_loader": "scripts/release_training.py",
    "checkpoint_identity_helper": "scripts/prepare_mlx_validation.py",
    "prospective_protocol": PROTOCOL,
}


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON constant: {value}")

    return json.loads(raw, parse_constant=reject_constant)


def _safe_name(name):
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or ":" in name
        or any(ord(char) < 32 for char in name)
        or name.startswith("/")
        or any(part in ("", ".", "..") for part in name.split("/"))
    ):
        raise ValueError(f"unsafe archive member name: {name!r}")
    return name


def _read_tree(root, *, allow_research=False):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"ordinary evidence directory required: {root}")
    files, omitted = {}, {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError(f"symlink cannot enter evidence archive: {path}")
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(f"regular evidence file required: {path}")
        name = _safe_name(path.relative_to(root).as_posix())
        reason = None
        if path.name in {"result.zip", "results.zip"}:
            reason = "redundant cloud export ZIP; extracted regular evidence retained"
        elif path.suffix.lower() in WEIGHT_SUFFIXES and not (
            allow_research
            and path.name in RESEARCH_BINARIES
            and len(PurePosixPath(name).parts) >= 4
            and PurePosixPath(name).parts[0] == "runs"
            and PurePosixPath(name).parts[2] in {"train", "train-recovered"}
        ):
            reason = "baseline or unrecognized model weights; not research head/features/adapter"
        if reason:
            omitted[name] = {"size_bytes": info.st_size, "reason": reason}
            continue
        if info.st_size > MAX_MEMBER_BYTES:
            raise ValueError(f"oversized evidence member: {path}")
        files[name] = path.read_bytes()
    return files, omitted


def frozen_sources(commit=SOURCE_COMMIT):
    """Read immutable preflight source bytes; these are not execution proof."""
    paths = [
        "src/s1",
        "apps/modal/breakthrough.py",
        *ALIASES.values(),
        "configs/benchmarks/jevbench-public.json",
        "examples/benchmarks/mps.jsonl",
        "pyproject.toml",
        "uv.lock",
        "LICENSE",
        "README.md",
    ]
    raw = subprocess.run(
        ["git", "-C", str(ROOT), "archive", "--format=tar", commit, *sorted(set(paths))],
        check=True,
        capture_output=True,
    ).stdout
    sources = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        for member in archive:
            _safe_name(member.name.rstrip("/"))
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"nonregular frozen source: {member.name}")
            sources[member.name] = archive.extractfile(member).read()
    return sources


def default_support_files():
    files, omitted = _read_tree(ROOT / "artifacts/release/datasets")
    if omitted:
        raise ValueError("unexpected weights or ZIP in complete release data")
    support = {f"artifacts/release/datasets/{name}": raw for name, raw in files.items()}
    paths = {
        "examples/benchmarks/mps.jsonl": ROOT / "examples/benchmarks/mps.jsonl",
        "configs/benchmarks/jevbench-public.json": ROOT / "configs/benchmarks/jevbench-public.json",
        "licenses/benchmarkheaven-LICENSE": Path(
            "/private/tmp/s1-jevbench-official-20261008/LICENSE"
        ),
        "licenses/coherence-LICENSE": ROOT / "artifacts/benchmark-tools/jevbench/LICENSE",
        "licenses/coherence-NOTICE": ROOT / "artifacts/benchmark-tools/jevbench/NOTICE",
        "tools/archive_breakthrough.py.txt": Path(__file__),
        "tools/analyze_breakthrough.py.txt": ROOT / "scripts/analyze_breakthrough.py",
        "publication/bf16-artifact-manifest.json": ROOT
        / "artifacts/release/hub-public-verification/bf16/artifact-manifest.json",
    }
    for name, path in paths.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"regular support file required: {path}")
        support[name] = path.read_bytes()
    return support


def default_outcome_reports(directory=None):
    """Capture available current reports separately from frozen/executed source."""
    directory = (
        Path(directory)
        if directory is not None
        else ROOT / "docs/validation/2026-10-08/breakthrough"
    )
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        raise ValueError("ordinary outcome report directory required")
    reports = {}
    for name in ("README.md", "results.md"):
        path = directory / name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError(f"regular outcome report required: {path}")
        if path.exists():
            reports[name] = path.read_bytes()
    return reports


def _hash_check(files, name, expected, scope):
    _safe_name(name)
    if name not in files or sha256(files[name]) != expected:
        raise ValueError(f"{scope} checksum mismatch or missing file: {name}")


def _input_checks(files, support, protocol):
    data = protocol["data"]
    synthetic = data["synthetic"]
    _hash_check(files, "data/manifest.json", synthetic["manifest_sha256"], "synthetic manifest")
    manifest = _json(files["data/manifest.json"])
    for name, digest in synthetic["files"].items():
        _hash_check(files, "data/" + _safe_name(name), digest, "frozen synthetic data")
    for name, dataset in manifest["datasets"].items():
        _hash_check(files, "data/" + _safe_name(name), dataset["file_sha256"], "synthetic dataset")
    release = data["release_regression"]
    prefix = release["directory"] + "/"
    _hash_check(support, prefix + "manifest.json", release["manifest_sha256"], "release manifest")
    release_manifest = _json(support[prefix + "manifest.json"])
    for name, digest in release_manifest["files"].items():
        _hash_check(support, prefix + _safe_name(name), digest, "complete release data")
    for name, spec in release.get("evaluation_files", {}).items():
        _hash_check(support, prefix + _safe_name(name), spec["file_sha256"], "release evaluation")
    fixture = data["native_fixture"]
    _hash_check(support, fixture["path"], fixture["file_sha256"], "native fixture")
    public = data["public231"]
    _hash_check(
        support,
        public["source_contract"],
        public["source_contract_sha256"],
        "public source contract",
    )
    return {
        "synthetic": "verified",
        "complete_release_data": "verified",
        "native_fixture": "verified",
        "public_source_contract": "verified",
    }


def _source_checks(files, prefix, receipt, sources, protocol_raw):
    declared = receipt.get("source_files", {})
    if not isinstance(declared, dict):
        raise ValueError("receipt source_files must be a mapping")
    expected = dict(ALIASES)
    expected.update(
        {
            name.removeprefix("src/"): name
            for name in sources
            if name.startswith("src/s1/") and name.endswith(".py")
        }
    )
    if not set(expected).issubset(declared):
        raise ValueError(f"receipt does not prove every frozen executed source: {prefix}")
    saved = {
        name.removeprefix(prefix + "sources/"): raw
        for name, raw in files.items()
        if name.startswith(prefix + "sources/")
    }
    if set(saved) != {f"{_safe_name(key)}.txt" for key in declared}:
        raise ValueError(f"receipt and saved executed source population differ: {prefix}")
    checks = {}
    for key, digest in sorted(declared.items()):
        raw = saved[f"{key}.txt"]
        if sha256(raw) != digest:
            raise ValueError(f"receipt/executed source checksum mismatch: {prefix}{key}")
        if key in expected and raw != sources[expected[key]]:
            raise ValueError(f"executed source differs from frozen preflight commit: {prefix}{key}")
        checks[key] = (
            "receipt_hash_and_frozen_bytes_verified"
            if key in expected
            else "receipt_hash_verified; no_preflight_alias"
        )
    if saved["prospective_protocol.txt"] != protocol_raw or receipt.get(
        "prospective_protocol"
    ) != _json(protocol_raw):
        raise ValueError(f"receipt prospective protocol differs: {prefix}")
    return checks


def _recovered_transport(files, omitted, prefix, run, receipt_raw):
    """Prove the one supported physical alias is an exact completed download."""
    metadata = prefix + ".evidence-download/"
    transport_raw = files.get(metadata + "receipt.json")
    if transport_raw is None:
        raise ValueError("train-recovered requires a completed transport receipt")
    transport = _json(transport_raw)
    receipt = _json(receipt_raw)
    if (
        receipt.get("status") not in {"completed", "failed"}
        or transport.get("status") != "completed"
        or transport.get("run") != run
        or transport.get("stage") != "train"
        or transport.get("volume") != "gemma-unified-system-one"
        or transport.get("remote_prefix") != f"breakthrough/{run}/train"
        or transport.get("gpu_receipt_sha256") != sha256(receipt_raw)
        or transport.get("gpu_receipt_status") != receipt.get("status")
        or transport.get("errors") != []
    ):
        raise ValueError("train-recovered terminal receipt/transport identity differs")
    payload = {
        name.removeprefix(prefix): raw
        for name, raw in files.items()
        if name.startswith(prefix) and not name.startswith(metadata)
    }
    if any(name.startswith(prefix) for name in omitted):
        raise ValueError("train-recovered transport payload cannot contain omitted files")
    checks = {}
    for field, collection in (
        ("members", payload),
        (
            "metadata_members",
            {
                name.removeprefix(metadata): raw
                for name, raw in files.items()
                if name.startswith(metadata) and name != metadata + "receipt.json"
            },
        ),
    ):
        records = transport.get(field)
        if not isinstance(records, list):
            raise ValueError("train-recovered transport member list missing")
        names = [_safe_name(record["path"]) for record in records]
        if len(set(names)) != len(names) or set(names) != set(collection):
            raise ValueError("train-recovered transport member population differs")
        for record in records:
            name = record["path"]
            if sha256(collection[name]) != record.get("sha256") or len(
                collection[name]
            ) != record.get("size_bytes"):
                raise ValueError("train-recovered transport member checksum/size differs")
        checks[field] = len(records)
    if (
        transport.get("expected_files") != len(payload)
        or transport.get("expected_size_bytes") != sum(map(len, payload.values()))
        or transport.get("executed_sources_verified") != len(receipt.get("source_files", {}))
        or sha256(files.get(metadata + "export_breakthrough_evidence.py.txt", b""))
        != transport.get("source_sha256")
    ):
        raise ValueError("train-recovered transport totals/helper source differ")
    failure = transport.get("original_export_failure")
    failure_raw = files.get(metadata + "original-export-failure.txt")
    if failure != (
        {"sha256": sha256(failure_raw), "size_bytes": len(failure_raw)}
        if failure_raw is not None
        else {"status": "not_supplied"}
    ):
        raise ValueError("train-recovered original export failure record differs")
    canonical_prefix = f"runs/{run}/train/"
    canonical = {
        name.removeprefix(canonical_prefix): raw
        for name, raw in files.items()
        if name.startswith(canonical_prefix)
        and not name.startswith(canonical_prefix + ".evidence-download/")
    }
    common = sorted(set(canonical) & set(payload))
    if any(canonical[name] != payload[name] for name in common):
        raise ValueError("canonical train and train-recovered common member bytes differ")
    return {
        "logical_stage": "train",
        "physical_directory": "train-recovered",
        "transport_status": "completed",
        "transport_receipt_sha256": sha256(transport_raw),
        "gpu_receipt_sha256": transport["gpu_receipt_sha256"],
        "remote_prefix": transport["remote_prefix"],
        "member_checks": checks,
        "canonical_transport_status": "evidence_present"
        if canonical or any(name.startswith(canonical_prefix) for name in omitted)
        else "not_downloaded",
        "canonical_common_members": len(common),
        "canonical_common_bytes": "all_identical",
        "canonical_only_members": sorted(set(canonical) - set(payload)),
        "recovered_only_members": sorted(set(payload) - set(canonical)),
        "original_cpu_export": {
            "status": "not_assessed_by_archiver",
            "transport_failure_record": transport.get("original_export_failure"),
            "scope": "Original CPU export pending/failure is separate from GPU and recovery status; consult preserved diagnostics and outcome reports",
        },
    }


def _execution(files, omitted, protocol, sources, protocol_raw):
    planned = {
        stage["id"]: [profile["id"] for profile in stage["profiles"]]
        for stage in protocol["stages"]
    }
    for stage, profiles in planned.items():
        _safe_name(stage)
        for profile in profiles:
            _safe_name(profile)
    all_names = set(files) | set(omitted)
    runs = sorted(
        {
            name.split("/")[1]
            for name in all_names
            if name.startswith("runs/") and len(name.split("/")) >= 4
        }
    )
    cells, recoveries = {}, {}
    for run in runs:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", run):
            raise ValueError("invalid run directory")
        actual_stages = {
            name.split("/")[2]
            for name in all_names
            if name.startswith(f"runs/{run}/") and len(name.split("/")) >= 4
        }
        allowed = set(planned) | ({"train-recovered"} if "train" in planned else set())
        if actual_stages - allowed:
            raise ValueError(f"unplanned stage directory: {run}")
        physical_stages = list(planned)
        if "train-recovered" in actual_stages:
            physical_stages.append("train-recovered")
        for physical_stage in physical_stages:
            stage = "train" if physical_stage == "train-recovered" else physical_stage
            profiles = planned[stage]
            prefix = f"runs/{run}/{physical_stage}/"
            stage_files = {name: raw for name, raw in files.items() if name.startswith(prefix)}
            stage_omitted = sum(name.startswith(prefix) for name in omitted)
            receipt_raw = files.get(prefix + "receipt.json")
            cell = {
                "declared_status": "not_run",
                "evidence_status": "not_run",
                "expected_profiles": profiles,
                "profiles": {name: "not_run" for name in profiles},
                "source_checks": {},
                "files": len(stage_files),
                "omitted_files": stage_omitted,
            }
            if physical_stage == "train-recovered":
                if receipt_raw is None:
                    raise ValueError("train-recovered requires a terminal GPU receipt")
                recoveries[f"{run}/{physical_stage}"] = _recovered_transport(
                    files, omitted, prefix, run, receipt_raw
                )
                cell["logical_stage"] = stage
                cell["physical_directory"] = physical_stage
            if not receipt_raw:
                cell["evidence_status"] = (
                    "missing_receipt" if stage_files or stage_omitted else "not_run"
                )
            else:
                receipt = _json(receipt_raw)
                if receipt.get("run") != run or receipt.get("stage") != stage:
                    raise ValueError(f"receipt run/stage mismatch: {prefix}")
                status = receipt.get("status")
                if status not in {"running", "completed", "failed", "blocked", "not_run"}:
                    raise ValueError(f"unknown receipt execution status: {prefix}")
                cell["declared_status"] = status
                if status in {"running", "completed", "failed"}:
                    cell["source_checks"] = _source_checks(
                        files, prefix, receipt, sources, protocol_raw
                    )
                reported = [profile["name"] for profile in receipt.get("profiles", [])]
                if len(reported) != len(set(reported)) or set(reported) - set(profiles):
                    raise ValueError(f"receipt profile population differs: {prefix}")
                for name in profiles:
                    profile_path = prefix + name + "/receipt.json"
                    if name in reported and profile_path in files:
                        identity = _json(files[profile_path]).get("identity", {})
                        if identity.get("variant") != name or identity.get(
                            "source_files"
                        ) != receipt.get("source_files"):
                            raise ValueError(
                                f"profile receipt identity/source differs: {profile_path}"
                            )
                        cell["profiles"][name] = "receipt_saved"
                    elif name in reported or profile_path in files:
                        cell["profiles"][name] = "incomplete_receipt_evidence"
                cell["evidence_status"] = (
                    "receipt_completed"
                    if status == "completed"
                    and all(value == "receipt_saved" for value in cell["profiles"].values())
                    else "incomplete"
                    if status == "completed"
                    else status
                )
                cell["gpu"] = receipt.get("gpu")
                cell["research_binary_checks"] = _binary_checks(files, prefix, receipt)
            cells[f"{run}/{physical_stage}"] = cell
        if "train-recovered" in actual_stages:
            canonical = cells[f"{run}/train"]
            if canonical["evidence_status"] == "not_run":
                canonical.update(
                    declared_status="not_observed_in_this_directory",
                    evidence_status="not_downloaded",
                    recovered_by=f"{run}/train-recovered",
                )
    execution_status = (
        _json(files["execution-status.json"]) if "execution-status.json" in files else None
    )
    if execution_status is not None:
        digest = execution_status.get("prospective_protocol_sha256")
        if digest is not None and digest != sha256(protocol_raw):
            raise ValueError("execution-status prospective protocol hash differs")
        if execution_status.get("gpu_started") is False and any(
            cell["declared_status"] in {"running", "completed", "failed"} for cell in cells.values()
        ):
            raise ValueError("no-GPU status conflicts with an executed stage receipt")
    state = (
        "no_gpu"
        if not runs and execution_status and execution_status.get("gpu_started") is False
        else "not_run"
        if not runs
        else "run_evidence_present"
    )
    return {
        "state": state,
        "state_scope": "Frozen Modal A100 stages only; no_gpu does not describe separate local MPS hardware use",
        "planned_stages": planned,
        "stage_receipts": cells,
        "recovery_transfers": recoveries,
        "local_experiments": _local_execution(files, protocol, sources),
        "local_status_record": execution_status,
        "population_validation": "not_performed_by_archiver; receipt_completed does not certify complete benchmark support or improvement",
    }


def _local_execution(files, protocol, sources):
    experiments = {}
    roots = sorted(
        {name.split("/")[0] for name in files if name.startswith("local-mps-") and "/" in name}
    )
    for root in roots:
        prefix = root + "/"
        receipt_raw = files.get(prefix + "receipt.json")
        if receipt_raw is None:
            experiments[root] = {
                "evidence_status": "missing_receipt",
                "scope": "Separate local MPS probe; not frozen A100 profiles",
            }
            continue
        receipt = _json(receipt_raw)
        if (
            receipt.get("device") != "mps"
            or receipt.get("cloud_calls") != 0
            or receipt.get("training_executed") is not False
        ):
            raise ValueError(f"local MPS scope differs from receipt: {root}")
        if receipt.get("data_sha256") != protocol["data"]["native_fixture"]["file_sha256"]:
            raise ValueError(f"local MPS input hash differs: {root}")
        declared = receipt.get("source_files", {})
        saved = {
            name.removeprefix(prefix + "sources/"): raw
            for name, raw in files.items()
            if name.startswith(prefix + "sources/")
        }
        if set(saved) != {f"{_safe_name(key)}.txt" for key in declared} or not declared:
            raise ValueError(f"local MPS source population differs: {root}")
        expected = {
            "scripts/breakthrough_prompt.py",
            "scripts/breakthrough_training.py",
            "scripts/prepare_mlx_validation.py",
        } | {name for name in sources if name.startswith("src/s1/") and name.endswith(".py")}
        if not expected.issubset(declared):
            raise ValueError(f"local MPS complete package source proof missing: {root}")
        checks = {}
        differences = {}
        for name, digest in sorted(declared.items()):
            raw = saved[name + ".txt"]
            if sha256(raw) != digest:
                raise ValueError(f"local MPS receipt/source hash differs: {root}/{name}")
            if name in sources and raw != sources[name]:
                differences[name] = {
                    "frozen_a100_sha256": sha256(sources[name]),
                    "executed_local_sha256": digest,
                }
            checks[name] = (
                "receipt_hash_verified; differs_from_frozen_A100_source"
                if name in differences
                else "receipt_hash_and_frozen_bytes_verified"
                if name in sources
                else "receipt_hash_verified; not_in_frozen_A100_source"
            )
        status = receipt.get("status")
        if status not in {"running", "completed", "failed"}:
            raise ValueError(f"local MPS receipt status differs: {root}")
        planned = receipt.get("profiles_declared", [])
        reported = [profile["name"] for profile in receipt.get("profiles", [])]
        if len(set(reported)) != len(reported) or set(reported) - set(planned):
            raise ValueError(f"local MPS profile population differs: {root}")
        profile_status = {
            name: "records_saved"
            if name in reported and prefix + _safe_name(name) + ".json" in files
            else "incomplete"
            for name in planned
        }
        experiments[root] = {
            "declared_status": status,
            "evidence_status": "receipt_completed"
            if status == "completed"
            and planned
            and all(value == "records_saved" for value in profile_status.values())
            else "incomplete"
            if status == "completed"
            else status,
            "scope": receipt.get("scope"),
            "source_checks": checks,
            "preflight_differences": differences,
            "profiles": profile_status,
            "population_validation": "not_performed_by_archiver; native fixture probe does not establish broad model quality",
        }
    return experiments


def _binary_checks(files, prefix, receipt):
    checks = {}
    for profile in receipt.get("profiles", []):
        identity = profile.get("identity", {})
        if "research_head_sha256" in identity:
            name = prefix + "decision-head.pt"
            _hash_check(files, name, identity["research_head_sha256"], "trained decision head")
            checks[name.removeprefix(prefix)] = "receipt_hash_verified"
        for name, spec in identity.get("research_adapter_files", {}).items():
            path = prefix + "mixed-lora/adapter/" + _safe_name(name)
            _hash_check(files, path, spec["sha256"], "trained adapter")
            if len(files[path]) != spec["bytes"]:
                raise ValueError(f"trained adapter size differs: {path}")
            checks[path.removeprefix(prefix)] = "receipt_hash_and_size_verified"
    training_raw = files.get(prefix + "mixed-lora/training.json")
    if training_raw:
        for name, spec in _json(training_raw).get("adapter_files", {}).items():
            path = prefix + "mixed-lora/adapter/" + _safe_name(name)
            _hash_check(files, path, spec["sha256"], "training adapter")
            if len(files[path]) != spec["bytes"]:
                raise ValueError(f"training adapter size differs: {path}")
            checks[path.removeprefix(prefix)] = "training_hash_and_size_verified"
    return checks


def _zip_bytes(members):
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system, info.external_attr = 3, 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, raw, compresslevel=9)
    return bundle.getvalue()


def archive_breakthrough(
    artifact_root,
    output,
    *,
    source_commit=SOURCE_COMMIT,
    package_sources=None,
    support_files=None,
    outcome_reports=None,
):
    artifact_root, output = Path(artifact_root), Path(output)
    manifest_path = output.with_suffix(".manifest.json")
    if (
        output.exists()
        or output.is_symlink()
        or manifest_path.exists()
        or manifest_path.is_symlink()
    ):
        raise FileExistsError("archive or manifest already exists")
    if output.resolve().is_relative_to(artifact_root.resolve()):
        raise ValueError("archive output cannot be inside its input evidence tree")
    files, omitted = _read_tree(artifact_root, allow_research=True)
    sources = frozen_sources(source_commit) if package_sources is None else package_sources
    support_from_disk = support_files is None
    support = default_support_files() if support_from_disk else support_files
    reports_from_disk = outcome_reports is None
    reports = default_outcome_reports() if reports_from_disk else outcome_reports
    protocol_raw = sources[PROTOCOL]
    protocol = _json(protocol_raw)
    input_checks = _input_checks(files, support, protocol)
    execution = _execution(files, omitted, protocol, sources, protocol_raw)
    members = {}
    for prefix, collection in (
        ("research", files),
        ("inputs", support),
        ("preflight", sources),
        ("reports", reports),
    ):
        for name, raw in collection.items():
            _safe_name(name)
            member = f"{prefix}/{name}" + (
                ".txt" if prefix == "preflight" and name.endswith(".py") else ""
            )
            if not isinstance(raw, bytes) or len(raw) > MAX_MEMBER_BYTES or member in members:
                raise ValueError(f"invalid or oversized member: {member}")
            members[member] = raw
    if sum(map(len, members.values())) > MAX_TOTAL_BYTES:
        raise ValueError("evidence exceeds conservative total size limit")
    zip_raw = _zip_bytes(members)
    with zipfile.ZipFile(io.BytesIO(zip_raw)) as zipped:
        member_records = [
            {
                "path": name,
                "sha256": sha256(raw),
                "size_bytes": len(raw),
                "crc32": f"{zipped.getinfo(name).CRC:08x}",
            }
            for name, raw in sorted(members.items())
        ]
    manifest = {
        "schema_version": 1,
        "archive": {"filename": output.name, "sha256": sha256(zip_raw), "size_bytes": len(zip_raw)},
        "members": member_records,
        "uncompressed_size_bytes": sum(map(len, members.values())),
        "preflight_source_commit": source_commit,
        "preflight_source_scope": "Immutable prospective source snapshot; execution proof only comes from saved stage sources and matching receipts",
        "archive_tool_scope": "Current offline archive tool saved under inputs/tools; it is not a model-cell executed source",
        "outcome_reports": {
            "scope": "Current outcome reports captured at archival time; not executed or frozen model source",
            "members": [
                record for record in member_records if record["path"].startswith("reports/")
            ],
        },
        "input_checks": input_checks,
        "execution": execution,
        "omitted_files": omitted,
        "binary_scope": "Only trained research decision heads, detached features and adapters; baseline checkpoint weights are excluded",
        "official": protocol.get("official", {}),
        "scope": "Exact byte archival and source/input integrity; does not assess metrics, costs, rank, promotion or GPU latency",
        "zip_metadata": {
            "timestamp": "1980-01-01T00:00:00",
            "member_order": "sorted",
            "compression": "deflate-9",
            "mode": "0644",
        },
    }
    if _read_tree(artifact_root, allow_research=True) != (files, omitted):
        raise ValueError("evidence changed while archiving")
    if support_from_disk and default_support_files() != support:
        raise ValueError("support files changed while archiving")
    if reports_from_disk and default_outcome_reports() != reports:
        raise ValueError("outcome reports changed while archiving")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        temporary = Path(temporary)
        (temporary / "evidence.zip").write_bytes(zip_raw)
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        os.link(temporary / "evidence.zip", output)
        try:
            os.link(temporary / "manifest.json", manifest_path)
        except BaseException:
            output.unlink()
            raise
    verify_archive(output)
    return manifest


def verify_archive(output, manifest_path=None):
    """Verify every stored member's exact name, metadata, CRC, size and SHA256."""
    output = Path(output)
    manifest_path = Path(manifest_path) if manifest_path else output.with_suffix(".manifest.json")
    if output.is_symlink() or manifest_path.is_symlink():
        raise ValueError("archive and manifest must be ordinary files")
    manifest = _json(manifest_path.read_bytes())
    raw = output.read_bytes()
    if manifest["archive"] != {
        "filename": output.name,
        "sha256": sha256(raw),
        "size_bytes": len(raw),
    }:
        raise ValueError("ZIP checksum/size/filename mismatch")
    expected = manifest["members"]
    names = [_safe_name(member["path"]) for member in expected]
    if names != sorted(set(names)):
        raise ValueError("manifest members must be sorted and unique")
    if "outcome_reports" in manifest and manifest["outcome_reports"]["members"] != [
        record for record in expected if record["path"].startswith("reports/")
    ]:
        raise ValueError("outcome report manifest index differs from hashed members")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if archive.namelist() != names or archive.testzip() is not None:
            raise ValueError("ZIP member population or CRC differs")
        for member in expected:
            info = archive.getinfo(member["path"])
            payload = archive.read(info)
            if (
                sha256(payload) != member["sha256"]
                or len(payload) != member["size_bytes"]
                or f"{info.CRC:08x}" != member["crc32"]
            ):
                raise ValueError(f"member checksum/size/CRC mismatch: {info.filename}")
            if (
                info.date_time != (1980, 1, 1, 0, 0, 0)
                or info.create_system != 3
                or info.external_attr != 0o100644 << 16
                or info.compress_type != zipfile.ZIP_DEFLATED
            ):
                raise ValueError(f"member metadata differs: {info.filename}")
    if manifest["uncompressed_size_bytes"] != sum(member["size_bytes"] for member in expected):
        raise ValueError("uncompressed member total differs")
    return {
        "verified": True,
        "members": len(expected),
        "archive": manifest["archive"],
        "execution_state": manifest["execution"]["state"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs/validation/2026-10-08/breakthrough/raw-evidence.zip",
    )
    parser.add_argument("--source-commit", default=SOURCE_COMMIT)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        result = verify_archive(args.verify)
    else:
        archive_breakthrough(args.artifact_root, args.output, source_commit=args.source_commit)
        result = verify_archive(args.output)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
