"""Archive finished JevBench research evidence without modifying its originals.

Members are sorted, use fixed ZIP metadata, and retain every raw run file.
The external manifest hashes every member and the ZIP itself. Model weights
are rejected, not silently omitted. This does not certify benchmark metrics,
run a sealed suite, calculate a cost, or submit a leaderboard result.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ("20261008-public-01", "20261008-public-02", "20261008-public-03")
PACKAGE_COMMIT = "501c8a466fa6e694b184283d0574526a500b1655"
RUNTIME_COMMIT = "1b581e90d4458cb658e565bc608990e8dee6174b"
RUNTIME_WHEEL_SHA256 = "8c16a6536480d6632d5f5478989ba1920d6ec2568eafe91647df2cb8591b38a1"
RUNTIME_WHEEL_SIZE = 195197
RUNTIME_SOURCE_MEMBERS = {"s1/backends.py", "s1/cli.py", "s1/evaluation/gemma.py", "s1/unified.py"}
SOURCE_PINS = {
    "20261008-public-01": {
        "jevbench_campaign.py": "627e73c5a1bd549838af33a78dc1ebd7881b2752de0a635c04e6d766e38550d8",
        "run_jevbench_public.py": "f303209795ac932e251c786642c434c2a561cfb1299e4065e96ff0247f436701",
    },
    "20261008-public-02": {
        "jevbench_campaign.py": "468100cf1f4c639e42acf9bbd779c2d9ba39383268d49379c81f8aa9a15deba0",
        "run_jevbench_public.py": "cb476d16a6e89ee5223b928e3a550af18de8534934ae5431563503db9f909bd1",
    },
    "20261008-public-03": {
        "jevbench_campaign.py": "0013717d6cc451479db3e3985d26bd5950e1f574bc90e4f2d14e67d1f2c44de6",
        "run_jevbench_public.py": "cb476d16a6e89ee5223b928e3a550af18de8534934ae5431563503db9f909bd1",
        "unified.py": "42ba5f076af164e0e54690a735e25a272bdd678279901946c2e5accefa385a6c",
    },
}
WEIGHT_SUFFIXES = {".safetensors", ".gguf", ".pt", ".pth", ".bin"}
MAX_MEMBER_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    return json.loads(raw)


def _read_tree(root):
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"ordinary evidence directory required: {root}")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink cannot enter evidence archive: {path}")
        if not path.is_file():
            continue
        if path.suffix in WEIGHT_SUFFIXES or path.stat().st_size > MAX_MEMBER_BYTES:
            raise ValueError(f"weight or oversized evidence member: {path}")
        files[path.relative_to(root).as_posix()] = path.read_bytes()
    if not files:
        raise ValueError(f"empty evidence directory: {root}")
    return files


def frozen_package_sources():
    raw = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "archive",
            "--format=tar",
            PACKAGE_COMMIT,
            "src/s1",
            "pyproject.toml",
            "uv.lock",
            "LICENSE",
            "README.md",
            "configs/benchmarks/models.toml",
            "examples/benchmarks/mps.jsonl",
            "scripts/prepare_mlx_validation.py",
        ],
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        return {
            member.name: archive.extractfile(member).read() for member in archive if member.isfile()
        }


def default_support_files(artifact_root=ROOT / "artifacts/jevbench"):
    artifact_root = Path(artifact_root)
    paths = {
        "configs/benchmarks/jevbench-public.json": ROOT / "configs/benchmarks/jevbench-public.json",
        "licenses/benchmarkheaven-LICENSE": Path(
            "/private/tmp/s1-jevbench-official-20261008/LICENSE"
        ),
        "licenses/coherence-LICENSE": ROOT / "artifacts/benchmark-tools/jevbench/LICENSE",
        "licenses/coherence-NOTICE": ROOT / "artifacts/benchmark-tools/jevbench/NOTICE",
        **{
            f"tools/{name}.txt": ROOT / "scripts" / name
            for name in (
                "run_jevbench_public.py",
                "summarize_jevbench_campaign.py",
                "archive_jevbench_campaign.py",
            )
        },
    }
    for pattern in ("compiled-native-audit-*.json", "independent-native-audit-*.json"):
        paths.update({f"audits/{path.name}": path for path in sorted(artifact_root.glob(pattern))})
    return {
        name: path.read_bytes() for name, path in paths.items()
    } | runtime_current_support_files(artifact_root / "runtime-current")


def _git_bytes(commit, path):
    return subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{commit}:{path}"], check=True, capture_output=True
    ).stdout


def runtime_current_support_files(root):
    """Retain the newer wrapper wheel separately from older executed sources."""
    root = Path(root)
    if not root.exists():
        return {}
    manifest_path = root / "manifest.json"
    if root.is_symlink() or manifest_path.is_symlink():
        raise ValueError("runtime bundle and manifest must not be symlinks")
    manifest_raw = manifest_path.read_bytes()
    manifest = _json(manifest_raw)
    if (
        manifest.get("source_commit") != RUNTIME_COMMIT
        or set(manifest.get("wheel_source_checked", [])) != RUNTIME_SOURCE_MEMBERS
    ):
        raise ValueError("current runtime manifest commit/source scope differs")
    wheels = [name for name in manifest.get("files", {}) if name.endswith(".whl")]
    if len(wheels) != 1 or Path(wheels[0]).name != wheels[0]:
        raise ValueError("one local current runtime wheel required")
    name = wheels[0]
    wheel_path = root / name
    if wheel_path.is_symlink():
        raise ValueError("runtime wheel must not be a symlink")
    wheel_raw = wheel_path.read_bytes()
    expected = {"sha256": RUNTIME_WHEEL_SHA256, "size_bytes": RUNTIME_WHEEL_SIZE}
    if (
        manifest["files"][name] != expected
        or sha256(wheel_raw) != expected["sha256"]
        or len(wheel_raw) != expected["size_bytes"]
    ):
        raise ValueError("current runtime wheel hash/size differs")
    with zipfile.ZipFile(io.BytesIO(wheel_raw)) as wheel:
        for member in sorted(RUNTIME_SOURCE_MEMBERS):
            if wheel.read(member) != _git_bytes(RUNTIME_COMMIT, "src/" + member):
                raise ValueError(f"wheel source differs from current wrapper commit: {member}")
    selection = {
        "scope": "Current opt-in CLI/backend wrapper build; not older model-cell executed source",
        "source_commit": RUNTIME_COMMIT,
        "included": [name, "manifest.json"],
        "not_archived": {
            file: "Redundant documentation-heavy sdist; older source bundle and current wheel are retained"
            for file in manifest["files"]
            if file != name
        },
        "wheel_source_verification": "Four source members match the current wrapper Git commit byte for byte",
        "gpu_http_smoke": manifest.get("gpu_http_smoke", "not_recorded"),
    }
    return {
        "runtime-current/manifest.json": manifest_raw,
        f"runtime-current/{name}": wheel_raw,
        "runtime-current/archive-selection.json": json.dumps(
            selection, indent=2, sort_keys=True, allow_nan=False
        ).encode()
        + b"\n",
    }


def _receipt_source_checks(run_files, source_files, expected, package_sources):
    for name, digest in expected.items():
        if name not in source_files or sha256(source_files[name]) != digest:
            raise ValueError(f"executed source pin differs: {name}")
    checks = {}
    cells = {name.split("/", 1)[0] for name in run_files if "/" in name}
    for cell in sorted(cells):
        receipt_path = f"{cell}/receipt.json"
        if receipt_path not in run_files:
            raise ValueError(f"unfinished model cell lacks receipt: {cell}")
        identity = _json(run_files[receipt_path]).get("identity", {})
        declared = identity.get("source_files", {})
        aliases = {
            "jevbench_campaign.py": ("campaign",),
            "run_jevbench_public.py": ("public_runner", "scripts/run_jevbench_public.py"),
        }
        source_bytes = dict(source_files)
        if source_bytes.get("unified.py") != package_sources.get("src/s1/unified.py"):
            raise ValueError("executed unified source differs from frozen package")
        for name, raw in package_sources.items():
            if name.startswith("src/s1/") and name.endswith(".py"):
                source_bytes[name] = raw
                aliases[name] = (name.removeprefix("src/s1/"), name)
        cell_checks = {}
        for name, keys in aliases.items():
            if name not in source_bytes:
                raise ValueError(f"missing executed source: {name}")
            recorded = [declared[key] for key in keys if key in declared]
            actual = sha256(source_bytes[name])
            if any(digest != actual for digest in recorded):
                raise ValueError(f"receipt/source hash differs: {cell}/{name}")
            cell_checks[name] = "verified" if recorded else "not_recorded_in_receipt"
        checks[cell] = cell_checks
    return checks


def native_regression_audit(raw, fixture_raw):
    """Compare every probability and argmax in the eight saved native cases."""
    return _native_regression_audit(
        raw, fixture_raw, "eager", ("cached", "compiled"), single_pass=True
    )


def independent_native_regression_audit(raw, fixture_raw):
    """Compare independent question batching with its sequential reference."""
    result = _native_regression_audit(
        raw, fixture_raw, "sequential", ("independent",), single_pass=False
    )
    result["comparison_scope"] = (
        "Independent per-question prompts: sequential reference versus padding/mask-aware batch; "
        "does not assert equivalence to causal multi-question prediction"
    )
    result["precision_scope"] = (
        "BF16 batched and sequential GEMM shapes may round differently; preserve measured drift "
        "and every question's argmax without selecting a tolerance from the results"
    )
    result["timing_scope"] = "No synchronized paired timing population; no speedup claim"
    return result


def _native_regression_audit(raw, fixture_raw, baseline, comparisons, *, single_pass):
    expected = [_json(line) for line in fixture_raw.splitlines() if line.strip()]
    records = _json(raw)
    if (
        not isinstance(records, list)
        or len(expected) != 8
        or [r.get("case_id") for r in records] != [r["id"] for r in expected]
    ):
        raise ValueError("native case population/order differs from the eight original fixtures")
    result = {
        "cases": len(records),
        "baseline_variant": baseline,
        "comparisons": {},
        "timing_scope": "Single ordered calls include shape/compile effects; no paired speedup claim",
        "tolerance_gate": "not_predeclared; report measured drift without inventing a threshold",
    }
    for variant in comparisons:
        rows = []
        by_modality = {}
        for case, record in zip(expected, records):
            modalities = [media["type"] for media in case["request"]["media"]]
            if record.get("modalities") != modalities or set(record.get("variants", {})) != {
                baseline,
                *comparisons,
            }:
                raise ValueError("native modalities/variant population differs")
            bucket = (
                "mixed" if len(set(modalities)) > 1 else modalities[0] if modalities else "text"
            )
            variants = record["variants"]
            answers = {name: response["answers"] for name, response in variants.items()}
            if single_pass and any(
                response.get("passes") != 1
                or any(
                    response.get(key) != variants[baseline].get(key)
                    for key in ("model", "revision", "temperature")
                )
                for response in variants.values()
            ):
                raise ValueError("native model identity/temperature/pass count differs")
            ids = [question["id"] for question in case["request"]["questions"]]
            if any(set(value) != set(ids) for value in answers.values()):
                raise ValueError("native answer population differs")
            if not single_pass:
                execution = variants[variant].get("execution", {})
                batch_sizes, tokens = execution.get("batch_sizes"), execution.get("sequence_tokens")
                if (
                    execution.get("independent_questions") is not True
                    or execution.get("readout") != "candidate"
                    or execution.get("scope") != "whole_adapter_batch"
                    or not isinstance(batch_sizes, list)
                    or not isinstance(tokens, list)
                    or not batch_sizes
                    or any(type(value) is not int or value <= 0 for value in batch_sizes + tokens)
                    or sum(batch_sizes) != len(ids)
                    or len(tokens) != len(ids)
                    or type(execution.get("forward_calls")) is not int
                    or execution.get("forward_calls") != len(batch_sizes)
                ):
                    raise ValueError("independent batch execution population/semantics differs")
            drift, matches, count = 0.0, 0, 0
            labels_changed = []
            for question in case["request"]["questions"]:
                qid = question["id"]
                base, other = answers[baseline][qid], answers[variant][qid]
                if base.get("type") != question["type"] or other.get("type") != question["type"]:
                    raise ValueError("native answer type differs")
                a, b = base["probabilities"], other["probabilities"]
                labels = (
                    ["false", "true"]
                    if question["type"] == "noul"
                    else list(question["criteria"])
                    if isinstance(question["criteria"], dict)
                    else [str(index) for index in range(len(question["criteria"]))]
                )
                if (
                    not a
                    or set(a) != set(b)
                    or set(a) != set(labels)
                    or any(
                        type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
                        for probabilities in (a, b)
                        for p in probabilities.values()
                    )
                ):
                    raise ValueError("native probability labels/values invalid")
                for probabilities in (a, b):
                    if not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-6):
                        raise ValueError("native probabilities do not sum to one")
                drift = max(drift, *(abs(a[label] - b[label]) for label in a))
                base_label, other_label = max(labels, key=a.get), max(labels, key=b.get)
                matches += base_label == other_label
                if base_label != other_label:
                    labels_changed.append(
                        {"question_id": qid, "baseline": base_label, "variant": other_label}
                    )
                count += len(a)
                field = "choice" if question["type"] == "choice" else "level"
                if question["type"] != "noul" and (
                    base.get(field) != base_label or other.get(field) != other_label
                ):
                    raise ValueError("native reported label differs from probability argmax")
            rows.append(
                {
                    "case_id": case["id"],
                    "questions": len(ids),
                    "probabilities": count,
                    "argmax_matches": matches,
                    "max_abs_probability_drift": drift,
                    "labels_changed": labels_changed,
                }
            )
            group = by_modality.setdefault(
                bucket,
                {
                    "cases": 0,
                    "questions": 0,
                    "probabilities": 0,
                    "argmax_matches": 0,
                    "max_abs_probability_drift": 0.0,
                },
            )
            for key, value in (
                ("cases", 1),
                ("questions", len(ids)),
                ("probabilities", count),
                ("argmax_matches", matches),
            ):
                group[key] += value
            group["max_abs_probability_drift"] = max(group["max_abs_probability_drift"], drift)
        result["comparisons"][variant] = {
            "by_case": rows,
            "by_modality": by_modality,
            "max_abs_probability_drift": max(row["max_abs_probability_drift"] for row in rows),
            "argmax_matches": sum(row["argmax_matches"] for row in rows),
            "questions": sum(row["questions"] for row in rows),
            "probabilities": sum(row["probabilities"] for row in rows),
        }
    return result


def archive_campaign(
    artifact_root,
    output,
    *,
    runs=RUNS,
    source_pins=None,
    fixture=ROOT / "examples/benchmarks/mps.jsonl",
    package_sources=None,
    support_files=None,
):
    artifact_root, output = Path(artifact_root), Path(output)
    manifest_path = output.with_suffix(".manifest.json")
    if (
        output.exists()
        or manifest_path.exists()
        or output.is_symlink()
        or manifest_path.is_symlink()
    ):
        raise FileExistsError("archive and manifest destinations must both be new")
    if not runs or len(set(runs)) != len(runs) or any(run not in RUNS for run in runs):
        raise ValueError("unique known campaign runs required")
    source_pins = SOURCE_PINS if source_pins is None else source_pins
    package_commit = PACKAGE_COMMIT if package_sources is None else None
    package_sources = frozen_package_sources() if package_sources is None else package_sources
    support_from_disk = support_files is None
    support_files = default_support_files(artifact_root) if support_files is None else support_files
    fixture_raw = Path(fixture).read_bytes()
    if package_commit and fixture_raw != package_sources["examples/benchmarks/mps.jsonl"]:
        raise ValueError("native input fixture differs from frozen package")
    members, snapshots, run_metadata = {}, [], {}
    for run in runs:
        source_name = "executed-public" + run.rsplit("-", 1)[1]
        run_files, source_files = (
            _read_tree(artifact_root / run),
            _read_tree(artifact_root / source_name),
        )
        checks = _receipt_source_checks(
            run_files, source_files, source_pins.get(run, {}), package_sources
        )
        snapshots.extend(
            ((artifact_root / run, run_files), (artifact_root / source_name, source_files))
        )
        members.update({f"{run}/{name}": raw for name, raw in run_files.items()})
        members.update(
            {
                f"{source_name}/{name}.txt"
                if name.endswith(".py")
                else f"{source_name}/{name}": raw
                for name, raw in source_files.items()
            }
        )
        native_audits = {
            "native-regression.json": native_regression_audit,
            "independent-native-regression.json": independent_native_regression_audit,
        }
        native = {
            name: native_audits[Path(name).name](raw, fixture_raw)
            for name, raw in run_files.items()
            if Path(name).name in native_audits
        }
        coherence = {}
        for cell in checks:
            requested = (
                _json(run_files[f"{cell}/receipt.json"])
                .get("identity", {})
                .get("coherence_requested", "not_recorded")
            )
            report_present = f"{cell}/coherence-mini.json" in run_files
            cache_files = sum(name.startswith(f"{cell}/coherence-cache/") for name in run_files)
            if requested != "not_recorded" and type(requested) is not bool:
                raise ValueError("coherence_requested must be a boolean when recorded")
            if requested is False and (report_present or cache_files):
                raise ValueError("coherence was not requested but evidence is present")
            coherence[cell] = {
                "requested": requested,
                "report_present": report_present,
                "cache_files": cache_files,
                "status": "not_requested"
                if requested is False
                else "report_saved"
                if report_present
                else "not_reported",
            }
        run_metadata[run] = {
            "files": len(run_files),
            "public_raw_files": sum("/raw/" in name for name in run_files),
            "coherence_cache_files": sum("/coherence-cache/" in name for name in run_files),
            "receipt_source_checks": checks,
            "native_regression": native,
            "coherence_execution": coherence,
        }
    members["inputs/native-regression.jsonl"] = fixture_raw
    members.update(
        {
            f"package/{name}.txt" if name.endswith(".py") else f"package/{name}": raw
            for name, raw in package_sources.items()
        }
    )
    for name, raw in support_files.items():
        if name in members or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("support evidence member name collides or escapes archive")
        members[name] = raw
    if sum(map(len, members.values())) > MAX_TOTAL_BYTES:
        raise ValueError("evidence archive exceeds conservative total size limit")
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system, info.external_attr = 3, 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, raw, compresslevel=9)
    zip_raw = bundle.getvalue()
    manifest = {
        "schema_version": 1,
        "archive": {"filename": output.name, "sha256": sha256(zip_raw), "size_bytes": len(zip_raw)},
        "members": [
            {"path": name, "sha256": sha256(raw), "size_bytes": len(raw)}
            for name, raw in sorted(members.items())
        ],
        "runs": run_metadata,
        "package_source_commit": package_commit,
        "runtime_current": _json(support_files["runtime-current/archive-selection.json"])
        if "runtime-current/archive-selection.json" in support_files
        else {"status": "not_included"},
        "tools_scope": "Current offline audit/archive tools; not the earlier model-cell executed source",
        "cost_usd": None,
        "sealed_evaluation": "not_run",
        "official_rank": None,
        "leaderboard_status": "not_submitted",
        "scope": "Byte archival and saved native regression comparison; metrics require independent audit",
        "zip_metadata": {
            "timestamp": "1980-01-01T00:00:00",
            "member_order": "sorted",
            "compression": "deflate-9",
            "mode": "0644",
        },
    }
    for path, before in snapshots:
        if _read_tree(path) != before:
            raise ValueError(f"evidence changed while archiving: {path}")
    if Path(fixture).read_bytes() != fixture_raw:
        raise ValueError("native input fixture changed while archiving")
    if support_from_disk and default_support_files(artifact_root) != support_files:
        raise ValueError("support files changed while archiving")
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
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "artifacts/jevbench")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "docs/validation/2026-10-08/jevbench/raw-evidence.zip"
    )
    parser.add_argument("--run", action="append", choices=RUNS)
    args = parser.parse_args()
    manifest = archive_campaign(args.artifact_root, args.output, runs=args.run or RUNS)
    print(
        json.dumps({"archive": manifest["archive"], "members": len(manifest["members"])}, indent=2)
    )


if __name__ == "__main__":
    main()
