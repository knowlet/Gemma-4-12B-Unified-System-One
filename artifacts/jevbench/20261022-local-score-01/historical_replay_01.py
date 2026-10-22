"""Bounded official CPU rescore of two historical A100 JevBench profiles."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import resource
import signal
import socket
import stat
import subprocess
import sys
import time
import zipfile

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[2]
OUTPUT = HERE / "historical-replay-01"
EXTRACTED = HERE / "historical-archive-01"
ARCHIVE = ROOT / "docs/validation/2026-10-08/jevbench/raw-evidence.zip"
MANIFEST = ARCHIVE.with_name("raw-evidence.manifest.json")
ARCHIVE_SHA = "b85b4471d655a05e5fc682d2d22c157ae86af9b42a05a926e2bd78655223f53b"
MANIFEST_SHA = "5cea21069490c69b59f8b9f12fd76062aeb2ed3b395a0ba3fe6a86d5d8d7b44d"
PROFILES = {
    "s1-bf16": "20261008-public-01/s1-bf16",
    "s1-independent": "20261008-public-02/s1-independent",
}
HEAVEN = HERE / "upstream/heaven"
COHERENCE = HERE / "upstream/coherence"


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def reference(path):
    raw = path.read_bytes()
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": digest(raw), "size_bytes": len(raw)}


def write(path, data):
    raw = (json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return reference(path)


def deadline(_signal, _frame):
    raise TimeoutError("Bounded historical CPU replay deadline")


def deny_network(*args, **kwargs):
    raise RuntimeError("Historical CPU replay cannot contact endpoints")


def main():
    started = time.monotonic()
    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        require(os.environ.get(key) == "1", "One CPU thread environment required: " + key)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "GPU must be invisible")
    resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (290, 290))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024**2, 64 * 1024**2))
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(295)
    socket.create_connection = deny_network
    require(not OUTPUT.exists() and not EXTRACTED.exists(), "Fresh replay and extraction directories required")
    OUTPUT.mkdir()
    sources_before = {str(p.relative_to(ROOT)): reference(p) for p in (
        Path(__file__).absolute(), ROOT / "scripts/summarize_jevbench_campaign.py",
        ROOT / "scripts/run_jevbench_public.py", ROOT / "configs/benchmarks/jevbench-public.json",
    )}
    receipt = {
        "schema_version": 1, "kind": "official_historical_cpu_jevbench_rescore",
        "status": "running", "started_at_utc": started_utc,
        "historical_inference_date": "2026-10-08", "replay_date": "2026-10-22",
        "model_calls": 0, "model_loaded": False, "training_executed": False,
        "fresh_gx10_model_inference": False,
        "limits": {"address_space_bytes": 1024**3, "cpu_seconds": 290, "wall_alarm_seconds": 295,
                   "per_file_bytes": 64 * 1024**2, "blas_threads": 1, "cuda_visible_devices": ""},
        "sources_before": sources_before,
        "argv": sys.argv, "python": sys.version,
        "scope": "Today's official CPU scoring of original historical A100 answers. No new model inference, serving speed measurement, sealed evaluation or leaderboard submission.",
    }
    primary = None
    try:
        raw, raw_manifest = ARCHIVE.read_bytes(), MANIFEST.read_bytes()
        require(digest(raw) == ARCHIVE_SHA and len(raw) == 10932994, "Pinned archive differs")
        require(digest(raw_manifest) == MANIFEST_SHA and len(raw_manifest) == 2127644, "Pinned archive manifest differs")
        inventory = json.loads(raw_manifest)
        EXTRACTED.mkdir()
        with zipfile.ZipFile(ARCHIVE) as bundle:
            require(bundle.namelist() == [row["path"] for row in inventory["members"]], "Exact ordered archive set required")
            for row in inventory["members"]:
                label = PurePosixPath(row["path"])
                require(not label.is_absolute() and ".." not in label.parts and "\\" not in row["path"], "Confined archive member required")
                info = bundle.getinfo(row["path"])
                mode = info.external_attr >> 16
                require(not stat.S_ISLNK(mode) and not info.is_dir(), "Ordinary non-symlink member required")
                payload = bundle.read(row["path"])
                require(len(payload) == row["size_bytes"] and digest(payload) == row["sha256"], "Archive member integrity differs")
                target = EXTRACTED.joinpath(*label.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(payload)
            require(bundle.testzip() is None, "Archive CRC differs")
        receipt["archive_integrity"] = {
            "archive": reference(ARCHIVE), "manifest": reference(MANIFEST),
            "members": len(inventory["members"]),
            "uncompressed_size_bytes": sum(row["size_bytes"] for row in inventory["members"]),
            "checks": ["exact ordered member set", "every member SHA256", "every member size", "ZIP CRC", "confined ordinary extraction"],
            "status": "PASS", "raw_evidence_scope": "Extracted only locally; not selected for Git export",
        }
        for path, wanted in ((HEAVEN, "bb05a335bc809e61b20c0f745d25499a82b326fc"),
                             (COHERENCE, "e18733694623aa93058e279c3534f8b8e2edefa3")):
            actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, timeout=15).strip()
            require(actual == wanted, "Official source revision differs")
        print(json.dumps({"stage": "archive_verified_start_official_scoring", "members": len(inventory["members"])}), flush=True)
        sys.path.insert(0, str(ROOT / "scripts"))
        from summarize_jevbench_campaign import summarize_campaign
        audit = summarize_campaign(
            EXTRACTED, HEAVEN, model_names=tuple(PROFILES),
            model_roots={name: EXTRACTED / path for name, path in PROFILES.items()},
            coherence_root=COHERENCE, bootstrap_samples=10000,
        )
        require(audit["status"] == "complete_public_audit", "Official public scoring incomplete")
        require(audit["coherence_audit"] == "complete_raw_replay_audit", "Official raw coherence replay incomplete")
        receipt["official_audit_artifact"] = write(OUTPUT / "official-audit.json", audit)
        summaries = {}
        for name, row in audit["models"].items():
            public, coherence, identity = row["public"], row["coherence"], row["identity"]
            original = json.loads((EXTRACTED / PROFILES[name] / "coherence-mini.json").read_bytes())
            require(public["n_planned"] == public["n_attempted"] == public["n_valid"] == 231, "Full public population required")
            require(coherence["replay"]["raw_cache_files"] == 1248, "Complete native coherence cache required")
            require(coherence["n_cases"] == 240 and coherence["n_checks"] == 50, "Complete mini cases and checks required")
            require(sum(original["result"]["planned_tests"].values()) == 1200, "Full fixed planned test population required")
            summaries[name] = {
                "historical_source_run": row["source_campaign"],
                "historical_inference_date": "2026-10-08", "historical_gpu": identity["gpu"],
                "identity": {key: identity.get(key) for key in ("model_id", "revision", "precision", "variant", "execution", "versions")},
                "checkpoint_inventory_sha256": identity["checkpoint_files"]["sha256"],
                "original_receipt_sha256": row["receipt_sha256"],
                "original_recorded_source_hashes": identity.get("source_files"),
                "original_source_proof_scope": "Recorded historical source hashes retained; run01 does not record a full campaign/package execution inventory. Archive bytes do not strengthen unrecorded execution provenance.",
                "public": {key: public[key] for key in ("n_planned", "n_attempted", "n_valid", "n_correct", "accuracy", "macro_accuracy", "brier_mean", "ece", "ordinal_mae", "schema_validity", "schema_validity_strict", "n_renormalized", "by_type", "verified_files", "manifest_sha256")},
                "coherence": {
                    **{key: coherence[key] for key in ("status", "overall", "ci95", "dimensions", "n_cases", "n_checks", "errors", "incomplete", "settings", "sha256")},
                    "planned_tests": 1200, "raw_cached_requests": 1248,
                    "outcome_rows_exactly_replayed": len(original["result"]["outcomes"]),
                    "raw_cache_inventory_sha256": coherence["replay"]["raw_cache_inventory_sha256"],
                    "ordered_request_fingerprints_sha256": coherence["replay"]["ordered_request_fingerprints_sha256"],
                    "coherence_source": {key: coherence["replay"]["source"][key] for key in ("repository", "revision", "inventory_sha256")},
                    "all_deterministic_outcome_fields_exactly_compared": True,
                    "scores_float_aggregate_tolerance": {"rel": 1e-12, "abs": 1e-12},
                },
            }
        receipt["profiles"] = summaries
        receipt["paired_public_accuracy"] = audit["paired_accuracy"]
        receipt["status"] = "PASS_OFFICIAL_PUBLIC_AND_COMPLETE_RAW_COHERENCE_REPLAY"
        require(not any(name == "torch" or name.startswith("torch.") or name == "transformers" or name.startswith("transformers.") for name in sys.modules), "ML framework imported into CPU replay")
        receipt["ml_frameworks_imported"] = False
        lines = ["# Historical JevBench official CPU rescore — 2026-10-22", "",
                 "Original inference: 2026-10-08, A100. These are newly recomputed scores from preserved historical answers, not new GX10 model inference.", "",
                 "| Historical profile | Public correct / planned | Accuracy | Coherence mini | 95% CI |",
                 "| --- | ---: | ---: | ---: | --- |"]
        for name, row in summaries.items():
            p, c = row["public"], row["coherence"]
            lines.append(f"| {name} | {p['n_correct']}/{p['n_planned']} | {p['accuracy']*100:.4f}% | {c['overall']*100:.4f} | {c['ci95'][0]*100:.4f}–{c['ci95'][1]*100:.4f} |")
        lines += ["", "Each coherence profile replayed all 1,248 original requests, 240 cases, 50 checks and 1,200 planned tests against the pinned official evaluator. Exact deterministic outcomes and original probabilities were compared; aggregate float tolerance remains 1e-12. Public scoring used all 231 tasks and the pinned official scorer.", "", "Public accuracy, five-dimension coherence and browser completion remain separate measurements. No calibration fit, sealed evaluation, official rank, cost estimate or current serving latency is claimed. Raw answers stay local and are excluded from Git publication.", ""]
        with (OUTPUT / "README.md").open("x") as stream:
            stream.write("\n".join(lines))
        receipt["markdown"] = reference(OUTPUT / "README.md")
    except BaseException as exc:
        primary = exc
        receipt["status"] = "FAILED"
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        signal.alarm(0)
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["finished_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        receipt["sources_after"] = {label: reference(ROOT / label) for label in sources_before}
        receipt["sources_unchanged"] = receipt["sources_after"] == sources_before
        receipt["usage"] = {name: {"user_seconds": r.ru_utime, "system_seconds": r.ru_stime, "max_rss_kib": r.ru_maxrss}
                            for name, r in (("self", resource.getrusage(resource.RUSAGE_SELF)), ("children", resource.getrusage(resource.RUSAGE_CHILDREN)))}
        receipt_ref = write(OUTPUT / "rescore-receipt.json", receipt)
        print(json.dumps({"status": receipt["status"], "receipt": receipt_ref, "seconds": receipt["elapsed_seconds"]}), flush=True)
    if primary is not None:
        raise primary
    require(receipt["sources_unchanged"], "Replay sources changed during computation")


if __name__ == "__main__":
    main()
