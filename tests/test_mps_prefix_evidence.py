"""Replay the real MPS experiment's rejection from preserved raw answers."""

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

from s1.contracts import DecisionRequest

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/validation/2026-10-28/mps-prefix"
SPEC = importlib.util.spec_from_file_location(
    "prefix_evidence_replay", ROOT / "scripts/benchmark_mps_prefix.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_all_raw_mps_samples_replay_the_predeclared_rejection():
    report = json.loads((EVIDENCE / "report.json").read_text())
    archive_path = EVIDENCE / report["archive"]["path"]
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == report["archive"]["sha256"]
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {row["path"] for row in report["archive"]["members"]}
        for row in report["archive"]["members"]:
            raw = archive.read(row["path"])
            assert len(raw) == row["size_bytes"]
            assert hashlib.sha256(raw).hexdigest() == row["sha256"]
        raw = archive.read("report.json")
        assert hashlib.sha256(raw).hexdigest() == report["source_report_sha256"]
        original = json.loads(raw)
        for name, expected in original["source_sha256"].items():
            assert hashlib.sha256(archive.read("sources/" + name)).hexdigest() == expected

    assert original["max_probability_delta"] == report["max_probability_delta"] == 0.01
    assert original["dataset"]["labels_used"] is False
    recomputed = []
    for case in original["cases"]:
        summary = RUNNER.summarize_case(
            DecisionRequest.model_validate(case["request"]), case["samples"], max_delta=0.01
        )
        assert summary == case["summary"]
        assert summary["samples_per_profile"] == 4
        assert summary["baseline_repeat_max_delta"] == 0
        recomputed.append({"id": case["id"], **summary})
    assert len(recomputed) == 11
    assert recomputed == report["cases"]
    assert {row["id"] for row in recomputed if not row["parity_gate_passed"]} == {
        "browser-dev-design-free-2",
        "browser-dev-design-free-3",
    }
    assert all(row["all_argmax_agree"] for row in recomputed)
    assert not report["parity_gate_passed"] and not original["parity_gate_passed"]
    assert report["status"] == "completed_candidate_rejected"
    assert not report["defaults_changed"] and not original["defaults_changed"]
