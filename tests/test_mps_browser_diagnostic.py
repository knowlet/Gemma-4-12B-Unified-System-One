"""Keep frozen results immutable while recording their oracle interpretation limit."""

import hashlib
import importlib.util
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/validation/2026-10-28/mps-browser"
ARCHIVE_SHA256 = "708907f750c37dee5f988e03be65c998c708b9416416eb974c30d31c2dd838c6"
SPEC = importlib.util.spec_from_file_location(
    "oracle_reader", ROOT / "scripts/ultrafast_browser.py"
)
READER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(READER)


def test_frozen_scores_and_archive_are_not_rewritten_by_diagnostic():
    report = json.loads((EVIDENCE / "report.json").read_text())
    diagnostic = json.loads((EVIDENCE / "oracle-diagnostic.json").read_text())
    raw = (EVIDENCE / "browser-evidence.zip").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == ARCHIVE_SHA256
    assert report["archive"]["sha256"] == diagnostic["archive_sha256"] == ARCHIVE_SHA256
    assert report["summary"]["causal_multislot"]["verified_successes"] == 15
    assert report["summary"]["independent"]["verified_successes"] == 18
    assert not diagnostic["classification"]["natural_language_task_failure_proven"]
    assert not report["diagnostic"]["task_success_advantage_proven"]


def test_all_three_frozen_rejections_are_only_empty_submit_count_format():
    diagnostic = json.loads((EVIDENCE / "oracle-diagnostic.json").read_text())
    comparable = []
    with zipfile.ZipFile(EVIDENCE / "browser-evidence.zip") as archive:
        for entry in diagnostic["repeats"]:
            root = entry["member_root"]
            task = json.loads(archive.read(root + "task.json"))
            final = json.loads(archive.read(root + "final-state.json"))
            summary = json.loads(archive.read(root + "summary.json"))
            observed = READER.verify_final_state(task["expected"], final)
            failed = {
                key: value for key, value in observed["checks"].items() if not value["passed"]
            }
            assert set(failed) == {"count"}
            assert failed["count"]["expected"] == "0 places"
            assert failed["count"]["actual"] == "0 places in "
            assert not observed["verified_success"] and not summary["verified_success"]
            assert summary["failure_class"] == entry["frozen_failure_class"] == "false_done"
            records = json.loads(archive.read(root + "decision-transport.json"))
            comparable.append(
                [
                    (record["body"], record["mapped_request"], record["raw_response"]["answers"])
                    for record in records
                ]
            )
    assert len(comparable) == 3
    assert comparable[0] == comparable[1] == comparable[2]
