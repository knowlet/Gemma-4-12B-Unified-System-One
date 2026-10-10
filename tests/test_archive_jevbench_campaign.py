"""Portable evidence archives retain exact bytes and reject inconsistent inputs."""

import copy
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/archive_jevbench_campaign.py"
spec = importlib.util.spec_from_file_location("archive_jevbench_campaign", SCRIPT)
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


@pytest.fixture
def evidence(tmp_path):
    root = tmp_path / "artifacts"
    source = root / "executed-public01"
    source.mkdir(parents=True)
    hashes = {}
    for name in ("jevbench_campaign.py", "run_jevbench_public.py", "unified.py"):
        raw = f"# exact {name}\n".encode()
        (source / name).write_bytes(raw)
        hashes[name] = archive.sha256(raw)
    cell = root / archive.RUNS[0] / "s1-bf16"
    (cell / "public231/raw").mkdir(parents=True)
    (cell / "coherence-cache").mkdir()
    (cell / "public231/raw/request.json").write_bytes(b'{"request": "original", "error": null}\n')
    (cell / "coherence-cache/response.json").write_bytes(b'{"response": "unchanged"}\n')
    receipt = {
        "identity": {
            "source_files": {
                "campaign": hashes["jevbench_campaign.py"],
                "public_runner": hashes["run_jevbench_public.py"],
                "unified.py": hashes["unified.py"],
            }
        }
    }
    (cell / "receipt.json").write_text(json.dumps(receipt))
    fixture = tmp_path / "native.jsonl"
    fixture.write_text("{}\n")
    return root, cell, fixture


def build(evidence, output):
    root, _, fixture = evidence
    return archive.archive_campaign(
        root,
        output,
        runs=[archive.RUNS[0]],
        source_pins={},
        fixture=fixture,
        package_sources={"src/s1/unified.py": (root / "executed-public01/unified.py").read_bytes()},
        support_files={},
    )


def test_archive_is_deterministic_complete_and_never_overwrites(evidence, tmp_path):
    outputs = [tmp_path / "first/raw-evidence.zip", tmp_path / "second/raw-evidence.zip"]
    manifests = [build(evidence, output) for output in outputs]
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    assert manifests[0] == manifests[1]
    manifest = manifests[0]
    with zipfile.ZipFile(outputs[0]) as bundled:
        assert bundled.namelist() == sorted(bundled.namelist())
        assert bundled.namelist() == [member["path"] for member in manifest["members"]]
        for member in manifest["members"]:
            raw = bundled.read(member["path"])
            assert archive.sha256(raw) == member["sha256"]
            assert len(raw) == member["size_bytes"]
            assert bundled.getinfo(member["path"]).date_time == (1980, 1, 1, 0, 0, 0)
        assert bundled.read("executed-public01/unified.py.txt") == b"# exact unified.py\n"
        assert any("coherence-cache/response.json" in name for name in bundled.namelist())
    before = outputs[0].read_bytes()
    with pytest.raises(FileExistsError):
        build(evidence, outputs[0])
    assert outputs[0].read_bytes() == before
    assert len(list((evidence[0] / archive.RUNS[0]).rglob("*.json"))) == 3


@pytest.mark.parametrize(
    "invalid", ["receipt", "core_receipt", "missing_receipt", "weights", "symlink"]
)
def test_archive_rejects_inconsistent_or_unsafe_evidence(evidence, tmp_path, invalid):
    _, cell, _ = evidence
    if invalid in ("receipt", "core_receipt"):
        receipt = json.loads((cell / "receipt.json").read_text())
        receipt["identity"]["source_files"][
            "campaign" if invalid == "receipt" else "unified.py"
        ] = "0" * 64
        (cell / "receipt.json").write_text(json.dumps(receipt))
    elif invalid == "missing_receipt":
        (cell / "receipt.json").unlink()
    elif invalid == "weights":
        (cell / "model.safetensors").write_bytes(b"model payload")
    else:
        (cell / "linked.json").symlink_to(cell / "receipt.json")
    output = tmp_path / "out.zip"
    with pytest.raises(ValueError):
        build(evidence, output)
    assert not output.exists()
    assert not output.with_suffix(".manifest.json").exists()


def test_old_receipt_missing_execution_hashes_is_explicitly_partial(evidence, tmp_path):
    _, cell, _ = evidence
    receipt = json.loads((cell / "receipt.json").read_text())
    recorded = receipt["identity"]["source_files"]
    recorded.pop("campaign")
    recorded.pop("unified.py")
    (cell / "receipt.json").write_text(json.dumps(receipt))
    manifest = build(evidence, tmp_path / "partial-source-proof.zip")
    checked = manifest["runs"][archive.RUNS[0]]["receipt_source_checks"]["s1-bf16"]
    assert checked == {
        "jevbench_campaign.py": "not_recorded_in_receipt",
        "run_jevbench_public.py": "verified",
        "src/s1/unified.py": "not_recorded_in_receipt",
    }
    assert manifest["official_rank"] is None
    assert manifest["cost_usd"] is None


def test_native_comparison_accounts_for_every_probability_and_modality():
    fixture, results = [], []
    for index in range(8):
        modalities = [[], ["image"], ["audio"], ["image", "audio"]][index % 4]
        fixture.append(
            {
                "id": str(index),
                "request": {
                    "questions": [
                        {"id": "answer", "type": "choice", "criteria": {"a": "A", "b": "B"}}
                    ],
                    "media": [{"type": kind} for kind in modalities],
                },
            }
        )
        variants = {}
        for variant in ("eager", "cached", "compiled"):
            drift = 0.01 if variant == "compiled" else 0
            variants[variant] = {
                "model": "same",
                "revision": "same",
                "temperature": 2,
                "passes": 1,
                "answers": {
                    "answer": {
                        "type": "choice",
                        "choice": "a",
                        "probabilities": {"a": 0.7 + drift, "b": 0.3 - drift},
                    }
                },
            }
        results.append({"case_id": str(index), "modalities": modalities, "variants": variants})
    fixture_raw = b"\n".join(json.dumps(case).encode() for case in fixture)
    audit = archive.native_regression_audit(json.dumps(results).encode(), fixture_raw)
    for variant in ("cached", "compiled"):
        row = audit["comparisons"][variant]
        assert row["questions"] == row["argmax_matches"] == 8
        assert row["probabilities"] == 16
        assert set(row["by_modality"]) == {"text", "image", "audio", "mixed"}
    assert audit["comparisons"]["cached"]["max_abs_probability_drift"] == 0
    assert audit["comparisons"]["compiled"]["max_abs_probability_drift"] == pytest.approx(0.01)
    with pytest.raises(ValueError, match="population"):
        archive.native_regression_audit(json.dumps(results[:-1]).encode(), fixture_raw)


@pytest.fixture
def independent_native():
    fixture_raw = (SCRIPT.parents[1] / "examples/benchmarks/mps.jsonl").read_bytes()
    records = []
    for case in (json.loads(line) for line in fixture_raw.splitlines()):
        answers = {}
        for question in case["request"]["questions"]:
            kind, criteria = question["type"], question["criteria"]
            labels = (
                ["false", "true"]
                if kind == "noul"
                else list(criteria)
                if isinstance(criteria, dict)
                else [str(i) for i in range(len(criteria))]
            )
            probabilities = {
                label: 0.7 if i == 0 else 0.3 / (len(labels) - 1) for i, label in enumerate(labels)
            }
            answer = {"type": kind, "probabilities": probabilities}
            answer["choice" if kind == "choice" else "level" if kind == "score" else "noul"] = (
                labels[0] if kind != "noul" else probabilities["true"]
            )
            answers[question["id"]] = answer
        count = len(answers)
        records.append(
            {
                "case_id": case["id"],
                "modalities": [m["type"] for m in case["request"]["media"]],
                "variants": {
                    "sequential": {"answers": answers},
                    "independent": {
                        "answers": copy.deepcopy(answers),
                        "execution": {
                            "forward_calls": 1,
                            "batch_sizes": [count],
                            "sequence_tokens": [100] * count,
                            "scope": "whole_adapter_batch",
                            "readout": "candidate",
                            "independent_questions": True,
                        },
                    },
                },
            }
        )
    return fixture_raw, records


def test_independent_native_audit_keeps_all_29_questions_and_shape_drift(independent_native):
    fixture_raw, records = independent_native
    probabilities = records[4]["variants"]["independent"]["answers"]["team"]["probabilities"]
    probabilities["billing"] += 0.005
    probabilities["technical"] -= 0.005
    audit = archive.independent_native_regression_audit(json.dumps(records).encode(), fixture_raw)
    row = audit["comparisons"]["independent"]
    assert audit["baseline_variant"] == "sequential"
    assert row["questions"] == row["argmax_matches"] == 29
    assert row["probabilities"] == 59
    assert set(row["by_modality"]) == {"text", "image", "audio", "mixed"}
    assert row["max_abs_probability_drift"] == pytest.approx(0.005)
    assert row["by_modality"]["audio"]["max_abs_probability_drift"] == pytest.approx(0.005)
    assert "does not assert equivalence to causal" in audit["comparison_scope"]


def test_independent_native_label_changes_are_reported_without_posthoc_tolerance(
    independent_native,
):
    fixture_raw, records = independent_native
    answer = records[-1]["variants"]["independent"]["answers"]["q15"]
    answer["probabilities"] = {"billing": 0.2, "technical": 0.8}
    answer["choice"] = "technical"
    audit = archive.independent_native_regression_audit(json.dumps(records).encode(), fixture_raw)
    row = audit["comparisons"]["independent"]
    assert row["argmax_matches"] == 28
    assert row["by_case"][-1]["labels_changed"] == [
        {"question_id": "q15", "baseline": "billing", "variant": "technical"}
    ]
    assert audit["tolerance_gate"].startswith("not_predeclared")


@pytest.mark.parametrize(
    "corruption", ["independent", "batch_population", "probability", "modality"]
)
def test_independent_native_rejects_incomplete_or_misdeclared_execution(
    independent_native, corruption
):
    fixture_raw, records = independent_native
    record = records[0]
    response = record["variants"]["independent"]
    if corruption == "independent":
        response["execution"]["independent_questions"] = False
    elif corruption == "batch_population":
        response["execution"]["batch_sizes"] = [1]
    elif corruption == "probability":
        response["answers"]["team"]["probabilities"].pop("technical")
    else:
        record["modalities"] = ["image"]
    with pytest.raises(ValueError):
        archive.independent_native_regression_audit(json.dumps(records).encode(), fixture_raw)


def test_supplemental_receipt_can_decline_coherence_explicitly(evidence, tmp_path):
    _, cell, _ = evidence
    (cell / "coherence-cache/response.json").unlink()
    receipt = json.loads((cell / "receipt.json").read_text())
    receipt["identity"]["coherence_requested"] = False
    (cell / "receipt.json").write_text(json.dumps(receipt))
    manifest = build(evidence, tmp_path / "supplemental.zip")
    scope = manifest["runs"][archive.RUNS[0]]["coherence_execution"]["s1-bf16"]
    assert scope == {
        "requested": False,
        "report_present": False,
        "cache_files": 0,
        "status": "not_requested",
    }


@pytest.fixture
def current_runtime(tmp_path, monkeypatch):
    root = tmp_path / "runtime-current"
    root.mkdir()
    source = {name: f"# new wrapper {name}\n".encode() for name in archive.RUNTIME_SOURCE_MEMBERS}
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as wheel:
        for name, raw in source.items():
            wheel.writestr(name, raw)
    raw = payload.getvalue()
    wheel_name = "current-wrapper.whl"
    (root / wheel_name).write_bytes(raw)
    manifest = {
        "source_commit": "new-wrapper",
        "files": {
            wheel_name: {"sha256": archive.sha256(raw), "size_bytes": len(raw)},
            "redundant-docs.tar.gz": {"sha256": "0" * 64, "size_bytes": 18042082},
        },
        "wheel_source_checked": sorted(source),
        "gpu_http_smoke": "not_run",
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(archive, "RUNTIME_COMMIT", "new-wrapper")
    monkeypatch.setattr(archive, "RUNTIME_WHEEL_SHA256", archive.sha256(raw))
    monkeypatch.setattr(archive, "RUNTIME_WHEEL_SIZE", len(raw))
    monkeypatch.setattr(
        archive, "_git_bytes", lambda commit, path: source[path.removeprefix("src/")]
    )
    return root


def test_optional_current_wheel_is_separate_from_older_cloud_source(current_runtime):
    support = archive.runtime_current_support_files(current_runtime)
    assert set(support) == {
        "runtime-current/manifest.json",
        "runtime-current/current-wrapper.whl",
        "runtime-current/archive-selection.json",
    }
    assert (
        support["runtime-current/manifest.json"] == (current_runtime / "manifest.json").read_bytes()
    )
    selection = json.loads(support["runtime-current/archive-selection.json"])
    assert selection["source_commit"] == "new-wrapper"
    assert "not older model-cell" in selection["scope"]
    assert set(selection["not_archived"]) == {"redundant-docs.tar.gz"}
    assert selection["gpu_http_smoke"] == "not_run"


@pytest.mark.parametrize("tamper", ["wheel", "git_reference", "commit"])
def test_current_runtime_bundle_requires_pinned_bytes_and_git_source(
    current_runtime, monkeypatch, tamper
):
    if tamper == "wheel":
        (current_runtime / "current-wrapper.whl").write_bytes(b"changed wheel")
    elif tamper == "git_reference":
        monkeypatch.setattr(archive, "_git_bytes", lambda *args: b"different source")
    else:
        manifest = json.loads((current_runtime / "manifest.json").read_text())
        manifest["source_commit"] = "old-cloud-source"
        (current_runtime / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        archive.runtime_current_support_files(current_runtime)


def test_current_runtime_bundle_is_optional(tmp_path):
    assert archive.runtime_current_support_files(tmp_path / "absent") == {}
