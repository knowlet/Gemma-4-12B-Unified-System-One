"""Research evidence archives are portable without implying GPU execution."""

import copy
import importlib.util
import io
import json
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/archive_breakthrough.py"
spec = importlib.util.spec_from_file_location("archive_breakthrough", SCRIPT)
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


def encoded(value):
    return json.dumps(value, sort_keys=True).encode() + b"\n"


def save(root, name, raw):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


@pytest.fixture
def evidence(tmp_path):
    root = tmp_path / "research"
    synthetic = {
        "train.jsonl": b'{"split":"train"}\n',
        "calibration.jsonl": b'{"split":"calibration"}\n',
        "test.jsonl": b'{"split":"test"}\n',
        "provenance.json": encoded({"source": "finite enumeration"}),
    }
    synthetic_manifest = encoded(
        {
            "datasets": {
                name: {"file_sha256": archive.sha256(raw)}
                for name, raw in synthetic.items()
                if name.endswith(".jsonl")
            }
        }
    )
    for name, raw in {**synthetic, "manifest.json": synthetic_manifest}.items():
        save(root, "data/" + name, raw)
    release = {
        "train.jsonl": b"original train\n",
        "media-test.jsonl": b"native image and audio bytes\n",
        "ATTRIBUTION.md": b"Upstream attribution\n",
    }
    release_manifest = encoded(
        {"files": {name: archive.sha256(raw) for name, raw in release.items()}}
    )
    support = {
        f"artifacts/release/datasets/{name}": raw
        for name, raw in {**release, "manifest.json": release_manifest}.items()
    }
    support["examples/benchmarks/mps.jsonl"] = b"native input\n"
    support["configs/benchmarks/jevbench-public.json"] = encoded({"revision": "fixed"})
    support["licenses/upstream-LICENSE"] = b"Preserved upstream license\n"
    support["tools/analyze_breakthrough.py.txt"] = b"# current offline analyzer source\n"
    support["publication/bf16-artifact-manifest.json"] = encoded(
        {"weight_sha256": "1" * 64, "scope": "metadata only"}
    )
    protocol = {
        "stages": [
            {"id": "ablate", "profiles": [{"id": "released-current"}]},
            {"id": "train", "profiles": [{"id": "released-user-head"}]},
        ],
        "data": {
            "synthetic": {
                "manifest_sha256": archive.sha256(synthetic_manifest),
                "files": {name: archive.sha256(raw) for name, raw in synthetic.items()},
            },
            "release_regression": {
                "directory": "artifacts/release/datasets",
                "manifest_sha256": archive.sha256(release_manifest),
                "evaluation_files": {
                    "media-test.jsonl": {"file_sha256": archive.sha256(release["media-test.jsonl"])}
                },
            },
            "native_fixture": {
                "path": "examples/benchmarks/mps.jsonl",
                "file_sha256": archive.sha256(support["examples/benchmarks/mps.jsonl"]),
            },
            "public231": {
                "source_contract": "configs/benchmarks/jevbench-public.json",
                "source_contract_sha256": archive.sha256(
                    support["configs/benchmarks/jevbench-public.json"]
                ),
            },
        },
        "official": {
            "leaderboard_status": "not_submitted",
            "sealed_evaluation": "not_run",
            "rank": None,
        },
    }
    sources = {path: f"# frozen {path}\n".encode() for path in archive.ALIASES.values()}
    sources[archive.PROTOCOL] = encoded(protocol)
    sources["src/s1/unified.py"] = b"# frozen model source\n"
    sources["src/s1/evaluation/gemma.py"] = b"# frozen native processor helper\n"
    save(
        root,
        "execution-status.json",
        encoded(
            {
                "status": "awaiting_explicit_cloud_authorization",
                "gpu_started": False,
                "training_executed": False,
                "prospective_protocol_sha256": archive.sha256(sources[archive.PROTOCOL]),
            }
        ),
    )
    return SimpleNamespace(root=root, support=support, sources=sources, protocol=protocol)


def build(evidence, output, **kwargs):
    kwargs.setdefault("outcome_reports", {})
    return archive.archive_breakthrough(
        evidence.root,
        output,
        package_sources=evidence.sources,
        support_files=evidence.support,
        **kwargs,
    )


def test_current_outcome_reports_are_optional_and_separate_from_frozen_source(evidence, tmp_path):
    report_directory = tmp_path / "current-docs"
    assert archive.default_outcome_reports(report_directory) == {}
    save(report_directory, "README.md", b"Current execution status\n")
    save(report_directory, "results.md", b"Current audited results\n")
    evidence.sources["README.md"] = b"Historical frozen status\n"
    reports = archive.default_outcome_reports(report_directory)
    output = tmp_path / "evidence.zip"
    manifest = build(evidence, output, outcome_reports=reports)
    index = manifest["outcome_reports"]
    assert "not executed or frozen" in index["scope"]
    assert [record["path"] for record in index["members"]] == [
        "reports/README.md",
        "reports/results.md",
    ]
    assert index["members"][0]["sha256"] == archive.sha256(reports["README.md"])
    assert index["members"][0]["size_bytes"] == len(reports["README.md"])
    with zipfile.ZipFile(output) as zipped:
        assert zipped.read("reports/README.md") == reports["README.md"]
        assert zipped.read("reports/results.md") == reports["results.md"]
        assert zipped.read("preflight/README.md") == b"Historical frozen status\n"
    assert archive.verify_archive(output)["verified"] is True
    manifest_path = output.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["outcome_reports"]["members"][0]["sha256"] = "0" * 64
    manifest_path.write_bytes(encoded(manifest))
    with pytest.raises(ValueError, match="outcome report manifest index"):
        archive.verify_archive(output)


@pytest.mark.parametrize("kind", ["symlink", "directory"])
def test_current_outcome_report_collector_requires_regular_files(tmp_path, kind):
    report_directory = tmp_path / "reports"
    report_directory.mkdir()
    if kind == "symlink":
        target = tmp_path / "real-report.md"
        target.write_bytes(b"report")
        (report_directory / "README.md").symlink_to(target)
    else:
        (report_directory / "README.md").mkdir()
    with pytest.raises(ValueError, match="regular outcome report"):
        archive.default_outcome_reports(report_directory)


def test_changing_current_reports_prevents_publication(evidence, tmp_path, monkeypatch):
    captures = iter([{"README.md": b"initial report"}, {"README.md": b"updated report"}])
    monkeypatch.setattr(archive, "default_outcome_reports", lambda: next(captures))
    output = tmp_path / "evidence.zip"
    with pytest.raises(ValueError, match="outcome reports changed"):
        build(evidence, output, outcome_reports=None)
    assert not output.exists()


def executed(evidence, *, stage="train", status="completed"):
    prefix = "runs/trial01/" + stage + "/"
    aliases = {
        **archive.ALIASES,
        **{
            name.removeprefix("src/"): name
            for name in evidence.sources
            if name.startswith("src/s1/")
        },
    }
    hashes = {}
    for key, name in aliases.items():
        raw = evidence.sources[name]
        hashes[key] = archive.sha256(raw)
        save(evidence.root, prefix + "sources/" + key + ".txt", raw)
    name = next(
        stage_spec["profiles"][0]["id"]
        for stage_spec in evidence.protocol["stages"]
        if stage_spec["id"] == stage
    )
    receipt = {
        "run": "trial01",
        "stage": stage,
        "status": status,
        "gpu": "recorded test board",
        "source_files": hashes,
        "prospective_protocol": evidence.protocol,
        "profiles": [{"name": name, "identity": {}}],
    }
    save(evidence.root, prefix + "receipt.json", encoded(receipt))
    save(
        evidence.root,
        prefix + name + "/receipt.json",
        encoded({"identity": {"variant": name, "source_files": hashes}}),
    )
    save(
        evidence.root,
        prefix + name + "/public231/raw/request.json",
        encoded({"request": "exact input", "response": "exact output", "error": None}),
    )
    save(
        evidence.root,
        prefix + name + "/coherence-cache/response.json",
        encoded({"answer": "raw answer"}),
    )
    (evidence.root / "execution-status.json").unlink()
    return prefix, receipt


def test_deterministic_no_gpu_archive_preserves_data_sources_and_status(evidence, tmp_path):
    outputs = [tmp_path / "a/evidence.zip", tmp_path / "b/evidence.zip"]
    manifests = [build(evidence, output) for output in outputs]
    assert outputs[0].read_bytes() == outputs[1].read_bytes()
    assert manifests[0] == manifests[1]
    manifest = manifests[0]
    assert manifest["execution"]["state"] == "no_gpu"
    assert manifest["execution"]["stage_receipts"] == {}
    assert manifest["execution"]["local_status_record"]["training_executed"] is False
    assert manifest["official"]["rank"] is None
    with zipfile.ZipFile(outputs[0]) as zipped:
        assert zipped.namelist() == sorted(zipped.namelist())
        assert zipped.read("research/data/train.jsonl") == b'{"split":"train"}\n'
        assert zipped.read("preflight/src/s1/unified.py.txt") == b"# frozen model source\n"
        assert zipped.read("inputs/licenses/upstream-LICENSE") == b"Preserved upstream license\n"
        assert (
            zipped.read("inputs/tools/analyze_breakthrough.py.txt")
            == evidence.support["tools/analyze_breakthrough.py.txt"]
        )
        assert (
            zipped.read("inputs/publication/bf16-artifact-manifest.json")
            == evidence.support["publication/bf16-artifact-manifest.json"]
        )
        assert "inputs/artifacts/release/datasets/train.jsonl" in zipped.namelist()
    result = archive.verify_archive(outputs[0])
    assert result["verified"] is True
    assert result["members"] == len(manifest["members"])
    assert result["execution_state"] == "no_gpu"


def test_absent_status_and_receipts_remain_not_run(evidence, tmp_path):
    (evidence.root / "execution-status.json").unlink()
    manifest = build(evidence, tmp_path / "evidence.zip")
    assert manifest["execution"]["state"] == "not_run"
    assert manifest["execution"]["local_status_record"] is None
    assert manifest["execution"]["planned_stages"] == {
        "ablate": ["released-current"],
        "train": ["released-user-head"],
    }


def test_executed_sources_raw_cache_and_trained_binaries_are_preserved(evidence, tmp_path):
    prefix, receipt = executed(evidence)
    trained = {
        "decision-head.pt": b"trained head exact bytes",
        "train-features.pt": b"detached features exact bytes",
        "mixed-lora/adapter/adapter_model.safetensors": b"trained adapter exact bytes",
        "mixed-lora/updates.jsonl": b'{"update":1}\n',
    }
    for name, raw in trained.items():
        save(evidence.root, prefix + name, raw)
    receipt["profiles"][0]["identity"]["research_head_sha256"] = archive.sha256(
        trained["decision-head.pt"]
    )
    adapter_spec = {
        "adapter_model.safetensors": {
            "sha256": archive.sha256(trained["mixed-lora/adapter/adapter_model.safetensors"]),
            "bytes": len(trained["mixed-lora/adapter/adapter_model.safetensors"]),
        }
    }
    receipt["profiles"][0]["identity"]["research_adapter_files"] = adapter_spec
    save(evidence.root, prefix + "receipt.json", encoded(receipt))
    save(
        evidence.root, prefix + "mixed-lora/training.json", encoded({"adapter_files": adapter_spec})
    )
    manifest = build(evidence, tmp_path / "evidence.zip")
    cell = manifest["execution"]["stage_receipts"]["trial01/train"]
    assert cell["evidence_status"] == "receipt_completed"
    assert len(cell["source_checks"]) == len(archive.ALIASES) + 2
    assert set(cell["source_checks"].values()) == {"receipt_hash_and_frozen_bytes_verified"}
    assert len(cell["research_binary_checks"]) == 2
    assert manifest["execution"]["stage_receipts"]["trial01/ablate"]["evidence_status"] == "not_run"
    with zipfile.ZipFile(tmp_path / "evidence.zip") as zipped:
        for name, raw in trained.items():
            assert zipped.read("research/" + prefix + name) == raw
        assert (
            zipped.read("research/" + prefix + "sources/s1/unified.py.txt")
            == evidence.sources["src/s1/unified.py"]
        )
        assert any("coherence-cache/response.json" in name for name in zipped.namelist())
        assert any("public231/raw/request.json" in name for name in zipped.namelist())


def recovered(evidence, *, status="completed", declared_binaries=False):
    prefix, receipt = executed(evidence, status=status)
    canonical = evidence.root / prefix
    target = canonical.with_name("train-recovered")
    canonical.rename(target)
    prefix = "runs/trial01/train-recovered/"
    research = {
        "decision-head.pt": b"exact trained head",
        "train-features.pt": b"exact detached features",
        "mixed-lora/adapter/adapter_model.safetensors": b"exact trained adapter",
        "mixed-lora/adapter/README.md": b"PEFT documentation absent from original ZIP filter\n",
    }
    for name, raw in research.items():
        save(evidence.root, prefix + name, raw)
    if declared_binaries:
        adapter_files = {
            name.removeprefix("mixed-lora/adapter/"): {
                "sha256": archive.sha256(raw),
                "bytes": len(raw),
            }
            for name, raw in research.items()
            if name.startswith("mixed-lora/adapter/")
        }
        receipt["profiles"][0]["identity"] = {
            "research_head_sha256": archive.sha256(research["decision-head.pt"]),
            "research_adapter_files": adapter_files,
        }
        save(evidence.root, prefix + "receipt.json", encoded(receipt))
        save(
            evidence.root,
            prefix + "mixed-lora/training.json",
            encoded({"adapter_files": adapter_files}),
        )
    payload = {
        path.relative_to(target).as_posix(): path.read_bytes()
        for path in sorted(target.rglob("*"))
        if path.is_file()
    }
    helper = b"# actual read-only download helper\n"
    listing = encoded({name: {"size_bytes": len(raw)} for name, raw in payload.items()})
    metadata = {
        "export_breakthrough_evidence.py.txt": helper,
        "remote-listing.json": listing,
        "original-export-failure.txt": b"Original CPU export timed out; GPU remained completed\n",
    }
    for name, raw in metadata.items():
        save(evidence.root, prefix + ".evidence-download/" + name, raw)
    transport = {
        "status": "completed",
        "run": "trial01",
        "stage": "train",
        "volume": "gemma-unified-system-one",
        "remote_prefix": "breakthrough/trial01/train",
        "gpu_receipt_status": status,
        "gpu_receipt_sha256": archive.sha256(payload["receipt.json"]),
        "source_sha256": archive.sha256(helper),
        "expected_files": len(payload),
        "expected_size_bytes": sum(map(len, payload.values())),
        "executed_sources_verified": len(receipt["source_files"]),
        "members": [
            {"path": name, "sha256": archive.sha256(raw), "size_bytes": len(raw)}
            for name, raw in payload.items()
        ],
        "metadata_members": [
            {"path": name, "sha256": archive.sha256(raw), "size_bytes": len(raw)}
            for name, raw in metadata.items()
        ],
        "original_export_failure": {
            "sha256": archive.sha256(metadata["original-export-failure.txt"]),
            "size_bytes": len(metadata["original-export-failure.txt"]),
        },
        "errors": [],
    }
    save(evidence.root, prefix + ".evidence-download/receipt.json", encoded(transport))
    return prefix, payload, metadata, transport


def test_recovered_train_preserves_all_bytes_and_discloses_original_missing_download(
    evidence, tmp_path
):
    prefix, payload, metadata, _ = recovered(evidence)
    manifest = build(evidence, tmp_path / "evidence.zip")
    execution = manifest["execution"]
    cell = execution["stage_receipts"]["trial01/train-recovered"]
    assert cell["logical_stage"] == "train"
    assert cell["evidence_status"] == "receipt_completed"
    assert len(cell["source_checks"]) == len(archive.ALIASES) + 2
    assert execution["stage_receipts"]["trial01/train"]["evidence_status"] == "not_downloaded"
    recovery = execution["recovery_transfers"]["trial01/train-recovered"]
    assert recovery["canonical_transport_status"] == "not_downloaded"
    assert recovery["member_checks"] == {"members": len(payload), "metadata_members": 3}
    assert recovery["original_cpu_export"]["status"] == "not_assessed_by_archiver"
    assert recovery["original_cpu_export"]["transport_failure_record"]["sha256"] == archive.sha256(
        metadata["original-export-failure.txt"]
    )
    with zipfile.ZipFile(tmp_path / "evidence.zip") as zipped:
        for name, raw in payload.items():
            assert zipped.read("research/" + prefix + name) == raw
        for name, raw in metadata.items():
            assert zipped.read("research/" + prefix + ".evidence-download/" + name) == raw
    assert archive.verify_archive(tmp_path / "evidence.zip")["verified"] is True


def test_canonical_and_recovered_common_bytes_are_verified_without_discarding_extras(
    evidence, tmp_path
):
    prefix, payload, _, _ = recovered(evidence)
    canonical_prefix = "runs/trial01/train/"
    for name, raw in payload.items():
        if not name.endswith("README.md"):
            save(evidence.root, canonical_prefix + name, raw)
    save(evidence.root, canonical_prefix + "original-main.log", b"original stdout exact bytes\n")
    manifest = build(evidence, tmp_path / "evidence.zip")
    recovery = manifest["execution"]["recovery_transfers"]["trial01/train-recovered"]
    assert recovery["canonical_transport_status"] == "evidence_present"
    assert recovery["canonical_common_members"] == len(payload) - 1
    assert recovery["canonical_common_bytes"] == "all_identical"
    assert recovery["canonical_only_members"] == ["original-main.log"]
    assert recovery["recovered_only_members"] == ["mixed-lora/adapter/README.md"]
    with zipfile.ZipFile(tmp_path / "evidence.zip") as zipped:
        assert (
            zipped.read("research/" + canonical_prefix + "receipt.json") == payload["receipt.json"]
        )
        assert zipped.read("research/" + prefix + "receipt.json") == payload["receipt.json"]
        assert zipped.read("research/" + canonical_prefix + "original-main.log") == (
            b"original stdout exact bytes\n"
        )


def test_original_export_filter_missing_declared_adapter_readme_is_explicitly_recovered(
    evidence, tmp_path
):
    prefix, payload, _, _ = recovered(evidence, declared_binaries=True)
    canonical_prefix = "runs/trial01/train/"
    for name, raw in payload.items():
        if not name.endswith(".md"):
            save(evidence.root, canonical_prefix + name, raw)
    manifest = build(evidence, tmp_path / "evidence.zip")
    cells = manifest["execution"]["stage_receipts"]
    canonical = cells["trial01/train"]
    assert canonical["declared_status"] == "completed"
    assert canonical["evidence_status"] == "incomplete_canonical_export"
    assert canonical["research_binary_checks"]["mixed-lora/adapter/README.md"] == (
        "missing_in_canonical; recovered_training_hash_and_size_verified"
    )
    assert cells["trial01/train-recovered"]["evidence_status"] == "receipt_completed"
    assert (
        cells["trial01/train-recovered"]["research_binary_checks"]["mixed-lora/adapter/README.md"]
        == "training_hash_and_size_verified"
    )
    with zipfile.ZipFile(tmp_path / "evidence.zip") as zipped:
        assert (
            "research/" + canonical_prefix + "mixed-lora/adapter/README.md" not in zipped.namelist()
        )
        assert (
            zipped.read("research/" + prefix + "mixed-lora/adapter/README.md")
            == payload["mixed-lora/adapter/README.md"]
        )


def test_partial_canonical_export_keeps_present_bytes_and_borrows_explicit_source_proof(
    evidence, tmp_path
):
    _, payload, _, _ = recovered(evidence, declared_binaries=True)
    canonical_prefix = "runs/trial01/train/"
    for name in ("receipt.json", "sources/campaign.txt"):
        save(evidence.root, canonical_prefix + name, payload[name])
    manifest = build(evidence, tmp_path / "evidence.zip")
    canonical = manifest["execution"]["stage_receipts"]["trial01/train"]
    assert canonical["evidence_status"] == "incomplete_canonical_export"
    assert canonical["source_checks"]["campaign"] == "receipt_hash_and_frozen_bytes_verified"
    assert canonical["source_checks"]["s1/unified.py"] == (
        "receipt_hash_and_frozen_bytes_verified; missing_in_canonical_verified_in_train-recovered"
    )
    assert canonical["research_binary_checks"]["decision-head.pt"] == (
        "missing_in_canonical; recovered_receipt_hash_verified"
    )
    with zipfile.ZipFile(tmp_path / "evidence.zip") as zipped:
        assert (
            zipped.read("research/" + canonical_prefix + "sources/campaign.txt")
            == payload["sources/campaign.txt"]
        )
        assert "research/" + canonical_prefix + "decision-head.pt" not in zipped.namelist()


def test_canonical_missing_declared_readme_is_still_rejected_without_valid_recovery(
    evidence, tmp_path
):
    prefix, receipt = executed(evidence)
    receipt["profiles"][0]["identity"]["research_adapter_files"] = {
        "README.md": {"sha256": archive.sha256(b"missing adapter README"), "bytes": 22}
    }
    save(evidence.root, prefix + "receipt.json", encoded(receipt))
    with pytest.raises(ValueError, match="trained adapter checksum mismatch or missing"):
        build(evidence, tmp_path / "evidence.zip")


def test_recovery_completed_transport_does_not_promote_failed_gpu_execution(evidence, tmp_path):
    recovered(evidence, status="failed")
    manifest = build(evidence, tmp_path / "evidence.zip")
    cell = manifest["execution"]["stage_receipts"]["trial01/train-recovered"]
    assert cell["declared_status"] == "failed"
    assert cell["evidence_status"] == "failed"


@pytest.mark.parametrize(
    "kind",
    [
        "missing_transport",
        "running_gpu",
        "partial_transport",
        "wrong_run",
        "wrong_stage",
        "wrong_prefix",
        "receipt_hash",
        "payload_hash",
        "payload_size",
        "extra_payload",
        "metadata_hash",
        "helper_hash",
        "original_failure_hash",
        "canonical_conflict",
    ],
)
def test_recovery_requires_exact_terminal_completed_transport_and_matching_bytes(
    evidence, tmp_path, kind
):
    prefix, payload, _, transport = recovered(evidence)
    transport_path = prefix + ".evidence-download/receipt.json"
    if kind == "missing_transport":
        (evidence.root / transport_path).unlink()
    elif kind == "running_gpu":
        receipt = json.loads(payload["receipt.json"])
        receipt["status"] = "running"
        save(evidence.root, prefix + "receipt.json", encoded(receipt))
    elif kind == "partial_transport":
        transport["status"] = "running"
    elif kind == "wrong_run":
        transport["run"] = "other01"
    elif kind == "wrong_stage":
        transport["stage"] = "ablate"
    elif kind == "wrong_prefix":
        transport["remote_prefix"] = "breakthrough/trial01/ablate"
    elif kind == "receipt_hash":
        transport["gpu_receipt_sha256"] = "0" * 64
    elif kind == "payload_hash":
        save(evidence.root, prefix + "decision-head.pt", b"different trained head")
    elif kind == "payload_size":
        transport["members"][0]["size_bytes"] += 1
    elif kind == "extra_payload":
        save(evidence.root, prefix + "not-declared.json", b"{}")
    elif kind == "metadata_hash":
        save(evidence.root, prefix + ".evidence-download/remote-listing.json", b"{}")
    elif kind == "helper_hash":
        transport["source_sha256"] = "0" * 64
    elif kind == "original_failure_hash":
        transport["original_export_failure"]["sha256"] = "0" * 64
    else:
        save(evidence.root, "runs/trial01/train/decision-head.pt", b"conflicting canonical head")
    if kind != "missing_transport":
        save(evidence.root, transport_path, encoded(transport))
    output = tmp_path / "evidence.zip"
    with pytest.raises(ValueError, match="train-recovered|canonical train"):
        build(evidence, output)
    assert not output.exists()


def test_baseline_weights_and_redundant_zip_are_explicitly_omitted_without_reading(
    evidence, tmp_path, monkeypatch
):
    baseline = evidence.root / "runs/trial01/train/model.safetensors"
    save(evidence.root, "runs/trial01/train/model.safetensors", b"baseline weights")
    save(evidence.root, "runs/trial01/train/results.zip", b"redundant export")
    read_bytes = Path.read_bytes

    def guarded_read(path):
        if path == baseline:
            pytest.fail("excluded baseline checkpoint must not be read into memory")
        return read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    manifest = build(evidence, tmp_path / "evidence.zip")
    assert set(manifest["omitted_files"]) == {
        "runs/trial01/train/model.safetensors",
        "runs/trial01/train/results.zip",
    }
    assert (
        manifest["execution"]["stage_receipts"]["trial01/train"]["evidence_status"]
        == "missing_receipt"
    )
    assert not any(name["path"].endswith(".safetensors") for name in manifest["members"])


@pytest.mark.parametrize("status", ["running", "failed", "blocked", "not_run"])
def test_partial_and_failed_stage_never_becomes_completed(evidence, tmp_path, status):
    prefix, _ = executed(evidence, status=status)
    save(evidence.root, prefix + "failure.json", encoded({"error": "original failure"}))
    manifest = build(evidence, tmp_path / "evidence.zip")
    assert manifest["execution"]["stage_receipts"]["trial01/train"]["evidence_status"] == status
    with zipfile.ZipFile(tmp_path / "evidence.zip") as zipped:
        assert zipped.read("research/" + prefix + "failure.json") == encoded(
            {"error": "original failure"}
        )


def test_missing_profile_receipt_is_explicitly_incomplete(evidence, tmp_path):
    prefix, receipt = executed(evidence)
    (evidence.root / (prefix + receipt["profiles"][0]["name"] + "/receipt.json")).unlink()
    manifest = build(evidence, tmp_path / "evidence.zip")
    cell = manifest["execution"]["stage_receipts"]["trial01/train"]
    assert cell["declared_status"] == "completed"
    assert cell["evidence_status"] == "incomplete"


def local_probe(evidence):
    prefix = "local-mps-01/"
    sources = {
        name: raw
        for name, raw in evidence.sources.items()
        if name.startswith("src/s1/")
        or name
        in {
            "scripts/breakthrough_prompt.py",
            "scripts/breakthrough_training.py",
            "scripts/prepare_mlx_validation.py",
        }
    }
    sources["scripts/probe_breakthrough_local.py"] = b"# actual local probe source\n"
    for name, raw in sources.items():
        save(evidence.root, prefix + "sources/" + name + ".txt", raw)
    receipt = {
        "status": "completed",
        "scope": "Local native prompt smoke only",
        "device": "mps",
        "cloud_calls": 0,
        "training_executed": False,
        "data_sha256": evidence.protocol["data"]["native_fixture"]["file_sha256"],
        "source_files": {name: archive.sha256(raw) for name, raw in sources.items()},
        "profiles_declared": ["current", "user_question"],
        "profiles": [{"name": "current"}, {"name": "user_question"}],
    }
    save(evidence.root, prefix + "receipt.json", encoded(receipt))
    for name in receipt["profiles_declared"]:
        save(
            evidence.root,
            prefix + name + ".json",
            encoded(
                [
                    {
                        "answers": {"q": {"probabilities": {"yes": 0.7, "no": 0.3}}},
                        "raw_logits": [2.0, 1.0],
                    }
                ]
            ),
        )
    return prefix, receipt


def test_local_mps_receipt_sources_and_raw_outputs_have_separate_scope(evidence, tmp_path):
    prefix, _ = local_probe(evidence)
    manifest = build(evidence, tmp_path / "evidence.zip")
    assert manifest["execution"]["state"] == "no_gpu"
    assert "Modal A100" in manifest["execution"]["state_scope"]
    assert manifest["execution"]["stage_receipts"] == {}
    local = manifest["execution"]["local_experiments"]["local-mps-01"]
    assert local["evidence_status"] == "receipt_completed"
    assert (
        local["source_checks"]["scripts/probe_breakthrough_local.py"]
        == "receipt_hash_verified; not_in_frozen_A100_source"
    )
    assert set(local["profiles"]) == {"current", "user_question"}
    with zipfile.ZipFile(tmp_path / "evidence.zip") as zipped:
        assert (
            "research/" + prefix + "sources/scripts/probe_breakthrough_local.py.txt"
            in zipped.namelist()
        )
        assert (
            zipped.read("research/" + prefix + "current.json")
            == (evidence.root / (prefix + "current.json")).read_bytes()
        )


@pytest.mark.parametrize("kind", ["cloud_call", "input_pin", "source_pin", "source_missing"])
def test_local_probe_scope_and_source_hashes_are_verified(evidence, tmp_path, kind):
    prefix, receipt = local_probe(evidence)
    if kind == "cloud_call":
        receipt["cloud_calls"] = 1
    elif kind == "input_pin":
        receipt["data_sha256"] = "0" * 64
    elif kind == "source_pin":
        receipt["source_files"]["scripts/probe_breakthrough_local.py"] = "0" * 64
    elif kind == "source_missing":
        (evidence.root / (prefix + "sources/src/s1/unified.py.txt")).unlink()
    save(evidence.root, prefix + "receipt.json", encoded(receipt))
    with pytest.raises(ValueError, match="local MPS"):
        build(evidence, tmp_path / "evidence.zip")


def test_local_source_revision_is_explicit_and_never_replaces_frozen_a100_source(
    evidence, tmp_path
):
    prefix, receipt = local_probe(evidence)
    path = "scripts/breakthrough_prompt.py"
    raw = b"# later local delimiter fix\n"
    save(evidence.root, prefix + "sources/" + path + ".txt", raw)
    receipt["source_files"][path] = archive.sha256(raw)
    save(evidence.root, prefix + "receipt.json", encoded(receipt))
    manifest = build(evidence, tmp_path / "evidence.zip")
    local = manifest["execution"]["local_experiments"]["local-mps-01"]
    assert local["preflight_differences"] == {
        path: {
            "frozen_a100_sha256": archive.sha256(evidence.sources[path]),
            "executed_local_sha256": archive.sha256(raw),
        }
    }
    assert local["source_checks"][path] == "receipt_hash_verified; differs_from_frozen_A100_source"
    with zipfile.ZipFile(tmp_path / "evidence.zip") as zipped:
        assert zipped.read("research/" + prefix + "sources/" + path + ".txt") == raw
        assert zipped.read("preflight/" + path + ".txt") == evidence.sources[path]


def test_incomplete_local_probe_is_not_called_completed(evidence, tmp_path):
    prefix, _ = local_probe(evidence)
    (evidence.root / (prefix + "user_question.json")).unlink()
    manifest = build(evidence, tmp_path / "evidence.zip")
    local = manifest["execution"]["local_experiments"]["local-mps-01"]
    assert local["declared_status"] == "completed"
    assert local["evidence_status"] == "incomplete"


@pytest.mark.parametrize(
    "kind",
    [
        "source_bytes",
        "source_pin",
        "source_missing",
        "source_extra",
        "source_vs_git",
        "protocol",
        "profile_source",
        "wrong_stage",
        "unplanned_stage",
        "no_gpu_conflict",
        "head_hash",
        "adapter_size",
    ],
)
def test_inconsistent_execution_evidence_is_rejected(evidence, tmp_path, kind):
    prefix, receipt = executed(evidence)
    if kind == "source_bytes":
        save(evidence.root, prefix + "sources/campaign.txt", b"tampered source")
    elif kind == "source_pin":
        receipt["source_files"]["campaign"] = "0" * 64
    elif kind == "source_missing":
        (evidence.root / (prefix + "sources/s1/unified.py.txt")).unlink()
    elif kind == "source_extra":
        save(evidence.root, prefix + "sources/unknown.txt", b"unrecorded source")
    elif kind == "source_vs_git":
        save(evidence.root, prefix + "sources/campaign.txt", b"different executed source")
        receipt["source_files"]["campaign"] = archive.sha256(b"different executed source")
    elif kind == "protocol":
        receipt["prospective_protocol"] = {"changed": True}
    elif kind == "profile_source":
        save(
            evidence.root,
            prefix + "released-user-head/receipt.json",
            encoded({"identity": {"variant": "released-user-head", "source_files": {}}}),
        )
    elif kind == "wrong_stage":
        receipt["stage"] = "ablate"
    elif kind == "unplanned_stage":
        save(evidence.root, "runs/trial01/unplanned/receipt.json", b"{}")
    elif kind == "no_gpu_conflict":
        save(evidence.root, "execution-status.json", encoded({"gpu_started": False}))
    elif kind == "head_hash":
        save(evidence.root, prefix + "decision-head.pt", b"trained head")
        receipt["profiles"][0]["identity"]["research_head_sha256"] = "0" * 64
    else:
        raw = b"trained adapter"
        save(evidence.root, prefix + "mixed-lora/adapter/adapter_model.safetensors", raw)
        receipt["profiles"][0]["identity"]["research_adapter_files"] = {
            "adapter_model.safetensors": {"sha256": archive.sha256(raw), "bytes": len(raw) + 1}
        }
    save(evidence.root, prefix + "receipt.json", encoded(receipt))
    output = tmp_path / "evidence.zip"
    with pytest.raises(ValueError):
        build(evidence, output)
    assert not output.exists()
    assert not output.with_suffix(".manifest.json").exists()


@pytest.mark.parametrize("kind", ["synthetic", "release", "native", "public", "status_protocol"])
def test_complete_frozen_input_hashes_are_required(evidence, tmp_path, kind):
    if kind == "synthetic":
        save(evidence.root, "data/train.jsonl", b"changed sample")
    elif kind == "release":
        evidence.support.pop("artifacts/release/datasets/train.jsonl")
    elif kind == "native":
        evidence.support["examples/benchmarks/mps.jsonl"] = b"changed media"
    elif kind == "public":
        evidence.support["configs/benchmarks/jevbench-public.json"] = b"changed contract"
    else:
        save(
            evidence.root,
            "execution-status.json",
            encoded({"gpu_started": False, "prospective_protocol_sha256": "0" * 64}),
        )
    with pytest.raises(ValueError, match="checksum|hash"):
        build(evidence, tmp_path / "evidence.zip")


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "/absolute",
        "part/../escape",
        "part//empty",
        "part\\escape",
        "C:escape",
        "bad\nname",
    ],
)
def test_archive_rejects_nonportable_and_traversal_member_names(evidence, tmp_path, name):
    evidence.support[name] = b"unsafe"
    with pytest.raises(ValueError, match="unsafe"):
        build(evidence, tmp_path / "evidence.zip")


@pytest.mark.parametrize("directory", [False, True])
def test_archive_rejects_symlinks_even_to_omitted_files(evidence, tmp_path, directory):
    target = evidence.root / "data" if directory else evidence.root / "data/train.jsonl"
    (evidence.root / ("linked" if directory else "model.safetensors")).symlink_to(
        target, target_is_directory=directory
    )
    with pytest.raises(ValueError, match="symlink"):
        build(evidence, tmp_path / "evidence.zip")


@pytest.mark.parametrize("existing", ["zip", "manifest"])
def test_archive_creation_never_overwrites_either_output(evidence, tmp_path, existing):
    output = tmp_path / "evidence.zip"
    occupied = output if existing == "zip" else output.with_suffix(".manifest.json")
    occupied.write_bytes(b"keep original bytes")
    with pytest.raises(FileExistsError):
        build(evidence, output)
    assert occupied.read_bytes() == b"keep original bytes"
    assert len(list(tmp_path.glob("evidence.*"))) == 1


def test_rejects_output_inside_evidence_tree(evidence):
    with pytest.raises(ValueError, match="inside"):
        build(evidence, evidence.root / "evidence.zip")


def test_oversized_research_binary_is_rejected_before_read(evidence, tmp_path, monkeypatch):
    save(evidence.root, "runs/trial01/train/decision-head.pt", b"x" * 4096)
    monkeypatch.setattr(archive, "MAX_MEMBER_BYTES", 4000)
    with pytest.raises(ValueError, match="oversized"):
        build(evidence, tmp_path / "evidence.zip")


def test_total_archive_size_limit_applies_to_all_sources_and_inputs(
    evidence, tmp_path, monkeypatch
):
    monkeypatch.setattr(archive, "MAX_TOTAL_BYTES", 100)
    with pytest.raises(ValueError, match="total size"):
        build(evidence, tmp_path / "evidence.zip")


def test_manifest_publish_race_rolls_back_only_new_zip(evidence, tmp_path, monkeypatch):
    output = tmp_path / "evidence.zip"
    original_link = archive.os.link

    def race_manifest(source, destination):
        if destination == output.with_suffix(".manifest.json"):
            destination.write_bytes(b"other writer's manifest")
        return original_link(source, destination)

    monkeypatch.setattr(archive.os, "link", race_manifest)
    with pytest.raises(FileExistsError):
        build(evidence, output)
    assert not output.exists()
    assert output.with_suffix(".manifest.json").read_bytes() == b"other writer's manifest"


def test_refuses_racing_evidence_mutation_without_publishing(evidence, tmp_path, monkeypatch):
    original = archive._zip_bytes

    def change_after_read(members):
        result = original(members)
        save(evidence.root, "late-error.json", encoded({"error": "late failure"}))
        return result

    monkeypatch.setattr(archive, "_zip_bytes", change_after_read)
    with pytest.raises(ValueError, match="changed"):
        build(evidence, tmp_path / "evidence.zip")
    assert not (tmp_path / "evidence.zip").exists()


@pytest.mark.parametrize(
    "tamper",
    ["zip_bytes", "member_hash", "member_size", "member_crc", "duplicate", "uncompressed_total"],
)
def test_verifier_detects_corruption_and_manifest_inconsistency(evidence, tmp_path, tamper):
    output = tmp_path / "evidence.zip"
    build(evidence, output)
    manifest_path = output.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if tamper == "zip_bytes":
        output.write_bytes(output.read_bytes() + b"unrecorded trailing bytes")
    elif tamper == "duplicate":
        manifest["members"].append(copy.deepcopy(manifest["members"][0]))
    elif tamper == "uncompressed_total":
        manifest["uncompressed_size_bytes"] += 1
    else:
        key = {"member_hash": "sha256", "member_size": "size_bytes", "member_crc": "crc32"}[tamper]
        manifest["members"][0][key] = (
            0 if key == "size_bytes" else "0" * (64 if key == "sha256" else 8)
        )
    manifest_path.write_bytes(encoded(manifest))
    with pytest.raises(ValueError):
        archive.verify_archive(output)


def test_actual_zip_crc_failure_is_detected_after_updated_container_hash(evidence, tmp_path):
    output = tmp_path / "evidence.zip"
    build(evidence, output)
    raw = bytearray(output.read_bytes())
    with zipfile.ZipFile(io.BytesIO(raw)) as zipped:
        info = zipped.infolist()[0]
        header = info.header_offset
        name_length = int.from_bytes(raw[header + 26 : header + 28], "little")
        extra_length = int.from_bytes(raw[header + 28 : header + 30], "little")
        raw[header + 30 + name_length + extra_length + info.compress_size // 2] ^= 1
    output.write_bytes(raw)
    manifest_path = output.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["archive"]["sha256"] = archive.sha256(raw)
    manifest_path.write_bytes(encoded(manifest))
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        archive.verify_archive(output)


def test_frozen_git_source_rejects_link_instead_of_resolving_it(monkeypatch):
    bundle = io.BytesIO()
    with tarfile.open(fileobj=bundle, mode="w") as tar:
        info = tarfile.TarInfo("src/s1/unified.py")
        info.type, info.linkname = tarfile.SYMTYPE, "../../outside"
        tar.addfile(info)
    monkeypatch.setattr(
        archive.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=bundle.getvalue())
    )
    with pytest.raises(ValueError, match="nonregular"):
        archive.frozen_sources()


def test_verify_cli_reports_offline_no_gpu_state(evidence, tmp_path, monkeypatch, capsys):
    output = tmp_path / "evidence.zip"
    build(evidence, output)
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "--verify", str(output)])
    archive.main()
    result = json.loads(capsys.readouterr().out)
    assert result["verified"] is True
    assert result["execution_state"] == "no_gpu"
