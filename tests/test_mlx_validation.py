"""Offline export preflight checks; no MLX import, model weights or GPU work."""

import builtins
import hashlib
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(scope="module")
def verifier():
    script = Path(__file__).resolve().parents[1] / "scripts" / "verify_mlx_export.py"
    return runpy.run_path(str(script))


@pytest.fixture
def aligned():
    manifest = {
        "prompt_version": 1,
        "dataset_sha256": "dataset-fingerprint",
        "source_model": "fixture/gemma4",
        "source_revision": "a" * 40,
        "temperature": 1.0,
        "letters": list(range(52)),
        "cases": [
            {
                "id": "case-1",
                "file": "case-1.npz",
                "slots": [2],
                "nopts": [2],
                "questions": [{"id": "route", "labels": ["left", "right"]}],
            }
        ],
    }
    reference = {
        "status": "ok",
        "dataset_sha256": manifest["dataset_sha256"],
        "pinned_revision": manifest["source_revision"],
        "model": {
            "name": manifest["source_model"],
            "resolved_revision": manifest["source_revision"],
            "temperature": manifest["temperature"],
        },
        "cases": [
            {
                "id": "case-1",
                "question_count": 1,
                "input_tokens": 3,
                "end_to_end_samples": [
                    {"answers": {"route": {"probabilities": {"right": 0.75, "left": 0.25}}}}
                ],
            }
        ],
    }
    return manifest, reference


@pytest.fixture
def model_config():
    return {
        "model_type": "gemma4_unified",
        "tie_word_embeddings": True,
        "text_config": {"final_logit_softcapping": 30.0},
        "vision_config": {"model_type": "fixture-vision"},
        "audio_config": {"model_type": "fixture-audio"},
    }


def test_matching_reference_follows_manifest_label_order(verifier, aligned):
    manifest, reference = aligned
    paired = verifier["validate_reference"](manifest, reference)
    assert list(paired) == ["case-1"]
    np.testing.assert_array_equal(paired["case-1"][0], [0.25, 0.75])


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("fingerprint", "dataset hashes differ"),
        ("revision", "source revisions differ"),
        ("temperature", "temperatures differ"),
        ("labels", "candidate labels differ"),
    ],
)
def test_reference_mismatch_rejected(verifier, aligned, mismatch, message):
    manifest, reference = aligned
    if mismatch == "fingerprint":
        reference["dataset_sha256"] = "another-dataset"
    elif mismatch == "revision":
        reference["model"]["resolved_revision"] = "b" * 40
    elif mismatch == "temperature":
        reference["model"]["temperature"] = 2.0
    else:
        reference["cases"][0]["end_to_end_samples"][0]["answers"]["route"]["probabilities"] = {
            "left": 0.25,
            "unknown": 0.75,
        }
    with pytest.raises(ValueError, match=message):
        verifier["validate_reference"](manifest, reference)


def test_complete_tied_model_config_accepted(verifier, model_config):
    assert verifier["validate_model_config"](model_config) == 30.0


@pytest.mark.parametrize("location", ["outer", "text"])
def test_untied_head_rejected(verifier, model_config, location):
    target = model_config if location == "outer" else model_config["text_config"]
    target["tie_word_embeddings"] = False
    with pytest.raises(ValueError, match="untied output heads are unsupported"):
        verifier["validate_model_config"](model_config)


@pytest.mark.parametrize("padded", [False, True])
def test_prepared_attention_contract(verifier, aligned, tmp_path, padded):
    case = aligned[0]["cases"][0]
    path = tmp_path / case["file"]
    ids = np.array([[10, 11, 12]], dtype=np.int64)
    attention = np.array([[1, 1, 0 if padded else 1]], dtype=np.int64)
    np.savez(path, input_ids=ids, attention_mask=attention)
    if padded:
        with pytest.raises(ValueError, match="all-ones, unpadded attention_mask"):
            verifier["load_case_arrays"](tmp_path, case)
    else:
        arrays, digest = verifier["load_case_arrays"](tmp_path, case)
        np.testing.assert_array_equal(arrays["input_ids"], ids)
        assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_failed_preflight_writes_report_before_mlx_import(verifier, aligned, tmp_path, monkeypatch):
    manifest, reference = aligned
    reference["dataset_sha256"] = "another-dataset"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    ref_path = tmp_path / "reference.json"
    ref_path.write_text(json.dumps(reference))
    output = tmp_path / "report.json"
    original_import = builtins.__import__

    def forbid_mlx(name, *args, **kwargs):
        if name == "mlx" or name.startswith("mlx.") or name == "mlx_vlm":
            pytest.fail("failed preflight must not initialize MLX")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_mlx)
    result = verifier["main"](
        [
            "--inputs",
            str(tmp_path),
            "--model",
            str(tmp_path / "unloaded-model"),
            "--reference",
            str(ref_path),
            "--output",
            str(output),
        ]
    )
    report = json.loads(output.read_text())
    assert result == 1
    assert report["status"] == "error"
    assert report["parity_accepted"] is False
    assert report["failure"]["stage"] == "preflight"
    assert "dataset hashes differ" in report["failure"]["message"]


@pytest.fixture(scope="module")
def release_evaluator():
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_mlx_release.py"
    return runpy.run_path(str(path))


@pytest.fixture
def release_inputs(aligned):
    import copy

    manifests, datasets = {}, {}
    for role, filename in (
        ("calibration", "calibration.jsonl"),
        ("test", "test.jsonl"),
        ("media", "media-test.jsonl"),
    ):
        manifest = copy.deepcopy(aligned[0])
        split = "calibration" if role == "calibration" else "test"
        digest = hashlib.sha256(role.encode()).hexdigest()
        manifest.update(
            split=split,
            dataset_sha256=digest,
            dataset_file_sha256=digest,
            source_checkpoint_sha256="c" * 64,
            source_s1_config_sha256="f" * 64,
        )
        case = manifest["cases"][0]
        case.update(
            id=role,
            split=split,
            group_id=role,
            request_sha256=digest,
            tensor_file_sha256="d" * 64,
            gold={"route": "left"},
        )
        case["questions"][0]["type"] = "choice"
        manifests[role] = manifest
        datasets[filename] = {
            "split": split,
            "cases": 1,
            "questions": 1,
            "dataset_sha256": digest,
            "file_sha256": digest,
            "case_ids": [role],
            "group_ids": [role],
            "request_sha256s": [digest],
        }
    datasets["train.jsonl"] = {
        "split": "train",
        "cases": 1,
        "case_ids": ["train"],
        "group_ids": ["train-group"],
        "request_sha256s": ["e" * 64],
    }
    return manifests, {"split_audit": {"valid": True}, "datasets": datasets}


def test_release_alignment_and_exact_counts(release_evaluator, release_inputs):
    manifests, receipt = release_inputs
    validate = release_evaluator["validate_manifests"]
    identity = validate(manifests, receipt, expected_counts={role: 1 for role in manifests})
    assert identity["source_checkpoint_sha256"] == "c" * 64
    assert identity["source_s1_config_sha256"] == "f" * 64
    assert identity["train_cases_audited"] == 1
    with pytest.raises(ValueError, match="expected exactly 256"):
        validate(manifests, receipt)


@pytest.mark.parametrize("key", ["id", "group_id", "request_sha256"])
def test_release_recomputes_train_calibration_leakage(release_evaluator, release_inputs, key):
    manifests, receipt = release_inputs
    receipt_key = {"id": "case_ids", "group_id": "group_ids", "request_sha256": "request_sha256s"}[
        key
    ]
    leaked = receipt["datasets"]["train.jsonl"][receipt_key][0]
    manifests["calibration"]["cases"][0][key] = leaked
    receipt["datasets"]["calibration.jsonl"][receipt_key] = [leaked]
    with pytest.raises(ValueError, match="split leakage"):
        release_evaluator["validate_manifests"](
            manifests, receipt, expected_counts={role: 1 for role in manifests}
        )


def test_release_binds_capture_to_dataset_receipt(release_evaluator, release_inputs):
    manifests, receipt = release_inputs
    manifests["test"]["dataset_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprint differs"):
        release_evaluator["validate_manifests"](
            manifests, receipt, expected_counts={role: 1 for role in manifests}
        )


def test_release_captures_must_share_exact_source_sidecar(release_evaluator, release_inputs):
    manifests, receipt = release_inputs
    manifests["test"]["source_s1_config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="different checkpoints"):
        release_evaluator["validate_manifests"](
            manifests, receipt, expected_counts={role: 1 for role in manifests}
        )


def test_checkpoint_digest_streams_weights_and_ignores_calibration(tmp_path, release_evaluator):
    _, preparation = release_evaluator["shared_scripts"]()
    (tmp_path / "config.json").write_text('{"model_type":"gemma4_unified"}')
    (tmp_path / "model.safetensors").write_bytes(b"fixture weights")
    before = preparation["checkpoint_identity"](tmp_path)
    assert [item["path"] for item in before["files"]] == ["config.json", "model.safetensors"]
    (tmp_path / "s1_config.json").write_text('{"temperature":2.0}')
    (tmp_path / "README.md").write_text("changed model card")
    assert preparation["checkpoint_identity"](tmp_path) == before
    (tmp_path / "model.safetensors").write_bytes(b"different weights")
    assert preparation["checkpoint_identity"](tmp_path)["sha256"] != before["sha256"]


@pytest.mark.parametrize("indexed", [False, True])
def test_checkpoint_identity_rejects_unpinned_extra_weights(tmp_path, release_evaluator, indexed):
    _, preparation = release_evaluator["shared_scripts"]()
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"fixture")
    if indexed:
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {"layer.weight": "model.safetensors"},
                }
            )
        )
    preparation["checkpoint_identity"](tmp_path)
    (tmp_path / "extra.safetensors").write_bytes(b"additional loader-visible weights")
    with pytest.raises(ValueError, match="unindexed"):
        preparation["checkpoint_identity"](tmp_path)


@pytest.mark.parametrize("mismatch", ["gold", "labels", "dataset_bytes"])
def test_capture_golds_and_label_order_are_bound_to_frozen_data(
    release_evaluator, release_inputs, tmp_path, mismatch
):
    from s1.evaluation.datasets import EvaluationCase, fingerprint, request_fingerprint

    manifests, receipt = release_inputs
    for filename, dataset in receipt["datasets"].items():
        case = EvaluationCase.model_validate(
            {
                "id": dataset["case_ids"][0],
                "group_id": dataset["group_ids"][0],
                "split": dataset["split"],
                "gold": {"route": "left"},
                "request": {
                    "state": filename,
                    "questions": [
                        {
                            "id": "route",
                            "type": "choice",
                            "instructions": "Choose a side",
                            "criteria": {"left": "Go left", "right": "Go right"},
                        }
                    ],
                },
            }
        )
        path = tmp_path / filename
        path.write_text(case.model_dump_json() + "\n")
        dataset.update(
            file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            dataset_sha256=fingerprint([case]),
            questions=1,
            request_sha256s=[request_fingerprint(case.request)],
        )
        role = next(
            (key for key, value in release_evaluator["DATASET_NAMES"].items() if value == filename),
            None,
        )
        if role:
            manifests[role].update(
                dataset_sha256=dataset["dataset_sha256"], dataset_file_sha256=dataset["file_sha256"]
            )
            manifests[role]["cases"][0]["request_sha256"] = dataset["request_sha256s"][0]
    validate = release_evaluator["validate_dataset_files"]
    validate(manifests, receipt, tmp_path)
    captured = manifests["calibration"]["cases"][0]
    if mismatch == "gold":
        captured["gold"]["route"] = "right"
        message = "gold differ"
    elif mismatch == "labels":
        captured["questions"][0]["labels"].reverse()
        message = "label ordering differs"
    else:
        with (tmp_path / "calibration.jsonl").open("a") as stream:
            stream.write("\n")
        message = "file SHA256 mismatch"
    with pytest.raises(ValueError, match=message):
        validate(manifests, receipt, tmp_path)


def test_conversion_receipt_binds_source_and_export_bytes(
    tmp_path, model_config, release_evaluator
):
    _, preparation = release_evaluator["shared_scripts"]()
    model_config["quantization"] = {"bits": 4, "group_size": 64}
    (tmp_path / "config.json").write_text(json.dumps(model_config))
    (tmp_path / "model.safetensors").write_bytes(b"packed fixture weights")
    exported = preparation["checkpoint_identity"](tmp_path)
    receipt = {"source_checkpoint_sha256": "c" * 64, "files": exported["files"]}
    identity = {"source_checkpoint_sha256": "c" * 64}
    assert release_evaluator["validate_conversion"](tmp_path, receipt, identity)[0] == exported
    (tmp_path / "model.safetensors").write_bytes(b"modified packed weights")
    with pytest.raises(ValueError, match="hashes differ"):
        release_evaluator["validate_conversion"](tmp_path, receipt, identity)


def test_release_fits_existing_numpy_algorithm_only_on_calibration(release_evaluator):
    from s1.training import fit_temperature

    rows = [[2.0, 0.0]] * 4
    golds = [0, 0, 0, 1]
    records = [
        {
            "status": "ok",
            "split": "calibration",
            "answers": {
                "q": {
                    "raw_logits": row,
                    "labels": ["left", "right"],
                    "gold": ["left", "right"][gold],
                }
            },
        }
        for row, gold in zip(rows, golds)
    ]
    fit = release_evaluator["fit_calibration"](records, expected_cases=4)
    expected = fit_temperature(rows, golds)
    assert fit["temperature"] == expected["temperature"]
    assert fit["nll_after"] < fit["nll_before"]
    assert fit["temperature"] > 1
    records[-1]["split"] = "test"
    with pytest.raises(ValueError, match="calibration-only"):
        release_evaluator["fit_calibration"](records, expected_cases=4)


def test_release_metrics_and_partial_coverage(release_evaluator, release_inputs):
    manifest = release_inputs[0]["test"]
    records = [
        {
            "status": "ok",
            "prepared_forward_samples_ms": [10.0],
            "answers": {
                "q": {"raw_logits": [0.0, 0.0], "labels": ["left", "right"], "gold": "left"}
            },
        }
    ]
    release_evaluator["apply_temperature"](records, 2.0)
    summary = release_evaluator["summarize_split"](records, manifest)
    assert summary["case_coverage"] == summary["decision_coverage"] == 1
    assert summary["metrics"]["accuracy"] == 1
    assert summary["metrics"]["nll"] == pytest.approx(np.log(2))
    assert summary["metrics"]["brier"] == 0.5
    assert summary["metrics"]["ece"] == 0.5
    empty = release_evaluator["summarize_split"]([], manifest)
    assert empty["decision_coverage"] == empty["case_coverage"] == 0
    assert empty["metrics"] == {"n": 0}


def test_raw_logits_and_parity_share_projection(verifier, aligned, monkeypatch):
    logits = np.arange(52, dtype=np.float32).reshape(1, 52)
    globals_ = verifier["selected_logits"].__globals__
    monkeypatch.setitem(globals_, "_selected_logits_mlx", lambda *args: logits)
    core = SimpleNamespace(
        eval=lambda value: None,
        arange=np.arange,
        array=np.array,
        int32=np.int32,
        inf=np.inf,
        where=np.where,
        softmax=lambda value, axis: (
            np.exp(value - value.max(axis=axis, keepdims=True))
            / np.exp(value - value.max(axis=axis, keepdims=True)).sum(axis=axis, keepdims=True)
        ),
    )
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=core))
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    case = aligned[0]["cases"][0]
    raw = verifier["selected_logits"](None, {}, case, list(range(52)), None)
    np.testing.assert_array_equal(raw[0], [0, 1])
    probability = verifier["selected_probabilities"](None, {}, case, list(range(52)), 2.0, None)
    np.testing.assert_allclose(probability[0], np.exp([0, 0.5]) / np.exp([0, 0.5]).sum())


def test_local_capture_preserves_evaluation_metadata(tmp_path, monkeypatch, benchmark_case):
    from s1 import unified
    from s1.evaluation.datasets import EvaluationCase, fingerprint, request_fingerprint

    preparation = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts" / "prepare_mlx_validation.py")
    )
    model = tmp_path / "merged"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"gemma4_unified"}')
    (model / "model.safetensors").write_bytes(b"local fixture weights")
    (model / "s1_config.json").write_text(
        json.dumps(
            {
                "source_model": "fixture/gemma4",
                "source_revision": "a" * 40,
                "temperature": 2.0,
            }
        )
    )
    case = EvaluationCase.model_validate(
        {
            **benchmark_case.model_dump(),
            "split": "calibration",
            "group_id": "source-group",
            "task_id": "routing",
            "language": "en",
            "schema_id": "v2-schema",
        }
    )
    dataset = tmp_path / "calibration.jsonl"
    dataset.write_text(case.model_dump_json() + "\n")
    processor = SimpleNamespace(tokenizer=object())
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoProcessor=SimpleNamespace(from_pretrained=lambda path: processor)),
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=lambda: pytest.fail("local capture must not resolve a Hub revision"),
            snapshot_download=lambda *a, **kw: pytest.fail("local capture must not download"),
        ),
    )
    monkeypatch.setattr(unified, "candidate_ids", lambda tokenizer: list(range(52)))

    def prepare(context, request):
        assert context.device == "cpu"
        values = np.array([[1, 2, 3]], dtype=np.int64)
        inputs = {
            "input_ids": SimpleNamespace(numpy=lambda: values),
            "attention_mask": SimpleNamespace(numpy=lambda: np.ones_like(values)),
        }
        return inputs, np.array([0, 1, 2]), np.array([2, 2, 2])

    monkeypatch.setattr(unified.UnifiedDecisionModel, "prepare", prepare)
    output = tmp_path / "captured"
    preparation["main"](
        [
            "--model",
            str(model),
            "--dataset",
            str(dataset),
            "--split",
            "calibration",
            "--output",
            str(output),
        ]
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["dataset_sha256"] == fingerprint([case])
    assert manifest["source_model"] == "fixture/gemma4"
    assert (
        manifest["source_s1_config_sha256"]
        == hashlib.sha256((model / "s1_config.json").read_bytes()).hexdigest()
    )
    assert (
        manifest["source_checkpoint_sha256"] == preparation["checkpoint_identity"](model)["sha256"]
    )
    captured = manifest["cases"][0]
    assert captured["gold"] == case.gold
    assert captured["group_id"] == "source-group"
    assert captured["request_sha256"] == request_fingerprint(case.request)
    assert (
        captured["tensor_file_sha256"]
        == hashlib.sha256((output / captured["file"]).read_bytes()).hexdigest()
    )


def test_release_bad_manifest_pin_fails_without_mlx_or_sidecar_write(
    release_evaluator, tmp_path, monkeypatch
):
    (tmp_path / "manifest.json").write_text("{}")
    sidecar = tmp_path / "s1_config.json"
    original_sidecar = '{"temperature":1.0}'
    sidecar.write_text(original_sidecar)
    output = tmp_path / "failed.json"
    original_import = builtins.__import__

    def forbid_mlx(name, *args, **kwargs):
        if name == "mlx" or name.startswith("mlx.") or name == "mlx_vlm":
            pytest.fail("failed release preflight must not initialize MLX")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_mlx)
    arguments = ["--model", str(tmp_path), "--output", str(output)]
    for role in ("calibration", "test", "media"):
        arguments.extend([f"--{role}-inputs", str(tmp_path), f"--{role}-manifest-sha256", "0" * 64])
    for kind in ("dataset", "conversion"):
        arguments.extend(
            [
                f"--{kind}-manifest",
                str(tmp_path / "manifest.json"),
                f"--{kind}-manifest-sha256",
                "0" * 64,
            ]
        )
    assert release_evaluator["main"](arguments) == 1
    report = json.loads(output.read_text())
    assert report["status"] == "error"
    assert report["release_validation_complete"] is False
    assert report["failure"]["stage"] == "preflight"
    assert "SHA256 mismatch" in report["failure"]["message"]
    assert sidecar.read_text() == original_sidecar
    assert not (tmp_path / "base_s1_config.json").exists()
