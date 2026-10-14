"""CPU-only replay integrity checks; synthetic tensors are never benchmark evidence."""

import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_jev_browser_fit_readout import FIT, collection

from s1 import browser_evidence, browser_readout, unified

torch = pytest.importorskip("torch")


def replay_fixture(tmp_path):
    train = collection(tmp_path, "train", group="training episode")
    development = collection(tmp_path, "development", group="development episode")
    cases, receipts = {}, {}
    for split, path in (("train", train), ("development", development)):
        cases[split], receipts[split], _snapshot = FIT._validated_snapshot(path, split)
    origin = tmp_path / "original-fit"
    origin.mkdir()
    # Nontrivial original token IDs prove exported artifacts retain their order,
    # while fit_head sees only the matching 52 local rows.
    model = SimpleNamespace(
        name="knowlet/Gemma-4-12B-Unified-System-One",
        revision="a66f836b56605039fe040f330180e336d19b3362",
        quantization="nf4",
        temperature=2.7830344470383452,
        softcap=30.0,
        letters=torch.arange(151, 99, -1),
        head=SimpleNamespace(
            weight=torch.arange(256 * 4).float().reshape(256, 4) / 1000,
            bias=torch.linspace(-0.2, 0.2, 256),
        ),
    )
    empty = {"ids": [], "groups": [], "request_fingerprints": []}
    browser_readout.write_readout(
        origin / "control",
        model,
        "grounded_user_question_v1",
        provenance={"source_scope": "development_only", "train": empty, "calibration": empty},
    )
    data = {}
    for split, selected in cases.items():
        data[split] = [
            {
                "case_id": case.id,
                "group_id": case.group_id,
                "split": case.split,
                "type": question.type,
                "question_id": question.id,
                "labels": question.labels(),
                "target": [case.soft_gold[question.id][label] for label in question.labels()],
                "logits": [0.1] * len(question.labels()),
                "hidden": torch.arange(4).float() + 0.2,
                "sequence_tokens": 100,
            }
            for case in selected
            for question in case.request.questions
        ]
    torch.save(data, origin / "features.pt")
    source_manifest = browser_evidence.capture_runtime_sources(
        origin, additional_files={"scripts/jev_browser_fit_readout.py": Path(FIT.__file__)}
    )
    receipt = {
        "stage": "complete",
        "scope": "development-only frozen-backbone readout fitting; not model task success",
        "full_lm_head_selected_rows_unchanged": True,
        "train_collection": receipts["train"],
        "development_collection": receipts["development"],
        "split_audit": {"valid": True},
        "objective_temperature": 1.0,
        "model": model.name,
        "revision": model.revision,
        "prompt_variant": "grounded_user_question_v1",
        "deployment_temperature": model.temperature,
        "quantization_details": {"mode": "nf4", "test_synthetic_control": True},
        "train_feature_rows": len(data["train"]),
        "development_feature_rows": len(data["development"]),
        "source_manifest": source_manifest,
        "source_verification": {
            "success": True,
            "identity_sha256": source_manifest["identity_sha256"],
            "current_identity_sha256": source_manifest["identity_sha256"],
            "changed": [],
            "missing": [],
            "added": [],
        },
    }
    (origin / "receipt.json").write_text(json.dumps(receipt) + "\n")
    return SimpleNamespace(
        train=train,
        development=development,
        cases=cases,
        receipts=receipts,
        origin=origin,
        model=model,
        data=data,
        receipt=receipt,
    )


def snapshot(fixture):
    return FIT._validated_replay(
        fixture.origin, fixture.receipts["train"], fixture.receipts["development"]
    )


def run_fit(fixture, output, **kwargs):
    return FIT.fit(
        train_cases=fixture.train,
        development_cases=fixture.development,
        output_dir=output,
        features_from=fixture.origin,
        steps=2,
        **kwargs,
    )


def forbidden(*_args, **_kwargs):
    raise AssertionError("Replay attempted a new model or CUDA workload")


def test_cpu_replay_preserves_bytes_pins_and_performs_only_selected_row_fit(tmp_path, monkeypatch):
    fixture = replay_fixture(tmp_path)
    original = {
        path.relative_to(fixture.origin).as_posix(): path.read_bytes()
        for path in fixture.origin.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(unified, "UnifiedDecisionModel", forbidden)
    monkeypatch.setattr(FIT, "feature_rows", forbidden)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", forbidden)
    monkeypatch.setattr(torch.backends.cuda, "enable_cudnn_sdp", forbidden)
    actual_load, load_options = torch.load, []

    def tracked_load(*args, **kwargs):
        load_options.append(kwargs)
        return actual_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", tracked_load)
    actual_fit, fitted = FIT.fit_head, []

    def checked_fit(model, rows, **kwargs):
        assert model.head.weight.shape == (52, 4)
        assert model.letters.tolist() == list(range(52))
        assert torch.equal(model.head.weight, fixture.model.head.weight[fixture.model.letters])
        assert all(row["split"] == "train" for row in rows)
        fitted.append(model)
        return actual_fit(model, rows, **kwargs)

    checked_fit.__code__ = checked_fit.__code__.replace(co_filename=actual_fit.__code__.co_filename)
    monkeypatch.setattr(FIT, "fit_head", checked_fit)
    output = tmp_path / "cpu-fit"
    receipt = run_fit(fixture, output)
    assert len(fitted) == 1 and load_options
    assert all(options == {"map_location": "cpu", "weights_only": True} for options in load_options)
    assert receipt["stage"] == "complete"
    assert receipt["execution_mode"] == "cpu_only_frozen_feature_replay"
    assert receipt["backbone_forward_calls_this_run"] == 0
    assert receipt["full_lm_head_loaded_this_run"] is False
    assert "full_lm_head_selected_rows_unchanged" not in receipt
    assert receipt["origin_full_lm_head_selected_rows_unchanged"] is True
    assert receipt["control_rows_unchanged"] is True
    assert "peak_cuda_allocated_bytes" not in receipt
    assert receipt["feature_replay"]["capture_sha_status"] == "retrospective_replay_freeze"
    assert receipt["feature_replay"]["origin_feature_sha_was_prospective"] is False
    assert "does not authenticate" in receipt["feature_replay"]["limitation"]
    assert receipt["feature_replay_verification"]["original"]["success"] is True
    assert receipt["feature_replay_verification"]["frozen"]["success"] is True
    assert receipt["feature_replay_verification"]["emitted_copies"]["success"] is True
    assert (output / "features.pt").read_bytes() == original["features.pt"]
    for label in ("manifest.json", "weights.pt"):
        assert (output / "control" / label).read_bytes() == original["control/" + label]
    for label, blob in original.items():
        assert (output / "feature-origin" / label).read_bytes() == blob
        assert (fixture.origin / label).read_bytes() == blob
    learned = browser_readout.load_readout(
        output / "learned", fixture.model, "grounded_user_question_v1"
    )
    info = learned.telemetry()
    assert info["candidate_ids"] == fixture.model.letters.tolist()
    assert info["base_temperature"] == info["temperature"] == fixture.model.temperature
    assert info["softcap"] == fixture.model.softcap
    assert info["quantization"] == "nf4"
    assert info["provenance"]["train"]["ids"] == [fixture.cases["train"][0].id]
    assert info["provenance"]["calibration"]["ids"] == []
    assert info["provenance"]["training_objective"]["backbone_forward_calls_this_run"] == 0
    saved = actual_load(io.BytesIO((output / "learned/weights.pt").read_bytes()), weights_only=True)
    assert not torch.equal(saved["weight"], fixture.model.head.weight[fixture.model.letters])


@pytest.mark.parametrize(
    "change",
    [
        "incomplete",
        "source_false",
        "source_identity",
        "collections",
        "unchanged_head",
        "source_bytes",
        "source_extra",
        "feature_hash",
    ],
)
def test_replay_requires_complete_matching_origin_and_exact_source_evidence(tmp_path, change):
    fixture = replay_fixture(tmp_path)
    receipt = copy.deepcopy(fixture.receipt)
    if change == "incomplete":
        receipt["stage"] = "fit"
    elif change == "source_false":
        receipt["source_verification"]["success"] = False
    elif change == "source_identity":
        receipt["source_verification"]["current_identity_sha256"] = "0" * 64
    elif change == "collections":
        receipt["train_collection"]["dataset_sha256"] = "0" * 64
    elif change == "unchanged_head":
        receipt["full_lm_head_selected_rows_unchanged"] = False
    elif change == "source_bytes":
        source = fixture.origin / "sources/scripts/jev_browser_fit_readout.py"
        source.write_bytes(source.read_bytes() + b"\n# altered\n")
    elif change == "source_extra":
        (fixture.origin / "sources/s1/extra.py").write_text("# unrecorded\n")
    else:
        receipt["features_sha256"] = "0" * 64
    (fixture.origin / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        snapshot(fixture)


@pytest.mark.parametrize(
    "change",
    [
        "case_id",
        "group_id",
        "split",
        "type",
        "question_id",
        "labels",
        "target",
        "count",
        "missing",
        "extra",
        "hidden_dim",
        "hidden_dtype",
        "hidden_nonfinite",
        "logits_count",
        "logits_nonfinite",
        "sequence_tokens",
    ],
)
def test_replay_validates_every_ordered_feature_row_against_current_cases(tmp_path, change):
    fixture = replay_fixture(tmp_path)
    data = copy.deepcopy(fixture.data)
    row = data["train"][0]
    if change in {"case_id", "group_id", "split", "type", "question_id"}:
        row[change] = "changed"
    elif change == "labels":
        row["labels"].reverse()
    elif change == "target":
        row["target"].reverse()
    elif change == "count":
        data["train"].append(copy.deepcopy(row))
    elif change == "missing":
        del row["labels"]
    elif change == "extra":
        row["unverified_oracle"] = "DONE"
    elif change == "hidden_dim":
        row["hidden"] = torch.zeros(5)
    elif change == "hidden_dtype":
        row["hidden"] = row["hidden"].half()
    elif change == "hidden_nonfinite":
        row["hidden"][0] = float("nan")
    elif change == "logits_count":
        row["logits"].pop()
    elif change == "logits_nonfinite":
        row["logits"][0] = float("inf")
    else:
        row["sequence_tokens"] = True
    torch.save(data, fixture.origin / "features.pt")
    with pytest.raises(ValueError, match="feature"):
        FIT._validated_feature_rows(snapshot(fixture), fixture.cases, 4)


@pytest.mark.parametrize(
    "change",
    [
        "learned",
        "provenance",
        "model",
        "revision",
        "prompt",
        "temperature",
        "base_temperature",
        "quantization",
        "weights_hash",
        "weights_shape",
        "weights_nonfinite",
    ],
)
def test_replay_control_requires_pinned_untrained_finite_rows(tmp_path, change):
    fixture = replay_fixture(tmp_path)
    manifest_path = fixture.origin / "control/manifest.json"
    control = json.loads(manifest_path.read_text())
    if change == "learned":
        control["kind"] = "learned_rows"
    elif change == "provenance":
        control["provenance"]["train"]["ids"] = ["some case"]
    elif change in {
        "model",
        "revision",
        "prompt",
        "temperature",
        "base_temperature",
        "quantization",
    }:
        field = {
            "model": "base_model_id",
            "revision": "base_revision",
            "prompt": "prompt_variant",
        }.get(change, change)
        control[field] = 1.0 if "temperature" in field else "changed"
    elif change == "weights_hash":
        control["weights_sha256"] = "0" * 64
    else:
        payload = {
            "weight": torch.zeros(51 if change == "weights_shape" else 52, 4),
            "bias": torch.zeros(52),
        }
        if change == "weights_nonfinite":
            payload["weight"][0, 0] = float("nan")
        stream = io.BytesIO()
        torch.save(payload, stream)
        (fixture.origin / "control/weights.pt").write_bytes(stream.getvalue())
        control["weights_sha256"] = FIT._sha(stream.getvalue())
    manifest_path.write_text(json.dumps(control))
    captured = snapshot(fixture)
    frozen = tmp_path / "frozen"
    FIT._freeze_collection(captured, frozen)
    with pytest.raises(ValueError):
        FIT._replay_descriptors(captured, frozen)


@pytest.mark.parametrize(
    "location,label",
    [
        ("original", "features.pt"),
        ("original", "control/weights.pt"),
        ("original", "receipt.json"),
        ("frozen", "features.pt"),
        ("frozen", "control/manifest.json"),
        ("frozen", "input_manifest.json"),
        ("emitted_copies", "features.pt"),
        ("emitted_copies", "control/weights.pt"),
    ],
)
def test_final_replay_integrity_invalidates_only_new_sidecars(
    tmp_path, monkeypatch, location, label
):
    fixture = replay_fixture(tmp_path)
    output = tmp_path / "cpu-fit"
    original_control = (fixture.origin / "control/manifest.json").read_bytes()
    actual_fit = FIT.fit_head

    def mutate(model, rows, **kwargs):
        roots = {
            "original": fixture.origin,
            "frozen": output / "feature-origin",
            "emitted_copies": output,
        }
        changed = roots[location] / label
        changed.write_bytes(changed.read_bytes() + b" ")
        return actual_fit(model, rows, **kwargs)

    mutate.__code__ = mutate.__code__.replace(co_filename=actual_fit.__code__.co_filename)
    monkeypatch.setattr(FIT, "fit_head", mutate)
    with pytest.raises(RuntimeError, match="replay bytes"):
        run_fit(fixture, output)
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["stage"] == "failed"
    assert receipt["feature_replay_verification"][location]["success"] is False
    for name in ("control", "learned"):
        assert not (output / name / "manifest.json").exists()
        assert (output / name / "manifest.invalid.json").exists()
        assert (output / name / "weights.pt").exists()
    assert (fixture.origin / "control/manifest.json").read_bytes() == original_control


def test_cli_explicit_features_from_is_forwarded_without_running_fit(monkeypatch, capsys):
    import sys

    called = []

    def capture(**kwargs):
        called.append(kwargs)
        return {
            "stage": "complete",
            "scope": "test",
            "learned_artifact": "learned",
            "control_artifact": "control",
        }

    monkeypatch.setattr(FIT, "fit", capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fit",
            "--train-cases",
            "train/cases.jsonl",
            "--development-cases",
            "dev/cases.jsonl",
            "--output-dir",
            "new-fit",
            "--features-from",
            "original-fit",
            "--steps",
            "512",
        ],
    )
    FIT.main()
    assert called[0]["features_from"] == "original-fit" and called[0]["steps"] == 512
    assert json.loads(capsys.readouterr().out)["stage"] == "complete"


def test_exact_control_snapshot_cannot_be_replaced_by_another_valid_frozen_control(tmp_path):
    fixture = replay_fixture(tmp_path)
    captured = snapshot(fixture)
    frozen = tmp_path / "frozen"
    FIT._freeze_collection(captured, frozen)
    altered = {
        "weight": fixture.model.head.weight[fixture.model.letters] + 1,
        "bias": fixture.model.head.bias[fixture.model.letters],
    }
    stream = io.BytesIO()
    torch.save(altered, stream)
    manifest = json.loads((frozen / "control/manifest.json").read_text())
    manifest["weights_sha256"] = FIT._sha(stream.getvalue())
    (frozen / "control/manifest.json").write_text(json.dumps(manifest))
    (frozen / "control/weights.pt").write_bytes(stream.getvalue())
    # This other artifact is independently valid, but it was not the frozen
    # source-control matrix that this training run is authorized to reuse.
    browser_readout.load_readout(frozen / "control", fixture.model, "grounded_user_question_v1")
    with pytest.raises(ValueError, match="Frozen control bytes"):
        FIT._replay_descriptors(captured, frozen)


@pytest.mark.parametrize("ordering", ["cases", "questions"])
def test_equal_counts_still_require_exact_ordered_case_and_question_alignment(tmp_path, ordering):
    fixture = replay_fixture(tmp_path)
    data = copy.deepcopy(fixture.data)
    case = fixture.cases["train"][0]
    if ordering == "cases":
        second = case.model_copy(update={"id": case.id + "-different"})
        fixture.cases["train"].append(second)
        row = copy.deepcopy(data["train"][0])
        row["case_id"] = second.id
    else:
        question = case.request.questions[0].model_copy(update={"id": "different_head"})
        fixture.cases["train"][0] = case.model_copy(
            update={
                "request": case.request.model_copy(
                    update={"questions": [*case.request.questions, question]}
                ),
                "soft_gold": {**case.soft_gold, question.id: case.soft_gold["operation"]},
            }
        )
        row = copy.deepcopy(data["train"][0])
        row["question_id"] = question.id
    data["train"].append(row)
    torch.save(data, fixture.origin / "features.pt")
    captured = snapshot(fixture)
    captured["receipt"]["train_feature_rows"] = 2
    FIT._validated_feature_rows(captured, fixture.cases, 4)
    data["train"].reverse()
    stream = io.BytesIO()
    torch.save(data, stream)
    captured["blobs"]["features.pt"] = stream.getvalue()
    with pytest.raises(ValueError, match="ordered case/question"):
        FIT._validated_feature_rows(captured, fixture.cases, 4)


def test_python_bytecode_cache_is_excluded_but_extra_source_remains_rejected(tmp_path):
    fixture = replay_fixture(tmp_path)
    baseline = snapshot(fixture)["manifest"]
    cache = fixture.origin / "sources/scripts/__pycache__"
    cache.mkdir()
    (cache / "jev_browser_benchmark.cpython-312.pyc").write_bytes(b"ignored runtime cache")
    assert snapshot(fixture)["manifest"] == baseline
    (fixture.origin / "sources/s1/new.py").write_text("# unrecorded source\n")
    with pytest.raises(ValueError, match="inventory"):
        snapshot(fixture)
