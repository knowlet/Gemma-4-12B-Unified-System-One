"""Offline release safety: immutable data, exact resume and calibration isolation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "release_training", ROOT / "scripts/release_training.py"
)
release = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = release
spec.loader.exec_module(release)


def _case(index, split="train"):
    from s1.evaluation.datasets import EvaluationCase

    return EvaluationCase.model_validate(
        {
            "id": f"{split}-{index}",
            "group_id": f"{split}-source-{index}",
            "split": split,
            "request": {
                "state": f"independent state {split} {index}",
                "questions": {
                    "answer": {
                        "type": "choice",
                        "instructions": "Classify the state",
                        "criteria": {"left": "left", "right": "right"},
                    }
                },
            },
            "gold": {"answer": "left" if index % 2 else "right"},
        }
    )


def _model():
    torch = pytest.importorskip("torch")

    class TinyModel:
        def __init__(self):
            torch.manual_seed(73)
            self.lm = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(0.2))
            self.temperature = 1.0
            self.calls = []

        def logits(self, request):
            self.calls.append(request.state)
            index = int(request.state.rsplit(" ", 1)[-1])
            values = self.lm(torch.tensor([[1.0, (index + 1) / 10.0]]))
            # Exercise semantic candidate permutation as the real interface does.
            order = [0 if label == "left" else 1 for label in request.questions[0].labels()]
            return values[:, order]

        def save_adapter(self, path):
            path = Path(path)
            path.mkdir()
            torch.save(self.lm.state_dict(), path / "adapter_model.bin")

        def set_temperature(self, value):
            self.temperature = value

    return TinyModel()


def _recipe(**kwargs):
    return release.Recipe(checkpoint_every=2, progress_every=10, **kwargs)


def test_checkpoint_resume_matches_uninterrupted_weights_and_canonical_ledger(tmp_path):
    torch = pytest.importorskip("torch")
    cases = [_case(i) for i in range(6)]
    reference = _model()
    expected = release.train_epoch(
        reference, cases, tmp_path / "reference", _recipe(), identity="locked"
    )
    interrupted = _model()
    original = interrupted.logits
    calls = 0

    def fail_after_uncheckpointed_update(request):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("simulated process interruption")
        return original(request)

    interrupted.logits = fail_after_uncheckpointed_update
    with pytest.raises(RuntimeError, match="simulated"):
        release.train_epoch(interrupted, cases, tmp_path / "resumed", _recipe(), identity="locked")
    # Step three is in the log, but only steps one and two are durable.
    assert len((tmp_path / "resumed/updates.jsonl").read_text().splitlines()) == 3
    restored = _model()
    actual = release.train_epoch(
        restored, cases, tmp_path / "resumed", _recipe(), identity="locked", resume=True
    )
    assert actual["status"] == expected["status"] == "completed"
    assert actual["step"] == expected["step"] == len(cases)
    for a, b in zip(reference.lm.parameters(), restored.lm.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    records = [
        json.loads(line) for line in (tmp_path / "resumed/updates.jsonl").read_text().splitlines()
    ]
    assert [row["step"] for row in records] == list(range(1, 7))
    assert len({row["case_id"] for row in records}) == 6
    assert len(list((tmp_path / "resumed").glob("attempt-*.jsonl"))) == 1


def test_resume_rejects_different_identity_or_modified_state(tmp_path):
    release.train_epoch(_model(), [_case(i) for i in range(2)], tmp_path, _recipe(), identity="a")
    with pytest.raises(ValueError, match="identity"):
        release.train_epoch(
            _model(), [_case(i) for i in range(2)], tmp_path, _recipe(), identity="b", resume=True
        )
    pointer = json.loads((tmp_path / "latest.json").read_text())
    state = tmp_path / "checkpoints" / pointer["directory"] / "state.pt"
    state.write_bytes(state.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        release.train_epoch(
            _model(), [_case(i) for i in range(2)], tmp_path, _recipe(), identity="a", resume=True
        )


def test_nonfinite_gradient_cannot_update_or_create_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    model = _model()
    parameter = next(model.lm.parameters())
    original = parameter.detach().clone()
    hook = parameter.register_hook(lambda grad: torch.full_like(grad, float("nan")))
    with pytest.raises(RuntimeError, match="non-finite"):
        release.train_epoch(model, [_case(0)], tmp_path, _recipe(), identity="a")
    hook.remove()
    torch.testing.assert_close(parameter, original, rtol=0, atol=0)
    assert not (tmp_path / "latest.json").exists()


def test_calibration_rejects_test_and_records_exact_candidate_population():
    model = _model()
    with pytest.raises(ValueError, match="calibration-only"):
        release.calibrate(model, [_case(0, "test")])
    assert model.calls == []
    report = release.calibrate(model, [_case(i, "calibration") for i in range(4)])
    assert report["n"] == len(report["records"]) == 4
    assert report["nll_after"] <= report["nll_before"]
    assert model.temperature == report["temperature"]
    drift = release.merge_drift(report["records"][:2], report["records"])
    assert drift["decisions"] == 2 and drift["argmax_matches"] == 2
    assert drift["max_logit_abs_delta"] == 0


def _dataset_fixture(root):
    from s1.evaluation.datasets import fingerprint, request_fingerprint

    manifest = {"files": {}, "datasets": {}}
    for name, split, size in (
        ("train", "train", 2),
        ("calibration", "calibration", 1),
        ("test", "test", 1),
        ("media-test", "test", 1),
    ):
        cases = [_case(i + (100 if name == "media-test" else 0), split) for i in range(size)]
        path = root / f"{name}.jsonl"
        path.write_text("".join(case.model_dump_json() + "\n" for case in cases))
        sha = release.file_sha256(path)
        manifest["files"][path.name] = sha
        manifest["datasets"][path.name] = {
            "cases": size,
            "dataset_sha256": fingerprint(cases),
            "file_sha256": sha,
            "case_ids": [case.id for case in cases],
            "group_ids": [case.group_id for case in cases],
            "request_sha256s": [request_fingerprint(case.request) for case in cases],
        }
    release.write_json(root / "manifest.json", manifest)
    return manifest


def test_prepared_manifest_binds_dataset_contents_and_aligned_ids(tmp_path):
    manifest = _dataset_fixture(tmp_path)
    data, identity = release.load_release_data(tmp_path, expected_counts=(2, 1, 1))
    assert len(data["train"]) == identity["train"]["cases"] == 2
    manifest["datasets"]["train.jsonl"]["case_ids"].reverse()
    release.write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="receipt"):
        release.load_release_data(tmp_path, expected_counts=(2, 1, 1))
    _dataset_fixture(tmp_path)
    with (tmp_path / "train.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="checksum"):
        release.load_release_data(tmp_path, expected_counts=(2, 1, 1))


def test_pilot_limit_and_longest_case_are_explicit(tmp_path):
    cases = [_case(i) for i in range(6)]
    result = release.train_epoch(
        _model(),
        cases,
        tmp_path,
        _recipe(),
        identity="pilot",
        max_steps=2,
        first_case_id=cases[5].id,
    )
    records = [json.loads(line) for line in (tmp_path / "updates.jsonl").read_text().splitlines()]
    assert result["step"] == result["target_steps"] == 2
    assert records[0]["case_id"] == cases[5].id
    assert result["order"] != list(range(6))


def test_implementation_identity_changes_when_trainer_changes(tmp_path, monkeypatch):
    trainer = tmp_path / "scripts/release_training.py"
    trainer.parent.mkdir()
    trainer.write_text("version_one")
    (tmp_path / "uv.lock").write_text("locked")
    monkeypatch.setattr(release, "__file__", str(trainer))
    monkeypatch.setattr(release, "_runtime", lambda: {"packages": {"torch": "locked"}})
    before = release.implementation_identity()
    trainer.write_text("version_two")
    after = release.implementation_identity()
    assert release.digest(before) != release.digest(after)


def test_budget_never_reports_unfinished_epoch_completed(tmp_path, monkeypatch):
    ticks = iter([0.0, 2.0, 3.0])
    monkeypatch.setattr(release.time, "monotonic", lambda: next(ticks))
    result = release.train_epoch(
        _model(), [_case(0)], tmp_path, _recipe(max_training_seconds=1), identity="limited"
    )
    assert result["status"] == "paused_budget" and result["step"] == 0
