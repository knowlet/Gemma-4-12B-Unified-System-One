"""Offline release safety: immutable data, exact resume and calibration isolation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


def _snapshot(directory):
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


@pytest.fixture
def phase_run(tmp_path, monkeypatch):
    """Exercise phase orchestration with no model loading or accelerator calls."""
    recipe = _recipe(pilot_steps=1)
    data = {"train": [_case(0)], "calibration": [_case(0, "calibration")]}
    data_identity = {"calibration": {"dataset_sha256": "frozen-calibration"}}
    implementation = {"trainer": "stub"}
    identity = release.digest(
        {
            "recipe": release.asdict(recipe),
            "datasets": data_identity,
            "implementation": implementation,
        }
    )
    output = tmp_path / "run"
    calls = []
    monkeypatch.setattr(release, "load_release_data", lambda _: (data, data_identity))
    monkeypatch.setattr(release, "implementation_identity", lambda: implementation)
    monkeypatch.setattr(release, "_runtime", lambda: {})
    monkeypatch.setattr(release, "_memory", lambda: {})
    monkeypatch.setattr(release, "_free_cuda", lambda: None)

    def save_adapter(path):
        path.mkdir()
        (path / "adapter_model.safetensors").write_bytes(b"stub adapter")

    lm = SimpleNamespace(
        save_pretrained=lambda path, **_: (path / "model.safetensors").write_bytes(b"stub model")
    )
    lm.merge_and_unload = lambda **_: lm

    def new_model(*args, **kwargs):
        calls.append("model")
        return SimpleNamespace(
            lm=lm,
            processor=SimpleNamespace(save_pretrained=lambda _: None),
            save_adapter=save_adapter,
            prepare=lambda _: ({"input_ids": SimpleNamespace(shape=(1, 4))}, None, None),
        )

    def epoch(model, cases, phase, recipe, *, identity, resume=False, **kwargs):
        calls.append(("epoch", resume))
        if (phase / "latest.json").exists() and not resume:
            raise ValueError("checkpoint exists; explicitly enable resume")
        release.write_json(phase / "latest.json", {"identity": identity})
        return {"status": "completed", "step": 1, "identity": identity}

    monkeypatch.setattr(release, "_new_model", new_model)
    monkeypatch.setattr(release, "train_epoch", epoch)
    monkeypatch.setattr(release, "capture_calibration_logits", lambda *args: [])
    monkeypatch.setattr(
        release, "calibrate", lambda *args: {"temperature": 1.0, "n": 1, "records": []}
    )
    monkeypatch.setattr(release, "merge_drift", lambda *args: {"decisions": 0, "records": []})

    def evaluate(*args):
        calls.append("evaluate")
        return {"status": "completed"}

    monkeypatch.setattr(release, "evaluate_release", evaluate)

    def seed(stage, status="completed", *, receipt=True):
        release.write_json(output / "manifest.json", {"identity": identity})
        phase = output / stage
        phase.mkdir(exist_ok=True)
        (phase / "events.jsonl").write_text('{"event":"previous_attempt"}\n')
        if stage in ("pilot", "train"):
            release.write_json(phase / "latest.json", {"identity": identity})
        previous = {
            "stage": stage,
            "identity": identity,
            "status": status,
            "runtime": {},
            "result": {"status": status, "step": 1, "identity": identity},
        }
        if receipt:
            release.write_json(phase / "receipt.json", previous)
        return previous

    def run(stage, *, resume=False):
        return release.run_phase(
            stage,
            tmp_path / "data",
            output,
            recipe,
            resume=resume,
            commit=lambda: calls.append("commit"),
        )

    return SimpleNamespace(output=output, calls=calls, seed=seed, run=run)


@pytest.mark.parametrize("stage", ["pilot", "train", "evaluate"])
@pytest.mark.parametrize("resume", [False, True])
def test_completed_phase_is_an_unchanged_noop(phase_run, stage, resume):
    if stage == "train":
        phase_run.seed("pilot")
    expected = phase_run.seed(stage)
    before = _snapshot(phase_run.output)
    try:
        actual = phase_run.run(stage, resume=resume)
    finally:
        assert _snapshot(phase_run.output) == before
    assert actual == expected
    assert phase_run.calls == []


@pytest.mark.parametrize("stage", ["pilot", "train", "evaluate"])
@pytest.mark.parametrize(
    "status", ["running", "failed", "paused_budget", "completed_with_errors", "no_receipt"]
)
def test_unfinished_phase_requires_resume_before_any_writes(phase_run, stage, status):
    phase_run.seed(stage, status, receipt=status != "no_receipt")
    before = _snapshot(phase_run.output)
    with pytest.raises(ValueError, match="--resume"):
        phase_run.run(stage)
    assert _snapshot(phase_run.output) == before
    assert phase_run.calls == []


@pytest.mark.parametrize("stage", ["pilot", "train", "evaluate"])
@pytest.mark.parametrize("resume", [False, True])
def test_initial_phase_and_explicit_resume_can_complete(phase_run, stage, resume):
    if stage == "train":
        phase_run.seed("pilot")
    if resume:
        phase_run.seed(stage, "failed")
    result = phase_run.run(stage, resume=resume)
    assert result["status"] == result["result"]["status"] == "completed"
    assert json.loads((phase_run.output / stage / "receipt.json").read_text()) == result
    expected_call = "evaluate" if stage == "evaluate" else ("epoch", resume)
    assert expected_call in phase_run.calls


def test_duplicate_completed_invocations_do_not_block_following_phases(phase_run):
    for stage in ("pilot", "train", "evaluate"):
        completed = phase_run.run(stage)
        assert completed["status"] == "completed"
        before = _snapshot(phase_run.output)
        calls_before = list(phase_run.calls)
        assert phase_run.run(stage) == completed
        assert _snapshot(phase_run.output) == before
        assert phase_run.calls == calls_before


@pytest.mark.parametrize(
    "damage", ["identity", "stage", "status", "not_object", "missing_result", "missing_manifest"]
)
def test_invalid_existing_phase_is_rejected_without_mutation(phase_run, damage):
    previous = phase_run.seed("pilot")
    if damage in ("identity", "stage", "status"):
        previous[damage] = "invalid"
    elif damage == "not_object":
        previous = []
    elif damage == "missing_result":
        previous.pop("result")
    else:
        (phase_run.output / "manifest.json").unlink()
    release.write_json(phase_run.output / "pilot/receipt.json", previous)
    before = _snapshot(phase_run.output)
    with pytest.raises(ValueError):
        phase_run.run("pilot", resume=True)
    assert _snapshot(phase_run.output) == before
    assert phase_run.calls == []


def test_completed_with_errors_evaluation_retries_only_evaluation(phase_run):
    phase_run.seed("evaluate", "completed_with_errors")
    result = phase_run.run("evaluate", resume=True)
    assert result["status"] == result["result"]["status"] == "completed"
    assert "evaluate" in phase_run.calls
    assert not any(isinstance(call, tuple) and call[0] == "epoch" for call in phase_run.calls)
