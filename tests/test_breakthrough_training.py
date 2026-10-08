"""Soft-target objectives, clean calibration, actual learning and durable progress."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from s1.contracts import DecisionRequest
from s1.evaluation.datasets import EvaluationCase

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

spec = importlib.util.spec_from_file_location(
    "breakthrough_training", Path(__file__).parents[1] / "scripts/breakthrough_training.py"
)
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)
pytestmark = pytest.mark.inference


def row(kind="choice", *, split="calibration", probabilities=(0.8, 0.2), target=None, scale=2):
    target = probabilities if target is None else target
    labels = ["false", "true"] if kind == "noul" else [str(i) for i in range(len(probabilities))]
    return {
        "case_id": kind,
        "group_id": kind,
        "split": split,
        "type": kind,
        "question_id": "decision",
        "labels": labels,
        "score_values": list(map(float, range(len(labels)))) if kind == "score" else None,
        "logits": [scale * math.log(value) for value in probabilities],
        "target": list(target),
        "sequence_tokens": 23,
    }


def calibration_rows():
    return [
        row("choice"),
        row("noul"),
        row("score", probabilities=(0.6, 0.3, 0.1)),
    ]


def case(split="train"):
    return EvaluationCase.model_validate(
        {
            "id": "independent-toy",
            "group_id": "fresh-source-group",
            "split": split,
            "request": {
                "state": "Synthetic observation",
                "questions": [
                    {"id": "answer", "type": "noul", "instructions": "Is the observation true?"}
                ],
            },
            "gold": {"answer": "true"},
            "soft_gold": {"answer": {"false": 0.1, "true": 0.9}},
        }
    )


def test_soft_ce_and_brier_use_complete_distribution_with_correct_gradient():
    logits = torch.tensor([0.0, 0.0], requires_grad=True)
    loss = training.soft_loss(logits, [0.2, 0.8], brier_weight=0.1)
    assert loss.item() == pytest.approx(math.log(2) + 0.1 * 0.18)
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.tensor([0.33, -0.33]), atol=1e-7, rtol=1e-6)


def test_soft_distribution_optimum_has_entropy_loss_and_zero_gradient():
    logits = torch.tensor([math.log(0.2), math.log(0.8)], requires_grad=True)
    loss = training.soft_loss(logits, [0.2, 0.8])
    assert loss.item() == pytest.approx(-0.2 * math.log(0.2) - 0.8 * math.log(0.8))
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros(2), atol=1e-7, rtol=0)


def test_zero_targets_and_extreme_logits_have_finite_loss_and_gradient():
    logits = torch.tensor([1000.0, -1000.0, 0.0], requires_grad=True)
    loss = training.soft_loss(logits, [0.0, 1.0, 0.0])
    assert loss.item() == pytest.approx(2000.2)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(logits.grad, torch.tensor([1.0, -1.0, 0.0]), atol=0, rtol=0)


@pytest.mark.parametrize(
    "logits,targets",
    [
        ([float("nan"), 0.0], [0.5, 0.5]),
        ([float("inf"), 0.0], [0.5, 0.5]),
        ([float("-inf"), 0.0], [0.5, 0.5]),
        ([0.0, 1.0], [1.0]),
        ([[0.0, 1.0]], [0.5, 0.5]),
        ([0.0, 1.0], [-0.1, 1.1]),
        ([0.0, 1.0], [0.2, 0.2]),
        ([0.0, 1.0], [float("nan"), 1.0]),
    ],
)
def test_soft_loss_rejects_nonfinite_misaligned_or_invalid_distribution(logits, targets):
    with pytest.raises(ValueError):
        training.soft_loss(torch.tensor(logits), targets)


@pytest.mark.parametrize("weight", [-0.1, float("nan"), float("inf")])
def test_soft_loss_rejects_invalid_brier_weight(weight):
    with pytest.raises(ValueError):
        training.soft_loss(torch.tensor([0.0, 1.0]), [0.5, 0.5], brier_weight=weight)


def test_soft_targets_follow_semantic_label_order():
    question = DecisionRequest(
        state="Observation",
        questions=[
            {
                "id": "pick",
                "type": "choice",
                "instructions": "Choose the item.",
                "criteria": {"second": "Second item", "first": "First item"},
            }
        ],
    ).questions[0]
    annotated = SimpleNamespace(soft_gold={"pick": {"first": 0.8, "second": 0.2}})
    assert training.target(annotated, question) == [0.2, 0.8]
    with pytest.raises(ValueError, match="soft target"):
        training.target(SimpleNamespace(soft_gold=None), question)


def test_per_type_temperature_recovers_known_distribution_without_changing_argmax():
    rows = calibration_rows()
    fitted = training.fit_temperatures(rows)
    assert set(fitted) == {"choice", "noul", "score"}
    for record in rows:
        result = fitted[record["type"]]
        assert result["n"] == 1 and not result["boundary"]
        assert result["temperature"] == pytest.approx(2.0, rel=0.005)
        assert result["soft_ce_after"] < result["soft_ce_before"]
        logits = np.asarray(record["logits"])
        assert logits.argmax() == (logits / result["temperature"]).argmax()


@pytest.mark.parametrize("split", ["train", "development", "test"])
def test_only_calibration_records_can_fit_temperature(split):
    rows = calibration_rows()
    rows[0]["split"] = split
    with pytest.raises(ValueError, match="calibration"):
        training.fit_temperatures(rows)


def test_calibration_requires_nonempty_complete_type_population():
    with pytest.raises(ValueError):
        training.fit_temperatures([])
    with pytest.raises(ValueError, match="primitive"):
        training.fit_temperatures(calibration_rows()[:-1])


@pytest.mark.parametrize("corruption", ["logit", "target_nan", "target_sum", "length", "dimension"])
def test_calibration_rejects_nonfinite_invalid_or_misaligned_records(corruption):
    rows = calibration_rows()
    if corruption == "logit":
        rows[0]["logits"][0] = float("nan")
    elif corruption == "target_nan":
        rows[0]["target"][0] = float("nan")
    elif corruption == "target_sum":
        rows[0]["target"] = [0.2, 0.2]
    elif corruption == "length":
        rows[0]["target"] = [1.0]
    else:
        rows[0]["logits"] = [[0.0, 1.0]]
    with pytest.raises(ValueError):
        training.fit_temperatures(rows)


def test_heldout_metrics_use_soft_gold_and_expected_score():
    records = [
        row("choice", split="test", probabilities=(0.75, 0.25), scale=1),
        row("noul", split="test", probabilities=(0.2, 0.8), target=(0.3, 0.7), scale=1),
        row("score", split="test", probabilities=(0.2, 0.3, 0.5), target=(0.5, 0.3, 0.2), scale=1),
    ]
    records[-1]["score_values"] = [0.0, 1.0, 4.0]
    metrics = training.summarize(records, dict.fromkeys(("choice", "noul", "score"), 1.0))
    assert metrics["all"]["n"] == 3
    assert metrics["choice"]["kl"] == pytest.approx(0, abs=1e-14)
    assert metrics["noul"]["brier"] == pytest.approx(0.02)
    assert metrics["score"]["brier"] == pytest.approx(0.18)
    assert metrics["score"]["kl"] == pytest.approx(0.5 * math.log(2.5) + 0.2 * math.log(0.4))
    assert metrics["score"]["expected_score_mae"] == pytest.approx(1.2)
    assert metrics["score"]["oracle_argmax_match"] == 0
    assert training.summarize([], {}) == {
        key: {"n": 0} for key in ("all", "choice", "noul", "score")
    }


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_metrics_reject_invalid_temperature(temperature):
    with pytest.raises(ValueError):
        training.summarize([row(split="test")], {"choice": temperature})


@pytest.mark.parametrize("corruption", ["logit", "target", "levels"])
def test_metrics_reject_invalid_probability_or_score_population(corruption):
    record = row("score", split="test", probabilities=(0.6, 0.3, 0.1))
    if corruption == "logit":
        record["logits"][0] = float("inf")
    elif corruption == "target":
        record["target"] = [-0.1, 0.3, 0.8]
    else:
        record["score_values"] = [0.0, 1.0]
    with pytest.raises(ValueError):
        training.summarize([record], {"score": 1.0})


@pytest.fixture
def head_model():
    head = torch.nn.Linear(2, 2, bias=True)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    return SimpleNamespace(head=head, letters=torch.tensor([0, 1]), softcap=3.0)


@pytest.fixture
def head_rows():
    result = []
    for index, (feature, truth) in enumerate(
        [
            ([1.0, 0.0], [0.9, 0.1]),
            ([-1.0, 0.0], [0.1, 0.9]),
            ([0.0, 1.0], [0.9, 0.1]),
            ([0.0, -1.0], [0.1, 0.9]),
        ]
    ):
        record = row(split="train", target=truth)
        record["case_id"] = record["group_id"] = str(index)
        record["hidden"] = torch.tensor(feature, requires_grad=True)
        result.append(record)
    return result


def test_frozen_head_really_learns_without_mutating_backbone_features_or_original_head(
    head_model, head_rows
):
    before = {key: value.detach().clone() for key, value in head_model.head.state_dict().items()}
    fitted, result = training.fit_head(head_model, head_rows, steps=48, learning_rate=0.05)
    assert result["steps"] == 48 and result["batch_size"] == 4
    assert result["losses"][-1] < result["losses"][0] - 0.2
    assert torch.isfinite(fitted["weight"]).all() and torch.isfinite(fitted["bias"]).all()
    assert not torch.equal(fitted["weight"], before["weight"])
    for key, value in head_model.head.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    assert all(record["hidden"].grad is None for record in head_rows)
    assert head_model.head.weight.grad is None
    features = torch.stack([record["hidden"].detach() for record in head_rows])
    predictions = torch.nn.functional.linear(features, fitted["weight"], fitted["bias"])
    assert predictions.argmax(-1).tolist() == [0, 1, 0, 1]


@pytest.mark.parametrize("split", ["calibration", "development", "test"])
def test_head_fitting_cannot_consume_heldout_features(head_model, head_rows, split):
    head_rows[0]["split"] = split
    with pytest.raises(ValueError, match="training"):
        training.fit_head(head_model, head_rows, steps=1)


@pytest.mark.parametrize("steps", [0, -1, True, 1.5])
def test_head_fitting_rejects_invalid_update_count(head_model, head_rows, steps):
    with pytest.raises(ValueError):
        training.fit_head(head_model, head_rows, steps=steps)


@pytest.mark.parametrize("rate", [0, -0.01, float("nan"), float("inf")])
def test_head_fitting_rejects_invalid_learning_rate(head_model, head_rows, rate):
    with pytest.raises(ValueError):
        training.fit_head(head_model, head_rows, steps=1, learning_rate=rate)


@pytest.mark.parametrize("corruption", ["feature_nan", "feature_dimension", "target_count"])
def test_head_fitting_rejects_nonfinite_or_mismatched_features(head_model, head_rows, corruption):
    if corruption == "feature_nan":
        head_rows[0]["hidden"] = torch.tensor([float("nan"), 0.0])
    elif corruption == "feature_dimension":
        head_rows[0]["hidden"] = torch.tensor([0.0, 1.0, 2.0])
    else:
        head_rows[0]["target"] = [0.1, 0.1, 0.8]
    with pytest.raises((ValueError, RuntimeError)):
        training.fit_head(head_model, head_rows, steps=2)


class TinyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter_delta = torch.nn.Parameter(torch.tensor(0.0))
        self.checkpointing = False
        self.saved_values = []

    def gradient_checkpointing_enable(self, **kwargs):
        assert kwargs == {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
        self.checkpointing = True

    def gradient_checkpointing_disable(self):
        self.checkpointing = False

    def save_pretrained(self, directory):
        directory = Path(directory)
        directory.mkdir(exist_ok=True)
        value = float(self.adapter_delta.detach())
        (directory / "adapter_state.json").write_text(json.dumps({"adapter_delta": value}))
        self.saved_values.append(value)


@pytest.fixture
def lora_model():
    return SimpleNamespace(lm=TinyAdapter())


def toy_capture(model, request, mode):
    assert mode == "user_question"
    return [
        {
            "question": request.questions[0],
            "logits": torch.stack([-model.lm.adapter_delta, model.lm.adapter_delta]),
        }
    ]


def test_lora_updates_real_parameter_and_saves_32_step_progress_and_hashes(
    lora_model, monkeypatch, tmp_path
):
    monkeypatch.setattr(training, "capture", toy_capture)
    output, commits = tmp_path / "run", []

    def commit():
        path = output / "progress.json"
        commits.append(json.loads(path.read_bytes())["step"] if path.exists() else None)

    result = training.train_lora(
        lora_model, [case()], output, steps=33, learning_rate=0.05, commit=commit
    )
    assert result["status"] == "completed" and result["steps"] == 33
    assert result["trainable_parameters"] == 1
    assert float(lora_model.lm.adapter_delta.detach()) > 0.5
    assert result["losses"][-1]["loss"] < result["losses"][0]["loss"] - 0.2
    updates = [json.loads(line) for line in (output / "updates.jsonl").read_bytes().splitlines()]
    assert len(updates) == 33 and [r["step"] for r in updates] == list(range(1, 34))
    assert all(math.isfinite(r["loss"]) and math.isfinite(r["gradient_norm"]) for r in updates)
    assert commits[:2] == [1, 32] and len(lora_model.lm.saved_values) == 3
    assert not lora_model.lm.training and not lora_model.lm.checkpointing
    assert json.loads((output / "training.json").read_bytes()) == result
    for name, identity in result["adapter_files"].items():
        raw = (output / "adapter" / name).read_bytes()
        assert identity == {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


@pytest.mark.parametrize("split", ["calibration", "development", "test"])
def test_lora_cannot_update_from_heldout_cases(lora_model, tmp_path, split):
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="training"):
        training.train_lora(lora_model, [case(split)], output, steps=1)
    assert not output.exists() and float(lora_model.lm.adapter_delta.detach()) == 0


@pytest.mark.parametrize("steps", [0, -1, True, 1.5])
def test_lora_rejects_invalid_update_count(lora_model, tmp_path, steps):
    with pytest.raises(ValueError):
        training.train_lora(lora_model, [case()], tmp_path / "run", steps=steps)


def test_lora_requires_fp32_trainable_adapter_parameters(lora_model, tmp_path):
    lora_model.lm.to(dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        training.train_lora(lora_model, [case()], tmp_path / "run", steps=1)


class FiniteForwardNaNBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return torch.full_like(gradient, float("nan"))


def test_nonfinite_backward_does_not_update_weights_or_publish_completed_result(
    lora_model, monkeypatch, tmp_path
):
    calls = 0

    def capture(model, request, mode):
        nonlocal calls
        calls += 1
        rows = toy_capture(model, request, mode)
        if calls == 33:
            rows[0]["logits"] = FiniteForwardNaNBackward.apply(rows[0]["logits"])
        return rows

    monkeypatch.setattr(training, "capture", capture)
    output = tmp_path / "run"
    with pytest.raises((ValueError, RuntimeError)):
        training.train_lora(lora_model, [case()], output, steps=33, learning_rate=0.05)
    assert float(lora_model.lm.adapter_delta.detach()) == lora_model.lm.saved_values[-1]
    assert json.loads((output / "progress.json").read_bytes())["step"] == 32
    assert len((output / "updates.jsonl").read_bytes().splitlines()) == 32
    assert not (output / "training.json").exists()


def test_accumulation_covers_each_case_once_and_uses_mean_loss_and_gradient(
    lora_model, monkeypatch, tmp_path
):
    cases, specifications = [], {}
    for index in range(8):
        annotated = case().model_copy(deep=True)
        annotated.id = annotated.group_id = f"fresh-{index}"
        annotated.request.state = annotated.id
        truth, scale = 0.55 + 0.05 * index, 0.5 + 0.05 * index
        annotated.soft_gold = {"answer": {"false": 1 - truth, "true": truth}}
        cases.append(annotated)
        specifications[annotated.id] = (truth, scale)
    calls, gradients, checkpoints = [], [], []
    original_optimizer = torch.optim.AdamW

    class InspectAdamW(original_optimizer):
        def step(self, *args, **kwargs):
            gradients.append(float(self.param_groups[0]["params"][0].grad.detach()))
            return super().step(*args, **kwargs)

    def capture(model, request, mode):
        assert mode == "user_question"
        truth, scale = specifications[request.state]
        theta = float(model.lm.adapter_delta.detach())
        probability = 1 / (1 + math.exp(-2 * scale * theta))
        difference = probability - truth
        # Closed-form binary CE + 0.1 Brier oracle, independent of soft_loss.
        expected_loss = (
            -(1 - truth) * math.log(1 - probability)
            - truth * math.log(probability)
            + 0.2 * difference**2
        )
        expected_gradient = 2 * scale * difference + 0.8 * scale * difference * probability * (
            1 - probability
        )
        calls.append((request.state, theta, expected_loss, expected_gradient))
        value = model.lm.adapter_delta * scale
        return [{"question": request.questions[0], "logits": torch.stack([-value, value])}]

    output = tmp_path / "accumulated"

    def commit():
        progress = json.loads((output / "progress.json").read_bytes())
        updates = [
            json.loads(line) for line in (output / "updates.jsonl").read_bytes().splitlines()
        ]
        checkpoints.append((progress["step"], updates))

    monkeypatch.setattr(torch.optim, "AdamW", InspectAdamW)
    monkeypatch.setattr(training, "capture", capture)
    result = training.train_lora(
        lora_model, cases, output, steps=2, accumulation=4, learning_rate=0.05, commit=commit
    )
    assert len(calls) == len({entry[0] for entry in calls}) == 8
    assert {entry[0] for entry in calls} == {annotated.id for annotated in cases}
    assert result["gradient_accumulation"] == 4
    assert result["n_training_calls"] == result["n_unique_training_cases"] == 8
    assert len(gradients) == len(result["losses"]) == 2
    assert [entry[1] for entry in calls[:4]] == [0.0] * 4
    assert calls[4][1] > 0 and len({entry[1] for entry in calls[4:]}) == 1
    for index, update in enumerate(result["losses"]):
        micro = calls[index * 4 : (index + 1) * 4]
        expected_gradient = sum(entry[3] for entry in micro) / 4
        assert abs(expected_gradient) < 1  # Clipping does not obscure the mean-gradient check.
        assert gradients[index] == pytest.approx(expected_gradient, abs=1e-7, rel=1e-6)
        assert update["gradient_norm"] == pytest.approx(abs(expected_gradient), abs=1e-7)
        assert update["loss"] == pytest.approx(sum(entry[2] for entry in micro) / 4, abs=1e-7)
        assert update["case_ids"] == [entry[0] for entry in micro]
        assert update["case_id"] == micro[-1][0]
    # At the first checkpoint, all four contributing IDs are already durable.
    assert checkpoints[0][0] == 1 and len(checkpoints[0][1]) == 1
    assert checkpoints[0][1][0]["case_ids"] == [entry[0] for entry in calls[:4]]
    assert checkpoints[-1][1] == result["losses"]
    assert json.loads((output / "training.json").read_bytes()) == result


@pytest.mark.parametrize("accumulation", [0, -1, True, 1.5])
def test_invalid_accumulation_is_rejected_before_any_update(lora_model, tmp_path, accumulation):
    output = tmp_path / "invalid-accumulation"
    with pytest.raises(ValueError):
        training.train_lora(lora_model, [case()], output, steps=1, accumulation=accumulation)
    assert not output.exists() and float(lora_model.lm.adapter_delta.detach()) == 0
    assert not lora_model.lm.checkpointing
