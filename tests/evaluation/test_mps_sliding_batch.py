"""MPS batching policy with CPU-only tensors/backbones; no accelerator calls."""

from __future__ import annotations

import copy
import weakref
from types import SimpleNamespace

import pytest

from s1.contracts import AudioInput, DecisionRequest, ImageInput
from s1.evaluation.gemma import predict_batch

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.inference


def request_for(lengths, *, request_index=0, media_shape=None, causal_length=None):
    questions = [
        {
            "id": f"q{index}",
            "type": "choice",
            "instructions": "Choose an observed target",
            "criteria": {"a": "First", "b": "Second", **({"c": "Third"} if index % 2 else {})},
        }
        for index in range(len(lengths))
    ]
    return DecisionRequest.model_validate(
        {
            "state": {
                "lengths": lengths,
                "request_index": request_index,
                "media_shape": media_shape,
                "causal_length": causal_length,
            },
            "questions": questions,
            "media": [ImageInput(data="AA==").model_dump(), AudioInput(samples=[0.2]).model_dump()]
            if media_shape is not None
            else [],
        }
    )


class CPUBackboneModel:
    """Device is metadata only; the asserted mask rule is evaluated on CPU."""

    def __init__(self, *, device="mps", window=4, sliding=True, per_layer=None):
        self.device = device
        self.config = SimpleNamespace(
            layer_types=["sliding_attention", "full_attention"] if sliding else ["full_attention"],
            sliding_window=window,
        )
        if per_layer is not None:
            self.config.layer_types = ["sliding_attention"] * len(per_layer)
            self.config.per_layer_config = [SimpleNamespace(sliding_window=x) for x in per_layer]
        self.window = min(per_layer) if per_layer is not None else window
        self.enforce_safe_mask = str(device).split(":", 1)[0].lower() == "mps" and sliding
        self.lm = SimpleNamespace(
            eval=lambda: None, config=SimpleNamespace(get_text_config=lambda: self.config)
        )
        self.tok = SimpleNamespace(pad_token_id=0)
        self.head = torch.nn.Linear(1, 4, bias=False, device="cpu")
        with torch.no_grad():
            self.head.weight[:, 0] = torch.tensor([-1.0, 0.5, 1.0, 10.0], device="cpu")
        self.letters = torch.tensor([0, 1, 2], device="cpu")
        self.temperature, self.softcap = 1.7, 2.0
        self.forwards, self.preparations, self.alive = [], [], []
        self.peak_prepared = 0

    def prepare(self, request):
        state, questions = request.state, request.questions
        length = (
            state["lengths"][int(questions[0].id[1:])]
            if len(questions) == 1
            else state["causal_length"]
        )
        request_marker = 100 + state["request_index"]
        ids = torch.full((1, length), request_marker, dtype=torch.long, device="cpu")
        slots = torch.arange(length - len(questions), length, device="cpu")
        for q, slot in zip(questions, slots, strict=True):
            ids[0, slot] = 1000 + state["request_index"] * 100 + int(q.id[1:])
        inputs = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "mm_token_type_ids": torch.zeros_like(ids),
            "token_type_ids": torch.zeros_like(ids),
            "position_ids": torch.arange(length, device="cpu").unsqueeze(0),
        }
        if state["media_shape"] is not None:
            size = state["media_shape"]
            inputs.update(
                pixel_values=torch.full((1, size, 2), float(request_marker), device="cpu"),
                input_features=torch.full((1, size, 3), float(request_marker), device="cpu"),
                input_features_mask=torch.ones((1, size), dtype=torch.long, device="cpu"),
            )
            assert len(request.media) == 2
        self.preparations.append((state["request_index"], [q.id for q in questions]))
        self.alive.append(weakref.ref(ids))
        self.peak_prepared = max(
            self.peak_prepared, sum(value() is not None for value in self.alive)
        )
        return inputs, slots, torch.tensor([len(q.labels()) for q in questions], device="cpu")

    def backbone(self, *, use_cache, return_dict, **inputs):
        assert use_cache is False and return_dict is True
        assert all(value.device.type == "cpu" for value in inputs.values())
        batch, length = inputs["input_ids"].shape
        positions = torch.arange(length, device="cpu")
        # Same causal/window/key-padding rule witnessed in the installed native
        # mask constructors. This is a mask assertion, not an attention forward.
        visible = (positions[:, None] >= positions[None, :]) & (
            (positions[:, None] - positions[None, :]) < self.window
        )
        visible = visible.unsqueeze(0) & inputs["attention_mask"].bool().unsqueeze(1)
        all_masked = ~visible.any(-1)
        if self.enforce_safe_mask:
            assert not all_masked.any(), "fully masked padding query would reach MPS SDPA"
        if "pixel_values" in inputs:
            markers = inputs["input_ids"][:, 0]
            for name in ("pixel_values", "input_features"):
                for index in range(batch):
                    assert torch.all(inputs[name][index] == markers[index])
            assert torch.all(inputs["input_features_mask"] == 1)
        self.forwards.append(
            {
                "batch": batch,
                "lengths": inputs["attention_mask"].sum(-1).tolist(),
                "ids": inputs["input_ids"].clone(),
                "all_masked_rows": all_masked.sum(-1).tolist(),
                "media_shape": tuple(inputs["pixel_values"].shape[1:])
                if "pixel_values" in inputs
                else None,
            }
        )
        return SimpleNamespace(last_hidden_state=inputs["input_ids"].float().unsqueeze(-1) / 1024)


def sequential_answers(request, *, readout="candidate"):
    reference = CPUBackboneModel(device="cpu")
    answers = {}
    for question in request.questions:
        part = request.model_copy(update={"questions": [question]})
        answers.update(predict_batch(reference, [part], readout=readout)[0]["answers"])
    return answers


@pytest.mark.parametrize("readout", ["candidate", "full"])
@pytest.mark.parametrize("long_length,expected", [(5, [2]), (6, [1, 1]), (9, [1, 1])])
def test_native_window_boundary_prevents_all_masked_padding_queries(readout, long_length, expected):
    model = CPUBackboneModel(window=4)
    request = request_for([2, long_length])
    before = copy.deepcopy(request.model_dump())
    result = predict_batch(model, [request], readout=readout)[0]
    assert [row["batch"] for row in model.forwards] == expected
    assert result["answers"] == sequential_answers(request, readout=readout)
    assert request.model_dump() == before
    assert result["execution"]["batch_sizes"] == expected
    assert result["execution"]["forward_calls"] == len(expected)
    assert result["execution"]["mps_sliding_window_batch_guard"] == 4
    assert all(not any(row["all_masked_rows"]) for row in model.forwards)


def test_partition_keeps_prepared_and_answer_order_without_length_sorting():
    model = CPUBackboneModel(window=4, device="MPS")
    request = request_for([3, 7, 4, 6, 2])
    result = predict_batch(model, [request])[0]
    assert [row["lengths"] for row in model.forwards] == [[3], [7, 4, 6], [2]]
    assert [
        int(row[mask.sum() - 1])
        for batch in model.forwards
        for row, mask in zip(batch["ids"], batch["ids"].ne(0), strict=True)
    ] == [1000, 1001, 1002, 1003, 1004]
    assert list(result["answers"]) == [q.id for q in request.questions]
    assert result["answers"] == sequential_answers(request)


def test_uses_smallest_actual_per_layer_window_instead_of_global_fallback():
    model = CPUBackboneModel(window=99, per_layer=[4, 8])
    result = predict_batch(model, [request_for([2, 6])])[0]
    assert result["execution"]["mps_sliding_window_batch_guard"] == 4
    assert result["execution"]["batch_sizes"] == [1, 1]


@pytest.mark.parametrize("device,sliding", [("cpu", True), ("cuda:0", True), ("mps", False)])
def test_non_mps_or_no_native_sliding_attention_keeps_original_batching(device, sliding):
    model = CPUBackboneModel(device=device, sliding=sliding)
    result = predict_batch(model, [request_for([2, 9])])[0]
    assert result["execution"]["batch_sizes"] == [2]
    assert "mps_sliding_window_batch_guard" not in result["execution"]
    assert model.forwards[0]["all_masked_rows"][0] > 0


def test_cpu_path_does_not_inspect_a_sliding_config():
    model = CPUBackboneModel(device="cpu")

    def unavailable():
        pytest.fail("CPU batching must not inspect MPS guard configuration")

    model.lm.config.get_text_config = unavailable
    assert predict_batch(model, [request_for([2, 9])])[0]["execution"]["batch_sizes"] == [2]


def test_cross_request_compatible_media_is_preserved_and_incompatible_shapes_stay_separate():
    model = CPUBackboneModel(window=4)
    requests = [
        request_for([3, 7], request_index=0, media_shape=2),
        request_for([4, 6], request_index=1, media_shape=2),
        request_for([5, 9], request_index=2, media_shape=3),
    ]
    before = [copy.deepcopy(request.model_dump()) for request in requests]
    results = predict_batch(model, requests)
    assert [row["lengths"] for row in model.forwards] == [[3], [7, 4, 6], [5], [9]]
    assert [row["media_shape"] for row in model.forwards] == [(2, 2), (2, 2), (3, 2), (3, 2)]
    assert [request.model_dump() for request in requests] == before
    for request, result in zip(requests, results, strict=True):
        assert list(result["answers"]) == [q.id for q in request.questions]
        assert result["answers"] == sequential_answers(request)


@pytest.mark.parametrize("readout", ["candidate", "full"])
def test_causal_requests_remain_whole_when_cross_request_padding_is_split(readout):
    model = CPUBackboneModel(window=4)
    requests = [
        request_for([2, 3, 4], request_index=0, causal_length=6),
        request_for([2, 3, 4], request_index=1, causal_length=10),
    ]
    results = predict_batch(model, requests, independent=False, readout=readout)
    assert model.preparations == [(0, ["q0", "q1", "q2"]), (1, ["q0", "q1", "q2"])]
    assert [row["batch"] for row in model.forwards] == [1, 1]
    reference = CPUBackboneModel(device="cpu")
    for request, result in zip(requests, results, strict=True):
        expected = predict_batch(reference, [request], independent=False, readout=readout)[0]
        assert result["answers"] == expected["answers"]
        assert list(result["answers"]) == [q.id for q in request.questions]
        assert result["execution"]["independent_questions"] is False


def test_mps_splits_preserve_eight_prompt_preparation_and_forward_caps():
    model = CPUBackboneModel(window=4)
    requests = [request_for([2, 6, 3, 7, 4, 8, 5, 9, 2], request_index=index) for index in range(3)]
    results = predict_batch(model, requests)
    assert model.peak_prepared <= 8
    assert len(model.preparations) == 27  # One prepare per question; no retry path.
    assert sum(row["batch"] for row in model.forwards) == 27
    assert max(row["batch"] for row in model.forwards) <= 8
    for row in model.forwards:
        assert max(row["lengths"]) - min(row["lengths"]) < 4
    for request, result in zip(requests, results, strict=True):
        assert result["answers"] == sequential_answers(request)
        assert result["execution"]["forward_calls"] == len(model.forwards)
        assert result["execution"]["max_forward_batch_size"] == 8
        assert result["execution"]["max_prepared_prompts"] == 8


def test_invalid_declared_native_sliding_window_fails_before_preparation():
    model = CPUBackboneModel(window=0)
    with pytest.raises(ValueError, match="positive native sliding windows"):
        predict_batch(model, [request_for([2, 6])])
    assert not model.preparations and not model.forwards
