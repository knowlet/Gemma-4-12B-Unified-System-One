"""Native tiny Gemma CPU checks; synthetic prompts are never benchmark evidence."""

import copy
import json
from types import SimpleNamespace

import pytest
import test_browser_prefix as prefix_tests
from test_jev_browser_fit_readout import FIT, collection, digest

from s1 import unified
from s1.contracts import DecisionRequest, ImageInput
from s1.evaluation import gemma
from s1.evaluation.datasets import EvaluationCase

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.inference

FEATURE_KEYS = {
    "case_id",
    "group_id",
    "split",
    "type",
    "question_id",
    "labels",
    "target",
    "logits",
    "hidden",
    "sequence_tokens",
}


@pytest.fixture
def model():
    return prefix_tests.model.__wrapped__()


def capture_fixture(model):
    """Short exact CPU processor rows, including original right padding."""
    cases, prepared = [], {}
    sizes = iter(((19, 2, 2), (85, 5, 52), (24, 0, 5)))
    for case_index, question_count in enumerate((1, 2)):
        questions, targets = [], {}
        for question_index in range(question_count):
            attended, extra, count = next(sizes)
            question_id = f"question-{case_index}-{question_index}"
            criteria = {f"target-{index}": f"Observed target {index}" for index in range(count)}
            questions.append(
                {
                    "id": question_id,
                    "instructions": "Choose an observed target",
                    "criteria": criteria,
                }
            )
            targets[question_id] = {
                label: (0.75 if index == 0 else 0.25 / (count - 1))
                for index, label in enumerate(criteria)
            }
            ids = (torch.arange(attended + extra) % 29 + 3).unsqueeze(0)
            mask = torch.ones_like(ids)
            if extra:
                ids[:, -extra:] = 0
                mask[:, -extra:] = 0
            inputs = {
                "input_ids": ids,
                "attention_mask": mask,
                "mm_token_type_ids": torch.zeros_like(ids),
                "token_type_ids": torch.full_like(ids, 7),
                "position_ids": torch.arange(ids.shape[1]).unsqueeze(0),
            }
            prepared[question_id] = (inputs, ids.shape[1])
        request = DecisionRequest(state=f"Observed case {case_index}", questions=questions)
        cases.append(
            EvaluationCase(
                id=f"case-{case_index}",
                group_id=f"source-{case_index}",
                split="train",
                request=request,
                gold={q.id: q.labels()[0] for q in request.questions},
                soft_gold=targets,
            )
        )
    prepare_calls = []

    def prepare(request, question):
        assert any(request is case.request and question in request.questions for case in cases)
        prepare_calls.append(question.id)
        return prepared[question.id]

    backend = SimpleNamespace(model=model, _prepare_question=prepare)
    return SimpleNamespace(
        backend=backend,
        cases=cases,
        prepared=prepared,
        prepare_calls=prepare_calls,
    )


def ordered_questions(fixture):
    return [(case, question) for case in fixture.cases for question in case.request.questions]


def native_features(fixture):
    output = []
    with torch.inference_mode():
        for _case, question in ordered_questions(fixture):
            inputs, _length = fixture.prepared[question.id]
            result = fixture.backend.model.backbone(**inputs, use_cache=False, return_dict=True)
            slot = int(inputs["attention_mask"][0].nonzero()[-1].item())
            hidden = result.last_hidden_state[:, slot]
            logits = unified.project_candidates(
                hidden,
                fixture.backend.model.head,
                fixture.backend.model.letters,
                torch.tensor([len(question.labels())]),
                fixture.backend.model.softcap,
            )
            output.append((hidden[0].float().clone(), logits[0, : len(question.labels())].tolist()))
    return output


def forbid_cuda(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("CPU feature capture attempted a CUDA operation")

    for name in (
        "synchronize",
        "reset_peak_memory_stats",
        "mem_get_info",
        "get_device_name",
        "memory_allocated",
        "memory_reserved",
        "max_memory_allocated",
        "max_memory_reserved",
    ):
        monkeypatch.setattr(torch.cuda, name, forbidden)


def assert_feature_rows(fixture, rows, expected, *, dtype=torch.float32):
    assert len(rows) == len(expected) == len(ordered_questions(fixture))
    tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-5
    for row, (case, question), (hidden, logits) in zip(
        rows, ordered_questions(fixture), expected, strict=True
    ):
        assert set(row) == FEATURE_KEYS
        assert row["case_id"] == case.id and row["group_id"] == case.group_id
        assert row["question_id"] == question.id and row["type"] == question.type
        assert row["split"] == case.split and row["labels"] == question.labels()
        assert row["target"] == [case.soft_gold[question.id][label] for label in question.labels()]
        assert row["sequence_tokens"] == fixture.prepared[question.id][1]
        assert row["hidden"].dtype == torch.float32 and row["hidden"].device.type == "cpu"
        assert row["hidden"].shape == hidden.shape and not row["hidden"].requires_grad
        torch.testing.assert_close(row["hidden"], hidden, atol=tolerance, rtol=tolerance)
        assert row["logits"] == pytest.approx(logits, abs=tolerance, rel=tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_features_match_single_forwards_and_real_two_row_collation(
    model, monkeypatch, dtype
):
    model.lm.to(dtype=dtype)
    fixture = capture_fixture(model)
    expected = native_features(fixture)
    originals = {
        question_id: {key: value.clone() for key, value in inputs.items()}
        for question_id, (inputs, _length) in fixture.prepared.items()
    }
    forbid_cuda(monkeypatch)
    calls, projections, collations = [], [], []
    actual_project, actual_collate = unified.project_candidates, gemma._collate

    def project(hidden, head, letters, counts, softcap):
        projections.append((tuple(hidden.shape), counts.tolist(), softcap))
        return actual_project(hidden, head, letters, counts, softcap)

    def collate(prepared, pad_id):
        collations.append(len(prepared))
        return actual_collate(prepared, pad_id)

    def before(_module, _args, kwargs):
        assert kwargs["use_cache"] is False and kwargs["return_dict"] is True
        assert 1 <= kwargs["input_ids"].shape[0] <= 2
        calls.append({key: kwargs[key].clone() for key in originals[next(iter(originals))]})

    monkeypatch.setattr(unified, "project_candidates", project)
    monkeypatch.setattr(gemma, "_collate", collate)
    hook = model.backbone.register_forward_pre_hook(before, with_kwargs=True)
    try:
        default_evidence = {}
        default = FIT.feature_rows(fixture.backend, fixture.cases, evidence=default_evidence)
        assert [call["input_ids"].shape[0] for call in calls] == [1, 1, 1]
        calls.clear()
        projections.clear()
        evidence = {}
        batched = FIT.feature_rows(
            fixture.backend,
            fixture.cases,
            feature_batch_size=2,
            evidence=evidence,
        )
    finally:
        hook.remove()
    assert_feature_rows(fixture, default, expected, dtype=dtype)
    assert_feature_rows(fixture, batched, expected, dtype=dtype)
    assert [call["input_ids"].shape[0] for call in calls] == [2, 1]
    assert 2 in collations
    assert [counts for _shape, counts, _softcap in projections] == [[2, 52], [5]]
    assert all(
        shape[1] == 16 and softcap == model.softcap for shape, _counts, softcap in projections
    )
    for index, (_case, question) in enumerate(ordered_questions(fixture)[:2]):
        width = fixture.prepared[question.id][1]
        for key, original in originals[question.id].items():
            torch.testing.assert_close(
                calls[0][key][index : index + 1, :width], original, atol=0, rtol=0
            )
            padded = calls[0][key][index, width:]
            assert (padded == (model.tok.pad_token_id if key == "input_ids" else 0)).all()
    for question_id, (inputs, _length) in fixture.prepared.items():
        for key, value in inputs.items():
            torch.testing.assert_close(value, originals[question_id][key], atol=0, rtol=0)
    assert evidence["status"] == default_evidence["status"] == "completed"
    assert evidence["forward_calls"] == 2 and evidence["batch_sizes"] == [2, 1]
    assert evidence["completed_row_indices"] == [0, 1, 2]
    original = sum(length for _inputs, length in fixture.prepared.values())
    logical = sum(
        int(inputs["attention_mask"].sum()) for inputs, _length in fixture.prepared.values()
    )
    physical = sum(call["input_ids"].numel() for call in calls)
    assert evidence["logical_input_tokens"] == logical
    assert evidence["original_input_tokens"] == original
    assert evidence["physical_input_tokens"] == physical
    assert evidence["padding_input_tokens"] == physical - logical > 0
    assert evidence["original_padding_input_tokens"] == original - logical == 7
    assert evidence["collation_padding_input_tokens"] == physical - original
    assert default_evidence["padding_input_tokens"] == original - logical
    assert default_evidence["collation_padding_input_tokens"] == 0
    assert evidence["elapsed_ms"] > 0
    assert evidence["peak_resources"]["process_max_rss_bytes"] > 0
    assert not any("cuda" in key for key in evidence["peak_resources"])
    assert len(evidence["row_records"]) == 3
    for forward, indices in zip(evidence["forwards"], ([0, 1], [2]), strict=True):
        assert forward["row_indices"] == indices and forward["batch_size"] == len(indices)
        assert forward["case_ids"] == [ordered_questions(fixture)[index][0].id for index in indices]
        assert forward["question_ids"] == [
            ordered_questions(fixture)[index][1].id for index in indices
        ]
        assert forward["original_lengths"] == [
            fixture.prepared[ordered_questions(fixture)[index][1].id][1] for index in indices
        ]
        assert forward["answer_slots"] == ([18, 84] if indices == [0, 1] else [23])
        assert len(forward["input_sha256"]) == 64
        assert (
            forward["physical_input_tokens"] - forward["logical_input_tokens"]
            == forward["padding_input_tokens"]
        )
        assert forward["status"] in ("returned", "completed")
        assert forward["latency_scope"] == "host_backbone_call_without_device_synchronize"
    json.dumps(evidence, allow_nan=False)


@pytest.mark.parametrize("change", ["optional_key", "optional_dtype"])
def test_valid_incompatible_processor_rows_split_without_dropping_tensors(
    model, monkeypatch, change
):
    fixture = capture_fixture(model)
    second = fixture.prepared[ordered_questions(fixture)[1][1].id][0]
    if change == "optional_key":
        del second["token_type_ids"]
    else:
        second["attention_mask"] = second["attention_mask"].bool()
    expected = native_features(fixture)
    calls = []

    def before(_module, _args, kwargs):
        calls.append(
            {key: value.clone() for key, value in kwargs.items() if torch.is_tensor(value)}
        )

    forbid_cuda(monkeypatch)
    hook = model.backbone.register_forward_pre_hook(before, with_kwargs=True)
    try:
        evidence = {}
        rows = FIT.feature_rows(
            fixture.backend, fixture.cases, feature_batch_size=2, evidence=evidence
        )
    finally:
        hook.remove()
    assert_feature_rows(fixture, rows, expected)
    assert evidence["batch_sizes"] == [1, 1, 1]
    for call, (_case, question) in zip(calls, ordered_questions(fixture), strict=True):
        original = fixture.prepared[question.id][0]
        assert set(call) == set(original)
        for key, value in original.items():
            assert call[key].dtype == value.dtype
            torch.testing.assert_close(call[key], value, atol=0, rtol=0)


@pytest.mark.parametrize("invalid", [None, True, False, 0, -1, 3, 1.0, "2"])
def test_invalid_feature_batch_size_is_rejected_before_touching_backend(invalid):
    class Untouchable:
        def __getattribute__(self, _name):
            raise AssertionError("invalid batch size touched the backend")

    with pytest.raises(ValueError):
        FIT.feature_rows(Untouchable(), [], feature_batch_size=invalid)


@pytest.mark.parametrize(
    "change",
    [
        "mask_value",
        "mask_gap",
        "all_masked",
        "missing_mask",
        "media_tensor",
        "shape",
        "reported_length",
        "context",
        "float_ids",
        "vocabulary",
        "non_cpu",
    ],
)
def test_entire_bounded_chunk_is_validated_on_cpu_before_transfer_or_forward(
    model, monkeypatch, change
):
    fixture = capture_fixture(model)
    question_id = ordered_questions(fixture)[1][1].id
    inputs, length = fixture.prepared[question_id]
    if change == "mask_value":
        inputs["attention_mask"][0, 4] = 2
    elif change == "mask_gap":
        inputs["attention_mask"][0, 4] = 0
    elif change == "all_masked":
        inputs["attention_mask"].zero_()
    elif change == "missing_mask":
        del inputs["attention_mask"]
    elif change == "media_tensor":
        inputs["pixel_values"] = torch.zeros(1, 3, 2, 2)
    elif change == "shape":
        inputs["token_type_ids"] = inputs["token_type_ids"][:, :-1]
    elif change == "reported_length":
        fixture.prepared[question_id] = (inputs, length - 1)
    elif change == "context":
        model.max_context = 40
    elif change == "float_ids":
        inputs["input_ids"] = inputs["input_ids"].float()
    elif change == "vocabulary":
        inputs["input_ids"][0, 4] = 256
    else:
        inputs["position_ids"] = torch.empty((1, length), device="meta", dtype=torch.long)
    forbid_cuda(monkeypatch)
    tensor_ids = {
        id(value) for tensors, _length in fixture.prepared.values() for value in tensors.values()
    }
    calls, transfers = [], []
    actual_to = torch.Tensor.to

    def tracked_to(tensor, *args, **kwargs):
        if id(tensor) in tensor_ids:
            transfers.append(id(tensor))
        return actual_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", tracked_to)
    hook = model.backbone.register_forward_pre_hook(lambda *_args: calls.append(None))
    evidence = {}
    try:
        with pytest.raises(ValueError) as error:
            FIT.feature_rows(
                fixture.backend, fixture.cases, feature_batch_size=2, evidence=evidence
            )
    finally:
        hook.remove()
    assert calls == transfers == []
    assert error.value.feature_capture_evidence is evidence
    assert error.value.completed_feature_rows == []
    assert evidence["status"] == "failed" and evidence["forward_calls"] == 0
    assert evidence["completed_row_indices"] == []
    json.dumps(evidence, allow_nan=False)


def test_media_is_rejected_before_preparation_or_native_forward(model, monkeypatch):
    fixture = capture_fixture(model)
    fixture.cases[-1].request.media = [ImageInput(data="not-decoded")]
    forbid_cuda(monkeypatch)
    with pytest.raises(ValueError) as error:
        FIT.feature_rows(fixture.backend, fixture.cases, feature_batch_size=2)
    assert fixture.prepare_calls == []
    assert error.value.feature_capture_evidence["forward_calls"] == 0
    assert error.value.completed_feature_rows == []


@pytest.mark.parametrize("failure", ["prepare", "context", "transfer", "forward", "projection"])
def test_later_failures_preserve_genuine_completed_cpu_rows_and_attempt_evidence(
    model, monkeypatch, failure
):
    fixture = capture_fixture(model)
    expected = native_features(fixture)
    forbid_cuda(monkeypatch)
    original_prepare, original_project = (
        fixture.backend._prepare_question,
        unified.project_candidates,
    )
    prepare_count, project_count, forward_count = 0, 0, 0
    if failure == "context":
        model.max_context = 90

    def prepare(request, question):
        nonlocal prepare_count
        prepare_count += 1
        if prepare_count == 3 and failure == "prepare":
            raise OSError("late preparation failure")
        inputs, length = original_prepare(request, question)
        if prepare_count == 3 and failure == "context":
            extra = model.max_context + 1 - length
            inputs = {
                key: torch.nn.functional.pad(
                    value, (0, extra), value=1 if key == "attention_mask" else 0
                )
                for key, value in inputs.items()
            }
            inputs["position_ids"] = torch.arange(length + extra).unsqueeze(0)
            return inputs, length + extra
        return inputs, length

    def project(*args, **kwargs):
        nonlocal project_count
        project_count += 1
        if project_count == 2 and failure == "projection":
            raise RuntimeError("late projection failure")
        return original_project(*args, **kwargs)

    def before(_module, _args, _kwargs):
        nonlocal forward_count
        forward_count += 1
        if forward_count == 2 and failure == "forward":
            raise RuntimeError("late native forward failure")

    if failure == "transfer":
        final_inputs = fixture.prepared[ordered_questions(fixture)[-1][1].id][0]
        actual_to = torch.Tensor.to

        def transfer(tensor, *args, **kwargs):
            target = kwargs.get("device", args[0] if args else None)
            if tensor is final_inputs["input_ids"] and str(target) == "cpu":
                raise RuntimeError("late CPU transfer failure")
            return actual_to(tensor, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "to", transfer)

    fixture.backend._prepare_question = prepare
    monkeypatch.setattr(unified, "project_candidates", project)
    hook = model.backbone.register_forward_pre_hook(before, with_kwargs=True)
    evidence = {}
    try:
        with pytest.raises((OSError, RuntimeError, ValueError)) as error:
            FIT.feature_rows(
                fixture.backend, fixture.cases, feature_batch_size=2, evidence=evidence
            )
    finally:
        hook.remove()
    trace = error.value.feature_capture_evidence
    assert trace is evidence and trace["status"] == "failed"
    assert trace["completed_row_indices"] == [0, 1]
    completed = error.value.completed_feature_rows
    assert len(completed) == 2
    first_chunk_fixture = copy.copy(fixture)
    first_chunk_fixture.cases = [
        fixture.cases[0],
        fixture.cases[1].model_copy(
            update={
                "request": fixture.cases[1].request.model_copy(
                    update={
                        "questions": fixture.cases[1].request.questions[:1],
                    }
                ),
            }
        ),
    ]
    assert_feature_rows(first_chunk_fixture, completed, expected[:2])
    expected_calls = 2 if failure in ("forward", "projection") else 1
    assert trace["forward_calls"] == forward_count == expected_calls
    assert trace["batch_sizes"] == ([2, 1] if expected_calls == 2 else [2])
    assert trace["forwards"][0]["row_indices"] == [0, 1]
    assert trace["forwards"][0]["original_lengths"] == [21, 90]
    if expected_calls == 2:
        attempted = trace["forwards"][-1]
        assert attempted["row_indices"] == [2] and attempted["original_lengths"] == [24]
        assert attempted["answer_slots"] == [23] and len(attempted["input_sha256"]) == 64
    if failure == "transfer":
        assert trace["failure_stage"] == "transfer"
        assert trace["active_chunk"]["row_indices"] == [2]
        assert trace["active_chunk"]["original_lengths"] == [24]
        assert trace["active_chunk"]["answer_slots"] == [23]
        assert len(trace["active_chunk"]["input_sha256"]) == 64
        assert len(trace["forwards"]) == 1
        assert trace["logical_input_tokens"] == 104
        assert trace["original_input_tokens"] == 111
        assert trace["physical_input_tokens"] == 180
        assert trace["padding_input_tokens"] == 76
    assert trace["elapsed_ms"] > 0 and trace["peak_resources"]["process_max_rss_bytes"] > 0
    json.dumps(trace, allow_nan=False)


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("failure", ["forward", "capture_receipt"])
def test_fit_preserves_native_partial_features_and_failed_receipt(
    model, monkeypatch, tmp_path, error_type, failure
):
    """Use the real tiny backbone; short CPU inputs isolate receipt correctness."""
    from s1 import browser_policy

    train = collection(tmp_path, "train", group="CPU training episode")
    development = collection(tmp_path, "development", group="CPU development episode")
    output = tmp_path / "partial-fit"
    name = "knowlet/Gemma-4-12B-Unified-System-One"
    revision = "a66f836b56605039fe040f330180e336d19b3362"
    model.name, model.revision = name, revision
    model.quantization_details = {"mode": "nf4", "diagnostic_native_tiny_cpu": True}
    initial_head = model.head.weight.detach().clone()
    prepared_calls, native_calls, completed_rows = [], [], []
    primary = error_type("CPU native capture interrupted on development")

    def load(requested_name, **options):
        assert requested_name == name and options["revision"] == revision
        assert options["device"] == "cuda"
        assert (output / "sources/manifest.json").is_file()
        assert (output / "inputs/train/cases.jsonl").read_bytes() == train.read_bytes()
        assert (output / "inputs/development/cases.jsonl").read_bytes() == development.read_bytes()
        assert str(model.device) == "cpu"
        return model

    def backend(loaded, *, prompt_mode):
        assert loaded is model and prompt_mode == "grounded"

        def prepare(request, question):
            assert question in request.questions
            prepared_calls.append(question.id)
            width = 17 + len(prepared_calls)
            ids = (torch.arange(width) % 29 + 3).unsqueeze(0)
            return {
                "input_ids": ids,
                "attention_mask": torch.ones_like(ids),
                "mm_token_type_ids": torch.zeros_like(ids),
            }, width

        return SimpleNamespace(
            model=model,
            prompt_variant="grounded_user_question_v1",
            _prepare_question=prepare,
        )

    actual_capture = FIT.feature_rows

    def capture(*args, **kwargs):
        rows = actual_capture(*args, **kwargs)
        completed_rows.extend(rows)
        return rows

    def before(_module, _args, kwargs):
        native_calls.append(tuple(kwargs["input_ids"].shape))
        if len(native_calls) == 2 and failure == "forward":
            raise primary

    if failure == "capture_receipt":
        from pathlib import Path

        actual_write = Path.write_text
        ledger_writes = 0

        def write(path, *args, **kwargs):
            nonlocal ledger_writes
            if path == output / "feature_capture.json":
                ledger_writes += 1
                if ledger_writes == 1:
                    raise primary
            return actual_write(path, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", write)

    monkeypatch.setattr(unified, "UnifiedDecisionModel", load)
    monkeypatch.setattr(browser_policy, "BrowserDecisionBackend", backend)
    monkeypatch.setattr(FIT, "feature_rows", capture)
    monkeypatch.setattr(torch.backends.cuda, "enable_cudnn_sdp", lambda _enabled: None)
    forbid_cuda(monkeypatch)
    hook = model.backbone.register_forward_pre_hook(before, with_kwargs=True)
    try:
        with pytest.raises(error_type) as caught:
            FIT.fit(
                train_cases=train,
                development_cases=development,
                output_dir=output,
                feature_batch_size=2,
            )
    finally:
        hook.remove()
    assert caught.value is primary
    expected_forwards = 2 if failure == "forward" else 1
    assert len(native_calls) == expected_forwards and len(completed_rows) == 1
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["stage"] == "failed" and receipt["error"]["type"] == error_type.__name__
    assert receipt["source_verification"]["success"] is True
    assert receipt["input_verification"]["train"]["success"] is True
    assert receipt["input_verification"]["development"]["success"] is True
    assert receipt["backbone_forward_calls_this_run"] == expected_forwards
    partial = receipt["partial_feature_capture"]
    assert (partial["train_rows"], partial["development_rows"]) == (1, 0)
    assert partial["sha256"] == digest((output / partial["artifact_path"]).read_bytes())
    features = torch.load(output / partial["artifact_path"], weights_only=True, map_location="cpu")
    assert features["development"] == []
    assert len(features["train"]) == 1 and set(features["train"][0]) == FEATURE_KEYS
    torch.testing.assert_close(
        features["train"][0]["hidden"], completed_rows[0]["hidden"], atol=0, rtol=0
    )
    assert features["train"][0]["logits"] == completed_rows[0]["logits"]
    assert features["train"][0]["hidden"].device.type == "cpu"
    trace_bytes = (output / "feature_capture.json").read_bytes()
    assert receipt["feature_capture_sha256"] == digest(trace_bytes)
    trace = json.loads(trace_bytes)
    assert trace == receipt["feature_capture"]
    assert trace["source_identity_sha256"] == receipt["source_manifest"]["identity_sha256"]
    train_capture = trace["split_captures"]["train"]
    assert train_capture["status"] == "completed"
    assert train_capture["completed_row_indices"] == [0]
    if failure == "forward":
        dev_capture = trace["split_captures"]["development"]
        assert dev_capture["failure_stage"] == "forward" and dev_capture["status"] == "failed"
        assert dev_capture["forwards"][0]["status"] == "failed"
        assert dev_capture["forwards"][0]["error"]["type"] == error_type.__name__
        assert dev_capture["completed_row_indices"] == []
    else:
        assert list(trace["split_captures"]) == ["train"] and ledger_writes == 2
    assert not (output / "features.pt").exists()
    assert not (output / "control").exists() and not (output / "learned").exists()
    torch.testing.assert_close(model.head.weight, initial_head, atol=0, rtol=0)
