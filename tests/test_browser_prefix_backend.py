"""Independent prefix backend tests use the genuine tiny Gemma CPU fixture."""

import json
from types import SimpleNamespace

import pytest
from test_browser_prefix import model as model  # Reuse the genuine CPU model fixture.

from s1.browser_policy import BrowserDecisionBackend
from s1.browser_prefix import BrowserPrefixUnsupported
from s1.contracts import DecisionRequest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.inference


def requests():
    state = {
        "page": {"title": "Observed controls", "text": "Shared page content. " * 8},
        "recent_actions": [],
    }
    instructions = json.dumps({"goal": "Open an observed control after applying changes"})
    result = []
    for index, counts in enumerate(([2, 5], [3, 52], [2])):
        result.append(
            DecisionRequest(
                state=state,
                questions=[
                    {
                        "id": f"head-{head}",
                        "instructions": instructions,
                        "criteria": {
                            f"node-{option}": f"Observed control {index}-{head}-{option}"
                            for option in range(count)
                        },
                    }
                    for head, count in enumerate(counts)
                ],
            )
        )
    return result


@pytest.mark.parametrize("mode", ["user", "grounded", "goal-first"])
def test_prefix_backend_aggregates_multiquestion_rows_matches_full_policy_and_counts_work_once(
    model, mode
):
    original = requests()
    snapshots = [request.model_dump() for request in original]
    baseline = BrowserDecisionBackend(model, prompt_mode=mode)
    expected = [baseline.predict(request) for request in original]
    backend = BrowserDecisionBackend(model, prompt_mode=mode, prefix_cache=True)
    head, letters, temperature = model.head, model.letters.clone(), model.temperature
    actual = backend.predict_batch(original)
    assert [request.model_dump() for request in original] == snapshots
    assert model.head is head and model.temperature == temperature
    torch.testing.assert_close(model.letters, letters, atol=0, rtol=0)
    assert backend.metadata["prefix_cache"] is True
    assert backend.metadata["prefix_cache_policy"] == "shared_prefix_dynamic_cache_v1"
    assert len(actual) == len(original) == 3
    for response, reference, request in zip(actual, expected, original, strict=True):
        assert response["model"] == backend.name == model.name
        assert response["prompt_variant"] == backend.prompt_variant
        assert response["temperature"] == temperature
        assert list(response["answers"]) == [question.id for question in request.questions]
        for question in request.questions:
            answer = response["answers"][question.id]
            prior = reference["answers"][question.id]
            assert answer["probabilities"] == pytest.approx(prior["probabilities"], abs=1e-5)
            assert answer["choice"] == prior["choice"]
            assert list(answer["probabilities"]) == question.labels()
        execution = response["execution"]
        assert execution["context_input_tokens"] == sum(execution["question_input_tokens"].values())
        assert execution["computed_input_tokens"] == response["usage"]["input_tokens"]
        assert execution["input_tokens"] == execution["computed_input_tokens"]
        assert execution["context_input_tokens"] == sum(execution["sequence_tokens"])
        assert execution["sequence_tokens_scope"] == "full_original_question_contexts"
        assert execution["batch_forward_calls"] == 6
        assert execution["max_forward_batch_size"] == 1
        assert execution["max_live_cache_forks"] == 1
        assert execution["forward_calls_scope"] == "allocated_once_per_shared_browser_batch"
        assert execution["latency_scope"] == response["latency_scope"] == "whole_browser_batch"
    assert [response["passes"] for response in actual] == [3, 2, 1]
    assert sum(response["execution"]["forward_calls"] for response in actual) == 6
    assert sum("prefix" in response for response in actual) == 1
    ledger = actual[0]["prefix"]
    assert ledger["row_to_request_index"] == [0, 0, 1, 1, 2]
    assert ledger["completed_response_indices"] == [0, 1, 2, 3, 4]
    assert ledger["request_aggregation"]["completed_response_indices"] == [0, 1, 2]
    assert ledger["request_aggregation"]["index_scope"] == "original_backend_request_indices"
    assert ledger["computed_input_tokens"] == sum(
        response["usage"]["input_tokens"] for response in actual
    )
    assert ledger["full_context_input_tokens"] == sum(
        response["execution"]["context_input_tokens"] for response in actual
    )
    assert ledger["computed_input_tokens"] < ledger["full_context_input_tokens"]
    for response in actual:
        assert (
            response["execution"]["batch_computed_input_tokens"] == ledger["computed_input_tokens"]
        )
        assert (
            response["execution"]["batch_full_context_input_tokens"]
            == ledger["full_context_input_tokens"]
        )
    assert all(
        "prefix" not in saved for saved in ledger["request_aggregation"]["completed_responses"]
    )
    assert len(actual[0]["execution"]["forward_refs"]) == 3
    assert len(actual[1]["execution"]["forward_refs"]) == 3
    assert len(actual[2]["execution"]["forward_refs"]) == 2
    json.dumps(actual, allow_nan=False)


def test_prefix_single_predict_routes_through_the_same_ordered_batch_path(model, monkeypatch):
    backend = BrowserDecisionBackend(model, prompt_mode="user", prefix_cache=True)
    batch = backend.predict_batch
    seen = []

    def record(supplied):
        seen.append(supplied)
        return batch(supplied)

    monkeypatch.setattr(backend, "predict_batch", record)
    request = requests()[0]
    actual = backend.predict(request)
    assert seen == [[request]]
    assert list(actual["answers"]) == [question.id for question in request.questions]
    assert actual["prefix"]["row_to_request_index"] == [0, 0]
    assert actual["passes"] == 3


@pytest.mark.parametrize("mode", ["native", "reasoned"])
def test_prefix_unsupported_modes_reject_before_any_model_access(mode):
    with pytest.raises(ValueError, match="prefix_cache supports only"):
        BrowserDecisionBackend(SimpleNamespace(), prompt_mode=mode, prefix_cache=True)


@pytest.mark.parametrize("flag", [None, 1, "true"])
def test_prefix_cache_requires_a_real_boolean(flag):
    with pytest.raises(ValueError, match="explicit boolean"):
        BrowserDecisionBackend(SimpleNamespace(), prefix_cache=flag)


def test_default_backend_stays_on_original_full_context_policy(model):
    backend = BrowserDecisionBackend(model, prompt_mode="user")
    result = backend.predict(requests()[0])
    assert backend.prefix_cache is False
    assert "prefix_cache" not in backend.metadata
    assert "prefix" not in result and "prefix_cache" not in result
    assert result["passes"] == 2


@pytest.mark.parametrize("learned", [False, True])
def test_prefix_backend_uses_genuine_sidecar_without_changing_helper(model, tmp_path, learned):
    from s1.browser_readout import load_readout, write_readout

    model.revision = "a" * 40
    variant = BrowserDecisionBackend(model, prompt_mode="grounded").prompt_variant
    provenance = {
        "source_scope": "development_only",
        "train": {
            "ids": ["development-row"] if learned else [],
            "groups": ["development-case"] if learned else [],
            "request_fingerprints": ["1" * 64] if learned else [],
        },
        "calibration": {"ids": [], "groups": [], "request_fingerprints": []},
    }
    path = tmp_path / "readout"
    kwargs = {}
    if learned:
        kwargs = {"weight": torch.zeros(52, 16), "bias": torch.linspace(-0.3, 0.8, 52)}
    write_readout(path, model, variant, provenance=provenance, **kwargs)
    readout = load_readout(path, model, variant)
    baseline = BrowserDecisionBackend(model, prompt_mode="grounded", decision_readout=readout)
    supplied = requests()
    expected = [baseline.predict(request) for request in supplied]
    backend = BrowserDecisionBackend(
        model, prompt_mode="grounded", decision_readout=readout, prefix_cache=True
    )
    head, weight, name = model.head, model.head.weight.clone(), model.name
    actual = backend.predict_batch(supplied)
    assert model.head is head and model.name == name
    torch.testing.assert_close(model.head.weight, weight, atol=0, rtol=0)
    for response, reference in zip(actual, expected, strict=True):
        assert response["decision_readout"] == readout.telemetry()
        assert response["model"] == backend.name
        assert readout.telemetry()["weights_sha256"] in backend.name
        for question_id, answer in response["answers"].items():
            assert answer["probabilities"] == pytest.approx(
                reference["answers"][question_id]["probabilities"], abs=1e-5
            )
            assert answer["choice"] == reference["answers"][question_id]["choice"]


@pytest.mark.parametrize(
    "fail_on,completed_rows,completed_requests", [(3, [0], []), (5, [0, 1, 2], [0])]
)
def test_prefix_failure_preserves_partial_rows_and_maps_only_complete_original_requests(
    model, monkeypatch, fail_on, completed_rows, completed_requests
):
    failure = RuntimeError("diagnostic cached suffix failure")
    failure.generation = {"existing": "generation retained"}
    failure.hierarchy = {"existing": "hierarchy retained"}
    original = model.backbone.forward
    calls = []

    def failing(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == fail_on:
            raise failure
        return original(*args, **kwargs)

    monkeypatch.setattr(model.backbone, "forward", failing)
    backend = BrowserDecisionBackend(model, prompt_mode="user", prefix_cache=True)
    with pytest.raises(RuntimeError) as caught:
        backend.predict_batch(requests())
    assert caught.value is failure
    assert caught.value.generation == {"existing": "generation retained"}
    assert caught.value.hierarchy == {"existing": "hierarchy retained"}
    prefix = caught.value.prefix_evidence
    assert prefix["completed_response_indices"] == completed_rows
    assert prefix["failure_stage"] == "suffixes"
    assert prefix["branches"][-1]["failure_stage"] == "forwarding"
    evidence = caught.value.batch_evidence
    assert evidence["index_scope"] == "original_backend_request_indices"
    assert evidence["row_to_request_index"] == [0, 0, 1, 1, 2]
    assert evidence["completed_row_response_indices"] == completed_rows
    assert evidence["completed_row_responses"] == prefix["completed_responses"]
    assert evidence["completed_response_indices"] == completed_requests
    assert evidence["prefix_evidence"] == prefix
    for response in evidence["completed_responses"]:
        assert list(response["answers"]) == ["head-0", "head-1"]
    assert len(calls) == fail_on
    assert evidence["aggregation_error"] is None
    json.dumps(evidence, allow_nan=False)


def test_prefix_checks_every_row_before_forward_and_keeps_full_validation_evidence(
    model, monkeypatch
):
    backend = BrowserDecisionBackend(model, prompt_mode="user", prefix_cache=True)
    prepare = backend._prepare_question
    prepared = []

    def unsupported_last(request, question):
        inputs, length = prepare(request, question)
        prepared.append(question.id)
        if len(prepared) == 5:
            inputs["mm_token_type_ids"] = torch.ones_like(inputs["input_ids"])
        return inputs, length

    monkeypatch.setattr(backend, "_prepare_question", unsupported_last)
    hook = model.backbone.register_forward_pre_hook(
        lambda *_: pytest.fail("invalid rows forwarded")
    )
    try:
        with pytest.raises(BrowserPrefixUnsupported) as caught:
            backend.predict_batch(requests())
    finally:
        hook.remove()
    assert len(prepared) == 5
    prefix = caught.value.prefix_evidence
    assert prefix["validation_row_index"] == 4
    assert prefix["forwards"] == []
    assert prefix["failure_stage"] == "validation"
    assert caught.value.batch_evidence["completed_response_indices"] == []
    assert caught.value.batch_evidence["completed_row_response_indices"] == []
    assert caught.value.generation is caught.value.batch_evidence


def test_prefix_rejects_unsupported_backbone_without_fallback(model):
    model.backbone = SimpleNamespace()
    backend = BrowserDecisionBackend(model, prompt_mode="grounded", prefix_cache=True)
    with pytest.raises(BrowserPrefixUnsupported, match="native Gemma4Unified") as caught:
        backend.predict_batch(requests())
    assert caught.value.prefix_evidence["forwards"] == []
    assert caught.value.batch_evidence["completed_response_indices"] == []


def compact_backend_preparer(backend, monkeypatch):
    """Exercise genuine native forwards with short, ordered prepared CPU rows."""
    counter = []

    def prepare(_request, question):
        index = len(counter)
        counter.append(question.id)
        prefix, suffix = 70, (4, 79, 9, 18, 5)[index]
        ids = torch.cat(
            (
                torch.arange(prefix) % 29 + 3,
                torch.full((suffix,), 40 + index),
            )
        ).unsqueeze(0)
        mask = torch.ones_like(ids)
        # Backend's canonical _prepare_question does not pad. This diagnostic
        # keeps the last token unmasked for its slot contract; utility-level
        # tests independently cover pre-existing right padding.
        inputs = {
            "input_ids": ids,
            "attention_mask": mask,
            "mm_token_type_ids": torch.zeros_like(ids),
            "position_ids": torch.arange(ids.shape[1]).unsqueeze(0),
        }
        return inputs, ids.shape[1]

    monkeypatch.setattr(backend, "_prepare_question", prepare)


@pytest.mark.parametrize("size", [True, False, None, 0, 3, 1.0, "2"])
def test_prefix_suffix_constructor_rejects_invalid_sizes_without_model_access(size):
    with pytest.raises(ValueError):
        BrowserDecisionBackend(
            SimpleNamespace(), prompt_mode="user", prefix_cache=True, prefix_suffix_batch_size=size
        )


def test_two_row_suffix_constructor_requires_explicit_prefix_cache():
    with pytest.raises(ValueError, match="prefix_cache"):
        BrowserDecisionBackend(SimpleNamespace(), prompt_mode="user", prefix_suffix_batch_size=2)


def test_optional_two_suffix_chunks_keep_backend_original_request_aggregation(model, monkeypatch):
    original = requests()
    one = BrowserDecisionBackend(model, prompt_mode="grounded", prefix_cache=True)
    compact_backend_preparer(one, monkeypatch)
    expected = one.predict_batch(original)
    two = BrowserDecisionBackend(
        model, prompt_mode="grounded", prefix_cache=True, prefix_suffix_batch_size=2
    )
    compact_backend_preparer(two, monkeypatch)
    actual = two.predict_batch(original)
    for response, reference, request in zip(actual, expected, original, strict=True):
        for question in request.questions:
            assert response["answers"][question.id]["probabilities"] == pytest.approx(
                reference["answers"][question.id]["probabilities"], abs=1e-5
            )
    ledger = actual[0]["prefix"]
    assert ledger["row_to_request_index"] == [0, 0, 1, 1, 2]
    assert ledger["completed_response_indices"] == [0, 1, 2, 3, 4]
    assert ledger["request_aggregation"]["completed_response_indices"] == [0, 1, 2]
    assert ledger["forward_calls"] == 4
    assert [response["passes"] for response in actual] == [2, 1, 1]
    assert (
        sum(response["usage"]["input_tokens"] for response in actual)
        == ledger["computed_input_tokens"]
    )
    assert sum(response["execution"]["forward_calls"] for response in actual) == 4
    assert ledger["padding_input_tokens"] > 0
    assert all(response["execution"]["max_forward_batch_size"] == 2 for response in actual)
    assert (
        sum(response["execution"]["logical_computed_input_tokens"] for response in actual)
        == ledger["logical_computed_input_tokens"]
    )
    assert (
        sum(response["execution"]["padding_input_tokens"] for response in actual)
        == ledger["padding_input_tokens"]
    )
    assert all(
        response["execution"]["computed_input_tokens"]
        == response["execution"]["logical_computed_input_tokens"]
        + response["execution"]["padding_input_tokens"]
        for response in actual
    )
    json.dumps(actual, allow_nan=False)


@pytest.mark.parametrize(
    "fail_on,completed_rows,completed_requests", [(3, [0, 1], [0]), (4, [0, 1, 2, 3], [0, 1])]
)
def test_two_suffix_chunk_failure_maps_complete_original_requests_and_retains_raw_rows(
    model, monkeypatch, fail_on, completed_rows, completed_requests
):
    backend = BrowserDecisionBackend(
        model, prompt_mode="user", prefix_cache=True, prefix_suffix_batch_size=2
    )
    compact_backend_preparer(backend, monkeypatch)
    calls = []

    def fail(_module, _args, _kwargs):
        calls.append(None)
        if len(calls) == fail_on:
            raise RuntimeError("batched suffix interrupted")

    hook = model.backbone.register_forward_pre_hook(fail, with_kwargs=True)
    try:
        with pytest.raises(RuntimeError, match="batched suffix interrupted") as caught:
            backend.predict_batch(requests())
    finally:
        hook.remove()
    evidence = caught.value.batch_evidence
    assert evidence["row_to_request_index"] == [0, 0, 1, 1, 2]
    assert evidence["completed_row_response_indices"] == completed_rows
    assert evidence["completed_response_indices"] == completed_requests
    prefix = evidence["prefix_evidence"]
    assert prefix["completed_response_indices"] == completed_rows
    assert prefix["branches"][0]["row_indices"] == [0, 1]
    assert len(prefix["branches"][0]["raw_probabilities"]) == 2
    assert evidence["aggregation_error"] is None
    json.dumps(evidence, allow_nan=False)
