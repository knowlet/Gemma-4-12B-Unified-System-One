import sys
from types import SimpleNamespace

import pytest

from s1.contracts import DecisionRequest
from s1.errors import RequestValidationError
from s1.evaluation.adapters import execution_blockers, load_adapter
from s1.evaluation.contracts import ProfileSpec
from s1.evaluation.integrity import validate_runtime_telemetry
from s1.evaluation.local_competitors import (
    CHECKPOINTS,
    LocalCompetitorBackend,
    local_competitor_spec,
)
from s1.evaluation.registry import Registry


@pytest.fixture
def registry():
    return Registry.load(
        "configs/benchmarks/models.toml",
        "configs/benchmarks/suites.toml",
        "configs/benchmarks/profiles.toml",
    )


def bare_backend(name):
    backend = LocalCompetitorBackend.__new__(LocalCompetitorBackend)
    backend.spec = local_competitor_spec(name)
    backend.identity = CHECKPOINTS[name]
    return backend


@pytest.mark.parametrize("name", CHECKPOINTS)
def test_pinned_specs_are_executable_and_match_registry(name, registry):
    spec = local_competitor_spec(name)
    registered = registry.models[spec.id]
    assert registered.model_id == spec.model_id
    assert registered.revision == spec.revision
    assert spec.capabilities.modalities == ("text",)
    assert execution_blockers(spec, ProfileSpec(id="p", suite="s", models=(spec.id,))) == []
    changed = spec.model_copy(update={"revision": "a" * 40})
    assert "local_competitor_checkpoint_unverified" in execution_blockers(
        changed, ProfileSpec(id="p", suite="s", models=(spec.id,))
    )
    with pytest.raises(ValueError, match="checkpoint identity"):
        LocalCompetitorBackend(changed)


@pytest.mark.parametrize("name", CHECKPOINTS)
def test_native_factory_uses_same_adapter_as_harness(name, monkeypatch):
    import s1.evaluation.local_competitors as local

    captured = []
    monkeypatch.setattr(
        local, "LocalCompetitorBackend", lambda spec: captured.append(spec) or object()
    )
    spec = local_competitor_spec(name)
    load_adapter(spec, {}).close()
    assert captured == [spec]


def test_decider_preserves_question_semantics_and_rejects_context(decision_request):
    backend = bare_backend("decider")
    calls = []

    def system_one(**payload):
        calls.append(payload)
        assert payload["questions"]["urgency"]["criteria"] == ["low", "high"]
        return {
            "answers": {
                "route": {"probabilities": {"billing": 0.8, "technical": 0.2}},
                "refund": {"noul": 0.7},
                "urgency": {"probabilities": {"0": 0.25, "1": 0.75}},
            }
        }

    backend.engine = SimpleNamespace(
        _system_one_items=lambda **kw: ({}, {}, [{"ids": [1, 2]}]), system_one=system_one
    )
    result = backend.predict(decision_request)
    assert result["answers"]["urgency"]["score"] == 17.5
    assert calls[0]["state"] == decision_request.state
    assert "gold" not in calls[0]
    backend.spec = backend.spec.model_copy(update={"context_limit": 1})
    with pytest.raises(RequestValidationError, match="context limit"):
        backend.predict(decision_request)
    assert len(calls) == 1


def test_agentjev_boolean_order_and_checkpoint_precision_are_explicit():
    backend = bare_backend("agentjev")
    request = DecisionRequest.model_validate(
        {"state": "A fact", "questions": [{"id": "q", "type": "noul", "instructions": "True?"}]}
    )

    def evaluate(payload):
        assert payload["questions"][0]["type"] == "boolean"
        return {
            "results": [
                {
                    "answers": [
                        {
                            "id": "q",
                            "type": "boolean",
                            "probability": 0.8,
                            "distribution": {"true": 0.8, "false": 0.2},
                        }
                    ]
                }
            ]
        }

    backend.engine = SimpleNamespace(evaluate=evaluate, temperatures={"boolean": 1.2})
    assert backend.predict(request)["answers"]["q"]["noul"] == 0.8
    telemetry = validate_runtime_telemetry(backend, backend.spec)
    assert telemetry["checkpoint_task"] == "coding_completion"
    assert telemetry["precision"] == "float32"
    assert telemetry["compute_precision"] == "bfloat16_autocast"
    assert telemetry["base_revision"] == CHECKPOINTS["agentjev"].base_revision


def test_kev_uses_strict_full_context_encoding_and_checkpoint_readout(monkeypatch):
    backend = bare_backend("kev")
    request = DecisionRequest.model_validate(
        {"state": "A fact", "questions": [{"id": "q", "type": "noul", "instructions": "True?"}]}
    )
    captured = {}

    def encode(tokenizer, record, **kwargs):
        captured.update(kwargs)
        return {"ids": [1, 2, 3]}

    fake_api = SimpleNamespace(
        SystemOneRequest=SimpleNamespace(model_validate=lambda payload: payload),
        to_record=lambda payload: (payload, [{"id": "q"}]),
        to_answers=lambda probs, mapping: {"q": {"noul": probs[0][1]}},
    )
    monkeypatch.setitem(sys.modules, "kev.api", fake_api)
    backend.tokenizer = object()
    backend.model = SimpleNamespace(
        encode=encode, probs=lambda enc: [SimpleNamespace(tolist=lambda: [0.2, 0.8])]
    )
    assert backend.predict(request)["answers"]["q"]["noul"] == 0.8
    assert captured == {"max_state": 8192, "max_branch": 8192, "strict": True}


@pytest.mark.parametrize("name", CHECKPOINTS)
def test_native_media_is_rejected_without_model_access(name, decision_request):
    backend = bare_backend(name)
    request = decision_request.model_copy(update={"media": [object()]})
    with pytest.raises(RequestValidationError, match="text-only"):
        backend.predict(request)


def test_quantized_cpu_and_wrong_telemetry_are_rejected(registry):
    spec = registry.models["gemma-g4"].model_copy(update={"quantization": "nf4", "device": "cpu"})
    assert "gemma_quantization_requires_cuda_bfloat16" in execution_blockers(
        spec, ProfileSpec(id="p", suite="s", models=(spec.id,))
    )
    runtime = SimpleNamespace(
        telemetry=lambda: {
            "revision": spec.revision,
            "precision": spec.precision,
            "context_policy": "reject",
        }
    )
    with pytest.raises(ValueError, match="quantization"):
        validate_runtime_telemetry(runtime, spec)
