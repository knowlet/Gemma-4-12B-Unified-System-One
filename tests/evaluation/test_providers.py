import json

import httpx
import pytest

from s1.errors import BackendResponseError
from s1.evaluation.providers import ProviderHTTPBackend, normalize_typesafe_rounding


def test_typesafe_rounding_only_repairs_four_decimal_quantization():
    valid = {"answers": {"q": {"probabilities": {"a": 0.3333, "b": 0.3333, "c": 0.3333}}}}
    result = normalize_typesafe_rounding(valid)
    assert sum(result["answers"]["q"]["probabilities"].values()) == 1
    assert result["probability_postprocessing"] == "bounded_four_decimal_renormalization"
    malformed = {"answers": {"q": {"probabilities": {"a": 0.2, "b": 0.2}}}}
    assert normalize_typesafe_rounding(malformed)["answers"]["q"]["probabilities"] == {
        "a": 0.2,
        "b": 0.2,
    }


@pytest.mark.parametrize("protocol", ["agentjev", "typesafe"])
def test_provider_contracts_preserve_semantic_score_levels(decision_request, protocol):
    def handler(request):
        payload = json.loads(request.content)
        assert "gold" not in payload
        if protocol == "typesafe":
            assert payload["questions"]["urgency"]["criteria"] == ["low", "high"]
            response = {
                "answers": {
                    "route": {
                        "choice": "billing",
                        "probabilities": {"billing": 0.8, "technical": 0.2},
                    },
                    "refund": {"noul": 0.7},
                    "urgency": {"probabilities": {"0": 0.25, "1": 0.75}},
                }
            }
        else:
            assert payload["questions"][1]["type"] == "boolean"
            assert payload["questions"][2]["levels"] == ["low", "high"]
            response = {
                "results": [
                    {
                        "answers": [
                            {
                                "id": "route",
                                "type": "choice",
                                "value": "billing",
                                "distribution": {"billing": 0.8, "technical": 0.2},
                            },
                            {
                                "id": "refund",
                                "type": "boolean",
                                "probability": 0.7,
                                "distribution": {"true": 0.7, "false": 0.3},
                            },
                            {
                                "id": "urgency",
                                "type": "score",
                                "distribution": {"0": 0.25, "1": 0.75},
                            },
                        ]
                    }
                ]
            }
        return httpx.Response(200, json=response)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        backend = ProviderHTTPBackend(
            "https://example.invalid/evaluate", protocol=protocol, client=client
        )
        response = backend.predict(decision_request)
    assert response["answers"]["urgency"]["score"] == 17.5
    assert response["answers"]["refund"]["noul"] == 0.7


def test_invalid_agentjev_envelope_is_backend_error(decision_request):
    with httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"results": []}))
    ) as client:
        with pytest.raises(BackendResponseError):
            ProviderHTTPBackend(
                "https://example.invalid/evaluate", protocol="agentjev", client=client
            ).predict(decision_request)
