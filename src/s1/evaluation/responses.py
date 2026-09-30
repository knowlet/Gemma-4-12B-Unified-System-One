"""Evaluation accepts hard labels without inventing probability distributions."""

from s1.backends import normalize_response
from s1.errors import BackendResponseError


def execution_info(response):
    info = response.get("execution") if isinstance(response, dict) else None
    if not isinstance(info, dict):
        return None
    result = {}
    for key in ("batch_sizes", "sequence_tokens"):
        values = info.get(key)
        if isinstance(values, list) and all(type(v) is int and v > 0 for v in values):
            result[key] = values
    if type(info.get("forward_calls")) is int and info["forward_calls"] > 0:
        result["forward_calls"] = info["forward_calls"]
    result["scope"] = "whole_adapter_batch"
    return result


def normalize(request, response, probability_mode="complete"):
    if probability_mode == "complete":
        return normalize_response(request, response)
    if probability_mode != "none":
        raise BackendResponseError("unknown probability capability")
    answers = response.get("answers") if isinstance(response, dict) else None
    if not isinstance(answers, dict) or set(answers) != {q.id for q in request.questions}:
        raise BackendResponseError("invalid hard-label response")
    normalized = {}
    for question in request.questions:
        answer = answers[question.id]
        label = answer.get("label") if isinstance(answer, dict) else None
        if not isinstance(label, str) or label not in question.labels():
            raise BackendResponseError("invalid hard label")
        if answer.get("probabilities") is not None:
            raise BackendResponseError("hard-label adapter must not claim probabilities")
        normalized[question.id] = {"label": label, "probabilities": None}
    return {"answers": normalized}
