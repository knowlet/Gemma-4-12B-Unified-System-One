"""Explicit wire mappings; service versions and weights remain operator-pinned."""

import math

from s1.backends import HTTPBackend, normalize_response
from s1.errors import BackendResponseError, RequestValidationError


def normalize_typesafe_rounding(response):
    """Kev serializes four decimals: accept only the corresponding bounded error.

    This policy is protocol-specific; general malformed distributions stay errors.
    """
    answers = response.get("answers") if isinstance(response, dict) else None
    entries = (
        answers.values()
        if isinstance(answers, dict)
        else answers
        if isinstance(answers, list)
        else []
    )
    adjusted = False
    for answer in entries:
        distribution = answer.get("probabilities") if isinstance(answer, dict) else None
        if not isinstance(distribution, dict):
            continue
        values = list(distribution.values())
        if not values or any(
            type(p) not in (float, int) or not math.isfinite(p) or not 0 <= p <= 1 for p in values
        ):
            continue  # The ordinary response validator reports the malformed data.
        mass = sum(values)
        if (
            mass > 0
            and abs(mass - 1) > 1e-5
            and abs(mass - 1) <= len(values) * 0.00005 + 1e-12
            and all(abs(p - round(p, 4)) < 1e-12 for p in values)
        ):
            answer["probabilities"] = {key: value / mass for key, value in distribution.items()}
            adjusted = True
    if adjusted:
        response["probability_postprocessing"] = "bounded_four_decimal_renormalization"
    return response


class ProviderHTTPBackend(HTTPBackend):
    def __init__(self, *args, protocol, **kwargs):
        super().__init__(*args, **kwargs)
        self.protocol = protocol

    def predict(self, request):
        if self.protocol == "compatible":
            return super().predict(request)
        if request.media:
            raise RequestValidationError("this provider wire adapter is text-only")
        if self.protocol == "typesafe":
            questions = request.named_questions_payload()
            for q in request.questions:
                if q.type == "score":
                    questions[q.id]["criteria"] = q.descriptions()
            response = self._post({"state": request.state, "questions": questions})
            response = normalize_typesafe_rounding(response)
            return normalize_response(request, response, ordinal_scores=True)
        if self.protocol != "agentjev":
            raise ValueError("unknown HTTP protocol")
        questions = []
        for q in request.questions:
            question = {
                "id": q.id,
                "type": "boolean" if q.type == "noul" else q.type,
                "question": q.instructions,
            }
            if q.type == "score":
                if len(q.labels()) > 10:
                    raise RequestValidationError("AgentJev score supports up to 10 levels")
                question["levels"] = q.descriptions()
            else:
                question["options" if q.type == "choice" else "criteria"] = dict(
                    zip(q.labels(), q.descriptions())
                )
            questions.append(question)
        response = self._post({"state": request.state, "questions": questions})
        try:
            if len(response["results"]) != 1:
                raise ValueError("wrong result count")
            raw = response["results"][0]["answers"]
            answers = {}
            for answer in raw:
                key = answer["id"]
                if key in answers:
                    raise ValueError("duplicate answer")
                converted = {"probabilities": answer["distribution"]}
                if answer["type"] == "boolean":
                    converted["noul"] = answer["probability"]
                elif answer["type"] == "choice":
                    converted["choice"] = answer["value"]
                answers[key] = converted
            return normalize_response(request, {"answers": answers}, ordinal_scores=True)
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendResponseError("invalid AgentJev response") from exc
