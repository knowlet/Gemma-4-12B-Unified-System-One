"""Small text baselines with explicit probability and fixed-schema contracts."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from s1.contracts import answer_from_probabilities
from s1.errors import RequestValidationError


def schema_key(question):
    # IDs and candidate ordering are transport details, descriptions are semantic.
    content = {
        "type": question.type,
        "instructions": question.instructions,
        "criteria": dict(zip(question.labels(), question.descriptions())),
    }
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def state_text(state):
    return (
        state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, sort_keys=True)
    )


def tokens(text):
    # Word tokens + individual CJK characters; fixed and serialized policy.
    return re.findall(r"[a-z0-9_]+|[^\x00-\x7f]", text.lower())


def tfidf_matrix(texts, vocabulary, idf):
    vectors = np.zeros((len(texts), len(vocabulary)), dtype=np.float64)
    for index, text in enumerate(texts):
        for term, count in Counter(tokens(text)).items():
            if term in vocabulary:
                vectors[index, vocabulary[term]] = count
    vectors *= np.asarray(idf)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class BaselineBackend:
    def __init__(self, spec):
        self.spec, self.model, self.tokenizer = spec, None, None
        self.artifact = None
        if spec.artifact_file:
            raw = Path(spec.artifact_file).read_bytes()
            if hashlib.sha256(raw).hexdigest() != spec.artifact_sha256:
                raise ValueError("baseline artifact hash mismatch")
            self.artifact = json.loads(raw)
        if spec.adapter == "embedding":
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                spec.model_id,
                revision=spec.revision,
                device=spec.device,
                trust_remote_code=False,
                model_kwargs={"torch_dtype": "float32"},
            )
        elif spec.adapter in ("nli", "cross_encoder"):
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(spec.model_id, revision=spec.revision)
            self.model = (
                AutoModelForSequenceClassification.from_pretrained(
                    spec.model_id, revision=spec.revision, dtype="float32"
                )
                .to(spec.device or "cpu")
                .eval()
            )
        elif spec.adapter == "setfit":
            from setfit import SetFitModel

            from .checkpoints import directory_digest

            if not self.artifact or directory_digest(spec.model_id) != self.artifact.get(
                "checkpoint_sha256"
            ):
                raise ValueError("SetFit checkpoint differs from its training artifact")

            self.model = SetFitModel.from_pretrained(
                spec.model_id, revision=spec.revision, device=spec.device
            )

    def predict(self, request):
        if request.media:
            raise RequestValidationError("text baseline cannot consume media")
        text, answers = state_text(request.state), {}
        for q in request.questions:
            if self.spec.adapter in ("tfidf", "prior", "setfit"):
                schema = (self.artifact or {}).get("schemas", {}).get(schema_key(q))
                if schema is None or set(schema["labels"]) != set(q.labels()):
                    raise RequestValidationError("unseen supervised schema")
                if self.spec.adapter == "prior":
                    probabilities = np.asarray(schema["prior"], dtype=float)
                elif self.spec.adapter == "tfidf":
                    x = tfidf_matrix([text], schema["vocabulary"], schema["idf"])
                    scores = (
                        x @ np.asarray(schema["coefficients"]).T + np.asarray(schema["intercept"])
                    )[0]
                    if len(scores) == 1:
                        # sklearn's binary LogisticRegression stores one log-odds row.
                        scores = np.array([0.0, scores[0]])
                    probabilities = np.exp(scores - scores.max())
                    probabilities /= probabilities.sum()
                else:
                    probabilities = np.asarray(self.model.predict_proba([text], as_numpy=True))[0]
                if len(probabilities) != len(schema["labels"]):
                    raise ValueError("classifier output labels do not match training manifest")
                mapped = dict(zip(schema["labels"], probabilities.tolist()))
                answers[q.id] = answer_from_probabilities(
                    q, [mapped[label] for label in q.labels()]
                )
            else:
                hypotheses = [
                    f"{q.instructions}\n{label}: {desc}"
                    for label, desc in zip(q.labels(), q.descriptions())
                ]
                if self.spec.adapter == "embedding":
                    vectors = self.model.encode(
                        [text, *hypotheses], normalize_embeddings=True, convert_to_numpy=True
                    )
                    scores = vectors[1:] @ vectors[0]
                else:
                    import torch

                    inputs = self.tokenizer(
                        [text] * len(hypotheses),
                        hypotheses,
                        padding=True,
                        truncation=False,
                        return_tensors="pt",
                    )
                    if inputs["input_ids"].shape[1] > self.spec.context_limit:
                        raise RequestValidationError("baseline context limit exceeded")
                    inputs = {
                        key: value.to(self.spec.device or "cpu") for key, value in inputs.items()
                    }
                    with torch.inference_mode():
                        logits = self.model(**inputs).logits.float().cpu().numpy()
                    if self.spec.adapter == "nli":
                        ids = [
                            int(i)
                            for i, label in self.model.config.id2label.items()
                            if label.lower() == "entailment"
                        ]
                        if len(ids) != 1:
                            raise ValueError("NLI checkpoint must identify its entailment class")
                        p = np.exp(logits - logits.max(axis=1, keepdims=True))
                        scores = (p / p.sum(axis=1, keepdims=True))[:, ids[0]]
                    else:
                        if logits.shape[1] != 1:
                            raise ValueError("cross encoder must emit one relevance score")
                        scores = logits[:, 0]
                # Similarity and entailment rankings are not candidate probabilities.
                if not np.isfinite(scores).all():
                    raise ValueError("nonfinite baseline ranking scores")
                answers[q.id] = {"label": q.labels()[int(np.argmax(scores))]}
        return {"answers": answers}

    def close(self):
        self.model = None
