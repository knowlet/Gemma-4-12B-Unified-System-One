"""Real local encoders and SetFit training, without pretrained downloads."""

import json

import pytest


@pytest.fixture
def tiny_encoder(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")
    from sentence_transformers import SentenceTransformer, models
    from transformers import BertConfig, BertModel, BertTokenizer

    torch.manual_seed(0)
    vocabulary = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "good",
        "bad",
        "happy",
        "sad",
        "positive",
        "negative",
        "classify",
        "sentiment",
    ]
    tokenizer = BertTokenizer(vocab={word: i for i, word in enumerate(vocabulary)})
    config = BertConfig(
        vocab_size=len(vocabulary),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
    )
    path = tmp_path / "bert"
    BertModel(config).save_pretrained(path)
    tokenizer.save_pretrained(path)
    encoder = SentenceTransformer(
        modules=[models.Transformer(str(path), max_seq_length=32), models.Pooling(16)]
    )
    output = tmp_path / "embedding"
    encoder.save(str(output))
    return output


def test_real_embedding_and_setfit_train_reload_without_downloads(tiny_encoder, tmp_path):
    pytest.importorskip("setfit")
    from s1.evaluation.baselines import BaselineBackend
    from s1.evaluation.contracts import ModelSpec
    from s1.evaluation.datasets import EvaluationCase
    from s1.evaluation.experiments import write_cases
    from s1.evaluation.responses import normalize
    from s1.evaluation.supervision import fit_baseline

    cases = [
        EvaluationCase.model_validate(
            {
                "id": f"train-{i}",
                "split": "train",
                "request": {
                    "state": "good happy" if i % 2 else "bad sad",
                    "questions": [
                        {
                            "id": "q",
                            "type": "choice",
                            "instructions": "classify sentiment",
                            "criteria": {"negative": "bad sad", "positive": "good happy"},
                        }
                    ],
                },
                "gold": {"q": "positive" if i % 2 else "negative"},
            }
        )
        for i in range(8)
    ]
    embedding = BaselineBackend(
        ModelSpec(
            id="embedding",
            adapter="embedding",
            model_id=str(tiny_encoder),
            revision="a" * 40,
            device="cpu",
        )
    )
    result = normalize(cases[0].request, embedding.predict(cases[0].request), "none")
    assert result["answers"]["q"]["probabilities"] is None
    write_cases(tmp_path / "train.jsonl", cases)
    artifact = fit_baseline(
        tmp_path / "train.jsonl",
        tmp_path / "setfit",
        method="setfit",
        seed=0,
        model_id=str(tiny_encoder),
        revision="a" * 40,
        steps=1,
    )
    fitted = BaselineBackend(
        ModelSpec(
            id="setfit",
            adapter="setfit",
            model_id=str(tmp_path / "setfit/model"),
            revision=artifact["artifact_sha256"],
            device="cpu",
            artifact_file=artifact["artifact_file"],
            artifact_sha256=artifact["artifact_sha256"],
        )
    )
    result = normalize(cases[0].request, fitted.predict(cases[0].request))
    assert sum(result["answers"]["q"]["probabilities"].values()) == pytest.approx(1)
    manifest = json.loads((tmp_path / "setfit/classifier.json").read_text())
    assert manifest["steps"] == 1
    assert manifest["checkpoint_sha256"]


@pytest.mark.parametrize("adapter", ["nli", "cross_encoder"])
def test_real_sequence_classifier_rankings_have_no_fake_probabilities(tmp_path, adapter):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

    from s1.contracts import DecisionRequest
    from s1.evaluation.baselines import BaselineBackend
    from s1.evaluation.contracts import ModelSpec
    from s1.evaluation.responses import normalize

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "good", "bad"]
    tokenizer = BertTokenizer(vocab={word: i for i, word in enumerate(vocab)})
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        num_labels=3 if adapter == "nli" else 1,
    )
    if adapter == "nli":
        config.id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}
    path = tmp_path / adapter
    BertForSequenceClassification(config).save_pretrained(path)
    tokenizer.save_pretrained(path)
    request = DecisionRequest(
        state="good",
        questions=[
            {
                "id": "q",
                "type": "choice",
                "instructions": "good or bad",
                "criteria": {"good": "good", "bad": "bad"},
            }
        ],
    )
    backend = BaselineBackend(
        ModelSpec(
            id=adapter, adapter=adapter, model_id=str(path), revision="a" * 40, context_limit=64
        )
    )
    response = normalize(request, backend.predict(request), "none")
    assert response["answers"]["q"]["probabilities"] is None
    assert response["answers"]["q"]["label"] in ("good", "bad")
