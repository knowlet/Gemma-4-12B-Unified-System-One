import json
import math
from dataclasses import dataclass, field

import jev_decision3_cpu_padding_candidate as candidate
import pytest


@dataclass
class Question:
    keys: list[str]
    kind: str = "choice"
    original: dict = field(default_factory=lambda: {"type": "choice"})
    rendered: dict = field(default_factory=lambda: {"type": "choice"})
    descriptions: list = field(default_factory=list)


@dataclass
class Prepared:
    keys: list[str]
    questions: dict
    sequences: dict
    errors: dict = field(default_factory=dict)
    images: list = field(default_factory=list)
    videos: list = field(default_factory=list)
    media: dict = field(default_factory=dict)
    video_texts: list = field(default_factory=list)
    texts: dict = field(default_factory=dict)
    lengths: dict = field(default_factory=dict)
    reversed_sequences: dict = field(default_factory=dict)
    reversed_texts: dict = field(default_factory=dict)
    reversed_lengths: dict = field(default_factory=dict)


class FakeRuntime:
    device = "cpu"
    batch_size = 8
    permutation_average = False
    fast = None
    model_name = "synthetic-decision3"

    def __init__(self):
        self.calls = []

    def run(self, prepared):
        self.calls.append(list(prepared.keys))
        results = {}
        for key in prepared.keys:
            count = len(prepared.questions[key].keys)
            results[key] = [1 / count] * count
        return results, sum(len(prepared.sequences[key]) for key in prepared.keys)

    def respond(self, prepared, probabilities, tokens):
        answers = {}
        for key in prepared.keys:
            question = prepared.questions[key]
            row = probabilities[key]
            shares = [value / sum(row) for value in row]
            if question.kind == "noul":
                answers[key] = {"type": "noul", "noul": shares[1]}
                continue
            confidence = (
                1.0
                if len(shares) == 1
                else max(
                    0.0,
                    min(
                        1.0,
                        1 + sum(p * math.log(p) for p in shares if p > 0) / math.log(len(shares)),
                    ),
                )
            )
            answer = {
                "type": question.kind,
                "probabilities": dict(zip(question.keys, shares, strict=True)),
                "confidence": confidence,
            }
            if question.kind == "choice":
                answer["choice"] = question.keys[max(range(len(shares)), key=shares.__getitem__)]
            else:
                answer["score"] = sum(i * p for i, p in enumerate(shares))
                answer["legend"] = {
                    label: description
                    if isinstance(description, str)
                    else json.dumps(
                        description, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    )
                    for label, description in zip(question.keys, question.descriptions, strict=True)
                }
            answers[key] = answer
        return {
            "model": self.model_name,
            "answers": answers,
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        }


def prepared(lengths=(1000, 1001, 2000, 2001)):
    keys = [f"q{i}" for i in range(len(lengths))]
    return Prepared(
        keys,
        {key: Question([f"o{n}" for n in range(i + 1)]) for i, key in enumerate(keys)},
        {key: [i + 1] * length for i, (key, length) in enumerate(zip(keys, lengths, strict=True))},
    )


@pytest.fixture(autouse=True)
def enable_reviewed_candidate(monkeypatch):
    monkeypatch.setattr(candidate, "SOURCE_ONLY", False)


def test_gate_and_explicit_opt_in(monkeypatch):
    runtime = FakeRuntime()
    with pytest.raises(ValueError, match="opt-in"):
        candidate.evaluate(runtime, prepared(), mode="whole")
    monkeypatch.setattr(candidate, "SOURCE_ONLY", True)
    with pytest.raises(ValueError, match="opt-in"):
        candidate.evaluate(runtime, prepared(), mode="whole", enabled=True)
    assert runtime.calls == []


def test_controls_keep_all_options_order_and_tokens_without_mutation():
    source = prepared()
    original_keys = list(source.keys)
    results = []
    calls = []
    for mode in ("whole", "ordered_pairs", "length_pairs"):
        runtime = FakeRuntime()
        result = candidate.evaluate(runtime, source, mode=mode, enabled=True)
        results.append(result)
        calls.append(runtime.calls)
    assert source.keys == original_keys
    assert calls == [[original_keys], [["q0", "q1"], ["q2", "q3"]], [["q3", "q2"], ["q1", "q0"]]]
    assert [r["response"] for r in results] == [results[0]["response"]] * 3
    assert [
        len(results[0]["response"]["answers"][key]["probabilities"]) for key in original_keys
    ] == [1, 2, 3, 4]
    assert results[0]["response"]["usage"]["input_tokens"] == 6002
    assert [r["padded_slots"] for r in results] == [8004, 6004, 6004]
    assert [r["native_run_calls"] for r in results] == [1, 2, 2]
    assert "unverified" in results[2]["scope"]


def test_length_pairs_reduce_padding_for_interleaved_heads():
    source = prepared((1000, 2001, 1001, 2000))
    ordered = candidate.evaluate(FakeRuntime(), source, mode="ordered_pairs", enabled=True)
    sorted_result = candidate.evaluate(FakeRuntime(), source, mode="length_pairs", enabled=True)
    assert ordered["padded_slots"] == 8002
    assert sorted_result["padded_slots"] == 6004
    assert sorted_result["response"] == ordered["response"]


def test_odd_tail_and_equal_lengths_are_stable():
    source = prepared((1, 100, 100, 100, 1))
    runtime = FakeRuntime()
    result = candidate.evaluate(runtime, source, mode="length_pairs", enabled=True)
    assert runtime.calls == [["q1", "q2"], ["q3", "q0"], ["q4"]]
    assert result["padded_slots"] == 401
    assert list(result["response"]["answers"]) == source.keys


def test_whole_padded_slots_respects_native_microbatch():
    source = prepared((1, 1, 1, 1, 1, 1, 1, 100, 200))
    result = candidate.evaluate(FakeRuntime(), source, mode="whole", enabled=True)
    assert result["padded_slots"] == 1000
    assert result["native_run_calls"] == 1
    assert result["forward_groups"] == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("images", [object()]),
        ("videos", [object()]),
        ("media", {"x": 1}),
        ("video_texts", ["x"]),
        ("texts", {"q0": "x"}),
        ("lengths", {"q0": 1}),
        ("errors", {"q0": "bad"}),
        ("reversed_sequences", {"q0": [1]}),
        ("reversed_texts", {"q0": "x"}),
        ("reversed_lengths", {"q0": 1}),
    ],
)
def test_nontext_or_incomplete_inputs_are_rejected(field, value):
    source = prepared()
    setattr(source, field, value)
    runtime = FakeRuntime()
    with pytest.raises(ValueError):
        candidate.evaluate(runtime, source, mode="length_pairs", enabled=True)
    assert runtime.calls == []


@pytest.mark.parametrize("length", [0, 16385])
def test_invalid_length_rejected_without_truncation(length):
    source = prepared((length,))
    with pytest.raises(ValueError, match="bounded"):
        candidate.evaluate(FakeRuntime(), source, mode="whole", enabled=True)
    assert len(source.sequences["q0"]) == length


def test_invalid_keys_or_cpu_policy_are_rejected():
    source = prepared()
    source.keys[1] = source.keys[0]
    with pytest.raises(ValueError):
        candidate.evaluate(FakeRuntime(), source, mode="whole", enabled=True)
    runtime = FakeRuntime()
    runtime.device = "cuda:0"
    with pytest.raises(ValueError, match="CPU"):
        candidate.evaluate(runtime, prepared(), mode="whole", enabled=True)
    runtime.device = "cpu"
    runtime.permutation_average = True
    with pytest.raises(ValueError, match="CPU"):
        candidate.evaluate(runtime, prepared(), mode="whole", enabled=True)


def test_bad_group_result_and_throw_release_candidate_lock():
    source = prepared()

    class Broken(FakeRuntime):
        def run(self, source):
            return {}, 0

    with pytest.raises(ValueError, match="per-group"):
        candidate.evaluate(Broken(), source, mode="ordered_pairs", enabled=True)

    class Raises(FakeRuntime):
        def run(self, source):
            raise RuntimeError("failed")

    with pytest.raises(RuntimeError, match="failed"):
        candidate.evaluate(Raises(), source, mode="length_pairs", enabled=True)
    assert (
        candidate.evaluate(FakeRuntime(), source, mode="whole", enabled=True)["unpadded_tokens"]
        == 6002
    )
    assert source.keys == ["q0", "q1", "q2", "q3"]


def test_mutated_original_input_is_refused():
    source = prepared()

    class Mutates(FakeRuntime):
        def run(self, part):
            result = super().run(part)
            part.sequences[part.keys[0]][0] = 999
            return result

    with pytest.raises(ValueError, match="changed"):
        candidate.evaluate(Mutates(), source, mode="whole", enabled=True)


def test_255_options_and_maximum_context_are_preserved():
    source = prepared((16384, 1))
    source.questions["q0"].keys = [f"item{i}" for i in range(255)]
    result = candidate.evaluate(FakeRuntime(), source, mode="length_pairs", enabled=True)
    assert len(result["response"]["answers"]["q0"]["probabilities"]) == 255
    assert len(source.sequences["q0"]) == 16384
    assert result["unpadded_tokens"] == 16385


def test_response_errors_and_group_option_count_are_not_hidden():
    class BadResponse(FakeRuntime):
        def respond(self, prepared, probabilities, tokens):
            result = super().respond(prepared, probabilities, tokens)
            result["usage"]["output_tokens"] = True
            return result

    with pytest.raises(ValueError, match="response order"):
        candidate.evaluate(BadResponse(), prepared(), mode="whole", enabled=True)

    class BadOptions(FakeRuntime):
        def run(self, prepared):
            results, tokens = super().run(prepared)
            results[prepared.keys[0]] = []
            return results, tokens

    with pytest.raises(ValueError, match="every original option"):
        candidate.evaluate(BadOptions(), prepared(), mode="whole", enabled=True)


@pytest.mark.parametrize("field", ["kind", "original", "rendered", "descriptions"])
def test_mutated_question_response_metadata_is_refused(field):
    source = prepared()

    class MutatesQuestion(FakeRuntime):
        def run(self, part):
            result = super().run(part)
            q = part.questions[part.keys[0]]
            if field == "kind":
                q.kind = "score"
            elif field == "descriptions":
                q.descriptions.append("changed")
            else:
                getattr(q, field)["changed"] = True
            return result

    with pytest.raises(ValueError, match="changed"):
        candidate.evaluate(MutatesQuestion(), source, mode="whole", enabled=True)


@pytest.mark.parametrize(
    "before,after",
    [
        ((1, 2), [1, 2]),
        ([1, 2], (1, 2)),
        (True, 1),
        (1, 1.0),
        (-0.0, 0.0),
    ],
)
def test_metadata_type_changes_during_run_are_refused(before, after):
    source = prepared()
    source.questions["q0"].original["nested"] = before

    class ChangesType(FakeRuntime):
        def run(self, part):
            result = super().run(part)
            part.questions["q0"].original["nested"] = after
            return result

    with pytest.raises(ValueError, match="changed"):
        candidate.evaluate(ChangesType(), source, mode="whole", enabled=True)


def test_shared_acyclic_metadata_is_allowed_but_cycle_is_rejected():
    source = prepared()
    shared = ["a", (1, True)]
    source.questions["q0"].original.update(first=shared, second=shared)
    candidate.evaluate(FakeRuntime(), source, mode="whole", enabled=True)
    shared.append(shared)
    runtime = FakeRuntime()
    with pytest.raises(ValueError, match="acyclic"):
        candidate.evaluate(runtime, source, mode="whole", enabled=True)
    assert runtime.calls == []


@pytest.mark.parametrize("bad", [object(), {"a", "b"}, float("nan"), float("inf"), chr(0xD800)])
def test_unsupported_metadata_is_rejected_before_runtime(bad):
    source = prepared()
    source.questions["q0"].original["nested"] = bad
    runtime = FakeRuntime()
    with pytest.raises(ValueError):
        candidate.evaluate(runtime, source, mode="whole", enabled=True)
    assert runtime.calls == []


def test_utf8_metadata_byte_budget_is_checked_before_copy(monkeypatch):
    monkeypatch.setattr(candidate, "_MAX_METADATA_BYTES", 4)
    assert candidate._freeze_metadata("aé", [0, 0], set()) == ("str", "aé")
    with pytest.raises(ValueError, match="bounded metadata bytes"):
        candidate._freeze_metadata("aéé", [0, 0], set())
    with pytest.raises(ValueError, match="bounded metadata bytes"):
        candidate._freeze_metadata("abcde", [0, 0], set())


def test_metadata_node_depth_and_integer_limits_are_enforced(monkeypatch):
    monkeypatch.setattr(candidate, "_MAX_METADATA_NODES", 5)
    with pytest.raises(ValueError, match="bounded metadata nodes"):
        candidate._freeze_metadata([None] * 5, [0, 0], set())
    monkeypatch.setattr(candidate, "_MAX_METADATA_NODES", 100)
    monkeypatch.setattr(candidate, "_MAX_METADATA_DEPTH", 2)
    with pytest.raises(ValueError, match="bounded metadata nesting"):
        candidate._freeze_metadata([[[None]]], [0, 0], set())
    monkeypatch.setattr(candidate, "_MAX_INTEGER_BITS", 8)
    with pytest.raises(ValueError, match="bounded metadata integer"):
        candidate._freeze_metadata(256, [0, 0], set())


def test_identifier_and_option_content_is_bounded_before_run(monkeypatch):
    source = prepared((1,))
    long_key = "x" * 32
    source.keys = [long_key]
    source.questions = {long_key: source.questions["q0"]}
    source.sequences = {long_key: source.sequences["q0"]}
    monkeypatch.setattr(candidate, "_MAX_METADATA_BYTES", 16)
    runtime = FakeRuntime()
    with pytest.raises(ValueError, match="bounded metadata bytes"):
        candidate.evaluate(runtime, source, mode="whole", enabled=True)
    assert runtime.calls == []
    source = prepared((1,))
    source.questions["q0"].keys = ["x" * 32]
    with pytest.raises(ValueError, match="bounded metadata bytes"):
        candidate.evaluate(runtime, source, mode="whole", enabled=True)
    assert runtime.calls == []


def test_out_of_range_token_id_is_rejected_before_run():
    source = prepared((1,))
    source.sequences["q0"][0] = 2**63
    runtime = FakeRuntime()
    with pytest.raises(ValueError, match="bounded token"):
        candidate.evaluate(runtime, source, mode="whole", enabled=True)
    assert runtime.calls == []


def test_publisher_choice_noul_and_score_shapes_and_all_native_probabilities():
    source = prepared((1, 2, 3))
    source.questions["q1"] = Question(["false", "true"], kind="noul")
    source.questions["q2"] = Question(
        ["0", "1", "2"], kind="score", descriptions=["low", {"level": 2}, "high"]
    )
    result = candidate.evaluate(FakeRuntime(), source, mode="length_pairs", enabled=True)
    answers = result["response"]["answers"]
    assert list(answers) == ["q0", "q1", "q2"]
    assert answers["q0"] == {
        "type": "choice",
        "probabilities": {"o0": 1.0},
        "confidence": 1.0,
        "choice": "o0",
    }
    assert answers["q1"] == {"type": "noul", "noul": 0.5}
    assert list(result["native_probabilities"]["q1"]) == [0.5, 0.5]
    assert answers["q2"]["score"] == pytest.approx(1.0)
    assert answers["q2"]["legend"] == {"0": "low", "1": '{"level":2}', "2": "high"}
    assert list(answers["q2"]["probabilities"]) == ["0", "1", "2"]


@pytest.mark.parametrize(
    "values",
    [[-3.0, 4.0], [1.1, -0.1], [0.2, 0.2], [float("nan"), 0.5], [True, False], []],
)
def test_invalid_native_probability_row_is_refused(values):
    class InvalidRow(FakeRuntime):
        def run(self, part):
            rows, tokens = super().run(part)
            rows["q1"] = values
            return rows, tokens

    with pytest.raises(ValueError, match="probabilities for every original option"):
        candidate.evaluate(InvalidRow(), prepared(), mode="whole", enabled=True)


@pytest.mark.parametrize(
    "fault",
    [
        "empty",
        "error",
        "missing_option",
        "reordered_option",
        "wrong_probability",
        "wrong_choice",
        "wrong_confidence",
    ],
)
def test_invalid_native_choice_answer_is_refused(fault):
    class InvalidAnswer(FakeRuntime):
        def respond(self, part, probabilities, tokens):
            result = super().respond(part, probabilities, tokens)
            answer = result["answers"]["q1"]
            if fault == "empty":
                result["answers"]["q1"] = []
            elif fault == "error":
                result["answers"]["q1"] = {"type": "choice", "error": "invalid_model_output"}
            elif fault == "missing_option":
                answer["probabilities"].pop("o1")
            elif fault == "reordered_option":
                answer["probabilities"] = dict(reversed(list(answer["probabilities"].items())))
            elif fault == "wrong_probability":
                answer["probabilities"]["o0"] = 0.9
            elif fault == "wrong_choice":
                answer["choice"] = "o1"
            else:
                answer["confidence"] = 0.9
            return result

    with pytest.raises(ValueError, match="native"):
        candidate.evaluate(InvalidAnswer(), prepared(), mode="whole", enabled=True)


@pytest.mark.parametrize(
    "kind,fault", [("noul", "wrong_value"), ("score", "wrong_score"), ("score", "wrong_legend")]
)
def test_invalid_noul_and_score_answers_are_refused(kind, fault):
    source = prepared((1, 2))
    if kind == "noul":
        source.questions["q1"] = Question(["false", "true"], kind="noul")
    else:
        source.questions["q1"] = Question(["0", "1"], kind="score", descriptions=["low", "high"])

    class InvalidAnswer(FakeRuntime):
        def respond(self, part, probabilities, tokens):
            result = super().respond(part, probabilities, tokens)
            answer = result["answers"]["q1"]
            if fault == "wrong_value":
                answer["noul"] = 0.9
            elif fault == "wrong_score":
                answer["score"] = 0.9
            else:
                answer["legend"]["1"] = "different"
            return result

    with pytest.raises(ValueError, match="native"):
        candidate.evaluate(InvalidAnswer(), source, mode="whole", enabled=True)


def test_native_probability_row_cannot_change_during_response():
    class MutatesRow(FakeRuntime):
        def respond(self, part, probabilities, tokens):
            result = super().respond(part, probabilities, tokens)
            probabilities["q1"][0] = 0.6
            return result

    with pytest.raises(ValueError, match="row changed"):
        candidate.evaluate(MutatesRow(), prepared(), mode="whole", enabled=True)


@pytest.mark.parametrize("fault", ["missing", "wrong", "wrong_type"])
def test_native_response_requires_the_selected_model_name(fault):
    class InvalidModel(FakeRuntime):
        def respond(self, part, probabilities, tokens):
            result = super().respond(part, probabilities, tokens)
            if fault == "missing":
                result.pop("model")
            elif fault == "wrong":
                result["model"] = "other-synthetic-model"
            else:
                result["model"] = True
            return result

    with pytest.raises(ValueError, match="response order"):
        candidate.evaluate(InvalidModel(), prepared(), mode="whole", enabled=True)


@pytest.mark.parametrize("phase", ["run", "respond"])
def test_runtime_model_name_changes_are_refused(phase):
    class MutatedModel(FakeRuntime):
        def run(self, part):
            result = super().run(part)
            if phase == "run":
                self.model_name = "changed-synthetic-model"
            return result

        def respond(self, part, probabilities, tokens):
            result = super().respond(part, probabilities, tokens)
            if phase == "respond":
                self.model_name = "changed-synthetic-model"
            return result

    with pytest.raises(ValueError, match="runtime changed|response order"):
        candidate.evaluate(MutatedModel(), prepared(), mode="whole", enabled=True)


@pytest.mark.parametrize("name", [None, "", "   ", True, "x" * 4097])
def test_missing_or_invalid_runtime_model_name_rejected_before_run(name):
    runtime = FakeRuntime()
    runtime.model_name = name
    with pytest.raises(ValueError, match="CPU reference runtime"):
        candidate.evaluate(runtime, prepared(), mode="whole", enabled=True)
    assert runtime.calls == []
