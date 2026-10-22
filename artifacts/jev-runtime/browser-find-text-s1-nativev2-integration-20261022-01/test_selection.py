"""Real c513 application/grammar/prompt/HTTP flow with simulated LM tokens."""

import copy
import importlib.util
import json
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("s1_find_v2_selection", HERE / "selection.py")
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)


class Tensor(np.ndarray):
    def __new__(cls, values):
        return np.asarray(values).view(cls)

    def to(self, device):
        assert device == "cpu"
        return self


class Tokenizer:
    pad_token_id, all_special_ids = 0, [0, 9000]

    def __init__(self):
        self.chunks, self.reverse = {0: "", 9000: ""}, {}

    def pack(self, text, width):
        ids = []
        for index in range(0, len(text), width):
            chunk = text[index : index + width]
            if chunk not in self.reverse:
                token = len(self.reverse) + 1
                self.reverse[chunk], self.chunks[token] = token, chunk
            ids.append(self.reverse[chunk])
        return ids

    def encode(self, text, **_kwargs):
        return [9000] if text == "<turn|>" else self.pack(text, 3)

    def decode(self, ids, **_kwargs):
        return "".join(self.chunks[int(i)] for i in ids)


class Processor:
    def __init__(self, tok):
        self.tok, self.native_envelope, self.messages = tok, True, []

    def apply_chat_template(self, messages, **_kwargs):
        from s1.browser_action_prompt import native_prompt

        self.messages.append(copy.deepcopy(messages))
        text = messages[0]["content"][0]["text"]
        return native_prompt(text) if self.native_envelope else text

    def __call__(self, *, text, **_kwargs):
        ids = Tensor([self.tok.pack(text, 64)])
        return {
            "input_ids": ids,
            "attention_mask": Tensor(np.ones_like(ids)),
            "mm_token_type_ids": Tensor(np.zeros_like(ids)),
        }


def model_fixture():
    from s1.browser_action_prompt import NATIVE_KIND, prompt_contract

    tok, hooks, calls, contexts = Tokenizer(), [], [], []

    def forward(**kwargs):
        for hook in hooks:
            hook(None, (), kwargs)

    def register(hook, **_kwargs):
        hooks.append(hook)
        return SimpleNamespace(remove=lambda: hooks.remove(hook))

    @contextmanager
    def decision_context():
        contexts.append("enter")
        try:
            yield
        finally:
            contexts.append("exit")

    forward.register_forward_pre_hook = register
    model = SimpleNamespace(
        name="fixture-numerical-output-only",
        revision="a" * 40,
        quantization="nf4",
        quantization_details={"fixture_only": True},
        max_context=8192,
        device="cpu",
        tok=tok,
        processor=Processor(tok),
        backbone=forward,
        calls=calls,
        hooks=hooks,
        contexts=contexts,
        corrupt=False,
    )
    model.action_adapter = SimpleNamespace(
        metadata={
            "kind": NATIVE_KIND,
            "prompt_format": "shared-elements-v2",
            "prompt_contract": prompt_contract(),
            "fixture_only": True,
        },
        decision_context=decision_context,
    )

    def generate(**kwargs):
        calls.append(kwargs)
        grammar = kwargs["prefix_allowed_tokens_fn"]
        index = next(
            i
            for i, value in enumerate(grammar.canonical_actions)
            if json.loads(value) == {"operation": "FIND_TEXT", "target": None}
        )
        tokens = list(grammar.encoded_actions[index])
        if model.corrupt:
            tokens = [*tok.encode('{"operation":"invented","target":null}'), 9000]
        prefix = kwargs["input_ids"][0].tolist()
        kwargs["streamer"].put(kwargs["input_ids"])
        for i, token in enumerate(tokens):
            if not model.corrupt:
                assert token in grammar(0, Tensor(prefix))
            forward(input_ids=kwargs["input_ids"] if i == 0 else Tensor([[token]]))
            kwargs["streamer"].put(Tensor([token]))
            prefix.append(token)
        kwargs["streamer"].end()
        return Tensor([prefix])

    model.lm = SimpleNamespace(
        generate=generate, eval=lambda: None, generation_config=SimpleNamespace(eos_token_id=[9000])
    )
    return model


def payload(width=2):
    operations = {
        op: op
        for op in [
            "CLICK",
            "TYPE_TEXT",
            "SELECT",
            "SCROLL_UP",
            "SCROLL_DOWN",
            "WAIT",
            "DONE",
            "BLOCKED",
            "FIND_TEXT",
        ]
    }
    questions = {
        "operation": {
            "type": "choice",
            "instructions": {"goal": "Find the requested text in the current document"},
            "criteria": operations,
        }
    }
    for name in ["click_target", "type_text_target", "select_target"]:
        questions[name] = {
            "type": "choice",
            "instructions": "Preserve all offered targets",
            "criteria": {str(i): f"Candidate {i}" for i in range(width)},
        }
    return {
        "model": "generated-actions-nf4",
        "state": {
            "page": {
                "url": "https://fixture.invalid/",
                "title": "Own fixture",
                "text": "Opening text",
            }
        },
        "questions": questions,
    }


@pytest.fixture
def selected(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    blob = (HERE / "selection.json").read_bytes()
    import hashlib

    with selector.selected(
        blob, hashlib.sha256(blob).hexdigest(), profile=selector.PROFILE
    ) as value:
        yield value
    assert not any(n == "s1" or n.startswith("s1.") for n in sys.modules)
    assert "jev_field_syntax_helper" not in sys.modules


def make_client(selected, tmp_path, model):
    _, driver, proof = selected
    events, gates = [], []
    output = tmp_path / "owned-output"
    output.mkdir()
    app = driver.make_application(
        model,
        None,
        events.append,
        SimpleNamespace(check=gates.append),
        output,
        "fixture-only",
        "shared-elements-v2",
        "unused-no-helper-call",
    )
    return TestClient(app), events, gates, output, proof


@pytest.mark.parametrize("width", [2, 255])
def test_actual_V2_service_uses_ninth_untargeted_grammar_path(selected, tmp_path, width):
    model = model_fixture()
    body = payload(width)
    client, events, gates, output, proof = make_client(selected, tmp_path, model)
    with client:
        response = client.post(
            "/v1/systemone", json=body, headers={"Authorization": "Bearer fixture-only"}
        )
    assert response.status_code == 200, response.text
    result = response.json()
    generation = result["generation"]
    assert result["answers"]["operation"]["choice"] == "FIND_TEXT"
    # The existing generated policy returns selected heads only, not native D3 posteriors.
    assert set(result["answers"]) == {"operation"}
    assert generation["prompt_format"] == "shared-elements-v2"
    assert generation["grammar"]["legal_action_count"] == 3 * width + 6
    assert generation["grammar"]["max_new_tokens"] == 32
    assert generation["grammar"]["max_legal_actions"] == 1205
    assert generation["candidate_labels"] == {
        q: list(v["criteria"]) for q, v in body["questions"].items()
    }
    assert list(generation["candidate_labels"]) == list(body["questions"])
    heads = {"CLICK": "click_target", "TYPE_TEXT": "type_text_target", "SELECT": "select_target"}
    expected_pairs = [
        {"operation": operation, "target": target}
        for operation in body["questions"]["operation"]["criteria"]
        for target in (
            body["questions"][heads[operation]]["criteria"] if operation in heads else [None]
        )
    ]
    grammar = model.calls[0]["prefix_allowed_tokens_fn"]
    assert [json.loads(value) for value in grammar.canonical_actions] == expected_pairs
    assert generation["decoded_action"] == {"operation": "FIND_TEXT", "target": None}
    assert model.contexts == ["enter", "exit"] and not model.hooks and len(model.calls) == 1
    assert gates == ["before-http-request", "after-http-request"]
    assert proof["declared_edits"] == 1 and proof["inverse_exact"]
    exchange = json.loads((output / "http/0000/exchange.json").read_text())
    assert (
        exchange["status"] == 200 and exchange["api_request_id"] == response.headers["x-request-id"]
    )
    assert any(e["event"] == "server_returned" for e in events)


def test_original_eight_operation_service_still_refuses_FIND(selected):
    from s1.api import create_app
    from s1.browser_policy import BrowserDecisionBackend

    model = model_fixture()
    app = create_app(
        BrowserDecisionBackend(model),
        jev_compatible=True,
        jev_generated_actions=True,
        jev_generated_action_grammar=True,
        jev_generated_prompt_format="shared-elements-v2",
        model_identity="generated-actions-nf4",
        api_key="fixture-only",
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/systemone", json=payload(), headers={"Authorization": "Bearer fixture-only"}
        )
    assert response.status_code == 422 and not model.calls


def test_full_inventory_over_existing_cap_is_refused_without_pruning(selected, tmp_path):
    model = model_fixture()
    client, *_ = make_client(selected, tmp_path, model)
    with client:
        response = client.post(
            "/v1/systemone", json=payload(400), headers={"Authorization": "Bearer fixture-only"}
        )
    assert response.status_code == 422 and not model.calls


def test_ignored_grammar_is_retained_as_failure(selected, tmp_path):
    model = model_fixture()
    model.corrupt = True
    client, *_ = make_client(selected, tmp_path, model)
    with client:
        response = client.post(
            "/v1/systemone", json=payload(), headers={"Authorization": "Bearer fixture-only"}
        )
    assert response.status_code == 502
    assert len(model.calls) == 1 and model.contexts == ["enter", "exit"] and not model.hooks
    assert "invented" in response.text


def test_wrong_native_prompt_envelope_refused_before_generation(selected, tmp_path):
    model = model_fixture()
    model.processor.native_envelope = False
    client, *_ = make_client(selected, tmp_path, model)
    with client:
        response = client.post(
            "/v1/systemone", json=payload(), headers={"Authorization": "Bearer fixture-only"}
        )
    assert response.status_code == 422 and not model.calls


def test_authentication_prevents_generation(selected, tmp_path):
    model = model_fixture()
    client, *_ = make_client(selected, tmp_path, model)
    with client:
        response = client.post("/v1/systemone", json=payload())
    assert response.status_code == 401 and not model.calls


@pytest.mark.parametrize("raise_body", [False, True])
def test_final_source_refusal_runs_on_both_context_exit_paths(monkeypatch, raise_body):
    import hashlib

    blob = (HERE / "selection.json").read_bytes()
    source_path = json.loads(blob)["source"]["path"]
    original, exiting = selector.checked, [False]
    primary = RuntimeError("owned fixture body failed")

    def checked(ref):
        if exiting[0] and ref["path"] == source_path:
            raise ValueError("fixture final source refused")
        return original(ref)

    monkeypatch.setattr(selector, "checked", checked)
    with pytest.raises(RuntimeError if raise_body else ValueError) as caught:
        with selector.selected(blob, hashlib.sha256(blob).hexdigest(), profile=selector.PROFILE):
            exiting[0] = True
            if raise_body:
                raise primary
    if raise_body:
        assert caught.value is primary
        assert any("Final nativeV2 FIND source validation failed" in n for n in primary.__notes__)
    else:
        assert str(caught.value) == "fixture final source refused"
    assert not any(n == "s1" or n.startswith("s1.") for n in sys.modules)
    assert "_selected_s1_find_native_v2" not in sys.modules
    assert "jev_field_syntax_helper" not in sys.modules


def test_changed_large_source_is_read_with_a_real_bound(monkeypatch, tmp_path):
    target = tmp_path / "oversized.py"
    target.write_bytes(b"x" * (2 * 1024**2))
    original_open, requested = Path.open, []

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def read(self, size=-1):
            requested.append(size)
            assert size == 1024**2 + 1
            return b"x" * size

    def opened(path, *args, **kwargs):
        return Stream() if path == target else original_open(path, *args, **kwargs)

    monkeypatch.setattr(selector, "ROOT", tmp_path)
    monkeypatch.setattr(Path, "open", opened)
    with pytest.raises(ValueError, match="Selected source bytes differ"):
        selector.checked({"path": "oversized.py", "sha256": "0" * 64, "size_bytes": 2 * 1024**2})
    assert requested == [1024**2 + 1]
