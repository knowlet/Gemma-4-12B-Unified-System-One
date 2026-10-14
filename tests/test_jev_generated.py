"""Direct native action decoding, bounded legal choices and truthful evidence."""

import asyncio
import copy
import importlib
import json
import os
import subprocess
import sys
import time
import types
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from s1 import jev_generated
from s1.api import create_app
from s1.backends import UniformBackend
from s1.errors import BackendResponseError, BackendTimeoutError, RequestValidationError
from s1.jev_generated import PROBABILITIES, JevGeneratedActionAdapter


class Tensor(np.ndarray):
    def __new__(cls, values):
        return np.asarray(values).view(cls)

    def to(self, device):
        assert device == "cpu"
        return self


class Tokenizer:
    pad_token_id = 0

    def __init__(self):
        self.chunks = {0: ""}

    def pack(self, text, width):
        ids = []
        for offset in range(0, len(text), width):
            token = len(self.chunks)
            self.chunks[token] = text[offset : offset + width]
            ids.append(token)
        return ids

    def decode(self, tokens, *, skip_special_tokens):
        assert skip_special_tokens is True
        return "".join(self.chunks[int(i)] for i in tokens)


class Processor:
    def __init__(self, tok):
        self.tok, self.messages, self.flags = tok, [], []
        self.length = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(copy.deepcopy(messages))
        self.flags.append(kwargs)
        return "<user>" + messages[0]["content"][0]["text"] + "</user><assistant>"

    def __call__(self, *, text, return_tensors, add_special_tokens):
        assert return_tensors == "pt" and add_special_tokens is False
        tokens = self.tok.pack(text, 64) if self.length is None else [0] * self.length
        ids = Tensor([tokens])
        return {
            "input_ids": ids,
            "attention_mask": Tensor(np.ones_like(ids)),
            "mm_token_type_ids": Tensor(np.zeros_like(ids)),
        }


class Backend(UniformBackend):
    name = "owned-native"

    def __init__(self, model):
        self.model, self.native_calls = model, []

    def predict(self, request):
        self.native_calls.append(request)
        return super().predict(request)


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext))
    tok = Tokenizer()
    hooks, forwards, calls = [], [], []

    def forward(**inputs):
        forwards.append(inputs)
        for hook in hooks:
            hook(None, (), inputs)

    def register(hook, *, with_kwargs):
        assert with_kwargs is True
        hooks.append(hook)
        return SimpleNamespace(remove=lambda: hooks.remove(hook))

    forward.register_forward_pre_hook = register
    model = SimpleNamespace(
        name="native-base",
        revision="a" * 40,
        quantization="nf4",
        quantization_details={"mode": "nf4", "dense_modules": ["lm_head"]},
        max_context=16384,
        device="cpu",
        tok=tok,
        processor=Processor(tok),
        backbone=forward,
        generated='{"operation":"CLICK","target":"1"}',
        calls=calls,
        forwards=forwards,
        hooks=hooks,
        after_generate=None,
        fail_after=None,
        interruption=RuntimeError,
    )

    def generate(**kwargs):
        calls.append(kwargs)
        tokens = tok.pack(model.generated, 3) + [0]
        kwargs["streamer"].put(kwargs["input_ids"])
        for index, token in enumerate(tokens):
            forward(input_ids=kwargs["input_ids"] if index == 0 else Tensor([[token]]))
            kwargs["streamer"].put(Tensor([token]))
            if model.fail_after == index + 1:
                raise model.interruption("actual generation interrupted")
        kwargs["streamer"].end()
        if model.after_generate is not None:
            model.after_generate()
        return Tensor(np.concatenate([kwargs["input_ids"], np.asarray([tokens])], axis=1))

    model.lm = SimpleNamespace(generate=generate, eval=lambda: None)
    return Backend(model)


@pytest.fixture
def payload():
    goal = "Apply the requested search before opening the matching record."
    questions = {
        "operation": {
            "type": "choice",
            "criteria": {
                "CLICK": "Click",
                "TYPE_TEXT": "Fill",
                "SELECT": "Select",
                "SCROLL_DOWN": "Scroll",
                "SCROLL_UP": "Scroll up",
                "WAIT": "Wait",
                "DONE": "All requirements satisfied",
                "BLOCKED": "No supported progress",
            },
            "instructions": {"goal": goal, "rules": "Edits are not applied state."},
        },
    }
    for head, op, labels in [
        ("click_target", "CLICK", ["1", "2"]),
        ("type_text_target", "TYPE_TEXT", ["3"]),
        ("select_target", "SELECT", ["4:1", "4:2"]),
    ]:
        questions[head] = {
            "type": "choice",
            "criteria": {
                key: {"element": key, "current_value": "", "checked": False} for key in labels
            },
            "instructions": {"goal": goal, "operation": op, "rules": ["Use current observations"]},
        }
    return {
        "model": "experiment",
        "state": {
            "page": {"title": "Records", "text": "Search · Apply · Open"},
            "elements": [{"index": "3", "value": "requested query"}],
            "recent_actions": [],
        },
        "questions": questions,
    }


def adapter(backend):
    return JevGeneratedActionAdapter(backend, model_identity="experiment")


@pytest.mark.parametrize(
    "operation,target",
    [
        ("CLICK", "2"),
        ("TYPE_TEXT", "3"),
        ("SELECT", "4:2"),
        ("SCROLL_DOWN", None),
        ("SCROLL_UP", None),
        ("WAIT", None),
        ("DONE", None),
        ("BLOCKED", None),
    ],
)
def test_all_upstream_operations_bind_exact_generated_action(backend, payload, operation, target):
    original = copy.deepcopy(payload)
    backend.model.generated = json.dumps({"operation": operation, "target": target})
    result = adapter(backend).predict(payload)
    assert payload == original and not backend.native_calls
    assert len(backend.model.calls) == 1 and not backend.model.hooks
    answers = result["answers"]
    assert answers["operation"]["choice"] == operation
    head = jev_generated.TARGETS.get(operation)
    assert set(answers) == {"operation", *([head] if head else [])}
    assert result["unused_target_heads_status"] == "not_evaluated"
    assert set(result["unused_target_heads"]) == set(payload["questions"]) - set(answers)
    for key, answer in answers.items():
        chosen = operation if key == "operation" else target
        assert answer["probabilities"] == {
            label: float(label == chosen) for label in payload["questions"][key]["criteria"]
        }
        assert (
            answer["probabilities_semantics"] == result["probabilities_semantics"] == PROBABILITIES
        )
    assert result["model"] == backend.model.name and result["revision"] == backend.model.revision
    generation = result["generation"]
    assert generation["policy"] == "greedy_legal_action_v2"
    assert generation["decoded_action"] == {"operation": operation, "target": target}
    assert generation["raw_output"] == backend.model.generated
    assert generation["forward_calls"] == len(backend.model.forwards) > 0
    assert generation["usage"]["output_tokens"] == len(generation["raw_token_ids"])
    assert generation["usage"]["input_tokens"] == len(generation["prompt_token_ids"])
    assert generation["generation_elapsed_ms"] >= 0 and generation["elapsed_ms"] >= 0
    kwargs = backend.model.calls[0]
    assert {
        key: kwargs[key]
        for key in [
            "max_new_tokens",
            "do_sample",
            "num_beams",
            "num_return_sequences",
            "return_dict_in_generate",
            "use_cache",
        ]
    } == {
        "max_new_tokens": 32,
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
        "return_dict_in_generate": False,
        "use_cache": True,
    }
    assert backend.model.processor.flags == [
        {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
    ]
    rendered_source = backend.model.processor.messages[0][0]["content"][0]["text"].split("\n", 1)[1]
    assert json.loads(rendered_source) == original
    assert result["execution"]["candidate_forward_calls"] == 0


def test_full_400_candidates_and_rich_fields_reach_model_in_original_order(backend, payload):
    full = {
        str(i): {"element": f"Observed {i}", "value": {"nested": [i, False]}} for i in range(1, 401)
    }
    payload["questions"]["click_target"]["criteria"] = full
    backend.model.generated = '{"operation":"CLICK","target":"400"}'
    result = adapter(backend).predict(payload)
    supplied = json.loads(
        backend.model.processor.messages[0][0]["content"][0]["text"].split("\n", 1)[1]
    )
    assert supplied == payload
    assert list(supplied["questions"]["click_target"]["criteria"]) == list(full)
    assert len(result["answers"]["click_target"]["probabilities"]) == 400
    assert result["answers"]["click_target"]["choice"] == "400"


def test_all_three_wide_heads_preserved_even_when_only_one_is_used(backend, payload):
    for head in jev_generated.TARGETS.values():
        payload["questions"][head]["criteria"] = {
            str(i): {"element": f"Observed {i}", "current_value": ""} for i in range(400)
        }
    backend.model.generated = '{"operation":"TYPE_TEXT","target":"399"}'
    result = adapter(backend).predict(payload)
    assert all(
        len(labels) == 400
        for key, labels in result["generation"]["candidate_labels"].items()
        if key != "operation"
    )
    assert set(result["answers"]) == {"operation", "type_text_target"}
    source = backend.model.processor.messages[0][0]["content"][0]["text"].split("\n", 1)[1]
    assert json.loads(source) == payload


@pytest.mark.parametrize(
    "generated",
    [
        '{"operation":"CLICK","target":null}',
        '{"operation":"CLICK","target":1}',
        '{"operation":"CLICK","target":"4:1"}',
        '{"operation":"CLICK","target":"absent"}',
        '{"operation":"DONE","target":"1"}',
        '{"operation":"INVENTED","target":null}',
        '{"operation":true,"target":null}',
        '{"operation":[],"target":null}',
        '{"operation":"DONE","target":null,"extra":true}',
        '{"operation":"DONE"}',
        '{"operation":"DONE","operation":"CLICK","target":null}',
        '{"operation":"DONE","target":NaN}',
        '{"operation":"DONE","target":Infinity}',
        '```json\n{"operation":"DONE","target":null}\n```',
        'text {"operation":"DONE","target":null}',
        '{"operation":"DONE","target":null} {}',
        "null",
        "[]",
        "",
        "thought\nthought",
    ],
)
def test_invalid_decoding_retains_actual_raw_and_never_falls_back(backend, payload, generated):
    backend.model.generated = generated
    with pytest.raises(BackendResponseError) as caught:
        adapter(backend).predict(payload)
    evidence = caught.value.generation
    assert evidence["raw_output"] == generated
    assert evidence["raw_token_ids"] is not None
    assert evidence["phase"] == "output_validation"
    assert evidence["forward_calls"] == len(backend.model.forwards)
    assert len(backend.model.calls) == 1 and not backend.native_calls and not backend.model.hooks


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(model="other-model"),
        lambda p: p.update(media=[{"type": "image", "data": "a"}]),
        lambda p: p.update(extra=True),
        lambda p: p["questions"]["click_target"].update(extra=True),
        lambda p: p["questions"]["click_target"].update(id="wrong"),
        lambda p: p["questions"]["click_target"].update(options=["1", "2"]),
        lambda p: p["questions"]["click_target"].update(
            criteria={str(i): "item" for i in range(401)}
        ),
        lambda p: p["questions"]["click_target"].update(criteria=["1", "2", "1"]),
        lambda p: p["questions"]["click_target"].update(
            criteria={**{str(i): "ok" for i in range(60)}, "late": 3}
        ),
        lambda p: p["questions"]["operation"]["criteria"].update(DELETE="Delete"),
        lambda p: p["questions"].pop("select_target"),
        lambda p: p["state"].update(nan=float("nan")),
        lambda p: p["state"].update(value="\ud800"),
        lambda p: p["state"].update(value="x" * 1_000_000),
        lambda p: p["questions"]["click_target"].update(criteria={"x" * 129: "item"}),
        lambda p: p["questions"]["click_target"].update(criteria={"1": "x" * 16001}),
    ],
)
def test_full_original_payload_validation_precedes_every_forward(backend, payload, mutation):
    mutation(payload)
    with pytest.raises(RequestValidationError):
        adapter(backend).predict(payload)
    assert not backend.model.calls and not backend.model.forwards


@pytest.mark.parametrize("model_limit,prompt_length", [(16384, 8161), (200, 169)])
def test_context_reserves_32_tokens_before_forward(backend, payload, model_limit, prompt_length):
    backend.model.max_context = model_limit
    backend.model.processor.length = prompt_length
    with pytest.raises(RequestValidationError, match="nothing truncated") as caught:
        adapter(backend).predict(payload)
    assert caught.value.generation["usage"]["input_tokens"] == prompt_length
    assert not backend.model.calls and not backend.model.forwards


@pytest.mark.parametrize("interruption", [RuntimeError, KeyboardInterrupt, asyncio.CancelledError])
def test_interrupted_generation_preserves_actual_emitted_prefix_and_forward_attempts(
    backend, payload, interruption
):
    backend.model.fail_after = 3
    backend.model.interruption = interruption
    with pytest.raises(interruption, match="interrupted") as caught:
        adapter(backend).predict(payload)
    e = caught.value.generation
    assert e["phase"] == "generation" and e["forward_calls"] == 3
    assert len(e["raw_token_ids"]) == e["usage"]["output_tokens"] == 3
    assert backend.model.generated.startswith(e["raw_output"])
    assert e["output_tokens_scope"] == "observed_emitted_before_failure"
    assert not backend.model.hooks and not backend.native_calls


def test_late_complete_generation_retained_but_never_returned_as_action(
    backend, payload, monkeypatch
):
    clock = [0.0]
    monkeypatch.setattr(
        jev_generated,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], perf_counter=time.perf_counter),
    )
    backend.model.after_generate = lambda: clock.__setitem__(0, 20.001)
    with pytest.raises(BackendTimeoutError) as caught:
        adapter(backend).predict(payload)
    e = caught.value.generation
    assert e["raw_output"] == backend.model.generated and e["forward_calls"] > 0
    assert e["budget_seconds"] == 20 and e["elapsed_ms"] == pytest.approx(20001)
    assert "decoded_action" not in e


def test_adapter_lock_wait_counts_in_unchanged_budget_before_any_forward(
    backend, payload, monkeypatch
):
    clock = [0.0]
    monkeypatch.setattr(
        jev_generated,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], perf_counter=time.perf_counter),
    )

    @contextmanager
    def wait_then_acquire():
        clock[0] = 20.001
        yield

    backend.model.action_adapter = SimpleNamespace(
        metadata={"manifest_sha256": "actual-adapter-identity"}, decision_context=wait_then_acquire
    )
    with pytest.raises(BackendTimeoutError) as caught:
        adapter(backend).predict(payload)
    assert caught.value.generation["action_adapter"]["manifest_sha256"] == "actual-adapter-identity"
    assert caught.value.generation["phase"] == "adapter_lock_wait"
    assert caught.value.generation["forward_calls"] == 0
    assert not backend.model.calls and not backend.model.hooks


def test_observer_mutation_failure_cannot_change_action_and_slow_observer_blocks_forward(
    backend, payload, monkeypatch
):
    def malicious(event):
        event["data"].clear()
        raise RuntimeError("observer only")

    result = adapter(backend).predict(payload, evidence_sink=malicious)
    assert result["answers"]["click_target"]["choice"] == "1"
    assert result["generation"]["observer_errors"]
    backend.model.calls.clear()
    clock = [0.0]
    monkeypatch.setattr(
        jev_generated,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], perf_counter=time.perf_counter),
    )
    with pytest.raises(BackendTimeoutError):
        adapter(backend).predict(payload, evidence_sink=lambda _event: clock.__setitem__(0, 21))
    assert not backend.model.calls and not backend.model.hooks


def test_api_auth_strict_core_and_generated_failure_journal(backend, payload):
    from fastapi.testclient import TestClient

    events = []
    app = create_app(
        backend,
        jev_compatible=True,
        jev_generated_actions=True,
        model_identity="experiment",
        api_key="secret",
        request_evidence_sink=events.append,
    )
    with TestClient(app) as client:
        assert client.post("/v1/systemone", json=payload).status_code == 401
        assert not backend.model.calls
        headers = {"Authorization": "Bearer secret"}
        assert client.post("/decide", json=payload, headers=headers).status_code == 422
        response = client.post("/v1/systemone", json=payload, headers=headers)
        assert (
            response.status_code == 200
            and response.json()["generation"]["decoded_action"]["target"] == "1"
        )
        backend.model.generated = "invalid JSON"
        response = client.post("/v1/systemone", json=payload, headers=headers)
        assert response.status_code == 502
        assert response.json()["detail"]["generation"]["raw_output"] == "invalid JSON"
    assert [e["event"] for e in events].count("server_returned") == 1
    assert [e["event"] for e in events].count("server_failed") == 1
    assert any(
        e["event"] == "generated_action_checkpoint" and e["data"]["event"] == "raw_returned"
        for e in events
    )
    assert not backend.native_calls


@pytest.mark.parametrize(
    "options",
    [
        {"jev_compatible": False},
        {"jev_compatible": True, "jev_hierarchical": True},
        {"jev_compatible": True, "jev_hierarchical_batch": True},
    ],
)
def test_api_rejects_mixed_policies_before_generation(backend, options):
    with pytest.raises(ValueError):
        create_app(backend, jev_generated_actions=True, **options)
    assert not backend.model.calls


@pytest.mark.parametrize(
    "key,value",
    [
        ("prompt_mode", "reasoned"),
        ("prompt_mode", "grounded"),
        ("decision_readout", object()),
        ("prefix_cache", True),
        ("prefix_suffix_batch_size", 2),
        ("analysis_style", "compact"),
    ],
)
def test_backend_policies_are_not_silently_ignored(backend, key, value):
    setattr(backend, key, value)
    with pytest.raises(ValueError):
        create_app(backend, jev_compatible=True, jev_generated_actions=True)
    assert not backend.model.calls


def test_default_and_ordinary_core_do_not_invoke_generation(backend, payload):
    from fastapi.testclient import TestClient

    native = {
        "state": "ordinary",
        "questions": [{"id": "q", "instructions": "Choose", "criteria": ["a", "b"]}],
    }
    with TestClient(
        create_app(backend, jev_compatible=True, model_identity="experiment")
    ) as client:
        assert client.post("/v1/systemone", json=payload).status_code == 200
    assert len(backend.native_calls) == 1 and not backend.model.calls
    with TestClient(create_app(backend, jev_compatible=True, jev_generated_actions=True)) as client:
        assert client.post("/decide", json=native).status_code == 200
        assert client.post("/decide/batch", json=[native]).status_code == 200
        native["questions"][0]["criteria"] = [str(i) for i in range(53)]
        assert client.post("/decide", json=native).status_code == 422
    assert len(backend.native_calls) == 3 and not backend.model.calls


def test_api_maps_model_mismatch_and_late_generation_with_full_evidence(
    backend, payload, monkeypatch
):
    from fastapi.testclient import TestClient

    clock = [0.0]
    monkeypatch.setattr(
        jev_generated,
        "time",
        SimpleNamespace(monotonic=lambda: clock[0], perf_counter=time.perf_counter),
    )
    with TestClient(
        create_app(
            backend, jev_compatible=True, jev_generated_actions=True, model_identity="experiment"
        )
    ) as client:
        wrong = {**payload, "model": "different"}
        assert client.post("/v1/systemone", json=wrong).status_code == 422
        assert not backend.model.calls
        backend.model.after_generate = lambda: clock.__setitem__(0, 21)
        response = client.post("/v1/systemone", json=payload)
    assert response.status_code == 504
    assert response.json()["detail"]["generation"]["raw_output"] == backend.model.generated


def test_invalid_utf8_request_with_journal_keeps_primary_validation_error(
    backend, payload, tmp_path
):
    from fastapi.testclient import TestClient

    from s1.browser_server_evidence import ServerEvidenceJournal

    payload["state"]["invalid"] = "\ud800"
    events = []
    journal = ServerEvidenceJournal(tmp_path / "journal")

    def sink(event):
        events.append(event)
        journal(event)

    app = create_app(
        backend,
        jev_compatible=True,
        jev_generated_actions=True,
        model_identity="experiment",
        request_evidence_sink=sink,
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/systemone",
            content=json.dumps(payload, ensure_ascii=True),
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 422
    assert "finite UTF-8 JSON" in response.json()["detail"]["message"]
    assert events[0]["data"]["original_request_sha256"] is None
    assert events[-1]["event"] == "server_failed"
    assert app.state.request_evidence_errors
    assert journal.snapshot()["requests"][0]["sealed"] is False
    assert not backend.model.forwards


@pytest.mark.inference
def test_genuine_tiny_cpu_native_generate_retains_invalid_tokens_without_classifier(payload):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(73)
    config = transformers.GPT2Config(
        vocab_size=64,
        n_layer=1,
        n_embd=16,
        n_head=2,
        n_positions=128,
        bos_token_id=1,
        eos_token_id=None,
        pad_token_id=0,
    )
    lm = transformers.GPT2LMHeadModel(config).eval()
    weight = lm.get_output_embeddings().weight.detach().clone()

    class NativeProcessor:
        def apply_chat_template(self, messages, **kwargs):
            assert json.loads(messages[0]["content"][0]["text"].split("\n", 1)[1]) == payload
            assert kwargs == {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            return "offline tiny generation boundary"

        def __call__(self, **_kwargs):
            ids = torch.tensor([[1, 2, 3, 4]])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    model = SimpleNamespace(
        name="offline-tiny-native",
        revision="test-in-memory",
        quantization="none",
        quantization_details={"mode": "none"},
        lm=lm,
        backbone=lm.transformer,
        processor=NativeProcessor(),
        device="cpu",
        max_context=128,
        tok=SimpleNamespace(
            pad_token_id=0,
            decode=lambda tokens, **_: "".join(chr(65 + int(i) % 26) for i in tokens),
        ),
    )
    with pytest.raises(BackendResponseError) as caught:
        adapter(Backend(model)).predict(payload)
    e = caught.value.generation
    assert e["phase"] == "output_validation"
    assert e["usage"] == {"input_tokens": 4, "output_tokens": 32}
    assert len(e["raw_token_ids"]) == len(e["forwards"]) == e["forward_calls"] == 32
    assert e["forwards"][0]["input_shape"] == [1, 4]
    assert all(row["input_shape"] == [1, 1] for row in e["forwards"][1:])
    assert torch.equal(weight, lm.get_output_embeddings().weight)
    assert not lm.transformer._forward_pre_hooks
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize(
    "operation,target,choice",
    [
        ("CLICK", "80", "observed-80"),
        ("SCROLL_UP", None, "scroll_up"),
    ],
)
def test_actual_pinned_upstream_choose_accepts_only_the_selected_head(
    backend, monkeypatch, operation, target, choice
):
    from fastapi.testclient import TestClient

    configured = os.environ.get("JEV_UPSTREAM_PATH")
    if not configured:
        pytest.skip("set JEV_UPSTREAM_PATH to the existing pinned offline checkout")
    checkout = Path(configured)
    assert (
        subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        == "1231850a0bf1a0c0341fe408ef1668dbbfdfac46"
    )
    for name in ("model.py", "questions.py"):
        relative = "jev_ultrafast/" + name
        assert (checkout / relative).read_bytes() == subprocess.check_output(
            ["git", "-C", str(checkout), "show", "HEAD:" + relative]
        )
    name = "_s1_generated_upstream"
    package = types.ModuleType(name)
    package.__path__ = [str(checkout / "jev_ultrafast")]
    monkeypatch.setitem(sys.modules, name, package)
    upstream = importlib.import_module(name + ".model")
    for imported in (name + ".model", name + ".questions"):
        monkeypatch.setitem(sys.modules, imported, sys.modules[imported])
    page = {
        "url": "http://example.invalid/records",
        "title": "Records",
        "text": "Choose a record",
        "actions": [
            {
                "id": f"observed-{i}",
                "node": i,
                "kind": "click",
                "label": f"Record {i}",
                "role": "button",
            }
            for i in range(1, 81)
        ],
    }
    page["actions"].extend(
        [
            {
                "id": "unused-fill",
                "node": 81,
                "kind": "fill",
                "label": "Search",
                "role": "searchbox",
                "value": "",
            },
            {
                "id": "unused-select",
                "node": 82,
                "kind": "select",
                "label": "Category → A",
                "role": "combobox",
                "value": "A",
                "current_value": "All",
            },
            {"id": "scroll_up", "kind": "scroll", "label": "Scroll up", "delta": -560},
            {"id": "scroll_down", "kind": "scroll", "label": "Scroll down", "delta": 560},
            {"id": "wait", "kind": "wait", "label": "Wait"},
        ]
    )
    backend.model.generated = json.dumps({"operation": operation, "target": target})
    with TestClient(
        create_app(
            backend, jev_compatible=True, jev_generated_actions=True, model_identity="experiment"
        )
    ) as client:

        def local_post(_url, _key, body):
            response = client.post("/v1/systemone", json=body)
            assert response.status_code == 200, response.text
            return response.json()

        monkeypatch.setattr(upstream, "post_json", local_post)
        monkeypatch.setenv("TYPESAFE_MODEL", "experiment")
        monkeypatch.setenv("TYPESAFE_API_KEY", "unused-test")
        result = upstream.choose(page, "Open the requested record.", [])
    assert result["choice"] == choice and result["target"] == target
    assert set(result["raw_answers"]) == {"operation", *(["click_target"] if target else [])}
    assert len(result["target_probabilities"]) == (80 if target else 0)
    assert set(result["operation_probabilities"]) == jev_generated.OPERATIONS
    assert result["raw_answers"]["operation"]["probabilities_semantics"] == PROBABILITIES
