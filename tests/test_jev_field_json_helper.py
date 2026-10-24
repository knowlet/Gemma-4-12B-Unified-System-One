"""Fake native LM plus actual cached tokenizer; no Torch/weights/HTTP/browser."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/jev_field_json_helper.py"
sys.path.insert(0, str(ROOT / "scripts"))
try:
    import jev_field_json_helper as H
finally:
    sys.path.remove(str(ROOT / "scripts"))
G = H.grammar
NATIVE = (
    ROOT
    / "artifacts/jev-local-hf/hub/models--knowlet--Gemma-4-12B-Unified-System-One/snapshots/a66f836b56605039fe040f330180e336d19b3362"
)
CLOSED = ROOT / "artifacts/jev-runtime/field-json-grammar-20261018-01/full-compile-01/cache"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Tensor:
    def __init__(self, data):
        self.data = copy.deepcopy(data)

    @property
    def shape(self):
        return (
            (len(self.data), len(self.data[0]))
            if self.data and isinstance(self.data[0], list)
            else (len(self.data),)
        )

    def tolist(self):
        return copy.deepcopy(self.data)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, column = key
            return Tensor(self.data[row][column])
        return Tensor(self.data[key])


class Batch(dict):
    def to(self, device):
        self.device = device
        return self


class Clock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value


class Tok:
    pad_token_id, eos_token_id, bos_token_id = 0, 1, 2
    clean_up_tokenization_spaces = False

    def __init__(self, backend, clock):
        self.backend_tokenizer, self.clock = backend, clock
        self.prompt = [2, 1005, 1005]
        self.calls, self.decode_calls = [], []
        self.decode_delay = self.tokenization_delay = 0
        self.decode_failure = None

    def get_vocab(self):
        return self.backend_tokenizer.get_vocab()

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        self.clock.value += self.tokenization_delay
        return Batch(
            input_ids=Tensor([self.prompt]), attention_mask=Tensor([[1] * len(self.prompt)])
        )

    def decode(self, value, **kwargs):
        values = value.tolist() if hasattr(value, "tolist") else value
        self.decode_calls.append((copy.deepcopy(values), kwargs))
        self.clock.value += self.decode_delay
        if self.decode_failure:
            raise self.decode_failure
        return self.backend_tokenizer.decode(values, **kwargs)


class Processor:
    def __init__(self, tok, clock):
        self.tokenizer, self.clock, self.calls = tok, clock, []
        self.delay = 0

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        self.clock.value += self.delay
        return "FAKE_NATIVE_TEMPLATE\n" + messages[0]["content"]


class LM:
    def __init__(self, clock, generation):
        self.clock, self.generation_config = clock, SimpleNamespace(**generation)
        self.tokens, self.calls, self.events = [], [], []
        self.adapter_enabled = True
        self.ignore_callback = self.ignore_streamer = False
        self.reject_allowed = True
        self.fail_at, self.fail_error = None, RuntimeError("fake native failure")
        self.post_generation_delay = 0
        self.changed_output_prompt = False
        self.extra_output = []
        self.callback_batch_id = 0

    def get_output_embeddings(self):
        return SimpleNamespace(weight=SimpleNamespace(shape=(262144, 42)))

    def generate(self, **kwargs):
        assert self.adapter_enabled is False
        self.calls.append(kwargs)
        prompt = kwargs["input_ids"].tolist()[0]
        capture = kwargs["streamer"]
        if not self.ignore_streamer:
            capture.put(Tensor([prompt]))
        emitted = []
        for index, token_id in enumerate(self.tokens):
            if self.fail_at == index:
                raise self.fail_error
            if not self.ignore_callback:
                allowed = kwargs["prefix_allowed_tokens_fn"](
                    self.callback_batch_id, Tensor(prompt + emitted)
                )
                assert allowed == sorted(set(allowed))
                self.events.append(
                    {
                        "index": index,
                        "allowed_count": len(allowed),
                        "chosen_allowed": token_id in allowed,
                    }
                )
                if self.reject_allowed and token_id not in allowed:
                    raise ValueError("fake prescribed token was refused by the native constraint")
            emitted.append(token_id)
            if not self.ignore_streamer:
                capture.put(Tensor([token_id]))
            if token_id in G.EOS_IDS:
                break
        if not self.ignore_streamer:
            capture.end()
        self.clock.value += self.post_generation_delay
        output_prompt = list(prompt)
        if self.changed_output_prompt:
            output_prompt[0] = 1005
        return Tensor([output_prompt + emitted + self.extra_output])


class Guard:
    """Explicit fake guard, source-bound but never claimed as public model proof."""

    def __init__(self, model, clock):
        self.model, self.clock, self.events = model, clock, []
        self.enter_delay = self.restore_delay = 0
        self.fail_restore = self.fail_disable = False
        self.sticky = False

    def check(self, *, full=False):
        self.events.append(("check", full, self.model.lm.adapter_enabled))
        if self.sticky or self.model.lm.adapter_enabled is not True:
            raise ValueError("fake guard enabled/restore integrity failed")

    def _check(self, *, disabled=False):
        self.events.append(("disabled_check", disabled, self.model.lm.adapter_enabled))
        if disabled and self.model.lm.adapter_enabled is not False:
            raise ValueError("fake adapter was not disabled")

    @contextmanager
    def base_context(self):
        self.check()
        self.clock.value += self.enter_delay
        if not self.fail_disable:
            self.model.lm.adapter_enabled = False
        primary = None
        try:
            yield
        except BaseException as exc:
            primary = exc
        finally:
            self.model.lm.adapter_enabled = True
            self.clock.value += self.restore_delay
            if self.fail_restore:
                self.sticky = True
                if primary is None:
                    primary = OSError("fake restore failure")
                else:
                    primary.add_note("fake restore failure")
        if primary is not None:
            raise primary


@pytest.fixture(scope="module")
def assets():
    sys.path.insert(0, str(ROOT / "artifacts/jev-runtime/overlay"))
    try:
        from tokenizers import Tokenizer
    finally:
        sys.path.remove(str(ROOT / "artifacts/jev-runtime/overlay"))
    backend = Tokenizer.from_file(str(NATIVE / "tokenizer.json"))
    generation = json.loads((NATIVE / "generation_config.json").read_bytes())
    selection = {
        "cache_dir": CLOSED,
        "tokenizer_path": NATIVE / "tokenizer.json",
        "generation_path": NATIVE / "generation_config.json",
        "expected_cache_manifest_sha256": "6e5ac130fa1f5b6bb315f71d17cb8d3d1b8f2552eef319a1e695c06ebf5b0dc6",
        "expected_tokenizer_sha256": sha(NATIVE / "tokenizer.json"),
        "expected_generation_sha256": sha(NATIVE / "generation_config.json"),
        "expected_grammar_source_sha256": sha(ROOT / "scripts/jev_field_json_grammar.py"),
        "expected_helper_source_sha256": sha(SOURCE),
        "expected_guard_source_sha256": sha(__file__),
        "expected_decoder_identity": G.DECODER_IDENTITY,
    }
    return backend, generation, selection


@pytest.fixture
def factory(assets, monkeypatch):
    backend, generation, selection = assets
    monkeypatch.setattr(H, "_inference_context", nullcontext)

    def build(*, text='{"text":"field x"}', eos=106, prepare=True):
        clock = Clock()
        tok = Tok(backend, clock)
        lm = LM(clock, copy.deepcopy(generation))
        processor = Processor(tok, clock)
        model = SimpleNamespace(
            lm=lm,
            tok=tok,
            processor=processor,
            name="fake-pinned-base",
            revision="fake-revision",
            quantization="bf16",
            max_context=8192,
            device="fake-device",
        )
        model.action_adapter = Guard(model, clock)
        lm.tokens = backend.encode(text, add_special_tokens=False).ids + (
            [] if eos is None else [eos]
        )
        prepared = H.prepare_helper(model, **selection) if prepare else None
        return SimpleNamespace(
            model=model,
            prepared=prepared,
            clock=clock,
            sink=[],
            context={
                "goal": "An independently supplied value",
                "page": {"text": "untrusted page"},
                "history": [],
            },
            selection=selection,
        )

    return build


def run(case, **kwargs):
    return H.generate_field_text(
        case.prepared,
        case.context,
        evidence_sink=kwargs.pop("evidence_sink", case.sink.append),
        clock=case.clock,
        **kwargs,
    )


def failed(case, **kwargs):
    with pytest.raises(H.FieldJSONError) as error:
        run(case, **kwargs)
    evidence = error.value.generation
    assert evidence["success"] is evidence["response_authorized"] is False
    assert case.model.lm.adapter_enabled is True
    return evidence


@pytest.mark.frozen_decoder
def test_actual_decoder_full_inventory_and_closed_cache_prepared_before_any_forward(factory):
    case = factory()
    receipt = case.prepared.receipt()
    assert receipt["prepared"] and not receipt["qualified"]
    assert receipt["native_vocab_size"] == 262144 and receipt["model_forward_calls"] == 0
    assert receipt["actual_decoder"]["version"] == "0.23.3"
    assert sha(receipt["actual_decoder"]["binary_path"]) == G.DECODER_IDENTITY["binary_sha256"]
    assert (
        receipt["mapping_rows_sha256"]
        == "eb334cd54294f30a74c64d0130efebf3818546a292c1425242dada2fbca1f217"
    )
    assert not case.model.lm.calls
    assert ("check", True, True) in case.model.action_adapter.events


@pytest.mark.frozen_decoder
def test_same_original_prompt_template_tokenization_generate_and_restored_base_contract(factory):
    case = factory(text='{"text":"  文字 field x 😀  "}')
    value, meta = run(case)
    assert value == "  文字 field x 😀  "
    assert meta["success"] and meta["response_authorized"] and meta["adapter_restore_complete"]
    assert meta["action_adapter_state"] == "disabled_for_pinned_base_helper"
    assert meta["terminated_by_actual_EOS"] and meta["grammar_terminal_accepted"]
    assert meta["raw_token_ids"] == case.model.lm.tokens
    assert meta["usage"]["output_tokens"] == len(case.model.lm.tokens)
    messages, template_kwargs = case.model.processor.calls[0]
    assert messages == [
        {
            "role": "user",
            "content": H.INSTRUCTIONS + "\n" + json.dumps(case.context, ensure_ascii=False),
        }
    ]
    assert template_kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    assert case.model.tok.calls[0][1] == {"return_tensors": "pt", "add_special_tokens": False}
    native = case.model.lm.calls[0]
    assert set(native) == {
        "input_ids",
        "attention_mask",
        "max_new_tokens",
        "do_sample",
        "num_beams",
        "return_dict_in_generate",
        "use_cache",
        "pad_token_id",
        "streamer",
        "prefix_allowed_tokens_fn",
    }
    assert {
        k: native[k]
        for k in [
            "max_new_tokens",
            "do_sample",
            "num_beams",
            "return_dict_in_generate",
            "use_cache",
            "pad_token_id",
        ]
    } == {
        "max_new_tokens": 128,
        "do_sample": False,
        "num_beams": 1,
        "return_dict_in_generate": False,
        "use_cache": True,
        "pad_token_id": 0,
    }
    assert len(meta["callback_events"]) == len(case.model.lm.tokens)
    assert all(event["chosen_allowed"] for event in case.model.lm.events)
    assert any(event["allowed_count"] > 100000 for event in case.model.lm.events)
    assert [row["event"] for row in case.sink] == [
        "raw_tokens_returned",
        "raw_original_decoder_output",
        "request_closed_before_response_gate",
    ]
    assert case.sink[0]["generation"]["raw_output"] is None
    assert case.sink[1]["generation"]["raw_output"] == meta["raw_output"]
    assert case.sink[-1]["generation"]["adapter_restore_complete"]
    assert case.sink[-1]["generation"]["response_authorized"] is False


def test_prompt_instruction_literal_matches_original_without_modifying_old_helper():
    path = ROOT / "src/s1/browser_text.py"
    assert sha(path) == "381299d3961afba4ba2a296b1ce9ee5e4b137f5494b14cb4291b70340462b577"
    tree = ast.parse(path.read_bytes())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "generate_field_text"
    )
    instruction = next(
        node
        for node in function.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "instructions" for target in node.targets
        )
    )
    assert H.INSTRUCTIONS == ast.literal_eval(instruction.value)


@pytest.mark.parametrize("eos", G.EOS_IDS)
@pytest.mark.frozen_decoder
def test_all_three_actual_eos_count_and_terminate_without_synthetic_tokens(factory, eos):
    case = factory(eos=eos)
    _, meta = run(case)
    assert meta["raw_token_ids"][-1] == eos
    assert meta["usage"]["output_tokens"] == len(case.model.lm.tokens)
    assert meta["terminated_by_actual_EOS"] and meta["finish_reason"] == "actual_EOS"


@pytest.mark.parametrize("at_cap", [False, True])
@pytest.mark.frozen_decoder
def test_complete_json_without_real_eos_is_refused_even_at128(factory, at_cap):
    case = factory(eos=None)
    if at_cap:
        space = case.model.tok.backend_tokenizer.token_to_id("\u2581")
        case.model.lm.tokens += [space] * (128 - len(case.model.lm.tokens))
    meta = failed(case)
    assert meta["failure_stage"] == "termination"
    assert meta["native_generation_returned"] and meta["generation_complete"]
    assert meta["terminated_by_actual_EOS"] is False
    assert meta["raw_token_ids"] == case.model.lm.tokens
    assert json.loads(meta["raw_output"]) == {"text": "field x"}
    assert meta["finish_reason"] == ("length_without_EOS" if at_cap else "returned_without_EOS")


@pytest.mark.frozen_decoder
def test_actual_eos_can_be_exact128th_emitted_token(factory):
    case = factory(eos=None)
    space = case.model.tok.backend_tokenizer.token_to_id("\u2581")
    case.model.lm.tokens += [space] * (127 - len(case.model.lm.tokens)) + [106]
    value, meta = run(case)
    assert value == "field x" and meta["usage"]["output_tokens"] == 128
    assert len(meta["raw_token_ids"]) == 128 and meta["raw_token_ids"][-1] == 106


@pytest.mark.parametrize("raw", ['{"text":null}', '{"text":""}', '{"text":"   "}'])
@pytest.mark.frozen_decoder
def test_nullable_grammar_is_not_permission_to_return_invalid_field_values(factory, raw):
    case = factory(text=raw)
    meta = failed(case)
    assert meta["failure_stage"] == "consumer" and meta["grammar_terminal_accepted"]
    assert meta["raw_output"] == raw and meta["raw_token_ids"] == case.model.lm.tokens


@pytest.mark.parametrize(
    "raw",
    [
        'thought {"text":"x"}',
        '```json\n{"text":"x"}\n```',
        '{"text":"x","text":"y"}',
        '{"text":"\\ud800"}',
        '{"text":1}',
        '{"text":"' + "_" * 2001 + '"}',
    ],
)
def test_strict_consumer_does_not_strip_repair_or_accept_oversized_text(raw):
    with pytest.raises((ValueError, UnicodeError)):
        H._consume_value(raw)


@pytest.mark.frozen_decoder
def test_over2000_actual_native_value_reaches_consumer_then_is_refused(factory):
    case = factory()
    backend = case.model.tok.backend_tokenizer
    # Actual generation can use legal native pieces other than the tokenizer's
    # canonical re-encoding. ID167 is31 ordinary spaces, so the field is x plus
    # exactly2000 spaces while the genuine emitted ID count remains below128.
    assert backend.decode([167], skip_special_tokens=True) == " " * 31
    prefix = [backend.token_to_id(piece) for piece in ["{", '"', "text", '"', ":", '"', "x"]]
    suffix = [backend.token_to_id(piece) for piece in ['"', "}"]]
    case.model.lm.tokens = (
        prefix + [167] * 64 + [backend.token_to_id("\u2581")] * 16 + suffix + [106]
    )
    assert len(case.model.lm.tokens) <= 128
    meta = failed(case)
    assert meta["grammar_terminal_accepted"] and meta["failure_stage"] == "consumer"
    assert len(json.loads(meta["raw_output"])["text"]) == 2001


@pytest.mark.parametrize("fault", ["no_callback", "no_streamer", "changed_prompt", "extra_return"])
@pytest.mark.frozen_decoder
def test_returned_native_tokens_are_bound_to_each_actual_constraint_and_capture(factory, fault):
    case = factory()
    if fault == "no_callback":
        case.model.lm.ignore_callback = True
    elif fault == "no_streamer":
        case.model.lm.ignore_streamer = True
    elif fault == "changed_prompt":
        case.model.lm.changed_output_prompt = True
    else:
        case.model.lm.extra_output = [1005]
    meta = failed(case)
    assert meta["failure_stage"] == "native_binding"
    assert meta["raw_token_ids"] is not None and meta["raw_output"] is not None
    assert any(row["event"] == "raw_original_decoder_output" for row in case.sink)


@pytest.mark.frozen_decoder
def test_fake_native_provider_cannot_ignore_grammar_to_return_thought_or_fences(factory):
    case = factory(text='thought {"text":"x"}')
    case.model.lm.ignore_callback = True
    meta = failed(case)
    assert meta["raw_output"].startswith("thought")
    assert meta["failure_stage"] == "native_binding"
    assert meta["response_authorized"] is False


@pytest.mark.frozen_decoder
def test_partial_native_failure_retains_actual_raw_ids_and_original_decode_then_restores(factory):
    case = factory()
    case.model.lm.fail_at = 3
    meta = failed(case)
    assert meta["raw_token_ids"] == case.model.lm.tokens[:3]
    assert meta["usage"]["output_tokens"] == 3
    assert meta["output_tokens_scope"] == "observed_emitted_before_failure"
    assert meta["raw_output"] == case.model.tok.backend_tokenizer.decode(
        case.model.lm.tokens[:3], skip_special_tokens=True
    )
    assert meta["adapter_restore_complete"]
    assert [row["event"] for row in case.sink][:2] == [
        "partial_raw_tokens",
        "partial_original_decoder_output",
    ]


@pytest.mark.frozen_decoder
def test_keyboard_interrupt_is_preserved_with_partial_generation_and_restore(factory):
    case = factory()
    interrupt = KeyboardInterrupt("owned cancellation")
    case.model.lm.fail_at, case.model.lm.fail_error = 2, interrupt
    with pytest.raises(KeyboardInterrupt) as error:
        run(case)
    assert error.value is interrupt
    assert interrupt.generation["raw_token_ids"] == case.model.lm.tokens[:2]
    assert interrupt.generation["adapter_restore_complete"]
    assert case.model.lm.adapter_enabled


@pytest.mark.parametrize(
    "stage",
    [
        "guard_entry",
        "template",
        "tokenization",
        "callback",
        "generation",
        "decode",
        "restore",
        "evidence",
    ],
)
@pytest.mark.frozen_decoder
def test_entire20seconds_includes_all_work_and_never_returns_late_text(factory, monkeypatch, stage):
    case = factory()
    if stage == "guard_entry":
        case.model.action_adapter.enter_delay = 21
    elif stage == "template":
        case.model.processor.delay = 21
    elif stage == "tokenization":
        case.model.tok.tokenization_delay = 21
    elif stage == "callback":
        original = H._allowed_ids

        def delayed(*args):
            result = original(*args)
            case.clock.value += 21
            return result

        monkeypatch.setattr(H, "_allowed_ids", delayed)
    elif stage == "generation":
        case.model.lm.post_generation_delay = 21
    elif stage == "decode":
        case.model.tok.decode_delay = 21
    elif stage == "restore":
        case.model.action_adapter.restore_delay = 21

    def sink(event):
        case.sink.append(event)
        if stage == "evidence" and event["event"] == "request_closed_before_response_gate":
            case.clock.value += 21

    meta = failed(case, evidence_sink=sink)
    assert meta["deadline_exceeded"] and meta["elapsed_seconds"] >= 21
    assert meta["adapter_restore_complete"]
    if stage in {"generation", "decode", "restore", "evidence"}:
        assert meta["raw_token_ids"] == case.model.lm.tokens and meta["raw_output"] is not None
    if stage in {"guard_entry", "template", "tokenization"}:
        assert not case.model.lm.calls


@pytest.mark.frozen_decoder
def test_outer_start_includes_prior_validation_queue_time_without_generation(factory):
    case = factory()
    meta = failed(case, started=case.clock.value - 20)
    assert meta["failure_stage"] == "guard_entry"
    assert not case.model.lm.calls and not meta["base_context_entered"]


@pytest.mark.parametrize(
    "event_name",
    ["raw_tokens_returned", "raw_original_decoder_output", "request_closed_before_response_gate"],
)
@pytest.mark.frozen_decoder
def test_synchronous_evidence_write_failure_never_authorizes_text(factory, event_name):
    case = factory()

    def sink(event):
        if event["event"] == event_name:
            raise OSError("durable evidence refused")
        case.sink.append(event)

    meta = failed(case, evidence_sink=sink)
    assert meta["evidence_failed"] and meta["adapter_restore_complete"]
    assert meta["raw_token_ids"] == case.model.lm.tokens
    assert meta["raw_output"] == case.model.tok.backend_tokenizer.decode(
        case.model.lm.tokens, skip_special_tokens=True
    )


@pytest.mark.frozen_decoder
def test_guard_restore_failure_is_integrity_failure_with_raw_evidence_retained(factory):
    case = factory()
    case.model.action_adapter.fail_restore = True
    meta = failed(case)
    assert meta["failure_stage"] == "guard_restore"
    assert not meta["adapter_restore_complete"]
    assert meta["raw_token_ids"] == case.model.lm.tokens and meta["raw_output"] is not None


@pytest.mark.frozen_decoder
def test_disabled_state_must_be_verified_before_native_generate(factory):
    case = factory()
    case.model.action_adapter.fail_disable = True
    meta = failed(case)
    assert not meta["disabled_check_passed"] and not case.model.lm.calls


@pytest.mark.parametrize("count,allowed", [(8064, True), (8065, False)])
@pytest.mark.frozen_decoder
def test_original8192_context_includes_full128_output_allowance(factory, count, allowed):
    case = factory()
    case.model.tok.prompt = [1005] * count
    if allowed:
        _, meta = run(case)
        assert meta["usage"]["input_tokens"] == count
    else:
        meta = failed(case)
        assert meta["usage"]["input_tokens"] == count and not case.model.lm.calls
        assert meta["error"]["message"] == "Text helper context exceeds the explicit token budget"


@pytest.mark.parametrize(
    "field",
    [
        "expected_cache_manifest_sha256",
        "expected_tokenizer_sha256",
        "expected_generation_sha256",
        "expected_grammar_source_sha256",
        "expected_helper_source_sha256",
        "expected_guard_source_sha256",
    ],
)
@pytest.mark.frozen_decoder
def test_prepare_requires_all_externally_selected_source_cache_metadata_pins(factory, field):
    case = factory(prepare=False)
    with pytest.raises(ValueError):
        H.prepare_helper(case.model, **{**case.selection, field: "0" * 64})
    assert not case.model.lm.calls


@pytest.mark.frozen_decoder
def test_prepare_reads_actual_decoder_version_instead_of_trusting_constant(factory, monkeypatch):
    import tokenizers

    case = factory(prepare=False)
    monkeypatch.setattr(tokenizers, "__version__", "unexpected-version")
    with pytest.raises(ValueError, match="Actual decoder version/binary"):
        H.prepare_helper(case.model, **case.selection)
    assert not case.model.lm.calls


@pytest.mark.frozen_decoder
def test_prepare_hashes_actual_native_binary_instead_of_trusting_constant(
    factory, tmp_path, monkeypatch
):
    case = factory(prepare=False)
    fake_binary = tmp_path / "substitute.so"
    fake_binary.write_bytes(b"not the native decoder")
    original = H.importlib.import_module

    def import_module(name):
        return (
            SimpleNamespace(__file__=str(fake_binary))
            if name == "tokenizers.tokenizers"
            else original(name)
        )

    monkeypatch.setattr(H.importlib, "import_module", import_module)
    with pytest.raises(ValueError, match="Actual decoder version/binary"):
        H.prepare_helper(case.model, **case.selection)
    assert not case.model.lm.calls


@pytest.mark.parametrize(
    "fault", ["missing_vocab", "float_id", "backend", "eos", "head", "context"]
)
@pytest.mark.frozen_decoder
def test_ready_rejects_native_inventory_decoder_and_generation_drift(factory, assets, fault):
    case = factory(prepare=False)
    if fault in {"missing_vocab", "float_id"}:
        vocab = case.model.tok.get_vocab()
        if fault == "missing_vocab":
            vocab.pop("text")
        else:
            vocab["text"] = float(vocab["text"])
        case.model.tok.get_vocab = lambda: vocab
    elif fault == "backend":
        case.model.tok.backend_tokenizer = SimpleNamespace(to_str=lambda: "{}")
    elif fault == "eos":
        case.model.lm.generation_config.eos_token_id = [106]
    elif fault == "head":
        case.model.lm.get_output_embeddings = lambda: SimpleNamespace(
            weight=SimpleNamespace(shape=(52, 42))
        )
    else:
        case.model.max_context = 9000
    with pytest.raises(ValueError):
        H.prepare_helper(case.model, **case.selection)
    assert not case.model.lm.calls


@pytest.mark.frozen_decoder
def test_model_identity_change_after_prepare_is_refused_before_generate(factory):
    case = factory()
    case.model.revision = "other-revision"
    meta = failed(case)
    assert not case.model.lm.calls and not meta["base_context_entered"]


@pytest.mark.parametrize(
    "field", ["pieces", "classes", "vocab_size", "mask_size", "byte_ids", "next_rows", "mask_rows"]
)
@pytest.mark.frozen_decoder
def test_prepared_cache_replacement_is_refused_before_any_native_forward(factory, field):
    case = factory()
    cache = case.prepared.cache
    original = getattr(cache, field)
    if type(original) is bytes:
        # Even byte-identical replacement is outside the authenticated immutable
        # object; changing it later must never silently become callback authority.
        replacement = bytes(bytearray(original))
    elif type(original) is tuple:
        replacement = tuple(list(original))
    else:
        replacement = original + 1
    assert replacement is not original
    setattr(cache, field, replacement)
    meta = failed(case)
    assert not case.model.lm.calls and not meta["base_context_entered"]
    assert meta["failure_stage"] == "guard_entry"


@pytest.mark.parametrize("method", ["step", "consume", "allowed_mask"])
@pytest.mark.frozen_decoder
def test_instance_cache_method_override_cannot_change_the_allowed_language(factory, method):
    case = factory()
    cache = case.prepared.cache
    if method == "allowed_mask":
        setattr(cache, method, lambda _: b"\xff" * cache.mask_size)
        assert cache.allowed_mask(G.State())[0] & 1  # Forbidden native special ID0.
    else:
        setattr(cache, method, lambda *args: G.State())
    meta = failed(case)
    assert not case.model.lm.calls and not meta["base_context_entered"]


@pytest.mark.frozen_decoder
def test_cache_subclass_with_changed_semantics_is_not_the_prepared_class(factory):
    case = factory()

    class ChangedGrammar(G.Grammar):
        def allowed_mask(self, state):
            return b"\xff" * self.mask_size

    case.prepared.cache.__class__ = ChangedGrammar
    meta = failed(case)
    assert not case.model.lm.calls and not meta["base_context_entered"]


@pytest.mark.frozen_decoder
def test_same_Rust_backend_decoder_mutation_is_refused_before_native_forward(factory):
    from tokenizers import decoders

    case = factory()
    backend = case.prepared.backend
    decoder = backend.decoder
    original = backend.decode(case.model.lm.tokens, skip_special_tokens=True)
    try:
        backend.decoder = decoders.ByteLevel()
        assert case.model.tok.backend_tokenizer is backend
        assert backend.decode(case.model.lm.tokens, skip_special_tokens=True) != original
        meta = failed(case)
        assert not case.model.lm.calls and not meta["base_context_entered"]
        assert "serialization changed" in meta["error"]["message"]
    finally:
        backend.decoder = decoder


@pytest.mark.parametrize("fault", ["cache", "decoder"])
@pytest.mark.frozen_decoder
def test_late_authority_mutation_refuses_text_and_retains_original_raw_evidence(factory, fault):
    from tokenizers import decoders

    case = factory()
    backend = case.prepared.backend
    decoder = backend.decoder

    def sink(event):
        case.sink.append(event)
        if event["event"] == "raw_original_decoder_output":
            if fault == "cache":
                cache = case.prepared.cache
                cache.mask_rows = b"\xff" * len(cache.mask_rows)
            else:
                backend.decoder = decoders.ByteLevel()

    try:
        meta = failed(case, evidence_sink=sink)
        assert len(case.model.lm.calls) == 1
        assert meta["raw_token_ids"] == case.model.lm.tokens
        assert meta["raw_output"] == '{"text":"field x"}'
        assert not meta["adapter_restore_complete"]
        assert any(row["event"] == "raw_original_decoder_output" for row in case.sink)
    finally:
        backend.decoder = decoder


@pytest.mark.parametrize("check_number", [1, 2])
@pytest.mark.frozen_decoder
def test_backend_serialization_validation_cost_is_inside_the20s_request(
    factory, monkeypatch, check_number
):
    case = factory()
    original = H._backend_serialization
    calls = 0

    def delayed(backend):
        nonlocal calls
        calls += 1
        result = original(backend)
        if calls == check_number:
            case.clock.value += 21
        return result

    monkeypatch.setattr(H, "_backend_serialization", delayed)
    meta = failed(case)
    assert meta["deadline_exceeded"] and meta["elapsed_seconds"] >= 21
    if check_number == 1:
        assert not case.model.lm.calls and not meta["base_context_entered"]
    else:
        assert len(case.model.lm.calls) == 1 and meta["raw_output"] is not None
        assert meta["failure_stage"] == "deadline_after_restore"


@pytest.mark.frozen_decoder
def test_sink_receives_detached_raw_evidence_and_no_real_framework_is_loaded(factory):
    case = factory()

    def sink(event):
        case.sink.append(copy.deepcopy(event))
        event["generation"]["raw_token_ids"] = ["external mutation"]

    _, meta = run(case, evidence_sink=sink)
    assert meta["raw_token_ids"] == case.model.lm.tokens
    assert not any(
        name.split(".")[0] in {"torch", "transformers", "peft", "modal"} for name in sys.modules
    )
