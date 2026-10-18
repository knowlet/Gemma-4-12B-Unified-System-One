"""Pure wiring/identity/consumer tests; no semantic model success is asserted."""

import ast
import importlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    H = importlib.import_module("jev_field_syntax_helper")
    Q = importlib.import_module("jev_qualify_public_field_syntax_bf16")
finally:
    sys.path.remove(str(ROOT / "scripts"))


def function(path, name):
    return next(
        n
        for n in ast.parse(path.read_bytes()).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )


def test_distinct_identity_shared_source_authority_and_fixed_budgets():
    assert Q.ROOT == ROOT
    assert Q.FIELD_PROFILE == H.PROFILE == "field-syntax-preserving-base-helper-v1"
    assert Q.KIND == "jev_public_action_field_syntax_bf16_qualification_v1"
    assert Q.HELPER_SHA == Q.pin(ROOT / Q.HELPER_SELF)["sha256"]
    assert Q._helper_instructions() == H.selected_instructions() == H.INSTRUCTIONS
    assert Q.digest(H.INSTRUCTIONS.encode()) == Q.HELPER_INSTRUCTIONS_SHA
    assert (H.SECONDS, H.TOKENS, H.CONTEXT) == (20.0, 128, 8192)
    assert H.grammar.EOS_IDS == (1, 106, 50)
    extension = Q.field_json_design_extension()
    assert extension["execution_source_count"] == 95
    assert extension["training_source_count"] == 90
    assert len(Q.FIELD_EXTENSIONS) == 5 and len(Q.SEMANTIC_EXTENSIONS) == 2
    assert extension["helper_contract"]["instructions_sha256"] == Q.HELPER_INSTRUCTIONS_SHA
    assert len(Q.expected_case_ids()) == 41


def test_mutated_prompt_global_refuses_both_prefix_authority_and_selected_helper(monkeypatch):
    monkeypatch.setattr(H, "INSTRUCTIONS", H.INSTRUCTIONS + " altered")
    with pytest.raises(ValueError, match="global drift"):
        H.selected_instructions()
    with pytest.raises(ValueError, match="global drift"):
        Q._helper_instructions()


def test_foreign_selected_module_origin_or_digest_is_refused(monkeypatch):
    monkeypatch.setattr(H, "__file__", str(ROOT / "scripts/jev_field_json_helper.py"))
    with pytest.raises(ValueError, match="Foreign field helper"):
        Q._helper_instructions()


def test_wrong_selected_instruction_digest_is_refused(monkeypatch):
    monkeypatch.setattr(Q, "HELPER_INSTRUCTIONS_SHA", "0" * 64)
    with pytest.raises(ValueError, match="instruction digest"):
        Q._helper_instructions()


class Ids:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        return Ids(self.values[index])

    def tolist(self):
        return self.values


def test_actual_generation_template_and_native_prefix_use_identical_reviewed_bytes(monkeypatch):
    context = {
        "goal": 'Search for the exact phrase "moss lantern".',
        "field": {"meaning": "query"},
        "history": [],
    }
    prefix_calls, generation_calls = [], []

    def processor(calls):
        def template(messages, **kwargs):
            calls.append((messages, kwargs))
            return "FAKE_TEMPLATE\n" + messages[0]["content"]

        return SimpleNamespace(apply_chat_template=template)

    model = SimpleNamespace(
        processor=processor(prefix_calls),
        tok=lambda *_a, **_k: {"input_ids": Ids([[2, 7]]), "attention_mask": Ids([[1, 1]])},
    )
    monkeypatch.setattr(Q, "parse_helper_request", lambda *_a, **_k: context)
    prefix, prompt = Q._prefix(model, {}, "helper", include_prompt=True)
    assert prefix["prompt_sha256"] == Q.digest(prompt.encode())

    @contextmanager
    def base_context():
        yield

    def stop_tokenization(*_a, **_k):
        raise RuntimeError("intentional fake stop before native tokenization or generation")

    model = SimpleNamespace(
        processor=processor(generation_calls),
        tok=stop_tokenization,
        name="fake",
        revision="fake",
        quantization="none",
    )
    guard = SimpleNamespace(base_context=base_context, _check=lambda **_k: None, check=lambda: None)
    prepared = object.__new__(H.PreparedHelper)
    for name, value in {
        "model": model,
        "guard": guard,
        "_token": H._VERIFIED,
        "_receipt": json.dumps(
            {"cache_manifest_sha256": "f" * 64, "instructions_sha256": Q.HELPER_INSTRUCTIONS_SHA}
        ).encode(),
    }.items():
        object.__setattr__(prepared, name, value)
    monkeypatch.setattr(H.PreparedHelper, "check", lambda _self: H.selected_instructions())
    events = []
    with pytest.raises(H.FieldJSONError, match="intentional fake stop") as caught:
        H.generate_field_text(prepared, context, evidence_sink=events.append, clock=lambda: 10.0)
    assert prefix_calls == generation_calls
    assert generation_calls[0][0][0]["content"] == H.INSTRUCTIONS + "\n" + json.dumps(
        context, ensure_ascii=False
    )
    assert generation_calls[0][1] == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    evidence = caught.value.generation
    assert evidence["instructions_sha256"] == Q.HELPER_INSTRUCTIONS_SHA
    assert evidence["native_generation_returned"] is False
    assert evidence["success"] is False and evidence["adapter_restore_complete"] is True
    assert events[-1]["event"] == "request_closed_before_response_gate"


@pytest.mark.parametrize(
    "value",
    [
        '"moss lantern"',
        "moss lantern",
        'title:"moss lantern" AND -draft',
        "Leda Noor",
        "leda.noor+trial@example.test",
        'print("moss")',
        'label says "north"',
        r"C:\notes\moss.txt",
        r"prefix\"moss\"",
        " (moss OR fern) -draft ",
        "x:y; a/b?c=d&e=f",
        "\t keep space \n",
    ],
)
def test_consumer_preserves_explicit_characters_without_repair(value):
    raw = json.dumps({"text": value}, ensure_ascii=False)
    assert H._consume_value(raw) == value


def test_valid_json_with_wrong_query_semantics_stays_wrong_without_repair():
    value = H._consume_value('{"text":"moss lantern"}')
    assert value == "moss lantern" and value != '"moss lantern"'


@pytest.mark.parametrize(
    "raw",
    [
        '{"text":null}',
        '{"text":42}',
        '{"text":""}',
        '{"text":"  "}',
        '```json\n{"text":"moss"}\n```',
        'thought\n{"text":"moss"}',
        '{"text":"moss","extra":true}',
        '{"text":"moss","text":"fern"}',
    ],
)
def test_consumer_never_repairs_invalid_or_missing_value(raw):
    with pytest.raises((ValueError, json.JSONDecodeError)):
        H._consume_value(raw)


def test_new_qualification_rejects_old_execution_kind_and_missing_instruction_evidence():
    baseline = Q.canonical(
        {
            "schema_id": "jev_public_action_field_json_qualification_execution_v1",
            "training_handoff": {},
            "files": {},
            "identity_sha256": "0" * 64,
            "baseline_field_json_execution_utf8": "",
        }
    )
    with pytest.raises(ValueError, match="Execution kind"):
        Q.verify_execution_sources(
            baseline,
            expected_sha256=Q.digest(baseline),
            expected_training_sha256=Q.TRAINING90_SHA,
            training_blob=b"not consumed",
        )
    with pytest.raises(ValueError, match="instruction identity"):
        Q._validate_field_generation({}, {}, 0, None)


@pytest.mark.parametrize("fault", [None, "projection", "baselinebytes", "extra", "oldkind"])
def test_exact_preserved93_projection_and_external_execution_identity(fault):
    base = ROOT / "artifacts/jev-runtime/public-field-syntax-helper-20261018-01"
    execution_path = base / "execution-95.json"
    training_path = ROOT / (
        "artifacts/jev-runtime/public-native-action-epoch-ablation-20261018-01/"
        "actual-fit-240-02-parallel-cli/public07-h200-240-20261018-01-fit/fit/source-handoff.json"
    )
    if not execution_path.is_file() or not training_path.is_file():
        pytest.skip("Requires locally saved execution95 and training90 source handoffs")
    value = Q.strict_json(Q.read(execution_path))
    training = Q.read(training_path)
    if fault == "projection":
        value["files"]["scripts/jev_field_json_helper.py"]["sha256"] = "0" * 64
        value["identity_sha256"] = Q.digest(Q.canonical(value["files"]))
    elif fault == "baselinebytes":
        value["baseline_field_json_execution_utf8"] += " "
    elif fault == "extra":
        value["files"]["scripts/unselected.py"] = {"sha256": "0" * 64, "size_bytes": 0}
    elif fault == "oldkind":
        value["schema_id"] = "jev_public_action_field_json_qualification_execution_v1"
    blob = Q.canonical(value)
    kwargs = dict(
        expected_sha256=Q.digest(blob),
        expected_training_sha256=Q.TRAINING90_SHA,
        training_blob=training,
    )
    if fault is not None:
        with pytest.raises(ValueError):
            Q.verify_execution_sources(blob, **kwargs)
        return
    # Saved-source authority only: no corpus/processor/model/runtime qualification.
    sources = Q.verify_execution_sources(blob, **kwargs)
    assert sources.manifest() == value
    assert len(value["files"]) == 95
    assert Q.digest(value["baseline_field_json_execution_utf8"].encode()) == Q.BASELINE93_SHA


def test_old_input_and_design_schemas_cannot_be_relabelled():
    value = {
        "schema_id": "jev_public_action_field_json_qualification_inputs_v1",
        "plan_sha256": "a" * 64,
        "execution_sha256": "b" * 64,
        "runtime_sha256": "c" * 64,
        "public": {},
        "cases": [],
        "materialization": [],
    }
    with pytest.raises(ValueError, match="input identities"):
        Q.validate_input_manifest(
            value,
            expected_plan_sha256="a" * 64,
            expected_execution_sha256="b" * 64,
            expected_runtime_sha256="c" * 64,
        )
    old_wrapper = Q.canonical(
        {
            "schema_id": "jev-public-field-json-bf16-qualification-design-v1",
            "base_plan": {},
            "historical_delta_utf8": "",
            "extension": {},
        }
    )
    with pytest.raises(ValueError, match="design wrapper"):
        Q.apply_design_delta(b"", old_wrapper)


def test_policy_prompt_branch_and_all_technical_recipe_functions_are_bytecode_AST_unchanged():
    old = ROOT / "scripts/jev_qualify_public_field_json_bf16.py"
    new = ROOT / Q.SELF
    old_prefix, new_prefix = function(old, "_prefix"), function(new, "_prefix")
    assert ast.dump(old_prefix, include_attributes=False) == ast.dump(
        new_prefix, include_attributes=False
    )
    for name in (
        "expected_case_ids",
        "fixture_payload",
        "validate_recipes",
        "parse_helper_request",
    ):
        assert ast.dump(function(old, name), include_attributes=False) == ast.dump(
            function(new, name), include_attributes=False
        )

    def gate(path):
        return next(
            n
            for n in ast.parse(path.read_bytes()).body
            if isinstance(n, ast.ClassDef) and n.name == "QualificationGate"
        )

    assert ast.dump(gate(old), include_attributes=False) == ast.dump(
        gate(new), include_attributes=False
    )
