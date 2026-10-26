"""Full raw native browser request/profile contracts, no browser or model calls."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from s1.api import create_app
from s1.contracts import DecisionRequest, Question
from s1.errors import BackendResponseError, BackendTimeoutError, RequestValidationError
from s1.evaluation import decision3 as d3
from s1.jev_decision3_native import (
    PROFILE,
    Decision3NativeQuestion,
    Decision3NativeRequest,
    JevDecision3NativeAdapter,
    parse_native_payload,
)


def payload(count=53):
    return {
        "model": "native-test",
        "state": {"z": [3, {"b": 2, "a": 1}], "a": "same goal"},
        "questions": {
            "operation": {
                "type": "choice",
                "instructions": {"z": "goal", "a": ["rules"]},
                "criteria": {
                    "CLICK": "click",
                    "TYPE_TEXT": "type",
                    "SELECT": "select",
                    "DONE": "done",
                },
            },
            "click_target": {
                "type": "choice",
                "instructions": {"scope": "all observed", "history": ["x"]},
                "criteria": {
                    f"node-{i}": {"text": "duplicate", "rect": [i, 2], "role": "link"}
                    for i in range(count)
                },
            },
            "type_text_target": {
                "type": "choice",
                "instructions": "field",
                "criteria": {"field": {"name": "query"}},
            },
            "select_target": {
                "type": "choice",
                "instructions": "select",
                "criteria": {"sel:1": "one", "sel:2": "two"},
            },
        },
    }


class FakeNative:
    def __init__(self):
        self.spec = d3.decision3_spec("d3-edge")
        self.name = self.spec.id
        self.calls = []
        self.after = None

    def predict(self, request, *, deadline=None, evidence_sink=None):
        self.calls.append(request)
        if deadline:
            deadline()
        native = d3.native_payload(request)
        if evidence_sink:
            evidence_sink(
                "native_preflight",
                {
                    "native_payload": native,
                    "preflight": {"lengths": {q.id: 100 for q in request.questions}},
                },
            )
        answers = {}
        for q in request.questions:
            keys = q.labels()
            values = [1.0 / len(keys)] * len(keys)
            answers[q.id] = {
                "type": "choice",
                "choice": keys[0],
                "probabilities": dict(zip(keys, values)),
                "confidence": 0.0,
            }
        raw = {
            "model": "d3-edge",
            "answers": answers,
            "usage": {"input_tokens": 400, "output_tokens": 0},
            "cost": {"usd": None},
        }
        if self.after:
            self.after(raw)
        if evidence_sink:
            evidence_sink("native_raw_returned", {"raw_response": raw})
        return raw


@pytest.mark.parametrize("count", [1, 20, 52, 53, 255])
def test_explicit_dto_accepts_full_native_width_and_all_heads(count):
    raw = payload(count)
    parsed, original, source, labels = parse_native_payload(raw, model_identity="native-test")
    assert (
        source
        == json.dumps(raw, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    )
    assert original == raw
    assert parsed.questions[1].labels() == list(raw["questions"]["click_target"]["criteria"])
    assert len(labels["click_target"]) == count
    backend = FakeNative()
    result = JevDecision3NativeAdapter(backend, model_identity="native-test").predict(raw)
    assert len(backend.calls) == 1
    assert [q.id for q in backend.calls[0].questions] == list(raw["questions"])
    assert len(result["answers"]) == 4
    assert result["answers"][2]["probabilities"] == {"field": 1.0}
    assert result["decision3_native"]["native_question_rows"] == 4
    assert result["decision3_native"]["original_request"] == raw
    assert result["cost"] == {"usd": None}
    assert result["usage"]["input_tokens"] == 400


@pytest.mark.parametrize("count", [256, 400, 401])
def test_wide_unsupported_heads_refused_before_backend_without_pruning(count):
    backend = FakeNative()
    original = payload(count)
    with pytest.raises(RequestValidationError) as caught:
        JevDecision3NativeAdapter(backend, model_identity="native-test").predict(original)
    assert backend.calls == []
    assert caught.value.decision3_native["original_request"] == original
    assert len(original["questions"]["click_target"]["criteria"]) == count


def test_native_json_order_and_mutation_protection_without_string_prompt_rewrite():
    raw = payload()
    backend = FakeNative()
    result = JevDecision3NativeAdapter(backend, model_identity="native-test").predict(raw)
    sent = d3.native_payload(backend.calls[0])
    assert isinstance(sent["questions"]["operation"]["instructions"], dict)
    assert list(sent["questions"]["operation"]["instructions"]) == ["z", "a"]
    assert list(sent["state"]) == ["z", "a"]
    assert list(sent["questions"]["click_target"]["criteria"]["node-0"]) == ["text", "rect", "role"]
    saved = copy.deepcopy(result["decision3_native"]["original_request"])
    raw["questions"]["click_target"]["criteria"]["node-0"]["text"] = "mutated"
    backend.calls[0].questions[1].criteria["node-0"]["rect"].append(9)
    assert result["decision3_native"]["original_request"] == saved


def test_generic52_contract_stays_strict_and_native_dto_validates_normally():
    with pytest.raises(ValidationError):
        Question(id="q", instructions="choose", criteria={str(i): "x" for i in range(53)})
    with pytest.raises(ValidationError):
        Decision3NativeQuestion(
            id="q", instructions="choose", criteria={str(i): "x" for i in range(256)}
        )
    with pytest.raises(ValidationError):
        Decision3NativeQuestion(id="q", instructions="choose", criteria={"x": "y"}, hidden_gold="x")
    assert DecisionRequest.model_fields["questions"].annotation == list[Question]


@pytest.mark.parametrize("target_count", [38, 255])
def test_diagnostic_native_questions_keep_their_payload_outside_browser_action_schema(target_count):
    counts = {"operation": 7, "target": target_count, "text": 36, "singleton": 1}
    original = {
        "state": {"description": "Invented diagnostic; no browser action or task answer."},
        "questions": {
            key: {
                "type": "choice",
                "instructions": "Choose one fabricated label.",
                "criteria": {f"fabricated_{index:03d}": None for index in range(count)},
            }
            for key, count in counts.items()
        },
    }
    backend = FakeNative()
    with pytest.raises(RequestValidationError, match="operation and named target heads"):
        JevDecision3NativeAdapter(backend, model_identity="native-test").predict(
            {"model": "native-test", **copy.deepcopy(original)}
        )
    assert backend.calls == []

    request = Decision3NativeRequest.model_validate(
        {
            "state": copy.deepcopy(original["state"]),
            "questions": [
                {"id": key, **copy.deepcopy(question)}
                for key, question in original["questions"].items()
            ],
        }
    )
    # General native diagnostics preserve their original names and full widths.
    # Renaming them as browser actions would change the serialized model input.
    encoded = json.dumps(original, ensure_ascii=False, separators=(",", ":")).encode()
    assert (
        json.dumps(d3.native_payload(request), ensure_ascii=False, separators=(",", ":")).encode()
        == encoded
    )
    assert [question.id for question in request.questions] == list(counts)
    assert [len(question.labels()) for question in request.questions] == list(counts.values())


@pytest.mark.parametrize(
    "field",
    [
        "private_gold",
        "bad_alias",
        "unoffered_op",
        "nonfinite",
        "wrong_model",
        "missing_target",
        "unknown_head",
        "bool_description",
    ],
)
def test_invalid_full_raw_wire_refused_before_backend(field):
    raw = payload()
    if field == "private_gold":
        raw["questions"]["operation"]["gold"] = "DONE"
    elif field == "bad_alias":
        raw["questions"]["click_target"]["options"] = {"x": "extra"}
    elif field == "unoffered_op":
        raw["questions"]["operation"]["criteria"]["NAVIGATE"] = "go"
    elif field == "nonfinite":
        raw["state"]["bad"] = float("nan")
    elif field == "wrong_model":
        raw["model"] = "other"
    elif field == "missing_target":
        raw["questions"].pop("select_target")
    elif field == "unknown_head":
        raw["questions"]["oracle"] = copy.deepcopy(raw["questions"]["operation"])
    else:
        raw["questions"]["click_target"]["criteria"]["node-0"] = True
    backend = FakeNative()
    with pytest.raises(RequestValidationError):
        JevDecision3NativeAdapter(backend, model_identity="native-test").predict(raw)
    assert backend.calls == []


def test_targetless_inventory_still_evaluates_complete_offered_operation():
    raw = {
        "model": "native-test",
        "state": "goal",
        "questions": {
            "operation": {
                "type": "choice",
                "instructions": "next",
                "criteria": {
                    "SCROLL_UP": "up",
                    "SCROLL_DOWN": "down",
                    "WAIT": "wait",
                    "DONE": "done",
                },
            }
        },
    }
    backend = FakeNative()
    result = JevDecision3NativeAdapter(backend, model_identity="native-test").predict(raw)
    assert len(backend.calls) == 1 and len(result["answers"]) == 1
    assert result["answers"][0]["probabilities"] == {
        "SCROLL_UP": 0.25,
        "SCROLL_DOWN": 0.25,
        "WAIT": 0.25,
        "DONE": 0.25,
    }


@pytest.mark.parametrize(
    "damage",
    ["missing_unused", "bad_choice", "bad_confidence", "wrong_model", "bad_mass", "native_error"],
)
def test_all_head_outputs_are_strictly_validated_with_raw_failure_evidence(damage):
    backend = FakeNative()

    def after(raw):
        if damage == "missing_unused":
            raw["answers"].pop("select_target")
        elif damage == "bad_choice":
            raw["answers"]["operation"]["choice"] = "invented"
        elif damage == "bad_confidence":
            raw["answers"]["operation"]["confidence"] = True
        elif damage == "wrong_model":
            raw["model"] = "other"
        elif damage == "bad_mass":
            raw["answers"]["type_text_target"]["probabilities"]["field"] = 0.5
        else:
            raw["answers"]["select_target"] = {"error": "invalid_question"}

    backend.after = after
    with pytest.raises(BackendResponseError) as caught:
        JevDecision3NativeAdapter(backend, model_identity="native-test").predict(payload())
    assert caught.value.decision3_native["raw_response"] is not None
    assert caught.value.decision3_native["backend_predict_calls"] == 1


def test_one_shared20s_timer_and_late_raw_return_is_not_actionable(monkeypatch):
    import s1.jev_decision3_native as module

    now = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    backend = FakeNative()
    backend.after = lambda raw: now.__setitem__(0, 20.0)
    events = []
    with pytest.raises(BackendTimeoutError) as caught:
        JevDecision3NativeAdapter(
            backend, model_identity="native-test", evidence_sink=events.append
        ).predict(payload(1))
    assert caught.value.decision3_native["raw_response"]["answers"]
    assert caught.value.decision3_native["status"] == "failed"
    assert events[-1]["event"] == "failed"


def test_parse_time_counts_and_observer_failure_is_visible(monkeypatch):
    import s1.jev_decision3_native as module

    now = [0.0]
    parse = module.parse_native_payload
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])

    def spent(*args, **kwargs):
        result = parse(*args, **kwargs)
        now[0] = 20.0
        return result

    monkeypatch.setattr(module, "parse_native_payload", spent)
    backend = FakeNative()
    with pytest.raises(BackendTimeoutError):
        JevDecision3NativeAdapter(backend, model_identity="native-test").predict(payload())
    assert backend.calls == []
    monkeypatch.setattr(module, "parse_native_payload", parse)
    now[0] = 0

    def broken(_event):
        raise OSError("journal failed")

    result = JevDecision3NativeAdapter(
        backend, model_identity="native-test", evidence_sink=broken
    ).predict(payload())
    assert result["decision3_native"]["evidence_sink_errors"][0]["message"] == "journal failed"


def test_completed_checkpoint_time_remains_inside_shared_budget(monkeypatch):
    import s1.jev_decision3_native as module

    now = [0.0]
    events = []
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])

    def sink(event):
        events.append(event)
        if event["event"] == "completed":
            now[0] = 20.0

    with pytest.raises(BackendTimeoutError) as caught:
        JevDecision3NativeAdapter(
            FakeNative(), model_identity="native-test", evidence_sink=sink
        ).predict(payload())
    assert caught.value.decision3_native["status"] == "failed"
    assert caught.value.decision3_native["raw_response"]["answers"]
    assert [event["event"] for event in events][-2:] == ["completed", "failed"]


def test_api_explicit_native_profile_journals_full_wire_and_errors():
    backend = FakeNative()
    events = []
    app = create_app(
        backend,
        jev_compatible=True,
        jev_decision3_native=True,
        model_identity="native-test",
        request_evidence_sink=events.append,
    )
    client = TestClient(app)
    result = client.post("/v1/systemone", json=payload(255))
    assert result.status_code == 200
    assert result.json()["decision3_native"]["version"] == PROFILE
    assert events[0]["data"]["payload"] == payload(255)
    assert any(event["event"] == "decision3_native_checkpoint" for event in events)
    refused = client.post("/v1/systemone", json=payload(256))
    assert refused.status_code == 422
    assert refused.json()["detail"]["decision3_native"]["status"] == "failed"
    assert len(backend.calls) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"jev_compatible": False},
        {"request_evidence_sink": None},
        {"jev_hierarchical": True},
        {"jev_hierarchical_batch": True},
        {"jev_browser_operation_first": True},
        {"jev_generated_actions": True},
        {"jev_generated_action_grammar": True},
    ],
)
def test_api_incompatible_profiles_fail_before_any_prediction(changes):
    options = {
        "jev_compatible": True,
        "jev_decision3_native": True,
        "request_evidence_sink": lambda e: None,
    }
    options.update(changes)
    with pytest.raises(ValueError):
        create_app(FakeNative(), **options)


def cli_driver():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    try:
        spec = importlib.util.spec_from_file_location(
            "_decision3_test_cli", root / "scripts/jev_local_model.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--competitor", "decider"],
        ["--competitor", "d3"],
        ["--competitor", "d3", "--quantization", "none"],
        [
            "--competitor",
            "d3",
            "--quantization",
            "none",
            "--server-evidence",
            "--hierarchical-candidates",
        ],
        ["--competitor", "d3", "--quantization", "none", "--server-evidence", "--model", "other"],
    ],
)
def test_cli_invalid_profile_flags_precede_any_source_or_model_access(tmp_path, monkeypatch, extra):
    driver = cli_driver()
    monkeypatch.setattr(
        driver, "upstream_manifest", lambda *_: pytest.fail("invalid flags reached source/model")
    )
    with pytest.raises(SystemExit) as caught:
        driver.main(
            [
                "--upstream-path",
                "unused",
                "--output-dir",
                str(tmp_path / "run"),
                "--decision3-native",
                *extra,
            ]
        )
    assert caught.value.code == 2 and not (tmp_path / "run").exists()


def test_cli_requires_profile_for_each_of_six_competitors_before_source(tmp_path, monkeypatch):
    driver = cli_driver()
    monkeypatch.setattr(
        driver, "upstream_manifest", lambda *_: pytest.fail("missing flag reached source/model")
    )
    for name in d3.CHECKPOINTS:
        with pytest.raises(SystemExit):
            driver.main(
                [
                    "--upstream-path",
                    "unused",
                    "--output-dir",
                    str(tmp_path / name),
                    "--competitor",
                    name,
                ]
            )


def test_cli_genuine_memory_refusal_precedes_torch_and_helper_load(tmp_path, monkeypatch):
    from s1 import browser_evidence

    driver = cli_driver()
    monkeypatch.setattr(driver, "upstream_manifest", lambda *_: {"revision": "cpu-test"})
    monkeypatch.setattr(browser_evidence, "capture_runtime_sources", lambda *_a, **_k: {})
    monkeypatch.setattr(
        browser_evidence, "verify_runtime_sources", lambda *_a, **_k: {"success": True}
    )
    original = Path.read_text

    def read(path, *args, **kwargs):
        if str(path) == "/proc/meminfo":
            return "MemTotal: 134217728 kB\nMemAvailable: 34603008 kB\nSwapTotal: 999999999 kB\nSwapFree: 999999999 kB\n"
        if str(path) == "/proc/sys/kernel/random/boot_id":
            return "cpu-test-boot"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    before_torch = sys.modules.get("torch")
    with pytest.raises(SystemExit) as caught:
        driver.main(
            [
                "--upstream-path",
                "unused",
                "--output-dir",
                str(tmp_path / "run"),
                "--decision3-native",
                "--competitor",
                "d3-edge",
                "--quantization",
                "none",
                "--server-evidence",
            ]
        )
    assert caught.value.code == 1
    assert sys.modules.get("torch") is before_torch
    receipt = json.loads((tmp_path / "run/receipt.json").read_bytes())
    assert receipt["browser_decision_profile"]["profile"] == PROFILE
    assert receipt["decision3_memory_refusal"]["admitted"] is False
    assert (
        receipt["decision3_memory_refusal"]["largest_declared_transient_bytes"]
        == d3.FIXED_HELPER_BYTES
    )
    assert receipt["decision3_memory_refusal"]["swap_credit_bytes"] == 0
    assert "versions" not in receipt and "model_reported" not in receipt


def test_cli_final_receipt_failure_preserves_primary_memory_refusal(tmp_path, monkeypatch):
    from s1 import browser_evidence

    driver = cli_driver()
    monkeypatch.setattr(driver, "upstream_manifest", lambda *_: {"revision": "cpu-test"})
    monkeypatch.setattr(browser_evidence, "capture_runtime_sources", lambda *_a, **_k: {})
    monkeypatch.setattr(
        browser_evidence, "verify_runtime_sources", lambda *_a, **_k: {"success": True}
    )
    primary = ValueError("primary pre-helper refusal")
    primary.decision3_memory = {"admitted": False, "boot_id": "test"}

    def refuse(*_args, **_kwargs):
        raise primary

    monkeypatch.setattr(d3, "require_memory", refuse)
    writer = driver._write_receipt

    def fail_final(path, receipt, *, qualified=False):
        if qualified:
            raise OSError("final fsync failed")
        return writer(path, receipt, qualified=False)

    monkeypatch.setattr(driver, "_write_receipt", fail_final)
    with pytest.raises(ValueError, match="primary pre-helper refusal") as caught:
        driver.main(
            [
                "--upstream-path",
                "unused",
                "--output-dir",
                str(tmp_path / "run"),
                "--decision3-native",
                "--competitor",
                "d3-edge",
                "--quantization",
                "none",
                "--server-evidence",
            ]
        )
    assert caught.value is primary
    assert any("final fsync failed" in note for note in primary.__notes__)
    # A failed final write leaves the existing receipt readable.
    assert (
        json.loads((tmp_path / "run/receipt.json").read_text())["browser_decision_profile"][
            "profile"
        ]
        == PROFILE
    )
