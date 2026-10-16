"""CPU scheduling/probability contracts; no real model or browser performance claim."""

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from s1.contracts import answer_from_probabilities
from s1.errors import BackendResponseError, BackendTimeoutError, RequestValidationError
from s1.jev_browser_operation_first import PROFILE, JevBrowserOperationFirstAdapter
from s1.jev_hierarchy import JevHierarchicalAdapter


def payload(count=21, unused=23):
    questions = {
        "operation": {
            "type": "choice",
            "instructions": {
                "goal": "Use the current observed control.",
                "rules": ["Keep all current candidates", "中文"],
            },
            "criteria": {
                operation: operation
                for operation in (
                    "CLICK",
                    "TYPE_TEXT",
                    "SELECT",
                    "SCROLL_DOWN",
                    "SCROLL_UP",
                    "WAIT",
                    "DONE",
                    "BLOCKED",
                )
            },
        }
    }
    for head, operation, count_ in (
        ("click_target", "CLICK", count),
        ("type_text_target", "TYPE_TEXT", 1),
        ("select_target", "SELECT", unused),
    ):
        questions[head] = {
            "type": "choice",
            "instructions": {
                "goal": "Use the current observed control.",
                "operation": operation,
                "rules": ["Original native rule"],
            },
            "criteria": {
                f"{head}-{i}": {"label": f"Observed {i}", "value": "unchanged"}
                for i in range(count_)
            },
        }
    return {
        "model": "native-specialist",
        "state": {
            "page": {
                "url": "https://example.invalid/controls",
                "title": "Controls",
                "text": "Current observed page",
            },
            "elements": [{"index": "1", "checked": False}],
            "recent_actions": [{"action": "previous", "kind": "click"}],
        },
        "questions": questions,
    }


def stage_info(question):
    return json.loads(question.instructions).get("hierarchy", {})


class NativeBackend:
    name = "native-specialist"

    def __init__(self, operation="CLICK", *, transform=None):
        self.operation, self.transform = operation, transform
        self.calls, self.preflights = [], []
        self.spec = SimpleNamespace(
            adapter="jev_omni",
            calibration="checkpoint",
            precision="bfloat16",
            capabilities=SimpleNamespace(independent_questions=True, max_options=20),
        )

    def preflight(self, request):
        self.preflights.append(request.model_copy(deep=True))
        return {"input_tokens": len(json.dumps(request.model_dump()))}

    def predict(self, request):
        self.calls.append(request.model_copy(deep=True))
        answers = {}
        for question in request.questions:
            labels = question.labels()
            info = stage_info(question)
            if question.id == "operation":
                values = [float(label == self.operation) for label in labels]
            elif info.get("stage") == "group" and len(labels) == 2:
                values = [0.51, 0.49]
            elif info.get("stage") == "conditional" and info["group"] == "group1":
                values = [0.9] + [0.1 / (len(labels) - 1)] * (len(labels) - 1)
            else:
                values = [1 / len(labels)] * len(labels)
            answers[question.id] = answer_from_probabilities(question, values)
        result = {
            "model": self.name,
            "answers": answers,
            "usage": {"input_tokens": 11, "output_tokens": 0},
            "native_metadata": {"calibration": "checkpoint"},
        }
        return self.transform(request, result) if self.transform else result


def adapter(backend, **kwargs):
    return JevBrowserOperationFirstAdapter(backend, model_identity="native-specialist", **kwargs)


@pytest.mark.parametrize("width", [1, 2, 20, 21, 53, 400])
def test_selected_complete_distribution_matches_independent_fake_full_head_baseline(width):
    wire = payload(width)
    before = copy.deepcopy(wire)
    full_backend, selected_backend = NativeBackend(), NativeBackend()
    baseline = JevHierarchicalAdapter(full_backend, model_identity="native-specialist").predict(
        wire
    )
    selected = adapter(selected_backend).predict(wire)
    assert selected["answers"] == {
        key: baseline["answers"][key] for key in ("operation", "click_target")
    }
    assert wire == before
    assert selected["jev_browser_profile"] == PROFILE
    assert selected["unused_target_heads"] == ["type_text_target", "select_target"]
    assert selected["unused_target_heads_status"] == "not_evaluated"
    assert list(selected["answers"]["click_target"]["probabilities"]) == list(
        wire["questions"]["click_target"]["criteria"]
    )
    assert all(
        question.id != "select_target"
        for request in selected_backend.calls
        for question in request.questions
    )
    assert selected["hierarchy"]["original_request"] == wire
    assert selected["hierarchy"]["native_question_rows"] == len(selected_backend.calls)
    # Kept native questions/state, including every hierarchical wrapper and
    # ordered criterion, equal their full-head counterparts. This fake backend
    # equality is not native-model numerical equivalence.
    full_questions = [
        (
            json.dumps(request.state, ensure_ascii=False, separators=(",", ":")),
            question.model_dump_json(),
        )
        for request in full_backend.calls
        for question in request.questions
    ]
    for request in selected_backend.calls:
        for question in request.questions:
            assert (
                json.dumps(request.state, ensure_ascii=False, separators=(",", ":")),
                question.model_dump_json(),
            ) in full_questions


def test_all_groups_are_evaluated_even_when_winning_leaf_is_in_nonwinning_group():
    backend = NativeBackend()
    result = adapter(backend).predict(payload(21))
    hierarchy = result["hierarchy"]
    head = hierarchy["heads"]["click_target"]
    assert max(head["group_probabilities"], key=head["group_probabilities"].get) == "group0"
    assert result["answers"]["click_target"]["choice"] == "click_target-11"
    assert [stage["group"] for stage in hierarchy["stages"] if stage["scope"] == "conditional"] == [
        "group0",
        "group1",
    ]
    assert sum(result["answers"]["click_target"]["probabilities"].values()) == pytest.approx(1)


@pytest.mark.parametrize("operation", ["SCROLL_DOWN", "SCROLL_UP", "WAIT", "DONE", "BLOCKED"])
def test_targetless_native_operations_only_evaluate_operation(operation):
    backend = NativeBackend(operation)
    result = adapter(backend).predict(payload(400, unused=400))
    assert set(result["answers"]) == {"operation"}
    assert len(backend.calls) == len(backend.preflights) == 1
    assert result["unused_target_heads"] == ["click_target", "type_text_target", "select_target"]
    assert result["hierarchy"]["original_request_wide"] is True


def test_selected_singleton_comes_from_real_inventory_without_model_call():
    backend = NativeBackend("TYPE_TEXT")
    result = adapter(backend).predict(payload(400, unused=400))
    assert len(backend.calls) == 1
    assert result["answers"]["type_text_target"]["probabilities"] == {"type_text_target-0": 1.0}
    assert result["jev_compatibility"]["singleton_heads"] == ["type_text_target"]
    assert result["hierarchy"]["stages"][-1]["status"] == "deterministic"


def test_select_operation_keeps_every_offered_select_candidate():
    backend = NativeBackend("SELECT")
    result = adapter(backend).predict(payload(53, unused=21))
    assert set(result["answers"]) == {"operation", "select_target"}
    assert len(result["answers"]["select_target"]["probabilities"]) == 21
    assert all(
        question.id != "click_target" for request in backend.calls for question in request.questions
    )


@pytest.mark.parametrize("head", ["click_target", "select_target"])
def test_entire_original_schema_including_unused_401_head_fails_before_any_call(head):
    wire = payload(2, unused=2)
    wire["questions"][head]["criteria"] = {str(i): "observed" for i in range(401)}
    backend = NativeBackend("TYPE_TEXT")
    with pytest.raises(RequestValidationError):
        adapter(backend).predict(wire)
    assert not backend.calls and not backend.preflights


@pytest.mark.parametrize(
    "mutate",
    [
        lambda wire: wire["questions"].update(
            other={
                "type": "choice",
                "instructions": "not browser",
                "criteria": {"a": "a", "b": "b"},
            }
        ),
        lambda wire: wire.update(private_oracle=True),
        lambda wire: wire["questions"]["operation"]["criteria"].update(ENTER="unsupported"),
    ],
)
def test_generic_or_unknown_browser_schemas_are_rejected(mutate):
    wire, backend = payload(), NativeBackend()
    mutate(wire)
    with pytest.raises(RequestValidationError) as caught:
        adapter(backend).predict(wire)
    assert backend.calls == []
    assert caught.value.hierarchy["original_request"] == wire


def test_invalid_operation_response_never_evaluates_any_target():
    def invalid(_request, result):
        result["answers"]["operation"]["choice"] = "ENTER"
        return result

    backend = NativeBackend(transform=invalid)
    with pytest.raises(BackendResponseError) as caught:
        adapter(backend).predict(payload(53))
    assert len(backend.calls) == 1
    assert (
        caught.value.hierarchy["stages"][0]["raw_response"]["answers"]["operation"]["choice"]
        == "ENTER"
    )


def test_unused_native_context_failures_are_not_preflighted_but_active_failures_remain_fatal():
    class ContextBackend(NativeBackend):
        def preflight(self, request):
            super().preflight(request)
            if any(question.id == "select_target" for question in request.questions):
                raise RequestValidationError("native context limit exceeded")

    unused = ContextBackend("CLICK")
    assert adapter(unused).predict(payload(2))["answers"]["operation"]["choice"] == "CLICK"
    active = ContextBackend("SELECT")
    with pytest.raises(RequestValidationError) as caught:
        adapter(active).predict(payload(2))
    assert caught.value.hierarchy["selected_operation"] == "SELECT"
    assert len(active.calls) == 1
    assert caught.value.hierarchy["stages"][-1]["error_stage"] == "preflight"


def test_active_probability_failure_retains_raw_response_and_no_partial_answer():
    def invalid(request, result):
        if request.questions[0].id != "operation":
            result["answers"][request.questions[0].id]["probabilities"] = {"wrong": 1.0}
        return result

    backend = NativeBackend(transform=invalid)
    with pytest.raises(BackendResponseError) as caught:
        adapter(backend).predict(payload(2))
    audit = caught.value.hierarchy
    assert audit["status"] == "failed"
    assert audit["stages"][-1]["raw_response"]["answers"]["click_target"]["probabilities"] == {
        "wrong": 1.0
    }
    assert caught.value.generation is audit


def test_narrow_exact_tie_choice_and_confidence_follow_native_normalization():
    def tied(request, result):
        question = request.questions[0]
        if question.id == "click_target":
            result["answers"][question.id] = {
                "choice": "click_target-1",
                "probabilities": {"click_target-0": 0.5, "click_target-1": 0.5},
                "confidence": 999,
            }
        return result

    result = adapter(NativeBackend(transform=tied)).predict(payload(2, unused=2))
    answer = result["answers"]["click_target"]
    assert answer["choice"] == "click_target-1"
    assert answer["confidence"] == 0.5 and answer["margin"] == 0


@pytest.mark.parametrize("wide", [False, True])
def test_original_any_head_width_retains_distinct_deadline_coverage(monkeypatch, wide):
    import s1.jev_browser_operation_first as module

    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])

    def consume(request, result):
        clock[0] += 21
        return result

    backend = NativeBackend("TYPE_TEXT", transform=consume)
    wire = payload(2, unused=21 if wide else 2)
    if wide:
        with pytest.raises(BackendTimeoutError) as caught:
            adapter(backend).predict(wire)
        assert len(backend.calls) == 1
        assert caught.value.hierarchy["stages"][0]["raw_response"]["model"] == backend.name
        assert caught.value.hierarchy["budget_coverage"] == "wide20_cooperative_and_HTTP25"
    else:
        result = adapter(backend).predict(wire)
        assert result["hierarchy"]["budget_coverage"] == "narrow_HTTP25_only_no_native20"


def test_one_wide_timer_is_shared_by_operation_and_all_active_target_stages(monkeypatch):
    import s1.jev_browser_operation_first as module

    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])

    def consume(request, result):
        clock[0] += 6
        return result

    backend = NativeBackend(transform=consume)
    with pytest.raises(BackendTimeoutError) as caught:
        adapter(backend).predict(payload(21))
    assert len(backend.calls) == 4
    assert caught.value.hierarchy["stages"][-1]["status"] == "failed"
    assert caught.value.hierarchy["stages"][-1]["raw_response"]


def test_evidence_sink_errors_remain_visible_and_never_replace_native_results():
    def failed_sink(_):
        raise OSError("journal unavailable")

    result = adapter(NativeBackend("DONE"), evidence_sink=failed_sink).predict(payload())
    assert result["hierarchy"]["evidence_sink_errors"]
    assert result["answers"]["operation"]["choice"] == "DONE"


def test_specialist_configuration_is_required_without_native_override():
    backend = NativeBackend()
    before = copy.deepcopy(backend.spec)
    adapter(backend)
    assert backend.spec == before
    backend.spec.calibration = "custom"
    with pytest.raises(ValueError, match="checkpoint-calibrated"):
        adapter(backend)


def test_explicit_api_profile_journals_full_request_and_only_active_answers():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from s1.api import create_app

    events, backend, wire = [], NativeBackend("TYPE_TEXT"), payload(53)
    app = create_app(
        backend,
        jev_compatible=True,
        jev_hierarchical=True,
        jev_browser_operation_first=True,
        model_identity="native-specialist",
        request_evidence_sink=events.append,
    )
    response = TestClient(app).post("/v1/systemone", json=wire)
    assert response.status_code == 200
    assert set(response.json()["answers"]) == {"operation", "type_text_target"}
    assert events[0]["data"]["payload"] == wire
    assert events[-1]["event"] == "server_returned"
    assert any(event["event"] == "browser_operation_first_checkpoint" for event in events)
    assert not any(event["event"] == "hierarchy_checkpoint" for event in events)


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"jev_compatible": True},
        {"jev_compatible": True, "jev_hierarchical": True, "jev_hierarchical_batch": True},
        {"jev_compatible": True, "jev_hierarchical": True, "jev_generated_actions": True},
    ],
)
def test_api_incompatible_profile_flags_fail_before_backend_access(options):
    pytest.importorskip("fastapi")
    from s1.api import create_app

    backend = NativeBackend()
    with pytest.raises(ValueError):
        create_app(backend, jev_browser_operation_first=True, **options)
    assert backend.calls == []


def cli_driver():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    try:
        spec = importlib.util.spec_from_file_location(
            "operation_first_cli", root / "scripts/jev_local_model.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


@pytest.mark.parametrize(
    "options",
    [
        [],
        ["--competitor", "decider"],
        ["--competitor", "decider", "--hierarchical-candidates"],
        [
            "--competitor",
            "decider",
            "--hierarchical-candidates",
            "--server-evidence",
            "--batch-hierarchy-stages",
        ],
        [
            "--competitor",
            "jev-omni",
            "--hierarchical-candidates",
            "--server-evidence",
            "--prompt-mode",
            "user",
        ],
    ],
)
def test_cli_invalid_profile_flags_precede_upstream_source_or_model_access(
    tmp_path, monkeypatch, options
):
    driver = cli_driver()
    monkeypatch.setattr(
        driver,
        "upstream_manifest",
        lambda *_: pytest.fail("invalid flags reached source/model setup"),
    )
    with pytest.raises(SystemExit) as caught:
        driver.main(
            [
                "--upstream-path",
                "unused",
                "--output-dir",
                str(tmp_path / "run"),
                "--browser-operation-first",
                *options,
            ]
        )
    assert caught.value.code == 2
    assert not (tmp_path / "run").exists()


def test_cli_profile_receipt_is_explicit_and_stops_before_model_imports(tmp_path, monkeypatch):
    from s1 import browser_evidence

    driver = cli_driver()
    monkeypatch.setattr(driver, "upstream_manifest", lambda *_: {"revision": "unit-test"})

    def stop(*_args, **_kwargs):
        raise RuntimeError("CPU-only stop before imports/models")

    monkeypatch.setattr(browser_evidence, "capture_runtime_sources", stop)
    with pytest.raises(SystemExit):
        driver.main(
            [
                "--upstream-path",
                "unused",
                "--output-dir",
                str(tmp_path / "run"),
                "--browser-operation-first",
                "--competitor",
                "decider",
                "--hierarchical-candidates",
                "--server-evidence",
            ]
        )
    receipt = json.loads((tmp_path / "run/receipt.json").read_bytes())
    assert receipt["browser_decision_profile"]["profile"] == PROFILE
    assert "unverified" in " ".join(receipt["browser_decision_profile"]["limitations"])
