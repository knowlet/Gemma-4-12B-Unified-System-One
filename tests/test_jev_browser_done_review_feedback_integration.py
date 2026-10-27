"""Invented requests through the real native parser and semantic binding."""

import copy
import hashlib
import json
import time
from pathlib import Path

import jev_browser_benchmark as runner
import jev_browser_done_review_owner as owner
import jev_decision3_completion_review_route as route
import jev_decision3_semantic_browser_binding as binding
import jev_decision3_semantic_operation_adapter as semantic
import jev_decision3_semantic_operation_audit as audit
import pytest
from test_jev_browser_done_review_owner import CHOOSE, Agent
from test_jev_decision3_native import payload
from test_jev_decision3_operation_gpu import RequestDTO, Tensor, Torch
from test_jev_decision3_semantic_operation_adapter import fake_sha

import s1.jev_decision3_native as native
from s1.evaluation.decision3 import native_payload

pytest_plugins = (
    "test_jev_browser_done_review_owner",
    "test_jev_decision3_semantic_browser_binding",
)


@pytest.mark.parametrize("verdict", ["unsupported", "unknown"])
@pytest.mark.parametrize("singleton", [False, True])
def test_feedback_flows_through_native_parser_and_semantic_binding(
    setup, semantic_binding_context, monkeypatch, verdict, singleton
):
    monkeypatch.setattr(route, "SOURCE_ONLY", False)
    operations = (
        "{'CLICK': 'click', 'DONE': 'done'}"
        if singleton
        else "{'CLICK': 'click', 'DONE': 'done', 'SELECT': 'select'}"
    )
    select_head = (
        ""
        if singleton
        else ", 'select_target': {'type':'choice', 'instructions':'Choose an option', 'criteria': {'1:1': 'option'}}"
    )
    action_source = (
        CHOOSE.replace(
            "{'CLICK': 'click', 'DONE': 'done', 'BLOCKED': 'blocked'}",
            operations,
        )
        .replace("'model': 'synthetic'", "'model': 'native-test'")
        .replace(
            "'click_target': {'type':'choice', 'criteria': {'1': 'continue'}}",
            "'click_target': {'type':'choice', 'instructions':'Choose a target', 'criteria': {'1': 'continue'}}"
            + select_head,
        )
    )
    assert action_source != CHOOSE
    exec(compile(action_source, __file__, "exec"), setup.model.__dict__)
    setup.agent_module.choose = setup.model.choose
    instance = semantic_binding_context.make()
    sent, events = [], []

    def force(raw):
        answers = raw["answers"]
        if owner.REVIEW_ID in answers:
            choice = verdict
            answer = answers[owner.REVIEW_ID]
        else:
            answer = answers["operation"]
            choice = "DONE" if "DONE" in answer["probabilities"] else "CLICK"
        answer["choice"] = choice
        answer["probabilities"] = {key: float(key == choice) for key in answer["probabilities"]}
        answer["confidence"] = 1.0

    semantic_binding_context.base.after = force

    def post(_url, _key, body):
        sent.append(copy.deepcopy(body))
        reply, receipt = instance.service.handle(
            json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode(),
            started=time.monotonic(),
        )
        instance.service.delivery(receipt, delivered=True)
        return json.loads(reply)

    setup.model.post_json = post
    try:
        with route.attach(native, binding, enabled=True):
            with owner.attach_to_runner(
                runner, enabled=True, feedback_enabled=True, event_sink=events.append
            ):
                with runner.model_adapter(
                    setup.agent_module,
                    setup.model,
                    endpoint="http://127.0.0.1:9000",
                    model="native-test",
                    api_key="key",
                ) as (records, _):
                    agent = Agent(setup.model)
                    if singleton:
                        with pytest.raises(ValueError, match="choice needs"):
                            agent.command("tick")
                        assert agent.state["status"] == "ready"
                        assert agent.state["browser"].actions == []
                    else:
                        assert agent.command("tick")["status"] == "ready"
                        assert agent.state["browser"].actions == ["item"]
                    assert agent.state["decision"] is None
        assert len(sent) == len(records) == 3
        if singleton:
            assert len(semantic_binding_context.bindings) == 2
            assert instance.service.failed is True
            assert [row["status"] for row in events][-1] == "replan_failed"
            return
        assert len(semantic_binding_context.bindings) == 3
        assert semantic_binding_context.bindings[-1]["original_request"] == sent[-1]
        assert semantic_binding_context.bindings[-1]["native_payload"]["state"] == sent[-1]["state"]
        assert semantic_binding_context.base.calls[-1].state == sent[-1]["state"]
        assert sent[-1]["state"][owner.FEEDBACK_KEY] == {
            "verdict": verdict,
            "guidance": owner.FEEDBACK_GUIDANCE[verdict],
        }
        assert owner.FEEDBACK_KEY not in sent[0]["state"]
        assert agent.state["decisions"][-1]["request"] == sent[-1]
        event = next(row for row in events if row["status"] == "replan_call")
        assert event["request_sha256"] == owner._body_sha256(sent[-1])
    finally:
        instance.close()


def test_feedback_reaches_real_semantic_adapter_prepare(
    semantic_binding_context, synthetic, monkeypatch, tmp_path
):
    import jev_decision3_candidate_http as http
    import numpy as np

    _, provider, deltas, hashes, _, _, _ = synthetic
    monkeypatch.setattr(semantic, "Decision3NativeRequest", native.Decision3NativeRequest)
    monkeypatch.setattr(semantic, "native_payload", native_payload)
    monkeypatch.setattr(native, "configuration_blockers", lambda _spec: [])
    monkeypatch.setattr(audit, "WIDTH", 2)
    monkeypatch.setattr(audit, "CODES", 3)
    monkeypatch.setattr(audit, "W0_SHA256", fake_sha(provider.runtime.readout))
    monkeypatch.setattr(
        Tensor, "numpy", lambda self: np.asarray(self.values, dtype=np.float32), raising=False
    )
    original_predict, original_prepare = provider.predict, provider.runtime.prepare
    prepared_states = []

    def prepare(*, state, questions):
        prepared_states.append(copy.deepcopy(state))
        return original_prepare(state=state, questions=questions)

    def predict(dto, **kwargs):
        return original_predict(RequestDTO.model_validate(dto.model_dump()), **kwargs)

    provider.predict = predict
    provider.runtime.prepare = prepare
    names = ["CLICK", "TYPE_TEXT"]
    names_sha = hashlib.sha256(binding._encode(names)).hexdigest()
    sink, receipts = audit.make_witness_writer(tmp_path / "feedback-witness", maximum_calls=1)
    records = []
    instance = binding.SemanticBrowserBinding(
        provider,
        deltas,
        hashes,
        Torch,
        lease_module=semantic_binding_context.module,
        lease=semantic_binding_context.lease,
        http_module=http,
        adapter_module=semantic,
        audit_module=audit,
        operation_names=names,
        operation_names_sha256=names_sha,
        model_identity="native-test",
        token="t" * 32,
        max_requests=1,
        identity_guard=lambda *_: {"invented": "stable"},
        owner_guard=lambda *_: {"invented": "owner"},
        witness_sink=sink,
        witness_snapshot=lambda: copy.deepcopy(receipts),
        read_witness=lambda receipt: (Path(receipt["directory"]) / "witness.json").read_bytes(),
        private_artifact_sink=records.append,
        event_sink=lambda _: None,
        closure_guard=lambda: True,
    )
    request = payload(2)
    request["state"][owner.FEEDBACK_KEY] = {
        "verdict": "unknown",
        "guidance": owner.FEEDBACK_GUIDANCE["unknown"],
    }
    request["questions"]["operation"]["criteria"] = {
        key: value
        for key, value in request["questions"]["operation"]["criteria"].items()
        if key in names
    }
    request["questions"] = {
        key: value
        for key, value in request["questions"].items()
        if key in {"operation", "click_target", "type_text_target"}
    }
    body = json.dumps(request, separators=(",", ":")).encode()
    try:
        response, row = instance.service.handle(body, started=time.monotonic())
        assert (
            binding.verify_client_exchange(records[0], body, response)["status"]
            == "provided_bytes_match"
        )
        assert records[0]["native_payload"]["state"] == request["state"]
        assert prepared_states and all(state == request["state"] for state in prepared_states)
        receipt = receipts[0]
        result = audit.audit_directory(
            receipt["directory"],
            expected_file_sha256=receipt["manifest_sha256"],
            expected_witness_sha256=receipt["witness_sha256"],
            expected_delta_sha256=hashes,
            expected_names_sha256=names_sha,
        )
        assert result["status"] == "numeric_consistency_only"
        instance.service.delivery(row, delivered=True)
    finally:
        instance.close()


@pytest.mark.parametrize("verdict", ["unsupported", "unknown"])
def test_singleton_stop_uses_real_parser_service_binding_without_replan(
    setup, semantic_binding_context, monkeypatch, verdict
):
    monkeypatch.setattr(route, "SOURCE_ONLY", False)
    source = (
        CHOOSE.replace(
            "{'CLICK': 'click', 'DONE': 'done', 'BLOCKED': 'blocked'}",
            "{'DONE': 'done', 'BLOCKED': 'blocked'}",
        )
        .replace("'model': 'synthetic'", "'model': 'native-test'")
        .replace(", 'click_target': {'type':'choice', 'criteria': {'1': 'continue'}}", "")
    )
    assert source != CHOOSE
    setup.model.action_space = lambda _actions: ([], {}, {})
    exec(compile(source, __file__, "exec"), setup.model.__dict__)
    setup.agent_module.choose = setup.model.choose
    instance = semantic_binding_context.make(
        operation_names=["BLOCKED", "CLICK", "DONE", "SELECT", "TYPE_TEXT"]
    )
    sent, events = [], []

    def force(raw):
        answers = raw["answers"]
        key = owner.REVIEW_ID if owner.REVIEW_ID in answers else "operation"
        choice = verdict if key == owner.REVIEW_ID else "DONE"
        answer = answers[key]
        answer["choice"] = choice
        answer["probabilities"] = {key: float(key == choice) for key in answer["probabilities"]}
        answer["confidence"] = 1.0

    semantic_binding_context.base.after = force

    def post(_url, _key, body):
        sent.append(copy.deepcopy(body))
        reply, receipt = instance.service.handle(
            json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode(),
            started=time.monotonic(),
        )
        instance.service.delivery(receipt, delivered=True)
        return json.loads(reply)

    setup.model.post_json = post
    try:
        with route.attach(native, binding, enabled=True):
            with owner.attach_to_runner(
                runner,
                enabled=True,
                feedback_enabled=True,
                singleton_stop_enabled=True,
                event_sink=events.append,
            ):
                with runner.model_adapter(
                    setup.agent_module,
                    setup.model,
                    endpoint="http://127.0.0.1:9000",
                    model="native-test",
                    api_key="key",
                ) as (records, _):
                    agent = Agent(setup.model)
                    agent.state["page"]["actions"] = []
                    assert agent.command("tick")["status"] == "blocked"
                    assert agent.state["decision"] is None and agent.state["history"] == []
                    assert agent.state["browser"].actions == []
                    assert len(agent.state["decisions"]) == 1
                    assert agent.state["decisions"][0]["choice"] == "DONE"
                    assert len(records) == len(sent) == 2
        assert len(semantic_binding_context.bindings) == 2
        assert instance.service.failed is False
        assert set(sent[0]["questions"]["operation"]["criteria"]) == {"DONE", "BLOCKED"}
        assert set(sent[0]["questions"]) == {"operation"}
        assert set(sent[1]["questions"]) == {owner.REVIEW_ID}
        assert [event["status"] for event in events] == ["review_call", "singleton_stop_requested"]
        assert events[-1]["review_request_sha256"] == owner._body_sha256(sent[-1])
    finally:
        instance.close()
