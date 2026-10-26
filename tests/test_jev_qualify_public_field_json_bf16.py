"""Independent field-helper qualification contracts; never deployment evidence.

Genuine source90/cache bytes are read, while generation and ASGI requests use
explicit fakes. No listening socket, HTTP client, weights, Torch or GPU is used.
"""

import ast
import asyncio
import copy
import importlib.util
import json
import subprocess
import sys
import threading
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import jev_qualify_public_actions_bf16 as OLD
import jev_qualify_public_field_json_bf16 as Q
import pytest
from local_evidence import require_local_evidence

ROOT = Path(__file__).resolve().parents[1]
HISTORY = (
    ROOT
    / "artifacts/jev-runtime/public-native-action-bf16-qualification-plan-20261016-01/plan-02.json"
)
HANDOFF = (
    ROOT
    / "artifacts/jev-runtime/public-native-action-epoch-ablation-20261018-01/actual-fit-240-02-parallel-cli/public07-h200-240-20261018-01-fit/fit/source-handoff.json"
)


def wrapper():
    require_local_evidence(HISTORY)
    historical = HISTORY.read_bytes()
    delta = Q.strict_json(historical)
    return (ROOT / delta["base_plan"]["path"]).read_bytes(), {
        "schema_id": "jev-public-field-json-bf16-qualification-design-v1",
        "base_plan": delta["base_plan"],
        "historical_delta_utf8": historical.decode(),
        "extension": Q.field_json_design_extension(),
    }


@pytest.mark.local_evidence
def test_historical_design_preserves_original_inference_design():
    base, selected = wrapper()
    effective = Q.apply_design_delta(base, Q.canonical(selected))
    assert effective["historical_inference_design"] == OLD.apply_design_delta(
        base, HISTORY.read_bytes()
    )


def test_all41_native_physical_contracts_are_preserved():
    assert Q.expected_case_ids() == OLD.expected_case_ids()
    assert len(Q.expected_case_ids()) == 41
    profile = Q.runtime_profile()
    helper = profile.pop("field_helper")
    assert profile == OLD.runtime_profile()
    assert helper["actual_tokens"] == 128 and helper["actual_EOS_ids"] == [1, 106, 50]
    assert helper["whole_seconds"] == 20 and helper["http_seconds"] == 25
    assert Q.KIND != OLD.KIND != "jev_public_cfg_action_bf16_qualification_v1"
    assert Q.field_json_design_extension()["extensions"] == sorted(Q.FIELD_EXTENSIONS)


@pytest.mark.parametrize("fault", ["historical", "base", "kind", "tokens", "extra"])
@pytest.mark.local_evidence
def test_rehashed_design_cannot_change_history_or_helper_contract(fault):
    base, selected = wrapper()
    if fault == "historical":
        selected["historical_delta_utf8"] += " "
    elif fault == "base":
        selected["base_plan"]["sha256"] = "0" * 64
    elif fault == "kind":
        selected["schema_id"] = "jev-public-cfg-action-bf16-qualification-design-v1"
    elif fault == "tokens":
        selected["extension"]["helper_contract"]["actual_tokens"] = 129
    else:
        selected["extra"] = True
    with pytest.raises(ValueError):
        Q.apply_design_delta(base, Q.canonical(selected))


def test_original_guard_and_tensor_physical_checks_are_not_reimplemented_or_weakened():
    old = ast.parse(Path(OLD.__file__).read_bytes())
    new = ast.parse(Path(Q.__file__).read_bytes())
    for name in ("PublicBF16Guard", "_TensorAPI"):

        def pick(tree):
            return next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)

        assert ast.dump(pick(old)) == ast.dump(pick(new))
    for node in ast.walk(new):
        if isinstance(node, ast.Import):
            assert all(a.name != "jev_qualify_public_actions_bf16" for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module != "jev_qualify_public_actions_bf16"


@pytest.mark.parametrize("scope,phase", [("cpu", "startup"), ("load", "base-hash")])
def test_original_fresh_resource_floor_has_no_resident_or_swap_credit(tmp_path, scope, phase):
    total = 130595860480
    reserve = max(24 * Q.GIB, (total + 4) // 5)
    floor = reserve + (4 * Q.GIB if scope == "cpu" else 2 * Q.BASE_BYTES)
    for offset in (0, -1):
        journal = tmp_path / f"{scope}-{offset}.jsonl"
        gate = Q.QualificationGate(
            journal,
            "boot",
            observe=lambda: {"MemTotal": total, "MemAvailable": floor + offset, "boot_id": "boot"},
            clock=lambda: 1.0,
        )
        if offset:
            with pytest.raises(ValueError, match="resource admission refused"):
                gate.check(phase, scope)
        else:
            assert gate.check(phase, scope)["admitted"] is True
        row = Q.strict_json(journal.read_bytes().splitlines()[0])
        assert row["required_bytes"] == floor and row["reserve_bytes"] == reserve
        assert row["swap_or_resident_credit_bytes"] == 0


@pytest.fixture
def genuine93():
    require_local_evidence(HANDOFF)
    training_blob = HANDOFF.read_bytes()
    assert Q.digest(training_blob) == Q.TRAINING90_SHA
    original = Q._artifact_module().verify_source_handoff(
        training_blob, expected_sha256=Q.TRAINING90_SHA
    )
    assert len(original["files"]) == 90
    files = {**original["files"], **{name: Q.pin(ROOT / name) for name in Q.FIELD_EXTENSIONS}}
    files = dict(sorted(files.items()))
    return training_blob, {
        "schema_id": "jev_public_action_field_json_qualification_execution_v1",
        "training_handoff": {
            "sha256": Q.TRAINING90_SHA,
            "size_bytes": len(training_blob),
            "identity_sha256": original["identity_sha256"],
        },
        "files": files,
        "identity_sha256": Q.digest(Q.canonical(files)),
    }


def verify93(fixture, changed=None):
    training, value = fixture
    blob = Q.canonical(value if changed is None else changed)
    return Q.verify_execution_sources(
        blob,
        expected_sha256=Q.digest(blob),
        expected_training_sha256=Q.TRAINING90_SHA,
        root=ROOT,
        training_blob=training,
    )


@pytest.mark.local_evidence
def test_real_original90_plus_exact_selected3_is_detached_immutable_and_rechecked(genuine93):
    verified = verify93(genuine93)
    assert len(verified.manifest()["files"]) == 93
    verified.manifest()["files"].clear()
    assert len(verified.manifest()["files"]) == 93
    with pytest.raises(FrozenInstanceError):
        verified.blob = b"{}"
    verified.check()


@pytest.mark.parametrize("fault", ["old91", "cfg95", "oldQ", "missing", "helper", "old90", "order"])
@pytest.mark.local_evidence
def test_even_rehashed_source_authority_cannot_substitute_old_or_CFG_closures(genuine93, fault):
    value = copy.deepcopy(genuine93[1])
    if fault == "old91":
        value["schema_id"] = "jev_public_action_qualification_execution_v1"
    elif fault == "cfg95":
        value["schema_id"] = "jev_public_cfg_action_qualification_execution_v1"
    elif fault == "oldQ":
        value["files"]["scripts/jev_qualify_public_actions_bf16.py"] = Q.pin(
            ROOT / "scripts/jev_qualify_public_actions_bf16.py"
        )
    elif fault == "missing":
        value["files"].pop(Q.HELPER_SELF)
    elif fault == "helper":
        value["files"][Q.HELPER_SELF]["sha256"] = "0" * 64
    elif fault == "old90":
        label = next(iter(set(value["files"]) - Q.FIELD_EXTENSIONS))
        value["files"][label]["sha256"] = "0" * 64
    else:
        value["files"] = dict(reversed(list(value["files"].items())))
    value["identity_sha256"] = Q.digest(Q.canonical(value["files"]))
    blob = Q.wire_bytes(value)
    with pytest.raises(ValueError):
        Q.verify_execution_sources(
            blob,
            expected_sha256=Q.digest(blob),
            expected_training_sha256=Q.TRAINING90_SHA,
            root=ROOT,
            training_blob=genuine93[0],
        )


def phases(message="Text helper context exceeds the explicit token budget"):
    return [
        {"name": "disabled_checked"},
        {"name": "helper_body_failed", "error": {"type": "FieldJSONError", "message": message}},
        {"name": "restored_checked"},
    ]


def context_failure():
    return {
        "profile": Q.FIELD_PROFILE,
        "cache_manifest_sha256": Q.FIELD_CACHE_SHA,
        "adapter_restore_complete": True,
        "evidence_failed": False,
        "secondary_errors": [],
        "success": False,
        "response_authorized": False,
        "base_context_entered": True,
        "disabled_check_passed": True,
        "failure_stage": "tokenization",
        "error": {
            "type": "ValueError",
            "message": "Text helper context exceeds the explicit token budget",
        },
        "usage": {"input_tokens": 8065},
        "raw_token_ids": None,
        "callback_events": [],
        "native_generation_returned": False,
    }


def test_case37_only_exact_wrapped_overflow_zero_forward_and_verified_restore_is422():
    g = context_failure()
    _, helper = Q._field_modules()
    exc = helper.FieldJSONError(g["error"]["message"], g)
    trace = {"prefix": {"token_count": 8065}, "forward_events": [], "helper_phases": phases()}
    status, body = Q._field_failure_response(exc, trace)
    assert status == 422 and trace["context_refusal"] is True
    assert body["error"]["type"] == "helper_context_budget" and body["generation"] is g


@pytest.mark.parametrize(
    "fault",
    [
        "cache",
        "restore",
        "evidence",
        "secondary",
        "inner_type",
        "stage",
        "forward",
        "phase",
        "prefix",
        "raw",
    ],
)
def test_context_lookalikes_and_integrity_errors_raise_the_original_exception(fault):
    g, prefix, count, ps = context_failure(), {"token_count": 8065}, 0, phases()
    if fault == "cache":
        g["cache_manifest_sha256"] = "0" * 64
    elif fault == "restore":
        g["adapter_restore_complete"] = False
    elif fault == "evidence":
        g["evidence_failed"] = True
    elif fault == "secondary":
        g["secondary_errors"] = [{"stage": "guard_restore_check"}]
    elif fault == "inner_type":
        g["error"]["type"] = "RuntimeError"
    elif fault == "stage":
        g["failure_stage"] = "guard_entry"
    elif fault == "forward":
        count = 1
    elif fault == "phase":
        ps.pop()
    elif fault == "prefix":
        prefix["token_count"] = 8064
    else:
        g["raw_token_ids"] = [106]
    assert Q._field_failure_kind(g, prefix, count, ps) is None
    _, helper = Q._field_modules()
    exc = helper.FieldJSONError("original integrity refusal", g)
    trace = {"prefix": prefix, "forward_events": [{}] * count, "helper_phases": ps}
    with pytest.raises(helper.FieldJSONError) as caught:
        Q._field_failure_response(exc, trace)
    assert caught.value is exc


def test_unknown_native_and_fatal_errors_keep_original_identity():
    for exc in (
        OSError("durability"),
        ValueError("unknown native"),
        KeyboardInterrupt("owned cancellation"),
    ):
        with pytest.raises(type(exc)) as caught:
            Q._field_failure_response(exc, {})
        assert caught.value is exc


@pytest.mark.parametrize("kind,status", [("consumer", 502), ("deadline", 504)])
def test_only_explicit_ordinary_consumer_and_deadline_failures_have_typed_responses(kind, status):
    generation = context_failure()
    if kind == "consumer":
        message = "Generated field value is null, blank or exceeds2000 characters"
        generation.update(
            failure_stage="consumer",
            error={"type": "ValueError", "message": message},
            native_generation_returned=True,
            generation_complete=True,
            terminated_by_actual_EOS=True,
            grammar_terminal_accepted=True,
            finish_reason="actual_EOS",
            candidate_valid=False,
        )
    else:
        message = "Whole constrained helper20s exceeded after evidence"
        generation.update(
            failure_stage="deadline_after_evidence",
            error={"type": "TimeoutError", "message": message},
            deadline_exceeded=True,
        )
    _, helper = Q._field_modules()
    trace = {"prefix": {"token_count": 2}, "forward_events": [], "helper_phases": phases(message)}
    observed, response = Q._field_failure_response(
        helper.FieldJSONError(message, generation), trace
    )
    assert observed == status and trace["helper_error"]["kind"] == kind
    assert response["generation"] is generation


@pytest.fixture(scope="module")
def actual_helper_transcript():
    spec = importlib.util.spec_from_file_location(
        "fieldQ_fake_LM_fixtures", ROOT / "tests/test_jev_field_json_helper.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch = pytest.MonkeyPatch()
    try:
        assets = module.assets.__wrapped__()
        case = module.factory.__wrapped__(assets, monkeypatch)(text='{"text":"x"}', prepare=False)
        case.model.name, case.model.revision, case.model.quantization = Q.MODEL, Q.REVISION, "none"
        case.prepared = module.H.prepare_helper(case.model, **case.selection)
        text, generation = module.run(case)
        assert text == "x"
        prefix = {
            "ids": case.model.tok.prompt,
            "token_count": len(case.model.tok.prompt),
            "prompt_sha256": generation["prompt_sha256"],
        }
        events = [{"request_id": "r" * 32, **event} for event in case.sink]
        yield generation, prefix, case.prepared.cache, events
    finally:
        monkeypatch.undo()


@pytest.mark.frozen_decoder
def test_actual_helper_fake_LM_transcript_validates_full_native_masks_EOS_and_preauthorization_events(
    actual_helper_transcript,
):
    g, p, cache, events = actual_helper_transcript
    assert Q._validate_field_generation(g, p, len(g["raw_token_ids"]), cache) == "x"
    Q._validate_field_events(b"\n".join(Q.canonical(e) for e in events), "r" * 32, g)
    assert (
        events[-1]["generation"]["response_authorized"] is False
        and g["response_authorized"] is True
    )


@pytest.mark.parametrize(
    "fault",
    [
        "EOS",
        "usage",
        "callback_prefix",
        "callback_mask",
        "callback_count",
        "prompt",
        "decoded_value",
        "fence",
        "null",
        "secondary",
        "deadline",
    ],
)
@pytest.mark.frozen_decoder
def test_native_generation_evidence_cannot_be_repaired_or_forge_prefix_masks_or_terminal(
    actual_helper_transcript, fault
):
    original, prefix, cache, _ = actual_helper_transcript
    g, p = copy.deepcopy(original), copy.deepcopy(prefix)
    if fault == "EOS":
        g["raw_token_ids"][-1] = 1005
    elif fault == "usage":
        g["usage"]["output_tokens"] -= 1
    elif fault == "callback_prefix":
        g["callback_events"][-1]["generated_ids"] = []
    elif fault == "callback_mask":
        g["callback_events"][0]["allowed_mask_sha256"] = "0" * 64
    elif fault == "callback_count":
        g["callback_events"][0]["allowed_count"] -= 1
    elif fault == "prompt":
        p["ids"] = [2, 4]
    elif fault == "decoded_value":
        g["raw_output"] = '{"text":"y"}'
    elif fault == "fence":
        g["raw_output"] = '```json\n{"text":"x"}\n```'
    elif fault == "null":
        g["raw_output"] = '{"text":null}'
    elif fault == "secondary":
        g["secondary_errors"] = [{"stage": "restore"}]
    else:
        g["deadline_exceeded"] = True
    with pytest.raises(ValueError):
        Q._validate_field_generation(g, p, len(g["raw_token_ids"]), cache)


@pytest.mark.parametrize(
    "pieces,expected",
    [
        ([0xE4, 0xBD, 0xA0], "你"),
        ([0x61, 0xFF], "\ufffd" * 2),
        ([0xE4, 0xBD, 0xA0, 0xFF], "\ufffd" * 4),
        ([0xE2, 0x96, 0x81], "\u2581"),
        ([0x22, 0x61, 0xFF], "\ufffd" * 3),
        ([0xE4, "x", 0xBD, 0xA0], "\ufffdx\ufffd\ufffd"),
    ],
)
@pytest.mark.frozen_decoder
def test_native_decoder_projection_preserves_whole_fallback_run_and_flush_boundaries(
    actual_helper_transcript, pieces, expected
):
    _, _, cache, _ = actual_helper_transcript
    grammar, _ = Q._field_modules()
    ordinary = {
        piece.decode("utf-8"): index
        for index, piece in enumerate(cache.pieces)
        if cache.classes[index] == grammar.ORDINARY
    }
    byte_ids = {cache.pieces[index][0]: index for index in cache.byte_ids}
    body = [ordinary[p] if isinstance(p, str) else byte_ids[p] for p in pieces]
    ids = [ordinary[p] for p in '{"text":"'] + body + [ordinary[p] for p in '"}'] + [106]
    assert Q._field_decode_ids(cache, ids) == '{"text":"' + expected + '"}'
    assert cache.consume(ids).terminated


@pytest.mark.frozen_decoder
def test_native_decoder_projection_cannot_skip_nonEOS_specials(actual_helper_transcript):
    _, _, cache, _ = actual_helper_transcript
    with pytest.raises(ValueError, match="nonEOS special"):
        Q._field_decode_ids(cache, [0])


@pytest.mark.parametrize(
    "fault", ["request", "order", "sequence", "raw", "callback", "authorized", "restore"]
)
@pytest.mark.frozen_decoder
def test_durable_event_sequence_and_transition_cannot_be_fabricated(
    actual_helper_transcript, fault
):
    generation, _, _, source = actual_helper_transcript
    events = copy.deepcopy(source)
    if fault == "request":
        events[0]["request_id"] = "other"
    elif fault == "order":
        events.reverse()
    elif fault == "sequence":
        events[0]["sequence"] = True
    elif fault == "raw":
        events[0]["generation"]["raw_output"] = generation["raw_output"]
    elif fault == "callback":
        events[1]["generation"]["callback_events"] = []
    elif fault == "authorized":
        events[-1]["generation"]["response_authorized"] = True
    else:
        events[-1]["generation"]["adapter_restore_complete"] = False
    with pytest.raises(ValueError):
        Q._validate_field_events(b"\n".join(Q.canonical(e) for e in events), "r" * 32, generation)


def app_state():
    return SimpleNamespace(
        state=SimpleNamespace(
            field_completion_lock=threading.RLock(),
            field_completions={},
            field_drained={},
            field_hard_halt=None,
            fatal_primary=None,
            request_evidence_errors=[],
        )
    )


def complete(app, *, rid="a" * 32, candidate=200):
    return Q._field_completion(
        app,
        started=100.0,
        request_id=rid,
        ordinal=36,
        raw_path="raw.json",
        raw_pin={"sha256": "1" * 64, "size_bytes": 1},
        candidate_status=candidate,
        candidate_sha="2" * 64,
        refusal_sha="3" * 64,
    )


@pytest.mark.parametrize("elapsed,status", [(19.999, 200), (20.0, 504), (20.001, 504)])
def test_final_outer_clock_has_exact20s_boundary_and_no_postgate_IO(monkeypatch, elapsed, status):
    app = app_state()
    monkeypatch.setattr(Q.time, "monotonic", lambda: 100 + elapsed)
    monkeypatch.setattr(Q, "create", lambda *args: pytest.fail("inline postgate I/O"))
    selected = complete(app)
    assert selected["selected_status"] == status and selected["outer_authorized"] is (status == 200)
    assert selected["selected_body_sha256"] == ("2" if status == 200 else "3") * 64
    selected["selected_status"] = 777
    assert dict(app.state.field_completions["a" * 32])["selected_status"] == status
    assert type(app.state.field_completions["a" * 32]) is tuple


def test_completed_rows_drain_after_existing_hardhalt_without_clearing_original_exception(
    tmp_path, monkeypatch
):
    app = app_state()
    monkeypatch.setattr(Q.time, "monotonic", lambda: 101.0)
    complete(app)
    primary = KeyboardInterrupt("prior owned fatal")
    Q._field_halt(app, primary)
    assert Q._drain_field_completions(app, tmp_path, qualifying=True) == 1
    assert app.state.field_hard_halt is app.state.fatal_primary is primary
    record = app.state.field_drained["a" * 32]
    assert Q.pin(tmp_path / record["path"]) == {k: record[k] for k in ("sha256", "size_bytes")}
    assert Q.strict_json((tmp_path / record["path"]).read_bytes())["selected_status"] == 200
    assert Q._drain_field_completions(app, tmp_path, qualifying=True) == 0


def test_drainer_write_failure_stickyhalts_and_never_replaces_prior_fatal(tmp_path, monkeypatch):
    app = app_state()
    monkeypatch.setattr(Q.time, "monotonic", lambda: 101.0)
    complete(app)
    original = OSError("completion evidence refused")
    monkeypatch.setattr(Q, "create", lambda *args: (_ for _ in ()).throw(original))
    with pytest.raises(OSError) as caught:
        Q._drain_field_completions(app, tmp_path, qualifying=False)
    assert caught.value is original and app.state.fatal_primary is original
    Q._field_halt(app, ValueError("secondary"))
    assert app.state.field_hard_halt is original and not app.state.field_drained


@pytest.fixture
def boundary(tmp_path, monkeypatch):
    from fastapi.responses import Response

    app, clock, live = app_state(), [100.0], set()
    prefix = {"ids": [2, 4], "attention_mask": [1, 1], "token_count": 2, "prompt_sha256": "a" * 64}
    context = {
        "goal": "Enter x",
        "field": {"label": "Label", "role": "textbox", "value": ""},
        "page": {"title": "Title", "text": "Page"},
        "recent_actions": [],
    }
    payload = {
        "model": Q.IDENTITY,
        "max_tokens": 1024,
        "response_format": {"type": "json_object"},
        "reasoning": {"enabled": False},
        "messages": [
            {"role": "system", "content": Q.TEXT_VALUE},
            {"role": "user", "content": json.dumps(context)},
        ],
    }
    encoded = Q.wire_bytes(payload)

    class Request:
        method = "POST"
        headers = {"authorization": "Bearer pure-local-token"}

        async def body(self):
            return encoded

    guard = SimpleNamespace(check=lambda: {"success": True})
    prepared = SimpleNamespace(check=lambda: None)
    checker = SimpleNamespace(check=lambda *a: None, freeze=lambda: None)
    source_checker = SimpleNamespace(check=lambda: None)
    monkeypatch.setattr(Q.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(Q, "_request_budget", lambda *a: None)
    monkeypatch.setattr(Q, "_sources", lambda *a: source_checker)
    monkeypatch.setattr(Q, "_source_copy", lambda *a: None)
    monkeypatch.setattr(Q, "_field_cache_bytes", lambda **kw: None)
    monkeypatch.setattr(Q, "_field_native_runtime", lambda *a: None)
    monkeypatch.setattr(Q, "serve_request_gate", lambda *a: checker)
    monkeypatch.setattr(Q, "_prefix", lambda *a, **kw: (prefix, "original synthetic prompt"))
    response_value = {"choices": [{"message": {"role": "assistant", "content": '{"text":"x"}'}}]}
    response_bytes = Q.wire_bytes(response_value)

    async def next_call(request):
        trace = Q.REQUEST_TRACE.get()
        trace["generation"] = {"success": True, "response_authorized": True}
        trace["helper_phases"] = [
            {"name": "disabled_checked"},
            {"name": "helper_body_returned"},
            {"name": "restored_checked"},
        ]
        clock[0] += 0.001
        return Response(response_bytes, status_code=200, media_type="application/json")

    cases = [None] * 41
    cases[36] = {"kind": "helper", "payload": payload, "prefix": prefix}

    def run(call=next_call, *, live_mode=False, application=None):
        return asyncio.run(
            Q._field_request_boundary(
                Request(),
                call,
                app=app if application is None else application,
                model=object(),
                guard=guard,
                prepared=prepared,
                config={"token": "pure-local-token", "boot_id": "synthetic-boot"},
                output=tmp_path,
                gate=checker,
                inputs=None if live_mode else {"cases": cases},
                claim_ordinal=lambda: 36,
                live=live,
            )
        )

    return SimpleNamespace(
        app=app,
        clock=clock,
        live=live,
        run=run,
        guard=guard,
        prepared=prepared,
        output=tmp_path,
        value=response_value,
        next_call=next_call,
        gate=checker,
    )


@pytest.mark.parametrize("phase", ["raw_fsync", "restore", "serialize"])
def test_direct_ASGI_boundary_includes_raw_restore_and_response_serialization_cost(
    boundary, monkeypatch, phase
):
    case = boundary
    if phase == "raw_fsync":
        original = Q.create

        def delayed(path, blob):
            original(path, blob)
            if Path(path).name == "036.json":
                case.clock[0] += 20

        monkeypatch.setattr(Q, "create", delayed)
    elif phase == "restore":

        def restored():
            case.clock[0] += 20
            return {"success": True}

        case.guard.check = restored
    else:
        original = Q.wire_bytes

        def serialized(value):
            blob = original(value)
            if value == case.value:
                case.clock[0] += 20
            return blob

        monkeypatch.setattr(Q, "wire_bytes", serialized)
    response = case.run()
    assert response.status_code == 504 and b'"helper_deadline"' in response.body
    raw = Q.strict_json((case.output / "run/cases/036.json").read_bytes())
    assert raw["candidate_response"] == case.value and raw["response"] is None
    assert raw["outer_authorized"] is raw["final_gate_measured"] is False
    final = dict(next(iter(case.app.state.field_completions.values())))
    assert final["candidate_status"] == 200 and final["selected_status"] == 504
    assert final["elapsed_ms"] >= 20000 and not final["outer_authorized"]
    assert not case.live and Q.REQUEST_TRACE.get() is None


def test_direct_boundary_raw_write_failure_never_queues_authorization_and_preserves_exception(
    boundary, monkeypatch
):
    original, primary = Q.create, OSError("protected raw fsync failed")

    def failed(path, blob):
        if Path(path).name == "036.json":
            raise primary
        return original(path, blob)

    monkeypatch.setattr(Q, "create", failed)
    with pytest.raises(OSError) as caught:
        boundary.run()
    assert caught.value is primary
    assert boundary.app.state.fatal_primary is boundary.app.state.field_hard_halt is primary
    assert not boundary.app.state.field_completions and not boundary.live
    assert Q.REQUEST_TRACE.get() is None


def test_original_fatal_type_survives_final_restore_failure(boundary):
    primary = KeyboardInterrupt("owned cancellation")

    async def fatal(request):
        raise primary

    def restore():
        raise ValueError("secondary restore failure")

    boundary.guard.check = restore
    with pytest.raises(KeyboardInterrupt) as caught:
        boundary.run(fatal)
    assert caught.value is primary and any("restoration" in n for n in primary.__notes__)
    assert boundary.app.state.fatal_primary is primary
    assert not boundary.app.state.field_completions and Q.REQUEST_TRACE.get() is None


@pytest.mark.parametrize("phase", ["final-cpu", "freeze"])
def test_live_boundary_includes_final_CPU_gate_and_freeze_cost(boundary, phase):
    def check(observed_phase, scope):
        if observed_phase == phase:
            boundary.clock[0] += 20

    boundary.gate.check = check
    if phase == "freeze":
        boundary.gate.freeze = lambda: boundary.clock.__setitem__(0, boundary.clock[0] + 20)
    response = boundary.run(live_mode=True)
    assert response.status_code == 504 and b'"helper_deadline"' in response.body
    row = dict(next(iter(boundary.app.state.field_completions.values())))
    assert row["elapsed_ms"] >= 20000 and row["candidate_status"] == 200
    assert row["selected_status"] == 504 and not row["outer_authorized"]


def test_current_native_decoder_binary_drift_is_rejected(tmp_path, monkeypatch):
    origin = tmp_path / "original-decoder.so"
    origin.write_bytes(b"owned fake native binary")
    root = tmp_path / "captured"
    metadata = {"origin": str(origin), **Q.pin(origin), "version": "fake-metadata-only"}
    prepared = {
        "actual_decoder": {
            "binary_path": str(origin),
            "binary_sha256": metadata["sha256"],
            "version": metadata["version"],
        }
    }
    Q.create(root / "field-helper-prepared.json", Q.canonical(prepared))
    Q.create(root / "helper-native-runtime/identity.json", Q.canonical(metadata))
    Q.create(root / "helper-native-runtime/tokenizers-native.bin", origin.read_bytes())
    monkeypatch.setitem(sys.modules, "tokenizers.tokenizers", SimpleNamespace(__file__=str(origin)))
    assert Q._field_native_runtime(root) == metadata
    origin.write_bytes(b"different decoder binary")
    with pytest.raises(ValueError, match="decoder source binary changed"):
        Q._field_native_runtime(root)


def test_installed_live_dispatch_halts_both_policy_and_helper_after_decoder_identity_failure(
    boundary, monkeypatch
):
    from fastapi import Request

    class Prepared:
        def __init__(self, model, guard):
            self.model, self.guard = model, guard

        def check(self):
            return None

    class Backbone:
        def register_forward_pre_hook(self, *args, **kwargs):
            return SimpleNamespace(remove=lambda: None)

    model = SimpleNamespace(
        name=Q.MODEL,
        revision=Q.REVISION,
        quantization="none",
        quantization_details={},
        processor=None,
        tok=None,
        device="no-device-pure-test",
        max_context=8192,
        backbone=Backbone(),
        lm=SimpleNamespace(generate=lambda *a, **kw: pytest.fail("fake LM must never forward")),
    )
    prepared = Prepared(model, boundary.guard)
    monkeypatch.setattr(
        Q, "_field_modules", lambda: (None, SimpleNamespace(PreparedHelper=Prepared))
    )
    app, _, _, _ = Q._make_application(
        model,
        boundary.guard,
        {"token": "pure-local-token"},
        boundary.output,
        boundary.gate,
        field_prepared=prepared,
    )
    primary = ValueError("actual native decoder binding changed")
    monkeypatch.setattr(Q, "_field_native_runtime", lambda *args: (_ for _ in ()).throw(primary))
    with pytest.raises(ValueError) as caught:
        boundary.run(application=app)
    assert caught.value is app.state.field_hard_halt is primary
    dispatch = app.user_middleware[0].kwargs["dispatch"]

    async def forbidden(request):
        pytest.fail("post-halt request reached native dispatch")

    for path in ("/chat/completions", "/v1/chat/completions"):
        request = Request({"type": "http", "method": "POST", "path": path, "headers": []})
        with pytest.raises(ValueError) as blocked:
            asyncio.run(dispatch(request, forbidden))
        assert blocked.value is primary


def test_pure_suite_never_imports_model_frameworks():
    # Whole-suite inference collection legitimately imports model frameworks.
    # Check this module's import closure without cleaning or weakening that state.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy, sys; "
            f"sys.path.insert(0, {str(ROOT / 'scripts')!r}); "
            f"sys.path.insert(0, {str(ROOT / 'tests')!r}); "
            f"runpy.run_path({str(Path(__file__).resolve())!r}); "
            "assert not any(name.split('.')[0] in "
            "{'torch', 'transformers', 'peft', 'modal'} for name in sys.modules)",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
