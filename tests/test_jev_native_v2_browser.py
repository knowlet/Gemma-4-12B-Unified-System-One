"""Portable production composition checks; no model, CUDA or browser is started."""

import importlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import jev_native_v2_browser as driver
import pytest


class FakeLM:
    def __init__(self, *, refuses_disable=False):
        self.enabled, self.refuses_disable = True, refuses_disable
        self.parameter = SimpleNamespace(requires_grad=False)

    def get_model_status(self):
        return SimpleNamespace(
            enabled=self.enabled,
            merged_adapters=[],
            active_adapters=["default"],
            available_adapters=["default"],
        )

    def parameters(self):
        return iter([self.parameter])

    @contextmanager
    def disable_adapter(self):
        self.enabled = self.refuses_disable
        try:
            yield
        finally:
            self.enabled = True
            self.parameter.requires_grad = True  # PEFT restoration is frozen by the real guard.

    def requires_grad_(self, value):
        self.parameter.requires_grad = value

    def eval(self):
        pass

    def generate(self, **_kwargs):
        pytest.fail("No real or fake model generation belongs in composition tests")


def model_fixture(*, refuses_disable=False, prompt_format="raw-json-v1"):
    from s1.browser_action_adapter import _AttachedActionAdapter
    from s1.browser_action_prompt import prompt_contract

    metadata = {"kind": "jev_action_lora_v1"}
    if prompt_format == "shared-elements-v2":
        metadata = {"kind": "jev_native_action_lora_v2", "prompt_contract": prompt_contract()}
    model = SimpleNamespace(
        lm=FakeLM(refuses_disable=refuses_disable),
        name=driver.BASE_ID,
        revision=driver.REVISION,
        quantization="nf4",
        quantization_details={},
        device="cpu",
        max_context=8192,
        processor=object(),
        tok=object(),
        head=SimpleNamespace(weight=SimpleNamespace(_version=0)),
        backbone=SimpleNamespace(register_forward_pre_hook=lambda *_a, **_k: None),
        temperature=2.783,
    )
    model.action_adapter = _AttachedActionAdapter(model, model.lm, metadata)
    return model


@pytest.mark.parametrize("fail", [False, True])
def test_bridge_delegates_real_attached_guard_and_restores_after_failure(fail):
    model = model_fixture()
    original, metadata = model.action_adapter, model.action_adapter.metadata
    guard = driver.CurrentNF4HelperGuard(model)
    assert guard.original is original and guard.metadata is metadata
    with guard.decision_context():
        assert model.lm.enabled
    try:
        with guard.base_context():
            assert not model.lm.enabled and not model.lm.parameter.requires_grad
            guard._check(disabled=True)
            if fail:
                raise RuntimeError("helper decoder failed")
    except RuntimeError:
        assert fail
    assert model.lm.enabled and not model.lm.parameter.requires_grad
    assert original.metadata is metadata and model.action_adapter is guard
    guard.check(full=True)
    with pytest.raises(ValueError, match="once"):
        driver.CurrentNF4HelperGuard(model)


def test_bridge_refuses_a_guard_that_does_not_actually_disable_peft():
    model = model_fixture(refuses_disable=True)
    guard = driver.CurrentNF4HelperGuard(model)
    with pytest.raises(ValueError, match="must be disabled"):
        with guard.base_context():
            pytest.fail("Enabled adapters reached helper generation")
    assert model.lm.enabled and not model.lm.parameter.requires_grad


@pytest.mark.parametrize("change", ["head", "version", "temperature", "metadata"])
def test_bridge_latches_runtime_identity_change(change):
    model = model_fixture()
    guard = driver.CurrentNF4HelperGuard(model)
    if change == "head":
        model.head = SimpleNamespace(weight=SimpleNamespace(_version=0))
    elif change == "version":
        model.head.weight._version += 1
    elif change == "temperature":
        model.temperature = 1.0
    else:
        guard.original.metadata = dict(guard.metadata)
    with pytest.raises(ValueError, match="identity changed"):
        guard.check(full=True)


def test_resource_formula_keeps_full_bf16_cost_and_latches_refusal(tmp_path):
    total = 130_595_860_480
    reserve = (total + 4) // 5
    available = [reserve + 2 * driver.BASE_BYTES, reserve + 2 * driver.BASE_BYTES - 1]
    gate = driver.ResourceJournal(
        tmp_path / "resources.jsonl",
        "boot",
        observe=lambda: {
            "boot_id": "boot",
            "MemTotal": total,
            "MemAvailable": available.pop(0),
        },
    )
    row = gate.check("constructor")
    assert row["required_bytes"] == 73_958_270_912
    assert row["swap_or_resident_credit_bytes"] == 0
    with pytest.raises(ValueError, match="floor refused"):
        gate.check("native-forward")
    with pytest.raises(ValueError, match="latched"):
        gate.check("retry")
    rows = [json.loads(line) for line in gate.path.read_text().splitlines()]
    assert [row["admitted"] for row in rows] == [True, False]


def test_native_gate_uses_actual_backbone_not_peft_forward():
    seen = []
    hooks = []
    model = SimpleNamespace(
        lm=SimpleNamespace(
            register_forward_pre_hook=lambda *_a, **_k: pytest.fail(
                "PEFT.generate can bypass this wrapper"
            )
        ),
        backbone=SimpleNamespace(
            register_forward_pre_hook=lambda hook, **kwargs: (
                hooks.append((hook, kwargs)) or "actual-hook"
            )
        ),
    )
    gate = SimpleNamespace(check=lambda phase: seen.append(phase) or {"admitted": True})
    assert driver.register_native_gate(model, gate) == "actual-hook"
    hook, kwargs = hooks[0]
    assert kwargs == {"with_kwargs": True} and hook(None, (), {}) is None
    assert seen == ["before-native-forward"]


@pytest.mark.parametrize("mode", driver.FORMATS)
def test_default_dry_does_not_verify_tensors_construct_models_or_create_output(
    tmp_path, monkeypatch, capsys, mode
):
    from s1.browser_action_prompt import prompt_contract

    adapter, upstream = tmp_path / "fit/adapter", tmp_path / "upstream"
    adapter.mkdir(parents=True)
    upstream.mkdir()
    metadata = {"kind": "jev_action_lora_v1"}
    if mode == "shared-elements-v2":
        metadata = {"kind": "jev_native_action_lora_v2", "prompt_contract": prompt_contract()}
    (adapter / "manifest.json").write_text(json.dumps(metadata))
    monkeypatch.setattr(driver, "verify_adapter", lambda *_a: pytest.fail("Dry verified tensors"))
    monkeypatch.setattr(driver, "construct_model", lambda: pytest.fail("Dry loaded model"))
    output = tmp_path / "browser"
    assert (
        driver.main(
            [
                "--action-adapter",
                str(adapter),
                "--output-dir",
                str(output),
                "--upstream-path",
                str(upstream),
                "--prompt-format",
                mode,
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert plan["case_ids"] == list(driver.KNOWN4 + driver.PUBLIC6)
    assert plan["metadata_authenticated"] is False and plan["model_execution"] is False
    assert not output.exists()


def test_mismatched_format_refuses_before_any_verifier_or_output(tmp_path, monkeypatch):
    adapter, upstream = tmp_path / "fit/adapter", tmp_path / "upstream"
    adapter.mkdir(parents=True)
    upstream.mkdir()
    (adapter / "manifest.json").write_text('{"kind":"jev_action_lora_v1"}')
    monkeypatch.setattr(
        driver, "verify_adapter", lambda *_a: pytest.fail("Wrong mode reached verifier")
    )
    with pytest.raises(ValueError, match="differs"):
        driver.metadata_plan(adapter, tmp_path / "browser", upstream, "shared-elements-v2")
    assert not (tmp_path / "browser").exists()


def test_adapter_verifier_receives_explicit_trusted_mode_and_nf4(monkeypatch):
    from s1 import browser_action_adapter

    calls = []

    def verify(path, **kwargs):
        calls.append((path, kwargs))
        return SimpleNamespace(telemetry=lambda: {"kind": "jev_action_lora_v1"})

    monkeypatch.setattr(browser_action_adapter, "verify_action_adapter", verify)
    driver.verify_adapter("parent/adapter", "raw-json-v1")
    assert calls == [
        (
            "parent/adapter",
            {
                "model_id": driver.BASE_ID,
                "revision": driver.REVISION,
                "quantization": "nf4",
                "expected_prompt_format": "raw-json-v1",
            },
        )
    ]


def test_all_ten_existing_v7_cases_run_in_separate_unchanged_bundles(tmp_path, monkeypatch):
    calls = []

    def client(argv, env, output):
        calls.append((argv, env))
        ids = argv[argv.index("--public-cases") + 1 :]
        assert len(ids) in {4, 6} and "private-test-token" not in argv
        assert env["TEXT_MODEL_REASONING"] == "none"
        assert env["S1_POLICY_API_KEY"] == env["S1_TEXT_API_KEY"] == "private-test-token"
        results = Path(output) / "results"
        results.mkdir()
        (results / "summary.json").write_text(json.dumps({"attempts": len(ids), "successes": 0}))

    monkeypatch.setattr(driver, "run_client", client)
    groups = driver.run_groups(
        tmp_path,
        "unused-upstream",
        "http://127.0.0.1:12345",
        "private-test-token",
        "unused-chromium",
    )
    assert [g["denominator"] for g in groups] == [4, 6]
    assert [c[0][c[0].index("--public-cases") + 1 :] for c in calls] == [
        list(driver.KNOWN4),
        list(driver.PUBLIC6),
    ]
    for argv, _env in calls:
        assert argv[argv.index("--max-steps") + 1] == "16"
        assert argv[argv.index("--seed") + 1] == "20261014"
        assert argv[argv.index("--observation-profile-helper-sha256") + 1] == driver.V7_SHA
    assert "jev_heldout_public_cases" in (tmp_path / "public6/runtime-bundle/runner.py").read_text()
    assert (
        "from jev_public_web_cases import"
        in (tmp_path / "known4/runtime-bundle/runner.py").read_text()
    )


def helper_payload():
    context = {
        "goal": 'Search the exact phrase "test phrase"',
        "field": {"label": "Search", "role": "textbox", "value": ""},
        "page": {"title": "Search", "text": "Search"},
        "recent_actions": [],
    }
    return {
        "model": driver.IDENTITY,
        "max_tokens": 1024,
        "response_format": {"type": "json_object"},
        "reasoning": {"enabled": False},
        "messages": [
            {"role": "system", "content": "original system"},
            {"role": "user", "content": json.dumps(context)},
        ],
    }


@pytest.mark.parametrize("fail", [False, True])
def test_actual_api_helper_contract_raw_evidence_and_adapter_isolation(tmp_path, monkeypatch, fail):
    from fastapi.testclient import TestClient

    from s1.browser_server_evidence import ServerEvidenceJournal

    helper = importlib.import_module("jev_field_syntax_helper")
    model = model_fixture()
    guard = driver.CurrentNF4HelperGuard(model)
    prepared = SimpleNamespace(guard=guard, model=model)
    journal = ServerEvidenceJournal(tmp_path / "server")
    gate = driver.ResourceJournal(
        tmp_path / "resource.jsonl",
        "boot",
        observe=lambda: {
            "boot_id": "boot",
            "MemTotal": 130_595_860_480,
            "MemAvailable": 118_000_000_000,
        },
    )
    calls = []

    def generate(actual, context, *, evidence_sink, started):
        assert actual is prepared and started <= driver.time.monotonic()
        with actual.guard.base_context():
            assert not model.lm.enabled
            calls.append(context)
            generation = {
                "success": not fail,
                "response_authorized": not fail,
                "raw_output": '{"text":"\\"test phrase\\""}',
                "raw_token_ids": [4, 1],
                "usage": {"input_tokens": 100, "output_tokens": 2},
            }
            evidence_sink({"sequence": 0, "event": "fake-native", "generation": generation})
            if fail:
                raise helper.FieldJSONError("decode failed", generation)
            return '"test phrase"', generation

    monkeypatch.setattr(helper, "generate_field_text", generate)
    app = driver.make_application(
        model,
        prepared,
        journal,
        gate,
        tmp_path,
        "private-test-token",
        "raw-json-v1",
        "original system",
    )
    with TestClient(app) as client:
        response = client.post(
            "/chat/completions",
            json=helper_payload(),
            headers={"Authorization": "Bearer private-test-token"},
        )
    assert response.status_code == (502 if fail else 200)
    assert model.lm.enabled and not model.lm.parameter.requires_grad and len(calls) == 1
    if not fail:
        assert json.loads(response.json()["choices"][0]["message"]["content"]) == {
            "text": '"test phrase"'
        }
    assert (tmp_path / "http/0000/response.bin").read_bytes() == response.content
    assert json.loads((tmp_path / "http/0000/request.bin").read_bytes()) == helper_payload()
    snapshot = journal.snapshot()
    assert snapshot["all_requests_sealed"] and not snapshot["pending_request_ids"]
    saved = next((tmp_path / "server").glob("*.jsonl")).read_text()
    assert "raw_token_ids" in saved and "private-test-token" not in saved


def test_invalid_helper_wire_never_enters_generation(tmp_path, monkeypatch):
    payload = helper_payload()
    payload["reasoning"] = {"enabled": 0}  # False and integer zero cannot compare as equal.
    with pytest.raises(ValueError, match="contract"):
        driver.helper_context(payload, "original system")


@pytest.mark.parametrize("fail_evidence", [False, True])
def test_policy_response_has_http_api_join_and_evidence_failure_cannot_authorize_action(
    tmp_path, monkeypatch, fail_evidence
):
    from fastapi.testclient import TestClient

    from s1 import jev_generated
    from s1.browser_server_evidence import ServerEvidenceJournal

    model = model_fixture()
    driver.CurrentNF4HelperGuard(model)
    saved = ServerEvidenceJournal(tmp_path / "server")

    def journal(event):
        if fail_evidence and event["event"] == "server_returned":
            raise OSError("disk refused terminal policy evidence")
        saved(event)

    gate = driver.ResourceJournal(
        tmp_path / "resources.jsonl",
        "boot",
        observe=lambda: {
            "boot_id": "boot",
            "MemTotal": 130_595_860_480,
            "MemAvailable": 118_000_000_000,
        },
    )
    expected = {"model": driver.IDENTITY, "answers": {"operation": {"choice": "DONE"}}}

    def predict(_self, payload, *, evidence_sink):
        assert payload == {"model": driver.IDENTITY}
        evidence_sink({"event": "fake-model-returned", "raw_output": "DONE"})
        return expected

    monkeypatch.setattr(jev_generated.JevGeneratedActionAdapter, "predict", predict)
    app = driver.make_application(
        model, None, journal, gate, tmp_path, "private-test-token", "raw-json-v1", "original system"
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/systemone",
            json={"model": driver.IDENTITY},
            headers={"Authorization": "Bearer private-test-token"},
        )
    assert response.status_code == (502 if fail_evidence else 200)
    if not fail_evidence:
        assert response.json() == expected
    exchange = json.loads((tmp_path / "http/0000/exchange.json").read_text())
    assert exchange["api_request_id"] == response.headers["X-Request-ID"]
    assert exchange["client_delivery"].startswith("unknown")
    assert (tmp_path / "http/0000/response.bin").read_bytes() == response.content
    assert json.loads((tmp_path / "http/0000/candidate-response.bin").read_bytes()) == expected


def test_prepare_helper_binds_current_wrapper_source_without_claiming_old_qualification(
    tmp_path, monkeypatch
):
    import jev_field_syntax_helper as helper

    cache = tmp_path / "cache"
    cache.mkdir()
    manifest = cache / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"tokenizer_sha256": "t", "generation_config_sha256": "g", "source_sha256": "grammar"}
        )
    )
    monkeypatch.setattr(driver, "CACHE", cache)
    monkeypatch.setattr(driver, "CACHE_SHA", driver.pin(manifest)["sha256"])
    seen = {}
    monkeypatch.setattr(
        helper, "prepare_helper", lambda model, **kwargs: seen.update(kwargs) or model
    )
    model = object()
    assert driver.prepare_helper(model) is model
    assert seen["expected_guard_source_sha256"] == driver.pin(driver.__file__)["sha256"]
    assert seen["expected_helper_source_sha256"] == driver.pin(helper.__file__)["sha256"]


@pytest.mark.parametrize("interrupted", [False, True])
@pytest.mark.parametrize("cleanup_failure", [None, "join", "receipt", "index"])
def test_campaign_cleanup_preserves_primary_and_failure_receipt(
    tmp_path, monkeypatch, interrupted, cleanup_failure
):
    from s1 import browser_action_adapter

    adapter, upstream, output = tmp_path / "fit/adapter", tmp_path / "upstream", tmp_path / "run"
    adapter.mkdir(parents=True)
    (upstream / "jev_ultrafast").mkdir(parents=True)
    (adapter / "manifest.json").write_text('{"kind":"jev_action_lora_v1"}')
    (upstream / "jev_ultrafast/questions.py").write_text('TEXT_VALUE = "original system"\n')
    plan = driver.metadata_plan(adapter, output, upstream, "raw-json-v1")
    verified = SimpleNamespace(
        manifest_sha256=plan["adapter_manifest"]["sha256"],
        receipt_sha256="receipt",
        files={"weights": b"fake"},
    )
    monkeypatch.setattr(driver, "verify_adapter", lambda *_a: verified)
    model = model_fixture()
    browser_error = RuntimeError("public6 observer interrupted")
    cleanup_error = OSError("fake " + str(cleanup_failure) + " failure")
    removed = []
    model.backbone.register_forward_pre_hook = lambda *_a, **_k: SimpleNamespace(
        remove=lambda: removed.append(True)
    )
    constructors = []
    monkeypatch.setattr(driver, "construct_model", lambda: constructors.append(True) or model)
    monkeypatch.setattr(browser_action_adapter, "attach_action_adapter", lambda *_a: {"fake": True})
    monkeypatch.setattr(
        driver, "prepare_helper", lambda _m: SimpleNamespace(receipt=lambda: {"fake": True})
    )
    monkeypatch.setattr(driver, "make_application", lambda *_a: object())
    monkeypatch.setattr(
        driver.ResourceJournal,
        "snapshot",
        staticmethod(
            lambda: {
                "boot_id": "boot",
                "MemTotal": 130_595_860_480,
                "MemAvailable": 118_000_000_000,
            }
        ),
    )
    monkeypatch.setattr(driver.os, "environ", dict(driver.os.environ))

    class Listener:
        def bind(self, addr):
            assert addr == ("127.0.0.1", 0)

        def getsockname(self):
            return ("127.0.0.1", 12345)

    monkeypatch.setattr(driver.socket, "socket", Listener)

    class Server:
        def __init__(self, _config):
            self.started, self.should_exit = False, False

        def run(self, **_kwargs):
            self.started = True

    monkeypatch.setitem(
        sys.modules, "uvicorn", SimpleNamespace(Server=Server, Config=lambda *_a, **_k: None)
    )

    class Thread:
        def __init__(self, *, target, kwargs, daemon):
            self.target, self.kwargs, self.live = target, kwargs, False

        def start(self):
            self.live = True
            self.target(**self.kwargs)

        def is_alive(self):
            return self.live

        def join(self, **_kwargs):
            if cleanup_failure == "join":
                raise cleanup_error
            self.live = False

    monkeypatch.setattr(driver.threading, "Thread", Thread)

    def groups(*_a, completed):
        completed({"group": "known4", "summary": {"attempts": 4, "successes": 0}})
        if interrupted:
            raise browser_error
        completed({"group": "public6", "summary": {"attempts": 6, "successes": 0}})

    monkeypatch.setattr(driver, "run_groups", groups)
    if cleanup_failure == "receipt":
        save = driver.write_final_receipt
        attempts = []

        def fail_once(*args):
            attempts.append(True)
            if len(attempts) == 1:
                raise cleanup_error
            return save(*args)

        monkeypatch.setattr(driver, "write_final_receipt", fail_once)
    elif cleanup_failure == "index":
        save = driver.write

        def fail_index(path, *args, **kwargs):
            if Path(path).name == "output-index.json":
                raise cleanup_error
            return save(path, *args, **kwargs)

        monkeypatch.setattr(driver, "write", fail_index)
    if interrupted or cleanup_failure:
        with pytest.raises(BaseException) as caught:
            driver.run(plan, expected_boot_id="boot", chromium="unused")
        assert caught.value is (browser_error if interrupted else cleanup_error)
        if interrupted and cleanup_failure:
            assert any(str(cleanup_error) in note for note in caught.value.__notes__)
    else:
        driver.run(plan, expected_boot_id="boot", chromium="unused")
    receipt = json.loads((output / "receipt.json").read_text())
    assert len(constructors) == 1 and removed == ([] if cleanup_failure == "join" else [True])
    assert receipt["completed_group_attempts"] == (4 if interrupted else 10)
    assert receipt["planned_denominator"] == 10
    assert receipt["process_exit_confirmed"] is False and receipt["outer_closure_required"] is True
    assert receipt["stage"] == (
        "failed" if interrupted or cleanup_failure else "complete_pending_external_closure"
    )
    assert receipt["metadata_authenticated"] and receipt["model_load_started"]
    if cleanup_failure:
        assert receipt["cleanup_errors"] and not (output / "output-index.json").exists()
        if cleanup_failure == "join":
            assert receipt["server_thread_drained"] is False
    else:
        index = json.loads((output / "output-index.json").read_text())
        assert index["files"]["receipt.json"] == driver.pin(output / "receipt.json")
        assert not receipt["cleanup_errors"]


@pytest.mark.parametrize("failure", ["stop", "publication", "both"])
def test_client_cleanup_and_publication_cannot_replace_browser_timeout(
    tmp_path, monkeypatch, failure
):
    browser_error = TimeoutError("actual first browser timeout")
    cleanup_error = OSError("stop_owned failed")
    publication_error = OSError("closure write failed")

    def poll():
        raise browser_error

    child = SimpleNamespace(pid=123, returncode=None, poll=poll)

    def stop(*_args):
        if failure in {"stop", "both"}:
            raise cleanup_error
        return {"confirmed": True}

    monkeypatch.setattr(
        driver,
        "matrix_common",
        lambda: SimpleNamespace(
            process_table=lambda: {
                123: {"pid": 123, "ppid": os.getpid(), "pgid": 123, "sid": 123, "start_ticks": 1}
            },
            extend_owned=lambda *_a: None,
            stop_owned=stop,
        ),
    )
    monkeypatch.setattr(driver.subprocess, "Popen", lambda *_a, **_k: child)
    if failure in {"publication", "both"}:

        def fail_write(*_args, **_kwargs):
            raise publication_error

        monkeypatch.setattr(driver, "write", fail_write)
    with pytest.raises(TimeoutError) as caught:
        driver.run_client(["fake-only"], {}, tmp_path)
    assert caught.value is browser_error
    notes = " ".join(caught.value.__notes__)
    if failure in {"stop", "both"}:
        assert str(cleanup_error) in notes
    if failure in {"publication", "both"}:
        assert str(publication_error) in notes
    if failure == "stop":
        closure = json.loads((tmp_path / "owned-client-closure.json").read_text())
        assert closure["confirmed"] is False


@pytest.mark.parametrize("code", ["pass", "import time; time.sleep(0.15)"])
def test_actual_owned_client_is_seeded_before_poll_and_closes(tmp_path, code):
    driver.run_client([sys.executable, "-I", "-B", "-c", code], dict(os.environ), tmp_path)
    closure = json.loads((tmp_path / "owned-client-closure.json").read_bytes())
    assert closure["confirmed"] is True and closure["remaining"] == []
    assert len(closure["tracked_PID_start_ticks"]) == 1
    pid, start = next(iter(closure["tracked_PID_start_ticks"].items()))
    assert int(pid) > 0 and start > 0 and closure["tracked_groups"] == [int(pid)]


@pytest.mark.parametrize("field", ["ppid", "pgid", "sid", "start_ticks"])
def test_wrong_initial_client_identity_never_becomes_owned(tmp_path, monkeypatch, field):
    row = {"pid": 123, "ppid": os.getpid(), "pgid": 123, "sid": 123, "start_ticks": 1}
    row[field] = 0
    child = SimpleNamespace(pid=123, returncode=None)
    observed = []

    def stop(_child, known, groups):
        observed.append((dict(known), set(groups)))
        return {"confirmed": False}

    monkeypatch.setattr(driver.subprocess, "Popen", lambda *_a, **_k: child)
    monkeypatch.setattr(
        driver,
        "matrix_common",
        lambda: SimpleNamespace(process_table=lambda: {123: row}, stop_owned=stop),
    )
    with pytest.raises(ValueError, match="Actual owned browser child"):
        driver.run_client(["fake-only"], {}, tmp_path)
    assert observed == [({}, set())]


@pytest.mark.parametrize(
    "timeout_phase", ["after_serial_lock", "before_dispatch", "after_candidate"]
)
def test_whole_http_timeout_returns_504_and_never_authorizes_action(
    tmp_path, monkeypatch, timeout_phase
):
    from fastapi.testclient import TestClient

    from s1 import jev_generated
    from s1.browser_server_evidence import ServerEvidenceJournal

    model = model_fixture()
    driver.CurrentNF4HelperGuard(model)
    journal = ServerEvidenceJournal(tmp_path / "server")
    gate = driver.ResourceJournal(
        tmp_path / "resources.jsonl",
        "boot",
        observe=lambda: {
            "boot_id": "boot",
            "MemTotal": 130_595_860_480,
            "MemAvailable": 118_000_000_000,
        },
    )
    calls = []
    candidate = {"model": driver.IDENTITY, "answers": {"operation": {"choice": "DONE"}}}

    def predict(_self, _payload, *, evidence_sink):
        calls.append(True)
        return candidate

    monkeypatch.setattr(jev_generated.JevGeneratedActionAdapter, "predict", predict)
    checks = []
    deadline = driver.http_deadline

    def measured(started, **_kwargs):
        checks.append(True)
        expires_at = {"after_serial_lock": 1, "before_dispatch": 2, "after_candidate": 3}
        expired = len(checks) >= expires_at[timeout_phase]
        deadline(started, clock=lambda: started + (20 if expired else 0))

    monkeypatch.setattr(driver, "http_deadline", measured)
    app = driver.make_application(
        model, None, journal, gate, tmp_path, "private-test-token", "raw-json-v1", "original system"
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/systemone",
            json={"model": driver.IDENTITY},
            headers={"Authorization": "Bearer private-test-token"},
        )
    assert response.status_code == 504
    assert response.json()["detail"]["type"] == "TimeoutError" and "answers" not in response.json()
    assert calls == ([True] if timeout_phase == "after_candidate" else [])
    exchange = json.loads((tmp_path / "http/0000/exchange.json").read_text())
    assert exchange["status"] == 504 and exchange["error"]["type"] == "TimeoutError"
    assert (tmp_path / "http/0000/response.bin").read_bytes() == response.content
    if timeout_phase == "after_candidate":
        assert json.loads((tmp_path / "http/0000/candidate-response.bin").read_bytes()) == candidate
    else:
        assert not (tmp_path / "http/0000/candidate-response.bin").exists()
        assert not journal.snapshot()["requests"]


class PublicationClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def publication_application(tmp_path, monkeypatch, clock, *, primary=None):
    """Real middleware/ASGI, fake policy; the framework's own clock is untouched."""
    from s1 import jev_generated
    from s1.browser_server_evidence import ServerEvidenceJournal

    model = model_fixture()
    driver.CurrentNF4HelperGuard(model)
    expected = {"model": driver.IDENTITY, "answers": {"operation": {"choice": "DONE"}}}
    monkeypatch.setattr(
        jev_generated.JevGeneratedActionAdapter,
        "predict",
        lambda _self, _payload, *, evidence_sink: expected,
    )

    def check(phase):
        if primary is not None and phase == "before-http-request":
            raise primary

    app = driver.make_application(
        model,
        None,
        ServerEvidenceJournal(tmp_path / "server"),
        SimpleNamespace(check=check),
        tmp_path,
        "private-test-token",
        "raw-json-v1",
        "original system",
        clock=clock,
    )
    return app, expected


def post_policy(app):
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        return client.post(
            "/v1/systemone",
            json={"model": driver.IDENTITY},
            headers={"Authorization": "Bearer private-test-token"},
        )


@pytest.mark.parametrize("phase", ["response_fsync", "response_pin", "exchange_fsync"])
@pytest.mark.parametrize("elapsed,status", [(19.999, 200), (20.0, 504), (26.0, 504)])
def test_actual_response_start_charges_all_final_publication(
    tmp_path, monkeypatch, phase, elapsed, status
):
    clock = PublicationClock()
    app, candidate = publication_application(tmp_path, monkeypatch, clock)
    original_write, original_pin = driver.write, driver.pin
    delays = []

    def slow_write(path, value, *, raw=False):
        original_write(path, value, raw=raw)
        selected = (phase == "response_fsync" and Path(path).name.startswith(".response.bin.")) or (
            phase == "exchange_fsync" and Path(path).name.startswith(".exchange.json.")
        )
        if selected and not delays:
            delays.append(phase)
            clock.now = elapsed

    def slow_pin(path):
        result = original_pin(path)
        if phase == "response_pin" and Path(path).name == "response.bin" and not delays:
            delays.append(phase)
            clock.now = elapsed
        return result

    monkeypatch.setattr(driver, "write", slow_write)
    monkeypatch.setattr(driver, "pin", slow_pin)
    response = post_policy(app)
    assert delays == [phase] and response.status_code == status
    folder = tmp_path / "http/0000"
    assert json.loads((folder / "candidate-response.bin").read_bytes()) == candidate
    assert (folder / "response.bin").read_bytes() == response.content
    record = json.loads((folder / "exchange.json").read_bytes())
    assert record["status"] == status
    assert record["response"] == original_pin(folder / "response.bin")
    assert record["request"] == original_pin(folder / "request.bin")
    assert record["api_request_id"] == response.headers["X-Request-ID"]
    assert record["deadline_gate"] == "after_all_publication_at_ASGI_http.response.start"
    assert record["elapsed_seconds_scope"] == "after_response_binding_before_this_exchange_fsync"
    assert int(response.headers["Content-Length"]) == len(response.content)
    assert response.headers["Content-Type"] == "application/json"
    if status == 504:
        assert record["elapsed_seconds"] >= 20
        assert response.json()["detail"]["type"] == "TimeoutError"
        assert record["error"] == response.json()["detail"] and "answers" not in response.json()
    else:
        assert response.json() == candidate
    assert not list(folder.glob(".*.pending"))
    assert (folder / "response.bin").stat().st_mode & 0o777 == 0o600
    assert (folder / "exchange.json").stat().st_mode & 0o777 == 0o600


def test_headers_are_built_before_last_successful_deadline_gate(tmp_path, monkeypatch):
    from starlette.responses import Response

    clock = PublicationClock()
    app, candidate = publication_application(tmp_path, monkeypatch, clock)
    init_headers = Response.init_headers
    header_work = []

    def slow_headers(response, headers=None):
        init_headers(response, headers)
        if (
            type(response).__name__ == "PublishedResponse"
            and (tmp_path / "http/0000/exchange.json").exists()
        ):
            header_work.append(response.status_code)
            clock.now = 20

    monkeypatch.setattr(Response, "init_headers", slow_headers)
    response = post_policy(app)
    assert header_work == [200, 504]
    assert response.status_code == 504
    assert int(response.headers["Content-Length"]) == len(response.content)
    folder = tmp_path / "http/0000"
    assert json.loads((folder / "candidate-response.bin").read_bytes()) == candidate
    assert (folder / "response.bin").read_bytes() == response.content


def test_event_loop_delay_before_response_start_is_charged(tmp_path, monkeypatch):
    from starlette.responses import Response

    clock = PublicationClock()
    app, _candidate = publication_application(tmp_path, monkeypatch, clock)
    original_call = Response.__call__

    async def delayed_call(response, scope, receive, send):
        if type(response).__name__ == "PublishedResponse":
            clock.now = 20  # After dispatch returned, before actual ASGI start.
        await original_call(response, scope, receive, send)

    monkeypatch.setattr(Response, "__call__", delayed_call)
    response = post_policy(app)
    assert response.status_code == 504
    record = json.loads((tmp_path / "http/0000/exchange.json").read_bytes())
    assert record["status"] == 504 and record["elapsed_seconds"] >= 20


@pytest.mark.parametrize(
    "primary,status",
    [
        (ValueError("original resource refusal"), 502),
        (TimeoutError("original deadline refusal"), 504),
    ],
)
def test_late_failure_publication_preserves_existing_refusal(
    tmp_path, monkeypatch, primary, status
):
    clock = PublicationClock()
    app, _candidate = publication_application(tmp_path, monkeypatch, clock, primary=primary)
    original_write = driver.write

    def slow_write(path, value, *, raw=False):
        original_write(path, value, raw=raw)
        if Path(path).name.startswith(".exchange.json."):
            clock.now = 26

    monkeypatch.setattr(driver, "write", slow_write)
    response = post_policy(app)
    assert response.status_code == status
    assert response.json()["detail"] == {"type": type(primary).__name__, "message": str(primary)}
    folder = tmp_path / "http/0000"
    record = json.loads((folder / "exchange.json").read_bytes())
    assert record["status"] == status and record["error"] == response.json()["detail"]
    assert (folder / "response.bin").read_bytes() == response.content
    assert not (folder / "candidate-response.bin").exists()


class TraceProbe:
    """Observe restoration across TestClient's thread without touching its contextvars."""

    def __init__(self, *, fail_reset=False):
        self.value, self.fail_reset, self.resets = None, fail_reset, 0

    def get(self):
        return self.value

    def set(self, value):
        previous, self.value = self.value, value
        return previous

    def reset(self, previous):
        self.resets += 1
        if self.fail_reset:
            raise OSError("TRACE reset refused")
        self.value = previous


@pytest.mark.parametrize("name", ["response.bin", "exchange.json"])
@pytest.mark.parametrize("has_primary", [False, True])
def test_publication_error_never_masks_request_error_or_leaks_trace(
    tmp_path, monkeypatch, name, has_primary
):
    clock = PublicationClock()
    primary = TimeoutError("original timeout") if has_primary else None
    app, candidate = publication_application(tmp_path, monkeypatch, clock, primary=primary)
    trace = TraceProbe()
    monkeypatch.setattr(driver, "TRACE", trace)
    publication_error = OSError("terminal fsync refused")
    replace = driver._replace_http_evidence

    def fail(directory, target, value, *, raw=False):
        if target == name:
            raise publication_error
        replace(directory, target, value, raw=raw)

    monkeypatch.setattr(driver, "_replace_http_evidence", fail)
    expected = primary if primary is not None else publication_error
    with pytest.raises(type(expected), match=str(expected)) as caught:
        post_policy(app)
    assert caught.value is expected
    if primary is not None:
        assert "terminal fsync refused" in " ".join(primary.__notes__)
    else:
        assert json.loads((tmp_path / "http/0000/candidate-response.bin").read_bytes()) == candidate
    assert trace.get() is None and trace.resets == 1


def test_late_candidate_refusal_write_preserves_primary_timeout(tmp_path, monkeypatch):
    clock = PublicationClock()
    app, candidate = publication_application(tmp_path, monkeypatch, clock)
    replace = driver._replace_http_evidence
    count = 0

    def fail_refusal(directory, name, value, *, raw=False):
        nonlocal count
        if name == "exchange.json":
            clock.now = 20
        if name == "response.bin":
            count += 1
            if count == 2:
                raise OSError("late refusal evidence unavailable")
        replace(directory, name, value, raw=raw)

    monkeypatch.setattr(driver, "_replace_http_evidence", fail_refusal)
    with pytest.raises(TimeoutError, match="Whole HTTP") as caught:
        post_policy(app)
    assert "late refusal evidence unavailable" in " ".join(caught.value.__notes__)
    assert json.loads((tmp_path / "http/0000/candidate-response.bin").read_bytes()) == candidate


@pytest.mark.parametrize("has_primary", [False, True])
def test_trace_reset_failure_restores_previous_context_and_preserves_primary(
    tmp_path, monkeypatch, has_primary
):
    clock = PublicationClock()
    primary = ValueError("original gate failure") if has_primary else None
    app, _candidate = publication_application(tmp_path, monkeypatch, clock, primary=primary)
    trace = TraceProbe(fail_reset=True)
    previous = {"outer": "context"}
    trace.set(previous)
    monkeypatch.setattr(driver, "TRACE", trace)
    with pytest.raises(type(primary) if has_primary else OSError) as caught:
        post_policy(app)
    if primary is not None:
        assert caught.value is primary
        assert "TRACE reset refused" in " ".join(primary.__notes__)
    else:
        assert str(caught.value) == "TRACE reset refused"
    assert trace.get() is previous and trace.resets == 1


def test_atomic_http_writer_is_confined_private_and_cleans_failed_temporary(tmp_path, monkeypatch):
    folder = tmp_path / "http/0000"
    with pytest.raises(ValueError, match="terminal evidence"):
        driver._replace_http_evidence(folder, "../escaped.bin", b"no", raw=True)
    assert not folder.exists()
    driver._replace_http_evidence(folder, "response.bin", b"original", raw=True)
    original_write = driver.write

    def fail_after_fsync(path, value, *, raw=False):
        original_write(path, value, raw=raw)
        raise OSError("temporary write failed")

    monkeypatch.setattr(driver, "write", fail_after_fsync)
    with pytest.raises(OSError, match="temporary write failed"):
        driver._replace_http_evidence(folder, "response.bin", b"replacement", raw=True)
    assert (folder / "response.bin").read_bytes() == b"original"
    assert (folder / "response.bin").stat().st_mode & 0o777 == 0o600
    assert not list(folder.glob(".*.pending"))


@pytest.mark.parametrize("alias", ["directory", "target"])
def test_atomic_http_writer_refuses_symlink_escapes(tmp_path, alias):
    outside = tmp_path / "outside"
    outside.mkdir()
    protected = outside / "response.bin"
    protected.write_bytes(b"unrelated")
    folder = tmp_path / "request"
    if alias == "directory":
        folder.symlink_to(outside, target_is_directory=True)
    else:
        folder.mkdir()
        (folder / "response.bin").symlink_to(protected)
    with pytest.raises(ValueError, match="Ordinary"):
        driver._replace_http_evidence(folder, "response.bin", b"must not escape", raw=True)
    assert protected.read_bytes() == b"unrelated"
    assert not list(outside.glob(".*.pending"))


@pytest.mark.parametrize("elapsed,status", [(0.0, 200), (20.0, 504)])
def test_concurrent_real_asgi_requests_keep_their_own_terminal_bindings(
    tmp_path, monkeypatch, elapsed, status
):
    import asyncio
    import threading

    import httpx

    from s1 import jev_generated

    clock = PublicationClock()
    app, _candidate = publication_application(tmp_path, monkeypatch, clock)
    publication_active = threading.Event()

    def predict(_self, payload, *, evidence_sink):
        assert not publication_active.is_set(), "Policy overlapped terminal publication"
        return {
            "model": driver.IDENTITY,
            "probe": payload["probe"],
            "answers": {"operation": {"choice": "DONE"}},
        }

    monkeypatch.setattr(jev_generated.JevGeneratedActionAdapter, "predict", predict)
    replace = driver._replace_http_evidence

    def publish(directory, name, value, *, raw=False):
        assert not publication_active.is_set()
        publication_active.set()
        try:
            replace(directory, name, value, raw=raw)
            clock.now = elapsed
        finally:
            publication_active.clear()

    monkeypatch.setattr(driver, "_replace_http_evidence", publish)

    async def requests():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            return await asyncio.gather(
                *[
                    client.post(
                        "/v1/systemone",
                        json={"model": driver.IDENTITY, "probe": value},
                        headers={"Authorization": "Bearer private-test-token"},
                    )
                    for value in ("alpha", "beta")
                ]
            )

    responses = asyncio.run(requests())
    assert [response.status_code for response in responses] == [status, status]
    assert len({response.headers["X-Request-ID"] for response in responses}) == 2
    exchanges = {
        json.loads(path.read_bytes())["api_request_id"]: path.parent
        for path in (tmp_path / "http").glob("*/exchange.json")
    }
    assert len(exchanges) == 2
    for expected, response in zip(("alpha", "beta"), responses, strict=True):
        folder = exchanges[response.headers["X-Request-ID"]]
        assert json.loads((folder / "request.bin").read_bytes())["probe"] == expected
        assert json.loads((folder / "candidate-response.bin").read_bytes())["probe"] == expected
        assert (folder / "response.bin").read_bytes() == response.content
        exchange = json.loads((folder / "exchange.json").read_bytes())
        assert exchange["status"] == status and exchange["response"] == driver.pin(
            folder / "response.bin"
        )
        assert int(response.headers["Content-Length"]) == len(response.content)
        if status == 200:
            assert response.json()["probe"] == expected
        else:
            assert "answers" not in response.json() and exchange["elapsed_seconds"] >= 20
