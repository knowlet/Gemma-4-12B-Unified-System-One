"""Service composition and owned loopback HTTP tests; no Modal or model load."""

import ast
import copy
import hashlib
import importlib.util
import io
import json
import socket
import struct
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from s1.evaluation.decision3 import decision3_spec
from s1.jev_decision3_native import JevDecision3NativeAdapter

ROOT = Path(__file__).resolve().parents[1]
SERVICE_PATH = ROOT / "apps/modal/decision3_browser_service.py"
WORKER_PATH = ROOT / "scripts/jev_decision3_browser_worker.py"


def load_source(name, path, *, before_modal=False):
    if before_modal:
        module = types.ModuleType(name)
        module.__file__ = str(path)
        tree = ast.parse(path.read_bytes())
        boundary = next(
            index
            for index, node in enumerate(tree.body)
            if isinstance(node, ast.Import) and any(alias.name == "modal" for alias in node.names)
        )
        exec(
            compile(ast.Module(body=tree.body[:boundary], type_ignores=[]), str(path), "exec"),
            module.__dict__,
        )
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SERVICE = load_source("browser_rpc_pure_service", SERVICE_PATH, before_modal=True)
WORKER = load_source("browser_rpc_pure_worker", WORKER_PATH)
WIRE = load_source("browser_rpc_pure_wire", ROOT / "scripts/jev_decision3_browser_wire.py")


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class NativeWithoutML:
    def __init__(self, clock, delay=0):
        self.spec = decision3_spec("d3-edge")
        self.name, self.clock, self.delay = self.spec.id, clock, delay
        self.calls = []

    def predict(self, request, *, deadline=None, evidence_sink=None):
        self.calls.append(copy.deepcopy(request.model_dump()))
        deadline()
        raw = {
            "model": "d3-edge",
            "answers": {},
            "usage": {"input_tokens": 123, "output_tokens": 0},
            "cost": {"usd": None},
        }
        for question in request.questions:
            labels = question.labels()
            choice = labels[0]
            raw["answers"][question.id] = {
                "type": "choice",
                "choice": choice,
                "confidence": 0.2,
                "probabilities": {
                    label: 0.75 if label == choice else 0.25 / (len(labels) - 1) for label in labels
                },
            }
        evidence_sink("native_raw_returned", {"raw_response": copy.deepcopy(raw)})
        self.clock.advance(self.delay)
        deadline()
        return raw


def payload():
    return {
        "model": "d3-edge",
        "state": {"z": "保留原始順序", "a": [3, 1, 2]},
        "questions": {
            "operation": {
                "type": "choice",
                "criteria": {"CLICK": "Click", "DONE": "Done", "BLOCKED": "Blocked"},
                "instructions": {"z": "First", "a": "Second"},
            },
            "click_target": {
                "type": "choice",
                "criteria": {"9": "Nine", "2": "Two"},
                "instructions": ["Original", "Order"],
            },
        },
    }


def ready_reply(model="d3-edge", *, nonce="nonce-test-only", pid=1001, gpu="GPU-test-only"):
    return {
        "success": True,
        "model": model,
        "nonce": nonce,
        "gpu": {"name": "H200", "uuid": gpu},
        "ready": {
            "event": "ready",
            "model": model,
            "alias": model,
            "nonce": nonce,
            "pid": pid,
            "gpu_uuid": gpu,
        },
    }


def closed_frame(
    model="d3-edge", *, success=True, nonce="nonce-test-only", pid=1001, gpu="GPU-test-only"
):
    error = (
        None
        if success
        else {"type": "TimeoutError", "message": "Actual request deadline exceeded", "notes": []}
    )
    receipt = {
        "profile": WORKER.PROFILE,
        "model": model,
        "alias": model,
        "nonce": nonce,
        "pid": pid,
        "gpu_uuid": gpu,
        "stage": "complete" if success else "failed",
        "success": success,
        "campaign_stop": not success,
        "error": error,
        "secondary_errors": [],
        "network_attempts_blocked": [],
        "final_resources": {"cuda": {"uuid": gpu}},
        "cache_before": {
            "full_payload_sha256_authenticated": True,
            "full_payload_sha256": "a" * 64,
        },
        "cache_after": {"full_payload_sha256_authenticated": True, "full_payload_sha256": "a" * 64},
        "requests": [],
    }
    if not success:
        request_id, digest = "failed-request-1", "b" * 64
        receipt["failed_request_response"] = {
            "event": "response",
            "model": model,
            "nonce": nonce,
            "gpu_uuid": gpu,
            "request_id": request_id,
            "http_sha256": digest,
            "success": False,
            "campaign_stop": True,
            "wire_body_utf8": None,
            "journal": {
                "request_id": request_id,
                "http_sha256": digest,
                "model": model,
                "nonce": nonce,
                "gpu_uuid": gpu,
                "error": copy.deepcopy(error),
                "raw_response": {"preserved": "native raw"},
            },
        }
        receipt["requests"] = [
            {
                "request_id": request_id,
                "http_sha256": digest,
                "success": False,
                "campaign_stop": True,
            }
        ]
    return {"event": "closed", "success": success, "campaign_stop": not success, "receipt": receipt}


def arm_reply(
    model="d3-edge",
    *,
    worker_success=True,
    arm_success=None,
    nonce="nonce-test-only",
    pid=1001,
    gpu="GPU-test-only",
):
    return {
        "schema_id": SERVICE.LIFECYCLE_SCHEMA,
        "model": model,
        "nonce": nonce,
        "gpu": {"name": "H200", "uuid": gpu},
        "owned_exit_confirmed": True,
        "returncode": 0 if worker_success else 1,
        "worker": closed_frame(model, success=worker_success, nonce=nonce, pid=pid, gpu=gpu),
        "success": worker_success if arm_success is None else arm_success,
        "continuation_authorized": True,
        "hard_halted": False,
        "lifecycle_errors": [],
    }


def session_harness(tmp_path, monkeypatch, *, failed_models=()):
    """Exercise real begin/predict/end/close with only process/GPU/storage boundaries fake."""
    session = object.__new__(SERVICE.Session)
    session.nonce = "nonce-test-only"
    session.gpu = {"name": "H200", "uuid": "GPU-test-only"}
    session.manifest = {"reviewed": "fake source inventory"}
    session.output = tmp_path / "cloud-output"
    session.output.mkdir(parents=True)
    session.mounts = {"/export": {"owned": "export-binding"}}
    session.sequence, session.next_arm, session.active = 0, 0, None
    session.blocked = session.halted = False
    session.lifecycle_errors, session.identity_errors, session.closures = [], [], []
    session.process = session.last_control = session.last_returncode = None
    session.protocol_history, session.observations = [], []
    records, launched, published = [], [], []

    def validate(blob):
        assert SERVICE.parse(blob) == session.manifest
        return copy.deepcopy(session.manifest)

    monkeypatch.setattr(SERVICE, "validate_inventory", validate)
    monkeypatch.setattr(SERVICE, "gpu_identity", lambda: copy.deepcopy(session.gpu))
    session.core = SimpleNamespace(
        write=lambda path, value, **_kwargs: SERVICE.save(path, value),
        resource_snapshot=lambda: {"boot_id": "owned-boot", "proc_meminfo": {"MemTotal": 1}},
    )

    def publish(path, binding, confirmed):
        published.append((path, copy.deepcopy(binding), confirmed))
        return {"sha256": "e" * 64, "size_bytes": 1}

    session.staging = SimpleNamespace(
        child_environment=lambda _kind: {},
        proc_record=lambda _path: {"start_ticks": 1},
        tree_inventory=lambda path: {p.name: SERVICE.pin(p) for p in path.iterdir() if p.is_file()},
        publish_output=publish,
    )

    def record(name, value):
        records.append((name, copy.deepcopy(value), session.next_arm))
        SERVICE.save(session.output / name, value)

    session.record = record

    def popen(command, **options):
        config = json.loads(Path(command[-1]).read_bytes())
        process = SimpleNamespace(
            pid=2000 + len(launched), stdin=io.BytesIO(), returncode=None, response_sent=False
        )
        process.poll = lambda: process.returncode
        process.config = config
        launched.append((process, options))
        return process

    monkeypatch.setattr(SERVICE.subprocess, "Popen", popen)

    def read_control():
        process = session.process
        model = process.config["model"]
        if session.active is None:
            value = ready_reply(model, pid=process.pid)["ready"]
        else:
            lines = process.stdin.getvalue().splitlines()
            envelope = SERVICE.parse(lines[-1]) if lines else {}
            if envelope.get("command") == "close" or process.response_sent:
                value = closed_frame(model, success=model not in failed_models, pid=process.pid)
                if process.response_sent:
                    response = process.last_response
                    value["receipt"]["requests"] = [
                        {
                            key: response[key]
                            for key in ("request_id", "http_sha256", "success", "campaign_stop")
                        }
                    ]
                    if not response["success"]:
                        value["receipt"]["failed_request_response"] = copy.deepcopy(response)
                        value["receipt"]["error"] = copy.deepcopy(response["journal"]["error"])
            else:
                value = {
                    "event": "response",
                    "request_id": envelope["request_id"],
                    "http_sha256": envelope["http_sha256"],
                    "nonce": session.nonce,
                    "model": model,
                    "gpu_uuid": session.gpu["uuid"],
                    "success": model not in failed_models,
                    "campaign_stop": model in failed_models,
                    "wire_body_utf8": None
                    if model in failed_models
                    else '{"model":"owned-fake","answers":{"operation":{"choice":"DONE"}}}',
                    "journal": {"raw_response": {"retained": model}},
                }
                if model in failed_models:
                    value["journal"].update(
                        request_id=envelope["request_id"],
                        http_sha256=envelope["http_sha256"],
                        model=model,
                        nonce=session.nonce,
                        gpu_uuid=session.gpu["uuid"],
                        error={
                            "type": "TimeoutError",
                            "message": "Actual request deadline exceeded",
                            "notes": [],
                        },
                    )
                process.last_response = copy.deepcopy(value)
                process.response_sent = True
                if model in failed_models:
                    process.returncode = 1
        session.last_control = copy.deepcopy(value)
        return value

    def reap():
        process = session.process
        session.last_returncode = 1 if process.config["model"] in failed_models else 0
        session.process = None
        return True

    session.read_control, session.reap = read_control, reap
    return session, records, launched, published


def composition(tmp_path, *, delay=0, after_rpc=0, wire=WIRE):
    clock = Clock()
    backend = NativeWithoutML(clock, delay)
    adapter = JevDecision3NativeAdapter(backend, model_identity="d3-edge")
    arm = tmp_path / "arm"
    arm.mkdir()
    calls, gates, fences = [], [], []
    torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda index: fences.append(index)))

    def rpc(nonce, request_id, body, remaining):
        calls.append(body)
        result = WORKER.process_request(
            SERVICE.encode(
                {
                    "request_id": request_id,
                    "payload_json": body.decode(),
                    "http_sha256": hashlib.sha256(body).hexdigest(),
                    "remaining_seconds": remaining,
                }
            ),
            adapter=adapter,
            wire=wire,
            torch=torch,
            gates=SimpleNamespace(check=lambda stage: gates.append(stage)),
            output=arm,
            model="d3-edge",
            nonce=nonce,
            gpu_uuid="GPU-test-only",
            clock=clock,
        )
        clock.advance(after_rpc)
        return result

    gateway = SERVICE.Gateway(
        rpc,
        tmp_path / "local",
        {"endpoint": "http://127.0.0.1:7", "qualification_sha256": "a" * 64},
        clock=clock,
    )
    gateway.accept_begin(ready_reply())
    return gateway, clock, backend, adapter, calls, gates, fences


def saved_request(tmp_path):
    paths = list((tmp_path / "local/requests").glob("*.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_bytes())


def test_original_ordered_http_bytes_reach_actual_legacy_adapter_and_reviewed_wire(tmp_path):
    gateway, _, backend, adapter, calls, gates, fences = composition(tmp_path)
    original = payload()
    body = json.dumps(original, ensure_ascii=False, indent=1).encode()
    status, response = gateway.request(body)
    assert status == 200 and calls == [body] and len(backend.calls) == 1
    result = json.loads(response)
    assert list(result["answers"]) == list(original["questions"])
    assert list(result["answers"]["click_target"]["probabilities"]) == ["9", "2"]
    assert result["decision3_native"]["original_request"] == original
    assert list(result["decision3_native"]["original_request"]["state"]) == ["z", "a"]
    assert (
        result["decision3_native"]["original_request_sha256"]
        == hashlib.sha256(SERVICE.encode(original)).hexdigest()
    )
    row = saved_request(tmp_path)
    assert row["raw_http_hex"] == body.hex() and row["gateway_action_authorized"] is True
    assert row["rpc"]["journal"]["raw_response"]["answers"]["operation"]["confidence"] == 0.2
    assert adapter.backend is backend and adapter.evidence_sink is None
    assert gates and fences == [0, 0]


@pytest.mark.parametrize("where", ["native", "rpc", "conversion"])
def test_all_late_paths_retain_raw_and_never_return_an_action(tmp_path, where):
    clock_ref = []

    def convert(request, response):
        result = WIRE.named_browser_response(request, response)
        clock_ref[0].advance(21)
        return result

    gateway, clock, backend, _, calls, _, _ = composition(
        tmp_path,
        delay=21 if where == "native" else 0,
        after_rpc=21 if where == "rpc" else 0,
        wire=SimpleNamespace(named_browser_response=convert) if where == "conversion" else WIRE,
    )
    clock_ref.append(clock)
    status, response = gateway.request(SERVICE.encode(payload()))
    assert status == 504 and "answers" not in json.loads(response) and gateway.blocked
    row = saved_request(tmp_path)
    assert row["rpc"]["journal"]["raw_returned"] is True
    assert row["gateway_action_authorized"] is False
    if where == "conversion":
        assert row["rpc"]["journal"]["wire_body_utf8"] is not None
        assert row["rpc"]["wire_body_utf8"] is None
    assert gateway.request(SERVICE.encode(payload()))[0] == 504
    assert len(calls) == len(backend.calls) == 1


def test_final_local_evidence_cost_is_inside_gateway_deadline(tmp_path, monkeypatch):
    gateway, clock, *_ = composition(tmp_path)
    original_save = SERVICE.save

    def slow_evidence(path, row):
        original_save(path, row)
        if row.get("gateway_action_authorized") is True:
            clock.advance(21)

    monkeypatch.setattr(SERVICE, "save", slow_evidence)
    status, response = gateway.request(SERVICE.encode(payload()))
    assert status == 504 and "answers" not in json.loads(response)
    row = saved_request(tmp_path)
    assert row["gateway_action_authorized"] is False and row["late_after_final_evidence"]


@pytest.mark.parametrize("fault", ["sha", "model", "duplicate_json", "budget"])
def test_worker_malformed_envelope_refuses_before_native_prediction(tmp_path, fault):
    clock = Clock()
    backend = NativeWithoutML(clock)
    adapter = JevDecision3NativeAdapter(backend, model_identity="d3-edge")
    body = SERVICE.encode(payload())
    envelope = {
        "request_id": "request-1",
        "payload_json": body.decode(),
        "http_sha256": hashlib.sha256(body).hexdigest(),
        "remaining_seconds": 20,
    }
    if fault == "sha":
        envelope["http_sha256"] = "0" * 64
    elif fault == "model":
        bad = {**payload(), "model": "d3"}
        envelope["payload_json"] = SERVICE.encode(bad).decode()
        envelope["http_sha256"] = hashlib.sha256(SERVICE.encode(bad)).hexdigest()
    elif fault == "duplicate_json":
        envelope["payload_json"] = '{"model":"d3-edge","model":"d3"}'
        envelope["http_sha256"] = hashlib.sha256(envelope["payload_json"].encode()).hexdigest()
    else:
        envelope["remaining_seconds"] = True
    result = WORKER.process_request(
        SERVICE.encode(envelope),
        adapter=adapter,
        wire=WIRE,
        torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda _index: None)),
        gates=SimpleNamespace(check=lambda stage: None),
        output=tmp_path,
        model="d3-edge",
        nonce="nonce",
        gpu_uuid="GPU-fake",
        clock=clock,
    )
    assert (
        result["success"] is False
        and result["campaign_stop"] is True
        and result["wire_body_utf8"] is None
    )
    assert backend.calls == []


def test_no_modals_or_framework_imports_for_source_only_preparation_and_exact17_inventory():
    before = set(sys.modules)
    manifest = SERVICE.source_inventory()
    assert manifest["files"] == len(manifest["rows"]) == 17
    assert len({row["path"] for row in manifest["rows"]}) == 17
    assert not manifest["collection_or_parent_uploaded"] and not manifest["weights_uploaded"]
    assert all(
        row["classification"] in {"reviewed_public_source12", "reviewed_service_composition_source"}
        for row in manifest["rows"]
    )
    assert not any(
        "admission" in row["path"] or "/public_collection/" in row["path"]
        for row in manifest["rows"]
    )
    assert "torch" not in set(sys.modules) - before and "modal" not in set(sys.modules) - before


def test_h200_rpc_configuration_has_no_public_endpoint_or_gemma_helper():
    tree = ast.parse(SERVICE_PATH.read_bytes())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PersistentWorker"
    )
    options = {keyword.arg: keyword.value for keyword in cls.decorator_list[0].keywords}
    for name, expected in {
        "gpu": "H200",
        "cpu": 2,
        "memory": 344064,
        "min_containers": 1,
        "max_containers": 1,
        "retries": 0,
        "include_source": False,
        "block_network": True,
    }.items():
        assert ast.literal_eval(options[name]) == expected
    assert [ast.literal_eval(key) for key in options["volumes"].keys] == ["/weights", "/export"]
    assert not any(
        isinstance(node, ast.Attribute)
        and node.attr in {"asgi_app", "web_endpoint", "fastapi_endpoint", "web_server"}
        for node in ast.walk(tree)
    )


def test_192_request_upper_bound_includes_full_known_and_heldout160(tmp_path):
    assert WORKER.MAX_REQUESTS == 192 and 4 * 16 + 6 * 16 < WORKER.MAX_REQUESTS
    directory = tmp_path / "requests"
    directory.mkdir()
    for index in range(WORKER.MAX_REQUESTS - 1):
        (directory / f"prior-{index}.json").write_text("{}")
    clock = Clock()
    backend = NativeWithoutML(clock)
    adapter = JevDecision3NativeAdapter(backend, model_identity="d3-edge")
    body = SERVICE.encode(payload())

    def request(request_id):
        return WORKER.process_request(
            SERVICE.encode(
                {
                    "request_id": request_id,
                    "payload_json": body.decode(),
                    "http_sha256": hashlib.sha256(body).hexdigest(),
                    "remaining_seconds": 20,
                }
            ),
            adapter=adapter,
            wire=WIRE,
            torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda _index: None)),
            gates=SimpleNamespace(check=lambda stage: None),
            output=tmp_path,
            model="d3-edge",
            nonce="nonce",
            gpu_uuid="GPU-fake",
            clock=clock,
        )

    assert request("last-allowed")["success"] is True
    refused = request("over-limit")
    assert refused["success"] is False and refused["campaign_stop"] is True
    assert "At most192" in refused["journal"]["error"]["message"]
    assert len(backend.calls) == 1
    assert json.loads((directory / "over-limit.json").read_bytes())["success"] is False


@pytest.mark.parametrize("fault", ["missing_journal", "cleanup_fence"])
def test_worker_preserves_primary_request_error_and_secondary_failure(tmp_path, monkeypatch, fault):
    records, sent = [], []
    original = {"type": "ValueError", "message": "ORIGINAL_REQUEST_FAILURE", "notes": []}
    journal = {
        "request_id": "request-1",
        "http_sha256": "a" * 64,
        "elapsed_seconds": 0.1,
        "journal_path": str(tmp_path / "request.json"),
        "error": original,
        "raw_response": {"retained": "exact raw"},
    }
    response = {"success": False, "campaign_stop": True, "wire_body_utf8": None, "journal": journal}

    def pin(_path):
        if fault == "missing_journal":
            raise FileNotFoundError("SECONDARY_JOURNAL_FAILURE")
        return {"sha256": "b" * 64, "size_bytes": 10}

    def fence(_index):
        if fault == "cleanup_fence":
            raise RuntimeError("CLEANUP_FENCE_FAILURE")

    torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=fence))
    adapter = SimpleNamespace(backend=SimpleNamespace(close=lambda: None))

    def initialize(_config, state, _output):
        state.update(resources_after_load={}, runtime={})
        return adapter, None, torch, None, None, []

    monkeypatch.setattr(WORKER, "validate_output", lambda _path: tmp_path)
    monkeypatch.setattr(WORKER, "initialize", initialize)
    monkeypatch.setattr(
        WORKER, "process_request", lambda *_args, **_kwargs: copy.deepcopy(response)
    )
    monkeypatch.setattr(WORKER, "authenticate_sources", lambda _manifest: {})
    monkeypatch.setattr(
        WORKER, "persist", lambda _path, value, **_kwargs: records.append(copy.deepcopy(value))
    )
    monkeypatch.setattr(WORKER, "send", lambda _control, value: sent.append(copy.deepcopy(value)))
    monkeypatch.setattr(
        WORKER,
        "CORE",
        SimpleNamespace(pin=pin, resource_snapshot=lambda _torch: {"cuda": {"uuid": "GPU-fake"}}),
    )
    monkeypatch.setattr(
        WORKER, "sys", SimpleNamespace(stdin=SimpleNamespace(buffer=io.BytesIO(b"{}\n")))
    )
    config = {
        "output_dir": str(tmp_path),
        "nonce": "nonce",
        "model": "d3-edge",
        "alias": "d3-edge",
        "expected_boot_id": "boot",
        "expected_memtotal": 1,
        "expected_gpu_uuid": "GPU-fake",
        "source_manifest": {},
    }
    assert WORKER.run(config, None) == 1
    terminal = sent[-1]
    assert terminal["event"] == "closed" and terminal["success"] is False
    receipt = terminal["receipt"]
    assert receipt["error"] == original and receipt["campaign_stop"] is True
    assert receipt["failed_request_response"] == response
    expected = (
        "SECONDARY_JOURNAL_FAILURE" if fault == "missing_journal" else "CLEANUP_FENCE_FAILURE"
    )
    assert any(expected in row["message"] for row in receipt["secondary_errors"])
    assert any(row.get("error") == original for row in records)


def test_persistent_nonce_uuid_change_blocks_actions_but_allows_owned_cleanup(monkeypatch):
    session = object.__new__(SERVICE.Session)
    session.nonce, session.gpu = "nonce", {"name": "H200", "uuid": "GPU-original"}
    session.blocked, session.identity_errors = False, []
    session.halted = False
    monkeypatch.setattr(
        SERVICE, "gpu_identity", lambda: {"name": "H200", "uuid": "GPU-replacement"}
    )
    with pytest.raises(ValueError, match="nonce"):
        session.check("wrong")
    with pytest.raises(ValueError, match="UUID changed"):
        session.check("nonce")
    assert session.blocked and session.halted and session.identity_errors
    assert session.check("nonce", cleanup=True) is None


@pytest.mark.parametrize("failed,local_late", [(False, False), (True, False), (False, True)])
def test_clean_closed_arm_can_advance_without_turning_a_failed_result_into_success(
    tmp_path, monkeypatch, failed, local_late
):
    session, records, _, _ = session_harness(
        tmp_path, monkeypatch, failed_models={"d3"} if failed else ()
    )
    session.begin("d3")
    session.blocked = failed or local_late
    result = session.end(session.nonce)
    assert result["success"] is (not failed and not local_late)
    assert result["worker"]["success"] is (not failed)
    assert result["owned_exit_confirmed"] and result["continuation_authorized"]
    assert not result["hard_halted"] and not session.halted and not session.blocked
    assert session.next_arm == 1 and session.active is None and session.process is None
    assert records[-1][0] == "arm-closures.json" and records[-1][2] == 0
    session.begin("d3-flash", session.nonce)
    assert session.active == "d3-flash"


def test_strict_json_rejects_duplicate_and_overflow_numbers_before_rpc(tmp_path):
    gateway, _, backend, *_ = composition(tmp_path)
    status, response = gateway.request(b'{"model":"d3-edge","state":1e999}')
    assert status == 504 and "answers" not in json.loads(response)
    assert backend.calls == [] and gateway.blocked


def test_sdk_checkpoint_failure_returns_available_raw_without_action(tmp_path):
    session = object.__new__(SERVICE.Session)
    session.nonce, session.active, session.gpu = "nonce", "d3-edge", {"uuid": "GPU-same"}
    session.blocked, session.last_control = False, None
    session.halted = False
    session.check = lambda *_args, **_kwargs: None
    session.process = SimpleNamespace(stdin=io.BytesIO())
    raw = {"journal": {"raw_response": {"exact": "native evidence"}}}

    def read_control():
        session.last_control = copy.deepcopy(raw)
        raise OSError("SDK protocol evidence checkpoint failed")

    def reap():
        session.process = None
        return True

    session.read_control, session.reap = read_control, reap
    result = session.predict("nonce", "request-1", SERVICE.encode(payload()), 20)
    assert result["success"] is False and result["campaign_stop"] is True
    assert result["wire_body_utf8"] is None and session.blocked
    assert result["available_worker_response"] == raw
    assert result["controller_error"]["type"] == "OSError"


def test_gateway_total_request_bound_does_not_forward_or_create_extra_journal(tmp_path):
    gateway, _, backend, *_ = composition(tmp_path)
    gateway.counter = 6 * 192
    status, response = gateway.request(SERVICE.encode(payload()))
    assert status == 503 and "answers" not in json.loads(response)
    assert gateway.blocked and backend.calls == []
    assert not (tmp_path / "local/requests").exists()


@pytest.mark.parametrize("failure", ["endpoint_save", "thread_start"])
def test_local_startup_failure_closes_server_and_joins_started_thread(
    tmp_path, monkeypatch, failure
):
    # Compile only the actual decorated entrypoint body; all SDK/socket/thread
    # boundaries are fake and no serving loop or actual thread is started.
    tree = ast.parse(SERVICE_PATH.read_bytes())
    node = copy.deepcopy(
        next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
    )
    node.decorator_list = []
    namespace = dict(SERVICE.__dict__)
    calls = []

    class Thread:
        def __init__(self, **_kwargs):
            calls.append("thread_constructed")

        def start(self):
            if failure == "thread_start":
                raise OSError("THREAD_START_FAILURE")
            calls.append("thread_started")

        def join(self):
            calls.append("thread_joined")

    server = SimpleNamespace(
        server_port=12345,
        serve_forever=lambda: None,
        shutdown=lambda: calls.append("shutdown"),
        server_close=lambda: calls.append("server_closed"),
    )
    original_save = SERVICE.save

    def save(path, value):
        if path.name == "endpoint.json":
            raise OSError("ENDPOINT_SAVE_FAILURE")
        original_save(path, value)

    namespace.update(
        PersistentWorker=lambda: SimpleNamespace(),
        http_server=lambda *_args: server,
        threading=SimpleNamespace(Thread=Thread),
        validate_inventory=lambda _blob: None,
        save=save,
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SERVICE_PATH), "exec"), namespace)
    manifest, profile = tmp_path / "manifest.json", tmp_path / "profile.json"
    manifest.write_bytes(b"{}")
    profile.write_bytes(b"{}")
    with pytest.raises(
        OSError,
        match="ENDPOINT_SAVE_FAILURE" if failure == "endpoint_save" else "THREAD_START_FAILURE",
    ):
        namespace["main"](
            output=str(tmp_path / "output"),
            manifest=str(manifest),
            weight_profile=str(profile),
            weight_profile_sha256=hashlib.sha256(b"{}").hexdigest(),
            helper_endpoint="http://127.0.0.1:1",
            helper_qualification_sha256="a" * 64,
            token_file=str(tmp_path / "private-token"),
            run_id="fake-test",
            run=True,
        )
    assert "server_closed" in calls
    assert ("shutdown" in calls) is (failure == "endpoint_save")
    assert ("thread_joined" in calls) is (failure == "endpoint_save")
    assert (
        json.loads((tmp_path / "output/controller-error.json").read_bytes())["campaign_stop"]
        is True
    )


def test_staging_failure_publishes_original_partial_tree_without_model_child(tmp_path, monkeypatch):
    failure = OSError("STAGING_PRIMARY_FAILURE")
    published = []
    core = SimpleNamespace(
        OUTPUT_ROOT=tmp_path,
        ordinary=load_source("real_file_only_replay_core", SERVICE.CORE).ordinary,
        authenticate_source=lambda _sha: {},
        write=lambda path, value, **_kwargs: SERVICE.save(path, value),
    )

    def stage_cache(config, _core, _identities):
        SERVICE.save(Path(config["output"]) / "partial-stage.json", {"actual_partial": True})
        raise failure

    def tree_inventory(path):
        return {entry.name: SERVICE.pin(entry) for entry in path.iterdir() if entry.is_file()}

    def publish(path, binding, confirmed):
        published.append((path, binding, confirmed, tree_inventory(path)))
        return SERVICE.pin(path / "export-index.json")

    staging = SimpleNamespace(
        load_core=lambda: core,
        capture_mount=lambda alias: {"alias": alias},
        stage_cache=stage_cache,
        tree_inventory=tree_inventory,
        publish_output=publish,
    )
    monkeypatch.setattr(SERVICE, "validate_inventory", lambda _blob: {})
    monkeypatch.setattr(SERVICE, "validate_weight_profile", lambda *_args: {})
    monkeypatch.setattr(SERVICE, "load_staging", lambda: staging)
    monkeypatch.setattr(SERVICE, "gpu_identity", lambda: {"name": "H200", "uuid": "GPU-fake"})
    with pytest.raises(OSError, match="STAGING_PRIMARY_FAILURE") as caught:
        SERVICE.Session("fake-stage", b"{}", b"{}", "a" * 64)
    assert caught.value is failure
    result = caught.value.decision3_initialization_failure
    assert result["success"] is False and result["owned_exit_confirmed"] is True
    assert result["model_loads"] == 0 and result["arms"] == [] and result["export_index"]
    assert len(published) == 1 and published[0][2] is True
    assert "partial-stage.json" in published[0][3]
    actual = json.loads(
        (tmp_path / "browser-fake-stage/initialization-failure-receipt.json").read_bytes()
    )
    assert actual["error"]["message"] == "STAGING_PRIMARY_FAILURE" and actual["success"] is False


@pytest.mark.parametrize("success", [True, False])
def test_known_terminal_outcomes_authenticate_without_rewriting_worker_evidence(success):
    closed = closed_frame(success=success)
    original = copy.deepcopy(closed)
    assert (
        SERVICE.closed_worker_success(
            closed,
            model="d3-edge",
            nonce="nonce-test-only",
            pid=1001,
            gpu_uuid="GPU-test-only",
            returncode=0 if success else 1,
        )
        is success
    )
    assert closed == original


def replace_field(value, path, replacement):
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = replacement


@pytest.mark.parametrize(
    "path,replacement",
    [
        (("event",), "response"),
        (("receipt", "profile"), "unreviewed-profile"),
        (("receipt", "model"), "d3"),
        (("receipt", "alias"), "d3"),
        (("receipt", "nonce"), "other-allocation"),
        (("receipt", "pid"), 1002),
        (("receipt", "pid"), True),
        (("receipt", "gpu_uuid"), "GPU-replaced"),
        (("receipt", "final_resources", "cuda", "uuid"), "GPU-replaced"),
        (("receipt", "secondary_errors"), [{"type": "OSError", "message": "cleanup failed"}]),
        (("receipt", "network_attempts_blocked"), ["https://unexpected.invalid"]),
        (("receipt", "cache_before", "full_payload_sha256_authenticated"), False),
        (("receipt", "cache_after", "full_payload_sha256"), "c" * 64),
        (("receipt", "cache_before"), None),
        (("receipt", "stage"), "closing"),
        (("receipt", "error"), {"message": "unacknowledged failure"}),
        (("campaign_stop",), True),
    ],
)
def test_terminal_authentication_refuses_foreign_or_unclean_receipts(path, replacement):
    closed = closed_frame()
    replace_field(closed, path, replacement)
    with pytest.raises(ValueError):
        SERVICE.closed_worker_success(
            closed,
            model="d3-edge",
            nonce="nonce-test-only",
            pid=1001,
            gpu_uuid="GPU-test-only",
            returncode=0,
        )


@pytest.mark.parametrize(
    "path,replacement",
    [
        (("receipt", "failed_request_response"), None),
        (("receipt", "failed_request_response", "event"), "ready"),
        (("receipt", "failed_request_response", "model"), "d3"),
        (("receipt", "failed_request_response", "nonce"), "other"),
        (("receipt", "failed_request_response", "gpu_uuid"), "GPU-other"),
        (("receipt", "failed_request_response", "wire_body_utf8"), '{"answers":{}}'),
        (("receipt", "failed_request_response", "success"), True),
        (("receipt", "failed_request_response", "campaign_stop"), False),
        (
            ("receipt", "failed_request_response", "journal", "error"),
            {"message": "different failure"},
        ),
        (("receipt", "failed_request_response", "journal", "request_id"), "another-request"),
        (("receipt", "failed_request_response", "journal", "http_sha256"), "c" * 64),
        (("receipt", "failed_request_response", "journal", "model"), "d3"),
        (("receipt", "failed_request_response", "journal", "nonce"), "other-allocation"),
        (("receipt", "failed_request_response", "journal", "gpu_uuid"), "GPU-other"),
        (("receipt", "requests"), []),
        (("receipt", "requests", 0, "request_id"), "another-request"),
        (("receipt", "requests", 0, "http_sha256"), "d" * 64),
        (("receipt", "requests", 0, "success"), True),
        (("receipt", "requests", 0, "campaign_stop"), False),
    ],
)
def test_failed_arm_continuation_requires_bound_failed_request_evidence(path, replacement):
    closed = closed_frame(success=False)
    replace_field(closed, path, replacement)
    with pytest.raises(ValueError):
        SERVICE.closed_worker_success(
            closed,
            model="d3-edge",
            nonce="nonce-test-only",
            pid=1001,
            gpu_uuid="GPU-test-only",
            returncode=1,
        )


@pytest.mark.parametrize("field", ["request_id", "http_sha256"])
def test_two_missing_request_identities_do_not_authenticate_each_other(field):
    closed = closed_frame(success=False)
    receipt = closed["receipt"]
    receipt["requests"][-1].pop(field)
    receipt["failed_request_response"].pop(field)
    receipt["failed_request_response"]["journal"].pop(field)
    with pytest.raises(ValueError):
        SERVICE.closed_worker_success(
            closed,
            model="d3-edge",
            nonce="nonce-test-only",
            pid=1001,
            gpu_uuid="GPU-test-only",
            returncode=1,
        )


@pytest.mark.parametrize(
    "success,returncode",
    [(True, 1), (True, -9), (False, 0), (False, -9), (True, False), (False, True), (True, None)],
)
def test_terminal_outcome_requires_exact_known_exit_code(success, returncode):
    with pytest.raises(ValueError):
        SERVICE.closed_worker_success(
            closed_frame(success=success),
            model="d3-edge",
            nonce="nonce-test-only",
            pid=1001,
            gpu_uuid="GPU-test-only",
            returncode=returncode,
        )


@pytest.mark.parametrize(
    "fault",
    [
        "unknown_exit",
        "unknown_closed",
        "abnormal_returncode",
        "source_change",
        "gpu_change",
        "secondary_cleanup",
    ],
)
def test_session_hard_failures_never_advance_or_become_recoverable(tmp_path, monkeypatch, fault):
    session, _, _, _ = session_harness(tmp_path, monkeypatch)
    session.begin("d3")
    original_reap = session.reap
    if fault == "unknown_exit":
        session.reap = lambda: (_ for _ in ()).throw(RuntimeError("Unknown owned descendant"))
    elif fault == "unknown_closed":
        session.read_control = lambda: {"event": "closed", "success": True, "campaign_stop": False}
    elif fault == "abnormal_returncode":

        def abnormal_reap():
            original_reap()
            session.last_returncode = -9
            return True

        session.reap = abnormal_reap
    elif fault == "source_change":
        monkeypatch.setattr(
            SERVICE,
            "validate_inventory",
            lambda _blob: (_ for _ in ()).throw(ValueError("SOURCE_CHANGED")),
        )
    elif fault == "gpu_change":
        monkeypatch.setattr(SERVICE, "gpu_identity", lambda: {"name": "H200", "uuid": "GPU-other"})
    else:
        closed = closed_frame("d3", pid=session.process.pid)
        closed["receipt"]["secondary_errors"] = [{"type": "OSError", "message": "cleanup"}]
        session.read_control = lambda: closed
    result = session.end(session.nonce)
    assert result["success"] is False and result["continuation_authorized"] is False
    assert result["hard_halted"] is True and session.halted and session.blocked
    assert session.next_arm == 0
    with pytest.raises(ValueError):
        session.begin("d3-flash", session.nonce)


def test_end_persistence_failure_cannot_advance_despite_clean_worker_exit(tmp_path, monkeypatch):
    session, _, _, _ = session_harness(tmp_path, monkeypatch)
    session.begin("d3")
    session.record = lambda *_args: (_ for _ in ()).throw(OSError("CLOSURE_NOT_DURABLE"))
    with pytest.raises(OSError, match="CLOSURE_NOT_DURABLE"):
        session.end(session.nonce)
    assert session.process is None and session.next_arm == 0
    assert session.halted and session.blocked
    with pytest.raises(ValueError, match="next ordered arm"):
        session.begin("d3-flash", session.nonce)


def test_new_arm_unknown_exit_does_not_reuse_previous_arms_returncode(tmp_path, monkeypatch):
    session, _, _, _ = session_harness(tmp_path, monkeypatch)
    session.begin("d3")
    session.end(session.nonce)
    assert session.last_returncode == 0
    session.begin("d3-flash", session.nonce)
    assert session.last_returncode is None
    session.reap = lambda: (_ for _ in ()).throw(RuntimeError("UNKNOWN_EXIT"))
    result = session.end(session.nonce)
    assert result["returncode"] is None and not result["owned_exit_confirmed"]
    assert not result["continuation_authorized"] and result["hard_halted"]
    assert session.next_arm == 1


def test_six_ordered_mixed_arms_keep_failures_and_complete_campaign_without_halt(
    tmp_path, monkeypatch
):
    failed = {"d3-flash", "d3-lite"}
    session, records, launched, published = session_harness(
        tmp_path, monkeypatch, failed_models=failed
    )
    gateway = SERVICE.Gateway(
        session.predict, tmp_path / "local", {"qualification_sha256": "a" * 64}
    )
    for index, model in enumerate(SERVICE.MODELS):
        gateway.accept_begin(session.begin(model, gateway.nonce))
        body = SERVICE.encode({**payload(), "model": model})
        status, response = gateway.request(body)
        assert status == (504 if model in failed else 200)
        if model in failed:
            assert "answers" not in json.loads(response)
            assert gateway.blocked and not gateway.halted
            assert gateway.request(body)[0] == 504 and not gateway.halted
        closed = session.end(session.nonce)
        original = copy.deepcopy(closed)
        local = gateway.accept_end(closed)
        assert closed == original
        assert closed["success"] is (model not in failed)
        assert local["gateway_arm_success"] is (model not in failed)
        assert local["gateway_arm_request_blocked"] is (model in failed)
        assert closed["continuation_authorized"] and closed["owned_exit_confirmed"]
        assert not gateway.blocked and not gateway.halted and gateway.active is None
        assert session.next_arm == index + 1 and not session.halted
    final = session.close(session.nonce)
    assert final["stage"] == "complete_with_failures"
    assert final["completed"] is True and final["success"] is False
    assert final["owned_exit_confirmed"] is True and not final["hard_halted"]
    assert final["failed_models"] == ["d3-flash", "d3-lite"] and final["unrun"] == []
    assert [row["model"] for row in final["arms"]] == list(SERVICE.MODELS)
    assert final["task_score"] is None and final["helper_calls"] == final["optimizer_updates"] == 0
    assert len(launched) == 6 and len({p.pid for p, _ in launched}) == 6
    assert all(options["start_new_session"] for _, options in launched)
    assert len(published) == 1 and published[0][2] is True
    assert [cursor for name, _, cursor in records if name == "arm-closures.json"] == list(range(6))


def test_partial_campaign_reports_unrun_cursor_and_keeps_failed_model(tmp_path, monkeypatch):
    session, _, _, _ = session_harness(tmp_path, monkeypatch, failed_models={"d3"})
    session.begin("d3")
    session.end(session.nonce)
    result = session.close(session.nonce)
    assert result["stage"] == "stopped" and result["completed"] is False
    assert result["success"] is False and result["failed_models"] == ["d3"]
    assert result["unrun"] == list(SERVICE.MODELS[1:])
    assert result["owned_exit_confirmed"] and not result["hard_halted"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("event", "closed"),
        ("model", "d3"),
        ("alias", "d3"),
        ("nonce", "other"),
        ("pid", 0),
        ("pid", True),
        ("gpu_uuid", "GPU-other"),
    ],
)
def test_gateway_refuses_unbound_ready_worker_and_latches_halt(tmp_path, field, value):
    gateway = SERVICE.Gateway(None, tmp_path, {})
    reply = ready_reply()
    reply["ready"][field] = value
    with pytest.raises(ValueError):
        gateway.accept_begin(reply)
    assert gateway.halted and gateway.blocked and gateway.active is None
    with pytest.raises(ValueError):
        gateway.accept_begin(ready_reply())


@pytest.mark.parametrize(
    "fault",
    [
        "schema",
        "model",
        "nonce",
        "gpu",
        "missing_worker",
        "worker_pid",
        "worker_nonce",
        "worker_cache",
        "failed_to_success",
    ],
)
def test_only_bound_authenticated_end_reply_can_release_gateway_arm_block(tmp_path, fault):
    gateway = SERVICE.Gateway(None, tmp_path, {})
    gateway.accept_begin(ready_reply())
    gateway.blocked = True
    reply = arm_reply(worker_success=fault != "failed_to_success")
    if fault == "schema":
        reply["schema_id"] = "decision3-browser-arm-lifecycle-v1"
    elif fault in {"model", "nonce"}:
        reply[fault] = "other"
    elif fault == "gpu":
        reply["gpu"] = {"name": "H200", "uuid": "GPU-other"}
    elif fault == "missing_worker":
        reply["worker"] = {"event": "closed", "success": True}
    elif fault == "worker_pid":
        reply["worker"]["receipt"]["pid"] += 1
    elif fault == "worker_nonce":
        reply["worker"]["receipt"]["nonce"] = "other"
    elif fault == "worker_cache":
        reply["worker"]["receipt"]["cache_after"] = {}
    else:
        reply["success"] = True
    original = copy.deepcopy(reply)
    with pytest.raises(ValueError):
        gateway.accept_end(reply)
    assert gateway.halted and gateway.blocked and reply == original


@pytest.mark.parametrize(
    "field,value",
    [("continuation_authorized", False), ("owned_exit_confirmed", False), ("hard_halted", True)],
)
def test_authenticated_but_unauthorized_closure_never_clears_gateway_halt(tmp_path, field, value):
    gateway = SERVICE.Gateway(None, tmp_path, {})
    gateway.accept_begin(ready_reply())
    gateway.blocked = True
    reply = arm_reply()
    reply[field] = value
    result = gateway.accept_end(reply)
    assert result["gateway_arm_request_blocked"] is True
    assert gateway.halted and gateway.blocked
    with pytest.raises(ValueError):
        gateway.accept_begin(ready_reply("d3-lite"))


def test_gateway_late_result_remains_failure_even_when_worker_closed_successfully(tmp_path):
    gateway, _, _, _, _, _, _ = composition(tmp_path, after_rpc=21)
    assert gateway.request(SERVICE.encode(payload()))[0] == 504
    assert gateway.blocked and not gateway.halted
    reply = arm_reply()
    original = copy.deepcopy(reply)
    result = gateway.accept_end(reply)
    assert result["success"] is True and result["gateway_arm_success"] is False
    assert result["gateway_arm_request_blocked"] is True
    assert reply == original and not gateway.halted and not gateway.blocked


@pytest.mark.parametrize(
    "fault",
    [
        "request_id",
        "http_sha256",
        "nonce",
        "model",
        "success_flags",
        "wire_with_failure",
        "missing_wire",
    ],
)
def test_gateway_unbound_or_malformed_rpc_is_sticky_and_cannot_be_cleared(tmp_path, fault):
    gateway, _, _, _, _, _, _ = composition(tmp_path)
    original_rpc = gateway.rpc

    def corrupted(*args):
        result = original_rpc(*args)
        if fault in {"request_id", "http_sha256", "nonce", "model"}:
            result[fault] = "other"
        elif fault == "success_flags":
            result["campaign_stop"] = True
        elif fault == "wire_with_failure":
            result["success"], result["campaign_stop"] = False, True
        else:
            result["wire_body_utf8"] = None
        return result

    gateway.rpc = corrupted
    status, response = gateway.request(SERVICE.encode(payload()))
    assert status == 504 and "answers" not in json.loads(response)
    assert gateway.halted and gateway.blocked
    result = gateway.accept_end(arm_reply())
    assert result["gateway_arm_success"] is False and gateway.halted and gateway.blocked


@pytest.mark.parametrize("failure_stage", ["first_evidence", "final_evidence", "queue_evidence"])
def test_gateway_evidence_write_failure_is_hard_and_not_released_by_clean_end(
    tmp_path, monkeypatch, failure_stage
):
    gateway, _, _, _, _, _, _ = composition(tmp_path)
    original_save, count = SERVICE.save, 0

    def broken_save(path, row):
        nonlocal count
        count += 1
        if count == (2 if failure_stage == "final_evidence" else 1):
            raise OSError("DURABLE_EVIDENCE_FAILURE")
        return original_save(path, row)

    monkeypatch.setattr(SERVICE, "save", broken_save)
    if failure_stage == "queue_evidence":
        gateway.lock = SimpleNamespace(acquire=lambda **_kwargs: False)
    if failure_stage == "first_evidence":
        status, response = gateway.request(SERVICE.encode(payload()))
        assert status == 504 and "answers" not in json.loads(response)
    else:
        with pytest.raises(OSError, match="DURABLE_EVIDENCE_FAILURE"):
            gateway.request(SERVICE.encode(payload()))
    assert gateway.halted and gateway.blocked
    result = gateway.accept_end(arm_reply())
    assert result["gateway_arm_success"] is False and gateway.halted and gateway.blocked


def delivery_gateway(tmp_path, *, failed=True):
    clock = Clock()

    def rpc(nonce, request_id, body, remaining):
        assert remaining > 0
        digest = hashlib.sha256(body).hexdigest()
        model = SERVICE.parse(body)["model"]
        if failed:
            clock.advance(38)
        return {
            "event": "response",
            "request_id": request_id,
            "http_sha256": digest,
            "model": model,
            "nonce": nonce,
            "gpu_uuid": "GPU-test-only",
            "success": not failed,
            "campaign_stop": failed,
            "wire_body_utf8": None if failed else '{"answers":{"operation":{"choice":"DONE"}}}',
            "journal": {
                "request_id": request_id,
                "http_sha256": digest,
                "model": model,
                "nonce": nonce,
                "gpu_uuid": "GPU-test-only",
                "error": {
                    "type": "TimeoutError",
                    "message": "Cold request exceeded20s",
                    "notes": [],
                }
                if failed
                else None,
                "raw_response": {"retained": "native raw even after client disconnect"},
            },
        }

    gateway = SERVICE.Gateway(rpc, tmp_path / "local", {}, clock=clock)
    gateway.accept_begin(ready_reply("d3"))
    return gateway, clock


def failed_delivery_case(tmp_path):
    gateway, clock = delivery_gateway(tmp_path)
    evidence = {}
    status, response = gateway.request(
        SERVICE.encode({**payload(), "model": "d3"}), delivery_evidence=evidence
    )
    assert status == 504 and "answers" not in SERVICE.parse(response)
    assert gateway.blocked and not gateway.halted
    assert evidence["known_no_action_failure"] is True
    path = gateway.output / "requests" / (evidence["request_id"] + ".json")
    assert evidence["request_pin"] == SERVICE.pin(path)
    return gateway, clock, status, response, evidence


def close_delivery_arm(gateway, *, failed):
    reply = arm_reply("d3", worker_success=not failed)
    if failed:
        paths = list((gateway.output / "requests").glob("*.json"))
        assert len(paths) == 1
        row = SERVICE.parse(paths[0].read_bytes())
        response = row["rpc"]
        receipt = reply["worker"]["receipt"]
        receipt["failed_request_response"] = copy.deepcopy(response)
        receipt["error"] = copy.deepcopy(response["journal"]["error"])
        receipt["requests"] = [
            {
                key: response[key]
                for key in ("request_id", "http_sha256", "success", "campaign_stop")
            }
        ]
    return gateway.accept_end(reply)


@pytest.mark.parametrize("failed", [True, False], ids=["durable504", "uncertain200"])
def test_real_loopback_client_rst_after_rpc_entry_classifies_delivery_without_guessing(
    tmp_path, failed
):
    gateway, _ = delivery_gateway(tmp_path, failed=failed)
    entered, release, delivered = threading.Event(), threading.Event(), threading.Event()
    original_rpc, original_delivery = gateway.rpc, gateway.delivery_failed
    receipts, errors = [], []

    def paused_rpc(*args):
        entered.set()
        if not release.wait(5):
            raise TimeoutError("Owned test RPC release missing")
        return original_rpc(*args)

    def observed_delivery(exc, **kwargs):
        try:
            receipts.append((type(exc), original_delivery(exc, **kwargs)))
        except BaseException as error:
            errors.append(error)
            raise
        finally:
            delivered.set()

    gateway.rpc, gateway.delivery_failed = paused_rpc, observed_delivery
    token = "owned-loopback-test-token"
    server = SERVICE.http_server(gateway, token)
    thread = threading.Thread(target=server.serve_forever, daemon=False)
    client = None
    thread.start()
    try:
        assert server.server_address[0] == "127.0.0.1"
        client = socket.create_connection(server.server_address, timeout=5)
        body = SERVICE.encode({**payload(), "model": "d3"})
        headers = (
            "POST /v1/systemone HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{server.server_port}\r\n"
            f"Authorization: Bearer {token}\r\n"
            f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
        ).encode()
        client.sendall(headers + body)
        assert entered.wait(5), "The owned HTTP request never reached RPC"
        client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        client.close()
        client = None
        release.set()
        assert delivered.wait(5), "Actual reset socket did not trigger delivery accounting"
        assert not errors and len(receipts) == 1
        error_type, receipt = receipts[0]
        assert issubclass(error_type, OSError)
        assert receipt["status"] == (504 if failed else 200)
        assert receipt["known_no_action_failure"] is failed
        assert receipt["validation_error"] is None
        saved = gateway.output / "http-delivery-errors" / (receipt["request_id"] + ".json")
        assert SERVICE.parse(saved.read_bytes()) == receipt
        row = SERVICE.parse(
            (gateway.output / "requests" / (receipt["request_id"] + ".json")).read_bytes()
        )
        assert row["rpc"]["journal"]["raw_response"] == {
            "retained": "native raw even after client disconnect"
        }
        assert row["gateway_action_authorized"] is (not failed)
        assert gateway.halted is (not failed)
        assert gateway.blocked
        local = close_delivery_arm(gateway, failed=failed)
        assert local["gateway_arm_success"] is False
        if failed:
            assert not gateway.blocked and not gateway.halted
            gateway.accept_begin(ready_reply("d3-flash", pid=1002))
            assert gateway.active == "d3-flash"
        else:
            assert gateway.blocked and gateway.halted
    finally:
        release.set()
        if client is not None:
            client.close()
        server.shutdown()
        server.server_close()
        thread.join(5)
        assert not thread.is_alive()


def test_old_arm_failed_delivery_cannot_pollute_a_new_active_arm(tmp_path):
    gateway, _, status, response, evidence = failed_delivery_case(tmp_path)
    assert close_delivery_arm(gateway, failed=True)["gateway_arm_success"] is False
    gateway.accept_begin(ready_reply("d3-flash", pid=1002))
    before = (
        gateway.active,
        gateway.nonce,
        copy.deepcopy(gateway.gpu),
        gateway.pid,
        gateway.blocked,
        gateway.halted,
    )
    receipt = gateway.delivery_failed(
        BrokenPipeError("Old HTTP client disappeared"),
        status=status,
        response=response,
        evidence=evidence,
    )
    after = (
        gateway.active,
        gateway.nonce,
        gateway.gpu,
        gateway.pid,
        gateway.blocked,
        gateway.halted,
    )
    assert after == before and gateway.active == "d3-flash"
    assert receipt["known_no_action_failure"] is True
    assert receipt["model"] == "d3" and receipt["request_id"] == evidence["request_id"]


def test_known_failed_delivery_never_clears_an_existing_hard_halt(tmp_path):
    gateway, _, status, response, evidence = failed_delivery_case(tmp_path)
    gateway.halted = gateway.blocked = True
    receipt = gateway.delivery_failed(
        BrokenPipeError("Disconnected"), status=status, response=response, evidence=evidence
    )
    assert receipt["known_no_action_failure"] is True
    assert gateway.halted and gateway.blocked


@pytest.mark.parametrize(
    "fault", ["missing", "response", "status", "model", "pin", "stored_row", "unknown_flag"]
)
def test_unknown_or_changed_delivery_evidence_is_a_sticky_hard_halt(tmp_path, fault):
    gateway, _, status, response, evidence = failed_delivery_case(tmp_path)
    if fault == "missing":
        evidence = None
    elif fault == "response":
        response = b'{"detail":"different response"}'
    elif fault == "status":
        status = 503
    elif fault == "model":
        evidence["model"] = "d3-flash"
    elif fault == "pin":
        evidence["request_pin"]["sha256"] = "c" * 64
    elif fault == "stored_row":
        path = gateway.output / "requests" / (evidence["request_id"] + ".json")
        row = SERVICE.parse(path.read_bytes())
        row["raw_http_hex"] = "changed after delivery snapshot"
        SERVICE.save(path, row)
    else:
        evidence["known_no_action_failure"] = False
    receipt = gateway.delivery_failed(
        BrokenPipeError("Disconnected"), status=status, response=response, evidence=evidence
    )
    assert receipt["known_no_action_failure"] is False
    assert gateway.halted and gateway.blocked
    if fault != "unknown_flag":
        assert receipt["validation_error"] is not None
    close_delivery_arm(gateway, failed=True)
    assert gateway.halted and gateway.blocked


@pytest.mark.parametrize("new_arm", [False, True])
def test_delivery_receipt_write_failure_latches_halt_even_for_old_known_failure(
    tmp_path, monkeypatch, new_arm
):
    gateway, _, status, response, evidence = failed_delivery_case(tmp_path)
    if new_arm:
        close_delivery_arm(gateway, failed=True)
        gateway.accept_begin(ready_reply("d3-flash", pid=1002))
    original_save = SERVICE.save

    def fail_receipt(path, value):
        if path.parent.name == "http-delivery-errors":
            raise OSError("DELIVERY_RECEIPT_NOT_DURABLE")
        return original_save(path, value)

    monkeypatch.setattr(SERVICE, "save", fail_receipt)
    with pytest.raises(OSError, match="DELIVERY_RECEIPT_NOT_DURABLE"):
        gateway.delivery_failed(
            BrokenPipeError("Disconnected"), status=status, response=response, evidence=evidence
        )
    assert gateway.halted and gateway.blocked


def test_delivery_metadata_cost_is_inside_original_gateway_20s(tmp_path):
    gateway, clock = delivery_gateway(tmp_path, failed=False)

    class SlowEvidence(dict):
        delayed = False

        def update(self, **kwargs):
            super().update(**kwargs)
            if not self.delayed:
                self.delayed = True
                clock.advance(21)

    evidence = SlowEvidence()
    status, response = gateway.request(
        SERVICE.encode({**payload(), "model": "d3"}), delivery_evidence=evidence
    )
    assert status == 504 and "answers" not in SERVICE.parse(response)
    assert gateway.blocked and not gateway.halted
    assert evidence["known_no_action_failure"] is True and evidence["status"] == 504
    path = gateway.output / "requests" / (evidence["request_id"] + ".json")
    assert evidence["request_pin"] == SERVICE.pin(path)
    row = SERVICE.parse(path.read_bytes())
    assert row["late_after_response_binding"] and row["gateway_action_authorized"] is False
