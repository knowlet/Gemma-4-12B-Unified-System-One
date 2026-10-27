"""Synthetic multi-request browser child tests; no GPU or systemd is started."""

import copy
import hashlib
import http.client as http_client
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import jev_decision3_candidate_http as http
import jev_decision3_candidate_service_lifecycle as lifecycle
import jev_decision3_link_navigation_features as features
import jev_decision3_semantic_browser_child as child
import jev_decision3_semantic_operation_audit as audit
import pytest
from test_jev_decision3_native import payload

pytest_plugins = ("test_jev_decision3_semantic_browser_binding",)


def plan_for(tmp_path):
    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    transport = {
        key: str(tmp_path / (key + ".py"))
        for key in (
            "http_release",
            "lease_release",
            "binding_release",
            "lifecycle_release",
            "semantic_adapter_release",
            "semantic_audit_release",
        )
    }
    for name in ("lock", "journal"):
        path = tmp_path / name
        path.touch()
        transport[name] = str(path)
    transport["nonce"] = "a" * 32
    transport["stat"] = {}
    for name in ("lock", "journal"):
        info = Path(transport[name]).stat()
        transport["stat"][name] = {"device": info.st_dev, "inode": info.st_ino}
    return {
        "schema": child.SCHEMA,
        "semantic_opt_in": True,
        "vocabulary": {
            "path": str(tmp_path / "vocabulary.json"),
            "file_sha256": "a" * 64,
            "names_sha256": "b" * 64,
        },
        "unit": "jev-browser-candidate-" + "c" * 32 + ".service",
        "output_dir": str(output),
        "stop_path": str(output / "stop.json"),
        "python": "/usr/bin/python3",
        "cache_root": str(tmp_path / "cache"),
        "modules_cache": str(tmp_path / "modules"),
        "overlay": str(
            Path(child.__file__).absolute().parent.parent / "artifacts/jev-runtime/overlay"
        ),
        "source_pins": {
            str(Path(child.__file__).absolute()): "a" * 64,
            str(Path(child.common.__file__).absolute()): "b" * 64,
            str(Path(child.semantic_owner.__file__).absolute()): "c" * 64,
        },
        "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
        "deltas": {
            name: {
                "path": str(tmp_path / (name + ".npy")),
                "file_sha256": "d" * 64,
                "fp32_sha256": "e" * 64,
            }
            for name in ("operation", "click_target")
        },
        "transport": transport,
        "model_alias": "native-test",
        "max_requests": 2,
        "wall_seconds": 120,
    }


def test_plan_source_gate_and_exact_handshake(tmp_path, monkeypatch):
    plan = plan_for(tmp_path)
    child._validate_plan(plan)
    with pytest.raises(ValueError, match="independent review"):
        child.main(["--plan", "/missing", "--plan-sha256", "a" * 64])
    with pytest.raises(ValueError, match="independent review"):
        child._serve(
            plan, "a" * 64, "b" * 32, None, None, (), Path(plan["output_dir"]), lambda: None
        )
    for key, invalid in (
        ("semantic_opt_in", False),
        ("max_requests", 17),
        ("max_requests", True),
        ("wall_seconds", 601),
        ("unit", "other.service"),
        ("stop_path", "/tmp/other"),
    ):
        value = copy.deepcopy(plan)
        value[key] = invalid
        with pytest.raises(ValueError):
            child._validate_plan(value)
    control = Path(plan["output_dir"]) / "go.json"
    value = {
        "schema": "decision3-candidate-browser-go-v1",
        "invocation": "a" * 32,
        "plan_sha256": "b" * 64,
    }
    control.write_text(json.dumps(value))
    assert (
        child._control(
            control,
            schema=value["schema"],
            invocation=value["invocation"],
            plan_sha=value["plan_sha256"],
        )
        == value
    )
    with pytest.raises(ValueError, match="bound parent control"):
        child._control(
            control, schema=value["schema"], invocation="c" * 32, plan_sha=value["plan_sha256"]
        )
    control.unlink()
    control.symlink_to(Path(plan["transport"]["lock"]))
    with pytest.raises(ValueError, match="ordinary private file"):
        child._control(
            control,
            schema=value["schema"],
            invocation=value["invocation"],
            plan_sha=value["plan_sha256"],
        )


def test_shared_journal_has_one_complete_pair_per_child_call(tmp_path):
    plan = plan_for(tmp_path)
    t, pid = plan["transport"], os.getpid()
    rows = [
        {
            "event": event,
            "kind": "decision3-gpu",
            "pid": pid,
            "thread": 42,
            "monotonic_ns": i,
            "nonce": t["nonce"],
        }
        for i, event in enumerate(("acquired", "completed", "acquired", "completed"), 1)
    ]
    Path(t["journal"]).write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert child._lease_summary(t, pid, 2)["own_completed_pairs"] == 2
    rows[-1]["thread"] = 43
    Path(t["journal"]).write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="same-thread"):
        child._lease_summary(t, pid, 2)


def _request(address, token, sequence):
    body = payload(2)
    body["state"] = {"synthetic": sequence}
    raw = json.dumps(body, ensure_ascii=False).encode()
    conn = http_client.HTTPConnection(*address, timeout=4)
    try:
        conn.request(
            "POST", "/v1/systemone", body=raw, headers={"Authorization": "Bearer " + token}
        )
        response = conn.getresponse()
        data = response.read()
        assert response.status == 200
        assert json.loads(data)["decision3_browser_wire"]["answer_values_changed"] is False
    finally:
        conn.close()


@pytest.mark.parametrize("end", ["request_limit", "parent_stop"])
@pytest.mark.parametrize("audit_invalid", [False, True])
def test_multiple_loopback_calls_ready_stop_and_private_records(
    tmp_path, monkeypatch, semantic_binding_context, end, audit_invalid
):
    monkeypatch.setattr(child, "SOURCE_ONLY", False)
    monkeypatch.setattr(lifecycle, "SOURCE_ONLY", False)
    plan = plan_for(tmp_path)
    if end == "parent_stop":
        plan["max_requests"] = 3
    semantic_binding_context.lease.held = semantic_binding_context.lease.fatal = None
    monkeypatch.setattr(child.common, "_shared_identity", lambda *_args: None)
    monkeypatch.setattr(child.common, "_verify_source_pins", lambda *_args: None)
    monkeypatch.setattr(child.semantic_owner, "_load_deltas", lambda *_args: {})
    monkeypatch.setattr(
        child.semantic_owner, "_load_vocabulary", lambda *_args: ["CLICK", "TYPE_TEXT"]
    )
    monkeypatch.setattr(
        child, "_lease_summary", lambda _transport, _pid, count: {"own_completed_pairs": count}
    )
    monkeypatch.setattr(features, "_validate_manifest", lambda *_args: None)
    monkeypatch.setattr(
        audit,
        "make_witness_writer",
        lambda *_args, **_kwargs: (lambda **_values: None, semantic_binding_context.receipts),
    )
    audited = []

    def audit_once(directory, **kwargs):
        receipt = semantic_binding_context.receipts[len(audited)]
        assert directory == receipt["directory"]
        assert kwargs["expected_file_sha256"] == receipt["manifest_sha256"]
        assert kwargs["expected_witness_sha256"] == receipt["witness_sha256"]
        assert kwargs["expected_names_sha256"] == plan["vocabulary"]["names_sha256"]
        assert kwargs["expected_delta_sha256"] == {
            head: row["fp32_sha256"] for head, row in plan["deltas"].items()
        }
        audited.append(directory)
        if audit_invalid:
            raise ValueError("synthetic numeric inconsistency")
        return {"status": "numeric_consistency_only", "browser_qualified": False}

    monkeypatch.setattr(audit, "audit_directory", audit_once)
    lease_module = SimpleNamespace(
        ForwardLease=lambda *_args, **_kwargs: semantic_binding_context.lease
    )

    class BindingModule:
        @staticmethod
        def SemanticBrowserBinding(*_args, **kwargs):
            instance = semantic_binding_context.make(max_requests=plan["max_requests"])
            original_sink = instance.service.event_sink

            def event(row):
                original_sink(row)
                if row["status"] == "prepared":
                    kwargs["private_artifact_sink"](semantic_binding_context.bindings[-1])
                kwargs["event_sink"](row)

            instance.service.event_sink = event
            return instance

    result, errors = [], []
    sha, invocation = "d" * 64, "f" * 32
    output = Path(plan["output_dir"])

    def run():
        try:
            result.append(
                child._serve(
                    plan,
                    sha,
                    invocation,
                    semantic_binding_context.base,
                    object(),
                    (http, lease_module, BindingModule, lifecycle, object(), audit),
                    output,
                    lambda: None,
                )
            )
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        deadline = time.monotonic() + 4
        while not (output / "ready.json").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        ready = json.loads((output / "ready.json").read_text())
        assert ready["host"] == "127.0.0.1" and ready["invocation"] == invocation
        assert (output / "ready.json").stat().st_mode & 0o077 == 0
        for index in range(2):
            _request((ready["host"], ready["port"]), ready["token"], index)
        if end == "parent_stop":
            stop = {
                "schema": "decision3-candidate-browser-stop-v1",
                "invocation": invocation,
                "plan_sha256": sha,
            }
            child._publish_control(Path(plan["stop_path"]), child._encode(stop))
        thread.join(timeout=5)
        assert not thread.is_alive()
        if audit_invalid:
            assert len(errors) == 1 and "synthetic numeric inconsistency" in str(errors[0])
            assert not result and len(audited) == 1
            assert semantic_binding_context.behavior["adapter_closed"]
            return
        assert not errors and len(audited) == 2
        assert len(result[0]["numeric_audits"]) == 2
        assert result[0]["request_calls"] == 2
        assert result[0]["stop_reason"] == (
            "requested" if end == "parent_stop" else "request_limit"
        )
        assert len(result[0]["private_bindings"]) == len(result[0]["witness_receipts"]) == 2
        for record in result[0]["private_bindings"]:
            raw = (output / record["path"]).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == record["sha256"]
            assert json.loads(raw)["status"] == "prepared_not_delivered"
    finally:
        if thread.is_alive():
            stop_path = Path(plan["stop_path"])
            if not stop_path.exists():
                child._publish_control(
                    stop_path,
                    child._encode(
                        {
                            "schema": "decision3-candidate-browser-stop-v1",
                            "invocation": invocation,
                            "plan_sha256": sha,
                        }
                    ),
                )
            thread.join(timeout=5)


@pytest.mark.parametrize(
    ("thread_alive", "reason"),
    [(True, "listener blocked"), (False, "lease still held"), (False, "socket still open")],
)
def test_failed_service_cleanup_preserves_primary_and_refuses_adapter_cleanup(
    tmp_path, monkeypatch, semantic_binding_context, thread_alive, reason
):
    monkeypatch.setattr(child, "SOURCE_ONLY", False)
    plan = plan_for(tmp_path)
    semantic_binding_context.lease.held = semantic_binding_context.lease.fatal = None
    monkeypatch.setattr(child.common, "_shared_identity", lambda *_args: None)
    monkeypatch.setattr(child.common, "_verify_source_pins", lambda *_args: None)
    monkeypatch.setattr(child.semantic_owner, "_load_deltas", lambda *_args: {})
    monkeypatch.setattr(
        child.semantic_owner, "_load_vocabulary", lambda *_args: ["CLICK", "TYPE_TEXT"]
    )
    monkeypatch.setattr(features, "_validate_manifest", lambda *_args: None)
    monkeypatch.setattr(
        audit,
        "make_witness_writer",
        lambda *_args, **_kwargs: (lambda **_values: None, semantic_binding_context.receipts),
    )
    monkeypatch.setattr(
        audit,
        "audit_directory",
        lambda *_args, **_kwargs: {
            "status": "numeric_consistency_only",
            "browser_qualified": False,
        },
    )
    lease_module = SimpleNamespace(
        ForwardLease=lambda *_args, **_kwargs: semantic_binding_context.lease
    )
    binding_module = SimpleNamespace(
        SemanticBrowserBinding=lambda *_args, **_kwargs: semantic_binding_context.make(
            max_requests=2
        )
    )
    primary = TimeoutError(reason) if thread_alive else None

    class BlockedLifecycle:
        def __init__(self, *_args, **kwargs):
            self.done = threading.Event()
            self.done.set()
            self.thread = SimpleNamespace(is_alive=lambda: thread_alive)
            self.server = SimpleNamespace(fileno=lambda: 3 if reason == "socket still open" else -1)
            self.cleanup_finished = False
            self.closure_guard = kwargs["closure_guard"]
            if reason == "lease still held":
                semantic_binding_context.lease.held = object()

        def start(self):
            return "127.0.0.1", 12345

        def stop(self, **_kwargs):
            if primary is not None:
                raise primary
            self.closure_guard()

    Path(plan["stop_path"]).write_text(
        json.dumps(
            {
                "schema": "decision3-candidate-browser-stop-v1",
                "invocation": "f" * 32,
                "plan_sha256": "d" * 64,
            }
        )
    )
    expected_error = TimeoutError if thread_alive else ValueError
    with pytest.raises(expected_error) as caught:
        child._serve(
            plan,
            "d" * 64,
            "f" * 32,
            semantic_binding_context.base,
            object(),
            (
                http,
                lease_module,
                binding_module,
                SimpleNamespace(CandidateServiceLifecycle=BlockedLifecycle),
                object(),
                audit,
            ),
            Path(plan["output_dir"]),
            lambda: None,
        )
    if primary is not None:
        assert caught.value is primary
    assert getattr(caught.value, "candidate_cleanup_unconfirmed") is True
    assert not semantic_binding_context.behavior["adapter_closed"]


def test_child_handshake_skips_backend_and_cuda_cleanup_for_live_listener(tmp_path, monkeypatch):
    import s1.evaluation.decision3 as decision3

    monkeypatch.setattr(child, "SOURCE_ONLY", False)
    monkeypatch.setenv("INVOCATION_ID", "a" * 32)
    plan = plan_for(tmp_path)
    sha = "d" * 64
    monkeypatch.setattr(child, "_load_plan", lambda *_args: plan)
    monkeypatch.setattr(child.common, "_own_cgroup", lambda *_args: None)
    monkeypatch.setattr(child.common, "_verify_source_pins", lambda *_args: None)
    monkeypatch.setattr(child.common, "_validate_interpreter", lambda *_args: None)
    monkeypatch.setattr(child.common, "_bind_project_source", lambda *_args: None)
    monkeypatch.setattr(child.common, "_shared_identity", lambda *_args: None)
    monkeypatch.setattr(child.common.systemd, "process_identity", lambda *_args: {"startticks": 1})
    monkeypatch.setattr(child.common.systemd, "boot_id", lambda: "boot")
    monkeypatch.setattr(child, "_runtime_modules", lambda *_args: ())
    monkeypatch.setattr(child, "_require_actual_sources", lambda *_args: None)
    monkeypatch.setattr(features, "_validate_manifest", lambda *_args: None)
    calls = []

    class Provider:
        runtime = object()

        def close(self):
            calls.append("backend.close")

    monkeypatch.setattr(decision3, "Decision3Backend", lambda *_args: Provider())
    monkeypatch.setattr(decision3, "decision3_spec", lambda *_args, **_kwargs: object())
    error = TimeoutError("active listener")
    error.candidate_cleanup_unconfirmed = True
    monkeypatch.setattr(child, "_serve", lambda *_args: (_ for _ in ()).throw(error))
    cuda = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_properties=lambda _index: SimpleNamespace(
            uuid=plan["gpu_uuid"], major=12, minor=1
        ),
        is_bf16_supported=lambda **_kwargs: True,
        synchronize=lambda *_args: calls.append("sync"),
        memory_allocated=lambda _index: 0,
        empty_cache=lambda: calls.append("empty_cache"),
    )
    torch = SimpleNamespace(
        set_num_threads=lambda *_args: None,
        set_num_interop_threads=lambda *_args: None,
        set_float32_matmul_precision=lambda *_args: None,
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True))),
        cuda=cuda,
        _C=SimpleNamespace(),
    )
    monkeypatch.setattr(child.common, "_bind_overlay", lambda *_args: torch)
    output = Path(plan["output_dir"])
    errors = []

    def run():
        try:
            child.run(tmp_path / "plan", sha)
        except BaseException as caught:
            errors.append(caught)

    worker = threading.Thread(target=run)
    worker.start()
    deadline = time.monotonic() + 3
    while not (output / "hello.json").exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    hello = json.loads((output / "hello.json").read_text())
    assert hello["invocation"] == "a" * 32 and hello["plan_sha256"] == sha
    child._publish_control(
        output / "go.json",
        child._encode(
            {
                "schema": "decision3-candidate-browser-go-v1",
                "invocation": "a" * 32,
                "plan_sha256": sha,
            }
        ),
    )
    worker.join(timeout=5)
    assert not worker.is_alive() and len(errors) == 1 and errors[0] is error
    terminal = json.loads((output / "terminal.json").read_text())
    assert terminal["status"] == "failed_cleanup_unconfirmed"
    assert "final_cuda_allocated" not in terminal
    assert calls == ["sync"]


@pytest.mark.parametrize(
    "name", ["hello.json", "ready.json", "terminal.json", "go.json", "stop.json"]
)
def test_control_publication_is_atomic_during_delayed_write(tmp_path, monkeypatch, name):
    started = threading.Event()
    complete = threading.Event()
    target = tmp_path / name
    errors = []

    def delayed_write(stream, data):
        stream.write(data[:1])
        stream.flush()
        started.set()
        if not complete.wait(5):
            raise TimeoutError("test write did not resume")
        stream.write(data[1:])
        stream.flush()
        os.fsync(stream.fileno())

    monkeypatch.setattr(child, "_write_control_bytes", delayed_write)

    def publish():
        try:
            child._publish_control(target, b'{"complete":true}')
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=publish)
    thread.start()
    try:
        assert started.wait(5)
        assert not target.exists()
    finally:
        complete.set()
        thread.join(5)
    assert not thread.is_alive()
    assert not errors
    assert json.loads(target.read_bytes()) == {"complete": True}
    assert target.stat().st_mode & 0o077 == 0
    assert list(tmp_path.iterdir()) == [target]


def test_control_publication_does_not_overwrite_collision(tmp_path):
    target = tmp_path / "ready.json"
    target.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        child._publish_control(target, b"new")
    assert target.read_bytes() == b"existing"
    assert list(tmp_path.iterdir()) == [target]


def test_control_temp_collision_preserves_unowned_file(tmp_path, monkeypatch):
    monkeypatch.setattr(child.secrets, "token_hex", lambda _: "a" * 32)
    temporary = tmp_path / (".hello.json." + "a" * 32 + ".tmp")
    temporary.write_bytes(b"unowned")
    with pytest.raises(FileExistsError):
        child._publish_control(tmp_path / "hello.json", b"new")
    assert temporary.read_bytes() == b"unowned"
    assert not (tmp_path / "hello.json").exists()


def test_binding_construction_failure_requires_external_termination(
    tmp_path, monkeypatch, semantic_binding_context
):
    monkeypatch.setattr(child, "SOURCE_ONLY", False)
    plan = plan_for(tmp_path)
    monkeypatch.setattr(child.common, "_shared_identity", lambda *_: None)
    monkeypatch.setattr(child.semantic_owner, "_load_deltas", lambda *_: {})
    monkeypatch.setattr(child.semantic_owner, "_load_vocabulary", lambda *_: ["CLICK"])
    monkeypatch.setattr(audit, "make_witness_writer", lambda *_a, **_k: (lambda **_: None, []))
    failure = RuntimeError("synthetic constructor failure")

    def failed(*_args, **_kwargs):
        raise failure

    lease_module = SimpleNamespace(ForwardLease=lambda *_a, **_k: semantic_binding_context.lease)
    binding_module = SimpleNamespace(SemanticBrowserBinding=failed)
    with pytest.raises(RuntimeError, match="constructor failure") as caught:
        child._serve(
            plan,
            "d" * 64,
            "f" * 32,
            semantic_binding_context.base,
            object(),
            (http, lease_module, binding_module, lifecycle, object(), audit),
            Path(plan["output_dir"]),
            lambda: None,
        )
    assert caught.value is failure
    assert caught.value.candidate_cleanup_unconfirmed is True


def test_no_success_on_zero_completed_calls(tmp_path, monkeypatch, semantic_binding_context):
    semantic_binding_context.lease.fatal = None
    monkeypatch.setattr(child, "SOURCE_ONLY", False)
    plan = plan_for(tmp_path)
    monkeypatch.setattr(child.common, "_shared_identity", lambda *_: None)
    monkeypatch.setattr(child.semantic_owner, "_load_deltas", lambda *_: {})
    monkeypatch.setattr(child.semantic_owner, "_load_vocabulary", lambda *_: ["CLICK"])
    monkeypatch.setattr(audit, "make_witness_writer", lambda *_a, **_k: (lambda **_: None, []))
    lease_module = SimpleNamespace(ForwardLease=lambda *_a, **_k: semantic_binding_context.lease)
    binding_module = SimpleNamespace(
        SemanticBrowserBinding=lambda *_a, **_k: semantic_binding_context.make()
    )

    class IdleLifecycle:
        def __init__(self, *_args, **kwargs):
            self.done = threading.Event()
            self.done.set()
            self.thread = SimpleNamespace(is_alive=lambda: False)
            self.server = SimpleNamespace(fileno=lambda: -1)
            self.cleanup_finished = False
            self.closure_guard, self.cleanup = kwargs["closure_guard"], kwargs["cleanup"]

        def start(self):
            return "127.0.0.1", 12345

        def stop(self, **_kwargs):
            self.closure_guard()
            self.cleanup()
            self.cleanup_finished = True
            return {"status": "stopped", "reason": "requested", "request_calls": 0}

    with pytest.raises(ValueError, match="completed request, witness"):
        child._serve(
            plan,
            "d" * 64,
            "f" * 32,
            semantic_binding_context.base,
            object(),
            (
                http,
                lease_module,
                binding_module,
                SimpleNamespace(CandidateServiceLifecycle=IdleLifecycle),
                object(),
                audit,
            ),
            Path(plan["output_dir"]),
            lambda: None,
        )
    assert semantic_binding_context.behavior["adapter_closed"]


def test_explicit_semantic_releases_are_loaded_through_reviewed_gate(tmp_path, monkeypatch):
    monkeypatch.setattr(child, "SOURCE_ONLY", False)
    plan = plan_for(tmp_path)
    found = []

    def load(path, held, pins, name):
        found.append((path, held, pins, name))
        return name

    monkeypatch.setattr(child.common, "_load_reviewed_release", load)
    modules = child._runtime_modules(plan)
    assert len(modules) == 6
    assert [row[1] for row in found] == [
        "jev_decision3_candidate_http.py",
        "jev_gpu_forward_lease.py",
        "jev_decision3_semantic_browser_binding.py",
        "jev_decision3_candidate_service_lifecycle.py",
        "jev_decision3_semantic_operation_adapter.py",
        "jev_decision3_semantic_operation_audit.py",
    ]
    assert all(row[2] is plan["source_pins"] for row in found)


def test_actual_semantic_sources_must_be_in_explicit_pin_inventory(tmp_path, monkeypatch):
    import jev_decision3_link_navigation_features as features

    plan = plan_for(tmp_path)
    fake_path = str(Path(__file__).absolute())
    reference = SimpleNamespace(__file__=fake_path)
    adapter = SimpleNamespace(__file__=fake_path, reference=reference)
    audit_module = SimpleNamespace(__file__=fake_path)
    binding_module = SimpleNamespace(__file__=fake_path)
    http_module = SimpleNamespace(wire=SimpleNamespace(__file__=fake_path), DeadlineBackend=object)
    backend = SimpleNamespace(model=object(), runtime=object())
    modules = (http_module, object(), binding_module, object(), adapter, audit_module)
    for value in (
        child.__file__,
        child.common.__file__,
        child.semantic_owner.__file__,
        features.__file__,
        fake_path,
    ):
        plan["source_pins"][str(Path(value).absolute())] = "0" * 64
    monkeypatch.setattr(child.inspect, "getfile", lambda *_: fake_path)
    monkeypatch.setattr(child.common, "_verify_source_pins", lambda *_: None)
    child._require_actual_sources(plan, backend, modules)
    del plan["source_pins"][str(Path(child.semantic_owner.__file__).absolute())]
    with pytest.raises(ValueError, match="actual owner, candidate"):
        child._require_actual_sources(plan, backend, modules)


def test_real_semantic_stack_and_witness_audit_through_child(
    tmp_path, monkeypatch, semantic_binding_context, synthetic
):
    """End-to-end invented CPU tensors; no browser, GPU or real lease is started."""
    import httpx
    import jev_browser_private_transport_capture as capture
    import jev_decision3_semantic_browser_binding as binding
    import jev_decision3_semantic_operation_adapter as semantic
    import jev_decision3_semantic_pair_evidence as pair_evidence
    import numpy as np
    from test_jev_decision3_operation_gpu import RequestDTO, Tensor, Torch
    from test_jev_decision3_semantic_operation_adapter import fake_sha

    import s1.jev_decision3_native as native
    from s1.evaluation.decision3 import native_payload

    _, provider, deltas, hashes, _, _, _ = synthetic
    monkeypatch.setattr(child, "SOURCE_ONLY", False)
    monkeypatch.setattr(capture, "SOURCE_ONLY", False)
    monkeypatch.setattr(pair_evidence, "SOURCE_ONLY", False)
    monkeypatch.setattr(lifecycle, "SOURCE_ONLY", False)
    monkeypatch.setattr(semantic, "Decision3NativeRequest", native.Decision3NativeRequest)
    monkeypatch.setattr(semantic, "native_payload", native_payload)
    monkeypatch.setattr(native, "configuration_blockers", lambda _: [])
    monkeypatch.setattr(audit, "WIDTH", 2)
    monkeypatch.setattr(audit, "CODES", 3)
    monkeypatch.setattr(audit, "W0_SHA256", fake_sha(provider.runtime.readout))
    monkeypatch.setattr(
        Tensor, "numpy", lambda self: np.asarray(self.values, dtype=np.float32), raising=False
    )
    monkeypatch.setattr(features, "_validate_manifest", lambda *_: None)
    monkeypatch.setattr(child.common, "_shared_identity", lambda *_: None)
    monkeypatch.setattr(child.common, "_verify_source_pins", lambda *_: None)
    monkeypatch.setattr(child.semantic_owner, "_load_deltas", lambda *_: deltas)
    names = ["CLICK", "TYPE_TEXT"]
    monkeypatch.setattr(child.semantic_owner, "_load_vocabulary", lambda *_: names)
    monkeypatch.setattr(
        child, "_lease_summary", lambda _transport, _pid, count: {"own_completed_pairs": count}
    )
    original_predict = provider.predict
    provider.predict = lambda dto, **kwargs: original_predict(
        RequestDTO.model_validate(dto.model_dump()), **kwargs
    )
    plan = plan_for(tmp_path)
    plan["max_requests"] = 1
    plan["vocabulary"]["names_sha256"] = hashlib.sha256(binding._encode(names)).hexdigest()
    for head in hashes:
        plan["deltas"][head]["fp32_sha256"] = hashes[head]

    class Lease(semantic_binding_context.module.ForwardLease):
        def __init__(self, *_args, **_kwargs):
            super().__init__()
            self.fatal = None

    leases = SimpleNamespace(
        SOURCE_ONLY=False,
        ForwardLease=Lease,
        SerialBackend=semantic_binding_context.module.SerialBackend,
    )
    output = Path(plan["output_dir"])
    result, errors = [], []

    def run():
        try:
            result.append(
                child._serve(
                    plan,
                    "d" * 64,
                    "f" * 32,
                    provider,
                    Torch,
                    (http, leases, binding, lifecycle, semantic, audit),
                    output,
                    lambda: None,
                )
            )
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        deadline = time.monotonic() + 4
        while not (output / "ready.json").exists() and time.monotonic() < deadline:
            if errors:
                raise errors[0]
            time.sleep(0.01)
        ready = json.loads((output / "ready.json").read_bytes())
        request = payload(2)
        request["questions"]["operation"]["criteria"] = {
            k: v for k, v in request["questions"]["operation"]["criteria"].items() if k in names
        }
        request["questions"] = {
            k: v
            for k, v in request["questions"].items()
            if k in {"operation", "click_target", "type_text_target"}
        }
        body = json.dumps(request).encode()
        capture_dir = tmp_path / "client-capture"
        capture_dir.mkdir()
        (tmp_path / "owned-client-closure.json").write_text(
            json.dumps(
                {
                    "confirmed": True,
                    "remaining": [],
                    "tracked_PID_start_ticks": {str(os.getpid()): 1},
                }
            )
        )

        def sink(event):
            attempt = capture_dir / f"attempt-{event['sequence']:04d}"
            attempt.mkdir()
            record = {
                k: v for k, v in event.items() if k not in {"request_bytes", "response_bytes"}
            }
            record["pid"] = os.getpid()
            for key in ("request_bytes", "response_bytes"):
                raw = event[key]
                (attempt / f"{key}.bin").write_bytes(raw)
                record[key] = {"sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
            (attempt / "record.json").write_text(json.dumps(record))

        with httpx.Client(trust_env=False) as client:
            with capture.capture_post(client, sink=sink, secrets=(ready["token"],)):
                response = client.post(
                    f"http://{ready['host']}:{ready['port']}/v1/systemone",
                    content=body,
                    headers={"Authorization": "Bearer " + ready["token"]},
                )
        response_bytes = response.content
        assert response.status_code == 200
        thread.join(timeout=5)
        assert not thread.is_alive() and not errors
        assert len(result[0]["numeric_audits"]) == 1
        assert result[0]["numeric_audits"][0]["status"] == "numeric_consistency_only"
        assert provider.runtime.forward_calls == 1
        record = json.loads((output / result[0]["private_bindings"][0]["path"]).read_bytes())
        assert (
            binding.verify_client_exchange(record, body, response_bytes)["status"]
            == "provided_bytes_match"
        )
        parent = pair_evidence.verify(
            service_dir=output,
            capture_dir=capture_dir,
            plan=plan,
            terminal={"service": result[0]},
            audit_module=audit,
            binding_module=binding,
            baseline_sha256=fake_sha(provider.runtime.readout),
            runner_policy_calls=1,
            runner_text_calls=0,
        )
        assert parent["status"] == "provided_bytes_and_fp64_match"
        assert parent["matched_sequences"] == [1] and parent["browser_success"] is None
    finally:
        if thread.is_alive():
            stop = output / "stop.json"
            if not stop.exists():
                child._publish_control(
                    stop,
                    child._encode(
                        {
                            "schema": "decision3-candidate-browser-stop-v1",
                            "invocation": "f" * 32,
                            "plan_sha256": "d" * 64,
                        }
                    ),
                )
            thread.join(timeout=5)
