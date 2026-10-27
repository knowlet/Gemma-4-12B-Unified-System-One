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

import jev_decision3_candidate_browser_child as child
import jev_decision3_candidate_http as http
import jev_decision3_candidate_service_lifecycle as lifecycle
import jev_decision3_link_navigation_adapter_audit as audit
import jev_decision3_link_navigation_features as features
import pytest
from test_jev_decision3_native import payload

pytest_plugins = ("test_jev_decision3_candidate_browser_binding",)


def plan_for(tmp_path):
    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    transport = {
        key: str(tmp_path / (key + ".py"))
        for key in ("http_release", "lease_release", "binding_release", "lifecycle_release")
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
def test_multiple_loopback_calls_ready_stop_and_private_records(
    tmp_path, monkeypatch, candidate_binding_context, end
):
    monkeypatch.setattr(child, "SOURCE_ONLY", False)
    monkeypatch.setattr(lifecycle, "SOURCE_ONLY", False)
    plan = plan_for(tmp_path)
    if end == "parent_stop":
        plan["max_requests"] = 3
    candidate_binding_context.lease.held = candidate_binding_context.lease.fatal = None
    monkeypatch.setattr(child.common, "_shared_identity", lambda *_args: None)
    monkeypatch.setattr(child.common, "_verify_source_pins", lambda *_args: None)
    monkeypatch.setattr(child.common, "_load_deltas", lambda *_args: {})
    monkeypatch.setattr(
        child, "_lease_summary", lambda _transport, _pid, count: {"own_completed_pairs": count}
    )
    monkeypatch.setattr(features, "_validate_manifest", lambda *_args: None)
    monkeypatch.setattr(
        audit,
        "make_witness_writer",
        lambda *_args, **_kwargs: (lambda **_values: None, candidate_binding_context.receipts),
    )
    lease_module = SimpleNamespace(
        ForwardLease=lambda *_args, **_kwargs: candidate_binding_context.lease
    )

    class BindingModule:
        @staticmethod
        def CandidateBrowserBinding(*_args, **kwargs):
            instance = candidate_binding_context.make(max_requests=plan["max_requests"])
            original_sink = instance.service.event_sink

            def event(row):
                original_sink(row)
                if row["status"] == "prepared":
                    kwargs["private_artifact_sink"](candidate_binding_context.bindings[-1])
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
                    candidate_binding_context.base,
                    object(),
                    (http, lease_module, BindingModule, lifecycle),
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
        assert not thread.is_alive() and not errors
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
    tmp_path, monkeypatch, candidate_binding_context, thread_alive, reason
):
    monkeypatch.setattr(child, "SOURCE_ONLY", False)
    plan = plan_for(tmp_path)
    candidate_binding_context.lease.held = candidate_binding_context.lease.fatal = None
    monkeypatch.setattr(child.common, "_shared_identity", lambda *_args: None)
    monkeypatch.setattr(child.common, "_verify_source_pins", lambda *_args: None)
    monkeypatch.setattr(child.common, "_load_deltas", lambda *_args: {})
    monkeypatch.setattr(features, "_validate_manifest", lambda *_args: None)
    monkeypatch.setattr(
        audit,
        "make_witness_writer",
        lambda *_args, **_kwargs: (lambda **_values: None, candidate_binding_context.receipts),
    )
    lease_module = SimpleNamespace(
        ForwardLease=lambda *_args, **_kwargs: candidate_binding_context.lease
    )
    binding_module = SimpleNamespace(
        CandidateBrowserBinding=lambda *_args, **_kwargs: candidate_binding_context.make(
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
                candidate_binding_context.lease.held = object()

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
            candidate_binding_context.base,
            object(),
            (
                http,
                lease_module,
                binding_module,
                SimpleNamespace(CandidateServiceLifecycle=BlockedLifecycle),
            ),
            Path(plan["output_dir"]),
            lambda: None,
        )
    if primary is not None:
        assert caught.value is primary
    assert getattr(caught.value, "candidate_cleanup_unconfirmed") is True
    assert not candidate_binding_context.behavior["adapter_closed"]


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
