"""Pure service composition tests: no sockets, Modal, framework or model load."""

import ast
import copy
import hashlib
import importlib.util
import io
import json
import sys
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
        # Keep the deployment's named-source bytes and digests unchanged while
        # pure tests use this checkout instead of Modal's copied absolute paths.
        deployed_root = module.ROOT
        for constant_name, value in list(vars(module).items()):
            if isinstance(value, Path) and value.is_relative_to(deployed_root):
                setattr(module, constant_name, ROOT / value.relative_to(deployed_root))
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
    gateway.active, gateway.nonce = "d3-edge", "nonce-test-only"
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
    monkeypatch.setattr(
        SERVICE, "gpu_identity", lambda: {"name": "H200", "uuid": "GPU-replacement"}
    )
    with pytest.raises(ValueError, match="nonce"):
        session.check("wrong")
    with pytest.raises(ValueError, match="UUID changed"):
        session.check("nonce")
    assert session.blocked and session.identity_errors
    assert session.check("nonce", cleanup=True) is None


@pytest.mark.parametrize("fault", [None, "unknown_exit", "nonzero_exit", "failed_worker"])
def test_next_arm_requires_successful_worker_and_confirmed_whole_exit(fault):
    session = object.__new__(SERVICE.Session)
    session.check = lambda *_args, **_kwargs: None
    session.process = SimpleNamespace(poll=lambda: 0)
    session.active, session.blocked, session.next_arm = "d3", False, 0
    session.nonce, session.gpu, session.closures = "nonce", {"uuid": "GPU-same"}, []
    session.read_control = lambda: {
        "event": "closed",
        "success": fault != "failed_worker",
        "campaign_stop": fault == "failed_worker",
    }
    session.record = lambda *_args: None

    def reap():
        if fault == "unknown_exit":
            raise RuntimeError("Unknown marked session remains")
        session.last_returncode = 1 if fault == "nonzero_exit" else 0
        session.process = None
        return True

    session.reap = reap
    result = session.end("nonce")
    assert result["success"] is (fault is None)
    assert session.next_arm == (1 if fault is None else 0)
    assert session.blocked is (fault is not None)
    if fault:
        with pytest.raises(ValueError, match="next ordered arm"):
            session.begin("d3-flash", "nonce")


def test_strict_json_rejects_duplicate_and_overflow_numbers_before_rpc(tmp_path):
    gateway, _, backend, *_ = composition(tmp_path)
    status, response = gateway.request(b'{"model":"d3-edge","state":1e999}')
    assert status == 504 and "answers" not in json.loads(response)
    assert backend.calls == [] and gateway.blocked


def test_sdk_checkpoint_failure_returns_available_raw_without_action(tmp_path):
    session = object.__new__(SERVICE.Session)
    session.nonce, session.active, session.gpu = "nonce", "d3-edge", {"uuid": "GPU-same"}
    session.blocked, session.last_control = False, None
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
