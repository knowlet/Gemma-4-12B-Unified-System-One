"""Owned CPU loopback fakes, synthetic source metadata; never real Q95/model calls."""

import copy
import hashlib
import importlib.util
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).absolute().parent
REPO = HERE.parents[2]
spec = importlib.util.spec_from_file_location("_q95_live_test", HERE / "live_helper.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class SafeFixture(dict):
    def __repr__(self):
        return "<owned CPU loopback fixture; private credential redacted>"


def no_private_data(token, blob):
    if token.encode() in blob:
        raise AssertionError("Private data appeared in public evidence")


def json_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(m.encode(value))
    path.chmod(0o600)


@pytest.fixture
def fake(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(m, "ROOT", root)
    source = root / "frozen"
    source.mkdir()
    entries = {}
    for i in range(94):
        p = source / f"source-{i:03d}.py"
        p.write_text("# synthetic source, never imported\n")
        entries[p.name] = m.pin(p)
    script = source / "scripts/jev_qualify_public_field_syntax_bf16.py"
    script.parent.mkdir()
    script.write_bytes((HERE / "fake_helper.py").read_bytes())
    entries[script.relative_to(source).as_posix()] = m.pin(script)
    runtime_entries = {}
    runtime_root = root / "runtime"
    runtime_root.mkdir()
    for i in range(348):
        p = runtime_root / f"file-{i:03d}.py"
        p.write_text("# CPU fixture only\n")
        runtime_entries[str(p)] = m.pin(p)
    selection_root = root / "selected"
    execution_path, runtime_path, qpath = [
        selection_root / n for n in ("execution.json", "runtime.json", "qualification.json")
    ]
    json_file(execution_path, {"files": entries})
    json_file(runtime_path, {"files": runtime_entries})
    monkeypatch.setattr(m, "E_SHA", m.pin(execution_path)["sha256"])
    monkeypatch.setattr(m, "R_SHA", m.pin(runtime_path)["sha256"])
    json_file(
        qpath,
        {
            "qualified": True,
            "task_performance_attested": False,
            "kind": "jev_public_action_field_syntax_bf16_qualification_v1",
            "execution_sha256": m.E_SHA,
            "runtime_sha256": m.R_SHA,
            "plan_sha256": "a" * 64,
            "inputs_sha256": "b" * 64,
            "public": {"synthetic_fixture": True},
        },
    )
    monkeypatch.setattr(m, "Q_SHA", m.pin(qpath)["sha256"])
    registry = root / "registry.py"
    shutil.copyfile(REPO / "src/s1/evaluation/decision3.py", registry)

    def need_ref(path):
        return {"path": path.relative_to(root).as_posix(), **m.pin(path)}

    service = root / "artifacts/jev-runtime/fake-helper"
    service.mkdir(parents=True, mode=0o700)
    token = secrets.token_hex(32)
    config_path = service / "controller-config.json"
    boot = m.boot()
    total = 130595864576
    config = {
        "token": token,
        "command": "serve",
        "output": str(service),
        "boot_id": boot,
        "mem_total": total,
        "qualification_sha256": m.Q_SHA,
        "execution_sha256": m.E_SHA,
        "runtime_sha256": m.R_SHA,
        "plan_sha256": "a" * 64,
        "inputs_sha256": "b" * 64,
        "public": {"synthetic_fixture": True},
        "qualification_path": str(selection_root),
        "execution_path": str(execution_path),
        "runtime_path": str(runtime_path),
    }
    json_file(config_path, config)
    monkeypatch.setenv("S1_POLICY_API_KEY", token)
    monkeypatch.setenv("S1_TEXT_API_KEY", token)
    interpreter = str(Path(sys.executable).resolve())
    child = subprocess.Popen(
        [interpreter, str(script), "--internal-child", str(config_path)],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        until = time.monotonic() + 3
        while not (service / "endpoint.json").is_file():
            assert child.poll() is None
            assert time.monotonic() < until
            time.sleep(0.005)
        endpoint = json.loads((service / "endpoint.json").read_bytes())["helper"]
        actual = m.process(child.pid)
        public = {
            "profile": m.PROFILE,
            "qualification_sha256": m.Q_SHA,
            "source_count": 95,
            "boot_id": boot,
            **{k: actual[k] for k in ("pid", "start_ticks", "pgid", "sid")},
            "endpoint": endpoint,
            "seconds": 20,
            "tokens": 128,
            "context": 8192,
        }
        value = {
            "schema_id": m.SCHEMA,
            "binding": public,
            "worker": {
                k: actual[k]
                for k in ("pid", "start_ticks", "pgid", "sid", "boot_id", "exe", "argv", "cgroup")
            },
            "listener": m.listener(child.pid, int(endpoint.rsplit(":", 1)[1])),
            "service_directory": str(service),
            "private_config": str(config_path),
            "interpreter": interpreter,
            "source_root": str(source),
            "qualification": need_ref(qpath),
            "execution": need_ref(execution_path),
            "runtime": need_ref(runtime_path),
            "registry": need_ref(registry),
            "expected_memtotal": total,
            "co_resident": {
                "model": "d3-edge",
                "resident_file_bytes": 1222198784,
                "cgroup": m.process(os.getpid())["cgroup"],
            },
        }
        # Use the exact registry's selected floor rather than a handpicked test value.
        import ast

        tree = ast.parse(registry.read_bytes())
        models = next(
            ast.literal_eval(n.value)
            for n in tree.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "CHECKPOINTS" for t in n.targets)
        )
        value["co_resident"]["resident_file_bytes"] = models["d3-edge"]["resident_file_bytes"]
        monkeypatch.setattr(
            m,
            "host_memory",
            lambda: {"boot_id": boot, "MemTotal": total, "MemAvailable": 120000000000},
        )
        monkeypatch.setattr(
            m,
            "cgroup_memory",
            lambda _g: [
                {
                    "path": "synthetic-memory",
                    "limit": 80 * m.GIB,
                    "current": m.GIB,
                    "events": {"oom": 0, "oom_kill": 0},
                }
            ],
        )
        receipts = root / "receipts"
        receipts.mkdir()

        def build(selected=None):
            blob = m.encode(value if selected is None else selected)
            return m.LiveHelperChecker(
                selection_blob=blob,
                selection_sha=hashlib.sha256(blob).hexdigest(),
                receipt_directory=receipts,
            )

        yield SafeFixture(
            {
                "value": value,
                "build": build,
                "receipts": receipts,
                "token": token,
                "child": child,
                "config": config,
                "config_path": config_path,
                "source": source,
                "service": service,
                "models": models,
            }
        )
    finally:
        if child.poll() is None:
            child.terminate()
        try:
            child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=2)


def test_actual_owned_loopback_process_private_config_and_callback_receipt(fake):
    checker = fake["build"]()
    result = checker(checker.binding, "before-cache-and-framework")
    assert set(result) == {
        "binding",
        "phase",
        "helper_admitted",
        "co_resident_admitted",
        "receipt_sha256",
    }
    receipt_path = fake["receipts"] / "helper-check-0000.json"
    blob = receipt_path.read_bytes()
    receipt = json.loads(blob)
    assert hashlib.sha256(blob).hexdigest() == result["receipt_sha256"]
    assert result["helper_admitted"] and result["co_resident_admitted"]
    assert [r["status"] for r in receipt["authentication"]["requests"]] == [500, 422]
    assert receipt["authentication"]["inference_requested"] is False
    no_private_data(fake["token"], blob)
    no_private_data(fake["token"], " ".join(m.process(fake["child"].pid)["argv"]).encode())
    assert receipt["whole_tree_exit_confirmed"] is False
    assert receipt["continuation_authorized"] is False
    assert receipt["qualified_model"] is False


def test_real_schema_delay_2_point_4_seconds_uses_remaining_shared_three_second_budget(fake):
    json_file(fake["service"] / "fake-control.json", {"selected_schema_delay_seconds": 2.4})
    checker = fake["build"]()
    started = time.monotonic()
    result = checker(checker.binding, "delayed-owned-schema")
    elapsed = time.monotonic() - started
    receipt = json.loads((fake["receipts"] / "helper-check-0000.json").read_bytes())
    assert m.CALL_SECONDS == 3.0
    assert 2.4 <= elapsed < 3.0
    assert result["helper_admitted"] and result["co_resident_admitted"]
    assert receipt["authentication"]["requests"][1]["status"] == 422
    assert receipt["authentication"]["requests"][1]["handler_evidence"] is not None
    assert receipt["authentication"]["inference_requested"] is False
    no_private_data(fake["token"], m.encode(receipt))


def test_real_schema_delay_beyond_shared_three_seconds_never_admits(fake):
    json_file(fake["service"] / "fake-control.json", {"selected_schema_delay_seconds": 3.2})
    checker = fake["build"]()
    started = time.monotonic()
    with pytest.raises((TimeoutError, m.BindingError)):
        checker(checker.binding, "late-owned-schema")
    elapsed = time.monotonic() - started
    receipt = json.loads((fake["receipts"] / "helper-check-0000.json").read_bytes())
    assert m.CALL_SECONDS == 3.0 and 2.9 <= elapsed < 4.0
    assert receipt["helper_admitted"] is receipt["co_resident_admitted"] is False
    assert receipt["error_type"] in {"TimeoutError", "BindingError"}
    no_private_data(fake["token"], m.encode(receipt))


def test_text_connection_is_explicit_memory_only_and_public_binding_is_copy(fake):
    checker = fake["build"]()
    public = checker.binding
    public["tokens"] = 999
    assert checker.binding["tokens"] == 128
    connection = checker.text_connection()
    assert connection["endpoint"] == checker.binding["endpoint"]
    assert connection["model"] == "generated-actions-none"
    if not secrets.compare_digest(connection["api_key"], fake["token"]):
        raise AssertionError("Private connection credential mismatch")
    no_private_data(fake["token"], next(fake["receipts"].iterdir()).read_bytes())


@pytest.mark.parametrize("field", ["start_ticks", "pgid", "sid", "boot_id"])
def test_wrong_selected_live_identity_refuses(fake, field):
    value = copy.deepcopy(fake["value"])
    if field == "boot_id":
        changed = "00000000-0000-4000-8000-000000000000"
        value["binding"][field] = value["worker"][field] = changed
    elif field == "start_ticks":
        value["binding"][field] += 1
        value["worker"][field] += 1
    else:
        value["worker"][field] += 1
    with pytest.raises(m.BindingError):
        checker = fake["build"](value)
        checker(checker.binding, "post-warmup")


@pytest.mark.parametrize(
    "kind", ["mode", "token", "qualified_sha", "stopping", "listener", "endpoint"]
)
def test_private_credential_profile_and_listener_mismatches_never_admit(fake, monkeypatch, kind):
    checker = fake["build"]()
    if kind == "mode":
        fake["config_path"].chmod(0o644)
    elif kind == "token":
        fake["config"]["token"] = secrets.token_hex(32)
        json_file(fake["config_path"], fake["config"])
    elif kind == "qualified_sha":
        fake["config"]["qualification_sha256"] = "0" * 64
        json_file(fake["config_path"], fake["config"])
    elif kind == "stopping":
        (fake["service"] / "shutdown.request").touch()
    elif kind == "listener":
        checker._value["listener"]["inode"] = "0"
    else:
        endpoint_path = fake["service"] / "endpoint.json"
        endpoint = json.loads(endpoint_path.read_bytes())
        endpoint["qualified"] = False
        json_file(endpoint_path, endpoint)
    with pytest.raises(m.BindingError, match="REFUSED"):
        checker(checker.binding, "post-warmup")
    receipt = next(fake["receipts"].iterdir()).read_bytes()
    no_private_data(fake["token"], receipt)
    assert json.loads(receipt)["helper_admitted"] is False


@pytest.mark.parametrize("control", ["ignore_credentials", "generation_instead_of_schema"])
def test_http_access_must_distinguish_auth_and_noninference_schema(fake, control):
    checker = fake["build"]()
    json_file(fake["service"] / "fake-control.json", {control: True})
    with pytest.raises(m.BindingError, match="AUTHENTICATED_SCHEMA_STATUS"):
        checker(checker.binding, "request-0")


@pytest.mark.parametrize(
    "control", ["unauthorized_creates_trace", "schema_without_handler_evidence"]
)
def test_generic_500_or_422_without_exact_handler_join_never_proves_authentication(fake, control):
    checker = fake["build"]()
    json_file(fake["service"] / "fake-control.json", {control: True})
    with pytest.raises(m.BindingError):
        checker(checker.binding, "request-0")


def test_source_drift_is_not_relabelled_or_admitted(fake):
    checker = fake["build"]()
    (fake["source"] / "source-000.py").write_text("# changed")
    with pytest.raises(m.BindingError, match="FROZEN_SOURCE95_DRIFT"):
        checker(checker.binding, "post-warmup")


def test_wrong_resident_floor_and_missing_sixth_model_cannot_admit(fake):
    value = copy.deepcopy(fake["value"])
    value["co_resident"]["resident_file_bytes"] -= 1
    with pytest.raises(m.BindingError, match="UNCHANGED_SELECTED_D3_RESIDENT_FLOOR"):
        fake["build"](value)
    value["co_resident"]["model"] = "unselected"
    with pytest.raises(m.BindingError, match="ALL_SIX_MODEL_POPULATION"):
        fake["build"](value)


def test_all_six_models_are_selected_and_full_model_capacity_refuses_before_http(fake, monkeypatch):
    value = copy.deepcopy(fake["value"])
    value["co_resident"].update(
        model="d3", resident_file_bytes=fake["models"]["d3"]["resident_file_bytes"]
    )
    checker = fake["build"](value)
    monkeypatch.setattr(checker, "_auth", lambda *_a: pytest.fail("capacity refusal entered HTTP"))
    with pytest.raises(m.BindingError, match="MEMORY_ADMISSION_REFUSED"):
        checker(checker.binding, "before-cache-and-framework")


def test_helper_host_floor_is_not_invented_inside_40gib_memcg():
    host = {"boot_id": "b", "MemTotal": 130595864576, "MemAvailable": 120000000000}
    helper = [{"limit": 40 * m.GIB, "current": 30 * m.GIB}]
    co = [{"limit": 80 * m.GIB, "current": m.GIB}]
    result = m.memory_decisions(host, helper, co, 1024**3)
    assert result["helper"]["admitted"] is True
    assert result["co_resident"]["admitted"] is True
    assert result["helper"]["required_host_available_bytes"] > helper[0]["limit"]
    assert result["swap_or_resident_credit_bytes"] == 0


def test_fresh_host_or_d3_cgroup_pressure_refuses(fake, monkeypatch):
    checker = fake["build"]()
    monkeypatch.setattr(
        m,
        "cgroup_memory",
        lambda _g: [
            {"path": "synthetic", "limit": 30 * m.GIB, "current": 29 * m.GIB, "events": {"oom": 0}}
        ],
    )
    with pytest.raises(m.BindingError, match="MEMORY_ADMISSION_REFUSED"):
        checker(checker.binding, "request-0")


def test_callback_binding_and_phase_cannot_authorize_another_helper(fake):
    checker = fake["build"]()
    changed = checker.binding
    changed["pid"] += 1
    with pytest.raises(m.BindingError, match="CALLBACK_BINDING"):
        checker(changed, "post-warmup")
    assert not list(fake["receipts"].iterdir())


def test_receipt_failure_never_returns_admitted_callback(fake, monkeypatch):
    checker = fake["build"]()
    json_file(fake["receipts"] / "helper-check-0000.json", {"old": True})
    with pytest.raises(m.BindingError, match="RECEIPT_FAILED"):
        checker(checker.binding, "post-warmup")
    assert json.loads((fake["receipts"] / "helper-check-0000.json").read_bytes()) == {"old": True}


def test_deadline_after_http_cannot_admit(fake, monkeypatch):
    checker = fake["build"]()
    original = checker._auth

    def late(token, deadline):
        result = original(token, deadline)
        monkeypatch.setattr(m, "CALL_SECONDS", 0.0)
        return result

    monkeypatch.setattr(checker, "_auth", late)
    with pytest.raises(m.BindingError, match="DEADLINE"):
        checker(checker.binding, "post-warmup")


def test_fresh_descriptor_digest_and_complete_source_population_required(fake):
    with pytest.raises(m.BindingError, match="EXTERNALLY_SELECTED"):
        m.LiveHelperChecker(
            selection_blob=b"{}", selection_sha="0" * 64, receipt_directory=fake["receipts"]
        )
    value = copy.deepcopy(fake["value"])
    value["binding"]["source_count"] = 93
    with pytest.raises(m.BindingError, match="EXACT_Q95"):
        fake["build"](value)


def test_selected_metadata_hashes_consumed_bytes_not_an_independent_second_read(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(m, "ROOT", tmp_path)
    selected = b'{"selected":true}'
    expected = {"sha256": hashlib.sha256(selected).hexdigest(), "size_bytes": len(selected)}
    monkeypatch.setattr(m, "read", lambda *_a, **_k: b'{"injected":true}')
    monkeypatch.setattr(m, "pin", lambda *_a: expected)
    with pytest.raises(m.BindingError, match="SELECTED_METADATA_DRIFT"):
        m.ref_blob({"path": "source.json", **expected}, expected["sha256"])


def test_receipt_quota_refuses_without_extra_writes_and_releases_lock(fake):
    checker = fake["build"]()
    checker._sequence = m.MAX_CHECKS
    with pytest.raises(m.BindingError, match="BOUNDED_CHECKER_RECEIPTS"):
        checker(checker.binding, "post-warmup")
    assert not list(fake["receipts"].iterdir())
    assert checker._lock.acquire(blocking=False)
    checker._lock.release()


def test_split_policy_and_helper_credentials_do_not_require_caller_token_env(fake, monkeypatch):
    monkeypatch.delenv("S1_TEXT_API_KEY", raising=False)
    monkeypatch.setenv("S1_POLICY_API_KEY", "separate-D3-policy-credential")
    checker = fake["build"]()
    connection = checker.text_connection()
    if not secrets.compare_digest(connection["api_key"], fake["token"]):
        raise AssertionError("Private helper credential mismatch")
