"""Fake binder/materializer behavior plus one bounded, owned, real tiny CPU child.

All test output uses temporary directories inside this new artifact namespace.
No existing service/private config, browser, model, processor or HTTP is used.
"""

import builtins
import contextlib
import copy
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location(
    "revisit_browser_observer_tested", HERE / "browser_observer.py"
)
o = importlib.util.module_from_spec(spec)
spec.loader.exec_module(o)
REAL_POPEN = o.subprocess.Popen


@pytest.fixture(autouse=True)
def no_framework_browser_network_or_unowned_process(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {
            "torch",
            "transformers",
            "numpy",
            "tokenizers",
            "peft",
            "safetensors",
            "playwright",
        }:
            pytest.fail("Pure observer test imported framework/browser: " + name)
        if name == "httpx":
            assert getattr(sys.modules.get(name), "EXPLICIT_FAKE_NO_HTTP", False)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(
        o.subprocess, "Popen", lambda *_a, **_kw: pytest.fail("Unexpected process actor")
    )


@pytest.fixture
def own_tmp():
    with tempfile.TemporaryDirectory(prefix=".pure-observer-test-", dir=HERE) as name:
        yield Path(name)


def write_json(path, value):
    path.write_bytes(o.encode(value))
    return o.reference(path)


def fake_preparation(own_tmp, monkeypatch, *, group="known4", runner="pass\n"):
    """Explicit fake capacity/service and fake bundle; real fixed case/arg parsing."""
    monkeypatch.setattr(o, "HERE", own_tmp)
    real_adapter = o.component("launcher_adapter")
    binding_component = o.component("service_binding")
    events, control = [], {"ready": True, "binding_calls": 0, "fail_at": None, "drain_error": None}
    token = "EXPLICIT_FAKE_PRIVATE_OWN_TEST_TOKEN_829301"
    service = own_tmp / "EXPLICIT_FAKE_SERVICE"
    service.mkdir()
    config = service / "controller-config.json"
    config.write_bytes(o.encode({"token": token}))
    config.chmod(0o600)
    legacy = {
        "schema_id": binding_component.OLD_SCHEMA,
        "service_directory": str(service.relative_to(o.ROOT)),
        "worker": "EXPLICIT_FAKE_NO_LIVE_SERVICE",
    }
    legacy_path = own_tmp / "EXPLICIT_FAKE_legacy_binding.json"
    legacy_ref = write_json(legacy_path, legacy)
    new = {
        "schema_id": binding_component.SCHEMA,
        "profile": o.PROFILE,
        "legacy_service_binding": legacy_ref,
        "revisit_capacity": {"EXPLICIT_FAKE_SELECTOR_NOT_ACTUAL75": True},
    }
    binding_path = own_tmp / "EXPLICIT_FAKE_new_binding.json"
    binding_ref = write_json(binding_path, new)
    public = {
        "schema_id": binding_component.PUBLIC_SCHEMA,
        "profile": o.PROFILE,
        "legacy_q95_service": {
            "endpoint": {
                "policy": "http://127.0.0.1:1/v1/systemone",
                "helper": "http://127.0.0.1:1",
            },
            "v8_capacity": {"EXPLICIT_FAKE_OLD23": True},
        },
        "revisit_capacity": {"EXPLICIT_FAKE_NEW75": True, "benchmark_ready": True},
        "task_quality_attested": False,
    }
    observers = []

    class FakeObserver:
        def __init__(self, policy, helper, *, secrets):
            assert (policy, helper) == (
                public["legacy_q95_service"]["endpoint"]["policy"],
                public["legacy_q95_service"]["endpoint"]["helper"],
            )
            assert secrets == (token, token)
            self.selected_attempt_total = self.omitted_count = 0
            self.primary_at_drain = None
            observers.append(self)

        @contextlib.contextmanager
        def install(self, module):
            assert module.EXPLICIT_FAKE_NO_HTTP is True
            events.append("install-fake-no-HTTP")
            yield
            events.append("uninstall-fake-no-HTTP")

        def drain(self, path, *, primary):
            events.append("drain")
            self.primary_at_drain = primary
            if control["drain_error"] is not None:
                raise control["drain_error"]
            path.mkdir()
            write_json(path / "EXPLICIT_FAKE_zero_HTTP.json", {"attempts": 0})

        def snapshot(self):
            return []

        def _message(self, error):
            return str(error).replace(token, "[REDACTED]")

    technical = SimpleNamespace(
        TransportObserver=FakeObserver,
        process_identity=lambda pid: {
            "pid": pid,
            "pgid": pid,
            "session": pid,
            "EXPLICIT_FAKE_IDENTITY": True,
        },
    )

    def read_legacy(value):
        assert value == new
        return o.parse(o.read(o.checked(value["legacy_service_binding"])))

    def state(value):
        assert value == new
        control["binding_calls"] += 1
        events.append("binding-state")
        assert tuple(os.environ.get(n) for n in o.SECRET_NAMES) == (token, token)
        if not control["ready"]:
            raise ValueError("EXPLICIT_FAKE_NEW75_CAPACITY_NOT_READY")
        if control["binding_calls"] == control["fail_at"]:
            raise ValueError("EXPLICIT_FAKE_POST_IDENTITY_CHANGED")
        return copy.deepcopy(public), (token, token)

    binder = SimpleNamespace(
        read_legacy_binding=read_legacy,
        binding_shape=binding_component.binding_shape,
        binding_state=state,
        technical_observer=lambda: technical,
    )
    import_depth = [0]

    @contextlib.contextmanager
    def case_imports():
        import_depth[0] += 1
        try:
            yield
        finally:
            import_depth[0] -= 1

    class FakeLauncher:
        def prepare_bundle(self, directory, *, case_module):
            assert import_depth[0] > 0 and case_module == real_adapter.GROUPS[group]
            events.append("fake-materialize")
            directory.mkdir()
            target = directory / "runner.py"
            target.write_text(runner)
            manifest = {"EXPLICIT_FAKE_BUNDLE": True, "runner": o.reference(target)}
            return directory, manifest

        def verify_bundle(self, directory, manifest):
            assert import_depth[0] > 0
            o.require(
                manifest
                == {"EXPLICIT_FAKE_BUNDLE": True, "runner": o.reference(directory / "runner.py")},
                "EXPLICIT_FAKE_BUNDLE_DRIFT",
            )
            events.append("fake-verify-bundle")

    adapter = SimpleNamespace(
        GROUPS=real_adapter.GROUPS,
        CASE_PLAN=real_adapter.CASE_PLAN,
        OBSERVER=real_adapter.OBSERVER,
        source_arguments=real_adapter.source_arguments,
        forward_arguments=real_adapter.forward_arguments,
        selected=real_adapter.selected,
        module=lambda label: (
            SimpleNamespace(checked_plan=lambda: events.append("old-plan-component"))
            if label == real_adapter.OBSERVER
            else pytest.fail("Unexpected adapter module")
        ),
        launcher=lambda: FakeLauncher(),
        case_imports=case_imports,
    )
    original_component = o.component
    monkeypatch.setattr(
        o,
        "component",
        lambda key: (
            {"service_binding": binder, "launcher_adapter": adapter}.get(key)
            or original_component(key)
        ),
    )
    output = own_tmp / (group + "-EXPLICIT_FAKE-01")

    def prepare():
        return o.prepare(
            group, binding_path, binding_ref["sha256"], output, o.reference(o.__file__)["sha256"]
        )

    return SimpleNamespace(
        group=group,
        output=output,
        prepare=prepare,
        binding_path=binding_path,
        binding_ref=binding_ref,
        binding=new,
        public=public,
        binder=binder,
        adapter=adapter,
        real_adapter=real_adapter,
        token=token,
        events=events,
        control=control,
        observers=observers,
        config=config,
    )


def prepared(own_tmp, monkeypatch, **kwargs):
    f = fake_preparation(own_tmp, monkeypatch, **kwargs)
    f.result = f.prepare()
    f.launch_path = f.output / "launch.json"
    f.launch = o.parse(o.read(f.launch_path))
    f.digest = f.result["launch"]["sha256"]
    return f


def child_environment(f, monkeypatch):
    for key, value in f.launch["environment"]["updates"].items():
        monkeypatch.setenv(key, value)
    for name in o.SECRET_NAMES:
        monkeypatch.setenv(name, f.token)
    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(EXPLICIT_FAKE_NO_HTTP=True))


def test_pending_capacity_refusal_creates_no_preparation_output_and_restores_env(
    own_tmp, monkeypatch
):
    f = fake_preparation(own_tmp, monkeypatch)
    f.control["ready"] = False
    monkeypatch.setenv(o.SECRET_NAMES[0], "EXPLICIT_FAKE_PREVIOUS")
    monkeypatch.delenv(o.SECRET_NAMES[1], raising=False)
    with pytest.raises(ValueError, match="NEW75_CAPACITY_NOT_READY"):
        f.prepare()
    assert not f.output.exists() and "fake-materialize" not in f.events
    assert os.environ[o.SECRET_NAMES[0]] == "EXPLICIT_FAKE_PREVIOUS"
    assert o.SECRET_NAMES[1] not in os.environ


@pytest.mark.parametrize("group,denominator", [("known4", 4), ("public6", 6)])
def test_prepare_and_validate_keep_exact_original_cases_args_and_independent_binding(
    own_tmp, monkeypatch, group, denominator
):
    f = prepared(own_tmp, monkeypatch, group=group)
    launch, binder, adapter, binding = o.validate_launch(f.launch_path, f.digest)
    selected, _ = f.real_adapter.source_arguments(group)
    assert launch["case_ids"] == selected["selected_case_ids_in_order"]
    assert launch["denominator"] == denominator and len(launch["case_ids"]) == denominator
    assert launch["forwarded_argv"] == f.real_adapter.forward_arguments(
        group,
        policy=f.public["legacy_q95_service"]["endpoint"]["policy"],
        helper=f.public["legacy_q95_service"]["endpoint"]["helper"],
        results=f.output / "results",
    )
    assert launch["public_service_identity"] == f.public and binding == f.binding
    assert binder is f.binder and adapter is f.adapter
    assert f.control["binding_calls"] == 2 and f.events.count("fake-materialize") == 1
    assert f.result["browser_model_HTTP_calls"] == 0 and not (f.output / "results").exists()
    assert f.token.encode() not in o.read(f.launch_path)


@pytest.mark.parametrize(
    "mode,message",
    [
        ("launch-sha", "launch SHA"),
        ("case-order", "cases/order"),
        ("denominator-bool", "cases/order"),
        ("argv", "full-case arguments"),
        ("environment", "measured conditions"),
        ("measurement", "measured conditions"),
        ("binding", "bytes changed"),
        ("dependency", "bytes changed"),
        ("target", "bytes changed"),
        ("bundle", "BUNDLE_DRIFT"),
        ("command", "fresh child command"),
        ("extra", "typed browser launch"),
    ],
)
def test_selected_launch_protocol_and_source_drift_are_rejected(
    own_tmp, monkeypatch, mode, message
):
    f = prepared(own_tmp, monkeypatch)
    value, digest = copy.deepcopy(f.launch), f.digest
    if mode == "launch-sha":
        digest = "0" * 64
    elif mode == "case-order":
        value["case_ids"] = value["case_ids"][::-1]
    elif mode == "denominator-bool":
        value["denominator"] = True
    elif mode == "argv":
        value["forwarded_argv"].append("--EXPLICIT_FAKE_UNREVIEWED")
    elif mode == "environment":
        value["environment"]["updates"]["TEXT_MODEL_REASONING"] = "changed"
    elif mode == "measurement":
        value["measurement_conditions"]["decisions"] = 1
    elif mode == "binding":
        f.binding_path.write_bytes(o.encode({**f.binding, "profile": "wrong"}))
    elif mode == "dependency":
        monkeypatch.setitem(
            o.PINS, "service_binding", {**o.PINS["service_binding"], "sha256": "0" * 64}
        )
    elif mode == "target":
        (f.output / "runtime-bundle/runner.py").write_text("raise AssertionError('drift')\n")
    elif mode == "bundle":
        value["bundle_manifest"]["EXPLICIT_FAKE_BUNDLE"] = False
    elif mode == "command":
        command = o.parse(o.read(f.output / "child-command.json"))
        command["argv"].append("--EXPLICIT_FAKE_EXTRA")
        write_json(f.output / "child-command.json", command)
    else:
        value["EXPLICIT_FAKE_EXTRA"] = True
    if value != f.launch:
        digest = write_json(f.launch_path, value)["sha256"]
    with pytest.raises(ValueError, match=message):
        o.validate_launch(f.launch_path, digest)
    assert not (f.output / "controller-intent.json").exists()


def test_default_execute_is_no_private_config_or_live_admission(own_tmp, monkeypatch):
    f = prepared(own_tmp, monkeypatch)
    f.config.unlink()  # validate_launch is source-only; admission must not occur.
    f.control["ready"] = False
    before = f.control["binding_calls"]
    result = o.execute(f.launch_path, f.digest)
    assert result["live_binding_attested"] is result["task_success_attested"] is False
    assert f.control["binding_calls"] == before
    assert not any((f.output / n).exists() for n in o.CONTROLLER_FILES)


def test_bound_environment_restores_callers_secrets_and_scrubs_unselected_env(own_tmp, monkeypatch):
    f = fake_preparation(own_tmp, monkeypatch)
    monkeypatch.setenv(o.SECRET_NAMES[0], "EXPLICIT_FAKE_PREVIOUS_A")
    monkeypatch.delenv(o.SECRET_NAMES[1], raising=False)
    monkeypatch.setenv("EXPLICIT_FAKE_UNRELATED_SECRET", "MUST_NOT_FORWARD")
    updates = {"HF_HUB_OFFLINE": "1", "TEXT_MODEL_REASONING": "none"}
    public, env = o.bound_environment(f.binder, f.binding, updates)
    assert public == f.public
    assert all(env[n] == f.token for n in o.SECRET_NAMES)
    assert all(env[n] == v for n, v in updates.items())
    assert set(env) <= set(o.SAFE_ENV) | set(updates) | set(o.SECRET_NAMES)
    assert "EXPLICIT_FAKE_UNRELATED_SECRET" not in env
    assert os.environ[o.SECRET_NAMES[0]] == "EXPLICIT_FAKE_PREVIOUS_A"
    assert o.SECRET_NAMES[1] not in os.environ


def test_child_primary_survives_drain_and_postidentity_failures_with_receipt_and_restoration(
    own_tmp, monkeypatch
):
    primary = RuntimeError("EXPLICIT_FAKE_RUNNER_PRIMARY")
    monkeypatch.setitem(sys.modules, "EXPLICIT_FAKE_BROWSER_TEST", SimpleNamespace(primary=primary))
    f = prepared(
        own_tmp,
        monkeypatch,
        runner="import sys\nraise sys.modules['EXPLICIT_FAKE_BROWSER_TEST'].primary\n",
    )
    child_environment(f, monkeypatch)
    f.control["drain_error"] = OSError("EXPLICIT_FAKE_DRAIN_FAILURE")
    f.control["fail_at"] = f.control["binding_calls"] + 2
    before_argv, before_path = sys.argv, list(sys.path)
    with pytest.raises(RuntimeError) as caught:
        o.child(f.launch_path, f.digest)
    assert caught.value is primary and f.observers[0].primary_at_drain is primary
    assert sys.argv is before_argv and sys.path == before_path
    assert "drain" in f.events
    assert any("Observer drain secondary" in n for n in primary.__notes__)
    assert any("Post-group identity secondary" in n for n in primary.__notes__)
    receipt = o.parse(o.read(f.output / "child-exit.json"))
    assert receipt["complete"] is False and receipt["error"]["type"] == "RuntimeError"
    assert (
        receipt["process_exit_is_task_success"]
        is receipt["zero_HTTP_is_task_success"]
        is receipt["all10_task_success_attested"]
        is False
    )
    assert f.token.encode() not in o.read(f.output / "child-exit.json")


def test_child_zero_http_and_exit0_are_explicitly_not_task_quality(own_tmp, monkeypatch):
    f = prepared(own_tmp, monkeypatch, runner="raise SystemExit(0)\n")
    child_environment(f, monkeypatch)
    assert o.child(f.launch_path, f.digest) == 0
    receipt = o.parse(o.read(f.output / "child-exit.json"))
    assert (
        receipt["complete"] is True
        and receipt["selected_HTTP_attempts"] == receipt["observer_rows"] == 0
    )
    assert (
        receipt["process_exit_is_task_success"]
        is receipt["zero_HTTP_is_task_success"]
        is receipt["all10_task_success_attested"]
        is False
    )
    assert f.control["binding_calls"] == 4 and f.events.count("drain") == 1


def test_child_postidentity_failure_is_primary_when_runner_succeeds(own_tmp, monkeypatch):
    f = prepared(own_tmp, monkeypatch)
    child_environment(f, monkeypatch)
    f.control["fail_at"] = f.control["binding_calls"] + 2
    with pytest.raises(ValueError, match="POST_IDENTITY_CHANGED"):
        o.child(f.launch_path, f.digest)
    receipt = o.parse(o.read(f.output / "child-exit.json"))
    assert receipt["complete"] is False and f.events.count("drain") == 1


def test_owned_tiny_cpu_child_has_exact_identity_and_whole_group_absence(own_tmp, monkeypatch):
    """Only this test permits one exact Python sleep child; never a browser target."""
    common = o.component("owned_lifecycle")
    technical = o.component("service_binding").technical_observer()
    output = own_tmp / "EXPLICIT_REAL_TINY_CPU_ONLY"
    output.mkdir()
    argv = [sys.executable, "-B", "-s", "-c", "import time; time.sleep(0.35)"]

    def own_child_only(actual, **kwargs):
        assert actual == argv and kwargs["start_new_session"] is True
        return REAL_POPEN(actual, **kwargs)

    monkeypatch.setattr(o.subprocess, "Popen", own_child_only)
    env = {name: os.environ[name] for name in o.SAFE_ENV if name in os.environ}
    result = o.run_owned(argv, env, output, common, technical, wall=5)
    identity = o.parse(o.read(output / "controller-process.json"))
    assert identity["pid"] == identity["pgid"] == identity["session"]
    assert identity["argv"] == argv
    assert result["actual_returncode"] == 0 and result["closure_confirmed"] is True
    assert result["cleanup"]["remaining"] == []
    current = common.process_table()
    assert identity["pid"] not in current
    assert not any(
        r["pgid"] == identity["pgid"] or r["sid"] == identity["session"] for r in current.values()
    )
    assert (
        result["cleanup"]["tracked_PID_start_ticks"][str(identity["pid"])]
        == identity["start_ticks"]
    )
    assert result["process_exit_is_task_success"] is False


def test_default_cli_is_metadata_only_without_binding_config_or_actors(capsys):
    assert o.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "DRY_SOURCE_ONLY" and result["browser_model_HTTP_calls"] == 0
    assert result["task_quality_attested"] is False


def test_owned_timeout_retains_primary_after_real_closure_then_fake_cleanup_error(
    own_tmp, monkeypatch
):
    common = o.component("owned_lifecycle")
    technical = o.component("service_binding").technical_observer()
    output = own_tmp / "EXPLICIT_REAL_TINY_TIMEOUT_CPU_ONLY"
    output.mkdir()
    argv = [sys.executable, "-B", "-s", "-c", "import time; time.sleep(30)"]

    def own_child_only(actual, **kwargs):
        assert actual == argv and kwargs["start_new_session"] is True
        return REAL_POPEN(actual, **kwargs)

    def close_then_fake_error(*args, **kwargs):
        closed = common.stop_owned(*args, **kwargs)
        assert closed["confirmed"] is True
        raise RuntimeError("EXPLICIT_FAKE_CLEANUP_REPORT_FAILURE_AFTER_REAL_CLOSURE")

    wrapper = SimpleNamespace(
        process_table=common.process_table,
        extend_owned=common.extend_owned,
        stop_owned=close_then_fake_error,
    )
    monkeypatch.setattr(o.subprocess, "Popen", own_child_only)
    env = {name: os.environ[name] for name in o.SAFE_ENV if name in os.environ}
    with pytest.raises(ValueError, match="wall deadline") as caught:
        o.run_owned(argv, env, output, wrapper, technical, wall=0.2)
    assert any("cleanup secondary: RuntimeError" in n for n in caught.value.__notes__)
    identity = o.parse(o.read(output / "controller-process.json"))
    terminal = o.parse(o.read(output / "controller-exit.json"))
    assert terminal["error_type"] == "ValueError" and terminal["cleanup_error_types"] == [
        "RuntimeError"
    ]
    assert terminal["closure_confirmed"] is False  # A failed report is not forged into evidence.
    current = common.process_table()
    assert identity["pid"] not in current
    assert not any(
        r["pgid"] == identity["pgid"] or r["sid"] == identity["session"] for r in current.values()
    )


def test_clean_env_standalone_default_cli_creates_no_browser_group_output(monkeypatch):
    argv = [sys.executable, "-B", "-s", str(HERE / "browser_observer.py")]
    before = {p.name for p in HERE.iterdir() if p.name.startswith(("known4-", "public6-"))}

    def own_dry_cli_only(actual, **kwargs):
        assert actual == argv
        return REAL_POPEN(actual, **kwargs)

    monkeypatch.setattr(o.subprocess, "Popen", own_dry_cli_only)
    env = {name: os.environ[name] for name in o.SAFE_ENV if name in os.environ}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    assert "PYTHONPATH" not in env and not any(n in env for n in o.SECRET_NAMES)
    result = o.subprocess.run(argv, cwd=o.ROOT, env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0 and result.stderr == ""
    value = json.loads(result.stdout)
    assert value["browser_model_HTTP_calls"] == 0 and value["task_quality_attested"] is False
    assert before == {p.name for p in HERE.iterdir() if p.name.startswith(("known4-", "public6-"))}
