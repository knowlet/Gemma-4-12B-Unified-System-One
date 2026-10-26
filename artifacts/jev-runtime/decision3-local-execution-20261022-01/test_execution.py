"""CPU-only actual subprocess/loopback tests; no model, browser, unit or Comfy calls."""

import copy
import hashlib
import http.client
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

HERE = Path(__file__).absolute().parent
SPEC = importlib.util.spec_from_file_location("_test_d3_execution", HERE / "execution.py")
E = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = E
exec(compile((HERE / "execution.py").read_bytes(), str(HERE / "execution.py"), "exec"), E.__dict__)


@pytest.fixture
def selected():
    with tempfile.TemporaryDirectory(prefix=".cpu-execution-", dir=HERE) as directory:
        root = Path(directory)
        helper = root / "helper-selection.json"
        helper.write_text('{"cpu_fixture_only":true}')
        bootstrap_ref = E.source_ref(
            E.ROOT
            / "artifacts/jev-runtime/decision3-current-source-bootstrap-20261022-01/current_bootstrap.py"
        )
        b = E.load_module(bootstrap_ref, "_test_selected_bootstrap")
        host = E.load_module(
            E.source_ref(
                E.ROOT
                / "artifacts/jev-runtime/decision3-local-transport-20261022-01/local_transport.py"
            ),
            "_test_host_transport",
        ).host_observation()
        binding = {
            "profile": b.HELPER_PROFILE,
            "qualification_sha256": b.HELPER_SHA,
            "source_count": 95,
            "boot_id": host["boot_id"],
            "pid": os.getpid(),
            "start_ticks": 1,
            "pgid": os.getpgrp(),
            "sid": os.getsid(0),
            "endpoint": "http://127.0.0.1:12009",
            "seconds": 20,
            "tokens": 128,
            "context": 8192,
        }
        plan = E.prepare_plan(
            model="d3-edge",
            helper_checker=HERE / "fake_components.py",
            helper_selections={g: helper for g in E.GROUPS},
            helper_bindings={g: binding for g in E.GROUPS},
            browser={},
            output=root / "actual",
            model_outputs={g: root / ("model-" + g) for g in E.GROUPS},
            boot_id=host["boot_id"],
            mem_total_bytes=host["MemTotal"],
            gpu_uuid="cpu-fixture-no-gpu",
            expires_monotonic=time.monotonic() + 90,
            fixture=True,
        )
        path = root / "plan.json"

        def seal(value=plan):
            blob = E.encode(value)
            path.write_bytes(blob)
            return hashlib.sha256(blob).hexdigest()

        digest = seal()
        yield SimpleNamespace(root=root, plan=plan, path=path, digest=digest, seal=seal, b=b)


def checked(value, group="known4", refuse=None):
    plan, b, _inputs, _identities = E.selected_plan(value.path, value.digest, fixture=True)
    transport = E.transport_for(plan)
    folder = value.root / ("transport-" + group)
    folder.mkdir()
    calls = []

    def checker(binding, phase):
        calls.append(phase)
        if refuse is not None and refuse in phase:
            raise ValueError("Private credential must not enter control evidence")
        return {
            "binding": binding,
            "phase": phase,
            "helper_admitted": True,
            "co_resident_admitted": True,
            "receipt_sha256": hashlib.sha256(E.encode([binding, phase])).hexdigest(),
        }

    session = E.checked_session_class(transport)(
        E.selection_for(plan, group, transport),
        folder,
        helper_binding=plan["arms"][group]["helper_binding"],
        checker=checker,
    )
    env = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "JEV_D3_FIXTURE_PARENT": json.dumps(transport.proc_record(os.getpid())),
    }
    command = [
        plan["interpreter"],
        "-B",
        "-s",
        str(HERE / "launch_execution.py"),
        "--execute",
        "--cpu-fixture",
        "--role",
        "worker",
        "--plan",
        str(value.path),
        "--plan-sha256",
        value.digest,
        "--group",
        group,
    ]
    return plan, b, transport, session, command, env, calls


def http_request(base, token, body):
    url = urlsplit(base)
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=3)
    try:
        connection.request(
            "POST",
            "/predict",
            body,
            {"Authorization": "Bearer " + token, "Content-Type": "application/json"},
        )
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


@pytest.mark.parametrize("group", ["known4", "public6"])
def test_actual_cpu_worker_loopback_preserves_full_255_and_original_journal(selected, group):
    plan, _b, t, session, argv, env, calls = checked(selected, group)
    http = None
    try:
        ready = session.start(argv, env=env, cwd=E.ROOT, timeout=12)
        assert ready["details"]["bootstrap"]["cpu_fixture_only"] is True
        assert ready["details"]["model_qualified"] is False
        gateway = t.LocalGateway(session)
        token = "fixture-private-policy-token-123456789"
        http = t.LoopbackHTTP(gateway, token=token)
        base = http.start()
        payload = {
            "model": plan["model"],
            "elements": [{"id": i, "text": "原始元素-" + str(i)} for i in range(255)],
        }
        body = json.dumps(payload, ensure_ascii=False, indent=1).encode()
        status, response = http_request(base, token, body)
        assert status == 200
        result = json.loads(response)
        assert result["elements_seen"] == 255 and result["last_element"] == payload["elements"][-1]
        assert result["native_calls"] == 1
        original = json.loads(
            next(
                (Path(plan["arms"][group]["model_output"]) / "requests").glob("*.json")
            ).read_bytes()
        )
        assert original["payload_json"].encode() == body
        assert original["original_request"] == payload
        assert original["http_sha256"] == hashlib.sha256(body).hexdigest()
        assert original["success"] is True
        assert calls[:3] == ["before-component-constructor", "before-request-1", "after-request-1"]
    finally:
        if http is not None:
            http.close()
        closure = session.close(grace=3, term=1, kill=1)
    assert closure["success"] is True and closure["direct_child_exit_confirmed"] is True
    assert (
        closure["whole_tree_exit_confirmed"] is False
        and closure["continuation_authorized"] is False
    )
    assert closure["closed"]["receipt"]["cpu_fixture_only"] is True
    assert calls[-1] == "finalize-before-close"


@pytest.mark.parametrize(
    "phase", ["before-component-constructor", "before-request", "after-request", "finalize"]
)
def test_real_parent_checker_refusal_blocks_and_owned_child_exits(selected, phase):
    plan, _b, t, session, argv, env, calls = checked(selected, refuse=phase)
    try:
        if phase == "before-component-constructor":
            with pytest.raises(Exception):
                session.start(argv, env=env, cwd=E.ROOT, timeout=10)
        else:
            session.start(argv, env=env, cwd=E.ROOT, timeout=10)
            if phase != "finalize":
                http = t.LoopbackHTTP(
                    t.LocalGateway(session), token="fixture-policy-private-123456789012345"
                )
                try:
                    base = http.start()
                    status, _ = http_request(
                        base, http.token, E.encode({"model": plan["model"], "elements": []})
                    )
                    assert status == 504
                finally:
                    http.close()
    finally:
        closure = session.close(grace=3, term=1, kill=1)
    assert closure["success"] is False
    assert closure["direct_child_exit_confirmed"] is True
    assert any(phase in row for row in calls)
    assert b"Private credential must not enter" not in b"".join(
        p.read_bytes() for p in session.output.glob("*.json")
    )


def test_actual_loopback_late_native_candidate_never_delivers_200(selected):
    plan, _b, t, session, argv, env, _calls = checked(selected)
    http = None
    try:
        session.start(argv, env=env, cwd=E.ROOT, timeout=10)
        http = t.LoopbackHTTP(
            t.LocalGateway(session, seconds=0.07), token="fixture-private-policy-123456789012345"
        )
        status, _ = http_request(
            http.start(),
            http.token,
            E.encode({"model": plan["model"], "fixture_delay": 0.15, "elements": []}),
        )
        assert status == 504
        assert session.blocked is True
    finally:
        if http is not None:
            http.close()
        closure = session.close(grace=3, term=1, kill=1)
    assert closure["success"] is False and closure["direct_child_exit_confirmed"] is True


@pytest.mark.parametrize(
    "fault",
    [
        "expired",
        "wrong-source",
        "missing-population",
        "wrong-boot",
        "wrong-helper",
        "duplicate-units",
        "duplicate-model-output",
        "changed-plan-sha",
        "actual-profile",
        "wrong-component",
    ],
)
def test_selected_authority_refuses_before_cpu_constructor(selected, fault):
    plan = copy.deepcopy(selected.plan)
    if fault == "expired":
        plan["expires_monotonic"] = time.monotonic() - 1
    elif fault == "wrong-source":
        plan["sources"][0]["sha256"] = "0" * 64
    elif fault == "missing-population":
        plan["sources"] = [row for row in plan["sources"] if row["path"] != "src/s1/jev.py"]
    elif fault == "wrong-boot":
        plan["boot_id"] = "different-selected-boot"
    elif fault == "wrong-helper":
        plan["arms"]["known4"]["helper_binding"]["qualification_sha256"] = "0" * 64
    elif fault == "duplicate-units":
        plan["arms"]["public6"]["unit"] = plan["arms"]["known4"]["unit"]
    elif fault == "duplicate-model-output":
        plan["arms"]["public6"]["model_output"] = plan["arms"]["known4"]["model_output"]
    elif fault == "actual-profile":
        plan["profile"] = E.PROFILE
    elif fault == "wrong-component":
        plan["components"]["boundary"]["sha256"] = "0" * 64
    digest = selected.seal(plan)
    if fault == "changed-plan-sha":
        digest = "0" * 64
    with pytest.raises(Exception):
        E.selected_plan(selected.path, digest, fixture=True)
    assert not Path(plan["arms"]["known4"]["model_output"]).exists()


def test_missing_actual_owner_refuses_before_model_or_gpu(selected):
    # Exercise actual owner admission directly: missing proof is refused before
    # any constructor/dependency callback, without invoking systemd at all.
    owner = SimpleNamespace(
        status=lambda _selection: pytest.fail("Must not query a unit without admission")
    )
    with pytest.raises(FileNotFoundError):
        E.owner_admission(selected.plan, "known4", selected.digest, owner, worker=True)
    assert not Path(selected.plan["arms"]["known4"]["model_output"]).exists()


def test_fixture_cannot_bypass_parent_identity(selected):
    plan, _b, _t, session, argv, env, _calls = checked(selected)
    env["JEV_D3_FIXTURE_PARENT"] = json.dumps({"pid": os.getpid(), "start_ticks": 1})
    with pytest.raises(Exception):
        session.start(argv, env=env, cwd=E.ROOT, timeout=10)
    assert session.closure["direct_child_exit_confirmed"] is True
    assert not Path(plan["arms"]["known4"]["model_output"]).exists()


def test_d3_only_owner_profile_has_80gib_and_unchanged_scope(selected):
    owner = E.load_module(selected.plan["components"]["boundary"], "_test_d3_80_owner")
    s = E.unit_selection(selected.plan, "known4", owner)
    assert s.memory_max_bytes == 80 * 1024**3
    assert s.properties()["MemoryMax"] == str(80 * 1024**3)
    assert s.properties()["MemorySwapMax"] == "0"
    assert s.properties()["KillMode"] == "control-group"
    assert s.properties()["RuntimeMaxUSec"] == "2h"
    assert s.properties()["Delegate"] == "no"
    with pytest.raises(Exception):
        owner.Selection(s.unit, s.nonce, s.boot_id, 80 * 1024**3, 7200, 10)


@pytest.mark.parametrize("group", ["known4", "public6"])
def test_original_exact_browser_workload_and_no_credential_in_argv(selected, group):
    plan = selected.plan
    plan["browser"] = {
        "interpreter": sys.executable,
        "upstream_path": "/tmp/selected-upstream",
        "chromium": {"path": "/tmp/selected-chromium"},
    }
    argv = E.browser_command(
        plan,
        group,
        Path("/tmp/bundle"),
        Path("/tmp/results"),
        "http://127.0.0.1:10000/predict",
        {
            "endpoint": "http://127.0.0.1:12009",
            "model": "generated-actions-none",
            "api_key": "private-test-key",
        },
    )
    assert "private-test-key" not in repr(argv)
    assert argv[argv.index("--public-cases") + 1 :] == list(E.CASES[group])
    assert argv[argv.index("--max-steps") + 1] == "16"
    assert argv[argv.index("--repeats") + 1] == "1"
    assert argv[argv.index("--seed") + 1] == "20261014"
    assert argv[argv.index("--observation-profile") + 1] == "composed-visible-text-settled-v7"
    startup = selected.b.load_worker().startup_configuration(
        "publisher-text48-v1", E.GROUPS[group][0]
    )
    assert startup["max_browser_envelopes"] == E.GROUPS[group][1]
    assert startup["native_batch_size"] == 8 and startup["images"] is False


@pytest.mark.parametrize(
    "fault", ["missing-case", "duplicate-case", "done-alone", "profile-failed", "false-arithmetic"]
)
def test_original_judge_and_full_denominator_are_required(selected, fault):
    outcomes = [
        {
            "task_id": case,
            "success": True,
            "status": "done",
            "verification": {"success": True},
            "error": None,
        }
        for case in E.CASES["known4"]
    ]
    summary = {
        "attempts": 4,
        "successes": 4,
        "outcomes": outcomes,
        "observation_profile_failed": False,
    }
    if fault == "missing-case":
        outcomes.pop()
    elif fault == "duplicate-case":
        outcomes[-1]["task_id"] = outcomes[0]["task_id"]
    elif fault == "done-alone":
        outcomes[-1]["verification"]["success"] = False
    elif fault == "profile-failed":
        summary["observation_profile_failed"] = True
    else:
        summary["successes"] = 3
    path = selected.root / "summary.json"
    path.write_bytes(E.encode(summary))
    with pytest.raises(Exception):
        E.browser_score(path, "known4")


def test_default_cli_does_not_import_components_or_start_any_actor():
    row = subprocess.run(
        [sys.executable, "-B", "-s", str(HERE / "launch_execution.py")],
        check=True,
        capture_output=True,
        text=True,
        timeout=3,
    )
    result = json.loads(row.stdout)
    assert result["mode"] == "DRY_SOURCE_ONLY" and result["actual_started"] is False
    assert result["model_qualified"] is False and result["groups"] == {"known4": 64, "public6": 96}


def test_helper_response_wrong_identity_or_sha_cannot_admit():
    # Actual pipe framing, not a boolean substitute. Preload a forged response.
    reader, writer = os.pipe()
    control_reader, control_writer = os.pipe()
    try:
        os.write(
            writer,
            E.encode(
                {
                    "event": "helper_checked",
                    "sequence": 1,
                    "check_sha256": "0" * 64,
                    "identity": {},
                    "evidence": {},
                    "error": None,
                }
            )
            + b"\n",
        )
        transport = E.load_module(
            E.source_ref(
                E.ROOT
                / "artifacts/jev-runtime/decision3-local-transport-20261022-01/local_transport.py"
            ),
            "_test_rpc_transport",
        )
        host = transport.host_observation()
        selection = transport.Selection(
            "cpu-fixture", "d3-edge", host["boot_id"], host["MemTotal"], "cpu-fixture"
        )
        with (
            os.fdopen(reader, "rb", buffering=0) as source,
            os.fdopen(control_writer, "wb", buffering=0) as control,
        ):
            reader = control_writer = None
            rpc = E.HelperRPC(transport, selection, {"fixture": True}, control, source)
            rpc.deadline = time.monotonic() + 0.2
            with pytest.raises(Exception, match="Authenticated parent"):
                rpc({"fixture": True}, "test")
    finally:
        for fd in [reader, writer, control_reader, control_writer]:
            if fd is not None:
                os.close(fd)


def test_selected_loader_executes_checked_source_despite_valid_stale_pyc(selected):
    import importlib.machinery
    import py_compile

    path = selected.root / "selected_source.py"
    path.write_text("VALUE = 'poison'\n")
    py_compile.compile(str(path), doraise=True)
    stamp = path.stat().st_mtime_ns
    path.write_text("VALUE = 'actual'\n")
    os.utime(path, ns=(stamp, stamp))
    # The ordinary loader would accept this same-size/same-timestamp cache.
    ordinary = importlib.machinery.SourceFileLoader("_ordinary_stale", str(path))
    spec = importlib.util.spec_from_loader(ordinary.name, ordinary)
    module = importlib.util.module_from_spec(spec)
    ordinary.exec_module(module)
    assert module.VALUE == "poison"
    assert E.load_module(E.source_ref(path), "_verified_source").VALUE == "actual"


def test_helper_control_wait_is_bounded_by_request_budget():
    reader, writer = os.pipe()
    control_reader, control_writer = os.pipe()
    try:
        transport = E.load_module(
            E.source_ref(
                E.ROOT
                / "artifacts/jev-runtime/decision3-local-transport-20261022-01/local_transport.py"
            ),
            "_test_deadline_transport",
        )
        host = transport.host_observation()
        selection = transport.Selection(
            "cpu-fixture", "d3-edge", host["boot_id"], host["MemTotal"], "cpu-fixture"
        )
        with (
            os.fdopen(reader, "rb", buffering=0) as source,
            os.fdopen(control_writer, "wb", buffering=0) as control,
        ):
            reader = control_writer = None
            rpc = E.HelperRPC(transport, selection, {"fixture": True}, control, source)
            started = time.monotonic()
            rpc.deadline = started + 0.03
            with pytest.raises(Exception, match="deadline"):
                rpc({"fixture": True}, "before-request")
            assert time.monotonic() - started < 0.3
    finally:
        for fd in [reader, writer, control_reader, control_writer]:
            if fd is not None:
                os.close(fd)


@pytest.mark.parametrize("fault", ["invocation", "cap", "parent", "readback-sha", "state-only"])
def test_external_owner_requires_fresh_actual_bound_identity_and_limits(
    selected, fault, monkeypatch
):
    plan = selected.plan
    boundary = E.load_module(plan["components"]["boundary"], "_test_owner_admission")
    selection = E.unit_selection(plan, "known4", boundary)
    out = E.group_output(plan, "known4")
    out.mkdir(parents=True)
    pid = os.getppid()
    actor = {
        "pid": pid,
        "start_ticks": 1234,
        "pgid": pid,
        "sid": pid,
        "uid": os.getuid(),
        "cgroup": selection.cgroup,
        "state": "S",
    }
    readback = {"actor": actor}
    E.save(out / "unit-readback.json", readback)
    invocation = "a" * 32
    admission = {
        "plan_sha256": selected.digest,
        "group": "known4",
        "nonce": selection.nonce,
        "invocation_id": invocation,
        "readback_sha256": hashlib.sha256((out / "unit-readback.json").read_bytes()).hexdigest(),
    }
    if fault == "readback-sha":
        admission["readback_sha256"] = "0" * 64
    E.save(out / "admission.json", admission)
    cgroot = selected.root / "fake-cgroup"
    cg = cgroot / selection.cgroup.lstrip("/")
    cg.mkdir(parents=True)
    (cg / "memory.max").write_text(str(E.CAP if fault != "cap" else 40 * 1024**3))
    (cg / "memory.swap.max").write_text("0")
    (cg / "cgroup.type").write_text("domain")
    status = {
        **selection.properties(),
        "MainPID": str(pid),
        "InvocationID": invocation if fault != "invocation" else "b" * 32,
    }

    def process(number):
        row = copy.deepcopy(actor)
        if number == os.getpid():
            row.update(pid=number, pgid=number, sid=number)
        if fault == "parent" and number == pid:
            row["start_ticks"] += 1
        if fault == "state-only":
            row["state"] = "R"
        return row

    owner = SimpleNamespace(
        Selection=boundary.Selection,
        status=lambda _selection: status,
        process=process,
        boot=lambda: plan["boot_id"],
        CGROUP_ROOT=cgroot,
    )
    if fault == "state-only":
        assert (
            E.owner_admission(plan, "known4", selected.digest, owner, worker=True)["invocation_id"]
            == invocation
        )
    else:
        with pytest.raises(Exception):
            E.owner_admission(plan, "known4", selected.digest, owner, worker=True)


def test_expired_service_refuses_before_any_helper_or_constructor(selected):
    plan = selected.plan
    plan["expires_monotonic"] = time.monotonic() - 1

    def rpc(*_args):
        pytest.fail("Expired launch must precede helper or constructor")

    service = E.ModelService(
        plan,
        "known4",
        selected.b,
        rpc,
        fixture_factory=lambda _service: pytest.fail("No constructor"),
    )
    with pytest.raises(Exception, match="expired"):
        service.initialize()


@pytest.mark.parametrize("failure", ["gateway", "listener", "session"])
def test_each_cleanup_is_attempted_and_primary_error_is_preserved(failure):
    calls = []
    primary = {"type": "OriginalPrimaryFailure"}

    def operation(stage):
        def run(**_kwargs):
            calls.append(stage)
            if stage == failure:
                raise RuntimeError("Secondary cleanup error")
            return {"success": True}

        return run

    closure, kept, errors = E.close_actor_resources(
        SimpleNamespace(block=operation("gateway")),
        SimpleNamespace(close=operation("listener")),
        SimpleNamespace(close=operation("session")),
        primary,
    )
    assert calls == ["gateway", "listener", "session"]
    assert kept == primary and len(errors) == 1
    assert closure["success"] is (failure != "session")


@pytest.mark.parametrize("complete", [False, True])
def test_complete_ten_task_score_is_distinct_from_partial_group_counts(selected, complete):
    groups = {"known4": {"actor_success": True, "score": {"attempts": 4, "successes": 2}}}
    if complete:
        groups["public6"] = {"actor_success": True, "score": {"attempts": 6, "successes": 3}}
    result = E.aggregate_result(selected.plan, selected.digest, groups, None)
    assert result["attempts_denominator"] == 10
    assert result["observed_complete_group_attempts"] == (10 if complete else 4)
    assert result["observed_successes"] == (5 if complete else 2)
    assert result["success_rate"] == (0.5 if complete else None)
    assert result["successes"] == (5 if complete else None)


def test_production_profile_rejects_arbitrary_helper_before_browser_or_constructor(selected):
    plan = selected.plan
    plan["profile"] = E.PROFILE
    for group in E.GROUPS:
        plan["arms"][group]["model_output"] = str(
            selected.b.REPLAY_ROOT / ("not-created-test-" + group)
        )
    selected.digest = selected.seal()
    with pytest.raises(Exception, match="reviewed live helper"):
        E.selected_plan(selected.path, selected.digest)


@pytest.mark.parametrize("fault", ["busy", "unknown", "reserve", "boot", "total"])
def test_continuous_host_guard_refuses_busy_unknown_or_reserve_crossing(selected, fault):
    plan = selected.plan
    row = {
        "boot_id": plan["boot_id"],
        "mem_total_bytes": plan["mem_total_bytes"],
        "mem_available_bytes": plan["mem_total_bytes"],
        "comfy_state": "absent_confirmed",
        "comfy_running": 0,
        "comfy_pending": 0,
    }
    if fault == "busy":
        row.update(comfy_state="http_busy", comfy_running=1)
    elif fault == "unknown":
        row["comfy_state"] = "uncertain"
    elif fault == "reserve":
        reserve = max(24 * 1024**3, (plan["mem_total_bytes"] + 4) // 5)
        row["mem_available_bytes"] = reserve + 2 * 1024**3 - 1
    elif fault == "boot":
        row["boot_id"] = "different"
    else:
        row["mem_total_bytes"] += 4096
    with pytest.raises(Exception):
        E.host_idle(plan, SimpleNamespace(observe=lambda _boot: row))


def test_continuous_host_guard_preserves_original_probe_and_exact_reserve_boundary(selected):
    plan = selected.plan
    reserve = max(24 * 1024**3, (plan["mem_total_bytes"] + 4) // 5)
    row = {
        "boot_id": plan["boot_id"],
        "mem_total_bytes": plan["mem_total_bytes"],
        "mem_available_bytes": reserve + 2 * 1024**3,
        "comfy_state": "http_idle",
        "comfy_running": 0,
        "comfy_pending": 0,
    }
    assert E.host_idle(plan, SimpleNamespace(observe=lambda _boot: row)) == row


def test_raw_driver_map_read_uses_numeric_own_pid_with_frozen_h15_reader(selected):
    # Actual own CPU process metadata only; no GPU/driver library is loaded.
    h = E.load_module(selected.plan["components"]["h15"], "_test_h15_maps_reader")
    with pytest.raises(Exception, match="Ordinary path"):
        h.read(Path("/proc/self/maps"), 8 * 1024**2)
    maps = E.driver_maps(h)
    assert "[stack]" in maps and len(maps) > 0
