"""Queued external launcher contracts; every process and wait is a test double."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SPEC = importlib.util.spec_from_file_location(
    "run_ultrafast_after", SCRIPTS / "run_ultrafast_after.py"
)
RUNNER = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(SCRIPTS))
try:
    SPEC.loader.exec_module(RUNNER)
finally:
    sys.path.remove(str(SCRIPTS))


def identity(state="S", *, start="Mon Oct 26 20:40:00 2026", command="python original.py"):
    assert len(start) == 24
    return f"{start} {state} {command}"


@pytest.fixture
def queued_args(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    for name in (*RUNNER.SCRIPT_FILES, "src/s1/example.py", "pyproject.toml", "uv.lock"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen " + name + "\n")
    manifest = root / "configs/benchmarks/ultrafast-external.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"tasks": [{"id": "external-one"}, {"id": "external-two"}]}\n')
    monkeypatch.setattr(RUNNER, "ROOT", root)
    monkeypatch.setattr(RUNNER.guard, "ROOT", root)
    return SimpleNamespace(
        predecessor_pid=[89007, 97017],
        output=tmp_path / "supervisor",
        run_output=tmp_path / "run",
        model=root / "artifacts/release/checkpoint",
        manifest=manifest,
        upstream=root / "artifacts/benchmark-tools/jev-ultrafast",
        repeats=3,
        task_timeout=180,
        mps_call_maintenance=False,
        oversized_head_policy="reject",
    )


def fake_processes(monkeypatch, observations, *, exit_code=0, identity_error=False):
    sequences = {pid: iter(rows) for pid, rows in observations.items()}
    events, sleeps, spawned = [], [], []

    def observe(pid):
        events.append(("observe", pid))
        if pid == 525252:
            if identity_error:
                raise RuntimeError("child ps observation interrupted")
            return identity(command="python frozen-child.py")
        row = next(sequences[pid])
        if isinstance(row, Exception):
            raise row
        if callable(row):
            return row()
        return row

    class Child:
        pid = 525252

        def wait(self):
            events.append(("wait_child", self.pid))
            return exit_code

    def spawn(argv, **kwargs):
        events.append(("spawn", tuple(argv)))
        kwargs["stdout"].write(b"child output retained\n")
        spawned.append((argv, kwargs))
        return Child()

    monkeypatch.setattr(RUNNER.guard, "process_identity", observe)
    monkeypatch.setattr(RUNNER.subprocess, "Popen", spawn)
    monkeypatch.setattr(RUNNER.time, "sleep", sleeps.append)
    return events, sleeps, spawned


def receipt(args):
    return json.loads((args.output / "receipt.json").read_text())


def test_capture_all_original_handles_then_wait_before_mps_child(queued_args, monkeypatch):
    events, sleeps, spawned = fake_processes(
        monkeypatch,
        {
            89007: [identity(), identity("R+"), None],
            97017: [identity(command="python prefix.py"), None],
        },
    )
    assert RUNNER.supervise(queued_args) == 0
    assert events[:2] == [("observe", 89007), ("observe", 97017)]
    assert events[2:5] == [("observe", 89007), ("observe", 89007), ("observe", 97017)]
    assert events[5][0] == "spawn"
    assert sleeps == [15]
    argv, kwargs = spawned[0]
    assert argv[:2] == [sys.executable, str(RUNNER.ROOT / "scripts/run_ultrafast.py")]
    assert argv[argv.index("--device") + 1] == "mps"
    assert argv[argv.index("--profiles") + 1] == "causal_multislot,independent"
    assert argv[argv.index("--repeats") + 1] == "3"
    assert argv[argv.index("--max-decisions") + 1] == "12"
    assert argv[argv.index("--task-timeout") + 1] == "180"
    assert "--native-text-helper" in argv
    assert "--mps-call-maintenance" not in argv and "--oversized-head-policy" not in argv
    assert "--stable-observation" not in argv
    assert "--heaven-root" not in argv and "--coherence-root" not in argv
    assert kwargs["shell"] is False
    assert kwargs["cwd"] == RUNNER.ROOT
    result = receipt(queued_args)
    assert result["status"] == "completed"
    assert result["child"]["pid"] == 525252
    assert result["child"]["exit_code"] == 0
    assert result["child"]["started_at_utc"]
    assert result["child"]["argv"] == argv
    assert result["candidate_prefix_requested"] is False
    assert result["mps_call_maintenance_requested"] is False
    assert result["oversized_head_policy"] == "reject"
    assert "stable_observation_requested" not in result
    assert result["predecessors"][1]["identity"] == identity(command="python prefix.py")
    assert result["predecessors"][1]["status"] == "original_exited"
    assert (queued_args.output / "child.log").read_bytes() == b"child output retained\n"
    assert "scripts/run_ultrafast_after.py" in result["source_sha256"]
    assert "configs/benchmarks/ultrafast-external.json" in result["source_sha256"]


def test_original_pid_reuse_is_terminal_without_waiting_on_replacement(queued_args, monkeypatch):
    replacement = identity(start="Tue Oct 27 20:40:00 2026")
    _, sleeps, spawned = fake_processes(
        monkeypatch, {89007: [identity(), replacement], 97017: [identity(), None]}
    )
    assert RUNNER.supervise(queued_args) == 0
    assert len(spawned) == 1
    assert sleeps == []
    result = receipt(queued_args)
    assert result["predecessors"][0]["status"] == "original_ended_pid_reused"
    assert result["predecessors"][0]["last_observation"] == replacement


def test_default_child_argv_is_identical_to_original_launcher(queued_args):
    assert RUNNER.child_argv(queued_args) == [
        sys.executable,
        str(RUNNER.ROOT / "scripts/run_ultrafast.py"),
        "--upstream",
        str(queued_args.upstream.resolve()),
        "--output",
        str(queued_args.run_output.resolve()),
        "--manifest",
        str(queued_args.manifest.resolve()),
        "--model",
        str(queued_args.model.resolve()),
        "--device",
        "mps",
        "--profiles",
        "causal_multislot,independent",
        "--repeats",
        "3",
        "--max-decisions",
        "12",
        "--task-timeout",
        "180",
        "--native-text-helper",
    ]


def test_stability_flag_is_declared_before_wait_and_frozen_sources_are_bound(
    queued_args, monkeypatch
):
    baseline = RUNNER.child_argv(queued_args)
    queued_args.stable_observation = True

    def waiting():
        captured = receipt(queued_args)
        assert captured["stable_observation_requested"] is True
        for name in (
            "scripts/run_ultrafast.py",
            "scripts/ultrafast_browser.py",
            "scripts/run_ultrafast_after.py",
        ):
            archived = queued_args.output / "sources" / name
            assert archived.read_bytes() == (RUNNER.ROOT / name).read_bytes()
            assert captured["source_sha256"][name] == RUNNER.guard.sha256(archived)
        return None

    _, _, spawned = fake_processes(
        monkeypatch, {89007: [identity(), waiting], 97017: [identity(), None]}
    )
    assert RUNNER.supervise(queued_args) == 0
    assert spawned[0][0] == baseline + ["--stable-observation"]
    assert receipt(queued_args)["child"]["argv"] == spawned[0][0]


def test_queued_stability_source_change_aborts_before_child(queued_args, monkeypatch):
    queued_args.stable_observation = True

    def changed():
        (RUNNER.ROOT / "scripts/ultrafast_browser.py").write_text("changed while queued")
        return None

    _, _, spawned = fake_processes(
        monkeypatch, {89007: [identity(), changed], 97017: [identity(), None]}
    )
    with pytest.raises(RuntimeError, match="changed while queued"):
        RUNNER.supervise(queued_args)
    assert not spawned and receipt(queued_args)["status"] == "error"


@pytest.mark.parametrize(
    "maintenance,policy", [(True, "reject"), (False, "hierarchical"), (True, "hierarchical")]
)
def test_opt_in_flags_pass_to_same_interpreter_child_and_are_declared_before_wait(
    queued_args, monkeypatch, maintenance, policy
):
    baseline = RUNNER.child_argv(queued_args)
    queued_args.mps_call_maintenance = maintenance
    queued_args.oversized_head_policy = policy

    def assert_receipt_before_wait():
        captured = receipt(queued_args)
        assert captured["mps_call_maintenance_requested"] is maintenance
        assert captured["oversized_head_policy"] == policy
        return None

    _, _, spawned = fake_processes(
        monkeypatch, {89007: [identity(), assert_receipt_before_wait], 97017: [identity(), None]}
    )
    assert RUNNER.supervise(queued_args) == 0
    expected = baseline + (["--mps-call-maintenance"] if maintenance else [])
    if policy != "reject":
        expected.extend(["--oversized-head-policy", policy])
    assert spawned[0][0] == expected
    assert spawned[0][0][0] == sys.executable
    assert receipt(queued_args)["child"]["argv"] == expected


def test_missing_initial_handle_fails_closed_before_launch(queued_args, monkeypatch):
    _, _, spawned = fake_processes(monkeypatch, {89007: [None]})
    with pytest.raises(ValueError, match="not live at startup"):
        RUNNER.supervise(queued_args)
    assert spawned == []
    assert receipt(queued_args)["status"] == "error"


def test_transient_observation_failure_cannot_start_overlapping_mps_child(queued_args, monkeypatch):
    _, sleeps, spawned = fake_processes(
        monkeypatch,
        {89007: [identity(), RuntimeError("transient ps failure")], 97017: [identity()]},
    )
    with pytest.raises(RuntimeError, match="transient ps failure"):
        RUNNER.supervise(queued_args)
    assert spawned == [] and sleeps == []
    result = receipt(queued_args)
    assert result["status"] == "error"
    assert result["child"] is None


@pytest.mark.parametrize(
    "relative", ["src/s1/example.py", "configs/benchmarks/ultrafast-external.json"]
)
def test_archive_precedes_wait_and_changed_bytes_abort_before_child(
    queued_args, monkeypatch, relative
):
    source = RUNNER.ROOT / relative
    original = source.read_bytes()

    def changed_while_waiting():
        assert (queued_args.output / "sources" / relative).read_bytes() == original
        source.write_bytes(original + b"\n# changed while queued\n")
        return None

    _, _, spawned = fake_processes(
        monkeypatch, {89007: [identity(), changed_while_waiting], 97017: [identity(), None]}
    )
    with pytest.raises(RuntimeError, match="source changed while queued"):
        RUNNER.supervise(queued_args)
    assert spawned == []
    assert receipt(queued_args)["status"] == "error"


def test_child_failure_is_retained_and_returned(queued_args, monkeypatch):
    _, _, spawned = fake_processes(
        monkeypatch, {89007: [identity(), None], 97017: [identity(), None]}, exit_code=7
    )
    assert RUNNER.supervise(queued_args) == 7
    assert len(spawned) == 1
    result = receipt(queued_args)
    assert result["status"] == "failed"
    assert result["child"]["exit_code"] == 7
    assert result["child"]["finished_at_utc"]
    assert (queued_args.output / "child.log").read_bytes() == b"child output retained\n"


def test_child_identity_observation_error_does_not_orphan_popen_handle(queued_args, monkeypatch):
    events, _, _ = fake_processes(
        monkeypatch, {89007: [identity(), None], 97017: [identity(), None]}, identity_error=True
    )
    assert RUNNER.supervise(queued_args) == 0
    assert ("wait_child", 525252) in events
    result = receipt(queued_args)
    assert result["status"] == "completed"
    assert (
        result["child"]["identity_observation_error"]["message"]
        == "child ps observation interrupted"
    )


def test_existing_child_output_is_never_overwritten(queued_args, monkeypatch):
    queued_args.run_output.mkdir()
    sentinel = queued_args.run_output / "existing"
    sentinel.write_bytes(b"preserve")
    monkeypatch.setattr(
        RUNNER.guard, "process_identity", lambda *_: pytest.fail("no process reads")
    )
    with pytest.raises(FileExistsError, match="fresh"):
        RUNNER.supervise(queued_args)
    assert sentinel.read_bytes() == b"preserve"
    assert not queued_args.output.exists()


def test_new_child_output_appearing_during_wait_is_rejected(queued_args, monkeypatch):
    def appeared():
        queued_args.run_output.mkdir()
        return None

    _, _, spawned = fake_processes(
        monkeypatch, {89007: [identity(), appeared], 97017: [identity(), None]}
    )
    with pytest.raises(FileExistsError, match="appeared while queued"):
        RUNNER.supervise(queued_args)
    assert spawned == []


@pytest.mark.parametrize(
    "extra",
    [
        ["--predecessor-pid", "0"],
        ["--predecessor-pid", "2", "--predecessor-pid", "2"],
        ["--predecessor-pid", "2", "--repeats", "0"],
        ["--predecessor-pid", "2", "--task-timeout", "nan"],
        ["--predecessor-pid", "2", "--oversized-head-policy", "unknown"],
    ],
)
def test_cli_rejects_invalid_queue_or_run_budgets(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(RUNNER, "supervise", lambda *_: pytest.fail("must not launch"))
    with pytest.raises(SystemExit) as error:
        RUNNER.main(
            ["--output", str(tmp_path / "receipt"), "--run-output", str(tmp_path / "child"), *extra]
        )
    assert error.value.code == 2


@pytest.mark.parametrize(
    "extra,maintenance,policy",
    [
        ([], False, "reject"),
        (
            ["--mps-call-maintenance", "--oversized-head-policy", "hierarchical"],
            True,
            "hierarchical",
        ),
    ],
)
def test_cli_policy_passthrough_is_opt_in(tmp_path, monkeypatch, extra, maintenance, policy):
    captured = []
    monkeypatch.setattr(RUNNER, "supervise", lambda args: captured.append(args) or 0)
    assert (
        RUNNER.main(
            [
                "--predecessor-pid",
                "2",
                "--output",
                str(tmp_path / "receipt"),
                "--run-output",
                str(tmp_path / "child"),
                *extra,
            ]
        )
        == 0
    )
    assert captured[0].mps_call_maintenance is maintenance
    assert captured[0].oversized_head_policy == policy
