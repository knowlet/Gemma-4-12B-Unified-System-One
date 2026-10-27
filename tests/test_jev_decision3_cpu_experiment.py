import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import jev_decision3_cpu_experiment_watcher as watcher
import jev_decision3_cpu_experiment_worker as worker
import jev_decision3_cpu_padding_candidate as candidate
import pytest


@dataclass
class Question:
    keys: list
    kind: str = "choice"
    original: dict = field(default_factory=lambda: {"type": "choice"})
    rendered: dict = field(default_factory=lambda: {"type": "choice"})
    descriptions: list = field(default_factory=list)


@dataclass
class Prepared:
    keys: list
    questions: dict
    sequences: dict
    errors: dict = field(default_factory=dict)
    images: list = field(default_factory=list)
    videos: list = field(default_factory=list)
    media: dict = field(default_factory=dict)
    video_texts: list = field(default_factory=list)
    texts: dict = field(default_factory=dict)
    lengths: dict = field(default_factory=dict)
    reversed_sequences: dict = field(default_factory=dict)
    reversed_texts: dict = field(default_factory=dict)
    reversed_lengths: dict = field(default_factory=dict)


class FakeRuntime:
    model_name = "synthetic"
    device = "cpu"
    batch_size = 8
    permutation_average = False
    fast = None

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return text.split()

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(ids)

    tokenizer = Tokenizer()

    def system_one(self, *, state, questions):
        prepared = self.prepare(state, questions)
        probabilities, count = self.run(prepared)
        return self.respond(prepared, probabilities, count)

    def prepare(self, state, questions):
        keys = list(questions)
        lengths = tuple(len(questions[key]["instructions"].split()) + 60 for key in keys)
        return Prepared(
            keys,
            {key: Question(["alpha", "beta", "gamma"]) for key in keys},
            {key: [index] * lengths[index] for index, key in enumerate(keys)},
        )

    def run(self, value):
        # Deterministic synthetic variation checks that comparison uses all options.
        scores = [0.2, 0.3, 0.5] if len(value.keys) > 2 else [0.21, 0.29, 0.5]
        return {key: scores[:] for key in value.keys}, sum(
            len(value.sequences[key]) for key in value.keys
        )

    def respond(self, value, probabilities, tokens):
        return {
            "model": "synthetic",
            "answers": {
                key: {
                    "type": "choice",
                    "choice": "gamma",
                    "confidence": max(
                        0.0,
                        min(
                            1.0,
                            1
                            + sum(x * math.log(x) for x in probabilities[key] if x > 0)
                            / math.log(len(probabilities[key])),
                        ),
                    ),
                    "probabilities": dict(
                        zip(["alpha", "beta", "gamma"], probabilities[key], strict=True)
                    ),
                }
                for key in value.keys
            },
            "usage": {"input_tokens": tokens, "output_tokens": 0},
        }


@pytest.fixture
def experiment(monkeypatch):
    monkeypatch.setattr(candidate, "SOURCE_ONLY", False)
    return FakeRuntime(), candidate


def test_comparison_records_complete_interleaved_modes_and_usage(experiment):
    runtime, module = experiment
    result = worker._experiment(runtime, module, 4)
    assert len(result["samples"]) == 20
    assert len(result["comparisons"]) == 12
    assert list(result["head_token_lengths"].items()) == [
        ("synthetic_0", 3117),
        ("synthetic_1", 4196),
        ("synthetic_2", 4579),
        ("synthetic_3", 3126),
    ]
    assert {
        sample["mode"]: sample["metrics"]["padded_slots"]
        for sample in result["samples"]
        if sample["phase"] == "measured" and sample["cycle"] == 0 and sample["metrics"]
    } == {"whole": 18316, "ordered_pairs": 17550, "length_pairs": 15410}
    assert [x["mode"] for x in result["samples"][8:12]] == [
        "whole",
        "ordered_pairs",
        "length_pairs",
        "native",
    ]
    assert all(
        list(x["response"]["answers"]) == list(result["questions"]) for x in result["samples"]
    )
    assert all(
        x["response"]["usage"] == {"input_tokens": 15018, "output_tokens": 0}
        for x in result["samples"]
    )
    assert all(
        len(a["probabilities"]) == 3
        for row in result["samples"]
        for a in row["response"]["answers"].values()
    )
    assert {
        x["mode"]: x["max_probability_abs_diff"] for x in result["comparisons"] if x["cycle"] == 0
    }["ordered_pairs"] == pytest.approx(0.01)
    assert result["scope"].startswith("synthetic-only")


def test_experiment_refuses_incomplete_response(experiment):
    runtime, module = experiment
    old = runtime.respond

    def bad(*args):
        value = old(*args)
        value["answers"].pop("synthetic_0")
        return value

    runtime.respond = bad
    with pytest.raises(ValueError, match="response order"):
        worker._experiment(runtime, module, 1)


def test_source_only_gates_do_not_read_plan():
    with pytest.raises(ValueError, match="release required"):
        worker.run("/not/read", "a" * 64)
    with pytest.raises(ValueError, match="release required"):
        watcher.supervise("/not/read", "a" * 64, "/not/create")


def make_plan(tmp_path, monkeypatch):
    base = tmp_path / "experiments"
    base.mkdir()
    run = base / ("run-" + "a" * 32)
    run.mkdir()
    monkeypatch.setattr(worker, "EXPERIMENTS", base)
    monkeypatch.setattr(worker, "_source_set", lambda: {"/approved/source"})
    monkeypatch.setattr(worker, "_source_set_v2", lambda: {"/approved/source"})
    plan = {
        "schema": worker.SCHEMA,
        "unit": "jev-cpu-" + "a" * 32 + ".service",
        "output_dir": str(run / "worker"),
        "python": worker.common.GPU_PYTHON,
        "source_pins": {"/approved/source": "a" * 64},
        "wall_seconds": 60,
        "memory_floor_bytes": worker.common.MEMORY_FLOOR,
        "rounds": 4,
    }
    path = run / "plan.json"

    def save(value):
        raw = json.dumps(value, separators=(",", ":")).encode()
        path.write_bytes(raw)
        return str(path), hashlib.sha256(raw).hexdigest()

    return plan, path, save


def test_plan_rejects_unapproved_source_before_opening_it(tmp_path, monkeypatch):
    plan, path, save = make_plan(tmp_path, monkeypatch)
    assert worker._plan(*save(plan)) == plan
    plan["source_pins"]["/forbidden/heldout.py"] = "b" * 64
    with pytest.raises(ValueError, match="allowlisted"):
        worker._plan(*save(plan))
    assert not Path("/forbidden/heldout.py").exists()
    with pytest.raises(ValueError, match="dedicated"):
        worker._plan(str(tmp_path / "other.json"), "a" * 64)


@pytest.mark.parametrize(
    "field,value",
    [
        ("wall_seconds", 0),
        ("rounds", 0),
        ("unit", "other.service"),
        ("output_dir", "/tmp/other"),
        ("memory_floor_bytes", 0),
    ],
)
def test_plan_rejects_bad_limits_or_paths(tmp_path, monkeypatch, field, value):
    plan, _path, save = make_plan(tmp_path, monkeypatch)
    plan[field] = value
    with pytest.raises(ValueError):
        worker._plan(*save(plan))


def test_reviewed_release_only_changes_gate(tmp_path):
    held, released = tmp_path / "held.py", tmp_path / "released.py"
    held.write_bytes(b"SOURCE_ONLY = True\n")
    released.write_bytes(b"SOURCE_ONLY = False\n")
    pins = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (held, released)}
    watcher._verify_release(held, released, pins)
    released.write_bytes(b"SOURCE_ONLY = False\n# extra\n")
    pins[str(released)] = hashlib.sha256(released.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="gate-only"):
        watcher._verify_release(held, released, pins)


@pytest.mark.parametrize("profile_name,expected_threads", [(None, 5), ("cpu5", 5), ("cpu10", 10)])
def test_worker_runs_complete_fake_model_and_writes_private_result(
    tmp_path, monkeypatch, experiment, profile_name, expected_threads
):
    runtime, module = experiment
    output = tmp_path / "worker"
    output.mkdir(mode=0o700)
    invocation = "b" * 32
    plan = {"output_dir": str(output), "unit": "synthetic.service", "source_pins": {}, "rounds": 4}
    if profile_name is not None:
        plan.update(schema=worker.SCHEMA_V2, cpu_profile=profile_name)
    monkeypatch.setattr(worker, "SOURCE_ONLY", False)
    monkeypatch.setattr(worker, "_plan", lambda *_: plan)
    monkeypatch.setattr(worker.os, "getpgrp", os.getpid)
    monkeypatch.setattr(worker.common, "_own_cgroup", lambda _: "/synthetic")
    monkeypatch.setattr(worker, "_own_profile_cgroup", lambda _unit, _profile: "/synthetic")
    monkeypatch.setattr(worker.common, "_verify_source_pins", lambda _: None)
    monkeypatch.setattr(worker.common, "_validate_interpreter", lambda _: None)
    monkeypatch.setattr(worker.common.systemd, "boot_id", lambda: "boot")
    monkeypatch.setattr(worker.common.systemd, "process_identity", lambda _: {"startticks": 7})
    monkeypatch.setenv("INVOCATION_ID", invocation)
    thread_calls = []
    torch = SimpleNamespace(
        set_num_threads=lambda value: thread_calls.append(("threads", value)),
        set_num_interop_threads=lambda value: thread_calls.append(("interop", value)),
    )
    monkeypatch.setattr(worker.common, "_bind_overlay", lambda *_: torch)
    monkeypatch.setattr(worker, "_verify_model_manifest", lambda *_: "manifest")

    def from_pretrained(path, **kwargs):
        assert kwargs == {
            "device": "cpu",
            "batch_size": 8,
            "verify": "full",
            "local_files_only": True,
            "permutation_average": False,
        }
        return runtime

    monkeypatch.setattr(
        worker,
        "_load_publisher",
        lambda *_: SimpleNamespace(D3=SimpleNamespace(from_pretrained=from_pretrained)),
    )
    release_calls = []

    def load_release(*args):
        release_calls.append(args)
        return module

    monkeypatch.setattr(worker.common, "_load_reviewed_release", load_release)
    write = worker.common._write_new

    def write_and_go(path, data):
        write(path, data)
        if path.name == "hello.json":
            write(
                output / "go.json",
                worker.child._encode(
                    {
                        "schema": "decision3-cpu-go-v1",
                        "invocation": invocation,
                        "plan_sha256": "c" * 64,
                    }
                ),
            )

    monkeypatch.setattr(worker.common, "_write_new", write_and_go)
    worker.run("synthetic", "c" * 64)
    expected_release = (
        "jev_decision3_cpu_padding_candidate_v2_release.py"
        if profile_name
        else "jev_decision3_cpu_padding_candidate_release.py"
    )
    assert Path(release_calls[0][0]).name == expected_release
    assert thread_calls == [("threads", expected_threads), ("interop", 1)]
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert os.environ[name] == str(expected_threads)
    result = json.loads((output / "result.json").read_bytes())
    terminal = json.loads((output / "terminal.json").read_bytes())
    assert len(result["samples"]) == 20 and len(result["comparisons"]) == 12
    if profile_name is None:
        assert "cpu_profile" not in result
    else:
        assert result["cpu_profile"] == profile_name
    assert (
        terminal["result_sha256"]
        == hashlib.sha256((output / "result.json").read_bytes()).hexdigest()
    )
    assert terminal["status"] == "completed_pending_external_audit"


@pytest.mark.parametrize("diagnostic_case", ["complete", "missing", "failure"])
def test_watcher_supervises_fake_unit_to_verified_closure(tmp_path, monkeypatch, diagnostic_case):
    run = tmp_path / "run"
    run.mkdir()
    output = run / "worker"
    output.mkdir(mode=0o700)
    report = run / "watcher"
    plan_path = run / "plan.json"
    plan = {
        "output_dir": str(output),
        "unit": "jev-cpu-" + "a" * 32 + ".service",
        "source_pins": {},
        "memory_floor_bytes": worker.common.MEMORY_FLOOR,
        "wall_seconds": 60,
        "python": worker.common.GPU_PYTHON,
    }
    invocation = "d" * 32
    digest = "e" * 64
    pid = 321
    group = "/user.slice/app.slice/" + plan["unit"]
    running = {
        "LoadState": "loaded",
        "ActiveState": "active",
        "SubState": "running",
        "InvocationID": invocation,
        "ControlGroup": group,
        "MainPID": str(pid),
        "Type": "exec",
        "MemoryAccounting": "yes",
        "MemoryMax": str(worker.common.MEMORY_MAX),
        "MemorySwapMax": "0",
        "OOMPolicy": "kill",
        "KillMode": "control-group",
        "RuntimeMaxUSec": "1min",
        "TimeoutStopUSec": "3s",
    }
    terminal = {
        **running,
        "SubState": "exited",
        "MainPID": "0",
        "ExecMainCode": "exited",
        "ExecMainStatus": "0",
        "Result": "success",
        "ControlGroup": "",
    }
    snapshots = iter(
        [
            {"LoadState": "not-found", "ActiveState": "inactive", "MainPID": "0"},
            running,
            terminal,
            terminal,
        ]
    )
    monkeypatch.setattr(watcher, "SOURCE_ONLY", False)
    monkeypatch.setattr(worker, "_plan", lambda *_: plan)
    monkeypatch.setattr(watcher.common, "_validate_interpreter", lambda *_: None)
    monkeypatch.setattr(watcher.common, "_verify_source_pins", lambda *_: None)
    monkeypatch.setattr(watcher, "_verify_release", lambda *_: None)
    monkeypatch.setattr(
        watcher.common,
        "_observe",
        lambda *_: {"available_bytes": 100000000000, "running": 0, "pending": 0},
    )
    monkeypatch.setattr(
        watcher.common.systemd, "service_show", lambda *_args, **_kwargs: next(snapshots)
    )
    monkeypatch.setattr(watcher.common.systemd, "boot_id", lambda: "boot")
    monkeypatch.setattr(watcher.common.systemd, "process_identity", lambda _: {"startticks": 7})
    monkeypatch.setattr(watcher.common.systemd, "cgroup_membership", lambda _: group)
    monkeypatch.setattr(
        watcher.common.systemd, "closed_unit_metadata", lambda *_args, **_kwargs: {"verified": True}
    )
    monkeypatch.setattr(watcher.os, "getpgid", lambda _: pid)
    monkeypatch.setattr(watcher.shared, "_live_limits", lambda *_: None)
    monkeypatch.setattr(watcher.common, "_closed_cgroup", lambda *_: "absent")
    monkeypatch.setattr(watcher, "_closed_group", lambda _: "absent")
    monkeypatch.setattr(watcher.time, "sleep", lambda _: None)

    def launch(*_args, **_kwargs):
        hello = {
            "schema": "decision3-cpu-hello-v1",
            "pid": pid,
            "startticks": 7,
            "boot_id": "boot",
            "unit": plan["unit"],
            "invocation": invocation,
            "plan_sha256": digest,
            "pgid": pid,
            "cgroup": group,
        }
        watcher.common._write_new(output / "hello.json", child_encode(hello))

    monkeypatch.setattr(watcher.subprocess, "run", launch)
    publish = watcher.child._publish_control

    def complete(path, raw):
        publish(path, raw)
        recorder = worker._StageRecorder(output, invocation, digest)
        if diagnostic_case != "missing":
            for stage in worker._STAGES:
                recorder(stage)
        if diagnostic_case == "failure":
            recorder.failure(ValueError("synthetic error"))
        result = b'{"fake":"synthetic"}'
        watcher.common._write_new(output / "result.json", result)
        status = {
            "schema": "decision3-cpu-terminal-v1",
            "status": "completed_pending_external_audit",
            "invocation": invocation,
            "plan_sha256": digest,
            "result_sha256": hashlib.sha256(result).hexdigest(),
        }
        watcher.common._write_new(output / "terminal.json", child_encode(status))

    monkeypatch.setattr(watcher.child, "_publish_control", complete)
    if diagnostic_case != "complete":
        with pytest.raises(ValueError, match="private diagnostic invalid"):
            watcher.supervise(str(plan_path), digest, str(report))
        closure = json.loads((report / "closure.json").read_bytes())
        assert closure["status"] == "failed"
        assert closure["cgroup"] == closure["process_group"] == "absent"
        return
    closure = watcher.supervise(str(plan_path), digest, str(report))
    assert closure["status"] == "completed_pending_external_audit"
    assert closure["cgroup"] == "absent" and closure["process_group"] == "absent"
    assert (output / "go.json").is_file() and (report / "closure.json").is_file()
    events = [json.loads(line) for line in (report / "events.jsonl").read_bytes().splitlines()]
    assert [event["event"] for event in events] == ["prelaunch", "host", "binding_verified", "host"]
    assert closure["machine_exclusive"] is False
    assert closure["diagnostic"] == {
        "last_entered_stage": "result_write",
        "failure_category": None,
    }


def child_encode(value):
    return json.dumps(value, separators=(",", ":")).encode()


def test_worker_model_manifest_rejects_outside_symlink_before_read(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    snapshot = cache / "snapshot"
    snapshot.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"must not read")
    (snapshot / "weights").symlink_to(outside)
    raw = json.dumps({"files_sha256": {"weights": "a" * 64}}).encode()
    (snapshot / "MODEL_MANIFEST.json").write_bytes(raw)
    monkeypatch.setattr(worker, "CACHE", cache)
    monkeypatch.setattr(worker, "SNAPSHOT", snapshot)
    monkeypatch.setattr(worker, "MANIFEST", hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="escaped"):
        worker._verify_model_manifest({})


def test_watcher_refuses_busy_host_before_launch(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    output = run / "worker"
    output.mkdir(mode=0o700)
    report = run / "watcher"
    plan = {
        "output_dir": str(output),
        "unit": "jev-cpu-" + "a" * 32 + ".service",
        "source_pins": {},
        "memory_floor_bytes": worker.common.MEMORY_FLOOR,
        "wall_seconds": 60,
        "python": worker.common.GPU_PYTHON,
    }
    monkeypatch.setattr(watcher, "SOURCE_ONLY", False)
    monkeypatch.setattr(worker, "_plan", lambda *_: plan)
    monkeypatch.setattr(watcher.common, "_validate_interpreter", lambda *_: None)
    monkeypatch.setattr(watcher.common, "_verify_source_pins", lambda *_: None)
    monkeypatch.setattr(watcher, "_verify_release", lambda *_: None)

    def busy(*_args):
        raise ValueError("Comfy queue is not idle")

    monkeypatch.setattr(watcher.common, "_observe", busy)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("must not launch")

    monkeypatch.setattr(watcher.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="not idle"):
        watcher.supervise(str(run / "plan.json"), "f" * 64, str(report))
    closure = json.loads((report / "closure.json").read_bytes())
    assert closure["status"] == "failed"
    assert closure["cgroup"] == closure["process_group"] == "not_started"
    assert list(output.iterdir()) == []


def test_unit_limits_are_hard_and_explicit():
    plan = {
        "unit": "jev-cpu-" + "a" * 32 + ".service",
        "python": worker.common.GPU_PYTHON,
        "wall_seconds": 600,
    }
    args = watcher._unit_args(plan, "/tmp/plan", "b" * 64)
    assert "--property=MemoryMax=8589934592" in args
    assert "--property=MemorySwapMax=0" in args
    assert "--property=CPUQuota=500%" in args
    assert "--property=AllowedCPUs=15-19" in args
    assert "--property=KillMode=control-group" in args
    assert "--property=RuntimeMaxSec=600" in args
    assert args[args.index("/usr/bin/taskset") + 1 :][:2] == ["-c", "15-19"]


def test_process_group_must_be_absent(monkeypatch):
    monkeypatch.setattr(watcher.os, "killpg", lambda *_: None)
    with pytest.raises(ValueError, match="still exists"):
        watcher._closed_group(456)

    def absent(*_args):
        raise ProcessLookupError()

    monkeypatch.setattr(watcher.os, "killpg", absent)
    assert watcher._closed_group(456) == "absent"


def test_terminal_result_must_match_raw_bytes(tmp_path):
    (tmp_path / "result.json").write_bytes(b"synthetic")
    terminal = {
        "schema": "decision3-cpu-terminal-v1",
        "status": "completed_pending_external_audit",
        "invocation": "a" * 32,
        "plan_sha256": "b" * 64,
        "result_sha256": "c" * 64,
    }
    (tmp_path / "terminal.json").write_bytes(child_encode(terminal))
    with pytest.raises(ValueError, match="hash differs"):
        watcher._terminal(tmp_path, "a" * 32, "b" * 64)


def test_watcher_stops_owned_worker_when_comfy_starts(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    output = run / "worker"
    output.mkdir(mode=0o700)
    report = run / "watcher"
    plan_path = run / "plan.json"
    plan = {
        "output_dir": str(output),
        "unit": "jev-cpu-" + "a" * 32 + ".service",
        "source_pins": {},
        "memory_floor_bytes": worker.common.MEMORY_FLOOR,
        "wall_seconds": 60,
        "python": worker.common.GPU_PYTHON,
    }
    invocation, digest, pid = "d" * 32, "e" * 64, 321
    group = "/user.slice/app.slice/" + plan["unit"]
    running = {
        "LoadState": "loaded",
        "ActiveState": "active",
        "SubState": "running",
        "InvocationID": invocation,
        "ControlGroup": group,
        "MainPID": str(pid),
        "Type": "exec",
        "MemoryAccounting": "yes",
        "MemoryMax": str(worker.common.MEMORY_MAX),
        "MemorySwapMax": "0",
        "OOMPolicy": "kill",
        "KillMode": "control-group",
        "RuntimeMaxUSec": "1min",
        "TimeoutStopUSec": "3s",
    }
    stopped = {
        **running,
        "ActiveState": "inactive",
        "SubState": "dead",
        "MainPID": "0",
        "ControlGroup": "",
    }
    snapshots = iter(
        [{"LoadState": "not-found", "ActiveState": "inactive", "MainPID": "0"}, running, stopped]
    )
    monkeypatch.setattr(watcher, "SOURCE_ONLY", False)
    monkeypatch.setattr(worker, "_plan", lambda *_: plan)
    monkeypatch.setattr(watcher.common, "_validate_interpreter", lambda *_: None)
    monkeypatch.setattr(watcher.common, "_verify_source_pins", lambda *_: None)
    monkeypatch.setattr(watcher, "_verify_release", lambda *_: None)
    observations = iter(
        [
            {"available_bytes": 100000000000, "running": 0, "pending": 0},
            {"available_bytes": 100000000000, "running": 0, "pending": 0},
            None,
        ]
    )

    def observe(*_args):
        value = next(observations)
        if value is None:
            raise ValueError("Comfy queue is not idle")
        return value

    monkeypatch.setattr(watcher.common, "_observe", observe)
    monkeypatch.setattr(
        watcher.common.systemd, "service_show", lambda *_args, **_kwargs: next(snapshots)
    )
    monkeypatch.setattr(watcher.common.systemd, "boot_id", lambda: "boot")
    monkeypatch.setattr(watcher.common.systemd, "process_identity", lambda _: {"startticks": 7})
    monkeypatch.setattr(watcher.common.systemd, "cgroup_membership", lambda _: group)
    monkeypatch.setattr(
        watcher.common.systemd, "closed_unit_metadata", lambda *_args, **_kwargs: {"verified": True}
    )
    monkeypatch.setattr(watcher.os, "getpgid", lambda _: pid)
    monkeypatch.setattr(watcher.shared, "_live_limits", lambda *_: None)
    monkeypatch.setattr(watcher.common, "_closed_cgroup", lambda *_: "absent")
    monkeypatch.setattr(watcher, "_closed_group", lambda _: "absent")
    monkeypatch.setattr(watcher.time, "sleep", lambda _: None)
    stopped_units = []
    monkeypatch.setattr(
        watcher.common, "_stop_same", lambda unit, token: stopped_units.append((unit, token))
    )

    def launch(*_args, **_kwargs):
        hello = {
            "schema": "decision3-cpu-hello-v1",
            "pid": pid,
            "startticks": 7,
            "boot_id": "boot",
            "unit": plan["unit"],
            "invocation": invocation,
            "plan_sha256": digest,
            "pgid": pid,
            "cgroup": group,
        }
        watcher.common._write_new(output / "hello.json", child_encode(hello))

    monkeypatch.setattr(watcher.subprocess, "run", launch)
    with pytest.raises(ValueError, match="not idle"):
        watcher.supervise(str(plan_path), digest, str(report))
    closure = json.loads((report / "closure.json").read_bytes())
    assert stopped_units == [(plan["unit"], invocation)]
    assert closure["status"] == "failed"
    assert closure["cgroup"] == closure["process_group"] == "absent"
    assert (output / "go.json").is_file() and not (output / "terminal.json").exists()


def test_publisher_loader_executes_only_pinned_synthetic_sources(tmp_path, monkeypatch):
    import sys

    staged, snapshot = tmp_path / "staged", tmp_path / "snapshot"
    staged.mkdir()
    snapshot.mkdir()
    sources = {
        staged / "d3_format.py": b"VALUE = 7\n",
        snapshot / "d3_fast.py": b"VALUE = 11\n",
        staged
        / "d3_runtime.py": b"from . import d3_fast, d3_format\nVALUE = d3_fast.VALUE + d3_format.VALUE\n",
    }
    for path, raw in sources.items():
        path.write_bytes(raw)
    monkeypatch.setattr(worker, "STAGED", staged)
    monkeypatch.setattr(worker, "SNAPSHOT", snapshot)
    pins = {str(path): hashlib.sha256(raw).hexdigest() for path, raw in sources.items()}
    try:
        module = worker._load_publisher(pins)
        assert module.VALUE == 18
        with pytest.raises(ValueError, match="fresh publisher"):
            worker._load_publisher(pins)
    finally:
        for name in tuple(sys.modules):
            if name == "_cpu_experiment_publisher" or name.startswith("_cpu_experiment_publisher."):
                del sys.modules[name]


def test_publisher_loader_rejects_changed_source_and_cleans_its_modules(tmp_path, monkeypatch):
    import sys

    staged, snapshot = tmp_path / "staged", tmp_path / "snapshot"
    staged.mkdir()
    snapshot.mkdir()
    sources = {
        staged / "d3_format.py": b"VALUE = 7\n",
        snapshot / "d3_fast.py": b"VALUE = 11\n",
        staged / "d3_runtime.py": b"VALUE = 18\n",
    }
    for path, raw in sources.items():
        path.write_bytes(raw)
    monkeypatch.setattr(worker, "STAGED", staged)
    monkeypatch.setattr(worker, "SNAPSHOT", snapshot)
    pins = {str(path): hashlib.sha256(raw).hexdigest() for path, raw in sources.items()}
    (snapshot / "d3_fast.py").write_bytes(b"VALUE = 12\n")
    with pytest.raises(ValueError, match="changed"):
        worker._load_publisher(pins)
    assert not any(
        name == "_cpu_experiment_publisher" or name.startswith("_cpu_experiment_publisher.")
        for name in sys.modules
    )


def test_manifest_pins_must_match_all_publisher_code(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    snapshot = cache / "snapshot"
    staged = tmp_path / "staged"
    snapshot.mkdir(parents=True)
    staged.mkdir()
    names = ("d3_format.py", "d3_runtime.py", "modeling_d3.py", "d3_fast.py")
    files = {name: "a" * 64 for name in names}
    raw = json.dumps({"files_sha256": files}).encode()
    (snapshot / "MODEL_MANIFEST.json").write_bytes(raw)
    monkeypatch.setattr(worker, "CACHE", cache)
    monkeypatch.setattr(worker, "SNAPSHOT", snapshot)
    monkeypatch.setattr(worker, "STAGED", staged)
    monkeypatch.setattr(worker, "MANIFEST", hashlib.sha256(raw).hexdigest())
    pins = {str((snapshot if name == "d3_fast.py" else staged) / name): "a" * 64 for name in names}
    assert worker._verify_model_manifest(pins) == hashlib.sha256(raw).hexdigest()
    pins[str(snapshot / "d3_fast.py")] = "b" * 64
    with pytest.raises(ValueError, match="publisher pin differs"):
        worker._verify_model_manifest(pins)


@pytest.mark.parametrize("launch_timeout", [True, False])
def test_watcher_recovers_identity_after_launch_race(tmp_path, monkeypatch, launch_timeout):
    import subprocess

    run = tmp_path / "run"
    run.mkdir()
    output = run / "worker"
    output.mkdir(mode=0o700)
    report = run / "watcher"
    plan = {
        "output_dir": str(output),
        "unit": "jev-cpu-" + "a" * 32 + ".service",
        "source_pins": {},
        "memory_floor_bytes": worker.common.MEMORY_FLOOR,
        "wall_seconds": 60,
        "python": worker.common.GPU_PYTHON,
    }
    invocation, digest, group = "d" * 32, "e" * 64, "/synthetic"
    hello = {"invocation": invocation, "plan_sha256": digest, "pgid": 321}
    monkeypatch.setattr(watcher, "SOURCE_ONLY", False)
    monkeypatch.setattr(worker, "_plan", lambda *_: plan)
    monkeypatch.setattr(watcher.common, "_validate_interpreter", lambda *_: None)
    monkeypatch.setattr(watcher.common, "_verify_source_pins", lambda *_: None)
    monkeypatch.setattr(watcher, "_verify_release", lambda *_: None)
    monkeypatch.setattr(watcher.common, "_observe", lambda *_: {})
    calls = 0

    def show(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"LoadState": "not-found", "ActiveState": "inactive", "MainPID": "0"}
        if calls == 2 and not launch_timeout:
            raise TimeoutError("first service query")
        return {"InvocationID": invocation}

    monkeypatch.setattr(watcher.common.systemd, "service_show", show)

    def bind(properties, received, checked_plan, checked_sha):
        assert properties["InvocationID"] == invocation
        assert received == hello and checked_plan == plan and checked_sha == digest
        return invocation, group

    monkeypatch.setattr(watcher, "_bind", bind)
    monkeypatch.setattr(
        watcher.common.systemd, "closed_unit_metadata", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(watcher.common, "_closed_cgroup", lambda *_: "absent")
    monkeypatch.setattr(watcher, "_closed_group", lambda *_: "absent")
    stopped = []
    monkeypatch.setattr(
        watcher.common, "_stop_same", lambda unit, token: stopped.append((unit, token))
    )

    def launch(*_args, **_kwargs):
        watcher.common._write_new(output / "hello.json", child_encode(hello))
        if launch_timeout:
            raise subprocess.TimeoutExpired("systemd-run", 10)

    monkeypatch.setattr(watcher.subprocess, "run", launch)
    with pytest.raises((subprocess.TimeoutExpired, TimeoutError)):
        watcher.supervise(str(run / "plan.json"), digest, str(report))
    assert stopped == [(plan["unit"], invocation)]
    closure = json.loads((report / "closure.json").read_bytes())
    assert closure["status"] == "failed"
    assert closure["cgroup"] == closure["process_group"] == "absent"
    assert not (output / "go.json").exists()


def test_recovery_never_stops_unproven_identity(tmp_path, monkeypatch):
    now = iter([0, 1, 2, 10])
    monkeypatch.setattr(watcher.time, "monotonic", lambda: next(now))
    monkeypatch.setattr(watcher.time, "sleep", lambda *_: None)
    monkeypatch.setattr(watcher.common.systemd, "service_show", lambda *_args, **_kwargs: {})
    with pytest.raises(ValueError, match="cannot prove"):
        watcher._recover_identity(tmp_path, {"unit": "synthetic"}, "a" * 64)


def test_watcher_does_not_stop_when_hello_cannot_be_bound(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    output = run / "worker"
    output.mkdir(mode=0o700)
    report = run / "watcher"
    plan = {
        "output_dir": str(output),
        "unit": "jev-cpu-" + "a" * 32 + ".service",
        "source_pins": {},
        "memory_floor_bytes": worker.common.MEMORY_FLOOR,
        "wall_seconds": 60,
        "python": worker.common.GPU_PYTHON,
    }
    invocation, group, digest = "d" * 32, "/untrusted", "e" * 64
    monkeypatch.setattr(watcher, "SOURCE_ONLY", False)
    monkeypatch.setattr(worker, "_plan", lambda *_: plan)
    monkeypatch.setattr(watcher.common, "_validate_interpreter", lambda *_: None)
    monkeypatch.setattr(watcher.common, "_verify_source_pins", lambda *_: None)
    monkeypatch.setattr(watcher, "_verify_release", lambda *_: None)
    monkeypatch.setattr(watcher.common, "_observe", lambda *_: {})
    snapshots = iter(
        [
            {"LoadState": "not-found", "ActiveState": "inactive", "MainPID": "0"},
            {"LoadState": "loaded", "InvocationID": invocation},
        ]
    )
    monkeypatch.setattr(watcher.common.systemd, "service_show", lambda *_a, **_kw: next(snapshots))
    monkeypatch.setattr(watcher.shared, "_unit_identity", lambda *_: (invocation, group))

    def reject(*_args):
        raise ValueError("untrusted handshake")

    monkeypatch.setattr(watcher, "_bind", reject)
    monkeypatch.setattr(watcher, "_recover_identity", reject)
    stopped = []
    monkeypatch.setattr(watcher.common, "_stop_same", lambda *args: stopped.append(args))
    monkeypatch.setattr(
        watcher.subprocess,
        "run",
        lambda *_a, **_kw: watcher.common._write_new(output / "hello.json", b"{}"),
    )
    with pytest.raises(ValueError, match="untrusted handshake"):
        watcher.supervise(str(run / "plan.json"), digest, str(report))
    closure = json.loads((report / "closure.json").read_bytes())
    assert not stopped
    assert closure["status"] == "cleanup_unconfirmed"
    assert closure["cgroup"] == closure["process_group"] == "unconfirmed"
    assert not (output / "go.json").exists()


@pytest.mark.parametrize("failed_stage", worker._STAGES)
def test_worker_records_private_failure_at_each_stage(tmp_path, monkeypatch, failed_stage):
    output = tmp_path / "worker"
    output.mkdir(mode=0o700)
    invocation, digest = "b" * 32, "c" * 64
    plan = {"output_dir": str(output), "unit": "synthetic.service", "source_pins": {}}
    monkeypatch.setattr(worker, "SOURCE_ONLY", False)
    monkeypatch.setattr(worker, "_plan", lambda *_: plan)
    monkeypatch.setattr(worker.os, "getpgrp", os.getpid)
    monkeypatch.setattr(worker.common, "_own_cgroup", lambda *_: "/synthetic")
    monkeypatch.setattr(worker.common, "_verify_source_pins", lambda *_: None)
    monkeypatch.setattr(worker.common, "_validate_interpreter", lambda *_: None)
    monkeypatch.setattr(worker.common.systemd, "boot_id", lambda: "boot")
    monkeypatch.setattr(worker.common.systemd, "process_identity", lambda *_: {"startticks": 7})
    monkeypatch.setenv("INVOCATION_ID", invocation)
    write = worker.common._write_new

    def write_and_go(path, data):
        write(path, data)
        if path.name == "hello.json":
            write(
                output / "go.json",
                worker.child._encode(
                    {
                        "schema": "decision3-cpu-go-v1",
                        "invocation": invocation,
                        "plan_sha256": digest,
                    }
                ),
            )

    monkeypatch.setattr(worker.common, "_write_new", write_and_go)

    def fail_at_stage(_plan, _output, _invocation, _digest, stage):
        for name in worker._STAGES:
            stage(name)
            if name == failed_stage:
                raise ValueError("SECRET-PROMPT private-path")

    monkeypatch.setattr(worker, "_run_after_go", fail_at_stage)
    with pytest.raises(ValueError, match="SECRET-PROMPT"):
        worker.run("synthetic", digest)
    assert watcher._diagnostic(output, invocation, digest) == {
        "last_entered_stage": failed_stage,
        "failure_category": "validation",
    }
    failure = (output / "failure.json").read_bytes()
    assert b"SECRET-PROMPT" not in failure and b"private-path" not in failure
    assert (output / "failure.json").stat().st_mode & 0o077 == 0
    assert not (output / "terminal.json").exists()


def test_hard_stop_leaves_last_entered_stage_without_inventing_a_reason(tmp_path):
    recorder = worker._StageRecorder(tmp_path, "a" * 32, "b" * 64)
    recorder("overlay")
    recorder("model_load")
    assert watcher._diagnostic(tmp_path, "a" * 32, "b" * 64) == {
        "last_entered_stage": "model_load",
        "failure_category": None,
    }
    assert not (tmp_path / "failure.json").exists()


@pytest.mark.parametrize(
    "fault", ["wrong_invocation", "wrong_plan", "extra_field", "gap", "wrong_category"]
)
def test_diagnostic_rejects_forged_or_incomplete_markers(tmp_path, fault):
    invocation, digest = "a" * 32, "b" * 64
    recorder = worker._StageRecorder(tmp_path, invocation, digest)
    recorder("overlay")
    recorder("model_load")
    if fault == "gap":
        (tmp_path / "stage-00.json").unlink()
    elif fault == "wrong_category":
        recorder.failure(ValueError("private"))
        path = tmp_path / "failure.json"
        value = json.loads(path.read_bytes())
        value["category"] = "secret"
        path.write_text(json.dumps(value))
    else:
        path = tmp_path / "stage-01.json"
        value = json.loads(path.read_bytes())
        if fault == "wrong_invocation":
            value["invocation"] = "c" * 32
        elif fault == "wrong_plan":
            value["plan_sha256"] = "d" * 64
        else:
            value["extra"] = "secret"
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="diagnostic"):
        watcher._diagnostic(tmp_path, invocation, digest)


def test_missing_markers_do_not_claim_a_failure_stage(tmp_path):
    assert watcher._diagnostic(tmp_path, "a" * 32, "b" * 64) == {
        "last_entered_stage": None,
        "failure_category": None,
    }


@pytest.mark.parametrize("name", ["cpu5", "cpu10"])
def test_v2_plan_requires_known_immutable_profile(tmp_path, monkeypatch, name):
    plan, _path, save = make_plan(tmp_path, monkeypatch)
    plan.update(schema=worker.SCHEMA_V2, cpu_profile=name)
    assert worker._plan(*save(plan)) == plan
    plan["cpu_profile"] = "arbitrary"
    with pytest.raises(ValueError, match="unknown CPU profile"):
        worker._plan(*save(plan))
    plan.pop("cpu_profile")
    with pytest.raises(ValueError, match="exact CPU experiment plan"):
        worker._plan(*save(plan))


def test_v1_plan_does_not_allow_profile_field(tmp_path, monkeypatch):
    plan, _path, save = make_plan(tmp_path, monkeypatch)
    plan["cpu_profile"] = "cpu10"
    with pytest.raises(ValueError, match="exact CPU experiment plan"):
        worker._plan(*save(plan))


@pytest.mark.parametrize(
    "name,cpulist,quota", [("cpu5", "15-19", "500%"), ("cpu10", "5-9,15-19", "1000%")]
)
def test_v2_unit_has_exact_profile_limits(name, cpulist, quota):
    plan = {
        "schema": worker.SCHEMA_V2,
        "cpu_profile": name,
        "unit": "synthetic.service",
        "python": worker.common.GPU_PYTHON,
        "wall_seconds": 600,
    }
    args = watcher._unit_args(plan, "/tmp/plan", "a" * 64)
    assert "--property=CPUQuota=" + quota in args
    assert "--property=AllowedCPUs=" + cpulist in args
    assert args[args.index("/usr/bin/taskset") + 1 :][:2] == ["-c", cpulist]
    assert "--property=MemoryMax=8589934592" in args
    assert "--property=MemorySwapMax=0" in args
    assert "--property=RuntimeMaxSec=600" in args


@pytest.mark.parametrize("name", ["cpu5", "cpu10"])
@pytest.mark.parametrize("fault", [None, "affinity", "cpuset", "quota", "memory", "swap"])
def test_v2_checks_actual_cgroup_limits(tmp_path, monkeypatch, name, fault):
    group = "/app.slice/synthetic.service"
    cg = tmp_path / group.lstrip("/")
    cg.mkdir(parents=True)
    monkeypatch.setattr(worker, "CGROUP_ROOT", tmp_path)
    profile = worker._profile({"schema": worker.SCHEMA_V2, "cpu_profile": name})
    (cg / "cpu.max").write_text(f"{profile['threads'] * 100000} 100000")
    (cg / "cpuset.cpus.effective").write_text(profile["cpu_list"])
    (cg / "memory.max").write_text(str(worker.common.MEMORY_MAX))
    (cg / "memory.swap.max").write_text("0")
    affinity = set(profile["cpus"])
    if fault == "affinity":
        affinity.remove(min(affinity))
    elif fault == "cpuset":
        (cg / "cpuset.cpus.effective").write_text("15-19") if name == "cpu10" else (
            cg / "cpuset.cpus.effective"
        ).write_text("15-18")
    elif fault == "quota":
        (cg / "cpu.max").write_text("100000 100000")
    elif fault == "memory":
        (cg / "memory.max").write_text("max")
    elif fault == "swap":
        (cg / "memory.swap.max").write_text("1")
    monkeypatch.setattr(worker.os, "sched_getaffinity", lambda _: affinity)
    if fault is None:
        worker._cpu_limits(group, 123, profile)
    else:
        with pytest.raises(ValueError, match="actual worker CPU"):
            worker._cpu_limits(group, 123, profile)


def test_v2_cpu_telemetry_records_ancestors_or_unverified(tmp_path, monkeypatch):
    leaf = tmp_path / "app.slice" / "run.service"
    leaf.mkdir(parents=True)
    monkeypatch.setattr(worker, "CGROUP_ROOT", tmp_path)
    for path, quota in (
        (leaf, "1000000 100000"),
        (leaf.parent, "500000 100000"),
        (tmp_path, "max 100000"),
    ):
        (path / "cpu.max").write_text(quota)
        (path / "cpu.stat").write_text("nr_periods 19\nnr_throttled 2\nthrottled_usec 345\n")
    value = watcher._cpu_observation("/app.slice/run.service")
    assert value["status"] == "observed"
    assert [x["cpu_max"] for x in value["levels"]] == [
        "1000000 100000",
        "500000 100000",
        "max 100000",
    ]
    assert value["levels"][1]["cpu_stat"]["throttled_usec"] == 345
    (leaf.parent / "cpu.stat").unlink()
    value = watcher._cpu_observation("/app.slice/run.service")
    assert value["status"] == "partial"
    assert value["levels"][0]["quota_status"] == "observed"
    assert value["levels"][1]["cpu_max"] == "500000 100000"
    assert value["levels"][1]["throttle_status"] == "unverified"
    (tmp_path / "cpu.max").unlink()
    value = watcher._cpu_observation("/app.slice/run.service")
    assert value["status"] == "partial"
    assert value["levels"][2]["quota_status"] == "root_missing"
    assert value["levels"][2]["cpu_max"] is None
    assert value["levels"][0]["cpu_max"] == "1000000 100000"


def test_v2_watcher_binding_checks_the_selected_profile(monkeypatch):
    invocation, unit, pid = "d" * 32, "synthetic.service", 123
    group, digest = "/user.slice/app.slice/" + unit, "e" * 64
    plan = {"schema": worker.SCHEMA_V2, "cpu_profile": "cpu10", "unit": unit, "wall_seconds": 60}
    properties = {
        "MainPID": str(pid),
        "ActiveState": "active",
        "Type": "exec",
        "MemoryAccounting": "yes",
        "MemoryMax": str(worker.common.MEMORY_MAX),
        "MemorySwapMax": "0",
        "OOMPolicy": "kill",
        "KillMode": "control-group",
        "RuntimeMaxUSec": "1min",
        "TimeoutStopUSec": "3s",
    }
    hello = {
        "schema": "decision3-cpu-hello-v1",
        "pid": pid,
        "pgid": pid,
        "startticks": 7,
        "boot_id": "boot",
        "unit": unit,
        "invocation": invocation,
        "plan_sha256": digest,
        "cgroup": group,
    }
    monkeypatch.setattr(watcher.shared, "_unit_identity", lambda *_: (invocation, group))
    monkeypatch.setattr(watcher.common.systemd, "boot_id", lambda: "boot")
    monkeypatch.setattr(watcher.common.systemd, "process_identity", lambda _: {"startticks": 7})
    monkeypatch.setattr(watcher.common.systemd, "cgroup_membership", lambda _: group)
    monkeypatch.setattr(watcher.os, "getpgid", lambda _: pid)
    calls = []
    monkeypatch.setattr(worker, "_cpu_limits", lambda *args: calls.append(args))
    assert watcher._bind(properties, hello, plan, digest) == (invocation, group)
    assert calls[0][:2] == (group, pid)
    assert calls[0][2]["threads"] == 10 and calls[0][2]["cpu_list"] == "5-9,15-19"


def test_v2_worker_binding_checks_owned_unit_and_profile(monkeypatch):
    profile = worker._profile({"schema": worker.SCHEMA_V2, "cpu_profile": "cpu10"})
    unit = "synthetic.service"
    group = "/user.slice/app.slice/" + unit
    monkeypatch.setattr(worker.common.systemd, "cgroup_membership", lambda _: group)
    calls = []
    monkeypatch.setattr(worker, "_cpu_limits", lambda *args: calls.append(args))
    assert worker._own_profile_cgroup(unit, profile) == group
    assert calls == [(group, os.getpid(), profile)]
    with pytest.raises(ValueError, match="selected unit"):
        worker._own_profile_cgroup("other.service", profile)


def test_v1_v2_release_inventories_and_worker_launch_do_not_overlap():
    v1, v2 = worker._source_set(), worker._source_set_v2()
    for name in ("worker", "watcher", "candidate"):
        old = str(worker._release_path(name, worker.SCHEMA))
        new = str(worker._release_path(name, worker.SCHEMA_V2))
        assert old != new and old.endswith("_release.py") and new.endswith("_v2_release.py")
        assert old in v1 and old not in v2
        assert new in v2 and new not in v1
    assert len(v1) == len(v2)
    base = {"unit": "synthetic.service", "python": worker.common.GPU_PYTHON, "wall_seconds": 600}
    old_args = watcher._unit_args(base, "/tmp/plan", "a" * 64)
    new_args = watcher._unit_args(
        {**base, "schema": worker.SCHEMA_V2, "cpu_profile": "cpu10"}, "/tmp/plan", "a" * 64
    )
    assert str(worker._release_path("worker", worker.SCHEMA)) in old_args
    assert str(worker._release_path("worker", worker.SCHEMA_V2)) in new_args
    assert str(worker._release_path("worker", worker.SCHEMA)) not in new_args


@pytest.mark.parametrize(
    "stage", ["process_group", "cgroup", "invocation", "sources", "interpreter", "hello"]
)
def test_v3_prehello_failure_records_only_bounded_categories(tmp_path, monkeypatch, stage):
    output = tmp_path / "worker"
    output.mkdir(mode=0o700)
    plan = {
        "schema": worker.SCHEMA_V3,
        "cpu_profile": "cpu10",
        "unit": "synthetic.service",
        "output_dir": str(output),
        "source_pins": {},
    }
    sha = "a" * 64
    monkeypatch.setattr(worker, "SOURCE_ONLY", False)
    monkeypatch.setattr(worker, "_plan", lambda *_: plan)
    monkeypatch.setattr(worker.os, "getpid", lambda: 123)
    monkeypatch.setattr(worker.os, "getpgrp", lambda: 122 if stage == "process_group" else 123)

    def fail(*_args):
        raise ValueError("private/path secret and traceback must not be recorded")

    monkeypatch.setattr(worker.os, "setsid", fail if stage == "process_group" else lambda: None)
    monkeypatch.setattr(
        worker, "_own_profile_cgroup", fail if stage == "cgroup" else lambda *_: "/synthetic"
    )
    monkeypatch.setenv("INVOCATION_ID", "bad" if stage == "invocation" else "b" * 32)
    monkeypatch.setattr(
        worker.common, "_verify_source_pins", fail if stage == "sources" else lambda *_: None
    )
    monkeypatch.setattr(
        worker.common, "_validate_interpreter", fail if stage == "interpreter" else lambda *_: None
    )
    monkeypatch.setattr(
        worker.common.systemd,
        "process_identity",
        fail if stage == "hello" else lambda *_: {"startticks": 1},
    )
    with pytest.raises(ValueError):
        worker.run("unused", sha)
    raw = (output / "prehello-failure.json").read_bytes()
    assert b"secret" not in raw and b"private/path" not in raw and b"traceback" not in raw
    assert json.loads(raw) == {
        "schema": "decision3-cpu-prehello-failure-v1",
        "plan_sha256": sha,
        "stage": stage,
        "category": "validation",
        "cpu_check": None,
    }
    assert watcher._prehello_diagnostic(output, sha) == {
        "source": "unbound_worker_report",
        "stage": stage,
        "category": "validation",
        "cpu_check": None,
    }


def test_v3_missing_cpuset_remains_fatal_and_records_fixed_check(tmp_path, monkeypatch):
    root = tmp_path / "cgroup"
    leaf = root / "app.slice/synthetic.service"
    leaf.mkdir(parents=True)
    monkeypatch.setattr(worker, "CGROUP_ROOT", root)
    profile = worker._profile({"schema": worker.SCHEMA_V3, "cpu_profile": "cpu10"})
    monkeypatch.setattr(worker.os, "sched_getaffinity", lambda _: set(profile["cpus"]))
    (leaf / "cpu.max").write_text("1000000 100000")
    (leaf / "memory.max").write_text(str(worker.common.MEMORY_MAX))
    (leaf / "memory.swap.max").write_text("0")
    with pytest.raises(worker._CpuLimitError) as error:
        worker._cpu_limits("/app.slice/synthetic.service", 123, profile)
    assert error.value.check == "cpuset"
    assert "No such file" not in str(error.value)


def test_v3_worker_records_cpu_check_without_exception_text(tmp_path, monkeypatch):
    output = tmp_path / "worker"
    output.mkdir(mode=0o700)
    plan = {
        "schema": worker.SCHEMA_V3,
        "cpu_profile": "cpu10",
        "unit": "synthetic.service",
        "output_dir": str(output),
        "source_pins": {},
    }
    monkeypatch.setattr(worker, "SOURCE_ONLY", False)
    monkeypatch.setattr(worker, "_plan", lambda *_: plan)
    monkeypatch.setattr(worker.os, "getpid", lambda: 123)
    monkeypatch.setattr(worker.os, "getpgrp", lambda: 123)

    def fail(*_args):
        raise worker._CpuLimitError("cpuset")

    monkeypatch.setattr(worker, "_own_profile_cgroup", fail)
    with pytest.raises(worker._CpuLimitError):
        worker.run("unused", "a" * 64)
    assert watcher._prehello_diagnostic(output, "a" * 64) == {
        "source": "unbound_worker_report",
        "stage": "cgroup",
        "category": "validation",
        "cpu_check": "cpuset",
    }
    assert (output / "prehello-failure.json").stat().st_mode & 0o077 == 0


def test_v3_unbound_failure_is_diagnostic_not_closure_proof(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    output = run / "worker"
    output.mkdir(mode=0o700)
    report = run / "watcher"
    plan = {
        "schema": worker.SCHEMA_V3,
        "cpu_profile": "cpu10",
        "output_dir": str(output),
        "unit": "jev-cpu-" + "a" * 32 + ".service",
        "source_pins": {},
        "memory_floor_bytes": worker.common.MEMORY_FLOOR,
        "wall_seconds": 60,
        "python": worker.common.GPU_PYTHON,
    }
    sha = "e" * 64
    monkeypatch.setattr(watcher, "SOURCE_ONLY", False)
    monkeypatch.setattr(worker, "_plan", lambda *_: plan)
    monkeypatch.setattr(watcher.common, "_validate_interpreter", lambda *_: None)
    monkeypatch.setattr(watcher.common, "_verify_source_pins", lambda *_: None)
    monkeypatch.setattr(watcher, "_verify_release", lambda *_: None)
    monkeypatch.setattr(watcher.common, "_observe", lambda *_: {})
    states = iter(
        [
            {"LoadState": "not-found", "ActiveState": "inactive", "MainPID": "0"},
            {"LoadState": "loaded", "InvocationID": "d" * 32},
        ]
    )
    monkeypatch.setattr(watcher.common.systemd, "service_show", lambda *_a, **_kw: next(states))
    monkeypatch.setattr(watcher.shared, "_unit_identity", lambda *_: ("d" * 32, "/untrusted"))

    def fail(*_args):
        raise ValueError("synthetic failed")

    monkeypatch.setattr(watcher.common, "_terminal_metadata", fail)
    monkeypatch.setattr(watcher, "_recover_identity", fail)
    stopped = []
    monkeypatch.setattr(watcher.common, "_stop_same", lambda *args: stopped.append(args))

    def launch(*_args, **_kwargs):
        watcher.common._write_new(
            output / "prehello-failure.json",
            watcher.child._encode(
                {
                    "schema": "decision3-cpu-prehello-failure-v1",
                    "plan_sha256": sha,
                    "stage": "cgroup",
                    "category": "validation",
                    "cpu_check": "cpuset",
                }
            ),
        )

    monkeypatch.setattr(watcher.subprocess, "run", launch)
    with pytest.raises(ValueError, match="synthetic failed"):
        watcher.supervise(str(run / "plan.json"), sha, str(report))
    closure = json.loads((report / "closure.json").read_bytes())
    assert closure["status"] == "cleanup_unconfirmed" and not stopped
    assert closure["cgroup"] == closure["process_group"] == "unconfirmed"
    assert closure["prehello_diagnostic"] == {
        "source": "unbound_worker_report",
        "stage": "cgroup",
        "category": "validation",
        "cpu_check": "cpuset",
    }


def test_v3_prehello_rejects_untrusted_content_and_wrong_plan(tmp_path):
    path = tmp_path / "prehello-failure.json"
    value = {
        "schema": "decision3-cpu-prehello-failure-v1",
        "plan_sha256": "a" * 64,
        "stage": "cgroup",
        "category": "validation",
        "cpu_check": "cpuset",
    }
    path.write_text(json.dumps(value))
    assert watcher._prehello_diagnostic(tmp_path, "a" * 64)["cpu_check"] == "cpuset"
    for bad in (
        {**value, "cpu_check": "private/path"},
        {**value, "message": "secret"},
        {**value, "plan_sha256": "b" * 64},
    ):
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            watcher._prehello_diagnostic(tmp_path, "a" * 64)


def test_v3_release_inventory_is_disjoint_from_older_releases():
    v1, v2, v3 = worker._source_set(), worker._source_set_v2(), worker._source_set_v3()
    assert len(v1) == len(v2) == len(v3)
    for name in ("worker", "watcher", "candidate"):
        old = str(worker._release_path(name, worker.SCHEMA))
        previous = str(worker._release_path(name, worker.SCHEMA_V2))
        new = str(worker._release_path(name, worker.SCHEMA_V3))
        assert new.endswith("_v3_release.py") and new in v3 and new not in v1 and new not in v2
        assert old not in v3 and previous not in v3
    args = watcher._unit_args(
        {
            "schema": worker.SCHEMA_V3,
            "cpu_profile": "cpu10",
            "unit": "synthetic.service",
            "python": worker.common.GPU_PYTHON,
            "wall_seconds": 600,
        },
        "/tmp/plan",
        "a" * 64,
    )
    assert "--property=AllowedCPUs=5-9,15-19" in args
    assert "--property=CPUQuota=1000%" in args
    assert str(worker._release_path("worker", worker.SCHEMA_V3)) in args


def test_v3_plan_uses_only_v3_source_inventory(tmp_path, monkeypatch):
    plan, _path, save = make_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(worker, "_source_set_v3", lambda: {"/approved/v3"})
    plan.update(
        schema=worker.SCHEMA_V3, cpu_profile="cpu10", source_pins={"/approved/v3": "b" * 64}
    )
    assert worker._plan(*save(plan)) == plan
    plan["source_pins"] = {"/approved/source": "a" * 64}
    with pytest.raises(ValueError, match="allowlisted"):
        worker._plan(*save(plan))
