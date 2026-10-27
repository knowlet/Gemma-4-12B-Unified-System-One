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


def test_worker_runs_complete_fake_model_and_writes_private_result(
    tmp_path, monkeypatch, experiment
):
    runtime, module = experiment
    output = tmp_path / "worker"
    output.mkdir(mode=0o700)
    invocation = "b" * 32
    plan = {"output_dir": str(output), "unit": "synthetic.service", "source_pins": {}, "rounds": 4}
    monkeypatch.setattr(worker, "SOURCE_ONLY", False)
    monkeypatch.setattr(worker, "_plan", lambda *_: plan)
    monkeypatch.setattr(worker.os, "getpgrp", os.getpid)
    monkeypatch.setattr(worker.common, "_own_cgroup", lambda _: "/synthetic")
    monkeypatch.setattr(worker.common, "_verify_source_pins", lambda _: None)
    monkeypatch.setattr(worker.common, "_validate_interpreter", lambda _: None)
    monkeypatch.setattr(worker.common.systemd, "boot_id", lambda: "boot")
    monkeypatch.setattr(worker.common.systemd, "process_identity", lambda _: {"startticks": 7})
    monkeypatch.setenv("INVOCATION_ID", invocation)
    torch = SimpleNamespace(set_num_threads=lambda _: None, set_num_interop_threads=lambda _: None)
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
    monkeypatch.setattr(worker.common, "_load_reviewed_release", lambda *_: module)
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
    result = json.loads((output / "result.json").read_bytes())
    terminal = json.loads((output / "terminal.json").read_bytes())
    assert len(result["samples"]) == 20 and len(result["comparisons"]) == 12
    assert (
        terminal["result_sha256"]
        == hashlib.sha256((output / "result.json").read_bytes()).hexdigest()
    )
    assert terminal["status"] == "completed_pending_external_audit"


def test_watcher_supervises_fake_unit_to_verified_closure(tmp_path, monkeypatch):
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
    closure = watcher.supervise(str(plan_path), digest, str(report))
    assert closure["status"] == "completed_pending_external_audit"
    assert closure["cgroup"] == "absent" and closure["process_group"] == "absent"
    assert (output / "go.json").is_file() and (report / "closure.json").is_file()
    events = [json.loads(line) for line in (report / "events.jsonl").read_bytes().splitlines()]
    assert [event["event"] for event in events] == ["prelaunch", "host", "binding_verified", "host"]
    assert closure["machine_exclusive"] is False


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
