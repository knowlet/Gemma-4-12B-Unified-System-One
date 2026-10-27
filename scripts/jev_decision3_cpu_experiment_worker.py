"""Held Decision3 CPU timing worker for a private, synthetic four-head request.

The external watcher owns cgroup/host admission and hard termination. Source,
model, input and memory evidence are limited to the checks explicitly reported
by that watcher; this worker cannot certify isolation or a benchmark score.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import resource
import sys
import time
import types
from pathlib import Path

import jev_decision3_candidate_browser_child as child
import jev_decision3_candidate_http_owner as common

SOURCE_ONLY = True
SCHEMA = "decision3-cpu-padding-experiment-plan-v1"
SCHEMA_V2 = "decision3-cpu-padding-experiment-plan-v2"
SCHEMA_V3 = "decision3-cpu-padding-experiment-plan-v3"
CGROUP_ROOT = Path("/sys/fs/cgroup")
ROOT = Path(__file__).absolute().parent.parent
SCRIPTS = ROOT / "scripts"
CACHE = ROOT / "artifacts/jev-local-hf/hub"
SNAPSHOT = CACHE / "models--vllm-sr--d3-edge/snapshots/cab71e6ad3461b8086d51ddb572c656babd63b0b"
STAGED = (
    ROOT
    / "artifacts/jev-runtime/decision3-link-navigation-diagnostic-20261027-01/staged-modules/transformers_modules/cab71e6ad3461b8086d51ddb572c656babd63b0b/5e85afd45ec2b404"
)
OVERLAY = ROOT / "artifacts/jev-runtime/overlay"
EXPERIMENTS = ROOT / "artifacts/jev-runtime/decision3-cpu-padding-experiment"
TORCH = Path("/home/albert_hsieh/ComfyUI/.venv/lib/python3.13/site-packages/torch/__init__.py")
MANIFEST = "3bae8989a6feb36a66bf4d8c44112dc63199d0dc69f564dc895fe368b1e08e59"
_STAGES = ("overlay", "model_load", "input_prepare", "inference", "result_write")
_HELD = {
    "worker": "jev_decision3_cpu_experiment_worker.py",
    "watcher": "jev_decision3_cpu_experiment_watcher.py",
    "candidate": "jev_decision3_cpu_padding_candidate.py",
}


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _profile(plan):
    if plan.get("schema", SCHEMA) == SCHEMA:
        return {"cpus": frozenset(range(15, 20)), "cpu_list": "15-19", "threads": 5}
    _need(
        plan.get("schema") in {SCHEMA_V2, SCHEMA_V3} and type(plan.get("cpu_profile")) is str,
        "explicit CPU profile required",
    )
    profiles = {
        "cpu5": {"cpus": frozenset(range(15, 20)), "cpu_list": "15-19", "threads": 5},
        "cpu10": {
            "cpus": frozenset((*range(5, 10), *range(15, 20))),
            "cpu_list": "5-9,15-19",
            "threads": 10,
        },
    }
    _need(plan["cpu_profile"] in profiles, "unknown CPU profile")
    return profiles[plan["cpu_profile"]]


def _parse_cpu_list(value):
    _need(type(value) is str and len(value) <= 4096 and value, "bounded effective cpuset required")
    result = set()
    for item in value.split(","):
        match = re.fullmatch(r"([0-9]+)(?:-([0-9]+))?", item)
        _need(match is not None, "invalid effective cpuset")
        start = int(match.group(1))
        end = int(match.group(2) or start)
        _need(0 <= start <= end < 4096, "bounded effective cpuset required")
        result.update(range(start, end + 1))
    return result


class _CpuLimitError(ValueError):
    def __init__(self, check):
        super().__init__("actual worker CPU, memory and swap limits required")
        self.check = check


def _cpu_limits(group, pid, profile):
    _need(
        type(group) is str and group.startswith("/") and ".." not in Path(group).parts,
        "owned cgroup path required",
    )
    path = CGROUP_ROOT / group.lstrip("/")
    _need(path.resolve().is_relative_to(CGROUP_ROOT.resolve()), "cgroup escaped root")

    def check(name, action):
        try:
            valid = action()
        except (OSError, ValueError, TypeError):
            raise _CpuLimitError(name) from None
        if not valid:
            raise _CpuLimitError(name)

    def quota_valid():
        quota, period = map(int, (path / "cpu.max").read_text().split())
        return period > 0 and quota == period * profile["threads"]

    check("affinity", lambda: os.sched_getaffinity(pid) == profile["cpus"])
    check(
        "cpuset",
        lambda: (
            _parse_cpu_list((path / "cpuset.cpus.effective").read_text().strip()) == profile["cpus"]
        ),
    )
    check("memory", lambda: (path / "memory.max").read_text().strip() == str(common.MEMORY_MAX))
    check("swap", lambda: (path / "memory.swap.max").read_text().strip() == "0")
    check("quota", quota_valid)


def _own_profile_cgroup(unit, profile):
    group = common.systemd.cgroup_membership(os.getpid())
    _need(group.endswith("/app.slice/" + unit), "selected unit cgroup required")
    _cpu_limits(group, os.getpid(), profile)
    return group


def _source_set():
    names = [SCRIPTS / name for name in _HELD.values()]
    names += [SCRIPTS / name.replace(".py", "_release.py") for name in _HELD.values()]
    names += [
        SCRIPTS / "jev_decision3_candidate_http_owner.py",
        SCRIPTS / "jev_decision3_candidate_browser_child.py",
        SCRIPTS / "jev_decision3_jevbench_heaven_supervisor.py",
        SCRIPTS / "jev_decision3_jevbench_heaven_owner.py",
        SCRIPTS / "jev_stage_memory_profile_v2.py",
        STAGED / "d3_runtime.py",
        STAGED / "d3_format.py",
        STAGED / "modeling_d3.py",
        SNAPSHOT / "d3_fast.py",
        OVERLAY / "transformers/__init__.py",
        OVERLAY / "transformers/models/qwen3_5/modeling_qwen3_5.py",
        TORCH,
        Path(common.GPU_PYTHON_REAL),
    ]
    return {str(path) for path in names}


def _release_path(name, schema):
    _need(name in _HELD and schema in {SCHEMA, SCHEMA_V2, SCHEMA_V3}, "known release required")
    suffix = {SCHEMA: "_release.py", SCHEMA_V2: "_v2_release.py", SCHEMA_V3: "_v3_release.py"}[
        schema
    ]
    return SCRIPTS / (_HELD[name].removesuffix(".py") + suffix)


def _source_set_v2():
    names = _source_set()
    for name in _HELD:
        names.remove(str(_release_path(name, SCHEMA)))
        names.add(str(_release_path(name, SCHEMA_V2)))
    return names


def _source_set_v3():
    names = _source_set()
    for name in _HELD:
        names.remove(str(_release_path(name, SCHEMA)))
        names.add(str(_release_path(name, SCHEMA_V3)))
    return names


def _plan(path, expected):
    _need(type(expected) is str and re.fullmatch(r"[0-9a-f]{64}", expected), "plan hash required")
    _need(type(path) is str, "explicit plan path required")
    selected = Path(path)
    _need(
        selected.is_absolute()
        and selected.name == "plan.json"
        and selected.parent.parent == EXPERIMENTS
        and re.fullmatch(r"run-[0-9a-f]{32}", selected.parent.name)
        and selected.resolve() == selected,
        "dedicated fresh experiment plan required",
    )
    raw = common._read_reviewed_source(selected, expected)
    _need(len(raw) <= 1024 * 1024, "bounded plan required")
    plan = common._decode(raw)
    _need(
        type(plan) is dict
        and set(plan)
        == {
            "schema",
            "unit",
            "output_dir",
            "python",
            "source_pins",
            "wall_seconds",
            "memory_floor_bytes",
            "rounds",
        }
        | ({"cpu_profile"} if plan.get("schema") in {SCHEMA_V2, SCHEMA_V3} else set())
        and plan["schema"] in {SCHEMA, SCHEMA_V2, SCHEMA_V3},
        "exact CPU experiment plan required",
    )
    _profile(plan)
    _need(
        type(plan["unit"]) is str and re.fullmatch(r"jev-cpu-[0-9a-f]{32}\.service", plan["unit"]),
        "fresh unit required",
    )
    _need(
        plan["unit"] == "jev-cpu-" + selected.parent.name.removeprefix("run-") + ".service"
        and plan["output_dir"] == str(selected.parent / "worker")
        and plan["python"] == common.GPU_PYTHON,
        "dedicated unit, output and interpreter required",
    )
    _need(type(plan["rounds"]) is int and plan["rounds"] in (4, 8), "bounded rounds required")
    _need(
        type(plan["wall_seconds"]) is int and 30 <= plan["wall_seconds"] <= 3600,
        "bounded hard wall required",
    )
    _need(
        type(plan["memory_floor_bytes"]) is int
        and plan["memory_floor_bytes"] >= common.MEMORY_FLOOR,
        "memory floor required",
    )
    pins = plan["source_pins"]
    _need(
        type(pins) is dict
        and set(pins)
        == {SCHEMA: _source_set, SCHEMA_V2: _source_set_v2, SCHEMA_V3: _source_set_v3}[
            plan["schema"]
        ]()
        and all(type(v) is str and re.fullmatch(r"[0-9a-f]{64}", v) for v in pins.values()),
        "exact allowlisted source inventory required",
    )
    return plan


def _verify_model_manifest(pins):
    """Reject unexpected model paths before publisher full verification reads them."""
    _need(
        SNAPSHOT.is_dir() and SNAPSHOT.resolve().is_relative_to(CACHE.resolve()),
        "selected local model required",
    )
    raw = common._read_reviewed_source(SNAPSHOT / "MODEL_MANIFEST.json", MANIFEST)
    manifest = common._decode(raw)
    files = manifest.get("files_sha256")
    _need(type(files) is dict and 0 < len(files) <= 128, "bounded model inventory required")
    for name, digest in files.items():
        _need(
            type(name) is str
            and name
            and not name.startswith("/")
            and ".." not in Path(name).parts
            and "\\" not in name
            and not any(x in name.lower() for x in ("heldout", "case", "root-done-feedback"))
            and type(digest) is str
            and re.fullmatch(r"[0-9a-f]{64}", digest),
            "unauthorized model inventory path",
        )
        path = SNAPSHOT / name
        _need(path.resolve().is_relative_to(CACHE.resolve()), "model path escaped selected cache")
    for name in ("d3_format.py", "d3_runtime.py", "modeling_d3.py", "d3_fast.py"):
        path = (SNAPSHOT if name == "d3_fast.py" else STAGED) / name
        _need(
            name in files and pins.get(str(path)) == files[name],
            "publisher pin differs from manifest",
        )
    return hashlib.sha256(raw).hexdigest()


def _load_publisher(pins):
    """Load the fixed local staged runtime; no benchmark case or scorer imports."""
    package_name = "_cpu_experiment_publisher"
    _need(package_name not in sys.modules, "fresh publisher imports required")
    package = types.ModuleType(package_name)
    package.__path__ = [str(STAGED)]
    sys.modules[package_name] = package
    registered = {package_name: package}
    try:
        for name in ("d3_format", "d3_fast", "d3_runtime"):
            source = (SNAPSHOT if name == "d3_fast" else STAGED) / (name + ".py")
            raw = common._read_reviewed_source(source, pins[str(source)])
            module_name = package_name + "." + name
            _need(module_name not in sys.modules, "fresh publisher module required")
            spec = importlib.util.spec_from_file_location(module_name, str(source))
            _need(spec is not None and spec.loader is not None, "publisher source loader required")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            registered[module_name] = module
            exec(compile(raw, str(source), "exec", dont_inherit=True), module.__dict__)
            _need(
                sys.modules.get(module_name) is module and sys.modules.get(package_name) is package,
                "publisher module identity changed",
            )
            setattr(package, name, module)
        return registered[package_name + ".d3_runtime"]
    except BaseException:
        for name, module in reversed(tuple(registered.items())):
            if sys.modules.get(name) is module:
                del sys.modules[name]
        raise


def _sample(runtime):
    # Synthetic text generated here, with varied sentences and no benchmark data.
    state = {"subject": "fictional package routing exercise", "note": "example only"}
    corpus = " ".join(
        f"Parcel {n} travels through fictional route {n % 37}; review condition {n % 53} "
        f"and select a sensible outcome for station {n % 71}."
        for n in range(1600)
    )
    encoded = runtime.tokenizer.encode(corpus, add_special_tokens=False)
    _need(len(encoded) >= 5000, "synthetic corpus is too short")
    targets = (3117, 4196, 4579, 3126)
    sizes = list(targets)
    for _ in range(12):
        questions = {
            f"synthetic_{index}": {
                "type": "choice",
                "instructions": runtime.tokenizer.decode(
                    encoded[: sizes[index]], skip_special_tokens=True
                ),
                "criteria": {"alpha": "first", "beta": "second", "gamma": "third"},
            }
            for index in range(4)
        }
        prepared = runtime.prepare(state, questions)
        _need(
            not prepared.errors and list(prepared.keys) == list(questions),
            "synthetic preparation failed",
        )
        lengths = [len(prepared.sequences[key]) for key in questions]
        if all(abs(length - target) <= 32 for length, target in zip(lengths, targets, strict=True)):
            return state, questions, lengths
        sizes = [
            max(1, min(len(encoded), size + target - length))
            for size, target, length in zip(sizes, targets, lengths, strict=True)
        ]
    raise ValueError("synthetic token targets not achieved")


def _response(response, order, expected_tokens, model_name):
    _need(
        type(response) is dict
        and response.get("model") == model_name
        and type(response.get("answers")) is dict
        and list(response["answers"]) == order,
        "response order differs",
    )
    usage = response.get("usage")
    _need(
        type(usage) is dict
        and type(usage.get("input_tokens")) is int
        and usage["input_tokens"] == expected_tokens
        and type(usage.get("output_tokens")) is int
        and usage["output_tokens"] == 0,
        "native usage differs",
    )
    for answer in response["answers"].values():
        _need(
            type(answer) is dict
            and answer.get("type") == "choice"
            and type(answer.get("probabilities")) is dict
            and list(answer["probabilities"]) == ["alpha", "beta", "gamma"],
            "complete ordered choice probabilities required",
        )
        values = list(answer["probabilities"].values())
        _need(
            all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in values)
            and math.isclose(sum(values), 1.0, rel_tol=1e-6, abs_tol=1e-6)
            and answer.get("choice") == list(answer["probabilities"])[values.index(max(values))]
            and type(answer.get("confidence")) in (int, float)
            and math.isfinite(answer["confidence"])
            and 0 <= answer["confidence"] <= 1,
            "valid ordered probabilities and confidence required",
        )


def _experiment(runtime, candidate, rounds, stage=lambda _: None):
    stage("input_prepare")
    state, questions, lengths = _sample(runtime)
    stage("inference")
    order = list(questions)
    tokens = sum(lengths)
    modes = ("native", "whole", "ordered_pairs", "length_pairs")
    samples = []
    for cycle in range(-1, rounds):
        scheduled = modes if cycle < 0 else modes[cycle % 4 :] + modes[: cycle % 4]
        for mode in scheduled:
            start = time.perf_counter_ns()
            if mode == "native":
                response = runtime.system_one(state=state, questions=questions)
                raw_probabilities = None
                metrics = None
            else:
                prepared = runtime.prepare(state, questions)
                _need(
                    not prepared.errors
                    and [len(prepared.sequences[key]) for key in order] == lengths,
                    "prepared lengths changed",
                )
                snapshot = candidate._snapshot(runtime, prepared)
                value = candidate.evaluate(runtime, prepared, mode=mode, enabled=True)
                response = value["response"]
                raw_probabilities = value["native_probabilities"]
                metrics = {
                    key: value[key]
                    for key in (
                        "native_run_calls",
                        "forward_groups",
                        "unpadded_tokens",
                        "padded_slots",
                    )
                }
                _need(candidate._snapshot(runtime, prepared) == snapshot, "prepared input changed")
            elapsed = time.perf_counter_ns() - start
            _response(response, order, tokens, runtime.model_name)
            _need(
                raw_probabilities is None
                if mode == "native"
                else (
                    type(raw_probabilities) is dict
                    and list(raw_probabilities) == order
                    and all(
                        type(raw_probabilities[key]) is list
                        and len(raw_probabilities[key]) == 3
                        and all(
                            type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1
                            for x in raw_probabilities[key]
                        )
                        and math.isclose(
                            sum(raw_probabilities[key]), 1.0, rel_tol=0.0, abs_tol=1e-5
                        )
                        for key in order
                    )
                ),
                "complete native probability rows required",
            )
            samples.append(
                {
                    "phase": "warmup" if cycle < 0 else "measured",
                    "cycle": cycle,
                    "mode": mode,
                    "elapsed_ns": elapsed,
                    "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "metrics": metrics,
                    "native_probabilities": raw_probabilities,
                    "response": response,
                }
            )
    comparisons = []
    for cycle in range(rounds):
        current = {
            row["mode"]: row
            for row in samples
            if row["phase"] == "measured" and row["cycle"] == cycle
        }
        _need(set(current) == set(modes), "complete cycle required")
        reference = current["native"]["response"]
        for mode in modes[1:]:
            actual = current[mode]["response"]
            differences = [
                abs(
                    actual["answers"][key]["probabilities"][option]
                    - reference["answers"][key]["probabilities"][option]
                )
                for key in order
                for option in ("alpha", "beta", "gamma")
            ]
            comparisons.append(
                {
                    "cycle": cycle,
                    "mode": mode,
                    "max_probability_abs_diff": max(differences),
                    "max_native_probability_abs_diff": None,  # system_one does not expose raw rows
                    "max_confidence_abs_diff": max(
                        abs(
                            actual["answers"][key]["confidence"]
                            - reference["answers"][key]["confidence"]
                        )
                        for key in order
                    ),
                    "same_choices": all(
                        actual["answers"][key]["choice"] == reference["answers"][key]["choice"]
                        for key in order
                    ),
                    "same_usage": actual["usage"] == reference["usage"],
                }
            )
    return {
        "schema": "decision3-cpu-padding-experiment-result-v1",
        "scope": "synthetic-only; no task success or benchmark score; peak_rss_kib is cumulative process lifetime peak",
        "state": state,
        "questions": questions,
        "input_tokens": tokens,
        "head_token_lengths": dict(zip(order, lengths, strict=True)),
        "timing_scope": "complete request: prepare, inference and response; candidate also validates snapshots",
        "native_raw_probabilities": "unavailable from publisher system_one",
        "samples": samples,
        "comparisons": comparisons,
    }


class _StageRecorder:
    def __init__(self, output, invocation, plan_sha):
        self.output = output
        self.invocation = invocation
        self.plan_sha = plan_sha
        self.current = None
        self.next_index = 0

    def __call__(self, name):
        _need(
            self.next_index < len(_STAGES) and name == _STAGES[self.next_index],
            "stage order required",
        )
        value = {
            "schema": "decision3-cpu-stage-v1",
            "invocation": self.invocation,
            "plan_sha256": self.plan_sha,
            "stage": name,
        }
        common._write_new(self.output / f"stage-{self.next_index:02d}.json", child._encode(value))
        self.current = name
        self.next_index += 1

    def failure(self, error):
        if self.current is None:
            return
        if isinstance(error, MemoryError):
            category = "memory"
        elif isinstance(error, ImportError):
            category = "import"
        elif isinstance(error, (ValueError, TypeError)):
            category = "validation"
        elif isinstance(error, TimeoutError):
            category = "timeout"
        else:
            category = "other"
        common._write_new(
            self.output / "failure.json",
            child._encode(
                {
                    "schema": "decision3-cpu-failure-v1",
                    "invocation": self.invocation,
                    "plan_sha256": self.plan_sha,
                    "stage": self.current,
                    "category": category,
                }
            ),
        )


def run(plan_path, plan_sha):
    _need(SOURCE_ONLY is False, "reviewed CPU worker release required")
    plan = _plan(plan_path, plan_sha)
    output = Path(plan["output_dir"])
    _need(
        output.is_dir()
        and output.resolve() == output
        and output.stat().st_mode & 0o077 == 0
        and not any(output.iterdir()),
        "fresh private worker output required",
    )
    stage = "process_group"
    try:
        if os.getpgrp() != os.getpid():
            os.setsid()
        _need(os.getpgrp() == os.getpid(), "isolated worker process group required")
        stage = "cgroup"
        group = (
            common._own_cgroup(plan["unit"])
            if plan.get("schema", SCHEMA) == SCHEMA
            else _own_profile_cgroup(plan["unit"], _profile(plan))
        )
        stage = "invocation"
        invocation = os.environ.get("INVOCATION_ID", "")
        _need(re.fullmatch(r"[0-9a-f]{32}", invocation), "unit invocation required")
        stage = "sources"
        common._verify_source_pins(plan["source_pins"])
        stage = "interpreter"
        common._validate_interpreter(plan)
        stage = "hello"
        hello = {
            "schema": "decision3-cpu-hello-v1",
            "pid": os.getpid(),
            "startticks": common.systemd.process_identity(os.getpid())["startticks"],
            "boot_id": common.systemd.boot_id(),
            "unit": plan["unit"],
            "invocation": invocation,
            "plan_sha256": plan_sha,
            "pgid": os.getpgrp(),
            "cgroup": group,
        }
        common._write_new(output / "hello.json", child._encode(hello))
    except Exception as error:
        if plan.get("schema") == SCHEMA_V3:
            check = error.check if isinstance(error, _CpuLimitError) else None
            category = (
                "memory"
                if isinstance(error, MemoryError)
                else "import"
                if isinstance(error, ImportError)
                else "timeout"
                if isinstance(error, TimeoutError)
                else "validation"
                if isinstance(error, (ValueError, TypeError))
                else "other"
            )
            try:
                common._write_new(
                    output / "prehello-failure.json",
                    child._encode(
                        {
                            "schema": "decision3-cpu-prehello-failure-v1",
                            "plan_sha256": plan_sha,
                            "stage": stage,
                            "category": category,
                            "cpu_check": check,
                        }
                    ),
                )
            except Exception:
                error.add_note("private prehello failure record unavailable")
        raise
    started = time.monotonic()
    while not (output / "go.json").exists():
        _need(time.monotonic() - started < 20, "watcher handshake timed out")
        time.sleep(0.1)
    child._control(
        output / "go.json", schema="decision3-cpu-go-v1", invocation=invocation, plan_sha=plan_sha
    )
    stage = _StageRecorder(output, invocation, plan_sha)
    try:
        _run_after_go(plan, output, invocation, plan_sha, stage)
    except Exception as error:
        try:
            stage.failure(error)
        except Exception:
            error.add_note("private failure record unavailable")
        raise


def _run_after_go(plan, output, invocation, plan_sha, stage):
    stage("overlay")
    threads = _profile(plan)["threads"]
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "OPENBLAS_NUM_THREADS": str(threads),
        }
    )
    common._verify_source_pins(plan["source_pins"])
    torch = common._bind_overlay(ROOT, plan["source_pins"])
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    stage("model_load")
    manifest_sha = _verify_model_manifest(plan["source_pins"])
    publisher = _load_publisher(plan["source_pins"])
    candidate = common._load_reviewed_release(
        str(_release_path("candidate", plan.get("schema", SCHEMA))),
        "jev_decision3_cpu_padding_candidate.py",
        plan["source_pins"],
        "_cpu_padding_candidate_release",
    )
    runtime = publisher.D3.from_pretrained(
        str(SNAPSHOT),
        device="cpu",
        batch_size=8,
        verify="full",
        local_files_only=True,
        permutation_average=False,
    )
    _need(str(runtime.device) == "cpu" and runtime.fast is None, "plain CPU runtime required")
    result = _experiment(runtime, candidate, plan["rounds"], stage)
    stage("result_write")
    if plan.get("schema") in {SCHEMA_V2, SCHEMA_V3}:
        result["cpu_profile"] = plan["cpu_profile"]
    result["publisher_manifest_sha256"] = manifest_sha
    result["model_verification"] = (
        "publisher full verify at load; execution-time weights and external dependencies unverified"
    )
    raw = json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    _need(len(raw) <= 4 * 1024 * 1024, "bounded result required")
    common._write_new(output / "result.json", raw)
    common._verify_source_pins(plan["source_pins"])
    terminal = {
        "schema": "decision3-cpu-terminal-v1",
        "status": "completed_pending_external_audit",
        "invocation": invocation,
        "plan_sha256": plan_sha,
        "result_sha256": hashlib.sha256(raw).hexdigest(),
    }
    common._write_new(output / "terminal.json", child._encode(terminal))


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    run(args.plan, args.plan_sha256)


if __name__ == "__main__":
    main()
