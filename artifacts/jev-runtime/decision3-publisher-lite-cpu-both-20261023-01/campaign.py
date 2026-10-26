#!/usr/bin/env python3
"""Dry-default, one-constructor CPU campaign for the two unchanged JevBench suites."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SMOKE = ROOT / "artifacts/jev-runtime/decision3-publisher-lite-cpu-20261023-01"
PROFILE = "decision3-lite-gx10-publisher-cpu-batch1-20261023-v1"
WALL_SECONDS = 6500
CALL_SECONDS = 180
SMOKE_ACTOR_SHA = "33859d8b57bc906e477dce6ebf9d76ba4175ce775abb55cd9c42510f6de0bc9e"
SMOKE_PLAN_SHA = "947359fa5d1d71a15b9d7294fc945cedf16447cd5128b378a6630f0200713349"
ADAPTER_SHA = "ed5b75eed21b25b2916d51540d0a572c556be98ed057be3864d241f7c54869a8"
BENCHMARK_SOURCE_PLAN_SHA = "b5caa794d864917a637034bd1cebe0487c9eec0e5c3c423835d24fca4f397021"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "duplicate JSON key")
        value[key] = item
    return value


def read_json(path):
    return json.loads(
        Path(path).read_text(),
        object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")),
    )


def write_json(path, value):
    """Serialize before opening an exclusive temporary file."""
    raw = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def diagnostic(value):
    text = repr(value)
    raw = text.encode("utf-8", errors="backslashreplace")
    return {
        "encoding": "python_repr_diagnostic_only",
        "repr_prefix": raw[:1_048_576].decode("utf-8", errors="backslashreplace"),
        "repr_bytes": len(raw),
        "repr_sha256": hashlib.sha256(raw).hexdigest(),
        "diagnostic_truncated": len(raw) > 1_048_576,
    }


class PreservedRuntime:
    """Retain actual raw output before the frozen adapter's strict JSON writer.

    Only evidence is encoded differently for malformed/nonfinite output. The
    exact original object is returned to the unchanged adapter and scorers.
    """

    def __init__(self, runtime, output):
        self.actual, self.output, self.calls = runtime, Path(output), 0
        self.output.mkdir(exist_ok=False)

    def __getattr__(self, name):
        return getattr(self.actual, name)

    def system_one(self, **request):
        raw = self.actual.system_one(**request)
        path = self.output / (str(self.calls).zfill(5) + ".json")
        self.calls += 1
        try:
            write_json(path, {"raw_native_response": raw, "encoding": "strict_json"})
        except (TypeError, ValueError, OverflowError):
            write_json(
                path,
                {
                    "raw_native_response_diagnostic": diagnostic(raw),
                    "encoding": "invalid_native_response_diagnostic",
                    "valid_response_claim": False,
                },
            )
        return raw


def import_verified(name, path, expected):
    require(sha(path) == expected, "selected shared source differs: " + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        # Execute exactly the reviewed bytes, never a previously cached pyc.
        raw = Path(path).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == expected, "shared source changed before import")
        exec(compile(raw, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def check_smoke(path, expected):
    require(expected and sha(path) == expected, "selected successful smoke receipt SHA differs")
    receipt = read_json(path)
    require(
        receipt.get("profile") == "decision3-publisher-lite-cpu-batch1-smoke-v1",
        "wrong prerequisite smoke profile",
    )
    require(
        receipt.get("status") == "publisher_cpu_single_request_complete_no_benchmark_claim",
        "real successful publisher CPU request required",
    )
    require(
        receipt.get("plan_sha256") == SMOKE_PLAN_SHA
        and receipt.get("model_loads") == 1
        and receipt.get("forward_calls") == 1
        and receipt.get("benchmark_calls") == 0,
        "smoke constructor/forward identity differs",
    )
    require(
        receipt.get("cuda_initialized_final") is False
        and receipt.get("validation", {}).get("contract_valid") is True,
        "smoke did not qualify genuine CPU answer",
    )
    require(
        receipt.get("forward_seconds", math.inf) <= 20
        and receipt.get("elapsed_seconds", math.inf) <= 120,
        "smoke did not meet its fixed deadlines",
    )
    require(
        receipt.get("native_provenance", {}).get("model_name") == "d3-lite"
        and receipt["native_provenance"].get("batch_size") == 1
        and receipt["native_provenance"].get("model_sha256")
        == "fda5cfc7bc9063fbb7cd40b50fa9a0963d83a4790a3d6a11a356833fd1171992"
        and receipt["native_provenance"].get("files_sha256", {}).get("MODEL_MANIFEST.json")
        == "31f0ff94ff26619c0a063ccac803edcfe34c56c92e5c9c50b703b62893f125dd",
        "smoke actual Lite checkpoint/batch/manifest differs",
    )
    require(
        receipt.get("final_full_model_manifest_runtime_identities_unchanged") is True
        and receipt.get("forward_validation_and_evidence_seconds", math.inf) <= 20,
        "smoke final identity/late-validation guard missing",
    )
    return {
        "path": str(Path(path).resolve()),
        "sha256": expected,
        "status": receipt["status"],
        "profile": receipt["profile"],
        "not_a_benchmark_or_browser_score": True,
    }


def load_plan(path, expected):
    require(expected and sha(path) == expected, "selected campaign plan SHA differs")
    plan = read_json(path)
    require(
        plan.get("profile") == PROFILE and plan.get("campaign_sha256") == sha(__file__),
        "selected campaign source/profile differs",
    )
    require(
        plan.get("whole_campaign_seconds") == WALL_SECONDS
        and plan.get("per_call_seconds") == CALL_SECONDS,
        "selected campaign deadline differs",
    )
    require(
        plan.get("smoke_actor_sha256") == SMOKE_ACTOR_SHA
        and plan.get("smoke_plan_sha256") == SMOKE_PLAN_SHA
        and plan.get("adapter_sha256") == ADAPTER_SHA
        and plan.get("benchmark_source_plan_sha256") == BENCHMARK_SOURCE_PLAN_SHA,
        "selected shared source pins differ",
    )
    require(
        plan.get("context_limit") == 16384
        and plan.get("concurrency") == 1
        and plan.get("constructor_count") == 1,
        "selected benchmark runtime policy differs",
    )
    require(
        plan.get("owner_unit") == "jev-d3-llama-cpp-cpu4-3g-20261023-04.service"
        and plan.get("owner_profile") == "llama-cpp-cpu4-3g-command-owned-v1",
        "selected owner04 differs",
    )
    require(
        plan.get("reserve_bytes") == 26_119_172_916
        and plan.get("memory_max_bytes") == 3_221_225_472,
        "campaign budgets differ",
    )
    return plan


def latency_distribution(values):
    if not values:
        return {
            "count": 0,
            "median_seconds": None,
            "p95_seconds": None,
            "max_seconds": None,
            "startup_included": False,
        }
    ordered = sorted(values)
    size = len(ordered)
    median = ordered[size // 2] if size % 2 else (ordered[size // 2 - 1] + ordered[size // 2]) / 2
    return {
        "count": size,
        "median_seconds": median,
        "p95_seconds": ordered[max(0, math.ceil(0.95 * size) - 1)],
        "max_seconds": ordered[-1],
        "quantile_method": "nearest_rank_p95; ordinary_median",
        "startup_included": False,
        "unit": "one complete state/questions callable, independent D3 questions run batch1",
        "browser_20_second_qualification": False,
    }


class TimedPredictor:
    def __init__(self, adapter, helper, output):
        self.adapter, self.helper, self.output = adapter, helper, Path(output)
        self.latencies, self.started, self.completed = {}, 0, 0

    @property
    def stage(self):
        return self.adapter.stage

    @stage.setter
    def stage(self, value):
        self.adapter.stage = value

    def __call__(self, state, questions):
        index = self.started
        self.started += 1
        write_json(
            self.output / "progress.json",
            {
                "stage": self.stage,
                "started_calls": self.started,
                "completed_calls": self.completed,
                "active_call": index,
                "per_call_deadline_seconds": CALL_SECONDS,
            },
        )
        began = time.monotonic()
        primary = None
        try:
            with self.helper.Deadline(CALL_SECONDS, self.output, "call-" + str(index).zfill(5)):
                raw = self.adapter(state, questions)
            elapsed = time.monotonic() - began
            require(elapsed <= CALL_SECONDS, "benchmark callable returned after its fixed deadline")
            self.latencies.setdefault(self.stage, []).append(elapsed)
            self.completed += 1
            return raw
        except BaseException as exc:
            primary = exc
            raise
        finally:
            try:
                write_json(
                    self.output / "progress.json",
                    {
                        "stage": self.stage,
                        "started_calls": self.started,
                        "completed_calls": self.completed,
                        "active_call": None,
                        "latencies": self.summary(),
                        "per_call_deadline_seconds": CALL_SECONDS,
                    },
                )
            except BaseException as secondary:
                if primary is None:
                    raise
                primary.add_note("Prediction progress evidence failed: " + repr(secondary))

    def answers(self, state, questions):
        return self(state, questions)["answers"]

    def summary(self):
        return {stage: latency_distribution(values) for stage, values in self.latencies.items()}


def execute(plan_path, plan_sha, smoke_receipt, smoke_sha, output):
    require(
        not {"torch", "transformers", "safetensors", "numpy", "jevbench", "s1"}.intersection(
            sys.modules
        ),
        "campaign began with model/harness modules imported",
    )
    require(sys.flags.no_user_site and sys.dont_write_bytecode, "runtime requires -B -s")
    plan = load_plan(plan_path, plan_sha)
    # A real request result, selected in the frozen owner command, is required
    # before this campaign can authenticate payloads or import model libraries.
    prerequisite = check_smoke(smoke_receipt, smoke_sha)
    helper = import_verified("_decision3_cpu_frozen_actor", SMOKE / "actor.py", SMOKE_ACTOR_SHA)
    adapter_module = import_verified(
        "_decision3_cpu_frozen_adapter", SMOKE / "benchmark_adapter.py", ADAPTER_SHA
    )
    require(
        sha(SMOKE / "benchmark-source-plan.json") == BENCHMARK_SOURCE_PLAN_SHA,
        "selected official-source plan changed",
    )
    benchmark = read_json(SMOKE / "benchmark-source-plan.json")
    native_plan = helper.load_plan(SMOKE / "plan.json", SMOKE_PLAN_SHA)
    identity = helper.process_identity()
    limits = helper.owned_limits(identity)
    helper.validate_owner(identity, limits)
    gate = helper.host_gate()
    output = Path(output).resolve(strict=False)
    require(
        output.is_relative_to(HERE.parent.resolve()) and not output.exists(),
        "new output must resolve inside runtime artifact tree",
    )
    output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    state = {
        "profile": PROFILE,
        "status": "authenticating_sources",
        "plan_sha256": plan_sha,
        "prerequisite": prerequisite,
        "process": identity,
        "owned_limits": limits,
        "host_startup_admission": gate,
        "whole_campaign_deadline_seconds": WALL_SECONDS,
        "per_call_deadline_seconds": CALL_SECONDS,
        "per_call_deadline_basis": "official benchmark HTTP default; not browser 20 seconds",
        "model_loads": 0,
        "startup_excluded_from_suite_latency": True,
        "browser_score": None,
        "benchmark_scores": None,
        "local_head_at_plan_creation": plan["local_head_at_creation"],
        "head_field_is_provenance_only": True,
    }
    write_json(output / "state.json", state)
    predictor = None
    primary = None
    with helper.Deadline(WALL_SECONDS, output, "campaign"):
        try:
            require(len(benchmark["source_files"]) == 78, "official source population differs")
            for row in benchmark["source_files"] + benchmark["existing_dataset_metadata"]:
                helper.check_record(row, full=True)
            state["official_source_files_authenticated"] = 78
            write_json(output / "state.json", state)
            runtime, torch = helper.load_publisher_cpu(native_plan, output, state)
            require(state["model_loads"] == 1, "campaign must have one actual model constructor")
            # Only model inputs enter the adapter. Benchmark data and scoring
            # remain entirely inside the unchanged authenticated harnesses.
            retained_runtime = PreservedRuntime(runtime, output / "raw-native-responses")
            adapter = adapter_module.PublisherCPUAdapter(
                retained_runtime,
                torch,
                output / "native-evidence",
                check_runtime=helper.validate_runtime,
                context_limit=16384,
            )
            predictor = TimedPredictor(adapter, helper, output)
            public = import_verified(
                "_decision3_cpu_public_harness",
                ROOT / "scripts/run_jevbench_public.py",
                next(
                    row["sha256"]
                    for row in benchmark["source_files"]
                    if row["path"] == str(ROOT / "scripts/run_jevbench_public.py")
                ),
            )
            coherence = Path(benchmark["coherence"]["root"])
            sys.path.insert(0, str(coherence / "src"))
            import jevbench as jb

            require(
                Path(jb.__file__).resolve() == (coherence / "src/jevbench/__init__.py").resolve(),
                "coherence import origin differs",
            )
            state["status"] = "running_original_public231_then_coherence_mini"
            state["model_identity"] = {
                "model_id": "vllm-sr/d3-lite",
                "revision": native_plan["revision"],
                "profile": PROFILE,
                "device": "cpu",
                "backbone_dtype": "bfloat16",
                "readout_dtype": "float32",
                "temperature": 1.0,
                "batch_size": 1,
                "context_limit": 16384,
                "context_policy": "reject_no_truncation",
                "permutation_average": False,
                "constructor_count": 1,
                "benchmark_heaven_revision": benchmark["public"]["revision"],
                "coherence_revision": benchmark["coherence"]["revision"],
                "source_plan_sha256": BENCHMARK_SOURCE_PLAN_SHA,
            }
            write_json(output / "state.json", state)
            result = adapter_module.run_unchanged_suites(
                predictor,
                state["model_identity"],
                output,
                run_public=public.run_public,
                heaven_root=benchmark["public"]["root"],
                jb=jb,
            )
            json.dumps(result, allow_nan=False)
            state["benchmark_scores"] = result
            state["latencies"] = predictor.summary()
            state["started_calls"], state["completed_calls"] = (
                predictor.started,
                predictor.completed,
            )
            require(
                predictor.started == predictor.completed == 1479,
                "both full suites require 1479 fresh actual model callable returns",
            )
            require(
                set(predictor.latencies) == {"legacy-public231", "jevbench-mini"}
                and len(predictor.latencies["legacy-public231"]) == 231
                and len(predictor.latencies["jevbench-mini"]) == 1248,
                "per-suite latency population differs",
            )
            helper.validate_runtime(runtime, torch)
            for row in benchmark["source_files"] + benchmark["existing_dataset_metadata"]:
                helper.check_record(row, full=True)
            for row in native_plan["model_files"]:
                helper.check_record(row, full=True)
            helper.check_record(native_plan["manifest"], full=True)
            for row in native_plan["runtime_files"]:
                helper.check_record(row, full="sha256" in row)
            for filename, expected in (
                ("actor.py", SMOKE_ACTOR_SHA),
                ("plan.json", SMOKE_PLAN_SHA),
                ("benchmark_adapter.py", ADAPTER_SHA),
                ("benchmark-source-plan.json", BENCHMARK_SOURCE_PLAN_SHA),
            ):
                require(
                    sha(SMOKE / filename) == expected,
                    "selected shared source changed during campaign",
                )
            state["final_official_sources_and_full_model_sha256_unchanged"] = True
            state["final_runtime_sources_and_selected_shared_files_unchanged"] = True
            state["torch_threads_final"] = {
                "intraop": torch.get_num_threads(),
                "interop": torch.get_num_interop_threads(),
            }
            state["closure_scope"] = (
                "actor results only; root must authenticate generic owner's external terminal/empty cgroup closure and measured peak separately"
            )
            state["cuda_initialized_final"] = torch.cuda.is_initialized()
            require(
                state["cuda_initialized_final"] is False,
                "CUDA initialized during final Lite campaign authentication",
            )
            helper.ensure_elapsed(began, WALL_SECONDS, "whole Lite campaign final authentication")
            state["status"] = "both_original_suites_complete_cpu_profile_only"
            write_json(
                output / "results.json",
                {
                    "benchmark_scores": result,
                    "latencies": state["latencies"],
                    "model_identity": state["model_identity"],
                    "browser_score": None,
                    "latency_startup_included": False,
                },
            )
            helper.ensure_elapsed(began, WALL_SECONDS, "whole Lite campaign results evidence")
            return state
        except BaseException as exc:
            primary = exc
            state["status"] = "failed_or_incomplete_no_complete_score_claim"
            state["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            if predictor is not None:
                state["latencies"] = predictor.summary()
                state["started_calls"], state["completed_calls"] = (
                    predictor.started,
                    predictor.completed,
                )
            state["elapsed_seconds"] = time.monotonic() - began
            try:
                write_json(output / "state.json", state)
            except BaseException as secondary:
                if primary is None:
                    raise
                primary.add_note("Final campaign state could not be written: " + repr(secondary))
                try:
                    write_json(
                        output / "final-evidence-error.json",
                        {
                            "status": "failed_no_complete_score_claim",
                            "primary_error": {
                                "type": type(primary).__name__,
                                "message": str(primary),
                            },
                            "secondary_error": {
                                "type": type(secondary).__name__,
                                "message": str(secondary),
                            },
                            "state_diagnostic": diagnostic(state),
                        },
                    )
                except BaseException as tertiary:
                    primary.add_note("Fallback evidence write also failed: " + repr(tertiary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--plan", type=Path, default=HERE / "plan.json")
    parser.add_argument("--plan-sha256")
    parser.add_argument("--smoke-receipt", type=Path)
    parser.add_argument("--smoke-receipt-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "profile": PROFILE,
                    "status": "dry_default_no_model_or_benchmark_execution",
                    "whole_campaign_seconds": WALL_SECONDS,
                    "per_call_seconds": CALL_SECONDS,
                    "constructor_count": 1,
                    "suites": ["legacy-public231", "jevbench-mini"],
                    "requires_successful_frozen_smoke_receipt": True,
                    "browser_20_second_qualification": False,
                }
            )
        )
        return
    require(
        args.output is not None and args.smoke_receipt is not None,
        "execution requires new output and exact successful smoke receipt",
    )
    result = execute(
        args.plan, args.plan_sha256, args.smoke_receipt, args.smoke_receipt_sha256, args.output
    )
    print(json.dumps({"profile": PROFILE, "status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
