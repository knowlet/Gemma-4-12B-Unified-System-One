#!/usr/bin/env python3
"""Dry-default native batch-four CPU campaign; unchanged official scorers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BATCH = ROOT / "artifacts/jev-runtime/decision3-publisher-batch4-cpu-20261024-04"
BATCH_ACTOR_SHA = "PENDING_NATIVE04_REVIEWED_ACTOR_SHA256"
BATCH_FREEZE_SHA = "PENDING_NATIVE04_REVIEWED_FREEZE_SHA256"
NATIVE_PLAN_SHA = {"edge": "PENDING_NATIVE04_REVIEWED_PLAN_SHA256"}
HOST_OBSERVATION_POLICY = "comfy-observed-cpu-coexist-v1"

BENCHMARK_PLAN = (
    ROOT
    / "artifacts/jev-runtime/decision3-publisher-lite-cpu-20261023-01/benchmark-source-plan.json"
)
BENCHMARK_PLAN_SHA = "b5caa794d864917a637034bd1cebe0487c9eec0e5c3c423835d24fca4f397021"
WALL_SECONDS = 10000
CALL_SECONDS = 180
OWNER_SECONDS = 10800
STAGE_SCHEMA = "authenticated-native-batch4-stage-completion-v1"
STAGE_POLICY = "full-auth-durable-stage-evidence-within-original-whole10000-root-audit-required"
STAGES = ("legacy-public231", "jevbench-mini")
COUNTS = {STAGES[0]: 231, STAGES[1]: 1248}
QUALIFICATIONS = (
    "smoke_complete",
    "external_authenticated_closure",
    "one_actual_constructor",
    "two_actual_four_row_events",
    "durable_call_20s",
    "whole_command_120s",
    "final_full_model_and_sources_authenticated",
)


class CampaignAbort(BaseException):
    """A deadline/evidence/admission failure must escape per-case scorer handlers."""


class DeadlineScope:
    """Finish the frozen watchdog while retaining any already propagating error."""

    def __init__(self, deadline):
        self.deadline = deadline

    def __enter__(self):
        return self.deadline.__enter__()

    def __exit__(self, kind, primary, traceback):
        try:
            self.deadline.__exit__(kind, primary, traceback)
        except BaseException as secondary:
            if primary is None:
                raise
            primary.add_note("Deadline cleanup also failed: " + repr(secondary))
        return False


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def read_json(path):
    return json.loads(
        Path(path).read_text(),
        object_pairs_hook=unique,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")),
    )


def selected_small_json(path, expected):
    p = Path(path)
    require(
        p.suffix == ".json" and p.stat().st_size <= 1048576,
        "selected small JSON receipt/source manifest required",
    )
    require(expected and sha(p) == expected, "selected JSON receipt/source SHA differs")
    return read_json(p)


def finite_seconds(value, cap):
    return type(value) in (float, int) and math.isfinite(value) and 0 <= value <= cap


def profile(key):
    return f"decision3-{key}-gx10-publisher-cpu16-4g-coexist-batch4-both-stage-completions-ece-schema-20261024-v1"


def load_plan(path, expected):
    plan = selected_small_json(path, expected)
    key = plan.get("model_key")
    require(
        key in NATIVE_PLAN_SHA
        and plan.get("profile") == profile(key)
        and plan.get("campaign_sha256") == sha(__file__),
        "campaign model/profile/source differs",
    )
    require(
        plan.get("batch_actor_sha256") == BATCH_ACTOR_SHA
        and plan.get("batch_freeze_sha256") == BATCH_FREEZE_SHA
        and plan.get("native_plan_sha256") == NATIVE_PLAN_SHA[key]
        and plan.get("benchmark_source_plan_sha256") == BENCHMARK_PLAN_SHA,
        "selected frozen dependencies differ",
    )
    native_metadata = selected_small_json(BATCH / (key + "-plan.json"), NATIVE_PLAN_SHA[key])
    require(
        plan.get("model") == native_metadata["model"]
        and plan.get("owner_sources") == native_metadata["owner_sources"]
        and plan.get("host_observation_policy")
        == native_metadata.get("host_observation_policy")
        == HOST_OBSERVATION_POLICY,
        "candidate model/owner identity metadata differs",
    )
    require(
        plan.get("whole_campaign_seconds") == WALL_SECONDS
        and plan.get("per_call_seconds") == CALL_SECONDS
        and plan.get("owner_runtime_seconds") == OWNER_SECONDS,
        "whole/per-call/owner deadlines differ",
    )
    require(
        plan.get("context_limit") == 16384
        and plan.get("context_policy") == "reject_no_truncation"
        and plan.get("concurrency") == 1
        and plan.get("constructor_count") == 1
        and plan.get("batch_size") == 4
        and plan.get("fresh_calls") == 1479,
        "native full-request batch4 policy differs",
    )
    require(
        plan.get("owner_unit") == "jev-d3-llama-cpp-cpu16-4g-coexist-20261024-08.service"
        and plan.get("owner_profile") == "llama-cpp-cpu16-4g-coexist-command-owned-v1"
        and plan.get("reserve_bytes") == 26119172916
        and plan.get("memory_max_bytes") == 4294967296
        and plan.get("startup_available_bytes") == 30414140212
        and plan.get("swap_max_bytes") == 0
        and plan.get("threads") == 16
        and plan.get("cpu_quota_percent") == 1600
        and plan.get("cpu_affinity") == [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        "owner08 CPU16/4GiB-coexist/memory policy differs",
    )
    require(
        plan.get("stage_completion_schema") == STAGE_SCHEMA
        and plan.get("stage_completion_policy") == STAGE_POLICY,
        "full authenticated stage completion policy differs",
    )
    return plan


def check_smoke(path, expected, native_plan):
    receipt = selected_small_json(path, expected)
    key, model = native_plan["model_key"], native_plan["model"]
    require(
        receipt.get("profile") == native_plan["profile"]
        and receipt.get("selected_owner_host_observation_policy")
        == native_plan.get("host_observation_policy")
        == HOST_OBSERVATION_POLICY
        and receipt.get("model_key") == key
        and receipt.get("plan_sha256") == NATIVE_PLAN_SHA[key]
        and receipt.get("status")
        == "native_batch4_two_call_synthetic_probe_complete_no_benchmark_claim",
        "same-model successful fresh native batch4 smoke required",
    )
    require(
        receipt.get("model_loads") == 1
        and receipt.get("native_request_calls") == 2
        and receipt.get("benchmark_calls") == 0
        and receipt.get("browser_calls") == 0
        and receipt.get("cuda_initialized_final") is False
        and receipt.get("final_full_model_manifest_runtime_source_identities_unchanged") is True
        and finite_seconds(receipt.get("elapsed_seconds"), 120),
        "smoke constructor/CPU/final-auth/whole deadline differs",
    )
    provenance = receipt.get("native_provenance", {})
    require(
        provenance.get("model_name") == model["name"]
        and provenance.get("batch_size") == 4
        and provenance.get("model_sha256") == model["model_sha256"]
        and provenance.get("files_sha256", {}).get("MODEL_MANIFEST.json")
        == model["manifest_sha256"],
        "smoke actual checkpoint/batch/manifest differs",
    )
    calls, events = receipt.get("calls", []), receipt.get("actual_first_layer_events", [])
    require(len(calls) == len(events) == 2, "two actual smoke calls/events required")
    for index, (call, event) in enumerate(zip(calls, events)):
        shape = event.get("shape", [])
        require(
            call.get("index") == index
            and call.get("status") == "validated"
            and call.get("validation", {}).get("contract_valid") is True
            and call.get("actual_first_layer_event") == event
            and finite_seconds(call.get("durable_validation_seconds"), 20)
            and finite_seconds(call.get("validation_seconds"), 20)
            and len(shape) == 3
            and shape[0] == 4
            and shape[2] == 1024
            and 0 < shape[1] <= 512
            and event.get("dtype") == "torch.bfloat16"
            and event.get("device") == "cpu"
            and event.get("inputs_or_outputs_modified") is False,
            "smoke actual four-row CPU BF16/durable 20s evidence differs",
        )
    return receipt


def check_audit(path, expected, smoke_path, smoke_sha, native_plan):
    """Root-selected independent audit cryptographically pins closed smoke evidence.

    Audit qualification is independently reviewed evidence, never inferred from
    actor status or this campaign's own source-only tests. All named small refs
    are authenticated without probing a process, unit, manager or model file.
    """
    audit = selected_small_json(path, expected)
    require(
        audit.get("schema") == "independent-native-batch4-synthetic-result-audit-v1"
        and audit.get("review_kind") == "closed_real_smoke_result"
        and audit.get("model_key") == native_plan["model_key"]
        and audit.get("result") in {"PASS", "PASS_WITH_LIMITS"},
        "closed independent native batch4 audit required",
    )
    require(
        all(audit.get("qualification", {}).get(k) is True for k in QUALIFICATIONS),
        "independent smoke admission/deadline/closure qualification incomplete",
    )
    model = native_plan["model"]
    require(
        audit.get("model_revision") == model["revision"]
        and audit.get("model_identity_sha256") == model["model_sha256"]
        and audit.get("manifest_sha256") == model["manifest_sha256"]
        and audit.get("owner_unit") == native_plan["owner_unit"]
        and audit.get("host_observation_policy")
        == native_plan.get("host_observation_policy")
        == HOST_OBSERVATION_POLICY
        and audit.get("memory_max_bytes") == native_plan["memory_max_bytes"]
        and audit.get("batch_size") == 4,
        "independent audit actual model/batch/cap differs",
    )
    refs = audit.get("artifact_refs", {})
    require(
        isinstance(refs, dict)
        and {"state", "plan", "owner_receipt", "root_execution", "spec", "closure_observation"}
        <= refs.keys(),
        "exact independent smoke state/plan/owner/root/spec/closure references required",
    )
    for name, ref in refs.items():
        p = Path(ref["path"])
        if not p.is_absolute():
            p = ROOT / p
        require(
            p.suffix.lower() not in {".safetensors", ".bin", ".pt", ".npz", ".gguf", ".so"}
            and p.name not in {"tokenizer.json", "merges.txt", "vocab.json"}
            and p.stat().st_size <= 1048576
            and p.stat().st_size == ref["size_bytes"]
            and sha(p) == ref["sha256"],
            "independent audit small artifact identity changed: " + name,
        )

    def resolved_ref(name):
        p = Path(refs[name]["path"])
        return (p if p.is_absolute() else ROOT / p).resolve()

    require(
        resolved_ref("state") == Path(smoke_path).resolve()
        and refs["state"]["sha256"] == smoke_sha
        and resolved_ref("plan") == (BATCH / (native_plan["model_key"] + "-plan.json")).resolve()
        and refs["plan"]["sha256"] == NATIVE_PLAN_SHA[native_plan["model_key"]],
        "audit does not qualify the exact selected smoke state/model plan",
    )
    return {
        "path": str(Path(path).resolve()),
        "sha256": expected,
        "result": audit["result"],
        "qualification": audit["qualification"],
        "artifact_refs": refs,
        "limits": audit.get("limits", []),
    }


def latency_distribution(values):
    ordered = sorted(values)
    n = len(ordered)
    return {
        "count": n,
        "median_seconds": (
            ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
        )
        if n
        else None,
        "p95_seconds": ordered[max(0, math.ceil(0.95 * n) - 1)] if n else None,
        "max_seconds": ordered[-1] if n else None,
        "quantile_method": "nearest_rank_p95; ordinary_median",
        "startup_included": False,
        "browser_20_second_qualification": False,
        "unit": "one unchanged complete state/questions callback; native question chunks of up to4",
    }


class TimedPredictor:
    """Evidence sink receives original native output before backend validation."""

    def __init__(self, backend, helper, output, counters):
        self.backend, self.helper, self.output, self.counters = (
            backend,
            helper,
            Path(output),
            counters,
        )
        self.stage, self.started, self.completed = "not_started", 0, 0
        self.latencies, self.active, self.raw_returned = {}, None, 0
        self.output.mkdir(exist_ok=False)
        backend.evidence_sink = self.record_native

    def record_native(self, event, value):
        require(self.active is not None, "native evidence outside selected callback")
        name = f"{self.active['index']:05d}-{event}.json"
        payload = {
            "index": self.active["index"],
            "stage": self.stage,
            "event": event,
            "value": value,
        }
        if event == "raw_returned_before_validation":
            self.raw_returned += 1
            try:
                self.helper.write_json(self.output / name, payload)
            except (TypeError, ValueError, OverflowError):
                self.helper.write_json(
                    self.output / name,
                    {
                        "index": self.active["index"],
                        "stage": self.stage,
                        "event": event,
                        "invalid_native_response_diagnostic": self.helper.diagnostic(value),
                        "valid_response_claim": False,
                    },
                )
        else:
            self.helper.write_json(self.output / name, payload)

    def summary(self):
        return {stage: latency_distribution(values) for stage, values in self.latencies.items()}

    def progress(self):
        return {
            "stage": self.stage,
            "started_calls": self.started,
            "completed_calls": self.completed,
            "raw_native_returns": self.raw_returned,
            "active_call": self.active,
            "latencies": self.summary(),
            "per_call_deadline_seconds": CALL_SECONDS,
        }

    def __call__(self, state, questions):
        require(
            self.active is None and self.stage in STAGES,
            "sequential original suite callback required",
        )
        began = time.monotonic()
        index = self.started
        self.started += 1
        primary = None
        counted = False
        record = self.active = {"index": index, "stage": self.stage, "status": "running"}
        try:
            with DeadlineScope(
                self.helper.Deadline(CALL_SECONDS, self.output, f"call-{index:05d}")
            ):
                try:
                    self.active["before_owned_counters"] = self.counters()
                    self.helper.write_json(self.output / "progress.json", self.progress())
                    raw = self.backend(state, questions)
                    self.active["after_owned_counters"] = self.counters()
                    self.active["validation_seconds"] = self.helper.ensure_elapsed(
                        began, CALL_SECONDS, "native callback validation"
                    )
                    self.active["status"] = "validated_pending_durable_evidence"
                    self.helper.write_json(self.output / f"{index:05d}-validated.json", self.active)
                    self.helper.ensure_elapsed(
                        began, CALL_SECONDS, "native callback durable evidence"
                    )
                    self.completed += 1
                    self.latencies.setdefault(self.stage, []).append(time.monotonic() - began)
                    counted = True
                    self.active["status"] = "complete"
                    self.helper.write_json(self.output / "progress.json", self.progress())
                    self.helper.ensure_elapsed(
                        began, CALL_SECONDS, "native callback durable progress"
                    )
                    # The final aggregate includes the durable progress write too.
                    self.latencies[self.stage][-1] = time.monotonic() - began
                except BaseException as exc:
                    primary = exc
                    if counted:
                        self.completed -= 1
                        self.latencies[self.stage].pop()
                    self.active["status"] = "failed_no_valid_return"
                    self.active["elapsed_seconds"] = time.monotonic() - began
                    self.active["error"] = {"type": type(exc).__name__, "message": str(exc)}
                    if time.monotonic() - began > CALL_SECONDS:
                        raise CampaignAbort(
                            "callback exceeded180s including validation and durable evidence"
                        ) from exc
                    raise
                finally:
                    try:
                        if primary is not None:
                            self.helper.write_json(
                                self.output / f"{index:05d}-failed.json", self.active
                            )
                            self.helper.write_json(self.output / "progress.json", self.progress())
                    except BaseException as secondary:
                        if primary is None:
                            raise CampaignAbort("callback final evidence failed") from secondary
                        primary.add_note(
                            "Callback failure evidence also failed: " + repr(secondary)
                        )
                    finally:
                        self.active = None
                    if primary is not None and time.monotonic() - began > CALL_SECONDS:
                        raise CampaignAbort(
                            "failure evidence exceeded180s; primary retained as cause"
                        ) from primary
            elapsed = self.helper.ensure_elapsed(
                began, CALL_SECONDS, "native callback after finalization and deadline cleanup"
            )
        except BaseException as exc:
            if primary is not None:
                if time.monotonic() - began > CALL_SECONDS and not isinstance(exc, CampaignAbort):
                    raise CampaignAbort(
                        "failure finalization exceeded180s after deadline cleanup; primary retained as cause"
                    ) from exc
                raise
            if counted:
                self.completed -= 1
                self.latencies[self.stage].pop()
            record.update(
                status="failed_no_valid_return",
                elapsed_seconds=time.monotonic() - began,
                error={"type": type(exc).__name__, "message": str(exc)},
            )
            self.active = record
            try:
                self.helper.write_json(self.output / f"{index:05d}-failed.json", record)
                self.helper.write_json(self.output / "progress.json", self.progress())
            except BaseException as secondary:
                exc.add_note(
                    "Post-cleanup callback failure evidence also failed: " + repr(secondary)
                )
            finally:
                self.active = None
            raise CampaignAbort("callback finalization or post-cleanup deadline failed") from exc
        self.latencies[self.stage][-1] = elapsed
        return raw

    def answers(self, state, questions):
        return self(state, questions)["answers"]


def project_public_ece(value, calibration_n):
    """Copy the pinned official global ECE object unchanged; never rescore it."""
    if value is None:
        require(calibration_n == 0, "absent official ECE requires zero calibration pairs")
        return None
    require(
        type(value) is dict and set(value) == {"ece", "n", "bins"},
        "official ECE object keys differ",
    )

    def unit(number):
        return type(number) in (int, float) and math.isfinite(number) and 0 <= number <= 1

    require(
        unit(value["ece"])
        and type(value["n"]) is int
        and 0 <= value["n"] <= 231
        and value["n"] == calibration_n,
        "official ECE metric/count differs",
    )
    bins = value["bins"]
    require(type(bins) is list and len(bins) == 10, "official ECE requires ten global bins")
    projected = []
    for index, item in enumerate(bins):
        require(
            type(item) is dict and set(item) == {"lo", "hi", "n", "mean_confidence", "accuracy"},
            "official ECE bin keys differ",
        )
        require(
            unit(item["lo"])
            and unit(item["hi"])
            and item["lo"] == index / 10
            and item["hi"] == (index + 1) / 10
            and type(item["n"]) is int
            and 0 <= item["n"] <= value["n"],
            "official ECE bin edges/count differ",
        )
        if item["n"] == 0:
            require(
                item["mean_confidence"] is None and item["accuracy"] is None,
                "empty official ECE bin means must be absent",
            )
        else:
            require(
                unit(item["mean_confidence"]) and unit(item["accuracy"]),
                "populated official ECE bin means must be finite unit metrics",
            )
        projected.append(dict(item))
    require(
        sum(item["n"] for item in bins) == value["n"],
        "official ECE bin total differs from calibration count",
    )
    return {"ece": value["ece"], "n": value["n"], "bins": projected}


def stage_metrics(stage, aggregate):
    """Copy only typed official global aggregates, never item records or responses."""
    require(type(aggregate) is dict, "official stage aggregate required")
    if stage == STAGES[0]:
        integers = (
            "n_planned",
            "n_attempted",
            "n_scorable",
            "n_valid",
            "n_correct",
            "n_renormalized",
            "calibration_n",
        )
        numbers = (
            "accuracy",
            "macro_accuracy",
            "coverage",
            "schema_validity",
            "schema_validity_strict",
            "operational_success",
            "brier_mean",
            "ordinal_mae",
            "price_per_1000_decisions_usd",
            "ledger_charged_usd",
        )
        require(
            aggregate.get("complete") is True
            and aggregate.get("n_planned") == aggregate.get("n_attempted") == 231,
            "complete original public231 official aggregate required",
        )
        result = {key: aggregate[key] for key in integers + numbers}
        require(
            all(type(result[key]) is int and 0 <= result[key] <= 231 for key in integers),
            "public aggregate integer counts differ",
        )
        result["ece"] = project_public_ece(aggregate["ece"], result["calibration_n"])
        result["complete"] = True
    else:
        require(stage == STAGES[1], "selected original stage required")
        require(
            {key: aggregate.get(key) for key in ("requests", "cases", "relations", "planned_tests")}
            == {"requests": 1248, "cases": 240, "relations": 50, "planned_tests": 1200},
            "complete original coherence counts required",
        )
        result = {
            key: aggregate[key]
            for key in ("overall", "interval", "requests", "cases", "relations", "planned_tests")
        }
        numbers = ("overall",)
        interval = result["interval"]
        require(
            interval is None
            or (
                type(interval) in (list, tuple)
                and len(interval) == 2
                and all(type(v) in (int, float) and math.isfinite(v) for v in interval)
            ),
            "official aggregate interval must be finite or absent",
        )
        if interval is not None:
            result["interval"] = list(interval)
    require(
        all(
            result[key] is None
            or (type(result[key]) in (int, float) and math.isfinite(result[key]))
            for key in numbers
        ),
        "official metric must be finite numeric or absent",
    )
    json.dumps(result, allow_nan=False)
    return result


def durable_stage_json(helper, path, value):
    """Strict original atomic file write, directory fsync and exact reread."""
    helper.write_json(path, value)
    descriptor = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    require(read_json(path) == value, "durable stage evidence reread differs")


class StageCompletions:
    """Additional full-auth stage evidence within the original whole budget."""

    def __init__(
        self,
        helper,
        batch,
        native,
        benchmark,
        identity,
        output,
        state,
        predictor,
        began,
        began_unix,
        authenticate,
        selected_sources,
    ):
        self.helper, self.batch, self.native, self.benchmark = helper, batch, native, benchmark
        self.identity, self.state, self.predictor = identity, state, predictor
        self.output = Path(output) / "stage-completions"
        self.output.mkdir(exist_ok=False)
        self.began, self.began_unix = began, began_unix
        self.authenticate, self.selected_sources = authenticate, selected_sources
        self.nonce = os.environ.get("JEV_COMMAND_NONCE")
        require(
            type(self.nonce) is str
            and self.nonce.startswith("jev-d3-boundary-")
            and len(self.nonce) == 48
            and all(v in "0123456789abcdef" for v in self.nonce[16:]),
            "original selected owner nonce required for stage binding",
        )
        self.completed = []

    def owned_identity(self):
        current = self.helper.process_identity()
        limits = self.helper.owned_limits(current)
        self.batch.validate_owner(current, limits, self.helper.START_ENV)
        require(
            all(
                current[key] == self.identity[key]
                for key in (
                    "pid",
                    "starttime_ticks",
                    "pgid",
                    "sid",
                    "cgroup",
                    "affinity",
                    "executable",
                )
            )
            and limits == self.state["owned_limits"]
            and os.environ.get("JEV_COMMAND_NONCE") == self.nonce,
            "stage owner/process/nonce binding changed",
        )
        return {
            "unit": self.native["owner_unit"],
            "profile": self.native["owner_profile"],
            "nonce": self.nonce,
            "process": current,
            "limits": limits,
        }

    def verify_all(self):
        require(self.completed == list(STAGES), "both authenticated stage completions required")
        require(
            set(self.state["completed_stage_receipts"]) == set(STAGES), "both stage refs required"
        )
        for stage, ref in self.state["completed_stage_receipts"].items():
            receipt = selected_small_json(ref["path"], ref["sha256"])
            require(
                receipt["schema"] == STAGE_SCHEMA
                and receipt["stage"] == stage
                and receipt["status"]
                == "authenticated_stage_complete_pending_independent_current_root_audit"
                and receipt["owner_nonce"] == self.nonce,
                "stage completion receipt changed or refused",
            )
            selected_small_json(receipt["evidence"]["path"], receipt["evidence"]["sha256"])

    def complete(self, stage, aggregate, stage_began):
        require(
            len(self.completed) < len(STAGES) and stage == STAGES[len(self.completed)],
            "both complete stages must preserve original order",
        )
        require(
            type(stage_began) in (int, float)
            and math.isfinite(stage_began)
            and self.began <= stage_began <= time.monotonic(),
            "stage start must use the original monotonic campaign clock",
        )
        cumulative = sum(COUNTS[s] for s in STAGES[: len(self.completed) + 1])
        p = self.predictor
        require(
            p.stage == stage
            and p.active is None
            and p.started == p.completed == p.raw_returned == cumulative
            and set(p.latencies) == set(STAGES[: len(self.completed) + 1])
            and all(len(p.latencies[s]) == COUNTS[s] for s in STAGES[: len(self.completed) + 1]),
            "stage exact fresh callback/native/latency populations required",
        )
        directory = self.output / stage
        directory.mkdir(exist_ok=False)
        evidence_path, receipt_path = directory / "evidence.json", directory / "completion.json"
        began = self.began
        try:
            remaining = WALL_SECONDS - (time.monotonic() - began)
            require(remaining > 0, "stage exhausted original whole campaign deadline")
            with DeadlineScope(self.helper.Deadline(remaining, directory, "stage-authentication")):
                metrics = stage_metrics(stage, aggregate)
                before = self.owned_identity()
                self.authenticate()
                after = self.owned_identity()
                require(before == after, "stage owner identity changed during authentication")
                elapsed = self.helper.ensure_elapsed(
                    began, WALL_SECONDS, "stage full authentication"
                )
                evidence = {
                    "schema": STAGE_SCHEMA,
                    "status": "evidence_only_not_a_stage_completion",
                    "stage": stage,
                    "profile": self.state["profile"],
                    "model_key": self.state["model_key"],
                    "metrics": metrics,
                    "counts": {
                        "stage_callbacks": COUNTS[stage],
                        "cumulative_callbacks": cumulative,
                        "started": p.started,
                        "completed": p.completed,
                        "raw_native_returns": p.raw_returned,
                    },
                    "latency": p.summary()[stage],
                    "owner": after,
                    "invocation": {
                        "output": str(self.output.parent),
                        "plan_sha256": self.state["plan_sha256"],
                        "campaign_started_unix": self.began_unix,
                        "campaign_started_monotonic": began,
                    },
                    "clock": {
                        "stage_started_monotonic": stage_began,
                        "authenticated_monotonic": time.monotonic(),
                        "elapsed_campaign_seconds": elapsed,
                        "whole_campaign_seconds": WALL_SECONDS,
                        "per_call_seconds": CALL_SECONDS,
                        "owner_seconds": OWNER_SECONDS,
                        "scope": "original execute-entry whole10000 includes stage scoring, full authentication, retention, publication and cleanup; no extra budget",
                    },
                    "full_authentication": {
                        "unchanged_batch_authenticate": True,
                        "actual_runtime_identity_and_cpu_BF16_FP32_dtypes_validated": True,
                        "all78_scorer_sources_plus2_dataset_metadata_authenticated": True,
                        "campaign_plan_prerequisites_and_every_audit_ref_authenticated": True,
                    },
                    "pins": {
                        "model": self.native["model"],
                        "manifest": self.native["manifest"],
                        "model_files": self.native["model_files"],
                        "runtime_files": self.native["runtime_files"],
                        "owner_sources": self.native["owner_sources"],
                        "benchmark_source_plan_sha256": BENCHMARK_PLAN_SHA,
                        "scorer_sources": self.benchmark["source_files"],
                        "dataset_metadata": self.benchmark["existing_dataset_metadata"],
                        "campaign_sources_and_prerequisites": [
                            {"path": str(path), "sha256": expected}
                            for path, expected in self.selected_sources
                        ],
                    },
                    "raw_tasks_gold_responses_or_per_item_scores_included": False,
                    "runtime_pin_scope": "original selected runtime SHA when supplied, plus original stat metadata; not full native DSO closure",
                }
                durable_stage_json(self.helper, evidence_path, evidence)
                self.helper.ensure_elapsed(
                    began, WALL_SECONDS, "stage durable authenticated evidence"
                )
            self.helper.ensure_elapsed(began, WALL_SECONDS, "stage after deadline cleanup")
            receipt = {
                "schema": STAGE_SCHEMA,
                "status": "authenticated_stage_complete_pending_independent_current_root_audit",
                "stage": stage,
                "profile": self.state["profile"],
                "evidence": {
                    "path": str(evidence_path),
                    "sha256": sha(evidence_path),
                    "size_bytes": evidence_path.stat().st_size,
                },
                "owner_nonce": self.nonce,
                "after_stage_deadline_cleanup_monotonic": time.monotonic(),
                "whole_campaign_seconds": WALL_SECONDS,
                "overall_campaign_complete": False,
                "external_current_command_and_owner_closure_required": True,
                "cooperative_publication_not_hard_realtime_or_no_escape": True,
            }
            durable_stage_json(self.helper, receipt_path, receipt)
            self.helper.ensure_elapsed(began, WALL_SECONDS, "stage durable completion publication")
            ref = {
                "path": str(receipt_path),
                "sha256": sha(receipt_path),
                "size_bytes": receipt_path.stat().st_size,
                "stage": stage,
                "elapsed_campaign_seconds_after_publication": time.monotonic() - began,
            }
            self.state["completed_stage_receipts"][stage] = ref
            self.helper.write_json(self.output.parent / "state.json", self.state)
            self.helper.ensure_elapsed(
                began, WALL_SECONDS, "stage completion handoff durable state"
            )
        except BaseException as primary:
            self.state["completed_stage_receipts"].pop(stage, None)
            try:
                durable_stage_json(
                    self.helper,
                    receipt_path,
                    {
                        "schema": STAGE_SCHEMA,
                        "stage": stage,
                        "status": "failed_no_stage_qualification",
                        "error": {"type": type(primary).__name__, "message": str(primary)},
                        "overall_campaign_complete": False,
                    },
                )
            except BaseException as secondary:
                primary.add_note("Stage failure demotion also failed: " + repr(secondary))
            raise
        self.completed.append(stage)
        return ref


def run_unchanged_suites(
    predictor, identity, output, *, run_public, heaven_root, jb, complete_stage
):
    """Original public entry then official coherence API; no new scoring math."""
    output = Path(output)
    require(not (output / "coherence-cache").exists(), "fresh coherence cache required")
    stage_began = time.monotonic()
    predictor.stage = STAGES[0]
    public = run_public(predictor, heaven_root, output / "public231", identity)
    require(
        read_json(output / "public231/manifest.json")["n_attempted"] == 231,
        "all original231 public tasks required",
    )
    complete_stage(STAGES[0], public, stage_began)
    stage_began = time.monotonic()
    predictor.stage = STAGES[1]
    report = jb.evaluate(
        jb.from_callable(predictor.answers, name=identity["profile"]),
        suite="jevbench-mini",
        cache=str(output / "coherence-cache"),
        concurrency=1,
        progress=True,
    )
    report.save(output / "coherence-mini.json")
    result = report.result
    require(
        result["requests"] == 1248
        and not result.get("truncated_by_budget")
        and len(result["case_domains"]) == 240
        and len(result["planned_tests"]) == 50
        and sum(result["planned_tests"].values()) == 1200,
        "all original1248 requests/240cases/50relations/1200tests required",
    )
    mini = {
        "overall": report.overall,
        "interval": report.interval,
        "dimensions": report.dimensions,
        "errors": report.errors,
        "scores": report.scores,
        "requests": 1248,
        "cases": 240,
        "relations": 50,
        "planned_tests": 1200,
    }
    complete_stage(STAGES[1], mini, stage_began)
    return {
        "profile": identity["profile"],
        "legacy_public231": public,
        "coherence_mini": mini,
        "official_rank": None,
        "browser_score": None,
    }


def check_counts(predictor):
    require(
        predictor.started == predictor.completed == predictor.raw_returned == 1479,
        "1479 fresh complete original callbacks/native envelopes required",
    )
    require(
        set(predictor.latencies) == set(STAGES)
        and all(len(predictor.latencies[s]) == n for s, n in COUNTS.items()),
        "per-suite231/1248 callback latency population differs",
    )


def durable_success(helper, output, state, result, began):
    """A late final write cannot leave a successful return to the controller."""
    helper.ensure_elapsed(began, WALL_SECONDS, "whole campaign final authentication")
    helper.write_json(
        output / "results.json",
        {
            "status": "pending_durable_campaign_validation",
            "benchmark_scores": result,
            "latencies": state["latencies"],
            "model_identity": state["model_identity"],
            "browser_score": None,
            "latency_startup_included": False,
        },
    )
    helper.ensure_elapsed(began, WALL_SECONDS, "whole campaign durable results")
    state["elapsed_seconds"] = time.monotonic() - began
    state["status"] = "both_original_suites_complete_native_batch4_cpu_profile_only"
    helper.write_json(output / "state.json", state)
    helper.ensure_elapsed(began, WALL_SECONDS, "whole campaign durable final state")


def execute(plan_path, plan_sha, smoke_path, smoke_sha, audit_path, audit_sha, output):
    began = time.monotonic()
    began_unix = time.time()
    require(
        sys.flags.no_user_site
        and sys.dont_write_bytecode
        and not {
            "torch",
            "transformers",
            "numpy",
            "safetensors",
            "s1",
            "jevbench",
            "d3_runtime",
        }.intersection(sys.modules),
        "fresh -B -s process without imported framework/harness required",
    )
    plan = load_plan(plan_path, plan_sha)
    # Import exactly reviewed small local actor bytes; its dry module imports no model library.
    import importlib.util

    raw = (BATCH / "actor.py").read_bytes()
    require(hashlib.sha256(raw).hexdigest() == BATCH_ACTOR_SHA, "batch4 shared source differs")
    spec = importlib.util.spec_from_file_location(
        "_shared_batch4_actor_campaign", BATCH / "actor.py"
    )
    batch = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = batch
    exec(compile(raw, str(BATCH / "actor.py"), "exec"), batch.__dict__)
    helper = batch.load_helper()
    native_path = BATCH / (plan["model_key"] + "-plan.json")
    native = batch.load_plan(helper, native_path, NATIVE_PLAN_SHA[plan["model_key"]])
    check_smoke(smoke_path, smoke_sha, native)
    prerequisite = check_audit(audit_path, audit_sha, smoke_path, smoke_sha, native)
    require(sha(BATCH / "freeze.json") == BATCH_FREEZE_SHA, "selected batch4 freeze differs")
    benchmark = selected_small_json(BENCHMARK_PLAN, BENCHMARK_PLAN_SHA)
    require(
        len(benchmark["source_files"]) == 78 and len(benchmark["existing_dataset_metadata"]) == 2,
        "original78 source files and2 dataset metadata manifests required",
    )
    identity = helper.process_identity()
    limits = helper.owned_limits(identity)
    batch.validate_owner(identity, limits, helper.START_ENV)
    gate = helper.host_gate()
    output = Path(output).resolve(strict=False)
    require(
        output.is_relative_to(HERE.parent.resolve()) and not output.exists(),
        "fresh resolved output in runtime artifact tree required",
    )
    output.mkdir(parents=True, exist_ok=False)
    state = {
        "profile": plan["profile"],
        "model_key": plan["model_key"],
        "status": "authenticating_sources",
        "plan_sha256": plan_sha,
        "prerequisite": prerequisite,
        "process": identity,
        "owned_limits": limits,
        "host_startup_admission": gate,
        "model_loads": 0,
        "whole_campaign_deadline_seconds": WALL_SECONDS,
        "per_call_deadline_seconds": CALL_SECONDS,
        "per_call_deadline_basis": "official offline benchmark180s; no browser20 qualification",
        "head_field_is_provenance_only": True,
        "local_head_at_creation": plan["local_head_at_creation"],
        "benchmark_scores": None,
        "completed_stage_receipts": {},
        "campaign_started_unix": began_unix,
        "browser_score": None,
        "batch1_score_or_parity_inherited": False,
    }
    primary, predictor = None, None
    # Remaining budget includes all preflight time in the whole10000s declaration.
    remaining = WALL_SECONDS - (time.monotonic() - began)
    require(remaining > 0, "campaign preflight exhausted whole deadline")
    try:
        with DeadlineScope(helper.Deadline(remaining, output, "campaign")):
            try:
                for ref in benchmark["source_files"] + benchmark["existing_dataset_metadata"]:
                    helper.check_record(ref, full=True)
                state["before_load_owned_counters"] = batch.own_counters(identity)
                helper.write_json(output / "state.json", state)
                runtime, torch = batch.load_runtime(helper, native, output, state)
                require(state["model_loads"] == 1, "one actual publisher constructor required")
                backend = batch.NativeBatch4Backend(
                    runtime, torch, native["model"], context_limit=16384
                )
                predictor = TimedPredictor(
                    backend,
                    helper,
                    output / "native-evidence",
                    lambda: batch.own_counters(identity),
                )
                public_path = ROOT / "scripts/run_jevbench_public.py"
                public_sha = next(
                    ref["sha256"]
                    for ref in benchmark["source_files"]
                    if ref["path"] == str(public_path)
                )
                public = batch.import_verified("_batch4_public_harness", public_path, public_sha)
                coherence = Path(benchmark["coherence"]["root"])
                sys.path.insert(0, str(coherence / "src"))
                import jevbench as jb

                require(
                    Path(jb.__file__).resolve()
                    == (coherence / "src/jevbench/__init__.py").resolve(),
                    "official coherence module origin differs",
                )
                state["model_identity"] = {
                    **native["model"],
                    "profile": plan["profile"],
                    "device": "cpu",
                    "backbone_dtype": "bfloat16",
                    "readout_dtype": "float32",
                    "temperature": 1.0,
                    "batch_size": 4,
                    "context_limit": 16384,
                    "context_policy": "reject_no_truncation",
                    "permutation_average": False,
                    "constructor_count": 1,
                    "benchmark_heaven_revision": benchmark["public"]["revision"],
                    "coherence_revision": benchmark["coherence"]["revision"],
                    "source_plan_sha256": BENCHMARK_PLAN_SHA,
                }
                state["status"] = "running_original_public231_then_coherence_mini"
                helper.write_json(output / "state.json", state)

                def authenticate_stage():
                    # Additional identical full authentication; final campaign gate remains below.
                    batch.validate_runtime(runtime, torch, native["model"])
                    batch.authenticate(helper, native)
                    for ref in benchmark["source_files"] + benchmark["existing_dataset_metadata"]:
                        helper.check_record(ref, full=True)
                    for source_path, expected in selected_stage_sources:
                        require(sha(source_path) == expected, "stage source/prerequisite changed")
                    check_audit(audit_path, audit_sha, smoke_path, smoke_sha, native)
                    batch.validate_runtime(runtime, torch, native["model"])

                selected_stage_sources = [
                    (Path(plan_path), plan_sha),
                    (Path(__file__), plan["campaign_sha256"]),
                    (BATCH / "actor.py", BATCH_ACTOR_SHA),
                    (BATCH / "freeze.json", BATCH_FREEZE_SHA),
                    (native_path, NATIVE_PLAN_SHA[plan["model_key"]]),
                    (BENCHMARK_PLAN, BENCHMARK_PLAN_SHA),
                    (batch.HELPER, batch.HELPER_SHA),
                    (BATCH / "source_audit.py", batch.SOURCE_AUDIT_SHA),
                    (Path(smoke_path), smoke_sha),
                    (Path(audit_path), audit_sha),
                ]
                stage_completions = StageCompletions(
                    helper,
                    batch,
                    native,
                    benchmark,
                    identity,
                    output,
                    state,
                    predictor,
                    began,
                    began_unix,
                    authenticate_stage,
                    selected_stage_sources,
                )
                result = run_unchanged_suites(
                    predictor,
                    state["model_identity"],
                    output,
                    run_public=public.run_public,
                    heaven_root=benchmark["public"]["root"],
                    jb=jb,
                    complete_stage=stage_completions.complete,
                )
                json.dumps(result, allow_nan=False)
                check_counts(predictor)
                stage_completions.verify_all()
                batch.validate_runtime(runtime, torch, native["model"])
                # Final full20 model+manifest, selected runtime and owner source authentication.
                batch.authenticate(helper, native)
                for ref in benchmark["source_files"] + benchmark["existing_dataset_metadata"]:
                    helper.check_record(ref, full=True)
                for p, expected in [
                    (Path(plan_path), plan_sha),
                    (Path(__file__), plan["campaign_sha256"]),
                    (BATCH / "actor.py", BATCH_ACTOR_SHA),
                    (BATCH / "freeze.json", BATCH_FREEZE_SHA),
                    (native_path, NATIVE_PLAN_SHA[plan["model_key"]]),
                    (BENCHMARK_PLAN, BENCHMARK_PLAN_SHA),
                    (batch.HELPER, batch.HELPER_SHA),
                    (BATCH / "source_audit.py", batch.SOURCE_AUDIT_SHA),
                    (Path(smoke_path), smoke_sha),
                    (Path(audit_path), audit_sha),
                ]:
                    require(
                        sha(p) == expected, "selected source/prerequisite changed during campaign"
                    )
                # Revalidate every small audit ref, including root/owner authenticated closure evidence.
                check_audit(audit_path, audit_sha, smoke_path, smoke_sha, native)
                batch.validate_runtime(runtime, torch, native["model"])
                state.update(
                    {
                        "benchmark_scores": result,
                        "latencies": predictor.summary(),
                        "started_calls": predictor.started,
                        "completed_calls": predictor.completed,
                        "raw_native_returns": predictor.raw_returned,
                        "final_full_model_manifest_runtime_and78_sources_plus2_metadata_authenticated": True,
                        "after_final_auth_owned_counters": batch.own_counters(identity),
                        "cuda_initialized_final": torch.cuda.is_initialized(),
                        "torch_threads_final": {
                            "intraop": torch.get_num_threads(),
                            "interop": torch.get_num_interop_threads(),
                        },
                        "closure_scope": "Actor result only. Root must separately authenticate current campaign owner terminal/twoempty/cgroupabsence and measured peak.",
                        "startup_excluded_from_suite_latency": True,
                    }
                )
                durable_success(helper, output, state, result, began)
            except BaseException as exc:
                primary = exc
                state["status"] = "failed_or_incomplete_no_complete_score_claim"
                state["error"] = {"type": type(exc).__name__, "message": str(exc)}
                raise
            finally:
                if primary is not None:
                    if predictor is not None:
                        state.update(
                            {
                                "latencies": predictor.summary(),
                                "started_calls": predictor.started,
                                "completed_calls": predictor.completed,
                                "raw_native_returns": predictor.raw_returned,
                            }
                        )
                    state["elapsed_seconds"] = time.monotonic() - began
                    helper.finalize_state(output, state, primary)
        helper.ensure_elapsed(
            began, WALL_SECONDS, "whole campaign after finalization and deadline cleanup"
        )
    except BaseException as exc:
        if primary is None:
            state["status"] = "failed_or_incomplete_no_complete_score_claim"
            state["error"] = {"type": type(exc).__name__, "message": str(exc)}
            state["benchmark_scores"] = None
            state["elapsed_seconds"] = time.monotonic() - began
            try:
                helper.finalize_state(output, state, exc)
            except BaseException as secondary:
                exc.add_note(
                    "Post-cleanup campaign failure evidence also failed: " + repr(secondary)
                )
        raise
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--smoke-receipt", type=Path)
    parser.add_argument("--smoke-receipt-sha256")
    parser.add_argument("--smoke-audit", type=Path)
    parser.add_argument("--smoke-audit-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "dry_default_no_model_or_benchmark_execution",
                    "models": list(NATIVE_PLAN_SHA),
                    "batch_size": 4,
                    "constructor_count": 1,
                    "suites": COUNTS,
                    "fresh_calls": 1479,
                    "whole_campaign_seconds": WALL_SECONDS,
                    "per_call_seconds": CALL_SECONDS,
                    "owner_runtime_seconds": OWNER_SECONDS,
                    "requires_exact_successful_smoke_and_closed_independent_audit": True,
                    "browser20_qualification_or_batch1_scores_inherited": False,
                    "stage_completion_schema": STAGE_SCHEMA,
                    "stage_receipts_require_independent_current_root_audit": True,
                }
            )
        )
        return
    require(
        all(x is not None for x in [args.plan, args.output, args.smoke_receipt, args.smoke_audit]),
        "explicit plan/newoutput/successfulstate/independently reviewed closed smoke required",
    )
    result = execute(
        args.plan,
        args.plan_sha256,
        args.smoke_receipt,
        args.smoke_receipt_sha256,
        args.smoke_audit,
        args.smoke_audit_sha256,
        args.output,
    )
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
