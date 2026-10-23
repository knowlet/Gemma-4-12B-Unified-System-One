"""Gold-free callable bridge for the two unchanged JevBench harnesses.

Consumes ONE already loaded publisher D3 runtime from actor.load_publisher_cpu.
This module never constructs a model, imports Torch, or starts evaluation.
"""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path

PROFILE = "decision3-lite-gx10-publisher-cpu-batch1-20261023-v1"


class CPUProfileAbort(BaseException):
    """Resource/identity failures escape upstream per-case Exception handlers."""


class PublisherCPUAdapter:
    def __init__(self, runtime, torch, evidence_dir, *, check_runtime, context_limit=16384):
        if context_limit != 16384:
            raise ValueError("separately declared benchmark context cap differs")
        self.runtime, self.torch = runtime, torch
        self.check_runtime = check_runtime
        self.context_limit = context_limit
        self.output = Path(evidence_dir)
        self.output.mkdir(exist_ok=False)
        self.calls, self.stage = 0, "not_started"
        self.verify()

    def verify(self):
        try:
            self.check_runtime(self.runtime, self.torch)
        except Exception as exc:
            raise CPUProfileAbort("CPU runtime identity/numerics changed") from exc

    def record(self, value):
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n"
        with (self.output / "native_calls.jsonl").open("a") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())

    def __call__(self, state, questions):
        self.verify()
        request = copy.deepcopy({"state": state, "questions": questions})
        index = self.calls
        self.calls += 1
        started = time.monotonic()
        # Only the harness' model input reaches this callable: no tasks, labels,
        # gold records, scorer results, or previous cache responses are accepted.
        self.record({"event": "started", "index": index, "stage": self.stage, "request": request})
        try:
            prepared = self.runtime.prepare(**copy.deepcopy(request))
            lengths = {key: len(tokens) for key, tokens in prepared.sequences.items()}
            lengths.update(prepared.lengths)
            if (
                prepared.errors
                or list(prepared.keys) != list(questions)
                or set(lengths) != set(questions)
            ):
                raise ValueError("publisher rejected complete benchmark request")
            if any(type(n) is not int or not 0 < n <= self.context_limit for n in lengths.values()):
                raise ValueError("benchmark context cap exceeded; nothing truncated")
            self.record(
                {"event": "prepared", "index": index, "stage": self.stage, "lengths": lengths}
            )
            raw = self.runtime.system_one(**copy.deepcopy(request))
            # Preserve the complete original envelope before checking it. Both
            # scorers continue to validate their ordinary answer wire formats.
            self.record(
                {
                    "event": "returned",
                    "index": index,
                    "stage": self.stage,
                    "seconds": time.monotonic() - started,
                    "response": raw,
                }
            )
            self.verify()
            if (
                not isinstance(raw, dict)
                or raw.get("model") != "d3-lite"
                or set(raw.get("answers", {})) != set(questions)
            ):
                raise ValueError("native publisher identity/question inventory differs")
            if raw.get("usage") != {"input_tokens": sum(lengths.values()), "output_tokens": 0}:
                raise ValueError("native publisher token usage differs")
            # Catch nonfinite response values without changing them or fitting
            # calibration. Publisher temperature and probabilities pass intact.
            json.dumps(raw, allow_nan=False)
            return raw
        except BaseException as exc:
            try:
                self.record(
                    {
                        "event": "failed",
                        "index": index,
                        "stage": self.stage,
                        "seconds": time.monotonic() - started,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
                )
            except BaseException as secondary:
                exc.add_note("Lite adapter failure evidence write failed: " + repr(secondary))
            raise

    def answers(self, state, questions):
        return self(state, questions)["answers"]


def run_unchanged_suites(adapter, identity, output, *, run_public, heaven_root, jb):
    """Sequential suites using one runtime; dependencies are verified by caller.

    No model constructor, data copy, case selection, scoring replacement or
    hidden retry occurs here. A future selected score worker supplies the exact
    source manifests and original owner with its separately reviewed deadline.
    """
    output = Path(output)
    if (output / "coherence-cache").exists():
        raise CPUProfileAbort("coherence cache must be fresh; no inherited predictions")
    adapter.stage = "legacy-public231"
    public = run_public(adapter, heaven_root, output / "public231", identity)
    manifest = json.loads((output / "public231/manifest.json").read_text())
    if manifest["n_attempted"] != 231:
        raise CPUProfileAbort("complete public231 population required")
    adapter.stage = "jevbench-mini"
    report = jb.evaluate(
        jb.from_callable(adapter.answers, name=PROFILE),
        suite="jevbench-mini",
        cache=str(output / "coherence-cache"),
        concurrency=1,
        progress=True,
    )
    report.save(output / "coherence-mini.json")
    result = report.result
    if (
        result["requests"] != 1248
        or result.get("truncated_by_budget")
        or len(result["case_domains"]) != 240
        or len(result["planned_tests"]) != 50
        or sum(result["planned_tests"].values()) != 1200
    ):
        raise CPUProfileAbort("complete original coherence mini population required")
    return {
        "profile": PROFILE,
        "legacy_public231": public,
        "coherence_mini": {
            "overall": report.overall,
            "interval": report.interval,
            "dimensions": report.dimensions,
            "errors": report.errors,
            "scores": report.scores,
            "requests": result["requests"],
            "cases": 240,
            "relations": 50,
        },
        "official_rank": None,
        "browser_score": None,
    }
