"""Resettable finite-state workflows with deterministic outcomes and routed cost.

Environments are replay fixtures: no arbitrary commands, browser actions or model
self-grading. A transition reaches an externally specified goal state; saying
`complete` before that state is always a false completion.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, model_validator

from s1.contracts import DecisionRequest, StrictModel

from .adapters import execution_blockers, load_adapter
from .artifacts import write_json
from .contracts import ProfileSpec
from .planning import _configuration_blockers, case_eligibility, runtime_identity
from .responses import normalize


class Transition(StrictModel):
    next_state: str
    tool_cost_usd: float = Field(default=0, ge=0)


class State(StrictModel):
    observation: DecisionRequest
    goal_satisfied: bool = False
    transitions: dict[str, Transition] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_actions(self):
        if len(self.observation.questions) != 1 or self.observation.questions[0].type != "choice":
            raise ValueError("workflow observation requires one Choice question")
        labels = set(self.observation.questions[0].labels())
        if not {"complete", "escalate"} <= labels or set(self.transitions) - labels:
            raise ValueError("workflow needs complete/escalate and valid transition labels")
        if labels - {"complete", "escalate"} != set(self.transitions):
            raise ValueError("every executable action needs a deterministic transition")
        return self


class Episode(StrictModel):
    id: str
    group_id: str
    family: Literal["ci", "ui_document"]
    seed: int = Field(ge=0)
    initial_state: str
    states: dict[str, State]
    max_steps: int = Field(default=8, ge=1, le=1000)

    @model_validator(mode="after")
    def valid_graph(self):
        if self.initial_state not in self.states or any(
            t.next_state not in self.states
            for s in self.states.values()
            for t in s.transitions.values()
        ):
            raise ValueError("workflow transition references an unknown state")
        return self


def load_episodes(path):
    episodes = [
        Episode.model_validate_json(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    if not episodes or len({e.id for e in episodes}) != len(episodes):
        raise ValueError("episodes must be nonempty with unique IDs")
    return episodes


def _call(adapter, spec, request):
    started = time.perf_counter()
    raw = adapter.predict(request.model_copy(deep=True))
    response = normalize(request, raw, spec.capabilities.probabilities)
    q = request.questions[0]
    answer = response["answers"][q.id]
    label = answer["choice"] if answer["probabilities"] is not None else answer["label"]
    confidence = (
        max(answer["probabilities"].values()) if answer["probabilities"] is not None else None
    )
    cost = raw.get("cost_usd")
    if cost is not None and (type(cost) not in (float, int) or not np.isfinite(cost) or cost < 0):
        raise ValueError("invalid reported model cost")
    return (
        label,
        confidence,
        {
            "model_id": spec.id,
            "latency_ms": (time.perf_counter() - started) * 1000,
            "reported_cost_usd": cost,
            "ceiling_cost_usd": spec.cost_per_request_usd or 0,
            "cache_hit": raw.get("cache_hit") if type(raw.get("cache_hit")) is bool else None,
        },
    )


def run_episode(episode, policy, small, strong, specs, threshold):
    current, events, calls = episode.initial_state, [], []
    started, tool_cost = time.perf_counter(), 0.0
    success = false_completion = False
    outcome = "step_budget_exhausted"
    fallback_count = retries = 0
    for step in range(episode.max_steps):
        state = episode.states[current]
        request = state.observation
        action, confidence = None, None
        try:
            if policy == "rules_first":
                # Only public observation fields; goal_satisfied is never read here.
                public = request.state if isinstance(request.state, dict) else {}
                if public.get("tests_passed") is True or public.get("ui_confirmed") is True:
                    action = "complete"
                elif public.get("retryable") is True and "retry" in state.transitions:
                    action = "retry"
            if action is None:
                use_strong = policy in ("always_strong", "rules_first")
                adapter, spec = (strong, specs[1]) if use_strong else (small, specs[0])
                action, confidence, call = _call(adapter, spec, request)
                calls.append(call)
            if policy == "cascade" and (
                action == "escalate" or confidence is None or confidence < threshold
            ):
                fallback_count += 1
                spec = specs[1]
                action, confidence, call = _call(strong, specs[1], request)
                calls.append(call)
        except Exception as exc:
            # A failed dispatched call still consumes the conservative cost ceiling.
            calls.append(
                {
                    "model_id": spec.id,
                    "latency_ms": None,
                    "reported_cost_usd": None,
                    "ceiling_cost_usd": spec.cost_per_request_usd or 0,
                    "cache_hit": None,
                    "error_type": type(exc).__name__,
                }
            )
            outcome = "model_error"
            break
        events.append({"step": step, "state_id": current, "action": action})
        if action == "complete":
            success = state.goal_satisfied
            false_completion = not success
            outcome = "success" if success else "false_completion"
            break
        if action == "escalate":
            outcome = "unresolved_escalation"
            break
        transition = state.transitions[action]
        retries += action == "retry"
        tool_cost += transition.tool_cost_usd
        current = transition.next_state
    reported = [c["reported_cost_usd"] for c in calls]
    return {
        "episode_id": episode.id,
        "group_id": episode.group_id,
        "family": episode.family,
        "seed": episode.seed,
        "policy": policy,
        "success": success,
        "false_completion": false_completion,
        "outcome": outcome,
        "events": events,
        "calls": calls,
        "fallback_count": fallback_count,
        "retry_steps": retries,
        "latency_ms": (time.perf_counter() - started) * 1000,
        "tool_replay_cost_usd": tool_cost,
        "ceiling_total_cost_usd": tool_cost + sum(c["ceiling_cost_usd"] for c in calls),
        "reported_total_cost_usd": tool_cost + sum(reported)
        if all(c is not None for c in reported)
        else None,
        "model_switches": sum(a["model_id"] != b["model_id"] for a, b in zip(calls, calls[1:])),
        "cache_loss_tokens": None,
    }


def summarize_episodes(rows):
    result = {}
    for policy in sorted({r["policy"] for r in rows}):
        selected = [r for r in rows if r["policy"] == policy]
        success = sum(r["success"] for r in selected)
        cost = sum(r["ceiling_total_cost_usd"] for r in selected)
        reported = [r["reported_total_cost_usd"] for r in selected]
        result[policy] = {
            "episodes": len(selected),
            "successful_episodes": success,
            "completion_rate": success / len(selected),
            "false_completions": sum(r["false_completion"] for r in selected),
            "fallback_episode_rate": sum(r["fallback_count"] > 0 for r in selected) / len(selected),
            "retry_steps": sum(r["retry_steps"] for r in selected),
            "p95_task_latency_ms": float(np.percentile([r["latency_ms"] for r in selected], 95)),
            "ceiling_cost_per_success_usd": cost / success if success else None,
            "reported_cost_per_success_usd": sum(reported) / success
            if success and all(c is not None for c in reported)
            else None,
            "reported_cost_episodes": sum(c is not None for c in reported),
        }
    return result


def compare_policies(rows, baseline="always_strong", *, seed=0):
    from .statistics import cluster_interval

    reference = {r["episode_id"]: r for r in rows if r["policy"] == baseline}
    comparisons = {}
    for policy in sorted({r["policy"] for r in rows} - {baseline}):
        target = {r["episode_id"]: r for r in rows if r["policy"] == policy}
        if target.keys() != reference.keys():
            raise ValueError("workflow policies need matched episodes")
        keys = sorted(reference)
        groups = [reference[k]["group_id"] for k in keys]
        comparisons[policy] = {
            "completion_delta": cluster_interval(
                [int(target[k]["success"]) - int(reference[k]["success"]) for k in keys],
                groups,
                seed=seed,
            ),
            "ceiling_episode_cost_delta": cluster_interval(
                [
                    target[k]["ceiling_total_cost_usd"] - reference[k]["ceiling_total_cost_usd"]
                    for k in keys
                ],
                groups,
                seed=seed,
            ),
        }
    return {"baseline": baseline, "comparisons": comparisons}


def execute_workflows(
    registry,
    episodes_path,
    small_id,
    strong_id,
    *,
    output,
    lockfile,
    environment=None,
    enable_models=(),
    threshold=0.8,
    max_calls=1000,
    max_cost_usd=0.0,
    adapter_factory=load_adapter,
):
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0,1]")
    if (
        type(max_calls) is not int
        or max_calls < 0
        or not np.isfinite(max_cost_usd)
        or max_cost_usd < 0
    ):
        raise ValueError("workflow budgets must be finite and nonnegative")
    episodes = load_episodes(episodes_path)
    specs = [registry.models[key] for key in (small_id, strong_id)]
    profile = ProfileSpec(
        id="workflow", suite="workflow", models=tuple(dict.fromkeys([small_id, strong_id]))
    )
    environment = dict(environment or {})
    reasons, service = [], {}
    for spec in specs:
        if not (spec.enabled or spec.id in enable_models):
            reasons.append(f"disabled:{spec.id}")
        blockers, identity = _configuration_blockers(spec, profile, environment)
        reasons.extend(blockers + execution_blockers(spec, profile))
        service[spec.id] = identity
        if spec.adapter != "uniform" and spec.cost_per_request_usd is None:
            reasons.append("unknown_cost_ceiling")
        for episode in episodes:
            for state in episode.states.values():
                case = type("CaseView", (), {"id": episode.id, "request": state.observation})()
                if case_eligibility(spec, case)["eligibility"] != "eligible":
                    reasons.append(f"unsupported_workflow:{spec.id}")
    # Four policies: small + strong + rules(worker worst case) + cascade(two calls).
    steps = sum(e.max_steps for e in episodes)
    reserve_calls = steps * 5
    reserve_cost = steps * (
        2 * (specs[0].cost_per_request_usd or 0) + 3 * (specs[1].cost_per_request_usd or 0)
    )
    reserve_cost += (
        sum(
            e.max_steps
            * max(
                (t.tool_cost_usd for s in e.states.values() for t in s.transitions.values()),
                default=0,
            )
            for e in episodes
        )
        * 4
    )
    if reserve_calls > max_calls or reserve_cost > max_cost_usd:
        reasons.append("workflow_budget_exceeded")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 2,
        "kind": "workflow_replay",
        "status": "not_run" if reasons else "running",
        "models": [s.model_dump(mode="json") for s in specs],
        "service": service,
        "runtime": runtime_identity(lockfile),
        "source_sha256": hashlib.sha256(Path(episodes_path).read_bytes()).hexdigest(),
        "threshold": threshold,
        "reserved_calls": reserve_calls,
        "reserved_cost_usd": reserve_cost,
        "reasons": sorted(set(reasons)),
        "provider_cache_policy": "unknown",
        "result_cache": False,
        "cost_policy": "reported provider costs when available; otherwise explicit ceilings; replay tool costs",
    }
    write_json(output / "workflow_manifest.json", manifest)
    rows, adapters = [], {}
    try:
        if not reasons:
            for spec in specs:
                if spec.id not in adapters:
                    adapters[spec.id] = adapter_factory(spec, environment)
            with (output / "episodes.jsonl").open("x", encoding="utf-8") as stream:
                for policy in ("always_small", "always_strong", "rules_first", "cascade"):
                    for episode in episodes:
                        row = run_episode(
                            episode,
                            policy,
                            adapters[small_id],
                            adapters[strong_id],
                            specs,
                            threshold,
                        )
                        rows.append(row)
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                        stream.flush()
            manifest["status"] = "completed"
    except BaseException:
        manifest["status"] = "interrupted"
        raise
    finally:
        for adapter in adapters.values():
            try:
                adapter.close()
            except Exception as exc:
                manifest.setdefault("cleanup_errors", []).append(type(exc).__name__)
                if manifest["status"] == "completed":
                    manifest["status"] = "completed_with_errors"
        write_json(output / "workflow_manifest.json", manifest)
        write_json(output / "workflow_summary.json", summarize_episodes(rows))
        if rows and manifest["status"] == "completed":
            write_json(output / "workflow_comparison.json", compare_policies(rows))
    return {
        "status": manifest["status"],
        "reasons": manifest["reasons"],
        "policies": summarize_episodes(rows),
    }
