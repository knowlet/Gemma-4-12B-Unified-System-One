"""Complete pinned public JevBench screening with the publisher's scoring code.

``run_public(predict, tasks_root, output, identity)`` calls a loaded model's
``predict(state, questions)`` once per original task. It accepts named answers
or the ordinary ``{"answers": named_answers}`` envelope. Loading, accelerator
synchronization, and hardware provenance belong to the caller. This module
does not contact endpoints, fit calibration, submit scores, or compute a rank.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

CONFIG = Path(__file__).resolve().parents[1] / "configs/benchmarks/jevbench-public.json"


def _bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()


def _hash(raw):
    return hashlib.sha256(raw).hexdigest()


def _write(path, value):
    path.write_bytes(_bytes(value) + b"\n")


def _verified_file(root, relative, expected):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("official source path escapes repository")
    raw = path.read_bytes()
    if _hash(raw) != expected:
        raise ValueError(f"official source hash mismatch: {relative}")
    return path


def _module(name, path, expected, package=False):
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if package else None
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        raw = path.read_bytes()
        if _hash(raw) != expected:
            raise ValueError("official source changed during import")
        # Execute the verified source bytes, never an unchecked cached .pyc.
        exec(compile(raw, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _load_official(root, config):
    """Verified isolated namespace: never import another installed jevbench."""
    paths = {
        relative: _verified_file(root, relative, digest)
        for relative, digest in config["source_files"].items()
    }
    alias = "_s1_public_jevbench_" + _hash(str(root).encode())[:16]
    # Reload verified bytes each time; a previous import must not bypass hashes.
    for name in tuple(sys.modules):
        if name == alias or name.startswith(alias + "."):
            del sys.modules[name]
    _module(
        alias,
        paths["jevbench/__init__.py"],
        config["source_files"]["jevbench/__init__.py"],
        package=True,
    )
    loaded = {}
    for name in ("tasks", "scoring", "metrics", "summarize"):
        relative = f"jevbench/{name}.py"
        loaded[name] = _module(
            alias + "." + name, paths[relative], config["source_files"][relative]
        )
    # Do not execute adapters/__init__.py: it imports unrelated optional models.
    adapters = types.ModuleType(alias + ".adapters")
    adapters.__path__ = [str(root / "jevbench/adapters")]
    sys.modules[adapters.__name__] = adapters
    loaded["base"] = _module(
        alias + ".adapters.base",
        paths["jevbench/adapters/base.py"],
        config["source_files"]["jevbench/adapters/base.py"],
    )
    return SimpleNamespace(**loaded)


def _repo_root(tasks_root):
    root = Path(tasks_root).resolve()
    for candidate in (root, root.parent, root.parent.parent):
        if (candidate / "jevbench/tasks.py").is_file():
            return candidate
    raise ValueError(
        "tasks_root must point to the pinned official repository or its public directory"
    )


def load_public(tasks_root, *, config=None):
    """Verify every source byte, ID and original order before model execution."""
    config = json.loads(CONFIG.read_text()) if config is None else config
    root = _repo_root(tasks_root)
    official = _load_official(root, config)
    tasks = []
    for entry in config["task_files"]:
        path = _verified_file(root, entry["path"], entry["sha256"])
        current = official.tasks.load_jsonl(str(path))
        if len(current) != entry["count"]:
            raise ValueError("official task file count differs from its pin")
        tasks.extend(current)
    ids = [task.id for task in tasks]
    if len(tasks) != config["expected_tasks"] or len(set(ids)) != len(ids):
        raise ValueError("incomplete public population or duplicate task IDs")
    if any(task.split != "public" for task in tasks):
        raise ValueError("private tasks cannot enter public screening")
    if official.tasks.dataset_hash(tasks) != config["canonical_dataset_sha256"]:
        raise ValueError("canonical official dataset hash differs from its pin")
    ordered = json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode()
    if _hash(ordered) != config["ordered_ids_sha256"]:
        raise ValueError("official task ordering differs from its pin")
    return tasks, official, config


def _native_probs(response, task):
    """Publisher TypeSafe adapter mapping; scoring stays entirely upstream."""
    if not isinstance(response, dict):
        raise ValueError("native answers must be a dict")
    answers = response.get("answers", response)
    if not isinstance(answers, dict) or set(answers) != {"decision"}:
        raise ValueError("native response must contain exactly the decision answer")
    answer = answers["decision"]
    kind = task.question["type"]
    if not isinstance(answer, dict) or answer.get("type") != kind:
        raise ValueError("native answer type mismatch")
    if kind == "noul":
        p = answer.get("noul")
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            raise ValueError("noul must be numeric")
        p = float(p)
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("noul out of range")
        return {"yes": p, "no": 1.0 - p}
    if kind == "choice" and answer.get("choice") not in task.labels:
        raise ValueError("invalid native choice")
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict):
        raise ValueError("native answer missing probabilities")
    return probabilities


def _raw_response(response):
    try:
        _bytes(response)
        return response, "json"
    except (TypeError, ValueError):
        # Preserve malformed/non-finite output without writing invalid JSON.
        return repr(response), "python_repr_not_json"


def _reported_usage(response):
    """Preserve reported usage; absence never becomes zero tokens or free cost."""
    if not isinstance(response, dict) or response.get("usage") is None:
        return None, "not_reported"
    usage = response["usage"]
    if not isinstance(usage, dict):
        return None, "invalid_not_used"
    try:
        _bytes(usage)
    except (TypeError, ValueError):
        return None, "invalid_not_used"
    # Provider names differ, but ordinary token counts must be nonnegative
    # integers. This is a schema check, not proof of provider billing.
    for key in (
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ):
        if key in usage and (type(usage[key]) is not int or usage[key] < 0):
            return None, "invalid_not_used"
    return copy.deepcopy(usage), "reported_unverified"


def run_public(predict, tasks_root, output, identity):
    """Run all pinned public tasks serially; return a durable research summary."""
    if not isinstance(identity, dict) or not all(
        isinstance(identity.get(key), str) and identity[key] for key in ("model_id", "revision")
    ):
        raise ValueError("identity requires explicit model_id and revision")
    tasks, official, config = load_public(tasks_root)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw_dir = output / "raw"
    raw_dir.mkdir()
    manifest = {
        "schema_version": 1,
        "status": "running",
        "dataset": config,
        "ordered_task_ids": [task.id for task in tasks],
        "identity": copy.deepcopy(identity),
        "started_at_unix": time.time(),
        "requests_per_task": 1,
        "retries": 0,
        "warmup": 0,
        "calibration_fitting": False,
        "latency_scope": "loaded-model predict wall time; caller owns accelerator synchronization",
        "cost_usd": None,
        "cost_basis": "unknown_local_compute_not_free",
        "official_composite": None,
        "official_rank": None,
        "leaderboard_status": "not_submitted",
        "sealed_evaluation": "not_run",
        "public_exposure": config["public_exposure"],
    }
    _write(output / "manifest.json", manifest)
    records = []
    with (output / "records.jsonl").open("x", encoding="utf-8") as stream:
        for index, task in enumerate(tasks):
            request = {
                "state": task.state,
                "questions": {"decision": official.base.build_question(task)},
            }
            frozen_request = copy.deepcopy(request)
            response, probs, error = None, None, None
            ok = False
            started = time.perf_counter()
            try:
                response = predict(
                    copy.deepcopy(request["state"]), copy.deepcopy(request["questions"])
                )
                probs = _native_probs(response, task)
                ok = True
            except Exception as exc:
                error = {"type": type(exc).__name__, "message": str(exc)[:500]}
            elapsed = time.perf_counter() - started
            scored = {
                "valid": False,
                "strict_valid": False,
                "renormalized": False,
                "correct": False,
                "predicted": None,
            }
            if ok:
                try:
                    scored = official.scoring.score_task(probs, task)
                except Exception as exc:
                    # An upstream scorer can itself fail on malformed output
                    # (e.g. sorting mixed label-key types). Preserve the raw
                    # answer and the failed item; never drop the remaining set.
                    ok = False
                    error = {
                        "phase": "official_scoring",
                        "type": type(exc).__name__,
                        "message": str(exc)[:500],
                    }
                    scored["error"] = f"official_scorer_exception: {type(exc).__name__}"
            returned, encoding = _raw_response(response)
            raw = {
                "request": frozen_request,
                "response": returned,
                "response_encoding": encoding,
                "error": error,
            }
            raw_bytes = _bytes(raw) + b"\n"
            raw_name = f"{index:04d}-{_hash(task.id.encode())}.json"
            (raw_dir / raw_name).write_bytes(raw_bytes)
            # A malformed raw map may contain NaN. Preserve its representation
            # in raw evidence, but never manufacture a valid scored distribution.
            returned_probs, probs_encoding = _raw_response(probs)
            usage, usage_status = _reported_usage(response)
            record = {
                "task_id": task.id,
                "family": task.family,
                "split": task.split,
                "group": task.group,
                "task_index": index,
                "ts": time.time(),
                "status": "ok" if ok else "failed",
                "ok": ok,
                "valid": scored["valid"],
                "correct": scored["correct"],
                "predicted": scored.get("predicted"),
                "ordinal_ev": scored.get("ordinal_ev"),
                "probs": scored.get("probs"),
                "probs_as_returned": returned_probs,
                "probs_as_returned_encoding": probs_encoding,
                "strict_valid": scored.get("strict_valid", False),
                "renormalized": scored.get("renormalized", False),
                "probs_source": "native",
                "model": identity["model_id"],
                "error": error,
                "schema_error": scored.get("error"),
                "status_code": None,
                "latency_s": elapsed,
                "usage": usage,
                "usage_status": usage_status,
                "cost_usd": None,
                "cost_basis": "unknown_local_compute_not_free",
                "request_sha256": _hash(_bytes(frozen_request)),
                "raw_path": "raw/" + raw_name,
                "raw_sha256": _hash(raw_bytes),
                "official_task_outcome": scored,
            }
            stream.write(_bytes(record).decode() + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            records.append(record)
            if (index + 1) % 10 == 0 or index + 1 == len(tasks):
                print(
                    f"JevBench public {identity['model_id']}: {index + 1}/{len(tasks)}", flush=True
                )
    summary = official.summarize.summarize(tasks, records)
    summary.update(
        dataset_label=config["dataset_label"],
        identity=copy.deepcopy(identity),
        source_revision=config["source_revision"],
        public_exposure=config["public_exposure"],
        official_composite=None,
        official_rank=None,
        leaderboard_status="not_submitted",
        sealed_evaluation="not_run",
    )
    _write(output / "summary.json", summary)
    manifest.update(
        status="completed"
        if all(record["ok"] and record["valid"] for record in records)
        else "completed_with_errors",
        finished_at_unix=time.time(),
        n_attempted=len(records),
        n_valid=summary["n_valid"],
        artifact_sha256={
            str(path.relative_to(output)): _hash(path.read_bytes())
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        },
    )
    _write(output / "manifest.json", manifest)
    return summary
