"""Offline integrity audit of pinned public screening and coherence raw answers.

This tool never runs models, changes prior evidence, or produces a leaderboard
rank. Accuracy intervals resample paired source groups. Cache timing uses the
predeclared interleaved sample, not the two sequential public-run timings.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import median

from run_jevbench_public import _bytes, _hash, _native_probs, _reported_usage, load_public

MODEL_NAMES = ("s1-bf16", "onejev-4b", "decider-2b")
SAMPLE_INDICES = (0, 17, 72, 103, 120, 151, 193, 230)
OUTCOME_KEYS = (
    "valid",
    "correct",
    "predicted",
    "ordinal_ev",
    "probs",
    "strict_valid",
    "renormalized",
)
METRIC_TOLERANCE = 1e-12
COHERENCE_REVISION = "e18733694623aa93058e279c3534f8b8e2edefa3"
COHERENCE_REPOSITORY = "https://github.com/JevBench/jevbench"
COHERENCE_SETTINGS = {"tolerance": 0.05, "min_n": 5, "bootstrap": 1000, "min_coverage": 0.95}
COHERENCE_RESULT_KEYS = (
    "jevbench_version",
    "backend",
    "seed",
    "repeats",
    "trials",
    "noise_multiplier",
    "noise_floor",
    "relations",
    "cases",
    "planned_tests",
    "truncated_by_budget",
    "backend_down",
    "bases",
    "outcomes",
    "errors",
    "suite",
    "case_domains",
    "selection",
)

# A separate interpreter avoids colliding with BenchmarkHeaven's jevbench package.
# Only the verified source copy and pre-existing raw answer bytes enter the replay.
COHERENCE_REPLAY = r"""
import hashlib, json, socket, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import jevbench as jb
def deny_network(*args, **kwargs):
    raise RuntimeError("offline coherence replay cannot use network")
socket.create_connection = deny_network
data = json.loads(Path(sys.argv[2]).read_text())
used, requests, missing = set(), [], set()
def answer(state, questions):
    blob = json.dumps({"b": data["backend"], "s": state, "q": questions, "r": 0},
                      ensure_ascii=False, separators=(",", ":"))
    filename = hashlib.sha256(blob.encode()).hexdigest() + ".json"
    requests.append(filename)
    if filename not in data["answers"]:
        missing.add(filename)
        raise RuntimeError("missing original cached answer: " + filename)
    used.add(filename)
    return json.loads(data["answers"][filename])
settings = data["settings"]
report = jb.evaluate(jb.from_callable(answer, name=data["backend"]), suite="jevbench-mini",
                     cache=":memory:", concurrency=1, progress=False,
                     tolerance=settings["tolerance"], min_n=settings["min_n"],
                     bootstrap=settings["bootstrap"])
print(json.dumps({"scores": report.scores, "result": report.result,
                  "used": sorted(used), "requests": requests, "missing": sorted(missing)},
                 ensure_ascii=False, allow_nan=False))
"""


class EvidenceError(ValueError):
    pass


def read_json(path):
    return json.loads(Path(path).read_text())


def _require(condition, message):
    if not condition:
        raise EvidenceError(message)


def _file(root, name):
    path = (root / name).resolve()
    _require(path.is_relative_to(root), "artifact path escapes evidence directory")
    _require(path.is_file(), f"missing artifact: {name}")
    return path


def _finite(value, *, positive=False):
    return (
        type(value) in (int, float)
        and math.isfinite(value)
        and value >= 0
        and (not positive or value > 0)
    )


def _metric_equal(actual, expected):
    """Portable float reductions only; populations and probabilities stay exact."""
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and set(actual) == set(expected)
            and all(_metric_equal(actual[key], value) for key, value in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_metric_equal(a, b) for a, b in zip(actual, expected))
        )
    if type(expected) is float:
        return (
            type(actual) in (int, float)
            and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=METRIC_TOLERANCE, abs_tol=METRIC_TOLERANCE)
        )
    return type(actual) is type(expected) and actual == expected


def audit_public(directory, tasks, official, config):
    """Recompute from source-pinned tasks and hashed raw answers, not headers."""
    directory = Path(directory).resolve()
    manifest = read_json(_file(directory, "manifest.json"))
    summary = read_json(_file(directory, "summary.json"))
    _require(
        manifest.get("status") in ("completed", "completed_with_errors"), "public run not finished"
    )
    _require(manifest.get("dataset") == config, "public source/dataset pins differ")
    ids = [task.id for task in tasks]
    _require(manifest.get("ordered_task_ids") == ids, "planned public task order differs")
    declared = manifest.get("artifact_sha256")
    _require(isinstance(declared, dict), "public artifact hashes missing")
    actual = {
        str(p.relative_to(directory))
        for p in directory.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    _require(set(declared) == actual, "declared artifact set differs from actual files")
    _require(
        {"records.jsonl", "summary.json"} <= set(declared), "public records/summary hashes missing"
    )
    for name, expected in declared.items():
        _require(
            _hash(_file(directory, name).read_bytes()) == expected,
            f"artifact hash mismatch: {name}",
        )
    records = [
        json.loads(line)
        for line in _file(directory, "records.jsonl").read_text().splitlines()
        if line.strip()
    ]
    _require(len(records) == len(tasks), "full public population not recorded")
    _require(
        [record.get("task_id") for record in records] == ids,
        "record IDs/order differ from official tasks",
    )
    identity = manifest.get("identity")
    _require(
        isinstance(identity, dict) and identity.get("model_id") and identity.get("revision"),
        "model identity incomplete",
    )
    _require(summary.get("identity") == identity, "summary/manifest model identities differ")
    raw_answers = []
    for index, (task, record) in enumerate(zip(tasks, records)):
        for key, expected in (
            ("task_index", index),
            ("family", task.family),
            ("split", task.split),
            ("group", task.group),
            ("model", identity["model_id"]),
        ):
            _require(record.get(key) == expected, f"record {key} differs: {task.id}")
        _require(_finite(record.get("latency_s")), f"invalid latency: {task.id}")
        request = {
            "state": task.state,
            "questions": {"decision": official.base.build_question(task)},
        }
        _require(
            record.get("request_sha256") == _hash(_bytes(request)),
            f"request fingerprint differs: {task.id}",
        )
        raw_name = record.get("raw_path")
        _require(
            isinstance(raw_name, str) and raw_name in declared, f"raw artifact unlisted: {task.id}"
        )
        raw_path = _file(directory, raw_name)
        _require(
            _hash(raw_path.read_bytes()) == record.get("raw_sha256"), f"raw hash differs: {task.id}"
        )
        raw = read_json(raw_path)
        _require(raw.get("request") == request, f"raw request differs: {task.id}")
        _require(raw.get("error") == record.get("error"), f"raw/record errors differ: {task.id}")
        returned = raw.get("response")
        if "usage_status" in record:
            usage, usage_status = _reported_usage(returned)
            _require(
                record.get("usage") == usage and record["usage_status"] == usage_status,
                f"reported usage differs from raw response: {task.id}",
            )
        raw_answers.append(
            returned.get("answers", returned) if isinstance(returned, dict) else None
        )
        if record.get("ok") is True:
            _require(
                raw.get("response_encoding") == "json", f"successful answer not JSON: {task.id}"
            )
            try:
                probs = _native_probs(returned, task)
                scored = official.scoring.score_task(probs, task)
            except Exception as exc:
                raise EvidenceError(f"successful record cannot be scored: {task.id}") from exc
            _require(
                record.get("probs_as_returned") == probs, f"raw probabilities differ: {task.id}"
            )
            _require(
                record.get("status") == "ok" and record.get("error") is None,
                f"success/error mismatch: {task.id}",
            )
        else:
            _require(
                record.get("ok") is False and record.get("status") == "failed",
                f"unattempted/ambiguous record: {task.id}",
            )
            _require(bool(record.get("error")), f"failed record lacks error: {task.id}")
            scored = {
                "valid": False,
                "strict_valid": False,
                "renormalized": False,
                "correct": False,
                "predicted": None,
            }
        for key in OUTCOME_KEYS:
            _require(
                record.get(key) == scored.get(key), f"official outcome {key} differs: {task.id}"
            )
        outcome = record.get("official_task_outcome")
        _require(isinstance(outcome, dict), f"task outcome missing: {task.id}")
        for key in OUTCOME_KEYS:
            _require(
                outcome.get(key) == scored.get(key), f"saved task outcome {key} differs: {task.id}"
            )
        _require(
            record.get("cost_usd") is None, "this local campaign has no measured per-decision cost"
        )
    recomputed = official.summarize.summarize(tasks, records)
    for key, expected in recomputed.items():
        _require(
            _metric_equal(summary.get(key), expected), f"official summary field differs: {key}"
        )
    _require(
        manifest.get("n_attempted") == len(tasks)
        and manifest.get("n_valid") == recomputed["n_valid"],
        "manifest population counters differ",
    )
    _require(
        summary.get("official_rank") is None and summary.get("official_composite") is None,
        "public screening cannot carry official rank/composite",
    )
    _require(
        summary.get("sealed_evaluation") == "not_run"
        and summary.get("leaderboard_status") == "not_submitted",
        "public/sealed/submission status differs",
    )
    kinds = {}
    for kind in sorted({task.question["type"] for task in tasks}):
        subset = [task for task in tasks if task.question["type"] == kind]
        kinds[kind] = official.summarize.metric(subset, records)
    return {
        "manifest": manifest,
        "summary": summary,
        "records": records,
        "raw_answers": raw_answers,
        "by_type": kinds,
        "verified_files": len(declared),
        "manifest_sha256": _hash((directory / "manifest.json").read_bytes()),
    }


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] * (hi - position) + values[hi] * (position - lo) if hi != lo else values[lo]


def _clusters(tasks):
    result = defaultdict(list)
    for index, task in enumerate(tasks):
        result["group:" + task.group if task.group else "task:" + task.id].append(index)
    return list(result.values())


def paired_accuracy(left, right, tasks, *, samples=10000, seed=20261008):
    """Candidate minus reference, keeping all decisions in sampled groups."""
    _require(
        [r["task_id"] for r in left] == [t.id for t in tasks] == [r["task_id"] for r in right],
        "paired task identities differ",
    )
    _require(
        all(a["request_sha256"] == b["request_sha256"] for a, b in zip(left, right)),
        "paired request fingerprints differ",
    )
    scorable = {
        i
        for i, task in enumerate(tasks)
        if task.expected is not None and not task.provenance.get("exclude_reason")
    }
    _require(bool(scorable), "no scorable paired tasks")
    delta = [int(bool(b["correct"])) - int(bool(a["correct"])) for a, b in zip(left, right)]
    groups = [[i for i in group if i in scorable] for group in _clusters(tasks)]
    groups = [group for group in groups if group]
    rng, estimates = random.Random(seed), []
    for _ in range(samples):
        drawn = [i for _ in groups for i in rng.choice(groups)]
        estimates.append(sum(delta[i] for i in drawn) / len(drawn))
    return {
        "n_paired": len(scorable),
        "source_groups": len(groups),
        "delta_accuracy": sum(delta[i] for i in scorable) / len(scorable),
        "ci95": [percentile(estimates, 0.025), percentile(estimates, 0.975)],
        "candidate_only_correct": sum(delta[i] == 1 for i in scorable),
        "reference_only_correct": sum(delta[i] == -1 for i in scorable),
        "bootstrap_samples": samples,
        "seed": seed,
        "scope": "paired public decisions; source-group bootstrap; descriptive, no multiple-comparison adjustment",
    }


def cache_equivalence(baseline, cached):
    a, b = baseline["records"], cached["records"]
    _require(
        [r["task_id"] for r in a] == [r["task_id"] for r in b], "cache task population differs"
    )
    _require(
        all(x["request_sha256"] == y["request_sha256"] for x, y in zip(a, b)),
        "cache requests differ",
    )
    baseline_identity = dict(baseline["manifest"]["identity"])
    cached_identity = dict(cached["manifest"]["identity"])
    _require(
        baseline_identity.pop("candidate_cache", False) is False
        and cached_identity.pop("candidate_cache", None) is True
        and cached_identity == baseline_identity,
        "cache model/runtime identity differs",
    )
    complete = all(x["valid"] and y["valid"] for x, y in zip(a, b))
    common = [(x, y) for x, y in zip(a, b) if x["valid"] and y["valid"]]
    probabilities_equal = complete and all(x["probs"] == y["probs"] for x, y in common)
    labels_equal = complete and all(x["predicted"] == y["predicted"] for x, y in common)
    delta = max(
        (abs(x["probs"][label] - y["probs"][label]) for x, y in common for label in x["probs"]),
        default=None,
    )
    return {
        "status": "passed"
        if probabilities_equal and labels_equal
        else "failed"
        if complete
        else "inconclusive",
        "expected": len(a),
        "common_valid": len(common),
        "exact_probabilities": probabilities_equal,
        "exact_labels": labels_equal,
        "max_probability_delta": delta,
    }


def cache_timing(path, baseline, tasks, *, samples=10000, seed=20261008):
    rows = read_json(path)
    selected = [tasks[index] for index in SAMPLE_INDICES]
    _require(
        isinstance(rows, list) and len(rows) == 96,
        "cache timing requires all 96 predeclared ABBA samples",
    )
    by_task, task_stats = {}, []
    expected_modes = [False] * 3 + [True] * 6 + [False] * 3
    for index, task in enumerate(selected):
        current = rows[index * 12 : (index + 1) * 12]
        _require(
            all(row.get("task_id") == task.id for row in current), "cache sample task order differs"
        )
        _require(
            [row.get("cached") for row in current] == expected_modes,
            "cache sample ABBA order differs",
        )
        _require(
            [row.get("repeat") for row in current] == [0, 1, 2] * 4,
            "cache sample repetition schedule differs",
        )
        _require(
            all(_finite(row.get("seconds"), positive=True) for row in current),
            "cache sample timing invalid",
        )
        answers = baseline["raw_answers"][SAMPLE_INDICES[index]]
        _require(
            answers is not None and all(row.get("answers") == answers for row in current),
            "cache sample outputs differ from baseline",
        )
        uncached = median(row["seconds"] for row in current if row["cached"] is False)
        cached = median(row["seconds"] for row in current if row["cached"] is True)
        by_task[task.id] = cached / uncached
        task_stats.append(
            {
                "task_id": task.id,
                "uncached_median_s": uncached,
                "cached_median_s": cached,
                "cached_over_uncached": cached / uncached,
            }
        )
    groups = _clusters(selected)
    ratios = [by_task[task.id] for task in selected]
    rng, boot = random.Random(seed), []
    for _ in range(samples):
        drawn = [ratios[i] for _ in groups for i in rng.choice(groups)]
        boot.append(median(drawn))
    interval = [percentile(boot, 0.025), percentile(boot, 0.975)]
    ratio = median(ratios)
    return {
        "status": "verified",
        "samples": len(rows),
        "tasks": len(selected),
        "source_groups": len(groups),
        "per_task": task_stats,
        "median_cached_over_uncached": ratio,
        "ratio_ci95": interval,
        "median_time_saved_fraction": 1 - ratio,
        "time_saved_ci95": [1 - interval[1], 1 - interval[0]],
        "timing_supports_reduction_on_sample": interval[1] < 1,
        "bootstrap_samples": samples,
        "seed": seed,
        "scope": "eight predeclared tasks; warmed interleaved ABBA samples, three repeats per block; one model/container; descriptive task-group bootstrap, not an independent deployment replication",
    }


def _freeze_coherence_source(source_root, destination):
    """Verify the pinned Git blobs, then import a private copy without stale .pyc."""
    source_root = Path(source_root).resolve()
    result = subprocess.run(
        ["git", "-C", str(source_root), "ls-tree", "-r", COHERENCE_REVISION, "--", "src/jevbench"],
        capture_output=True,
        text=True,
        check=False,
    )
    _require(
        result.returncode == 0 and result.stdout.strip(), "pinned coherence source unavailable"
    )
    inventory = {}
    for line in result.stdout.splitlines():
        metadata, name = line.split("\t", 1)
        mode, kind, blob_hash = metadata.split()
        _require(
            kind == "blob" and mode in ("100644", "100755"), "unexpected coherence source object"
        )
        relative = Path(name)
        _require(relative.is_relative_to("src/jevbench"), "coherence source path escapes package")
        raw = _file(source_root, name).read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        _require(actual == blob_hash, f"coherence source differs from pin: {name}")
        inventory[name] = _hash(raw)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    actual = {
        str(p.relative_to(source_root))
        for p in (source_root / "src/jevbench").rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }
    _require(actual == set(inventory), "untracked or missing coherence package files")
    return {
        "repository": COHERENCE_REPOSITORY,
        "revision": COHERENCE_REVISION,
        "file_sha256": inventory,
        "inventory_sha256": _hash(_bytes(inventory)),
    }


def replay_coherence(report, cache_root, source_root):
    """Rebuild every fixed-suite request and score immutable cached answers."""
    settings, saved = report.get("settings"), report["result"]
    _require(settings == COHERENCE_SETTINGS, "coherence settings differ from predeclared defaults")
    _require(
        saved.get("selection")
        == {
            "domains": None,
            "dimensions": None,
            "groups": None,
            "relations": None,
            "diagnostics": False,
        }
        and saved.get("seed") == 0
        and saved.get("repeats") == 1
        and saved.get("trials") == 1,
        "coherence selection differs from full frozen mini suite",
    )
    _require(saved.get("suite", {}).get("name") == "jevbench-mini", "coherence suite differs")
    cache_root = Path(cache_root).resolve()
    _require(cache_root.is_dir(), "coherence raw answer cache unavailable")
    paths = sorted(cache_root.iterdir())
    answers, hashes = {}, {}
    for path in paths:
        _require(
            path.is_file()
            and not path.is_symlink()
            and len(path.stem) == 64
            and all(c in "0123456789abcdef" for c in path.stem)
            and path.suffix == ".json",
            "unexpected coherence cache entry",
        )
        raw = path.read_bytes()
        try:
            answer = json.loads(raw)
            _bytes(answer)  # Reject non-JSON floats such as NaN before official validation.
            answers[path.name] = raw.decode("utf-8")
        except (ValueError, UnicodeError) as exc:
            raise EvidenceError(f"invalid coherence cache JSON: {path.name}") from exc
        hashes[path.name] = _hash(raw)
    _require(len(answers) == 1248, "full mini suite requires exactly 1248 original cached requests")
    with tempfile.TemporaryDirectory(prefix="s1-coherence-audit-") as temporary:
        directory = Path(temporary)
        source = _freeze_coherence_source(source_root, directory)
        inputs = directory / "replay-inputs.json"
        inputs.write_bytes(
            _bytes({"backend": saved["backend"], "settings": settings, "answers": answers})
        )
        result = subprocess.run(
            [sys.executable, "-I", "-c", COHERENCE_REPLAY, str(directory / "src"), str(inputs)],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        _require(
            result.returncode == 0,
            "official offline coherence replay failed: " + result.stderr[-1000:],
        )
        try:
            replay = json.loads(result.stdout)
        except ValueError as exc:
            raise EvidenceError("offline coherence replay did not return JSON") from exc
    _require(not replay["missing"], "coherence replay requests lack original answers")
    _require(
        set(replay["used"]) == set(answers),
        "coherence cache does not exactly match frozen request plan",
    )
    _require(
        _metric_equal(
            {k: v for k, v in report["scores"].items() if k != "environment"},
            {k: v for k, v in replay["scores"].items() if k != "environment"},
        ),
        "coherence scores differ from raw-answer replay",
    )
    for key in COHERENCE_RESULT_KEYS:
        _require(
            _bytes(saved.get(key)) == _bytes(replay["result"].get(key)),
            f"coherence outcome field differs from raw-answer replay: {key}",
        )
    _require(
        replay["scores"].get("n_cases") == 240 and replay["scores"].get("n_checks") == 50,
        "coherence population differs from full mini suite",
    )
    return {
        "source": source,
        "raw_cache_files": len(hashes),
        "raw_cache_sha256": hashes,
        "raw_cache_inventory_sha256": _hash(_bytes(hashes)),
        "ordered_request_fingerprints_sha256": _hash(_bytes(replay["requests"])),
        "original_environment": report["scores"].get("environment"),
        "audit_environment": replay["scores"].get("environment"),
        "deterministic_outcome_comparison": "exact canonical JSON, including native base probabilities and per-test gaps",
        "raw_cache_hash_scope": "observed audit bytes; original campaign did not declare cache hashes",
        "scope": "official evaluate replay from 1248 native cached answers; no model invocation; timing and environment telemetry excluded from outcome equality",
    }


def coherence_summary(path, receipt, *, source_root=None, identity=None, model_name=None):
    """Read saved fields and optionally verify them against every native answer."""
    report = read_json(path)
    scores, result = report.get("scores"), report.get("result")
    _require(
        isinstance(scores, dict) and isinstance(result, dict), "saved coherence report malformed"
    )
    if receipt and receipt.get("coherence") is not None:
        stored = receipt["coherence"]
        _require(
            stored.get("scores") == scores and stored.get("errors") == result.get("errors"),
            "coherence report/receipt differ",
        )
        _require(
            stored.get("overall") == scores.get("overall")
            and stored.get("dimensions") == scores.get("pillars"),
            "coherence headline/receipt differ",
        )
    replay = None
    if source_root is not None:
        identity = identity or (receipt or {}).get("identity", {})
        _require(
            identity.get("coherence_commit") == COHERENCE_REVISION,
            "coherence identity source pin differs",
        )
        if model_name is not None:
            _require(
                result.get("backend") == f"{model_name}@{identity['revision']}",
                "coherence backend/model identity differs",
            )
        replay = replay_coherence(report, Path(path).parent / "coherence-cache", source_root)
    checks = scores.get("checks") or {}
    return {
        "status": "verified_raw_replay" if replay else "saved_report_only_not_recomputed",
        "replay": replay,
        "sha256": _hash(Path(path).read_bytes()),
        "overall": scores.get("overall"),
        "ci95": (scores.get("ci") or {}).get("overall"),
        "dimensions": scores.get("pillars"),
        "dimension_ci95": (scores.get("ci") or {}).get("pillars"),
        "dimension_ci95_status": "not_provided_by_pinned_official_scorer",
        "n_cases": scores.get("n_cases"),
        "n_checks": scores.get("n_checks"),
        "incomplete": scores.get("incomplete"),
        "check_coverage": {key: value.get("coverage") for key, value in checks.items()},
        "errors": len(result.get("errors") or []),
        "settings": report.get("settings"),
        "suite": result.get("suite"),
        "scope": "JevBench coherence and stability; not task accuracy or a leaderboard rank",
    }


def summarize_campaign(
    root,
    tasks_root,
    *,
    model_names=MODEL_NAMES,
    model_roots=None,
    coherence_root=None,
    bootstrap_samples=10000,
    seed=20261008,
):
    _require(
        type(bootstrap_samples) is int and bootstrap_samples >= 100,
        "at least 100 bootstrap resamples required",
    )
    tasks, official, config = load_public(tasks_root)
    root = Path(root).resolve()
    model_roots = model_roots or {}
    _require(set(model_roots) <= set(model_names), "model root overrides must name expected models")
    _require(
        len(set(model_names)) == len(model_names) and bool(model_names),
        "model names must be nonempty and unique",
    )
    models, audited, warnings = {}, {}, []
    for name in model_names:
        directory = Path(model_roots.get(name, root / name)).resolve()
        row = {
            "status": "not_run",
            "source_campaign": directory.parent.name,
            "source_directory": str(directory),
            "public": None,
            "coherence": {
                "status": "expected_not_run" if name == "s1-compiled" else "not_run",
                "reason": "predeclared public-only compiled experiment"
                if name == "s1-compiled"
                else "coherence report unavailable",
            },
            "setup_seconds": None,
            "cost_usd": None,
            "memory": None,
        }
        models[name] = row
        if not directory.exists():
            continue
        try:
            receipt = (
                read_json(directory / "receipt.json")
                if (directory / "receipt.json").is_file()
                else None
            )
        except (OSError, ValueError) as exc:
            row.update(status="integrity_failed", error=f"cell receipt unreadable: {exc}")
            continue
        if (directory / "failure.json").is_file():
            row["cell_failure"] = read_json(directory / "failure.json")
        try:
            evidence = audit_public(directory / "public231", tasks, official, config)
            if receipt:
                _require(
                    receipt.get("public") == evidence["summary"],
                    "public receipt differs from saved summary",
                )
                _require(
                    receipt.get("identity") == evidence["manifest"]["identity"],
                    "receipt/public model identity differs",
                )
            audited[name] = evidence
            summary = evidence["summary"]
            row.update(
                status="verified",
                identity=evidence["manifest"]["identity"],
                public={
                    "n_planned": summary["n_planned"],
                    "n_attempted": summary["n_attempted"],
                    "n_valid": summary["n_valid"],
                    "n_correct": summary["n_correct"],
                    "accuracy": summary["accuracy"],
                    "macro_accuracy": summary["macro_accuracy"],
                    "brier_mean": summary["brier_mean"],
                    "ece": summary["ece"],
                    "ordinal_mae": summary["ordinal_mae"],
                    "schema_validity": summary["schema_validity"],
                    "schema_validity_strict": summary["schema_validity_strict"],
                    "n_renormalized": summary["n_renormalized"],
                    "latency": summary["latency"],
                    "per_family": summary["per_family"],
                    "by_type": evidence["by_type"],
                    "verified_files": evidence["verified_files"],
                    "manifest_sha256": evidence["manifest_sha256"],
                },
                setup_seconds=evidence["manifest"]["identity"].get("setup_seconds"),
            )
        except (EvidenceError, OSError, ValueError, KeyError, TypeError) as exc:
            row.update(status="integrity_failed", error=str(exc))
            continue
        if receipt:
            row["receipt_sha256"] = _hash((directory / "receipt.json").read_bytes())
            row["memory"] = {
                "peak_allocated_bytes": receipt.get("memory_peak_allocated_bytes"),
                "peak_reserved_bytes": receipt.get("memory_peak_reserved_bytes"),
                "scope": receipt.get("memory_peak_scope")
                or "unknown; recorded cell peaks are not a controlled per-stage comparison",
                "phase_peaks": receipt.get("phase_memory"),
            }
        else:
            warnings.append(f"{name}: public evidence verified; full-cell receipt unavailable")
        if (directory / "coherence-mini.json").is_file():
            try:
                row["coherence"] = coherence_summary(
                    directory / "coherence-mini.json",
                    receipt,
                    source_root=coherence_root,
                    identity=evidence["manifest"]["identity"],
                    model_name=name,
                )
            except (
                EvidenceError,
                OSError,
                ValueError,
                KeyError,
                TypeError,
                subprocess.SubprocessError,
            ) as exc:
                row["coherence"] = {"status": "integrity_failed", "error": str(exc)}
        if name == "s1-bf16" and (directory / "cached-public231").exists():
            try:
                cached = audit_public(directory / "cached-public231", tasks, official, config)
                if receipt:
                    _require(
                        receipt.get("cached_public") == cached["summary"],
                        "cached public receipt differs",
                    )
                row["candidate_cache"] = {"equivalence": cache_equivalence(evidence, cached)}
                sample_path = directory / "candidate-cache-samples.json"
                if sample_path.is_file():
                    row["candidate_cache"]["timing"] = cache_timing(
                        sample_path, evidence, tasks, samples=bootstrap_samples, seed=seed
                    )
                    row["candidate_cache"]["timing"]["sample_file_sha256"] = _hash(
                        sample_path.read_bytes()
                    )
                    row["candidate_cache"]["time_reduction_supported"] = (
                        row["candidate_cache"]["equivalence"]["status"] == "passed"
                        and row["candidate_cache"]["timing"]["timing_supports_reduction_on_sample"]
                    )
                else:
                    row["candidate_cache"]["timing"] = {"status": "not_run"}
                    row["candidate_cache"]["time_reduction_supported"] = False
            except (EvidenceError, OSError, ValueError, KeyError, TypeError) as exc:
                row["candidate_cache"] = {
                    "status": "integrity_failed",
                    "error": str(exc),
                    "time_reduction_supported": False,
                }
    comparisons = []
    for reference, candidate in itertools.combinations(model_names, 2):
        pair = {"reference": reference, "candidate": candidate, "status": "inconclusive"}
        if reference in audited and candidate in audited:
            try:
                reference_identity = audited[reference]["manifest"]["identity"]
                candidate_identity = audited[candidate]["manifest"]["identity"]
                pair["hardware"] = {
                    "reference_gpu": reference_identity.get("gpu"),
                    "candidate_gpu": candidate_identity.get("gpu"),
                    "same_gpu_label": reference_identity.get("gpu") is not None
                    and reference_identity.get("gpu") == candidate_identity.get("gpu"),
                }
                pair["configuration_differences"] = [
                    key
                    for key in (
                        "model_id",
                        "revision",
                        "variant",
                        "gpu",
                        "precision",
                        "execution",
                        "candidate_cache",
                        "processor_files",
                        "source_files",
                        "versions",
                    )
                    if _bytes(reference_identity.get(key)) != _bytes(candidate_identity.get(key))
                ]
                pair["comparison_scope"] = (
                    "Observed quality across recorded named profiles and hardware; configuration differences "
                    "may be confounded. This quality interval does not establish a causal change or latency speedup."
                )
                pair.update(
                    status="computed",
                    **paired_accuracy(
                        audited[reference]["records"],
                        audited[candidate]["records"],
                        tasks,
                        samples=bootstrap_samples,
                        seed=seed,
                    ),
                )
            except (EvidenceError, KeyError, TypeError) as exc:
                pair["reason"] = str(exc)
        comparisons.append(pair)
    return {
        "schema_version": 1,
        "auditor_source_sha256": _hash(Path(__file__).read_bytes()),
        "campaign": root.name,
        "status": "complete_public_audit"
        if all(row["status"] == "verified" for row in models.values())
        else "incomplete_or_failed_public_audit",
        "dataset_label": config["dataset_label"],
        "dataset_sha256": config["canonical_dataset_sha256"],
        "expected_models": list(model_names),
        "source_campaigns": sorted({row["source_campaign"] for row in models.values()}),
        "coherence_audit": "complete_raw_replay_audit"
        if all(
            row["coherence"]["status"] in ("verified_raw_replay", "expected_not_run")
            for row in models.values()
        )
        else "incomplete_or_unrecomputed_coherence_audit",
        "models": models,
        "paired_accuracy": comparisons,
        "warnings": warnings,
        "official_rank": None,
        "official_composite": None,
        "leaderboard_status": "not_submitted",
        "sealed_evaluation": "not_run",
        "public_exposure": config["public_exposure"],
        "scope": "Frozen public accuracy screening; coherence raw-answer replay when pinned source supplied; source-group intervals are descriptive, not an official leaderboard result",
        "floating_aggregate_recompute_tolerance": {
            "relative": METRIC_TOLERANCE,
            "absolute": METRIC_TOLERANCE,
            "scope": "aggregate float reductions may differ across Python/platform; task populations, request hashes and cache probabilities remain exact",
        },
    }


def _number(value, *, percent=False):
    if value is None:
        return "unknown"
    return f"{100 * value:.2f}%" if percent else f"{value:.4f}"


def markdown(report):
    lines = [
        f"JevBench public screening — {report['campaign']}",
        "",
        report["public_exposure"],
        "",
        "Official rank: not submitted. Sealed evaluation: not run.",
        "",
        "| Model | Source run | Audit | Valid / planned | Correct | Accuracy | Coherence | Coherence audit | Setup seconds |",
        "|---|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for name, row in report["models"].items():
        p, c = row.get("public") or {}, row.get("coherence") or {}
        valid = f"{p['n_valid']}/{p['n_planned']}" if p else "unknown"
        lines.append(
            f"| {name} | {row['source_campaign']} | {row['status']} | {valid} | {p.get('n_correct', 'unknown')} | {_number(p.get('accuracy'), percent=True)} | {_number(c.get('overall'), percent=True)} | {c.get('status', 'unknown')} | {_number(row.get('setup_seconds'))} |"
        )
    lines.extend(
        [
            "",
            "Accuracy intervals use paired source groups; failures remain incorrect decisions in the complete population. Coherence measures stability. Its audit status distinguishes raw-answer replay, saved-only metrics and expected omissions. The pinned scorer supplies the overall interval, but no dimension intervals.",
            "",
        ]
    )
    for pair in report["paired_accuracy"]:
        if pair["status"] == "computed":
            lo, hi = pair["ci95"]
            lines.append(
                f"- {pair['candidate']} minus {pair['reference']}: {100 * pair['delta_accuracy']:+.2f} percentage points; descriptive 95% CI [{100 * lo:+.2f}, {100 * hi:+.2f}]."
            )
            if not pair["hardware"]["same_gpu_label"]:
                lines.append(
                    f"  Recorded hardware: {pair['hardware']['reference_gpu']} versus {pair['hardware']['candidate_gpu']}; observed quality only, with hardware/configuration confounding."
                )
        else:
            lines.append(f"- {pair['candidate']} vs {pair['reference']}: inconclusive.")
    for name, row in report["models"].items():
        memory = row.get("memory") or {}
        lines.extend(
            [
                "",
                f"{name} memory scope: {memory.get('scope', 'unknown')}. Per-decision cost: unknown.",
            ]
        )
        cache = row.get("candidate_cache")
        if cache:
            timing = cache.get("timing") or {}
            lines.extend(
                [
                    "",
                    f"Candidate-cache equivalence: {(cache.get('equivalence') or {}).get('status', cache.get('status', 'unknown'))}.",
                ]
            )
            if timing.get("status") == "verified":
                lines.append(
                    f"Warmed ABBA median cached/uncached ratio: {timing['median_cached_over_uncached']:.4f}; 95% CI {timing['ratio_ci95']}. Time reduction supported on the measured sample: {cache['time_reduction_supported']}."
                )
                lines.append(timing["scope"] + ".")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--coherence-root", type=Path)
    parser.add_argument("--model-root", action="append", default=[], metavar="NAME=DIRECTORY")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    args = parser.parse_args(argv)
    roots = {}
    for override in args.model_root:
        name, separator, directory = override.partition("=")
        if not separator or not name or not directory or name in roots:
            parser.error("--model-root needs a unique NAME=DIRECTORY")
        roots[name] = Path(directory)
    report = summarize_campaign(
        args.campaign,
        args.tasks_root,
        model_names=tuple(roots) if roots else MODEL_NAMES,
        model_roots=roots,
        coherence_root=args.coherence_root,
        bootstrap_samples=args.bootstrap_samples,
    )
    outputs = []
    if args.output is not None:
        outputs.append((args.output, _bytes(report).decode() + "\n"))
    if args.markdown is not None:
        outputs.append((args.markdown, markdown(report)))
    for path, _ in outputs:
        if path.exists():
            raise FileExistsError(f"report exists: {path}")
    for path, content in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(content)
    if args.output is None:
        print(_bytes(report).decode())
    complete = report["status"] == "complete_public_audit" and (
        args.coherence_root is None or report.get("coherence_audit") == "complete_raw_replay_audit"
    )
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
