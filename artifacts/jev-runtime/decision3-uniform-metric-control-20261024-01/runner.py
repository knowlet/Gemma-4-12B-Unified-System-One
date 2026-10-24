#!/usr/bin/env python3
"""Held, non-model uniform-primitive control for the unchanged official MINI.

Only the Python standard library is imported before execute().  No official
package, pin metadata or benchmark payload is opened in the default path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import stat
import sys


# Root must review this source and choose an execution/closure boundary before
# deliberately changing this constant.  CLI flags cannot release this hold.
SOURCE_ONLY = True
NAME = "diagnostic:uniform-primitive-mini-20261024-01"
REPO = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
BENCHMARK_PLAN = REPO / (
    "artifacts/jev-runtime/decision3-publisher-lite-cpu-20261023-01/"
    "benchmark-source-plan.json"
)
BENCHMARK_PLAN_SHA = "b5caa794d864917a637034bd1cebe0487c9eec0e5c3c423835d24fca4f397021"
COHERENCE = REPO / "artifacts/jevbench/20261022-local-score-01/upstream/coherence"
REVISION = "e18733694623aa93058e279c3534f8b8e2edefa3"
DIMENSIONS = ("REP", "BAT", "MEA", "LOG", "CHO")
GROUPS = (
    "REP.invariance", "REP.equivariance", "REP.binding", "REP.type",
    "BAT.isolation", "MEA.complement", "MEA.partition", "MEA.marginalisation",
    "LOG.bounds", "LOG.monotonicity", "CHO.conditioning",
)
SETTINGS = {"tolerance": 0.05, "min_n": 5, "bootstrap": 1000, "min_coverage": 0.95}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def uniform_answers(state, questions):
    """Ignore state/text; preserve every supplied question/criterion order and id."""
    del state
    require(type(questions) is dict and bool(questions), "questions must be a nonempty object")
    answers = {}
    for key, question in questions.items():
        require(type(key) is str and type(question) is dict, "invalid question shape")
        kind = question.get("type")
        criteria = question.get("criteria")
        if kind == "noul":
            require(criteria is None or criteria == {}, "Noul has no criteria")
            answers[key] = {"type": "noul", "noul": 0.5}
        elif kind == "choice":
            require(type(criteria) is dict and len(criteria) >= 2, "Choice needs options")
            require(all(type(option) is str for option in criteria), "Choice ids must be strings")
            answers[key] = {
                "type": "choice",
                "probabilities": {option: 1.0 / len(criteria) for option in criteria},
            }
        elif kind == "score":
            require(type(criteria) is list and len(criteria) >= 2, "Score needs levels")
            answers[key] = {
                "type": "score",
                "probabilities": {str(i): 1.0 / len(criteria) for i in range(len(criteria))},
            }
        else:
            raise ValueError("unsupported primitive")
    return answers


class UniformControl:
    """Aggregate-only observation of actual callable requests; no text/ids retained."""

    def __init__(self):
        self.requests = 0
        self.heads = {kind: 0 for kind in ("noul", "choice", "score")}
        self.arities = {kind: {} for kind in self.heads}

    def __call__(self, state, questions):
        answers = uniform_answers(state, questions)
        self.requests += 1
        for question in questions.values():
            kind = question["type"]
            arity = 2 if kind == "noul" else len(question["criteria"])
            self.heads[kind] += 1
            histogram = self.arities[kind]
            histogram[str(arity)] = histogram.get(str(arity), 0) + 1
        return answers

    def diagnostic(self):
        return {
            "callable_requests": self.requests,
            "heads_by_primitive": dict(self.heads),
            "arity_histograms": {kind: dict(histogram) for kind, histogram in self.arities.items()},
            "all_tied_heads": sum(self.heads.values()),
            "reads_state_or_instructions": False,
            "argmax_action_or_correctness_measured": False,
        }


def integer(value, label):
    require(type(value) is int and value >= 0, label + " must be a nonnegative integer")
    return value


def number(value, label, nullable=False, upper=1.0):
    if nullable and value is None:
        return None
    require(type(value) in (int, float) and math.isfinite(value), label + " must be finite")
    require(0.0 <= value <= upper, label + " out of range")
    return value


def metric_map(value, names, label, nullable=True, exact=True):
    require(type(value) is dict, label + " must be an object")
    require(set(value) == set(names) if exact else set(value) <= set(names), label + " keys differ")
    return {key: number(item, label, nullable=nullable) for key, item in value.items()}


def project_report(report, control, checks):
    """Copy typed official aggregates; never write Report.result, card or examples.

    checks maps the official relation names to their official primary check ids.
    No metric is recomputed here.  Official errors are counted and hashed, not
    forced to zero; the original scorer retains its own coverage policy.
    """
    result, scores = report.result, report.scores
    require(type(checks) is dict and len(checks) == 50, "50 official relations required")
    require(report.settings == SETTINGS, "official score settings differ")
    require(result["backend"] == NAME, "control identity differs")
    require(result["jevbench_version"] == "0.1.0", "official version differs")
    require(all(type(result[key]) is int for key in ("seed", "trials", "repeats", "concurrency")), "run count types differ")
    require((result["seed"], result["trials"], result["repeats"]) == (0, 1, 1), "frozen run differs")
    require(result["concurrency"] == 1 and result["truncated_by_budget"] is False, "incomplete run")
    require(result["backend_down"] is None, "backend stopped")
    require(result["suite"]["name"] == "jevbench-mini", "suite differs")
    require(result["selection"] == {
        "domains": None, "dimensions": None, "groups": None, "relations": None, "diagnostics": False,
    }, "suite selection differs")
    require(integer(result["requests"], "requests") == 1248, "1248 distinct requests required")
    require(integer(control.requests, "callable requests") == 1248, "callable request count differs")
    require(type(result["cases"]) is list and len(result["cases"]) == 240, "240 cases required")
    require(all(type(case) is str for case in result["cases"]) and len(set(result["cases"])) == 240, "case ids differ")
    require(type(result["case_domains"]) is dict and len(result["case_domains"]) == 240, "240 domains rows required")
    require(set(result["case_domains"]) == set(result["cases"]), "case identities differ")
    require(type(result["relations"]) is list and len(result["relations"]) == 50, "50 relations required")
    require(set(result["relations"]) == set(checks), "official relations differ")
    planned = result["planned_tests"]
    require(type(planned) is dict and set(planned) == set(checks), "planned relation keys differ")
    require(all(integer(value, "planned tests") == 24 for value in planned.values()), "1200 original tests required")
    require(type(result["errors"]) is list, "official errors must be a list")
    error_sha = hashlib.sha256(json.dumps(
        result["errors"], sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode()).hexdigest()

    require(scores["backend"] == NAME and scores["tolerance"] == "fixed 0.05", "scorer identity differs")
    require(integer(scores["min_n"], "min_n") == 5 and integer(scores["bootstrap"], "bootstrap") == 1000, "scorer settings differ")
    require(type(scores["min_coverage"]) is float and scores["min_coverage"] == 0.95, "coverage differs")
    official = {
        "overall": number(scores["overall"], "overall", nullable=True),
        "dimensions": metric_map(scores["pillars"], DIMENSIONS, "dimensions"),
        "groups": metric_map(scores["groups"], GROUPS, "groups"),
        "relations": metric_map(scores["relations"], checks, "relations", nullable=False, exact=False),
    }
    graded = scores["graded"]
    require(type(graded) is dict and set(graded) == {"overall", "pillars", "groups", "relations"}, "graded keys differ")
    official["graded"] = {
        "overall": number(graded["overall"], "graded overall", nullable=True),
        "dimensions": metric_map(graded["pillars"], DIMENSIONS, "graded dimensions"),
        "groups": metric_map(graded["groups"], GROUPS, "graded groups"),
        "relations": metric_map(graded["relations"], checks, "graded relations", nullable=False, exact=False),
    }
    require(type(scores["ci"]) is dict and set(scores["ci"]) <= {"overall", "graded_overall"}, "CI keys differ")
    official["intervals"] = {}
    for key, interval in scores["ci"].items():
        require(type(interval) is list and len(interval) == 2, "interval shape differs")
        low, high = [number(item, "interval") for item in interval]
        require(low <= high, "interval order differs")
        official["intervals"][key] = [low, high]

    allowed_checks = {relation + "/" + check for relation, check in checks.items()}
    require(type(scores["checks"]) is dict and set(scores["checks"]) == allowed_checks, "check inventory differs")
    official["relation_checks"] = {}
    for relation, check in checks.items():
        row = scores["checks"][relation + "/" + check]
        require(type(row) is dict and set(row) <= {"n", "planned", "coverage", "score", "graded", "excluded"}, "check row differs")
        require(integer(row["planned"], "planned") == 24, "planned count differs")
        projected = {
            "check": check, "n": integer(row["n"], "judged tests"), "planned": 24,
            "coverage": number(row["coverage"], "coverage"), "excluded": "excluded" in row,
        }
        for key in ("score", "graded"):
            projected[key] = number(row[key], key) if key in row else None
        official["relation_checks"][relation] = projected
    require(type(scores["excluded"]) is dict and set(scores["excluded"]) == {"low_n", "exploratory", "diagnostic"}, "excluded keys differ")
    official["excluded"] = {}
    for key, values in scores["excluded"].items():
        require(type(values) is list and set(values) <= allowed_checks, "excluded identifiers differ")
        official["excluded"][key] = list(values)
    require(type(scores["incomplete"]) is list and set(scores["incomplete"]) <= set(checks), "incomplete identifiers differ")
    official["incomplete_relations"] = list(scores["incomplete"])
    official["settings"] = dict(SETTINGS)
    official["counts"] = {key: integer(scores[key], key) for key in (
        "n_cases", "n_checks", "n_outcomes", "skipped_outcomes",
    )}
    return {
        "schema": "decision3-uniform-primitive-metric-control-v1",
        "status": "computed_unreviewed_control_only",
        "scope": "uniform_primitive_metric_control",
        "uses_model": False, "model_qualification": False,
        "browser_qualification": False, "training_qualification": False,
        "ranking_qualification": False, "competitive_win_claim": False,
        "universal_qualification": False, "root_closure_audit_required": True,
        "policy": {"noul": 0.5, "choice": "1/K per original criterion id", "score": "1/L per original level index"},
        "inventory": {"requests": 1248, "cases": 240, "relations": 50, "planned_tests": 1200,
                      "cache_hits": integer(result["cache_hits"], "cache hits"), "cache": "fresh :memory:"},
        "official_metrics": official,
        "control_diagnostic": control.diagnostic(),
        "official_errors": {"count": len(result["errors"]), "canonical_sha256": error_sha},
    }


def checked_file(path, expected_sha, expected_size=None, expected_stat=None):
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode), "ordinary pinned file required")
    if expected_size is not None:
        require(before.st_size == expected_size, "pinned size differs")
    require(before.st_size <= 1024 * 1024, "small source/metadata file required")
    payload = path.read_bytes()
    after = path.lstat()
    fields = ("st_dev", "st_ino", "st_mtime_ns", "st_size")
    require(all(getattr(before, key) == getattr(after, key) for key in fields), "pinned file changed while reading")
    if expected_stat is not None:
        require(set(expected_stat) == set(fields), "stat pin keys differ")
        require(all(type(expected_stat[key]) is int and getattr(after, key) == expected_stat[key] for key in fields), "pinned stat differs")
    require(hashlib.sha256(payload).hexdigest() == expected_sha, "pinned SHA differs")
    return payload


def benchmark_pins():
    plan = json.loads(checked_file(BENCHMARK_PLAN, BENCHMARK_PLAN_SHA))
    require(plan["schema"] == "decision3-cpu-two-suite-source-plan-v1", "source plan schema differs")
    require(plan["coherence"] == {
        "root": str(COHERENCE), "repository": "https://github.com/JevBench/jevbench",
        "revision": REVISION, "entry": "jevbench.evaluate/jevbench.from_callable", "suite": "jevbench-mini",
        "concurrency": 1, "requests": 1248, "cases": 240, "relations": 50, "planned_tests": 1200,
        "fresh_cache_required": True,
    }, "frozen coherence metadata differs")
    sources, metadata = plan["source_files"], plan["existing_dataset_metadata"]
    require(type(sources) is list and len(sources) == 78, "78 original source refs required")
    require(type(metadata) is list and len(metadata) == 2, "2 original metadata refs required")
    refs = sources + metadata
    require(len({ref["path"] for ref in refs}) == 80, "duplicate selected pin")
    # The SHA-authenticated metadata is the selection authority.  Its model,
    # adapter, resource and old owner fields confer no qualification here.
    for ref in refs:
        require(set(ref) == {"path", "size", "stat", "sha256"}, "pin row keys differ")
        checked_file(Path(ref["path"]), ref["sha256"], ref["size"], ref["stat"])
    return refs


def execute(args):
    # Must remain the first operation: held --execute cannot read pins, own
    # source, output paths, suite payload or any official dependency.
    require(SOURCE_ONLY is False, "SOURCE_ONLY: root source/boundary review has not released execution")
    require(sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode, "isolated -I -S -B interpreter required")
    require(args.output is not None and args.expected_runner_sha256 is not None, "output and selected runner SHA required")
    selected_sha = args.expected_runner_sha256
    require(len(selected_sha) == 64 and all(c in "0123456789abcdef" for c in selected_sha), "runner SHA shape differs")
    checked_file(Path(__file__), selected_sha)
    refs = benchmark_pins()
    require(not any(name == "jevbench" or name.startswith("jevbench.") for name in sys.modules), "official package already imported")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(COHERENCE / "src"))
    jb = importlib.import_module("jevbench")
    require(Path(jb.__file__).resolve() == (COHERENCE / "src/jevbench/__init__.py").resolve(), "official module origin differs")
    control = UniformControl()
    report = jb.evaluate(
        jb.from_callable(control, name=NAME, concurrent=False),
        suite="jevbench-mini", cache=":memory:", concurrency=1,
        seed=0, repeats=1, trials=1, minimize=False, max_requests=None,
        tolerance=0.05, min_n=5, bootstrap=1000, progress=False,
    )
    checks = {name: jb.describe(name)["check"] for name in jb.select_relations()}
    aggregate = project_report(report, control, checks)
    # Recheck the same original pins after the official computation.  No raw
    # report.save()/card()/table(), examples or benchmark contents are written.
    for ref in refs:
        checked_file(Path(ref["path"]), ref["sha256"], ref["size"], ref["stat"])
    checked_file(BENCHMARK_PLAN, BENCHMARK_PLAN_SHA)
    checked_file(Path(__file__), selected_sha)
    aggregate["provenance"] = {
        "runner_sha256": selected_sha, "benchmark_source_plan_sha256": BENCHMARK_PLAN_SHA,
        "coherence_revision": REVISION, "suite": "jevbench-mini",
        "source_refs_verified_before_after": 78, "metadata_refs_verified_before_after": 2,
        "official_suite_loader_payload_hash_checks": True,
        "model_or_previous_owner_qualification_inherited": False,
    }
    # Exclusive output preserves an existing result.  Root chooses the output
    # and performs its own post-terminal closure/source audit separately.
    with Path(args.output).open("x", encoding="utf-8") as stream:
        json.dump(aggregate, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--expected-runner-sha256")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"status": "held_source_only" if SOURCE_ONLY else "not_executed", "source_only": SOURCE_ONLY,
                          "uses_model": False, "official_imported": False,
                          "benchmark_read": False, "qualification": False}))
        return 0
    try:
        return execute(args)
    except Exception as error:
        # Official exception messages may contain question text; do not copy
        # them into this aggregate-only runner's output or terminal stream.
        print(json.dumps({"status": "refused_or_failed", "error_type": type(error).__name__,
                          "qualification": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
