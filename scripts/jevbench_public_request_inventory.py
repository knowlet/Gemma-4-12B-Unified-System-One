"""Inventory the complete pinned public JevBench populations without model calls.

Pass local checkouts of the revisions in configs/benchmarks. This tool never writes
requests or answers and prints aggregate counts only. The coherence default is
its official mini suite; --coherence-suite jevbench-240 selects its full suite.
The frozen plan is expanded without minimization and deduplicated using the
publisher's case-id/question rule. Inventory is not a model score.

For pinned public checkouts, run from the repository root:
  .venv/bin/python -B -s scripts/jevbench_public_request_inventory.py \
      --heaven-root /path/to/legacy-public-jevbench \
      --coherence-root /path/to/coherence-jevbench

The JSON pin files record the exact official revisions and source SHA-256s.
Use --coherence-suite jevbench-240 to inventory the larger official plan.
No network, model, browser, or output artifact is used.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.abc
import importlib.readers
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import run_jevbench_public as heaven_runner

from s1.contracts import DecisionRequest
from s1.evaluation.decision3 import native_payload, request_blockers

ROOT = Path(__file__).resolve().parents[1]
COHERENCE_CONFIG = ROOT / "configs/benchmarks/jevbench-coherence-public.json"

# The pinned official MINI loader recursively authenticates every base data file,
# including the full base plan, before using the MINI plan. This is a fixed
# inventory, not filesystem discovery or a claim that those base trials run.
_MINI_ASSETS = frozenset(
    "src/jevbench/suites/" + name
    for name in (
        "jevbench-mini/manifest.json",
        "jevbench-mini/tests.jsonl.gz",
        "jevbench-240/manifest.json",
        "jevbench-240/cases.jsonl",
        "jevbench-240/rewrites.jsonl",
        "jevbench-240/tests.jsonl.gz",
    )
)
_MINI_SDK = frozenset(
    "src/jevbench/" + name
    for name in (
        "__init__.py",
        "__main__.py",
        "api.py",
        "backends.py",
        "cli.py",
        "datagen/__init__.py",
        "datagen/domains/__init__.py",
        "datagen/domains/_rules.py",
        "datagen/domains/agent_logs.py",
        "datagen/domains/content_moderation.py",
        "datagen/domains/customer_support.py",
        "datagen/domains/ecommerce_returns.py",
        "datagen/domains/education_grading.py",
        "datagen/domains/finance_ops.py",
        "datagen/domains/healthcare_admin.py",
        "datagen/domains/hr_recruiting.py",
        "datagen/domains/it_incidents.py",
        "datagen/domains/legal_contracts.py",
        "datagen/domains/logistics.py",
        "datagen/domains/security_ops.py",
        "datagen/freeze.py",
        "datagen/sample.py",
        "datagen/spec.py",
        "datagen/text.py",
        "engine.py",
        "plan.py",
        "relations/__init__.py",
        "relations/_shared.py",
        "relations/base.py",
        "relations/bat.py",
        "relations/cho.py",
        "relations/diagnostics.py",
        "relations/log.py",
        "relations/mea.py",
        "relations/rep.py",
        "render.py",
        "results.py",
        "rewrite.py",
        "schema.py",
        "score.py",
        "suites/__init__.py",
        "taxonomy.py",
    )
)


def _mini_directory_check(root):
    """Reject inputs the official loader would otherwise read before hash checks.

    This pre/post check alone does not prevent transient concurrent replacement;
    the execution owner must provide a separately controlled immutable checkout.
    """
    for suite, expected in (
        ("jevbench-mini", {"tests.jsonl.gz"}),
        ("jevbench-240", {"cases.jsonl", "rewrites.jsonl", "tests.jsonl.gz"}),
    ):
        directory = root / "src/jevbench/suites" / suite
        if directory.resolve() != directory or not directory.is_dir():
            raise ValueError("ordinary official suite directory required")
        found = set()
        for entry in directory.iterdir():
            if entry.name.endswith((".jsonl", ".jsonl.gz")):
                if entry.is_symlink() or not entry.is_file():
                    raise ValueError("ordinary official suite data required")
                found.add(entry.name)
        if found != expected:
            raise ValueError("official suite data inventory differs")


def _ordered_bytes(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


class _VerifiedFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Load the official package from checked bytes, never installed modules or pyc."""

    def __init__(self, root, sources):
        self.root = root
        self.sources = sources

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "jevbench" and not fullname.startswith("jevbench."):
            return None
        base = "src/" + fullname.replace(".", "/")
        for relative, is_package in ((base + "/__init__.py", True), (base + ".py", False)):
            if relative in self.sources:
                return importlib.util.spec_from_file_location(
                    fullname,
                    self.root / relative,
                    loader=self,
                    submodule_search_locations=[str((self.root / relative).parent)]
                    if is_package
                    else None,
                )
        raise ImportError("official package requested an unpinned module")

    def get_resource_reader(self, fullname):
        module = sys.modules.get(fullname)
        return (
            importlib.readers.FileReader(SimpleNamespace(path=module.__file__)) if module else None
        )

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        base = "src/" + module.__name__.replace(".", "/")
        package = base + "/__init__.py" in self.sources
        relative = base + ("/__init__.py" if package else ".py")
        module.__file__ = str(self.root / relative)
        if package:
            module.__path__ = [str((self.root / relative).parent)]
        exec(compile(self.sources[relative], module.__file__, "exec"), module.__dict__)


def _check_coherence_files(root, config, *, suite_name=None):
    supplied_root = Path(root)
    root = supplied_root.resolve()
    sources = {}
    if suite_name not in (None, "jevbench-mini"):
        raise ValueError("unsupported selective coherence suite")
    files = config["files"]
    if suite_name == "jevbench-mini":
        if not supplied_root.is_absolute() or supplied_root != root:
            raise ValueError("ordinary absolute coherence checkout required")
        selected = _MINI_SDK | _MINI_ASSETS
        if not selected.issubset(files):
            raise ValueError("missing required MINI source pin")
        # Preflight all paths and both directories before opening any payload.
        _mini_directory_check(root)
        for relative in selected:
            path = root / relative
            if path.resolve() != path or not path.is_file():
                raise ValueError("ordinary official MINI source required")
        names = sorted(selected)
    else:
        # Preserve the original complete-config inventory when unspecified.
        names = files
    for relative in names:
        expected = files[relative]
        path = root / relative
        if not path.resolve().is_relative_to(root):
            raise ValueError("official coherence source escapes checkout")
        raw = path.read_bytes()
        if _sha(raw) != expected:
            raise ValueError("official coherence source hash mismatch")
        if relative.endswith(".py"):
            sources[relative] = raw
    return sources


def _verified_coherence(root, config, *, suite_name=None):
    sources = _check_coherence_files(root, config, suite_name=suite_name)
    root = Path(root).resolve()
    if any(name == "jevbench" or name.startswith("jevbench.") for name in sys.modules):
        raise RuntimeError("coherence module already loaded")
    return _VerifiedFinder(root, sources)


def _counter(counter):
    return {str(k): v for k, v in sorted(counter.items(), key=lambda x: str(x[0]))}


def summarize(items, validate_question=None):
    totals = collections.Counter()
    heads = collections.Counter()
    kinds = collections.Counter()
    states = collections.Counter()
    options = collections.Counter()
    errors = collections.Counter()
    blockers = collections.Counter()
    changes = collections.Counter()
    targets = collections.Counter()
    known_blockers = {
        "decision3_audio_unsupported",
        "decision3_timestamped_frames_unsupported",
        "max_options_exceeded",
        "decision3_score_levels_exceeded",
        "decision3_score_levels_must_be_distinct_nonempty",
    }
    for state, questions in items:
        totals["requests"] += 1
        heads[len(questions)] += 1
        states[{str: "str", dict: "dict"}.get(type(state), "other")] += 1
        for key, question in questions.items():
            kind = question.get("type")
            kinds[kind if kind in ("choice", "noul", "score") else "other"] += 1
            if key in ("operation", "click_target"):
                targets[key] += 1
            if kind in ("choice", "score") and isinstance(question.get("criteria"), (dict, list)):
                options[len(question["criteria"])] += 1
            if validate_question is not None:
                try:
                    validate_question(key, question)
                except Exception:
                    totals["official_schema_rejected_questions"] += 1
                else:
                    totals["official_schema_accepted_questions"] += 1
        if 1 <= len(questions) <= 4 and all(q.get("type") == "choice" for q in questions.values()):
            totals["browser_adapter_shape_eligible"] += 1
        try:
            parsed = DecisionRequest.model_validate({"state": state, "questions": questions})
        except Exception as error:
            totals["contract_rejected"] += 1
            if callable(getattr(error, "errors", None)):
                for detail in error.errors():
                    # Error categories, never input, location, message, or context.
                    error_type = detail.get("type")
                    safe_types = {
                        "value_error",
                        "missing",
                        "extra_forbidden",
                        "string_type",
                        "dict_type",
                        "list_type",
                        "literal_error",
                        "too_short",
                        "too_long",
                        "model_type",
                        "int_type",
                        "float_type",
                    }
                    errors[error_type if error_type in safe_types else "other"] += 1
            else:
                errors["other"] += 1
            continue
        totals["contract_accepted"] += 1
        reasons = request_blockers(parsed)
        if reasons:
            totals["decision3_static_blocked"] += 1
            blockers.update(r if r in known_blockers else "other" for r in reasons)
            continue
        totals["decision3_static_accepted"] += 1
        original = {"state": state, "questions": questions}
        try:
            converted = native_payload(parsed)
        except Exception:
            totals["native_conversion_rejected"] += 1
            continue
        if converted == original:
            totals["native_structural_equal"] += 1
        else:
            totals["native_structural_changed"] += 1
        if _ordered_bytes(converted) == _ordered_bytes(original):
            totals["native_ordered_json_equal"] += 1
        else:
            totals["native_ordered_json_changed"] += 1
            if converted == original:
                changes["field_order_only_requests"] += 1
        if list(questions) != list(converted.get("questions", {})):
            changes["head_order_or_membership_changed_requests"] += 1
        if converted.get("state") != state:
            changes["state_value_changed_requests"] += 1
        for key, question in questions.items():
            out = converted.get("questions", {}).get(key)
            if not isinstance(out, dict):
                changes["missing_or_invalid_output_heads"] += 1
                continue
            before, after = question.get("criteria"), out.get("criteria")
            if "criteria" in question and before is None and "criteria" not in out:
                changes["null_criteria_omitted_heads"] += 1
            elif before != after or ("criteria" in question) != ("criteria" in out):
                changes["other_criteria_changes_heads"] += 1
            if isinstance(before, dict) and isinstance(after, dict) and list(before) != list(after):
                changes["criteria_order_or_membership_changed_heads"] += 1
            if set(question) == set(out) and list(question) != list(out):
                changes["question_field_order_changed_heads"] += 1
            left = {k: v for k, v in question.items() if k != "criteria"}
            right = {k: v for k, v in out.items() if k != "criteria"}
            if left != right:
                changes["other_question_field_changes_heads"] += 1
    return {
        "totals": _counter(totals),
        "heads_per_request": _counter(heads),
        "question_types": _counter(kinds),
        "state_types": _counter(states),
        "choice_score_option_counts": _counter(options),
        "contract_error_types": _counter(errors),
        "decision3_blockers": _counter(blockers),
        "native_changes": _counter(changes),
        "target_question_occurrences": _counter(targets),
    }


def coherence_inventory(suite, manifest, decode_requests, validate_question):
    cases = suite.cases
    plan = suite.plan
    by_id = {case.id: case for case in cases}
    if len(by_id) != len(cases):
        raise ValueError("duplicate official case IDs")
    seen = {
        (case.id, json.dumps(case.questions, ensure_ascii=False)): (case.state, case.questions)
        for case in cases
    }
    occurrences = 0
    for rows in plan.by.values():
        for row in rows:
            case = by_id[row["case"]]
            for questions in decode_requests(case, row["requests"]).values():
                occurrences += 1
                seen[(case.id, json.dumps(questions, ensure_ascii=False))] = (case.state, questions)
    if len(seen) != manifest["plan"]["requests"] or len(plan) != manifest["plan"]["trials"]:
        raise ValueError("official plan population differs from manifest")
    return {
        "cases": len(cases),
        "plan_rows": len(plan),
        "trial_request_occurrences": occurrences,
        "case_question_unique_requests": len(seen),
        "state_question_unique_requests": len(
            {_ordered_bytes({"state": s, "questions": q}) for s, q in seen.values()}
        ),
        "summary": summarize(seen.values(), validate_question),
    }


def run(heaven_root, coherence_root, suite_name):
    if suite_name not in ("jevbench-mini", "jevbench-240"):
        raise ValueError("unsupported official suite")
    tasks, official, config = heaven_runner.load_public(heaven_root)
    heaven = summarize(
        (task.state, {"decision": official.base.build_question(task)}) for task in tasks
    )
    pin_bytes = COHERENCE_CONFIG.read_bytes()
    pin = json.loads(pin_bytes)
    selected = "jevbench-mini" if suite_name == "jevbench-mini" else None
    finder = _verified_coherence(coherence_root, pin, suite_name=selected)
    sys.meta_path.insert(0, finder)
    try:
        from jevbench.plan import decode_requests
        from jevbench.schema import validate_question
        from jevbench.suites import load_suite

        suite = load_suite(suite_name)
        manifest_path = Path(coherence_root) / "src/jevbench/suites" / suite_name / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        inventory = coherence_inventory(suite, manifest, decode_requests, validate_question)
        # The official suite reads data from disk; reject ordinary mid-run changes.
        _check_coherence_files(coherence_root, pin, suite_name=selected)
    finally:
        sys.meta_path.remove(finder)
        for name in list(sys.modules):
            if name == "jevbench" or name.startswith("jevbench."):
                del sys.modules[name]
    return {
        "kind": "offline_request_inventory_not_a_score",
        "minimization": False,
        "heaven": {
            "source_repository": config["source_repository"],
            "source_revision": config["source_revision"],
            "dataset_sha256": config["canonical_dataset_sha256"],
            "source_pin_sha256": _sha(heaven_runner.CONFIG.read_bytes()),
            "summary": heaven,
        },
        "coherence": {
            "source_repository": pin["source_repository"],
            "source_revision": pin["source_revision"],
            "suite": suite_name,
            "source_pin_sha256": _sha(pin_bytes),
            "manifest_sha256": _sha(manifest_path.read_bytes()),
            "plan_sha256": manifest["files"]["tests.jsonl.gz"],
            **inventory,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heaven-root", required=True, type=Path)
    parser.add_argument("--coherence-root", required=True, type=Path)
    parser.add_argument(
        "--coherence-suite", choices=("jevbench-mini", "jevbench-240"), default="jevbench-mini"
    )
    args = parser.parse_args()
    try:
        result = run(args.heaven_root, args.coherence_root, args.coherence_suite)
    except Exception as exc:
        # Exception messages may include request content. Do not emit them.
        print(f"inventory failed ({type(exc).__name__}); no result produced", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
