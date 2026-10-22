"""Explicit scoped continuation of Edge02; importing and default CLI start nothing."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).absolute().parents[3]
HERE = Path(__file__).absolute().parent
EXECUTION = ROOT / "artifacts/jev-runtime/decision3-local-execution-retry-20261022-01/execution.py"
EXECUTION_SHA = "248687b7dddc5cf7697482e85b2ec909f1c04cbecc5e258d0f23478ecbf3745f"
PLAN = ROOT / "artifacts/jev-runtime/decision3-edge-browser-retry-plan-20261022-01/plan-02.json"
PLAN_SHA = "852b162a5d2a9b7c5f5516ce2ac913be5510491e6c85dd1603baaeb31d6db6ca"
PROFILE = "decision3-edge-external-scoped-continuation-v1"
PRIOR_NAMES = {
    "actor": "actor-result.json",
    "transport": "transport/closure.json",
    "outer": "external-boundary-closure.json",
    "final": "controller-final.json",
    "summary": "results/summary.json",
    "readback": "unit-readback.json",
    "admission": "admission.json",
}


def need(value, message):
    if not value:
        raise ValueError(message)


def read_selected(ref):
    need(set(ref) == {"path", "sha256", "size_bytes"}, "Exact selected reference")
    path = Path(ref["path"])
    need(not path.is_absolute() and ".." not in path.parts, "Repository-relative selection")
    path = ROOT / path
    need(not any(p.is_symlink() for p in (path, *path.parents)), "Ordinary selected path")
    need(0 < ref["size_bytes"] <= 16 * 1024**2, "Bounded small evidence/source")
    before = path.stat()
    blob = path.read_bytes()
    after = path.stat()
    need(
        all(
            getattr(before, k) == getattr(after, k)
            for k in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        )
        and len(blob) == ref["size_bytes"]
        and hashlib.sha256(blob).hexdigest() == ref["sha256"],
        "Selected bytes changed",
    )
    return path, blob


def load_execution():
    ref = {"path": str(EXECUTION.relative_to(ROOT)), "sha256": EXECUTION_SHA, "size_bytes": 57544}
    path, blob = read_selected(ref)
    spec = importlib.util.spec_from_file_location("_edge_scoped_original", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(blob, str(path), "exec", dont_inherit=True), module.__dict__)
    return module, blob


def protocol_complete(E, plan, group, actor, transport, summary_path):
    """Worker/protocol facts only; generic observation booleans remain unchanged."""
    need(
        actor.get("primary_error") is None and actor.get("cleanup_errors") == [],
        "Actor primary/cleanup failure",
    )
    need(
        actor.get("browser_exit_confirmed") is True and actor.get("model_child_started") is True,
        "Actual browser and worker required",
    )
    need(
        actor.get("group") == group and actor.get("plan_sha256") == PLAN_SHA,
        "Original selected arm result",
    )
    score = E.browser_score(summary_path, group)
    need(actor.get("score") == score, "Original complete browser score binding")
    need(
        transport.get("direct_child_exit_confirmed") is True
        and type(transport.get("returncode")) is int
        and transport["returncode"] == 0
        and transport.get("forced_cleanup") is False
        and transport.get("blocked") is False
        and transport.get("errors") == [],
        "Clean actual worker protocol exit required",
    )
    closed, identity = transport.get("closed"), transport.get("identity")
    need(type(closed) is dict and type(identity) is dict, "Authenticated CLOSED identity")
    need(
        all(closed.get(k) == v and type(closed.get(k)) is type(v) for k, v in identity.items()),
        "CLOSED worker identity differs",
    )
    expected = {
        "model": plan["model"],
        "nonce": plan["arms"][group]["nonce"],
        "boot_id": plan["boot_id"],
        "mem_total_bytes": plan["mem_total_bytes"],
        "gpu_uuid": plan["gpu_uuid"],
    }
    need(all(identity.get(k) == v for k, v in expected.items()), "Selected worker identity differs")
    receipt = closed.get("receipt")
    need(
        closed.get("event") == "closed"
        and closed.get("success") is True
        and closed.get("campaign_stop") is False
        and closed.get("error") is None
        and type(receipt) is dict
        and receipt.get("success") is True
        and receipt.get("errors") == []
        and receipt.get("source_cache_runtime_authenticated") is True
        and receipt.get("profile") == E.PROFILE
        and type(closed.get("envelopes_seen")) is int
        and closed["envelopes_seen"] == receipt.get("requests")
        and 0 <= closed["envelopes_seen"] <= E.GROUPS[group][1],
        "Successful final source/cache/runtime and original envelope accounting required",
    )
    return score


def scoped_group(E, plan, group, refs):
    """Derive a new scoped fact; never change the original actor/inner flags."""
    need(set(refs) == set(PRIOR_NAMES), "Complete bound group evidence")
    out = E.group_output(plan, group)
    rows = {}
    for key, name in PRIOR_NAMES.items():
        path, blob = read_selected(refs[key])
        need(path == out / name, "Exact original group evidence path")
        rows[key] = E.parse(blob)
    actor, inner, outer, final = (rows[k] for k in ("actor", "transport", "outer", "final"))
    score = protocol_complete(E, plan, group, actor, inner, out / PRIOR_NAMES["summary"])
    owner = E.load_module(plan["components"]["boundary"], "_edge_scoped_proof_owner")
    selection = E.unit_selection(plan, group, owner)
    readback, admission = rows["readback"], rows["admission"]
    need(
        outer.get("selection") == asdict(selection)
        and outer.get("invocation_id")
        == readback["unit"]["InvocationID"]
        == admission["invocation_id"]
        and bool(outer.get("invocation_id"))
        and admission.get("plan_sha256") == PLAN_SHA
        and admission.get("group") == group
        and admission.get("nonce") == selection.nonce
        and admission.get("readback_sha256") == refs["readback"]["sha256"],
        "Exact prior admitted unit/nonce/Invocation ownership",
    )
    need(
        all(
            outer["actor"][k] == readback["actor"][k]
            for k in ("pid", "start_ticks", "pgid", "sid", "uid", "cgroup")
        )
        and outer.get("whole_unit_exit_confirmed") is True
        and outer.get("observed_cgroup_cohort_exit_confirmed") is True
        and outer.get("sources_unchanged") is True
        and outer.get("cgroup_absent") is True
        and outer.get("post_stop_terminal_verified") is True
        and outer.get("closure_errors") == []
        and len(outer.get("two_process_scans", [])) == 2
        and all(
            s.get("matches") == [] and s.get("unknown") == [] for s in outer["two_process_scans"]
        )
        and final.get("closure") == outer,
        "Complete original external selected-unit/cohort/source closure",
    )
    # Reuse the original typed host predicate on the saved final observation.
    E.host_idle(plan, SimpleNamespace(observe=lambda _boot: final["final_host"]))
    for ref in plan["sources"]:
        E.read_ref(ref)
    return {
        "group": group,
        "score": score,
        "worker_protocol_success": True,
        "externally_scoped_group_complete": True,
        "original_actor_success": actor["actor_success"],
        "original_transport_success": inner["success"],
        "original_observed_process_exit_confirmed": inner["observed_process_exit_confirmed"],
        "evidence": copy.deepcopy(refs),
        "whole_tree_exit_confirmed": False,
        "no_escape_enforced": False,
        "model_qualified": False,
        "scope": "Original authenticated selected unit and observed cohort; inner flags are preserved.",
    }


def group_refs(E, plan, group):
    return {
        key: E.source_ref(E.group_output(plan, group) / name) for key, name in PRIOR_NAMES.items()
    }


def aggregate(plan, digest, known, groups, primary):
    combined = {"known4": known, **groups}
    complete = (
        primary is None
        and set(combined) == {"known4", "public6"}
        and all(r.get("externally_scoped_group_complete") is True for r in combined.values())
    )
    successes = sum((r.get("score") or {}).get("successes", 0) for r in combined.values())
    return {
        "profile": PROFILE,
        "model": plan["model"],
        "plan_sha256": digest,
        "completed": complete,
        "attempts_denominator": 10,
        "observed_complete_group_attempts": sum(
            (r.get("score") or {}).get("attempts", 0) for r in combined.values()
        ),
        "observed_successes": successes,
        "successes": successes if complete else None,
        "success_rate": successes / 10 if complete else None,
        "groups": combined,
        "primary_error": primary,
        "retained_known4_attempts": 4,
        "known4_rerun": False,
        "continued_group": "public6",
        "model_qualified": False,
        "whole_tree_exit_confirmed": False,
        "no_escape_enforced": False,
        "scope": "Explicit continuation of all original ten tasks, using selected unit/cohort closure.",
    }


def controller_source(original_blob):
    text = original_blob.decode()
    node = next(
        n
        for n in ast.parse(text).body
        if isinstance(n, ast.FunctionDef) and n.name == "controller_main"
    )
    original = "\n".join(text.splitlines()[node.lineno - 1 : node.end_lineno]) + "\n"
    edits = [
        (
            "def controller_main(plan, digest, blob):",
            "def continued_controller(plan, digest, blob):",
        ),
        ('    output = ordinary(plan["output"])', "    output = ordinary(_continuation_output)"),
        ("    for group in GROUPS:\n", '    for group in ("public6",):\n'),
        (
            '            require(result.get("actor_success") is True, "Actual selected group refused")\n',
            "            # Authoritative scoped completion is checked after external closure.\n",
        ),
        (
            "        if (\n            primary is not None",
            "        if primary is None:\n            try:\n                groups[group] = _scoped_completed(plan, group)\n            except BaseException as error:\n                primary = error_record(error)\n        if (\n            primary is not None",
        ),
        (
            "    report = aggregate_result(plan, digest, groups, primary)",
            "    report = _continued_aggregate(plan, digest, groups, primary)",
        ),
    ]
    derivative = original
    for before, after in edits:
        need(derivative.count(before) == 1, "Exact frozen controller continuation anchor")
        derivative = derivative.replace(before, after, 1)
    inverse = derivative
    for before, after in reversed(edits):
        need(inverse.count(after) == 1, "Exact inverse continuation anchor")
        inverse = inverse.replace(after, before, 1)
    need(inverse == original, "No unrelated controller change")
    return derivative, {
        "edits": edits,
        "inverse_bytes_equal": True,
        "source_sha256": hashlib.sha256(derivative.encode()).hexdigest(),
    }


def selected_spec(path, digest):
    blob = Path(path).read_bytes()
    need(
        len(blob) <= 1024**2 and hashlib.sha256(blob).hexdigest() == digest,
        "External continuation spec SHA",
    )
    spec = json.loads(blob)
    need(
        spec["profile"] == PROFILE
        and spec["run_group"] == "public6"
        and spec["retained_group"] == "known4",
        "Explicit original continuation only",
    )
    source, _ = read_selected(spec["controller"])
    need(source == Path(__file__).absolute(), "Exact continuation source")
    plan_path, _ = read_selected(spec["original_plan"])
    need(
        plan_path == PLAN and spec["original_plan"]["sha256"] == PLAN_SHA,
        "Fixed original Edge02 plan",
    )
    output = Path(spec["output"])
    need(output.parent == HERE and output.name == "continue-01", "New scoped continuation output")
    E, original_blob = load_execution()
    plan, _b, _inputs, _identities = E.selected_plan(PLAN, PLAN_SHA)
    known = scoped_group(E, plan, "known4", spec["retained_evidence"])
    return spec, E, plan, known, original_blob


def execute(spec_path, digest):
    spec, E, plan, known, original_blob = selected_spec(spec_path, digest)
    source, delta = controller_source(original_blob)
    namespace = dict(E.__dict__)
    namespace.update(
        _continuation_output=spec["output"],
        _scoped_completed=lambda selected, group: scoped_group(
            E, selected, group, group_refs(E, selected, group)
        ),
        _continued_aggregate=lambda selected, sha, groups, primary: aggregate(
            selected, sha, known, groups, primary
        ),
    )
    exec(
        compile(source, str(HERE / "selected-original-controller.py"), "exec", dont_inherit=True),
        namespace,
    )
    # Original guards, admissions, stop and final-host proof run unchanged.
    code = namespace["continued_controller"](plan, PLAN_SHA, PLAN.read_bytes())
    E.save(
        Path(spec["output"]) / "continuation-authority.json",
        {
            "spec_sha256": digest,
            "controller_delta": delta,
            "retained_evidence": spec["retained_evidence"],
            "known4_rerun": False,
        },
    )
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--spec")
    parser.add_argument("--spec-sha256")
    args = parser.parse_args(argv)
    if not args.execute:
        print(
            json.dumps(
                {
                    "mode": "DRY_SOURCE_ONLY",
                    "retained_group": "known4",
                    "run_group": "public6",
                    "fixed_denominator": 10,
                    "actual_started": False,
                    "model_qualified": False,
                }
            )
        )
        return 0
    if not args.spec or not args.spec_sha256:
        parser.error("Execution requires independently selected continuation spec and SHA")
    return execute(args.spec, args.spec_sha256)


if __name__ == "__main__":
    raise SystemExit(main())
