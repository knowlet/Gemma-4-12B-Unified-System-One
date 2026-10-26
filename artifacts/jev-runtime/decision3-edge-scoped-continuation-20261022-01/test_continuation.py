"""CPU-only saved-evidence and fake-owner checks; no actual launch or HTTP."""

import ast
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location("_test_scoped_continuation", HERE / "continuation.py")
C = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = C
exec(
    compile((HERE / "continuation.py").read_bytes(), str(HERE / "continuation.py"), "exec"),
    C.__dict__,
)


@pytest.fixture
def saved():
    E, source = C.load_execution()
    plan = json.loads(C.PLAN.read_bytes())
    original = E.group_output(plan, "known4")
    with tempfile.TemporaryDirectory(prefix=".cpu-scoped-", dir=HERE) as directory:
        folder = Path(directory)
        plan["output"] = str(folder / "original-output")
        out = E.group_output(plan, "known4")
        for name in C.PRIOR_NAMES.values():
            target = out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original / name, target)

        def update(key, callback):
            target = out / C.PRIOR_NAMES[key]
            row = json.loads(target.read_bytes())
            callback(row)
            target.write_text(json.dumps(row))
            if key == "outer":
                final = out / C.PRIOR_NAMES["final"]
                value = json.loads(final.read_bytes())
                value["closure"] = row
                final.write_text(json.dumps(value))
            return C.group_refs(E, plan, "known4")

        yield SimpleNamespace(E=E, source=source, plan=plan, out=out, folder=folder, update=update)


def test_actual_known4_is_retained_without_changing_inner_false_flags(saved):
    before = {
        name: hashlib.sha256((saved.out / name).read_bytes()).hexdigest()
        for name in C.PRIOR_NAMES.values()
    }
    result = C.scoped_group(
        saved.E, saved.plan, "known4", C.group_refs(saved.E, saved.plan, "known4")
    )
    assert result["score"]["attempts"] == 4 and result["score"]["successes"] == 0
    assert result["worker_protocol_success"] is True
    assert result["externally_scoped_group_complete"] is True
    assert result["original_actor_success"] is False
    assert result["original_transport_success"] is False
    assert result["original_observed_process_exit_confirmed"] is False
    assert result["whole_tree_exit_confirmed"] is False
    assert result["no_escape_enforced"] is False
    assert before == {
        name: hashlib.sha256((saved.out / name).read_bytes()).hexdigest()
        for name in C.PRIOR_NAMES.values()
    }


@pytest.mark.parametrize(
    "fault",
    [
        "primary",
        "cleanup",
        "browser-live",
        "model-missing",
        "returncode",
        "direct-exit",
        "forced",
        "blocked",
        "transport-error",
        "closed-error",
        "closed-failed",
        "campaign-stop",
        "cache-runtime",
        "receipt-error",
        "count",
        "identity",
        "unit",
        "invocation",
        "outer-unit",
        "outer-cohort",
        "outer-source",
        "outer-cgroup",
        "outer-terminal",
        "outer-error",
        "outer-scans",
        "host-busy",
        "host-low",
        "source-hash",
        "summary-judge",
    ],
)
def test_protocol_or_outer_proof_failure_cannot_be_reconciled(saved, fault):
    def actor(row):
        if fault == "primary":
            row["primary_error"] = {"type": "RuntimeError"}
        elif fault == "cleanup":
            row["cleanup_errors"] = [{"stage": "listener_close"}]
        elif fault == "browser-live":
            row["browser_exit_confirmed"] = False
        else:
            row["model_child_started"] = False

    def transport(row):
        if fault == "returncode":
            row["returncode"] = 1
        elif fault == "direct-exit":
            row["direct_child_exit_confirmed"] = False
        elif fault == "forced":
            row["forced_cleanup"] = True
        elif fault == "blocked":
            row["blocked"] = True
        elif fault == "transport-error":
            row["errors"] = [{"stage": "closed_frame"}]
        elif fault == "closed-error":
            row["closed"]["error"] = {"type": "RuntimeError"}
        elif fault == "closed-failed":
            row["closed"]["success"] = False
        elif fault == "campaign-stop":
            row["closed"]["campaign_stop"] = True
        elif fault == "cache-runtime":
            row["closed"]["receipt"]["source_cache_runtime_authenticated"] = False
        elif fault == "receipt-error":
            row["closed"]["receipt"]["errors"] = [{"stage": "final_cache"}]
        elif fault == "count":
            row["closed"]["envelopes_seen"] += 1
        else:
            row["closed"]["pid"] += 1

    def outer(row):
        if fault == "unit":
            row["selection"]["unit"] = "jev-d3-unselected.service"
        elif fault == "invocation":
            row["invocation_id"] = "0" * 32
        elif fault == "outer-error":
            row["closure_errors"] = [{"type": "PermissionError"}]
        elif fault == "outer-scans":
            row["two_process_scans"][1]["unknown"] = [{"pid": 1}]
        else:
            row[
                {
                    "outer-unit": "whole_unit_exit_confirmed",
                    "outer-cohort": "observed_cgroup_cohort_exit_confirmed",
                    "outer-source": "sources_unchanged",
                    "outer-cgroup": "cgroup_absent",
                    "outer-terminal": "post_stop_terminal_verified",
                }[fault]
            ] = False

    if fault in {"primary", "cleanup", "browser-live", "model-missing"}:
        saved.update("actor", actor)
    elif fault in {
        "returncode",
        "direct-exit",
        "forced",
        "blocked",
        "transport-error",
        "closed-error",
        "closed-failed",
        "campaign-stop",
        "cache-runtime",
        "receipt-error",
        "count",
        "identity",
    }:
        saved.update("transport", transport)
    elif fault.startswith("outer") or fault in {"unit", "invocation"}:
        saved.update("outer", outer)
    elif fault.startswith("host"):
        saved.update(
            "final",
            lambda r: r["final_host"].update(
                {"comfy_running": 1} if fault == "host-busy" else {"mem_available_bytes": 1}
            ),
        )
    elif fault == "source-hash":
        saved.plan["sources"][0]["sha256"] = "0" * 64
    else:
        saved.update("summary", lambda r: r["outcomes"][0].update(success=True))
    with pytest.raises((ValueError, KeyError)):
        C.scoped_group(saved.E, saved.plan, "known4", C.group_refs(saved.E, saved.plan, "known4"))


def test_controller_is_inverse_exact_and_selects_only_unused_public6(saved):
    source, delta = C.controller_source(saved.source)
    assert delta["inverse_bytes_equal"] is True and len(delta["edits"]) == 6
    tree = ast.parse(source)
    loops = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "group"
    ]
    assert len(loops) == 1 and ast.literal_eval(loops[0].iter) == ("public6",)
    assert 'require(result.get("actor_success") is True' not in source
    assert source.index("closure = boundary.close") < source.index(
        "groups[group] = _scoped_completed"
    )
    for original_guard in [
        "host_idle(plan, observer)",
        "Fresh exact unit absent",
        "Fresh selected cgroup absent",
        "Fresh separate model output",
        "boundary.owned_status()",
        'closure.get("sources_unchanged") is True',
    ]:
        assert original_guard in source
    with pytest.raises(ValueError):
        C.controller_source(
            saved.source.replace(b"    for group in GROUPS:\n", b"    for group in (): # drift\n")
        )


def test_partial_public6_never_creates_a_ten_task_score(saved):
    known = C.scoped_group(
        saved.E, saved.plan, "known4", C.group_refs(saved.E, saved.plan, "known4")
    )
    partial = C.aggregate(saved.plan, C.PLAN_SHA, known, {}, {"type": "Public6NotComplete"})
    assert partial["completed"] is False and partial["success_rate"] is None
    assert partial["observed_complete_group_attempts"] == 4 and partial["observed_successes"] == 0
    public = {"externally_scoped_group_complete": True, "score": {"attempts": 6, "successes": 2}}
    full = C.aggregate(saved.plan, C.PLAN_SHA, known, {"public6": public}, None)
    assert full["completed"] is True and full["success_rate"] == 0.2
    assert full["observed_complete_group_attempts"] == full["attempts_denominator"] == 10
    assert full["known4_rerun"] is False


def test_failed_public6_null_score_keeps_original_failure_and_incomplete_receipt(saved):
    known = C.scoped_group(
        saved.E, saved.plan, "known4", C.group_refs(saved.E, saved.plan, "known4")
    )
    primary = {"type": "TimeoutError"}
    failed = {"actor_success": False, "score": None, "primary_error": primary}
    report = C.aggregate(saved.plan, C.PLAN_SHA, known, {"public6": failed}, primary)
    assert report["completed"] is False
    assert report["primary_error"] == primary
    assert report["observed_complete_group_attempts"] == 4
    assert report["observed_successes"] == 0
    assert report["successes"] is None and report["success_rate"] is None
    assert report["groups"]["known4"] == known
    assert report["groups"]["public6"] == failed


def test_default_dry_cannot_load_or_launch(monkeypatch, capsys):
    monkeypatch.setattr(C, "load_execution", lambda: pytest.fail("No project import"))
    monkeypatch.setattr(C, "execute", lambda *_: pytest.fail("No actor"))
    assert C.main([]) == 0
    assert json.loads(capsys.readouterr().out)["actual_started"] is False
    with pytest.raises(SystemExit):
        C.main(["--execute"])
