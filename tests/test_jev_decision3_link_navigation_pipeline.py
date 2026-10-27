"""Synthetic pipeline tests; these files do not assert real browser evidence."""

import copy
import json
from types import SimpleNamespace

import jev_decision3_link_navigation_collection as links
import jev_decision3_link_navigation_pipeline as pipeline
import jev_decision3_link_navigation_scenarios as scenarios
import jev_decision3_text_history_collection as base
import pytest

ORIGIN = "http://127.0.0.1:46789"


def _synthetic_exclusions(path):
    value = {
        "schema_id": base.EXCLUSION_SCHEMA,
        "normalization": base.NORMALIZATION,
        "goal_sha256": ["0" * 64],
        "scenario_ids": [],
        "ancestor_ids": [],
    }
    path.write_bytes(base._encode(value))
    return path


def test_plan_rebuild_binds_sources_exclusions_chromium_and_v7(tmp_path, monkeypatch):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    exclusions = _synthetic_exclusions(tmp_path / "exclusions.json")
    fake_source = {
        "scripts/jev_decision3_link_navigation_pipeline.py": {"sha256": "a" * 64, "bytes": 1}
    }
    monkeypatch.setattr(pipeline, "_source_inventory", lambda: fake_source)
    monkeypatch.setattr(pipeline.benchmark, "upstream_manifest", lambda _: {"revision": "pinned"})
    monkeypatch.setattr(
        pipeline.base,
        "_prepare_observation_profile",
        lambda *_: SimpleNamespace(
            materialized_manifest=lambda: {"v7": "reviewed", "artifact_files": {}}
        ),
    )
    monkeypatch.setattr(
        pipeline.base,
        "_executable_pin",
        lambda x: {"path": str(x), "resolved_path": str(x), "sha256": "b" * 64},
    )
    plan = pipeline.collection_plan(
        output_dir=tmp_path / "out",
        upstream=upstream,
        collection_id="unit",
        seed=7,
        exclusion_manifest_path=exclusions,
        chromium="/example/chromium",
        observation_helper_sha256="c" * 64,
    )
    assert len(plan["episodes"]) == 27 and not (tmp_path / "out").exists()
    assert plan["observation_profile_manifest"]["v7"] == "reviewed"
    manifest_bytes = (
        json.dumps(
            plan["observation_profile_manifest"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode()
    assert plan["observation_profile_manifest_sha256"] == base._sha(manifest_bytes)
    pipeline._verify_plan(plan, fresh=True)
    for field in ("sources", "exclusions", "chromium", "observation_profile_manifest", "episodes"):
        modified = copy.deepcopy(plan)
        if field == "episodes":
            modified[field].pop()
        else:
            modified[field]["modified"] = True
        with pytest.raises(ValueError, match="changed"):
            pipeline._verify_plan(modified, fresh=True)
    exclusions.write_bytes(base._encode({"bad": True}))
    with pytest.raises(ValueError):
        pipeline._verify_plan(plan, fresh=True)


def _wire():
    return {
        "model": base.MODEL_ALIAS,
        "state": {"page": {"title": "Synthetic"}},
        "questions": {
            "operation": {
                "type": "choice",
                "instructions": "choose",
                "criteria": {"CLICK": "click", "DONE": "done", "BLOCKED": "blocked"},
            },
            "click_target": {
                "type": "choice",
                "instructions": "choose",
                "criteria": {"1": {"element": "other"}, "2": {"element": "target"}},
            },
        },
    }


def _finish(spec):
    return {
        "pathname": spec["paths"]["detail"],
        "view": "detail",
        "ready": "complete",
        "heading": spec["record_label"],
        "status": spec["completed_text"],
        "count": 1,
        "links": [],
        "events": [],
    }


def _pack(output, path, value):
    target = output / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(base._encode(value))
    return base._sha(target.read_bytes())


def _synthetic_collection(tmp_path, monkeypatch):
    """Mock protocol replay boundaries; only test wiring, never real evidence."""
    spec = next(s for s in scenarios.scenarios(44, {"0" * 64}) if s["start_mode"] == "workspace")
    output = tmp_path / "out"
    output.mkdir()
    episode_id = "unit." + spec["id"]
    source = b"synthetic public source"
    source_path = output / "sources/scripts/synthetic.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(source)
    plan = {
        "schema_id": pipeline.SCHEMA,
        "output_dir": str(output),
        "upstream_path": "synthetic",
        "sources": {"scripts/synthetic.py": {"sha256": base._sha(source), "bytes": len(source)}},
        "observation_helper_sha256": "c" * 64,
        "observation_profile_manifest": {"v7": "test"},
        "observation_profile_manifest_sha256": base._sha(b"{}"),
        "episodes": [{"episode_id": episode_id, "scenario": spec}],
    }
    monkeypatch.setattr(pipeline, "_verify_plan", lambda *_args, **_kwargs: None)
    runtime = SimpleNamespace(read_state="READ", manifest=lambda: {"v7": "test"})
    monkeypatch.setattr(
        base,
        "_profile_module",
        lambda _: SimpleNamespace(verify_materialized_profile=lambda *_args, **_kwargs: runtime),
    )
    manifest = output / "observation-profile/profile-manifest.json"
    manifest.parent.mkdir()
    manifest.write_bytes(b"{}")
    monkeypatch.setattr(links, "_verify_ownership", lambda *_: None)
    monkeypatch.setattr(links, "_verify_step", lambda *_: {"synthetic": "proof"})
    start = {"pathname": spec["paths"]["workspace"]}
    detail = {"pathname": spec["paths"]["detail"]}
    end = _finish(spec)
    states = [(start, detail, "CLICK", "2"), (detail, end, "CLICK", "2"), (end, end, "DONE", None)]
    steps, rows, history = [], [], []
    for i, (before, after, op, target) in enumerate(states):
        after_history = history + ([{"index": i}] if op == "CLICK" else [])
        step = {
            "scenario": spec,
            "index": i,
            "before": before,
            "after": after,
            "page": {"url": ORIGIN + before["pathname"]},
            "page_after": {"url": ORIGIN + after["pathname"]},
            "history_before": history,
            "history_after": after_history,
            "teacher": {"operation": op},
            "payload": _wire(),
            "target": target,
            "proof": {"synthetic": "proof"},
        }
        steps.append(step)
        rows.append(
            links._verified_training_row(
                step["payload"],
                row_id=episode_id + f".step-{i:02d}",
                scenario_id=spec["id"],
                ancestor_id=spec["ancestor_id"],
                episode_id=episode_id,
                step=i,
                operation=op,
                target=target,
                proof_refs={"transition": episode_id + f"#step-{i}"},
            )
        )
        history = after_history
    trace = {
        "scenario": spec,
        "scenario_profile": links.DEFAULT_SCENARIO_PROFILE,
        "episode_id": episode_id,
        "schema_id": links.SCHEMA,
        "fit_eligible": False,
        "model_score": None,
        "status": "completed",
        "completed": True,
        "errors": [],
        "agent_closed": True,
        "cdp_start": 0,
        "cdp_end": 1,
        "steps": steps,
        "rows": rows,
        "network": [ORIGIN + spec["paths"]["workspace"], ORIGIN + spec["paths"]["detail"]],
        "observation_settling": {"fault": False, "records": [{"synthetic": True}]},
    }
    journal = [{"sequence": 0}]
    events = [
        {"url": value, "method": "GET", "redirected": False, "main_document": True}
        for value in trace["network"]
    ]
    prefix = "episodes/" + episode_id
    summary = {"episode_id": episode_id, "context_closed": True}
    for key, value in (("trace", trace), ("journal", journal), ("network", events)):
        summary[key] = prefix + "." + key + ".json"
        summary[key + "_sha256"] = _pack(output, summary[key], value)
    plan_sha = _pack(output, "plan.json", plan)
    receipt = {
        "schema_id": pipeline.SCHEMA,
        "plan_sha256": plan_sha,
        "status": "completed_pending_admission",
        "completed": True,
        "errors": [],
        "browser_closed": True,
        "sources_unchanged": True,
        "model_calls": 0,
        "model_score": None,
        "fit_eligible": False,
        "episodes": [summary],
    }
    receipt_sha = _pack(output, "receipt.json", receipt)
    return output, plan, receipt, trace, journal, events, plan_sha, receipt_sha


def test_independent_replay_preserves_target_mask_and_pending_fit(tmp_path, monkeypatch):
    output, _, _, _, _, _, plan_sha, receipt_sha = _synthetic_collection(tmp_path, monkeypatch)
    admitted = pipeline.admit_training_collection(
        output, expected_receipt_sha256=receipt_sha, expected_plan_sha256=plan_sha
    )
    rows = pipeline.compile_wire_projection(admitted)
    assert [row["loss_mask"]["click_target"] for row in rows] == [True, True, False]
    assert all(row["fit_eligible"] is False for row in rows)
    with pytest.raises(ValueError, match="alias"):
        pipeline.compile_wire_projection(admitted, model_alias="rewritten")
    with pytest.raises(ValueError, match="admission"):
        pipeline.compile_wire_projection(None)


@pytest.mark.parametrize(
    "change",
    ["mask", "wire", "incomplete", "network", "journal", "source", "history", "goal", "manifest"],
)
def test_independent_replay_rejects_tampering_even_after_updated_digests(
    tmp_path, monkeypatch, change
):
    output, _, receipt, trace, journal, events, plan_sha, _ = _synthetic_collection(
        tmp_path, monkeypatch
    )
    summary = receipt["episodes"][0]
    if change == "mask":
        trace["rows"][0]["loss_mask"]["click_target"] = False
    elif change == "wire":
        trace["rows"][0]["payload"]["questions"]["click_target"]["criteria"].pop("2")
    elif change == "incomplete":
        trace["completed"] = False
    elif change == "network":
        events[1]["redirected"] = True
    elif change == "journal":
        journal[0]["sequence"] = 9
    elif change == "source":
        (output / "sources/scripts/synthetic.py").write_bytes(b"changed")
    elif change == "manifest":
        (output / "observation-profile/profile-manifest.json").write_bytes(b"changed")
    elif change == "history":
        trace["steps"][1]["history_before"] = []
    else:
        trace["steps"][-1]["after"]["status"] = "unfinished"
    for key, value in (("trace", trace), ("journal", journal), ("network", events)):
        summary[key + "_sha256"] = _pack(output, summary[key], value)
    receipt_sha = _pack(output, "receipt.json", receipt)
    with pytest.raises((ValueError, KeyError)):
        pipeline.admit_training_collection(
            output, expected_receipt_sha256=receipt_sha, expected_plan_sha256=plan_sha
        )


def test_rejects_missing_or_wrong_receipt_pins(tmp_path, monkeypatch):
    output, _, _, _, _, _, plan_sha, receipt_sha = _synthetic_collection(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="SHA256"):
        pipeline.admit_training_collection(
            output, expected_receipt_sha256="f" * 64, expected_plan_sha256=plan_sha
        )
    with pytest.raises(ValueError, match="SHA256"):
        pipeline.admit_training_collection(
            output, expected_receipt_sha256=receipt_sha, expected_plan_sha256="f" * 64
        )
