"""Closed saved-file v8 checks; no browser, service or credential access."""

import hashlib
import importlib.util
import json
import math
import re
from collections import Counter
from pathlib import Path

PROFILE = "composed-visible-hover-entries-settled-v8"


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _json(path):
    return json.loads(Path(path).read_bytes())


def _finite(value):
    assert type(value) in (int, float) and math.isfinite(value) and value >= 0
    return value


def check_sidecar(value, observations, states):
    """Pure arithmetic/native snapshot join; timeout history never becomes task failure."""
    assert value["schema_id"] == "jev-v8-native-observation-settling-sidecar-v1"
    assert value["profile"] == PROFILE and value["complete"] is True
    assert value["timing_scope"] == "per_native_observe_operation_combined_not_individual_CDP"
    assert value["task_success_attested"] is False
    rows = value["records"]
    assert len(rows) <= 4096
    assert value["terminal_unknown"] is any(row["terminal_unknown"] for row in rows)
    native = Counter()
    for observation in observations:
        page = observation["response"].get("result", {}).get("value")
        if isinstance(page, dict):
            assert page["observation_profile"]["id"] == PROFILE
            assert page["w"] == 1120 and page["h"] == 780
            page = dict(page)
            page["fingerprint"] = _sha(
                json.dumps(
                    {k: page[k] for k in ("url", "text", "actions", "scroll")}, sort_keys=True
                ).encode()
            )
            native[
                _sha(json.dumps(page, sort_keys=True, ensure_ascii=False, allow_nan=False).encode())
            ] += 1
    available = native.copy()
    accepted = snapshots = nulls = 0
    statuses = Counter()
    previous_id, previous_end = None, None
    accepted_digests = set()
    for row in rows:
        call_id = row["observe_call_id"]
        assert type(call_id) is int and call_id > 0
        assert previous_id is None or call_id == previous_id + 1
        previous_id = call_id
        start, deadline, elapsed = (
            _finite(row[k]) for k in ("started_monotonic", "deadline_monotonic", "elapsed_seconds")
        )
        assert math.isclose(deadline - start, 5, abs_tol=1e-6)
        assert previous_end is None or start >= previous_end
        previous_end = start + elapsed
        assert row["target_seconds"] == 5 and row["scope"] == "per_observe_call"
        assert row["synchronous_cdp_hard_deadline"] is False
        assert row["operation_timing_scope"] == "native_observe_combined_not_individual_CDP"
        assert math.isclose(_finite(row["overshoot_seconds"]), max(0, elapsed - 5), abs_tol=1e-6)
        operations = row["operations"]
        assert row["attempts"] == len(operations) <= 1024
        status = row["status"]
        assert status in {"success", "timeout", "unexpected_error"}
        assert row["terminal_unknown"] is (status != "success")
        if status == "success":
            assert elapsed < 5 and not row["late_snapshot"] and operations
            assert operations[-1]["returned_snapshot_sha256"] is not None
            accepted_digests.add(operations[-1]["returned_snapshot_sha256"])
            accepted += 1
        elif status == "timeout":
            assert elapsed >= 5
        else:
            assert row["error_type"] is not None
        statuses[status] += 1
        for operation in operations:
            a, z, spent = (
                _finite(operation[k])
                for k in ("started_monotonic", "ended_monotonic", "elapsed_seconds")
            )
            assert start <= a <= z <= start + elapsed + 1e-6
            assert math.isclose(z - a, spent, abs_tol=1e-6)
            assert operation["native_protocol_call_count"] is None
            if operation["null_navigation"]:
                assert operation["error_type"] == "StalePage"
                assert operation["error"] == "Document is navigating"
                assert operation["returned_snapshot_sha256"] is None
                nulls += 1
            digest = operation["returned_snapshot_sha256"]
            if digest is not None:
                assert re.fullmatch(r"[0-9a-f]{64}", digest) and available[digest] > 0
                available[digest] -= 1
                snapshots += 1
    assert not any(available.values()), (
        "Every native snapshot reply belongs to a recorded operation"
    )
    for state in states:
        if state and state.get("page") is not None:
            page = state["page"]
            assert page["observation_profile"]["id"] == PROFILE
            digest = _sha(
                json.dumps(page, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
            )
            assert digest in accepted_digests, (
                "Saved Agent page must join an accepted native reply plus fingerprint"
            )
    return {
        "observe_records": len(rows),
        "statuses": dict(statuses),
        "native_snapshot_SHA_joins": snapshots,
        "accepted_observe_snapshots": accepted,
        "null_navigation_operations": nulls,
        "terminal_unknown": value["terminal_unknown"],
        "max_elapsed_seconds": max((x["elapsed_seconds"] for x in rows), default=0),
        "max_overshoot_seconds": max((x["overshoot_seconds"] for x in rows), default=0),
        "timeout_unknown_is_not_task_failure_or_recovery_rejection": True,
    }


def negative_helper_kind(Q, raw, outer, event_bytes, cache):
    """Selected504 after a valid candidate is neither delivered text nor inner failure."""
    if outer["candidate_status"] == 200:
        assert outer["selected_status"] == 504 and outer["outer_authorized"] is False
        assert outer["elapsed_ms"] >= 20000
        Q._validate_field_generation(
            raw["generation"], raw["prefix"], len(raw["forward_events"]), cache
        )
        Q._validate_field_events(event_bytes, raw["request_id"], raw["generation"])
        return "outer_deadline_after_valid_candidate"
    candidate = raw["candidate_response"]
    if outer["selected_status"] == outer["candidate_status"] == 422 and raw["prefix"] is None:
        assert raw["generation"] == {} and raw["forward_events"] == []
        assert raw["helper_phases"] == [] and candidate["error"]["type"] == "helper_schema"
        return "helper_schema"
    kind = Q._field_failure_kind(
        raw["generation"], raw["prefix"], len(raw["forward_events"]), raw["helper_phases"]
    )
    assert kind in {"context", "consumer", "deadline"}
    if kind == "context":
        Q._validate_field_events(event_bytes, raw["request_id"], raw["generation"], context=True)
    return kind


def review_profile_group(ROOT, root, launch, plan, group):
    """Only call after the parent confirms this whole browser group has closed."""
    config = _json(root / "results/configuration.json")
    status = _json(root / "results/observation-profile-status.json")
    expected = plan["browser_profile"]
    for key in (
        "id",
        "manifest_path",
        "manifest_sha256",
        "helper_source_sha256",
        "canonical_source_sha256",
        "runtime_source_sha256",
        "effective_snapshot_js_sha256",
        "marker_expression_sha256",
    ):
        assert config["observation_profile"][key] == expected[key]
    assert status["profile"] == config["observation_profile"]
    assert not config["observation_profile_integrity_errors"] and not status["integrity_errors"]
    assert status["failed"] is False
    for key, value in (("max_steps", 16), ("repeats", 1), ("seed", 20261014)):
        assert config[key] == value
    assert config["model_requested"] == config["text_model_requested"] == "generated-actions-none"
    manifest = _json(root / "runtime-bundle/bundle-manifest.json")
    assert manifest == launch["bundle_manifest"]
    assert manifest["case_module"] == ("original" if group["group"] == "known4" else "heldout")
    assert (
        manifest["profile"] == PROFILE
        and manifest["profile_helper_sha256"] == expected["helper_source_sha256"]
    )
    assert (
        manifest["source_delta"]["inverse_bytes_equal"]
        and manifest["source_delta"]["inverse_AST_equal"]
    )
    assert config["runner_sha256"] == manifest["files"]["runner.py"]["sha256"]
    for label, source_name in (
        ("wrapper", "jev_browser_profile_v8_benchmark.py"),
        ("profile", "jev_observation_profile_v8.py"),
    ):
        source = ROOT / "scripts" / source_name
        expected_source = next(
            x for x in plan["benchmark_source_refs"] if x["path"] == str(source.relative_to(ROOT))
        )
        assert _sha(source.read_bytes()) == expected_source["sha256"]
        spec = importlib.util.spec_from_file_location("_saved_v8_audit_" + label, source)
        module = importlib.util.module_from_spec(spec)
        import sys

        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        if label == "wrapper":
            module.verify_bundle(root / "runtime-bundle", manifest)
        else:
            module.verify_materialized_profile(
                root / "results/observation-profile",
                expected_manifest_sha256=expected["manifest_sha256"],
                expected_source_sha256=expected["helper_source_sha256"],
            )
    fixed = _json(ROOT / plan["fixed_case_manifest"]["path"])
    cases = next(x["cases"] for x in fixed["groups"] if x["group"] == group["group"])
    assert config["public_web_cases"] == [x["case"] for x in cases]
    reports = []
    for case_id in group["selected_case_ids_in_order"]:
        folder = root / "results/attempts" / (case_id + "-000")
        sidecar_path = folder / "navigation-settling.json"
        assert sidecar_path.stat().st_size <= 8 * 1024**2
        observations = [
            json.loads(x) for x in (folder / "observations.jsonl").read_bytes().splitlines()
        ]
        reports.append(
            {
                "case_id": case_id,
                **check_sidecar(
                    _json(sidecar_path),
                    observations,
                    [*_json(folder / "states.json"), _json(folder / "final_state.json")],
                ),
            }
        )
    return {
        "profile": PROFILE,
        "bundle_and_materialized_profile_exact": True,
        "benchmark_screenshots": False,
        "case_reports": reports,
        "scope": "Native observe operation timing, not individual CDP; timeout unknown history preserved. No LLM score derived from sidecar.",
    }
