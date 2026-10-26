"""Small metadata-only port builder; no candidate imports, model or actual reads."""

import ast
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "artifacts/jev-runtime/decision3-publisher-batch4-cpu-both-20261023-04"
NATIVE = ROOT / "artifacts/jev-runtime/decision3-publisher-batch4-cpu-20261024-03"
POLICY = "comfy-observed-cpu-coexist-v1"
PINS = {
    BASE / "campaign.py": "a96dd36184db3c54fab7a7bfa397dbce89ea1737119f33e5cfc53572ffe52747",
    BASE / "edge-plan.json": "363f59055ad7c830e3728fed1f6e262534161fbc0273c721060de64d4cd5df96",
    BASE / "freeze.json": "6ad7726a720990b3ccdf322510aace93137a14782eb95b656e6a06a80ebced31",
    NATIVE / "actor.py": "4a08be7703d8559f5dc86593f0e3138e9c9546ea1cc4613a951ed796a13d9d3f",
    NATIVE
    / "resource_helper.py": "7d90f15adee339ed292510d3f5c10c0b33b505279de32643fe1c3da425554552",
    NATIVE / "source_audit.py": "ae299d87cbf7338bb9ff29055efc335e96a44b73a29370e08428aad2fb2b68d3",
    NATIVE / "edge-plan.json": "5efcef370b0ae09a78d334d6e00c4a4e39ebbaae9ad18fd822581e8979e1b6e4",
    NATIVE / "freeze.json": "a3c5a2adeea6e1b18650d63fd2f1cb09ec5d8cbd0b204f2e1196de4fe234ea1f",
}


def ref(path):
    with path.open("rb") as stream:
        raw = stream.read(1048577)
    assert 0 < len(raw) <= 1048576
    return {
        "path": str(path.relative_to(ROOT)),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def load(path):
    return json.loads(path.read_bytes())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    for path, expected in PINS.items():
        assert ref(path)["sha256"] == expected
    native = load(NATIVE / "edge-plan.json")
    old = load(BASE / "edge-plan.json")
    plan = dict(old)
    assert native["model"] == old["model"]
    assert native["host_observation_policy"] == POLICY
    assert native["owner_unit"] == "jev-d3-llama-cpp-cpu4-2g-coexist-20261024-07.service"
    assert native["owner_profile"] == "llama-cpp-cpu4-2g-coexist-command-owned-v1"
    plan.update(
        profile="decision3-edge-gx10-publisher-cpu2g-coexist-batch4-both-stage-completions-20261024-v1",
        campaign_sha256=ref(HERE / "campaign.py")["sha256"],
        batch_actor_sha256=PINS[NATIVE / "actor.py"],
        batch_freeze_sha256=PINS[NATIVE / "freeze.json"],
        native_plan_sha256=PINS[NATIVE / "edge-plan.json"],
        owner_unit=native["owner_unit"],
        owner_profile=native["owner_profile"],
        owner_sources=native["owner_sources"],
        host_observation_policy=POLICY,
        local_head_at_creation=native.get("local_head_at_creation", old["local_head_at_creation"]),
        prerequisite={
            **old["prerequisite"],
            "selection": "Future root selection must provide genuinely new owner07/native03 smoke state and independent accepted closed audit. No owner06/idle-only or other profile fit, authentication, timing or quality transfers.",
            "native_plan_sha256": PINS[NATIVE / "edge-plan.json"],
            "required_host_observation_policy": POLICY,
        },
        score_scope="Fresh Edge native03 batch4 CPU2GiB coexist candidate only. No prior owner06/idle-only or3GiB results, fit, performance, authentication, scores, batch1 or browser inheritance.",
        execution_selection="CANDIDATE_ONLY_PENDING_GENUINE_OWNER07_NATIVE03_SMOKE_AND_CLOSED_AUDIT; fresh root absence/fullR+entire2GiB admission required under explicit CPU-coexist policy.",
    )
    assert plan["whole_campaign_seconds"] == 10000
    assert plan["per_call_seconds"] == 180 and plan["owner_runtime_seconds"] == 10800
    assert plan["fresh_calls"] == 1479 and plan["context_limit"] == 16384
    assert plan["suites"] == old["suites"]
    assert plan["owner_sources"] == native["owner_sources"]
    assert json.loads(json.dumps(plan))["owner_sources"] == native["owner_sources"]
    save(HERE / "edge-plan.json", plan)
    pending = load(BASE / "pending-prerequisites.json")
    pending.update(
        profile=plan["profile"],
        owner_profile=native["owner_profile"],
        owner_unit=native["owner_unit"],
        host_observation_policy=POLICY,
        new_native_plan_sha256=PINS[NATIVE / "edge-plan.json"],
        new_native_profile=native["profile"],
        prior_owner06_qualification_reused=False,
        required_future_action="Root must run genuinely new owner07/native03 smoke, authenticate its original closure and independent audit including host_observation_policy. Pending nulls cannot execute; root later supplies exact accepted CLI refs without mutating this source record.",
    )
    assert pending["smoke_receipt"] is pending["closed_independent_audit"] is None
    assert pending["actual2GiB_smoke_or_audit_available"] is False
    save(HERE / "pending-prerequisites.json", pending)
    original = (BASE / "campaign.py").read_text()
    current = (HERE / "campaign.py").read_text()
    before = ast.parse(original)
    after = ast.parse(current)

    def definitions(tree):
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }

    old_defs, new_defs = definitions(before), definitions(after)
    changed = {"profile", "load_plan", "check_smoke", "check_audit"}
    assert set(old_defs) == set(new_defs)
    identical = sorted(set(old_defs) - changed)
    assert all(old_defs[name] == new_defs[name] for name in identical)
    allowed_names = {
        "BATCH",
        "BATCH_ACTOR_SHA",
        "BATCH_FREEZE_SHA",
        "NATIVE_PLAN_SHA",
        "HOST_OBSERVATION_POLICY",
    }

    def unchanged_nodes(tree):
        return [
            ast.dump(node, include_attributes=False)
            for node in tree.body
            if not (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in changed)
            and not (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id in allowed_names
                    for target in node.targets
                )
            )
        ]

    assert unchanged_nodes(before) == unchanged_nodes(after)
    (HERE / "campaign.patch").write_text(
        "".join(
            difflib.unified_diff(
                original.splitlines(True),
                current.splitlines(True),
                fromfile="frozen-full04/campaign.py",
                tofile="coexist-full05/campaign.py",
            )
        )
    )
    proof = {
        "schema": "native-batch4-coexist-campaign-source-port-equivalence-v1",
        "base": ref(BASE / "campaign.py"),
        "successor": ref(HERE / "campaign.py"),
        "unchanged_definitions": identical,
        "changed_existing_definitions": sorted(changed),
        "changed_or_added_top_level_bindings": sorted(allowed_names),
        "all_other_top_level_AST_identical": True,
        "original_execute_run_suites_timing_primary_stage_projection_and_durability_AST_identical": True,
        "original_benchmark_source_plan_sha256_unchanged": plan["benchmark_source_plan_sha256"],
        "original_suites_counts_order_model_math_context_and_budgets_unchanged": True,
        "complete_byte_delta": ref(HERE / "campaign.patch"),
        "plan_changed_keys": sorted(
            key for key in set(old) | set(plan) if old.get(key) != plan.get(key)
        ),
        "new_native_owner_metadata_copied_with_exact_python_JSON_integers": True,
        "verified_small_source_inputs": [ref(path) for path in PINS],
        "no_candidate_import_or_test_model_payload_task_gold_response_or_live_probe": True,
        "new_owner07_native03_actual_smoke_or_audit_qualification": None,
    }
    save(HERE / "equivalence.json", proof)
    print(
        json.dumps(
            {
                "status": "METADATA_ONLY_PLAN_AND_SOURCE_EQUALITY_PROOF_WRITTEN_TESTS_HELD",
                "campaign": ref(HERE / "campaign.py"),
                "plan": ref(HERE / "edge-plan.json"),
                "proof": ref(HERE / "equivalence.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
