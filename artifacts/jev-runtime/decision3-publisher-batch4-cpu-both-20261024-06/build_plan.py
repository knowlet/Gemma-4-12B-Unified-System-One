"""Bounded metadata-only full05 ECE-schema successor builder; no runtime imports."""

import ast
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "artifacts/jev-runtime/decision3-publisher-batch4-cpu-both-20261024-05"
OFFICIAL = ROOT / "artifacts/jevbench/20261022-local-score-01/upstream/heaven/jevbench"
PINS = {
    BASE / "campaign.py": "11d81a7e5e1010e23870667f936a29bb978e82ee0f5c5a264752ce3ae1cfd017",
    BASE / "edge-plan.json": "4bacfb19400f48425f334ab5f4f930a4063adfdab0ef4fe0276a1db2aa908b3e",
    BASE / "freeze.json": "ed9953b105e645c5e6440c4620e07684b158ffa68249ef62ee8bac9f9e8f7cc1",
    OFFICIAL / "metrics.py": "4038c8423ad1cf28601afb6babe777b71f08e379b9ca125b09b52c0ac7d60515",
    OFFICIAL / "summarize.py": "aa5f24da407d004cd80bad1cd957b6987aba152591084c096d1de51dd9775cbe",
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


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    for path, expected in PINS.items():
        assert ref(path)["sha256"] == expected
    old = json.loads((BASE / "edge-plan.json").read_bytes())
    plan = dict(old)
    plan.update(
        profile=old["profile"].replace(
            "both-stage-completions-", "both-stage-completions-ece-schema-"
        ),
        campaign_sha256=ref(HERE / "campaign.py")["sha256"],
        global_metric_projection={
            "public_ece": "Original official structured ECE object retained unchanged after exact typed global schema validation; no scalar assumption, recomputation, rounding or clamping.",
            "official_source_pins": [
                ref(OFFICIAL / name) for name in ("metrics.py", "summarize.py")
            ],
            "scope": "Output projection schema only; every original model/scorer/input/count/context/authentication/deadline invariant unchanged.",
        },
    )
    save(HERE / "edge-plan.json", plan)
    pending = json.loads((BASE / "pending-prerequisites.json").read_bytes())
    pending.update(
        profile=plan["profile"],
        required_future_action="Root must select exact genuine native03 smoke and independent closed audit and fresh full06 output/cache; no old full05 score or completion inherits. Source-only null prerequisite slots cannot qualify execution.",
    )
    save(HERE / "pending-prerequisites.json", pending)
    original = (BASE / "campaign.py").read_text()
    current = (HERE / "campaign.py").read_text()
    before, after = ast.parse(original), ast.parse(current)

    def definitions(tree):
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }

    old_defs, new_defs = definitions(before), definitions(after)
    changed, added = {"profile", "stage_metrics"}, {"project_public_ece"}
    assert set(new_defs) == set(old_defs) | added
    identical = sorted(set(old_defs) - changed)
    assert all(old_defs[name] == new_defs[name] for name in identical)

    def remainder(tree, excluded):
        return [
            ast.dump(node, include_attributes=False)
            for node in tree.body
            if not (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in excluded)
        ]

    assert remainder(before, changed) == remainder(after, changed | added)
    (HERE / "campaign.patch").write_text(
        "".join(
            difflib.unified_diff(
                original.splitlines(True),
                current.splitlines(True),
                fromfile="frozen-full05/campaign.py",
                tofile="ece-schema-full06/campaign.py",
            )
        )
    )
    save(
        HERE / "equivalence.json",
        {
            "schema": "native-batch4-full05-to-full06-ece-projection-equivalence-v1",
            "base": ref(BASE / "campaign.py"),
            "successor": ref(HERE / "campaign.py"),
            "unchanged_definitions": identical,
            "changed_existing_definitions": sorted(changed),
            "added_definitions": sorted(added),
            "all_other_top_level_AST_identical": True,
            "execute_suites_TimedPredictor_StageCompletions_DeadlineScope_authentication_scoring_and_finalization_AST_identical": True,
            "plan_changed_keys": sorted(
                key for key in set(old) | set(plan) if old.get(key) != plan.get(key)
            ),
            "all_native_owner_policy_model_math_counts_context_and_budgets_unchanged": True,
            "complete_byte_delta": ref(HERE / "campaign.patch"),
            "verified_small_source_inputs": [ref(path) for path in PINS],
            "actual_state_tasks_gold_responses_caches_model_native_payload_read_or_runtime_execution": False,
            "new_campaign_actual_qualification": None,
        },
    )
    print(
        json.dumps(
            {
                "status": "METADATA_BUILDER_AND_AST_PASS_NO_MODEL",
                "campaign": ref(HERE / "campaign.py"),
                "plan": ref(HERE / "edge-plan.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
