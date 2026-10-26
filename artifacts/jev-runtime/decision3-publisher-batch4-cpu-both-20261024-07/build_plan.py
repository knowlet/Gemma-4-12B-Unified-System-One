"""Metadata-only CPU16/4GiB successor builder; HELD until current NN is closed."""

import ast
import copy
import difflib
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "artifacts/jev-runtime/decision3-publisher-batch4-cpu-both-20261024-06"
NATIVE = ROOT / "artifacts/jev-runtime/decision3-publisher-batch4-cpu-20261024-04"
PRIOR_NATIVE = ROOT / "artifacts/jev-runtime/decision3-publisher-batch4-cpu-20261024-03"
PINS = {
    BASE / "campaign.py": "81113f1e2e1f1342c547b156e307ef78844c5947a9fa47f85fb054579eb4d966",
    BASE / "edge-plan.json": "351500bcdfdd0b1c3c6096b21bd767321e761a0af79a64ed080dad518d9d575c",
    BASE / "freeze.json": "a10ece98ee8f5e9cfdc3dc00facc661cfb77f12ae2d92dc1c682bf5f6218e9ea",
    PRIOR_NATIVE / "edge-plan.json": "5efcef370b0ae09a78d334d6e00c4a4e39ebbaae9ad18fd822581e8979e1b6e4",
}


def ref(path):
    assert path.suffix in {".py", ".json"}
    with path.open("rb") as stream:
        raw = stream.read(1048577)
    assert 0 < len(raw) <= 1048576
    raw.decode("utf-8")
    return {"path": str(path.relative_to(ROOT)), "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def definitions(tree):
    return {node.name: ast.dump(node, include_attributes=False) for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def main():
    for path, expected in PINS.items():
        assert ref(path)["sha256"] == expected
    original = (BASE / "campaign.py").read_text()
    current = (HERE / "campaign.py").read_text()
    before, after = ast.parse(original), ast.parse(current)
    old_defs, new_defs = definitions(before), definitions(after)
    changed = {"profile", "load_plan"}
    assert set(old_defs) == set(new_defs)
    unchanged = sorted(set(old_defs) - changed)
    assert all(old_defs[name] == new_defs[name] for name in unchanged)
    changing_constants = {"BATCH", "BATCH_ACTOR_SHA", "BATCH_FREEZE_SHA", "NATIVE_PLAN_SHA"}

    def remainder(tree):
        result = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in changed:
                continue
            if isinstance(node, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id in changing_constants
                    for target in node.targets):
                continue
            result.append(ast.dump(node, include_attributes=False))
        return result

    assert remainder(before) == remainder(after)
    constants = {target.id: ast.literal_eval(node.value)
                 for node in after.body if isinstance(node, ast.Assign)
                 for target in node.targets if isinstance(target, ast.Name)
                 and target.id in changing_constants - {"BATCH"}}
    assert set(constants) == changing_constants - {"BATCH"}
    actual_pins = {
        "BATCH_ACTOR_SHA": ref(NATIVE / "actor.py")["sha256"],
        "BATCH_FREEZE_SHA": ref(NATIVE / "freeze.json")["sha256"],
        "NATIVE_PLAN_SHA": {"edge": ref(NATIVE / "edge-plan.json")["sha256"]},
    }
    assert all(re.fullmatch(r"[0-9a-f]{64}", value)
               for value in [constants["BATCH_ACTOR_SHA"], constants["BATCH_FREEZE_SHA"],
                             constants["NATIVE_PLAN_SHA"]["edge"]])
    assert constants == actual_pins
    native = json.loads((NATIVE / "edge-plan.json").read_text())
    old_native = json.loads((PRIOR_NATIVE / "edge-plan.json").read_text())
    assert all(native[key] == old_native[key]
               for key in ["model", "model_files", "runtime_files", "manifest", "snapshot"])
    assert native["memory_max_bytes"] == 4294967296
    assert native["startup_available_bytes"] == 30414140212
    assert native["reserve_bytes"] == 26119172916
    assert native["cpu_affinity"] == list(range(4, 20))
    assert native["cpu_quota_percent"] == 1600 and native["threads"] == 16
    old = json.loads((BASE / "edge-plan.json").read_text())
    plan = copy.deepcopy(old)
    plan.update(
        profile="decision3-edge-gx10-publisher-cpu16-4g-coexist-batch4-both-stage-completions-ece-schema-20261024-v1",
        campaign_sha256=ref(HERE / "campaign.py")["sha256"],
        batch_actor_sha256=actual_pins["BATCH_ACTOR_SHA"],
        batch_freeze_sha256=actual_pins["BATCH_FREEZE_SHA"],
        native_plan_sha256=actual_pins["NATIVE_PLAN_SHA"]["edge"],
    )
    for key in ["model", "owner_unit", "owner_profile", "owner_sources", "reserve_bytes",
                "memory_max_bytes", "startup_available_bytes", "swap_max_bytes", "threads",
                "cpu_affinity", "cpu_quota_percent", "host_observation_policy"]:
        plan[key] = native[key]
    plan["prerequisite"].update(
        selection="Fresh owner08/native04 genuine CPU16/4GiB smoke and accepted independent closed audit required; no old profile qualification transfer.",
        native_plan_sha256=plan["native_plan_sha256"],
    )
    plan["execution_selection"] = "CANDIDATE_ONLY_PENDING_GENUINE_OWNER08_NATIVE04_SMOKE_AND_CLOSED_AUDIT; fresh absence/fullR+entire4GiB admission required."
    plan["score_scope"] = "Fresh Edge native04 batch4 CPU16/4GiB profile only. No prior fit, quality, scores or browser inheritance; CPU-vs-GPU timing is not a speed comparison."
    save(HERE / "edge-plan.json", plan)
    patch = "".join(difflib.unified_diff(original.splitlines(True), current.splitlines(True),
                    fromfile="frozen-full06/campaign.py", tofile="cpu16-4g-full07/campaign.py"))
    with (HERE / "campaign.patch").open("x") as stream:
        stream.write(patch)
    save(HERE / "equivalence.json", {
        "schema": "full06-to-full07-resource-only-equivalence-v1",
        "base": ref(BASE / "campaign.py"), "successor": ref(HERE / "campaign.py"),
        "unchanged_definitions": unchanged,
        "changed_definitions": sorted(changed),
        "changed_top_level_constants": sorted(changing_constants),
        "all_other_AST_identical": True,
        "execute_scoring_ECE_stages_callbacks_context_deadlines_and_finalization_AST_identical": True,
        "native_model20_runtime_manifest_and_snapshot_metadata_copied_exact": True,
        "plan_changed_keys": sorted(key for key in set(old) | set(plan) if old.get(key) != plan.get(key)),
        "source_inputs": [ref(path) for path in PINS],
        "delta": {
            "path": str((HERE / "campaign.patch").relative_to(ROOT)),
            "sha256": hashlib.sha256(patch.encode()).hexdigest(),
            "size_bytes": len(patch.encode()),
        },
        "actual_state_tasks_gold_responses_caches_model_native_payload_or_runtime_execution": False,
        "actual_new_profile_qualification": None,
    })
    print(json.dumps({"status": "METADATA_AND_AST_PASS_NO_MODEL_OR_FIT_CLAIM",
                      "plan": ref(HERE / "edge-plan.json")}))


if __name__ == "__main__":
    main()
