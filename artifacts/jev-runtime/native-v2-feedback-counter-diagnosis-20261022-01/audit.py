"""Read saved Counter inputs/actions; replay pure serialization, no inference."""

import ast
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).absolute().parents[3]
HERE = Path(__file__).absolute().parent
CASE = "public-python-counter-most-common-v1-000"
BASES = {
    "control": ROOT / "artifacts/jev-runtime/native-v2-browser-execution-20261022-04/v2-01/browser",
    "feedback": ROOT / "artifacts/jev-runtime/browser-stale-feedback-bundle-20261022-01/v2-feedback-01/browser",
}


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def ref(path):
    blob = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(blob), "size_bytes": len(blob)}


def load(path):
    return json.loads(path.read_bytes())


def main():
    bootstrap = ROOT / "artifacts/jev-runtime/decision3-current-source-bootstrap-20261022-01/current_bootstrap.py"
    blob = bootstrap.read_bytes()
    assert sha(blob) == "a39e811764f83eff793f3f5ade7fb9defd7e21fdc7d86207b5e12ce7c22b8581"
    spec = importlib.util.spec_from_file_location("_counter_pure_bootstrap", bootstrap)
    boot = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = boot
    exec(compile(blob, str(bootstrap), "exec", dont_inherit=True), boot.__dict__)
    core = bootstrap.parent / "current_core.py"
    core_blob = core.read_bytes()
    assert sha(core_blob) == "13b77299787d77587de2253a5f953b101113fc8d694edd27ee651d709be0e435"
    pins = next(ast.literal_eval(n.value) for n in ast.parse(core_blob).body
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
                and t.id == "SOURCE_PINS" for t in n.targets))
    with boot.ProjectSources(pins):
        from s1.jev_generated import format_action_request
    model = BASES["control"] / "known4/results/observation-profile/runtime/jev_ultrafast/model.py"
    model_blob = model.read_bytes()
    assert model_blob == (BASES["feedback"] / "known4/results/observation-profile/runtime/jev_ultrafast/model.py").read_bytes()
    functions = [n for n in ast.parse(model_blob).body if isinstance(n, ast.FunctionDef)
                 and n.name in {"action_space", "validate_choice"}]
    namespace = {"math": math}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(model), "exec"), namespace)
    result, inputs, refs = {}, {}, [ref(model), ref(bootstrap), ref(core)]
    for arm, base in BASES.items():
        folder = base / "known4/results/attempts" / CASE
        calls = [c for c in load(folder / "model_calls.json") if c["kind"] == "policy"]
        states = load(folder / "states.json")
        raw = [json.loads(line) for line in (folder / "observations.jsonl").read_bytes().splitlines()]
        outcome = load(folder / "outcome.json")
        assert len(states) == len(calls) + 1
        rows = []
        for i, call in enumerate(calls):
            request, response = call["request"], call["response"]
            generation = response["generation"]
            assert len(call["transport_attempts"]) == 1
            assert call["transport_attempts"][0]["http_status"] == 200
            assert generation["phase"] == "validated" and generation["do_sample"] is False
            text, source, labels, representation = format_action_request(
                request, request["model"], prompt_format="shared-elements-v2")
            contract = generation["action_adapter"]["prompt_contract"]
            assert generation["prompt"] == contract["native_prefix"] + text + contract["native_suffix"]
            assert generation["original_request_sha256"] == sha(source)
            assert generation["prompt_representation"] == representation
            assert representation["format"] == "shared-elements-v2"
            assert generation["candidate_labels"] == labels
            assert json.loads(generation["raw_output"]) == generation["decoded_action"]
            action = generation["decoded_action"]
            operation, target = action["operation"], action["target"]
            assert response["answers"]["operation"]["choice"] == operation
            namespace["validate_choice"](response["answers"]["operation"], labels["operation"])
            target_head = {"CLICK": "click_target", "TYPE_TEXT": "type_text_target", "SELECT": "select_target"}.get(operation)
            if target_head:
                assert target in labels[target_head]
                namespace["validate_choice"](response["answers"][target_head], labels[target_head])
                assert response["answers"][target_head]["choice"] == target
            history = states[i]["history"]
            expected_recent = [{k: h.get(k) for k in ("action", "kind", "text", "page_changed")}
                               for h in history[-10:]]
            assert request["state"]["recent_actions"] == expected_recent
            matches = []
            choices = set()
            for observation in raw:
                page = observation["response"]["result"]["value"]
                if not isinstance(page, dict) or "actions" not in page:
                    continue
                elements, targets, controls = namespace["action_space"](page["actions"])
                if ({k: page[k] for k in ("url", "title", "text")} == request["state"]["page"]
                        and elements == request["state"]["elements"]):
                    matches.append(observation["sequence"])
                    choices.add(targets[operation][target]["id"] if target_head else
                                controls[operation]["id"] if operation in controls else operation)
            assert matches and len(choices) == 1
            choice = next(iter(choices))
            after = states[i + 1]["history"]
            delta = len(after) - len(history)
            assert delta in (0, 1)
            if delta:
                assert after[-1]["operation"] == operation and after[-1]["target"] == target
                assert after[-1]["choice"] == choice
            feedback = request["state"].get("action_feedback", [])
            rows.append({"policy_index": i, "request_sha256": sha(source),
                         "prompt_sha256": sha(generation["prompt"].encode()),
                         "matching_saved_observation_sequences": matches,
                         "offered_target_and_original_choice_validation_passed": True,
                         "decoded_action_sha256": sha(encode(action)),
                         "actual_choice_identity_sha256": sha(choice.encode()),
                         "history_before": len(history), "history_after": len(after),
                         "recorded_action_delta": delta,
                         "feedback_attempt_ids": [x["attempt"] for x in feedback],
                         "feedback_physical_input_outcome": sorted({x["physical_input_outcome"] for x in feedback}),
                         "generation_ms": generation["generation_elapsed_ms"],
                         "handler_ms": generation["elapsed_ms"],
                         "shared_v2_complete_prompt_replay_equal": True,
                         "page_and_candidate_input_sha256": sha(encode(request["state"])),
                         "all_client_status200": True})
        final = outcome["verification"]["observed"]
        geometry = {key: {k: final["python"][key][k] for k in
                         ("rect", "css_visible", "center_unobscured", "intersection_unobscured")}
                    for key in ("method", "description", "first_paragraph")}
        result[arm] = {"success": outcome["success"], "status": outcome["status"],
                       "error_present": outcome["error"] is not None,
                       "policy_calls": len(calls), "text_calls": outcome["text_calls"],
                       "recorded_actions": len(states[-1]["history"]),
                       "mismatch_keys": list(outcome["verification"].get("mismatches", {})),
                       "final_geometry": geometry, "rows": rows,
                       "identical_request_run": max((sum(1 for _ in group) for _key, group
                           in __import__("itertools").groupby(x["request_sha256"] for x in rows)), default=0)}
        inputs[arm] = calls
        refs.extend(ref(folder / n) for n in ("model_calls.json", "states.json", "observations.jsonl", "outcome.json"))
    equalities = []
    for i, (a, b) in enumerate(zip(inputs["control"], inputs["feedback"])):
        old, new = a["request"], b["request"]
        stripped = {**new, "state": {k: v for k, v in new["state"].items() if k != "action_feedback"}}
        equalities.append({"policy_index": i, "request_equal": old == new,
                           "without_feedback_equal": old == stripped,
                           "page_equal": old["state"]["page"] == new["state"]["page"],
                           "candidate_elements_equal": old["state"]["elements"] == new["state"]["elements"],
                           "history_equal": old["state"]["recent_actions"] == new["state"]["recent_actions"],
                           "questions_equal": old["questions"] == new["questions"]})
    assert all(x["request_equal"] for x in equalities[:2])
    assert equalities[2]["page_equal"] is False and not inputs["feedback"][2]["request"]["state"].get("action_feedback")
    assert result["control"]["success"] is True and result["feedback"]["success"] is False
    assert not any(n == "torch" or n.startswith("torch.") or n == "transformers" for n in sys.modules)
    report = {"schema_id": "counter-feedback-paired-diagnosis-v1", "audit_passed": True,
              "case_id": CASE, "arms": result, "paired_input_equalities": equalities,
              "findings": ["All13 policy calls were HTTP200, grammar-validated native generated actions with offered targets; exact stored shared-V2 prompt and original request replay passed. No invalid-target/format/default fallback or client HTTP failure is evidenced.",
                           "Both arms had two early attempted clicks with no new action-history entry; the feedback arm records two typed StalePage outcomes. Physical guard branch/input outcome remains unknown.",
                           "Both arms have no identical full-request repetition. Visible page text differs at policy2 before any feedback exists; policy3/4 retain equal offered elements, question descriptions and histories but differing page text plus feedback. The later choice divergence cannot isolate feedback causality.",
                           "Control finishes at the requested method anchor with visible signature/description. Feedback executes broader module/section navigation then native DONE; its exact method and first paragraph remain below the viewport (y1370/1404 at viewport780), so DONE is premature.",
                           "The actionable generic gap is evidence-sensitive completion and precise current-page progress; stale feedback delivery alone does not ensure either. Preserve denominator10, judges/candidate safety and existing action history; do not derive benchmark answer routes or labels."],
              "limitations": ["Single live-page pair; dynamic search rendering and guard timing were not controlled.",
                              "No tokenizer/model rerun or physical guard-branch reconstruction; actual saved prompt/token metadata is evidence only."],
              "evidence": refs, "auditor": ref(Path(__file__).absolute())}
    path = HERE / "summary.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"audit_passed": True, "summary": ref(path)}))


if __name__ == "__main__":
    main()
