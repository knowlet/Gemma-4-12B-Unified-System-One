"""Source-only v7 -> v8 observer derivative; no live binding or execution."""

import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
OLD = ROOT / "artifacts/jev-runtime/public-field-syntax-v7-browser-retest-20261018-01"
FORMAL = ROOT / "artifacts/jev-runtime/public-native-hover-settled-v8-20261019-01"
CAPACITY = ROOT / "artifacts/jev-runtime/public-v8-capacity-preflight-20261019-02"
PROFILE = "composed-visible-hover-entries-settled-v8"


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def ref(path):
    blob = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(blob), "size_bytes": len(blob)}


def save(path, blob):
    with path.open("xb") as stream:
        stream.write(blob)


def save_json(path, value):
    save(path, (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode())


def main():
    assert not (BASE / "source-HOLD-01").exists(), "Frozen derivative must never be rebuilt"
    for filename, expected in {
        "plan-01.json": "7ed4b3c9b8de46eaee0b53d78e324621986cc599e9f99f813f51185750fb3416",
        "browser_observer_runner.py": "0aa07134ea272c2a62b6534e3a3506b9e2006e9886bc506a210b349e4b03c960",
        "root_owned_group_controller.py": "1b6b83b1e8338d5051f9e217996a219f3c08de89d8746a4c435e97bb760633f1",
        "review_v7_profile_evidence.py": "5538b71d4e409b6236692868cee1654960e1571764fb5b251361029d6179178e",
    }.items():
        assert ref(OLD / filename)["sha256"] == expected
    selection_ref = ref(CAPACITY / "root-selected-capacity-inputs-01.json")
    assert (
        selection_ref["sha256"]
        == "724a74e741db5f7ad1b88a0bc91a0b9ed778931e344983b581f5ba7cfd386995"
    )
    selection = json.loads((CAPACITY / "root-selected-capacity-inputs-01.json").read_bytes())
    capacity_plan = json.loads((CAPACITY / "plan-01.json").read_bytes())
    materialized = json.loads((FORMAL / "profile-01/profile-manifest.json").read_bytes())
    helper_ref = ref(ROOT / "scripts/jev_observation_profile_v8.py")
    snapshot = (FORMAL / "profile-01/runtime/jev_ultrafast/snapshot.js").read_text()
    plan = json.loads((OLD / "plan-01.json").read_bytes())
    plan["schema_id"] = "jev-field-syntax-v8-capacity-bound-browser-retest-plan-v1"
    plan["status"] = "SOURCE_ONLY_CAPACITY_SELECTED_NO_LIVE_SERVICE_OR_BROWSER_EXECUTION"
    plan["purpose"] = (
        "Retest all fixed10 exposed cases with the same qualified field-syntaxQ/public240 policy, "
        "using reviewed v5 CSSOM hover/v4 proxies plus v7 settling in formal v8. The separately "
        "selected genuine CPU actual16+7 capacity receipt is not model qualification or task quality."
    )
    plan["frozen_syntax_v7_parent"] = {
        "plan": ref(OLD / "plan-01.json"),
        "observer": ref(OLD / "browser_observer_runner.py"),
        "controller": ref(OLD / "root_owned_group_controller.py"),
        "closed_score": "5/10 (2/4+3/6); exposed one-repeat baseline, not a v8 result",
    }
    plan["browser_profile"].update(
        id=PROFILE,
        helper_source_sha256=helper_ref["sha256"],
        manifest_sha256=ref(FORMAL / "profile-01/profile-manifest.json")["sha256"],
        runtime_source_sha256=materialized["runtime_source_sha256"],
        effective_snapshot_js_sha256=sha(snapshot.encode()),
        marker_expression_sha256=sha(
            f"(() => {{ const state={snapshot}; return state?.marker ?? null; }})()".encode()
        ),
        scope="Formal v5 hover/v4 proxies/v3 parent plus unchanged v7 settling; no native CLICK change.",
    )
    plan["v8_capacity"] = {
        "plan": ref(CAPACITY / "plan-01.json"),
        "validator_source": ref(CAPACITY / "v8_capacity_validation.py"),
        "binding_source": ref(BASE / "v8_capacity_binding.py"),
        "root_selection": selection_ref,
        "selected_inputs": selection["inputs"],
        "selected_outer": selection["outer_receipt"],
        "prepare_directory": str(Path(selection["inputs"]["path"]).parent),
        "observation": capacity_plan["observation"],
        "interpretation": "Original technical Q remains separate. Fixed actual16 accepted;20/1/2/0 total23. No universal400/8192 bound. New complete bodies face original server parser/context gates before decode, with failures retained. Capacity metadata checks occur before/after group, outside attempt/model timing.",
    }
    refs = {r["path"]: r for r in plan["benchmark_source_refs"]}
    refs.pop("scripts/jev_browser_profile_v7_benchmark.py")
    for path in [
        ROOT / "scripts/jev_browser_profile_v8_benchmark.py",
        ROOT / "scripts/jev_observation_profile_v8.py",
        ROOT / "scripts/jev_observation_profile_v5.py",
        ROOT / "scripts/jev_observation_profile_v4.py",
        FORMAL / "profile-01/profile-manifest.json",
        FORMAL / "source-HOLD-01/HOLD.json",
        BASE / "v8_capacity_binding.py",
        CAPACITY / "plan-01.json",
        CAPACITY / "v8_capacity_validation.py",
        CAPACITY / "root-selected-capacity-inputs-01.json",
    ]:
        item = ref(path)
        refs[item["path"]] = item
    # Original v7 helper remains pinned as one of five runtime source dependencies.
    plan["benchmark_source_refs"] = list(refs.values())
    plan["judge_and_budget"]["observation_profile"] = PROFILE
    del plan["judge_and_budget"]["browser_condition_unchanged_from_Q93_v7"]
    plan["judge_and_budget"]["v8_condition"] = (
        "V5 CSSOM/hover +v4 proxies +v7 settling, new v8 tracker identity and inherited CSS-cap classification. Same native action/tick, original20s model/helper,25s SDK,16decisions and strict DONE/DOM judge. No extra calls or response processing."
    )
    for group in plan["groups_in_execution_order"]:
        group["future_output"] = str(BASE / (group["group"] + "-v8-01"))
        group["runner_selection"] = group["runner_selection"].replace("v7", "v8")
        group["argv_template"] = [
            x.replace(str(OLD), str(BASE))
            .replace("-v7-01", "-v8-01")
            .replace("jev_browser_profile_v7_benchmark.py", "jev_browser_profile_v8_benchmark.py")
            .replace("composed-visible-text-settled-v7", PROFILE)
            .replace(
                "d9f4236d1d46c9472c3d6c1551342639ecb870e6c0dd419246927645410ea868",
                helper_ref["sha256"],
            )
            for x in group["argv_template"]
        ]
    plan["service_start_proposal"]["serve_argv"] = copy.deepcopy(
        plan["service_start_proposal"]["serve_argv"]
    )
    serve = plan["service_start_proposal"]["serve_argv"]
    serve[serve.index("--output") + 1] = str(BASE / "service-01")
    plan["service_start_proposal"].pop("verify_argv")
    plan["service_start_proposal"]["binding_template"] = (
        "New schema jev-field-syntax-v8-browser-owned-binding-v1 plus v8_capacity selected inputs/outer/root-selection proof; actual fresh root-owned worker/READY/PID/start/exe/argv/listener and env-only credentials still required. No fabricated PID or endpoint and no auto-run service."
    )
    plan["service_start_proposal"]["qualification_path_contract"] = (
        "Original --qualification-path is qualify-01 DIRECTORY; original Q/41inputs unchanged. Separate capacity receipt is observer admission, never substituted into original strict41 schema."
    )
    plan["interpretation_limits"] = [
        "All10 are exposed regression cases; one live-web repeat cannot establish causality, generalization or superiority.",
        "Compare against syntax-v7 closed5/10 with all failures. Observation profile and tracker admission differ; policy/helper prompt/math/budgets do not.",
        "Native250 then proxies/hover has no universal count bound. Capacity20/1/2/0 is scoped to fixed23 CPU payloads, not every browser state or semantic quality.",
        "Prior v5 already offered curl Options while policy scrolled; extraction completion does not guarantee useful action choice.",
    ]
    plan["prelaunch_gates"] = [
        "Root full source/HOLD review, exact inverse proof and selected CPU inputs+outer closure+root selection SHA. No processor/browser/model execution during source preparation.",
        "Fresh original qualified field-syntaxQ owned service only after technical services wholeclose; root confirms prepared forward0 and exact live binding. Old CLOSED v7 or CPU worker identities are not live authority.",
        "Known4 then public6, exact fixed10/16 decisions/repeat1/seed20261014/golds/20s/25s/no selected retries. Controller default remains dry.",
    ]
    plan["failure_retention"] = [
        "Retain all10 outcomes including403/blocked/provider/schema/context/stale/geometry/budget/DONE/observer/cleanup failures. No substituted or retried selected subset.",
        "Hard service/source/cache/durability identity failure stops calls and retains partial evidence plus unrun denominator.",
    ]
    plan["execution_so_far"] = {
        "source_only": True,
        "browser": 0,
        "model_HTTP": 0,
        "new_service": 0,
        "capacity_CPU_execution_in_this_branch": 0,
    }
    save_json(BASE / "plan-01.json", plan)
    replacements = {
        'PLAN_SHA = "7ed4b3c9b8de46eaee0b53d78e324621986cc599e9f99f813f51185750fb3416"': 'PLAN_SHA = "'
        + ref(BASE / "plan-01.json")["sha256"]
        + '"',
        "isolated v7 Jev runners": "isolated v8 Jev runners",
        "jev-field-syntax-browser-owned-binding-v1": "jev-field-syntax-v8-browser-owned-binding-v1",
        "q93_v7_isolated_launcher": "syntax_v8_isolated_launcher",
        "jev_browser_profile_v7_benchmark.py": "jev_browser_profile_v8_benchmark.py",
        "jev-field-syntax-v7-observed-browser-launch-v1": "jev-field-syntax-v8-observed-browser-launch-v1",
        "Explicit v7 isolated runner/profile": "Explicit v8 isolated runner/profile",
        "selected v7 isolated benchmark runner": "selected v8 isolated benchmark runner",
        "jev-field-syntax-v7-observed-browser-child-exit-v1": "jev-field-syntax-v8-observed-browser-child-exit-v1",
    }
    observer = (OLD / "browser_observer_runner.py").read_text()
    for before, after in replacements.items():
        assert before in observer
        observer = observer.replace(before, after)
    addition = '''def capacity_state(binding):
    """Pure saved CPU inputs/closure check; separate from technical model Q."""
    plan = checked_plan()
    expected = plan["v8_capacity"]
    path = checked_ref(expected["binding_source"])
    blob = path.read_bytes()
    if sha(blob) != expected["binding_source"]["sha256"]:
        raise EvidenceError("selected v8 capacity binding source changed")
    spec = importlib.util.spec_from_file_location("_v8_capacity_binding", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(blob, str(path), "exec"), module.__dict__)
    return module.validate_selection(binding.get("v8_capacity"), expected, root=ROOT)


'''
    observer = observer.replace(
        "def binding_state(binding):", addition + "def binding_state(binding):", 1
    )
    observer = observer.replace(
        '    if binding.get("schema_id") != "jev-field-syntax-v8-browser-owned-binding-v1":',
        '    capacity = capacity_state(binding)\n    if binding.get("schema_id") != "jev-field-syntax-v8-browser-owned-binding-v1":',
        1,
    )
    observer = observer.replace(
        '        "qualification_sha256": Q_SHA,\n',
        '        "qualification_sha256": Q_SHA,\n        "v8_capacity": capacity,\n',
        1,
    )
    save(BASE / "browser_observer_runner.py", observer.encode())
    controller = (
        (OLD / "root_owned_group_controller.py")
        .read_text()
        .replace(
            "0aa07134ea272c2a62b6534e3a3506b9e2006e9886bc506a210b349e4b03c960",
            ref(BASE / "browser_observer_runner.py")["sha256"],
        )
    )
    save(BASE / "root_owned_group_controller.py", controller.encode())
    profile = (OLD / "review_v7_profile_evidence.py").read_text()
    for before, after in (
        ("v7 checks", "v8 checks"),
        ("composed-visible-text-settled-v7", PROFILE),
        (
            "jev-v7-native-observation-settling-sidecar-v1",
            "jev-v8-native-observation-settling-sidecar-v1",
        ),
        ("jev_browser_profile_v7_benchmark.py", "jev_browser_profile_v8_benchmark.py"),
        ("jev_observation_profile_v7.py", "jev_observation_profile_v8.py"),
        ("_saved_v7_audit_", "_saved_v8_audit_"),
    ):
        assert before in profile
        profile = profile.replace(before, after)
    save(BASE / "review_v8_profile_evidence.py", profile.encode())
    for filename in (
        "browser_observer_runner.py",
        "root_owned_group_controller.py",
        "plan-01.json",
    ):
        save(
            BASE / (filename + ".patch"),
            "".join(
                difflib.unified_diff(
                    (OLD / filename).read_text().splitlines(keepends=True),
                    (BASE / filename).read_text().splitlines(keepends=True),
                    fromfile=str((OLD / filename).relative_to(ROOT)),
                    tofile=str((BASE / filename).relative_to(ROOT)),
                )
            ).encode(),
        )
    save(
        BASE / "review-profile-v7-to-v8.patch",
        "".join(
            difflib.unified_diff(
                (OLD / "review_v7_profile_evidence.py").read_text().splitlines(keepends=True),
                profile.splitlines(keepends=True),
                fromfile=str((OLD / "review_v7_profile_evidence.py").relative_to(ROOT)),
                tofile=str((BASE / "review_v8_profile_evidence.py").relative_to(ROOT)),
            )
        ).encode(),
    )
    save_json(BASE / "observer-literal-replacements.json", replacements)
    for filename in (
        "browser_observer_runner.py",
        "root_owned_group_controller.py",
        "review_v8_profile_evidence.py",
    ):
        ast.parse((BASE / filename).read_bytes())


if __name__ == "__main__":
    main()
