"""Export selected sources, aggregate counts and hash references; no raw payloads."""

import collections
import hashlib
import json
import statistics
import subprocess
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[2]


def load(path):
    return json.loads(path.read_bytes())


def pin(path):
    blob = path.read_bytes()
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "size_bytes": len(blob),
    }


def write(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def main():
    post = load(HERE / "posthoc-validation-01.json")
    actual = HERE / "prepare-actual-01"
    closure = load(HERE / "prepare-root-external-closure-01.json")
    worker = load(actual / "terminal.json")
    assert post["recorded_whole_closure_verified"] and post["review"]["training_ready"]
    assert post["strict_native_artifact_replay"]["compiled_native_records_verified"] == 179
    assert post["strict_native_artifact_replay"]["compiled_native_CFG_inventories_verified"] == 179
    assert worker["primary_error"] is None and closure["worker_primary_error"] is None
    test_dir = HERE / "tests-04"
    test = load(test_dir / "integration-tests-terminal.json")
    test_exit = load(test_dir / "integration-tests-root-closure.json")
    suite = ET.parse(test_dir / "integration-tests.xml").getroot().find("testsuite")
    assert suite is not None and suite.get("errors") == suite.get("failures") == "0"
    assert test["pytest_returncode"] == 0 and test["sources_unchanged"]
    assert not test["CUDA_initialized"] and test_exit["cgroup_absent_after_stop"]
    common = set(test["sources_before"]) & set(worker["sources"])
    assert common and all(
        test["sources_before"][label] == worker["sources"][label]["sha256"] for label in common
    )
    assert test_exit["reason"] == "terminal"
    assert all(
        test_exit["terminal_before_stop"][k] == v
        for k, v in {
            "Result": "success",
            "ExecMainCode": "1",
            "ExecMainStatus": "0",
            "MainPID": "0",
        }.items()
    )
    rows = [load(p) for p in sorted((actual / "worker/rows").glob("*.json"))]
    assert len(rows) == 179
    tokens = {}
    for mode in ("raw-json-v1", "shared-elements-v2"):
        values = [r["modes"][mode]["compiled"]["record"]["prompt_tokens"] for r in rows]
        tokens[mode] = {
            "rows": len(values),
            "total": sum(values),
            "minimum": min(values),
            "maximum": max(values),
            "median": statistics.median(values),
        }
    effective = collections.Counter(
        r["modes"]["shared-elements-v2"]["compiled"]["record"]["prompt_representation"]["format"]
        for r in rows
    )
    gates = [
        json.loads(line)
        for line in (actual / "worker/resource-gates.jsonl").read_bytes().splitlines()
    ]
    probes = [
        json.loads(line)
        for line in (HERE / "native-fit-resource-observation-01.jsonl").read_bytes().splitlines()
    ]
    assert len(gates) == 1080 and all(g["admitted"] for g in gates)
    assert all(not p["admitted"] for p in probes)
    source_names = (
        "prepare_cpu.py",
        "test_prepare_cpu.py",
        "launch_prepare_cpu.py",
        "integration_tests.py",
        "run_integration_tests.py",
        "validate_prepare_saved.py",
        "test_validate_prepare_saved.py",
        "export_results.py",
    )
    evidence_paths = [
        HERE / "prepare-plan-01.json",
        actual / "intent.json",
        actual / "terminal.json",
        actual / "worker/resource-gates.jsonl",
        actual / "worker/processor-before.json",
        actual / "worker/processor-after.json",
        actual / "worker/runtime-before.json",
        actual / "worker/runtime-after.json",
        HERE / "prepare-root-launch-01.json",
        HERE / "prepare-root-unit-exit-01.json",
        HERE / "prepare-root-external-closure-01.json",
        HERE / "posthoc-validation-01.json",
        HERE / "validator-tests.xml",
        HERE / "native-fit-resource-observation-01.jsonl",
        test_dir / "integration-tests-intent.json",
        test_dir / "integration-tests-terminal.json",
        test_dir / "integration-tests-root-closure.json",
        test_dir / "integration-tests.xml",
        test_dir / "integration-tests-service.log",
    ]
    index = {
        "schema_id": "jev_native_v2_selected_evidence_index_v1",
        "sources": [pin(HERE / name) for name in source_names],
        "evidence_hash_references_only": [pin(p) for p in evidence_paths],
        "raw_wires_prompts_tokens_processor_runtime_copies_published": False,
    }
    index_path = HERE / "selected-evidence-index-01.json"
    write(index_path, index)
    implementation = subprocess.check_output(
        ["git", "rev-parse", "bb0fd14"], cwd=ROOT, text=True
    ).strip()
    total_tests, skipped = int(suite.get("tests")), int(suite.get("skipped"))
    report = {
        "schema_id": "jev_native_v2_fit_processor_results_v1",
        "at_utc": datetime.now(UTC).isoformat(),
        "status": "NATIVE_V2_PIPELINE_TESTED_FULL179_PROCESSOR_READY_REAL12B_FIT_RESOURCE_REFUSED",
        "implementation_commit": implementation,
        "profile": {
            "artifact_kind": "jev_native_action_lora_v2",
            "schema_version": 2,
            "requested_format": "shared-elements-v2",
            "training_quantization": "none",
            "training_dtype": "bfloat16",
            "deployment_quantization": "nf4",
            "inference_identity": "generated-actions-nf4",
            "objective": "full-vocabulary-action-token-CE",
            "CFG_loss_training_performed": False,
            "old_raw_artifact_profile_preserved": True,
            "BF16_generated_actions_none_prompt_equivalence_attested": False,
        },
        "final_GX10_CPU_regression": {
            "passed": total_tests - skipped,
            "skipped": skipped,
            "failed": 0,
            "pytest_seconds": float(suite.get("time")),
            "source_files_stable": len(test["sources_before"]),
            "invocation": test_exit["invocation_id"],
            "cached_terminal_MemoryPeak_bytes": int(
                test_exit["terminal_before_stop"]["MemoryPeak"]
            ),
            "process_peak_RSS_bytes": test["process_peak_RSS_bytes"],
            "CUDA_initialized_before_after": False,
            "CUDA_driver_probe_absence_attested": False,
            "whole_owned_cgroup_removed": True,
            "scope": "Genuine tiny CPU LoRA with synthetic tokenizer and mocked12B gates/checkpoint/model admission; no real12B qualification.",
            "native_tiny_fit": {
                "train_updates": 142,
                "development_examples": 37,
                "development_updates": 0,
                "warm_start_tensor_initialization_verified": True,
                "partial_update_primary_failure_preserved": True,
            },
        },
        "preserved_earlier_test_attempts": [
            {
                "attempt": "tests-01",
                "passed": 10,
                "failed": 1,
                "reason": "Overbroad immutable-input output guard; corrected before final tests.",
            },
            {
                "attempt": "tests-02",
                "passed": 13,
                "errors": 1,
                "reason": "Tiny fixture deepcopy attempted to pickle torch module; corrected fixture memo.",
            },
            {
                "attempt": "tests-03",
                "passed": 489,
                "skipped": 2,
                "scope": "Passed before final attached-native processor envelope guard; final tests-04 supersedes it.",
            },
        ],
        "processor_and_posthoc_focused_tests": {
            "initial_actor_tests_before_actual": {"passed": 14, "seconds": 2.28},
            "after_actual_combined_actor_posthoc": {
                "passed": 31,
                "failed": 0,
                "deselected": 1,
                "junit_seconds": 3.376,
                "deselected_reason": "Sealed fresh-output-only actor test was already passed before actual output existed; original test and actual output preserved.",
            },
        },
        "actual_full179_processor": {
            "counts": {"train": 142, "development": 37, "total": 179},
            "selected_historical_data_files": 45,
            "new_labels": 0,
            "review": post["review"],
            "strict_native_artifact_replay": post["strict_native_artifact_replay"],
            "prompt_tokens": tokens,
            "V2_effective_formats": dict(effective),
            "total_prompt_token_reduction_fraction": 1
            - tokens["shared-elements-v2"]["total"] / tokens["raw-json-v1"]["total"],
            "same_teacher_token_paths": sum(
                r["modes"]["raw-json-v1"]["compiled"]["targets"]
                == r["modes"]["shared-elements-v2"]["compiled"]["targets"]
                for r in rows
            ),
            "full_RAW_V2_grammar_target_parity_rows": sum(
                r["both_compiled_grammar_target_parity"] is True for r in rows
            ),
            "processor_references": 58,
            "runtime_files": 348,
            "source_files_stable": 110,
            "invocation": worker["InvocationID"],
            "actor": worker["actor"],
            "resource_gates": len(gates),
            "required_bytes_every_gate": gates[0]["required_bytes"],
            "minimum_observed_MemAvailable_bytes": min(
                g["observation"]["MemAvailable"] for g in gates
            ),
            "largest_live_gate_cgroup_peak_bytes": max(
                g["unit"]["memory_peak_bytes"] for g in gates
            ),
            "unit_memory_max_bytes": 3 * 1024**3,
            "unit_swap_max_bytes": 0,
            "whole_owned_closed": True,
            "unit_exit_code": 0,
            "CUDA_initialized": False,
            "original_root_wrapper_exit_code": 1,
            "original_posthoc_error": post["original_posthoc_error"],
            "posthoc_replay": "Separate verifier uses complete ordered JSON data-evidence bytes; tuples/list differ only after JSON storage. Original actor/plan/closure unchanged.",
            "data_evidence_join": post["data_join"],
            "original_179_full_wires_sha256": post["original_179_full_wires_sha256"],
            "ordered_saved_179_full_wires_verified": post["ordered_saved_179_full_wires_verified"],
            "final_postexit_cgroup_counters": None,
            "counter_scope": "Recorded live counters only; no final zero-OOM claim after cgroup removal.",
            "native12B_model_constructor_forward_train_HTTP_browser_calls": 0,
            "readiness_scope": "Processor/compiler readiness only; real12B fitting, resource qualification and browser quality pending.",
        },
        "native_fit_resource_only_probe": {
            "actual_fit_launched": False,
            "framework_or_model_loads": 0,
            "observations": [
                {
                    "scope": p["scope"],
                    "admitted": p["admitted"],
                    "available_bytes": p["observation"]["MemAvailable"],
                    "required_bytes": p["required_bytes"],
                }
                for p in probes
            ],
        },
        "unchanged_goal_limits": {
            "latest_real_public_V7_browser_passed": 5,
            "latest_real_public_V7_browser_total": 10,
            "new_browser_trials": 0,
            "six_Decision3_matched_warm_comparisons_complete": False,
            "task_success_or_model_superiority_improvement_claimed": False,
            "Comfy_interrupt_unload_free_calls": 0,
            "queue_observation_only": True,
            "goal_complete": False,
        },
        "selected_evidence": pin(index_path),
    }
    report_path = ROOT / "docs/validation/2026-10-21/jev-native-v2-fit-processor-results.json"
    write(report_path, report)
    previous = (
        ROOT / "artifacts/jev-runtime/composed-observation-integration-work-state-20261016-97.json"
    )
    state = {
        "schema_id": "jev_real_browser_goal_work_state_v98",
        "previous_state": pin(previous),
        "goal_status": "active",
        "turn_classification": "PROGRESS_NATIVE_V2_PIPELINE_FULL179_PROCESSOR_READY_REAL12B_RESOURCE_REFUSED",
        "implementation_git": implementation,
        "latest_results": pin(report_path),
        "completed": [
            "Native179 V2 BF16-fit/NF4-export/serve contract implemented and pushed; legacy raw preserved.",
            "Final CPU regression suite passed, including genuine tiny142-update fit, warm-start and dev37zero-updates.",
            "Real processor all179 RAW/V2/CFG compiled, strict native artifact rows/inventories replayed; original teachers/wires retained.",
            "Single owned CPU worker exited0 and closed; original posthoc list/tuple comparison failure preserved and independently revalidated.",
        ],
        "remaining": [
            "Qualify resource-admitted real12B native V2 training/export/deployment; current BF16 load/train probe refused.",
            "Run matched actual Jev browser tasks and all6 Decision3 warm comparisons; latest real publicV7 remains5/10.",
        ],
        "constraints": [
            "UseGX10; no Modal/Colab budget.",
            "Preserve resource gates; no Comfy interruption/unload/free without its pending answer.",
            "Publish narrowed source/tests/aggregate/hashreferences only; raw payloads stay local.",
        ],
        "scope": "Input readiness and tiny fixtures establish no real12B quality or model superiority; goal remains unfinished.",
    }
    write(
        ROOT / "artifacts/jev-runtime/composed-observation-integration-work-state-20261016-98.json",
        state,
    )
    print(
        json.dumps(
            {
                "report": pin(report_path),
                "final_passed": total_tests - skipped,
                "processor_rows": len(rows),
            }
        )
    )


if __name__ == "__main__":
    main()
