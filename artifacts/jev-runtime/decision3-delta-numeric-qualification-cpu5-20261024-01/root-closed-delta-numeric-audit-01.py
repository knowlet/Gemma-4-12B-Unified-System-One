"""Read-only root audit of the closed CPU5 frozen-delta numeric qualification; no model/scorer execution."""

import argparse
import array
import ast
import hashlib
import json
import math
import struct
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
HERE = Path(__file__).resolve().parent
OWNER = ROOT / "artifacts/jev-runtime/llama-cpp-owner-20261024-09"
OUT = OWNER / "actual-21"
UNIT = "jev-d3-llama-cpp-cpu5-fast-4g-coexist-20261024-09.service"
PROFILE = "llama-cpp-cpu5-fast-same-cluster-4g-coexist-command-owned-v1"
PLAN_SHA = "1b371920fa7c2c007704183df957a4cd2194bf6e001b99e369bb4800666d31b0"
SPEC_SHA = "e3a051b88213678d96f230ef20ed41ffaa4709aef67a1a2b92653164b2eef746"
CONTROLLER_SHA = "81846efe2ee0357da07283c1a77a606283a920cbddef668c8f62def291057dee"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ref(path):
    return {
        "path": str(path.relative_to(ROOT)),
        "size_bytes": path.stat().st_size,
        "sha256": sha(path),
    }


def read(path):
    require(
        path.is_file() and not path.is_symlink() and path.stat().st_size <= 1048576,
        "Bounded original JSON required",
    )
    return json.loads(path.read_text())


def authenticate(row):
    p = Path(row["path"])
    p = p if p.is_absolute() else ROOT / p
    info = p.stat()
    size = row.get("size_bytes", row.get("size"))
    require(size is None or size == info.st_size, "Selected file size differs")
    if "stat" in row:
        require(
            all(type(v) is int and getattr(info, k) == v for k, v in row["stat"].items()),
            "Selected original stat identity differs",
        )
    if "sha256" in row:
        require(sha(p) == row["sha256"], "Selected file hash differs")
    else:
        require(bool(row.get("stat")), "Original stat-only scope required")


def npy_payload(row, shape, *, diagnostic):
    authenticate(row)
    path = Path(row["path"])
    require(
        path.is_absolute() and path.is_relative_to(ROOT) and not path.is_symlink(),
        "ordinary private array",
    )
    require(
        row["shape"] == list(shape)
        and row["dtype"] == "float32"
        and row["format"] == "npy_no_pickle",
        "exact FP32 array contract",
    )
    if diagnostic:
        require(
            path.is_relative_to(OUT / "delta-numeric")
            and row["status"] == "PENDING_NONTRAINING"
            and row["training_eligible"] is False,
            "permanently diagnostic array",
        )
    raw = path.read_bytes()
    require(len(raw) < 2 * 1024**2 and raw[:8] == b"\x93NUMPY\x01\x00", "bounded NPY1 no pickle")
    n = struct.unpack("<H", raw[8:10])[0]
    require(
        n <= 4096
        and ast.literal_eval(raw[10 : 10 + n].decode("latin1"))
        == {"descr": "<f4", "fortran_order": False, "shape": tuple(shape)},
        "exact Corder little-endian FP32 header",
    )
    data = raw[10 + n :]
    require(len(data) == math.prod(shape) * 4, "complete FP32 bytes")
    values = array.array("f")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    require(all(math.isfinite(v) for v in values), "finite diagnostic array")
    return data, values


def probabilities(raw, questions, counts, model):
    require(
        raw["model"] == model and list(raw["answers"]) == list(questions),
        "explicit alias and full original head order",
    )
    result = []
    for (head, question), count in zip(questions.items(), counts):
        a = raw["answers"][head]
        p = a["probabilities"]
        values = list(p.values())
        require(
            a["type"] == "choice"
            and list(p) == list(question["criteria"])
            and len(values) == count
            and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in values)
            and abs(sum(values) - 1) <= 1e-6,
            "all ordered finite normalized legal probabilities",
        )
        require(
            a["choice"] == list(p)[max(range(count), key=lambda k: values[k])], "ordered argmax"
        )
        entropy = -sum(v * math.log(v) for v in values if v > 0)
        confidence = 1.0 if count < 2 else max(0.0, min(1.0, 1 - entropy / math.log(count)))
        require(
            type(a["confidence"]) in (int, float) and abs(a["confidence"] - confidence) <= 1e-6,
            "unchanged entropy confidence/singleton",
        )
        result.append(values)
    return result


def mixed(a, b):
    return (
        type(a) in (int, float)
        and math.isfinite(a)
        and math.isfinite(b)
        and abs(a - b) <= 1e-6 + 1e-5 * abs(b)
    )


def execute():
    terminal_path = OWNER / "root-execution-delta-numeric-21.json"
    require(
        terminal_path.is_file() and (OUT / "receipt.json").is_file(),
        "Original root and owner must be terminal first",
    )
    terminal, receipt = read(terminal_path), read(OUT / "receipt.json")
    closure = read(OUT / "external-boundary-closure.json")
    require(
        terminal["exit_code"] == 0
        and receipt["profile"] == PROFILE
        and receipt["completed"] is True
        and receipt["primary_error"] is None
        and receipt["secondary_errors"] == []
        and receipt["sources_unchanged"] is True
        and receipt["comfy_service_modified"] is False,
        "Original successful owner and unchanged sources required",
    )
    require(
        closure["closure_mode"] == "authenticated_stop"
        and closure["closure_errors"] == []
        and all(
            closure[k] is True
            for k in (
                "whole_unit_exit_confirmed",
                "observed_cgroup_cohort_exit_confirmed",
                "post_stop_terminal_verified",
                "cgroup_absent",
                "sources_unchanged",
            )
        )
        and len(closure["two_process_scans"]) == 2
        and all(not row["matches"] and not row["unknown"] for row in closure["two_process_scans"]),
        "Original current authenticated scoped closure required",
    )
    for k in (
        "no_escape_enforced",
        "whole_tree_exit_confirmed",
        "continuation_authorized",
        "model_qualified",
    ):
        require(closure[k] is False, "Original universal nonqualification flags required")
    config = receipt["selected_config"]
    require(
        config == read(OUT / "sealed-config.json")
        and config["profile"] == PROFILE
        and config["unit"] == UNIT
        and config["output"] == str(OUT)
        and config["selection_sha256"] == SPEC_SHA,
        "Exact current selected config required",
    )
    admission = read(OUT / "admission.json")
    require(
        admission["nonce"] == config["nonce"] == closure["selection"]["nonce"]
        and admission["invocation_id"] == closure["invocation_id"]
        and closure["selection"]["unit"] == UNIT
        and all(
            admission["actor"][k] == closure["actor"][k]
            for k in ("pid", "start_ticks", "pgid", "sid", "uid", "cgroup")
        )
        and admission["config_sha256"] == sha(OUT / "sealed-config.json"),
        "Current admission/config/closure invocation differs",
    )
    spec_path, plan_path = HERE / "root-command-spec-01.json", HERE / "selected-plan-01.json"
    require(
        sha(spec_path) == SPEC_SHA
        and sha(plan_path) == PLAN_SHA
        and sha(OWNER / "controller.py") == CONTROLLER_SHA,
        "Selected source/plan pins differ",
    )
    spec, plan = read(spec_path), read(plan_path)
    base = [
        OWNER / "controller.py",
        spec_path,
        ROOT / "artifacts/jev-runtime/decision3-local-owner-terminal-20261022-01/owner_boundary.py",
        ROOT / "artifacts/jev-runtime/observer-system-services-20261022-01/comfy_observer.py",
        ROOT / "artifacts/jev-runtime/observer-system-services-20261022-01/system-services.json",
    ]
    expected = {r["path"]: r for r in [*spec["sources"], *(ref(p) for p in base)]}
    require(
        len(spec["sources"]) == 55
        and len(expected) == 60
        and {r["path"]: r for r in config["sources"]} == expected,
        "Exact55 plus5 original source union required",
    )
    for row in config["sources"] + spec["binary_stats"]:
        authenticate(row)
    native_plan = read(Path(plan["dependencies"]["native_plan"]["path"]))
    for row in [
        *native_plan["model_files"],
        *native_plan["runtime_files"],
        native_plan["manifest"],
        *native_plan["owner_sources"],
    ]:
        authenticate(row)
    require(
        len(native_plan["model_files"]) == 20 and len(native_plan["runtime_files"]) == 28,
        "Original model/runtime inventory differs",
    )
    expected_argv = [
        str(ROOT / ".venv/bin/python"),
        "-B",
        "-s",
        str(OWNER / "controller.py"),
        "--run",
        "--expected-entry-sha256",
        CONTROLLER_SHA,
        "--selection",
        str(spec_path),
        "--selection-sha256",
        SPEC_SHA,
        "--output",
        str(OUT),
    ]
    require(
        terminal["argv"] == expected_argv
        and terminal["profile"] == "root-selected-delta-numeric-cpu5-terminal-v1",
        "Exact root CPU5 invocation differs",
    )
    command = read(OUT / "command-01-output.json")["result"]
    started = read(OUT / "command-01-started.json")
    intent = read(OUT / "command-01-intent.json")
    require(
        command["returncode"] == 0
        and command["pid"] == started["pid"]
        and command["argv"]
        == [v.replace("{OUTPUT}", str(OUT)) for v in spec["commands"][0]["argv"]],
        "Original successful selected child command required",
    )
    observed = started["observed_identity"]
    child, wrapper = observed["process"], admission["actor"]
    require(
        child["pid"] == started["pid"] == command["pid"]
        and child["pid"] != wrapper["pid"]
        and all(child[k] == wrapper[k] for k in ("pgid", "sid", "uid", "cgroup"))
        and all(
            child[k] == command["observed_identity"]["process"][k]
            for k in ("pid", "start_ticks", "pgid", "sid", "uid", "cgroup")
        )
        and all(observed[k] == command["observed_identity"][k] for k in ("exe", "argv", "cwd"))
        and observed["argv"] == intent["argv"] == command["argv"]
        and observed["cwd"]
        == intent["cwd"]
        == command["cwd"]
        == spec["commands"][0]["cwd"].replace("{OUTPUT}", str(OUT))
        and observed["exe"]
        == intent["resolved_executable"]
        == command["resolved_executable"]
        == str(Path(command["argv"][0]).resolve(strict=True)),
        "Current admitted wrapper cohort and exact child identity/exe/argv/cwd required",
    )
    state_path = OUT / "delta-numeric/state.json"
    state = read(state_path)
    require(
        state["profile"] == plan["profile"]
        and state["plan_sha256"] == PLAN_SHA
        and state["completed"] is True
        and state["model_loads"] == 1
        and state["native_request_calls"] == len(state["calls"]) == 4
        and state["training_eligible"] is False
        and state["labels"] is None
        and state["semantic_expected_answers"] is None
        and state["benchmark_scores"] is None
        and all(
            state[k] is False
            for k in (
                "browser_qualified",
                "model_qualified",
                "no_escape_audited",
                "whole_tree_audited",
            )
        ),
        "one model/four calls and scoped nontraining status",
    )
    require(
        state["initial_full_model_manifest_runtime_sources_authenticated"] is True
        and state["original_full_model_runtime_source_authentication"] is True
        and state["baseline_metadata_values_and_bits_unchanged"] is True,
        "original full authentication and unchanged baseline bits",
    )
    proc = state["process"]
    require(
        proc["pid"] == child["pid"]
        and proc["starttime_ticks"] == child["start_ticks"]
        and all(proc[k] == child[k] for k in ("pgid", "sid"))
        and proc["cgroup"].strip() == "0::" + child["cgroup"]
        and proc["affinity"] == list(range(15, 20)),
        "actual child binding",
    )
    require(
        state["owned_limits"]
        == {"memory.max": "4294967296", "memory.swap.max": "0", "cpu.max": "500000 100000"}
        and state["host_startup_admission"]["required_available_bytes"] == 30414140212
        and state["host_startup_admission"]["MemAvailable"] >= 30414140212,
        "same CPU5/full4GiB admission",
    )
    provenance = state["native_provenance"]
    require(
        provenance["model_sha256"] == plan["model"]["model_sha256"]
        and provenance["backbone_dtype"] == "bfloat16"
        and provenance["readout_dtype"] == "float32"
        and provenance["temperature"] == 1.0
        and provenance["batch_size"] == 4
        and provenance["attention_mode"] == "noncausal_full_attention"
        and state["native_runtime_info"]["device"] == "cpu"
        and state["actual_environment"]["OMP_NUM_THREADS"] == "5"
        and state["actual_environment"]["CUDA_VISIBLE_DEVICES"] == "",
        "original genuine native provenance",
    )
    require(
        0
        < state["elapsed_before_final_retention_seconds"]
        < command["elapsed_seconds"]
        < terminal["wall_seconds"]
        < 120
        and abs(
            (
                datetime.fromisoformat(terminal["finished_at_utc"])
                - datetime.fromisoformat(terminal["started_at_utc"])
            ).total_seconds()
            - terminal["wall_seconds"]
        )
        < 0.01,
        "whole120 actual command/root bounds",
    )
    for k in ("capability_ref", "root_acceptance_ref", "delta_ref"):
        authenticate(plan[k])
    capability = read(Path(plan["capability_ref"]["path"]))
    acceptance = read(Path(plan["root_acceptance_ref"]["path"]))
    require(
        capability["status"] == "ROOT_ACCEPTED_TRAIN_ONLY_READOUT"
        and capability["schema"] == "root-accepted-dsl01-train-delta-v1"
        and capability["delta_ref"] == plan["delta_ref"]
        and capability["root_acceptance_ref"] == plan["root_acceptance_ref"]
        and capability["model"] == plan["model"]
        and capability["frozen_before_dev_sealed_benchmark_or_browser"] is True
        and acceptance["status"] == "accepted"
        and acceptance["schema"] == "root-accepted-dsl01-trained-readout-v1"
        and acceptance["root_exit"] == 0
        and acceptance["owner_completed"] is True
        and acceptance["current_original_scoped_closure_positive"] is True,
        "true frozen train-only accepted candidate joins",
    )
    delta_bytes, delta = npy_payload(plan["delta_ref"], (255, 1024), diagnostic=False)
    require(any(v != 0 for v in delta), "true nonzero delta artifact")
    fixture = read(Path(plan["dependencies"]["fixture"]["path"]))["request"]
    baseline_bytes = None
    all_calls = []
    catalog_identity = None
    diagnostic_refs = []
    for index, (q, (kind, rows)) in enumerate(zip(state["calls"], plan["call_order"]), 1):
        target = OUT / "delta-numeric" / f"call-{index:02d}-{kind}-{rows}"
        evidence = read(target / "evidence-pending.json")
        require(
            q["index"] == index
            and q["kind"] == kind
            and q["actual_rows_required"] == rows
            and q["completed"] is True
            and q["training_eligible"] is False
            and q["labels"] is None
            and q["outer_hooks_removed"] is True
            and q["original_model_full_authentication_unchanged"] is True
            and evidence == {k: v for k, v in q.items() if k != "complete_prediction_seconds"},
            "closed call evidence and cleanup",
        )
        require(
            0 < q["elapsed_before_final_retention_seconds"] < q["complete_prediction_seconds"] < 20,
            "full call20 after required retention and cleanup",
        )
        questions = dict(list(fixture["questions"].items())[:rows])
        counts = [len(v["criteria"]) for v in questions.values()]
        hbytes, h = npy_payload(q["private_feature_payload"], (rows, 1024), diagnostic=True)
        wbytes, w = npy_payload(q["private_baseline_payload"], (255, 1024), diagnostic=True)
        require(
            hashlib.sha256(wbytes).hexdigest()
            == capability["baseline_fp32_bytes_sha256"]
            == q["baseline_fp32_bytes_sha256"]
            and (baseline_bytes is None or baseline_bytes == wbytes),
            "exact original FP32 baseline bytes across calls/train capability",
        )
        baseline_bytes = wbytes
        raw_envelope = read(target / "raw-response-pending.json")
        require(
            raw_envelope["status"] == "PENDING_NONTRAINING"
            and raw_envelope["training_eligible"] is False,
            "private raw permanently nontraining",
        )
        raw = raw_envelope["raw_response"]
        probs = probabilities(
            raw,
            questions,
            counts,
            plan["model"]["name"] if kind == "original" else capability["alias"],
        )
        prepared = read(target / "backend-prepared.json")
        require(
            prepared["status"] == "PENDING_NONTRAINING"
            and prepared["training_eligible"] is False
            and prepared["data"]["original_question_order"] == list(questions),
            "same original question order",
        )
        lengths = prepared["data"]["lengths"]
        require(
            list(lengths) == list(questions)
            and all(type(v) is int and 0 < v <= 16384 for v in lengths.values())
            and raw["usage"]["input_tokens"] == sum(lengths.values()),
            "complete untruncated native usage",
        )
        events = [[rows, max(lengths.values()), 1024]]
        require(
            q["outer_first_layer_events"] == q["outer_final_norm_events"] == events,
            "exact true 4/2 rows, one encoder/norm pass",
        )
        retained_before_alias = read(target / "backend-raw_returned_before_validation.json")["data"]
        require(
            retained_before_alias == {**raw, "model": plan["model"]["name"]},
            "exact retained native response before explicit candidate alias",
        )
        actual_projection = q["reconstructed_same_forward_logits"]
        if kind == "candidate":
            a = q["candidate_backend_evidence"]
            chunk = a["chunks"][0]
            require(
                a["status"] == "completed"
                and a["raw_response"] == raw
                and a["capability"] == capability
                and a["lengths"] == lengths
                and len(a["chunks"]) == 1
                and chunk["head_order"] == list(questions)
                and chunk["ordered_criteria_ids"]
                == [list(v["criteria"]) for v in questions.values()]
                and chunk["first_layer_events"] == chunk["final_norm_events"] == events
                and chunk["legal_counts"] == counts
                and chunk["rows"] == rows
                and chunk["padded_width"] == max(lengths.values()),
                "candidate one actual forward and same true trained capability",
            )
            for name in ("base_legal_logits", "delta_legal_logits", "adapted_legal_logits"):
                actual = chunk["projection"][name]
                expected = actual_projection[name]
                require(
                    len(actual) == len(expected) == rows
                    and all(
                        len(a) == len(b) == n and all(mixed(x, y) for x, y in zip(a, b))
                        for a, b, n in zip(actual, expected, counts)
                    ),
                    "all candidate actual FP32 logits vs same-forward reconstruction",
                )
            for event in ("chunk_returned", "completed"):
                retained = read(target / ("backend-" + event + ".json"))
                require(
                    retained["status"] == "PENDING_NONTRAINING"
                    and retained["training_eligible"] is False,
                    "required successful candidate sink retention",
                )
            require(
                read(target / "backend-completed.json")["data"]["raw_response"] == raw,
                "completed sink exact raw",
            )
        else:
            require(
                q["zero_primitive_probabilities"]["all_legal_probabilities_matched"] is True
                and q["original_torch_replay"]["all_legal_probabilities_matched"] is True,
                "original raw zero primitive contract",
            )
        require(
            q["same_forward_probabilities"]["all_legal_probabilities_matched"] is True
            and q["same_forward_probabilities"]["all_ordered_argmaxes_matched"] is True
            and q["response_contract"]["contract_valid"] is True
            and q["entropy"]["all_entropy_confidences_matched"] is True,
            "pinned actual FP32 replay/response proof",
        )
        witness = []
        for j, (head, count, p) in enumerate(zip(questions, counts, probs)):
            base_logits = [
                math.fsum(h[j * 1024 + k] * w[code * 1024 + k] for k in range(1024))
                for code in range(count)
            ]
            delta_logits = (
                [
                    math.fsum(h[j * 1024 + k] * delta[code * 1024 + k] for k in range(1024))
                    for code in range(count)
                ]
                if kind == "candidate"
                else [0.0] * count
            )
            logits = [a + b for a, b in zip(base_logits, delta_logits)]
            e = [math.exp(x - max(logits)) for x in logits]
            total = math.fsum(e)
            reference = [x / total for x in e]
            require(
                all(mixed(a, b) for a, b in zip(reference, p))
                and max(range(count), key=lambda k: reference[k])
                == max(range(count), key=lambda k: p[k]),
                "independent stdlib all-legal FP64 probability/ordered-argmax witness",
            )
            witness.append(
                {
                    "head": head,
                    "legal_count": count,
                    "maximum_probability_absolute_error": max(
                        abs(a - b) for a, b in zip(reference, p)
                    ),
                    "ordered_argmax_matched": True,
                }
            )
        catalogs = q["numpy_catalog_discovery"]
        before, after = catalogs["before"], catalogs["after"]
        require(
            {k: v for k, v in before.items() if k != "own_maps_sha256"}
            == {k: v for k, v in after.items() if k != "own_maps_sha256"},
            "per-call runtime projection unchanged; heap map hashes may vary",
        )
        if catalog_identity is None:
            critical = after["critical_files_and_mapped_math_candidates"]
            gpu = after["mapped_gpu_math_candidates_excluded_from_CPU_projection"]
            require(
                after["numpy_version"] == "2.5.2"
                and after["python"]["version"] == [3, 13, 14]
                and after["python"]["resolved_path"] == command["resolved_executable"]
                and len(critical) == 22
                and len(gpu) == 2
                and sum(r["size_bytes"] for r in critical) <= 512 * 1024**2
                and all(0 < r["size_bytes"] <= 256 * 1024**2 for r in critical),
                "bounded genuine NumPy critical projection",
            )
            for r in critical:
                require("sha256" in r, "CPU full hashes")
                authenticate(r)
            for r in gpu:
                require(
                    "sha256" not in r
                    and r["hashed"] is False
                    and r["included_in_CPU_projection"] is False
                    and r["execution_or_symbol_binding_proven"] is False
                    and Path(r["path"]).name in ("libcublas.so.13", "libcublasLt.so.13"),
                    "GPU stat-only proof scope",
                )
                authenticate(r)
            catalog_identity = after
        else:
            require(
                {k: v for k, v in after.items() if k != "own_maps_sha256"}
                == {k: v for k, v in catalog_identity.items() if k != "own_maps_sha256"},
                "runtime catalog identical across calls",
            )
        diagnostic_refs.extend(
            [
                ref(target / "raw-response-pending.json"),
                ref(target / "evidence-pending.json"),
                ref(Path(q["private_feature_payload"]["path"])),
            ]
        )
        all_calls.append(
            {
                "index": index,
                "kind": kind,
                "actual_rows": rows,
                "events": events,
                "legal_counts": counts,
                "full_call_seconds": q["complete_prediction_seconds"],
                "all_actual_FP32_probabilities_logits_argmax_entropy_matched": True,
                "additional_stdlib_float64_witness": witness,
            }
        )
    counters = state["after_final_auth_owned_counters"]["counters"]
    require(
        all(v == 0 for v in counters["memory.events"].values()),
        "no memory event in original owned diagnostic",
    )
    return {
        "schema": "root-accepted-frozen-delta-cpu5-numeric-v1",
        "status": "accepted",
        "authority": "root",
        "independent_actual_review_claimed": False,
        "separate_reviewer_actual_bytes_read": False,
        "source_peer_review_scope": "source only",
        "root_exit": 0,
        "owner_completed": True,
        "current_original_scoped_closure_positive": True,
        "source_union": 60,
        "model_files_authenticated": 20,
        "runtime_files_authenticated": 28,
        "runtime_sha256_files": 22,
        "runtime_stat_only_files": 6,
        "root_execution": ref(terminal_path),
        "owner_receipt": ref(OUT / "receipt.json"),
        "current_closure": ref(OUT / "external-boundary-closure.json"),
        "selected_plan": ref(plan_path),
        "command_spec": ref(spec_path),
        "actor_source": ref(HERE / "actor_released_v1.py"),
        "audit_source": ref(Path(__file__)),
        "backend_source": plan["dependencies"]["backend"],
        "candidate_capability": plan["capability_ref"],
        "training_acceptance": plan["root_acceptance_ref"],
        "delta_ref": plan["delta_ref"],
        "baseline_fp32_bytes_sha256": capability["baseline_fp32_bytes_sha256"],
        "model": plan["model"],
        "candidate_alias": capability["alias"],
        "model_loads": 1,
        "genuine_forward_calls": 4,
        "calls": all_calls,
        "actor_seconds_before_final_retention": state["elapsed_before_final_retention_seconds"],
        "root_seconds": terminal["wall_seconds"],
        "owned_peak_bytes": counters["memory.peak"],
        "memory_events": counters["memory.events"],
        "numpy_critical_files": 22,
        "numpy_critical_bytes": sum(
            r["size_bytes"] for r in catalog_identity["critical_files_and_mapped_math_candidates"]
        ),
        "GPU_maps_stat_only": 2,
        "full_DSO_or_transitive_closure_qualified": False,
        "diagnostic_arrays_permanently_nontraining": True,
        "training_eligible": False,
        "semantic_correctness_evaluated": False,
        "browser_qualified": False,
        "benchmark_scores": None,
        "quality_or_win_claim": False,
        "diagnostic_evidence_refs": diagnostic_refs,
        "scope": "Root checked original terminal/current owner closure and60 sources, original20 model files and28 runtime identities (22 hashes/6 stat-only), actual legal FP32 candidate logits/replays/raws, independent all-legal stdlib FP64 witness, unchanged baseline byte join and true trained artifact. Numeric 4/2-row composition only; no independent actual reader or HTTP/browser/semantic/benchmark pass.",
    }


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = execute()
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "source_union",
                    "model_loads",
                    "genuine_forward_calls",
                    "root_seconds",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
