"""Held root audit of closed CPU5 operation-only numerical evidence.

Unexecuted source candidate. No model/scorer/NumPy is imported. Future root
selection must pin exact terminal/spec/plan references and source inventory
counts; no old shared-head or fixed-step qualification is inherited.
"""

import argparse
import array
import ast
import hashlib
import json
import math
import re
import struct
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OWNER = ROOT / "artifacts/jev-runtime/llama-cpp-owner-20261024-09"
SOURCE_ONLY = False
SELECTED_REFS = {
    "terminal": {
        "path": "/home/albert_hsieh/Gemma-4-12B-Unified-System-One/artifacts/jev-runtime/llama-cpp-owner-20261024-09/root-execution-operation-numeric-29.json",
        "sha256": "86f5f10567e8ea3b75b096236c4992d69c9a75e7d6ecf85c8c3699f47b4ee1a1",
        "size_bytes": 1759,
    },
    "spec": {
        "path": "/home/albert_hsieh/Gemma-4-12B-Unified-System-One/artifacts/jev-runtime/decision3-operation-numeric-qualification-cpu5-20261024-01/root-command-spec-01.json",
        "sha256": "3a312b8f8c0bb4c907596af0fcc6819c58308cffe9dd5b25c80c6de2f6fb8acf",
        "size_bytes": 13681,
    },
    "plan": {
        "path": "/home/albert_hsieh/Gemma-4-12B-Unified-System-One/artifacts/jev-runtime/decision3-operation-numeric-qualification-cpu5-20261024-01/selected-plan-01.json",
        "sha256": "f650e599662489a6283bda8b78de8bb5d3758275391f35fba87c81cf61660bb7",
        "size_bytes": 5894,
    },
}
SPEC_SOURCE_COUNT = 54
SOURCE_UNION_COUNT = 59
UNIT = "jev-d3-llama-cpp-cpu5-fast-4g-coexist-20261024-09.service"
PROFILE = "llama-cpp-cpu5-fast-same-cluster-4g-coexist-command-owned-v1"
CONTROLLER_SHA = "81846efe2ee0357da07283c1a77a606283a920cbddef668c8f62def291057dee"
NUMERIC_PROFILE = "decision3-operation-four-call-numeric-cpu5-4g-nontraining-v1"
CANDIDATE_PROFILE = "decision3-dsl01-operation-readout-cpu5-4g-candidate-v1"
CANDIDATE_ALIAS = "d3-edge-dsl01-operation-delta-cpu5-candidate"
HEADS = ("click_target", "operation", "select_target", "type_text_target")
COUNTS = (1, 8, 2, 3)
CALLS = (("original", 4), ("candidate", 4), ("original", 2), ("candidate", 2))


def require(value, message):
    if not value:
        raise ValueError(message)


def release_gate():
    require(SOURCE_ONLY is False, "SOURCE_ONLY: operation numeric root audit remains held")
    require(
        all(
            type(row) is dict
            and type(row.get("path")) is str
            and re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256"))) is not None
            and type(row.get("size_bytes")) is int
            and 0 < row["size_bytes"] <= 1048576
            for row in SELECTED_REFS.values()
        )
        and type(SPEC_SOURCE_COUNT) is int
        and SPEC_SOURCE_COUNT > 0
        and type(SOURCE_UNION_COUNT) is int
        and SOURCE_UNION_COUNT == SPEC_SOURCE_COUNT + 5,
        "root-selected terminal/spec/plan refs and exact source counts remain pending",
    )


def selected(name, supplied_path=None, supplied_sha=None):
    row = SELECTED_REFS[name]
    path = Path(row["path"])
    path = path if path.is_absolute() else ROOT / path
    require(
        path.is_relative_to(ROOT)
        and ".." not in path.parts
        and not any(p.is_symlink() for p in (path, *path.parents))
        and (supplied_path is None or supplied_path.absolute() == path)
        and (supplied_sha is None or supplied_sha == row["sha256"]),
        "exact root-selected reference required",
    )
    authenticate(row)
    return path, read(path), row["sha256"]


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


def npy_payload(row, shape, *, diagnostic, diagnostic_root=None):
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
            diagnostic_root is not None
            and path.is_relative_to(diagnostic_root)
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


def source_fixture(actor_path):
    """Read the selected actor's literal fixture, never import/execute it."""
    tree = ast.parse(actor_path.read_text())
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "fixture_request"
    ]
    require(len(functions) == 1, "one source-defined numerical fixture required")
    returns = [node for node in functions[0].body if isinstance(node, ast.Return)]
    require(len(returns) == 1, "one literal fixture return required")
    fixture = ast.literal_eval(returns[0].value)
    require(
        list(fixture) == ["state", "questions"]
        and type(fixture["state"]) is str
        and list(fixture["questions"]) == list(HEADS)
        and [len(q["criteria"]) for q in fixture["questions"].values()] == list(COUNTS)
        and all(q["type"] == "choice" for q in fixture["questions"].values()),
        "new fixture actual order/counts differ",
    )
    return fixture


def target_bits(actual_projection, raw, control, heads, counts):
    require(actual_projection["ordered_head_ids"] == list(heads), "actual projection head order")
    preserved = []
    for i, (head, count) in enumerate(zip(heads, counts)):
        if head == "operation":
            continue
        base = actual_projection["base_legal_logits"][i]
        adapted = actual_projection["adapted_legal_logits"][i]
        delta = actual_projection["delta_legal_logits"][i]
        actual = raw["answers"][head]["probabilities"]
        native = control["answers"][head]["probabilities"]
        require(
            len(base) == len(adapted) == len(delta) == len(actual) == len(native) == count
            and list(actual) == list(native)
            and all(type(v) in (int, float) and math.isfinite(v) for v in (*base, *adapted, *delta))
            and all(v == 0 for v in delta)
            and all(struct.pack("<f", a) == struct.pack("<f", b) for a, b in zip(base, adapted))
            and all(
                struct.pack("<d", a) == struct.pack("<d", b)
                for a, b in zip(actual.values(), native.values())
            ),
            "target logits bits or original-formatted target probabilities changed",
        )
        preserved.append(head)
    return preserved


def execute(
    *, terminal=None, terminal_sha=None, spec=None, spec_sha=None, plan=None, plan_sha=None
):
    release_gate()
    terminal_path, terminal, _ = selected("terminal", terminal, terminal_sha)
    spec_path, spec, SPEC_SHA = selected("spec", spec, spec_sha)
    plan_path, plan, PLAN_SHA = selected("plan", plan, plan_sha)
    require(
        type(terminal.get("argv")) is list and terminal["argv"].count("--output") == 1,
        "one selected original owner output argument required",
    )
    OUT = Path(terminal["argv"][terminal["argv"].index("--output") + 1])
    require(
        OUT.parent == OWNER and re.fullmatch(r"actual-[0-9]{2}", OUT.name) is not None,
        "fresh selected original-owner output branch required",
    )
    require(
        terminal_path.is_file() and (OUT / "receipt.json").is_file(),
        "Original root and owner must be terminal first",
    )
    receipt = read(OUT / "receipt.json")
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
    require(
        sha(spec_path) == SPEC_SHA
        and sha(plan_path) == PLAN_SHA
        and sha(OWNER / "controller.py") == CONTROLLER_SHA,
        "Selected source/plan pins differ",
    )
    require(
        plan["schema"] == "decision3-operation-four-call-numeric-plan-v1"
        and plan["profile"] == NUMERIC_PROFILE
        and plan["executable"] is True
        and plan["ordered_head_ids"] == list(HEADS)
        and plan["legal_counts"] == list(COUNTS)
        and plan["call_order"] == [list(item) for item in CALLS]
        and plan["wall_seconds"] == 120
        and plan["call_seconds"] == 20
        and plan["absolute_tolerance"] == 1e-6
        and plan["relative_tolerance"] == 1e-5
        and plan["training_eligible"] is False
        and plan["semantic_expected_answers"] is None,
        "new operation-only nontraining plan contract required",
    )
    for row in [plan["actor_source"], *plan["dependencies"].values()]:
        authenticate(row)
    base = [
        OWNER / "controller.py",
        spec_path,
        ROOT / "artifacts/jev-runtime/decision3-local-owner-terminal-20261022-01/owner_boundary.py",
        ROOT / "artifacts/jev-runtime/observer-system-services-20261022-01/comfy_observer.py",
        ROOT / "artifacts/jev-runtime/observer-system-services-20261022-01/system-services.json",
    ]
    expected = {r["path"]: r for r in [*spec["sources"], *(ref(p) for p in base)]}
    require(
        len(spec["sources"]) == SPEC_SOURCE_COUNT
        and len(expected) == SOURCE_UNION_COUNT
        and {r["path"]: r for r in config["sources"]} == expected,
        "Exact selected source count plus5 original source union required",
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
    require(plan["model"] == native_plan["model"], "exact original native model identity")
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
        and terminal["profile"] == "root-selected-operation-numeric-cpu5-terminal-v1",
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
    actor_path = Path(plan["actor_source"]["path"])
    actor_path = actor_path if actor_path.is_absolute() else ROOT / actor_path
    diagnostic_root = OUT / "operation-numeric"
    require(
        command["argv"]
        == [
            command["argv"][0],
            "-B",
            "-s",
            str(actor_path),
            "--plan",
            str(plan_path),
            "--plan-sha256",
            PLAN_SHA,
            "--output",
            str(diagnostic_root),
        ],
        "selected child must execute this exact authenticated actor and numerical plan",
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
    state_path = diagnostic_root / "state.json"
    state = read(state_path)
    require(
        state["profile"] == plan["profile"]
        and state["plan_sha256"] == PLAN_SHA
        and state["completed"] is True
        and state["status"] == "NUMERICAL_CANDIDATE_PENDING_ORIGINAL_ROOT_OWNER_CLOSURE_ACCEPTANCE"
        and state["model_loads"] == 1
        and state["native_request_calls"] == len(state["calls"]) == 4
        and state["training_eligible"] is False
        and state["labels"] is None
        and state["semantic_expected_answers"] is None
        and state["benchmark_scores"] is None
        and state["only_adapted_head_id"] == "operation"
        and state["candidate_alias"] == CANDIDATE_ALIAS
        and state["dev_used_for_model_selection"] is True
        and state["optimizer_input_dev"] is False
        and state["optimizer_input_sealed"] is False
        and state["candidate_selection_before_dev_or_sealed"] is False
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
        capability["status"] == "ROOT_ACCEPTED_OPERATION_TRAIN_ONLY_READOUT"
        and capability["schema"] == "root-accepted-dsl01-operation-train-delta-v1"
        and capability["delta_ref"] == plan["delta_ref"]
        and capability["root_acceptance_ref"] == plan["root_acceptance_ref"]
        and capability["model"] == plan["model"]
        and capability["split"] == "train"
        and capability["training_eligible"] is True
        and capability["nontraining"] is False
        and capability["only_head"] == "operation"
        and capability["profile"] == CANDIDATE_PROFILE
        and capability["alias"] == CANDIDATE_ALIAS
        and capability["owner_profile"] == PROFILE
        and all(
            capability[k] is expected and plan[k] is expected
            for k, expected in (
                ("dev_used_for_model_selection", True),
                ("optimizer_input_dev", False),
                ("optimizer_input_sealed", False),
                ("candidate_selection_before_dev_or_sealed", False),
            )
        )
        and acceptance["status"] == "accepted"
        and acceptance["schema"] == "root-accepted-dsl01-operation-trained-readout-v1"
        and acceptance["root_exit"] == 0
        and acceptance["owner_completed"] is True
        and acceptance["current_original_scoped_closure_positive"] is True,
        "new operation-only accepted train candidate and honest dev-selection joins",
    )
    authenticate(capability["fit_capability"])
    fit_input = read(Path(capability["fit_capability"]["path"]))
    require(
        fit_input["schema"] == "root-committed-train-operation-fit-v1"
        and fit_input["status"] == "ROOT_COMMITTED_TRAIN_OPERATION_FIT",
        "operation fitter input authority remains separate from accepted artifact",
    )
    delta_bytes, delta = npy_payload(plan["delta_ref"], (255, 1024), diagnostic=False)
    require(any(v != 0 for v in delta), "true nonzero delta artifact")
    require("fixture" not in plan["dependencies"], "old exposed fixture dependency refused")
    fixture = source_fixture(actor_path)
    baseline_bytes = None
    all_calls = []
    catalog_identity = None
    diagnostic_refs = []
    for index, (q, (kind, rows)) in enumerate(zip(state["calls"], plan["call_order"]), 1):
        target = diagnostic_root / f"call-{index:02d}-{kind}-{rows}"
        evidence = read(target / "evidence-pending.json")
        require(
            q["index"] == index
            and q["kind"] == kind
            and q["actual_rows_required"] == rows
            and q["completed"] is True
            and q["status"] == "NUMERICAL_CANDIDATE_PENDING_OUTER_CLEANUP_AND_ROOT_ACCEPTANCE"
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
        require(
            list(questions) == list(HEADS[:rows]) and counts == list(COUNTS[:rows]),
            "actual operation-second 4/2 head projection required",
        )
        hbytes, h = npy_payload(
            q["private_feature_payload"],
            (rows, 1024),
            diagnostic=True,
            diagnostic_root=diagnostic_root,
        )
        wbytes, w = npy_payload(
            q["private_baseline_payload"],
            (255, 1024),
            diagnostic=True,
            diagnostic_root=diagnostic_root,
        )
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
        control = q["native_response_control"]
        native_control_probabilities = probabilities(
            control, questions, counts, plan["model"]["name"]
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
        require(
            control["usage"]["input_tokens"] == sum(lengths.values()),
            "same-forward native formatted control keeps actual input usage",
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
        require(
            actual_projection["ordered_head_ids"] == list(questions)
            and actual_projection["target_full_logits_bitwise_unchanged"] is True
            and actual_projection["target_probabilities_unchanged"] is True,
            "independent original/adapted FP32 target witness required",
        )
        if kind == "candidate":
            a = q["candidate_backend_evidence"]
            chunk = a["chunks"][0]
            require(
                a["status"] == "completed"
                and a["profile"] == CANDIDATE_PROFILE
                and a["alias"] == CANDIDATE_ALIAS
                and a["only_adapted_head_id"] == "operation"
                and a["capability_sha256"] == plan["capability_ref"]["sha256"]
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
            require(
                chunk["projection"]["ordered_head_ids"] == list(questions)
                and chunk["projection"]["adapted_head_ids"] == ["operation"]
                and chunk["projection"]["preserved_head_ids"]
                == [head for head in questions if head != "operation"]
                and chunk["projection"]["target_logits_bitwise_unchanged"] is True,
                "operation-only candidate row identity and preservation proof required",
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
        observed_projection = chunk["projection"] if kind == "candidate" else actual_projection
        preserved = target_bits(observed_projection, raw, control, list(questions), counts)
        require(
            q["independent_target_preservation"]
            == {
                "ordered_head_ids": list(questions),
                "preserved_head_ids": preserved,
                "target_legal_logits_bitwise_unchanged": True,
                "target_native_probabilities_bitwise_unchanged": True,
            },
            "actor private target proof joins exact retained raw/control and logit bits",
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
                if kind == "candidate" and head == "operation"
                else [0.0] * count
            )
            logits = (
                [a + b for a, b in zip(base_logits, delta_logits)]
                if kind == "candidate" and head == "operation"
                else base_logits
            )
            e = [math.exp(x - max(logits)) for x in logits]
            total = math.fsum(e)
            reference = [x / total for x in e]
            require(
                all(mixed(a, b) for a, b in zip(reference, p))
                and max(range(count), key=lambda k: reference[k])
                == max(range(count), key=lambda k: p[k]),
                "independent stdlib all-legal FP64 probability/ordered-argmax witness",
            )
            native_exps = [math.exp(x - max(base_logits)) for x in base_logits]
            native_total = math.fsum(native_exps)
            native_reference = [x / native_total for x in native_exps]
            require(
                all(mixed(a, b) for a, b in zip(native_reference, native_control_probabilities[j]))
                and max(range(count), key=native_reference.__getitem__)
                == max(range(count), key=native_control_probabilities[j].__getitem__),
                "independent FP64 baseline vs original-formatted native control",
            )
            witness.append(
                {
                    "head": head,
                    "legal_count": count,
                    "maximum_probability_absolute_error": max(
                        abs(a - b) for a, b in zip(reference, p)
                    ),
                    "ordered_argmax_matched": True,
                    "delta_applied": kind == "candidate" and head == "operation",
                    "target_preserved": head != "operation",
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
                "ordered_head_ids": list(questions),
                "target_logit_bits_and_original_formatted_probabilities_unchanged": True,
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
        "schema": "root-accepted-operation-cpu5-numeric-v1",
        "status": "accepted",
        "authority": "root",
        "independent_actual_review_claimed": False,
        "separate_reviewer_actual_bytes_read": False,
        "source_peer_review_scope": "source only",
        "root_exit": 0,
        "owner_completed": True,
        "current_original_scoped_closure_positive": True,
        "source_union": SOURCE_UNION_COUNT,
        "model_files_authenticated": 20,
        "runtime_files_authenticated": 28,
        "runtime_sha256_files": 22,
        "runtime_stat_only_files": 6,
        "root_execution": ref(terminal_path),
        "owner_receipt": ref(OUT / "receipt.json"),
        "current_closure": ref(OUT / "external-boundary-closure.json"),
        "selected_plan": ref(plan_path),
        "command_spec": ref(spec_path),
        "actor_source": plan["actor_source"],
        "audit_source": ref(Path(__file__)),
        "backend_source": plan["dependencies"]["backend"],
        "candidate_capability": plan["capability_ref"],
        "training_acceptance": plan["root_acceptance_ref"],
        "delta_ref": plan["delta_ref"],
        "baseline_fp32_bytes_sha256": capability["baseline_fp32_bytes_sha256"],
        "model": plan["model"],
        "candidate_alias": capability["alias"],
        "candidate_profile": CANDIDATE_PROFILE,
        "only_adapted_head_id": "operation",
        "dev_used_for_model_selection": True,
        "optimizer_input_dev": False,
        "optimizer_input_sealed": False,
        "candidate_selection_before_dev_or_sealed": False,
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
        "scope": "Root checked this selected terminal/current owner closure and exact pinned source union, original20 model files and28 runtime identities (22 hashes/6 stat-only), actual legal operation-only FP32 logits/replays/raws, original publisher-formatted same-forward control, target legal logit bits/exact control probabilities, independent all-legal operation-only stdlib FP64 witness, unchanged baseline bytes and new operation artifact. Numeric 4/2-row composition only; dev-informed candidate selection is disclosed. No shared-head numeric qualification or fixed-step identity inherited; no independent actual reader or HTTP/browser/semantic/benchmark pass.",
    }


def main():
    release_gate()  # Before argument parsing or any artifact access.
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("terminal", "spec", "plan"):
        parser.add_argument("--" + name, type=Path)
        parser.add_argument("--" + name + "-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = execute(
        terminal=args.terminal,
        terminal_sha=args.terminal_sha256,
        spec=args.spec,
        spec_sha=args.spec_sha256,
        plan=args.plan,
        plan_sha=args.plan_sha256,
    )
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
