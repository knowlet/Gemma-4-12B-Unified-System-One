"""HELD DSL01 dev-only frozen-candidate feature collection; no labels or optimizer.

Ported from the unchanged train collector by source only. No check/import/model
execution is claimed. Development features remain permanently nontraining.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE_ONLY = False
DEV_CAPABILITY_SHA256 = "917f208594a715ad19dc788682f5a41bcbe55c4667d1f726a2c4d909cd01aed2"
NUMERIC_ACCEPTANCE_SHA256 = "91f80677a7956dbdee721661c3f20b8dfcbf5df8d7948b7221debdbd84d174d1"
FROZEN_CANDIDATE_SHA256 = "239165d486bad16963a6e5c94c1f9503fc145ea5b85b114e9fbf94d84eb1fd76"
FROZEN_TRAIN_ACCEPTANCE_SHA256 = "cc629a1893ce53c232cc323253f39e86496b0e7a7347c5d11df8fa259ccb5baa"
PROFILE = "decision3-dsl01-dev-features-frozen-cpu5-4g-candidate-v1"
SCHEMA = "decision3-dsl01-dev-feature-chunk-v1"
ROLE = "dev_feature_collector"
SPLIT = "dev"
DSL_VERSION = "independent-browser-workflow-dsl/1"
DSL = ROOT / "artifacts/jev-runtime/browser-decision-dsl-20261023-01/corpus/data/dev"
NATIVE = ROOT / "artifacts/jev-runtime/decision3-publisher-batch4-cpu-20261024-05"
NP5 = ROOT / "artifacts/jev-runtime/decision3-readout-numpy-cpu5-qualification-20261024-02"
CORE = ROOT / "artifacts/jev-runtime/decision3-cached-head-core-20261023-01/head_core.py"
OWNER = ROOT / "artifacts/jev-runtime/llama-cpp-owner-20261024-09"
UNIT = "jev-d3-llama-cpp-cpu5-fast-4g-coexist-20261024-09.service"
OWNER_PROFILE = "llama-cpp-cpu5-fast-same-cluster-4g-coexist-command-owned-v1"
NON_DEV_TRAIN_FAMILIES = frozenset(
    {"appointment_edit", "document_upload", "booking_inventory", "admin_filters"}
)
HEADS = frozenset({"operation", "click_target", "type_text_target", "select_target"})
WALL_SECONDS, CALL_SECONDS = 120, 20
PENDING = "PENDING_DEV_NONTRAINING"
MODEL_MODULES = {"torch", "transformers", "safetensors", "numpy", "s1"}
IDENTITY_KEYS = {
    "case_id",
    "dsl_version",
    "family_id",
    "ancestor_id",
    "request_sha256",
    "native_payload_sha256",
}
DEPENDENCIES = {
    "native_actor": NATIVE / "actor.py",
    "native_plan": NATIVE / "edge-plan.json",
    "native_helper": NATIVE / "resource_helper.py",
    "np5_actor": NP5 / "actor.py",
    "observer": NP5 / "observer.py",
    "numpy_probe": NP5 / "numpy_probe.py",
    "head_core": CORE,
    "owner_controller": OWNER / "controller.py",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def release_gate():
    require(
        SOURCE_ONLY is False,
        "SOURCE_ONLY: no arguments, plan, imports, output, model or collection authorized",
    )
    require(
        all(
            type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value)
            for value in (DEV_CAPABILITY_SHA256, NUMERIC_ACCEPTANCE_SHA256)
        ),
        "Selected dev capability/numeric acceptance pins remain pending",
    )


def canonical(value):
    # Exact DSL01 digest encoding, including original mapping insertion order.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def ordinary(path):
    path = Path(path).absolute()
    require(
        path.is_relative_to(ROOT) and ".." not in path.parts, "Ordinary repository path required"
    )
    dsl_root = DSL.parents[2]
    require(
        not path.is_relative_to(dsl_root) or path in {DSL / "requests.jsonl", DSL / "summary.json"},
        "Dev collector authoring/train/sealed/labels input refused before open",
    )
    require(
        all(not item.is_symlink() for item in (path, *path.parents)),
        "Selected paths cannot contain symlinks",
    )
    return path


def read(path, maximum=4 * 1024**2):
    path = ordinary(path)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(before.st_mode) and 0 < before.st_size <= maximum,
            "Bounded regular selected input required",
        )
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(
        len(raw) == before.st_size
        and all(getattr(before, k) == getattr(after, k) == getattr(path.stat(), k) for k in keys),
        "Selected input changed during read",
    )
    return raw


def pairs(entries):
    result = {}
    for key, value in entries:
        require(key not in result, "Duplicate JSON key")
        result[key] = value
    return result


def decode(raw):
    return json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_constant=lambda _v: require(False, "Nonfinite JSON refused"),
    )


def verify_record(record, expected_path=None):
    require(
        type(record) is dict and set(record) == {"path", "sha256", "size_bytes"},
        "Exact selected file identity required",
    )
    require(
        re.fullmatch(r"[0-9a-f]{64}", record["sha256"] or "") and type(record["size_bytes"]) is int,
        "Pending or malformed file pin",
    )
    path = ordinary(record["path"])
    require(
        expected_path is None or path == expected_path,
        "Selected path outside fixed input projection",
    )
    raw = read(path)
    require(
        len(raw) == record["size_bytes"] and hashlib.sha256(raw).hexdigest() == record["sha256"],
        "Selected file bytes differ",
    )
    return raw


def source(name, record):
    raw = verify_record(record)
    path = Path(record["path"])
    require(
        path.suffix == ".py" and len(raw) < 100_000 and name not in sys.modules,
        "Fresh bounded Python dependency required",
    )
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "Dependency loader unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def capability_inventory(capability):
    """Root-selected dev ancestry metadata; no authoring or label import."""
    families = capability.get("family_ids")
    require(
        capability.get("role") == ROLE
        and capability.get("split") == SPLIT
        and capability.get("dsl_version") == DSL_VERSION
        and type(capability.get("case_count")) is int
        and capability["case_count"] == 24
        and type(families) is list
        and len(families) == 1
        and type(families[0]) is str
        and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", families[0])
        and families[0] not in NON_DEV_TRAIN_FAMILIES,
        "Exact root-selected dev role/24-case/one-family metadata required",
    )
    ancestors = [f"{families[0]}/workflow-{i}" for i in (1, 2, 3)]
    require(
        capability.get("ancestor_ids") == ancestors, "Exact three existing dev ancestors required"
    )
    return families, ancestors


def frozen_candidate(plan):
    """Trust only the exact root acceptance, without reading train arrays/data."""
    require(
        plan["frozen_candidate_ref"]["sha256"] == FROZEN_CANDIDATE_SHA256
        and plan["frozen_train_acceptance_ref"]["sha256"] == FROZEN_TRAIN_ACCEPTANCE_SHA256
        and plan["numeric_root_acceptance_ref"]["sha256"] == NUMERIC_ACCEPTANCE_SHA256,
        "Frozen candidate/train acceptance/numeric root pins differ",
    )
    candidate = decode(verify_record(plan["frozen_candidate_ref"]))
    training = decode(verify_record(plan["frozen_train_acceptance_ref"]))
    numeric = decode(verify_record(plan["numeric_root_acceptance_ref"]))
    require(
        candidate.get("schema") == "root-accepted-dsl01-train-delta-v1"
        and candidate.get("status") == "ROOT_ACCEPTED_TRAIN_ONLY_READOUT"
        and candidate.get("split") == "train"
        and candidate.get("training_eligible") is True
        and candidate.get("nontraining") is False
        and candidate.get("model") == plan["model"]
        and candidate.get("frozen_before_dev_sealed_benchmark_or_browser") is True
        and candidate.get("W0_merge_allowed") is False
        and candidate.get("quality_or_win_claim") is False
        and candidate.get("root_acceptance_ref") == plan["frozen_train_acceptance_ref"],
        "Exact train-only candidate must already be frozen before dev exposure",
    )
    require(
        training.get("schema") == "root-accepted-dsl01-trained-readout-v1"
        and training.get("status") == "accepted"
        and type(training.get("root_exit")) is int
        and training["root_exit"] == 0
        and training.get("owner_completed") is True
        and training.get("current_original_scoped_closure_positive") is True,
        "Genuine closed trained-readout root acceptance required",
    )
    require(
        numeric.get("schema") == "root-accepted-frozen-delta-cpu5-numeric-v1"
        and numeric.get("status") == "accepted"
        and numeric.get("authority") == "root"
        and numeric.get("independent_actual_review_claimed") is False
        and type(numeric.get("root_exit")) is int
        and numeric["root_exit"] == 0
        and numeric.get("owner_completed") is True
        and numeric.get("current_original_scoped_closure_positive") is True
        and numeric.get("model") == plan["model"]
        and numeric.get("candidate_capability") == plan["frozen_candidate_ref"]
        and numeric.get("training_acceptance") == plan["frozen_train_acceptance_ref"]
        and numeric.get("delta_ref") == candidate.get("delta_ref")
        and numeric.get("training_eligible") is False
        and numeric.get("semantic_correctness_evaluated") is False
        and numeric.get("browser_qualified") is False,
        "Closed genuine numeric composition prerequisite required, not dev quality",
    )
    for key in ("root_execution", "owner_receipt", "current_closure"):
        verify_record(numeric[key])
    return candidate


def validate_plan(plan, path, expected_sha):
    require(
        plan.get("schema") == "decision3-dsl01-dev-collector-plan-v1"
        and plan.get("profile") == PROFILE
        and plan.get("status") == "selected_after_separate_review",
        "Pending/other collector plan refused",
    )
    require(
        plan.get("dsl_version") == DSL_VERSION
        and plan.get("split") == SPLIT
        and plan.get("role") == ROLE
        and plan.get("training_eligible") is False
        and plan.get("requests") == 8
        and plan.get("wall_seconds") == 120
        and plan.get("call_seconds") == 20,
        "Fixed nontraining dev chunk and original budgets required",
    )
    require(
        plan.get("owner_unit") == UNIT
        and plan.get("owner_profile") == OWNER_PROFILE
        and plan.get("memory_max_bytes") == 4294967296
        and plan.get("swap_max_bytes") == 0
        and plan.get("reserve_bytes") == 26119172916
        and plan.get("startup_available_bytes") == 30414140212
        and plan.get("cpu_affinity") == [15, 16, 17, 18, 19]
        and plan.get("cpu_quota_percent") == 500
        and plan.get("threads") == 5,
        "Original CPU5/full4GiB owner contract required",
    )
    require(
        set(plan["dependencies"]) == set(DEPENDENCIES),
        "Exact executed dependency projection required",
    )
    for key, dependency in DEPENDENCIES.items():
        verify_record(plan["dependencies"][key], dependency)
    verify_record(plan["collector_source"], Path(__file__).absolute())
    require(
        type(expected_sha) is str
        and len(expected_sha) == 64
        and hashlib.sha256(read(path)).hexdigest() == expected_sha,
        "Exact selected collector plan required",
    )
    require(
        set(plan["dev_inputs"]) == {"requests", "summary"},
        "Dev collector cannot select labels, train or sealed artifacts",
    )
    verify_record(plan["dev_inputs"]["requests"], DSL / "requests.jsonl")
    verify_record(plan["dev_inputs"]["summary"], DSL / "summary.json")
    prerequisite = decode(verify_record(plan["np5_root_acceptance"]))
    required_checks = {
        "root_exit0",
        "owner_completed",
        "sources_unchanged",
        "current_closure_positive",
        "full_authentication",
        "all_legal_probabilities_matched",
        "all_ordered_argmaxes_matched",
        "FP32_copy_exact",
    }
    require(
        prerequisite.get("schema") == "root-accepted-np5-qualification-v1"
        and prerequisite.get("status") == "accepted"
        and prerequisite.get("independent_actual_review_claimed") is False
        and set(prerequisite.get("checks", {})) == required_checks
        and all(prerequisite["checks"][key] is True for key in required_checks),
        "Genuine closed NP5 root prerequisite remains pending",
    )
    require(
        prerequisite.get("source_refs") == plan["dependencies"]
        and prerequisite.get("owner_unit") == UNIT
        and prerequisite.get("owner_profile") == OWNER_PROFILE,
        "NP5 root prerequisite is for different source/native/owner bytes",
    )
    require(
        set(prerequisite.get("proof_refs", {}))
        == {"root_execution", "owner_receipt", "current_closure"},
        "NP5 acceptance must retain exact original terminal/owner/current-closure refs",
    )
    for record in prerequisite["proof_refs"].values():
        verify_record(record)
    catalog = decode(verify_record(prerequisite["numpy_runtime_catalog"]))
    require(
        catalog.get("status") == "PENDING_CLOSED_ROOT_ACCEPTANCE"
        and catalog.get("training_eligible") is False
        and type(catalog.get("after")) is dict,
        "Original closed NP5 runtime catalog metadata required; no diagnostic arrays",
    )
    require(
        plan["dev_capability"]["sha256"] == DEV_CAPABILITY_SHA256,
        "Exact selected dev export capability pin required",
    )
    capability = decode(verify_record(plan["dev_capability"]))
    require(
        capability.get("schema") == "root-selected-dsl01-dev-collector-capability-v1"
        and capability.get("status") == "ROOT_SELECTED_FROZEN_DEV_COLLECTION"
        and capability.get("dev_inputs") == plan["dev_inputs"]
        and capability.get("frozen_candidate_ref") == plan["frozen_candidate_ref"]
        and capability.get("numeric_root_acceptance_ref") == plan["numeric_root_acceptance_ref"]
        and capability.get("training_eligible") is False
        and capability.get("labels_allowed") is False
        and capability.get("sealed_reveal_authorized") is False,
        "Root dev-only nontraining export capability required",
    )
    _families, ancestors = capability_inventory(capability)
    require(
        plan.get("ancestor_id") in ancestors, "Selected ancestor is not in exact dev capability"
    )
    frozen_candidate(plan)
    return catalog["after"], capability


def select_cases(request_bytes, summary_bytes, ancestor, capability):
    """Pure dev-export validation; never import reconstructive authoring code."""
    summary = decode(summary_bytes)
    families, ancestors = capability_inventory(capability)
    require(
        summary
        == {
            "dsl_version": DSL_VERSION,
            "split": SPLIT,
            "case_count": 24,
            "family_ids": families,
            "ancestor_ids": ancestors,
            "is_authored_synthetic_not_NN": True,
        },
        "Exact DSL01 dev summary required",
    )
    require(ancestor in ancestors, "Unknown/non-dev ancestor refused")
    rows = [decode(line) for line in request_bytes.splitlines() if line.strip()]
    require(
        len(rows) == 24 and len({row["case_id"] for row in rows}) == 24,
        "Exact unique dev request population required",
    )
    selected = []
    counts = {key: 0 for key in ancestors}
    for row in rows:
        require(
            set(row) == IDENTITY_KEYS | {"request"}
            and row["dsl_version"] == DSL_VERSION
            and row["family_id"] in families
            and row["ancestor_id"] in ancestors
            and row["ancestor_id"].split("/")[0] == row["family_id"]
            and re.fullmatch(r"[0-9a-f]{24}", row["case_id"]),
            "Malformed/non-dev request lineage",
        )
        request = row["request"]
        projection = {key: request[key] for key in ("state", "questions")}
        require(
            digest(request) == row["request_sha256"]
            and digest(projection) == row["native_payload_sha256"],
            "Original request/projection SHA mismatch",
        )
        questions = projection["questions"]
        require(
            type(questions) is dict
            and len(questions) == 4
            and set(questions) == HEADS
            and next(iter(questions)) == "operation",
            "All four original ordered heads required",
        )
        for key, question in questions.items():
            require(
                question.get("type") == "choice"
                and type(question.get("criteria")) is dict
                and len(question["criteria"]) == (8 if key == "operation" else 4),
                "Original DSL01 head/count projection required",
            )
        counts[row["ancestor_id"]] += 1
        if row["ancestor_id"] == ancestor:
            selected.append(row)
    require(
        all(count == 8 for count in counts.values()) and len(selected) == 8,
        "All eight existing branches per dev ancestor required",
    )
    return selected


def mapping(row, index):
    return [
        {
            "case_id": row["case_id"],
            "native_payload_sha256": row["native_payload_sha256"],
            "head_id": key,
            "ordered_criteria_ids": list(question["criteria"]),
            "legal_count": len(question["criteria"]),
            "h_index": position,
            "same_call_id": f"call-{index:04d}",
            "native_call_index": index,
        }
        for position, (key, question) in enumerate(row["request"]["questions"].items())
    ]


def file_record(path, output):
    raw = read(path)
    return {
        "path": str(Path(path).relative_to(output)),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "split": SPLIT,
        "status": PENDING,
        "training_eligible": False,
    }


def check_catalog(actual, accepted, *, complete):
    """Exact closed NP5 critical projection; no full/transitive DSO assertion."""
    for key in ("numpy_version", "python", "numpy_build_configuration"):
        require(
            actual.get(key) == accepted.get(key),
            "Actual NumPy/runtime/build differs from closed NP5 prerequisite",
        )
    expected_rows = accepted["critical_files_and_mapped_math_candidates"]
    actual_rows = actual["critical_files_and_mapped_math_candidates"]
    expected = {row["resolved_path"]: row for row in expected_rows}
    observed = {row["resolved_path"]: row for row in actual_rows}
    require(
        len(expected) == len(expected_rows)
        and len(observed) == len(actual_rows)
        and all(expected.get(path) == row for path, row in observed.items()),
        "Actual critical/mapped file identity differs from closed NP5 prerequisite",
    )
    gpu_key = "mapped_gpu_math_candidates_excluded_from_CPU_projection"
    require(
        actual.get(gpu_key) == accepted.get(gpu_key),
        "Actual GPU stat-only projection differs from closed NP5 prerequisite",
    )
    if complete:
        require(
            observed == expected
            and actual["mapped_math_candidates"] == accepted["mapped_math_candidates"],
            "Complete observed NP5 critical/mapped projection required after computation",
        )


def authenticate_catalog_before_import(probe, native, helper, accepted, began):
    rows = accepted["critical_files_and_mapped_math_candidates"]
    gpu = accepted["mapped_gpu_math_candidates_excluded_from_CPU_projection"]
    require(
        0 < len(rows) + len(gpu) <= probe.MAX_SELECTED_FILES
        and sum(row["size_bytes"] for row in rows) <= probe.MAX_SELECTED_BYTES,
        "Original closed CPU/GPU catalog bounds required before numerical import",
    )
    for row in rows:
        require(
            probe.file_identity(row["path"], native, helper, began) == row,
            "Closed CPU catalog bytes differ before numerical import",
        )
    for row in gpu:
        require(
            probe.gpu_candidate_metadata(row["path"], native, helper, began) == row,
            "Closed GPU stat-only catalog differs before numerical import",
        )


def numerical_context(shared, probe, native, helper, state, core_record, accepted, began):
    if not shared:
        authenticate_catalog_before_import(probe, native, helper, accepted, began)
        np = probe.load_numpy(native, helper, state)
        core = source("_dev_collector_head_core", core_record)
        shared.update(np=np, core=core)
    return shared["np"], shared["core"]


def sync_directories(paths, primary):
    try:
        for path in paths:
            descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except BaseException as secondary:
        if primary is None:
            raise
        primary.add_note("Directory sync also failed: " + repr(secondary))


def save_array(np, array, path, shape):
    require(
        type(array) is np.ndarray
        and array.dtype == np.float32
        and array.shape == shape
        and array.flags.c_contiguous
        and bool(np.isfinite(array).all()),
        "Exact independent finite FP32 payload required",
    )
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def retain_call(
    np5,
    native,
    helper,
    observer_module,
    probe,
    shared,
    core_record,
    runtime,
    torch,
    base,
    item,
    index,
    output,
    state,
    baseline,
    accepted_catalog,
):
    began, handles, published = time.monotonic(), [], {}
    case_output = output / "cases" / item["case_id"]
    case_output.mkdir(mode=0o700)
    request = {key: copy.deepcopy(item["request"][key]) for key in ("state", "questions")}
    row = {
        "case_id": item["case_id"],
        "split": SPLIT,
        "role": ROLE,
        "labels": None,
        "completed": False,
        "status": PENDING,
        "training_eligible": False,
    }
    state["calls"].append(row)
    evidence = {
        "schema": "decision3-dsl01-dev-case-evidence-v1",
        "split": SPLIT,
        "role": ROLE,
        "labels": None,
        "status": PENDING,
        "training_eligible": False,
        "identity": {
            **{k: item[k] for k in ("case_id", "request_sha256", "native_payload_sha256")},
            "same_call_id": f"call-{index:04d}",
            "native_call_index": index,
        },
        "ordered_heads": list(request["questions"]),
        "rows": mapping(item, index),
        "native_payload": request,
        "completed": False,
    }

    def demote(error):
        row["completed"] = evidence["completed"] = False
        row["error"] = evidence["error"] = {"type": type(error).__name__, "message": str(error)}

    def finalize(primary):
        evidence["timing"] = {
            "budget_seconds": 20,
            "elapsed_before_last_evidence_write_seconds": time.monotonic() - began,
            "hooks_removed": not handles,
            "post_watchdog_check_required": True,
        }
        try:
            helper.write_json(case_output / "evidence.json", evidence)
        except BaseException as secondary:
            if primary is None:
                raise
            primary.add_note("Case evidence write also failed: " + repr(secondary))
        if primary is None:
            # File identities are charged to the same20s before its final gate.
            published.update(
                features={
                    **file_record(case_output / "features.npy", output),
                    "format": "npy_no_pickle",
                    "dtype": "float32",
                    "shape": [4, 1024],
                },
                raw=file_record(case_output / "raw-native-response.json", output),
                evidence=file_record(case_output / "evidence.json", output),
            )
        helper.finalize_state(output, state, primary)
        sync_directories((case_output, case_output.parent, output), primary)

    def operation():
        np, core = numerical_context(
            shared, probe, native, helper, state, core_record, accepted_catalog, began
        )
        native.validate_runtime(runtime, torch, base["model"])
        probe.owned(native, helper)
        before = observer_module.baseline_identity(runtime)
        initial_catalog = probe.catalog(np, native, helper, began)
        check_catalog(initial_catalog, accepted_catalog, complete=False)
        require(
            runtime.encoder.language_model is runtime.backbone.language_model,
            "Original encoder/final-norm join differs",
        )

        def sink(kind, value):
            if kind == "raw_returned_before_validation":
                # The exact native result is retained without normalizing/relabeling.
                helper.write_json(case_output / "raw-native-response.json", value)

        backend = native.NativeBatch4Backend(runtime, torch, base["model"], evidence_sink=sink)
        prepared, lengths = backend.prepare(**request)
        keys = list(request["questions"])
        require(
            prepared.keys == keys
            and list(lengths) == keys
            and list(prepared.sequences) == keys
            and not prepared.errors
            and max(lengths.values()) <= 16384,
            "Original complete ordered prompts required; nothing truncated",
        )
        counts = [len(request["questions"][key]["criteria"]) for key in keys]
        require(
            all(
                list(prepared.questions[key].keys) == list(request["questions"][key]["criteria"])
                for key in keys
            ),
            "Native option slots differ from original criteria order",
        )
        witness = observer_module.FinalNormObserver(torch, 4, max(lengths.values()))
        handles.append(
            runtime.backbone.language_model.layers[0].register_forward_pre_hook(witness.first_layer)
        )
        handles.append(runtime.backbone.language_model.norm.register_forward_hook(witness))
        state["native_request_calls"] += 1
        raw = backend(**copy.deepcopy(request))
        evidence["native_contract"] = probe.validate_answers(
            native, raw, request, lengths, base["model"]
        )
        witness.require_single_pass()
        torch_replay = observer_module.replay_probabilities(runtime, torch, witness, counts)
        torch_result = observer_module.compare_distributions(
            raw, request["questions"], torch_replay
        )
        features = probe.copy_fp32(np, torch, witness.features, (4, 1024))
        w0 = probe.copy_fp32(np, torch, runtime.readout, (255, 1024))
        array_before = [hashlib.sha256(a.tobytes(order="C")).hexdigest() for a in (features, w0)]
        if not baseline:
            save_array(np, w0, output / "baseline.npy", (255, 1024))
            baseline["bytes_sha256"] = hashlib.sha256(w0.tobytes(order="C")).hexdigest()
        require(
            baseline["bytes_sha256"] == hashlib.sha256(w0.tobytes(order="C")).hexdigest(),
            "W0 changed between same-load dev calls",
        )
        zero = np.zeros((255, 1024), dtype=np.float32)
        zero.flags.writeable = False
        legal = np.asarray(counts, dtype=np.int64)
        legal.flags.writeable = False
        probabilities = core.probabilities(features, w0, zero, legal)
        require(
            probabilities.shape == (4, 255)
            and probabilities.dtype == np.float32
            and bool(np.isfinite(probabilities).all())
            and all(bool((probabilities[i, n:] == 0).all()) for i, n in enumerate(counts)),
            "Exact finite legal NumPy projection required",
        )
        numpy_result = observer_module.compare_distributions(
            raw, request["questions"], [p[:n].tolist() for p, n in zip(probabilities, counts)]
        )
        require(
            array_before
            == [hashlib.sha256(a.tobytes(order="C")).hexdigest() for a in (features, w0)],
            "Projection modified immutable feature/baseline copies",
        )
        save_array(np, features, case_output / "features.npy", (4, 1024))
        final_catalog = probe.catalog(np, native, helper, began)
        check_catalog(final_catalog, accepted_catalog, complete=True)
        prior = {
            record["resolved_path"]: record
            for record in initial_catalog["critical_files_and_mapped_math_candidates"]
        }
        after = {
            record["resolved_path"]: record
            for record in final_catalog["critical_files_and_mapped_math_candidates"]
        }
        require(
            all(after.get(path) == record for path, record in prior.items()),
            "Actual NumPy/math identity changed",
        )
        require(
            observer_module.baseline_identity(runtime) == before
            and torch.equal(torch.from_numpy(w0.copy()), runtime.readout.detach()),
            "Original readout/backbone changed",
        )
        native.validate_runtime(runtime, torch, base["model"])
        native.authenticate(helper, base)
        evidence.update(
            math={
                "feature_shape": [4, 1024],
                "baseline_shape": [255, 1024],
                "temperature": 1.0,
                "FP32_copy_exact": True,
                "first_layer_events": witness.first_layer_events,
                "final_norm_events": witness.norm_events,
                "baseline_tensor_identity_and_versions_unchanged": True,
                "torch_replay": torch_result,
                "numpy_zero_delta": numpy_result,
                "illegal_probabilities_exact_zero": True,
            },
            input_tokens_per_question=lengths,
            native_full_authentication_unchanged=True,
            numpy_runtime=final_catalog,
            completed=True,
        )
        row["completed"] = True
        return row

    result, elapsed = np5.transaction(
        native,
        helper,
        output,
        f"dev-call-{index:04d}",
        began,
        CALL_SECONDS,
        operation,
        lambda: np5.remove_handles(handles),
        finalize,
        demote,
    )
    result["complete_prediction_seconds"] = elapsed
    return {
        **{key: item[key] for key in IDENTITY_KEYS},
        "split": SPLIT,
        "role": ROLE,
        "status": PENDING,
        "training_eligible": False,
        "labels": None,
        "ordered_heads": list(request["questions"]),
        "rows": mapping(item, index),
        **published,
        "complete_prediction_seconds": elapsed,
    }


def execute(plan_path, expected_sha, output):
    release_gate()
    began = time.monotonic()
    require(
        sys.flags.no_user_site
        and sys.dont_write_bytecode
        and not MODEL_MODULES.intersection(sys.modules),
        "Fresh -B -s process without model libraries required",
    )
    raw_plan = read(plan_path)
    require(
        hashlib.sha256(raw_plan).hexdigest() == expected_sha, "Selected collector plan SHA differs"
    )
    plan = decode(raw_plan)
    accepted_catalog, dev_capability = validate_plan(plan, plan_path, expected_sha)
    selected = select_cases(
        verify_record(plan["dev_inputs"]["requests"]),
        verify_record(plan["dev_inputs"]["summary"]),
        plan["ancestor_id"],
        dev_capability,
    )
    native = source("_dev_collector_native05", plan["dependencies"]["native_actor"])
    helper = native.load_helper()
    base = native.load_plan(
        helper, NATIVE / "edge-plan.json", plan["dependencies"]["native_plan"]["sha256"]
    )
    require(base["model"] == plan["model"], "Frozen candidate/native model identity differs")
    np5 = source("_dev_collector_np5", plan["dependencies"]["np5_actor"])
    observer_module = source("_dev_collector_observer", plan["dependencies"]["observer"])
    probe = source("_dev_collector_numpy_probe", plan["dependencies"]["numpy_probe"])
    identity = helper.process_identity()
    limits = helper.owned_limits(identity)
    native.validate_owner(identity, limits, helper.START_ENV)
    gate = helper.host_gate()
    output = ordinary(output)
    require(
        output.name == "dev-chunk"
        and output.parent.parent == OWNER
        and re.fullmatch(r"actual-[0-9]{2}", output.parent.name)
        and not output.exists(),
        "Fresh current original-owner dev output only",
    )
    output.mkdir(mode=0o700)
    (output / ".gitignore").write_text("*\n", encoding="utf-8")
    (output / "cases").mkdir(mode=0o700)
    state = {
        "profile": PROFILE,
        "role": ROLE,
        "split": SPLIT,
        "labels": None,
        "evaluation_only": True,
        "frozen_candidate_ref": plan["frozen_candidate_ref"],
        "numeric_root_acceptance_ref": plan["numeric_root_acceptance_ref"],
        "status": PENDING,
        "training_eligible": False,
        "completed": False,
        "plan_sha256": expected_sha,
        "model_loads": 0,
        "native_request_calls": 0,
        "calls": [],
        "process": identity,
        "owned_limits": limits,
        "host_startup_admission": gate,
        "no_escape_audited": False,
        "whole_tree_audited": False,
        "model_qualified": False,
    }
    chunk = {
        "schema": SCHEMA,
        "profile": PROFILE,
        "status": PENDING,
        "training_eligible": False,
        "completed": False,
        "dsl_version": DSL_VERSION,
        "split": SPLIT,
        "role": ROLE,
        "family_id": selected[0]["family_id"],
        "ancestor_id": plan["ancestor_id"],
        "dev_inputs": plan["dev_inputs"],
        "source_records": plan["dependencies"],
        "collector_source": plan["collector_source"],
        "selected_plan_sha256": expected_sha,
        "root_np5_prerequisite_ref": plan["np5_root_acceptance"],
        "dev_capability_ref": plan["dev_capability"],
        "frozen_candidate_ref": plan["frozen_candidate_ref"],
        "frozen_train_acceptance_ref": plan["frozen_train_acceptance_ref"],
        "numeric_root_acceptance_ref": plan["numeric_root_acceptance_ref"],
        "labels": None,
        "evaluation_only": True,
        "sealed_reveal_authorized": False,
        "train_COMMITTED_authority": False,
        "owner": {
            "unit": UNIT,
            "profile": OWNER_PROFILE,
            "process": identity,
            "owned_limits": limits,
        },
        "model": base["model"],
        "native_plan_sha256": plan["dependencies"]["native_plan"]["sha256"],
        "cases": [],
        "whole_seconds": 120,
        "call_seconds": 20,
        "external_root_acceptance_required": True,
    }

    def demote(error):
        state["completed"] = chunk["completed"] = False
        state["error"] = chunk["error"] = {"type": type(error).__name__, "message": str(error)}

    def finalize(primary):
        chunk["elapsed_before_last_manifest_write_seconds"] = time.monotonic() - began
        try:
            helper.write_json(output / "chunk.json", chunk)
        except BaseException as secondary:
            if primary is None:
                raise
            primary.add_note("Pending chunk manifest write also failed: " + repr(secondary))
        helper.finalize_state(output, state, primary)
        sync_directories((output, output.parent), primary)

    def operation():
        helper.write_json(output / "state.json", state)
        runtime, torch = native.load_runtime(helper, base, output, state)
        # First call20 includes closed-catalog authentication and imports.
        shared, baseline = {}, {}
        for index, item in enumerate(selected, 1):
            chunk["cases"].append(
                retain_call(
                    np5,
                    native,
                    helper,
                    observer_module,
                    probe,
                    shared,
                    plan["dependencies"]["head_core"],
                    runtime,
                    torch,
                    base,
                    item,
                    index,
                    output,
                    state,
                    baseline,
                    accepted_catalog,
                )
            )
        require(
            state["model_loads"] == 1
            and state["native_request_calls"] == len(chunk["cases"]) == 8
            and all(row["completed"] for row in state["calls"]),
            "One constructor and eight complete original calls required",
        )
        native.authenticate(helper, base)
        native.validate_runtime(runtime, torch, base["model"])
        require(
            validate_plan(plan, plan_path, expected_sha) == (accepted_catalog, dev_capability),
            "Closed NP5/frozen candidate/dev prerequisites changed during chunk",
        )
        select_cases(
            verify_record(plan["dev_inputs"]["requests"]),
            verify_record(plan["dev_inputs"]["summary"]),
            plan["ancestor_id"],
            dev_capability,
        )
        chunk["baseline"] = {
            **file_record(output / "baseline.npy", output),
            "format": "npy_no_pickle",
            "dtype": "float32",
            "shape": [255, 1024],
            "FP32_bytes_sha256": baseline["bytes_sha256"],
        }
        chunk["final_full_model_runtime_source_input_authentication"] = True
        chunk["after_final_auth_owned_counters"] = native.own_counters(identity)
        chunk["completed"] = state["completed"] = True
        return chunk

    # NP5's original transaction includes evidence, cleanup and watchdog exit,
    # then a literal final120s check. Any later root decision stays external.
    result, _elapsed = np5.transaction(
        native,
        helper,
        output,
        "whole-dev-chunk",
        began,
        WALL_SECONDS,
        operation,
        lambda: None,
        finalize,
        demote,
    )
    return result


def main(argv=None):
    release_gate()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    execute(args.plan, args.plan_sha256, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
