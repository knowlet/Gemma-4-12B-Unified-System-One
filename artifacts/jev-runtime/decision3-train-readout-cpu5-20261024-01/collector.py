"""Source-qualified DSL01 train-only collector; no optimizer or acceptance API.

Execution requires an exact selected plan, closed NP02 root prerequisite, fixed
train-only capability and the original CPU5 owner admission. Outputs stay PENDING.
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
PROFILE = "decision3-dsl01-train-features-cpu5-4g-source-candidate-v1"
SCHEMA = "decision3-dsl01-train-feature-chunk-v1"
DSL_VERSION = "independent-browser-workflow-dsl/1"
DSL = ROOT / "artifacts/jev-runtime/browser-decision-dsl-20261023-01/corpus/data/train"
NATIVE = ROOT / "artifacts/jev-runtime/decision3-publisher-batch4-cpu-20261024-05"
NP5 = ROOT / "artifacts/jev-runtime/decision3-readout-numpy-cpu5-qualification-20261024-02"
CORE = ROOT / "artifacts/jev-runtime/decision3-cached-head-core-20261023-01/head_core.py"
OWNER = ROOT / "artifacts/jev-runtime/llama-cpp-owner-20261024-09"
UNIT = "jev-d3-llama-cpp-cpu5-fast-4g-coexist-20261024-09.service"
OWNER_PROFILE = "llama-cpp-cpu5-fast-same-cluster-4g-coexist-command-owned-v1"
TRAIN_FAMILIES = frozenset(
    {"appointment_edit", "document_upload", "booking_inventory", "admin_filters"}
)
HEADS = frozenset({"operation", "click_target", "type_text_target", "select_target"})
WALL_SECONDS, CALL_SECONDS = 120, 20
PENDING = "PENDING_TRAIN_ONLY"
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
        "Collector authoring/dev/sealed/labels input refused before open",
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


def validate_plan(plan, path, expected_sha):
    require(
        plan.get("schema") == "decision3-dsl01-train-collector-plan-v1"
        and plan.get("profile") == PROFILE
        and plan.get("status") == "selected_after_separate_review",
        "Pending/other collector plan refused",
    )
    require(
        plan.get("dsl_version") == DSL_VERSION
        and plan.get("split") == "train"
        and plan.get("requests") == 8
        and plan.get("wall_seconds") == 120
        and plan.get("call_seconds") == 20,
        "Fixed train chunk and original budgets required",
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
        set(plan["train_inputs"]) == {"requests", "summary"},
        "Collector cannot select labels or non-train artifacts",
    )
    verify_record(plan["train_inputs"]["requests"], DSL / "requests.jsonl")
    verify_record(plan["train_inputs"]["summary"], DSL / "summary.json")
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
    capability = decode(verify_record(plan["train_capability"]))
    require(
        capability.get("schema") == "root-selected-dsl01-train-collector-capability-v1"
        and capability.get("role") == "train_collector"
        and capability.get("dsl_version") == DSL_VERSION
        and capability.get("split") == "train"
        and capability.get("train_inputs") == plan["train_inputs"],
        "Root train-only export capability required",
    )
    require(
        capability.get("family_ids") == sorted(TRAIN_FAMILIES)
        and capability.get("ancestor_ids")
        == sorted(f"{family}/workflow-{i}" for family in TRAIN_FAMILIES for i in (1, 2, 3)),
        "Exact existing train ancestry capability required",
    )
    require(
        plan.get("ancestor_id") in capability["ancestor_ids"], "Selected ancestor is not train-only"
    )
    return catalog["after"]


def select_cases(request_bytes, summary_bytes, ancestor):
    """Pure train-export validation; never import reconstructive authoring code."""
    summary = decode(summary_bytes)
    ancestors = sorted(f"{family}/workflow-{i}" for family in TRAIN_FAMILIES for i in (1, 2, 3))
    require(
        summary
        == {
            "dsl_version": DSL_VERSION,
            "split": "train",
            "case_count": 96,
            "family_ids": sorted(TRAIN_FAMILIES),
            "ancestor_ids": ancestors,
            "is_authored_synthetic_not_NN": True,
        },
        "Exact DSL01 train summary required",
    )
    require(ancestor in ancestors, "Unknown/non-train ancestor refused")
    rows = [decode(line) for line in request_bytes.splitlines() if line.strip()]
    require(
        len(rows) == 96 and len({row["case_id"] for row in rows}) == 96,
        "Exact unique train request population required",
    )
    selected = []
    counts = {key: 0 for key in ancestors}
    for row in rows:
        require(
            set(row) == IDENTITY_KEYS | {"request"}
            and row["dsl_version"] == DSL_VERSION
            and row["family_id"] in TRAIN_FAMILIES
            and row["ancestor_id"] in ancestors
            and row["ancestor_id"].split("/")[0] == row["family_id"]
            and re.fullmatch(r"[0-9a-f]{24}", row["case_id"]),
            "Malformed/non-train request lineage",
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
        "All eight existing branches per train ancestor required",
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
        core = source("_train_collector_head_core", core_record)
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
        "completed": False,
        "status": PENDING,
        "training_eligible": False,
    }
    state["calls"].append(row)
    evidence = {
        "schema": "decision3-dsl01-train-case-evidence-v1",
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
            "W0 changed between same-load train calls",
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
        f"train-call-{index:04d}",
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
    accepted_catalog = validate_plan(plan, plan_path, expected_sha)
    selected = select_cases(
        verify_record(plan["train_inputs"]["requests"]),
        verify_record(plan["train_inputs"]["summary"]),
        plan["ancestor_id"],
    )
    native = source("_train_collector_native05", plan["dependencies"]["native_actor"])
    helper = native.load_helper()
    base = native.load_plan(
        helper, NATIVE / "edge-plan.json", plan["dependencies"]["native_plan"]["sha256"]
    )
    np5 = source("_train_collector_np5", plan["dependencies"]["np5_actor"])
    observer_module = source("_train_collector_observer", plan["dependencies"]["observer"])
    probe = source("_train_collector_numpy_probe", plan["dependencies"]["numpy_probe"])
    identity = helper.process_identity()
    limits = helper.owned_limits(identity)
    native.validate_owner(identity, limits, helper.START_ENV)
    gate = helper.host_gate()
    output = ordinary(output)
    require(
        output.name == "train-chunk"
        and output.parent.parent == OWNER
        and re.fullmatch(r"actual-[0-9]{2}", output.parent.name)
        and not output.exists(),
        "Fresh current original-owner train output only",
    )
    output.mkdir(mode=0o700)
    (output / ".gitignore").write_text("*\n", encoding="utf-8")
    (output / "cases").mkdir(mode=0o700)
    state = {
        "profile": PROFILE,
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
        "split": "train",
        "family_id": selected[0]["family_id"],
        "ancestor_id": plan["ancestor_id"],
        "train_inputs": plan["train_inputs"],
        "source_records": plan["dependencies"],
        "collector_source": plan["collector_source"],
        "selected_plan_sha256": expected_sha,
        "root_np5_prerequisite_ref": plan["np5_root_acceptance"],
        "train_capability_ref": plan["train_capability"],
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
            validate_plan(plan, plan_path, expected_sha) == accepted_catalog,
            "Closed NP5 prerequisite changed during chunk",
        )
        select_cases(
            verify_record(plan["train_inputs"]["requests"]),
            verify_record(plan["train_inputs"]["summary"]),
            plan["ancestor_id"],
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
        "whole-train-chunk",
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
