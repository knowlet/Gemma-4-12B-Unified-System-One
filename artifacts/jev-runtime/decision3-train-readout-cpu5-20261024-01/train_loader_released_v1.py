"""HELD DSL01 train-only COMMITTED-feature loader; standard library only.

This module never imports authoring code, a corpus manifest, a model or NumPy.
The selected caller supplies the authenticated NumPy/core modules after its hold
and current original-owner admission checks.  A root-selected acceptance hash,
not a collector's Boolean or filename, is the data acceptance authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from pathlib import Path

SOURCE_ONLY = False
ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
HERE = ROOT / "artifacts/jev-runtime/decision3-train-readout-cpu5-20261024-01"
DSL_ROOT = ROOT / "artifacts/jev-runtime/browser-decision-dsl-20261023-01"
TRAIN = DSL_ROOT / "corpus/data/train"
DSL_VERSION = "independent-browser-workflow-dsl/1"
FAMILIES = ("appointment_edit", "document_upload", "booking_inventory", "admin_filters")
ANCESTORS = tuple(family + "/workflow-" + str(i) for family in FAMILIES for i in (1, 2, 3))
IDENTITY = (
    "case_id",
    "dsl_version",
    "family_id",
    "ancestor_id",
    "request_sha256",
    "native_payload_sha256",
)
OWNER = ROOT / "artifacts/jev-runtime/llama-cpp-owner-20261024-09"
UNIT = "jev-d3-llama-cpp-cpu5-fast-4g-coexist-20261024-09.service"
OWNER_PROFILE = "llama-cpp-cpu5-fast-same-cluster-4g-coexist-command-owned-v1"
CORE = ROOT / "artifacts/jev-runtime/decision3-cached-head-core-20261023-01/head_core.py"
CORE_SHA = "6761dd654481424f4d6c908a7c9fd2d43e4762eeb5420af9f22612ffe77657ed"
HYPERPARAMETERS = {
    "steps": 32,
    "learning_rate": 0.001,
    "l2": 0.0001,
    "seed": 20261023,
    "batch_size": 64,
}
PROJECTION = {
    "id": "publisher-final-norm-last-position-fp32-255x1024-t1-v1",
    "feature_width": 1024,
    "codes": 255,
    "temperature": 1.0,
    "absolute_tolerance": 1e-6,
    "relative_tolerance": 1e-5,
    "legal_mask": "code_index_less_than_actual_criteria_count",
}
NP_CHECKS = frozenset(
    {
        "root_exit0",
        "owner_completed",
        "sources_unchanged",
        "current_closure_positive",
        "full_authentication",
        "all_legal_probabilities_matched",
        "all_ordered_argmaxes_matched",
        "FP32_copy_exact",
    }
)


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    # The existing DSL exports preserve insertion order; do not sort keys.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def decode(raw):
    return json.loads(
        raw, object_pairs_hook=pairs, parse_constant=lambda _: require(False, "nonfinite JSON")
    )


def ordinary(path):
    path = Path(path)
    require(
        path.is_absolute() and ".." not in path.parts and path.is_relative_to(ROOT),
        "absolute repository path required",
    )
    # The only authoring-directory files this loader may read are these three
    # exact train exports.  No full manifest/generator/dev/sealed fallback.
    require(
        not path.is_relative_to(DSL_ROOT)
        or path in {TRAIN / name for name in ("requests.jsonl", "labels.jsonl", "summary.json")},
        "authoring/dev/sealed input refused",
    )
    for parent in (path, *path.parents):
        require(not parent.is_symlink(), "symlink input refused")
    return path


def normalized(ref, parent=None):
    require(
        type(ref) is dict and {"path", "sha256", "size_bytes"} <= set(ref),
        "explicit file pin required",
    )
    path = Path(ref["path"])
    if not path.is_absolute():
        require(
            parent is not None and ".." not in path.parts,
            "relative payload needs its selected chunk root",
        )
        path = parent / path
        require(path.is_relative_to(parent), "payload escaped chunk root")
    return {**ref, "path": str(ordinary(path))}


def read_ref(ref, check_deadline, *, parent=None, maximum=2 * 1024**2):
    ref = normalized(ref, parent)
    require(
        type(ref["size_bytes"]) is int and 0 < ref["size_bytes"] <= maximum,
        "bounded exact file size required",
    )
    require(
        type(ref["sha256"]) is str
        and len(ref["sha256"]) == 64
        and all(c in "0123456789abcdef" for c in ref["sha256"]),
        "SHA256 pin required",
    )
    path = Path(ref["path"])
    check_deadline()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(before.st_mode) and before.st_size == ref["size_bytes"],
            "ordinary exact-size input required",
        )
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(
        len(raw) == ref["size_bytes"]
        and all(getattr(before, k) == getattr(after, k) == getattr(path.stat(), k) for k in fields),
        "input changed during read",
    )
    require(hashlib.sha256(raw).hexdigest() == ref["sha256"], "input SHA differs")
    check_deadline()
    return raw


def same_identity(left, right):
    return {k: left[k] for k in IDENTITY} == {k: right[k] for k in IDENTITY}


def validate_train_inventory(requests, labels, summary):
    """Pure metadata/content checks; no authoring import and no file access."""
    require(
        type(summary) is dict
        and summary.get("dsl_version") == DSL_VERSION
        and summary.get("split") == "train",
        "DSL01 train summary required",
    )
    require(
        type(summary.get("case_count")) is int
        and summary["case_count"] == 96
        and summary.get("is_authored_synthetic_not_NN") is True,
        "96 authored train states required",
    )
    require(
        summary.get("family_ids") == sorted(FAMILIES)
        and summary.get("ancestor_ids") == sorted(ANCESTORS),
        "exact train families/ancestors required",
    )
    require(len(requests) == len(labels) == 96, "96 train requests and labels required")
    require(len({row["case_id"] for row in requests}) == 96, "duplicate train case")
    require(
        [row["case_id"] for row in labels] == [row["case_id"] for row in requests],
        "label/request case order differs",
    )
    grouped = {ancestor: [] for ancestor in ANCESTORS}
    for request, label in zip(requests, labels):
        require(
            set(request) == {*IDENTITY, "request"} and set(label) == {*IDENTITY, "heads"},
            "exact train export row schema required",
        )
        require(
            same_identity(request, label) and request["dsl_version"] == DSL_VERSION,
            "request/label identity differs",
        )
        family, ancestor = request["family_id"], request["ancestor_id"]
        require(
            family in FAMILIES and ancestor in grouped and ancestor.split("/")[0] == family,
            "non-train family/ancestor refused",
        )
        case = request["case_id"]
        require(
            type(case) is str
            and 0 < len(case) <= 128
            and "/" not in case
            and "\\" not in case
            and ".." not in case,
            "case id is not a payload path",
        )
        payload = request["request"]
        require(
            digest(payload) == request["request_sha256"]
            and digest({key: payload[key] for key in ("state", "questions")})
            == request["native_payload_sha256"],
            "request/native projection digest differs",
        )
        questions = payload["questions"]
        require(
            type(questions) is dict
            and set(questions)
            == {"operation", "click_target", "type_text_target", "select_target"},
            "all four actual original heads required",
        )
        require(
            type(label["heads"]) is dict
            and "operation" in label["heads"]
            and set(label["heads"]) <= set(questions),
            "invalid supervised head selection",
        )
        for head, question in questions.items():
            criteria = question.get("criteria")
            require(
                question.get("type") == "choice"
                and type(criteria) is dict
                and 1 <= len(criteria) <= 255
                and all(type(k) is str for k in criteria),
                "complete ordered Choice criteria required",
            )
            if head in label["heads"]:
                target_vector(label["heads"][head], list(criteria))
        grouped[ancestor].append(request)
    require(
        all(len(rows) == 8 for rows in grouped.values()), "eight states per train ancestor required"
    )
    return grouped


def target_vector(label, criteria):
    require(
        type(label) is dict
        and set(label) == {"criteria_ids", "valid_ids", "valid_positions", "soft_target"},
        "exact label schema required",
    )
    require(label["criteria_ids"] == criteria, "label criteria order differs")
    valid = label["valid_ids"]
    require(
        type(valid) is list
        and bool(valid)
        and len(set(valid)) == len(valid)
        and set(valid) <= set(criteria),
        "empty or invalid target set",
    )
    positions = [i for i, key in enumerate(criteria) if key in valid]
    require(
        valid == [criteria[i] for i in positions]
        and label["valid_positions"] == positions
        and all(type(i) is int for i in label["valid_positions"]),
        "valid target positions differ",
    )
    soft = label["soft_target"]
    expected = [1.0 / len(valid) if key in valid else 0.0 for key in criteria]
    require(
        type(soft) is list
        and len(soft) == len(criteria)
        and all(
            type(v) in (int, float) and math.isfinite(v) and abs(v - e) <= 1e-12
            for v, e in zip(soft, expected)
        ),
        "authored soft target differs",
    )
    # Never manufacture supervision for unused target heads or accept an
    # all-zero row.  The 255-code padding is outside the valid criteria only.
    return soft + [0.0] * (255 - len(criteria))


def validate_commit_header(acceptance, entry, selection):
    require(
        acceptance.get("schema") == "root-closed-dsl01-train-chunk-acceptance-v1"
        and acceptance.get("status") == "COMMITTED"
        and acceptance.get("training_eligible") is True,
        "genuine separate COMMITTED acceptance required",
    )
    require(
        acceptance.get("split") == "train"
        and acceptance.get("dsl_version") == DSL_VERSION
        and acceptance.get("ancestor_id") == entry["ancestor_id"]
        and acceptance.get("family_id") == entry["family_id"],
        "acceptance split/ancestor differs",
    )
    for key in (
        "train_inputs",
        "model",
        "projection",
        "source_records",
        "collector_source",
        "root_np5_prerequisite_ref",
    ):
        require(
            acceptance[key] == selection[key], "root-selected acceptance identity differs: " + key
        )
    require(
        acceptance["producer_chunk"] == entry["chunk"]
        and acceptance["baseline_sha256"] == selection["baseline"]["sha256"],
        "accepted chunk/baseline pin differs",
    )
    transaction = acceptance["transaction"]
    require(
        all(
            type(transaction[key]) is int and transaction[key] == expected
            for key, expected in (
                ("case_count", 8),
                ("call_budget_seconds", 20),
                ("whole_budget_seconds", 120),
            )
        ),
        "original collector transaction differs",
    )
    times = transaction["complete_call_seconds"]
    require(
        type(times) is list
        and len(times) == 8
        and all(type(t) in (int, float) and math.isfinite(t) and 0 < t <= 20 for t in times),
        "eight complete call20 proofs required",
    )
    total = transaction["whole_seconds"]
    require(
        type(total) in (int, float) and math.isfinite(total) and max(times) <= total <= 120,
        "whole120 proof required",
    )


def validate_original_closure(refs, check_deadline):
    require(
        set(refs) == {"root_execution", "owner_receipt", "closure", "admission", "sealed_config"},
        "exact original proof refs required",
    )
    raw = {key: read_ref(ref, check_deadline) for key, ref in refs.items()}
    proof = {key: decode(value) for key, value in raw.items()}
    terminal, receipt, closure = (
        proof[key] for key in ("root_execution", "owner_receipt", "closure")
    )
    config, admission = proof["sealed_config"], proof["admission"]
    require(
        type(terminal["exit_code"]) is int
        and terminal["exit_code"] == 0
        and receipt["profile"] == OWNER_PROFILE
        and receipt["completed"] is True
        and receipt["primary_error"] is None
        and receipt["secondary_errors"] == []
        and receipt["sources_unchanged"] is True,
        "successful original root/owner required",
    )
    require(
        receipt["selected_config"] == config
        and config["profile"] == OWNER_PROFILE
        and config["unit"] == UNIT
        and Path(config["output"]).parent == OWNER,
        "current selected owner09 output required",
    )
    require(
        receipt["closure"] == closure, "owner receipt and separately pinned original closure differ"
    )
    require(
        admission["config_sha256"] == hashlib.sha256(raw["sealed_config"]).hexdigest()
        and admission["nonce"] == config["nonce"] == closure["selection"]["nonce"]
        and admission["invocation_id"] == closure["invocation_id"]
        and closure["selection"]["unit"] == UNIT,
        "current original nonce/invocation differs",
    )
    require(
        all(
            admission["actor"][key] == closure["actor"][key]
            for key in ("pid", "start_ticks", "pgid", "sid", "uid", "cgroup")
        ),
        "original admitted wrapper differs",
    )
    require(
        closure["closure_mode"] == "authenticated_stop"
        and closure["closure_errors"] == []
        and all(
            closure[key] is True
            for key in (
                "whole_unit_exit_confirmed",
                "observed_cgroup_cohort_exit_confirmed",
                "post_stop_terminal_verified",
                "cgroup_absent",
                "sources_unchanged",
            )
        ),
        "original authenticated scoped closure required",
    )
    require(
        len(closure["two_process_scans"]) == 2
        and all(not row["matches"] and not row["unknown"] for row in closure["two_process_scans"]),
        "two original empty cohort scans required",
    )
    require(
        all(
            closure[key] is False
            for key in (
                "no_escape_enforced",
                "whole_tree_exit_confirmed",
                "continuation_authorized",
                "model_qualified",
            )
        ),
        "universal qualification must remain false",
    )
    argv = terminal["argv"]
    require(
        argv[3] == str(OWNER / "controller.py")
        and argv[4] == "--run"
        and argv[argv.index("--selection-sha256") + 1] == config["selection_sha256"]
        and argv[argv.index("--output") + 1] == config["output"],
        "root command/current config join differs",
    )
    require(
        Path(refs["owner_receipt"]["path"]).parent == Path(config["output"])
        and all(
            Path(refs[key]["path"]).parent == Path(config["output"])
            for key in ("closure", "admission", "sealed_config")
        ),
        "proof refs belong to different invocations",
    )
    return proof


def validate_np_header(prerequisite, selection):
    """A pinned pending/failed NP record is not a closed runtime acceptance."""
    require(
        prerequisite.get("schema") == "root-accepted-np5-qualification-v1"
        and prerequisite.get("status") == "accepted"
        and prerequisite.get("independent_actual_review_claimed") is False,
        "accepted honest-root NP5 prerequisite required",
    )
    checks = prerequisite.get("checks", {})
    require(
        type(checks) is dict
        and set(checks) == NP_CHECKS
        and all(checks[key] is True for key in NP_CHECKS),
        "complete true NP5 numeric/closure checks required",
    )
    require(
        prerequisite.get("source_refs") == selection["source_records"]
        and prerequisite.get("owner_unit") == UNIT
        and prerequisite.get("owner_profile") == OWNER_PROFILE,
        "NP5 source/native/owner prerequisite differs",
    )
    require(
        set(prerequisite.get("proof_refs", {}))
        == {"root_execution", "owner_receipt", "current_closure"},
        "exact original NP5 proof refs required",
    )


def selected_sources(config, records):
    rows = config["sources"]
    selected = {str(ROOT / row["path"]): row for row in rows}
    require(len(selected) == len(rows), "duplicate original config source path")
    for ref in records:
        row = selected.get(ref["path"])
        require(
            row is not None
            and row["sha256"] == ref["sha256"]
            and row["size_bytes"] == ref["size_bytes"],
            "producer source is not selected by its original owner config",
        )
    return selected


def validate_np_acceptance(prerequisite, selection, check_deadline):
    # This runs before any NumPy import and is repeated by the train loader.
    validate_np_header(prerequisite, selection)
    refs = prerequisite["proof_refs"]
    proof = {key: decode(read_ref(ref, check_deadline)) for key, ref in refs.items()}
    terminal, receipt, closure = (
        proof[key] for key in ("root_execution", "owner_receipt", "current_closure")
    )
    config = receipt["selected_config"]
    require(
        type(terminal["exit_code"]) is int
        and terminal["exit_code"] == 0
        and receipt["profile"] == OWNER_PROFILE
        and receipt["completed"] is True
        and receipt["primary_error"] is None
        and receipt["secondary_errors"] == []
        and receipt["sources_unchanged"] is True,
        "original successful NP5 terminal/owner required",
    )
    require(
        config["profile"] == OWNER_PROFILE
        and config["unit"] == UNIT
        and Path(config["output"]).parent == OWNER
        and receipt["closure"] == closure,
        "NP5 owner/closure belongs to another selection",
    )
    require(
        closure["selection"]["unit"] == UNIT
        and closure["selection"]["nonce"] == config["nonce"]
        and closure["selection"]["boot_id"] == config["boot_id"]
        and closure["closure_mode"] == "authenticated_stop"
        and closure["closure_errors"] == [],
        "original closed NP5 selection differs",
    )
    require(
        all(
            closure[key] is True
            for key in (
                "whole_unit_exit_confirmed",
                "observed_cgroup_cohort_exit_confirmed",
                "post_stop_terminal_verified",
                "cgroup_absent",
                "sources_unchanged",
            )
        )
        and len(closure["two_process_scans"]) == 2
        and all(not row["matches"] and not row["unknown"] for row in closure["two_process_scans"]),
        "original NP5 current positive closure required",
    )
    require(
        all(
            closure[key] is False
            for key in (
                "no_escape_enforced",
                "whole_tree_exit_confirmed",
                "continuation_authorized",
                "model_qualified",
            )
        ),
        "NP5 universal qualification must remain false",
    )
    argv = terminal["argv"]
    require(
        argv[3] == str(OWNER / "controller.py")
        and argv[4] == "--run"
        and argv[argv.index("--selection-sha256") + 1] == config["selection_sha256"]
        and argv[argv.index("--output") + 1] == config["output"],
        "NP5 root command/current config join differs",
    )
    require(
        all(
            Path(refs[key]["path"]).parent == Path(config["output"])
            for key in ("owner_receipt", "current_closure")
        ),
        "NP5 proof refs belong to different invocations",
    )
    # Transitive native/runtime identities are authenticated by the pinned
    # accepted source projection; its executed actor must also be an outer
    # owner-selected source. Do not require redundant outer pins for every DSO.
    selected_sources(config, [selection["source_records"]["np5_actor"]])
    return normalized(prerequisite["numpy_runtime_catalog"], ROOT)


def validate_producer_binding(chunk, parent, proof, selection, check_deadline):
    """Join immutable features to the actual successful collector child."""
    config, receipt = proof["sealed_config"], proof["owner_receipt"]
    require(
        parent == Path(config["output"]) / "train-chunk"
        and chunk["owner"]["unit"] == UNIT
        and chunk["owner"]["profile"] == OWNER_PROFILE,
        "chunk is not from this original owner output",
    )
    actor = receipt["state"]["actor_exit"]
    require(
        actor["config_sha256"] == proof["admission"]["config_sha256"]
        and actor["entry_exit_code"] == 0
        and actor["error"] is None
        and actor["planned_command_count"] == config["command_count"] == 1
        and len(actor["command_results"]) == 1,
        "one completed original collector command required",
    )
    command = actor["command_results"][0]
    require(
        type(command["returncode"]) is int
        and command["returncode"] == 0
        and command["observed_identity"] is not None,
        "actual successful collector child observation required",
    )
    observed, process = command["observed_identity"], chunk["owner"]["process"]
    child = observed["process"]
    require(
        type(process["pid"]) is int
        and type(process["starttime_ticks"]) is int
        and process["pid"] == command["pid"] == child["pid"]
        and process["starttime_ticks"] == child["start_ticks"]
        and process["pgid"] == child["pgid"]
        and process["sid"] == child["sid"]
        and process["cgroup"] == "0::" + child["cgroup"]
        and process["executable"] == command["resolved_executable"] == observed["exe"],
        "chunk producer PID/start/cohort/executable differs from original child",
    )
    require(
        all(
            child[key] == proof["admission"]["actor"][key]
            for key in ("pgid", "sid", "uid", "cgroup")
        ),
        "collector child is outside original admitted cohort",
    )
    argv = command["argv"]
    require(
        type(argv) is list
        and len(argv) == 10
        and argv[:4]
        == [
            "/home/albert_hsieh/ComfyUI/.venv/bin/python",
            "-B",
            "-s",
            selection["collector_source"]["path"],
        ]
        and argv[4] == "--plan"
        and argv[6] == "--plan-sha256"
        and argv[7] == chunk["selected_plan_sha256"]
        and argv[8:] == ["--output", str(parent)]
        and observed["argv"] == argv
        and command["cwd"] == observed["cwd"] == str(ROOT),
        "original actual collector argv/plan/output differs",
    )
    selected = selected_sources(
        config, [*selection["source_records"].values(), selection["collector_source"]]
    )
    plan_source = selected.get(argv[5])
    require(
        plan_source is not None and plan_source["sha256"] == chunk["selected_plan_sha256"],
        "collector plan was not selected by original config sources",
    )
    plan_ref = {
        "path": argv[5],
        "sha256": plan_source["sha256"],
        "size_bytes": plan_source["size_bytes"],
    }
    plan = decode(read_ref(plan_ref, check_deadline))
    require(
        plan["schema"] == "decision3-dsl01-train-collector-plan-v1"
        and plan["profile"] == chunk["profile"]
        and plan["status"] == "selected_after_separate_review"
        and plan["ancestor_id"] == chunk["ancestor_id"]
        and plan["dependencies"] == selection["source_records"]
        and plan["collector_source"] == selection["collector_source"]
        and plan["train_inputs"] == chunk["train_inputs"]
        and plan["np5_root_acceptance"] == selection["root_np5_prerequisite_ref"],
        "actual selected collector plan does not bind this chunk",
    )
    return plan_ref


def load_array(np, ref, parent, shape, check_deadline):
    require(
        ref.get("format") == "npy_no_pickle"
        and ref.get("dtype") == "float32"
        and ref.get("shape") == list(shape),
        "exact FP32 array declaration required",
    )
    import io

    raw = read_ref(ref, check_deadline, parent=parent)
    array = np.load(io.BytesIO(raw), allow_pickle=False)
    require(
        type(array) is np.ndarray
        and array.dtype == np.float32
        and array.shape == shape
        and array.flags.c_contiguous
        and bool(np.isfinite(array).all()),
        "actual finite C-order FP32 array differs",
    )
    if "FP32_bytes_sha256" in ref:
        require(
            hashlib.sha256(array.tobytes(order="C")).hexdigest() == ref["FP32_bytes_sha256"],
            "raw FP32 bytes differ from NPY file identity",
        )
    array.flags.writeable = False
    check_deadline()
    return array


def compare_native(np, core, features, baseline, request, raw):
    questions = request["questions"]
    require(
        raw["model"] == "d3-edge" and list(raw["answers"]) == list(questions),
        "same-call raw model/head order differs",
    )
    counts = np.asarray([len(q["criteria"]) for q in questions.values()], dtype=np.int64)
    zero = np.zeros((255, 1024), dtype=np.float32)
    replay = core.probabilities(features, baseline, zero, counts)
    require(
        replay.dtype == np.float32 and replay.shape == (4, 255) and bool(np.isfinite(replay).all()),
        "zero-delta projection differs",
    )
    for i, (head, question) in enumerate(questions.items()):
        criteria, answer = list(question["criteria"]), raw["answers"][head]
        probabilities = answer["probabilities"]
        require(
            answer["type"] == "choice"
            and list(probabilities) == criteria
            and bool((replay[i, len(criteria) :] == 0).all()),
            "raw criteria order/illegal mass differs",
        )
        values = list(probabilities.values())
        require(
            all(type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1 for p in values)
            and abs(sum(values) - 1) <= 1e-6,
            "invalid complete native probabilities",
        )
        require(
            all(abs(float(replay[i, j]) - p) <= 1e-6 + 1e-5 * abs(p) for j, p in enumerate(values)),
            "selected feature/W0 does not reproduce same-call raw",
        )
        require(
            answer["choice"]
            == max(probabilities, key=probabilities.get)
            == criteria[int(np.argmax(replay[i, : len(criteria)]))],
            "ordered argmax differs",
        )
        entropy = -sum(p * math.log(p) for p in values if p > 0)
        confidence = (
            1.0 if len(values) == 1 else max(0.0, min(1.0, 1 - entropy / math.log(len(values))))
        )
        require(
            type(answer["confidence"]) in (int, float)
            and math.isfinite(answer["confidence"])
            and abs(answer["confidence"] - confidence) <= 1e-6,
            "native singleton/entropy confidence differs",
        )


def validate_case_evidence(evidence, case, original, index):
    identity = evidence["identity"]
    require(
        identity
        == {
            "case_id": case["case_id"],
            "request_sha256": original["request_sha256"],
            "native_payload_sha256": original["native_payload_sha256"],
            "same_call_id": "call-" + str(index).zfill(4),
            "native_call_index": index,
        },
        "same-call evidence identity differs",
    )
    require(
        evidence["native_payload"]
        == {key: original["request"][key] for key in ("state", "questions")}
        and digest(evidence["native_payload"]) == original["native_payload_sha256"],
        "same-call original native payload differs",
    )
    proof = evidence["math"]
    require(
        proof["feature_shape"] == [4, 1024]
        and proof["baseline_shape"] == [255, 1024]
        and type(proof["temperature"]) in (int, float)
        and proof["temperature"] == 1.0,
        "original numerical projection differs",
    )
    for key in ("first_layer_events", "final_norm_events"):
        events = proof[key]
        require(
            type(events) is list
            and len(events) == 1
            and type(events[0]) is list
            and len(events[0]) == 3
            and all(type(n) is int for n in events[0])
            and events[0][0] == 4
            and events[0][1] > 0
            and events[0][2] == 1024,
            "one complete original forward event required",
        )
    require(
        proof["first_layer_events"] == proof["final_norm_events"],
        "original first/final forward shape differs",
    )
    require(
        all(
            proof[key] is True
            for key in (
                "FP32_copy_exact",
                "baseline_tensor_identity_and_versions_unchanged",
                "illegal_probabilities_exact_zero",
            )
        ),
        "collector mathematical evidence incomplete",
    )
    timing = evidence["timing"]
    elapsed = timing["elapsed_before_last_evidence_write_seconds"]
    require(
        type(timing["budget_seconds"]) is int
        and timing["budget_seconds"] == 20
        and type(elapsed) in (int, float)
        and math.isfinite(elapsed)
        and 0 <= elapsed <= case["complete_prediction_seconds"] <= 20
        and timing["hooks_removed"] is True
        and timing["post_watchdog_check_required"] is True,
        "complete call20 evidence differs",
    )
    # These producer diagnostics are necessary but never sufficient: independent
    # acceptance pins them, and compare_native recomputes actual array parity.
    require(
        evidence["native_full_authentication_unchanged"] is True,
        "original per-call full authentication required",
    )


def load_training(selection, np, core, *, check_deadline):
    require(SOURCE_ONLY is False, "SOURCE_ONLY: train loader is not released")
    require(callable(check_deadline), "whole owned-fit deadline required")
    require(
        selection["schema"] == "root-selected-dsl01-committed-train-readout-v1"
        and selection["status"] == "ROOT_SELECTED_COMMITTED_TRAIN_ONLY"
        and selection["split"] == "train",
        "root-selected train-only capability required",
    )
    require(
        selection["projection"] == PROJECTION
        and selection["hyperparameters"] == HYPERPARAMETERS
        and selection["fit_budget_seconds"] == 120,
        "predeclared numerical/fit policy differs",
    )
    model = selection["model"]
    require(
        model["model_id"] == "vllm-sr/d3-edge"
        and model["revision"] == "cab71e6ad3461b8086d51ddb572c656babd63b0b"
        and model["model_sha256"]
        == "6e00d04abbdbdf6d35c9e12065769d9edaacc3bfe9404583891797718d239dc8"
        and model["manifest_sha256"]
        == "3bae8989a6feb36a66bf4d8c44112dc63199d0dc69f564dc895fe368b1e08e59",
        "fixed model identity differs",
    )
    pinned = []
    inputs = selection["train_inputs"]
    require(set(inputs) == {"requests", "labels", "summary"}, "only three train exports selected")
    for key in inputs:
        require(
            inputs[key]["path"] == str(TRAIN / (key + (".json" if key == "summary" else ".jsonl"))),
            "non-train export path refused",
        )
    requests = [
        decode(line)
        for line in read_ref(inputs["requests"], check_deadline, maximum=16 * 1024**2).splitlines()
        if line.strip()
    ]
    labels = [
        decode(line)
        for line in read_ref(inputs["labels"], check_deadline, maximum=4 * 1024**2).splitlines()
        if line.strip()
    ]
    summary = decode(read_ref(inputs["summary"], check_deadline))
    grouped = validate_train_inventory(requests, labels, summary)
    label_by_id = {row["case_id"]: row for row in labels}
    pinned.extend(inputs.values())
    # Root selects these hashes independently of a collector-authored manifest.
    sources = selection["source_records"]
    require(
        type(sources) is dict
        and set(sources)
        == {
            "native_actor",
            "native_plan",
            "native_helper",
            "np5_actor",
            "observer",
            "numpy_probe",
            "head_core",
            "owner_controller",
        },
        "exact selected collector dependency projection required",
    )
    require(
        sources["head_core"]["path"] == str(CORE) and sources["head_core"]["sha256"] == CORE_SHA,
        "frozen core differs",
    )
    require(
        Path(selection["collector_source"]["path"]).parent == HERE,
        "selected train-only collector source required",
    )
    for ref in [*sources.values(), selection["collector_source"]]:
        read_ref(ref, check_deadline)
        pinned.append(ref)
    prerequisite_ref = selection["root_np5_prerequisite_ref"]
    prerequisite = decode(read_ref(prerequisite_ref, check_deadline))
    catalog_ref = validate_np_acceptance(prerequisite, selection, check_deadline)
    read_ref(catalog_ref, check_deadline)
    pinned.extend([prerequisite_ref, catalog_ref, *prerequisite["proof_refs"].values()])
    require(
        type(selection["baseline"].get("FP32_bytes_sha256")) is str
        and len(selection["baseline"]["FP32_bytes_sha256"]) == 64,
        "selected raw FP32 W0 pin required",
    )
    baseline = load_array(np, selection["baseline"], None, (255, 1024), check_deadline)
    pinned.append(selection["baseline"])
    entries = selection["chunks"]
    require(
        type(entries) is list
        and len(entries) == 12
        and {e["ancestor_id"] for e in entries} == set(ANCESTORS),
        "all twelve committed train ancestors required",
    )
    feature_rows, target_rows, counts = [], [], []
    seen, skipped = set(), 0
    for entry in entries:
        check_deadline()
        require(
            entry["family_id"] == entry["ancestor_id"].split("/")[0],
            "selected ancestor family differs",
        )
        accepted = decode(read_ref(entry["acceptance"], check_deadline))
        validate_commit_header(accepted, entry, selection)
        original_proof = validate_original_closure(accepted["proof_refs"], check_deadline)
        chunk = decode(read_ref(entry["chunk"], check_deadline))
        parent = Path(entry["chunk"]["path"]).parent
        require(
            chunk["schema"] == "decision3-dsl01-train-feature-chunk-v1"
            and chunk["status"] == "PENDING_TRAIN_ONLY"
            and chunk["training_eligible"] is False
            and chunk["split"] == "train"
            and chunk["dsl_version"] == DSL_VERSION,
            "original immutable pending collector chunk required",
        )
        require(
            chunk["ancestor_id"] == entry["ancestor_id"]
            and chunk["family_id"] == entry["family_id"]
            and chunk["model"] == model
            and chunk["source_records"] == sources
            and chunk["collector_source"] == selection["collector_source"],
            "accepted source/model/ancestor differs",
        )
        require(
            chunk["train_inputs"] == {key: inputs[key] for key in ("requests", "summary")},
            "collector read a different train projection",
        )
        require(
            chunk["native_plan_sha256"] == sources["native_plan"]["sha256"],
            "native plan binding differs",
        )
        require(
            chunk["root_np5_prerequisite_ref"] == prerequisite_ref,
            "collector used another NP5 prerequisite",
        )
        pinned.append(
            validate_producer_binding(chunk, parent, original_proof, selection, check_deadline)
        )
        require(
            [c["case_id"] for c in chunk["cases"]]
            == [r["case_id"] for r in grouped[entry["ancestor_id"]]],
            "case inventory/order differs",
        )
        same_w0 = load_array(np, chunk["baseline"], parent, (255, 1024), check_deadline)
        require(
            chunk["baseline"]["sha256"] == selection["baseline"]["sha256"]
            and chunk["baseline"]["FP32_bytes_sha256"] == selection["baseline"]["FP32_bytes_sha256"]
            and np.array_equal(same_w0, baseline),
            "chunk W0 is not the selected baseline",
        )
        payloads = [normalized(chunk["baseline"], parent)]
        for index, (case, original) in enumerate(
            zip(chunk["cases"], grouped[entry["ancestor_id"]]), 1
        ):
            require(
                same_identity(case, original) and case["case_id"] not in seen,
                "case/native identity reused or differs",
            )
            seen.add(case["case_id"])
            request = original["request"]
            heads = list(request["questions"])
            require(
                case["ordered_heads"] == heads and len(case["rows"]) == 4,
                "actual four-row head order differs",
            )
            features = load_array(np, case["features"], parent, (4, 1024), check_deadline)
            raw = decode(read_ref(case["raw"], check_deadline, parent=parent))
            evidence = decode(read_ref(case["evidence"], check_deadline, parent=parent))
            require(
                evidence["schema"] == "decision3-dsl01-train-case-evidence-v1"
                and evidence["status"] == "PENDING_TRAIN_ONLY"
                and evidence["training_eligible"] is False
                and evidence["ordered_heads"] == heads
                and evidence["rows"] == case["rows"],
                "original case evidence differs",
            )
            require(
                case["complete_prediction_seconds"]
                == accepted["transaction"]["complete_call_seconds"][index - 1],
                "post-watchdog call timing differs",
            )
            validate_case_evidence(evidence, case, original, index)
            compare_native(np, core, features, baseline, request, raw)
            for h_index, (row, head) in enumerate(zip(case["rows"], heads)):
                criteria = list(request["questions"][head]["criteria"])
                require(
                    row["case_id"] == case["case_id"]
                    and row["native_payload_sha256"] == original["native_payload_sha256"]
                    and row["head_id"] == head
                    and type(row["h_index"]) is int
                    and row["h_index"] == h_index
                    and row["ordered_criteria_ids"] == criteria
                    and type(row["legal_count"]) is int
                    and row["legal_count"] == len(criteria)
                    and row["same_call_id"] == "call-" + str(index).zfill(4)
                    and type(row["native_call_index"]) is int
                    and row["native_call_index"] == index,
                    "feature row/actual ordered criterion join differs",
                )
                labeled = label_by_id[case["case_id"]]["heads"]
                if head not in labeled:
                    skipped += 1
                    continue
                feature_rows.append(features[h_index])
                target_rows.append(target_vector(labeled[head], criteria))
                counts.append(len(criteria))
            payloads.extend(
                normalized(case[key], parent) for key in ("features", "raw", "evidence")
            )
        require(
            accepted["accepted_files"] == payloads,
            "root acceptance does not pin every original payload",
        )
        pinned.extend(
            [entry["acceptance"], entry["chunk"], *accepted["proof_refs"].values(), *payloads]
        )
    require(
        len(seen) == 96 and 96 <= len(feature_rows) <= 384,
        "complete labeled train inventory required",
    )
    x = np.stack(feature_rows).astype(np.float32, copy=False)
    y = np.asarray(target_rows, dtype=np.float32)
    n = np.asarray(counts, dtype=np.int64)
    require(
        x.shape == (len(feature_rows), 1024)
        and y.shape == (len(feature_rows), 255)
        and bool(np.isfinite(x).all())
        and bool(np.isfinite(y).all()),
        "actual loss matrices differ",
    )
    require(
        bool(np.allclose(y.sum(axis=1, dtype=np.float64), 1.0, rtol=0, atol=1e-6))
        and bool((y[np.arange(255)[None] >= n[:, None]] == 0).all()),
        "all-zero target or illegal-column label mass",
    )
    for array in (x, y, n):
        array.flags.writeable = False
    check_deadline()
    return {
        "features": x,
        "baseline": baseline,
        "counts": n,
        "targets": y,
        "pinned_refs": pinned,
        "aggregate": {
            "train_states": 96,
            "ancestors": 12,
            "families": 4,
            "supervised_heads": len(feature_rows),
            "unused_heads_skipped": skipped,
            "projection": PROJECTION,
            "dev_or_sealed_selected": False,
            "OS_access_sealing_claim": False,
        },
    }


def reauthenticate(refs, check_deadline):
    for ref in refs:
        read_ref(ref, check_deadline, maximum=16 * 1024**2)
