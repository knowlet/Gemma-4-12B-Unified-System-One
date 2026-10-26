"""HELD root-only closed dev-feature acceptance; stdlib, no labels or model calls.

Run separately with root-selected parameters for each of the three closed dev
chunks. This copies only pure metadata/numerical validators from the prior train
audit/loader and the selected dev collector; no target module is imported. The
original PENDING_DEV_NONTRAINING producer files remain unchanged. Root must
review, check and explicitly release a new copy before any actual reads.
"""

from __future__ import annotations

import argparse
import array
import ast
import hashlib
import json
import math
import os
import re
import stat
import struct
import sys
from datetime import datetime
from pathlib import Path

SOURCE_ONLY = False
ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
HERE = ROOT / "artifacts/jev-runtime/decision3-frozen-heldout-cpu5-20261024-01"
OWNER = ROOT / "artifacts/jev-runtime/llama-cpp-owner-20261024-09"
DSL_ROOT = ROOT / "artifacts/jev-runtime/browser-decision-dsl-20261023-01"
DSL = DSL_ROOT / "corpus/data/dev"
DSL_VERSION = "independent-browser-workflow-dsl/1"
PROFILE = "decision3-dsl01-dev-features-frozen-cpu5-4g-candidate-v1"
ROLE, SPLIT, PENDING = "dev_feature_collector", "dev", "PENDING_DEV_NONTRAINING"
UNIT = "jev-d3-llama-cpp-cpu5-fast-4g-coexist-20261024-09.service"
OWNER_PROFILE = "llama-cpp-cpu5-fast-same-cluster-4g-coexist-command-owned-v1"
COLLECTOR_SHA = "2c7cce27f5e801caf7a2d4e19868b485d288c452dee8932e485871fa259e8f70"
DEV_CAPABILITY_SHA256 = "917f208594a715ad19dc788682f5a41bcbe55c4667d1f726a2c4d909cd01aed2"
NUMERIC_ACCEPTANCE_SHA256 = "91f80677a7956dbdee721661c3f20b8dfcbf5df8d7948b7221debdbd84d174d1"
FROZEN_CANDIDATE_SHA256 = "239165d486bad16963a6e5c94c1f9503fc145ea5b85b114e9fbf94d84eb1fd76"
FROZEN_TRAIN_ACCEPTANCE_SHA256 = "cc629a1893ce53c232cc323253f39e86496b0e7a7347c5d11df8fa259ccb5baa"
CONTROLLER_SHA = "81846efe2ee0357da07283c1a77a606283a920cbddef668c8f62def291057dee"
IDENTITY_KEYS = {
    "case_id",
    "dsl_version",
    "family_id",
    "ancestor_id",
    "request_sha256",
    "native_payload_sha256",
}
IDENTITY = tuple(sorted(IDENTITY_KEYS))
NON_DEV_TRAIN_FAMILIES = frozenset(
    {"appointment_edit", "document_upload", "booking_inventory", "admin_filters"}
)
HEADS = frozenset({"operation", "click_target", "type_text_target", "select_target"})
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
STAT_KEYS = {"st_dev", "st_ino", "st_size", "st_mtime_ns"}
READS = {}


def require(value, message):
    if not value:
        raise ValueError(message)


def released():
    require(SOURCE_ONLY is False, "SOURCE_ONLY: root chunk audit is not released")


def dev_path_scope(path):
    """Apply the same corpus/payload deny rules before every file read/stat."""
    path = Path(path)
    require(
        (
            not path.is_relative_to(DSL_ROOT)
            or path in {DSL / "requests.jsonl", DSL / "summary.json"}
        )
        and "train-chunk" not in path.parts,
        "audit refuses authoring/full manifest/train/sealed/labels bytes",
    )
    if path.suffix == ".npy":
        require(
            any(
                path.is_relative_to(OWNER / name / "dev-chunk")
                for name in ("actual-23", "actual-24", "actual-25")
            ),
            "only selected closed dev feature/baseline arrays may be read",
        )


def local(path):
    path = Path(path)
    require(
        path.is_absolute()
        and path.is_relative_to(ROOT)
        and ".." not in path.parts
        and not any(p.is_symlink() for p in (path, *path.parents)),
        "ordinary absolute repository path required",
    )
    dev_path_scope(path)
    return path


def digest_ok(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def fetch(path, expected=None, *, maximum=2 * 1024**2):
    path = local(path)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(before.st_mode) and 0 < before.st_size <= maximum,
            "bounded regular source/metadata/payload required",
        )
        blob = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    keys = (*STAT_KEYS, "st_ctime_ns")
    require(
        len(blob) == before.st_size
        and all(getattr(before, k) == getattr(after, k) == getattr(path.stat(), k) for k in keys),
        "selected file changed while read",
    )
    ref = {"path": str(path), "sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)}
    require(
        expected is None or digest_ok(expected) and ref["sha256"] == expected,
        "independently selected SHA differs",
    )
    require(str(path) not in READS or READS[str(path)] == ref, "file changed across audit reads")
    READS[str(path)] = ref
    return blob, ref


def verify(ref, parent=None, *, maximum=2 * 1024**2):
    require(
        type(ref) is dict and {"path", "sha256", "size_bytes"} <= set(ref),
        "explicit SHA/size ref required",
    )
    path = Path(ref["path"])
    if not path.is_absolute():
        require(
            parent is not None and ".." not in path.parts, "relative ref requires selected parent"
        )
        path = parent / path
    blob, actual = fetch(path, ref["sha256"], maximum=maximum)
    require(
        type(ref["size_bytes"]) is int and actual["size_bytes"] == ref["size_bytes"],
        "selected file size differs",
    )
    return blob, {**ref, "path": actual["path"]}


def normalized(ref, parent=None):
    require(
        type(ref) is dict and {"path", "sha256", "size_bytes"} <= set(ref),
        "explicit file pin required",
    )
    path = Path(ref["path"])
    if not path.is_absolute():
        require(
            parent is not None and ".." not in path.parts,
            "relative pin requires its selected parent",
        )
        path = Path(parent) / path
    return {**ref, "path": str(local(path))}


def read_ref(ref, check_deadline, *, parent=None, maximum=2 * 1024**2):
    check_deadline()
    blob, _ = verify(normalized(ref, parent), maximum=maximum)
    check_deadline()
    return blob


def verify_record(ref):
    # Used by copied pure dev-capability checks; never imports a target module.
    return verify(ref)[0]


def authenticate(row):
    """Honor each selected inventory's original SHA or stat-only scope."""
    path = Path(row["path"])
    path = path if path.is_absolute() else ROOT / path
    require(
        path.is_absolute()
        and ".." not in path.parts
        and (
            not path.is_relative_to(DSL_ROOT)
            or path in {DSL / "requests.jsonl", DSL / "summary.json"}
        ),
        "inventory cannot select other corpus bytes",
    )
    dev_path_scope(path)
    resolved = path.resolve(strict=True)
    dev_path_scope(resolved)
    first = path.stat()
    require(stat.S_ISREG(first.st_mode), "regular selected inventory file required")
    size = row.get("size_bytes", row.get("size", first.st_size))
    require(type(size) is int and size == first.st_size, "selected inventory size differs")
    if "stat" in row:
        require(
            set(row["stat"]) == STAT_KEYS
            and all(type(v) is int and getattr(first, k) == v for k, v in row["stat"].items()),
            "selected inventory stat differs",
        )
    if "sha256" in row:
        require(digest_ok(row["sha256"]), "explicit inventory SHA required")
        digest, seen = hashlib.sha256(), 0
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            require(
                all(getattr(opened, k) == getattr(first, k) for k in STAT_KEYS),
                "inventory changed before open",
            )
            for block in iter(lambda: stream.read(1024**2), b""):
                seen += len(block)
                require(seen <= size, "inventory grew while hashed")
                digest.update(block)
            last = os.fstat(stream.fileno())
        require(
            seen == size
            and digest.hexdigest() == row["sha256"]
            and all(
                getattr(first, k) == getattr(last, k) == getattr(path.stat(), k)
                for k in (*STAT_KEYS, "st_ctime_ns")
            ),
            "inventory bytes/stat changed",
        )
    else:
        require("stat" in row, "stat-only inventory needs explicit original stat")


def npy(ref, parent, shape):
    require(
        ref.get("format") == "npy_no_pickle"
        and ref.get("dtype") == "float32"
        and ref.get("shape") == list(shape),
        "exact FP32 shape declaration required",
    )
    blob, normalized = verify(ref, parent)
    require(blob[:8] == b"\x93NUMPY\x01\x00" and len(blob) >= 10, "NPY1 header required")
    length = struct.unpack("<H", blob[8:10])[0]
    require(0 < length <= len(blob) - 10, "bounded complete NPY header required")
    header = ast.literal_eval(blob[10 : 10 + length].decode("latin1"))
    require(
        header == {"descr": "<f4", "fortran_order": False, "shape": shape},
        "finite C-order FP32/no-pickle header required",
    )
    data = blob[10 + length :]
    require(len(data) == math.prod(shape) * 4, "complete FP32 data bytes required")
    raw_sha = hashlib.sha256(data).hexdigest()
    if "FP32_bytes_sha256" in ref:
        require(
            digest_ok(ref["FP32_bytes_sha256"]) and raw_sha == ref["FP32_bytes_sha256"],
            "original raw FP32 digest differs from NPY file SHA",
        )
    values = array.array("f")
    require(values.itemsize == 4, "32-bit stdlib float storage required")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    require(all(math.isfinite(value) for value in values), "finite actual FP32 data required")
    return values, normalized, raw_sha


def producer_metrics(result, questions):
    require(
        result["absolute_tolerance"] == 1e-6
        and result["relative_tolerance"] == 1e-5
        and result["comparison"] == "abs(replay-native) <= 1e-6 + 1e-5*abs(native)"
        and result["all_legal_probabilities_matched"] is True
        and result["all_ordered_argmaxes_matched"] is True
        and result["semantic_correctness_evaluated"] is False,
        "original mixed-tolerance FP32 producer comparison required",
    )
    rows = result["rows"]
    require(
        [(r["head"], r["legal_options"]) for r in rows]
        == [(head, len(q["criteria"])) for head, q in questions.items()]
        and all(
            type(r["max_absolute_error"]) in (int, float)
            and math.isfinite(r["max_absolute_error"])
            and 0 <= r["max_absolute_error"] <= 1.1e-5
            for r in rows
        ),
        "complete original legal-slot FP32 metrics required",
    )


def witness(features, baseline, request, raw, lengths):
    questions = request["questions"]
    require(
        raw["model"] == "d3-edge"
        and list(raw["answers"]) == list(questions)
        and list(lengths) == list(questions)
        and all(type(n) is int and 0 < n <= 16384 for n in lengths.values())
        and raw["usage"] == {"input_tokens": sum(lengths.values()), "output_tokens": 0},
        "same-call complete raw native model/usage/order differs",
    )
    rows = []
    for i, (head, question) in enumerate(questions.items()):
        criteria, answer = list(question["criteria"]), raw["answers"][head]
        require(
            answer["type"] == "choice" and list(answer["probabilities"]) == criteria,
            "original complete ordered Choice output required",
        )
        original = list(answer["probabilities"].values())
        require(
            all(type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1 for p in original)
            and abs(sum(original) - 1) <= 1e-6,
            "finite normalized original probabilities required",
        )
        logits = [
            math.fsum(features[i * 1024 + j] * baseline[k * 1024 + j] for j in range(1024))
            for k in range(len(criteria))
        ]
        largest = max(logits)
        exps = [math.exp(value - largest) for value in logits]
        denominator = math.fsum(exps)
        probabilities = [value / denominator for value in exps]
        require(
            all(
                abs(p - native) <= 1e-6 + 1e-5 * abs(native)
                for p, native in zip(probabilities, original)
            ),
            "separate stdlib float64 legal-slot witness differs",
        )
        chosen = max(range(len(criteria)), key=original.__getitem__)
        require(
            answer["choice"]
            == criteria[chosen]
            == criteria[max(range(len(criteria)), key=probabilities.__getitem__)],
            "exact ordered argmax differs",
        )
        entropy = -sum(p * math.log(p) for p in original if p > 0)
        confidence = (
            1.0 if len(criteria) == 1 else max(0.0, min(1.0, 1 - entropy / math.log(len(criteria))))
        )
        require(
            type(answer["confidence"]) in (int, float)
            and math.isfinite(answer["confidence"])
            and abs(answer["confidence"] - confidence) <= 1e-6,
            "singleton/entropy confidence differs",
        )
        rows.append(
            {
                "head": head,
                "legal_count": len(criteria),
                "max_absolute_error": max(
                    abs(p - native) for p, native in zip(probabilities, original)
                ),
            }
        )
    return rows


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


def same_identity(left, right):
    return {k: left[k] for k in IDENTITY} == {k: right[k] for k in IDENTITY}


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
    # Pure copied NP closure validation; this auditor never imports NumPy.
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
        parent == Path(config["output"]) / "dev-chunk"
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
        plan["schema"] == "decision3-dsl01-dev-collector-plan-v1"
        and plan["profile"] == chunk["profile"]
        and plan["status"] == "selected_after_separate_review"
        and plan["ancestor_id"] == chunk["ancestor_id"]
        and plan["dependencies"] == selection["source_records"]
        and plan["collector_source"] == selection["collector_source"]
        and plan["dev_inputs"] == chunk["dev_inputs"]
        and plan["np5_root_acceptance"] == selection["root_np5_prerequisite_ref"],
        "actual selected collector plan does not bind this chunk",
    )
    return plan_ref


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


def validate_dev_plan(plan):
    """Selected metadata only; never import a producer or open any label/delta."""
    require(
        plan["schema"] == "decision3-dsl01-dev-collector-plan-v1"
        and plan["profile"] == PROFILE
        and plan["status"] == "selected_after_separate_review"
        and plan["dsl_version"] == DSL_VERSION
        and plan["split"] == SPLIT
        and plan["role"] == ROLE
        and plan["training_eligible"] is False
        and plan["requests"] == 8
        and plan["wall_seconds"] == 120
        and plan["call_seconds"] == 20,
        "exact root-selected dev-only plan and original budgets required",
    )
    require(
        plan["owner_unit"] == UNIT
        and plan["owner_profile"] == OWNER_PROFILE
        and plan["memory_max_bytes"] == 4294967296
        and plan["swap_max_bytes"] == 0
        and plan["reserve_bytes"] == 26119172916
        and plan["startup_available_bytes"] == 30414140212
        and plan["cpu_affinity"] == list(range(15, 20))
        and plan["cpu_quota_percent"] == 500
        and plan["threads"] == 5,
        "original CPU5/full4GiB/no-swap resource selection differs",
    )
    native = ROOT / "artifacts/jev-runtime/decision3-publisher-batch4-cpu-20261024-05"
    np5 = ROOT / "artifacts/jev-runtime/decision3-readout-numpy-cpu5-qualification-20261024-02"
    dependencies = {
        "native_actor": native / "actor.py",
        "native_plan": native / "edge-plan.json",
        "native_helper": native / "resource_helper.py",
        "np5_actor": np5 / "actor.py",
        "observer": np5 / "observer.py",
        "numpy_probe": np5 / "numpy_probe.py",
        "head_core": ROOT
        / "artifacts/jev-runtime/decision3-cached-head-core-20261023-01/head_core.py",
        "owner_controller": OWNER / "controller.py",
    }
    require(set(plan["dependencies"]) == set(dependencies), "exact eight dependencies required")
    for key, path in dependencies.items():
        require(plan["dependencies"][key]["path"] == str(path), "selected dependency path differs")
        verify(plan["dependencies"][key])
    require(
        plan["collector_source"]["path"] == str(HERE / "eval_collector_released_v1.py")
        and plan["collector_source"]["sha256"] == COLLECTOR_SHA
        and plan["collector_source"]["size_bytes"] == 39229,
        "exact executed dev collector source required",
    )
    verify(plan["collector_source"])
    require(
        plan["dev_capability"]["sha256"] == DEV_CAPABILITY_SHA256,
        "selected dev export capability differs",
    )
    capability = decode(verify_record(plan["dev_capability"]))
    require(
        capability["schema"] == "root-selected-dsl01-dev-collector-capability-v1"
        and capability["status"] == "ROOT_SELECTED_FROZEN_DEV_COLLECTION"
        and capability["dev_inputs"] == plan["dev_inputs"]
        and capability["frozen_candidate_ref"] == plan["frozen_candidate_ref"]
        and capability["numeric_root_acceptance_ref"] == plan["numeric_root_acceptance_ref"]
        and capability["training_eligible"] is False
        and capability["labels_allowed"] is False
        and capability["sealed_reveal_authorized"] is False,
        "nontraining frozen dev export authority required",
    )
    _, ancestors = capability_inventory(capability)
    require(plan["ancestor_id"] in ancestors, "selected non-dev ancestor refused")
    candidate = frozen_candidate(plan)
    return capability, candidate


def dev_taint(value):
    require(
        value["split"] == SPLIT
        and value["role"] == ROLE
        and value["labels"] is None
        and value["training_eligible"] is False,
        "permanent dev NONTRAINING role/labels taint required",
    )


def payload_taint(ref):
    require(
        ref["split"] == SPLIT and ref["status"] == PENDING and ref["training_eligible"] is False,
        "original dev payload taint required",
    )


def execute(args):
    released()
    out = local(args.owner_output)
    require(
        out.parent == OWNER and out.name in {"actual-23", "actual-24", "actual-25"},
        "root-selected original owner09 dev output23/24/25 required",
    )
    terminal_path = local(args.terminal_path)
    require(
        terminal_path.parent == OWNER
        and terminal_path.is_file()
        and (out / "receipt.json").is_file(),
        "original root terminal and owner receipt must exist before actual chunk reads",
    )
    require(
        not any(
            name.split(".")[0] in {"numpy", "torch", "transformers", "safetensors", "s1"}
            for name in sys.modules
        ),
        "fresh root stdlib audit without numerical/model imports required",
    )
    _, terminal_ref = fetch(terminal_path, args.terminal_sha)
    proof_refs = {"root_execution": terminal_ref}
    for key, name in (
        ("owner_receipt", "receipt.json"),
        ("closure", "external-boundary-closure.json"),
        ("admission", "admission.json"),
        ("sealed_config", "sealed-config.json"),
    ):
        _, proof_refs[key] = fetch(out / name)
    proof = validate_original_closure(proof_refs, lambda: None)
    terminal, receipt, config = (
        proof["root_execution"],
        proof["owner_receipt"],
        proof["sealed_config"],
    )
    require(
        receipt["comfy_service_modified"] is False
        and proof["closure"]["selection"]["boot_id"] == config["boot_id"]
        and config["output"] == str(out),
        "current successful untouched original owner required",
    )
    plan_blob, plan_ref = fetch(args.plan, args.plan_sha)
    spec_blob, spec_ref = fetch(args.spec, args.spec_sha)
    plan, spec = decode(plan_blob), decode(spec_blob)
    require(
        config["selection"] == str(args.spec)
        and config["selection_sha256"] == args.spec_sha
        and terminal["profile"] == "root-selected-frozen-dev-cpu5-terminal-v1",
        "root command/spec selection differs",
    )
    expected_root = [
        str(ROOT / ".venv/bin/python"),
        "-B",
        "-s",
        str(OWNER / "controller.py"),
        "--run",
        "--expected-entry-sha256",
        CONTROLLER_SHA,
        "--selection",
        str(args.spec),
        "--selection-sha256",
        args.spec_sha,
        "--output",
        str(out),
    ]
    require(terminal["argv"] == expected_root, "exact original root argv differs")
    base_paths = [
        OWNER / "controller.py",
        args.spec,
        ROOT / "artifacts/jev-runtime/decision3-local-owner-terminal-20261022-01/owner_boundary.py",
        ROOT / "artifacts/jev-runtime/observer-system-services-20261022-01/comfy_observer.py",
        ROOT / "artifacts/jev-runtime/observer-system-services-20261022-01/system-services.json",
    ]
    expected_sources = {row["path"]: row for row in spec["sources"]}
    require(
        len(expected_sources) == len(spec["sources"]) == 57,
        "exact57 selected dev collector sources required",
    )
    for path in base_paths:
        _, ref = fetch(path, CONTROLLER_SHA if path == OWNER / "controller.py" else None)
        row = {**ref, "path": path.relative_to(ROOT).as_posix()}
        require(row["path"] not in expected_sources, "base/selected source collision")
        expected_sources[row["path"]] = row
    require(
        len(expected_sources) == len(config["sources"]) == 62
        and {r["path"]: r for r in config["sources"]} == expected_sources
        and config["binary_stats"] == spec["binary_stats"],
        "exact original62 source union/tool stat selection required",
    )
    for row in [*config["sources"], *spec["binary_stats"]]:
        authenticate(row)
    chunk_parent = out / "dev-chunk"
    chunk_blob, chunk_ref = fetch(chunk_parent / "chunk.json", args.chunk_sha)
    chunk = decode(chunk_blob)
    require(
        chunk["schema"] == "decision3-dsl01-dev-feature-chunk-v1"
        and chunk["status"] == PENDING
        and chunk["training_eligible"] is False
        and chunk["completed"] is True
        and chunk["split"] == SPLIT
        and chunk["role"] == ROLE
        and chunk["dsl_version"] == DSL_VERSION
        and chunk["labels"] is None
        and chunk["evaluation_only"] is True
        and chunk["sealed_reveal_authorized"] is False
        and chunk["train_COMMITTED_authority"] is False
        and chunk["external_root_acceptance_required"] is True
        and chunk["final_full_model_runtime_source_input_authentication"] is True
        and not chunk.get("error"),
        "immutable completed PENDING dev NONTRAINING chunk required",
    )
    selection = {
        "source_records": plan["dependencies"],
        "collector_source": plan["collector_source"],
        "root_np5_prerequisite_ref": plan["np5_root_acceptance"],
    }
    require(
        chunk["source_records"] == selection["source_records"]
        and chunk["collector_source"] == selection["collector_source"]
        and chunk["selected_plan_sha256"] == args.plan_sha
        and chunk["root_np5_prerequisite_ref"] == selection["root_np5_prerequisite_ref"]
        and chunk["dev_capability_ref"] == plan["dev_capability"]
        and chunk["dev_inputs"] == plan["dev_inputs"]
        and all(
            chunk[key] == plan[key]
            for key in (
                "frozen_candidate_ref",
                "frozen_train_acceptance_ref",
                "numeric_root_acceptance_ref",
            )
        ),
        "root independently selected source/input/plan identity differs",
    )
    require(
        validate_producer_binding(chunk, chunk_parent, proof, selection, lambda: None) == plan_ref,
        "actual selected plan differs",
    )
    require(
        set(plan["dev_inputs"]) == {"requests", "summary"}
        and all(
            plan["dev_inputs"][key]["path"] == str(DSL / name)
            for key, name in (("requests", "requests.jsonl"), ("summary", "summary.json"))
        ),
        "only fixed dev request/summary exports may be opened",
    )
    capability, candidate = validate_dev_plan(plan)
    requests, _ = verify(plan["dev_inputs"]["requests"], maximum=16 * 1024**2)
    summary, _ = verify(plan["dev_inputs"]["summary"])
    original = select_cases(requests, summary, plan["ancestor_id"], capability)
    require(
        chunk["ancestor_id"] == plan["ancestor_id"]
        and chunk["family_id"] == original[0]["family_id"]
        and [c["case_id"] for c in chunk["cases"]] == [r["case_id"] for r in original],
        "all eight dev states in original order required",
    )
    prerequisite_blob, _ = verify(plan["np5_root_acceptance"])
    prerequisite = decode(prerequisite_blob)
    catalog_ref = validate_np_acceptance(prerequisite, selection, lambda: None)
    catalog_blob, _ = verify(catalog_ref)
    pending_catalog = decode(catalog_blob)
    require(
        pending_catalog["status"] == "PENDING_CLOSED_ROOT_ACCEPTANCE"
        and pending_catalog["training_eligible"] is False,
        "original accepted NP catalog remains nontraining",
    )
    accepted_catalog = pending_catalog["after"]
    for row in accepted_catalog["critical_files_and_mapped_math_candidates"]:
        require(
            digest_ok(row.get("sha256"))
            and set(row["stat"]) == STAT_KEYS
            and row["size_bytes"] == row["stat"]["st_size"]
            and str(Path(row["path"]).resolve(strict=True)) == row["resolved_path"],
            "closed CPU catalog SHA/stat/resolved identity required",
        )
        authenticate(row)
    for row in accepted_catalog["mapped_gpu_math_candidates_excluded_from_CPU_projection"]:
        require(
            "sha256" not in row
            and row["hashed"] is False
            and row["included_in_CPU_projection"] is False
            and row["execution_or_symbol_binding_proven"] is False,
            "GPU candidates remain stat-only/unqualified",
        )
        authenticate(row)
    native_blob, _ = verify(plan["dependencies"]["native_plan"])
    native = decode(native_blob)
    require(
        chunk["native_plan_sha256"] == plan["dependencies"]["native_plan"]["sha256"]
        and chunk["model"] == native["model"] == plan["model"]
        and len(native["model_files"]) == 20
        and all("sha256" in r for r in native["model_files"])
        and len(native["runtime_files"]) == 28
        and sum("sha256" in r for r in native["runtime_files"]) == 22,
        "exact original20 model hashes and28 runtime identities (22 hashes/6 stat-only) required",
    )
    for row in [
        *native["model_files"],
        native["manifest"],
        *native["runtime_files"],
        *native["owner_sources"],
    ]:
        authenticate(row)
    state_blob, state_ref = fetch(chunk_parent / "state.json")
    state = decode(state_blob)
    dev_taint(chunk)
    dev_taint(state)
    require(
        state["profile"] == chunk["profile"] == plan["profile"]
        and state["plan_sha256"] == args.plan_sha
        and state["status"] == "constructing_original_publisher_CPU_batch4"
        and state["training_eligible"] is False
        and state["completed"] is True
        and state["model_loads"] == 1
        and state["native_request_calls"] == len(state["calls"]) == 8
        and state["initial_full_model_manifest_runtime_sources_authenticated"] is True
        and state["process"] == chunk["owner"]["process"]
        and not state.get("error")
        and all(
            state[k] is False
            for k in ("no_escape_audited", "whole_tree_audited", "model_qualified")
        ),
        "one actual constructor/eight completed calls and nonqualification required",
    )
    require(
        state["evaluation_only"] is True
        and state["frozen_candidate_ref"] == plan["frozen_candidate_ref"]
        and state["numeric_root_acceptance_ref"] == plan["numeric_root_acceptance_ref"],
        "state frozen candidate/numeric/dev scope differs",
    )
    require(
        state["process"]["affinity"] == list(range(15, 20))
        and state["owned_limits"] == chunk["owner"]["owned_limits"]
        and state["owned_limits"]["memory.max"] == "4294967296"
        and state["owned_limits"]["memory.swap.max"] == "0",
        "original CPU5/full4GiB/no-swap contract required",
    )
    quota = state["owned_limits"]["cpu.max"].split()
    require(
        len(quota) == 2
        and all(v.isdigit() for v in quota)
        and int(quota[1]) > 0
        and int(quota[0]) == 5 * int(quota[1]),
        "original kernel quota ratio5 required",
    )
    provenance = state["native_provenance"]
    require(
        provenance["model_sha256"] == chunk["model"]["model_sha256"]
        and provenance["backbone_dtype"] == "bfloat16"
        and provenance["readout_dtype"] == "float32"
        and provenance["temperature"] == 1.0
        and provenance["batch_size"] == 4
        and state["native_runtime_info"]["device"] == "cpu"
        and state["actual_environment"]["CUDA_VISIBLE_DEVICES"] == ""
        and all(
            state["actual_environment"][k] == "5"
            for k in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        ),
        "original native CPU/BF16/FP32/T1 provenance required",
    )
    baseline, baseline_ref, baseline_sha = npy(chunk["baseline"], chunk_parent, (255, 1024))
    payload_taint(baseline_ref)
    require(
        "FP32_bytes_sha256" in chunk["baseline"]
        and Path(baseline_ref["path"]) == chunk_parent / "baseline.npy"
        and baseline_sha == candidate["baseline_fp32_bytes_sha256"],
        "original baseline path/raw FP32/frozen candidate pin required",
    )
    require(
        len(chunk["cases"]) == len(original) == len(state["calls"]) == 8,
        "all eight cases/calls required before ordered zip",
    )
    accepted_files, numeric, times, accepted_cases = [baseline_ref], [], [], []
    for index, (case, request_row, call) in enumerate(
        zip(chunk["cases"], original, state["calls"]), 1
    ):
        dev_taint(case)
        dev_taint(call)
        require(
            same_identity(case, request_row)
            and case["rows"] == mapping(request_row, index)
            and case["ordered_heads"] == list(request_row["request"]["questions"])
            and case["status"] == PENDING,
            "actual four-row request/native/head/criteria join differs",
        )
        elapsed = case["complete_prediction_seconds"]
        require(
            type(elapsed) in (int, float)
            and math.isfinite(elapsed)
            and 0 < elapsed <= 20
            and call["complete_prediction_seconds"] == elapsed
            and call["case_id"] == case["case_id"]
            and call["completed"] is True
            and call["status"] == PENDING
            and call["training_eligible"] is False
            and not call.get("error"),
            "all original post-watchdog call20 completions required",
        )
        times.append(elapsed)
        features, feature_ref, feature_sha = npy(case["features"], chunk_parent, (4, 1024))
        raw_blob, raw_ref = verify(case["raw"], chunk_parent)
        evidence_blob, evidence_ref = verify(case["evidence"], chunk_parent)
        for ref, name in (
            (feature_ref, "features.npy"),
            (raw_ref, "raw-native-response.json"),
            (evidence_ref, "evidence.json"),
        ):
            payload_taint(ref)
            require(
                Path(ref["path"]) == chunk_parent / "cases" / case["case_id"] / name,
                "exact same-case payload path required",
            )
        raw, evidence = decode(raw_blob), decode(evidence_blob)
        dev_taint(evidence)
        require(
            evidence["schema"] == "decision3-dsl01-dev-case-evidence-v1"
            and evidence["status"] == PENDING
            and evidence["training_eligible"] is False
            and evidence["completed"] is True
            and evidence["rows"] == case["rows"]
            and evidence["ordered_heads"] == case["ordered_heads"]
            and not evidence.get("error"),
            "complete immutable same-case evidence required",
        )
        validate_case_evidence(evidence, case, request_row, index)
        lengths = evidence["input_tokens_per_question"]
        expected_event = [[4, max(lengths.values()), 1024]]
        require(
            evidence["math"]["first_layer_events"]
            == evidence["math"]["final_norm_events"]
            == expected_event,
            "one original complete same-forward batch4 required",
        )
        for key in ("torch_replay", "numpy_zero_delta"):
            producer_metrics(evidence["math"][key], request_row["request"]["questions"])
        check_catalog(evidence["numpy_runtime"], accepted_catalog, complete=True)
        rows = witness(features, baseline, request_row["request"], raw, lengths)
        accepted_cases.append(
            {
                **{key: case[key] for key in IDENTITY_KEYS},
                "split": SPLIT,
                "role": ROLE,
                "labels": None,
                "training_eligible": False,
                "status": PENDING,
                "ordered_heads": case["ordered_heads"],
                "rows": case["rows"],
                "features_ref": {**feature_ref, "FP32_bytes_sha256": feature_sha},
                "raw_ref": raw_ref,
                "evidence_ref": evidence_ref,
            }
        )
        numeric.append(
            {
                "case_id": case["case_id"],
                "feature_FP32_bytes_sha256_computed": feature_sha,
                "baseline_FP32_bytes_sha256_verified": baseline_sha,
                "producer_FP32_comparisons": {
                    key: evidence["math"][key] for key in ("torch_replay", "numpy_zero_delta")
                },
                "additional_stdlib_float64_accumulation_witness": rows,
            }
        )
        accepted_files.extend((feature_ref, raw_ref, evidence_ref))
    command = receipt["state"]["actor_exit"]["command_results"][0]
    whole = command["elapsed_seconds"]
    before_write, wall = (
        chunk["elapsed_before_last_manifest_write_seconds"],
        terminal["wall_seconds"],
    )
    require(
        type(whole) in (int, float)
        and math.isfinite(whole)
        and type(before_write) in (int, float)
        and math.isfinite(before_write)
        and type(wall) in (int, float)
        and math.isfinite(wall)
        and sum(times) <= before_write <= whole <= 120
        and whole <= wall
        and abs(
            (
                datetime.fromisoformat(terminal["finished_at_utc"])
                - datetime.fromisoformat(terminal["started_at_utc"])
            ).total_seconds()
            - wall
        )
        < 0.1
        and chunk["whole_seconds"] == 120
        and chunk["call_seconds"] == 20,
        "whole120 verified by enclosing original command; original timing order required",
    )
    counters = chunk["after_final_auth_owned_counters"]["counters"]
    require(
        type(counters["memory.peak"]) is int
        and counters["memory.peak"] >= 0
        and type(counters["memory.events"]) is dict
        and all(type(v) is int and v >= 0 for v in counters["memory.events"].values()),
        "typed original memory diagnostics required",
    )
    acceptance = {
        "schema": "root-closed-dsl01-dev-chunk-acceptance-v1",
        "status": "ROOT_ACCEPTED_DEV_FEATURES_NONTRAINING",
        "training_eligible": False,
        "split": SPLIT,
        "role": ROLE,
        "labels": None,
        "evaluation_only": True,
        "sealed_reveal_authorized": False,
        "train_COMMITTED_authority": False,
        "dsl_version": DSL_VERSION,
        "owner_output": str(out),
        "ancestor_id": chunk["ancestor_id"],
        "family_id": chunk["family_id"],
        "chunk_ref": chunk_ref,
        "dev_inputs": plan["dev_inputs"],
        "dev_capability_ref": plan["dev_capability"],
        "frozen_candidate_ref": plan["frozen_candidate_ref"],
        "frozen_train_acceptance_ref": plan["frozen_train_acceptance_ref"],
        "numeric_root_acceptance_ref": plan["numeric_root_acceptance_ref"],
        "source_records": plan["dependencies"],
        "collector_source": plan["collector_source"],
        "model": chunk["model"],
        "projection": PROJECTION,
        "root_np5_prerequisite_ref": plan["np5_root_acceptance"],
        "baseline_ref": {**baseline_ref, "FP32_bytes_sha256": baseline_sha},
        "cases": accepted_cases,
        "accepted_files": accepted_files,
        "proof_refs": proof_refs,
        "transaction": {
            "case_count": 8,
            "call_budget_seconds": 20,
            "whole_budget_seconds": 120,
            "complete_call_seconds": times,
            "whole_seconds": whole,
        },
        "authority": "root",
        "separate_reviewer_actual_bytes_read": False,
        "independent_actual_review_claimed": False,
        "audit_source": fetch(Path(__file__).absolute())[1],
        "selected_plan": plan_ref,
        "command_spec": spec_ref,
        "actor_state": state_ref,
        "producer_state_phase_label_at_close": state["status"],
        "phase_label_scope": "Original native loader left its construction phase label; completed=true, original terminal/owner/closure and every actual call/evidence are checked separately. No producer file edited.",
        "source_union": 62,
        "model_files_sha256": 20,
        "runtime_files": {"sha256": 22, "stat_only": 6},
        "dev_states": 8,
        "same_forward_feature_rows": 32,
        "labels_bytes_read": False,
        "labels_selected": False,
        "original_manifest_write_elapsed_seconds": before_write,
        "whole_seconds_scope": "Original completed child command elapsed; enclosing upper bound including entry/startup/final writes/watchdog cleanup, not a fabricated actor elapsed_seconds field",
        "numeric_evidence": numeric,
        "memory_diagnostics": {
            "peak_bytes": counters["memory.peak"],
            "events": counters["memory.events"],
        },
        "FP32_reexecuted_by_root": False,
        "semantic_correctness_evaluated": False,
        "new_model_calls": 0,
        "fit_performed": False,
        "browser_qualified": False,
        "benchmark_qualified": False,
        "model_qualified": False,
        "ranking_or_win_claim": False,
        "full_DSO_or_transitive_closure_qualified": False,
        "OS_access_sealing_claim": False,
        "scope": "Root closed audit accepts only this original dev ancestor's permanent NONTRAINING features. File hashes/finite FP32 bytes and every legal-slot stdlib float64 witness are checked; original FP32 Torch/NumPy parity remains producer evidence. No labels, optimizer, model, scorer, delta array or train feature payload is read or executed here. No semantic metric, sealed exposure, browser qualification or quality claim. Original PENDING producer files remain unchanged.",
    }
    # Reauthenticate only the bounded source/metadata/payload bytes parsed here.
    # Inventory model/runtime files already received their original-scope check.
    for ref in tuple(READS.values()):
        verify(ref, maximum=16 * 1024**2)
    return acceptance


def main(argv=None):
    released()  # root source release precedes args, IO, imports or acceptance
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha", required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--spec-sha", required=True)
    parser.add_argument("--terminal-path", type=Path, required=True)
    parser.add_argument("--terminal-sha", required=True)
    parser.add_argument("--owner-output", type=Path, required=True)
    parser.add_argument("--chunk-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    target = local(args.output)
    require(
        target.parent == HERE and not target.exists(),
        "fresh exclusive root acceptance output required",
    )
    result = execute(args)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    print(
        json.dumps(
            {
                "status": result["status"],
                "ancestor_id": result["ancestor_id"],
                "dev_states": 8,
                "same_forward_feature_rows": 32,
                "acceptance": str(target),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
