"""HELD root-only closed train-chunk acceptance; stdlib, no NN or optimizer.

Root must release this source explicitly after review. The collector's pending
manifest is never edited. The held train loader supplies pure validators only;
load_training, NumPy, model/generator imports and diagnostic arrays are unused.
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
import types
from datetime import datetime
from pathlib import Path

SOURCE_ONLY = False
ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
HERE = ROOT / "artifacts/jev-runtime/decision3-train-readout-cpu5-20261024-01"
OWNER = ROOT / "artifacts/jev-runtime/llama-cpp-owner-20261024-09"
DSL_ROOT = ROOT / "artifacts/jev-runtime/browser-decision-dsl-20261023-01"
TRAIN = DSL_ROOT / "corpus/data/train"
CONTROLLER_SHA = "81846efe2ee0357da07283c1a77a606283a920cbddef668c8f62def291057dee"
STAT_KEYS = {"st_dev", "st_ino", "st_size", "st_mtime_ns"}
READS = {}


def require(value, message):
    if not value:
        raise ValueError(message)


def released():
    require(SOURCE_ONLY is False, "SOURCE_ONLY: root chunk audit is not released")


def local(path):
    path = Path(path)
    require(
        path.is_absolute()
        and path.is_relative_to(ROOT)
        and ".." not in path.parts
        and not any(p.is_symlink() for p in (path, *path.parents)),
        "ordinary absolute repository path required",
    )
    require(
        not path.is_relative_to(DSL_ROOT)
        or path in {TRAIN / "requests.jsonl", TRAIN / "summary.json"},
        "audit refuses authoring/full manifest/dev/sealed/labels bytes",
    )
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


def load_source(name, path, expected):
    blob, ref = fetch(path, expected, maximum=1024**2)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(blob, str(path), "exec", dont_inherit=True), vars(module))
    return module, ref


def authenticate(row):
    """Honor each selected inventory's original SHA or stat-only scope."""
    path = Path(row["path"])
    path = path if path.is_absolute() else ROOT / path
    require(
        path.is_absolute()
        and ".." not in path.parts
        and (
            not path.is_relative_to(DSL_ROOT)
            or path in {TRAIN / "requests.jsonl", TRAIN / "summary.json"}
        ),
        "inventory cannot select other corpus bytes",
    )
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


def execute(args):
    released()
    out = local(args.owner_output)
    require(
        out.parent == OWNER and re.fullmatch(r"actual-[0-9]{2}", out.name),
        "exact original owner09 output required",
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
    loader, loader_ref = load_source(
        "_root_closed_train_loader_metadata", HERE / "train_loader.py", args.loader_sha
    )
    decode = loader.decode
    _, terminal_ref = fetch(terminal_path, args.terminal_sha)
    proof_refs = {"root_execution": terminal_ref}
    for key, name in (
        ("owner_receipt", "receipt.json"),
        ("closure", "external-boundary-closure.json"),
        ("admission", "admission.json"),
        ("sealed_config", "sealed-config.json"),
    ):
        _, proof_refs[key] = fetch(out / name)
    proof = loader.validate_original_closure(proof_refs, lambda: None)
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
        and terminal["profile"] == "root-selected-train-collector-cpu5-terminal-v1",
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
        len(expected_sources) == len(spec["sources"]) == 56,
        "exact56 selected collector sources required",
    )
    for path in base_paths:
        _, ref = fetch(path, CONTROLLER_SHA if path == OWNER / "controller.py" else None)
        row = {**ref, "path": path.relative_to(ROOT).as_posix()}
        require(row["path"] not in expected_sources, "base/selected source collision")
        expected_sources[row["path"]] = row
    require(
        len(expected_sources) == len(config["sources"]) == 61
        and {r["path"]: r for r in config["sources"]} == expected_sources
        and config["binary_stats"] == spec["binary_stats"],
        "exact original61 source union/tool stat selection required",
    )
    for row in [*config["sources"], *spec["binary_stats"]]:
        authenticate(row)
    chunk_parent = out / "train-chunk"
    chunk_blob, chunk_ref = fetch(chunk_parent / "chunk.json", args.chunk_sha)
    chunk = decode(chunk_blob)
    require(
        chunk["schema"] == "decision3-dsl01-train-feature-chunk-v1"
        and chunk["status"] == "PENDING_TRAIN_ONLY"
        and chunk["training_eligible"] is False
        and chunk["completed"] is True
        and chunk["split"] == "train"
        and chunk["dsl_version"] == loader.DSL_VERSION
        and chunk["external_root_acceptance_required"] is True
        and chunk["final_full_model_runtime_source_input_authentication"] is True
        and not chunk.get("error"),
        "immutable completed PENDING train chunk required",
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
        and chunk["train_capability_ref"] == plan["train_capability"]
        and chunk["train_inputs"] == plan["train_inputs"],
        "root independently selected source/input/plan identity differs",
    )
    require(
        loader.validate_producer_binding(chunk, chunk_parent, proof, selection, lambda: None)
        == plan_ref,
        "actual selected plan differs",
    )
    collector, _ = load_source(
        "_root_closed_train_collector_metadata",
        Path(plan["collector_source"]["path"]),
        plan["collector_source"]["sha256"],
    )
    require(
        set(plan["train_inputs"]) == {"requests", "summary"}
        and all(
            plan["train_inputs"][key]["path"] == str(TRAIN / name)
            for key, name in (("requests", "requests.jsonl"), ("summary", "summary.json"))
        ),
        "only fixed train request/summary exports may be opened",
    )
    requests, _ = verify(plan["train_inputs"]["requests"], maximum=16 * 1024**2)
    summary, _ = verify(plan["train_inputs"]["summary"])
    original = collector.select_cases(requests, summary, plan["ancestor_id"])
    require(
        chunk["ancestor_id"] == plan["ancestor_id"]
        and chunk["family_id"] == original[0]["family_id"]
        and [c["case_id"] for c in chunk["cases"]] == [r["case_id"] for r in original],
        "all eight train states in original order required",
    )
    prerequisite_blob, _ = verify(plan["np5_root_acceptance"])
    prerequisite = decode(prerequisite_blob)
    catalog_ref = loader.validate_np_acceptance(prerequisite, selection, lambda: None)
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
        and chunk["model"] == native["model"]
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
    require(
        state["profile"] == chunk["profile"] == plan["profile"]
        and state["plan_sha256"] == args.plan_sha
        and state["status"] == "PENDING_TRAIN_ONLY"
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
    require(
        "FP32_bytes_sha256" in chunk["baseline"]
        and Path(baseline_ref["path"]) == chunk_parent / "baseline.npy",
        "original baseline path/raw FP32 pin required",
    )
    accepted_files, numeric, times = [baseline_ref], [], []
    for index, (case, request_row, call) in enumerate(
        zip(chunk["cases"], original, state["calls"]), 1
    ):
        require(
            loader.same_identity(case, request_row)
            and case["rows"] == collector.mapping(request_row, index)
            and case["ordered_heads"] == list(request_row["request"]["questions"]),
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
            and call["status"] == "PENDING_TRAIN_ONLY"
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
            require(
                Path(ref["path"]) == chunk_parent / "cases" / case["case_id"] / name,
                "exact same-case payload path required",
            )
        raw, evidence = decode(raw_blob), decode(evidence_blob)
        require(
            evidence["schema"] == "decision3-dsl01-train-case-evidence-v1"
            and evidence["status"] == "PENDING_TRAIN_ONLY"
            and evidence["training_eligible"] is False
            and evidence["completed"] is True
            and evidence["rows"] == case["rows"]
            and evidence["ordered_heads"] == case["ordered_heads"]
            and not evidence.get("error"),
            "complete immutable same-case evidence required",
        )
        loader.validate_case_evidence(evidence, case, request_row, index)
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
        collector.check_catalog(evidence["numpy_runtime"], accepted_catalog, complete=True)
        rows = witness(features, baseline, request_row["request"], raw, lengths)
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
    require(
        digest_ok(args.labels_sha) and type(args.labels_size) is int and args.labels_size > 0,
        "root independently selected labels metadata pin required; labels bytes are not read here",
    )
    inputs = {
        **plan["train_inputs"],
        "labels": {
            "path": str(TRAIN / "labels.jsonl"),
            "sha256": args.labels_sha,
            "size_bytes": args.labels_size,
        },
    }
    acceptance = {
        "schema": "root-closed-dsl01-train-chunk-acceptance-v1",
        "status": "COMMITTED",
        "training_eligible": True,
        "split": "train",
        "dsl_version": loader.DSL_VERSION,
        "ancestor_id": chunk["ancestor_id"],
        "family_id": chunk["family_id"],
        "producer_chunk": chunk_ref,
        "train_inputs": inputs,
        "source_records": plan["dependencies"],
        "collector_source": plan["collector_source"],
        "model": chunk["model"],
        "projection": loader.PROJECTION,
        "root_np5_prerequisite_ref": plan["np5_root_acceptance"],
        "baseline_sha256": baseline_ref["sha256"],
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
        "loader_source": loader_ref,
        "selected_plan": plan_ref,
        "command_spec": spec_ref,
        "actor_state": state_ref,
        "source_union": 61,
        "model_files_sha256": 20,
        "runtime_files": {"sha256": 22, "stat_only": 6},
        "train_states": 8,
        "same_forward_feature_rows": 32,
        "labels_bytes_read": False,
        "labels_pin_authority": "independent root CLI metadata; authenticated against actual train labels before loss by held loader",
        "original_manifest_write_elapsed_seconds": before_write,
        "whole_seconds_scope": "Original completed child command elapsed; enclosing upper bound including entry/startup/final writes/watchdog cleanup, not a fabricated actor elapsed_seconds field",
        "numeric_evidence": numeric,
        "memory_diagnostics": {
            "peak_bytes": counters["memory.peak"],
            "events": counters["memory.events"],
        },
        "FP32_reexecuted_by_root": False,
        "new_model_calls": 0,
        "fit_performed": False,
        "browser_qualified": False,
        "benchmark_qualified": False,
        "model_qualified": False,
        "ranking_or_win_claim": False,
        "full_DSO_or_transitive_closure_qualified": False,
        "OS_access_sealing_claim": False,
        "scope": "Root actual audit accepts only this original train ancestor chunk. Verified file hashes/finite FP32 bytes and all legal-slot stdlib float64 witnesses; actual FP32 Torch/NumPy parity is retained from the independently selected completed producer source and evidence. No optimizer/model/scorer execution, diagnostic-array promotion, labels read, dev/sealed or quality qualification. Original PENDING producer files remain unchanged.",
    }
    entry = {
        "ancestor_id": chunk["ancestor_id"],
        "family_id": chunk["family_id"],
        "chunk": chunk_ref,
    }
    check_selection = {
        **selection,
        "train_inputs": inputs,
        "model": chunk["model"],
        "projection": loader.PROJECTION,
        "baseline": baseline_ref,
    }
    loader.validate_commit_header(acceptance, entry, check_selection)
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
    parser.add_argument("--loader-sha", required=True)
    parser.add_argument("--labels-sha", required=True)
    parser.add_argument("--labels-size", type=int, required=True)
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
                "train_states": 8,
                "same_forward_feature_rows": 32,
                "acceptance": str(target),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
