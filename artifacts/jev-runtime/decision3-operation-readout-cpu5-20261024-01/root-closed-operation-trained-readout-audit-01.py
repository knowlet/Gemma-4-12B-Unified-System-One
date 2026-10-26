"""Root closed fit audit; stdlib only, no optimizer/model or evaluation rerun."""

import array
import ast
import hashlib
import importlib.util
import json
import math
import struct
import sys
from pathlib import Path

ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
HERE = Path(__file__).resolve().parent
OWNER = ROOT / "artifacts/jev-runtime/llama-cpp-owner-20261024-09"
OUT = OWNER / "actual-27"
SOURCE_ONLY = False
CAP_SHA = "3cbb5cb38afacdb0f16ca3c770f6565e1895e722246172ed01b957512c41ea4e"
SPEC_SHA = "fc565da496941626e833ba988d05a8b39df57e30ca9fbd89e025221781f17c31"
LOADER_SHA = "2a147760e6a18338d12a4bb33a53a92860ba5f362b2d91567c991f96baa6fe57"
FIT_SHA = "3f0819956854c070b707369e1aba0e25637d8cecb9c996e80b64285210453211"


def need(value, message):
    if not value:
        raise ValueError(message)


def blob(path, maximum=2 * 1024**2):
    path = Path(path)
    need(
        path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= maximum,
        "bounded ordinary input",
    )
    return path.read_bytes()


def critical_sha(path):
    first = path.stat()
    need(0 < first.st_size <= 256 * 1024**2, "original critical CPU file256MiB bound")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    last = path.stat()
    need(
        all(
            getattr(first, k) == getattr(last, k)
            for k in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        ),
        "critical file changed while hashed",
    )
    return digest.hexdigest()


def ref(path):
    raw = blob(path)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def read(path):
    return json.loads(blob(path))


def pinned(row):
    path = Path(row["path"])
    path = path if path.is_absolute() else ROOT / path
    actual = ref(path)
    need(
        actual["sha256"] == row["sha256"]
        and actual["size_bytes"] == row.get("size_bytes", row.get("size")),
        "exact selected file identity",
    )
    return path


def source(path, expected, name):
    need(ref(path)["sha256"] == expected, "pinned stdlib source")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def execute():
    need(SOURCE_ONLY is False, "SOURCE_ONLY: root operation audit held")
    terminal = OWNER / "root-execution-operation-fit-27.json"
    need(read(terminal)["exit_code"] == 0, "original terminal0 before actual fit reads")
    loader = source(HERE / "train_loader_released_v1.py", LOADER_SHA, "_root_fit_closed_loader")
    refs = {
        k: ref(p)
        for k, p in {
            "root_execution": terminal,
            "owner_receipt": OUT / "receipt.json",
            "closure": OUT / "external-boundary-closure.json",
            "admission": OUT / "admission.json",
            "sealed_config": OUT / "sealed-config.json",
        }.items()
    }
    proof = loader.validate_original_closure(refs, lambda: None)
    cfg, receipt = proof["sealed_config"], proof["owner_receipt"]
    cap_path = HERE / "root-committed-operation-fit-capability-01.json"
    spec_path = HERE / "root-fit-command-spec-01.json"
    need(
        ref(cap_path)["sha256"] == CAP_SHA
        and ref(spec_path)["sha256"] == SPEC_SHA
        and ref(HERE / "fit_head_released_v2.py")["sha256"] == FIT_SHA,
        "independent capability/spec/fit pins",
    )
    cap, spec = read(cap_path), read(spec_path)
    need(
        cfg["selection"] == str(spec_path)
        and cfg["selection_sha256"] == SPEC_SHA
        and cfg["output"] == str(OUT)
        and receipt["comfy_service_modified"] is False,
        "current unchanged original owner",
    )
    need(
        proof["root_execution"]["profile"] == "root-selected-operation-fit-cpu5-terminal-v1",
        "root fit profile",
    )
    base = [
        OWNER / "controller.py",
        spec_path,
        ROOT / "artifacts/jev-runtime/decision3-local-owner-terminal-20261022-01/owner_boundary.py",
        ROOT / "artifacts/jev-runtime/observer-system-services-20261022-01/comfy_observer.py",
        ROOT / "artifacts/jev-runtime/observer-system-services-20261022-01/system-services.json",
    ]
    expected = {row["path"]: row for row in spec["sources"]}
    for p in base:
        expected[str(p.relative_to(ROOT))] = {**ref(p), "path": str(p.relative_to(ROOT))}
    need(
        len(spec["sources"]) == 29
        and len(expected) == len(cfg["sources"]) == 34
        and expected == {r["path"]: r for r in cfg["sources"]}
        and cfg["binary_stats"] == spec["binary_stats"],
        "exact29+5 source union",
    )
    for row in cfg["sources"]:
        pinned(row)
    for row in spec["binary_stats"]:
        info = Path(row["path"]).stat()
        need(
            all(type(v) is int and getattr(info, k) == v for k, v in row["stat"].items()),
            "original binary stat identity",
        )
    command = receipt["state"]["actor_exit"]["command_results"][0]
    started = read(OUT / "command-01-started.json")
    intent = read(OUT / "command-01-intent.json")
    need(
        command["returncode"] == 0
        and command["pid"] == started["pid"]
        and command["argv"]
        == [s.replace("{OUTPUT}", str(OUT)) for s in spec["commands"][0]["argv"]]
        and command["argv"] == intent["argv"],
        "original successful exact child command",
    )
    identity = command["observed_identity"]
    child = identity["process"]
    wrapper = proof["admission"]["actor"]
    need(
        all(
            child[k] == started["observed_identity"]["process"][k]
            for k in ("pid", "start_ticks", "pgid", "sid", "uid", "cgroup")
        )
        and all(child[k] == wrapper[k] for k in ("pgid", "sid", "uid", "cgroup"))
        and all(identity[k] == started["observed_identity"][k] for k in ("exe", "argv", "cwd"))
        and identity["exe"] == intent["resolved_executable"]
        and identity["cwd"] == str(ROOT),
        "original child/start/cohort/exe/cwd",
    )
    summary_path = OUT / "fit-operation-head/fit-summary-pending.json"
    s = read(summary_path)
    need(
        s["schema"] == "dsl01-owned-operation-readout-pending-v1"
        and s["status"] == "PENDING_OPERATION_READOUT_ROOT_ACCEPTANCE"
        and s["training_eligible"] is False
        and s["selection_sha256"] == CAP_SHA
        and s["process"] == identity
        and s["original_owner_nonce"] == cfg["nonce"],
        "pending fit summary exact current child",
    )
    need(
        s["hyperparameters"] == cap["hyperparameters"] == loader.HYPERPARAMETERS
        and s["projection"] == cap["projection"] == loader.PROJECTION
        and s["loss"]["steps"] == 512,
        "fixed operation-only512-step fit policy",
    )
    need(
        s["profile"] == cap["profile"]
        and s["alias"] == cap["alias"]
        and s["training_strategy"] == "operation_only",
        "exact new operation profile/alias",
    )
    loader.validate_operation_selection(cap)
    need(
        cap["candidate_selection_before_dev_or_sealed"] is False
        and cap["only_head"] == "operation",
        "post-dev operation policy",
    )
    need(
        s["dev_used_for_model_selection"] is True
        and s["optimizer_input_dev"] is False
        and s["optimizer_input_sealed"] is False,
        "honest dev model-selection scope",
    )
    need(
        all(
            s[k] is False
            for k in (
                "model_qualified",
                "browser_qualified",
                "benchmark_qualified",
                "dev_or_sealed_selected",
                "ranking_or_win_claim",
                "OS_access_sealing_claim",
                "full_DSO_or_transitive_closure_qualified",
            )
        ),
        "quality/DSO/access qualification stays false",
    )
    for row in cap["train_inputs"].values():
        pinned(row)
    requests = [
        loader.decode(line)
        for line in blob(cap["train_inputs"]["requests"]["path"]).splitlines()
        if line.strip()
    ]
    labels = [
        loader.decode(line)
        for line in blob(cap["train_inputs"]["labels"]["path"]).splitlines()
        if line.strip()
    ]
    grouped = loader.validate_train_inventory(
        requests, labels, read(cap["train_inputs"]["summary"]["path"])
    )
    supervised = sum(len(row["heads"]) for row in labels)
    need(
        supervised == s["train"]["total_verified_supervised_heads"] == 132
        and s["train"]["supervised_heads"] == 96
        and s["train"]["supervised_target_heads_excluded"] == 36
        and s["train"]["training_strategy"] == "operation_only"
        and s["train"]["optimizer_head_ids"] == ["operation"]
        and s["train"]["unused_heads_skipped"] == 384 - supervised == 252
        and s["train"]["train_states"] == 96
        and s["train"]["ancestors"] == 12
        and s["train"]["families"] == 4,
        "all and only valid supervised train rows",
    )
    ancestors = set()
    chunk_refs = []
    row_map = []
    label_by_id = {r["case_id"]: r for r in labels}
    for entry in cap["chunks"]:
        accepted = read(pinned(entry["acceptance"]))
        loader.validate_commit_header(accepted, entry, cap)
        loader.validate_original_closure(accepted["proof_refs"], lambda: None)
        chunk = read(pinned(entry["chunk"]))
        ancestors.add(entry["ancestor_id"])
        need(
            [c["case_id"] for c in chunk["cases"]]
            == [r["case_id"] for r in grouped[entry["ancestor_id"]]],
            "original per-ancestor cases unchanged",
        )
        for row in accepted["accepted_files"]:
            pinned(row)
        for case, original in zip(chunk["cases"], grouped[entry["ancestor_id"]]):
            need(loader.same_identity(case, original), "independent row case/native identity")
            heads = list(original["request"]["questions"])
            need(
                case["ordered_heads"] == heads and len(case["rows"]) == 4,
                "actual original head order",
            )
            labeled = label_by_id[case["case_id"]]["heads"]
            for h_index, (row, head) in enumerate(zip(case["rows"], heads)):
                criteria = list(original["request"]["questions"][head]["criteria"])
                need(
                    row["head_id"] == head
                    and row["h_index"] == h_index
                    and row["case_id"] == case["case_id"]
                    and row["native_payload_sha256"] == original["native_payload_sha256"]
                    and row["legal_count"] == len(criteria)
                    and row["ordered_criteria_ids"] == criteria,
                    "independent original row mapping",
                )
                if head in labeled:
                    loader.target_vector(labeled[head], criteria)
                    row_map.append(
                        {
                            **{key: case[key] for key in loader.IDENTITY},
                            **row,
                            "supervised_row_index": len(row_map),
                        }
                    )
        chunk_refs.append(entry["acceptance"])
    need(len(ancestors) == 12 and ancestors == set(loader.ANCESTORS), "complete train ancestry")
    need(len(row_map) == 132, "complete independent supervised mapping")
    operation_rows = [r for r in row_map if r["head_id"] == "operation"]
    need(
        len(operation_rows) == len({r["case_id"] for r in operation_rows}) == 96
        and all(r["legal_count"] == 8 for r in operation_rows),
        "96 genuine operation rows",
    )
    mapping_bytes = json.dumps(
        operation_rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    mapping_sha = hashlib.sha256(mapping_bytes).hexdigest()
    need(
        s["train"]["row_mapping_sha256"] == s["row_mapping_sha256"] == mapping_sha,
        "independent actual selected operation row hash",
    )
    prereq = read(pinned(cap["root_np5_prerequisite_ref"]))
    catalog_ref = loader.validate_np_acceptance(prereq, cap, lambda: None)
    catalog = read(pinned(catalog_ref))["after"]
    need(
        s["runtime_catalog"] == catalog_ref
        and set(s["mapped_math_candidates"]) <= set(catalog["mapped_math_candidates"])
        and not s["mapped_gpu_math_candidates_stat_only"],
        "actual NumPy-only mapped candidates subset",
    )
    for row in catalog["critical_files_and_mapped_math_candidates"]:
        p = Path(row["resolved_path"])
        need(
            critical_sha(p) == row["sha256"]
            and p.stat().st_size == row["size_bytes"]
            and all(getattr(p.stat(), k) == v for k, v in row["stat"].items()),
            "accepted critical runtime hash/stat unchanged",
        )
    dref = {**s["delta"], "path": str(OUT / "fit-operation-head" / s["delta"]["path"])}
    delta_path = pinned(dref)
    data = blob(delta_path)
    need(
        dref["shape"] == [255, 1024]
        and dref["dtype"] == "float32"
        and dref["format"] == "npy_no_pickle"
        and data[:8] == b"\x93NUMPY\x01\x00",
        "finite FP32 delta declaration",
    )
    length = struct.unpack("<H", data[8:10])[0]
    header = ast.literal_eval(data[10 : 10 + length].decode("latin1"))
    need(
        header == {"descr": "<f4", "fortran_order": False, "shape": (255, 1024)},
        "exact C-order FP32 no-pickle header",
    )
    raw = data[10 + length :]
    need(len(raw) == 255 * 1024 * 4, "complete delta bytes")
    values = array.array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    need(
        all(math.isfinite(v) for v in values)
        and any(v != 0 for v in values)
        and all(v == 0 for v in values[8 * 1024 :]),
        "finite nonzero delta and unchanged globally unused code rows",
    )
    need(
        0 <= s["elapsed_before_final_write_seconds"] <= command["elapsed_seconds"] <= 120
        and command["elapsed_seconds"] <= proof["root_execution"]["wall_seconds"],
        "original enclosing whole120 completion",
    )
    events = receipt["state"]["last_live_properties"]["kernel_memory_events"]["counters"]
    peak = receipt["state"]["last_live_properties"]["kernel_memory_peak_bytes"]
    result = {
        "schema": "root-accepted-dsl01-operation-trained-readout-v1",
        "status": "accepted",
        "authority": "root",
        "training_eligible": True,
        "independent_actual_review_claimed": False,
        "separate_reviewer_actual_bytes_read": False,
        "original_proof_refs": refs,
        "capability": ref(cap_path),
        "command_spec": ref(spec_path),
        "fit_source": ref(HERE / "fit_head_released_v2.py"),
        "loader_source": ref(HERE / "train_loader_released_v1.py"),
        "audit_source": ref(Path(__file__)),
        "summary": ref(summary_path),
        "source_union": 34,
        "root_exit": 0,
        "owner_completed": True,
        "current_original_scoped_closure_positive": True,
        "model": cap["model"],
        "delta_ref": dref,
        "baseline_ref": cap["baseline"],
        "delta_statistics": {
            "FP32_raw_bytes_sha256": hashlib.sha256(raw).hexdigest(),
            "finite": True,
            "nonzero_elements": sum(v != 0 for v in values),
            "minimum": min(values),
            "maximum": max(values),
            "L2_norm": math.sqrt(math.fsum(v * v for v in values)),
            "rows8_to254_exact_zero": True,
        },
        "train": s["train"],
        "hyperparameters": s["hyperparameters"],
        "minibatch_loss": s["loss"],
        "loss_scope": "Different seeded minibatches; first/last are not a full-train loss comparison or a quality improvement claim",
        "root_seconds": proof["root_execution"]["wall_seconds"],
        "original_child_seconds": command["elapsed_seconds"],
        "elapsed_before_final_write_seconds": s["elapsed_before_final_write_seconds"],
        "owned_unit_peak_bytes": peak,
        "memory_events_diagnostic": events,
        "COMMITTED_chunks": chunk_refs,
        "original_pending_fit_files_unchanged": True,
        "no_optimizer_rerun_by_audit": True,
        "publisher_model_loaded_for_fit": False,
        "train_labels_bytes_checked_by_root_fit_audit": True,
        "dev_or_sealed_selected": False,
        "dev_used_for_model_selection": True,
        "optimizer_input_dev": False,
        "optimizer_input_sealed": False,
        "candidate_selection_before_dev_or_sealed": False,
        "independently_reconstructed_operation_row_mapping_sha256": mapping_sha,
        "browser_or_benchmark_qualified": False,
        "model_quality_improved": False,
        "ranking_or_win_claim": False,
        "scope": "Root checked original source/child/owner/current closure, all96 COMMITTED train records and payload identities, NumPy critical projection and finite nonzero FP32 delta. Operation-only strategy and512 steps chosen after dev exposure. Optimizer uses original train only; sealed remains unrevealed. No evaluation or win claim.",
    }
    return result


def main():
    result = execute()
    target = HERE / "root-accepted-trained-readout-01.json"
    with target.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "train_states": 96,
                "supervised_heads": 96,
                "delta_nonzero_elements": result["delta_statistics"]["nonzero_elements"],
                "root_seconds": result["root_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
