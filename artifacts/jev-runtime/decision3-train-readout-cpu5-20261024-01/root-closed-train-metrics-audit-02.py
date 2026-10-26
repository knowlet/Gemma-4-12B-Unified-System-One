"""Root full-train metrics audit with a stdlib FP64 witness; no optimizer/model."""

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
OUT = OWNER / "actual-20"
CAP_SHA = "51d87342a04e1121e669da018591d62778cbc52cb4599f8763df97317d25a45b"
SPEC_SHA = "e7c9cda4a36d84cb86a6f65daf6a09b07e58522049428121a06affb3ed34e5b3"
LOADER_SHA = "c2b1e8cbeca4d78ee276ebac071c0c15be0a8cf7073fe5526a91ab14a8741763"
FIT_SHA = "2019360a1a6093fae4d2eeb1250b85041c17622053e761dd3553c989fe425820"


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


def npy_data(row):
    p = pinned(row)
    raw = blob(p)
    need(raw[:8] == b"\x93NUMPY\x01\x00", "NPY1 header")
    n = struct.unpack("<H", raw[8:10])[0]
    h = ast.literal_eval(raw[10 : 10 + n].decode("latin1"))
    need(
        h == {"descr": "<f4", "fortran_order": False, "shape": tuple(row["shape"])},
        "exact finite FP32 Corder shape",
    )
    data = raw[10 + n :]
    need(len(data) == math.prod(row["shape"]) * 4, "completeFP32bytes")
    values = array.array("f")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    need(all(math.isfinite(v) for v in values), "finiteFP32values")
    return data, values


def execute():
    terminal = OWNER / "root-execution-train-metrics-20.json"
    need(read(terminal)["exit_code"] == 0, "originalterminal0first")
    loader = source(HERE / "train_loader_released_v1.py", LOADER_SHA, "_root_closed_metrics_loader")
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
    cfg = proof["sealed_config"]
    receipt = proof["owner_receipt"]
    sp = HERE / "root-train-metrics-command-spec-01.json"
    cp = HERE / "root-committed-train-fit-capability-01.json"
    candidate_path = HERE / "root-frozen-delta-candidate-capability-01.json"
    need(
        ref(sp)["sha256"] == SPEC_SHA
        and ref(cp)["sha256"] == CAP_SHA
        and ref(candidate_path)["sha256"]
        == "239165d486bad16963a6e5c94c1f9503fc145ea5b85b114e9fbf94d84eb1fd76",
        "independentlypinned spec/cap/candidate",
    )
    spec, cap, candidate = read(sp), read(cp), read(candidate_path)
    need(
        cfg["selection"] == str(sp)
        and cfg["selection_sha256"] == SPEC_SHA
        and cfg["output"] == str(OUT)
        and receipt["comfy_service_modified"] is False,
        "currentoriginalselection",
    )
    bases = [
        OWNER / "controller.py",
        sp,
        ROOT / "artifacts/jev-runtime/decision3-local-owner-terminal-20261022-01/owner_boundary.py",
        ROOT / "artifacts/jev-runtime/observer-system-services-20261022-01/comfy_observer.py",
        ROOT / "artifacts/jev-runtime/observer-system-services-20261022-01/system-services.json",
    ]
    expected = {r["path"]: r for r in spec["sources"]}
    for p in bases:
        expected[str(p.relative_to(ROOT))] = {**ref(p), "path": str(p.relative_to(ROOT))}
    need(
        len(spec["sources"]) == 35
        and len(expected) == len(cfg["sources"]) == 40
        and expected == {r["path"]: r for r in cfg["sources"]}
        and cfg["binary_stats"] == spec["binary_stats"],
        "exact35plus5sources",
    )
    for row in cfg["sources"]:
        pinned(row)
    for row in spec["binary_stats"]:
        info = Path(row["path"]).stat()
        need(
            all(type(v) is int and getattr(info, k) == v for k, v in row["stat"].items()),
            "actualbinaryoriginalstat",
        )
    command = receipt["state"]["actor_exit"]["command_results"][0]
    started = read(OUT / "command-01-started.json")
    intent = read(OUT / "command-01-intent.json")
    identity = command["observed_identity"]
    child = identity["process"]
    need(
        proof["root_execution"]["profile"] == "root-selected-train-metrics-cpu5-terminal-v1"
        and command["returncode"] == 0
        and command["pid"] == started["pid"] == child["pid"]
        and command["argv"]
        == [v.replace("{OUTPUT}", str(OUT)) for v in spec["commands"][0]["argv"]]
        and command["argv"] == intent["argv"],
        "actualselectedsuccessfulchild",
    )
    need(
        all(
            child[k] == started["observed_identity"]["process"][k]
            for k in ("pid", "start_ticks", "pgid", "sid", "uid", "cgroup")
        )
        and all(
            child[k] == proof["admission"]["actor"][k] for k in ("pgid", "sid", "uid", "cgroup")
        )
        and all(identity[k] == started["observed_identity"][k] for k in ("exe", "argv", "cwd"))
        and identity["cwd"] == str(ROOT)
        and identity["exe"] == intent["resolved_executable"],
        "exactchildcohortidentity",
    )
    result_path = OUT / "train-metrics/metrics-pending.json"
    s = read(result_path)
    need(
        s["schema"] == "dsl01-full-train-zero-frozen-delta-metrics-pending-v1"
        and s["status"] == "PENDING_ROOT_ACCEPTANCE"
        and s["training_eligible"] is False
        and s["selection"] == ref(cp)
        and s["candidate"] == ref(candidate_path)
        and s["process"] == identity
        and s["original_owner_nonce"] == cfg["nonce"],
        "pendingmetricscurrentchild/caps",
    )
    need(
        s["optimizer_steps"] == 0
        and s["array_values_unchanged"] is True
        and all(
            s[k] is False
            for k in (
                "dev_or_sealed_selected",
                "browser_or_official_tasks_selected",
                "raw_arrays_or_probabilities_written",
                "ranking_or_win_claim",
                "model_qualified",
                "browser_qualified",
                "full_DSO_or_transitive_closure_qualified",
            )
        ),
        "readonlytrainonlynoqualityclaim",
    )
    need(
        0 <= s["elapsed_before_final_write_seconds"] <= command["elapsed_seconds"] <= 120
        and command["elapsed_seconds"] <= proof["root_execution"]["wall_seconds"],
        "originalwhole120bounds",
    )
    for row in cap["train_inputs"].values():
        pinned(row)
    requests = [
        loader.decode(v)
        for v in blob(cap["train_inputs"]["requests"]["path"]).splitlines()
        if v.strip()
    ]
    labels = [
        loader.decode(v)
        for v in blob(cap["train_inputs"]["labels"]["path"]).splitlines()
        if v.strip()
    ]
    grouped = loader.validate_train_inventory(
        requests, labels, read(cap["train_inputs"]["summary"]["path"])
    )
    targets = {r["case_id"]: r for r in labels}
    baseline_raw, w0 = npy_data(cap["baseline"])
    delta_raw, delta = npy_data(candidate["delta_ref"])
    need(candidate["delta_ref"] == s["delta"], "samefrozensnapshotdelta")
    xhash, yhash, nhash = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    acc = [0, 0]
    ce = [[], []]
    rows = 0
    for entry in cap["chunks"]:
        acceptance = read(pinned(entry["acceptance"]))
        loader.validate_commit_header(acceptance, entry, cap)
        loader.validate_original_closure(acceptance["proof_refs"], lambda: None)
        chunk = read(pinned(entry["chunk"]))
        parent = Path(entry["chunk"]["path"]).parent
        need(
            [c["case_id"] for c in chunk["cases"]]
            == [r["case_id"] for r in grouped[entry["ancestor_id"]]],
            "allcaseorderunchanged",
        )
        for r in acceptance["accepted_files"]:
            pinned(r)
        for case in chunk["cases"]:
            feature_ref = {**case["features"], "path": str(parent / case["features"]["path"])}
            feature_raw, h = npy_data(feature_ref)
            for i, mapping in enumerate(case["rows"]):
                head = mapping["head_id"]
                label = targets[case["case_id"]]["heads"].get(head)
                if label is None:
                    continue
                n = mapping["legal_count"]
                soft = loader.target_vector(label, mapping["ordered_criteria_ids"])
                xhash.update(feature_raw[i * 4096 : (i + 1) * 4096])
                yhash.update(struct.pack("<255f", *soft))
                nhash.update(struct.pack("<q", n))
                rows += 1
                base = [
                    math.fsum(h[i * 1024 + j] * w0[k * 1024 + j] for j in range(1024))
                    for k in range(n)
                ]
                added = [
                    math.fsum(h[i * 1024 + j] * delta[k * 1024 + j] for j in range(1024))
                    for k in range(n)
                ]
                for which, logits in enumerate((base, [a + b for a, b in zip(base, added)])):
                    chosen = max(range(n), key=logits.__getitem__)
                    acc[which] += int(soft[chosen] > 0)
                    maximum = max(logits)
                    logden = math.log(math.fsum(math.exp(v - maximum) for v in logits))
                    ce[which].append(
                        -math.fsum(
                            soft[k] * (logits[k] - maximum - logden)
                            for k in range(n)
                            if soft[k] > 0
                        )
                    )
    need(
        rows == 132
        and s["train"]["train_states"] == 96
        and s["train"]["ancestors"] == 12
        and s["train"]["families"] == 4
        and s["train"]["supervised_heads"] == rows
        and s["train"]["unused_heads_skipped"] == 252,
        "complete132supervisedtrainrows",
    )
    fp = s["readonly_input_fingerprints"]
    expected_hashes = {
        "features": xhash.hexdigest(),
        "targets": yhash.hexdigest(),
        "counts": nhash.hexdigest(),
        "baseline": hashlib.sha256(baseline_raw).hexdigest(),
        "candidate_delta": hashlib.sha256(delta_raw).hexdigest(),
        "zero_delta": hashlib.sha256(b"\0" * (255 * 1024 * 4)).hexdigest(),
    }
    need(
        {k: row["c_order_bytes_sha256"] for k, row in fp.items()} == expected_hashes,
        "independentlyreconstructedallinputFP32/int64bytes",
    )
    witness = []
    for i, name in enumerate(("zero_delta", "frozen_candidate")):
        metric = s["metrics"][name]
        mean = math.fsum(ce[i]) / rows
        published = metric["full_mean_soft_ce"]
        accuracy = metric["valid_target_argmax"]
        penalty = 0.0 if i == 0 else 0.5 * 0.0001 * math.fsum(v * v for v in delta)
        need(
            accuracy["rows"] == rows
            and accuracy["correct"] == acc[i]
            and accuracy["accuracy"] == acc[i] / rows
            and metric["l2"] == 0.0001
            and abs(mean - published) <= 1e-6 + 1e-5 * abs(published)
            and abs(penalty - metric["half_l2_delta_squared"]) < 1e-12
            and abs(
                (published + metric["half_l2_delta_squared"])
                - metric["full_mean_soft_ce_plus_same_l2"]
            )
            < 1e-6,
            "exactaccuracyandmixed-tolerancefloat64fullobjectivewitness",
        )
        witness.append(
            {
                "profile": name,
                "stdlib_float64_mean_soft_ce": mean,
                "published_FP32_mean_soft_ce": published,
                "absolute_error": abs(mean - published),
                "correct": acc[i],
                "rows": rows,
            }
        )
    prereq = read(pinned(cap["root_np5_prerequisite_ref"]))
    catalog_ref = loader.validate_np_acceptance(prereq, cap, lambda: None)
    catalog = read(pinned(catalog_ref))["after"]
    need(s["runtime_catalog"] == catalog_ref, "sameacceptedruntime")
    for row in catalog["critical_files_and_mapped_math_candidates"]:
        p = Path(row["resolved_path"])
        need(
            critical_sha(p) == row["sha256"]
            and all(getattr(p.stat(), k) == v for k, v in row["stat"].items()),
            "criticalruntimeidentitiesunchanged",
        )
    return {
        "schema": "root-accepted-full-train-readout-metrics-v1",
        "status": "accepted",
        "authority": "root",
        "separate_reviewer_actual_bytes_read": False,
        "original_proof_refs": refs,
        "source_union": 40,
        "command_spec": ref(sp),
        "summary": ref(result_path),
        "candidate_capability": ref(candidate_path),
        "audit_source": ref(Path(__file__)),
        "metrics": s["metrics"],
        "train": s["train"],
        "all_input_byte_fingerprints_reconstructed": expected_hashes,
        "additional_stdlib_float64_witness": witness,
        "FP32_reexecuted_by_root": False,
        "optimizer_steps": 0,
        "model_forwards": 0,
        "root_seconds": proof["root_execution"]["wall_seconds"],
        "original_child_seconds": command["elapsed_seconds"],
        "train_accuracy_gain_percentage_points": 100 * (acc[1] - acc[0]) / rows,
        "full_mean_soft_ce_change": s["metrics"]["frozen_candidate"]["full_mean_soft_ce"]
        - s["metrics"]["zero_delta"]["full_mean_soft_ce"],
        "dev_sealed_browser_or_JevBench_quality_qualified": False,
        "ranking_or_win_claim": False,
        "scope": "Same132trainrows; actual pinned-source FP32 fullobjective with separate root stdlibFP64 witness and allinputbyte hashes. No optimizer/update/model/official evaluation; training-set improvement only.",
    }


def main():
    result = execute()
    with (HERE / "root-accepted-full-train-metrics-02.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "train_accuracy_gain_pp": result["train_accuracy_gain_percentage_points"],
                "full_mean_soft_ce_change": result["full_mean_soft_ce_change"],
                "witness": result["additional_stdlib_float64_witness"],
            }
        )
    )


if __name__ == "__main__":
    main()
