"""HELD root dev24 metrics audit; stdlib FP64 witness, permanently NONTRAINING.

Authoring only: not imported, checked, hashed or executed; no actual/capability/
labels/array bytes were read. Root supplies every concrete selected ref later.
The accepted root result never grants training eligibility or quality promotion.
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
from pathlib import Path

SOURCE_ONLY = True
ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
HERE = ROOT / "artifacts/jev-runtime/decision3-frozen-heldout-cpu5-20261024-01"
OWNER = ROOT / "artifacts/jev-runtime/llama-cpp-owner-20261024-09"
DEV = ROOT / "artifacts/jev-runtime/browser-decision-dsl-20261023-01/corpus/data/dev"
DSL = DEV.parents[2]
HEADS = ("operation", "click_target", "type_text_target", "select_target")
IDENTITY = (
    "case_id",
    "dsl_version",
    "family_id",
    "ancestor_id",
    "request_sha256",
    "native_payload_sha256",
)
CANDIDATE_SHA = "239165d486bad16963a6e5c94c1f9503fc145ea5b85b114e9fbf94d84eb1fd76"
DELTA_SHA = "812256139b35429dc55f4f178af8c177d2a31c023414e668e9b39cdc217d14be"
READS = {}


def need(value, message):
    if not value:
        raise ValueError(message)


def released():
    need(SOURCE_ONLY is False, "SOURCE_ONLY: no arguments, imports or actual reads authorized")


def local(path):
    path = Path(path)
    path = path if path.is_absolute() else ROOT / path
    need(
        path.is_relative_to(ROOT)
        and ".." not in path.parts
        and not any(p.is_symlink() for p in (path, *path.parents)),
        "ordinary repository path required",
    )
    need(
        not path.is_relative_to(DSL)
        or path in {DEV / name for name in ("requests.jsonl", "labels.jsonl", "summary.json")},
        "only original dev exports; no train/sealed/authoring",
    )
    need("train-chunk" not in path.parts, "no train feature access")
    return path


def fetch(path, sha=None, size=None):
    path = local(path)
    need(stat.S_ISREG(path.lstat().st_mode), "regular selected input required")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        need(0 < before.st_size <= 16 * 1024**2, "bounded metadata/private metric array")
        raw = stream.read(16 * 1024**2 + 1)
        after = os.fstat(stream.fileno())
    actual = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
    need(
        len(raw) == before.st_size
        and (sha is None or actual["sha256"] == sha)
        and (size is None or len(raw) == size)
        and all(
            getattr(before, key) == getattr(after, key) == getattr(path.stat(), key)
            for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        ),
        "exact stable selected bytes required",
    )
    need(str(path) not in READS or READS[str(path)] == actual, "input changed across audit reads")
    READS[str(path)] = actual
    return raw, actual


def pinned(record):
    return fetch(record["path"], record["sha256"], record.get("size_bytes", record.get("size")))[0]


def decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate metadata key")
            result[key] = value
        return result

    return json.loads(
        raw, object_pairs_hook=pairs, parse_constant=lambda _s: need(False, "nonfinite metadata")
    )


def normalized(record):
    return {
        "path": str(local(record["path"])),
        "sha256": record["sha256"],
        "size_bytes": record.get("size_bytes", record.get("size")),
    }


def npy(record, shape, dtype="<f4"):
    raw = pinned(record)
    need(raw[:8] == b"\x93NUMPY\x01\x00", "private NPYv1 required")
    width = struct.unpack("<H", raw[8:10])[0]
    need(
        0 < width <= 4096
        and ast.literal_eval(raw[10 : 10 + width].decode("latin1"))
        == {"descr": dtype, "fortran_order": False, "shape": tuple(shape)},
        "exact bounded plain Corder NPY header",
    )
    data = raw[10 + width :]
    need(len(data) == math.prod(shape) * (8 if dtype == "<i8" else 4), "complete array bytes")
    values = array.array("q" if dtype == "<i8" else "f")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    need(all(math.isfinite(value) for value in values), "finite private values required")
    need(
        "FP32_bytes_sha256" not in record
        or hashlib.sha256(data).hexdigest() == record["FP32_bytes_sha256"],
        "actual FP32 value bytes differ",
    )
    return data, values


def load_helper(path, sha):
    need(
        local(path).parent == HERE and Path(path).name == "root-closed-dev-chunk-audit-02.py",
        "existing corrected02 stdlib metadata-closure helper required",
    )
    raw, record = fetch(path, sha)
    module = types.ModuleType("_root_dev_metrics_closed_metadata_helper")
    module.__file__ = record["path"]
    sys.modules[module.__name__] = module
    exec(compile(raw, record["path"], "exec", dont_inherit=True), vars(module))
    return module


def target_vector(label, criteria):
    need(
        set(label) == {"criteria_ids", "valid_ids", "valid_positions", "soft_target"}
        and label["criteria_ids"] == criteria,
        "exact dev label/criteria join",
    )
    valid = label["valid_ids"]
    need(
        type(valid) is list
        and valid
        and len(set(valid)) == len(valid)
        and set(valid) <= set(criteria),
        "actual multiple valid targets required",
    )
    positions = [i for i, key in enumerate(criteria) if key in valid]
    need(
        valid == [criteria[i] for i in positions] and label["valid_positions"] == positions,
        "actual ordered valid slots required",
    )
    expected = [1 / len(valid) if key in valid else 0.0 for key in criteria]
    need(
        len(label["soft_target"]) == len(criteria)
        and all(
            type(value) in (int, float)
            and math.isfinite(value)
            and abs(value - expected[i]) <= 1e-12
            for i, value in enumerate(label["soft_target"])
        ),
        "true uniform multi-valid target mass",
    )
    return expected + [0.0] * (255 - len(criteria))


def rederive(cap, helper):
    exports = cap["dev_inputs"]
    need(set(exports) == {"requests", "labels", "summary"}, "only three dev exports")
    for key, record in exports.items():
        need(
            local(record["path"]) == DEV / (key + (".json" if key == "summary" else ".jsonl")),
            "original dev export identity",
        )
    requests = [decode(line) for line in pinned(exports["requests"]).splitlines() if line.strip()]
    labels = [decode(line) for line in pinned(exports["labels"]).splitlines() if line.strip()]
    summary = decode(pinned(exports["summary"]))
    need(
        len(requests) == len(labels) == summary["case_count"] == 24
        and summary["split"] == "dev"
        and [r["case_id"] for r in requests] == [r["case_id"] for r in labels]
        and len({r["case_id"] for r in requests}) == 24,
        "complete original dev24 population/order",
    )
    need(len(cap["closed_dev_acceptances"]) == 3, "all three closed dev chunks required")
    cases, baseline = {}, None
    for record in cap["closed_dev_acceptances"]:
        accepted = decode(pinned(record))
        need(
            accepted["schema"] == "root-closed-dsl01-dev-chunk-acceptance-v1"
            and accepted["status"] == "ROOT_ACCEPTED_DEV_FEATURES_NONTRAINING"
            and accepted["split"] == "dev"
            and accepted["training_eligible"] is False
            and len(accepted["cases"]) == 8
            and accepted["frozen_candidate_ref"] == cap["candidate_capability"]
            and accepted["model"] == cap["model"]
            and accepted["dev_inputs"] == {key: exports[key] for key in ("requests", "summary")},
            "exact closed dev acceptance/model/candidate joins",
        )
        helper.validate_original_closure(accepted["proof_refs"], lambda: None)
        raw, _ = npy(accepted["baseline_ref"], (255, 1024))
        need(baseline is None or raw == baseline, "same original W0 across dev chunks")
        baseline = raw
        for case in accepted["cases"]:
            need(case["case_id"] not in cases, "duplicate accepted dev case")
            cases[case["case_id"]] = case
    need(set(cases) == {r["case_id"] for r in requests}, "all24 accepted cases exactly once")
    features, targets, counts, case_map, row_map = [], [], [], [], []

    def canonical(value):
        return json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()

    for request, label in zip(requests, labels):
        need(
            set(request) == set(IDENTITY) | {"request"}
            and set(label) == set(IDENTITY) | {"heads"}
            and all(request[key] == label[key] for key in IDENTITY),
            "original request/label identity",
        )
        case, payload = cases[request["case_id"]], request["request"]
        need(
            all(case[key] == request[key] for key in IDENTITY)
            and hashlib.sha256(canonical(payload)).hexdigest() == request["request_sha256"]
            and hashlib.sha256(
                canonical({key: payload[key] for key in ("state", "questions")})
            ).hexdigest()
            == request["native_payload_sha256"],
            "accepted actual projection identity",
        )
        questions, offered = payload["questions"], list(payload["questions"])
        need(
            len(offered) == 4
            and set(offered) == set(HEADS)
            and offered[0] == "operation"
            and offered == case["ordered_heads"]
            and "operation" in label["heads"]
            and set(label["heads"]) <= set(offered),
            "actual head insertion order/supervised subset",
        )
        raw, _ = npy(case["features_ref"], (4, 1024))
        supervised = [head for head in offered if head in label["heads"]]
        case_map.append(
            {
                "case_id": request["case_id"],
                "offered_heads": offered,
                "supervised_heads": supervised,
            }
        )
        for slot, head in enumerate(offered):
            if head not in label["heads"]:
                continue
            criteria = list(questions[head]["criteria"])
            target = target_vector(label["heads"][head], criteria)
            features.append(raw[slot * 4096 : (slot + 1) * 4096])
            targets.extend(target)
            counts.append(len(criteria))
            row_map.append(
                {
                    "row_index": len(row_map),
                    "case_id": request["case_id"],
                    "head_id": head,
                    "ordered_criteria_ids": criteria,
                }
            )
    need(
        len(row_map) == 33 and cap["cases"] == case_map and cap["rows"] == row_map,
        "complete33-supervised/24-case exact private mapping",
    )
    return {
        "features": b"".join(features),
        "counts": struct.pack("<33q", *counts),
        "soft_targets": struct.pack("<8415f", *targets),
        "W0": baseline,
    }


def witness(cap, x, w0, delta, counts, targets, candidate):
    hits, losses = [], []
    for i, count in enumerate(counts):
        need(
            count == len(cap["rows"][i]["ordered_criteria_ids"]) and 1 <= count <= 255,
            "actual legal count/slot mapping",
        )
        soft = targets[i * 255 : (i + 1) * 255]
        need(
            all(0 <= value <= 1 for value in soft)
            and abs(math.fsum(soft) - 1) <= 1e-6
            and all(value == 0 for value in soft[count:]),
            "normalized legal soft targets",
        )
        h = x[i * 1024 : (i + 1) * 1024]
        base = [math.fsum(h[j] * w0[k * 1024 + j] for j in range(1024)) for k in range(count)]
        added = (
            [math.fsum(h[j] * delta[k * 1024 + j] for j in range(1024)) for k in range(count)]
            if candidate
            else [0.0] * count
        )
        logits = [a + b for a, b in zip(base, added)]
        chosen = max(range(count), key=logits.__getitem__)
        hits.append(soft[chosen] > 0)
        maximum = max(logits)
        logden = math.log(math.fsum(math.exp(value - maximum) for value in logits))
        losses.append(
            -math.fsum(
                soft[k] * (logits[k] - maximum - logden) for k in range(count) if soft[k] > 0
            )
        )
    return hits, losses


def compare(cap, metric, hits, losses):
    n = len(hits)
    need(
        metric["supervised_rows"] == n
        and metric["valid_target_argmax_correct"] == sum(hits)
        and metric["valid_target_argmax_accuracy"] == sum(hits) / n
        and metric["l2_applied"] is False
        and metric["optimizer_steps"] == 0
        and abs(metric["full_mean_soft_ce"] - math.fsum(losses) / n) <= 1e-5,
        "exact full-dev hit/FP64 CE tolerance1e-5",
    )
    need(set(metric["by_head"]) == set(HEADS), "all original head groups required")
    for head in HEADS:
        ids = [i for i, row in enumerate(cap["rows"]) if row["head_id"] == head]
        row, total = metric["by_head"][head], len(ids)
        need(
            row["rows"] == total and row["correct"] == sum(hits[i] for i in ids),
            "per-head complete denominator/hits",
        )
        need(
            (row["accuracy"] is None and row["mean_soft_ce"] is None)
            if not total
            else (
                row["accuracy"] == sum(hits[i] for i in ids) / total
                and abs(row["mean_soft_ce"] - math.fsum(losses[i] for i in ids) / total) <= 1e-5
            ),
            "per-head CE/hit witness",
        )
    correct_cases = sum(
        all(hits[i] for i, row in enumerate(cap["rows"]) if row["case_id"] == case["case_id"])
        for case in cap["cases"]
    )
    need(
        metric["all_supervised_heads_correct"]
        == {"cases": 24, "correct": correct_cases, "accuracy": correct_cases / 24},
        "all-heads complete24-case denominator",
    )
    return {
        "stdlib_FP64_full_mean_soft_ce": math.fsum(losses) / n,
        "published_FP32_full_mean_soft_ce": metric["full_mean_soft_ce"],
        "CE_absolute_tolerance": 1e-5,
        "correct": sum(hits),
        "rows": n,
        "all_heads_correct_cases": correct_cases,
    }


def execute(args):
    released()
    _, source_ref = fetch(Path(__file__).absolute(), args.source_sha256)
    helper = load_helper(args.helper, args.helper_sha256)
    terminal_raw, terminal_ref = fetch(args.terminal, args.terminal_sha256)
    spec_raw, spec_ref = fetch(args.spec, args.spec_sha256)
    cap_raw, cap_ref = fetch(args.cap, args.cap_sha256)
    summary_raw, summary_ref = fetch(args.output, args.output_sha256)
    terminal, spec, cap, summary = map(decode, (terminal_raw, spec_raw, cap_raw, summary_raw))
    need(
        terminal["exit_code"] == 0 and len(spec["commands"]) == 1,
        "originalterminal0/single selected child first",
    )
    out = local(terminal["argv"][terminal["argv"].index("--output") + 1])
    proof_refs = {
        "root_execution": terminal_ref,
        **{
            key: fetch(out / name)[1]
            for key, name in {
                "owner_receipt": "receipt.json",
                "closure": "external-boundary-closure.json",
                "admission": "admission.json",
                "sealed_config": "sealed-config.json",
            }.items()
        },
    }
    proof = helper.validate_original_closure(proof_refs, lambda: None)
    cfg, receipt = proof["sealed_config"], proof["owner_receipt"]
    need(
        out.parent == OWNER
        and cfg["output"] == str(out)
        and local(cfg["selection"]) == local(args.spec)
        and cfg["selection_sha256"] == args.spec_sha256
        and local(args.output) == out / "dev-metrics/metrics-nontraining-pending.json"
        and receipt["comfy_service_modified"] is False,
        "current exact owner/spec/metrics output joins",
    )
    bases = [
        OWNER / "controller.py",
        local(args.spec),
        ROOT / "artifacts/jev-runtime/decision3-local-owner-terminal-20261022-01/owner_boundary.py",
        ROOT / "artifacts/jev-runtime/observer-system-services-20261022-01/comfy_observer.py",
        ROOT / "artifacts/jev-runtime/observer-system-services-20261022-01/system-services.json",
    ]
    expected = {str(local(row["path"])): normalized(row) for row in spec["sources"]}
    for path in bases:
        expected[str(path)] = fetch(path)[1]
    need(
        len(expected) == len(spec["sources"]) + 5 == len(cfg["sources"])
        and expected == {str(local(row["path"])): normalized(row) for row in cfg["sources"]}
        and cfg["binary_stats"] == spec["binary_stats"],
        "exact selected source union plus5 and binaries",
    )
    # The corrected dev helper is used for metadata closure only. Its payload
    # deny rules intentionally do not admit metric bundle/delta/dev-label IO.
    for row in cfg["sources"]:
        pinned(row)
    for row in spec["binary_stats"]:
        binary = Path(row["path"])
        need(
            binary.is_absolute()
            and ".." not in binary.parts
            and not any(item.is_symlink() for item in (binary, *binary.parents)),
            "ordinary absolute selected binary stat path",
        )
        info = binary.stat()
        need(
            all(
                type(value) is int and getattr(info, key) == value
                for key, value in row["stat"].items()
            ),
            "original binary stat-only identity",
        )
    command = receipt["state"]["actor_exit"]["command_results"][0]
    started, intent = (
        decode(fetch(out / name)[0])
        for name in ("command-01-started.json", "command-01-intent.json")
    )
    identity, observed = command["observed_identity"], started["observed_identity"]
    argv = [value.replace("{OUTPUT}", str(out)) for value in spec["commands"][0]["argv"]]
    need(
        command["returncode"] == 0
        and command["argv"] == identity["argv"] == observed["argv"] == intent["argv"] == argv
        and command["pid"]
        == started["pid"]
        == identity["process"]["pid"]
        != proof["admission"]["actor"]["pid"]
        and identity == observed
        and identity["cwd"] == intent["cwd"] == str(ROOT)
        and identity["exe"] == intent["resolved_executable"]
        and intent["shell"] is False
        and all(
            identity["process"][key] == proof["admission"]["actor"][key]
            for key in ("pgid", "sid", "uid", "cgroup")
        ),
        "exact actual command-started child/cohort/cwd/exe",
    )
    need(
        summary["schema"] == "dev24-cached-head-metrics-nontraining-v1"
        and summary["status"] == "NONTRAINING_PENDING_ROOT_ACCEPTANCE"
        and summary["nontraining"] is True
        and summary["training_eligible"] is False
        and summary["split"] == "dev"
        and summary["complete_cases"] == 24
        and summary["supervised_rows"] == 33
        and summary["unused_heads_skipped"] == 63
        and summary["process"] == identity
        and summary["original_owner_nonce"] == cfg["nonce"]
        and normalized(summary["capability"]) == cap_ref
        and summary["optimizer_steps"] == 0
        and all(
            summary[key] is False
            for key in (
                "l2_applied",
                "raw_arrays_or_probabilities_written",
                "browser_or_official_score",
                "full_DSO_or_transitive_closure_qualified",
            )
        ),
        "actual readonly complete dev-only summary",
    )
    need(
        0 <= summary["elapsed_before_terminal_write_seconds"] < command["elapsed_seconds"] < 120
        and command["elapsed_seconds"] <= terminal["wall_seconds"],
        "whole120 includes final retention",
    )
    need(
        cap["schema"] == "root-selected-frozen-dev24-metrics-v1"
        and cap["status"] == "ROOT_SELECTED_FROZEN_DEV24_METRICS_ONLY"
        and cap["split"] == "dev"
        and cap["nontraining"] is True
        and cap["training_eligible"] is False
        and cap["frozen_candidate_steps"] == 32
        and cap["candidate_capability"]["sha256"] == CANDIDATE_SHA
        and cap["delta_ref"]["sha256"] == DELTA_SHA,
        "same root-frozen32 candidate/dev-only capability",
    )
    candidate = decode(pinned(cap["candidate_capability"]))
    need(
        candidate["status"] == "ROOT_ACCEPTED_TRAIN_ONLY_READOUT"
        and candidate["model"] == cap["model"]
        and candidate["delta_ref"] == cap["delta_ref"]
        and candidate["frozen_before_dev_sealed_benchmark_or_browser"] is True
        and candidate["W0_merge_allowed"] is False
        and candidate["baseline_fp32_bytes_sha256"] == cap["arrays"]["W0"]["FP32_bytes_sha256"]
        and summary["candidate_capability"] == cap["candidate_capability"],
        "actual frozen artifact/model/W0 identity",
    )
    need(
        set(cap["arrays"]) == {"features", "counts", "soft_targets", "W0"}
        and len(cap["cases"]) == 24
        and len(cap["rows"]) == 33,
        "all four complete private arrays/mapping",
    )
    shapes = {"features": (33, 1024), "counts": (33,), "soft_targets": (33, 255), "W0": (255, 1024)}
    loaded = {
        key: npy(record, shapes[key], "<i8" if key == "counts" else "<f4")
        for key, record in cap["arrays"].items()
    }
    delta_raw, delta = npy(cap["delta_ref"], (255, 1024))
    originals = rederive(cap, helper)
    need(
        all(originals[key] == loaded[key][0] for key in originals),
        "all33 actual accepted head rows/targets/counts/W0 bytes independently rederived",
    )
    fingerprints = {key: hashlib.sha256(raw).hexdigest() for key, (raw, _values) in loaded.items()}
    fingerprints.update(
        delta=hashlib.sha256(delta_raw).hexdigest(),
        zero_delta=hashlib.sha256(b"\0" * (255 * 1024 * 4)).hexdigest(),
    )
    need(
        summary["readonly_value_hashes"] == fingerprints,
        "actual unchanged readonly array fingerprints",
    )
    comparisons = {}
    need(
        set(summary["results"]) == {"zero_delta", "frozen_candidate"},
        "both exact same-denominator profiles",
    )
    for name in ("zero_delta", "frozen_candidate"):
        hits, losses = witness(
            cap,
            loaded["features"][1],
            loaded["W0"][1],
            delta,
            loaded["counts"][1],
            loaded["soft_targets"][1],
            name == "frozen_candidate",
        )
        comparisons[name] = compare(cap, summary["results"][name], hits, losses)
    need(
        normalized(summary["runtime_catalog"])
        == normalized(cap["runtime"]["numpy_runtime_catalog"]),
        "actual root-selected runtime catalog",
    )
    for record in cap["runtime"].values():
        pinned(record)
    for record in list(READS.values()):
        pinned(record)
    return {
        "schema": "root-accepted-complete-dev24-cached-metrics-nontraining-v1",
        "status": "accepted",
        "authority": "root",
        "nontraining": True,
        "training_eligible": False,
        "separate_reviewer_actual_bytes_read": False,
        "independent_actual_review_claimed": False,
        "original_proof_refs": proof_refs,
        "command_spec": spec_ref,
        "capability": cap_ref,
        "summary": summary_ref,
        "source_union": len(cfg["sources"]),
        "candidate_capability": cap["candidate_capability"],
        "delta_ref": cap["delta_ref"],
        "model": cap["model"],
        "complete_cases": 24,
        "supervised_rows": 33,
        "unused_heads_skipped": 63,
        "metrics": summary["results"],
        "additional_stdlib_FP64_witness": comparisons,
        "all_bundle_values_independently_rederived": True,
        "readonly_value_fingerprints": fingerprints,
        "optimizer_steps": 0,
        "model_forwards": 0,
        "browser_or_official_score": False,
        "promotion_or_quality_win_claim": False,
        "audit_source": source_ref,
        "argmax_comparison_scope": "FP64 uses first ordered argmax per row; actual FP32 publishes aggregate hits only, not independently retained per-row choices",
        "scope": "Complete frozen dev24 cached-head metrics only; root actual read with source-only peer review. Permanent NONTRAINING; no promotion, browser/official score or independent peer actual-read claim.",
    }


def main(argv=None):
    released()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ("spec", "terminal", "cap", "output", "helper"):
        parser.add_argument("--" + key, type=Path, required=True)
        parser.add_argument("--" + key + "-sha256", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--acceptance-output", type=Path, required=True)
    args = parser.parse_args(argv)
    need(
        all(
            re.fullmatch(r"[0-9a-f]{64}", getattr(args, key + "_sha256"))
            for key in ("spec", "terminal", "cap", "output", "helper", "source")
        ),
        "exact root-selected input/source hashes",
    )
    result = execute(args)
    target = local(args.acceptance_output)
    need(target.parent == HERE, "fresh local NONTRAINING acceptance output required")
    with target.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(HERE, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    print(
        json.dumps(
            {"status": "accepted", "nontraining": True, "complete_cases": 24, "supervised_rows": 33}
        )
    )


if __name__ == "__main__":
    main()
