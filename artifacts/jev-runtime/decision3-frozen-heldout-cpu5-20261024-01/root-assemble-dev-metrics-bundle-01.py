"""Root-only stdlib assembly of accepted complete dev24 metric inputs; no model/fit.

Exact root-selected closed acceptances and original dev labels are required.
Every derived file remains private and permanently nontraining.
"""

from __future__ import annotations

import argparse
import array
import ast
import hashlib
import json
import math
import os
import struct
import sys
from pathlib import Path

ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
HERE = ROOT / "artifacts/jev-runtime/decision3-frozen-heldout-cpu5-20261024-01"
DSL = ROOT / "artifacts/jev-runtime/browser-decision-dsl-20261023-01"
IDENTITY = (
    "case_id",
    "dsl_version",
    "family_id",
    "ancestor_id",
    "request_sha256",
    "native_payload_sha256",
)
HEADS = {"operation", "click_target", "type_text_target", "select_target"}
CANDIDATE_SHA = "239165d486bad16963a6e5c94c1f9503fc145ea5b85b114e9fbf94d84eb1fd76"
READS = {}


def need(value, message):
    if not value:
        raise ValueError(message)


def ordinary(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    need(
        path.is_relative_to(ROOT)
        and ".." not in path.parts
        and not any(p.is_symlink() for p in (path, *path.parents)),
        "ordinary repository file",
    )
    need(
        not path.is_relative_to(DSL)
        or path
        in {
            DSL / "corpus/data/dev" / n for n in ("requests.jsonl", "labels.jsonl", "summary.json")
        },
        "only exact dev exports; no train/sealed/authoring",
    )
    return path


def read_ref(record):
    path = ordinary(record["path"])
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        need(0 < before.st_size <= 4 * 1024**2, "bounded metadata/private array")
        raw = stream.read(4 * 1024**2 + 1)
        after = os.fstat(stream.fileno())
    need(
        len(raw) == record["size_bytes"] == before.st_size
        and hashlib.sha256(raw).hexdigest() == record["sha256"]
        and all(
            getattr(before, k) == getattr(after, k) == getattr(path.stat(), k)
            for k in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        ),
        "exact stable selected bytes",
    )
    actual = {"path": str(path), "sha256": record["sha256"], "size_bytes": len(raw)}
    need(str(path) not in READS or READS[str(path)] == actual, "changed across assembly reads")
    READS[str(path)] = actual
    return raw


def decode(raw):
    def pairs(items):
        d = {}
        for k, v in items:
            need(k not in d, "duplicate metadata key")
            d[k] = v
        return d

    return json.loads(
        raw, object_pairs_hook=pairs, parse_constant=lambda _s: need(False, "nonfinite metadata")
    )


def file_ref(path):
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def npy_data(record, shape):
    raw = read_ref(record)
    need(raw[:8] == b"\x93NUMPY\x01\x00", "NPYv1 only")
    n = struct.unpack("<H", raw[8:10])[0]
    need(0 < n <= 4096, "bounded literal NPY header")
    need(
        ast.literal_eval(raw[10 : 10 + n].decode("latin1"))
        == {"descr": "<f4", "fortran_order": False, "shape": tuple(shape)},
        "exact FP32 Corder no pickle",
    )
    data = raw[10 + n :]
    need(len(data) == math.prod(shape) * 4, "complete array bytes")
    values = array.array("f")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    need(all(math.isfinite(v) for v in values), "finite actual features/W0")
    return data


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def save_npy(path, data, shape, dtype):
    text = repr({"descr": dtype, "fortran_order": False, "shape": tuple(shape)}).encode("ascii")
    header = text + b" " * (64 - (10 + len(text) + 1) % 64) + b"\n"
    with path.open("xb") as stream:
        stream.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + data)
        stream.flush()
        os.fsync(stream.fileno())
    return {
        **file_ref(path),
        "format": "npy_no_pickle",
        "dtype": {"<f4": "float32", "<i8": "int64"}[dtype],
        "shape": list(shape),
        "raw_bytes_sha256": hashlib.sha256(data).hexdigest(),
    }


def target_vector(label, criteria):
    need(
        set(label) == {"criteria_ids", "valid_ids", "valid_positions", "soft_target"}
        and label["criteria_ids"] == criteria,
        "exact label/criteria mapping",
    )
    valid = label["valid_ids"]
    need(
        type(valid) is list
        and valid
        and len(set(valid)) == len(valid)
        and set(valid) <= set(criteria),
        "all valid offered targets",
    )
    positions = [i for i, k in enumerate(criteria) if k in valid]
    need(
        valid == [criteria[i] for i in positions]
        and label["valid_positions"] == positions
        and all(type(i) is int for i in positions),
        "actual valid slot positions",
    )
    expected = [1.0 / len(valid) if k in valid else 0.0 for k in criteria]
    soft = label["soft_target"]
    need(
        type(soft) is list
        and len(soft) == len(criteria)
        and all(
            type(v) in (int, float) and math.isfinite(v) and abs(v - e) <= 1e-12
            for v, e in zip(soft, expected)
        ),
        "complete uniform multi-valid soft targets",
    )
    return soft + [0.0] * (255 - len(criteria))


def execute(selection_ref):
    selection = decode(read_ref(selection_ref))
    need(
        selection["schema"] == "root-selected-accepted-dev24-metric-assembly-v1"
        and selection["training_eligible"] is False
        and len(selection["chunk_acceptances"]) == 3,
        "exact root-selected three accepted chunks",
    )
    candidate = decode(read_ref(selection["frozen_candidate_ref"]))
    need(
        selection["frozen_candidate_ref"]["sha256"] == CANDIDATE_SHA
        and candidate["status"] == "ROOT_ACCEPTED_TRAIN_ONLY_READOUT"
        and candidate["frozen_before_dev_sealed_benchmark_or_browser"] is True,
        "same frozen candidate before dev exposure",
    )
    exports = selection["dev_inputs"]
    need(set(exports) == {"requests", "labels", "summary"}, "dedicated dev metric role inputs")
    for key in exports:
        need(
            ordinary(exports[key]["path"])
            == DSL
            / "corpus/data/dev"
            / (
                {"requests": "requests.jsonl", "labels": "labels.jsonl", "summary": "summary.json"}[
                    key
                ]
            ),
            "fixed original dev export",
        )
    requests = [decode(line) for line in read_ref(exports["requests"]).splitlines() if line.strip()]
    labels = [decode(line) for line in read_ref(exports["labels"]).splitlines() if line.strip()]
    summary = decode(read_ref(exports["summary"]))
    need(
        summary["split"] == "dev"
        and summary["case_count"] == 24
        and summary["family_ids"] == ["expense_claim"]
        and len(requests) == len(labels) == 24
        and len({r["case_id"] for r in requests}) == 24
        and [r["case_id"] for r in requests] == [label_row["case_id"] for label_row in labels],
        "complete dev24 original population/order",
    )
    accepted = {}
    baseline = None
    proofs = []
    for record in selection["chunk_acceptances"]:
        a = decode(read_ref(record))
        need(
            a["status"] == "ROOT_ACCEPTED_DEV_FEATURES_NONTRAINING"
            and a["split"] == "dev"
            and a["training_eligible"] is False
            and a["dev_inputs"] == {k: exports[k] for k in ("requests", "summary")}
            and len(a["cases"]) == 8,
            "closed accepted dev chunk only",
        )
        b = npy_data(a["baseline_ref"], (255, 1024))
        need(
            hashlib.sha256(b).hexdigest() == candidate["baseline_fp32_bytes_sha256"]
            and (baseline is None or baseline == b),
            "same original W0 byte identity across all three chunks",
        )
        baseline = b
        proofs.append(record)
        for case in a["cases"]:
            need(case["case_id"] not in accepted, "duplicate accepted case")
            accepted[case["case_id"]] = case
    need(
        set(accepted) == {r["case_id"] for r in requests},
        "all24 exactly once; no partial metric bundle",
    )
    features = []
    counts = []
    targets = []
    case_map = []
    row_map = []
    for request, label in zip(requests, labels):
        need(
            set(request) == set(IDENTITY) | {"request"}
            and set(label) == set(IDENTITY) | {"heads"}
            and all(request[k] == label[k] for k in IDENTITY),
            "exact request/label identity",
        )
        need(
            request["family_id"] == "expense_claim"
            and request["ancestor_id"] in summary["ancestor_ids"],
            "dev family/ancestor only",
        )
        payload = request["request"]
        questions = payload["questions"]
        case = accepted[request["case_id"]]
        need(
            all(case[k] == request[k] for k in IDENTITY)
            and hashlib.sha256(canonical(payload)).hexdigest() == request["request_sha256"]
            and hashlib.sha256(
                canonical({k: payload[k] for k in ("state", "questions")})
            ).hexdigest()
            == request["native_payload_sha256"],
            "accepted actual native projection joins original dev request",
        )
        offered = list(questions)
        need(
            len(offered) == 4
            and set(offered) == HEADS
            and offered[0] == "operation"
            and case["ordered_heads"] == offered,
            "complete actual offered head order",
        )
        need(
            type(label["heads"]) is dict
            and "operation" in label["heads"]
            and set(label["heads"]) <= set(offered),
            "legal supervised head subset",
        )
        raw = npy_data(case["features_ref"], (4, 1024))
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
            need(
                questions[head]["type"] == "choice"
                and len(criteria) == (8 if head == "operation" else 4),
                "exact original offered criteria",
            )
            target = target_vector(label["heads"][head], criteria)
            features.append(raw[slot * 1024 * 4 : (slot + 1) * 1024 * 4])
            counts.append(len(criteria))
            targets.extend(target)
            row_map.append(
                {
                    "row_index": len(row_map),
                    "case_id": request["case_id"],
                    "head_id": head,
                    "ordered_criteria_ids": criteria,
                }
            )
    bundle = HERE / "dev-metrics-bundle-01"
    need(not bundle.exists(), "fresh private bundle only")
    bundle.mkdir(mode=0o700)
    (bundle / ".gitignore").write_text("*\n")
    n = len(row_map)
    arrays = {
        "features": save_npy(bundle / "features.npy", b"".join(features), (n, 1024), "<f4"),
        "counts": save_npy(
            bundle / "counts.npy", struct.pack("<" + str(n) + "q", *counts), (n,), "<i8"
        ),
        "soft_targets": save_npy(
            bundle / "soft_targets.npy",
            struct.pack("<" + str(len(targets)) + "f", *targets),
            (n, 255),
            "<f4",
        ),
        "W0": save_npy(bundle / "W0.npy", baseline, (255, 1024), "<f4"),
    }
    arrays["W0"]["FP32_bytes_sha256"] = candidate["baseline_fp32_bytes_sha256"]
    for original in READS.values():
        read_ref(original)
    np5 = decode(read_ref(selection["root_np5_prerequisite_ref"]))
    runtime = {
        "root_np5_prerequisite_ref": selection["root_np5_prerequisite_ref"],
        "numpy_runtime_catalog": np5["numpy_runtime_catalog"],
    }
    cap = {
        "schema": "root-selected-frozen-dev24-metrics-v1",
        "status": "ROOT_SELECTED_FROZEN_DEV24_METRICS_ONLY",
        "split": "dev",
        "nontraining": True,
        "training_eligible": False,
        "frozen_candidate_steps": 32,
        "bundle_root": str(bundle),
        "arrays": arrays,
        "runtime": runtime,
        "candidate_capability": selection["frozen_candidate_ref"],
        "delta_ref": candidate["delta_ref"],
        "model": candidate["model"],
        "cases": case_map,
        "rows": row_map,
        "root_assembly_selection": selection_ref,
        "closed_dev_acceptances": proofs,
        "dev_inputs": exports,
        "sealed_reveal_authorized": False,
        "scope": "Complete same frozen candidate/dev24 cached-head only. Original actual native baseline parity checked in root chunk acceptance. No browser/official benchmark/fit/OS-sealing claim.",
    }
    save(HERE / "root-selected-dev24-metrics-capability-01.json", cap)
    save(
        bundle / "manifest.json",
        {
            "status": "PERMANENT_DEV_NONTRAINING",
            "arrays": arrays,
            "complete_cases": 24,
            "supervised_rows": n,
            "unused_heads_skipped": 96 - n,
            "training_eligible": False,
        },
    )
    for p in (bundle, HERE):
        fd = os.open(p, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return {
        "status": "ASSEMBLED_COMPLETE_DEV24_NONTRAINING",
        "complete_cases": 24,
        "supervised_rows": n,
        "unused_heads_skipped": 96 - n,
        "capability": file_ref(HERE / "root-selected-dev24-metrics-capability-01.json"),
        "source": file_ref(Path(__file__)),
        "input_refs": list(READS.values()),
        "training_eligible": False,
        "sealed_inputs_opened": False,
    }


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--source-sha256", required=True)
    args = parser.parse_args()
    need(file_ref(Path(__file__))["sha256"] == args.source_sha256, "root-selected assembly source")
    r = file_ref(args.selection)
    need(r["sha256"] == args.selection_sha256, "root-selected assembly inputs")
    result = execute(r)
    save(HERE / "root-dev24-bundle-assembly-result-01.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "complete_cases", "supervised_rows", "unused_heads_skipped")
            }
        )
    )


if __name__ == "__main__":
    main()
