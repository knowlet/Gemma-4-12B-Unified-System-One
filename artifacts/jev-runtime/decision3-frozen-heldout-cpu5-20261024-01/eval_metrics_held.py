"""Held full-dev24 cached-head evaluation; permanently NONTRAINING.

Only a root-selected private metric bundle, the frozen delta and trusted runtime
metadata are read. No DSL, train loader, optimizer, model or raw-dev input API.
Source authoring only: no import/check/AST/test/hash/actual read has been run.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import stat
import sys
import time
import types
from pathlib import Path

SOURCE_ONLY = True
ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
HERE = ROOT / "artifacts/jev-runtime/decision3-frozen-heldout-cpu5-20261024-01"
FIT = (
    ROOT / "artifacts/jev-runtime/decision3-train-readout-cpu5-20261024-01/fit_head_released_v2.py"
)
FIT_SHA = "2019360a1a6093fae4d2eeb1250b85041c17622053e761dd3553c989fe425820"
CAPABILITY = HERE / "root-selected-dev24-metrics-capability-01.json"
CAPABILITY_SHA = None
CANDIDATE_SHA = "239165d486bad16963a6e5c94c1f9503fc145ea5b85b114e9fbf94d84eb1fd76"
DELTA_SHA = "812256139b35429dc55f4f178af8c177d2a31c023414e668e9b39cdc217d14be"
PROFILE = "dsl01-frozen-dev24-cached-head-metrics-cpu5-4g-v1"
HEADS = ("operation", "click_target", "type_text_target", "select_target")
SCHEMA = "root-selected-frozen-dev24-metrics-v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def released():
    require(SOURCE_ONLY is False, "SOURCE_ONLY: dev metrics is held before arguments/IO/imports")
    require(
        isinstance(CAPABILITY_SHA, str) and re.fullmatch(r"[0-9a-f]{64}", CAPABILITY_SHA),
        "dev metric capability pin remains pending",
    )


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate metadata key")
        result[key] = value
    return result


def decode(raw):
    return json.loads(
        raw, object_pairs_hook=pairs, parse_constant=lambda _s: require(False, "nonfinite metadata")
    )


def normalized(ref):
    path = Path(ref["path"])
    path = path if path.is_absolute() else ROOT / path
    require(
        path.is_relative_to(ROOT) and ".." not in path.parts, "selected repository path required"
    )
    require(
        not any(p.is_symlink() for p in (path, *path.parents)), "ordinary selected files required"
    )
    return {"path": str(path), "sha256": ref["sha256"], "size_bytes": ref["size_bytes"]}


def read_fixed(path, sha, deadline, size=None):
    deadline()
    path = Path(normalized({"path": str(path), "sha256": sha, "size_bytes": size})["path"])
    require(stat.S_ISREG(path.lstat().st_mode), "regular selected metric file required")
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(0 < before.st_size <= 16 * 1024**2, "bounded metric metadata/array required")
        raw = stream.read(16 * 1024**2 + 1)
        after = os.fstat(stream.fileno())
    keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(
        len(raw) == before.st_size
        and (size is None or len(raw) == size)
        and all(
            getattr(before, key) == getattr(after, key) == getattr(path.stat(), key) for key in keys
        )
        and hashlib.sha256(raw).hexdigest() == sha,
        "selected metric bytes changed",
    )
    deadline()
    return raw


class RuntimeReader:
    """Minimal read-only facade for fit.numpy_runtime; no train-loader role."""

    decode = staticmethod(decode)

    def __init__(self, cap):
        refs = list(cap["arrays"].values()) + list(cap["runtime"].values())
        refs += [cap["candidate_capability"], cap["delta_ref"]]
        self.allowed = {tuple(normalized(ref).items()) for ref in refs}
        self.runtime = cap["runtime"]

    def read_ref(self, ref, deadline):
        record = normalized(ref)
        require(tuple(record.items()) in self.allowed, "unselected/non-metric input refused")
        return read_fixed(record["path"], record["sha256"], deadline, record["size_bytes"])

    def validate_np_acceptance(self, prerequisite, _selection, deadline):
        deadline()
        require(
            prerequisite["schema"] == "root-accepted-np5-qualification-v1"
            and prerequisite["status"] == "accepted"
            and prerequisite["owner_profile"]
            == "llama-cpp-cpu5-fast-same-cluster-4g-coexist-command-owned-v1"
            and normalized(prerequisite["numpy_runtime_catalog"])
            == normalized(self.runtime["numpy_runtime_catalog"]),
            "root-selected accepted runtime metadata/catalog differs",
        )
        return self.runtime["numpy_runtime_catalog"]


def validate_mapping(cap):
    require(
        cap["schema"] == SCHEMA
        and cap["status"] == "ROOT_SELECTED_FROZEN_DEV24_METRICS_ONLY"
        and cap["split"] == "dev"
        and cap["nontraining"] is True
        and cap["training_eligible"] is False
        and cap["frozen_candidate_steps"] == 32
        and set(cap["arrays"]) == {"features", "counts", "soft_targets", "W0"}
        and set(cap["runtime"]) == {"root_np5_prerequisite_ref", "numpy_runtime_catalog"},
        "complete dev-only metric capability required",
    )
    cases, rows = cap["cases"], cap["rows"]
    ids = [case["case_id"] for case in cases]
    require(
        len(ids) == len(set(ids)) == 24 and all(isinstance(value, str) and value for value in ids),
        "exact24 unique cases required; no partial score",
    )
    expected, unused = set(), 0
    for case in cases:
        offered, supervised = case["offered_heads"], case["supervised_heads"]
        require(
            type(offered) is list
            and len(offered) == 4
            and len(set(offered)) == 4
            and set(offered) == set(HEADS)
            and offered[0] == "operation"
            and supervised
            and len(supervised) == len(set(supervised))
            and supervised == [head for head in offered if head in supervised]
            and "operation" in supervised,
            "actual complete offered order and legal supervised subset required",
        )
        expected.update((case["case_id"], head) for head in supervised)
        unused += len(offered) - len(supervised)
    require(
        24 <= len(rows) <= 96 and len(rows) == len(expected),
        "complete supervised denominator required",
    )
    seen = []
    for i, row in enumerate(rows):
        labels = row["ordered_criteria_ids"]
        require(
            row["row_index"] == i
            and (row["case_id"], row["head_id"]) in expected
            and type(labels) is list
            and 1 <= len(labels) <= 255
            and len(labels) == len(set(labels))
            and all(isinstance(label, str) and label for label in labels),
            "exact case/head/slot order required",
        )
        seen.append((row["case_id"], row["head_id"]))
    ordered = [(case["case_id"], head) for case in cases for head in case["supervised_heads"]]
    require(
        len(set(seen)) == len(seen) and set(seen) == expected and seen == ordered,
        "missing/duplicate/reordered supervised head",
    )
    bundle = Path(normalized({"path": cap["bundle_root"], "sha256": "", "size_bytes": 0})["path"])
    require(
        bundle.parent == HERE
        and all(Path(normalized(ref)["path"]).parent == bundle for ref in cap["arrays"].values()),
        "only dedicated private dev metric bundle arrays allowed",
    )
    return unused


def load_array(np, reader, ref, shape, dtype, deadline):
    require(
        ref["format"] == "npy_no_pickle" and ref["shape"] == list(shape) and ref["dtype"] == dtype,
        "exact metric array declaration required",
    )
    value = np.load(
        io.BytesIO(reader.read_ref(ref, deadline)), allow_pickle=False, max_header_size=4096
    )
    require(
        type(value) is np.ndarray
        and value.dtype == np.dtype(dtype)
        and value.shape == shape
        and value.flags.c_contiguous
        and bool(np.isfinite(value).all()),
        "finite exact metric array required",
    )
    if "FP32_bytes_sha256" in ref:
        require(
            hashlib.sha256(value.tobytes(order="C")).hexdigest() == ref["FP32_bytes_sha256"],
            "original W0 value bytes differ",
        )
    value.flags.writeable = False
    return value


def aggregate_rows(rows, cases, correct, losses):
    """Only positive-target row hits; all-heads denominator is always24 cases."""
    require(
        len(rows) == len(correct) == len(losses)
        and len(cases) == 24
        and all(any(row["case_id"] == case["case_id"] for row in rows) for case in cases),
        "complete aggregate inputs required",
    )
    by_head = {}
    for head in HEADS:
        indices = [i for i, row in enumerate(rows) if row["head_id"] == head]
        count = len(indices)
        by_head[head] = {
            "rows": count,
            "correct": sum(correct[i] for i in indices),
            "accuracy": sum(correct[i] for i in indices) / count if count else None,
            "mean_soft_ce": sum(losses[i] for i in indices) / count if count else None,
        }
    case_hits = [
        all(correct[i] for i, row in enumerate(rows) if row["case_id"] == case["case_id"])
        for case in cases
    ]
    return {
        "supervised_rows": len(rows),
        "valid_target_argmax_correct": sum(correct),
        "valid_target_argmax_accuracy": sum(correct) / len(rows),
        "by_head": by_head,
        "all_supervised_heads_correct": {
            "cases": 24,
            "correct": sum(case_hits),
            "accuracy": sum(case_hits) / 24,
        },
    }


def valid_target_hit(probabilities, targets, count):
    require(
        type(count) is int and 1 <= count <= 255 and len(probabilities) == len(targets) == 255,
        "complete legal metric row required",
    )
    index = max(range(count), key=probabilities.__getitem__)
    return targets[index] > 0


def metrics(np, core, x, w0, counts, targets, delta, cap, deadline):
    deadline()
    illegal = core._inputs(x, w0, delta, counts, targets)
    probabilities, logp = core._probabilities(x, w0, delta, illegal)
    # No L2, gradient or optimizer. Stable log probabilities avoid log(0) after
    # FP32 probability underflow. The full mean uses the frozen FP32 reduction.
    positive = targets > 0
    loss = -(targets[positive] * logp[positive]).sum(dtype=np.float32) / np.float32(len(x))
    with np.errstate(invalid="raise", over="raise", under="ignore"):
        row_losses = -(targets * np.where(positive, logp, 0)).sum(axis=1, dtype=np.float32)
    correct = [
        valid_target_hit(row, target, count)
        for row, target, count in zip(probabilities.tolist(), targets.tolist(), counts.tolist())
    ]
    require(
        math.isfinite(float(loss)) and bool(np.isfinite(row_losses).all()),
        "finite full-dev objective required",
    )
    result = aggregate_rows(cap["rows"], cap["cases"], correct, row_losses.tolist())
    result.update(full_mean_soft_ce=float(loss), l2_applied=False, optimizer_steps=0)
    deadline()
    return result


def fingerprints(arrays):
    require(
        all(array.flags.writeable is False for array in arrays.values()),
        "metric inputs must remain readonly",
    )
    return {
        name: hashlib.sha256(array.tobytes(order="C")).hexdigest() for name, array in arrays.items()
    }


def execute(owner_output):
    released()
    began = time.monotonic()

    def deadline():
        require(time.monotonic() - began < 120, "whole owned dev-metrics120 exceeded")

    require(
        sys.flags.no_user_site
        and sys.dont_write_bytecode
        and not any(
            name.split(".")[0] in {"numpy", "torch", "transformers", "safetensors", "s1"}
            for name in sys.modules
        ),
        "fresh -B -s numerical/model-free child required",
    )
    raw = read_fixed(FIT, FIT_SHA, deadline)
    fit = types.ModuleType("_dev24_metrics_original_helpers")
    fit.__file__ = str(FIT)
    sys.modules[fit.__name__] = fit
    exec(compile(raw, str(FIT), "exec", dont_inherit=True), vars(fit))
    owner = fit.source(
        "_dev24_metrics_original_owner09", fit.OWNER / "controller.py", fit.OWNER_SHA
    )
    out, cfg, _spec, identity, resources = fit.owned(owner_output, owner, deadline)
    cap_raw = read_fixed(CAPABILITY, CAPABILITY_SHA, deadline)
    cap = decode(cap_raw)
    unused = validate_mapping(cap)
    reader = RuntimeReader(cap)
    candidate = decode(reader.read_ref(cap["candidate_capability"], deadline))
    require(
        cap["candidate_capability"]["sha256"] == CANDIDATE_SHA
        and cap["delta_ref"]["sha256"] == DELTA_SHA
        and candidate["schema"] == "root-accepted-dsl01-train-delta-v1"
        and candidate["status"] == "ROOT_ACCEPTED_TRAIN_ONLY_READOUT"
        and candidate["delta_ref"] == cap["delta_ref"]
        and candidate["model"] == cap["model"]
        and candidate["frozen_before_dev_sealed_benchmark_or_browser"] is True
        and candidate["W0_merge_allowed"] is False
        and candidate["baseline_fp32_bytes_sha256"] == cap["arrays"]["W0"]["FP32_bytes_sha256"],
        "same frozen32-step candidate/model/original W0 required",
    )
    pool = {str(ROOT / row["path"]): row["sha256"] for row in cfg["sources"]}
    for path, sha in (
        (FIT, FIT_SHA),
        (fit.CORE, fit.CORE_SHA),
        (CAPABILITY, CAPABILITY_SHA),
        (Path(cap["candidate_capability"]["path"]), CANDIDATE_SHA),
        (
            Path(__file__).absolute(),
            hashlib.sha256(owner.read(Path(__file__).absolute())).hexdigest(),
        ),
    ):
        require(
            pool.get(str(path)) == sha,
            "original owner did not select actor/helper/core/metadata bytes",
        )
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "5"
    np, catalog, catalog_ref, mapped_before = fit.numpy_runtime(reader, cap["runtime"], deadline)
    core = fit.source("_dev24_metrics_frozen_core", fit.CORE, fit.CORE_SHA)
    n = len(cap["rows"])
    declarations = {
        "features": ((n, 1024), "float32"),
        "counts": ((n,), "int64"),
        "soft_targets": ((n, 255), "float32"),
        "W0": ((255, 1024), "float32"),
    }
    arrays = {
        key: load_array(np, reader, cap["arrays"][key], shape, dtype, deadline)
        for key, (shape, dtype) in declarations.items()
    }
    delta = load_array(np, reader, cap["delta_ref"], (255, 1024), "float32", deadline)
    require(
        arrays["counts"].tolist() == [len(row["ordered_criteria_ids"]) for row in cap["rows"]],
        "actual counts/criteria slot mapping differs",
    )
    zero = np.zeros_like(delta, dtype=np.float32)
    zero.flags.writeable = False
    arrays.update(delta=delta, zero_delta=zero)
    before = fingerprints(arrays)
    results = {
        name: metrics(
            np,
            core,
            arrays["features"],
            arrays["W0"],
            arrays["counts"],
            arrays["soft_targets"],
            value,
            cap,
            deadline,
        )
        for name, value in (("zero_delta", zero), ("frozen_candidate", delta))
    }
    require(
        fingerprints(arrays) == before,
        "evaluation mutated private features/counts/targets/W0/delta",
    )
    for record in reader.allowed:
        reader.read_ref(dict(record), deadline)
    read_fixed(CAPABILITY, CAPABILITY_SHA, deadline, len(cap_raw))
    _, catalog_after, ref_after, mapped_after = fit.numpy_runtime(reader, cap["runtime"], deadline)
    require(catalog_after == catalog and ref_after == catalog_ref, "selected NumPy runtime changed")
    owner.sources_match(cfg["sources"])
    owner.command_selection(cfg["selection"], cfg["selection_sha256"], cfg["base_sources"])
    summary = {
        "schema": "dev24-cached-head-metrics-nontraining-v1",
        "profile": PROFILE,
        "status": "NONTRAINING_PENDING_ROOT_ACCEPTANCE",
        "nontraining": True,
        "training_eligible": False,
        "split": "dev",
        "complete_cases": 24,
        "supervised_rows": n,
        "unused_heads_skipped": unused,
        "capability": {
            "path": str(CAPABILITY),
            "sha256": CAPABILITY_SHA,
            "size_bytes": len(cap_raw),
        },
        "candidate_capability": cap["candidate_capability"],
        "results": results,
        "case_head_criteria_mapping_retained_in_private_capability": True,
        "readonly_value_hashes": before,
        "raw_arrays_or_probabilities_written": False,
        "optimizer_steps": 0,
        "l2_applied": False,
        "browser_or_official_score": False,
        "runtime_catalog": catalog_ref,
        "mapped_math_before": mapped_before,
        "mapped_math_after": mapped_after,
        "full_DSO_or_transitive_closure_qualified": False,
        "process": identity,
        "owned_resources": resources,
        "original_owner_nonce": cfg["nonce"],
        "elapsed_before_terminal_write_seconds": time.monotonic() - began,
    }
    target = out / "dev-metrics"
    target.mkdir(exist_ok=False)
    (target / ".gitignore").write_text("*\n")
    with (target / "metrics-nontraining-pending.json").open("xb") as stream:
        stream.write(
            json.dumps(summary, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(target, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    deadline()  # final retention/fsync is charged; no partial score is returned
    return summary


def main(argv=None):
    released()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--owner-output", type=Path, required=True)
    execute(parser.parse_args(argv).owner_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
