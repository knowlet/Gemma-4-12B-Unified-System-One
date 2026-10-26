"""SOURCE-ONLY full-train zero/candidate objective diagnostic; no optimizer.

Uses the two fixed root capabilities, released fit admission/runtime helpers,
released COMMITTED loader and frozen core. No model or non-train corpus is
imported. Output is aggregate-only, pending new original root/owner acceptance.
This source has not been executed, imported, parsed, formatted or tested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import types
from pathlib import Path

SOURCE_ONLY = False
ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
HERE = ROOT / "artifacts/jev-runtime/decision3-train-readout-cpu5-20261024-01"
FIT = HERE / "fit_head_released_v2.py"
FIT_SHA256 = "2019360a1a6093fae4d2eeb1250b85041c17622053e761dd3553c989fe425820"
SELECTION = HERE / "root-committed-train-fit-capability-01.json"
SELECTION_SHA256 = "51d87342a04e1121e669da018591d62778cbc52cb4599f8763df97317d25a45b"
CANDIDATE = HERE / "root-frozen-delta-candidate-capability-01.json"
CANDIDATE_SHA256 = "239165d486bad16963a6e5c94c1f9503fc145ea5b85b114e9fbf94d84eb1fd76"
LOADER_SHA256 = "c2b1e8cbeca4d78ee276ebac071c0c15be0a8cf7073fe5526a91ab14a8741763"
ACCEPTANCE_SHA256 = "cc629a1893ce53c232cc323253f39e86496b0e7a7347c5d11df8fa259ccb5baa"
DELTA_SHA256 = "812256139b35429dc55f4f178af8c177d2a31c023414e668e9b39cdc217d14be"
PROFILE = "dsl01-full-train-zero-frozen-delta-metrics-cpu5-4g-v1"
ROWS, WIDTH, CODES, WHOLE_SECONDS = 132, 1024, 255, 120


def require(condition, message):
    if not condition:
        raise ValueError(message)


def released():
    require(SOURCE_ONLY is False, "SOURCE_ONLY: full-train metrics actor is held")


def bootstrap_fit():
    """Only bootstrap duplication: fixed bytes before reusing fit.source below."""
    released()
    require(
        not any(path.is_symlink() for path in (FIT, *FIT.parents)),
        "ordinary fixed fitter source required",
    )
    with FIT.open("rb") as stream:
        raw = stream.read(1024**2 + 1)
    require(
        len(raw) <= 1024**2 and hashlib.sha256(raw).hexdigest() == FIT_SHA256,
        "released fitter source SHA differs",
    )
    module = types.ModuleType("_dsl01_metrics_released_fit_v2")
    module.__file__ = str(FIT)
    sys.modules[module.__name__] = module
    exec(compile(raw, str(FIT), "exec", dont_inherit=True), vars(module))
    require(
        module.SOURCE_ONLY is False
        and module.SELECTION_SHA256 == SELECTION_SHA256
        and module.LOADER_SHA256 == LOADER_SHA256,
        "original released fit pins/status differ",
    )
    return module


def validate_candidate(candidate, selection, train_ref):
    """Two immutable root pins bind this one delta to these accepted train rows."""
    require(
        selection["schema"] == "root-selected-dsl01-committed-train-readout-v1"
        and selection["status"] == "ROOT_SELECTED_COMMITTED_TRAIN_ONLY"
        and selection["split"] == "train"
        and candidate["schema"] == "root-accepted-dsl01-train-delta-v1"
        and candidate["status"] == "ROOT_ACCEPTED_TRAIN_ONLY_READOUT"
        and candidate["split"] == "train"
        and candidate["training_eligible"] is True
        and candidate["nontraining"] is False
        and candidate["profile"] == "decision3-dsl01-delta-readout-cpu5-4g-candidate-v1"
        and candidate["alias"] == "d3-edge-dsl01-train-delta-cpu5-candidate"
        and candidate["owner_profile"]
        == "llama-cpp-cpu5-fast-same-cluster-4g-coexist-command-owned-v1"
        and candidate["model"] == selection["model"]
        and candidate["fit_capability"] == train_ref
        and candidate["baseline_fp32_bytes_sha256"] == selection["baseline"]["FP32_bytes_sha256"]
        and candidate["root_acceptance_ref"]["path"]
        == str(HERE / "root-accepted-trained-readout-01.json")
        and candidate["root_acceptance_ref"]["sha256"] == ACCEPTANCE_SHA256
        and candidate["delta_ref"]["sha256"] == DELTA_SHA256
        and candidate["delta_ref"]["format"] == "npy_no_pickle"
        and candidate["delta_ref"]["dtype"] == "float32"
        and candidate["delta_ref"]["shape"] == [CODES, WIDTH]
        and candidate["frozen_before_dev_sealed_benchmark_or_browser"] is True
        and candidate["W0_merge_allowed"] is False,
        "exact root-frozen train-only candidate/fit/model/W0 binding differs",
    )


def valid_target_accuracy(probabilities, targets, counts):
    """First ordered argmax is correct iff its authored soft target is positive.

    Accepts bounded ordinary lists after the frozen core has validated matrices.
    Multiple valid targets remain valid; no fabricated single gold is chosen.
    """
    require(
        len(probabilities) == len(targets) == len(counts) > 0, "complete objective rows required"
    )
    correct = 0
    for probabilities_row, target_row, count in zip(probabilities, targets, counts):
        require(
            type(count) is int
            and 1 <= count <= CODES
            and len(probabilities_row) == len(target_row) == CODES,
            "complete legal probability/target row required",
        )
        index = max(range(count), key=probabilities_row.__getitem__)
        correct += int(target_row[index] > 0)
    return {
        "correct": correct,
        "rows": len(counts),
        "accuracy": correct / len(counts),
        "criterion": "first ordered argmax has positive authored target mass",
    }


def objective(np, core, data, delta, l2, deadline):
    """Frozen full-matrix arithmetic and stable soft CE, without any gradient/update."""
    deadline()
    x, w0, counts, targets = (data[name] for name in ("features", "baseline", "counts", "targets"))
    illegal = core._inputs(x, w0, delta, counts, targets)
    probabilities, log_probabilities = core._probabilities(x, w0, delta, illegal)
    coefficient = core._positive(l2, "l2", zero=True)
    positive = targets > 0
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        # Same FP32 reduction/casts as frozen loss_gradient, using stable log
        # probabilities. No log(rounded probability), optimizer or gradient.
        soft_ce = -(targets[positive] * log_probabilities[positive]).sum(
            dtype=np.float32
        ) / np.float32(len(x))
        penalty = np.float32(0.5) * coefficient * (delta * delta).sum(dtype=np.float32)
        full_objective = soft_ce + penalty
    require(
        all(math.isfinite(float(value)) for value in (soft_ce, penalty, full_objective)),
        "nonfinite complete-train objective",
    )
    accuracy = valid_target_accuracy(probabilities.tolist(), targets.tolist(), counts.tolist())
    deadline()
    return {
        "full_mean_soft_ce": float(soft_ce),
        "half_l2_delta_squared": float(penalty),
        "full_mean_soft_ce_plus_same_l2": float(full_objective),
        "l2": l2,
        "valid_target_argmax": accuracy,
    }


def array_fingerprints(arrays):
    """Bounded in-memory retention proof; never export tensors/probabilities."""
    result = {}
    for name, array in arrays.items():
        require(array.flags.writeable is False, "diagnostic inputs must remain readonly")
        result[name] = {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "c_order_bytes_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    return result


def execute(owner_output):
    released()  # precedes argument paths/dependencies/model/numerical imports
    began = time.monotonic()

    def deadline():
        require(time.monotonic() - began <= WHOLE_SECONDS, "whole owned train-metrics120 exceeded")

    require(
        sys.flags.no_user_site
        and sys.dont_write_bytecode
        and not any(
            name.split(".")[0] in {"numpy", "torch", "transformers", "safetensors", "s1"}
            for name in sys.modules
        ),
        "fresh -B -s numerical/model-free child required",
    )
    fit = bootstrap_fit()
    controller = fit.source(
        "_dsl01_metrics_original_owner09", fit.OWNER / "controller.py", fit.OWNER_SHA
    )
    out, cfg, _spec, identity, resources = fit.owned(owner_output, controller, deadline)
    loader_path = HERE / "train_loader_released_v1.py"
    loader = fit.source("_dsl01_metrics_committed_train_loader", loader_path, LOADER_SHA256)
    require(loader.SOURCE_ONLY is False, "original train loader remains held")
    blobs = {SELECTION: controller.read(SELECTION), CANDIDATE: controller.read(CANDIDATE)}
    refs = []
    for path, expected in ((SELECTION, SELECTION_SHA256), (CANDIDATE, CANDIDATE_SHA256)):
        raw = blobs[path]
        require(
            len(raw) <= 64 * 1024 and hashlib.sha256(raw).hexdigest() == expected,
            "fixed root capability bytes differ",
        )
        refs.append({"path": str(path), "sha256": expected, "size_bytes": len(raw)})
    selection, candidate = (loader.decode(blobs[path]) for path in (SELECTION, CANDIDATE))
    validate_candidate(candidate, selection, refs[0])
    require(
        selection["hyperparameters"] == loader.HYPERPARAMETERS
        and selection["projection"] == loader.PROJECTION
        and selection["fit_budget_seconds"] == WHOLE_SECONDS,
        "unchanged accepted train numerical policy required",
    )
    pool = {str(ROOT / row["path"]): row["sha256"] for row in cfg["sources"]}
    for path, sha in (
        (FIT, FIT_SHA256),
        (loader_path, LOADER_SHA256),
        (fit.CORE, fit.CORE_SHA),
        (SELECTION, SELECTION_SHA256),
        (CANDIDATE, CANDIDATE_SHA256),
        (
            Path(__file__).absolute(),
            hashlib.sha256(controller.read(Path(__file__).absolute())).hexdigest(),
        ),
    ):
        require(
            pool.get(str(path)) == sha,
            "original owner did not select metrics/capability/helper/core source bytes",
        )
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "5"
    np, catalog, catalog_ref, mapped_before = fit.numpy_runtime(loader, selection, deadline)
    core = fit.source("_dsl01_metrics_frozen_head_core", fit.CORE, fit.CORE_SHA)
    data = loader.load_training(selection, np, core, check_deadline=deadline)
    aggregate = data["aggregate"]
    require(
        all(
            aggregate[key] == value
            for key, value in {
                "train_states": 96,
                "ancestors": 12,
                "families": 4,
                "supervised_heads": ROWS,
                "unused_heads_skipped": 252,
                "dev_or_sealed_selected": False,
            }.items()
        )
        and data["features"].shape == (ROWS, WIDTH)
        and data["targets"].shape == (ROWS, CODES),
        "same complete96-state/132-supervised/252-unused train inventory required",
    )
    # Acceptance is selected by the root-frozen capability, never by a path's
    # name/status. Reauthenticate its exact accepted metadata and array bytes.
    loader.read_ref(candidate["root_acceptance_ref"], deadline)
    delta = loader.load_array(np, candidate["delta_ref"], HERE, (CODES, WIDTH), deadline)
    zero = np.zeros_like(data["baseline"], dtype=np.float32)
    zero.flags.writeable = False
    arrays = {name: data[name] for name in ("features", "baseline", "counts", "targets")}
    arrays.update(candidate_delta=delta, zero_delta=zero)
    before = array_fingerprints(arrays)
    metrics = {
        "zero_delta": objective(np, core, data, zero, loader.HYPERPARAMETERS["l2"], deadline),
        "frozen_candidate": objective(
            np, core, data, delta, loader.HYPERPARAMETERS["l2"], deadline
        ),
    }
    require(
        array_fingerprints(arrays) == before, "objective mutated W0/features/labels/counts/delta"
    )
    loader.reauthenticate(
        data["pinned_refs"] + refs + [candidate["root_acceptance_ref"], candidate["delta_ref"]],
        deadline,
    )
    _, catalog_after, catalog_ref_after, mapped_after = fit.numpy_runtime(
        loader, selection, deadline
    )
    require(
        catalog_after == catalog and catalog_ref_after == catalog_ref,
        "selected numerical runtime catalog changed",
    )
    controller.sources_match(cfg["sources"])
    controller.command_selection(cfg["selection"], cfg["selection_sha256"], cfg["base_sources"])
    deadline()
    summary = {
        "schema": "dsl01-full-train-zero-frozen-delta-metrics-pending-v1",
        "profile": PROFILE,
        "status": "PENDING_ROOT_ACCEPTANCE",
        "training_eligible": False,
        "selection": refs[0],
        "candidate": refs[1],
        "delta": candidate["delta_ref"],
        "train": aggregate,
        "metrics": metrics,
        "readonly_input_fingerprints": before,
        "array_values_unchanged": True,
        "optimizer_steps": 0,
        "evaluation_scope": "same full132 train rows, not distinct SGD minibatches",
        "runtime_catalog": catalog_ref,
        "mapped_math_before": mapped_before,
        "mapped_math_after": mapped_after,
        "full_DSO_or_transitive_closure_qualified": False,
        "process": identity,
        "owned_resources": resources,
        "original_owner_nonce": cfg["nonce"],
        "dev_or_sealed_selected": False,
        "browser_or_official_tasks_selected": False,
        "raw_arrays_or_probabilities_written": False,
        "ranking_or_win_claim": False,
        "model_qualified": False,
        "browser_qualified": False,
        "elapsed_before_final_write_seconds": time.monotonic() - began,
        "scope": "Train-only fixed full-objective diagnostic. New root terminal0, owner completion and current positive closure acceptance remain required.",
    }
    target = out / "train-metrics"
    target.mkdir(exist_ok=False)
    (target / ".gitignore").write_text("*\n")
    with (target / "metrics-pending.json").open("xb") as stream:
        stream.write(
            json.dumps(summary, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(target, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    deadline()  # auth, input/runtime retention, final fsync and cleanup are charged
    return summary


def main(argv=None):
    released()  # before even parsing arguments or opening root capabilities
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-output", type=Path, required=True)
    args = parser.parse_args(argv)
    execute(args.owner_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
