"""Owned runtime discovery and nontraining numeric checks; stdlib imports only.

No caller may invoke payload/catalog functions until the genuine native loader
has completed. The actor supplies already authenticated native/helper modules.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import stat
import sys
from pathlib import Path

MAX_FILE = 256 * 1024**2
MAX_MAP_BYTES = 2 * 1024**2
MAX_SELECTED_FILES = 24
MAX_SELECTED_BYTES = 512 * 1024**2
MATH_LIBRARY = re.compile(
    r"(?:openblas|lapack|blas|gfortran|quadmath|libgomp|libiomp|libomp)", re.I
)
GPU_MATH_LIBRARY = re.compile(r"^lib(?:cublas(?:lt)?|nvblas)\.so(?:\.[0-9]+)*$", re.I)
POLICY = "comfy-observed-cpu-coexist-v1"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def owned(native, helper):
    identity = helper.process_identity()
    native.validate_owner(
        identity, helper.owned_limits(identity), {**helper.START_ENV, **helper.CPU_ENV}
    )
    require(helper.HOST_OBSERVATION_POLICY == POLICY, "new owner09 policy required")
    require(
        Path(sys.executable).resolve() == helper.RUNTIME_PYTHON.resolve(),
        "selected ML interpreter required",
    )
    return identity


def file_identity(path, native, helper, began):
    """Hash selected actual files only inside this admitted process, in chunks."""
    owned(native, helper)
    logical = Path(path)
    resolved = logical.resolve(strict=True)
    first = resolved.stat()
    require(
        stat.S_ISREG(first.st_mode) and 0 < first.st_size <= MAX_FILE,
        "bounded regular owned catalog/payload file required",
    )
    keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    digest, seen = hashlib.sha256(), 0
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            helper.ensure_elapsed(began, 20, "complete prediction during owned file hashing")
            seen += len(block)
            require(seen <= MAX_FILE, "owned file grew beyond bound")
            digest.update(block)
    last = resolved.stat()
    require(
        seen == first.st_size and all(getattr(first, key) == getattr(last, key) for key in keys),
        "owned file identity changed during hashing",
    )
    return {
        "path": str(logical),
        "resolved_path": str(resolved),
        "size_bytes": seen,
        "stat": {key: getattr(first, key) for key in keys},
        "sha256": digest.hexdigest(),
    }


def load_numpy(native, helper, state):
    require(
        state.get("model_loads") == 1
        and state.get("initial_full_model_manifest_runtime_sources_authenticated") is True,
        "load_runtime/full authentication must finish before NumPy",
    )
    owned(native, helper)
    np = importlib.import_module("numpy")
    expected = (
        helper.RUNTIME_PYTHON.parent.parent / "lib/python3.13/site-packages/numpy/__init__.py"
    )
    require(Path(np.__file__).resolve() == expected.resolve(), "actual ML3.13 NumPy origin differs")
    require(
        type(np.__version__) is str
        and re.fullmatch(r"2\.[0-9A-Za-z.+_-]+", np.__version__) is not None,
        "NumPy2 discovery projection required; version is not prequalified",
    )
    return np


def mapped_paths():
    with Path("/proc/self/maps").open("rb") as stream:
        raw = stream.read(MAX_MAP_BYTES + 1)
    require(len(raw) <= MAX_MAP_BYTES, "bounded own mapped-library inventory required")
    lines = raw.decode().splitlines()
    require(len(lines) <= 16384, "bounded own map rows required")
    paths = set()
    for line in lines:
        fields = line.split(None, 5)
        if len(fields) == 6 and fields[5].startswith("/"):
            name = fields[5]
            if MATH_LIBRARY.search(Path(name).name):
                require(not name.endswith(" (deleted)"), "mapped mathematical dependency deleted")
            if not name.endswith(" (deleted)"):
                paths.add(str(Path(name).resolve(strict=True)))
    return paths, hashlib.sha256(raw).hexdigest()


def split_math_candidates(mapped):
    """Separate only explicit CUDA BLAS basenames; NVPL remains a CPU candidate."""
    candidates = sorted(Path(path) for path in mapped if MATH_LIBRARY.search(Path(path).name))
    require(len(candidates) <= MAX_SELECTED_FILES, "bounded total mapped math inventory required")
    gpu = [path for path in candidates if GPU_MATH_LIBRARY.fullmatch(path.name)]
    cpu = [path for path in candidates if not GPU_MATH_LIBRARY.fullmatch(path.name)]
    return cpu, gpu


def gpu_candidate_metadata(path, native, helper, began):
    owned(native, helper)
    helper.ensure_elapsed(began, 20, "complete prediction during GPU map metadata")
    path = Path(path).resolve(strict=True)
    require(GPU_MATH_LIBRARY.fullmatch(path.name), "explicit GPU math basename required")
    before = path.stat()
    require(
        stat.S_ISREG(before.st_mode) and before.st_size > 0, "regular mapped GPU candidate required"
    )
    after = path.stat()
    keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    require(
        all(getattr(before, k) == getattr(after, k) for k in keys), "GPU candidate metadata changed"
    )
    return {
        "path": str(path),
        "size_bytes": before.st_size,
        "stat": {k: getattr(before, k) for k in keys},
        "hashed": False,
        "included_in_CPU_projection": False,
        "execution_or_symbol_binding_proven": False,
    }


def catalog(np, native, helper, began):
    owned(native, helper)
    package = Path(np.__file__).resolve().parent
    modules = [
        np,
        np.__config__,
        importlib.import_module("numpy.version"),
        importlib.import_module("numpy._core._multiarray_umath"),
        importlib.import_module("numpy.linalg._umath_linalg"),
    ]
    selected = []
    for module in modules:
        path = Path(module.__file__).resolve(strict=True)
        require(
            path.is_relative_to(package), "NumPy critical module origin escaped selected package"
        )
        selected.append(path)
    mapped, maps_sha = mapped_paths()
    for module in modules[-2:]:
        require(
            str(Path(module.__file__).resolve()) in mapped,
            "actual NumPy native extension is not mapped",
        )
    dependencies, gpu_dependencies = split_math_candidates(mapped)
    selected = list(dict.fromkeys([*selected, *dependencies]))
    require(
        len(selected) + len(gpu_dependencies) <= MAX_SELECTED_FILES,
        "bounded CPU plus excluded GPU catalog count required",
    )
    excluded_gpu = [
        gpu_candidate_metadata(path, native, helper, began) for path in gpu_dependencies
    ]
    require(
        len(selected) <= MAX_SELECTED_FILES, "bounded NumPy/math dependency projection required"
    )
    require(
        sum(path.stat().st_size for path in selected) <= MAX_SELECTED_BYTES,
        "bounded selected identity bytes before reading required",
    )
    records = [file_identity(path, native, helper, began) for path in selected]
    require(
        sum(row["size_bytes"] for row in records) <= MAX_SELECTED_BYTES,
        "bounded NumPy/math identity bytes required",
    )
    configuration = getattr(np.__config__, "CONFIG", None)
    require(type(configuration) is dict, "NumPy2 build-dependency configuration required")
    encoded = json.dumps(configuration, allow_nan=False)
    require(len(encoded.encode()) <= 65536, "bounded NumPy configuration required")
    blas = configuration.get("Build Dependencies", {}).get("blas")
    require(
        type(blas) is dict and type(blas.get("name")) is str and blas["name"],
        "explicit NumPy BLAS build-provider evidence required",
    )
    return {
        "schema": "owned-ML313-numpy-discovery-critical-projection-v1",
        "numpy_version": np.__version__,
        "python": {
            "logical_path": sys.executable,
            "resolved_path": str(Path(sys.executable).resolve()),
            "version": list(sys.version_info[:3]),
        },
        "critical_files_and_mapped_math_candidates": records,
        "numpy_build_configuration": configuration,
        "mapped_math_candidates": [str(path) for path in dependencies],
        "mapped_gpu_math_candidates_excluded_from_CPU_projection": excluded_gpu,
        "own_maps_sha256": maps_sha,
        "actual_discovery_after_native_load": True,
        "preselected_catalog": False,
        "full_DSO_or_transitive_closure_qualified": False,
        "scope": "Observed critical NumPy modules, mapped CPU mathematical-library candidates and build-provider configuration. Explicit CUDA BLAS maps are retained separately as stat-only candidates with execution/symbol binding unproven and are outside the CPU hash projection. Mapping candidates does not prove every symbol/dependency binding. Root accepted closed catalog must pin future optimizer runtime first.",
    }


def validate_answers(native, raw, request, lengths, model):
    """Original published confidence rule, including K=1; no answer repairs."""
    native.validate_native(raw, request["questions"], lengths, model)
    for key, question in request["questions"].items():
        answer = raw["answers"][key]
        probabilities = answer.get("probabilities", {})
        require(
            answer.get("type") == "choice" and list(probabilities) == list(question["criteria"]),
            "all original legal choices/order required",
        )
        values = list(probabilities.values())
        require(
            values
            and all(
                type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
                for value in values
            )
            and abs(sum(values) - 1) <= 1e-6,
            "finite normalized legal probabilities required",
        )
        require(
            answer.get("choice") == max(probabilities, key=probabilities.__getitem__),
            "original ordered native argmax required",
        )
        entropy = -sum(value * math.log(value) for value in values if value > 0)
        expected = (
            1.0 if len(values) < 2 else max(0.0, min(1.0, 1 - entropy / math.log(len(values))))
        )
        require(
            type(answer.get("confidence")) in (float, int)
            and math.isfinite(answer["confidence"])
            and abs(answer["confidence"] - expected) <= 1e-6,
            "published singleton/entropy confidence differs",
        )
    return {
        "contract_valid": True,
        "singleton_confidence_rule": "1.0 for K<2",
        "semantic_expected_answer_or_quality_claim": False,
    }


def copy_fp32(np, torch, tensor, shape):
    require(
        tensor.dtype == torch.float32
        and tensor.device.type == "cpu"
        and tuple(tensor.shape) == shape,
        "exact original CPU FP32 shape required before conversion",
    )
    source = tensor.detach().contiguous()
    result = source.numpy().copy(order="C")
    require(
        type(result) is np.ndarray
        and result.dtype == np.float32
        and result.shape == shape
        and result.flags.c_contiguous
        and bool(np.isfinite(result).all()),
        "finite independent C-order FP32 NumPy copy required",
    )
    require(
        torch.equal(torch.from_numpy(result), source)
        and result.tobytes(order="C") == source.view(torch.uint8).numpy().tobytes(order="C"),
        "exact Torch to NumPy FP32 value/byte conversion required",
    )
    result.flags.writeable = False
    return result


def write_pending_array(np, array, path, native, helper, began):
    owned(native, helper)
    require(
        type(array) is np.ndarray
        and array.dtype == np.float32
        and array.ndim == 2
        and array.nbytes <= 2 * 1024**2,
        "bounded non-object FP32 payload required",
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
    record = file_identity(path, native, helper, began)
    return {
        **record,
        "dtype": "float32",
        "shape": list(array.shape),
        "format": "npy_no_pickle",
        "status": "PENDING_NONTRAINING",
        "training_eligible": False,
    }


def qualify(
    np,
    core,
    observer_module,
    native,
    helper,
    runtime,
    torch,
    witness,
    questions,
    raw,
    counts,
    output,
    began,
    initial_catalog,
):
    owned(native, helper)
    witness.require_single_pass()
    features = copy_fp32(np, torch, witness.features, (4, 1024))
    baseline = copy_fp32(np, torch, runtime.readout, (255, 1024))
    before = [hashlib.sha256(array.tobytes()).hexdigest() for array in (features, baseline)]
    delta = np.zeros((255, 1024), dtype=np.float32)
    delta.flags.writeable = False
    legal = np.asarray(list(counts), dtype=np.int64)
    legal.flags.writeable = False
    probabilities = core.probabilities(features, baseline, delta, legal)
    require(
        type(probabilities) is np.ndarray
        and probabilities.dtype == np.float32
        and probabilities.shape == (4, 255)
        and bool(np.isfinite(probabilities).all()),
        "complete finite NumPy FP32 probabilities required",
    )
    require(
        all(bool((probabilities[index, count:] == 0).all()) for index, count in enumerate(counts)),
        "illegal code probability must be exactly zero",
    )
    result = observer_module.compare_distributions(
        raw, questions, [row[:count].tolist() for row, count in zip(probabilities, counts)]
    )
    require(
        before == [hashlib.sha256(array.tobytes()).hexdigest() for array in (features, baseline)],
        "NumPy modified immutable feature/baseline copies",
    )
    require(
        torch.equal(torch.from_numpy(baseline.copy()), runtime.readout.detach()),
        "original live W0 changed",
    )
    final_catalog = catalog(np, native, helper, began)
    before_files = {
        row["resolved_path"]: row
        for row in initial_catalog["critical_files_and_mapped_math_candidates"]
    }
    after_files = {
        row["resolved_path"]: row
        for row in final_catalog["critical_files_and_mapped_math_candidates"]
    }
    require(
        all(after_files.get(path) == row for path, row in before_files.items()),
        "initial NumPy/math identities changed during computation",
    )
    require(
        initial_catalog["numpy_version"] == final_catalog["numpy_version"]
        and initial_catalog["numpy_build_configuration"]
        == final_catalog["numpy_build_configuration"],
        "NumPy runtime configuration changed",
    )
    payloads = [
        write_pending_array(np, features, output / "features-pending.npy", native, helper, began),
        write_pending_array(np, baseline, output / "W0-pending.npy", native, helper, began),
    ]
    helper.write_json(
        output / "catalog-actual-pending.json",
        {
            "status": "PENDING_CLOSED_ROOT_ACCEPTANCE",
            "before": initial_catalog,
            "after": final_catalog,
            "training_eligible": False,
        },
    )
    helper.write_json(
        output / "payloads-pending.json",
        {
            "status": "PENDING_NONTRAINING",
            "payloads": payloads,
            "ordered_heads": list(questions),
            "legal_counts": list(counts),
            "baseline_feature_FP32_sha256": before,
            "labels": None,
            "training_eligible": False,
        },
    )
    witness.require_single_pass()
    helper.ensure_elapsed(
        began, 20, "complete prediction after NumPy conversion/catalog/hash/retention"
    )
    return {
        **result,
        "illegal_probabilities_exact_zero": True,
        "zero_delta_shape": [255, 1024],
        "feature_shape": [4, 1024],
        "FP32_Torch_NumPy_copy_exact": True,
        "catalog_is_actual_discovery_pending_root_acceptance": True,
        "preselected_full_DSO_closure_qualified": False,
        "raw_payload_status": "PENDING_NONTRAINING",
        "training_eligible": False,
    }
