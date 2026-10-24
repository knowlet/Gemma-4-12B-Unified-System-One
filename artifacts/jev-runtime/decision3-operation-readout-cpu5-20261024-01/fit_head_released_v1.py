"""HELD operation-only optimizer over the complete accepted DSL01 train view.

The original owner09 supervises this child; this module creates no guardian.
Release needs new source review, fixed capability/loader hashes and an original
owner command spec. No CLI argument can release the source hold. Dev results
informed this model-selection policy; dev/sealed features and labels are never
optimizer inputs. Whether512 steps fit within the original120s is unmeasured.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
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

SOURCE_ONLY = False
SELECTION_SHA256 = '3cbb5cb38afacdb0f16ca3c770f6565e1895e722246172ed01b957512c41ea4e'
LOADER_SHA256 = '2a147760e6a18338d12a4bb33a53a92860ba5f362b2d91567c991f96baa6fe57'
ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
HERE = ROOT / "artifacts/jev-runtime/decision3-operation-readout-cpu5-20261024-01"
LOADER = HERE / "train_loader_released_v1.py"
PROFILE = "decision3-dsl01-operation-readout-cpu5-4g-candidate-v1"
ALIAS = "d3-edge-dsl01-operation-delta-cpu5-candidate"
OWNER = ROOT / "artifacts/jev-runtime/llama-cpp-owner-20261024-09"
OWNER_SHA = "81846efe2ee0357da07283c1a77a606283a920cbddef668c8f62def291057dee"
CORE = ROOT / "artifacts/jev-runtime/decision3-cached-head-core-20261023-01/head_core.py"
CORE_SHA = "6761dd654481424f4d6c908a7c9fd2d43e4762eeb5420af9f22612ffe77657ed"
MATH_LIBRARY = re.compile(
    r"(?:openblas|lapack|blas|gfortran|quadmath|libgomp|libiomp|libomp)", re.I
)
GPU_MATH_LIBRARY = re.compile(r"^lib(?:cublas(?:lt)?|nvblas)\.so(?:\.[0-9]+)*$", re.I)
GPU_CATALOG_KEY = "mapped_gpu_math_candidates_excluded_from_CPU_projection"


def require(value, message):
    if not value:
        raise ValueError(message)


def released():
    require(SOURCE_ONLY is False, "SOURCE_ONLY: fit caller is not released")
    require(
        all(
            type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value)
            for value in (SELECTION_SHA256, LOADER_SHA256)
        ),
        "independent capability and released-loader pins remain pending",
    )


def source(name, path, expected):
    # Called only after released(); never use a generator/full manifest module.
    require(
        path.is_absolute()
        and path.is_relative_to(ROOT)
        and not any(p.is_symlink() for p in (path, *path.parents)),
        "ordinary fixed source required",
    )
    with path.open("rb") as stream:
        blob = stream.read(1024**2 + 1)
    require(
        len(blob) <= 1024**2 and hashlib.sha256(blob).hexdigest() == expected,
        "fixed source SHA differs",
    )
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(blob, str(path), "exec", dont_inherit=True), vars(module))
    return module


def child_properties(controller, owner, selected, pid):
    """Reuse original kernel/proc checks without a child user-bus dependency."""
    group = owner.CGROUP_ROOT / selected.cgroup.lstrip("/")
    quota = (group / "cpu.max").read_text().split()
    require(
        len(quota) == 2
        and quota[0].isdigit()
        and quota[1].isdigit()
        and int(quota[1]) > 0
        and int(quota[0]) == 5 * int(quota[1]),
        "Actual CPU500% quota kernel readback",
    )
    limits = (Path("/proc") / str(pid) / "limits").read_text()
    for name, number, unit in (
        ("Max core file size", 0, "bytes"),
        ("Max file size", 256 * 1024**2, "bytes"),
        ("Max open files", 512, "files"),
    ):
        require(
            re.search(
                r"(?m)^"
                + name
                + r"\s+"
                + str(number)
                + r"\s+"
                + str(number)
                + r"\s+"
                + unit
                + r"\s*$",
                limits,
            ),
            "Actual process limit: " + name,
        )
    peak = (group / "memory.peak").read_text().strip()
    require(peak.isdigit(), "Actual kernel memory.peak")
    affinity = controller.main_affinity(owner, pid)
    cpuset = group / "cpuset.cpus.effective"
    return {
        "scope": "Child kernel/proc readback only; original parent owner performs systemd unit property verification",
        "kernel_memory_peak_bytes": int(peak),
        "kernel_memory_events": controller.memory_events(group),
        "kernel_cpu_max": quota,
        "cpuset_cpus_effective_raw": cpuset.read_text().strip() if cpuset.exists() else None,
        "main_thread_affinity": affinity,
        "affinity_scope": "Observed main-thread affinity only; no delegated cpuset prevention of later widening or all-thread lifetime claim",
    }


def owned(owner_output, controller, check_deadline):
    """Join the actual child to original admission, intent and started receipt."""
    out = controller.ordinary(owner_output)
    require(
        out.parent == OWNER and re.fullmatch(r"actual-[0-9]{2}", out.name),
        "new original owner09 actual output required",
    )
    config_raw = controller.read(out / "sealed-config.json")
    cfg = json.loads(config_raw)
    admitted = json.loads(controller.read(out / "admission.json"))
    boundary = controller.load_fixed("boundary")
    require(
        cfg["output"] == str(out)
        and cfg["profile"] == controller.PROFILE
        and cfg["unit"] == controller.UNIT
        and boundary.boot() == cfg["boot_id"]
        and os.environ.get("JEV_COMMAND_NONCE") == cfg["nonce"],
        "current original owner09 selection required",
    )
    require(
        admitted["config_sha256"] == hashlib.sha256(config_raw).hexdigest()
        and admitted["nonce"] == cfg["nonce"]
        and re.fullmatch(r"[0-9a-f]{32}", admitted["invocation_id"]),
        "current original admission differs",
    )
    controller.sources_match(cfg["sources"])
    spec = controller.command_selection(
        cfg["selection"], cfg["selection_sha256"], cfg["base_sources"]
    )
    require(len(spec["commands"]) == 1, "one directly selected fit child required")
    wrapper = boundary.process(admitted["actor"]["pid"])
    require(
        controller.stable_process(wrapper) == controller.stable_process(admitted["actor"]),
        "admitted wrapper changed",
    )
    me = controller.process_identity(boundary, os.getpid())
    require(
        all(me["process"][key] == wrapper[key] for key in ("pgid", "sid", "uid", "cgroup"))
        and me["process"]["cgroup"] == controller.selection(cfg, boundary).cgroup,
        "actual fit child is outside the admitted wrapper cohort",
    )
    expected = {
        "argv": [s.replace("{OUTPUT}", str(out)) for s in spec["commands"][0]["argv"]],
        "cwd": spec["commands"][0]["cwd"].replace("{OUTPUT}", str(out)),
    }
    require(
        me["argv"] == expected["argv"] and me["cwd"] == expected["cwd"] == str(ROOT),
        "selected fit argv/cwd differs",
    )
    # Parent writes -started after Popen.  Do not mistake this bounded race for
    # a missing child proof, or accept intent alone as actual child identity.
    until = time.monotonic() + 10
    while not (out / "command-01-started.json").exists():
        check_deadline()
        require(time.monotonic() < until, "bounded original child-start observation wait")
        time.sleep(0.01)
    started = json.loads(controller.read(out / "command-01-started.json"))
    intent = json.loads(controller.read(out / "command-01-intent.json"))
    require(
        started["pid"] == os.getpid()
        and started["observed_identity"] is not None
        and controller.same_identity(started["observed_identity"], me)
        and intent["argv"] == expected["argv"]
        and intent["cwd"] == expected["cwd"]
        and intent["shell"] is False
        and me["exe"] == intent["resolved_executable"],
        "original actual child/intent joins differ",
    )
    selected = controller.selection(cfg, boundary)
    group = boundary.CGROUP_ROOT / selected.cgroup.lstrip("/")
    require(
        (group / "memory.max").read_text().strip() == str(4 * 1024**3)
        and (group / "memory.swap.max").read_text().strip() == "0",
        "actual original 4GiB/no-swap limits required",
    )
    resources = child_properties(controller, boundary, selected, os.getpid())
    require(
        controller.stable_process(boundary.process(wrapper["pid"]))
        == controller.stable_process(wrapper),
        "wrapper changed during admission checks",
    )
    check_deadline()
    return out, cfg, spec, me, resources


def runtime_files(catalog, check_deadline):
    rows = catalog["critical_files_and_mapped_math_candidates"]
    require(
        type(rows) is list
        and 1 <= len(rows) <= 24
        and len({r["resolved_path"] for r in rows}) == len(rows)
        and sum(r["size_bytes"] for r in rows) <= 512 * 1024**2,
        "bounded selected runtime projection required",
    )
    for row in rows:
        path = Path(row["resolved_path"])
        require(
            path.is_absolute()
            and path == Path(row["path"]).resolve(strict=True)
            and path.resolve(strict=True) == path
            and type(row["size_bytes"]) is int
            and 0 < row["size_bytes"] <= 256 * 1024**2
            and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]),
            "explicit runtime file identity required",
        )
        digest, size = hashlib.sha256(), 0
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            require(stat.S_ISREG(before.st_mode), "regular selected runtime file required")
            while True:
                check_deadline()
                block = stream.read(1024**2)
                if not block:
                    break
                size += len(block)
                require(size <= row["size_bytes"], "runtime file grew")
                digest.update(block)
            after = os.fstat(stream.fileno())
        require(
            set(row["stat"]) == {"st_dev", "st_ino", "st_size", "st_mtime_ns"}
            and all(type(v) is int for v in row["stat"].values()),
            "typed runtime stat projection required",
        )
        require(
            size == row["size_bytes"]
            and digest.hexdigest() == row["sha256"]
            and all(
                getattr(before, key) == getattr(after, key) == getattr(path.stat(), key) == value
                for key, value in row["stat"].items()
            )
            and before.st_ctime_ns == after.st_ctime_ns == path.stat().st_ctime_ns,
            "selected runtime bytes/stat changed",
        )
    return rows


def gpu_runtime_files(catalog, check_deadline):
    """Authenticate excluded GPU metadata only; never open/hash a GPU payload."""
    rows = catalog[GPU_CATALOG_KEY]
    require(
        type(rows) is list
        and len(rows) + len(catalog["critical_files_and_mapped_math_candidates"]) <= 24,
        "bounded combined CPU/GPU catalog required",
    )
    names = set()
    for row in rows:
        require(
            set(row)
            == {
                "path",
                "size_bytes",
                "stat",
                "hashed",
                "included_in_CPU_projection",
                "execution_or_symbol_binding_proven",
            }
            and row["hashed"] is False
            and row["included_in_CPU_projection"] is False
            and row["execution_or_symbol_binding_proven"] is False,
            "original stat-only GPU candidate declaration required",
        )
        path = Path(row["path"])
        require(
            path.is_absolute()
            and str(path) not in names
            and path.resolve(strict=True) == path
            and GPU_MATH_LIBRARY.fullmatch(path.name),
            "explicit distinct GPU math basename required",
        )
        names.add(str(path))
        check_deadline()
        before = path.stat()
        after = path.stat()
        require(
            stat.S_ISREG(before.st_mode)
            and type(row["size_bytes"]) is int
            and row["size_bytes"] > 0
            and before.st_size == row["size_bytes"]
            and set(row["stat"]) == {"st_dev", "st_ino", "st_size", "st_mtime_ns"}
            and all(type(v) is int and v >= 0 for v in row["stat"].values())
            and all(
                getattr(before, key) == getattr(after, key) == value
                for key, value in row["stat"].items()
            ),
            "excluded GPU candidate stat changed",
        )
        check_deadline()
    return names


def mapped_math(raw):
    require(len(raw) <= 2 * 1024**2, "bounded own maps required")
    lines = raw.decode().splitlines()
    require(len(lines) <= 16384, "bounded own map rows required")
    cpu, gpu = set(), set()
    for line in lines:
        fields = line.split(None, 5)
        if (
            len(fields) == 6
            and fields[5].startswith("/")
            and MATH_LIBRARY.search(Path(fields[5]).name)
        ):
            require(not fields[5].endswith(" (deleted)"), "mapped mathematical candidate deleted")
            path = Path(fields[5]).resolve(strict=True)
            (gpu if GPU_MATH_LIBRARY.fullmatch(path.name) else cpu).add(str(path))
    return cpu, gpu


def numpy_runtime(loader, selection, check_deadline):
    prerequisite = loader.decode(
        loader.read_ref(selection["root_np5_prerequisite_ref"], check_deadline)
    )
    catalog_ref = loader.validate_np_acceptance(prerequisite, selection, check_deadline)
    pending = loader.decode(loader.read_ref(catalog_ref, check_deadline))
    require(
        pending["status"] == "PENDING_CLOSED_ROOT_ACCEPTANCE"
        and pending["training_eligible"] is False,
        "original nontraining catalog must remain unchanged",
    )
    catalog = pending["after"]
    require(
        catalog["schema"] == "owned-ML313-numpy-discovery-critical-projection-v1"
        and catalog["actual_discovery_after_native_load"] is True
        and catalog["preselected_catalog"] is False
        and catalog["full_DSO_or_transitive_closure_qualified"] is False,
        "selected NP5 runtime projection differs",
    )
    require(
        catalog["python"]
        == {
            "logical_path": sys.executable,
            "resolved_path": str(Path(sys.executable).resolve()),
            "version": list(sys.version_info[:3]),
        },
        "selected runtime Python differs",
    )
    rows = runtime_files(catalog, check_deadline)
    selected_gpu = gpu_runtime_files(catalog, check_deadline)
    np = importlib.import_module("numpy")
    require(
        np.__version__ == catalog["numpy_version"]
        and getattr(np.__config__, "CONFIG", None) == catalog["numpy_build_configuration"],
        "selected NumPy version/build differs",
    )
    modules = [
        np,
        np.__config__,
        importlib.import_module("numpy.version"),
        importlib.import_module("numpy._core._multiarray_umath"),
        importlib.import_module("numpy.linalg._umath_linalg"),
    ]
    selected_paths = {row["resolved_path"] for row in rows}
    require(
        all(
            str(Path(module.__file__).resolve(strict=True)) in selected_paths for module in modules
        ),
        "actual critical NumPy module origin differs",
    )
    with Path("/proc/self/maps").open("rb") as stream:
        maps = stream.read(2 * 1024**2 + 1)
    mapped, mapped_gpu = mapped_math(maps)
    require(
        mapped <= set(catalog["mapped_math_candidates"]) <= selected_paths,
        "actual mapped mathematical candidate lacks selected file identity",
    )
    require(mapped_gpu <= selected_gpu, "actual GPU math map lacks selected stat-only metadata")
    # Native qualification also mapped Torch libraries; a NumPy-only optimizer
    # need not map every historical candidate. No transitive DSO claim follows.
    runtime_files(catalog, check_deadline)
    gpu_runtime_files(catalog, check_deadline)
    return np, catalog, catalog_ref, {"cpu": sorted(mapped), "gpu_stat_only": sorted(mapped_gpu)}


def execute(selection_path, owner_output):
    released()  # before plan/arguments' paths, dependencies, arrays or optimizer
    began = time.monotonic()

    def deadline():
        require(time.monotonic() - began <= 120, "whole owned fit120 exceeded")

    require(
        sys.flags.no_user_site
        and sys.dont_write_bytecode
        and not any(
            name.split(".")[0] in {"numpy", "torch", "transformers", "safetensors", "s1"}
            for name in sys.modules
        ),
        "fresh -B -s child without numerical/model imports required",
    )
    controller = source("_dsl01_fit_original_owner09", OWNER / "controller.py", OWNER_SHA)
    out, cfg, spec, identity, resources = owned(owner_output, controller, deadline)
    loader = source("_dsl01_operation_committed_train_loader", LOADER, LOADER_SHA256)
    require(loader.SOURCE_ONLY is False, "train loader source remains held")
    cap_path = loader.ordinary(selection_path)
    cap_bytes = controller.read(cap_path)
    require(
        hashlib.sha256(cap_bytes).hexdigest() == SELECTION_SHA256,
        "independently selected train capability SHA differs",
    )
    selection = loader.decode(cap_bytes)
    pool = {str(ROOT / row["path"]): row["sha256"] for row in cfg["sources"]}
    for path, sha in (
        (cap_path, SELECTION_SHA256),
        (LOADER, LOADER_SHA256),
        (CORE, CORE_SHA),
        (
            Path(__file__).absolute(),
            hashlib.sha256(controller.read(Path(__file__).absolute())).hexdigest(),
        ),
    ):
        require(
            pool.get(str(path)) == sha,
            "original owner spec did not select fit/capability/loader/core bytes",
        )
    require(
        selection["hyperparameters"] == loader.HYPERPARAMETERS
        and selection["projection"] == loader.PROJECTION
        and selection["fit_budget_seconds"] == 120,
        "fixed predeclared train policy required before NumPy",
    )
    loader.validate_operation_selection(selection)
    require(
        selection["profile"] == PROFILE and selection["alias"] == ALIAS,
        "selected operation-only alias/profile differs before NumPy",
    )
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "5"
    np, catalog, catalog_ref, mapped = numpy_runtime(loader, selection, deadline)
    core = source("_dsl01_frozen_train_head_core", CORE, CORE_SHA)
    data = loader.load_training(selection, np, core, check_deadline=deadline)
    require(
        data["aggregate"]["supervised_heads"] == 96
        and data["aggregate"]["total_verified_supervised_heads"] == 132
        and data["aggregate"]["supervised_target_heads_excluded"] == 36
        and data["aggregate"]["unused_heads_skipped"] == 252
        and len(data["row_map"]) == 96
        and len({row["case_id"] for row in data["row_map"]}) == 96
        and all(
            row["head_id"] == "operation" and row["legal_count"] == 8 for row in data["row_map"]
        )
        and data["row_mapping_sha256"]
        == data["aggregate"]["row_mapping_sha256"]
        == loader.row_mapping_sha256(data["row_map"]),
        "complete132-to96 operation-only row selection required before optimizer",
    )
    result = core.fit_arrays(
        data["features"],
        data["baseline"],
        data["counts"],
        data["targets"],
        check_deadline=deadline,
        **loader.HYPERPARAMETERS,
    )
    delta, losses = result["delta"], result["losses"]
    require(
        type(delta) is np.ndarray
        and delta.dtype == np.float32
        and delta.shape == (255, 1024)
        and bool(np.isfinite(delta).all()),
        "finite FP32 delta255x1024 required",
    )
    require(
        result["steps"] == 512
        and result["seed"] == 20261023
        and len(losses) == 512
        and all(type(value) in (int, float) and math.isfinite(value) for value in losses)
        and result["data_qualification_supplied"] is False,
        "fixed bounded core result differs",
    )
    loader.reauthenticate(data["pinned_refs"], deadline)
    runtime_files(catalog, deadline)
    gpu_runtime_files(catalog, deadline)
    controller.sources_match(cfg["sources"])
    controller.command_selection(cfg["selection"], cfg["selection_sha256"], cfg["base_sources"])
    require(
        hashlib.sha256(controller.read(cap_path)).hexdigest() == SELECTION_SHA256,
        "train capability changed",
    )
    target = out / "fit-operation-head"
    target.mkdir(exist_ok=False)
    (target / ".gitignore").write_text("*\n")
    buffer = io.BytesIO()
    np.save(buffer, delta, allow_pickle=False)
    delta_bytes = buffer.getvalue()
    with (target / "delta-pending.npy").open("xb") as stream:
        stream.write(delta_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    summary = {
        "schema": "dsl01-owned-operation-readout-pending-v1",
        "status": "PENDING_OPERATION_READOUT_ROOT_ACCEPTANCE",
        "profile": PROFILE,
        "alias": ALIAS,
        "training_strategy": "operation_only",
        "training_eligible": False,
        "selection_sha256": SELECTION_SHA256,
        "hyperparameters": loader.HYPERPARAMETERS,
        "projection": loader.PROJECTION,
        "train": data["aggregate"],
        "row_mapping_sha256": data["row_mapping_sha256"],
        "supervised_heads": data["aggregate"]["supervised_heads"],
        "total_verified_supervised_heads": data["aggregate"]["total_verified_supervised_heads"],
        "supervised_target_heads_excluded": data["aggregate"]["supervised_target_heads_excluded"],
        "unused_heads_skipped": data["aggregate"]["unused_heads_skipped"],
        "row_mapping_encoding": "UTF-8 JSON; ordered96 rows; sort_keys=true; separators=(',',':'); ensure_ascii=false; allow_nan=false",
        "runtime_catalog": catalog_ref,
        "mapped_math_candidates": mapped["cpu"],
        "mapped_gpu_math_candidates_stat_only": mapped["gpu_stat_only"],
        "full_DSO_or_transitive_closure_qualified": False,
        "process": identity,
        "owned_resources": resources,
        "original_owner_nonce": cfg["nonce"],
        "loss": {
            "steps": 512,
            "first": losses[0],
            "last": losses[-1],
            "minimum": min(losses),
            "maximum": max(losses),
        },
        "delta": {
            "path": "delta-pending.npy",
            "sha256": hashlib.sha256(delta_bytes).hexdigest(),
            "size_bytes": len(delta_bytes),
            "format": "npy_no_pickle",
            "dtype": "float32",
            "shape": [255, 1024],
        },
        "elapsed_before_final_write_seconds": time.monotonic() - began,
        "model_qualified": False,
        "browser_qualified": False,
        "benchmark_qualified": False,
        "dev_or_sealed_selected": False,
        "dev_used_for_model_selection": True,
        "optimizer_input_dev": False,
        "optimizer_input_sealed": False,
        "ranking_or_win_claim": False,
        "OS_access_sealing_claim": False,
        "scope": "Operation-only bounded delta SGD over96 true operation rows after all132 supervised rows and96 COMMITTED train states are verified. W0 stays frozen;36 supervised targets are excluded and252 unused heads skipped. Dev informed policy selection but supplies no optimizer input. Pending new original root terminal0, owner09 completion and authenticated closure acceptance;512 steps within120s is not prequalified. No quality or browser evidence.",
    }
    deadline()
    with (target / "fit-summary-pending.json").open("xb") as stream:
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
    deadline()  # includes final write and cleanup; no local COMMITTED authority
    return summary


def main(argv=None):
    released()  # deliberately precedes argument parser and all file IO
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--owner-output", type=Path, required=True)
    args = parser.parse_args(argv)
    execute(args.selection, args.owner_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
