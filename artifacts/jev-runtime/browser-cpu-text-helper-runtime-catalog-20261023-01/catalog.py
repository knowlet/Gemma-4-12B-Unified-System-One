"""Dry-default selected native identity catalog; no model loading or generation."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import re
import stat
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[2]
OWNER = ROOT / "artifacts/jev-runtime/llama-cpp-owner-20261023-05/controller.py"
OWNER_SHA = "81a492bb00a430c8728e0fcafe94a4220961312e521e8c0d7863c7f601c8ef2e"
UNIT = "jev-d3-llama-cpp-cpu4-3g-long-20261023-05.service"
OWNER_PROFILE = "llama-cpp-cpu4-3g-long-command-owned-v1"
PROFILE = "owned-helper-runtime-native-catalog-v1"
PYTHON = Path("/home/albert_hsieh/ComfyUI/.venv/bin/python")
TORCH = PYTHON.parent.parent / "lib/python3.13/site-packages/torch"
OVERLAY = ROOT / "artifacts/jev-runtime/overlay"
NATIVE_PATHS = (
    TORCH / "_C.cpython-313-aarch64-linux-gnu.so",
    *(
        TORCH / "lib" / name
        for name in (
            "libtorch_cpu.so",
            "libtorch_python.so",
            "libc10.so",
            "libtorch_global_deps.so",
            "libtorch.so",
            "libshm.so",
        )
    ),
    OVERLAY / "tokenizers/tokenizers.abi3.so",
    OVERLAY / "safetensors/_safetensors_rust.abi3.so",
)
KNOWN_SHA = {
    str(NATIVE_PATHS[0]): "575d3e4b00f6766f1ea5139b9e2b533c6e768d02cb8b20077a5ca576f83d208a",
    str(NATIVE_PATHS[-2]): "eb4d4fab7a667086711cfc6b7f0fba6c0275c2059da0f07d52c917bfa854a20d",
}
MODULES = ("torch", "transformers", "tokenizers", "safetensors", "accelerate")
EXTENSIONS = {
    "torch._C": str(NATIVE_PATHS[0]),
    "tokenizers.tokenizers": str(NATIVE_PATHS[-2]),
    "safetensors._safetensors_rust": str(NATIVE_PATHS[-1]),
}
STAT_KEYS = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
CHUNK, SMALL, WALL = 1024**2, 1024**2, 600
MAX_NATIVE_BYTES = 512 * 1024**2
ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "ACCELERATE_USE_CPU": "true",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
CPU_ENV = {
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}


def need(value, message):
    if not value:
        raise ValueError(message)


def ordinary(path, *, output=False):
    path = Path(path).absolute()
    need(
        ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "Ordinary selected path only",
    )
    need(not output or path.is_relative_to(ROOT), "Owned repository output only")
    return path


def small(path):
    with ordinary(path).open("rb") as stream:
        blob = stream.read(SMALL + 1)
    need(len(blob) <= SMALL, "Selected small metadata/source only")
    return blob


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def file_stat(path):
    row = ordinary(path).stat()
    need(stat.S_ISREG(row.st_mode), "Selected ordinary regular file")
    return {key: getattr(row, key) for key in STAT_KEYS}


def check_ref(row):
    need(set(row) == {"path", "size_bytes", "sha256"}, "Exact small source ref")
    blob = small(row["path"])
    need(len(blob) == row["size_bytes"] and digest(blob) == row["sha256"], "Selected source drift")
    return blob


def check_stat(row):
    need(
        row["size_bytes"] == row["stat"]["st_size"] and file_stat(row["path"]) == row["stat"],
        "Selected stat drift",
    )


def write(path, value):
    blob = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    need(len(blob) <= SMALL, "Bounded catalog receipt")
    fd, pending = tempfile.mkstemp(prefix=".catalog-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pending, path)
    finally:
        Path(pending).unlink(missing_ok=True)


def elapsed(began):
    value = time.monotonic() - began
    need(0 <= value <= WALL, "Catalog whole600s deadline")
    return value


def load_plan(path, expected):
    blob = small(path)
    need(digest(blob) == expected, "Exact root-selected catalog plan SHA")
    plan = json.loads(blob)
    need(
        plan["profile"] == PROFILE
        and plan["whole_seconds"] == WALL
        and plan["native_total_max_bytes"] == MAX_NATIVE_BYTES
        and plan["actor_sha256"] == digest(small(__file__)),
        "Fixed catalog/source policy",
    )
    need(
        plan["owner_source"] == {"path": str(OWNER), "sha256": OWNER_SHA, "size_bytes": 40319},
        "Exact unchanged owner05 source",
    )
    need(
        set(plan["module_origins"]) == set(MODULES) and set(plan["versions"]) == set(MODULES),
        "Exactly selected five runtime modules",
    )
    template = json.loads(check_ref(plan["pending_descriptor"]))
    need(
        plan["pending_descriptor"]["sha256"]
        == "b5e0c09b60b2de3dadfb48d98febc8e61c236bb9e61a0d03ae9b126a26223e26"
        and plan["module_origins"] == template["module_origins"]
        and plan["versions"] == template["versions"]
        and plan["runtime_sources"] == template["runtime_files"],
        "Exact inherited runtime source roots/versions",
    )
    rows = plan["native_files"]
    need(
        type(rows) is list
        and len(rows) == len(NATIVE_PATHS)
        and {row["path"] for row in rows} == set(map(str, NATIVE_PATHS)),
        "Exact nine native paths",
    )
    need(sum(row["size_bytes"] for row in rows) <= MAX_NATIVE_BYTES, "Bounded native total")
    for row in rows:
        need(
            type(row["size_bytes"]) is int
            and 0 < row["size_bytes"] <= MAX_NATIVE_BYTES
            and set(row["stat"]) == set(STAT_KEYS)
            and all(type(v) is int and v >= 0 for v in row["stat"].values()),
            "Typed selected native stat",
        )
        expected_sha = row["expected_sha256"]
        need(
            expected_sha is None or re.fullmatch(r"[a-f0-9]{64}", expected_sha),
            "Expected SHA or null",
        )
        need(
            row["path"] not in KNOWN_SHA or expected_sha == KNOWN_SHA[row["path"]],
            "Historical owned Torch/tokenizer SHA retained",
        )
    refs = [plan["owner_source"], *plan["sources"], *plan["runtime_sources"]]
    need(
        1 <= len(refs) <= 64 and len({row["path"] for row in refs}) == len(refs),
        "Bounded distinct small source refs",
    )
    return plan


def load_guard(plan):
    blob = check_ref(plan["owner_source"])
    module = types.ModuleType("_catalog_unchanged_owner05")
    module.__file__ = str(OWNER)
    exec(compile(blob, str(OWNER), "exec"), module.__dict__)
    need(
        module.UNIT == UNIT
        and module.PROFILE == OWNER_PROFILE
        and module.fixed_memory() == 3221225472,
        "Unchanged owner05 resource profile",
    )
    return module


class Admission:
    def __init__(self, guard):
        self.guard = guard
        self.owner, self.observer = guard.load_fixed("boundary"), guard.load_fixed("observer")
        self.boot = self.owner.boot()
        self.selected = guard.selection(
            {
                "nonce": os.environ.get("JEV_COMMAND_NONCE"),
                "boot_id": self.boot,
                "runtime_seconds": 10800,
            },
            self.owner,
        )
        self.invocation = None
        self.main_identity = None
        self.actor_identity = None
        self.total = None
        self.last = 0.0

    def check(self, *, initial=False, force=False):
        if not initial and not force and time.monotonic() - self.last < 1:
            return None
        row = self.owner.status(self.selected)
        need(
            all(row.get(k) == v for k, v in self.selected.properties().items())
            and row.get("ControlGroup") == self.selected.cgroup
            and row.get("ActiveState") == "active"
            and row.get("SubState") == "running",
            "Actual exact live owner properties",
        )
        invocation = row.get("InvocationID", "")
        need(
            re.fullmatch(r"[a-f0-9]{32}", invocation)
            and (self.invocation is None or self.invocation == invocation),
            "Stable own invocation",
        )
        self.invocation = invocation
        me = self.guard.process_identity(self.owner, os.getpid())
        main = self.owner.process(int(row["MainPID"]))
        need(
            me["process"]["cgroup"] == main["cgroup"] == self.selected.cgroup
            and main["pid"] == main["pgid"] == main["sid"]
            and me["process"]["pgid"] == main["pgid"]
            and me["process"]["sid"] == main["sid"]
            and me["process"]["uid"] == main["uid"] == os.getuid(),
            "Own admitted wrapper cohort",
        )
        main_identity = self.guard.stable_process(main)
        actor_identity = self.guard.stable_process(me["process"])
        need(
            (self.main_identity is None or self.main_identity == main_identity)
            and (self.actor_identity is None or self.actor_identity == actor_identity),
            "Stable actual wrapper/catalog process identities",
        )
        self.main_identity, self.actor_identity = main_identity, actor_identity
        self.guard.main_affinity(self.owner, os.getpid())
        resources = self.guard.properties(self.owner, self.selected, main["pid"])
        host = self.guard.host(self.observer, self.boot, self.total, initial=initial)
        need(host["reserve_bytes"] == 26119172916, "Same original host reserve")
        self.total = host["observation"]["mem_total_bytes"]
        self.last = time.monotonic()
        return {"identity": me, "unit": row, "resources": resources, "host": host}


def hash_native(row, began, checkpoint):
    check_stat(row)
    started = time.monotonic()
    h, count = hashlib.sha256(), 0
    with ordinary(row["path"]).open("rb") as stream:
        need(
            {key: getattr(os.fstat(stream.fileno()), key) for key in STAT_KEYS} == row["stat"],
            "Opened native stat agrees with selection",
        )
        while block := stream.read(CHUNK):
            elapsed(began)
            checkpoint()
            count += len(block)
            need(count <= row["size_bytes"], "Bounded selected native bytes")
            h.update(block)
        need(
            {key: getattr(os.fstat(stream.fileno()), key) for key in STAT_KEYS} == row["stat"],
            "Native opened identity remains stable",
        )
    check_stat(row)
    need(count == row["size_bytes"], "Complete selected native stream")
    observed = h.hexdigest()
    need(
        row["expected_sha256"] is None or observed == row["expected_sha256"],
        "Known native content SHA differs",
    )
    return {
        "path": row["path"],
        "size_bytes": count,
        "sha256": observed,
        "stat": row["stat"],
        "elapsed_seconds": time.monotonic() - started,
    }


def mapped_native():
    blob = small(Path("/proc") / str(os.getpid()) / "maps")
    paths = set()
    for line in blob.decode().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and re.search(r"\.so(?:\.[0-9]+)*$", parts[5]):
            paths.add(parts[5])
    need(len(paths) <= 192, "Bounded observed native mapping population")
    return [{"path": path, "stat": file_stat(path)} for path in sorted(paths)]


def import_runtime(plan):
    need(not any(name in sys.modules for name in MODULES), "Fresh runtime process")
    sys.path.insert(0, str(OVERLAY))
    for name, expected in plan["module_origins"].items():
        found = importlib.util.find_spec(name)
        need(
            found is not None and str(Path(found.origin).resolve()) == expected,
            "Pre-import selected module origin",
        )
    modules = {name: importlib.import_module(name) for name in MODULES}
    identities = {}
    for name, module in modules.items():
        identity = {"origin": str(Path(module.__file__).resolve()), "version": module.__version__}
        need(
            identity == {"origin": plan["module_origins"][name], "version": plan["versions"][name]},
            "Actual module origin/version differs",
        )
        identities[name] = identity
    torch = modules["torch"]
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    need(
        torch.get_num_threads() == 4
        and torch.get_num_interop_threads() == 1
        and not torch.cuda.is_initialized(),
        "CPU4 runtime without CUDA context",
    )
    extensions = {
        name: str(Path(importlib.import_module(name).__file__).resolve()) for name in EXTENSIONS
    }
    need(extensions == EXTENSIONS, "Actual selected native extension origins")
    return identities, extensions, torch


def execute(args):
    began = time.monotonic()
    plan = load_plan(args.plan, args.plan_sha256)
    need(args.expected_entry_sha256 == digest(small(__file__)), "Explicit selected actor SHA")
    need(
        sys.flags.no_user_site
        and sys.dont_write_bytecode
        and all(os.environ.get(k) == v for k, v in ENV.items()),
        "Fresh -B -s CPU/offline environment",
    )
    need(
        plan["logical_interpreter"] == str(PYTHON)
        and sys.executable == plan["logical_interpreter"]
        and str(Path(sys.executable).resolve()) == plan["interpreter"]["path"],
        "Selected interpreter",
    )
    check_stat(plan["interpreter"])
    guard = load_guard(plan)
    admission = Admission(guard)
    initial = admission.check(initial=True)
    out = ordinary(args.output, output=True)
    need(
        out.is_relative_to(OWNER.parent) and not out.exists(), "Fresh owner05 child catalog output"
    )
    out.mkdir(mode=0o700)
    state = {
        "schema": PROFILE,
        "profile": PROFILE,
        "status": "running",
        "completed": False,
        "runtime_imported": False,
        "initial_admission": initial,
        "runtime_python": {
            "path": sys.executable,
            "resolved_path": plan["interpreter"]["path"],
            "stat": plan["interpreter"]["stat"],
            "version": list(sys.version_info[:3]),
        },
        "plan_sha256": args.plan_sha256,
        "hashes_before_import": [],
        "hashes_final": [],
        "model_loaded": False,
        "helper_qualified": False,
        "full_native_DSO_closure": False,
    }
    timer = None
    try:
        for row in [plan["owner_source"], *plan["sources"], *plan["runtime_sources"]]:
            check_ref(row)
        for row in plan["package_files"]:
            check_stat(row)  # Model/tokenizer data is never opened by this catalog.
        elapsed(began)

        def expire():
            try:
                write(
                    out / "timeout.json", {"status": "catalog_deadline_exceeded", "seconds": WALL}
                )
            finally:
                os._exit(124)

        timer = threading.Timer(max(0, WALL - (time.monotonic() - began)), expire)
        timer.daemon = True
        timer.start()
        os.environ.update(CPU_ENV)
        os.environ["HF_HOME"] = str(out / "hf-cache")
        for row in plan["native_files"]:
            state["hashes_before_import"].append(hash_native(row, began, admission.check))
            write(out / "progress.json", state)
        state["before_import_admission"] = admission.check(force=True)
        state["modules"], state["extension_origins"], torch = import_runtime(plan)
        state["runtime_imported"] = True
        state["mapped_native_DSOs"] = mapped_native()
        mapped = {row["path"] for row in state["mapped_native_DSOs"]}
        need(set(EXTENSIONS.values()) <= mapped, "Selected native extensions actually mapped")
        for row in plan["native_files"]:
            state["hashes_final"].append(hash_native(row, began, admission.check))
            write(out / "progress.json", state)
        need(
            [
                {k: v for k, v in row.items() if k != "elapsed_seconds"}
                for row in state["hashes_before_import"]
            ]
            == [
                {k: v for k, v in row.items() if k != "elapsed_seconds"}
                for row in state["hashes_final"]
            ],
            "Selected native hashes unchanged after import",
        )
        for row in [plan["owner_source"], *plan["sources"], *plan["runtime_sources"]]:
            check_ref(row)
        need(digest(small(args.plan)) == args.plan_sha256, "Final exact plan source authentication")
        for row in plan["package_files"]:
            check_stat(row)
        final_modules = {
            name: {
                "origin": str(Path(sys.modules[name].__file__).resolve()),
                "version": sys.modules[name].__version__,
            }
            for name in MODULES
        }
        need(
            final_modules == state["modules"]
            and torch.get_num_threads() == 4
            and torch.get_num_interop_threads() == 1
            and not torch.cuda.is_initialized(),
            "Same final CPU runtime identities without CUDA context",
        )
        state["final_admission"] = admission.check(force=True)
        check_stat(plan["interpreter"])
        need(
            state["final_admission"]["identity"]["exe"] == state["runtime_python"]["resolved_path"],
            "Final actual catalog interpreter identity",
        )
        state["runtime_source_files"] = plan["runtime_sources"]
        state["package_receipt"] = plan["package_receipt"]
        state["package_file_stats"] = plan["package_files"]
        state["runtime_binary_files"] = [
            {k: row[k] for k in ("path", "size_bytes", "sha256")} for row in state["hashes_final"]
        ]
        state["final_authentication"] = True
        state["status"] = "catalog_complete_runtime_identity_only"
        state["completed"] = True
        state["elapsed_before_final_write_seconds"] = elapsed(began)
        write(out / "receipt.json", state)
        elapsed(began)
        timer.cancel()
        elapsed(began)
        return 0
    except BaseException as primary:
        if timer is not None:
            try:
                timer.cancel()
            except BaseException as secondary:
                primary.add_note("Catalog watchdog cleanup also failed: " + repr(secondary))
        state["status"] = "failed_no_runtime_or_helper_qualification"
        state["completed"] = False
        state["error"] = {"type": type(primary).__name__, "message": str(primary)}
        try:
            write(out / "receipt.json", state)
        except BaseException as secondary:
            primary.add_note("Catalog failure receipt also failed: " + repr(secondary))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-entry-sha256")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "dry_source_only_no_runtime_import",
                    "model_loaded": False,
                    "native_payload_read": False,
                    "runtime_imported": False,
                }
            )
        )
        return 0
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
