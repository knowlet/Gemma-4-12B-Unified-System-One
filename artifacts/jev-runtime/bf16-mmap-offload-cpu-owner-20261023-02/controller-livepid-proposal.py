"""Small stdlib owner for one tiny true-Torch CPU storage-hook fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import runpy
import secrets
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[2]
UNIT = "jev-d3-mmap-cpu-20261023-01.service"
PROFILE = "bf16-mmap-offload-cpu-owned-v1"
ACTOR_PROFILE = "bf16-mmap-offload-cpu-fixture-v1"
ACTOR = ROOT / "artifacts/jev-runtime/bf16-mmap-offload-cpu-20261023-01/fixture.py"
TORCH_PYTHON = Path("/home/albert_hsieh/ComfyUI/.venv/bin/python")
REPO_PYTHON = ROOT / ".venv/bin/python"
GIB, MAX_SOURCE = 1024**3, 1024**2
FIXED = {
    "boundary": (
        "artifacts/jev-runtime/decision3-local-owner-terminal-20261022-01/owner_boundary.py",
        "7a50c47d6d7f696973c666cb7a40ce47dd0a7624667a18422c8c60c0db72e4e5",
    ),
    "observer": (
        "artifacts/jev-runtime/observer-system-services-20261022-01/comfy_observer.py",
        "fd977040d659770e26a8fd6cc2b94e6c5440a6a3d3f05099e07d7597d8309e45",
    ),
    "authority": (
        "artifacts/jev-runtime/observer-system-services-20261022-01/system-services.json",
        "a2446af344f084dcf556256fd04c9754fcba940f33b3866990de85a7374c0727",
    ),
}


def need(value, message):
    if not value:
        raise ValueError(message)


def ordinary(path, *, local=True):
    path = Path(path).absolute()
    need(
        ".." not in path.parts
        and not any(p.is_symlink() for p in (path, *path.parents))
        and (not local or path.is_relative_to(ROOT)),
        "Ordinary selected path",
    )
    return path


def read(path):
    with ordinary(path, local=False).open("rb") as stream:
        blob = stream.read(MAX_SOURCE + 1)
    need(len(blob) <= MAX_SOURCE, "Small selected metadata/source only")
    return blob


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def ref(path):
    path = ordinary(path)
    blob = read(path)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(blob), "size_bytes": len(blob)}


def encoded(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"


def load_fixed(name):
    label, digest = FIXED[name]
    path, blob = ROOT / label, read(ROOT / label)
    need(sha(blob) == digest, "Exact unchanged " + name + " authority")
    module = types.ModuleType("_mmap_cpu_" + name)
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(blob, str(path), "exec", dont_inherit=True), vars(module))
    return module


def sources_match(rows):
    need(type(rows) is list and 1 <= len(rows) <= 64, "At most64 small source pins")
    labels = set()
    for row in rows:
        need(
            set(row) == {"path", "sha256", "size_bytes"}
            and not Path(row["path"]).is_absolute()
            and row["path"] not in labels
            and 0 <= row["size_bytes"] <= MAX_SOURCE,
            "Distinct typed small source references",
        )
        labels.add(row["path"])
        need(ref(ROOT / row["path"]) == row, "Selected small source bytes unchanged")
    return True


def actor_selection(path, digest, local_sources):
    blob = read(ordinary(path))
    need(sha(blob) == digest, "Independently selected fixture manifest SHA")
    value = json.loads(blob)
    need(
        set(value) == {"profile", "sources", "versions", "binary_stats"}
        and value["profile"] == ACTOR_PROFILE,
        "Exact tiny actor selection",
    )
    need(
        type(value["sources"]) is dict
        and set(value["versions"]) == {"torch", "accelerate", "peft", "safetensors"},
        "Explicit critical source/version set",
    )
    need(
        value["sources"].get("entry") == {**ref(ACTOR), "path": str(ACTOR)},
        "Manifest binds exact fixed actor source",
    )
    labels = {str(ROOT / row["path"]) for row in local_sources}
    for row in value["sources"].values():
        need(
            set(row) == {"path", "sha256", "size_bytes"}
            and Path(row["path"]).is_absolute()
            and 0 <= row["size_bytes"] <= MAX_SOURCE,
            "Small absolute critical source ref",
        )
        data = read(row["path"])
        need(
            len(data) == row["size_bytes"] and sha(data) == row["sha256"],
            "Selected critical actor/runtime source bytes",
        )
        labels.add(row["path"])
    need(
        len(labels) <= 64 and 1 <= len(value["binary_stats"]) <= 32,
        "Bounded critical refs/stat population, not full runtime",
    )
    for row in value["binary_stats"]:
        need(
            set(row) == {"path", "stat"}
            and Path(row["path"]).is_absolute()
            and set(row["stat"]) == {"st_dev", "st_ino", "st_size", "st_mtime_ns"},
            "Explicit binary stat-only identity",
        )
        actual = ordinary(row["path"], local=False).stat()
        need(
            {k: getattr(actual, k) for k in row["stat"]} == row["stat"],
            "Selected large binary stat unchanged",
        )
    return value


def reserve(total):
    need(type(total) is int and total > 0, "Actual positive host total")
    return max(24 * GIB, (total + 4) // 5)


def host(observer, boot_id, total=None, *, initial=False):
    row = observer.observe(boot_id)
    required = None
    try:
        need(
            row["boot_id"] == boot_id and (total is None or row["mem_total_bytes"] == total),
            "Same actual boot/host",
        )
        need(
            row["comfy_state"] in {"http_idle", "absent_confirmed"}
            and row["comfy_running"] == row["comfy_pending"] == 0,
            "Original read-only Comfy idle gate",
        )
        required = reserve(row["mem_total_bytes"]) + (2 * GIB if initial else 0)
        need(
            type(row["mem_available_bytes"]) is int and row["mem_available_bytes"] >= required,
            "Original whole-host reserve" + (" plus tiny2GiB" if initial else ""),
        )
    except BaseException as error:
        error.evidence = {
            "observation": row,
            "required_available_bytes": required,
            "initial": initial,
        }
        raise
    return {
        "observation": row,
        "reserve_bytes": reserve(row["mem_total_bytes"]),
        "required_available_bytes": required,
        "initial": initial,
    }


def failure(error):
    def captured(value):
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        return value[:32768] if isinstance(value, str) else value

    return {
        "type": type(error).__name__,
        "message": str(error),
        "evidence": getattr(error, "evidence", None),
        "returncode": getattr(error, "returncode", None),
        "stdout": captured(getattr(error, "stdout", None)),
        "stderr": captured(getattr(error, "stderr", None)),
    }


def keep(primary, errors, phase, error):
    row = {"phase": phase, **failure(error)}
    if primary is None:
        return row
    errors.append(row)
    return primary


def environment(cfg):
    return {
        "PATH": os.defpath,
        "LANG": "C.UTF-8",
        "PYTHONPATH": str(ROOT / "artifacts/jev-runtime/overlay"),
        "CUDA_VISIBLE_DEVICES": "",
        "ACCELERATE_USE_CPU": "true",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "USE_TF": "0",
        "USE_FLAX": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "XDG_CACHE_HOME": str(Path(cfg["output"]) / "cache"),
        "TMPDIR": str(Path(cfg["output"]) / "tmp"),
        "JEV_MMAP_NONCE": cfg["nonce"],
    }


def selection(cfg, owner):
    return owner.Selection(UNIT, cfg["nonce"], cfg["boot_id"], 2 * GIB, 30, 3)


def stable_process(row):
    return {key: row[key] for key in ("pid", "start_ticks", "pgid", "sid", "uid", "cgroup")}


def same_identity(left, right):
    return stable_process(left["process"]) == stable_process(right["process"]) and all(
        left[key] == right[key] for key in ("exe", "argv", "cwd")
    )


def process_identity(owner, pid):
    base = Path("/proc") / str(pid)
    before = owner.process(pid)
    value = {
        "process": before,
        "exe": str((base / "exe").resolve(strict=True)),
        "argv": [part.decode() for part in (base / "cmdline").read_bytes().split(b"\0") if part],
        "cwd": str((base / "cwd").resolve(strict=True)),
    }
    need(
        stable_process(owner.process(pid)) == stable_process(before),
        "Stable actual process across exec identity observation",
    )
    return value


def properties(owner, selected, pid):
    keys = (
        "CPUQuotaPerSecUSec",
        "AllowedCPUs",
        "LimitCORE",
        "LimitNOFILE",
        "LimitFSIZE",
        "WorkingDirectory",
        "MemoryPeak",
    )
    text = owner.command(
        "systemctl", "--user", "show", UNIT, *("--property=" + k for k in keys), timeout=1
    )
    row = dict(line.split("=", 1) for line in text.splitlines())
    need(
        all(
            row.get(k) == v
            for k, v in {
                "CPUQuotaPerSecUSec": "1s",
                "AllowedCPUs": "0",
                "LimitCORE": "0",
                "LimitNOFILE": "512",
                "LimitFSIZE": "8388608",
                "WorkingDirectory": str(ROOT),
            }.items()
        ),
        "Actual extra unit resource properties",
    )
    group = owner.CGROUP_ROOT / selected.cgroup.lstrip("/")
    quota = (group / "cpu.max").read_text().split()
    need(
        len(quota) == 2
        and quota[0].isdigit()
        and quota[1].isdigit()
        and int(quota[0]) == int(quota[1]) > 0
        and (group / "cpuset.cpus.effective").read_text().strip() == "0",
        "Actual CPU100% and CPU0 kernel readback",
    )
    limits = (Path("/proc") / str(pid) / "limits").read_text()
    for name, number, unit in (
        ("Max core file size", 0, "bytes"),
        ("Max file size", 8388608, "bytes"),
        ("Max open files", 512, "files"),
    ):
        need(
            re.search(
                r"(?m)^"
                + name
                + r"\s+"
                + str(number)
                + r"\s+"
                + str(number)
                + r"\s+"
                + unit
                + r"$",
                limits,
            ),
            "Actual process limit: " + name,
        )
    peak = (group / "memory.peak").read_text().strip()
    need(peak.isdigit(), "Actual kernel memory.peak")
    return {
        "unit": row,
        "kernel_memory_peak_bytes": int(peak),
        "kernel_cpu_max": quota,
        "kernel_allowed_cpus": "0",
    }


def ensure_absent(owner, selected):
    need(
        owner.status(selected).get("LoadState") == "not-found"
        and not (owner.CGROUP_ROOT / selected.cgroup.lstrip("/")).exists(),
        "Fresh exact unit and cgroup absent; never reuse",
    )


def actor_argv(cfg):
    return [
        str(ACTOR),
        "--run",
        "--profile",
        ACTOR_PROFILE,
        "--selection",
        cfg["selection"],
        "--selection-sha256",
        cfg["selection_sha256"],
        "--entry-sha256",
        cfg["actor_entry_sha256"],
        "--output",
        str(Path(cfg["output"]) / "fixture"),
    ]


def bootstrap_argv(cfg, digest, executable, stage):
    return [
        str(executable),
        "-B",
        "-s",
        str(HERE / "controller.py"),
        "--internal-stage",
        stage,
        "--config",
        str(Path(cfg["output"]) / "sealed-config.json"),
        "--config-sha256",
        digest,
    ]


def bootstrap(cfg, digest, stage, owner):
    selected = selection(cfg, owner)
    need(
        os.environ.get("JEV_MMAP_NONCE") == cfg["nonce"] and owner.boot() == cfg["boot_id"],
        "Exact bootstrap nonce/boot",
    )
    sources_match(cfg["sources"])
    me = process_identity(owner, os.getpid())
    p = me["process"]
    need(
        p["pid"] == p["pgid"] == p["sid"]
        and p["cgroup"] == selected.cgroup
        and p["uid"] == os.getuid()
        and me["cwd"] == str(ROOT),
        "Actual selected bootstrap session/cgroup",
    )
    out = ordinary(cfg["output"])
    need("torch" not in sys.modules, "No true Torch import before parent admission")
    identity_file, admission_file = (
        ("bootstrap-identity.json", "admission.json")
        if stage == "wait"
        else ("exec-identity.json", "exec-admission.json")
    )
    owner.save(
        out,
        identity_file,
        {
            "config_sha256": digest,
            "nonce": cfg["nonce"],
            "identity": me,
            "torch_not_imported": True,
            "sys_executable": sys.executable,
            "python_flags": {
                "no_user_site": sys.flags.no_user_site,
                "dont_write_bytecode": sys.dont_write_bytecode,
            },
        },
    )
    until = time.monotonic() + 10
    while not (out / admission_file).exists():
        need(time.monotonic() < until, "Bounded parent admission wait")
        time.sleep(0.01)
    admitted = json.loads(read(out / admission_file))
    need(
        admitted["config_sha256"] == digest
        and admitted["nonce"] == cfg["nonce"]
        and stable_process(admitted["actor"]) == stable_process(p)
        and re.fullmatch(r"[0-9a-f]{32}", admitted["invocation_id"]),
        "Exact parent admission identity",
    )
    sources_match(cfg["sources"])
    if stage == "wait":
        os.execve(
            str(TORCH_PYTHON), bootstrap_argv(cfg, digest, TORCH_PYTHON, "actor"), environment(cfg)
        )
    code, error = 0, None
    try:
        sys.argv = actor_argv(cfg)
        runpy.run_path(str(ACTOR), run_name="__main__")
    except SystemExit as result:
        code = result.code if type(result.code) is int else (0 if result.code is None else 1)
        if code:
            error = {"type": "SystemExit", "message": str(result.code)}
    except BaseException as result:
        code, error = 1, failure(result)
    owner.save(
        out,
        "actor-exit.json",
        {
            "config_sha256": digest,
            "actor": p,
            "entry_exit_code": code,
            "error": error,
            "postreceipt_hold_seconds": 5,
            "reason": "Keep verified main live for original authenticated boundary.close; not an actor success exit claim",
        },
    )
    time.sleep(5)
    # Success must not become active/exited under RemainAfterExit if parent was
    # unable to close in time. This bounded fallback is an explicit owner failure.
    return 125


def finalize(owner, boundary, attempted, cfg, observer, out, primary, state):
    errors, closure, final_host, source_ok = [], None, None, False
    if attempted:
        try:
            keys = (
                "Description",
                "InvocationID",
                "MainPID",
                "ExecMainPID",
                "ExecMainCode",
                "ExecMainStatus",
                "Result",
                "MemoryPeak",
            )
            raw = owner.command(
                "systemctl",
                "--user",
                "show",
                UNIT,
                *("--property=" + key for key in keys),
                timeout=1,
            )
            row = dict(line.split("=", 1) for line in raw.splitlines())
            state["preclose_exit_memory_raw"] = row
            need(
                row.get("Description") == cfg["nonce"]
                and boundary.invocation is not None
                and row.get("InvocationID") == boundary.invocation,
                "Memory/exit observation belongs to exact admitted invocation",
            )
            state["preclose_exit_memory_authenticated"] = True
            need(boundary.actor is not None and row.get("MainPID") == str(boundary.actor["pid"]), "Postreceipt hold must remain live until authenticated close; expiry125 is owner failure")
            state["preclose_main_live"] = True
        except BaseException as error:
            primary = keep(primary, errors, "preclose_exit_memory", error)
        try:
            closure = boundary.close(primary_error=primary)
        except BaseException as error:
            primary = keep(primary, errors, "boundary_close", error)
    try:
        sources_match(cfg["sources"])
        actor_selection(cfg["selection"], cfg["selection_sha256"], cfg["sources"])
        source_ok = True
    except BaseException as error:
        primary = keep(primary, errors, "final_source", error)
    try:
        final_host = host(observer, cfg["boot_id"], cfg.get("mem_total_bytes"))
    except BaseException as error:
        primary = keep(primary, errors, "final_host", error)
    completed = bool(
        primary is None
        and state.get("actor_success")
        and source_ok
        and final_host
        and closure
        and closure.get("whole_unit_exit_confirmed")
        and closure.get("observed_cgroup_cohort_exit_confirmed")
    )
    owner.save(
        out,
        "receipt.json",
        {
            "profile": PROFILE,
            "selected_config": cfg,
            "completed": completed,
            "primary_error": primary,
            "secondary_errors": errors,
            "closure": closure,
            "final_host": final_host,
            "sources_unchanged": source_ok,
            "state": state,
            "unit_attempted": attempted,
            "tiny_true_cpu_only": True,
            "large_model_GPU_20s_or_quality_qualified": False,
            "whole_tree_exit_confirmed": False,
            "comfy_service_modified": False,
        },
    )
    return 0 if completed else 1


def run(args):
    need(
        args.expected_entry_sha256 == sha(read(HERE / "controller.py")),
        "Explicit independently reviewed controller entry SHA",
    )
    need(args.selection is not None, "Explicit fixture selection path")
    need(
        re.fullmatch(r"[0-9a-f]{64}", args.actor_entry_sha256 or "")
        and re.fullmatch(r"[0-9a-f]{64}", args.selection_sha256 or ""),
        "Explicit actor and selection SHA",
    )
    need(sha(read(ACTOR)) == args.actor_entry_sha256, "Selected actor entry SHA")
    selected_manifest = ordinary(args.selection)
    need(
        selected_manifest.parent == ACTOR.parent, "Selection belongs to fixed tiny actor directory"
    )
    refs = [
        ref(HERE / "controller.py"),
        ref(ACTOR),
        ref(selected_manifest),
        *(ref(ROOT / label) for label, _ in FIXED.values()),
    ]
    sources_match(refs)
    actor_selection(selected_manifest, args.selection_sha256, refs)
    for name, (label, expected) in FIXED.items():
        need(sha(read(ROOT / label)) == expected, "Exact initial " + name + " authority")
    owner, observer = load_fixed("boundary"), load_fixed("observer")
    out = ordinary(args.output)
    need(
        out.parent == HERE and re.fullmatch(r"actual-[0-9]{2}", out.name) and not out.exists(),
        "Fresh scoped controller output",
    )
    cfg = {
        "profile": PROFILE,
        "unit": UNIT,
        "nonce": "jev-d3-boundary-" + secrets.token_hex(16),
        "boot_id": owner.boot(),
        "output": str(out),
        "sources": refs,
        "selection": str(selected_manifest),
        "selection_sha256": args.selection_sha256,
        "actor_entry_sha256": args.actor_entry_sha256,
    }
    selected = selection(cfg, owner)
    ensure_absent(owner, selected)
    out.mkdir(mode=0o700)
    (out / "tmp").mkdir(mode=0o700)
    boundary = owner.UnitBoundary(selected, out, refs)
    attempted, primary, state = False, None, {}
    try:
        initial = host(observer, cfg["boot_id"], initial=True)
        cfg["mem_total_bytes"] = initial["observation"]["mem_total_bytes"]
        state["initial_host"] = initial
        owner.save(out, "sealed-config.json", cfg)
        digest = sha(read(out / "sealed-config.json"))
        props = {
            **selected.properties(),
            "RuntimeMaxSec": "30",
            "TimeoutStopSec": "3",
            "WorkingDirectory": str(ROOT),
            "CPUQuota": "100%",
            "AllowedCPUs": "0",
            "LimitCORE": "0",
            "LimitNOFILE": "512",
            "LimitFSIZE": "8388608",
            "StandardOutput": "append:" + str(out / "actor.stdout"),
            "StandardError": "append:" + str(out / "actor.stderr"),
        }
        del props["RuntimeMaxUSec"], props["TimeoutStopUSec"]
        command = [
            "systemd-run",
            "--user",
            "--unit=" + UNIT,
            *("--property=" + k + "=" + v for k, v in props.items()),
            "/usr/bin/env",
            "-i",
            *(k + "=" + v for k, v in environment(cfg).items()),
            *bootstrap_argv(cfg, digest, REPO_PYTHON, "wait"),
        ]
        owner.save(
            out,
            "launch-intent.json",
            {
                "argv": command,
                "source_count": len(refs),
                "runtime_seconds": 30,
                "memory_max_bytes": 2 * GIB,
                "memory_swap_max_bytes": 0,
            },
        )
        attempted = True
        state["systemd_run_stdout"] = owner.command(*command, timeout=5)
        actor = None
        for name, admission, executable in (
            ("bootstrap-identity.json", "admission.json", REPO_PYTHON),
            ("exec-identity.json", "exec-admission.json", TORCH_PYTHON),
        ):
            until = time.monotonic() + 10
            while not (out / name).exists():
                need(time.monotonic() < until, "Bounded bootstrap/exec readiness")
                host(observer, cfg["boot_id"], cfg["mem_total_bytes"])
                time.sleep(0.02)
            identity = json.loads(read(out / name))
            rb = (
                boundary.verify()
                if actor is None
                else {"actor": owner.process(actor["pid"]), "unit": boundary.owned_status()}
            )
            need(
                actor is None or stable_process(actor) == stable_process(rb["actor"]),
                "Exec preserves original admitted PID/start/session/cgroup",
            )
            actor = rb["actor"]
            stage = "wait" if name == "bootstrap-identity.json" else "actor"
            need(
                identity["config_sha256"] == digest
                and identity["nonce"] == cfg["nonce"]
                and stable_process(identity["identity"]["process"]) == stable_process(actor)
                and identity["torch_not_imported"] is True
                and identity["identity"]["exe"] == str(executable.resolve(strict=True))
                and identity["identity"]["argv"] == bootstrap_argv(cfg, digest, executable, stage)
                and identity["sys_executable"] == str(executable)
                and identity["python_flags"] == {"no_user_site": 1, "dont_write_bytecode": True},
                "Actual pre-Torch PID/start/exec identity",
            )
            need(
                same_identity(identity["identity"], process_identity(owner, actor["pid"])),
                "Fresh exact actual exec identity",
            )
            props_readback = properties(owner, selected, actor["pid"])
            host(observer, cfg["boot_id"], cfg["mem_total_bytes"])
            sources_match(refs)
            state[name] = {"identity": identity, "properties": props_readback}
            owner.save(
                out,
                admission,
                {
                    "config_sha256": digest,
                    "nonce": cfg["nonce"],
                    "invocation_id": boundary.invocation,
                    "actor": actor,
                },
            )
        until = time.monotonic() + 30
        state["runtime_observations"] = []
        state["runtime_observation_count"] = 0
        while not (out / "actor-exit.json").exists():
            need(time.monotonic() < until, "Tiny actor deadline; unit runtime remains original30s")
            row = boundary.owned_status()
            need(
                row.get("MainPID") == str(actor["pid"])
                and row.get("ActiveState") == "active"
                and row.get("SubState") == "running",
                "Selected main remains live",
            )
            boundary.members()
            observation = {
                "host": host(observer, cfg["boot_id"], cfg["mem_total_bytes"]),
                "properties": properties(owner, selected, actor["pid"]),
            }
            state["runtime_observation_count"] += 1
            if len(state["runtime_observations"]) < 64:
                state["runtime_observations"].append(observation)
            else:
                state["runtime_observations"][-1] = observation
            state["runtime_observation_retention"] = (
                "First63 plus latest when more than64; every observation still gates execution"
            )
            time.sleep(0.05)
        result = json.loads(read(out / "actor-exit.json"))
        state["actor_exit"] = result
        state["last_live_properties"] = properties(owner, selected, actor["pid"])
        need(
            result["config_sha256"] == digest
            and stable_process(result["actor"]) == stable_process(actor)
            and result["entry_exit_code"] == 0
            and result["error"] is None,
            "Tiny actor entry completed before authenticated stop",
        )
        receipt = json.loads(read(out / "fixture/receipt.json"))
        need(
            receipt["status"] == "completed_mechanism_observations"
            and receipt["passed"] is True
            and receipt["CPU_ONLY"] is True
            and receipt["mechanism_checks_completed"] is True
            and receipt["source_integrity_final"] is True
            and receipt["GPU_transfer_measured"] is False
            and receipt["large_model_runtime_or_quality_qualified"] is False
            and receipt["CUDA"]["attempts"] == []
            and receipt["CUDA"]["initialized_final"] is False,
            "Tiny true CPU mechanism only",
        )
        state["actor_receipt"] = receipt
        state["actor_success"] = True
    except BaseException as error:
        primary = {"phase": "controller", **failure(error)}
    finally:
        result = finalize(owner, boundary, attempted, cfg, observer, out, primary, state)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--internal-stage", choices=("wait", "actor"))
    parser.add_argument("--expected-entry-sha256")
    parser.add_argument("--actor-entry-sha256")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--selection-sha256")
    parser.add_argument("--output", type=Path, default=HERE / "actual-01")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--config-sha256")
    args = parser.parse_args(argv)
    if args.internal_stage:
        blob = read(args.config)
        need(sha(blob) == args.config_sha256, "Exact sealed bootstrap config SHA")
        cfg = json.loads(blob)
        need(
            cfg["profile"] == PROFILE
            and cfg["unit"] == UNIT
            and ordinary(cfg["output"]).parent == HERE,
            "Fixed bootstrap profile/output",
        )
        return bootstrap(cfg, args.config_sha256, args.internal_stage, load_fixed("boundary"))
    if not args.run:
        print(
            json.dumps(
                {
                    "mode": "DRY_SOURCE_ONLY",
                    "unit_started": False,
                    "torch_imported": False,
                    "unit": UNIT,
                    "memory_max_bytes": 2 * GIB,
                    "runtime_seconds": 30,
                    "model_or_quality_qualified": False,
                }
            )
        )
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
