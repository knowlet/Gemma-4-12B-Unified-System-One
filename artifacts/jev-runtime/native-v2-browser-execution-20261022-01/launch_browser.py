"""Own one unchanged full V7 current-NF4 browser arm; default dry."""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import resource
import stat
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[2]
CGROUP_ROOT = Path("/sys/fs/cgroup")
DRIVER = "scripts/jev_native_v2_browser.py"
PROFILE = "jev-current-nf4-v7-browser-owned-v1"
CAP, GIB, BASE_BYTES = 40 * 1024**3, 1024**3, 23_919_549_408
FORMATS = {"raw": "raw-json-v1", "v2": "shared-elements-v2"}
GUARD_DIRECTORY = "artifacts/jev-runtime/v2-action-training-prepare-20261021-01"
GUARD_INDEX = GUARD_DIRECTORY + "/selected-evidence-index-01.json"
GUARD_SOURCE = GUARD_DIRECTORY + "/run_compiler_tests.py"
OBSERVER_SOURCE = (
    "artifacts/jev-runtime/comfy-observer-vanished-context-20261022-01/comfy_observer.py"
)
MODEL_HUB = Path("artifacts/jev-local-hf/hub")
MODEL_REPOSITORY = MODEL_HUB / "models--knowlet--Gemma-4-12B-Unified-System-One"
MODEL_SNAPSHOT = MODEL_REPOSITORY / "snapshots/a66f836b56605039fe040f330180e336d19b3362"
MODEL_CACHE_NAMES = {
    "artifact-manifest.json",
    "chat_template.jinja",
    "config.json",
    "evaluation-report.json",
    "generation_config.json",
    "model.safetensors",
    "processor_config.json",
    "s1_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "training-receipt.json",
}
EXPECTED_BASE_CHECKPOINT = {
    "sha256": "11144621122123135eb79c9cfd0466a2c51f230109124aefe401cdbec8c71a53",
    "size_bytes": BASE_BYTES,
}


def require(value, message):
    if not value:
        raise ValueError(message)


def error_record(error):
    row = {"type": type(error).__name__, "message": str(error)}
    evidence = getattr(error, "evidence", None)
    if evidence is None:
        return row
    hidden = {
        "argv",
        "cmdline",
        "environ",
        "environment",
        "headers",
        "authorization",
        "cookie",
        "cookies",
        "api_key",
        "token",
        "password",
        "secret",
    }

    def redacted(value, depth=0):
        if depth > 16:
            return {"redacted": "depth_bound"}
        if type(value) is dict:
            result = {}
            for key, item in value.items():
                safe_digest = (
                    type(item) is dict
                    and set(item) == {"sha256", "size_bytes"}
                    and type(item["sha256"]) is str
                    and re.fullmatch(r"[0-9a-f]{64}", item["sha256"])
                    and type(item["size_bytes"]) is int
                    and item["size_bytes"] >= 0
                )
                result[str(key)] = (
                    "[REDACTED]"
                    if str(key).lower().replace("-", "_") in hidden and not safe_digest
                    else redacted(item, depth + 1)
                )
            return result
        if type(value) is list:
            return [redacted(item, depth + 1) for item in value]
        if value is None or type(value) in (str, int, float, bool):
            return value
        return {"unsupported_type": type(value).__name__}

    try:
        safe = redacted(evidence)
        blob = json.dumps(
            safe, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode()
        row["observation_evidence"] = (
            safe
            if len(blob) <= 64 * 1024
            else {
                "sha256": hashlib.sha256(blob).hexdigest(),
                "size_bytes": len(blob),
                "truncated": True,
            }
        )
    except (TypeError, ValueError, RecursionError):
        row["observation_evidence"] = {"unsupported_type": type(evidence).__name__}
    return row


def observe_with_evidence(guard, callback):
    def captured():
        try:
            return callback()
        except BaseException as error:
            return {"observation_error": error_record(error)}

    return guard.observe(captured)


def ordinary(path):
    path = Path(path).absolute()
    require(
        ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "Ordinary nontraversing owned path required",
    )
    return path


def pin(path):
    path = ordinary(path)
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while block := stream.read(1024**2):
            digest.update(block)
            size += len(block)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": digest.hexdigest(),
        "size_bytes": size,
    }


def authenticate(ref):
    require(type(ref) is dict and set(ref) == {"path", "sha256", "size_bytes"}, "Exact source ref")
    path = ordinary(ROOT / ref["path"])
    require(path.is_relative_to(ROOT) and pin(path) == ref, "Pinned source differs")
    return path


def verify_cache_aliases(plan, expected=None):
    """Bind every selected HF logical name and its bounded link chain; no content rehash."""
    require(
        plan.get("expected_base_checkpoint") == EXPECTED_BASE_CHECKPOINT,
        "Exact selected base checkpoint required",
    )
    aliases = plan.get("model_cache_aliases")
    require(
        type(aliases) is list and len(aliases) == len(MODEL_CACHE_NAMES),
        "Complete selected model cache aliases required",
    )
    snapshot = ordinary(ROOT / MODEL_SNAPSHOT)
    require(
        snapshot.is_dir() and {p.name for p in snapshot.iterdir()} == MODEL_CACHE_NAMES,
        "Exact selected snapshot file population",
    )
    observed, names = [], set()
    allowed = (ROOT / MODEL_REPOSITORY / "blobs", ROOT / MODEL_HUB / "blobs")
    for alias in aliases:
        require(
            type(alias) is dict and set(alias) == {"logical", "resolved"},
            "Exact logical/resolved cache alias schema",
        )
        logical, resolved = alias["logical"], alias["resolved"]
        require(
            all(
                type(r) is dict and set(r) == {"path", "sha256", "size_bytes"}
                for r in (logical, resolved)
            ),
            "Exact cache source references",
        )
        require(
            type(logical["path"]) is str and type(resolved["path"]) is str,
            "Relative selected cache paths required",
        )
        name = Path(logical["path"]).name
        require(
            name in MODEL_CACHE_NAMES
            and name not in names
            and logical["path"] == (MODEL_SNAPSHOT / name).as_posix(),
            "Exact distinct selected snapshot logical names",
        )
        names.add(name)
        require(
            logical["sha256"] == resolved["sha256"]
            and logical["size_bytes"] == resolved["size_bytes"]
            and type(resolved["size_bytes"]) is int
            and resolved["size_bytes"] > 0
            and type(resolved["sha256"]) is str
            and re.fullmatch(r"[0-9a-f]{64}", resolved["sha256"])
            and resolved in plan["sources"],
            "Pinned matching logical/resolved cache identity",
        )
        target = ordinary(ROOT / resolved["path"])
        require(
            any(target.is_relative_to(p) for p in allowed), "Cache target outside allowed blobs"
        )
        if name == "model.safetensors":
            require(
                {k: resolved[k] for k in EXPECTED_BASE_CHECKPOINT} == EXPECTED_BASE_CHECKPOINT,
                "Selected base checkpoint content differs",
            )
        selected, links = ROOT / logical["path"], []
        ordinary(selected.parent)
        while stat.S_ISLNK(selected.lstat().st_mode):
            require(len(links) < 4, "Bounded nonrecursive model cache alias chain")
            link = os.readlink(selected)
            require(0 < len(link) <= 4096, "Bounded model cache link text")
            links.append({"path": selected.relative_to(ROOT).as_posix(), "link_text": link})
            selected = Path(os.path.abspath(selected.parent / link))
            require(
                any(selected.is_relative_to(p) for p in allowed),
                "Cache alias hop outside allowed blobs",
            )
            ordinary(selected.parent)
        actual = selected.lstat()
        require(
            links
            and selected == target
            and stat.S_ISREG(actual.st_mode)
            and actual.st_size == resolved["size_bytes"],
            "Actual model cache alias target differs",
        )
        observed.append(
            {
                "logical": logical,
                "resolved": resolved,
                "links": links,
                "stat": {
                    "device": actual.st_dev,
                    "inode": actual.st_ino,
                    "size_bytes": actual.st_size,
                    "mtime_ns": actual.st_mtime_ns,
                    "ctime_ns": actual.st_ctime_ns,
                },
            }
        )
    require(names == MODEL_CACHE_NAMES, "Complete selected snapshot names")
    require(expected is None or observed == expected, "Model cache alias chain changed")
    return observed


def current_boot():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def memtotal():
    row = next(
        x for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemTotal:")
    )
    return int(row.split()[1]) * 1024


def reserve(total):
    require(type(total) is int and total > 0, "Positive actual MemTotal")
    return max(24 * GIB, (total + 4) // 5)


def run_paths(plan):
    directory = ordinary(ROOT / plan["run_directory"])
    require(
        directory.parent == HERE and directory.name == plan["arm"] + "-" + plan["attempt"],
        "Explicit fresh arm directory required",
    )
    require(
        plan["output"] == (directory / "browser").relative_to(ROOT).as_posix(),
        "Browser output must be the owned arm child",
    )
    return directory, directory / "browser"


def validate_plan(plan):
    require(plan["schema_id"] == PROFILE and plan["root"] == str(ROOT), "Exact local profile/root")
    require(
        plan["arm"] in FORMATS and re.fullmatch(r"[0-9]{2}", plan["attempt"]),
        "Explicit arm/attempt",
    )
    require(plan["prompt_format"] == FORMATS[plan["arm"]], "Matched explicit prompt format")
    wanted_unit = f"jev-current-nf4-{plan['arm']}-20261022-{plan['attempt']}.service"
    require(plan["unit"] == wanted_unit, "Explicit owned unit")
    wanted_cgroup = (
        f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{wanted_unit}"
    )
    require(
        plan["cgroup"] == wanted_cgroup and plan["unit_memory_max_bytes"] == CAP, "Exact cgroup/cap"
    )
    require(plan["boot_id"] == current_boot(), "Fresh same-boot plan")
    require(
        type(plan["nonce"]) is str and re.fullmatch(r"jev-nf4-browser-[0-9a-f]{32}", plan["nonce"]),
        "Fresh bounded owner nonce",
    )
    run_paths(plan)
    require(type(plan["sources"]) is list and plan["sources"], "Nonempty source pins")
    labels = [r["path"] for r in plan["sources"]]
    require(len(set(labels)) == len(labels), "Unique source pins")
    required = {
        DRIVER,
        GUARD_INDEX,
        GUARD_SOURCE,
        OBSERVER_SOURCE,
        str((HERE / "browser_entry.py").relative_to(ROOT)),
        str((HERE / "launch_browser.py").relative_to(ROOT)),
        plan["adapter"] + "/manifest.json",
    }
    require(required <= set(labels), "Owner/entry/driver/adapter manifest must be pinned")
    require(
        plan.get("comfy_observer") in plan["sources"]
        and plan["comfy_observer"]["path"] == OBSERVER_SOURCE,
        "Exact pinned fresh Comfy observer required",
    )
    for ref in plan["sources"]:
        authenticate(ref)
    verify_cache_aliases(plan)
    for key in ("adapter", "upstream"):
        path = ordinary(ROOT / plan[key])
        require(path.is_relative_to(ROOT) and path.is_dir(), "Local ordinary input directory")
    require(ordinary(plan["chromium"]).is_file(), "Selected local Chromium executable")
    return plan


def load_plan(path, digest):
    blob = ordinary(path).read_bytes()
    require(hashlib.sha256(blob).hexdigest() == digest, "Sealed plan SHA differs")
    return validate_plan(json.loads(blob))


def browser_arguments(plan):
    return [
        "--run",
        "--action-adapter",
        str(ROOT / plan["adapter"]),
        "--output-dir",
        str(ROOT / plan["output"]),
        "--prompt-format",
        plan["prompt_format"],
        "--upstream-path",
        str(ROOT / plan["upstream"]),
        "--chromium-executable",
        plan["chromium"],
        "--expected-boot-id",
        plan["boot_id"],
    ]


def save(directory, name, value):
    path = directory / name
    with path.open("x") as stream:
        os.chmod(path, 0o600)
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def command(*args, timeout=15):
    return subprocess.run(args, capture_output=True, text=True, check=True, timeout=timeout).stdout


def description(unit):
    return command("systemctl", "--user", "show", unit, "--value", "--property=Description").strip()


def load_guard(plan):
    index = json.loads((ROOT / GUARD_INDEX).read_bytes())
    ref = next(r for r in index["sources"] if r["path"].endswith("/run_compiler_tests.py"))
    require(ref["path"] == GUARD_SOURCE and ref in plan["sources"], "Sealed guardian projection")
    path = authenticate(ref)
    spec = importlib.util.spec_from_file_location("_owned_nf4_browser_status", path)
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    return guard, ref


def load_observer(plan):
    ref = plan["comfy_observer"]
    require(
        ref in plan["sources"] and ref["path"] == OBSERVER_SOURCE,
        "Exact pinned fresh Comfy observer required",
    )
    path = authenticate(ref)
    spec = importlib.util.spec_from_file_location("_owned_nf4_comfy_observer", path)
    observer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(observer)
    return observer


def observe_host(observer, plan):
    row = observer.observe(expected_boot_id=plan["boot_id"])
    require(type(row) is dict, "Typed fresh Comfy observer result")
    require(row.get("boot_id") == plan["boot_id"], "Fresh observer boot binding")
    require(
        type(row.get("mem_total_bytes")) is int
        and type(row.get("mem_available_bytes")) is int
        and 0 < row["mem_available_bytes"] <= row["mem_total_bytes"]
        and all(
            type(row.get(k)) is int and row[k] >= 0 for k in ("comfy_running", "comfy_pending")
        ),
        "Complete typed host/queue observation",
    )
    require(
        row.get("comfy_state") in {"http_idle", "http_busy", "absent_confirmed"},
        "Explicit HTTP or independently confirmed absent Comfy state",
    )
    if row["comfy_state"] in {"http_idle", "absent_confirmed"}:
        require(
            row["comfy_running"] == row["comfy_pending"] == 0,
            "Idle/confirmed absence requires zero queues",
        )
    else:
        require(row["comfy_running"] + row["comfy_pending"] > 0, "HTTP busy queue evidence")
    return row


def scan(actor):
    matches, unreadable = [], []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            raw = (path / "stat").read_text()
            fields = raw[raw.rfind(")") + 2 :].split()
            pid, pgid, sid = int(path.name), int(fields[2]), int(fields[3])
            if pid == actor["pid"] or pgid == actor["pgid"] or sid == actor["sid"]:
                matches.append(
                    {"pid": pid, "pgid": pgid, "sid": sid, "start_ticks": int(fields[19])}
                )
        except FileNotFoundError:
            continue
        except (PermissionError, OSError, ValueError, IndexError) as error:
            unreadable.append({"pid": path.name, "error": type(error).__name__})
    return {"matches": matches, "unreadable": unreadable}


def readback(plan, invocation, directory):
    wanted = {
        "Type": "exec",
        "MemoryAccounting": "yes",
        "MemoryMax": str(CAP),
        "MemoryHigh": "infinity",
        "MemorySwapMax": "0",
        "OOMPolicy": "kill",
        "KillMode": "control-group",
        "RuntimeMaxUSec": "2h",
        "KillSignal": "2",
        "TimeoutStopUSec": "20s",
        "RemainAfterExit": "yes",
        "InvocationID": invocation,
    }
    properties = dict(
        line.split("=", 1)
        for line in command(
            "systemctl", "--user", "show", plan["unit"], *["--property=" + name for name in wanted]
        ).splitlines()
    )
    group = CGROUP_ROOT / plan["cgroup"].lstrip("/")
    limits = {
        name: (group / name).read_text().strip() for name in ("memory.max", "memory.swap.max")
    }
    save(directory, "root-unit-readback.json", {"properties": properties, "limits": limits})
    require(
        properties == wanted and limits == {"memory.max": str(CAP), "memory.swap.max": "0"},
        "Actual unit/cgroup bounds differ",
    )


def verify_closed(result, receipt, plan, output):
    require(
        result["reason"] == "terminal" and result["unit_readback_verified"] is True,
        "Successful terminal/readback required",
    )
    require(
        type(result["host_after"]) is dict
        and result["host_after"].get("comfy_state")
        in {"http_idle", "http_busy", "absent_confirmed"}
        and result["host_after"].get("boot_id") == plan["boot_id"]
        and result["host_after"].get("mem_total_bytes") == result["selected_memtotal_bytes"],
        "Fresh known final host/Comfy observation required",
    )
    require(
        type(result["model_cache_aliases_after"]) is list
        and result["model_cache_aliases_before"]
        and result["model_cache_aliases_after"] == result["model_cache_aliases_before"],
        "Known unchanged complete model cache alias closure required",
    )
    require(
        result["cgroup_absent_after_stop"] and result["after_stop"]["MainPID"] == "0",
        "External cgroup/main process closure required",
    )
    actor, terminal = result["actor"], result["terminal_before_stop"]
    require(
        type(result["stop_result"]) is str
        and type(result["invocation_id"]) is str
        and len(result["invocation_id"]) == 32
        and terminal.get("InvocationID") == result["invocation_id"],
        "Authenticated owned stop/invocation closure required",
    )
    require(
        actor
        and all(
            type(actor.get(k)) is int and actor[k] > 0
            for k in ("pid", "pgid", "sid", "start_ticks")
        )
        and actor["pid"] == actor["pgid"] == actor["sid"]
        and str(actor["pid"]) == terminal["ExecMainPID"],
        "Actual owned actor binding",
    )
    require(
        len(result["two_proc_scans"]) == 2
        and all(not r["matches"] and not r["unreadable"] for r in result["two_proc_scans"])
        and not result["changed_sources"],
        "Owned descendants/source closure required",
    )
    require(
        all(
            terminal.get(k) == v
            for k, v in {
                "Result": "success",
                "ExecMainCode": "1",
                "ExecMainStatus": "0",
                "MainPID": "0",
            }.items()
        ),
        "Successful actual worker exit required",
    )
    require(
        receipt["stage"] == "complete_pending_external_closure"
        and receipt["attempts"] == receipt["planned_denominator"] == 10
        and receipt["prompt_format"] == plan["prompt_format"],
        "Complete unchanged browser arm",
    )
    groups = receipt["groups"]
    require(
        [(g["group"], g["denominator"], g["summary"]["attempts"]) for g in groups]
        == [("known4", 4, 4), ("public6", 6, 6)],
        "Fixed original group denominators",
    )
    require((output / "output-index.json").is_file(), "Driver output index required")


def execute(plan, digest, guard, guardian, *, plan_blob):
    require(
        hashlib.sha256(plan_blob).hexdigest() == digest and json.loads(plan_blob) == plan,
        "Original sealed plan bytes required",
    )
    directory, output = run_paths(plan)
    aliases_before = verify_cache_aliases(plan)
    observer = load_observer(plan)
    total = memtotal()
    before = observe_host(observer, plan)
    require(before["mem_total_bytes"] == total, "Same selected physical total")
    require(
        before["mem_available_bytes"] >= reserve(total) + 2 * BASE_BYTES + 64 * 1024**2,
        "Full unchanged startup admission refused",
    )
    require(before["comfy_running"] == before["comfy_pending"] == 0, "ComfyUI must be idle")
    require(
        guard.status(plan["unit"])["LoadState"] == "not-found"
        and not directory.exists()
        and not output.exists(),
        "Fresh unit/output required",
    )
    directory.mkdir(mode=0o700)
    env = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "HF_HUB_CACHE": str(ROOT / "artifacts/jev-local-hf/hub"),
        "PYTHONPATH": ":".join(
            str(ROOT / p) for p in ("artifacts/jev-runtime/overlay", "src", "scripts")
        ),
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTORCH_ALLOC_CONF": "backend:native",
        "JEV_NATIVE_BROWSER_EXECUTE": PROFILE,
        "JEV_NATIVE_BROWSER_UNIT": plan["unit"],
    }
    launch = [
        "systemd-run",
        "--user",
        "--unit=" + plan["unit"],
        "--working-directory=" + str(ROOT),
        *[
            "--property=" + p
            for p in (
                "Type=exec",
                "MemoryAccounting=yes",
                f"MemoryMax={CAP}",
                "MemoryHigh=infinity",
                "MemorySwapMax=0",
                "OOMPolicy=kill",
                "KillMode=control-group",
                "RuntimeMaxSec=7200",
                "KillSignal=SIGINT",
                "TimeoutStopSec=20",
                "RemainAfterExit=yes",
                "Description=" + plan["nonce"],
                "StandardOutput=append:" + str(directory / "browser-service.log"),
                "StandardError=append:" + str(directory / "browser-service.log"),
            )
        ],
        *["--setenv=" + k + "=" + v for k, v in env.items()],
        "/home/albert_hsieh/ComfyUI/.venv/bin/python",
        "-B",
        "-s",
        str(HERE / "browser_entry.py"),
        "--run",
        "--plan",
        str(directory / "selected-plan.json"),
        "--plan-sha256",
        digest,
    ]
    with (directory / "selected-plan.json").open("xb") as stream:
        os.chmod(stream.name, 0o600)
        stream.write(plan_blob)
        stream.flush()
        os.fsync(stream.fileno())
    save(
        directory,
        "root-launch.json",
        {
            "plan_sha256": digest,
            "host_before": before,
            "argv": launch,
            "guardian": guardian,
            "model_cache_aliases_before": aliases_before,
            "scope": "40GiB memcg/swap0 only; no process-native CUDA cap claim.",
        },
    )
    observations, invocation, primary = [], None, None
    reason, terminal, stopped, stop_result = "terminal", None, None, None
    started, attempted, last_print, verified = time.monotonic(), False, -30, False
    try:
        attempted = True
        print(command(*launch), flush=True)
        first = guard.status(plan["unit"])
        invocation = first["InvocationID"]
        require(
            len(invocation) == 32 and description(plan["unit"]) == plan["nonce"], "Owned invocation"
        )
        require(first["ControlGroup"] == plan["cgroup"], "Actual owned cgroup")
        readback(plan, invocation, directory)
        verified = True
        while True:
            state, host = guard.status(plan["unit"]), observe_host(observer, plan)
            verify_cache_aliases(plan, aliases_before)
            require(state["InvocationID"] == invocation, "Same owned invocation")
            require(
                current_boot() == plan["boot_id"]
                and memtotal() == total
                and host["mem_total_bytes"] == total,
                "Same boot/physical total",
            )
            elapsed = time.monotonic() - started
            row = {"elapsed_seconds": elapsed, "unit": state, "host": host}
            observations.append(row)
            if elapsed - last_print >= 30 or state["SubState"] in ("exited", "dead", "failed"):
                print(json.dumps(row), flush=True)
                last_print = elapsed
            if state["SubState"] in ("exited", "dead", "failed"):
                break
            if host["mem_available_bytes"] < reserve(total) + 2 * GIB:
                reason = "host_reserve_margin_crossed"
                break
            if host["comfy_running"] or host["comfy_pending"]:
                reason = "Comfy_became_busy_stop_own_unit_only"
                break
            if elapsed > 7180:
                reason = "owner_deadline"
                break
            time.sleep(2)
    except BaseException as error:
        primary, reason = error, "owner_error"
    finally:
        try:
            if attempted:
                terminal = guard.observe(lambda: guard.status(plan["unit"]))
                same_nonce = guard.observe(lambda: description(plan["unit"])) == plan["nonce"]
                current = terminal.get("InvocationID")
                owned = (
                    same_nonce
                    and isinstance(current, str)
                    and len(current) == 32
                    and (invocation is None or current == invocation)
                )
                if owned:
                    invocation = current
                    stop_result = guard.observe(
                        lambda: command("systemctl", "--user", "stop", plan["unit"], timeout=30)
                    )
                else:
                    stop_result = {"closure": "UNKNOWN_NO_AUTHENTICATED_OWNED_INVOCATION"}
                stopped = guard.observe(lambda: guard.status(plan["unit"]))
            actor_read = guard.observe(
                lambda: json.loads((directory / "actor-intent.json").read_bytes())
            )
            actor = actor_read.get("actor") if type(actor_read) is dict else None
            if actor is not None:
                require(
                    actor_read["plan_sha256"] == digest
                    and actor_read["unit"] == plan["unit"]
                    and actor_read["nonce"] == plan["nonce"],
                    "Actor intent bound to plan",
                )
            scans = []
            if actor is not None:
                scans.append(scan(actor))
                time.sleep(0.2)
                scans.append(scan(actor))
            result = {
                "plan_sha256": digest,
                "selected_memtotal_bytes": total,
                "model_cache_aliases_before": aliases_before,
                "model_cache_aliases_after": guard.observe(
                    lambda: verify_cache_aliases(plan, aliases_before)
                ),
                "invocation_id": invocation,
                "reason": reason,
                "unit_readback_verified": verified,
                "primary_error": None if primary is None else error_record(primary),
                "observations": observations,
                "terminal_before_stop": terminal,
                "stop_result": stop_result,
                "after_stop": stopped,
                "cgroup_absent_after_stop": not (CGROUP_ROOT / plan["cgroup"].lstrip("/")).exists(),
                "actor": actor,
                "actor_read": actor_read,
                "two_proc_scans": scans,
                "changed_sources": [
                    r["path"]
                    for r in plan["sources"]
                    if guard.observe(lambda r=r: pin(ROOT / r["path"])) != r
                ],
                "host_after": observe_with_evidence(guard, lambda: observe_host(observer, plan)),
                "final_postexit_cgroup_counters": None,
                "counter_scope": "Live samples only; no final zero-OOM claim after cgroup removal.",
            }
            save(directory, "root-external-closure.json", result)
        except BaseException as secondary:
            if primary is not None:
                primary.add_note("Owner finalization also failed: " + repr(secondary))
            else:
                primary = secondary
            try:
                save(
                    directory,
                    "root-finalization-error.json",
                    {
                        "closure": "UNKNOWN_INCOMPLETE_FINALIZATION",
                        "primary_error": error_record(primary),
                        "finalization_error": error_record(secondary),
                    },
                )
            except BaseException as persistence_error:
                primary.add_note(
                    "Could not preserve finalization error: " + repr(persistence_error)
                )
    if primary is not None:
        raise primary
    receipt = json.loads((output / "receipt.json").read_bytes())
    verify_closed(result, receipt, plan, output)
    print(
        json.dumps(
            {
                "whole_owned_closed": True,
                "attempts": 10,
                "successes": receipt["successes"],
                "InvocationID": invocation,
            }
        ),
        flush=True,
    )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if not args.run:
        print("DRY_NO_ACTOR_OR_MODEL; explicit --run and sealed plan required")
        return 0
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
    require(args.plan and args.plan_sha256, "Explicit sealed plan required")
    plan = load_plan(args.plan, args.plan_sha256)
    guard, guardian = load_guard(plan)
    return execute(
        plan, args.plan_sha256, guard, guardian, plan_blob=ordinary(args.plan).read_bytes()
    )


if __name__ == "__main__":
    raise SystemExit(main())
