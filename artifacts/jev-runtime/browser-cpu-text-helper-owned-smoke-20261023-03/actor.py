"""Dry-default future owned Qwen smoke. Stdlib only until the verified worker loader.

The parent is the real external request watchdog, in the original owner03 cgroup.
The actual NN worker uses frozen02; ASGI transport is in-process, never real HTTP.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import importlib.util
import json
import math
import os
import secrets
import selectors
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OWNER = ROOT / "artifacts/jev-runtime/llama-cpp-owner-20261023-03/controller.py"
OWNER_SHA = "882ba1579b17b31dca0ee01fe2e89f46b1bc8c17b7635a0c4aa05042fc847ac0"
HELPER = ROOT / "artifacts/jev-runtime/browser-cpu-text-helper-20261023-02/helper.py"
HELPER_SHA = "38f4e346c47c4091b22f396e38aca583f33729bd357c04a3e5abeb329cf2b62d"
PENDING = (
    ROOT
    / "artifacts/jev-runtime/browser-cpu-text-helper-20261023-01/runtime-selection.pending.json"
)
PENDING_SHA = "b5e0c09b60b2de3dadfb48d98febc8e61c236bb9e61a0d03ae9b126a26223e26"
ACQUISITION = ROOT / "artifacts/jev-runtime/llama-cpp-owner-20261023-02/actual-02/helper-package"
PACKAGE_SHA = "fce23ca9ae2e3aa00dc5e412c9c9a06eed3ee53083447cadbeb7a0d7e471638e"
PYTHON = "/home/albert_hsieh/ComfyUI/.venv/bin/python"
CATALOG_PLAN = (
    ROOT / "artifacts/jev-runtime/browser-cpu-text-helper-runtime-catalog-20261023-02/plan.json"
)
CATALOG_PLAN_SHA = "c41d9c2030a84faf2170579132d872688b9d427c4af2542fc9eb4e44747db5ef"
PROFILE = "qwen25-05b-cpu4-2g-owned-native-smoke-v1"
MODEL_PROFILE = "qwen25-05b-bf16-cpu4-field-text-v1"
OWNER_PROFILE = "llama-cpp-cpu4-command-owned-v1"
UNIT = "jev-d3-llama-cpp-cpu4-20261023-03.service"
FILES = frozenset(
    {
        "LICENSE",
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "merges.txt",
        "vocab.json",
        "tokenizer.json",
        "model.safetensors",
    }
)
CAP, RESERVE, OWNER_RUNTIME = 2 * 1024**3, 26119172916, 7200
WHOLE_SECONDS, REQUEST_SECONDS = 180.0, 20.0
ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4",
    "ACCELERATE_USE_CPU": "true",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONNOUSERSITE": "1",
    "USE_HUB_KERNELS": "0",
    "HF_ENABLE_PARALLEL_LOADING": "false",
}


def need(value, message):
    if not value:
        raise ValueError(message)


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def small(path, expected=None):
    path = Path(path)
    need(
        path.is_absolute()
        and path.is_file()
        and not path.is_symlink()
        and path.stat().st_size <= 1024 * 1024,
        "Bounded ordinary source/metadata",
    )
    blob = path.read_bytes()
    need(expected is None or digest(blob) == expected, "Immutable selected metadata/source SHA")
    return blob


def read_json(path, expected=None):
    return json.loads(small(path, expected))


def write_new(path, value):
    path = Path(path)
    blob = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return {"path": str(path), "size_bytes": len(blob), "sha256": digest(blob)}


def load_source(path, expected, name):
    blob = small(path, expected)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(blob, str(path), "exec"), module.__dict__)
    return module


def elapsed_check(began, clock=time.monotonic):
    need(clock() - began < WHOLE_SECONDS, "Whole smoke budget including preparation exceeded")


def checked_ref(ref):
    need(type(ref) is dict and set(ref) == {"path", "size_bytes", "sha256"}, "Exact metadata ref")
    blob = small(ref["path"], ref["sha256"])
    need(len(blob) == ref["size_bytes"], "Metadata ref size")
    return json.loads(blob)


def accepted_catalog(path, expected):
    selected = read_json(path, expected)
    need(
        selected.get("schema") == "owned-helper-runtime-catalog-acceptance-v1"
        and set(selected.get("refs", {}))
        == {"catalog_receipt", "root_execution", "owner_receipt", "closure", "catalog_plan"},
        "Root-selected closed catalog acceptance required",
    )
    need(
        selected["refs"]["catalog_plan"]["path"] == str(CATALOG_PLAN)
        and selected["refs"]["catalog_plan"]["sha256"] == CATALOG_PLAN_SHA,
        "Exact shared catalog plan ref",
    )
    values = {key: checked_ref(ref) for key, ref in selected["refs"].items()}
    catalog_plan = values["catalog_plan"]
    catalog, root, owner, closure = (
        values[k] for k in ("catalog_receipt", "root_execution", "owner_receipt", "closure")
    )
    need(
        catalog.get("profile") == "owned-helper-runtime-native-catalog-2g-v1"
        and catalog.get("status") == "catalog_complete_runtime_identity_only"
        and catalog.get("completed") is True
        and catalog.get("plan_sha256") == CATALOG_PLAN_SHA
        and catalog.get("final_authentication") is True
        and catalog.get("model_loaded") is False
        and catalog.get("helper_qualified") is False
        and catalog.get("full_native_DSO_closure") is False,
        "Actual catalog identity-only scope",
    )
    need(
        root.get("exit_code") == 0
        and owner.get("completed") is True
        and owner.get("primary_error") is None
        and owner.get("secondary_errors") == []
        and owner.get("sources_unchanged") is True
        and owner.get("closure") == closure,
        "Catalog root/owner success and bound closure",
    )
    actor = owner.get("state", {}).get("actor_exit", {})
    results = actor.get("command_results", [])
    need(
        actor.get("entry_exit_code") == 0
        and actor.get("error") is None
        and len(results) == 1
        and results[0].get("returncode") == 0,
        "One zero-exit catalog command",
    )
    need(
        closure.get("closure_mode") == "authenticated_stop"
        and closure.get("primary_error") is None
        and closure.get("closure_errors") == []
        and closure.get("two_process_scans")
        == [{"matches": [], "unknown": []}, {"matches": [], "unknown": []}]
        and closure.get("cgroup_absent") is True
        and closure.get("whole_unit_exit_confirmed") is True
        and closure.get("observed_cgroup_cohort_exit_confirmed") is True
        and closure.get("continuation_authorized") is False,
        "Actual authenticated closed catalog cohort",
    )
    cfg = owner["selected_config"]
    policy = catalog_plan["owner_policy"]
    need(
        owner.get("profile") == OWNER_PROFILE
        and cfg.get("profile") == OWNER_PROFILE
        and cfg.get("unit") == UNIT
        and cfg.get("runtime_seconds") == OWNER_RUNTIME
        and closure["selection"].get("unit") == UNIT
        and closure["selection"].get("memory_max_bytes") == CAP
        and closure["selection"].get("runtime_seconds") == OWNER_RUNTIME
        and catalog_plan["owner_source"]["sha256"] == OWNER_SHA
        and policy["profile"] == OWNER_PROFILE
        and policy["unit"] == UNIT
        and policy["memory_max_bytes"] == CAP
        and policy["swap_max_bytes"] == 0
        and policy["reserve_bytes"] == RESERVE
        and policy["initial_available_bytes"] == RESERVE + CAP
        and policy["runtime_available_bytes"] == RESERVE
        and policy["owner_runtime_seconds"] == OWNER_RUNTIME,
        "Exact selected owner03 2GiB/full-cap admission catalog only; no mixed owner/cap",
    )
    need(
        cfg["nonce"] == closure["selection"]["nonce"] == closure["before_stop"]["Description"]
        and cfg["boot_id"] == closure["selection"]["boot_id"]
        and closure["invocation_id"] == closure["before_stop"]["InvocationID"],
        "Catalog nonce/boot/invocation joins",
    )
    observed_actor = results[0].get("observed_identity")
    need(type(observed_actor) is dict, "Actual catalog command identity observed by owner")
    stable = ("pid", "start_ticks", "pgid", "sid", "uid", "cgroup")
    for phase in ("initial_admission", "final_admission"):
        admission = catalog[phase]
        process = admission["identity"]["process"]
        need(
            admission["unit"]["Description"] == cfg["nonce"]
            and admission["unit"]["InvocationID"] == closure["invocation_id"]
            and admission["host"]["observation"]["boot_id"] == cfg["boot_id"]
            and {key: process[key] for key in stable}
            == {key: observed_actor["process"][key] for key in stable}
            and all(
                admission["identity"][key] == observed_actor[key] for key in ("exe", "argv", "cwd")
            ),
            "Catalog actual admission identity joins original owner command/invocation/boot",
        )
    need(
        type(catalog.get("runtime_source_files")) is list
        and len(catalog["runtime_source_files"]) >= 11
        and type(catalog.get("runtime_binary_files")) is list
        and len(catalog["runtime_binary_files"]) == 9
        and set(catalog.get("modules", {}))
        == {"torch", "transformers", "tokenizers", "safetensors", "accelerate"},
        "Complete selected catalog source/module/nine-binary inventory",
    )
    need(
        catalog["runtime_source_files"] == catalog_plan["runtime_sources"],
        "Exact catalog source plan join",
    )
    need(
        {k: v["origin"] for k, v in catalog["modules"].items()} == catalog_plan["module_origins"]
        and {k: v["version"] for k, v in catalog["modules"].items()} == catalog_plan["versions"],
        "Exact catalog module origin/version plan join",
    )
    native = {row["path"]: row for row in catalog_plan["native_files"]}
    observed = {row["path"]: row for row in catalog["runtime_binary_files"]}
    need(len(observed) == 9 and set(observed) == set(native), "Exact selected nine native paths")
    for path, row in observed.items():
        need(
            row["size_bytes"] == native[path]["size_bytes"]
            and type(row.get("sha256")) is str
            and len(row["sha256"]) == 64
            and all(c in "0123456789abcdef" for c in row["sha256"])
            and native[path].get("expected_sha256") in (None, row["sha256"]),
            "Catalog native identity",
        )
    python = catalog["runtime_python"]
    need(
        python["path"] == catalog_plan["logical_interpreter"]
        and python["resolved_path"] == catalog_plan["interpreter"]["path"]
        and python["stat"] == catalog_plan["interpreter"]["stat"],
        "Catalog interpreter plan join",
    )
    return catalog


def exact_snapshot(package, output, records, checkpoint):
    need(set(records) == FILES, "Exact eight native snapshot files")
    output = Path(output)
    output.mkdir(mode=0o700, exist_ok=False)
    result = {}
    for name in sorted(FILES):
        checkpoint()
        source = Path(package) / name
        target = output / name
        need(
            source.is_file() and not any(p.is_symlink() for p in (source, *source.parents)),
            "Ordinary acquired file",
        )
        need(source.stat().st_size == records[name]["size_bytes"], "Acquired size drift")
        os.link(source, target)  # Never chmod a hardlink: original inode must remain unchanged.
        actual = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                checkpoint()
                actual.update(block)
        need(actual.hexdigest() == records[name]["sha256"], "Owned payload reauthentication failed")
        result[name] = {
            "path": str(target),
            "size_bytes": records[name]["size_bytes"],
            "sha256": actual.hexdigest(),
        }
        if "publisher_git_blob_sha1" in records[name]:
            result[name]["publisher_git_blob_sha1"] = records[name]["publisher_git_blob_sha1"]
    need({p.name for p in output.iterdir()} == FILES, "No README/journals in exact snapshot")
    return result


def failure_receipt(output, state, primary, label):
    """Demote candidate names and retain the actual primary across evidence errors."""
    output = Path(output)
    candidate = output / "receipt.json"
    if candidate.exists():
        try:
            candidate.rename(output / ("failed-" + label + "-candidate-receipt.json"))
        except BaseException as secondary:
            primary.add_note("Success-receipt demotion also failed: " + repr(secondary))
    state.update(
        status="failed_no_helper_qualification",
        actual_generation_qualified=False,
        error={"type": type(primary).__name__, "message": str(primary)},
    )
    try:
        write_new(output / "failure.json", state)
    except BaseException as secondary:
        primary.add_note("Failure evidence also failed: " + repr(secondary))


class RequestWatchdog:
    """Pure parent state: no worker message may disarm or extend its deadline."""

    def __init__(self, secret, ticket_sha, whole_deadline, clock=time.monotonic):
        self.secret, self.ticket_sha, self.whole_deadline, self.clock = (
            secret,
            ticket_sha,
            whole_deadline,
            clock,
        )
        self.sequence, self.deadline, self.request_id, self.armed = 0, None, None, False

    def check(self):
        now = self.clock()
        if now >= self.whole_deadline or (self.deadline is not None and now >= self.deadline):
            raise TimeoutError("External parent watchdog deadline reached")

    def handle(self, message):
        self.check()
        need(
            message.get("secret") == self.secret
            and message.get("ticket_sha256") == self.ticket_sha
            and type(message.get("sequence")) is int
            and message["sequence"] == self.sequence + 1,
            "Authenticated ordered control",
        )
        operation = message.get("operation")
        need(operation in {"check", "arm"}, "No cancel/finish/disarm control operation")
        if "deadline" in message:
            value, rid = message["deadline"], message.get("request_id")
            need(
                type(value) in (int, float)
                and math.isfinite(value)
                and type(rid) is str
                and self.clock() < value <= self.clock() + REQUEST_SECONDS,
                "Original bounded request deadline",
            )
            if self.deadline is None:
                self.deadline, self.request_id = value, rid
            need(
                self.deadline == value and self.request_id == rid,
                "No request deadline extension or replacement",
            )
        if operation == "arm":
            need(
                not self.armed
                and self.deadline is not None
                and type(message.get("started")) in (int, float)
                and message["deadline"] == message["started"] + REQUEST_SECONDS,
                "One actual original arm",
            )
            self.armed = True
        self.sequence += 1
        self.check()
        return {"ok": True, "sequence": self.sequence}


class ParentGate:
    def __init__(self, output, protocol):
        self.protocol = protocol
        self.owner = load_source(OWNER, OWNER_SHA, "_helper_smoke_exact_owner03")
        self.boundary = self.owner.load_fixed("boundary")
        self.directory = Path(output).parent
        self.cfg_blob = small(self.directory / "sealed-config.json")
        self.cfg, self.admission = (
            json.loads(self.cfg_blob),
            read_json(self.directory / "admission.json"),
        )
        need(
            self.cfg["profile"] == OWNER_PROFILE
            and self.cfg["unit"] == UNIT
            and self.cfg["runtime_seconds"] == OWNER_RUNTIME
            and self.cfg["output"] == str(self.directory)
            and self.admission["config_sha256"] == digest(self.cfg_blob)
            and self.admission["nonce"] == self.cfg["nonce"],
            "Real original owner03 admission/config",
        )
        self.selected = self.owner.selection(self.cfg, self.boundary)
        self.parent = protocol.identity(os.getpid())
        self.main = protocol.identity(self.admission["actor"]["pid"])
        self.check(full=True)

    def check(self, full=False):
        need(self.boundary.boot() == self.cfg["boot_id"], "Same live boot")
        need(
            self.protocol.same_identity(self.parent, self.protocol.identity(self.parent["pid"]))
            and self.protocol.same_identity(self.main, self.protocol.identity(self.main["pid"]))
            and self.parent["cgroup"] == self.main["cgroup"] == self.selected.cgroup,
            "Live actual parent/main identity and original cgroup",
        )
        need(
            self.main["start_ticks"] == self.admission["actor"]["start_ticks"],
            "Original admitted main PID start",
        )
        group = self.boundary.CGROUP_ROOT / self.selected.cgroup.lstrip("/")
        need(
            (group / "memory.max").read_text().strip() == str(CAP)
            and (group / "memory.swap.max").read_text().strip() == "0",
            "Original 2GiB/swap0",
        )
        quota, period = map(int, (group / "cpu.max").read_text().split())
        need(
            period > 0 and quota == 4 * period and set(os.sched_getaffinity(0)) == {16, 17, 18, 19},
            "Original CPU4",
        )
        if full:
            row = self.boundary.status(self.selected)
            need(
                all(row.get(k) == v for k, v in self.selected.properties().items())
                and row.get("Description") == self.cfg["nonce"]
                and row.get("InvocationID") == self.admission["invocation_id"]
                and row.get("ControlGroup") == self.selected.cgroup
                and row.get("MainPID") == str(self.main["pid"]),
                "Live exact original unit/invocation",
            )
        # Original owner03 controller independently retains its existing R/swap/Comfy gates.
        # No per-token observer or new full-host policy is introduced here.

    def child(self, pid):
        row = self.protocol.identity(pid)
        need(
            row["ppid"] == self.parent["pid"]
            and row["uid"] == self.parent["uid"]
            and row["cgroup"] == self.parent["cgroup"]
            and row["state"] not in {"Z", "X"},
            "Exact same-unit worker child",
        )
        return row


def stop_exact_worker(protocol, expected, pidfd):
    need(
        protocol.same_identity(expected, protocol.identity(expected["pid"])),
        "Never signal a reused/migrated PID",
    )
    signal.pidfd_send_signal(pidfd, signal.SIGKILL)


def run_parent(args, plan, protocol):
    began = time.monotonic()
    output = Path(args.output).absolute()
    gate = ParentGate(output, protocol)
    need(
        output.parent == gate.directory
        and not output.exists()
        and not any(p.is_symlink() for p in (output, *output.parents)),
        "Fresh private smoke output",
    )
    output.mkdir(mode=0o700)

    def checkpoint():
        elapsed_check(began)
        gate.check()

    catalog = accepted_catalog(args.catalog_acceptance, args.catalog_acceptance_sha256)
    python = catalog["runtime_python"]
    info = Path(PYTHON).resolve().stat()
    need(
        python["path"] == PYTHON
        and python["resolved_path"] == str(Path(PYTHON).resolve())
        and {key: getattr(info, key) for key in python["stat"]} == python["stat"]
        and set(python["stat"]) == {"st_dev", "st_ino", "st_size", "st_mtime_ns"},
        "Closed-catalog interpreter stat before worker spawn",
    )
    acquired = read_json(ACQUISITION / "package-receipt.json", PACKAGE_SHA)
    need(
        acquired["status"] == "complete_public_package_stream_identity_verified",
        "Selected acquired package",
    )
    rows = {row["name"]: row for row in acquired["files"]}
    pending = read_json(PENDING, PENDING_SHA)
    records = {
        name: {
            **rows[name],
            **(
                {"publisher_git_blob_sha1": pending["model_files"][name]["publisher_git_blob_sha1"]}
                if name != "model.safetensors"
                else {}
            ),
        }
        for name in FILES
    }
    model_files = exact_snapshot(ACQUISITION, output / "snapshot", records, checkpoint)
    descriptor = copy.deepcopy(pending)
    descriptor.update(
        model_files=model_files,
        runtime_files=catalog["runtime_source_files"],
        runtime_binary_files=catalog["runtime_binary_files"],
        module_origins={k: v["origin"] for k, v in catalog["modules"].items()},
        versions={k: v["version"] for k, v in catalog["modules"].items()},
        status="ROOT_SELECTED_OWNED_SMOKE_ONLY",
        runtime_python=python,
    )
    # Child process gets this explicit offline import path; the parent environment stays unchanged.
    env = {
        **os.environ,
        **ENVIRONMENT,
        "PYTHONPATH": plan["overlay"],
        "HELPER_AUTHORITY_SHA256": plan["authority_sha256"],
    }
    need(
        hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"),
        "Exact pidfd kill support required",
    )
    parent_socket, child_socket = socket.socketpair()
    inherited_fd = child_socket.fileno()
    child = None
    pidfd = None
    primary = None
    state = {
        "profile": PROFILE,
        "status": "preparing_worker",
        "transport": "in_process_ASGI_not_HTTP",
        "actual_generation_qualified": False,
        "browser_score": None,
        "full_native_DSO_closure": False,
        "catalog_acceptance_sha256": args.catalog_acceptance_sha256,
        "model_files": model_files,
    }
    try:
        child = subprocess.Popen(
            [
                PYTHON,
                "-B",
                "-s",
                str(HERE / "actor.py"),
                "--worker",
                "--control-fd",
                str(child_socket.fileno()),
            ],
            cwd=str(ROOT),
            env=env,
            pass_fds=(child_socket.fileno(),),
            close_fds=True,
        )
        child_socket.close()
        worker = gate.child(child.pid)
        pidfd = os.pidfd_open(child.pid)
        need(
            protocol.same_identity(worker, protocol.identity(child.pid)),
            "Stable child around pidfd creation",
        )
        secret = secrets.token_hex(32)
        ticket = {
            "schema": "owned-helper-smoke-ticket-v1",
            "profile": OWNER_PROFILE,
            "unit": UNIT,
            "boot_id": gate.cfg["boot_id"],
            "nonce": gate.cfg["nonce"],
            "invocation_id": gate.admission["invocation_id"],
            "config_sha256": digest(gate.cfg_blob),
            "parent": gate.parent,
            "worker": worker,
            "unit_main": gate.main,
            "control_fd": inherited_fd,
            "secret": secret,
            "seconds": REQUEST_SECONDS,
            "memory_max_bytes": CAP,
            "swap_max_bytes": 0,
            "runtime_seconds": OWNER_RUNTIME,
        }
        ticket_ref = write_new(output / "owner-ticket.json", ticket)
        verifier = next(row for row in plan["sources"] if row["path"] == str(HERE / "authority.py"))
        descriptor["owner"] = {
            "profile": OWNER_PROFILE,
            "ticket": ticket_ref,
            "verifier": verifier,
            "hard_watchdog_seconds": REQUEST_SECONDS,
            "watchdog_external": True,
            "listen_host": "127.0.0.1",
        }
        selection_ref = write_new(output / "runtime-selection.json", descriptor)
        channel = protocol.Channel(parent_socket)
        channel.send(
            {"selection": selection_ref, "output": str(output), "helper_sha256": HELPER_SHA}
        )
        watchdog = RequestWatchdog(secret, ticket_ref["sha256"], began + WHOLE_SECONDS)
        while child.poll() is None:
            gate.check()
            watchdog.check()
            wait = min(0.05, began + WHOLE_SECONDS - time.monotonic())
            if watchdog.deadline is not None:
                wait = min(wait, watchdog.deadline - time.monotonic())
            ready = selectors.DefaultSelector()
            try:
                ready.register(parent_socket, selectors.EVENT_READ)
                events = ready.select(max(0.001, wait))
            finally:
                ready.close()
            if events:
                try:
                    message = channel.receive(max(0.001, wait))
                except EOFError:
                    try:
                        child.wait(
                            timeout=min(0.2, max(0.001, began + WHOLE_SECONDS - time.monotonic()))
                        )
                    except subprocess.TimeoutExpired:
                        raise RuntimeError(
                            "Worker closed control channel while still live"
                        ) from None
                    watchdog.check()
                    break
                gate.child(child.pid)
                if message.get("operation") == "arm":
                    gate.check(full=True)
                reply = watchdog.handle(message)
                if message.get("operation") == "arm":
                    write_new(
                        output / "watchdog-armed.json",
                        {
                            "request_id": watchdog.request_id,
                            "original_deadline_monotonic": watchdog.deadline,
                            "observed_parent_monotonic": time.monotonic(),
                            "parent": gate.parent,
                            "worker": worker,
                            "pidfd_bound": True,
                            "worker_cannot_disarm": True,
                        },
                    )
                    watchdog.check()
                channel.send(reply)
        code = child.wait(timeout=1)
        watchdog.check()
        need(code == 0 and watchdog.armed, "One successful externally supervised worker request")
        worker_receipt = read_json(output / "worker-receipt.json")
        need(
            worker_receipt.get("status") == "one_actual_native_ASGI_request_complete"
            and worker_receipt.get("runtime_selection_sha256") == selection_ref["sha256"]
            and worker_receipt.get("actual_generation_count") == 1,
            "Bound actual native worker receipt",
        )
        state.update(
            status="owned_helper_smoke_complete_pending_external_closure",
            worker=worker,
            worker_receipt=worker_receipt,
            worker_exit_code=code,
            elapsed_seconds=time.monotonic() - began,
            actual_generation_qualified=True,
            transport="in_process_ASGI_not_HTTP",
            external_request_deadline=watchdog.deadline,
            external_watchdog_armed=True,
        )
        checkpoint()
        write_new(output / "receipt.json", state)
        checkpoint()
        watchdog.check()
    except BaseException as exc:
        primary = exc
        state["watchdog_failure_observed_monotonic"] = time.monotonic()
        if child is not None and child.poll() is None and pidfd is not None:
            try:
                stop_exact_worker(protocol, worker, pidfd)
            except BaseException as secondary:
                exc.add_note("Exact worker kill also failed: " + repr(secondary))
        if child is not None:
            try:
                child.wait(timeout=3)
            except BaseException as secondary:
                exc.add_note("Worker wait also failed: " + repr(secondary))
        failure_receipt(output, state, exc, "request")
        raise
    finally:
        cleanup_error = None
        for operation in [
            parent_socket.close,
            child_socket.close,
            *([lambda: os.close(pidfd)] if pidfd is not None else []),
        ]:
            try:
                operation()
            except BaseException as secondary:
                if primary is not None:
                    primary.add_note("Parent cleanup also failed: " + repr(secondary))
                elif cleanup_error is None:
                    cleanup_error = secondary
                else:
                    cleanup_error.add_note("Additional cleanup failed: " + repr(secondary))
        if cleanup_error is not None:
            failure_receipt(output, state, cleanup_error, "cleanup")
            raise cleanup_error
        if primary is None and child is not None:
            need(child.poll() == 0, "Actual worker closure before parent return")
    try:
        checkpoint()
        watchdog.check()
    except BaseException as exc:
        failure_receipt(output, state, exc, "post-cleanup")
        raise
    return state


def worker(control_fd):
    # This branch is reachable only from the future pinned owned parent command.
    need(
        str(Path(sys.executable).absolute()) == PYTHON
        and sys.flags.no_user_site
        and sys.dont_write_bytecode
        and not {"torch", "transformers", "tokenizers"}.intersection(sys.modules),
        "Fresh exact selected worker interpreter -B -s",
    )
    protocol = load_source(
        HERE / "authority.py",
        os.environ["HELPER_AUTHORITY_SHA256"],
        "_helper_smoke_worker_protocol",
    )
    channel = protocol.Channel(socket.socket(fileno=control_fd))
    bootstrap = channel.receive(10)
    selection = bootstrap["selection"]
    descriptor = checked_ref(selection)
    selected_python = descriptor["runtime_python"]
    info = Path(sys.executable).resolve().stat()
    need(
        selected_python["path"] == PYTHON
        and selected_python["resolved_path"] == str(Path(sys.executable).resolve())
        and selected_python["version"] == list(sys.version_info[:3])
        and {key: getattr(info, key) for key in selected_python["stat"]} == selected_python["stat"],
        "Actual worker interpreter matches closed owned catalog",
    )
    need(
        descriptor["owner"]["verifier"]["sha256"] == os.environ["HELPER_AUTHORITY_SHA256"],
        "Worker verifier source binding",
    )
    helper = load_source(HELPER, HELPER_SHA, "_helper_smoke_exact02")
    runtime = helper.load_runtime(selection["path"], selection["sha256"])
    output = Path(bootstrap["output"])
    opaque = "probe-" + secrets.token_hex(6)
    context = {
        "goal": "Enter the reference code " + opaque + " exactly as written.",
        "field": {"label": "Reference code", "role": "textbox", "value": ""},
        "page": {"title": "", "text": ""},
        "recent_actions": [],
    }
    payload = {
        "model": helper.MODEL_ID,
        "max_tokens": 1024,
        "response_format": {"type": "json_object"},
        "reasoning": {"enabled": False},
        "messages": [
            {"role": "system", "content": helper.TEXT_VALUE},
            {"role": "user", "content": json.dumps(context)},
        ],
    }
    key = secrets.token_hex(24)
    app = helper.make_application(runtime, helper.DirectoryJournal(output / "http-journal"), key)
    sent = []

    async def receive():
        return {"type": "http.request", "body": json.dumps(payload).encode(), "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/chat/completions",
        "server": ("127.0.0.1", 0),
        "client": ("127.0.0.1", 0),
        "headers": [(b"authorization", ("Bearer " + key).encode())],
    }
    asyncio.run(app(scope, receive, send))
    need(
        len(sent) == 2
        and sent[0]["type"] == "http.response.start"
        and sent[0]["status"] == 200
        and sent[1]["type"] == "http.response.body"
        and not sent[1].get("more_body"),
        "Complete actual ASGI response",
    )
    response = json.loads(sent[1]["body"])
    raw = response["choices"][0]["message"]["content"]
    parsed = json.loads(raw)
    need(
        set(parsed) == {"text"} and parsed["text"] == opaque,
        "NN must emit the fresh opaque value exactly; no fallback",
    )
    write_new(
        output / "worker-receipt.json",
        {
            "profile": PROFILE,
            "status": "one_actual_native_ASGI_request_complete",
            "runtime_selection_sha256": selection["sha256"],
            "actual_generation_count": 1,
            "transport": "in_process_ASGI_not_HTTP",
            "response": response,
            "expected_opaque_value": opaque,
            "semantic_smoke_match": True,
            "helper_general_quality_qualified": False,
            "browser_score": None,
        },
    )


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--control-fd", type=int)
    parser.add_argument("--plan")
    parser.add_argument("--plan-sha256")
    parser.add_argument("--catalog-acceptance")
    parser.add_argument("--catalog-acceptance-sha256")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.worker:
        worker(args.control_fd)
        return
    if not args.execute:
        print(
            json.dumps(
                {
                    "profile": PROFILE,
                    "status": "source_only_not_ready",
                    "model_started": False,
                    "owner_started": False,
                    "transport": "in_process_ASGI_not_HTTP",
                    "requires_closed_catalog_acceptance_and_root_command_spec": True,
                }
            )
        )
        return
    need(
        all(
            getattr(args, key)
            for key in (
                "plan",
                "plan_sha256",
                "catalog_acceptance",
                "catalog_acceptance_sha256",
                "output",
            )
        ),
        "Explicit future root selection",
    )
    plan = read_json(args.plan, args.plan_sha256)
    need(
        plan.get("profile") == PROFILE and plan.get("owner_source_sha256") == OWNER_SHA,
        "Exact smoke plan",
    )
    for entry in plan["sources"]:
        need(
            len(small(entry["path"], entry["sha256"])) == entry["size_bytes"],
            "Selected smoke source size",
        )
    authority_ref = next(
        row for row in plan["sources"] if row["path"] == str(HERE / "authority.py")
    )
    # Parent mutates only the spawned child's environment, never this process environment.
    plan["authority_sha256"] = authority_ref["sha256"]
    protocol = load_source(
        HERE / "authority.py", authority_ref["sha256"], "_helper_smoke_parent_protocol"
    )
    run_parent(args, plan, protocol)


if __name__ == "__main__":
    main()
