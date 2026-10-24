"""HELD root-owned CPU-controller pump for the fixed matched V7 ten cases.

No process or artifact IO is authorized until the source/template hold releases.
The pump owns the original controller Popen outside the GPU unit. A stopped GPU
cgroup never substitutes for the original separately owned CPU closure.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import secrets
import signal
import stat
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

SOURCE_ONLY = True
TEMPLATE_SHA256 = None
ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
OWNER = ROOT / "artifacts/jev-runtime/llama-cpp-owner-20261024-09"
CONTROLLER = OWNER / "controller.py"
CONTROLLER_SHA256 = "81846efe2ee0357da07283c1a77a606283a920cbddef668c8f62def291057dee"
CPU_PROFILE = "llama-cpp-cpu5-fast-same-cluster-4g-coexist-command-owned-v1"
CPU_UNIT = "jev-d3-llama-cpp-cpu5-fast-4g-coexist-20261024-09.service"
ALIAS = "d3-edge-dsl01-operation-delta-cpu5-candidate"
KNOWN4 = (
    "public-arxiv-title-filter-results-v1",
    "public-arxiv-original-paper-open-v1",
    "public-python-counter-most-common-v1",
    "public-nasa-mars-facts-moons-v1",
)
PUBLIC6 = tuple(f"heldout-public-{index:03d}" for index in range(6))
CASE_IDS = KNOWN4 + PUBLIC6
MAX_BYTES, CLOSE_SECONDS = 1024**2, 45


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def is_sha(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def released():
    require(SOURCE_ONLY is False, "SOURCE_ONLY: root pump is held before IO or processes")
    require(is_sha(TEMPLATE_SHA256), "selected CPU template pin remains pending")


def ordinary(path):
    path = Path(path)
    path = path if path.is_absolute() else ROOT / path
    require(
        path.is_relative_to(ROOT) and ".." not in path.parts, "fixed repository pump path required"
    )
    require(
        not any(part.is_symlink() for part in (path, *path.parents)), "ordinary pump paths required"
    )
    return path


def decode(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, "duplicate pump metadata key")
            result[key] = value
        return result

    return json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_constant=lambda _value: require(False, "nonfinite metadata"),
    )


def encoded(value):
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode()
        + b"\n"
    )


def read(path, *, private=False):
    path = ordinary(path)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(before.st_mode) and 0 < before.st_size <= MAX_BYTES,
            "bounded regular pump metadata/source required",
        )
        if private:
            require(
                before.st_uid == os.getuid() and stat.S_IMODE(before.st_mode) == 0o600,
                "same-user mode0600 handoff required",
            )
        raw = stream.read(MAX_BYTES + 1)
        after = os.fstat(stream.fileno())
    keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(
        len(raw) == before.st_size
        and all(
            getattr(before, key) == getattr(after, key) == getattr(path.stat(), key) for key in keys
        ),
        "pump metadata/source changed while read",
    )
    return raw


def ref(path):
    path = ordinary(path)
    raw = read(path)
    return {"path": str(path), "sha256": sha(raw), "size_bytes": len(raw)}


def normalize(record):
    require(
        type(record) is dict
        and {"path", "sha256", "size_bytes"} <= set(record)
        and is_sha(record["sha256"])
        and type(record["size_bytes"]) is int
        and 0 < record["size_bytes"] <= MAX_BYTES,
        "exact bounded selected ref required",
    )
    return {
        "path": str(ordinary(record["path"])),
        "sha256": record["sha256"],
        "size_bytes": record["size_bytes"],
    }


def authenticate(record):
    record = normalize(record)
    raw = read(record["path"])
    require(
        len(raw) == record["size_bytes"] and sha(raw) == record["sha256"],
        "selected pump reference changed",
    )
    return raw


def relative(record):
    record = normalize(record)
    return {**record, "path": ordinary(record["path"]).relative_to(ROOT).as_posix()}


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish(path, value):
    """Complete mode0600 bytes become visible once; no final-file overwrite."""
    path = ordinary(path)
    parent = path.parent.stat()
    require(
        stat.S_ISDIR(parent.st_mode)
        and parent.st_uid == os.getuid()
        and stat.S_IMODE(parent.st_mode) == 0o700,
        "private same-user handoff directory required",
    )
    raw = encoded(value)
    require(len(raw) <= MAX_BYTES and not path.exists(), "new bounded handoff only")
    temporary = path.parent / (".pending-" + secrets.token_hex(16))
    primary = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        sync_directory(path.parent)
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except BaseException as secondary:
            if primary is None:
                raise
            primary.add_note("Pump temporary cleanup also failed: " + repr(secondary))
    return {"path": str(path), "sha256": sha(raw), "size_bytes": len(raw)}


def identity(pid):
    path = Path("/proc") / str(pid)
    raw = (path / "stat").read_text()
    fields = raw[raw.rfind(")") + 2 :].split()
    return {
        "pid": pid,
        "pgid": int(fields[2]),
        "sid": int(fields[3]),
        "start_ticks": int(fields[19]),
        "cgroup": (path / "cgroup").read_text().strip(),
    }


def load_controller(record):
    require(
        normalize(record)["path"] == str(CONTROLLER) and record["sha256"] == CONTROLLER_SHA256,
        "original controller09 only",
    )
    raw = authenticate(record)
    name = "_matched_pump_original_controller09"
    require(name not in sys.modules, "fresh original controller source import required")
    module = types.ModuleType(name)
    module.__file__ = str(CONTROLLER)
    sys.modules[name] = module
    try:
        exec(compile(raw, str(CONTROLLER), "exec", dont_inherit=True), vars(module))
        require(
            module.PROFILE == CPU_PROFILE and module.UNIT == CPU_UNIT,
            "original CPU owner profile/unit required",
        )
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def validate_template(template, plan):
    require(
        template["schema"] == "jev-v7-matched-cpu-service-template-v1"
        and template["status"] == "ROOT_SELECTED_TEMPLATE_ONLY",
        "selected nonexecuting CPU template required",
    )
    cases = template["cases"]
    require(
        type(cases) is list
        and len(cases) == 10
        and [row["case_id"] for row in cases] == list(CASE_IDS),
        "fixed ordered known4/public6 template required",
    )
    outputs, nonces = [], []
    for index, case in enumerate(cases, 1):
        output = ordinary(case["owner_output"])
        require(
            case["case_index"] == index
            and case["group"] == ("known4" if index <= 4 else "public6")
            and output.parent == OWNER
            and re.fullmatch(r"actual-[0-9]{2}", output.name)
            and type(case["nonce"]) is str
            and re.fullmatch(r"[0-9a-f]{32}", case["nonce"]),
            "fixed original CPU output/case nonce required",
        )
        outputs.append(str(output))
        nonces.append(case["nonce"])
    require(len(set(outputs)) == len(set(nonces)) == 10, "distinct CPU outputs/nonces required")
    spec = template["command_spec"]
    require(
        spec["profile"] == "llama-cpp-command-spec-v1"
        and spec["runtime_seconds"] == 7200
        and spec["file_size_max_bytes"] == 256 * 1024**2
        and len(spec["commands"]) == 1
        and len(spec["sources"]) <= 56,
        "original7200 CPU spec with room for two runtime refs required",
    )
    require(
        template["service_plan"]["cooperative_budget_seconds"] == 1200
        and template["service_plan"]["owner_hard_runtime_seconds"] == 7200,
        "separately declared service cooperative1200 required",
    )
    serial_source = template["service_plan"]["serial_forward"]
    selected_serial = normalize(plan["matched"]["serial_source"])
    require(
        set(serial_source) == {"path", "sha256"}
        and str(ordinary(serial_source["path"])) == selected_serial["path"]
        and serial_source["sha256"] == selected_serial["sha256"],
        "exact worker two-field serial source must match selected full source ref",
    )
    require(
        normalize(template["service_plan"]["candidate_capability"])
        == normalize(template["candidate_capability_ref"])
        and normalize(template["service_plan"]["numeric_acceptance"])
        == normalize(template["numeric_acceptance_ref"]),
        "same static service artifact/numeric/serialization source selection required",
    )
    capability = decode(authenticate(template["candidate_capability_ref"]))
    numeric = decode(authenticate(template["numeric_acceptance_ref"]))
    require(
        template["candidate_capability_ref"]["sha256"] == plan["matched"]["capability_sha256"]
        and capability["schema"] == "root-accepted-dsl01-operation-train-delta-v1"
        and capability["status"] == "ROOT_ACCEPTED_OPERATION_TRAIN_ONLY_READOUT"
        and capability["alias"] == plan["matched"]["candidate_alias"] == ALIAS
        and capability["only_head"] == "operation"
        and numeric["schema"] == "root-accepted-operation-cpu5-numeric-v1"
        and numeric["status"] == "accepted"
        and normalize(numeric["candidate_capability"])
        == normalize(template["candidate_capability_ref"])
        and numeric["candidate_alias"] == ALIAS,
        "same accepted operation artifact and numeric authority required",
    )
    return cases


def case_root(directory, index):
    return directory / "handoff" / ("case-" + str(index).zfill(2))


def compile_case(template, case, root_sha, directory, serialization, serialization_ref, parent):
    """Pure runtime expansion; the parent plan pins template/generator, not this output."""
    target = case_root(directory, case["case_index"])
    service = copy.deepcopy(template["service_plan"])
    service.update(
        case_index=case["case_index"],
        case_id=case["case_id"],
        root_plan_sha256=root_sha,
        parent_identity=copy.deepcopy(parent),
        owner_output=case["owner_output"],
        serialization=copy.deepcopy(serialization),
        serialization_ref=copy.deepcopy(serialization_ref),
        handoff={
            "ready_path": str(target / "service-ready.json"),
            "stop_path": str(target / "service-stop.json"),
            "closed_path": str(target / "service-closed.json"),
            "nonce": case["nonce"],
        },
    )
    service_path = target / "service-plan.json"
    service_raw = encoded(service)
    service_ref = {
        "path": str(service_path),
        "sha256": sha(service_raw),
        "size_bytes": len(service_raw),
    }
    spec = copy.deepcopy(template["command_spec"])
    substitutions = {
        "{SERVICE_PLAN}": str(service_path),
        "{SERVICE_PLAN_SHA256}": service_ref["sha256"],
    }
    for command in spec["commands"]:
        command["argv"] = [substitutions.get(value, value) for value in command["argv"]]
        require(
            not any("{SERVICE_PLAN" in value for value in command["argv"]),
            "service placeholders must be whole argv tokens",
        )
    spec["sources"] += [relative(service_ref), relative(serialization_ref)]
    require(len(spec["sources"]) <= 58, "unchanged58 selected-source bound")
    return service, service_ref, spec


def controller_argv(spec_ref, output):
    return [
        str(ROOT / ".venv/bin/python"),
        "-B",
        "-s",
        str(CONTROLLER),
        "--run",
        "--expected-entry-sha256",
        CONTROLLER_SHA256,
        "--selection",
        spec_ref["path"],
        "--selection-sha256",
        spec_ref["sha256"],
        "--output",
        str(output),
    ]


def binding(case, root_sha, service_sha):
    return {
        "case_index": case["case_index"],
        "case_id": case["case_id"],
        "root_plan_sha256": root_sha,
        "plan_sha256": service_sha,
        "nonce": case["nonce"],
    }


def service_stop(case, root_sha, service_sha):
    return {"schema": "jev-v7-full-choice-service-stop-v1", **binding(case, root_sha, service_sha)}


def verify_original(record, controller, *, complete):
    """Reauthenticate original CPU unit proof; no model, training or raw-task API."""
    terminal = decode(authenticate(record["owner_terminal_ref"]))
    receipt = decode(authenticate(record["owner_receipt_ref"]))
    closure = decode(authenticate(record["closure_ref"]))
    cfg_raw = authenticate(record["sealed_config_ref"])
    cfg = decode(cfg_raw)
    output = ordinary(record["owner_output"])
    spec = decode(authenticate(record["command_spec_ref"]))
    service = decode(authenticate(record["service_plan_ref"]))
    require(
        cfg == receipt["selected_config"]
        and cfg["profile"] == receipt["profile"] == CPU_PROFILE
        and cfg["unit"] == CPU_UNIT
        and cfg["output"] == str(output)
        and output.parent == OWNER
        and cfg["selection"] == record["command_spec_ref"]["path"]
        and cfg["selection_sha256"] == record["command_spec_ref"]["sha256"]
        and cfg["runtime_seconds"] == spec["runtime_seconds"] == 7200
        and cfg["boot_id"] == record["boot_id"]
        and receipt["closure"] == closure
        and receipt["sources_unchanged"] is True,
        "same original selected CPU config/receipt/sources required",
    )
    require(
        terminal["schema"] == "jev-v7-matched-controller-terminal-v1"
        and terminal["argv"] == controller_argv(record["command_spec_ref"], output)
        and terminal["exit_code"] == record["controller_exit_code"]
        and terminal["root_plan_sha256"] == record["root_plan_sha256"]
        and terminal["case_index"] == record["case_index"]
        and terminal["case_id"] == record["case_id"]
        and terminal["controller_identity"] == record["controller_identity"],
        "actual owned controller terminal join",
    )
    selected, actor = closure["selection"], closure["actor"]
    require(
        selected["unit"] == CPU_UNIT
        and selected["nonce"] == cfg["nonce"]
        and selected["boot_id"] == cfg["boot_id"]
        and selected["memory_max_bytes"] == 4 * 1024**3
        and selected["runtime_seconds"] == 7200
        and selected["stop_seconds"] == 3
        and closure["closure_mode"] == "authenticated_stop"
        and closure["closure_errors"] == []
        and type(closure["invocation_id"]) is str
        and len(closure["invocation_id"]) == 32
        and closure["before_stop"]["Description"] == cfg["nonce"]
        and closure["before_stop"]["InvocationID"] == closure["invocation_id"]
        and closure["before_stop"]["MainPID"] == str(actor["pid"])
        and all(
            closure[key] is True
            for key in (
                "whole_unit_exit_confirmed",
                "observed_cgroup_cohort_exit_confirmed",
                "post_stop_terminal_verified",
                "cgroup_absent",
                "sources_unchanged",
            )
        )
        and len(closure["two_process_scans"]) == 2
        and all(row["matches"] == row["unknown"] == [] for row in closure["two_process_scans"])
        and all(
            closure[key] is False
            for key in (
                "no_escape_enforced",
                "whole_tree_exit_confirmed",
                "continuation_authorized",
                "model_qualified",
            )
        ),
        "positive original authenticated CPU scope closure required",
    )
    controller.sources_match(cfg["sources"])
    expected_base = [
        controller.ref(CONTROLLER),
        controller.ref(record["command_spec_ref"]["path"]),
        *(controller.ref(ROOT / label) for label, _ in controller.FIXED.values()),
    ]
    require(
        len(cfg["sources"]) <= 64 and len(spec["sources"]) <= 58,
        "original source-union bounds required",
    )
    require(
        cfg["base_sources"] == expected_base
        and cfg["sources"] == [*expected_base, *spec["sources"]],
        "original exact base plus selected-source union required",
    )
    controller.command_selection(cfg["selection"], cfg["selection_sha256"], cfg["base_sources"])
    require(
        service["case_index"] == record["case_index"]
        and service["case_id"] == record["case_id"]
        and service["root_plan_sha256"] == record["root_plan_sha256"]
        and service["owner_output"] == str(output)
        and service["handoff"]["nonce"] == record["nonce"]
        and record["plan_sha256"] == record["service_plan_ref"]["sha256"],
        "same runtime service plan required",
    )
    if complete:
        admission = decode(authenticate(record["admission_ref"]))
        worker = decode(authenticate(record["worker_receipt_ref"]))
        local = decode(authenticate(record["service_closed_ref"]))
        require(
            terminal["exit_code"] == 0
            and receipt["completed"] is True
            and receipt["primary_error"] is None
            and receipt["secondary_errors"] == []
            and admission["config_sha256"] == sha(cfg_raw)
            and admission["nonce"] == cfg["nonce"]
            and admission["invocation_id"] == closure["invocation_id"]
            and controller.stable_process(admission["actor"]) == controller.stable_process(actor)
            and local["schema"] == "jev-v7-full-choice-service-closed-v1"
            and all(
                local[key] == value
                for key, value in binding(
                    record, record["root_plan_sha256"], record["plan_sha256"]
                ).items()
            )
            and local["owner_output"] == str(output)
            and local["service_success"] is True
            and local["server_thread_drained"] is True
            and local["final_native_artifact_authentication"] is True
            and local["sources_unchanged"] is True
            and local["worker_receipt"] == record["worker_receipt_ref"]["path"]
            and local["external_current_owner_closure_required"] is True
            and local["whole_tree_exit_confirmed"] is False
            and local["task_success"] is None
            and worker["schema"] == "jev-v7-full-choice-service-receipt-v1"
            and all(
                worker[key] == value
                for key, value in binding(
                    record, record["root_plan_sha256"], record["plan_sha256"]
                ).items()
            )
            and worker["owner_output"] == str(output)
            and record["worker_receipt_ref"]["path"]
            == str(output / "v7-full-choice-service/worker-receipt.json")
            and worker["service_success"] is True
            and worker["server_thread_drained"] is True
            and worker["final_native_artifact_authentication"] is True
            and worker["sources_unchanged"] is True
            and worker["cleanup_errors"] == []
            and worker["pending_native_or_server"] is False
            and worker["helper_model_loaded"] is False
            and worker["helper_model_calls"] == 0,
            "successful original controller, service and worker completion required",
        )
        results = receipt["state"]["actor_exit"]["command_results"]
        require(
            len(results) == 1
            and results[0]["returncode"] == 0
            and results[0]["argv"]
            == [value.replace("{OUTPUT}", str(output)) for value in spec["commands"][0]["argv"]]
            and controller.same_identity(results[0]["observed_identity"], worker["worker"]),
            "same actual original service child/process command result required",
        )
    return True


class Pump:
    def __init__(self, plan, directory, guard):
        released()
        self.directory, self.plan, self.guard = ordinary(directory), copy.deepcopy(plan), guard
        raw = read(self.directory / "selected-plan.json", private=True)
        require(decode(raw) == plan, "exact root-selected plan bytes required")
        self.root_sha, self.closed, self.active, self.records = sha(raw), False, None, []
        self.failure, self.gpu_invocation = None, None
        matched = plan["matched"]
        require(
            matched["schema"] == "jev-v7-matched-settings-v1"
            and ordinary(matched["handoff_root"]) == self.directory / "handoff"
            and ordinary(matched["serialization_path"]) == self.directory / "serialization.json"
            and matched["cpu_template"]["sha256"] == TEMPLATE_SHA256,
            "fixed selected campaign handoff/serialization/template required",
        )
        self.template_ref = normalize(matched["cpu_template"])
        self.template = decode(authenticate(self.template_ref))
        self.cases = validate_template(self.template, plan)
        self.controller = load_controller(self.template["controller_source"])
        require(
            Path("/proc/sys/kernel/random/boot_id").read_text().strip() == plan["boot_id"],
            "same actual root/CPU boot required",
        )
        (self.directory / "handoff").mkdir(mode=0o700, exist_ok=False)
        for case in self.cases:
            case_root(self.directory, case["case_index"]).mkdir(mode=0o700, exist_ok=False)
        paths, observed = {}, {}
        for name, filename in (("lock", "forward.lock"), ("journal", "forward-journal.jsonl")):
            path = self.directory / filename
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            try:
                os.fsync(descriptor)
                info = os.fstat(descriptor)
                observed[name] = {"device": info.st_dev, "inode": info.st_ino}
            finally:
                os.close(descriptor)
            paths[name] = str(path)
        self.serialization = {
            "lock_path": paths["lock"],
            "journal_path": paths["journal"],
            "expected_stat": observed,
            "nonce": secrets.token_hex(16),
        }
        self.serialization_ref = publish(matched["serialization_path"], self.serialization)
        sync_directory(self.directory)

    def _parent(self, request):
        intent = decode(read(self.directory / "actor-intent.json", private=True))
        actor = intent["actor"]
        state = self.guard.status(self.plan["unit"])
        require(
            intent["plan_sha256"] == request["root_plan_sha256"] == self.root_sha
            and intent["nonce"] == self.plan["nonce"]
            and intent["boot_id"] == self.plan["boot_id"]
            and intent["unit"] == self.plan["unit"]
            and request["parent_identity"] == actor
            and state["ControlGroup"] == self.plan["cgroup"]
            and state["ExecMainPID"] == str(actor["pid"])
            and type(state["InvocationID"]) is str
            and len(state["InvocationID"]) == 32,
            "request belongs to the admitted original GPU parent invocation",
        )
        current = identity(actor["pid"])
        require(
            all(current[key] == actor[key] for key in ("pid", "pgid", "sid", "start_ticks"))
            and current["cgroup"] == "0::" + self.plan["cgroup"],
            "fresh GPU actor identity/cgroup required before CPU launch",
        )
        require(
            self.gpu_invocation in (None, state["InvocationID"]), "same original GPU invocation"
        )
        self.gpu_invocation = state["InvocationID"]
        return actor

    def _start(self, case, request):
        require(
            request
            == {
                "schema": "jev-v7-matched-case-request-v1",
                "root_plan_sha256": self.root_sha,
                "case_index": case["case_index"],
                "case_id": case["case_id"],
                "group": case["group"],
                "parent_identity": request["parent_identity"],
            },
            "exact ordered case request",
        )
        parent = self._parent(request)
        require(
            sha(read(self.directory / "selected-plan.json", private=True)) == self.root_sha,
            "root plan changed",
        )
        authenticate(self.template_ref)
        validate_template(self.template, self.plan)
        authenticate(self.template["controller_source"])
        authenticate(self.serialization_ref)
        require(
            not ordinary(case["owner_output"]).exists(), "fresh original CPU owner output required"
        )
        service, service_ref, spec = compile_case(
            self.template,
            case,
            self.root_sha,
            self.directory,
            self.serialization,
            self.serialization_ref,
            parent,
        )
        target = case_root(self.directory, case["case_index"])
        require(publish(service_ref["path"], service) == service_ref, "same runtime service bytes")
        spec_ref = publish(target / "command-spec.json", spec)
        base_refs = [
            self.controller.ref(CONTROLLER),
            self.controller.ref(spec_ref["path"]),
            *(self.controller.ref(ROOT / label) for label, _ in self.controller.FIXED.values()),
        ]
        self.controller.command_selection(spec_ref["path"], spec_ref["sha256"], base_refs)
        argv = controller_argv(spec_ref, case["owner_output"])
        descriptor = os.open(target / "controller.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        stream = os.fdopen(descriptor, "wb")
        active = {
            "case": case,
            "target": target,
            "service_ref": service_ref,
            "spec_ref": spec_ref,
            "parent": parent,
            "argv": argv,
            "stream": stream,
            "process": None,
            "identity": None,
            "began": time.monotonic(),
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "ready": False,
        }
        self.active = active
        process = subprocess.Popen(
            argv,
            cwd=str(ROOT),
            stdin=subprocess.DEVNULL,
            stdout=stream,
            stderr=subprocess.STDOUT,
            shell=False,
            close_fds=True,
            start_new_session=True,
        )
        active["process"] = process  # Retain before any further read can fail.
        active["identity"] = identity(process.pid)
        publish(
            target / "start.json",
            {
                "schema": "jev-v7-matched-case-start-v1",
                **binding(case, self.root_sha, service_ref["sha256"]),
                "owner_output": case["owner_output"],
                "command_spec_ref": spec_ref,
                "controller_identity": active["identity"],
            },
        )

    def _ready(self, active):
        path, case = active["target"] / "service-ready.json", active["case"]
        if active["ready"] or not path.exists():
            return
        ready = decode(read(path, private=True))
        expected = binding(case, self.root_sha, active["service_ref"]["sha256"])
        require(
            ready["schema"] == "jev-v7-full-choice-service-ready-v1"
            and all(ready[key] == value for key, value in expected.items())
            and ready["owner_output"] == case["owner_output"]
            and ready["model"] == ALIAS,
            "same original CPU service-ready binding required",
        )
        address = urlsplit(ready["endpoint"])
        require(
            address.scheme == "http"
            and address.hostname in {"127.0.0.1", "localhost", "::1"}
            and address.port is not None
            and address.path == "/v1/systemone"
            and not address.query
            and not address.fragment
            and address.username is None,
            "original random loopback endpoint only",
        )
        output = ordinary(case["owner_output"])
        cfg_raw = read(output / "sealed-config.json")
        cfg, admission = decode(cfg_raw), decode(read(output / "admission.json"))
        owned, worker = ready["verify_owned"], ready["worker_identity"]
        capability = decode(authenticate(self.template["candidate_capability_ref"]))
        require(
            cfg["selection_sha256"] == active["spec_ref"]["sha256"]
            and cfg["profile"] == CPU_PROFILE
            and cfg["unit"] == CPU_UNIT
            and cfg["output"] == str(output)
            and cfg["boot_id"] == self.plan["boot_id"]
            and cfg["selection"] == active["spec_ref"]["path"]
            and admission["config_sha256"] == sha(cfg_raw)
            and admission["nonce"] == cfg["nonce"] == owned["nonce"]
            and admission["invocation_id"] == owned["invocation_id"]
            and owned["schema"] == "owned-decision3-cpu5-browser-binding-v1"
            and owned["owner_profile"] == CPU_PROFILE
            and owned["owner_unit"] == CPU_UNIT
            and owned["selection_sha256"] == cfg["selection_sha256"]
            and owned["config_sha256"] == sha(cfg_raw)
            and owned["plan_sha256"] == active["service_ref"]["sha256"]
            and owned["plan_path"] == active["service_ref"]["path"]
            and owned["worker_pid"] == worker["process"]["pid"]
            and owned["worker_start_ticks"] == worker["process"]["start_ticks"]
            and owned["memory_max_bytes"] == 4 * 1024**3
            and owned["swap_max_bytes"] == 0
            and owned["cpu_affinity"] == [15, 16, 17, 18, 19]
            and owned["cpu_quota_percent"] == 500
            and owned["expected_threads"] == 5
            and owned["expected_interop_threads"] == 1
            and owned["device"] == "cpu"
            and owned["precision"] == "bfloat16"
            and owned["context_limit"] == 16384
            and owned["native_model"] == capability["model"]
            and all(
                worker["process"][key] == admission["actor"][key]
                for key in ("pgid", "sid", "uid", "cgroup")
            ),
            "ready must join original CPU admission/config/worker identity and resources",
        )
        self.controller.sources_match(cfg["sources"])
        actual = identity(worker["process"]["pid"])
        require(
            all(
                actual[key] == worker["process"][key]
                for key in ("pid", "start_ticks", "pgid", "sid")
            )
            and actual["cgroup"] == "0::" + worker["process"]["cgroup"],
            "fresh actual ready worker required",
        )
        publish(
            active["target"] / "ready.json",
            {
                "schema": "jev-v7-matched-case-ready-v1",
                **expected,
                "owner_output": case["owner_output"],
                "endpoint": ready["endpoint"],
                "model": ready["model"],
                "service_ready_ref": ref(path),
            },
        )
        active["ready"] = True

    def _stop(self, active, *, forced=False):
        case, path = active["case"], active["target"] / "stop.json"
        target = active["target"] / "service-stop.json"
        if target.exists():
            require(
                decode(read(target, private=True))
                == service_stop(case, self.root_sha, active["service_ref"]["sha256"]),
                "same immutable service stop",
            )
            return
        if not forced:
            if not path.exists():
                return
            row = decode(read(path, private=True))
            require(
                row["schema"] == "jev-v7-matched-case-stop-v1"
                and all(
                    row[key] == value
                    for key, value in binding(
                        case, self.root_sha, active["service_ref"]["sha256"]
                    ).items()
                )
                and set(row)
                == {
                    "schema",
                    "root_plan_sha256",
                    "case_index",
                    "case_id",
                    "plan_sha256",
                    "nonce",
                    "caller_record_ref",
                },
                "exact original callback stop binding required",
            )
            if row["caller_record_ref"] is not None:
                authenticate(row["caller_record_ref"])
        publish(target, service_stop(case, self.root_sha, active["service_ref"]["sha256"]))

    def _finish(self, active, *, complete):
        process, case, target = active["process"], active["case"], active["target"]
        code = process.poll()
        require(type(code) is int, "held original CPU controller must actually exit")
        if not active["stream"].closed:
            active["stream"].flush()
            os.fsync(active["stream"].fileno())
            active["stream"].close()
        if "terminal_ref" not in active:
            active["terminal_ref"] = publish(
                target / "controller-terminal.json",
                {
                    "schema": "jev-v7-matched-controller-terminal-v1",
                    "argv": active["argv"],
                    "exit_code": code,
                    "controller_identity": active["identity"],
                    "started_at_utc": active["started_at_utc"],
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "wall_seconds": time.monotonic() - active["began"],
                    "root_plan_sha256": self.root_sha,
                    "case_index": case["case_index"],
                    "case_id": case["case_id"],
                },
            )
        terminal_ref = active["terminal_ref"]
        output = ordinary(case["owner_output"])
        record = {
            "schema": "jev-v7-matched-case-closed-v1",
            **binding(case, self.root_sha, active["service_ref"]["sha256"]),
            "boot_id": self.plan["boot_id"],
            "owner_output": str(output),
            "controller_exit_code": code,
            "owner_completed": False,
            "original_scoped_closure_positive": False,
            "parent_identity": active["parent"],
            "owner_terminal_ref": terminal_ref,
            "controller_identity": active["identity"],
            "owner_receipt_ref": ref(output / "receipt.json"),
            "closure_ref": ref(output / "external-boundary-closure.json"),
            "sealed_config_ref": ref(output / "sealed-config.json"),
            "command_spec_ref": active["spec_ref"],
            "service_plan_ref": active["service_ref"],
            "worker_receipt_ref": None,
            "admission_ref": None,
            "service_closed_ref": None,
            "interrupted_for_cleanup": not complete,
        }
        for key, path in (
            ("admission_ref", output / "admission.json"),
            ("worker_receipt_ref", output / "v7-full-choice-service/worker-receipt.json"),
            ("service_closed_ref", target / "service-closed.json"),
        ):
            if path.exists():
                record[key] = ref(path)
        verify_original(record, self.controller, complete=complete)
        owner_receipt = decode(authenticate(record["owner_receipt_ref"]))
        record.update(
            owner_completed=owner_receipt["completed"], original_scoped_closure_positive=True
        )
        publish(target / "closed.json", record)
        self.records.append(record)
        self.active = None
        if complete:
            require(
                code == 0 and record["owner_completed"] is True,
                "failed original controller cannot authorize next case",
            )
        return record

    def poll(self):
        released()
        require(
            not self.closed and self.failure is None, "closed or failed root pump cannot advance"
        )
        try:
            if self.active is not None:
                active = self.active
                self._ready(active)
                self._stop(active)
                if active["process"].poll() is not None:
                    require(
                        (active["target"] / "stop.json").exists() and active["ready"],
                        "controller ended before original ready/callback stop",
                    )
                    self._finish(active, complete=True)
                return
            index = len(self.records) + 1
            if index > 10:
                return
            for later in range(index + 1, 11):
                require(
                    not (case_root(self.directory, later) / "request.json").exists(),
                    "original case order only; no concurrent or skipped tasks",
                )
            path = case_root(self.directory, index) / "request.json"
            if path.exists():
                self._start(self.cases[index - 1], decode(read(path, private=True)))
        except BaseException as error:
            self.failure = error
            raise

    def close(self):
        released()
        if self.closed:
            return self.final
        cleanup_primary = None

        def keep(error):
            nonlocal cleanup_primary
            if cleanup_primary is None:
                cleanup_primary = error
            elif error is not cleanup_primary:
                cleanup_primary.add_note("Root CPU cleanup also failed: " + repr(error))

        if self.active is not None:
            active = self.active
            process = active["process"]
            if process is None and not active["stream"].closed:
                active["stream"].close()
            require(
                process is not None, "CPU process creation was incomplete; explicit pending cleanup"
            )
            if process.poll() is None:
                try:
                    self._stop(active, forced=True)
                except BaseException as error:
                    keep(error)  # A broken handoff must not prevent original controller cleanup.
                try:
                    try:
                        current = identity(process.pid)
                    except FileNotFoundError:
                        require(
                            process.poll() is not None,
                            "held controller identity unavailable while live",
                        )
                        current = None
                    if current is not None:
                        if active["identity"] is None:
                            require(
                                current["pid"] == process.pid
                                and current["pid"] == current["pgid"] == current["sid"],
                                "held new-session Popen identity before cleanup",
                            )
                            active["identity"] = current
                        require(current == active["identity"], "same held controller before SIGINT")
                        process.send_signal(
                            signal.SIGINT
                        )  # Its original finally closes its CPU unit.
                    until = time.monotonic() + CLOSE_SECONDS
                    while process.poll() is None:
                        require(
                            time.monotonic() < until,
                            "CPU controller remains live: no kill/closure claim; original cleanup pending",
                        )
                        time.sleep(0.05)
                except BaseException as error:
                    keep(error)
            if process.poll() is not None:
                try:
                    self._finish(active, complete=False)
                except BaseException as error:
                    keep(error)
            if self.active is not None:
                if cleanup_primary is not None:
                    raise cleanup_primary
                raise RuntimeError(
                    "CPU controller closure remains pending; no positive pump result"
                )
        if cleanup_primary is not None and self.failure is None:
            self.failure = cleanup_primary
        self.final = {
            "schema": "jev-v7-matched-root-pump-closed-v1",
            "children": self.records,
            "active_controller_absent": self.active is None,
            "independently_closed": all(
                row["original_scoped_closure_positive"] is True for row in self.records
            ),
            "complete_ten_cases": len(self.records) == 10
            and all(
                row["owner_completed"] is True
                and row["controller_exit_code"] == 0
                and row["interrupted_for_cleanup"] is False
                for row in self.records
            ),
            "failure": None
            if self.failure is None
            else {"type": type(self.failure).__name__, "message": str(self.failure)},
            "independent_actual_review_claimed": False,
            "scope": "Original separate CPU units/cohorts only; outer40GiB does not cover siblings",
        }
        self.closed = True
        if cleanup_primary is not None:
            cleanup_primary.jev_v7_pump_closure = self.final
            raise cleanup_primary
        return self.final


def verify_children(result, arm, plan, output):
    """Outer owner delegate: independently reread every original child proof ref."""
    released()
    directory = ordinary(output).parent
    root_raw = read(directory / "selected-plan.json", private=True)
    require(decode(root_raw) == plan, "same sealed outer plan for closed child verification")
    root_sha = sha(root_raw)
    template_ref = plan["matched"]["cpu_template"]
    require(template_ref["sha256"] == TEMPLATE_SHA256, "same selected CPU template pin")
    template = decode(authenticate(template_ref))
    cases = validate_template(template, plan)
    controller = (
        load_controller(template["controller_source"])
        if "_matched_pump_original_controller09" not in sys.modules
        else sys.modules["_matched_pump_original_controller09"]
    )
    authenticate(template["controller_source"])
    require(
        controller.__file__ == str(CONTROLLER)
        and controller.PROFILE == CPU_PROFILE
        and controller.UNIT == CPU_UNIT,
        "same trusted process-local original controller module",
    )
    pump = result["candidate_pump_closure"]
    records = arm["children"]
    require(
        pump["children"] == records
        and len(records) == 10
        and pump["active_controller_absent"] is True
        and pump["independently_closed"] is True
        and pump["complete_ten_cases"] is True
        and pump["failure"] is None,
        "all original ten independently owned CPU cases closed without a live controller",
    )
    serialization_ref = ref(directory / "serialization.json")
    serialization = decode(authenticate(serialization_ref))
    for record, case in zip(records, cases):
        require(
            record["schema"] == "jev-v7-matched-case-closed-v1"
            and record["case_index"] == case["case_index"]
            and record["case_id"] == case["case_id"]
            and record["root_plan_sha256"] == root_sha
            and record["nonce"] == case["nonce"]
            and record["owner_output"] == case["owner_output"]
            and record["boot_id"] == plan["boot_id"]
            and record["parent_identity"] == result["actor"]
            and record["owner_completed"] is True
            and record["controller_exit_code"] == 0
            and record["original_scoped_closure_positive"] is True
            and record["interrupted_for_cleanup"] is False,
            "exact original case/root/parent/CPU closed record required",
        )
        service, service_ref, spec = compile_case(
            template,
            case,
            root_sha,
            directory,
            serialization,
            serialization_ref,
            record["parent_identity"],
        )
        require(
            service_ref == normalize(record["service_plan_ref"])
            and decode(authenticate(record["service_plan_ref"])) == service
            and decode(authenticate(record["command_spec_ref"])) == spec,
            "original selected template runtime expansion is unchanged",
        )
        verify_original(record, controller, complete=True)
        try:
            current = identity(record["controller_identity"]["pid"])
        except FileNotFoundError:
            current = None
        require(
            current is None or current != record["controller_identity"],
            "original owned root CPU controller still exists",
        )
    return {
        "original_cpu_children": 10,
        "original_scoped_closures_positive": True,
        "outer_memcg_covers_candidate_siblings": False,
        "independent_actual_review_claimed": False,
    }


def main(_argv=None):
    released()
    raise RuntimeError("Only the pinned integrated root owner may construct this pump")


if __name__ == "__main__":
    raise SystemExit(main())
