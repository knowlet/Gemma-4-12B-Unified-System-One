"""Fresh selected Q95 owner with bounded host-refusal diagnostics.

Only explicit SHA-selected launch can create its own fifth unit/service slot.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[2]
FROZEN = ROOT / "artifacts/jev-runtime/decision3-q95-frozen-root-20261022-01"
SERVICE = FROZEN / "serve-05"
PROFILE = "decision3-q95-guarded-owner-host-diagnostics-v3"
UNIT = "jev-d3-q95-helper-20261023-05.service"
BASE, GIB = 23919549408, 1024**3
FIXED = {
    "boundary": (
        "decision3-local-owner-terminal-20261022-01/owner_boundary.py",
        "7a50c47d6d7f696973c666cb7a40ce47dd0a7624667a18422c8c60c0db72e4e5",
    ),
    "observer": (
        "observer-system-services-20261022-01/comfy_observer.py",
        "fd977040d659770e26a8fd6cc2b94e6c5440a6a3d3f05099e07d7597d8309e45",
    ),
    "observer_manifest": (
        "observer-system-services-20261022-01/system-services.json",
        "a2446af344f084dcf556256fd04c9754fcba940f33b3866990de85a7374c0727",
    ),
    "checker": (
        "decision3-local-helper-budget-20261022-01/live_helper.py",
        "85f2eef27f4455710635514feebe90611a52b22c6e6781da09899dfaa27482ba",
    ),
    "serve": (
        "decision3-q95-frozen-root-20261022-01/serve-plan-02.json",
        "a14708dbc42ff8526ada639f989ba9effe633c7f64d7dd5c4abd66ba3ff03247",
    ),
    "assembly": (
        "decision3-q95-frozen-root-20261022-01/assembly-inventory.json",
        "1e3c2dcd631f24e614df6d183cf5b3d8237e1a8a8284501c5ae4e1eb56570df6",
    ),
    "tensor": (
        "decision3-q95-frozen-root-20261022-01/tensor-link-01.json",
        "f077b17fd0c13adee724b4649284dedd134dc5d3579264d99b1e6011dcba1ff4",
    ),
    "errors": (
        "overlay/starlette/middleware/errors.py",
        "6e899351f408a3b088c27650f1f173e0108363989d427aecae1192a593621feb",
    ),
}


def need(value, message):
    if not value:
        raise ValueError(message)


def ordinary(path):
    path = Path(path).absolute()
    need(
        ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "Ordinary selected path",
    )
    return path


def encode(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"


def read(path, limit=4 * 1024**2):
    path = ordinary(path)
    with path.open("rb") as stream:
        blob = stream.read(limit + 1)
    need(len(blob) <= limit, "Bounded selected metadata")
    return blob


def pin(path):
    path = ordinary(path)
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while block := stream.read(1024**2):
            digest.update(block)
            size += len(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": size}


def check_ref(ref):
    need(type(ref) is dict and set(ref) == {"path", "sha256", "size_bytes"}, "Exact reference")
    need(
        Path(ref["path"]).is_absolute()
        and type(ref["size_bytes"]) is int
        and ref["size_bytes"] >= 0,
        "Absolute typed reference",
    )
    need(pin(ref["path"]) == ref, "Selected source drift")
    return ordinary(ref["path"])


def relative_ref(ref):
    return {**ref, "path": ordinary(ref["path"]).relative_to(ROOT).as_posix()}


def load(ref, name):
    path = check_ref(ref)
    blob = read(path)
    need(hashlib.sha256(blob).hexdigest() == ref["sha256"], "Consumed selected source")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(blob, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


def stamp(path):
    s = ordinary(path).stat()
    need(s.st_mode & 0o170000 == 0o100000, "Ordinary regular source")
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def population(plan, serve):
    assembly = json.loads(read(check_ref(plan["components"]["assembly"])))
    need(len(assembly["files"]) == 119, "Complete frozen assembly119")
    source = json.loads(read(ROOT / serve["execution"]["path"]))["files"]
    runtime = json.loads(read(ROOT / serve["runtime"]["path"]))["files"]
    need(len(source) == 95 and len(runtime) == 348, "Exact source95/runtime348")
    need(
        all(assembly["files"].get(k) == v for k, v in source.items()),
        "Assembly retains all exact95",
    )
    tensor = json.loads(read(check_ref(plan["components"]["tensor"])))
    tensor_label = ordinary(tensor["target"]).relative_to(FROZEN).as_posix()
    need(
        serve["source_files"]
        == {**assembly["files"], tensor_label: tensor["expected_payload_pin"]},
        "Exact assembly119 plus separately selected tensor pin",
    )
    result = {str(FROZEN / k): {"path": str(FROZEN / k), **v} for k, v in assembly["files"].items()}
    for path, value in runtime.items():
        need(Path(path).is_absolute(), "Original absolute runtime origins")
        result[path] = {"path": path, **value}
    for ref in plan["components"].values():
        result[ref["path"]] = ref
    for name in ("qualification", "execution", "runtime", "design", "inputs"):
        ref = {**serve[name], "path": str(ROOT / serve[name]["path"])}
        result[ref["path"]] = ref
    return [result[key] for key in sorted(result)]


def check_population(refs, *, expected_stamps=None):
    reject_project_bytecode()
    result = {}
    for ref in refs:
        before = stamp(ref["path"])
        if expected_stamps is None:
            check_ref(ref)
        else:
            need(before == expected_stamps[ref["path"]], "Source metadata changed during service")
        need(before == stamp(ref["path"]), "Source changed during observation")
        result[ref["path"]] = before
    return result


def reject_project_bytecode():
    # -B prohibits writes, but Python can still consume a previously valid pyc.
    # These are exactly the selected copied project source roots, not an
    # installed-runtime or universal native-loader attestation claim.
    for root in (FROZEN / "src", FROZEN / "scripts"):
        if not root.exists():
            continue
        count = 0
        for path in root.rglob("*"):
            count += 1
            need(count <= 4096, "Bounded copied project source inventory")
            need(not path.is_symlink(), "Ordinary copied project source tree")
            need(
                path.name != "__pycache__" and path.suffix not in {".pyc", ".pyo"},
                "Cached project bytecode is not selected source",
            )


def tensor_identity(plan):
    value = json.loads(read(check_ref(plan["components"]["tensor"])))
    rows = []
    for key in ("source", "target"):
        path = ordinary(value[key])
        s = path.stat()
        need(
            [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns]
            == [value[key + "_after"][k] for k in ("device", "inode", "size_bytes", "mtime_ns")],
            "Selected confined tensor identity",
        )
        rows.append(stamp(path))
    need(rows[0] == rows[1] == plan["tensor_stat"], "Tensor alias/stat drift")
    return rows


def selection(plan, owner):
    result = owner.Selection(plan["unit"], plan["nonce"], plan["boot_id"], 40 * GIB, 10800, 20)
    need(result.cgroup == plan["cgroup"], "Selected exact helper cgroup")
    return result


def selected_serve(serve):
    """Declare one output-only retry; qualified controller/listener stay unchanged."""
    argv = list(serve["argv"])
    need(argv.count("--output") == 1, "One original qualified output argument")
    index = argv.index("--output") + 1
    need(
        index < len(argv) and argv[index] == str(FROZEN / "serve-01"),
        "Original selected serve01 argument",
    )
    argv[index] = str(SERVICE)
    return {**serve, "argv": argv}


def validate(path, digest):
    need(
        type(digest) is str and re.fullmatch("[0-9a-f]{64}", digest),
        "External selected plan digest",
    )
    blob = read(path, 1024**2)
    need(hashlib.sha256(blob).hexdigest() == digest, "Selected plan bytes")
    plan = json.loads(blob)
    need(
        set(plan)
        == set(
            "profile unit nonce boot_id mem_total_bytes created_monotonic expires_monotonic cgroup output components tensor_stat".split()
        ),
        "Exact owner plan schema",
    )
    need(plan["profile"] == PROFILE and plan["unit"] == UNIT, "Exact helper owner profile/unit")
    need(re.fullmatch("jev-d3-boundary-[0-9a-f]{32}", plan["nonce"]), "Fresh selected nonce")
    need(
        type(plan["mem_total_bytes"]) is int and plan["mem_total_bytes"] > 0,
        "Selected positive host total",
    )
    need(
        type(plan["expires_monotonic"]) in (float, int)
        and type(plan["created_monotonic"]) in (float, int)
        and 0
        <= plan["created_monotonic"]
        < plan["expires_monotonic"]
        <= plan["created_monotonic"] + 3600,
        "Bounded plan freshness",
    )
    need(
        ordinary(plan["output"]).parent == HERE and Path(plan["output"]).name == "owner-05",
        "Scoped fresh owner output",
    )
    need(set(plan["components"]) == set(FIXED) | {"owner", "entry"}, "Complete selected components")
    for name, (label, sha) in FIXED.items():
        ref = plan["components"][name]
        need(
            ref["path"] == str(ROOT / "artifacts/jev-runtime" / label) and ref["sha256"] == sha,
            "Fixed reviewed component",
        )
    for name, filename in (("owner", "q95_owner.py"), ("entry", "launch_q95.py")):
        need(plan["components"][name]["path"] == str(HERE / filename), "Selected owner entry path")
    for ref in plan["components"].values():
        check_ref(ref)
    serve = json.loads(read(plan["components"]["serve"]["path"]))
    need(
        serve["source_root"] == serve["cwd"] == str(FROZEN)
        and serve["source_count"] == 95
        and serve["runtime_file_count"] == 348,
        "Frozen qualified root identity",
    )
    refs = population(plan, serve)
    check_population(refs)
    tensor_identity(plan)
    return plan, selected_serve(serve), refs, blob


def host_field(row, key):
    """Typed scalar only; no arbitrary observer evidence or object stringification."""
    if key not in row:
        return {"present": False}
    value = row[key]
    result = {"present": True, "type": type(value).__name__[:64]}
    if value is None or type(value) in (bool, int):
        # Bound malformed integer evidence while preserving ordinary host facts.
        if type(value) is int and value.bit_length() > 128:
            result["integer_bits"] = value.bit_length()
        else:
            result["value"] = value
    elif type(value) is str:
        result["value"] = value[:128]
        result["truncated"] = len(value) > 128
    return result


def host_gate_evidence(plan, row, *, initial, gate):
    reserve = max(24 * GIB, (plan["mem_total_bytes"] + 4) // 5)
    margin = 2 * BASE if initial else 2 * GIB
    return {
        "schema_id": "q95-host-gate-refusal-v1",
        "gate": gate,
        "phase": "initial" if initial else "runtime",
        "observed": {
            key: host_field(row, key)
            for key in (
                "boot_id",
                "mem_total_bytes",
                "mem_available_bytes",
                "comfy_state",
                "comfy_running",
                "comfy_pending",
            )
        },
        "thresholds": {
            "boot_id": plan["boot_id"],
            "mem_total_bytes": plan["mem_total_bytes"],
            "reserve_bytes": reserve,
            "margin_bytes": margin,
            "min_mem_available_bytes": reserve + margin,
        },
    }


def emit_host_refusal(evidence):
    """Also retain pre-output startup refusals in the caller's stderr capture."""
    try:
        blob = (
            json.dumps(
                {"host_gate_refusal": evidence},
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
            + b"\n"
        )
        if len(blob) <= 4096:
            os.write(2, blob)
    except OSError:
        # Diagnostic transport failure cannot replace or relax the original gate.
        pass


def host(plan, observer, *, initial=False):
    row = observer.observe(expected_boot_id=plan["boot_id"])
    gate = "host_identity_and_comfy_idle"
    try:
        need(
            row.get("boot_id") == plan["boot_id"]
            and row.get("mem_total_bytes") == plan["mem_total_bytes"]
            and type(row.get("mem_available_bytes")) is int
            and 0 < row["mem_available_bytes"] <= plan["mem_total_bytes"]
            and row.get("comfy_state") in {"http_idle", "absent_confirmed"}
            and type(row.get("comfy_running")) is int
            and row["comfy_running"] == 0
            and type(row.get("comfy_pending")) is int
            and row["comfy_pending"] == 0,
            "Explicit same-host idle/absent observation",
        )
        gate = "memory_reserve"
        reserve = max(24 * GIB, (plan["mem_total_bytes"] + 4) // 5)
        need(
            row["mem_available_bytes"] >= reserve + (2 * BASE if initial else 2 * GIB),
            "Selected host reserve gate",
        )
    except ValueError as error:
        evidence = host_gate_evidence(plan, row, initial=initial, gate=gate)
        error.evidence = evidence
        emit_host_refusal(evidence)
        raise
    return row


def error_record(error):
    value = {"type": type(error).__name__, "message": "Selected Q95 owner stage refused"}
    evidence = getattr(error, "evidence", None)
    if evidence is not None:
        blob = encode(evidence)
        value["observation_evidence"] = (
            evidence
            if len(blob) <= 65536
            else {
                "sha256": hashlib.sha256(blob).hexdigest(),
                "size_bytes": len(blob),
                "truncated": True,
            }
        )
    return value


def actor_admission(plan, digest, owner):
    out = Path(plan["output"])
    value = json.loads(read(out / "admission.json"))
    selected = selection(plan, owner)
    row = owner.status(selected)
    need(
        owner.boot() == plan["boot_id"]
        and os.environ.get("JEV_D3_BOUNDARY_NONCE") == plan["nonce"],
        "Selected actor boot/nonce",
    )
    need(
        value["plan_sha256"] == digest
        and value["nonce"] == plan["nonce"]
        and value["invocation_id"] == row.get("InvocationID")
        and row.get("Description") == plan["nonce"]
        and row.get("ControlGroup") == selected.cgroup
        and row.get("MainPID") == str(os.getpid())
        and row.get("ActiveState") == "active"
        and row.get("SubState") == "running",
        "Actual actor owner admission",
    )
    need(
        all(row.get(k) == v for k, v in selected.properties().items()),
        "Actual selected unit properties",
    )
    actual = owner.process(os.getpid())
    need(
        all(
            actual[k] == value["actor"][k]
            for k in ("pid", "start_ticks", "pgid", "sid", "uid", "cgroup")
        )
        and actual["pid"] == actual["pgid"] == actual["sid"],
        "Exact admitted actor identity",
    )
    group = owner.CGROUP_ROOT / selected.cgroup.lstrip("/")
    need(
        (group / "memory.max").read_text().strip() == str(40 * GIB)
        and (group / "memory.swap.max").read_text().strip() == "0",
        "Fresh actor kernel caps",
    )
    return value


def actor_main(plan, serve, refs, digest, owner):
    if os.getpid() != os.getsid(0):
        os.setsid()
    out = Path(plan["output"])
    owner.save(
        out, "actor-intent.json", {"plan_sha256": digest, "actor": owner.process(os.getpid())}
    )
    until = time.monotonic() + 60
    while not (out / "admission.json").exists():
        need(time.monotonic() < until, "External owner admission timeout")
        time.sleep(0.05)
    actor_admission(plan, digest, owner)
    need(time.monotonic() < plan["expires_monotonic"], "Fresh actual launch plan")
    observer = load(plan["components"]["observer"], "_q95_actor_observer")
    host(plan, observer, initial=True)
    check_population(refs)
    tensor_identity(plan)
    env = {
        k: v
        for k, v in os.environ.items()
        if not any(x in k.upper() for x in ("TOKEN", "SECRET", "PASSWORD", "API_KEY"))
    }
    for key in serve["environment_unset"]:
        env.pop(key, None)
    env.update(serve["environment_updates"])
    env.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    primary, child = None, None
    try:
        with (out / "qualified-controller.log").open("xb") as log:
            child = subprocess.Popen(
                serve["argv"], cwd=serve["cwd"], env=env, stdout=log, stderr=log
            )
            controller = owner.process(child.pid)
            raw = Path(f"/proc/{child.pid}/stat").read_text()
            need(
                int(raw[raw.rfind(")") + 2 :].split()[1]) == os.getpid()
                and controller["cgroup"] == plan["cgroup"],
                "Actual direct original controller",
            )
            owner.save(
                out, "controller-started.json", {"plan_sha256": digest, "controller": controller}
            )
            while child.poll() is None:
                need(
                    time.monotonic() < plan["created_monotonic"] + 10720,
                    "Actor independent cooperative deadline",
                )
                actor_admission(plan, digest, owner)
                if (out / "shutdown-intent.json").exists():
                    request = SERVICE / "shutdown.request"
                    if not request.exists():
                        request.parent.mkdir(parents=True, exist_ok=True)
                        with request.open("xb") as stream:
                            stream.write(
                                encode(
                                    {
                                        "requested_by": "selected_external_owner",
                                        "plan_sha256": digest,
                                    }
                                )
                            )
                time.sleep(0.2)
        need(child.returncode == 0, "Original qualified controller failed")
        receipt = json.loads(read(SERVICE / "controller-receipt.json"))
        need(
            receipt.get("stage") == "complete" and receipt.get("owned_exit_confirmed") is True,
            "Original controller terminal receipt",
        )
    except BaseException as error:
        primary = error_record(error)
    finally:
        owner.save(
            out,
            "actor-result.json",
            {
                "plan_sha256": digest,
                "primary_error": primary,
                "controller_exit_confirmed": child is not None and child.poll() is not None,
                "controller_returncode": None if child is None else child.poll(),
                "external_scoped_closure_required": True,
                "model_qualified": False,
            },
        )
    while True:
        time.sleep(1)


def helper_identity(plan, serve, owner, checker, boundary):
    service = SERVICE
    endpoint = json.loads(read(service / "endpoint.json"))
    config_path = service / "controller-config.json"
    config = checker.parse(checker.read(config_path, private=True))
    need(
        config.get("command") == "serve"
        and config.get("output") == str(service)
        and config.get("qualification_sha256") == checker.Q_SHA
        and config.get("execution_sha256") == checker.E_SHA
        and config.get("runtime_sha256") == checker.R_SHA
        and config.get("boot_id") == plan["boot_id"]
        and config.get("mem_total") == plan["mem_total_bytes"],
        "Actual original private helper profile",
    )
    expected = [
        serve["argv"][0],
        str(FROZEN / "scripts/jev_qualify_public_field_syntax_bf16.py"),
        "--internal-child",
        str(config_path),
    ]
    candidates = []
    for member in boundary.members()["members"]:
        try:
            row = checker.process(member["pid"])
        except (FileNotFoundError, ProcessLookupError):
            continue
        if row["argv"] == expected:
            candidates.append(row)
    need(len(candidates) == 1, "One actual original helper process")
    worker = candidates[0]
    need(
        worker["pid"] == worker["pgid"] == worker["sid"] and worker["cgroup"] == plan["cgroup"],
        "Original helper own session within selected cgroup",
    )
    from urllib.parse import urlsplit

    url = urlsplit(endpoint["helper"])
    need(
        url.scheme == "http"
        and url.hostname == "127.0.0.1"
        and url.path == ""
        and type(url.port) is int,
        "Actual original loopback endpoint",
    )
    listener = checker.listener(worker["pid"], url.port)
    binding = {
        "profile": checker.PROFILE,
        "qualification_sha256": checker.Q_SHA,
        "source_count": 95,
        "boot_id": plan["boot_id"],
        **{k: worker[k] for k in ("pid", "start_ticks", "pgid", "sid")},
        "endpoint": endpoint["helper"],
        "seconds": 20,
        "tokens": 128,
        "context": 8192,
    }
    need(checker.process(worker["pid"]) == worker, "Fresh helper identity sandwich")
    return {
        "schema_id": "decision3-q95-owner-live-identity-seed-v1",
        "owner_plan_sha256": plan["_digest"],
        "binding": binding,
        "worker": {k: v for k, v in worker.items() if k != "uid"},
        "listener": listener,
        "service_directory": str(service),
        "private_config": str(config_path),
        "interpreter": expected[0],
        "source_root": str(FROZEN),
        **{k: serve[k] for k in ("qualification", "execution", "runtime")},
        "registry": {
            "path": "src/s1/evaluation/decision3.py",
            **{
                k: v for k, v in pin(ROOT / "src/s1/evaluation/decision3.py").items() if k != "path"
            },
        },
        "expected_memtotal": plan["mem_total_bytes"],
        "authentication_admitted": False,
        "co_resident_admitted": False,
        "model_qualified": False,
        "scope": "Public actual identity seed only. Fresh b20 checker with selected D3 co_resident is still mandatory; no credentials serialized.",
    }


def close_requested(plan, digest):
    path = Path(plan["output"]) / "close-request.json"
    if not path.exists():
        return False
    need(
        json.loads(read(path)) == {"plan_sha256": digest, "nonce": plan["nonce"]},
        "Exact selected close request",
    )
    return True


def helper_selection_seed(identity, checker):
    keys = "binding worker listener service_directory private_config interpreter source_root qualification execution runtime registry expected_memtotal".split()
    return {"schema_id": checker.SCHEMA, **{key: identity[key] for key in keys}}


def controller_main(plan, serve, refs, blob, digest, owner):
    selected = selection(plan, owner)
    observer = load(plan["components"]["observer"], "_q95_owner_observer")
    need(
        owner.boot() == plan["boot_id"] and time.monotonic() < plan["expires_monotonic"],
        "Same boot and fresh launch",
    )
    host(plan, observer, initial=True)
    need(
        owner.status(selected).get("LoadState") == "not-found"
        and not (owner.CGROUP_ROOT / selected.cgroup.lstrip("/")).exists(),
        "Fresh exact unit/cgroup absent",
    )
    out = ordinary(plan["output"])
    need(
        not out.exists() and not SERVICE.exists(),
        "Never reuse owner or original service output",
    )
    out.mkdir(mode=0o700)
    with (out / "sealed-plan.json").open("xb") as stream:
        stream.write(blob)
    small = [relative_ref(r) for r in plan["components"].values()]
    boundary = owner.UnitBoundary(selected, out, small)
    snapshots = check_population(refs)
    props = {
        **selected.properties(),
        "RuntimeMaxSec": "10800",
        "TimeoutStopSec": "20",
        "WorkingDirectory": str(ROOT),
        "LimitCORE": "0",
        "StandardOutput": "append:" + str(out / "actor.log"),
        "StandardError": "append:" + str(out / "actor.log"),
    }
    del props["RuntimeMaxUSec"], props["TimeoutStopUSec"]
    argv = [
        "systemd-run",
        "--user",
        "--unit=" + selected.unit,
        *["--property=" + k + "=" + v for k, v in props.items()],
        "--setenv=JEV_D3_BOUNDARY_NONCE=" + selected.nonce,
        serve["argv"][0],
        "-B",
        "-s",
        str(HERE / "launch_q95.py"),
        "--run",
        "--actor",
        "--plan",
        str(out / "sealed-plan.json"),
        "--plan-sha256",
        digest,
    ]
    owner.save(
        out,
        "launch-intent.json",
        {"argv": argv, "cap_bytes": 40 * GIB, "swap_bytes": 0, "runtime_seconds": 10800},
    )
    primary, closure, final_host, population_ok, ready = None, None, None, False, None
    try:
        owner.command(*argv, timeout=8)
        until = time.monotonic() + 60
        while not (out / "actor-intent.json").exists():
            need(time.monotonic() < until, "Actual owned actor timed readiness")
            host(plan, observer)
            time.sleep(0.1)
        rb = boundary.verify()
        intent = json.loads(read(out / "actor-intent.json"))
        need(
            intent.get("plan_sha256") == digest
            and all(
                intent["actor"][k] == rb["actor"][k]
                for k in ("pid", "start_ticks", "pgid", "sid", "uid", "cgroup")
            ),
            "Exact external actor intent",
        )
        owner.save(
            out,
            "admission.json",
            {
                "plan_sha256": digest,
                "nonce": selected.nonce,
                "invocation_id": rb["unit"]["InvocationID"],
                "actor": rb["actor"],
            },
        )
        checker = load(plan["components"]["checker"], "_q95_owner_identity_checker")
        startup = time.monotonic() + 1800
        shutdown_at = None
        while True:
            row = boundary.owned_status()
            need(
                row.get("ActiveState") == "active"
                and row.get("SubState") == "running"
                and row.get("MainPID") == str(rb["actor"]["pid"]),
                "Selected main actor became terminal",
            )
            host(plan, observer)
            check_population(refs, expected_stamps=snapshots)
            tensor_identity(plan)
            if (out / "actor-result.json").exists():
                result = json.loads(read(out / "actor-result.json"))
                need(
                    result.get("plan_sha256") == digest
                    and result.get("primary_error") is None
                    and result.get("controller_exit_confirmed") is True
                    and result.get("controller_returncode") == 0
                    and shutdown_at is not None,
                    "Original controller completed only after requested shutdown",
                )
                break
            if ready is None and (SERVICE / "endpoint.json").exists():
                ready = helper_identity(
                    {**plan, "_digest": digest}, serve, owner, checker, boundary
                )
                owner.save(out, "helper-identity.json", ready)
                owner.save(out, "helper-selection-seed.json", helper_selection_seed(ready, checker))
            if ready is not None and shutdown_at is None:
                fresh = checker.process(ready["binding"]["pid"])
                need(
                    all(fresh[k] == ready["worker"][k] for k in ready["worker"]),
                    "Actual helper identity changed",
                )
                need(
                    checker.listener(fresh["pid"], ready["listener"]["port"]) == ready["listener"],
                    "Actual helper listener changed",
                )
            need(
                ready is not None or time.monotonic() < startup, "Original helper startup deadline"
            )
            requested = (
                close_requested(plan, digest)
                or time.monotonic() >= plan["created_monotonic"] + 10680
            )
            if requested and shutdown_at is None:
                shutdown_at = time.monotonic()
                owner.save(
                    out, "shutdown-intent.json", {"plan_sha256": digest, "nonce": selected.nonce}
                )
            need(
                shutdown_at is None or time.monotonic() - shutdown_at < 60,
                "Original cooperative shutdown deadline",
            )
            time.sleep(2)
    except BaseException as error:
        primary = error_record(error)
    finally:
        try:
            closure = boundary.close(primary_error=primary)
        except BaseException as error:
            primary = primary or error_record(error)
            closure = {"closure_error": error_record(error)}
        try:
            check_population(refs)
            tensor_identity(plan)
            population_ok = True
        except BaseException as error:
            primary = primary or error_record(error)
        try:
            final_host = host(plan, observer)
        except BaseException as error:
            primary = primary or error_record(error)
        success = (
            primary is None
            and population_ok
            and final_host is not None
            and closure.get("whole_unit_exit_confirmed") is True
            and closure.get("observed_cgroup_cohort_exit_confirmed") is True
            and ready is not None
        )
        owner.save(
            out,
            "owner-final.json",
            {
                "profile": PROFILE,
                "plan_sha256": digest,
                "primary_error": primary,
                "closure": closure,
                "full_population_unchanged": population_ok,
                "final_host": final_host,
                "ready_identity_was_published": ready is not None,
                "completed": success,
                "model_qualified": False,
                "whole_tree_exit_confirmed": False,
                "continuation_authorized": False,
            },
        )
    return 0 if success else 1


def prepare(output):
    import secrets

    components = {
        key: pin(ROOT / "artifacts/jev-runtime" / label) for key, (label, _sha) in FIXED.items()
    }
    components.update(owner=pin(HERE / "q95_owner.py"), entry=pin(HERE / "launch_q95.py"))
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    total = (
        int(
            next(
                line.split()[1]
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemTotal:")
            )
        )
        * 1024
    )
    tensor = json.loads(read(components["tensor"]["path"]))
    now = time.monotonic()
    unit = UNIT
    uid = os.getuid()
    plan = {
        "profile": PROFILE,
        "unit": unit,
        "nonce": "jev-d3-boundary-" + secrets.token_hex(16),
        "boot_id": boot,
        "mem_total_bytes": total,
        "created_monotonic": now,
        "expires_monotonic": now + 3600,
        "cgroup": f"/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/{unit}",
        "output": str(HERE / output),
        "components": components,
        "tensor_stat": stamp(tensor["target"]),
    }
    return plan
