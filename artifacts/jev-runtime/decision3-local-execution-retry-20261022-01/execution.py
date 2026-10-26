"""Thin selected local D3 worker composition; importing this module starts nothing."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import secrets
import select
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).absolute().parents[3]
HERE = Path(__file__).absolute().parent
PROFILE = "decision3-gx10-native-v7-local-execution-retry-v2"
FIXTURE = "decision3-local-execution-cpu-fixture-v1"
MODELS = ("d3", "d3-flash", "d3-mini", "d3-nano", "d3-lite", "d3-edge")
GROUPS = {"known4": ("known-four", 64, 4), "public6": ("public-six", 96, 6)}
BOOTSTRAP_SHA = "a39e811764f83eff793f3f5ade7fb9defd7e21fdc7d86207b5e12ce7c22b8581"
DESCRIPTOR_SHA = "034cad18a33e9cb988d5b8b697df249651cf5f1d18be711174184948ed83aa03"
TRANSPORT_SHA = "0490fe27872265662d7008a97876d42e81f35585f37f177928e802649445b5d8"
BOUNDARY_SHA = "7a50c47d6d7f696973c666cb7a40ce47dd0a7624667a18422c8c60c0db72e4e5"
OBSERVER_SHA = "2bd125f17db7739d19979565fcf8013c897f620caaa948d8ffe4de1559bf9fd6"
V7_SHA = "31450f9002ad216c32b77751997eb27748c491d7bc757c4410294d3704fdfb9e"
H15_SHA = "2ed7cda3054da62e7440bf57e4bfae15e26ee6d22de8b80aede6c8f53973e3cb"
DRIVER_SHA = "1ad0a1749e1ab69235cf6356939ef55edf904ba8942794f6241ef9580c676582"
HELPER_SHA = "85f2eef27f4455710635514feebe90611a52b22c6e6781da09899dfaa27482ba"
CAP = 80 * 1024**3
SHA = re.compile(r"[0-9a-f]{64}\Z")
CASES = {
    "known4": (
        "public-arxiv-title-filter-results-v1",
        "public-arxiv-original-paper-open-v1",
        "public-python-counter-most-common-v1",
        "public-nasa-mars-facts-moons-v1",
    ),
    "public6": tuple("heldout-public-" + str(i).zfill(3) for i in range(6)),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def encode(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def parse(blob):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "Duplicate selected JSON field")
            result[key] = value
        return result

    return json.loads(
        blob,
        object_pairs_hook=pairs,
        parse_constant=lambda _value: require(False, "Nonfinite selected JSON"),
    )


def ordinary(path):
    path = Path(path).absolute()
    require(
        ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "Ordinary selected path",
    )
    return path


def source_ref(path):
    path = ordinary(path)
    blob = path.read_bytes()
    require(len(blob) <= 2 * 1024**2, "Small explicit source only")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "size_bytes": len(blob),
    }


def read_ref(ref):
    require(
        type(ref) is dict and set(ref) == {"path", "sha256", "size_bytes"},
        "Exact selected reference",
    )
    require(
        type(ref["path"]) is str and not Path(ref["path"]).is_absolute(),
        "Repository relative reference",
    )
    path = ordinary(ROOT / ref["path"])
    require(
        path.is_relative_to(ROOT)
        and type(ref["sha256"]) is str
        and SHA.fullmatch(ref["sha256"])
        and type(ref["size_bytes"]) is int
        and 0 < ref["size_bytes"] <= 2 * 1024**2,
        "Bounded selected source identity",
    )
    blob = path.read_bytes()
    require(
        len(blob) == ref["size_bytes"] and hashlib.sha256(blob).hexdigest() == ref["sha256"],
        "Selected source bytes changed",
    )
    return path, blob


def load_module(ref, name):
    path, blob = read_ref(ref)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(blob, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


def save(path, value):
    path = ordinary(path)
    blob = encode(value) + b"\n"
    require(len(blob) <= 16 * 1024**2, "Bounded integration evidence")
    temporary = path.with_name(path.name + ".pending-" + str(os.getpid()))
    with temporary.open("xb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def selected_plan(path, expected_sha, *, fixture=False, check_time=True):
    require(
        type(expected_sha) is str and SHA.fullmatch(expected_sha),
        "Independently selected plan SHA required",
    )
    path = ordinary(path)
    blob = path.read_bytes()
    require(
        len(blob) <= 2 * 1024**2 and hashlib.sha256(blob).hexdigest() == expected_sha,
        "Execution plan SHA differs",
    )
    plan = parse(blob)
    require(
        type(plan) is dict
        and set(plan)
        == {
            "schema_id",
            "profile",
            "model",
            "boot_id",
            "mem_total_bytes",
            "gpu_uuid",
            "expires_monotonic",
            "output",
            "interpreter",
            "components",
            "sources",
            "arms",
            "browser",
        },
        "Exact selected execution plan",
    )
    require(
        plan["schema_id"] == "decision3-local-execution-plan-v1"
        and plan["profile"] == (FIXTURE if fixture else PROFILE)
        and plan["model"] in MODELS,
        "Explicit selected execution/model profile",
    )
    require(
        type(plan["expires_monotonic"]) in (float, int)
        and math.isfinite(plan["expires_monotonic"]),
        "Finite launch expiry",
    )
    require(
        not check_time or time.monotonic() < plan["expires_monotonic"],
        "Selected launch has expired",
    )
    require(
        type(plan["mem_total_bytes"]) is int and plan["mem_total_bytes"] > 0,
        "Positive selected host total",
    )
    require(type(plan["gpu_uuid"]) is str and bool(plan["gpu_uuid"]), "Selected GPU identity")
    require(set(plan["arms"]) == set(GROUPS), "Both original browser groups required")
    require(
        set(plan["components"])
        == {
            "execution",
            "entry",
            "bootstrap",
            "bootstrap_selection",
            "transport",
            "boundary",
            "observer",
            "helper_checker",
            "v7",
            "h15",
            "driver_runtime",
        },
        "Exact integration components",
    )
    require(
        type(plan["sources"]) is list and 1 <= len(plan["sources"]) <= 64,
        "Explicit bounded source union",
    )
    refs = {row["path"]: row for row in plan["sources"]}
    require(len(refs) == len(plan["sources"]), "No duplicate source authority")
    for ref in [*plan["sources"], *plan["components"].values()]:
        read_ref(ref)
        require(refs.get(ref["path"]) == ref, "Every component in sealed source authority")
    for key, digest in [
        ("bootstrap", BOOTSTRAP_SHA),
        ("bootstrap_selection", DESCRIPTOR_SHA),
        ("transport", TRANSPORT_SHA),
        ("boundary", BOUNDARY_SHA),
        ("observer", OBSERVER_SHA),
        ("v7", V7_SHA),
        ("h15", H15_SHA),
        ("driver_runtime", DRIVER_SHA),
    ]:
        require(
            plan["components"][key]["sha256"] == digest, "Exact frozen component required: " + key
        )
    require(
        read_ref(plan["components"]["execution"])[0] == Path(__file__).absolute(),
        "Exact integration execution source",
    )
    require(
        read_ref(plan["components"]["entry"])[0] == HERE / "launch_execution.py",
        "Exact integration entry source",
    )
    b = load_module(plan["components"]["bootstrap"], "_d3_execution_bootstrap")
    descriptor = read_ref(plan["components"]["bootstrap_selection"])[1]
    inputs = b.selected_inputs(descriptor, DESCRIPTOR_SHA)
    required = {str(Path(row["path"]).relative_to(ROOT)) for row in inputs["inventory"]["rows"]}
    require(
        required <= set(refs), "Complete selected nineteen-source population in outer authority"
    )
    worker = b.load_worker()
    require(
        inputs["inventory"] == b.source_inventory(worker),
        "Selected bootstrap source composition drift",
    )
    identities = worker.authenticate_sources(inputs["inventory"]["selected_projection"])
    require(tuple(identities) == MODELS, "All six checkpoint identities retained")
    for group, (workload, cap, _count) in GROUPS.items():
        arm = plan["arms"][group]
        require(
            type(arm) is dict
            and set(arm) == {"nonce", "unit", "helper_binding", "helper_selection", "model_output"},
            "Exact selected arm",
        )
        require(
            type(arm["nonce"]) is str
            and re.fullmatch(r"jev-d3-boundary-[0-9a-f]{32}", arm["nonce"]),
            "Fresh selected arm nonce",
        )
        require(
            type(arm["unit"]) is str and re.fullmatch(r"jev-d3-[a-z0-9-]+\.service", arm["unit"]),
            "Selected D3 unit",
        )
        require(arm["helper_binding"]["boot_id"] == plan["boot_id"], "Same selected helper boot")
        read_ref(arm["helper_selection"])
        require(
            refs.get(arm["helper_selection"]["path"]) == arm["helper_selection"],
            "Selected helper authority",
        )
        require(
            b.validate_helper(arm["helper_binding"], {"expected_boot_id": plan["boot_id"]})
            == arm["helper_binding"],
            "Exact Q95 helper binding",
        )
        require(
            b.load_worker().startup_configuration("publisher-text48-v1", workload)[
                "max_browser_envelopes"
            ]
            == cap,
            "Original envelope cap",
        )
        model_output = ordinary(arm["model_output"])
        require(
            model_output.is_relative_to(HERE if fixture else b.REPLAY_ROOT),
            "Original model output boundary",
        )
    require(
        len({a["model_output"] for a in plan["arms"].values()}) == 2, "Fresh separate model outputs"
    )
    require(
        len({a["nonce"] for a in plan["arms"].values()})
        == len({a["unit"] for a in plan["arms"].values()})
        == 2,
        "Independent fresh group units/nonces",
    )
    require(ordinary(plan["output"]).is_relative_to(HERE), "Owned integration output root")
    require(Path(plan["interpreter"]).is_absolute(), "Selected absolute interpreter")
    if not fixture:
        require(
            plan["components"]["helper_checker"]["sha256"] == HELPER_SHA
            and read_ref(plan["components"]["helper_checker"])[0]
            == ROOT
            / "artifacts/jev-runtime/decision3-local-helper-budget-20261022-01/live_helper.py",
            "Exact independently reviewed live helper checker",
        )
        require(
            plan["interpreter"] == parse(inputs["runtime_blob"])["interpreter"],
            "Selected model interpreter",
        )
        browser = plan["browser"]
        require(
            type(browser) is dict
            and set(browser) == {"interpreter", "upstream_path", "upstream_commit", "chromium"},
            "Exact browser selection",
        )
        require(browser["interpreter"] == plan["interpreter"], "Selected browser interpreter")
        require(
            type(browser["upstream_commit"]) is str
            and re.fullmatch("[0-9a-f]{40}", browser["upstream_commit"]),
            "Selected upstream commit",
        )
        ordinary(browser["upstream_path"])
        verify_binary(browser["chromium"])
    return plan, b, inputs, identities


def prepare_plan(
    *,
    model,
    helper_checker,
    helper_selections,
    helper_bindings,
    browser,
    output,
    model_outputs,
    boot_id,
    mem_total_bytes,
    gpu_uuid,
    expires_monotonic,
    fixture=False,
    units=None,
    nonces=None,
):
    """Preparation only. Caller reviews then independently selects returned bytes/SHA."""
    base = ROOT / "artifacts/jev-runtime"
    component_paths = {
        "execution": HERE / "execution.py",
        "entry": HERE / "launch_execution.py",
        "bootstrap": base / "decision3-current-source-bootstrap-20261022-01/current_bootstrap.py",
        "bootstrap_selection": base
        / "decision3-current-source-bootstrap-20261022-01/bootstrap-selection.json",
        "transport": base / "decision3-local-transport-20261022-01/local_transport.py",
        "boundary": base / "decision3-local-owner-terminal-20261022-01/owner_boundary.py",
        "observer": base / "comfy-observer-zombie-exit-20261022-01/comfy_observer.py",
        "helper_checker": ordinary(helper_checker),
        "v7": ROOT / "scripts/jev_browser_profile_v7_benchmark.py",
        "h15": base / "public-native-fit-memory-uma-probe-20261020-01/native_uma_peak_probe_01.py",
        "driver_runtime": base
        / "public-native-fit-memory-peak-probe-20261018-01/native-toolchain-runtime-pins-02.json",
    }
    components = {k: source_ref(v) for k, v in component_paths.items()}
    b = load_module(components["bootstrap"], "_d3_execution_prepare_bootstrap")
    selected = b.selected_inputs(read_ref(components["bootstrap_selection"])[1], DESCRIPTOR_SHA)
    descriptor = parse(read_ref(components["bootstrap_selection"])[1])
    paths = {
        *component_paths.values(),
        *(Path(row["path"]) for row in selected["inventory"]["rows"]),
        *(Path(row["path"]) for row in descriptor["files"].values()),
        *(ordinary(v) for v in helper_selections.values()),
        ROOT / "scripts/jev_observation_profile_v7.py",
        ROOT / "scripts/jev_public_web_cases.py",
        ROOT / "scripts/jev_heldout_public_cases.py",
        ROOT / "scripts/jev_browser_benchmark.py",
        ROOT / "src/s1/jev_observation_profile.py",
        ROOT / "src/s1/jev_observation_events.py",
    }
    if fixture:
        paths.add(HERE / "fake_components.py")
    refs = sorted((source_ref(path) for path in paths), key=lambda ref: ref["path"])
    token = secrets.token_hex(8)
    plan = {
        "schema_id": "decision3-local-execution-plan-v1",
        "profile": FIXTURE if fixture else PROFILE,
        "model": model,
        "boot_id": boot_id,
        "mem_total_bytes": mem_total_bytes,
        "gpu_uuid": gpu_uuid,
        "expires_monotonic": expires_monotonic,
        "output": str(ordinary(output)),
        "interpreter": str(Path(sys.executable).absolute())
        if fixture
        else parse(selected["runtime_blob"])["interpreter"],
        "components": components,
        "sources": refs,
        "browser": browser,
        "arms": {},
    }
    for group in GROUPS:
        plan["arms"][group] = {
            "nonce": nonces[group]
            if nonces is not None
            else "jev-d3-boundary-" + secrets.token_hex(32 // 2),
            "unit": units[group]
            if units is not None
            else "jev-d3-" + model + "-" + group + "-" + token + ".service",
            "helper_selection": source_ref(helper_selections[group]),
            "helper_binding": copy.deepcopy(helper_bindings[group]),
            "model_output": str(ordinary(model_outputs[group])),
        }
    return plan


def verify_binary(ref):
    require(
        type(ref) is dict and set(ref) == {"path", "sha256", "size_bytes"},
        "Selected binary identity",
    )
    path = ordinary(ref["path"])
    require(
        path.is_absolute()
        and type(ref["size_bytes"]) is int
        and ref["size_bytes"] > 0
        and type(ref["sha256"]) is str
        and SHA.fullmatch(ref["sha256"]),
        "Typed binary selection",
    )
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
            size += len(chunk)
    require(
        size == ref["size_bytes"] and digest.hexdigest() == ref["sha256"], "Selected binary changed"
    )


def transport_for(plan):
    return load_module(plan["components"]["transport"], "_d3_execution_transport")


def unit_selection(plan, group, owner):
    """Only the explicitly approved D3 80-GiB cap differs from frozen owner policy."""

    class D3Selection(owner.Selection):
        def __post_init__(self):
            require(self.memory_max_bytes == CAP, "Only approved explicit D3 80GiB cap")
            owner.Selection(
                self.unit,
                self.nonce,
                self.boot_id,
                40 * 1024**3,
                self.runtime_seconds,
                self.stop_seconds,
            )

    arm = plan["arms"][group]
    return D3Selection(arm["unit"], arm["nonce"], plan["boot_id"], CAP, 7200, 10)


def actual_parent(owner, transport):
    row = transport.proc_record(os.getppid())
    require(
        row["pid"] == owner["pid"] and row["start_ticks"] == owner["start_ticks"],
        "Actual selected parent identity",
    )


class HelperRPC:
    """Selected checker in parent; no credentials or socket access in model child."""

    def __init__(self, transport, selection, binding, control, source):
        self.t, self.selection, self.binding, self.control, self.source = (
            transport,
            selection,
            copy.deepcopy(binding),
            control,
            source,
        )
        self.sequence, self.deadline = 0, None
        self.identity = transport.binding(
            selection, os.getpid(), transport.proc_record(os.getpid())["start_ticks"]
        )

    def __call__(self, binding, phase):
        require(
            binding == self.binding and type(phase) is str and 0 < len(phase) <= 128,
            "Exact helper request binding",
        )
        require(self.deadline is None or time.monotonic() < self.deadline, "Helper request expired")
        self.sequence += 1
        require(self.sequence <= 330, "Bounded helper admission calls")
        request = {
            "event": "helper_check",
            "binding": binding,
            "phase": phase,
            "sequence": self.sequence,
            **self.identity,
        }
        digest = hashlib.sha256(encode(request)).hexdigest()
        self.control.write(encode({**request, "check_sha256": digest}) + b"\n")
        self.control.flush()
        # The pipe is unbuffered; a stalled parent cannot extend a request's budget.
        deadline = self.deadline if self.deadline is not None else time.monotonic() + 120
        line = bytearray()
        while not line.endswith(b"\n"):
            require(time.monotonic() < deadline, "Helper control deadline")
            readable, _, _ = select.select(
                [self.source], [], [], max(0, deadline - time.monotonic())
            )
            require(readable, "Helper control response deadline")
            chunk = os.read(self.source.fileno(), 1)
            require(chunk and len(line) <= 1024**2, "Bounded complete helper control frame")
            line.extend(chunk)
        line = bytes(line)
        require(line.endswith(b"\n") and len(line) <= 1024**2, "Bounded helper admission response")
        reply = parse(line)
        require(
            type(reply) is dict
            and set(reply) == {"event", "sequence", "check_sha256", "identity", "evidence", "error"}
            and reply["event"] == "helper_checked"
            and reply["sequence"] == self.sequence
            and reply["check_sha256"] == digest
            and reply["identity"] == self.identity,
            "Authenticated parent helper response",
        )
        require(reply["error"] is None, "Fresh parent helper/owner observation refused")
        require(
            self.deadline is None or time.monotonic() < self.deadline,
            "Fresh helper response arrived late",
        )
        return reply["evidence"]


def checked_session_class(transport):
    class CheckedSession(transport.OwnedSession):
        def __init__(self, *args, helper_binding, checker, **kwargs):
            super().__init__(*args, **kwargs)
            self.helper_binding, self.checker, self.helper_sequence = (
                copy.deepcopy(helper_binding),
                checker,
                0,
            )

        def _read(self, deadline):
            while True:
                frame = super()._read(deadline)
                if frame.get("event") != "helper_check":
                    return frame
                self.helper_sequence += 1
                require(
                    frame.get("binding") == self.helper_binding
                    and frame.get("sequence") == self.helper_sequence
                    and self.helper_sequence <= 330
                    and type(frame.get("phase")) is str
                    and 0 < len(frame["phase"]) <= 128,
                    "Selected helper control request",
                )
                digest = frame.get("check_sha256")
                require(
                    digest
                    == hashlib.sha256(
                        encode({k: v for k, v in frame.items() if k != "check_sha256"})
                    ).hexdigest(),
                    "Helper control request SHA",
                )
                evidence, error = None, None
                try:
                    evidence = self.checker(copy.deepcopy(self.helper_binding), frame["phase"])
                except Exception as refusal:
                    error = {
                        "type": type(refusal).__name__,
                        "message": "Selected helper or host observation refused",
                    }
                self._send(
                    {
                        "event": "helper_checked",
                        "sequence": self.helper_sequence,
                        "check_sha256": digest,
                        "identity": self.identity,
                        "evidence": evidence,
                        "error": error,
                    },
                    deadline,
                )

    return CheckedSession


def driver_maps(helper):
    # H15's selected ordinary-path reader intentionally rejects /proc/self.
    return helper.read(Path("/proc") / str(os.getpid()) / "maps", 8 * 1024**2).decode()


class RawIdentity:
    def __init__(self, plan):
        self.plan = plan
        self.h = load_module(plan["components"]["h15"], "_d3_execution_driver_mechanisms")
        self.runtime = parse(read_ref(plan["components"]["driver_runtime"])[1])

    def check(self, torch):
        import ctypes

        torch.cuda.synchronize(0)
        require(
            torch.cuda.device_count() == 1 and torch.cuda.current_device() == 0,
            "One selected CUDA device",
        )
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        metadata = self.h.uma_driver_metadata(self.runtime)
        library = ctypes.CDLL(self.h.UMA_LIBCUDA, mode=os.RTLD_NOLOAD | os.RTLD_NOW | os.RTLD_LOCAL)
        symbols = {name: getattr(library, name) for name in self.h.UMA_SYMBOLS}
        addresses = {name: ctypes.cast(fn, ctypes.c_void_p).value for name, fn in symbols.items()}
        self.h.uma_symbol_mappings(driver_maps(self.h), addresses, metadata)
        raw = self.h.uma_driver_read(symbols, ctypes)
        host = transport_for(self.plan).host_observation()

        def normalized(value):
            return str(value).lower().removeprefix("gpu-").replace("-", "")

        require(
            raw["context_device"] == 0
            and raw["integrated"] == 1
            and raw["name"] == props.name == "NVIDIA GB10"
            and raw["total_bytes"]
            == props.total_memory
            == total
            == host["MemTotal"]
            == self.plan["mem_total_bytes"]
            and host["boot_id"] == self.plan["boot_id"]
            and normalized(raw["uuid"])
            == normalized(props.uuid)
            == normalized(self.plan["gpu_uuid"])
            and 0 < free <= total,
            "Fresh raw Driver/Torch/host/UUID identity differs",
        )
        require(self.h.uma_driver_metadata(self.runtime) == metadata, "Driver metadata changed")
        self.h.uma_symbol_mappings(driver_maps(self.h), addresses, metadata)
        return {
            "driver": raw,
            "torch_uuid": str(props.uuid),
            "total_bytes": total,
            "boot_id": host["boot_id"],
            "library_metadata": metadata,
            "scope": "selected driver metadata/maps and actual identity; no universal DSO claim",
        }


class ModelService:
    def __init__(self, plan, group, bootstrap, rpc, *, fixture_factory=None):
        self.plan, self.group, self.b, self.rpc = plan, group, bootstrap, rpc
        self.fixture_factory, self.components, self.requests = fixture_factory, None, 0
        self.failure, self.raw_identity = None, None
        self.output = ordinary(plan["arms"][group]["model_output"])
        self.config = {
            "profile": bootstrap.PROFILE,
            "model": plan["model"],
            "workload": GROUPS[group][0],
            "nonce": plan["arms"][group]["nonce"],
            "expected_boot_id": plan["boot_id"],
            "expected_memtotal": plan["mem_total_bytes"],
            "expected_gpu_uuid": plan["gpu_uuid"],
            "output_dir": str(self.output),
        }

    def initialize(self):
        require(
            time.monotonic() < self.plan["expires_monotonic"], "Launch expired before constructor"
        )
        binding = self.plan["arms"][self.group]["helper_binding"]
        self.b.live_helper_check(self.rpc, binding, "before-component-constructor")
        if self.fixture_factory is not None:
            require(
                self.plan["profile"] == FIXTURE,
                "Fake model only under explicit CPU fixture profile",
            )
            self.components = self.fixture_factory(self)
        else:
            require(self.plan["profile"] == PROFILE, "Actual worker profile")
            guard = RawIdentity(self.plan)

            def dependencies(selection):
                deps = self.b.dependencies(selection)
                self.raw_identity = guard.check(deps.torch)
                return deps

            descriptor = read_ref(self.plan["components"]["bootstrap_selection"])[1]
            self.components = self.b.bootstrap_selected(
                self.config,
                selection_blob=descriptor,
                selection_sha=DESCRIPTOR_SHA,
                helper=binding,
                helper_check=self.rpc,
                load_dependencies=dependencies,
            )
            self.raw_identity = guard.check(self.components.torch)
        return {
            "profile": self.plan["profile"],
            "component_complete": True,
            "model_qualified": False,
            "bootstrap_descriptor_sha256": DESCRIPTOR_SHA,
            "raw_identity": self.raw_identity,
            "startup": self.components.startup,
            "bootstrap": self.components.state,
        }

    def process_request(self, line):
        require(self.components is not None, "Initialized selected worker")
        t = transport_for(self.plan)
        envelope = t.parse_envelope(line)
        started = time.monotonic()
        self.rpc.deadline = started + envelope["remaining_seconds"]
        self.requests += 1
        require(self.requests <= GROUPS[self.group][1], "Original per-group envelope cap")
        candidate = None
        try:
            self.b.live_helper_check(
                self.rpc, self.rpc.binding, "before-request-" + str(self.requests)
            )
            c = self.components
            candidate = c.worker.process_request(
                line,
                adapter=c.adapter,
                wire=c.wire,
                torch=c.torch,
                gates=c.gates,
                output=c.output,
                model=self.plan["model"],
                nonce=self.config["nonce"],
                gpu_uuid=self.plan["gpu_uuid"],
                startup=c.startup,
                started=started,
            )
            self.b.live_helper_check(
                self.rpc, self.rpc.binding, "after-request-" + str(self.requests)
            )
            require(
                time.monotonic() < self.rpc.deadline, "Final admission exceeded request deadline"
            )
            if not candidate["success"]:
                self.failure = {"type": "NativeRequestRefused"}
            return candidate
        except BaseException as error:
            self.failure = {"type": type(error).__name__, "message": str(error)[:1024]}
            save(
                self.output / ("service-refusal-" + str(self.requests) + ".json"),
                {
                    "error": self.failure,
                    "native_candidate": candidate,
                    "request_sha256": hashlib.sha256(line).hexdigest(),
                },
            )
            raise
        finally:
            self.rpc.deadline = None

    def finalize(self):
        report = {
            "success": False,
            "profile": self.plan["profile"],
            "requests": self.requests,
            "model_qualified": False,
            "errors": [],
            "source_cache_runtime_authenticated": False,
        }
        c = self.components
        if c is None:
            report["errors"].append({"stage": "bootstrap", "type": "ComponentUnavailable"})
        else:
            try:
                self.b.live_helper_check(self.rpc, self.rpc.binding, "finalize-before-close")
                if self.fixture_factory is None:
                    worker = c.worker
                    inventory = self.b.selected_inputs(
                        read_ref(self.plan["components"]["bootstrap_selection"])[1], DESCRIPTOR_SHA
                    )["inventory"]
                    identities = worker.authenticate_sources(inventory["selected_projection"])
                    identity = identities[self.plan["model"]]
                    require(
                        self.b.geometry(c.adapter.backend, identity, c.torch, self.plan["model"])
                        == c.state["geometry"],
                        "Final native geometry/provenance changed",
                    )
                    initial = parse((c.output / "initial-cache.json").read_bytes())
                    c.torch.cuda.synchronize(0)
                    self.raw_identity = RawIdentity(self.plan).check(c.torch)
                    c.close()
                    final = worker.CORE.authenticate_cache(
                        self.plan["model"], identity, c.output, c.gates, "final"
                    )
                    require(final == initial, "Final full cache authentication changed")
                    worker.authenticate_sources(inventory["selected_projection"])
                    runtime = self.b.selected_inputs(
                        read_ref(self.plan["components"]["bootstrap_selection"])[1], DESCRIPTOR_SHA
                    )
                    report["runtime_final"] = self.b.RuntimeSelection(
                        runtime["runtime_blob"], runtime["runtime_sha"], worker
                    ).verify(loaded=True)
                    require(
                        c.state["network_attempts_blocked"] == [], "Offline model attempted network"
                    )
                    report["source_cache_runtime_authenticated"] = True
                    report["raw_identity"] = RawIdentity(self.plan).check(c.torch)
                else:
                    c.close()
                    report["cpu_fixture_only"] = True
                report["success"] = self.failure is None
            except BaseException as error:
                report["errors"].append(
                    {
                        "stage": "finalize",
                        "type": type(error).__name__,
                        "message": str(error)[:1024],
                    }
                )
                if not c.closed:
                    try:
                        c.close()
                    except BaseException as secondary:
                        report["errors"].append(
                            {"stage": "close", "type": type(secondary).__name__}
                        )
        # A failed pre-constructor admission may leave the model directory empty.
        self.output.mkdir(parents=True, exist_ok=True, mode=0o700)
        save(self.output / "service-final.json", report)
        return report


def selection_for(plan, group, transport):
    return transport.Selection(
        plan["arms"][group]["nonce"],
        plan["model"],
        plan["boot_id"],
        plan["mem_total_bytes"],
        plan["gpu_uuid"],
    )


def group_output(plan, group):
    require(group in GROUPS, "Selected original group")
    return ordinary(plan["output"]) / group


def host_idle(plan, observer):
    row = observer.observe(plan["boot_id"])
    require(
        row.get("boot_id") == plan["boot_id"]
        and row.get("mem_total_bytes") == plan["mem_total_bytes"]
        and type(row.get("mem_available_bytes")) is int
        and 0 < row["mem_available_bytes"] <= plan["mem_total_bytes"]
        and row.get("comfy_state") in {"http_idle", "absent_confirmed"}
        and type(row.get("comfy_running")) is int
        and row["comfy_running"] == 0
        and type(row.get("comfy_pending")) is int
        and row["comfy_pending"] == 0,
        "Fresh known idle/absent Comfy observation required",
    )
    reserve = max(24 * 1024**3, (row["mem_total_bytes"] + 4) // 5)
    require(
        row["mem_available_bytes"] >= reserve + 2 * 1024**3,
        "Continuous owner host reserve margin crossed",
    )
    return row


def error_record(error):
    # Exception messages from external helper libraries can contain credentials.
    row = {
        "type": type(error).__name__,
        "message": "Selected execution refused; see bounded stage evidence",
    }
    evidence = getattr(error, "evidence", None)
    if evidence is not None:
        blob = encode(evidence)
        row["observation_evidence_sha256"] = hashlib.sha256(blob).hexdigest()
        row["observation_evidence_bytes"] = len(blob)
    return row


def owner_admission(plan, group, digest, owner, *, worker=False):
    """Read actual external controller proof; no caller boolean creates admission."""
    out = group_output(plan, group)
    admission_blob = (out / "admission.json").read_bytes()
    require(len(admission_blob) <= 65536, "Bounded external owner admission")
    admission = parse(admission_blob)
    selection = unit_selection(plan, group, owner)
    readback_blob = (out / "unit-readback.json").read_bytes()
    require(len(readback_blob) <= 2 * 1024**2, "Bounded actual unit readback")
    readback = parse(readback_blob)
    require(
        set(admission) == {"plan_sha256", "group", "nonce", "invocation_id", "readback_sha256"}
        and admission["plan_sha256"] == digest
        and admission["group"] == group
        and admission["nonce"] == selection.nonce
        and admission["readback_sha256"] == hashlib.sha256(readback_blob).hexdigest(),
        "Exact external plan/nonce/readback authority",
    )
    row = owner.status(selection)
    require(
        owner.boot() == plan["boot_id"]
        and row.get("InvocationID") == admission["invocation_id"]
        and re.fullmatch("[0-9a-f]{32}", admission["invocation_id"])
        and all(row.get(k) == v for k, v in selection.properties().items())
        and row.get("MainPID") == str(readback["actor"]["pid"]),
        "Fresh actual selected unit properties/Invocation",
    )
    actor = owner.process(readback["actor"]["pid"])
    fields = ("pid", "start_ticks", "pgid", "sid", "uid", "cgroup")
    require(
        all(actor[k] == readback["actor"][k] for k in fields)
        and actor["pid"] == actor["pgid"] == actor["sid"]
        and actor["cgroup"] == selection.cgroup
        and actor["uid"] == os.getuid(),
        "Stable admitted owner actor",
    )
    current = owner.process(os.getpid())
    require(
        current["cgroup"] == selection.cgroup and current["uid"] == os.getuid(),
        "Actual selected cgroup membership",
    )
    if worker:
        require(os.getppid() == actor["pid"], "Model child actual admitted actor parent")
    else:
        require(all(current[k] == actor[k] for k in fields), "Actual admitted main actor")
    cg = owner.CGROUP_ROOT / selection.cgroup.lstrip("/")
    require(
        (cg / "memory.max").read_text().strip() == str(CAP)
        and (cg / "memory.swap.max").read_text().strip() == "0"
        and (cg / "cgroup.type").read_text().strip() == "domain",
        "Fresh actual 80GiB/swap0/domain limits",
    )
    return {"actor": actor, "invocation_id": admission["invocation_id"], "cgroup": selection.cgroup}


def worker_main(plan, group, digest, bootstrap, *, fixture_factory=None):
    transport = transport_for(plan)
    selection = selection_for(plan, group, transport)
    selection.check_host()
    if fixture_factory is None:
        owner = load_module(plan["components"]["boundary"], "_d3_execution_owner_worker")
        owner_admission(plan, group, digest, owner, worker=True)
    else:
        require(plan["profile"] == FIXTURE, "CPU fixture profile only")
        actual_parent(parse(os.environ["JEV_D3_FIXTURE_PARENT"]), transport)
    output = ordinary(plan["arms"][group]["model_output"])
    require(not output.exists(), "Fresh model output before constructor")
    output.mkdir(mode=0o700, parents=True)
    if fixture_factory is None:
        modules = (
            ROOT
            / "artifacts/jev-d3-browser-module-cache"
            / plan["arms"][group]["nonce"]
            / plan["model"]
        )
        require(not modules.exists(), "Fresh owned publisher module cache")
        ordinary(modules).mkdir(mode=0o700, parents=True)
    # RPC must read raw stdin, without a BufferedReader retaining a following frame.
    source = os.fdopen(os.dup(0), "rb", buffering=0)
    control = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    rpc = HelperRPC(transport, selection, plan["arms"][group]["helper_binding"], control, source)
    service = ModelService(plan, group, bootstrap, rpc, fixture_factory=fixture_factory)
    try:
        return transport.serve_worker(
            selection,
            initialize=service.initialize,
            process_request=service.process_request,
            finalize=service.finalize,
            control=control,
            source=source,
        )
    finally:
        control.close()
        source.close()


def browser_command(plan, group, bundle, output, endpoint, text):
    require(
        set(text) == {"endpoint", "model", "api_key"} and text["model"] == "generated-actions-none",
        "Exact helper text connection",
    )
    return [
        plan["browser"]["interpreter"],
        "-B",
        "-s",
        str(bundle / "runner.py"),
        "--upstream-path",
        plan["browser"]["upstream_path"],
        "--endpoint",
        endpoint,
        "--model",
        plan["model"],
        "--output-dir",
        str(output),
        "--text-endpoint",
        text["endpoint"],
        "--text-model",
        text["model"],
        "--api-key-env",
        "S1_POLICY_API_KEY",
        "--text-api-key-env",
        "S1_TEXT_API_KEY",
        "--repeats",
        "1",
        "--max-steps",
        "16",
        "--seed",
        "20261014",
        "--suite-split",
        "development",
        "--chromium-executable",
        plan["browser"]["chromium"]["path"],
        "--observation-profile",
        "composed-visible-text-settled-v7",
        "--observation-profile-helper-sha256",
        "d9f4236d1d46c9472c3d6c1551342639ecb870e6c0dd419246927645410ea868",
        "--public-cases",
        *CASES[group],
    ]


def browser_score(path, group):
    blob = ordinary(path).read_bytes()
    require(len(blob) <= 16 * 1024**2, "Bounded unchanged browser summary")
    summary = parse(blob)
    outcomes = summary["outcomes"]
    require(
        type(outcomes) is list and len(outcomes) == GROUPS[group][2],
        "Complete original group denominator",
    )
    # Original judge writes success only after verification + done + no error.
    require(
        all(type(row.get("success")) is bool for row in outcomes), "Original typed task success"
    )
    require(
        tuple(row["task_id"] for row in outcomes) == CASES[group],
        "Exact complete ordered original cases",
    )
    require(
        all(
            row["success"]
            == bool(
                row.get("verification", {}).get("success")
                and row.get("status") == "done"
                and row.get("error") is None
            )
            for row in outcomes
        ),
        "Original verification/done/no-error judge",
    )
    require(summary.get("observation_profile_failed") is False, "V7 profile integrity required")
    require(
        summary["attempts"] == len(outcomes)
        and summary["successes"] == sum(row["success"] for row in outcomes),
        "Original score arithmetic",
    )
    return {
        "attempts": len(outcomes),
        "successes": summary["successes"],
        "summary_sha256": hashlib.sha256(blob).hexdigest(),
    }


def close_actor_resources(gateway, http, session, primary):
    """Every owned cleanup is attempted, even when an earlier one failed."""
    errors, closure = [], {"success": False}
    operations = []
    if gateway is not None and primary is not None:
        operations.append(("gateway_block", gateway.block))
    if http is not None:
        operations.append(("listener_close", http.close))
    if session is not None:
        operations.append(("model_session_close", lambda: session.close(grace=300, term=5, kill=5)))
    for stage, operation in operations:
        try:
            value = operation()
            if stage == "model_session_close":
                closure = value
        except BaseException as error:
            recorded = {"stage": stage, **error_record(error)}
            errors.append(recorded)
            primary = primary or recorded
    return closure, primary, errors


def actor_main(plan, group, digest, plan_path):
    """One group inside the selected external unit; controller owns scoped stop."""
    require(plan["profile"] == PROFILE, "Actual actor profile only")
    if os.getpid() != os.getsid(0):
        os.setsid()
    out = group_output(plan, group)
    owner = load_module(plan["components"]["boundary"], "_d3_execution_owner_actor")
    observer = load_module(plan["components"]["observer"], "_d3_execution_observer_actor")
    actor = owner.process(os.getpid())
    save(out / "actor-intent.json", {"plan_sha256": digest, "group": group, "actor": actor})
    session = None
    gateway, http, browser, closure, primary, score = None, None, None, None, None, None
    try:
        deadline = time.monotonic() + 30
        while not (out / "admission.json").exists():
            require(time.monotonic() < deadline, "External owner admission timed out")
            time.sleep(0.02)
        owner_admission(plan, group, digest, owner)
        host_idle(plan, observer)
        checker_module = load_module(plan["components"]["helper_checker"], "_d3_execution_helper")
        helper_ref = plan["arms"][group]["helper_selection"]
        checker_out = out / "helper"
        checker_out.mkdir(mode=0o700)
        checker = checker_module.LiveHelperChecker(
            selection_blob=read_ref(helper_ref)[1],
            selection_sha=helper_ref["sha256"],
            receipt_directory=checker_out,
        )
        require(
            checker.binding == plan["arms"][group]["helper_binding"]
            and checker.model == plan["model"],
            "Independently selected live helper/model",
        )

        def admitted(binding, phase):
            owner_admission(plan, group, digest, owner)
            host_idle(plan, observer)
            for ref in plan["sources"]:
                read_ref(ref)
            return checker(binding, phase)

        admitted(checker.binding, "actor-before-model-child")
        text = checker.text_connection()
        transport = transport_for(plan)
        transport_out = out / "transport"
        transport_out.mkdir(mode=0o700)
        session = checked_session_class(transport)(
            selection_for(plan, group, transport),
            transport_out,
            helper_binding=checker.binding,
            checker=admitted,
        )
        environment = {
            k: v
            for k, v in os.environ.items()
            if not any(part in k.upper() for part in ("TOKEN", "KEY", "SECRET", "PASSWORD"))
        }
        environment.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_CACHE": str(ROOT / "artifacts/jev-local-hf/hub"),
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
                "HF_TOKEN_PATH": "/nonexistent-d3-public-token",
                "TOKENIZERS_PARALLELISM": "false",
                "HF_ENABLE_PARALLEL_LOADING": "false",
                "HF_PARALLEL_LOADING_WORKERS": "1",
                "HF_MODULES_CACHE": str(
                    ROOT
                    / "artifacts/jev-d3-browser-module-cache"
                    / plan["arms"][group]["nonce"]
                    / plan["model"]
                ),
                "PYTHONPATH": str(ROOT / "artifacts/jev-runtime/overlay"),
            }
        )
        session.start(
            [
                plan["interpreter"],
                "-B",
                "-s",
                str(HERE / "launch_execution.py"),
                "--execute",
                "--role",
                "worker",
                "--plan",
                str(plan_path),
                "--plan-sha256",
                digest,
                "--group",
                group,
            ],
            env=environment,
            cwd=ROOT,
            timeout=3600,
        )
        gateway = transport.LocalGateway(session)
        token = secrets.token_urlsafe(32)
        http = transport.LoopbackHTTP(gateway, token=token)
        endpoint = http.start() + "/predict"
        v7 = load_module(plan["components"]["v7"], "_d3_execution_v7")
        bundle, manifest = v7.prepare_bundle(
            out / "bundle", case_module="original" if group == "known4" else "heldout"
        )
        command = browser_command(plan, group, bundle, out / "results", endpoint, text)
        environment.update({"S1_POLICY_API_KEY": token, "S1_TEXT_API_KEY": text["api_key"]})
        # The selected runtime's overlay is the same ordinary runtime profile as bootstrap.
        environment["PYTHONPATH"] = (
            str(ROOT / "src") + ":" + str(ROOT / "artifacts/jev-runtime/overlay")
        )
        with (out / "browser.log").open("xb") as log:
            browser = subprocess.Popen(command, env=environment, cwd=ROOT, stdout=log, stderr=log)
            finish = time.monotonic() + 3300
            while browser.poll() is None:
                require(time.monotonic() < finish, "Owned browser group deadline")
                owner_admission(plan, group, digest, owner)
                host_idle(plan, observer)
                require(not session.blocked, "Model arm refused; stop browser group")
                time.sleep(2)
        require(browser.returncode == 0, "Original browser process failed")
        v7.verify_bundle(bundle, manifest)
        score = browser_score(out / "results/summary.json", group)
    except BaseException as error:
        primary = error_record(error)
    finally:
        # On failure, the external selected-unit owner stops any live browser
        # and Chromium descendants. Do not reopen and signal a numeric PID here.
        closure, primary, cleanup_errors = close_actor_resources(gateway, http, session, primary)
        try:
            host_idle(plan, observer)
            owner_admission(plan, group, digest, owner)
            for ref in plan["sources"]:
                read_ref(ref)
        except BaseException as error:
            primary = primary or error_record(error)
        save(
            out / "actor-result.json",
            {
                "group": group,
                "plan_sha256": digest,
                "score": score,
                "primary_error": primary,
                "cleanup_errors": cleanup_errors,
                "browser_exit_confirmed": browser is not None and browser.poll() is not None,
                "browser_stop_scope": "external authenticated selected unit and observed cohort",
                "transport_success": closure["success"],
                "model_child_started": session is not None and session.process is not None,
                "actor_success": primary is None and closure["success"] and score is not None,
                "model_qualified": False,
                "external_scoped_closure_required": True,
            },
        )
    # Remain the selected live MainPID until its external owner reauthenticates stop.
    while True:
        time.sleep(1)


def controller_main(plan, digest, blob):
    require(plan["profile"] == PROFILE, "Actual controller profile only")
    owner = load_module(plan["components"]["boundary"], "_d3_execution_owner_controller")
    observer = load_module(plan["components"]["observer"], "_d3_execution_observer_controller")
    output = ordinary(plan["output"])
    require(not output.exists(), "Fresh overall output")
    host_idle(plan, observer)
    require(
        subprocess.run(
            ["git", "-C", plan["browser"]["upstream_path"], "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        == plan["browser"]["upstream_commit"],
        "Exact upstream checkout",
    )
    require(
        not subprocess.run(
            [
                "git",
                "-C",
                plan["browser"]["upstream_path"],
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        ).stdout.strip(),
        "Clean upstream tracked sources",
    )
    output.mkdir(mode=0o700, parents=True)
    sealed = output / "sealed-plan.json"
    with sealed.open("xb") as stream:
        stream.write(blob)
    groups, primary = {}, None
    for group in GROUPS:
        selection = unit_selection(plan, group, owner)
        out = group_output(plan, group)
        boundary, closure = None, None
        try:
            require(
                time.monotonic() < plan["expires_monotonic"], "Fresh selected next group launch"
            )
            host_idle(plan, observer)
            require(
                owner.status(selection).get("LoadState") == "not-found", "Fresh exact unit absent"
            )
            require(
                not (owner.CGROUP_ROOT / selection.cgroup.lstrip("/")).exists(),
                "Fresh selected cgroup absent",
            )
            require(
                not Path(plan["arms"][group]["model_output"]).exists(),
                "Fresh separate model output",
            )
            out.mkdir(mode=0o700)
            boundary = owner.UnitBoundary(selection, out, plan["sources"])
            props = {
                **selection.properties(),
                "RuntimeMaxSec": "7200",
                "TimeoutStopSec": "10",
                "WorkingDirectory": str(ROOT),
                "LimitCORE": "0",
                "StandardOutput": "append:" + str(out / "actor.log"),
                "StandardError": "append:" + str(out / "actor.log"),
            }
            del props["RuntimeMaxUSec"], props["TimeoutStopUSec"]
            argv = [
                "systemd-run",
                "--user",
                "--unit=" + selection.unit,
                *["--property=" + k + "=" + v for k, v in props.items()],
                "--setenv=JEV_D3_BOUNDARY_NONCE=" + selection.nonce,
                plan["interpreter"],
                "-B",
                "-s",
                str(HERE / "launch_execution.py"),
                "--execute",
                "--role",
                "actor",
                "--plan",
                str(sealed),
                "--plan-sha256",
                digest,
                "--group",
                group,
            ]
            save(
                out / "launch-intent.json",
                {"argv": argv, "cap_bytes": CAP, "swap_bytes": 0, "runtime_seconds": 7200},
            )
            owner.command(*argv, timeout=8)
            deadline = time.monotonic() + 60
            while not (out / "actor-intent.json").is_file():
                require(time.monotonic() < deadline, "Timed owned actor readiness")
                host_idle(plan, observer)
                time.sleep(0.1)
            readback = boundary.verify()
            intent = parse((out / "actor-intent.json").read_bytes())
            require(
                intent["plan_sha256"] == digest
                and intent["group"] == group
                and all(
                    intent["actor"][k] == readback["actor"][k]
                    for k in ("pid", "start_ticks", "pgid", "sid", "uid", "cgroup")
                ),
                "Exact actual actor intent",
            )
            save(
                out / "admission.json",
                {
                    "plan_sha256": digest,
                    "group": group,
                    "nonce": selection.nonce,
                    "invocation_id": readback["unit"]["InvocationID"],
                    "readback_sha256": hashlib.sha256(
                        (out / "unit-readback.json").read_bytes()
                    ).hexdigest(),
                },
            )
            deadline = time.monotonic() + 7080
            while not (out / "actor-result.json").is_file():
                require(time.monotonic() < deadline, "Owned group outer deadline")
                row = boundary.owned_status()
                require(
                    row.get("ActiveState") == "active"
                    and row.get("SubState") == "running"
                    and row.get("MainPID") == str(readback["actor"]["pid"]),
                    "Selected actor became terminal",
                )
                host_idle(plan, observer)
                time.sleep(2)
            result = parse((out / "actor-result.json").read_bytes())
            require(
                result.get("plan_sha256") == digest and result.get("group") == group,
                "Bound actual group result",
            )
            groups[group] = result
            require(result.get("actor_success") is True, "Actual selected group refused")
        except BaseException as error:
            primary = error_record(error)
        finally:
            if boundary is not None:
                try:
                    closure = boundary.close(primary_error=primary)
                except BaseException as error:
                    primary = primary or error_record(error)
                    closure = {"closure_error": error_record(error)}
            if out.exists():
                final_host = None
                try:
                    final_host = host_idle(plan, observer)
                except BaseException as error:
                    primary = primary or error_record(error)
                save(
                    out / "controller-final.json",
                    {"primary_error": primary, "closure": closure, "final_host": final_host},
                )
        if (
            primary is not None
            or closure is None
            or not (
                closure.get("whole_unit_exit_confirmed") is True
                and closure.get("observed_cgroup_cohort_exit_confirmed") is True
                and closure.get("sources_unchanged") is True
            )
        ):
            primary = primary or {"type": "ExternalScopedClosureUnproven"}
            break
    report = aggregate_result(plan, digest, groups, primary)
    save(output / "execution-result.json", report)
    return 0 if report["completed"] else 1


def aggregate_result(plan, digest, groups, primary):
    successes = sum(
        (groups.get(group, {}).get("score") or {}).get("successes", 0) for group in GROUPS
    )
    complete = (
        primary is None
        and set(groups) == set(GROUPS)
        and all(row["actor_success"] for row in groups.values())
    )
    return {
        "profile": PROFILE,
        "model": plan["model"],
        "plan_sha256": digest,
        "completed": complete,
        "attempts_denominator": 10,
        "observed_complete_group_attempts": sum(
            (row.get("score") or {}).get("attempts", 0) for row in groups.values()
        ),
        "observed_successes": successes,
        "successes": successes if complete else None,
        "success_rate": successes / 10 if complete else None,
        "primary_error": primary,
        "groups": groups,
        "model_qualified": False,
        "scope": "Selected two units and observed cgroup cohorts; no universal no-escape claim",
    }
