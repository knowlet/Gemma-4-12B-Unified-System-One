#!/usr/bin/env python3
"""Current-source GX10 native Decision3 bootstrap component.

Explicit current18/union19 profile; historical17/H15 remain untouched.
All six identities retained, unchanged floors/native255/text48. No actor/READY.
The CLI is dry and refuses --run; use the sealed callable under a future owner.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
SELF = Path(__file__).resolve()
SOURCE17 = ROOT / "artifacts/jev-runtime/decision3-matched-v7-warm-readiness-20261018-01"
HISTORICAL_MANIFEST = SOURCE17 / "source-inventory17.json"
HISTORICAL_SHA = "d8c886416bfc54192a3f861a8cb386e25754606a42410317056d2eef2f638985"
MANIFEST = SELF.parent / "current-source18.json"
MANIFEST_SHA = "be987eabd8bea9b9aa2c412983f702a40008d9dd552981fd6c92aac0da817ba4"
WORKER = SELF.parent / "current_worker.py"
WORKER_SHA = "edb7e6ed3e6b250a76d00189998d403c1cdda2bdec2cedafacc08aa8f97306c3"
PROFILE = "decision3-gx10-current-q95-text48-component-v1"
INVENTORY_KIND = "decision3-gx10-current-component-source-inventory-v1"
RUNTIME_KIND = "decision3-gx10-local-runtime-selection-v1"
SELECTION_KIND = "decision3-gx10-current-bootstrap-selection-v1"
HELPER_SHA = "d33bfcac49f90c47b626c3915baf3d5fcbfb2e32522bfaccbbfec59dce35d8db"
HELPER_PROFILE = "field-syntax-preserving-base-helper-v1"
MODELS = ("d3", "d3-flash", "d3-mini", "d3-nano", "d3-lite", "d3-edge")
CACHE_CANDIDATES = MODELS
GROUPS = {"known-four": (4, 64), "public-six": (6, 96)}
REPLAY_ROOT = ROOT / "artifacts/jev-d3-native"
REQUIRED_DISTRIBUTIONS = set(
    (
        "torch torchvision transformers tokenizers safetensors accelerate huggingface-hub numpy"
    ).split()
)

SHA = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fields(value, names, message):
    require(type(value) is dict and set(value) == set(names.split()), message)


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def checked_blob(path, expected=None):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Ordinary source path")
    require(path.is_file() and path.stat().st_size <= 1024**2, "Small source/metadata only")
    blob = path.read_bytes()
    if expected is not None:
        require(hashlib.sha256(blob).hexdigest() == expected, "Selected source SHA differs")
    return blob


def selected_inputs(blob, expected_sha):
    """Read an externally sealed self/union/runtime selection; never reseal live bytes."""
    require(type(blob) is bytes and type(expected_sha) is str and SHA.fullmatch(expected_sha),
            "Externally selected bootstrap descriptor SHA required")
    require(hashlib.sha256(blob).hexdigest() == expected_sha, "Bootstrap descriptor SHA differs")
    value = json.loads(blob)
    fields(value, "schema_id profile files", "Exact selected bootstrap descriptor")
    require(value["schema_id"] == SELECTION_KIND and value["profile"] == PROFILE,
            "Explicit current-source bootstrap selection")
    fields(value["files"], "bootstrap inventory runtime", "Exact selected input files")
    expected_paths = {"bootstrap": SELF, "inventory": SELF.parent / "source-inventory19.json",
                      "runtime": SELF.parent / "runtime-selection.json"}
    blobs = {}
    for key, path in expected_paths.items():
        row = value["files"][key]
        fields(row, "path sha256 size_bytes", "Exact selected file authority")
        require(row["path"] == str(path) and type(row["sha256"]) is str
                and SHA.fullmatch(row["sha256"]) and type(row["size_bytes"]) is int
                and 0 < row["size_bytes"] <= 1024**2, "Bounded fixed selected file")
        blobs[key] = checked_blob(path, row["sha256"])
        require(len(blobs[key]) == row["size_bytes"], "Selected file size changed")
    inventory = load_worker().parse(blobs["inventory"])
    return {"inventory": inventory,
            "inventory_sha": hashlib.sha256(encode(inventory)).hexdigest(),
            "runtime_blob": blobs["runtime"],
            "runtime_sha": value["files"]["runtime"]["sha256"]}


def load_worker():
    blob = checked_blob(WORKER, WORKER_SHA)
    spec = importlib.util.spec_from_file_location("gx10_selected_current18_worker", WORKER)
    require(spec is not None and spec.loader is not None, "Original worker module spec")
    module = importlib.util.module_from_spec(spec)
    exec(compile(blob, str(WORKER), "exec", dont_inherit=True), module.__dict__)
    return module


def source_inventory(worker):
    """Preparation only: actual calls must supply a previously sealed union SHA."""
    selected = worker.parse(checked_blob(MANIFEST, MANIFEST_SHA))
    historical = worker.parse(checked_blob(HISTORICAL_MANIFEST, HISTORICAL_SHA))
    rows = copy.deepcopy(selected["rows"])
    require(len(rows) == 18 and len({r["path"] for r in rows}) == 18, "Exact current18")
    require(str(SELF) not in {r["path"] for r in rows}, "Independent component source")
    rows.append({"path": str(SELF), "pin": worker.pin(SELF), "classification": "current_bootstrap"})
    return {
        "schema_id": INVENTORY_KIND, "profile": PROFILE, "files": 19, "rows": rows,
        "selected_manifest_sha256": MANIFEST_SHA, "selected_projection": selected,
        "historical_source17_manifest_sha256": HISTORICAL_SHA,
        "historical_source17_projection": historical,
        "helper_qualification_sha256": HELPER_SHA, "transport_ready": False,
        "whole_tree_exit_confirmed": False, "continuation_authorized": False,
    }


def validate_config(config, worker):
    fields(
        config,
        "profile model workload nonce expected_boot_id expected_memtotal expected_gpu_uuid output_dir",
        "Exact component configuration fields",
    )
    require(config["profile"] == PROFILE and config["model"] in MODELS, "Explicit current six-model profile")
    require(
        config["model"] in CACHE_CANDIDATES, "All six selected; full cache authentication still required"
    )
    require(config["workload"] in GROUPS, "Original split workload only")
    require(
        type(config["expected_memtotal"]) is int and config["expected_memtotal"] > 0,
        "Exact positive MemTotal",
    )
    for key in ("nonce", "expected_boot_id", "expected_gpu_uuid"):
        require(type(config[key]) is str and bool(config[key]), "Nonempty identity: " + key)
    require(worker.SAFE_ID.fullmatch(config["nonce"]), "Safe fresh nonce")
    output = Path(config["output_dir"]).absolute()
    require(output.is_relative_to(REPLAY_ROOT), "Original bounded replay output root required")
    require(not any(p.is_symlink() for p in (output, *output.parents)), "No linked output root")
    require(output.is_dir() and not list(output.iterdir()), "Fresh empty owned arm directory")
    return worker.startup_configuration("publisher-text48-v1", config["workload"])


def validate_helper(binding, config):
    """Static selection only. Freshness and coexistence require a separate live checker."""
    fields(
        binding,
        "profile qualification_sha256 source_count boot_id pid start_ticks pgid sid endpoint seconds tokens context",
        "Exact public helper identity, without private token",
    )
    require(
        binding["profile"] == HELPER_PROFILE and binding["qualification_sha256"] == HELPER_SHA,
        "New Q95 helper required; original Q93 cannot be relabelled",
    )
    for key, expected in (
        ("source_count", 95),
        ("seconds", 20),
        ("tokens", 128),
        ("context", 8192),
    ):
        require(type(binding[key]) is int and binding[key] == expected, "Pinned helper " + key)
    require(binding["boot_id"] == config["expected_boot_id"], "Same boot helper identity")
    require(
        all(
            type(binding[k]) is int and binding[k] > 0
            for k in ("pid", "start_ticks", "pgid", "sid")
        ),
        "Typed owned helper identity",
    )
    endpoint = urlsplit(binding["endpoint"])
    require(
        endpoint.scheme == "http"
        and endpoint.hostname == "127.0.0.1"
        and endpoint.port is not None
        and endpoint.path == ""
        and not endpoint.query
        and not endpoint.fragment
        and endpoint.username is None,
        "Public loopback helper endpoint only",
    )
    return copy.deepcopy(binding)


def live_helper_check(checker, binding, phase):
    evidence = checker(copy.deepcopy(binding), phase)
    fields(
        evidence,
        "binding phase helper_admitted co_resident_admitted receipt_sha256",
        "Exact admission evidence",
    )
    require(
        evidence["binding"] == binding
        and evidence["phase"] == phase
        and evidence["helper_admitted"] is True
        and evidence["co_resident_admitted"] is True
        and type(evidence["receipt_sha256"]) is str
        and SHA.fullmatch(evidence["receipt_sha256"]),
        "Fresh full helper and co-resident admission evidence required",
    )
    return evidence


def offline_environment(config, core):
    modules = ROOT / "artifacts/jev-d3-browser-module-cache" / config["nonce"] / config["model"]
    required = {
        "HF_HUB_CACHE": str(core.HUB),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_TOKEN_PATH": "/nonexistent-d3-public-token",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_ENABLE_PARALLEL_LOADING": "false",
        "HF_PARALLEL_LOADING_WORKERS": "1",
        "HF_MODULES_CACHE": str(modules),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    require(
        all(os.environ.get(k) == v for k, v in required.items()),
        "Future owned launcher must establish the exact offline environment",
    )
    require(
        modules.is_dir()
        and not list(modules.iterdir())
        and not any(p.is_symlink() for p in (modules, *modules.parents)),
        "Fresh owned publisher module cache required",
    )
    blocked = []

    def deny_network(event, _arguments):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.sendto"}:
            blocked.append(event)
            raise RuntimeError("Network forbidden in offline model component")

    sys.addaudithook(deny_network)
    return blocked


class RuntimeSelection:
    """Bounded selected source/METADATA pins and module origins; not all indirect DSOs."""

    def __init__(self, blob, expected_sha, worker):
        require(
            type(blob) is bytes and type(expected_sha) is str and SHA.fullmatch(expected_sha),
            "External local runtime SHA required",
        )
        require(hashlib.sha256(blob).hexdigest() == expected_sha, "Local runtime selection SHA")
        value = worker.parse(blob)
        fields(
            value,
            "schema_id machine interpreter python files distributions modules kernel_policy",
            "Exact local runtime schema",
        )
        require(
            value["schema_id"] == RUNTIME_KIND and value["machine"] == "aarch64",
            "Separate aarch64 runtime selection required",
        )
        policy = {
            "fast": False,
            "permutation_average": False,
            "native_batch_size": 8,
            "temperature": 1.0,
            "backbone": "bfloat16",
            "readout": "float32",
            "context": 8192,
        }
        require(
            json.dumps(value["kernel_policy"], sort_keys=True)
            == json.dumps(policy, sort_keys=True),
            "Original numerical policy",
        )
        require(
            type(value["python"]) is list
            and len(value["python"]) == 2
            and all(type(v) is int for v in value["python"]),
            "Selected Python version",
        )
        require(Path(value["interpreter"]).is_absolute(), "Selected absolute interpreter")
        require(type(value["files"]) is dict and bool(value["files"]), "No placeholder source pins")
        for path, pin in value["files"].items():
            fields(pin, "sha256 size_bytes", "Selected file pin")
            require(
                Path(path).is_absolute()
                and type(pin["sha256"]) is str
                and SHA.fullmatch(pin["sha256"])
                and type(pin["size_bytes"]) is int
                and pin["size_bytes"] > 0,
                "Exact selected file pin",
            )
        distributions = value["distributions"]
        require(
            type(distributions) is dict and REQUIRED_DISTRIBUTIONS <= set(distributions),
            "Required runtime distributions",
        )
        for row in distributions.values():
            fields(row, "version metadata_path", "Selected distribution metadata")
            require(
                type(row["version"]) is str
                and bool(row["version"])
                and row["metadata_path"] in value["files"],
                "Distribution metadata pin required",
            )
        require(distributions["transformers"]["version"] == "5.17.0", "Publisher Transformers pin")
        require(
            type(value["modules"]) is dict
            and {"torch", "torch._C", "transformers"} <= set(value["modules"]),
            "Selected framework entry origins required",
        )
        for row in value["modules"].values():
            fields(row, "path kind size_bytes", "Selected module origin")
            require(
                Path(row["path"]).is_absolute()
                and row["kind"] in {"source", "native_origin"}
                and type(row["size_bytes"]) is int
                and row["size_bytes"] > 0,
                "Exact selected module origin metadata",
            )
            require(
                row["kind"] == "native_origin" or row["path"] in value["files"],
                "Selected source module must have a byte pin",
            )
        self.value, self.worker, self.sha = value, worker, expected_sha

    def verify(self, *, loaded=False):
        value = self.value
        require(
            platform.machine() == value["machine"]
            and sys.executable == value["interpreter"]
            and list(sys.version_info[:2]) == value["python"]
            and sys.flags.no_user_site
            and sys.dont_write_bytecode,
            "Selected local interpreter -B -s required",
        )
        for path, pin in value["files"].items():
            require(self.worker.pin(path) == pin, "Selected runtime file changed: " + path)
        for name, row in value["distributions"].items():
            actual = importlib.metadata.distribution(name)
            require(
                actual.version == row["version"]
                and str(Path(actual._path) / "METADATA") == row["metadata_path"],
                "Selected distribution version/origin differs: " + name,
            )
        observed = {}
        for name, row in value["modules"].items():
            path = Path(row["path"])
            require(
                path.is_file() and path.stat().st_size == row["size_bytes"],
                "Module origin stat differs",
            )
            module = sys.modules.get(name) if loaded else None
            if module is not None:
                require(
                    Path(module.__file__).resolve() == path and path.suffix != ".pyc",
                    "Selected loaded runtime origin differs: " + name,
                )
                observed[name] = row
        return {
            "runtime_sha256": self.sha,
            "loaded_origins_checked": loaded,
            "observed_selected_modules": observed,
            "indirect_DSO_payloads_authenticated": False,
        }


class ProjectSources:
    """Import selected fresh project modules from their authenticated source bytes."""

    def __init__(self, source_pins):
        self.sources = {}
        for label, expected in source_pins.items():
            path = ROOT / label
            require(label.startswith("src/s1/") and label.endswith(".py")
                    and type(expected) is str and SHA.fullmatch(expected),
                    "Exact selected project source required")
            name = label.removeprefix("src/").removesuffix(".py").replace("/", ".")
            package = name.endswith(".__init__")
            name = name.removesuffix(".__init__")
            require(name not in self.sources, "Unique selected project module")
            self.sources[name] = (path, expected, package)

    def __enter__(self):
        require(not (set(self.sources) & set(sys.modules)),
                "Fresh selected project module cache required")
        sys.meta_path.insert(0, self)
        return self

    def __exit__(self, *_error):
        sys.meta_path.remove(self)

    def find_spec(self, fullname, _path=None, _target=None):
        if fullname not in self.sources:
            require(fullname != "s1" and not fullname.startswith("s1."),
                    "Unselected project module import refused: " + fullname)
            return None
        path, _expected, package = self.sources[fullname]
        return importlib.util.spec_from_file_location(
            fullname, path, loader=self,
            submodule_search_locations=[str(path.parent)] if package else None,
        )

    def create_module(self, _spec):
        return None

    def exec_module(self, module):
        path, expected, _package = self.sources[module.__name__]
        blob = checked_blob(path, expected)
        exec(compile(blob, str(path), "exec", dont_inherit=True), module.__dict__)


def dependencies(selection):
    """Real lazy original Backend entry; selected local metadata must pass first."""
    selection.verify()
    sys.path.insert(0, str(ROOT / "src"))
    with ProjectSources(selection.worker.CORE.SOURCE_PINS) as selected:
        import torch

        from s1.evaluation.decision3 import CHECKPOINTS, Decision3Backend, decision3_spec
        from s1.jev_decision3_native import JevDecision3NativeAdapter

        for name, (path, _expected, _package) in selected.sources.items():
            module = sys.modules.get(name)
            require(
                module is not None and Path(module.__file__).resolve() == path
                and module.__loader__ is selected,
                "Executed selected project source differs: " + name,
            )
    selection.verify(loaded=True)
    return SimpleNamespace(
        torch=torch,
        checkpoints=CHECKPOINTS,
        backend=Decision3Backend,
        spec=decision3_spec,
        adapter=JevDecision3NativeAdapter,
    )


class UUIDGates:
    """Keep exact original phase labels/floors and bind every CUDA sample to one GPU."""

    def __init__(self, original, worker, gpu_uuid):
        self.original, self.worker, self.gpu_uuid = original, worker, gpu_uuid

    def check(self, stage):
        row = self.original.check(stage)
        cuda = row["snapshot"]["cuda"]
        if cuda is not None:
            require(self.worker.same_uuid(cuda["uuid"], self.gpu_uuid), "Actual CUDA UUID changed")
        return row


def geometry(backend, identity, torch, model):
    runtime = backend.runtime
    width = identity["readout_tensor_bytes"] // (255 * 4)
    require(
        all(
            p.device == torch.device("cuda:0") and p.dtype == torch.bfloat16
            for p in runtime.backbone.parameters()
        )
        and runtime.readout.device == torch.device("cuda:0")
        and runtime.readout.dtype == torch.float32
        and list(runtime.readout.shape) == [255, width]
        and sum(p.numel() for p in runtime.backbone.parameters()) + runtime.readout.numel()
        == identity["loaded_parameters"],
        "Original CUDA placement/full geometry differs",
    )
    provenance = runtime.provenance()
    require(
        provenance["model_name"] == model
        and provenance["repo_id"] == identity["model_id"]
        and provenance["pooling"] == "last",
        "Publisher provenance differs",
    )
    require(
        runtime.fast is None
        and runtime.permutation_average is False
        and runtime.batch_size == 8
        and runtime.readout_dtype == "float32"
        and runtime.temperature == 1.0
        and runtime.max_length is None
        and runtime.attention_mode == "noncausal_full_attention"
        and backend.spec.context_limit == 8192,
        "Original native numerical policy differs",
    )
    return {
        "provenance": provenance,
        "readout_shape": list(runtime.readout.shape),
        "loaded_parameters": identity["loaded_parameters"],
    }


def bootstrap(
    config,
    *,
    inventory,
    inventory_sha,
    runtime_blob,
    runtime_sha,
    helper,
    helper_check,
    worker=None,
    load_dependencies=dependencies,
):
    """Build components for a future owned controller; never emit READY or serve.

    helper_check must freshly verify exact helper identity and its full independent
    admission plus co-resident admission. No saved qualification can replace it.
    Partial constructor effects cannot be closed by this library: the future
    process supervisor must own the whole worker even when construction raises.
    """
    worker = load_worker() if worker is None else worker
    startup = validate_config(config, worker)
    selected_helper = validate_helper(helper, config)
    require(callable(helper_check), "Fresh helper/co-resident checker required")
    require(
        hashlib.sha256(encode(inventory)).hexdigest() == inventory_sha
        and inventory == source_inventory(worker),
        "Independent new source inventory required",
    )
    selection = RuntimeSelection(runtime_blob, runtime_sha, worker)
    original = inventory["selected_projection"]
    identities = worker.authenticate_sources(original)
    require(tuple(identities) == MODELS or set(identities) == set(MODELS), "Six models retained")
    core, output, name = worker.CORE, Path(config["output_dir"]), config["model"]
    require(core is not None, "Explicit current source core with unchanged resource formulas required")
    identity = identities[name]
    original_gates = core.Gates(
        output, identity, config["expected_boot_id"], config["expected_memtotal"]
    )
    gates = UUIDGates(original_gates, worker, config["expected_gpu_uuid"])
    state = {
        "profile": PROFILE,
        "model": name,
        "startup": startup,
        "component_complete": False,
        "transport_ready": False,
        "qualified": False,
        "external_owner_boundary_required": True,
        "whole_tree_exit_confirmed": False,
        "continuation_authorized": False,
        "source_inventory_sha256": inventory_sha,
        "helper": selected_helper,
        "runtime_sha256": runtime_sha,
        "error": None,
        "secondary_errors": [],
    }
    backend, torch = None, None
    try:
        gates.check("before-input-and-framework")
        state["helper_before"] = live_helper_check(
            helper_check, selected_helper, "before-cache-and-framework"
        )
        before = core.authenticate_cache(name, identity, output, gates, "initial")
        # Environment/module-cache setup belongs to the future owned launcher.
        state["network_attempts_blocked"] = offline_environment(config, core)
        deps = load_dependencies(selection)
        torch = deps.torch
        require(
            deps.checkpoints == identities
            and torch.cuda.is_available()
            and torch.cuda.device_count() == 1,
            "Pinned identities and exactly one CUDA GPU",
        )
        original_gates.torch = torch
        gates.check("before-model-constructor")  # Exact label carries original extra CUDA floor.
        spec = worker.matched_context_spec(deps.spec(name, device="cuda:0"))
        state["constructor_entered"] = True
        backend = deps.backend(
            spec,
            helper_already_loaded=True,
            expected_boot_id=config["expected_boot_id"],
            expected_memtotal=config["expected_memtotal"],
        )
        initial = geometry(backend, identity, torch, name)
        worker.publisher_warmup(
            backend.runtime,
            torch,
            gates,
            output,
            state,
            config["expected_gpu_uuid"],
            before["files"]["d3_runtime.py"],
        )
        worker.authenticate_sources(original)
        require(
            core.authenticate_cache(name, identity, output, gates, "post-warmup") == before,
            "Whole cache changed after warmup",
        )
        require(geometry(backend, identity, torch, name) == initial, "Postwarm geometry/provenance")
        state["runtime_after"] = selection.verify(loaded=True)
        state["actual_runtime_info"] = worker.evidence_value(backend.runtime.runtime_info())
        state["actual_kernels"] = worker.evidence_value(backend.runtime.kernels)
        state["helper_after"] = live_helper_check(helper_check, selected_helper, "post-warmup")
        torch.cuda.synchronize(0)
        gates.check("post-warmup-final")
        wire = worker.load_file("gx10_original_named_wire", worker.WIRE_PATH)
        adapter = deps.adapter(backend, model_identity=name)
        state.update(component_complete=True, geometry=initial)
        worker.persist(output / "bootstrap.json", state)
        return Components(worker, adapter, wire, torch, gates, output, config, startup, state)
    except BaseException as primary:
        state["component_complete"] = False
        state["error"] = worker.error_info(primary)
        if state.get("constructor_entered") and backend is None:
            state["constructor_partial_effects"] = "unknown; whole-owned-worker closure required"
        for stage, operation in (
            ("owned_backend_close", backend.close if backend else None),
            ("final_cuda_fence", (lambda: torch.cuda.synchronize(0)) if torch else None),
        ):
            if operation is not None:
                try:
                    operation()
                except BaseException as secondary:
                    state["secondary_errors"].append(
                        {"stage": stage, **worker.error_info(secondary)}
                    )
                    primary.add_note(stage + ": " + repr(secondary))
        try:
            worker.persist(output / "bootstrap.json", state)
        except BaseException as secondary:
            primary.add_note("bootstrap evidence: " + repr(secondary))
        raise


def bootstrap_selected(config, *, selection_blob, selection_sha, helper, helper_check,
                       worker=None, load_dependencies=dependencies):
    """Concrete future-worker entry; fresh helper/owner enforcement remains external."""
    inputs = selected_inputs(selection_blob, selection_sha)
    return bootstrap(config, **inputs, helper=helper, helper_check=helper_check,
                     worker=worker, load_dependencies=load_dependencies)


class Components:
    """Internal candidate response seam; no listener, scheduling, delivery or process owner."""

    def __init__(self, worker, adapter, wire, torch, gates, output, config, startup, state):
        self.worker, self.adapter, self.wire, self.torch = worker, adapter, wire, torch
        self.gates, self.output = gates, output
        self.config, self.startup, self.state = copy.deepcopy(config), startup, state
        self.closed = False

    def close(self):
        require(not self.closed, "Component already closed")
        self.closed = True
        primary = None
        for stage, operation in (
            ("backend_close", self.adapter.backend.close),
            ("final_cuda_fence", lambda: self.torch.cuda.synchronize(0)),
        ):
            try:
                operation()
            except BaseException as exc:
                if primary is None:
                    primary = exc
                else:
                    primary.add_note(stage + ": " + repr(exc))
        if primary is not None:
            raise primary
        return {
            "component_closed": True,
            "whole_worker_exit_confirmed": False,
            "external_owner_boundary_required": True,
            "whole_tree_exit_confirmed": False,
            "continuation_authorized": False,
            "final_source_cache_and_UUID_authentication_complete": False,
            "transport_ready": False,
            "qualified": False,
        }


def dry_report():
    return {
        "profile": PROFILE,
        "mode": "DRY_SOURCE_ONLY",
        "actual_started": False,
        "transport_ready": False,
        "qualified": False,
        "full_scope_models": list(MODELS),
        "cache_candidates_only": list(CACHE_CANDIDATES),
        "workloads": list(GROUPS),
        "helper_qualification_sha256": HELPER_SHA,
        "blockers": [
            "No root-selected GX10 selected-source/runtime-metadata profile yet",
            "No fresh owned Q95 helper",
            "Full helper and co-resident admission unmeasured",
            "No owned supervisor/gateway/delivery/whole-worker cleanup binding",
            "All six local caches present; actual full SHA/header/runtime qualification pending",
            "Six caches selected; each requires fresh full payload/header authentication",
        ],
        "limits": "D3 arm floor alone cannot admit the full10-task comparison; no six-model result",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    report = dry_report()
    if args.run:
        report.update(mode="ACTUAL_REFUSED_BEFORE_DYNAMIC_IMPORT", error="Bootstrap source only")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 2 if args.run else 0


if __name__ == "__main__":
    raise SystemExit(main())
