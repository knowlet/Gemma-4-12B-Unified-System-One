"""Selected constructor+publisher-text48 component, with no actor or listener.

An external owner must authenticate sources before importing this module, run
the separate50ms host sampler, supply its authenticated helper pipe callback,
and close the whole selected unit even after constructor partial effects.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).absolute().parents[3]
HERE = Path(__file__).absolute().parent
PROFILE = "decision3-gx10-serial-mmap-measured-phase1-v1"
BOOTSTRAP = (
    ROOT
    / "artifacts/jev-runtime/decision3-current-source-bootstrap-20261022-01/current_bootstrap.py"
)
BOOTSTRAP_SHA = "a39e811764f83eff793f3f5ade7fb9defd7e21fdc7d86207b5e12ce7c22b8581"
DESCRIPTOR = BOOTSTRAP.parent / "bootstrap-selection.json"
DESCRIPTOR_SHA = "034cad18a33e9cb988d5b8b697df249651cf5f1d18be711174184948ed83aa03"
POLICY_SHA = "af6c4082a8f2fec084267b1760088c7f02dfdbbaf4bbc4cfae231b6ae09e7432"
PROPOSAL = (
    ROOT
    / "artifacts/jev-runtime/decision3-measured-serial-profile-20261022-01/profile-proposal.json"
)
PROPOSAL_SHA = "3a72ba74900b28e8e505c9acba2ed9c23eb6d7107afa6a8c1f4e394e553c9235"
ASYNC_SOURCE = ROOT / "artifacts/jev-runtime/overlay/transformers/core_model_loading.py"
ASYNC_SHA = "c5b6bcdb6401a3cfdf825bc979d41ac3e7e075e09459021363dce5e68c54de22"
SHA = re.compile(r"[0-9a-f]{64}\Z")


def require(value, message):
    if not value:
        raise ValueError(message)


def encode(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode()


def parse(blob):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "Duplicate selected field")
            result[key] = value
        return result

    return json.loads(
        blob, object_pairs_hook=pairs, parse_constant=lambda _: require(False, "Nonfinite JSON")
    )


def checked(path, expected):
    path = Path(path).absolute()
    require(
        path.is_relative_to(ROOT)
        and ".." not in path.parts
        and not any(p.is_symlink() for p in (path, *path.parents)),
        "Ordinary selected component source",
    )
    with path.open("rb") as stream:
        blob = stream.read(1024**2 + 1)
    require(
        len(blob) <= 1024**2 and hashlib.sha256(blob).hexdigest() == expected,
        "Selected source SHA differs",
    )
    return blob


def load(path, expected, name):
    blob = checked(path, expected)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(blob, str(path), "exec", dont_inherit=True), vars(module))
    return module


def selected_components(blob, expected_sha):
    require(
        type(blob) is bytes
        and len(blob) <= 65536
        and type(expected_sha) is str
        and SHA.fullmatch(expected_sha)
        and hashlib.sha256(blob).hexdigest() == expected_sha,
        "Independently selected phase1 descriptor",
    )
    value = parse(blob)
    require(
        type(value) is dict
        and set(value) == {"schema_id", "profile", "files"}
        and value["schema_id"] == "decision3-measured-phase1-component-selection-v1"
        and value["profile"] == PROFILE,
        "Explicit measured component descriptor",
    )
    paths = {
        "phase1": HERE / "phase1.py",
        "backend": HERE / "measured_backend.py",
        "observations": HERE / "observations.py",
        "policy": HERE / "resource_policy.py",
        "helper": HERE / "measured_helper.py",
        "owner_sampler": HERE / "owner_sampler.py",
    }
    require(
        type(value["files"]) is dict and set(value["files"]) == set(paths),
        "Exact measured source set",
    )
    for key, path in paths.items():
        row = value["files"][key]
        require(
            type(row) is dict
            and set(row) == {"path", "sha256", "size_bytes"}
            and row["path"] == str(path.relative_to(ROOT))
            and type(row["size_bytes"]) is int
            and 0 < row["size_bytes"] <= 1024**2
            and type(row["sha256"]) is str
            and SHA.fullmatch(row["sha256"]),
            "Exact component source reference",
        )
        require(
            len(checked(path, row["sha256"])) == row["size_bytes"], "Selected source size differs"
        )
    require(value["files"]["policy"]["sha256"] == POLICY_SHA, "Selected reviewed resource policy")
    checked(PROPOSAL, PROPOSAL_SHA)
    checked(ASYNC_SOURCE, ASYNC_SHA)
    return value


@contextmanager
def serial_environment():
    """Select the checked Transformers async switch before frameworks; restore."""
    checked(ASYNC_SOURCE, ASYNC_SHA)
    require(
        not any(
            name == "torch"
            or name.startswith("torch.")
            or name == "transformers"
            or name.startswith("transformers.")
            for name in sys.modules
        ),
        "Fresh model process before framework imports",
    )
    previous = os.environ.get("HF_DEACTIVATE_ASYNC_LOAD")
    os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("HF_DEACTIVATE_ASYNC_LOAD", None)
        else:
            os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = previous


class Gates:
    """Original warmup/cache Gates interface, explicitly new resource profile."""

    def __init__(self, policy, observations, worker, output, expected):
        self.policy, self.observations, self.worker = policy, observations, worker
        self.expected = policy.selected_expected(expected)
        self.output, self.torch, self.driver = output, None, None
        self.sequence, self.baseline, self.cap_installed = 0, None, False
        (output / "gates").mkdir()

    def attach_cuda(self, torch):
        require(
            not self.cap_installed and self.torch is None and self.baseline is not None,
            "One CUDA cap installation after successful initial gate",
        )
        self.torch = torch
        self.driver = self.observations.DriverIdentity()
        host = self.observations.host_snapshot(self.expected)
        raw = self.driver.check(torch, host)
        self.policy.driver_identity(raw, self.expected)
        # This denominator is obtained now from the authenticated actual Driver,
        # Torch properties/mem_get_info and host selection; never a saved total.
        fraction = self.policy.allocator_fraction(raw["total_bytes"])
        require(torch.cuda.memory.get_allocator_backend() == "native", "Native allocator required")
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
        self.cap_installed = True
        return self.check("before-model-constructor")

    def check(self, stage):
        phase = (
            "before-framework"
            if self.torch is None
            else ("before-constructor" if stage == "before-model-constructor" else "steady")
        )
        snapshot = self.observations.host_snapshot(self.expected)
        if self.torch is not None:
            require(self.cap_installed, "Allocator cap installed before CUDA gate")
            snapshot["raw_driver"] = self.driver.check(self.torch, snapshot)
            snapshot["cuda"] = self.observations.cuda_snapshot(self.torch)
        decision = self.policy.decision(snapshot, self.expected, phase, self.baseline)
        row = {
            "sequence": self.sequence,
            "stage": stage,
            "phase": phase,
            "monotonic_ns": time.monotonic_ns(),
            "snapshot": snapshot,
            "decision": decision,
        }
        self.worker.persist(self.output / "gates" / f"gate-{self.sequence:04d}.json", row)
        self.sequence += 1
        require(
            decision["admitted"] is True,
            "Measured resource refusal: " + ",".join(decision["refusal_reasons"]),
        )
        self.baseline = decision["baseline"]
        return row


def validate_config(config, bootstrap, worker):
    require(
        type(config) is dict
        and set(config) == {"profile", "model", "nonce", "expected", "output_dir"},
        "Exact phase1 configuration",
    )
    require(config["profile"] == PROFILE and config["model"] == "d3", "Only explicit fullD3 phase1")
    require(
        type(config["nonce"]) is str and worker.SAFE_ID.fullmatch(config["nonce"]),
        "Fresh safe nonce",
    )
    output = Path(config["output_dir"]).absolute()
    require(
        output.is_relative_to(bootstrap.REPLAY_ROOT)
        and not any(p.is_symlink() for p in (output, *output.parents))
        and output.is_dir()
        and not list(output.iterdir()),
        "Fresh ordinary phase1 output",
    )
    return output


def run_selected(config, *, selection_blob, selection_sha, helper, helper_check):
    """Future owned model-worker callable; no constructor runs at import/dry CLI.

    helper_check must be the external controller's nonce/PID/start/sequence/SHA
    authenticated pipe RPC, not a socket callback in the offline model worker.
    This library does not manufacture that owner proof or authorize continuation.
    """
    selected_components(selection_blob, selection_sha)
    with serial_environment():
        return _run_selected(
            config,
            selection_blob=selection_blob,
            selection_sha=selection_sha,
            helper=helper,
            helper_check=helper_check,
        )


def _run_selected(config, *, selection_blob, selection_sha, helper, helper_check):
    selected = selected_components(selection_blob, selection_sha)
    refs = selected["files"]
    bootstrap = load(BOOTSTRAP, BOOTSTRAP_SHA, "_measured_d3_selected_bootstrap")
    worker = bootstrap.load_worker()
    inputs = bootstrap.selected_inputs(checked(DESCRIPTOR, DESCRIPTOR_SHA), DESCRIPTOR_SHA)
    identities = worker.authenticate_sources(inputs["inventory"]["selected_projection"])
    require(set(identities) == set(bootstrap.MODELS), "All six unchanged checkpoint identities")
    policy = load(HERE / "resource_policy.py", POLICY_SHA, "_measured_d3_policy")
    observations = load(
        HERE / "observations.py", refs["observations"]["sha256"], "_measured_d3_observations"
    )
    backend_source = load(
        HERE / "measured_backend.py", refs["backend"]["sha256"], "_measured_d3_backend"
    )
    output = validate_config(config, bootstrap, worker)
    expected = policy.selected_expected(config["expected"])
    require(
        expected["resident_bytes"] == identities["d3"]["resident_file_bytes"],
        "Exact fullD3 resident",
    )
    original_config = {
        "profile": bootstrap.PROFILE,
        "model": "d3",
        "nonce": config["nonce"],
        "expected_boot_id": expected["boot_id"],
        "expected_memtotal": expected["mem_total_bytes"],
        "expected_gpu_uuid": expected["gpu_uuid"],
        "output_dir": str(output),
    }
    binding = bootstrap.validate_helper(helper, original_config)
    require(callable(helper_check), "Authenticated external helper control RPC required")
    require(
        os.environ.get("HF_DEACTIVATE_ASYNC_LOAD") == "1",
        "Serial environment before framework imports",
    )
    gates = Gates(policy, observations, worker, output, expected)
    state = {
        "profile": PROFILE,
        "scope": "constructor-and-publisher-text48-only",
        "model": "d3",
        "component_selection_sha256": selection_sha,
        "measurement_complete": False,
        "qualified": False,
        "score": None,
        "transport_ready": False,
        "browser_started": False,
        "context8192_peak_qualified": False,
        "external_sampler_and_owner_required": True,
        "whole_tree_exit_confirmed": False,
        "continuation_authorized": False,
        "error": None,
        "secondary_errors": [],
        "allocator_peak_reset_performed": False,
    }
    backend, torch, primary = None, None, None
    try:
        gates.check("before-input-and-framework")
        state["helper_before"] = bootstrap.live_helper_check(
            helper_check, binding, "before-cache-and-framework"
        )
        before = worker.CORE.authenticate_cache("d3", identities["d3"], output, gates, "initial")
        state["network_attempts_blocked"] = bootstrap.offline_environment(
            original_config, worker.CORE
        )
        selection = bootstrap.RuntimeSelection(
            inputs["runtime_blob"], inputs["runtime_sha"], worker
        )
        deps = bootstrap.dependencies(selection)
        torch = deps.torch
        require(
            deps.checkpoints == identities
            and torch.cuda.is_available()
            and torch.cuda.device_count() == 1,
            "Selected dependencies and exactly one CUDA device",
        )
        state["cap_installation"] = gates.attach_cuda(torch)
        original = sys.modules[deps.backend.__module__]
        constructor = backend_source.backend_type(original, gates)
        spec = worker.matched_context_spec(deps.spec("d3", device="cuda:0"))
        state["constructor_entered"] = True
        started = time.monotonic()
        backend = constructor(
            spec,
            helper_already_loaded=True,
            expected_boot_id=expected["boot_id"],
            expected_memtotal=expected["mem_total_bytes"],
        )
        torch.cuda.synchronize(0)
        state["constructor_elapsed_seconds"] = time.monotonic() - started
        gates.check("after-model-constructor")
        initial = bootstrap.geometry(backend, identities["d3"], torch, "d3")
        worker.publisher_warmup(
            backend.runtime,
            torch,
            gates,
            output,
            state,
            expected["gpu_uuid"],
            before["files"]["d3_runtime.py"],
        )
        worker.authenticate_sources(inputs["inventory"]["selected_projection"])
        require(
            worker.CORE.authenticate_cache("d3", identities["d3"], output, gates, "post-warmup")
            == before,
            "Whole selected cache changed after warmup",
        )
        require(
            bootstrap.geometry(backend, identities["d3"], torch, "d3") == initial,
            "Postwarm geometry/provenance",
        )
        state["runtime_after"] = selection.verify(loaded=True)
        state["actual_runtime_info"] = worker.evidence_value(backend.runtime.runtime_info())
        state["actual_kernels"] = worker.evidence_value(backend.runtime.kernels)
        state["helper_after"] = bootstrap.live_helper_check(helper_check, binding, "post-warmup")
        torch.cuda.synchronize(0)
        state["final_resources"] = gates.check("post-warmup-final")
        selected_components(selection_blob, selection_sha)
        state.update(measurement_complete=True, geometry=initial, driver=gates.driver.evidence)
    except BaseException as error:
        primary = error
        state["error"] = worker.error_info(error)
        if state.get("constructor_entered") and backend is None:
            state["constructor_partial_effects"] = "Unknown; external whole-unit closure required"
    finally:
        for stage, operation in (
            ("owned_backend_close", backend.close if backend else None),
            ("final_cuda_fence", (lambda: torch.cuda.synchronize(0)) if torch else None),
        ):
            if operation is not None:
                try:
                    operation()
                except BaseException as error:
                    state["secondary_errors"].append({"stage": stage, **worker.error_info(error)})
                    if primary is None:
                        primary = error
        if primary is not None:
            state["measurement_complete"] = False
            state["error"] = state["error"] or worker.error_info(primary)
        try:
            worker.persist(output / "phase1.json", state)
        except BaseException as error:
            if primary is None:
                primary = error
            else:
                primary.add_note("Phase1 evidence persistence also failed: " + repr(error))
    if primary is not None:
        raise primary
    return state


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    require(not args.run, "No actor entrypoint; independently selected owner integration required")
    print(
        json.dumps(
            {
                "profile": PROFILE,
                "actual_model_started": False,
                "score": None,
                "phase1_only": True,
                "owner_integration_required": True,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
