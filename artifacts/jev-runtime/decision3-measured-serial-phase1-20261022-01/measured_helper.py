"""Q95 authentication reused under an explicitly different measured host policy.

This is not helper85's original full-host-floor co-resident qualification. The
initial check keeps Q95 R+2BASE; postwarm keeps its actual owner's R+2GiB rule.
The model child independently enforces the new CPU/CUDA resource policy.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import os
import re
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).absolute().parents[3]
HERE = Path(__file__).absolute().parent
PARENT = ROOT / "artifacts/jev-runtime/decision3-local-helper-budget-20261022-01/live_helper.py"
PARENT_SHA = "85f2eef27f4455710635514feebe90611a52b22c6e6781da09899dfaa27482ba"
POLICY_SHA = "af6c4082a8f2fec084267b1760088c7f02dfdbbaf4bbc4cfae231b6ae09e7432"
SCHEMA = "decision3-measured-phase1-q95-helper-selection-v1"
PROFILE = "decision3-gx10-serial-mmap-measured-phase1-v1"


def load(path, expected, name):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Ordinary selected measured helper source")
    blob = path.read_bytes()
    if len(blob) > 1024**2 or hashlib.sha256(blob).hexdigest() != expected:
        raise ValueError("Selected measured helper mechanism source differs")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(blob, str(path), "exec", dont_inherit=True), vars(module))
    return module


class MeasuredHelperChecker:
    def __init__(self, selection_blob, selection_sha, receipt_directory):
        p = self.p = load(PARENT, PARENT_SHA, "_measured_selected_q95_auth")
        self.policy = load(HERE / "resource_policy.py", POLICY_SHA, "_measured_helper_policy")
        p.need(
            type(selection_blob) is bytes
            and len(selection_blob) <= 1024**2
            and type(selection_sha) is str
            and re.fullmatch(r"[0-9a-f]{64}", selection_sha)
            and hashlib.sha256(selection_blob).hexdigest() == selection_sha,
            "EXTERNALLY_SELECTED_MEASURED_HELPER",
        )
        value = p.parse(selection_blob)
        p.exact_fields(value, "schema_id helper_seed_ref measured_co_resident observations_ref")
        p.need(value["schema_id"] == SCHEMA, "EXPLICIT_MEASURED_HELPER_PROFILE")
        seed = p.parse(p.ref_blob(value["helper_seed_ref"], value["helper_seed_ref"]["sha256"]))
        p.exact_fields(
            seed,
            "schema_id binding worker listener service_directory private_config interpreter source_root qualification execution runtime registry expected_memtotal",
        )
        p.need(seed["schema_id"] == p.SCHEMA, "ORIGINAL_FRESH_Q95_OWNER_SEED")
        co = value["measured_co_resident"]
        p.exact_fields(co, "profile unit expected")
        p.need(
            co["profile"] == PROFILE
            and type(co["unit"]) is str
            and re.fullmatch(r"jev-d3-measured-[a-z0-9-]{1,70}\.service", co["unit"]),
            "EXPLICIT_FRESH_MEASURED_UNIT",
        )
        expected = self.policy.selected_expected(co["expected"])
        group = f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{co['unit']}"
        p.need(
            expected["cgroup_path"] == "/sys/fs/cgroup" + group
            and expected["boot_id"] == seed["binding"]["boot_id"]
            and expected["mem_total_bytes"] == seed["expected_memtotal"],
            "MEASURED_HELPER_HOST_AND_UNIT",
        )
        observer = value["observations_ref"]
        p.need(
            observer["path"] == str((HERE / "observations.py").relative_to(ROOT)),
            "EXACT_MEASURED_OBSERVATIONS_SOURCE",
        )
        p.ref_blob(observer, observer["sha256"])
        observations = load(
            HERE / "observations.py", observer["sha256"], "_measured_helper_observations"
        )
        self._descriptor, self.expected, self.observations = value, expected, observations
        self._mode, self._initial_admitted, self._host_baseline = "initial", False, None
        self._phase_lock = threading.Lock()
        # Only the static schema is projected to the checked original constructor.
        # Its live/auth/source methods remain unchanged; _memory below is named
        # separately and never invokes the old CORE R+2D+2GiB decision.
        projected = {
            **seed,
            "co_resident": {
                "model": "d3",
                "resident_file_bytes": expected["resident_bytes"],
                "cgroup": group,
            },
        }
        blob = p.encode(projected)
        p.LiveHelperChecker.__init__(
            self,
            selection_blob=blob,
            selection_sha=hashlib.sha256(blob).hexdigest(),
            receipt_directory=receipt_directory,
        )
        self._sha = selection_sha

    @property
    def binding(self):
        return copy.deepcopy(self._value["binding"])

    @property
    def model(self):
        return "d3"

    @property
    def resident_bytes(self):
        return self.expected["resident_bytes"]

    @property
    def selection_sha(self):
        return self._sha

    def _static(self):
        p = self.p
        ref = self._descriptor["helper_seed_ref"]
        seed = p.parse(p.ref_blob(ref, ref["sha256"]))
        observation = self._descriptor["observations_ref"]
        p.ref_blob(observation, observation["sha256"])
        p.need(
            seed == {k: v for k, v in self._value.items() if k != "co_resident"},
            "EXACT_SELECTED_FRESH_Q95_SEED",
        )
        return p.LiveHelperChecker._static(self)

    def _live(self):
        return self.p.LiveHelperChecker._live(self)

    def _auth(self, token, deadline):
        return self.p.LiveHelperChecker._auth(self, token, deadline)

    def _memory(self):
        p = self.p
        snapshot = self.observations.host_snapshot(self.expected)
        decision = self.policy.host_decision(
            snapshot, self.expected, self._host_baseline, initial=self._mode == "initial"
        )
        p.need(decision["admitted"] is True, "MEASURED_HOST_CGROUP_REFUSED")
        reserve = decision["host"]["reserve_bytes"]
        helper_floor = reserve + (2 * p.BASE_BYTES if self._mode == "initial" else 2 * p.GIB)
        p.need(
            snapshot["proc_meminfo"]["MemAvailable"] >= helper_floor,
            "Q95_INITIAL_OR_STEADY_HOST_FLOOR",
        )
        helper_groups = p.cgroup_memory(self._value["worker"]["cgroup"])
        p.need(bool(helper_groups), "COMPLETE_Q95_CGROUP_OBSERVATION")
        self._host_baseline = decision["baseline"]
        return {
            "profile": PROFILE,
            "phase": self._mode,
            "helper": {
                "admitted": True,
                "required_host_available_bytes": helper_floor,
                "basis": "Q95 initial R+2BASE; actual selected-owner steady R+2GiB",
                "cgroups_recorded_only": helper_groups,
            },
            "co_resident": decision,
            "cuda_observed": False,
            "original85_full_host_floor_qualification_claimed": False,
        }

    def __call__(self, binding, phase):
        p = self.p
        p.need(phase in {"before-cache-and-framework", "post-warmup"}, "PHASE1_HELPER_CHECK_ONLY")
        p.need(self._phase_lock.acquire(blocking=False), "MEASURED_CHECK_ALREADY_RUNNING")
        try:
            p.need(
                phase != "post-warmup" or self._initial_admitted,
                "SUCCESSFUL_INITIAL_HELPER_REQUIRED",
            )
            p.need(
                phase != "before-cache-and-framework" or self._mode == "initial",
                "NO_RETURN_TO_INITIAL_PHASE",
            )
            self._mode = "initial" if phase == "before-cache-and-framework" else "steady"
            result = p.LiveHelperChecker.__call__(self, binding, phase)
            if self._mode == "initial":
                self._initial_admitted = True
            return result
        finally:
            self._phase_lock.release()
