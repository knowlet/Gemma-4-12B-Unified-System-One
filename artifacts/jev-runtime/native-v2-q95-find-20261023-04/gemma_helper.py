"""Same Q95 helper, explicitly separate original Gemma NF4 co-resident rules."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).absolute().parents[3]
PARENT = ROOT / "artifacts/jev-runtime/decision3-local-helper-budget-20261022-01/live_helper.py"
PARENT_SHA = "85f2eef27f4455710635514feebe90611a52b22c6e6781da09899dfaa27482ba"
SCHEMA = "gemma-native-nf4-v2-q95-helper-selection-v1"
NATIVE_PROFILE = "gemma-native-nf4-v2-with-q95-v1"
ADAPTER_SHA = "8d4f014be160f632936678db32be8c73d3413f301c368eb86d6c69644ac7727c"


def load_parent():
    blob = PARENT.read_bytes()
    if hashlib.sha256(blob).hexdigest() != PARENT_SHA:
        raise ValueError("Selected shared helper mechanism source differs")
    spec = importlib.util.spec_from_file_location("_gemma_selected_q95_mechanisms", PARENT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(blob, str(PARENT), "exec", dont_inherit=True), module.__dict__)
    return module


class GemmaHelperChecker:
    """Reuse Q95 methods, never the D3 constructor schema or CORE memory floor."""

    def __init__(self, selection_blob, selection_sha, receipt_directory):
        p = self.p = load_parent()
        p.need(
            hashlib.sha256(selection_blob).hexdigest() == selection_sha,
            "SELECTED_GEMMA_Q95_DESCRIPTOR",
        )
        value = p.parse(selection_blob)
        p.exact_fields(value, "schema_id helper_seed_ref native_co_resident")
        p.need(value["schema_id"] == SCHEMA, "EXPLICIT_GEMMA_Q95_CONTROL_SCHEMA")
        seed_ref = value["helper_seed_ref"]
        seed = p.parse(p.ref_blob(seed_ref, seed_ref["sha256"]))
        p.exact_fields(
            seed,
            "schema_id binding worker listener service_directory private_config interpreter source_root qualification execution runtime registry expected_memtotal",
        )
        p.need(seed["schema_id"] == p.SCHEMA, "ORIGINAL_Q95_OWNER_SEED")
        binding = seed["binding"]
        p.exact_fields(
            binding,
            "profile qualification_sha256 source_count boot_id pid start_ticks pgid sid endpoint seconds tokens context",
        )
        for key, expected in {
            "profile": p.PROFILE,
            "qualification_sha256": p.Q_SHA,
            "source_count": 95,
            "seconds": 20,
            "tokens": 128,
            "context": 8192,
        }.items():
            p.need(
                binding[key] == expected and type(binding[key]) is type(expected),
                "SAME_EXACT_Q95_HELPER",
            )
        p.need(
            all(
                type(binding[k]) is int and binding[k] > 0
                for k in ("pid", "start_ticks", "pgid", "sid")
            )
            and binding["pid"] == binding["pgid"] == binding["sid"],
            "ORIGINAL_HELPER_SESSION",
        )
        p.exact_fields(seed["worker"], "pid start_ticks pgid sid boot_id exe argv cgroup")
        p.need(
            all(
                seed["worker"][k] == binding[k]
                for k in ("pid", "start_ticks", "pgid", "sid", "boot_id")
            ),
            "EXACT_HELPER_PROCESS_BINDING",
        )
        source = p.ordinary(seed["source_root"])
        service = p.ordinary(seed["service_directory"])
        p.need(
            source.is_relative_to(p.ROOT)
            and service.is_relative_to(p.ROOT / "artifacts/jev-runtime")
            and p.ordinary(seed["private_config"]) == service / "controller-config.json",
            "ORIGINAL_CONFINED_Q95_PATHS",
        )
        p.need(
            seed["worker"]["argv"]
            == [
                seed["interpreter"],
                str(source / "scripts/jev_qualify_public_field_syntax_bf16.py"),
                "--internal-child",
                seed["private_config"],
            ]
            and seed["worker"]["exe"] == str(Path(seed["interpreter"]).resolve(strict=True)),
            "EXACT_ORIGINAL_Q95_COMMAND",
        )
        endpoint = p.urlsplit(binding["endpoint"])
        p.need(
            endpoint.scheme == "http"
            and endpoint.hostname == "127.0.0.1"
            and endpoint.port
            and endpoint.path == ""
            and not endpoint.query
            and not endpoint.fragment
            and endpoint.username is None,
            "EXACT_Q95_LOOPBACK_ENDPOINT",
        )
        p.exact_fields(seed["listener"], "pid address port inode")
        p.need(
            seed["listener"]["pid"] == binding["pid"]
            and seed["listener"]["address"] == "127.0.0.1"
            and seed["listener"]["port"] == endpoint.port
            and type(seed["listener"]["inode"]) is str
            and seed["listener"]["inode"].isdecimal(),
            "EXACT_Q95_LISTENER_SELECTION",
        )
        p.need(
            type(seed["expected_memtotal"]) is int and seed["expected_memtotal"] > 0,
            "SELECTED_HOST_TOTAL",
        )
        co = value["native_co_resident"]
        p.exact_fields(co, "profile unit cgroup memory_max_bytes adapter_manifest")
        p.need(
            co["profile"] == NATIVE_PROFILE
            and co["unit"] == "jev-current-nf4-v2-q95-find-20261023-04.service"
            and co["memory_max_bytes"] == 40 * p.GIB,
            "EXPLICIT_ORIGINAL_GEMMA_NATIVE40_PROFILE",
        )
        p.need(
            co["cgroup"]
            == f"/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service/app.slice/{co['unit']}",
            "EXACT_GEMMA_OWNER_CGROUP",
        )
        self._descriptor, self._value, self._co, self._sha = value, seed, co, selection_sha
        self._directory = p.ordinary(receipt_directory)
        p.need(
            self._directory.is_dir() and not list(self._directory.iterdir()),
            "FRESH_GEMMA_HELPER_RECEIPTS",
        )
        self._sequence, self._lock = 0, threading.Lock()
        self._qualification = self._static()

    @property
    def binding(self):
        return copy.deepcopy(self._value["binding"])

    def _static(self):
        p, value = self.p, self._value
        seed_ref = self._descriptor["helper_seed_ref"]
        p.need(
            p.parse(p.ref_blob(seed_ref, seed_ref["sha256"])) == value,
            "SELECTED_Q95_SEED_UNCHANGED",
        )
        qualification = p.parse(p.ref_blob(value["qualification"], p.Q_SHA))
        p.need(
            qualification["qualified"] is True
            and qualification["task_performance_attested"] is False
            and qualification["kind"] == "jev_public_action_field_syntax_bf16_qualification_v1"
            and qualification["execution_sha256"] == p.E_SHA
            and qualification["runtime_sha256"] == p.R_SHA,
            "SAME_SAVED_Q95_METADATA",
        )
        source = p.parse(p.ref_blob(value["execution"], p.E_SHA))["files"]
        p.need(len(source) == 95, "EXACT_ORIGINAL_Q95_SOURCE95")
        for label, expected in source.items():
            p.need(
                not Path(label).is_absolute()
                and ".." not in Path(label).parts
                and p.pin(Path(value["source_root"]) / label) == expected,
                "EXACT_FROZEN_Q95_SOURCE",
            )
        runtime = p.parse(p.ref_blob(value["runtime"], p.R_SHA))["files"]
        p.need(len(runtime) == 348, "EXACT_ORIGINAL_Q95_RUNTIME348")
        for path, expected in runtime.items():
            p.need(Path(path).is_absolute() and p.pin(path) == expected, "EXACT_Q95_RUNTIME_SOURCE")
        p.ref_blob(self._co["adapter_manifest"], ADAPTER_SHA)
        return qualification

    def _live(self):
        return self.p.LiveHelperChecker._live(self)

    def _auth(self, token, deadline):
        return self.p.LiveHelperChecker._auth(self, token, deadline)

    def _memory(self):
        p = self.p
        host = p.host_memory()
        p.need(
            host["boot_id"] == self.binding["boot_id"]
            and host["MemTotal"] == self._value["expected_memtotal"],
            "SAME_GEMMA_Q95_HOST",
        )
        p.need(p.process(os.getpid())["cgroup"] == self._co["cgroup"], "ACTUAL_GEMMA_OWNER_CGROUP")
        native_groups = p.cgroup_memory(self._co["cgroup"])
        helper_groups = p.cgroup_memory(self._value["worker"]["cgroup"])
        p.need(
            native_groups
            and native_groups[0]["limit"] == 40 * p.GIB
            and (p.CGROOT / self._co["cgroup"].lstrip("/") / "memory.swap.max").read_text().strip()
            == "0",
            "ACTUAL_GEMMA_NATIVE40_SWAP0",
        )
        reserve = max(24 * p.GIB, (host["MemTotal"] + 4) // 5)
        required = reserve + 2 * p.BASE_BYTES
        p.need(host["MemAvailable"] >= required, "ORIGINAL_Q95_AND_NATIVE_NF4_HOST_FLOOR")
        return {
            "host": host,
            "reserve_bytes": reserve,
            "helper": {
                "admitted": True,
                "required_host_available_bytes": required,
                "basis": "Original Q95 host-only R+2BASE",
                "cgroups_recorded_only": helper_groups,
            },
            "co_resident": {
                "admitted": True,
                "profile": NATIVE_PROFILE,
                "required_host_available_bytes": required,
                "basis": "Original NF4 ResourceJournal host R+2BASE plus native40GiB/swap0 owner; no D3 CORE or memcg substitution",
                "cgroups_recorded_only": native_groups,
                "native_forward_gates_remain_external": True,
            },
            "swap_or_resident_credit_bytes": 0,
            "cuda_observed": False,
        }

    def __call__(self, binding, phase):
        return self.p.LiveHelperChecker.__call__(self, binding, phase)

    def text_connection(self):
        return self.p.LiveHelperChecker.text_connection(self)
