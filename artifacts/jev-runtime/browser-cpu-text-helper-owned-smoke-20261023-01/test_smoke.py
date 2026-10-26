"""Stdlib control/identity/snapshot fakes; no NN, owner, payload or runtime tests."""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


a = module("owned_helper_actor_test", "actor.py")
p = module("owned_helper_authority_test", "authority.py")


class Clock:
    value = 100.0

    def __call__(self):
        return self.value


def process(pid, start=7, parent=1, group=None):
    return {
        "pid": pid,
        "start_ticks": start,
        "ppid": parent,
        "uid": 99,
        "state": "S",
        "cgroup": group or "/owned/" + p.UNIT,
    }


def ticket():
    return {
        "schema": "owned-helper-smoke-ticket-v1",
        "profile": p.PROFILE,
        "unit": p.UNIT,
        "seconds": 20.0,
        "boot_id": "independent-fake-boot",
        "worker": process(20, parent=10),
        "parent": process(10),
        "unit_main": process(5),
        "control_fd": 6,
        "secret": "a" * 64,
    }


def watchdog(clock):
    return a.RequestWatchdog("secret", "hash", 280, clock)


def control(seq=1, **fields):
    return {
        "operation": "check",
        "sequence": seq,
        "secret": "secret",
        "ticket_sha256": "hash",
        **fields,
    }


class WatchdogTests(unittest.TestCase):
    def test_original_request_is_armed_once_and_no_control_can_disarm(self):
        clock = Clock()
        w = watchdog(clock)
        w.handle(control(request_id="r", deadline=120.0))
        w.handle(control(2, operation="arm", request_id="r", started=100.0, deadline=120.0))
        self.assertTrue(w.armed)
        for operation in ("cancel", "finish", "disarm"):
            with self.assertRaises(ValueError):
                w.handle(control(3, operation=operation))
        clock.value = 120
        with self.assertRaises(TimeoutError):
            w.check()

    def test_deadline_cannot_extend_or_change_request(self):
        for fields in (
            {"deadline": 121.0, "request_id": "r"},
            {"deadline": 120.0, "request_id": "s"},
        ):
            w = watchdog(Clock())
            w.handle(control(deadline=120.0, request_id="r"))
            with self.assertRaises(ValueError):
                w.handle(control(2, **fields))

    def test_bad_sequence_auth_nonfinite_deadline_and_rearm_refuse(self):
        for fields in (
            {"sequence": 2},
            {"secret": "wrong"},
            {"ticket_sha256": "wrong"},
            {"deadline": float("inf"), "request_id": "r"},
        ):
            with self.assertRaises(ValueError):
                watchdog(Clock()).handle(control(**fields))
        w = watchdog(Clock())
        w.handle(control(operation="arm", request_id="r", started=100.0, deadline=120.0))
        with self.assertRaises(ValueError):
            w.handle(control(2, operation="arm", request_id="r", started=100.0, deadline=120.0))

    def test_whole_budget_also_expires_without_any_arm(self):
        clock = Clock()
        w = watchdog(clock)
        clock.value = 280
        with self.assertRaises(TimeoutError):
            w.check()


class IdentityTests(unittest.TestCase):
    def inspect(self, value):
        return {20: process(20, parent=10), 10: process(10), 5: process(5)}[value]

    def test_real_ticket_shape_requires_live_matching_parent_worker_and_main(self):
        p.check_ticket(
            ticket(), 20, inspect=self.inspect, boot="independent-fake-boot", parent_pid=10
        )
        for change in (
            lambda t: t["worker"].update(start_ticks=8),
            lambda t: t["parent"].update(cgroup="/different"),
            lambda t: t.update(boot_id="wrong"),
            lambda t: t.update(control_fd=None),
        ):
            value = ticket()
            change(value)
            with self.assertRaises(ValueError):
                p.check_ticket(
                    value, 20, inspect=self.inspect, boot="independent-fake-boot", parent_pid=10
                )

    def test_wrong_actual_parent_or_zombie_refuses(self):
        with self.assertRaises(ValueError):
            p.check_ticket(
                ticket(), 20, inspect=self.inspect, boot="independent-fake-boot", parent_pid=11
            )
        with self.assertRaises(ValueError):
            p.check_ticket(
                ticket(),
                20,
                inspect=lambda pid: {**self.inspect(pid), "state": "Z"},
                boot="independent-fake-boot",
                parent_pid=10,
            )

    def test_pidfd_kill_requires_exact_current_identity(self):
        expected = process(20)
        fake = SimpleNamespace(
            identity=lambda _pid: {**expected, "start_ticks": 8}, same_identity=p.same_identity
        )
        with patch.object(a.signal, "pidfd_send_signal") as signal_call:
            with self.assertRaises(ValueError):
                a.stop_exact_worker(fake, expected, 99)
            signal_call.assert_not_called()
        fake.identity = lambda _pid: expected
        with patch.object(a.signal, "pidfd_send_signal") as signal_call:
            a.stop_exact_worker(fake, expected, 99)
            signal_call.assert_called_once_with(99, a.signal.SIGKILL)

    def test_lost_or_refused_external_parent_cannot_authorize(self):
        class Channel:
            def send(self, _value):
                pass

            def receive(self, _timeout):
                return {"ok": False, "sequence": 1}

        authority = p.Authority(ticket(), "hash", Channel(), Clock())
        with patch.object(p, "check_ticket"), self.assertRaises(ValueError):
            authority.check("synthetic-boundary")


class SnapshotTests(unittest.TestCase):
    def package(self, root):
        directory = root / "acquired"
        directory.mkdir()
        records = {}
        for index, name in enumerate(sorted(a.FILES)):
            raw = ("independent tiny bytes " + str(index)).encode()
            (directory / name).write_bytes(raw)
            records[name] = {"size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        (directory / "README.md").write_text("excluded synthetic metadata")
        return directory, records

    def test_exact8_hardlinks_preserve_original_inodes_and_omit_journals(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            acquired, records = self.package(root)
            before = {name: (acquired / name).stat() for name in a.FILES}
            result = a.exact_snapshot(acquired, root / "snapshot", records, lambda: None)
            self.assertEqual(set(result), a.FILES)
            self.assertEqual({p.name for p in (root / "snapshot").iterdir()}, a.FILES)
            for name in a.FILES:
                self.assertEqual((root / "snapshot" / name).stat().st_ino, before[name].st_ino)
                self.assertEqual((acquired / name).stat().st_mode, before[name].st_mode)

    def test_corrupt_same_length_or_symlink_source_refuses(self):
        for symlink in (False, True):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                acquired, records = self.package(root)
                name = sorted(a.FILES)[0]
                if symlink:
                    source = acquired / name
                    source.rename(root / "redirect")
                    source.symlink_to(root / "redirect")
                else:
                    records[name]["sha256"] = "0" * 64
                with self.assertRaises(ValueError):
                    a.exact_snapshot(acquired, root / "snapshot", records, lambda: None)

    def test_incomplete_selection_and_expired_checkpoint_never_qualify(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            acquired, records = self.package(root)
            with self.assertRaises(ValueError):
                a.exact_snapshot(acquired, root / "snapshot", {}, lambda: None)
            with self.assertRaises(TimeoutError):
                a.exact_snapshot(
                    acquired,
                    root / "snapshot",
                    records,
                    lambda: (_ for _ in ()).throw(TimeoutError("synthetic")),
                )
            self.assertEqual(list((root / "snapshot").iterdir()), [])


class FailureTests(unittest.TestCase):
    def test_late_or_cleanup_failure_demotes_success_and_clears_qualification(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "receipt.json").write_text("synthetic candidate")
            state = {"actual_generation_qualified": True}
            primary = TimeoutError("synthetic late cleanup")
            a.failure_receipt(root, state, primary, "post-cleanup")
            self.assertFalse((root / "receipt.json").exists())
            self.assertTrue((root / "failed-post-cleanup-candidate-receipt.json").is_file())
            self.assertFalse(
                json.loads((root / "failure.json").read_text())["actual_generation_qualified"]
            )

    def test_failure_evidence_cannot_replace_primary(self):
        with tempfile.TemporaryDirectory() as temp:
            primary = RuntimeError("original synthetic failure")
            with patch.object(a, "write_new", side_effect=OSError("synthetic evidence failure")):
                a.failure_receipt(Path(temp), {}, primary, "request")
            self.assertIn("Failure evidence also failed", primary.__notes__[0])


class CatalogTests(unittest.TestCase):
    def values(self):
        closure = {
            "closure_mode": "authenticated_stop",
            "primary_error": None,
            "closure_errors": [],
            "two_process_scans": [{"matches": [], "unknown": []}, {"matches": [], "unknown": []}],
            "cgroup_absent": True,
            "whole_unit_exit_confirmed": True,
            "observed_cgroup_cohort_exit_confirmed": True,
            "continuation_authorized": False,
            "selection": {"nonce": "nonce", "boot_id": "boot"},
            "before_stop": {"Description": "nonce", "InvocationID": "invocation"},
            "invocation_id": "invocation",
        }
        names = ("torch", "transformers", "tokenizers", "safetensors", "accelerate")
        sources = [
            {"path": "/synthetic/source" + str(i), "size_bytes": 1, "sha256": "a" * 64}
            for i in range(13)
        ]
        natives = [
            {"path": "/synthetic/native" + str(i), "size_bytes": 1, "sha256": "b" * 64}
            for i in range(9)
        ]
        origins = {name: "/synthetic/" + name + "/__init__.py" for name in names}
        versions = {name: "independent.fake" for name in names}
        interpreter = {
            "path": "/synthetic/python-resolved",
            "stat": {"st_dev": 1, "st_ino": 2, "st_size": 3, "st_mtime_ns": 4},
        }
        observed = {
            "process": {
                "pid": 55,
                "start_ticks": 9,
                "pgid": 50,
                "sid": 50,
                "uid": 99,
                "cgroup": "/synthetic/owned",
            },
            "exe": "/synthetic/python-resolved",
            "argv": ["synthetic-catalog"],
            "cwd": "/synthetic",
        }
        admission = {
            "identity": observed,
            "unit": {"Description": "nonce", "InvocationID": "invocation"},
            "host": {"observation": {"boot_id": "boot"}},
        }
        return {
            "catalog_plan": {
                "runtime_sources": sources,
                "native_files": [{**row, "expected_sha256": None} for row in natives],
                "module_origins": origins,
                "versions": versions,
                "logical_interpreter": "/synthetic/python",
                "interpreter": interpreter,
            },
            "catalog_receipt": {
                "profile": "owned-helper-runtime-native-catalog-v1",
                "status": "catalog_complete_runtime_identity_only",
                "final_authentication": True,
                "completed": True,
                "plan_sha256": a.CATALOG_PLAN_SHA,
                "initial_admission": copy.deepcopy(admission),
                "final_admission": copy.deepcopy(admission),
                "model_loaded": False,
                "helper_qualified": False,
                "full_native_DSO_closure": False,
                "runtime_source_files": copy.deepcopy(sources),
                "runtime_binary_files": copy.deepcopy(natives),
                "modules": {
                    key: {"origin": origins[key], "version": versions[key]} for key in names
                },
                "runtime_python": {
                    "path": "/synthetic/python",
                    "resolved_path": interpreter["path"],
                    "stat": copy.deepcopy(interpreter["stat"]),
                    "version": [3, 13, 14],
                },
            },
            "root_execution": {"exit_code": 0},
            "closure": closure,
            "owner_receipt": {
                "completed": True,
                "primary_error": None,
                "secondary_errors": [],
                "sources_unchanged": True,
                "closure": closure,
                "selected_config": {"nonce": "nonce", "boot_id": "boot"},
                "state": {
                    "actor_exit": {
                        "entry_exit_code": 0,
                        "error": None,
                        "command_results": [
                            {"returncode": 0, "observed_identity": copy.deepcopy(observed)}
                        ],
                    }
                },
            },
        }

    def invoke(self, values):
        accepted = {
            "schema": "owned-helper-runtime-catalog-acceptance-v1",
            "refs": {
                key: {"key": key, "path": str(a.CATALOG_PLAN), "sha256": a.CATALOG_PLAN_SHA}
                for key in values
            },
        }
        with (
            patch.object(a, "read_json", return_value=accepted),
            patch.object(a, "checked_ref", side_effect=lambda row: values[row["key"]]),
        ):
            return a.accepted_catalog("unused", "unused")

    def test_catalog_status_alone_does_not_authorize_unclosed_or_failed_owner(self):
        self.invoke(self.values())
        for change in (
            lambda d: d["root_execution"].update(exit_code=1),
            lambda d: d["owner_receipt"].update(completed=False),
            lambda d: d["closure"].update(cgroup_absent=False),
            lambda d: d["closure"]["two_process_scans"][0].update(unknown=[1]),
            lambda d: d["catalog_receipt"].update(model_loaded=True),
            lambda d: d["catalog_receipt"]["final_admission"]["unit"].update(InvocationID="wrong"),
            lambda d: d["catalog_receipt"]["initial_admission"]["identity"]["process"].update(
                start_ticks=10
            ),
            lambda d: d["catalog_receipt"]["final_admission"]["host"]["observation"].update(
                boot_id="wrong"
            ),
        ):
            value = copy.deepcopy(self.values())
            change(value)
            with self.assertRaises(ValueError):
                self.invoke(value)

    def test_catalog_plan_inventory_origin_hash_and_interpreter_joins_refuse_drift(self):
        for change in (
            lambda d: d["catalog_receipt"].update(plan_sha256="c" * 64),
            lambda d: d["catalog_receipt"]["runtime_source_files"][0].update(sha256="c" * 64),
            lambda d: d["catalog_receipt"]["runtime_binary_files"][0].update(path="/wrong/tree"),
            lambda d: d["catalog_receipt"]["runtime_binary_files"][0].update(sha256=None),
            lambda d: d["catalog_plan"]["native_files"][0].update(expected_sha256="c" * 64),
            lambda d: d["catalog_receipt"]["modules"]["transformers"].update(origin="/wrong/tree"),
            lambda d: d["catalog_receipt"]["modules"]["torch"].update(version="wrong"),
            lambda d: d["catalog_receipt"]["runtime_python"]["stat"].update(st_ino=5),
        ):
            values = self.values()
            change(values)
            with self.assertRaises(ValueError):
                self.invoke(values)

    def test_dry_default_never_loads_source_owner_or_model(self):
        output = io.StringIO()
        with (
            patch.object(sys, "argv", ["actor.py"]),
            patch.object(a, "load_source", side_effect=AssertionError("must stay dry")),
            contextlib.redirect_stdout(output),
        ):
            a.main()
        self.assertEqual(json.loads(output.getvalue())["status"], "source_only_not_ready")
        self.assertNotIn("torch", sys.modules)


if __name__ == "__main__":
    unittest.main()
