"""Authored stdlib fakes only; this source has not been executed by its author."""

import copy
import importlib.util
import io
import signal
import types
import unittest
from pathlib import Path
from unittest import mock

PATH = Path(__file__).with_name("pump_held.py")
SPEC = importlib.util.spec_from_file_location("_held_v7_pump_source_fakes", PATH)
pump = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pump)


def source_ref(path):
    path = Path(path)
    return {"path": path.relative_to(pump.ROOT).as_posix(), "sha256": "a" * 64, "size_bytes": 1}


def stable(row):
    return {key: row[key] for key in ("pid", "start_ticks", "pgid", "sid", "uid", "cgroup")}


def same_identity(left, right):
    return stable(left["process"]) == stable(right["process"]) and all(
        left[key] == right[key] for key in ("exe", "argv", "cwd")
    )


def proof():
    """Invented metadata exercises original-shaped joins, never actual evidence."""
    root = pump.ROOT / "artifacts/jev-runtime/pump-invented-only"
    output = pump.OWNER / "actual-99"
    controller = types.SimpleNamespace(
        FIXED={
            name: (f"artifacts/pump-invented-{name}.py", "a" * 64)
            for name in ("boundary", "observer", "authority")
        },
        ref=source_ref,
        stable_process=stable,
        same_identity=same_identity,
        sources_match=mock.Mock(),
        command_selection=mock.Mock(),
    )
    process = {
        "pid": 31,
        "start_ticks": 9,
        "pgid": 30,
        "sid": 30,
        "uid": 1001,
        "cgroup": "/invented-cpu-cgroup",
        "state": "R",
    }
    worker_identity = {
        "process": process,
        "exe": "/invented/python",
        "argv": ["/invented/python", "--output", str(output)],
        "cwd": str(pump.ROOT),
    }
    data = {}

    def keep(name, value, path=None):
        path = path or root / name
        raw = pump.encoded(value)
        data[str(path)] = raw
        return {"path": str(path), "sha256": pump.sha(raw), "size_bytes": len(raw)}

    service = {
        "case_index": 1,
        "case_id": pump.CASE_IDS[0],
        "root_plan_sha256": "b" * 64,
        "owner_output": str(output),
        "handoff": {"nonce": "c" * 32},
    }
    service_ref = keep("service-plan.json", service)
    spec = {
        "runtime_seconds": 7200,
        "sources": [],
        "commands": [{"argv": ["/invented/python", "--output", "{OUTPUT}"], "cwd": str(pump.ROOT)}],
    }
    spec_ref = keep("command-spec.json", spec)
    base = [
        controller.ref(pump.CONTROLLER),
        controller.ref(spec_ref["path"]),
        *(controller.ref(pump.ROOT / label) for label, _ in controller.FIXED.values()),
    ]
    cfg = {
        "profile": pump.CPU_PROFILE,
        "unit": pump.CPU_UNIT,
        "output": str(output),
        "selection": spec_ref["path"],
        "selection_sha256": spec_ref["sha256"],
        "runtime_seconds": 7200,
        "boot_id": "invented-boot",
        "nonce": "jev-d3-boundary-" + "d" * 32,
        "sources": base,
        "base_sources": base,
    }
    cfg_ref = keep("sealed-config.json", cfg)
    actor = {**process, "pid": 30, "start_ticks": 8, "state": "S"}
    closure = {
        "selection": {
            "unit": pump.CPU_UNIT,
            "nonce": cfg["nonce"],
            "boot_id": cfg["boot_id"],
            "memory_max_bytes": 4 * 1024**3,
            "runtime_seconds": 7200,
            "stop_seconds": 3,
        },
        "actor": actor,
        "invocation_id": "e" * 32,
        "closure_mode": "authenticated_stop",
        "closure_errors": [],
        "before_stop": {"Description": cfg["nonce"], "InvocationID": "e" * 32, "MainPID": "30"},
        "whole_unit_exit_confirmed": True,
        "observed_cgroup_cohort_exit_confirmed": True,
        "post_stop_terminal_verified": True,
        "cgroup_absent": True,
        "sources_unchanged": True,
        "two_process_scans": [{"matches": [], "unknown": []} for _ in range(2)],
        "no_escape_enforced": False,
        "whole_tree_exit_confirmed": False,
        "continuation_authorized": False,
        "model_qualified": False,
    }
    # Original snapshots may observe S and R for the same stable process.
    command_identity = copy.deepcopy(worker_identity)
    command_identity["process"]["state"] = "S"
    receipt = {
        "profile": pump.CPU_PROFILE,
        "selected_config": cfg,
        "completed": True,
        "primary_error": None,
        "secondary_errors": [],
        "closure": closure,
        "sources_unchanged": True,
        "state": {
            "actor_exit": {
                "command_results": [
                    {
                        "returncode": 0,
                        "argv": worker_identity["argv"],
                        "observed_identity": command_identity,
                    }
                ]
            }
        },
    }
    record = {
        "root_plan_sha256": "b" * 64,
        "case_index": 1,
        "case_id": pump.CASE_IDS[0],
        "plan_sha256": service_ref["sha256"],
        "nonce": "c" * 32,
        "owner_output": str(output),
        "boot_id": cfg["boot_id"],
        "command_spec_ref": spec_ref,
        "service_plan_ref": service_ref,
        "sealed_config_ref": cfg_ref,
        "controller_exit_code": 0,
        "controller_identity": {"pid": 21, "start_ticks": 5},
    }
    record["closure_ref"] = keep("closure.json", closure)
    record["owner_receipt_ref"] = keep("receipt.json", receipt)
    record["owner_terminal_ref"] = keep(
        "controller-terminal.json",
        {
            "schema": "jev-v7-matched-controller-terminal-v1",
            "argv": pump.controller_argv(spec_ref, output),
            "exit_code": 0,
            "controller_identity": record["controller_identity"],
            "root_plan_sha256": record["root_plan_sha256"],
            "case_index": 1,
            "case_id": record["case_id"],
        },
    )
    record["admission_ref"] = keep(
        "admission.json",
        {
            "config_sha256": cfg_ref["sha256"],
            "nonce": cfg["nonce"],
            "invocation_id": closure["invocation_id"],
            "actor": {**actor, "state": "R"},
        },
    )
    common = pump.binding(record, record["root_plan_sha256"], record["plan_sha256"])
    worker = {
        "schema": "jev-v7-full-choice-service-receipt-v1",
        **common,
        "owner_output": str(output),
        "service_success": True,
        "server_thread_drained": True,
        "final_native_artifact_authentication": True,
        "sources_unchanged": True,
        "cleanup_errors": [],
        "pending_native_or_server": False,
        "helper_model_loaded": False,
        "helper_model_calls": 0,
        "worker": worker_identity,
    }
    record["worker_receipt_ref"] = keep(
        "worker.json", worker, output / "v7-full-choice-service/worker-receipt.json"
    )
    record["service_closed_ref"] = keep(
        "service-closed.json",
        {
            "schema": "jev-v7-full-choice-service-closed-v1",
            **common,
            "owner_output": str(output),
            "service_success": True,
            "server_thread_drained": True,
            "final_native_artifact_authentication": True,
            "sources_unchanged": True,
            "worker_receipt": record["worker_receipt_ref"]["path"],
            "external_current_owner_closure_required": True,
            "whole_tree_exit_confirmed": False,
            "task_success": None,
        },
    )
    return controller, record, data


class HeldPumpFakes(unittest.TestCase):
    def test_template_accepts_exact_real_worker_two_field_serial_source(self):
        root = pump.ROOT / "artifacts/jev-runtime/invented-pump-protocol"
        cap = {"path": str(root / "cap.json"), "sha256": "a" * 64, "size_bytes": 1}
        numeric = {"path": str(root / "numeric.json"), "sha256": "b" * 64, "size_bytes": 1}
        serial = {"path": str(root / "serial.py"), "sha256": "c" * 64, "size_bytes": 1}
        template = {
            "schema": "jev-v7-matched-cpu-service-template-v1",
            "status": "ROOT_SELECTED_TEMPLATE_ONLY",
            "cases": [
                {
                    "case_index": index,
                    "case_id": case_id,
                    "group": "known4" if index <= 4 else "public6",
                    "owner_output": str(pump.OWNER / f"actual-{80 + index:02d}"),
                    "nonce": str(index).zfill(32),
                }
                for index, case_id in enumerate(pump.CASE_IDS, 1)
            ],
            "command_spec": {
                "profile": "llama-cpp-command-spec-v1",
                "runtime_seconds": 7200,
                "file_size_max_bytes": 256 * 1024**2,
                "commands": [{}],
                "sources": [],
            },
            "service_plan": {
                "cooperative_budget_seconds": 1200,
                "owner_hard_runtime_seconds": 7200,
                "candidate_capability": cap,
                "numeric_acceptance": numeric,
                "serial_forward": {key: serial[key] for key in ("path", "sha256")},
            },
            "candidate_capability_ref": cap,
            "numeric_acceptance_ref": numeric,
        }
        plan = {
            "matched": {
                "serial_source": serial,
                "capability_sha256": cap["sha256"],
                "candidate_alias": pump.ALIAS,
            }
        }
        fake_cap = {
            "schema": "root-accepted-dsl01-operation-train-delta-v1",
            "status": "ROOT_ACCEPTED_OPERATION_TRAIN_ONLY_READOUT",
            "alias": pump.ALIAS,
            "only_head": "operation",
        }
        fake_numeric = {
            "schema": "root-accepted-operation-cpu5-numeric-v1",
            "status": "accepted",
            "candidate_capability": cap,
            "candidate_alias": pump.ALIAS,
        }
        with mock.patch.object(
            pump,
            "authenticate",
            side_effect=lambda row: pump.encoded(
                fake_cap if row["path"] == cap["path"] else fake_numeric
            ),
        ):
            self.assertEqual(pump.validate_template(template, plan), template["cases"])
            template["service_plan"]["serial_forward"] = dict(serial)
            with self.assertRaisesRegex(ValueError, "two-field serial"):
                pump.validate_template(template, plan)

    def test_entry_holds_before_io_or_process(self):
        with (
            mock.patch.object(pump, "SOURCE_ONLY", True),
            mock.patch.object(pump, "ordinary", side_effect=AssertionError("IO")),
            mock.patch.object(pump, "read", side_effect=AssertionError("read")),
            mock.patch.object(pump.subprocess, "Popen", side_effect=AssertionError("Popen")),
        ):
            for call in (
                lambda: pump.main(["--run"]),
                lambda: pump.Pump({}, "/bad", None),
                lambda: pump.verify_children({}, {}, {}, "/bad"),
            ):
                with self.assertRaisesRegex(ValueError, "SOURCE_ONLY"):
                    call()

    def test_template_pin_is_second_hold(self):
        with (
            mock.patch.object(pump, "SOURCE_ONLY", False),
            mock.patch.object(pump, "TEMPLATE_SHA256", None),
            mock.patch.object(pump, "ordinary", side_effect=AssertionError("IO")),
        ):
            with self.assertRaisesRegex(ValueError, "pin remains pending"):
                pump.Pump({}, "/bad", None)

    def test_pure_runtime_expansion_preserves_original_output_token_and_limits(self):
        target = pump.ROOT / "artifacts/jev-runtime/invented-pump-campaign"
        serial_ref = {
            "path": str(target / "serialization.json"),
            "sha256": "a" * 64,
            "size_bytes": 10,
        }
        template = {
            "service_plan": {
                "cooperative_budget_seconds": 1200,
                "owner_hard_runtime_seconds": 7200,
            },
            "command_spec": {
                "runtime_seconds": 7200,
                "sources": [],
                "commands": [
                    {
                        "argv": [
                            "python",
                            "--plan",
                            "{SERVICE_PLAN}",
                            "--plan-sha256",
                            "{SERVICE_PLAN_SHA256}",
                            "--owner-output",
                            "{OUTPUT}",
                        ]
                    }
                ],
            },
        }
        case = {
            "case_index": 2,
            "case_id": pump.CASE_IDS[1],
            "nonce": "b" * 32,
            "owner_output": str(pump.OWNER / "actual-99"),
        }
        with mock.patch.object(pump, "ordinary", side_effect=lambda path: Path(path)):
            service, selected, spec = pump.compile_case(
                template, case, "c" * 64, target, {"nonce": "d" * 32}, serial_ref, {"pid": 7}
            )
        self.assertEqual(service["root_plan_sha256"], "c" * 64)
        self.assertEqual(
            service["handoff"]["ready_path"], str(target / "handoff/case-02/service-ready.json")
        )
        self.assertEqual(service["serialization_ref"], serial_ref)
        self.assertEqual(spec["commands"][0]["argv"][2], selected["path"])
        self.assertEqual(spec["commands"][0]["argv"][4], selected["sha256"])
        self.assertEqual(spec["commands"][0]["argv"][-1], "{OUTPUT}")
        self.assertEqual(spec["runtime_seconds"], 7200)
        self.assertEqual(len(spec["sources"]), 2)
        self.assertNotIn("case_index", template["service_plan"])

    def test_stop_has_exact_service_schema_and_distinct_nonce(self):
        case = {"case_index": 4, "case_id": pump.CASE_IDS[3], "nonce": "b" * 32}
        row = pump.service_stop(case, "a" * 64, "c" * 64)
        self.assertEqual(
            row,
            {
                "schema": "jev-v7-full-choice-service-stop-v1",
                "case_index": 4,
                "case_id": pump.CASE_IDS[3],
                "nonce": "b" * 32,
                "root_plan_sha256": "a" * 64,
                "plan_sha256": "c" * 64,
            },
        )

    def test_original_shaped_new_service_receipt_and_stable_identity_pass(self):
        controller, record, data = proof()
        with (
            mock.patch.object(pump, "authenticate", side_effect=lambda row: data[row["path"]]),
            mock.patch.object(pump, "ordinary", side_effect=lambda path: Path(path)),
        ):
            self.assertTrue(pump.verify_original(record, controller, complete=True))

    def test_old_toy_receipt_keys_are_not_an_accepted_fallback(self):
        controller, record, data = proof()
        path = record["worker_receipt_ref"]["path"]
        worker = pump.decode(data[path])
        worker["success"] = worker.pop("service_success")
        worker["final_native_and_artifact_authentication"] = worker.pop(
            "final_native_artifact_authentication"
        )
        data[path] = pump.encoded(worker)
        with (
            mock.patch.object(pump, "authenticate", side_effect=lambda row: data[row["path"]]),
            mock.patch.object(pump, "ordinary", side_effect=lambda path: Path(path)),
        ):
            with self.assertRaises(KeyError):
                pump.verify_original(record, controller, complete=True)

    def test_original_closure_and_service_join_rejections(self):
        changes = (
            ("closure_ref", lambda row: row.update(closure_errors=["invented failure"])),
            ("closure_ref", lambda row: row["two_process_scans"][1].update(unknown=[17])),
            ("closure_ref", lambda row: row["selection"].update(memory_max_bytes=5 * 1024**3)),
            ("service_closed_ref", lambda row: row.update(nonce="0" * 32)),
            ("worker_receipt_ref", lambda row: row.update(service_success=False)),
            ("admission_ref", lambda row: row.update(invocation_id="0" * 32)),
        )
        for key, mutate in changes:
            with self.subTest(key=key):
                controller, record, data = proof()
                path = record[key]["path"]
                changed = pump.decode(data[path])
                mutate(changed)
                data[path] = pump.encoded(changed)
                with (
                    mock.patch.object(
                        pump, "authenticate", side_effect=lambda row: data[row["path"]]
                    ),
                    mock.patch.object(pump, "ordinary", side_effect=lambda path: Path(path)),
                ):
                    with self.assertRaises(ValueError):
                        pump.verify_original(record, controller, complete=True)

    def test_close_still_interrupts_original_controller_after_stop_write_failure(self):
        subject = object.__new__(pump.Pump)
        subject.closed, subject.failure, subject.records = False, None, []
        original = OSError("invented stop durability error")
        process = mock.Mock()
        process.poll.side_effect = [None, None, 1, 1]
        identity = {"pid": 20, "start_ticks": 7, "pgid": 20, "sid": 20, "cgroup": "root"}
        active = {"process": process, "identity": identity, "stream": io.BytesIO()}
        subject.active = active
        process.pid = 20
        subject._stop = mock.Mock(side_effect=original)

        def finish(_active, *, complete):
            self.assertFalse(complete)
            subject.records.append(
                {
                    "original_scoped_closure_positive": True,
                    "owner_completed": False,
                    "controller_exit_code": 1,
                    "interrupted_for_cleanup": True,
                }
            )
            subject.active = None

        subject._finish = finish
        with (
            mock.patch.object(pump, "released"),
            mock.patch.object(pump, "identity", return_value=identity),
            mock.patch.object(pump.time, "sleep"),
        ):
            with self.assertRaises(OSError) as found:
                subject.close()
        self.assertIs(found.exception, original)
        process.send_signal.assert_called_once_with(signal.SIGINT)
        process.kill.assert_not_called()
        self.assertTrue(original.jev_v7_pump_closure["active_controller_absent"])
        self.assertFalse(original.jev_v7_pump_closure["complete_ten_cases"])

    def test_no_live_controller_can_be_reported_closed(self):
        subject = object.__new__(pump.Pump)
        subject.closed, subject.failure, subject.records = False, None, []
        process = mock.Mock()
        process.pid = 20
        process.poll.return_value = None
        identity = {"pid": 20, "start_ticks": 7, "pgid": 20, "sid": 20, "cgroup": "root"}
        subject.active = {"process": process, "identity": identity, "stream": io.BytesIO()}
        subject._stop = mock.Mock()
        subject._finish = mock.Mock()
        with (
            mock.patch.object(pump, "released"),
            mock.patch.object(pump, "identity", return_value=identity),
            mock.patch.object(pump.time, "monotonic", side_effect=[0, 46]),
            mock.patch.object(pump.time, "sleep"),
        ):
            with self.assertRaisesRegex(ValueError, "remains live"):
                subject.close()
        self.assertFalse(subject.closed)
        subject._finish.assert_not_called()
        process.kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
