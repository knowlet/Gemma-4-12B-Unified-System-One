"""Unexecuted stdlib fakes for the held matched outer ownership boundary.

All files/identities here are invented temporary fixtures. No real systemd,
ComfyUI, cgroup, cache, model, child owner, browser or request is used.
"""

import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def source(filename):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location("_matched_fake_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def host_row(*, busy=False):
    return {
        "boot_id": "invented-boot",
        "mem_total_bytes": 130_595_864_576,
        "mem_available_bytes": 125_000_000_000,
        "comfy_running": int(busy),
        "comfy_pending": 0,
        "comfy_state": "http_busy" if busy else "http_idle",
    }


class OwnerSourceTests(unittest.TestCase):
    def setUp(self):
        self.owner = source("owner_held.py")
        self.entry = source("campaign_entry_held.py")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "repo"
        self.here = self.root / "artifacts/jev-runtime/invented-matched-source"
        self.here.mkdir(parents=True)
        for name, value in (
            ("ROOT", self.root),
            ("HERE", self.here),
            ("CGROUP_ROOT", Path(self.temporary.name) / "cgroups"),
        ):
            self.addCleanup(patch.stopall)
            patch.object(self.owner, name, value).start()
        self.plan = {
            "arm": "matched",
            "attempt": "01",
            "unit": self.owner.UNIT,
            "run_directory": (self.here / "matched-01").relative_to(self.root).as_posix(),
            "output": (self.here / "matched-01/campaign").relative_to(self.root).as_posix(),
            "cgroup": "/invented/matched-unit",
            "boot_id": "invented-boot",
            "nonce": "jev-nf4-d3-matched-" + "a" * 32,
            "prompt_format": "shared-elements-v2",
            "sources": [{"path": "invented-source", "sha256": "a" * 64, "size_bytes": 1}],
        }

    def test_source_holds_precede_parser_source_plan_and_launch(self):
        for module in (self.owner, self.entry):
            with (
                patch.object(module.argparse, "ArgumentParser") as parser,
                patch.object(Path, "read_bytes", side_effect=AssertionError("IO")),
            ):
                with self.assertRaisesRegex(ValueError, "SOURCE_ONLY"):
                    module.main(["--run"])
                parser.assert_not_called()
        with patch.object(self.owner, "verify_cache_aliases") as aliases:
            with self.assertRaisesRegex(ValueError, "SOURCE_ONLY"):
                self.owner.execute(None, None, None, None, plan_blob=None)
            aliases.assert_not_called()
        with patch.object(self.entry, "OWNER_SOURCE_SHA256", "a" * 64):
            with self.assertRaisesRegex(ValueError, "SOURCE_ONLY"):
                self.entry.load_owner()

    def test_changing_only_source_hold_cannot_bypass_pending_pins_or_child_contract(self):
        with patch.object(self.owner, "SOURCE_ONLY", False):
            with self.assertRaisesRegex(ValueError, "pins remain pending"):
                self.owner.main([])
            with (
                patch.object(self.owner, "CAMPAIGN_SOURCE_SHA256", "a" * 64),
                patch.object(self.owner, "PUMP_SOURCE_SHA256", "b" * 64),
            ):
                with self.assertRaisesRegex(ValueError, "closure contract remains pending"):
                    self.owner.main([])
        with patch.object(self.entry, "SOURCE_ONLY", False):
            with self.assertRaisesRegex(ValueError, "pin remains pending"):
                self.entry.main([])

    def test_new_time_profile_does_not_relabel_old7200_and_preserves_original_cap(self):
        self.assertEqual(
            (self.owner.OUTER_SECONDS, self.owner.INNER_SECONDS, self.owner.OWNER_STOP_SECONDS),
            (10800, 10740, 10780),
        )
        self.assertEqual(self.owner.CAP, 40 * 1024**3)
        self.assertEqual(self.owner.BASE_BYTES, 23_919_549_408)
        self.assertEqual(
            self.owner.EXPECTED_BASE_CHECKPOINT["sha256"],
            "11144621122123135eb79c9cfd0466a2c51f230109124aefe401cdbec8c71a53",
        )
        self.assertEqual(self.owner.reserve(130_595_864_576), 26_119_172_916)
        self.assertEqual(self.owner.PROFILE, "jev-nf4-d3-matched-v7-owned-v1")
        self.assertEqual(self.owner.UNIT, "jev-nf4-d3-matched-v7-20261024-01.service")

    def test_fixed_fresh_path_and_campaign_arguments(self):
        directory, output = self.owner.run_paths(self.plan)
        self.assertEqual(output, directory / "campaign")
        self.assertEqual(
            self.owner.campaign_arguments(self.plan, "d" * 64),
            ["--run", "--plan", str(directory / "selected-plan.json"), "--plan-sha256", "d" * 64],
        )
        changed = {**self.plan, "output": "foreign-output"}
        with self.assertRaisesRegex(ValueError, "Campaign output"):
            self.owner.run_paths(changed)

    def test_pending_pump_and_child_closure_cannot_accept_aggregate_booleans(self):
        with self.assertRaisesRegex(ValueError, "PENDING"):
            self.owner.candidate_pump(self.plan, self.here, object())
        with self.assertRaisesRegex(ValueError, "PENDING"):
            self.owner.verify_candidate_children(
                {"candidate_pump_closure": True}, {"children": True}, self.plan, self.here
            )

    def test_readback_requires_new3h_and_original40g_swap0_properties(self):
        group = self.owner.CGROUP_ROOT / self.plan["cgroup"].lstrip("/")
        group.mkdir(parents=True)
        (group / "memory.max").write_text(str(self.owner.CAP))
        (group / "memory.swap.max").write_text("0")
        properties = {
            "Type": "exec",
            "MemoryAccounting": "yes",
            "MemoryMax": str(self.owner.CAP),
            "MemoryHigh": "infinity",
            "MemorySwapMax": "0",
            "OOMPolicy": "kill",
            "KillMode": "control-group",
            "RuntimeMaxUSec": "3h",
            "KillSignal": "2",
            "TimeoutStopUSec": "20s",
            "RemainAfterExit": "yes",
            "InvocationID": "b" * 32,
        }
        for change in (None, "old-duration", "wrong-cap", "swap"):
            actual = dict(properties)
            if change == "old-duration":
                actual["RuntimeMaxUSec"] = "2h"
            elif change == "wrong-cap":
                actual["MemoryMax"] = "1"
            elif change == "swap":
                actual["MemorySwapMax"] = "1"
            raw = "\n".join(key + "=" + value for key, value in actual.items())
            with (
                self.subTest(change=change),
                patch.object(self.owner, "command", return_value=raw),
                patch.object(self.owner, "save"),
            ):
                if change is None:
                    self.owner.readback(self.plan, "b" * 32, self.here)
                else:
                    with self.assertRaisesRegex(ValueError, "bounds differ"):
                        self.owner.readback(self.plan, "b" * 32, self.here)

    def runtime(self, *, busy=False, monitor_error=None, pump_close_error=None):
        owner, plan = self.owner, self.plan
        directory, output = owner.run_paths(plan)
        events, state = [], {"started": False, "stopped": False, "status": 0, "hosts": 0}
        invocation = "b" * 32
        actor = {"pid": 77, "pgid": 77, "sid": 77, "start_ticks": 99}
        blob = json.dumps(plan, sort_keys=True).encode()
        digest = hashlib.sha256(blob).hexdigest()
        records = [
            {
                "case_id": case,
                "group": "known4" if index < 4 else "public6",
                "task_success": False,
                "infrastructure_complete": True,
            }
            for index, case in enumerate(owner.KNOWN4 + owner.PUBLIC6)
        ]
        arms = [
            {
                "arm": name,
                "planned_denominator": 10,
                "attempted_cases": 10,
                "known4_attempts": 4,
                "public6_attempts": 6,
                "records": copy.deepcopy(records),
                "infrastructure_complete": True,
                "successes": 0,
            }
            for name in ("baseline", "candidate")
        ]
        receipt = {
            "schema": owner.CAMPAIGN_SCHEMA,
            "stage": "complete_pending_external_closure",
            "arms": arms,
        }
        native = {
            "stage": "complete_pending_external_closure",
            "attempts": 10,
            "planned_denominator": 10,
            "prompt_format": "shared-elements-v2",
            "successes": 0,
            "groups": [
                {
                    "group": name,
                    "denominator": len(ids),
                    "summary": {
                        "attempts": len(ids),
                        "outcomes": [{"task_id": case} for case in ids],
                    },
                }
                for name, ids in (("known4", owner.KNOWN4), ("public6", owner.PUBLIC6))
            ],
        }

        def command(*args, **_kwargs):
            if args[0] == "systemd-run":
                events.append("launch-outer")
                self.assertIn("--property=RuntimeMaxSec=10800", args)
                state["started"] = True
                owner.save(
                    directory,
                    "actor-intent.json",
                    {
                        "actor": actor,
                        "unit": plan["unit"],
                        "nonce": plan["nonce"],
                        "plan_sha256": digest,
                    },
                )
                (output / "native").mkdir(parents=True)
                (output / "receipt.json").write_text(json.dumps(receipt))
                (output / "native/receipt.json").write_text(json.dumps(native))
                (output / "native/output-index.json").write_text("{}")
                return "synthetic launch; no process"
            self.assertEqual(args, ("systemctl", "--user", "stop", plan["unit"]))
            events.append("stop-outer")
            state["stopped"] = True
            return ""

        def status(_unit):
            self.assertEqual(_unit, plan["unit"])
            if not state["started"]:
                return {"LoadState": "not-found"}
            state["status"] += 1
            running = not state["stopped"] and (state["status"] == 1 or busy or monitor_error)
            return {
                "InvocationID": invocation,
                "ControlGroup": plan["cgroup"],
                "SubState": "dead" if state["stopped"] else "running" if running else "exited",
                "MainPID": "77" if running else "0",
                "ExecMainPID": "77",
                "Result": "success",
                "ExecMainCode": "1",
                "ExecMainStatus": "0",
            }

        def observed(**_kwargs):
            state["hosts"] += 1
            if monitor_error is not None and state["hosts"] == 2:
                raise monitor_error
            return host_row(busy=busy and state["hosts"] >= 2)

        def observe(callback):
            try:
                return callback()
            except BaseException as error:
                return {"observation_error": owner.error_record(error)}

        def close():
            events.append("close-candidate-pump")
            if pump_close_error is not None:
                raise pump_close_error
            return {"source_fake_only": True}

        pump = SimpleNamespace(poll=lambda: events.append("poll-candidate-pump"), close=close)
        guard = SimpleNamespace(status=status, observe=observe)
        aliases = [{"invented": "cache closure"}]
        patches = (
            patch.object(owner, "released"),
            patch.object(owner, "verify_cache_aliases", return_value=aliases),
            patch.object(owner, "load_observer", return_value=SimpleNamespace(observe=observed)),
            patch.object(owner, "memtotal", return_value=130_595_864_576),
            patch.object(owner, "current_boot", return_value="invented-boot"),
            patch.object(owner, "command", side_effect=command),
            patch.object(owner, "description", return_value=plan["nonce"]),
            patch.object(owner, "readback"),
            patch.object(owner, "candidate_pump", return_value=pump),
            patch.object(owner, "scan", return_value={"matches": [], "unreadable": []}),
            patch.object(
                owner,
                "pin",
                side_effect=lambda path: next(
                    ref for ref in plan["sources"] if self.root / ref["path"] == path
                ),
            ),
            patch.object(owner.time, "sleep"),
        )
        return SimpleNamespace(
            plan=plan,
            blob=blob,
            digest=digest,
            guard=guard,
            events=events,
            state=state,
            directory=directory,
            output=output,
            receipt=receipt,
            patches=patches,
        )

    @contextlib.contextmanager
    def patched_runtime(self, runtime):
        with contextlib.ExitStack() as stack:
            for mocker in runtime.patches:
                stack.enter_context(mocker)
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            yield

    def test_outer_fake_waits_for_pump_close_before_stopping_only_its_gpu_unit(self):
        fake = self.runtime()
        with self.patched_runtime(fake):
            # Focused outer-only fake; this stub is not real child acceptance.
            with patch.object(self.owner, "verify_candidate_children"):
                self.assertEqual(
                    self.owner.execute(fake.plan, fake.digest, fake.guard, {}, plan_blob=fake.blob),
                    0,
                )
        self.assertLess(fake.events.index("close-candidate-pump"), fake.events.index("stop-outer"))
        result = json.loads((fake.directory / "root-external-closure.json").read_bytes())
        self.assertFalse(result["outer_memcg_covers_candidate_siblings"])
        self.assertFalse(result["candidate_child_closure_verified_by_outer_stop"])
        self.assertEqual(len(result["two_proc_scans"]), 2)

    def test_even_complete_all_false20_results_cannot_bypass_pending_child_acceptance(self):
        fake = self.runtime()
        with self.patched_runtime(fake):
            with self.assertRaisesRegex(ValueError, "PENDING"):
                self.owner.execute(fake.plan, fake.digest, fake.guard, {}, plan_blob=fake.blob)
        self.assertTrue(fake.state["stopped"])

    def test_comfy_busy_stops_pump_then_only_outer_and_refuses_success(self):
        fake = self.runtime(busy=True)
        with self.patched_runtime(fake):
            with self.assertRaisesRegex(ValueError, "terminal"):
                self.owner.execute(fake.plan, fake.digest, fake.guard, {}, plan_blob=fake.blob)
        self.assertLess(fake.events.index("close-candidate-pump"), fake.events.index("stop-outer"))
        result = json.loads((fake.directory / "root-external-closure.json").read_bytes())
        self.assertEqual(result["reason"], "Comfy_became_busy_stop_own_unit_only")

    def test_monitor_primary_survives_pump_close_failure_and_outer_still_stops(self):
        primary = KeyboardInterrupt("invented monitor abort")
        secondary = OSError("invented CPU closure failed")
        fake = self.runtime(monitor_error=primary, pump_close_error=secondary)
        with self.patched_runtime(fake):
            with self.assertRaises(KeyboardInterrupt) as raised:
                self.owner.execute(fake.plan, fake.digest, fake.guard, {}, plan_blob=fake.blob)
        self.assertIs(raised.exception, primary)
        self.assertTrue(fake.state["stopped"])
        self.assertTrue(any("Candidate pump closure" in note for note in primary.__notes__))

    def test_missing_or_partial_arm_records_refuse_before_child_acceptance(self):
        fake = self.runtime()
        with self.patched_runtime(fake):
            with patch.object(self.owner, "verify_candidate_children"):
                self.owner.execute(fake.plan, fake.digest, fake.guard, {}, plan_blob=fake.blob)
            result = json.loads((fake.directory / "root-external-closure.json").read_bytes())
            for mutation in ("partial", "duplicate", "wrong-group", "unknown-task", "infra"):
                receipt = copy.deepcopy(fake.receipt)
                records = receipt["arms"][1]["records"]
                if mutation == "partial":
                    records.pop()
                elif mutation == "duplicate":
                    records[1]["case_id"] = records[0]["case_id"]
                elif mutation == "wrong-group":
                    records[0]["group"] = "public6"
                elif mutation == "unknown-task":
                    records[0]["task_success"] = None
                else:
                    records[0]["infrastructure_complete"] = False
                with (
                    self.subTest(mutation=mutation),
                    patch.object(self.owner, "verify_candidate_children") as children,
                ):
                    with self.assertRaises(ValueError):
                        self.owner.verify_closed(result, receipt, fake.plan, fake.output)
                    children.assert_not_called()


if __name__ == "__main__":
    unittest.main()
