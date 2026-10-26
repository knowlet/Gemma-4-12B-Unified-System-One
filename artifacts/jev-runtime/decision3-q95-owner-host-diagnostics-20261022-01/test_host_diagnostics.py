"""Pure fake-observer tests; no processes, host probes, HTTP or model runtime."""

import ast
import hashlib
import importlib.util
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).absolute().parent
PARENT = HERE.parent / "decision3-q95-owner-20261022-04/q95_owner.py"
PARENT_SHA = "68e44e99bfbb1c9114a5b5b378cac38f80b78c37d23f8e898b290c23e3a0a4a6"


def checked_module(path, name, digest=None):
    data = path.read_bytes()
    if digest is not None:
        assert hashlib.sha256(data).hexdigest() == digest
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    exec(compile(data, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


M = checked_module(HERE / "q95_owner.py", "_q95_diagnostics_test")
OLD = checked_module(PARENT, "_q95_diagnostics_parent", PARENT_SHA)


class HostDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.plan = {"boot_id": "fixture-boot", "mem_total_bytes": 130595864576}
        self.reserve = max(24 * M.GIB, (self.plan["mem_total_bytes"] + 4) // 5)
        self.row = {
            "boot_id": "fixture-boot",
            "mem_total_bytes": self.plan["mem_total_bytes"],
            "mem_available_bytes": self.plan["mem_total_bytes"],
            "comfy_state": "http_idle",
            "comfy_running": 0,
            "comfy_pending": 0,
        }

    def observer(self, row):
        calls = []

        def observe(**kwargs):
            calls.append(kwargs)
            return row

        return SimpleNamespace(observe=observe), calls

    def refuse(self, row, *, initial=False):
        observer, calls = self.observer(row)
        with patch.object(M.os, "write") as output:
            with self.assertRaises(ValueError) as raised:
                M.host(self.plan, observer, initial=initial)
        self.assertEqual(calls, [{"expected_boot_id": "fixture-boot"}])
        self.assertEqual(output.call_count, 1)
        fd, blob = output.call_args.args
        self.assertEqual(fd, 2)
        self.assertLessEqual(len(blob), 4096)
        self.assertEqual(blob.count(b"\n"), 1)
        self.assertEqual(json.loads(blob), {"host_gate_refusal": raised.exception.evidence})
        return raised.exception

    def test_success_returns_same_row_without_diagnostic_and_observes_once(self):
        for state in ("http_idle", "absent_confirmed"):
            for initial in (False, True):
                with self.subTest(state=state, initial=initial):
                    row = {**self.row, "comfy_state": state}
                    observer, calls = self.observer(row)
                    with patch.object(M.os, "write") as output:
                        self.assertIs(M.host(self.plan, observer, initial=initial), row)
                    self.assertEqual(len(calls), 1)
                    output.assert_not_called()

    def test_busy_refusal_keeps_typed_counters_memory_and_original_message(self):
        row = {
            **self.row,
            "comfy_state": "http_busy",
            "comfy_running": 1,
            "mem_available_bytes": 37384560640,
        }
        error = self.refuse(row)
        self.assertEqual(str(error), "Explicit same-host idle/absent observation")
        evidence = error.evidence
        self.assertEqual(evidence["gate"], "host_identity_and_comfy_idle")
        self.assertEqual(evidence["phase"], "runtime")
        self.assertEqual(
            evidence["observed"]["comfy_running"], {"present": True, "type": "int", "value": 1}
        )
        self.assertEqual(evidence["observed"]["mem_available_bytes"]["value"], 37384560640)
        self.assertEqual(M.error_record(error)["observation_evidence"], evidence)

    def test_memory_refusal_records_exact_original_initial_and_runtime_thresholds(self):
        for initial, margin in ((False, 2 * M.GIB), (True, 2 * M.BASE)):
            with self.subTest(initial=initial):
                minimum = self.reserve + margin
                row = {**self.row, "mem_available_bytes": minimum - 1}
                error = self.refuse(row, initial=initial)
                self.assertEqual(str(error), "Selected host reserve gate")
                evidence = error.evidence
                self.assertEqual(evidence["gate"], "memory_reserve")
                self.assertEqual(evidence["phase"], "initial" if initial else "runtime")
                self.assertEqual(
                    evidence["thresholds"],
                    {
                        "boot_id": "fixture-boot",
                        "mem_total_bytes": 130595864576,
                        "reserve_bytes": self.reserve,
                        "margin_bytes": margin,
                        "min_mem_available_bytes": minimum,
                    },
                )
                observer, _ = self.observer({**row, "mem_available_bytes": minimum})
                with patch.object(M.os, "write") as output:
                    self.assertEqual(
                        M.host(self.plan, observer, initial=initial)["mem_available_bytes"], minimum
                    )
                output.assert_not_called()

    def test_unrelated_secrets_queue_prompt_and_observer_evidence_are_excluded(self):
        secret = "not-for-diagnostics-credential"
        row = {
            **self.row,
            "comfy_running": 1,
            "unrelated_secret": secret,
            "queue_running": [[1, secret]],
            "prompt": secret,
            "comfy_evidence": {"unrelated_payload": secret},
            "token": secret,
        }
        error = self.refuse(row)
        encoded = M.encode(M.error_record(error))
        self.assertNotIn(secret.encode(), encoded)
        self.assertNotIn(b"queue_running", encoded)
        self.assertEqual(set(error.evidence["observed"]), set(self.row))

    def test_bad_typed_fields_are_rejected_without_coercion_or_object_stringification(self):
        class UnsafeObject:
            def __str__(self):
                raise AssertionError("No object stringification")

        for key, value in (
            ("comfy_running", True),
            ("comfy_pending", "0"),
            ("mem_available_bytes", False),
            ("comfy_state", UnsafeObject()),
        ):
            with self.subTest(key=key):
                error = self.refuse({**self.row, key: value})
                self.assertEqual(error.evidence["observed"][key]["type"], type(value).__name__)

    def test_missing_field_and_bounded_malformed_values_remain_evidence_only(self):
        row = dict(self.row)
        del row["comfy_pending"]
        error = self.refuse(row)
        self.assertEqual(error.evidence["observed"]["comfy_pending"], {"present": False})
        error = self.refuse(
            {**self.row, "comfy_state": "x" * 100000, "mem_available_bytes": 1 << 100000}
        )
        self.assertEqual(len(error.evidence["observed"]["comfy_state"]["value"]), 128)
        self.assertTrue(error.evidence["observed"]["comfy_state"]["truncated"])
        self.assertEqual(error.evidence["observed"]["mem_available_bytes"]["integer_bits"], 100001)

    def test_stderr_failure_never_changes_original_refusal(self):
        observer, _ = self.observer({**self.row, "comfy_running": 1})
        with patch.object(M.os, "write", side_effect=OSError("closed diagnostic stream")):
            with self.assertRaisesRegex(ValueError, "Explicit same-host idle/absent") as raised:
                M.host(self.plan, observer)
        self.assertEqual(raised.exception.evidence["observed"]["comfy_running"]["value"], 1)

    def test_observer_error_identity_and_its_existing_evidence_are_preserved(self):
        failure = ValueError("fixture observer refusal")
        failure.evidence = {"existing": "evidence"}

        def observe(**kwargs):
            raise failure

        with patch.object(M.os, "write") as output:
            with self.assertRaises(ValueError) as raised:
                M.host(self.plan, SimpleNamespace(observe=observe))
        self.assertIs(raised.exception, failure)
        self.assertEqual(failure.evidence, {"existing": "evidence"})
        output.assert_not_called()

    def test_original_gate_acceptance_and_error_messages_match_over_fault_matrix(self):
        faults = [(None, None)]
        for key, values in {
            "boot_id": ["other", None],
            "mem_total_bytes": [130595864575, None],
            "mem_available_bytes": [
                None,
                True,
                0,
                -1,
                1,
                self.reserve + 2 * M.GIB - 1,
                self.reserve + 2 * M.GIB,
                self.reserve + 2 * M.BASE - 1,
                self.reserve + 2 * M.BASE,
                130595864577,
            ],
            "comfy_state": ["http_busy", "absent_confirmed", "unknown", None],
            "comfy_running": [1, -1, False, "0", None],
            "comfy_pending": [1, -1, False, "0", None],
        }.items():
            faults.extend((key, value) for value in values)
        for initial in (False, True):
            for key, value in faults:
                with self.subTest(initial=initial, key=key, value=value):
                    row = {**self.row, **({key: value} if key else {})}
                    outcomes = []
                    with patch.object(M.os, "write"):
                        for module in (OLD, M):
                            observer, _ = self.observer(row)
                            try:
                                result = module.host(self.plan, observer, initial=initial)
                                outcomes.append(("accepted", result))
                            except Exception as error:
                                outcomes.append((type(error).__name__, str(error)))
                    self.assertEqual(outcomes[0], outcomes[1])

    def test_existing_functions_including_cleanup_are_ast_identical_except_host(self):
        original = ast.parse(PARENT.read_bytes())
        current = ast.parse((HERE / "q95_owner.py").read_bytes())
        old_functions = {
            n.name: n for n in original.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        }
        new_functions = {
            n.name: n for n in current.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        }
        self.assertEqual(
            set(new_functions) - set(old_functions),
            {"host_field", "host_gate_evidence", "emit_host_refusal"},
        )
        for name in old_functions:
            if name != "host":
                self.assertEqual(ast.dump(old_functions[name]), ast.dump(new_functions[name]), name)
        before = old_functions["host"]
        after = new_functions["host"]

        def need_calls(n):
            return [
                ast.dump(x)
                for x in ast.walk(n)
                if isinstance(x, ast.Call) and isinstance(x.func, ast.Name) and x.func.id == "need"
            ]

        self.assertEqual(need_calls(before), need_calls(after))
        original_math = [n for n in before.body if isinstance(n, ast.Assign)]
        current_math = [
            n
            for n in ast.walk(after)
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in {"row", "reserve"} for t in n.targets)
        ]
        self.assertEqual(
            sorted(ast.dump(n) for n in original_math), sorted(ast.dump(n) for n in current_math)
        )

    def test_component_has_no_launch_entry_or_selected_plan_and_parent_is_unchanged(self):
        self.assertFalse((HERE / "launch_q95.py").exists())
        self.assertFalse(any(HERE.glob("launch-plan*.json")))
        self.assertIn("source-only", M.PROFILE)
        self.assertEqual(hashlib.sha256(PARENT.read_bytes()).hexdigest(), PARENT_SHA)


if __name__ == "__main__":
    unittest.main()
