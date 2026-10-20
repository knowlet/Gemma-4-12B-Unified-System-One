"""Pure receipts and tiny owned children; no processor/model/GPU imports."""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location(
    "revisit_controller_tests_subject", HERE / "revisit_capacity_controller.py"
)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cpu, cls.common = c.component(c.CPU), c.component(c.COMMON)

    def test_pinned_module_refuses_drift_before_execution(self):
        with self.assertRaisesRegex(ValueError, "Source pin"):
            c.pinned_module(
                {"path": c.CPU, "sha256": "0" * 64, "size_bytes": 57238}, "never_execute"
            )
        self.assertNotIn("never_execute", sys.modules)

    def test_original_floor_and_boot_identity(self):
        obs = {"MemTotal": 130595860480, "MemAvailable": 36856590336, "boot_id": "fake-boot"}
        fake = SimpleNamespace(
            memory_observation=lambda: obs, resource_formula=self.cpu.resource_formula
        )
        self.assertEqual(
            c.prospective(fake, "fake-boot", obs["MemTotal"])["required_bytes"], 36856590336
        )
        obs["MemAvailable"] -= 1
        with self.assertRaisesRegex(ValueError, "floor refused"):
            c.prospective(fake, "fake-boot", obs["MemTotal"])
        with self.assertRaises(ValueError):
            c.prospective(fake, "wrong-boot", obs["MemTotal"])

    def owned(self, body, *, wall=2, refuse_monitor=False, available=100 * 1024**3):
        cpu = SimpleNamespace(
            **{
                k: getattr(self.cpu, k)
                for k in ("resource_formula", "child_argv", "save", "encode", "caps", "error_value")
            }
        )
        cpu.memory_observation = lambda: {
            "MemTotal": 130595860480,
            "MemAvailable": available,
            "boot_id": "fake-boot",
        }

        class FakeMonitor:
            def __init__(self, *_args):
                self.calls = 0

            def check(self, rows):
                self.calls += 1
                if refuse_monitor and self.calls > 1:
                    raise ValueError("explicit fake monitor refusal")

        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            output = Path(temp)
            parent = {
                "pid": os.getpid(),
                "start_ticks": 1,
                "pgid": os.getpgrp(),
                "sid": os.getsid(0),
                "mem_total": 130595860480,
            }
            result, _monitor = c.run_owned(
                cpu,
                self.common,
                output,
                [sys.executable, "-B", "-c", body],
                {"PATH": os.environ["PATH"]},
                "fake-boot",
                parent,
                wall=wall,
                monitor_factory=FakeMonitor,
            )
            if result["started"]:
                self.assertTrue(result["cleanup"]["confirmed"], result)
                self.assertEqual(result["cleanup"]["remaining"], [])
                self.assertTrue((output / "owned-cleanup.json").is_file())
                table = self.common.process_table()
                self.assertFalse(
                    any(
                        row["pgid"] in result["cleanup"]["tracked_groups"] for row in table.values()
                    )
                )
            return result

    def test_tiny_child_success_is_owned_and_closed(self):
        result = self.owned("import time; print('tiny CPU fixture'); time.sleep(0.2)")
        self.assertEqual(result["returncode"], 0)
        self.assertIsNone(result["primary_error"])

    def test_timeout_closes_TERM_resistant_child(self):
        result = self.owned(
            "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)",
            wall=0.25,
        )
        self.assertIn("deadline", result["primary_error"]["message"])
        self.assertTrue(any(x["signal"] == 9 for x in result["cleanup"]["signals"]))

    def test_monitor_refusal_closes_child_and_preserves_primary(self):
        result = self.owned("import time; time.sleep(30)", refuse_monitor=True)
        self.assertEqual(result["primary_error"]["message"], "explicit fake monitor refusal")

    def test_fresh_low_memory_creates_no_child(self):
        result = self.owned("raise RuntimeError('must never execute')", available=1)
        self.assertFalse(result["started"])
        self.assertIsNone(result["cleanup"])
        self.assertIn("floor refused", result["primary_error"]["message"])

    def test_log_limit_closes_child(self):
        result = self.owned("import os,time; os.write(1,b'x'*(2*1024**2)); time.sleep(30)")
        self.assertIn("stdout/stderr", result["primary_error"]["message"])

    def test_receipt_cannot_select_without_exact_owned_closure(self):
        for cleanup in (
            None,
            {"confirmed": False, "remaining": []},
            {"confirmed": True, "remaining": [{"pid": 1}]},
        ):
            with self.assertRaisesRegex(ValueError, "Whole owned closure"):
                c.validate_worker(self.cpu, None, None, HERE, "0" * 64, "a" * 32, cleanup)

    def test_full75_fake_receipt_join_and_incomplete_or_corrupt_prefix_refusal(self):
        # Reuse the explicitly fake processor fixture, never the actual loader.
        fixture_spec = importlib.util.spec_from_file_location(
            "fake_revisit_worker_fixture", HERE / "test_revisit_capacity_worker.py"
        )
        fixture = importlib.util.module_from_spec(fixture_spec)
        fixture_spec.loader.exec_module(fixture)
        w = fixture.w
        sys.path[:0] = [str(w.ROOT / "src"), str(w.ROOT / "scripts")]
        q, validator, v8 = w.module(w.Q), w.module(w.VALIDATOR), w.module(w.V8)
        q._helper_instructions = lambda: "EXPLICIT_FAKE_HELPER_TEMPLATE"
        from s1.errors import RequestValidationError
        from s1.jev_generated import _request

        def ref(path):
            blob = path.read_bytes()
            return {
                "path": str(path.relative_to(w.ROOT)),
                "sha256": w.sha(blob),
                "size_bytes": len(blob),
            }

        cohort = w.parse(w.checked(w.fixed(w.COHORT)))
        technical = w.parse(w.checked(w.fixed(w.TECHNICAL)))
        processor_plan = w.parse(w.checked(w.fixed(w.PROCESSOR)))
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            output, nonce = Path(temp), "a" * 32
            work = output / "worker"
            work.mkdir()
            config = {
                "schema_id": w.KIND,
                "cohort": w.fixed(w.COHORT),
                "technical_plan": w.fixed(w.TECHNICAL),
                "processor_plan": w.fixed(w.PROCESSOR),
                "worker_source": ref(HERE / "revisit_capacity_worker.py"),
                "parent_source": ref(HERE / "revisit_capacity_controller.py"),
                "owner": {"pid": 10, "start_ticks": 20, "pgid": 10, "sid": 10},
                "nonce": nonce,
                "boot_id": "explicit-fake-boot",
                "mem_total": 130595860480,
                "output": str(output),
            }
            config_blob = w.wire(config)
            (output / "worker-config.json").write_bytes(config_blob)
            union = w.source_union(config, cohort, technical)
            w.save(
                work,
                "source-union.json",
                w.canonical(
                    {
                        "schema_id": w.KIND + "_source_union",
                        "files": union,
                        "count": len(union),
                        "old95_modified": False,
                    }
                ),
            )
            self.cpu.capture_reference(processor_plan, work)
            w.save(
                work,
                "runtime-component.json",
                w.canonical(
                    {
                        "runtime": technical["technical"]["runtime"],
                        "runtime_files": 348,
                        "original_execution": technical["technical"]["execution"],
                        "original_execution_files": 95,
                        "processor_plan": config["processor_plan"],
                        "processor_reference_items": 58,
                        "loaded_mandatory_generation_imports_requested": False,
                    }
                ),
            )
            evidence = w.canonical(
                {
                    "fixture": "EXPLICIT_FAKE_PROCESSOR_EVIDENCE",
                    "tokenizer_is_processor_tokenizer": True,
                    "max_context": 8192,
                    "generation_EOS": [1, 106, 50],
                    "CPU_threads": 2,
                    "CPU_interop_threads": 2,
                }
            )
            w.save(work, "processor-evidence-before.json", evidence)
            w.save(work, "processor-evidence-after.json", evidence)
            phases = [
                "startup",
                "source-before",
                "runtime-before",
                "cpu-reference-before-processor-load",
                "processor-after",
                *(["prepare"] * 75),
                "final-cpu",
            ]
            obs = {
                "MemTotal": config["mem_total"],
                "MemAvailable": 100 * 1024**3,
                "boot_id": config["boot_id"],
            }
            formula = self.cpu.resource_formula(obs, config["boot_id"], prospective=True)
            gates = [
                {
                    "schema_id": w.KIND + "_resource_gate",
                    "sequence": i,
                    "phase": p,
                    "scope": "cpu",
                    "monotonic": 100 + i,
                    "worker_elapsed_seconds": i,
                    "admitted": True,
                    "error": None,
                    "observation": obs,
                    **formula,
                    "limits": {
                        "role": "worker",
                        "soft_bytes": 8 * 1024**3,
                        "hard_bytes": 8 * 1024**3,
                    },
                    "cuda": {
                        "CUDA_VISIBLE_DEVICES": "",
                        "torch_loaded": False,
                        "CUDA_initialized": False,
                    },
                }
                for i, p in enumerate(phases)
            ]
            w.save(work, "resource-gates.jsonl", b"".join(w.wire(r) + b"\n" for r in gates))
            fake = fixture.FakeProcessor()
            rows = w.run_rows(
                cohort,
                payload_root=(w.ROOT / w.COHORT).parent,
                q=q,
                validator=validator,
                v8=v8,
                model=SimpleNamespace(processor=fake, tok=fake),
                gate=fixture.FakeGate(),
                output=work,
                request=_request,
                schema_error=RequestValidationError,
            )
            terminal = {
                "schema_id": w.KIND,
                "profile": c.PROFILE,
                "config_sha256": w.sha(config_blob),
                "nonce": nonce,
                "child": {"pid": 11, "start_ticks": 30, "pgid": 11, "sid": 11},
                "complete": True,
                "status": "CPU_PREFIXES_COMPLETE_AWAIT_WHOLE_OWNED_CLOSURE",
                "primary_error": None,
                "secondary_errors": [],
                "processor_capacity_only": True,
                "task_quality_attested": False,
                "universal_profile_capacity_attested": False,
                "native_model_constructor_forward_HTTP_browser_calls": 0,
                "rows": rows,
                "counts": {
                    name: sum(r["outcome"] == name for r in rows)
                    for name in ("ACCEPTED", "REFUSED_SCHEMA", "REFUSED_CONTEXT", "UNKNOWN")
                },
            }
            terminal_path = work / "terminal.json"
            terminal_path.write_bytes(w.canonical(terminal))
            cleanup = {
                "confirmed": True,
                "remaining": [],
                "tracked_PID_start_ticks": {"11": 30},
                "tracked_groups": [11],
            }
            result = c.validate_worker(
                self.cpu, w, cohort, output, w.sha(config_blob), nonce, cleanup
            )
            self.assertEqual(sum(result["counts"].values()), 75)
            terminal["rows"] = rows[:-1]
            terminal_path.write_bytes(w.canonical(terminal))
            with self.assertRaisesRegex(ValueError, "All75"):
                c.validate_worker(self.cpu, w, cohort, output, w.sha(config_blob), nonce, cleanup)
            terminal["rows"] = rows
            rows[0]["prefix"]["attention_mask"][0] = 0
            (work / "rows/00.json").write_bytes(w.canonical(rows[0]))
            terminal_path.write_bytes(w.canonical(terminal))
            with self.assertRaises(ValueError):
                c.validate_worker(self.cpu, w, cohort, output, w.sha(config_blob), nonce, cleanup)
        self.assertFalse(
            any(
                n.split(".")[0] in {"torch", "transformers", "numpy", "tokenizers"}
                for n in sys.modules
            )
        )


if __name__ == "__main__":
    unittest.main()
