"""Critical replay integrity checks; stdlib only, no Torch/model/network imports."""

import copy
import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/jev_decision3_public_dev22_gpu_replay.py"
SPEC = importlib.util.spec_from_file_location("decision3_gpu_replay_tested_source", SCRIPT)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class ReplayTests(unittest.TestCase):
    def test_projection_preserves_ordered_context_and_scoring_uses_active_target(self):
        original = {
            "model": "original",
            "state": {"goal": "open requested document", "history": ["observed"]},
            "questions": {
                "operation": {
                    "type": "choice",
                    "instructions": {"stop": "only if complete"},
                    "criteria": {"CLICK": "open", "DONE": "complete"},
                },
                "click_target": {
                    "type": "choice",
                    "instructions": ["all candidates"],
                    "criteria": {"71": "requested", "3": "unrelated"},
                },
            },
        }
        saved = copy.deepcopy(original)

        def parser(wire, *, model_identity):
            self.assertEqual(model_identity, "alias")
            request = SimpleNamespace(questions=wire["questions"], state=wire["state"])
            return request, wire, m.encode(wire), {}

        def publisher(request):
            return {"state": request.state, "questions": request.questions}

        _, alias, native = m.project(original, "alias", parser, publisher)
        self.assertEqual(original, saved)
        self.assertEqual(alias["model"], "alias")
        self.assertEqual(
            m.encode(native), m.encode({"state": saved["state"], "questions": saved["questions"]})
        )
        action = m.select_action(
            original,
            {"answers": {"operation": {"choice": "DONE"}, "click_target": {"choice": "3"}}},
        )
        case = {"gold": {"operation": "DONE"}, "valid_next_targets": {"DONE": []}}
        self.assertIsNone(action["target"])
        self.assertTrue(m.score_action(case, action)["valid_next_targets_match"])
        self.assertFalse(
            m.score_action(case, {"operation": "CLICK", "target": "3"})["valid_next_targets_match"]
        )

    def test_late_raw_is_retained_and_scored_without_becoming_timely(self):
        clock = SimpleNamespace(now=0)
        original = {
            "model": "original",
            "state": "complete observation",
            "questions": {
                "operation": {
                    "type": "choice",
                    "instructions": "stop when complete",
                    "criteria": {"DONE": "complete"},
                }
            },
        }
        question = SimpleNamespace(id="operation", labels=lambda: ["DONE"])
        request = SimpleNamespace(questions=[question])
        raw = {
            "model": "d3-edge",
            "answers": {
                "operation": {
                    "type": "choice",
                    "choice": "DONE",
                    "probabilities": {"DONE": 1.0},
                    "confidence": 1.0,
                }
            },
            "usage": {"input_tokens": 2, "output_tokens": 0},
        }
        prepared = SimpleNamespace(
            errors={},
            keys=["operation"],
            sequences={"operation": [7, 8]},
            questions={"operation": SimpleNamespace(rendered="rendered")},
        )
        runtime = SimpleNamespace(
            prepare=lambda **kwargs: prepared, text=lambda *args: "native prompt"
        )

        def predict(actual_request, *, deadline, evidence_sink):
            self.assertIs(actual_request, request)
            clock.now = 25
            evidence = {"raw_response": raw, "native_payload": {}}
            evidence_sink("native_raw_returned", evidence)
            try:
                deadline()
            except TimeoutError as exc:
                exc.decision3_provider_evidence = evidence
                raise

        backend = SimpleNamespace(name="alias", runtime=runtime, predict=predict)
        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda device: None))
        gates = SimpleNamespace(check=lambda stage: None)
        functions = (lambda *args, **kwargs: (request, {}, b"", {}), lambda r: {}, lambda r, v: v)
        case = {
            "id": "complete-0",
            "gold": {"operation": "DONE"},
            "valid_next_targets": {"DONE": []},
            "state_role": "ordinary_complete",
        }
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch.object(m, "project", return_value=(request, original, original)),
        ):
            directory = Path(temporary)
            result = m.replay_row(
                case,
                {"native_payload": original},
                "d3-edge",
                backend,
                torch,
                directory,
                gates,
                functions,
                clock=lambda: clock.now,
            )
            self.assertEqual(result["raw_response"], raw)
            self.assertEqual(result["status"], "late_valid_response")
            self.assertTrue(result["offline_scoring"]["valid_next_targets_match"])
            self.assertFalse(result["within20s_action_valid"])
            self.assertEqual(json.loads((directory / "complete-0.json").read_bytes()), result)
        missing = {**case, "id": "not-returned-1"}
        summary = m.summarize([case, missing], [result])["all22"]
        self.assertEqual(
            summary,
            {
                "planned": 2,
                "raw_returned": 1,
                "contract_valid": 1,
                "raw_action_valid": 1,
                "within20s_action_valid": 0,
                "late_valid": 1,
                "unavailable": 1,
            },
        )

    def test_full_shard_union_and_model_specific_readout_are_verified(self):
        def tensor(path, entries):
            offset, header = 0, {}
            for key, (dtype, shape) in entries.items():
                size = m.math.prod(shape) * {"BF16": 2, "F32": 4}[dtype]
                header[key] = {
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [offset, offset + size],
                }
                offset += size
            blob = m.encode(header)
            path.write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(offset))

        with tempfile.TemporaryDirectory() as temporary:
            hub = Path(temporary) / "hub"
            snapshot = hub / "models--vllm-sr--fake" / "snapshots" / ("a" * 40)
            snapshot.mkdir(parents=True)
            tensor(snapshot / "model-1.safetensors", {"first": ("BF16", [2, 3])})
            tensor(snapshot / "model-2.safetensors", {"second": ("BF16", [2, 4])})
            tensor(snapshot / "readout.safetensors", {"weight": ("F32", [255, 2])})
            (snapshot / "model.safetensors.index.json").write_bytes(
                m.encode(
                    {
                        "weight_map": {
                            "first": "model-1.safetensors",
                            "second": "model-2.safetensors",
                        }
                    }
                )
            )
            pins = {p.name: m.pin(p) for p in snapshot.iterdir()}
            manifest = {
                "schema": "d3-package-manifest/1",
                "model_name": "fake",
                "repo_id": "vllm-sr/fake",
                "files_sha256": {name: pin["sha256"] for name, pin in pins.items()},
                "files_bytes": {name: pin["size_bytes"] for name, pin in pins.items()},
            }
            blob = m.encode(manifest)
            (snapshot / "MODEL_MANIFEST.json").write_bytes(blob)
            identity = {
                "model_id": "vllm-sr/fake",
                "revision": "a" * 40,
                "manifest_sha256": hashlib.sha256(blob).hexdigest(),
                "loaded_parameters": 524,
                "readout_tensor_bytes": 2040,
                "resident_file_bytes": sum(
                    pin["size_bytes"] for name, pin in pins.items() if name.endswith(".safetensors")
                ),
            }
            output = Path(temporary) / "output"
            output.mkdir()
            gates = SimpleNamespace(check=lambda stage: None)
            with patch.object(m, "HUB", hub):
                report = m.authenticate_cache("fake", identity, output, gates, "valid")
                self.assertEqual(
                    report["all_shards"], ["model-1.safetensors", "model-2.safetensors"]
                )
                self.assertEqual(report["readout_shape"], [255, 2])
                with self.assertRaisesRegex(RuntimeError, "readout differs"):
                    m.authenticate_cache(
                        "fake",
                        {**identity, "readout_tensor_bytes": 4080},
                        output,
                        gates,
                        "bad-width",
                    )
                (snapshot / "model-2.safetensors").unlink()
                with self.assertRaises(FileNotFoundError):
                    m.authenticate_cache("fake", identity, output, gates, "missing-shard")

    def test_cgroup_can_only_restrict_existing_full_memory_floor(self):
        snapshot = {
            "proc_meminfo": {"MemTotal": 160 * 1024**3, "MemAvailable": 150 * 1024**3},
            "cgroup_memory": [{"limit": 100 * 1024**3, "current": 60 * 1024**3}],
        }
        decision = m.memory_decision(snapshot, 20 * 1024**3)
        self.assertEqual(decision["required_available_bytes"], 74 * 1024**3)
        self.assertEqual(decision["available_bytes"], 40 * 1024**3)
        self.assertFalse(decision["admitted"])
        self.assertEqual(decision["swap_or_resident_credit_bytes"], 0)

    def test_setup_failure_retains_receipt_before_framework_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "d3"
            args = SimpleNamespace(
                output_dir=output,
                worker="d3",
                source_sha256="reviewed",
                expected_boot_id="host",
                expected_memtotal=1,
            )
            with patch.object(
                m, "authenticate_source", side_effect=RuntimeError("source mismatch")
            ):
                self.assertEqual(m.worker(args), 1)
            receipt = json.loads((output / "receipt.json").read_bytes())
            self.assertEqual(receipt["stage"], "failed")
            self.assertEqual(receipt["failure_stage"], "initializing")
            self.assertEqual(receipt["errors"][0]["message"], "source mismatch")
            self.assertEqual(receipt["rows"], [])

    def test_unknown_cgroup_is_refused_but_observed_unlimited_is_complete(self):
        snapshot = {
            "proc_meminfo": {"MemTotal": 160 * 1024**3, "MemAvailable": 150 * 1024**3},
            "cgroup_memory": [],
        }
        for groups in (
            [],
            [{}],
            [{"limit": None}],
            [{"limit": None, "current": -1}],
            [{"limit": 0, "current": 0}],
        ):
            with self.subTest(cgroup=groups):
                unknown = m.memory_decision({**snapshot, "cgroup_memory": groups}, 20 * 1024**3)
                self.assertFalse(unknown["admitted"])
                self.assertFalse(unknown["cgroup_observation_complete"])
                self.assertEqual(unknown["refusal_reason"], "cgroup_memory_observation_incomplete")
                self.assertIsNone(unknown["available_bytes"])
        observed = m.memory_decision(
            {**snapshot, "cgroup_memory": [{"limit": None, "current": 0}]}, 20 * 1024**3
        )
        self.assertTrue(observed["admitted"])
        self.assertTrue(observed["cgroup_observation_complete"])
        self.assertEqual(observed["required_available_bytes"], 74 * 1024**3)
        self.assertEqual(observed["available_bytes"], snapshot["proc_meminfo"]["MemAvailable"])
        self.assertEqual(observed["swap_or_resident_credit_bytes"], 0)

    def test_secondary_fatal_overrides_ordinary_primary_at_every_finalization_boundary(self):
        original = {
            "model": "original",
            "state": "observed",
            "questions": {
                "operation": {
                    "type": "choice",
                    "instructions": "decide",
                    "criteria": {"DONE": "complete"},
                }
            },
        }
        request = SimpleNamespace(
            questions=[SimpleNamespace(id="operation", labels=lambda: ["DONE"])]
        )
        raw = {
            "model": "d3",
            "answers": {
                "operation": {
                    "type": "choice",
                    "choice": "DONE",
                    "probabilities": {"DONE": 1.0},
                    "confidence": 1.0,
                }
            },
            "usage": {"input_tokens": 2, "output_tokens": 0},
        }
        prepared = SimpleNamespace(
            errors={},
            keys=["operation"],
            sequences={"operation": [7, 8]},
            questions={"operation": SimpleNamespace(rendered="rendered")},
        )
        runtime = SimpleNamespace(
            prepare=lambda **kwargs: prepared, text=lambda *args: "native prompt"
        )
        case = {"id": "case-0", "gold": {"operation": "DONE"}, "valid_next_targets": {"DONE": []}}
        functions = (None, None, lambda request, value: value)
        real_write = m.write
        for first_type in (TimeoutError, ValueError):
            for fatal_type in (KeyboardInterrupt, SystemExit, m.AdmissionError, OSError):
                for boundary in ("synchronize", "verify", "checkpoint"):
                    with self.subTest(
                        primary=first_type.__name__, fatal=fatal_type.__name__, boundary=boundary
                    ):
                        primary, fatal = first_type("first failure"), fatal_type("second failure")
                        after_predict, injected = False, False

                        def predict(request, *, deadline, evidence_sink):
                            nonlocal after_predict
                            evidence_sink("native_raw_returned", {"raw_response": raw})
                            after_predict = True
                            raise primary

                        def persist(path, value, **kwargs):
                            nonlocal injected
                            if boundary == "checkpoint" and after_predict and not injected:
                                injected = True
                                raise fatal
                            return real_write(path, value, **kwargs)

                        synchronize = Mock(
                            side_effect=[None, fatal] if boundary == "synchronize" else None
                        )
                        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=synchronize))
                        backend = SimpleNamespace(name="alias", runtime=runtime, predict=predict)
                        with (
                            tempfile.TemporaryDirectory() as temporary,
                            patch.object(m, "project", return_value=(request, original, original)),
                            patch.object(m, "write", side_effect=persist),
                            patch.object(
                                m,
                                "verify_response",
                                side_effect=fatal if boundary == "verify" else None,
                                return_value=raw,
                            ),
                        ):
                            directory = Path(temporary)
                            with self.assertRaises(fatal_type) as caught:
                                m.replay_row(
                                    case,
                                    {"native_payload": original},
                                    "d3",
                                    backend,
                                    torch,
                                    directory,
                                    SimpleNamespace(check=lambda stage: None),
                                    functions,
                                )
                            self.assertIs(caught.exception, fatal)
                            self.assertIs(caught.exception.__cause__, primary)
                            row = caught.exception.replay_row
                            self.assertEqual(row["error"]["type"], first_type.__name__)
                            self.assertEqual(
                                row["secondary_errors"][0]["type"], fatal_type.__name__
                            )
                            self.assertTrue(row["secondary_errors"][0]["fatal"])
                            self.assertTrue(row["campaign_stop"])
                            self.assertEqual(
                                json.loads((directory / "case-0.json").read_bytes()), row
                            )

    def test_primary_fatal_object_survives_secondary_error(self):
        for first_type, second_type in (
            (KeyboardInterrupt, ValueError),
            (m.AdmissionError, SystemExit),
        ):
            with self.subTest(primary=first_type.__name__, secondary=second_type.__name__):
                primary, secondary = first_type("primary"), second_type("secondary")
                synchronize = Mock(side_effect=[None, secondary])
                backend = SimpleNamespace(name="alias")
                original = {"model": "alias", "questions": {}}
                case = {"id": "primary-fatal"}
                with (
                    tempfile.TemporaryDirectory() as temporary,
                    patch.object(m, "project", side_effect=primary),
                ):
                    with self.assertRaises(first_type) as caught:
                        m.replay_row(
                            case,
                            {"native_payload": original},
                            "d3",
                            backend,
                            SimpleNamespace(cuda=SimpleNamespace(synchronize=synchronize)),
                            Path(temporary),
                            None,
                            (None, None, None),
                        )
                    self.assertIs(caught.exception, primary)
                    self.assertTrue(any("row_finalization" in note for note in primary.__notes__))
                    self.assertTrue(primary.replay_row["campaign_stop"])

    def test_close_failure_keeps_primary_and_durable_terminal_receipt(self):
        for close_type in (ValueError, OSError, KeyboardInterrupt, SystemExit):
            for has_primary in (False, True):
                with (
                    self.subTest(close=close_type.__name__, has_primary=has_primary),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    primary = TimeoutError("original request error") if has_primary else None
                    close = Mock(side_effect=close_type("close failed"))
                    state = {
                        "stage": "failed" if has_primary else "complete",
                        "rows": [],
                        "errors": [m.error_info(primary)] if has_primary else [],
                    }
                    m.finish_worker(
                        state,
                        Path(temporary),
                        SimpleNamespace(close=close),
                        None,
                        m.time.monotonic(),
                        primary=primary,
                    )
                    saved = json.loads((Path(temporary) / "receipt.json").read_bytes())
                    self.assertEqual(saved, state)
                    self.assertEqual(saved["stage"], "failed")
                    self.assertTrue(saved["campaign_stop"])
                    self.assertEqual(saved["stop_reason"], "backend_close")
                    self.assertEqual(saved["cleanup_errors"][0]["type"], close_type.__name__)
                    if has_primary:
                        self.assertEqual(saved["errors"][0]["type"], "TimeoutError")
                        self.assertTrue(any("backend_close" in note for note in primary.__notes__))
                    close.assert_called_once_with()

    def test_failed_terminal_write_preserves_original_fatal_object(self):
        primary = KeyboardInterrupt("original stop")
        original_cause = TimeoutError("earlier request")
        primary.__cause__ = original_cause
        state = {"stage": "failed", "rows": [], "errors": [m.error_info(primary)]}
        with patch.object(m, "write", side_effect=OSError("cannot persist")):
            with self.assertRaises(KeyboardInterrupt) as caught:
                m.finish_worker(
                    state, Path("/unused"), None, None, m.time.monotonic(), primary=primary
                )
        self.assertIs(caught.exception, primary)
        self.assertIs(caught.exception.__cause__, original_cause)
        self.assertIs(caught.exception.worker_receipt, state)
        self.assertEqual(state["cleanup_errors"][0]["stage"], "terminal_receipt_persistence")

    def test_controller_stops_before_next_model_after_cleanup_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            output = output_root / "new-run"
            child = SimpleNamespace(pid=123)
            identities = {name: {"resident_file_bytes": 1} for name in m.MODEL_ORDER}
            case = {
                "id": "case-0",
                "gold": {"operation": "DONE"},
                "state_role": "ordinary_complete",
            }
            resource = {"boot_id": "boot", "proc_meminfo": {"MemTotal": 1}}
            gates = SimpleNamespace(check=lambda stage: None)
            events = []

            def drain(process, control):
                self.assertIs(process, child)
                directory = output / "d3"
                directory.mkdir()
                m.write(
                    directory / "receipt.json",
                    {
                        "stage": "failed",
                        "rows": [],
                        "campaign_stop": True,
                        "stop_reason": "backend_close",
                        "cleanup_errors": [
                            {"stage": "backend_close", "type": "ValueError", "message": "failed"}
                        ],
                    },
                )
                events.append("drained")
                return 1

            def terminate(process):
                if process is not None:
                    events.append("confirmed_exit")

            args = SimpleNamespace(output_dir=output, source_sha256="reviewed")
            with (
                patch.object(m, "sys", SimpleNamespace(modules={}, executable=m.sys.executable)),
                patch.object(m, "OUTPUT_ROOT", output_root),
                patch.object(m, "authenticate_source", return_value=identities),
                patch.object(m, "authenticate_inputs", return_value=([case], [])),
                patch.object(m, "resource_snapshot", return_value=resource),
                patch.object(m, "Gates", return_value=gates),
                patch.object(m.subprocess, "Popen", return_value=child) as launch,
                patch.object(m, "drain", side_effect=drain),
                patch.object(m, "terminate", side_effect=terminate),
            ):
                self.assertEqual(m.controller(args), 1)
            saved = json.loads((output / "receipt.json").read_bytes())
            self.assertEqual(launch.call_count, 1)
            self.assertTrue(saved["campaign_stop"])
            self.assertEqual(saved["unrun_models"], list(m.MODEL_ORDER[1:]))
            self.assertTrue(saved["arms"][0]["owned_process_group_exited"])
            self.assertEqual(events[:2], ["drained", "confirmed_exit"])

    def test_durable_prelaunch_admission_refusal_allows_next_smaller_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            output = output_root / "new-run"
            child = SimpleNamespace(pid=123)
            identities = {name: {"resident_file_bytes": 1} for name in m.MODEL_ORDER}
            case = {
                "id": "case-0",
                "gold": {"operation": "DONE"},
                "state_role": "ordinary_complete",
            }
            resource = {"boot_id": "boot", "proc_meminfo": {"MemTotal": 1}}

            class Gates:
                def __init__(self, directory, *args):
                    self.directory = directory

                def check(self, stage):
                    if self.directory.name == "d3-controller":
                        (self.directory / "gates").mkdir()
                        row = {"sequence": 0, "stage": stage, "decision": {"admitted": False}}
                        m.write(self.directory / "gates/0000.json", row)
                        error = m.AdmissionError("full floor refused")
                        error.resource_gate = row
                        raise error

            def drain(process, control):
                self.assertIs(process, child)
                self.assertEqual(control.name, "d3-flash-controller")
                directory = output / "d3-flash"
                directory.mkdir()
                # The second arm stops for an independent cleanup failure; the
                # first genuine resource refusal must not prevent its launch.
                m.write(
                    directory / "receipt.json",
                    {
                        "stage": "failed",
                        "rows": [],
                        "campaign_stop": True,
                        "stop_reason": "backend_close",
                        "cleanup_errors": [
                            {"stage": "backend_close", "type": "ValueError", "message": "failed"}
                        ],
                    },
                )
                return 1

            args = SimpleNamespace(output_dir=output, source_sha256="reviewed")
            with (
                patch.object(m, "sys", SimpleNamespace(modules={}, executable=m.sys.executable)),
                patch.object(m, "OUTPUT_ROOT", output_root),
                patch.object(m, "authenticate_source", return_value=identities),
                patch.object(m, "authenticate_inputs", return_value=([case], [])),
                patch.object(m, "resource_snapshot", return_value=resource),
                patch.object(m, "Gates", Gates),
                patch.object(m.subprocess, "Popen", return_value=child) as launch,
                patch.object(m, "drain", side_effect=drain),
                patch.object(m, "terminate"),
            ):
                self.assertEqual(m.controller(args), 1)
            saved = json.loads((output / "receipt.json").read_bytes())
            self.assertEqual(launch.call_count, 1)
            first, second = saved["arms"]
            self.assertEqual(first["stage"], "unavailable_resource_admission")
            self.assertFalse(first["worker_spawned"])
            self.assertEqual(first["summary"]["all22"]["unavailable"], 1)
            self.assertEqual(
                first["initial_resource_refusal"]["gate"]["decision"], {"admitted": False}
            )
            self.assertEqual(second["model"], "d3-flash")
            self.assertTrue(second["worker_spawned"])
            self.assertEqual(saved["unrun_models"], list(m.MODEL_ORDER[2:]))


if __name__ == "__main__":
    unittest.main()
