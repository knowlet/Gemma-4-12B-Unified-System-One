"""Unexecuted stdlib fakes; no real DTO/framework/server/browser/model loaded."""

import copy
import importlib.util
import json
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def source(name):
    path = Path(__file__).with_name(name + "_held.py")
    spec = importlib.util.spec_from_file_location("_held_operation_browser_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def payload():
    # Deliberately offered BLOCKED first and two non-goal-bearing fake targets;
    # no policy choice is edited by the adapter/provider plumbing.
    return {
        "model": "d3-edge-dsl01-operation-delta-cpu5-candidate",
        "state": "Click the second offered control",
        "questions": {
            "operation": {
                "type": "choice",
                "instructions": "Choose operation",
                "criteria": {"BLOCKED": "Blocked", "DONE": "Done", "CLICK": "Click"},
            },
            "click_target": {
                "type": "choice",
                "instructions": "Choose target",
                "criteria": {"target-z": "First offered", "target-a": "Second offered"},
            },
        },
    }


class Request:
    def __init__(self, data):
        self.data = copy.deepcopy(data)
        self.questions = [types.SimpleNamespace(id=key) for key in data["questions"]]

    def model_dump(self):
        return copy.deepcopy(self.data)

    @classmethod
    def model_validate(cls, data):
        return cls(data)


def dependencies(module, seen):
    def parse(data, *, model_identity):
        assert data["model"] == model_identity == module.ALIAS
        original = copy.deepcopy(data)
        raw = json.dumps(
            original, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
        labels = {key: list(value["criteria"]) for key, value in data["questions"].items()}
        return Request(data), original, raw, labels

    def normalize(request, raw):
        seen.append((request.model_dump(), copy.deepcopy(raw)))
        return copy.deepcopy(raw)

    return {
        "parse_native_payload": parse,
        "request_type": Request,
        "normalize_native_response": normalize,
        "timeout_type": TimeoutError,
        "validation_type": ValueError,
        "response_error_type": ValueError,
        "configuration_blockers": lambda spec: ["decision3_requires_cuda_bfloat16"],
    }


class FakeProvider:
    def __init__(self, module, *, raw_model=None, after=None):
        self.name = module.ALIAS
        self.spec = types.SimpleNamespace(
            adapter="decision3", device="cpu", precision="bfloat16", context_limit=16384
        )
        self.model = module.ALIAS if raw_model is None else raw_model
        self.after, self.received = after, []

    def predict(self, request, *, deadline, evidence_sink):
        self.received.append(request.model_dump())
        raw = {
            "model": self.model,
            "answers": {
                "operation": {
                    "id": "operation",
                    "type": "choice",
                    "choice": "BLOCKED",
                    "probabilities": {"BLOCKED": 0.8, "DONE": 0.1, "CLICK": 0.1},
                },
                "click_target": {
                    "id": "click_target",
                    "type": "choice",
                    "choice": "target-z",
                    "probabilities": {"target-z": 0.6, "target-a": 0.4},
                },
            },
            "usage": {"input_tokens": 123, "output_tokens": 0},
        }
        evidence_sink("real_fake_raw", raw)
        if self.after is not None:
            self.after()
        return raw


class AdapterSourceTests(unittest.TestCase):
    def setUp(self):
        self.module = source("adapter")
        self.module.SOURCE_ONLY = False
        self.seen, self.now = [], types.SimpleNamespace(value=0.0)
        self.deps = dependencies(self.module, self.seen)

    def kind(self):
        return self.module.adapter_type(
            self.deps, verify_owned=lambda: {}, clock=lambda: self.now.value
        )

    def test_default_hold_precedes_dependency_or_capability_access(self):
        self.module.SOURCE_ONLY = True

        class Bomb:
            def __getattr__(self, name):
                raise AssertionError("unexpected dependency access: " + name)

        with self.assertRaisesRegex(ValueError, "SOURCE_ONLY"):
            self.module.adapter_type(Bomb(), verify_owned=Bomb())
        with self.assertRaisesRegex(ValueError, "SOURCE_ONLY"):
            self.module.CandidateProvider(
                spec=Bomb(),
                native=Bomb(),
                observer=Bomb(),
                delta_backend=Bomb(),
                runtime=Bomb(),
                torch=Bomb(),
                model=Bomb(),
                delta=Bomb(),
                capability=Bomb(),
                capability_sha256=Bomb(),
                native_payload=Bomb(),
                verify_owned=Bomb(),
            )

    def test_wrong_legal_BLOCKED_and_target_remain_raw_with_complete_order(self):
        offered = payload()
        provider = FakeProvider(self.module)
        result = self.kind()(provider).predict(offered)
        self.assertEqual(provider.received, [offered])
        self.assertEqual(len(self.seen), 1)
        self.assertEqual(
            [answer["id"] for answer in result["answers"]], ["operation", "click_target"]
        )
        self.assertEqual(result["answers"][0]["choice"], "BLOCKED")
        self.assertEqual(result["answers"][1]["choice"], "target-z")
        self.assertEqual(list(result["answers"][0]["probabilities"]), ["BLOCKED", "DONE", "CLICK"])
        self.assertEqual(result["model"], self.module.ALIAS)
        self.assertFalse(result["decision3_native"]["publisher_native_numerics"])
        self.assertEqual(result["decision3_native"]["original_request"], offered)

    def test_original_publisher_alias_is_rejected_before_normalization_and_retained(self):
        provider = FakeProvider(self.module, raw_model="d3-edge")
        with self.assertRaisesRegex(ValueError, "distinct delta alias") as caught:
            self.kind()(provider).predict(payload())
        self.assertEqual(self.seen, [])
        self.assertEqual(caught.exception.decision3_native["raw_response"]["model"], "d3-edge")

    def test_cpu_spec_and_other_configuration_blockers_cannot_be_spoofed(self):
        for attr, value in (("device", "cuda"), ("precision", "float32"), ("context_limit", 8192)):
            provider = FakeProvider(self.module)
            setattr(provider.spec, attr, value)
            with self.subTest(attr=attr), self.assertRaisesRegex(ValueError, "honest original CPU"):
                self.kind()(provider)
        self.deps["configuration_blockers"] = lambda spec: [
            "decision3_requires_cuda_bfloat16",
            "decision3_checkpoint_unverified",
        ]
        with self.assertRaisesRegex(ValueError, "checkpoint_unverified"):
            self.kind()(FakeProvider(self.module))

    def test_exact20_after_raw_or_completed_callback_refuses_return_without_losing_raw(self):
        for stage in ("raw", "completed"):
            with self.subTest(stage=stage):
                self.now.value = 0.0
                provider = FakeProvider(
                    self.module,
                    after=(lambda: setattr(self.now, "value", 20.0)) if stage == "raw" else None,
                )

                def sink(event):
                    if stage == "completed" and event["event"] == "completed":
                        self.now.value = 20.0

                with self.assertRaises(TimeoutError) as caught:
                    self.kind()(provider, evidence_sink=sink).predict(payload())
                self.assertEqual(caught.exception.decision3_native["status"], "failed")
                self.assertEqual(
                    caught.exception.decision3_native["raw_response"]["model"], self.module.ALIAS
                )

    def test_primary_BaseException_survives_sink_failure(self):
        original = KeyboardInterrupt("owned abort")
        provider = FakeProvider(self.module)

        def fail(request, **kwargs):
            raise original

        provider.predict = fail

        def sink(event):
            if event["event"] == "failed":
                raise OSError("failed-event evidence sink")

        with self.assertRaises(KeyboardInterrupt) as caught:
            self.kind()(provider, evidence_sink=sink).predict(payload())
        self.assertIs(caught.exception, original)
        self.assertTrue(original.decision3_native["evidence_sink_errors"])
        self.assertTrue(any("Failed-event retention" in note for note in original.__notes__))

    def test_required_sink_failure_at_planned_raw_or_completed_refuses_result(self):
        for stage in ("planned", "backend_raw_returned", "completed"):
            with self.subTest(stage=stage):
                original = OSError("required retention failed at " + stage)
                provider = FakeProvider(self.module)

                def sink(event):
                    if event["event"] == stage:
                        raise original

                with self.assertRaises(OSError) as caught:
                    self.kind()(provider, evidence_sink=sink).predict(payload())
                self.assertIs(caught.exception, original)
                self.assertEqual(original.decision3_native["status"], "failed")
                self.assertEqual(len(provider.received), 0 if stage == "planned" else 1)

    def test_provider_failure_evidence_copy_cannot_replace_original_abort(self):
        original = KeyboardInterrupt("original candidate forward abort")

        class BrokenEvidence:
            def __deepcopy__(self, memo):
                raise OSError("secondary evidence copy")

        class Backend:
            def __init__(self, *args, **kwargs):
                self.last_evidence = BrokenEvidence()

            def __call__(self, **kwargs):
                raise original

        backend_module = types.SimpleNamespace(
            SOURCE_ONLY=False,
            ALIAS=self.module.ALIAS,
            PROFILE="fake-profile",
            CAPABILITY_SCHEMA="root-accepted-dsl01-operation-train-delta-v1",
            CAPABILITY_SHA256="a" * 64,
            DeltaHeadBackend=Backend,
        )
        provider = self.module.CandidateProvider(
            spec=types.SimpleNamespace(),
            native=object(),
            observer=object(),
            delta_backend=backend_module,
            runtime=object(),
            torch=object(),
            model={},
            delta=object(),
            capability={"profile": "fake-profile"},
            capability_sha256="a" * 64,
            native_payload=lambda request: {
                "state": "unchanged",
                "questions": {"operation": {"criteria": {"BLOCKED": "Blocked"}}},
            },
            verify_owned=lambda: {},
        )
        with self.assertRaises(KeyboardInterrupt) as caught:
            provider.predict(
                object(), deadline=lambda: None, evidence_sink=lambda event, data: None
            )
        self.assertIs(caught.exception, original)
        self.assertTrue(any("failure evidence retention" in note for note in original.__notes__))
        self.assertEqual(original.decision3_provider_evidence["question_order"], ["operation"])


class WorkerSourceTests(unittest.TestCase):
    def setUp(self):
        self.module = source("worker")

    def test_source_hold_precedes_arguments_helpers_and_paths(self):
        class Bomb:
            def __iter__(self):
                raise AssertionError("arguments parsed")

            def __fspath__(self):
                raise AssertionError("path accessed")

        with self.assertRaisesRegex(RuntimeError, "SOURCE_ONLY"):
            self.module.main(Bomb())
        with self.assertRaisesRegex(RuntimeError, "SOURCE_ONLY"):
            self.module.run(Bomb(), Bomb(), Bomb())

    def test_source_release_alone_cannot_bypass_pending_new_artifact_pins(self):
        with (
            patch.object(self.module, "SOURCE_ONLY", False),
            patch.object(self.module.argparse, "ArgumentParser") as parser,
            patch.object(self.module, "bootstrap") as importer,
        ):
            self.assertIsNone(self.module.CAPABILITY_SHA256)
            self.assertIsNone(self.module.NUMERIC_ACCEPTANCE_SHA256)
            with self.assertRaisesRegex(RuntimeError, "pins remain pending"):
                self.module.main([])
            parser.assert_not_called()
            importer.assert_not_called()

    def test_pending_operation_plan_is_rejected_before_dependency_or_path_access(self):
        class Bomb:
            def __getattr__(self, name):
                raise AssertionError("pending plan touched " + name)

        plan = {
            "schema": "owned-decision3-operation-cpu5-browser-worker-plan-v1",
            "profile": self.module.PROFILE,
            "status": "PENDING",
        }
        with self.assertRaisesRegex(RuntimeError, "pending/foreign"):
            self.module.validate_plan(Bomb(), plan, Bomb(), Bomb(), Bomb())

    def test_fixed_npy_reader_requires_exact_shape_dtype_layout_and_payload_length(self):
        def record(metadata, body=b"\0" * (255 * 1024 * 4)):
            header = (repr(metadata) + "\n").encode("ascii")
            return b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header + body

        metadata = {"descr": "<f4", "fortran_order": False, "shape": (255, 1024)}
        self.assertEqual(len(self.module.fp32_payload(record(metadata))), 255 * 1024 * 4)
        for changed in (
            {**metadata, "descr": "|O"},
            {**metadata, "fortran_order": True},
            {**metadata, "shape": (254, 1024)},
        ):
            with self.assertRaises(RuntimeError):
                self.module.fp32_payload(record(changed))
        with self.assertRaises(RuntimeError):
            self.module.fp32_payload(record(metadata, b"short"))
        # The literal reader cannot execute expressions in a supplied header.
        header = b"__import__('not_executed')\n"
        with self.assertRaises(ValueError):
            self.module.fp32_payload(
                b"\x93NUMPY\x01\x00"
                + len(header).to_bytes(2, "little")
                + header
                + b"\0" * (255 * 1024 * 4)
            )

    def test_numeric_acceptance_exact_candidate_backend_model_and_baseline_joins(self):
        ref = {"path": "/fake/root-metadata", "sha256": "a" * 64, "size_bytes": 123}
        cap = {
            "root_acceptance_ref": dict(ref),
            "delta_ref": {**ref, "dtype": "float32"},
            "baseline_fp32_bytes_sha256": "b" * 64,
        }
        plan = {
            "candidate_capability": dict(ref),
            "delta_backend": {"path": "/fake/backend.py", "sha256": "c" * 64},
        }
        model = {"model_id": "vllm-sr/d3-edge", "revision": "fixed"}
        numeric = {
            "schema": "root-accepted-operation-cpu5-numeric-v1",
            "authority": "root",
            "status": "accepted",
            "root_exit": 0,
            "owner_completed": True,
            "current_original_scoped_closure_positive": True,
            "candidate_alias": self.module.ALIAS,
            "candidate_profile": self.module.CANDIDATE_PROFILE,
            "only_adapted_head_id": "operation",
            "dev_used_for_model_selection": True,
            "optimizer_input_dev": False,
            "optimizer_input_sealed": False,
            "candidate_selection_before_dev_or_sealed": False,
            "candidate_capability": dict(ref),
            "training_acceptance": dict(ref),
            "delta_ref": copy.deepcopy(cap["delta_ref"]),
            "baseline_fp32_bytes_sha256": "b" * 64,
            "model": copy.deepcopy(model),
            "backend_source": {**plan["delta_backend"], "size_bytes": 100},
            "browser_qualified": False,
            "quality_or_win_claim": False,
        }
        self.module.validate_numeric_acceptance(numeric, cap, model, plan)
        for key, value in (
            ("status", "PENDING"),
            ("schema", "root-accepted-frozen-delta-cpu5-numeric-v1"),
            ("root_exit", 1),
            ("owner_completed", False),
            ("candidate_alias", "d3-edge"),
            ("only_adapted_head_id", "all_heads"),
            ("candidate_profile", "decision3-dsl01-delta-readout-cpu5-4g-candidate-v1"),
            ("dev_used_for_model_selection", False),
            ("optimizer_input_dev", True),
            ("optimizer_input_sealed", True),
            ("candidate_selection_before_dev_or_sealed", True),
            ("quality_or_win_claim", True),
            ("browser_qualified", True),
            ("model", {"model_id": "foreign"}),
            ("delta_ref", {}),
            ("baseline_fp32_bytes_sha256", "d" * 64),
            ("backend_source", {"path": "/wrong", "sha256": "c" * 64}),
        ):
            changed = copy.deepcopy(numeric)
            changed[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "binding differs"):
                self.module.validate_numeric_acceptance(changed, cap, model, plan)


class OperationProviderWiringTests(unittest.TestCase):
    def test_authenticated_capability_sha_reaches_required_backend_keyword(self):
        adapter = source("adapter")
        adapter.SOURCE_ONLY = False
        captured = {}
        sha = "a" * 64
        offered = payload()

        class Backend:
            def __init__(
                self, *args, root_capability, root_capability_sha256, deadline, evidence_sink
            ):
                captured.update(
                    capability=copy.deepcopy(root_capability), sha=root_capability_sha256
                )
                self.last_evidence = {"source_fake_only": True}

            def __call__(self, **native):
                captured["native"] = copy.deepcopy(native)
                return FakeProvider(adapter).predict(
                    Request(native),
                    deadline=lambda: None,
                    evidence_sink=lambda *_args: None,
                )

        backend_module = types.SimpleNamespace(
            SOURCE_ONLY=False,
            ALIAS=adapter.ALIAS,
            PROFILE="invented-operation-profile",
            CAPABILITY_SCHEMA="root-accepted-dsl01-operation-train-delta-v1",
            CAPABILITY_SHA256=sha,
            DeltaHeadBackend=Backend,
        )
        cap = {"profile": backend_module.PROFILE, "only_head": "operation"}

        def native_payload(request):
            return {key: request.model_dump()[key] for key in ("state", "questions")}

        options = dict(
            spec=types.SimpleNamespace(),
            native=object(),
            observer=object(),
            delta_backend=backend_module,
            runtime=object(),
            torch=object(),
            model={},
            delta=object(),
            capability=cap,
            capability_sha256=sha,
            native_payload=native_payload,
            verify_owned=lambda: {},
        )
        provider = adapter.CandidateProvider(**options)
        raw = provider.predict(
            Request(offered), deadline=lambda: None, evidence_sink=lambda *_args: None
        )
        self.assertEqual(captured["sha"], sha)
        self.assertEqual(captured["capability"], cap)
        self.assertEqual(captured["native"], {key: offered[key] for key in ("state", "questions")})
        self.assertEqual(list(captured["native"]["questions"]), ["operation", "click_target"])
        self.assertEqual(raw["answers"]["operation"]["choice"], "BLOCKED")
        self.assertEqual(
            list(raw["answers"]["operation"]["probabilities"]), ["BLOCKED", "DONE", "CLICK"]
        )
        self.assertEqual(raw["answers"]["click_target"]["choice"], "target-z")
        for wrong in (None, "x" * 64, "b" * 64):
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                adapter.CandidateProvider(**{**options, "capability_sha256": wrong})


if __name__ == "__main__":
    unittest.main()
