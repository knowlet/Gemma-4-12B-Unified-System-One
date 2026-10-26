"""Bounded stdlib synthetic controls only: no ML, owner, HTTP, or payload."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


a = load("canary_actor_tests", HERE / "actor.py")
b = load("canary_builder_tests", HERE / "builder.py")
n = a.load_native()
h = n.load_helper()


def record(path):
    raw = path.read_bytes()
    return {"path": str(path), "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


MODEL = record(a.ROOT / "artifacts/jev-runtime/upstream/jev_ultrafast/model.py")
QUESTIONS = record(a.ROOT / "artifacts/jev-runtime/upstream/jev_ultrafast/questions.py")
FIXTURES = json.loads((HERE / "fixtures.json").read_text())


def fake_answer(body, operation="CLICK"):
    answers = {}
    for key, question in body["questions"].items():
        labels = list(question["criteria"])
        choice = operation if key == "operation" else labels[0]
        values = {label: float(label == choice) for label in labels}
        answers[key] = {
            "id": key,
            "type": "choice",
            "choice": choice,
            "probabilities": values,
            "confidence": 1.0,
        }
    return {"model": "d3-edge", "answers": answers, "usage": {}}


class BuilderTests(unittest.TestCase):
    def test_original_functions_build_complete_browser_questions(self):
        submitted = []

        def callback(url, key, body):
            self.assertEqual(url, "https://api.typesafe.ai/v1/systemone")
            self.assertEqual(key, "in-process-canary")
            submitted.append(copy.deepcopy(body))
            return fake_answer(body)

        choose, proof = b.load_builder(MODEL, QUESTIONS, callback)
        decision = choose(FIXTURES["cases"]["actionable"]["page"], FIXTURES["goal"], [])
        self.assertEqual(decision["choice"], "click:lantern-switch")
        self.assertEqual(list(submitted[0]["questions"]), ["operation", "click_target"])
        self.assertEqual(
            submitted[0]["questions"]["operation"]["instructions"]["goal"], FIXTURES["goal"]
        )
        self.assertEqual(
            list(submitted[0]["questions"]["operation"]["criteria"]),
            ["CLICK", "WAIT", "DONE", "BLOCKED"],
        )
        self.assertTrue(proof["compile_dont_inherit"])

    def test_source_changes_refused_before_compile(self):
        bad = {**MODEL, "sha256": "0" * 64}
        with self.assertRaisesRegex(RuntimeError, "source changed"):
            b.load_builder(bad, QUESTIONS, lambda *_: None)

    def test_ast_exact_and_no_import_or_future_inherited(self):
        choose, proof = b.load_builder(MODEL, QUESTIONS, lambda *_: None)
        tree = ast.parse(Path(MODEL["path"]).read_bytes())
        original = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "choose"
        )
        self.assertEqual(
            proof["functions"]["choose"]["ast_sha256"],
            hashlib.sha256(ast.dump(original).encode()).hexdigest(),
        )
        self.assertEqual(choose.__code__.co_filename, MODEL["path"])
        self.assertFalse(
            choose.__code__.co_flags & __import__("__future__").annotations.compiler_flag
        )
        self.assertNotIn("httpx", choose.__globals__)

    def test_reversal_submits_every_actual_criteria_in_reverse(self):
        requests = []
        adapter = NS(predict=lambda body: requests.append(copy.deepcopy(body)) or fake_answer(body))
        wire = NS(named_browser_response=lambda body, raw: raw)
        evidence = {}
        transport = a.make_transport(b, adapter, wire, "reversed", evidence)
        choose, _ = b.load_builder(MODEL, QUESTIONS, transport)
        decision = choose(FIXTURES["cases"]["actionable"]["page"], FIXTURES["goal"], [])
        self.assertEqual(
            list(requests[0]["questions"]["operation"]["criteria"]),
            ["BLOCKED", "DONE", "WAIT", "CLICK"],
        )
        self.assertEqual(decision["operation"], "CLICK")
        self.assertEqual(evidence["submitted_request"], requests[0])
        self.assertEqual(
            list(evidence["original_constructed_request"]["questions"]["operation"]["criteria"]),
            ["CLICK", "WAIT", "DONE", "BLOCKED"],
        )

    def test_all_six_predeclared_cells_not_adaptive(self):
        self.assertEqual(
            a.CELLS,
            [
                (key, order)
                for key in ("actionable", "achieved", "unavailable")
                for order in ("original", "reversed")
            ],
        )
        self.assertEqual(FIXTURES["controls"], [])
        self.assertFalse(FIXTURES["adaptive_controls"])

    def test_unavailable_keeps_canonical_wait_but_no_invented_target(self):
        captured = []
        choose, _ = b.load_builder(
            MODEL,
            QUESTIONS,
            lambda u, k, body: captured.append(body) or fake_answer(body, "BLOCKED"),
        )
        result = choose(FIXTURES["cases"]["unavailable"]["page"], FIXTURES["goal"], [])
        self.assertEqual(list(captured[0]["questions"]), ["operation"])
        self.assertEqual(
            list(captured[0]["questions"]["operation"]["criteria"]), ["WAIT", "DONE", "BLOCKED"]
        )
        self.assertEqual(result["operation"], "BLOCKED")

    def test_incorrect_legal_blocked_is_diagnostic_not_error_or_fallback(self):
        choose, _ = b.load_builder(
            MODEL, QUESTIONS, lambda u, k, body: fake_answer(body, "BLOCKED")
        )
        result = choose(FIXTURES["cases"]["actionable"]["page"], FIXTURES["goal"], [])
        diag = b.semantic_diagnostic(result, FIXTURES["cases"]["actionable"]["expected"])
        self.assertEqual(result["choice"], "BLOCKED")
        self.assertFalse(diag["matches_self_authored_predicate"])
        self.assertFalse(diag["helper_called"])

    def test_native_choice_cannot_be_replaced_by_missing_helper(self):
        choose, _ = b.load_builder(MODEL, QUESTIONS, lambda u, k, body: fake_answer(body, "CLICK"))
        result = choose(FIXTURES["cases"]["actionable"]["page"], FIXTURES["goal"], [])
        diag = b.semantic_diagnostic(result, {"operation": "DONE", "target": None})
        self.assertEqual(result["operation"], "CLICK")
        self.assertEqual(diag["subsequent_state"], "not_acted; diagnostic preserves native choice")

    def test_expected_predicates_not_in_submitted_model_input(self):
        choose, _ = b.load_builder(
            MODEL,
            QUESTIONS,
            lambda u, k, body: self.assertNotIn("expected", json.dumps(body)) or fake_answer(body),
        )
        choose(FIXTURES["cases"]["actionable"]["page"], FIXTURES["goal"], [])

    def test_transport_cannot_repeat_same_cell(self):
        evidence = {}
        transport = a.make_transport(
            b,
            NS(predict=lambda body: fake_answer(body)),
            NS(named_browser_response=lambda body, raw: raw),
            "original",
            evidence,
        )
        choose, _ = b.load_builder(MODEL, QUESTIONS, transport)
        page = FIXTURES["cases"]["actionable"]["page"]
        choose(page, FIXTURES["goal"], [])
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            choose(page, FIXTURES["goal"], [])

    def test_diagnostic_wire_keeps_original_AST_with_only_explicit_profile_binding(self):
        wire_record = record(a.ROOT / "scripts/jev_decision3_browser_wire.py")
        wire, proof = b.load_diagnostic_wire(wire_record, a.PROFILE, a.PROFILE + "-named-answers")
        body = {
            "model": "d3-edge",
            "state": {},
            "questions": {"operation": {"criteria": {"DONE": "done", "BLOCKED": "blocked"}}},
        }
        encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        answer = {"id": "operation", "choice": "BLOCKED"}
        raw = {
            "answers": [answer],
            "decision3_native": {
                "version": a.PROFILE,
                "original_request": body,
                "original_request_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
            },
        }
        result = wire.named_browser_response(body, raw)
        self.assertEqual(result["answers"]["operation"], answer)
        self.assertEqual(result["decision3_browser_wire"]["profile"], a.PROFILE + "-named-answers")
        self.assertTrue(all(row["exact_function_AST"] for row in proof.values()))
        raw["decision3_native"]["version"] = "jev_decision3_native_v1"
        with self.assertRaises(ValueError):
            wire.named_browser_response(body, raw)

    def test_original_native_predict_AST_not_GPU_constructor_extracted(self):
        source_record = record(a.ROOT / "src/s1/jev_decision3_native.py")
        names = (
            "_evidence",
            "parse_native_payload",
            "Decision3NativeRequest",
            "BackendTimeoutError",
            "BackendResponseError",
            "normalize_native_response",
        )
        dto = NS(**dict.fromkeys(names, object()))
        predict, proof = b.load_native_predict(source_record, dto, a.PROFILE, a.profile_manifest)
        tree = ast.parse(Path(source_record["path"]).read_bytes())
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "JevDecision3NativeAdapter"
        )
        original = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "predict"
        )
        self.assertEqual(
            proof["ast_sha256"], hashlib.sha256(ast.dump(original).encode()).hexdigest()
        )
        self.assertNotIn("configuration_blockers", predict.__globals__)
        self.assertEqual(predict.__globals__["PROFILE"], a.PROFILE)

    def test_cpu_constructor_truthful_spec_and_owner_required(self):
        identity = {
            "cgroup": "0::/owned/" + n.UNIT,
            "affinity": [16, 17, 18, 19],
            "environment": h.START_ENV,
        }
        limits = {
            "memory.max": str(n.MEMORY_MAX),
            "memory.swap.max": "0",
            "cpu.max": "400000 100000",
        }
        helper = NS(
            START_ENV=h.START_ENV, process_identity=lambda: identity, owned_limits=lambda _: limits
        )
        spec = NS(
            device="cpu",
            precision="bfloat16",
            context_limit=8192,
            model_id=n.MODELS["edge"]["model_id"],
            revision=n.MODELS["edge"]["revision"],
        )
        bridge = NS(spec=spec, backend=NS(check_runtime=lambda: None))
        adapter = a.make_cpu_adapter(n, helper, bridge, lambda *_: None)
        self.assertIs(adapter.backend, bridge)
        spec.device = "cuda"
        with self.assertRaisesRegex(RuntimeError, "truthful"):
            a.make_cpu_adapter(n, helper, bridge, lambda *_: None)
        spec.device = "cpu"
        limits["memory.max"] = str(n.MEMORY_MAX + 1)
        with self.assertRaisesRegex(RuntimeError, "2GiB"):
            a.make_cpu_adapter(n, helper, bridge, lambda *_: None)

    def test_actual_partial_row_observer_never_pads_or_changes_values(self):
        torch = NS(bfloat16="bf16")
        for rows in (1, 2):
            witness = a.RowWitness(torch, rows)
            hidden = NS(shape=(rows, 135, 1024), dtype="bf16", device=NS(type="cpu"))
            self.assertIsNone(witness(None, [hidden]))
            self.assertEqual(witness.events[0]["shape"], [rows, 135, 1024])
            self.assertFalse(witness.events[0]["inputs_or_outputs_modified"])
            hidden.shape = (4, 135, 1024)
            with self.assertRaises(RuntimeError):
                witness(None, [hidden])

    def test_raw_evidence_secondary_never_replaces_native_primary(self):
        primary = ValueError("native")
        helper = NS(
            write_json=lambda *_: (_ for _ in ()).throw(OSError("evidence")),
            diagnostic=lambda x: repr(x),
        )
        a.persist_local_evidence(helper, Path("/unused"), {"raw": []}, primary)
        self.assertIn("evidence", primary.__notes__[0])

    def test_bridge_checks_actual_DTO_order_before_native_backend_entry(self):
        submitted = {
            "state": {"page": "toy"},
            "questions": {
                "operation": {
                    "type": "choice",
                    "instructions": "goal",
                    "criteria": {"BLOCKED": "no progress", "DONE": "achieved", "CLICK": "advance"},
                }
            },
        }
        entries = []

        class Backend:
            def __init__(self, *_, **__):
                pass

            def prepare(self, **payload):
                return None, {key: 42 for key in payload["questions"]}

            def __call__(self, **payload):
                entries.append(payload)
                return {"model": "d3-edge", "answers": {}}

        class Spec:
            def model_copy(self, *, update):
                return NS(**update)

        provider = NS(
            decision3_spec=lambda *_, **__: Spec(),
            native_payload=lambda request: copy.deepcopy(request),
        )
        native = NS(NativeBatch4Backend=Backend)
        evidence = {"submitted_request": copy.deepcopy(submitted)}
        bridge = a.BrowserBridge(native, None, None, {}, provider, evidence)
        bridge.predict(copy.deepcopy(submitted))
        self.assertEqual(
            list(entries[0]["questions"]["operation"]["criteria"]), ["BLOCKED", "DONE", "CLICK"]
        )
        tampered = copy.deepcopy(submitted)
        tampered["questions"]["operation"]["criteria"] = dict(
            reversed(list(tampered["questions"]["operation"]["criteria"].items()))
        )
        with self.assertRaisesRegex(RuntimeError, "DTO changed actual"):
            bridge.predict(tampered)
        self.assertEqual(len(entries), 1)

    def test_bridge_rejects_full_context_over8192_without_truncation(self):
        body = {
            "state": "toy",
            "questions": {
                "operation": {
                    "type": "choice",
                    "instructions": "goal",
                    "criteria": {"DONE": "achieved", "BLOCKED": "none"},
                }
            },
        }

        class Backend:
            def __init__(self, *_, **__):
                pass

            def prepare(self, **payload):
                return None, {"operation": 8193}

            def __call__(self, **payload):
                raise AssertionError("native must not run")

        provider = NS(
            decision3_spec=lambda *_, **__: NS(model_copy=lambda **__: None),
            native_payload=lambda value: value,
        )
        bridge = a.BrowserBridge(
            NS(NativeBatch4Backend=Backend), None, None, {}, provider, {"submitted_request": body}
        )
        with self.assertRaisesRegex(RuntimeError, "8192"):
            bridge.predict(body)


class TimingTests(unittest.TestCase):
    def run_case(
        self,
        *,
        whole=False,
        exit_delay=0,
        cleanup_error=None,
        body_error=None,
        write_delay=0,
        write_error=None,
        start_clock=0,
    ):
        clock = [start_clock]
        writes = []

        class Deadline:
            def __init__(self, *_):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_):
                clock[0] += exit_delay
                if cleanup_error is not None:
                    raise cleanup_error

        def writer(path, value):
            clock[0] += write_delay
            writes.append(copy.deepcopy(value))
            if write_error is not None and (
                "error" in value or any("error" in row for row in value.get("calls", []))
            ):
                raise write_error

        helper = NS(Deadline=Deadline, write_json=writer, ensure_elapsed=h.ensure_elapsed)

        def finalizer(output, state, primary):
            try:
                writer(output / "state.json", state)
            except BaseException as exc:
                if primary is None:
                    raise
                primary.add_note("fake secondary evidence: " + repr(exc))

        helper.finalize_state = finalizer
        row = {"index": 0, "completed": False}
        state = {"completed": False, "calls": [row], "benchmark_scores": None}

        def work():
            if body_error is not None:
                raise body_error
            return {"matches_self_authored_predicate": False}

        with patch.object(a.time, "monotonic", lambda: clock[0]):
            if whole:
                result = a.finish_whole(
                    n, helper, Path("/unused"), state, 0, 120 - start_clock, work
                )
            else:
                result = a.run_cell(n, helper, Path("/unused"), state, row, work)
        return result, state, row, writes

    def test_call_cleanup_expiry_demotes_after_validation(self):
        with self.assertRaisesRegex(RuntimeError, "after deadline cleanup"):
            self.run_case(exit_delay=21)

    def test_call_durable_write_expiry_refused(self):
        with self.assertRaisesRegex(RuntimeError, "durable evidence"):
            self.run_case(write_delay=11)

    def test_call_cleanup_raise_not_success(self):
        error = KeyboardInterrupt("cleanup")
        with self.assertRaises(KeyboardInterrupt) as caught:
            self.run_case(cleanup_error=error)
        self.assertIs(caught.exception, error)

    def test_call_primary_retained_despite_cleanup_and_evidence_failure(self):
        error = ValueError("native primary")
        with self.assertRaises(ValueError) as caught:
            self.run_case(
                body_error=error,
                cleanup_error=KeyboardInterrupt("cleanup"),
                write_error=OSError("write"),
            )
        self.assertIs(caught.exception, error)
        self.assertGreaterEqual(len(error.__notes__), 2)

    def test_whole_preflight_time_not_reset(self):
        with self.assertRaisesRegex(RuntimeError, "whole canary after"):
            self.run_case(whole=True, start_clock=119, exit_delay=2)

    def test_whole_final_evidence_time_counts(self):
        with self.assertRaisesRegex(RuntimeError, "whole canary after"):
            self.run_case(whole=True, write_delay=121)

    def test_whole_cleanup_raise_demotes(self):
        error = RuntimeError("cleanup")
        with self.assertRaises(RuntimeError) as caught:
            self.run_case(whole=True, cleanup_error=error)
        self.assertIs(caught.exception, error)

    def test_whole_primary_survives_finalwrite_and_cleanup(self):
        error = ValueError("native primary")
        with self.assertRaises(ValueError) as caught:
            self.run_case(
                whole=True,
                body_error=error,
                cleanup_error=KeyboardInterrupt("cleanup"),
                write_error=OSError("write"),
            )
        self.assertIs(caught.exception, error)
        self.assertGreaterEqual(len(error.__notes__), 2)

    def test_success_can_have_zero_semantic_matches_without_score(self):
        _, state, row, _ = self.run_case()
        self.assertTrue(row["completed"])
        self.assertFalse(row["semantic_diagnostic"]["matches_self_authored_predicate"])
        self.assertIsNone(state["benchmark_scores"])

    def test_no_model_imports_in_source_test_process(self):
        self.assertFalse(
            {"torch", "transformers", "safetensors", "numpy", "s1"}.intersection(sys.modules)
        )


if __name__ == "__main__":
    unittest.main()
