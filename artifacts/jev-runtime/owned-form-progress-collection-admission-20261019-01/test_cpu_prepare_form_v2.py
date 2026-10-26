"""Synthetic stdlib tests: no real limits, child, admission, processor or framework."""

import ast
import copy
import hashlib
import importlib.util
import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "_form_cpu_reference_v2_pure", HERE / "cpu_prepare_form_v2.py"
)
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


def observation(available=100 * C.GIB, total=128 * C.GIB, boot="same"):
    return {"MemTotal": total, "MemAvailable": available, "boot_id": boot}


def limits(role):
    return {"role": role, "soft_bytes": C.limit_target(role)[0], "hard_bytes": C.CHILD_AS}


def cuda():
    return {"CUDA_VISIBLE_DEVICES": "", "torch_loaded": False, "CUDA_initialized": False}


def identity(pid=123, start=456, group=123):
    return {
        "pid": pid,
        "ppid": 100,
        "pgid": group,
        "sid": group,
        "start_ticks": start,
        "state": "R",
    }


class FakeResource:
    RLIMIT_AS = 9
    RLIM_INFINITY = -1

    def __init__(self, before=(-1, -1), wrong=False):
        self.value, self.calls, self.wrong = before, [], wrong

    def getrlimit(self, kind):
        assert kind == self.RLIMIT_AS
        return self.value

    def setrlimit(self, kind, value):
        self.calls.append((kind, value))
        self.value = (value[0] - 1, value[1]) if self.wrong else value


def fake_compile():
    cases = []
    for index in range(23):
        raw = C.encode(
            {
                "model": "generated-actions-none",
                "state": {"full": "Moss 星", "unused": ["z", "a"]},
                "questions": {"operation": {}, "click_target": {}, "select_target": {}},
            }
        )
        cases.append(
            types.SimpleNamespace(
                id=f"fake-{index}", wire_sha256=hashlib.sha256(raw).hexdigest(), raw=raw
            )
        )

    class Collection:
        _plan = b'{"sources":{}}'
        train_cases = tuple(cases[:13])
        development_cases = tuple(cases[13:])

        def check(self):
            pass

        def compile_wire_projection(self, split):
            selected = self.train_cases if split == "train" else self.development_cases
            return [
                {
                    "case_id": c.id,
                    "payload": json.loads(c.raw),
                    "teacher_choice": {"operation": "CLICK", "target": "2"},
                    "valid_next_targets": {"CLICK": ["2"]},
                }
                for c in selected
            ]

    compiled, payloads = [], []

    def compile_actions(model, selected, snapshot, *, identity):
        case = selected[0]
        wire = json.loads(snapshot["blobs"]["wire_requests.jsonl"])
        assert C.encode(wire["payload"]) == case.raw
        assert identity == "generated-actions-none"
        compiled.append(case.id)
        return [{"record": {"original_wire_sha256": case.wire_sha256}}]

    def compile_cfg(row, payload, tok):
        payloads.append(copy.deepcopy(payload))
        return {"fake": True}

    kernel = types.SimpleNamespace(compile_actions=compile_actions)
    cfg = types.SimpleNamespace(compile_action_plan=compile_cfg)
    common = types.SimpleNamespace(
        parse=json.loads,
        encode=C.encode,
        IDENTITY="generated-actions-none",
        trusted_module=mock.Mock(side_effect=[kernel, cfg]),
        verify_reference_environment=mock.Mock(),
    )
    forms = types.SimpleNamespace(
        AdmittedFormProgressCollection=Collection,
        common=common,
        authenticate_sources=mock.Mock(),
        COUNTS={"train": 13, "development": 10},
    )
    model = types.SimpleNamespace(
        processor=object(),
        tok=object(),
        max_context=8192,
        lm=types.SimpleNamespace(
            config=types.SimpleNamespace(),
            generation_config=types.SimpleNamespace(eos_token_id=[1, 106, 50]),
        ),
    )
    return forms, model, Collection(), kernel, cfg, compiled, payloads


def fixture_reference(output):
    """Synthetic closed-file joins, explicitly not a genuine processor receipt."""
    ordered = [
        {"id": f"fake-{i}", "split": "train" if i < 13 else "development", "family": "fake"}
        for i in range(23)
    ]
    closure = {"components": {"processor": {"class": "Fake", "path": "/fake"}}, "versions": {}}
    closure_blob = C.encode(closure)
    C.save(output, "reference-environment/processor-source-closure.json", closure_blob)
    evidence = {
        **closure,
        "tokenizer_is_processor_tokenizer": True,
        "max_context": 8192,
        "generation_EOS": [1, 106, 50],
        "CPU_threads": 2,
        "CPU_interop_threads": 2,
    }
    for side in ["before", "after"]:
        C.save(output, f"actual-processor-{side}.json", C.encode(evidence))
    rows = []
    for i, item in enumerate(ordered):
        raw = C.encode({"synthetic_case": item["id"]})
        wire_sha = hashlib.sha256(raw).hexdigest()
        tensors = {"input_ids": [[11, 12, 10, 106]], "attention_mask": [[1, 1, 1, 1]]}
        record = {
            "case_id": item["id"],
            "split": item["split"],
            "original_wire_sha256": wire_sha,
            "normalized_wire_sha256": wire_sha,
            "prompt_tokens": 2,
            "input_tokens": 4,
            "logits_positions": [1, 2],
            "target_token_ids": [10, 106],
            "native_stop_token_id": 106,
            "processor_tensors": tensors,
        }
        row = {"record": record, "inputs": tensors, "positions": [1, 2], "targets": [10, 106]}
        cfg = {
            "compiled_row_sha256": hashlib.sha256(C.encode(row)).hexdigest(),
            "request_sha256": wire_sha,
            "target_ids": [10, 106],
            "positions": [1, 2],
            "allowed_ids": [[10], [106]],
        }
        body = C.save(output, f"compiled/{i:02d}.native-wire.json", raw)
        ref = C.save(
            output,
            f"compiled/{i:02d}.json",
            C.encode({"case": {"id": item["id"]}, "row": row, "cfg_plan": cfg}),
        )
        rows.append(
            {**item, "status": "COMPLETE_REFERENCE", "native_wire": body, "compiled_reference": ref}
        )
    terminal = {
        "schema_id": "owned-form-progress-native-CPU-reference-only-v2",
        "status": C.SUCCESS,
        "nonce": "owned",
        "plan_sha256": "selected",
        "primary_error": None,
        "secondary_errors": [],
        "counts": {"saved_rows": 23, "compiled_rows_in_memory": 23, "compile_returned": True},
        "rows": rows,
        "profile": C.PROFILE,
        "runtime_guard_count": len(C.RUNTIME_GUARDS),
        "containment_limits": C.LIMITS,
    }
    for role in ["parent", "worker"]:
        gate = C.CPUReferenceGate(
            output,
            role + "-resource-gates.jsonl",
            role,
            "same",
            observe=observation,
            limits=limits,
            cuda=cuda,
        )
        for phase in C.phase_sequence(role):
            gate.check(phase, "cpu")
    monitor = C.OwnedMemoryMonitor(
        output,
        "same",
        identity(100, 200, 100),
        observe=observation,
        rss=lambda r: {
            **{k: r[k] for k in ("pid", "start_ticks", "pgid", "sid")},
            "rss_bytes": C.MIB,
        },
    )
    with mock.patch.object(C, "verify_limits", side_effect=limits):
        monitor.check({123: identity()})
    C.save(output, "parent-monitor-identity.json", C.encode(identity(100, 200, 100)))
    C.save(
        output,
        "owned-cleanup.json",
        C.encode(
            {
                "confirmed": True,
                "remaining": [],
                "tracked_PID_start_ticks": {"123": 456},
                "tracked_groups": [123],
            }
        ),
    )
    guards = [{"phase": phase, "limits": limits("worker"), **cuda()} for phase in C.RUNTIME_GUARDS]
    C.save(output, "runtime-guards.json", C.encode(guards))
    C.save(output, "worker-terminal.json", C.encode(terminal))
    return {
        "ordered_cases": ordered,
        "reference_items": {
            "processor-source-closure.json": {
                "sha256": hashlib.sha256(closure_blob).hexdigest(),
                "size_bytes": len(closure_blob),
            }
        },
    }


class CPUReferenceV2Tests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.addCleanup(self.folder.cleanup)

    def gate(self, role="worker", observe=None):
        return C.CPUReferenceGate(
            self.root,
            role + ".jsonl",
            role,
            "same",
            observe=observe or observation,
            limits=limits,
            cuda=cuda,
        )

    def test_fake_limits_parent_child_targets_and_readback(self):
        for role, target in [
            ("parent", (2 * C.GIB, 16 * C.GIB)),
            ("worker", (16 * C.GIB, 16 * C.GIB)),
        ]:
            with self.subTest(role=role):
                api = FakeResource()
                evidence = C.configure_limits(role, api=api)
                self.assertEqual(api.calls, [(api.RLIMIT_AS, target)])
                self.assertEqual((evidence["soft_bytes"], evidence["hard_bytes"]), target)

    def test_fake_limits_refuse_lower_hard_or_readback_drift(self):
        api = FakeResource(before=(C.GIB, C.GIB))
        with self.assertRaisesRegex(ValueError, "cannot be raised"):
            C.configure_limits("worker", api=api)
        self.assertEqual(api.calls, [])
        with self.assertRaisesRegex(ValueError, "readback"):
            C.configure_limits("worker", api=FakeResource(wrong=True))
        with self.assertRaises(ValueError):
            C.limit_target("train")

    def test_formula_preserves_original_floor_and_adds18GiB_without_credit(self):
        for total in [100 * C.GIB + 3, 128 * C.GIB]:
            item = observation(total=total)
            f = C.resource_formula(item, "same", prospective=True)
            reserve = max(24 * C.GIB, (total + 4) // 5)
            self.assertEqual(f["reserve_bytes"], reserve)
            self.assertEqual(f["original_CPU_floor_bytes"], reserve + 4 * C.GIB)
            self.assertEqual(f["required_bytes"], reserve + 18 * C.GIB)
            self.assertEqual(f["swap_or_resident_credit_bytes"], 0)

    def test_formula_refuses_wrong_boot_boolean_impossible_or_extra_observations(self):
        for mutation in ["boot", "boolean", "impossible", "extra"]:
            item = observation()
            if mutation == "boot":
                item["boot_id"] = "foreign"
            elif mutation == "boolean":
                item["MemAvailable"] = True
            elif mutation == "impossible":
                item["MemAvailable"] = item["MemTotal"] + 1
            else:
                item["resident_credit"] = C.GIB
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                C.resource_formula(item, "same", prospective=True)

    def test_all_exact_worker_phases_and_fresh_gate_journal(self):
        gate = self.gate()
        for phase in C.WORKER_PHASES:
            gate.check(phase, "cpu")
        blob = (self.root / "worker.jsonl").read_bytes()
        self.assertEqual(C.validate_gate_evidence(blob, "worker"), "same")
        rows = [C.parse(line) for line in blob.splitlines()]
        self.assertEqual(
            len([r for r in rows if r["phase"].startswith("cpu-reference-compile-")]), 23
        )
        with self.assertRaises(FileExistsError):
            self.gate()

    def test_gate_refuses_train_load_forward_reorder_duplicate_or_wrong_scope(self):
        gate = self.gate()
        for phase, scope in [
            ("before-native-train-forward", "train"),
            ("before-checkpoint-payload-hash-and-BF16-loader", "load"),
            ("before-BF16-processor-compile-train-0", "train"),
            (C.WORKER_PHASES[1], "cpu"),
            (C.WORKER_PHASES[0], "train"),
        ]:
            with (
                self.subTest(phase=phase, scope=scope),
                self.assertRaisesRegex(ValueError, "phase/scope"),
            ):
                gate.check(phase, scope)
        gate.check(C.WORKER_PHASES[0], "cpu")
        with self.assertRaises(ValueError):
            gate.check(C.WORKER_PHASES[0], "cpu")

    def test_gate_refusal_is_durable_and_primary_survives_IO_failure(self):
        gate = self.gate(observe=lambda: observation(available=30 * C.GIB))
        with self.assertRaisesRegex(ValueError, "memory refused"):
            gate.check(C.WORKER_PHASES[0], "cpu")
        row = C.parse((self.root / "worker.jsonl").read_bytes())
        self.assertFalse(row["admitted"])
        self.assertEqual(row["error"]["phase"], "CPU-reference-gate")
        with self.assertRaisesRegex(ValueError, "memory refused") as caught:
            with mock.patch.object(C, "append_owned", side_effect=OSError("synthetic disk error")):
                gate.check(C.WORKER_PHASES[1], "cpu")
        self.assertIn("persistence failed", caught.exception.__notes__[0])

    def test_gate_saved_refuses_missing_bool_sequence_and_formula_drift(self):
        gate = self.gate("parent")
        for phase in C.PARENT_PHASES:
            gate.check(phase, "cpu")
        rows = [C.parse(line) for line in (self.root / "parent.jsonl").read_bytes().splitlines()]
        for mutation in ["missing", "bool", "formula", "scope", "boot"]:
            bad = copy.deepcopy(rows)
            if mutation == "missing":
                bad.pop()
            elif mutation == "bool":
                bad[0]["sequence"] = False
            elif mutation == "formula":
                bad[0]["required_bytes"] -= 1
            elif mutation == "scope":
                bad[0]["scope"] = "train"
            else:
                bad[-1]["expected_boot_id"] = "foreign"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                C.validate_gate_evidence(b"".join(C.encode(r) + b"\n" for r in bad), "parent")

    def test_cuda_guards_do_not_query_initialize_or_require_real_torch(self):
        fake = types.SimpleNamespace(cuda=types.SimpleNamespace(_initialized=False))
        with (
            mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}),
            mock.patch.dict(C.sys.modules, {"torch": fake}),
        ):
            self.assertTrue(C.cuda_state()["torch_loaded"])
            fake.cuda._initialized = True
            with self.assertRaisesRegex(ValueError, "uninitialized"):
                C.cuda_state()
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            with self.assertRaisesRegex(ValueError, "hidden"):
                C.cuda_state()

    def monitor(self, *, observe=observation, rss=None):
        rss = rss or (
            lambda r: {
                **{k: r[k] for k in ("pid", "start_ticks", "pgid", "sid")},
                "rss_bytes": C.GIB,
            }
        )
        return C.OwnedMemoryMonitor(
            self.root, "same", identity(100, 200, 100), observe=observe, rss=rss
        )

    def test_monitor_counts_parent_and_owned_tree_and_validates_sameboot_journal(self):
        monitor = self.monitor()
        with mock.patch.object(C, "verify_limits", side_effect=limits):
            row = monitor.check({123: identity()})
        self.assertEqual(row["tracked_RSS_bytes"], 2 * C.GIB)
        self.assertEqual(
            C.validate_monitor_evidence(
                (self.root / "owned-memory-monitor.jsonl").read_bytes(),
                "same",
                parent=identity(100, 200, 100),
                cleanup={
                    "confirmed": True,
                    "remaining": [],
                    "tracked_PID_start_ticks": {"123": 456},
                    "tracked_groups": [123],
                },
            ),
            1,
        )

    def test_monitor_refuses_host_floor_or_tree_budget_and_persists_refusal(self):
        monitor = self.monitor(observe=lambda: observation(available=20 * C.GIB))
        with (
            mock.patch.object(C, "verify_limits", side_effect=limits),
            self.assertRaisesRegex(ValueError, "host reserve"),
        ):
            monitor.check({123: identity()})
        self.assertFalse(
            C.parse((self.root / "owned-memory-monitor.jsonl").read_bytes())["admitted"]
        )
        (self.root / "owned-memory-monitor.jsonl").unlink()
        monitor = self.monitor(
            rss=lambda r: {
                **{k: r[k] for k in ("pid", "start_ticks", "pgid", "sid")},
                "rss_bytes": 10 * C.GIB,
            }
        )
        with (
            mock.patch.object(C, "verify_limits", side_effect=limits),
            self.assertRaisesRegex(ValueError, "18GiB RSS"),
        ):
            monitor.check({123: identity()})

    def test_monitor_unknown_identity_is_not_zero_RSS_credit_and_preserves_primary(self):
        monitor = self.monitor(rss=mock.Mock(side_effect=ProcessLookupError("synthetic PID gone")))
        with self.assertRaises(ProcessLookupError):
            monitor.check({123: identity()})
        record = C.parse((self.root / "owned-memory-monitor.jsonl").read_bytes())
        self.assertFalse(record["admitted"])
        with (
            mock.patch.object(C, "append_owned", side_effect=OSError("disk")),
            self.assertRaises(ProcessLookupError) as caught,
        ):
            monitor.check({123: identity()})
        self.assertIn("persistence failed", caught.exception.__notes__[0])

    def test_process_RSS_joins_PID_start_PG_SID_and_rejects_reuse(self):
        proc = self.root / "proc"
        folder = proc / "123"
        folder.mkdir(parents=True)
        tail = ["R", "100", "123", "123"] + ["0"] * 16
        tail[19] = "456"
        (folder / "stat").write_text("123 (synthetic process) " + " ".join(tail))
        (folder / "status").write_text("Name:\tfake\nVmRSS:\t1024 kB\n")
        self.assertEqual(C.process_rss(identity(), proc)["rss_bytes"], C.MIB)
        for field in ["start_ticks", "pid", "pgid", "sid"]:
            wrong = identity()
            wrong[field] += 1
            with self.subTest(field=field), self.assertRaises((ValueError, FileNotFoundError)):
                C.process_rss(wrong, proc)

    def test_new_adapter_routes_all23_exact_full_wires_and_CFG_without_old_adapter(self):
        forms, model, collection, _, _, compiled, payloads = fake_compile()
        gate = types.SimpleNamespace(check=mock.Mock())
        with mock.patch.object(C, "runtime_guard", side_effect=lambda p: {"phase": p}):
            rows, plans, guards = C.compile_references_v2(forms, model, collection, (), gate)
        self.assertEqual(len(rows), 23)
        self.assertEqual(len(plans), 23)
        self.assertEqual(len(guards), 23)
        self.assertEqual(compiled, [f"fake-{i}" for i in range(23)])
        self.assertEqual(
            [c.args for c in gate.check.call_args_list], [(phase, "cpu") for phase in C.ROW_PHASES]
        )
        self.assertTrue(
            all(
                p["state"]["unused"] == ["z", "a"] and "select_target" in p["questions"]
                for p in payloads
            )
        )
        forms.common.verify_reference_environment.assert_called_once()

    def test_new_adapter_refuses_LM_objects_or_wrong_SHA_without_CFG_or_selection(self):
        forms, model, collection, kernel, cfg, _, _ = fake_compile()
        model.lm = object()
        with self.assertRaises(ValueError):
            C.compile_references_v2(forms, model, collection, (), None)
        forms, model, collection, kernel, cfg, _, _ = fake_compile()
        kernel.compile_actions = lambda *a, **k: [{"record": {"original_wire_sha256": "0" * 64}}]
        cfg.compile_action_plan = mock.Mock()
        with self.assertRaisesRegex(ValueError, "ordered form"):
            C.compile_references_v2(
                forms, model, collection, (), types.SimpleNamespace(check=lambda *a: None)
            )
        cfg.compile_action_plan.assert_not_called()

    def test_release_strict_identity_ignores_only_scheduling_state(self):
        actual = identity()
        release = {
            "nonce": "owned",
            "plan_sha256": "selected",
            "identity": {**actual, "state": "S"},
        }
        C.verify_worker_binding(release, actual, "owned", "selected", 123)
        for field in ["pid", "ppid", "pgid", "sid", "start_ticks"]:
            bad = copy.deepcopy(release)
            bad["identity"][field] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                C.verify_worker_binding(bad, actual, "owned", "selected", 123)

    def test_parent_refusal_never_starts_child_and_retains_all23(self):
        plan = {
            "ordered_cases": [{"id": str(i)} for i in range(23)],
            "python": "/fake/not-executed",
        }
        gate = types.SimpleNamespace(
            check=mock.Mock(side_effect=ValueError("synthetic resource refusal"))
        )
        output = self.root / "fresh"
        with (
            mock.patch.object(C, "load_plan", return_value=(plan, {})),
            mock.patch.object(C, "configure_limits", return_value={"synthetic": True}),
            mock.patch.object(C, "cuda_state", return_value=cuda()),
            mock.patch.object(C, "resource_gate", return_value=gate),
            mock.patch.object(
                C.subprocess, "Popen", side_effect=AssertionError("must not spawn")
            ) as spawn,
        ):
            self.assertEqual(C.controller(Path("fake-plan"), "selected", output, run=True), 2)
        result = C.parse((output / "controller-terminal.json").read_bytes())
        self.assertEqual(result["primary_error"]["message"], "synthetic resource refusal")
        self.assertFalse(result["process_started"])
        self.assertIsNone(result["reference_ready"])
        self.assertEqual(len(result["all23_rows"]), 23)
        self.assertTrue(all(r["status"] == "NOT_EXECUTED" for r in result["all23_rows"]))
        spawn.assert_not_called()

    def test_terminal_reserve_and_write_failure_preserve_primary(self):
        with (
            mock.patch.object(C, "MAX_BYTES", 100),
            mock.patch.object(C, "MAX_FILE", 80),
            mock.patch.object(C, "RESERVE", 20),
        ):
            C.save(self.root, "payload", b"x" * 75)
            with self.assertRaises(ValueError):
                C.save(self.root, "ordinary", b"y" * 10)
            C.save(self.root, "terminal", b"z" * 10, terminal=True)
            self.assertEqual((self.root / "payload").read_bytes(), b"x" * 75)
        primary = {"type": "ValueError", "message": "original"}
        secondary = []
        value = {"primary_error": primary, "secondary_errors": secondary}
        with mock.patch.object(C, "save", side_effect=OSError("synthetic IO failure")):
            self.assertFalse(C.terminal(self.root, "failed-terminal", value, secondary))
        self.assertIs(value["primary_error"], primary)
        self.assertEqual(secondary[0]["phase"], "terminal-publication")

    def test_old_source_invariants_and_successor_no_native_train_or_old_adapter_calls(self):
        pins = {
            HERE
            / "cpu_prepare_form.py": "11f8cd050afd65c2fc2a6ecb47425a1b1eb91d6840d6ff56c08ba6dedc361c2a",
            C.ROOT
            / "scripts/jev_owned_form_progress.py": "5211edc548683af254b652bf2e291f3c3f7cc0ab972295324ad40e54a46ba571",
            C.ROOT
            / "scripts/jev_public_action_artifact.py": "dbd1eb660d3bebc8dbc0b3f7e58b7bac66f5ab0a4ea9d687dc463ce830f1672d",
            C.ROOT
            / "scripts/jev_browser_fit_actions.py": "4bf6585008e434a469dafb4a9997dee595b079044e03c9afe00ceffa5e031f82",
            C.ROOT
            / "scripts/jev_cfg_action_loss.py": "b422f055cfe22c8d4b6c34780c189d8c78f21bb78e5ea564e67cd9ab79e80aad",
        }
        for path, digest in pins.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        tree = ast.parse((HERE / "cpu_prepare_form_v2.py").read_bytes())
        forbidden = {
            "compile_form_references",
            "UnifiedDecisionModel",
            "train_actions",
            "action_loss",
            "forward",
            "ResourceGate",
        }
        calls = [
            n.func.attr
            if isinstance(n.func, ast.Attribute)
            else n.func.id
            if isinstance(n.func, ast.Name)
            else ""
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
        ]
        self.assertFalse(forbidden & set(calls))
        self.assertEqual(len(C.ROW_PHASES), 23)
        self.assertTrue(
            all("BF16" not in phase and "forward" not in phase for phase in C.WORKER_PHASES)
        )
        original = json.loads(
            (
                C.ROOT
                / "artifacts/jev-runtime/public-native-action-actual07-fit-20261016-01/source-handoff-03.json"
            ).read_bytes()
        )["files"]
        for label, pin in original.items():
            blob = (C.ROOT / label).read_bytes()
            self.assertEqual(
                {"sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)}, pin
            )

    def test_complete_saved23_reference_and_new_profile_gates(self):
        plan = fixture_reference(self.root)
        result = C.select_reference(self.root, plan, "selected", "owned")
        self.assertEqual(result["compiled_rows"], 23)
        self.assertEqual(result["profile"], C.PROFILE)
        self.assertEqual(result["monitor_rows"], 1)

    def test_saved_reference_rejects_partial_body_EOS_CFG_and_new_runtime_evidence(self):
        for mutation in [
            "denominator",
            "partial",
            "source",
            "body",
            "EOS",
            "CFG",
            "gates",
            "CUDA",
            "profile",
        ]:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as folder:
                output = Path(folder)
                plan = fixture_reference(output)
                terminal_path = output / "worker-terminal.json"
                terminal = C.parse(terminal_path.read_bytes())
                if mutation == "denominator":
                    terminal["rows"].pop()
                elif mutation == "partial":
                    terminal["counts"]["compile_returned"] = False
                elif mutation == "source":
                    terminal["plan_sha256"] = "foreign"
                elif mutation == "profile":
                    terminal["profile"] = "train"
                elif mutation == "gates":
                    path = output / "worker-resource-gates.jsonl"
                    path.write_bytes(b"\n".join(path.read_bytes().splitlines()[:-1]) + b"\n")
                elif mutation == "CUDA":
                    path = output / "runtime-guards.json"
                    guards = C.parse(path.read_bytes())
                    guards[1]["CUDA_initialized"] = True
                    path.write_bytes(C.encode(guards))
                else:
                    item = terminal["rows"][0]
                    if mutation == "body":
                        (output / item["native_wire"]["path"]).write_bytes(b"changed")
                    else:
                        ref = item["compiled_reference"]
                        path = output / ref["path"]
                        data = C.parse(path.read_bytes())
                        if mutation == "EOS":
                            data["row"]["targets"][-1] = 99
                        else:
                            data["cfg_plan"]["allowed_ids"][0] = [99]
                        blob = C.encode(data)
                        path.write_bytes(blob)
                        ref.update(sha256=hashlib.sha256(blob).hexdigest(), size_bytes=len(blob))
                terminal_path.write_bytes(C.encode(terminal))
                with self.assertRaises(ValueError):
                    C.select_reference(output, plan, "selected", "owned")

    def test_compile_failure_does_not_skip_failed_row_or_return_success_subset(self):
        forms, model, collection, _, _, compiled, _ = fake_compile()
        calls = []

        def guard(phase):
            calls.append(phase)
            if phase == "after-row-3":
                raise ValueError("synthetic CUDA/AS drift")
            return {"phase": phase}

        with (
            mock.patch.object(C, "runtime_guard", side_effect=guard),
            self.assertRaisesRegex(ValueError, "drift"),
        ):
            C.compile_references_v2(
                forms, model, collection, (), types.SimpleNamespace(check=lambda *a: None)
            )
        self.assertEqual(compiled, ["fake-0", "fake-1", "fake-2", "fake-3"])
        self.assertEqual(len(calls), 4)

    def test_worker_compile_failure_retains_all23_unknown_and_primary(self):
        ordered = [
            {"id": f"fake-{i}", "split": "train" if i < 13 else "development", "family": "fake"}
            for i in range(23)
        ]
        cases = [
            types.SimpleNamespace(
                id=item["id"], split=item["split"], annotation=lambda: {"family": "fake"}
            )
            for item in ordered
        ]
        collection = types.SimpleNamespace(
            train_cases=tuple(cases[:13]), development_cases=tuple(cases[13:]), provenance={}
        )
        forms = types.SimpleNamespace(
            validate_form_progress_collection=mock.Mock(return_value=collection)
        )
        plan = {
            "python": "/synthetic/python",
            "ordered_cases": ordered,
            "environment": {},
            "admission_plan": {"sha256": "fake"},
        }
        C.save(
            self.root,
            "controller-intent.json",
            C.encode({"nonce": "owned", "plan_sha256": "selected"}),
        )
        pid = C.os.getpid()
        actual = identity(pid, 456, pid)
        C.save(
            self.root,
            "controller-release.json",
            C.encode({"nonce": "owned", "plan_sha256": "selected", "identity": actual}),
        )
        owned = types.SimpleNamespace(process_table=lambda: {pid: actual})
        with (
            mock.patch.object(C, "load_plan", return_value=(plan, {})),
            mock.patch.object(C.sys, "version_info", (3, 13)),
            mock.patch.object(C.sys, "executable", plan["python"]),
            mock.patch.object(C, "configure_limits", return_value={"synthetic": True}),
            mock.patch.object(C, "cuda_state", return_value=cuda()),
            mock.patch.object(C, "ownership", return_value=owned),
            mock.patch.object(
                C, "resource_gate", return_value=types.SimpleNamespace(check=lambda *a: None)
            ),
            mock.patch.object(C, "authenticate_sources"),
            mock.patch.object(C, "module", return_value=forms),
            mock.patch.object(C, "checked", return_value=b'{"sources":{}}'),
            mock.patch.object(C, "capture_reference", return_value=()),
            mock.patch.object(C, "load_processor", return_value=object()),
            mock.patch.object(C, "processor_evidence", return_value={}),
            mock.patch.object(C, "runtime_guard", return_value={"synthetic": True}),
            mock.patch.object(
                C,
                "compile_references_v2",
                side_effect=ValueError("original synthetic compile error"),
            ),
        ):
            self.assertEqual(C.worker(Path("fake-plan"), "selected", self.root, "owned"), 1)
        result = C.parse((self.root / "worker-terminal.json").read_bytes())
        self.assertEqual(result["primary_error"]["message"], "original synthetic compile error")
        self.assertTrue(result["counts"]["compile_started"])
        self.assertFalse(result["counts"]["compile_returned"])
        self.assertIsNone(result["counts"]["compiled_rows_in_memory"])
        self.assertEqual(result["counts"]["saved_rows"], 0)
        self.assertEqual(len(result["rows"]), 23)
        self.assertTrue(all(r["status"] == "COMPILE_RETURN_UNKNOWN" for r in result["rows"]))

    def test_monitor_closed_evidence_rejects_foreign_parent_or_unowned_PID_start(self):
        monitor = self.monitor()
        with mock.patch.object(C, "verify_limits", side_effect=limits):
            monitor.check({123: identity()})
        blob = (self.root / "owned-memory-monitor.jsonl").read_bytes()
        parent = identity(100, 200, 100)
        cleanup = {
            "confirmed": True,
            "remaining": [],
            "tracked_PID_start_ticks": {"123": 456},
            "tracked_groups": [123],
        }
        for mutation in ["parent", "start", "group", "not_closed"]:
            p, c = copy.deepcopy(parent), copy.deepcopy(cleanup)
            if mutation == "parent":
                p["start_ticks"] += 1
            elif mutation == "start":
                c["tracked_PID_start_ticks"]["123"] += 1
            elif mutation == "group":
                c["tracked_groups"] = [124]
            else:
                c["confirmed"] = False
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                C.validate_monitor_evidence(blob, "same", parent=p, cleanup=c)

    def test_final_launch_metadata_is_separate_exact105_and_dry_never_sets_limits(self):
        plan_path = HERE / "CPU-v2-launch-plan-02.json"
        blob = plan_path.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        plan, files = C.load_plan(plan_path, digest)
        self.assertEqual(len(files), 105)
        self.assertNotIn((HERE / "cpu_prepare_form.py").relative_to(C.ROOT).as_posix(), files)
        with (
            mock.patch.object(
                C, "configure_limits", side_effect=AssertionError("must not set real limits")
            ) as setter,
            mock.patch.object(
                C.subprocess, "Popen", side_effect=AssertionError("must not spawn")
            ) as spawn,
        ):
            result = C.controller(plan_path, digest, self.root / "dry-never-created", run=False)
        self.assertEqual(result["fixed_rows"], 23)
        self.assertFalse((self.root / "dry-never-created").exists())
        setter.assert_not_called()
        spawn.assert_not_called()
        for mutation in ["profile", "phase", "environment", "extra"]:
            bad = copy.deepcopy(plan)
            if mutation == "profile":
                bad["profile"] = "train"
            elif mutation == "phase":
                bad["worker_phases"][5] = "before-native-train-forward"
            elif mutation == "environment":
                bad["environment"]["PYTHONPATH"] = "/foreign"
            else:
                bad["waiver"] = True
            raw = C.encode(bad)
            path = self.root / (mutation + ".json")
            path.write_bytes(raw)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                C.load_plan(path, hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    unittest.main(verbosity=2)
