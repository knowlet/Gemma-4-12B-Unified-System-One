"""Pure fake/source tests. No actual admission, processor, browser or model."""

import ast
import copy
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import jev_owned_form_progress as F  # noqa: E402

OPTIONAL_ROOT = ROOT
OPTIONAL_NATIVE_SOURCES = {
    F.BASE
    + "/collector_core.py": "665232daa0671e0c8f2ed1a3adf73802c39df5cf576f3763d8a3250ab79f8323",
    F.MATRIX
    + "/native-request-source/model.py": "a85ada8458d23642c5c8a4095acf0565e6268f81dcda26f6308942b653e9c9f3",
    F.MATRIX
    + "/native-request-source/questions.py": "66a7054b85040b949e1757311257ea0c5a843c635f9a40c99f6d15ad660fb422",
    F.MATRIX
    + "/runtime/v5/snapshot.js": "17a2e57362f9654abc8acea94a714dcaf6945594ba2ac0d57b8494f0b458a9a1",
}


def optional_evidence(label, expected_sha):
    path = OPTIONAL_ROOT / label
    assert not path.is_symlink(), "Optional evidence must be an ordinary file"
    if not path.exists():
        raise unittest.SkipTest("Optional closed research evidence is absent in this checkout")
    blob = path.read_bytes()
    assert F.common.sha(blob) == expected_sha, "Pinned optional research evidence changed"
    return blob


def fake_case(index, split, family, operation="CLICK"):
    payload = {
        "model": "generated-actions-none",
        "state": {"unmodified": "Moss 星", "unused": ["z", "a"]},
        "questions": {"operation": {}, "click_target": {}, "select_target": {}},
    }
    raw = json.dumps(payload, ensure_ascii=False, indent=1).encode()
    decision = {
        "id": f"fake-{index}",
        "selected": {"operation": operation, "target": "2:1" if operation == "SELECT" else "2"},
    }
    wire = {
        "native_wire_utf8": raw.decode(),
        "native_wire_sha256": F.common.sha(raw),
        "native_wire_size_bytes": len(raw),
        "proof_refs": {
            "guarded_transition": {
                "artifact_path": "separate-text-proof",
                "sequence": 0,
                "sha256": "a" * 64,
            }
        },
    }
    return F._case(decision, wire, {"id": f"episode-{index}", "family": family, "split": split})


def fake_collection():
    specs, cases = [], []
    sizes = [2, 1, 2, 1, 2, 1, 3, 1, 3, 2, 3, 2]
    for i, count in enumerate(sizes):
        split = "train" if i < 8 else "development"
        family = f"family-{i // 2}"
        specs.append({"id": f"scene-{i}", "family": family, "split": split})
        offset = len(cases)
        cases.extend(fake_case(offset + j, split, family) for j in range(count))
    snapshot = F._Snapshot((("fake-only", b"retained fake bytes"),))
    cases = tuple(cases)
    plan = F.common.encode({"sources": {}})
    provenance = b"{}"
    seal = F.common.sha(
        plan
        + b"\0"
        + provenance
        + b"\0"
        + F.common.encode([c._seal for c in cases])
        + snapshot.digest().encode()
    )
    return F.AdmittedFormProgressCollection(
        snapshot, cases, plan, provenance, F._PROOF, seal
    ), specs


def fake_episode(*, counter=False):
    """Synthetic IO/schema joins; DOM semantic verifier is an explicit stub."""
    for label, expected_sha in OPTIONAL_NATIVE_SOURCES.items():
        optional_evidence(label, expected_sha)
    real = F._core()
    constructor, space = F.native_constructor()
    action = {
        "id": "e1",
        "node": 1,
        "kind": "click",
        "role": "button",
        "label": "Commit draft",
        "value": "",
    }
    pages = []
    for text in ["Draft view", "Applied view"]:
        page = {
            "url": "http://127.0.0.1:4321/s/fake/index.html",
            "title": "Fake schema test",
            "text": text,
            "actions": [action],
            "scroll": {"y": 0, "height": 780},
        }
        page["fingerprint"] = real.fingerprint(page)
        pages.append(page)
    private = [
        {"url": pages[0]["url"], "done": False},
        {"url": pages[1]["url"], "done": not counter},
    ]
    resolution = {"selected": {"operation": "CLICK", "target": "1"}, "action": action}
    history = [{"action": action["label"], "kind": "click", "text": None, "page_changed": True}]
    expression = (ROOT / F.MATRIX / "runtime/v5/snapshot.js").read_text()
    private_expression = "fake-private-only-expression"
    calls, observations = [], []
    for i, page in enumerate(pages):
        if i:
            for kind in ["mousePressed", "mouseReleased"]:
                calls.append(
                    {
                        "method": "Input.dispatchMouseEvent",
                        "session": "owned-matrix-fixture",
                        "params": {"type": kind},
                        "response": {},
                    }
                )
        response = {"result": {"value": {k: v for k, v in page.items() if k != "fingerprint"}}}
        calls.append(
            {
                "method": "Runtime.evaluate",
                "session": "owned-matrix-fixture",
                "params": {"expression": expression},
                "response": response,
            }
        )
        observations.append(
            {
                "protocol_call": len(calls),
                "session": "owned-matrix-fixture",
                "response": copy.deepcopy(response),
            }
        )
        calls.append(
            {
                "method": "Runtime.evaluate",
                "session": "owned-matrix-fixture",
                "params": {"expression": private_expression},
                "response": {"result": {"value": private[i]}},
            }
        )
    inputs = [c for c in calls if c["method"].startswith("Input.")]
    transition = {
        "resolution": resolution,
        "before": private[0],
        "after": private[1],
        "fresh": True,
        "executor_result": {"executed": "e1"},
        "call_range": [2, 6],
        "inputs": inputs,
        "text": None,
        "before_fingerprint": pages[0]["fingerprint"],
        "after_fingerprint": pages[1]["fingerprint"],
    }
    requests = []
    for i, page in enumerate(pages):
        payload = constructor(page, "Keep the requested draft applied.", [] if i == 0 else history)
        raw = F.common.encode(payload)
        requests.append(
            {
                "sequence": i,
                "phase": "final_proof_only" if i else "counterexample" if counter else "teacher",
                "native_payload": payload,
                "native_wire_utf8": raw.decode(),
                "native_wire_sha256": F.common.sha(raw),
                "raw_observation_sequence": i,
                "compiled": None,
                "compilation_status": "NOT_PERFORMED",
            }
        )
    proof = {
        "schema_id": real.PROOF_KIND,
        "case_id": "fake",
        "goal": "Keep the requested draft applied.",
        "admitted": False,
        "compiled": False,
        "semantic_complete": not counter,
        "requests": requests,
        "native_pages": pages,
        "private_states": private,
        "transitions": [transition],
        "history": history,
        "decisions": [] if counter else [{"fake_placeholder": True}],
    }
    row = {
        "case_id": "fake",
        "counterexample": counter,
        "status": "COMPLETE_WRONG_DOCUMENT_COUNTEREXAMPLE"
        if counter
        else "COMPLETE_NATIVE_ROUTE_UNADMITTED",
        "cleanup_errors": [],
        "blocked": [],
        "page_errors": [],
        "console": [],
        "admitted": False,
        "network": [],
        "calls": calls,
        "inputs": inputs,
        "ownership": {"fake": True},
        "events": [],
        "observations": observations,
        "proof": proof,
    }
    core = types.SimpleNamespace(
        PROOF_KIND=real.PROOF_KIND,
        PRIVATE=private_expression,
        encode=real.encode,
        fingerprint=real.fingerprint,
        assert_initial=mock.Mock(),
        document=mock.Mock(),
        resolve_teacher=mock.Mock(return_value=resolution),
        verify_native_transition=mock.Mock(),
        semantic_complete=lambda _, state, __: state["done"],
        assert_demonstration=mock.Mock(),
    )
    spec = {
        "id": "fake",
        "goal": proof["goal"],
        "teacher_steps": [
            {"operation": "CLICK", "document": "index.html", "private_dom_id": "commit"}
        ],
    }
    return row, core, spec, constructor, space


def snapshot_for(row):
    proof = row["proof"]
    prefix = "episodes/00"
    arrays = {
        "native-cdp.jsonl": row["calls"],
        "observations.jsonl": row["observations"],
        "ownership.jsonl": [row["ownership"]],
        "local-request-ledger.jsonl": [{"responses": row["network"], "blocked": row["blocked"]}],
        "progress-events.jsonl": row["events"],
        "native-requests.jsonl": proof["requests"],
        "transitions.jsonl": proof["transitions"],
        "private-dom.jsonl": proof["private_states"],
    }
    items = {
        prefix + "/" + k: b"".join(F.common.encode(v) + b"\n" for v in values)
        for k, values in arrays.items()
    }
    items[prefix + "/proof.json"] = F.common.encode(row) + b"\n"
    bodies = []
    for i, request in enumerate(proof["requests"]):
        label = prefix + f"/native-request-bodies/{i:02d}.json"
        items[label] = request["native_wire_utf8"].encode()
        bodies.append(
            {"path": label, "sha256": F.common.sha(items[label]), "size_bytes": len(items[label])}
        )
    summary = {
        "ordinal": 0,
        "case_id": "fake",
        "counterexample": row["counterexample"],
        "status": row["status"],
        "proof": {
            "path": prefix + "/proof.json",
            "sha256": F.common.sha(items[prefix + "/proof.json"]),
            "size_bytes": len(items[prefix + "/proof.json"]),
        },
        "journals": {
            k: {"path": prefix + "/" + k, "sha256": F.common.sha(items[prefix + "/" + k])}
            for k in arrays
        },
        "native_wire_bodies": bodies,
    }
    items[prefix + ".json"] = F.common.encode(summary)
    return F._Snapshot(tuple(sorted(items.items())))


class FormProgressTests(unittest.TestCase):
    def test_optional_evidence_skips_missing_but_refuses_corruption_or_symlink(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "optional.py"
            with mock.patch.dict(globals(), OPTIONAL_ROOT=Path(folder)):
                with self.assertRaises(unittest.SkipTest):
                    optional_evidence("optional.py", "a" * 64)
                path.write_bytes(b"corrupt")
                with self.assertRaisesRegex(AssertionError, "evidence changed"):
                    optional_evidence("optional.py", "a" * 64)
                path.unlink()
                path.symlink_to(Path(folder) / "missing-target")
                with self.assertRaisesRegex(AssertionError, "ordinary file"):
                    optional_evidence("optional.py", "a" * 64)

    def test_snapshot_accepts_frozen_Path_order_and_refuses_population_or_hash_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(strict=True)
            proof = root / F.BASE / "actual-01"
            proof.mkdir(parents=True)
            names = ["episodes/00/trace.json", "episodes/00.json"] + [
                f"leaf-{i:03d}.json" for i in range(234)
            ]
            for name in names:
                path = proof / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
            # Exercise the fixed complete population contract without reading
            # or admitting any actual browser evidence.
            (proof / names[-1]).write_bytes(b"x" * (11_438_498 - 235))
            refs = [
                {"path": p.relative_to(root).as_posix(), **F.common.pin(p)}
                for p in sorted(proof.rglob("*"))
                if p.is_file()
            ]
            paths = [r["path"] for r in refs]
            self.assertNotEqual(paths, sorted(paths))
            index = {
                "schema": "root-closed-owned-form-native-proof-index-v1",
                "ordinary_file_count": 236,
                "total_bytes": 11_438_498,
                "files": refs,
            }
            with mock.patch.object(F, "ROOT", root):
                snapshot = F.capture_snapshot(index)
                self.assertEqual(
                    [name for name, _ in snapshot.items],
                    [Path(p).relative_to(Path(F.BASE) / "actual-01").as_posix() for p in paths],
                )
                extra = proof / "unexpected.json"
                extra.write_bytes(b"x")
                with self.assertRaisesRegex(ValueError, "Missing/extra"):
                    F.capture_snapshot(index)
                extra.unlink()
                missing = proof / names[0]
                missing.unlink()
                with self.assertRaisesRegex(ValueError, "Missing/extra"):
                    F.capture_snapshot(index)
                missing.write_bytes(b"x")
                duplicate = copy.deepcopy(index)
                duplicate["files"][1] = duplicate["files"][0]
                with self.assertRaisesRegex(ValueError, "Unique complete index"):
                    F.capture_snapshot(duplicate)
                missing.write_bytes(b"y")
                with self.assertRaisesRegex(ValueError, "reference drift"):
                    F.capture_snapshot(index)

    def test_original_package_population_has_no_new_module_waiver(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            package = root / "src/s1"
            package.mkdir(parents=True)
            (package / "a.py").write_text("# original source\n")
            with mock.patch.object(F, "ROOT", root):
                F.authenticate_original_package_population({"src/s1/a.py": {}})
                (package / "extra.py").write_text("# unregistered\n")
                with self.assertRaises(ValueError):
                    F.authenticate_original_package_population({"src/s1/a.py": {}})

    def test_new_case_type_and_original_TARGETS_not_legacy_coercion(self):
        for operation, head in F._targets().items():
            case = fake_case(0, "train", "family", operation)
            self.assertIs(type(case), F.FormNativeActionCase)
            self.assertEqual(
                case.gold, {"operation": operation, head: "2:1" if operation == "SELECT" else "2"}
            )
            self.assertIsNone(case.annotation()["compiled_input_sequence"])
            self.assertNotIn("text", case.gold)
            self.assertFalse(case.annotation()["public_registry_admitted"])

    def test_request_calls_original_NativeWireRequest_with_exact_bytes(self):
        case = fake_case(1, "development", "family")
        validator = mock.Mock(return_value=object())
        module = types.ModuleType("s1.browser_public_training")
        module.NativeWireRequest = types.SimpleNamespace(model_validate_json=validator)
        package = types.ModuleType("s1")
        package.__path__ = []
        with mock.patch.dict(sys.modules, {"s1": package, "s1.browser_public_training": module}):
            result = case.request
        self.assertIs(result, validator.return_value)
        validator.assert_called_once_with(case._wire)
        self.assertNotEqual(case._wire, F.common.encode(F.common.parse(case._wire)))

    def test_case_mutation_and_forged_proof_rejected(self):
        case = fake_case(0, "train", "family")
        object.__setattr__(case, "_wire", b"{}")
        with self.assertRaises(ValueError):
            case.check()
        forged = F.FormNativeActionCase(b"{}", b"{}", object(), F.common.sha(b"{}\0{}"))
        with self.assertRaises(ValueError):
            forged.check()

    def test_all23_whole_family_split_and_projection_preserve_full_payload(self):
        collection, specs = fake_collection()
        F._validate_split(collection._cases, specs)
        self.assertEqual((len(collection.train_cases), len(collection.development_cases)), (13, 10))
        projection = collection.compile_wire_projection("train")
        self.assertEqual(projection[0]["payload"], F.common.parse(collection.train_cases[0]._wire))
        self.assertEqual(projection[0]["payload"]["state"]["unused"], ["z", "a"])
        self.assertEqual(projection[0]["valid_next_targets"], {"CLICK": ["2"]})
        self.assertNotIn("gold", projection[0]["payload"])

    def test_family_leakage_duplicate_or_missing_state_rejected(self):
        collection, specs = fake_collection()
        bad = copy.deepcopy(specs)
        bad[0]["split"] = "development"
        with self.assertRaises(ValueError):
            F._validate_split(collection._cases, bad)
        with self.assertRaises(ValueError):
            F._validate_split(collection._cases[:-1], specs)
        with self.assertRaises(ValueError):
            F._validate_split(collection._cases[:-1] + collection._cases[:1], specs)

    def test_original_constructor_has_all_heads_and_no_client_import(self):
        row, _, _, _, _ = fake_episode()
        payload = row["proof"]["requests"][0]["native_payload"]
        self.assertEqual(payload["model"], "generated-actions-none")
        self.assertEqual(list(payload["questions"]), ["operation", "click_target"])
        self.assertEqual(payload["state"]["recent_actions"], [])
        self.assertNotIn("teacher", F.common.encode(payload).decode())

    def test_fake_complete_episode_checks_every_raw_and_native_join(self):
        row, core, spec, constructor, space = fake_episode()
        F._verify_episode(snapshot_for(row), 0, spec, False, core, constructor, space)
        core.assert_initial.assert_called_once()
        core.resolve_teacher.assert_called_once()
        core.verify_native_transition.assert_called_once()
        core.assert_demonstration.assert_called_once()

    def test_wronglocal_clone_semantic_false_is_proof_only(self):
        row, core, spec, constructor, space = fake_episode(counter=True)
        F._verify_episode(snapshot_for(row), 0, spec, True, core, constructor, space)
        core.assert_demonstration.assert_not_called()
        self.assertEqual(row["proof"]["decisions"], [])

    def test_rebuilt_body_refuses_goal_history_head_pruning_and_reorder(self):
        for mutation in ["goal", "history", "head", "order"]:
            row, core, spec, constructor, space = fake_episode()
            request = row["proof"]["requests"][0]
            if mutation == "goal":
                spec["goal"] = "Different hidden goal"
            elif mutation == "history":
                request["native_payload"]["state"]["recent_actions"] = [{"action": "invented"}]
            elif mutation == "head":
                del request["native_payload"]["questions"]["click_target"]
            else:
                request["native_payload"] = dict(reversed(list(request["native_payload"].items())))
            if mutation != "goal":
                raw = F.common.encode(request["native_payload"])
                request.update(native_wire_utf8=raw.decode(), native_wire_sha256=F.common.sha(raw))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                F._verify_episode(snapshot_for(row), 0, spec, False, core, constructor, space)

    def test_false_fresh_wrong_snapshot_method_and_semantic_final_rejected(self):
        for mutation in ["fresh", "snapshot", "final", "Inputerror", "text"]:
            row, core, spec, constructor, space = fake_episode()
            if mutation == "fresh":
                row["proof"]["transitions"][0]["fresh"] = False
            elif mutation == "snapshot":
                row["calls"][0]["params"]["expression"] = "fake raw state"
            elif mutation == "final":
                row["proof"]["semantic_complete"] = False
            elif mutation == "text":
                row["proof"]["transitions"][0]["text"] = "unexpected instruction"
            else:
                row["calls"][2]["error"] = {"type": "Incomplete"}
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                F._verify_episode(snapshot_for(row), 0, spec, False, core, constructor, space)

    def test_exact_source_union_refuses_extra_missing_old_mutated_or_review_drift(self):
        original = optional_evidence(F.ORIGINAL, F.ORIGINAL_SHA)
        old = F.common.parse(original)["files"]
        pins = {
            k: old.get(k, {"sha256": "a" * 64, "size_bytes": 1})
            for k in sorted(set(old) | F.EXTRAS)
        }
        plan = {"original_source_handoff_utf8": original.decode(), "sources": pins}
        with mock.patch.object(
            F.common, "pin", side_effect=lambda p: pins[p.relative_to(ROOT).as_posix()]
        ):
            F.authenticate_sources(plan, pins)
            for mutation in ["extra", "missing", "old", "review"]:
                bad = copy.deepcopy(plan)
                reviewed = copy.deepcopy(pins)
                if mutation == "extra":
                    bad["sources"]["scripts/unknown.py"] = {"sha256": "b" * 64, "size_bytes": 1}
                elif mutation == "missing":
                    del bad["sources"][next(iter(bad["sources"]))]
                elif mutation == "old":
                    bad["sources"][next(iter(old))] = {"sha256": "c" * 64, "size_bytes": 1}
                    reviewed = bad["sources"]
                else:
                    reviewed["scripts/jev_owned_form_progress.py"] = {
                        "sha256": "d" * 64,
                        "size_bytes": 1,
                    }
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    F.authenticate_sources(bad, reviewed)

    def test_fresh_gate_required_before_invalid_kind_without_admission(self):
        raw = b"{}"
        gate = types.SimpleNamespace(check=mock.Mock())
        with self.assertRaises(ValueError):
            F.validate_form_progress_collection(
                raw, expected_plan_sha256=F.common.sha(raw), reviewed_source_pins={}, gate=gate
            )
        gate.check.assert_called_once_with("before-any-project-framework-or-parent-payload", "cpu")
        self.assertEqual(F.common.resource_phase_scope(gate.check.call_args.args[0]), "cpu")

    def test_fake_compile_routes_all23_through_original_kernel_and_full_CFG_payload(self):
        collection, _ = fake_collection()
        compiled, cfg_payloads = [], []

        def compile_actions(_model, cases, snapshot, *, identity):
            case = cases[0]
            wire = F.common.parse(snapshot["blobs"]["wire_requests.jsonl"].strip())
            self.assertEqual(wire["payload"], F.common.parse(case._wire))
            self.assertEqual(identity, "generated-actions-none")
            compiled.append(case.id)
            return [{"record": {"case_id": case.id, "original_wire_sha256": case.wire_sha256}}]

        kernel = types.SimpleNamespace(compile_actions=compile_actions)
        cfg = types.SimpleNamespace(
            compile_action_plan=lambda row, payload, tok: (
                cfg_payloads.append(payload) or {"fake_plan_for": row["record"]["case_id"]}
            )
        )
        gate = types.SimpleNamespace(check=mock.Mock())
        with (
            mock.patch.object(F, "authenticate_sources"),
            mock.patch.object(F.common, "verify_reference_environment") as origins,
            mock.patch.object(F.common, "trusted_module", side_effect=[kernel, cfg]),
        ):
            rows, plans = F.compile_form_references(
                types.SimpleNamespace(tok="fake-not-tokenizer"), collection, (), gate=gate
            )
        self.assertEqual((len(rows), len(plans)), (23, 23))
        self.assertEqual(
            compiled, [c.id for c in collection.train_cases + collection.development_cases]
        )
        self.assertEqual(len(cfg_payloads), 23)
        origins.assert_called_once()
        self.assertEqual(gate.check.call_count, 23)
        for call in gate.check.call_args_list:
            self.assertEqual(F.common.resource_phase_scope(call.args[0]), call.args[1])

    def test_compile_rejects_old_collection_before_processor_and_wrong_wire_SHA(self):
        with mock.patch.object(F.common, "verify_reference_environment") as origins:
            with self.assertRaises(ValueError):
                F.compile_form_references(object(), object(), (), gate=None)
            origins.assert_not_called()
        collection, _ = fake_collection()
        kernel = types.SimpleNamespace(
            compile_actions=lambda *_a, **_k: [{"record": {"original_wire_sha256": "0" * 64}}]
        )
        cfg = types.SimpleNamespace(compile_action_plan=mock.Mock())
        with (
            mock.patch.object(F, "authenticate_sources"),
            mock.patch.object(F.common, "verify_reference_environment"),
            mock.patch.object(F.common, "trusted_module", side_effect=[kernel, cfg]),
            self.assertRaises(ValueError),
        ):
            F.compile_form_references(
                types.SimpleNamespace(tok=None),
                collection,
                (),
                gate=types.SimpleNamespace(check=lambda *_: None),
            )
        cfg.compile_action_plan.assert_not_called()

    def test_import_is_stdlib_plus_original_pure_common_only(self):
        tree = ast.parse((ROOT / "scripts/jev_owned_form_progress.py").read_bytes())
        imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        modules = {
            n.module.split(".")[0] if isinstance(n, ast.ImportFrom) else a.name.split(".")[0]
            for n in imports
            for a in (n.names if isinstance(n, ast.Import) else [types.SimpleNamespace(name="")])
        }
        self.assertFalse(
            modules & {"torch", "transformers", "tokenizers", "playwright", "s1", "peft"}
        )


if __name__ == "__main__":
    unittest.main()
