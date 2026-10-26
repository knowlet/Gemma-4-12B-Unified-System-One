"""Invented stdlib fixtures only; not executed by this source-only author."""

import copy
import importlib.util
import unittest
from pathlib import Path
from unittest import mock

PATH = Path(__file__).with_name("actor_held.py")
SPEC = importlib.util.spec_from_file_location("_held_coherence_resume_source_fakes", PATH)
actor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(actor)


def invented():
    def ref(name):
        return {
            "path": str(actor.HERE / ("invented-" + name + ".json")),
            "sha256": "a" * 64,
            "size_bytes": 1,
        }

    rows = [
        {"request_fingerprint": f"{index:064x}", "payload_sha256": "b" * 64}
        for index in range(actor.TOTAL)
    ]
    manifest = {
        "schema": actor.INPUT_SCHEMA,
        "status": actor.ACCEPTED,
        "original_backend_identity": actor.BACKEND,
        "settings": dict(actor.SETTINGS),
        "accepted_prior_count": 1,
        "accepted_entry_shards": [ref("shard")],
        "request_manifest_ref": ref("requests"),
        "root_prefix_audit_ref": ref("audit"),
        "prior_closure_ref": ref("closure"),
        "prior_source_union_ref": ref("union"),
        "original_full06_plan_ref": {
            **ref("old-plan"),
            "path": str(actor.FULL06 / "edge-plan.json"),
            "sha256": actor.ORIGINAL_PLAN_SHA,
        },
        "benchmark_source_plan_ref": {**ref("benchmark"), "sha256": actor.BENCHMARK_SHA},
    }
    requests = {"schema": "decision3-original-coherence1248-request-manifest-v1", "requests": rows}
    return manifest, requests


class HeldResumeFakes(unittest.TestCase):
    def pure_paths(self):
        return self.enterContext(mock.patch.object(actor, "ordinary", side_effect=Path))

    def test_source_hold_precedes_argument_parsing_and_artifact_reads(self):
        with (
            mock.patch.object(actor, "SOURCE_ONLY", True),
            mock.patch.object(actor, "read_small", side_effect=AssertionError("artifact IO")),
            mock.patch.object(actor, "import_selected", side_effect=AssertionError("import")),
        ):
            with self.assertRaisesRegex(actor.ResumeAbort, "SOURCE_ONLY"):
                actor.main(["--execute"])

    def test_input_pin_remains_a_separate_hold(self):
        with mock.patch.object(actor, "SOURCE_ONLY", False):
            with self.assertRaisesRegex(actor.ResumeAbort, "input SHA remains pending"):
                actor.released()

    def test_valid_inventory_uses_explicit_audited_count(self):
        manifest, requests = invented()
        self.pure_paths()
        self.assertEqual(actor.validate_manifest(manifest, requests), requests["requests"])
        manifest["accepted_prior_count"] = 1234
        self.assertEqual(len(actor.validate_manifest(manifest, requests)), 1248)

    def test_invalid_or_progress_only_counts_refuse(self):
        self.pure_paths()
        for count in (None, True, 0, 1248, 1234.0):
            with self.subTest(count=count):
                manifest, requests = invented()
                manifest["accepted_prior_count"] = count
                with self.assertRaisesRegex(actor.ResumeAbort, "audited cache count"):
                    actor.validate_manifest(manifest, requests)

    def test_unknown_or_duplicate_original_request_refuses(self):
        self.pure_paths()
        for mutation in (
            lambda r: r["requests"].pop(),
            lambda r: r["requests"].__setitem__(1, copy.deepcopy(r["requests"][0])),
            lambda r: r["requests"][0].update(payload_sha256="invalid"),
        ):
            manifest, requests = invented()
            mutation(requests)
            with self.assertRaises(actor.ResumeAbort):
                actor.validate_manifest(manifest, requests)

    def test_backend_or_settings_cannot_mix_cpu5_delta_or_modified_suite(self):
        self.pure_paths()
        for mutation in (
            lambda m: m.update(
                original_backend_identity="d3-edge-dsl01-operation-delta-cpu5-candidate"
            ),
            lambda m: m["settings"].update(concurrency=2),
            lambda m: m["settings"].update(seed=1),
            lambda m: m["settings"].update(diagnostics=True),
        ):
            manifest, requests = invented()
            mutation(manifest)
            with self.assertRaisesRegex(actor.ResumeAbort, "original backend/settings"):
                actor.validate_manifest(manifest, requests)

    def test_request_fingerprint_preserves_option_and_question_order(self):
        q1 = {
            "one": {"type": "choice", "criteria": {"a": "A", "b": "B"}},
            "two": {"type": "noul", "instructions": "invented"},
        }
        q2 = dict(reversed(list(q1.items())))
        key1, payload1 = actor.request_identity("invented", q1)
        key2, payload2 = actor.request_identity("invented", q2)
        self.assertNotEqual(key1, key2)
        self.assertNotEqual(payload1, payload2)
        q3 = copy.deepcopy(q1)
        q3["one"]["criteria"] = {"b": "B", "a": "A"}
        self.assertNotEqual(actor.request_identity("invented", q3)[0], key1)

    def test_probability_hash_does_not_normalize_or_repair(self):
        answers = {
            "one": {"probabilities": {"a": 0.2, "b": 0.8}, "choice": "b"},
            "two": {"noul": 0.7},
        }
        self.assertEqual(
            actor.probability_sha(answers),
            actor.digest(actor.compact({"one": {"a": 0.2, "b": 0.8}, "two": 0.7})),
        )
        changed = copy.deepcopy(answers)
        changed["one"]["probabilities"]["a"] = 0.20001
        self.assertNotEqual(actor.probability_sha(changed), actor.probability_sha(answers))


if __name__ == "__main__":
    unittest.main()
