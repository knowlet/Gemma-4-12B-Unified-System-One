"""Synthetic profile/API boundaries only; no model, browser or network execution.

API tests require the separately reviewed API patch to be applied. During the
active campaign this file is source only; no tests/imports have been run.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import inspect
import json
import os
import sys
import tempfile
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from s1 import jev_action_profiles as profiles
from s1.errors import RequestValidationError

EXPERIMENTAL = "jev-find-text-experimental-v1"


def request():
    operations = (
        "CLICK",
        "TYPE_TEXT",
        "SELECT",
        "SCROLL_DOWN",
        "SCROLL_UP",
        "WAIT",
        "DONE",
        "BLOCKED",
        "FIND_TEXT",
    )
    questions = {
        "operation": {
            "type": "choice",
            "instructions": "Synthetic operation choice",
            "criteria": {key: "Synthetic offered operation" for key in operations},
        }
    }
    for head in ("click_target", "type_text_target", "select_target"):
        questions[head] = {
            "type": "choice",
            "instructions": "Synthetic target choice",
            "criteria": {"candidate": "Synthetic offered target"},
        }
    return {
        "state": {"goal": "Synthetic control-flow fixture"},
        "questions": questions,
        "model": "synthetic",
    }


def legacy_response(payload, profile):
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return {
        "model": "synthetic",
        "answers": [
            {
                "id": key,
                "choice": next(iter(value["criteria"])),
                "probabilities": {label: 1 / len(value["criteria"]) for label in value["criteria"]},
            }
            for key, value in payload["questions"].items()
        ],
        "decision3_native": {
            "version": profile,
            "original_request": copy.deepcopy(payload),
            "original_request_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        },
    }


class ProtocolTests(unittest.TestCase):
    def test_profile_names_and_unknown_values_are_explicit(self):
        self.assertEqual(profiles.ACTION_PROFILES, ("standard", EXPERIMENTAL))
        self.assertEqual(profiles.DEFAULT_ACTION_PROFILE, "standard")
        for value in (None, True, 1, "", "find_text", "STANDARD"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                profiles.resolve_protocol(value)

    def test_standard_returns_original_classes_without_replacement(self):
        native = importlib.import_module("s1.jev_decision3_native")
        generated = importlib.import_module("s1.jev_generated")
        result = profiles.resolve_protocol("standard")
        self.assertIs(result.native_adapter_class, native.JevDecision3NativeAdapter)
        self.assertIs(result.generated_adapter_class, generated.JevGeneratedActionAdapter)
        self.assertEqual(result.provenance["operation_count"], 8)
        self.assertEqual(result.provenance["native_profile"], "jev_decision3_native_v1")
        self.assertFalse(result.provenance["original_modules_mutated"])

    def test_experimental_does_not_mutate_original_or_frozen_canary_namespaces(self):
        originals = {name: importlib.import_module("s1." + name) for name in profiles._ORIGINALS}
        before = {
            name: {
                key: module.__dict__[key]
                for key in ("PROFILE", "OPERATIONS", "UNTARGETED_OPERATIONS")
                if key in module.__dict__
            }
            for name, module in originals.items()
        }
        old_names = {
            "s1._experimental_find_" + name: sys.modules.get("s1._experimental_find_" + name)
            for name in profiles._ORIGINALS
        }
        result = profiles.resolve_protocol(EXPERIMENTAL)
        self.assertEqual(result.provenance["operation_count"], 9)
        self.assertEqual(len(result.provenance["source_dependencies"]), 7)
        self.assertEqual(len({row["path"] for row in result.provenance["source_dependencies"]}), 7)
        for name, module in originals.items():
            self.assertIs(sys.modules["s1." + name], module)
            for key, value in before[name].items():
                self.assertIs(module.__dict__[key], value)
        for name, value in old_names.items():
            self.assertIs(sys.modules.get(name), value)

    def test_all_nine_operations_and_original_singleton_heads_are_parsed_intact(self):
        result = profiles.resolve_protocol(EXPERIMENTAL)
        native = sys.modules[result.native_adapter_class.__module__]
        payload = request()
        parsed, original, source, labels = native.parse_native_payload(
            payload, model_identity="synthetic"
        )
        self.assertEqual(original, payload)
        self.assertEqual(json.loads(source), payload)
        self.assertEqual(list(labels), list(payload["questions"]))
        self.assertEqual(len(labels["operation"]), 9)
        self.assertEqual([len(q.criteria) for q in parsed.questions], [9, 1, 1, 1])
        standard = importlib.import_module("s1.jev_decision3_native")
        with self.assertRaises(RequestValidationError):
            standard.parse_native_payload(payload, model_identity="synthetic")

    def test_native255_bound_still_rejects256_without_mutating_request(self):
        result = profiles.resolve_protocol(EXPERIMENTAL)
        native = sys.modules[result.native_adapter_class.__module__]
        payload = request()
        payload["questions"]["click_target"]["criteria"] = {
            str(i): "Synthetic candidate" for i in range(256)
        }
        before = copy.deepcopy(payload)
        with self.assertRaises(RequestValidationError):
            native.parse_native_payload(payload, model_identity="synthetic")
        self.assertEqual(payload, before)

    def test_private_grammar_remains_available_for_deferred_import_and_same_limits(self):
        result = profiles.resolve_protocol(EXPERIMENTAL)
        source = result.provenance["sources"]["jev_action_grammar"]
        grammar = importlib.import_module(source["private_module"])
        self.assertIn("FIND_TEXT", grammar.UNTARGETED_OPERATIONS)
        labels = {key: list(value["criteria"]) for key, value in request()["questions"].items()}
        inventory = {}

        class Tokenizer:
            all_special_ids = [106]

            def encode(self, text, add_special_tokens=False):
                inventory.setdefault(text, len(inventory) + 1000)
                return [inventory[text]]

        trie = grammar.LegalActionTokenTrie(labels, Tokenizer(), prompt_length=1)
        find = '{"operation":"FIND_TEXT","target":null}'
        self.assertEqual(
            json.loads(trie.assert_complete([inventory[find], 106])),
            {"operation": "FIND_TEXT", "target": None},
        )
        self.assertEqual(
            (trie.max_new_tokens, trie.max_legal_actions, trie.max_encoded_tokens),
            (32, 1205, 38560),
        )
        self.assertIn(source["private_module"], sys.modules)
        with self.assertRaises(ValueError):
            trie.assert_complete([inventory[find]])

    def test_concurrent_standard_and_experimental_resolution_keeps_stable_classes(self):
        def resolve(index):
            profile = "standard" if index % 2 else EXPERIMENTAL
            return profile, profiles.resolve_protocol(profile)

        with ThreadPoolExecutor(max_workers=4) as pool:
            values = list(pool.map(resolve, range(16)))
        for profile, result in values:
            expected = profiles.resolve_protocol(profile)
            self.assertIs(result.native_adapter_class, expected.native_adapter_class)
            self.assertIs(result.generated_adapter_class, expected.generated_adapter_class)
        self.assertIsNot(values[0][1].provenance, values[2][1].provenance)
        values[0][1].provenance["sources"].clear()
        self.assertEqual(len(profiles.resolve_protocol(EXPERIMENTAL).provenance["sources"]), 3)

    def test_registered_private_module_replacement_is_refused(self):
        result = profiles.resolve_protocol(EXPERIMENTAL)
        name = result.provenance["sources"]["jev_generated"]["private_module"]
        with patch.dict(sys.modules, {name: types.ModuleType(name)}):
            with self.assertRaisesRegex(ValueError, "registry changed"):
                profiles.resolve_protocol(EXPERIMENTAL)
        self.assertIs(
            profiles.resolve_protocol(EXPERIMENTAL).generated_adapter_class,
            result.generated_adapter_class,
        )

    def test_wire_profile_pair_and_answer_values_are_preserved(self):
        payload = request()
        for profile, native_profile in (
            ("standard", "jev_decision3_native_v1"),
            (EXPERIMENTAL, "jev_decision3_native_find_text_experimental_v1"),
        ):
            with self.subTest(profile=profile):
                legacy = legacy_response(payload, native_profile)
                before = copy.deepcopy(legacy)
                converted = profiles.named_browser_response(profile, payload, legacy)
                self.assertEqual(list(converted["answers"]), list(payload["questions"]))
                self.assertEqual(list(converted["answers"].values()), legacy["answers"])
                self.assertEqual(legacy, before)
                self.assertFalse(converted["decision3_browser_wire"]["answer_values_changed"])
                other = EXPERIMENTAL if profile == "standard" else "standard"
                with self.assertRaises(ValueError):
                    profiles.named_browser_response(other, payload, legacy)
                legacy["answers"].reverse()
                with self.assertRaises(ValueError):
                    profiles.named_browser_response(profile, payload, legacy)

    def test_standard_agent_identity_and_experimental_subclass_are_separate(self):
        class BaseAgent:
            pass

        agent = NS(Agent=BaseAgent, MAX_STEPS=16, StalePage=ValueError)
        model = NS(NEXT_ACTION="Synthetic instruction", post_json=lambda *a: None)
        self.assertIs(
            profiles.make_agent_class("standard", agent, model, marker_expression="null"), BaseAgent
        )
        derived = profiles.make_agent_class(EXPERIMENTAL, agent, model, marker_expression="null")
        self.assertTrue(issubclass(derived, BaseAgent))
        self.assertEqual(derived.experimental_profile, EXPERIMENTAL)
        self.assertIs(agent.Agent, BaseAgent)
        self.assertFalse(hasattr(model, "action_space"))
        agent.MAX_STEPS = 60
        canonical = profiles.make_agent_class(EXPERIMENTAL, agent, model, marker_expression="null")
        self.assertEqual(agent.MAX_STEPS, 60)
        self.assertIs(agent.Agent, BaseAgent)
        command = inspect.getclosurevars(canonical.command).nonlocals["original_command"]
        self.assertEqual(command.__globals__["MAX_STEPS"], 16)
        agent.MAX_STEPS = 17
        with self.assertRaises(ValueError):
            profiles.make_agent_class(EXPERIMENTAL, agent, model, marker_expression="null")
        with self.assertRaises(ValueError):
            profiles.make_agent_class(EXPERIMENTAL, agent, model, marker_expression="")

    def test_changed_source_is_rejected_before_component_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory).resolve() / "source.py"
            path.write_text("raise AssertionError('must never execute')\n")
            with self.assertRaisesRegex(ValueError, "source changed"):
                profiles._checked_source(path, "0" * 64)


class FakeHTTPException(Exception):
    def __init__(self, status_code, detail):
        self.status_code, self.detail = status_code, detail


class FakeApp:
    def __init__(self, **kwargs):
        self.state, self.routes = NS(), {}

    def route(self, method, path):
        def register(function):
            self.routes[method, path] = function
            return function

        return register

    def get(self, path, **kwargs):
        return self.route("GET", path)

    def post(self, path, **kwargs):
        return self.route("POST", path)


class APIProfileTests(unittest.TestCase):
    """Actual create_app control flow, with stdlib fake app/adapters and no HTTP."""

    def setUp(self):
        candidate = os.environ.get("JEV_ACTION_PROFILES_API_CANDIDATE_SHA256")
        if candidate:
            path = Path("/tmp/jev_action_profiles_api_candidate.py")
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != candidate:
                raise ValueError("Explicit API candidate SHA differs")
            api = types.ModuleType("s1._action_profiles_api_candidate_tests")
            api.__file__, api.__package__ = str(path), "s1"
            exec(compile(raw, str(path), "exec", dont_inherit=True), vars(api))
        else:
            from s1 import api
        self.api = api
        fastapi = NS(
            FastAPI=FakeApp,
            Depends=lambda function: function,
            Header=lambda **kwargs: None,
            HTTPException=FakeHTTPException,
        )
        self.fake_fastapi = patch.dict(sys.modules, {"fastapi": fastapi})
        self.fake_fastapi.start()
        self.addCleanup(self.fake_fastapi.stop)
        self.backend = NS(name="synthetic")

    def test_default_health_is_exact_and_does_not_resolve_new_protocol(self):
        with patch.object(
            profiles, "resolve_protocol", side_effect=AssertionError("default must keep old path")
        ):
            app = self.api.create_app(self.backend)
        self.assertEqual(app.routes["GET", "/healthz"](), {"status": "ok", "model": "synthetic"})

    def test_unsupported_experimental_combinations_and_unknown_profiles_refuse(self):
        cases = (
            {},
            {"jev_compatible": True},
            {"jev_compatible": True, "jev_generated_actions": True, "jev_decision3_native": True},
            {"jev_compatible": True, "jev_generated_actions": True, "jev_hierarchical": True},
            {
                "jev_compatible": True,
                "jev_decision3_native": True,
                "jev_browser_operation_first": True,
            },
        )
        with patch.object(
            profiles, "resolve_protocol", side_effect=AssertionError("must refuse first")
        ):
            for options in cases:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    self.api.create_app(self.backend, jev_action_profile=EXPERIMENTAL, **options)
        for value in (None, True, "find_text"):
            with self.assertRaises(ValueError):
                self.api.create_app(self.backend, jev_action_profile=value)

    def test_native_and_generated_factories_receive_full_request_and_report_opt_in(self):
        for native in (False, True):
            with self.subTest(native=native):
                calls, evidence = [], []

                class Adapter:
                    def __init__(self, backend, **kwargs):
                        calls.append(("constructor", backend, kwargs))

                    def predict(self, payload, **kwargs):
                        calls.append(("predict", payload, kwargs))
                        return {"model": "synthetic", "answers": {}}

                provenance = {"action_profile": EXPERIMENTAL, "operation_count": 9}
                selected = profiles.ActionProtocol(Adapter, Adapter, provenance)
                with (
                    patch.object(profiles, "resolve_protocol", return_value=selected),
                    patch.dict(
                        sys.modules, {"s1.backends": NS(normalize_response=lambda *a: None)}
                    ),
                ):
                    app = self.api.create_app(
                        self.backend,
                        jev_compatible=True,
                        jev_decision3_native=native,
                        jev_generated_actions=not native,
                        jev_generated_action_grammar=not native,
                        jev_action_profile=EXPERIMENTAL,
                        request_evidence_sink=evidence.append,
                    )
                    payload = request()
                    app.routes["POST", "/v1/systemone"](payload)
                predicted = next(row for row in calls if row[0] == "predict")
                self.assertIs(predicted[1], payload)
                health = app.routes["GET", "/healthz"]()
                self.assertEqual(health["action_profile"], EXPERIMENTAL)
                self.assertEqual(health["action_profile_provenance"], provenance)
                self.assertEqual(
                    next(row for row in evidence if row["event"] == "request_started")["data"][
                        "action_profile"
                    ],
                    EXPERIMENTAL,
                )
                health["action_profile_provenance"].clear()
                self.assertEqual(selected.provenance, provenance)

    def test_standard_and_experimental_apps_keep_independent_health(self):
        class Adapter:
            def __init__(self, *a, **kwargs):
                pass

        standard = self.api.create_app(self.backend)
        with patch.object(
            profiles,
            "resolve_protocol",
            return_value=profiles.ActionProtocol(
                Adapter, Adapter, {"action_profile": EXPERIMENTAL}
            ),
        ):
            experimental = self.api.create_app(
                self.backend,
                jev_compatible=True,
                jev_generated_actions=True,
                jev_action_profile=EXPERIMENTAL,
            )
        self.assertNotIn("action_profile", standard.routes["GET", "/healthz"]())
        self.assertEqual(experimental.routes["GET", "/healthz"]()["action_profile"], EXPERIMENTAL)


if __name__ == "__main__":
    unittest.main()
