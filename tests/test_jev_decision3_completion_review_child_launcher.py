"""Synthetic launcher wiring, no browser or real model."""

import json
import sys
import time
import types
from pathlib import Path

import jev_decision3_completion_review_child_launcher as launcher
import jev_decision3_completion_review_route as route
import jev_decision3_semantic_browser_binding as binding
import pytest
from test_jev_decision3_completion_review_route import review_payload

pytest_plugins = ("test_jev_decision3_semantic_browser_binding",)


def test_owned_launcher_connects_real_service_and_restores(context, monkeypatch):
    monkeypatch.setattr(launcher, "SOURCE_ONLY", False)
    monkeypatch.setattr(route, "SOURCE_ONLY", False)
    instance = context.make()
    child = types.ModuleType("synthetic_owned_child")
    child.__file__ = __file__
    child.instance = instance
    child.body = json.dumps(review_payload()).encode()
    child.clock = time.monotonic
    exec(
        "def _serve(plan, plan_sha, invocation, backend, torch, modules, output, check):\n"
        "    response, row = instance.service.handle(body, started=clock())\n"
        "    instance.service.delivery(row, delivered=True)\n"
        "    return response\n",
        child.__dict__,
    )

    def load(released, held, pins, name):
        assert released == str(route_release)
        assert held == "jev_decision3_completion_review_route.py"
        assert name not in sys.modules
        sys.modules[name] = route
        return route

    child.common = types.SimpleNamespace(
        _load_reviewed_release=load, _verify_source_pins=lambda _pins: None
    )
    route_release = Path(route.__file__).with_name(launcher.ROUTE_RELEASE_NAME)
    pins = {
        str(route_release): launcher.ROUTE_RELEASE_SHA,
        route.__file__: launcher.ROUTE_HELD_SHA,
        child.__file__: "b" * 64,
    }
    plan = {"source_pins": pins}
    modules = (None, None, binding, None, None, None)
    original = child._serve
    try:
        with launcher.attach_owned_child(child, route_release):
            result = child._serve(plan, "sha", "invocation", None, None, modules, None, None)
            assert list(json.loads(result)["answers"]) == [route.HEAD]
            assert context.bindings[0]["original_request"] == review_payload()
            assert "_candidate_completion_review_route" not in sys.modules
        assert child._serve is original
    finally:
        instance.close()


def test_launcher_source_gate_and_explicit_gate_only():
    held = b"before\nSOURCE_ONLY = True\nafter\n"
    launcher._release(held, held.replace(b"True", b"False"))
    with pytest.raises(ValueError, match="gate-only"):
        launcher._release(held, held.replace(b"after", b"changed"))
    with pytest.raises(ValueError, match="independent review"):
        with launcher.attach_owned_child(None, Path("/unused")):
            pass
