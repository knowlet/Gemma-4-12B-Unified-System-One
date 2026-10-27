"""Held opt-in launcher for the existing owned semantic candidate child.

The original child, model, transport, action parser, witness and supervision
remain in charge. This launcher only installs the reviewed completion-review
route for that child's listener, after its original source checks have run.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path

SOURCE_ONLY = True
CHILD_HELD_SHA = "d7f46bbe12b87dbea1cf39bdf460331f32c527fb00889bde7478c7c4015b2e1e"
CHILD_RELEASE_SHA = "2e7142ef2b3dbcb29d4bab97552d5663fe6b6244916058d8599259cf94ecb8ec"
ROUTE_HELD_SHA = "01642cb14972e1381b7873f4f7e588e7b0ceddcb69281bc47c2e8a1e466250c8"
ROUTE_RELEASE_SHA = "3966b952d21851de7b3e5e0fec2603150948e5a8d83b2321b2b041f1517827ca"
ROUTE_RELEASE_NAME = "jev_decision3_completion_review_route_pending_review_release_20261027.py"


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _read(path, expected):
    _need(
        path.is_absolute() and path.is_file() and path.resolve() == path, "ordinary source required"
    )
    raw = path.read_bytes()
    _need(
        0 < len(raw) <= 1024 * 1024 and hashlib.sha256(raw).hexdigest() == expected,
        "source pin differs",
    )
    return raw


def _release(held, released):
    marker = b"SOURCE_ONLY" + b" = True"
    disabled = b"SOURCE_ONLY" + b" = False"
    _need(
        held.count(marker) == 1 and released == held.replace(marker, disabled, 1),
        "gate-only source release required",
    )


@contextmanager
def attach_owned_child(child, route_release):
    """Defer route import until the pinned child has bound its native modules."""
    _need(SOURCE_ONLY is False, "held review child launcher requires independent review")
    original = child._serve
    _need(
        callable(original) and original.__globals__ is child.__dict__,
        "original child callback required",
    )

    def serve(plan, plan_sha, invocation, backend, torch, modules, output, check):
        pins = plan["source_pins"]
        _need(
            pins.get(str(route_release)) == ROUTE_RELEASE_SHA
            and pins.get(str(route_release.with_name("jev_decision3_completion_review_route.py")))
            == ROUTE_HELD_SHA,
            "explicit review route source pins required",
        )
        route = child.common._load_reviewed_release(
            str(route_release),
            "jev_decision3_completion_review_route.py",
            pins,
            "_candidate_completion_review_route",
        )
        import s1.jev_decision3_native as native

        primary = None
        try:
            _need(child._serve is serve, "child callback identity changed")
            child._serve = original
            with route.attach_to_child(child, native, enabled=True):
                return child._serve(
                    plan, plan_sha, invocation, backend, torch, modules, output, check
                )
        except BaseException as error:
            primary = error
            raise
        finally:
            problems = []
            if child._serve is original:
                child._serve = serve
            else:
                problems.append("child callback")
            if sys.modules.get("_candidate_completion_review_route") is route:
                del sys.modules["_candidate_completion_review_route"]
            else:
                problems.append("review route module")
            if problems:
                if primary is None:
                    raise RuntimeError("review launcher cleanup unconfirmed")
                primary.add_note("review launcher cleanup unconfirmed")

    child._serve = serve
    primary = None
    try:
        yield
    except BaseException as error:
        primary = error
        raise
    finally:
        if child._serve is serve:
            child._serve = original
        elif primary is None:
            raise RuntimeError("child service cleanup unconfirmed")
        else:
            primary.add_note("child service cleanup unconfirmed")


def main(argv=None):
    _need(SOURCE_ONLY is False, "held review child launcher requires independent review")
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--completion-review-opt-in", action="store_true", required=True)
    args = parser.parse_args(argv)
    scripts = Path(__file__).absolute().parent
    child_held = scripts / "jev_decision3_semantic_browser_child.py"
    child_release = scripts / "jev_decision3_semantic_pair_child_pending_review_release_20261027.py"
    held = _read(child_held, CHILD_HELD_SHA)
    released = _read(child_release, CHILD_RELEASE_SHA)
    _release(held, released)
    name = "_owned_completion_review_child"
    _need(name not in sys.modules, "fresh child module name required")
    spec = importlib.util.spec_from_file_location(name, child_release)
    _need(spec is not None and spec.loader is not None, "child source loader required")
    child = importlib.util.module_from_spec(spec)
    sys.modules[name] = child
    try:
        exec(compile(released, str(child_release), "exec", dont_inherit=True), child.__dict__)
        _need(child.SOURCE_ONLY is False, "released semantic child required")
        plan = child._load_plan(args.plan, args.plan_sha256)
        pins = plan["source_pins"]
        held_launcher = scripts / "jev_decision3_completion_review_child_launcher.py"
        release_launcher = Path(__file__).absolute()
        route_held = scripts / "jev_decision3_completion_review_route.py"
        route_release = scripts / ROUTE_RELEASE_NAME
        _need(
            all(
                str(path) in pins
                for path in (
                    held_launcher,
                    release_launcher,
                    child_held,
                    child_release,
                    route_held,
                    route_release,
                )
            )
            and pins[str(child_held)] == CHILD_HELD_SHA
            and pins[str(child_release)] == CHILD_RELEASE_SHA
            and pins[str(route_held)] == ROUTE_HELD_SHA
            and pins[str(route_release)] == ROUTE_RELEASE_SHA,
            "pinned launcher, route and child required",
        )
        child.common._verify_source_pins(pins)
        _release(held_launcher.read_bytes(), release_launcher.read_bytes())
        original_argv = sys.argv
        try:
            sys.argv = [str(child_release), "--plan", args.plan, "--plan-sha256", args.plan_sha256]
            with attach_owned_child(child, route_release):
                return child.main()
        finally:
            sys.argv = original_argv
    finally:
        if sys.modules.get(name) is child:
            del sys.modules[name]
        else:
            raise RuntimeError("review child module cleanup unconfirmed")


if __name__ == "__main__":
    raise SystemExit(main())
