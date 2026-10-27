"""Source-only same-process bootstrap for a pinned Jev browser runner.

An external owner must authenticate this source and independently supervise the
child, model and private sink. This module neither selects a task nor starts a
process. It executes the explicitly pinned runner bytes in a module namespace,
injects the explicitly pinned capture adapter, and calls the unchanged main().
"""

from __future__ import annotations

import ast
import hashlib
import importlib.abc
import importlib.util
import json
import os
import re
import stat
import sys
from pathlib import Path

SOURCE_ONLY = True
MAX_SOURCE_BYTES = 8 * 1024**2
FILES = frozenset(
    {
        "original_jev_browser_benchmark.py",
        "runner.py",
        "jev_public_web_cases.py",
        "jev_heldout_public_cases.py",
        "launcher.py",
        "runner-import.patch",
    }
)
ORIGINAL_IMPORT = b"        from jev_public_web_cases import (\n"
ISOLATED_IMPORT = b"        from jev_heldout_public_cases import (\n"
V7_PROFILE = "composed-visible-text-settled-v7"
V7_FILES = frozenset(
    {
        "original_jev_browser_benchmark.py",
        "runner.py",
        "jev_public_web_cases.py",
        "jev_observation_profile_v7.py",
        "jev_observation_profile_v3_parent.py",
        "jev_observation_events_v3_parent.py",
        "jev_observation_events_v7.py",
        "launcher.py",
        "runner-profile.patch",
    }
)
V7_OLD_IMPORT = b"from s1 import jev_observation_profile as profile"
V7_NEW_IMPORT = b"import jev_observation_profile_v7 as profile"
V7_OLD_CHOICE = b'PRUNED_OBSERVATION_PROFILE = "composed-visible-text-v3"'
V7_NEW_CHOICE = b'PRUNED_OBSERVATION_PROFILE = "composed-visible-text-settled-v7"'
V7_OLD_EVENTS_IMPORT = b"from s1.jev_observation_events import ObservationProfileTracker"
V7_NEW_EVENTS_IMPORT = b"from jev_observation_events_v7 import ObservationProfileTracker"
V7_OLD_EVENTS_CHOICE = b'PRUNED_PROFILE = "composed-visible-text-v3"'
V7_NEW_EVENTS_CHOICE = b'PRUNED_PROFILE = "composed-visible-text-settled-v7"'
V7_SIDECARS = (
    (
        b"                        snapshots = []\n",
        b"                        snapshots = []\n"
        b"                        _settling_primary = None\n"
        b"                        _settling_cursor = profile.settling_cursor(agent_module.Browser)\n",
    ),
    (
        b'                        except Exception as exc:\n                            error = f"{type(exc).__name__}: {exc}"\n',
        b"                        except Exception as exc:\n                            _settling_primary = exc\n"
        b'                            error = f"{type(exc).__name__}: {exc}"\n',
    ),
    (
        b'                            _write_json(attempt / "final_state.json", final_state, secrets)\n',
        b'                            _write_json(attempt / "final_state.json", final_state, secrets)\n'
        b"                            profile.write_settling_sidecar(agent_module.Browser, _settling_cursor,\n"
        b'                                attempt / "navigation-settling.json", _write_json, secrets,\n'
        b"                                _settling_primary if _settling_primary is not None else sys.exception(), _redact)\n",
    ),
)


def _need(value, message):
    if not value:
        raise ValueError(message)


def _gate():
    _need(SOURCE_ONLY is False, "held private bootstrap requires independent review")


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _read_pinned(path, expected, *, limit=MAX_SOURCE_BYTES):
    _need(_digest(expected), "explicit source SHA256 required")
    path = Path(path)
    _need(
        path.is_absolute() and ".." not in path.parts and path.resolve() == path,
        "absolute ordinary source path required",
    )
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        _need(
            stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= limit,
            "bounded ordinary source file required",
        )
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        after = os.fstat(fd)
        current = os.stat(path, follow_symlinks=False)
        fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        _need(
            tuple(getattr(before, key) for key in fields)
            == tuple(getattr(after, key) for key in fields)
            == tuple(getattr(current, key) for key in fields)
            and len(raw) == before.st_size
            and hashlib.sha256(raw).hexdigest() == expected,
            "source identity or SHA changed",
        )
        return raw
    finally:
        os.close(fd)


def _parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, "duplicate bundle manifest key")
            result[key] = value
        return result

    def invalid(_value):
        raise ValueError("nonfinite bundle manifest")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def _v7_sources(bundle, manifest, contents, runner_sha256, profile_helper_sha256):
    """Reconstruct the pinned V7 edits, then invoke its own bundle verifier."""
    _need(
        manifest.get("profile") == V7_PROFILE
        and manifest.get("case_module") in {"original", "heldout"}
        and _digest(profile_helper_sha256)
        and manifest.get("profile_helper_sha256") == profile_helper_sha256
        and hashlib.sha256(contents["jev_observation_profile_v7.py"]).hexdigest()
        == profile_helper_sha256,
        "exact independently pinned V7 profile and case selection required",
    )
    original = contents["original_jev_browser_benchmark.py"]
    parent_events = contents["jev_observation_events_v3_parent.py"]
    _need(
        original.count(V7_OLD_IMPORT) == 3
        and original.count(V7_OLD_CHOICE) == 1
        and parent_events.count(V7_OLD_EVENTS_CHOICE) == 1,
        "original V7 profile and tracker anchors required",
    )
    expected = original.replace(V7_OLD_IMPORT, V7_NEW_IMPORT).replace(V7_OLD_CHOICE, V7_NEW_CHOICE)
    _need(expected.count(V7_OLD_EVENTS_IMPORT) == 1, "original tracker import anchor required")
    expected = expected.replace(V7_OLD_EVENTS_IMPORT, V7_NEW_EVENTS_IMPORT)
    if manifest["case_module"] == "heldout":
        _need(expected.count(ORIGINAL_IMPORT) == 1, "original case import anchor required")
        expected = expected.replace(ORIGINAL_IMPORT, ISOLATED_IMPORT, 1)
    for before, after in V7_SIDECARS:
        _need(expected.count(before) == 1, "original V7 sidecar anchor required")
        expected = expected.replace(before, after, 1)
    _need(
        expected == contents["runner.py"]
        and hashlib.sha256(expected).hexdigest() == runner_sha256
        and contents["jev_observation_events_v7.py"]
        == parent_events.replace(V7_OLD_EVENTS_CHOICE, V7_NEW_EVENTS_CHOICE),
        "V7 derivative or tracker differs from explicit source transformations",
    )
    inverse = expected
    for before, after in reversed(V7_SIDECARS):
        inverse = inverse.replace(after, before, 1)
    inverse = inverse.replace(V7_NEW_IMPORT, V7_OLD_IMPORT).replace(V7_NEW_CHOICE, V7_OLD_CHOICE)
    inverse = inverse.replace(V7_NEW_EVENTS_IMPORT, V7_OLD_EVENTS_IMPORT)
    if manifest["case_module"] == "heldout":
        inverse = inverse.replace(ISOLATED_IMPORT, ORIGINAL_IMPORT, 1)
    _need(
        inverse == original
        and ast.dump(ast.parse(inverse), include_attributes=False)
        == ast.dump(ast.parse(original), include_attributes=False),
        "V7 inverse source and AST equality required",
    )
    _need(
        manifest.get("source_delta")
        == {
            "profile_import_substitutions": 3,
            "diagnostic_only_sidecar_substitutions": 3,
            "sidecar_max_bytes": 8 * 1024 * 1024,
            "profile_choice_literal_substitutions": 1,
            "tracker_import_substitutions": 1,
            "tracker_profile_literal_substitutions": 1,
            "tracker_inverse_bytes_equal": True,
            "heldout_case_import_substitutions": int(manifest["case_module"] == "heldout"),
            "inverse_bytes_equal": True,
            "inverse_AST_equal": True,
            "original_sha256": hashlib.sha256(original).hexdigest(),
            "derivative_sha256": runner_sha256,
        }
        and manifest.get("case005_interface_exposed") is True
        and manifest.get("model_executed") is False
        and manifest.get("runtime_qualification_attested") is False,
        "exact original V7 source-delta and preparation scope required",
    )
    launcher_sha = manifest["files"]["launcher.py"]["sha256"]
    name = "_jev_verified_v7_launcher_" + launcher_sha[:16]
    launcher = _module(bundle / "launcher.py", contents["launcher.py"], name)
    try:
        _need(
            launcher.PROFILE == V7_PROFILE
            and callable(launcher.verify_bundle)
            and hashlib.sha256(original).hexdigest() == launcher.BENCHMARK_SHA256
            and hashlib.sha256(contents["jev_public_web_cases.py"]).hexdigest()
            == launcher.HELPERS_SHA256
            and hashlib.sha256(contents["jev_observation_profile_v3_parent.py"]).hexdigest()
            == launcher.PARENT_SHA256
            and hashlib.sha256(parent_events).hexdigest() == launcher.EVENTS_SHA256
            and (
                manifest["case_module"] != "heldout"
                or hashlib.sha256(contents["jev_heldout_public_cases.py"]).hexdigest()
                == launcher.HELDOUT_SHA256
            ),
            "V7 launcher source identities required",
        )
        launcher.verify_bundle(bundle, manifest)
    finally:
        _need(sys.modules.get(name) is launcher, "V7 verifier module identity changed")
        del sys.modules[name]


def _bundle(bundle_dir, manifest_sha256, runner_sha256, profile_helper_sha256=None):
    bundle = Path(bundle_dir)
    _need(
        bundle.is_absolute() and bundle.is_dir() and bundle.resolve() == bundle,
        "ordinary absolute verified bundle required",
    )
    raw = _read_pinned(bundle / "bundle-manifest.json", manifest_sha256, limit=1024**2)
    manifest = _parse(raw)
    _need(
        type(manifest) is dict and type(manifest.get("files")) is dict, "bundle inventory required"
    )
    schema = manifest.get("schema_id")
    if schema == "jev-heldout-evaluator-bundle-v1":
        allowed_files = FILES
    elif schema == "jev-v7-isolated-runner-bundle-v1":
        allowed_files = V7_FILES | (
            {"jev_heldout_public_cases.py"} if manifest.get("case_module") == "heldout" else set()
        )
    else:
        raise ValueError("unsupported pinned bundle schema")
    _need(set(manifest["files"]) == allowed_files, "complete pinned bundle inventory required")
    _need(
        set(os.listdir(bundle)) <= allowed_files | {"bundle-manifest.json", "launch-receipt.json"},
        "unexpected bundle entry refused",
    )
    contents = {}
    for name, record in manifest["files"].items():
        _need(
            type(record) is dict
            and set(record) == {"sha256", "size_bytes"}
            and _digest(record["sha256"])
            and type(record["size_bytes"]) is int
            and record["size_bytes"] >= 0,
            "exact source file digest and size required",
        )
        contents[name] = _read_pinned(bundle / name, record["sha256"])
        _need(len(contents[name]) == record["size_bytes"], "bundle source size differs")
    if schema == "jev-v7-isolated-runner-bundle-v1":
        _v7_sources(bundle, manifest, contents, runner_sha256, profile_helper_sha256)
        return bundle, contents
    change = manifest.get("source_change")
    _need(
        type(change) is dict
        and change.get("kind") == "one_explicit_public_case_import_substitution"
        and change.get("derivative_sha256") == runner_sha256
        and manifest["files"]["runner.py"]["sha256"] == runner_sha256
        and change.get("original_sha256")
        == manifest["files"]["original_jev_browser_benchmark.py"]["sha256"],
        "externally pinned runner and original source identity required",
    )
    original = contents["original_jev_browser_benchmark.py"]
    _need(
        original.count(ORIGINAL_IMPORT) == 1
        and ISOLATED_IMPORT not in original
        and contents["runner.py"] == original.replace(ORIGINAL_IMPORT, ISOLATED_IMPORT, 1),
        "runner must preserve the single approved import substitution",
    )
    return bundle, contents


def _capture_source(held_path, held_sha256, released_path, released_sha256):
    held = _read_pinned(held_path, held_sha256)
    released = _read_pinned(released_path, released_sha256)
    marker = b"SOURCE_ONLY = True"
    _need(
        held.count(marker) == 1 and released == held.replace(marker, b"SOURCE_ONLY = False", 1),
        "capture release must differ only by the reviewed source gate",
    )
    return released


def _module(path, raw, name):
    _need(name not in sys.modules, "fresh isolated module name required")
    spec = importlib.util.spec_from_file_location(name, path)
    _need(spec is not None and spec.loader is not None, "ordinary source module spec required")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(raw, str(path), "exec"), module.__dict__)
    except BaseException:
        if sys.modules.get(name) is module:
            del sys.modules[name]
        raise
    return module


class _PinnedFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Load bundle sidecars from verified in-memory bytes, never preloaded names."""

    def __init__(self, bundle, contents):
        selected = {
            name.removesuffix(".py"): (bundle / name, raw)
            for name, raw in contents.items()
            if name.endswith(".py")
            and name not in {"runner.py", "launcher.py", "original_jev_browser_benchmark.py"}
        }
        self.selected, self.loaded = selected, {}

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.selected:
            return None
        source, _raw = self.selected[fullname]
        return importlib.util.spec_from_file_location(fullname, source, loader=self)

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        path, raw = self.selected[module.__name__]
        _need(
            _read_pinned(path, hashlib.sha256(raw).hexdigest()) == raw,
            "pinned bundle module changed before import",
        )
        self.loaded[module.__name__] = module
        exec(compile(raw, str(path), "exec"), module.__dict__)


def _sink_proxy(sink, *, destination_attribution=False):
    _need(type(destination_attribution) is bool, "explicit destination attribution required")
    sequence = 0

    def private_event(event):
        nonlocal sequence
        _need(
            type(event) is dict
            and set(event)
            == {
                "sequence",
                "status",
                "request_source",
                "request_bytes",
                "response_bytes",
                "http_status",
                "exception_type",
                "scope",
            }
            | ({"destination_match"} if destination_attribution else set())
            and type(event["sequence"]) is int
            and event["sequence"] == sequence + 1
            and event["status"] in {"response", "transport_exception"}
            and event["request_source"]
            in {
                "response-attached-request",
                "exception-attached-request-not-delivery-proof",
                "unavailable",
            }
            and (event["request_bytes"] is None or type(event["request_bytes"]) is bytes)
            and (event["response_bytes"] is None or type(event["response_bytes"]) is bytes)
            and (event["http_status"] is None or type(event["http_status"]) is int)
            and (event["exception_type"] is None or type(event["exception_type"]) is str)
            and type(event["scope"]) is str,
            "bounded private body event without headers or URL required",
        )
        if destination_attribution:
            _need(
                type(event["destination_match"]) is str
                and event["destination_match"]
                in {"policy_endpoint", "text_endpoint", "shared_endpoint", "unknown"},
                "fixed destination classification required",
            )
        _need(
            (event["request_bytes"] is None or len(event["request_bytes"]) <= 1_000_000)
            and (event["response_bytes"] is None or len(event["response_bytes"]) <= 16 * 1024**2),
            "bounded private HTTP body bytes required",
        )
        result = sink(event)
        sequence += 1
        return result

    return private_event


def _require_v7_arguments(arguments, helper_sha256):
    """Require the original V7 launcher defaults already forwarded verbatim."""
    for flag, expected in (
        ("--observation-profile", V7_PROFILE),
        ("--observation-profile-helper-sha256", helper_sha256),
    ):
        found = []
        for index, value in enumerate(arguments):
            if value == flag:
                _need(index + 1 < len(arguments), "complete explicit V7 runner arguments required")
                found.append(arguments[index + 1])
            elif value.startswith(flag + "="):
                found.append(value[len(flag) + 1 :])
        _need(found == [expected], "exact explicit V7 runner profile and helper arguments required")


def run(
    *,
    bundle_dir,
    manifest_sha256,
    runner_sha256,
    held_capture_path,
    held_capture_sha256,
    released_capture_path,
    released_capture_sha256,
    profile_helper_sha256=None,
    destination_attribution=False,
    sink,
    arguments,
):
    """Call the unchanged runner.main() under opt-in same-process capture.

    ``arguments`` are the original forwarded CLI arguments, without argv[0].
    The private sink receives sensitive body bytes only, never HTTP headers or
    URLs. Actual model/browser execution, source ownership and numerical audit
    remain the responsibility of the supervising process.
    """
    _gate()
    _need(
        callable(sink)
        and sys.dont_write_bytecode is True
        and type(destination_attribution) is bool
        and type(arguments) in (list, tuple)
        and all(type(item) is str for item in arguments),
        "no-bytecode child, explicit private sink and original string arguments required",
    )
    bundle, contents = _bundle(bundle_dir, manifest_sha256, runner_sha256, profile_helper_sha256)
    if "jev_observation_profile_v7.py" in contents:
        _require_v7_arguments(arguments, profile_helper_sha256)
    runner_bytes = contents["runner.py"]
    finder = _PinnedFinder(bundle, contents)
    parent_alias = "_jev_observation_profile_v7_parent_8d199ac1564a"
    protected = set(finder.selected)
    if "jev_observation_profile_v7.py" in contents:
        protected.add(parent_alias)
    _need(not protected.intersection(sys.modules), "preloaded bundle or V7 parent module refused")
    capture_bytes = _capture_source(
        held_capture_path, held_capture_sha256, released_capture_path, released_capture_sha256
    )
    runner_path = bundle / "runner.py"
    release_path = Path(released_capture_path)
    runner_name = "_jev_private_runner_" + runner_sha256[:16]
    capture_name = "_jev_private_capture_" + released_capture_sha256[:16]
    old_argv, old_path = sys.argv, list(sys.path)
    runner = capture = None
    primary = None
    try:
        sys.argv = [str(runner_path), *arguments]
        sys.path[:] = [str(bundle), *old_path[1:]]
        sys.meta_path.insert(0, finder)
        capture = _module(release_path, capture_bytes, capture_name)
        _need(
            capture.SOURCE_ONLY is False and callable(capture.attach_to_runner),
            "released capture adapter required",
        )
        runner = _module(runner_path, runner_bytes, runner_name)
        _need(
            callable(getattr(runner, "main", None))
            and callable(getattr(runner, "model_adapter", None)),
            "pinned runner entry points required",
        )
        options = {"sink": _sink_proxy(sink, destination_attribution=destination_attribution)}
        if destination_attribution:
            options["destination_attribution"] = True
        with capture.attach_to_runner(runner, **options):
            return runner.main()
    except BaseException as error:
        primary = error
        raise
    finally:
        problems = []
        try:
            try:
                _bundle(bundle_dir, manifest_sha256, runner_sha256, profile_helper_sha256)
                _capture_source(
                    held_capture_path,
                    held_capture_sha256,
                    released_capture_path,
                    released_capture_sha256,
                )
            except BaseException as error:
                problems.append(error)
            for name, module in ((runner_name, runner), (capture_name, capture)):
                if module is not None:
                    try:
                        _need(sys.modules.get(name) is module, "executed module identity changed")
                        del sys.modules[name]
                    except BaseException as error:
                        problems.append(error)
            for name in protected:
                module = sys.modules.get(name)
                if module is None:
                    continue
                try:
                    if name == parent_alias:
                        path = bundle / "jev_observation_profile_v3_parent.py"
                        raw = contents[path.name]
                    else:
                        path, raw = finder.selected[name]
                        _need(
                            finder.loaded.get(name) is module,
                            "bundle module bypassed pinned loader",
                        )
                    _need(
                        Path(module.__file__).absolute() == path
                        and _read_pinned(path, hashlib.sha256(raw).hexdigest()) == raw,
                        "bundle module origin or source changed",
                    )
                    del sys.modules[name]
                except BaseException as error:
                    problems.append(error)
        finally:
            if finder in sys.meta_path:
                sys.meta_path.remove(finder)
            else:
                problems.append(RuntimeError("pinned bundle import finder disappeared"))
            sys.argv = old_argv
            sys.path[:] = old_path
        if problems:
            if primary is None:
                raise problems[0]
            for error in problems:
                try:
                    primary.add_note(
                        "Private bootstrap integrity or cleanup failed: " + type(error).__name__
                    )
                except BaseException:
                    pass
