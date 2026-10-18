"""Pure v7 source contracts and native Browser/Agent clock/CDP fakes.

No Chromium, sockets, policy/helper, model, GPU or unit is started. Fake CDP
exercises the complete derived native Browser and unchanged Agent source; these
checks do not substitute for the separately gated real streaming fixtures.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import sys
import types
from collections import deque
from pathlib import Path

import jev_browser_profile_v7_benchmark as launcher
import jev_observation_profile_v7 as profile
import pytest

from s1 import jev_observation_profile as parent

ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA = "8d199ac1564af05892e9b21e225d1fdf500f1953840855c902160ed8d56df89d"
SOURCE_SHA = hashlib.sha256(Path(profile.__file__).read_bytes()).hexdigest()
UPSTREAM = ROOT / "artifacts/jev-runtime/upstream"
BROWSER, SNAPSHOT = profile.BROWSER, profile.SNAPSHOT


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


@pytest.fixture(scope="module")
def canonical():
    assert UPSTREAM.is_dir(), "Installed canonical checkout is required; never fetches"
    return profile.read_canonical_sources(UPSTREAM)


@pytest.fixture(scope="module")
def bundle(canonical):
    return profile.build_profile(canonical, expected_source_sha256=SOURCE_SHA)


def method_span(blob, name):
    tree = ast.parse(blob)
    browser = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Browser")
    method = next(n for n in browser.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return b"".join(blob.splitlines(keepends=True)[method.lineno - 1 : method.end_lineno])


def test_exact_v3_parent_and_only_observe_identity_delta(canonical, bundle):
    assert sha(Path(parent.__file__).read_bytes()) == PARENT_SHA
    old = parent.build_profile(
        canonical, expected_source_sha256=PARENT_SHA, profile=parent.PRUNED_PROFILE
    )
    assert dict(bundle.canonical_files) == canonical
    assert set(bundle.runtime_files) == set(canonical)
    new_browser, old_browser = bundle.runtime_files[BROWSER], old.runtime_files[BROWSER]
    before, after = method_span(old_browser, "observe"), method_span(new_browser, "observe")
    inverse = new_browser.replace(after, before, 1)
    assert inverse == old_browser == canonical[BROWSER]
    assert ast.dump(ast.parse(inverse)) == ast.dump(ast.parse(old_browser))
    for name in ("fresh", "act", "call", "evaluate", "__init__", "close"):
        assert method_span(new_browser, name) == method_span(old_browser, name)
    before_id = ("id:'" + parent.PRUNED_PROFILE + "'").encode()
    after_id = ("id:'" + profile.PROFILE + "'").encode()
    assert bundle.runtime_files[SNAPSHOT].count(after_id) == 1
    assert (
        bundle.runtime_files[SNAPSHOT].replace(after_id, before_id, 1)
        == old.runtime_files[SNAPSHOT]
    )
    assert all(
        bundle.runtime_files[k] == old.runtime_files[k] == canonical[k]
        for k in canonical
        if k not in {BROWSER, SNAPSHOT}
    )
    # The native operation function, including all act and snapshot/screenshot logic,
    # has not been inlined, wrapped, replaced or given a new callback parameter.
    old_tree, new_tree = ast.parse(old_browser), ast.parse(new_browser)
    for name in ("browser_operation", "fingerprint"):

        def select(tree):
            return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)

        assert ast.dump(select(old_tree)) == ast.dump(select(new_tree))
    assert bundle.manifest()["navigation_settling"]["target_seconds"] == 5.0


@pytest.mark.parametrize("name", tuple(parent.CANONICAL_SHA256))
def test_canonical_drift_refused(canonical, name):
    changed = dict(canonical)
    changed[name] += b"\n"
    with pytest.raises(ValueError):
        profile.build_profile(changed, expected_source_sha256=SOURCE_SHA)


@pytest.mark.parametrize(
    "choice", ["composed-visible-text-v3", "composed-visible-controls-settled-v6", "v7"]
)
def test_explicit_profile_only(canonical, choice):
    with pytest.raises(ValueError):
        profile.build_profile(canonical, expected_source_sha256=SOURCE_SHA, profile=choice)


def test_complete_materialized_two_helper_closure(tmp_path):
    result = profile.materialize_profile(
        UPSTREAM, tmp_path / "profile", expected_source_sha256=SOURCE_SHA
    )
    handle = profile.verify_materialized_profile(
        tmp_path / "profile",
        expected_manifest_sha256=result["manifest_sha256"],
        expected_source_sha256=SOURCE_SHA,
    )
    manifest = handle.manifest()
    assert manifest["parent_helper_source_sha256"] == PARENT_SHA
    assert set(manifest["artifact_files"]) == {
        *("canonical/" + k for k in parent.CANONICAL_SHA256),
        *("runtime/" + k for k in parent.CANONICAL_SHA256),
        "sources/jev_observation_profile_v7.py",
        "sources/jev_observation_profile_v3_parent.py",
    }
    assert handle.read_state == (handle.runtime_root / SNAPSHOT).read_text()
    for label, pin in manifest["artifact_files"].items():
        blob = (handle.path / label).read_bytes()
        assert sha(blob) == pin["sha256"] and len(blob) == pin["size_bytes"]
    # Rehashing a modified derived browser does not authorize new bytes.
    browser = handle.runtime_root / BROWSER
    browser.write_bytes(browser.read_bytes() + b"\n# drift\n")
    with pytest.raises(ValueError):
        profile.verify_materialized_profile(
            handle.path,
            expected_manifest_sha256=result["manifest_sha256"],
            expected_source_sha256=SOURCE_SHA,
        )


@pytest.mark.parametrize("kind", ["extra", "directory", "symlink"])
def test_closed_population_refuses_unknown_or_alias(tmp_path, kind):
    result = profile.materialize_profile(
        UPSTREAM, tmp_path / "profile", expected_source_sha256=SOURCE_SHA
    )
    other = tmp_path / "profile" / "extra"
    if kind == "extra":
        other.write_bytes(b"unknown")
    elif kind == "directory":
        other.mkdir()
    else:
        other.symlink_to(UPSTREAM)
    with pytest.raises(ValueError):
        profile.verify_materialized_profile(
            tmp_path / "profile",
            expected_manifest_sha256=result["manifest_sha256"],
            expected_source_sha256=SOURCE_SHA,
        )


@pytest.mark.parametrize("cases", ["original", "heldout"])
def test_runner_inverse_all_three_export_anchors_and_tracker(tmp_path, cases):
    path, manifest = launcher.prepare_bundle(tmp_path / "runner", case_module=cases)
    launcher.verify_bundle(path, manifest)
    original, derived = (
        (path / "original_jev_browser_benchmark.py").read_bytes(),
        (path / "runner.py").read_bytes(),
    )
    # Independently remove only the diagnostic additions, then the declared import
    # and identity substitutions. Everything else must be the actual old benchmark.
    inverse = derived.replace(b"                        _settling_primary = None\n", b"")
    inverse = inverse.replace(
        b"                        _settling_cursor = profile.settling_cursor(agent_module.Browser)\n",
        b"",
    )
    inverse = inverse.replace(b"                            _settling_primary = exc\n", b"")
    export = (
        b"                            profile.write_settling_sidecar(agent_module.Browser, _settling_cursor,\n"
        b'                                attempt / "navigation-settling.json", _write_json, secrets,\n'
        b"                                _settling_primary if _settling_primary is not None else sys.exception(), _redact)\n"
    )
    assert inverse.count(export) == 1
    inverse = inverse.replace(export, b"")
    for before, after in (
        (launcher.NEW_IMPORT, launcher.OLD_IMPORT),
        (launcher.NEW_CHOICE, launcher.OLD_CHOICE),
        (launcher.NEW_EVENTS_IMPORT, launcher.OLD_EVENTS_IMPORT),
    ):
        inverse = inverse.replace(before, after)
    if cases == "heldout":
        inverse = inverse.replace(launcher.NEW_CASE, launcher.OLD_CASE, 1)
    assert inverse == original
    assert ast.dump(ast.parse(inverse)) == ast.dump(ast.parse(original))
    events = (path / "jev_observation_events_v7.py").read_bytes()
    assert (
        events.replace(launcher.NEW_EVENTS_CHOICE, launcher.OLD_EVENTS_CHOICE)
        == (ROOT / "src/s1/jev_observation_events.py").read_bytes()
    )
    spec = importlib.util.spec_from_file_location(
        "_test_v7_events", path / "jev_observation_events_v7.py"
    )
    tracker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tracker)
    tracker.ObservationProfileTracker(
        read_state="read",
        marker_expression="marker",
        manifest_sha256="a" * 64,
        profile=profile.PROFILE,
        sink=lambda row: None,
    )
    assert manifest["source_delta"]["diagnostic_only_sidecar_substitutions"] == 3
    assert not manifest["runtime_qualification_attested"]


class Clock:
    def __init__(self):
        self.now, self.sleeps = 0.0, []

    def monotonic(self):
        return self.now

    def perf_counter(self):
        return self.now

    def sleep(self, seconds):
        assert 0 <= seconds <= 0.1
        self.sleeps.append(seconds)
        self.now += seconds


def state(text="page"):
    return {
        "url": "https://owned.invalid/fixture",
        "text": text,
        "actions": [
            {
                "id": "e1",
                "node": 1,
                "kind": "click",
                "label": "Open",
                "rect": {"x": 1, "y": 1, "w": 10, "h": 10},
            }
        ],
        "scroll": {"x": 0, "y": 0},
        "marker": "m",
        "page_key": "p",
        "guards": {"1": ["guard"]},
    }


class CDP:
    def __init__(self, clock):
        self.clock, self.script, self.calls = clock, deque(), []
        self.default = (0.0, None)
        self.ready = (0.0, None)
        self.shot_delay = 0.0
        self.fresh = True

    def __call__(self, method, **params):
        self.calls.append((method, copy.deepcopy(params)))
        if method == "Target.createTarget":
            return {"targetId": "t"}
        if method == "Target.attachToTarget":
            return {"sessionId": "s"}
        if method == "Page.captureScreenshot":
            self.clock.now += self.shot_delay
            return {"data": "unchanged-image-bytes"}
        if method != "Runtime.evaluate":
            return {}
        expression = params["expression"]
        if expression == "document.readyState":
            value = "complete"
        elif params.get("awaitPromise"):
            delay, value = self.ready
            self.clock.now += delay
        elif expression == self.module.READ_STATE:
            delay, value = self.script.popleft() if self.script else self.default
            self.clock.now += delay
        elif expression == self.module.MARKER:
            value = "m" if self.fresh else "changed"
        elif "c.guard(c.nodes.get(" in expression:
            value = ["p", ["guard"]] if self.fresh else None
        else:
            value = {"x": 6, "y": 6}
        if isinstance(value, BaseException):
            raise value
        return {"result": {"value": copy.deepcopy(value)}}

    @property
    def inputs(self):
        return [row for row in self.calls if row[0].startswith("Input.")]


@pytest.fixture
def native(monkeypatch, tmp_path, bundle):
    clock, cdp = Clock(), None
    cdp = CDP(clock)
    package = types.ModuleType("browser_harness")
    package.__path__ = []
    admin, helpers = (
        types.ModuleType("browser_harness.admin"),
        types.ModuleType("browser_harness.helpers"),
    )
    admin.ensure_daemon = lambda: None
    helpers.cdp = cdp
    for module in (package, admin, helpers):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    path = tmp_path / "browser.py"
    path.write_bytes(bundle.runtime_files[BROWSER])
    path.with_name("snapshot.js").write_bytes(bundle.runtime_files[SNAPSHOT])
    module = types.ModuleType("_v7_native_browser")
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    module.time = clock
    cdp.module = module
    browser = module.Browser.__new__(module.Browser)
    browser.session, browser.target = "s", "t"
    return module, browser, cdp, clock


def test_complete_fast_snapshot_no_extra_model_fields_or_wait(native):
    module, browser, cdp, clock = native
    cdp.script.append((0.001, state()))
    page = browser.observe(screenshot=False)
    expected = state()
    expected["fingerprint"] = module.fingerprint(expected)
    assert page == expected and not clock.sleeps and not cdp.inputs
    row = browser.observation_settling
    assert row["status"] == "success" and row["attempts"] == 1
    assert (
        row["elapsed_seconds"] == 0.001
        and row["operations"][0]["native_protocol_call_count"] is None
    )
    assert row["operation_timing_scope"] == "native_observe_combined_not_individual_CDP"


def test_native_combined_snapshot_screenshot_timing_not_individual(native):
    _, browser, cdp, _ = native
    cdp.script.append((0.2, state()))
    cdp.shot_delay = 0.4
    page = browser.observe()
    row = browser.observation_settling
    assert page["screenshot"] == "unchanged-image-bytes"
    assert row["operations"][0]["elapsed_seconds"] == pytest.approx(0.6)
    assert row["operations"][0]["returned_snapshot_sha256"] == sha(
        json.dumps(page, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    )


def test_transient_navigation_recovers_without_native_input(native):
    _, browser, cdp, clock = native
    cdp.script.extend(((2.4, None), (0.1, state("arrived"))))
    page = browser.observe(False)
    assert page["text"] == "arrived" and not cdp.inputs
    assert clock.sleeps == [0.02]
    row = browser.observation_settling
    assert row["attempts"] == 2 and row["navigation_retries"] == 1
    assert row["operations"][0]["null_navigation"] is True


@pytest.mark.parametrize("delay", [5.0, 5.4])
def test_late_complete_snapshot_is_refused_and_true_overshoot_recorded(native, delay):
    module, browser, cdp, _ = native
    cdp.script.append((delay, state()))
    with pytest.raises(module.StalePage, match="Navigation settling target exceeded") as caught:
        browser.observe(False)
    row = browser.observation_settling
    assert row["late_snapshot"] and row["terminal_unknown"] and row["status"] == "timeout"
    assert row["elapsed_seconds"] == delay and row["overshoot_seconds"] == pytest.approx(delay - 5)
    assert caught.value.observation_settling == row and not cdp.inputs


def test_diagnostic_serialization_cost_is_in_target_and_late_gate(native):
    module, browser, cdp, clock = native
    cdp.script.append((0.1, state()))
    real_dumps = json.dumps

    def slow_dumps(value, *args, **kwargs):
        result = real_dumps(value, *args, **kwargs)
        if isinstance(value, dict) and value.get("status") == "success" and "operations" in value:
            clock.now += 5.1
        return result

    module.json = types.SimpleNamespace(dumps=slow_dumps, loads=json.loads)
    with pytest.raises(module.StalePage) as caught:
        browser.observe(False)
    row = browser.observation_settling
    assert row["status"] == "timeout" and row["late_snapshot"]
    assert row["elapsed_seconds"] == pytest.approx(5.2)
    assert len(type(browser).observation_settling_records) == 1
    assert caught.value.observation_settling == row


@pytest.mark.parametrize("initial_delay", [0.1, 4.8])
def test_final_diagnostic_commit_cost_in_reported_elapsed_and_gate(native, initial_delay):
    module, browser, cdp, clock = native

    class SlowCommit(type):
        def __setattr__(cls, name, value):
            super().__setattr__(name, value)
            if name == "observation_settling_records":
                clock.now += 0.4

    class SlowBrowser(module.Browser, metaclass=SlowCommit):
        pass

    browser.__class__ = SlowBrowser
    cdp.script.append((initial_delay, state()))
    if initial_delay > 4:
        with pytest.raises(module.StalePage):
            browser.observe(False)
        assert browser.observation_settling["status"] == "timeout"
    else:
        browser.observe(False)
        assert browser.observation_settling["status"] == "success"
    assert browser.observation_settling["elapsed_seconds"] == pytest.approx(initial_delay + 0.4)
    assert not cdp.inputs


def test_rehashed_modified_runtime_is_not_authorized_by_new_manifest(tmp_path):
    result = profile.materialize_profile(
        UPSTREAM, tmp_path / "profile", expected_source_sha256=SOURCE_SHA
    )
    root = Path(result["output"])
    target = root / "runtime" / BROWSER
    target.write_bytes(target.read_bytes() + b"\n# unauthorized\n")
    path = root / "profile-manifest.json"
    manifest = json.loads(path.read_bytes())
    digest = sha(target.read_bytes())
    manifest["artifact_files"]["runtime/" + BROWSER] = {
        "sha256": digest,
        "size_bytes": target.stat().st_size,
    }
    manifest["runtime_source_sha256"][BROWSER] = digest
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    )
    with pytest.raises(ValueError, match="metadata differs"):
        profile.verify_materialized_profile(
            root, expected_manifest_sha256=sha(path.read_bytes()), expected_source_sha256=SOURCE_SHA
        )


@pytest.mark.parametrize(
    "kind", ["other_stale", "subclass_same_message", "runtime", "keyboard", "exception_details"]
)
def test_only_exact_native_null_navigation_retries(native, kind):
    module, browser, cdp, _ = native

    class Child(module.StalePage):
        pass

    errors = {
        "other_stale": module.StalePage("Target changed"),
        "subclass_same_message": Child("Document is navigating"),
        "runtime": RuntimeError("transport failure"),
        "keyboard": KeyboardInterrupt(),
    }
    if kind == "exception_details":
        original = cdp.__class__.__call__

        def changed(self, method, **params):
            if method == "Runtime.evaluate" and params["expression"] == module.READ_STATE:
                return {"exceptionDetails": {"text": "script failure"}}
            return original(self, method, **params)

        cdp.__class__.__call__ = changed
        try:
            with pytest.raises(module.StalePage, match="Document changed during evaluation"):
                browser.observe(False)
        finally:
            cdp.__class__.__call__ = original
    else:
        exc = errors[kind]
        cdp.script.append((0.1, exc))
        with pytest.raises(type(exc)) as caught:
            browser.observe(False)
        assert caught.value is exc
    assert browser.observation_settling["attempts"] == 1
    assert browser.observation_settling["navigation_retries"] == 0
    assert browser.observation_settling["status"] == "unexpected_error"
    assert not cdp.inputs


def test_after_input_baseline_runtime_swallow_once_includes_wait(native):
    _, browser, cdp, clock = native
    browser.after_input = state()["actions"][0]
    cdp.ready = (0.7, RuntimeError("baseline interrupted readiness"))
    cdp.script.append((0.2, state()))
    browser.observe(False)
    assert browser.after_input is None and clock.now == pytest.approx(0.9)
    row = browser.observation_settling
    assert row["baseline_after_input_runtime_error_swallowed"] and row["after_input_consumed"]
    cdp.script.append((0, state()))
    browser.observe(False)
    assert not browser.observation_settling["after_input_consumed"]
    assert len(browser.observation_settling_history) == 2
    assert len(type(browser).observation_settling_records) == 2


def test_after_input_alone_can_exceed_target_without_snapshot_or_replay(native):
    module, browser, cdp, _ = native
    browser.after_input = state()["actions"][0]
    cdp.ready = (5.2, None)
    with pytest.raises(module.StalePage):
        browser.observe(False)
    assert browser.observation_settling["attempts"] == 0
    assert browser.observation_settling["overshoot_seconds"] == pytest.approx(0.2)
    assert browser.after_input is None and not cdp.inputs


def test_old_stale_action_guard_unchanged_no_input(native):
    module, browser, cdp, _ = native
    cdp.fresh = False
    with pytest.raises(module.StalePage, match="Page changed since this decision"):
        browser.act(state()["actions"][0], state())
    assert not cdp.inputs and not hasattr(browser, "observation_settling")


def load_agent(native, canonical, monkeypatch):
    browser_module, _, cdp, clock = native
    package = types.ModuleType("_v7_fake_package")
    package.__path__ = []
    model, questions = (
        types.ModuleType("_v7_fake_package.model"),
        types.ModuleType("_v7_fake_package.questions"),
    )
    choices = []

    def choose(page, goal, history):
        choices.append((copy.deepcopy(page), goal, copy.deepcopy(history)))
        return dict(
            choice="e1",
            probabilities={"e1": 1.0},
            confidence=1.0,
            latency_ms=0,
            operation="CLICK",
            target="e1",
            usage={},
        )

    model.action_space = lambda actions: ([], {})
    model.choose = choose
    model.field_context = model.field_text = lambda *args: pytest.fail("No field helper is allowed")
    questions.MAX_STEPS = 16
    for key, value in (
        (package.__name__, package),
        ("_v7_fake_package.browser", browser_module),
        (model.__name__, model),
        (questions.__name__, questions),
    ):
        monkeypatch.setitem(sys.modules, key, value)
    agent_module = types.ModuleType("_v7_fake_package.agent")
    agent_module.__package__ = package.__name__
    exec(
        compile(canonical["jev_ultrafast/agent.py"], "<unchanged-native-agent>", "exec"),
        agent_module.__dict__,
    )
    agent_module.time = clock
    return agent_module, cdp, choices


@pytest.mark.parametrize("second_timeout", [False, True])
def test_unchanged_tick_two_observe_passes_one_choice_one_native_click(
    native, canonical, monkeypatch, second_timeout
):
    agent_module, cdp, choices = load_agent(native, canonical, monkeypatch)
    cdp.script.append((0, state("initial")))
    agent = agent_module.Agent("https://owned.invalid", "Open the offered item", screenshots=False)
    cursor = profile.settling_cursor(agent_module.Browser)
    cdp.script.extend(
        (
            (5.2, None),
            (5.3 if second_timeout else 0.3, None if second_timeout else state("arrived")),
        )
    )
    if second_timeout:
        with pytest.raises(native[0].StalePage):
            agent.command("tick")
    else:
        assert agent.command("tick")["page"]["text"] == "arrived"
    records = agent_module.Browser.observation_settling_records[cursor:]
    assert len(records) == 2 and records[0]["status"] == "timeout"
    assert records[1]["status"] == ("timeout" if second_timeout else "success")
    assert records[0]["after_input_consumed"] and not records[1]["after_input_consumed"]
    assert len(choices) == 1 and len(cdp.inputs) == 2 and len(agent.state["history"]) == 1
    assert agent.state["history"][0]["page_changed"] is None
    assert [r["elapsed_seconds"] for r in records] == pytest.approx(
        [5.2, 5.3 if second_timeout else 0.3]
    )


def test_constructor_failure_agent_none_still_exports_trace(native, canonical, monkeypatch):
    agent_module, cdp, _ = load_agent(native, canonical, monkeypatch)
    cdp.script.append((5.1, None))
    agent, captured = None, []
    cursor = profile.settling_cursor(agent_module.Browser)
    with pytest.raises(native[0].StalePage):
        agent = agent_module.Agent("https://owned.invalid", "Read", screenshots=False)
    assert agent is None
    profile.write_settling_sidecar(
        agent_module.Browser,
        cursor,
        Path("unused"),
        lambda p, value, secrets: captured.append(value),
        (),
    )
    assert len(captured[0]["records"]) == 1 and captured[0]["terminal_unknown"]
    assert not captured[0]["task_success_attested"] and not cdp.inputs


def test_two_attempt_cursors_isolate_rows_and_success_writer_preserves_caught_flow(native):
    _, browser, cdp, _ = native
    captured = []
    for text in ("attempt one", "attempt two"):
        cursor = profile.settling_cursor(type(browser))
        cdp.script.append((0, state(text)))
        browser.observe(False)
        original = ValueError("baseline caught failure")
        profile.write_settling_sidecar(
            type(browser), cursor, Path("unused"), lambda p, v, s: captured.append(v), (), original
        )
    assert [len(v["records"]) for v in captured] == [1, 1]
    assert [v["records"][0]["observe_call_id"] for v in captured] == [1, 2]


@pytest.mark.parametrize("primary_kind", ["none", "caught", "keyboard"])
def test_sidecar_evidence_failure_sticky_preserves_primary_identity(native, primary_kind):
    _, browser, cdp, _ = native
    cdp.script.append((0, state()))
    browser.observe(False)
    failure = OSError("durable writer refused")
    primary = (
        None
        if primary_kind == "none"
        else ValueError("original")
        if primary_kind == "caught"
        else KeyboardInterrupt()
    )

    def writer(*args):
        raise failure

    if primary_kind == "keyboard":
        with pytest.raises(KeyboardInterrupt) as caught:
            try:
                raise primary
            finally:
                profile.write_settling_sidecar(
                    type(browser), 0, Path("unused"), writer, (), sys.exception()
                )
    else:
        with pytest.raises(type(primary or failure)) as caught:
            profile.write_settling_sidecar(type(browser), 0, Path("unused"), writer, (), primary)
    assert caught.value is (primary or failure)
    if primary is not None:
        assert any("durable writer refused" in note for note in primary.__notes__)
    with pytest.raises(ValueError, match="diagnostics failed"):
        profile.settling_cursor(type(browser))


def test_sidecar_and_record_byte_bounds_refuse_completion(native, monkeypatch):
    _, browser, cdp, _ = native
    cdp.script.append((0, state()))
    browser.observe(False)
    monkeypatch.setattr(profile, "MAX_SIDECAR_BYTES", 1)
    with pytest.raises(ValueError, match="byte bound"):
        profile.write_settling_sidecar(
            type(browser), 0, Path("unused"), lambda *a: pytest.fail("must not write"), ()
        )
    assert type(browser)._observation_settling_fault


def test_sidecar_bound_uses_actual_pinned_redaction_and_pretty_bytes(native, monkeypatch, tmp_path):
    _, browser, cdp, _ = native
    cdp.script.append((0, state()))
    browser.observe(False)
    tree = ast.parse((ROOT / "scripts/jev_browser_benchmark.py").read_bytes())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in {"_redact", "_write_json"}
    ]
    scope = {"Any": object, "Path": Path, "json": json}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<pinned-json-writer>", "exec"), scope)
    target = tmp_path / "trace.json"
    value = profile.write_settling_sidecar(
        type(browser), 0, target, scope["_write_json"], ("a",), redactor=scope["_redact"]
    )
    actual = target.read_bytes()
    assert (
        actual
        == (
            json.dumps(scope["_redact"](value, ["a"]), indent=2, ensure_ascii=False) + "\n"
        ).encode()
    )
    monkeypatch.setattr(profile, "MAX_SIDECAR_BYTES", len(actual) - 1)
    other = tmp_path / "refused.json"
    with pytest.raises(ValueError, match="byte bound"):
        profile.write_settling_sidecar(
            type(browser), 0, other, scope["_write_json"], ("a",), redactor=scope["_redact"]
        )
    assert not other.exists() and type(browser)._observation_settling_fault


def test_record_bound_overflow_unknown_and_original_error_not_replaced(native):
    module, browser, cdp, _ = native
    type(browser)._observation_settling_bytes = 64 * 1024 * 1024
    original = RuntimeError("original transport failure")
    cdp.script.append((0, original))
    with pytest.raises(RuntimeError) as caught:
        browser.observe(False)
    assert caught.value is original
    assert any("diagnostic bound" in note for note in original.__notes__)
    assert type(browser)._observation_settling_fault
    with pytest.raises(RuntimeError, match="diagnostic integrity"):
        browser.observe(False)


def test_five_seconds_is_not_claimed_superset_of_old_count_window(native, canonical, tmp_path):
    module, browser, cdp, clock = native
    cdp.script.extend([(0.6, None)] * 9 + [(0.6, state("late old success"))])
    with pytest.raises(module.StalePage):
        browser.observe(False)
    assert browser.observation_settling["elapsed_seconds"] > 5
    # Execute the actual original observe method against the same native operation,
    # same fake CDP and clock. It can return a tenth snapshot later than five seconds.
    cdp.script = deque([(0.6, None)] * 9 + [(0.6, state("late old success"))])
    clock.now, clock.sleeps = 0, []
    scope = dict(module.__dict__)
    exec(
        compile(
            method_span(canonical[BROWSER], "observe").decode().replace("    ", "", 1),
            "<original-observe>",
            "exec",
        ),
        scope,
    )
    page = scope["observe"](browser, False)
    assert page["text"] == "late old success" and clock.now > 5 and not cdp.inputs
