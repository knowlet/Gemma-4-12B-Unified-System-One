"""V4 source contracts and owned Chromium interaction regressions.

The browser checks use local HTML, real Chromium input, and the verified copied
native Browser. They never call a policy, helper, public site, or Modal. Opt in
with JEV_BROWSER_TESTS=1; no browser or dependency is downloaded.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import jev_browser_profile_v4_benchmark as launcher
import jev_observation_profile_v4 as profile
import pytest

from s1 import jev_observation_profile as parent

ROOT = Path(__file__).resolve().parents[1]
PROFILE = "composed-visible-controls-v4"
PARENT_SHA = "8d199ac1564af05892e9b21e225d1fdf500f1953840855c902160ed8d56df89d"
BENCHMARK_SHA = "fe1410675e4decb06a8491a5a83d2ec965629c99f8912b797446c4cb2c155f6c"
EVENTS_SHA = "ee6d3ea04d948fd8947d12934b63564494aab6a67cb8d8063ea7e12fb84a0664"
HELPER_SHA = hashlib.sha256(Path(profile.__file__).read_bytes()).hexdigest()
CHROMIUM = "/home/albert_hsieh/.cache/ms-playwright/chromium_headless_shell-1208/chrome-linux/headless_shell"


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


@pytest.fixture(scope="module")
def canonical_root():
    root = Path(os.environ.get("JEV_UPSTREAM_PATH", ROOT / "artifacts/jev-runtime/upstream"))
    if not root.is_dir():
        pytest.skip("An existing pinned Jev checkout is required; never fetches")
    return root.resolve()


@pytest.fixture
def canonical(canonical_root):
    return profile.read_canonical_sources(canonical_root)


def build(canonical):
    return profile.build_profile(canonical, expected_source_sha256=HELPER_SHA, profile=PROFILE)


def materialize(root, target):
    return profile.materialize_profile(
        root, target, expected_source_sha256=HELPER_SHA, profile=PROFILE
    )


def verify(target, digest):
    return profile.verify_materialized_profile(
        target,
        expected_manifest_sha256=digest,
        expected_source_sha256=HELPER_SHA,
        profile=PROFILE,
    )


def test_v3_and_original_evaluator_remain_frozen(canonical):
    assert sha(Path(parent.__file__).read_bytes()) == PARENT_SHA
    assert sha((ROOT / "scripts/jev_browser_benchmark.py").read_bytes()) == BENCHMARK_SHA
    old = parent.build_profile(
        canonical, expected_source_sha256=PARENT_SHA, profile=parent.PRUNED_PROFILE
    )
    assert sha(old.runtime_files[parent.SNAPSHOT]) == (
        "c694f308f29ed5000b7bb60e9374afd9398413d2811e07741cf0f0a6321be552"
    )
    new = build(canonical)
    assert new.manifest()["profile"] == PROFILE
    assert set(new.runtime_files) == set(canonical)
    assert {name for name in canonical if canonical[name] != new.runtime_files[name]} == {
        parent.SNAPSHOT,
        "jev_ultrafast/browser.py",
    }
    # The model request builder and native execution contract outside Browser
    # retain their reviewed bytes, including operation and candidate heads.
    for name in (
        "jev_ultrafast/model.py",
        "jev_ultrafast/questions.py",
        "jev_ultrafast/agent.py",
    ):
        assert new.runtime_files[name] == canonical[name]


@pytest.mark.parametrize("alias", ["v4", PROFILE + " ", "composed-visible-text-v4"])
def test_no_implicit_profile_alias(canonical, alias):
    with pytest.raises(ValueError):
        profile.build_profile(canonical, expected_source_sha256=HELPER_SHA, profile=alias)


@pytest.mark.parametrize("name", tuple(parent.CANONICAL_SHA256))
def test_each_original_source_is_authenticated(canonical, name):
    canonical[name] += b"\n"
    with pytest.raises(ValueError):
        build(canonical)


def test_complete_materialization_is_reverified_and_detached(tmp_path, canonical_root):
    target = tmp_path / "profile"
    result = materialize(canonical_root, target)
    manifest_blob = (target / "profile-manifest.json").read_bytes()
    assert result["manifest_sha256"] == sha(manifest_blob)
    handle = verify(target, result["manifest_sha256"])
    assert handle.path == target and handle.runtime_root == target / "runtime"
    assert handle.read_state == (handle.runtime_root / parent.SNAPSHOT).read_text()
    assert handle.marker_expression == (
        f"(() => {{ const state={handle.read_state}; return state?.marker ?? null; }})()"
    )
    manifest = handle.manifest()
    assert manifest["profile"] == PROFILE
    assert manifest["helper_source_sha256"] == HELPER_SHA
    assert manifest["canonical_source_sha256"] == dict(parent.CANONICAL_SHA256)
    for name, record in manifest["artifact_files"].items():
        blob = (target / name).read_bytes()
        assert sha(blob) == record["sha256"] and len(blob) == record["size_bytes"]
    expected = {*manifest["artifact_files"], "profile-manifest.json"}
    assert {p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()} == expected
    copies = [p.read_bytes() for p in target.rglob("*.py")]
    assert Path(parent.__file__).read_bytes() in copies
    assert Path(profile.__file__).read_bytes() in copies
    manifest["runtime_source_sha256"][parent.SNAPSHOT] = "0" * 64
    assert handle.manifest()["runtime_source_sha256"][parent.SNAPSHOT] != "0" * 64
    with pytest.raises(ValueError):
        materialize(canonical_root, target)


@pytest.mark.parametrize("name", [parent.SNAPSHOT, "jev_ultrafast/browser.py"])
def test_runtime_mutation_cannot_be_accepted_with_original_manifest(tmp_path, canonical_root, name):
    target = tmp_path / "profile"
    result = materialize(canonical_root, target)
    path = target / "runtime" / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        verify(target, result["manifest_sha256"])


@pytest.mark.parametrize("mutation", ["limits", "runtime", "parent_copy"])
def test_rehashing_does_not_authorize_changed_source_or_limits(tmp_path, canonical_root, mutation):
    target = tmp_path / "profile"
    materialize(canonical_root, target)
    path = target / "profile-manifest.json"
    manifest = json.loads(path.read_bytes())
    if mutation == "limits":
        manifest["limits"]["visited_nodes"] = 20001
    else:
        label = (
            "runtime/jev_ultrafast/browser.py"
            if mutation == "runtime"
            else manifest["parent_helper_copy"]
        )
        changed = (target / label).read_bytes() + b"\n# changed\n"
        (target / label).write_bytes(changed)
        manifest["artifact_files"][label] = {"sha256": sha(changed), "size_bytes": len(changed)}
        if mutation == "runtime":
            manifest["runtime_source_sha256"]["jev_ultrafast/browser.py"] = sha(changed)
        else:
            manifest["parent_helper_source_sha256"] = sha(changed)
    blob = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(blob + b"\n")
    with pytest.raises(ValueError):
        verify(target, sha(path.read_bytes()))


@pytest.mark.parametrize("mutation", ["extra_file", "extra_directory", "symlink"])
def test_materialized_population_is_closed(tmp_path, canonical_root, mutation):
    target = tmp_path / "profile"
    result = materialize(canonical_root, target)
    if mutation == "extra_file":
        (target / "unexpected.py").write_bytes(b"pass\n")
    elif mutation == "extra_directory":
        (target / "unexpected").mkdir()
    else:
        path = target / "runtime" / parent.SNAPSHOT
        copy = tmp_path / "snapshot.js"
        copy.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(copy)
    with pytest.raises(ValueError):
        verify(target, result["manifest_sha256"])


@pytest.mark.parametrize("case_module", ["original", "heldout"])
def test_runner_changes_only_explicit_profile_tracker_imports_and_optional_cases(
    tmp_path, case_module
):
    original = (ROOT / "scripts/jev_browser_benchmark.py").read_bytes()
    bundle, manifest = launcher.prepare_bundle(tmp_path / "bundle", case_module=case_module)
    assert manifest["case005_interface_exposed"] is True
    assert (bundle / "original_jev_browser_benchmark.py").read_bytes() == original
    assert (bundle / "jev_observation_profile_v4.py").read_bytes() == Path(
        profile.__file__
    ).read_bytes()
    assert (bundle / "jev_observation_profile_v3_parent.py").read_bytes() == Path(
        parent.__file__
    ).read_bytes()
    derived = ast.parse((bundle / "runner.py").read_bytes())
    events = (ROOT / "src/s1/jev_observation_events.py").read_bytes()
    assert sha(events) == EVENTS_SHA
    assert (bundle / "jev_observation_events_v3_parent.py").read_bytes() == events
    copied_events = (bundle / "jev_observation_events_v4.py").read_bytes()
    before, after = (
        b'PRUNED_PROFILE = "composed-visible-text-v3"',
        b'PRUNED_PROFILE = "composed-visible-controls-v4"',
    )
    assert copied_events.count(after) == 1
    assert copied_events.replace(after, before, 1) == events
    imports = literals = case_imports = tracker_imports = 0

    class Restore(ast.NodeTransformer):
        def visit_Import(self, node):
            nonlocal imports
            if len(node.names) == 1 and (
                node.names[0].name,
                node.names[0].asname,
            ) == ("jev_observation_profile_v4", "profile"):
                imports += 1
                return ast.ImportFrom(
                    module="s1",
                    names=[ast.alias(name="jev_observation_profile", asname="profile")],
                    level=0,
                )
            return self.generic_visit(node)

        def visit_Constant(self, node):
            nonlocal literals
            if node.value == PROFILE:
                literals += 1
                return ast.Constant(value=parent.PRUNED_PROFILE)
            return node

        def visit_ImportFrom(self, node):
            nonlocal case_imports, tracker_imports
            if node.module == "jev_heldout_public_cases":
                case_imports += 1
                node.module = "jev_public_web_cases"
            if node.module == "jev_observation_events_v4":
                tracker_imports += 1
                node.module = "s1.jev_observation_events"
            return self.generic_visit(node)

    restored = Restore().visit(derived)
    assert (imports, literals, case_imports, tracker_imports) == (
        3,
        1,
        int(case_module == "heldout"),
        1,
    )
    assert ast.dump(restored, include_attributes=False) == ast.dump(
        ast.parse(original), include_attributes=False
    )
    launcher.verify_bundle(bundle, manifest)


def _memory_admission():
    if not Path("/proc/meminfo").is_file():
        pytest.skip("Owned Chromium requires a fresh Linux memory check")
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable"}:
            amount, unit = value.split()
            assert unit == "kB"
            memory[key] = int(amount) * 1024
    reserve = max(24 * 1024**3, (memory["MemTotal"] + 4) // 5)
    if memory["MemAvailable"] < reserve + 2 * 1024**3:
        pytest.skip("Host reserve plus 2 GiB owned Chromium is unavailable")


@pytest.fixture(scope="module")
def browser_runtime(tmp_path_factory, canonical_root):
    if os.environ.get("JEV_BROWSER_TESTS") != "1":
        pytest.skip("Set JEV_BROWSER_TESTS=1 for real local Chromium checks")
    _memory_admission()
    playwright = pytest.importorskip("playwright.sync_api")
    pytest.importorskip("httpx")
    executable = Path(os.environ.get("JEV_CHROMIUM_PATH", CHROMIUM))
    if not executable.is_file():
        pytest.skip("An installed local Chromium executable is required; never downloads")
    bundle, manifest = launcher.prepare_bundle(tmp_path_factory.mktemp("v4-runner") / "bundle")
    name = "jev_v4_owned_fixture_runner"
    spec = importlib.util.spec_from_file_location(name, bundle / "runner.py")
    harness = importlib.util.module_from_spec(spec)
    sys.modules[name] = harness
    previous_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(harness)
        tracker_spec = importlib.util.spec_from_file_location(
            "jev_v4_owned_fixture_tracker", bundle / "jev_observation_events_v4.py"
        )
        tracker_module = importlib.util.module_from_spec(tracker_spec)
        tracker_spec.loader.exec_module(tracker_module)
    finally:
        sys.dont_write_bytecode = previous_bytecode
    output = tmp_path_factory.mktemp("v4-materialization") / "profile"
    result = materialize(canonical_root, output)
    verified = verify(output, result["manifest_sha256"])
    old = parent.build_profile(
        parent.read_canonical_sources(canonical_root),
        expected_source_sha256=PARENT_SHA,
        profile=parent.PRUNED_PROFILE,
    )
    blocked = []
    with playwright.sync_playwright() as runtime:
        chromium = runtime.chromium.launch(
            headless=True,
            executable_path=str(executable),
            args=[
                "--disable-gpu",
                "--disable-software-rasterizer",
                "--disable-dev-shm-usage",
                "--disable-background-networking",
            ],
        )
        try:
            context = chromium.new_context(
                viewport={"width": 1120, "height": 780}, service_workers="block"
            )

            def refuse(route):
                blocked.append((route.request.method, route.request.url))
                route.abort("blockedbyclient")

            context.route("**/*", refuse)

            class Transport(harness.PlaywrightCDP):
                def __init__(self):
                    super().__init__(context)
                    self.inputs = []

                def cdp(self, method, session_id=None, **params):
                    if method.startswith("Input."):
                        self.inputs.append((method, dict(params)))
                    if method == "Page.navigate":
                        assert params["url"] == "about:blank"
                    return super().cdp(method, session_id=session_id, **params)

            transport = Transport()
            transport.v4_profile = verified
            transport.v4_tracker_class = tracker_module.ObservationProfileTracker
            with harness.load_upstream(canonical_root, transport, observation_runtime=verified) as (
                _,
                model,
            ):
                native = sys.modules["jev_ultrafast.browser"]
                try:
                    assert Path(native.__file__).resolve() == verified.runtime_root / (
                        "jev_ultrafast/browser.py"
                    )
                    assert native.READ_STATE == verified.read_state
                    yield native, transport, old
                finally:
                    model.CLIENT.close()
            context.close()
            assert blocked == [], "Owned fixtures attempted an external request"
        finally:
            chromium.close()
            launcher.verify_bundle(bundle, manifest)
            sys.modules.pop(name, None)


@pytest.fixture
def owned_page(browser_runtime):
    _memory_admission()
    native, transport, old = browser_runtime
    browser = native.Browser("about:blank")
    verified = transport.v4_profile
    tracker = transport.v4_tracker_class(
        read_state=verified.read_state,
        marker_expression=verified.marker_expression,
        manifest_sha256=verified.manifest_sha256,
        profile=PROFILE,
    )
    transport.v4_tracker = tracker
    transport.observation_profile_sink = tracker.capture
    try:
        yield browser, transport.targets[browser.target], transport, old, native
    finally:
        transport.observation_profile_sink = None
        browser.close()
        assert tracker.rows and tracker.failed is False


def html(body):
    return (
        "<!doctype html><meta charset=utf-8><title>Owned controls</title><style>"
        "body{margin:20px;font:16px sans-serif}.line{margin:12px 0}"
        ".away{position:absolute;left:-15934px;width:18px;height:18px}"
        "label{display:inline-block;padding:10px;border:1px solid #777}"
        "</style><body>"
        + body
        + "<script>window.events=[];for(const t of ['click','input','change'])"
        "document.addEventListener(t,e=>events.push({type:e.type,id:e.target.id,"
        "trusted:e.isTrusted,checked:e.target.checked??null}));</script>"
    )


def proxy_for(browser, observed, selector):
    owner = browser.evaluate(
        f"window.__jevFast.ids.get(document.querySelector({json.dumps(selector)}))"
    )
    actions = [a for a in observed["actions"] if a.get("control_node") == owner]
    assert len(actions) == 1, observed["actions"]
    return actions[0]


def assert_clicks(transport, before):
    calls = transport.inputs[before:]
    assert [method for method, _ in calls] == ["Input.dispatchMouseEvent"] * 2
    assert [params["type"] for _, params in calls] == ["mousePressed", "mouseReleased"]
    assert all(params["button"] == "left" and params["clickCount"] == 1 for _, params in calls)


def test_actual_snapshot_and_marker_are_accepted_by_copied_v4_tracker(owned_page):
    browser, page, transport, _, _ = owned_page
    page.set_content(
        html("<input class=away id=owner type=checkbox><label for=owner>Tracked</label>")
    )
    observed = browser.observe(screenshot=False)
    assert browser.fresh(observed) is True
    rows = transport.v4_tracker.rows
    assert [row["expression_kind"] for row in rows] == ["READ_STATE", "MARKER"]
    assert rows[0]["response"]["result"]["value"]["observation_profile"]["id"] == PROFILE
    assert all(row["failure"] is None for row in rows)


@pytest.mark.parametrize("checked", [False, True])
def test_offscreen_checkbox_label_uses_trusted_native_click_and_updates_observation(
    owned_page, checked
):
    browser, page, transport, _, _ = owned_page
    page.set_content(
        html(
            "<input class=away id=owner type=checkbox "
            + ("checked" if checked else "")
            + "><label id=label for=owner>Visible filter</label>"
        )
    )
    assert browser.evaluate("document.querySelector('#owner').getBoundingClientRect().x") < -15000
    observed = browser.observe(screenshot=False)
    action = proxy_for(browser, observed, "#owner")
    assert action["kind"] == "click" and action["role"] == action["control_type"] == "checkbox"
    assert action["checked"] == str(checked).lower()
    assert action["node"] == browser.evaluate(
        "window.__jevFast.ids.get(document.querySelector('#label'))"
    )
    assert action["node"] != action["control_node"]
    assert "Visible filter" in action["label"] and "Visible filter" in observed["text"]
    assert browser.fresh(observed, action) is True
    first = len(transport.inputs)
    assert browser.act(action, observed) == {"executed": action["id"]}
    assert_clicks(transport, first)
    assert browser.evaluate("document.querySelector('#owner').checked") is not checked
    events = browser.evaluate("window.events")
    changes = [e for e in events if e["type"] == "change" and e["id"] == "owner"]
    assert changes == [{"type": "change", "id": "owner", "trusted": True, "checked": not checked}]
    after = browser.observe(screenshot=False)
    assert proxy_for(browser, after, "#owner")["checked"] == str(not checked).lower()
    assert after["omitted_actions"] == 0


def test_label_prevent_default_is_honored_without_assigning_checked(owned_page):
    browser, page, transport, _, _ = owned_page
    page.set_content(
        html(
            "<input class=away id=owner type=checkbox><label id=label for=owner>Cancel toggle</label>"
        )
    )
    browser.evaluate(
        "document.querySelector('#label').addEventListener('click',e=>e.preventDefault())"
    )
    observed = browser.observe(screenshot=False)
    action = proxy_for(browser, observed, "#owner")
    first = len(transport.inputs)
    browser.act(action, observed)
    assert_clicks(transport, first)
    assert browser.evaluate("document.querySelector('#owner').checked") is False
    assert browser.evaluate("events.some(e=>e.id==='label'&&e.type==='click'&&e.trusted)") is True
    assert browser.evaluate("events.some(e=>e.id==='owner'&&e.type==='change')") is False


def test_radio_label_respects_native_group_exclusivity(owned_page):
    browser, page, transport, _, _ = owned_page
    page.set_content(
        html(
            "<input class=away id=a name=group type=radio checked><label for=a>First</label>"
            "<input class=away id=b name=group type=radio><label for=b>Second</label>"
        )
    )
    observed = browser.observe(screenshot=False)
    action = proxy_for(browser, observed, "#b")
    assert action["role"] == action["control_type"] == "radio" and action["checked"] == "false"
    first = len(transport.inputs)
    browser.act(action, observed)
    assert_clicks(transport, first)
    assert browser.evaluate("['a','b'].map(id=>document.getElementById(id).checked)") == [
        False,
        True,
    ]
    assert browser.evaluate("events.filter(e=>e.type==='change').map(e=>[e.id,e.trusted])") == [
        ["b", True]
    ]


@pytest.mark.parametrize(
    "owner",
    [
        "<input class=away id=owner type=checkbox disabled>",
        "<fieldset disabled><input class=away id=owner type=checkbox></fieldset>",
        "<input class=away id=owner type=checkbox aria-disabled=true>",
        "<div aria-disabled=true><input class=away id=owner type=checkbox></div>",
        "<div inert><input class=away id=owner type=checkbox></div>",
    ],
)
def test_disabled_or_inert_owner_is_not_offered_through_its_visible_label(owned_page, owner):
    browser, page, transport, _, _ = owned_page
    page.set_content(html(owner + "<label for=owner>Unavailable control</label>"))
    first = len(transport.inputs)
    observed = browser.observe(screenshot=False)
    assert not any("control_node" in action for action in observed["actions"])
    assert len(transport.inputs) == first
    assert browser.evaluate("document.querySelector('#owner').checked") is False


def test_multiple_labels_deduplicate_owner_and_visible_native_control_needs_no_proxy(owned_page):
    browser, page, transport, _, _ = owned_page
    page.set_content(
        html(
            "<input class=away id=owner type=checkbox><label for=owner>First label</label>"
            "<label for=owner>Second label</label><div class=line>"
            "<input id=visible type=checkbox><label for=visible>Native visible control</label></div>"
        )
    )
    observed = browser.observe(screenshot=False)
    action = proxy_for(browser, observed, "#owner")
    assert len([a for a in observed["actions"] if "control_node" in a]) == 1
    native = [
        a for a in observed["actions"] if a.get("role") == "checkbox" and "control_node" not in a
    ]
    assert len(native) == 1 and native[0]["label"] == "Native visible control"
    first = len(transport.inputs)
    browser.act(action, observed)
    assert_clicks(transport, first)
    assert browser.evaluate("document.querySelector('#owner').checked") is True


def test_implicit_label_with_plain_text_descendant_activates_its_exact_owner(owned_page):
    browser, page, transport, _, _ = owned_page
    page.set_content(
        html(
            "<label id=label><input class=away id=owner type=checkbox>"
            "<span>Implicit association</span></label>"
        )
    )
    observed = browser.observe(screenshot=False)
    action = proxy_for(browser, observed, "#owner")
    first = len(transport.inputs)
    browser.act(action, observed)
    assert_clicks(transport, first)
    assert browser.evaluate("document.querySelector('#owner').checked") is True
    assert browser.evaluate("events.filter(e=>e.type==='change').map(e=>[e.id,e.trusted])") == [
        ["owner", True]
    ]


@pytest.mark.parametrize("tag", ["button", "a"])
def test_label_center_interactive_descendant_is_not_misrepresented_as_control(owned_page, tag):
    browser, page, transport, _, _ = owned_page
    attributes = 'href="#owned-fragment"' if tag == "a" else "type=button"
    page.set_content(
        html(
            "<input class=away id=owner type=checkbox>"
            '<label id=label for=owner style="padding:0;width:240px;height:70px">'
            f'<{tag} id=interactive {attributes} style="display:block;box-sizing:border-box;'
            f'width:100%;height:100%">Independent native target</{tag}></label>'
        )
    )
    first = len(transport.inputs)
    observed = browser.observe(screenshot=False)
    assert not any("control_node" in action for action in observed["actions"])
    assert any(a.get("role") == ("link" if tag == "a" else "button") for a in observed["actions"])
    assert len(transport.inputs) == first
    assert browser.evaluate("document.querySelector('#owner').checked") is False


def test_new_interactive_descendant_makes_old_label_action_stale(owned_page):
    browser, page, transport, _, native = owned_page
    page.set_content(
        html(
            "<input class=away id=owner type=checkbox>"
            '<label id=label for=owner style="padding:0;width:240px;height:70px">Plain label</label>'
        )
    )
    observed = browser.observe(screenshot=False)
    action = proxy_for(browser, observed, "#owner")
    browser.evaluate(
        "document.querySelector('#label').innerHTML="
        "'<button type=button style=\"width:100%;height:100%\">Independent target</button>'"
    )
    assert browser.fresh(observed, action) is False
    first = len(transport.inputs)
    with pytest.raises(native.StalePage):
        browser.act(action, observed)
    assert len(transport.inputs) == first
    assert browser.evaluate("document.querySelector('#owner').checked") is False


MUTATIONS = {
    "association": "document.querySelector('#label').htmlFor='other'",
    "type": "document.querySelector('#owner').type='radio'",
    "state": "document.querySelector('#owner').checked=true",
    "disabled": "document.querySelector('#owner').disabled=true",
    "fieldset": "document.querySelector('#fieldset').disabled=true",
    "owner_replacement": (
        "document.querySelector('#owner').replaceWith(document.querySelector('#owner').cloneNode(true))"
    ),
    "radio_name": "document.querySelector('#owner').name='changed-group'",
    "radio_form": "document.querySelector('#owner').setAttribute('form','changed-form')",
}


def setup_mutable(page, *, radio=False):
    control_type = "radio name=original-group form=original-form" if radio else "checkbox"
    page.set_content(
        html(
            "<form id=original-form></form><form id=changed-form></form>"
            f"<fieldset id=fieldset><input class=away id=owner type={control_type}></fieldset>"
            "<input class=away id=other type=checkbox><label id=label for=owner>Mutable association</label>"
        )
    )


@pytest.mark.parametrize("mutation", MUTATIONS)
def test_stale_owner_identity_type_state_and_disabled_refuse_before_input(owned_page, mutation):
    browser, page, transport, _, native = owned_page
    setup_mutable(page, radio=mutation in {"radio_name", "radio_form"})
    observed = browser.observe(screenshot=False)
    action = proxy_for(browser, observed, "#owner")
    browser.evaluate(MUTATIONS[mutation])
    assert browser.fresh(observed, action) is False
    first = len(transport.inputs)
    with pytest.raises(native.StalePage):
        browser.act(action, observed)
    assert len(transport.inputs) == first
    assert browser.evaluate("events.some(e=>e.trusted)") is False
    if mutation in {"radio_name", "radio_form"}:
        assert browser.evaluate("document.querySelector('#owner').checked") is False


@pytest.mark.parametrize("mutation", MUTATIONS)
def test_executor_rechecks_owner_after_freshness_before_dispatch(owned_page, monkeypatch, mutation):
    browser, page, transport, _, native = owned_page
    setup_mutable(page, radio=mutation in {"radio_name", "radio_form"})
    observed = browser.observe(screenshot=False)
    action = proxy_for(browser, observed, "#owner")
    original = browser.fresh

    def change_after_fresh(page, candidate=None):
        assert original(page, candidate) is True
        browser.evaluate(MUTATIONS[mutation])
        return True

    monkeypatch.setattr(browser, "fresh", change_after_fresh)
    first = len(transport.inputs)
    with pytest.raises(native.StalePage):
        browser.act(action, observed)
    assert len(transport.inputs) == first
    assert browser.evaluate("events.some(e=>e.trusted)") is False
    if mutation in {"radio_name", "radio_form"}:
        assert browser.evaluate("document.querySelector('#owner').checked") is False


def test_proxy_label_becoming_native_button_stales_old_proxy_and_executes_new_action(owned_page):
    browser, page, transport, _, native = owned_page
    page.set_content(
        html(
            "<input class=away id=owner type=checkbox>"
            "<label id=label for=owner>Dynamic native label</label>"
        )
    )
    before = browser.observe(screenshot=False)
    old_action = proxy_for(browser, before, "#owner")
    browser.evaluate("document.querySelector('#label').setAttribute('role','button')")
    observed = browser.observe(screenshot=False)
    action = next(a for a in observed["actions"] if a.get("node") == old_action["node"])
    assert action["kind"] == "click" and action["role"] == "button"
    assert "control_node" not in action
    assert observed["guards"][str(action["node"])] is not None
    assert browser.fresh(before, old_action) is False
    first = len(transport.inputs)
    with pytest.raises(native.StalePage):
        browser.act(old_action, before)
    assert len(transport.inputs) == first
    assert browser.evaluate("document.querySelector('#owner').checked") is False
    assert browser.fresh(observed, action) is True
    browser.act(action, observed)
    assert_clicks(transport, first)
    assert browser.evaluate("document.querySelector('#owner').checked") is True
    assert browser.evaluate("events.filter(e=>e.type==='change').map(e=>[e.id,e.trusted])") == [
        ["owner", True]
    ]


@pytest.mark.parametrize("after_observation", [False, True])
def test_covered_label_is_never_clicked(owned_page, after_observation):
    browser, page, transport, _, native = owned_page
    page.set_content(
        html("<input class=away id=owner type=checkbox><label id=label for=owner>Covered</label>")
    )
    observed = browser.observe(screenshot=False)
    action = proxy_for(browser, observed, "#owner")
    browser.evaluate(
        "(() => {const r=document.querySelector('#label').getBoundingClientRect();"
        "const cover=document.createElement('div');cover.id='cover';Object.assign(cover.style,"
        "{position:'fixed',left:r.x+'px',top:r.y+'px',width:r.width+'px',height:r.height+'px',"
        "zIndex:'9999',background:'black'});document.body.append(cover)})()"
    )
    first = len(transport.inputs)
    if after_observation:
        with pytest.raises(native.StalePage):
            browser.act(action, observed)
    else:
        covered = browser.observe(screenshot=False)
        assert not any("control_node" in a for a in covered["actions"])
    assert len(transport.inputs) == first
    assert browser.evaluate("document.querySelector('#owner').checked") is False


def test_normal_actions_text_ids_and_guards_match_unchanged_v3(owned_page):
    browser, page, transport, old, _ = owned_page
    page.set_content(
        html(
            '<button id=button>Ordinary button</button><input aria-label="Editable" value="old">'
            '<select aria-label="Native choice"><option value=a>First</option><option value=b>Second</option></select>'
            "<input id=visible type=checkbox><label for=visible>Visible checkbox</label>"
            "<input class=away id=owner type=checkbox><label for=owner>Offscreen owner</label>"
            '<div id=clicked></div><script>document.querySelector("#button").onclick=()=>{'
            'document.querySelector("#clicked").textContent="Clicked through native executor"}</script>'
        )
    )
    before = browser.evaluate(old.read_state)
    observed = browser.observe(screenshot=False)
    assert [a for a in observed["actions"] if "control_node" not in a] == before["actions"]
    assert observed["text"] == before["text"] and observed["page_key"] == before["page_key"]
    for node, guard in before["guards"].items():
        assert observed["guards"][node] == guard
    assert observed["observation_profile"]["id"] == PROFILE
    assert observed["omitted_actions"] == before["omitted_actions"] == 0
    action = next(a for a in observed["actions"] if a.get("label") == "Ordinary button")
    first = len(transport.inputs)
    browser.act(action, observed)
    assert_clicks(transport, first)
    assert browser.evaluate("document.querySelector('#clicked').textContent") == (
        "Clicked through native executor"
    )


def test_proxy_append_preserves_original_candidate_cap_and_native_order(owned_page):
    browser, page, _, old, _ = owned_page
    page.set_content(
        html(
            '<div style="display:grid;grid-template-columns:repeat(25,28px)">'
            + "".join(
                f'<button style="width:26px;height:24px;padding:0">{i}</button>' for i in range(251)
            )
            + "</div><input class=away id=owner type=checkbox><label for=owner>After native candidates</label>"
        )
    )
    before = browser.evaluate(old.read_state)
    observed = browser.observe(screenshot=False)
    assert [a for a in observed["actions"] if "control_node" not in a] == before["actions"]
    assert proxy_for(browser, observed, "#owner")["checked"] == "false"
    assert before["omitted_actions"] == observed["omitted_actions"] == 1
    assert observed["label_proxy_actions"] == 1
