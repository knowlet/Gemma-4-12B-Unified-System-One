"""Closed v5 sources and real owned Chromium fixtures; no model or public site."""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
import os
import types
from pathlib import Path

import jev_browser_profile_v5_benchmark as launcher
import jev_observation_profile_v5 as profile
import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER_SHA = hashlib.sha256(Path(profile.__file__).read_bytes()).hexdigest()
CHROMIUM = "/home/albert_hsieh/.cache/ms-playwright/chromium_headless_shell-1208/chrome-linux/headless_shell"


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


@pytest.fixture(scope="module")
def canonical_root():
    path = Path(os.environ.get("JEV_UPSTREAM_PATH", ROOT / "artifacts/jev-runtime/upstream"))
    if not path.is_dir():
        pytest.skip("Existing pinned source checkout required; never downloads")
    return path


@pytest.fixture(scope="module")
def bundle(canonical_root):
    return profile.build_profile(
        profile.read_canonical_sources(canonical_root), expected_source_sha256=HELPER_SHA
    )


def test_exact_v4_executor_and_all_native_policy_sources_unchanged(bundle):
    old = bundle._parent_bundle
    assert {k for k in old.runtime_files if old.runtime_files[k] != bundle.runtime_files[k]} == {
        profile.SNAPSHOT
    }
    assert sha(Path(profile._PARENT.__file__).read_bytes()) == profile.PARENT_SOURCE_SHA256
    assert bundle.manifest()["derivation"]["executor_bytes_equal_to_v4"] is True
    assert bundle.manifest()["derivation"]["operation_heads_unchanged"] is True
    assert bundle.manifest()["known_inherited_limitations"]


def test_complete_materialization_is_closed_and_rederived(tmp_path, canonical_root):
    target = tmp_path / "profile"
    result = profile.materialize_profile(canonical_root, target, expected_source_sha256=HELPER_SHA)
    handle = profile.verify_materialized_profile(
        target,
        expected_manifest_sha256=result["manifest_sha256"],
        expected_source_sha256=HELPER_SHA,
    )
    assert handle.runtime_root == target / "runtime"
    manifest = handle.manifest()
    assert manifest["profile"] == profile.PROFILE
    assert len(manifest["artifact_files"]) + 1 == 28
    assert handle.read_state == (target / "runtime" / profile.SNAPSHOT).read_text()
    for name, record in manifest["artifact_files"].items():
        blob = (target / name).read_bytes()
        assert (sha(blob), len(blob)) == (record["sha256"], record["size_bytes"])
    assert set(manifest["artifact_files"]) | {"profile-manifest.json"} == {
        p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()
    }
    # A rehashed self-description cannot authorize changed derived source bytes.
    path = target / "runtime" / profile.SNAPSHOT
    path.write_bytes(path.read_bytes() + b"\n// unauthorized\n")
    manifest["artifact_files"]["runtime/" + profile.SNAPSHOT] = {
        "sha256": sha(path.read_bytes()),
        "size_bytes": path.stat().st_size,
    }
    blob = profile._encode(manifest) + b"\n"
    (target / "profile-manifest.json").write_bytes(blob)
    with pytest.raises(ValueError, match="source/metadata"):
        profile.verify_materialized_profile(
            target, expected_manifest_sha256=sha(blob), expected_source_sha256=HELPER_SHA
        )


@pytest.mark.parametrize("case_module", ["original", "heldout"])
def test_runner_has_only_named_import_literal_and_case_substitutions(tmp_path, case_module):
    target, manifest = launcher.prepare_bundle(tmp_path / "runner", case_module=case_module)
    original = (target / "original_jev_browser_benchmark.py").read_bytes()
    current = (target / "runner.py").read_bytes()
    inverse = (
        current.replace(launcher.NEW_IMPORT, launcher.OLD_IMPORT)
        .replace(launcher.NEW_CHOICE, launcher.OLD_CHOICE)
        .replace(launcher.NEW_EVENTS_IMPORT, launcher.OLD_EVENTS_IMPORT)
    )
    if case_module == "heldout":
        inverse = inverse.replace(launcher.NEW_CASE, launcher.OLD_CASE, 1)
    assert inverse == original
    assert ast.dump(ast.parse(inverse), include_attributes=False) == ast.dump(
        ast.parse(original), include_attributes=False
    )
    assert manifest["source_delta"]["profile_import_substitutions"] == 3
    assert manifest["source_delta"]["tracker_import_substitutions"] == 1
    assert (target / "jev_observation_profile_v4.py").read_bytes() == Path(
        profile._PARENT.__file__
    ).read_bytes()
    launcher.verify_bundle(target, manifest)
    with pytest.raises(SystemExit):
        launcher._arguments(
            [
                "--upstream-path",
                "x",
                "--endpoint",
                "x",
                "--model",
                "x",
                "--output-dir",
                "x",
                "--max-steps",
                "17",
            ],
            "heldout",
            HELPER_SHA,
        )


def _admit_browser():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, text = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable"}:
            number, unit = text.split()
            assert unit == "kB"
            values[key] = int(number) * 1024
    if values["MemAvailable"] < max(24 * 1024**3, (values["MemTotal"] + 4) // 5) + 2 * 1024**3:
        pytest.skip("Host reserve plus owned Chromium unavailable")


@pytest.fixture(scope="module")
def browser_context(tmp_path_factory, canonical_root):
    if os.environ.get("JEV_BROWSER_TESTS") != "1":
        pytest.skip("Set JEV_BROWSER_TESTS=1 for actual local Chromium")
    _admit_browser()
    playwright = pytest.importorskip("playwright.sync_api")
    if not Path(CHROMIUM).is_file():
        pytest.skip("Installed Chromium required; never downloads")
    output = tmp_path_factory.mktemp("v5-runtime") / "profile"
    result = profile.materialize_profile(canonical_root, output, expected_source_sha256=HELPER_SHA)
    handle = profile.verify_materialized_profile(
        output,
        expected_manifest_sha256=result["manifest_sha256"],
        expected_source_sha256=HELPER_SHA,
    )
    blocked = []
    with playwright.sync_playwright() as runtime:
        chromium = runtime.chromium.launch(
            headless=True,
            executable_path=CHROMIUM,
            args=[
                "--disable-gpu",
                "--disable-software-rasterizer",
                "--disable-dev-shm-usage",
                "--disable-background-networking",
            ],
        )
        try:
            context = chromium.new_context(
                viewport={"width": 1120, "height": 780}, offline=True, service_workers="block"
            )

            def refuse(route):
                blocked.append(route.request.url)
                route.abort("blockedbyclient")

            context.route("**/*", refuse)
            try:
                yield context, handle
            finally:
                context.close()
        finally:
            chromium.close()
    assert not blocked, "Fixtures attempted an external request"
    profile.verify_materialized_profile(
        output,
        expected_manifest_sha256=result["manifest_sha256"],
        expected_source_sha256=HELPER_SHA,
    )


@pytest.fixture
def owned(browser_context):
    context, handle = browser_context
    page = context.new_page()
    session = context.new_cdp_session(page)
    inputs = []

    def cdp(method, session_id=None, **params):
        assert session_id == "owned-v5-fixture"
        assert method in {"Runtime.evaluate", "Input.dispatchMouseEvent"}
        if method.startswith("Input."):
            inputs.append((method, params))
        return session.send(method, params)

    def import_owned(name, *args, **kwargs):
        if name == "browser_harness.admin":
            return types.SimpleNamespace(
                ensure_daemon=lambda: (_ for _ in ()).throw(AssertionError("No daemon"))
            )
        if name == "browser_harness.helpers":
            return types.SimpleNamespace(cdp=cdp)
        return builtins.__import__(name, *args, **kwargs)

    module = types.ModuleType("actual_v5_owned_browser")
    module.__file__ = str(handle.runtime_root / profile.BROWSER)
    module.__dict__["__builtins__"] = {**vars(builtins), "__import__": import_owned}
    exec(compile(Path(module.__file__).read_bytes(), module.__file__, "exec"), module.__dict__)
    browser = object.__new__(module.Browser)
    browser.session = "owned-v5-fixture"
    try:
        yield browser, page, session, inputs, module
    finally:
        page.close()


def html(body, css=""):
    return (
        "<!doctype html><meta charset=utf-8><meta http-equiv=Content-Security-Policy "
        "content=\"default-src 'none';style-src 'unsafe-inline' data:;script-src 'unsafe-inline'\">"
        "<style>body{margin:0;font:16px sans-serif}.dropdown{position:absolute;left:100px;top:100px;"
        "width:440px;height:110px}.trigger{display:block;width:160px;height:40px;background:#ddd}"
        ".menu{display:none;position:absolute;top:45px;background:#afa}.dropdown:hover .menu{display:block}"
        + css
        + "</style><body>"
        + body
        + "<script>window.events=[];for(const type of ['pointerdown','pointerup','click','input','change'])"
        "document.addEventListener(type,e=>events.push({type,id:e.target.id||e.target.tagName,trusted:e.isTrusted}),true);</script>"
    )


MENU = '<div id=owner class=dropdown><div id=trigger class=trigger>Navigation</div><div id=popup class=menu><a id=item href="#destination">Hidden menu option text</a></div></div>'


def prepare(owned, body=MENU, css=""):
    browser, page, session, inputs, _ = owned
    page.set_content(html(body, css), wait_until="load")
    session.send(
        "Input.dispatchMouseEvent", {"type": "mouseMoved", "x": 1000, "y": 700, "button": "none"}
    )
    inputs.clear()
    return browser.observe(screenshot=False)


def hover(observed):
    return [a for a in observed["actions"] if a.get("affordance") == "css-hover-entry-v5"]


def test_hover_owner_child_uses_exact_leaf_and_short_label_after_expansion(owned):
    browser, page, _, inputs, _ = owned
    before = prepare(owned)
    (action,) = hover(before)
    assert action["label"] == "Navigation" and action["role"] == "generic"
    assert action["node"] == browser.evaluate(
        "window.__jevFast.ids.get(document.querySelector('#trigger'))"
    )
    assert action["hover_owner"] == browser.evaluate(
        "window.__jevFast.ids.get(document.querySelector('#owner'))"
    )
    assert "Hidden menu option" not in json.dumps(action)
    assert "Hidden menu option" not in json.dumps(before["guards"][str(action["node"])])
    assert before["hover_discovery"]["coverage_complete"] is True
    assert browser.fresh(before, action) is True
    assert browser.act(action, before) == {"executed": action["id"]}
    assert [v[1]["type"] for v in inputs] == ["mousePressed", "mouseReleased"]
    assert page.evaluate("events.filter(e=>e.type==='click')") == [
        {"type": "click", "id": "trigger", "trusted": True}
    ]
    after = browser.observe(screenshot=False)
    assert page.evaluate(
        "document.querySelector('#owner').innerText.includes('Hidden menu option text')"
    )
    (again,) = hover(after)
    assert (again["node"], again["label"]) == (action["node"], "Navigation")
    assert browser.fresh(before, action) is True
    assert any(a["label"] == "Hidden menu option text" for a in after["actions"])


def test_multiple_real_hover_owners_deduplicate_the_same_visible_leaf(owned):
    observed = prepare(
        owned, "<div class=outer>" + MENU + "</div>", ".outer:hover .menu{display:block}"
    )
    assert len(hover(observed)) == 1
    assert observed["hover_discovery"]["duplicate_bindings"] == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "document.querySelector('#trigger').textContent='Different entry'",
        "document.querySelector('#owner').className='no-hover-rule'",
        "document.querySelector('#trigger').outerHTML='<div id=trigger class=trigger>Navigation</div>'",
        "document.querySelector('#trigger').setAttribute('role','button')",
        "document.querySelector('#owner').inert=true",
        "(() => {const s=document.styleSheets[0];for(let i=s.cssRules.length-1;i>=0;i--)if(s.cssRules[i].selectorText?.includes(':hover'))s.deleteRule(i)})()",
        "document.querySelector('#popup').outerHTML='<div id=popup class=menu><a href=#new>Different tree</a></div>'",
    ],
)
def test_changed_binding_is_stale_before_any_input(owned, mutation):
    browser, _, _, inputs, native = owned
    before = prepare(owned)
    (action,) = hover(before)
    browser.evaluate(mutation)
    assert browser.fresh(before, action) is False
    with pytest.raises(native.StalePage):
        browser.act(action, before)
    assert inputs == []


@pytest.mark.parametrize(
    "body",
    [
        "<div class=dropdown><div class=trigger>One</div><div>Two</div><div class=menu><a href=#x>Popup</a></div></div>",
        "<div class=dropdown><div class=menu><a href=#x>Popup</a></div></div>",
        "<div class=dropdown><button class=trigger>Existing native</button><div class=menu><a href=#x>Popup</a></div></div>",
        "<div class=trigger>Ordinary roleless text</div>",
    ],
)
def test_no_ambiguous_native_or_ordinary_div_promotion(owned, body):
    observed = prepare(owned, body)
    assert hover(observed) == []


def test_v4_default_actions_and_label_proxy_preserve_order_ids_and_state(owned, bundle):
    browser, page, session, inputs, _ = owned
    body = (
        MENU
        + '<input id=check type=checkbox style="position:absolute;left:-10000px"><label for=check style="position:absolute;left:600px;top:100px">Visible setting</label><button id=normal style="position:absolute;left:600px;top:180px">Ordinary button</button>'
    )
    page.set_content(html(body))
    session.send(
        "Input.dispatchMouseEvent", {"type": "mouseMoved", "x": 1000, "y": 700, "button": "none"}
    )
    old = page.evaluate(bundle._parent_bundle.read_state)
    new = browser.observe(screenshot=False)
    assert [a for a in new["actions"] if a.get("affordance") != "css-hover-entry-v5"] == old[
        "actions"
    ]
    assert new["omitted_actions"] == old["omitted_actions"] == 0
    assert new["label_proxy_actions"] == old["label_proxy_actions"] == 1
    (action,) = [a for a in new["actions"] if a.get("control_node") is not None]
    inputs.clear()
    browser.act(action, new)
    assert page.evaluate("document.querySelector('#check').checked") is True
    assert [v[1]["type"] for v in inputs] == ["mousePressed", "mouseReleased"]


def test_inherited_native_cap_preserved_new_candidate_not_pruned(owned, bundle):
    browser, page, session, _, _ = owned
    buttons = "".join(
        '<button style="position:absolute;left:0;top:0">Native ' + str(i) + "</button>"
        for i in range(260)
    )
    page.set_content(html(buttons + MENU))
    session.send(
        "Input.dispatchMouseEvent", {"type": "mouseMoved", "x": 1000, "y": 700, "button": "none"}
    )
    old = page.evaluate(bundle._parent_bundle.read_state)
    new = browser.observe(screenshot=False)
    assert new["omitted_actions"] == old["omitted_actions"] == 10
    assert [a for a in new["actions"] if a.get("affordance") != "css-hover-entry-v5"] == old[
        "actions"
    ]
    assert len(hover(new)) == new["hover_entry_actions"] == 1
    assert new["hover_discovery"]["additional_candidate_pruning"] is False


def test_complex_css_has_explicit_incomplete_coverage(owned):
    browser, page, _, inputs, _ = owned
    page.set_content(
        html(MENU).replace(
            ".dropdown:hover .menu{display:block}", ":is(.dropdown):hover .menu{display:block}"
        )
    )
    observed = browser.observe(screenshot=False)
    assert hover(observed) == []
    assert observed["hover_discovery"]["unsupported_selectors"] == 1
    assert observed["hover_discovery"]["coverage_complete"] is False
    assert inputs == []


def test_pending_stylesheet_has_explicit_incomplete_coverage(owned):
    browser, page, _, inputs, _ = owned
    url = "http://127.0.0.1:39093/pending.css"
    held = []
    page.route(url, lambda route: held.append(route))
    page.set_content(
        html(MENU).replace(
            "style-src 'unsafe-inline' data:", "style-src 'unsafe-inline' http://127.0.0.1:39093"
        )
    )
    page.evaluate(
        """url => {const link=document.createElement('link');link.rel='stylesheet';
      link.href=url;document.head.appendChild(link)}""",
        url,
    )
    page.wait_for_timeout(10)
    try:
        assert len(held) == 1  # Intercepted and paused before any actual I/O.
        observed = browser.observe(screenshot=False)
        assert observed["hover_discovery"]["pending_stylesheets"] == 1
        assert observed["hover_discovery"]["coverage_complete"] is False
        assert len(hover(observed)) == 1 and inputs == []
    finally:
        for route in held:
            route.abort("aborted")


def test_cross_origin_cssom_security_error_is_not_guessed(owned):
    browser, page, _, inputs, _ = owned
    document_url = "http://127.0.0.1:39091/fixture"
    stylesheet_url = "http://127.0.0.1:39092/fixture.css"
    intercepted = []

    def serve_local(route):
        intercepted.append(route.request.url)
        if route.request.url == document_url:
            content = (
                html(MENU).replace(
                    "style-src 'unsafe-inline' data:",
                    "style-src 'unsafe-inline' http://127.0.0.1:39092",
                )
                + '<link rel=stylesheet href="'
                + stylesheet_url
                + '">'
            )
            route.fulfill(status=200, content_type="text/html", body=content)
        else:
            assert route.request.url == stylesheet_url
            route.fulfill(
                status=200, content_type="text/css", body=".external:hover .menu{display:block}"
            )

    # Different loopback ports are real distinct renderer origins. Both responses
    # come from memory before network I/O; no server, DNS or outside URL is used.
    page.route(document_url, serve_local)
    page.route(stylesheet_url, serve_local)
    page.goto(document_url, wait_until="load")
    assert intercepted == [document_url, stylesheet_url]
    assert (
        page.evaluate("""() => {try{document.styleSheets[1].cssRules;return null}
      catch(error){return error.name}}""")
        == "SecurityError"
    )
    observed = browser.observe(screenshot=False)
    report = observed["hover_discovery"]
    assert report["unreadable_stylesheets"] == 1
    assert report["coverage_complete"] is False
    assert len(hover(observed)) == 1  # Known readable proof remains explicit.
    assert inputs == []


@pytest.mark.parametrize(
    "css,stage,maximum",
    [
        ("".join(".unused" + str(i) + "{color:black}" for i in range(520)), "rules", 512),
        (
            "".join(".dropdown:hover .menu{display:block}" for _ in range(130)),
            "hover-selectors",
            128,
        ),
        (
            '.dropdown:hover .menu{display:block;--large:"' + "x" * 2050 + '"}',
            "declaration-chars",
            2048,
        ),
    ],
)
def test_css_budget_refuses_only_hover_batch_with_explicit_unknown_count(
    owned, bundle, css, stage, maximum
):
    browser, page, _, inputs, _ = owned
    page.set_content(html(MENU, css))
    old = page.evaluate(bundle._parent_bundle.read_state)
    observed = browser.observe(screenshot=False)
    report = observed["hover_discovery"]
    assert (
        observed["actions"] == old["actions"]
        and observed["omitted_actions"] == old["omitted_actions"]
    )
    assert observed["label_proxy_actions"] == old["label_proxy_actions"]
    assert observed["hover_entry_actions"] == report["actions"] == 0
    assert (
        report["extension_status"] == "refused_budget"
        and report["candidate_batch_committed"] is False
    )
    assert (
        report["coverage_complete"] is False and report["unscanned_hover_candidates"] == "unknown"
    )
    assert report["extension_refusal"] == {
        "kind": "css_discovery_budget",
        "stage": stage,
        "scanned": report["extension_refusal"]["scanned"],
        "maximum": maximum,
    }
    assert report["extension_refusal"]["scanned"] > maximum
    # The real menu rule occurs before the later exhaustion: its partial prefix
    # must not become an offered action even when readable proof was found first.
    assert report["hover_selectors"] >= 1 and inputs == []


def test_budget_refusal_is_valid_native_observation_and_tracker_can_recover(
    owned, browser_context, tmp_path
):
    browser, page, session, inputs, native = owned
    prepared, manifest = launcher.prepare_bundle(tmp_path / "tracker-bundle")
    tracker_module = types.ModuleType("v5_actual_copied_tracker")
    exec(
        compile(
            (prepared / "jev_observation_events_v5.py").read_bytes(), "copied-v5-tracker", "exec"
        ),
        tracker_module.__dict__,
    )
    handle = browser_context[1]
    tracker = tracker_module.ObservationProfileTracker(
        read_state=native.READ_STATE,
        marker_expression=native.MARKER,
        manifest_sha256=handle.manifest_sha256,
        profile=profile.PROFILE,
    )
    page.set_content(html(MENU, "".join(".unused" + str(i) + "{color:black}" for i in range(520))))
    first = session.send(
        "Runtime.evaluate", {"expression": native.READ_STATE, "returnByValue": True}
    )
    assert "exceptionDetails" not in first
    assert first["result"]["value"]["hover_discovery"]["extension_status"] == "refused_budget"
    tracker.capture(
        {
            "protocol_call": 1,
            "session": "owned-v5-fixture",
            "expression": native.READ_STATE,
            "response": first,
        }
    )
    assert tracker.failed is False and tracker.failures == []
    page.set_content(html(MENU))
    second = session.send(
        "Runtime.evaluate", {"expression": native.READ_STATE, "returnByValue": True}
    )
    tracker.capture(
        {
            "protocol_call": 2,
            "session": "owned-v5-fixture",
            "expression": native.READ_STATE,
            "response": second,
        }
    )
    assert tracker.failed is False and tracker.rows[-1]["profile_failure_count"] == 0
    assert tracker.rows[-1]["failure"] is None and inputs == []
    assert manifest["source_delta"]["tracker_inverse_bytes_equal"] is True


def test_old_hover_becomes_stale_on_budget_refusal_native_guards_still_work(owned, bundle):
    browser, page, _, inputs, native = owned
    body = (
        MENU
        + '<button id=normal style="position:absolute;left:650px;top:220px">Native action</button><input id=check type=checkbox style="position:absolute;left:-10000px"><label for=check style="position:absolute;left:650px;top:100px">Visible setting</label>'
    )
    before = prepare(owned, body)
    (action,) = hover(before)
    (normal,) = [a for a in before["actions"] if a.get("label") == "Native action"]
    browser.evaluate("""(() => {const s=document.createElement('style');s.id='budget';
      s.textContent=Array.from({length:520},(_,i)=>'.unused'+i+'{color:black}').join('');
      document.head.appendChild(s)})()""")
    assert browser.fresh(before, action) is False
    with pytest.raises(native.StalePage):
        browser.act(action, before)
    assert inputs == []
    # Without even rebuilding the snapshot, the old normal action keeps its
    # original v4 guard; the failed optional exploration must not poison it.
    assert browser.fresh(before, normal) is True
    browser.act(normal, before)
    assert page.evaluate("events.filter(e=>e.type==='click')") == [
        {"type": "click", "id": "normal", "trusted": True}
    ]
    assert [v[1]["type"] for v in inputs] == ["mousePressed", "mouseReleased"]
    capped = browser.observe(screenshot=False)
    old = page.evaluate(bundle._parent_bundle.read_state)
    assert (
        capped["actions"] == old["actions"]
        and capped["label_proxy_actions"] == old["label_proxy_actions"] == 1
    )
    assert capped["hover_entry_actions"] == 0
    browser.evaluate("document.querySelector('#budget').remove()")
    restored = browser.observe(screenshot=False)
    assert len(hover(restored)) == 1
    assert browser.fresh(before, action) is False  # Refused guard epoch never revives.


@pytest.mark.parametrize(
    "mutation",
    [
        "Object.defineProperty(document.styleSheets[0],'cssRules',{get(){throw Error('fixture unexpected CSS')},configurable:true})",
        "Element.prototype.checkVisibility=()=>{throw Error('fixture unexpected native')}",
    ],
)
def test_unexpected_css_and_native_failures_are_not_swallowed(owned, mutation):
    browser, page, _, inputs, native = owned
    prepare(owned)
    browser.evaluate(mutation)
    with pytest.raises(Exception, match="fixture unexpected"):
        page.evaluate(native.READ_STATE)
    with pytest.raises(native.StalePage):
        browser.observe(screenshot=False)
    assert inputs == []


def test_delayed_visibility_does_not_implicitly_wait_or_open(owned):
    browser, page, _, inputs, _ = owned
    css = ".menu{display:block;visibility:hidden;transition:visibility 0s linear 180ms}.dropdown:hover .menu{visibility:visible}"
    before = prepare(owned, css=css)
    (action,) = hover(before)
    assert not page.evaluate(
        "document.querySelector('#popup').checkVisibility({checkVisibilityCSS:true,checkOpacity:true})"
    )
    assert inputs == []
    browser.act(action, before)
    assert not page.evaluate(
        "document.querySelector('#popup').checkVisibility({checkVisibilityCSS:true,checkOpacity:true})"
    )
    assert [v[1]["type"] for v in inputs] == ["mousePressed", "mouseReleased"]
    page.wait_for_timeout(230)
    assert page.evaluate(
        "document.querySelector('#popup').checkVisibility({checkVisibilityCSS:true,checkOpacity:true})"
    )
    after = browser.observe(screenshot=False)
    assert hover(after)[0]["label"] == "Navigation"
    assert after["hover_discovery"]["synchronous_only"] is True


def test_inherited_hover_overlay_target_split_is_explicitly_not_fixed(owned):
    browser, page, _, inputs, _ = owned
    # Overlay is outside the controlled popup, so its first hidden state does not
    # add another visible trigger. It can cover the selected leaf after hover.
    body = MENU.replace("<div id=popup", "<div id=overlay>Overlay</div><div id=popup")
    before = prepare(
        owned,
        body,
        "#overlay{display:none;position:absolute;left:0;top:0;width:160px;height:40px;background:red;z-index:9}.dropdown:hover #overlay{display:block}",
    )
    (action,) = hover(before)
    browser.act(action, before)
    clicked = page.evaluate("events.filter(e=>e.type==='click')")
    assert clicked == [{"type": "click", "id": "owner", "trusted": True}]
    assert [v[1]["type"] for v in inputs] == ["mousePressed", "mouseReleased"]
