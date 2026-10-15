"""Opt-in owned Chromium regressions; fake policy, no model or public requests.

Set JEV_BROWSER_TESTS=1 and JEV_UPSTREAM_PATH to an existing pinned checkout.
JEV_CHROMIUM_PATH optionally selects an already installed Chromium executable.
Nothing downloads or installs a browser. These checks are not task scores.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import pytest

from s1 import jev_observation_profile as profile

pytestmark = pytest.mark.skipif(
    os.environ.get("JEV_BROWSER_TESTS") != "1",
    reason="Set JEV_BROWSER_TESTS=1 and JEV_UPSTREAM_PATH for owned Chromium checks",
)
ROOT = Path(__file__).resolve().parents[1]
CONTROLS = Path(__file__).with_name("fixtures") / "jev_composed_text_controls.html"
VISIBLE = (
    "ASSIGNED_VISIBLE",
    "SHADOW_BEFORE",
    "SHADOW_AFTER",
    "NESTED_VISIBLE",
    "DISPLAY_CONTENTS_VISIBLE",
    "DIRECT_SLOT_VISIBLE",
    "RANGE_VISIBLE",
)
HIDDEN = (
    "UNASSIGNED_SECRET",
    "CLOSED_SECRET",
    "HIDE_SECRET",
    "INERT_SECRET",
    "ARIA_SECRET",
    "OPACITY_SECRET",
    "VISIBILITY_SECRET",
    "OFFSCREEN_SECRET",
    "DIRECT_OPACITY_SECRET",
    "DIRECT_ARIA_SECRET",
    "DIRECT_INERT_SECRET",
    "RANGE_HIDDEN",
)
SHADOW_DONE = """<!doctype html><title>Owned native DONE guard</title><body>
<style>body{font:14px monospace}x-code{display:block}</style>
<button>Ordinary button</button><label>Editable <input value="unchanged"></label>
<select><option>First</option><option>Second</option></select><x-code id="code"></x-code>
<script>
const root=document.querySelector('#code').attachShadow({mode:'open'});
root.innerHTML='<pre><code>alpha_method(x)</code></pre>';
window.changeOwnedShadowText=()=>{root.querySelector('code').textContent='omega_method(x)';};
</script></body>"""
CAP_HTML = {
    "node": "<!doctype html><title>Node cap</title><body>" + "<span></span>" * 20010 + "</body>",
    "traversal": "<!doctype html><title>Traversal cap</title><style>div{display:contents}</style><body>"
    + "<div>" * 130
    + "</div>" * 130
    + "</body>",
    "ancestry": """<!doctype html><title>Ancestry cap</title>
<style>div{display:contents}</style><body><script>
let parent=document.body;for(let i=0;i<127;i++){const next=document.createElement('div');parent.append(next);parent=next;}
parent.append(document.createTextNode('ANCESTRY_LIMIT'));
</script></body>""",
}
STRUCTURE = """(() => {
 const body=document.body;if(!body)return {body:false};
 const stack=[{node:body,depth:0}],texts=[];let maxDepth=0;
 while(stack.length){const {node,depth}=stack.pop();
  if(node.nodeType===Node.ELEMENT_NODE)maxDepth=Math.max(maxDepth,depth);
  if(node.nodeType===Node.TEXT_NODE&&node.textContent==='ANCESTRY_LIMIT'){
   const tags=[];for(let p=node.parentElement;p;p=p.parentElement)tags.push(p.tagName);
   const range=document.createRange();range.selectNodeContents(node);const r=range.getBoundingClientRect();
   texts.push({depth,tags,visible:r.width>0&&r.height>0&&r.bottom>0&&r.top<innerHeight&&r.right>0&&r.left<innerWidth});
  }
  for(let i=node.childNodes.length-1;i>=0;i--)stack.push({node:node.childNodes[i],depth:depth+1});
 }
 return {body:true,spans:body.querySelectorAll('span').length,divs:body.querySelectorAll('div').length,maxDepth,texts};
})()"""


def _memory_admission():
    """No swap/resident credits; skip if a fresh local resource check is unavailable."""
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        pytest.skip("Owned CPU Chromium resource admission requires Linux MemAvailable")
    memory = {}
    for line in meminfo.read_text().splitlines():
        key, value = line.split(":", 1)
        if key in {"MemTotal", "MemAvailable"}:
            amount, unit = value.split()
            assert unit == "kB"
            memory[key] = int(amount) * 1024
    reserve = max(24 * 1024**3, (memory["MemTotal"] + 4) // 5)
    if memory["MemAvailable"] < reserve + 2 * 1024**3:
        pytest.skip("Insufficient MemAvailable for host reserve plus 2GiB owned CPU Chromium")


@pytest.fixture(scope="module")
def composed_runtime(tmp_path_factory):
    configured = os.environ.get("JEV_UPSTREAM_PATH")
    if not configured:
        pytest.skip("Set JEV_UPSTREAM_PATH to the existing pinned checkout; never fetches")
    _memory_admission()
    playwright = pytest.importorskip("playwright.sync_api")
    pytest.importorskip("httpx")
    spec = importlib.util.spec_from_file_location(
        "observation_profile_browser_harness", ROOT / "scripts/jev_browser_benchmark.py"
    )
    harness = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = harness
    spec.loader.exec_module(harness)
    canonical = Path(configured).resolve()
    manifest = harness.upstream_manifest(canonical)
    assert manifest["source_sha256"] == dict(profile.CANONICAL_SHA256)
    helper_sha = hashlib.sha256(Path(profile.__file__).read_bytes()).hexdigest()
    materialized = profile.materialize_profile(
        canonical,
        tmp_path_factory.mktemp("composed-profile") / "sources",
        expected_source_sha256=helper_sha,
        profile=profile.COMPOSED_PROFILE,
    )
    runtime_root = Path(materialized["runtime_root"])
    _memory_admission()
    with playwright.sync_playwright() as runtime:
        executable = os.environ.get("JEV_CHROMIUM_PATH")
        if executable and not Path(executable).is_file():
            pytest.skip("JEV_CHROMIUM_PATH must name an existing local executable")
        options = {"executable_path": executable} if executable else {}
        try:
            chromium = runtime.chromium.launch(
                headless=True,
                args=[
                    "--disable-gpu",
                    "--disable-software-rasterizer",
                    "--disable-dev-shm-usage",
                    "--disable-background-networking",
                ],
                **options,
            )
        except playwright.Error as exc:
            if "Executable doesn't exist" in str(exc):
                pytest.skip("No installed Playwright Chromium; never downloads")
            raise
        try:
            context = chromium.new_context(
                viewport={"width": 1120, "height": 780}, service_workers="block"
            )
            blocked = []

            def refuse(route):
                blocked.append((route.request.method, route.request.url))
                route.abort("blockedbyclient")

            context.route("**/*", refuse)

            class Transport(harness.PlaywrightCDP):
                def __init__(self):
                    super().__init__(context)
                    self.evaluations = []

                def cdp(self, method, session_id=None, **params):
                    assert not method.startswith("Input."), (
                        "Synthetic freshness tests never send input"
                    )
                    if method == "Page.navigate":
                        assert params["url"] == "about:blank"
                    response = super().cdp(method, session_id=session_id, **params)
                    if method == "Runtime.evaluate":
                        self.evaluations.append((params["expression"], copy.deepcopy(response)))
                    return response

            transport = Transport()
            with harness.load_upstream(runtime_root, transport) as (agent, model):
                native = sys.modules["jev_ultrafast.browser"]
                try:
                    assert native.READ_STATE == (runtime_root / profile.SNAPSHOT).read_text()
                    assert (
                        native.MARKER
                        == f"(() => {{ const state={native.READ_STATE}; return state?.marker ?? null; }})()"
                    )
                    assert (
                        Path(native.__file__).resolve() == runtime_root / "jev_ultrafast/browser.py"
                    )
                    for label, digest in profile.CANONICAL_SHA256.items():
                        if label != profile.SNAPSHOT:
                            assert (
                                hashlib.sha256((runtime_root / label).read_bytes()).hexdigest()
                                == digest
                            )
                    yield native, agent, model, transport, canonical
                finally:
                    model.CLIENT.close()
        finally:
            try:
                if "context" in locals():
                    context.close()
                    assert not blocked, "An owned fixture attempted a blocked request"
            finally:
                chromium.close()


@pytest.fixture
def owned_page(composed_runtime):
    _memory_admission()
    native, _, _, transport, _ = composed_runtime
    browser = native.Browser("about:blank")
    try:
        yield browser, transport.targets[browser.target]
    finally:
        browser.close()


def test_composed_slots_visibility_range_and_native_controls(composed_runtime, owned_page):
    browser, page = owned_page
    page.set_content(CONTROLS.read_text(), wait_until="load")
    observed = browser.observe(screenshot=False)
    canonical = browser.evaluate((composed_runtime[4] / profile.SNAPSHOT).read_text())
    for key in set(canonical) - {"text", "marker"}:
        assert canonical[key] == observed[key]
    for text in VISIBLE:
        assert text in observed["text"]
    for text in HIDDEN:
        assert text not in observed["text"]
    assert (
        observed["text"].count("DIRECT_SLOT_VISIBLE")
        == observed["text"].count("ASSIGNED_VISIBLE")
        == 1
    )
    assert (
        observed["text"].index("SHADOW_BEFORE")
        < observed["text"].index("ASSIGNED_VISIBLE")
        < observed["text"].index("SHADOW_AFTER")
    )
    geometry = browser.evaluate(
        "(() => {const r=document.createRange();r.selectNodeContents(document.querySelector('#range').firstChild);return r.getBoundingClientRect().toJSON()})()"
    )
    assert geometry["width"] > 0 and geometry["height"] > 0


def test_composed_text_keeps_6000_character_bound(owned_page):
    browser, page = owned_page
    page.set_content(
        '<!doctype html><body><p style="white-space:nowrap">' + "A" * 6100 + "</p></body>"
    )
    observed = browser.observe(screenshot=False)
    assert observed["text"] == "A" * 6000
    assert not observed["observation_profile"]["node_limit_reached"]
    assert not observed["observation_profile"]["depth_limit_reached"]


@pytest.mark.parametrize("kind", ["node", "traversal", "ancestry"])
def test_caps_reject_read_state_and_native_marker_freshness(composed_runtime, owned_page, kind):
    native, _, _, transport, _ = composed_runtime
    browser, page = owned_page
    page.set_content("<!doctype html><body><p>Before cap</p></body>")
    baseline = browser.observe(screenshot=False)
    page.set_content(CAP_HTML[kind], wait_until="load")
    proof = browser.evaluate(STRUCTURE)
    assert proof["body"]
    if kind == "node":
        assert proof["spans"] == 20010
    elif kind == "traversal":
        assert proof["divs"] == proof["maxDepth"] == 130
    else:
        assert proof["divs"] == proof["maxDepth"] == 127
        assert proof["texts"] == [
            {"depth": 128, "tags": ["DIV"] * 127 + ["BODY", "HTML"], "visible": True}
        ]
    first = len(transport.evaluations)
    with pytest.raises(native.StalePage):
        browser.evaluate(native.READ_STATE)
    with pytest.raises(native.StalePage):
        browser.fresh(baseline)
    errors = transport.evaluations[first:]
    assert [expression for expression, _ in errors] == [native.READ_STATE, native.MARKER]
    for _, response in errors:
        description = response["exceptionDetails"]["exception"]["description"]
        assert "ComposedVisibleTextLimitError" in description
        metadata = json.loads(
            re.search(r"COMPOSED_VISIBLE_TEXT_LIMIT (\{[^\n]*\})", description).group(1)
        )
        assert metadata["node_limit_reached" if kind == "node" else "depth_limit_reached"] is True
    # CLICK/SELECT use the unchanged native page_key+guard branch, not MARKER.
    # This is not a new unconditional cap check between a model reply and input.


def test_native_predict_and_done_reject_changed_shadow_text(composed_runtime, monkeypatch):
    _memory_admission()
    httpx = pytest.importorskip("httpx")
    native, agent_module, model, transport, _ = composed_runtime
    calls = []

    def fake_policy(request):
        assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
        body = json.loads(request.content)
        calls.append(body)
        answers = {}
        for name, question in body["questions"].items():
            ids = list(question["criteria"])
            choice = "DONE" if name == "operation" else ids[0]
            assert choice in ids
            answers[name] = {
                "choice": choice,
                "confidence": 1.0,
                "probabilities": {key: float(key == choice) for key in ids},
            }
        return httpx.Response(200, json={"model": "owned-fake-DONE-transport", "answers": answers})

    with (
        httpx.Client(transport=httpx.MockTransport(fake_policy)) as client,
        monkeypatch.context() as patch,
    ):
        patch.setattr(model, "CLIENT", client)
        patch.setenv("TYPESAFE_API_KEY", "owned-local-fake-key")
        agent = agent_module.Agent("about:blank", "Owned freshness regression", screenshots=False)
        try:
            transport.targets[agent.browser.target].set_content(SHADOW_DONE, wait_until="load")
            predicted = agent.command("predict")
            before = copy.deepcopy(predicted["page"])
            assert predicted["decision"]["choice"] == "DONE" and "alpha_method(x)" in before["text"]
            agent.browser.evaluate("window.changeOwnedShadowText()")
            assert agent.browser.fresh(before) is False
            with pytest.raises(native.StalePage):
                agent.command("act", {"fingerprint": before["fingerprint"]})
            rejected = copy.deepcopy(agent.snapshot())
            assert (
                rejected["status"] == "ready"
                and rejected["decision"] is None
                and rejected["history"] == []
            )
            assert len(rejected["decisions"]) == 1
            after = agent.browser.observe(screenshot=False)
            for key in ("actions", "page_key", "guards"):
                assert before[key] == after[key]
            assert before["marker"] != after["marker"] and "omega_method(x)" in after["text"]
            refreshed = agent.command("predict")
            completed = agent.command("act", {"fingerprint": refreshed["page"]["fingerprint"]})
            assert completed["status"] == "done" and completed["history"] == []
            assert len(rejected["decisions"]) == 1 and len(completed["decisions"]) == 2
            assert len(calls) == 2  # Fake HTTP transport calls; zero real model/helper calls.
        finally:
            agent.close()
