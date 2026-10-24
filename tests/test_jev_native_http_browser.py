"""Opt-in wiring check against an ALREADY RUNNING genuine native HTTP service.

No model or API service is constructed here. Source references for this test:
  scripts/jev_browser_benchmark.py: FixtureCase, fixture_server, PlaywrightCDP,
      upstream_manifest, load_upstream, model_adapter.
  jev-ultrafast@1231850a0bf1a0c0341fe408ef1668dbbfdfac46:
      model.choose/post_json, Agent.command('tick'), Browser.act, snapshot.js.
The unmodified upstream path is the standard eight-operation protocol; only
currently available operations are offered. No experimental profile is loaded.

Future execution requires JEV_NATIVE_HTTP_BROWSER_TESTS=1 and explicit
JEV_NATIVE_HTTP_ENDPOINT, JEV_NATIVE_HTTP_MODEL, JEV_UPSTREAM_PATH,
JEV_CHROMIUM_EXECUTABLE, JEV_NATIVE_HTTP_BROWSER_OUTPUT (a NEW directory).
JEV_NATIVE_HTTP_API_KEY is optional and never recorded. This test does not start
the external service or independently authenticate its weights: join the saved
HTTP body hashes/request IDs with that service's native execution receipts.
This invented two-control page is an integration diagnostic, not a benchmark.

Implementation was written from source only while full06 was running; no test,
import, browser, model, formatter, or resource observation was executed then.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
STANDARD_OPERATIONS = {
    "CLICK",
    "TYPE_TEXT",
    "SELECT",
    "SCROLL_DOWN",
    "SCROLL_UP",
    "WAIT",
    "DONE",
    "BLOCKED",
}
FIXTURE = """<!doctype html><html><head><meta charset="utf-8">
<title>Two lamp controls</title><style>
body{font:20px sans-serif;margin:32px}button{padding:16px;margin:12px;font:inherit}
</style></head><body><h1>Two lamp controls</h1>
<p id="confirmation" role="status" data-active="none">Neither lamp armed</p>
<button id="copper" type="button">Arm copper lamp</button>
<button id="silver" type="button">Arm silver lamp</button>
<script>
window.lampEvidence={clicks:[],lastTrusted:false};
for(const button of document.querySelectorAll('button')){
  button.addEventListener('click',event=>{
    window.lampEvidence.clicks.push({id:button.id,trusted:event.isTrusted});
    window.lampEvidence.lastTrusted=event.isTrusted;
    const status=document.getElementById('confirmation');
    status.dataset.active=button.id;
    status.textContent=button.id==='copper'?'Copper lamp armed':'Silver lamp armed';
    for(const other of document.querySelectorAll('button'))other.disabled=other===button;
  });
}
</script></body></html>"""
JUDGE = """(() => {
  const status=document.getElementById('confirmation');
  const visible=element=>{
    if(!element?.isConnected)return false;
    const rect=element.getBoundingClientRect();
    const x=rect.x+rect.width/2,y=rect.y+rect.height/2;
    return rect.width>0&&rect.height>0&&x>=0&&y>=0&&x<innerWidth&&y<innerHeight&&
      element.checkVisibility({checkOpacity:true,checkVisibilityCSS:true})&&
      element.contains(document.elementFromPoint(x,y));
  };
  return {url:location.href,active:status?.dataset.active,text:status?.textContent,
    visible:visible(status),clicks:window.lampEvidence?.clicks,
    last_trusted:window.lampEvidence?.lastTrusted,
    buttons:[...document.querySelectorAll('button')].map(button=>({
      id:button.id,text:button.textContent,disabled:button.disabled,visible:visible(button)
    }))};
})()"""


@pytest.mark.integration
def test_real_native_http_choose_agent_and_guarded_click():
    if os.environ.get("JEV_NATIVE_HTTP_BROWSER_TESTS") != "1":
        pytest.skip("Explicit external-native HTTP browser opt-in required")

    required = (
        "JEV_NATIVE_HTTP_ENDPOINT",
        "JEV_NATIVE_HTTP_MODEL",
        "JEV_UPSTREAM_PATH",
        "JEV_CHROMIUM_EXECUTABLE",
        "JEV_NATIVE_HTTP_BROWSER_OUTPUT",
    )
    missing = [name for name in required if not os.environ.get(name, "").strip()]
    assert not missing, "Missing explicit configuration: " + ", ".join(missing)
    endpoint = os.environ["JEV_NATIVE_HTTP_ENDPOINT"]
    address = urlsplit(endpoint)
    assert address.scheme == "http" and address.hostname in {"127.0.0.1", "localhost", "::1"}
    assert address.path == "/v1/systemone" and not address.query and not address.fragment
    assert address.username is None and address.password is None
    model = os.environ["JEV_NATIVE_HTTP_MODEL"]
    upstream = Path(os.environ["JEV_UPSTREAM_PATH"]).resolve(strict=True)
    executable = Path(os.environ["JEV_CHROMIUM_EXECUTABLE"]).resolve(strict=True)
    output = Path(os.environ["JEV_NATIVE_HTTP_BROWSER_OUTPUT"]).absolute()
    key = os.environ.get("JEV_NATIVE_HTTP_API_KEY", "")
    playwright = pytest.importorskip("playwright.sync_api")
    pytest.importorskip("httpx")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    (output / ".gitignore").write_text("*\n", encoding="utf-8")

    module_name = "_jev_native_http_browser_test_harness"
    assert module_name not in sys.modules
    source = ROOT / "scripts/jev_browser_benchmark.py"
    source_before = hashlib.sha256(source.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location(module_name, source)
    assert spec is not None and spec.loader is not None
    harness = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = harness
    report = {
        "scope": "invented two-control real-native HTTP wiring diagnostic; not a model score",
        "endpoint": endpoint,
        "requested_model": model,
        "max_ticks": 16,
        "action_profile": "standard",
        "observation_profile": "native-1231850",
        "client_source_sha256": source_before,
        "test_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(FIXTURE.encode()).hexdigest(),
        "native_authenticity": "external operator/server receipts required; no local model loaded",
        "snapshots": [],
        "observations": [],
        "http": [],
        "records": [],
        "secondary_errors": [],
        "error": None,
        "goal_satisfied": False,
        "success": False,
    }
    agent = chromium = None
    primary = None
    secrets = [key] if key else []
    try:
        spec.loader.exec_module(harness)
        report["upstream_before"] = harness.upstream_manifest(upstream)
        assert report["upstream_before"]["revision"] == harness.UPSTREAM_REVISION
        (output / "fixture.html").write_text(FIXTURE, encoding="utf-8")
        case = harness.FixtureCase(
            case_id="native-http-two-controls",
            task_id="native_http_wiring",
            goal="Arm the copper lamp. Stop when Copper lamp armed is visibly confirmed.",
            html=FIXTURE,
            expected={},
        )
        report["goal"] = case.goal
        with playwright.sync_playwright() as runtime, harness.fixture_server([case]) as base:
            chromium = runtime.chromium.launch(headless=True, executable_path=str(executable))
            try:
                transport = harness.PlaywrightCDP(chromium)
                transport.observation_expression = (
                    upstream / "jev_ultrafast/snapshot.js"
                ).read_text()
                transport.observation_sink = lambda row: report["observations"].append(
                    copy.deepcopy(row)
                )
                with harness.load_upstream(upstream, transport) as (agent_module, model_module):
                    original_http_post = model_module.CLIENT.post

                    def observed_http_post(url, **kwargs):
                        # Pass through exactly once to the real HTTP client. No answer replacement.
                        row = {"endpoint": str(url), "started_unix": time.time()}
                        report["http"].append(row)
                        started = time.perf_counter()
                        try:
                            response = original_http_post(url, **kwargs)
                        except BaseException as exc:
                            row["error"] = f"{type(exc).__name__}: {exc}"
                            raise
                        else:
                            try:
                                request_raw = response.request.content
                                response_raw = response.content
                                row.update(
                                    status=response.status_code,
                                    request_sha256=hashlib.sha256(request_raw).hexdigest(),
                                    response_sha256=hashlib.sha256(response_raw).hexdigest(),
                                    request_utf8=request_raw.decode("utf-8"),
                                    response_utf8=response_raw.decode("utf-8"),
                                    server_request_id=response.headers.get("X-Request-ID"),
                                )
                            except Exception as exc:
                                report["secondary_errors"].append(
                                    f"HTTP observation: {type(exc).__name__}: {exc}"
                                )
                            return response
                        finally:
                            row["elapsed_ms"] = (time.perf_counter() - started) * 1000

                    model_module.CLIENT.post = observed_http_post
                    try:
                        with harness.model_adapter(
                            agent_module, model_module, endpoint=endpoint, model=model, api_key=key
                        ) as (records, adapter_secrets):
                            report["records"] = records
                            secrets.extend(adapter_secrets)
                            try:
                                agent = agent_module.Agent(
                                    base + case.path, case.goal, screenshots=False
                                )
                                report["snapshots"].append(copy.deepcopy(agent.snapshot()))
                                initial = agent.state["browser"].evaluate(JUDGE)
                                report["initial_judge"] = initial
                                assert initial["active"] == "none" and not initial["clicks"]
                                assert len(initial["buttons"]) == 2
                                assert all(
                                    button["visible"] is True and button["disabled"] is False
                                    for button in initial["buttons"]
                                )
                                assert agent.state["page"]["omitted_actions"] == 0
                                for _ in range(16):
                                    if agent.state["status"] in {"done", "blocked"}:
                                        break
                                    report["snapshots"].append(copy.deepcopy(agent.command("tick")))
                            finally:
                                if agent is not None:
                                    try:
                                        final = agent.state["browser"].evaluate(JUDGE)
                                        report["final_judge"] = final
                                        report["goal_satisfied"] = (
                                            final["url"] == base + case.path
                                            and final["active"] == "copper"
                                            and final["text"] == "Copper lamp armed"
                                            and final["visible"] is True
                                            and final["last_trusted"] is True
                                            and final["clicks"]
                                            == [{"id": "copper", "trusted": True}]
                                        )
                                        report["final_html"] = agent.state["browser"].evaluate(
                                            "document.documentElement.outerHTML"
                                        )
                                        report["final_snapshot"] = copy.deepcopy(agent.snapshot())
                                    except Exception as exc:
                                        report["secondary_errors"].append(
                                            f"Final observation: {type(exc).__name__}: {exc}"
                                        )
                                    try:
                                        agent.close()
                                    except Exception as exc:
                                        report["secondary_errors"].append(
                                            f"Agent close: {type(exc).__name__}: {exc}"
                                        )
                    finally:
                        model_module.CLIENT.post = original_http_post
            finally:
                try:
                    chromium.close()
                except Exception as exc:
                    report["secondary_errors"].append(
                        f"Chromium close: {type(exc).__name__}: {exc}"
                    )
                finally:
                    chromium = None
        policy = [row for row in report["records"] if row["kind"] == "policy"]
        assert policy and report["http"], "No real native HTTP request was made"
        first = policy[0]["request"]
        assert len(first["questions"]["click_target"]["criteria"]) == 2
        for row in policy:
            assert set(row["request"]["questions"]["operation"]["criteria"]) <= STANDARD_OPERATIONS
            if "response" in row:
                assert row["response"]["model"] == model
        assert all(row["kind"] == "policy" for row in report["records"])
    except BaseException as exc:
        primary = exc
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if hasattr(harness, "upstream_manifest"):
                report["upstream_after"] = harness.upstream_manifest(upstream)
            report["source_unchanged"] = hashlib.sha256(
                source.read_bytes()
            ).hexdigest() == source_before and report.get("upstream_before") == report.get(
                "upstream_after"
            )
        except Exception as exc:
            report["secondary_errors"].append(f"Source verification: {type(exc).__name__}: {exc}")
            report["source_unchanged"] = False
        final_status = report.get("final_snapshot", {}).get("status")
        report["final_status"] = final_status
        report["success"] = (
            report["goal_satisfied"]
            and final_status == "done"
            and report["error"] is None
            and not report["secondary_errors"]
            and report["source_unchanged"]
        )
        try:
            # Redact only evidence copies; the real HTTP body and choices were never changed.
            safe = harness._redact(report, secrets) if hasattr(harness, "_redact") else report
            (output / "report.json").write_text(
                json.dumps(safe, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
            )
        except Exception as exc:
            if primary is None:
                raise
            primary.add_note(f"Evidence write also failed: {type(exc).__name__}: {exc}")
        finally:
            sys.modules.pop(module_name, None)

    assert report["success"], (
        f"Native HTTP browser diagnostic failed; evidence: {output / 'report.json'}"
    )
