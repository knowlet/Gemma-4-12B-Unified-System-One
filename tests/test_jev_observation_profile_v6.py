"""V6 source closure and real local Chromium navigation regressions.

Only observation receives a per-call settling target. These tests use genuine
streaming loopback documents and native mouse input; no policy, text helper,
public website, Modal, dependency download, or model is used.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import jev_browser_profile_v6_benchmark as launcher
import jev_observation_profile_v4 as parent
import jev_observation_profile_v6 as profile
import pytest

from s1 import jev_observation_profile as grandparent

ROOT = Path(__file__).resolve().parents[1]
PROFILE = "composed-visible-controls-settled-v6"
PARENT_PROFILE = "composed-visible-controls-v4"
PARENT_SHA = "f96fb0553c4c8ef1a163633577bfb3ddbb03ddbc1ff6644c36d8fbe2c06d4b43"
GRANDPARENT_SHA = "8d199ac1564af05892e9b21e225d1fdf500f1953840855c902160ed8d56df89d"
BENCHMARK_SHA = "fe1410675e4decb06a8491a5a83d2ec965629c99f8912b797446c4cb2c155f6c"
EVENTS_SHA = "ee6d3ea04d948fd8947d12934b63564494aab6a67cb8d8063ea7e12fb84a0664"
HELPER_SHA = hashlib.sha256(Path(profile.__file__).read_bytes()).hexdigest()
CHROMIUM = "/home/albert_hsieh/.cache/ms-playwright/chromium_headless_shell-1208/chrome-linux/headless_shell"
BROWSER = "jev_ultrafast/browser.py"
SNAPSHOT = "jev_ultrafast/snapshot.js"
DOCUMENT = """(() => ({url:location.href,ready:document.readyState,
body:!!document.body,text:document.body?.innerText??null,
width:innerWidth,height:innerHeight}))()"""


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


@pytest.fixture(scope="module")
def canonical_root():
    root = Path(os.environ.get("JEV_UPSTREAM_PATH", ROOT / "artifacts/jev-runtime/upstream"))
    if not root.is_dir():
        pytest.skip("An installed pinned Jev checkout is required; never fetches")
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


def without_observe(blob):
    """Compare all other source bytes, independently locating the one method."""
    lines = blob.splitlines(keepends=True)
    tree = ast.parse(blob)
    browser = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Browser")
    method = next(n for n in browser.body if isinstance(n, ast.FunctionDef) and n.name == "observe")
    lines[method.lineno - 1 : method.end_lineno] = [b"    # OBSERVE METHOD\n"]
    return b"".join(lines)


def test_only_v4_observe_and_one_snapshot_identity_literal_change(canonical):
    assert sha(Path(parent.__file__).read_bytes()) == PARENT_SHA
    assert sha(Path(grandparent.__file__).read_bytes()) == GRANDPARENT_SHA
    assert sha((ROOT / "scripts/jev_browser_benchmark.py").read_bytes()) == BENCHMARK_SHA
    old = parent.build_profile(canonical, expected_source_sha256=PARENT_SHA, profile=PARENT_PROFILE)
    new = build(canonical)
    assert new.manifest()["profile"] == PROFILE
    assert set(new.canonical_files) == set(new.runtime_files) == set(canonical)
    assert dict(new.canonical_files) == canonical
    assert without_observe(new.runtime_files[BROWSER]) == without_observe(
        old.runtime_files[BROWSER]
    )
    before = ("id:'" + PARENT_PROFILE + "'").encode()
    after = ("id:'" + PROFILE + "'").encode()
    assert new.runtime_files[SNAPSHOT].count(after) == 1
    assert new.runtime_files[SNAPSHOT].replace(after, before, 1) == old.runtime_files[SNAPSHOT]
    for name in canonical:
        if name not in {BROWSER, SNAPSHOT}:
            assert new.runtime_files[name] == old.runtime_files[name] == canonical[name]
    # This includes Agent history, decision budgets, all request heads and model HTTP.
    assert {k for k in canonical if new.runtime_files[k] != old.runtime_files[k]} == {
        BROWSER,
        SNAPSHOT,
    }


@pytest.mark.parametrize("alias", [PARENT_PROFILE, "v6", PROFILE + " "])
def test_profile_requires_exact_explicit_v6_identity(canonical, alias):
    with pytest.raises(ValueError):
        profile.build_profile(canonical, expected_source_sha256=HELPER_SHA, profile=alias)


@pytest.mark.parametrize("name", tuple(grandparent.CANONICAL_SHA256))
def test_every_canonical_source_still_authenticated(canonical, name):
    canonical[name] += b"\n"
    with pytest.raises(ValueError):
        build(canonical)


def test_materialization_contains_complete_three_helper_chain(tmp_path, canonical_root):
    target = tmp_path / "profile"
    result = materialize(canonical_root, target)
    manifest_blob = (target / "profile-manifest.json").read_bytes()
    handle = verify(target, result["manifest_sha256"])
    assert result["manifest_sha256"] == sha(manifest_blob)
    assert handle.runtime_root == target / "runtime"
    assert handle.read_state == (handle.runtime_root / SNAPSHOT).read_text()
    assert handle.marker_expression == (
        f"(() => {{ const state={handle.read_state}; return state?.marker ?? null; }})()"
    )
    manifest = handle.manifest()
    assert manifest["profile"] == PROFILE
    assert manifest["helper_source_sha256"] == HELPER_SHA
    assert manifest["canonical_source_sha256"] == dict(grandparent.CANONICAL_SHA256)
    for name, record in manifest["artifact_files"].items():
        blob = (target / name).read_bytes()
        assert sha(blob) == record["sha256"] and len(blob) == record["size_bytes"]
    assert {p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()} == {
        *manifest["artifact_files"],
        "profile-manifest.json",
    }
    assert (target / "sources/jev_observation_profile_v6.py").read_bytes() == Path(
        profile.__file__
    ).read_bytes()
    assert (target / "sources/jev_observation_profile_v4.py").read_bytes() == Path(
        parent.__file__
    ).read_bytes()
    assert (target / "sources/jev_observation_profile_v3_parent.py").read_bytes() == Path(
        grandparent.__file__
    ).read_bytes()
    manifest["runtime_source_sha256"][BROWSER] = "0" * 64
    assert handle.manifest()["runtime_source_sha256"][BROWSER] != "0" * 64
    with pytest.raises(ValueError):
        materialize(canonical_root, target)


def replace_digest(value, before, after):
    if isinstance(value, dict):
        return {k: replace_digest(v, before, after) for k, v in value.items()}
    if isinstance(value, list):
        return [replace_digest(v, before, after) for v in value]
    return after if value == before else value


@pytest.mark.parametrize(
    "label",
    [
        "runtime/" + BROWSER,
        "runtime/" + SNAPSHOT,
        "sources/jev_observation_profile_v4.py",
        "sources/jev_observation_profile_v3_parent.py",
    ],
)
def test_rehashing_runtime_or_parent_copy_never_authorizes_new_source(
    tmp_path, canonical_root, label
):
    target = tmp_path / "profile"
    materialize(canonical_root, target)
    path = target / "profile-manifest.json"
    manifest = json.loads(path.read_bytes())
    changed = (target / label).read_bytes() + b"\n# unauthorized replacement\n"
    old_sha = manifest["artifact_files"][label]["sha256"]
    (target / label).write_bytes(changed)
    manifest = replace_digest(manifest, old_sha, sha(changed))
    manifest["artifact_files"][label]["size_bytes"] = len(changed)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    )
    with pytest.raises(ValueError):
        verify(target, sha(path.read_bytes()))


@pytest.mark.parametrize("mutation", ["extra_file", "extra_directory", "symlink"])
def test_complete_profile_population_is_closed(tmp_path, canonical_root, mutation):
    target = tmp_path / "profile"
    result = materialize(canonical_root, target)
    if mutation == "extra_file":
        (target / "unreviewed.py").write_text("raise RuntimeError('unreviewed')\n")
    elif mutation == "extra_directory":
        (target / "unreviewed").mkdir()
    else:
        path = target / "runtime" / SNAPSHOT
        other = tmp_path / "snapshot.js"
        other.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(other)
    with pytest.raises(ValueError):
        verify(target, result["manifest_sha256"])


@pytest.mark.parametrize("case_module", ["original", "heldout"])
def test_runner_and_tracker_inverse_bytes_and_ast(tmp_path, case_module):
    bundle, manifest = launcher.prepare_bundle(tmp_path / "bundle", case_module=case_module)
    original = (ROOT / "scripts/jev_browser_benchmark.py").read_bytes()
    derived = (bundle / "runner.py").read_bytes()
    assert (bundle / "original_jev_browser_benchmark.py").read_bytes() == original
    assert manifest["profile"] == PROFILE and manifest["case005_interface_exposed"] is True
    for name, source in (
        ("jev_observation_profile_v6.py", profile.__file__),
        ("jev_observation_profile_v4.py", parent.__file__),
        ("jev_observation_profile_v3_parent.py", grandparent.__file__),
    ):
        assert (bundle / name).read_bytes() == Path(source).read_bytes()
    substitutions = [
        (
            b"import jev_observation_profile_v6 as profile",
            b"from s1 import jev_observation_profile as profile",
            3,
        ),
        (
            (f'PRUNED_OBSERVATION_PROFILE = "{PROFILE}"').encode(),
            b'PRUNED_OBSERVATION_PROFILE = "composed-visible-text-v3"',
            1,
        ),
        (
            b"from jev_observation_events_v6 import ObservationProfileTracker",
            b"from s1.jev_observation_events import ObservationProfileTracker",
            1,
        ),
    ]
    if case_module == "heldout":
        substitutions.append(
            (
                b"        from jev_heldout_public_cases import (\n",
                b"        from jev_public_web_cases import (\n",
                1,
            )
        )
    restored = derived
    for before, after, count in substitutions:
        assert restored.count(before) == count
        restored = restored.replace(before, after)
    assert restored == original
    assert ast.dump(ast.parse(restored), include_attributes=False) == ast.dump(
        ast.parse(original), include_attributes=False
    )
    events = (ROOT / "src/s1/jev_observation_events.py").read_bytes()
    assert sha(events) == EVENTS_SHA
    copied = (bundle / "jev_observation_events_v6.py").read_bytes()
    after = (f'PRUNED_PROFILE = "{PROFILE}"').encode()
    before = b'PRUNED_PROFILE = "composed-visible-text-v3"'
    assert copied.count(after) == 1 and copied.replace(after, before, 1) == events
    launcher.verify_bundle(bundle, manifest)


def test_bundle_verifies_profile_without_reading_repository_helpers(tmp_path, canonical_root):
    bundle, manifest = launcher.prepare_bundle(tmp_path / "bundle")
    result = materialize(canonical_root, tmp_path / "profile")
    # An isolated fresh interpreter cannot reuse already-imported repository parents.
    # Audit rejects even an attempted source open beneath src/ or scripts/.
    code = r"""
import json, pathlib, sys
bundle, target, digest, helper_sha, root = sys.argv[1:]
root = pathlib.Path(root)
def audit(event, arguments):
    if event == 'open' and isinstance(arguments[0], (str, bytes)):
        name = arguments[0].decode() if isinstance(arguments[0], bytes) else arguments[0]
        path = pathlib.Path(name).resolve()
        if path.is_relative_to(root / 'scripts') or path.is_relative_to(root / 'src'):
            raise AssertionError('repository helper access: ' + str(path))
sys.addaudithook(audit)
sys.path.insert(0, bundle)
import jev_observation_profile_v6 as p
h = p.verify_materialized_profile(target, expected_manifest_sha256=digest,
    expected_source_sha256=helper_sha, profile=p.PROFILE)
loaded = sorted(str(pathlib.Path(m.__file__).resolve()) for m in sys.modules.values()
    if getattr(m, '__file__', None) and 'jev_observation_profile' in str(m.__file__))
assert len(loaded) == 3, loaded
assert all(pathlib.Path(f).parent == pathlib.Path(bundle) for f in loaded), loaded
print(json.dumps({'profile': h.manifest()['profile'], 'loaded': loaded}))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            code,
            str(bundle),
            str(tmp_path / "profile"),
            result["manifest_sha256"],
            HELPER_SHA,
            str(ROOT),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    assert json.loads(completed.stdout)["profile"] == PROFILE
    launcher.verify_bundle(bundle, manifest)


def memory_admission():
    memory = {}
    if not Path("/proc/meminfo").is_file():
        pytest.skip("A fresh Linux memory admission check is required")
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
    memory_admission()
    playwright = pytest.importorskip("playwright.sync_api")
    pytest.importorskip("httpx")
    executable = Path(os.environ.get("JEV_CHROMIUM_PATH", CHROMIUM))
    if not executable.is_file():
        pytest.skip("Only an installed Chromium is allowed; never downloads")
    bundle, manifest = launcher.prepare_bundle(tmp_path_factory.mktemp("v6-runner") / "bundle")
    name = "jev_v6_owned_fixture_runner"
    spec = importlib.util.spec_from_file_location(name, bundle / "runner.py")
    harness = importlib.util.module_from_spec(spec)
    sys.modules[name] = harness
    before_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(harness)
        output = tmp_path_factory.mktemp("v6-materialized") / "profile"
        result = materialize(canonical_root, output)
        verified = verify(output, result["manifest_sha256"])
        assert harness.upstream_manifest(canonical_root)["source_sha256"] == dict(
            grandparent.CANONICAL_SHA256
        )
        with playwright.sync_playwright() as runtime:
            chromium = runtime.chromium.launch(
                executable_path=str(executable),
                headless=True,
                args=[
                    "--disable-gpu",
                    "--disable-software-rasterizer",
                    "--disable-dev-shm-usage",
                    "--disable-background-networking",
                ],
            )
            try:
                yield harness, verified, chromium, canonical_root
            finally:
                chromium.close()
                verify(output, result["manifest_sha256"])
                launcher.verify_bundle(bundle, manifest)
    finally:
        sys.dont_write_bytecode = before_bytecode
        sys.modules.pop(name, None)


@contextmanager
def open_browser(runtime, url="about:blank", *, origin=None):
    memory_admission()
    harness, verified, chromium, canonical_root = runtime
    blocked, post_attempts = [], []
    with chromium.new_context(
        viewport={"width": 1120, "height": 780}, service_workers="block"
    ) as context:

        def route_owned(route):
            request = route.request
            parsed = urlsplit(request.url)
            allowed = origin is not None and (
                request.method in {"GET", "HEAD"}
                and parsed.scheme == "http"
                and parsed.hostname == "127.0.0.1"
                and parsed.port == urlsplit(origin).port
                and parsed.username is None
                and parsed.password is None
            )
            if allowed:
                route.continue_()
            else:
                blocked.append((request.method, request.url))
                route.abort("blockedbyclient")

        context.route("**/*", route_owned)

        class Transport(harness.PlaywrightCDP):
            def __init__(self):
                super().__init__(context)
                self.inputs = []
                self.reads = []
                self.delay_snapshot = 0
                self.delay_screenshot = 0
                self.after_input_error = None

            def cdp(self, method, session_id=None, **params):
                read = (
                    method == "Runtime.evaluate" and params.get("expression") == verified.read_state
                )
                if method.startswith("Input."):
                    self.inputs.append((method, copy.deepcopy(params)))
                if (
                    self.after_input_error is not None
                    and method == "Runtime.evaluate"
                    and ("action => new Promise" in params.get("expression", ""))
                ):
                    exc, self.after_input_error = self.after_input_error, None
                    raise exc
                at = time.monotonic()
                try:
                    response = super().cdp(method, session_id=session_id, **params)
                except BaseException as exc:
                    if read:
                        self.reads.append({"error": exc, "at": at})
                    raise
                if read:
                    self.reads.append({"response": copy.deepcopy(response), "at": at})
                    delay, self.delay_snapshot = self.delay_snapshot, 0
                    if delay:
                        time.sleep(delay)
                if method == "Page.captureScreenshot" and self.delay_screenshot:
                    delay, self.delay_screenshot = self.delay_screenshot, 0
                    time.sleep(delay)
                return response

        transport = Transport()
        assert transport.public_navigation_hosts is None
        with harness.load_upstream(canonical_root, transport, observation_runtime=verified) as (
            _,
            model,
        ):
            native = sys.modules["jev_ultrafast.browser"]
            assert Path(native.__file__).resolve() == verified.runtime_root / BROWSER

            def refuse(*_args, **_kwargs):
                post_attempts.append(True)
                raise AssertionError("No model/helper HTTP is permitted")

            model.post_json = refuse
            model.CLIENT.post = refuse
            browser = native.Browser(url)
            try:
                yield browser, transport.targets[browser.target], transport, native
            finally:
                browser.close()
                model.CLIENT.close()
                assert blocked == []
                assert post_attempts == []


def html(body):
    return (
        "<!doctype html><html><head><meta charset=utf-8><title>Owned v6</title></head>"
        "<body style='margin:24px;font:18px sans-serif'>"
        + body
        + "<script>window.events=[];for(const t of ['click','input','change'])"
        "document.addEventListener(t,e=>events.push({type:e.type,id:e.target.id,"
        "trusted:e.isTrusted,checked:e.target.checked??null}));</script></body></html>"
    )


def assert_two_click_inputs(transport):
    assert [method for method, _ in transport.inputs] == ["Input.dispatchMouseEvent"] * 2
    assert [params["type"] for _, params in transport.inputs] == ["mousePressed", "mouseReleased"]


def assert_diagnostic(browser, status):
    diagnostic = browser.observation_settling
    assert diagnostic["target_seconds"] == 2.0
    assert diagnostic["scope"] == "per_observe_call"
    assert diagnostic["status"] == status
    assert diagnostic["synchronous_cdp_hard_deadline"] is False
    assert diagnostic["elapsed_seconds"] >= 0
    assert diagnostic["overshoot_seconds"] == pytest.approx(
        max(0, diagnostic["elapsed_seconds"] - 2.0), abs=1e-6
    )
    return copy.deepcopy(diagnostic)


def test_complete_snapshot_fast_path_one_read_and_no_payload_diagnostic(browser_runtime):
    with open_browser(browser_runtime) as (browser, page, transport, _):
        page.set_content(html("<button id=button>Available control</button>"))
        observed = browser.observe(screenshot=False)
        diagnostic = assert_diagnostic(browser, "success")
        assert len(transport.reads) == diagnostic["attempts"] == 1
        assert diagnostic["navigation_retries"] == 0 and diagnostic["late_snapshot"] is False
        assert diagnostic["elapsed_seconds"] < 0.5
        assert "observation_settling" not in observed
        assert observed["observation_profile"]["id"] == PROFILE
        assert any(a["label"] == "Available control" for a in observed["actions"])
        assert transport.inputs == []


class StreamingServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self):
        super().__init__(("127.0.0.1", 0), StreamingHandler)
        self.cases = {}
        self.events = []
        self.lock = threading.Lock()

    def register(self, delay, *, body_first=False):
        with self.lock:
            case_id = str(len(self.cases) + 1)
            self.cases[case_id] = (delay, body_first)
        origin = f"http://127.0.0.1:{self.server_port}"
        return case_id, origin, origin + f"/{case_id}/start", origin + f"/{case_id}/target"

    def record(self, case_id, phase):
        with self.lock:
            self.events.append((case_id, phase, time.monotonic()))


class StreamingHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        self.respond()

    def do_HEAD(self):
        self.respond(head_only=True)

    def respond(self, head_only=False):
        parts = urlsplit(self.path).path.strip("/").split("/")
        if len(parts) != 2 or parts[0] not in self.server.cases:
            self.send_error(404)
            return
        case_id, stage = parts
        delay, body_first = self.server.cases[case_id]
        self.server.record(case_id, "request_" + stage)
        if stage == "start":
            prefix = html(f"<a href='/{case_id}/target'>Continue to delayed document</a>").encode()
            suffix, delay = b"", 0
        elif stage == "target":
            prefix = (
                "<!doctype html><html><head><meta charset=utf-8><title>Delayed target</title>"
                "<!--" + "p" * 16384 + "-->\n"
            ).encode()
            body = f"</head><body><h1>Arrived {case_id}</h1><p>Owned document</p>".encode()
            if body_first:
                prefix, suffix = prefix + body, b"</body></html>"
            else:
                suffix = body + b"</body></html>"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(prefix) + len(suffix)))
        self.end_headers()
        if head_only:
            return
        try:
            self.wfile.write(prefix)
            self.wfile.flush()
            self.server.record(case_id, "prefix_flushed")
            if delay:
                time.sleep(delay)
            self.wfile.write(suffix)
            self.wfile.flush()
            self.server.record(case_id, "body_flushed")
        except OSError:
            self.server.record(case_id, "owned_client_closed")

    def log_message(self, *_args):
        pass


@pytest.fixture(scope="module")
def streaming_server(browser_runtime):
    server = StreamingServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


def click_offered_link_once(browser, page, transport, target_url):
    observed = browser.observe(screenshot=False)
    links = [a for a in observed["actions"] if a["kind"] == "click"]
    assert len(links) == 1
    trusted = []
    page.on(
        "console",
        lambda message: (
            trusted.append(message.text) if message.text == "owned-v6-trusted-click" else None
        ),
    )
    browser.evaluate(
        "document.querySelector('a').addEventListener('click',e=>{"
        "if(e.isTrusted)console.log('owned-v6-trusted-click')})"
    )
    assert browser.fresh(observed, links[0]) is True
    assert browser.act(links[0], observed) == {"executed": links[0]["id"]}
    assert_two_click_inputs(transport)
    # Fixture commit synchronization is separate from the method being measured.
    deadline = time.monotonic() + 0.1
    current = None
    while time.monotonic() < deadline:
        try:
            current = browser.evaluate(DOCUMENT)
            if current["url"] == target_url:
                break
        except Exception:
            pass
        time.sleep(0.002)
    assert current and current["url"] == target_url
    assert trusted == ["owned-v6-trusted-click"]
    transport.reads.clear()
    return current


@pytest.mark.parametrize("delay", [0.35, 0.8, 1.5])
def test_streaming_navigation_recovers_without_replaying_input(
    browser_runtime, streaming_server, delay
):
    case_id, origin, start, target = streaming_server.register(delay)
    with open_browser(browser_runtime, start, origin=origin) as (browser, page, transport, _):
        before = click_offered_link_once(browser, page, transport, target)
        assert before["ready"] == "loading" and before["body"] is False
        observed = browser.observe(screenshot=False)
        diagnostic = assert_diagnostic(browser, "success")
        assert observed["url"] == target and f"Arrived {case_id}" in observed["text"]
        assert any(
            row["response"].get("result", {}).get("subtype") == "null" for row in transport.reads
        )
        assert diagnostic["navigation_retries"] > 0
        assert diagnostic["attempts"] == len(transport.reads)
        assert diagnostic["elapsed_seconds"] <= 2.0
        assert diagnostic["late_snapshot"] is False
        assert_two_click_inputs(transport)
        final = browser.evaluate(DOCUMENT)
        assert final["width"] == 1120 and final["height"] == 780
        events = [phase for cid, phase, _ in streaming_server.events if cid == case_id]
        assert events.count("request_target") == 1
        assert events.count("body_flushed") == 2  # start and target each complete once


def test_navigation_timeout_is_bounded_and_later_observe_resets_diagnostic(
    browser_runtime, streaming_server
):
    _, origin, start, target = streaming_server.register(3.2)
    with open_browser(browser_runtime, start, origin=origin) as (browser, page, transport, native):
        before = click_offered_link_once(browser, page, transport, target)
        assert before["body"] is False and before["ready"] == "loading"
        at = time.monotonic()
        with pytest.raises(
            native.StalePage, match="^Navigation settling target exceeded"
        ) as caught:
            browser.observe(screenshot=False)
        elapsed = time.monotonic() - at
        diagnostic = assert_diagnostic(browser, "timeout")
        assert 2.0 <= elapsed < 3.0
        assert diagnostic["elapsed_seconds"] == pytest.approx(elapsed, abs=0.05)
        assert diagnostic["navigation_retries"] > 0
        assert diagnostic["late_snapshot"] is False
        assert diagnostic["attempts"] == len(transport.reads)
        assert caught.value.observation_settling == diagnostic
        assert caught.value.observation_settling is not browser.observation_settling
        assert all(
            row["response"].get("result", {}).get("subtype") == "null" for row in transport.reads
        )
        assert_two_click_inputs(transport)
        # This final proof is outside the timed call, with no second act().
        page.wait_for_function("document.readyState==='complete'", timeout=2500)
        transport.reads.clear()
        observed = browser.observe(screenshot=False)
        assert observed["url"] == target
        later = assert_diagnostic(browser, "success")
        assert later["attempts"] == 1 and later["navigation_retries"] == 0
        assert later["late_snapshot"] is False
        assert caught.value.observation_settling == diagnostic
        assert_two_click_inputs(transport)


def test_loading_document_with_available_body_returns_without_waiting_for_complete(
    browser_runtime, streaming_server
):
    _, origin, start, target = streaming_server.register(0.8, body_first=True)
    with open_browser(browser_runtime, start, origin=origin) as (browser, page, transport, _):
        click_offered_link_once(browser, page, transport, target)
        # URL commit can precede parsing this already-delivered body. This
        # bounded read-only fixture gate precedes the observe call being tested.
        page.wait_for_function("document.body !== null", timeout=600)
        before = browser.evaluate(DOCUMENT)
        assert before["ready"] == "loading" and before["body"] is True
        observed = browser.observe(screenshot=False)
        diagnostic = assert_diagnostic(browser, "success")
        assert observed["url"] == target
        assert diagnostic["attempts"] == len(transport.reads) == 1
        assert diagnostic["navigation_retries"] == 0
        assert diagnostic["elapsed_seconds"] < 0.5
        assert browser.evaluate("document.readyState") == "loading"
        assert_two_click_inputs(transport)
        page.wait_for_function("document.readyState==='complete'", timeout=2000)


def test_javascript_evaluation_exception_is_not_retried_as_navigation(browser_runtime):
    with open_browser(browser_runtime) as (browser, page, transport, native):
        page.set_content(html("<button>Available</button>"))
        browser.evaluate(
            "Object.defineProperty(document,'body',{configurable:true,"
            "get(){throw new Error('owned unexpected body getter')}})"
        )
        with pytest.raises(native.StalePage, match="^Document changed during evaluation$"):
            browser.observe(screenshot=False)
        diagnostic = assert_diagnostic(browser, "unexpected_error")
        assert len(transport.reads) == diagnostic["attempts"] == 1
        assert diagnostic["navigation_retries"] == 0
        assert diagnostic["elapsed_seconds"] < 0.5
        assert "exceptionDetails" in transport.reads[0]["response"]
        assert transport.inputs == []


def test_profile_capture_failure_propagates_once_without_retry(browser_runtime):
    with open_browser(browser_runtime) as (browser, page, transport, _):
        page.set_content(html("<button>Available</button>"))
        sentinel = ValueError("owned observation evidence sink failed")

        def fail(_event):
            raise sentinel

        transport.observation_profile_sink = fail
        with pytest.raises(ValueError) as caught:
            browser.observe(screenshot=False)
        assert caught.value is sentinel
        diagnostic = assert_diagnostic(browser, "unexpected_error")
        assert diagnostic["attempts"] == len(transport.reads) == 1
        assert diagnostic["navigation_retries"] == 0
        assert transport.inputs == []
        transport.observation_profile_sink = None


def test_after_input_runtime_error_is_not_swallowed_or_retried(browser_runtime):
    with open_browser(browser_runtime) as (browser, page, transport, _):
        page.set_content(html("<button id=button>Owned button</button>"))
        observed = browser.observe(screenshot=False)
        action = next(a for a in observed["actions"] if a["label"] == "Owned button")
        browser.act(action, observed)
        transport.reads.clear()
        sentinel = RuntimeError("owned after-input transport failure")
        transport.after_input_error = sentinel
        with pytest.raises(RuntimeError) as caught:
            browser.observe(screenshot=False)
        assert caught.value is sentinel
        diagnostic = assert_diagnostic(browser, "unexpected_error")
        assert diagnostic["navigation_retries"] == 0
        assert transport.reads == []
        assert_two_click_inputs(transport)


def test_after_input_actual_javascript_exception_details_fail_fast(browser_runtime):
    with open_browser(browser_runtime) as (browser, page, transport, native):
        page.set_content(html("<button id=button>Owned button</button>"))
        observed = browser.observe(screenshot=False)
        action = next(a for a in observed["actions"] if a["label"] == "Owned button")
        browser.act(action, observed)
        transport.reads.clear()
        browser.evaluate(
            "window.requestAnimationFrame=()=>{throw new Error('owned after-input rejected promise')}"
        )
        with pytest.raises(native.StalePage, match="^Document changed during evaluation$"):
            browser.observe(screenshot=False)
        diagnostic = assert_diagnostic(browser, "unexpected_error")
        assert diagnostic["navigation_retries"] == 0
        assert transport.reads == []
        assert_two_click_inputs(transport)


@pytest.mark.parametrize("screenshot", [False, True])
def test_complete_but_late_actual_cdp_snapshot_is_rejected(browser_runtime, screenshot):
    with open_browser(browser_runtime) as (browser, page, transport, native):
        page.set_content(html("<button>Available</button>"))
        if screenshot:
            transport.delay_screenshot = 2.05
        else:
            transport.delay_snapshot = 2.05
        at = time.monotonic()
        with pytest.raises(
            native.StalePage, match="^Navigation settling target exceeded"
        ) as caught:
            browser.observe(screenshot=screenshot)
        diagnostic = assert_diagnostic(browser, "timeout")
        assert time.monotonic() - at >= 2.05
        assert diagnostic["late_snapshot"] is True
        assert diagnostic["attempts"] == len(transport.reads) == 1
        assert diagnostic["navigation_retries"] == 0
        assert diagnostic["overshoot_seconds"] >= 0.04
        assert caught.value.observation_settling == diagnostic
        actual = transport.reads[0]["response"]["result"]["value"]
        assert actual["observation_profile"]["id"] == PROFILE
        assert any(a["label"] == "Available" for a in actual["actions"])
        assert transport.inputs == []


@pytest.mark.parametrize("after_fresh", [False, True])
def test_settling_does_not_relax_old_proxy_freshness_or_executor_guard(
    browser_runtime, monkeypatch, after_fresh
):
    with open_browser(browser_runtime) as (browser, page, transport, native):
        page.set_content(
            html(
                "<input id=owner type=checkbox style='position:absolute;left:-9000px'>"
                "<input id=other type=checkbox style='position:absolute;left:-9000px'>"
                "<label id=label for=owner style='display:inline-block;padding:16px'>Bound label</label>"
            )
        )
        observed = browser.observe(screenshot=False)
        action = next(a for a in observed["actions"] if "control_node" in a)
        mutate = "document.querySelector('#label').htmlFor='other'"
        if after_fresh:
            original = browser.fresh

            def reassign_after_fresh(old_page, candidate=None):
                assert original(old_page, candidate) is True
                browser.evaluate(mutate)
                return True

            monkeypatch.setattr(browser, "fresh", reassign_after_fresh)
        else:
            browser.evaluate(mutate)
            assert browser.fresh(observed, action) is False
        with pytest.raises(native.StalePage):
            browser.act(action, observed)
        assert transport.inputs == []
        assert browser.evaluate("document.querySelector('#owner').checked") is False
        assert browser.evaluate("document.querySelector('#other').checked") is False
        assert browser.evaluate("events.filter(e=>e.trusted).length") == 0
