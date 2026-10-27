"""Synthetic bootstrap composition; no benchmark cases, browser, or GPU."""

import hashlib
import json
import sys
from pathlib import Path

import httpx
import jev_browser_done_review_bootstrap as bridge
import jev_browser_private_capture_bootstrap as capture
import jev_browser_private_transport_capture as transport_capture
import pytest
from test_jev_browser_done_review_owner import Agent

pytest_plugins = ("test_jev_browser_done_review_owner",)


RUNNER = b"import os\nimport jev_browser_benchmark\nmodel_adapter = jev_browser_benchmark.model_adapter\n"


class Bootstrap:
    _module = staticmethod(capture._module)
    _capture_source = staticmethod(capture._capture_source)

    def __init__(self, case, verdict, fail=False):
        self.case, self.verdict, self.fail = case, verdict, fail
        self.records = None
        self.agent = None
        self.captured = []

    def run(self, *, runner_sha256):
        name = "_jev_private_runner_" + runner_sha256[:16]
        module = self._module(Path("synthetic_runner.py"), RUNNER, name)
        try:
            self.case.options["review"] = self.verdict
            with transport_capture.attach_to_runner(module, sink=self.captured.append):
                with module.model_adapter(
                    self.case.agent_module,
                    self.case.model,
                    endpoint="http://127.0.0.1:9000",
                    model="synthetic",
                    api_key="private-token-123",
                ) as (self.records, _):
                    self.agent = Agent(self.case.model)
                    if self.fail:
                        raise RuntimeError("synthetic runner failure")
                    return self.agent.command("tick")
        finally:
            if sys.modules.get(name) is module:
                del sys.modules[name]


@pytest.fixture
def transport(setup, monkeypatch):
    monkeypatch.setattr(transport_capture, "SOURCE_ONLY", False)
    network = setup.model.post_json

    def handler(request):
        return httpx.Response(
            200, json=network(str(request.url), "key", json.loads(request.content))
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    setup.model.CLIENT = client
    setup.model.post_json = lambda url, key, body: client.post(url, json=body).json()
    yield
    client.close()


@pytest.fixture
def pins(tmp_path, monkeypatch):
    monkeypatch.setattr(bridge, "SOURCE_ONLY", False)
    held = Path(bridge.__file__).with_name("jev_browser_done_review_owner.py").read_bytes()
    release = held.replace(b"SOURCE_ONLY = True", b"SOURCE_ONLY = False", 1)
    held_path, release_path = tmp_path / "held.py", tmp_path / "released.py"
    held_path.write_bytes(held)
    release_path.write_bytes(release)
    return {
        "held_owner_path": held_path,
        "held_owner_sha256": hashlib.sha256(held).hexdigest(),
        "released_owner_path": release_path,
        "released_owner_sha256": hashlib.sha256(release).hexdigest(),
        "runner_sha256": hashlib.sha256(RUNNER).hexdigest(),
    }


@pytest.mark.parametrize(
    "verdict, expected, calls", [("supported", "done", 2), ("unknown", "ready", 3)]
)
def test_bootstrap_wraps_loaded_runner_and_restores_callbacks(
    setup, pins, transport, verdict, expected, calls
):
    bootstrap = Bootstrap(setup, verdict)
    original = bootstrap._module
    command = Agent.command
    result = bridge.run(bootstrap, enabled=True, **pins)
    assert result["status"] == expected
    assert len(bootstrap.records) == calls
    assert len(bootstrap.captured) == calls
    assert all(event["status"] == "response" for event in bootstrap.captured)
    assert bootstrap.agent.state["browser"].actions == ([] if verdict == "supported" else ["item"])
    assert bootstrap._module is original and Agent.command is command
    assert "_jev_done_review_" + pins["released_owner_sha256"][:16] not in sys.modules


def test_bootstrap_failure_cleans_up(setup, pins, transport):
    bootstrap = Bootstrap(setup, "unknown", fail=True)
    original, command = bootstrap._module, Agent.command
    with pytest.raises(RuntimeError, match="synthetic runner failure"):
        bridge.run(bootstrap, enabled=True, **pins)
    assert bootstrap._module is original and Agent.command is command
    assert "_jev_done_review_" + pins["released_owner_sha256"][:16] not in sys.modules


def test_disabled_directly_delegates_without_release(setup, pins, transport, monkeypatch):
    monkeypatch.setattr(bridge, "SOURCE_ONLY", True)
    bootstrap = Bootstrap(setup, "supported")
    original = bootstrap._module
    assert (
        bridge.run(bootstrap, enabled=False, runner_sha256=pins["runner_sha256"])["status"]
        == "done"
    )
    assert len(bootstrap.records) == 1 and bootstrap._module is original
