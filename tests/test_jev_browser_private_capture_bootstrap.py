"""Bootstrap uses synthetic source and MockTransport; never starts a browser."""

import hashlib
import importlib
import json
import sys
import types
from pathlib import Path

import jev_browser_private_capture_bootstrap as bootstrap
import jev_browser_private_transport_capture as held_capture
import pytest


def sha(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(bootstrap, "SOURCE_ONLY", False)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    for name in (
        "jev_public_web_cases",
        "jev_heldout_public_cases",
        "jev_observation_profile_v7",
        "jev_observation_events_v7",
        "jev_observation_profile_v3_parent",
        "jev_observation_events_v3_parent",
        "_jev_observation_profile_v7_parent_8d199ac1564a",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)
    source = b"""import os, sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import httpx

if False:
    def untouched_public_case_import():
        from jev_public_web_cases import (
            SOME_CASE,
        )

@contextmanager
def model_adapter(agent, model, **kwargs):
    original = model.CLIENT.post
    def harness_post(*args, **kw):
        return original(*args, **kw)
    model.CLIENT.post = harness_post
    try:
        yield [], []
    finally:
        model.CLIENT.post = original

def main():
    if "--mutate" in sys.argv:
        (Path(__file__).parent / "jev_public_web_cases.py").write_bytes(b"changed")
        raise KeyError("original failure")
    client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, content=b'{"ok":true}')))
    model = SimpleNamespace(CLIENT=client)
    try:
        with model_adapter(None, model, api_key="synthetic-private-token"):
            response = client.post("https://example.invalid", json={"invented": 1}, headers={"Authorization": "Bearer synthetic-private-token"})
        return {"args": sys.argv[:], "path0": sys.path[0], "status": response.status_code, "name": __name__}
    finally:
        client.close()
"""
    assert source.count(bootstrap.ORIGINAL_IMPORT) == 1
    derivative = source.replace(bootstrap.ORIGINAL_IMPORT, bootstrap.ISOLATED_IMPORT, 1)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    files = {name: b"# synthetic source\n" for name in bootstrap.FILES}
    files["original_jev_browser_benchmark.py"] = source
    files["runner.py"] = derivative
    for name, data in files.items():
        (bundle / name).write_bytes(data)
    manifest = {
        "schema_id": "jev-heldout-evaluator-bundle-v1",
        "source_change": {
            "kind": "one_explicit_public_case_import_substitution",
            "original_sha256": sha(source),
            "derivative_sha256": sha(derivative),
        },
        "files": {
            name: {"sha256": sha(data), "size_bytes": len(data)} for name, data in files.items()
        },
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
    (bundle / "bundle-manifest.json").write_bytes(manifest_bytes)
    held = Path(held_capture.__file__).read_bytes()
    released = held.replace(b"SOURCE_ONLY = True", b"SOURCE_ONLY = False", 1)
    held_path, release_path = tmp_path / "held.py", tmp_path / "released.py"
    held_path.write_bytes(held)
    release_path.write_bytes(released)
    events = []
    arguments = {
        "bundle_dir": str(bundle),
        "manifest_sha256": sha(manifest_bytes),
        "runner_sha256": sha(derivative),
        "held_capture_path": str(held_path),
        "held_capture_sha256": sha(held),
        "released_capture_path": str(release_path),
        "released_capture_sha256": sha(released),
        "sink": events.append,
        "arguments": ["--example", "unchanged"],
    }
    return arguments, events


def test_same_process_main_retains_argv_and_uses_private_sink(fixture):
    config, events = fixture
    before_argv, before_path = sys.argv, list(sys.path)
    result = bootstrap.run(**config)
    assert result["args"] == [
        str(Path(config["bundle_dir"]) / "runner.py"),
        "--example",
        "unchanged",
    ]
    assert result["path0"] == config["bundle_dir"] and result["status"] == 200
    assert result["name"].startswith("_jev_private_runner_")
    assert sys.argv is before_argv and sys.path == before_path
    assert len(events) == 1 and events[0]["request_bytes"] == b'{"invented":1}'
    assert events[0]["response_bytes"] == b'{"ok":true}'
    assert "headers" not in events[0] and "url" not in events[0]
    assert b"synthetic-private-token" not in events[0]["request_bytes"]
    assert not any(
        name.startswith("_jev_private_runner_") or name.startswith("_jev_private_capture_")
        for name in sys.modules
    )


@pytest.mark.parametrize("item", ["manifest", "runner", "held", "release", "extra"])
def test_invalid_source_never_reaches_runner(fixture, item):
    config, events = fixture
    paths = {
        "manifest": Path(config["bundle_dir"]) / "bundle-manifest.json",
        "runner": Path(config["bundle_dir"]) / "runner.py",
        "held": Path(config["held_capture_path"]),
        "release": Path(config["released_capture_path"]),
        "extra": Path(config["bundle_dir"]) / "untracked.py",
    }
    paths[item].write_bytes(b"changed")
    with pytest.raises(ValueError):
        bootstrap.run(**config)
    assert events == []


def test_primary_exception_and_source_failure_are_both_visible_and_modules_removed(fixture):
    config, events = fixture
    config["arguments"] = ["--mutate"]
    before_argv, before_path = sys.argv, list(sys.path)
    with pytest.raises(KeyError, match="original failure") as caught:
        bootstrap.run(**config)
    assert caught.value.__notes__ == ["Private bootstrap integrity or cleanup failed: ValueError"]
    assert events == [] and sys.argv is before_argv and sys.path == before_path
    assert not any(
        name.startswith("_jev_private_runner_") or name.startswith("_jev_private_capture_")
        for name in sys.modules
    )


def test_source_gate_and_bare_child_without_no_bytecode_fail_before_model(fixture, monkeypatch):
    config, events = fixture
    monkeypatch.setattr(bootstrap, "SOURCE_ONLY", True)
    with pytest.raises(ValueError, match="independent review"):
        bootstrap.run(**config)
    monkeypatch.setattr(bootstrap, "SOURCE_ONLY", False)
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    with pytest.raises(ValueError, match="no-bytecode"):
        bootstrap.run(**config)
    assert events == []


def test_private_sink_proxy_refuses_headers_or_out_of_order_events():
    seen = []
    proxy = bootstrap._sink_proxy(seen.append)
    event = {
        "sequence": 1,
        "status": "response",
        "request_source": "response-attached-request",
        "request_bytes": b"{}",
        "response_bytes": b"{}",
        "http_status": 200,
        "exception_type": None,
        "scope": "httpx-observed content; response may be decoded",
    }
    with pytest.raises(ValueError):
        proxy({**event, "headers": {"Authorization": "private"}})
    with pytest.raises(ValueError):
        proxy({**event, "sequence": 2})
    assert seen == []
    proxy(event)
    assert seen == [event]


@pytest.mark.parametrize("case_module", ["original", "heldout"])
def test_real_v7_bundle_inventory_and_transform_are_verified(tmp_path, case_module):
    import jev_browser_profile_v7_benchmark as v7

    bundle, manifest = v7.prepare_bundle(tmp_path / "v7", case_module=case_module)
    manifest_bytes = (bundle / "bundle-manifest.json").read_bytes()
    verified, contents = bootstrap._bundle(
        str(bundle),
        sha(manifest_bytes),
        manifest["source_delta"]["derivative_sha256"],
        manifest["profile_helper_sha256"],
    )
    assert verified == bundle and contents["runner.py"] == (bundle / "runner.py").read_bytes()
    with pytest.raises(ValueError, match="profile and case"):
        bootstrap._bundle(
            str(bundle),
            sha(manifest_bytes),
            manifest["source_delta"]["derivative_sha256"],
            "0" * 64,
        )


def test_real_v7_runner_loads_pinned_bundle_modules_in_child_namespace(
    fixture, tmp_path, monkeypatch
):
    import jev_browser_profile_v7_benchmark as v7

    config, events = fixture
    bundle, manifest = v7.prepare_bundle(tmp_path / "v7-original", case_module="original")
    config.update(
        bundle_dir=str(bundle),
        manifest_sha256=sha((bundle / "bundle-manifest.json").read_bytes()),
        runner_sha256=manifest["source_delta"]["derivative_sha256"],
        profile_helper_sha256=manifest["profile_helper_sha256"],
        arguments=[
            "--observation-profile",
            bootstrap.V7_PROFILE,
            "--observation-profile-helper-sha256",
            manifest["profile_helper_sha256"],
        ],
    )
    original_module = bootstrap._module

    def instrumented(path, raw, name):
        module = original_module(path, raw, name)
        if name.startswith("_jev_private_runner_"):

            def synthetic_main():
                names = (
                    "jev_public_web_cases",
                    "jev_observation_profile_v7",
                    "jev_observation_events_v7",
                )
                return [importlib.import_module(name).__file__ for name in names]

            module.main = synthetic_main
        return module

    monkeypatch.setattr(bootstrap, "_module", instrumented)
    result = bootstrap.run(**config)
    assert result == [
        str(bundle / (name + ".py"))
        for name in (
            "jev_public_web_cases",
            "jev_observation_profile_v7",
            "jev_observation_events_v7",
        )
    ]
    assert events == []
    assert not any(
        name in sys.modules
        for name in (
            "jev_public_web_cases",
            "jev_observation_profile_v7",
            "jev_observation_events_v7",
            "_jev_observation_profile_v7_parent_8d199ac1564a",
        )
    )


@pytest.mark.parametrize(
    "name",
    [
        "jev_public_web_cases",
        "jev_heldout_public_cases",
        "jev_observation_profile_v7",
        "jev_observation_events_v7",
        "_jev_observation_profile_v7_parent_8d199ac1564a",
    ],
)
def test_preloaded_bundle_names_are_rejected_without_overwriting(
    fixture, tmp_path, monkeypatch, name
):
    import jev_browser_profile_v7_benchmark as v7

    config, events = fixture
    bundle, manifest = v7.prepare_bundle(tmp_path / "v7-heldout", case_module="heldout")
    config.update(
        bundle_dir=str(bundle),
        manifest_sha256=sha((bundle / "bundle-manifest.json").read_bytes()),
        runner_sha256=manifest["source_delta"]["derivative_sha256"],
        profile_helper_sha256=manifest["profile_helper_sha256"],
        arguments=[
            "--observation-profile",
            bootstrap.V7_PROFILE,
            "--observation-profile-helper-sha256",
            manifest["profile_helper_sha256"],
        ],
    )
    foreign = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, name, foreign)
    with pytest.raises(ValueError, match="preloaded bundle"):
        bootstrap.run(**config)
    assert sys.modules[name] is foreign and events == []


@pytest.mark.parametrize("mode", ["missing", "wrong", "duplicate"])
def test_v7_requires_the_original_explicit_forwarded_profile(fixture, tmp_path, mode):
    import jev_browser_profile_v7_benchmark as v7

    config, events = fixture
    bundle, manifest = v7.prepare_bundle(tmp_path / "v7-arguments", case_module="original")
    arguments = [
        "--observation-profile",
        bootstrap.V7_PROFILE,
        "--observation-profile-helper-sha256",
        manifest["profile_helper_sha256"],
    ]
    if mode == "missing":
        arguments = []
    elif mode == "wrong":
        arguments[1] = "native-1231850"
    else:
        arguments += ["--observation-profile", bootstrap.V7_PROFILE]
    config.update(
        bundle_dir=str(bundle),
        manifest_sha256=sha((bundle / "bundle-manifest.json").read_bytes()),
        runner_sha256=manifest["source_delta"]["derivative_sha256"],
        profile_helper_sha256=manifest["profile_helper_sha256"],
        arguments=arguments,
    )
    with pytest.raises(ValueError, match="V7 runner profile"):
        bootstrap.run(**config)
    assert events == []
