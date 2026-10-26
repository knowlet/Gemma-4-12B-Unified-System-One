"""Shared helper/profile composition; no model, network or browser starts."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from local_evidence import require_local_evidence


@pytest.fixture
def driver():
    path = Path(
        os.environ.get("JEV_NATIVE_V2_ACTION_PROFILE_CANDIDATE", "scripts/jev_native_v2_browser.py")
    )
    spec = importlib.util.spec_from_file_location("_native_action_profile_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def local_experimental_profile_sources():
    from s1 import jev_action_profiles

    require_local_evidence(
        *(
            jev_action_profiles._ROOT / label
            for label, _digest in jev_action_profiles._COMPONENTS.values()
        )
    )


def test_legacy_settings_remain_empty(driver):
    assert driver.action_settings() == {}


@pytest.mark.parametrize("profile", [None, 1, [], "find", ""])
def test_unknown_profile_refused_before_inputs_or_model(driver, profile):
    with pytest.raises(ValueError, match="profile"):
        driver.metadata_plan(
            "absent", "absent", "absent", "shared-elements-v2", action_profile=profile
        )


@pytest.mark.parametrize("endpoint,model", [(None, "small"), ("http://127.0.0.1:8123", None)])
def test_half_helper_configuration_refused(driver, endpoint, model):
    with pytest.raises(ValueError, match="both"):
        driver.action_settings(text_endpoint=endpoint, text_model=model)


def test_find_requires_explicit_shared_http_helper(driver):
    with pytest.raises(ValueError, match="shared external"):
        driver.action_settings("jev-find-text-experimental-v1")


def test_server_refuses_half_selected_experiment_before_backend_construction(driver):
    with pytest.raises(ValueError, match="shared external helper"):
        driver.make_application(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            action_profile="jev-find-text-experimental-v1",
        )


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://example.invalid:443",
        "http://127.0.0.1",
        "http://u:p@127.0.0.1:8123",
        "http://127.0.0.1:8123?q=secret",
        "http://127.0.0.1:8123#secret",
    ],
)
def test_endpoint_cannot_smuggle_remote_destination_or_credentials(driver, endpoint):
    with pytest.raises(ValueError):
        driver.action_settings(text_endpoint=endpoint, text_model="small")


@pytest.mark.parametrize(
    "key",
    [
        "S1_POLICY_API_KEY",
        "TEXT_MODEL_REASONING",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONNOUSERSITE",
        "PYTHONPATH",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "TOKENIZERS_PARALLELISM",
        "HF_HOME",
        "HF_HUB_CACHE",
        "A-B",
        "",
        None,
    ],
)
def test_external_helper_key_cannot_collide_with_policy_or_invalid_environment_name(driver, key):
    with pytest.raises(ValueError, match="environment"):
        driver.action_settings(
            text_endpoint="http://127.0.0.1:8123", text_model="small", text_api_key_env=key
        )


def test_selected_metadata_contains_environment_name_without_secret(driver, monkeypatch):
    monkeypatch.setenv("SHARED_TEST_KEY", "secret-value-must-not-appear")
    selected = driver.action_settings(
        "jev-find-text-experimental-v1", "http://127.0.0.1:8123/", "small", "SHARED_TEST_KEY"
    )
    assert selected["action_profile"] == "jev-find-text-experimental-v1"
    assert selected["text_helper"]["endpoint"] == "http://127.0.0.1:8123"
    assert selected["text_helper"]["api_key_env"] == "SHARED_TEST_KEY"
    assert selected["helper_tokens"] is None
    assert selected["text_helper"]["original_request_max_tokens"] == 1024
    assert selected["text_helper"]["native_limit_authenticated"] is False
    assert "secret-value-must-not-appear" not in json.dumps(selected)


@pytest.mark.parametrize("experimental", [False, True])
def test_groups_keep_cases_budget_and_common_helper_without_overwriting_key(
    driver, monkeypatch, tmp_path, experimental
):
    calls, bundles = [], []
    profile = "jev-find-text-experimental-v1" if experimental else "standard"

    def prepare(path, *, case_module):
        path.mkdir(parents=True)
        bundles.append(case_module)
        return path, {"fake": "bounded composition test"}

    def verify(path, manifest):
        assert path.exists() and manifest["fake"]

    module = SimpleNamespace(prepare_bundle=prepare, verify_bundle=verify)
    monkeypatch.setitem(
        sys.modules,
        "jev_find_text_browser" if experimental else "jev_browser_profile_v7_benchmark",
        module,
    )
    monkeypatch.setenv("S1_TEXT_API_KEY", "shared-secret")

    def client(argv, env, output):
        calls.append((argv, env))
        ids = argv[argv.index("--public-cases") + 1 :]
        (output / "results").mkdir()
        (output / "results/summary.json").write_text(
            json.dumps({"attempts": len(ids), "successes": 0})
        )

    monkeypatch.setattr(driver, "run_client", client)
    groups = driver.run_groups(
        tmp_path,
        tmp_path / "upstream",
        "http://127.0.0.1:9000",
        "policy-secret",
        "chromium-not-started",
        action_profile=profile,
        text_helper={
            "endpoint": "http://127.0.0.1:8123",
            "model": "small",
            "api_key_env": "S1_TEXT_API_KEY",
        },
    )
    assert bundles == ["original", "heldout"]
    assert [group["denominator"] for group in groups] == [4, 6]
    for argv, env in calls:
        assert argv[argv.index("--max-steps") + 1] == "16"
        assert argv[argv.index("--repeats") + 1] == "1"
        assert argv[argv.index("--text-endpoint") + 1] == "http://127.0.0.1:8123"
        assert argv[argv.index("--text-model") + 1] == "small"
        assert env["S1_TEXT_API_KEY"] == "shared-secret"
        assert env["S1_POLICY_API_KEY"] == "policy-secret"
        assert env["TEXT_MODEL_REASONING"] == "none"


def test_legacy_groups_still_use_original_local_helper(driver, monkeypatch, tmp_path):
    calls = []

    def prepare(path, **kwargs):
        path.mkdir(parents=True)
        return path, {}

    monkeypatch.setitem(
        sys.modules,
        "jev_browser_profile_v7_benchmark",
        SimpleNamespace(prepare_bundle=prepare, verify_bundle=lambda *_: None),
    )

    def client(argv, env, output):
        calls.append((argv, env))
        (output / "results").mkdir()
        ids = argv[argv.index("--public-cases") + 1 :]
        (output / "results/summary.json").write_text(
            json.dumps({"attempts": len(ids), "successes": 0})
        )

    monkeypatch.setattr(driver, "run_client", client)
    driver.run_groups(tmp_path, tmp_path, "http://127.0.0.1:9000", "policy", "not-started")
    for argv, env in calls:
        assert argv[argv.index("--text-endpoint") + 1] == "http://127.0.0.1:9000"
        assert argv[argv.index("--text-model") + 1] == driver.IDENTITY
        assert env["S1_TEXT_API_KEY"] == "policy"


@pytest.mark.local_evidence
def test_experimental_server_selects_profile_and_refuses_local_helper_fallback(
    driver, monkeypatch, tmp_path, local_experimental_profile_sources
):
    import jev_field_syntax_helper
    from fastapi.testclient import TestClient
    from test_jev_native_v2_browser import helper_payload, model_fixture

    from s1.browser_server_evidence import ServerEvidenceJournal

    def forbidden(*_args, **_kwargs):
        pytest.fail("An external helper selection called local generation")

    monkeypatch.setattr(jev_field_syntax_helper, "generate_field_text", forbidden)
    journal = ServerEvidenceJournal(tmp_path / "server")
    gate = driver.ResourceJournal(
        tmp_path / "resources.jsonl",
        "test-only-boot",
        observe=lambda: {
            "boot_id": "test-only-boot",
            "MemTotal": 130_595_860_480,
            "MemAvailable": 118_000_000_000,
        },
    )
    app = driver.make_application(
        model_fixture(),
        None,
        journal,
        gate,
        tmp_path,
        "policy-token",
        "raw-json-v1",
        "original system",
        action_profile="jev-find-text-experimental-v1",
        allow_local_helper=False,
    )
    with TestClient(app) as client:
        health = client.get("/healthz").json()
        response = client.post(
            "/chat/completions",
            json=helper_payload(),
            headers={"Authorization": "Bearer policy-token"},
        )
    assert health["action_profile"] == "jev-find-text-experimental-v1"
    assert health["action_profile_provenance"]["operation_count"] == 9
    assert response.status_code == 404
    assert "answers" not in response.json()


@pytest.mark.local_evidence
def test_experimental_composition_closes_all_selected_sources(
    driver, monkeypatch, tmp_path, local_experimental_profile_sources
):
    import jev_native_v2_browser as installed

    monkeypatch.setattr(driver, "ROOT", installed.ROOT)
    require_local_evidence(driver.MATRIX)
    paths = driver.composition_paths("jev-find-text-experimental-v1")
    assert "scripts/jev_find_text_browser.py" in paths
    assert "src/s1/jev_action_profiles.py" in paths
    assert "artifacts/jev-runtime/browser-find-text-profile-20261022-01/find_profile.py" in paths
    expected = driver.capture_composition(tmp_path, "jev-find-text-experimental-v1")
    driver.verify_composition(tmp_path, expected, "jev-find-text-experimental-v1")
    copied = tmp_path / "composition-sources/src/s1/jev_action_profiles.py"
    copied.write_bytes(copied.read_bytes() + b"\n# synthetic tamper\n")
    with pytest.raises(ValueError, match="bytes changed"):
        driver.verify_composition(tmp_path, expected, "jev-find-text-experimental-v1")
