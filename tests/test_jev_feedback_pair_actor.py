"""Only synthetic cases and fake bootstraps; no browser, GPU or benchmark data."""

import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import jev_blind_feedback_pair_plan as pair_plan
import pytest
from test_jev_blind_feedback_pair_plan import synthetic_plan
from test_jev_browser_done_review_bootstrap import transport as transport
from test_jev_browser_done_review_owner import setup as setup

ROOT = Path(__file__).resolve().parents[1]
ACTOR = ROOT / "scripts" / "jev_feedback_pair_actor.py"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode()


def source_pair(directory, name, source):
    held, released = directory / f"{name}-held.py", directory / f"{name}-release.py"
    held.write_bytes(source)
    released.write_bytes(source.replace(b"SOURCE_ONLY = True", b"SOURCE_ONLY = False", 1))
    return {
        "held": str(held),
        "held_sha256": sha(held.read_bytes()),
        "released": str(released),
        "released_sha256": sha(released.read_bytes()),
    }


def load_actor(pair):
    spec = importlib.util.spec_from_file_location("_synthetic_pair_actor", pair["released"])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BRIDGE = b"""SOURCE_ONLY = True\nimport json, os\ndef run(bootstrap, **kw):\n    assert kw["enabled"] is True\n    assert os.environ["JEV_FEEDBACK_CHILD_TOKEN"] == "X" * 40\n    extra = {"destination_match": "unknown"} if kw.get("destination_attribution") else {}\n    kw["sink"]({**extra, "sequence": 1, "status": "response", "request_source": "response-attached-request", "request_bytes": b"{\\"alpha\\":1}", "response_bytes": b"{\\"beta\\":2}", "http_status": 200, "exception_type": None, "scope": "test"})\n    kw["sink"]({**extra, "sequence": 2, "status": "transport_exception", "request_source": "unavailable", "request_bytes": None, "response_bytes": None, "http_status": None, "exception_type": "SyntheticFailure", "scope": "test"})\n    kw["event_sink"]({"status": "review_call"})\n    print(json.dumps({"feedback": kw["feedback_enabled"], "arguments": kw["arguments"]}))\n    if os.environ.get("SYNTHETIC_FAIL") == "1": raise RuntimeError("synthetic failure")\n"""
BOOTSTRAP = b"""SOURCE_ONLY = True\ndef _bundle(*args):\n    return None\ndef run(**kwargs):\n    raise AssertionError("bridge must own call")\n"""


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    sources = tmp_path / "sources"
    sources.mkdir()
    actor_pair = source_pair(sources, "actor", ACTOR.read_bytes())
    actor = load_actor(actor_pair)
    try:
        bridge = source_pair(sources, "bridge", BRIDGE)
        capture_bootstrap = source_pair(sources, "bootstrap", BOOTSTRAP)
        owner = source_pair(sources, "owner", b"SOURCE_ONLY = True\n")
        capture = source_pair(sources, "capture", b"SOURCE_ONLY = True\n")
        files = {}
        for name in pair_plan._PIN_NAMES:
            path = sources / (name + ".txt")
            if name == "review_owner":
                path = Path(owner["held"])
            else:
                path.write_text("synthetic-" + name)
            files[name] = {"path": str(path), "sha256": sha(path.read_bytes())}
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        manifest = {
            "case_module": "original",
            "files": {
                "original_jev_browser_benchmark.py": {"sha256": files["runner"]["sha256"]},
                "jev_public_web_cases.py": {"sha256": files["case_and_judge"]["sha256"]},
            },
        }
        (bundle / "bundle-manifest.json").write_bytes(canonical(manifest))
        upstream = tmp_path / "upstream"
        upstream.mkdir()
        chromium = tmp_path / "chromium"
        chromium.write_text("synthetic")
        verifier_path = Path(pair_plan.__file__).resolve()
        config = {
            "schema": "jev-feedback-original-actor-v1",
            "source_files": files,
            "actor_source": actor_pair,
            "pair_verifier": {
                "path": str(verifier_path),
                "sha256": sha(verifier_path.read_bytes()),
            },
            "review_bridge": bridge,
            "capture_bootstrap": capture_bootstrap,
            "owner": owner,
            "capture": capture,
            "bundle_dir": str(bundle),
            "bundle_manifest_sha256": sha((bundle / "bundle-manifest.json").read_bytes()),
            "runner_sha256": "a" * 64,
            "profile_helper_sha256": None,
            "upstream_path": str(upstream),
            "model": "synthetic-model",
            "chromium_executable": str(chromium),
            "observation_profile": "native",
            "text_endpoint": "http://127.0.0.1:1235/v1",
            "text_model": "synthetic-text",
            "max_reviews": 2,
            "destination_attribution": False,
        }
        plan = synthetic_plan()
        plan["selected_cases"] = 1
        plan["planned_runs"] = 2
        plan["schedule"] = [
            {"token": actor._case_token(b"s" * 32, "fake_case"), "seed": 42, "order": ["off", "on"]}
        ]
        for name in ("off", "on"):
            plan["arms"][name]["pins"] = {key: item["sha256"] for key, item in files.items()}
            plan["arms"][name]["configuration_sha256"] = sha(canonical(config))
        yield actor, config, plan
    finally:
        sys.modules.pop("_synthetic_pair_actor", None)


def invoke(prepared, out, arm="off", **changes):
    actor, config, plan = prepared
    raw = canonical(plan)
    args = {
        "pair_index": 0,
        "arm_name": arm,
        "case_id": "fake_case",
        "selector_key": b"s" * 32,
        "endpoint": "http://127.0.0.1:1234",
        "api_key": "X" * 40,
        "private_dir": str(out),
    }
    args.update(changes)
    return actor.run_arm(raw, sha(raw), config, **args)


def test_cached_verifier_cannot_accept_different_arm_budgets(prepared, tmp_path, monkeypatch):
    _, _, plan = prepared
    plan["arms"]["on"]["budgets"]["actions"] += 1
    monkeypatch.setattr(pair_plan, "validate_plan", lambda _: None)
    out = tmp_path / "different-budgets"
    out.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="arm configurations differ"):
        invoke(prepared, out)
    assert list(out.iterdir()) == []
    assert "_jev_feedback_pair_verifier" not in sys.modules


def test_cached_verifier_cannot_reject_valid_pinned_plan(prepared, tmp_path, monkeypatch):
    def reject(_):
        raise ValueError("untrusted cached verifier")

    monkeypatch.setattr(pair_plan, "validate_plan", reject)
    out = tmp_path / "valid-plan"
    out.mkdir(mode=0o700)
    assert invoke(prepared, out)["capture_count"] == 2
    assert "_jev_feedback_pair_verifier" not in sys.modules


def test_verifier_identity_change_fails_closed(prepared, tmp_path):
    _, config, plan = prepared
    verifier = tmp_path / "synthetic-verifier.py"
    verifier.write_text("import sys\ndef validate_plan(plan):\n    del sys.modules[__name__]\n")
    config["pair_verifier"] = {"path": str(verifier), "sha256": sha(verifier.read_bytes())}
    for arm in ("off", "on"):
        plan["arms"][arm]["configuration_sha256"] = sha(canonical(config))
    out = tmp_path / "changed-verifier"
    out.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="pair-plan verifier identity changed"):
        invoke(prepared, out)
    assert list(out.iterdir()) == []
    assert "_jev_feedback_pair_verifier" not in sys.modules


def test_two_arms_only_change_feedback_and_preserve_all_capture(prepared, tmp_path, monkeypatch):
    monkeypatch.setenv("JEV_FEEDBACK_CHILD_TOKEN", "original")
    results = []
    output = []
    for arm in ("off", "on"):
        out = tmp_path / arm
        out.mkdir(mode=0o700)
        results.append(invoke(prepared, out, arm))
        output.append(json.loads((out / "runner-stdout.txt").read_text()))
        events = [
            json.loads(line) for line in (out / "capture-events.jsonl").read_text().splitlines()
        ]
        assert [event["status"] for event in events] == ["response", "transport_exception"]
        assert (out / events[0]["request_file"]).read_bytes() == b'{"alpha":1}'
        assert events[1]["request_file"] is None
        assert (out / events[0]["request_file"]).stat().st_mode & 0o077 == 0
        assert "fake_case" not in str(results[-1])
        assert results[-1]["capture_count"] == 2
        assert results[-1]["browser_success"] is None
        assert results[-1]["os_closure"] == "unverified"
        assert os.environ["JEV_FEEDBACK_CHILD_TOKEN"] == "original"
    assert [result["feedback"] for result in output] == [False, True]
    left, right = output[0]["arguments"], output[1]["arguments"]
    left[left.index("--output-dir") + 1] = "private-output"
    right[right.index("--output-dir") + 1] = "private-output"
    assert left == right
    assert left[left.index("--seed") + 1] == "42"
    assert left[left.index("--public-cases") + 1] == "fake_case"
    assert all("X" * 40 not in arg for arg in left)


@pytest.mark.parametrize(
    "changes",
    [
        {"case_id": "other_case"},
        {"selector_key": b"z" * 32},
        {"endpoint": "http://example.org:1234"},
        {"endpoint": "http://127.0.0.1:1234?token=secret"},
        {"api_key": "short"},
        {"arm_name": "other"},
        {"pair_index": 1},
    ],
)
def test_invalid_binding_refused_before_output(prepared, tmp_path, changes):
    out = tmp_path / "out"
    out.mkdir(mode=0o700)
    with pytest.raises((ValueError, KeyError)):
        invoke(prepared, out, **changes)
    assert list(out.iterdir()) == []


def test_unpinned_config_and_heldout_bundle_refused(prepared, tmp_path):
    actor, config, plan = prepared
    out = tmp_path / "out"
    out.mkdir(mode=0o700)
    changed = copy.deepcopy(config)
    changed["model"] = "different"
    raw = canonical(plan)
    with pytest.raises(ValueError, match="configuration"):
        actor.run_arm(
            raw,
            sha(raw),
            changed,
            pair_index=0,
            arm_name="off",
            case_id="fake_case",
            selector_key=b"s" * 32,
            endpoint="http://127.0.0.1:1234",
            api_key="X" * 40,
            private_dir=str(out),
        )
    manifest_path = Path(config["bundle_dir"]) / "bundle-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["case_module"] = "heldout"
    manifest_path.write_bytes(canonical(manifest))
    config["bundle_manifest_sha256"] = sha(manifest_path.read_bytes())
    for name in ("off", "on"):
        plan["arms"][name]["configuration_sha256"] = sha(canonical(config))
    with pytest.raises(ValueError, match="public-case bundle"):
        invoke(prepared, out)
    assert list(out.iterdir()) == []


def test_changed_source_and_nonprivate_output_refused(prepared, tmp_path):
    actor, config, _ = prepared
    public = tmp_path / "public"
    public.mkdir(mode=0o755)
    with pytest.raises(ValueError, match="private directory"):
        invoke(prepared, public)
    out = tmp_path / "out"
    out.mkdir(mode=0o700)
    Path(config["source_files"]["runner"]["path"]).write_text("tampered")
    with pytest.raises(ValueError, match="source bytes"):
        invoke(prepared, out)
    assert list(out.iterdir()) == []


def test_failure_preserves_partial_private_evidence_and_restores_environment(
    prepared, tmp_path, monkeypatch
):
    out = tmp_path / "failure"
    out.mkdir(mode=0o700)
    monkeypatch.setenv("SYNTHETIC_FAIL", "1")
    monkeypatch.setenv("JEV_FEEDBACK_CHILD_TOKEN", "previous")
    with pytest.raises(RuntimeError, match="synthetic failure"):
        invoke(prepared, out)
    assert len((out / "capture-events.jsonl").read_text().splitlines()) == 2
    assert len((out / "review-events.jsonl").read_text().splitlines()) == 1
    assert json.loads((out / "runner-stdout.txt").read_text())["feedback"] is False
    assert os.environ["JEV_FEEDBACK_CHILD_TOKEN"] == "previous"
    assert "_jev_feedback_review_bridge" not in sys.modules
    assert "_jev_feedback_capture_bootstrap" not in sys.modules


def test_budget_overflow_preserves_final_capture_before_refusal(prepared, tmp_path):
    _, config, plan = prepared
    source = b"""SOURCE_ONLY = True\ndef run(bootstrap, **kw):\n    for n in range(1, 40):\n        kw["sink"]({"sequence": n, "status": "transport_exception", "request_source": "unavailable", "request_bytes": None, "response_bytes": None, "http_status": None, "exception_type": "SyntheticFailure", "scope": "test"})\n"""
    config["review_bridge"] = source_pair(tmp_path, "flood", source)
    for arm in ("off", "on"):
        plan["arms"][arm]["configuration_sha256"] = sha(canonical(config))
    out = tmp_path / "flood-out"
    out.mkdir(mode=0o700)
    with pytest.raises(ValueError, match="captured call budget"):
        invoke(prepared, out)
    assert len((out / "capture-events.jsonl").read_text().splitlines()) == 21
    assert (out / "runner-stdout.txt").read_text() == ""


# Reuse existing synthetic model and HTTPX transport without registering pytest
# plugins globally; both imported fixtures operate solely on synthetic data.

REAL_BOOTSTRAP_ADAPTER = b"""SOURCE_ONLY = True\nimport json, sys\nfrom jev_browser_private_capture_bootstrap import _module, _capture_source\nfrom test_jev_browser_done_review_bootstrap import Bootstrap\ndef _bundle(*args): return None\ndef run(*, sink, runner_sha256, **kwargs):\n    ctx = sys.modules["_synthetic_feedback_test_context"]\n    wrapped = Bootstrap(ctx.setup, "unknown")\n    wrapped._module = sys.modules[__name__]._module\n    class Captured:\n        def append(self, event): sink(event)\n    wrapped.captured = Captured()\n    result = wrapped.run(runner_sha256=runner_sha256)\n    print(json.dumps({"status": result["status"]}))\n"""


@pytest.mark.parametrize("arm", ["off", "on"])
def test_real_review_bridge_and_transport_reach_synthetic_runner(
    prepared, tmp_path, monkeypatch, setup, transport, arm
):
    import types

    actor, config, plan = prepared
    context = types.ModuleType("_synthetic_feedback_test_context")
    context.setup = setup
    monkeypatch.setitem(sys.modules, context.__name__, context)
    bridge_source = (ROOT / "scripts" / "jev_browser_done_review_bootstrap.py").read_bytes()
    owner_source = (ROOT / "scripts" / "jev_browser_done_review_owner.py").read_bytes()
    config["review_bridge"] = source_pair(tmp_path, "real-bridge", bridge_source)
    config["capture_bootstrap"] = source_pair(tmp_path, "fake-bootstrap", REAL_BOOTSTRAP_ADAPTER)
    config["owner"] = source_pair(tmp_path, "real-owner", owner_source)
    config["source_files"]["review_owner"] = {
        "path": config["owner"]["held"],
        "sha256": config["owner"]["held_sha256"],
    }
    for name in ("off", "on"):
        plan["arms"][name]["pins"]["review_owner"] = config["owner"]["held_sha256"]
        plan["arms"][name]["configuration_sha256"] = sha(canonical(config))
    out = tmp_path / ("bridge-" + arm)
    out.mkdir(mode=0o700)
    result = invoke(prepared, out, arm)
    assert result["capture_count"] == 3
    assert json.loads((out / "runner-stdout.txt").read_text())["status"] == "ready"
    captures = [
        json.loads(line) for line in (out / "capture-events.jsonl").read_text().splitlines()
    ]
    events = [json.loads(line) for line in (out / "review-events.jsonl").read_text().splitlines()]
    assert [item["status"] for item in events] == ["review_call", "replan_call", "replanned"]
    replan = json.loads((out / captures[-1]["request_file"]).read_bytes())
    assert ("completion_review_advisory" in replan["state"]) is (arm == "on")
    assert "_jev_feedback_review_bridge" not in sys.modules
    assert "_jev_feedback_capture_bootstrap" not in sys.modules


def test_optional_destination_label_preserves_unknown(prepared, tmp_path):
    _, config, plan = prepared
    config["destination_attribution"] = True
    for arm in ("off", "on"):
        plan["arms"][arm]["configuration_sha256"] = sha(canonical(config))
    out = tmp_path / "attribution"
    out.mkdir(mode=0o700)
    result = invoke(prepared, out)
    events = [json.loads(line) for line in (out / "capture-events.jsonl").read_text().splitlines()]
    assert [event["destination_match"] for event in events] == ["unknown", "unknown"]
    assert result["capture_count"] == 2
    assert result["browser_success"] is None


def test_token_escaping_never_written(prepared, tmp_path):
    actor, _, _ = prepared
    token = "X" * 40
    with pytest.raises(ValueError, match="private token"):
        actor._no_secret(('"' + r"\u0058" * 40 + '"').encode(), token)
