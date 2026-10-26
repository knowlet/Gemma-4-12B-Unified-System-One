"""Original V7 script/Agent/model adapter with explicit CPU fake transport."""

import ast
import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


selector = load("_v7_find_selector_test", HERE / "find_client_selector.py")
shared_path = (
    ROOT / "artifacts/jev-runtime/browser-find-text-profile-20261022-01/test_find_profile.py"
)
assert (
    hashlib.sha256(shared_path.read_bytes()).hexdigest()
    == "498c05819cb153c2d01717675847eb43899eab33cbc0f095942ad779e5c009ac"
)
shared = load("_v7_find_original_cpu_fixture", shared_path)
fixture = shared.fixture


@pytest.fixture
def runner(tmp_path):
    bundle, manifest = selector.prepare_bundle(tmp_path / "bundle", profile=selector.PROFILE)
    sys.path.insert(0, str(bundle))
    try:
        yield load("_v7_find_private_runner", bundle / "runner.py"), bundle, manifest
    finally:
        sys.path.remove(str(bundle))
        sys.modules.pop("find_client_selector", None)
        sys.modules.pop("_v7_find_private_runner", None)
        selector.verify_bundle(bundle, manifest)


class Client:
    def __init__(self, system):
        self.sent, self.attempts, self.query, self.system = (
            [],
            0,
            "an owned fixture section",
            system,
        )
        self.schema = {"json": json, "math": math, "IDENTITY": "generated-actions-none"}
        shared.functions(
            ROOT / "scripts/jev_qualify_public_field_syntax_bf16.py",
            ["require", "canonical", "strict_json", "exact", "keys", "parse_helper_request"],
            self.schema,
        )

    def post(self, url, **kwargs):
        body = copy.deepcopy(kwargs["json"])
        self.sent.append({"url": url, "request": body})
        status = 200
        if "questions" in body:
            self.attempts += 1
            if self.attempts == 1:
                status, response = 503, {"detail": "owned simulated retry"}
            else:
                answers = {}
                for key, question in body["questions"].items():
                    labels = list(question["criteria"])
                    choice = "FIND_TEXT" if key == "operation" else labels[0]
                    answers[key] = {
                        "choice": choice,
                        "confidence": 1.0,
                        "probabilities": {v: float(v == choice) for v in labels},
                    }
                response = {"model": "actual-fixture-identity", "answers": answers, "usage": {}}
        else:
            context = self.schema["parse_helper_request"](body, system_prompt=self.system)
            assert context["field"]["role"] == "browser_find_query"
            response = {
                "choices": [{"message": {"content": json.dumps({"text": self.query})}}],
                "usage": {},
            }
        return SimpleNamespace(
            status_code=status,
            is_error=status >= 400,
            json=lambda: copy.deepcopy(response),
            text=json.dumps(response),
        )


def arguments(group="original"):
    cases = selector.KNOWN4 if group == "original" else selector.PUBLIC6
    return [
        "--upstream-path",
        "owned-upstream",
        "--endpoint",
        "http://fixture.invalid/v1/systemone",
        "--model",
        "fixture-configured",
        "--output-dir",
        "owned-output",
        "--text-endpoint",
        "http://fixture.invalid",
        "--text-model",
        "generated-actions-none",
        "--chromium-executable",
        "owned-chromium",
        "--repeats",
        "1",
        "--max-steps",
        "16",
        "--seed",
        "20261014",
        "--suite-split",
        "development",
        "--observation-profile",
        selector.V7,
        "--observation-profile-helper-sha256",
        selector.V7_SHA,
        "--public-cases",
        *cases,
    ]


def test_explicit_dry_preparation(tmp_path, capsys):
    with pytest.raises(ValueError, match="Explicit opt-in"):
        selector.prepare_bundle(tmp_path / "refused")
    assert not (tmp_path / "refused").exists()
    selector.main([])
    assert json.loads(capsys.readouterr().out)["actual_model_or_browser_started"] is False


@pytest.mark.parametrize("group", ["original", "heldout"])
def test_original_V7_files_instrumentation_and_judges_preserved(tmp_path, group):
    bundle, manifest = selector.prepare_bundle(
        tmp_path / group, case_module=group, profile=selector.PROFILE
    )
    parent = json.loads((bundle / "v7-parent-manifest.json").read_bytes())
    for name, ref in parent["files"].items():
        if name != "runner.py":
            assert manifest["files"][name] == ref
    assert manifest["profile_helper_sha256"] == selector.V7_SHA
    assert manifest["fixed_case_ids"] == list(
        selector.KNOWN4 if group == "original" else selector.PUBLIC6
    )
    source = (bundle / "runner.py").read_bytes()
    insertion = selector.install(manifest["client_selection_sha256"], group)
    inverse = source.replace(insertion + selector.MAIN, selector.MAIN).replace(
        selector.SELECTED_CONSTRUCTOR, selector.CONSTRUCTOR
    )
    assert hashlib.sha256(inverse).hexdigest() == parent["files"]["runner.py"]["sha256"]
    names = {
        "model_adapter",
        "verify_fixture",
        "_task_success",
        "_premature_done",
    }

    def selected(data):
        return {
            n.name: ast.dump(n, include_attributes=False)
            for n in ast.parse(data).body
            if isinstance(n, ast.FunctionDef) and n.name in names
        }

    assert selected(inverse) == selected(source) and set(selected(source)) == names

    def verifier_import(data):
        return [
            ast.dump(n, include_attributes=False)
            for n in ast.walk(ast.parse(data))
            if isinstance(n, ast.ImportFrom)
            and any(a.name == "verify_public_web_case" for a in n.names)
        ]

    assert verifier_import(inverse) == verifier_import(source) and len(verifier_import(source)) == 1


def test_real_Agent_preserves_adapter_retry_text_records_redaction_and_restoration(runner, fixture):
    module, bundle, manifest = runner
    agent_module, model_module = fixture.module, fixture.model
    original_agent, original_choose = agent_module.Agent, agent_module.choose
    model_module.httpx = httpx
    client = model_module.CLIENT = Client(model_module.TEXT_VALUE)
    shared.functions(shared.profile.UPSTREAM / "model.py", ["post_json"], vars(model_module))
    fixture.browser.MARKER = "'owned-fixture-marker'"

    def call(self, method, **kwargs):
        assert method == "Runtime.evaluate" and client.query in kwargs["expression"]
        self.page.update(fingerprint="after-owned-find", marker="after-owned-marker")
        return {
            "result": {
                "value": {"status": "found", "scroll_requested": True, "scroll_changed": True}
            }
        }

    fixture.browser.Browser.call = call
    original_post, original_http = model_module.post_json, client.post
    before = {
        key: os.environ.get(key)
        for key in [
            "TYPESAFE_MODEL",
            "TYPESAFE_API_KEY",
            "TEXT_MODEL_BASE_URL",
            "TEXT_MODEL",
            "TEXT_MODEL_API_KEY",
        ]
    }
    with module.model_adapter(
        agent_module,
        model_module,
        endpoint="http://fixture.invalid/v1/systemone",
        model="fixture-configured",
        api_key="owned-policy-secret",
        text_endpoint="http://fixture.invalid",
        text_model="generated-actions-none",
        text_api_key="owned-text-secret",
    ) as (records, secrets):
        cls = module._selected_find_agent_class(agent_module, model_module)
        agent = cls("http://fixture.invalid/document", "Find the requested owned section")
        agent.state["page"]["title"] = "Owned owned-policy-secret owned-text-secret"
        result = agent.command("tick")
        assert result["history"][-1]["operation"] == "FIND_TEXT"
        assert result["find_text_feedback"][-1]["query"] == client.query
        assert len(result["history"]) == len(result["decisions"]) == len(result["text_calls"]) == 1
        assert [r["kind"] for r in records] == ["policy", "text"]
        assert [a["http_status"] for a in records[0]["transport_attempts"]] == [503, 200]
        assert records[1]["transport_attempts"][0]["http_status"] == 200
        assert records[0]["endpoint"] == "http://fixture.invalid/v1/systemone"
        assert records[1]["endpoint"] == "http://fixture.invalid/chat/completions"
        assert records[0]["request"] == module._redact(result["decisions"][0]["request"], secrets)
        assert records[0]["request"]["model"] == "fixture-configured"
        assert result["decisions"][0]["model"] == "actual-fixture-identity"
        assert "owned-policy-secret" not in json.dumps(
            records
        ) and "owned-text-secret" not in json.dumps(records)
        assert list(records[0]["request"]["questions"]) == [
            "operation",
            "click_target",
            "type_text_target",
            "select_target",
        ]
        assert len(records[0]["request"]["questions"]["operation"]["criteria"]) == 9
        assert agent_module.Agent is original_agent and agent_module.choose is original_choose
        agent.close()
    assert model_module.post_json is original_post and client.post == original_http
    assert {key: os.environ.get(key) for key in before} == before
    selector.verify_bundle(bundle, manifest)


def test_callback_only_helper_refused_before_adapter(runner, fixture):
    module, *_ = runner
    with pytest.raises(ValueError, match="explicit HTTP"):
        with module.model_adapter(
            fixture.module,
            fixture.model,
            endpoint="http://fixture.invalid",
            model="fixture",
            api_key="none",
            text_helper=lambda context: None,
        ):
            pytest.fail("callback-only profile admitted")
    assert not fixture.sent


@pytest.mark.parametrize("group", ["original", "heldout"])
def test_current_group_arguments_are_accepted_without_execution(group):
    result = selector.validate_arguments(arguments(group), group)
    assert result.max_steps == 16 and result.seed == 20261014


@pytest.mark.parametrize(
    "change",
    [
        "subset",
        "reorder",
        "steps",
        "seed",
        "profile",
        "helper_sha",
        "extra_task",
        "missing_http_helper",
    ],
)
def test_scope_or_budget_change_is_refused(change):
    args = arguments()
    if change == "subset":
        args.pop()
    elif change == "reorder":
        args[-1], args[-2] = args[-2], args[-1]
    elif change == "steps":
        args[args.index("--max-steps") + 1] = "17"
    elif change == "seed":
        args[args.index("--seed") + 1] = "1"
    elif change == "profile":
        args[args.index("--observation-profile") + 1] = "native-1231850"
    elif change == "helper_sha":
        args[args.index("--observation-profile-helper-sha256") + 1] = "0" * 64
    elif change == "extra_task":
        args += ["--tasks", "filter_open"]
    else:
        i = args.index("--text-endpoint")
        del args[i : i + 2]
    with pytest.raises((ValueError, SystemExit)):
        selector.validate_arguments(args, "original")


def test_private_main_validates_then_delegates_without_benchmark(runner, monkeypatch):
    module, *_ = runner
    seen = []
    module._original_v7_main = lambda: seen.append("original-main")
    monkeypatch.setattr(sys, "argv", ["runner.py", *arguments()])
    module.main()
    assert seen == ["original-main"]


def test_changed_bundle_is_refused(tmp_path):
    bundle, manifest = selector.prepare_bundle(tmp_path / "bundle", profile=selector.PROFILE)
    (bundle / "find_client_selector.py").write_bytes(b"changed\n")
    with pytest.raises(ValueError, match="bundle changed"):
        selector.verify_bundle(bundle, manifest)


def test_genuine_canonical_sixty_works_with_private_sixteen_and_no_global_changes(runner, tmp_path):
    """Actual load_upstream, constructor hook and command guards; no CDP/HTTP."""
    module, bundle, manifest = runner
    calls = []

    def no_cdp(*args, **kwargs):
        calls.append((args, kwargs))
        pytest.fail("Budget-only regression must not create a browser or issue CDP")

    parent_path = (
        ROOT
        / "artifacts/jev-runtime/browser-find-text-v7-client-integration-20261023-01/find_client_selector.py"
    )
    parent = load("_original_v7_budget_refusal", parent_path)
    old_bundle, old_manifest = parent.prepare_bundle(
        tmp_path / "old-parent", profile=parent.PROFILE
    )
    with module.load_upstream(
        ROOT / "artifacts/jev-runtime/upstream", SimpleNamespace(cdp=no_cdp)
    ) as (agent_module, model_module):
        original_agent, original_choose = agent_module.Agent, agent_module.choose
        assert agent_module.MAX_STEPS == 60
        assert original_agent.command.__globals__["MAX_STEPS"] == 60
        try:
            with pytest.raises(ValueError, match="sixteen-action cap"):
                parent.agent_class(
                    agent_module,
                    model_module,
                    selection_path=old_bundle / "find-client-selection.json",
                    selection_sha=old_manifest["client_selection_sha256"],
                )
            with module.model_adapter(
                agent_module,
                model_module,
                endpoint="http://fixture.invalid/v1/systemone",
                model="fixture-only",
                api_key="owned-fixture-only",
                text_endpoint="http://fixture.invalid",
                text_model="generated-actions-none",
                text_api_key="owned-fixture-only",
            ) as (records, _secrets):
                installed_choose = agent_module.choose
                cls = module._selected_find_agent_class(agent_module, model_module)
                assert issubclass(cls, original_agent)
                assert agent_module.MAX_STEPS == 60
                assert agent_module.Agent is original_agent
                assert agent_module.choose is installed_choose
                assert original_agent.command.__globals__["MAX_STEPS"] == 60
                selected = object.__new__(cls)
                selected.state = {
                    "decision": {"operation": "FIND_TEXT", "choice": "find_text", "target": None},
                    "page": {"fingerprint": "owned-marker"},
                    "history": [{} for _ in range(16)],
                    "status": "ready",
                }
                with pytest.raises(ValueError, match="sixteen-action demo budget"):
                    selected.command("act", {"fingerprint": "owned-marker"})
                assert selected.state["status"] == "blocked"
                assert selected.state["decision"] is None
                assert len(selected.state["history"]) == 16
                selected.state = {
                    "browser": SimpleNamespace(fresh=lambda _page: True),
                    "page": {"fingerprint": "owned-marker"},
                    "started_at": 0,
                    "status": "ready",
                    "decision": None,
                    "decisions": [{} for _ in range(32)],
                }
                with pytest.raises(ValueError, match="model-call budget"):
                    selected.command("predict")
                assert len(selected.state["decisions"]) == 32
                assert not records and not calls
                assert agent_module.MAX_STEPS == 60
            assert agent_module.choose is original_choose
            assert agent_module.Agent is original_agent and agent_module.MAX_STEPS == 60
        finally:
            model_module.CLIENT.close()
    selector.verify_bundle(bundle, manifest)
    parent.verify_bundle(old_bundle, old_manifest)


def test_unknown_internal_cap_refused_without_mutating_canonical(runner, fixture, monkeypatch):
    module, *_ = runner
    monkeypatch.setattr(fixture.module, "MAX_STEPS", 61)
    fixture.browser.MARKER = "'owned-marker'"
    before = (fixture.module.Agent, fixture.module.choose)
    with pytest.raises(ValueError, match="Known original sixty-action"):
        module._selected_find_agent_class(fixture.module, fixture.model)
    assert fixture.module.MAX_STEPS == 61
    assert before == (fixture.module.Agent, fixture.module.choose)
