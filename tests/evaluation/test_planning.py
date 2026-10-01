import copy
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from s1.benchmark import Case, fingerprint
from s1.evaluation.contracts import Budget, Capabilities, ModelSpec, ProfileSpec, SuiteSpec
from s1.evaluation.planning import case_eligibility, create_plan
from s1.evaluation.registry import Registry

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "benchmarks"


@pytest.fixture
def registry():
    return Registry.load(CONFIG / "models.toml", CONFIG / "suites.toml", CONFIG / "profiles.toml")


def plan(registry, **kwargs):
    return create_plan(registry, "smoke", lockfile=ROOT / "uv.lock", **kwargs)


def configured(registry, *, models=None, profile=None):
    return replace(
        registry,
        models=models if models is not None else registry.models,
        profiles={"smoke": profile if profile is not None else registry.profiles["smoke"]},
    )


def uniform(registry, **updates):
    data = registry.models["uniform"].model_dump()
    data.update(enabled=True, **updates)
    return ModelSpec.model_validate(data)


def profile(registry, **updates):
    data = registry.profiles["smoke"].model_dump()
    data.update(updates)
    return ProfileSpec.model_validate(data)


def test_disabled_inventory_is_not_a_zero_score(registry):
    result = create_plan(registry, "inventory", lockfile=ROOT / "uv.lock")
    assert result["status_counts"] == {"ready": 0, "blocked": 0, "not_run": 8}
    assert not result["can_execute"]
    assert result["budget"]["candidate_requests"] == 0
    for cell in result["models"]:
        assert "disabled" in cell["reasons"]
        assert "metrics" not in cell
        assert "accuracy" not in cell


def test_smoke_plan_counts_warmup_per_repetition(registry):
    result = plan(registry, enable_models=["uniform"])
    cell = result["models"][0]
    assert result["can_execute"]
    assert result["dry_run"]
    assert cell["coverage"]["requests"] == {"eligible": 3, "unsupported": 0, "unknown": 0}
    assert cell["coverage"]["decisions"]["eligible"] == 5
    assert cell["estimate"] == {
        "measured_requests": 9,
        "warmup_requests": 3,
        "total_requests": 12,
        "measured_decisions": 15,
        "cost_usd": 0,
    }


def test_media_unknown_and_unsupported_are_distinct(registry, benchmark_case):
    case = benchmark_case.model_copy(deep=True)
    data = case.model_dump()
    data["request"]["media"] = [{"type": "image", "data": "fixture"}]
    case = Case.model_validate(data)
    text = uniform(registry).model_copy(update={"capabilities": Capabilities(modalities=("text",))})
    unsupported = case_eligibility(text, case)
    assert unsupported["eligibility"] == "unsupported"
    assert "unsupported_modality:image" in unsupported["reasons"]
    unknown = case_eligibility(ModelSpec(id="pending", adapter="unimplemented"), case)
    assert unknown["eligibility"] == "unknown"
    assert "unknown_modality_support" in unknown["reasons"]


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"max_questions": 1}, "max_questions_exceeded"),
        ({"primitives": ("choice",)}, "unsupported_primitive:noul"),
        ({"modalities": ()}, "unsupported_modality:text"),
    ],
)
def test_request_capability_limits(registry, benchmark_case, changes, reason):
    model = uniform(registry)
    caps = model.capabilities.model_copy(update=changes)
    result = case_eligibility(model.model_copy(update={"capabilities": caps}), benchmark_case)
    assert result["eligibility"] == "unsupported"
    assert reason in result["reasons"]


def test_max_options_applies_to_actual_request(registry, benchmark_case):
    model = uniform(registry)
    assert case_eligibility(model, benchmark_case)["eligibility"] == "eligible"
    data = benchmark_case.model_dump()
    data["request"]["questions"][0]["criteria"]["other"] = "Other"
    model = model.model_copy(
        update={"capabilities": model.capabilities.model_copy(update={"max_options": 2})}
    )
    assert "max_options_exceeded" in case_eligibility(model, Case.model_validate(data))["reasons"]


@pytest.mark.parametrize(
    "capability,setting,reason",
    [
        (
            {"batch": "loop_emulated"},
            {"batch_size": 4, "require_native_batch": True},
            "native_batch_unavailable",
        ),
        ({"batch": "none"}, {"batch_size": 2}, "batch_unavailable"),
        ({"concurrent_requests": None}, {"concurrency": 2}, "concurrent_requests_unavailable"),
        (
            {"independent_questions": False},
            {"require_independent_questions": True},
            "independent_questions_unavailable",
        ),
        ({"probabilities": "none"}, {}, "probabilities_none"),
    ],
)
def test_profile_requirements_cannot_be_claimed_by_loop_or_unknown(
    registry, capability, setting, reason
):
    model = uniform(registry)
    model = model.model_copy(
        update={"capabilities": model.capabilities.model_copy(update=capability)}
    )
    result = plan(
        configured(registry, models={"uniform": model}, profile=profile(registry, **setting))
    )
    assert not result["can_execute"]
    assert reason in result["models"][0]["reasons"]


def test_partial_support_keeps_all_denominators(registry):
    model = uniform(registry)
    model = model.model_copy(
        update={
            "capabilities": model.capabilities.model_copy(update={"primitives": ("choice", "noul")})
        }
    )
    cell = plan(configured(registry, models={"uniform": model}))["models"][0]
    assert cell["coverage"]["requests"] == {"eligible": 2, "unsupported": 1, "unknown": 0}
    assert cell["coverage"]["decisions"] == {"eligible": 4, "unsupported": 1, "unknown": 0}
    assert cell["estimate"]["total_requests"] == 9


def test_request_budget_includes_warmup(registry):
    result = plan(
        configured(registry, profile=profile(registry, budget={"max_requests": 11})),
        enable_models=["uniform"],
    )
    assert not result["can_execute"]
    assert result["budget"]["candidate_requests"] == 12
    assert result["models"][0]["status"] == "blocked"
    assert "request_budget_exceeded" in result["models"][0]["reasons"]


def paid(registry, model_id, price=0.1):
    return ModelSpec(
        id=model_id,
        adapter="http",
        enabled=True,
        revision="provider-snapshot-1",
        precision="provider",
        execution_mode="provider",
        calibration="provider",
        endpoint_env="BENCH_ENDPOINT",
        token_env="BENCH_TOKEN",
        capabilities=registry.models["uniform"].capabilities,
        cost_per_request_usd=price,
    )


SERVICE = {"BENCH_ENDPOINT": "https://example.invalid/decide", "BENCH_TOKEN": "never-print-this"}


def test_cost_budget_is_aggregate_and_decimal_safe(registry):
    models = {name: paid(registry, name) for name in ("one", "two")}
    settings = dict(
        models=tuple(models), budget={"max_requests": 24, "max_estimated_cost_usd": 2.4}
    )
    exact = configured(registry, models=models, profile=profile(registry, **settings))
    result = plan(exact, environment=SERVICE)
    assert result["can_execute"]
    assert result["budget"]["candidate_cost_usd"] == 2.4
    settings["budget"]["max_estimated_cost_usd"] = 2.3
    result = plan(
        configured(registry, models=models, profile=profile(registry, **settings)),
        environment=SERVICE,
    )
    assert not result["can_execute"]
    assert all("cost_budget_exceeded" in c["reasons"] for c in result["models"])


@pytest.mark.parametrize(
    "price,reason", [(None, "unknown_cost_estimate"), (0.1, "cost_budget_exceeded")]
)
def test_unknown_cost_is_not_free(registry, price, reason):
    result = plan(
        configured(
            registry,
            models={"service": paid(registry, "service", price)},
            profile=profile(registry, models=("service",)),
        ),
        environment=SERVICE,
    )
    assert not result["can_execute"]
    assert reason in result["models"][0]["reasons"]


def test_secrets_and_request_contents_never_appear_in_plan(registry):
    current = configured(
        registry,
        models={"service": paid(registry, "service", 0)},
        profile=profile(registry, models=("service",)),
    )
    result = plan(current, environment=SERVICE)
    serialized = json.dumps(result)
    assert "never-print-this" not in serialized
    assert SERVICE["BENCH_ENDPOINT"] not in serialized
    assert "charged twice" not in serialized
    assert '"gold"' not in serialized
    assert "endpoint_sha256" in serialized
    missing = plan(current)
    assert "missing_credentials" in missing["models"][0]["reasons"]
    assert "missing_endpoint" in missing["models"][0]["reasons"]


@pytest.mark.parametrize(
    "url",
    [
        "file:///tmp/model",
        "https://user:secret@example.invalid/x",
        "https://example.invalid?token=secret",
        "https://[broken",
        "https://example.invalid:bad",
    ],
)
def test_invalid_endpoints_are_not_reflected_in_errors(registry, url):
    current = configured(
        registry,
        models={"service": paid(registry, "service", 0)},
        profile=profile(registry, models=("service",)),
    )
    result = plan(current, environment={**SERVICE, "BENCH_ENDPOINT": url})
    assert "invalid_endpoint" in result["models"][0]["reasons"]
    assert url not in json.dumps(result)


def test_unpinned_weights_remain_blocked_even_when_enabled(registry):
    result = create_plan(
        registry, "inventory", lockfile=ROOT / "uv.lock", enable_models=["gemma-g3"]
    )
    model = next(c for c in result["models"] if c["model_id"] == "gemma-g3")
    assert model["status"] == "blocked"
    assert "unresolved_model_revision" in model["reasons"]
    assert "unresolved_processor_revision" in model["reasons"]


def test_identity_is_stable_but_tracks_material_changes(registry, tmp_path):
    first = plan(registry, enable_models=["uniform"])
    repeated = plan(registry, enable_models=["uniform"], purpose="preflight")
    assert first["plan_id"] == repeated["plan_id"]
    altered = configured(registry, profile=profile(registry, seed=42))
    assert plan(altered, enable_models=["uniform"])["plan_id"] != first["plan_id"]
    lock = tmp_path / "changed.lock"
    lock.write_text("changed dependencies")
    changed = create_plan(registry, "smoke", lockfile=lock, enable_models=["uniform"])
    assert changed["plan_id"] != first["plan_id"]


def test_dataset_order_and_candidate_order_are_part_of_identity(registry, tmp_path):
    source = registry.dataset_path(registry.suites["smoke"])
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    path = tmp_path / "cases.jsonl"
    suite = registry.suites["smoke"].model_copy(update={"dataset": str(path)})
    current = replace(registry, suites={"smoke": suite})
    path.write_text("\n".join(json.dumps(row) for row in rows))
    before = plan(current)["dataset_sha256"]
    criteria = rows[0]["request"]["questions"]["team"]["criteria"]
    rows[0]["request"]["questions"]["team"]["criteria"] = dict(reversed(list(criteria.items())))
    path.write_text("\n".join(json.dumps(row) for row in rows))
    assert plan(current)["dataset_sha256"] != before
    rows.reverse()
    path.write_text("\n".join(json.dumps(row) for row in rows))
    assert plan(current)["dataset_sha256"] != before


def test_dataset_paths_are_relative_to_suite_file(registry, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert plan(registry, enable_models=["uniform"])["can_execute"]


def test_dataset_pin_and_split_checked_before_planning(registry, benchmark_case, tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(benchmark_case.model_dump_json())
    suite = registry.suites["smoke"].model_copy(
        update={"dataset": str(path), "dataset_sha256": "0" * 64}
    )
    current = replace(registry, suites={"smoke": suite})
    with pytest.raises(ValueError, match="fingerprint"):
        plan(current)
    suite = suite.model_copy(update={"dataset_sha256": fingerprint([benchmark_case])})
    current = replace(registry, suites={"smoke": suite})
    assert plan(current)["dataset_sha256"] == suite.dataset_sha256
    benchmark_case.split = "calibration"
    path.write_text(benchmark_case.model_dump_json())
    with pytest.raises(ValueError, match="test-only"):
        plan(current)


@pytest.mark.parametrize("kind", ["models", "suites", "profiles"])
def test_duplicate_registry_ids_rejected(tmp_path, kind):
    paths = {name: CONFIG / f"{name}.toml" for name in ("models", "suites", "profiles")}
    content = paths[kind].read_text()
    last_entry = content[content.rfind(f"[[{kind}]]") :]
    duplicate = tmp_path / f"{kind}.toml"
    duplicate.write_text(content + "\n" + last_entry)
    paths[kind] = duplicate
    with pytest.raises(ValueError, match="duplicate"):
        Registry.load(**paths)


def test_unknown_references_and_configuration_typos_rejected(tmp_path):
    original = (CONFIG / "profiles.toml").read_text()
    path = tmp_path / "profiles.toml"
    for content, message in (
        (original.replace('suite = "smoke"', 'suite = "absent"'), "unknown suite"),
        (original.replace('models = ["uniform"]', 'models = ["absent"]'), "unknown models"),
        (original.replace("warmup = 1", "warmupp = 1"), "Extra inputs"),
    ):
        path.write_text(content)
        with pytest.raises(ValueError, match=message):
            Registry.load(CONFIG / "models.toml", CONFIG / "suites.toml", path)


def test_invalid_numeric_settings_rejected():
    for data in (
        {"max_requests": -1},
        {"max_estimated_cost_usd": float("nan")},
        {"max_estimated_cost_usd": float("inf")},
    ):
        with pytest.raises(ValueError):
            Budget(**data)
    with pytest.raises(ValueError, match="unique"):
        ProfileSpec(id="p", suite="s", models=("one", "one"))
    with pytest.raises(ValueError, match="independent"):
        ModelSpec(
            id="m",
            adapter="gemma",
            execution_mode="causal_multislot",
            capabilities=Capabilities(independent_questions=True),
        )


def test_enable_override_does_not_mutate_registry(registry):
    before = copy.deepcopy(registry.models)
    plan(registry, enable_models=["uniform"])
    assert registry.models == before
    with pytest.raises(ValueError, match="selected"):
        plan(registry, enable_models=["typo"])


def test_configuration_flags_and_budgets_do_not_coerce_strings_or_booleans():
    with pytest.raises(ValueError):
        ModelSpec(id="m", adapter="uniform", enabled="yes")
    with pytest.raises(ValueError):
        Budget(max_requests=True)
    with pytest.raises(ValueError):
        Budget(max_estimated_cost_usd=False)
    with pytest.raises(ValueError):
        ProfileSpec(id="p", suite="s", models=("m",), warmup="1")


@pytest.mark.parametrize("track", ["native_multimodal", "pipeline"])
def test_track_requires_matching_information_view(track):
    with pytest.raises(ValueError, match="information view"):
        SuiteSpec(id="s", dataset="test.jsonl", track=track, information_view="raw_text")


def test_media_is_not_silently_classified_as_text(registry, benchmark_case, tmp_path):
    data = benchmark_case.model_dump()
    data["request"]["media"] = [{"type": "audio", "samples": [0.0]}]
    path = tmp_path / "media.jsonl"
    path.write_text(Case.model_validate(data).model_dump_json())
    suite = registry.suites["smoke"].model_copy(update={"dataset": str(path)})
    with pytest.raises(ValueError, match="native_media"):
        plan(replace(registry, suites={"smoke": suite}))


def test_cli_is_offline_and_writes_blocked_plan(tmp_path):
    output = tmp_path / "plan.json"
    # A socket connection or inference runtime import fails the subprocess test.
    script = """
import socket, sys
class DenyRuntime:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'transformers', 'laya', 'huggingface_hub'}:
            raise AssertionError('inference runtime imported')
sys.meta_path.insert(0, DenyRuntime())
def deny(*args, **kwargs):
    raise AssertionError('network attempted')
socket.socket.connect = deny
from s1.cli import main
raise SystemExit(main(sys.argv[1:]))
"""
    command = [
        sys.executable,
        "-c",
        script,
        "eval",
        "plan",
        "--profile",
        "smoke",
        "--output",
        str(output),
    ]
    blocked = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert blocked.returncode == 1, blocked.stderr
    assert json.loads(output.read_text())["models"][0]["status"] == "not_run"
    ready = subprocess.run(
        command + ["--enable-model", "uniform"], cwd=ROOT, capture_output=True, text=True
    )
    assert ready.returncode == 0, ready.stderr
    assert json.loads(output.read_text())["can_execute"]
