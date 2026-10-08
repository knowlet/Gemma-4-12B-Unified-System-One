"""Offline campaign audit rejects score forgery and unsupported cache claims."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "audit_public_campaign", ROOT / "scripts/summarize_jevbench_campaign.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def refresh_hashes(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["artifact_sha256"] = {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    write_json(directory / "manifest.json", manifest)


@pytest.fixture
def evidence(audit, tmp_path):
    tasks = [
        SimpleNamespace(
            id=f"task-{i}",
            family="policy",
            split="public",
            group=f"source-{i // 2}",
            state=f"state-{i}",
            question={"type": "noul", "instructions": "Allow?"},
            labels=["no", "yes"],
            expected="yes",
            provenance={},
        )
        for i in range(4)
    ]

    def score(probs, task):
        predicted = "yes" if probs["yes"] > probs["no"] else "no"
        return {
            "valid": True,
            "strict_valid": True,
            "renormalized": False,
            "probs": probs,
            "predicted": predicted,
            "correct": predicted == task.expected,
        }

    def metric(subset, records):
        rs = [r for r in records if r["task_id"] in {task.id for task in subset}]
        return {
            "n_planned": len(subset),
            "n_attempted": len(rs),
            "n_valid": sum(r["valid"] for r in rs),
            "n_correct": sum(r["correct"] for r in rs),
            "accuracy": sum(r["correct"] for r in rs) / len(rs) if rs else None,
            "brier_mean": 0.1,
            "ece": {"ece": 0.1, "n": len(rs)},
            "ordinal_mae": None,
            "schema_validity": 1.0,
            "schema_validity_strict": 1.0,
            "n_renormalized": 0,
            "latency": {"n": len(rs), "p50_s": 1.0, "p95_s": 1.0},
        }

    def summarize(subset, records):
        out = metric(subset, records)
        return {
            **out,
            "macro_accuracy": out["accuracy"],
            "per_family": {"policy": out},
            "complete": len(subset) == len(records),
            "price_per_1000_decisions_usd": None,
        }

    official = SimpleNamespace(
        base=SimpleNamespace(build_question=lambda task: copy.deepcopy(task.question)),
        scoring=SimpleNamespace(score_task=score),
        summarize=SimpleNamespace(summarize=summarize, metric=metric),
    )
    config = {
        "dataset_label": "fixture-public",
        "canonical_dataset_sha256": "a" * 64,
        "public_exposure": "public fixture, not sealed",
    }
    root = tmp_path / "campaign"
    root.mkdir()
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(audit, "load_public", lambda *_: (tasks, official, config))

    def create(name, p_yes=0.8, *, cached=False):
        base = root / name
        base.mkdir(exist_ok=True)
        directory = base / ("cached-public231" if cached else "public231")
        directory.mkdir()
        (directory / "raw").mkdir()
        identity = {
            "model_id": name,
            "revision": "b" * 40,
            "gpu": "same A100",
            "setup_seconds": 8.0,
        }
        if cached:
            identity["candidate_cache"] = True
        records = []
        raw_answers = []
        for i, task in enumerate(tasks):
            request = {"state": task.state, "questions": {"decision": task.question}}
            response = {"answers": {"decision": {"type": "noul", "noul": p_yes}}}
            raw_name = f"raw/{i}.json"
            write_json(
                directory / raw_name,
                {
                    "request": request,
                    "response": response,
                    "response_encoding": "json",
                    "error": None,
                },
            )
            probs = {"yes": p_yes, "no": 1 - p_yes}
            outcome = score(probs, task)
            records.append(
                {
                    "task_id": task.id,
                    "family": task.family,
                    "split": task.split,
                    "group": task.group,
                    "task_index": i,
                    "model": name,
                    "latency_s": 1.0,
                    "request_sha256": audit._hash(audit._bytes(request)),
                    "raw_path": raw_name,
                    "raw_sha256": audit._hash((directory / raw_name).read_bytes()),
                    "probs_as_returned": probs,
                    "status": "ok",
                    "ok": True,
                    "error": None,
                    "official_task_outcome": outcome,
                    "cost_usd": None,
                    **outcome,
                }
            )
            raw_answers.append(response["answers"])
        (directory / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
        summary = {
            **summarize(tasks, records),
            "identity": identity,
            "official_rank": None,
            "official_composite": None,
            "sealed_evaluation": "not_run",
            "leaderboard_status": "not_submitted",
        }
        write_json(directory / "summary.json", summary)
        write_json(
            directory / "manifest.json",
            {
                "status": "completed",
                "dataset": config,
                "ordered_task_ids": [task.id for task in tasks],
                "identity": identity,
                "n_attempted": len(tasks),
                "n_valid": len(tasks),
            },
        )
        refresh_hashes(directory)
        return directory, summary, records, identity

    yield SimpleNamespace(root=root, tasks=tasks, official=official, config=config, create=create)
    monkeypatch.undo()


def test_hash_mutation_excludes_model_from_pairing(audit, evidence):
    directory, *_ = evidence.create("a")
    evidence.create("b")
    with (directory / "raw/0.json").open("a") as stream:
        stream.write(" ")
    report = audit.summarize_campaign(
        evidence.root, "unused", model_names=("a", "b"), bootstrap_samples=100
    )
    assert report["models"]["a"]["status"] == "integrity_failed"
    assert report["paired_accuracy"][0]["status"] == "inconclusive"
    assert report["official_rank"] is None


def test_score_forgery_fails_even_after_refreshing_artifact_hashes(audit, evidence):
    directory, *_ = evidence.create("a")
    rows = [json.loads(line) for line in (directory / "records.jsonl").read_text().splitlines()]
    rows[0]["correct"] = False
    rows[0]["official_task_outcome"]["correct"] = False
    (directory / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    refresh_hashes(directory)
    with pytest.raises(audit.EvidenceError, match="official outcome correct"):
        audit.audit_public(directory, evidence.tasks, evidence.official, evidence.config)


@pytest.mark.parametrize("change", ["missing", "duplicate", "reorder", "request"])
def test_population_and_request_integrity_survive_refreshed_hashes(audit, evidence, change):
    directory, *_ = evidence.create("a")
    rows = [json.loads(line) for line in (directory / "records.jsonl").read_text().splitlines()]
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[1] = rows[0]
    elif change == "reorder":
        rows.reverse()
    else:
        rows[0]["request_sha256"] = "c" * 64
    (directory / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    refresh_hashes(directory)
    with pytest.raises(audit.EvidenceError):
        audit.audit_public(directory, evidence.tasks, evidence.official, evidence.config)


def test_summary_recomputed_not_accepted_from_headers(audit, evidence):
    directory, summary, *_ = evidence.create("a")
    summary["accuracy"] = 0.25
    write_json(directory / "summary.json", summary)
    refresh_hashes(directory)
    with pytest.raises(audit.EvidenceError, match="summary field differs: accuracy"):
        audit.audit_public(directory, evidence.tasks, evidence.official, evidence.config)


def test_portable_aggregate_roundoff_is_bounded_but_population_counters_exact(audit, evidence):
    directory, summary, *_ = evidence.create("a")
    summary["brier_mean"] += 1e-16
    write_json(directory / "summary.json", summary)
    refresh_hashes(directory)
    audit.audit_public(directory, evidence.tasks, evidence.official, evidence.config)
    summary["n_correct"] = 4.0
    write_json(directory / "summary.json", summary)
    refresh_hashes(directory)
    with pytest.raises(audit.EvidenceError, match="n_correct"):
        audit.audit_public(directory, evidence.tasks, evidence.official, evidence.config)


def test_valid_campaign_reports_missing_models_setup_and_unknown_phase_memory(audit, evidence):
    directory, summary, _, identity = evidence.create("a")
    write_json(
        directory.parent / "receipt.json",
        {
            "identity": identity,
            "public": summary,
            "coherence": None,
            "memory_peak_allocated_bytes": 123,
        },
    )
    report = audit.summarize_campaign(
        evidence.root, "unused", model_names=("a", "missing"), bootstrap_samples=100
    )
    a = report["models"]["a"]
    assert a["status"] == "verified" and a["public"]["n_attempted"] == 4
    assert a["setup_seconds"] == 8.0 and a["memory"]["phase_peaks"] is None
    assert a["memory"]["scope"].startswith("unknown")
    assert a["cost_usd"] is None
    assert report["models"]["missing"]["status"] == "not_run"
    assert report["status"] == "incomplete_or_failed_public_audit"


def test_paired_interval_preserves_source_group_cluster(audit):
    tasks = [
        SimpleNamespace(id=str(i), group="one shared source", expected="yes", provenance={})
        for i in range(4)
    ]
    left = [{"task_id": str(i), "request_sha256": str(i), "correct": False} for i in range(4)]
    right = [{**row, "correct": i < 2} for i, row in enumerate(left)]
    pair = audit.paired_accuracy(left, right, tasks, samples=100)
    assert pair["source_groups"] == 1 and pair["delta_accuracy"] == 0.5
    assert pair["ci95"] == [0.5, 0.5]


def test_hardware_mismatch_keeps_observed_quality_pair_and_records_confound(audit, evidence):
    evidence.create("a")
    directory, summary, _, identity = evidence.create("b")
    identity["gpu"] = "different GPU"
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["identity"] = identity
    write_json(directory / "manifest.json", manifest)
    summary["identity"] = identity
    write_json(directory / "summary.json", summary)
    refresh_hashes(directory)
    report = audit.summarize_campaign(
        evidence.root, "unused", model_names=("a", "b"), bootstrap_samples=100
    )
    pair = report["paired_accuracy"][0]
    assert pair["status"] == "computed" and pair["delta_accuracy"] == 0
    assert pair["hardware"] == {
        "reference_gpu": "same A100",
        "candidate_gpu": "different GPU",
        "same_gpu_label": False,
    }
    assert "gpu" in pair["configuration_differences"]
    assert "does not establish a causal change or latency speedup" in pair["comparison_scope"]


def test_unknown_hardware_does_not_suppress_valid_paired_quality(audit, evidence):
    directory, summary, _, identity = evidence.create("a")
    evidence.create("b")
    identity.pop("gpu")
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest["identity"] = identity
    write_json(directory / "manifest.json", manifest)
    summary["identity"] = identity
    write_json(directory / "summary.json", summary)
    refresh_hashes(directory)
    report = audit.summarize_campaign(
        evidence.root, "unused", model_names=("a", "b"), bootstrap_samples=100
    )
    pair = report["paired_accuracy"][0]
    assert pair["status"] == "computed"
    assert pair["hardware"]["reference_gpu"] is None and not pair["hardware"]["same_gpu_label"]


def test_cache_requires_all_valid_exact_distributions_not_only_same_labels(audit, evidence):
    baseline_path, *_ = evidence.create("s1-bf16", 0.8)
    cached_path, *_ = evidence.create("s1-bf16", 0.7, cached=True)
    baseline = audit.audit_public(baseline_path, evidence.tasks, evidence.official, evidence.config)
    cached = audit.audit_public(cached_path, evidence.tasks, evidence.official, evidence.config)
    comparison = audit.cache_equivalence(baseline, cached)
    assert comparison["exact_labels"] is True and comparison["exact_probabilities"] is False
    assert comparison["status"] == "failed" and comparison[
        "max_probability_delta"
    ] == pytest.approx(0.1)


def test_explicit_baseline_cache_disabled_identity_is_accepted(audit, evidence):
    baseline_path, *_ = evidence.create("s1-bf16", 0.8)
    cached_path, *_ = evidence.create("s1-bf16", 0.8, cached=True)
    baseline = audit.audit_public(baseline_path, evidence.tasks, evidence.official, evidence.config)
    cached = audit.audit_public(cached_path, evidence.tasks, evidence.official, evidence.config)
    baseline["manifest"]["identity"]["candidate_cache"] = False
    assert audit.cache_equivalence(baseline, cached)["status"] == "passed"


def test_new_usage_records_cannot_claim_counts_absent_from_raw_answer(audit, evidence):
    directory, *_ = evidence.create("a")
    rows = [json.loads(line) for line in (directory / "records.jsonl").read_text().splitlines()]
    rows[0]["usage_status"] = "reported_unverified"
    rows[0]["usage"] = {"input_tokens": 0}
    (directory / "records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    refresh_hashes(directory)
    with pytest.raises(audit.EvidenceError, match="usage differs"):
        audit.audit_public(directory, evidence.tasks, evidence.official, evidence.config)


def sample_rows(tasks, baseline, ratios):
    rows = []
    for index, task in enumerate(tasks):
        for cached in (False, True, True, False):
            for repeat in range(3):
                rows.append(
                    {
                        "task_id": task.id,
                        "cached": cached,
                        "repeat": repeat,
                        "seconds": ratios[index] if cached else 1.0,
                        "answers": baseline["raw_answers"][index],
                    }
                )
    return rows


def test_timing_interval_crossing_one_prevents_speedup_claim(audit, monkeypatch, tmp_path):
    monkeypatch.setattr(audit, "SAMPLE_INDICES", tuple(range(8)))
    tasks = [SimpleNamespace(id=str(i), group=None) for i in range(8)]
    baseline = {"raw_answers": [{"decision": {"type": "noul", "noul": 0.8}}] * 8}
    path = tmp_path / "samples.json"
    write_json(path, sample_rows(tasks, baseline, [0.5] * 4 + [1.5] * 4))
    timing = audit.cache_timing(path, baseline, tasks, samples=1000)
    assert timing["ratio_ci95"][0] < 1 < timing["ratio_ci95"][1]
    assert timing["timing_supports_reduction_on_sample"] is False


@pytest.mark.parametrize("change", ["drop", "reorder", "output"])
def test_cache_sample_schedule_and_output_integrity(audit, monkeypatch, tmp_path, change):
    monkeypatch.setattr(audit, "SAMPLE_INDICES", tuple(range(8)))
    tasks = [SimpleNamespace(id=str(i), group=None) for i in range(8)]
    baseline = {"raw_answers": [{"decision": {"type": "noul", "noul": 0.8}}] * 8}
    rows = sample_rows(tasks, baseline, [0.8] * 8)
    if change == "drop":
        rows.pop()
    elif change == "reorder":
        rows[0], rows[3] = rows[3], rows[0]
    else:
        rows[0]["answers"] = {"decision": {"type": "noul", "noul": 0.7}}
    path = tmp_path / "samples.json"
    write_json(path, rows)
    with pytest.raises(audit.EvidenceError):
        audit.cache_timing(path, baseline, tasks, samples=100)


def test_saved_coherence_error_and_coverage_disclosed_without_accuracy_claim(audit, tmp_path):
    path = tmp_path / "coherence.json"
    scores = {
        "overall": None,
        "pillars": {"BAT": None},
        "ci": {},
        "n_cases": 4,
        "n_checks": 1,
        "incomplete": ["batch"],
        "checks": {"batch/x": {"coverage": 0.5}},
    }
    write_json(
        path,
        {
            "scores": scores,
            "result": {"errors": [{"error": "refused"}], "suite": {"name": "mini"}},
            "settings": {"min_coverage": 0.95},
        },
    )
    result = audit.coherence_summary(path, None)
    assert result["status"] == "saved_report_only_not_recomputed"
    assert result["overall"] is None and result["errors"] == 1
    assert result["check_coverage"] == {"batch/x": 0.5}
    assert "accuracy" not in result


def test_cli_never_overwrites_old_reports(audit, monkeypatch, tmp_path):
    output = tmp_path / "report.json"
    output.write_text("old evidence")
    monkeypatch.setattr(
        audit, "summarize_campaign", lambda *args, **kwargs: {"status": "complete_public_audit"}
    )
    with pytest.raises(FileExistsError):
        audit.main(["campaign", "--tasks-root", "repo", "--output", str(output)])
    assert output.read_text() == "old evidence"


def test_combined_model_roots_preserve_run_ids_and_expected_compiled_omission(
    audit, evidence, tmp_path
):
    first, *_ = evidence.create("a")
    second, *_ = evidence.create("s1-compiled")
    run01, run02 = tmp_path / "run01", tmp_path / "run02"
    run01.mkdir()
    run02.mkdir()
    first.parent.rename(run01 / "a")
    second.parent.rename(run02 / "s1-compiled")
    roots = {
        "a": run01 / "a",
        "s1-compiled": run02 / "s1-compiled",
        "jev-omni": tmp_path / "run03/jev-omni",
    }
    report = audit.summarize_campaign(
        tmp_path / "combined",
        "unused",
        model_names=tuple(roots),
        model_roots=roots,
        bootstrap_samples=100,
    )
    assert report["models"]["a"]["source_campaign"] == "run01"
    assert report["models"]["s1-compiled"]["source_campaign"] == "run02"
    assert report["models"]["s1-compiled"]["coherence"]["status"] == "expected_not_run"
    assert report["models"]["a"]["coherence"]["status"] == "not_run"
    assert report["models"]["jev-omni"]["source_campaign"] == "run03"
    assert report["models"]["jev-omni"]["status"] == "not_run"
    assert report["paired_accuracy"][0]["status"] == "computed"
    assert report["coherence_audit"] == "incomplete_or_unrecomputed_coherence_audit"


def test_cli_accepts_all_six_named_profiles_without_symlink_projection(
    audit, monkeypatch, tmp_path
):
    supplied = {}

    def summarize(*args, **kwargs):
        supplied.update(kwargs)
        return {"status": "complete_public_audit", "coherence_audit": "complete_raw_replay_audit"}

    monkeypatch.setattr(audit, "summarize_campaign", summarize)
    names = ("s1-bf16", "onejev-4b", "decider-2b", "s1-independent", "s1-compiled", "jev-omni")
    overrides = [arg for name in names for arg in ("--model-root", f"{name}={tmp_path / name}")]
    assert (
        audit.main(
            ["combined", "--tasks-root", "repo", "--coherence-root", "coherence", *overrides]
        )
        == 0
    )
    assert supplied["model_names"] == names
    assert set(supplied["model_roots"]) == set(names)
    assert supplied["coherence_root"] == Path("coherence")


def test_cli_requested_coherence_failure_has_nonzero_exit_despite_valid_public(audit, monkeypatch):
    monkeypatch.setattr(
        audit,
        "summarize_campaign",
        lambda *args, **kwargs: {
            "status": "complete_public_audit",
            "coherence_audit": "incomplete_or_unrecomputed_coherence_audit",
        },
    )
    assert audit.main(["combined", "--tasks-root", "repo", "--coherence-root", "coherence"]) == 1


@pytest.fixture
def coherence_replay(audit, monkeypatch, tmp_path):
    """Small frozen package exercises isolated replay, not the publisher metric itself."""
    source = tmp_path / "source"
    package = source / "src/jevbench"
    package.mkdir(parents=True)
    package.joinpath("__init__.py").write_text("""
from types import SimpleNamespace
def from_callable(fn, name):
    return SimpleNamespace(fn=fn, name=name)
def evaluate(model, **kwargs):
    answers = []
    for i in range(1248):
        answer = model.fn(str(i), {"q": {"type": "noul", "instructions": "Allow?"}})
        assert answer["q"]["type"] == "noul"
        answers.append(answer["q"]["noul"])
    return SimpleNamespace(scores={"overall": sum(answers)/1248, "n_cases": 240, "n_checks": 50},
        result={"backend": model.name, "seed": 0, "repeats": 1, "trials": 1,
                "suite": {"name": "jevbench-mini"},
                "selection": {"domains": None, "dimensions": None, "groups": None,
                              "relations": None, "diagnostics": False}, "bases": answers})
""")
    subprocess.run(["git", "init", "-q", str(source)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(source), "add", "src"], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "Frozen fixture",
        ],
        check=True,
        capture_output=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    monkeypatch.setattr(audit, "COHERENCE_REVISION", revision)
    cache = tmp_path / "coherence-cache"
    cache.mkdir()
    backend, value = "fixture@revision", 0.8
    for i in range(1248):
        blob = json.dumps(
            {
                "b": backend,
                "s": str(i),
                "q": {"q": {"type": "noul", "instructions": "Allow?"}},
                "r": 0,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        filename = hashlib.sha256(blob.encode()).hexdigest() + ".json"
        write_json(cache / filename, {"q": {"type": "noul", "noul": value}})
    report = {
        "settings": audit.COHERENCE_SETTINGS,
        "scores": {"overall": sum([value] * 1248) / 1248, "n_cases": 240, "n_checks": 50},
        "result": {
            "backend": backend,
            "seed": 0,
            "repeats": 1,
            "trials": 1,
            "suite": {"name": "jevbench-mini"},
            "selection": {
                "domains": None,
                "dimensions": None,
                "groups": None,
                "relations": None,
                "diagnostics": False,
            },
            "bases": [value] * 1248,
        },
    }
    return SimpleNamespace(source=source, package=package, cache=cache, report=report)


def test_coherence_replays_all_native_answers_and_preserves_original_bytes(audit, coherence_replay):
    c = coherence_replay
    before = {p.name: p.read_bytes() for p in c.cache.iterdir()}
    replay = audit.replay_coherence(c.report, c.cache, c.source)
    assert replay["raw_cache_files"] == 1248
    assert replay["source"]["revision"] == audit.COHERENCE_REVISION
    assert replay["raw_cache_hash_scope"].startswith("observed audit bytes")
    assert before == {p.name: p.read_bytes() for p in c.cache.iterdir()}


def test_coherence_changed_native_answer_fails_even_when_saved_scores_unchanged(
    audit, coherence_replay
):
    c = coherence_replay
    write_json(next(c.cache.iterdir()), {"q": {"type": "noul", "noul": 0.2}})
    with pytest.raises(audit.EvidenceError, match="scores differ"):
        audit.replay_coherence(c.report, c.cache, c.source)


def test_coherence_saved_outcome_forgery_fails_even_with_matching_scores(audit, coherence_replay):
    c = coherence_replay
    c.report["result"]["bases"][0] = 0.2
    with pytest.raises(audit.EvidenceError, match="outcome field.*bases"):
        audit.replay_coherence(c.report, c.cache, c.source)


def test_coherence_raw_probability_equality_has_no_aggregate_tolerance(audit, coherence_replay):
    c = coherence_replay
    c.report["result"]["bases"][0] += 1e-14
    with pytest.raises(audit.EvidenceError, match="outcome field.*bases"):
        audit.replay_coherence(c.report, c.cache, c.source)


def test_coherence_audit_environment_is_telemetry_not_a_metric(audit, coherence_replay):
    c = coherence_replay
    c.report["scores"]["environment"] = {
        "python": "original-runtime",
        "torch": "original-gpu-runtime",
    }
    replay = audit.replay_coherence(c.report, c.cache, c.source)
    assert replay["original_environment"]["torch"] == "original-gpu-runtime"
    assert replay["audit_environment"] is None


@pytest.mark.parametrize("change", ["missing", "substituted"])
def test_coherence_missing_or_wrong_request_hash_is_not_skipped(audit, coherence_replay, change):
    c = coherence_replay
    path = next(c.cache.iterdir())
    if change == "missing":
        path.unlink()
    else:
        path.rename(c.cache / ("f" * 64 + ".json"))
    with pytest.raises(
        audit.EvidenceError, match="1248|lack original answers|missing original cached answer"
    ):
        audit.replay_coherence(c.report, c.cache, c.source)


@pytest.mark.parametrize("change", ["modified", "untracked"])
def test_coherence_source_pin_rejects_changed_or_extra_importable_code(
    audit, coherence_replay, tmp_path, change
):
    c = coherence_replay
    if change == "modified":
        with (c.package / "__init__.py").open("a") as stream:
            stream.write("\n# mutation\n")
    else:
        (c.package / "extra.py").write_text("raise RuntimeError('untracked')")
    with pytest.raises(audit.EvidenceError, match="source differs|untracked"):
        audit._freeze_coherence_source(c.source, tmp_path / "copy")
