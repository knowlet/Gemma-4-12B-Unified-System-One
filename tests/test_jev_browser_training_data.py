"""Supervised development trajectories stay outside benchmark policy inputs."""

import importlib.util
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    SPEC = importlib.util.spec_from_file_location(
        "jev_browser_training_data", ROOT / "scripts/jev_browser_training_data.py"
    )
    COLLECTOR = importlib.util.module_from_spec(SPEC)
    sys.modules[SPEC.name] = COLLECTOR
    SPEC.loader.exec_module(COLLECTOR)
finally:
    sys.path.pop(0)


def test_supervision_excludes_unused_target_heads_and_preserves_original_inputs():
    payload = {
        "model": "development-oracle",
        "state": {"page": {"text": "Search pending"}, "elements": []},
        "questions": {
            "operation": {
                "instructions": {"goal": "Find item"},
                "criteria": {"CLICK": "Click", "SELECT": "Select", "DONE": "Finished"},
            },
            "click_target": {
                "instructions": {"goal": "Find item", "operation": "CLICK"},
                "criteria": {"1": {"element": "Enable filter"}, "2": {"element": "Open result"}},
            },
            "select_target": {
                "instructions": {"goal": "Find item", "operation": "SELECT"},
                "criteria": {"3:1": "Choose region", "3:2": "Other region"},
            },
        },
    }
    before = json.loads(json.dumps(payload))
    row = COLLECTOR.training_case(
        payload,
        case_id="train-step",
        group_id="train-episode",
        task_id="filter_open",
        split="train",
        selected={"operation": "CLICK", "target": "1"},
        valid_targets={"CLICK": ["1"], "SELECT": ["3:1"]},
    )
    assert payload == before
    assert row.request.state == payload["state"]
    assert [question.id for question in row.request.questions] == ["operation", "click_target"]
    assert row.gold == {"operation": "CLICK", "click_target": "1"}
    assert row.soft_gold["operation"] == {"CLICK": 0.5, "SELECT": 0.5, "DONE": 0.0}
    assert row.soft_gold["click_target"] == {"1": 1.0, "2": 0.0}
    assert (
        json.loads(row.request.questions[0].instructions)
        == payload["questions"]["operation"]["instructions"]
    )
    serialized = row.request.model_dump_json()
    assert "gold" not in serialized and "train-episode" not in serialized


def test_collector_rejects_holdout_and_invalid_configuration_before_writing(tmp_path):
    for split, repeats in [("holdout", 1), ("train", 0), ("test", 1)]:
        with pytest.raises(ValueError, match="explicit fitting"):
            COLLECTOR.collect(
                upstream_path="unused", output_dir=tmp_path / split, split=split, repeats=repeats
            )
        assert not (tmp_path / split).exists()


@pytest.mark.parametrize(
    "options",
    [
        {"task_ids": []},
        {"task_ids": ["upstream_travel", "upstream_travel"]},
        {"task_ids": ["not_a_task"]},
        {"task_ids": ["upstream_travel"], "split": "development"},
        {"travel_recovery_mode": "unknown"},
        {"travel_recovery_mode": "partial", "split": "development"},
    ],
)
def test_collector_rejects_invalid_partial_recovery_options_before_sources(tmp_path, options):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        COLLECTOR.collect(upstream_path="unused", output_dir=output, **options)
    assert not output.exists()


def isolated_collection(monkeypatch, tmp_path, *, during_generation=None):
    """Exercise actual source capture/finalization without any browser or model."""
    source = tmp_path / "caller" / "jev_browser_training_data.py"
    source.parent.mkdir()
    source.write_bytes(Path(COLLECTOR.__file__).read_bytes())
    monkeypatch.setattr(COLLECTOR, "__file__", str(source))
    monkeypatch.setattr(COLLECTOR, "upstream_manifest", lambda _path: {"revision": "unit"})

    def cases(*_args):
        assert (
            tmp_path / "run" / "sources" / "scripts" / source.name
        ).read_bytes() == source.read_bytes()
        if during_generation:
            during_generation(source)
        return []

    monkeypatch.setattr(COLLECTOR, "make_cases", cases)
    # The capture must name the actual imported harness file rather than this
    # test's monkeypatched generator code filename.
    cases.__code__ = cases.__code__.replace(
        co_filename=str(ROOT / "scripts" / "jev_browser_benchmark.py")
    )
    monkeypatch.setattr(COLLECTOR, "fixture_server", lambda _cases: nullcontext("http://unit"))
    monkeypatch.setattr(COLLECTOR, "load_upstream", lambda *_args: nullcontext((None, None)))
    browser = SimpleNamespace(close=lambda: None)
    runtime = SimpleNamespace(chromium=SimpleNamespace(launch=lambda **_kwargs: browser))
    monkeypatch.setitem(
        sys.modules,
        "playwright.sync_api",
        SimpleNamespace(sync_playwright=lambda: nullcontext(runtime)),
    )
    return source, tmp_path / "run"


def test_collector_source_drift_writes_failed_receipt_and_raises(monkeypatch, tmp_path):
    source, output = isolated_collection(
        monkeypatch,
        tmp_path,
        during_generation=lambda path: path.write_bytes(path.read_bytes() + b"\n# changed\n"),
    )
    monkeypatch.setenv("TYPESAFE_API_KEY", "unit-key-must-be-restored")
    with pytest.raises(RuntimeError, match="runtime source changed"):
        COLLECTOR.collect(upstream_path="unused", output_dir=output, repeats=1)
    receipt = json.loads((output / "receipt.json").read_bytes())
    assert receipt["stage"] == "failed"
    assert (
        receipt["source_preservation_timing"] == "before fixture generation and browser collection"
    )
    assert receipt["source_verification"]["changed"] == ["scripts/jev_browser_training_data.py"]
    assert (output / "sources" / "scripts" / source.name).read_bytes() != source.read_bytes()
    assert (output / "cases.jsonl").exists() and (output / "wire_requests.jsonl").exists()
    assert os.environ["TYPESAFE_API_KEY"] == "unit-key-must-be-restored"
    assert "unit-key-must-be-restored" not in (output / "receipt.json").read_text()


def test_collector_keeps_primary_failure_when_source_also_changes(monkeypatch, tmp_path):
    primary = LookupError("trajectory failed first")

    def fail(path):
        path.write_bytes(path.read_bytes() + b"\n# changed\n")
        raise primary

    _source, output = isolated_collection(monkeypatch, tmp_path, during_generation=fail)
    with pytest.raises(LookupError) as caught:
        COLLECTOR.collect(upstream_path="unused", output_dir=output, repeats=1)
    assert caught.value is primary
    receipt = json.loads((output / "receipt.json").read_bytes())
    assert receipt["error"]["type"] == "LookupError"
    assert receipt["stage"] == "failed" and not receipt["source_verification"]["success"]


@pytest.mark.integration
@pytest.mark.parametrize("split,expected_episodes", [("train", 11), ("calibration", 9)])
def test_real_training_trajectories_use_guarded_actions_and_fresh_final_dom(
    tmp_path, split, expected_episodes
):
    if os.environ.get("JEV_BROWSER_TESTS") != "1":
        pytest.skip("Set JEV_BROWSER_TESTS=1 for guarded Chromium collection")
    pytest.importorskip("playwright.sync_api")
    from s1.evaluation.datasets import load_cases

    upstream = Path(os.environ.get("JEV_UPSTREAM_PATH", "/tmp/s1-jev-ultrafast"))
    before = COLLECTOR.upstream_manifest(upstream)
    output = tmp_path / split
    receipt = COLLECTOR.collect(
        upstream_path=upstream, output_dir=output, split=split, seed=91023, repeats=1
    )
    assert receipt["stage"] == "complete"
    assert receipt["source_verification"]["success"]
    assert receipt["upstream_verification"]["success"]
    assert (
        receipt["source_manifest"]["files"]["scripts/jev_browser_training_data.py"]["sha256"]
        == receipt["collector_sha256"]
    )
    assert len(receipt["episodes"]) == expected_episodes
    assert all(episode["verified_dom"]["success"] for episode in receipt["episodes"])
    assert COLLECTOR.upstream_manifest(upstream) == before
    rows = load_cases(output / "cases.jsonl")
    wires = [json.loads(line) for line in (output / "wire_requests.jsonl").read_text().splitlines()]
    assert len(rows) == len(wires) == receipt["training_cases"]
    assert all(row.split == split for row in rows)
    assert all(
        set(row.gold) <= {"operation", row.gold["operation"].lower() + "_target"} for row in rows
    )
    assert all(
        row.request.state == wire["payload"]["state"] for row, wire in zip(rows, wires, strict=True)
    )
    if split != "train":
        assert all(episode["task_id"] != "upstream_travel" for episode in receipt["episodes"])
    else:
        recovery = next(
            episode
            for episode in receipt["episodes"]
            if episode["initialization"] == "guarded early-open recovery"
        )
        first = next(row for row in rows if row.id == recovery["episode_id"] + "-step-000")
        assert first.gold["operation"] == "CLICK"
        assert "Casa Flora" in first.request.state["page"]["title"]
        assert "Destination anywhere" in first.request.state["page"]["text"]
    assert sum(row.gold["operation"] == "DONE" for row in rows) == expected_episodes
    with pytest.raises(FileExistsError):
        COLLECTOR.collect(upstream_path=upstream, output_dir=output, split=split, repeats=1)


@pytest.mark.integration
def test_partial_travel_recovery_collects_actual_missing_query_and_never_positive_false_done(
    tmp_path,
):
    if os.environ.get("JEV_BROWSER_TESTS") != "1":
        pytest.skip("Set JEV_BROWSER_TESTS=1 for guarded Chromium collection")
    pytest.importorskip("playwright.sync_api")
    from s1.evaluation.datasets import load_cases

    upstream = Path(os.environ.get("JEV_UPSTREAM_PATH", "/tmp/s1-jev-ultrafast"))
    output = tmp_path / "partial"
    receipt = COLLECTOR.collect(
        upstream_path=upstream,
        output_dir=output,
        seed=91024,
        repeats=15,
        task_ids=["upstream_travel"],
        travel_recovery_mode="partial",
    )
    assert receipt["stage"] == "complete" and receipt["source_verification"]["success"]
    assert receipt["upstream_verification"]["success"]
    assert len(receipt["episodes"]) == 30
    assert {episode["task_id"] for episode in receipt["episodes"]} == {"upstream_travel"}
    assert all(episode["verified_dom"]["success"] for episode in receipt["episodes"])
    rows = load_cases(output / "cases.jsonl")
    recovery = [e for e in receipt["episodes"] if e["episode_id"].endswith("-recovery")]
    configurations = {
        (tuple(e.get("initial_prerequisite_indices", [])), e.get("initial_search_submitted", False))
        for e in recovery
    }
    assert len(configurations) == 15
    assert ((0, 1, 2), True) not in configurations
    # Reproduce the actual learned policy failure: Design/free/submitted, no Lisbon.
    missing_query = next(
        e
        for e in recovery
        if e.get("initial_prerequisite_indices") == [1, 2] and e["initial_search_submitted"]
    )
    assert not missing_query["initial_verified_dom"]["success"]
    assert set(missing_query["initial_verified_dom"]["mismatches"]) == {"query"}
    first = next(r for r in rows if r.id == missing_query["episode_id"] + "-step-000")
    assert first.gold["operation"] == "CLICK"
    assert "Destination anywhere" in first.request.state["page"]["text"]
    assert "Design" in first.request.state["page"]["text"]
    assert "Free cancellation enabled" in first.request.state["page"]["text"]
    assert all(
        row.gold["operation"] == "CLICK"
        for row in rows
        if row.id.endswith("recovery-step-000")
    )
    assert sum(row.gold["operation"] == "DONE" for row in rows) == 30
    # Typing without submission is lost when the native detail/back path rerenders
    # the search. The recovery teacher must type again from that fresh observation.
    typed_unsubmitted = next(
        e
        for e in recovery
        if e.get("initial_prerequisite_indices") == [0, 1, 2]
        and not e["initial_search_submitted"]
    )
    episode_rows = [r for r in rows if r.id.startswith(typed_unsubmitted["episode_id"] + "-")]
    assert any(r.gold["operation"] == "TYPE_TEXT" for r in episode_rows)
    assert receipt["scope"] == "development-only scripted training collection; no model evaluation"
