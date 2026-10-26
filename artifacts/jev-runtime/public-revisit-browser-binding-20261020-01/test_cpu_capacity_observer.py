"""Pure observer protocol fixtures; native limits, ML and browser runs forbidden."""

import builtins
import importlib.util
import resource
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location(
    "tested_cpu_capacity_external_observer", HERE / "cpu_capacity_observer.py"
)
o = importlib.util.module_from_spec(spec)
spec.loader.exec_module(o)


@pytest.fixture(autouse=True)
def pure_only(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {
            "torch",
            "transformers",
            "numpy",
            "tokenizers",
            "peft",
            "playwright",
            "httpx",
        }:
            raise AssertionError("Observer pure test imported runtime: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(
        o.resource, "setrlimit", lambda *a: pytest.fail("Pure observer test changed a native limit")
    )


def components():
    c = o.controller_module()
    b = c.pinned_module(o.BINDING, "_pure_observer_protocol_binding")
    return c, b, b.expected_sources()


def fake_record_inputs():
    _c, b, expected = components()
    parent = {"pid": 101, "start_ticks": 202, "pgid": 101, "sid": 101}
    config = {"nonce": "a" * 32, "boot_id": "EXPLICIT_FAKE_BOOT", "owner": parent}
    outer = {"nonce": config["nonce"], "parent": parent}
    cleanup = {"tracked_PID_start_ticks": {"101": 202, "303": 404}, "tracked_groups": [101, 303]}
    terminal_ref = {
        "path": b.CPU_BASE + "/actual-01/controller-terminal.json",
        "sha256": "d" * 64,
        "size_bytes": 10,
    }
    return (
        b,
        expected,
        {
            "config": config,
            "outer": outer,
            "terminal_ref": terminal_ref,
            "identity": parent,
            "argv": o.parent_argv(expected, "/home/albert_hsieh/ComfyUI/.venv/bin/python"),
            "sources": {k: expected[k] for k in ("controller_source", "worker_source")},
            "missing": o.absence({}, parent, cleanup),
            "started": 1000,
            "finished": 1100,
            "observed": 1101,
        },
        cleanup,
    )


def test_emitted_external_receipt_matches_independent_new_selector_schema():
    b, expected, kwargs, _cleanup = fake_record_inputs()
    value = o.external_record(b, expected, **kwargs)
    assert value["schema_id"] == b.EXTERNAL and value["returncode"] == 0
    assert value["parent"] == kwargs["config"]["owner"]
    assert value["source_before"] == value["source_after"]
    assert len(value["argv"]) == 9 and value["argv"][4:6] == ["--run", "--plan"]


@pytest.mark.parametrize(
    "mode",
    [
        "live-parent",
        "reused-parent",
        "PG-member",
        "SID-member",
        "tracked-child",
        "untracked-group-member",
    ],
)
def test_live_or_reused_owned_identity_cannot_emit_closed_receipt(mode):
    b, expected, kwargs, cleanup = fake_record_inputs()
    parent = kwargs["identity"]
    row = {"pid": 999, "start_ticks": 888, "pgid": 999, "sid": 999}
    if mode == "live-parent":
        row = parent
    elif mode == "reused-parent":
        row = {**parent, "start_ticks": 203}
    elif mode == "PG-member":
        row["pgid"] = 101
    elif mode == "SID-member":
        row["sid"] = 101
    elif mode == "tracked-child":
        row.update(pid=303, start_ticks=404)
    else:
        row["pgid"] = 303
    kwargs["missing"] = o.absence({row["pid"]: row}, parent, cleanup)
    with pytest.raises(ValueError, match="absence"):
        o.external_record(b, expected, **kwargs)
    if mode == "reused-parent":
        assert kwargs["missing"]["matching_parent_start_absent"] is True
        assert kwargs["missing"]["parent_pid_absent"] is False


@pytest.mark.parametrize("mode", ["source", "argv", "owner", "time"])
def test_protocol_binding_drift_never_creates_ready_receipt(mode):
    b, expected, kwargs, _cleanup = fake_record_inputs()
    if mode == "source":
        kwargs["sources"] = {**kwargs["sources"], "worker_source": {}}
    elif mode == "argv":
        kwargs["argv"] = [*kwargs["argv"][:4], "--dry", *kwargs["argv"][5:]]
    elif mode == "owner":
        kwargs["identity"] = {**kwargs["identity"], "start_ticks": 203}
    else:
        kwargs["observed"] = 1099
    with pytest.raises(ValueError):
        o.external_record(b, expected, **kwargs)


def test_actual_defaultdry_reads_only_small_source_and_creates_no_output():
    output = HERE / "cpu-observer-dry-not-created"
    before = resource.getrlimit(resource.RLIMIT_AS)
    result = o.observe(output)
    assert result["status"] == "DRY_SOURCE_ONLY_NO_CHILD_PROCESSOR_OR_OUTPUT"
    assert (
        result["benchmark_ready"]
        is result["output_created"]
        is result["actual_output_exists"]
        is False
    )
    assert not output.exists() and resource.getrlimit(resource.RLIMIT_AS) == before


def test_fresh_floor_refusal_precedes_limits_outputs_and_child(monkeypatch):
    c, b, expected, plan, cpu, worker, cohort, files, actual = o.sources_and_plan()
    fake_cpu = SimpleNamespace(**cpu.__dict__)
    fake_cpu.memory_observation = lambda: {
        "MemTotal": 130595860480,
        "MemAvailable": 36856590335,
        "boot_id": "EXPLICIT_FAKE_BOOT",
    }
    fake_cpu.configure_limits = lambda *a: pytest.fail("low memory changed AS")
    monkeypatch.setattr(
        o,
        "sources_and_plan",
        lambda: (c, b, expected, plan, fake_cpu, worker, cohort, files, actual),
    )
    for key, value in plan["environment"].items():
        monkeypatch.setenv(key, value)
    output = HERE / "cpu-observer-low-not-created"
    digest = o.sha(Path(o.__file__).read_bytes())
    result = o.observe(output, digest, run=True)
    assert result["status"] == "NOT_STARTED_ORIGINAL_RESOURCE_FLOOR_REFUSED"
    assert result["formula"]["required_bytes"] == 36856590336
    assert (
        result["benchmark_ready"] is result["process_started"] is result["output_created"] is False
    )
    assert not output.exists() and not actual.exists()


def test_controller_source_drift_refused_before_execution(monkeypatch):
    monkeypatch.setitem(o.CONTROLLER, "sha256", "0" * 64)
    with pytest.raises(ValueError, match="Frozen controller bytes"):
        o.controller_module()


def test_run_requires_explicit_observer_source_pin():
    with pytest.raises(ValueError, match="Explicit reviewed observer source"):
        o.main(["--run", "--output", str(HERE / "cpu-observer-never-created")])


def test_combined_budget_counts_both_disjoint_trees_and_terminal_reserve(tmp_path, monkeypatch):
    observer, actual = tmp_path / "observer", tmp_path / "actual"
    observer.mkdir()
    actual.mkdir()
    (observer / "log").write_bytes(b"x" * 10)
    (actual / "row").write_bytes(b"x" * 10)
    monkeypatch.setattr(o, "MAX_OUTPUT", 20)
    monkeypatch.setattr(o, "RESERVE", 2)
    # Either tree is below its own cap, but their combined nonterminal bytes fail.
    with pytest.raises(ValueError, match="Combined observed"):
        o.combined_caps(observer, actual)
    o.combined_caps(observer, actual, terminal=True)
    with pytest.raises(ValueError, match="Combined observed"):
        o.combined_caps(observer, actual, 1, terminal=True)
    cpu = SimpleNamespace(save=lambda *a, **kw: pytest.fail("overbudget publication wrote output"))
    with pytest.raises(ValueError, match="Combined observed"):
        o.combined_save(cpu, observer, "never", b"x", observer_output=observer, actual=actual)


def test_composite_monitor_keeps_original_memory_check_and_combined_budget(monkeypatch, tmp_path):
    calls = []

    class MemoryMonitor:
        def __init__(self, output, boot, parent):
            calls.append(("construct-original", output, boot, parent))

        def check(self, rows):
            calls.append(("original-memory-check", rows))
            return "original-component-record"

    observer, actual, parent = tmp_path / "observer", tmp_path / "actual", {"pid": 10}
    monkeypatch.setattr(
        o, "combined_caps", lambda out, act: calls.append(("combined-budget", out, act))
    )
    monitor = o.CombinedMonitor(
        SimpleNamespace(OwnedMemoryMonitor=MemoryMonitor), observer, "fake-boot", parent, actual
    )
    rows = {11: {"pid": 11}}
    assert monitor.check(rows) == "original-component-record"
    assert calls == [
        ("construct-original", observer, "fake-boot", parent),
        ("original-memory-check", rows),
        ("combined-budget", observer, actual),
    ]


def test_combined_budget_refuses_overlapping_or_symlink_populations(tmp_path):
    outer = tmp_path / "outer"
    outer.mkdir()
    with pytest.raises(ValueError, match="Disjoint"):
        o.combined_caps(outer, outer / "nested")
    other = tmp_path / "other"
    other.mkdir()
    (outer / "bad-link").symlink_to(other)
    with pytest.raises(ValueError, match="Ordinary"):
        o.combined_caps(outer, other)
