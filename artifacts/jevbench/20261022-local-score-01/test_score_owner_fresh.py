"""Real controller entry rejects every stale output before invoking a service."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "stale",
    [
        "actual-02",
        "actor-intent-02.json",
        "root-launch-02b.json",
        "root-unit-readback-02b.json",
        "root-external-closure-02b.json",
        "root-finalization-error-02b.json",
        "score-service-02b.log",
    ],
)
def test_stale_evidence_refuses_before_any_service_or_output(tmp_path, monkeypatch, stale):
    source = Path(__file__).parent / "launch_score_02b.py"
    spec = importlib.util.spec_from_file_location("_fresh_owned_score_test", source)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    here = tmp_path / "owned"
    here.mkdir()
    monkeypatch.setattr(owner, "ROOT", tmp_path)
    monkeypatch.setattr(owner, "HERE", here)
    monkeypatch.setattr(owner.resource, "setrlimit", lambda *_: None)
    monkeypatch.setattr(
        owner,
        "load_guard",
        lambda: (SimpleNamespace(status=lambda _: {"LoadState": "not-found"}), {}),
    )
    host = {
        "mem_total_bytes": 130595864576,
        "mem_available_bytes": 130000000000,
        "comfy_running": 0,
        "comfy_pending": 0,
    }
    monkeypatch.setattr(owner, "load_observer", lambda _: lambda: host)

    def forbidden(*_, **__):
        pytest.fail("Controller invoked an actor/service with stale output")

    monkeypatch.setattr(owner, "command", forbidden)
    (here / stale).write_text("retained historical evidence")
    plan = {
        "unit": owner.UNIT,
        "cgroup": owner.CGROUP,
        "output": "owned/actual-02",
        "unit_memory_max_bytes": owner.CAP,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        "expected_memtotal": host["mem_total_bytes"],
        "sources": [],
    }
    blob = json.dumps(plan).encode()
    path = tmp_path / "plan.json"
    path.write_bytes(blob)
    with pytest.raises(AssertionError, match="Fresh owned output/evidence"):
        owner.main(
            ["--plan", str(path), "--plan-sha256", hashlib.sha256(blob).hexdigest(), "--run"]
        )
    assert {p.name for p in here.iterdir()} == {stale}
