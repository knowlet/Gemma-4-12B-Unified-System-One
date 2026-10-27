"""Pure synthetic official-SDK stand-ins; no private cases, browser or GPU."""

import hashlib
import importlib.abc
import json
import sys
from types import ModuleType, SimpleNamespace

import jev_decision3_jevbench_coherence_owner as owner
import jev_decision3_runtime_integrity as features
import jevbench_public_request_inventory as inventory
import pytest


class Tensor:
    def __init__(self, raw):
        self.raw = raw

    def detach(self):
        return self

    def contiguous(self):
        return self

    def numpy(self):
        return self

    def tobytes(self):
        return self.raw


class Lease:
    def __init__(self, *_args, **_kwargs):
        self.fd = self.held = self.fatal = None


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, *_args):
        return None


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    monkeypatch.setattr(owner, "SOURCE_ONLY", False)
    monkeypatch.setattr(owner.common, "_shared_identity", lambda *_: None)
    monkeypatch.setattr(owner.common, "_verify_source_pins", lambda *_: None)
    monkeypatch.setattr(owner.common, "_load_deltas", lambda *_: deltas.copy())
    monkeypatch.setattr(owner.common, "_fsync_directory", lambda *_: None, raising=False)
    monkeypatch.setattr(owner, "_fsync_directory", lambda *_: None)
    monkeypatch.setattr(owner, "_runtime_guard", lambda *_: None)
    monkeypatch.setattr(owner, "_model_stat_snapshot", lambda *_: ("synthetic",))
    monkeypatch.setattr(
        owner.child, "_lease_summary", lambda _t, _pid, n: {"own_completed_pairs": n}
    )
    monkeypatch.setattr(features, "_validate_manifest", lambda *_: None)
    monkeypatch.setattr(inventory, "_check_coherence_files", lambda *_: None)
    original_append = owner.child._append

    def append(path, row):
        if path.name == "callback-timing.jsonl":
            if controls.drop_timing and row["call"] == 3:
                return
            original_append(path, row)

    monkeypatch.setattr(owner.child, "_append", append)
    deltas = {key: Tensor(key.encode()) for key in ("operation", "click_target")}
    plan = {
        "unit": "synthetic",
        "tasks_root": str(tmp_path),
        "cache_root": str(tmp_path),
        "source_pins": {},
        "gpu_uuid": "synthetic",
        "request_seconds": 180,
        "transport": {"lock": "x", "journal": "y", "stat": {}, "nonce": "a" * 32},
        "deltas": {k: {"fp32_sha256": hashlib.sha256(k.encode()).hexdigest()} for k in deltas},
    }
    manifest = tmp_path / "src/jevbench/suites/jevbench-mini/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    cases = [
        SimpleNamespace(id=f"case-{i}", state=f"state-{i}", questions={"head": {"value": i}})
        for i in range(240)
    ]
    rows = [
        {
            "case": cases[i % 240].id,
            "relation": "synthetic",
            "key": f"key-{i}",
            "t": 0,
            "v": 0,
            "requests": i,
        }
        for i in range(1200)
    ]

    rows[0]["relation"] = "batch_order"
    rows[0]["targets"] = ["head", "other"]

    class Plan:
        by = {"synthetic": rows}

        def __len__(self):
            return len(rows)

        def trial(self, row, _case):
            return SimpleNamespace(targets=row.get("targets", ["head"]), params={})

    suite = SimpleNamespace(cases=cases, plan=Plan())

    def decode(case, number):
        return {"one": {"head": {"value": f"trial-{number}"}} if number < 1008 else case.questions}

    expected_outcomes = []
    for row in rows:
        trial_id = f"{row['relation']}:{row['case']}:{row['key']}:0"
        targets = (
            [[target] for target in row["targets"]]
            if row["relation"] == "batch_order"
            else [["head"]]
        )
        for target_set in targets:
            expected_outcomes.append(
                {
                    "trial_id": trial_id,
                    "relation": row["relation"],
                    "case_id": row["case"],
                    "targets": target_set,
                }
            )
    expected_requests = [(case.state, case.questions) for case in cases]
    case_by_id = {case.id: case for case in cases}
    expected_requests.extend(
        (case_by_id[row["case"]].state, decode(case_by_id[row["case"]], row["requests"])["one"])
        for row in rows[:1008]
    )
    controls = SimpleNamespace(
        fail_at=None, missing_trial=False, drop_timing=False, duplicate_outcome=False
    )

    class FakeGeneric:
        def __init__(self, backend, *, lease, identity_guard, event_sink, **_kwargs):
            self.backend, self.lease, self.guard, self.sink = (
                backend,
                lease,
                identity_guard,
                event_sink,
            )
            self.calls = 0

        def predict(self, state, questions):
            self.calls += 1
            self.guard(self.backend, self.lease)
            request = {"state": state, "questions": questions}
            row = {
                "route": "jevbench-generic-native-v1",
                "call": self.calls,
                "request_sha256": owner.child._sha(owner.child._encode(request)),
                "delta_applied": [],
                "delta_bypassed": ["operation", "click_target"],
            }
            self.sink({**row, "event": "planned", "request": request})
            if self.calls == controls.fail_at:
                self.sink({**row, "event": "failed"})
                raise RuntimeError("synthetic failure")
            self.sink({**row, "event": "native_preflight"})
            self.sink({**row, "event": "native_raw_returned"})
            response = {"answers": {"head": {"noul": 0.5}}}
            self.sink(
                {
                    **row,
                    "event": "prepared_not_delivered",
                    "response": response,
                    "response_sha256": owner.child._sha(owner.child._encode(response)),
                }
            )
            return response

    class Backend:
        def __init__(self, fn, **_kwargs):
            self.fn = fn

    class Report:
        overall = 0.5

        def __init__(self, result):
            self.result = result

        def save(self, path):
            path.write_text(json.dumps({"synthetic": True}))

    def evaluate(backend, **settings):
        assert settings["cache"] == ":memory:"
        assert settings["concurrency"] == 1
        assert settings["minimize"] is False
        for state, questions in expected_requests:
            backend.fn(state, questions)
        outcomes = list(expected_outcomes)
        if controls.missing_trial:
            outcomes.pop()
        if controls.duplicate_outcome:
            outcomes.append(outcomes[-1].copy())
        return Report(
            {
                "requests": 1248,
                "cache_hits": 193,
                "bases": dict.fromkeys(range(240)),
                "planned_tests": {"synthetic": 1200},
                "outcomes": outcomes,
                "errors": [],
                "truncated_by_budget": False,
                "backend_down": None,
            }
        )

    def finder(*_args):
        package = ModuleType("jevbench")
        package.__path__ = []
        package.evaluate = evaluate
        modules = {"jevbench": package}
        for name in ("backends", "plan", "schema", "suites"):
            modules["jevbench." + name] = ModuleType("jevbench." + name)
        modules["jevbench.backends"].CallableBackend = Backend
        modules["jevbench.plan"].decode_requests = decode
        modules["jevbench.schema"].validate_question = lambda *_: None
        modules["jevbench.suites"].load_suite = lambda _name: suite
        for key, module in modules.items():
            monkeypatch.setitem(sys.modules, key, module)
        return Finder()

    monkeypatch.setattr(inventory, "_verified_coherence", finder)
    monkeypatch.setattr(
        inventory,
        "coherence_inventory",
        lambda *_: {
            "case_question_unique_requests": 1248,
            "state_question_unique_requests": 1248,
            "summary": {"totals": {"contract_accepted": 1248, "decision3_static_accepted": 1248}},
        },
    )
    return SimpleNamespace(
        plan=plan,
        backend=SimpleNamespace(runtime=object()),
        generic=SimpleNamespace(GenericJevBenchAdapter=FakeGeneric),
        lease=SimpleNamespace(ForwardLease=Lease),
        output=tmp_path,
        controls=controls,
    )


def run(env):
    return owner._run_coherence(
        env.plan, env.backend, object(), env.generic, env.lease, env.output, lambda: None
    )


def test_complete_fresh_cache_original_scorer_run_is_quarantined(synthetic):
    result = run(synthetic)
    assert result["calls"] == 1248 and result["planned_trials"] == 1200
    assert result["outcome_rows"] == 1201
    assert result["event_counts"]["failed"] == 0 and result["score_accepted"] is False
    assert result["callback_timing"]["count"] == 1248
    assert result["cost_usd"] is None
    timing = (synthetic.output / "callback-timing.jsonl").read_bytes().splitlines()
    assert len(timing) == 1248
    assert (
        hashlib.sha256(b"\n".join(timing) + b"\n").hexdigest()
        == result["callback_timing"]["sha256"]
    )
    stage = synthetic.output / "staging"
    assert json.loads((stage / "status.json").read_text())["score_accepted"] is False
    assert (stage / "coherence-mini/report.json").is_file()
    assert (
        json.loads((stage / "coherence-mini/STATUS_NOT_ACCEPTED.json").read_text())[
            "score_accepted"
        ]
        is False
    )


def test_missing_official_trial_cannot_be_scored(synthetic):
    synthetic.controls.missing_trial = True
    with pytest.raises(ValueError, match="1200 trial"):
        run(synthetic)
    assert not (synthetic.output / "staging/coherence-mini/report.json").exists()
    assert (synthetic.output / "staging/coherence-mini/STATUS_NOT_ACCEPTED.json").is_file()


def test_one_failed_call_aborts_and_leaves_no_scorer_report(synthetic):
    synthetic.controls.fail_at = 3
    with pytest.raises(owner.EvaluationAbort, match="entire evaluation invalid"):
        run(synthetic)
    assert not (synthetic.output / "staging/coherence-mini/report.json").exists()
    assert (synthetic.output / "staging/coherence-mini/STATUS_NOT_ACCEPTED.json").is_file()


def test_worker_is_source_only_by_default(monkeypatch):
    monkeypatch.setattr(owner, "SOURCE_ONLY", True)
    with pytest.raises(ValueError, match="independent review"):
        owner.main(["--plan", "/missing", "--plan-sha256", "a" * 64])


def test_missing_callback_timing_cannot_publish_report(synthetic):
    synthetic.controls.drop_timing = True
    with pytest.raises(ValueError, match="complete callback timing"):
        run(synthetic)
    assert not (synthetic.output / "staging/coherence-mini/report.json").exists()
    assert (synthetic.output / "staging/coherence-mini/STATUS_NOT_ACCEPTED.json").is_file()


def test_duplicate_outcome_for_existing_trial_cannot_be_scored(synthetic):
    synthetic.controls.duplicate_outcome = True
    with pytest.raises(ValueError, match="1200 trial"):
        run(synthetic)
    assert not (synthetic.output / "staging/coherence-mini/report.json").exists()
    assert (synthetic.output / "staging/coherence-mini/STATUS_NOT_ACCEPTED.json").is_file()
