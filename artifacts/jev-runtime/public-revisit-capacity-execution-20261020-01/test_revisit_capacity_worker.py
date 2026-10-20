"""Explicit fake processor tests; no actual processor/framework/model/browser."""

import builtins
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location(
    "revisit_worker_tested", HERE / "revisit_capacity_worker.py"
)
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


@pytest.fixture(autouse=True)
def block_frameworks(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {
            "torch",
            "transformers",
            "numpy",
            "tokenizers",
            "safetensors",
            "peft",
        }:
            raise AssertionError("Fake-only test attempted framework import: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.fixture
def deps(monkeypatch):
    monkeypatch.syspath_prepend(str(w.ROOT / "src"))
    monkeypatch.syspath_prepend(str(w.ROOT / "scripts"))
    q, v, v8 = w.module(w.Q), w.module(w.VALIDATOR), w.module(w.V8)
    monkeypatch.setattr(q, "_helper_instructions", lambda: "EXPLICIT_FAKE_HELPER_TEMPLATE")
    from s1.errors import RequestValidationError
    from s1.jev_generated import _request

    return q, v, v8, _request, RequestValidationError


class FakeTensorRow:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values[:]


class FakeProcessor:
    """Original q._prefix consumes this clearly fake chat/tokenizer interface."""

    def __init__(self):
        self.tokenizer = self
        self.prompts, self.templates = [], []

    def apply_chat_template(self, messages, **kwargs):
        self.templates.append(copy.deepcopy((messages, kwargs)))
        content = messages[0]["content"]
        return content[0]["text"] if isinstance(content, list) else content

    def __call__(self, text, **kwargs):
        assert kwargs == {"return_tensors": "pt", "add_special_tokens": False}
        self.prompts.append(text)
        payload = json.loads(text.split("\n", 1)[1])
        page = payload.get("state", {}).get("page", payload.get("page", {}))
        count = len(page.get("text", "")) // 2
        size = (33 if "questions" in payload else 100) + count
        return {
            "input_ids": [FakeTensorRow(list(range(size)))],
            "attention_mask": [FakeTensorRow([1] * size)],
        }


class FakeGate:
    def __init__(self):
        self.calls = []

    def check(self, *args):
        self.calls.append(args)


@pytest.fixture
def output(tmp_path, monkeypatch):
    ordinary = w.ordinary
    monkeypatch.setattr(
        w,
        "ordinary",
        lambda p: (
            Path(p).absolute() if Path(p).absolute().is_relative_to(tmp_path) else ordinary(p)
        ),
    )
    result = tmp_path / "worker"
    result.mkdir()
    return result


def prefix(count, prompt="fake prompt"):
    return {
        "prompt_sha256": w.sha(prompt.encode()),
        "ids": list(range(count)),
        "attention_mask": [1] * count,
        "token_count": count,
    }


def test_all75_actual_closed_payloads_original_schema_and_prefix_with_fake_processor(
    tmp_path, monkeypatch, deps
):
    q, v, v8, request, error = deps
    cohort = w.parse(w.checked(w.fixed(w.COHORT)))
    root, out = w.ROOT, tmp_path / "worker"
    out.mkdir()
    # Permit only owned test output; original read authorities retain root.
    ordinary = w.ordinary
    monkeypatch.setattr(
        w,
        "ordinary",
        lambda p: (
            Path(p).absolute() if Path(p).absolute().is_relative_to(tmp_path) else ordinary(p)
        ),
    )
    processor, gate = FakeProcessor(), FakeGate()
    model = SimpleNamespace(processor=processor, tok=processor)
    rows = w.run_rows(
        cohort,
        payload_root=(root / w.COHORT).parent,
        q=q,
        validator=v,
        v8=v8,
        model=model,
        gate=gate,
        output=out,
        request=request,
        schema_error=error,
    )
    assert len(rows) == 75 and all(r["expectation_met"] for r in rows)
    assert rows[69]["outcome"] == "REFUSED_SCHEMA" and rows[69]["prefix"] is None
    assert rows[68]["offered_candidate_keys"]["click_target"] == [str(n) for n in range(1, 401)]
    assert (
        rows[71]["outcome"] == "ACCEPTED"
        and rows[72]["outcome"] == rows[74]["outcome"] == "REFUSED_CONTEXT"
    )
    assert rows[73]["kind"] == "helper" and rows[73]["outcome"] == "ACCEPTED"
    assert [r["id"] for r in rows[:68]] == [r["case_id"] for r in cohort["rows"]]
    for n, row in enumerate(rows):
        if row["prefix"]:
            assert row["prefix"]["ids"] == list(range(row["prefix"]["token_count"]))
            assert row["prefix"]["attention_mask"] == [1] * row["prefix"]["token_count"]
        if n < 68:
            old = w.checked(cohort["rows"][n]["payload_ref"], (root / w.COHORT).parent)
            assert (out / row["payload"]["path"]).read_bytes() == old
    factual = [(n, r) for n, r in enumerate(cohort["rows"]) if r["metadata"]["observation_notes"]]
    assert factual and any("arxiv" in r["case_id"] for _, r in factual)
    assert any("observation_revisit" in p for p in processor.prompts)
    assert all(
        t[1] == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        for t in processor.templates
    )
    assert len(list((out / "rows").glob("*.json"))) == 75
    assert not any(
        n.split(".")[0] in {"torch", "transformers", "numpy", "tokenizers"} for n in sys.modules
    )


@pytest.mark.parametrize(
    "kind,count,result",
    [
        ("policy", 8160, "ACCEPTED"),
        ("policy", 8161, "REFUSED_CONTEXT"),
        ("helper", 8064, "ACCEPTED"),
        ("helper", 8065, "REFUSED_CONTEXT"),
    ],
)
def test_exact_context_boundaries_use_original_validator(deps, kind, count, result):
    _, v, _, _, _ = deps
    assert v.validate_prefix(prefix(count), "fake prompt")
    assert v.outcome(prefix(count), kind) == result


@pytest.mark.parametrize("mode", ["exception", "invalid-mask", "mutate"])
def test_processor_failure_never_becomes_schema_or_admission(output, deps, mode):
    q, v, _, request, error = deps
    payload = q.fixture_payload(32)

    def fake_prefix(_model, body, _kind, **_kw):
        if mode == "exception":
            raise RuntimeError("fake processor failed")
        if mode == "mutate":
            body["state"]["changed"] = True
        p = prefix(3)
        if mode == "invalid-mask":
            p["attention_mask"][1] = True
        return p, "fake prompt"

    fake = SimpleNamespace(_prefix=fake_prefix)
    row = w.slots({"rows": [{"case_id": str(i)} for i in range(68)]})[0]
    w.prepare_row(
        row,
        payload,
        {},
        q=fake,
        validator=v,
        model=None,
        gate=FakeGate(),
        output=output,
        request=request,
        schema_error=error,
    )
    assert row["outcome"] == "UNKNOWN" and not row["expectation_met"]
    assert row["error"] is not None


def test_partial_loop_keeps_prior_rows_and_unexecuted_slots(tmp_path, monkeypatch, deps):
    q, v, v8, request, error = deps
    cohort = w.parse(w.checked(w.fixed(w.COHORT)))
    old = w.checked
    calls = 0

    def checked(ref, base=w.ROOT):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise ValueError("third payload pin failed")
        return old(ref, base)

    monkeypatch.setattr(w, "checked", checked)
    ordinary = w.ordinary
    monkeypatch.setattr(
        w,
        "ordinary",
        lambda p: (
            Path(p).absolute() if Path(p).absolute().is_relative_to(tmp_path) else ordinary(p)
        ),
    )
    out = tmp_path / "worker"
    out.mkdir()
    rows, processor = w.slots(cohort), FakeProcessor()
    with pytest.raises(ValueError, match="third payload"):
        w.run_rows(
            cohort,
            payload_root=(w.ROOT / w.COHORT).parent,
            q=q,
            validator=v,
            v8=v8,
            model=SimpleNamespace(processor=processor, tok=processor),
            gate=FakeGate(),
            output=out,
            request=request,
            schema_error=error,
            rows=rows,
        )
    assert rows[0]["expectation_met"] and rows[1]["expectation_met"]
    assert all(r["outcome"] == "NOT_EXECUTED" for r in rows[2:])


def test_source_union_mechanical_and_no_old_authority_mutation():
    cohort = w.parse(w.checked(w.fixed(w.COHORT)))
    technical = w.parse(w.checked(w.fixed(w.TECHNICAL)))
    source = HERE / "revisit_capacity_worker.py"
    ref = {
        "path": str(source.relative_to(w.ROOT)),
        "sha256": w.sha(source.read_bytes()),
        "size_bytes": source.stat().st_size,
    }
    parent = {**ref, "path": str((HERE / "revisit_capacity_controller.py").relative_to(w.ROOT))}
    before = copy.deepcopy((cohort, technical))
    union = w.source_union({"worker_source": ref, "parent_source": parent}, cohort, technical)
    old = w.parse(w.checked(technical["technical"]["execution"]))["files"]
    assert len(old) == 95 and all(union[k] == v for k, v in old.items())
    assert len(union) == len(
        set(old)
        | {str(Path(k).relative_to(w.ROOT)) for k in cohort["source_union"]}
        | {w.Q, w.V8, w.VALIDATOR, w.CPU, w.COMMON, ref["path"], parent["path"]}
    )
    assert (cohort, technical) == before


def test_owner_reuse_or_wrong_command_refused_without_signal(monkeypatch):
    parent = {"pid": 10, "start_ticks": 20, "pgid": 10, "sid": 10}
    child = {"pid": 11, "start_ticks": 30, "pgid": 11, "sid": 11}
    config = {"owner": parent, "parent_source": {}}
    monkeypatch.setattr(w, "checked", lambda ref: b"fake reviewed parent source")
    argv = [
        "/home/albert_hsieh/ComfyUI/.venv/bin/python",
        "-B",
        "-s",
        str(HERE / "revisit_capacity_controller.py"),
        "--run",
        "--plan",
        "/tmp/plan.json",
        "--plan-sha256",
        "a" * 64,
    ]
    assert (
        w.validate_owner(
            config, lookup=lambda pid: parent if pid == 10 else child, cmdline=argv, pid=11, ppid=10
        )
        == child
    )
    with pytest.raises(ValueError, match="Exact live parent"):
        w.validate_owner(
            config, lookup=lambda pid: {**parent, "start_ticks": 21}, cmdline=argv, pid=11, ppid=10
        )
    with pytest.raises(ValueError, match="command shape"):
        w.validate_owner(
            config,
            lookup=lambda pid: parent if pid == 10 else child,
            cmdline=[*argv[:4], "--dry", *argv[5:]],
            pid=11,
            ppid=10,
        )


def test_gate_preserves_resource_primary_when_append_also_fails(output):
    class CPU:
        memory_observation = staticmethod(
            lambda: (_ for _ in ()).throw(RuntimeError("fresh memory failure"))
        )
        append_owned = staticmethod(lambda *a: (_ for _ in ()).throw(OSError("journal failure")))

    gate = w.Gate(output, {"boot_id": "fake", "mem_total": 10}, CPU, clock=lambda: 1)
    with pytest.raises(RuntimeError, match="fresh memory failure") as caught:
        gate.check("startup", "cpu")
    assert any("journal failure" in note for note in caught.value.__notes__)


def test_dry_does_not_load_modules_set_limits_or_write(monkeypatch):
    cohort = {"rows": [{"case_id": str(i)} for i in range(68)]}
    monkeypatch.setattr(w, "load_config", lambda *a: ({}, cohort, {}, {}))
    monkeypatch.setattr(w, "source_union", lambda *a: {"fake-selected-source": {}})
    monkeypatch.setattr(w, "module", lambda *a: pytest.fail("dry module import"))
    monkeypatch.setattr(w, "save", lambda *a, **kw: pytest.fail("dry output write"))
    monkeypatch.setattr(w.resource, "setrlimit", lambda *a: pytest.fail("dry limit mutation"))
    result = w.worker("unused", "0" * 64)
    assert (
        result["rows"] == 75
        and result["processor_calls"] == 0
        and result["output_created"] is False
    )


def test_budget_counts_parent_files_too(output, monkeypatch):
    (output.parent / "controller-existing.json").write_bytes(b"a" * 16)
    monkeypatch.setattr(w, "MAX_OUTPUT", 20)
    monkeypatch.setattr(w, "RESERVE", 0)
    with pytest.raises(ValueError, match="1GiB output"):
        w.save(output, "new.json", b"12345")
    assert not (output / "new.json").exists()


@pytest.mark.parametrize("delta,elapsed,admitted", [(0, 0, True), (-1, 0, False), (0, 600, False)])
def test_new_gate_original_floor_and_cooperative_deadline(output, delta, elapsed, admitted):
    cpu = w.module(w.CPU)
    total = 130595860480
    floor = max(24 * w.GIB, (total + 4) // 5) + 10 * w.GIB
    observed = {"MemTotal": total, "MemAvailable": floor + delta, "boot_id": "fake-boot"}

    def append(out, label, row, previous):
        data = w.canonical(row) + b"\n"
        with (out / label).open("ab") as stream:
            stream.write(data)
        return previous + len(data)

    fake = SimpleNamespace(
        memory_observation=lambda: observed,
        resource_formula=cpu.resource_formula,
        verify_limits=lambda role: {"role": role, "soft_bytes": 8 * w.GIB, "hard_bytes": 8 * w.GIB},
        cuda_state=lambda: {"CUDA_VISIBLE_DEVICES": "", "CUDA_initialized": False},
        append_owned=append,
    )
    clock = iter([0, elapsed])
    gate = w.Gate(
        output, {"mem_total": total, "boot_id": "fake-boot"}, fake, clock=lambda: next(clock)
    )
    if admitted:
        row = gate.check("startup", "cpu")
        assert row["required_bytes"] == floor == 36856590336
        assert row["swap_or_resident_credit_bytes"] == 0
    else:
        with pytest.raises(ValueError):
            gate.check("startup", "cpu")
    saved = w.parse((output / "resource-gates.jsonl").read_bytes())
    assert saved["admitted"] is admitted and (saved["error"] is None) is admitted


def test_real_config_selection_dry_is_small_metadata_only(tmp_path, monkeypatch):
    source = HERE / "revisit_capacity_worker.py"
    parent = HERE / "revisit_capacity_controller.py"
    source_ref = {
        "path": str(source.relative_to(w.ROOT)),
        "sha256": w.sha(source.read_bytes()),
        "size_bytes": source.stat().st_size,
    }
    parent_ref = {"path": str(parent.relative_to(w.ROOT)), "sha256": "b" * 64, "size_bytes": 1}
    config = {
        "schema_id": w.KIND,
        "cohort": w.fixed(w.COHORT),
        "technical_plan": w.fixed(w.TECHNICAL),
        "processor_plan": w.fixed(w.PROCESSOR),
        "worker_source": source_ref,
        "parent_source": parent_ref,
        "owner": {"pid": 1, "start_ticks": 2, "pgid": 1, "sid": 1},
        "nonce": "a" * 32,
        "boot_id": "fake-boot",
        "mem_total": 130595860480,
        "output": str(w.ROOT / "artifacts/jev-runtime/unused-capacity-test-output"),
    }
    ordinary = w.ordinary
    monkeypatch.setattr(
        w,
        "ordinary",
        lambda p: (
            Path(p).absolute() if Path(p).absolute().is_relative_to(tmp_path) else ordinary(p)
        ),
    )
    checked = w.checked
    monkeypatch.setattr(
        w, "checked", lambda ref, base=w.ROOT: b"x" if ref == parent_ref else checked(ref, base)
    )
    path = tmp_path / "config.json"
    data = w.canonical(config)
    path.write_bytes(data)
    monkeypatch.setattr(w, "module", lambda *a: pytest.fail("dry dynamic import"))
    result = w.worker(path, w.sha(data))
    assert (
        result["rows"] == 75
        and result["processor_calls"] == 0
        and result["new_source_union_files"] > 95
    )
    config["cohort"] = {**config["cohort"], "sha256": "0" * 64}
    data = w.canonical(config)
    path.write_bytes(data)
    with pytest.raises(ValueError, match="immutable input selection"):
        w.worker(path, w.sha(data))


def test_worker_entry_fake_limits_and_double_failure_preserve_primary(output, monkeypatch):
    config = {
        "output": str(output.parent),
        "nonce": "a" * 32,
        "boot_id": "fake",
        "mem_total": 130595860480,
    }
    # The entry creates this subtree itself; no actor or native limits are used.
    output.rmdir()
    cohort = {"rows": [{"case_id": str(i)} for i in range(68)]}
    monkeypatch.setattr(w, "load_config", lambda *a: (config, cohort, {}, {}))
    monkeypatch.setattr(w, "source_union", lambda *a: {})
    monkeypatch.setattr(
        w, "validate_owner", lambda *a: {"pid": 11, "pgid": 11, "sid": 11, "start_ticks": 20}
    )
    calls = []
    monkeypatch.setattr(w, "sys", SimpleNamespace(modules={}, path=[]))
    monkeypatch.setattr(
        w,
        "module",
        lambda label: SimpleNamespace(
            configure_limits=lambda role: calls.append(("fake_AS8", role))
        ),
    )
    monkeypatch.setattr(
        w,
        "resource",
        SimpleNamespace(
            RLIMIT_CORE=1, RLIMIT_FSIZE=2, setrlimit=lambda *a: calls.append(("fake_limit", *a))
        ),
    )
    monkeypatch.setattr(
        w,
        "signal",
        SimpleNamespace(
            SIGALRM=14, getsignal=lambda *a: None, signal=lambda *a: None, alarm=lambda *a: None
        ),
    )

    class RefusedGate:
        def __init__(self, *a):
            pass

        def check(self, *a):
            raise RuntimeError("primary fresh resource refusal")

    monkeypatch.setattr(w, "Gate", RefusedGate)
    captured = []

    def failed_terminal(_output, label, blob, **_kwargs):
        assert label == "terminal.json"
        captured.append(w.parse(blob))
        raise OSError("secondary terminal write failure")

    monkeypatch.setattr(w, "save", failed_terminal)
    with pytest.raises(RuntimeError, match="primary fresh resource refusal") as caught:
        w.worker("fake-config", "b" * 64, run=True)
    assert any("secondary terminal write failure" in n for n in caught.value.__notes__)
    assert captured[0]["counts"]["UNKNOWN"] == 75 and captured[0]["complete"] is False
    assert calls == [
        ("fake_AS8", "worker"),
        ("fake_limit", 1, (0, 0)),
        ("fake_limit", 2, (64 * w.MIB, 64 * w.MIB)),
    ]


def valid_gate_rows():
    cpu = w.module(w.CPU)
    config = {"boot_id": "fake-boot", "mem_total": 130595860480}
    phases = [
        "startup",
        "source-before",
        "runtime-before",
        "cpu-reference-before-processor-load",
        "processor-after",
        *(["prepare"] * 75),
        "final-cpu",
    ]
    obs = {"boot_id": "fake-boot", "MemTotal": config["mem_total"], "MemAvailable": 36856590336}
    formula = cpu.resource_formula(obs, config["boot_id"], prospective=True)
    rows = [
        {
            "schema_id": w.KIND + "_resource_gate",
            "scope": "cpu",
            "sequence": i,
            "phase": phase,
            "monotonic": 1000 + i / 1000,
            "worker_elapsed_seconds": i / 1000,
            "admitted": True,
            "error": None,
            "observation": obs.copy(),
            **formula,
            "limits": {"role": "worker", "soft_bytes": 8 * w.GIB, "hard_bytes": 8 * w.GIB},
            "cuda": {"CUDA_VISIBLE_DEVICES": "", "torch_loaded": i >= 4, "CUDA_initialized": False},
        }
        for i, phase in enumerate(phases)
    ]
    return rows, config


def test_complete_new_gate_journal_and_original_component_formula():
    rows, config = valid_gate_rows()
    blob = b"\n".join(w.canonical(r) for r in rows) + b"\n"
    result = w.validate_gates(blob, config)
    assert result["rows"] == 81 and result["sha256"] == w.sha(blob)
    assert result["last_worker_elapsed_seconds"] == 0.08


@pytest.mark.parametrize(
    "change",
    [
        "missing-stage",
        "out-of-order",
        "low-floor",
        "wrong-boot",
        "wrong-total",
        "AS",
        "cuda",
        "credit",
        "elapsed600",
        "nonmonotonic",
        "not-admitted",
    ],
)
def test_gate_validator_refuses_incomplete_or_contradictory_evidence(change):
    rows, config = valid_gate_rows()
    if change == "missing-stage":
        del rows[3]
    elif change == "out-of-order":
        rows[2]["phase"] = "prepare"
    elif change == "low-floor":
        rows[10]["observation"]["MemAvailable"] -= 1
    elif change == "wrong-boot":
        rows[10]["observation"]["boot_id"] = "other-boot"
    elif change == "wrong-total":
        rows[10]["observation"]["MemTotal"] += 1
    elif change == "AS":
        rows[10]["limits"]["soft_bytes"] -= 1
    elif change == "cuda":
        rows[10]["cuda"]["CUDA_initialized"] = True
    elif change == "credit":
        rows[10]["swap_or_resident_credit_bytes"] = False
    elif change == "elapsed600":
        rows[-1]["worker_elapsed_seconds"] = 600
    elif change == "nonmonotonic":
        rows[10]["monotonic"] = 999
    else:
        rows[10]["admitted"] = False
    with pytest.raises(ValueError):
        w.validate_gates(b"\n".join(w.canonical(r) for r in rows), config)
