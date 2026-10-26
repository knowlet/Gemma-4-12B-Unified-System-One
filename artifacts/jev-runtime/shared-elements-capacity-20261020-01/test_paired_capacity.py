"""Explicit fake processor/lifecycle tests; no model, GPU, unit or HTTP actors."""

import ast
import builtins
import copy
import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location("paired_capacity_tested", HERE / "paired_capacity.py")
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
BOOT = "11111111-2222-4333-8444-555555555555"


@pytest.fixture(autouse=True)
def no_heavy_runtime(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {
            "torch",
            "transformers",
            "tokenizers",
            "numpy",
            "peft",
            "playwright",
        }:
            pytest.fail("Pure paired test imported heavy runtime: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(d.subprocess, "Popen", lambda *_a, **_k: pytest.fail("Test launched actor"))


def test_resource03_exact_identity_inverse_and_whole_function_ast():
    old = d.read(d.V2 / "cpu_prepare_form_3g.py").decode()
    new = d.read(HERE / "cpu_prepare_form_3g.py").decode()
    inverse = (
        new.replace("CPU_REFERENCE_ONLY_3G_V3", "CPU_REFERENCE_ONLY_3G_V2")
        .replace("3g-v3", "3g-v2")
        .replace("jev_cpu3g_unit_observation_v3", "jev_cpu3g_unit_observation_v2")
        .replace("jev-cpu3g-20261020-03.service", "jev-cpu3g-20261020-02.service")
    )
    assert inverse == old

    def functions(text):
        return {
            n.name: ast.dump(n, include_attributes=False)
            for n in ast.parse(text).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        }

    assert functions(old) == functions(new)
    cpu = d.module(d.pin(HERE / "cpu_prepare_form_3g.py"))
    assert cpu.PROFILE == "CPU_REFERENCE_ONLY_3G_V3" and cpu.EXPECTED_CGROUP == d.CGROUP
    assert cpu.TREE_RSS == 3 * 1024**3 and cpu.limit_target("parent") == (2 * 1024**3, 8 * 1024**3)


def test_new_current_union_has93_unchanged2_replacements_and_new_codec():
    plan = d.make_plan(BOOT)
    old = d.module(d.WORKER)
    files, technical, _ = d.sources(plan, old)
    prior = old.parse(old.checked(technical["technical"]["execution"]))["files"]
    assert len(prior) == 95
    assert sum(files[p] == item for p, item in prior.items()) == 93
    assert all(files[p] == plan["production_sources"][p] for p in d.PRODUCTION)
    assert files[old.Q] == {k: v for k, v in old.fixed(old.Q).items() if k != "path"}
    d.authenticate(files)


def test_default_dry_creates_no_output_or_limits(capsys, monkeypatch):
    monkeypatch.setattr(d, "run_parent", lambda *_: pytest.fail("Dry parent"))
    monkeypatch.setattr(d, "run_worker", lambda *_: pytest.fail("Dry worker"))
    assert d.main([]) == 0
    assert json.loads(capsys.readouterr().out)["processor_calls"] == 0
    with pytest.raises(ValueError, match="selected plan"):
        d.main(["--run"])


def test_bound_plan_and_source_drift_refuse_without_output():
    plan = d.make_plan(BOOT)
    with tempfile.TemporaryDirectory(prefix="pure-plan-", dir=HERE) as folder:
        path = Path(folder) / "plan.json"
        path.write_bytes(d.wire(plan))
        assert d.load_plan(path, d.sha(path.read_bytes()))[0] == plan
        plan["production_sources"]["src/s1/jev_generated.py"]["sha256"] = "0" * 64
        path.write_bytes(d.wire(plan))
        with pytest.raises(ValueError, match="pin differs"):
            d.load_plan(path, d.sha(path.read_bytes()))
        with pytest.raises(ValueError, match="plan SHA"):
            d.load_plan(path, "0" * 64)
    assert not (HERE / "actual-01").exists()


class Tokens(list):
    def tolist(self):
        return list(self)


class FakeProcessor:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        return "EXPLICIT_FAKE_CHAT[" + messages[0]["content"][0]["text"] + "]"

    def __call__(self, *, text, **kwargs):
        assert kwargs == {"return_tensors": "pt", "add_special_tokens": False}
        values = list(text.encode())
        return {"input_ids": [Tokens(values)], "attention_mask": [Tokens([1] * len(values))]}


class FakeGate:
    def __init__(self):
        self.calls = []

    def check(self, phase, scope):
        self.calls.append((phase, scope))


def save(root, label, blob, **_):
    path = root / label
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(blob)
    return {"path": label, "sha256": d.sha(blob), "size_bytes": len(blob)}


def refread(ref, root):
    raw = (root / ref["path"]).read_bytes()
    assert len(raw) == ref["size_bytes"] and d.sha(raw) == ref["sha256"]
    return raw


@pytest.fixture
def deps():
    from s1 import jev_prompt
    from s1.jev_generated import format_action_request

    old = d.module(d.WORKER)
    validator = old.module(old.VALIDATOR)
    model = SimpleNamespace(processor=FakeProcessor())

    def helper_prefix(model, payload, kind, *, include_prompt=False):
        assert kind == "helper"
        pair = d.prefix_from_text(model, "EXPLICIT_FAKE_HELPER:" + d.wire(payload).decode())
        return pair if include_prompt else pair[0]

    q = SimpleNamespace(
        TEXT_VALUE="fake-helper", parse_helper_request=lambda p, **_: p, _prefix=helper_prefix
    )
    return dict(
        model=model,
        q=q,
        validator=validator,
        formatter=format_action_request,
        codec=jev_prompt,
        gate=FakeGate(),
    )


def test_true_production_formatter_codec_and_fake_processor_preserve_full_curl(deps):
    baseline = json.loads(d.checked(d.BASELINE))
    row = baseline["rows"][51]
    payload = json.loads(refread(row["payload"], (d.ROOT / d.BASELINE["path"]).parent))
    original = copy.deepcopy(payload)
    raw = d.measure(payload, "policy", "raw-json-v1", **deps)
    shared = d.measure(payload, "policy", "shared-elements-v1", **deps)
    assert payload == original
    assert raw["labels"] == shared["labels"] == row["offered_candidate_keys"]
    assert shared["representation"]["criterion_refs"] == 250
    assert d.wire(deps["codec"].unpack_payload(shared["encoded"])) == d.wire(payload)
    assert len(shared["encoded"].encode()) < len(d.wire(payload))
    assert len(raw["prefix"]["ids"]) == raw["prefix"]["token_count"]


def test_401_stops_before_fake_processor_and_both_modes_refuse(deps):
    baseline = json.loads(d.checked(d.BASELINE))
    row = baseline["rows"][69]
    payload = json.loads(refread(row["payload"], (d.ROOT / d.BASELINE["path"]).parent))
    deps["model"].processor = lambda *_a, **_k: pytest.fail("Schema401 reached processor")
    for mode in ("raw-json-v1", "shared-elements-v1"):
        value = d.measure(payload, "policy", mode, **deps)
        assert value["outcome"] == "REFUSED_SCHEMA" and value["prefix"] is None


def test_changed_roundtrip_and_full_ids_masks_are_not_silently_accepted(deps):
    from s1.jev_generated import _request

    baseline = json.loads(d.checked(d.BASELINE))
    row = baseline["rows"][0]
    payload = json.loads(refread(row["payload"], (d.ROOT / d.BASELINE["path"]).parent))
    original = deps["formatter"]

    def wrong(payload, identity, **kwargs):
        text, source, labels, meta = original(payload, identity, **kwargs)
        return text, source + b" ", labels, meta

    deps["formatter"] = wrong
    value = d.measure(payload, "policy", "raw-json-v1", **deps)
    assert value["outcome"] == "UNKNOWN" and value["prefix"] is None
    assert _request(payload, "generated-actions-none")[1] == d.wire(payload)


@pytest.fixture
def fake75(tmp_path, deps):
    actual = json.loads(d.checked(d.BASELINE))
    oldroot = (d.ROOT / d.BASELINE["path"]).parent
    baseline = {"rows": []}
    for oldrow in actual["rows"]:
        payload = json.loads(refread(oldrow["payload"], oldroot))
        row = copy.deepcopy(oldrow)
        measured = d.measure(payload, row["kind"], "raw-json-v1", **deps)
        row["prefix"], row["outcome"] = measured["prefix"], measured["outcome"]
        row["offered_candidate_keys"] = measured["labels"]
        row["payload"] = save(
            tmp_path / "baseline", f"payloads/{row['ordinal']:02d}.json", d.wire(payload)
        )
        baseline["rows"].append(row)
    old = SimpleNamespace(checked=lambda ref, root: refread(ref, root))
    root = tmp_path / "trial/worker"
    rows = d.paired_rows(
        baseline,
        input_root=tmp_path / "baseline",
        cpu=SimpleNamespace(save=save),
        root=root,
        old=old,
        **deps,
    )
    return baseline, rows, root, deps


def test_full75_pairs_use_identical_payloads_and_keep_every_original_control(fake75):
    baseline, rows, root, _ = fake75
    assert len(rows) == 75 and [r["id"] for r in rows] == [r["id"] for r in baseline["rows"]]
    assert len(list((root / "payloads").iterdir())) == 75
    assert all(r["raw_matches_frozen"] for r in rows)
    assert rows[69]["measurements"]["shared-elements-v1"]["outcome"] == "REFUSED_SCHEMA"
    assert all(
        r["measurements"]["raw-json-v1"]["prefix"] == p["prefix"]
        for r, p in zip(rows, baseline["rows"], strict=True)
    )
    assert rows[51]["measurements"]["shared-elements-v1"]["representation"]["criterion_refs"] == 250


def test_raw_baseline_mismatch_retains_earlier_partial_rows(fake75, tmp_path):
    baseline, _, _, deps = fake75
    baseline["rows"][1]["prefix"]["ids"][0] += 1
    partial = []
    with pytest.raises(ValueError, match="frozen75 baseline"):
        d.paired_rows(
            baseline,
            input_root=tmp_path / "baseline",
            cpu=SimpleNamespace(save=save),
            root=tmp_path / "other",
            old=SimpleNamespace(checked=refread),
            rows=partial,
            **deps,
        )
    assert len(partial) == 1 and partial[0]["ordinal"] == 0


def test_parent_independently_accepts_complete_fake75_closure_and_rejects_missing_control(
    fake75, tmp_path, monkeypatch
):
    baseline, rows, root, _ = fake75
    realroot = d.ROOT
    cpu = d.module(d.pin(HERE / "cpu_prepare_form_3g.py"))
    validator = d.module(d.WORKER).module(d.module(d.WORKER).VALIDATOR)
    for name in ("jev_generated.py", "jev_prompt.py"):
        target = tmp_path / "src/s1" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((realroot / "src/s1" / name).read_bytes())
    monkeypatch.setattr(d, "ROOT", tmp_path)
    baseline_path = tmp_path / "baseline/terminal.json"
    baseline_path.write_bytes(d.wire(baseline))
    monkeypatch.setattr(d, "BASELINE", d.pin(baseline_path))
    production = {
        "src/s1/jev_prompt.py": {
            k: v for k, v in d.pin(tmp_path / "src/s1/jev_prompt.py").items() if k != "path"
        }
    }
    plan = {"unit": {"boot_id": BOOT, "cgroup_path": d.CGROUP}, "production_sources": production}
    technical = {
        "technical": {"runtime": {"path": "original-runtime", "sha256": "a" * 64, "size_bytes": 1}}
    }
    monkeypatch.setattr(d, "sources", lambda *_: ({}, technical, {}))
    processor = {
        "reference_items": {
            f"fake-ref-{n}": {"sha256": d.sha(b"x"), "size_bytes": 1} for n in range(58)
        }
    }
    processor_path = tmp_path / "processor.json"
    processor_path.write_bytes(d.wire(processor))
    processor_ref = d.pin(processor_path)
    old = SimpleNamespace(
        PROCESSOR="processor.json",
        VALIDATOR="explicit-validator",
        parse=json.loads,
        fixed=lambda _: processor_ref,
        checked=lambda ref, base=None: refread(ref, tmp_path if base is None else base),
        module=lambda _: validator,
    )
    for label in processor["reference_items"]:
        save(root / "reference-environment", label, b"x")
    save(
        root,
        "source-union.json",
        d.wire(
            {
                "schema_id": d.KIND + "_current_sources",
                "files": {},
                "count": 0,
                "prior95_qualification_attested": False,
            }
        ),
    )
    save(
        root,
        "runtime-component.json",
        d.wire(
            {
                "runtime_ref": technical["technical"]["runtime"],
                "runtime_files": 348,
                "processor_plan": processor_ref,
                "reference_count": 58,
                "original95_current_qualification_attested": False,
            }
        ),
    )
    evidence = {
        "tokenizer_is_processor_tokenizer": True,
        "max_context": 8192,
        "generation_EOS": [1, 106, 50],
        "CPU_threads": 2,
        "CPU_interop_threads": 2,
    }
    save(root, "processor-before.json", d.wire(evidence))
    save(root, "processor-after.json", d.wire(evidence))
    child = {"pid": 22222, "start_ticks": 33333, "pgid": 22222, "sid": 22222}
    unit = {
        "schema_id": cpu.UNIT_SCHEMA,
        "boot_id": BOOT,
        "cgroup_path": d.CGROUP,
        "cgroup_dev": 0,
        "cgroup_inode": 1,
        "caller": child,
        "observed_monotonic": 1.0,
        "memory_max_bytes": 3 * 1024**3,
        "memory_high": "max",
        "memory_current_bytes": 1024,
        "memory_peak_bytes": 2048,
        "memory_swap_max_bytes": 0,
        "memory_swap_current_bytes": 0,
        "memory_events": {k: 0 for k in cpu.UNIT_EVENTS},
        "pids_current": 1,
        "processes": [child["pid"]],
        "cgroup_events": {"populated": 1, "frozen": 0},
        "nr_descendants": 0,
        "nr_dying_descendants": 0,
    }
    obs = {"MemTotal": 130595860480, "MemAvailable": 31000000000, "boot_id": BOOT}
    phases = (
        [
            "startup",
            "source-before",
            "runtime-before",
            "cpu-reference-before-processor-load",
            "processor-after",
        ]
        + ["prepare"] * 75
        + ["final-cpu"]
    )
    gates = [
        {
            "schema_id": d.KIND + "_resource_gate",
            "sequence": i,
            "phase": phase,
            "monotonic": 10 + i / 100,
            "elapsed": i / 100,
            "admitted": True,
            "error": None,
            "observation": obs,
            **cpu.resource_formula(obs, BOOT, prospective=True),
            "unit": unit,
            "limits": {"role": "worker", "soft_bytes": 8 * 1024**3, "hard_bytes": 8 * 1024**3},
            "cuda": {"CUDA_VISIBLE_DEVICES": "", "torch_loaded": True, "CUDA_initialized": False},
        }
        for i, phase in enumerate(phases)
    ]
    save(root, "resource-gates.jsonl", b"".join(d.wire(r) + b"\n" for r in gates))
    save(root.parent, "intent.json", d.wire({"mem_total": obs["MemTotal"]}))
    extra = []
    for band in (8128, 8161, 8065):
        prompt = str(band)
        extra.append(
            {
                "id": "compact-band-" + str(band),
                "band": band,
                "kind": "helper" if band == 8065 else "policy",
                "outcome": "ACCEPTED" if band == 8128 else "REFUSED_CONTEXT",
                "payload": save(root, f"controls/{band}.json", b"{}"),
                "prefix": {
                    "prompt_sha256": d.sha(prompt.encode()),
                    "token_count": band,
                    "ids": [1] * band,
                    "attention_mask": [1] * band,
                },
                "prompt": save(root, f"controls/{band}.txt", prompt.encode()),
                "scope": "SUPPLEMENTAL_NEW_FORMAT_RECIPE_NOT_GENUINE_QUALITY",
            }
        )
    result = {
        "schema_id": d.KIND + "_worker",
        "plan_sha256": "a" * 64,
        "nonce": "b" * 32,
        "complete": True,
        "primary_error": None,
        "child": child,
        "rows": rows,
        "supplemental_controls": extra,
        "processor_only": True,
        "native_model_forward_HTTP_browser_calls": 0,
        "task_quality_attested": False,
    }
    save(root, "terminal.json", d.wire(result))
    cleanup = {
        "confirmed": True,
        "remaining": [],
        "tracked_PID_start_ticks": {str(child["pid"]): child["start_ticks"]},
        "tracked_groups": [child["pgid"]],
    }
    assert d.validate_result(plan, "a" * 64, "b" * 32, cpu, old, root.parent, cleanup)["rows"] == 75
    result["supplemental_controls"].pop()
    (root / "terminal.json").write_bytes(d.wire(result))
    with pytest.raises(ValueError, match="Complete independently typed"):
        d.validate_result(plan, "a" * 64, "b" * 32, cpu, old, root.parent, cleanup)


def test_supplemental_controls_use_explicit_prefix_facade_and_expected_context(deps, tmp_path):
    seen = []

    def scan(facade, model, gate, output, *, helper):
        assert callable(facade._prefix) and callable(facade.fixture_payload)
        seen.append(helper)
        bands = [8065] if helper else [8128, 8161]
        return {
            n: (
                {"fake_control": n},
                (
                    {
                        "prompt_sha256": d.sha(str(n).encode()),
                        "ids": [1] * n,
                        "attention_mask": [1] * n,
                        "token_count": n,
                    },
                    str(n),
                ),
            )
            for n in bands
        }

    deps["q"].fixture_payload = lambda *_a, **_k: {"EXPLICIT_FAKE": True}
    values = d.controls(
        cpu=SimpleNamespace(save=save),
        root=tmp_path,
        old=None,
        v8=SimpleNamespace(scan=scan),
        **deps,
    )
    assert seen == [False, True]
    assert [r["outcome"] for r in values] == ["ACCEPTED", "REFUSED_CONTEXT", "REFUSED_CONTEXT"]
    assert all(r["scope"].startswith("SUPPLEMENTAL") for r in values)


def test_gate_preserves_primary_if_persistence_also_fails(tmp_path):
    primary = MemoryError("EXPLICIT fresh floor failure")

    def fail():
        raise primary

    cpu = SimpleNamespace(
        save=save,
        memory_observation=fail,
        caps=lambda *_: None,
        append_owned=lambda *_: (_ for _ in ()).throw(OSError("EXPLICIT evidence failure")),
    )
    gate = d.Gate(tmp_path, {"unit": {"boot_id": BOOT}}, cpu, 123)
    with pytest.raises(MemoryError) as caught:
        gate.check("startup", "cpu")
    assert caught.value is primary and isinstance(caught.value.__cause__, OSError)


def test_parent_low_memory_refuses_before_output_or_limit_change(monkeypatch, tmp_path):
    cpu = SimpleNamespace(
        EXPECTED_ENVIRONMENT={},
        SECRET_FREE_ENV=(),
        assert_current_unit=lambda *_: {},
        memory_observation=lambda: {"MemTotal": 123},
        configure_limits=lambda *_: pytest.fail("Pre-refusal limit mutation"),
    )
    controller = SimpleNamespace(
        prospective=lambda *_: (_ for _ in ()).throw(MemoryError("floor refused"))
    )
    old = SimpleNamespace(
        COMMON="fake",
        module=lambda *_: None,
        identity=lambda p: {"pid": p, "start_ticks": 1, "pgid": p, "sid": p},
    )
    plan = {"environment": {}, "unit": {"boot_id": BOOT}, "output": "actual-unused"}
    monkeypatch.setattr(d, "module", lambda *_: controller)
    monkeypatch.setattr(d.sys, "executable", d.PYTHON)
    monkeypatch.setattr(d.sys, "version_info", (3, 13))
    monkeypatch.setenv("JEV_CPU3G_BOOT", BOOT)
    monkeypatch.setenv("JEV_CPU3G_CGROUP", d.CGROUP)
    monkeypatch.setattr(d.os, "getpgrp", d.os.getpid)
    monkeypatch.setattr(d.os, "getsid", lambda _: d.os.getpid())
    fields = {
        "InvocationID": "a" * 32,
        "MainPID": str(d.os.getpid()),
        "ControlGroup": d.CGROUP,
        "MemoryMax": str(3 * 1024**3),
        "MemoryHigh": "infinity",
        "MemorySwapMax": "0",
        "OOMPolicy": "kill",
        "KillMode": "control-group",
        "RuntimeMaxUSec": "11min",
        "TimeoutStopUSec": "10s",
        "Type": "exec",
    }
    monkeypatch.setattr(
        d.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(
            stdout="\n".join(k + "=" + v for k, v in fields.items()).encode(), stderr=b""
        ),
    )
    with pytest.raises(MemoryError, match="floor refused"):
        d.run_parent(tmp_path / "plan", "a" * 64, plan, cpu, old, {})
    assert not (d.ROOT / plan["output"]).exists()
