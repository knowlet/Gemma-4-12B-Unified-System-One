"""Pure V2 trial tests with an explicit byte-token fake processor; no actors."""

import builtins
import copy
import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).absolute().parent
spec = importlib.util.spec_from_file_location("repeat_capacity_tested", HERE / "repeat_capacity.py")
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)
BOOT = "11111111-2222-4333-8444-555555555555"


@pytest.fixture(autouse=True)
def no_actors_or_frameworks(monkeypatch):
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
            pytest.fail("Heavy import: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(d.subprocess, "Popen", lambda *_a, **_k: pytest.fail("Actor started"))


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
        ids = list(text.encode())
        for band in (8128, 8161, 8065):
            if "EXPLICIT_FAKE_BAND" + str(band) in text:
                ids = [1] * band
        return {"input_ids": [Tokens(ids)], "attention_mask": [Tokens([1] * len(ids))]}


def save(root, label, blob, **_):
    path = root / label
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(blob)
    return {"path": label, "sha256": d.core().sha(blob), "size_bytes": len(blob)}


def checked(ref, root):
    raw = (root / ref["path"]).read_bytes()
    assert len(raw) == ref["size_bytes"] and d.core().sha(raw) == ref["sha256"]
    return raw


@pytest.fixture
def deps():
    from s1 import jev_prompt
    from s1.jev_generated import format_action_request

    c = d.core()
    old = c.module(c.WORKER)
    validator = old.module(old.VALIDATOR)
    model = SimpleNamespace(processor=FakeProcessor())

    def helper(model, payload, kind, *, include_prompt=False):
        assert kind == "helper"
        pair = c.prefix_from_text(model, "EXPLICIT_FAKE_HELPER:" + c.wire(payload).decode())
        return pair if include_prompt else pair[0]

    q = SimpleNamespace(
        TEXT_VALUE="fake-helper",
        parse_helper_request=lambda p, **_: p,
        _prefix=helper,
        fixture_payload=lambda *_a, **_k: {},
    )
    return c, dict(
        formatter=format_action_request,
        codec=jev_prompt,
        q=q,
        model=model,
        validator=validator,
        gate=SimpleNamespace(check=lambda *_: None),
    )


def baseline(c):
    return json.loads(c.checked(c.BASELINE))


def payload(c, ordinal):
    row = baseline(c)["rows"][ordinal]
    return row, json.loads(checked(row["payload"], (d.ROOT / c.BASELINE["path"]).parent))


def test_reuses_frozen03_without_helper_or_driver_copy():
    c = d.core()
    assert c.pin(d.ROOT / d.CORE["path"]) == d.CORE and c.pin(d.ROOT / d.CPU["path"]) == d.CPU
    assert not (HERE / "cpu_prepare_form_3g.py").exists()
    plan = d.make_plan(BOOT)
    files, technical = d.source_union(c, plan, c.module(c.WORKER))
    prior = json.loads(c.module(c.WORKER).checked(technical["technical"]["execution"]))["files"]
    assert sum(files[p] == item for p, item in prior.items()) == 93
    assert files[d.CORE["path"]]["sha256"] == d.CORE["sha256"]
    assert files[plan["self_source"]["path"]]["sha256"] == plan["self_source"]["sha256"]
    c.authenticate(files)


def test_default_and_strict_plan_refuse_drift_without_output(capsys):
    assert d.main([]) == 0 and json.loads(capsys.readouterr().out)["processor_calls"] == 0
    with pytest.raises(ValueError):
        d.main(["--run"])
    plan = d.make_plan(BOOT)
    with tempfile.TemporaryDirectory(prefix="pure-plan-", dir=HERE) as folder:
        path = Path(folder) / "plan.json"
        path.write_bytes(d.core().wire(plan))
        assert d.load_plan(path, d.core().sha(path.read_bytes()))[1] == plan
        plan["production_sources"]["src/s1/jev_prompt.py"]["sha256"] = "0" * 64
        path.write_bytes(d.core().wire(plan))
        with pytest.raises(ValueError, match="pin differs"):
            d.load_plan(path, d.core().sha(path.read_bytes()))
    assert not (HERE / "actual-01").exists()


def test_actual_v2_codec_full_curl_roundtrip_and_candidate_keys(deps):
    c, args = deps
    prior, value = payload(c, 51)
    original = copy.deepcopy(value)
    measured = d.measure(c, value, "policy", d.V2, **args)
    assert measured["outcome"] != "UNKNOWN" and value == original
    assert c.wire(args["codec"].unpack_payload(measured["encoded"])) == c.wire(value)
    assert measured["representation"]["requested_format"] == d.V2
    assert (
        measured["labels"] == c.frozen_labels(prior)
        and len(measured["labels"]["click_target"]) == 250
    )
    assert (
        measured["representation"]["candidate_user_text_bytes"][d.V2]
        < measured["representation"]["candidate_user_text_bytes"][d.RAW]
    )


def test_raw_fallback_full_prompt_ids_mask_and_parent_verification(deps, tmp_path):
    c, args = deps
    prior, value = payload(c, 71)
    value["state"] = {"goal": "tiny"}
    raw = d.measure(c, value, "policy", d.RAW, **args)
    selected = d.measure(c, value, "policy", d.V2, **args)
    assert (
        selected["representation"]["format"] == d.RAW
        and selected["prefix"] == raw["prefix"]
        and selected["prompt"] == raw["prompt"]
    )
    row = {
        "kind": "policy",
        "measurements": {
            mode: d.retain(SimpleNamespace(save=save), tmp_path, mode, result)
            for mode, result in ((d.RAW, raw), (d.V2, selected))
        },
    }
    fakeprior = {
        **prior,
        "prefix": raw["prefix"],
        "outcome": raw["outcome"],
        "offered_candidate_keys": raw["labels"],
        "offered_head_order": list(raw["labels"]),
    }
    params = dict(
        root=tmp_path,
        old=SimpleNamespace(checked=checked),
        validator=args["validator"],
        format_action_request=args["formatter"],
        jev_prompt=args["codec"],
    )
    d.validate_values(c, row, value, fakeprior, **params)
    selected["prefix"] = copy.deepcopy(selected["prefix"])
    selected["prefix"]["ids"][0] += 1
    with pytest.raises(ValueError, match="prefix equality"):
        d.validate_values(c, row, value, fakeprior, **params)


def test_original401_is_rejected_before_processor(deps):
    c, args = deps
    _, value = payload(c, 69)
    args["model"].processor = lambda *_a, **_k: pytest.fail("401 reached processor")
    for mode in (d.RAW, d.V2):
        result = d.measure(c, value, "policy", mode, **args)
        assert result["outcome"] == "REFUSED_SCHEMA" and result["prefix"] is None


def test_forged_requested_format_or_byte_selection_never_admits(deps):
    c, args = deps
    _, value = payload(c, 0)
    original = args["formatter"]

    def forged(payload, identity, **kwargs):
        text, source, labels, meta = original(payload, identity, **kwargs)
        meta = {**meta, "requested_format": "not-v2"}
        return text, source, labels, meta

    args["formatter"] = forged
    assert d.measure(c, value, "policy", d.V2, **args)["outcome"] == "UNKNOWN"


def test_full75_original_pairs_and_raw_ids_are_preserved(deps, tmp_path):
    c, args = deps
    original = baseline(c)
    fake = copy.deepcopy(original)
    for row in fake["rows"]:
        _, value = payload(c, row["ordinal"])
        result = d.measure(c, value, row["kind"], d.RAW, **args)
        row["prefix"], row["outcome"] = result["prefix"], result["outcome"]
    rows = []
    d.pairs(
        c,
        fake,
        cpu=SimpleNamespace(save=save),
        root=tmp_path,
        old=SimpleNamespace(checked=checked),
        rows=rows,
        **args,
    )
    assert len(rows) == 75 and [r["id"] for r in rows] == [r["id"] for r in original["rows"]]
    assert all(r["raw_matches_frozen"] for r in rows)
    assert rows[69]["measurements"][d.V2]["outcome"] == "REFUSED_SCHEMA"
    assert all(
        r["measurements"][d.RAW]["prefix"] == p["prefix"]
        for r, p in zip(rows, fake["rows"], strict=True)
    )


def test_three_new_controls_keep_literal_bands_and_effective_format(deps, tmp_path):
    c, args = deps
    _, original = payload(c, 71)
    seen = []

    def scan(facade, model, gate, root, *, helper):
        seen.append(helper)
        result = {}
        for band in (8065,) if helper else (8128, 8161):
            value = copy.deepcopy(original)
            value["state"] = {"goal": "EXPLICIT_FAKE_BAND" + str(band)}
            result[band] = (
                value,
                facade._prefix(model, value, "helper" if helper else "policy", include_prompt=True),
            )
        return result

    rows = []
    d.supplemental(
        c,
        cpu=SimpleNamespace(save=save),
        root=tmp_path,
        rows=rows,
        v8=SimpleNamespace(scan=scan),
        **args,
    )
    assert seen == [False, True] and [r["band"] for r in rows] == [8128, 8161, 8065]
    assert [r["measurements"][d.V2]["outcome"] for r in rows] == [
        "ACCEPTED",
        "REFUSED_CONTEXT",
        "REFUSED_CONTEXT",
    ]


def test_worker_exact_new_entry_binding_refuses_before_any_limits(tmp_path):
    c = d.core()
    with tempfile.TemporaryDirectory(prefix="pure-owner-", dir=HERE) as folder:
        output = Path(folder)
        owner = {"pid": 11, "start_ticks": 22, "pgid": 11, "sid": 11}
        (output / "intent.json").write_bytes(
            c.wire(
                {
                    "nonce": "a" * 32,
                    "plan_sha256": "b" * 64,
                    "parent": owner,
                    "plan_path": str(output / "plan.json"),
                }
            )
        )
        cpu = SimpleNamespace(
            child_argv=lambda _: ["wrong-entry"],
            configure_limits=lambda _: pytest.fail("Limit changed"),
        )
        old = SimpleNamespace(identity=lambda _: owner)
        with pytest.raises(ValueError, match="new entry command"):
            d.worker(
                c, {"output": str(output.relative_to(d.ROOT))}, "b" * 64, "a" * 32, cpu, old, {}, {}
            )
        assert not (output / "worker").exists()
