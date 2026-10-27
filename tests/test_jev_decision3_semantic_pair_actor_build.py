"""Synthetic actor-versioning tests; no cases, browser or GPU are used."""

import hashlib
import json
from pathlib import Path

import jev_decision3_semantic_pair_actor_build as builder
import pytest


def _sha(value):
    return hashlib.sha256(value).hexdigest()


ACTOR = """\
from pathlib import Path
import json
SOURCE_ONLY = True
SERVICE = Path('/old-child.py')
PINS = {SERVICE: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'}
def main():
    def start_candidate():
        static_path = Path('/old-template.json')
        raw = read(static_path)
        need(sha(raw) == 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb', 'template')
        static = json.loads(raw)
        need(static['schema'] == 'decision3-browser-candidate-static-template-v1', 'schema')
        transport = {'binding_release': '/old-binding'}
        plan = {'schema': 'decision3-candidate-browser-child-plan-v1', 'transport': transport}
        return plan
    def stop_candidate():
        try:
            terminal = {'service': {}}
            summary_path = run_dir / 'summary.json'
            if summary_path.exists():
                summary = json.loads(read(summary_path))
                need(True, 'old summary preserved')
        except Exception:
            raise
    return start_candidate, stop_candidate

def baseline():
    return 'unaltered baseline'

def judge():
    return 'unaltered judge'
"""


def _inputs(tmp_path):
    actor = tmp_path / "actor.py"
    actor.write_text(ACTOR)
    paths = {}
    for name in ("child", "binding", "audit", "evidence"):
        held, released = tmp_path / f"{name}-held.py", tmp_path / f"{name}-release.py"
        held.write_text(f"SOURCE_ONLY = True\nNAME = {name!r}\n")
        released.write_text(held.read_text().replace("SOURCE_ONLY = True", "SOURCE_ONLY = False"))
        paths[name + "_held"] = str(held)
        paths[name + "_release"] = str(released)
    template = tmp_path / "template.json"
    data = {
        "schema": "decision3-semantic-browser-static-template-v1",
        "python": "/usr/bin/python3",
        "cache_root": "/tmp/cache",
        "modules_cache": "/tmp/modules",
        "overlay": "/tmp/overlay",
        "source_pins": {
            paths[name + "_release"]: _sha(Path(paths[name + "_release"]).read_bytes())
            for name in ("child", "binding", "audit", "evidence")
        },
        "deltas": {},
        "vocabulary": {"names_sha256": "c" * 64},
        "binding_release": paths["binding_release"],
        "semantic_adapter_release": "/tmp/adapter.py",
        "semantic_audit_release": paths["audit_release"],
        "baseline_sha256": "d" * 64,
    }
    template.write_text(json.dumps(data))
    return dict(
        actor_path=str(actor),
        actor_sha256=_sha(actor.read_bytes()),
        template_path=str(template),
        template_sha256=_sha(template.read_bytes()),
        **paths,
    ), data


def test_actor_preserves_baseline_and_adds_explicit_semantic_parent_check(tmp_path):
    args, data = _inputs(tmp_path)
    raw = builder.build(**args)
    text = raw.decode()
    assert "SOURCE_ONLY = True" in text and "SOURCE_ONLY = False" not in text
    assert "return 'unaltered baseline'" in text and "return 'unaltered judge'" in text
    called = []

    class FakeEvidence:
        def verify(self, **kwargs):
            called.append(kwargs)
            return {"status": "provided_bytes_and_fp64_match"}

    def source(path, _name):
        return FakeEvidence() if str(path) == args["evidence_release"] else object()

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    plan_path = run_dir / "candidate-plan.json"
    plan_path.write_text(json.dumps({"sentinel": "owned-plan"}))
    (run_dir / "summary.json").write_text(json.dumps({"policy_calls": 2, "text_calls": 1}))
    captured = []

    def need(value, message):
        if not value:
            raise ValueError(message)

    env = {
        "read": lambda p: Path(p).read_bytes(),
        "sha": _sha,
        "need": need,
        "source": source,
        "run_dir": run_dir,
        "candidate_output": run_dir / "candidate-service",
        "child_plan_sha": _sha(plan_path.read_bytes()),
        "private": lambda p, raw: captured.append((p, raw)),
    }
    exec(compile(raw, "<synthetic-actor>", "exec"), env)
    start, stop = env["main"]()
    plan = start()
    assert plan["schema"] == "decision3-semantic-browser-child-plan-v1"
    assert plan["semantic_opt_in"] is True and plan["vocabulary"] == data["vocabulary"]
    assert plan["transport"]["binding_release"] == args["binding_release"]
    assert plan["transport"]["semantic_adapter_release"] == data["semantic_adapter_release"]
    assert plan["transport"]["semantic_audit_release"] == args["audit_release"]
    stop()
    assert len(called) == 1 and called[0]["plan"] == {"sentinel": "owned-plan"}
    assert called[0]["baseline_sha256"] == "d" * 64
    assert called[0]["runner_policy_calls"] == 2 and called[0]["runner_text_calls"] == 1
    assert captured[0][0] == run_dir / "semantic-evidence.json"
    plan_path.write_text("{}")
    with pytest.raises(ValueError, match="plan changed"):
        stop()


@pytest.mark.parametrize("fault", ["actor", "template", "release", "missing_pin", "schema"])
def test_actor_builder_rejects_unreviewed_inputs(tmp_path, fault):
    args, data = _inputs(tmp_path)
    if fault == "actor":
        args["actor_sha256"] = "0" * 64
    elif fault == "template":
        args["template_sha256"] = "0" * 64
    elif fault == "release":
        Path(args["child_release"]).write_text("SOURCE_ONLY = False\nchanged = True\n")
    elif fault == "missing_pin":
        data["source_pins"].pop(args["evidence_release"])
        Path(args["template_path"]).write_text(json.dumps(data))
        args["template_sha256"] = _sha(Path(args["template_path"]).read_bytes())
    elif fault == "schema":
        data["schema"] = "other"
        Path(args["template_path"]).write_text(json.dumps(data))
        args["template_sha256"] = _sha(Path(args["template_path"]).read_bytes())
    with pytest.raises(ValueError):
        builder.build(**args)
