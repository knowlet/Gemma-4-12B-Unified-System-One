"""Build a reviewable semantic variant of a pinned existing paired actor.

Only the candidate protocol plan and its semantic evidence verification are
changed. The original runner, capture, baseline, DOM judge and resource
supervision remain in the byte-for-byte copied actor. No actor is executed here.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import stat
from pathlib import Path


def _need(value, message):
    if not value:
        raise ValueError(message)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _read(path, limit=8 * 1024**2):
    path = Path(path)
    _need(path.is_absolute() and path.resolve() == path, "ordinary absolute source path required")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        _need(
            stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit,
            "bounded regular source required",
        )
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(limit + 1)
        after = os.fstat(fd)
        _need(
            len(raw) == before.st_size
            and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            "source changed during read",
        )
        return raw
    finally:
        os.close(fd)


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _unique(nodes, name):
    values = list(nodes)
    _need(len(values) == 1, "exactly one " + name + " required")
    return values[0]


def _assignment(nodes, target):
    return _unique(
        (
            n
            for n in nodes
            if isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == target
        ),
        target + " assignment",
    )


def _release(held_path, release_path):
    held, release = _read(held_path), _read(release_path)
    marker = b"SOURCE_ONLY = True"
    _need(
        held.count(marker) == 1 and release == held.replace(marker, b"SOURCE_ONLY = False", 1),
        "release must change only the held gate",
    )
    return _sha(release)


def build(
    *,
    actor_path,
    actor_sha256,
    template_path,
    template_sha256,
    child_held,
    child_release,
    binding_held,
    binding_release,
    audit_held,
    audit_release,
    evidence_held,
    evidence_release,
):
    """Return the held actor bytes; caller is responsible for external review."""
    _need(
        _digest(actor_sha256) and _digest(template_sha256),
        "explicit source and template hashes required",
    )
    raw, template_raw = _read(actor_path), _read(template_path)
    _need(
        _sha(raw) == actor_sha256 and _sha(template_raw) == template_sha256,
        "actor or template hash differs",
    )
    _need(
        raw.count(b"SOURCE_ONLY = True") == 1 and b"SOURCE_ONLY = False" not in raw,
        "held actor required",
    )
    template = json.loads(template_raw)
    keys = {
        "schema",
        "python",
        "cache_root",
        "modules_cache",
        "overlay",
        "source_pins",
        "deltas",
        "vocabulary",
        "binding_release",
        "semantic_adapter_release",
        "semantic_audit_release",
        "baseline_sha256",
    }
    _need(
        type(template) is dict
        and set(template) == keys
        and template["schema"] == "decision3-semantic-browser-static-template-v1"
        and _digest(template["baseline_sha256"]),
        "exact pinned semantic template required",
    )
    releases = {
        "child": (child_held, child_release),
        "binding": (binding_held, binding_release),
        "audit": (audit_held, audit_release),
        "evidence": (evidence_held, evidence_release),
    }
    hashes = {key: _release(held, released) for key, (held, released) in releases.items()}
    _need(
        template["binding_release"] == str(binding_release)
        and template["semantic_audit_release"] == str(audit_release),
        "template releases differ",
    )
    pins = template["source_pins"]
    _need(
        type(pins) is dict
        and all(pins.get(str(path)) == hashes[key] for key, (_, path) in releases.items()),
        "template must pin each released source",
    )
    source = raw.decode("utf-8")
    tree = ast.parse(source)
    service = _assignment(tree.body, "SERVICE")
    pins_node = _assignment(tree.body, "PINS")
    _need(isinstance(pins_node.value, ast.Dict), "literal actor pin inventory required")
    service_pin = _unique(
        (
            value
            for key, value in zip(pins_node.value.keys, pins_node.value.values, strict=True)
            if isinstance(key, ast.Name) and key.id == "SERVICE"
        ),
        "service source pin",
    )
    main = _unique(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"), "main"
    )
    start = _unique(
        (n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == "start_candidate"),
        "start candidate",
    )
    stop = _unique(
        (n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == "stop_candidate"),
        "stop candidate",
    )
    static_path = _assignment(start.body, "static_path")
    transport = _assignment(start.body, "transport")
    plan = _assignment(start.body, "plan")
    _need(
        isinstance(transport.value, ast.Dict) and isinstance(plan.value, ast.Dict),
        "literal original plan and transport required",
    )
    static_check = _unique(
        (
            n
            for n in ast.walk(start)
            if isinstance(n, ast.Compare)
            and isinstance(n.left, ast.Call)
            and isinstance(n.left.func, ast.Name)
            and n.left.func.id == "sha"
            and len(n.left.args) == 1
            and isinstance(n.left.args[0], ast.Name)
            and n.left.args[0].id == "raw"
            and len(n.comparators) == 1
            and isinstance(n.comparators[0], ast.Constant)
            and _digest(n.comparators[0].value)
        ),
        "static template hash check",
    )
    static_schema = _unique(
        (
            n
            for n in ast.walk(start)
            if isinstance(n, ast.Constant)
            and n.value == "decision3-browser-candidate-static-template-v1"
        ),
        "static template schema",
    )
    summary = _unique(
        (
            n
            for n in ast.walk(stop)
            if isinstance(n, ast.If)
            and isinstance(n.test, ast.Call)
            and isinstance(n.test.func, ast.Attribute)
            and isinstance(n.test.func.value, ast.Name)
            and n.test.func.value.id == "summary_path"
            and n.test.func.attr == "exists"
        ),
        "candidate summary check",
    )
    _need(
        static_path.col_offset == transport.col_offset == plan.col_offset
        and summary.col_offset > stop.col_offset,
        "unexpected actor indentation",
    )
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line.encode("utf-8")))

    def position(line, column):
        return offsets[line - 1] + column

    def span(node):
        return position(node.lineno, node.col_offset), position(
            node.end_lineno, node.end_col_offset
        )

    edits = []

    def replace(node, value):
        a, b = span(node)
        edits.append((a, b, value.encode()))

    def after(node, value):
        _, b = span(node)
        edits.append((b, b, value.encode()))

    replace(service.value, f"Path({str(child_release)!r})")
    replace(service_pin, repr(hashes["child"]))
    after(
        pins_node,
        "\n"
        + "\n".join(
            [
                f"SEMANTIC_BINDING = Path({str(binding_release)!r})",
                f"SEMANTIC_AUDIT = Path({str(audit_release)!r})",
                f"SEMANTIC_EVIDENCE = Path({str(evidence_release)!r})",
                f"PINS.update({{SEMANTIC_BINDING: {hashes['binding']!r}, SEMANTIC_AUDIT: {hashes['audit']!r}, SEMANTIC_EVIDENCE: {hashes['evidence']!r}}})",
            ]
        ),
    )
    replace(static_path.value, f"Path({str(template_path)!r})")
    replace(static_check.comparators[0], repr(template_sha256))
    replace(static_schema, repr("decision3-semantic-browser-static-template-v1"))
    indent = " " * transport.col_offset
    after(
        transport,
        "\n"
        + "\n".join(
            indent + line
            for line in (
                "transport['binding_release'] = static['binding_release']",
                "transport['semantic_adapter_release'] = static['semantic_adapter_release']",
                "transport['semantic_audit_release'] = static['semantic_audit_release']",
            )
        ),
    )
    after(
        plan,
        "\n"
        + "\n".join(
            indent + line
            for line in (
                "plan['schema'] = 'decision3-semantic-browser-child-plan-v1'",
                "plan['semantic_opt_in'] = True",
                "plan['vocabulary'] = static['vocabulary']",
            )
        ),
    )
    indent = " " * summary.col_offset
    after(
        summary,
        "\n"
        + "\n".join(
            indent + line
            for line in (
                "semantic_evidence = source(SEMANTIC_EVIDENCE, '_semantic_pair_evidence')",
                "semantic_audit = source(SEMANTIC_AUDIT, '_semantic_pair_audit')",
                "semantic_binding = source(SEMANTIC_BINDING, '_semantic_pair_binding')",
                "candidate_plan_raw = read(run_dir / 'candidate-plan.json')",
                "need(sha(candidate_plan_raw) == child_plan_sha, 'semantic child plan changed')",
                "candidate_plan = json.loads(candidate_plan_raw)",
                "need(summary_path.exists(), 'runner summary required for policy and text counters')",
                f"semantic_result = semantic_evidence.verify(service_dir=candidate_output, capture_dir=run_dir / 'native' / 'text-history-candidate' / 'client-capture', plan=candidate_plan, terminal=terminal, audit_module=semantic_audit, binding_module=semantic_binding, baseline_sha256={template['baseline_sha256']!r}, runner_policy_calls=summary['policy_calls'], runner_text_calls=summary['text_calls'])",
                "need(semantic_result['status'] == 'provided_bytes_and_fp64_match', 'semantic parent evidence required')",
                "private(run_dir / 'semantic-evidence.json', (json.dumps(semantic_result, sort_keys=True, allow_nan=False) + '\\n').encode())",
            )
        ),
    )
    result = raw
    previous = len(raw) + 1
    for a, b, data in sorted(edits, reverse=True):
        _need(0 <= a <= b <= len(raw) and b <= previous, "overlapping actor edits")
        result = result[:a] + data + result[b:]
        previous = a
    _need(result.count(b"SOURCE_ONLY = True") == 1, "generated actor must remain held")
    compile(result, str(actor_path), "exec", dont_inherit=True)
    return result
