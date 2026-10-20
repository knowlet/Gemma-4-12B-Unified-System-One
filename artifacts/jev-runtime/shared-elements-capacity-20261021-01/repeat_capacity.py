"""Independent V2 paired trial using the sealed03 resource/lifecycle libraries.

Default dry. Unit03 must have a fresh invocation and this new output must not
exist. No native model, forward, HTTP or browser interface is present.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[2]
KIND = "jev_shared_elements_v2_paired_cpu_capacity_v1"
RAW, V2 = "raw-json-v1", "shared-elements-v2"
CORE = {
    "path": "artifacts/jev-runtime/shared-elements-capacity-20261020-01/paired_capacity.py",
    "sha256": "bc5aea1547ce483e9f94023e54cbff01fed9cf5ef4efdb13b9d1cbb9d2cf869f",
    "size_bytes": 39812,
}
CPU = {
    "path": "artifacts/jev-runtime/shared-elements-capacity-20261020-01/cpu_prepare_form_3g.py",
    "sha256": "a3da1fde8a2e542e622db249452d95b7b7ae0e5dee5b1eda4ad8d60128948e22",
    "size_bytes": 70678,
}


def core():
    path = ROOT / CORE["path"]
    blob = path.read_bytes()
    if (
        len(blob) != CORE["size_bytes"]
        or hashlib.sha256(blob).hexdigest() != CORE["sha256"]
        or any(p.is_symlink() for p in (path, *path.parents))
    ):
        raise ValueError("Frozen03 library source differs")
    spec = importlib.util.spec_from_file_location("_repeat_frozen03", path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = obj
    exec(compile(blob, str(path), "exec"), obj.__dict__)
    return obj


def make_plan(boot_id):
    c = core()
    cpu = c.module(CPU)
    return {
        "schema_id": KIND,
        "self_source": c.pin(Path(__file__)),
        "production_sources": {
            p: {k: v for k, v in c.pin(ROOT / p).items() if k != "path"}
            for p in sorted(c.PRODUCTION)
        },
        "unit": {"boot_id": boot_id, "cgroup_path": c.CGROUP},
        "output": str((HERE / "actual-01").relative_to(ROOT)),
        "python": c.PYTHON,
        "environment": cpu.EXPECTED_ENVIRONMENT,
        "resource_limits": c.LIMITS,
    }


def source_union(c, plan, old):
    selection = {
        "self_source": CORE,
        "cpu_source": CPU,
        "production_sources": plan["production_sources"],
    }
    files, technical, _ = c.sources(selection, old)
    files[plan["self_source"]["path"]] = {
        k: v for k, v in plan["self_source"].items() if k != "path"
    }
    return dict(sorted(files.items())), technical


def load_plan(path, expected):
    c = core()
    blob = c.read(path)
    c.require(type(expected) is str and c.sha(blob) == expected, "External repeat plan SHA")
    plan = json.loads(blob)
    c.require(
        set(plan)
        == {
            "schema_id",
            "self_source",
            "production_sources",
            "unit",
            "output",
            "python",
            "environment",
            "resource_limits",
        }
        and plan["schema_id"] == KIND,
        "Independent repeat plan",
    )
    c.require(
        ROOT / plan["self_source"]["path"] == Path(__file__).absolute(), "Exact new entry source"
    )
    c.checked(plan["self_source"])
    cpu, old = c.module(CPU), c.module(c.WORKER)
    cpu.unit_binding(plan["unit"])
    c.require(
        plan["python"] == c.PYTHON
        and plan["environment"] == cpu.EXPECTED_ENVIRONMENT
        and plan["resource_limits"] == c.LIMITS
        and set(plan["production_sources"]) == c.PRODUCTION
        and ROOT / plan["output"] == HERE / "actual-01",
        "Same resource contract and fresh separate namespace",
    )
    files, technical = source_union(c, plan, old)
    c.authenticate(files)
    c.checked(c.BASELINE)
    return c, plan, cpu, old, files, technical


def measure(c, payload, kind, mode, *, formatter, codec, **deps):
    if mode == RAW or kind == "helper":
        return c.measure(payload, kind, RAW, formatter=formatter, codec=codec, **deps)
    result = {
        "outcome": "UNKNOWN",
        "prefix": None,
        "error": None,
        "labels": None,
        "representation": None,
    }
    original, stage = c.wire(payload), "schema"
    try:
        deps["gate"].check("prepare", "cpu")
        text, source, labels, metadata = formatter(
            payload, "generated-actions-none", prompt_format=V2
        )
        c.require(
            source == original
            and metadata["requested_format"] == V2
            and metadata["format"] in {RAW, V2}
            and metadata["ordered_roundtrip_verified"] is True,
            "Explicit production requested/effective format and full source",
        )
        candidate, _ = codec.format_payload(payload, prompt_format=V2)
        c.require(
            c.wire(codec.unpack_payload(candidate)) == original,
            "V2 candidate complete ordered roundtrip",
        )
        candidate_sha = c.sha(candidate.encode())
        if metadata["format"] == RAW:
            raw_text, raw_source, raw_labels, _ = formatter(
                payload, "generated-actions-none", prompt_format=RAW
            )
            c.require(
                text == raw_text
                and source == raw_source
                and c.wire(labels) == c.wire(raw_labels)
                and metadata["encoded_sha256"] == c.sha(original)
                and metadata["candidate_shared_representation"]["encoded_sha256"] == candidate_sha,
                "Transparent raw fallback is exact original production text",
            )
        else:
            c.require(metadata["encoded_sha256"] == candidate_sha, "Selected V2 encoded SHA")
        sizes = metadata["candidate_user_text_bytes"]
        c.require(
            metadata["selection_basis"] == "complete_user_text_utf8_bytes"
            and type(sizes[RAW]) is int
            and type(sizes[V2]) is int
            and (metadata["format"] == V2) == (sizes[V2] < sizes[RAW]),
            "Explicit full-user-text byte selection",
        )
        result.update(labels=labels, representation=metadata, encoded=candidate)
        stage = "processor"
        prefix, prompt = c.prefix_from_text(deps["model"], text)
        deps["gate"].check("prepare", "cpu")
        deps["validator"].validate_prefix(prefix, prompt)
        result.update(prefix=prefix, prompt=prompt, outcome=deps["validator"].outcome(prefix, kind))
    except Exception as exc:
        from s1.errors import RequestValidationError

        result.update(
            outcome="REFUSED_SCHEMA"
            if stage == "schema" and isinstance(exc, RequestValidationError)
            else "UNKNOWN",
            error={"type": type(exc).__name__, "message": str(exc)[:2048]},
        )
    c.require(c.wire(payload) == original, "Complete input remains unchanged")
    return result


def retain(cpu, root, label, value):
    for key, suffix in (("prompt", "txt"), ("encoded", "json")):
        if key in value:
            value[key + "_ref"] = cpu.save(
                root, label + "-" + key + "." + suffix, value.pop(key).encode()
            )
    return value


def pairs(c, baseline, *, cpu, root, old, rows, **deps):
    for prior in baseline["rows"]:
        payload = json.loads(old.checked(prior["payload"], (ROOT / c.BASELINE["path"]).parent))
        values = {
            mode: retain(
                cpu,
                root,
                f"measurements/{prior['ordinal']:02d}-{mode}",
                measure(c, payload, prior["kind"], mode, **deps),
            )
            for mode in (RAW, V2)
        }
        raw, selected = values[RAW], values[V2]
        c.require(
            raw["prefix"] == prior["prefix"]
            and raw["outcome"] == prior["outcome"]
            and selected["outcome"] != "UNKNOWN"
            and c.wire(raw["labels"])
            == c.wire(selected["labels"])
            == c.wire(c.frozen_labels(prior)),
            "All75 frozen raw IDs/masks and full ordered heads",
        )
        if (
            prior["kind"] == "helper"
            or selected["prefix"] is not None
            and selected["representation"]["format"] == RAW
        ):
            c.require(
                selected["prefix"] == raw["prefix"]
                and selected["outcome"] == raw["outcome"]
                and selected["prompt_ref"]
                == {**raw["prompt_ref"], "path": selected["prompt_ref"]["path"]},
                "Raw fallback/helper full native prefix byte equality",
            )
        row = {
            "ordinal": prior["ordinal"],
            "id": prior["id"],
            "kind": prior["kind"],
            "provenance": prior["provenance"],
            "payload": cpu.save(root, f"payloads/{prior['ordinal']:02d}.json", c.wire(payload)),
            "measurements": values,
            "raw_matches_frozen": True,
        }
        cpu.save(root, f"rows/{prior['ordinal']:02d}.json", c.wire(row))
        rows.append(row)


def supplemental(c, *, cpu, root, rows, q, v8, **deps):
    def prefix(model, payload, kind, *, include_prompt=False):
        value = measure(c, payload, kind, V2, q=q, **deps)
        c.require(
            value["prefix"] is not None and value["error"] is None,
            "Actual new-format control prefix",
        )
        return (value["prefix"], value["prompt"]) if include_prompt else value["prefix"]

    facade = SimpleNamespace(fixture_payload=q.fixture_payload, _prefix=prefix)
    for helper, bands in ((False, (8128, 8161)), (True, (8065,))):
        selected = v8.scan(facade, deps["model"], deps["gate"], root, helper=helper)
        for band in bands:
            c.require(band in selected, "Original bounded grid reaches new-format band")
            payload, _ = selected[band]
            kind = "helper" if helper else "policy"
            values = {
                mode: retain(
                    cpu,
                    root,
                    f"controls/{band}-{mode}",
                    measure(c, payload, kind, mode, q=q, **deps),
                )
                for mode in (RAW, V2)
            }
            value = values[V2]
            c.require(
                value["outcome"] == ("ACCEPTED" if band == 8128 else "REFUSED_CONTEXT")
                and band <= value["prefix"]["token_count"] <= (8160 if band == 8128 else 8192),
                "Regenerated near/over context controls",
            )
            rows.append(
                {
                    "id": "compact-band-" + str(band),
                    "band": band,
                    "kind": kind,
                    "scope": "SUPPLEMENTAL_NEW_FORMAT_RECIPE_NOT_GENUINE_QUALITY",
                    "payload": cpu.save(root, f"controls/{band}-payload.json", c.wire(payload)),
                    "measurements": values,
                }
            )


def worker(c, plan, expected, nonce, cpu, old, files, technical):
    output = ROOT / plan["output"]
    intent = json.loads(c.read(output / "intent.json"))
    c.require(
        intent["nonce"] == nonce
        and intent["plan_sha256"] == expected
        and intent["parent"] == old.identity(os.getppid()),
        "Exact new parent nonce and identity",
    )
    c.require(
        cpu.child_argv(os.getppid())
        == [
            c.PYTHON,
            "-B",
            "-s",
            str(Path(__file__).absolute()),
            "--run",
            "--plan",
            intent["plan_path"],
            "--plan-sha256",
            expected,
        ],
        "Genuine new entry command, not old03 CLI",
    )
    child = old.identity(os.getpid())
    c.require(child["pid"] == child["pgid"] == child["sid"], "Fresh owned worker session")
    root = output / "worker"
    c.require(not root.exists(), "Fresh worker output")
    root.mkdir(mode=0o700)
    cpu.configure_limits("worker")
    gate = c.Gate(root, plan, cpu, intent["mem_total"])
    rows, controls, primary = [], [], None
    previous = signal.signal(
        signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Worker600s"))
    )
    signal.alarm(600)
    try:
        gate.check("startup", "cpu")
        gate.check("source-before", "cpu")
        c.authenticate(files)
        q, validator, v8 = (old.module(p) for p in (old.Q, old.VALIDATOR, old.V8))
        q._configure_environment("prepare")
        sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
        gate.check("runtime-before", "cpu")
        runtime = q._runtime(technical["technical"]["config"])
        processor = old.parse(old.checked(old.fixed(old.PROCESSOR)))
        refs = cpu.capture_reference(processor, root)
        model = cpu.load_processor(refs, gate)
        gate.check("processor-after", "cpu")
        before = c.wire(cpu.processor_evidence(model, refs))
        cpu.save(root, "processor-before.json", before)
        from s1 import jev_prompt
        from s1.jev_generated import format_action_request

        deps = dict(
            model=model,
            q=q,
            validator=validator,
            formatter=format_action_request,
            codec=jev_prompt,
            gate=gate,
        )
        pairs(c, json.loads(c.checked(c.BASELINE)), cpu=cpu, root=root, old=old, rows=rows, **deps)
        supplemental(c, cpu=cpu, root=root, rows=controls, v8=v8, **deps)
        c.authenticate(files, loaded=True)
        q._runtime(technical["technical"]["config"])
        after = c.wire(cpu.processor_evidence(model, refs))
        c.require(
            before == after and len(refs) == 58 and len(runtime["files"]) == 348,
            "Original processor58/runtime348 unchanged",
        )
        cpu.save(root, "processor-after.json", after)
        gate.check("final-cpu", "cpu")
    except BaseException as exc:
        primary = exc
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        result = {
            "schema_id": KIND + "_worker",
            "plan_sha256": expected,
            "nonce": nonce,
            "child": child,
            "complete": primary is None and len(rows) == 75 and len(controls) == 3,
            "primary_error": None
            if primary is None
            else {"type": type(primary).__name__, "message": str(primary)[:2048]},
            "rows": rows,
            "controls": controls,
            "sources": files,
            "processor_only": True,
            "native_model_forward_HTTP_browser_calls": 0,
            "task_quality_attested": False,
        }
        try:
            cpu.save(root, "terminal.json", c.wire(result), terminal=True)
        except BaseException as secondary:
            if primary is not None:
                primary.add_note("Terminal also failed: " + repr(secondary))
                raise primary from secondary
            raise
    if primary is not None:
        raise primary


def validate_values(
    c, row, payload, prior, *, root, old, validator, format_action_request, jev_prompt
):
    raw, chosen = row["measurements"][RAW], row["measurements"][V2]
    raw_prompt = None
    for mode, value in ((RAW, raw), (V2, chosen)):
        if value["prefix"] is None:
            c.require(
                prior is not None
                and prior["id"] == "native-click-width401"
                and value["outcome"] == "REFUSED_SCHEMA"
                and value["error"] is not None,
                "Only original401 schema refusal",
            )
            continue
        prompt = old.checked(value["prompt_ref"], root).decode()
        validator.validate_prefix(value["prefix"], prompt)
        c.require(
            value["outcome"] == validator.outcome(value["prefix"], row["kind"])
            and value["error"] is None,
            "Full IDs/mask and original32/128 context math",
        )
        if mode == RAW:
            raw_prompt = prompt
        if row["kind"] == "policy":
            text, source, labels, metadata = format_action_request(
                payload, "generated-actions-none", prompt_format=mode
            )
            c.require(
                source == c.wire(payload)
                and c.wire(labels) == c.wire(value["labels"])
                and value["representation"] == metadata,
                "Exact production effective format, full labels and source",
            )
            if mode == V2:
                candidate = old.checked(value["encoded_ref"], root).decode()
                c.require(
                    c.wire(jev_prompt.unpack_payload(candidate)) == source,
                    "Saved V2 candidate exact roundtrip even when falling back",
                )
                raw_text = format_action_request(
                    payload, "generated-actions-none", prompt_format=RAW
                )[0]
                c.require(
                    raw_prompt.count(raw_text) == 1
                    and raw_prompt.replace(raw_text, text, 1) == prompt,
                    "Exact native chat template and chosen production text",
                )
    c.require(c.wire(raw["labels"]) == c.wire(chosen["labels"]), "Full ordered target heads")
    if (
        row["kind"] == "helper"
        or chosen["prefix"] is not None
        and chosen["representation"]["format"] == RAW
    ):
        c.require(
            chosen["prefix"] == raw["prefix"] and chosen["outcome"] == raw["outcome"],
            "Raw fallback/helper exact full prefix equality",
        )
    if prior is not None:
        c.require(
            raw["prefix"] == prior["prefix"]
            and raw["outcome"] == prior["outcome"]
            and c.wire(raw["labels"]) == c.wire(c.frozen_labels(prior)),
            "Frozen raw baseline complete equality",
        )


def validate(c, plan, expected, nonce, cpu, old, files, output, cleanup):
    c.require(
        cleanup["confirmed"] is True and cleanup["remaining"] == [], "Whole owned exact cleanup"
    )
    root = output / "worker"
    result = json.loads(c.read(root / "terminal.json"))
    baseline = json.loads(c.checked(c.BASELINE))
    c.require(
        result["schema_id"] == KIND + "_worker"
        and result["plan_sha256"] == expected
        and result["nonce"] == nonce
        and result["complete"] is True
        and result["primary_error"] is None
        and len(result["rows"]) == 75
        and len(result["controls"]) == 3
        and result["sources"] == files
        and result["processor_only"] is True
        and result["native_model_forward_HTTP_browser_calls"] == 0
        and result["task_quality_attested"] is False,
        "Complete independent V2 trial",
    )
    child = result["child"]
    c.require(
        cleanup["tracked_PID_start_ticks"].get(str(child["pid"])) == child["start_ticks"]
        and child["pgid"] == child["sid"] == child["pid"]
        and child["pgid"] in cleanup["tracked_groups"],
        "Exact closed worker join",
    )
    from s1 import jev_prompt
    from s1.jev_generated import format_action_request

    validator = old.module(old.VALIDATOR)
    for row, prior in zip(result["rows"], baseline["rows"], strict=True):
        c.require(
            all(row[k] == prior[k] for k in ("ordinal", "id", "kind", "provenance"))
            and row["raw_matches_frozen"] is True
            and json.loads(c.read(root / f"rows/{row['ordinal']:02d}.json")) == row,
            "Exact75 ordered identity and saved rows",
        )
        raw = old.checked(row["payload"], root)
        c.require(
            raw == old.checked(prior["payload"], (ROOT / c.BASELINE["path"]).parent),
            "Original complete payload bytes",
        )
        validate_values(
            c,
            row,
            json.loads(raw),
            prior,
            root=root,
            old=old,
            validator=validator,
            format_action_request=format_action_request,
            jev_prompt=jev_prompt,
        )
    c.require(
        {r["band"] for r in result["controls"]} == {8128, 8161, 8065},
        "All3 separate new context controls",
    )
    for row in result["controls"]:
        band = row["band"]
        selected = row["measurements"][V2]
        c.require(
            row["id"] == "compact-band-" + str(band)
            and row["kind"] == ("helper" if band == 8065 else "policy")
            and row["scope"] == "SUPPLEMENTAL_NEW_FORMAT_RECIPE_NOT_GENUINE_QUALITY"
            and band <= selected["prefix"]["token_count"] <= (8160 if band == 8128 else 8192)
            and selected["outcome"] == ("ACCEPTED" if band == 8128 else "REFUSED_CONTEXT"),
            "Explicit new positive/negative controls",
        )
        validate_values(
            c,
            row,
            json.loads(old.checked(row["payload"], root)),
            None,
            root=root,
            old=old,
            validator=validator,
            format_action_request=format_action_request,
            jev_prompt=jev_prompt,
        )
    before = c.read(root / "processor-before.json")
    evidence = json.loads(before)
    c.require(
        before == c.read(root / "processor-after.json")
        and evidence["tokenizer_is_processor_tokenizer"] is True
        and evidence["max_context"] == 8192
        and evidence["generation_EOS"] == [1, 106, 50]
        and evidence["CPU_threads"] == evidence["CPU_interop_threads"] == 2,
        "Stable original processor/EOS/context/threads",
    )
    processor = old.parse(old.checked(old.fixed(old.PROCESSOR)))
    c.require(len(processor["reference_items"]) == 58, "Original58 processor source references")
    for label, item in processor["reference_items"].items():
        old.checked({"path": label, **item}, root / "reference-environment")
    gates = [json.loads(x) for x in c.read(root / "resource-gates.jsonl").splitlines()]
    intent = json.loads(c.read(output / "intent.json"))
    c.require(
        81 <= len(gates) <= 2048
        and [r["phase"] for r in gates[:5]]
        == [
            "startup",
            "source-before",
            "runtime-before",
            "cpu-reference-before-processor-load",
            "processor-after",
        ]
        and gates[-1]["phase"] == "final-cpu"
        and all(r["phase"] == "prepare" for r in gates[5:-1]),
        "Preserved bounded03 resource component phases",
    )
    previous, start = -math.inf, None
    for i, row in enumerate(gates):
        cpu.validate_unit_evidence(row["unit"], plan["unit"])
        formula = cpu.resource_formula(
            row["observation"], plan["unit"]["boot_id"], prospective=True
        )
        c.require(
            row["unit"]["caller"] == child
            and cpu.unit_identity(row["unit"]) == cpu.unit_identity(intent["unit_before"])
            and row["observation"]["MemTotal"] == intent["mem_total"],
            "Same unit/worker/physical memory",
        )
        c.require(
            type(row["monotonic"]) in (int, float)
            and math.isfinite(row["monotonic"])
            and row["monotonic"] >= previous
            and type(row["elapsed"]) in (int, float)
            and math.isfinite(row["elapsed"])
            and 0 <= row["elapsed"] < 600,
            "Finite same-origin worker clock",
        )
        start = row["monotonic"] - row["elapsed"] if start is None else start
        c.require(
            math.isclose(row["monotonic"] - row["elapsed"], start, rel_tol=0, abs_tol=1e-6),
            "One worker clock origin",
        )
        previous = row["monotonic"]
        c.require(
            row["schema_id"] == c.KIND + "_resource_gate"
            and type(row["sequence"]) is int
            and row["sequence"] == i
            and row["admitted"] is True
            and row["error"] is None
            and all(type(row[k]) is int and row[k] == n for k, n in formula.items())
            and row["observation"]["MemAvailable"] >= formula["required_bytes"]
            and row["limits"]
            == {"role": "worker", "soft_bytes": 8 * 1024**3, "hard_bytes": 8 * 1024**3}
            and row["cuda"]["CUDA_VISIBLE_DEVICES"] == ""
            and type(row["cuda"]["torch_loaded"]) is bool
            and row["cuda"]["CUDA_initialized"] is False,
            "Unchanged03 reserve/caps/AS/zero credit",
        )
    return {
        "rows": 75,
        "controls": 3,
        "raw_matches_frozen": True,
        "policy68_accepted": sum(
            r["measurements"][V2]["outcome"] == "ACCEPTED" for r in result["rows"][:68]
        ),
        "task_quality_attested": False,
    }


def parent(c, path, expected, plan, cpu, old, files):
    controller, common = c.module(c.CONTROLLER), old.module(old.COMMON)
    c.require(
        sys.executable == c.PYTHON
        and sys.version_info[:2] == (3, 13)
        and all(os.environ.get(k) == v for k, v in plan["environment"].items())
        and os.environ.get("JEV_CPU3G_BOOT") == plan["unit"]["boot_id"]
        and os.environ.get("JEV_CPU3G_CGROUP") == c.CGROUP,
        "Reviewed CPU runtime and existing03 unit",
    )
    if os.getpgrp() != os.getpid() or os.getsid(0) != os.getpid():
        os.setsid()
    before = cpu.assert_current_unit(plan["unit"])
    keys = (
        "InvocationID",
        "MainPID",
        "ControlGroup",
        "MemoryMax",
        "MemoryHigh",
        "MemorySwapMax",
        "OOMPolicy",
        "KillMode",
        "RuntimeMaxUSec",
        "TimeoutStopUSec",
        "Type",
    )
    response = subprocess.run(
        ["systemctl", "--user", "show", c.UNIT, *["--property=" + k for k in keys]],
        capture_output=True,
        timeout=5,
        check=True,
    )
    c.require(
        len(response.stdout) < 1024**2 and not response.stderr, "Bounded current unit properties"
    )
    props = dict(line.split("=", 1) for line in response.stdout.decode().splitlines())
    c.require(
        all(
            props[k] == v
            for k, v in {
                "MainPID": str(os.getpid()),
                "ControlGroup": c.CGROUP,
                "MemoryMax": str(3 * 1024**3),
                "MemoryHigh": "infinity",
                "MemorySwapMax": "0",
                "OOMPolicy": "kill",
                "KillMode": "control-group",
                "RuntimeMaxUSec": "11min",
                "TimeoutStopUSec": "10s",
                "Type": "exec",
            }.items()
        )
        and len(props["InvocationID"]) == 32,
        "Strict reused03 unit has fresh externally attested invocation",
    )
    obs = cpu.memory_observation()
    actor = {**old.identity(os.getpid()), "mem_total": obs["MemTotal"]}
    c.require(actor["pid"] == actor["pgid"] == actor["sid"], "Exact parent owned session")
    controller.prospective(cpu, plan["unit"]["boot_id"], actor["mem_total"])
    output = ROOT / plan["output"]
    c.require(not output.exists(), "Fresh new trial output")
    cpu.configure_limits("parent")
    output.mkdir(mode=0o700)
    nonce = uuid.uuid4().hex
    cpu.save(
        output,
        "intent.json",
        c.wire(
            {
                "parent": {k: actor[k] for k in ("pid", "start_ticks", "pgid", "sid")},
                "mem_total": actor["mem_total"],
                "nonce": nonce,
                "plan_path": str(path),
                "plan_sha256": expected,
                "unit_before": before,
                "properties": props,
                "sources": files,
            }
        ),
    )
    argv = [
        c.PYTHON,
        "-B",
        "-s",
        str(Path(__file__).absolute()),
        "--run",
        "--plan",
        str(path),
        "--plan-sha256",
        expected,
        "--worker",
        nonce,
    ]
    env = {k: os.environ[k] for k in cpu.SECRET_FREE_ENV if k in os.environ}
    env.update(plan["environment"])
    execution, monitor = controller.run_owned(
        cpu, common, output, argv, env, plan["unit"]["boot_id"], actor, wall=600
    )
    primary, review, after = execution["primary_error"], None, None
    try:
        c.require(
            primary is None
            and execution["returncode"] == 0
            and execution["secondary_errors"] == [],
            "Successful owned worker",
        )
        review = validate(c, plan, expected, nonce, cpu, old, files, output, execution["cleanup"])
        cpu.validate_monitor_evidence(
            c.read(output / "owned-memory-monitor.jsonl"),
            plan["unit"]["boot_id"],
            parent=actor,
            cleanup=execution["cleanup"],
            unit=plan["unit"],
        )
        c.authenticate(files, loaded=True)
        controller.prospective(cpu, plan["unit"]["boot_id"], actor["mem_total"])
        after = cpu.assert_current_unit(plan["unit"])
    except BaseException as exc:
        if primary is None:
            primary = {"type": type(exc).__name__, "message": str(exc)[:2048]}
    signal.alarm(0)
    result = {
        "schema_id": KIND + "_parent",
        "plan_sha256": expected,
        "nonce": nonce,
        "parent": actor,
        "InvocationID": props["InvocationID"],
        "execution": execution,
        "review": review,
        "primary_error": primary,
        "unit_after": after,
        "complete_pending_external_unit_exit": primary is None,
        "original95_qualification_attested": False,
        "task_quality_attested": False,
    }
    try:
        cpu.save(output, "parent-terminal.json", c.wire(result), terminal=True)
    except BaseException as secondary:
        if primary is not None:
            raise RuntimeError("Primary retained: " + str(primary)) from secondary
        raise
    c.require(primary is None, "Repeat trial failed; partial facts retained")


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--plan")
    parser.add_argument("--plan-sha256")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker")
    args = parser.parse_args(argv)
    if args.plan is None:
        if args.run or args.worker:
            raise ValueError("Explicit selected plan required")
        print(
            json.dumps(
                {
                    "schema_id": KIND,
                    "processor_calls": 0,
                    "status": "SOURCE_COMPONENT_ONLY_PLAN_REQUIRED",
                }
            )
        )
        return 0
    path = Path(args.plan).absolute()
    c, plan, cpu, old, files, technical = load_plan(path, args.plan_sha256)
    if not args.run:
        c.require(args.worker is None, "Dry cannot authorize worker")
        print(
            json.dumps(
                {
                    "schema_id": KIND,
                    "source_files": len(files),
                    "processor_calls": 0,
                    "output_exists": (ROOT / plan["output"]).exists(),
                }
            )
        )
        return 0
    if args.worker:
        worker(c, plan, args.plan_sha256, args.worker, cpu, old, files, technical)
    else:
        previous = signal.signal(
            signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Parent630s"))
        )
        signal.alarm(630)
        try:
            parent(c, path, args.plan_sha256, plan, cpu, old, files)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
