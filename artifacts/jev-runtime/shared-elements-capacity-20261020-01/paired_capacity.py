"""Default-dry paired CPU prompt trial; no native model, forward, HTTP or browser.

One unitMainPID owns one worker via the frozen V2 lifecycle. This independently
named trial retains the original75 payloads and adds3 separately named controls.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import signal
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[2]
KIND = "jev_shared_elements_paired_cpu_capacity_v1"
UNIT = "jev-cpu3g-20261020-03.service"
CGROUP = "/user.slice/user-1001.slice/user@1001.service/app.slice/" + UNIT
PYTHON = "/home/albert_hsieh/ComfyUI/.venv/bin/python"
V2 = ROOT / "artifacts/jev-runtime/public-revisit-capacity-3g-20261020-02"
WORKER = {
    "path": str((V2 / "revisit_capacity_worker.py").relative_to(ROOT)),
    "sha256": "9166eede7ffd8806b45fa8b1a66ec136551f4e6a462491d175c5dfbf9bfd80a3",
    "size_bytes": 28322,
}
CONTROLLER = {
    "path": str((V2 / "revisit_capacity_controller.py").relative_to(ROOT)),
    "sha256": "280500922695da0d37166dfe8da64bbf3b63bcffc7ea012edf4dbf2c580f6a7c",
    "size_bytes": 21598,
}
BASELINE = {
    "path": str((V2 / "actual-01/worker/terminal.json").relative_to(ROOT)),
    "sha256": "5ee726d2ec8707e22075d94451fbc46c1f990ced017f291e3802c40ce208ddfc",
    "size_bytes": 1744782,
}
PRODUCTION = {"src/s1/jev_generated.py", "src/s1/api.py", "src/s1/jev_prompt.py"}
REPLACED = PRODUCTION - {"src/s1/jev_prompt.py"}
LIMITS = {
    "unit_memory_max": 3 * 1024**3,
    "unit_swap_max": 0,
    "parent_AS_soft": 2 * 1024**3,
    "AS_hard": 8 * 1024**3,
    "worker_wall": 600,
    "parent_wall": 630,
    "unit_wall": 660,
    "output_bytes": 1024**3,
    "file_bytes": 64 * 1024**2,
}
_MODULES = {}


def require(value, message):
    if not value:
        raise ValueError(message)


def wire(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def read(path):
    path = Path(path).absolute()
    require(
        path.is_relative_to(ROOT)
        and ".." not in path.parts
        and not any(x.is_symlink() for x in (path, *path.parents)),
        "Ordinary repository path",
    )
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(before.st_mode) and before.st_size <= LIMITS["file_bytes"],
            "Bounded regular source/evidence file",
        )
        blob = stream.read(LIMITS["file_bytes"] + 1)
        after = os.fstat(stream.fileno())

    def sig(s):
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns

    require(
        sig(before) == sig(after) == sig(path.stat()) and len(blob) == before.st_size,
        "Stable complete file",
    )
    return blob


def pin(path):
    path = Path(path).absolute()
    blob = read(path)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(blob), "size_bytes": len(blob)}


def checked(ref):
    require(
        type(ref) is dict
        and set(ref) == {"path", "sha256", "size_bytes"}
        and type(ref["path"]) is str
        and not Path(ref["path"]).is_absolute()
        and type(ref["size_bytes"]) is int
        and ref["size_bytes"] >= 0,
        "Typed relative source/evidence ref",
    )
    require(pin(ROOT / ref["path"]) == ref, "Selected source/evidence pin differs")
    return read(ROOT / ref["path"])


def module(ref):
    blob = checked(ref)
    if ref["path"] not in _MODULES:
        name = "_paired_" + Path(ref["path"]).stem + "_" + ref["sha256"][:12]
        spec = importlib.util.spec_from_file_location(name, ROOT / ref["path"])
        obj = importlib.util.module_from_spec(spec)
        sys.modules[name] = obj
        exec(compile(blob, str(ROOT / ref["path"]), "exec"), obj.__dict__)
        _MODULES[ref["path"]] = obj
    return _MODULES[ref["path"]]


def make_plan(boot_id, production_sources=None):
    cpu = module(pin(HERE / "cpu_prepare_form_3g.py"))
    return {
        "schema_id": KIND,
        "self_source": pin(Path(__file__)),
        "cpu_source": pin(HERE / "cpu_prepare_form_3g.py"),
        "production_sources": production_sources
        if production_sources is not None
        else {
            p: {k: v for k, v in pin(ROOT / p).items() if k != "path"} for p in sorted(PRODUCTION)
        },
        "baseline_terminal": BASELINE,
        "output": str((HERE / "actual-01").relative_to(ROOT)),
        "unit": {"boot_id": boot_id, "cgroup_path": CGROUP},
        "python": PYTHON,
        "environment": cpu.EXPECTED_ENVIRONMENT,
        "resource_limits": LIMITS,
    }


def sources(plan, old):
    technical = old.parse(old.checked(old.fixed(old.TECHNICAL)))
    execution = old.parse(old.checked(technical["technical"]["execution"]))
    require(
        len(execution["files"]) == 95 and REPLACED <= set(execution["files"]),
        "Original95 retained only as prior authority",
    )
    files = dict(execution["files"])
    files.update(plan["production_sources"])
    cohort = old.parse(old.checked(old.fixed(old.COHORT)))
    for path, item in cohort["source_union"].items():
        absolute = (ROOT / path).absolute()
        require(
            absolute.is_relative_to(ROOT) and ".." not in absolute.parts,
            "Confined cohort source origin",
        )
        path = absolute.relative_to(ROOT).as_posix()
        if path in PRODUCTION:
            continue
        require(path not in files or files[path] == item, "No conflicting source authority")
        files[path] = item
    for ref in [
        WORKER,
        CONTROLLER,
        plan["self_source"],
        plan["cpu_source"],
        *[old.fixed(p) for p in old.PINS],
    ]:
        item = {k: v for k, v in ref.items() if k != "path"}
        require(ref["path"] not in files or files[ref["path"]] == item, "One current source pin")
        files[ref["path"]] = item
    return dict(sorted(files.items())), technical, cohort


def authenticate(files, *, loaded=False):
    for path, item in files.items():
        checked({"path": path, **item})
    if loaded:
        for name, obj in tuple(sys.modules.items()):
            if name.startswith("s1.") and getattr(obj, "__file__", None):
                origin = Path(obj.__file__).resolve()
                require(
                    origin.is_relative_to(ROOT) and origin.relative_to(ROOT).as_posix() in files,
                    "Loaded project origin belongs to new current source union",
                )


def load_plan(path, expected):
    blob = read(path)
    require(
        type(expected) is str and len(expected) == 64 and sha(blob) == expected,
        "Externally selected plan SHA",
    )
    plan = json.loads(blob)
    require(
        set(plan)
        == {
            "schema_id",
            "self_source",
            "cpu_source",
            "production_sources",
            "baseline_terminal",
            "output",
            "unit",
            "python",
            "environment",
            "resource_limits",
        }
        and plan["schema_id"] == KIND,
        "Strict separately named trial plan",
    )
    require(
        ROOT / plan["self_source"]["path"] == Path(__file__).absolute()
        and ROOT / plan["cpu_source"]["path"] == HERE / "cpu_prepare_form_3g.py",
        "Exact sibling source origins",
    )
    checked(plan["self_source"])
    cpu, old = module(plan["cpu_source"]), module(WORKER)
    require(
        plan["python"] == PYTHON
        and plan["environment"] == cpu.EXPECTED_ENVIRONMENT
        and plan["resource_limits"] == LIMITS,
        "Unchanged CPU resource/runtime contract",
    )
    cpu.unit_binding(plan["unit"])
    require(
        set(plan["production_sources"]) == PRODUCTION
        and plan["baseline_terminal"] == BASELINE
        and ROOT / plan["output"] == HERE / "actual-01",
        "Explicit new production authority and fixed original75 baseline",
    )
    checked(BASELINE)
    files, technical, cohort = sources(plan, old)
    authenticate(files)
    return plan, cpu, old, files, technical, cohort


class Gate:
    def __init__(self, root, plan, cpu, mem_total):
        self.root, self.plan, self.cpu = root, plan, cpu
        self.mem_total = mem_total
        self.start, self.sequence, self.used = time.monotonic(), 0, 0
        cpu.save(root, "resource-gates.jsonl", b"")

    def check(self, phase, scope):
        require(
            scope == "cpu"
            and phase
            in {
                "startup",
                "source-before",
                "runtime-before",
                "cpu-reference-before-processor-load",
                "processor-after",
                "prepare",
                "final-cpu",
            }
            and self.sequence < 2048,
            "Bounded typed CPU phase",
        )
        now = time.monotonic()
        row = {
            "schema_id": KIND + "_resource_gate",
            "sequence": self.sequence,
            "phase": phase,
            "monotonic": now,
            "elapsed": now - self.start,
            "admitted": False,
            "error": None,
        }
        primary = None
        try:
            require(row["elapsed"] < 600, "Worker600s cooperative deadline")
            obs = self.cpu.memory_observation()
            require(obs["MemTotal"] == self.mem_total, "Stable physical memory identity")
            row.update(
                observation=obs,
                **self.cpu.resource_formula(obs, self.plan["unit"]["boot_id"], prospective=True),
            )
            row["unit"] = self.cpu.assert_current_unit(self.plan["unit"])
            row["limits"], row["cuda"] = self.cpu.verify_limits("worker"), self.cpu.cuda_state()
            require(
                obs["MemAvailable"] >= row["required_bytes"], "Same reserve plus prospective3GiB"
            )
            row["admitted"] = True
        except BaseException as exc:
            primary, row["error"] = exc, {"type": type(exc).__name__, "message": str(exc)[:2048]}
        try:
            self.cpu.caps(self.root.parent)
            self.used = self.cpu.append_owned(self.root, "resource-gates.jsonl", row, self.used)
            self.sequence += 1
        except BaseException as secondary:
            if primary is not None:
                primary.add_note("Gate persistence also failed: " + repr(secondary))
                raise primary from secondary
            raise
        if primary is not None:
            raise primary
        return row


def prefix_from_text(model, text):
    prompt = model.processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": text}]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = dict(model.processor(text=prompt, return_tensors="pt", add_special_tokens=False))
    ids, mask = inputs["input_ids"][0].tolist(), inputs["attention_mask"][0].tolist()
    return {
        "prompt_sha256": sha(prompt.encode()),
        "ids": ids,
        "attention_mask": mask,
        "token_count": len(ids),
    }, prompt


def measure(payload, kind, mode, *, model, q, validator, formatter, codec, gate):
    original = wire(payload)
    row = {
        "outcome": "UNKNOWN",
        "prefix": None,
        "error": None,
        "labels": None,
        "representation": None,
    }
    stage = "schema"
    try:
        gate.check("prepare", "cpu")
        if kind == "policy":
            text, source, labels, meta = formatter(
                payload, "generated-actions-none", prompt_format=mode
            )
            require(
                source == original
                and meta["ordered_roundtrip_verified"] is True
                and meta["format"] == mode,
                "Full ordered original request",
            )
            row.update(labels=labels, representation=meta)
            if mode == "shared-elements-v1":
                packed, _ = codec.format_payload(payload, prompt_format=mode)
                require(
                    wire(codec.unpack_payload(packed)) == original
                    and sha(packed.encode()) == meta["encoded_sha256"],
                    "Independently verified ordered shared roundtrip",
                )
                row["encoded"] = packed
            stage = "processor"
            prefix, prompt = prefix_from_text(model, text)
        else:
            q.parse_helper_request(payload, system_prompt=q.TEXT_VALUE)
            stage = "processor"
            prefix, prompt = q._prefix(model, payload, "helper", include_prompt=True)
            row["representation"] = {
                "format": "unchanged-helper-v1",
                "ordered_roundtrip_verified": True,
            }
        gate.check("prepare", "cpu")
        validator.validate_prefix(prefix, prompt)
        row.update(prefix=prefix, prompt=prompt, outcome=validator.outcome(prefix, kind))
    except Exception as exc:
        from s1.errors import RequestValidationError

        row.update(
            outcome="REFUSED_SCHEMA"
            if stage == "schema" and isinstance(exc, RequestValidationError)
            else "UNKNOWN",
            error={"type": type(exc).__name__, "message": str(exc)[:2048]},
        )
    require(wire(payload) == original, "No input mutation")
    return row


def frozen_labels(row):
    labels = row.get("offered_candidate_keys")
    if labels is None:
        return None
    order = row["offered_head_order"]
    require(
        type(order) is list and len(set(order)) == len(order) and set(order) == set(labels),
        "Frozen explicit head order",
    )
    return {head: labels[head] for head in order}


def paired_rows(baseline, *, input_root, cpu, root, rows=None, **deps):
    rows = [] if rows is None else rows
    for oldrow in baseline["rows"]:
        payload = json.loads(deps["old"].checked(oldrow["payload"], input_root))
        pairs = {}
        for mode in ("raw-json-v1", "shared-elements-v1"):
            result = measure(
                payload, oldrow["kind"], mode, **{k: v for k, v in deps.items() if k != "old"}
            )
            if "prompt" in result:
                result["prompt_ref"] = cpu.save(
                    root,
                    f"prompts/{oldrow['ordinal']:02d}-{mode}.txt",
                    result.pop("prompt").encode(),
                )
            if "encoded" in result:
                result["encoded_ref"] = cpu.save(
                    root, f"encoded/{oldrow['ordinal']:02d}.json", result.pop("encoded").encode()
                )
            pairs[mode] = result
        raw, compact = pairs["raw-json-v1"], pairs["shared-elements-v1"]
        require(
            raw["prefix"] == oldrow["prefix"] and raw["outcome"] == oldrow["outcome"],
            "Actual raw IDs/mask/prompt/outcome match frozen75 baseline",
        )
        require(
            compact["outcome"] != "UNKNOWN" and wire(raw["labels"]) == wire(compact["labels"]),
            "Complete compact measurement and unchanged ordered full heads",
        )
        require(
            wire(raw["labels"]) == wire(frozen_labels(oldrow)),
            "Complete original ordered head keys",
        )
        item = {
            "ordinal": oldrow["ordinal"],
            "id": oldrow["id"],
            "kind": oldrow["kind"],
            "provenance": oldrow["provenance"],
            "payload": cpu.save(root, f"payloads/{oldrow['ordinal']:02d}.json", wire(payload)),
            "raw_matches_frozen": True,
            "measurements": pairs,
        }
        cpu.save(root, f"rows/{oldrow['ordinal']:02d}.json", wire(item))
        rows.append(item)
    return rows


def controls(*, cpu, root, old, q, v8, model, validator, formatter, codec, gate, rows=None):
    common = dict(
        model=model, q=q, validator=validator, formatter=formatter, codec=codec, gate=gate
    )
    facade = SimpleNamespace(
        fixture_payload=q.fixture_payload,
        _prefix=lambda m, p, k, include_prompt=False: _control_prefix(p, k, common, include_prompt),
    )
    rows = [] if rows is None else rows
    for helper, bands in ((False, (8128, 8161)), (True, (8065,))):
        selected = v8.scan(facade, model, gate, root, helper=helper)
        for band in bands:
            require(band in selected, "New compact literal context band reached")
            payload, (prefix, prompt) = selected[band]
            kind = "helper" if helper else "policy"
            outcome = validator.outcome(prefix, kind)
            require(
                outcome == ("ACCEPTED" if band == 8128 else "REFUSED_CONTEXT"),
                "Regenerated compact context negative controls",
            )
            rows.append(
                {
                    "id": "compact-band-" + str(band),
                    "band": band,
                    "kind": kind,
                    "outcome": outcome,
                    "payload": cpu.save(root, f"controls/{band}-payload.json", wire(payload)),
                    "prefix": prefix,
                    "prompt": cpu.save(root, f"controls/{band}-prompt.txt", prompt.encode()),
                    "scope": "SUPPLEMENTAL_NEW_FORMAT_RECIPE_NOT_GENUINE_QUALITY",
                }
            )
    return rows


def _control_prefix(payload, kind, deps, include_prompt):
    result = measure(payload, kind, "shared-elements-v1", **deps)
    require(
        result["prefix"] is not None and result["error"] is None, "Actual compact prefix control"
    )
    return (result["prefix"], result["prompt"]) if include_prompt else result["prefix"]


def run_worker(plan, expected, nonce, cpu, old, files, technical):
    output = ROOT / plan["output"]
    intent = json.loads(read(output / "intent.json"))
    require(
        intent["nonce"] == nonce
        and intent["plan_sha256"] == expected
        and intent["parent"] == old.identity(os.getppid()),
        "Exact live parent identity and nonce",
    )
    require(
        old.identity(os.getpid())["pgid"] == old.identity(os.getpid())["sid"] == os.getpid(),
        "Owned fresh worker PG/SID",
    )
    parent_args = cpu.child_argv(os.getppid())
    require(
        parent_args
        == [
            PYTHON,
            "-B",
            "-s",
            str(Path(__file__).absolute()),
            "--run",
            "--plan",
            intent["plan_path"],
            "--plan-sha256",
            expected,
        ],
        "Exact bound parent command",
    )
    root = output / "worker"
    require(not root.exists(), "Fresh worker subtree")
    root.mkdir(mode=0o700)
    cpu.configure_limits("worker")
    gate, primary, rows, extra = Gate(root, plan, cpu, intent["mem_total"]), None, [], []
    previous = signal.signal(
        signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Worker600s"))
    )
    signal.alarm(600)
    try:
        gate.check("startup", "cpu")
        gate.check("source-before", "cpu")
        authenticate(files)
        q, validator, v8 = (old.module(p) for p in (old.Q, old.VALIDATOR, old.V8))
        q._configure_environment("prepare")
        sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
        gate.check("runtime-before", "cpu")
        runtime = q._runtime(technical["technical"]["config"])
        processor_plan = old.parse(old.checked(old.fixed(old.PROCESSOR)))
        refs = cpu.capture_reference(processor_plan, root)
        model = cpu.load_processor(refs, gate)
        gate.check("processor-after", "cpu")
        before = wire(cpu.processor_evidence(model, refs))
        cpu.save(root, "processor-before.json", before)
        from s1 import jev_prompt
        from s1.jev_generated import format_action_request

        baseline = json.loads(checked(BASELINE))
        paired_rows(
            baseline,
            input_root=(ROOT / BASELINE["path"]).parent,
            cpu=cpu,
            root=root,
            old=old,
            q=q,
            validator=validator,
            model=model,
            gate=gate,
            formatter=format_action_request,
            codec=jev_prompt,
            rows=rows,
        )
        controls(
            cpu=cpu,
            root=root,
            old=old,
            q=q,
            v8=v8,
            model=model,
            validator=validator,
            formatter=format_action_request,
            codec=jev_prompt,
            gate=gate,
            rows=extra,
        )
        authenticate(files, loaded=True)
        q._runtime(technical["technical"]["config"])
        after = wire(cpu.processor_evidence(model, refs))
        require(
            before == after and len(runtime["files"]) == 348 and len(refs) == 58,
            "Unchanged actual processor/runtime components",
        )
        cpu.save(root, "processor-after.json", after)
        cpu.save(
            root,
            "runtime-component.json",
            wire(
                {
                    "runtime_ref": technical["technical"]["runtime"],
                    "runtime_files": 348,
                    "processor_plan": old.fixed(old.PROCESSOR),
                    "reference_count": 58,
                    "original95_current_qualification_attested": False,
                }
            ),
        )
        cpu.save(
            root,
            "source-union.json",
            wire(
                {
                    "schema_id": KIND + "_current_sources",
                    "files": files,
                    "count": len(files),
                    "prior95_qualification_attested": False,
                }
            ),
        )
        gate.check("final-cpu", "cpu")
    except BaseException as exc:
        primary = exc
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
        result = {
            "schema_id": KIND + "_worker",
            "nonce": nonce,
            "plan_sha256": expected,
            "child": old.identity(os.getpid()),
            "complete": primary is None and len(rows) == 75 and len(extra) == 3,
            "primary_error": None
            if primary is None
            else {"type": type(primary).__name__, "message": str(primary)[:2048]},
            "rows": rows,
            "supplemental_controls": extra,
            "processor_only": True,
            "native_model_forward_HTTP_browser_calls": 0,
            "task_quality_attested": False,
        }
        try:
            cpu.save(root, "terminal.json", wire(result), terminal=True)
        except BaseException as secondary:
            if primary is not None:
                primary.add_note("Worker terminal failed: " + repr(secondary))
                raise primary from secondary
            raise
    if primary is not None:
        raise primary
    return result


def validate_result(plan, expected, nonce, cpu, old, output, cleanup):
    require(cleanup["confirmed"] is True and cleanup["remaining"] == [], "Whole owned cleanup")
    root = output / "worker"
    result = json.loads(read(root / "terminal.json"))
    baseline = json.loads(checked(BASELINE))
    require(
        result["schema_id"] == KIND + "_worker"
        and result["plan_sha256"] == expected
        and result["nonce"] == nonce
        and result["complete"] is True
        and result["primary_error"] is None
        and len(result["rows"]) == 75
        and len(result["supplemental_controls"]) == 3,
        "Complete independently typed paired trial",
    )
    require(
        result["processor_only"] is True
        and result["native_model_forward_HTTP_browser_calls"] == 0
        and result["task_quality_attested"] is False,
        "Processor-only scoped evidence",
    )
    child = result["child"]
    require(
        cleanup["tracked_PID_start_ticks"].get(str(child["pid"])) == child["start_ticks"]
        and child["pgid"] in cleanup["tracked_groups"]
        and child["pgid"] == child["sid"] == child["pid"],
        "Closed exact worker join",
    )
    validator = old.module(old.VALIDATOR)
    codec = module(
        {"path": "src/s1/jev_prompt.py", **plan["production_sources"]["src/s1/jev_prompt.py"]}
    )
    instructions = next(
        ast.literal_eval(n.value)
        for n in ast.parse(read(ROOT / "src/s1/jev_generated.py")).body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "INSTRUCTIONS" for t in n.targets)
    )
    for row, prior in zip(result["rows"], baseline["rows"], strict=True):
        require(
            all(row[k] == prior[k] for k in ("ordinal", "id", "kind", "provenance"))
            and row["raw_matches_frozen"] is True,
            "Exact full75 pair identity/provenance",
        )
        require(
            json.loads(read(root / f"rows/{row['ordinal']:02d}.json")) == row, "Saved row equality"
        )
        original = old.checked(prior["payload"], (ROOT / BASELINE["path"]).parent)
        require(
            old.checked(row["payload"], root) == original, "All original payload bytes retained"
        )
        raw, shared = (row["measurements"][mode] for mode in ("raw-json-v1", "shared-elements-v1"))
        require(
            raw["prefix"] == prior["prefix"]
            and raw["outcome"] == prior["outcome"]
            and wire(raw["labels"]) == wire(shared["labels"]) == wire(frozen_labels(prior)),
            "Frozen raw IDs/masks and full ordered heads",
        )
        for value in (raw, shared):
            if value["prefix"] is not None:
                prompt = old.checked(value["prompt_ref"], root).decode()
                validator.validate_prefix(value["prefix"], prompt)
                require(
                    value["outcome"] == validator.outcome(value["prefix"], row["kind"])
                    and value["error"] is None
                    and value["representation"]["ordered_roundtrip_verified"] is True,
                    "Complete actual prefix and outcome",
                )
            else:
                require(
                    value["outcome"] == "REFUSED_SCHEMA" and row["id"] == "native-click-width401",
                    "Only preserved401 schema refusal",
                )
        if row["kind"] == "policy" and shared["prefix"] is not None:
            encoded = old.checked(shared["encoded_ref"], root).decode()
            require(
                wire(codec.unpack_payload(encoded)) == original
                and sha(encoded.encode()) == shared["representation"]["encoded_sha256"],
                "Saved shared encoding exact roundtrip",
            )
            raw_prompt = old.checked(raw["prompt_ref"], root).decode()
            raw_text = instructions + "\n" + original.decode()
            shared_text = instructions + "\n" + codec.SHARED_ELEMENT_INSTRUCTIONS + "\n" + encoded
            require(
                raw_prompt.count(raw_text) == 1
                and raw_prompt.replace(raw_text, shared_text, 1)
                == old.checked(shared["prompt_ref"], root).decode(),
                "Same genuine chat template and exact production user text",
            )
    require(
        {row["band"] for row in result["supplemental_controls"]} == {8128, 8161, 8065},
        "Three distinct new context controls",
    )
    for row in result["supplemental_controls"]:
        band = row["band"]
        require(
            row["id"] == "compact-band-" + str(band)
            and row["kind"] == ("helper" if band == 8065 else "policy")
            and row["scope"] == "SUPPLEMENTAL_NEW_FORMAT_RECIPE_NOT_GENUINE_QUALITY"
            and band <= row["prefix"]["token_count"] <= (8160 if band == 8128 else 8192),
            "Exact new context control identity, range and scope",
        )
        require(
            row["band"] in {8128, 8161, 8065}
            and row["outcome"] == ("ACCEPTED" if row["band"] == 8128 else "REFUSED_CONTEXT"),
            "Separate regenerated negative controls",
        )
        validator.validate_prefix(row["prefix"], old.checked(row["prompt"], root).decode())
        require(
            row["outcome"] == validator.outcome(row["prefix"], row["kind"]),
            "New control context math",
        )
        old.checked(row["payload"], root)
    before = read(root / "processor-before.json")
    evidence = json.loads(before)
    require(
        before == read(root / "processor-after.json")
        and evidence["tokenizer_is_processor_tokenizer"] is True
        and evidence["max_context"] == 8192
        and evidence["generation_EOS"] == [1, 106, 50]
        and evidence["CPU_threads"] == evidence["CPU_interop_threads"] == 2,
        "Processor provenance and unchanged context/EOS/threads",
    )
    processor = old.parse(old.checked(old.fixed(old.PROCESSOR)))
    require(len(processor["reference_items"]) == 58, "Original58 references")
    for label, item in processor["reference_items"].items():
        old.checked({"path": label, **item}, root / "reference-environment")
    files, technical, _ = sources(plan, old)
    require(
        json.loads(read(root / "source-union.json"))
        == {
            "schema_id": KIND + "_current_sources",
            "files": files,
            "count": len(files),
            "prior95_qualification_attested": False,
        },
        "Genuine new current source authority",
    )
    require(
        json.loads(read(root / "runtime-component.json"))
        == {
            "runtime_ref": technical["technical"]["runtime"],
            "runtime_files": 348,
            "processor_plan": old.fixed(old.PROCESSOR),
            "reference_count": 58,
            "original95_current_qualification_attested": False,
        },
        "Original348/58 components remain independent",
    )
    gates = [json.loads(x) for x in read(root / "resource-gates.jsonl").splitlines()]
    require(
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
        "Complete typed phase journal",
    )
    intent = json.loads(read(output / "intent.json"))
    previous, start = -math.inf, None
    for i, row in enumerate(gates):
        cpu.validate_unit_evidence(row["unit"], plan["unit"])
        formula = cpu.resource_formula(
            row["observation"], plan["unit"]["boot_id"], prospective=True
        )
        require(
            row["unit"]["caller"] == child
            and row["observation"]["MemTotal"] == intent["mem_total"],
            "Same exact worker and physical memory",
        )
        require(
            type(row["monotonic"]) in (int, float)
            and math.isfinite(row["monotonic"])
            and row["monotonic"] >= previous
            and type(row["elapsed"]) in (int, float)
            and math.isfinite(row["elapsed"])
            and 0 <= row["elapsed"] < 600,
            "Finite cooperative ordered worker clock",
        )
        start = row["monotonic"] - row["elapsed"] if start is None else start
        require(
            math.isclose(row["monotonic"] - row["elapsed"], start, rel_tol=0, abs_tol=1e-6),
            "One worker clock origin",
        )
        previous = row["monotonic"]
        require(
            row["schema_id"] == KIND + "_resource_gate"
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
            "Original reserve/caps/AS/zero-credit actual gates",
        )
    return {
        "rows": 75,
        "supplemental_controls": 3,
        "raw_matches_frozen": True,
        "compact_policy68_accepted": sum(
            r["measurements"]["shared-elements-v1"]["outcome"] == "ACCEPTED"
            for r in result["rows"][:68]
        ),
        "task_quality_attested": False,
    }


def run_parent(path, expected, plan, cpu, old, files):
    controller, common = module(CONTROLLER), old.module(old.COMMON)
    require(
        sys.executable == PYTHON
        and sys.version_info[:2] == (3, 13)
        and all(os.environ.get(k) == v for k, v in plan["environment"].items()),
        "Exact reviewed CPU environment",
    )
    require(
        os.environ.get("JEV_CPU3G_BOOT") == plan["unit"]["boot_id"]
        and os.environ.get("JEV_CPU3G_CGROUP") == CGROUP,
        "Explicit original unit environment",
    )
    if os.getpgrp() != os.getpid() or os.getsid(0) != os.getpid():
        os.setsid()
    before = cpu.assert_current_unit(plan["unit"])
    properties = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            UNIT,
            *[
                "--property=" + k
                for k in (
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
            ],
        ],
        capture_output=True,
        timeout=5,
        check=True,
    )
    require(len(properties.stdout) < 1024**2 and not properties.stderr, "Bounded unit readback")
    props = dict(line.split("=", 1) for line in properties.stdout.decode().splitlines())
    require(
        all(
            props[k] == v
            for k, v in {
                "MainPID": str(os.getpid()),
                "ControlGroup": CGROUP,
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
        "Same strict ownunit properties",
    )
    obs = cpu.memory_observation()
    parent = {**old.identity(os.getpid()), "mem_total": obs["MemTotal"]}
    require(parent["pid"] == parent["pgid"] == parent["sid"], "Unit parent owns fresh session")
    controller.prospective(cpu, plan["unit"]["boot_id"], parent["mem_total"])
    output = ROOT / plan["output"]
    require(not output.exists(), "Fresh actual namespace")
    cpu.configure_limits("parent")
    output.mkdir(mode=0o700)
    nonce = uuid.uuid4().hex
    cpu.save(
        output,
        "intent.json",
        wire(
            {
                "nonce": nonce,
                "plan_sha256": expected,
                "plan_path": str(path),
                "parent": {k: parent[k] for k in ("pid", "start_ticks", "pgid", "sid")},
                "mem_total": parent["mem_total"],
                "unit_before": before,
                "properties": props,
                "sources": files,
            }
        ),
    )
    argv = [
        PYTHON,
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
        cpu, common, output, argv, env, plan["unit"]["boot_id"], parent, wall=600
    )
    primary, review, after = execution["primary_error"], None, None
    try:
        require(
            primary is None
            and execution["returncode"] == 0
            and execution["secondary_errors"] == [],
            "Successful owned worker execution",
        )
        review = validate_result(plan, expected, nonce, cpu, old, output, execution["cleanup"])
        cpu.validate_monitor_evidence(
            read(output / "owned-memory-monitor.jsonl"),
            plan["unit"]["boot_id"],
            parent=parent,
            cleanup=execution["cleanup"],
            unit=plan["unit"],
        )
        authenticate(files)
        controller.prospective(cpu, plan["unit"]["boot_id"], parent["mem_total"])
        after = cpu.assert_current_unit(plan["unit"])
    except BaseException as exc:
        if primary is None:
            primary = {"type": type(exc).__name__, "message": str(exc)[:2048]}
    signal.alarm(0)
    terminal = {
        "schema_id": KIND + "_parent",
        "plan_sha256": expected,
        "nonce": nonce,
        "parent": parent,
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
        cpu.save(output, "parent-terminal.json", wire(terminal), terminal=True)
    except BaseException as secondary:
        if primary is not None:
            raise RuntimeError("Primary retained: " + str(primary)) from secondary
        raise
    require(primary is None, "Paired trial failed; retain partial evidence")
    return terminal


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--plan")
    parser.add_argument("--plan-sha256")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker")
    args = parser.parse_args(argv)
    if args.plan is None:
        require(not args.run and args.worker is None, "Explicit externally selected plan required")
        print(
            json.dumps(
                {
                    "schema_id": KIND,
                    "status": "SOURCE_COMPONENT_ONLY_PLAN_REQUIRED",
                    "processor_calls": 0,
                }
            )
        )
        return 0
    path = Path(args.plan).absolute()
    plan, cpu, old, files, technical, _ = load_plan(path, args.plan_sha256)
    if not args.run:
        require(args.worker is None, "Worker cannot dry-authorize execution")
        print(
            json.dumps(
                {
                    "schema_id": KIND,
                    "source_files": len(files),
                    "output_exists": (ROOT / plan["output"]).exists(),
                    "processor_calls": 0,
                }
            )
        )
        return 0
    if args.worker:
        run_worker(plan, args.plan_sha256, args.worker, cpu, old, files, technical)
    else:
        previous = signal.signal(
            signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Parent630s"))
        )
        signal.alarm(630)
        try:
            run_parent(path, args.plan_sha256, plan, cpu, old, files)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
