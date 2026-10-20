"""Default-dry owned CPU form admission/reference worker and controller.

No browser, native toolchain, base model, forward, HTTP or Modal interface.
Actual execution requires an externally selected plan and explicit --run.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import os
import signal
import stat
import subprocess
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GIB, MIB = 1024**3, 1024**2
MAX_BYTES, MAX_FILE, RESERVE = GIB, 64 * MIB, MIB
SCHEMA = "owned-form-progress-CPU-launch-v1"
SOURCE_SCHEMA = "owned-form-progress-CPU-source-union-v1"
SUCCESS = "COMPLETE_CPU_FORM_REFERENCE_PENDING_EXTERNAL_CONTROLLER_EXIT"
SECRET_FREE_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR")


def require(value, message):
    if not value:
        raise ValueError(message)


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def parse(blob):
    def pairs(values):
        out = {}
        for key, value in values:
            require(key not in out, "Duplicate CPU preparation JSON key")
            out[key] = value
        return out

    return json.loads(
        blob, object_pairs_hook=pairs, parse_constant=lambda _: require(False, "Nonfinite JSON")
    )


def ordinary(path):
    path = Path(path).absolute()
    require(
        ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "Ordinary CPU preparation path",
    )
    return path


def read(path, maximum=MAX_FILE):
    path = ordinary(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(before.st_mode) and before.st_size <= maximum,
            "Bounded ordinary CPU evidence",
        )
        blob = stream.read(before.st_size + 1)
        after = os.fstat(stream.fileno())
    require(
        len(blob) == before.st_size
        and (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
        "CPU evidence changed while read",
    )
    return blob


def checked(ref):
    require(
        type(ref) is dict and set(ref) == {"path", "sha256", "size_bytes"},
        "Typed CPU input reference",
    )
    p = Path(ref["path"])
    require(not p.is_absolute() and ".." not in p.parts, "Confined CPU input reference")
    blob = read(ROOT / p)
    require(
        len(blob) == ref["size_bytes"] and hashlib.sha256(blob).hexdigest() == ref["sha256"],
        "CPU input/source drift: " + ref["path"],
    )
    return blob


def caps(output, pending=0, *, terminal=False):
    files = [p for p in output.rglob("*") if p.is_file()]
    require(
        len(files) < 10000
        and all(not p.is_symlink() and p.stat().st_size <= MAX_FILE for p in files),
        "CPU file count/per-file cap",
    )
    require(
        sum(p.stat().st_size for p in files) + pending <= MAX_BYTES - (0 if terminal else RESERVE),
        "CPU total evidence/terminal reserve",
    )


def save(output, label, blob, *, terminal=False):
    require(type(blob) is bytes and len(blob) <= MAX_FILE, "CPU output file cap")
    path = ordinary(output / label)
    require(path.is_relative_to(output), "Confined CPU output")
    caps(output, len(blob), terminal=terminal)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        require(stream.write(blob) == len(blob), "Complete CPU evidence write")
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": label, "sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)}


def error_value(error, phase):
    return {"type": type(error).__name__, "message": str(error)[:2048], "phase": phase}


def terminal(output, label, value, secondary):
    try:
        save(output, label + ".pending", encode(value) + b"\n", terminal=True)
        require(not (output / label).exists(), "Fresh CPU terminal")
        (output / (label + ".pending")).rename(output / label)
        return True
    except BaseException as error:
        secondary.append(error_value(error, "terminal-publication"))
        return False


def report_unknown(status, primary, secondary):
    try:
        print(
            encode(
                {"status": status, "primary_error": primary, "secondary_errors": secondary}
            ).decode(),
            file=sys.stderr,
        )
    except BaseException:
        pass


def load_plan(path, expected):
    blob = read(path, 2 * MIB)
    require(hashlib.sha256(blob).hexdigest() == expected, "Externally selected CPU launch plan")
    plan = parse(blob)
    require(
        plan["schema_id"] == SCHEMA and plan["source_schema_id"] == SOURCE_SCHEMA,
        "Separate CPU form execution kind",
    )
    require(
        plan["python"] == "/home/albert_hsieh/ComfyUI/.venv/bin/python"
        and plan["caps"]
        == {
            "seconds": 600,
            "bytes": MAX_BYTES,
            "file_bytes": MAX_FILE,
            "terminal_reserve_bytes": RESERVE,
            "stdout_or_stderr_bytes": MIB,
            "TERM_KILL_grace_each_seconds": 2,
        },
        "Fixed CPU resources/watchdog",
    )
    require(
        len(plan["ordered_cases"]) == 23
        and len({r["id"] for r in plan["ordered_cases"]}) == 23
        and [r["split"] for r in plan["ordered_cases"]].count("train") == 13
        and [r["split"] for r in plan["ordered_cases"]].count("development") == 10,
        "Fixed complete23 ordered references",
    )
    handoff = parse(checked(plan["source_handoff"]))
    admission = parse(checked(plan["admission_plan"]))
    require(
        handoff["schema_id"] == SOURCE_SCHEMA
        and len(handoff["files"]) == 105
        and set(handoff["files"])
        == set(admission["sources"]) | {Path(__file__).resolve().relative_to(ROOT).as_posix()},
        "Exact105=original104 plus this CPU driver",
    )
    require(
        all(handoff["files"][k] == v for k, v in admission["sources"].items()),
        "Original104 projection exact",
    )
    driver_label = Path(__file__).resolve().relative_to(ROOT).as_posix()
    checked({"path": driver_label, **handoff["files"][driver_label]})
    require(
        plan["environment"]["CUDA_VISIBLE_DEVICES"] == ""
        and plan["environment"]["HF_HUB_OFFLINE"]
        == plan["environment"]["TRANSFORMERS_OFFLINE"]
        == "1",
        "CPU offline/CUDA hidden",
    )
    return plan, handoff["files"]


def authenticate_sources(files):
    for label, pin in files.items():
        checked({"path": label, **pin})


def module(name, label, files):
    path = ordinary(ROOT / label)
    checked({"path": label, **files[label]})
    if name in sys.modules:
        loaded = sys.modules[name]
        require(Path(loaded.__file__).resolve() == path, "CPU module origin: " + name)
        return loaded
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


class BoundedGate:
    def __init__(self, inner, output):
        self.inner, self.output = inner, output

    def check(self, phase, scope):
        caps(self.output, 16384)
        result = self.inner.check(phase, scope)
        caps(self.output)
        return result


def resource_gate(output, label, files):
    module("jev_public_action_artifact", "scripts/jev_public_action_artifact.py", files)
    public = module(
        "jev_browser_fit_public_actions", "scripts/jev_browser_fit_public_actions.py", files
    )
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    return BoundedGate(public.ResourceGate(output / label, boot), output)


def ownership(files):
    label = "artifacts/jev-runtime/public-affordance-matrix-harness-20261019-01/matrix_common.py"
    return module("_form_CPU_original_owned", label, files)


def child_argv(pid):
    return [
        x.decode()
        for x in (Path("/proc") / str(pid) / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    ]


def fresh_rows(plan, status):
    return [{**r, "status": status, "compiled_reference": None} for r in plan["ordered_cases"]]


def verify_worker_binding(binding, actual, nonce, expected, pid):
    # Scheduling state changes between parent and child observations; PID/start
    # and the independent process group/session are the ownership identity.
    fields = {"pid", "ppid", "pgid", "sid", "start_ticks"}
    released = binding.get("identity")
    require(
        type(released) is type(actual) is dict
        and fields <= set(released)
        and fields <= set(actual)
        and all(type(actual[k]) is type(released[k]) is int for k in fields)
        and all(actual[k] == released[k] for k in fields)
        and actual["start_ticks"] > 0
        and actual["pid"] == actual["pgid"] == actual["sid"] == pid
        and binding.get("nonce") == nonce
        and binding.get("plan_sha256") == expected,
        "Released exact owned PID/start/PG/SID",
    )


def capture_reference(plan, output):
    receipt_blob = checked(plan["original_compiler_receipt"])
    require(
        hashlib.sha256(receipt_blob).hexdigest()
        == "a8a5ebdefccc4ef1be01e673c36d36452efd30368b02921b81aae9cabd096b82",
        "Original genuine CPU compiler authority",
    )
    receipt = parse(receipt_blob)
    expected = {
        k: v
        for k, v in receipt["output_inventory_excluding_receipt"].items()
        if k == "processor-source-closure.json"
        or k.startswith(("processor-runtime-sources/", "processor-metadata/"))
    }
    require(
        expected == plan["reference_items"] and len(expected) == 58,
        "Original51 origins plus6 metadata and closure; complete58 items",
    )
    root = ROOT / plan["reference_root"]
    result = []
    for label, pin in expected.items():
        path = ordinary(root / label)
        require(path.is_relative_to(root), "Original reference confinement")
        blob = read(path)
        require(
            {"sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)} == pin,
            "Original processor reference bytes",
        )
        save(output, "reference-environment/" + label, blob)
        result.append((label, blob))
    return tuple(result)


def load_processor(reference_items, gate):
    import jev_public_action_artifact as public

    public.verify_reference_environment(reference_items)
    gate.check("before-any-project-framework-or-parent-payload", "cpu")
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        public.MODEL_ID, revision=public.REVISION, local_files_only=True
    )
    refs = dict(reference_items)
    model = SimpleNamespace(
        processor=processor,
        tok=processor.tokenizer,
        max_context=8192,
        lm=SimpleNamespace(
            config=SimpleNamespace(**parse(refs["processor-metadata/config.json"])),
            generation_config=SimpleNamespace(eos_token_id=[1, 106, 50]),
        ),
    )
    public.verify_reference_environment(reference_items, model=model)
    import torch

    torch.set_num_threads(2)
    require(
        getattr(torch.cuda, "_initialized", False) is False,
        "No initialized CUDA in CPU processor worker",
    )
    return model


def processor_evidence(model, reference_items):
    import jev_public_action_artifact as public

    versions = public.verify_reference_environment(reference_items, model=model)
    components = {
        "processor": model.processor,
        "tokenizer": model.tok,
        "feature_extractor": model.processor.feature_extractor,
        "image_processor": model.processor.image_processor,
        "video_processor": model.processor.video_processor,
    }
    return {
        "versions": versions,
        "components": {
            k: {"class": type(v).__name__, "path": str(Path(inspect.getfile(type(v))).resolve())}
            for k, v in components.items()
        },
        "tokenizer_is_processor_tokenizer": model.tok is model.processor.tokenizer,
        "max_context": model.max_context,
        "generation_EOS": model.lm.generation_config.eos_token_id,
    }


def worker(plan_path, expected, output, nonce):
    plan, files = load_plan(plan_path, expected)
    require(
        sys.version_info[:2] == (3, 13) and sys.executable == plan["python"],
        "Actual reviewed Comfy3.13",
    )
    output = ordinary(output)
    intent = parse(read(output / "controller-intent.json", MIB))
    require(
        intent["nonce"] == nonce and intent["plan_sha256"] == expected, "Owned CPU worker intent"
    )
    rows, primary, secondary = fresh_rows(plan, "NOT_EXECUTED"), None, []
    state = {
        "admission_started": False,
        "admission_returned": False,
        "processor_load_started": False,
        "processor_load_returned": False,
        "compile_started": False,
        "compile_returned": False,
        "compiled_rows_in_memory": 0,
        "saved_rows": 0,
    }
    model = collection = None
    try:
        require(
            all(os.environ.get(k) == v for k, v in plan["environment"].items()),
            "Exact offline CPU worker environment",
        )
        require(
            not any(
                n.split(".")[0] in {"torch", "transformers", "tokenizers", "peft", "s1"}
                for n in sys.modules
            ),
            "Fresh pre-framework CPU child",
        )
        deadline = time.monotonic() + 10
        release = output / "controller-release.json"
        while not release.exists():
            require(time.monotonic() < deadline, "Owned CPU release deadline")
            time.sleep(0.02)
        binding = parse(read(release, MIB))
        owned = ownership(files)
        actual = owned.process_table().get(os.getpid())
        verify_worker_binding(binding, actual, nonce, expected, os.getpid())
        gate = resource_gate(output, "worker-resource-gates.jsonl", files)
        gate.check("before-any-project-framework-or-parent-payload", "cpu")
        authenticate_sources(files)
        forms = module("jev_owned_form_progress", "scripts/jev_owned_form_progress.py", files)
        state["admission_started"] = True
        admission_blob = checked(plan["admission_plan"])
        admission_plan = parse(admission_blob)
        collection = forms.validate_form_progress_collection(
            admission_blob,
            expected_plan_sha256=plan["admission_plan"]["sha256"],
            reviewed_source_pins=admission_plan["sources"],
            gate=gate,
        )
        state["admission_returned"] = True
        cases = collection.train_cases + collection.development_cases
        require(
            [{"id": c.id, "split": c.split, "family": c.annotation()["family"]} for c in cases]
            == plan["ordered_cases"],
            "Actual complete typed form case order",
        )
        save(output, "admission-provenance.json", encode(collection.provenance) + b"\n")
        gate.check("before-any-project-framework-or-parent-payload", "cpu")
        reference_items = capture_reference(plan, output)
        state["processor_load_started"] = True
        model = load_processor(reference_items, gate)
        state["processor_load_returned"] = True
        save(
            output,
            "actual-processor-before.json",
            encode(processor_evidence(model, reference_items)) + b"\n",
        )
        state["compile_started"] = True
        state["compiled_rows_in_memory"] = None
        rows = fresh_rows(plan, "COMPILE_RETURN_UNKNOWN")
        compiled, cfg = forms.compile_form_references(model, collection, reference_items, gate=gate)
        state.update(compile_returned=True, compiled_rows_in_memory=len(compiled))
        rows = fresh_rows(plan, "COMPILED_IN_MEMORY_EVIDENCE_INCOMPLETE")
        require(
            len(compiled) == len(cfg) == 23 and set(compiled) == set(cfg) == {c.id for c in cases},
            "Complete23 returned compile/CFG rows",
        )
        save(
            output,
            "actual-processor-after.json",
            encode(processor_evidence(model, reference_items)) + b"\n",
        )
        for ordinal, case in enumerate(cases):
            blob = (
                encode(
                    {
                        "case": case.annotation(),
                        "row": forms.common.serialized_row(compiled[case.id]),
                        "cfg_plan": cfg[case.id].evidence(),
                    }
                )
                + b"\n"
            )
            body = save(output, f"compiled/{ordinal:02d}.native-wire.json", case.wire_bytes)
            saved = save(output, f"compiled/{ordinal:02d}.json", blob)
            rows[ordinal].update(
                status="COMPLETE_REFERENCE", compiled_reference=saved, native_wire=body
            )
            state["saved_rows"] += 1
        for split in forms.COUNTS:
            save(
                output,
                "compile-projection-" + split + ".jsonl",
                b"".join(encode(r) + b"\n" for r in collection.compile_wire_projection(split)),
            )
        authenticate_sources(files)
        gate.check("before-any-project-framework-or-parent-payload", "cpu")
    except BaseException as error:
        primary = error_value(error, "worker")
    finally:
        model = collection = None
        try:
            gc.collect()
        except BaseException as error:
            secondary.append(error_value(error, "CPU-object-release"))
        complete = (
            primary is None
            and not secondary
            and state["compile_returned"]
            and state["saved_rows"] == 23
            and all(r["status"] == "COMPLETE_REFERENCE" for r in rows)
        )
        result = {
            "schema_id": "owned-form-progress-native-CPU-reference-v1",
            "status": SUCCESS if complete else "FAILED_OR_INCOMPLETE_CPU_REFERENCE",
            "nonce": nonce,
            "plan_sha256": expected,
            "rows": rows,
            "counts": state,
            "primary_error": primary,
            "secondary_errors": secondary,
            "model_constructor_calls": 0,
            "model_forward_calls": 0,
            "optimizer_updates": 0,
            "HTTP_browser_Modal_calls": 0,
            "scope": "Actual processor-only CPU references if complete; not capacity/training/qualification. External owned controller/whole exit mandatory.",
        }
        published = terminal(output, "worker-terminal.json", result, secondary)
        if not published:
            report_unknown("UNKNOWN_WORKER_TERMINAL_EVIDENCE", primary, secondary)
    return 0 if complete and published else 1


def worker_summary(output, plan, expected, nonce):
    terminal_blob = read(output / "worker-terminal.json", MIB)
    value = parse(terminal_blob)
    require(
        value["schema_id"] == "owned-form-progress-native-CPU-reference-v1"
        and value["nonce"] == nonce
        and value["plan_sha256"] == expected
        and [{k: r[k] for k in ("id", "split", "family")} for r in value["rows"]]
        == plan["ordered_cases"],
        "Authenticated full23 worker summary, including failures",
    )
    return terminal_blob, value


def select_reference(output, plan, expected, nonce, *, summary=None):
    terminal_blob, value = (
        worker_summary(output, plan, expected, nonce) if summary is None else summary
    )
    require(
        value["status"] == SUCCESS
        and value["primary_error"] is None
        and not value["secondary_errors"]
        and value["counts"]["saved_rows"] == value["counts"]["compiled_rows_in_memory"] == 23
        and value["counts"]["compile_returned"] is True,
        "Complete genuine CPU child terminal",
    )
    for label, pin in plan["reference_items"].items():
        blob = read(output / "reference-environment" / label)
        require(
            {"sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)} == pin,
            "Closed original58 processor reference population",
        )
    closure = parse(read(output / "reference-environment/processor-source-closure.json"))
    for label in ("actual-processor-before.json", "actual-processor-after.json"):
        evidence = parse(read(output / label, MIB))
        require(
            evidence["components"] == closure["components"]
            and evidence["versions"] == closure["versions"]
            and evidence["tokenizer_is_processor_tokenizer"] is True
            and evidence["max_context"] == 8192
            and evidence["generation_EOS"] == [1, 106, 50],
            "Genuine actual processor components/origins before/after",
        )
    for item in value["rows"]:
        require(item["status"] == "COMPLETE_REFERENCE", "No success selection from partial run")
        bodies = []
        for key in ("compiled_reference", "native_wire"):
            ref = item[key]
            blob = read(output / ref["path"])
            require(
                len(blob) == ref["size_bytes"]
                and hashlib.sha256(blob).hexdigest() == ref["sha256"],
                "Closed CPU row/body SHA",
            )
            bodies.append(blob)
        data, raw = parse(bodies[0]), bodies[1]
        row, cfg = data["row"], data["cfg_plan"]
        record = row["record"]
        p, length = record["prompt_tokens"], record["input_tokens"]
        require(
            data["case"]["id"] == record["case_id"] == item["id"]
            and record["split"] == item["split"]
            and record["original_wire_sha256"] == hashlib.sha256(raw).hexdigest(),
            "Closed CPU original case/raw request identity",
        )
        require(
            1 <= p < length
            and 1 <= length - p <= 32
            and p + 32 <= 8192
            and row["positions"] == record["logits_positions"] == list(range(p - 1, length - 1))
            and row["targets"] == record["target_token_ids"] == row["inputs"]["input_ids"][0][p:]
            and row["targets"][-1] == record["native_stop_token_id"] == 106
            and not any(t in {1, 106, 50} for t in row["targets"][:-1])
            and row["inputs"] == record["processor_tensors"],
            "Closed full token/mask/readout/firstEOS reference",
        )
        require(
            cfg["compiled_row_sha256"] == hashlib.sha256(encode(row)).hexdigest()
            and cfg["request_sha256"]
            == record["normalized_wire_sha256"]
            == hashlib.sha256(raw).hexdigest()
            and cfg["target_ids"] == row["targets"]
            and cfg["positions"] == row["positions"]
            and len(cfg["allowed_ids"]) == len(row["targets"])
            and all(
                t in allowed for t, allowed in zip(row["targets"], cfg["allowed_ids"], strict=True)
            ),
            "Closed full request-bound CFG path",
        )
    return {
        "worker_terminal_sha256": hashlib.sha256(terminal_blob).hexdigest(),
        "compiled_rows": 23,
        "counts": {"train": 13, "development": 10},
    }


def controller(plan_path, expected, output, *, run=False):
    plan, files = load_plan(plan_path, expected)
    output = ordinary(output)
    require(not output.exists(), "Fresh CPU output namespace")
    if not run:
        return {"status": "DRY_SOURCE_ONLY_NO_CHILD_FRAMEWORK_OR_PROCESSOR", "fixed_rows": 23}
    output.mkdir(mode=0o700)
    nonce = uuid.uuid4().hex
    argv = [
        plan["python"],
        "-B",
        "-s",
        str(Path(__file__).resolve()),
        "--role",
        "worker",
        "--plan",
        str(Path(plan_path).absolute()),
        "--plan-sha256",
        expected,
        "--output",
        str(output),
        "--nonce",
        nonce,
        "--run",
    ]
    save(
        output,
        "controller-intent.json",
        encode({"nonce": nonce, "plan_sha256": expected, "argv": argv}) + b"\n",
    )
    process = primary = gate = None
    secondary, known, groups, interrupted = [], {}, set(), []
    cleanup, selected, summary, returncode = None, None, None, None
    old_handlers = {}

    def stop_signal(number, _frame):
        interrupted.append(number)
        raise InterruptedError("Owned form CPU controller signal")

    try:
        require(
            not any(
                n.split(".")[0] in {"torch", "transformers", "tokenizers", "peft", "s1"}
                for n in sys.modules
            ),
            "Fresh CPU controller, no framework/model",
        )
        gate = resource_gate(output, "parent-resource-gates.jsonl", files)
        gate.check("before-any-project-framework-or-parent-payload", "cpu")
        authenticate_sources(files)
        owned = ownership(files)
        old_handlers = {
            sig: signal.signal(sig, stop_signal) for sig in (signal.SIGINT, signal.SIGTERM)
        }
        env = {k: os.environ[k] for k in SECRET_FREE_ENV if k in os.environ}
        env.update(plan["environment"])
        with (
            (output / "child-stdout.txt").open("xb") as stdout,
            (output / "child-stderr.txt").open("xb") as stderr,
        ):
            os.chmod(stdout.name, 0o600)
            os.chmod(stderr.name, 0o600)
            process = subprocess.Popen(
                argv,
                cwd=ROOT,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            identity = owned.process_table().get(process.pid)
            require(
                identity is not None and identity["pgid"] == identity["sid"] == process.pid,
                "Fresh owned CPU PID/PG/SID",
            )
            require(child_argv(process.pid) == argv, "Exact CPU child argv")
            known[process.pid] = identity["start_ticks"]
            groups.add(process.pid)
            save(
                output,
                "controller-release.json",
                encode({"nonce": nonce, "plan_sha256": expected, "identity": identity}) + b"\n",
            )
            deadline, recorded = time.monotonic() + 600, set()
            while True:
                rows, _ = owned.extend_owned(process.pid, known, groups)
                for pid, row in rows.items():
                    key = (pid, row["start_ticks"])
                    if key not in recorded:
                        save(
                            output,
                            f"processes/{pid}-{row['start_ticks']}.json",
                            encode(row) + b"\n",
                        )
                        recorded.add(key)
                if process.poll() is not None:
                    break
                require(time.monotonic() < deadline, "Owned CPU600s deadline")
                caps(output)
                require(stdout.tell() < MIB and stderr.tell() < MIB, "CPU stdout/stderr1MiB bound")
                time.sleep(0.05)
    except BaseException as error:
        primary = error_value(error, "controller")
    finally:
        for sig, handler in old_handlers.items():
            try:
                signal.signal(sig, handler)
            except BaseException as error:
                secondary.append(error_value(error, "signal-handler-restore"))
        if process is not None:
            try:
                owned = ownership(files)
                if not known:
                    identity = owned.process_table().get(process.pid)
                    if identity is not None and identity["pgid"] == identity["sid"] == process.pid:
                        known[process.pid] = identity["start_ticks"]
                        groups.add(process.pid)
                cleanup = owned.stop_owned(process, known, groups)
            except BaseException as error:
                secondary.append(error_value(error, "owned-cleanup"))
            try:
                returncode = process.poll()
            except BaseException as error:
                secondary.append(error_value(error, "child-final-poll"))
        if cleanup and cleanup["confirmed"]:
            try:
                summary = worker_summary(output, plan, expected, nonce)
            except BaseException as error:
                secondary.append(error_value(error, "closed-worker-summary"))
        if (
            primary is None
            and not secondary
            and returncode == 0
            and cleanup
            and cleanup["confirmed"]
            and summary is not None
        ):
            try:
                authenticate_sources(files)
                gate.check("before-any-project-framework-or-parent-payload", "cpu")
                selected = select_reference(output, plan, expected, nonce, summary=summary)
            except BaseException as error:
                primary = error_value(error, "closed-reference-validation")
        result = {
            "schema_id": SCHEMA,
            "status": "CLOSED_CPU_REFERENCE_READY_PENDING_EXTERNAL_CONTROLLER_EXIT"
            if selected is not None
            else "FAILED_OR_UNKNOWN_CPU_REFERENCE",
            "nonce": nonce,
            "plan_sha256": expected,
            "process_started": process is not None,
            "actual_returncode": returncode,
            "primary_error": primary,
            "secondary_errors": secondary,
            "worker_primary_error": summary[1]["primary_error"] if summary else None,
            "worker_counts": summary[1]["counts"] if summary else None,
            "cleanup": cleanup,
            "closure_confirmed": bool(cleanup and cleanup["confirmed"]),
            "signals_received": interrupted,
            "reference_ready": selected,
            "selection_requires_external_controller_exit0_and_group_absent": True,
            "all23_rows": summary[1]["rows"]
            if summary is not None
            else fresh_rows(
                plan, "UNKNOWN_AFTER_CHILD_FAILURE" if process is not None else "NOT_EXECUTED"
            ),
            "scope": "Whole exact child closure plus saved references only. External root must confirm this controller exit0/group absent before selecting reference; no capacity/training/model/browser score.",
        }
        published = terminal(output, "controller-terminal.json", result, secondary)
        if not published:
            report_unknown("UNKNOWN_CONTROLLER_TERMINAL_EVIDENCE", primary, secondary)
    return 0 if selected is not None and published else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("controller", "worker"), default="controller")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nonce")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.role == "worker":
        require(args.run and args.nonce, "Owned explicit worker execution")
        result = worker(args.plan, args.plan_sha256, args.output, args.nonce)
    else:
        result = controller(args.plan, args.plan_sha256, args.output, run=args.run)
    print(encode(result).decode())
    raise SystemExit(result if isinstance(result, int) else 0)
