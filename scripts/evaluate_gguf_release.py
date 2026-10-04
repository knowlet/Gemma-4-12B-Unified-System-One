#!/usr/bin/env python3
"""Calibrate and evaluate a provenance-bound GGUF S1 backend on frozen populations.

The persistent backend receives gold-free requests and returns candidate logits
after the native softcap, before temperature. No generation or test-set fitting
is used. Torch, llama.cpp and accelerator initialization belong to the adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import runpy
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

EXPECTED_CASES = {"calibration": 256, "test": 256, "media": 52}
LOGITS_SEMANTICS = "post_native_softcap_pre_temperature"


@lru_cache(maxsize=1)
def shared_scripts():
    root = Path(__file__).resolve().parent
    return {
        name: runpy.run_path(str(root / filename))
        for name, filename in {
            "mlx": "evaluate_mlx_release.py",
            "summary": "summarize_release.py",
            "capture": "prepare_mlx_validation.py",
            "verifier": "verify_mlx_export.py",
        }.items()
    }


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_sha256(path):
    return shared_scripts()["capture"]["file_sha256"](path)


class BackendFailure(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class AdapterUnavailable(RuntimeError):
    """The persistent transport cannot safely accept another request."""


def _integer_list(value, *, length=None):
    return (
        isinstance(value, list)
        and (length is None or len(value) == length)
        and all(type(item) is int and item >= 0 for item in value)
    )


def validate_response(response, payload, *, context_size=16384):
    """Enforce case/slot/candidate alignment without asserting media tensor parity."""
    if not isinstance(response, dict):
        raise ValueError("adapter response must be a JSON object")
    json.dumps(response, allow_nan=False)
    if response.get("case_id") != payload["case_id"]:
        raise ValueError("adapter case identity differs")
    status = response.get("status")
    if status in ("error", "unsupported"):
        message = response.get("error")
        if not isinstance(message, str) or not message:
            raise ValueError("unsuccessful adapter response requires an explicit error")
        raise BackendFailure(status, message)
    if status != "ok":
        raise ValueError("adapter status must be ok, error or unsupported")
    count = len(payload["question_ids"])
    if response.get("question_ids") != payload["question_ids"]:
        raise ValueError("adapter question identities or ordering differ")
    if (
        not _integer_list(response.get("slots"), length=count)
        or response["slots"] != payload["slots"]
    ):
        raise ValueError("adapter must echo the original captured slots")
    slots, tokens = response.get("runtime_slots"), response.get("native_token_count")
    if (
        not _integer_list(slots, length=count)
        or slots != sorted(set(slots))
        or type(tokens) is not int
        or not 0 < tokens <= context_size
        or not slots
        or max(slots) >= tokens
    ):
        raise ValueError("native slots must be ordered and within the declared context")
    if not payload["has_media"] and (
        slots != payload["slots"] or tokens != payload["source_input_token_count"]
    ):
        raise ValueError("text token count and slots must match exact captured input_ids")
    rows = response.get("raw_logits")
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError("raw logits must cover every question exactly")
    for row, size in zip(rows, payload["nopts"]):
        if (
            not isinstance(row, list)
            or len(row) != size
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in row)
        ):
            raise ValueError("raw logits must be finite complete legal-candidate vectors")
    runtime = response.get("runtime")
    if (
        not isinstance(runtime, dict)
        or not isinstance(runtime.get("engine"), str)
        or not runtime["engine"]
        or runtime.get("logits_semantics") != LOGITS_SEMANTICS
    ):
        raise ValueError("runtime must declare post-native-softcap, pre-temperature logits")
    return response


def initialize_records(manifests):
    return {
        role: [
            {
                "id": case["id"],
                "split": case["split"],
                "group_id": case["group_id"],
                "request_sha256": case["request_sha256"],
                "tensor_file_sha256": case["tensor_file_sha256"],
                "status": "not_run",
                "reason": "evaluation has not reached this case",
                "answers": {},
                "backend_call_samples_ms": [],
            }
            for case in manifest["cases"]
        ]
        for role, manifest in manifests.items()
    }


def summarize_split(records, manifest):
    # Reuse the established metrics, explicitly relabeling the different timing scope.
    normalized = [
        {**r, "prepared_forward_samples_ms": r["backend_call_samples_ms"]} for r in records
    ]
    result = shared_scripts()["mlx"]["summarize_split"](normalized, manifest)
    result["backend_call_latency_ms"] = result.pop("prepared_forward_latency_ms")
    result["case_status_counts"] = dict(Counter(r["status"] for r in records))
    decisions = Counter()
    for record, case in zip(records, manifest["cases"]):
        status = record["status"]
        if status == "ok" and any(
            "probabilities" not in answer for answer in record["answers"].values()
        ):
            status = "unscored"
        decisions[status] += len(case["questions"])
    result["decision_status_counts"] = dict(decisions)
    result["metrics_population"] = "successful scored decisions only; coverage retains all cases"
    return result


def evaluate_backend(
    backend,
    manifests,
    payload_factory,
    *,
    warmup=1,
    repeat=1,
    context_size=16384,
    clock=time.perf_counter,
    phase_guard=lambda: None,
    result=None,
):
    """Run one fixed backend recipe; unsuccessful calibration prevents held-out scoring."""
    if type(warmup) is not int or warmup < 0 or type(repeat) is not int or repeat < 1:
        raise ValueError("warmup must be nonnegative and repeat must be positive")
    records = initialize_records(manifests)
    result = {} if result is None else result
    result.update(cases=records, calibration=None, backend_runtime=None)
    temperature, stopped = None, False
    mlx = shared_scripts()["mlx"]
    for role in EXPECTED_CASES:
        if stopped:
            break
        for index, (case, record) in enumerate(zip(manifests[role]["cases"], records[role])):
            try:
                payload = payload_factory(role, case)
                json.dumps(payload, allow_nan=False)
                responses, samples = [], []
                for iteration in range((warmup if index == 0 else 0) + repeat):
                    started = clock()
                    response = validate_response(
                        backend(payload), payload, context_size=context_size
                    )
                    elapsed = (clock() - started) * 1000
                    if not math.isfinite(elapsed) or elapsed < 0:
                        raise ValueError("backend call duration must be finite and nonnegative")
                    if result["backend_runtime"] is None:
                        result["backend_runtime"] = response["runtime"]
                    elif result["backend_runtime"] != response["runtime"]:
                        raise AdapterUnavailable(
                            "backend runtime identity changed during evaluation"
                        )
                    if iteration >= (warmup if index == 0 else 0):
                        responses.append(response)
                        samples.append(elapsed)
                first = responses[0]
                for response in responses[1:]:
                    if (
                        response["runtime_slots"] != first["runtime_slots"]
                        or response["native_token_count"] != first["native_token_count"]
                    ):
                        raise ValueError("native token/slot layout changed between repetitions")
                record.update(
                    status="ok",
                    backend_call_samples_ms=samples,
                    source_slots=case["slots"],
                    runtime_slots=first["runtime_slots"],
                    source_input_token_count=payload["source_input_token_count"],
                    native_token_count=first["native_token_count"],
                    preprocessing=first.get("preprocessing"),
                    max_repeat_logit_drift=max(
                        (
                            max(
                                float(np.abs(np.array(a) - b).max())
                                for a, b in zip(first["raw_logits"], response["raw_logits"])
                            )
                            for response in responses[1:]
                        ),
                        default=0.0,
                    ),
                )
                record.pop("reason", None)
                record["answers"] = {
                    q["id"]: {
                        "type": q["type"],
                        "labels": q["labels"],
                        "gold": case["gold"][q["id"]],
                        "raw_logits": row,
                    }
                    for q, row in zip(case["questions"], first["raw_logits"])
                }
            except Exception as exc:
                record.update(status=getattr(exc, "status", "error"), reason=str(exc), answers={})
                record["error_type"] = type(exc).__name__
                if isinstance(exc, AdapterUnavailable):
                    stopped = True
                    break
            if (index + 1) % 32 == 0 and index + 1 < len(records[role]):
                print(
                    json.dumps(
                        {
                            "split": role,
                            "processed_cases": index + 1,
                            "expected_cases": len(records[role]),
                            "case_status_counts": dict(Counter(r["status"] for r in records[role])),
                        }
                    ),
                    file=sys.stderr,
                    flush=True,
                )
        print(
            json.dumps(
                {
                    "split": role,
                    "split_finished": True,
                    "expected_cases": len(records[role]),
                    "case_status_counts": dict(Counter(r["status"] for r in records[role])),
                }
            ),
            file=sys.stderr,
            flush=True,
        )
        phase_guard()
        if role == "calibration":
            if stopped or any(r["status"] != "ok" for r in records[role]):
                stopped = True
                result["calibration_failure"] = "complete calibration-only coverage is required"
            else:
                result["calibration"] = mlx["fit_calibration"](
                    records[role], expected_cases=len(manifests[role]["cases"])
                )
                temperature = result["calibration"]["temperature"]
        if temperature is not None:
            mlx["apply_temperature"](records[role], temperature)
    result["summary"] = {
        role: summarize_split(records[role], manifests[role]) for role in EXPECTED_CASES
    }
    result["population_complete"] = temperature is not None and all(
        item["case_coverage"] == item["decision_coverage"] == 1
        for item in result["summary"].values()
    )
    return result


def validate_inventory(files):
    if not isinstance(files, list) or not files:
        raise ValueError("immutable file inventory must be nonempty")
    names = []
    for item in files:
        name = item.get("path") if isinstance(item, dict) else None
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name in (".", "..")
            or type(item.get("size_bytes")) is not int
            or item["size_bytes"] <= 0
            or not shared_scripts()["mlx"]["_digest"](item.get("sha256"))
        ):
            raise ValueError("invalid immutable file inventory entry")
        names.append(name)
    if names != sorted(set(names)):
        raise ValueError("immutable file inventory must be unique and sorted")
    return set(names)


def validate_conversion(model_path, conversion, identity, manifests):
    """Bind source inventories, sidecar and every immutable exported byte."""
    if conversion.get("format") != "gguf" or conversion.get("status") != (
        "converted_and_structurally_verified"
    ):
        raise ValueError("GGUF conversion must be structurally verified")
    for key in (
        "source_model",
        "source_revision",
        "source_checkpoint_sha256",
        "source_s1_config_sha256",
    ):
        if conversion.get(key) != identity[key]:
            raise ValueError(f"conversion {key} differs from captured checkpoint")
    source_files = conversion.get("source_checkpoint_files")
    validate_inventory(source_files)
    if digest(source_files) != identity["source_checkpoint_sha256"] or any(
        manifest.get("source_checkpoint_files") != source_files for manifest in manifests.values()
    ):
        raise ValueError("source checkpoint inventory differs from captured checkpoint digest")
    files = conversion.get("files")
    names = validate_inventory(files)
    required = {
        "base_s1_config.json",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "chat_template.jinja",
        "tensor-inventory.json",
    }
    model, mmproj = conversion.get("model_file"), conversion.get("mmproj_file")
    if (
        not isinstance(model, str)
        or not isinstance(mmproj, str)
        or model == mmproj
        or {name for name in names if name.endswith(".gguf")} != {model, mmproj}
        or not required <= names
        or "s1_config.json" in names
        or {path.name for path in Path(model_path).glob("*.gguf")} != {model, mmproj}
    ):
        raise ValueError(
            "GGUF inventory must include exactly the selected model/projector and processor"
        )
    for item in files:
        path = Path(model_path) / item["path"]
        if path.stat().st_size != item["size_bytes"] or file_sha256(path) != item["sha256"]:
            raise ValueError(f"exported file changed: {item['path']}")
    base_bytes = (Path(model_path) / "base_s1_config.json").read_bytes()
    if hashlib.sha256(base_bytes).hexdigest() != identity["source_s1_config_sha256"]:
        raise ValueError("base sidecar differs from captured source sidecar")
    base = json.loads(base_bytes)
    for key in ("source_model", "source_revision", "prompt_version"):
        if base.get(key) != identity[key]:
            raise ValueError(f"base sidecar {key} differs from source")
    if (
        not shared_scripts()["mlx"]["_digest"](conversion.get("training_recipe_sha256"))
        or base.get("training_recipe_sha256") != conversion["training_recipe_sha256"]
    ):
        raise ValueError("conversion training recipe differs from source sidecar")
    config = json.loads((Path(model_path) / "config.json").read_text())
    shared_scripts()["verifier"]["validate_model_config"](config)
    return {"sha256": digest(files), "files": files}, base


def make_payload(role, case, *, directories, manifests, population):
    arrays, actual_digest = shared_scripts()["verifier"]["load_case_arrays"](
        directories[role], case
    )
    if actual_digest != case["tensor_file_sha256"]:
        raise ValueError("captured tensors changed after preflight")
    request = population[role][case["id"]].request
    return {
        "case_id": case["id"],
        "request": request.model_dump(mode="json"),
        "question_ids": [question["id"] for question in case["questions"]],
        "slots": case["slots"],
        "letters": manifests[role]["letters"],
        "nopts": case["nopts"],
        "input_ids": arrays["input_ids"][0].tolist(),
        "has_media": bool(request.media),
        "source_input_token_count": int(arrays["input_ids"].shape[-1]),
    }


class RuntimeGuard:
    """Freeze actual code/interpreter/native bytes independently from conversion provenance."""

    def __init__(self, revision, command, extra_files=(), *, root=None):
        self.root = Path(root or Path(__file__).resolve().parents[1]).resolve()
        self.revision = revision
        if (
            len(revision) != 40
            or any(c not in "0123456789abcdef" for c in revision)
            or self.git("rev-parse", "HEAD").decode().strip() != revision
        ):
            raise ValueError("runtime source revision must equal the current full Git HEAD")
        paths = {Path(__file__).resolve(), Path(sys.executable).resolve()}
        paths.update((self.root / "src" / "s1").rglob("*.py"))
        paths.update(
            self.root / "scripts" / name
            for name in (
                "evaluate_mlx_release.py",
                "summarize_release.py",
                "prepare_mlx_validation.py",
                "verify_mlx_export.py",
            )
        )
        # Bind the bundled native adapter to its build inputs and all loadable
        # backend libraries (Metal/CPU may be loaded dynamically by ggml).
        if any(Path(value).name == "gguf_adapter.py" for value in command):
            if "--native-runner" not in command:
                raise ValueError("bundled GGUF adapter requires an explicit native runner")
            runner = Path(command[command.index("--native-runner") + 1]).resolve()
            cache = runner.parent.parent / "CMakeCache.txt"
            paths.update(
                (
                    self.root / "tools" / "gguf" / "s1_gguf.cpp",
                    self.root / "tools" / "gguf" / "CMakeLists.txt",
                    cache,
                )
            )
            entries = dict(
                line.split("=", 1)
                for line in cache.read_text().splitlines()
                if line.startswith("LLAMA_CPP_DIR:PATH=")
            )
            library_root = Path(entries["LLAMA_CPP_DIR:PATH"]) / "build" / "bin"
            libraries = {
                path.resolve()
                for path in library_root.iterdir()
                if path.is_file() and (".so" in path.name or path.suffix == ".dylib")
            }
            if not libraries:
                raise ValueError("native build must expose its shared backend libraries")
            paths.update(libraries)
        for value in command:
            candidate = Path(value)
            if candidate.is_file():
                paths.add(candidate.resolve())
        executable = shutil.which(command[0])
        if not executable:
            raise ValueError("adapter executable could not be resolved")
        paths.add(Path(executable).resolve())
        for value in extra_files:
            candidate = Path(value).resolve()
            if not candidate.is_file():
                raise ValueError(f"runtime file does not exist: {candidate}")
            paths.add(candidate)
        self.files = []
        for path in sorted(paths):
            actual = file_sha256(path)
            relative = path.relative_to(self.root) if path.is_relative_to(self.root) else None
            tracked = False
            if relative is not None:
                exists = (
                    subprocess.run(
                        [
                            "git",
                            "-c",
                            "core.fsmonitor=false",
                            "-C",
                            str(self.root),
                            "cat-file",
                            "-e",
                            f"HEAD:{relative}",
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    ).returncode
                    == 0
                )
                tracked = exists
                if exists:
                    if hashlib.sha256(self.git("show", f"HEAD:{relative}")).hexdigest() != actual:
                        raise ValueError(f"runtime file differs from pinned Git HEAD: {relative}")
                elif relative.suffix in (".py", ".cpp", ".h") or relative.name == "CMakeLists.txt":
                    raise ValueError(f"runtime source file must be committed: {relative}")
            self.files.append(
                {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": actual,
                    "tracked_at_runtime_revision": tracked,
                }
            )

    def git(self, *args):
        return subprocess.check_output(
            ["git", "-c", "core.fsmonitor=false", "-C", str(self.root), *args]
        )

    def check(self):
        if self.git("rev-parse", "HEAD").decode().strip() != self.revision:
            raise ValueError("Git HEAD changed during evaluation")
        for item in self.files:
            path = Path(item["path"])
            if path.stat().st_size != item["size_bytes"] or file_sha256(path) != item["sha256"]:
                raise ValueError(f"runtime file changed during evaluation: {path}")

    def receipt(self):
        return {
            "source_revision": self.revision,
            "files": self.files,
            "files_sha256": digest(self.files),
        }


class JsonLineAdapter:
    """Persistent shell-free transport with bounded calls and process-group cleanup."""

    def __init__(self, command, timeout=300):
        if not command or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("adapter command and positive finite timeout are required")
        self.timeout = timeout
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
            bufsize=1,
            start_new_session=True,
        )
        self.closed = False

    def __call__(self, payload):
        if self.closed:
            raise AdapterUnavailable("adapter transport is closed")
        result = queue.Queue(maxsize=1)

        def exchange():
            try:
                self.process.stdin.write(json.dumps(payload, allow_nan=False) + "\n")
                self.process.stdin.flush()
                line = self.process.stdout.readline(16 * 1024 * 1024)
                if not line or not line.endswith("\n"):
                    raise ValueError("adapter EOF or oversized response")
                result.put((True, json.loads(line)))
            except Exception as exc:
                result.put((False, exc))

        thread = threading.Thread(target=exchange, daemon=True)
        thread.start()
        try:
            success, value = result.get(timeout=self.timeout)
        except queue.Empty as exc:
            self.close(force=True)
            raise AdapterUnavailable("adapter response timed out") from exc
        if not success:
            self.close(force=True)
            raise AdapterUnavailable(f"adapter transport failed: {value}") from value
        return value

    def close(self, *, force=False):
        if self.closed:
            return
        self.closed = True
        if not force:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
        # The adapter may have a persistent native child, including after adapter EOF.
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            # Restricted macOS hosts can report EPERM for an already reaped
            # process group. Never suppress a failure to stop a live adapter.
            if self.process.poll() is None:
                raise
        self.process.wait(timeout=5)
        for stream in (self.process.stdin, self.process.stdout):
            try:
                stream.close()
            except OSError:
                pass


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    for role in EXPECTED_CASES:
        parser.add_argument(f"--{role}-inputs", type=Path, required=True)
        parser.add_argument(f"--{role}-manifest-sha256", required=True)
    for kind in ("dataset", "conversion"):
        parser.add_argument(f"--{kind}-manifest", type=Path, required=True)
        parser.add_argument(f"--{kind}-manifest-sha256", required=True)
    parser.add_argument("--runtime-source-revision", required=True)
    parser.add_argument("--runtime-file", action="append", default=[], type=Path)
    parser.add_argument("--adapter-timeout", type=float, default=300)
    parser.add_argument("--context-size", type=int, default=16384)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--adapter-command",
        nargs=argparse.REMAINDER,
        required=True,
        help="Persistent NDJSON command; this option must be last",
    )
    return parser.parse_args(argv)


def main(argv=None):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    args = parse_args(argv)
    report = {
        "schema_version": 1,
        "status": "error",
        "release_validation_complete": False,
        "population_complete": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_path": str(args.model.resolve()),
        "cases": {},
        "calibration": None,
        "measurement": {
            "scope": "persistent adapter request serialization through raw-logit response",
            "includes": "transport, HF preprocessing and native decoding; first call may include loading",
            "excludes": "captured-tensor file I/O, NumPy softmax and temperature fitting",
            "warmup": "first case per split only; cold loading is excluded only when warmup > 0",
            "warmup_per_split": args.warmup,
            "repeat_per_case": args.repeat,
            "not_comparable_to_cuda_or_mlx_timing_scope": True,
        },
        "metric_definitions": {"nll_probability_floor": 1e-12, "ece_equal_width_bins": 15},
        "processor_parity_claimed": False,
    }
    stage, manifests, adapter = "preflight", {}, None
    try:
        if args.warmup < 0 or args.repeat < 1 or args.context_size < 1 or not args.adapter_command:
            raise ValueError("invalid warmup, repeat, context size or adapter command")
        if not math.isfinite(args.adapter_timeout) or args.adapter_timeout <= 0:
            raise ValueError("adapter timeout must be positive and finite")
        shared = shared_scripts()
        read = shared["mlx"]["read_pinned_json"]
        directories = {role: getattr(args, f"{role}_inputs") for role in EXPECTED_CASES}
        pins = {role: getattr(args, f"{role}_manifest_sha256") for role in EXPECTED_CASES}
        manifests = {
            role: read(directory / "manifest.json", pins[role])
            for role, directory in directories.items()
        }
        dataset = read(args.dataset_manifest, args.dataset_manifest_sha256)
        identity = shared["mlx"]["validate_manifests"](manifests, dataset)
        report["cases"] = initialize_records(manifests)
        shared["mlx"]["validate_dataset_files"](manifests, dataset, args.dataset_manifest.parent)
        data, _, actual_dataset_pin = shared["summary"]["load_population"](
            args.dataset_manifest.parent
        )
        if actual_dataset_pin != args.dataset_manifest_sha256:
            raise ValueError("frozen population must use the pinned manifest.json")
        population = {role: {case.id: case for case in data[role]} for role in EXPECTED_CASES}
        conversion = read(args.conversion_manifest, args.conversion_manifest_sha256)
        if (
            args.conversion_manifest.resolve()
            != (args.model / "conversion-manifest.json").resolve()
        ):
            raise ValueError(
                "conversion receipt must be the model package conversion-manifest.json"
            )
        exported, base = validate_conversion(args.model, conversion, identity, manifests)
        report.update(identity)
        report.update(
            source_checkpoint_files=conversion["source_checkpoint_files"],
            input_manifest_sha256=pins,
            dataset_manifest_sha256=args.dataset_manifest_sha256,
            conversion_manifest_sha256=args.conversion_manifest_sha256,
            export_checkpoint_sha256=exported["sha256"],
            export_files=exported["files"],
            quantization=conversion.get("quantization"),
            llama_cpp_revision=conversion.get("llama_cpp_revision"),
            conversion_record_source_revision=conversion.get("conversion_record_source_revision"),
            adapter_command=args.adapter_command,
            context_size=args.context_size,
        )
        for role, manifest in manifests.items():
            for case in manifest["cases"]:
                payload = make_payload(
                    role, case, directories=directories, manifests=manifests, population=population
                )
                if payload["source_input_token_count"] > args.context_size:
                    raise ValueError(f"{case['id']}: captured input exceeds context size")
        guard = RuntimeGuard(args.runtime_source_revision, args.adapter_command, args.runtime_file)
        report["runtime_source_revision"] = args.runtime_source_revision
        report["runtime_implementation"] = guard.receipt()
        guard.check()
        stage = "evaluate_backend"
        adapter = JsonLineAdapter(args.adapter_command, args.adapter_timeout)
        evaluate_backend(
            adapter,
            manifests,
            lambda role, case: make_payload(
                role, case, directories=directories, manifests=manifests, population=population
            ),
            warmup=args.warmup,
            repeat=args.repeat,
            context_size=args.context_size,
            phase_guard=guard.check,
            result=report,
        )
        adapter.close()
        if not report["population_complete"]:
            raise ValueError("complete successful calibration/test/media coverage is required")
        if report["backend_runtime"].get("engine") != "llama.cpp" or report["backend_runtime"].get(
            "llama_cpp_revision"
        ) != conversion.get("llama_cpp_revision"):
            raise ValueError("native runtime differs from the pinned conversion engine/revision")
        stage = "final_provenance_validation"
        guard.check()
        read(args.conversion_manifest, args.conversion_manifest_sha256)
        validate_conversion(args.model, conversion, identity, manifests)
        for role, directory in directories.items():
            read(directory / "manifest.json", pins[role])
        read(args.dataset_manifest, args.dataset_manifest_sha256)
        stage = "save_calibrated_sidecar"
        calibrated = {
            **base,
            "temperature": report["calibration"]["temperature"],
            "runtime": "llama.cpp",
            "runtime_format": "gguf",
            "runtime_source_revision": args.runtime_source_revision,
            "conversion_manifest_sha256": args.conversion_manifest_sha256,
            "source_checkpoint_sha256": identity["source_checkpoint_sha256"],
            "export_checkpoint_sha256": exported["sha256"],
            "calibration_dataset_sha256": manifests["calibration"]["dataset_sha256"],
            "gguf_calibration": {
                **report["calibration"],
                "split": "calibration",
                "runtime": "llama.cpp",
                "manifest_sha256": pins["calibration"],
                "dataset_manifest_sha256": args.dataset_manifest_sha256,
                "runtime_implementation_sha256": report["runtime_implementation"]["files_sha256"],
                "source_temperature": base.get("temperature"),
            },
        }
        serialized = (json.dumps(calibrated, indent=2, allow_nan=False) + "\n").encode()
        temporary = args.model / ".s1_config.gguf-calibrated.tmp"
        temporary.write_bytes(serialized)
        temporary.replace(args.model / "s1_config.json")
        report["calibrated_s1_config_sha256"] = hashlib.sha256(serialized).hexdigest()
        report.update(status="ok", release_validation_complete=True)
    except Exception as exc:
        report["failure"] = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
    finally:
        if adapter is not None:
            adapter.close(force=True)
    if set(report["cases"]) == set(EXPECTED_CASES):
        try:
            report["summary"] = {
                role: summarize_split(report["cases"][role], manifests[role])
                for role in EXPECTED_CASES
            }
        except Exception as exc:
            report["summary_failure"] = {"type": type(exc).__name__, "message": str(exc)}
            report.update(status="error", release_validation_complete=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{report['status']}: {args.output}")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
