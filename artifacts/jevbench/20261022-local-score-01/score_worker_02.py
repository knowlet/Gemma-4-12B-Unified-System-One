"""One owned BF16 model, full public231 then official JevBench coherence mini."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[2]
OUTPUT = HERE / "actual-02"
UNIT = "jevbench-gx10-20261022-02.service"
EXECUTE = "jevbench-gx10-bf16-score-v1"
MODEL = "knowlet/Gemma-4-12B-Unified-System-One"
REVISION = "a66f836b56605039fe040f330180e336d19b3362"
PUBLIC_REVISION = "bb05a335bc809e61b20c0f745d25499a82b326fc"
COHERENCE_REVISION = "e18733694623aa93058e279c3534f8b8e2edefa3"
PROFILE = "released-gx10-bf16-causal-multislot-20261022-v1"
TEMPERATURE = 2.7830344470383452
CACHE = (
    ROOT
    / "artifacts/jev-local-hf/hub/models--knowlet--Gemma-4-12B-Unified-System-One"
    / "snapshots"
    / REVISION
)
HEAVEN = HERE / "upstream/heaven"
COHERENCE = HERE / "upstream/coherence"
CAP = 40 * 1024**3
STOP = CAP - 1024**3 // 2


class ScoreAbort(BaseException):
    """An environment/evidence refusal must escape upstream Exception handlers."""


def require(condition, message):
    if not condition:
        raise ScoreAbort(message)


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()


def write(path, value):
    try:
        blob = encode(value) + b"\n"
        with Path(path).open("wb") as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException as exc:
        if isinstance(exc, ScoreAbort):
            raise
        raise ScoreAbort("Durable score evidence write failed: " + str(path)) from exc


def append(path, value):
    try:
        with Path(path).open("ab") as stream:
            stream.write(encode(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException as exc:
        raise ScoreAbort("Durable prediction evidence failed") from exc


def pin(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Ordinary source file required")
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def git_revision(path):
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, timeout=15
    ).strip()


def authenticate_checkpoint(cache, resolver, pinner, expected):
    checkpoint = resolver(cache)
    observed = pinner(checkpoint, limit=32 * 1024**3)
    require(observed == expected, "Exact published base checkpoint SHA/size required")
    return checkpoint, observed


def source_paths():
    paths = [
        Path(__file__).absolute(),
        ROOT / "scripts/run_jevbench_public.py",
        ROOT / "scripts/jev_native_v2_browser.py",
        ROOT / "scripts/jev_public_action_artifact.py",
        ROOT / "configs/benchmarks/jevbench-public.json",
    ]
    from s1.browser_native_resources import COMPONENTS

    paths.extend(ROOT / ref["path"] for ref in COMPONENTS.values())
    for repository, source in ((HEAVEN, "jevbench"), (COHERENCE, "src/jevbench")):
        paths.extend(sorted((repository / source).rglob("*.py")))
        paths.extend(repository / name for name in ("LICENSE", "pyproject.toml"))
    paths.append(COHERENCE / "NOTICE")
    paths.extend(sorted((HEAVEN / "datasets/public").glob("*.jsonl")))
    suite = COHERENCE / "src/jevbench/suites"
    paths.extend(
        p for p in sorted(suite.rglob("*")) if p.is_file() and "__pycache__" not in p.parts
    )
    return {str(path.relative_to(ROOT)): path for path in sorted(set(paths))}


def capture_sources(output):
    paths = source_paths()
    references = {}
    for label, source in paths.items():
        references[label] = pin(source)
        target = output / "score-sources" / label
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        require(pin(target) == references[label], "Captured score source differs")
    write(output / "score-source-manifest.json", references)
    return references


def verify_sources(references):
    return {label: pin(path) for label, path in source_paths().items()} == references


class Admission:
    def __init__(self, gate, torch=None):
        self.gate, self.torch = gate, torch

    def check(self, phase, scope="load"):
        try:
            row = self.gate.check(phase, scope)
            if self.torch is not None and scope == "load":
                allocated = self.torch.cuda.memory_allocated(0)
                reserved = self.torch.cuda.memory_reserved(0)
                require(0 <= allocated <= reserved <= STOP, "Own native allocator soft stop")
            return row
        except ScoreAbort:
            raise
        except BaseException as exc:
            raise ScoreAbort("Score resource/evidence refusal at " + phase) from exc


class Predictor:
    def __init__(self, model, torch, admission, output, request_factory, model_check=None):
        self.model, self.torch, self.admission = model, torch, admission
        self.output, self.request_factory = Path(output), request_factory
        self.calls, self.stage, self.active = 0, "not-started", None
        self.model_check = model_check

    def progress(self):
        write(
            self.output / "progress.json",
            {
                "stage": self.stage,
                "started_predictions": self.calls,
                "completed_predictions": self.calls if self.active is None else self.calls - 1,
                "active_prediction": self.active,
                "monotonic": time.monotonic(),
            },
        )

    def hook(self, _module, _args):
        self.admission.check("actual-backbone-forward:" + self.stage)

    def __call__(self, state, questions):
        self.admission.check("before-predict:" + self.stage)
        if self.model_check is not None:
            self.model_check()
        index = self.calls
        self.calls += 1
        request = {"state": state, "questions": questions}
        started = time.perf_counter()
        self.active = index
        self.progress()
        append(
            self.output / "predictions.jsonl",
            {
                "event": "started",
                "index": index,
                "stage": self.stage,
                "request": request,
            },
        )
        try:
            self.torch.cuda.synchronize()
            response = self.model.predict(self.request_factory(state=state, questions=questions))
            self.torch.cuda.synchronize()
        except BaseException as exc:
            try:
                append(
                    self.output / "predictions.jsonl",
                    {
                        "event": "failed",
                        "index": index,
                        "stage": self.stage,
                        "seconds": time.perf_counter() - started,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    },
                )
            except BaseException as secondary:
                exc.add_note("Prediction failure evidence failed: " + repr(secondary))
            if isinstance(exc, getattr(self.torch.cuda, "OutOfMemoryError", ())):
                raise ScoreAbort("CUDA capacity refusal during " + self.stage) from exc
            raise
        finally:
            self.active = None
        append(
            self.output / "predictions.jsonl",
            {
                "event": "returned",
                "index": index,
                "stage": self.stage,
                "seconds": time.perf_counter() - started,
                "response": response,
            },
        )
        self.progress()
        if self.calls % 10 == 0:
            print(json.dumps({"stage": self.stage, "model_calls": self.calls}), flush=True)
        return response

    def answers(self, state, questions):
        return self(state, questions)["answers"]


def run_stages(predictor, identity, output, run_public, jb):
    predictor.stage = "legacy-public231"
    predictor.progress()
    public = run_public(predictor, HEAVEN, output / "public231", identity)
    manifest = json.loads((output / "public231/manifest.json").read_text())
    require(manifest["n_attempted"] == 231, "All legacy231 tasks must be attempted")
    result = {"legacy_public231": public, "coherence_mini": None}
    write(output / "stage-results.json", result)
    predictor.stage = "jevbench-mini"
    predictor.progress()
    report = jb.evaluate(
        jb.from_callable(predictor.answers, name=PROFILE),
        suite="jevbench-mini",
        cache=str(output / "coherence-cache"),
        concurrency=1,
        progress=True,
    )
    report.save(output / "coherence-mini.json")
    require(report.result["requests"] == 1248, "Complete fresh mini native request population")
    require(not report.result.get("truncated_by_budget"), "Mini budget must not truncate")
    require(len(report.result["case_domains"]) == 240, "Complete mini case population")
    require(len(report.result["planned_tests"]) == 50, "Complete mini relation population")
    require(
        sum(report.result["planned_tests"].values()) == 1200,
        "Complete mini planned test population",
    )
    result["coherence_mini"] = {
        "overall": report.overall,
        "interval": report.interval,
        "dimensions": report.dimensions,
        "errors": report.errors,
        "scores": report.scores,
        "requests": report.result["requests"],
        "cases": 240,
        "relations": 50,
    }
    write(output / "stage-results.json", result)
    return result


def install_stop_handlers():
    def stop(signum, _frame):
        reason = "7140-second deadline" if signum == signal.SIGALRM else "external stop"
        raise ScoreAbort("Owned cooperative " + reason)

    for signum in (signal.SIGALRM, signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
    signal.alarm(7140)


def actor_identity(expected_boot):
    if os.getsid(0) != os.getpid() or os.getpgrp() != os.getpid():
        os.setsid()
    raw = (Path("/proc") / str(os.getpid()) / "stat").read_text()
    fields = raw[raw.rfind(")") + 2 :].split()
    actor = {
        "pid": os.getpid(),
        "pgid": os.getpgrp(),
        "sid": os.getsid(0),
        "start_ticks": int(fields[19]),
    }
    require(actor["pid"] == actor["pgid"] == actor["sid"], "Owned main process/session")
    require(
        Path("/proc/sys/kernel/random/boot_id").read_text().strip() == expected_boot,
        "Expected boot identity",
    )
    require(
        Path("/proc/self/cgroup").read_text().strip().endswith("/" + UNIT),
        "Exact score cgroup unit required",
    )
    return actor


def execute(expected_boot):
    require(os.environ.get("JEV_JEVBENCH_EXECUTE") == EXECUTE, "Explicit execution environment")
    require(os.environ.get("JEV_JEVBENCH_UNIT") == UNIT, "Explicit expected owned unit")
    for key, value in {
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTORCH_ALLOC_CONF": "backend:native",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }.items():
        require(os.environ.get(key) == value, "Required execution environment: " + key)
    require(not OUTPUT.exists(), "Fresh immutable score output required")
    require(not (HERE / "actor-intent-02.json").exists(), "Fresh score actor intent required")
    actor = actor_identity(expected_boot)
    OUTPUT.mkdir(parents=True)
    write(
        OUTPUT / "actor-intent.json",
        {
            "actor": actor,
            "unit": UNIT,
            "argv": sys.argv,
            "boot_id": expected_boot,
            "outer_closure_required": True,
        },
    )
    write(
        HERE / "actor-intent-02.json",
        {
            "actor": actor,
            "unit": UNIT,
            "argv": sys.argv,
            "boot_id": expected_boot,
            "output": str(OUTPUT),
            "outer_closure_required": True,
        },
    )
    write(OUTPUT / "progress.json", {"stage": "setup", "completed_predictions": 0})

    install_stop_handlers()
    from jev_native_v2_browser import ResourceJournal
    from jev_public_action_artifact import BASE_CHECKPOINT, resolve_base_checkpoint
    from jev_public_action_artifact import pin as stream_pin

    from s1.browser_evidence import capture_runtime_sources, verify_runtime_sources
    from s1.browser_native_resources import EXPECTED_UUID, capture_components, mechanism

    gate = Admission(ResourceJournal(OUTPUT / "resources.jsonl", expected_boot))
    receipt = {
        "schema_id": "jevbench-gx10-local-score-v1",
        "profile": PROFILE,
        "stage": "setup",
        "model_constructor_calls": 0,
        "model_load_returned": False,
        "outer_closure_required": True,
        "process_exit_confirmed": False,
        "official_rank": None,
        "training_executed": False,
        "plan": ["legacy-public231", "jevbench-mini"],
        "unit": UNIT,
        "actor": actor,
    }
    primary, runtime, references, checkpoint, checkpoint_pin, handle = (
        None,
        None,
        None,
        None,
        None,
        None,
    )
    try:
        gate.check("before-framework-import")
        require(git_revision(HEAVEN) == PUBLIC_REVISION, "Pinned legacy public source commit")
        require(
            git_revision(COHERENCE) == COHERENCE_REVISION, "Current official mini source commit"
        )
        references = capture_sources(OUTPUT)
        runtime = capture_runtime_sources(OUTPUT)
        receipt["resource_components"] = capture_components(OUTPUT / "resource-components")
        checkpoint, checkpoint_pin = authenticate_checkpoint(
            CACHE, resolve_base_checkpoint, stream_pin, BASE_CHECKPOINT
        )
        metadata = {
            p.name: stream_pin(p.resolve(strict=True))
            for p in sorted(CACHE.iterdir())
            if p.is_file() and p.suffix in {".json", ".jinja", ".txt", ".model"}
        }
        receipt["checkpoint"] = {"path": str(checkpoint), **checkpoint_pin}
        receipt["metadata_files"] = metadata
        gate.check("immediately-before-torch-import")
        import torch
        from run_jevbench_public import run_public

        from s1.contracts import DecisionRequest
        from s1.unified import UnifiedDecisionModel

        gate.torch = torch
        torch.set_num_threads(2)
        torch.manual_seed(20261022)
        torch.backends.cuda.enable_cudnn_sdp(False)
        receipt["allocator_cap"] = mechanism().allocator_cap(torch, EXPECTED_UUID)
        gate.check("immediately-before-model-constructor")
        receipt["model_constructor_calls"] += 1
        write(
            OUTPUT / "model-constructor-intent.json",
            {
                "model_id": MODEL,
                "revision": REVISION,
                "quantization": "none",
                "precision": "bfloat16",
                "context": 16384,
                "time_monotonic": time.monotonic(),
            },
        )
        model = UnifiedDecisionModel(
            str(CACHE),
            revision=REVISION,
            device="cuda",
            precision="bfloat16",
            quantization="none",
            attn_implementation="sdpa",
            max_context=16384,
        )
        receipt["model_load_returned"] = True
        require(
            model.temperature == TEMPERATURE and model.quantization == "none",
            "Original BF16 released model/temperature",
        )
        require(
            str(model.head.weight.dtype) == "torch.bfloat16"
            and not hasattr(model.lm, "peft_config"),
            "Original BF16 head without LoRA",
        )
        require(
            not model.candidate_cache_enabled and model.compile_mode is None,
            "Declared no cache/no compile",
        )
        original_lm, original_head, original_backbone = model.lm, model.head, model.backbone
        head_version = model.head.weight._version

        def model_check():
            require(
                model.lm is original_lm
                and model.head is original_head
                and model.backbone is original_backbone
                and model.head.weight._version == head_version
                and model.temperature == TEMPERATURE
                and model.quantization == "none"
                and not hasattr(model.lm, "peft_config")
                and not model.candidate_cache_enabled
                and model.compile_mode is None
                and model.max_context == 16384,
                "Original model/head/temperature/profile identity changed",
            )

        identity = {
            "model_id": MODEL,
            "revision": REVISION,
            "profile": PROFILE,
            "precision": "bfloat16",
            "quantization": "none",
            "temperature": model.temperature,
            "context": model.max_context,
            "question_mode": "causal_multislot",
            "head": {
                "shape": list(model.head.weight.shape),
                "dtype": str(model.head.weight.dtype),
                "version": model.head.weight._version,
            },
            "gpu": torch.cuda.get_device_name(0),
            "allocator": receipt["allocator_cap"],
            "versions": {
                p: importlib.metadata.version(p)
                for p in ("torch", "transformers", "numpy", "httpx", "safetensors")
            },
            "checkpoint": receipt["checkpoint"],
            "metadata_files": metadata,
            "public_commit": PUBLIC_REVISION,
            "coherence_commit": COHERENCE_REVISION,
            "source_manifest_sha256": runtime["identity_sha256"],
            "cost_usd": None,
            "official_rank": None,
            "training_executed": False,
        }
        write(OUTPUT / "model-loaded.json", identity)
        predictor = Predictor(model, torch, gate, OUTPUT, DecisionRequest, model_check)
        handle = model.backbone.register_forward_pre_hook(predictor.hook)
        sys.path.insert(0, str(COHERENCE / "src"))
        import jevbench as jb

        require(
            Path(jb.__file__).resolve().is_relative_to(COHERENCE / "src"),
            "Official isolated coherence import origin",
        )
        receipt["identity"] = identity
        receipt["stage"] = "evaluating"
        write(OUTPUT / "receipt.json", receipt)
        receipt["results"] = run_stages(predictor, identity, OUTPUT, run_public, jb)
        gate.check("after-final-inference")
        receipt["model_calls"] = predictor.calls
        receipt["stage"] = "completed"
        write(
            OUTPUT / "progress.json",
            {
                "stage": "completed",
                "completed_predictions": predictor.calls,
            },
        )
    except BaseException as exc:
        primary = exc
        receipt["stage"] = "failed"
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        signal.alarm(0)
        try:
            if handle is not None:
                handle.remove()
            gate.check("post-model-work-drain", "cpu")
            receipt["runtime_sources_unchanged"] = (
                verify_runtime_sources(runtime) if runtime is not None else None
            )
            receipt["score_sources_unchanged"] = (
                verify_sources(references) if references is not None else None
            )
            receipt["checkpoint_unchanged"] = (
                stream_pin(checkpoint, limit=32 * 1024**3) == checkpoint_pin
                if checkpoint is not None and checkpoint_pin is not None
                else None
            )
            receipt["metadata_unchanged"] = (
                {
                    p.name: stream_pin(p.resolve(strict=True))
                    for p in sorted(CACHE.iterdir())
                    if p.is_file() and p.suffix in {".json", ".jinja", ".txt", ".model"}
                }
                == receipt["metadata_files"]
                if "metadata_files" in receipt
                else None
            )
            if receipt["stage"] == "completed":
                require(
                    receipt["runtime_sources_unchanged"]["success"]
                    and receipt["score_sources_unchanged"]
                    and receipt["checkpoint_unchanged"]
                    and receipt["metadata_unchanged"],
                    "Source/checkpoint identity must remain unchanged",
                )
        except BaseException as exc:
            receipt["finalization_error"] = {"type": type(exc).__name__, "message": str(exc)}
            receipt["stage"] = "failed"
            if primary is None:
                primary = exc
            else:
                primary.add_note("Score finalizer failed: " + repr(exc))
        try:
            write(OUTPUT / "receipt.json", receipt)
        except BaseException as exc:
            if primary is None:
                primary = exc
            else:
                primary.add_note("Score receipt final write failed: " + repr(exc))
    if primary is not None:
        raise primary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--expected-boot-id")
    args = parser.parse_args(argv)
    if not args.run:
        print("DRY_NO_MODEL_OR_OUTPUT")
        return
    require(args.expected_boot_id, "Explicit expected boot ID required")
    execute(args.expected_boot_id)


if __name__ == "__main__":
    main()
