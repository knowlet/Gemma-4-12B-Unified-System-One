"""Actual local OCR/ASR HTTP pipelines and workflow replay on a shared checkpoint.

The caller owns cloud allocation, model loading, and the job budget. Only Whisper
Tiny is downloaded here, at an immutable public revision and explicitly on CPU.
Gemma remains alive when individual adapters and the local HTTP service close.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

from s1.backends import HTTPBackend
from s1.evaluation.adapters import ReferenceAdapter
from s1.evaluation.artifacts import read_records, write_json
from s1.evaluation.contracts import Budget, ProfileSpec, SuiteSpec
from s1.evaluation.datasets import fingerprint, load_cases
from s1.evaluation.multimodal import PipelineAdapter, counterfactual_summary, prepare_media_views
from s1.evaluation.registry import Registry
from s1.evaluation.runner import execute
from s1.evaluation.workflow import execute_workflows

WHISPER_MODEL = "openai/whisper-tiny"
WHISPER_REVISION = "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
REQUEST_COST_CEILING_USD = 0.02


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _digest(value) -> str:
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _sync(model):
    if str(model.device).startswith("cuda"):
        import torch

        torch.cuda.synchronize(model.device)


class _SharedAdapter(ReferenceAdapter):
    """Measure real synchronized decisions while preserving caller-owned weights."""

    def __init__(self, spec, model, trace):
        super().__init__(spec, SimpleNamespace(model=model))
        self.model, self.trace = model, Path(trace)

    def predict(self, request):
        _sync(self.model)
        started, error = time.perf_counter(), None
        try:
            result = super().predict(request)
            _sync(self.model)
            return result
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)[:1000]}
            raise
        finally:
            with self.trace.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "model_id": self.spec.id,
                            "questions": len(request.questions),
                            "modalities": sorted({"text", *(m.type for m in request.media)}),
                            "synchronized_decision_wall_seconds": time.perf_counter() - started,
                            "error": error,
                        }
                    )
                    + "\n"
                )

    def close(self):
        self.backend = None


class _Preprocessor:
    """Recognize supplied bytes; this class never receives case IDs or labels."""

    def __init__(self, output: Path):
        self.output = Path(output)
        self.trace = self.output / "preprocessing-http-calls.jsonl"
        self.lock = threading.Lock()
        self.whisper = self.processor = None
        self.tesseract = shutil.which("tesseract")
        self.setup_errors = {}
        self.metadata = {
            "transport": "real loopback HTTP; FastAPI and uvicorn",
            "response_cost_usd": None,
            "billing_policy": "no provider bill; missing cost remains unknown",
            "result_cache": False,
            "gold_or_oracle_input": False,
            "whisper": {
                "model_id": WHISPER_MODEL,
                "requested_revision": WHISPER_REVISION,
                "license": "Apache-2.0",
                "source": f"https://huggingface.co/{WHISPER_MODEL}/tree/{WHISPER_REVISION}",
                "device": "cpu",
                "precision": "float32",
                "generation": {
                    "language": "en",
                    "task": "transcribe",
                    "do_sample": False,
                    "max_new_tokens": 32,
                    "return_timestamps": False,
                },
            },
            "tesseract": {
                "arguments": [
                    "--psm",
                    "10",
                    "-l",
                    "eng",
                    "--dpi",
                    "300",
                    "-c",
                    "tessedit_char_whitelist=0123456789",
                ],
                "source": "https://github.com/tesseract-ocr/tesseract",
                "license": "Apache-2.0",
            },
        }

    def setup(self):
        started = time.perf_counter()
        try:
            if not self.tesseract:
                raise FileNotFoundError("tesseract executable is not installed")
            banner = subprocess.run(
                [self.tesseract, "--version"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
            languages = subprocess.run(
                [self.tesseract, "--list-langs"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout
            self.metadata["tesseract"].update(
                resolved_version=banner.splitlines()[0],
                executable_sha256=_sha(Path(self.tesseract).read_bytes()),
                available_languages=languages,
            )
            location = re.search(r'"([^"\n]+)"', languages)
            if not location or not (Path(location[1]) / "eng.traineddata").is_file():
                raise FileNotFoundError("cannot resolve the actual Tesseract English weights")
            self.metadata["tesseract"]["eng_traineddata_sha256"] = _sha(
                (Path(location[1]) / "eng.traineddata").read_bytes()
            )
        except Exception as exc:
            self.setup_errors["ocr"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
        try:
            import torch
            from huggingface_hub import HfApi, snapshot_download
            from transformers import WhisperForConditionalGeneration, WhisperProcessor

            info = HfApi(token=False).model_info(WHISPER_MODEL, revision=WHISPER_REVISION)
            if info.sha != WHISPER_REVISION:
                raise ValueError("resolved Whisper revision differs from its immutable pin")
            snapshot = Path(
                snapshot_download(
                    WHISPER_MODEL,
                    revision=WHISPER_REVISION,
                    token=False,
                    allow_patterns=["*.json", "*.safetensors", "*.txt", "README.md"],
                )
            )
            self.processor = WhisperProcessor.from_pretrained(snapshot, local_files_only=True)
            self.whisper = (
                WhisperForConditionalGeneration.from_pretrained(
                    snapshot,
                    local_files_only=True,
                    use_safetensors=True,
                    dtype=torch.float32,
                )
                .to("cpu")
                .eval()
            )
            self.whisper.generation_config.forced_decoder_ids = None
            self.metadata["whisper"].update(
                resolved_revision=info.sha,
                resolved_device=str(next(self.whisper.parameters()).device),
                resolved_precision=str(next(self.whisper.parameters()).dtype),
                files_sha256={
                    p.name: _sha(p.read_bytes()) for p in sorted(snapshot.iterdir()) if p.is_file()
                },
            )
        except Exception as exc:
            self.setup_errors["asr"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
        self.metadata.update(
            setup_seconds=time.perf_counter() - started,
            setup_errors=self.setup_errors,
            packages={
                name: version(name) for name in ("fastapi", "uvicorn", "transformers", "torch")
            },
        )
        identity = {
            key: value
            for key, value in self.metadata.items()
            if key not in ("setup_seconds", "setup_errors")
        }
        identity["service_source_sha256"] = _sha(Path(__file__).read_bytes())
        self.metadata["service_revision"] = _digest(identity)
        write_json(self.output / "preprocessor-manifest.json", self.metadata)

    def _ocr(self, medium):
        from PIL import Image

        if "ocr" in self.setup_errors:
            raise RuntimeError("OCR setup failed; see preprocessor-manifest.json")
        raw = base64.b64decode(medium["data"], validate=True)
        with tempfile.TemporaryDirectory(prefix="s1-live-ocr-") as directory:
            image_path = Path(directory) / "input.png"
            Image.open(io.BytesIO(raw)).convert("RGB").save(image_path)
            result = subprocess.run(
                [
                    self.tesseract,
                    str(image_path),
                    "stdout",
                    *self.metadata["tesseract"]["arguments"],
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
        return result.stdout.strip()

    def _asr(self, medium):
        import numpy as np
        import torch

        if "asr" in self.setup_errors:
            raise RuntimeError("ASR setup failed; see preprocessor-manifest.json")
        if medium.get("sampling_rate", 16000) != 16000:
            raise ValueError("ASR input must use the 16kHz media contract")
        inputs = self.processor(
            np.asarray(medium["samples"], dtype=np.float32),
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )
        with torch.inference_mode():
            tokens = self.whisper.generate(
                input_features=inputs.input_features.to("cpu"),
                attention_mask=inputs.attention_mask.to("cpu"),
                **self.metadata["whisper"]["generation"],
            )
        return self.processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()

    def process(self, payload):
        started = time.perf_counter()
        stages, error = [], None
        record = {
            "payload_keys": sorted(payload),
            "media_sha256": _digest(payload.get("media")),
            "stages": stages,
        }
        try:
            if set(payload) != {"media"} or not payload["media"]:
                raise ValueError("preprocessing receives only nonempty media")
            observations = []
            for medium in payload["media"]:
                stage_started = time.perf_counter()
                stage = {"type": medium.get("type"), "recognized_text": None, "error": None}
                try:
                    if medium.get("type") == "image":
                        text = self._ocr(medium)
                        observations.append("OCR text: " + (text or "[no recognized characters]"))
                    elif medium.get("type") == "audio":
                        text = self._asr(medium)
                        observations.append(
                            "Audio transcription: " + (text or "[no recognized speech]")
                        )
                    else:
                        raise ValueError("unsupported preprocessing media type")
                    stage["recognized_text"] = text
                except Exception as exc:
                    stage["error"] = {"type": type(exc).__name__, "message": str(exc)[:1000]}
                    raise
                finally:
                    stage["elapsed_ms"] = (time.perf_counter() - stage_started) * 1000
                    stages.append(stage)
            record["status"] = "ok"
            return {"text": "\n".join(observations)}
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)[:1000]}
            raise
        finally:
            record.update(error=error, elapsed_ms=(time.perf_counter() - started) * 1000)
            with self.lock, self.trace.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")


class _LocalHTTPService:
    def __init__(self, preprocessor):
        import uvicorn
        from fastapi import FastAPI, HTTPException

        self.preprocessor = preprocessor
        app = FastAPI()

        @app.post("/preprocess")
        def preprocess(payload: dict):
            try:
                return preprocessor.process(payload)
            except Exception as exc:
                raise HTTPException(status_code=500, detail="preprocessing stage failed") from exc

        self.socket = socket.socket()
        self.socket.bind(("127.0.0.1", 0))
        self.endpoint = f"http://127.0.0.1:{self.socket.getsockname()[1]}/preprocess"
        self.server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
        self.thread = threading.Thread(
            target=self.server.run,
            kwargs={"sockets": [self.socket]},
            daemon=True,
        )

    def __enter__(self):
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started:
            if not self.thread.is_alive() or time.monotonic() >= deadline:
                self.close()
                raise RuntimeError("local preprocessing HTTP server failed to start")
            time.sleep(0.05)
        return self

    def close(self):
        self.server.should_exit = True
        if self.thread.is_alive():
            self.thread.join(timeout=10)
        self.socket.close()

    def __exit__(self, *args):
        self.close()


def run(model, root: Path, output: Path, dataset_dir: Path) -> dict:
    """Execute actual M0/M2/M3 and workflow paths while retaining shared Gemma."""
    root, output, dataset_dir = Path(root), Path(output), Path(dataset_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    original_temperature = model.temperature
    views = output / "media-views"
    manifest = prepare_media_views(dataset_dir / "media-bundle.jsonl", views)
    write_json(
        output / "media-source-provenance.json",
        json.loads((dataset_dir / "provenance.json").read_text()),
    )
    registry = Registry.load(
        root / "configs/benchmarks/models.toml",
        root / "configs/benchmarks/suites.toml",
        root / "configs/benchmarks/profiles.toml",
    )
    models = {}
    for name in ("gemma-g2", "gemma-g4"):
        models[name] = registry.models[name].model_copy(
            update={
                "enabled": True,
                "model_id": model.name,
                "revision": model.revision,
                "processor_revision": model.revision,
                "device": str(model.device),
                "context_limit": model.max_context,
                "calibration": "none",
                "cost_per_request_usd": REQUEST_COST_CEILING_USD,
            }
        )
    registry = replace(registry, models=models)
    trace = output / "decision-calls.jsonl"
    summary = {
        "kind": "live_ocr_asr_http_pipeline_and_workflow_replay",
        "status": "running",
        "checkpoint": model.name,
        "revision": model.revision,
        "device": str(model.device),
        "media_view_manifest": manifest,
        "runs": {},
        "failures": [],
        "reported_total_cost_usd": None,
        "cost_ceiling_policy": "operator reservation only; GPU billing is owned by parent job",
        "limitations": [
            "32 handwritten digits and 20 spoken digits are task-specific screening samples.",
            "M2 is a source-label oracle diagnostic; no independent new human verification.",
            "Tesseract and Whisper predictions are passed verbatim, including recognition errors.",
            "Workflow small/strong share checkpoint weights; G2/G4 differ only in execution policy.",
            "Replay fixtures do not execute actual CI repairs or UI actions.",
        ],
    }

    def persist():
        summary["allocated_device_wall_seconds"] = time.perf_counter() - started
        calls = (
            [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
        )
        summary["synchronized_decision_wall_seconds"] = sum(
            row["synchronized_decision_wall_seconds"] for row in calls
        )
        summary["decision_calls"] = len(calls)
        write_json(output / "receipt.json", summary)

    def benchmark(view, spec, environment=None):
        cases = load_cases(views / f"{view}.jsonl")
        name = f"media-{view.lower()}"
        suite = SuiteSpec(
            id=name,
            dataset=str((views / f"{view}.jsonl").resolve()),
            track="pipeline" if view == "M3" else "generalist",
            information_view="pipeline_output"
            if view == "M3"
            else "verified_transcript"
            if view == "M2"
            else "raw_text",
            dataset_sha256=fingerprint(cases),
        )
        profile = ProfileSpec(
            id=name,
            suite=name,
            models=(spec.id,),
            budget=Budget(
                max_requests=len(cases),
                max_estimated_cost_usd=len(cases) * REQUEST_COST_CEILING_USD + 0.01,
            ),
        )
        configured = replace(
            registry, models={spec.id: spec}, suites={name: suite}, profiles={name: profile}
        )
        write_json(
            output / f"{name}-config.json",
            {
                "model": spec.model_dump(mode="json"),
                "suite": suite.model_dump(mode="json"),
                "profile": profile.model_dump(mode="json"),
            },
        )

        def factory(selected, env):
            adapter = _SharedAdapter(selected, model, trace)
            if view == "M3":
                return PipelineAdapter(
                    adapter, HTTPBackend(env["LIVE_MEDIA_PREPROCESSOR"], timeout=120)
                )
            return adapter

        print(f"LIVE_PIPELINE {view}: {len(cases)} cases", flush=True)
        result = execute(
            configured,
            name,
            output=output / name,
            lockfile=root / "uv.lock",
            environment=environment,
            adapter_factory=factory,
        )
        predictions = read_records(output / name, "predictions")
        entry = {
            "run_status": result["run_status"],
            "expected_predictions": len(cases),
            "actual_predictions": len(predictions),
            "records_complete": len(predictions) == len(cases),
            "summary": result,
        }
        if (
            view == "M3"
            and entry["records_complete"]
            and result["run_status"] in ("completed", "completed_with_errors")
        ):
            entry["counterfactual"] = counterfactual_summary(
                output / name, spec.id, views / "media-views.json"
            )
        summary["runs"][view] = entry
        if result["run_status"] != "completed" or not entry["records_complete"]:
            summary["failures"].append({"check": view, "status": result["run_status"]})
        persist()

    model.set_temperature(1.0)
    try:
        benchmark("M0", models["gemma-g2"])
        benchmark("M2", models["gemma-g2"])
        preprocessor = _Preprocessor(output)
        preprocessor.setup()
        summary["preprocessor"] = preprocessor.metadata
        with _LocalHTTPService(preprocessor) as service:
            spec = models["gemma-g2"].model_copy(
                update={
                    "id": "gemma-g2-m3",
                    "preprocessing": "http_media",
                    "preprocessor_endpoint_env": "LIVE_MEDIA_PREPROCESSOR",
                    "preprocessor_revision": preprocessor.metadata["service_revision"],
                }
            )
            benchmark("M3", spec, {"LIVE_MEDIA_PREPROCESSOR": service.endpoint})
        print("LIVE_PIPELINE workflow replay: G2/G4 shared checkpoint", flush=True)
        episodes_source = root / "examples/benchmarks/workflows.jsonl"
        episodes_archive = output / "workflow-source.jsonl"
        shutil.copyfile(episodes_source, episodes_archive)
        write_json(
            output / "workflow-config.json",
            {key: spec.model_dump(mode="json") for key, spec in models.items()},
        )
        result = execute_workflows(
            registry,
            episodes_archive,
            "gemma-g2",
            "gemma-g4",
            output=output / "workflow",
            lockfile=root / "uv.lock",
            max_calls=1000,
            max_cost_usd=20.0,
            adapter_factory=lambda spec, environment: _SharedAdapter(spec, model, trace),
        )
        summary["workflow"] = result
        if result["status"] != "completed":
            summary["failures"].append({"check": "workflow", "status": result["status"]})
        summary["status"] = "completed_with_errors" if summary["failures"] else "completed"
    except Exception as exc:
        summary["status"] = "completed_with_errors"
        summary["failures"].append(
            {"check": "module", "error_type": type(exc).__name__, "message": str(exc)[:1000]}
        )
    finally:
        model.set_temperature(original_temperature)
        persist()
    return summary
