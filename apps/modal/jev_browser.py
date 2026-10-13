"""Ephemeral, matched real-browser runs using an immutable Jev checkout."""

from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/workspace")
UPSTREAM_REVISION = "1231850a0bf1a0c0341fe408ef1668dbbfdfac46"
BASE_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
SPECIALIST_REVISION = "a66f836b56605039fe040f330180e336d19b3362"
app = modal.App("gemma-s1-jev-browser")
volume = modal.Volume.from_name("gemma-unified-system-one", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_sync(
        uv_project_dir=ROOT,
        extras=["inference", "train", "api"],
        frozen=True,
        extra_options="--no-dev",
    )
    .uv_pip_install(
        "torch==2.10.0", "torchvision==0.25.0", "playwright==1.58.0", "httpx[http2]==0.28.1"
    )
    .run_commands(
        "python -m playwright install --with-deps chromium",
        "git clone https://github.com/browser-use/jev-ultrafast.git /opt/jev-ultrafast",
        f"git -C /opt/jev-ultrafast checkout {UPSTREAM_REVISION}",
    )
    .env(
        {
            "HF_HOME": "/vol/hf",
            "TOKENIZERS_PARALLELISM": "false",
            "USE_TF": "0",
            "PYTHONPATH": "/workspace/src",
        }
    )
    .add_local_dir(ROOT / "src" / "s1", "/workspace/src/s1")
    .add_local_dir(ROOT / "scripts", "/workspace/scripts")
    .add_local_dir(ROOT / "configs", "/workspace/configs")
)


@app.function(image=image, volumes={"/vol": volume}, timeout=120, max_containers=1)
def probe():
    import importlib.metadata
    import subprocess

    specialist_snapshot = (
        Path("/vol/hf/hub/models--knowlet--Gemma-4-12B-Unified-System-One/snapshots")
        / SPECIALIST_REVISION
    )

    return {
        "upstream_revision": subprocess.check_output(
            ["git", "-C", "/opt/jev-ultrafast", "rev-parse", "HEAD"], text=True
        ).strip(),
        "specialist_cached": (specialist_snapshot / "config.json").is_file()
        and (specialist_snapshot / "model.safetensors").is_file(),
        "specialist_revision": SPECIALIST_REVISION,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "playwright", "httpx", "pydantic")
        },
    }


def field_text_helper(model, context):
    """Use the same genuine base-model generation helper in every cloud arm."""
    from s1.browser_text import generate_field_text

    return generate_field_text(model, context)


def _record_failure(receipt, stage, exc):
    error = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
    receipt.setdefault("failures", []).append(error)
    # Preserve the original failure if cleanup or persistence also fails.
    if "error" not in receipt:
        receipt["error"] = error
        receipt["stage"] = stage
    receipt["status"] = "failed"


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4,
    memory=49152,
    volumes={"/vol": volume},
    timeout=3600,
    max_containers=1,
    single_use_containers=True,
    scaledown_window=2,
)
def benchmark_cell(run: str, cell: str, repeats: int, split: str, tasks: str):
    import time
    import traceback

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", run):
        raise ValueError("run must be a unique lowercase identifier")
    valid_cells = {
        "native-base",
        "native-specialist",
        "user-base",
        "user-specialist",
        "reasoned-base",
        "reasoned-specialist",
        "jev-omni",
    }
    if cell not in valid_cells or split not in {"development", "holdout"} or not 1 <= repeats <= 10:
        raise ValueError("invalid browser campaign configuration")
    output = Path("/vol/browser-runs") / run / cell
    if output.exists():
        raise ValueError("refusing to overwrite a previous browser attempt")
    output.mkdir(parents=True)
    started = time.perf_counter()
    receipt = {
        "run": run,
        "cell": cell,
        "split": split,
        "repeats": repeats,
        "upstream_revision": UPSTREAM_REVISION,
        "status": "initializing",
        "stage": "runtime_import",
        "model": None,
        "text_helper": None,
        "requested_text_helper": {"model": "google/gemma-4-12B-it", "revision": BASE_REVISION},
        "gpu": None,
        "setup_ms": None,
        "modal_app_id": None,
        "modal_call_id": None,
        "cost_usd": None,
        "versions": {},
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    server_socket = server = thread = None
    thread_started = False
    source_manifest = None
    source_files = {"apps/modal/jev_browser.py": Path(__file__)}
    try:
        from s1.browser_evidence import capture_runtime_sources, verify_runtime_sources

        receipt["stage"] = "source_capture"
        source_manifest = capture_runtime_sources(output, additional_files=source_files)
        receipt["source_manifest"] = source_manifest
        receipt["stage"] = "runtime_import"
        import importlib.metadata
        import socket
        import sys
        import threading

        import torch
        import uvicorn

        from s1.api import create_app
        from s1.browser_policy import BrowserDecisionBackend
        from s1.unified import UnifiedDecisionModel

        receipt["stage"] = "benchmark_import"
        sys.path.insert(0, "/workspace/scripts")
        from jev_browser_benchmark import run_benchmark

        receipt["stage"] = "runtime_metadata"
        receipt["modal_app_id"] = app.app_id
        receipt["modal_call_id"] = modal.current_function_call_id()
        for name in ("torch", "transformers", "playwright", "httpx", "pydantic"):
            receipt["versions"][name] = importlib.metadata.version(name)
        receipt["gpu"] = torch.cuda.get_device_name()
        torch.backends.cuda.enable_cudnn_sdp(False)
        receipt["attention_policy"] = {
            "cudnn_sdp_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
            "scope": "owned benchmark process, identical policy for every comparison arm",
        }

        receipt["stage"] = "helper_model_loading"
        helper = UnifiedDecisionModel(
            "google/gemma-4-12B-it",
            revision=BASE_REVISION,
            device="cuda",
            precision="bfloat16",
            temperature=1.0,
        )
        receipt["text_helper"] = {"model": helper.name, "revision": helper.revision}
        if helper.revision != BASE_REVISION:
            raise ValueError("loaded text helper differs from the requested immutable revision")
        receipt["stage"] = "backend_loading"
        if cell == "jev-omni":
            from s1.evaluation.jev_omni import JevOmniBackend
            from s1.evaluation.registry import Registry

            config = Path("/workspace/configs/benchmarks")
            registry = Registry.load(
                config / "models.toml", config / "suites.toml", config / "profiles.toml"
            )
            backend = JevOmniBackend(registry.models["jev-omni-local"])
            backend.name = cell
            telemetry = backend.telemetry()
        else:
            model = (
                helper
                if cell.endswith("base")
                else UnifiedDecisionModel(
                    "knowlet/Gemma-4-12B-Unified-System-One",
                    revision=SPECIALIST_REVISION,
                    device="cuda",
                    precision="bfloat16",
                )
            )
            expected_revision = BASE_REVISION if cell.endswith("base") else SPECIALIST_REVISION
            if model.revision != expected_revision:
                raise ValueError(
                    "loaded decision model differs from the requested immutable revision"
                )
            backend = BrowserDecisionBackend(model, prompt_mode=cell.split("-")[0])
            telemetry = {
                "checkpoint": model.name,
                "revision": model.revision,
                "temperature": model.temperature,
                "prompt_mode": cell.split("-")[0],
            }
        # Reject malformed telemetry before it can prevent receipt persistence.
        json.dumps(telemetry, allow_nan=False)
        receipt["model"] = telemetry
        receipt["stage"] = "server_start"
        server_socket = socket.socket()
        server_socket.bind(("127.0.0.1", 0))
        port = server_socket.getsockname()[1]
        application = create_app(backend, jev_compatible=True, model_identity=cell)
        server = uvicorn.Server(uvicorn.Config(application, log_level="error"))
        thread = threading.Thread(
            target=server.run, kwargs={"sockets": [server_socket]}, daemon=True
        )
        thread.start()
        thread_started = True
        deadline = time.monotonic() + 15
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.02)
        if not server.started:
            raise RuntimeError("local model API failed to start")
        receipt["setup_ms"] = (time.perf_counter() - started) * 1000
        receipt["stage"] = "benchmark"
        receipt["status"] = "running"
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        summary = run_benchmark(
            upstream_path="/opt/jev-ultrafast",
            endpoint=f"http://127.0.0.1:{port}/v1/systemone",
            model=cell,
            output_dir=output / "benchmark",
            repeats=repeats,
            max_steps=16,
            suite_split=split,
            task_ids=tasks.split(",") if tasks else None,
            text_helper=lambda context: field_text_helper(helper, context),
        )
        json.dumps(summary, allow_nan=False)
        receipt["summary"] = summary
        receipt["stage"] = "final_telemetry"
        receipt["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        receipt["stage"] = "completed"
        receipt["status"] = "completed"
    except Exception as exc:
        _record_failure(receipt, receipt["stage"], exc)
        (output / "failure.txt").write_text(traceback.format_exc())
    finally:
        if source_manifest is not None:
            try:
                receipt["source_verification"] = verify_runtime_sources(
                    source_manifest, additional_files=source_files
                )
                if not receipt["source_verification"]["success"]:
                    raise RuntimeError("runtime source changed during the model campaign")
            except Exception as exc:
                _record_failure(receipt, "source_verification", exc)
        if server is not None:
            try:
                server.should_exit = True
            except Exception as exc:
                _record_failure(receipt, "cleanup_server", exc)
        if thread_started:
            try:
                thread.join(timeout=10)
                if thread.is_alive():
                    raise RuntimeError("local model API did not stop within 10 seconds")
            except Exception as exc:
                _record_failure(receipt, "cleanup_thread", exc)
        if server_socket is not None:
            try:
                server_socket.close()
            except Exception as exc:
                _record_failure(receipt, "cleanup_socket", exc)
        receipt["elapsed_ms"] = (time.perf_counter() - started) * 1000
        if receipt["setup_ms"] is None:
            receipt["setup_ms"] = receipt["elapsed_ms"]
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        try:
            volume.commit()
        except Exception as exc:
            # A returned archive can still recover evidence when volume commit fails.
            _record_failure(receipt, "volume_commit", exc)
            (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output))
    return {"receipt": receipt, "archive": archive_buffer.getvalue()}


@app.local_entrypoint()
def main(
    stage: str = "probe",
    run: str = "",
    cells: str = "native-specialist,user-specialist",
    repeats: int = 1,
    split: str = "development",
    tasks: str = "",
):
    if stage == "probe":
        result = probe.remote()
        print(json.dumps(result, indent=2))
        return
    if stage != "benchmark" or not run:
        raise ValueError("benchmark requires --stage benchmark and --run <unique-name>")
    for cell in cells.split(","):
        destination = ROOT / "artifacts/jev-browser" / run / cell
        if destination.exists():
            raise ValueError("local browser evidence already exists")
        result = benchmark_cell.remote(run, cell, repeats, split, tasks)
        destination.mkdir(parents=True)
        with zipfile.ZipFile(io.BytesIO(result["archive"])) as archive:
            archive.extractall(destination)
        print(json.dumps(result["receipt"], indent=2), flush=True)
