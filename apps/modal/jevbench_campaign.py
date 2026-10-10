"""Sequential A100 public screening; official leaderboard submission is separate.

uv run --no-sync modal run apps/modal/jevbench_campaign.py --run <unique-id>
"""

import json
import re
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/workspace")
app = modal.App("gemma-jevbench-public")
volume = modal.Volume.from_name("gemma-unified-system-one", create_if_missing=False)
HEAVEN_REV = "bb05a335bc809e61b20c0f745d25499a82b326fc"
COHERENCE_REV = "e18733694623aa93058e279c3534f8b8e2edefa3"
ONEJEV_CODE = "7e3007d7829c7f59f5af15c60fb8454a11e37170"
DECIDER_CODE = "e50e549b47e2da69223734fee4efa1ddd4528e93"
MODELS = {
    "s1-bf16": (
        "knowlet/Gemma-4-12B-Unified-System-One",
        "a66f836b56605039fe040f330180e336d19b3362",
    ),
    "s1-compiled": (
        "knowlet/Gemma-4-12B-Unified-System-One",
        "a66f836b56605039fe040f330180e336d19b3362",
    ),
    "s1-independent": (
        "knowlet/Gemma-4-12B-Unified-System-One",
        "a66f836b56605039fe040f330180e336d19b3362",
    ),
    "onejev-4b": ("OmniJev/OneJev-4B", "c88e18653ceb7a8770716287f55fdefc79d6b588"),
    "decider-2b": ("Mapika/decider-2b", "533964dae8be954c5b5e19fa4948e48408094c1e"),
    "jev-omni": ("akhilaaa3/Jev-Omni", "5addda86ddee081a68fb067477ea100c221b8917"),
}
base = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "torch==2.10.0",
        "torchvision==0.25.0",
        "transformers==5.17.0",
        "accelerate==1.15.0",
        "huggingface-hub==1.33.0",
        "safetensors==0.8.0",
        "pillow==12.3.0",
        "soundfile==0.14.0",
        "librosa==0.11.0",
        "sentencepiece==0.2.2",
        "protobuf==6.33.6",
        "numpy==2.5.3",
        "pydantic==2.13.5",
        "httpx==0.28.1",
    )
    .run_commands(
        "git clone https://github.com/fstandhartinger/jevbench /opt/heaven",
        f"git -C /opt/heaven checkout {HEAVEN_REV}",
        "git clone https://github.com/JevBench/jevbench /opt/coherence",
        f"git -C /opt/coherence checkout {COHERENCE_REV}",
    )
    .env({"HF_HOME": "/vol/hf", "TOKENIZERS_PARALLELISM": "false", "USE_TF": "0"})
)
onejev_image = base.uv_pip_install(
    f"qev @ git+https://github.com/OmniJev/OneJev@{ONEJEV_CODE}", extra_options="--no-deps"
)
decider_image = base.uv_pip_install(
    f"decider-ai @ git+https://github.com/Mapika/decider@{DECIDER_CODE}",
    extra_options="--no-deps",
).uv_pip_install("flash-linear-attention==0.4.2")


def sources(image):
    return (
        image.add_local_python_source("s1")
        .add_local_dir(ROOT / "scripts", "/workspace/scripts")
        .add_local_dir(ROOT / "configs", "/workspace/configs")
        .add_local_file(
            ROOT / "examples/benchmarks/mps.jsonl", "/workspace/native-regression.jsonl"
        )
    )


def _run_cell(run, name, coherence):
    import hashlib
    import importlib.metadata
    import subprocess
    import sys
    import time

    import torch
    from huggingface_hub import snapshot_download

    import s1

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", run) or name not in MODELS:
        raise ValueError("unique lowercase run and known model required")
    destination = Path("/vol/jevbench") / run / name
    destination.mkdir(parents=True, exist_ok=False)
    model_id, revision = MODELS[name]
    identity = {
        "model_id": model_id,
        "revision": revision,
        "variant": name,
        "gpu": torch.cuda.get_device_name(),
        "precision": "bfloat16",
        "benchmarkheaven_commit": subprocess.check_output(
            ["git", "-C", "/opt/heaven", "rev-parse", "HEAD"], text=True
        ).strip(),
        "coherence_commit": COHERENCE_REV,
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "numpy", "huggingface-hub", "safetensors")
        },
        "source_files": {
            "campaign": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "public_runner": hashlib.sha256(
                Path("/workspace/scripts/run_jevbench_public.py").read_bytes()
            ).hexdigest(),
            **{
                str(p.relative_to(Path(s1.__file__).parent)): hashlib.sha256(
                    p.read_bytes()
                ).hexdigest()
                for p in sorted(Path(s1.__file__).parent.rglob("*.py"))
            },
        },
        "reported_cost_usd": None,
        "official_rank": None,
        "public_exposure": "Development screening. No fitting, checkpoint selection, or temperature tuning on these cases in this campaign; upstream model training exposure may differ.",
        "coherence_requested": coherence,
        "modal_call_id": modal.current_function_call_id(),
        "modal_app_id": app.app_id,
    }
    started = time.perf_counter()
    directory = Path(
        snapshot_download(
            model_id,
            revision=revision,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.model",
                "*.jinja",
                "*.txt",
                *(["jev_omni.py", "head.pt"] if name == "jev-omni" else []),
            ],
        )
    )
    identity["config_sha256"] = hashlib.sha256((directory / "config.json").read_bytes()).hexdigest()
    sys.path.insert(0, "/workspace/scripts")
    from prepare_mlx_validation import checkpoint_identity

    identity["checkpoint_files"] = checkpoint_identity(directory)
    identity["processor_files"] = {
        p.name: {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "size_bytes": p.stat().st_size,
        }
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix in (".json", ".model", ".jinja", ".txt")
    }
    for calibration in ("s1_config.json", "calibration.json", "decider_config.json"):
        path = directory / calibration
        if path.exists():
            identity[calibration] = {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "content": json.loads(path.read_text()),
            }
    if name.startswith("s1-"):
        from s1.contracts import DecisionRequest
        from s1.unified import UnifiedDecisionModel

        model = UnifiedDecisionModel(
            str(directory), device="cuda", precision="bfloat16", attn_implementation="sdpa"
        )
        identity["execution"] = "causal_multislot; SDPA; no cache; selected 52 rows"
        if name == "s1-compiled":
            model.enable_compile("decoder-default-dynamic")
            identity["compile"] = "decoder-default-dynamic; dynamic=True; fullgraph=False"
        if name == "s1-independent":
            identity["execution"] = (
                "Existing G4 independent question batching; separate per-question prompts with real right-padding masks; SDPA; shipped calibration"
            )
            identity["development_exposure"] = (
                "This explicitly different profile was selected after observing causal-multislot BAT failures; not an untouched evaluation or the published causal profile."
            )

        def predict(state, questions):
            if name == "s1-independent":
                from s1.evaluation.gemma import predict_batch

                return predict_batch(
                    model, [DecisionRequest(state=state, questions=questions)], independent=True
                )[0]
            return model.predict(DecisionRequest(state=state, questions=questions))

    elif name == "onejev-4b":
        from qev.calibrate import Calibration
        from qev.mm_engine import MMDecisionEngine
        from qev.schema import SystemOneRequest

        model = MMDecisionEngine(
            str(directory),
            device="cuda:0",
            dtype="bfloat16",
            calibration=Calibration.load(directory / "calibration.json")
            if (directory / "calibration.json").is_file()
            else None,
            cuda_graphs=True,
        )
        identity["code_revision"] = ONEJEV_CODE
        if not (directory / "calibration.json").is_file():
            identity["calibration_policy"] = (
                "Publisher CLI default: no sidecar provided; Calibration() temperatures default to 1.0. No benchmark fitting."
            )
        identity["execution"] = (
            "publisher MMDecisionEngine; auto fork; CUDA graphs; default float32 head; debias=1"
        )

        def predict(state, questions):
            response, meta = model.decide(
                SystemOneRequest(state=state, questions=questions), debias=1
            )
            return {**response.model_dump(), "runtime": meta}

    elif name == "jev-omni":
        import tomllib

        from s1.contracts import DecisionRequest
        from s1.evaluation.contracts import ModelSpec
        from s1.evaluation.jev_omni import JevOmniBackend

        registry_path = Path("/workspace/configs/benchmarks/models.toml")
        spec_data = next(
            spec
            for spec in tomllib.loads(registry_path.read_text())["models"]
            if spec["id"] == "jev-omni-local"
        )
        model = JevOmniBackend(ModelSpec.model_validate(spec_data))
        identity["code_revision"] = revision
        identity["native_telemetry"] = model.telemetry()
        identity["spec"] = spec_data
        identity["registry_sha256"] = hashlib.sha256(registry_path.read_bytes()).hexdigest()
        identity["head_file"] = {
            "sha256": hashlib.sha256((directory / "head.pt").read_bytes()).hexdigest(),
            "size_bytes": (directory / "head.pt").stat().st_size,
        }
        identity["execution"] = (
            "Pinned publisher BF16 backbone, FP32 head; original independent per-question prompt, no graph/cache, no fitting"
        )

        def predict(state, questions):
            return model.predict(DecisionRequest(state=state, questions=questions))

    else:
        from decider.infer import Decider

        model = Decider(str(directory), device="cuda", dtype=torch.bfloat16, use_graphs=False)
        identity["code_revision"] = DECIDER_CODE
        identity["execution"] = (
            "publisher Decider eager reference; independent=True; shipped by-type temperatures/isolated-level policy"
        )

        def predict(state, questions):
            return model.system_one(state, questions)

    torch.cuda.synchronize()
    identity["setup_seconds"] = time.perf_counter() - started
    identity["memory_after_load_bytes"] = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    def synchronized_predict(state, questions):
        torch.cuda.synchronize()
        result = predict(state, questions)
        torch.cuda.synchronize()
        return result

    sys.path.insert(0, "/workspace/scripts")
    from run_jevbench_public import run_public

    public = run_public(synchronized_predict, "/opt/heaven", destination / "public231", identity)
    volume.commit()
    result = {"identity": identity, "public": public, "coherence": None, "official_rank": None}
    if name == "s1-compiled":
        from s1.evaluation.datasets import load_cases

        compiled = model.backbone.language_model
        original = compiled._orig_mod
        native = []
        for case in load_cases("/workspace/native-regression.jsonl"):
            variants = {}
            for variant, decoder, cached in (
                ("eager", original, False),
                ("cached", original, True),
                ("compiled", compiled, False),
            ):
                model.backbone.language_model = decoder
                model.enable_candidate_cache(cached)
                variants[variant] = model.predict(case.request)
                torch.cuda.synchronize()
            native.append(
                {
                    "case_id": case.id,
                    "modalities": [media.type for media in case.request.media],
                    "variants": variants,
                }
            )
        model.backbone.language_model = compiled
        model.enable_candidate_cache(False)
        (destination / "native-regression.json").write_text(
            json.dumps(native, indent=2, allow_nan=False) + "\n"
        )
        volume.commit()
    if name == "s1-independent":
        from s1.evaluation.datasets import load_cases
        from s1.evaluation.gemma import predict_batch, predict_sequential

        native = []
        for case in load_cases("/workspace/native-regression.jsonl"):
            native.append(
                {
                    "case_id": case.id,
                    "modalities": [media.type for media in case.request.media],
                    "variants": {
                        "sequential": predict_sequential(model, case.request),
                        "independent": predict_batch(model, [case.request], independent=True)[0],
                    },
                }
            )
            torch.cuda.synchronize()
        (destination / "independent-native-regression.json").write_text(
            json.dumps(native, indent=2, allow_nan=False) + "\n"
        )
        volume.commit()
    if name == "s1-bf16":
        from run_jevbench_public import load_public

        tasks, official, _ = load_public("/opt/heaven")
        # Fixed positions span all three files; selection never reads labels or scores.
        sample_indices = [0, 17, 72, 103, 120, 151, 193, 230]
        samples = []
        for index in sample_indices:
            task = tasks[index]
            questions = {"decision": official.base.build_question(task)}
            for cached in (False, True):
                model.enable_candidate_cache(cached)
                synchronized_predict(task.state, questions)
            for cached in (False, True, True, False):
                model.enable_candidate_cache(cached)
                # Prime after explicit invalidation; exclude row materialization.
                synchronized_predict(task.state, questions)
                for repeat in range(3):
                    began = time.perf_counter()
                    response = synchronized_predict(task.state, questions)
                    samples.append(
                        {
                            "task_id": task.id,
                            "cached": cached,
                            "repeat": repeat,
                            "seconds": time.perf_counter() - began,
                            "answers": response["answers"],
                        }
                    )
        model.enable_candidate_cache(True)
        cached_identity = {**identity, "candidate_cache": True}
        result["cached_public"] = run_public(
            synchronized_predict, "/opt/heaven", destination / "cached-public231", cached_identity
        )
        (destination / "candidate-cache-samples.json").write_text(
            json.dumps(samples, indent=2, allow_nan=False) + "\n"
        )
        model.enable_candidate_cache(False)
        volume.commit()
    if coherence:
        sys.path.insert(0, "/opt/coherence/src")
        import jevbench as jb

        def standard_predict(state, questions):
            response = synchronized_predict(state, questions)
            return response["answers"] if "answers" in response else response

        report = jb.evaluate(
            jb.from_callable(standard_predict, name=f"{name}@{revision}"),
            suite="jevbench-mini",
            cache=str(destination / "coherence-cache"),
            concurrency=1,
            progress=True,
        )
        report.save(destination / "coherence-mini.json")
        result["coherence"] = {
            "overall": report.overall,
            "interval": report.interval,
            "dimensions": report.dimensions,
            "errors": report.errors,
            "scores": report.scores,
        }
    result["memory_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    result["memory_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    result["memory_peak_scope"] = (
        "Entire loaded-model cell: public231, cache experiment/native regressions when applicable, and requested coherence-mini; not a controlled per-stage memory comparison."
    )
    (destination / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    volume.commit()
    import io
    import zipfile

    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(destination))
    return bundle.getvalue()


def run_cell(run, name, coherence):
    """Persist setup/stage failure without erasing finished public evidence."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", run) or name not in MODELS:
        raise ValueError("unique lowercase run and known model required")
    destination = Path("/vol/jevbench") / run / name
    if destination.exists():
        raise ValueError("output already exists; use a distinct recovery run")
    try:
        return _run_cell(run, name, coherence)
    except Exception as exc:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "failure.json").write_text(
            json.dumps(
                {
                    "status": "error",
                    "model": name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "official_rank": None,
                },
                indent=2,
            )
            + "\n"
        )
        raise
    finally:
        volume.commit()


_settings = dict(
    gpu="A100-80GB",
    cpu=4,
    memory=49152,
    volumes={"/vol": volume},
    timeout=7200,
    max_containers=1,
    single_use_containers=True,
    scaledown_window=2,
)


@app.function(image=sources(base), **_settings)
def gemma_cell(run, name, coherence):
    return run_cell(run, name, coherence)


@app.function(image=sources(onejev_image), **_settings)
def onejev_cell(run, name, coherence):
    return run_cell(run, name, coherence)


@app.function(image=sources(decider_image), **_settings)
def decider_cell(run, name, coherence):
    return run_cell(run, name, coherence)


@app.local_entrypoint()
def main(
    run: str,
    models: str = "s1-bf16,onejev-4b,decider-2b,jev-omni",
    coherence: bool = True,
    coherence_models: str = "",
):
    import io
    import zipfile

    selected = models.split(",")
    if len(set(selected)) != len(selected) or set(selected) - MODELS.keys():
        raise ValueError("unique known models required")
    coherence_selected = set(coherence_models.split(",")) if coherence_models else set(selected)
    if coherence_selected - set(selected):
        raise ValueError("coherence models must be included in selected models")
    output = ROOT / "artifacts/jevbench" / run
    output.mkdir(parents=True, exist_ok=False)
    failures = []
    for name in selected:
        print(f"Starting {name}", flush=True)
        function = (
            onejev_cell
            if name == "onejev-4b"
            else decider_cell
            if name == "decider-2b"
            else gemma_cell
        )
        try:
            data = function.remote(
                run, name, coherence and name in coherence_selected and name != "s1-compiled"
            )
        except Exception as exc:
            failures.append(name)
            (output / f"{name}-failure.json").write_text(
                json.dumps(
                    {
                        "status": "error",
                        "model": name,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "remote_evidence": f"/jevbench/{run}/{name}",
                        "official_rank": None,
                    },
                    indent=2,
                )
                + "\n"
            )
            print(f"Failed {name}; persisted failure, continuing remaining models", flush=True)
            continue
        target = output / name
        target.mkdir()
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for entry in archive.infolist():
                if not (target / entry.filename).resolve().is_relative_to(target.resolve()):
                    raise ValueError("archive member escapes output")
            archive.extractall(target)
        receipt = json.loads((target / "receipt.json").read_text())
        metrics = {
            key: receipt["public"][key]
            for key in ("n_planned", "n_valid", "n_correct", "accuracy", "ece", "latency")
        }
        print(
            json.dumps(
                {
                    "model": name,
                    "public_metrics": metrics,
                    "strict_coherence": receipt["coherence"]["overall"]
                    if receipt["coherence"]
                    else None,
                    "evidence": str(target),
                    "official_rank": None,
                }
            ),
            flush=True,
        )
    if failures:
        raise RuntimeError(f"Incomplete campaign: {', '.join(failures)}; see preserved evidence")
