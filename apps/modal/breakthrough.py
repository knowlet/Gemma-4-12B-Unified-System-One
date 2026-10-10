"""Prospectively fixed prompt/weight ablations and bounded mixed-target research.

uv run --no-sync modal run apps/modal/breakthrough.py --run <unique-id> --stage ablate
uv run --no-sync modal run apps/modal/breakthrough.py --run <unique-id> --stage train
"""

from __future__ import annotations

import io
import json
import os
import re
import zipfile
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/workspace")
DATA = ROOT / "artifacts/jevbench/breakthrough-20261008/data"
app = modal.App("gemma-s1-breakthrough")
volume = modal.Volume.from_name("gemma-unified-system-one", create_if_missing=False)
WEIGHTS = {
    "released": (
        "knowlet/Gemma-4-12B-Unified-System-One",
        "a66f836b56605039fe040f330180e336d19b3362",
    ),
    "base": ("google/gemma-4-12B-it", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"),
}
HEAVEN = "bb05a335bc809e61b20c0f745d25499a82b326fc"
COHERENCE = "e18733694623aa93058e279c3534f8b8e2edefa3"
runtime = (
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
        "peft==0.21.0",
    )
    .run_commands(
        "git clone https://github.com/fstandhartinger/jevbench /opt/heaven",
        f"git -C /opt/heaven checkout {HEAVEN}",
        "git clone https://github.com/JevBench/jevbench /opt/coherence",
        f"git -C /opt/coherence checkout {COHERENCE}",
    )
    .env({"HF_HOME": "/vol/hf", "TOKENIZERS_PARALLELISM": "false", "USE_TF": "0"})
    .add_local_python_source("s1")
    .add_local_dir(ROOT / "scripts", "/workspace/scripts")
    .add_local_dir(ROOT / "configs", "/workspace/configs")
    .add_local_dir(DATA, "/workspace/data")
    .add_local_dir(ROOT / "artifacts/release/datasets", "/workspace/release-data")
    .add_local_file(ROOT / "examples/benchmarks/mps.jsonl", "/workspace/native-regression.jsonl")
)
secrets = (
    [modal.Secret.from_name(os.environ["S1_HF_SECRET"])] if os.environ.get("S1_HF_SECRET") else []
)


def _path(run, stage):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", run) or stage not in ("ablate", "train"):
        raise ValueError("unique lowercase run and ablate/train stage required")
    return Path("/vol/breakthrough") / run / stage


def _run(run, stage):
    import gc
    import hashlib
    import importlib.metadata
    import sys

    import torch
    from huggingface_hub import snapshot_download

    import s1

    sys.path.insert(0, "/workspace/scripts")
    from breakthrough_training import (
        capture,
        fit_head,
        fit_temperatures,
        records,
        summarize,
        train_lora,
        write_json,
    )
    from prepare_mlx_validation import checkpoint_identity
    from release_training import load_release_data
    from run_jevbench_public import run_public

    from s1.contracts import DecisionRequest, answer_from_probabilities
    from s1.evaluation.datasets import audit_splits, load_cases
    from s1.unified import UnifiedDecisionModel

    destination = _path(run, stage)
    destination.mkdir(parents=True, exist_ok=False)
    source_files = {
        "campaign": Path(__file__),
        **{p.name: p for p in Path("/workspace/scripts").glob("breakthrough*.py")},
        "data_generator": Path("/workspace/scripts/prepare_breakthrough_data.py"),
        "public_runner": Path("/workspace/scripts/run_jevbench_public.py"),
        "release_data_loader": Path("/workspace/scripts/release_training.py"),
        "checkpoint_identity_helper": Path("/workspace/scripts/prepare_mlx_validation.py"),
        "prospective_protocol": Path(
            "/workspace/configs/experiments/jevbench-breakthrough-20261008.json"
        ),
        **{
            f"s1/{p.relative_to(Path(s1.__file__).parent)}": p
            for p in Path(s1.__file__).parent.rglob("*.py")
        },
    }
    source_hashes = {
        key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in source_files.items()
    }
    if source_hashes["prospective_protocol"] != (
        "3a8dcceba9fbc0c7ce31bddbb85c0d10acdfa855ddf237f4d5e81982ec6a926d"
    ):
        raise ValueError("frozen prospective protocol checksum mismatch")
    protocol = json.loads(source_files["prospective_protocol"].read_text())
    (destination / "sources").mkdir()
    for key, path in source_files.items():
        saved = destination / "sources" / f"{key}.txt"
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(path.read_bytes())
    paths = [
        Path("/workspace/data") / f"{split}.jsonl" for split in ("train", "calibration", "test")
    ]
    split_audit = audit_splits(paths)
    if split_audit["overlaps"]:
        raise ValueError("synthetic splits overlap")
    train, calibration, test = [load_cases(path) for path in paths]
    if [len(train), len(calibration), len(test)] != [1024, 192, 384]:
        raise ValueError("prospective dataset counts changed")
    release, release_identity = load_release_data("/workspace/release-data")
    data_manifest_path = Path("/workspace/data/manifest.json")
    for path, expected in (
        (data_manifest_path, protocol["data"]["synthetic"]["manifest_sha256"]),
        (
            Path("/workspace/release-data/manifest.json"),
            protocol["data"]["release_regression"]["manifest_sha256"],
        ),
        (
            Path("/workspace/native-regression.jsonl"),
            protocol["data"]["native_fixture"]["file_sha256"],
        ),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("frozen prospective input checksum mismatch")
    data_manifest = json.loads(data_manifest_path.read_text())
    for filename, expected in protocol["data"]["synthetic"]["files"].items():
        if (
            Path(filename).name != filename
            or hashlib.sha256((Path("/workspace/data") / filename).read_bytes()).hexdigest()
            != expected
        ):
            raise ValueError("frozen synthetic dataset checksum mismatch")
    for filename, dataset in data_manifest["datasets"].items():
        if (
            Path(filename).name != filename
            or hashlib.sha256((Path("/workspace/data") / filename).read_bytes()).hexdigest()
            != dataset["file_sha256"]
        ):
            raise ValueError("frozen synthetic dataset checksum mismatch")
    receipt = {
        "status": "running",
        "stage": stage,
        "run": run,
        "modal_app_id": app.app_id,
        "modal_call_id": modal.current_function_call_id(),
        "gpu": torch.cuda.get_device_name(),
        "precision": "bfloat16",
        "source_files": source_hashes,
        "prospective_protocol": protocol,
        "versions": {
            p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "numpy")
        },
        "dataset": data_manifest,
        "split_audit": split_audit,
        "release_data": release_identity,
        "training_recipe": {
            "steps": 128,
            "seed": 42,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_gradient_accumulation": 8,
            "lora_lr": 1e-5,
            "head_lr": 1e-4,
            "head_batch": 64,
            "loss": "soft CE + 0.1 Brier; head only adds 0.01 weight anchor",
            "temperature": "per type; calibration-only soft CE; fixed geomspace .1..10 81values",
        },
        "public_exposure": "All public231/coherence development data exposed before this trial; none used for fitting or recipe selection. Synthetic heldout template family fixed before GPU trial; not a general-world benchmark.",
        "execution": "Sequential independent native forward per question; no compilation, CUDA graph, candidate cache or probability constraints.",
        "reported_cost_usd": None,
        "official_rank": None,
        "profiles": [],
    }
    write_json(destination / "receipt.json", receipt)
    volume.commit()

    class Predictor:
        def __init__(self, model, prompt, temperatures, head=None):
            self.model, self.prompt, self.temperatures, self.head = (
                model,
                prompt,
                temperatures,
                head,
            )
            self.raw, self.replay = {}, False

        def __call__(self, state, questions):
            request = DecisionRequest(state=state, questions=questions)
            key = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
            if self.replay:
                rows = self.raw[key]
            else:
                self.model.lm.eval()
                torch.cuda.synchronize()
                with torch.inference_mode():
                    rows = [
                        {
                            "id": r["question"].id,
                            "labels": r["question"].labels(),
                            "logits": r["logits"].float().cpu().tolist(),
                            "sequence_tokens": r["sequence_tokens"],
                        }
                        for r in capture(self.model, request, self.prompt, self.head)
                    ]
                torch.cuda.synchronize()
                self.raw[key] = rows
            answers = {}
            for question, row in zip(request.questions, rows, strict=True):
                if row["id"] != question.id or row["labels"] != question.labels():
                    raise ValueError("raw-logit replay changed label identity")
                values = (
                    (
                        torch.tensor(row["logits"], dtype=torch.float32)
                        / self.temperatures[question.type]
                    )
                    .softmax(-1)
                    .tolist()
                )
                answers[question.id] = answer_from_probabilities(question, values)
            return {
                "answers": answers,
                "runtime": {
                    "raw_logits": rows,
                    "prompt": self.prompt,
                    "sequential_independent_questions": True,
                    "forward_calls": 0 if self.replay else len(rows),
                    "calibration_replay": self.replay,
                },
            }

    def evaluate(model, name, prompt, identity, head=None):
        torch.cuda.reset_peak_memory_stats()
        root = destination / name
        root.mkdir(exist_ok=False)
        cal = records(model, calibration, prompt, head)
        fitted = fit_temperatures(cal)
        temps = {kind: value["temperature"] for kind, value in fitted.items()}
        heldout = records(model, test, prompt, head)
        write_json(root / "calibration.json", {"temperatures": fitted, "records": cal})
        write_json(
            root / "heldout.json",
            {
                "records": heldout,
                "raw_metrics": summarize(heldout, {k: 1.0 for k in temps}),
                "calibrated_metrics": summarize(heldout, temps),
            },
        )
        predictor = Predictor(model, prompt, temps, head)
        ident = {
            **identity,
            "variant": name,
            "gpu": receipt["gpu"],
            "precision": "bfloat16",
            "benchmarkheaven_commit": HEAVEN,
            "coherence_commit": COHERENCE,
            "source_files": source_hashes,
            "calibration": fitted,
            "execution": receipt["execution"],
            "reported_cost_usd": None,
            "official_rank": None,
        }
        public = run_public(predictor, "/opt/heaven", root / "public231", ident)
        regression = {}
        for dataset in ("test", "media", "regression_text"):
            saved, n_correct, n_questions = [], 0, 0
            for case in release[dataset]:
                response = (
                    predictor(case.request.state, case.request.named_questions_payload())
                    if not case.request.media
                    else None
                )
                if response is None:
                    with torch.inference_mode():
                        values = capture(model, case.request, prompt, head)
                    response = {
                        "answers": {
                            r["question"].id: answer_from_probabilities(
                                r["question"],
                                (r["logits"].float() / temps[r["question"].type])
                                .softmax(-1)
                                .cpu()
                                .tolist(),
                            )
                            for r in values
                        }
                    }
                for q in case.request.questions:
                    n_questions += 1
                    probs = response["answers"][q.id]["probabilities"]
                    n_correct += int(max(probs, key=probs.__getitem__) == case.gold[q.id])
                saved.append({"case_id": case.id, "gold": case.gold, "response": response})
            regression[dataset] = {
                "n_cases": len(saved),
                "n_questions": n_questions,
                "n_correct": n_correct,
                "accuracy": n_correct / n_questions,
                "records": saved,
            }
            print(f"{name} regression {dataset}: {n_correct}/{n_questions}", flush=True)
        write_json(root / "regression.json", regression)
        native = []
        for case in load_cases("/workspace/native-regression.jsonl"):
            with torch.inference_mode():
                rows = capture(model, case.request, prompt, head)
            native.append(
                {
                    "case_id": case.id,
                    "modalities": [m.type for m in case.request.media],
                    "answers": {
                        r["question"].id: answer_from_probabilities(
                            r["question"],
                            (r["logits"].float() / temps[r["question"].type])
                            .softmax(-1)
                            .cpu()
                            .tolist(),
                        )
                        for r in rows
                    },
                }
            )
        write_json(root / "native-fixture.json", native)
        sys.path.insert(0, "/opt/coherence/src")
        import jevbench as jb

        coherence = jb.evaluate(
            jb.from_callable(
                lambda state, questions: predictor(state, questions)["answers"],
                name=f"{name}@{identity['revision']}",
            ),
            suite="jevbench-mini",
            cache=str(root / "coherence-cache"),
            concurrency=1,
            progress=True,
        )
        # Library returns its report model, retaining the full official suite outcomes.
        coherence.save(str(root / "coherence-mini.json"))
        predictor.replay, predictor.temperatures = True, {k: 1.0 for k in temps}
        coherence_unit = jb.evaluate(
            jb.from_callable(
                lambda state, questions: predictor(state, questions)["answers"],
                name=f"{name}-unit-replay@{identity['revision']}",
            ),
            suite="jevbench-mini",
            cache=str(root / "coherence-cache-unit"),
            concurrency=1,
            progress=False,
        )
        coherence_unit.save(root / "coherence-unit.json")
        write_json(root / "raw-logits.json", predictor.raw)
        volume.commit()
        raw_public = {}
        for policy, temperatures in (
            ("unit", {k: 1.0 for k in temps}),
            ("published_global", {k: 2.7830344470383452 for k in temps}),
        ):
            predictor.replay, predictor.temperatures = True, temperatures
            raw_public[policy] = run_public(
                predictor,
                "/opt/heaven",
                root / f"public231-{policy}",
                {
                    **ident,
                    "variant": f"{name}-{policy}",
                    "calibration": temperatures,
                    "execution": "raw-logit calibration replay; recorded durations are CPU replay only, NOT model inference latency",
                },
            )
        entry = {
            "name": name,
            "prompt": prompt,
            "identity": identity,
            "public": public,
            "raw_public": raw_public,
            "heldout": summarize(heldout, temps),
            "coherence": coherence.overall,
            "unit_coherence": coherence_unit.overall,
            "regression": {
                key: {k: v for k, v in value.items() if k != "records"}
                for key, value in regression.items()
            },
            "calibration": fitted,
        }
        receipt["profiles"].append(entry)
        write_json(
            root / "receipt.json",
            {
                "identity": ident,
                "public": public,
                "coherence": {
                    "overall": coherence.overall,
                    "interval": coherence.interval,
                    "dimensions": coherence.dimensions,
                    "errors": coherence.errors,
                    "scores": coherence.scores,
                },
                "memory_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "memory_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "memory_peak_scope": "Whole profile including independent synthetic calibration/test, public231, release native regression and coherence; not minimum deployment VRAM",
                "official_rank": None,
            },
        )
        write_json(destination / "receipt.json", receipt)
        volume.commit()
        print(
            json.dumps(
                {
                    "profile": name,
                    "public_correct": public["n_correct"],
                    "coherence": entry["coherence"],
                    "unit_coherence": entry["unit_coherence"],
                    "heldout": entry["heldout"],
                }
            ),
            flush=True,
        )

    def load(kind, *, lora=False):
        model_id, revision = WEIGHTS[kind]
        cached = (
            Path("/vol/hf/hub") / f"models--{model_id.replace('/', '--')}" / "snapshots" / revision
        )
        directory = (
            cached
            if (cached / "config.json").is_file()
            else Path(
                snapshot_download(
                    model_id,
                    revision=revision,
                    allow_patterns=["*.json", "*.safetensors", "*.model", "*.jinja", "*.txt"],
                )
            )
        )
        checkpoint = checkpoint_identity(directory)
        processor_files = {
            p.name: {
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "size_bytes": p.stat().st_size,
            }
            for p in sorted(directory.iterdir())
            if p.is_file() and p.suffix in (".json", ".model", ".jinja", ".txt")
        }
        options = {
            "device": "cuda",
            "precision": "bfloat16",
            "attn_implementation": "sdpa",
            "temperature": 1.0,
        }
        if lora:
            options["lora"] = {
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "bias": "none",
            }
        torch.manual_seed(42)
        loaded = UnifiedDecisionModel(str(directory), **options)
        if lora:
            for parameter in loaded.lm.parameters():
                if parameter.requires_grad:
                    parameter.data = parameter.data.float()
        return loaded, {
            "model_id": model_id,
            "revision": revision,
            "checkpoint_files": checkpoint,
            "processor_files": processor_files,
        }

    try:
        if stage == "ablate":
            for kind in ("released", "base"):
                model, identity = load(kind)
                for prompt in ("current", "user_question"):
                    evaluate(model, f"{kind}-{prompt}", prompt, identity)
                del model
                gc.collect()
                torch.cuda.empty_cache()
        else:
            model, identity = load("released")
            evaluate(model, "released-user-control", "user_question", identity)
            initial_bias = getattr(model.head, "bias", None)
            initial_head = {
                "weight": model.head.weight.detach().index_select(0, model.letters).float().clone(),
                "bias": initial_bias.detach().index_select(0, model.letters).float().clone()
                if initial_bias is not None
                else torch.zeros(len(model.letters), device=model.device),
            }
            evaluate(
                model,
                "released-user-head-init",
                "user_question",
                {
                    **identity,
                    "head_precision": "float32",
                    "head_training": "not_run; kernel/readout control",
                },
                initial_head,
            )
            del initial_head
            features = records(model, train, "user_question", features=True)
            trained_head, training = fit_head(model, features)
            torch.save(
                {
                    "features": torch.stack([row.pop("hidden") for row in features]),
                    "records": features,
                },
                destination / "train-features.pt",
            )
            torch.save(trained_head, destination / "decision-head.pt")
            write_json(destination / "head-training.json", training)
            head_hash = hashlib.sha256((destination / "decision-head.pt").read_bytes()).hexdigest()
            head = {key: value.to(model.device) for key, value in trained_head.items()}
            evaluate(
                model,
                "released-user-head",
                "user_question",
                {**identity, "research_head_sha256": head_hash, "head_precision": "float32"},
                head,
            )
            del model, head, trained_head, features
            gc.collect()
            torch.cuda.empty_cache()
            model, identity = load("released", lora=True)
            train_lora(
                model, train, destination / "mixed-lora", accumulation=8, commit=volume.commit
            )
            evaluate(
                model,
                "released-user-lora",
                "user_question",
                {
                    **identity,
                    "research_adapter_files": json.loads(
                        (destination / "mixed-lora/training.json").read_text()
                    )["adapter_files"],
                },
            )
        receipt["status"] = "completed"
    except Exception as exc:
        receipt["status"], receipt["error"] = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_json(destination / "receipt.json", receipt)
        volume.commit()
    return receipt


@app.function(
    image=runtime,
    gpu="A100-80GB",
    cpu=4,
    memory=98304,
    volumes={"/vol": volume},
    secrets=secrets,
    timeout=14400,
    max_containers=1,
    single_use_containers=True,
)
def experiment(run: str, stage: str):
    try:
        return _run(run, stage)
    except Exception as exc:
        root = _path(run, stage)
        root.mkdir(parents=True, exist_ok=True)
        (root / "failure.json").write_text(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "official_rank": None,
                }
            )
            + "\n"
        )
        raise
    finally:
        volume.commit()


@app.function(
    image=modal.Image.debian_slim(python_version="3.12"),
    volumes={"/vol": volume},
    timeout=600,
    memory=4096,
    cpu=2,
)
def export(run: str, stage: str):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(_path(run, stage).rglob("*")):
            if path.is_file() and (
                path.suffix in (".json", ".jsonl", ".txt", ".safetensors")
                or path.name in ("decision-head.pt", "train-features.pt")
            ):
                archive.write(path, path.relative_to(_path(run, stage)))
    return buffer.getvalue()


def _download_evidence(run: str, stage: str, destination: Path):
    """Use bounded, source-verified local Volume reads after a terminal GPU call."""
    import asyncio
    import importlib.util

    helper = Path(__file__).resolve().parents[2] / "scripts/export_breakthrough_evidence.py"
    spec = importlib.util.spec_from_file_location("breakthrough_evidence_download", helper)
    downloader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(downloader)
    return asyncio.run(
        downloader.download(volume, run, stage, destination, allow_empty_output=True)
    )


@app.local_entrypoint()
def main(run: str = "", stage: str = "ablate"):
    _path(run, stage)
    destination = ROOT / "artifacts/jevbench/breakthrough-20261008/runs" / run / stage
    destination.mkdir(parents=True, exist_ok=False)
    error = None
    try:
        result = experiment.remote(run, stage)
        print(
            json.dumps(
                {"status": result["status"], "profiles": [p["name"] for p in result["profiles"]]}
            ),
            flush=True,
        )
    except Exception as exc:
        error = exc
    print(f"Downloading evidence to {destination}", flush=True)
    try:
        verification = _download_evidence(run, stage, destination)
    except Exception as download_error:
        if error is not None:
            raise error from download_error
        raise
    print(json.dumps(verification, sort_keys=True), flush=True)
    print(f"Saved {destination}", flush=True)
    if error is not None:
        raise error
