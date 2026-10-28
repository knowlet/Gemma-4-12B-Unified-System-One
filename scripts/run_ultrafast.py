"""Run the pinned Jev Ultrafast policy with S1 and independently verified DOM outcomes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from s1.backends import GemmaBackend, UniformBackend
from s1.contracts import DecisionRequest
from s1.evaluation.ultrafast import UltrafastDecisionBridge

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_checkpoint(path):
    """Verify local published files, including weights; a directory name is not an identity."""
    path = Path(path).resolve()
    manifest = json.loads((path / "artifact-manifest.json").read_text())
    required = {
        "model.safetensors",
        "config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "s1_config.json",
    }
    members = manifest.get("files", [])
    names = [row["path"] for row in members]
    if not required <= set(names) or len(set(names)) != len(names):
        raise ValueError("checkpoint manifest needs unique required model/config members")
    for row in manifest["files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != row["path"]:
            raise ValueError("checkpoint member path must be a safe relative path")
        target = (path / row["path"]).resolve()
        if not target.is_relative_to(path) or not target.is_file():
            raise ValueError("checkpoint member missing or outside directory")
        if target.stat().st_size != row["size_bytes"] or digest(target) != row["sha256"]:
            raise ValueError(f"checkpoint member differs: {row['path']}")
    return {
        "path": str(path),
        "manifest_sha256": digest(path / "artifact-manifest.json"),
        "verified_files": manifest["files"],
    }


def summarize(rows):
    result = {}
    for profile in dict.fromkeys(row["profile"] for row in rows):
        group = [row for row in rows if row["profile"] == profile]
        latencies = [row["wall_ms"] for row in group]
        result[profile] = {
            "attempted": len(group),
            "unique_tasks": len({row["task_id"] for row in group}),
            "verified_successes": sum(row["verified_success"] for row in group),
            "task_success_rate": sum(row["verified_success"] for row in group) / len(group),
            "task_latency_ms_p50_p95": np.percentile(latencies, [50, 95]).tolist(),
            "model_calls": sum(row["backend_calls"] + row.get("text_calls", 0) for row in group),
            "decision_calls": sum(row["backend_calls"] for row in group),
            "text_calls": sum(row.get("text_calls", 0) for row in group),
            "backbone_forwards": sum(row.get("backbone_forwards", 0) for row in group),
            "failure_classes": dict(
                Counter(row["failure_class"] for row in group if not row["verified_success"])
            ),
            "reported_cost_usd": None,
            "cost_scope": "Local compute cost not measured; unknown is not free.",
            "by_split": {
                split: {
                    "attempted": len(subset),
                    "verified_successes": sum(row["verified_success"] for row in subset),
                }
                for split in dict.fromkeys(row["split"] for row in group)
                if (subset := [row for row in group if row["split"] == split])
            },
        }
    return result


def native_field_text(model, instruction, records):
    """Optional shared text helper: original upstream instruction, local generation, no gold values."""

    def generate(context):
        import torch

        started = time.perf_counter()
        record = {
            "context": copy.deepcopy(context),
            "instruction": instruction,
            "model": model.name,
            "max_new_tokens": 128,
            "do_sample": False,
            "raw_text": None,
            "error": None,
            "reported_cost_usd": None,
        }
        try:
            text = model.processor.apply_chat_template(
                [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = model.processor(text=text, return_tensors="pt", add_special_tokens=False)
            if inputs["input_ids"].shape[1] > model.max_context:
                raise ValueError("text helper context exceeded; nothing truncated")
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            with torch.inference_mode():
                output = model.lm.generate(**inputs, max_new_tokens=128, do_sample=False)
            new_tokens = output[0, inputs["input_ids"].shape[1] :]
            raw = model.tok.decode(new_tokens, skip_special_tokens=True)
            record["raw_text"] = raw
            record["input_tokens"] = inputs["input_ids"].shape[1]
            record["output_tokens"] = len(new_tokens)
            parsed = json.loads(raw)
            value = parsed["text"]
            if (
                set(parsed) != {"text"}
                or not isinstance(value, str)
                or not value.strip()
                or len(value) > 2000
            ):
                raise ValueError("Text helper returned no valid field value; nothing typed")
            return value, {
                "model": model.name,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "usage": {
                    "input_tokens": record["input_tokens"],
                    "output_tokens": record["output_tokens"],
                },
            }
        except Exception as exc:
            record["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            record["latency_ms"] = (time.perf_counter() - started) * 1000
            records.append(record)

    return generate


def run_benchmarks(backend, profile, output, args, checkpoint):
    from run_jevbench_public import run_public

    identity = {
        "model_id": backend.name,
        "revision": checkpoint["manifest_sha256"],
        "revision_kind": "verified_local_artifact_manifest_sha256",
        "runtime": backend.metadata,
        "platform": platform.platform(),
        "profile": profile,
    }

    def predict(state, questions):
        import torch

        if backend.model.device == "mps":
            torch.mps.synchronize()
        response = backend.predict(
            DecisionRequest.model_validate({"state": state, "questions": questions})
        )
        if backend.model.device == "mps":
            torch.mps.synchronize()
        return response

    if args.heaven_root:
        print(f"Starting public231: {profile}", flush=True)
        run_public(predict, args.heaven_root, output / profile / "public231", identity)
    if args.coherence_root:
        actual = subprocess.check_output(
            ["git", "-C", str(args.coherence_root), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != "e18733694623aa93058e279c3534f8b8e2edefa3":
            raise ValueError("coherence source pin differs")
        if subprocess.check_output(
            [
                "git",
                "-C",
                str(args.coherence_root),
                "status",
                "--porcelain",
                "--untracked-files=no",
            ],
            text=True,
        ):
            raise ValueError("coherence source has tracked modifications")
        sys.path.insert(0, str(args.coherence_root.resolve() / "src"))
        import jevbench as jb

        print(f"Starting coherence-mini: {profile}", flush=True)
        report = jb.evaluate(
            jb.from_callable(
                lambda state, questions: predict(state, questions)["answers"], name=profile
            ),
            suite="jevbench-mini",
            cache=str(output / profile / "coherence-cache"),
            concurrency=1,
            progress=True,
        )
        report.save(output / profile / "coherence-mini.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "configs/benchmarks/ultrafast-local.json"
    )
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/release/checkpoint")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--profiles", default="causal_multislot,independent")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-decisions", type=int, default=12)
    parser.add_argument("--task-timeout", type=float, default=90)
    parser.add_argument(
        "--tasks", help="Comma-separated task IDs; omitted runs full frozen manifest"
    )
    parser.add_argument("--native-text-helper", action="store_true")
    parser.add_argument("--heaven-root", type=Path)
    parser.add_argument("--coherence-root", type=Path)
    args = parser.parse_args()
    profiles = args.profiles.split(",")
    if len(set(profiles)) != len(profiles) or set(profiles) - {
        "uniform",
        "causal_multislot",
        "independent",
    }:
        parser.error("profiles must be unique uniform/causal_multislot/independent")
    if args.repeats < 1 or args.max_decisions < 1 or args.task_timeout <= 0:
        parser.error("budgets must be positive")
    from ultrafast_browser import IsolatedCDP, fixture_server, load_upstream, run_task

    modules, upstream_identity = load_upstream(args.upstream)
    manifest = json.loads(args.manifest.read_text())
    tasks = manifest["tasks"]
    if args.tasks:
        selected = args.tasks.split(",")
        if len(set(selected)) != len(selected) or set(selected) - {task["id"] for task in tasks}:
            parser.error("tasks must be unique manifest IDs")
        tasks = [task for task in tasks if task["id"] in selected]
    args.output.mkdir(parents=True, exist_ok=False)
    sources = args.output / "sources"
    for path in [
        *ROOT.glob("src/s1/**/*.py"),
        *[ROOT / "scripts" / name for name in ["run_ultrafast.py", "ultrafast_browser.py"]],
        args.manifest,
    ]:
        target = sources / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    identity = {
        "status": "running",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            str(path.relative_to(sources)): digest(path)
            for path in sources.rglob("*")
            if path.is_file()
        },
        "upstream": upstream_identity,
        "manifest_sha256": digest(args.manifest),
        "selected_tasks": [task["id"] for task in tasks],
        "profiles": profiles,
        "repeats": args.repeats,
        "max_decisions": args.max_decisions,
        "task_timeout": args.task_timeout,
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["torch", "transformers", "playwright", "browser-harness", "numpy"]
        },
        "exposure": {key: manifest[key] for key in ("scope", "split_policy")},
        "official_rank": None,
        "native_text_helper": args.native_text_helper,
        "reported_cost_usd": None,
    }
    write_json(args.output / "identity.json", identity)
    rows, backends, forward_records = [], {}, []
    try:
        if set(profiles) - {"uniform"}:
            print("Verifying checkpoint files", flush=True)
            identity["checkpoint"] = verify_checkpoint(args.model)
            print("Loading real checkpoint", flush=True)
            started = time.perf_counter()
            shared = GemmaBackend(
                str(args.model.resolve()),
                device=args.device,
                dtype="bfloat16",
                attn_implementation="sdpa",
                question_mode="causal_multislot",
            )
            identity["load_seconds"] = time.perf_counter() - started
            identity["runtime"] = shared.metadata
            import torch

            identity["hardware"] = {
                "processor": platform.processor(),
                "machine": platform.machine(),
                "torch_num_threads": torch.get_num_threads(),
                "mps_available": torch.backends.mps.is_available(),
            }
            if sys.platform == "darwin":
                identity["hardware"]["chip"] = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
                ).strip()
                identity["hardware"]["memory_bytes"] = int(
                    subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
                )

            def record_forward(_module, _args, kwargs, _output):
                ids = kwargs.get("input_ids")
                forward_records.append(
                    {"input_shape": list(ids.shape) if ids is not None else None}
                )

            shared.model.backbone.register_forward_hook(record_forward, with_kwargs=True)
            for profile in set(profiles) - {"uniform"}:
                backend = copy.copy(shared)
                backend.question_mode = profile
                backend.metadata = {**shared.metadata, "question_mode": profile}
                backends[profile] = backend
        backends["uniform"] = UniformBackend()
        write_json(args.output / "identity.json", identity)
        key_before = os.environ.get("TYPESAFE_API_KEY")
        os.environ["TYPESAFE_API_KEY"] = "local-transport-no-credential"
        original_text_helper = modules["agent"].field_text
        try:
            with (
                fixture_server(args.upstream) as base_url,
                IsolatedCDP(modules["browser"], args.output) as transport,
            ):
                identity["chromium_version"] = transport.browser_version
                write_json(args.output / "identity.json", identity)
                for repeat in range(args.repeats):
                    for task_index, task in enumerate(tasks):
                        # Alternate first profile to reduce systematic warm/order effects.
                        order = (
                            profiles if (repeat + task_index) % 2 == 0 else list(reversed(profiles))
                        )
                        for profile in order:
                            destination = args.output / profile / f"{task['id']}-{repeat + 1}"
                            bridge = UltrafastDecisionBridge(backends[profile])
                            modules["model"].post_json = bridge
                            text_records = []
                            modules["agent"].field_text = original_text_helper
                            if args.native_text_helper and profile != "uniform":
                                modules["agent"].field_text = native_field_text(
                                    backends[profile].model,
                                    modules["model"].TEXT_VALUE,
                                    text_records,
                                )
                            transport.set_output(destination)
                            forward_start = len(forward_records)
                            row = run_task(
                                modules["agent"],
                                task,
                                f"{base_url}/fixture.html?scenario={task['scenario']}",
                                destination,
                                max_decisions=args.max_decisions,
                                task_timeout=args.task_timeout,
                            )
                            transport.close_task()
                            row.update(
                                profile=profile,
                                repeat=repeat + 1,
                                backend_calls=bridge.backend_calls,
                                backbone_forwards=len(forward_records) - forward_start,
                                text_calls=len(text_records),
                                reported_cost_usd=None,
                            )
                            write_json(destination / "decision-transport.json", bridge.records)
                            write_json(destination / "text-transport.json", text_records)
                            write_json(
                                destination / "backbone-forwards.json",
                                forward_records[forward_start:],
                            )
                            write_json(destination / "summary.json", row)
                            rows.append(row)
                            write_json(args.output / "results.json", rows)
                            write_json(args.output / "summary.json", summarize(rows))
                            print(json.dumps(row), flush=True)
        finally:
            if key_before is None:
                os.environ.pop("TYPESAFE_API_KEY", None)
            else:
                os.environ["TYPESAFE_API_KEY"] = key_before
        for profile in profiles:
            if profile != "uniform":
                run_benchmarks(
                    backends[profile], profile, args.output, args, identity["checkpoint"]
                )
        identity["status"] = "completed"
    except Exception as exc:
        identity.update(status="error", error={"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        write_json(args.output / "identity.json", identity)
    return 0 if rows and all(row["verified_success"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
