"""One saved native field-helper call on the existing local Gemma instruction model.

This sequential MPS control does not score tasks, adopt a helper, download files,
or infer that a cache directory proves an absence of fine-tuning. Source and
metadata bytes are frozen before waiting; every base file is hashed before load.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BASE_REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
BASE_MODEL = (
    Path("/Users/knowlet/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots")
    / BASE_REVISION
)
REQUIRED_FILES = {
    "model.safetensors",
    "config.json",
    "generation_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_hash(value):
    raw = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def load_module(path, name):
    module = SimpleNamespace(__file__=str(path), __name__=name)
    exec(compile(Path(path).read_bytes(), str(path), "exec"), module.__dict__)
    return module


def inventory(root):
    return {
        path.relative_to(root).as_posix(): digest(path)
        for path in sorted(Path(root).rglob("*"))
        if path.is_file()
    }


def select_helper(path, index):
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list) or not 1 <= index <= len(rows):
        raise ValueError("helper record must select an existing one-based entry")
    row = rows[index - 1]
    if (
        not isinstance(row.get("instruction"), str)
        or not row["instruction"].strip()
        or not isinstance(row.get("context"), dict)
        or row.get("do_sample") is not False
        or type(row.get("max_new_tokens")) is not int
        or row["max_new_tokens"] != 128
    ):
        raise ValueError(
            "helper needs its original instruction/context and 128-token greedy settings"
        )
    helper = {key: row[key] for key in ("instruction", "context", "do_sample", "max_new_tokens")}
    json_hash(helper)  # Reject nonfinite/non-JSON input rather than silently changing it.
    return helper


def stat_identity(path):
    stat = Path(path).stat()
    return {
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
    }


def freeze(args):
    """Copy source/metadata only; do not read or copy the weight payload yet."""
    frozen = args.output / "sources"
    for source in sorted((ROOT / "src/s1").rglob("*.py")):
        target = frozen / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for name in (
        "diagnose_mps_base_helper.py",
        "diagnose_mps_numerics.py",
        "resume_mps_benchmarks.py",
    ):
        target = frozen / "scripts" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "scripts" / name, target)
    trace = frozen / "inputs/helper-trace.json"
    trace.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.helper_trace, trace)
    helper = select_helper(trace, args.helper_record)
    write_json(frozen / "inputs/helper-record.json", helper)

    names = {path.name for path in args.base_model.iterdir() if path.is_file()}
    if not REQUIRED_FILES <= names or "s1_config.json" in names:
        raise ValueError(
            "base control requires complete native files and no S1 calibration sidecar"
        )
    if any(path.is_dir() for path in args.base_model.iterdir()):
        raise ValueError("base snapshot must contain only top-level model files")
    config = json.loads((args.base_model / "config.json").read_text())
    if (
        config.get("model_type") != "gemma4_unified"
        or "Gemma4UnifiedForConditionalGeneration" not in config.get("architectures", [])
        or config.get("quantization")
        or config.get("quantization_config")
    ):
        raise ValueError("base control requires the unquantized complete Unified architecture")
    files = {}
    for name in sorted(names):
        source = args.base_model / name
        row = {"resolved_path": str(source.resolve()), "stat": stat_identity(source)}
        if name.endswith((".safetensors", ".bin", ".gguf")):
            if name != "model.safetensors":
                raise ValueError("this bounded control requires one native model.safetensors")
            row["kind"] = "weight_not_copied"
        else:
            if source.stat().st_size > 64 * 1024 * 1024:
                raise ValueError("unexpected oversized metadata file")
            target = frozen / "base-metadata" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            row.update(kind="frozen_metadata", sha256=digest(target))
        files[name] = row
    return {
        "source_inventory": inventory(frozen),
        "helper_sha256": json_hash(helper),
        "base_files": files,
        "declared_base": {
            "repository": "google/gemma-4-12B-it"
            if args.base_model == BASE_MODEL.resolve()
            else None,
            "revision": BASE_REVISION if args.base_model == BASE_MODEL.resolve() else None,
            "path": str(args.base_model),
            "identity_scope": "Local file inventory; cache naming is not independent publisher-byte attestation",
            "absence_of_fine_tuning_proven": False,
            "derivative_artifact_manifest_used": False,
        },
    }


def verify_base_files(args, receipt):
    """Hash every actual base file, including weights, after waits and before load."""
    names = {path.name for path in args.base_model.iterdir() if path.is_file()}
    if names != set(receipt["base_files"]):
        raise ValueError("base file inventory changed while queued")
    verified = {}
    for name, captured in receipt["base_files"].items():
        path = args.base_model / name
        if (
            str(path.resolve()) != captured["resolved_path"]
            or stat_identity(path) != captured["stat"]
        ):
            raise ValueError("base file identity changed while queued")
        value = digest(path)
        if stat_identity(path) != captured["stat"]:
            raise ValueError("base file changed during hashing")
        if captured["kind"] == "frozen_metadata" and value != captured["sha256"]:
            raise ValueError("base metadata hash differs from frozen source")
        verified[name] = {"sha256": value, "size_bytes": captured["stat"]["size_bytes"]}
    return verified


def prepare_runtime_model(args, receipt):
    """Load frozen metadata and the verified original weight file without duplicating it."""
    path = args.output / "runtime-model"
    path.mkdir(exist_ok=False)
    for name, row in receipt["base_files"].items():
        if row["kind"] == "weight_not_copied":
            (path / name).symlink_to(row["resolved_path"])
        else:
            shutil.copyfile(args.output / "sources/base-metadata" / name, path / name)
    return path


def load_runtime(output):
    source = (Path(output) / "sources/src").resolve()
    for name, module in tuple(sys.modules.items()):
        if name == "s1" or name.startswith("s1."):
            location = getattr(module, "__file__", None)
            if not location or not Path(location).resolve().is_relative_to(source):
                raise RuntimeError(
                    "S1 imported before the frozen base-control source was installed"
                )
    # Enforce local-only library behavior before importing Torch/Transformers.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.path.insert(0, str(source))
    return SimpleNamespace(
        torch=importlib.import_module("torch"),
        transformers=importlib.import_module("transformers"),
        unified=importlib.import_module("s1.unified"),
        profiling=importlib.import_module("s1.profiling"),
    )


def execute(args, receipt, generator):
    helper = json.loads((args.output / "sources/inputs/helper-record.json").read_text())
    if json_hash(helper) != receipt["helper_sha256"]:
        raise ValueError("frozen helper input changed before model load")
    runtime = load_runtime(args.output)
    torch = runtime.torch
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable; no alternate device or helper is allowed")
    receipt["memory_before_load"] = runtime.profiling.memory_snapshot("mps")
    model_path = prepare_runtime_model(args, receipt)
    started = time.perf_counter()
    processor = runtime.transformers.AutoProcessor.from_pretrained(
        str(model_path.resolve()), local_files_only=True, trust_remote_code=False
    )
    lm = runtime.transformers.AutoModelForMultimodalLM.from_pretrained(
        str(model_path.resolve()),
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    if lm.config.model_type != "gemma4_unified":
        raise ValueError("loaded base is not the complete native Unified model")
    model = runtime.unified.UnifiedDecisionModel.from_components(
        lm, processor, device="mps", temperature=1.0, max_context=16384
    )
    torch.mps.synchronize()
    receipt["load_seconds"] = time.perf_counter() - started
    if model.device != "mps" or model.dtype != "bfloat16" or model.attn_implementation != "sdpa":
        raise RuntimeError("base control requires the requested native MPS BF16 SDPA runtime")
    receipt["runtime"] = {
        "model_path": str(model_path),
        "device": model.device,
        "dtype": model.dtype,
        "attn_implementation": model.attn_implementation,
        "temperature": model.temperature,
        "max_context": model.max_context,
        "compile_mode": getattr(model, "compile_mode", None),
        "quantization": model.quantization,
        "packages": runtime.profiling.environment_metadata(),
    }
    receipt["memory_before_generation"] = runtime.profiling.memory_snapshot("mps")
    receipt["status"] = "running_control"
    write_json(args.output / "receipt.json", receipt)
    started = time.perf_counter()
    receipt["generation_calls"] = 1
    result = generator.generate_native(model, torch, helper)
    # Preserve the exact helper result, including raw JSON/text and original error.
    write_json(args.output / "generation.json", result)
    receipt["generation_sha256"] = digest(args.output / "generation.json")
    receipt["generation_error"] = result.get("error")
    torch.mps.synchronize()
    receipt["generation_wall_ms"] = (time.perf_counter() - started) * 1000
    receipt["memory_after_generation"] = runtime.profiling.memory_snapshot("mps")
    return result.get("error") is None


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema_version": 1,
        "status": "freezing",
        "pid": os.getpid(),
        "scope": "One saved general field-helper input; no task scoring or production adoption",
        "helper_record": args.helper_record,
        "scoring": "not performed",
        "official_rank": None,
        "reported_cost_usd": None,
        "generation_calls": 0,
        "defaults_changed": False,
        "network_policy": "HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 before runtime import",
        "predecessors": [],
    }

    def save():
        write_json(args.output / "receipt.json", receipt)

    save()
    try:
        receipt.update(freeze(args))
        frozen = args.output / "sources"
        guard = load_module(frozen / "scripts/resume_mps_benchmarks.py", "_base_helper_guard")
        receipt["predecessors"] = [guard.capture_handle(pid) for pid in args.wait_for_pid]
        receipt["status"] = "waiting"
        save()
        for handle in receipt["predecessors"]:
            guard.wait_handle(handle, save, 15)
        if inventory(frozen) != receipt["source_inventory"]:
            raise ValueError("frozen source/input hash differs after waiting")
        receipt["verified_base_files"] = verify_base_files(args, receipt)
        if inventory(frozen) != receipt["source_inventory"]:
            raise ValueError("frozen source/input hash differs during base verification")
        generator_path = frozen / "scripts/diagnose_mps_numerics.py"
        receipt["generate_native_source"] = {
            "path": "scripts/diagnose_mps_numerics.py",
            "sha256": digest(generator_path),
            "function": "generate_native",
        }
        generator = load_module(generator_path, "_base_helper_generation")
        save()
        succeeded = execute(args, receipt, generator)
        receipt["status"] = "completed_control"
        save()
        return 0 if succeeded else 1
    except Exception as exc:
        receipt.update(status="error", error={"type": type(exc).__name__, "message": str(exc)})
        save()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helper-trace", type=Path, required=True)
    parser.add_argument("--helper-record", type=int, required=True)
    parser.add_argument("--base-model", type=Path, default=BASE_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-for-pid", type=int, action="append", default=[])
    args = parser.parse_args(argv)
    if (
        args.helper_record < 1
        or any(pid <= 0 for pid in args.wait_for_pid)
        or len(set(args.wait_for_pid)) != len(args.wait_for_pid)
    ):
        parser.error("positive helper index and unique positive predecessor PIDs required")
    for name in ("helper_trace", "base_model", "output"):
        setattr(args, name, getattr(args, name).resolve())
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
