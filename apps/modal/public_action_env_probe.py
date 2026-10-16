"""CPU-only check of the immutable public07 processor environment on Modal.

Review the source and upload inventory before running:
  .venv/bin/modal run apps/modal/public_action_env_probe.py --output /tmp/probe.json

This does not load a processor, model, tensor payload, GPU, secret, or volume.
The image installs genuine wheels; preserved reference files are never copied
over installed packages or their distribution metadata.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
SCRIPT = ROOT / "apps/modal/public_action_env_probe.py"
ARTIFACT = ROOT / "scripts/jev_public_action_artifact.py"
REFERENCE = ROOT / "artifacts/jev-runtime/public-native-action-compiler-07-root-20261016-01/actual-compile-01"
OVERLAY = ROOT / "artifacts/jev-runtime/overlay"
INTERPRETER = "/home/albert_hsieh/ComfyUI/.venv/bin/python"
REVISION = "a66f836b56605039fe040f330180e336d19b3362"
CACHE = ROOT / "artifacts/jev-local-hf/hub/models--knowlet--Gemma-4-12B-Unified-System-One/snapshots" / REVISION
ARTIFACT_SHA = "8c474af9edcc4bc78632713d242c0be9c78aacd6079758d0fcef143459082268"
CLOSURE_SHA = "ff87ceb2563bdee706c4b8cc1064a4682534be2925372c0c02c7ccd2d04e7bbc"
MAX_BYTES = 64 * 1024**2
MAX_LOG_BYTES = 128 * 1024
FORBIDDEN = ("torch", "transformers", "tokenizers", "safetensors", "numpy", "PIL", "jinja2", "peft", "playwright")
CACHE_PINS = {
    "chat_template.jinja": {"sha256": "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4", "size_bytes": 18683},
    "config.json": {"sha256": "b50e760a9107b629e1a140ab7a4392e875c37baefee022f27c98b90b9d947cf0", "size_bytes": 4953},
    "generation_config.json": {"sha256": "040d0ccb9e10df02df9738424c1ce6d20d5d15b09362da922093ccd97573dbb4", "size_bytes": 255},
    "processor_config.json": {"sha256": "6b938e76555b3e9946890770e1abcd442a4718f34041a58e8139dc8ad34545c9", "size_bytes": 1382},
    "tokenizer.json": {"sha256": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f", "size_bytes": 32169626},
    "tokenizer_config.json": {"sha256": "794a39f8330ce05020774c70c091225bc5f031b9cacf41fc20e52eb54b4b52d8", "size_bytes": 3736},
}


def _read(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_BYTES:
        raise ValueError("Missing, symbolic, or oversized input: " + str(path))
    blob = path.read_bytes()
    if len(blob) > MAX_BYTES:
        raise ValueError("Input grew beyond byte bound")
    return blob


def _pin(path):
    blob = _read(path)
    return {"sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)}


def _closure():
    blob = _read(REFERENCE / "processor-source-closure.json")
    if hashlib.sha256(blob).hexdigest() != CLOSURE_SHA:
        raise ValueError("Fixed public07 processor closure changed")
    closure = json.loads(blob)
    if len(closure["files"]) != 51 or len(closure["packages"]) != 9:
        raise ValueError("Fixed 51-file/nine-package closure required")
    return closure


def upload_inventory():
    """Only these 60 named files may be uploaded; no recursive local mounts."""
    closure = _closure()
    pairs = [(SCRIPT, SCRIPT), (ARTIFACT, ARTIFACT),
             (REFERENCE / "processor-source-closure.json", REFERENCE / "processor-source-closure.json")]
    if _pin(ARTIFACT)["sha256"] != ARTIFACT_SHA:
        raise ValueError("Unchanged public artifact verifier required")
    for label, spec in sorted(closure["files"].items()):
        path = REFERENCE / "processor-runtime-sources" / label
        expected = {key: spec[key] for key in ("sha256", "size_bytes")}
        if _pin(path) != expected:
            raise ValueError("Preserved processor source changed: " + label)
        pairs.append((path, path))
    for label, expected in sorted(CACHE_PINS.items()):
        path = REFERENCE / "processor-metadata" / label
        if _pin(path) != expected:
            raise ValueError("Preserved processor metadata changed: " + label)
        pairs.append((path, CACHE / label))
    if len(pairs) != 60 or len({str(remote) for _, remote in pairs}) != 60:
        raise ValueError("Exact unique 60-file payload required")
    return [{"local_path": str(local), "remote_path": str(remote), **_pin(local)} for local, remote in pairs]


def _child():
    """Fresh target interpreter: standard library plus unchanged ART only."""
    started = time.monotonic()
    report = {"schema_id": "public07-modal-cpu-environment-probe-v1", "success": False,
              "scope": "51 sources/nine installed distribution metadata/six processor cache assets; no framework binary attestation",
              "interpreter": sys.executable, "python": list(sys.version_info[:3]),
              "processor_initializations": 0, "model_loads": 0, "model_calls": 0,
              "training_updates": 0, "GPU_requested": False, "versions": {},
              "source_mismatches": [], "package_mismatches": [], "cache_mismatches": []}
    try:
        closure = _closure()
        if _pin(ARTIFACT)["sha256"] != ARTIFACT_SHA:
            raise ValueError("Public artifact verifier bytes changed")
        reference = {"processor-source-closure.json": _read(REFERENCE / "processor-source-closure.json")}
        for label, spec in closure["files"].items():
            expected = {key: spec[key] for key in ("sha256", "size_bytes")}
            captured = REFERENCE / "processor-runtime-sources" / label
            if _pin(captured) != expected:
                raise ValueError("Preserved reference changed: " + label)
            reference["processor-runtime-sources/" + label] = _read(captured)
            try:
                actual = _pin(spec["path"])
            except (OSError, ValueError) as exc:
                actual = {"error_type": type(exc).__name__, "message": str(exc)[:1000]}
            if actual != expected:
                report["source_mismatches"].append({"label": label, "expected": expected, "actual": actual})
        for name, spec in closure["packages"].items():
            expected = {"version": closure["versions"][name], "metadata_path": spec["metadata_path"],
                        **{key: spec[key] for key in ("sha256", "size_bytes")}}
            try:
                distribution = importlib.metadata.distribution(name)
                path = Path(distribution._path) / "METADATA"
                actual = {"version": distribution.version, "metadata_path": str(path.absolute()), **_pin(path)}
                report["versions"][name] = distribution.version
            except (OSError, ValueError, importlib.metadata.PackageNotFoundError) as exc:
                actual = {"error_type": type(exc).__name__, "message": str(exc)[:1000]}
            if actual != expected:
                report["package_mismatches"].append({"name": name, "expected": expected, "actual": actual})
        for label, expected in CACHE_PINS.items():
            blob = _read(CACHE / label)
            actual = {"sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)}
            reference["processor-metadata/" + label] = blob
            if actual != expected:
                report["cache_mismatches"].append({"name": label, "expected": expected, "actual": actual})
        sys.path.insert(0, str(ROOT / "scripts"))
        import jev_public_action_artifact as artifact
        report["verified_versions"] = artifact.verify_reference_environment(tuple(reference.items()), model=None)
        report["success"] = not any(report[key] for key in ("source_mismatches", "package_mismatches", "cache_mismatches"))
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    report["forbidden_imports"] = sorted(name for name in sys.modules if name.split(".")[0] in FORBIDDEN)
    if report["forbidden_imports"]:
        report["success"] = False
        report.setdefault("error", {"type": "UnexpectedImport", "message": "Framework module imported during metadata-only probe"})
    report["elapsed_seconds"] = time.monotonic() - started
    blob = json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    if len(blob) > MAX_LOG_BYTES:
        raise ValueError("Probe report exceeds log bound")
    sys.stdout.buffer.write(blob + b"\n")
    return 0 if report["success"] else 1


if __name__ == "__main__" and sys.argv[1:] == ["--metadata-child"]:
    raise SystemExit(_child())

# No Modal imports or image definitions are reached by the metadata-only child.
import modal  # noqa: E402 -- the metadata child exits before importing this SDK.


def _install(index, packages, *, target=None):
    command = [INTERPRETER, "-m", "pip", "--isolated", "install", "--disable-pip-version-check",
               "--no-cache-dir", "--only-binary=:all:", "--no-deps", "--timeout", "60", "--retries", "1",
               "--index-url", index]
    if target is not None:
        command.extend(["--target", str(target)])
    return shlex.join([*command, *packages])


app = modal.App("gemma-public07-environment-probe", include_source=False)
image = (
    modal.Image.debian_slim(python_version="3.13")
    .run_commands(
        shlex.join(["python", "-m", "venv", "--copies", "/home/albert_hsieh/ComfyUI/.venv"]),
        _install("https://download.pytorch.org/whl/cu130", ["torch==2.13.0+cu130", "torchvision==0.28.0+cu130"]),
        _install("https://pypi.org/simple", ["numpy==2.5.2", "pillow==12.3.0", "jinja2==3.1.6"]),
        _install("https://pypi.org/simple", ["transformers==5.17.0", "tokenizers==0.23.3", "huggingface_hub==1.33.0", "safetensors==0.8.0"], target=OVERLAY),
    )
    .env({"PYTHONPATH": str(SCRIPT.parent) + os.pathsep + str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"})
)
if modal.is_local():
    for item in upload_inventory():
        image = image.add_local_file(item["local_path"], item["remote_path"], copy=True)


def _stop_child(process):
    """Terminate and reap only this invocation's private session/process group."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            continue
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
    raise RuntimeError("Owned metadata-child process group exit unconfirmed")


@app.function(image=image, cpu=2, memory=4096, timeout=240, max_containers=1,
              single_use_containers=True, scaledown_window=2, include_source=False)
def probe():
    environment = {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8",
                   "PYTHONPATH": str(OVERLAY), "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                   "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "HF_HUB_CACHE": str(ROOT / "artifacts/jev-local-hf/hub"), "TOKENIZERS_PARALLELISM": "false"}
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen([INTERPRETER, "-B", "-s", str(SCRIPT), "--metadata-child"],
                                   env=environment, cwd=str(ROOT), stdin=subprocess.DEVNULL,
                                   stdout=stdout, stderr=stderr, start_new_session=True)
        started = time.monotonic()
        primary = None
        try:
            while process.poll() is None:
                if max(os.fstat(stdout.fileno()).st_size, os.fstat(stderr.fileno()).st_size) > MAX_LOG_BYTES:
                    raise RuntimeError("Owned metadata child exceeded bounded output")
                if time.monotonic() - started >= 180:
                    raise TimeoutError("Owned metadata child exceeded 180 seconds")
                time.sleep(0.05)
        except BaseException as exc:
            primary = exc
        try:
            _stop_child(process)
        except BaseException as exc:
            if primary is None:
                primary = exc
            else:
                primary.add_note("Owned child cleanup: " + repr(exc))
        if primary is not None and not isinstance(primary, Exception):
            raise primary
        stdout.seek(0)
        stderr.seek(0)
        out, err = stdout.read(MAX_LOG_BYTES + 1), stderr.read(MAX_LOG_BYTES + 1)
        if len(out) > MAX_LOG_BYTES or len(err) > MAX_LOG_BYTES:
            raise ValueError("Owned child final logs exceed bounds")
        try:
            result = json.loads(out)
            if type(result) is not dict or type(result.get("success")) is not bool:
                raise ValueError("Child report must contain a boolean success")
        except (ValueError, UnicodeError) as exc:
            result = {"success": False, "error": {"type": type(exc).__name__, "message": str(exc)[:2000]},
                      "stdout": out.decode("utf-8", errors="replace")}
        if primary is not None:
            result["success"] = False
            result["controller_error"] = {"type": type(primary).__name__, "message": str(primary)[:2000],
                                          "notes": getattr(primary, "__notes__", [])}
        return {"child_returncode": process.returncode, "child": result,
                "stderr": err.decode("utf-8", errors="replace"), "modal_app_id": app.app_id,
                "modal_call_id": modal.current_function_call_id()}


@app.local_entrypoint()
def main(output: str):
    destination = Path(output).absolute()
    if destination.exists() or not destination.parent.is_dir():
        raise ValueError("Provide a fresh result path in an existing directory")
    before = upload_inventory()
    result = probe.remote()
    after = upload_inventory()
    result["upload_inventory"] = before
    if before != after:
        result["child"]["success"] = False
        result["local_source_error"] = "Upload payload changed while the CPU environment probe was running"
    blob = json.dumps(result, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    with destination.open("xb") as stream:
        stream.write(blob)
    print(json.dumps({"result": str(destination), "child_returncode": result["child_returncode"],
                      "success": result["child"]["success"]}, sort_keys=True))
    if result["child_returncode"] != 0 or result["child"]["success"] is not True:
        raise RuntimeError("Modal CPU environment did not match the fixed processor reference; see saved result")
