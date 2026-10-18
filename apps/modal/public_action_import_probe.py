"""CPU-only negative importer-cache regression using the cached public07 image.

Only this public probe source is added to the image. No volumes, model, package
installation, network or genuine project package import is involved. Review
before running: .venv/bin/modal run apps/modal/public_action_import_probe.py
  --run --output NEW.json
"""

import hashlib
import importlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
SCRIPT = ROOT / "apps/modal/public_action_import_probe.py"
OVERLAY = ROOT / "artifacts/jev-runtime/overlay"
INTERPRETER = "/home/albert_hsieh/ComfyUI/.venv/bin/python"
TARGET = "s1.browser_public_training_validation"
MAX_LOG = 64 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def encode(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, ensure_ascii=False).encode()


def cache_child(source):
    """Find specs only; never execute the synthetic target module."""
    source = Path(source).absolute()
    report = {
        "success": False,
        "interpreter": sys.executable,
        "python_major_minor": list(sys.version_info[:2]),
        "pythonpath": os.environ.get("PYTHONPATH"),
        "source": str(source),
        "target": TARGET,
        "model_loads": 0,
        "training_updates": 0,
        "target_imported": TARGET in sys.modules,
    }
    try:
        require(
            sys.executable == INTERPRETER and sys.version_info[:2] == (3, 13),
            "Exact fresh ComfyPython3.13 required",
        )
        require(
            os.environ.get("PYTHONPATH") == os.pathsep.join((str(OVERLAY), str(source))),
            "Exact overlay and nonexistent temporary src PYTHONPATH required",
        )
        require(
            str(source) in sys.path and not source.exists(),
            "Temporary src must not exist at process startup",
        )
        require(
            "s1" not in sys.modules and TARGET not in sys.modules,
            "Fresh process must not have imported the synthetic package",
        )
        # Force a real path search even if these stdlib imports happened to
        # resolve without visiting every PYTHONPATH entry on a Python build.
        require(
            importlib.util.find_spec("jev_import_probe_missing_module") is None,
            "Missing probe name unexpectedly exists",
        )
        report["before_create"] = {
            "source_exists": source.exists(),
            "cache_entry_present": str(source) in sys.path_importer_cache,
            "cache_entry_is_none": sys.path_importer_cache.get(str(source), False) is None,
        }
        require(
            report["before_create"]
            == {"source_exists": False, "cache_entry_present": True, "cache_entry_is_none": True},
            "Did not observe the actual negative importer cache",
        )
        package = source / "s1"
        package.mkdir(parents=True)
        (package / "__init__.py").write_bytes(b"# Empty synthetic package, not project code.\n")
        target = package / "browser_public_training_validation.py"
        target.write_bytes(
            b"raise AssertionError('The spec-only probe target must never execute')\n"
        )
        report["created"] = {
            "target": str(target),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "size_bytes": target.stat().st_size,
        }
        try:
            baseline = importlib.util.find_spec(TARGET)
            report["baseline"] = {
                "spec": None if baseline is None else baseline.origin,
                "error": None,
            }
        except ModuleNotFoundError as exc:
            report["baseline"] = {
                "spec": None,
                "error": {"type": type(exc).__name__, "name": exc.name, "message": str(exc)[:2000]},
            }
        require(
            report["baseline"]["error"] is not None and report["baseline"]["error"]["name"] == "s1",
            "Late-created package did not reproduce the baseline ModuleNotFoundError",
        )
        require(TARGET not in sys.modules, "Spec lookup imported the target")
        importlib.invalidate_caches()
        report["after_invalidate"] = {
            "cache_entry_present": str(source) in sys.path_importer_cache,
            "cache_entry_is_none": sys.path_importer_cache.get(str(source), False) is None,
        }
        spec = importlib.util.find_spec(TARGET)
        report["resolved"] = {
            "origin": None if spec is None else spec.origin,
            "target_imported": TARGET in sys.modules,
        }
        require(
            spec is not None and spec.origin == str(target) and TARGET not in sys.modules,
            "Invalidation must find the exact same unexecuted target",
        )
        report["success"] = True
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
    report["target_imported"] = TARGET in sys.modules
    print(encode(report).decode(), flush=True)
    return 0 if report["success"] else 1


def group_alive(process):
    process.poll()
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        return False
    return True


def stop_child(process):
    for signum in (signal.SIGTERM, signal.SIGKILL):
        if not group_alive(process):
            break
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass
        until = time.monotonic() + 2
        while group_alive(process) and time.monotonic() < until:
            time.sleep(0.02)
    require(
        process.poll() is not None and not group_alive(process),
        "Owned import probe exit unconfirmed",
    )


def run_owned_child():
    primary, process, confirmed = None, None, False
    with (
        tempfile.TemporaryDirectory(prefix="jev-import-probe-") as temporary,
        tempfile.TemporaryFile() as stdout,
        tempfile.TemporaryFile() as stderr,
    ):
        source = Path(temporary) / "src"
        environment = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "PYTHONPATH": os.pathsep.join((str(OVERLAY), str(source))),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
        }
        try:
            process = subprocess.Popen(
                [INTERPRETER, "-B", "-s", str(SCRIPT), "--cache-child", str(source)],
                cwd=ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            started = time.monotonic()
            while process.poll() is None:
                require(
                    max(os.fstat(stdout.fileno()).st_size, os.fstat(stderr.fileno()).st_size)
                    <= MAX_LOG,
                    "Owned import probe logs exceed64KiB",
                )
                if time.monotonic() - started >= 30:
                    raise TimeoutError("Owned import probe exceeded30s")
                time.sleep(0.02)
        except BaseException as exc:
            primary = exc
        finally:
            if process is not None:
                try:
                    stop_child(process)
                    confirmed = True
                except BaseException as secondary:
                    if primary is None:
                        primary = secondary
                    else:
                        primary.add_note("Owned cleanup: " + repr(secondary))
        stdout.seek(0)
        stderr.seek(0)
        out, err = stdout.read(MAX_LOG), stderr.read(MAX_LOG)
        sizes = {
            "stdout": os.fstat(stdout.fileno()).st_size,
            "stderr": os.fstat(stderr.fileno()).st_size,
        }
        if max(sizes.values()) > MAX_LOG and primary is None:
            primary = ValueError("Owned import probe final logs exceed64KiB")
        try:
            child = json.loads(out)
            require(
                type(child) is dict and type(child.get("success")) is bool,
                "Actual child JSON success required",
            )
        except (ValueError, UnicodeError) as exc:
            child = {
                "success": False,
                "parse_error": str(exc)[:2000],
                "stdout": out.decode(errors="replace"),
            }
        result = {
            "success": primary is None
            and confirmed
            and process.returncode == 0
            and child["success"],
            "owned_exit_confirmed": confirmed,
            "child_returncode": None if process is None else process.returncode,
            "child": child,
            "stderr": err.decode(errors="replace"),
            "log_bytes": sizes,
            "error": None
            if primary is None
            else {
                "type": type(primary).__name__,
                "message": str(primary)[:2000],
                "notes": getattr(primary, "__notes__", []),
            },
        }
    if primary is not None and not isinstance(primary, Exception):
        raise primary
    return result


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--cache-child":
    raise SystemExit(cache_child(sys.argv[2]))

import modal  # noqa: E402 -- the stdlib-only child exits before any SDK/image import.

from apps.modal.public_action_fit import image  # noqa: E402

app = modal.App("public07-negative-import-cache-probe", include_source=False)
probe_image = image.add_local_file(str(SCRIPT), str(SCRIPT), copy=True)


@app.function(
    image=probe_image,
    cpu=1,
    memory=4096,
    timeout=60,
    max_containers=1,
    retries=0,
    single_use_containers=True,
    include_source=False,
    block_network=True,
)
def probe_import_cache():
    result = run_owned_child()
    result["modal_app_id"] = app.app_id
    result["modal_call_id"] = modal.current_function_call_id()
    result["probe_source_sha256"] = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    return result


@app.local_entrypoint()
def main(output: str, run: bool = False):
    require(run, "Explicit --run after source review required")
    destination = Path(output).absolute()
    require(
        not destination.exists() and destination.parent.is_dir(),
        "Fresh local output in an existing directory required",
    )
    before = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    report = probe_import_cache.remote()
    report["local_probe_source_sha256"] = before
    if (
        before != hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
        or report.get("probe_source_sha256") != before
    ):
        report["success"] = False
        report["source_error"] = "Local/remote probe source identity differs"
    with destination.open("xb") as stream:
        stream.write(encode(report) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"output": str(destination), "success": report["success"]}, sort_keys=True))
    require(report["success"], "Import cache regression probe failed; inspect the saved result")
