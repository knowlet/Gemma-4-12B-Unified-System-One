"""Run one externally pinned browser child in its own group; default is metadata dry."""

import argparse
import importlib.util
import os
import signal
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OBSERVER = Path(__file__).with_name("browser_observer_runner.py")
OBSERVER_SHA = "8e00743c7b3caa48a18692e652571f8e03b8cbb9906f7462f243e2ac062150f9"
QUALIFIER = ROOT / "scripts/jev_qualify_public_field_syntax_bf16.py"
QUALIFIER_SHA = "cbfb7bdd8301174a97ac18389a998ff3256ac32d6f7a770978f2f9b722c3b43b"
PYTHON = "/home/albert_hsieh/ComfyUI/.venv/bin/python"
SECRET_NAMES = ("S1_POLICY_API_KEY", "S1_TEXT_API_KEY")
FILES = tuple(
    "controller-intent.json controller-process.json controller-exit.json "
    "controller-stdout.txt controller-stderr.txt".split()
)


def ordinary(path):
    path = Path(path).absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Ordinary controller path required")
    return path


def load(path, name, expected):
    import hashlib

    path = ordinary(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Reviewed controller dependency drift")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def prepare(launch_path, expected):
    launch_path = ordinary(launch_path)
    r = load(OBSERVER, "owned_group_observer", OBSERVER_SHA)
    if r.pin(launch_path)["sha256"] != expected:
        raise ValueError("Externally reviewed launch SHA mismatch")
    launch = r.read_json(launch_path)
    r.checked_ref(launch["observer_source"])
    if launch["observer_source"]["path"] != str(OBSERVER.relative_to(ROOT)):
        raise ValueError("Exact current observer required")
    plan = r.checked_plan()
    if launch["environment"] != {
        "updates": plan["environment"]["updates"],
        "secret_names_only": list(SECRET_NAMES),
    }:
        raise ValueError("Original environment and exact secret names required")
    argv = [PYTHON, "-B", "-s", str(OBSERVER)] + ["child", "--run", "--launch", str(launch_path)]
    argv += ["--launch-sha256", expected]
    command = r.read_json(ordinary(launch_path.with_name("child-command.json")))
    if command["argv"] != argv or command["owned_fresh_session_required"] is not True:
        raise ValueError("Exact prepared child argv required")
    return r, launch, argv


def bound_environment(r, launch):
    binding = r.read_json(r.checked_ref(launch["binding"]))
    config = ordinary(ROOT / binding["service_directory"] / "controller-config.json")
    if stat.S_IMODE(config.stat().st_mode) != 0o600 or not config.is_file():
        raise ValueError("Bound ordinary mode0600 service config required")
    token = r.read_json(config)["token"]
    if type(token) is not str or not token:
        raise ValueError("Nonempty private credential required")
    previous = {name: os.environ.get(name) for name in SECRET_NAMES}
    try:
        os.environ.update(dict.fromkeys(SECRET_NAMES, token))
        state, _ = r.binding_state(binding)
        if r.canonical(state) != r.canonical(launch["public_service_identity"]):
            raise ValueError("Owned service binding changed")
    finally:
        for name, value in previous.items():
            os.environ.pop(name, None) if value is None else os.environ.__setitem__(name, value)
    return {**os.environ, **launch["environment"]["updates"], **dict.fromkeys(SECRET_NAMES, token)}


def execute(launch_path, expected, *, run=False):
    r, launch, argv = prepare(launch_path, expected)
    if not run:
        return 0  # No private-config read, process, browser, HTTP or binding attestation.
    q = load(QUALIFIER, "owned_group_qualifier", QUALIFIER_SHA)
    output = ordinary(launch_path).parent
    if any((output / name).exists() for name in FILES):
        raise ValueError("Fresh controller files required; refuse overwrite")
    env = bound_environment(r, launch)
    receipt = {
        "launch_sha256": expected,
        "argv": argv,
        "process_exit_is_task_success": False,
        "closure_confirmed": False,
        "actual_returncode": None,
        "error_type": None,
    }
    r.write(output / FILES[0], r.canonical(receipt))
    process = primary = None
    try:
        with (output / FILES[3]).open("xb") as stdout, (output / FILES[4]).open("xb") as stderr:
            os.chmod(stdout.name, 0o600)
            os.chmod(stderr.name, 0o600)
            streams = {"stdin": subprocess.DEVNULL, "stdout": stdout, "stderr": stderr}
            process = subprocess.Popen(argv, env=env, start_new_session=True, **streams)
            identity = r.process_identity(process.pid)
            r.write(output / FILES[1], r.canonical(identity))
            if (
                identity["pgid"] != process.pid
                or identity["session"] != process.pid
                or list(identity["argv"]) != argv
            ):
                raise ValueError("Owned child session/command mismatch")
            process.wait(timeout=7200)
    except BaseException as exc:
        primary = exc
        receipt["error_type"] = type(exc).__name__
        raise
    finally:
        cleanup = []
        if process is not None:
            if primary is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGINT)
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                except BaseException as exc:
                    cleanup.append(exc)
            try:
                receipt["closure_confirmed"] = q._stop_owned(process, drain=2)
                if not receipt["closure_confirmed"]:
                    raise RuntimeError("Owned browser group closure unknown")
            except BaseException as exc:
                cleanup.append(exc)
            receipt["actual_returncode"] = process.poll()
        receipt["cleanup_error_types"] = [type(e).__name__ for e in cleanup]
        try:
            r.write(output / FILES[2], r.canonical(receipt))
        except BaseException as exc:
            cleanup.append(exc)
        if primary is not None:
            for exc in cleanup:
                primary.add_note("Owned group cleanup: " + type(exc).__name__)
        elif cleanup:
            raise cleanup[0]
    return process.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--launch-sha256", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    raise SystemExit(execute(args.launch, args.launch_sha256, run=args.run))
