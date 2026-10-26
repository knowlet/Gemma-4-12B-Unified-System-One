"""Capacity-bound revisit browser preparation, transport observation and ownership.

Default invocation is metadata only. Preparation requires both the new closed75
capacity and the original live Q95 service. Each explicit execution owns only
its browser child, samples PID/start ancestry, and records whole-group cleanup.
The separately owned model service is never stopped here. Exit0 and complete
HTTP capture do not establish any task success or an all10 benchmark result.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import runpy
import stat
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).absolute().parents[3]
HERE = Path(__file__).absolute().parent
PROFILE = "composed-visible-hover-entries-settled-revisit-v1"
KIND = "jev_revisit_observed_browser_launch_v1"
PYTHON = "/home/albert_hsieh/ComfyUI/.venv/bin/python"
SECRET_NAMES = ("S1_POLICY_API_KEY", "S1_TEXT_API_KEY")
SAFE_ENV = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR")
MAX_FILE, MAX_OUTPUT, TERMINAL_RESERVE = 64 * 1024**2, 1024**3, 1024**2
LOG_BYTES, WALL_SECONDS, SAMPLE_SECONDS = 8 * 1024**2, 7200, 0.1
PINS = {
    "service_binding": {
        "path": "artifacts/jev-runtime/public-revisit-browser-observer-20261020-01/service_binding.py",
        "sha256": "19e55e3dc8f71405281594d846e3f5aeb74060df2874bf7e3dd039f86dbb6dac",
        "size_bytes": 9050,
    },
    "launcher_adapter": {
        "path": "artifacts/jev-runtime/public-revisit-browser-binding-20261020-01/launcher_adapter.py",
        "sha256": "b8fc5033ea4a28d35875122565d0b817332ea2c23e2999340af8b2d1c8e72fd5",
        "size_bytes": 9708,
    },
    "owned_lifecycle": {
        "path": "artifacts/jev-runtime/public-affordance-matrix-harness-20261019-01/matrix_common.py",
        "sha256": "71251036c63e8df852ae22814d7234db0daaee90436c265e8a5549de468c6291",
        "size_bytes": 7229,
    },
}
LAUNCH_KEYS = {
    "schema_id",
    "profile",
    "group",
    "case_ids",
    "denominator",
    "target",
    "forwarded_argv",
    "binding",
    "public_service_identity",
    "sources",
    "environment",
    "bundle_manifest",
    "measurement_conditions",
}
CONTROLLER_FILES = (
    "controller-intent.json",
    "controller-process.json",
    "controller-exit.json",
    "controller-stdout.txt",
    "controller-stderr.txt",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def encode(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def ordinary(path):
    path = Path(path).absolute()
    require(path.is_relative_to(ROOT) and ".." not in path.parts, "Repository-confined path")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Ordinary source/output path")
    return path


def read(path, limit=2 * 1024**2):
    path = ordinary(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit, "Bounded regular file"
        )
        blob = stream.read(before.st_size + 1)
        after = os.fstat(stream.fileno())

    def identity(s):
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns

    require(
        identity(before) == identity(after) == identity(path.stat())
        and len(blob) == before.st_size,
        "Stable descriptor/path identity",
    )
    return blob


def parse(blob):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    return json.loads(blob, object_pairs_hook=pairs, parse_constant=lambda x: require(False, x))


def reference(path):
    path = ordinary(path)
    blob = read(path)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "size_bytes": len(blob),
    }


def checked(ref):
    require(type(ref) is dict and set(ref) == {"path", "sha256", "size_bytes"}, "Exact file ref")
    name = ref["path"]
    require(
        type(name) is str
        and name
        and not Path(name).is_absolute()
        and Path(name).as_posix() == name
        and ".." not in Path(name).parts,
        "Relative file ref",
    )
    require(
        type(ref["sha256"]) is str
        and len(ref["sha256"]) == 64
        and all(c in "0123456789abcdef" for c in ref["sha256"])
        and type(ref["size_bytes"]) is int
        and 0 < ref["size_bytes"] <= 2 * 1024**2,
        "Bounded SHA256 file ref",
    )
    path = ordinary(ROOT / name)
    require(reference(path) == ref, "Selected source/evidence bytes changed")
    return path


def component(key):
    path = checked(PINS[key])
    blob = read(path)
    name = "_revisit_browser_" + key + "_" + PINS[key]["sha256"][:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(blob, str(path), "exec"), module.__dict__)
    require(reference(path) == PINS[key], "Component changed during load")
    return module


def sources():
    for ref in PINS.values():
        checked(ref)
    return {"observer": reference(__file__), **PINS}


def caps(output, pending=0, *, terminal=False):
    output = ordinary(output)
    sizes = []
    for path in output.rglob("*"):
        ordinary(path)
        if path.is_file():
            sizes.append(path.stat().st_size)
    require(
        len(sizes) < 10000 and all(s <= MAX_FILE for s in sizes), "Observed file population cap"
    )
    require(
        sum(sizes) + pending <= MAX_OUTPUT - (0 if terminal else TERMINAL_RESERVE),
        "Observed aggregate output/reserve cap",
    )


def save(output, label, value, *, terminal=False):
    blob = encode(value)
    require(len(blob) <= MAX_FILE, "Evidence file cap")
    caps(output, len(blob), terminal=terminal)
    path = ordinary(output / label)
    require(path.parent == output, "Direct owned evidence path")
    with path.open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    return reference(path)


@contextlib.contextmanager
def credentials(token):
    previous = {name: os.environ.get(name) for name in SECRET_NAMES}
    try:
        os.environ.update(dict.fromkeys(SECRET_NAMES, token))
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def bound_environment(binder, binding, updates):
    """Obtain original private token locally; admit only through both binders."""
    legacy = binder.read_legacy_binding(binding)  # Metadata only, never admission.
    service = ordinary(ROOT / legacy["service_directory"])
    config = ordinary(service / "controller-config.json")
    require(stat.S_IMODE(config.stat().st_mode) == 0o600, "Original private config mode0600")
    token = parse(read(config))["token"]
    require(type(token) is str and token, "Nonempty original env-only credential")
    with credentials(token):
        public, secrets = binder.binding_state(binding)
    require(secrets == (token, token), "Both original credentials select the bound service")
    env = {name: os.environ[name] for name in SAFE_ENV if name in os.environ}
    env.update(updates)
    env.update(dict.fromkeys(SECRET_NAMES, token))
    return public, env


def group_conditions(adapter, group):
    selected, _ = adapter.source_arguments(group)
    plan = parse(adapter.selected(adapter.CASE_PLAN))
    # Recheck original full source/upstream authority, in addition to new profile.
    adapter.module(adapter.OBSERVER).checked_plan()
    return selected, plan["environment"]["updates"]


def child_argv(launch_path, digest):
    return [
        PYTHON,
        "-B",
        "-s",
        str(Path(__file__).absolute()),
        "child",
        "--run",
        "--launch",
        str(launch_path),
        "--launch-sha256",
        digest,
    ]


def prepare(group, binding_path, binding_sha, output, source_sha):
    require(reference(__file__)["sha256"] == source_sha, "Explicit reviewed observer source SHA")
    source_refs = sources()
    binder, adapter = component("service_binding"), component("launcher_adapter")
    binding_ref = reference(binding_path)
    require(binding_ref["sha256"] == binding_sha, "Externally selected new binding bytes")
    binding = parse(read(checked(binding_ref)))
    selected, updates = group_conditions(adapter, group)
    public, _env = bound_environment(binder, binding, updates)
    endpoint = public["legacy_q95_service"]["endpoint"]
    output = ordinary(output)
    require(
        output.parent == HERE and output.name.startswith(group + "-") and not output.exists(),
        "Fresh separate selected browser group output",
    )
    forwarded = adapter.forward_arguments(
        group, policy=endpoint["policy"], helper=endpoint["helper"], results=output / "results"
    )
    output.mkdir(mode=0o700)
    with adapter.case_imports():
        launcher = adapter.launcher()
        bundle, manifest = launcher.prepare_bundle(
            output / "runtime-bundle", case_module=adapter.GROUPS[group]
        )
        launcher.verify_bundle(bundle, manifest)
    # Materialization may take time; repeat both independent gates afterward.
    current, _ = bound_environment(binder, binding, updates)
    require(encode(current) == encode(public), "Bound service/capacity changed during preparation")
    require(
        sources() == source_refs and reference(binding_path) == binding_ref,
        "Prepared source/binding bytes stable",
    )
    launch = {
        "schema_id": KIND,
        "profile": PROFILE,
        "group": group,
        "case_ids": selected["selected_case_ids_in_order"],
        "denominator": selected["denominator"],
        "target": reference(bundle / "runner.py"),
        "forwarded_argv": forwarded,
        "binding": binding_ref,
        "public_service_identity": public,
        "sources": source_refs,
        "environment": {"updates": updates, "secret_names_only": list(SECRET_NAMES)},
        "bundle_manifest": manifest,
        "measurement_conditions": conditions(),
    }
    save(output, "launch.json", launch)
    launch_ref = reference(output / "launch.json")
    command = child_argv(output / "launch.json", launch_ref["sha256"])
    save(
        output,
        "child-command.json",
        {
            "argv": command,
            "owned_fresh_session_required": True,
            "whole_benchmark_denominator": 10,
            "group_result_is_all10_score": False,
        },
    )
    return {
        "launch": launch_ref,
        "prepared_child_argv": command,
        "browser_model_HTTP_calls": 0,
        "task_quality_attested": False,
    }


def conditions():
    return {
        "profile": PROFILE,
        "fixed_total_cases": 10,
        "group_order": ["known4", "public6"],
        "repeats": 1,
        "decisions": 16,
        "seed": 20261014,
        "native_model_deadline_seconds": 20,
        "native_HTTP_deadline_seconds": 25,
        "policy_helper_math_and_candidate_heads_changed": False,
        "transport": "Pinned synchronous httpx observer; immutable capture/hash within native send, final fsync outside attempt timing. Native timeout/retry/request/response semantics retained.",
        "strict_success": "Independent fresh DOM verification AND native done AND no execution error. Assess every declared case, including failed or missing outcomes, after whole closure.",
        "authority": "New closed75/all68 acceptance is mandatory; unchanged legacy Q95/V8 technical proof is an additional independent condition. Neither establishes task quality.",
    }


def validate_launch(launch_path, expected_sha):
    launch_path = ordinary(launch_path)
    require(
        launch_path.name == "launch.json" and launch_path.parent.parent == HERE,
        "Selected prepared browser group launch path",
    )
    require(reference(launch_path)["sha256"] == expected_sha, "Externally selected launch SHA")
    launch = parse(read(launch_path))
    require(
        type(launch) is dict
        and set(launch) == LAUNCH_KEYS
        and launch["schema_id"] == KIND
        and launch["profile"] == PROFILE,
        "Exact new typed browser launch",
    )
    require(launch["sources"] == sources(), "Exact current reviewed source authority")
    binder, adapter = component("service_binding"), component("launcher_adapter")
    selected, updates = group_conditions(adapter, launch["group"])
    require(launch_path.parent.name.startswith(launch["group"] + "-"), "Group namespace")
    require(
        launch["case_ids"] == selected["selected_case_ids_in_order"]
        and type(launch["denominator"]) is int
        and launch["denominator"] == selected["denominator"],
        "All fixed original group cases/order/denominator",
    )
    require(
        launch["environment"] == {"updates": updates, "secret_names_only": list(SECRET_NAMES)}
        and launch["measurement_conditions"] == conditions(),
        "Original measured conditions",
    )
    binding_path = checked(launch["binding"])
    binding = parse(read(binding_path))
    binder.binding_shape(binding)
    state = launch["public_service_identity"]
    require(
        state["profile"] == PROFILE and state["task_quality_attested"] is False,
        "Explicit new profile, no task-quality attestation",
    )
    endpoint = state["legacy_q95_service"]["endpoint"]
    require(
        launch["forwarded_argv"]
        == adapter.forward_arguments(
            launch["group"],
            policy=endpoint["policy"],
            helper=endpoint["helper"],
            results=launch_path.parent / "results",
        ),
        "Exact native full-case arguments",
    )
    target = checked(launch["target"])
    require(
        target == launch_path.parent / "runtime-bundle/runner.py", "Exact isolated runner target"
    )
    with adapter.case_imports():
        adapter.launcher().verify_bundle(target.parent, launch["bundle_manifest"])
    command = parse(read(launch_path.with_name("child-command.json")))
    require(
        command
        == {
            "argv": child_argv(launch_path, expected_sha),
            "owned_fresh_session_required": True,
            "whole_benchmark_denominator": 10,
            "group_result_is_all10_score": False,
        },
        "Exact prepared fresh child command",
    )
    return launch, binder, adapter, binding


def child(launch_path, expected_sha):
    launch_path = ordinary(launch_path)
    launch, binder, adapter, binding = validate_launch(launch_path, expected_sha)
    require(
        all(os.environ.get(k) == v for k, v in launch["environment"]["updates"].items()),
        "Original benchmark environment updates",
    )
    technical = binder.technical_observer()
    own = technical.process_identity(os.getpid())
    require(own["pid"] == own["pgid"] == own["session"], "Fresh owned browser child session")
    state, secrets = binder.binding_state(binding)
    require(
        encode(state) == encode(launch["public_service_identity"]), "Live service binding changed"
    )
    endpoint = state["legacy_q95_service"]["endpoint"]
    target, output = checked(launch["target"]), launch_path.parent
    require(
        not any((output / n).exists() for n in ("child-exit.json", "http-transport", "results")),
        "Fresh child output; no selected retry or overwrite",
    )
    import httpx

    observer = technical.TransportObserver(endpoint["policy"], endpoint["helper"], secrets=secrets)
    primary = None
    old_argv, old_path = sys.argv, list(sys.path)
    started = time.monotonic()
    try:
        sys.path.insert(0, str(target.parent))
        sys.argv = [str(target), *launch["forwarded_argv"]]
        with observer.install(httpx):
            runpy.run_path(str(target), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (0, None):
            primary = exc
    except BaseException as exc:
        primary = exc
    finally:
        sys.argv, sys.path = old_argv, old_path
        try:
            observer.drain(output / "http-transport", primary=primary)
        except BaseException as exc:
            if primary is None:
                primary = exc
            else:
                primary.add_note("Observer drain secondary: " + observer._message(exc))
        try:
            validate_launch(launch_path, expected_sha)
            current, _ = binder.binding_state(parse(read(checked(launch["binding"]))))
            require(encode(current) == encode(state), "Live service changed during browser group")
        except BaseException as exc:
            if primary is None:
                primary = exc
            else:
                primary.add_note("Post-group identity secondary: " + observer._message(exc))
        receipt = {
            "schema_id": "jev_revisit_observed_browser_child_exit_v1",
            "complete": primary is None,
            "profile": PROFILE,
            "group": launch["group"],
            "declared_denominator": launch["denominator"],
            "selected_HTTP_attempts": observer.selected_attempt_total,
            "observer_rows": len(observer.snapshot()),
            "observer_omitted": observer.omitted_count,
            "error": None
            if primary is None
            else {
                "type": type(primary).__name__,
                "message": observer._message(primary),
                "notes": [observer._message(n) for n in getattr(primary, "__notes__", [])],
            },
            "elapsed_ms_including_final_drain": (time.monotonic() - started) * 1000,
            "process_exit_is_task_success": False,
            "zero_HTTP_is_task_success": False,
            "all10_task_success_attested": False,
        }
        try:
            save(output, "child-exit.json", receipt)
        except BaseException as exc:
            if primary is None:
                primary = exc
            else:
                primary.add_note("Child receipt secondary: " + observer._message(exc))
    if primary is not None:
        raise primary
    return 0


def run_owned(argv, env, output, common, technical, *, wall=WALL_SECONDS):
    """Sample ancestry throughout execution; signal only matching owned starts."""
    receipt = {
        "argv": argv,
        "closure_confirmed": False,
        "actual_returncode": None,
        "process_exit_is_task_success": False,
        "error_type": None,
        "cleanup": None,
        "output_limits_are_sampled_not_atomic": True,
        "wall_seconds": wall,
        "sample_seconds": SAMPLE_SECONDS,
        "ownership_scope": "Cleanup covers tracked PID/start identities and observed groups. Ancestry is sampled; a descendant detaching and reparenting entirely between samples may remain undiscovered.",
    }
    save(output, CONTROLLER_FILES[0], receipt)
    process = primary = None
    known, groups = {}, set()
    started = time.monotonic()
    try:
        with (
            (output / CONTROLLER_FILES[3]).open("xb") as stdout,
            (output / CONTROLLER_FILES[4]).open("xb") as stderr,
        ):
            for stream in (stdout, stderr):
                os.fchmod(stream.fileno(), 0o600)
            process = subprocess.Popen(
                argv,
                env=env,
                cwd=ROOT,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
            )
            row = None
            identity_deadline = time.monotonic() + 0.25
            # Do not reap the child before recording its start identity. A
            # transient /proc read failure gets a bounded retry, never a signal
            # based only on the integer returned by Popen.
            while row is None:
                row = common.process_table().get(process.pid)
                if row is not None or time.monotonic() >= identity_deadline:
                    break
                time.sleep(0.01)
            require(row is not None, "Fresh child start identity must be observed")
            known[process.pid] = row["start_ticks"]
            groups.add(row["pgid"])
            require(
                row["pgid"] == row["sid"] == process.pid, "Exact fresh owned child group/session"
            )
            identity = technical.process_identity(process.pid)
            require(
                identity["start_ticks"] == known[process.pid] and list(identity["argv"]) == argv,
                "Observed child PID/start/exact command",
            )
            save(output, CONTROLLER_FILES[1], identity)
            while True:
                common.extend_owned(process.pid, known, groups)
                caps(output)
                require(
                    all(os.fstat(s.fileno()).st_size <= LOG_BYTES for s in (stdout, stderr)),
                    "Sampled owned stdout/stderr cap",
                )
                require(time.monotonic() - started < wall, "Owned browser wall deadline")
                if process.poll() is not None:
                    break
                time.sleep(SAMPLE_SECONDS)
    except BaseException as exc:
        primary = exc
        receipt["error_type"] = type(exc).__name__
    finally:
        cleanup_errors = []
        if process is not None:
            try:
                cleanup = common.stop_owned(process, known, groups, grace=2.0)
                receipt["cleanup"] = cleanup
                receipt["closure_confirmed"] = cleanup["confirmed"]
                require(cleanup["confirmed"], "Owned browser child/descendant closure unknown")
            except BaseException as exc:
                cleanup_errors.append(exc)
            receipt["actual_returncode"] = process.poll()
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["cleanup_error_types"] = [type(e).__name__ for e in cleanup_errors]
        try:
            save(output, CONTROLLER_FILES[2], receipt, terminal=True)
        except BaseException as exc:
            cleanup_errors.append(exc)
        if primary is not None:
            for exc in cleanup_errors:
                primary.add_note("Owned browser cleanup secondary: " + type(exc).__name__)
        elif cleanup_errors:
            primary = cleanup_errors[0]
    if primary is not None:
        raise primary
    return receipt


def execute(launch_path, expected_sha, *, run=False):
    launch_path = ordinary(launch_path)
    launch, binder, _adapter, binding = validate_launch(launch_path, expected_sha)
    if not run:
        return {
            "status": "DRY_NO_PRIVATE_CONFIG_PROCESS_BROWSER_OR_HTTP",
            "profile": PROFILE,
            "group": launch["group"],
            "denominator": launch["denominator"],
            "live_binding_attested": False,
            "task_success_attested": False,
        }
    output = launch_path.parent
    require(
        not any(
            (output / name).exists()
            for name in (*CONTROLLER_FILES, "child-exit.json", "http-transport", "results")
        ),
        "Fresh controller/browser evidence; no selected retry",
    )
    public, env = bound_environment(binder, binding, launch["environment"]["updates"])
    require(
        encode(public) == encode(launch["public_service_identity"]), "Current live binding differs"
    )
    result = run_owned(
        child_argv(launch_path, expected_sha),
        env,
        output,
        component("owned_lifecycle"),
        binder.technical_observer(),
    )
    # The model service is separately owned and remains up for the next group.
    require(sources() == launch["sources"], "Sources stable after whole browser closure")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    sub = parser.add_subparsers(dest="command")
    prep = sub.add_parser("prepare", allow_abbrev=False)
    prep.add_argument("--group", required=True, choices=("known4", "public6"))
    prep.add_argument("--binding", type=Path, required=True)
    prep.add_argument("--binding-sha256", required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--source-sha256", required=True)
    for name in ("execute", "child"):
        p = sub.add_parser(name, allow_abbrev=False)
        p.add_argument("--launch", type=Path, required=True)
        p.add_argument("--launch-sha256", required=True)
        p.add_argument("--run", action="store_true", required=(name == "child"))
    args = parser.parse_args(argv)
    if args.command is None:
        print(
            json.dumps(
                {
                    "status": "DRY_SOURCE_ONLY",
                    "profile": PROFILE,
                    "sources": sources(),
                    "browser_model_HTTP_calls": 0,
                    "task_quality_attested": False,
                }
            )
        )
        return 0
    if args.command == "prepare":
        result = prepare(
            args.group, args.binding, args.binding_sha256, args.output, args.source_sha256
        )
    elif args.command == "execute":
        result = execute(args.launch, args.launch_sha256, run=args.run)
    else:
        return child(args.launch, args.launch_sha256)
    print(json.dumps(result, ensure_ascii=False))
    return result.get("actual_returncode", 0) or 0


if __name__ == "__main__":
    raise SystemExit(main())
