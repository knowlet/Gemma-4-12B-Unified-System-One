"""Explicit isolated composition of the unchanged native driver with Q95 text."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[2]
PROFILE = "jev-current-nf4-v7-q95-browser-v1"
NATIVE = ROOT / "scripts/jev_native_v2_browser.py"
NATIVE_SHA = "c513dc5bcb091de4fd0d4bd7a0776078e82be16106328b6de9d8424e32f2ee11"


def need(value, reason):
    if not value:
        raise ValueError(reason)


def selected_module(path, sha, name):
    path = Path(path).absolute()
    need(
        not any(p.is_symlink() for p in (path, *path.parents)),
        "Selected ordinary composition source",
    )
    blob = path.read_bytes()
    need(
        len(blob) <= 1024**2 and hashlib.sha256(blob).hexdigest() == sha,
        "Selected consumed composition source",
    )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(blob, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


def selected_local(plan, name):
    path = HERE / name
    refs = [r for r in plan["sources"] if ROOT / r["path"] == path]
    need(len(refs) == 1, "Exact selected control component")
    return selected_module(path, refs[0]["sha256"], "_native_q95_" + path.stem)


def bind_text_client(argv, env, connection):
    """Only these text fields change; original policy route/model/key stay intact."""
    result = list(argv)
    for flag, value in (
        ("--text-endpoint", connection["endpoint"]),
        ("--text-model", connection["model"]),
    ):
        need(result.count(flag) == 1, "One explicit original text argument")
        index = result.index(flag) + 1
        need(index < len(result), "Original text argument value")
        result[index] = value
    need(
        result.count("--text-api-key-env") == 1
        and result[result.index("--text-api-key-env") + 1] == "S1_TEXT_API_KEY",
        "Original text credential channel",
    )
    return result, {**env, "S1_TEXT_API_KEY": connection["api_key"]}


def request_ids(service):
    root = Path(service) / "requests"
    paths = list(root.iterdir()) if root.exists() else []
    need(
        len(paths) <= 5000
        and all(
            p.is_dir()
            and not p.is_symlink()
            and len(p.name) == 32
            and all(c in "0123456789abcdef" for c in p.name)
            for p in paths
        ),
        "Bounded actual Q95 request population",
    )
    return {p.name for p in paths}


def index_external_requests(service, before, native, output):
    ids = request_ids(service) - before
    until = time.monotonic() + 3
    rows = {}
    for request_id in sorted(ids):
        directory = Path(service) / "requests" / request_id
        while not (directory / "outer-completion.json").is_file():
            need(time.monotonic() < until, "Actual Q95 final evidence drain deadline")
            time.sleep(0.02)
        raw_path, final_path = directory / "raw.json", directory / "outer-completion.json"
        raw, final = json.loads(raw_path.read_bytes()), json.loads(final_path.read_bytes())
        need(
            raw.get("request_id") == final.get("request_id") == request_id
            and raw.get("guard_restored") is True
            and raw.get("server_terminal") is True
            and raw.get("error") is None
            and final.get("completed") is True
            and final.get("final_gate_measured") is True
            and final.get("raw_sha256") == native.pin(raw_path)["sha256"],
            "Actual unpruned Q95 raw/final evidence join",
        )
        rows[request_id] = {
            "raw": {"path": str(raw_path), **native.pin(raw_path)},
            "outer_completion": {"path": str(final_path), **native.pin(final_path)},
            "selected_status": final["selected_status"],
            "scope": "Q95 request population during owned browser group, including source-bound authentication schema probes; not a fabricated per-client request-ID join",
        }
    native.write(
        output / "external-q95-request-index.json",
        {
            "request_ids": sorted(ids),
            "requests": rows,
            "pruned": False,
            "scope": "External Q95 traces are separate from the native policy server journal",
        },
    )
    return len(rows)


def execute(plan, descriptor, helper_module, boundary_module, rest):
    native = selected_module(NATIVE, NATIVE_SHA, "_native_q95_original_driver")
    run_directory = ROOT / plan["run_directory"]
    control_dir = run_directory / "q95-control"
    control_dir.mkdir(mode=0o700)
    helper_out = control_dir / "helper"
    helper_out.mkdir(mode=0o700)
    checker = helper_module.GemmaHelperChecker(
        descriptor, plan["helper_selection"]["sha256"], helper_out
    )
    checker(checker.binding, "before-native-driver-imports")
    connection = checker.text_connection()
    control = boundary_module.ControlState()
    original = {
        key: getattr(native, key)
        for key in (
            "construct_model",
            "make_application",
            "run_client",
            "matrix_common",
            "composition_paths",
            "metadata_plan",
            "run",
        )
    }
    extra_sources = {
        str((HERE / name).relative_to(ROOT)): HERE / name
        for name in ("control_driver.py", "gemma_helper.py", "policy_boundary.py")
    }
    extra_sources[str(helper_module.PARENT.relative_to(ROOT))] = helper_module.PARENT
    group_indexes = []

    def construct():
        control.healthy()
        checker(checker.binding, "before-native-constructor")
        return original["construct_model"]()

    def application(*args, **kwargs):
        app = original["make_application"](*args, **kwargs)
        output = control_dir / "policy"
        output.mkdir(mode=0o700)
        return boundary_module.PolicyBoundary(app, checker, output, args[5], control)

    def matrix():
        common = original["matrix_common"]()

        class CheckedMatrix:
            def __getattr__(self, name):
                return getattr(common, name)

            def extend_owned(self, *args, **kwargs):
                control.healthy()
                return common.extend_owned(*args, **kwargs)

        return CheckedMatrix()

    def client(argv, env, output):
        control.healthy()
        checker(checker.binding, "before-browser-group-" + str(len(group_indexes)))
        before = request_ids(checker._value["service_directory"])
        selected_argv, selected_env = bind_text_client(argv, env, connection)
        original["run_client"](selected_argv, selected_env, output)
        control.healthy()
        checker(checker.binding, "after-browser-group-" + str(len(group_indexes)))
        count = index_external_requests(
            checker._value["service_directory"], before, native, Path(output)
        )
        group_indexes.append({"group_directory": str(output), "external_q95_requests": count})

    def metadata(*args, **kwargs):
        value = original["metadata_plan"](*args, **kwargs)
        need(value["prompt_format"] == "shared-elements-v2", "Matched selected V2 only")
        return {
            **value,
            "external_text_helper": {
                "profile": checker.binding["profile"],
                "binding": checker.binding,
                "selection_sha256": plan["helper_selection"]["sha256"],
                "seconds": 20,
                "tokens": 128,
                "context": 8192,
            },
            "internal_nf4_helper_role": "Original preparation retained as unused evidence; local text route refuses",
            "source_composition": "Original checked native driver with explicitly selected private bindings",
            "native_policy_deadline_including_helper_checks": 20,
        }

    def run(value, **kwargs):
        primary, receipt = None, None
        try:
            receipt = original["run"](value, **kwargs)
            control.healthy()
            checker(checker.binding, "after-native-service-drain")
            need(
                receipt.get("stage") == "complete_pending_external_closure"
                and receipt.get("attempts") == 10
                and len(group_indexes) == 2,
                "Complete original ten plus both external helper indexes",
            )
        except BaseException as error:
            primary = error
            control.fail(error)
        finally:
            try:
                native.write(
                    control_dir / "receipt.json",
                    {
                        "profile": PROFILE,
                        "stage": "complete_pending_external_closure"
                        if primary is None
                        else "failed",
                        "helper_binding": checker.binding,
                        "selection_sha256": plan["helper_selection"]["sha256"],
                        "native_receipt": native.pin(ROOT / plan["output"] / "receipt.json")
                        if (ROOT / plan["output"] / "receipt.json").exists()
                        else None,
                        "group_indexes": group_indexes,
                        "policy_candidates_and_publication_attempts": control.requests,
                        "primary_error": control.failure,
                        "native_journal_covers_external_helper": False,
                        "qualified": False,
                        "external_owned_closure_required": True,
                    },
                )
            except BaseException as persistence:
                if primary is None:
                    primary = persistence
                    control.fail(persistence)
                else:
                    primary.add_note(
                        "Control receipt persistence also failed: " + type(persistence).__name__
                    )
        if primary is not None:
            raise RuntimeError("Gemma/Q95 matched control refused; see retained evidence") from None
        return receipt

    native.PROFILE = PROFILE
    native.construct_model, native.make_application = construct, application
    native.run_client, native.matrix_common = client, matrix
    native.composition_paths = lambda: {**original["composition_paths"](), **extra_sources}
    native.metadata_plan, native.run = metadata, run
    return native.main(rest)


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--q95-control-plan")
    parser.add_argument("--q95-control-plan-sha256")
    args, rest = parser.parse_known_args(argv)
    if "--run" not in rest:
        print(
            json.dumps(
                {
                    "mode": "DRY_NO_MODEL_OR_BROWSER",
                    "profile": PROFILE,
                    "matched_helper": "Q95",
                    "model_execution": False,
                }
            )
        )
        return 0
    need(args.q95_control_plan and args.q95_control_plan_sha256, "Explicit selected control plan")
    blob = Path(args.q95_control_plan).read_bytes()
    need(
        len(blob) <= 2 * 1024**2
        and hashlib.sha256(blob).hexdigest() == args.q95_control_plan_sha256,
        "Selected control plan bytes",
    )
    plan = json.loads(blob)
    helper = selected_local(plan, "gemma_helper.py")
    boundary = selected_local(plan, "policy_boundary.py")
    ref = plan["helper_selection"]
    descriptor = (ROOT / ref["path"]).read_bytes()
    need(
        len(descriptor) == ref["size_bytes"]
        and hashlib.sha256(descriptor).hexdigest() == ref["sha256"],
        "Selected Gemma/Q95 descriptor bytes",
    )
    return execute(plan, descriptor, helper, boundary, rest)


if __name__ == "__main__":
    main()
