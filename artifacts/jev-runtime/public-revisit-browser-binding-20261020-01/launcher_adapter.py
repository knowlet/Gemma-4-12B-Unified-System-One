"""Source-only adapter for the pinned two-argument revisit launcher API.

The fixed ten cases, budgets, seed, model and credential variable names come
from the old benchmark plan as case authority only. This adapter never starts
a browser/service/process or authorizes one. A new owned observer must first
validate CPU capacity and its live technical service independently.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).absolute().parents[3]
HERE = Path(__file__).absolute().parent
PROFILE = "composed-visible-hover-entries-settled-revisit-v1"
KIND = "jev_revisit_two_argument_launcher_adapter_v1"
LAUNCHER = "scripts/jev_browser_revisit_benchmark.py"
HELPER = "scripts/jev_observation_profile_revisit.py"
MANIFEST = (
    "artifacts/jev-runtime/public-revisit-profile-20261020-01/profile-01/profile-manifest.json"
)
CASE_PLAN = "artifacts/jev-runtime/public-field-syntax-v8-browser-retest-20261019-01/plan-01.json"
OBSERVER = "artifacts/jev-runtime/public-field-syntax-v8-browser-retest-20261019-01/browser_observer_runner.py"
PINS = {
    LAUNCHER: ("f7f0c4f708203e107563fa70656bbeb3e04afcba463a0a9cf9eb156fbd8e51b3", 13193),
    HELPER: ("873abc6aa2a02272fd9b54d299d683598bdb3d5c0a848cc43a356daa5a36a0ca", 13642),
    MANIFEST: ("f3e65535836518581deab8487f6c20c837bcecfa03173db85f953cc02975ce88", 12379),
    CASE_PLAN: ("f909d8174e349b4e7facf3ef7866f05d37ed3f1f17eafe26a04c299fe089794c", 33428),
    OBSERVER: ("8e00743c7b3caa48a18692e652571f8e03b8cbb9906f7462f243e2ac062150f9", 32298),
}
GROUPS = {"known4": "original", "public6": "heldout"}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def encode(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def ordinary(path):
    path = Path(path).absolute()
    require(
        path.is_relative_to(ROOT) and ".." not in path.parts, "Repository-confined ordinary path"
    )
    require(not any(p.is_symlink() for p in (path, *path.parents)), "No symlink source/output")
    return path


def read(path, limit=1024**2):
    path = ordinary(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit, "Bounded regular source"
        )
        blob = stream.read(before.st_size + 1)
        after = os.fstat(stream.fileno())

    def identity(s):
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns

    require(
        identity(before) == identity(after) == identity(path.stat())
        and len(blob) == before.st_size,
        "Stable source descriptor identity",
    )
    return blob


def selected(label):
    digest, size = PINS[label]
    blob = read(ROOT / label)
    require(
        len(blob) == size and sha(blob) == digest, "Pinned adapter dependency differs: " + label
    )
    return blob


def module(label):
    blob = selected(label)
    name = "_revisit_adapter_" + Path(label).stem
    spec = importlib.util.spec_from_file_location(name, ROOT / label)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    exec(compile(blob, str(ROOT / label), "exec"), result.__dict__)
    return result


@contextlib.contextmanager
def case_imports():
    """Scope transitive imports for both loading and calling the launcher API."""
    names = ("jev_heldout_public_cases", "jev_public_web_cases")
    previous, old_path = {name: sys.modules.get(name) for name in names}, list(sys.path)
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        yield
    finally:
        sys.path[:] = old_path
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def launcher():
    with case_imports():
        result = module(LAUNCHER)
        require(
            result.PROFILE == PROFILE and result.PROFILE_SHA == PINS[HELPER][0],
            "Exact new launcher profile/helper",
        )
        return result


def replace_one(arguments, flag, expected, value):
    require(arguments.count(flag) == 1, "Exactly one fixed argument: " + flag)
    index = arguments.index(flag) + 1
    require(
        index < len(arguments) and arguments[index] == expected,
        "Original argument binding differs: " + flag,
    )
    arguments[index] = value


def source_arguments(group):
    require(group in GROUPS, "Original fixed benchmark group")
    plan = json.loads(selected(CASE_PLAN))
    source_group = next(g for g in plan["groups_in_execution_order"] if g["group"] == group)
    original = source_group["argv_template"]
    arguments = original[original.index("--") + 1 :]
    replace_one(arguments, "--observation-profile", plan["browser_profile"]["id"], PROFILE)
    replace_one(
        arguments,
        "--observation-profile-helper-sha256",
        plan["browser_profile"]["helper_source_sha256"],
        PINS[HELPER][0],
    )
    replace_one(
        arguments,
        "--output-dir",
        source_group["future_output"] + "/results",
        "<REVISIT_GROUP_RESULTS>",
    )
    # The pinned new API takes two arguments; the helper SHA is in arguments.
    with case_imports():
        forwarded = launcher()._arguments(arguments, GROUPS[group])
    ids = forwarded[forwarded.index("--public-cases") + 1 :]
    require(
        ids == source_group["selected_case_ids_in_order"]
        and len(ids) == source_group["denominator"],
        "All original case IDs and order retained",
    )
    for flag, value in (
        ("--repeats", "1"),
        ("--max-steps", "16"),
        ("--seed", "20261014"),
        ("--model", "generated-actions-none"),
        ("--text-model", "generated-actions-none"),
        ("--api-key-env", "S1_POLICY_API_KEY"),
        ("--text-api-key-env", "S1_TEXT_API_KEY"),
    ):
        require(
            forwarded.count(flag) == 1 and forwarded[forwarded.index(flag) + 1] == value,
            "Unchanged native benchmark condition: " + flag,
        )
    selected(HELPER)
    selected(MANIFEST)
    return source_group, forwarded


def forward_arguments(group, *, policy, helper, results):
    """Build concrete arguments only; no capacity, service or launch permission."""
    module(OBSERVER).endpoints(policy, helper)
    results = ordinary(results)
    _group, arguments = source_arguments(group)
    replace_one(arguments, "--endpoint", "<FIELD_SYNTAX_POLICY_ENDPOINT>", policy)
    replace_one(arguments, "--text-endpoint", "<FIELD_SYNTAX_HELPER_BASE_ENDPOINT>", helper)
    replace_one(arguments, "--output-dir", "<REVISIT_GROUP_RESULTS>", str(results))
    with case_imports():
        return launcher()._arguments(arguments, GROUPS[group])


def prepare_source(group, output):
    """Materialize source for review; no endpoint binding or browser child."""
    source_group, arguments = source_arguments(group)
    output = ordinary(output)
    require(not output.exists() and output.parent == HERE, "Fresh separate adapter source output")
    output.mkdir(mode=0o700)
    with case_imports():
        entry = launcher()
        bundle, manifest = entry.prepare_bundle(
            output / "runtime-bundle", case_module=GROUPS[group]
        )
        entry.verify_bundle(bundle, manifest)
    receipt = {
        "schema_id": KIND,
        "status": "SOURCE_PREPARED_NOT_CAPACITY_SERVICE_OR_BROWSER_READY",
        "group": group,
        "profile": PROFILE,
        "case_module": GROUPS[group],
        "case_ids": source_group["selected_case_ids_in_order"],
        "denominator": source_group["denominator"],
        "arguments_template": arguments,
        "launcher_API_positional_arguments": 2,
        "bundle_manifest": {
            "path": str((bundle / "bundle-manifest.json").relative_to(ROOT)),
            "sha256": sha(read(bundle / "bundle-manifest.json")),
            "size_bytes": len(read(bundle / "bundle-manifest.json")),
        },
        "capacity_selected": False,
        "service_bound": False,
        "browser_model_HTTP_calls": 0,
        "benchmark_ready": False,
        "old_V8_Q95_admits_new_prompt": False,
    }
    blob = encode(receipt)
    with (output / "adapter-source-preparation.json").open("xb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--group", choices=tuple(GROUPS), required=True)
    parser.add_argument("--prepare-source", type=Path)
    args = parser.parse_args(argv)
    if args.prepare_source is not None:
        result = prepare_source(args.group, args.prepare_source)
    else:
        group, arguments = source_arguments(args.group)
        result = {
            "schema_id": KIND,
            "status": "DRY_SOURCE_ARGUMENTS_ONLY",
            "group": args.group,
            "case_ids": group["selected_case_ids_in_order"],
            "denominator": group["denominator"],
            "arguments_template": arguments,
            "launcher_API_positional_arguments": 2,
            "benchmark_ready": False,
            "output_created": False,
            "browser_model_HTTP_calls": 0,
        }
    print(encode(result).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
