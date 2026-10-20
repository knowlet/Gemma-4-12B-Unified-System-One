"""Compose new revisit CPU capacity with the unchanged original Q95 service check.

The old23/V8 proof remains an independent technical-service condition. It never
stands in for the new75/all68 acceptance. The original component still compares
private-config credentials with the existing environment. This wrapper does not
obtain credentials, start an actor, or claim browser task quality.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).absolute().parents[3]
PROFILE = "composed-visible-hover-entries-settled-revisit-v1"
SCHEMA = "jev_revisit_browser_owned_binding_v1"
PUBLIC_SCHEMA = "jev_revisit_browser_public_binding_state_v1"
OLD_SCHEMA = "jev-field-syntax-v8-browser-owned-binding-v1"
OLD_BASE = "artifacts/jev-runtime/public-field-syntax-v8-browser-retest-20261019-01"
CAP_BASE = "artifacts/jev-runtime/public-revisit-browser-binding-20261020-01"
PINS = {
    "technical_observer_source": (
        OLD_BASE + "/browser_observer_runner.py",
        "8e00743c7b3caa48a18692e652571f8e03b8cbb9906f7462f243e2ac062150f9",
        32298,
    ),
    "capacity_binding_source": (
        CAP_BASE + "/revisit_capacity_binding.py",
        "0ef8275bd2f4f57f50710ec70bde5ce3c809f743cf783405e8db2cbbbdeb29f7",
        15758,
    ),
    "profile_manifest": (
        "artifacts/jev-runtime/public-revisit-profile-20261020-01/profile-01/profile-manifest.json",
        "f3e65535836518581deab8487f6c20c837bcecfa03173db85f953cc02975ce88",
        12379,
    ),
    "profile_helper": (
        "scripts/jev_observation_profile_revisit.py",
        "873abc6aa2a02272fd9b54d299d683598bdb3d5c0a848cc43a356daa5a36a0ca",
        13642,
    ),
    "launcher_source": (
        "scripts/jev_browser_revisit_benchmark.py",
        "f7f0c4f708203e107563fa70656bbeb3e04afcba463a0a9cf9eb156fbd8e51b3",
        13193,
    ),
    "launcher_adapter": (
        CAP_BASE + "/launcher_adapter.py",
        "b8fc5033ea4a28d35875122565d0b817332ea2c23e2999340af8b2d1c8e72fd5",
        9708,
    ),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def parse(blob):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key")
            result[key] = value
        return result

    return json.loads(blob, object_pairs_hook=pairs, parse_constant=lambda n: require(False, n))


def expected_sources():
    return {
        "schema_id": "jev_revisit_browser_service_binding_sources_v1",
        "profile": PROFILE,
        **{
            key: {"path": path, "sha256": digest, "size_bytes": size}
            for key, (path, digest, size) in PINS.items()
        },
    }


def checked(ref):
    require(
        type(ref) is dict and set(ref) == {"path", "sha256", "size_bytes"}, "Exact file reference"
    )
    name, digest, size = ref["path"], ref["sha256"], ref["size_bytes"]
    require(
        type(name) is str
        and name
        and not Path(name).is_absolute()
        and ".." not in Path(name).parts
        and Path(name).as_posix() == name,
        "Repository-relative ordinary reference",
    )
    require(
        type(digest) is str
        and len(digest) == 64
        and all(c in "0123456789abcdef" for c in digest)
        and type(size) is int
        and 0 < size <= 2 * 1024**2,
        "Bounded SHA256 reference",
    )
    path = ROOT / name
    require(not any(p.is_symlink() for p in (path, *path.parents)), "No symlink reference")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size == size, "Pinned regular size")
        blob = stream.read(size + 1)
        after = os.fstat(stream.fileno())

    def identity(s):
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns

    require(
        identity(before) == identity(after) == identity(path.stat())
        and len(blob) == size
        and hashlib.sha256(blob).hexdigest() == digest,
        "Pinned stable source/evidence bytes",
    )
    return blob


def component(key):
    ref = expected_sources()[key]
    blob = checked(ref)
    name = "_revisit_service_" + key + "_" + ref["sha256"][:16]
    spec = importlib.util.spec_from_file_location(name, ROOT / ref["path"])
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        exec(compile(blob, str(ROOT / ref["path"]), "exec"), module.__dict__)
    except BaseException:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return module


def technical_observer():
    """Load the pinned original observer unchanged; no service check by itself."""
    return component("technical_observer_source")


def binding_shape(binding):
    require(
        type(binding) is dict
        and set(binding) == {"schema_id", "profile", "legacy_service_binding", "revisit_capacity"}
        and binding["schema_id"] == SCHEMA
        and binding["profile"] == PROFILE,
        "Exact new typed revisit browser binding",
    )
    # Copy JSON values to prevent either returned metadata or delegated code from
    # mutating the caller's selection. This does not adapt any old authority.
    return parse(canonical(binding))


def read_legacy_binding(binding):
    """Read only pinned public metadata. This is NOT a live-service admission.

    An owned launcher can use service_directory to supply the original env-only
    credentials. It must subsequently call binding_state with those variables.
    """
    binding = binding_shape(binding)
    legacy = parse(checked(binding["legacy_service_binding"]))
    require(
        type(legacy) is dict and legacy.get("schema_id") == OLD_SCHEMA, "Original typed binding"
    )
    return legacy


def binding_state(binding):
    """Return (new public state, original secret tuple), with both checks required.

    This is a sequence of fresh checks, not an atomic lease on processes/files.
    Callers must repeat it at their existing before/after observer boundaries.
    """
    binding = binding_shape(binding)
    expected = expected_sources()
    for key in PINS:
        checked(expected[key])
    selector = component("capacity_binding_source")
    capacity = selector.validate_selection(
        binding["revisit_capacity"], selector.expected_sources(), root=ROOT
    )
    require(
        capacity.get("kind") == selector.KIND
        and capacity.get("profile") == PROFILE
        and capacity.get("closed_evidence") is True
        and capacity.get("benchmark_ready") is True,
        "New closed75 evidence and all68 policy ACCEPTED required",
    )
    legacy = read_legacy_binding(binding)
    technical, secrets = technical_observer().binding_state(copy.deepcopy(legacy))
    require(
        type(secrets) is tuple
        and len(secrets) == 2
        and all(type(s) is str and s for s in secrets)
        and secrets[0] == secrets[1],
        "Original two env-only credentials required",
    )
    # Retain the complete original technical result, including its genuine
    # old v8_capacity. Do not rename it or merge it with new revisit_capacity.
    public = {
        "schema_id": PUBLIC_SCHEMA,
        "profile": PROFILE,
        "client_profile": {
            key: expected[key]
            for key in ("profile_manifest", "profile_helper", "launcher_source", "launcher_adapter")
        },
        "legacy_service_binding": binding["legacy_service_binding"],
        "legacy_q95_service": copy.deepcopy(technical),
        "revisit_capacity": copy.deepcopy(capacity),
        "task_quality_attested": False,
    }
    require(
        not any(
            secret.encode() in canonical(value) for secret in secrets for value in (binding, public)
        ),
        "Private credential must not occur in the complete new public binding/state",
    )
    require(read_legacy_binding(binding) == legacy, "Original binding stable during checks")
    for key in PINS:
        checked(expected[key])
    return public, secrets


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    print(
        json.dumps(
            {
                "schema_id": "jev_revisit_browser_service_binding_source_only_v1",
                "status": "DRY_SOURCE_ONLY_NOT_ADMITTED",
                "benchmark_ready": False,
                "expected_sources": expected_sources(),
                "model_processor_browser_service_calls": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
