"""Authenticated historical source fixtures, never runtime source substitutions."""

import ast
import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = Path("scripts/jev_decision3_public_dev22_gpu_replay.py")
REFERENCE = (
    ROOT
    / "artifacts/jev-runtime/decision3-current-source-bootstrap-20261022-01"
    / "frozen-generated-reference.txt"
)


def frozen_source_root(destination):
    """Copy the exact historical composition to a fresh test-only checkout.

    Current production sources and their rejected drift remain unchanged. The
    reference is a tracked historical byte artifact, not a regenerated fixture.
    """
    tree = ast.parse((ROOT / CORE).read_bytes())
    pins = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(name, ast.Name) and name.id == "SOURCE_PINS" for name in node.targets)
    )
    destination.mkdir()
    for label, expected in pins.items():
        source = REFERENCE if label == "src/s1/jev_generated.py" else ROOT / label
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
        target = destination / label
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for label in (
        CORE,
        Path("apps/modal/decision3_public_replay.py"),
        Path("apps/modal/decision3_browser_service.py"),
        Path("apps/modal/decision3_public_weights.py"),
        Path("scripts/jev_decision3_browser_wire.py"),
        Path("scripts/jev_decision3_browser_worker.py"),
    ):
        target = destination / label
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / label, target)
    return destination
