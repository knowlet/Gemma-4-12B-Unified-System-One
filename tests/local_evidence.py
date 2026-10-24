"""Availability gate for tests that replay untracked local historical evidence.

This does not validate evidence. Existing source, hash, and closure guards must
still reject present but invalid inputs, including directories and symlinks.
"""

import os
from pathlib import Path

import pytest


def require_local_evidence(*paths: str | Path) -> None:
    """Skip only absent inputs, or fail when replay was explicitly required."""
    candidates = (Path(value) for value in paths)
    missing = [str(path) for path in candidates if not path.exists() and not path.is_symlink()]
    if not missing:
        return
    message = (
        "Missing local historical replay evidence: "
        + ", ".join(missing)
        + ". Restore these untracked local inputs to run the replay. "
        "JEV_REQUIRE_LOCAL_EVIDENCE=1 requires all inputs and fails instead of skipping."
    )
    if os.environ.get("JEV_REQUIRE_LOCAL_EVIDENCE") == "1":
        pytest.fail(message, pytrace=False)
    pytest.skip(message)
