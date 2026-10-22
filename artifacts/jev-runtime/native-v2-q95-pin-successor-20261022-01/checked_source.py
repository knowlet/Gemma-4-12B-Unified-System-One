"""Consume exact selected ordinary Python bytes; never load timestamp bytecode."""

import hashlib
import sys
import types
from pathlib import Path

ROOT = Path(__file__).absolute().parents[3]


def module(relative, sha, name):
    path = ROOT / relative
    if not path.is_relative_to(ROOT) or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Ordinary selected composition source")
    if path.stat().st_size > 1024**2:
        raise ValueError("Bounded selected composition source")
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != sha:
        raise ValueError("Selected consumed composition bytes differ")
    selected = types.ModuleType(name)
    selected.__file__ = str(path)
    sys.modules[name] = selected
    exec(compile(blob, str(path), "exec", dont_inherit=True), selected.__dict__)
    return selected
