"""Content identity for local decision adapters and trained classifiers."""

import hashlib
import json
from pathlib import Path


def directory_digest(directory):
    root = Path(directory)
    files = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
    if not files:
        raise ValueError("checkpoint directory is empty")
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
