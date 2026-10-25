"""Fetch only checksum-pinned public tokenizer assets needed by CPU contracts.

No model weights, authentication, browser traffic, or acceptance evidence is fetched.
The captured native wire fixture must be supplied separately; never fabricate it.
"""

import hashlib
import urllib.request
from pathlib import Path

from jev_browser_collect_public_training import ASSET_SHA, CACHE

REVISION = "a66f836b56605039fe040f330180e336d19b3362"
BASE = f"https://huggingface.co/knowlet/Gemma-4-12B-Unified-System-One/resolve/{REVISION}"
MAX_BYTES = 40 * 1024 * 1024


def prepare(destination: Path = CACHE) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name, expected in ASSET_SHA.items():
        path = destination / name
        if path.exists():
            blob = path.read_bytes()
        else:
            with urllib.request.urlopen(f"{BASE}/{name}", timeout=60) as response:
                blob = response.read(MAX_BYTES + 1)
        if len(blob) > MAX_BYTES or hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError(f"Pinned tokenizer asset differs: {name}")
        if not path.exists():
            path.write_bytes(blob)
        print(f"Verified {name}: {expected}")


if __name__ == "__main__":
    prepare()
