"""Append records as work completes; never overwrite an existing run directory."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path


def write_json(path: Path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class Artifacts:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.streams = {}
        self.stack = ExitStack()

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=False)
        try:
            for name in ("requests", "predictions", "errors"):
                self.streams[name] = self.stack.enter_context(
                    (self.directory / f"{name}.jsonl").open("x", encoding="utf-8")
                )
        except BaseException:
            self.stack.close()
            raise
        return self

    def append(self, name, record):
        stream = self.streams[name]
        stream.write(json.dumps(record, allow_nan=False) + "\n")
        stream.flush()

    def manifest(self, value):
        write_json(self.directory / "run_manifest.json", value)

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)


def read_records(directory, name):
    with (Path(directory) / f"{name}.jsonl").open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
