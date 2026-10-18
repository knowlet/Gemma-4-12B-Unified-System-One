"""Local bounded parallel, read-only receive of one externally pinned closed export.

OUTPUT/tree contains the unchanged remote index and every indexed file.
OUTPUT/transfer-receipt.json is separate from that exact received tree.
Modal is imported only after explicit --run; no image, GPU or upload is used.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import stat
import time
from contextlib import aclosing
from pathlib import Path, PurePosixPath

VOLUME = "jev-public07-fit-results-01"
MAX_FILES, MAX_BYTES, MAX_INDEX = 10000, 1024**3, 8 * 1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def ordinary(path):
    path = Path(path).absolute()
    require(
        ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
        "Ordinary ancestors required",
    )
    return path


def relative(name):
    require(type(name) is str and 0 < len(name) <= 4096, "Bounded relative filename required")
    path = PurePosixPath(name)
    require(
        not path.is_absolute()
        and str(path) == name
        and name != "."
        and ".." not in path.parts
        and "\\" not in name
        and not any(ord(c) < 32 or ord(c) == 127 for c in name),
        "Confined relative filename required",
    )
    return path


def valid_pin(value, limit=MAX_BYTES):
    require(
        type(value) is dict
        and set(value) == {"sha256", "size_bytes"}
        and type(value["sha256"]) is str
        and re.fullmatch(r"[0-9a-f]{64}", value["sha256"])
        and type(value["size_bytes"]) is int
        and 0 <= value["size_bytes"] <= limit,
        "Exact bounded SHA/size pin required",
    )


def parse(blob):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate JSON key refused")
            result[key] = value
        return result

    return json.loads(
        blob.decode("utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON refused")),
    )


def inventory(tree):
    files, directories, total = {}, set(), 0
    for parent, children, names in os.walk(tree, followlinks=False):
        ordinary(parent)
        for name in children:
            path = ordinary(Path(parent) / name)
            directories.add(path.relative_to(tree).as_posix())
            require(len(directories) <= MAX_FILES, "Received directory count cap")
        for name in names:
            path, digest, count = ordinary(Path(parent) / name), hashlib.sha256(), 0
            require(stat.S_ISREG(path.stat().st_mode), "Only ordinary received files allowed")
            total += path.stat().st_size
            require(total <= MAX_BYTES, "Received total byte cap")
            with path.open("rb") as stream:
                while chunk := stream.read(1024**2):
                    count += len(chunk)
                    require(count <= MAX_BYTES, "Received file byte cap")
                    digest.update(chunk)
            files[path.relative_to(tree).as_posix()] = {
                "sha256": digest.hexdigest(),
                "size_bytes": count,
            }
            require(len(files) <= MAX_FILES, "Received file count cap")
    return files, directories


def error_info(exc):
    result = {"type": type(exc).__name__, "message": str(exc)}
    if isinstance(exc, BaseExceptionGroup):
        result["errors"] = [error_info(item) for item in exc.exceptions]
    return result


async def receive(volume, *, remote_root, index_sha256, index_size_bytes, output, concurrency=8):
    require(concurrency in (8, 16), "Concurrency must be8 or16")
    require(len(relative(remote_root).parts) == 1, "One remote export root required")
    index_pin = {"sha256": index_sha256, "size_bytes": index_size_bytes}
    valid_pin(index_pin, MAX_INDEX)
    output = ordinary(output)
    require(not output.exists(), "Fresh output directory required; no resume or overwrite")
    output.mkdir(parents=True)
    tree = output / "tree"
    tree.mkdir()
    started, semaphore = time.monotonic(), asyncio.Semaphore(concurrency)
    receipt = {
        "status": "failed",
        "volume": VOLUME,
        "remote_root": remote_root,
        "index_pin": index_pin,
        "concurrency": concurrency,
        "completed_files": 0,
        "completed_bytes": 0,
        "stage": "index",
        "scope": "Read-only local transfer and full byte/population authentication",
    }

    async def transfer(name, expected):
        async with semaphore:
            path = ordinary(tree / relative(name))
            path.parent.mkdir(parents=True, exist_ok=True)
            ordinary(path)
            count, digest = 0, hashlib.sha256()
            # SDK1.6 Volume.read_file.aio returns an async iterator, not an
            # awaitable bytes result. Closing it also closes SDK block prefetch.
            with path.open("xb") as stream:
                async with aclosing(volume.read_file.aio(remote_root + "/" + name)) as chunks:
                    async for chunk in chunks:
                        require(type(chunk) is bytes, "Volume stream must yield bytes")
                        count += len(chunk)
                        require(
                            count <= expected["size_bytes"], "Stream exceeded pinned size: " + name
                        )
                        require(stream.write(chunk) == len(chunk), "Short local write")
                        digest.update(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            require(
                {"sha256": digest.hexdigest(), "size_bytes": count} == expected,
                "Stream SHA/size differs: " + name,
            )
            receipt["completed_files"] += 1
            receipt["completed_bytes"] += count

    primary = None
    try:
        await transfer("export-index.json", index_pin)
        index = parse((tree / "export-index.json").read_bytes())
        require(
            type(index) is dict
            and index.get("root") == remote_root
            and index.get("complete_owned_exit") is True
            and type(index.get("files")) is dict,
            "Matching closed owned export index required",
        )
        files = index["files"]
        require(len(files) + 1 <= MAX_FILES, "Whole export file cap")
        paths = {name: relative(name) for name in files}
        require("export-index.json" not in files, "Index cannot overwrite itself")
        for name, expected in files.items():
            valid_pin(expected)
            require(
                not any(parent.as_posix() in files for parent in paths[name].parents),
                "File/directory prefix collision",
            )
        require(
            sum(p["size_bytes"] for p in files.values()) + index_size_bytes <= MAX_BYTES,
            "Whole export byte cap",
        )
        receipt.update(
            stage="files",
            expected_files=len(files) + 1,
            expected_bytes=sum(p["size_bytes"] for p in files.values()) + index_size_bytes,
        )
        # TaskGroup cancels and awaits every owned transfer on any failure or
        # cancellation, including closing each active SDK asynchronous iterator.
        async with asyncio.TaskGroup() as group:
            for name, expected in files.items():
                group.create_task(transfer(name, expected))
        receipt["stage"] = "verification"
        expected_files = {**files, "export-index.json": index_pin}
        expected_dirs = {
            parent.as_posix()
            for path in paths.values()
            for parent in path.parents
            if parent.as_posix() != "."
        }
        require(
            inventory(tree) == (expected_files, expected_dirs),
            "Final received SHA/size/population differs",
        )
        require(set(output.iterdir()) == {tree}, "Unknown output entries refused")
        receipt.update(status="complete", stage="complete", complete_owned_exit=True)
    except BaseException as exc:
        primary = exc
        receipt["error"] = error_info(exc)
    receipt["elapsed_seconds"] = time.monotonic() - started
    path = ordinary(output / "transfer-receipt.json")
    try:
        with path.open("xb") as stream:
            stream.write((json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException as secondary:
        if primary is not None:
            primary.add_note("Transfer receipt persistence also failed: " + repr(secondary))
            raise primary from secondary
        raise
    if primary is not None:
        raise primary
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--volume", choices=[VOLUME], required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--index-sha256", required=True)
    parser.add_argument("--index-size-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, choices=[8, 16], default=8)
    args = parser.parse_args(argv)
    import modal

    volume = modal.Volume.from_name(args.volume, create_if_missing=False)
    result = asyncio.run(
        receive(
            volume,
            remote_root=args.remote_root,
            index_sha256=args.index_sha256,
            index_size_bytes=args.index_size_bytes,
            output=args.output,
            concurrency=args.concurrency,
        )
    )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
