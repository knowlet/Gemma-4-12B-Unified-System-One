"""Recover terminal breakthrough stage evidence with bounded local CPU I/O.

Only the named existing Modal Volume is read. No App, Function, GPU, mount,
commit, upload or remote mutation is created. Files are downloaded losslessly
with exclusive local writes; a separate receipt records the original export
failure and every downloaded SHA256. The original GPU receipt is unchanged.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

VOLUME = "gemma-unified-system-one"
METADATA = ".evidence-download"
MAX_FILE_BYTES = 128 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
MAX_FILES = 50000
WEIGHTS = {".pt", ".pth", ".bin", ".safetensors", ".gguf"}
RESEARCH = {
    "decision-head.pt",
    "train-features.pt",
    "adapter_model.safetensors",
    "adapter.safetensors",
}


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def _safe(name):
    if (
        not isinstance(name, str)
        or not name
        or name.startswith("/")
        or "\\" in name
        or ":" in name
        or any(ord(c) < 32 for c in name)
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise ValueError("unsafe evidence path")
    return name


def _stage(run, stage):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", run) or stage not in {"ablate", "train"}:
        raise ValueError("existing lowercase run and ablate/train stage required")
    return f"breakthrough/{run}/{stage}"


def _plan(entries, prefix, stage):
    files, seen = {}, set()
    for entry in entries:
        name = _safe(entry.path.removeprefix("/"))
        if name in seen:
            raise ValueError("duplicate remote listing path")
        seen.add(name)
        if name == prefix and int(entry.type) == 2:
            continue
        if not name.startswith(prefix + "/"):
            raise ValueError("remote listing escapes requested stage")
        relative = _safe(name.removeprefix(prefix + "/"))
        if METADATA in relative.split("/"):
            raise ValueError("remote path collides with transfer metadata")
        if int(entry.type) == 2:
            continue
        if int(entry.type) != 1:
            raise ValueError("remote symlink or nonregular file rejected")
        if type(entry.size) is not int or not 0 <= entry.size <= MAX_FILE_BYTES:
            raise ValueError("remote member size exceeds guard")
        path = Path(relative)
        if path.suffix.lower() in WEIGHTS and not (stage == "train" and path.name in RESEARCH):
            raise ValueError("baseline or unrecognized weights rejected")
        files[relative] = {"remote_path": name, "size_bytes": entry.size, "mtime": entry.mtime}
    if (
        "receipt.json" not in files
        or len(files) > MAX_FILES
        or sum(r["size_bytes"] for r in files.values()) > MAX_TOTAL_BYTES
    ):
        raise ValueError("missing receipt or evidence population/size exceeds guard")
    return dict(sorted(files.items()))


async def _read(volume, remote_path, expected_size):
    raw = bytearray()
    async for chunk in volume.read_file.aio(remote_path):
        raw.extend(chunk)
        if len(raw) > expected_size:
            raise ValueError("download exceeds listed size")
    if len(raw) != expected_size:
        raise ValueError("download differs from listed size")
    return bytes(raw)


def _write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _terminal(raw, run, stage):
    receipt = json.loads(raw)
    if (
        receipt.get("run") != run
        or receipt.get("stage") != stage
        or receipt.get("status") not in {"completed", "failed"}
    ):
        raise ValueError("remote GPU receipt must match a terminal requested stage")
    return receipt


def _sources(output, receipt):
    declared = receipt.get("source_files", {})
    if not isinstance(declared, dict) or not declared:
        raise ValueError("remote receipt lacks executed source hashes")
    actual = {
        p.relative_to(output / "sources").as_posix()
        for p in (output / "sources").rglob("*")
        if p.is_file()
    }
    if actual != {f"{_safe(name)}.txt" for name in declared}:
        raise ValueError("executed source population differs from GPU receipt")
    for name, digest in declared.items():
        if sha256((output / "sources" / (name + ".txt")).read_bytes()) != digest:
            raise ValueError("executed source hash differs from GPU receipt")
    return len(declared)


async def download(
    volume,
    run,
    stage,
    output,
    *,
    concurrency=16,
    file_timeout=120,
    attempts=3,
    allow_empty_output=False,
    original_failure=None,
):
    """Read one terminal stage; injected public Volume APIs permit CPU tests."""
    prefix = _stage(run, stage)
    if (
        type(concurrency) is not int
        or not 1 <= concurrency <= 32
        or type(attempts) is not int
        or not 1 <= attempts <= 5
        or type(file_timeout) is not int
        or not 1 <= file_timeout <= 600
    ):
        raise ValueError("bounded positive concurrency/attempts/timeout required")
    output = Path(output)
    if any(parent.is_symlink() for parent in (output, *output.parents)):
        raise ValueError("local output must not traverse a symlink")
    if output.exists() and not (
        allow_empty_output and output.is_dir() and not any(output.iterdir())
    ):
        raise FileExistsError("output exists and is not an explicitly allowed empty directory")
    plan = _plan(await volume.listdir.aio(prefix, recursive=True), prefix, stage)
    initial = await asyncio.wait_for(
        _read(volume, prefix + "/receipt.json", plan["receipt.json"]["size_bytes"]), file_timeout
    )
    receipt = _terminal(initial, run, stage)
    failure_raw = None
    if original_failure is not None:
        original_failure = Path(original_failure)
        if original_failure.is_symlink() or not original_failure.is_file():
            raise ValueError("original export failure must be a regular file")
        failure_raw = original_failure.read_bytes()
    output.mkdir(parents=True, exist_ok=allow_empty_output)
    metadata = output / METADATA
    metadata.mkdir(exist_ok=False)
    source_raw = Path(__file__).read_bytes()
    (metadata / "export_breakthrough_evidence.py.txt").write_bytes(source_raw)
    _write_json(metadata / "remote-listing.json", plan)
    if failure_raw is not None:
        (metadata / "original-export-failure.txt").write_bytes(failure_raw)
    metadata_members = [
        {"path": path.name, "sha256": sha256(path.read_bytes()), "size_bytes": path.stat().st_size}
        for path in sorted(metadata.iterdir())
    ]
    state = {
        "schema_version": 1,
        "status": "running",
        "volume": VOLUME,
        "run": run,
        "stage": stage,
        "remote_prefix": prefix,
        "gpu_receipt_status": receipt["status"],
        "gpu_receipt_sha256": sha256(initial),
        "source_sha256": sha256(source_raw),
        "settings": {
            "concurrency": concurrency,
            "file_timeout_seconds": file_timeout,
            "attempts": attempts,
            "remote_mutations": False,
            "gpu_calls": 0,
        },
        "expected_files": len(plan),
        "expected_size_bytes": sum(r["size_bytes"] for r in plan.values()),
        "original_export_failure": {"sha256": sha256(failure_raw), "size_bytes": len(failure_raw)}
        if failure_raw is not None
        else {"status": "not_supplied"},
        "members": [],
        "metadata_members": metadata_members,
        "errors": [],
    }
    _write_json(metadata / "receipt.json", state)
    semaphore = asyncio.Semaphore(concurrency)

    async def member(name, spec):
        async with semaphore:
            target = output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(1, attempts + 1):
                temporary = None
                try:

                    async def transfer():
                        nonlocal temporary
                        handle, temporary = tempfile.mkstemp(prefix=".download-", dir=target.parent)
                        digest, size = hashlib.sha256(), 0
                        with os.fdopen(handle, "wb") as file:
                            async for chunk in volume.read_file.aio(spec["remote_path"]):
                                size += len(chunk)
                                if size > spec["size_bytes"]:
                                    raise ValueError("download exceeds listed size")
                                digest.update(chunk)
                                file.write(chunk)
                        if size != spec["size_bytes"]:
                            raise ValueError("download differs from listed size")
                        os.link(temporary, target)
                        return {
                            "path": name,
                            "sha256": digest.hexdigest(),
                            "size_bytes": size,
                            "attempts": attempt,
                        }

                    record = await asyncio.wait_for(transfer(), file_timeout)
                    state["members"].append(record)
                    if len(state["members"]) % 200 == 0:
                        print(
                            json.dumps(
                                {
                                    "downloaded_files": len(state["members"]),
                                    "expected_files": len(plan),
                                }
                            ),
                            flush=True,
                        )
                    return
                except Exception as exc:
                    if isinstance(exc, FileExistsError) or attempt == attempts:
                        state["errors"].append(
                            {"path": name, "error_type": type(exc).__name__, "attempts": attempt}
                        )
                        return
                    await asyncio.sleep(min(attempt, 2))
                finally:
                    if temporary is not None:
                        Path(temporary).unlink(missing_ok=True)

    try:
        await asyncio.gather(*(member(name, spec) for name, spec in plan.items()))
        if state["errors"]:
            raise ValueError("download incomplete; original/partial evidence preserved")
        after = _plan(await volume.listdir.aio(prefix, recursive=True), prefix, stage)
        if (
            after != plan
            or await asyncio.wait_for(
                _read(volume, prefix + "/receipt.json", plan["receipt.json"]["size_bytes"]),
                file_timeout,
            )
            != initial
        ):
            raise ValueError("remote evidence changed during download")
        state["executed_sources_verified"] = _sources(output, receipt)
        if (output / "receipt.json").read_bytes() != initial:
            raise ValueError("downloaded GPU receipt differs from original")
        _verify_payload(output, state)
        state["status"] = "completed"
        state["remote_consistency"] = (
            "Pre/post listing sizes/mtimes and exact GPU receipt unchanged; downloaded SHA256s, not remote-provided file checksums"
        )
    except BaseException as exc:
        state["status"], state["error_type"] = "failed", type(exc).__name__
        raise
    finally:
        state["members"].sort(key=lambda r: r["path"])
        _write_json(metadata / "receipt.json", state)
    return verify_download(output)


def verify_download(output):
    output = Path(output)
    state = json.loads((output / METADATA / "receipt.json").read_text())
    if (
        state["status"] != "completed"
        or state["errors"]
        or len(state["members"]) != state["expected_files"]
    ):
        raise ValueError("download receipt is incomplete")
    return _verify_payload(output, state)


def _verify_payload(output, state):
    if state["errors"] or len(state["members"]) != state["expected_files"]:
        raise ValueError("download receipt is incomplete")
    expected = {_safe(record["path"]): record for record in state["members"]}
    if len(expected) != len(state["members"]):
        raise ValueError("duplicate download manifest member")
    actual = set()
    for path in output.rglob("*"):
        if path.is_symlink():
            raise ValueError("download must not contain symlinks")
        if path.is_file() and METADATA not in path.relative_to(output).parts:
            actual.add(path.relative_to(output).as_posix())
    if actual != set(expected):
        raise ValueError("downloaded member population differs")
    for name, record in expected.items():
        raw = (output / name).read_bytes()
        if len(raw) != record["size_bytes"] or sha256(raw) != record["sha256"]:
            raise ValueError("downloaded member hash/size differs")
    if (
        sha256((output / METADATA / "export_breakthrough_evidence.py.txt").read_bytes())
        != state["source_sha256"]
    ):
        raise ValueError("download helper source hash differs")
    for record in state["metadata_members"]:
        raw = (output / METADATA / _safe(record["path"])).read_bytes()
        if len(raw) != record["size_bytes"] or sha256(raw) != record["sha256"]:
            raise ValueError("download metadata hash/size differs")
    raw = (output / "receipt.json").read_bytes()
    if sha256(raw) != state["gpu_receipt_sha256"]:
        raise ValueError("original GPU receipt hash differs")
    receipt = _terminal(raw, state["run"], state["stage"])
    _sources(output, receipt)
    return {
        "verified": True,
        "files": len(expected),
        "size_bytes": sum(r["size_bytes"] for r in expected.values()),
        "gpu_receipt_status": receipt["status"],
        "download_status": state["status"],
        "executed_sources_verified": state["executed_sources_verified"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run")
    parser.add_argument("--stage", choices=("ablate", "train"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--file-timeout", type=int, default=120)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--allow-empty-output", action="store_true")
    parser.add_argument("--original-export-failure", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        result = verify_download(args.verify)
    else:
        if not args.run or not args.stage or args.output is None:
            parser.error("--run, --stage and --output are required for download")
        import modal

        volume = modal.Volume.from_name(VOLUME)
        result = asyncio.run(
            download(
                volume,
                args.run,
                args.stage,
                args.output,
                concurrency=args.concurrency,
                file_timeout=args.file_timeout,
                attempts=args.attempts,
                allow_empty_output=args.allow_empty_output,
                original_failure=args.original_export_failure,
            )
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
