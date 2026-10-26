"""Dry-default acquisition of one immutable public CPU helper checkpoint.

The stream identity checks follow the previously reviewed Q8 fetcher, extended
to authenticate non-LFS files with the publisher's Git blob identity. No model
or tokenizer is imported. Execute only inside the separately admitted download
owner02; its file-size limit already accommodates the original public Q8 files.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import signal
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PROFILE = "qwen25-05b-cpu-helper-public-acquisition-v1"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
METADATA_SHA = "524ffed976ac1ba768ca111598ddd9f5c1a744d4bb615e8c78c7a1f3167c46ed"
GUARD = ROOT / "artifacts/jev-runtime/llama-cpp-owner-20261023-02/controller.py"
GUARD_SHA = "2d250d1cbe79bdd179381b766c5b4c9499801f79a61c94d5a15adc16fefffb25"
FILES = ("LICENSE", "README.md", "config.json", "generation_config.json",
         "tokenizer_config.json", "merges.txt", "vocab.json", "tokenizer.json",
         "model.safetensors")
WALL_SECONDS = 600
CHUNK = 1024**2


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def ordinary(path):
    path = Path(path).absolute()
    require(".." not in path.parts and path.is_relative_to(ROOT), "repository output only")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "no symlink output")
    return path


def metadata():
    raw = (HERE / "publisher-model-metadata.json").read_bytes()
    require(len(raw) <= CHUNK and digest(raw) == METADATA_SHA, "selected metadata differs")
    data = json.loads(raw)
    require(data["id"] == MODEL and data["sha"] == REVISION, "immutable publisher identity differs")
    rows = data["siblings"]
    require(len({row["rfilename"] for row in rows}) == len(rows), "duplicate publisher file")
    selected = {row["rfilename"]: row for row in rows if row["rfilename"] in FILES}
    require(set(selected) == set(FILES), "complete selected public package required")
    for name, row in selected.items():
        require(type(row["size"]) is int and 0 < row["size"] <= 988_097_824,
                "fixed positive publisher size")
        require(len(row["blobId"]) == 40 and all(c in "0123456789abcdef" for c in row["blobId"]),
                "publisher Git blob identity")
        if "lfs" in row:
            require(name == "model.safetensors" and row["lfs"]["size"] == row["size"]
                    and row["lfs"]["sha256"] == "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
                    "fixed public LFS identity")
    return selected


def write_new(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def fetch_one(name, row, directory, deadline, *, opener=urllib.request.urlopen, clock=time.monotonic):
    require(name in FILES and row["rfilename"] == name, "selected filename only")
    directory = ordinary(directory)
    target, partial = directory / name, directory / (name + ".partial")
    require(not target.exists() and not partial.exists(), "fresh file paths required")
    expected_size = row["size"]
    sha256 = hashlib.sha256()
    gitblob = hashlib.sha1(("blob " + str(expected_size) + "\0").encode())
    count, began = 0, clock()
    require(began < deadline, "whole acquisition deadline exceeded")
    request = urllib.request.Request(
        "https://huggingface.co/" + MODEL + "/resolve/" + REVISION + "/" + name + "?download=true",
        headers={"User-Agent": "jev-cpu-helper-acquisition/1"},
    )
    with opener(request, timeout=min(30, deadline - began)) as reply:
        require(reply.status == 200, "complete HTTP200 required")
        length = reply.headers.get("Content-Length")
        require(length is None or int(length) == expected_size, "HTTP size differs")
        with partial.open("xb") as stream:
            while True:
                require(clock() < deadline, "whole acquisition deadline exceeded")
                block = reply.read(min(CHUNK, expected_size - count + 1))
                if not block:
                    break
                count += len(block)
                require(count <= expected_size, "publisher byte limit exceeded")
                stream.write(block)
                sha256.update(block)
                gitblob.update(block)
            stream.flush()
            os.fsync(stream.fileno())
    require(count == expected_size, "incomplete public file")
    kind = "lfs_sha256" if "lfs" in row else "git_blob_sha1"
    expected = row["lfs"]["sha256"] if "lfs" in row else row["blobId"]
    actual = sha256.hexdigest() if "lfs" in row else gitblob.hexdigest()
    require(actual == expected, "publisher file identity mismatch")
    require(clock() < deadline, "late download cannot qualify")
    # Hardlink publishes without replacing a file created during the transfer.
    os.link(partial, target)
    partial.unlink()
    record = {"name": name, "size_bytes": count, "sha256": sha256.hexdigest(),
              "publisher_hash_kind": kind, "publisher_expected_hash": expected,
              "publisher_observed_hash": actual, "verified": True,
              "elapsed_seconds": clock() - began, "http_status": 200,
              "hash_scope": "exact downloaded stream before publication; no second payload read"}
    write_new(directory / (name + ".download.json"), record)
    require(clock() < deadline, "late durable evidence cannot qualify")
    return record


def load_guard():
    raw = GUARD.read_bytes()
    require(len(raw) <= CHUNK and digest(raw) == GUARD_SHA, "selected owner guard differs")
    spec = importlib.util.spec_from_file_location("_cpu_helper_acquisition_guard", GUARD)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(raw, str(GUARD), "exec"), module.__dict__)
    return module


def owned_context(guard):
    boundary, observer = guard.load_fixed("boundary"), guard.load_fixed("observer")
    identity = guard.process_identity(boundary, os.getpid())
    group_name = identity["process"]["cgroup"]
    require(group_name.endswith("/" + guard.UNIT), "selected download owner unit required")
    require(sorted(os.sched_getaffinity(0)) == [0], "selected download CPU affinity")
    group = Path("/sys/fs/cgroup") / group_name.lstrip("/")
    limits = {name: (group / name).read_text().strip()
              for name in ("memory.max", "memory.swap.max", "cpu.max")}
    require(limits["memory.max"] == str(2 * 1024**3) and limits["memory.swap.max"] == "0",
            "selected download memory/swap limits")
    quota, period = map(int, limits["cpu.max"].split())
    require(quota == period and period > 0, "selected single-core download quota")
    require(resource.getrlimit(resource.RLIMIT_FSIZE) == (guard.FILE_SIZE_MAX, guard.FILE_SIZE_MAX),
            "selected existing32GiB download file limit")
    return identity, limits, lambda: guard.host(observer, boundary.boot(), initial=False)


class Deadline:
    def __enter__(self):
        require(signal.getitimer(signal.ITIMER_REAL) == (0, 0), "fresh whole-download timer")
        self.previous = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self.expire)
        signal.setitimer(signal.ITIMER_REAL, WALL_SECONDS)

    def expire(self, *_):
        raise TimeoutError("whole public acquisition deadline exceeded")

    def __exit__(self, *_):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.previous)


def execute(output):
    require(sys.flags.no_user_site and sys.dont_write_bytecode, "requires -B -s")
    require(not {"torch", "transformers", "safetensors", "numpy"}.intersection(sys.modules),
            "stdlib-only acquisition")
    selected = metadata()
    guard = load_guard()
    identity, limits, host_gate = owned_context(guard)
    gate = host_gate()
    output = ordinary(output)
    require(not output.exists() and output.parent.is_dir(), "fresh owned output directory")
    disk = os.statvfs(output.parent)
    require(disk.f_bavail * disk.f_frsize >= sum(row["size"] for row in selected.values()) + 64 * CHUNK,
            "available disk must cover whole selected public package")
    output.mkdir(mode=0o700)
    began, records = time.monotonic(), []
    receipt = {"profile": PROFILE, "model_id": MODEL, "revision": REVISION,
               "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "publisher_metadata_sha256": METADATA_SHA, "process": identity,
               "owned_limits": limits, "host_start": gate, "files": records,
               "status": "incomplete_no_model_or_helper_qualification"}
    primary = None
    try:
        with Deadline():
            for name in FILES:
                host_gate()
                records.append(fetch_one(name, selected[name], output, began + WALL_SECONDS))
            require(metadata() == selected and digest(GUARD.read_bytes()) == GUARD_SHA,
                    "selected source/metadata changed during acquisition")
            final_identity, final_limits, _ = owned_context(guard)
            require(guard.same_identity(identity, final_identity) and final_limits == limits,
                    "selected download process/limits changed")
            receipt["host_final"] = host_gate()
            require(not {"torch", "transformers", "safetensors", "numpy"}.intersection(sys.modules),
                    "acquisition imported a model library")
            require(time.monotonic() - began < WALL_SECONDS, "late package cannot qualify")
            receipt.update(status="complete_public_package_stream_identity_verified",
                           elapsed_seconds=time.monotonic() - began,
                           model_loaded=False, helper_qualified=False, browser_score=None)
            write_new(output / "package-receipt.json", receipt)
            if time.monotonic() - began >= WALL_SECONDS:
                # Preserve evidence, but remove the success receipt's name.
                (output / "package-receipt.json").rename(output / "late-package-receipt.json")
                raise TimeoutError("package evidence completed late; no success receipt")
        return receipt
    except BaseException as exc:
        primary = exc
        if (output / "package-receipt.json").exists():
            try:
                (output / "package-receipt.json").rename(output / "failed-package-receipt.json")
            except BaseException as secondary:
                primary.add_note("Success receipt could not be demoted: " + type(secondary).__name__)
        receipt.update(status="failed_or_incomplete_no_package_claim",
                       elapsed_seconds=time.monotonic() - began,
                       error_type=type(exc).__name__,
                       error_message="Acquisition failed; secret redirect URLs and headers omitted")
        try:
            write_new(output / "download-failure.json", receipt)
        except BaseException as secondary:
            primary.add_note("Acquisition failure evidence could not be saved: " + type(secondary).__name__)
        raise RuntimeError("Public helper acquisition failed; inspect bounded local evidence") from None


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"profile": PROFILE, "status": "dry_default_no_download",
                          "model_id": MODEL, "revision": REVISION, "files": list(FILES),
                          "whole_seconds": WALL_SECONDS, "owner_required": "owner02_download_only_2GiB_CPU0_32GiBfile"}))
        return
    require(args.output is not None, "explicit fresh output required")
    result = execute(args.output)
    print(json.dumps({"profile": PROFILE, "status": result["status"], "files": len(result["files"])}))


if __name__ == "__main__":
    main()
