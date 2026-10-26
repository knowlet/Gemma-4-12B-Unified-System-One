#!/usr/bin/env python3
"""Storage-only pinned public checkpoint transfer; never load a model/framework.

Uses the installed official HF client with Xet/concurrency disabled, then hashes
each file in bounded blocks. Future original model/cache/header admission is
still required. This program does not issue a model readiness/quality result.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import resource
import signal
import sys
import threading
import time
from pathlib import Path

ROOT = Path("/home/albert_hsieh/Gemma-4-12B-Unified-System-One")
HERE = Path(__file__).absolute().parent
HUB = ROOT / "artifacts/jev-local-hf/hub"
METADATA = ROOT / "artifacts/jev-d3-native/20261015-native-source-01"
IDENTITIES = ROOT / "src/s1/evaluation/decision3.py"
IDENTITIES_SHA = "aaa21ab6f2ecf5ae0e00694b43755541dc75fc0310ba10d945ade2a4511add52"
MODELS = ("d3-nano", "d3-mini", "d3-flash", "d3")
ADDRESS_SPACE = 512 * 1024**2
DISK_RESERVE = 32 * 1024**3
WALL_SECONDS = 10800
CORE_SOURCE = ROOT / "scripts/jev_decision3_public_dev22_gpu_replay.py"
CORE_SHA = "9919e3b1dd5d5a68377b8e74c1b313c9a368a1716400d32af72e8a548ef02b4c"
CORE = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for blob in iter(lambda: stream.read(1024**2), b""):
            digest.update(blob)
    return digest.hexdigest()


def selection(name):
    require(sha(IDENTITIES) == IDENTITIES_SHA, "Frozen six-model identities differ")
    tree = ast.parse(IDENTITIES.read_bytes())
    identities = next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "CHECKPOINTS" for t in n.targets)
    )
    identity = identities[name]
    path = METADATA / name / "MODEL_MANIFEST.json"
    require(sha(path) == identity["manifest_sha256"], "Official pinned manifest differs")
    manifest = json.loads(path.read_bytes())
    require(
        manifest["repo_id"] == identity["model_id"] and manifest["model_name"] == name,
        "Publisher manifest identity differs",
    )
    files = {
        k: {"size_bytes": v, "sha256": manifest["files_sha256"][k]}
        for k, v in manifest["files_bytes"].items()
        if k not in {"LICENSE", "NOTICE", "requirements.txt", ".gitattributes"}
        and not k.endswith(".md")
        and not k.startswith(("assets/", "LICENSES/"))
    }
    files["MODEL_MANIFEST.json"] = {
        "size_bytes": path.stat().st_size,
        "sha256": identity["manifest_sha256"],
    }
    require(
        all(Path(k).name == k and k not in {".", ".."} for k in files), "Flat pinned runtime only"
    )
    return identity, files


def gate():
    global CORE
    if CORE is None:
        require(sha(CORE_SOURCE) == CORE_SHA, "Pinned stdlib resource observer changed")
        spec = importlib.util.spec_from_file_location("storage_transfer_resource_only", CORE_SOURCE)
        CORE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(CORE)
    snapshot = CORE.resource_snapshot()
    values = {
        k: int(v.split()[0]) * 1024
        for k, v in (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        if k in {"MemAvailable", "MemTotal"}
    }
    floor = max(24 * 1024**3, (values["MemTotal"] + 4) // 5) + ADDRESS_SPACE + 128 * 1024**2
    groups = snapshot["cgroup_memory"]
    require(
        isinstance(groups, list) and bool(groups), "Inherited memory cgroup observation required"
    )
    available = values["MemAvailable"]
    for group in groups:
        require(type(group["current"]) is int and group["current"] >= 0, "Typed cgroup current")
        if group["limit"] is not None:
            require(type(group["limit"]) is int and group["limit"] > 0, "Typed cgroup limit")
            available = min(available, max(0, group["limit"] - group["current"]))
    require(available >= floor, "Storage-only host/cgroup reserve refused")
    fs = os.statvfs(HUB)
    require(fs.f_bavail * fs.f_frsize >= DISK_RESERVE, "Storage-only disk reserve refused")
    return {
        "available_bytes": available,
        "required_bytes": floor,
        "disk_available_bytes": fs.f_bavail * fs.f_frsize,
    }


def persist(path, value):
    blob = (json.dumps(value, indent=2) + "\n").encode()
    require(len(blob) <= 512 * 1024, "Small transfer receipt only")
    temporary = path.with_suffix(".pending")
    with temporary.open("wb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    identity, files = selection(args.model)
    selected_bytes = sum(row["size_bytes"] for row in files.values())
    if not args.run:
        print(
            json.dumps(
                {
                    "model": args.model,
                    "files": len(files),
                    "bytes": selected_bytes,
                    "runtime_qualified": False,
                    "model_loaded": False,
                }
            )
        )
        return 0
    for path in (HUB, HERE, METADATA):
        require(not any(p.is_symlink() for p in (path, *path.parents)), "Ordinary selected roots")
    require(not any(n in sys.modules for n in ("torch", "transformers", "numpy")), "No frameworks")
    initial = gate()
    require(
        initial["disk_available_bytes"] >= selected_bytes + DISK_RESERVE,
        "Whole model plus disk reserve",
    )
    receipt = HERE / (args.model + "-transfer-01.json")
    require(not receipt.exists(), "Fresh transfer receipt required")
    resource.setrlimit(resource.RLIMIT_AS, (ADDRESS_SPACE, ADDRESS_SPACE))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    state = {
        "schema_id": "decision3-public-checkpoint-storage-transfer-v1",
        "model": args.model,
        "identity": identity,
        "initial": initial,
        "expected_files": len(files),
        "expected_bytes": selected_bytes,
        "files": {},
        "model_loaded": False,
        "runtime_qualified": False,
        "error": None,
        "started_unix": time.time(),
        "pid": os.getpid(),
        "pgid": os.getpgid(0),
        "sid": os.getsid(0),
        "address_space_limit_bytes": ADDRESS_SPACE,
        "disk_reserve_bytes": DISK_RESERVE,
    }
    stopped = threading.Event()
    failure = []

    def interrupted(_number, _frame):
        raise RuntimeError(failure[0] if failure else "Owned transfer interrupted")

    signal.signal(signal.SIGTERM, interrupted)

    def monitor():
        while not stopped.wait(1):
            try:
                gate()
                require(time.time() - state["started_unix"] < WALL_SECONDS, "Transfer wall limit")
            except BaseException as exc:
                failure.append(str(exc))
                os.kill(os.getpid(), signal.SIGTERM)
                return

    os.environ.update(
        {
            "HF_HUB_DISABLE_XET": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_TOKEN_PATH": "/nonexistent-d3-public-token",
            "HF_HUB_OFFLINE": "0",
            "HF_HUB_DOWNLOAD_TIMEOUT": "30",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        }
    )
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    primary = None
    try:
        persist(receipt, state)
        from huggingface_hub import hf_hub_download

        monitor_thread.start()
        for label, pin in sorted(files.items()):
            gate()
            print(
                json.dumps(
                    {
                        "event": "fetch",
                        "model": args.model,
                        "file": label,
                        "bytes": pin["size_bytes"],
                    }
                ),
                flush=True,
            )
            path = Path(
                hf_hub_download(
                    identity["model_id"],
                    label,
                    revision=identity["revision"],
                    cache_dir=str(HUB),
                    token=False,
                    endpoint="https://huggingface.co",
                )
            )
            require(
                path.stat().st_size == pin["size_bytes"] and sha(path) == pin["sha256"],
                "Transferred complete payload SHA/size differs: " + label,
            )
            state["files"][label] = {**pin, "path": str(path)}
            persist(receipt, state)
        require(
            not any(n in sys.modules for n in ("torch", "transformers", "numpy")),
            "Framework imported during transfer",
        )
        state["status"] = "COMPLETE_TRANSFER_HASHED_NOT_MODEL_QUALIFIED"
    except BaseException as exc:
        primary = exc
        state["status"] = "PARTIAL_TRANSFER_UNQUALIFIED"
        state["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        stopped.set()
        if monitor_thread.is_alive():
            monitor_thread.join(timeout=2)
        state["finished_unix"] = time.time()
        try:
            persist(receipt, state)
        except BaseException as secondary:
            if primary is None:
                primary = secondary
            else:
                primary.add_note("Transfer receipt also failed: " + repr(secondary))
    if primary is not None:
        raise primary
    print(
        json.dumps({"status": state["status"], "model": args.model, "files": len(state["files"])})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
