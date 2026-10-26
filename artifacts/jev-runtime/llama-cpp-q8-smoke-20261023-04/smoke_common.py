"""Small stdlib identity and payload checks shared by the two smoke stages."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).absolute().parent
ROOT = HERE.parents[2]
PROFILE = "llama-cpp-q8-smoke-selection-v1"
REVISION = "a66f836b56605039fe040f330180e336d19b3362"
PROCESSOR = (
    ROOT
    / "artifacts/jev-local-hf/hub/models--knowlet--Gemma-4-12B-Unified-System-One/snapshots"
    / REVISION
)
MAX_JSON = 1024**2
THRESHOLD = 0.001
SCORING_SECONDS = 600
MODEL_SHA = "9d3bc61851e5b7ed8e595b84327b85271830778d0e62fa9c1349c61843f369ed"
MMPROJ_SHA = "cffbff05efd290fef28d084ebcb653de7594f5e26bae0eabbedf30d6c782c1d1"
STAT_KEYS = ("st_dev", "st_ino", "st_size", "st_mtime_ns")


def need(value, message):
    if not value:
        raise ValueError(message)


def ordinary(path):
    path = Path(path).absolute()
    need(
        path.is_relative_to(ROOT)
        and ".." not in path.parts
        and not any(p.is_symlink() for p in (path, *path.parents)),
        "Ordinary owned repository path",
    )
    return path


def read(path):
    with ordinary(path).open("rb") as stream:
        data = stream.read(MAX_JSON + 1)
    need(len(data) <= MAX_JSON, "Bounded metadata/payload")
    return data


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (
        json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode() + b"\n"
    )


def save(path, value):
    blob = encoded(value)
    need(len(blob) <= MAX_JSON, "Bounded smoke result")
    path = ordinary(path)
    pending = path.with_name(".pending-" + path.name)
    with pending.open("xb") as stream:
        stream.write(blob)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(pending, path)
    pending.unlink()


def ref(path):
    path = ordinary(path)
    data = read(path)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(data), "size_bytes": len(data)}


def check_ref(row):
    need(
        type(row) is dict
        and set(row) == {"path", "sha256", "size_bytes"}
        and not Path(row["path"]).is_absolute(),
        "Exact relative small source reference",
    )
    need(ref(ROOT / row["path"]) == row, "Selected small source unchanged")
    return row


def file_stat(path):
    value = Path(path).stat()
    return {key: getattr(value, key) for key in STAT_KEYS}


def check_stat(row):
    need(
        type(row) is dict and set(row) >= {"path", "stat"} and set(row["stat"]) == set(STAT_KEYS),
        "Explicit selected stat identity",
    )
    path = ordinary(row["path"])
    need(path.is_file() and file_stat(path) == row["stat"], "Selected file stat unchanged")
    return row


def selection(path, digest, entry, *, runtime=False):
    blob = read(path)
    need(sha(blob) == digest, "Explicit selected smoke metadata SHA")
    value = json.loads(blob)
    need(
        set(value)
        == {
            "profile",
            "sources",
            "processor",
            "native",
            "package",
            "research_scoring_deadline_seconds",
        }
        and value["profile"] == PROFILE
        and type(value["research_scoring_deadline_seconds"]) is int
        and value["research_scoring_deadline_seconds"] == SCORING_SECONDS,
        "Explicit research600s scoring selection, not browser20s qualification",
    )
    need(
        type(value["sources"]) is list and 1 <= len(value["sources"]) <= 40,
        "Bounded critical small source set",
    )
    refs = [check_ref(row) for row in value["sources"]]
    required = [
        entry,
        HERE / "smoke_common.py",
        HERE / "prepare_smoke.py",
        HERE / "run_smoke.py",
        ROOT / "scripts/gguf_adapter.py",
        *(
            ROOT / ("src/s1/" + name)
            for name in (
                "__init__.py",
                "unified.py",
                "contracts.py",
                "schema.py",
                "checkpoint.py",
                "runtime.py",
                "errors.py",
            )
        ),
    ]
    need(
        len({row["path"] for row in refs}) == len(refs)
        and all(ref(path) in refs for path in required),
        "Entry/shared source in distinct selected source pool",
    )
    need(
        set(value["processor"]) == {"path", "artifact_manifest"}
        and value["processor"]["path"] == str(PROCESSOR)
        and any(
            row["path"] == value["processor"]["artifact_manifest"]
            and row["sha256"] == "1646a81f062fa3fe5727c835de569b669f288bd646749b2c3b96426bff4bf8f0"
            for row in refs
        ),
        "Same a66 processor and selected public identity manifest",
    )
    if runtime:
        native, package = value["native"], value["package"]
        need(
            set(native) == {"path", "sha256", "size_bytes", "stat", "libraries"},
            "Native executable identity",
        )
        check_stat(native)
        need(
            native["stat"]["st_size"] == native["size_bytes"] <= 64 * 1024**2,
            "Bounded native code executable",
        )
        h = hashlib.sha256()
        with ordinary(native["path"]).open("rb") as stream:
            while block := stream.read(1024**2):
                h.update(block)
        need(h.hexdigest() == native["sha256"], "Selected native executable code SHA")
        need(
            type(native["libraries"]) is list and len(native["libraries"]) <= 32,
            "Bounded selected native library stats",
        )
        for row in native["libraries"]:
            check_stat(row)
        need(
            set(package) == {"receipt", "model", "mmproj"}
            and check_ref(package["receipt"]) in refs,
            "Selected downloaded package receipt",
        )
        receipt = json.loads(read(ROOT / package["receipt"]["path"]))
        need(
            receipt["status"] == "PASS"
            and receipt["mode"] == "full native model/projector"
            and receipt["Torch_imported"] is False,
            "Completed public download only",
        )
        downloaded = {row["name"]: row for row in receipt["files"]}
        need(
            set(downloaded) == {"s1-boolq-Q8_0.gguf", "mmproj-s1-boolq-f16.gguf"},
            "Exactly two published payload receipts",
        )
        for key, name, digest, size in (
            ("model", "s1-boolq-Q8_0.gguf", MODEL_SHA, 12669647264),
            ("mmproj", "mmproj-s1-boolq-f16.gguf", MMPROJ_SHA, 122031744),
        ):
            row = package[key]
            need(
                set(row) == {"path", "sha256", "stat"}
                and Path(row["path"]).name == name
                and Path(row["path"]).parent == (ROOT / package["receipt"]["path"]).parent
                and row["sha256"] == digest
                and row["stat"]["st_size"] == size,
                "Exact published Q8/projector identity",
            )
            check_stat(row)  # Payload content is not reopened or hashed here.
            need(
                downloaded[name]["verified"] is True
                and downloaded[name]["size_bytes"] == size
                and downloaded[name]["sha256"] == downloaded[name]["expected_sha256"] == digest,
                "Public stream receipt joins selected payload stat; no second payload read",
            )
    return value


def payload_check(value, *, long):
    need(
        set(value)
        == {
            "case_id",
            "question_ids",
            "slots",
            "nopts",
            "letters",
            "source_input_token_count",
            "parts",
        },
        "Exact text-only native payload",
    )
    count = value["source_input_token_count"]
    ids = [token for part in value["parts"] for token in part.get("tokens", [])]
    need(
        type(count) is int
        and 0 < count <= 2048
        and count == len(ids)
        and all(set(p) == {"type", "tokens"} and p["type"] == "text" for p in value["parts"])
        and all(type(n) is int and n >= 0 for n in ids),
        "Exact bounded text input count/tokens",
    )
    names = ["synthetic-early", "synthetic-late"] if long else ["synthetic-warm"]
    need(
        value["case_id"] == ("synthetic-cache-long-v1" if long else "synthetic-cache-warm-v1")
        and value["question_ids"] == names
        and value["nopts"] == [2] * len(names)
        and len(value["slots"]) == len(names)
        and all(type(n) is int and 0 <= n < count for n in value["slots"])
        and value["slots"] == sorted(set(value["slots"])),
        "Exact ordered synthetic questions/slots",
    )
    need(
        type(value["letters"]) is list
        and len(value["letters"]) == len(set(value["letters"])) == 52
        and all(type(n) is int and n >= 0 for n in value["letters"]),
        "Exactly52 distinct candidate token IDs",
    )
    need(
        value["slots"][0] < 512 and (not long or value["slots"][1] > 1536),
        "Observed early/late slot geometry",
    )
    return value


def response_check(value, payload, context, full):
    need(value.get("status") == "ok", "Native request returned status ok")
    for key in ("case_id", "question_ids", "slots"):
        need(value.get(key) == payload[key], "Native response exact " + key)
    need(
        type(value.get("slots")) is list
        and type(value.get("runtime_slots")) is list
        and all(type(n) is int for n in value["slots"] + value["runtime_slots"])
        and value.get("runtime_slots") == payload["slots"]
        and type(value.get("native_token_count")) is int
        and value.get("native_token_count") == payload["source_input_token_count"],
        "Native exact runtime slots/token count",
    )
    rows = value.get("raw_logits")
    need(
        type(rows) is list and len(rows) == len(payload["nopts"]),
        "All native legal logit rows present",
    )
    for row, count in zip(rows, payload["nopts"]):
        need(
            type(row) is list
            and len(row) == count
            and all(type(n) in (int, float) and math.isfinite(n) for n in row),
            "Finite exact legal logit row width",
        )
    runtime = value["runtime"]
    need(
        runtime["engine"] == "llama.cpp"
        and runtime["llama_cpp_revision"] == "184f6536024668e8908ab8c86033d56708e62b40"
        and type(runtime["context_size"]) is int
        and runtime["context_size"] == context
        and runtime["batch_size"] == runtime["microbatch_size"] == 512
        and type(runtime["gpu_layers_requested"]) is int
        and runtime["gpu_layers_requested"] == 0
        and type(runtime["swa_full"]) is bool
        and runtime["swa_full"] is full
        and runtime["cpu_mmap_only"] is True
        and runtime["decision_passes"] == 1
        and runtime["autoregressive_tokens"] == 0
        and runtime["cache_scope"] == "one_request_cleared_between_requests"
        and runtime["logits_semantics"] == "post_native_softcap_pre_temperature",
        "Actual selected native runtime contract",
    )
    return rows


def compare(left, right):
    need(
        len(left) == len(right) and all(len(a) == len(b) for a, b in zip(left, right)),
        "Comparable legal rows",
    )
    maximum = max(abs(a - b) for x, y in zip(left, right) for a, b in zip(x, y))
    same = all(
        max(range(len(a)), key=a.__getitem__) == max(range(len(b)), key=b.__getitem__)
        for a, b in zip(left, right)
    )
    return {
        "max_abs_logit_difference": maximum,
        "preselected_threshold": THRESHOLD,
        "same_argmax": same,
        "passed": maximum <= THRESHOLD and same,
        "scope": "CPU cache arithmetic agreement on synthetic input; no answer accuracy or benchmark qualification",
    }


def error_row(error):
    return {
        "type": type(error).__name__,
        "message": str(error)[:32768],
        "evidence": getattr(error, "evidence", None),
    }
