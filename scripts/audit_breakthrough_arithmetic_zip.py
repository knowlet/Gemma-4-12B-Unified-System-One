"""Exact CPU-only arithmetic replay in the original frozen Modal image.

Replays an exact verified local stage snapshot passed as a compact ZIP argument.
Reads only terminal stage receipts from a server-side read-only Volume. It
never constructs a model, calls its forward, downloads HF assets, trains, or
modifies the Volume. Expected vectors are retained, including failed checks.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import stat
import sys
import tempfile
import types
import zipfile
from pathlib import Path

IMAGE_ID = "im-hWNAPF8eRYjs4tA8VG582u"
VOLUME = "gemma-unified-system-one"
COHERENCE_COMMIT = "e18733694623aa93058e279c3534f8b8e2edefa3"
PROFILES = {
    "ablate": ["released-current", "released-user_question", "base-current", "base-user_question"],
    "train": [
        "released-user-control",
        "released-user-head-init",
        "released-user-head",
        "released-user-lora",
    ],
}
MAX_MEMBER_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_ZIP_BYTES = 128 * 1024 * 1024
MAX_RESULT_BYTES = 128 * 1024 * 1024
RESULT_CHUNK_BYTES = 256 * 1024
RESULT_STREAM_PROTOCOL = "s1-cpu-audit-result-v1"
PARENT_ARITHMETIC_SOURCE_SHA256 = "b22966075af74816d03764cae47f1e210a400b9d06c1b79f38d9a51f5c03381d"
_FROZEN_CONTRACTS = None


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def contracts():
    if _FROZEN_CONTRACTS is not None:
        return _FROZEN_CONTRACTS
    from s1 import contracts as module

    return module


def cpu_answer(request, rows, temperatures):
    import torch

    require(
        isinstance(rows, list)
        and [row.get("id") for row in rows] == [q.id for q in request.questions],
        "full ordered raw question population required",
    )
    answers, question_records = {}, []
    for question, row in zip(request.questions, rows, strict=True):
        labels = question.labels()
        require(row.get("labels") == labels, "ordered raw labels differ")
        require(
            isinstance(temperatures, dict) and question.type in temperatures,
            "positive finite type temperature required",
        )
        temperature = temperatures[question.type]
        require(
            type(temperature) in (int, float) and math.isfinite(temperature) and temperature > 0,
            "positive finite type temperature required",
        )
        values = row.get("logits")
        require(
            isinstance(values, list)
            and len(values) == len(labels)
            and all(type(x) in (int, float) and math.isfinite(x) for x in values),
            "aligned finite raw logits required",
        )
        tensor = torch.tensor(values, dtype=torch.float32, device="cpu") / temperature
        require(bool(torch.isfinite(tensor).all()), "nonfinite FP32 temperature transform")
        probabilities = tensor.softmax(-1).tolist()
        answer = contracts().answer_from_probabilities(question, probabilities)
        answers[question.id] = answer
        question_records.append(
            {
                "id": question.id,
                "type": question.type,
                "labels": labels,
                "temperature": temperature,
                "expected_probabilities": probabilities,
                "expected_answer": answer,
            }
        )
    return {"answers": answers, "question_records": question_records}


def compare_answers(request, expected_answers, actual_answers):
    ids = {q.id for q in request.questions}
    require(
        isinstance(expected_answers, dict)
        and isinstance(actual_answers, dict)
        and set(expected_answers) == set(actual_answers) == ids,
        "full answer question population required",
    )
    maximum, mismatches, fields, probabilities = 0.0, 0, 0, 0
    for question in request.questions:
        expected, actual = expected_answers[question.id], actual_answers[question.id]
        require(
            isinstance(actual, dict)
            and actual.get("id") == question.id
            and actual.get("type") == question.type,
            "native question identity/type differs",
        )
        observed = actual.get("probabilities")
        require(
            isinstance(observed, dict) and list(observed) == question.labels(),
            "native label population differs",
        )
        require(
            all(
                type(value) in (int, float) and math.isfinite(value) for value in observed.values()
            ),
            "finite native probabilities required",
        )
        values = contracts().validate_distribution(list(observed.values()), len(question.labels()))
        for label, value in zip(question.labels(), values, strict=True):
            reference = expected["probabilities"][label]
            maximum = max(maximum, abs(value - reference))
            mismatches += value != reference
            probabilities += 1
        fields += actual != expected
    return {
        "questions": len(request.questions),
        "probabilities": probabilities,
        "max_abs_error": maximum,
        "mismatch_count": mismatches,
        "native_field_mismatch_count": fields,
    }


def _safe(name):
    require(
        isinstance(name, str)
        and name
        and not name.startswith("/")
        and "\\" not in name
        and ":" not in name
        and all(ord(c) >= 32 for c in name)
        and all(part not in {"", ".", ".."} for part in name.split("/")),
        "unsafe arithmetic input path",
    )
    return name


def collect_input_manifest(roots, *, include_public=False):
    """Bind every used local byte before the only CPU cloud call."""
    output = {}
    for stage, directory in roots.items():
        directory = Path(directory)
        names = {"receipt.json"}
        names.update(
            p.relative_to(directory).as_posix() for p in (directory / "sources").rglob("*")
        )
        for profile in PROFILES[stage]:
            prefix = profile + "/"
            names.update(
                prefix + name
                for name in (
                    "receipt.json",
                    "calibration.json",
                    "raw-logits.json",
                    "coherence-mini.json",
                    "coherence-unit.json",
                )
            )
            folders = ["coherence-cache", "coherence-cache-unit"]
            if include_public:
                folders += ["public231", "public231-unit", "public231-published_global"]
            for folder in folders:
                names.update(
                    p.relative_to(directory).as_posix()
                    for p in (directory / profile / folder).rglob("*")
                    if p.is_file()
                )
        for name in sorted(names):
            path = directory / _safe(name)
            if path.is_dir():
                continue
            require(path.is_file() and not path.is_symlink(), "regular arithmetic input required")
            raw = path.read_bytes()
            require(len(raw) <= 16 * 1024 * 1024, "arithmetic JSON input exceeds bound")
            output[stage + "/" + name] = {"sha256": digest(raw), "size_bytes": len(raw)}
    require(
        sum(r["size_bytes"] for r in output.values()) <= 256 * 1024 * 1024,
        "input total exceeds bound",
    )
    return output


def _aggregate(checks):
    return {
        key: max((row[key] for row in checks), default=0.0)
        if key == "max_abs_error"
        else sum(row[key] for row in checks)
        for key in (
            "questions",
            "probabilities",
            "max_abs_error",
            "mismatch_count",
            "native_field_mismatch_count",
        )
    }


def validate_input_zip(payload, expected_inputs):
    """Validate all uploaded bytes in memory; never extract ZIP paths."""
    require(
        isinstance(payload, bytes) and len(payload) <= MAX_ZIP_BYTES,
        "compressed ZIP bound exceeded",
    )
    require(isinstance(expected_inputs, dict) and bool(expected_inputs), "input manifest required")
    for name, identity in expected_inputs.items():
        _safe(name)
        require(name.split("/")[0] in PROFILES, "unexpected input stage")
        require(
            isinstance(identity, dict)
            and type(identity.get("size_bytes")) is int
            and 0 <= identity["size_bytes"] <= MAX_MEMBER_BYTES
            and isinstance(identity.get("sha256"), str)
            and bool(re.fullmatch(r"[0-9a-f]{64}", identity.get("sha256", ""))),
            "invalid bounded input member identity",
        )
    require(
        sum(row["size_bytes"] for row in expected_inputs.values()) <= MAX_TOTAL_BYTES,
        "input total bound exceeded",
    )
    with zipfile.ZipFile(io.BytesIO(payload)) as zipped:
        infos = zipped.infolist()
        names = [info.filename for info in infos]
        require(len(names) == len(set(names)), "duplicate ZIP member")
        require(set(names) == set(expected_inputs), "exact ZIP member population required")
        require(
            infos[0].header_offset == 0
            and not zipped.comment
            and payload.rfind(b"PK\x05\x06") + 22 == len(payload),
            "unambiguous single ZIP container required",
        )
        output = {}
        for info in infos:
            _safe(info.filename)
            require(
                not info.is_dir()
                and stat.S_IFMT(info.external_attr >> 16) in {0, stat.S_IFREG}
                and not info.flag_bits & 1
                and info.compress_type in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED},
                "regular unencrypted supported ZIP member required",
            )
            identity = expected_inputs[info.filename]
            require(
                info.file_size == identity["size_bytes"],
                "ZIP member size differs: " + info.filename,
            )
            with zipped.open(info) as handle:
                raw = handle.read(MAX_MEMBER_BYTES + 1)
            require(
                len(raw) == identity["size_bytes"], "ZIP actual byte size differs: " + info.filename
            )
            require(digest(raw) == identity["sha256"], "ZIP member SHA differs: " + info.filename)
            output[info.filename] = raw
    return output


def pack_input_zip(roots, expected_inputs):
    """Sort/fix ZIP metadata and recheck every locally transferred member."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for name, identity in sorted(expected_inputs.items()):
            _safe(name)
            stage, separator, relative = name.partition("/")
            require(separator and stage in roots and stage in PROFILES, "unexpected snapshot stage")
            root = Path(roots[stage])
            path = root / relative
            ancestors = [path, *list(path.parents)[: len(Path(relative).parts)]]
            require(
                not any(p.is_symlink() for p in ancestors) and path.is_file(),
                "regular snapshot member required",
            )
            with path.open("rb") as handle:
                raw = handle.read(MAX_MEMBER_BYTES + 1)
            require(
                len(raw) == identity["size_bytes"] and digest(raw) == identity["sha256"],
                "local snapshot member size/SHA differs: " + name,
            )
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zipped.writestr(info, raw)
    payload = buffer.getvalue()
    validate_input_zip(payload, expected_inputs)
    return payload


def snapshot_origin(roots, expected_inputs, run):
    output = {}
    for stage, directory in roots.items():
        raw = (Path(directory) / ".evidence-download/receipt.json").read_bytes()
        receipt = json.loads(raw)
        require(
            receipt["status"] == "completed"
            and not receipt["errors"]
            and receipt["run"] == run
            and receipt["stage"] == stage
            and receipt["volume"] == VOLUME
            and receipt["remote_prefix"] == "breakthrough/" + run + "/" + stage
            and receipt["gpu_receipt_sha256"] == expected_inputs[stage + "/receipt.json"]["sha256"],
            "verified transfer origin/terminal receipt differs",
        )
        members = {row["path"]: row for row in receipt["members"]}
        require(
            len(members) == len(receipt["members"]) == receipt["expected_files"],
            "complete transfer origin population required",
        )
        require(
            sum(record["size_bytes"] for record in members.values())
            == receipt["expected_size_bytes"],
            "complete transfer origin byte population required",
        )
        for name, identity in expected_inputs.items():
            if name.startswith(stage + "/"):
                record = members[name.removeprefix(stage + "/")]
                require(
                    all(record[k] == identity[k] for k in ("sha256", "size_bytes")),
                    "snapshot differs from verified transfer member",
                )
        output[stage] = {
            "run": run,
            "stage": stage,
            "volume": VOLUME,
            "remote_prefix": receipt["remote_prefix"],
            "transfer_receipt_sha256": digest(raw),
            "terminal_gpu_receipt_sha256": receipt["gpu_receipt_sha256"],
            "transfer_source_sha256": receipt["source_sha256"],
            "transfer_files": receipt["expected_files"],
            "transfer_size_bytes": receipt["expected_size_bytes"],
        }
    return output


def read_bound_member(base, name, identity, *, observation=None):
    """Reject payload symlinks; system ancestors above the authorized root are outside this check."""
    _safe(name)
    require(name.split("/")[0] in PROFILES, "unexpected input stage")
    path = Path(base) / name
    ancestors = [path, *list(path.parents)[: len(Path(name).parts)]]
    metadata = path.stat()
    require(
        not any(p.is_symlink() for p in ancestors) and stat.S_ISREG(metadata.st_mode),
        "remote arithmetic input type differs: " + name,
    )
    with path.open("rb") as handle:
        raw = handle.read(16 * 1024 * 1024 + 1)
    require(
        len(raw) == identity["size_bytes"], "remote arithmetic input byte size differs: " + name
    )
    require(digest(raw) == identity["sha256"], "remote arithmetic input hash differs: " + name)
    if observation is not None:
        observation.update(
            {"path": name, "metadata_size_bytes": metadata.st_size, "actual_size_bytes": len(raw)}
        )
    return raw


def audit_run(run, expected_inputs, source_sha256, input_zip, input_origin, include_public=False):
    """Remote entry point; all mutations are restricted to ephemeral /tmp."""
    import importlib.metadata
    import importlib.util
    import platform
    import subprocess
    from datetime import datetime, timezone

    import numpy as np
    import torch

    global _FROZEN_CONTRACTS
    require(bool(re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", run)), "invalid existing run")
    source_raw = Path(__file__).read_bytes()
    require(digest(source_raw) == source_sha256, "actual verifier source bytes differ")
    require(
        platform.python_version() == "3.12.10"
        and importlib.metadata.version("torch") == "2.10.0"
        and importlib.metadata.version("numpy") == "2.5.3"
        and platform.system() == "Linux"
        and platform.machine() in {"x86_64", "amd64"}
        and not torch.cuda.is_available()
        and torch.cuda.device_count() == 0,
        "original pinned CPU arithmetic environment/no-GPU guard failed",
    )
    require(
        subprocess.check_output(["git", "-C", "/opt/coherence", "rev-parse", "HEAD"])
        .decode()
        .strip()
        == COHERENCE_COMMIT,
        "pinned official coherence source differs",
    )
    base = Path("/vol/breakthrough") / run
    mount_ancestry = [
        {"path": str(path), "is_symlink": path.is_symlink(), "metadata_mode": path.stat().st_mode}
        for path in (base, *base.parents)
    ]

    require(len(expected_inputs) == 20142, "exact original 20,142 input population required")
    raw_files = validate_input_zip(input_zip, expected_inputs)
    receipt_observations = []
    for stage in PROFILES:
        require(
            input_origin[stage]["run"] == run and input_origin[stage]["stage"] == stage,
            "snapshot origin run/stage differs",
        )
        observation = {}
        name = stage + "/receipt.json"
        before = read_bound_member(base, name, expected_inputs[name], observation=observation)
        require(
            before == raw_files[name], "read-only Volume terminal receipt differs from snapshot"
        )
        receipt_observations.append(observation)
    size_discrepancies = [
        obs
        for obs in receipt_observations
        if obs["metadata_size_bytes"] != obs["actual_size_bytes"]
    ]
    data = {name: json.loads(raw) for name, raw in raw_files.items() if name.endswith(".json")}
    source_maps = {}
    for stage in PROFILES:
        receipt = data[stage + "/receipt.json"]
        require(
            receipt["status"] == "completed"
            and receipt["run"] == run
            and receipt["stage"] == stage,
            "existing GPU stage receipt must be terminal completed",
        )
        source_maps[stage] = receipt["source_files"]
        require(len(receipt["source_files"]) == 66, "complete 66-source proof required")
        for key, sha in receipt["source_files"].items():
            require(
                digest(raw_files[stage + "/sources/" + _safe(key) + ".txt"]) == sha,
                "executed source hash differs",
            )
    require(source_maps["ablate"] == source_maps["train"], "executed stage sources differ")
    with tempfile.TemporaryDirectory(prefix="s1-arithmetic-") as temporary:
        temporary = Path(temporary)

        def module(name, raw):
            path = temporary / (name + ".py")
            path.write_bytes(raw)
            spec = importlib.util.spec_from_file_location(name, path)
            value = importlib.util.module_from_spec(spec)
            sys.modules[name] = value
            spec.loader.exec_module(value)
            return value

        _FROZEN_CONTRACTS = module(
            "arithmetic_frozen_contracts", raw_files["train/sources/s1/contracts.py.txt"]
        )
        package = types.ModuleType("s1")
        package.__path__ = []
        sys.modules["s1"] = package
        sys.modules["s1.contracts"] = _FROZEN_CONTRACTS
        fitted_helper = module(
            "arithmetic_frozen_training", raw_files["train/sources/breakthrough_training.py.txt"]
        )
        sys.path.insert(0, "/opt/coherence/src")
        import jevbench as jb

        profiles = {}
        for stage, names in PROFILES.items():
            for name in names:
                prefix = stage + "/" + name + "/"
                receipt = data[prefix + "receipt.json"]
                calibration = data[prefix + "calibration.json"]
                rows = calibration["records"]
                require(
                    len({r["case_id"] for r in rows}) == 192, "full 192 calibration cases required"
                )
                require(
                    all(r["split"] == "calibration" for r in rows),
                    "calibration-only records required",
                )
                fitted = fitted_helper.fit_temperatures(rows)
                grid = np.geomspace(0.1, 10.0, 81)
                grid_checks = {}
                for kind in ("choice", "noul", "score"):
                    selected = [r for r in rows if r["type"] == kind]
                    losses = []
                    for t in grid:
                        values = []
                        for row in selected:
                            x = np.asarray(row["logits"], dtype=np.float64) / t
                            truth = np.asarray(row["target"], dtype=np.float64)
                            shifted = x - x.max()
                            logp = shifted - np.log(np.exp(shifted).sum())
                            values.append(float(-(truth * logp).sum()))
                        losses.append(float(np.mean(values)))
                    grid_checks[kind] = {
                        "n": len(selected),
                        "argmin_index": int(np.argmin(losses)),
                        "losses": losses,
                        "expected_metadata": fitted[kind],
                        "saved_metadata": calibration["temperatures"][kind],
                        "exact_metadata_match": fitted[kind] == calibration["temperatures"][kind],
                    }
                temperatures = {
                    kind: value["temperature"]
                    for kind, value in receipt["identity"]["calibration"].items()
                }
                require(
                    receipt["identity"]["calibration"] == calibration["temperatures"],
                    "child calibration identity differs",
                )
                raw = data[prefix + "raw-logits.json"]
                result = {
                    "stage": stage,
                    "identity": receipt["identity"],
                    "file_sha256": {
                        path.removeprefix(prefix): identity["sha256"]
                        for path, identity in expected_inputs.items()
                        if path.startswith(prefix)
                    },
                    "calibration": {
                        "cases": 192,
                        "record_count": len(rows),
                        "record_identity_order_sha256": digest(
                            encoded(
                                [
                                    [r["case_id"], r["question_id"], r["type"], r["labels"]]
                                    for r in rows
                                ]
                            )
                        ),
                        "records": rows,
                        "grid": grid.tolist(),
                        "types": grid_checks,
                    },
                    "public": {},
                    "coherence": {},
                }
                public_policies = (
                    (
                        ("calibrated", "public231"),
                        ("unit", "public231-unit"),
                        ("published_global", "public231-published_global"),
                    )
                    if include_public
                    else ()
                )
                for policy, folder in public_policies:
                    public_prefix = prefix + folder + "/"
                    manifest = data[public_prefix + "manifest.json"]
                    records = [
                        json.loads(line)
                        for line in raw_files[public_prefix + "records.jsonl"].splitlines()
                        if line
                    ]
                    require(
                        len(records) == 231
                        and [r["task_id"] for r in records] == manifest["ordered_task_ids"],
                        "full ordered public231 required",
                    )
                    ts = (
                        temperatures
                        if policy == "calibrated"
                        else manifest["identity"]["calibration"]
                    )
                    vectors, checks = [], []
                    for row in records:
                        raw_path = _safe(row["raw_path"])
                        payload = data[public_prefix + raw_path]
                        require(
                            row["ok"] is True and payload["error"] is None,
                            "public successful complete answers required",
                        )
                        request = contracts().DecisionRequest(**payload["request"])
                        runtime = payload["response"]["runtime"]
                        supplied_rows = runtime["raw_logits"]
                        key = digest(request.model_dump_json().encode())
                        require(supplied_rows == raw[key], "public changed recorded raw logits")
                        answers = cpu_answer(request, supplied_rows, ts)
                        checks.append(
                            compare_answers(
                                request, answers["answers"], payload["response"]["answers"]
                            )
                        )
                        vectors.append(
                            {
                                "task_id": row["task_id"],
                                "raw_file": raw_path,
                                "request_sha256": row["request_sha256"],
                                "raw_key": key,
                                "request": payload["request"],
                                "raw_rows": supplied_rows,
                                "questions": answers["question_records"],
                                "expected_answers": answers["answers"],
                            }
                        )
                    result["public"][policy] = {
                        "backend_identity": manifest["identity"],
                        "ordered_task_ids": manifest["ordered_task_ids"],
                        "requests": vectors,
                        **_aggregate(checks),
                    }
                for policy, report_name, folder, ts in (
                    ("calibrated", "coherence-mini.json", "coherence-cache", temperatures),
                    (
                        "unit",
                        "coherence-unit.json",
                        "coherence-cache-unit",
                        {kind: 1.0 for kind in temperatures},
                    ),
                ):
                    report = data[prefix + report_name]
                    backend = report["result"]["backend"]
                    cache_prefix = prefix + folder + "/"
                    cache = {
                        path.removeprefix(cache_prefix): answer
                        for path, answer in data.items()
                        if path.startswith(cache_prefix)
                    }
                    require(len(cache) == 1248, "complete 1248 coherence cache files required")
                    vectors, checks = [], []

                    def answer(state, questions):
                        filename = (
                            digest(encoded({"b": backend, "s": state, "q": questions, "r": 0}))
                            + ".json"
                        )
                        request = contracts().DecisionRequest(state=state, questions=questions)
                        key = digest(request.model_dump_json().encode())
                        require(
                            filename in cache and key in raw,
                            "official request/cache/raw key missing",
                        )
                        expected = cpu_answer(request, raw[key], ts)
                        checks.append(
                            compare_answers(request, expected["answers"], cache[filename])
                        )
                        vectors.append(
                            {
                                "raw_key": key,
                                "cache_file": filename,
                                "expected_answers": expected["answers"],
                            }
                        )
                        return cache[filename]

                    settings = report["settings"]
                    replay = jb.evaluate(
                        jb.from_callable(answer, name=backend),
                        suite="jevbench-mini",
                        cache=":memory:",
                        concurrency=1,
                        progress=False,
                        tolerance=settings["tolerance"],
                        min_n=settings["min_n"],
                        bootstrap=settings["bootstrap"],
                    )
                    require(
                        not replay.result.get("errors"), "official CPU reconstruction had errors"
                    )
                    ordered = [row["cache_file"] for row in vectors]
                    require(
                        len(ordered) == len(set(ordered)) == 1248 and set(ordered) == set(cache),
                        "full official ordered cache population differs",
                    )
                    result["coherence"][policy] = {
                        "backend": backend,
                        "cache_sha256": {
                            name: expected_inputs[cache_prefix + name]["sha256"]
                            for name in sorted(cache)
                        },
                        "ordered_cache_files": ordered,
                        "ordered_cache_sha256": digest(encoded(ordered)),
                        "ordered_raw_keys_sha256": digest(
                            encoded([row["raw_key"] for row in vectors])
                        ),
                        "requests": vectors,
                        **_aggregate(checks),
                    }
                profiles[name] = result
                print(
                    json.dumps(
                        {
                            "arithmetic_profile": name,
                            "exact_public": all(
                                v["mismatch_count"] == v["native_field_mismatch_count"] == 0
                                for v in result["public"].values()
                            ),
                            "exact_coherence": all(
                                v["mismatch_count"] == v["native_field_mismatch_count"] == 0
                                for v in result["coherence"].values()
                            ),
                        }
                    ),
                    flush=True,
                )
    # Exact receipt recheck binds reads to immutable finalized stages.
    for stage in PROFILES:
        require(
            digest(
                read_bound_member(
                    base, stage + "/receipt.json", expected_inputs[stage + "/receipt.json"]
                )
            )
            == expected_inputs[stage + "/receipt.json"]["sha256"],
            "GPU receipt changed during CPU audit",
        )
    proof = {
        "schema_version": 1,
        "status": "completed",
        "run": run,
        "method": "pinned_linux_cpu_arithmetic; original strict macOS audit remains separate",
        "public_replay": "included"
        if include_public
        else "not_replayed; already independently strict-verified locally",
        "verifier_source_sha256": digest(source_raw),
        "original_image_id": IMAGE_ID,
        "parent_arithmetic_source_sha256": PARENT_ARITHMETIC_SOURCE_SHA256,
        "input_delivery": {
            "method": "verified_local_snapshot_zip_argument",
            "sha256": digest(input_zip),
            "size_bytes": len(input_zip),
            "volume_receipt_reads": 4,
            "origin": input_origin,
        },
        "source_files": source_maps,
        "input_members": expected_inputs,
        "input_manifest_sha256": digest(encoded(expected_inputs)),
        "filesystem_observations": {
            "mount_ancestry": mount_ancestry,
            "absolute_ancestor_guard_would_reject": any(
                row["is_symlink"] for row in mount_ancestry
            ),
            "metadata_size_vs_actual_byte_discrepancies": len(size_discrepancies),
            "metadata_size_discrepancy_examples": size_discrepancies[:20],
            "scope": "Stage receipts only; other members use uploaded SHA-bound snapshot. Prior CPU failures preserved separately",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": importlib.metadata.version("torch"),
            "numpy": importlib.metadata.version("numpy"),
            "torch_threads": torch.get_num_threads(),
            "visible_gpu_count": torch.cuda.device_count(),
            "cpu_info": Path("/proc/cpuinfo").read_text().split("\n\n", 1)[0],
            "torch_build_configuration": torch.__config__.show(),
        },
        "execution": {
            "cpu_only": True,
            "no_model_call": True,
            "volume_read_only": True,
            "cpu": 2,
            "memory_mib": 4096,
            "timeout_seconds": 600,
            "network_blocked": True,
            "model_forwards": 0,
            "training_updates": 0,
            "hf_downloads": 0,
            "volume_writes": 0,
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "profiles": profiles,
        "exact_arithmetic_match": all(
            all(row["exact_metadata_match"] for row in profile["calibration"]["types"].values())
            and all(
                row["mismatch_count"] == row["native_field_mismatch_count"] == 0
                for row in (*profile["coherence"].values(), *profile["public"].values())
            )
            for profile in profiles.values()
        ),
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        zipped.writestr("proof.json", encoded(proof))
        zipped.writestr("verifier.py.txt", source_raw)
    return buffer.getvalue()


def result_stream(payload, run, source_sha256):
    """Keep every yielded serialized result safely below Modal's 2 MiB blob threshold."""
    require(
        isinstance(payload, bytes) and 0 < len(payload) <= MAX_RESULT_BYTES,
        "bounded nonempty CPU result required",
    )
    header = {
        "kind": "header",
        "protocol": RESULT_STREAM_PROTOCOL,
        "run": run,
        "verifier_source_sha256": source_sha256,
        "size_bytes": len(payload),
        "sha256": digest(payload),
        "chunk_bytes": RESULT_CHUNK_BYTES,
        "chunk_count": (len(payload) + RESULT_CHUNK_BYTES - 1) // RESULT_CHUNK_BYTES,
    }
    yield header
    for index, offset in enumerate(range(0, len(payload), RESULT_CHUNK_BYTES)):
        raw = payload[offset : offset + RESULT_CHUNK_BYTES]
        yield {
            "kind": "chunk",
            "index": index,
            "size_bytes": len(raw),
            "sha256": digest(raw),
            "data": raw,
        }
    yield {
        "kind": "complete",
        "size_bytes": len(payload),
        "sha256": header["sha256"],
        "chunk_count": header["chunk_count"],
    }


def audit_run_stream(
    run, expected_inputs, source_sha256, input_zip, input_origin, include_public=False
):
    """Only the delivery changes; the pinned audit computes the same complete result."""
    payload = audit_run(
        run, expected_inputs, source_sha256, input_zip, input_origin, include_public
    )
    yield from result_stream(payload, run, source_sha256)


def assemble_result_stream(stream, run, source_sha256, *, partial_path=None, receipt_path=None):
    """Require all ordered bounded bytes, hashes and terminal receipt; retain failed input."""
    require(
        (partial_path is None) == (receipt_path is None),
        "partial bytes and transport receipt paths must be paired",
    )
    if partial_path is not None:
        require(
            all(
                not Path(p).exists() and not Path(p).is_symlink()
                for p in (partial_path, receipt_path)
            ),
            "exclusive new CPU transport evidence paths required",
        )
    stream = iter(stream)
    buffer = io.BytesIO()
    observed = {"protocol": RESULT_STREAM_PROTOCOL, "status": "incomplete", "chunks": []}
    partial = None

    def next_record():
        try:
            return next(stream)
        except StopIteration as exc:
            raise ValueError("truncated CPU result stream") from exc

    try:
        if partial_path is not None:
            partial = Path(partial_path).open("xb")
        header = next_record()
        require(
            isinstance(header, dict)
            and set(header)
            == {
                "kind",
                "protocol",
                "run",
                "verifier_source_sha256",
                "size_bytes",
                "sha256",
                "chunk_bytes",
                "chunk_count",
            }
            and header["kind"] == "header"
            and header["protocol"] == RESULT_STREAM_PROTOCOL
            and header["run"] == run
            and header["verifier_source_sha256"] == source_sha256
            and type(header["size_bytes"]) is int
            and 0 < header["size_bytes"] <= MAX_RESULT_BYTES
            and isinstance(header["sha256"], str)
            and bool(re.fullmatch(r"[0-9a-f]{64}", header["sha256"]))
            and type(header["chunk_bytes"]) is int
            and header["chunk_bytes"] == RESULT_CHUNK_BYTES
            and type(header["chunk_count"]) is int
            and header["chunk_count"]
            == (header["size_bytes"] + RESULT_CHUNK_BYTES - 1) // RESULT_CHUNK_BYTES,
            "CPU result stream header identity/bounds differ",
        )
        header = dict(header)
        observed["header"] = header
        for index in range(header["chunk_count"]):
            record = next_record()
            size = min(RESULT_CHUNK_BYTES, header["size_bytes"] - buffer.tell())
            require(
                isinstance(record, dict)
                and set(record) == {"kind", "index", "size_bytes", "sha256", "data"}
                and record["kind"] == "chunk"
                and type(record["index"]) is int
                and record["index"] == index
                and type(record["size_bytes"]) is int
                and record["size_bytes"] == size
                and isinstance(record["data"], bytes)
                and len(record["data"]) == size
                and record["sha256"] == digest(record["data"]),
                "CPU result stream chunk order/size/hash differs",
            )
            buffer.write(record["data"])
            if partial is not None:
                partial.write(record["data"])
                partial.flush()
            observed["chunks"].append({k: v for k, v in record.items() if k != "data"})
        terminal = next_record()
        require(
            isinstance(terminal, dict)
            and set(terminal) == {"kind", "size_bytes", "sha256", "chunk_count"}
            and terminal["kind"] == "complete"
            and type(terminal["size_bytes"]) is int
            and terminal["size_bytes"] == header["size_bytes"]
            and type(terminal["chunk_count"]) is int
            and terminal["chunk_count"] == header["chunk_count"]
            and terminal["sha256"] == header["sha256"],
            "CPU result stream terminal digest receipt differs",
        )
        observed["terminal"] = dict(terminal)
        try:
            next(stream)
        except StopIteration:
            pass
        else:
            raise ValueError("extra CPU result stream record")
        payload = buffer.getvalue()
        require(
            len(payload) == header["size_bytes"] and digest(payload) == header["sha256"],
            "CPU result stream complete byte hash differs",
        )
        observed["status"] = "completed"
        return payload
    except BaseException as exc:
        observed.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        if partial is not None:
            partial.close()
        if receipt_path is not None:
            observed.update(
                {"received_size_bytes": buffer.tell(), "received_sha256": digest(buffer.getvalue())}
            )
            with Path(receipt_path).open("xb") as handle:
                handle.write(encoded(observed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--ablate", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-public",
        action="store_true",
        help="Optional public arithmetic replay; default is the required coherence/calibration scope only",
    )
    args = parser.parse_args()
    require(
        not args.output.exists() and not args.output.is_symlink(),
        "exclusive new CPU audit directory required",
    )
    expected_inputs = collect_input_manifest(
        {"ablate": args.ablate, "train": args.train}, include_public=args.include_public
    )
    source_raw = Path(__file__).read_bytes()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "verifier.py.txt").write_bytes(source_raw)
    (args.output / "input-manifest.json").write_bytes(encoded(expected_inputs))
    print(
        json.dumps(
            {
                "prepared": str(args.output),
                "input_members": len(expected_inputs),
                "input_bytes": sum(value["size_bytes"] for value in expected_inputs.values()),
                "verifier_source_sha256": digest(source_raw),
                "scope": "CPU coherence/calibration only"
                if not args.include_public
                else "CPU coherence/calibration/public",
            }
        ),
        flush=True,
    )
    input_zip = pack_input_zip({"ablate": args.ablate, "train": args.train}, expected_inputs)
    input_origin = snapshot_origin(
        {"ablate": args.ablate, "train": args.train}, expected_inputs, args.run
    )
    (args.output / "input.zip").write_bytes(input_zip)
    (args.output / "input-origin.json").write_bytes(encoded(input_origin))
    print(
        json.dumps({"input_zip_bytes": len(input_zip), "input_zip_sha256": digest(input_zip)}),
        flush=True,
    )
    import modal

    app = modal.App("s1-breakthrough-arithmetic-zip-cpu-audit")
    image = modal.Image.from_id(IMAGE_ID)
    volume = modal.Volume.from_name(VOLUME, create_if_missing=False).with_mount_options(
        read_only=True
    )
    remote = app.function(
        image=image,
        cpu=2,
        memory=4096,
        timeout=600,
        volumes={"/vol": volume},
        block_network=True,
        max_containers=1,
        single_use_containers=True,
    )(audit_run_stream)
    try:
        with app.run():
            (args.output / "modal-app.json").write_bytes(
                encoded(
                    {
                        "app_id": app.app_id,
                        "function_id": remote.object_id,
                        "image_id": IMAGE_ID,
                        "verifier_source_sha256": digest(source_raw),
                    }
                )
            )
            payload = assemble_result_stream(
                remote.remote_gen(
                    args.run,
                    expected_inputs,
                    digest(source_raw),
                    input_zip,
                    input_origin,
                    args.include_public,
                ),
                args.run,
                digest(source_raw),
                partial_path=args.output / "result.zip.partial",
                receipt_path=args.output / "result-stream.json",
            )
        with (args.output / "result.zip").open("xb") as handle:
            handle.write(payload)
        with zipfile.ZipFile(io.BytesIO(payload)) as zipped:
            require(
                zipped.namelist() == ["proof.json", "verifier.py.txt"],
                "unexpected CPU proof archive members",
            )
            require(
                zipped.read("verifier.py.txt") == source_raw, "CPU verifier source bytes differ"
            )
            with (args.output / "proof.json").open("xb") as handle:
                handle.write(zipped.read("proof.json"))
        (args.output / "result.zip.partial").unlink()
    except BaseException as exc:
        (args.output / "failure.json").write_bytes(
            encoded(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "scope": "CPU arithmetic audit only; original GPU receipts unchanged",
                }
            )
        )
        raise
    print(
        json.dumps(
            {
                "saved": str(args.output),
                "proof_sha256": digest((args.output / "proof.json").read_bytes()),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
