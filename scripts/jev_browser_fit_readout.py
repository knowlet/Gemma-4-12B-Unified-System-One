#!/usr/bin/env python3
"""Fit a small CPU decision readout on frozen real development-browser features.

The NF4 backbone supplies genuine hidden states with the exact grounded prompt.
Only detached selected LM-head rows are optimized; the full text helper and
backbone remain unchanged. This script reports fitting diagnostics, not task
success. A separate closed-loop browser run must evaluate the saved sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import time
from pathlib import Path
from types import SimpleNamespace

from breakthrough_training import fit_head
from jev_browser_training_data import training_case


def _sha(blob):
    return hashlib.sha256(blob).hexdigest()


def _json(blob):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    def invalid_constant(_value):
        raise ValueError("nonfinite JSON number")

    return json.loads(blob, object_pairs_hook=object_pairs, parse_constant=invalid_constant)


def _input_manifest(blobs):
    files = {
        label: {"sha256": _sha(blob), "size_bytes": len(blob)}
        for label, blob in sorted(blobs.items())
    }
    return {
        "schema_version": 1,
        "files": files,
        "identity_sha256": _sha(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()),
    }


def _input_paths(root):
    paths = {
        label: root / label for label in ("cases.jsonl", "wire_requests.jsonl", "receipt.json")
    }
    fixture_dir = root / "fixtures"
    if fixture_dir.is_symlink() or not fixture_dir.is_dir():
        raise ValueError("Collection fixtures directory is missing or symlinked")
    for path in fixture_dir.rglob("*"):
        if path.is_symlink():
            raise ValueError("Collection inputs must not be symlinks")
        if path.is_file():
            label = path.relative_to(root).as_posix()
            if re.fullmatch(r"fixtures/[0-9]{4}\.html", label) is None:
                raise ValueError("Unexpected collection fixture path")
            paths[label] = path
    source_dir = root / "sources"
    if source_dir.exists() or source_dir.is_symlink():
        if source_dir.is_symlink() or not source_dir.is_dir():
            raise ValueError("Collection source evidence must be an ordinary directory")
        for path in source_dir.rglob("*"):
            if path.is_symlink():
                raise ValueError("Collection source evidence must not be symlinked")
            if path.is_file():
                label = path.relative_to(root).as_posix()
                if label != "sources/manifest.json" and not label.endswith(".py"):
                    raise ValueError("Unexpected collection source evidence path")
                paths[label] = path
    if any(path.is_symlink() for path in paths.values()):
        raise ValueError("Collection inputs must not be symlinks")
    return paths


def _validated_snapshot(path, split):
    """Validate one coherent byte snapshot, including native wire/episode identities."""
    from jev_browser_benchmark import UPSTREAM_FILES, UPSTREAM_REVISION, UPSTREAM_TRAVEL_SHA256

    from s1.evaluation.datasets import EvaluationCase

    configured = Path(path)
    if configured.is_symlink() or configured.name != "cases.jsonl":
        raise ValueError("Collection cases.jsonl must be an ordinary named input")
    root = configured.resolve().parent
    paths = _input_paths(root)
    # Every input is read exactly once. Parsing, hashing and freezing use these bytes.
    blobs = {label: source.read_bytes() for label, source in paths.items()}
    receipt = _json(blobs["receipt.json"])
    if (
        receipt.get("stage") != "complete"
        or receipt.get("split") != split
        or receipt.get("dataset_sha256") != _sha(blobs["cases.jsonl"])
        or receipt.get("wire_sha256") != _sha(blobs["wire_requests.jsonl"])
        or receipt.get("scope")
        != "development-only scripted training collection; no model evaluation"
    ):
        raise ValueError("A complete, unchanged development-only collection is required")
    upstream = receipt.get("upstream", {})
    upstream_files = upstream.get("source_sha256", {})
    sha_pattern = re.compile(r"[0-9a-f]{64}\Z")
    if (
        type(receipt.get("schema_version")) is not int
        or receipt["schema_version"] != 1
        or type(receipt.get("training_cases")) is not int
        or receipt["training_cases"] < 1
        or type(receipt.get("seed")) is not int
        or type(receipt.get("repeats")) is not int
        or receipt["repeats"] < 1
        or not isinstance(receipt.get("collector_sha256"), str)
        or sha_pattern.fullmatch(receipt["collector_sha256"]) is None
        or upstream.get("repository") != "https://github.com/browser-use/jev-ultrafast"
        or upstream.get("revision") != UPSTREAM_REVISION
        or set(upstream_files) != {"jev_ultrafast/" + name for name in UPSTREAM_FILES}
        or any(
            not isinstance(value, str) or sha_pattern.fullmatch(value) is None
            for value in upstream_files.values()
        )
        or upstream_files.get("jev_ultrafast/static/fixture.html") != UPSTREAM_TRAVEL_SHA256
    ):
        raise ValueError("Collection receipt identities are invalid")
    try:
        cases = [
            EvaluationCase.model_validate(_json(line), strict=True)
            for line in blobs["cases.jsonl"].splitlines()
            if line.strip()
        ]
        wires = [_json(line) for line in blobs["wire_requests.jsonl"].splitlines() if line.strip()]
    except (TypeError, ValueError):
        raise ValueError("Collection cases or wire records are invalid") from None
    if (
        not cases
        or len({case.id for case in cases}) != len(cases)
        or len(cases) != receipt.get("training_cases")
        or len(wires) != len(cases)
        or any(
            case.split != split or case.schema_id != "jev-development-supervision-v1"
            for case in cases
        )
        or not receipt.get("episodes")
        or any(
            not episode.get("verified_dom", {}).get("success") for episode in receipt["episodes"]
        )
    ):
        raise ValueError("Development collection identities or guarded outcomes are invalid")
    episodes = {}
    referenced_fixtures = set()
    for episode in receipt["episodes"]:
        episode_id = episode.get("episode_id")
        task_id = episode.get("task_id")
        prefix = re.escape(f"{split}-{receipt['seed']}-")
        match = re.fullmatch(
            prefix + r"([0-9]{4})-" + re.escape(str(task_id)) + r"-(fresh|recovery)",
            str(episode_id),
        )
        if (
            not match
            or episode_id in episodes
            or type(episode.get("steps")) is not int
            or episode["steps"] < 1
            or episode.get("verified_dom", {}).get("success") is not True
            or episode.get("verified_dom", {}).get("error") is not None
            or episode.get("scope") != "scripted training collection, no model score"
            or (task_id == "upstream_travel" and split != "train")
        ):
            raise ValueError("Collection episode identities are invalid")
        label = "fixtures/" + match[1] + ".html"
        if (
            label not in blobs
            or episode.get("fixture_path", label) != label
            or episode.get("fixture_sha256") != _sha(blobs[label])
            or episode.get("group_id") != "browser-development:" + _sha(blobs[label])
        ):
            raise ValueError("Collection fixture bytes differ from their episode identity")
        referenced_fixtures.add(label)
        episodes[episode_id] = episode
    if referenced_fixtures != {label for label in blobs if label.startswith("fixtures/")}:
        raise ValueError("Collection fixture inventory differs from its episode identities")
    expected_ids = [
        f"{episode_id}-step-{index:03d}"
        for episode_id, episode in episodes.items()
        for index in range(episode["steps"])
    ]
    if [case.id for case in cases] != expected_ids:
        raise ValueError("Collection cases differ from their ordered episode steps")
    for case, wire in zip(cases, wires, strict=True):
        episode = episodes[case.id.rsplit("-step-", 1)[0]]
        try:
            if (
                wire["case_id"] != case.id
                or wire["label_source"]
                != "scripted development teacher, fresh observed candidate IDs"
            ):
                raise ValueError("wire identity mismatch")
            goal = wire["payload"]["questions"]["operation"]["instructions"]["goal"]
            if not isinstance(goal, str) or _sha(goal.encode()) != episode["goal_sha256"]:
                raise ValueError("wire goal identity mismatch")
            rebuilt = training_case(
                wire["payload"],
                case_id=case.id,
                group_id=episode["group_id"],
                task_id=episode["task_id"],
                split=split,
                selected=wire["teacher_choice"],
                valid_targets=wire["valid_next_targets"],
            )
            # Candidate mappings are ordered: ordinary dict equality would hide
            # an altered answer-letter assignment despite identical label sets.
            if rebuilt.model_dump_json() != case.model_dump_json():
                raise ValueError("wire supervision mismatch")
            last = case.id == f"{episode['episode_id']}-step-{episode['steps'] - 1:03d}"
            if (case.gold["operation"] == "DONE") != last:
                raise ValueError("DONE must be the verified final episode step")
        except (KeyError, TypeError, ValueError, IndexError):
            raise ValueError(
                "Collection case does not match its native wire or verified episode"
            ) from None
    source_manifest = receipt.get("source_manifest")
    if source_manifest is not None:
        from s1.browser_evidence import _ALLOWED_CALLERS

        source_files = source_manifest.get("files", {})
        source_record = source_files.get("scripts/jev_browser_training_data.py", {})
        identity = _sha(
            json.dumps(
                source_files, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode()
        )
        if (
            source_manifest.get("schema_version") != 1
            or source_manifest.get("identity_sha256") != identity
            or source_record.get("sha256") != receipt["collector_sha256"]
            or receipt.get("source_verification", {}).get("success") is not True
            or receipt.get("upstream_verification", {}).get("success") is not True
            or "sources/manifest.json" not in blobs
            or _json(blobs["sources/manifest.json"]) != source_manifest
        ):
            raise ValueError("Collection prospective source identities are invalid")
        expected_source_paths = {"sources/manifest.json"}
        for label, record in source_files.items():
            relative = Path(label)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not (
                    label in _ALLOWED_CALLERS or (label.startswith("s1/") and label.endswith(".py"))
                )
                or record.get("artifact_path") != "sources/" + label
                or record["artifact_path"] not in blobs
                or record.get("sha256") != _sha(blobs[record["artifact_path"]])
                or record.get("size_bytes") != len(blobs[record["artifact_path"]])
            ):
                raise ValueError("Collection preserved source bytes differ from their identities")
            expected_source_paths.add(record["artifact_path"])
        if expected_source_paths != {label for label in blobs if label.startswith("sources/")}:
            raise ValueError("Collection preserved source inventory differs from its identity")
    elif any(label.startswith("sources/") for label in blobs):
        raise ValueError("Collection source evidence has no prospective receipt identity")
    return cases, receipt, {"root": root, "blobs": blobs, "manifest": _input_manifest(blobs)}


def _freeze_collection(snapshot, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for label, blob in snapshot["blobs"].items():
        target = destination / label
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
    (destination / "input_manifest.json").write_text(
        json.dumps(snapshot["manifest"], indent=2) + "\n"
    )
    return snapshot["manifest"]


def _verify_collection(snapshot):
    result = {"success": False, "changed": [], "missing": [], "added": []}
    try:
        paths = _input_paths(snapshot["root"])
        current = _input_manifest({label: path.read_bytes() for label, path in paths.items()})
    except (OSError, TypeError, ValueError) as exc:
        return {**result, "error": type(exc).__name__ + " while verifying collection inputs"}
    expected = snapshot["manifest"]["files"]
    actual = current["files"]
    changed = sorted(
        label for label in expected.keys() & actual.keys() if expected[label] != actual[label]
    )
    missing, added = (
        sorted(expected.keys() - actual.keys()),
        sorted(actual.keys() - expected.keys()),
    )
    return {
        "success": not changed and not missing and not added,
        "changed": changed,
        "missing": missing,
        "added": added,
        "identity_sha256": snapshot["manifest"]["identity_sha256"],
        "current_identity_sha256": current["identity_sha256"],
    }


def validated_collection(path, split, *, freeze_dir=None):
    cases, receipt, snapshot = _validated_snapshot(path, split)
    if freeze_dir is not None:
        _freeze_collection(snapshot, freeze_dir)
    return cases, receipt


def _audit_snapshots(collections):
    from s1.evaluation.datasets import request_fingerprint

    seen = {"id": {}, "group": {}, "request": {}}
    counts, overlaps = {}, []
    for cases in collections:
        for case in cases:
            counts[case.split] = counts.get(case.split, 0) + 1
            for kind, key in (
                ("id", case.id),
                ("group", case.group_id or case.id),
                ("request", request_fingerprint(case.request)),
            ):
                previous = seen[kind].setdefault(key, set())
                if previous - {case.split}:
                    overlaps.append(
                        {
                            "kind": kind,
                            "splits": sorted(previous | {case.split}),
                            "key_sha256": _sha(key.encode()),
                        }
                    )
                previous.add(case.split)
    return {"valid": not overlaps, "split_counts": counts, "overlaps": overlaps}


def _write_owned_readout(writer, destination, owned_artifacts, *args, **kwargs):
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Sidecar destination already exists; this fit does not own it")
    owned_artifacts.append(destination)
    try:
        return writer(destination, *args, **kwargs)
    except FileExistsError:
        # The native writer atomically creates its directory with exist_ok=False.
        # A competing creator owns that directory even after our initial check.
        owned_artifacts.remove(destination)
        raise


def _replay_paths(root):
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Replay inputs require an ordinary directory")
    paths = {label: root / label for label in ("features.pt", "receipt.json")}
    for directory in ("control", "sources"):
        folder = root / directory
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError("Replay control and source directories must be ordinary directories")
        for path in folder.rglob("*"):
            if path.is_symlink():
                raise ValueError("Replay inputs must not contain symlinks")
            if path.is_file():
                label = path.relative_to(root).as_posix()
                # Running a preserved caller may create this cache, which is
                # not an input or a manifest-recorded source file.
                if directory == "sources" and re.fullmatch(
                    r"sources/(?:[^/]+/)*__pycache__/[^/]+\.cpython-[0-9]+\.pyc", label
                ):
                    continue
                if directory == "control" and label not in {
                    "control/manifest.json",
                    "control/weights.pt",
                }:
                    raise ValueError("Unexpected replay control file")
                if (
                    directory == "sources"
                    and label != "sources/manifest.json"
                    and not label.endswith(".py")
                ):
                    raise ValueError("Unexpected replay source file")
                paths[label] = path
    if any(path.is_symlink() or not path.is_file() for path in paths.values()):
        raise ValueError("Replay inputs must be ordinary files")
    if not {"control/manifest.json", "control/weights.pt", "sources/manifest.json"} <= paths.keys():
        raise ValueError("Replay control or source evidence is incomplete")
    return paths


def _validated_replay(path, train_receipt, development_receipt):
    """Capture current bytes; old captures without a feature hash stay retrospective."""
    from s1.browser_evidence import _ALLOWED_CALLERS

    configured = Path(path)
    if configured.is_symlink() or not configured.is_dir():
        raise ValueError("--features-from requires an ordinary completed fit directory")
    root = configured.resolve()
    blobs = {label: source.read_bytes() for label, source in _replay_paths(root).items()}
    origin = _json(blobs["receipt.json"])
    if not isinstance(origin, dict):
        raise ValueError("Replay receipt must be an object")
    manifest = origin.get("source_manifest", {})
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise ValueError("Replay source manifest must record exact source files")
    records = manifest.get("files", {})
    identity = _sha(
        json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    )
    verification = origin.get("source_verification", {})
    if not isinstance(verification, dict):
        raise ValueError("Replay source verification must be an object")
    if not isinstance(origin.get("split_audit"), dict):
        raise ValueError("Replay split audit must be an object")
    if (
        origin.get("stage") != "complete"
        or origin.get("scope")
        != "development-only frozen-backbone readout fitting; not model task success"
        or origin.get("full_lm_head_selected_rows_unchanged") is not True
        or origin.get("train_collection") != train_receipt
        or origin.get("development_collection") != development_receipt
        or origin.get("split_audit", {}).get("valid") is not True
        or type(origin.get("objective_temperature")) is not float
        or origin["objective_temperature"] != 1.0
        or manifest.get("schema_version") != 1
        or not records
        or manifest.get("identity_sha256") != identity
        or _json(blobs["sources/manifest.json"]) != manifest
        or verification.get("success") is not True
        or verification.get("identity_sha256") != identity
        or verification.get("current_identity_sha256") != identity
        or any(verification.get(key) != [] for key in ("changed", "missing", "added"))
    ):
        raise ValueError(
            "Replay requires a complete source-verified fit of these exact collections"
        )
    expected = {"sources/manifest.json"}
    for label, record in records.items():
        relative = Path(label)
        artifact_path = "sources/" + label
        if (
            not isinstance(record, dict)
            or relative.is_absolute()
            or ".." in relative.parts
            or not (
                label in _ALLOWED_CALLERS or (label.startswith("s1/") and label.endswith(".py"))
            )
            or record.get("artifact_path") != artifact_path
            or artifact_path not in blobs
            or record.get("sha256") != _sha(blobs[artifact_path])
            or record.get("size_bytes") != len(blobs[artifact_path])
        ):
            raise ValueError("Replay preserved source bytes differ from their receipt")
        expected.add(artifact_path)
    if expected != {label for label in blobs if label.startswith("sources/")}:
        raise ValueError("Replay preserved source inventory differs from its receipt")
    if "features_sha256" in origin and origin["features_sha256"] != _sha(blobs["features.pt"]):
        raise ValueError("Replay features differ from their prospective capture hash")
    return {"root": root, "blobs": blobs, "manifest": _input_manifest(blobs), "receipt": origin}


def _verify_replay(snapshot):
    try:
        actual = _input_manifest(
            {label: path.read_bytes() for label, path in _replay_paths(snapshot["root"]).items()}
        )
        expected, current = snapshot["manifest"]["files"], actual["files"]
        changed = sorted(
            label for label in expected.keys() & current.keys() if expected[label] != current[label]
        )
        missing, added = (
            sorted(expected.keys() - current.keys()),
            sorted(current.keys() - expected.keys()),
        )
        return {
            "success": not changed and not missing and not added,
            "changed": changed,
            "missing": missing,
            "added": added,
            "identity_sha256": snapshot["manifest"]["identity_sha256"],
            "current_identity_sha256": actual["identity_sha256"],
        }
    except (OSError, TypeError, ValueError) as exc:
        return {"success": False, "error": type(exc).__name__ + " while verifying replay bytes"}


def _replay_descriptors(snapshot, frozen_root):
    """No backbone: real local 52-row training matrix, meta-only artifact pin descriptor."""
    import torch

    from s1.browser_readout import load_readout

    control = _json(snapshot["blobs"]["control/manifest.json"])
    if not isinstance(control, dict):
        raise ValueError("Replay control manifest must be an object")
    origin = snapshot["receipt"]
    if not isinstance(origin.get("quantization_details"), dict) or not isinstance(
        control.get("provenance"), dict
    ):
        raise ValueError("Replay quantization and provenance must be objects")
    ids, hidden_size = control.get("candidate_ids"), control.get("hidden_size")
    if (
        not isinstance(ids, list)
        or len(ids) != 52
        or any(type(index) is not int or not 0 <= index < 10_000_000 for index in ids)
        or len(set(ids)) != 52
        or type(hidden_size) is not int
        or not 1 <= hidden_size <= 65_536
        or control.get("kind") != "untrained_fp32_selected_rows_control"
        or origin.get("model") != "knowlet/Gemma-4-12B-Unified-System-One"
        or origin.get("revision") != "a66f836b56605039fe040f330180e336d19b3362"
        or origin.get("prompt_variant") != "grounded_user_question_v1"
        or control.get("base_model_id") != origin["model"]
        or control.get("base_revision") != origin["revision"]
        or control.get("prompt_variant") != origin["prompt_variant"]
        or control.get("temperature") != origin.get("deployment_temperature")
        or control.get("quantization") != "nf4"
        or origin.get("quantization_details", {}).get("mode") != "nf4"
        or any(
            control.get("provenance", {}).get(split)
            != {"ids": [], "groups": [], "request_fingerprints": []}
            for split in ("train", "calibration")
        )
    ):
        raise ValueError("Replay requires the unchanged pinned grounded NF4 untrained control")
    pins = SimpleNamespace(
        name=origin["model"],
        revision=origin["revision"],
        quantization="nf4",
        temperature=control["temperature"],
        softcap=control.get("softcap"),
        letters=torch.tensor(ids, dtype=torch.long),
        head=SimpleNamespace(weight=torch.empty((max(ids) + 1, hidden_size), device="meta")),
    )
    # The strict production loader validates exact hash, schema, shapes, finite
    # FP32 rows, provenance and all deployment pins against the frozen copy.
    control_root = Path(frozen_root) / "control"
    if any(
        (control_root / label).read_bytes() != snapshot["blobs"]["control/" + label]
        for label in ("manifest.json", "weights.pt")
    ):
        raise ValueError("Frozen control bytes differ from the replay snapshot")
    load_readout(control_root, pins, origin["prompt_variant"])
    payload = snapshot["blobs"]["control/weights.pt"]
    if _sha(payload) != control.get("weights_sha256"):
        raise ValueError("Captured control weights SHA256 differs from its manifest")
    try:
        rows = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("Captured control weights must be safe CPU tensors") from exc
    if (
        not isinstance(rows, dict)
        or set(rows) != {"weight", "bias"}
        or any(
            not torch.is_tensor(value)
            or value.dtype != torch.float32
            or value.device.type != "cpu"
            or not torch.isfinite(value).all().item()
            for value in rows.values()
        )
        or tuple(rows["weight"].shape) != (52, hidden_size)
        or tuple(rows["bias"].shape) != (52,)
    ):
        raise ValueError("Captured control weights must be finite FP32 selected rows")
    training = SimpleNamespace(
        head=SimpleNamespace(weight=rows["weight"].clone(), bias=rows["bias"].clone()),
        letters=torch.arange(52),
        softcap=pins.softcap,
    )
    return pins, training


def _validated_feature_rows(snapshot, collections, hidden_size):
    import torch

    try:
        data = torch.load(
            io.BytesIO(snapshot["blobs"]["features.pt"]), map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise ValueError("Replay features must load as CPU tensors with weights_only=True") from exc
    if not isinstance(data, dict) or set(data) != {"train", "development"}:
        raise ValueError("Replay features need exactly train and development rows")
    keys = {
        "case_id",
        "group_id",
        "split",
        "type",
        "question_id",
        "labels",
        "target",
        "logits",
        "hidden",
        "sequence_tokens",
    }
    for split, cases in collections.items():
        expected = [(case, question) for case in cases for question in case.request.questions]
        rows = data[split]
        if (
            not isinstance(rows, list)
            or len(rows) != len(expected)
            or type(snapshot["receipt"].get(split + "_feature_rows")) is not int
            or snapshot["receipt"][split + "_feature_rows"] != len(expected)
        ):
            raise ValueError("Replay feature counts differ from the ordered collection questions")
        for row, (case, question) in zip(rows, expected, strict=True):
            labels = question.labels()
            target = [case.soft_gold[question.id][label] for label in labels]
            if (
                not isinstance(row, dict)
                or set(row) != keys
                or any(
                    type(row.get(key)) is not type(value) or row.get(key) != value
                    for key, value in {
                        "case_id": case.id,
                        "group_id": case.group_id,
                        "split": case.split,
                        "type": question.type,
                        "question_id": question.id,
                        "labels": labels,
                    }.items()
                )
                or not isinstance(row.get("target"), list)
                or len(row["target"]) != len(target)
                or any(
                    type(value) not in (int, float) or not math.isfinite(value)
                    for value in row["target"]
                )
                or row["target"] != target
                or not 1 <= len(labels) <= 52
                or not isinstance(row.get("logits"), list)
                or len(row["logits"]) != len(labels)
                or any(
                    type(value) not in (int, float) or not math.isfinite(value)
                    for value in row["logits"]
                )
                or type(row.get("sequence_tokens")) is not int
                or not 1 <= row["sequence_tokens"] <= 8192
                or not torch.is_tensor(row.get("hidden"))
                or row["hidden"].dtype != torch.float32
                or row["hidden"].device.type != "cpu"
                or tuple(row["hidden"].shape) != (hidden_size,)
                or not torch.isfinite(row["hidden"]).all().item()
            ):
                raise ValueError(
                    "Replay feature row differs from its ordered case/question/targets or finite hidden state"
                )
    return data["train"], data["development"]


def _finalize_fit(
    output,
    receipt,
    *,
    source_manifest,
    source_files,
    input_snapshots,
    started,
    primary_error=None,
    owned_artifacts=(),
    replay_snapshot=None,
):
    """Write the failure receipt before surfacing final integrity failures."""
    from s1.browser_evidence import verify_runtime_sources

    failures = []
    if replay_snapshot is not None:
        receipt["feature_replay_verification"] = {}
        for name, root in (
            ("original", replay_snapshot["root"]),
            ("frozen", Path(output) / "feature-origin"),
        ):
            try:
                verification = _verify_replay({**replay_snapshot, "root": root})
            except Exception as exc:
                verification = {
                    "success": False,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            if name == "frozen":
                try:
                    if (
                        _sha((root / "input_manifest.json").read_bytes())
                        != receipt["feature_replay"]["manifest_sha256"]
                    ):
                        verification.update(success=False, manifest_changed=True)
                except (OSError, KeyError):
                    verification.update(success=False, manifest_missing=True)
            receipt["feature_replay_verification"][name] = verification
            if not verification["success"]:
                failures.append(
                    f"{name.capitalize()} replay bytes changed or could not be verified"
                )
        emitted = {"success": True, "changed": [], "missing": []}
        for label in ("features.pt", "control/manifest.json", "control/weights.pt"):
            try:
                path = Path(output) / label
                if path.is_symlink() or path.read_bytes() != replay_snapshot["blobs"][label]:
                    emitted["changed"].append(label)
            except OSError:
                emitted["missing"].append(label)
        emitted["success"] = not emitted["changed"] and not emitted["missing"]
        receipt["feature_replay_verification"]["emitted_copies"] = emitted
        if not emitted["success"]:
            failures.append("Emitted replay bytes changed or could not be verified")
    if source_manifest is not None:
        try:
            verification = verify_runtime_sources(source_manifest, additional_files=source_files)
        except Exception as exc:
            verification = {
                "success": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        receipt["source_verification"] = verification
        if not verification.get("success"):
            failures.append("Training runtime source changed or could not be verified")
    receipt["input_verification"] = {}
    for split, snapshot in input_snapshots.items():
        try:
            verification = _verify_collection(snapshot)
        except Exception as exc:
            verification = {
                "success": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        receipt["input_verification"][split] = verification
        if not verification.get("success"):
            failures.append(f"Original {split} collection changed or could not be verified")
    if "input_manifests" in receipt:
        receipt["frozen_input_verification"] = {}
        for split, record in receipt["input_manifests"].items():
            try:
                frozen_root = Path(output) / "inputs" / split
                frozen_snapshot = {**input_snapshots[split], "root": frozen_root}
                verification = _verify_collection(frozen_snapshot)
                if record.get("manifest_sha256") != _sha(
                    (frozen_root / "input_manifest.json").read_bytes()
                ):
                    verification.update(success=False, manifest_changed=True)
            except Exception as exc:
                verification = {
                    "success": False,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            receipt["frozen_input_verification"][split] = verification
            if not verification.get("success"):
                failures.append(f"Frozen {split} collection changed or could not be verified")
    if failures:
        receipt.update(stage="failed", integrity_errors=failures)
    if receipt["stage"] == "failed":
        invalidated, invalidation_errors = [], []
        output = Path(output).resolve()
        for configured in owned_artifacts:
            artifact = Path(configured)
            try:
                if artifact.is_symlink() or artifact.resolve().parent != output:
                    raise ValueError("Sidecar invalidation path is outside this owned run")
                manifest_path = artifact / "manifest.json"
                invalid_path = artifact / "manifest.invalid.json"
                if manifest_path.exists() or manifest_path.is_symlink():
                    if invalid_path.exists() or invalid_path.is_symlink():
                        raise ValueError("Invalid sidecar evidence already exists")
                    manifest_path.rename(invalid_path)
                    invalidated.append(artifact.name + "/manifest.invalid.json")
            except Exception as exc:
                invalidation_errors.append(
                    {"artifact": artifact.name, "type": type(exc).__name__, "message": str(exc)}
                )
        receipt["invalidated_artifacts"] = invalidated
        if invalidation_errors:
            receipt["invalidation_errors"] = invalidation_errors
            failures.append("Could not invalidate a task-owned sidecar manifest")
    receipt["elapsed_ms"] = (time.perf_counter() - started) * 1000
    try:
        (Path(output) / "receipt.json").write_text(
            json.dumps(receipt, indent=2, allow_nan=False) + "\n"
        )
    except Exception as exc:
        if primary_error is not None:
            primary_error.add_note("Failed to write final training receipt: " + type(exc).__name__)
            return
        raise
    if failures and primary_error is None:
        raise RuntimeError("; ".join(failures))


def _feature_tensor_sha(inputs):
    return _sha(
        json.dumps(
            {
                key: {"dtype": str(value.dtype), "values": value.tolist()}
                for key, value in sorted(inputs.items())
            },
            separators=(",", ":"),
        ).encode()
    )


def _feature_resources(model):
    import resource

    import torch

    result = {
        "process_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "scope": "process_high_water_not_capture_delta",
    }
    if torch.device(model.device).type == "cuda":
        result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(model.device)
    return result


def _feature_prepared(backend, record):
    import torch

    from s1.evaluation.gemma import SEQUENCE_KEYS

    inputs, length = backend._prepare_question(record["case"].request, record["question"])
    if (
        not isinstance(inputs, dict)
        or not {"input_ids", "attention_mask"} <= inputs.keys()
        or set(inputs) - SEQUENCE_KEYS
    ):
        raise ValueError("Feature capture requires explicit text-only processor sequence tensors")
    ids = inputs["input_ids"]
    if (
        not torch.is_tensor(ids)
        or ids.dtype != torch.long
        or ids.ndim != 2
        or ids.shape[0] != 1
        or type(length) is not int
        or length != ids.shape[1]
        or not 1 <= length <= backend.model.max_context
    ):
        raise ValueError(
            "Feature capture requires a complete original context within its token budget"
        )
    for value in inputs.values():
        if (
            not torch.is_tensor(value)
            or value.device.type != "cpu"
            or tuple(value.shape) != (1, length)
        ):
            raise ValueError("Feature capture processor tensors must be aligned on CPU")
    mask = inputs["attention_mask"][0]
    if not ((mask == 0) | (mask == 1)).all() or not (mask[1:] <= mask[:-1]).all() or not mask.any():
        raise ValueError("Feature capture requires a nonempty right-padding attention mask")
    if "mm_token_type_ids" in inputs and (inputs["mm_token_type_ids"] != 0).any():
        raise ValueError("Feature capture supports text-only modality IDs")
    config = getattr(backend.model.backbone, "config", None)
    if config is not None:
        text_config = config.get_text_config()
        if (ids < 0).any() or (ids >= text_config.vocab_size).any():
            raise ValueError("Feature capture token IDs exceed the native vocabulary")
        for name in ("image_token_id", "audio_token_id", "video_token_id"):
            token = getattr(config, name, None)
            if token is not None and (ids == token).any():
                raise ValueError("Feature capture does not accept media placeholder tokens")
    slot = int(mask.sum().item()) - 1
    return inputs, torch.tensor([slot]), torch.tensor([len(record["question"].labels())])


def feature_rows(backend, cases, *, feature_batch_size=1, evidence=None):
    """Capture only real frozen-backbone rows; optional two-row native forwards.

    Every bounded chunk is prepared and checked on CPU before any of its rows
    reaches the model device. Original processor keys and masks are preserved;
    incompatible valid rows use separate forwards. Output annotations keep the
    historical feature archive format. ``evidence`` receives a JSON-safe ledger.
    """
    if type(feature_batch_size) is not int or feature_batch_size not in (1, 2):
        raise ValueError("feature_batch_size must be the integer 1 or 2")
    if evidence is not None and not isinstance(evidence, dict):
        raise ValueError("feature capture evidence must be a dictionary")
    import torch

    from s1.contracts import validate_distribution
    from s1.evaluation.gemma import _bucket_key, _collate
    from s1.unified import project_candidates

    trace = {} if evidence is None else evidence
    trace.update(
        schema_version=1,
        status="validation",
        feature_batch_size=feature_batch_size,
        preparation="exact backend._prepare_question; frozen native backbone; no generation",
        model=backend.model.name,
        revision=getattr(backend.model, "revision", None),
        prompt_variant=getattr(backend, "prompt_variant", None),
        row_records=[],
        forwards=[],
        forward_calls=0,
        batch_sizes=[],
        logical_input_tokens=0,
        original_input_tokens=0,
        physical_input_tokens=0,
        padding_input_tokens=0,
        original_padding_input_tokens=0,
        collation_padding_input_tokens=0,
        completed_row_indices=[],
    )
    started = time.perf_counter()
    rows = []
    flat = []
    try:
        trace["peak_resources_before"] = _feature_resources(backend.model)
        for case in cases:
            if case.request.media:
                raise ValueError(
                    "Browser feature training is text-only; media requests are unsupported"
                )
            for question in case.request.questions:
                labels = question.labels()
                if not case.soft_gold or question.id not in case.soft_gold:
                    raise ValueError(
                        "Feature capture requires explicit complete soft training targets"
                    )
                targets = [case.soft_gold[question.id][label] for label in labels]
                validate_distribution(targets, len(labels))
                flat.append({"case": case, "question": question, "targets": targets})
                trace["row_records"].append(
                    {
                        "row_index": len(flat) - 1,
                        "case_id": case.id,
                        "question_id": question.id,
                        "status": "pending",
                    }
                )
        if not flat:
            raise ValueError("Feature capture requires at least one training question")
        backend.model.lm.eval()
        with torch.inference_mode():
            for start in range(0, len(flat), feature_batch_size):
                indices = list(range(start, min(start + feature_batch_size, len(flat))))
                prepared = []
                for index in indices:
                    trace.update(status="preparation", active_row_index=index)
                    item = _feature_prepared(backend, flat[index])
                    prepared.append(item)
                    trace["row_records"][index].update(
                        status="prepared",
                        sequence_tokens=item[0]["input_ids"].shape[1],
                        attended_tokens=int(item[0]["attention_mask"].sum().item()),
                        answer_slot=item[1].item(),
                        option_count=item[2].item(),
                        prepared_sha256=_feature_tensor_sha(item[0]),
                    )
                # _collate intentionally assumes compatible keys/dtypes. A
                # different valid optional tensor set must not disappear.
                groups = [list(range(len(indices)))]
                if len(prepared) == 2 and _bucket_key(prepared[0][0]) != _bucket_key(
                    prepared[1][0]
                ):
                    groups = [[0], [1]]
                for group in groups:
                    selected = [prepared[offset] for offset in group]
                    row_indices = [indices[offset] for offset in group]
                    pad_id = getattr(backend.model.tok, "pad_token_id", None)
                    config = getattr(backend.model.backbone, "config", None)
                    vocabulary = config.get_text_config().vocab_size if config is not None else None
                    if len(selected) > 1 and (
                        type(pad_id) is not int
                        or pad_id < 0
                        or (vocabulary is not None and pad_id >= vocabulary)
                    ):
                        raise ValueError("Feature batching requires the native padding token")
                    cpu_inputs = (
                        selected[0][0] if len(selected) == 1 else _collate(selected, pad_id)
                    )
                    width = cpu_inputs["input_ids"].shape[1]
                    original = sum(item[0]["input_ids"].shape[1] for item in selected)
                    logical = sum(int(item[0]["attention_mask"].sum().item()) for item in selected)
                    physical = cpu_inputs["input_ids"].numel()
                    forward = {
                        "forward_index": len(trace["forwards"]),
                        "row_indices": row_indices,
                        "case_ids": [flat[index]["case"].id for index in row_indices],
                        "question_ids": [flat[index]["question"].id for index in row_indices],
                        "batch_size": len(selected),
                        "original_lengths": [item[0]["input_ids"].shape[1] for item in selected],
                        "answer_slots": [item[1].item() for item in selected],
                        "sequence_width": width,
                        "logical_input_tokens": logical,
                        "original_input_tokens": original,
                        "physical_input_tokens": physical,
                        "padding_input_tokens": physical - logical,
                        "original_padding_input_tokens": original - logical,
                        "collation_padding_input_tokens": physical - original,
                        "input_sha256": _feature_tensor_sha(cpu_inputs),
                        "status": "started",
                        "latency_scope": "host_backbone_call_without_device_synchronize",
                    }
                    trace.update(
                        status="transfer",
                        active_chunk=forward,
                        active_forward_index=forward["forward_index"],
                    )
                    inputs = {
                        key: value.to(backend.model.device) for key, value in cpu_inputs.items()
                    }
                    trace["forwards"].append(forward)
                    trace["status"] = "forward"
                    began = time.perf_counter()
                    try:
                        output = backend.model.backbone(**inputs, use_cache=False, return_dict=True)
                        forward["status"] = "returned"
                    except BaseException as exc:
                        forward.update(
                            status="failed", error={"type": type(exc).__name__, "message": str(exc)}
                        )
                        raise
                    finally:
                        forward["elapsed_ms"] = (time.perf_counter() - began) * 1000
                    trace["status"] = "readout"
                    slots = torch.cat([item[1] for item in selected]).to(backend.model.device)
                    counts = torch.cat([item[2] for item in selected]).to(backend.model.device)
                    hidden = output.last_hidden_state[
                        torch.arange(len(selected), device=backend.model.device), slots
                    ]
                    logits = project_candidates(
                        hidden,
                        backend.model.head,
                        backend.model.letters,
                        counts,
                        backend.model.softcap,
                    )
                    if (
                        not torch.is_tensor(logits)
                        or not logits.is_floating_point()
                        or tuple(logits.shape) != (len(selected), len(backend.model.letters))
                        or not torch.isfinite(hidden).all()
                        or not torch.isfinite(
                            logits.masked_fill(
                                torch.arange(len(backend.model.letters), device=logits.device)[
                                    None, :
                                ]
                                >= counts[:, None],
                                0,
                            )
                        ).all()
                    ):
                        raise ValueError(
                            "Feature capture returned nonfinite hidden states or candidate logits"
                        )
                    hidden_cpu, logits_cpu = hidden.float().cpu(), logits.float().cpu()
                    trace["status"] = "capture"
                    for batch_index, index in enumerate(row_indices):
                        case, question = flat[index]["case"], flat[index]["question"]
                        rows.append(
                            {
                                "case_id": case.id,
                                "group_id": case.group_id,
                                "split": case.split,
                                "type": question.type,
                                "question_id": question.id,
                                "labels": question.labels(),
                                "target": flat[index]["targets"],
                                "logits": logits_cpu[
                                    batch_index, : len(question.labels())
                                ].tolist(),
                                "hidden": hidden_cpu[batch_index].clone(),
                                "sequence_tokens": selected[batch_index][0]["input_ids"].shape[1],
                            }
                        )
                        trace["row_records"][index]["status"] = "captured"
                    forward["status"] = "completed"
                    # Keep at most the current native chunk on the model device.
                    del output, hidden, inputs, slots, counts, logits, hidden_cpu, logits_cpu
                if (start + len(indices)) % 32 == 0:
                    print(
                        f"Captured {len(rows)}/{len(flat)} actual development question rows",
                        flush=True,
                    )
        trace["status"] = "completed"
    except BaseException as exc:
        trace.update(
            failure_stage=trace["status"],
            status="failed",
            error={"type": type(exc).__name__, "message": str(exc)},
        )
        exc.feature_capture_evidence = trace
        exc.completed_feature_rows = rows
        raise
    finally:
        trace.update(
            elapsed_ms=(time.perf_counter() - started) * 1000,
            latency_scope="whole_feature_capture_including_cpu_readout_copy",
            forward_calls=len(trace["forwards"]),
            batch_sizes=[item["batch_size"] for item in trace["forwards"]],
            logical_input_tokens=sum(item["logical_input_tokens"] for item in trace["forwards"]),
            original_input_tokens=sum(item["original_input_tokens"] for item in trace["forwards"]),
            physical_input_tokens=sum(item["physical_input_tokens"] for item in trace["forwards"]),
            padding_input_tokens=sum(item["padding_input_tokens"] for item in trace["forwards"]),
            original_padding_input_tokens=sum(
                item["original_padding_input_tokens"] for item in trace["forwards"]
            ),
            collation_padding_input_tokens=sum(
                item["collation_padding_input_tokens"] for item in trace["forwards"]
            ),
            logical_token_scope="original attended tokens; original widths retained separately",
            token_accounting_scope="issued_backbone_forwards_including_failed_unknown_completion",
            completed_row_indices=list(range(len(rows))),
        )
        try:
            trace["peak_resources"] = _feature_resources(backend.model)
        except Exception as exc:
            trace["resource_measurement_error"] = type(exc).__name__
    return rows


def fit(
    *,
    train_cases,
    development_cases,
    output_dir,
    steps=128,
    learning_rate=1e-3,
    seed=42,
    features_from=None,
    feature_batch_size=1,
):
    if type(feature_batch_size) is not int or feature_batch_size not in (1, 2):
        raise ValueError("feature_batch_size must be the integer 1 or 2")
    if features_from is not None and feature_batch_size != 1:
        raise ValueError("feature_batch_size=2 does not apply to CPU feature replay")
    if (
        type(steps) is not int
        or steps < 1
        or not math.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError("positive steps and finite learning rate required")
    from s1.evaluation.datasets import audit_splits, request_fingerprint

    train, train_receipt, train_snapshot = _validated_snapshot(train_cases, "train")
    development, development_receipt, development_snapshot = _validated_snapshot(
        development_cases, "development"
    )
    # Detect overlap before creating a run, using exactly the validated bytes.
    audit = _audit_snapshots([train, development])
    if not audit["valid"]:
        raise ValueError("Training and development episode/source/request groups overlap")
    replay_snapshot = (
        _validated_replay(features_from, train_receipt, development_receipt)
        if features_from is not None
        else None
    )
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "stage": "setup",
        "scope": "development-only frozen-backbone readout fitting; not model task success",
        "steps": steps,
        "learning_rate": learning_rate,
        "seed": seed,
        "feature_batch_size": feature_batch_size,
        "split_audit": audit,
        "train_collection": train_receipt,
        "development_collection": development_receipt,
    }
    started = time.perf_counter()
    manifest = None
    primary_error = None
    owned_artifacts = []
    train_rows, development_rows = None, None
    active_capture_split = None
    input_snapshots = {"train": train_snapshot, "development": development_snapshot}
    source_files = {
        "scripts/jev_browser_fit_readout.py": Path(__file__),
        "scripts/breakthrough_training.py": Path(fit_head.__code__.co_filename),
        "scripts/jev_browser_training_data.py": Path(training_case.__code__.co_filename),
    }
    try:
        from jev_browser_benchmark import make_cases

        from s1.browser_evidence import capture_runtime_sources

        source_files["scripts/jev_browser_benchmark.py"] = Path(make_cases.__code__.co_filename)
        manifest = capture_runtime_sources(output, additional_files=source_files)
        receipt["source_manifest"] = manifest
        receipt["input_manifests"] = {}
        for split, snapshot in input_snapshots.items():
            destination = output / "inputs" / split
            receipt["input_manifests"][split] = {
                **_freeze_collection(snapshot, destination),
                "manifest_sha256": _sha((destination / "input_manifest.json").read_bytes()),
                "artifact_path": "inputs/" + split,
                "preservation_timing": "validated bytes frozen before model loading or feature extraction",
            }
        audit = audit_splits(
            [output / "inputs" / split / "cases.jsonl" for split in input_snapshots]
        )
        receipt["split_audit"] = audit
        receipt["split_audit_paths"] = [
            "inputs/" + split + "/cases.jsonl" for split in input_snapshots
        ]
        if not audit["valid"]:
            raise ValueError(
                "Frozen training and development episode/source/request groups overlap"
            )

        import torch

        from s1.browser_readout import write_readout

        if replay_snapshot is None:
            from s1.browser_policy import BrowserDecisionBackend
            from s1.unified import UnifiedDecisionModel

            torch.backends.cuda.enable_cudnn_sdp(False)
            model = UnifiedDecisionModel(
                "knowlet/Gemma-4-12B-Unified-System-One",
                revision="a66f836b56605039fe040f330180e336d19b3362",
                device="cuda",
                precision="bfloat16",
                quantization="nf4",
                max_context=8192,
                attn_implementation="sdpa",
            )
            if model.revision != "a66f836b56605039fe040f330180e336d19b3362":
                raise ValueError("Loaded model differs from the immutable training checkpoint")
            backend = BrowserDecisionBackend(model, prompt_mode="grounded")
            prompt_variant, training_model = backend.prompt_variant, model
            initial_rows = model.head.weight.detach().index_select(0, model.letters).cpu().clone()
            receipt.update(
                stage="feature_capture",
                model=model.name,
                revision=model.revision,
                prompt_variant=prompt_variant,
                quantization_details=model.quantization_details,
                deployment_temperature=model.temperature,
                objective_temperature=1.0,
                execution_mode="genuine_frozen_backbone_feature_capture",
                feature_capture={
                    "schema_version": 1,
                    "feature_batch_size": feature_batch_size,
                    "source_identity_sha256": manifest["identity_sha256"],
                    "split_captures": {},
                },
            )
            (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            for split, cases in (("train", train), ("development", development)):
                active_capture_split = split
                capture = {"source_identity_sha256": manifest["identity_sha256"], "split": split}
                receipt["feature_capture"]["split_captures"][split] = capture
                captured = feature_rows(
                    backend, cases, feature_batch_size=feature_batch_size, evidence=capture
                )
                if split == "train":
                    train_rows = captured
                else:
                    development_rows = captured
                (output / "feature_capture.json").write_text(
                    json.dumps(receipt["feature_capture"], indent=2, allow_nan=False) + "\n"
                )
            active_capture_split = None
            captures = receipt["feature_capture"]["split_captures"].values()
            receipt["backbone_forward_calls_this_run"] = sum(
                item.get("forward_calls", 0) for item in captures
            )
            receipt["feature_capture_sha256"] = _sha((output / "feature_capture.json").read_bytes())
            torch.save(
                {"train": train_rows, "development": development_rows}, output / "features.pt"
            )
        else:
            origin_root = output / "feature-origin"
            _freeze_collection(replay_snapshot, origin_root)
            origin = replay_snapshot["receipt"]
            receipt["feature_replay"] = {
                **replay_snapshot["manifest"],
                "artifact_path": "feature-origin",
                "manifest_sha256": _sha((origin_root / "input_manifest.json").read_bytes()),
                "preservation_timing": "replay input bytes frozen before CPU fitting",
                "capture_sha_status": "retrospective_replay_freeze",
                "origin_feature_sha_was_prospective": "features_sha256" in origin,
                "limitation": "This replay freeze does not authenticate previously unhashed feature or control-manifest bytes at their original capture time.",
                "origin_source_identity_sha256": origin["source_manifest"]["identity_sha256"],
                "training_candidate_indexing": "local 52-row positions in the pinned original candidate-token order",
            }
            model, training_model = _replay_descriptors(replay_snapshot, origin_root)
            initial_rows = (
                training_model.head.weight.detach()
                .index_select(0, training_model.letters)
                .cpu()
                .clone()
            )
            prompt_variant = origin["prompt_variant"]
            train_rows, development_rows = _validated_feature_rows(
                replay_snapshot,
                {"train": train, "development": development},
                model.head.weight.shape[1],
            )
            # Preserve the exact tensor archive; no new backbone output is generated.
            (output / "features.pt").write_bytes(replay_snapshot["blobs"]["features.pt"])
            receipt.update(
                model=model.name,
                revision=model.revision,
                prompt_variant=prompt_variant,
                quantization_details=origin["quantization_details"],
                deployment_temperature=model.temperature,
                objective_temperature=1.0,
                execution_mode="cpu_only_frozen_feature_replay",
                backbone_forward_calls_this_run=0,
                full_lm_head_loaded_this_run=False,
                original_head_verification_scope="original capture receipt; full LM head not loaded by this replay",
            )
        receipt.update(
            stage="fit",
            train_feature_rows=len(train_rows),
            development_feature_rows=len(development_rows),
            features_sha256=_sha((output / "features.pt").read_bytes()),
        )

        def inventory(cases):
            return {
                "ids": sorted({case.id for case in cases}),
                "groups": sorted({case.group_id or case.id for case in cases}),
                "request_fingerprints": sorted(
                    {request_fingerprint(case.request) for case in cases}
                ),
            }

        empty = {"ids": [], "groups": [], "request_fingerprints": []}
        control_provenance = {
            "source_scope": "development_only",
            "train": empty,
            "calibration": empty,
        }
        if replay_snapshot is None:
            _write_owned_readout(
                write_readout,
                output / "control",
                owned_artifacts,
                model,
                prompt_variant,
                provenance=control_provenance,
            )
        else:

            def copy_control(destination):
                destination.mkdir(parents=True, exist_ok=False)
                for label in ("manifest.json", "weights.pt"):
                    (destination / label).write_bytes(replay_snapshot["blobs"]["control/" + label])

            _write_owned_readout(copy_control, output / "control", owned_artifacts)
        learned, history = fit_head(
            training_model, train_rows, steps=steps, seed=seed, learning_rate=learning_rate
        )
        provenance = {
            "source_scope": "development_only",
            "train": inventory(train),
            "calibration": empty,
            "training_objective": {
                "description": "soft CE + 0.1 Brier + 0.01 selected-row anchor",
                "objective_temperature": 1.0,
                "deployment_temperature": model.temperature,
                "steps": steps,
                "learning_rate": learning_rate,
                "seed": seed,
                "train_dataset_sha256": train_receipt["dataset_sha256"],
            },
        }
        if replay_snapshot is not None:
            provenance["training_objective"].update(
                feature_replay_identity_sha256=replay_snapshot["manifest"]["identity_sha256"],
                feature_sha256=receipt["features_sha256"],
                feature_capture_sha_status="retrospective_replay_freeze",
                backbone_forward_calls_this_run=0,
            )
        _write_owned_readout(
            write_readout,
            output / "learned",
            owned_artifacts,
            model,
            prompt_variant,
            weight=learned["weight"],
            bias=learned["bias"],
            provenance=provenance,
        )
        if not torch.equal(
            initial_rows,
            training_model.head.weight.detach().index_select(0, training_model.letters).cpu(),
        ):
            raise ValueError("Original LM-head rows changed during detached readout fitting")
        receipt.update(
            stage="complete",
            training=history,
            full_lm_head_selected_rows_unchanged=True,
            learned_artifact=str(output / "learned"),
            control_artifact=str(output / "control"),
        )
        if replay_snapshot is None:
            receipt["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        else:
            receipt["control_rows_unchanged"] = True
            receipt.pop("full_lm_head_selected_rows_unchanged", None)
            receipt["origin_full_lm_head_selected_rows_unchanged"] = True
    except BaseException as exc:
        primary_error = exc
        receipt.update(stage="failed", error={"type": type(exc).__name__, "message": str(exc)})
        if "feature_capture" in receipt:
            failed_capture = getattr(exc, "feature_capture_evidence", None)
            if failed_capture is not None and active_capture_split is not None:
                receipt["feature_capture"]["split_captures"][active_capture_split] = failed_capture
            captures = receipt["feature_capture"]["split_captures"].values()
            receipt["backbone_forward_calls_this_run"] = sum(
                item.get("forward_calls", 0) for item in captures
            )
            try:
                if failed_capture is not None:
                    if active_capture_split == "train":
                        train_rows = exc.completed_feature_rows
                    elif active_capture_split == "development":
                        development_rows = exc.completed_feature_rows
                partial_path = output / "features.partial.pt"
                torch.save(
                    {"train": train_rows or [], "development": development_rows or []}, partial_path
                )
                receipt["partial_feature_capture"] = {
                    "artifact_path": "features.partial.pt",
                    "sha256": _sha(partial_path.read_bytes()),
                    "train_rows": len(train_rows or []),
                    "development_rows": len(development_rows or []),
                    "scope": "actual completed CPU features from failed capture; not a complete fit input",
                }
                trace_path = output / "feature_capture.json"
                trace_path.write_text(
                    json.dumps(receipt["feature_capture"], indent=2, allow_nan=False) + "\n"
                )
                receipt["feature_capture_sha256"] = _sha(trace_path.read_bytes())
            except Exception as evidence_exc:
                receipt["partial_feature_capture_write_error"] = {
                    "type": type(evidence_exc).__name__,
                    "message": str(evidence_exc),
                }
                exc.add_note(
                    "Failed to preserve partial feature evidence: " + type(evidence_exc).__name__
                )
        raise
    finally:
        _finalize_fit(
            output,
            receipt,
            source_manifest=manifest,
            source_files=source_files,
            input_snapshots=input_snapshots,
            started=started,
            primary_error=primary_error,
            owned_artifacts=owned_artifacts,
            replay_snapshot=replay_snapshot,
        )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cases", required=True)
    parser.add_argument("--development-cases", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-batch-size", type=int, choices=[1, 2], default=1)
    parser.add_argument(
        "--features-from",
        help="Completed original fit directory; replay frozen features on CPU without loading the backbone",
    )
    receipt = fit(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in ["stage", "scope", "learned_artifact", "control_artifact"]
            }
        )
    )


if __name__ == "__main__":
    main()
