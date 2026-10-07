"""Freeze a BoolQ specialist release split from checksum-pinned local source bytes.

This command performs no network requests. Fresh calibration and test passages
are absent from the entire official training split and all prior sampled data.
The old text and media tests remain separate regression sets, never training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261002
BOOLQ_REVISION = "35b264d03638db9f4ce671b711558bf7ff0f80d5"
BOOLQ_UPSTREAM_REVISION = "90af34107399cc7a446b373dc4ee35b8001da7c2"
SOURCE_FILES = {
    "boolq-train.parquet": "4f028e992c0bd4df30b9f056f4946b64f5c23028034ff0ed5ea467d8538cc623",
    "boolq-validation.parquet": "52355d11524b4b874a9b9dcc278feb10f672d52c4f4eff9872e695ede59820f8",
    "boolq-README.md": "0afe70d7cdbb644637592212f0cb15339389910bf00f35764ec3c545257aa663",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def passage_group(passage):
    return "boolq-passage-" + sha256(passage.encode())


def rank(seed, purpose, value):
    return sha256(canonical([seed, purpose, value]))


def select_rows(rows, *, source_split, split, count, forbidden, seed):
    """Pick one stable row per passage, then equal numbers of true/false labels."""
    if count < 2 or count % 2:
        raise ValueError("split counts must be positive even numbers")
    groups = {}
    for index, row in enumerate(rows):
        if (
            type(row.get("answer")) is not bool
            or not isinstance(row.get("passage"), str)
            or not row["passage"].strip()
            or not isinstance(row.get("question"), str)
            or not row["question"].strip()
        ):
            raise ValueError("BoolQ rows require original boolean labels, passages and questions")
        group = passage_group(row["passage"])
        if group in forbidden:
            continue
        item = (index, row)
        priority = rank(seed, "group-representative", [source_split, index])
        if group not in groups or priority < groups[group][0]:
            groups[group] = (priority, item)
    selected = []
    for label in (False, True):
        candidates = [item for _, item in groups.values() if item[1]["answer"] is label]
        candidates.sort(key=lambda item: rank(seed, f"select:{split}:{label}", item[0]))
        if len(candidates) < count // 2:
            raise ValueError(f"insufficient disjoint {label} passage groups for {split}")
        selected.extend(candidates[: count // 2])
    selected.sort(key=lambda item: rank(seed, f"order:{split}", item[0]))
    return selected


def make_cases(selected, *, source_split, split):
    from s1.evaluation.datasets import EvaluationCase, request_fingerprint

    cases, provenance = [], []
    for index, row in selected:
        case = EvaluationCase.model_validate(
            {
                "id": f"boolq-{source_split}-{index}",
                "split": split,
                "group_id": passage_group(row["passage"]),
                "task_id": "boolq",
                "schema_id": "boolq",
                "language": "en",
                "request": {
                    "state": {"passage": row["passage"], "question": row["question"]},
                    "questions": [
                        {
                            "id": "answer",
                            "type": "noul",
                            "instructions": "Answer the question using the supplied passage.",
                        }
                    ],
                },
                "gold": {"answer": "true" if row["answer"] else "false"},
            }
        )
        cases.append(case)
        provenance.append(
            {
                "case_id": case.id,
                "split": split,
                "source_split": source_split,
                "source_row_index": index,
                "source_row_sha256": sha256(canonical(row)),
                "group_id": case.group_id,
                "request_sha256": request_fingerprint(case.request),
                "label": case.gold["answer"],
            }
        )
    return cases, provenance


def prior_groups(directory):
    """Include archived formats and every support selection, not just current files."""
    all_groups, protected_groups, files = set(), set(), []
    for path in sorted(directory.rglob("*.jsonl")):
        raw = path.read_bytes()
        relevant = False
        for line in raw.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            item = item.get("case", item)
            if not str(item.get("id", "")).startswith("boolq-"):
                continue
            state = item.get("request", {}).get("state")
            # v1 stored its passage under passage and its question in instructions.
            passage = state.get("passage") if isinstance(state, dict) else state
            if not isinstance(passage, str) or not passage.strip():
                raise ValueError(f"cannot resolve previous BoolQ passage: {path.name}")
            group = passage_group(passage)
            relevant = True
            all_groups.add(group)
            if item["split"] != "train":
                protected_groups.add(group)
        if relevant:
            files.append({"path": str(path.relative_to(directory)), "file_sha256": sha256(raw)})
    if not files:
        raise ValueError("prior dataset directory contains no sampled BoolQ records")
    return all_groups, protected_groups, files


def write_immutable(path, raw):
    if path.exists() and path.read_bytes() != raw:
        raise FileExistsError(f"refusing to replace a different frozen release file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(raw)


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def prepare(source_dir, prior_dir, output, *, seed=SEED, train=2048, calibration=256, test=256):
    import pyarrow.parquet as parquet

    from s1.evaluation.datasets import audit_splits, fingerprint, load_cases, request_fingerprint

    source_dir, prior_dir, output = map(Path, (source_dir, prior_dir, output))
    if output.resolve().is_relative_to(prior_dir.resolve()):
        raise ValueError("release output must be outside the prior dataset directory")
    for filename, expected in SOURCE_FILES.items():
        if sha256((source_dir / filename).read_bytes()) != expected:
            raise ValueError(f"cached source checksum mismatch: {filename}")
    original = {
        split: parquet.read_table(source_dir / f"boolq-{split}.parquet").to_pylist()
        for split in ("train", "validation")
    }
    previous, protected, prior_files = prior_groups(prior_dir)
    official_train_groups = {passage_group(row["passage"]) for row in original["train"]}
    forbidden = official_train_groups | previous
    datasets, selected_records = {}, []
    for split, count, source_split, excluded in (
        ("calibration", calibration, "validation", forbidden),
        ("test", test, "validation", forbidden),
        ("train", train, "train", protected),
    ):
        selected = select_rows(
            original[source_split],
            source_split=source_split,
            split=split,
            count=count,
            forbidden=excluded,
            seed=seed,
        )
        cases, records = make_cases(selected, source_split=source_split, split=split)
        datasets[f"{split}.jsonl"] = cases
        selected_records.extend(records)
        if split != "train":
            forbidden.update(case.group_id for case in cases)
    for destination, source in (
        ("regression-text.jsonl", "text-test.jsonl"),
        ("media-test.jsonl", "media-test.jsonl"),
    ):
        cases = load_cases(prior_dir / source)
        if any(case.split != "test" for case in cases):
            raise ValueError("regression inputs must remain labeled test")
        datasets[destination] = cases
    for filename, cases in datasets.items():
        raw = b"".join(case.model_dump_json().encode() + b"\n" for case in cases)
        write_immutable(output / filename, raw)
    # Publish the complete exclusion identities with the frozen data so split
    # independence can be audited without the original workspace's archives.
    exclusions = {
        "schema_version": 1,
        "group_definition": "boolq-passage- plus SHA-256 of the original UTF-8 passage",
        "official_train_source_sha256": SOURCE_FILES["boolq-train.parquet"],
        "all_prior_sampled_group_ids": sorted(previous),
        "prior_nontraining_group_ids": sorted(protected),
        "all_official_train_group_ids": sorted(official_train_groups),
    }
    write_immutable(output / "exclusions.json", json_bytes(exclusions))
    split_audit = audit_splits([output / name for name in datasets])
    if not split_audit["valid"]:
        raise ValueError("release datasets failed source-group/ID/request split audit")
    fresh = datasets["calibration.jsonl"] + datasets["test.jsonl"]
    fresh_groups = {case.group_id for case in fresh}
    if fresh_groups & (official_train_groups | previous):
        raise ValueError("fresh holdouts include official train or previously sampled passages")
    # State-only checks catch reused passage/question state even if prompts differ.
    states = {}
    for cases in datasets.values():
        for case in cases:
            key = sha256(canonical(case.request.state))
            if case.task_id == "boolq":
                previous_split = states.setdefault(key, case.split)
                if previous_split != case.split:
                    raise ValueError("release datasets contain identical states across splits")
    provenance = {
        "schema_version": 1,
        "sources": {
            "boolq": {
                "publisher": "Google Research; Christopher Clark et al. (2019)",
                "source_url": "https://huggingface.co/datasets/google/boolq",
                "source_revision": BOOLQ_REVISION,
                "upstream_revision": BOOLQ_UPSTREAM_REVISION,
                "license": "CC-BY-SA-3.0",
                "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
                "source_files": SOURCE_FILES,
                "raw_split_counts": {key: len(rows) for key, rows in original.items()},
            },
        },
        "selected_source_records": selected_records,
        "regression_provenance": json.loads((prior_dir / "provenance.json").read_text()),
    }
    write_immutable(output / "provenance.json", json_bytes(provenance))
    write_immutable(
        output / "BOOLQ_SOURCE_README.md", (source_dir / "boolq-README.md").read_bytes()
    )
    attribution = (
        "# Release dataset attribution\n\n"
        "This release adapts BoolQ by Christopher Clark, Kenton Lee, Ming-Wei Chang, "
        "Tom Kwiatkowski, Michael Collins and Kristina Toutanova (2019), Google Research. "
        "BoolQ and these transformed BoolQ dataset files are CC-BY-SA-3.0: "
        "https://creativecommons.org/licenses/by-sa/3.0/\n\n"
        "Original: https://github.com/google-research-datasets/boolean-questions\n\n"
        "Changes: deterministic passage-group sampling, balanced labels, separate "
        "train/calibration/test splits, and conversion to the System One NOUL request "
        "schema. Original questions, passages and boolean labels are preserved.\n\n"
        "The media regression file retains earlier MNIST (MIT) and FSDD "
        "(CC-BY-SA-4.0) samples. Authors, immutable source revisions, transformations "
        "and license links are preserved in provenance.json. It is not training data.\n\n"
        "These are specialist English reading-comprehension data, with balanced "
        "labels rather than natural deployment prevalence. No broad reasoning, "
        "multimodal training or foundation-model pretraining-disjointness claim is made.\n"
    )
    write_immutable(output / "ATTRIBUTION.md", attribution.encode())
    manifest = {
        "schema_version": 1,
        "task_scope": "boolq-specialist",
        "seed": seed,
        "algorithm": "sha256-group-representative-and-balanced-selection-v1",
        "preparation_script_sha256": sha256(Path(__file__).read_bytes()),
        "sources": provenance["sources"],
        "selection_policy": {
            "train": "official train; excludes all previous non-training passage groups",
            "calibration_test": "official validation; excludes every official train passage "
            "and every previously sampled passage; disjoint group representatives",
            "regression": "prior text and media tests retained without training use",
            "freshness_limit": "fresh relative to recorded local sampled data; upstream "
            "foundation-model pretraining overlap is unknown",
        },
        "prior_dataset_files": prior_files,
        "prior_dataset_directory": str(prior_dir),
        "exclusions_file": "exclusions.json",
        "excluded_groups": {
            "all_prior_sampled": len(previous),
            "prior_nontraining": len(protected),
            "all_official_train": len(official_train_groups),
        },
        "datasets": {},
        "split_audit": split_audit,
        "holdout_checks": {
            "official_train_passage_overlap": 0,
            "prior_sampled_passage_overlap": 0,
            "state_overlap_across_splits": 0,
        },
        "files": {},
    }
    for name, cases in datasets.items():
        manifest["datasets"][name] = {
            "split": cases[0].split,
            "cases": len(cases),
            "questions": sum(len(case.request.questions) for case in cases),
            "label_counts": dict(Counter(label for case in cases for label in case.gold.values())),
            "file_sha256": sha256((output / name).read_bytes()),
            "dataset_sha256": fingerprint(cases),
            "case_ids": [case.id for case in cases],
            "group_ids": [case.group_id or case.id for case in cases],
            "request_sha256s": [request_fingerprint(case.request) for case in cases],
        }
    for name in (
        *datasets,
        "provenance.json",
        "exclusions.json",
        "BOOLQ_SOURCE_README.md",
        "ATTRIBUTION.md",
    ):
        manifest["files"][name] = sha256((output / name).read_bytes())
    write_immutable(output / "manifest.json", json_bytes(manifest))
    return manifest


def main(argv=None):
    sys.path.insert(0, str(ROOT / "src"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prior-dir", type=Path, default=Path("artifacts/live-validation/datasets")
    )
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/release/datasets"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train", type=int, default=2048)
    parser.add_argument("--calibration", type=int, default=256)
    parser.add_argument("--test", type=int, default=256)
    args = parser.parse_args(argv)
    manifest = prepare(
        args.source_dir or args.prior_dir / "raw",
        args.prior_dir,
        args.output,
        seed=args.seed,
        train=args.train,
        calibration=args.calibration,
        test=args.test,
    )
    print(json.dumps({name: item["cases"] for name, item in manifest["datasets"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
