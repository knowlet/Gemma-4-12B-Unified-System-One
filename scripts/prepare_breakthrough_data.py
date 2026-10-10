#!/usr/bin/env python3
"""Freeze independent finite-world typed decisions, without benchmark input.

Targets describe known sampling distributions, not outcomes that have occurred.
The hard label is only the oracle's most likely category. Test template families
are reserved before generation. All answers and computation receipts remain
outside the inference payload. No downloads, model calls, or GPU work occur.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "typed-finite-worlds-v1"
SEED = 20261008
TYPES = ("choice", "noul", "score")
COUNTS = {"train": 1024, "calibration": 192, "test": 384}
TRAIN_FAMILIES = {
    "choice": ("choice_inventory", "choice_census"),
    "noul": ("noul_flag_table", "noul_colored_tiles"),
    "score": ("score_rating_counts", "score_quality_ledger"),
}
TEST_FAMILIES = {
    "choice": ("choice_route_manifest", "choice_partition_boxes"),
    "noul": ("noul_access_policy", "noul_eligibility_records"),
    "score": ("score_damage_ledger", "score_count_bands"),
}
CHOICE_LABELS = ("cedar", "coral", "amber", "violet", "birch", "silver", "indigo", "olive")
OPERATIONS = (
    "x",
    "not_x",
    "x_and_y",
    "x_or_y",
    "x_xor_y",
    "x_implies_y",
    "x_given_y",
    "not_x_given_not_y",
)
OPERATION_TEXT = {
    "x": "X holds",
    "not_x": "X does not hold",
    "x_and_y": "both X and Y hold",
    "x_or_y": "at least one of X or Y holds (inclusive OR)",
    "x_xor_y": "exactly one of X and Y holds",
    "x_implies_y": "the material implication X implies Y is true (false only for X and not Y)",
    "x_given_y": "X holds, conditional on Y holding",
    "not_x_given_not_y": "X does not hold, conditional on Y not holding",
}
LEVEL_SETS = ((0, 1, 2), (0, 1, 2, 3, 4), (0, 2, 5))


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def type_counts(total):
    if type(total) is not int or total < 3:
        raise ValueError("each split needs at least three cases")
    return {kind: total // 3 + int(i < total % 3) for i, kind in enumerate(TYPES)}


def oracle(world, operation=None):
    """Calculate exact integer masses before converting to probability floats."""
    if world["kind"] == "boolean_table":
        selected = []
        for (x, y), weight in zip(
            ((False, False), (False, True), (True, False), (True, True)),
            world["counts"],
            strict=True,
        ):
            if operation == "x_given_y" and not y:
                continue
            if operation == "not_x_given_not_y" and y:
                continue
            truth = {
                "x": x,
                "not_x": not x,
                "x_and_y": x and y,
                "x_or_y": x or y,
                "x_xor_y": x != y,
                "x_implies_y": not x or y,
                "x_given_y": x,
                "not_x_given_not_y": not x,
            }[operation]
            selected.append((truth, weight))
        labels = ["false", "true"]
        masses = [
            sum(weight for truth, weight in selected if truth is value) for value in (False, True)
        ]
    else:
        labels = [str(label) for label in world["labels"]]
        masses = list(world["counts"])
    denominator = sum(masses)
    if denominator <= 0:
        raise ValueError("sampling population must be nonempty")
    probabilities = dict(zip(labels, (mass / denominator for mass in masses), strict=True))
    best = max(labels, key=probabilities.__getitem__)
    return {
        "labels": labels,
        "integer_masses": masses,
        "denominator": denominator,
        "probabilities": probabilities,
        "most_likely_label": best,
        "hard_label_policy": "most likely only; ties use canonical world label order",
        "expected_score": sum(
            float(label) * probability for label, probability in probabilities.items()
        )
        if world["kind"] == "ordinal"
        else None,
    }


def make_candidate(split, kind, index, rng):
    from s1.evaluation.datasets import EvaluationCase, request_fingerprint

    families = TEST_FAMILIES if split == "test" else TRAIN_FAMILIES
    family = families[kind][index % len(families[kind])]
    question = {"id": "answer", "type": kind}
    operation = None
    if kind == "choice":
        size = (2, 4, 8)[index % 3]
        labels = list(CHOICE_LABELS[:size])
        counts = [rng.randint(1, 80) for _ in labels]
        world = {"kind": "categorical", "labels": labels, "counts": counts}
        display = labels[:]
        rng.shuffle(display)
        question["criteria"] = {label: f"The sampled item belongs to {label}." for label in display}
        if family == "choice_inventory":
            state = {
                "bag_inventory": dict(zip(labels, counts, strict=True)),
                "sampling": "One individual item is selected uniformly from the whole bag.",
            }
            text = "Give the probability distribution of the selected item's category. Counts are exact; each item is equally likely."
        elif family == "choice_census":
            state = {
                "census": [
                    {"category": label, "residents": count}
                    for label, count in zip(labels, counts, strict=True)
                ],
                "sampling": "Select one resident uniformly from all listed residents.",
            }
            text = "What is the category distribution for the sampled resident? Weight categories by their numbers of residents."
        else:
            rows = []
            for label, count in zip(labels, counts, strict=True):
                first = rng.randint(0, count)
                for part, mass in enumerate((first, count - first)):
                    rows.append(
                        {
                            "site": f"site-{len(rows)}",
                            "category": label,
                            "items": mass,
                            "compartment": part,
                        }
                    )
            rng.shuffle(rows)
            state = {
                "route_manifest"
                if family == "choice_route_manifest"
                else "partitioned_boxes": rows,
                "sampling": "Every individual listed item has equal selection probability; locations are not equally weighted.",
            }
            text = "Aggregate all locations for each category and report the category probabilities of one uniformly selected item."
    elif kind == "noul":
        counts = [rng.randint(1, 80) for _ in range(4)]
        world = {"kind": "boolean_table", "counts": counts}
        operation = OPERATIONS[index % len(OPERATIONS)]
        fields = ("red", "round") if split != "test" else ("credential_valid", "policy_eligible")
        rows = [
            {fields[0]: x, fields[1]: y, "individuals": count}
            for (x, y), count in zip(
                ((False, False), (False, True), (True, False), (True, True)), counts, strict=True
            )
        ]
        if split == "test":
            partitions = []
            for row in rows:
                first = rng.randint(0, row["individuals"])
                for site, mass in enumerate((first, row["individuals"] - first)):
                    partitions.append(
                        {
                            "site": f"site-{len(partitions)}",
                            "record_attributes": {field: row[field] for field in fields},
                            "records": mass,
                            "subdivision": site,
                        }
                    )
            rows = partitions
        rng.shuffle(rows)
        state = {
            family: rows,
            "symbols": {"X": fields[0], "Y": fields[1]},
            "sampling": "Select an individual uniformly. Counts are exact; use conditional sampling only if the question says conditional.",
        }
        question["criteria"] = {
            "false": "The specified proposition does not hold for the sampled individual.",
            "true": "The specified proposition holds for the sampled individual.",
        }
        text = f"What is the probability that {OPERATION_TEXT[operation]} for the sampled individual? Return the distribution over false and true."
        if split == "test":
            text = f"An auditor samples one individual record across all sites, weighting by record counts. Attribute flags are nested inside each row's record_attributes. Evaluate the policy proposition: {OPERATION_TEXT[operation]}. Give its false/true sampling probabilities, rather than certainty about the entire table."
    else:
        levels = list(LEVEL_SETS[index % len(LEVEL_SETS)])
        counts = [rng.randint(1, 80) for _ in levels]
        world = {"kind": "ordinal", "labels": levels, "counts": counts}
        question["criteria"] = {
            str(level): f"The sampled outcome receives the numeric score {level}."
            for level in levels
        }
        if family in TRAIN_FAMILIES["score"]:
            state = {
                family: [
                    {"numeric_score": level, "observations": count}
                    for level, count in zip(levels, counts, strict=True)
                ],
                "sampling": "Select one observed outcome uniformly; every observation is equally likely.",
            }
            text = "Return the full distribution of the sampled outcome's numeric score. Preserve numeric level order and the exact frequency weights."
        else:
            observations, bands = [], []
            for band, (level, count) in enumerate(zip(levels, counts, strict=True)):
                bands.append(
                    {
                        "measurement_min": 2 * band,
                        "measurement_max": 2 * band + 1,
                        "assigned_score": level,
                    }
                )
                first = rng.randint(0, count)
                observations.extend(
                    (
                        {"measurement": 2 * band, "frequency": first},
                        {"measurement": 2 * band + 1, "frequency": count - first},
                    )
                )
            rng.shuffle(observations)
            state = {
                family: observations,
                "score_band_rules": bands,
                "sampling": "Draw one observation uniformly from the total frequencies, then apply its inclusive measurement band.",
            }
            text = "Map each measurement through its score band, aggregate the observation frequencies, and report the resulting numeric score distribution."
    question["instructions"] = text
    truth = oracle(world, operation)
    group = "finite-world-" + sha256(canonical(world))
    case = EvaluationCase.model_validate(
        {
            "id": f"typed-{split}-{kind}-{index:05d}-{group[-12:]}",
            "split": split,
            "group_id": group,
            "task_id": family,
            "schema_id": f"{VERSION}-{kind}-{len(truth['labels'])}",
            "language": "en",
            "request": {"state": state, "questions": [question]},
            "gold": {"answer": truth["most_likely_label"]},
            "soft_gold": {"answer": truth["probabilities"]},
        }
    )
    receipt = {
        "case_id": case.id,
        "split": split,
        "group_id": group,
        "template_family": family,
        "world": world,
        "operation": operation,
        "oracle": truth,
        "request_sha256": request_fingerprint(case.request),
    }
    return case, receipt


def prepare(
    output, *, seed=SEED, train=1024, calibration=192, test=384, forbidden_request_sha256s=()
):
    from s1.evaluation.datasets import audit_splits, fingerprint

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    counts = {"train": train, "calibration": calibration, "test": test}
    for total in counts.values():
        type_counts(total)
    forbidden = set(forbidden_request_sha256s)
    if any(
        not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
        for value in forbidden
    ):
        raise ValueError("forbidden requests must be SHA256 strings, never benchmark contents")
    output = Path(output)
    if output.exists():
        raise FileExistsError("refusing to overwrite an existing frozen dataset directory")
    datasets, records, seen_groups, seen_requests = {}, [], set(), set()
    rng = random.Random(seed)
    for split, total in counts.items():
        cases = []
        for kind, count in type_counts(total).items():
            for index in range(count):
                for _ in range(10000):
                    case, record = make_candidate(split, kind, index, rng)
                    request_hash = record["request_sha256"]
                    if (
                        case.group_id not in seen_groups
                        and request_hash not in seen_requests | forbidden
                    ):
                        break
                else:
                    raise ValueError("insufficient distinct finite worlds for this recipe")
                seen_groups.add(case.group_id)
                seen_requests.add(request_hash)
                cases.append(case)
                records.append(record)
        rng.shuffle(cases)
        datasets[f"{split}.jsonl"] = cases
    # The declaration is immutable and separate from the sampled worlds.
    if set(sum(TRAIN_FAMILIES.values(), ())) & set(sum(TEST_FAMILIES.values(), ())):
        raise AssertionError("test template families must be held out")
    output.mkdir(parents=True, exist_ok=False)
    for name, cases in datasets.items():
        (output / name).write_bytes(
            b"".join(case.model_dump_json().encode() + b"\n" for case in cases)
        )
    audit = audit_splits(output / name for name in datasets)
    if not audit["valid"]:
        raise ValueError("source-group/id/request split audit failed")
    provenance = {
        "schema_version": 1,
        "generator_version": VERSION,
        "seed": seed,
        "source": "Locally generated finite populations; exact integer enumeration; no downloaded corpus or benchmark input.",
        "template_families": {
            "train_and_calibration": TRAIN_FAMILIES,
            "heldout_test": TEST_FAMILIES,
        },
        "selected_source_records": records,
    }
    (output / "provenance.json").write_bytes(json_bytes(provenance))
    (output / "split-audit.json").write_bytes(json_bytes(audit))
    (output / "ATTRIBUTION.md").write_text(
        "# Synthetic dataset scope\n\nGenerated by this repository's MIT-licensed preparation script. No external corpus or model outputs are used. Targets are exact sampling probabilities in finite worlds; hard gold means most likely category, not an observed event. This is a synthetic mathematical development experiment, not broad real-world or foundation-pretraining-disjoint evidence. Public231 and JevBench coherence inputs are not generation sources or selection criteria. Test template families were reserved in the generator before execution.\n"
    )
    manifest = {
        "schema_version": 1,
        "generator_version": VERSION,
        "seed": seed,
        "preparation_script_sha256": sha256(Path(__file__).read_bytes()),
        "source": provenance["source"],
        "counts": counts,
        "template_families": provenance["template_families"],
        "heldout_scope": "Test reserves partitioned aggregation, nested policy records, and measurement-to-score band mappings. Primitive logic operations and target level sets remain shared; this is template transfer, not an unseen mathematical rule or real-world heldout claim.",
        "split_audit": audit,
        "forbidden_request_count": len(forbidden),
        "forbidden_request_set_sha256": sha256(canonical(sorted(forbidden))),
        "target_semantics": "Full known sampling distribution; hard labels are oracle most likely only. Evaluate soft CE/KL/Brier and expected-score MAE, not realized-event accuracy.",
        "datasets": {},
        "files": {},
    }
    receipt_requests = {record["case_id"]: record["request_sha256"] for record in records}
    for name, cases in datasets.items():
        manifest["datasets"][name] = {
            "cases": len(cases),
            "questions": len(cases),
            "type_counts": dict(Counter(case.request.questions[0].type for case in cases)),
            "candidate_counts": dict(
                Counter(str(len(case.request.questions[0].labels())) for case in cases)
            ),
            "candidate_counts_by_type": {
                kind: dict(
                    Counter(
                        str(len(case.request.questions[0].labels()))
                        for case in cases
                        if case.request.questions[0].type == kind
                    )
                )
                for kind in TYPES
            },
            "template_counts": dict(Counter(case.task_id for case in cases)),
            "dataset_sha256": fingerprint(cases),
            "file_sha256": sha256((output / name).read_bytes()),
            "case_ids": [case.id for case in cases],
            "group_ids": [case.group_id for case in cases],
            "request_sha256s": [receipt_requests[case.id] for case in cases],
        }
    for path in sorted(output.iterdir()):
        manifest["files"][path.name] = sha256(path.read_bytes())
    (output / "manifest.json").write_bytes(json_bytes(manifest))
    return manifest


def main(argv=None):
    sys.path.insert(0, str(ROOT / "src"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/jevbench/breakthrough-20261008/data")
    )
    parser.add_argument("--seed", type=int, default=SEED)
    for split, count in COUNTS.items():
        parser.add_argument("--" + split, type=int, default=count)
    parser.add_argument(
        "--forbidden-request-hashes",
        type=Path,
        help="Optional JSON array of SHA256 request identities only.",
    )
    args = parser.parse_args(argv)
    forbidden = (
        json.loads(args.forbidden_request_hashes.read_text())
        if args.forbidden_request_hashes
        else []
    )
    if not isinstance(forbidden, list):
        raise ValueError("forbidden request file must contain an array of SHA256 strings")
    manifest = prepare(
        args.output,
        seed=args.seed,
        train=args.train,
        calibration=args.calibration,
        test=args.test,
        forbidden_request_sha256s=forbidden,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "counts": manifest["counts"],
                "manifest_sha256": sha256((args.output / "manifest.json").read_bytes()),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
