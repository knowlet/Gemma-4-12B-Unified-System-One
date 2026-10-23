"""Deterministic source-corpus authoring only; dry default, no model/feature IO."""

import argparse
import hashlib
import json
import pathlib

from audit import EXPECTED_COUNTS, audit_cases
from dsl import DSL_VERSION, FAMILIES, SEED, SPLITS, canonical, generate_cases
from request_builder import OPERATIONS, SOURCE_PINS

HERE = pathlib.Path(__file__).resolve().parent


def encoded(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def artifacts(cases):
    audit = audit_cases(cases)
    files = {}
    for split in SPLITS:
        selected = [c for c in cases if c["split"] == split]
        requests, labels = [], []
        for c in selected:
            identity = {
                k: c[k]
                for k in (
                    "case_id",
                    "dsl_version",
                    "family_id",
                    "ancestor_id",
                    "request_sha256",
                    "native_payload_sha256",
                )
            }
            requests.append({**identity, "request": c["request"]})
            labels.append({**identity, "heads": c["labels"]})
        files[f"data/{split}/requests.jsonl"] = (
            "\n".join(canonical(v) for v in requests) + "\n"
        ).encode()
        files[f"data/{split}/labels.jsonl"] = (
            "\n".join(canonical(v) for v in labels) + "\n"
        ).encode()
        files[f"data/{split}/summary.json"] = encoded(
            {
                "dsl_version": DSL_VERSION,
                "split": split,
                "case_count": len(selected),
                "family_ids": sorted({c["family_id"] for c in selected}),
                "ancestor_ids": sorted({c["ancestor_id"] for c in selected}),
                "is_authored_synthetic_not_NN": True,
            }
        )
    # Full authoring states/labels must never be selected by the optimizer role.
    files["authored-cases.jsonl"] = ("\n".join(canonical(c) for c in cases) + "\n").encode()
    files["static-audit.json"] = encoded(audit)
    rows = [
        {"path": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in sorted(files.items())
    ]
    sources = []
    for name in (
        "dsl.py",
        "request_builder.py",
        "audit.py",
        "generate.py",
        "role-policy.json",
        "UPSTREAM-LICENSE",
    ):
        data = (HERE / name).read_bytes()
        sources.append(
            {"path": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        )
    manifest = {
        "schema": "independent-workflow-dsl-corpus-manifest-v1",
        "dsl_version": DSL_VERSION,
        "seed": SEED,
        "counts": EXPECTED_COUNTS,
        "families": {f[0]: f[1] for f in FAMILIES},
        "split_unit": "Independently authored domain workflow family, before all entity/phrasing/branch/permutation variants",
        "ancestor_count": 21,
        "states_per_ancestor": 8,
        "trajectory_scope": "Eight counterfactual branch snapshots per workflow ancestor, not eight executed sequential browser steps",
        "operation_contract": list(OPERATIONS),
        "selected_operation_profile": "standard",
        "operation_permutation_schedule": "Eight fixed cyclic rotations; shift=ancestor_ordinal mod8; every ancestor has one label at each operation position",
        "target_position_schedule": "ancestor_ordinal mod4, balanced per selected target head; all four offered target candidates retained",
        "all_labels_authored_before_NN": True,
        "model_failure_outputs_or_existing_canary_used": False,
        "official_tasks_gold_or_browser_DOM_used": False,
        "FIND_TEXT_used": False,
        "supervision": "Complete valid next-action sets; uniform mass on all valid actual criteria positions, zero on remaining positions; unused target heads not supervised",
        "model_routing_tag": "Original upstream default jev-latest retained as source construction metadata, not a runtime model binding; exact native state/questions projection has a separate hash on both requests and labels",
        "question_order": "Operation first; actual target-head insertion order from exact upstream action_space varies with observed-control order. Future collector must join features/labels by actual ordered IDs and native projection hash, never a constant list.",
        "role_checker_placement": "Root-only metadata process; audit.py imports full authoring DSL and must never be imported into a train-only optimizer process",
        "sealed_scope": "Source fixtures may be committed, but future optimizer process must not select authoring source/full manifest or dev/sealed artifacts. Metadata role check is not an OS sandbox.",
        "upstream_source_pins": SOURCE_PINS,
        "authoring_source_pins": sources,
        "artifacts": rows,
        "derived_native_features_or_model_outputs_in_Git": False,
        "optimizer_or_owned_loader_implemented": False,
        "runtime_or_real_browser_quality_claim": False,
    }
    files["manifest.json"] = encoded(manifest)
    return files


def publish(files, destination):
    destination = pathlib.Path(destination)
    if destination.exists():
        raise ValueError("Refuse to overwrite any existing dataset output")
    destination.mkdir()
    for name, data in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-corpus", type=pathlib.Path)
    args = parser.parse_args()
    if args.write_corpus is None:
        print(
            json.dumps(
                {
                    "status": "SOURCE_ONLY_DRY_DEFAULT",
                    "dsl_version": DSL_VERSION,
                    "counts": EXPECTED_COUNTS,
                    "ML_or_dataset_payload_opened": False,
                }
            )
        )
        return
    # Authoring outputs stay in this isolated package; no arbitrary path writer.
    if args.write_corpus.absolute() != HERE / "corpus":
        raise ValueError("Only fresh isolated package/corpus is an allowed destination")
    files = artifacts(generate_cases())
    publish(files, args.write_corpus)
    print(
        json.dumps(
            {
                "status": "AUTHORED_SYNTHETIC_SOURCE_ONLY",
                "files": len(files),
                "counts": EXPECTED_COUNTS,
                "NN_executed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
