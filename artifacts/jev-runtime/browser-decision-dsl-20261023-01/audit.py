"""Pure semantic, ancestry, label-position and role/taint checks; no ML IO."""

import collections
import math
import pathlib

from dsl import DSL_VERSION, FAMILIES, SPLITS, STAGES, authored_snapshot, digest, solve
from request_builder import OPERATIONS, capture_request

EXPECTED_COUNTS = {"train": 96, "dev": 24, "sealed": 48}
ROLE_SPLIT = {"optimizer": "train", "dev_evaluator": "dev", "sealed_evaluator": "sealed"}
HEAD_FOR = {"CLICK": "click_target", "TYPE_TEXT": "type_text_target", "SELECT": "select_target"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_cases(cases):
    require(len(cases) == 168, "Exact bounded population required")
    counts = collections.Counter()
    families = {split: set() for split in SPLITS}
    ancestors = {split: set() for split in SPLITS}
    seen_ids, seen_requests = set(), set()
    positions = {split: {op: [0] * 8 for op in OPERATIONS} for split in SPLITS}
    targets = {split: {op: [0] * 4 for op in HEAD_FOR} for split in SPLITS}
    stages = collections.defaultdict(set)
    source_families = {f[0]: f[1] for f in FAMILIES}
    for case in cases:
        split = case["split"]
        require(split in SPLITS and case["dsl_version"] == DSL_VERSION, "Unknown split/version")
        require(source_families.get(case["family_id"]) == split, "Family crossed predeclared split")
        require(case["ancestor_id"].split("/")[0] == case["family_id"], "Ancestor changed family")
        require(
            case["ancestor_id"] in {f"{case['family_id']}/workflow-{i}" for i in (1, 2, 3)},
            "Unknown workflow ancestor",
        )
        ancestor_index = int(case["ancestor_id"].rsplit("-", 1)[1]) - 1
        expected_state, expected_page, expected_goal, expected_ids = authored_snapshot(
            case["family_id"], ancestor_index, case["branch_stage"]
        )
        require(
            case["state"] == expected_state
            and case["page"] == expected_page
            and case["goal"] == expected_goal
            and case["semantic_control_ids"] == expected_ids,
            "Semantic state/page/goal/control binding differs from authored branch",
        )
        prefix = f"{case['family_id']}-{ancestor_index}-"
        require(
            all(
                case["semantic_control_ids"][key] == prefix + suffix
                for key, suffix in {
                    "details": "details",
                    "classification": "classification-option-0",
                    "acknowledgement": "acknowledgement",
                    "submit": "submit",
                }.items()
            ),
            "Semantic control identity changed",
        )
        require(case["case_id"] not in seen_ids, "Duplicate case identity")
        seen_ids.add(case["case_id"])
        request = case["request"]
        require(digest(request) == case["request_sha256"], "Actual request changed after labeling")
        require(
            digest({k: request[k] for k in ("state", "questions")})
            == case["native_payload_sha256"],
            "Native state/question projection changed",
        )
        require(case["request_sha256"] not in seen_requests, "Duplicate request across population")
        seen_requests.add(case["request_sha256"])
        questions = request["questions"]
        constructed, original_targets = capture_request(case["page"], case["goal"])
        require(
            request == constructed, "Offered request values differ from exact upstream construction"
        )
        require(
            list(questions) == list(constructed["questions"]),
            "Original question head order changed",
        )
        require(
            set(questions) == {"operation", *HEAD_FOR.values()}, "Original offered heads changed"
        )
        require(set(questions["operation"]["criteria"]) == set(OPERATIONS), "Standard8 changed")
        require(
            all(len(questions[h]["criteria"]) == 4 for h in HEAD_FOR.values()),
            "Target distractors removed",
        )
        valid = solve(case["state"])
        valid_ops = {op for op, _ in valid}
        labels = case["labels"]
        require(
            set(labels) == {"operation", *(HEAD_FOR[op] for op in valid_ops if op in HEAD_FOR)},
            "Unused target head mislabeled",
        )
        require(
            set(labels["operation"]["valid_ids"]) == valid_ops, "Semantic operation set changed"
        )
        for op in valid_ops & HEAD_FOR.keys():
            expected_ids = {
                candidate
                for candidate, action in original_targets[op].items()
                if any(
                    action["id"] == case["semantic_control_ids"][semantic]
                    for valid_op, semantic in valid
                    if valid_op == op
                )
            }
            require(
                set(labels[HEAD_FOR[op]]["valid_ids"]) == expected_ids,
                "Target label does not identify semantic control",
            )
        for head, label in labels.items():
            criteria = list(questions[head]["criteria"])
            valid_ids = set(label["valid_ids"])
            require(
                criteria == label["criteria_ids"] and valid_ids and valid_ids <= set(criteria),
                "Labels do not match actual options",
            )
            require(
                label["valid_positions"]
                == [i for i, key in enumerate(criteria) if key in valid_ids],
                "Fixed/stale position label",
            )
            expected = [1 / len(valid_ids) if key in valid_ids else 0.0 for key in criteria]
            require(
                label["soft_target"] == expected and all(math.isfinite(v) for v in expected),
                "Valid-set distribution changed",
            )
            require(math.isclose(sum(expected), 1.0), "Label mass is not one")
        for op in valid_ops:
            positions[split][op][list(questions["operation"]["criteria"]).index(op)] += 1
            if op in HEAD_FOR:
                for position in labels[HEAD_FOR[op]]["valid_positions"]:
                    targets[split][op][position] += 1
        counts[split] += 1
        families[split].add(case["family_id"])
        ancestors[split].add(case["ancestor_id"])
        stages[case["ancestor_id"]].add(case["branch_stage"])
    require(dict(counts) == EXPECTED_COUNTS, "Population counts changed")
    for left in SPLITS:
        for right in SPLITS:
            if left != right:
                require(
                    not families[left] & families[right] and not ancestors[left] & ancestors[right],
                    "Ancestry leakage",
                )
    require(
        len(stages) == 21 and all(v == set(STAGES) for v in stages.values()),
        "Ancestor branch coverage changed",
    )
    for split in SPLITS:
        for table in (positions[split], targets[split]):
            require(
                all(max(row) - min(row) <= 1 for row in table.values()),
                "Position schedule is not balanced",
            )
    return {
        "schema": "independent-workflow-dsl-static-audit-v1",
        "counts": dict(counts),
        "ancestor_counts": {k: len(v) for k, v in ancestors.items()},
        "family_counts": {k: len(v) for k, v in families.items()},
        "operation_by_actual_label_position_counts": positions,
        "selected_target_by_actual_label_position_counts": targets,
        "operation_position_total_counts": {
            split: [sum(positions[split][op][i] for op in OPERATIONS) for i in range(8)]
            for split in SPLITS
        },
        "all_family_and_ancestor_splits_disjoint": True,
        "all_actual_request_labels_bound": True,
        "all_original_offered_heads_options_retained": True,
        "ambiguous_states_arbitrarily_tiebroken": False,
        "model_prediction_confusion_matrix": None,
        "label_position_contingencies_only": True,
        "NN_or_browser_or_official_quality_qualified": False,
    }


def check_role_selection(role, selected_paths, taints, family_splits):
    """Root-side capability-plan check, not an OS sandbox or secrecy claim.

    Caller must authenticate immutable selected files and metadata before this
    check. A future owned loader must enforce the returned exact allowlist.
    Authoring code/full manifest are intentionally unavailable to optimizer.
    """
    require(role in ROLE_SPLIT, "Unknown role")
    require(
        family_splits == {f[0]: f[1] for f in FAMILIES},
        "Caller family map differs from frozen definition",
    )
    split = ROLE_SPLIT[role]
    allowed = {f"data/{split}/{name}.jsonl" for name in ("requests", "labels")}
    allowed.add(f"data/{split}/summary.json")
    paths = list(selected_paths)
    require(paths and len(paths) == len(set(paths)), "Empty or duplicate selection")
    for path in paths:
        relative = pathlib.PurePosixPath(path)
        require(
            not relative.is_absolute() and ".." not in relative.parts and str(relative) == path,
            "Noncanonical selection path",
        )
        require(path in allowed, "Role attempted unselected source/dev/sealed access")
    require(set(taints) == set(paths), "Every selected artifact requires exact taint metadata")
    for path, taint in taints.items():
        require(taint.get("dsl_version") == DSL_VERSION, "Taint version changed")
        require(taint.get("splits") == [split], "Cross-split derived-artifact taint")
        require(taint.get("family_ids") and taint.get("ancestor_ids"), "Missing ancestry taint")
        require(
            all(family_splits.get(family) == split for family in taint["family_ids"]),
            "Family taint violates role",
        )
        require(
            all(
                ancestor.split("/")[0] in taint["family_ids"] for ancestor in taint["ancestor_ids"]
            ),
            "Ancestor taint missing family",
        )
        require(
            all(
                ancestor
                in {f"{family}/workflow-{i}" for family in taint["family_ids"] for i in (1, 2, 3)}
                for ancestor in taint["ancestor_ids"]
            ),
            "Unknown ancestor taint",
        )
        require(
            len(taint.get("sha256", "")) == 64
            and all(ch in "0123456789abcdef" for ch in taint["sha256"]),
            "Missing artifact identity",
        )
    return {
        "role": role,
        "split": split,
        "allowed_selected_paths": paths,
        "scope": "Metadata check only; future independently reviewed root loader must enforce file access.",
    }
