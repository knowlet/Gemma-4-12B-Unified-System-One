"""Pinned historical DATA only; no old admission type or execution qualification.

Default CLI is dry. Loading checks selected immutable bytes and recorded teacher
joins, then returns the ordinary case protocol for the CURRENT compiler. It does
not replay browsers, old proof verifiers, processors, or model/source qualification.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).absolute().parents[3]
KIND = "jev_historical_action_data_projection_v1"
PUB = "artifacts/jev-browser/20261016-public-native-training-collection-07"
OWN = "artifacts/jev-runtime/owned-generic-collection-admission-20261018-01"
FORM = "artifacts/jev-runtime/owned-form-progress-collection-admission-20261019-01"
PROOF = "artifacts/jev-runtime/generic-form-progress-native-proof-20261019-01"
MIX = "artifacts/jev-runtime/owned-public-mixed-composition-20261018-01/actual-admit-02"
AUTHORITIES = {
    MIX
    + "/mixed-manifest-A.json": "624b631a0a89dd9265970ef5602e509aa07b7d0d4ab7b72a2e4219210cb81912",
    MIX + "/receipt.json": "1d5796c0a3c4aff2da69fe0d12b7d18e8549f820e65e86bddd8f887ebe4f5cc7",
    "artifacts/jev-browser/20261016-public-native-training-07-admission-01/receipt.json": "bb799a48e8bbf1fb0af5afaf23cb38fb09bb67c3d26c833e96ba33bc34b0fcdb",
    OWN
    + "/actual-admission-01-independent-closed-index.json": "f1d81a5f72f3d45f26c7183acc619f0e0f110c100393fd2466ae745c2d476b6d",
    OWN
    + "/admission-plan-02.json": "1391920f12433241598549c69892619f4c1ace890318f1f9aa54539186ae7243",
    FORM
    + "/actual-collection-verification-01/terminal.json": "1b1f3ddc1b5220bc8982ddb7f8e7f85b48edc1cd7144abe543457c993268258e",
    FORM
    + "/root-collection-verification01-exit-review-01.json": "d2a0cc75871938c3ebd0d84f5e9c926ba6f05100829875a2a119028a4d922fb8",
    FORM
    + "/admission-plan-02.json": "01518a2859b4483631d19c2592498a3dac29ba4221dc6f908ea6f9c19611cc34",
    PROOF
    + "/root-actual01-closed-index-01.json": "00eed5a57f169145cd661f91c65db3ea547b847123ee2d2719f9b3cdc9c098eb",
}
TARGETS = {"CLICK": "click_target", "TYPE_TEXT": "type_text_target", "SELECT": "select_target"}
COUNTS = {"public": (120, 22), "owned": (9, 5), "form": (13, 10)}


def require(value, message):
    if not value:
        raise ValueError(message)


def wire(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def parse(value):
    def pairs(items):
        result = {}
        for key, item in items:
            require(key not in result, "Duplicate historical JSON key")
            result[key] = item
        return result

    return json.loads(
        value,
        object_pairs_hook=pairs,
        parse_constant=lambda _: require(False, "Nonfinite historical JSON"),
    )


def load_historical_data(root=ROOT):
    """Return all 179 ordered cases/wires, authenticated refs and historical provenance."""
    root, refs, cache = Path(root).absolute(), {}, {}

    def checked(label, pin):
        relative = Path(label)
        require(
            type(label) is str and not relative.is_absolute() and ".." not in relative.parts,
            "Confined historical data path",
        )
        require(
            type(pin) is dict
            and set(pin) == {"sha256", "size_bytes"}
            and type(pin["size_bytes"]) is int
            and 0 <= pin["size_bytes"] <= 8 * 1024**2,
            "Bounded typed historical data pin",
        )
        if label in refs:
            require(refs[label] == pin, "Conflicting historical data authority")
            return cache[label]
        path = root / relative
        require(
            not any(p.is_symlink() for p in (path, *path.parents)), "No historical data symlink"
        )
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            require(
                stat.S_ISREG(before.st_mode) and before.st_size == pin["size_bytes"],
                "Bounded regular historical data",
            )
            value = stream.read(pin["size_bytes"] + 1)
            after = os.fstat(stream.fileno())

        def signature(s):
            return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

        require(
            signature(before) == signature(after) == signature(path.stat())
            and len(value) == pin["size_bytes"]
            and sha(value) == pin["sha256"],
            "Historical data bytes changed",
        )
        refs[label], cache[label] = dict(pin), value
        return value

    def initial(label, digest):
        size = (root / label).lstat().st_size
        value = parse(checked(label, {"sha256": digest, "size_bytes": size}))
        require(type(value) is dict, "Historical authority must be an object")
        return value

    def referenced(ref):
        require(
            type(ref) is dict and set(ref) == {"path", "sha256", "size_bytes"},
            "Exact historical reference",
        )
        return checked(ref["path"], {k: v for k, v in ref.items() if k != "path"})

    def lines(value):
        return [parse(line) for line in value.splitlines() if line.strip()]

    authority = {p: initial(p, digest) for p, digest in AUTHORITIES.items()}
    mixed, receipt = authority[MIX + "/mixed-manifest-A.json"], authority[MIX + "/receipt.json"]
    require(
        receipt["status"] == "complete-typed-mixed-admission-not-fit"
        and receipt["mixed_admissions"] == 2
        and mixed["profile"] == "public120-owned9-row-equal-1pass-v1"
        and receipt["external_manifest_pins"]["A"] == AUTHORITIES[MIX + "/mixed-manifest-A.json"],
        "Historical mixed admission",
    )
    public_receipt = authority[
        "artifacts/jev-browser/20261016-public-native-training-07-admission-01/receipt.json"
    ]
    inventory = mixed["components"]["public"]["public_inventory"]["files"]
    require(
        public_receipt["admission_passed"] is True
        and wire(public_receipt["full_closed_input_inventory"]["files"]) == wire(inventory),
        "Historical public inventory join",
    )
    population = {name: {split: [] for split in ("train", "development")} for name in COUNTS}

    def append(component, split, case_id, group_id, payload, teacher, valid, gold=None):
        from s1.browser_public_training import NativeWireRequest

        request = NativeWireRequest.model_validate(payload)
        require(
            type(case_id) is str and bool(case_id) and type(group_id) is str and bool(group_id),
            "Historical case/group identity",
        )
        require(
            type(teacher) is dict and set(teacher) == {"operation", "target"}, "Recorded teacher"
        )
        operation, target = teacher["operation"], teacher["target"]
        labels, head = request.candidate_labels, TARGETS.get(operation)
        require(
            operation in labels["operation"]
            and type(valid) is dict
            and set(valid) == {operation}
            and wire(valid[operation]) == wire([] if target is None else [target])
            and (
                (head is not None and type(target) is str and target in labels[head])
                or (head is None and target is None)
            ),
            "Historical teacher/target join",
        )
        derived = {"operation": operation, **({head: target} if head else {})}
        require(
            gold is None or wire(gold) == wire(derived),
            "Historical annotation/teacher disagreement",
        )
        case = SimpleNamespace(
            id=case_id,
            split=split,
            group_id=group_id,
            gold=derived if gold is None else gold,
            request=request,
        )
        projection = {
            "case_id": case_id,
            "payload": parse(request.wire_bytes),
            "teacher_choice": teacher,
            "valid_next_targets": valid,
        }
        population[component][split].append((case, projection))

    for split in ("train", "development"):
        annotations = lines(
            checked(PUB + f"/{split}/cases.jsonl", inventory[f"{split}/cases.jsonl"])
        )
        records = lines(
            checked(
                PUB + f"/{split}/wire_requests.jsonl", inventory[f"{split}/wire_requests.jsonl"]
            )
        )
        require(
            len(annotations) == len(records) == COUNTS["public"][split == "development"],
            "Complete public split",
        )
        for annotation, record in zip(annotations, records, strict=True):
            require(
                annotation["id"] == record["case_id"]
                and annotation["split"] == split
                and annotation["wire_row_id"] == record["wire_row_id"]
                and wire(annotation["request"]) == wire(record["native_payload"])
                and sha(wire(record["native_payload"])) == record["native_wire_sha256"]
                and wire(annotation["valid_next_targets"]) == wire(record["valid_next_targets"]),
                "Public annotation/wire join",
            )
            append(
                "public",
                split,
                annotation["id"],
                annotation["group_id"],
                record["native_payload"],
                record["teacher_choice"],
                record["valid_next_targets"],
                annotation["gold"],
            )

    owned_index = authority[OWN + "/actual-admission-01-independent-closed-index.json"]
    require(
        owned_index["exit_code"] == 0
        and owned_index["file_count"] == len(owned_index["files"]) == 5,
        "Closed owned projection index",
    )
    owned_files = {
        r["path"]: checked(
            OWN + "/actual-admission-01/" + r["path"], {k: v for k, v in r.items() if k != "path"}
        )
        for r in owned_index["files"]
    }
    owned_provenance = parse(owned_files["provenance.json"])
    owned_plan = authority[OWN + "/admission-plan-02.json"]
    scenes = parse(referenced(owned_plan["specification"]))["scene_and_goal_records"]
    for split in ("train", "development"):
        rows = lines(owned_files[f"admitted-wire-projection-{split}.jsonl"])
        require(
            [r["case_id"] for r in rows] == owned_provenance["splits"][split], "Owned split order"
        )
        for row in rows:
            scene = [s for s in scenes if row["case_id"].startswith(s["id"] + "-")]
            require(len(scene) == 1 and scene[0]["split"] == split, "Owned scene/split join")
            append(
                "owned",
                split,
                row["case_id"],
                "owned-semantic-scene:" + scene[0]["id"],
                row["payload"],
                row["teacher_choice"],
                row["valid_next_targets"],
            )
        actual = [
            {"component": name, "row_id": c.id, "wire_sha256": c.request.wire_sha256}
            for name in ("public", "owned")
            for c, _ in population[name][split]
        ]
        require(wire(actual) == wire(mixed["rows"][split]), "Original public/owned data manifest")

    form_terminal = authority[FORM + "/actual-collection-verification-01/terminal.json"]
    form_exit = authority[FORM + "/root-collection-verification01-exit-review-01.json"]
    form_index = authority[PROOF + "/root-actual01-closed-index-01.json"]
    require(
        form_terminal["status"] == "ADMITTED_COLLECTION_NOT_COMPILED_NOT_FIT_NOT_QUALIFIED"
        and form_terminal["error"] is None
        and form_exit["tool_exit_code"] == 0
        and form_exit["terminal"]["sha256"]
        == AUTHORITIES[FORM + "/actual-collection-verification-01/terminal.json"]
        and form_terminal["provenance"]["proof_index_sha256"]
        == AUTHORITIES[PROOF + "/root-actual01-closed-index-01.json"]
        and form_terminal["plan_sha256"] == AUTHORITIES[FORM + "/admission-plan-02.json"]
        and form_index["ordinary_file_count"] == len(form_index["files"]) == 236,
        "Historical form admission/closed index",
    )
    plan = parse(
        referenced(authority[FORM + "/admission-plan-02.json"]["anchors"]["collection_plan"])
    )
    manifest = parse(referenced(plan["candidate_manifest"]))
    require(wire(plan["cases"]) == wire(manifest["cases"]), "Original authored form manifest")
    specs = {s["id"]: s for s in manifest["cases"]}
    indexed = {r["path"]: r for r in form_index["files"]}
    require(len(indexed) == 236, "Unique closed form leaves")
    records = lines(referenced(indexed[PROOF + "/actual-01/candidate-native-wire-requests.jsonl"]))
    by_id = {r["case_id"]: r for r in records}
    require(
        len(by_id) == len(records) == len(form_terminal["all23_rows"]) == 23,
        "Exactly23 form teachers",
    )
    require(
        {item["id"] for item in form_terminal["all23_rows"]} == set(by_id),
        "Complete historical form identity join",
    )
    for item in form_terminal["all23_rows"]:
        row = by_id[item["id"]]
        spec = specs[row["episode_id"]]
        raw = row["native_wire_utf8"].encode()
        require(
            item["status"] == "VERIFIED"
            and item["split"] == spec["split"]
            and item["family"] == spec["family"]
            and sha(raw) == row["native_wire_sha256"] == item["native_wire_sha256"]
            and len(raw) == row["native_wire_size_bytes"]
            and raw == wire(row["native_payload"]),
            "Form wire/terminal/family join",
        )
        for proof_ref in row["proof_refs"].values():
            proof_path = PROOF + "/actual-01/" + proof_ref["artifact_path"]
            require(
                indexed[proof_path]["sha256"] == proof_ref["sha256"], "Form teacher proof reference"
            )
        body = row["proof_refs"]["native_wire_body"]
        require(
            referenced(indexed[PROOF + "/actual-01/" + body["artifact_path"]]) == raw,
            "Original form native body",
        )
        append(
            "form",
            item["split"],
            item["id"],
            "owned-form-family:" + item["family"],
            row["native_payload"],
            row["teacher_choice"],
            row["valid_next_targets"],
        )

    ordered = []
    for index, split in enumerate(("train", "development")):
        for component, counts in COUNTS.items():
            require(
                len(population[component][split]) == counts[index], "Complete historical component"
            )
            ordered.extend(population[component][split])
    cases, wires = map(list, zip(*ordered, strict=True))
    require(len({c.id for c in cases}) == 179, "Unique179 historical identities")
    train, dev = (
        [c for c in cases if c.split == "train"],
        [c for c in cases if c.split == "development"],
    )
    require(
        not {c.request.wire_sha256 for c in train} & {c.request.wire_sha256 for c in dev}
        and not {c.group_id for c in train} & {c.group_id for c in dev},
        "Historical train/dev leakage",
    )
    return {
        "cases": cases,
        "wires": wires,
        "refs": dict(sorted(refs.items())),
        "train_ids": [c.id for c in train],
        "dev_ids": [c.id for c in dev],
        "counts": {"train": 142, "development": 37, "total": 179},
        "provenance": {
            "schema_id": KIND,
            "source_kind": "historical-data-projection",
            "components": COUNTS,
            "old_admission_types_constructed": False,
            "old_execution_qualification_attested": False,
            "public_registry_admission_performed": False,
            "new_labels_created": False,
            "source_scope": "Historical data authority only; current compiler/codec/runtime need separate pins and actual CPU evidence.",
        },
    }


if __name__ == "__main__":
    print(json.dumps({"schema_id": KIND, "status": "DRY_NO_DATA_OR_PROCESSOR_LOADED"}))
