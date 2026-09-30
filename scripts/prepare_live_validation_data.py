"""Pinned public screening data for explicitly invoked live model validation.

Requires the repository's train and inference extras. No dataset loader executes
remote code. Source bytes are cached and checked against embedded SHA-256 pins.
This small task-specific sample does not establish broad multimodal quality.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import random
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

SEED = 20260930
BOOLQ_REVISION = "35b264d03638db9f4ce671b711558bf7ff0f80d5"
BOOLQ_UPSTREAM_REVISION = "90af34107399cc7a446b373dc4ee35b8001da7c2"
MNIST_REVISION = "77f3279092a1c1579b2250db8eafed0ad422088c"
FSDD_REVISION = "26eb9aaf76e81b692f806f9140c2d2777410d7a1"
BOOLQ_REPO = "https://huggingface.co/datasets/google/boolq"
MNIST_REPO = "https://huggingface.co/datasets/ylecun/mnist"
FSDD_REPO = "https://github.com/Jakobovski/free-spoken-digit-dataset"
FSDD_RAW = (
    f"https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/{FSDD_REVISION}/"
)
SOURCE_FILES = {
    "boolq-train.parquet": (
        f"{BOOLQ_REPO}/resolve/{BOOLQ_REVISION}/data/train-00000-of-00001.parquet",
        "4f028e992c0bd4df30b9f056f4946b64f5c23028034ff0ed5ea467d8538cc623",
    ),
    "boolq-validation.parquet": (
        f"{BOOLQ_REPO}/resolve/{BOOLQ_REVISION}/data/validation-00000-of-00001.parquet",
        "52355d11524b4b874a9b9dcc278feb10f672d52c4f4eff9872e695ede59820f8",
    ),
    "mnist-test.parquet": (
        f"{MNIST_REPO}/resolve/{MNIST_REVISION}/mnist/test-00000-of-00001.parquet",
        "d49fcf556ce25b002b302e318ce4a11098bbfe5d4499c3f35d7c72297c52374b",
    ),
    "boolq-README.md": (
        "https://raw.githubusercontent.com/google-research-datasets/boolean-questions/"
        f"{BOOLQ_UPSTREAM_REVISION}/README.md",
        "0afe70d7cdbb644637592212f0cb15339389910bf00f35764ec3c545257aa663",
    ),
    "mnist-README.md": (
        f"{MNIST_REPO}/raw/{MNIST_REVISION}/README.md",
        "8cd17b7ebe5c115eb401f2ce57f9b3c8b16a6e293dc312c35e4f37f7a81fc4a2",
    ),
    "fsdd-README.md": (
        FSDD_RAW + "README.md",
        "523de3cafa0f54707a0ab2760a2d32b640ccfe834a081d7d35c894f1fb6d79f5",
    ),
    "fsdd-metadata.py": (
        FSDD_RAW + "metadata.py",
        "b7fcb538c27ec10ef94031de437b6b9da4724e8fdd2a1c6f25e7b3d76a067c76",
    ),
}
FSDD_FILES = {
    "0_jackson_4.wav": "967c3b14efad3501be65f167466f008b4e8abfc219b2bd009c443054bb2a6d3d",
    "0_george_3.wav": "d5255e12fcc9ec9be0fa5d2dcca7c15ce7d1d39f0f581e6338d8930c09d6e52d",
    "1_theo_0.wav": "a856fff7bcb68328a76b269c0d8b1d4269c28ac7e3d6725fe9da5bdd05405b55",
    "1_nicolas_0.wav": "d57a29689b0abf1ab609ab401dd4053dbc18c317a4ed9ebed6a9aa034e9bab0a",
    "2_lucas_4.wav": "e50e731cb732f7b7ff4e293f5bc80cbea35834b532d914ac9b1b5c62386aa2b7",
    "2_yweweler_2.wav": "464fc9f6602fbbda43687c01ccb3edc8b46fc35e4d42841b88314fb7c07bcefe",
    "3_jackson_3.wav": "1dfd194038b7837c1feaafee606e9616ae75c9b582ede0b6eff04e5372edbab0",
    "3_george_4.wav": "ae8ce7634a2a3a20a2307ca208645a4716fd2040eee49aba35b2e078b5e21a6d",
    "4_theo_3.wav": "c1f1d9e77651d7f31715f732aabdb0479c73278d6bc7020435445dcf6557390e",
    "4_nicolas_0.wav": "dc6ae853b163bebf1774701a4f7e6e27bbace3472ba3628ec2d61b2e6ab1d0c7",
    "5_lucas_0.wav": "266bab64975aebcb2f8e734cb8f4721ef362fb506190c35bd7a31c4bd9e418a2",
    "5_yweweler_3.wav": "0508ac7bea35a189194bd4edd45f727be1667116535efeeecebecebb92c1a7cb",
    "6_jackson_0.wav": "fe7705fdfaddc378d72c479664ab8aacd53fa78a99c3d130ad74b1ff40212595",
    "6_george_1.wav": "7084ce484c7390bbf565b2f36ae0feea079a4cffec18ddbcc39f5a60fd4624aa",
    "7_theo_4.wav": "3baf71f47769277d7123bc78719fab659de4ab50676e46c77fff538bbb45d67e",
    "7_nicolas_4.wav": "5ff214a30aee4a90bfa7705399c4a784213906ed6d4ebab7cec95cf3f115d50d",
    "8_lucas_2.wav": "86f5cd61b08f75409ae641707e6e4d9e6303f81ac2244b885b7207b9ae8d6cc7",
    "8_yweweler_0.wav": "ef1973bb1d3111d6b251aee38ea7afe93a987fff84b2861729648f90acfa47b3",
    "9_jackson_3.wav": "e31bd5b6bae99c8f706029461a89e3fdf71e0625d7d662f73782c20be0f625b5",
    "9_george_2.wav": "ca35b3a937cc95d2682ad27688f23dec1dc6f6dc88d05fdd0d710fb87cacb46e",
}
DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def _fetch(directory: Path, name: str, url: str, expected_sha256: str) -> Path:
    target = directory / name
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=90) as response:
                    raw = response.read()
                break
            except (urllib.error.URLError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        if _sha(raw) != expected_sha256:
            raise ValueError(f"downloaded public source checksum mismatch: {name}")
        target.write_bytes(raw)
    if _sha(target.read_bytes()) != expected_sha256:
        raise ValueError(f"cached public source checksum mismatch: {name}")
    return target


def _write(path: Path, raw: bytes) -> None:
    if path.exists() and path.read_bytes() != raw:
        raise FileExistsError(f"refusing to replace a different prepared dataset: {path}")
    if not path.exists():
        path.write_bytes(raw)


def _write_cases(path: Path, cases) -> None:
    _write(path, b"".join(case.model_dump_json().encode() + b"\n" for case in cases))


def _boolq(rows, original_split: str, split: str, count: int, forbidden: set[str]):
    from s1.evaluation.datasets import EvaluationCase

    rng = random.Random(f"{SEED}:boolq:{split}")
    by_label = {False: [], True: []}
    for index, row in enumerate(rows):
        if type(row["answer"]) is not bool:
            raise ValueError("BoolQ source must contain the published boolean labels")
        by_label[row["answer"]].append((index, row))
    cases, selected = [], []
    for answer, pool in by_label.items():
        rng.shuffle(pool)
        accepted = 0
        for index, row in pool:
            group = "boolq-passage-" + _sha(row["passage"].encode())
            if group in forbidden:
                continue
            case = EvaluationCase.model_validate(
                {
                    "id": f"boolq-{original_split}-{index}",
                    "split": split,
                    "group_id": group,
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
                    "gold": {"answer": "true" if answer else "false"},
                }
            )
            cases.append(case)
            selected.append(
                {
                    "case_id": case.id,
                    "source_split": original_split,
                    "source_row_index": index,
                    "source_row_sha256": _sha(_canonical(row)),
                }
            )
            forbidden.add(group)
            accepted += 1
            if accepted == count // 2:
                break
        if accepted != count // 2:
            raise ValueError("insufficient disjoint BoolQ passage groups for the fixed sample")
    pairs = list(zip(cases, selected))
    rng.shuffle(pairs)
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _digit_case(identifier: str, group: str, task: str, media, label: int):
    from s1.evaluation.datasets import EvaluationCase

    modality = "image" if task == "mnist" else "audio recording"
    return EvaluationCase.model_validate(
        {
            "id": identifier,
            "split": "test",
            "group_id": group,
            "task_id": task,
            "schema_id": "digit-11-label-choice",
            "language": "en",
            "request": {
                "state": f"Identify the single digit in the supplied {modality}.",
                "media": [media],
                "questions": [
                    {
                        "id": "digit",
                        "type": "choice",
                        "instructions": (
                            "Which digit is present? Select unknown only when the digit cannot "
                            "be determined from the supplied information."
                        ),
                        "criteria": {
                            **{
                                f"digit-{i}": f"The digit {word} ({i})."
                                for i, word in enumerate(DIGIT_WORDS)
                            },
                            "unknown": "The digit cannot be determined from supplied information.",
                        },
                    }
                ],
            },
            "gold": {"digit": f"digit-{label}"},
        }
    )


def _media(raw: Path):
    import numpy as np
    import pyarrow.parquet as parquet
    import soundfile as sf
    from PIL import Image, ImageOps
    from scipy.signal import resample_poly

    rows = parquet.read_table(raw / "mnist-test.parquet").to_pylist()
    if len(rows) != 10000:
        raise ValueError("unexpected official MNIST test size")
    rng = random.Random(f"{SEED}:mnist")
    by_label = {label: [] for label in range(10)}
    for index, row in enumerate(rows):
        by_label[row["label"]].append(index)
    selected_images = []
    for label, indexes in by_label.items():
        rng.shuffle(indexes)
        selected_images.extend(indexes[: 4 if label < 2 else 3])
    rng.shuffle(selected_images)
    cases, bundle, selections = [], [], []
    for index in selected_images:
        row = rows[index]
        source_bytes = row["image"]["bytes"]
        source_image = Image.open(io.BytesIO(source_bytes)).convert("L")
        # Preserve the entire digit; nearest-neighbor enlargement adds no detail.
        image = ImageOps.invert(source_image).resize((224, 224), Image.Resampling.NEAREST)
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        encoded_bytes = buffer.getvalue()
        case = _digit_case(
            f"mnist-test-{index}",
            f"mnist-test-index-{index}",
            "mnist",
            {"type": "image", "data": base64.b64encode(encoded_bytes).decode()},
            row["label"],
        )
        cases.append(case)
        bundle.append(
            {
                "case": case.model_dump(mode="json"),
                "missing_media_gold": {"digit": "unknown"},
                "verified_text": (
                    f"The handwritten digit is {DIGIT_WORDS[row['label']]} ({row['label']})."
                ),
                "annotation_source": "published human source label; label-derived oracle text",
            }
        )
        selections.append(
            {
                "case_id": case.id,
                "source_split": "test",
                "source_row_index": index,
                "original_image_sha256": _sha(source_bytes),
                "converted_image_sha256": _sha(encoded_bytes),
                "label": row["label"],
            }
        )
    for name, sha256 in FSDD_FILES.items():
        url = FSDD_RAW + "recordings/" + name
        source = _fetch(raw, "fsdd/" + name, url, sha256)
        label, speaker, index = name.removesuffix(".wav").split("_")
        if int(index) not in range(5):
            raise ValueError("FSDD selection must use the official test recording indices 0-4")
        samples, rate = sf.read(source, dtype="float32")
        if samples.ndim != 1 or rate != 8000:
            raise ValueError("unexpected original FSDD channel count or sample rate")
        converted = np.clip(resample_poly(samples, 2, 1), -1, 1).astype(np.float32)
        case = _digit_case(
            f"fsdd-test-{name.removesuffix('.wav')}",
            f"fsdd-speaker-{speaker}",
            "fsdd",
            {"type": "audio", "samples": converted.tolist(), "sampling_rate": 16000},
            int(label),
        )
        cases.append(case)
        bundle.append(
            {
                "case": case.model_dump(mode="json"),
                "missing_media_gold": {"digit": "unknown"},
                "verified_text": f"The speaker says '{DIGIT_WORDS[int(label)]}'.",
                "annotation_source": "published contributor digit label; label-derived oracle text",
            }
        )
        selections.append(
            {
                "case_id": case.id,
                "source_split": "official-test-index-0-to-4",
                "source_url": url,
                "source_filename": name,
                "source_sha256": sha256,
                "speaker": speaker,
                "recording_index": int(index),
                "label": int(label),
                "original_sampling_rate": rate,
                "resampled_sampling_rate": 16000,
                "resampled_float32_sha256": _sha(converted.astype("<f4").tobytes()),
            }
        )
    return cases, bundle, selections


def prepare(output: Path) -> dict:
    """Create reproducible, strictly disjoint text splits and heldout media cases."""
    import pyarrow.parquet as parquet

    from s1.evaluation.datasets import audit_splits, fingerprint, request_fingerprint

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "raw"
    for name, (url, digest) in SOURCE_FILES.items():
        _fetch(raw, name, url, digest)
    train_rows = parquet.read_table(raw / "boolq-train.parquet").to_pylist()
    validation_rows = parquet.read_table(raw / "boolq-validation.parquet").to_pylist()
    if len(train_rows) != 9427 or len(validation_rows) != 3270:
        raise ValueError("unexpected official labeled BoolQ split sizes")
    # Select heldout rows first, then remove their groups from every fitting split.
    forbidden: set[str] = set()
    test, test_selection = _boolq(validation_rows, "validation", "test", 128, forbidden)
    calibration, cal_selection = _boolq(train_rows, "train", "calibration", 16, forbidden)
    train, train_selection = _boolq(train_rows, "train", "train", 128, forbidden)
    # Reuse the same deterministic label pools, skipping all already selected
    # passage groups, so the full curve is a strict superset of the pilot train.
    extra_train, extra_train_selection = _boolq(train_rows, "train", "train", 128, forbidden)
    full_train = train + extra_train
    media, bundle, media_selection = _media(raw)
    datasets = {
        "text-train.jsonl": train,
        "text-train-full.jsonl": full_train,
        "text-calibration.jsonl": calibration,
        "text-test.jsonl": test,
        "media-test.jsonl": media,
    }
    for name, cases in datasets.items():
        _write_cases(output / name, cases)
    _write(output / "media-bundle.jsonl", b"".join(_canonical(row) + b"\n" for row in bundle))
    heldout_names = ("text-calibration.jsonl", "text-test.jsonl", "media-test.jsonl")
    audit = audit_splits([output / name for name in ("text-train.jsonl", *heldout_names)])
    full_train_audit = audit_splits(
        [output / name for name in ("text-train-full.jsonl", *heldout_names)]
    )
    if not audit["valid"] or not full_train_audit["valid"]:
        raise ValueError("prepared data failed ID/source-group/request split leakage audit")
    selection_by_id = {
        row["case_id"]: row
        for row in (
            train_selection
            + extra_train_selection
            + cal_selection
            + test_selection
            + media_selection
        )
    }
    for cases in datasets.values():
        for case in cases:
            selection_by_id[case.id].update(
                group_id=case.group_id,
                request_sha256=request_fingerprint(case.request),
                evaluation_split=case.split,
            )
    summary = {
        "schema_version": 1,
        "dataset_format_version": 2,
        "seed": SEED,
        "purpose": "task-specific public heldout live validation screening",
        "scope_limit": (
            "Balanced small samples; not broad representative media or general reasoning evidence. "
            "Upstream foundation-model pretraining overlap cannot be audited from these sources."
        ),
        "selection_policy": {
            "boolq": "64 true/64 false train and test; 8/8 calibration; unique passage SHA groups",
            "boolq_full_train": (
                "128/class (256 total), strict superset of the 128-case pilot train; "
                "same seeded label pools skip every existing train/calibration/test passage group"
            ),
            "boolq_schema": (
                "One fixed NOUL schema; original passage and question preserved in request state."
            ),
            "mnist": "official test only; fixed-seed stratified sample; 4 digits 0/1 and 3 others",
            "fsdd": "official test recordings 0-4; 2 per digit; all 6 speakers; speaker groups",
            "ontology": "complete false/true or digit-0..digit-9 plus unknown; no truncation",
        },
        "sources": {
            "boolq": {
                "publisher": "Google Research; Christopher Clark et al. (2019)",
                "source_url": BOOLQ_REPO,
                "source_revision": BOOLQ_REVISION,
                "upstream_url": "https://github.com/google-research-datasets/boolean-questions",
                "upstream_revision": BOOLQ_UPSTREAM_REVISION,
                "license": "CC-BY-SA-3.0",
                "license_source": SOURCE_FILES["boolq-README.md"][0],
                "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
                "label_policy": "official labeled validation used for test; unlabeled test not used",
                "transport_note": "Original official GCS files returned 403; Google HF mirror used.",
            },
            "mnist": {
                "publisher": "Yann LeCun, Corinna Cortes, Christopher J.C. Burges",
                "source_url": MNIST_REPO,
                "source_revision": MNIST_REVISION,
                "upstream_url": "http://yann.lecun.com/exdb/mnist/",
                "license": "MIT",
                "license_source": SOURCE_FILES["mnist-README.md"][0],
                "original_split": "test",
                "transform": "28x28 grayscale inverted; nearest enlargement to 224x224 RGB PNG",
            },
            "fsdd": {
                "publisher": "Zohar Jackson and FSDD contributors",
                "source_url": FSDD_REPO,
                "source_revision": FSDD_REVISION,
                "license": "CC-BY-SA-4.0",
                "license_source": SOURCE_FILES["fsdd-README.md"][0],
                "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
                "original_split": "official-test-recording-indices-0-to-4",
                "transform": "8kHz mono PCM normalized by decoder; polyphase upsample to 16kHz; clip",
                "source_group": "speaker; six groups limit independent statistical evidence",
            },
        },
        "source_files": {
            name: {"url": url, "sha256": digest} for name, (url, digest) in SOURCE_FILES.items()
        },
        "datasets": {
            name: {
                "cases": len(cases),
                "dataset_sha256": fingerprint(cases),
                "file_sha256": _sha((output / name).read_bytes()),
                "groups": len({case.group_id for case in cases}),
                "label_counts": dict(Counter(case.gold[next(iter(case.gold))] for case in cases)),
                "questions_per_case": sorted({len(case.request.questions) for case in cases}),
            }
            for name, cases in datasets.items()
        },
        "media_bundle_sha256": _sha((output / "media-bundle.jsonl").read_bytes()),
        "M0_annotation": "unknown is separately assigned because the sole digit evidence is removed",
        "M2_annotation": (
            "Oracle descriptions derived from published human digit labels. No new independent "
            "human listening or transcription review. Diagnostic only; not a production capability."
        ),
        "split_audit": audit,
        "full_train_split_audit": full_train_audit,
        "full_train_superset": {
            "pilot_dataset": "text-train.jsonl",
            "full_dataset": "text-train-full.jsonl",
            "pilot_cases_preserved": len(train),
            "additional_cases": len(extra_train),
            "support_shots_per_class": [8, 32, 128],
            "support_seeds": [0, 1, 2],
        },
        "selected_source_records": list(selection_by_id.values()),
    }
    _write(
        output / "provenance.json",
        json.dumps(summary, indent=2, ensure_ascii=False).encode() + b"\n",
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2, ensure_ascii=False))
