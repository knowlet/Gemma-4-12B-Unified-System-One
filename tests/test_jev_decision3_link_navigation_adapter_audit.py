"""Synthetic CPU witnesses only: no model, real training data, or GPU."""

import copy
import math

import jev_decision3_link_navigation_adapter_audit as audit
import numpy as np
import pytest


def softmax(logits):
    values = np.asarray(logits, dtype=np.float64)
    exp = np.exp(values - values.max())
    return (exp / exp.sum()).tolist()


def response(heads, criteria, lengths, logits):
    answers = {}
    for head, labels, row in zip(heads, criteria, logits, strict=True):
        values = softmax(row[: len(labels)])
        entropy = -sum(value * math.log(value) for value in values if value > 0)
        confidence = 1.0 if len(values) == 1 else 1.0 - entropy / math.log(len(values))
        answers[head] = {
            "type": "choice",
            "probabilities": dict(zip(labels, values, strict=True)),
            "choice": labels[int(np.argmax(values))],
            "confidence": confidence,
        }
    return {
        "model": "d3-edge",
        "answers": answers,
        "usage": {"input_tokens": sum(lengths.values()), "output_tokens": 0},
    }


def synthetic(heads=("operation", "click_target", "select_target")):
    heads = list(heads)
    criteria = [
        ["first", "second", "third"] if head != "select_target" else ["left", "right"]
        for head in heads
    ]
    payload = {
        "state": "authored synthetic",
        "questions": {
            head: {
                "type": "choice",
                "instructions": "choose",
                "criteria": {key: "description" for key in labels},
            }
            for head, labels in zip(heads, criteria, strict=True)
        },
    }
    lengths = {head: index + 4 for index, head in enumerate(heads)}
    x = np.zeros((len(heads), audit.WIDTH), dtype=np.float32)
    x[:, 0] = 1.0
    base = np.zeros((audit.CODES, audit.WIDTH), dtype=np.float32)
    base[1, 0] = 0.2
    delta = {head: np.zeros((audit.CODES, audit.WIDTH), dtype=np.float32) for head in audit.HEADS}
    delta["operation"][0, 0] = 0.5
    delta["click_target"][2, 0] = 0.8
    before = x.astype(np.float64) @ base.astype(np.float64).T
    after = before.copy()
    for index, head in enumerate(heads):
        if head in audit.HEADS:
            after[index] += x[index].astype(np.float64) @ delta[head].astype(np.float64).T
    raw = response(heads, criteria, lengths, before)
    return {
        "native_payload": payload,
        "lengths": lengths,
        "ordered_head_ids": heads,
        "ordered_criteria_ids": criteria,
        "features": x,
        "baseline": base,
        "deltas": delta,
        "delta_sha256": {head: audit._matrix_sha(delta[head]) for head in audit.HEADS},
        "baseline_raw": raw,
        "zero_raw": copy.deepcopy(raw),
        "composed_raw": response(heads, criteria, lengths, after),
        "projection": {
            "ordered_head_ids": heads,
            "applied_head_ids": [head for head in heads if head in audit.HEADS],
            "absent_head_ids": [head for head in audit.HEADS if head not in heads],
            "legal_counts": [len(labels) for labels in criteria],
            "same_call_baseline_replayed": True,
            "unmodified_heads_preserved": True,
        },
    }


def verify(witness, monkeypatch):
    monkeypatch.setattr(audit, "W0_SHA256", audit._matrix_sha(witness["baseline"]))
    manifest_sha = audit._sha(audit._encode(audit.witness_manifest(witness)))
    return audit.audit(
        witness,
        expected_manifest_sha256=manifest_sha,
        expected_delta_sha256=witness["delta_sha256"],
    )


def test_independent_fp64_replay_both_heads_and_unadapted_head(monkeypatch):
    witness = synthetic()
    result = verify(witness, monkeypatch)
    assert result["status"] == "numeric_consistency_only"
    assert result["applied_head_ids"] == ["operation", "click_target"]
    assert result["absent_head_ids"] == []
    assert result["legal_counts"] == [3, 3, 2]
    assert max(result["max_probability_absolute_error"].values()) < 1e-12
    assert result["model_score"] is None and result["fit_eligible"] is False


@pytest.mark.parametrize(
    "heads, absent",
    [
        (("click_target", "select_target"), ["operation"]),
        (("operation",), ["click_target"]),
        (("select_target",), ["operation", "click_target"]),
    ],
)
def test_absent_heads_are_explicit_and_never_remapped(heads, absent, monkeypatch):
    witness = synthetic(heads)
    assert verify(witness, monkeypatch)["absent_head_ids"] == absent
    witness["projection"]["absent_head_ids"] = []
    with pytest.raises(ValueError, match="head mapping"):
        verify(witness, monkeypatch)


def test_external_manifest_and_deltas_bind_all_bytes(monkeypatch):
    witness = synthetic()
    monkeypatch.setattr(audit, "W0_SHA256", audit._matrix_sha(witness["baseline"]))
    pin = audit._sha(audit._encode(audit.witness_manifest(witness)))
    wrong = dict(witness["delta_sha256"], operation="0" * 64)
    with pytest.raises(ValueError, match="delta bytes"):
        audit.audit(witness, expected_manifest_sha256=pin, expected_delta_sha256=wrong)
    witness["features"][0, 0] = 2
    with pytest.raises(ValueError, match="commitment"):
        audit.audit(
            witness, expected_manifest_sha256=pin, expected_delta_sha256=witness["delta_sha256"]
        )


@pytest.mark.parametrize("which", ["baseline_raw", "zero_raw", "composed_raw"])
def test_numeric_tampering_rejected_even_if_manifest_is_resealed(which, monkeypatch):
    witness = synthetic()
    answer = witness[which]["answers"]["operation"]
    answer["probabilities"] = {"first": 1 / 3, "second": 1 / 3, "third": 1 / 3}
    answer["choice"] = "first"
    answer["confidence"] = 0.0
    with pytest.raises(ValueError, match="replay or argmax"):
        verify(witness, monkeypatch)


@pytest.mark.parametrize("which", ["baseline_raw", "zero_raw", "composed_raw"])
@pytest.mark.parametrize("field", ["probability", "confidence"])
def test_singleton_must_be_exact_even_if_manifest_is_resealed(which, field, monkeypatch):
    witness = synthetic(("operation",))
    witness["ordered_criteria_ids"][0] = ["first"]
    witness["native_payload"]["questions"]["operation"]["criteria"] = {"first": "description"}
    witness["projection"]["legal_counts"] = [1]
    for name in ("baseline_raw", "zero_raw", "composed_raw"):
        answer = witness[name]["answers"]["operation"]
        answer.update(probabilities={"first": 1.0}, choice="first", confidence=1.0)
    assert verify(witness, monkeypatch)["legal_counts"] == [1]
    answer = witness[which]["answers"]["operation"]
    if field == "probability":
        answer["probabilities"]["first"] = 0.9999995
    else:
        answer["confidence"] = 0.9999995
    with pytest.raises(ValueError, match="singleton requires an exact point mass"):
        verify(witness, monkeypatch)


def test_untouched_head_must_preserve_raw_bytes(monkeypatch):
    witness = synthetic()
    witness["composed_raw"]["answers"]["select_target"]["extra"] = "changed"
    with pytest.raises(ValueError, match="unadapted raw"):
        verify(witness, monkeypatch)


def test_projection_head_order_cannot_be_rewritten(monkeypatch):
    witness = synthetic()
    witness["projection"]["ordered_head_ids"] = list(reversed(witness["ordered_head_ids"]))
    with pytest.raises(ValueError, match="head mapping"):
        verify(witness, monkeypatch)


def test_refuses_invalid_or_missing_witness_and_bad_native_metadata(monkeypatch):
    witness = synthetic()
    del witness["zero_raw"]
    with pytest.raises(ValueError, match="exact persisted"):
        audit.witness_manifest(witness)
    witness = synthetic()
    witness["deltas"]["operation"] = witness["deltas"]["operation"].astype(np.float64)
    with pytest.raises(ValueError, match="CPU FP32"):
        audit.witness_manifest(witness)
    witness = synthetic()
    witness["composed_raw"]["usage"]["input_tokens"] += 1
    with pytest.raises(ValueError, match="usage"):
        verify(witness, monkeypatch)
    witness = synthetic()
    witness["composed_raw"]["answers"]["click_target"]["confidence"] = 0.5
    with pytest.raises(ValueError, match="confidence"):
        verify(witness, monkeypatch)
    witness = synthetic()
    witness["ordered_criteria_ids"][0] = list(reversed(witness["ordered_criteria_ids"][0]))
    with pytest.raises(ValueError, match="candidate inventory"):
        verify(witness, monkeypatch)


class FakeTensor:
    """Explicit CPU fake used only to exercise the writer contract."""

    def __init__(self, array):
        self.array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def numpy(self):
        return self.array


def write_synthetic(root, witness):
    sink, receipts = audit.make_witness_writer(root, maximum_calls=1)
    payload = {
        key: copy.deepcopy(value)
        for key, value in witness.items()
        if key not in {"features", "baseline", "deltas"}
    }
    sink(
        torch=None,
        features=FakeTensor(witness["features"]),
        baseline=FakeTensor(witness["baseline"]),
        deltas={key: FakeTensor(value) for key, value in witness["deltas"].items()},
        **payload,
    )
    assert len(receipts) == 1
    return receipts[0], sink, payload


def test_writer_to_cpu_file_audit_roundtrip_and_cli(tmp_path, monkeypatch):
    witness = synthetic()
    monkeypatch.setattr(audit, "W0_SHA256", audit._matrix_sha(witness["baseline"]))
    receipt, sink, payload = write_synthetic(tmp_path / "private", witness)
    result = audit.audit_directory(
        receipt["directory"],
        expected_file_sha256=receipt["manifest_sha256"],
        expected_witness_sha256=receipt["witness_sha256"],
        expected_delta_sha256=receipt["delta_sha256"],
    )
    assert result["status"] == "numeric_consistency_only" and result["legal_counts"] == [3, 3, 2]
    out = tmp_path / "result.json"
    assert (
        audit.main(
            [
                "--witness-dir",
                receipt["directory"],
                "--manifest-sha256",
                receipt["manifest_sha256"],
                "--witness-sha256",
                receipt["witness_sha256"],
                "--operation-delta-sha256",
                receipt["delta_sha256"]["operation"],
                "--click-target-delta-sha256",
                receipt["delta_sha256"]["click_target"],
                "--output",
                str(out),
            ]
        )
        == 0
    )
    assert out.is_file()
    with pytest.raises(ValueError, match="count exceeded"):
        sink(
            torch=None,
            features=FakeTensor(witness["features"]),
            baseline=FakeTensor(witness["baseline"]),
            deltas={key: FakeTensor(value) for key, value in witness["deltas"].items()},
            **payload,
        )


def test_file_audit_refuses_tampering_missing_extra_and_symlink(tmp_path, monkeypatch):
    witness = synthetic()
    monkeypatch.setattr(audit, "W0_SHA256", audit._matrix_sha(witness["baseline"]))
    receipt, _, _ = write_synthetic(tmp_path / "private", witness)
    directory = __import__("pathlib").Path(receipt["directory"])

    def check():
        return audit.audit_directory(
            directory,
            expected_file_sha256=receipt["manifest_sha256"],
            expected_witness_sha256=receipt["witness_sha256"],
            expected_delta_sha256=receipt["delta_sha256"],
        )

    original = (directory / "features.npy").read_bytes()
    (directory / "features.npy").write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(ValueError, match="file pin"):
        check()
    (directory / "features.npy").write_bytes(original)
    (directory / "extra").write_text("x")
    with pytest.raises(ValueError, match="inventory"):
        check()
    (directory / "extra").unlink()
    (directory / "features.npy").unlink()
    with pytest.raises(ValueError, match="inventory"):
        check()
    (directory / "features.npy").symlink_to(directory / "baseline.npy")
    with pytest.raises(OSError):
        check()


def test_npy_parser_refuses_wrong_shapes_and_pickled_objects():
    import io

    def encode(value):
        stream = io.BytesIO()
        np.save(stream, value, allow_pickle=True)
        return stream.getvalue()

    with pytest.raises(ValueError, match="header"):
        audit._load_npy(encode(np.zeros((2, 2), dtype=np.float32)), (3, audit.WIDTH))
    with pytest.raises(ValueError, match="header"):
        audit._load_npy(encode(np.asarray([object()], dtype=object)), (1,))
