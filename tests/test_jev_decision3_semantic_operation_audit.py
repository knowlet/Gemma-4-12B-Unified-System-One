"""Synthetic semantic witness tests; no GPU or private artifacts."""

import copy
import hashlib

import jev_decision3_semantic_operation_audit as audit
import numpy as np
import pytest
from test_jev_decision3_operation_gpu import Tensor
from test_jev_decision3_semantic_operation_adapter import fake_sha, request

pytest_plugins = ("test_jev_decision3_semantic_operation_adapter",)


def test_same_call_semantic_witness_writer_and_fp64_audit(synthetic, monkeypatch, tmp_path):
    instance, provider, _, hashes, names, names_sha, _ = synthetic
    monkeypatch.setattr(audit, "SOURCE_ONLY", False)
    monkeypatch.setattr(audit, "WIDTH", 2)
    monkeypatch.setattr(audit, "CODES", 3)
    monkeypatch.setattr(audit, "W0_SHA256", fake_sha(provider.runtime.readout))
    monkeypatch.setattr(
        Tensor, "numpy", lambda self: np.array(self.values, dtype=np.float32), raising=False
    )
    sink, receipts = audit.make_witness_writer(tmp_path / "witnesses", maximum_calls=1)
    instance.witness_sink = sink
    instance.predict(request(list(reversed(names))), deadline=lambda: None)
    assert len(receipts) == 1
    receipt = receipts[0]
    result = audit.audit_directory(
        receipt["directory"],
        expected_file_sha256=receipt["manifest_sha256"],
        expected_witness_sha256=receipt["witness_sha256"],
        expected_delta_sha256=hashes,
        expected_names_sha256=names_sha,
    )
    assert result["status"] == "numeric_consistency_only"
    assert result["operation_names_sha256"] == names_sha
    assert result["browser_qualified"] is False
    assert provider.runtime.forward_calls == 1
    with pytest.raises(ValueError, match="metadata"):
        audit.audit_directory(
            receipt["directory"],
            expected_file_sha256=receipt["manifest_sha256"],
            expected_witness_sha256=receipt["witness_sha256"],
            expected_delta_sha256=hashes,
            expected_names_sha256="0" * 64,
        )


def test_unknown_operation_and_vocab_tampering_are_rejected(synthetic, monkeypatch):
    instance, provider, _, hashes, names, _, witnesses = synthetic
    monkeypatch.setattr(audit, "SOURCE_ONLY", False)
    monkeypatch.setattr(audit, "WIDTH", 2)
    monkeypatch.setattr(audit, "CODES", 3)
    monkeypatch.setattr(audit, "W0_SHA256", fake_sha(provider.runtime.readout))
    instance.predict(request(names), deadline=lambda: None)
    witness = copy.deepcopy(witnesses[0])
    witness.pop("torch")
    for key in ("features", "baseline"):
        witness[key] = np.asarray(witness[key].values, dtype=np.float32)
    witness["deltas"] = {
        head: np.asarray(value.values, dtype=np.float32)
        for head, value in witness["deltas"].items()
    }
    witness["operation_names"] = ["invented-0", "invented-z"]
    names_sha = hashlib.sha256(audit._encode(witness["operation_names"])).hexdigest()
    witness["operation_names_sha256"] = names_sha
    witness["projection"]["operation_names"] = witness["operation_names"]
    commitment = audit._sha(audit._encode(audit.witness_manifest(witness)))
    with pytest.raises(ValueError, match="unknown semantic operation"):
        audit.audit(
            witness,
            expected_manifest_sha256=commitment,
            expected_delta_sha256=hashes,
            expected_names_sha256=names_sha,
        )


def test_audit_gate_prevents_file_access(tmp_path):
    with pytest.raises(ValueError, match="independent review"):
        audit.make_witness_writer(tmp_path / "not-created", maximum_calls=1)
    assert not (tmp_path / "not-created").exists()
