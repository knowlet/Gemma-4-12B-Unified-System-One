"""Local verifier integrity and fail-closed evidence; no model/network/GPU."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.inference
ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "verify_breakthrough_prompt", ROOT / "scripts/verify_breakthrough_prompt.py"
)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def test_tensor_receipt_retains_exact_native_bytes_and_complete_masks():
    tensor = torch.tensor([[0, 1, 1], [1, 0, 1]], dtype=torch.int64).T
    evidence = verifier.tensor_evidence(tensor, values=True)
    assert evidence["shape"] == [3, 2]
    assert evidence["values"] == tensor.tolist()
    assert evidence["dtype"] == "torch.int64" and evidence["device"] == "cpu"
    assert evidence["sha256"] == hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()
    assert evidence["bytes"] == 48


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_processor_features_are_not_recorded_as_verified(bad):
    with pytest.raises(ValueError, match="nonfinite"):
        verifier.tensor_evidence(torch.tensor([1.0, bad]))


def test_processor_failure_preserves_offline_partial_evidence(monkeypatch, tmp_path):
    from transformers import AutoProcessor

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "processor_config.json").write_text("{}")
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text("local fixture\n")
    output = tmp_path / "receipt.json"
    calls = []

    def failed_processor(path, **options):
        calls.append((path, options))
        raise RuntimeError("local processor cannot be loaded")

    monkeypatch.setattr(AutoProcessor, "from_pretrained", failed_processor)
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    with pytest.raises(RuntimeError, match="local processor"):
        verifier.verify(checkpoint, fixture, output)
    assert calls == [(str(checkpoint), {"local_files_only": True})]
    report = json.loads(output.read_text())
    assert report["status"] == "failed_local_processor_check"
    assert report["offline"] == {
        "local_files_only": True,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    assert report["records"] == [] and report["errors"][0]["completed_questions"] == 0
    assert report["inference"] == "not_run" and report["gpu_performance"] == "not_measured"
    assert (
        report["processor_files"]["processor_config.json"]["sha256"]
        == hashlib.sha256(b"{}").hexdigest()
    )


def test_existing_evidence_is_never_overwritten_or_reexecuted(monkeypatch, tmp_path):
    from transformers import AutoProcessor

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text("local fixture\n")
    output = tmp_path / "receipt.json"
    original = b'{"previous_execution":"preserved"}\n'
    output.write_bytes(original)

    def forbidden_processor(*args, **kwargs):
        pytest.fail("existing output must reject before processor construction")

    monkeypatch.setattr(AutoProcessor, "from_pretrained", forbidden_processor)
    with pytest.raises(FileExistsError):
        verifier.verify(checkpoint, fixture, output)
    assert output.read_bytes() == original
