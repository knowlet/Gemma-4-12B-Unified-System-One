"""Pure resource/refusal/publication tests; no real Torch, weights or fitting."""

import json
import os
from types import SimpleNamespace

import jev_browser_fit_public_actions as FIT
import jev_public_action_artifact as ART
import pytest

MEM_TOTAL = 130595860480


def observation(available):
    return {"MemTotal": MEM_TOTAL, "MemAvailable": available, "boot_id": "boot"}


@pytest.mark.parametrize("scope,required", [("cpu", 30414139392), ("load", 73958270912), ("train", 106438961088)])
def test_resource_formula_exact_boundary_is_durable_before_return(tmp_path, scope, required):
    phase = {"cpu": "before-any-project-framework-or-parent-payload", "load": "before-checkpoint-payload-hash-and-BF16-loader", "train": "before-any-BF16-model-construction"}[scope]
    gate = FIT.ResourceGate(tmp_path / "gates.jsonl", "boot", observe=lambda: observation(required))
    row = gate.check(phase, scope)
    assert row["admitted"] is True
    assert row["required_bytes"] == required
    assert json.loads((tmp_path / "gates.jsonl").read_text()) == row
    refused = FIT.ResourceGate(tmp_path / "refused.jsonl", "boot", observe=lambda: observation(required - 1))
    with pytest.raises(ValueError, match="refused"):
        refused.check(phase, scope)
    assert json.loads((tmp_path / "refused.jsonl").read_text())["admitted"] is False


def test_wrong_boot_and_journal_mutation_refuse(tmp_path):
    gate = FIT.ResourceGate(tmp_path / "wrong.jsonl", "different", observe=lambda: observation(MEM_TOTAL))
    with pytest.raises(ValueError, match="same-boot"):
        gate.check("before-any-project-framework-or-parent-payload", "cpu")
    gate = FIT.ResourceGate(tmp_path / "changed.jsonl", "boot", observe=lambda: observation(MEM_TOTAL))
    (tmp_path / "changed.jsonl").write_text("unexpected")
    with pytest.raises(ValueError, match="journal changed"):
        gate.check("before-any-project-framework-or-parent-payload", "cpu")


def test_gate_refusal_primary_survives_failed_persistence(tmp_path, monkeypatch):
    gate = FIT.ResourceGate(tmp_path / "gates.jsonl", "boot", observe=lambda: observation(1))
    monkeypatch.setattr(FIT.os, "write", lambda *_args: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(ValueError, match="admission refused") as captured:
        gate.check("before-any-project-framework-or-parent-payload", "cpu")
    assert "persistence failed" in " ".join(captured.value.__notes__)


def test_cli_requires_run_before_any_work(tmp_path):
    with pytest.raises(ValueError, match="--run"):
        FIT.main(["--output", str(tmp_path / "out"), "--source-handoff", "unused",
                  "--expected-source-handoff", "a" * 64, "--expected-boot-id", "boot"])
    assert not (tmp_path / "out").exists()


def test_initial_memory_refusal_never_reaches_project_or_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(FIT, "_fresh_environment", lambda: None)
    class Refuse:
        def __init__(self, *_args):
            pass

        def check(self, *_args):
            raise ValueError("CPU memory refused")
    monkeypatch.setattr(FIT, "ResourceGate", Refuse)
    monkeypatch.setattr(ART, "trusted_module", lambda *_args: pytest.fail("Project import reached"))
    with pytest.raises(ValueError, match="CPU memory refused"):
        FIT.fit_public_actions(output=tmp_path / "out", source_handoff="unused",
            expected_source_handoff="a" * 64, expected_boot_id="boot")
    receipt = json.loads((tmp_path / "out/receipt.json").read_text())
    assert receipt["stage"] == "failed"
    assert receipt["model_loads"] == receipt["training_updates"] == 0
    assert not (tmp_path / "out/adapter/manifest.json").exists()


def candidate_fixture(tmp_path, monkeypatch):
    output = tmp_path / "fit"
    output.mkdir()
    (output / "adapter").mkdir()
    (output / "receipt.json").write_text("{}")
    (output / "resource-gates.jsonl").write_text("before\n")
    (output / "adapter/adapter_config.json").write_text("{}")
    (output / "adapter/adapter_model.safetensors").write_bytes(b"synthetic-not-tensors")
    (output / "adapter/manifest.pending.json").write_text("{}")
    monkeypatch.setattr(FIT, "_parent_checks", lambda _captured: None)
    monkeypatch.setattr(ART, "validate_manifest", lambda _manifest: None)
    monkeypatch.setattr(ART, "validate_training_receipt", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ART, "validate_resource_evidence", lambda *_args, **_kwargs: None)
    def semantic(_path, _blob, **kwargs):
        assert kwargs["_producer_journal"] is True
        with (output / "resource-gates.jsonl").open("a") as stream:
            stream.write("final semantic gate\n")
    monkeypatch.setattr(ART, "_verify_public_material", semantic)
    inputs = SimpleNamespace(check=lambda: None, source_handoff_sha256="a" * 64)
    return output, {"training_receipt": {}, "provenance": {}, "tensor_metadata": {"tensor_shapes": {}}}, {"stage": "pending", "steps": 120}, inputs


def test_publication_seals_fresh_semantic_gates_and_rename_is_last_operation(tmp_path, monkeypatch):
    output, manifest, receipt, inputs = candidate_fixture(tmp_path, monkeypatch)
    real_replace = os.replace
    events = []
    def replace(source, target):
        events.append(str(target))
        real_replace(source, target)
        if str(target).endswith("adapter/manifest.json"):
            monkeypatch.setattr(ART, "pin", lambda *_args: pytest.fail("Fallible validation after marker replacement"))
    monkeypatch.setattr(FIT.os, "replace", replace)
    FIT.publish_candidate(output, manifest, receipt, inputs=inputs, captured={}, gate=object())
    assert events[-1].endswith("adapter/manifest.json")
    saved = json.loads((output / "receipt.json").read_text())
    assert saved["output_inventory_excluding_receipt_and_manifest"]["resource-gates.jsonl"]["size_bytes"] == len(b"before\nfinal semantic gate\n")
    assert json.loads((output / "adapter/manifest.json").read_text())["training_receipt"]["sha256"] == ART.sha((output / "receipt.json").read_bytes())


@pytest.mark.parametrize("failure", ["semantic", "source", "temporary-fsync", "marker-rename"])
def test_failed_candidate_never_leaves_valid_marker_and_retains_partial(tmp_path, monkeypatch, failure):
    output, manifest, receipt, inputs = candidate_fixture(tmp_path, monkeypatch)
    if failure == "semantic":
        monkeypatch.setattr(ART, "_verify_public_material", lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("semantic")))
    elif failure == "source":
        checks = []
        def changed():
            checks.append(1)
            if len(checks) == 2:
                raise ValueError("source")
        inputs.check = changed
    elif failure == "temporary-fsync":
        original = ART.create
        def failed(path, blob):
            if path.name.startswith("manifest.") and path.name.endswith(".tmp"):
                raise ValueError("temporary-fsync")
            original(path, blob)
        monkeypatch.setattr(ART, "create", failed)
    else:
        original = os.replace
        def failed(source, target):
            if str(target).endswith("adapter/manifest.json"):
                raise ValueError("marker-rename")
            original(source, target)
        monkeypatch.setattr(FIT.os, "replace", failed)
    with pytest.raises(ValueError, match=failure):
        FIT.publish_candidate(output, manifest, receipt, inputs=inputs, captured={}, gate=object())
    assert not (output / "adapter/manifest.json").exists()
    assert (output / "adapter/manifest.pending.json").exists()
    if failure == "marker-rename":
        assert list((output / "adapter").glob("manifest.*.tmp"))


def test_whole_reserved_byte_and_file_caps(tmp_path, monkeypatch):
    monkeypatch.setattr(ART, "tree_sizes", lambda _path: {"file": ART.GIB - 64 * 1024**2 - 1024**2})
    FIT.check_cap(tmp_path)
    with pytest.raises(ValueError, match="cap"):
        FIT.check_cap(tmp_path, pending_bytes=1)
    monkeypatch.setattr(ART, "tree_sizes", lambda _path: {str(i): 0 for i in range(ART.MAX_FILES - 264)})
    FIT.check_cap(tmp_path)
    with pytest.raises(ValueError, match="cap"):
        FIT.check_cap(tmp_path, pending_files=1)


def test_main_returns_without_fallible_output_after_marker(monkeypatch):
    result = {"stage": "complete", "qualified": False}
    monkeypatch.setattr(FIT, "fit_public_actions", lambda **_kwargs: result)
    monkeypatch.setattr("builtins.print", lambda *_args, **_kwargs: pytest.fail("Output after final marker"))
    assert FIT.main(["--run", "--output", "unused", "--source-handoff", "unused",
                    "--expected-source-handoff", "a" * 64, "--expected-boot-id", "boot"]) is result
